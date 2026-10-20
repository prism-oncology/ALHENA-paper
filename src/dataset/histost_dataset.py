import logging
logger = logging.getLogger(__name__)

import pandas.api.types as ptypes
from pathlib import Path
from typing import Callable, Union, List, Optional
import os
import hashlib
from tqdm import tqdm
import zipfile
import io
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.sparse import load_npz
import re
from natsort import natsorted
from contextlib import ExitStack
from torch.utils.data import Dataset
from torch_geometric.data import Data
from collections import defaultdict
import gc
import pickle, lmdb
import torch.distributed as dist
import shutil
from filelock import FileLock
import json
import h5py
import random

from src.utils.utils import get_dist_rank
from src.utils.cell_ids import str_cell_id
from src.dataset.hed_augmentation import hed_augment_image
from src.dataset.novae_teacher import fetch_scconcept_for_cells


# LMDB blob format version (changing it forces a cache rebuild)
LMDB_CACHE_SCHEMA_VERSION = 2



class HistoStDataset(Dataset):
    """Dataset for paired H&E patches and ST teachers (precomputed Novae st_feat, optional scConcept cell embeddings).

    Args:
        dataset_path (Union[Path, str]): Path to the training-dataset folder.
        fold (str): Fold to use for this dataset ('train', 'valid').
        training_id (str): Training dataset ID.
        transforms (Callable, optional): Albumentations H&E transforms (image + mask + keypoints). Defaults to None.
        n_min_cells (int, optional): Minimum number of cells required to keep a patch. Defaults to 0.
        mode (str, optional): Mode of the dataset ('train', 'valid'). Defaults to 'train'.
        hed_augm_p (float, optional): Probability of applying HED augmentation. Defaults to None.
        hed_augm_sigma (float, optional): Sigma for HED augmentation. Defaults to None.
        cached_dataset (str, optional): Dataset caching policy. Allowed: "no", "lmdb_init". Defaults to "no".
        use_scconcept_embeddings (bool, optional): Whether to use scConcept cell embeddings (token loss). Defaults to False.
        scconcept_embeddings_dir (Union[Path, str, None], optional): Per-slide scConcept ``{slide_id}.h5`` store.
        scconcept_max_cells (int, optional): Fixed number K of cells sampled per patch for the token loss. Pads with -1/0 if fewer.
        novae_teacher_cache_dir (Union[Path, str, None]): Required. The precomputed Novae-teacher st_feat cache dir.
        novae_teacher_fingerprint (str, optional): Expected cache fingerprint (checked at each per-slide H5 open).
    """

    def __init__(
        self,
        dataset_path: Union[Path, str],
        fold: str,
        training_id: str,
        transforms: Callable = None,
        n_min_cells: int = 0,
        mode: str = "train",
        hed_augm_p: float = None,
        hed_augm_sigma: float = None,
        cached_dataset: str = "no",
        use_scconcept_embeddings: bool = False,
        scconcept_embeddings_dir: Union[Path, str, None] = None,
        scconcept_max_cells: int = 32,
        novae_teacher_cache_dir: Union[Path, str, None] = None,
        novae_teacher_fingerprint: str | None = None,
    ) -> None:

        self.dataset = Path(dataset_path).resolve()
        self.transforms = transforms
        self.images = []
        self.masks = []
        self.patch_ids = []
        self.fold = fold
        self.training_id = training_id
        self.n_min_cells = n_min_cells
        self.mode = mode
        self.hed_augm_p = hed_augm_p
        self.hed_augm_sigma = hed_augm_sigma
        # Per-slide nucleus centroids (str cell id -> x/y), loaded lazily from parquet
        self._centroids_cache = {}

        # Caching policy: only "no" and "lmdb_init" are supported
        cached_dataset = (cached_dataset or "no")
        cached_dataset = str(cached_dataset).strip().lower()
        if cached_dataset not in ("no", "lmdb_init"):
            raise ValueError(f"Invalid cached_dataset='{cached_dataset}'. Allowed: 'no', 'lmdb_init'.")
        self.cached_dataset = cached_dataset

        # scConcept cell embeddings (precomputed with scConcept, and used only for token-level loss)
        self.use_scconcept_embeddings = bool(use_scconcept_embeddings)
        self.scconcept_max_cells = int(scconcept_max_cells)
        if self.scconcept_max_cells <= 0:
            raise ValueError("scconcept_max_cells must be a positive integer (fixed K per patch).")

        self.scconcept_embeddings_dir = None
        if self.use_scconcept_embeddings:
            if scconcept_embeddings_dir is None:
                raise ValueError("use_scconcept_embeddings=True requires scconcept_embeddings_dir to be provided.")
            self.scconcept_embeddings_dir = Path(scconcept_embeddings_dir).resolve()
            if not self.scconcept_embeddings_dir.is_dir():
                raise FileNotFoundError(f"scConcept embeddings directory not found: {self.scconcept_embeddings_dir}")

        # Lazily opened in each DataLoader worker process (h5py is not fork-safe)
        self._scconcept_handles = {}       # slide_id -> h5py.File
        self._scconcept_cell_id_int = {}   # slide_id -> np.ndarray[int32] (sorted ascending)
        self._scconcept_emb_dim = None     # inferred on first access  

        # LMDB-related mapping helpers (to be able to reuse LMDB with full dataset even when n_min_cells > 0)
        self._lmdb_patch_ids: list[str] | None = None          # _lmdb_patch_ids: list of patch ids stored in the LMDB with full data (order == LMDB index)
        self._lmdb_index_for_local: list[int] | None = None      # _lmdb_index_for_local[i]: LMDB index corresponding to dataset index i
        
        # Precomputed Novae-teacher st_feat cache (src/dataset/precompute_novae_teacher.py), one {slide_id}.h5 per slide
        if novae_teacher_cache_dir is None:
            raise ValueError(
                "novae_teacher_cache_dir is required (precomputed st_feat cache). "
                "Run src/dataset/precompute_novae_teacher.py first."
            )
        self.novae_teacher_cache_dir = Path(novae_teacher_cache_dir).resolve()
        if not self.novae_teacher_cache_dir.is_dir():
            raise FileNotFoundError(f"novae_teacher_cache_dir not found: {self.novae_teacher_cache_dir}")
        self.novae_teacher_fingerprint = novae_teacher_fingerprint  # verified at each h5 open
        self._st_feat_handles = {}       # slide_id -> h5py.File
        self._st_feat_patch_idx = {}     # slide_id -> np.ndarray[int64] (sorted patch_idx)
        self._st_feat_dim = None         # inferred on first access

        # LMDB key encoding: fixed-width (8 bytes) big-endian, so lexicographic order == insert order
        self._lmdb_key_bytes = 8

        # Data loading info ----
        logger.info(f"---> Loading dataset for mode {self.mode} and fold {self.fold}:")

        # Tissue types
        if (self.dataset / "types.csv").is_file():
            logger.info(f"Loading types.csv file")
            self.types_df = pd.read_csv(self.dataset / "types.csv")
        else:
            raise FileNotFoundError(f"types.csv file not found for fold {self.fold}")

        # Validate mode
        if self.mode not in ("train", "valid"):
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'train' or 'valid'.")

        logger.info(f"Using transform: {self.transforms}")
        
        # Image and mask zip files
        self.img_zip_path = self.dataset / "images.zip"
        logger.info(f"img_zip_path is {self.img_zip_path}")
        if not self.img_zip_path.is_file():
            raise FileNotFoundError(f"Image zip file not found for fold {self.fold}")
        
        self.mask_zip_path = self.dataset / "masks_cell_ids_nuclei.zip"
        if not self.mask_zip_path.is_file():
            raise FileNotFoundError(f"Cell ID masks zip file not found for fold {self.fold}")

        # List of patch IDs for this fold, filtered by n_min_cells if applicable
        if self.n_min_cells > 0:
            fold_patch_ids = pd.read_csv(self.dataset / f"cell_count_{self.fold}_{self.n_min_cells}.csv")['patch_id'].tolist()
        else:
            fold_patch_ids = pd.read_csv(self.dataset / f"cell_count_{self.fold}.csv")['patch_id'].tolist()
        fold_patch_ids = natsorted(fold_patch_ids)

        # Also keep the unfiltered list of patch IDs for this fold
        # This is used to build an LMDB cache name that is independent of n_min_cells
        try:
            all_fold_patch_ids = pd.read_csv(self.dataset / f"cell_count_{self.fold}.csv")['patch_id'].tolist()
            self._all_fold_patch_ids = natsorted(all_fold_patch_ids)
        except FileNotFoundError:
            raise FileNotFoundError(f"Unfiltered cell count file not found for fold {self.fold} to build unfiltered patch ID list")
        

        # Check existence of image and mask files for the patch IDs in the fold, and build the lists of images/masks/patch_ids to use
        with ExitStack() as stack:
            img_zip = stack.enter_context(zipfile.ZipFile(self.img_zip_path, 'r'))
            mask_zip = stack.enter_context(zipfile.ZipFile(self.mask_zip_path, 'r'))

            # Precompute file lists for images and cell IDs masks
            existing_img_files = set(img_zip.namelist())
            existing_mask_files = set(mask_zip.namelist())

            # Now in the loop, check accordingly
            for fold_pid in tqdm(fold_patch_ids, desc=f"{self.fold}"):

                img_fname = f"{fold_pid}.png"
                mask_fname = f"{fold_pid}.npz"

                image_ok = img_fname in existing_img_files
                mask_ok = mask_fname in existing_mask_files

                if image_ok and mask_ok:
                    self.images.append(img_fname)
                    self.masks.append(mask_fname)
                    self.patch_ids.append(fold_pid)
                else:
                    raise FileNotFoundError(f"Missing files in zip for patch_id {fold_pid}: "
                                            f"{'image missing ' if not image_ok else ''}"
                                            f"{'mask missing' if not mask_ok else ''}")

        # Keep in type only row with patch_id is in self.patch_ids, and check that all patch_ids have a type
        fold_type_dict = (
            self.types_df[self.types_df["patch_id"].isin(set(self.patch_ids))]
            .set_index("patch_id")["type"]
            .to_dict()
        )
        missing_types = set(self.patch_ids) - set(fold_type_dict.keys())
        if missing_types:
            raise ValueError(f"Missing types for patch IDs: {missing_types}")
        self.types = fold_type_dict                  # map patch_id -> tissue type


        # Cached dataset information ----
        self._cache_backend = "lmdb" if self.cached_dataset == "lmdb_init" else "none"
        self._lmdb_complete = False
        self._cache_f = None
        self._dist = dist.is_available() and dist.is_initialized()
        self._rank = get_dist_rank() if self._dist else 0

        if self._cache_backend == "lmdb":
            cache_root = Path(self.dataset) / ".lmdb_cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            # Cache name: hash of the fold, schema version and unfiltered fold patch list (independent of n_min_cells)
            base_patch_ids = getattr(self, "_all_fold_patch_ids")
            params_str = f"{self.fold}_schema{LMDB_CACHE_SCHEMA_VERSION}_{hashlib.md5('_'.join(base_patch_ids).encode()).hexdigest()}"
            dataset_hash = hashlib.md5(params_str.encode()).hexdigest()[:10]
            self._cache_f = cache_root / f"{self.fold}_{dataset_hash}.lmdb"

            # Probe completeness without building
            self._lmdb_complete = self._lmdb_is_complete(self._cache_f)

            if self._lmdb_complete:
                env = lmdb.open(str(self._cache_f), readonly=True, lock=False, readahead=False, max_readers=1)
                with env.begin() as txn:
                    meta = pickle.loads(txn.get(b"__meta__"))
                env.close()
                # Patch ids stored in the LMDB correspond to the dataset that built the cache (n_min_cells=0)
                cached_patch_ids = meta.get("patch_ids", [])
                self._lmdb_patch_ids = cached_patch_ids
                # Extra patch ids in the LMDB are fine: they are simply never indexed
                missing = set(self.patch_ids) - set(cached_patch_ids)
                if missing:
                    logger.warning(f"LMDB cache exists but is missing {len(missing)} patch id(s) required by this dataset. Cache will be rebuilt.")
                    self._lmdb_complete = False
                else:
                    # dataset index -> LMDB index (LMDB key i corresponds to cached_patch_ids[i])
                    name_to_lmdb_idx = {name: idx for idx, name in enumerate(cached_patch_ids)}
                    self._lmdb_index_for_local = [name_to_lmdb_idx[name] for name in self.patch_ids]
                    logger.info(
                        "Existing LMDB cache detected. "
                        f"Reusing {len(self.patch_ids)}/{len(cached_patch_ids)} patches from cache."
                    )

        # Raise if the LMDB cache must be built but this dataset is not the full fold
        self._raise_if_lmdb_build_would_be_invalid()

        # Slides present in this dataset (from patch_ids)
        self.slide_id_list = natsorted({self._split_patch_id(pid)[0] for pid in self.patch_ids})

        # Open parquet file with bbox information for patches
        if (self.dataset / "patch_bboxes.parquet").is_file():
            logger.info(f"Loading patch bboxes")
            self.bboxes = pd.read_parquet(self.dataset / "patch_bboxes.parquet", columns=["slide_id", "patch_id", "x_min", "y_min", "x_max", "y_max"])
        else:
            raise FileNotFoundError(f"patch_bboxes.parquet file not found for fold {self.fold}")
        self.bboxes["patch_id"] = self.bboxes["patch_id"].astype(str)
        self.bboxes.set_index("patch_id", inplace=True)
        self.bboxes.drop(columns=["slide_id"], inplace=True)
        self.bboxes = self.bboxes.loc[self.bboxes.index.isin(set(self.patch_ids))]

        logger.info(f"Created dataset by using fold {self.fold}")
        logger.info(f"Resulting dataset length: {self.__len__()}")
        logger.info(f"Number of unique slides: {len(self.slide_id_list)}")
        logger.info(f"ST-graph teacher: precomputed st_feat cache at {self.novae_teacher_cache_dir}\n")

        # Initialize as None; open in the worker process
        self.opened_img_zip = None 
        self.opened_mask_zip = None

        # Dataset caching
        if self._cache_backend == "lmdb":
            self._setup_lmdb_cache(build_if_missing=True)
        # nothing to do for backend == "none"

        # Check that the st_feat cache covers every slide, with the expected fingerprint
        self._validate_st_feat_cache_inventory()

        # Fail fast if the token loss is enabled but the requested scConcept run is incomplete
        if self.use_scconcept_embeddings:
            self._validate_scconcept_embeddings_inventory()

        # Build a slide_id to slide_idx mapping
        self.slide_id_to_idx = {sid: i for i, sid in enumerate(self.slide_id_list)}

        # Precompute per-patch slide membership for efficient slide-aware sampling
        # patch_ids are like "{slide_id}_{patch_id}" (slide_id may contain underscores)
        self.slide_id_per_patch = [name.rsplit("_", 1)[0] for name in self.patch_ids]
        self.slide_idx_per_patch = torch.empty(len(self.slide_id_per_patch), dtype=torch.long)

        indices_by_slide: List[List[int]] = [[] for _ in range(len(self.slide_id_list))]
        for i, sid in enumerate(self.slide_id_per_patch):
            sidx = self.slide_id_to_idx.get(sid, None)
            if sidx is None:
                raise ValueError(f"slide_id '{sid}' from patch_id not found in slide_id_list for fold={self.fold}")
            self.slide_idx_per_patch[i] = int(sidx)
            indices_by_slide[sidx].append(i)

        # Store as CPU tensors for samplers
        self.indices_by_slide_idx = [torch.tensor(lst, dtype=torch.long) for lst in indices_by_slide]
        self.slide_patch_counts = torch.tensor([len(lst) for lst in indices_by_slide], dtype=torch.long)


    def _init_zipfiles(self):
        
        if self.opened_img_zip is None:
            self.opened_img_zip = zipfile.ZipFile(self.img_zip_path, 'r')
        
        if self.opened_mask_zip is None:
            self.opened_mask_zip = zipfile.ZipFile(self.mask_zip_path, 'r')


    def __del__(self):
        # Clean up open files
        if hasattr(self, 'opened_img_zip') and self.opened_img_zip is not None:
            self.opened_img_zip.close()
            self.opened_img_zip = None
        
        if hasattr(self, 'opened_mask_zip') and self.opened_mask_zip is not None:
            self.opened_mask_zip.close()
            self.opened_mask_zip = None
        
        # Clean up LMDB environment
        if hasattr(self, "_lmdb_env") and self._lmdb_env is not None:
            self._lmdb_env.close()
            self._lmdb_env = None

        # Close per-worker H5 handles (scConcept token store + Novae-teacher st_feat cache)
        for _handles_attr, _idx_attr in (("_scconcept_handles", "_scconcept_cell_id_int"),
                                         ("_st_feat_handles", "_st_feat_patch_idx")):
            handles = getattr(self, _handles_attr, None)
            if handles:
                for _sid, _f in list(handles.items()):
                    try:
                        _f.close()
                    except Exception:
                        pass
                handles.clear()
            idx = getattr(self, _idx_attr, None)
            if idx:
                idx.clear()

        # Force garbage collection after cleanup
        gc.collect()



    def __getitem__(self, index: int) -> dict:

        def _read_raw(i: int):
            if self._cache_backend == "lmdb":
                self._ensure_lmdb_env()
                # Map local dataset index -> LMDB index (the LMDB may hold a superset of the patches)
                lmdb_idx = (
                    self._lmdb_index_for_local[i]
                    if getattr(self, "_lmdb_index_for_local", None) is not None
                    else i
                )
                key = self._encode_lmdb_key(lmdb_idx)
                with self._lmdb_env.begin(write=False) as txn:
                    buf = txn.get(key)
                if buf is None:
                    # Should not happen for a complete lmdb_init cache
                    raise KeyError(f"LMDB cache is missing key for index {i} (LMDB index {lmdb_idx})")
                else:
                    raw = pickle.loads(buf)
                return raw
            else:
                return self._load_raw_data(i)

        def _is_dummy(d) -> bool:
            # "Dummy" patch: no cell visible after the inside-patch / keypoint filtering
            try:
                return int(d.num_visible_cells.item()) == 0
            except Exception:
                return True

        raw_loaded = _read_raw(index)
        data = self._transform_and_build_output(raw_loaded)

        # Dataset index (the name must not contain "index": PyG offsets `*index*` attributes at collate time)
        data.dataset_idx = torch.tensor([index], dtype=torch.long)

        if self.mode != "train":
            return data

        # No leading underscore: PyG does not collate `_`-prefixed attributes
        is_dummy = 1 if _is_dummy(data) else 0
        data.dummy_initial = torch.tensor([is_dummy], dtype=torch.int8)
        return data



    def __len__(self) -> int:
        """Length of Dataset

        Returns:
            int: Length of Dataset
        """
        return len(self.patch_ids)



    @staticmethod
    def _split_patch_id(patch_id: str) -> tuple[str, str]:
        """
        Split patch_id into (slide_id, patch_suffix) on the last '_' (slide_id may contain underscores).
        """
        try:
            slide_id, suffix = str(patch_id).rsplit("_", 1)
        except ValueError:
            raise ValueError(f"Unexpected patch_id format (missing final '_'): {patch_id}")
        if not suffix.isdigit():
            raise ValueError(f"Unexpected patch_id format (final token is not digits): {patch_id}")
        return slide_id, suffix



    def _encode_lmdb_key(self, i: int) -> bytes:
        """
        Encode an integer index as fixed-width big-endian bytes (lexicographic order == numeric order).
        """
        return int(i).to_bytes(self._lmdb_key_bytes, byteorder="big", signed=False)



    def load_imgfile(self, img_name: str) -> np.ndarray:
        """Load image from file (disk)
        
        Args:
            img_name (str): Name of image file
        """
        img_data = self.opened_img_zip.read(img_name)
        img = Image.open(io.BytesIO(img_data))
        return np.array(img).astype(np.uint8)
    


    def load_cell_count(self):
        """Load Cell count from cell_count_{fold}.csv file. File must be located inside the dataset folder
        and named "cell_count_{fold}.csv" or "cell_count_{fold}_{n_min_cells}.csv" if n_min_cells > 0
        """
        if self.n_min_cells > 0:
            csv_path = self.dataset / f"cell_count_{self.fold}_{self.n_min_cells}.csv"
        else:
            csv_path = self.dataset / f"cell_count_{self.fold}.csv"
        self.cell_count = pd.read_csv(csv_path)
        self.cell_count['patch_id'] = self.cell_count['patch_id'].astype(str)
        self.cell_count.set_index('patch_id', inplace=True)
        self.cell_count = self.cell_count.reindex(self.patch_ids)
        # missing rows -> NaN; make them safe for weighting
        self.cell_count = self.cell_count.fillna(0)
        # enforce numeric dtype where possible (ignore non-numeric columns like slide_set_scope)
        for c in self.cell_count.columns:
            if ptypes.is_numeric_dtype(self.cell_count[c]):
                self.cell_count[c] = pd.to_numeric(self.cell_count[c], errors="coerce").fillna(0)


    def get_sampling_weights_tissue(self, gamma: float = 1) -> torch.Tensor:
        """
        Per-patch sampling weights balancing tissue types: w(t) = (n / count(t)) ** gamma.

        Args:
            gamma (float): Gamma in [0, 1]. 1 -> full balancing; 0 -> original distribution.
        Returns:
            torch.Tensor: Per-sample weights aligned with dataset ordering.
        """
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"

        # Compute counts from the actual dataset being sampled (post-filtering).
        tissues = [self.types[n] for n in self.patch_ids]
        tissue_counts = pd.Series(tissues).value_counts().to_dict()
        n = float(len(tissues))
        if n <= 0:
            raise ValueError("Empty dataset; cannot compute tissue weights")
        logger.info(f"[weights/tissue] counts from dataset: {tissue_counts}")

        # Power-law interpolation between uniform (gamma=0) and fully balanced (gamma=1)
        # w(t) = (n / count(t)) ** gamma
        w_by_tissue = {t: (n / float(c)) ** gamma for t, c in tissue_counts.items()}
        weights = torch.tensor([w_by_tissue[self.types[nm]] for nm in self.patch_ids], dtype=torch.float32)
        return weights


    def get_sampling_weights_slide_headcap(
        self,
        max_oversampling: float = 1.3,
        headcap_C: Optional[float] = None,
        exempt_single_slide: bool = True,
    ) -> torch.Tensor:
        """
        Per-slide head-cap sampling weights w(slide) = min(1, C / count(slide)) (slides with more than C patches are
        downsampled to C). C is the smallest cap keeping the oversampling <= max_oversampling, unless headcap_C is
        given. With exempt_single_slide, the only slide of a tissue keeps weight 1.

        Args:
            max_oversampling (float): Maximum oversampling of an uncapped patch (>= 1). Ignored when ``headcap_C`` is given.
            headcap_C (float|None): Optional fixed cap (> 0). Default None -> derived.
            exempt_single_slide (bool): Exempt single-slide-tissue slides from the cap. Default True.
        Returns:
            torch.Tensor: Per-sample weights aligned with dataset ordering (self.patch_ids).
        """
        # Per-slide train patch counts (after filtering)
        count_by_slide = {sid: int(c) for sid, c in zip(self.slide_id_list, self.slide_patch_counts.tolist())}
        N = int(sum(count_by_slide.values()))
        if N <= 0:
            raise ValueError("Empty dataset; cannot compute slide head-cap weights")

        # Exempt slides: the only train slide of a tissue (a slide spanning several tissues is exempt if it
        # is the only slide of one of them)
        exempt_slides = set()
        if exempt_single_slide:
            slides_by_tissue = defaultdict(set)
            for pid, sid in zip(self.patch_ids, self.slide_id_per_patch):
                slides_by_tissue[self.types[pid]].add(sid)
            for _tissue, _slides in slides_by_tissue.items():
                if len(_slides) == 1:
                    exempt_slides |= _slides

        cappable_counts = {sid: c for sid, c in count_by_slide.items() if sid not in exempt_slides}
        exempt_mass = float(sum(count_by_slide[sid] for sid in exempt_slides))

        if headcap_C is not None:
            C = float(headcap_C)
            if not (C > 0.0):
                raise ValueError(f"headcap_C must be > 0 (got {headcap_C}).")
            logger.info(f"[weights/slide_headcap] using fixed headcap_C={C:.6g}")
        else:
            C = self._derive_headcap_C(N, exempt_mass, list(cappable_counts.values()), float(max_oversampling))

        w_by_slide = {
            sid: (1.0 if sid in exempt_slides else min(1.0, C / float(count_by_slide[sid])))
            for sid in count_by_slide
        }
        # Achieved max oversampling of an uncapped patch (w=1) = N / total effective mass.
        eff_mass = exempt_mass + sum(min(float(count_by_slide[sid]), C) for sid in cappable_counts)
        achieved = N / eff_mass if eff_mass > 0 else float("inf")
        capped = sorted((sid for sid in cappable_counts if count_by_slide[sid] > C),
                        key=lambda s: -count_by_slide[s])
        logger.info(
            f"[weights/slide_headcap] N={N} derived_C={C:.6g} exempt_slides={len(exempt_slides)} "
            f"cappable={len(cappable_counts)} capped={len(capped)} "
            f"achieved_max_oversampling={achieved:.3f} (target<={float(max_oversampling):.3f}); "
            f"capped_slides={capped[:8]}"
        )
        weights = torch.tensor([w_by_slide[sid] for sid in self.slide_id_per_patch], dtype=torch.float32)
        return weights


    @staticmethod
    def _derive_headcap_C(N: int, exempt_mass: float, cappable_counts: list, max_oversampling: float) -> float:
        """
        Smallest per-slide cap C with N / (exempt_mass + sum_cappable min(count, C)) <= max_oversampling (binary search).
        Returns +inf if no slide is cappable, and max(count) (no capping) if the bound already holds at C=1.
        """
        if max_oversampling < 1.0:
            raise ValueError(f"headcap_max_oversampling must be >= 1 (got {max_oversampling}).")
        c = np.asarray(cappable_counts, dtype=np.float64)
        if c.size == 0:
            logger.warning("[weights/slide_headcap] no cappable slides (all tissues single-slide); weights uniform.")
            return float("inf")
        Cmax = float(c.max())
        # oversampling(C) is monotone decreasing: max at C=1 (= N/(exempt_mass + n_cappable)), ->1 at C=Cmax.
        over_at_floor = N / (exempt_mass + float(c.size))
        if over_at_floor <= max_oversampling:
            logger.warning(
                f"[weights/slide_headcap] uncapped oversampling already <= tol "
                f"(floor={over_at_floor:.3f} <= {max_oversampling:.3f}); no cap applied."
            )
            return Cmax
        lo, hi = 1.0, Cmax   # invariant: over(lo) > tol >= over(hi)
        for _ in range(64):
            C = 0.5 * (lo + hi)
            over = N / (exempt_mass + float(np.minimum(c, C).sum()))
            if over > max_oversampling:
                lo = C
            else:
                hi = C
        return float(hi)


    def get_sampling_weights_cell(self, gamma: float = 1, ignore_cat: list = None) -> torch.Tensor:
        """Get sampling weights calculated by cell type statistics

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.
            ignore_cat (list): List of cell type columns to exclude or "ignore" from the
                weighting logic. Defaults to None.

        Returns:
            torch.Tensor: Weights for each sample
        """
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        assert hasattr(self, "cell_count"), "Please run .load_cell_count() in advance!"

        ignore = set(ignore_cat or [])
        ignore.add("slide_set_scope")

        # Determine which columns to *actually* use for weighting
        relevant_cols = [
            col for col in self.cell_count.columns
            if (col not in ignore) and ptypes.is_numeric_dtype(self.cell_count[col])
        ]
        logger.info(f"Relevant columns for weighting cell strategy to balance the dataset: {relevant_cols}")
        if len(relevant_cols) == 0:
            raise ValueError("No columns left for weighting after ignoring categories")
        
        counts = self.cell_count[relevant_cols].to_numpy(dtype=np.float32, copy=False)
        # presence/absence of a cell type in a patch
        present = (counts > 0).astype(np.float32)

        # drop columns with zero frequency (can happen after filtering)
        class_counts = present.sum(axis=0)  # (#patches containing the type)
        keep = class_counts > 0
        if not np.all(keep):
            dropped = [c for c, k in zip(relevant_cols, keep) if not k]
            logger.warning(f"Dropping {len(dropped)} cell-type columns with zero frequency in this dataset: {dropped}")
            present = present[:, keep]
            class_counts = class_counts[keep]

        n = float(present.shape[0])
        if n <= 0:
            raise ValueError("Empty dataset; cannot compute cell weights")

        # Power-law interpolation: gamma=0 -> 1, gamma=1 -> n/class_counts
        # (class_counts is #patches where that type is present)
        class_w = (n / class_counts.astype(np.float32, copy=False)) ** gamma

        # per-sample weight: average class weights among the classes present in the patch
        n_present = present.sum(axis=1)
        img_w = (present @ class_w) / np.maximum(1.0, n_present)

        # final safety
        img_w = np.where(np.isfinite(img_w) & (img_w > 0), img_w, 1.0).astype(np.float32, copy=False)
        return torch.from_numpy(img_w)


    def get_sampling_weights_cell_tissue(self, gamma: float = 1, ignore_cat: list = None) -> torch.Tensor:
        """Get combined sampling weights by calculating tissue and cell sampling weights,
        normalizing them and adding them up to yield one score.

        Args:
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.
            ignore_cat (list): List of cell type columns to exclude or "ignore" from the
                weighting logic. Defaults to None.

        Returns:
            torch.Tensor: Weights for each sample
        """
        assert 0 <= gamma <= 1, "Gamma must be between 0 and 1"
        tw = self.get_sampling_weights_tissue(gamma)
        cw = self.get_sampling_weights_cell(gamma, ignore_cat=ignore_cat)
        # normalize each component to mean~1 to avoid arbitrary scaling differences
        tw = tw / tw.mean().clamp_min(1e-12)
        cw = cw / cw.mean().clamp_min(1e-12)
        weights = tw + cw
        # keep overall scale stable for logging/debugging (sampler only cares about ratios)
        weights = weights / weights.mean().clamp_min(1e-12)
        return weights



    def _load_raw_data(self, index: int):
        """
        Raw inputs of one patch, without augmentation (H&E patch, cell-id mask bytes, cell ids, nucleus
        centroids, tissue type, bbox). This is what the LMDB cache stores.
        """
        self._init_zipfiles()
        img_name = self.images[index]
        mask_name = self.masks[index]
        patch_id = self.patch_ids[index]
        slide_id, _ = self._split_patch_id(patch_id)

        img = self.load_imgfile(img_name)
        mask_npz_bytes = self.opened_mask_zip.read(mask_name)

        # Unique non-zero cell ids present in the mask (0 = background, excluded implicitly).
        mask_sparse = load_npz(io.BytesIO(mask_npz_bytes))
        if mask_sparse.nnz == 0:
            cell_ids_patch = np.array([], dtype=np.int32)
        else:
            cell_ids_patch = np.unique(mask_sparse.data.astype(np.int32, copy=False))

        tissue_type = self.types[patch_id]

        if len(cell_ids_patch) == 0:
            raise ValueError(f"No valid cell IDs found in cell token mask for patch {patch_id}.")

        # Nucleus centroids (slide-level px) of the patch cells, aligned with cell_ids_patch
        centroids = self._get_slide_centroids(slide_id)
        cell_ids_str = [str_cell_id(int(cid)) for cid in cell_ids_patch.tolist()]
        coord_df = centroids.reindex(cell_ids_str)[["x", "y"]]
        have = coord_df.notna().all(axis=1).to_numpy()
        if not have.all():
            missing = [s for s, ok in zip(cell_ids_str, have.tolist()) if not ok]
            logger.warning(
                f"Patch '{patch_id}': {len(missing)} mask cell id(s) missing a nucleus centroid "
                f"(dropped). Example(s): {missing[:5]}"
            )
            keep = np.flatnonzero(have)
            cell_ids_patch = cell_ids_patch[keep]
            coord_df = coord_df.iloc[keep]
        if len(cell_ids_patch) == 0:
            raise ValueError(
                f"Patch '{patch_id}' has no mask cell with a matching nucleus centroid "
                "(preprocessing/alignment issue)."
            )
        coord = coord_df.to_numpy().astype(np.float32, copy=False)
        cell_ids_patch = [int(c) for c in cell_ids_patch.tolist()]

        if patch_id not in self.bboxes.index:
            raise ValueError(f"Bounding box for {patch_id} not found in self.bboxes.")
        bbox_series = self.bboxes.loc[patch_id]
        bbox = {"x_min": int(bbox_series["x_min"]), "y_min": int(bbox_series["y_min"]),
                "x_max": int(bbox_series["x_max"]), "y_max": int(bbox_series["y_max"])}

        return {
            "patch_id": patch_id,
            "slide_id": slide_id,
            "raw_img": img,
            "raw_mask_npz": mask_npz_bytes,    # bytes
            "cell_ids_patch": cell_ids_patch,  # list[int], aligned with raw_coord
            "raw_coord": coord,                # np.float32 [n_cells, 2], slide-level px
            "tissue_type": tissue_type,
            "bbox": bbox,
        }


    def _get_slide_centroids(self, slide_id: str):
        """Lazy-load a slide's nucleus centroids (str cell id -> x/y) from parquet, cached per slide."""
        df = self._centroids_cache.get(slide_id)
        if df is None:
            path = self.dataset / "nucleus_centroids" / f"nucleus_centroids_{slide_id}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Centroids file not found for slide {slide_id}: {path}")
            df = pd.read_parquet(path).set_index("cell_id")
            self._centroids_cache[slide_id] = df
        return df


    def _validate_st_feat_cache_inventory(self) -> None:
        """
        Check that every slide has an st_feat cache file with the expected fingerprint and the same dim.
        """
        missing = [sid for sid in self.slide_id_list
                   if not (self.novae_teacher_cache_dir / f"{sid}.h5").is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} slide(s) have no Novae-teacher st_feat cache in "
                f"{self.novae_teacher_cache_dir}. Run src/dataset/precompute_novae_teacher.py. "
                f"Example missing slide_id(s): {missing[:5]}"
            )
        dims = set()
        for sid in self.slide_id_list:
            h5_path = self.novae_teacher_cache_dir / f"{sid}.h5"
            with h5py.File(h5_path, "r") as f:
                if "patch_idx" not in f or "st_feat" not in f:
                    raise KeyError(f"Invalid st_feat cache (missing 'patch_idx' or 'st_feat'): {h5_path}")
                fp = f.attrs.get("fingerprint")
                if self.novae_teacher_fingerprint is not None and fp != self.novae_teacher_fingerprint:
                    raise ValueError(
                        f"Novae-teacher st_feat cache fingerprint mismatch for slide '{sid}': "
                        f"cache={fp!r} vs configured={self.novae_teacher_fingerprint!r}. The cache was "
                        "built for a different teacher / graph params -- re-run precompute_novae_teacher.py."
                    )
                dims.add(int(f.attrs.get("dim", f["st_feat"].shape[1])))
        if len(dims) != 1:
            raise ValueError(f"Inconsistent st_feat dims across slides: {sorted(dims)}")
        self._st_feat_dim = dims.pop()
        logger.info(
            f"Novae-teacher st_feat cache OK for {len(self.slide_id_list)} slide(s) "
            f"(dim={self._st_feat_dim}, dir={self.novae_teacher_cache_dir})"
        )


    def _get_st_feat_handle(self, slide_id: str):
        """Lazy-open the st_feat H5 of a slide in the current worker (fingerprint checked).
        Returns (h5_file, patch_idx_sorted).
        """
        f = self._st_feat_handles.get(slide_id)
        patch_idx = self._st_feat_patch_idx.get(slide_id)
        if f is not None and patch_idx is not None:
            return f, patch_idx

        h5_path = self.novae_teacher_cache_dir / f"{slide_id}.h5"
        if not h5_path.is_file():
            raise FileNotFoundError(f"Novae-teacher st_feat cache not found for slide_id='{slide_id}': {h5_path}")
        f = h5py.File(h5_path, "r")
        if "patch_idx" not in f or "st_feat" not in f:
            f.close()
            raise KeyError(f"Invalid st_feat cache (missing 'patch_idx' or 'st_feat'): {h5_path}")
        fp = f.attrs.get("fingerprint")
        if self.novae_teacher_fingerprint is not None and fp != self.novae_teacher_fingerprint:
            f.close()
            raise ValueError(
                f"Novae-teacher st_feat cache fingerprint mismatch for slide '{slide_id}': cache={fp!r} "
                f"vs configured={self.novae_teacher_fingerprint!r}. Re-run precompute_novae_teacher.py."
            )
        patch_idx = f["patch_idx"][:].astype(np.int64, copy=False)
        if patch_idx.size > 1 and np.any(patch_idx[1:] < patch_idx[:-1]):
            f.close()
            raise ValueError(f"'patch_idx' must be sorted ascending in {h5_path} (searchsorted lookup).")
        if self._st_feat_dim is None:
            self._st_feat_dim = int(f.attrs.get("dim", f["st_feat"].shape[1]))
        self._st_feat_handles[slide_id] = f
        self._st_feat_patch_idx[slide_id] = patch_idx
        return f, patch_idx


    def _get_st_feat_for_patch(self, slide_id: str, patch_id: str) -> torch.Tensor:
        """Return the precomputed frozen Novae-teacher st_feat [dim] (float32) for one patch."""
        _, patch_suffix = self._split_patch_id(patch_id)
        pidx = int(patch_suffix)
        f, patch_idx = self._get_st_feat_handle(slide_id)
        j = int(np.searchsorted(patch_idx, pidx))
        if j >= patch_idx.shape[0] or int(patch_idx[j]) != pidx:
            raise KeyError(
                f"patch_id '{patch_id}' (idx {pidx}) not found in the Novae-teacher st_feat cache "
                f"for slide '{slide_id}'. Re-run src/dataset/precompute_novae_teacher.py for this dataset."
            )
        vec = f["st_feat"][j]  # [dim] float32
        return torch.from_numpy(np.asarray(vec, dtype=np.float32))


    def _validate_scconcept_embeddings_inventory(self) -> None:
        """
        Check once, at dataset construction time, that the selected scConcept run
        contains one H5 file per slide needed by the current dataset.
        """
        if not self.use_scconcept_embeddings:
            return
        if self.scconcept_embeddings_dir is None:
            raise RuntimeError("scConcept embeddings are enabled but scconcept_embeddings_dir is None.")
        if not self.scconcept_embeddings_dir.is_dir():
            raise FileNotFoundError(f"scConcept embeddings directory not found: {self.scconcept_embeddings_dir}")

        needed_slide_ids = sorted({self._split_patch_id(pid)[0] for pid in self.patch_ids})
        missing = [sid for sid in needed_slide_ids if not (self.scconcept_embeddings_dir / f"{sid}.h5").is_file()]
        if missing:
            examples = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"Missing scConcept embedding file(s) for {len(missing)}/{len(needed_slide_ids)} slide(s) "
                f"in {self.scconcept_embeddings_dir}. Example missing slide_id(s): {examples}"
            )

        dims = {}
        for sid in needed_slide_ids:
            h5_path = self.scconcept_embeddings_dir / f"{sid}.h5"
            with h5py.File(h5_path, "r") as f:
                if "cell_id_int" not in f or "cls_emb" not in f:
                    raise KeyError(f"Invalid scConcept embeddings file: {h5_path}")
                dims[sid] = int(f.attrs.get("embedding_dim", f["cls_emb"].shape[1]))

        unique_dims = sorted(set(dims.values()))
        if len(unique_dims) != 1:
            raise ValueError(f"Inconsistent scConcept embedding dimensions across slides: {dims}")

        self._scconcept_emb_dim = unique_dims[0]
        logger.info(
            f"Validated scConcept embeddings inventory: {len(needed_slide_ids)} slide file(s), "
            f"embedding_dim={self._scconcept_emb_dim}, dir={self.scconcept_embeddings_dir}"
        )




    def _get_scconcept_handle(self, slide_id: str):
        """Lazy-open scConcept H5 file for a slide in the current worker process.

        Returns:
            (h5_file, cell_id_int_array)
        """
        if not self.use_scconcept_embeddings:
            return None, None

        if self.scconcept_embeddings_dir is None:
            raise RuntimeError("scConcept embeddings are enabled but scconcept_embeddings_dir is None.")

        # Check if already opened in this worker
        f = self._scconcept_handles.get(slide_id, None)
        cell_id_int = self._scconcept_cell_id_int.get(slide_id, None)
        if f is not None and cell_id_int is not None:
            return f, cell_id_int

        h5_path = self.scconcept_embeddings_dir / f"{slide_id}.h5"
        if not h5_path.is_file():
            raise FileNotFoundError(f"scConcept embeddings not found for slide_id='{slide_id}' at: {h5_path}")

        f = h5py.File(h5_path, "r")
        if "cell_id_int" not in f or "cls_emb" not in f:
            f.close()
            raise KeyError(f"Invalid scConcept embeddings file (missing 'cell_id_int' or 'cls_emb'): {h5_path}")

        # cell_id_int is kept in RAM for searchsorted lookups
        cell_id_int = f["cell_id_int"][:].astype(np.int32, copy=False)
        if cell_id_int.ndim != 1:
            f.close()
            raise ValueError(f"Expected 1D 'cell_id_int' in {h5_path}, got shape={cell_id_int.shape}")

        if cell_id_int.size > 1 and np.any(cell_id_int[1:] < cell_id_int[:-1]):
            f.close()
            raise ValueError(
                f"'cell_id_int' must be sorted ascending in {h5_path} because lookups use np.searchsorted(). "
                "Fix the export step or sort the file contents before training."
            )

        # Infer embedding dim once
        if self._scconcept_emb_dim is None:
            if "embedding_dim" in f.attrs:
                self._scconcept_emb_dim = int(f.attrs["embedding_dim"])
            else:
                self._scconcept_emb_dim = int(f["cls_emb"].shape[1])

        self._scconcept_handles[slide_id] = f
        self._scconcept_cell_id_int[slide_id] = cell_id_int
        return f, cell_id_int


    def _get_scconcept_embeddings_for_cells(self, slide_id: str, cell_ids_patch) -> torch.Tensor:
        """Fetch scConcept embeddings for a list/array of integer cell IDs in a given slide.
        The returned tensor is aligned with the order of cell_ids_patch.
        """
        if not self.use_scconcept_embeddings:
            raise RuntimeError("scConcept embeddings requested but 'use_scconcept_embeddings' is False.")

        # D must be known even when there are 0 cells (PyG will still concatenate)
        if self._scconcept_emb_dim is None:
            self._get_scconcept_handle(slide_id)

        ids = np.asarray(cell_ids_patch, dtype=np.int32)
        if ids.size == 0:
            return torch.zeros((0, int(self._scconcept_emb_dim)), dtype=torch.float16)

        f, cell_id_int = self._get_scconcept_handle(slide_id)

        # Binary-search lookup (shared with the Novae-teacher precompute, src/dataset/novae_teacher.py)
        emb = fetch_scconcept_for_cells(f, cell_id_int, ids, emb_dim=self._scconcept_emb_dim)  # [N, D], stored dtype

        # Convert to torch (keep dtype as stored, projection head will handle casting under AMP)
        return torch.from_numpy(emb)





    def _transform_and_build_output(self, raw_loaded: dict) -> "Data":
        """
        Apply the H&E augmentations to the output of `_load_raw_data` and build the PyG `Data` of one patch:
        st_feat, H&E patch and, for the token branch, the K sampled scConcept cell embeddings + cell-id map.
        """
        patch_id = raw_loaded["patch_id"]
        slide_id = raw_loaded["slide_id"]
        img = raw_loaded["raw_img"]
        cell_ids_patch = raw_loaded["cell_ids_patch"]
        tissue_type = raw_loaded["tissue_type"]
        bbox = raw_loaded["bbox"]
        coord = raw_loaded["raw_coord"]      # nucleus centroids (slide-level px), aligned to cell_ids_patch

        # ------------------------- 0) Prepare data -------------------------
        # 0.1. Reconstruct the dense cell-id map (only needed for the token branch)
        mask_cell_ids = None
        need_cell_id_map = bool(self.use_scconcept_embeddings)
        if need_cell_id_map:
            mask_bytes = raw_loaded["raw_mask_npz"]
            mask_sparse = load_npz(io.BytesIO(mask_bytes))
            mask_cell_ids = mask_sparse.toarray().astype(np.int32, copy=False)  # [H, W] cell IDs

        # 0.2. Slide-level centroids -> patch-level; drop cells whose centroid lies outside the patch.
        if coord is not None:
            coord = coord.astype(np.float32, copy=False)
            coord[:, 0] = coord[:, 0] - float(bbox["x_min"])
            coord[:, 1] = coord[:, 1] - float(bbox["y_min"])

            H_patch, W_patch = img.shape[:2]     # img is H x W x C (numpy)
            inside_mask = (
                (coord[:, 0] >= 0.0) & (coord[:, 0] < float(W_patch)) &
                (coord[:, 1] >= 0.0) & (coord[:, 1] < float(H_patch))
            )
            if not inside_mask.all():
                keep_idx = np.flatnonzero(inside_mask)
                if keep_idx.size == 0:
                    coord = None
                    cell_ids_patch = []
                else:
                    coord = coord[keep_idx]
                    cell_ids_patch = [cell_ids_patch[i] for i in keep_idx]
            # Clip remaining centroids so Albumentations never sees x==W or y==H (expects x in [0,W), y in [0,H))
            if coord is not None and coord.size > 0:
                eps = 1e-4
                coord[:, 0] = np.clip(coord[:, 0], 0.0, float(W_patch) - eps)
                coord[:, 1] = np.clip(coord[:, 1], 0.0, float(H_patch) - eps)

        # ------------------------- 1) H&E data -------------------------
        # HED (stain) augmentation -- train only
        if self.mode == "train" and self.hed_augm_p is not None:
            if 0 < self.hed_augm_p <= 1:
                img = hed_augment_image(img, slide_id=slide_id, p=self.hed_augm_p, sigma=self.hed_augm_sigma)

        # Albumentations (color + flips/rotate90 + resize/center-crop) on the image, cell-id mask and nucleus keypoints
        if self.transforms is not None:
            data_to_transform = {"image": img}
            if mask_cell_ids is not None:
                data_to_transform["mask"] = mask_cell_ids
            # Albumentations requires the KeypointParams.label_fields on every call, even if empty.
            if coord is not None:
                data_to_transform["keypoints"] = [tuple(p) for p in coord.tolist()]
                data_to_transform["keypoint_ids"] = list(range(len(coord)))
            else:
                data_to_transform["keypoints"] = []
                data_to_transform["keypoint_ids"] = []

            transformed = self.transforms(**data_to_transform)
            img = transformed["image"]
            mask_cell_ids = transformed.get("mask", None)
            keypoint_ids = transformed.get("keypoint_ids", [])

            # Sync the token-branch cell list to the surviving keypoints (order defined by keypoint_ids)
            sel = np.asarray(keypoint_ids, dtype=int)
            if sel.size == 0:
                cell_ids_patch = []
            else:
                cell_ids_patch = [cell_ids_patch[i] for i in sel]
        else:
            raise ValueError("No transforms provided (at least Normalization and ToTensor transforms are required).")

        # Shape / dtype checks for img and the cell-id map
        if img.ndim != 3 or img.shape[0] != 3:
            raise ValueError(f"Image shape {img.shape} is not valid, expected (3, H, W).")
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Image should be a torch.Tensor, got {type(img)}.")
        cell_id_map = None
        if need_cell_id_map:
            if mask_cell_ids is None:
                raise ValueError("Token loss is enabled but the transformed cell-id map is missing.")
            if not isinstance(mask_cell_ids, torch.Tensor):
                raise TypeError(f"Cell-id map should be a torch.Tensor, got {type(mask_cell_ids)}.")
            if mask_cell_ids.ndim != 2:
                raise ValueError(f"Cell-id map shape {mask_cell_ids.shape} is not valid, expected (H, W).")
            # Masks must stay integer-valued; a float here means non-nearest interpolation corrupted IDs.
            if mask_cell_ids.is_floating_point():
                raise ValueError("Cell-id map contains non-integer values after transformation. Ensure nearest-neighbor interpolation is used for masks.")
            cell_id_map = mask_cell_ids.to(torch.int32)

        # ------------------------- 2) ST-graph teacher (precomputed) -------------------------
        # Node-less PyG Data (graph-level attributes collate to [B, ...])
        st_feat = self._get_st_feat_for_patch(slide_id, patch_id)  # torch.float32 [dim]
        data = Data()
        data.num_nodes = 0   # no nodes (set explicitly, PyG cannot infer it)
        data.st_feat = st_feat.unsqueeze(0)   # [1, dim] -> collates to [B, dim]
        data.num_visible_cells = torch.tensor([len(cell_ids_patch)], dtype=torch.long)

        # ------------------------- 3) scConcept token teacher (optional) -------------------------
        # Precomputed, not trained here (only projected in CLIP space):
        #   - data.scconcept_emb      : [1,K,D]
        #   - data.scconcept_cell_id  : [1,K] (pad = -1)
        #   - data.cell_id_map        : [1,H,W] int32 (used to pool image tokens per cell in clip.py)
        if self.use_scconcept_embeddings:
            K = self.scconcept_max_cells
            candidates = [int(x) for x in (cell_ids_patch or [])]
            if self.mode == "train":
                chosen = random.sample(candidates, K) if len(candidates) > K else candidates
            else:
                if len(candidates) > K:
                    seed = int(hashlib.md5(f"{slide_id}|{patch_id}|scconcept_eval".encode()).hexdigest()[:8], 16)
                    rng = np.random.default_rng(seed)
                    chosen = rng.choice(candidates, size=K, replace=False).tolist()
                else:
                    chosen = candidates
            n_sel = len(chosen)

            if self._scconcept_emb_dim is None:
                self._get_scconcept_handle(slide_id)
            D = int(self._scconcept_emb_dim)

            ids_k = torch.full((K,), -1, dtype=torch.long)
            if n_sel > 0:
                ids_k[:n_sel] = torch.tensor(chosen, dtype=torch.long)

            if n_sel > 0:
                emb_sel = self._get_scconcept_embeddings_for_cells(slide_id, chosen)  # [n_sel,D] (float16)
                if emb_sel.ndim != 2 or emb_sel.shape[1] != D:
                    raise ValueError(f"scConcept embedding shape mismatch: got {tuple(emb_sel.shape)}, expected (*,{D})")
                emb_k = torch.zeros((K, D), dtype=emb_sel.dtype)
                emb_k[:n_sel] = emb_sel
            else:
                emb_k = torch.zeros((K, D), dtype=torch.float16)

            data.scconcept_emb = emb_k.unsqueeze(0)        # [1,K,D]
            data.scconcept_cell_id = ids_k.unsqueeze(0)    # [1,K]

        # ------------------------- 4) H&E + metadata -------------------------
        data.img = img.unsqueeze(0)     # [1, 3, H, W]
        if need_cell_id_map:
            data.cell_id_map = cell_id_map.unsqueeze(0)   # [1, H, W]
        data.tissue_type = tissue_type
        data.patch_id = patch_id
        data.slide_id = slide_id
        data.slide_idx = torch.tensor(self.slide_id_to_idx[slide_id], dtype=torch.long)

        return data


    #  --------- DDP-safe LMDB helpers ----------

    def _raise_if_lmdb_build_would_be_invalid(self) -> None:
        """
        Fail early when this dataset instance would not be allowed to build
        the reusable LMDB cache.
        """
        if self._cache_backend != "lmdb" or self.cached_dataset != "lmdb_init":
            return
        if self._lmdb_complete:
            return
        if self.patch_ids == self._all_fold_patch_ids:
            return
        raise RuntimeError(
            "LMDB cache is missing/incomplete, but this dataset instance is not the full fold (filtered and/or subsampled). "
            "To build the reusable LMDB, instantiate the dataset once with n_min_cells=0, "
            f"then rerun with n_min_cells={self.n_min_cells}."
        )


    def _setup_lmdb_cache(self, build_if_missing: bool = False):
        """
        Setup the LMDB cache for the dataset.
        """

        # First, let all ranks check if the cache is complete
        if os.path.exists(self._cache_f):
            try:
                self._lmdb_complete = self._lmdb_is_complete(self._cache_f)
            except lmdb.Error:
                # If file exists but is not accessible yet, assume incomplete
                self._lmdb_complete = False
        else:
            self._lmdb_complete = False
        
        logger.info(f"LMDB cache at {self._cache_f} is complete: {self._lmdb_complete}\n")

        # Now handle cache building based on setting
        if build_if_missing and self.cached_dataset == "lmdb_init":
            if (not self._lmdb_complete) and (self.patch_ids != self._all_fold_patch_ids):
                raise RuntimeError(
                    "LMDB cache is missing/incomplete, but this dataset instance is filtered (n_min_cells > 0). "
                    "To build the reusable LMDB, instantiate the dataset once with n_min_cells=0 "
                    f"for fold='{self.fold}', then rerun with n_min_cells={self.n_min_cells}."
                )
            # Build the cache under a file lock (one process builds it, the others wait)
            lock = FileLock(str(self._cache_f) + ".lock")
            with lock:
                # Re-check inside the lock: another rank/process may have built the cache while this one waited
                if os.path.exists(self._cache_f):
                    try:
                        self._lmdb_complete = self._lmdb_is_complete(self._cache_f)
                    except lmdb.Error:
                        self._lmdb_complete = False
                else:
                    self._lmdb_complete = False

                if not self._lmdb_complete:
                    os.makedirs(os.path.dirname(self._cache_f), exist_ok=True)
                    if os.path.exists(self._cache_f):
                        self._delete_path(self._cache_f)
                        logger.info(f"Deleted existing LMDB cache at {self._cache_f}.")
                    self._build_lmdb_cache()

            # all ranks re-check once the lock is released
            self._lmdb_complete = self._lmdb_is_complete(self._cache_f)
            logger.info(f"LMDB cache at {self._cache_f} is complete: {self._lmdb_complete}")

        self._lmdb_env = None  # lazy-open in workers


    def _build_lmdb_cache(self):
        """
        Create the LMDB cache
        """
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(self._cache_f), exist_ok=True)
        
        # Clean slate
        # Only rank 0 should handle cache deletion
        if self._rank == 0:
            if os.path.exists(self._cache_f):
                logger.info(f"Deleting existing LMDB cache at {self._cache_f}...")
                self._delete_path(self._cache_f)
            
        # Create environment with sufficient space
        env = lmdb.open(str(self._cache_f),
                        map_size=int(1e12),
                        writemap=False,   # ↓ less RAM than writemap=True
                        map_async=True,
                        meminit=False)

        # When the LMDB is built, self.patch_ids must equal self._all_fold_patch_ids.
        # Raise otherwise, to avoid building a cache that will never be considered complete
        if self.patch_ids != self._all_fold_patch_ids:
            raise ValueError("patch_ids != _all_fold_patch_ids when building LMDB cache. "
                             "Ensure that the dataset is properly initialised with the full set of patch ids before building the cache.")

        logger.info("Building LMDB cache (chunked commits)...")
        commit_every = 1000      # choose chunk size for commits
        txn = env.begin(write=True)

        for i in tqdm(range(len(self)), desc="LMDB-cache (init)"):
            raw = self._load_raw_data(i)
            txn.put(self._encode_lmdb_key(i),
                    pickle.dumps(raw, protocol=pickle.HIGHEST_PROTOCOL),
                    append=True)   # keys are strictly increasing lexicographically

            if (i + 1) % commit_every == 0:
                txn.commit()          # flush dirty pages to free RAM
                txn = env.begin(write=True)
                gc.collect()

        txn.commit()  # commit the final (possibly incomplete) chunk

        # store metadata in its own tiny transaction
        with env.begin(write=True) as meta_txn:
            meta_txn.put(b"__meta__", pickle.dumps({
                "schema_version": LMDB_CACHE_SCHEMA_VERSION,
                "slide_id_list": self.slide_id_list,
                "patch_ids": self.patch_ids,
            }, protocol=pickle.HIGHEST_PROTOCOL))

        env.sync(); env.close()
        logger.info(f"LMDB cache built at {self._cache_f}.\n")


    def _ensure_lmdb_env(self):
        """
        Ensure the LMDB environment is opened.
        """
        if self._lmdb_env is None:                 # called in __getitem__
            # Keep reader environments lock-free for maximum performance
            # Use memory mapping for readonly operations to reduce duplicated memory
            self._lmdb_env = lmdb.open(
                str(self._cache_f), readonly=True,
                lock=False, readahead=True, max_readers=2048,
                meminit=False)  # Skip zeroing memory which improves opening time
    

    def _lmdb_is_complete(self, path: Path) -> bool:
        """
        True if the LMDB cache has a __meta__ entry and one entry per patch id of meta["patch_ids"].
        """
        if not path.exists():
            return False
        
        try:
            env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, max_readers=1)
            with env.begin() as txn:
                meta = txn.get(b"__meta__")
                if meta is None:
                    complete = False
                else:
                    meta_dict = pickle.loads(meta)
                    stored_patch_ids = meta_dict.get("patch_ids", [])
                    # Cache is complete if it has exactly one entry per stored patch id
                    # (self.patch_ids is not compared, so the same LMDB can be reused by filtered datasets)
                    complete = (txn.stat()["entries"] - 1 == len(stored_patch_ids))
            env.close()
            return complete
        except lmdb.Error:
            # If there's any LMDB error, consider the cache incomplete
            return False
    

    def _delete_path(self, path: Path) -> None:
        """
        Remove path whether it is a directory or a file.
        """
        if not path.exists():
            return
        if path.is_dir():
            logger.info(f"Deleting {path}...")
            shutil.rmtree(path)
        else:
            logger.info(f"Deleting {path}...")
            path.unlink()


    # -------- Pickle support –----- 
    # Required when the DataLoader workers are launched with “spawn” or “forkserver”
    # Every handle that is not serialisable is dropped here and reopened lazily inside the worker.
    def __getstate__(self):
        state = self.__dict__.copy()
        # Close h5 files already opened by a worker before pickling (re-opened lazily per worker)
        for _handles_key in ("_scconcept_handles", "_st_feat_handles"):
            try:
                if state.get(_handles_key):
                    for _sid, _f in list(state[_handles_key].items()):
                        try:
                            _f.close()
                        except Exception:
                            pass
            except Exception:
                pass
        for k in (
            "opened_img_zip",
            "opened_mask_zip",
            "_lmdb_env",
            "_scconcept_handles",
            "_scconcept_cell_id_int",
            "_st_feat_handles",
            "_st_feat_patch_idx",
        ):
            state[k] = None          # will be recreated lazily

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # ensure the lazily-opened attributes exist and are (re)initialised per worker
        self.opened_img_zip = None
        self.opened_mask_zip = None
        self._lmdb_env = None
        self._scconcept_handles = {}
        self._scconcept_cell_id_int = {}
        self._st_feat_handles = {}
        self._st_feat_patch_idx = {}
        self._scconcept_cell_id_int = {}