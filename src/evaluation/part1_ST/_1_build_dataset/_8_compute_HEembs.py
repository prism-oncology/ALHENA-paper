#!/usr/bin/env python
"""
Compute the H&E encoder embeddings of the HEST / STHELAR eval slides: per patch, and for STHELAR per (patch, cell) group
(tokens pooled with the cell-to-token membership weights). No L2 normalization.

Saved in <ROOT_HistoST> / eval / eval_datasets / st_eval_data / eval_st_<training_dataset_id> / he_embeddings / HEembs_t<training_num>e<checkpoint_num> / <slide_id>.h5
    - patch_id, patch_emb_he, patch_emb_he_clip: patch embeddings
    - patch_cell_patch_index, patch_cell_cell_id_int, patch_cell_alpha, patch_cell_emb_he, patch_cell_emb_he_clip: (patch, cell) group embeddings
"""

import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from anndata import read_h5ad
from tqdm import tqdm
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import sys
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import os

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.models_settings import BACKBONE_TRANSFORM
from src.evaluation.extract_encoders import load_exported_he_encoder


def _configure_native_libs_for_dataloader() -> None:
    """Limit the native thread pools and disable OpenCL (OpenCV) for the DataLoader workers."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def _dataloader_worker_init_fn(_worker_id: int) -> None:
    # Re-apply inside each worker process
    _configure_native_libs_for_dataloader()


# Apply once in the main process too
_configure_native_libs_for_dataloader()



def _tensor_to_numpy_for_storage(
    t: torch.Tensor,
    np_dtype: np.dtype,
    non_blocking: bool = False,
) -> np.ndarray:
    """
    Move a tensor to CPU and cast to the appropriate dtype for HDF5 storage
    in a single step, then return as a NumPy array
    """
    # Map numpy dtype -> torch dtype
    if np_dtype is np.float16:
        target_torch_dtype = torch.float16
    else:
        target_torch_dtype = torch.float32

    # Device -> CPU copy (blocking by default)
    t_cpu = t.to(
        device="cpu",
        dtype=target_torch_dtype,
        non_blocking=non_blocking,
    ).contiguous()
    return t_cpu.numpy()



def _decode_barcode(x) -> str:
    """Decode a single barcode element coming from an HDF5 string dataset."""
    # Expected shape from the writer: (N, 1) strings
    # f["barcode"][i] returns something like ['TENX96_0.png'] or a bytes object.
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8")
    # h5py may already return a (1,) array of str/bytes
    if hasattr(x, "__len__") and not isinstance(x, str):
        elem = x[0]
        if isinstance(elem, (bytes, bytearray)):
            return elem.decode("utf-8")
        return str(elem)
    return str(x)



def _img_to_hwc_uint8(img_np: np.ndarray) -> np.ndarray:
    """
    Convert an HDF5-stored image patch to a contiguous HWC uint8 numpy array.
    Supports both HWC (H, W, 3) and CHW (3, H, W).
    """
    if img_np.ndim != 3:
        raise ValueError(f"Expected img with 3 dims, got shape={img_np.shape}")

    # Layout detection
    if img_np.shape[-1] == 3:
        arr = img_np
    elif img_np.shape[0] == 3:
        arr = np.transpose(img_np, (1, 2, 0))
    else:
        raise ValueError(f"Unrecognized image layout, expected HWC or CHW with 3 channels. shape={img_np.shape}")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if float(arr.max()) <= 1.5:
                arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
            else:
                arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)



def _encoder_rgb_stats_to_list(x: torch.Tensor) -> List[float]:
    vals = x.detach().cpu().view(-1).tolist()
    if len(vals) != 3:
        raise ValueError(f"Expected 3 RGB values, got {len(vals)}.")
    return [float(v) for v in vals]



class HEAlbumentationsCollate:
    """
    Collate applying the training preprocessing (A.Resize -> A.CenterCrop -> A.Normalize -> ToTensorV2).
    Returns (x [B, 3, H, W] float32, patch_ids).
    """
    def __init__(
        self,
        resize_size: int,
        crop_size: int,
        mean: List[float],
        std: List[float],
        interpolation: int,
    ) -> None:
        self.resize_size = int(resize_size)
        self.crop_size = int(crop_size)
        self.interpolation = int(interpolation)

        tfms = []
        if self.resize_size > 0:
            tfms.append(
                A.Resize(
                    height=self.resize_size,
                    width=self.resize_size,
                    interpolation=self.interpolation,
                )
            )
        if self.crop_size > 0:
            tfms.append(A.CenterCrop(height=self.crop_size, width=self.crop_size))
        tfms.append(A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0))
        tfms.append(ToTensorV2())
        self.transform = A.Compose(tfms)

    def __call__(self, batch):
        imgs_np, patch_ids = zip(*batch)
        x = torch.stack(
            [self.transform(image=img)["image"] for img in imgs_np],
            dim=0,
        )
        return x, list(patch_ids)



def _make_collate_for_encoder(encoder: torch.nn.Module):
    """
    Build the external preprocessing collate_fn from encoder-stored metadata.
    """

    if not hasattr(encoder, "expects_preprocessed") or not bool(getattr(encoder, "expects_preprocessed")):
        warnings.warn(
            "Encoder does not expose expects_preprocessed=True. "
            "Assuming it expects external preprocessing, but metadata might be missing."
        )

    resize_size = int(getattr(encoder, "resize_size", 0))
    crop_size = int(getattr(encoder, "crop_size", 0))
    interpolation = int(getattr(encoder, "interpolation_code", cv2.INTER_LINEAR))

    mean = getattr(encoder, "mean", torch.zeros((1, 3, 1, 1), dtype=torch.float32))
    std = getattr(encoder, "std", torch.ones((1, 3, 1, 1), dtype=torch.float32))
    return HEAlbumentationsCollate(
        resize_size=resize_size,
        crop_size=crop_size,
        mean=_encoder_rgb_stats_to_list(mean),
        std=_encoder_rgb_stats_to_list(std),
        interpolation=interpolation,
    )



class H5PatchesDataset(Dataset):
    """
    Torch Dataset over a single HEST-style .h5 slide.

    Exposes:
        - image_u8: np.ndarray, shape (H, W, 3) uint8 in [0..255]
        - patch_id: str, the patch image name (same as 'barcode' in the .h5)
    """

    def __init__(self, h5_path: Path):
        super().__init__()
        self.h5_path = str(h5_path)
        self._file: Optional[h5py.File] = None

        with h5py.File(self.h5_path, "r") as f:
            if "img" not in f or "barcode" not in f:
                raise KeyError(f"{self.h5_path} must contain 'img' and 'barcode' datasets.")
            self._length = f["img"].shape[0]

    def __len__(self) -> int:
        return self._length

    def _get_file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __getitem__(self, idx: int):
        f = self._get_file()
        img_np = f["img"][idx]          # (H, W, 3), uint8
        barcode_raw = f["barcode"][idx] # shape (1,) or scalar

        patch_id = _decode_barcode(barcode_raw)

        # Keep HWC uint8 because Albumentations/OpenCV expects numpy HWC input
        img_u8 = _img_to_hwc_uint8(img_np)

        return img_u8, patch_id
    
    def __del__(self):
        # Make sure the HDF5 file handle is closed when the dataset is GC'ed
        if getattr(self, "_file", None) is not None:
            try:
                self._file.close()
            except Exception:
                pass



MEMBERSHIP_BACKBONES = ("uni2h", "virchow2", "hoptimus0", "hoptimus1")


def _membership_groups_from_backbone_transform():
    by_sig = {}
    for m in MEMBERSHIP_BACKBONES:
        cfg = BACKBONE_TRANSFORM[m]
        sig = (int(cfg.get("resize_size", 0)), int(cfg.get("crop_size", 0)))
        by_sig.setdefault(sig, []).append(m)
    model_to_group = {}
    sig_to_group = {}
    for sig, models in by_sig.items():
        models_sorted = sorted(models)
        gname = "_".join(models_sorted)
        sig_to_group[sig] = gname
        for m in models_sorted:
            model_to_group[m] = gname
    return model_to_group, sig_to_group


_MODEL_TO_GROUP, _SIG_TO_GROUP = _membership_groups_from_backbone_transform()


def membership_group_for_encoder(encoder: torch.nn.Module) -> str:
    """Membership folder of the encoder: from encoder.backbone, else from its (resize_size, crop_size)."""
    backbone = str(getattr(encoder, "backbone", "")).lower()
    if backbone in _MODEL_TO_GROUP:
        return _MODEL_TO_GROUP[backbone]
    rs = int(getattr(encoder, "resize_size", 0))
    cs = int(getattr(encoder, "crop_size", 0))
    sig = (rs, cs)
    if sig in _SIG_TO_GROUP:
        return _SIG_TO_GROUP[sig]
    # Other geometries (not built by _1_download_eval_data.py)
    return f"custom_r{rs}_c{cs}"


def membership_parquet_path(eval_root: Path, group_name: str, slide_id: str) -> Path:
    return eval_root / f"cell_token_membership_{group_name}" / f"{slide_id}.parquet"



def load_slide_cells_from_h5ad(slide_h5ad_path: Path, slide_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (cell_id_int, cell_key) for STHELAR slides.
    Requires .h5ad to contain cell rows with obs['is_cell']==True and obs['cell_id_int'].
    """
    if not slide_h5ad_path.exists():
        raise FileNotFoundError(f"Missing .h5ad for slide '{slide_id}': {slide_h5ad_path}")
    ad = read_h5ad(slide_h5ad_path, backed="r")
    try:
        obs = ad.obs
        if "is_cell" not in obs.columns:
            raise KeyError(f"{slide_h5ad_path} has no 'is_cell' in .obs (expected STHELAR-built .h5ad).")
        cell_mask = obs["is_cell"].astype(bool).to_numpy()
        if not np.any(cell_mask):
            return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=object)
        if "cell_id_int" not in obs.columns:
            raise KeyError(f"{slide_h5ad_path} is missing obs['cell_id_int'] for cell rows.")
        cell_id_int = obs.loc[cell_mask, "cell_id_int"].to_numpy(dtype=np.int32, copy=True)
        cell_key = obs.index.to_numpy(dtype=object, copy=False)[cell_mask]
        # Ensure unique
        if len(np.unique(cell_id_int)) != cell_id_int.shape[0]:
            raise ValueError(f"Duplicate cell_id_int values found in cell rows for slide '{slide_id}'.")
        return cell_id_int, cell_key
    finally:
        if hasattr(ad, "file") and ad.file is not None:
            ad.file.close()



def load_he_encoder_bundle(
    encoder_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """
    Load the compact H&E encoder bundle exported by `extract_encoders.py`.
    The bundle is reconstructed into a regular nn.Module and loaded from its saved state_dict.
    """
    encoder = load_exported_he_encoder(str(encoder_path), device=device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder



def encode_batch_raw_with_tokens(
    encoder: torch.nn.Module,
    images: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[int, int]]]:
    """
    Single backbone pass:
      returns (patch_feat_he, token_feat_he, grid_hw) with no projection and no L2.
    """
    img_he, tok_he, grid_hw = encoder.encode_with_tokens(
        images,
        project_img=False,
        project_tokens=False,
    )
    return img_he, tok_he, grid_hw



def encode_batch_raw_no_tokens(
    encoder: torch.nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """
    Single backbone pass:
      returns patch_feat_he with no projection and no L2.
    """
    return encoder.encode_image(images, project=False)



def project_patch_clip(encoder: torch.nn.Module, patch_he: torch.Tensor) -> torch.Tensor:
    """patch_emb_clip = image_proj(patch_emb_he), without normalization."""
    if not hasattr(encoder, "image_proj"):
        raise RuntimeError("Encoder has no image_proj; cannot compute patch_emb_clip.")
    z = encoder.image_proj(patch_he)
    return z



def project_cell_clip(encoder: torch.nn.Module, cell_he: torch.Tensor) -> torch.Tensor:
    """z_cell = img_tok_proj(cell_he), without normalization."""
    if not hasattr(encoder, "img_tok_proj") or encoder.img_tok_proj is None:
        raise RuntimeError("Encoder has no img_tok_proj; cannot compute cell_emb_clip.")
    z = encoder.img_tok_proj(cell_he)
    return z



def _strip_cls_if_present(token_emb: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
    gh, gw = int(grid_hw[0]), int(grid_hw[1])
    T_expected = gh * gw
    if token_emb.shape[1] == T_expected + 1:
        return token_emb[:, 1:, :]
    if token_emb.shape[1] != T_expected:
        raise RuntimeError(
            f"Token count mismatch: got {token_emb.shape[1]}, expected {T_expected} (or {T_expected+1} with CLS)."
        )
    return token_emb



def read_membership_for_patch_batch(
    parquet_ds: ds.Dataset,
    patch_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns arrays (patch_local_idx, cell_id_int, token_index, nuc_px) for rows
    where patch_id is in patch_ids.
    """
    if len(patch_ids) == 0:
        return (np.zeros((0,), np.int64),
                np.zeros((0,), np.int32),
                np.zeros((0,), np.int32),
                np.zeros((0,), np.int32))

    patch_set = pa.array(patch_ids, type=pa.string())
    expr = ds.field("patch_id").isin(patch_set)
    table = parquet_ds.to_table(
        columns=["patch_id", "cell_id_int", "token_index", "nuc_px"],
        filter=expr,
    )
    if table.num_rows == 0:
        return (np.zeros((0,), np.int64),
                np.zeros((0,), np.int32),
                np.zeros((0,), np.int32),
                np.zeros((0,), np.int32))

    # patch_local_idx in [0..B-1], vectorized in Arrow
    patch_local = pc.index_in(table["patch_id"], value_set=patch_set).to_numpy(zero_copy_only=False)
    cell_id_int = table["cell_id_int"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    token_index = table["token_index"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    nuc_px = table["nuc_px"].to_numpy(zero_copy_only=False).astype(np.int32, copy=False)

    # Drop rows whose patch_id is not in patch_ids (-1)
    ok = patch_local >= 0
    if not np.all(ok):
        patch_local = patch_local[ok]
        cell_id_int = cell_id_int[ok]
        token_index = token_index[ok]
        nuc_px = nuc_px[ok]
    return patch_local.astype(np.int64, copy=False), cell_id_int, token_index, nuc_px



def extract_embeddings_for_slide(
    encoder: torch.nn.Module,
    slide_h5: Path,
    out_h5: Path,
    eval_root: Path,
    membership_group: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    out_dtype: str,
    model_checkpoint: str,
):
    """
    Run the H&E encoder over all patches of a single slide and save embeddings.
    (preprocessing is applied in the DataLoader; encoder expects already-preprocessed input)

    Args:
        encoder: H&E encoder module (already on `device` and in eval mode).
        slide_h5: path to patches/{slide_id}.h5
        out_h5: path to output embeddings file
        eval_root: root directory for evaluation data
        membership_group: which membership folder to look for (e.g. "uni2h", "virchow2", etc.)
        device: computation device
        batch_size: inference batch size
        num_workers: dataloader num_workers
        out_dtype: 'fp16' or 'fp32' for embeddings on disk
        model_checkpoint: string identifier or path for the encoder checkpoint (stored in HDF5 attrs)
    """

    # Check for membership parquet to decide whether tokens must be loaded and cells pooled
    slide_id = slide_h5.stem
    mpath = membership_parquet_path(eval_root, membership_group, slide_id)
    has_membership = mpath.exists()
    print(f"Slide '{slide_id}': membership_parquet={has_membership}")

    # Projection heads may be absent (pretrained encoder, or no token head when lambda_token = 0)
    has_image_proj = hasattr(encoder, "image_proj") and (getattr(encoder, "image_proj") is not None)
    has_token_proj = hasattr(encoder, "img_tok_proj") and (getattr(encoder, "img_tok_proj") is not None)

    # Dataset
    dataset = H5PatchesDataset(slide_h5)
    n_patches = len(dataset)
    if n_patches == 0:
        warnings.warn(f"No patches found in {slide_h5}, skipping.")
        return

    collate_fn = _make_collate_for_encoder(encoder)

    # Dataloader
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=_dataloader_worker_init_fn,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(**loader_kwargs)

    # Map out_dtype -> numpy dtype for storage
    if out_dtype == "fp16":
        np_dtype = np.float16
    else:
        np_dtype = np.float32

    out_h5.parent.mkdir(parents=True, exist_ok=True)

    # Embedding dims are needed to preallocate the patch datasets
    first_batch = True
    patch_offset = 0

    parquet_ds: Optional[ds.Dataset] = None
    grid_hw_seen: Optional[Tuple[int, int]] = None
    patch_cell_he_dim: Optional[int] = None
    patch_cell_clip_dim: Optional[int] = None

    if has_membership:
          parquet_ds = ds.dataset(str(mpath), format="parquet")
          print(f"  Loaded membership parquet from {mpath}")

    with h5py.File(out_h5, "w") as f_out, torch.inference_mode():

        # Patch-cell group datasets (append-only, chunked)
        # Created lazily on first non-empty group batch
        pc_initialized = False
        pc_size = 0
        pc_cap = 0
        
        for imgs, patch_ids in tqdm(loader, desc=f"  Embedding {slide_id}", total=len(loader)):

            patch_base = int(patch_offset)  # global patch index start for this batch
            
            imgs = imgs.to(device, non_blocking=True)

            # One forward pass:
            # - always compute patch_emb_he (no proj, no L2)
            # - if membership exists: also get raw tokens for cell pooling (no proj, no L2)
            if has_membership:
                patch_he, token_he, grid_hw = encode_batch_raw_with_tokens(encoder, imgs)
                if token_he is None or grid_hw is None:
                    raise RuntimeError("Membership exists but encoder did not return tokens/grid_hw.")
                if grid_hw_seen is None:
                    grid_hw_seen = (int(grid_hw[0]), int(grid_hw[1]))
                else:
                    if (int(grid_hw[0]), int(grid_hw[1])) != grid_hw_seen:
                        raise RuntimeError(f"grid_hw changed across batches: {grid_hw_seen} -> {(int(grid_hw[0]), int(grid_hw[1]))}")
                token_he = _strip_cls_if_present(token_he, grid_hw_seen)
                if patch_cell_he_dim is None:
                    patch_cell_he_dim = int(token_he.shape[-1])
                if patch_cell_clip_dim is None:
                    patch_cell_clip_dim = (
                        int(getattr(encoder.img_tok_proj, "out_features"))
                        if has_token_proj else 0
                    )
            else:
                patch_he = encode_batch_raw_no_tokens(encoder, imgs)
                token_he, grid_hw = None, None

            # Compute patch_emb_clip (projection, no L2) from patch_emb_he
            patch_clip = project_patch_clip(encoder, patch_he) if has_image_proj else None

            # Move back to CPU and cast for storage (single step)
            patch_he_np = _tensor_to_numpy_for_storage(patch_he, np_dtype)
            patch_clip_np = _tensor_to_numpy_for_storage(patch_clip, np_dtype) if patch_clip is not None else None

            B, D_he = patch_he_np.shape
            D_clip = 0 if patch_clip_np is None else int(patch_clip_np.shape[1])

            if first_batch:
                # Allocate patch-level datasets
                str_dt = h5py.string_dtype(encoding="utf-8")
                f_out.create_dataset(
                    "patch_id",
                    shape=(n_patches,),
                    dtype=str_dt,
                )
                f_out.create_dataset(
                    "patch_emb_he",
                    shape=(n_patches, D_he),
                    dtype=np_dtype,
                    compression="gzip",
                    compression_opts=4,
                    chunks=True,
                )

                if D_clip > 0:
                    f_out.create_dataset(
                        "patch_emb_he_clip",
                        shape=(n_patches, D_clip),
                        dtype=np_dtype,
                        compression="gzip",
                        compression_opts=4,
                        chunks=True,
                    )

                # Common attributes
                f_out.attrs["patch_he_dim"] = int(D_he)
                f_out.attrs["patch_he_clip_dim"] = int(D_clip)
                f_out.attrs["has_image_proj"] = bool(has_image_proj)
                f_out.attrs["model_checkpoint"] = model_checkpoint
                f_out.attrs["patches_file"] = str(slide_h5)
                f_out.attrs["membership_parquet"] = str(mpath) if has_membership else ""
                f_out.attrs["preprocess_backend"] = "albumentations_opencv"

                # Allocate cell-level datasets at the end (after aggregation), but record intent
                f_out.attrs["has_cells"] = bool(has_membership)
                first_batch = False

            # Write patch-level embeddings and ids
            patch_slice = slice(patch_offset, patch_offset + B)
            f_out["patch_emb_he"][patch_slice, :] = patch_he_np
            if D_clip > 0:
                f_out["patch_emb_he_clip"][patch_slice, :] = patch_clip_np
            f_out["patch_id"][patch_slice] = list(patch_ids)

            # Cell aggregation (STHELAR only)
            if has_membership:
                assert parquet_ds is not None
                assert token_he is not None
                assert grid_hw_seen is not None

                # Read membership rows for this patch batch
                patch_ids_list = list(patch_ids)
                patch_local, cell_id_int, token_index, nuc_px = read_membership_for_patch_batch(
                    parquet_ds=parquet_ds,
                    patch_ids=patch_ids_list,
                )

                if patch_local.shape[0] > 0:

                    # Check token indices (a mismatch comes from a crop/resize different from the membership build)
                    max_tok = int(token_index.max())
                    # token_he is after CLS strip => shape [B, T, Ct]
                    T = int(token_he.shape[1])
                    if max_tok >= T:
                        raise RuntimeError(f"Membership token_index out of bounds: max token_index={max_tok} but model returned T={T} tokens.")

                    # Move membership arrays to device for fast token gather/pooling
                    patch_local_t = torch.from_numpy(patch_local).to(device=device, dtype=torch.long)
                    cell_id_t = torch.from_numpy(cell_id_int.astype(np.int64, copy=False)).to(device=device, dtype=torch.long)
                    tok_t = torch.from_numpy(token_index.astype(np.int64, copy=False)).to(device=device, dtype=torch.long)
                    nuc_t = torch.from_numpy(nuc_px.astype(np.float32, copy=False)).to(device=device, dtype=torch.float32)

                    # Group by (patch_local, cell_id_int)
                    key = (patch_local_t << 32) | (cell_id_t & 0xFFFFFFFF)
                    uniq_key, inv = torch.unique(key, return_inverse=True)

                    # coverage alpha_p = sum nuc_px within (patch, cell)
                    alpha = torch.zeros((uniq_key.shape[0],), device=device, dtype=torch.float32)
                    alpha.scatter_add_(0, inv, nuc_t)

                    # weights within each (patch,cell): nuc_px / sum(nuc_px)
                    w = nuc_t / (alpha[inv] + 1e-6)

                    # gather raw tokens: token_he[patch_local, token_index]
                    tok_vec = token_he[patch_local_t, tok_t, :]  # [N_rows, Ct]
                    Ct = tok_vec.shape[-1]

                    # pooled mean per (patch,cell)
                    pooled = torch.zeros((uniq_key.shape[0], Ct), device=device, dtype=torch.float32)
                    pooled.scatter_add_(
                        0,
                        inv.unsqueeze(1).expand(-1, Ct),
                        tok_vec.to(dtype=torch.float32) * w.unsqueeze(1),
                    )  # pooled == weighted mean (sum weights == 1)

                    # Decode group identifiers
                    patch_local_group = (uniq_key >> 32).to(dtype=torch.long)  # [G]
                    cell_id_group = (uniq_key & 0xFFFFFFFF).to(dtype=torch.long)  # [G]
                    patch_index_global = patch_local_group + patch_base  # [G] index into patch_id arrays

                    # Optional: token-projected representation (no L2)
                    z_patch_cell = None
                    if has_token_proj:
                        z_patch_cell = project_cell_clip(encoder, pooled)  # [G, D_tok]

                    # Move small group-level tensors to CPU for HDF5 append
                    patch_index_np = patch_index_global.cpu().numpy().astype(np.int32, copy=False)
                    cell_id_np = cell_id_group.cpu().numpy().astype(np.int32, copy=False)
                    alpha_np = alpha.cpu().numpy().astype(np.float32, copy=False)

                    pooled_np = _tensor_to_numpy_for_storage(pooled, np_dtype)
                    if z_patch_cell is not None:
                        z_np = _tensor_to_numpy_for_storage(z_patch_cell, np_dtype)
                        D_tok = int(z_np.shape[1])
                    else:
                        z_np = None
                        D_tok = 0

                    G = int(patch_index_np.shape[0])
                    if G > 0:
                        # Lazily create / grow append-only datasets with capacity to reduce resize calls
                        if not pc_initialized:
                            pc_size = 0
                            pc_cap = max(4096, G)
                            f_out.create_dataset(
                                "patch_cell_patch_index",
                                shape=(pc_cap,),
                                maxshape=(None,),
                                dtype=np.int32,
                                compression="gzip",
                                compression_opts=4,
                                chunks=(min(pc_cap, 4096),),
                            )
                            f_out.create_dataset(
                                "patch_cell_cell_id_int",
                                shape=(pc_cap,),
                                maxshape=(None,),
                                dtype=np.int32,
                                compression="gzip",
                                compression_opts=4,
                                chunks=(min(pc_cap, 4096),),
                            )
                            f_out.create_dataset(
                                "patch_cell_alpha",
                                shape=(pc_cap,),
                                maxshape=(None,),
                                dtype=np.float32,
                                compression="gzip",
                                compression_opts=4,
                                chunks=(min(pc_cap, 4096),),
                            )
                            f_out.create_dataset(
                                "patch_cell_emb_he",
                                shape=(pc_cap, int(pooled_np.shape[1])),
                                maxshape=(None, int(pooled_np.shape[1])),
                                dtype=np_dtype,
                                compression="gzip",
                                compression_opts=4,
                                chunks=(min(pc_cap, 4096), int(pooled_np.shape[1])),
                            )
                            f_out.create_dataset(
                                "patch_cell_emb_he_clip",
                                shape=(pc_cap, int(D_tok)),
                                maxshape=(None, int(D_tok)),
                                dtype=np_dtype,
                                compression="gzip",
                                compression_opts=4,
                                chunks=True if int(D_tok) == 0 else (min(pc_cap, 4096), int(D_tok)),
                            )
                            f_out.attrs["patch_cell_he_dim"] = int(pooled_np.shape[1])
                            f_out.attrs["patch_cell_he_clip_dim"] = int(D_tok)
                            f_out.attrs["token_grid_h"] = int(grid_hw_seen[0])
                            f_out.attrs["token_grid_w"] = int(grid_hw_seen[1])
                            pc_initialized = True

                        # Ensure capacity
                        need = pc_size + G
                        if need > pc_cap:
                            new_cap = max(pc_cap * 2, need)
                            f_out["patch_cell_patch_index"].resize((new_cap,))
                            f_out["patch_cell_cell_id_int"].resize((new_cap,))
                            f_out["patch_cell_alpha"].resize((new_cap,))
                            f_out["patch_cell_emb_he"].resize((new_cap, f_out["patch_cell_emb_he"].shape[1]))
                            f_out["patch_cell_emb_he_clip"].resize((new_cap, f_out["patch_cell_emb_he_clip"].shape[1]))
                            pc_cap = new_cap

                        # Append
                        sl = slice(pc_size, pc_size + G)
                        f_out["patch_cell_patch_index"][sl] = patch_index_np
                        f_out["patch_cell_cell_id_int"][sl] = cell_id_np
                        f_out["patch_cell_alpha"][sl] = alpha_np
                        f_out["patch_cell_emb_he"][sl, :] = pooled_np
                        if z_np is not None and f_out["patch_cell_emb_he_clip"].shape[1] > 0:
                            f_out["patch_cell_emb_he_clip"][sl, :] = z_np
                        pc_size += G

            patch_offset += B

        assert patch_offset == n_patches, (f"Written patches ({patch_offset}) does not match total ({n_patches}).")

        # Finalize patch-cell group datasets (shrink to actual size)
        f_out.attrs["slide_id"] = str(slide_id)
        f_out.attrs["membership_parquet"] = str(mpath) if has_membership else ""
        f_out.attrs["has_patch_cells"] = bool(has_membership)
        f_out.attrs["has_token_proj"] = bool(has_token_proj)
        if pc_initialized:
            f_out["patch_cell_patch_index"].resize((pc_size,))
            f_out["patch_cell_cell_id_int"].resize((pc_size,))
            f_out["patch_cell_alpha"].resize((pc_size,))
            f_out["patch_cell_emb_he"].resize((pc_size, f_out["patch_cell_emb_he"].shape[1]))
            f_out["patch_cell_emb_he_clip"].resize((pc_size, f_out["patch_cell_emb_he_clip"].shape[1]))
        else:
            he_dim = int(patch_cell_he_dim or 0)
            clip_dim = int(patch_cell_clip_dim or 0)
            # If membership exists but no groups were found, write empty placeholders for consistency
            f_out.create_dataset("patch_cell_patch_index", shape=(0,), maxshape=(None,), dtype=np.int32)
            f_out.create_dataset("patch_cell_cell_id_int", shape=(0,), maxshape=(None,), dtype=np.int32)
            f_out.create_dataset("patch_cell_alpha", shape=(0,), maxshape=(None,), dtype=np.float32)
            f_out.create_dataset("patch_cell_emb_he", shape=(0, he_dim), maxshape=(None, he_dim), dtype=np_dtype)
            f_out.create_dataset("patch_cell_emb_he_clip", shape=(0, clip_dim), maxshape=(None, clip_dim), dtype=np_dtype)
            f_out.attrs["patch_cell_he_dim"] = he_dim
            f_out.attrs["patch_cell_he_clip_dim"] = clip_dim
            if grid_hw_seen is not None:
                f_out.attrs["token_grid_h"] = int(grid_hw_seen[0])
                f_out.attrs["token_grid_w"] = int(grid_hw_seen[1])

    print(f"  -> Saved embeddings to {out_h5}")




def main(args):

    if sys.platform == "darwin" and args.num_workers > 0:
        warnings.warn(
            "macOS + PyTorch DataLoader workers + Albumentations/OpenCV is fragile. "
            "For stability, forcing num_workers=0 in this script."
        )
        args.num_workers = 0

    # Data directories
    eval_data_root = get_full_path('eval', 'eval_datasets', 'st_eval_data', f'eval_st_{args.training_dataset_id}', scratch=True)
    patches_dir = eval_data_root / "patches"

    if not patches_dir.exists():
        raise FileNotFoundError(f"'patches' directory not found under {eval_data_root}")

    # Output directory
    output_dir = eval_data_root / "he_embeddings" / f"HEembs_t{args.training_num}e{args.checkpoint_num}"

    # Decide device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # small speed-up for fixed-size convolutions on CUDA

    print(f"Using device: {device}")

    # Load exported H&E encoder bundle (exported by extract_encoders.py)
    checkpoint_path = get_full_path('train', 'trainings', 'trainings_HistoST', f'training_{args.training_num}', 'trained_encoders', f'HEencoder_t{args.training_num}e{args.checkpoint_num}.pt')
    encoder = load_he_encoder_bundle(encoder_path=checkpoint_path, device=device)
    print(f"Loaded H&E encoder bundle from {checkpoint_path}")
    print(f"Reconstructed encoder: {type(encoder).__name__}")

    # Print if model contains image_proj and img_tok_proj
    has_image_proj = hasattr(encoder, "image_proj") and (getattr(encoder, "image_proj") is not None)
    has_token_proj = hasattr(encoder, "img_tok_proj") and (getattr(encoder, "img_tok_proj") is not None)
    print(f"Encoder has_image_proj: {has_image_proj}, has_token_proj: {has_token_proj}")

    membership_group = membership_group_for_encoder(encoder)
    print(f"Membership group for encoder: {membership_group}")

    # 1. Glob files, filtering out Mac OS cache files starting with "._"
    slide_h5_files = [
        f for f in sorted(patches_dir.glob("*.h5")) 
        if not f.name.startswith("._")
    ]
    
    if not slide_h5_files:
        raise FileNotFoundError(f"No valid '*.h5' files found in {patches_dir}")

    # 2. Cross-check globbed files with metadata CSV
    metadata_csv_path = eval_data_root / f"metadata_eval_st_{args.training_dataset_id}.csv"
    if not metadata_csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv_path}")
        
    df_meta = pd.read_csv(metadata_csv_path)
    if "slide_id" not in df_meta.columns:
        raise ValueError(f"'slide_id' column not found in {metadata_csv_path}")
        
    csv_slide_ids = set(df_meta["slide_id"].astype(str))
    glob_slide_ids = set(f.stem for f in slide_h5_files)
    
    if csv_slide_ids != glob_slide_ids:
        missing_in_glob = csv_slide_ids - glob_slide_ids
        missing_in_csv = glob_slide_ids - csv_slide_ids
        error_msg = "Mismatch between metadata CSV slide_ids and globbed .h5 files.\n"
        if missing_in_glob:
            error_msg += f"  - In CSV but missing in patches dir: {missing_in_glob}\n"
        if missing_in_csv:
            error_msg += f"  - In patches dir but missing in CSV: {missing_in_csv}\n"
        raise ValueError(error_msg)

    print(f"Found {len(slide_h5_files)} valid slide(s) in {patches_dir} (perfectly matching metadata)")

    # 2b. Optional --sets restriction (metadata 'source' or 'subset' column), applied after the check above
    allowed_slide_ids = None
    if args.sets is not None:
        for col in ("source", "subset"):
            if col not in df_meta.columns:
                raise ValueError(f"--sets requires a '{col}' column in {metadata_csv_path}")
        requested = set(args.sets)
        umbrellas = requested & {"STHELAR", "HEST"}          # match the `source` column
        fine = requested - umbrellas                          # match the `subset` column
        mask = (
            df_meta["source"].astype(str).isin(umbrellas)
            | df_meta["subset"].astype(str).isin(fine)
        )
        allowed_slide_ids = set(df_meta.loc[mask, "slide_id"].astype(str))
        unmatched = (
            {t for t in umbrellas if t not in set(df_meta["source"].astype(str))}
            | {t for t in fine if t not in set(df_meta["subset"].astype(str))}
        )
        if unmatched:
            print(f"WARNING: requested --sets not present in metadata (0 slides): {sorted(unmatched)}")
        if not allowed_slide_ids:
            raise ValueError(f"--sets {sorted(requested)} matched zero slides in {metadata_csv_path}")
        print(
            f"--sets restriction active: computing {len(allowed_slide_ids)} of "
            f"{len(slide_h5_files)} slides (requested: {sorted(requested)})."
        )

    for i, slide_h5 in enumerate(slide_h5_files):
        print(f"\n* Processing slide: {slide_h5.name} ({i + 1} / {len(slide_h5_files)})")
        
        slide_id = slide_h5.stem

        # Skip slides not in the requested --sets (if any restriction is active).
        if allowed_slide_ids is not None and slide_id not in allowed_slide_ids:
            print(f"  -> Skipping '{slide_id}', not in requested --sets.")
            continue

        out_h5 = output_dir / f"{slide_id}.h5"

        # 3. Resume capability: Skip if output embeddings already exist
        if out_h5.exists():
            print(f"  -> Skipping '{slide_id}', output file already exists.")
            continue

        extract_embeddings_for_slide(
            encoder=encoder,
            slide_h5=slide_h5,
            out_h5=out_h5,
            eval_root=eval_data_root,
            membership_group=membership_group,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            out_dtype=args.dtype,
            model_checkpoint=str(checkpoint_path),
        )



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract H&E encoder embeddings for HEST & STHELAR evaluation slides.")

    # Data

    parser.add_argument("--training_dataset_id", type=str,
                        required=True,
                        help="Training dataset ID used (e.g. 'ds_ct1_20x_224_o112_1')")
    
    parser.add_argument("--training_num", type=str,
                        required=True,
                        help="Num from HistoST training_id (e.g. '46' for training_id=='training_46')")
    
    parser.add_argument("--checkpoint_num", type=str,
                        required=True,
                        help="Checkpoint number to load (e.g. '13' for 'checkpoint_epoch_13.pth')")

    parser.add_argument("--sets", type=str, nargs="+",
                        default=None,
                        choices=["STHELAR", "HEST",
                                 "STHELAR_train", "STHELAR_test",
                                 "other_technologies", "other_preservations",
                                 "other_magnifications", "other_species", "other_organs"],
                        help="Data sets to compute embeddings for (default: all): 'STHELAR' / 'HEST' match the "
                             "metadata 'source' column, the other values its 'subset' column.")

    # Technical parameters
    
    parser.add_argument("--batch_size", type=int,
                        default=256,
                        help="Batch size for encoder inference.")
    
    parser.add_argument("--num_workers", type=int,
                        default=22,
                        help="Number of DataLoader workers.")
    
    parser.add_argument("--device", type=str,
                        default=None,
                        help="Device for inference (e.g. 'cuda', 'cuda:0', 'cpu', 'mps'). Default: cuda, else mps, else cpu.")
    
    parser.add_argument("--dtype", type=str,
                        default="fp32",
                        choices=["fp16", "fp32"],
                        help="Numeric precision of the embeddings on disk (HDF5): fp16 or fp32 (default: fp32).")

    args = parser.parse_args()
    main(args)
