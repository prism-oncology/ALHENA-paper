#!/usr/bin/env python3
"""
Infer the Novae domains of the cells of the STHELAR eval slides (scConcept node features, fresh radius graph) and convert
them into patch-level labels for each H&E patch by nucleus-pixel-weighted vote (unassigned cells do not vote).

Saved in <ROOT_HistoST> / eval / eval_datasets / st_eval_data / eval_st_<training_dataset_id> / novae_domains_HEpatches / <training_id> /
    - <slide_id>.parquet: patch domain (novae_patch_domain_<K>) and winner fraction (novae_patch_domain_frac_<K>)
    - manifest.json: parameters and Novae cell-domain keys per slide
"""

import argparse
import gc
import io
import json
import sys
import traceback
import warnings
import zipfile
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from anndata import AnnData, read_h5ad
from scipy.sparse import load_npz
from tqdm import tqdm

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.cell_ids import int_cell_id, int_cell_id_vec

import novae
from novae._constants import Keys

# Novae GAT node features: precomputed per-cell scConcept embeddings (obsm["X_scConcept"])
EMBEDDING_DIM = 512          # scConcept node-feature dim (corpus40M-model30M)
OBSM_KEY = "X_scConcept"     # per-cell scConcept node feature (the Novae GAT input)
SLIDE_KEY = "slide_id"       # passed to novae.spatial_neighbors as slide_key (real slide id)

try:
    import torch
except Exception:
    torch = None


# -----------------------------
# Helpers
# -----------------------------


def read_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_novae_scconcept_provenance(training_id: str) -> Dict[str, object]:
    """scConcept node-feature model and graph radius of the Novae teacher, read and checked from trainings_Novae/<training_id>/manifest.json."""
    mp = get_full_path("train", "trainings", "trainings_Novae", training_id, "manifest.json")
    if not mp.exists():
        raise FileNotFoundError(
            f"Novae manifest not found: {mp}. It is written by _2_novae_pretraining.py and gives the "
            "scConcept node-feature model."
        )
    with open(mp, "r") as f:
        manifest = json.load(f)
    scc = manifest.get("scConcept") or {}
    scc_id = scc.get("own_scconcept_training_id")
    if not scc_id:
        raise KeyError(
            f"Novae manifest {mp} has no scConcept.own_scconcept_training_id (not a scConcept-node run?)."
        )
    if int(scc.get("adapt_steps", -1)) != 0:
        raise ValueError(f"scConcept teacher must be raw (adapt_steps==0); got {scc.get('adapt_steps')}.")
    if int(scc.get("embedding_size", -1)) != EMBEDDING_DIM:
        raise ValueError(f"scConcept embedding_size={scc.get('embedding_size')} != {EMBEDDING_DIM}.")
    graph = manifest.get("graph") or {}
    return {
        "scconcept_training_id": str(scc_id),
        "model_name": scc.get("model_name"),
        "scConcept_version": scc.get("scConcept_version"),
        "species": scc.get("species"),
        "radius_um": graph.get("radius_um"),
    }


def attach_scconcept_obsm(adata: AnnData, scc_h5_path: Path) -> None:
    """Attach obsm['X_scConcept'] (aligned on obs cell_id) from a scConcept <slide_id>.h5; raises on missing cells."""
    if "cell_id" not in adata.obs:
        raise KeyError("adata.obs['cell_id'] missing (needed for the scConcept join).")
    cell_int = int_cell_id_vec(adata.obs["cell_id"].astype(str).values)
    with h5py.File(scc_h5_path, "r") as f:
        if "cell_id_int" not in f or "cls_emb" not in f:
            raise KeyError(f"Invalid scConcept h5 (missing 'cell_id_int'/'cls_emb'): {scc_h5_path}")
        cid = f["cell_id_int"][:].astype(np.int64)
        if cid.size > 1 and np.any(cid[1:] < cid[:-1]):
            raise ValueError(f"'cell_id_int' must be sorted ascending in {scc_h5_path}")
        emb_dim = int(f.attrs["embedding_dim"]) if "embedding_dim" in f.attrs else int(f["cls_emb"].shape[1])
        if emb_dim != EMBEDDING_DIM:
            raise ValueError(f"scConcept embedding_dim={emb_dim} != {EMBEDDING_DIM} in {scc_h5_path}")
        idx = np.searchsorted(cid, cell_int)
        idx_clip = np.clip(idx, 0, cid.shape[0] - 1)
        ok = (idx < cid.shape[0]) & (cid[idx_clip] == cell_int)
        if not np.all(ok):
            missing = cell_int[~ok]
            raise KeyError(
                f"{missing.size} cell_id_int not found in {Path(scc_h5_path).name}; "
                f"e.g. {missing[:5].tolist()}. The scConcept run must cover all cells of this eval slide."
            )
        uniq, inv = np.unique(idx.astype(np.int64), return_inverse=True)
        emb = f["cls_emb"][uniq][inv]  # (N, 512), float16 as stored (L2-normalized)
    adata.obsm[OBSM_KEY] = np.asarray(emb, dtype=np.float16)


def get_requested_n_domains(config_path: Path, override: Sequence[int] | None) -> List[int]:
    if override is not None and len(override) > 0:
        out = [int(x) for x in override]
        if len(out) == 0:
            raise ValueError("--n_domains was provided but empty.")
        return out

    cfg = read_yaml(config_path)
    post = cfg.get("post_training", {})
    n_domains = post.get("n_domains", None)
    if n_domains is None:
        raise KeyError(
            f"Could not find 'post_training.n_domains' in config: {config_path}. "
            "Provide --n_domains explicitly."
        )
    out = [int(x) for x in n_domains]
    if len(out) == 0:
        raise ValueError(f"'post_training.n_domains' is empty in config: {config_path}")
    return out


def decode_h5_strings(arr: np.ndarray) -> List[str]:
    out: List[str] = []
    for x in arr.reshape(-1):
        if isinstance(x, (bytes, np.bytes_)):
            out.append(bytes(x).decode("utf-8"))
        else:
            out.append(str(x))
    return out


def load_eval_patch_ids(eval_patch_h5: Path) -> List[str]:
    """
    Read only the HEST-style barcode dataset from eval patches/{slide_id}.h5
    """
    if not eval_patch_h5.exists():
        raise FileNotFoundError(f"Eval patch file not found: {eval_patch_h5}")
    with h5py.File(eval_patch_h5, "r") as f:
        if "barcode" not in f:
            raise KeyError(f"'barcode' dataset not found in {eval_patch_h5}")
        patch_ids = decode_h5_strings(f["barcode"][:])

    if len(patch_ids) == 0:
        raise ValueError(f"No patch IDs found in {eval_patch_h5}")

    if len(set(patch_ids)) != len(patch_ids):
        dup = pd.Series(patch_ids)[pd.Series(patch_ids).duplicated()].unique().tolist()[:10]
        raise ValueError(f"Duplicate patch IDs found in {eval_patch_h5}. Examples: {dup}")

    return patch_ids


def ensure_slide_id_key(adata: AnnData) -> None:
    """Add the Novae slide-id key from obs['slide_id'] if missing."""
    if Keys.SLIDE_ID not in adata.obs:
        if "slide_id" not in adata.obs:
            raise KeyError("Expected adata.obs['slide_id'] to exist.")
        adata.obs[Keys.SLIDE_ID] = adata.obs["slide_id"].astype("category")


def prepare_slide_for_inference(adata: AnnData, scc_h5_path: Path, radius: float) -> AnnData:
    """
    Prepare one whole-slide AnnData for Novae inference: attach obsm['X_scConcept'], set the slide-id key and build
    a fresh Delaunay graph of radius `radius` µm (novae.spatial_neighbors, as in the teacher pretraining).
    """
    ad = adata
    if not ad.obs_names.is_unique:
        ad.obs_names_make_unique()
    if "spatial" not in ad.obsm:
        raise KeyError("adata.obsm['spatial'] missing (needed for novae.spatial_neighbors).")

    attach_scconcept_obsm(ad, Path(scc_h5_path))
    ensure_slide_id_key(ad)

    # Rebuild the graph at the teacher's radius (the stored obsp may use another radius)
    for k in list(ad.obsp.keys()):
        del ad.obsp[k]
    print(f"* Building fresh radius={radius} µm Delaunay graph (slide_key={SLIDE_KEY}) ...")
    novae.spatial_neighbors(ad, slide_key=SLIDE_KEY, radius=float(radius))
    return ad


def load_trained_novae_model(ckpt_path: Path):
    """Load the trained Novae checkpoint in pretrained-inference mode (node features: obsm['X_scConcept'], set in the checkpoint)."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = novae.Novae.load_from_checkpoint(str(ckpt_path), map_location="cpu", strict=False)

    # zero_shot=False, trained=True, pretrained=True (required by compute_representations and assign_domains)
    model.mode.as_pretrained()
    model.eval()
    return model


def read_sparse_mask_value_counts_from_zip(mask_zip: zipfile.ZipFile, patch_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read patch mask {patch_id}.npz from masks_cell_ids_nuclei.zip and return:
        unique_cell_ids_int, nucleus_pixel_counts
    Only uses sparse .data values, so it avoids densifying the mask.
    """
    mask_name = f"{patch_id}.npz"
    try:
        raw = mask_zip.read(mask_name)
    except KeyError as e:
        raise KeyError(f"Mask '{mask_name}' not found in masks_cell_ids_nuclei.zip") from e

    sp = load_npz(io.BytesIO(raw)).tocoo()
    vals = np.asarray(sp.data)

    if vals.size == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int32),
        )

    vals = vals[vals > 0]
    if vals.size == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int32),
        )

    unique_ids, counts = np.unique(vals.astype(np.int64, copy=False), return_counts=True)
    return unique_ids.astype(np.int64, copy=False), counts.astype(np.int32, copy=False)


def infer_cell_domains_for_slide(
    model,
    adata: AnnData,
    requested_n_domains: Sequence[int],
    accelerator: str,
    num_workers: int,
) -> Tuple[np.ndarray, Dict[int, str], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Run Novae inference on one whole slide and prepare compact encodings for fast patch voting.

    Returns
    -------
    sorted_cell_ids_int:
        sorted int cell IDs, shape (n_cells,)
    req_to_actual_key:
        {requested_n_domains -> actual adata.obs key returned by model.assign_domains}
    actual_key_to_codes_sorted:
        {actual_key -> int domain code per cell, aligned with sorted_cell_ids_int}
    actual_key_to_codebook:
        {actual_key -> np.array of domain labels indexed by code}
    """
    # Compute latent + leaves
    model.compute_representations(
        adata,
        zero_shot=False,
        accelerator=accelerator,
        num_workers=num_workers,
    )

    if Keys.LEAVES not in adata.obs:
        raise RuntimeError(
            f"Novae inference did not create adata.obs['{Keys.LEAVES}'] on slide "
            f"{adata.obs['slide_id'].iloc[0] if 'slide_id' in adata.obs else '<unknown>'}."
        )

    # Assign requested resolutions
    req_to_actual_key: Dict[int, str] = {}
    for nd in requested_n_domains:
        actual_key = model.assign_domains(adata, n_domains=int(nd))
        if actual_key not in adata.obs:
            raise RuntimeError(
                f"Novae returned key '{actual_key}' for n_domains={nd}, "
                "but it is not present in adata.obs."
            )
        req_to_actual_key[int(nd)] = str(actual_key)

    # Prepare integer cell IDs
    if "cell_id" not in adata.obs:
        raise KeyError("Expected adata.obs['cell_id'] to exist.")

    cell_ids_int = np.fromiter(
        (int_cell_id(str(x)) for x in adata.obs["cell_id"].astype(str).values),
        dtype=np.int64,
        count=adata.n_obs,
    )

    if np.unique(cell_ids_int).size != adata.n_obs:
        raise ValueError("Duplicate integer cell IDs found after converting adata.obs['cell_id'].")

    order = np.argsort(cell_ids_int)
    sorted_cell_ids_int = cell_ids_int[order]

    actual_key_to_codes_sorted: Dict[str, np.ndarray] = {}
    actual_key_to_codebook: Dict[str, np.ndarray] = {}

    for actual_key in sorted(set(req_to_actual_key.values())):
        labels = adata.obs[actual_key].astype(str).values
        # Unassigned cells (NaN domain) get code -1 and are excluded from the patch vote
        codes, uniques = pd.factorize(labels, sort=True)
        n_unassigned = int(np.count_nonzero(codes < 0))
        if n_unassigned:
            print(f"  [{actual_key}] {n_unassigned}/{codes.size} cells unassigned "
                  f"(invalid neighbourhood) -> excluded from patch voting")

        actual_key_to_codes_sorted[actual_key] = codes.astype(np.int32, copy=False)[order]
        actual_key_to_codebook[actual_key] = np.asarray(uniques, dtype=object)

    return sorted_cell_ids_int, req_to_actual_key, actual_key_to_codes_sorted, actual_key_to_codebook


def patch_vote_from_cell_domains(
    cell_ids_in_patch: np.ndarray,
    nuc_px_in_patch: np.ndarray,
    sorted_cell_ids_int: np.ndarray,
    codes_sorted: np.ndarray,
    codebook: np.ndarray,
) -> Tuple[str | None, float]:
    """
    Nucleus-pixel-weighted majority vote for one patch and one requested domain resolution.
    """
    if cell_ids_in_patch.size == 0:
        return None, np.nan

    idx = np.searchsorted(sorted_cell_ids_int, cell_ids_in_patch)
    valid = idx < sorted_cell_ids_int.size
    if np.any(valid):
        valid_idx = idx[valid]
        valid[valid] &= (sorted_cell_ids_int[valid_idx] == cell_ids_in_patch[valid])

    if not np.all(valid):
        missing = cell_ids_in_patch[~valid][:10].tolist()
        raise KeyError(
            "Some cell IDs found in the patch mask are missing from the slide AnnData after preprocessing/inference. "
            f"Examples: {missing}"
        )

    dom_codes = codes_sorted[idx]

    # Unassigned cells (code -1) do not vote; a patch with only unassigned cells gets no domain
    assigned = dom_codes >= 0
    if not np.any(assigned):
        return None, np.nan
    dom_codes = dom_codes[assigned]
    nuc_px_in_patch = nuc_px_in_patch[assigned]

    weights = np.bincount(
        dom_codes,
        weights=nuc_px_in_patch.astype(np.float64, copy=False),
        minlength=codebook.shape[0],
    )
    winner = int(np.argmax(weights))
    total = float(weights.sum())

    if total <= 0:
        return None, np.nan

    return str(codebook[winner]), float(weights[winner] / total)


def write_slide_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        str(out_path),
        compression="zstd",
        use_dictionary=True,
    )


def cleanup_memory() -> None:
    """
    Memory cleanup for CPU / CUDA / MPS.
    """
    gc.collect()

    if torch is None:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def process_single_slide(
    slide_id: str,
    eval_patch_h5: str,
    slide_h5ad: str,
    masks_zip_path: str,
    ckpt_path: str,
    out_path: str,
    scc_h5_path: str,
    requested_n_domains: Sequence[int],
    accelerator: str,
    num_workers: int,
    radius: float,
) -> Dict[str, object]:
    """
    Process exactly one slide in a fresh process so all memory is released
    when the child exits.
    """
    eval_patch_h5 = Path(eval_patch_h5)
    slide_h5ad = Path(slide_h5ad)
    masks_zip_path = Path(masks_zip_path)
    ckpt_path = Path(ckpt_path)
    out_path = Path(out_path)

    try:
        try:
            novae.settings.auto_preprocessing = False
        except Exception:
            warnings.warn("Could not set novae.settings.auto_preprocessing = False. Continuing anyway.")

        print("* Loading trained Novae model ...")
        model = load_trained_novae_model(ckpt_path)

        print("* Reading eval patch IDs ...")
        patch_ids = load_eval_patch_ids(eval_patch_h5)
        print(f"  n_patches = {len(patch_ids)}")

        print("* Reading whole-slide graph AnnData ...")
        ad = read_h5ad(slide_h5ad)

        if "slide_id" not in ad.obs.columns:
            ad.obs["slide_id"] = str(slide_id)

        print("* Attaching scConcept node features (obsm['X_scConcept']) + fresh graph ...")
        ad = prepare_slide_for_inference(ad, scc_h5_path=Path(scc_h5_path), radius=float(radius))

        print("* Running Novae inference on whole slide ...")
        sorted_cell_ids_int, req_to_actual_key, actual_key_to_codes_sorted, actual_key_to_codebook = infer_cell_domains_for_slide(
            model=model,
            adata=ad,
            requested_n_domains=requested_n_domains,
            accelerator=str(accelerator),
            num_workers=int(num_workers),
        )

        cols: Dict[str, list] = {
            "slide_id": [],
            "patch_id": [],
            "n_cells": [],
            "total_nuc_px": [],
        }
        for nd in requested_n_domains:
            cols[f"novae_patch_domain_{int(nd)}"] = []
            cols[f"novae_patch_domain_frac_{int(nd)}"] = []

        print("* Aggregating cell domains -> patch domains ...")
        with zipfile.ZipFile(masks_zip_path, "r") as mask_zip:
            for patch_id in tqdm(patch_ids, desc=f"Patch voting ({slide_id})"):
                cell_ids_in_patch, nuc_px_in_patch = read_sparse_mask_value_counts_from_zip(mask_zip, patch_id)

                cols["slide_id"].append(slide_id)
                cols["patch_id"].append(patch_id)
                cols["n_cells"].append(int(cell_ids_in_patch.size))
                cols["total_nuc_px"].append(int(nuc_px_in_patch.sum()) if nuc_px_in_patch.size > 0 else 0)

                for nd in requested_n_domains:
                    actual_key = req_to_actual_key[int(nd)]
                    label, frac = patch_vote_from_cell_domains(
                        cell_ids_in_patch=cell_ids_in_patch,
                        nuc_px_in_patch=nuc_px_in_patch,
                        sorted_cell_ids_int=sorted_cell_ids_int,
                        codes_sorted=actual_key_to_codes_sorted[actual_key],
                        codebook=actual_key_to_codebook[actual_key],
                    )
                    cols[f"novae_patch_domain_{int(nd)}"].append(label)
                    cols[f"novae_patch_domain_frac_{int(nd)}"].append(frac)

        df = pd.DataFrame(cols)

        df["slide_id"] = df["slide_id"].astype("string")
        df["patch_id"] = df["patch_id"].astype("string")
        df["n_cells"] = df["n_cells"].astype(np.int32)
        df["total_nuc_px"] = df["total_nuc_px"].astype(np.int32)
        for nd in requested_n_domains:
            df[f"novae_patch_domain_{int(nd)}"] = df[f"novae_patch_domain_{int(nd)}"].astype("string")
            df[f"novae_patch_domain_frac_{int(nd)}"] = df[f"novae_patch_domain_frac_{int(nd)}"].astype(np.float32)

        print(f"* Writing parquet -> {out_path}")
        write_slide_parquet(df, out_path)

        # Patches without domain (no cells or only unassigned cells), counted on the first resolution
        first_dom_col = f"novae_patch_domain_{int(requested_n_domains[0])}"
        n_patches_unlabeled = int(df[first_dom_col].isna().sum()) if first_dom_col in df.columns else 0

        result = {
            "output_parquet": str(out_path),
            "status": "written",
            "n_patches": int(len(df)),
            "n_patches_unlabeled": n_patches_unlabeled,
            "requested_to_actual_cell_domain_key": {
                str(int(k)): str(v) for k, v in req_to_actual_key.items()
            },
        }

        del model
        del ad
        del df
        del cols
        del patch_ids
        del sorted_cell_ids_int
        del req_to_actual_key
        del actual_key_to_codes_sorted
        del actual_key_to_codebook
        cleanup_memory()

        return result

    finally:
        cleanup_memory()


def _slide_worker(slide_id: str, worker_kwargs: Dict[str, object], conn) -> None:
    """
    Child process entrypoint. Sends back either:
      {"ok": True, "result": ...}
    or
      {"ok": False, "error": "...traceback..."}
    """
    try:
        result = process_single_slide(slide_id=slide_id, **worker_kwargs)
        conn.send({"ok": True, "result": result})
    except Exception:
        conn.send({"ok": False, "error": traceback.format_exc()})
    finally:
        try:
            conn.close()
        except Exception:
            pass
        cleanup_memory()


# -----------------------------
# Main
# -----------------------------


def main(args) -> None:
    
    dataset_id = str(args.dataset_id)
    training_dataset_id = str(args.training_dataset_id)
    training_id = str(args.training_id)

    # Paths
    eval_root = get_full_path("eval", "eval_datasets", "st_eval_data", f"eval_st_{training_dataset_id}")
    eval_patches_dir = eval_root / "patches"
    metadata_csv = eval_root / f"metadata_eval_st_{training_dataset_id}.csv"

    train_dataset_dir = get_full_path("train", "training_datasets", dataset_id, training_dataset_id)
    masks_zip_path = train_dataset_dir / "masks_cell_ids_nuclei.zip"

    adata_graph_dir = get_full_path("STHELAR_HistoST", "adata_graph")

    novae_run_dir = get_full_path("train", "trainings", "trainings_Novae", training_id)
    config_path = novae_run_dir / f"{training_id}_novae.yaml"
    ckpt_path = novae_run_dir / "checkpoints" / str(args.ckpt)

    # scConcept node-feature model + graph radius of this Novae teacher
    prov = read_novae_scconcept_provenance(training_id)
    scconcept_training_id = prov["scconcept_training_id"]
    radius = float(args.radius) if args.radius is not None else float(prov["radius_um"] or 100.0)
    scconcept_dir = get_full_path("train", "trainings", "trainings_scConcept", scconcept_training_id, "embeddings")
    print(f"scConcept node features: training_id={scconcept_training_id} "
          f"(model={prov['model_name']} v{prov['scConcept_version']}) | graph radius={radius} µm")

    out_dir = get_full_path("eval", "eval_datasets", "st_eval_data", f"eval_st_{training_dataset_id}", "novae_domains_HEpatches", training_id)
    manifest_path = out_dir / "manifest.json"

    # Checks
    for p in [eval_root, eval_patches_dir, metadata_csv, train_dataset_dir, masks_zip_path, adata_graph_dir, novae_run_dir, config_path, ckpt_path, scconcept_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Required path does not exist: {p}")

    # Read metadata (STHELAR slides only)
    metadata = pd.read_csv(metadata_csv)
    if "slide_id" not in metadata.columns:
        raise KeyError(f"'slide_id' column missing from {metadata_csv}")
    if "source" in metadata.columns:
        metadata = metadata[metadata["source"] == "STHELAR"].copy()

    slide_ids = sorted(metadata["slide_id"].astype(str).unique().tolist())
    if len(slide_ids) == 0:
        raise RuntimeError(f"No STHELAR slides found in {metadata_csv}")

    requested_n_domains = get_requested_n_domains(config_path=config_path, override=args.n_domains)

    # Match the training behavior
    try:
        novae.settings.auto_preprocessing = False
    except Exception:
        warnings.warn("Could not set novae.settings.auto_preprocessing = False. Continuing anyway.")

    print("\n====== Novae slide inference ======")
    print(f"Training ID           : {training_id}")
    print(f"Checkpoint            : {ckpt_path}")
    print(f"Requested n_domains   : {requested_n_domains}")
    print("Execution mode        : 1 fresh subprocess per slide")
    print(f"Novae num_workers     : {int(args.num_workers)}")

    # Manifest
    manifest = {
        "dataset_id": dataset_id,
        "training_dataset_id": training_dataset_id,
        "training_id": training_id,
        "config_path": str(config_path),
        "checkpoint_path": str(ckpt_path),
        "requested_n_domains": [int(x) for x in requested_n_domains],
        "graph_radius_um": float(radius),
        "scConcept": {
            "scconcept_training_id": scconcept_training_id,
            "model_name": prov["model_name"],
            "scConcept_version": prov["scConcept_version"],
            "species": prov["species"],
            "embedding_size": EMBEDDING_DIM,
            "obsm_key": OBSM_KEY,
        },
        "slides": {},
    }

    ctx = get_context("spawn")

    # Process each slide
    print("\n====== Processing STHELAR eval slides ======")
    for i, slide_id in enumerate(slide_ids, start=1):
        print(f"\n--- Slide {i}/{len(slide_ids)}: {slide_id} ---")

        out_path = out_dir / f"{slide_id}.parquet"
        if out_path.exists() and not args.overwrite:
            print(f"Output exists, skipping: {out_path}")
            manifest["slides"][slide_id] = {
                "output_parquet": str(out_path),
                "status": "skipped_existing",
            }
            continue

        eval_patch_h5 = eval_patches_dir / f"{slide_id}.h5"
        slide_h5ad = adata_graph_dir / f"{slide_id}.h5ad"
        scc_h5_path = scconcept_dir / f"{slide_id}.h5"

        if not eval_patch_h5.exists():
            raise FileNotFoundError(f"Missing eval patch file for slide '{slide_id}': {eval_patch_h5}")
        if not slide_h5ad.exists():
            raise FileNotFoundError(f"Missing graph AnnData for slide '{slide_id}': {slide_h5ad}")
        if not scc_h5_path.exists():
            raise FileNotFoundError(
                f"Missing scConcept embeddings for slide '{slide_id}' in run '{scconcept_training_id}': "
                f"{scc_h5_path}. The scConcept run must cover this eval slide's cells."
            )

        worker_kwargs = {
            "eval_patch_h5": str(eval_patch_h5),
            "slide_h5ad": str(slide_h5ad),
            "masks_zip_path": str(masks_zip_path),
            "ckpt_path": str(ckpt_path),
            "out_path": str(out_path),
            "scc_h5_path": str(scc_h5_path),
            "requested_n_domains": [int(x) for x in requested_n_domains],
            "accelerator": str(args.accelerator),
            "num_workers": int(args.num_workers),
            "radius": float(radius),
        }

        recv_conn, send_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=_slide_worker,
            args=(slide_id, worker_kwargs, send_conn),
        )
        proc.start()
        send_conn.close()

        msg = None
        try:
            try:
                msg = recv_conn.recv()
            except EOFError:
                msg = None
        finally:
            recv_conn.close()
            proc.join()

        if proc.exitcode != 0:
            if msg is not None and not msg.get("ok", False):
                raise RuntimeError(f"Slide '{slide_id}' failed in worker:\n{msg['error']}")
            raise RuntimeError(
                f"Slide '{slide_id}' worker exited with code {proc.exitcode}. "
                "This usually indicates the child was killed by the OS (often OOM)."
            )

        if msg is None:
            raise RuntimeError(f"Slide '{slide_id}' worker exited without returning a result.")

        if not msg.get("ok", False):
            raise RuntimeError(f"Slide '{slide_id}' failed in worker:\n{msg['error']}")

        manifest["slides"][slide_id] = msg["result"]
        cleanup_memory()

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nDone.")
    print(f"Outputs written to: {out_dir}")
    print(f"Manifest written to: {manifest_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer Novae patch-domain labels for STHELAR eval H&E patches.")

    # Data & model

    parser.add_argument("--dataset_id", type=str,
                        default="ds_ct1_20x_224_o112",
                        help="Dataset ID, e.g. ds_ct1_20x_224_o112")
    
    parser.add_argument("--training_dataset_id", type=str,
                        default="ds_ct1_20x_224_o112_1",
                        help="Training dataset ID, e.g. ds_ct1_20x_224_o112_1")
    
    parser.add_argument("--training_id", type=str,
                        default="training_4",
                        help="Novae training ID, e.g. training_4")

    parser.add_argument("--ckpt", type=str,
                        default="last.ckpt",
                        help="Checkpoint filename under train/trainings/trainings_Novae/{training_id}/checkpoints/")
    
    # Parameters
    
    parser.add_argument("--n_domains", type=int,
                        nargs="+",
                        default=None,
                        help="Override the list of requested domain resolutions. Example: --n_domains 15 20 25")
    
    parser.add_argument("--accelerator", type=str,
                        default="auto",
                          help="Novae inference accelerator, e.g. 'gpu', 'cpu', or 'auto'.")
    
    parser.add_argument("--num_workers", type=int,
                        default=5,
                        help="Number of dataloader workers for Novae inference.")
    
    parser.add_argument("--radius", type=float,
                        default=None,
                        help="Delaunay edge-pruning radius in µm for the fresh per-slide graph. "
                             "Default: the Novae teacher's graph.radius_um from its manifest (else 100).")
    
    parser.add_argument("--overwrite", 
                        action="store_true",
                        help="Overwrite existing per-slide parquet outputs.")

    args = parser.parse_args()
    main(args)