#!/usr/bin/env python
"""
Precompute the teacher-centering means of the CLIP soft target: mean over the TRAIN split of the L2-normalized
Novae st_feat (per patch) and scConcept cls_emb (per cell). Runs in the `histost` env.

Saved in <ROOT_HistoST> / train / training_datasets / <dataset_id> / <training_dataset_id> / novae_teacher_cache / <novae_training_id> /
    - teacher_center_means.h5: novae_mean [128], scconcept_mean [512]
    - teacher_center_means_manifest.json
"""

import os
import sys
import json
import logging
import argparse
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import h5py
from tqdm import tqdm

# Add the project root to sys.path so that 'src' can be imported.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Torch-free imports only (helpers of src.dataset.novae_teacher are copied below)
from src.utils.constants import get_full_path, TEACHER_CENTER_MEANS_SCHEMA_VERSION  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT_SIZE = 128      # Novae st_feat dim
EMBEDDING_SIZE = 512   # scConcept cls_emb dim
_NORM_EPS = 1e-8       # L2-normalization floor
_SCC_READ_BLOCK_ROWS = 65_536  # target rows per contiguous cls_emb read (see _scc_block_rows)


# --------------------------------------------------------------------------- helpers
def _split_patch_id(patch_id):
    """``{slide_id}_{n}`` -> ``(slide_id, int(n))`` (slide_id may contain underscores)."""
    slide_id, suffix = str(patch_id).rsplit("_", 1)
    if not suffix.isdigit():
        raise ValueError(f"Unexpected patch_id (numeric suffix expected): '{patch_id}'")
    return slide_id, int(suffix)


def _list_train_patch_ids(dataset_dir, n_min_cells):
    """TRAIN patch ids from ``cell_count_train.csv`` (or ``cell_count_train_{n_min_cells}.csv``)."""
    import pandas as pd  # local import: keeps the module import pandas-optional
    if n_min_cells and int(n_min_cells) > 0:
        csv = os.path.join(str(dataset_dir), f"cell_count_train_{int(n_min_cells)}.csv")
    else:
        csv = os.path.join(str(dataset_dir), "cell_count_train.csv")
    if not os.path.isfile(csv):
        raise FileNotFoundError(f"TRAIN cell-count CSV not found: {csv}")
    return pd.read_csv(csv)["patch_id"].astype(str).tolist()


def _l2_normalize_drop_zero(X, eps=_NORM_EPS):
    """Row-L2-normalize ``X`` and drop rows with norm <= eps. Returns ``(Xn[float32], n_dropped)``."""
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1)
    keep = norms > eps
    Xn = (X[keep] / norms[keep, None]).astype(np.float32, copy=False)
    return Xn, int((~keep).sum())


def _scc_block_rows(dset):
    """Rows per contiguous ``cls_emb`` read, rounded down to a whole number of HDF5 chunks."""
    chunk_rows = int(dset.chunks[0]) if dset.chunks else _SCC_READ_BLOCK_ROWS
    return max(chunk_rows, (_SCC_READ_BLOCK_ROWS // chunk_rows) * chunk_rows)


def _read_masked_blocks(dset, mask, pbar):
    """Rows of ``dset`` selected by the boolean ``mask`` (ascending order), read as contiguous chunk-aligned blocks. Ticks ``pbar``."""
    block_rows = _scc_block_rows(dset)
    blocks = []
    for start in range(0, dset.shape[0], block_rows):
        stop = min(start + block_rows, dset.shape[0])
        blk_mask = mask[start:stop]
        n_sel = int(blk_mask.sum())
        if n_sel:
            blocks.append(dset[start:stop][blk_mask])  # native dtype; cast once by the caller
        pbar.update(n_sel)
    return blocks


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _git_commit():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "-C", project_root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- per-teacher streaming mean
def compute_novae_mean(cache_dir, dataset_dir, n_min_cells, limit):
    """Streaming TRAIN mean of the L2-normalized Novae ``st_feat`` (fp64 accumulate, drop zero-norm)."""
    train_ids = _list_train_patch_ids(dataset_dir, n_min_cells)
    train_idx_by_slide = defaultdict(list)
    for pid in train_ids:
        sid, pidx = _split_patch_id(pid)
        train_idx_by_slide[sid].append(pidx)
    train_idx_by_slide = {s: np.asarray(sorted(v), dtype=np.int64) for s, v in train_idx_by_slide.items()}

    sum64 = np.zeros(OUTPUT_SIZE, dtype=np.float64)
    n_rows = 0
    n_zero = 0
    with tqdm(total=len(train_ids), desc="[novae] st_feat", unit="patch", unit_scale=True) as pbar:
        for sid in sorted(train_idx_by_slide.keys()):
            pbar.set_postfix_str(sid, refresh=False)
            h5_path = os.path.join(str(cache_dir), f"{sid}.h5")
            if not os.path.isfile(h5_path):
                raise FileNotFoundError(
                    f"Novae st_feat cache missing for slide '{sid}': {h5_path}. Run precompute_novae_teacher.py."
                )
            with h5py.File(h5_path, "r") as f:
                if "patch_idx" not in f or "st_feat" not in f:
                    raise KeyError(f"Invalid st_feat cache (missing 'patch_idx'/'st_feat'): {h5_path}")
                patch_idx = f["patch_idx"][:].astype(np.int64, copy=False)
                st = f["st_feat"][:].astype(np.float32, copy=False)
            if st.ndim != 2 or st.shape[1] != OUTPUT_SIZE:
                raise ValueError(f"{sid}: st_feat shape {st.shape} != (*, {OUTPUT_SIZE}).")
            mask = np.isin(patch_idx, train_idx_by_slide[sid])
            Xs = st[mask]
            pbar.update(int(train_idx_by_slide[sid].size))  # by patches requested, so the bar ends at 100%
            if Xs.shape[0] == 0:
                continue
            Xn, n_drop = _l2_normalize_drop_zero(Xs)
            n_zero += n_drop
            sum64 += Xn.sum(axis=0, dtype=np.float64)
            n_rows += int(Xn.shape[0])
            if limit and n_rows >= int(limit):
                logger.info("Novae: --limit reached (%d rows), stopping early.", n_rows)
                break
    if n_rows == 0:
        raise RuntimeError("No TRAIN Novae st_feat rows loaded (check split / cache / n_min_cells).")
    mean = (sum64 / n_rows).astype(np.float32)
    logger.info("Novae mean: %d TRAIN rows (%d zero-norm dropped), ||m||=%.4f", n_rows, n_zero,
                float(np.linalg.norm(mean)))
    return mean, n_rows, n_zero


def compute_scconcept_mean(scc_dir, cell_ids_sets_path, limit):
    """Streaming TRAIN mean of the L2-normalized scConcept ``cls_emb`` (fp64 accumulate, drop zero-norm).

    TRAIN cells come from ``cell_ids_sets.npz::train_only_by_slide`` (n_min_cells-independent).
    """
    if not os.path.isfile(cell_ids_sets_path):
        raise FileNotFoundError(f"cell_ids_sets.npz not found: {cell_ids_sets_path}")
    z = np.load(str(cell_ids_sets_path), allow_pickle=True)
    if "train_only_by_slide" not in z:
        raise KeyError(f"'train_only_by_slide' missing in {cell_ids_sets_path}")
    train_by_slide = z["train_only_by_slide"].item()

    # Slides to read (with TRAIN cells and an embeddings file)
    slides = [s for s in sorted(train_by_slide.keys())
              if len(train_by_slide[s]) and os.path.isfile(os.path.join(str(scc_dir), f"{s}.h5"))]
    n_planned = int(sum(len(train_by_slide[s]) for s in slides))

    sum64 = np.zeros(EMBEDDING_SIZE, dtype=np.float64)
    n_rows = 0
    n_zero = 0
    with tqdm(total=n_planned, desc="[scConcept] cls_emb", unit="cell", unit_scale=True) as pbar:
        for sid in slides:
            pbar.set_postfix_str(sid, refresh=False)
            train_ids = np.asarray(train_by_slide[sid], dtype=np.int64)
            h5_path = os.path.join(str(scc_dir), f"{sid}.h5")
            with h5py.File(h5_path, "r") as f:
                if "cell_id_int" not in f or "cls_emb" not in f:
                    raise KeyError(f"Invalid scConcept embeddings (missing 'cell_id_int'/'cls_emb'): {h5_path}")
                cell_id_int = f["cell_id_int"][:].astype(np.int64, copy=False)
                emb_dim = int(f.attrs.get("embedding_dim", f["cls_emb"].shape[1]))
                if emb_dim != EMBEDDING_SIZE:
                    raise ValueError(f"{sid}: scConcept embedding_dim={emb_dim} != {EMBEDDING_SIZE}.")
                # boolean mask of the TRAIN cells, in ascending row order (see _read_masked_blocks).
                mask = np.isin(cell_id_int, train_ids, assume_unique=True)
                if not mask.any():
                    continue
                blocks = _read_masked_blocks(f["cls_emb"], mask, pbar)
            Xs = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)  # fp16 -> fp32 is exact
            del blocks
            Xn, n_drop = _l2_normalize_drop_zero(Xs)
            n_zero += n_drop
            sum64 += Xn.sum(axis=0, dtype=np.float64)
            n_rows += int(Xn.shape[0])
            if limit and n_rows >= int(limit):
                logger.info("scConcept: --limit reached (%d rows), stopping early.", n_rows)
                break
    if n_rows == 0:
        raise RuntimeError("No TRAIN scConcept cells loaded (check embeddings dir / cell_ids_sets.npz).")
    mean = (sum64 / n_rows).astype(np.float32)
    logger.info("scConcept mean: %d TRAIN cells (%d zero-norm dropped), ||m||=%.4f", n_rows, n_zero,
                float(np.linalg.norm(mean)))
    return mean, n_rows, n_zero


# --------------------------------------------------------------------------- write
def write_means(out_path, novae_mean, scconcept_mean, attrs, overwrite):
    """Atomic write of the two means + provenance attrs (refuse to clobber unless --overwrite)."""
    if os.path.isfile(out_path) and not overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite to recompute.")
    tmp = str(out_path) + ".tmp"
    with h5py.File(tmp, "w") as f:
        if novae_mean is not None:
            f.create_dataset("novae_mean", data=novae_mean.astype(np.float32, copy=False), dtype="float32")
        if scconcept_mean is not None:
            f.create_dataset("scconcept_mean", data=scconcept_mean.astype(np.float32, copy=False), dtype="float32")
        for k, v in attrs.items():
            f.attrs[k] = "" if v is None else v
    os.replace(tmp, out_path)


# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Precompute the teacher-centering means of the CLIP soft target."
    )
    parser.add_argument("--novae_training_id", type=str, required=True,
                        help="Novae training id whose st_feat cache defines the patch teacher (e.g. training_4).")
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset id, e.g. ds_ct1_20x_224_o112.")
    parser.add_argument("--training_dataset_id", type=str, required=True,
                        help="Training dataset id, e.g. ds_ct1_20x_224_o112_1 (defines the TRAIN split).")
    parser.add_argument("--scconcept_training_id", type=str, default=None,
                        help="scConcept training id of the cell teacher (default: the one of the Novae cache "
                             "manifest, which it must match).")
    parser.add_argument("--n_min_cells", type=int, default=0,
                        help="Min-cell filter of the Novae mean (must match data_select.n_min_cells of the "
                             "training run). Default 0.")
    parser.add_argument("--which", type=str, choices=["both", "novae", "scconcept"], default="both",
                        help="Which mean(s) to compute (default 'both').")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing teacher_center_means.h5 (default: refuse).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on the rows accumulated per teacher (for testing): writes "
                             "teacher_center_means.smoke.h5, without manifest.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    is_smoke = args.limit is not None

    # --- paths --------------------------------------------------------------------------------
    dataset_dir = get_full_path("train", "training_datasets", args.dataset_id, args.training_dataset_id,
                                as_str=True, scratch=True)
    cache_dir = get_full_path("train", "training_datasets", args.dataset_id, args.training_dataset_id,
                              "novae_teacher_cache", args.novae_training_id, as_str=True, scratch=True)
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Novae teacher cache dir not found: {cache_dir}. Run precompute_novae_teacher.py.")

    # --- resolve scConcept provenance from the Novae cache manifest ---------------------------
    manifest_path = os.path.join(cache_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Novae cache manifest not found: {manifest_path}. Run precompute_novae_teacher.py.")
    cache_manifest = _load_json(manifest_path)
    novae_fingerprint = cache_manifest.get("fingerprint")
    if novae_fingerprint is None:
        raise KeyError(f"Novae cache manifest has no 'fingerprint': {manifest_path}")
    if int(cache_manifest.get("dim", -1)) != OUTPUT_SIZE:
        raise ValueError(f"Novae cache manifest dim={cache_manifest.get('dim')} != {OUTPUT_SIZE}.")
    scc_id = args.scconcept_training_id or cache_manifest.get("scconcept_training_id")
    if scc_id is None:
        raise KeyError("scconcept_training_id not given and absent from the Novae cache manifest.")
    if args.scconcept_training_id and args.scconcept_training_id != cache_manifest.get("scconcept_training_id"):
        raise ValueError(
            f"--scconcept_training_id={args.scconcept_training_id} != Novae cache manifest's "
            f"scconcept_training_id={cache_manifest.get('scconcept_training_id')}. The cell teacher must match "
            "the scConcept model the Novae node features came from."
        )
    scc_model_name = cache_manifest.get("scconcept_model_name")
    scc_model_version = cache_manifest.get("scconcept_model_version")
    scconcept_dir = get_full_path("train", "trainings", "trainings_scConcept", scc_id, "embeddings", as_str=True)
    cell_ids_sets_path = os.path.join(dataset_dir, "cell_ids_sets.npz")

    logger.info("Teacher-centering means: novae=%s (fp=%s), scConcept=%s (%s / %s), n_min_cells=%d, which=%s",
                args.novae_training_id, novae_fingerprint, scc_id, scc_model_name, scc_model_version,
                args.n_min_cells, args.which)

    # --- compute ------------------------------------------------------------------------------
    novae_mean = scconcept_mean = None
    n_rows_novae = n_rows_scc = n_zero_novae = n_zero_scc = -1
    if args.which in ("both", "novae"):
        novae_mean, n_rows_novae, n_zero_novae = compute_novae_mean(cache_dir, dataset_dir, args.n_min_cells, args.limit)
    if args.which in ("both", "scconcept"):
        scconcept_mean, n_rows_scc, n_zero_scc = compute_scconcept_mean(scconcept_dir, cell_ids_sets_path, args.limit)

    # --- provenance attrs (checked at model build by Experiment.get_model) --------------------
    attrs = {
        "schema_version": int(TEACHER_CENTER_MEANS_SCHEMA_VERSION),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "precompute_teacher_mean.py",
        "git_commit": _git_commit(),
        "which": args.which,
        "split": "train",
        "novae_training_id": args.novae_training_id,
        "novae_teacher_fingerprint": novae_fingerprint,
        "novae_n_min_cells": int(args.n_min_cells),
        "novae_dim": int(OUTPUT_SIZE),
        "scconcept_training_id": scc_id,
        "scconcept_model_name": scc_model_name,
        "scconcept_model_version": scc_model_version,
        "scconcept_dim": int(EMBEDDING_SIZE),
        "dataset_id": args.dataset_id,
        "training_dataset_id": args.training_dataset_id,
        "n_rows_novae": int(n_rows_novae),
        "n_rows_scconcept": int(n_rows_scc),
        "n_zero_novae": int(n_zero_novae),
        "n_zero_scconcept": int(n_zero_scc),
    }

    out_name = "teacher_center_means.smoke.h5" if is_smoke else "teacher_center_means.h5"
    out_path = os.path.join(cache_dir, out_name)
    write_means(out_path, novae_mean, scconcept_mean, attrs, overwrite=(args.overwrite or is_smoke))
    logger.info("Wrote %s", out_path)

    if is_smoke:
        logger.info("--limit: manifest not written.")
    else:
        manifest_out = os.path.join(cache_dir, "teacher_center_means_manifest.json")
        with open(manifest_out, "w") as f:
            json.dump(attrs, f, indent=2)
        logger.info("Wrote manifest -> %s", manifest_out)
    logger.info("Done.")


if __name__ == "__main__":
    main()
