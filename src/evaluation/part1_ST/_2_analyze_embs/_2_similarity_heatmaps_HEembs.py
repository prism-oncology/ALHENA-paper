#!/usr/bin/env python
"""
Compare each patch embedding with the per-(patch, cell) H&E embeddings of its cells, at the (patch, cell) level
and/or projected onto the ViT token grid (--output_level), with intra-patch and shuffled baselines.

Saved in <ROOT_HistoST> / eval / eval_datasets / st_eval_data / eval_st_<training_dataset_id> / embeddings_analysis / similarity_heatmaps_HEembs / HEembs_t<training_num>e<checkpoint_num>_<space> /
    - run_config.json: analysis parameters
    Token level (--output_level token or both):
    - heatmaps/<slide_id>_similarity_heatmaps.h5: token heatmaps
    - metrics/<slide_id>_similarity_metrics.csv.gz, metrics/similarity_metrics_all_slides.csv.gz: token-focus metrics per patch
    - metrics/<slide_id>_token_slide_summary.csv, metrics/similarity_metrics_by_slide.csv: summary per slide
    Patch-cell level (--output_level patch_cell or both):
    - patch_cell/<slide_id>_patch_cell_similarity.parquet: similarity per (patch, cell)
    - metrics/<slide_id>_patch_cell_patch_summary.csv.gz, metrics/patch_cell_patch_summary_all_slides.csv.gz: summary per patch
    - metrics/<slide_id>_patch_cell_slide_summary.csv, metrics/patch_cell_similarity_by_slide.csv: summary per slide
    - metrics/similarity_metrics_by_slide_headline.csv: summary per slide over patches with >= --headline_min_cells cells
    - metrics/patch_cell_similarity_by_slide_by_npc_bin.csv: summary per (slide, n_patch_cells bin)
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from anndata import read_h5ad
from tqdm import tqdm

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.utils import set_global_seed, stable_seed

# Sibling modules in this folder.
from a0_io import (  # noqa: E402
    FALLBACK_RDCC_NBYTES,
    FALLBACK_RDCC_NSLOTS,
    atomic_output,
    matrix_fits_in_memory,
    take_rows,
)
from celltype_axis import (  # noqa: E402
    EXCLUDED_FROM_SCORED,
    HEADROOM_NORM_GAP_COLUMN,
    apply_label_map_to_dict,
    assert_vocabulary_covered,
    axis_cell_types,
    axis_column,
    build_axes,
    is_scored_flavour,
    parse_shuffled_ct_column,
    parse_shuffled_gap_column,
    scored_flavour,
    scored_mean_column,
    shuffled_ct_column,
    shuffled_gap_column,
)


EPS = 1e-12

# Output format version of this analysis (written into run_config.json, checked before writing into an existing folder)
A0_ANALYSIS_VERSION = 4

# Virchow2 token / embed dim: its patch embedding is concat(CLS, mean_patch_tokens) = 2x this (2560),
# its per-(patch, cell) embeddings are 1280-d (see reconcile_patch_dim_to_cell)
VIRCHOW2_EMBED_DIM = 1280

# The patch and cell CLIP heads are separately supervised, so their outputs are not compared
CLIP_PATCH_AND_CELL_HEADS_ARE_INDEPENDENT = True

# Bin edges (right-closed) and labels of the n_patch_cells stratification
NPC_BIN_EDGES: Tuple[int, ...] = (1, 1, 4, 9, 10**9)
NPC_BIN_LABELS: Tuple[str, ...] = ("1", "2-4", "5-9", "10+")

# Per-(patch, cell) embedding matrix, in RAM or as an open h5py Dataset (see --max_in_memory_gb)
PatchCellMatrix = Union[np.ndarray, "h5py.Dataset"]

PATCH_CELL_SCHEMA = pa.schema(
    [
        pa.field("slide_id", pa.string()),
        pa.field("patch_index", pa.int32()),
        pa.field("patch_id", pa.string()),
        pa.field("cell_id_int", pa.int32()),
        pa.field("cell_type", pa.string()),
        pa.field("alpha", pa.float32()),
        pa.field("similarity_score", pa.float32()),
    ]
)



# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------



def _decode_str_list(arr) -> List[str]:
    out = []
    for x in arr:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(bytes(x).decode("utf-8"))
        else:
            out.append(str(x))
    return out



def _safe_float(x) -> float:
    return float(x) if np.isfinite(x) else np.nan



def softmax_1d(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    z = x.astype(np.float32, copy=False)
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / (np.sum(ez) + EPS)



def token_focus_logits(
    scores: np.ndarray,
    *,
    tau_mode: str,
    tau: float,
    tau_scale: float,
) -> Tuple[Optional[np.ndarray], float, float]:
    """
    Softmax logits of the token focus map. Returns (logits_or_None, tau_realized, score_std), None if the
    score profile is degenerate. `scale_free`: (s - mean(s)) / (tau_scale * std(s)); `fixed`: s / tau.
    """
    n = int(scores.size)
    if n < 2:
        return None, float("nan"), float("nan")
    std = float(np.std(scores))
    if not np.isfinite(std):
        return None, float("nan"), float("nan")

    if tau_mode == "fixed":
        if not (tau > 0):
            raise ValueError(f"--tau_mode fixed requires --tau > 0, got {tau}")
        return scores / float(tau), float(tau), std
    if tau_mode != "scale_free":
        raise ValueError(f"Unknown tau_mode='{tau_mode}' (expected 'scale_free' or 'fixed').")

    if not (tau_scale > 0):
        raise ValueError(f"--tau_scale must be > 0, got {tau_scale}")
    if std <= EPS:
        # Every token has the same score: no focus map
        return None, float("nan"), std
    tau_realized = float(tau_scale) * std
    return (scores - float(np.mean(scores))) / tau_realized, tau_realized, std


FOCUS_SCALAR_COLUMNS: Tuple[str, ...] = (
    "peak_raw",
    "peak_p",
    "entropy",
    "entropy_norm",
    "eff_tokens",
    "eff_tokens_frac_nonempty",
    "topk_mass",
    "hhi",
    "focus_total_cells",
    "focus_total_nuc_px",
    "corr_p_total_cells",
    "corr_p_total_nuc_px",
    "spearman_score_vs_total_cells",
    "spearman_score_vs_total_nuc_px",
    "com_x",
    "com_y",
    "radius",
)

SPREAD_COLUMNS: Tuple[str, ...] = (
    "score_spread_std",
    "score_spread_iqr",
    "score_spread_p95_minus_p05",
    "score_peak_minus_median",
)


def blank_focus_metrics(
    patch_metrics: Dict[str, object],
    *,
    axis_states: Dict[str, Dict[str, object]],
    compute_celltype_advantage: bool,
    n_nonempty_tokens: int = 0,
    frac_nonempty_tokens: float = 0.0,
    keep_spread: bool = False,
) -> None:
    """
    Set every focus-family column to NaN for a patch where the focus map is not defined
    (`keep_spread=True` keeps the `score_spread_*` columns).
    """
    patch_metrics["n_nonempty_tokens"] = int(n_nonempty_tokens)
    patch_metrics["frac_nonempty_tokens"] = float(frac_nonempty_tokens)
    patch_metrics.setdefault("tau_realized", np.nan)
    patch_metrics["focus_family_degenerate"] = True
    for col in FOCUS_SCALAR_COLUMNS:
        patch_metrics[col] = np.nan
    if not keep_spread:
        for col in SPREAD_COLUMNS:
            patch_metrics[col] = np.nan
    for axis, ax_state in axis_states.items():
        for tname_safe in ax_state["safe_type_names"]:
            patch_metrics[axis_column("focus_frac", axis, tname_safe)] = np.nan
            patch_metrics[axis_column("mass_on_tokens", axis, tname_safe)] = np.nan
            patch_metrics[axis_column("corr_p_cells", axis, tname_safe)] = np.nan
            if compute_celltype_advantage:
                patch_metrics[axis_column("expected_focus_frac", axis, tname_safe)] = np.nan
                patch_metrics[axis_column("focus_advantage", axis, tname_safe)] = np.nan


def score_spread_stats(scores: np.ndarray) -> Dict[str, float]:
    """Spread of the within-patch token scores (no temperature, no softmax)."""
    if scores.size == 0:
        return {
            "score_spread_std": np.nan,
            "score_spread_iqr": np.nan,
            "score_spread_p95_minus_p05": np.nan,
            "score_peak_minus_median": np.nan,
        }
    q05, q25, q50, q75, q95 = np.percentile(scores, [5, 25, 50, 75, 95])
    return {
        "score_spread_std": float(np.std(scores)),
        "score_spread_iqr": float(q75 - q25),
        "score_spread_p95_minus_p05": float(q95 - q05),
        "score_peak_minus_median": float(np.max(scores) - q50),
    }


def spearman_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation for 1D arrays (Pearson on average ranks); NaN if ill-defined."""
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return np.nan
    ra = pd.Series(np.asarray(a, dtype=np.float64)).rank().to_numpy()
    rb = pd.Series(np.asarray(b, dtype=np.float64)).rank().to_numpy()
    return corr_1d(ra.astype(np.float32), rb.astype(np.float32))


def corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation for 1D arrays.
    Returns NaN if ill-defined.
    """
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return np.nan
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    am = a.mean()
    bm = b.mean()
    a0 = a - am
    b0 = b - bm
    av = np.mean(a0 * a0)
    bv = np.mean(b0 * b0)
    if av <= EPS or bv <= EPS:
        return np.nan
    cov = np.mean(a0 * b0)
    return float(cov / (math.sqrt(av) * math.sqrt(bv) + EPS))



def l2_norm_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 norms of `x`, shape (n,) (not the normalized rows)."""
    return np.sqrt(np.sum(x * x, axis=1) + EPS)



def sanitize_col_suffix(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name).strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "unknown"



def wants_token_output(output_level: str) -> bool:
    return output_level in {"token", "both"}



def wants_patch_cell_output(output_level: str) -> bool:
    return output_level in {"patch_cell", "both"}



# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------



def resolve_membership_parquet(emb_h5_path: Path, eval_root: Path, slide_id: str) -> Path:
    with h5py.File(emb_h5_path, "r") as f:
        attr_path = str(f.attrs.get("membership_parquet", "")).strip()

    if attr_path:
        p = Path(attr_path)
        if p.exists():
            return p

    candidates = []
    for d in sorted(eval_root.glob("cell_token_membership_*")):
        # do not take cell_token_membership_missing_cells_* into account
        if "missing_cells" not in str(d):
            cand = d / f"{slide_id}.parquet"
            if cand.exists():
                candidates.append(cand)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Could not resolve membership parquet for slide '{slide_id}'. "
            f"No valid path in embeddings attrs and no unique fallback found under {eval_root}."
        )
    raise RuntimeError(
        f"Ambiguous membership parquet for slide '{slide_id}'. "
        f"Found multiple candidates: {[str(p) for p in candidates]}"
    )



def load_celltype_map_from_h5ad(slide_h5ad_path: Path) -> Tuple[Dict[int, str], List[str]]:
    """
    Returns:
        cell_id_int_to_cell_type
        sorted unique cell type names
    """
    if not slide_h5ad_path.exists():
        raise FileNotFoundError(f"Slide h5ad not found: {slide_h5ad_path}")

    ad = read_h5ad(slide_h5ad_path, backed="r")
    try:
        obs = ad.obs
        if "is_cell" not in obs.columns:
            raise KeyError(f"{slide_h5ad_path} does not contain obs['is_cell'].")
        if "cell_id_int" not in obs.columns:
            raise KeyError(f"{slide_h5ad_path} does not contain obs['cell_id_int'].")
        if "cell_type" not in obs.columns:
            raise KeyError(f"{slide_h5ad_path} does not contain obs['cell_type'].")

        is_cell = obs["is_cell"].astype(bool).to_numpy()
        obs_cell = obs.loc[is_cell, ["cell_id_int", "cell_type"]].copy()
        obs_cell["cell_id_int"] = obs_cell["cell_id_int"].astype(np.int32)
        # A missing cell_type becomes `UNKNOWN` (as in `_1`)
        obs_cell["cell_type"] = obs_cell["cell_type"].astype("string").fillna("UNKNOWN").astype(str)

        if obs_cell["cell_id_int"].duplicated().any():
            dup = obs_cell.loc[obs_cell["cell_id_int"].duplicated(), "cell_id_int"].unique()[:10].tolist()
            raise ValueError(f"Duplicate cell_id_int values found in {slide_h5ad_path}. Examples: {dup}")

        mapping = dict(zip(obs_cell["cell_id_int"].tolist(), obs_cell["cell_type"].tolist()))
        types = sorted(obs_cell["cell_type"].dropna().astype(str).unique().tolist())
        return mapping, types
    finally:
        if hasattr(ad, "file") and ad.file is not None:
            ad.file.close()



# ---------------------------------------------------------------------
# Core numeric helpers
# ---------------------------------------------------------------------


def assert_patch_cell_space_comparable(space: str, *, slide_id: str = "") -> None:
    """Raise for `--space clip`: the patch and cell embeddings come from independent CLIP heads."""
    if space == "clip" and CLIP_PATCH_AND_CELL_HEADS_ARE_INDEPENDENT:
        where = f" (slide '{slide_id}')" if slide_id else ""
        raise RuntimeError(
            f"--space clip is not a valid patch-vs-cell comparison{where}: patch_emb_he_clip and "
            "patch_cell_emb_he_clip come from independent CLIP heads (patch and token losses). "
            "Use --space he (projected cell embeddings: _1_cell_view_consistency_HEembs.py --spaces he_clip)."
        )


def reconcile_patch_dim_to_cell(
    patch_emb: np.ndarray,
    cell_dim: int,
    *,
    slide_id: str,
    space: str,
) -> np.ndarray:
    """
    Make the per-patch embedding comparable to the per-(patch, cell) embedding: unchanged if the widths
    match, CLS half for the Virchow2 layout (patch_dim == 2 * cell_dim); raises otherwise.
    """
    patch_dim = int(patch_emb.shape[1])
    if patch_dim == cell_dim:
        return patch_emb
    if patch_dim == 2 * cell_dim and cell_dim == VIRCHOW2_EMBED_DIM:
        return patch_emb[:, :cell_dim]
    raise RuntimeError(
        f"Cannot compare patch vs cell embeddings for slide '{slide_id}' (space='{space}'): "
        f"patch_dim={patch_dim}, cell_dim={cell_dim}. Expected patch_dim == cell_dim, or the "
        f"Virchow2 layout patch_dim == 2*cell_dim with cell_dim == {VIRCHOW2_EMBED_DIM}."
    )


def compute_group_similarity_scores(
    patch_emb: np.ndarray,
    group_patch_index: np.ndarray,
    group_emb: "PatchCellMatrix",
    similarity: str,
    chunk_size: int = 200_000,
) -> np.ndarray:
    """
    Compute sim(patch_emb[patch_index], patch_cell_emb[group]) for all groups.
    `group_emb` is an in-RAM ndarray or an open h5py Dataset (read in contiguous slices).
    """
    n_groups = int(group_patch_index.shape[0])
    out = np.empty((n_groups,), dtype=np.float32)

    for start in tqdm(range(0, n_groups, chunk_size), desc="  patch-vs-cell sim", leave=False):
        end = min(n_groups, start + chunk_size)
        patch_vec = patch_emb[group_patch_index[start:end]].astype(np.float32, copy=False)
        cell_vec = np.asarray(group_emb[start:end], dtype=np.float32)

        if cell_vec.ndim != 2:
            raise RuntimeError(f"Expected 2D group embeddings, got shape={cell_vec.shape}")

        if patch_vec.shape != cell_vec.shape:
            raise RuntimeError(
                f"Shape mismatch when computing similarity: patch chunk {patch_vec.shape} vs cell chunk {cell_vec.shape}"
            )

        if similarity == "cosine":
            num = np.sum(patch_vec * cell_vec, axis=1)
            den = l2_norm_rows(patch_vec) * l2_norm_rows(cell_vec) + EPS
            out[start:end] = (num / den).astype(np.float32, copy=False)
        elif similarity == "dot":
            out[start:end] = np.sum(patch_vec * cell_vec, axis=1).astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unknown similarity='{similarity}'")

    return out



def build_key_u64(patch_index: np.ndarray, cell_id_int: np.ndarray) -> np.ndarray:
    return (patch_index.astype(np.uint64) << np.uint64(32)) | cell_id_int.astype(np.uint32).astype(np.uint64)



def map_membership_rows_to_group_rows(
    mem_patch_index: np.ndarray,
    mem_cell_id_int: np.ndarray,
    group_patch_index: np.ndarray,
    group_cell_id_int: np.ndarray,
) -> np.ndarray:
    """
    Returns:
        mem_to_group: (N_mem,) int64 index into group arrays for each membership row.
    """
    key_group = build_key_u64(group_patch_index, group_cell_id_int)
    order = np.argsort(key_group, kind="mergesort")
    key_group_sorted = key_group[order]

    key_mem = build_key_u64(mem_patch_index, mem_cell_id_int)
    pos = np.searchsorted(key_group_sorted, key_mem, side="left")
    ok = pos < key_group_sorted.shape[0]
    ok_exact = np.zeros_like(ok, dtype=bool)
    if np.any(ok):
        ok_exact[ok] = key_group_sorted[pos[ok]] == key_mem[ok]
    ok = ok_exact

    if not np.all(ok):
        bad = np.flatnonzero(~ok)
        examples = []
        for i in bad[:10]:
            examples.append(
                {
                    "patch_index": int(mem_patch_index[i]),
                    "cell_id_int": int(mem_cell_id_int[i]),
                }
            )
        raise RuntimeError(
            "Some membership rows could not be matched to patch-cell embeddings. "
            f"First unmatched examples: {examples}"
        )

    return order[pos].astype(np.int64, copy=False)



def compute_patch_level_group_summary(
    n_patches: int,
    group_patch_index: np.ndarray,
    group_alpha: np.ndarray,
    group_score: np.ndarray,
) -> pd.DataFrame:
    """
    Per-patch summary over patch-cell groups.
    """
    n_cells_patch = np.bincount(group_patch_index, minlength=n_patches).astype(np.int32)

    sum_score = np.bincount(group_patch_index, weights=group_score, minlength=n_patches).astype(np.float32)
    mean_score = np.full((n_patches,), np.nan, dtype=np.float32)
    valid = n_cells_patch > 0
    mean_score[valid] = sum_score[valid] / n_cells_patch[valid]

    sum_alpha = np.bincount(group_patch_index, weights=group_alpha, minlength=n_patches).astype(np.float32)
    sum_alpha_score = np.bincount(group_patch_index, weights=group_alpha * group_score, minlength=n_patches).astype(np.float32)
    alpha_mean = np.full((n_patches,), np.nan, dtype=np.float32)
    valid_alpha = sum_alpha > 0
    alpha_mean[valid_alpha] = sum_alpha_score[valid_alpha] / (sum_alpha[valid_alpha] + EPS)

    max_score = np.full((n_patches,), np.nan, dtype=np.float32)
    if group_score.shape[0] > 0:
        tmp = np.full((n_patches,), -np.inf, dtype=np.float32)
        np.maximum.at(tmp, group_patch_index, group_score)
        tmp[np.isneginf(tmp)] = np.nan
        max_score = tmp

    out = pd.DataFrame(
        {
            "patch_index": np.arange(n_patches, dtype=np.int32),
            "n_patch_cells": n_cells_patch,
            "cell_sim_mean": mean_score,
            "cell_sim_alpha_mean": alpha_mean,
            "cell_sim_max": max_score,
        }
    )
    return out



def compute_intra_patch_cell_cell(
    n_patches: int,
    group_patch_index: np.ndarray,
    group_emb: "PatchCellMatrix",
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    """
    For each patch with at least two cells, pairwise cosines between the patch-cell embeddings of its cells.
    Returns one row per patch: patch_index, intra_patch_cc_n_pairs, intra_patch_cc_cos_{mean,min,std}.
    """
    n_groups = int(group_patch_index.shape[0])
    if n_groups == 0:
        return pd.DataFrame({
            "patch_index": np.arange(n_patches, dtype=np.int32),
            "intra_patch_cc_n_pairs": np.zeros(n_patches, dtype=np.int32),
            "intra_patch_cc_cos_mean": np.full(n_patches, np.nan, dtype=np.float32),
            "intra_patch_cc_cos_min": np.full(n_patches, np.nan, dtype=np.float32),
            "intra_patch_cc_cos_std": np.full(n_patches, np.nan, dtype=np.float32),
        })

    # Group rows by patch
    order = np.argsort(group_patch_index, kind="mergesort")

    out_mean = np.full(n_patches, np.nan, dtype=np.float32)
    out_min = np.full(n_patches, np.nan, dtype=np.float32)
    out_std = np.full(n_patches, np.nan, dtype=np.float32)
    out_n = np.zeros(n_patches, dtype=np.int32)

    sorted_patch = group_patch_index[order]
    # Identify contiguous patch groups in sorted order.
    change = np.r_[True, sorted_patch[1:] != sorted_patch[:-1], True]
    bounds = np.flatnonzero(change)
    starts = bounds[:-1]
    ends = bounds[1:]

    for s, e in tqdm(zip(starts, ends), total=len(starts), desc="  intra-patch cell-cell", leave=False):
        npc = int(e - s)
        if npc < 2:
            continue
        pidx = int(sorted_patch[s])
        # Read the cells of this patch (sorted-back indices).
        sel = np.sort(order[s:e])
        # Contiguous slice when possible, else take_rows
        if sel[-1] - sel[0] + 1 == sel.size:
            block = np.asarray(group_emb[int(sel[0]):int(sel[-1]) + 1], dtype=np.float32)
        else:
            block = take_rows(group_emb, sel)
        # Cosine matrix on L2-normalized rows.
        norms = l2_norm_rows(block)
        Ehat = block / np.clip(norms[:, None], EPS, None)
        sim = np.clip(Ehat @ Ehat.T, -1.0, 1.0)
        iu = np.triu_indices(npc, k=1)
        vals = sim[iu]
        out_n[pidx] = int(vals.size)
        if vals.size:
            out_mean[pidx] = float(np.mean(vals))
            out_min[pidx] = float(np.min(vals))
            out_std[pidx] = float(np.std(vals))

    return pd.DataFrame({
        "patch_index": np.arange(n_patches, dtype=np.int32),
        "intra_patch_cc_n_pairs": out_n,
        "intra_patch_cc_cos_mean": out_mean,
        "intra_patch_cc_cos_min": out_min,
        "intra_patch_cc_cos_std": out_std,
    })



def _sample_cross_patch_rows_vectorized(
    pool: np.ndarray,
    self_rows: np.ndarray,
    self_patch: np.ndarray,
    group_patch_index: np.ndarray,
    rng: np.random.Generator,
    max_reject_tries: int = 8,
) -> np.ndarray:
    """
    For each query i, draw one row uniformly from `pool` that is neither `self_rows[i]` nor in patch
    `self_patch[i]` (vectorized bounded rejection, then an exact fallback). Returns int64 indices, -1 if none.
    """
    n = int(self_rows.shape[0])
    out = np.full(n, -1, dtype=np.int64)
    if n == 0 or pool.size == 0:
        return out

    # Fast path: vectorized bounded rejection
    unresolved = np.ones(n, dtype=bool)
    for _ in range(max_reject_tries):
        loc = np.flatnonzero(unresolved)
        if loc.size == 0:
            break
        cand = pool[rng.integers(0, pool.size, size=loc.size)]
        ok = (cand != self_rows[loc]) & (group_patch_index[cand] != self_patch[loc])
        good = loc[ok]
        out[good] = cand[ok]
        unresolved[good] = False

    # Exact fallback (rare): draw uniformly from the per-query filtered pool.
    straggler = np.flatnonzero(unresolved)
    if straggler.size:
        pool_patch = group_patch_index[pool]
        for i in straggler:
            valid = pool[(pool != self_rows[i]) & (pool_patch != self_patch[i])]
            if valid.size:
                out[i] = int(valid[int(rng.integers(0, valid.size))])
    return out



def compute_shuffled_baseline(
    patch_emb: np.ndarray,
    group_patch_index: np.ndarray,
    group_cell_id_int: np.ndarray,
    group_type_idx_by_axis: Dict[str, np.ndarray],
    group_emb: "PatchCellMatrix",
    n_shuffled_per_group: int,
    rng_for_stream: Callable[[int], np.random.Generator],
    primary_axis: Optional[str] = None,
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """
    For each (patch p, cell c) row, mean cosine between patch_emb[p] and random other rows of the slide: any row
    (`any`), or a row of another patch with the same (`sameCT`) or a different (`diffCT`) cell type, per cell-type
    axis (unsuffixed columns = `primary_axis`). NaN where no row qualifies.
    """
    if not group_type_idx_by_axis:
        raise ValueError("group_type_idx_by_axis must contain at least one cell-type axis.")
    axes = list(group_type_idx_by_axis)
    primary = primary_axis if primary_axis is not None else axes[0]
    if primary not in group_type_idx_by_axis:
        raise ValueError(f"primary_axis='{primary}' is not among the axes {axes}.")

    n_groups = int(group_patch_index.shape[0])
    cos_any = np.full(n_groups, np.nan, dtype=np.float32)
    cos_sameCT = {ax: np.full(n_groups, np.nan, dtype=np.float32) for ax in axes}
    cos_diffCT = {ax: np.full(n_groups, np.nan, dtype=np.float32) for ax in axes}

    def _assemble() -> pd.DataFrame:
        out = {
            "patch_index": group_patch_index.astype(np.int32, copy=False),
            "cell_id_int": group_cell_id_int.astype(np.int32, copy=False),
            "shuffled_cos_any_mean": cos_any,
        }
        for ax in axes:
            out[shuffled_ct_column("sameCT", ax)] = cos_sameCT[ax]
            out[shuffled_ct_column("diffCT", ax)] = cos_diffCT[ax]
        # Unsuffixed columns: primary axis
        out["shuffled_cos_sameCT_mean"] = cos_sameCT[primary]
        out["shuffled_cos_diffCT_mean"] = cos_diffCT[primary]
        return pd.DataFrame(out)

    if n_groups < 2:
        return _assemble()

    # Per-axis pools: rows of each type, and its complement.
    pools_same: Dict[str, Dict[int, np.ndarray]] = {}
    pools_diff: Dict[str, Dict[int, np.ndarray]] = {}
    unique_types_by_axis: Dict[str, List[int]] = {}
    for ax in axes:
        type_idx = group_type_idx_by_axis[ax]
        unique_types = [int(t) for t in np.unique(type_idx) if t >= 0]
        unique_types_by_axis[ax] = unique_types
        pools_same[ax] = {t: np.flatnonzero(type_idx == t) for t in unique_types}
        pools_diff[ax] = {t: np.flatnonzero(type_idx != t) for t in unique_types}

    # Cache norms of patch embeddings (one per patch).
    patch_norms = l2_norm_rows(patch_emb)

    # One generator per flavour family: 0 = `any`, 1 + i = the i-th axis
    rng_any = rng_for_stream(0)
    rng_by_axis = {ax: rng_for_stream(1 + i) for i, ax in enumerate(axes)}

    def _accumulate(arr_j: np.ndarray, patch_vec: np.ndarray, patch_norm: np.ndarray,
                    sums: np.ndarray, n_acc: np.ndarray) -> None:
        valid = arr_j >= 0
        if not np.any(valid):
            return
        uniq, inv = np.unique(arr_j[valid], return_inverse=True)
        block = take_rows(group_emb, uniq)
        foreign = block[inv]
        foreign_norm = l2_norm_rows(foreign) + EPS
        num = np.sum(patch_vec[valid] * foreign, axis=1)
        cos_vals = num / (patch_norm[valid] * foreign_norm)
        sums[valid] += np.clip(cos_vals, -1.0, 1.0).astype(np.float64)
        n_acc[valid] += 1

    # Process in chunks to bound HDF5 random reads.
    for start in tqdm(range(0, n_groups, chunk_size), desc="  shuffled baseline", leave=False):
        end = min(n_groups, start + chunk_size)
        # Patch vectors for this chunk (same for all flavours and all axes).
        chunk_p = group_patch_index[start:end].astype(np.int64, copy=False)
        patch_vec = patch_emb[chunk_p].astype(np.float32, copy=False)
        patch_norm = patch_norms[chunk_p].astype(np.float32, copy=False) + EPS

        # 'any' flavour: sample random rows excluding self. Cell-type-free, computed once.
        sums_any = np.zeros(end - start, dtype=np.float64)
        n_any = np.zeros(end - start, dtype=np.int32)
        for _ in range(int(n_shuffled_per_group)):
            j = rng_any.integers(0, n_groups, size=end - start)
            valid = j != np.arange(start, end)
            if not np.any(valid):
                continue
            # Read the foreign embeddings (sorted unique indices)
            uniq, inv = np.unique(j[valid], return_inverse=True)
            block = take_rows(group_emb, uniq)
            foreign = block[inv]
            foreign_norm = l2_norm_rows(foreign) + EPS
            num = np.sum(patch_vec[valid] * foreign, axis=1)
            cos_vals = num / (patch_norm[valid] * foreign_norm)
            sums_any[valid] += np.clip(cos_vals, -1.0, 1.0).astype(np.float64)
            n_any[valid] += 1
        with np.errstate(invalid="ignore"):
            cos_any[start:end] = np.where(n_any > 0, sums_any / np.maximum(n_any, 1), np.nan)

        # sameCT / diffCT flavours: per cell-type axis.
        # Cross-patch draws per cell type (rows of unknown type, t == -1, stay at -1)
        chunk_self_rows = np.arange(start, end, dtype=np.int64)
        for ax in axes:
            rng_ax = rng_by_axis[ax]
            chunk_t = group_type_idx_by_axis[ax][start:end].astype(np.int64, copy=False)
            sums_s = np.zeros(end - start, dtype=np.float64)
            n_s = np.zeros(end - start, dtype=np.int32)
            sums_d = np.zeros(end - start, dtype=np.float64)
            n_d = np.zeros(end - start, dtype=np.int32)
            for _ in range(int(n_shuffled_per_group)):
                j_same = np.full(end - start, -1, dtype=np.int64)
                j_diff = np.full(end - start, -1, dtype=np.int64)
                for t in unique_types_by_axis[ax]:
                    local = np.flatnonzero(chunk_t == t)
                    if local.size == 0:
                        continue
                    self_rows_t = chunk_self_rows[local]
                    self_patch_t = chunk_p[local]
                    # sameCT: cross-patch rows of the same cell type.
                    pool_same = pools_same[ax].get(t)
                    if pool_same is not None and pool_same.size > 1:
                        j_same[local] = _sample_cross_patch_rows_vectorized(
                            pool_same, self_rows_t, self_patch_t, group_patch_index, rng_ax
                        )
                    # diffCT: cross-patch rows of a different cell type.
                    pool_diff = pools_diff[ax].get(t)
                    if pool_diff is not None and pool_diff.size > 0:
                        j_diff[local] = _sample_cross_patch_rows_vectorized(
                            pool_diff, self_rows_t, self_patch_t, group_patch_index, rng_ax
                        )

                _accumulate(j_same, patch_vec, patch_norm, sums_s, n_s)
                _accumulate(j_diff, patch_vec, patch_norm, sums_d, n_d)

            with np.errstate(invalid="ignore"):
                cos_sameCT[ax][start:end] = np.where(n_s > 0, sums_s / np.maximum(n_s, 1), np.nan)
                cos_diffCT[ax][start:end] = np.where(n_d > 0, sums_d / np.maximum(n_d, 1), np.nan)

    return _assemble()



def bin_npc(n_patch_cells: np.ndarray) -> np.ndarray:
    """Bin `n_patch_cells` into NPC_BIN_LABELS ('1', '2-4', '5-9', '10+'); empty string for NaN or values below 1."""
    x = np.asarray(n_patch_cells, dtype=np.float64)
    out = np.full(x.shape, "", dtype=object)
    one = x == 1
    two_four = (x >= 2) & (x <= 4)
    five_nine = (x >= 5) & (x <= 9)
    ten_plus = x >= 10
    out[one] = NPC_BIN_LABELS[0]
    out[two_four] = NPC_BIN_LABELS[1]
    out[five_nine] = NPC_BIN_LABELS[2]
    out[ten_plus] = NPC_BIN_LABELS[3]
    return out



def compute_patch_cell_slide_summary(
    slide_id: str,
    n_patches: int,
    group_cell_id_int: np.ndarray,
    group_alpha: np.ndarray,
    group_score: np.ndarray,
    unique_cell_types: List[str],
    space: str,
    similarity: str,
    embeddings_h5: Path,
    slide_h5ad: Path,
    patch_cell_out_parquet: Path,
    df_group_summary: Optional[pd.DataFrame] = None,
    shuffled_per_group: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    n_groups = int(group_score.shape[0])
    if n_groups > 0:
        score_mean = float(np.mean(group_score))
        score_std = float(np.std(group_score))
        score_min = float(np.min(group_score))
        score_max = float(np.max(group_score))
    else:
        score_mean = np.nan
        score_std = np.nan
        score_min = np.nan
        score_max = np.nan

    alpha_sum = float(np.sum(group_alpha))
    alpha_mean = float(np.mean(group_alpha)) if group_alpha.shape[0] > 0 else np.nan
    score_alpha_mean = float(np.sum(group_alpha * group_score) / (alpha_sum + EPS)) if alpha_sum > 0 else np.nan

    out = {
        "schema_version": int(PER_SLIDE_SCHEMA_VERSION),
        "slide_id": slide_id,
        "n_patches": int(n_patches),
        "n_patch_cells_total": n_groups,
        "n_unique_cells": int(np.unique(group_cell_id_int).shape[0]) if group_cell_id_int.shape[0] > 0 else 0,
        "n_cell_types": int(len(unique_cell_types)),
        "group_score_mean": score_mean,
        "group_score_std": score_std,
        "group_score_min": score_min,
        "group_score_max": score_max,
        "group_score_alpha_mean": score_alpha_mean,
        "group_alpha_mean": alpha_mean,
        "group_alpha_sum": alpha_sum,
        "embedding_space": space,
        "similarity": similarity,
        "patch_cell_parquet": str(patch_cell_out_parquet),
        "embeddings_h5": str(embeddings_h5),
        "slide_h5ad": str(slide_h5ad),
    }

    def _slide_mean(frame: pd.DataFrame, col: str) -> float:
        arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else np.nan

    # Slide-level means of the per-patch diagnostics (shuffled and gap columns found from their names)
    if df_group_summary is not None and not df_group_summary.empty:
        fixed_cols = (
            "intra_patch_cc_cos_mean",
            "intra_patch_cc_cos_min",
            "within_patch_cell_separation",
            # Not matched by parse_shuffled_gap_column, so listed explicitly
            HEADROOM_NORM_GAP_COLUMN,
        )
        # Per-axis scored means (not matched by the column parsers either)
        fixed_cols = fixed_cols + tuple(
            sorted(
                {
                    scored_mean_column(axis)
                    for col in df_group_summary.columns
                    for flavour, axis in [parse_shuffled_ct_column(col) or ("", None)]
                    if axis and is_scored_flavour(flavour)
                }
            )
        )
        discovered_cols = [
            c for c in df_group_summary.columns
            if parse_shuffled_ct_column(c) is not None
            or parse_shuffled_gap_column(c) is not None
        ]
        for col in list(fixed_cols) + sorted(discovered_cols):
            out[col + "_slide_mean"] = (
                _slide_mean(df_group_summary, col)
                if col in df_group_summary.columns else np.nan
            )

    # Slide-level means of the per-(patch, cell) shuffled-baseline columns
    if shuffled_per_group is not None and not shuffled_per_group.empty:
        pergroup_cols = [
            c for c in shuffled_per_group.columns
            if parse_shuffled_ct_column(c) is not None
        ]
        for col in sorted(pergroup_cols):
            out[col + "_pergroup_slide_mean"] = _slide_mean(shuffled_per_group, col)

    return out



# ---------------------------------------------------------------------
# Patch-cell outputs
# ---------------------------------------------------------------------


def write_patch_cell_similarity_parquet(
    out_path: Path,
    slide_id: str,
    patch_ids: List[str],
    group_patch_index: np.ndarray,
    group_cell_id_int: np.ndarray,
    group_alpha: np.ndarray,
    group_score: np.ndarray,
    cell_id_to_type: Dict[int, str],
) -> None:
    """
    Write one row per stored `(patch_id, cell_id_int)` group.

    Columns:
      - slide_id
      - patch_index
      - patch_id
      - cell_id_int
      - cell_type
      - alpha            (sum nuc_px within the patch for that cell)
      - similarity_score
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = int(group_patch_index.shape[0])
    if n == 0:
        table = pa.Table.from_arrays(
            [
                pa.array([], type=pa.string()),
                pa.array([], type=pa.int32()),
                pa.array([], type=pa.string()),
                pa.array([], type=pa.int32()),
                pa.array([], type=pa.string()),
                pa.array([], type=pa.float32()),
                pa.array([], type=pa.float32()),
            ],
            schema=PATCH_CELL_SCHEMA,
        )
        with atomic_output(out_path) as tmp:
            pq.write_table(table, str(tmp), compression="zstd", use_dictionary=True)
        return

    order = np.lexsort((group_cell_id_int, group_patch_index))
    patch_index_sorted = group_patch_index[order].astype(np.int32, copy=False)
    cell_id_sorted = group_cell_id_int[order].astype(np.int32, copy=False)
    alpha_sorted = group_alpha[order].astype(np.float32, copy=False)
    score_sorted = group_score[order].astype(np.float32, copy=False)

    patch_ids_arr = np.asarray(patch_ids, dtype=object)
    patch_id_sorted = patch_ids_arr[patch_index_sorted]
    cell_type_sorted = np.fromiter(
        (cell_id_to_type.get(int(cid), "") for cid in cell_id_sorted),
        dtype=object,
        count=n,
    )
    slide_id_arr = np.full((n,), str(slide_id), dtype=object)

    table = pa.Table.from_pydict(
        {
            "slide_id": slide_id_arr,
            "patch_index": patch_index_sorted,
            "patch_id": patch_id_sorted,
            "cell_id_int": cell_id_sorted,
            "cell_type": cell_type_sorted,
            "alpha": alpha_sorted,
            "similarity_score": score_sorted,
        },
        schema=PATCH_CELL_SCHEMA,
    )
    with atomic_output(out_path) as tmp:
        pq.write_table(table, str(tmp), compression="zstd", use_dictionary=True)



# ---------------------------------------------------------------------
# Token-level outputs
# ---------------------------------------------------------------------


def build_token_heatmaps_and_metrics(
    slide_id: str,
    patch_ids: List[str],
    group_patch_index: np.ndarray,
    group_cell_id_int: np.ndarray,
    group_alpha: np.ndarray,
    group_score: np.ndarray,
    group_type_idx_by_axis: Dict[str, np.ndarray],
    df_group_summary: pd.DataFrame,
    patch_id_to_index: Dict[str, int],
    membership_parquet_path: Path,
    heatmap_out_h5: Path,
    metrics_out_csv_gz: Path,
    space: str,
    similarity: str,
    tau: float,
    tau_mode: str,
    tau_scale: float,
    topk: int,
    heatmap_dtype: str,
    axis_states: Dict[str, Dict[str, object]],
    emb_h5_path: Path,
    slide_h5ad_path: Path,
    gh: int,
    gw: int,
    compute_celltype_advantage: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    
    n_patches = len(patch_ids)
    T = int(gh * gw)

    print("  3. Loading membership parquet and projecting cell scores back to token space...")
    table = pq.read_table(
        str(membership_parquet_path),
        columns=["patch_id", "cell_id_int", "token_index", "nuc_px"],
    )
    mem_patch_id = table["patch_id"].to_pylist()
    mem_cell_id_int = table["cell_id_int"].to_numpy().astype(np.int32, copy=False)
    mem_token_index = table["token_index"].to_numpy().astype(np.int32, copy=False)
    mem_nuc_px = table["nuc_px"].to_numpy().astype(np.float32, copy=False)

    mem_patch_index = np.fromiter(
        (patch_id_to_index.get(str(pid), -1) for pid in mem_patch_id),
        dtype=np.int32,
        count=len(mem_patch_id),
    )

    valid = (
        (mem_patch_index >= 0)
        & (mem_token_index >= 0)
        & (mem_token_index < T)
        & (mem_nuc_px > 0)
    )

    if not np.all(valid):
        n_bad = int((~valid).sum())
        print(f"  [warn] Dropping {n_bad} invalid membership row(s) for slide '{slide_id}'.")
        mem_patch_index = mem_patch_index[valid]
        mem_cell_id_int = mem_cell_id_int[valid]
        mem_token_index = mem_token_index[valid]
        mem_nuc_px = mem_nuc_px[valid]

    mem_to_group = map_membership_rows_to_group_rows(
        mem_patch_index=mem_patch_index,
        mem_cell_id_int=mem_cell_id_int,
        group_patch_index=group_patch_index,
        group_cell_id_int=group_cell_id_int,
    )

    mem_group_score = group_score[mem_to_group]
    mem_group_type_idx_by_axis = {ax: arr[mem_to_group] for ax, arr in group_type_idx_by_axis.items()}

    # Sort membership rows by patch for efficient per-patch iteration
    order = np.argsort(mem_patch_index, kind="mergesort")
    mem_patch_index = mem_patch_index[order]
    mem_token_index = mem_token_index[order]
    mem_nuc_px = mem_nuc_px[order]
    mem_group_score = mem_group_score[order]
    mem_group_type_idx_by_axis = {ax: arr[order] for ax, arr in mem_group_type_idx_by_axis.items()}

    # Patch boundaries in sorted membership arrays
    patch_ids_unique, patch_starts, patch_counts = np.unique(
        mem_patch_index, return_index=True, return_counts=True
    )

    patch_start_map = np.full((n_patches,), -1, dtype=np.int64)
    patch_count_map = np.zeros((n_patches,), dtype=np.int64)
    patch_start_map[patch_ids_unique] = patch_starts.astype(np.int64, copy=False)
    patch_count_map[patch_ids_unique] = patch_counts.astype(np.int64, copy=False)

    print("  4. Building token-space heatmaps and per-patch metrics...")

    heatmap_out_h5.parent.mkdir(parents=True, exist_ok=True)
    metrics_out_csv_gz.parent.mkdir(parents=True, exist_ok=True)

    if heatmap_dtype == "fp16":
        hm_dtype = np.float16
    elif heatmap_dtype == "fp32":
        hm_dtype = np.float32
    else:
        raise ValueError(f"Unknown heatmap_dtype='{heatmap_dtype}'")

    rows = []

    # Heatmap H5 written through atomic_output (temp file moved into place)
    with atomic_output(heatmap_out_h5) as heatmap_tmp:
        with h5py.File(heatmap_tmp, "w") as hf:
            str_dt = h5py.string_dtype(encoding="utf-8")
            hf.create_dataset("patch_id", data=np.asarray(patch_ids, dtype=object), dtype=str_dt)
            hf.create_dataset("score_mean", shape=(n_patches, gh, gw), dtype=hm_dtype, compression="gzip", compression_opts=4)
            hf.create_dataset("score_max", shape=(n_patches, gh, gw), dtype=hm_dtype, compression="gzip", compression_opts=4)
            hf.create_dataset("prob_nonempty", shape=(n_patches, gh, gw), dtype=hm_dtype, compression="gzip", compression_opts=4)
            hf.create_dataset("total_nuc_px", shape=(n_patches, gh, gw), dtype=np.float32, compression="gzip", compression_opts=4)
            hf.create_dataset("total_cells", shape=(n_patches, gh, gw), dtype=np.float32, compression="gzip", compression_opts=4)

            hf.attrs["slide_id"] = slide_id
            hf.attrs["token_grid_h"] = gh
            hf.attrs["token_grid_w"] = gw
            hf.attrs["embedding_space"] = space
            hf.attrs["similarity"] = similarity
            hf.attrs["tau"] = float(tau)
            hf.attrs["softmax_support"] = "nonempty_tokens_only"
            hf.attrs["membership_parquet"] = str(membership_parquet_path)
            hf.attrs["slide_h5ad"] = str(slide_h5ad_path)
            hf.attrs["embeddings_h5"] = str(emb_h5_path)
            hf.attrs["score_definition"] = (
                "For each token, score_mean is the nuc_px-weighted mean of patch-vs-(patch,cell) similarity "
                "over all cells overlapping the token."
            )
            hf.attrs["cell_type_names"] = json.dumps(
                {ax: list(st["unique_cell_types"]) for ax, st in axis_states.items()}
            )

            group_summary_lookup = df_group_summary.set_index("patch_index")
            token_x = np.tile(np.arange(gw, dtype=np.float32), gh)
            token_y = np.repeat(np.arange(gh, dtype=np.float32), gw)

            for patch_index in tqdm(range(n_patches), desc=f"  Patches {slide_id}"):
                start = int(patch_start_map[patch_index])
                count = int(patch_count_map[patch_index])

                score_mean = np.full((T,), np.nan, dtype=np.float32)
                score_max = np.full((T,), np.nan, dtype=np.float32)
                prob = np.zeros((T,), dtype=np.float32)
                total_nuc_px = np.zeros((T,), dtype=np.float32)
                total_cells = np.zeros((T,), dtype=np.float32)

                patch_metrics = {
                    "schema_version": int(PER_SLIDE_SCHEMA_VERSION),
                    "slide_id": slide_id,
                    "patch_id": patch_ids[patch_index],
                    "patch_index": int(patch_index),
                    "token_grid_h": int(gh),
                    "token_grid_w": int(gw),
                    "n_tokens_total": int(T),
                    "embedding_space": space,
                    "similarity": similarity,
                    "tau": float(tau),
                }

                # group-level summary
                gs = group_summary_lookup.loc[patch_index]
                patch_metrics["n_patch_cells"] = int(gs["n_patch_cells"])
                patch_metrics["cell_sim_mean"] = _safe_float(gs["cell_sim_mean"])
                patch_metrics["cell_sim_alpha_mean"] = _safe_float(gs["cell_sim_alpha_mean"])
                patch_metrics["cell_sim_max"] = _safe_float(gs["cell_sim_max"])

                if count > 0:
                    sl = slice(start, start + count)
                    tok_idx = mem_token_index[sl]
                    nuc_px = mem_nuc_px[sl]
                    cell_score = mem_group_score[sl]

                    # score_mean by nuc_px-weighted average
                    total_nuc_px = np.bincount(tok_idx, weights=nuc_px, minlength=T).astype(np.float32)
                    score_sum = np.bincount(tok_idx, weights=nuc_px * cell_score, minlength=T).astype(np.float32)
                    nonempty = total_nuc_px > 0
                    score_mean[nonempty] = score_sum[nonempty] / (total_nuc_px[nonempty] + EPS)

                    # score_max
                    tmp = np.full((T,), -np.inf, dtype=np.float32)
                    np.maximum.at(tmp, tok_idx, cell_score)
                    tmp[np.isneginf(tmp)] = np.nan
                    score_max = tmp

                    # cell counts per token (# distinct cell-token memberships -> one row per cell/token in parquet)
                    total_cells = np.bincount(tok_idx, minlength=T).astype(np.float32)

                    n_nonempty = int(nonempty.sum())
                    patch_metrics["n_nonempty_tokens"] = n_nonempty
                    patch_metrics["frac_nonempty_tokens"] = float(n_nonempty / T)

                    scores_nonempty = score_mean[nonempty]
                    patch_metrics.update(score_spread_stats(scores_nonempty))
                    logits, tau_realized, score_std = token_focus_logits(
                        scores_nonempty, tau_mode=tau_mode, tau=tau, tau_scale=tau_scale
                    )
                    patch_metrics["tau_realized"] = tau_realized
                    patch_metrics["focus_family_degenerate"] = bool(logits is None)

                    if n_nonempty > 0 and logits is not None:
                        p_nonempty = softmax_1d(logits)
                        prob[nonempty] = p_nonempty

                        topk_eff = min(int(topk), n_nonempty)
                        if topk_eff > 0:
                            top_vals = np.partition(p_nonempty, -topk_eff)[-topk_eff:]
                            topk_mass = float(top_vals.sum())
                        else:
                            topk_mass = np.nan

                        ent = float(-(p_nonempty * np.log(p_nonempty + EPS)).sum())
                        eff_tokens = float(np.exp(ent))
                        peak_raw = float(np.nanmax(score_mean[nonempty]))
                        peak_p = float(np.max(p_nonempty))
                        hhi = float(np.sum(p_nonempty * p_nonempty))

                        com_x = float(np.sum(prob * token_x))
                        com_y = float(np.sum(prob * token_y))
                        radius = float(
                            np.sqrt(np.sum(prob * ((token_x - com_x) ** 2 + (token_y - com_y) ** 2)) + EPS)
                        )

                        patch_metrics["peak_raw"] = peak_raw
                        patch_metrics["peak_p"] = peak_p
                        patch_metrics["entropy"] = ent
                        patch_metrics["entropy_norm"] = float(ent / math.log(max(n_nonempty, 2)))
                        patch_metrics["eff_tokens"] = eff_tokens
                        patch_metrics["eff_tokens_frac_nonempty"] = float(eff_tokens / max(n_nonempty, 1))
                        patch_metrics["topk_mass"] = topk_mass
                        patch_metrics["hhi"] = hhi
                        patch_metrics["focus_total_cells"] = float(np.sum(prob * total_cells))
                        patch_metrics["focus_total_nuc_px"] = float(np.sum(prob * total_nuc_px))
                        patch_metrics["corr_p_total_cells"] = corr_1d(prob[nonempty], total_cells[nonempty])
                        patch_metrics["corr_p_total_nuc_px"] = corr_1d(prob[nonempty], total_nuc_px[nonempty])
                        # Rank correlation of the raw token scores with cell density (no temperature, no softmax)
                        patch_metrics["spearman_score_vs_total_cells"] = spearman_1d(
                            scores_nonempty, total_cells[nonempty]
                        )
                        patch_metrics["spearman_score_vs_total_nuc_px"] = spearman_1d(
                            scores_nonempty, total_nuc_px[nonempty]
                        )
                        patch_metrics["com_x"] = com_x
                        patch_metrics["com_y"] = com_y
                        patch_metrics["radius"] = radius

                        # Per-type metrics, one block per cell-type axis
                        focus_total_cells = float(np.sum(prob * total_cells))
                        for axis, ax_state in axis_states.items():
                            ax_types = ax_state["safe_type_names"]
                            if not ax_types:
                                continue
                            ax_type_idx = mem_group_type_idx_by_axis[axis][sl]
                            type_cells = np.zeros((len(ax_types), T), dtype=np.float32)
                            valid_type = ax_type_idx >= 0
                            if np.any(valid_type):
                                np.add.at(type_cells, (ax_type_idx[valid_type], tok_idx[valid_type]), 1.0)

                            # Patch-level totals per cell type (abundance baseline)
                            patch_type_totals = type_cells.sum(axis=1)  # (K,)
                            n_patch_cells_for_advantage = float(np.sum(patch_type_totals))
                            for k, tname_safe in enumerate(ax_types):
                                type_cells_k = type_cells[k]
                                weighted_cells_k = float(np.sum(prob * type_cells_k))
                                focus_frac_k = (
                                    weighted_cells_k / (focus_total_cells + EPS) if focus_total_cells > 0 else np.nan
                                )
                                patch_metrics[axis_column("focus_frac", axis, tname_safe)] = focus_frac_k
                                patch_metrics[axis_column("mass_on_tokens", axis, tname_safe)] = float(
                                    np.sum(prob[type_cells_k > 0])
                                )
                                patch_metrics[axis_column("corr_p_cells", axis, tname_safe)] = corr_1d(
                                    prob[nonempty],
                                    type_cells_k[nonempty],
                                )

                                # Cell-type abundance-corrected focus
                                if compute_celltype_advantage:
                                    if n_patch_cells_for_advantage > 0:
                                        expected_frac = float(patch_type_totals[k]) / n_patch_cells_for_advantage
                                    else:
                                        expected_frac = np.nan
                                    patch_metrics[axis_column("expected_focus_frac", axis, tname_safe)] = expected_frac
                                    if np.isfinite(focus_frac_k) and np.isfinite(expected_frac):
                                        patch_metrics[axis_column("focus_advantage", axis, tname_safe)] = (
                                            focus_frac_k - expected_frac
                                        )
                                    else:
                                        patch_metrics[axis_column("focus_advantage", axis, tname_safe)] = np.nan
                    else:
                        # Degenerate focus map (fewer than 2 non-empty tokens or zero-spread scores): NaN columns
                        blank_focus_metrics(
                            patch_metrics,
                            axis_states=axis_states,
                            compute_celltype_advantage=compute_celltype_advantage,
                            n_nonempty_tokens=n_nonempty,
                            frac_nonempty_tokens=float(n_nonempty / T),
                            keep_spread=True,
                        )
                else:
                    blank_focus_metrics(
                        patch_metrics,
                        axis_states=axis_states,
                        compute_celltype_advantage=compute_celltype_advantage,
                    )

                hf["score_mean"][patch_index] = score_mean.reshape(gh, gw).astype(hm_dtype)
                hf["score_max"][patch_index] = score_max.reshape(gh, gw).astype(hm_dtype)
                hf["prob_nonempty"][patch_index] = prob.reshape(gh, gw).astype(hm_dtype)
                hf["total_nuc_px"][patch_index] = total_nuc_px.reshape(gh, gw)
                hf["total_cells"][patch_index] = total_cells.reshape(gh, gw)

                rows.append(patch_metrics)

    df = pd.DataFrame(rows)
    with atomic_output(metrics_out_csv_gz) as tmp_csv:
        df.to_csv(tmp_csv, index=False, compression="gzip")

    slide_summary = {
        "schema_version": int(PER_SLIDE_SCHEMA_VERSION),
        "slide_id": slide_id,
        "n_patches": int(n_patches),
        "n_patch_cells_total": int(group_patch_index.shape[0]),
        "n_membership_rows": int(mem_patch_index.shape[0]),
        "token_grid_h": int(gh),
        "token_grid_w": int(gw),
        "embedding_space": space,
        "similarity": similarity,
        "tau": float(tau),
        # `n_cell_types`: primary axis; `n_cell_types_by_axis`: all axes
        "n_cell_types": int(len(axis_states[next(iter(axis_states))]["unique_cell_types"])),
        "n_cell_types_by_axis": json.dumps(
            {ax: len(st["unique_cell_types"]) for ax, st in axis_states.items()}
        ),
        "membership_parquet": str(membership_parquet_path),
        "embeddings_h5": str(emb_h5_path),
        "slide_h5ad": str(slide_h5ad_path),
    }
    return df, slide_summary



# ---------------------------------------------------------------------
# Slide processing
# ---------------------------------------------------------------------


def process_slide(
    emb_h5_path: Path,
    slide_h5ad_path: Path,
    membership_parquet_path: Optional[Path],
    heatmap_out_h5: Optional[Path],
    token_metrics_out_csv_gz: Optional[Path],
    token_slide_summary_out_csv: Optional[Path],
    patch_cell_out_parquet: Optional[Path],
    patch_cell_summary_out_csv_gz: Optional[Path],
    patch_cell_slide_summary_out_csv: Optional[Path],
    similarity: str,
    space: str,
    tau: float,
    tau_mode: str,
    tau_scale: float,
    celltype_axes: Dict[str, Optional[Dict[str, str]]],
    topk: int,
    heatmap_dtype: str,
    output_level: str,
    compute_intra_patch: bool = True,
    compute_shuffled: bool = True,
    n_shuffled_per_group: int = 4,
    rng_for_stream: Optional[Callable[[int], np.random.Generator]] = None,
    compute_celltype_advantage: bool = True,
    max_in_memory_bytes: int = 0,
) -> Dict[str, Optional[object]]:
    slide_id = emb_h5_path.stem
    do_token = wants_token_output(output_level)
    do_patch_cell = wants_patch_cell_output(output_level)

    # Check the space before any read (see CLIP_PATCH_AND_CELL_HEADS_ARE_INDEPENDENT)
    assert_patch_cell_space_comparable(space, slide_id=slide_id)

    if do_token:
        if tau_mode == "fixed" and tau <= 0:
            raise ValueError(f"--tau_mode fixed requires tau > 0, got {tau}")
        if topk <= 0:
            raise ValueError(f"topk must be > 0, got {topk}")
        if membership_parquet_path is None:
            raise ValueError("membership_parquet_path must be provided when token-level output is requested.")
        if heatmap_out_h5 is None:
            raise ValueError("heatmap_out_h5 must be provided when token-level output is requested.")
        if token_metrics_out_csv_gz is None:
            raise ValueError("token_metrics_out_csv_gz must be provided when token-level output is requested.")
        if token_slide_summary_out_csv is None:
            raise ValueError("token_slide_summary_out_csv must be provided when token-level output is requested.")

    if do_patch_cell:
        if patch_cell_out_parquet is None:
            raise ValueError("patch_cell_out_parquet must be provided when patch-cell output is requested.")
        if patch_cell_summary_out_csv_gz is None:
            raise ValueError("patch_cell_summary_out_csv_gz must be provided when patch-cell output is requested.")
        if patch_cell_slide_summary_out_csv is None:
            raise ValueError("patch_cell_slide_summary_out_csv must be provided when patch-cell output is requested.")

    cell_id_to_type_fine, unique_cell_types_fine = load_celltype_map_from_h5ad(slide_h5ad_path)

    # One label state per cell-type axis; `cell_id_to_type` / `unique_cell_types` refer to the primary axis
    axis_states: Dict[str, Dict[str, object]] = {}
    for axis, mapping in celltype_axes.items():
        ax_map = apply_label_map_to_dict(cell_id_to_type_fine, mapping)
        ax_types = axis_cell_types(unique_cell_types_fine, mapping)
        axis_states[axis] = {
            "cell_id_to_type": ax_map,
            "unique_cell_types": ax_types,
            "type_to_idx": {t: i for i, t in enumerate(ax_types)},
            "safe_type_names": [sanitize_col_suffix(t) for t in ax_types],
        }
    primary_axis = next(iter(axis_states))
    cell_id_to_type = axis_states[primary_axis]["cell_id_to_type"]
    unique_cell_types = axis_states[primary_axis]["unique_cell_types"]

    # Larger HDF5 chunk cache (used when the patch-cell matrix is not read into RAM)
    with h5py.File(
        emb_h5_path, "r",
        rdcc_nbytes=FALLBACK_RDCC_NBYTES, rdcc_nslots=FALLBACK_RDCC_NSLOTS,
    ) as f:
        patch_ids = _decode_str_list(f["patch_id"][:])
        n_patches = len(patch_ids)
        patch_id_to_index = {pid: i for i, pid in enumerate(patch_ids)}

        if space == "he":
            patch_name = "patch_emb_he"
            patch_cell_name = "patch_cell_emb_he"
        elif space == "clip":
            patch_name = "patch_emb_he_clip"
            patch_cell_name = "patch_cell_emb_he_clip"
        else:
            raise ValueError(f"Unknown space='{space}'")

        if patch_name not in f:
            raise KeyError(f"{emb_h5_path} missing dataset '{patch_name}'")
        if patch_cell_name not in f:
            raise KeyError(f"{emb_h5_path} missing dataset '{patch_cell_name}'")
        if "patch_cell_patch_index" not in f or "patch_cell_cell_id_int" not in f or "patch_cell_alpha" not in f:
            raise KeyError(
                f"{emb_h5_path} is missing patch-cell datasets "
                "('patch_cell_patch_index', 'patch_cell_cell_id_int', 'patch_cell_alpha')."
            )

        patch_emb = f[patch_name][:].astype(np.float32, copy=False)
        if patch_emb.shape[0] != n_patches:
            raise RuntimeError(
                f"patch embeddings shape mismatch for {slide_id}: got {patch_emb.shape[0]} rows, expected {n_patches}"
            )

        group_patch_index = f["patch_cell_patch_index"][:].astype(np.int32, copy=False)
        group_cell_id_int = f["patch_cell_cell_id_int"][:].astype(np.int32, copy=False)
        group_alpha = f["patch_cell_alpha"][:].astype(np.float32, copy=False)
        group_emb_h5 = f[patch_cell_name]

        if group_emb_h5.ndim != 2:
            raise RuntimeError(f"{patch_cell_name} must be 2D, got shape={group_emb_h5.shape}")
        if group_emb_h5.shape[0] != group_patch_index.shape[0]:
            raise RuntimeError(
                f"{patch_cell_name} rows ({group_emb_h5.shape[0]}) do not match patch_cell_patch_index "
                f"({group_patch_index.shape[0]})"
            )
        if group_emb_h5.shape[1] == 0:
            raise RuntimeError(
                f"{patch_cell_name} has dim 0 for slide '{slide_id}'. "
                "This usually means the token projection head was not exported, so `--space clip` cannot be used."
            )

        # Reduce the patch embedding to the per-cell width (CLS half for Virchow2)
        patch_emb = reconcile_patch_dim_to_cell(
            patch_emb, int(group_emb_h5.shape[1]), slide_id=slide_id, space=space
        )

        # Read the patch-cell matrix into RAM when it fits in --max_in_memory_gb
        matrix_in_memory = matrix_fits_in_memory(group_emb_h5, max_in_memory_bytes)
        if matrix_in_memory:
            n_bytes = int(group_emb_h5.shape[0]) * int(group_emb_h5.shape[1]) * 4
            print(
                f"  0. Reading the patch-cell matrix into memory once "
                f"({group_emb_h5.shape[0]:,} x {group_emb_h5.shape[1]}, {n_bytes / 1e9:.2f} GB)."
            )
            group_emb: "PatchCellMatrix" = np.asarray(group_emb_h5[:], dtype=np.float32)
        else:
            print(
                f"  0. Patch-cell matrix stays on disk (over --max_in_memory_gb); reads go "
                f"through a {FALLBACK_RDCC_NBYTES / 1e6:.0f} MB HDF5 chunk cache."
            )
            group_emb = group_emb_h5

        gh = int(f.attrs.get("token_grid_h", 0)) if do_token else 0
        gw = int(f.attrs.get("token_grid_w", 0)) if do_token else 0
        if do_token and (gh <= 0 or gw <= 0):
            raise RuntimeError(
                f"Could not read token grid size from {emb_h5_path}. "
                "Expected attrs 'token_grid_h' and 'token_grid_w'."
            )

        # -----------------------------------------------------------------
        # 1) Compute patch-vs-cell-group similarities
        # -----------------------------------------------------------------
        print(f"  1. Computing patch-vs-cell similarities in '{space}' space with '{similarity}' similarity...")
        group_score = compute_group_similarity_scores(
            patch_emb=patch_emb,
            group_patch_index=group_patch_index,
            group_emb=group_emb,
            similarity=similarity,
        )

        # Per-group cell type index, one array per cell-type axis.
        group_type_idx_by_axis: Dict[str, np.ndarray] = {}
        for axis, ax_state in axis_states.items():
            ax_map = ax_state["cell_id_to_type"]
            ax_t2i = ax_state["type_to_idx"]
            group_type_idx_by_axis[axis] = np.fromiter(
                (ax_t2i.get(ax_map.get(int(cid), None), -1) for cid in group_cell_id_int),
                dtype=np.int32,
                count=group_cell_id_int.shape[0],
            )

        # Per-patch summary over groups
        df_group_summary = compute_patch_level_group_summary(
            n_patches=n_patches,
            group_patch_index=group_patch_index,
            group_alpha=group_alpha,
            group_score=group_score,
        )

        # Intra-patch cell-cell cosine (collapse detector)
        if compute_intra_patch:
            df_intra = compute_intra_patch_cell_cell(
                n_patches=n_patches,
                group_patch_index=group_patch_index,
                group_emb=group_emb,
            )
            df_group_summary = df_group_summary.merge(
                df_intra, on="patch_index", how="left"
            )
            # Within-patch cell separation: cell_sim_mean - intra_patch_cc_cos_mean
            df_group_summary["within_patch_cell_separation"] = (
                df_group_summary["cell_sim_mean"].astype(float)
                - df_group_summary["intra_patch_cc_cos_mean"].astype(float)
            )
        else:
            for col in [
                "intra_patch_cc_n_pairs",
                "intra_patch_cc_cos_mean",
                "intra_patch_cc_cos_min",
                "intra_patch_cc_cos_std",
                "within_patch_cell_separation",
            ]:
                df_group_summary[col] = np.nan

        # Shuffled patch-cell baseline (per-group cosines + per-patch aggregates)
        shuffled_per_group_df = pd.DataFrame()
        if compute_shuffled and rng_for_stream is not None:
            shuffled_per_group_df = compute_shuffled_baseline(
                patch_emb=patch_emb,
                group_patch_index=group_patch_index,
                group_cell_id_int=group_cell_id_int,
                group_type_idx_by_axis=group_type_idx_by_axis,
                primary_axis=primary_axis,
                group_emb=group_emb,
                n_shuffled_per_group=int(n_shuffled_per_group),
                rng_for_stream=rng_for_stream,
            )

            # Per-patch unweighted mean (same aggregation as cell_sim_mean)
            def _patch_unweighted_mean(values, row_mask: Optional[np.ndarray] = None) -> np.ndarray:
                """Per-patch unweighted mean over the selected rows; NaN where none is finite."""
                vals = pd.to_numeric(pd.Series(np.asarray(values)), errors="coerce").to_numpy()
                finite = np.isfinite(vals)
                if row_mask is not None:
                    finite = finite & np.asarray(row_mask, dtype=bool)
                sum_v = np.bincount(
                    group_patch_index,
                    weights=np.where(finite, vals, 0.0),
                    minlength=n_patches,
                ).astype(np.float64)
                # Count of finite contributions per patch (for NaN propagation).
                n_finite = np.bincount(
                    group_patch_index,
                    weights=finite.astype(np.float64),
                    minlength=n_patches,
                ).astype(np.float64)
                out = np.full(n_patches, np.nan, dtype=np.float32)
                valid = n_finite > 0
                out[valid] = (sum_v[valid] / n_finite[valid]).astype(np.float32)
                return out

            agg_rows = {
                "patch_index": np.arange(n_patches, dtype=np.int32),
            }
            # Minuend of each gap (the scored flavours use the scored rows only)
            minuend_by_col: Dict[str, str] = {}

            shuffled_value_cols = [
                c for c in shuffled_per_group_df.columns
                if c.startswith("shuffled_cos_")
            ]
            for col in shuffled_value_cols:
                agg_rows[col] = _patch_unweighted_mean(shuffled_per_group_df[col].to_numpy())
                minuend_by_col[col] = "cell_sim_mean"

            # Scored flavours per axis: sameCT / diffCT over the rows whose cell type is not in EXCLUDED_FROM_SCORED
            for axis, ax_state in axis_states.items():
                ax_types = list(ax_state["unique_cell_types"])
                excluded_idx = [
                    i for i, t in enumerate(ax_types) if str(t).strip() in set(EXCLUDED_FROM_SCORED)
                ]
                ax_idx = group_type_idx_by_axis[axis]
                scored_mask = (ax_idx >= 0) & ~np.isin(ax_idx, excluded_idx)

                minuend_col = scored_mean_column(axis)
                agg_rows[minuend_col] = _patch_unweighted_mean(group_score, scored_mask)
                for flavour in ("sameCT", "diffCT"):
                    src = shuffled_ct_column(flavour, axis)
                    if src not in shuffled_per_group_df.columns:
                        continue
                    dst = shuffled_ct_column(scored_flavour(flavour), axis)
                    agg_rows[dst] = _patch_unweighted_mean(
                        shuffled_per_group_df[src].to_numpy(), scored_mask
                    )
                    minuend_by_col[dst] = minuend_col

            df_shuffled_patch = pd.DataFrame(agg_rows)
            df_group_summary = df_group_summary.merge(df_shuffled_patch, on="patch_index", how="left")
            # One gap per shuffled column: minuend - shuffled baseline
            for col, minuend_col in minuend_by_col.items():
                parsed = parse_shuffled_ct_column(col)
                if parsed is None:
                    raise RuntimeError(
                        f"Unparseable shuffled-baseline column '{col}'. Every such column must be "
                        "built by shuffled_ct_column() or be the axis-free 'shuffled_cos_any_mean'."
                    )
                flavour, axis = parsed
                gap_col = shuffled_gap_column(flavour, axis)
                df_group_summary[gap_col] = (
                    df_group_summary[minuend_col].astype(float)
                    - df_group_summary[col].astype(float)
                )

            # Headroom-normalised `any` gap: gap / (1 - shuffled_cos_any_mean)
            any_gap_col = shuffled_gap_column("any")
            if any_gap_col in df_group_summary.columns and "shuffled_cos_any_mean" in df_group_summary.columns:
                headroom = 1.0 - df_group_summary["shuffled_cos_any_mean"].astype(float)
                df_group_summary[HEADROOM_NORM_GAP_COLUMN] = (
                    df_group_summary[any_gap_col].astype(float)
                    / np.where(headroom > EPS, headroom, np.nan)
                )
        else:
            blank_cols = [
                "shuffled_cos_any_mean",
                "shuffled_cos_sameCT_mean",
                "shuffled_cos_diffCT_mean",
                shuffled_gap_column("any"),
                shuffled_gap_column("sameCT"),
                shuffled_gap_column("diffCT"),
                HEADROOM_NORM_GAP_COLUMN,
            ]
            for ax in axis_states:
                blank_cols.append(scored_mean_column(ax))
                for flavour in ("sameCT", "diffCT"):
                    for fl in (flavour, scored_flavour(flavour)):
                        blank_cols.append(shuffled_ct_column(fl, ax))
                        blank_cols.append(shuffled_gap_column(fl, ax))
            for col in blank_cols:
                df_group_summary[col] = np.nan

    result = {
        "token_patch_metrics": None,
        "token_slide_summary": None,
        "patch_cell_patch_summary": None,
        "patch_cell_slide_summary": None,
        "shuffled_per_group": shuffled_per_group_df if compute_shuffled else pd.DataFrame(),
    }

    # ---------------------------------------------------------------------
    # 2) Optional patch-cell outputs
    # ---------------------------------------------------------------------
    if do_patch_cell:
        print("  2. Writing raw patch-cell similarity table...")
        write_patch_cell_similarity_parquet(
            out_path=patch_cell_out_parquet,
            slide_id=slide_id,
            patch_ids=patch_ids,
            group_patch_index=group_patch_index,
            group_cell_id_int=group_cell_id_int,
            group_alpha=group_alpha,
            group_score=group_score,
            cell_id_to_type=cell_id_to_type,
        )

        df_patch_cell_summary = df_group_summary.copy()
        df_patch_cell_summary.insert(0, "schema_version", int(PER_SLIDE_SCHEMA_VERSION))
        df_patch_cell_summary.insert(1, "slide_id", slide_id)
        df_patch_cell_summary.insert(2, "patch_id", np.asarray(patch_ids, dtype=object))
        df_patch_cell_summary["embedding_space"] = space
        df_patch_cell_summary["similarity"] = similarity
        with atomic_output(patch_cell_summary_out_csv_gz) as tmp_csv:
            df_patch_cell_summary.to_csv(tmp_csv, index=False, compression="gzip")

        patch_cell_slide_summary = compute_patch_cell_slide_summary(
            slide_id=slide_id,
            n_patches=n_patches,
            group_cell_id_int=group_cell_id_int,
            group_alpha=group_alpha,
            group_score=group_score,
            unique_cell_types=unique_cell_types,
            space=space,
            similarity=similarity,
            embeddings_h5=emb_h5_path,
            slide_h5ad=slide_h5ad_path,
            patch_cell_out_parquet=patch_cell_out_parquet,
            df_group_summary=df_group_summary,
            shuffled_per_group=result.get("shuffled_per_group", pd.DataFrame()),
        )

        result["patch_cell_patch_summary"] = df_patch_cell_summary
        result["patch_cell_slide_summary"] = patch_cell_slide_summary

        # Per-slide file used to rebuild patch_cell_similarity_by_slide.csv
        with atomic_output(patch_cell_slide_summary_out_csv) as tmp_csv:
            pd.DataFrame([patch_cell_slide_summary]).to_csv(tmp_csv, index=False)

    # ---------------------------------------------------------------------
    # 3) Optional token-level outputs
    # ---------------------------------------------------------------------
    if do_token:
        df_token, token_slide_summary = build_token_heatmaps_and_metrics(
            slide_id=slide_id,
            patch_ids=patch_ids,
            group_patch_index=group_patch_index,
            group_cell_id_int=group_cell_id_int,
            group_alpha=group_alpha,
            group_score=group_score,
            group_type_idx_by_axis=group_type_idx_by_axis,
            df_group_summary=df_group_summary,
            patch_id_to_index=patch_id_to_index,
            membership_parquet_path=membership_parquet_path,
            heatmap_out_h5=heatmap_out_h5,
            metrics_out_csv_gz=token_metrics_out_csv_gz,
            space=space,
            similarity=similarity,
            tau=tau,
            tau_mode=tau_mode,
            tau_scale=tau_scale,
            topk=topk,
            heatmap_dtype=heatmap_dtype,
            axis_states=axis_states,
            emb_h5_path=emb_h5_path,
            slide_h5ad_path=slide_h5ad_path,
            gh=gh,
            gw=gw,
            compute_celltype_advantage=compute_celltype_advantage,
        )
        result["token_patch_metrics"] = df_token
        result["token_slide_summary"] = token_slide_summary

        # Per-slide file used to rebuild similarity_metrics_by_slide.csv
        with atomic_output(token_slide_summary_out_csv) as tmp_csv:
            pd.DataFrame([token_slide_summary]).to_csv(tmp_csv, index=False)

    return result



# ---------------------------------------------------------------------
# Resume / aggregation helpers
# ---------------------------------------------------------------------


def assert_focus_family_informative(
    *,
    metrics_dir: Path,
    max_median_entropy_norm: float,
    tau_mode: str,
) -> None:
    """Raise if the token focus map is degenerate over the run (median entropy_norm above max_median_entropy_norm, or all NaN)."""
    combined = metrics_dir / "similarity_metrics_all_slides.csv.gz"
    if not combined.exists():
        return

    # Read only the columns present in the file
    wanted = ["entropy_norm", "focus_family_degenerate"]
    try:
        header = pd.read_csv(combined, nrows=0).columns
    except pd.errors.EmptyDataError:
        return
    cols = [c for c in wanted if c in header]
    if "entropy_norm" not in cols:
        return
    frame = pd.read_csv(combined, usecols=cols)

    values = pd.to_numeric(frame["entropy_norm"], errors="coerce").dropna()

    degenerate_share = np.nan
    if "focus_family_degenerate" in frame.columns:
        flags = frame["focus_family_degenerate"].map(
            lambda v: str(v).strip().lower() in {"true", "1", "1.0"}
        )
        degenerate_share = float(flags.mean()) if len(flags) else np.nan
    share_txt = (
        "unknown" if not np.isfinite(degenerate_share) else f"{100.0 * degenerate_share:.1f}%"
    )

    # All-NaN entropy_norm: every patch is degenerate
    if values.empty:
        raise RuntimeError(
            f"Token-focus family is degenerate: entropy_norm is NaN for all "
            f"{len(frame)} patches (fewer than 2 non-empty tokens or zero-spread scores; "
            f"focus_family_degenerate = {share_txt}, tau_mode='{tau_mode}'). Check that the membership "
            "parquet projects cells onto the token grid for these slides."
        )

    median = float(values.median())
    if median <= max_median_entropy_norm:
        print(f"  Token-focus family OK: median entropy_norm = {median:.4f} over {len(values)} "
              f"patches (gate {max_median_entropy_norm:.4f}, tau_mode={tau_mode}); "
              f"degenerate patches: {share_txt}.")
        return
    raise RuntimeError(
        f"Token-focus family is degenerate: median entropy_norm = {median:.4f} over "
        f"{len(values)} patches, above --max_median_entropy_norm = "
        f"{max_median_entropy_norm:.4f} (degenerate patches: {share_txt}, tau_mode='{tau_mode}'). "
        "With --tau_mode fixed, lower --tau or use scale_free; with scale_free, lower --tau_scale."
    )


# Schema version of the per-slide files (stamped into each one and checked on read)
PER_SLIDE_SCHEMA_VERSION = 4


def assert_sidecar_schema(df: pd.DataFrame, path: Path) -> None:
    """Raise if a per-slide file was written with another PER_SLIDE_SCHEMA_VERSION."""
    found = df["schema_version"].iloc[0] if "schema_version" in df.columns and len(df) else 1
    try:
        found = int(found)
    except (TypeError, ValueError):
        found = 1
    if found != PER_SLIDE_SCHEMA_VERSION:
        raise RuntimeError(
            f"{path} was written with per-slide schema_version={found}, but this run is at "
            f"{PER_SLIDE_SCHEMA_VERSION}. Delete the stale per-slide outputs (or use a clean output "
            "directory) and re-run those slides."
        )


def per_slide_output_paths(
    slide_id: str,
    metrics_dir: Path,
    heatmaps_dir: Path,
    patch_cell_dir: Path,
    output_level: str,
) -> Dict[str, Path]:
    """Expected per-slide output files for output_level (a slide is done for --resume when all of them exist)."""
    paths: Dict[str, Path] = {}
    if wants_token_output(output_level):
        paths["token_heatmap_h5"] = heatmaps_dir / f"{slide_id}_similarity_heatmaps.h5"
        paths["token_metrics_csv_gz"] = metrics_dir / f"{slide_id}_similarity_metrics.csv.gz"
        paths["token_slide_summary_csv"] = metrics_dir / f"{slide_id}_token_slide_summary.csv"
    if wants_patch_cell_output(output_level):
        paths["patch_cell_parquet"] = patch_cell_dir / f"{slide_id}_patch_cell_similarity.parquet"
        paths["patch_cell_patch_summary_csv_gz"] = metrics_dir / f"{slide_id}_patch_cell_patch_summary.csv.gz"
        paths["patch_cell_slide_summary_csv"] = metrics_dir / f"{slide_id}_patch_cell_slide_summary.csv"
    return paths


def build_combined_outputs(
    selected_slide_ids: List[str],
    metrics_dir: Path,
    do_token: bool,
    do_patch_cell: bool,
    no_stratified_aggregation: bool,
    headline_min_cells: int,
) -> None:
    """
    Rebuild the cross-slide tables from the per-slide files on disk for `selected_slide_ids` (the full
    cohort, not the --slides subset). Missing per-slide files are skipped with a warning.
    """

    def _existing(suffix: str) -> List[Path]:
        out: List[Path] = []
        for sid in selected_slide_ids:
            p = metrics_dir / f"{sid}{suffix}"
            if p.exists():
                out.append(p)
            else:
                print(f"  [warn] missing per-slide file for aggregation: {p.name}")
        return out

    def _concat(paths: List[Path]) -> pd.DataFrame:
        frames = []
        for path in paths:
            frame = pd.read_csv(path)  # gzip inferred from .gz suffix
            # Check the per-slide schema version
            assert_sidecar_schema(frame, path)
            frames.append(frame)
        return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()

    if do_token:
        token_all = _concat(_existing("_similarity_metrics.csv.gz"))
        token_by_slide = _concat(_existing("_token_slide_summary.csv"))
        if not token_all.empty:
            with atomic_output(metrics_dir / "similarity_metrics_all_slides.csv.gz") as tmp:
                token_all.to_csv(tmp, index=False, compression="gzip")
            print(f"Saved combined token-level patch metrics ({len(token_all)} rows).")
        if not token_by_slide.empty:
            with atomic_output(metrics_dir / "similarity_metrics_by_slide.csv") as tmp:
                token_by_slide.to_csv(tmp, index=False)
            print(f"Saved token-level slide summary ({len(token_by_slide)} slides).")

    if do_patch_cell:
        df_patch_all = _concat(_existing("_patch_cell_patch_summary.csv.gz"))
        pc_by_slide = _concat(_existing("_patch_cell_slide_summary.csv"))
        if not df_patch_all.empty:
            with atomic_output(metrics_dir / "patch_cell_patch_summary_all_slides.csv.gz") as tmp:
                df_patch_all.to_csv(tmp, index=False, compression="gzip")
            print(f"Saved combined patch-cell patch summary ({len(df_patch_all)} rows).")
        if not pc_by_slide.empty:
            with atomic_output(metrics_dir / "patch_cell_similarity_by_slide.csv") as tmp:
                pc_by_slide.to_csv(tmp, index=False)
            print(f"Saved patch-cell slide summary ({len(pc_by_slide)} slides).")

        # n_patch_cells stratification: by-slide table over patches with >= headline_min_cells cells,
        # and by-(slide, n_patch_cells_bin) table
        if not no_stratified_aggregation and not df_patch_all.empty and "n_patch_cells" in df_patch_all.columns:
            df_strat_src = df_patch_all.copy()
            df_strat_src["n_patch_cells_bin"] = bin_npc(df_strat_src["n_patch_cells"].to_numpy())

            min_cells = int(headline_min_cells)
            mask_headline = pd.to_numeric(df_strat_src["n_patch_cells"], errors="coerce").fillna(0).astype(int) >= min_cells
            df_headline_src = df_strat_src[mask_headline].copy()
            if not df_headline_src.empty:
                num_cols = [
                    c for c in df_headline_src.columns
                    if c not in {"slide_id", "patch_id", "patch_index", "embedding_space", "similarity",
                                 "source", "subset", "n_patch_cells_bin"}
                    and pd.api.types.is_numeric_dtype(df_headline_src[c])
                ]
                df_headline = (
                    df_headline_src.groupby(["slide_id"], dropna=False)[num_cols]
                    .mean(numeric_only=True)
                    .reset_index()
                )
                df_headline.insert(1, "n_patches_with_geq_min_cells", df_headline_src.groupby(["slide_id"]).size().values)
                df_headline.insert(2, "headline_min_cells", min_cells)
                with atomic_output(metrics_dir / "similarity_metrics_by_slide_headline.csv") as tmp:
                    df_headline.to_csv(tmp, index=False)
                print("Saved headline-only slide summary.")

            num_cols_strat = [
                c for c in df_strat_src.columns
                if c not in {"slide_id", "patch_id", "patch_index", "embedding_space", "similarity",
                             "source", "subset", "n_patch_cells_bin"}
                and pd.api.types.is_numeric_dtype(df_strat_src[c])
            ]
            if num_cols_strat:
                df_strat = (
                    df_strat_src.groupby(["slide_id", "n_patch_cells_bin"], dropna=False)[num_cols_strat]
                    .mean(numeric_only=True)
                    .reset_index()
                )
                df_strat["n_patches_in_bin"] = (
                    df_strat_src.groupby(["slide_id", "n_patch_cells_bin"]).size().reset_index(name="_n")["_n"].values
                )
                with atomic_output(metrics_dir / "patch_cell_similarity_by_slide_by_npc_bin.csv") as tmp:
                    df_strat.to_csv(tmp, index=False)
                print("Saved n_patch_cells-stratified summary.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(args):

    # Reproducibility (the sampling-based diagnostics use the streams below)
    set_global_seed(args.seed)

    # Independent RNG stream per key (seed, stable_seed(slide_id), sub-stream)
    def stream_rng(*key: int) -> np.random.Generator:
        return np.random.default_rng([int(args.seed), *(int(k) for k in key)])

    # Paths
    eval_root = get_full_path("eval", "eval_datasets", "st_eval_data", f"eval_st_{args.training_dataset_id}")
    
    emb_dir = eval_root / "he_embeddings" / f"HEembs_t{args.training_num}e{args.checkpoint_num}"
    if not emb_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")
    
    out_dir = get_full_path("eval", "eval_datasets", "st_eval_data", f"eval_st_{args.training_dataset_id}", "embeddings_analysis", "similarity_heatmaps_HEembs", f"HEembs_t{args.training_num}e{args.checkpoint_num}_{args.space}")

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    do_token = wants_token_output(args.output_level)
    do_patch_cell = wants_patch_cell_output(args.output_level)

    heatmaps_dir = out_dir / "heatmaps"
    patch_cell_dir = out_dir / "patch_cell"
    if do_token:
        heatmaps_dir.mkdir(parents=True, exist_ok=True)
    if do_patch_cell:
        patch_cell_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the cell-type axes once, so a bad --coarse_mapping_version fails before any slide.
    celltype_axes = build_axes(
        args.ct_cat_id, args.celltype_axes, args.coarse_mapping_version
    )

    # Refuse to write into a folder written by another analysis version (unless --overwrite)
    existing_cfg_path = out_dir / "run_config.json"
    if existing_cfg_path.exists():
        try:
            existing_version = int(json.loads(existing_cfg_path.read_text()).get("a0_analysis_version", 1))
        except (json.JSONDecodeError, TypeError, ValueError):
            existing_version = 1
        if existing_version != A0_ANALYSIS_VERSION and not args.overwrite:
            raise RuntimeError(
                f"{out_dir} was written by A0_ANALYSIS_VERSION={existing_version}, but this is "
                f"version {A0_ANALYSIS_VERSION}. Re-run into a clean directory, or pass "
                "--overwrite to replace it."
            )

    run_cfg = {
        "training_dataset_id": args.training_dataset_id,
        "training_num": str(args.training_num),
        "checkpoint_num": str(args.checkpoint_num),
        "output_level": args.output_level,
        "space": args.space,
        "similarity": args.similarity,
        "a0_analysis_version": int(A0_ANALYSIS_VERSION),
        "tau": float(args.tau),
        "tau_mode": str(args.tau_mode),
        "tau_scale": float(args.tau_scale),
        "max_median_entropy_norm": float(args.max_median_entropy_norm),
        "ct_cat_id": str(args.ct_cat_id),
        "celltype_axes": list(args.celltype_axes),
        "coarse_mapping_version": str(args.coarse_mapping_version),
        "topk": int(args.topk),
        "heatmap_dtype": args.heatmap_dtype,
        "seed": int(args.seed),
        "no_intra_patch_cellcell": bool(args.no_intra_patch_cellcell),
        "no_shuffled_baseline": bool(args.no_shuffled_baseline),
        "n_shuffled_pairs_per_group": int(args.n_shuffled_pairs_per_group),
        "no_celltype_advantage": bool(args.no_celltype_advantage),
        # Read strategy only (not checked on --resume)
        "max_in_memory_gb": float(args.max_in_memory_gb),
        "headline_min_cells": int(args.headline_min_cells),
        "no_stratified_aggregation": bool(args.no_stratified_aggregation),
        "embeddings_dir": str(emb_dir),
        "output_dir": str(out_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    # On --resume, the existing run_config.json must match on the guarded parameters
    run_config_path = out_dir / "run_config.json"
    if args.resume and run_config_path.exists():
        prev_cfg = json.loads(run_config_path.read_text())
        guarded_keys = [
            "training_dataset_id", "training_num", "checkpoint_num", "space",
            "similarity", "tau", "topk", "heatmap_dtype", "seed",
            "no_intra_patch_cellcell", "no_shuffled_baseline",
            "n_shuffled_pairs_per_group", "no_celltype_advantage",
            "a0_analysis_version", "tau_mode", "tau_scale",
            "ct_cat_id", "celltype_axes", "coarse_mapping_version",
        ]
        mismatches = {
            k: {"previous": prev_cfg.get(k), "current": run_cfg.get(k)}
            for k in guarded_keys
            if prev_cfg.get(k) != run_cfg.get(k)
        }
        if mismatches:
            raise RuntimeError(
                "Cannot --resume: the existing run_config.json was produced with different "
                f"parameters than this invocation: {json.dumps(mismatches, sort_keys=True)}. "
                "Re-run without --resume to overwrite, or use a clean output directory."
            )

    with atomic_output(run_config_path) as tmp_cfg:
        with open(tmp_cfg, "w") as f:
            json.dump(run_cfg, f, indent=2, sort_keys=True)
    
    # Get slide IDs list
    # STHELAR slides only in evaluation dataset
    metadata_csv = eval_root / f"metadata_eval_st_{args.training_dataset_id}.csv"
    metadata = pd.read_csv(metadata_csv)
    if "source" not in metadata.columns or "slide_id" not in metadata.columns:
        raise KeyError("metadata csv must contain 'source' and 'slide_id'")
    # cohort_slide_ids: all STHELAR slides (combined tables); process_slide_ids: slides to process (--slides)
    cohort_slide_ids = sorted(metadata.loc[metadata["source"] == "STHELAR", "slide_id"].astype(str).unique().tolist())
    if len(cohort_slide_ids) == 0:
        raise ValueError("No STHELAR slides found in metadata.")

    # Optionally filter by --slides (processing scope only)
    process_slide_ids = list(cohort_slide_ids)
    if args.slides:
        wanted = set(args.slides)
        process_slide_ids = [p for p in cohort_slide_ids if p in wanted]
        if not process_slide_ids:
            raise RuntimeError("After filtering with --slides, no slide IDs remain.")
        print(
            f"--slides: processing {len(process_slide_ids)} of {len(cohort_slide_ids)} STHELAR "
            "slide(s); the combined cross-slide tables are still rebuilt over the full cohort."
        )

    # Get embedding files list (processing scope)
    emb_files = sorted([p for p in emb_dir.glob("*.h5") if p.stem in process_slide_ids], key=lambda p: p.stem)

    print(f"Eval root:      {eval_root}")
    print(f"Embeddings dir: {emb_dir}")
    print(f"Output dir:     {out_dir}")
    print(f"Output level:   {args.output_level}")
    print(f"Found {len(emb_files)} slide embedding file(s).")

    # Check the cell-type vocabulary of every slide before processing
    print("Preflight: checking the cell-type vocabulary against every requested axis...")
    observed_labels: set = set()
    for p in emb_files:
        observed_labels.update(load_celltype_map_from_h5ad(eval_root / "st" / f"{p.stem}.h5ad")[1])
    assert_vocabulary_covered(
        observed_labels, celltype_axes, where=f"{len(emb_files)} slides under {eval_root / 'st'}"
    )

    # Process each slide (the combined tables are rebuilt from the per-slide files afterwards)
    n_processed = 0
    n_skipped = 0
    for i, emb_h5_path in enumerate(emb_files, start=1):
        
        slide_id = emb_h5_path.stem
        out_paths = per_slide_output_paths(
            slide_id, metrics_dir, heatmaps_dir, patch_cell_dir, args.output_level
        )

        print(f"\n=== [{i}/{len(emb_files)}] Slide '{slide_id}' ===")

        if args.resume and all(p.exists() for p in out_paths.values()):
            n_skipped += 1
            print("  [resume] all outputs already present -> skipping.")
            continue

        slide_h5ad_path = eval_root / "st" / f"{slide_id}.h5ad"
        if not slide_h5ad_path.exists():
            raise FileNotFoundError(f"Missing slide h5ad for '{slide_id}': {slide_h5ad_path}")

        membership_parquet_path = None
        if do_token:
            membership_parquet_path = resolve_membership_parquet(
                emb_h5_path=emb_h5_path,
                eval_root=eval_root,
                slide_id=slide_id,
            )

        print(f"  embeddings : {emb_h5_path}")
        print(f"  st h5ad    : {slide_h5ad_path}")
        if do_token:
            print(f"  membership : {membership_parquet_path}")
        if do_patch_cell:
            print(f"  patch_cell : {out_paths['patch_cell_parquet']}")

        process_slide(
            emb_h5_path=emb_h5_path,
            slide_h5ad_path=slide_h5ad_path,
            membership_parquet_path=membership_parquet_path,
            heatmap_out_h5=out_paths.get("token_heatmap_h5"),
            token_metrics_out_csv_gz=out_paths.get("token_metrics_csv_gz"),
            token_slide_summary_out_csv=out_paths.get("token_slide_summary_csv"),
            patch_cell_out_parquet=out_paths.get("patch_cell_parquet"),
            patch_cell_summary_out_csv_gz=out_paths.get("patch_cell_patch_summary_csv_gz"),
            patch_cell_slide_summary_out_csv=out_paths.get("patch_cell_slide_summary_csv"),
            similarity=args.similarity,
            space=args.space,
            tau=args.tau,
            tau_mode=args.tau_mode,
            tau_scale=args.tau_scale,
            celltype_axes=celltype_axes,
            topk=args.topk,
            heatmap_dtype=args.heatmap_dtype,
            output_level=args.output_level,
            compute_intra_patch=(not args.no_intra_patch_cellcell),
            compute_shuffled=(not args.no_shuffled_baseline),
            n_shuffled_per_group=int(args.n_shuffled_pairs_per_group),
            # One stream family per slide (keyed on the slide id), sub-keyed by flavour
            rng_for_stream=(lambda sub, _sid=slide_id: stream_rng(stable_seed(_sid), sub)),
            compute_celltype_advantage=(not args.no_celltype_advantage),
            max_in_memory_bytes=int(float(args.max_in_memory_gb) * 1e9),
        )
        n_processed += 1

    # Rebuild the combined tables from the per-slide files on disk (full cohort)
    build_combined_outputs(
        selected_slide_ids=cohort_slide_ids,
        metrics_dir=metrics_dir,
        do_token=do_token,
        do_patch_cell=do_patch_cell,
        no_stratified_aggregation=args.no_stratified_aggregation,
        headline_min_cells=int(args.headline_min_cells),
    )

    if do_token:
        assert_focus_family_informative(
            metrics_dir=metrics_dir,
            max_median_entropy_norm=float(args.max_median_entropy_norm),
            tau_mode=str(args.tau_mode),
        )

    print(f"\nDone. Processed {n_processed} slide(s); skipped {n_skipped} already-present slide(s).")
    if do_token:
        print(f"Saved token heatmaps under:                   {heatmaps_dir}")
    if do_patch_cell:
        print(f"Saved patch-cell tables under:               {patch_cell_dir}")



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute similarity analysis for STHELAR H&E embeddings using patch embeddings + per-(patch, cell) embeddings, with optional token-grid projection.")

    ap.add_argument("--training_dataset_id", type=str,
                    required=True,
                    help="Training dataset ID used for eval dataset construction, e.g. ds_ct1_20x_224_o112_1")
    
    ap.add_argument("--training_num", type=str,
                    required=True,
                    help="Training number, e.g. '46' for training_46")
    
    ap.add_argument("--checkpoint_num", type=str,
                    required=True,
                    help="Checkpoint epoch number, e.g. '13' for checkpoint_epoch_13.pth")

    ap.add_argument("--output_level", type=str,
                    default="token",
                    choices=["token", "patch_cell", "both"],
                    help=("Outputs to write: 'token' (token-grid heatmaps and metrics), 'patch_cell' "
                          "(per-(patch, cell) similarity table and patch summaries) or 'both'."))
    
    ap.add_argument("--ct_cat_id", type=str,
                    default="ct_1",
                    help=("Cell-type category id of data/annots/annot_dicts_{ct_cat_id}/ (e.g. ct_1)."))
    ap.add_argument("--celltype_axes", type=str, nargs="+",
                    default=["fine", "coarse"], choices=["fine", "coarse"],
                    help=("Cell-type axes to report (the first one is the primary axis, used for the unsuffixed columns)."))
    ap.add_argument("--coarse_mapping_version", type=str,
                    default="v1",
                    help=("Version of the coarse cell-type mapping data/annots/annot_dicts_{ct_cat_id}/"
                          "coarse_ct_mapping_{version}.json (required with 'coarse')."))
    ap.add_argument("--overwrite", action="store_true",
                    help=("Allow writing into an output folder produced by a different "
                          f"A0_ANALYSIS_VERSION (current: {A0_ANALYSIS_VERSION})."))
    ap.add_argument("--space", type=str,
                    default="he", choices=["he"],
                    help=("Embedding space of the patch-vs-cell similarity (only 'he': the 'clip' patch and "
                          "cell heads are independent)."))

    ap.add_argument("--similarity",type=str,
                    default="cosine",
                    choices=["cosine", "dot"],
                    help="Similarity used between patch embeddings and patch-cell embeddings. Default: cosine.")
    
    ap.add_argument("--tau_mode", type=str,
                    default="scale_free", choices=["scale_free", "fixed"],
                    help=("Scaling of the token-focus softmax logits: 'scale_free' (z-scored token scores) "
                          "or 'fixed' (scores / --tau)."))
    ap.add_argument("--tau_scale", type=float,
                    default=1.0,
                    help=("Sharpness of --tau_mode scale_free: logits are "
                          "(s - mean(s)) / (tau_scale * std(s)). Smaller = sharper focus."))
    ap.add_argument("--max_median_entropy_norm", type=float,
                    default=0.99,
                    help=("Raise after all slides if the median entropy_norm of the token-focus map exceeds this value."))
    ap.add_argument("--tau", type=float,
                    default=0.07,
                    help="Softmax temperature with --tau_mode fixed (used only when --output_level includes token).")
    
    ap.add_argument("--topk", type=int,
                    default=10,
                    help="Top-k mass metric computed on non-empty tokens (used only when --output_level includes token).")
    
    ap.add_argument("--heatmap_dtype", type=str,
                    default="fp16",
                    choices=["fp16", "fp32"],
                    help="Storage dtype for token heatmap datasets (used only when --output_level includes token).")
    
    ap.add_argument("--slides", nargs="*",
                    default=None,
                    help=("Optional subset of slide IDs to process (the combined tables still cover the "
                          "full STHELAR cohort)."))

    ap.add_argument("--resume", action="store_true",
                    help=("Skip slides whose per-slide outputs for the requested --output_level already "
                          "exist (the existing run_config.json must match the guarded parameters)."))

    ap.add_argument("--max_in_memory_gb", type=float,
                    default=6.0,
                    help=("Read a slide's per-(patch, cell) embedding matrix into RAM when it is at most "
                          "this many GB (float32); 0 = always read from disk."))

    # Reproducibility of the sampling-based diagnostics
    ap.add_argument("--seed", type=int,
                    default=42,
                    help="Random seed used for shuffled-baseline sampling (set_global_seed + rng).")

    # Intra-patch cell-cell cosine
    ap.add_argument("--no_intra_patch_cellcell", action="store_true",
                    help="Disable the intra-patch cell-cell cosine diagnostic.")

    # Shuffled patch-cell baseline
    ap.add_argument("--n_shuffled_pairs_per_group", type=int,
                    default=4,
                    help="Random shuffled (patch, cell) pairs per group for the shuffled baseline.")
    ap.add_argument("--no_shuffled_baseline", action="store_true",
                    help="Disable the shuffled patch-cell baseline.")

    # Cell-type abundance-corrected focus
    ap.add_argument("--no_celltype_advantage", action="store_true",
                    help="Disable the cell-type abundance-corrected focus columns.")

    # n_patch_cells stratification
    ap.add_argument("--headline_min_cells", type=int,
                    default=5,
                    help="Minimum n_patch_cells for the headline-only by-slide summary.")
    ap.add_argument("--no_stratified_aggregation", action="store_true",
                    help="Disable the n_patch_cells-stratified slide aggregation.")

    args = ap.parse_args()
    main(args)