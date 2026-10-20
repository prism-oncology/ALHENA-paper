#!/usr/bin/env python
"""
Analyze the similarity of the H&E cell embeddings (he, he_clip) of a same cell_id across overlapping patches.

Saved in <ROOT_HistoST> / eval / eval_datasets / st_eval_data / eval_st_<training_dataset_id> / embeddings_analysis / cell_view_consistency_HEembs / HEembs_t<training_num>e<checkpoint_num> /
    - run_config.json: analysis parameters
    - per_cell_metrics.parquet, per_view_metrics.parquet: metrics per cell and per (cell, patch) view
    - summary_global.csv, summary_by_{slide,n_views,cell_type,tissue}.csv: summaries of the per-cell metrics
    - alpha_bin_summary.csv, alpha_correlation_{global,by_slide}.csv: alpha weight vs leave-one-out cosine
    - random_pair_baseline_{by_slide,global}.csv: cosine between random views of different cells
    - representation_geometry.csv: anisotropy and effective rank per space
    - celltype_cosine_matrix.csv: mean cosine between cell types
    - plots/*.png
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import h5py
import numpy as np
import pandas as pd
from anndata import read_h5ad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.utils import set_global_seed, stable_seed
# Streaming parquet writer (sibling module a0_io.py)
from a0_io import StreamingParquetWriter  # noqa: E402

# Sibling module in this folder
from celltype_axis import (  # noqa: E402
    PAIR_SCOPE_POOLED,
    PAIR_SCOPE_WITHIN_SLIDE,
    PAIR_SCOPES,
    apply_label_map,
    apply_label_map_to_dict,
    assert_vocabulary_covered,
    build_axes,
)


# Output format version of this analysis (written into run_config.json, checked before writing into an existing folder)
A0_ANALYSIS_VERSION = 4


EPS = 1e-8

# Minimum number of other-cell views on a slide for the impostor leave-one-out cosine (NaN otherwise)
MIN_IMPOSTOR_POOL_SIZE = 10


# RNG stream tags, one independent stream per sampling diagnostic
SLIDE_STREAM_IMPOSTOR_LOO = 0
SLIDE_STREAM_RANDOM_BASELINE = 1
SLIDE_STREAM_ANISOTROPY = 2
SLIDE_STREAM_CELLTYPE_MATRIX = 3
GLOBAL_STREAM_ANISOTROPY = 0xA0
GLOBAL_STREAM_CELLTYPE_MATRIX = 0xA1
# Sub-tags of the cell-type matrix stream (cap draw and pair draw)
MATRIX_STREAM_CAP = 0
MATRIX_STREAM_PAIRS = 1


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def normalize_vector(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n <= eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / n).astype(np.float32, copy=False)



def normalize_rows(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(norms, eps, None)).astype(np.float32, copy=False)



def cosine_between(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= eps or nb <= eps:
        return 0.0
    val = float(np.dot(a, b) / (na * nb))
    return float(np.clip(val, -1.0, 1.0))



def safe_mean(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    return float(np.mean(x))



def safe_median(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    return float(np.median(x))



def safe_min(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    return float(np.min(x))



def normalized_entropy(weights: np.ndarray, eps: float = EPS) -> float:
    w = np.asarray(weights, dtype=np.float64)
    s = float(w.sum())
    if s <= eps or w.size <= 1:
        return 0.0
    p = w / s
    p = p[p > 0]
    h = -float(np.sum(p * np.log(p)))
    hmax = float(np.log(len(weights)))
    if hmax <= eps:
        return 0.0
    return h / hmax



def load_slide_to_tissue_map(metadata_csv: Optional[Path]) -> Dict[str, str]:
    if metadata_csv is None or (not metadata_csv.exists()):
        return {}

    meta = pd.read_csv(metadata_csv)
    slide_col = "slide_id"
    if slide_col is None:
        return {}

    tissue_col = "tissue_HistoST"
    if tissue_col is None:
        return {}

    out = (
        meta[[slide_col, tissue_col]]
        .dropna(subset=[slide_col])
        .astype({slide_col: str})
        .copy()
    )
    out[tissue_col] = out[tissue_col].fillna("UNKNOWN").astype(str)
    return dict(zip(out[slide_col].values, out[tissue_col].values))



def load_celltype_map_from_h5ad(slide_h5ad_path: Path) -> Dict[int, str]:
    """
    Load {cell_id_int -> cell_type} from STHELAR .h5ad cell rows.
    """
    if not slide_h5ad_path.exists():
        return {}

    ad = read_h5ad(slide_h5ad_path, backed="r")
    try:
        obs = ad.obs
        if "is_cell" not in obs.columns or "cell_id_int" not in obs.columns:
            return {}

        mask = obs["is_cell"].astype(bool).to_numpy()
        if not np.any(mask):
            return {}

        if "cell_type" in obs.columns:
            sub = obs.loc[mask, ["cell_id_int", "cell_type"]].copy()
            sub["cell_type"] = sub["cell_type"].astype("string").fillna("UNKNOWN")
        else:
            sub = obs.loc[mask, ["cell_id_int"]].copy()
            sub["cell_type"] = "UNKNOWN"

        sub["cell_id_int"] = sub["cell_id_int"].astype(np.int64)
        sub = sub.drop_duplicates(subset=["cell_id_int"], keep="first")
        return dict(zip(sub["cell_id_int"].tolist(), sub["cell_type"].tolist()))
    finally:
        if hasattr(ad, "file") and ad.file is not None:
            ad.file.close()



def load_spaces_from_embeddings_h5(
    h5_path: Path,
    embedding_spaces: List[str],
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]:
    """
    Returns:
      cell_id_int: (N,)
      alpha:       (N,)
      emb_by_space: dict[str, (N, D)]
    """
    dset_names = {
        "he": "patch_cell_emb_he",
        "he_clip": "patch_cell_emb_he_clip",
    }
    unknown = sorted(set(embedding_spaces) - set(dset_names))
    if unknown:
        raise ValueError(f"Unknown embedding space(s): {unknown}")

    with h5py.File(h5_path, "r") as f:
        required = ["patch_cell_cell_id_int", "patch_cell_alpha"]
        if any(name not in f for name in required):
            return None

        cell_id = f["patch_cell_cell_id_int"][:].astype(np.int64, copy=False)
        alpha = f["patch_cell_alpha"][:].astype(np.float32, copy=False)
        emb_by_space: Dict[str, np.ndarray] = {}
        for embedding_space in embedding_spaces:
            dset_name = dset_names[embedding_space]
            if dset_name not in f:
                continue

            emb = f[dset_name][:]
            if emb.ndim != 2:
                raise ValueError(f"{h5_path}::{dset_name} must be 2D, got shape={emb.shape}")

            if emb.shape[0] != cell_id.shape[0] or emb.shape[0] != alpha.shape[0]:
                raise ValueError(
                    f"Shape mismatch in {h5_path} for space='{embedding_space}': "
                    f"emb={emb.shape}, cell_id={cell_id.shape}, alpha={alpha.shape}"
                )

            if emb.shape[1] == 0:
                continue

            emb_by_space[embedding_space] = emb.astype(np.float32, copy=False)

    if cell_id.size == 0:
        return (cell_id, alpha, emb_by_space)

    return (cell_id, alpha, emb_by_space)



# ---------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------

def analyze_one_cell(
    cell_id_int: int,
    E: np.ndarray,
    A: np.ndarray,
    cell_type: str,
    slide_id: str,
    tissue: str,
    embedding_space: str,
    save_per_view: bool,
    impostor_pool_ehat: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict, List[Dict]]:
    """
    E: [n_views, D] raw embeddings
    A: [n_views] alpha weights (sum nuc_px in that patch for that cell)
    impostor_pool_ehat: optional [M, D] L2-normalized views of other cells of the same slide, for the
        impostor leave-one-out cosine (see `compute_impostor_loo_for_cell`)
    rng: numpy Generator of the impostor sampling (required with impostor_pool_ehat)
    """
    n_views, dim = E.shape
    if n_views < 2:
        raise ValueError("analyze_one_cell expects n_views >= 2")

    A = np.asarray(A, dtype=np.float32)
    E = np.asarray(E, dtype=np.float32)

    norms = np.linalg.norm(E, axis=1).astype(np.float32, copy=False)
    zero_norm_views = int(np.sum(norms <= EPS))
    Ehat = normalize_rows(E)
    norm_mean = float(np.mean(norms))
    norm_std = float(np.std(norms))
    norm_min = float(np.min(norms))
    norm_max = float(np.max(norms))

    # Pairwise metrics
    iu = np.triu_indices(n_views, k=1)
    pair_i, pair_j = iu
    pair_count = int(len(iu[0]))

    cos_mat = np.clip(Ehat @ Ehat.T, -1.0, 1.0)
    cos_pairs = cos_mat[iu]

    pair_norm_i = norms[pair_i]
    pair_norm_j = norms[pair_j]
    pair_sq_i = pair_norm_i * pair_norm_i
    pair_sq_j = pair_norm_j * pair_norm_j

    raw_l2_pairs = np.sqrt(
        np.maximum(pair_sq_i + pair_sq_j - 2.0 * pair_norm_i * pair_norm_j * cos_pairs, 0.0)
    ).astype(np.float32, copy=False)

    unit_l2_pairs = np.sqrt(np.maximum(2.0 - 2.0 * cos_pairs, 0.0)).astype(np.float32, copy=False)

    pair_norm_max = np.maximum(pair_norm_i, pair_norm_j)
    pair_norm_min = np.minimum(pair_norm_i, pair_norm_j)
    norm_ratio_pairs = (
        pair_norm_min / np.clip(pair_norm_max, EPS, None)
    ).astype(np.float32, copy=False)

    # Weighted / unweighted centroids
    alpha_sum = float(A.sum())
    alpha_max = float(A.max()) if A.size > 0 else 0.0
    alpha_min = float(A.min()) if A.size > 0 else 0.0
    alpha_mean = float(A.mean()) if A.size > 0 else 0.0
    alpha_min_over_max = float(alpha_min / max(alpha_max, EPS)) if A.size > 0 else np.nan
    alpha_entropy_norm = normalized_entropy(A)

    w = A / max(alpha_sum, EPS)

    raw_cent_alpha = np.sum(w[:, None] * E, axis=0)
    pre_cent_alpha = np.sum(w[:, None] * Ehat, axis=0)
    raw_cent_unw = np.mean(E, axis=0)
    pre_cent_unw = np.mean(Ehat, axis=0)

    raw_cent_alpha_u = normalize_vector(raw_cent_alpha)
    pre_cent_alpha_u = normalize_vector(pre_cent_alpha)
    raw_cent_unw_u = normalize_vector(raw_cent_unw)
    pre_cent_unw_u = normalize_vector(pre_cent_unw)

    # Direction concentration on normalized embeddings
    resultant_length_alpha = float(np.linalg.norm(pre_cent_alpha))
    resultant_length_unweighted = float(np.linalg.norm(pre_cent_unw))

    # "Total cancellation" ratio on raw embeddings
    raw_cancellation_ratio_alpha = float(
        np.linalg.norm(raw_cent_alpha) / max(float(np.sum(w * norms)), EPS)
    )
    raw_cancellation_ratio_unweighted = float(
        np.linalg.norm(raw_cent_unw) / max(norm_mean, EPS)
    )

    # Cosine to consensus directions
    cos_to_alpha_raw_centroid = np.clip(Ehat @ raw_cent_alpha_u, -1.0, 1.0)
    cos_to_alpha_prenorm_centroid = np.clip(Ehat @ pre_cent_alpha_u, -1.0, 1.0)
    cos_to_unweighted_raw_centroid = np.clip(Ehat @ raw_cent_unw_u, -1.0, 1.0)
    cos_to_unweighted_prenorm_centroid = np.clip(Ehat @ pre_cent_unw_u, -1.0, 1.0)

    # Merge sensitivity
    merge_cos_alpha_raw_vs_alpha_prenorm = cosine_between(raw_cent_alpha_u, pre_cent_alpha_u)
    merge_cos_unweighted_raw_vs_unweighted_prenorm = cosine_between(raw_cent_unw_u, pre_cent_unw_u)
    merge_cos_alpha_raw_vs_unweighted_raw = cosine_between(raw_cent_alpha_u, raw_cent_unw_u)
    merge_cos_alpha_prenorm_vs_unweighted_prenorm = cosine_between(pre_cent_alpha_u, pre_cent_unw_u)

    # Leave-one-out consensus
    weighted_E = A[:, None] * E
    weighted_Ehat = A[:, None] * Ehat
    sum_raw = np.sum(weighted_E, axis=0)
    sum_pre = np.sum(weighted_Ehat, axis=0)

    denom = (alpha_sum - A).astype(np.float32, copy=False)
    valid_loo = denom > EPS
    loo_cos_raw_arr = np.full(n_views, np.nan, dtype=np.float32)
    loo_cos_prenorm_arr = np.full(n_views, np.nan, dtype=np.float32)
    per_view_rows = []

    if np.any(valid_loo):
        loo_raw_cent = np.zeros_like(E, dtype=np.float32)
        loo_pre_cent = np.zeros_like(Ehat, dtype=np.float32)

        loo_raw_cent[valid_loo] = normalize_rows(
            (sum_raw - weighted_E)[valid_loo] / denom[valid_loo, None]
        )
        loo_pre_cent[valid_loo] = normalize_rows(
            (sum_pre - weighted_Ehat)[valid_loo] / denom[valid_loo, None]
        )

        loo_cos_raw_arr[valid_loo] = np.clip(np.sum(Ehat[valid_loo] * loo_raw_cent[valid_loo], axis=1), -1.0, 1.0)
        loo_cos_prenorm_arr[valid_loo] = np.clip(np.sum(Ehat[valid_loo] * loo_pre_cent[valid_loo], axis=1), -1.0, 1.0)

    # Per-view impostor LOO: the other views of the LOO consensus are replaced by random views of other cells
    loo_cos_prenorm_impostor_arr = np.full(n_views, np.nan, dtype=np.float32)
    if (
        impostor_pool_ehat is not None
        and rng is not None
        and impostor_pool_ehat.ndim == 2
        and impostor_pool_ehat.shape[0] >= MIN_IMPOSTOR_POOL_SIZE
    ):
        loo_cos_prenorm_impostor_arr = compute_impostor_loo_for_cell(
            Ehat=Ehat,
            A=A,
            impostor_pool_ehat=impostor_pool_ehat,
            rng=rng,
        )

    if save_per_view:
        for i in range(n_views):
            per_view_rows.append(
                {
                    "embedding_space": embedding_space,
                    "slide_id": slide_id,
                    "tissue": tissue,
                    "cell_id_int": int(cell_id_int),
                    "cell_type": str(cell_type),
                    "n_views": int(n_views),
                    "alpha": float(A[i]),
                    "alpha_frac_sum": float(A[i] / max(alpha_sum, EPS)),
                    "alpha_frac_max": float(A[i] / max(alpha_max, EPS)),
                    "view_norm": float(norms[i]),
                    "zero_norm_view": bool(norms[i] <= EPS),
                    "cos_to_alpha_raw_centroid": float(cos_to_alpha_raw_centroid[i]),
                    "cos_to_alpha_prenorm_centroid": float(cos_to_alpha_prenorm_centroid[i]),
                    "cos_to_unweighted_raw_centroid": float(cos_to_unweighted_raw_centroid[i]),
                    "cos_to_unweighted_prenorm_centroid": float(cos_to_unweighted_prenorm_centroid[i]),
                    "loo_cos_raw": float(loo_cos_raw_arr[i]),
                    "loo_cos_prenorm": float(loo_cos_prenorm_arr[i]),
                    "loo_cos_prenorm_impostor": float(loo_cos_prenorm_impostor_arr[i]),
                }
            )

    cell_row = {
        "embedding_space": embedding_space,
        "slide_id": slide_id,
        "tissue": tissue,
        "cell_id_int": int(cell_id_int),
        "cell_type": str(cell_type),
        "n_views": int(n_views),
        "pair_count": int(pair_count),
        "embedding_dim": int(dim),

        # alpha / coverage
        "alpha_sum": alpha_sum,
        "alpha_mean": alpha_mean,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "alpha_min_over_max": alpha_min_over_max,
        "alpha_entropy_norm": alpha_entropy_norm,

        # norm stability
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "norm_cv": float(norm_std / max(norm_mean, EPS)),
        "norm_max_over_min": float(norm_max / max(norm_min, EPS)),
        "pair_norm_ratio_mean": safe_mean(norm_ratio_pairs),
        "pair_norm_ratio_median": safe_median(norm_ratio_pairs),
        "pair_norm_ratio_min": safe_min(norm_ratio_pairs),
        "zero_norm_views": zero_norm_views,

        # direction consistency
        "pair_cos_mean": safe_mean(cos_pairs),
        "pair_cos_median": safe_median(cos_pairs),
        "pair_cos_min": safe_min(cos_pairs),
        "resultant_length_alpha": resultant_length_alpha,
        "resultant_length_unweighted": resultant_length_unweighted,
        "mean_cos_to_alpha_raw_centroid": safe_mean(cos_to_alpha_raw_centroid),
        "mean_cos_to_alpha_prenorm_centroid": safe_mean(cos_to_alpha_prenorm_centroid),
        "mean_cos_to_unweighted_raw_centroid": safe_mean(cos_to_unweighted_raw_centroid),
        "mean_cos_to_unweighted_prenorm_centroid": safe_mean(cos_to_unweighted_prenorm_centroid),
        "mean_loo_cos_raw": safe_mean(loo_cos_raw_arr),
        "mean_loo_cos_prenorm": safe_mean(loo_cos_prenorm_arr),

        # discriminability vs random impostor LOO
        "mean_loo_cos_prenorm_impostor": safe_mean(loo_cos_prenorm_impostor_arr),
        "loo_discriminability": (
            safe_mean(loo_cos_prenorm_arr) - safe_mean(loo_cos_prenorm_impostor_arr)
            if np.isfinite(safe_mean(loo_cos_prenorm_arr))
            and np.isfinite(safe_mean(loo_cos_prenorm_impostor_arr))
            else np.nan
        ),

        # total discrepancy
        "pair_l2_raw_mean": safe_mean(raw_l2_pairs),
        "pair_l2_unit_mean": safe_mean(unit_l2_pairs),
        "raw_cancellation_ratio_alpha": raw_cancellation_ratio_alpha,
        "raw_cancellation_ratio_unweighted": raw_cancellation_ratio_unweighted,

        # merge sensitivity
        "merge_cos_alpha_raw_vs_alpha_prenorm": merge_cos_alpha_raw_vs_alpha_prenorm,
        "merge_cos_unweighted_raw_vs_unweighted_prenorm": merge_cos_unweighted_raw_vs_unweighted_prenorm,
        "merge_cos_alpha_raw_vs_unweighted_raw": merge_cos_alpha_raw_vs_unweighted_raw,
        "merge_cos_alpha_prenorm_vs_unweighted_prenorm": merge_cos_alpha_prenorm_vs_unweighted_prenorm,
    }

    return cell_row, per_view_rows



def attach_secondary_axis_columns(
    frame: pd.DataFrame,
    celltype_map_by_axis: Dict[str, Dict[int, str]],
    primary_axis: str,
) -> pd.DataFrame:
    """Add one `cell_type_{axis}` column per non-primary cell-type axis, from its `{cell_id_int: label}` map of the slide."""
    if frame.empty or "cell_id_int" not in frame.columns:
        return frame
    ids = frame["cell_id_int"].astype(np.int64)
    for axis, axis_map in celltype_map_by_axis.items():
        if axis == primary_axis:
            continue
        # `UNKNOWN`: no annotation row for this cell
        frame[f"cell_type_{axis}"] = ids.map(axis_map).fillna("UNKNOWN").astype(str)
    return frame


def analyze_one_slide_space(
    slide_id: str,
    tissue: str,
    embedding_space: str,
    cell_id: np.ndarray,
    alpha: np.ndarray,
    emb: np.ndarray,
    celltype_map: Dict[int, str],
    min_views: int,
    save_per_view: bool,
    order: Optional[np.ndarray] = None,
    starts: Optional[np.ndarray] = None,
    ends: Optional[np.ndarray] = None,
    compute_impostor_loo: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze one slide for one embedding space.
    With `compute_impostor_loo` and `rng`, the impostor LOO uses a slide-wide pool of L2-normalized views.
    """
    if cell_id.size == 0:
        return pd.DataFrame(), pd.DataFrame()

    if order is None:
        order = np.argsort(cell_id, kind="mergesort")
    cell_id = cell_id[order]
    alpha = alpha[order]
    emb = emb[order]

    if starts is None or ends is None:
        change = np.r_[True, cell_id[1:] != cell_id[:-1], True]
        bounds = np.flatnonzero(change)
        starts = bounds[:-1]
        ends = bounds[1:]

    # Slide-wide L2-normalized pool for the impostor LOO
    if compute_impostor_loo and rng is not None and emb.shape[0] >= MIN_IMPOSTOR_POOL_SIZE:
        slide_pool_ehat: Optional[np.ndarray] = normalize_rows(emb)
    else:
        slide_pool_ehat = None

    cell_rows: List[Dict] = []
    view_rows: List[Dict] = []

    for s, e in zip(starts, ends):
        cid = int(cell_id[s])
        n_views = int(e - s)
        if n_views < min_views:
            continue

        E = emb[s:e]
        A = alpha[s:e]
        cell_type = celltype_map.get(cid, "UNKNOWN")

        row_cell, rows_view = analyze_one_cell(
            cell_id_int=cid,
            E=E,
            A=A,
            cell_type=cell_type,
            slide_id=slide_id,
            tissue=tissue,
            embedding_space=embedding_space,
            save_per_view=save_per_view,
            impostor_pool_ehat=slide_pool_ehat,
            rng=rng,
        )
        cell_rows.append(row_cell)
        if save_per_view:
            view_rows.extend(rows_view)

    df_cell = pd.DataFrame(cell_rows)
    df_view = pd.DataFrame(view_rows) if save_per_view else pd.DataFrame()

    return df_cell, df_view



# ---------------------------------------------------------------------
# Discriminability / collapse helpers
# ---------------------------------------------------------------------

def compute_impostor_loo_for_cell(
    Ehat: np.ndarray,
    A: np.ndarray,
    impostor_pool_ehat: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Per-view impostor leave-one-out cosine of a cell: for view i, the n_views - 1 other views of the
    alpha-weighted LOO consensus are replaced by random rows of `impostor_pool_ehat` (same alpha weights).
    """
    n_views, dim = Ehat.shape
    out = np.full(n_views, np.nan, dtype=np.float32)
    if n_views < 2 or impostor_pool_ehat.shape[0] < (n_views - 1):
        return out

    pool_size = impostor_pool_ehat.shape[0]
    for i in range(n_views):
        idx = rng.integers(0, pool_size, size=n_views - 1)
        imp_views = impostor_pool_ehat[idx]               # (n_views-1, D)
        A_others = np.delete(A, i).astype(np.float32)     # (n_views-1,)
        denom = float(A_others.sum())
        if denom <= EPS:
            continue
        consensus = np.sum((A_others[:, None] * imp_views), axis=0) / denom
        consensus_u = normalize_vector(consensus)
        out[i] = float(np.clip(np.dot(Ehat[i], consensus_u), -1.0, 1.0))
    return out



def compute_random_pair_baseline(
    emb: np.ndarray,
    cell_id: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Mean / median / p10 / p90 cosine between random pairs of rows whose
    `cell_id_int` differs. Returns NaN-valued summary when fewer than two
    distinct cells are present.
    """
    out = {
        "n_rows": int(emb.shape[0]),
        "n_pairs": 0,
        "cos_mean": np.nan,
        "cos_median": np.nan,
        "cos_p10": np.nan,
        "cos_p90": np.nan,
    }
    n = int(emb.shape[0])
    if n < 2 or np.unique(cell_id).size < 2:
        return out

    Ehat = normalize_rows(emb)
    # Sample pairs (i, j) with replacement and reject within-cell pairs.
    target = int(n_pairs)
    accepted_cos: List[float] = []
    # Sample in batches for efficiency.
    batch = max(target, 1024)
    while len(accepted_cos) < target:
        ii = rng.integers(0, n, size=batch)
        jj = rng.integers(0, n, size=batch)
        keep = cell_id[ii] != cell_id[jj]
        if not np.any(keep):
            continue
        cos_vals = np.sum(Ehat[ii[keep]] * Ehat[jj[keep]], axis=1)
        accepted_cos.extend(np.clip(cos_vals, -1.0, 1.0).tolist())
        if len(accepted_cos) >= target * 4:
            # Safety stop: should never happen unless target is tiny.
            break

    arr = np.asarray(accepted_cos[:target], dtype=np.float64)
    if arr.size == 0:
        return out

    out["n_pairs"] = int(arr.size)
    out["cos_mean"] = float(np.mean(arr))
    out["cos_median"] = float(np.median(arr))
    out["cos_p10"] = float(np.quantile(arr, 0.10))
    out["cos_p90"] = float(np.quantile(arr, 0.90))
    return out



def compute_anisotropy(emb: np.ndarray) -> Dict[str, float]:
    """
    Anisotropy diagnostic of (a sample of) the patch-cell embedding matrix, by uncentered SVD:
    top1_var_fraction = sigma_1^2 / sum(sigma_i^2), top5_var_fraction, effective_rank = exp(H(sigma_i^2 / sum sigma_i^2)).
    """
    out = {
        "n_rows_sampled": int(emb.shape[0]),
        "embedding_dim": int(emb.shape[1]) if emb.ndim == 2 else 0,
        "top1_var_fraction": np.nan,
        "top5_var_fraction": np.nan,
        "effective_rank": np.nan,
        "mean_random_cosine": np.nan,
    }
    if emb.ndim != 2 or emb.shape[0] < 2 or emb.shape[1] < 1:
        return out

    X = emb.astype(np.float32, copy=False)
    try:
        # np.linalg.svd returns singular values in descending order.
        s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        return out

    s2 = (s.astype(np.float64) ** 2)
    total = float(s2.sum())
    if total <= EPS:
        return out

    p = s2 / total
    p_pos = p[p > 0]
    H = float(-np.sum(p_pos * np.log(p_pos))) if p_pos.size else 0.0
    out["top1_var_fraction"] = float(p[0])
    out["top5_var_fraction"] = float(p[: min(5, p.size)].sum())
    out["effective_rank"] = float(np.exp(H))
    return out



def compute_mean_random_cosine(
    emb: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
) -> float:
    """Mean cosine between random pairs of rows (no cell-id filter)."""
    n = int(emb.shape[0])
    if n < 2:
        return float("nan")
    Ehat = normalize_rows(emb)
    target = int(n_pairs)
    ii = rng.integers(0, n, size=target)
    jj = rng.integers(0, n, size=target)
    mask = ii != jj
    if not np.any(mask):
        return float("nan")
    vals = np.sum(Ehat[ii[mask]] * Ehat[jj[mask]], axis=1)
    return float(np.clip(np.mean(vals), -1.0, 1.0))



def compute_celltype_cosine_matrix(
    emb: np.ndarray,
    cell_id: np.ndarray,
    celltype_per_row: np.ndarray,
    min_cells_per_type: int,
    n_pairs_per_type_pair: int,
    cap_rng: np.random.Generator,
    pair_rng: np.random.Generator,
    max_rows_per_type: Optional[int] = None,
    slide_per_row: Optional[np.ndarray] = None,
    pair_scope: str = PAIR_SCOPE_POOLED,
) -> pd.DataFrame:
    """
    Long-format K x K mean-cosine matrix between cell types, for one pair scope: for each (t1, t2), cosine of
    random (row in t1, row in t2) pairs, drawn from the cross-slide pool (`pooled`) or within one slide (`within_slide`).
    Each type's pool is capped at `max_rows_per_type` (`cap_rng`); pairs are drawn with `pair_rng`.
    """
    if pair_scope not in PAIR_SCOPES:
        raise ValueError(f"Unknown pair_scope={pair_scope!r}. Expected one of {list(PAIR_SCOPES)}.")
    if pair_scope == PAIR_SCOPE_WITHIN_SLIDE and slide_per_row is None:
        raise ValueError(
            "pair_scope='within_slide' requires slide_per_row."
        )
    n = int(emb.shape[0])
    if n < 2 or celltype_per_row.size != n:
        return pd.DataFrame()

    Ehat = normalize_rows(emb)
    types, counts = np.unique(celltype_per_row, return_counts=True)
    keep = counts >= int(min_cells_per_type)
    types_ok = [str(t) for t, k in zip(types, keep) if k]
    if len(types_ok) < 2:
        return pd.DataFrame()

    idx_by_type: Dict[str, np.ndarray] = {
        t: np.flatnonzero(celltype_per_row == t) for t in types_ok
    }

    # Class size and slide concentration, on the full per-type pool (before the cap)
    n_rows_by_type: Dict[str, int] = {t: int(pool.size) for t, pool in idx_by_type.items()}
    slide_stats: Dict[str, Dict[str, float]] = {}
    if slide_per_row is not None and len(slide_per_row) == n:
        for t, pool in idx_by_type.items():
            slides_t = np.asarray(slide_per_row, dtype=object)[pool]
            uniq, cnt = np.unique(slides_t, return_counts=True)
            slide_stats[t] = {
                "n_slides": int(uniq.size),
                "max_slide_share": float(cnt.max() / cnt.sum()) if cnt.size else np.nan,
            }

    if max_rows_per_type is not None and int(max_rows_per_type) > 0:
        cap = int(max_rows_per_type)
        for t, pool in idx_by_type.items():
            if pool.size > cap:
                # Same capped pool for both pair scopes
                idx_by_type[t] = np.sort(cap_rng.choice(pool, size=cap, replace=False))

    # For `within_slide`: each type's (capped) pool, split by slide
    slide_arr = (
        np.asarray(slide_per_row, dtype=object)
        if (slide_per_row is not None and len(slide_per_row) == n) else None
    )
    pool_by_type_slide: Dict[str, Dict[object, np.ndarray]] = {}
    if pair_scope == PAIR_SCOPE_WITHIN_SLIDE:
        if slide_arr is None:
            raise ValueError(
                "pair_scope='within_slide' needs a slide_per_row array of length n_rows "
                f"(got {None if slide_per_row is None else len(slide_per_row)} vs {n})."
            )
        for t, pool in idx_by_type.items():
            by_slide: Dict[object, np.ndarray] = {}
            slides_t = slide_arr[pool]
            order_t = np.argsort(slides_t.astype(str), kind="mergesort")
            pool_sorted = pool[order_t]
            slides_sorted = slides_t[order_t].astype(str)
            change = np.r_[True, slides_sorted[1:] != slides_sorted[:-1], True]
            bounds = np.flatnonzero(change)
            for s, e in zip(bounds[:-1], bounds[1:]):
                by_slide[slides_sorted[s]] = pool_sorted[s:e]
            pool_by_type_slide[t] = by_slide

    rows: List[Dict] = []
    for t1 in types_ok:
        for t2 in types_ok:
            i_pool = idx_by_type[t1]
            j_pool = idx_by_type[t2]
            target = int(n_pairs_per_type_pair)
            i = pair_rng.choice(i_pool, size=target, replace=True)
            if pair_scope == PAIR_SCOPE_POOLED:
                j = pair_rng.choice(j_pool, size=target, replace=True)
                mask = i != j
            else:
                # `i` drawn as in `pooled`; `j` drawn from t2's rows on i's slide (j = -1 if none, dropped)
                j = np.full(target, -1, dtype=np.int64)
                i_slides = slide_arr[i].astype(str)
                t2_by_slide = pool_by_type_slide[t2]
                for slide_value in np.unique(i_slides):
                    partner_pool = t2_by_slide.get(slide_value)
                    if partner_pool is None or partner_pool.size == 0:
                        continue
                    local = np.flatnonzero(i_slides == slide_value)
                    j[local] = pair_rng.choice(partner_pool, size=local.size, replace=True)
                mask = (j >= 0) & (i != j)
            if t1 == t2:
                # Same type: exclude pairs of views of the same cell
                mask &= cell_id[i] != cell_id[np.maximum(j, 0)]
            stats1 = slide_stats.get(t1, {})
            base_row = {
                "pair_scope": pair_scope,
                "cell_type_1": t1,
                "cell_type_2": t2,
                "cell_type_1_n_rows": n_rows_by_type.get(t1, np.nan),
                "cell_type_1_n_slides": stats1.get("n_slides", np.nan),
                "cell_type_1_max_slide_share": stats1.get("max_slide_share", np.nan),
            }
            if not np.any(mask):
                rows.append({**base_row, "n_pairs": 0, "n_pair_slides": 0,
                             "cos_mean": np.nan, "cos_median": np.nan})
                continue
            vals = np.clip(np.sum(Ehat[i[mask]] * Ehat[j[mask]], axis=1), -1.0, 1.0)
            # Number of slides contributing queries to this (t1, t2) entry
            n_pair_slides = (
                int(np.unique(slide_arr[i[mask]].astype(str)).size) if slide_arr is not None else np.nan
            )
            rows.append({
                **base_row,
                "n_pairs": int(vals.size),
                "n_pair_slides": n_pair_slides,
                "cos_mean": float(np.mean(vals)),
                "cos_median": float(np.median(vals)),
            })

    return pd.DataFrame(rows)



# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

SUMMARY_NUMERIC_COLS = [
    "n_views",
    "alpha_min_over_max",
    "alpha_entropy_norm",
    "norm_cv",
    "norm_max_over_min",
    "pair_norm_ratio_mean",
    "pair_cos_mean",
    "pair_cos_min",
    "resultant_length_alpha",
    "resultant_length_unweighted",
    "mean_cos_to_alpha_prenorm_centroid",
    "mean_loo_cos_prenorm",
    "mean_loo_cos_prenorm_impostor",
    "loo_discriminability",
    "pair_l2_raw_mean",
    "pair_l2_unit_mean",
    "raw_cancellation_ratio_alpha",
    "merge_cos_alpha_raw_vs_alpha_prenorm",
    "merge_cos_unweighted_raw_vs_unweighted_prenorm",
    "merge_cos_alpha_raw_vs_unweighted_raw",
    "merge_cos_alpha_prenorm_vs_unweighted_prenorm",
]


def summarize_per_cell(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        rec = {col: val for col, val in zip(group_cols, key)}
        rec["n_cells"] = int(len(sub))
        rec["n_unique_slides"] = int(sub["slide_id"].nunique()) if "slide_id" in sub.columns else np.nan
        # Fraction of this group's cells on its most common slide
        if "slide_id" in sub.columns and len(sub):
            counts = sub["slide_id"].value_counts()
            rec["max_slide_share"] = float(counts.iloc[0] / counts.sum())
        else:
            rec["max_slide_share"] = np.nan

        for col in SUMMARY_NUMERIC_COLS:
            arr = sub[col].dropna().to_numpy(dtype=np.float64)
            rec[f"{col}_mean"] = float(np.mean(arr)) if arr.size > 0 else np.nan
            rec[f"{col}_median"] = float(np.median(arr)) if arr.size > 0 else np.nan
            rec[f"{col}_p10"] = float(np.quantile(arr, 0.10)) if arr.size > 0 else np.nan
            rec[f"{col}_p90"] = float(np.quantile(arr, 0.90)) if arr.size > 0 else np.nan

        rows.append(rec)

    out = pd.DataFrame(rows)
    sort_cols = [c for c in group_cols if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out



def build_alpha_bin_summary(df_view: pd.DataFrame) -> pd.DataFrame:
    if df_view.empty:
        return pd.DataFrame()

    bins = [0.0, 0.25, 0.50, 0.75, 0.90, 1.000001]
    labels = ["[0,0.25]", "(0.25,0.50]", "(0.50,0.75]", "(0.75,0.90]", "(0.90,1.00]"]

    x = df_view.copy()
    x["alpha_frac_max_bin"] = pd.cut(
        x["alpha_frac_max"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    rows = []
    for key, sub in x.groupby(["embedding_space", "alpha_frac_max_bin"], dropna=False):
        emb_space, alpha_bin = key
        arr_raw = sub["loo_cos_raw"].dropna().to_numpy(dtype=np.float64)
        arr_pre = sub["loo_cos_prenorm"].dropna().to_numpy(dtype=np.float64)

        rows.append(
            {
                "embedding_space": emb_space,
                "alpha_frac_max_bin": str(alpha_bin),
                "n_views": int(len(sub)),
                "loo_cos_raw_mean": float(np.mean(arr_raw)) if arr_raw.size > 0 else np.nan,
                "loo_cos_raw_median": float(np.median(arr_raw)) if arr_raw.size > 0 else np.nan,
                "loo_cos_prenorm_mean": float(np.mean(arr_pre)) if arr_pre.size > 0 else np.nan,
                "loo_cos_prenorm_median": float(np.median(arr_pre)) if arr_pre.size > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)



def correlation_table(
    df: pd.DataFrame,
    group_cols: List[str],
    x_col: str,
    y_col: str,
    min_rows: int,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        sub2 = sub[[x_col, y_col]].dropna()
        if len(sub2) < min_rows:
            continue

        rec = {col: val for col, val in zip(group_cols, key)}
        rec["n_rows"] = int(len(sub2))
        rec["pearson"] = float(sub2[x_col].corr(sub2[y_col], method="pearson"))
        rec["spearman"] = float(sub2[x_col].corr(sub2[y_col], method="spearman"))
        rows.append(rec)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(group_cols).reset_index(drop=True)
    return out



# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_histograms(per_cell: pd.DataFrame, plot_dir: Path) -> None:
    if per_cell.empty:
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("pair_cos_mean", "Per-cell mean pairwise cosine"),
        ("norm_cv", "Per-cell norm CV"),
        ("merge_cos_alpha_raw_vs_alpha_prenorm", "Cosine between alpha-weighted raw merge and alpha-weighted pre-normalized merge"),
        ("resultant_length_alpha", "Weighted resultant length on normalized views"),
        ("mean_loo_cos_prenorm", "Mean leave-one-out cosine (pre-normalized consensus)"),
    ]

    for emb_space in sorted(per_cell["embedding_space"].dropna().unique()):
        sub = per_cell[per_cell["embedding_space"] == emb_space].copy()
        if sub.empty:
            continue

        for col, xlabel in metrics:
            vals = sub[col].dropna().to_numpy(dtype=np.float64)
            if vals.size == 0:
                continue

            plt.figure(figsize=(6.0, 4.2))
            plt.hist(vals, bins=50)
            plt.xlabel(xlabel)
            plt.ylabel("Number of cells")
            plt.title(f"{emb_space} - {col}")
            plt.tight_layout()
            plt.savefig(plot_dir / f"{emb_space}_{col}.png", dpi=200)
            plt.close()



def plot_celltype_cosine_matrix(
    matrix_df: pd.DataFrame,
    plot_dir: Path,
) -> None:
    """Plot one K x K mean-cosine heatmap per (embedding space, cell-type axis, pair scope)."""
    if matrix_df.empty:
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    axes_present = (
        sorted(matrix_df["celltype_axis"].dropna().unique())
        if "celltype_axis" in matrix_df.columns else [None]
    )
    scopes_present = (
        sorted(matrix_df["pair_scope"].dropna().unique())
        if "pair_scope" in matrix_df.columns else [None]
    )
    for emb_space in sorted(matrix_df["embedding_space"].dropna().unique()):
      for ct_axis in axes_present:
       for scope in scopes_present:
        sub = matrix_df[matrix_df["embedding_space"] == emb_space].copy()
        if ct_axis is not None:
            sub = sub[sub["celltype_axis"] == ct_axis].copy()
        if scope is not None:
            sub = sub[sub["pair_scope"] == scope].copy()
        if sub.empty:
            continue
        types = sorted(set(sub["cell_type_1"].astype(str)).union(set(sub["cell_type_2"].astype(str))))
        if len(types) < 2:
            continue

        idx = {t: i for i, t in enumerate(types)}
        K = len(types)
        M = np.full((K, K), np.nan, dtype=np.float32)
        for _, row in sub.iterrows():
            i = idx.get(str(row["cell_type_1"]))
            j = idx.get(str(row["cell_type_2"]))
            if i is None or j is None:
                continue
            v = float(row["cos_mean"]) if pd.notna(row["cos_mean"]) else np.nan
            M[i, j] = v

        finite = M[np.isfinite(M)]
        if finite.size == 0:
            continue
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1e-6

        fig_size = max(4.5, 0.35 * K + 2.0)
        plt.figure(figsize=(fig_size, fig_size))
        im = plt.imshow(M, vmin=vmin, vmax=vmax, cmap="viridis")
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Mean cosine")
        plt.xticks(range(K), types, rotation=45, ha="right", fontsize=8)
        plt.yticks(range(K), types, fontsize=8)
        axis_label = "" if ct_axis is None else f" [{ct_axis}]"
        scope_label = "" if scope is None else f" ({scope} pairs)"
        plt.title(f"{emb_space}{axis_label}{scope_label} - cell-type cosine matrix")
        plt.tight_layout()
        # Axis and scope in the filename
        suffix = "" if ct_axis is None else f"_{ct_axis}"
        suffix += "" if scope is None else f"_{scope}"
        plt.savefig(plot_dir / f"{emb_space}{suffix}_celltype_cosine_matrix.png", dpi=200)
        plt.close()



def plot_alpha_bin_summary(alpha_bin_summary: pd.DataFrame, plot_dir: Path) -> None:
    if alpha_bin_summary.empty:
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    order = ["[0,0.25]", "(0.25,0.50]", "(0.50,0.75]", "(0.75,0.90]", "(0.90,1.00]"]

    for emb_space in sorted(alpha_bin_summary["embedding_space"].dropna().unique()):
        sub = alpha_bin_summary[alpha_bin_summary["embedding_space"] == emb_space].copy()
        if sub.empty:
            continue

        sub["alpha_frac_max_bin"] = pd.Categorical(sub["alpha_frac_max_bin"], categories=order, ordered=True)
        sub = sub.sort_values("alpha_frac_max_bin")

        plt.figure(figsize=(6.5, 4.2))
        plt.plot(sub["alpha_frac_max_bin"].astype(str), sub["loo_cos_prenorm_mean"].to_numpy(), marker="o")
        plt.xlabel("alpha_frac_max bin")
        plt.ylabel("Mean leave-one-out cosine")
        plt.title(f"{emb_space} - alpha support vs consensus")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{emb_space}_alpha_bin_vs_loo_cos_prenorm.png", dpi=200)
        plt.close()



# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(args) -> None:

    # Reproducibility
    set_global_seed(args.seed)

    # Independent RNG stream per key (e.g. slide, space, diagnostic)
    def stream_rng(*key: int) -> np.random.Generator:
        return np.random.default_rng([int(args.seed), *(int(k) for k in key)])

    # Data paths
    eval_root = get_full_path("eval", "eval_datasets", "st_eval_data", f"eval_st_{args.training_dataset_id}")
    st_dir = eval_root / "st"
    embeddings_dir = eval_root / "he_embeddings" / f"HEembs_t{args.training_num}e{args.checkpoint_num}"

    if not st_dir.exists():
        raise FileNotFoundError(f"ST directory not found: {st_dir}")
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

    metadata_csv = get_full_path("eval", "eval_datasets", "st_eval_data", f"eval_st_{args.training_dataset_id}", f"metadata_eval_st_{args.training_dataset_id}.csv")
    if not metadata_csv.exists():
        metadata_csv = None

    # Load slide->tissue mapping
    slide_to_tissue = load_slide_to_tissue_map(metadata_csv)

    # Output paths
    out_dir = get_full_path("eval", "eval_datasets", "st_eval_data", f"eval_st_{args.training_dataset_id}", "embeddings_analysis", f"cell_view_consistency_HEembs", f"HEembs_t{args.training_num}e{args.checkpoint_num}")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the cell-type axes once, so a bad --coarse_mapping_version fails before any slide.
    celltype_axes = build_axes(args.ct_cat_id, args.celltype_axes, args.coarse_mapping_version)
    primary_axis = next(iter(celltype_axes))

    # Refuse to write into a folder written by another analysis version (unless --overwrite)
    run_config_path = out_dir / "run_config.json"
    if run_config_path.exists():
        try:
            prev_version = int(json.loads(run_config_path.read_text()).get("a0_analysis_version", 1))
        except (json.JSONDecodeError, TypeError, ValueError):
            prev_version = 1
        if prev_version != A0_ANALYSIS_VERSION and not args.overwrite:
            raise RuntimeError(
                f"{out_dir} was written by A0_ANALYSIS_VERSION={prev_version}, but this is "
                f"version {A0_ANALYSIS_VERSION}. Re-run into a clean directory, or pass "
                "--overwrite to replace it."
            )
    run_config_path.write_text(json.dumps({
        "a0_analysis_version": int(A0_ANALYSIS_VERSION),
        "training_dataset_id": args.training_dataset_id,
        "training_num": args.training_num,
        "checkpoint_num": args.checkpoint_num,
        "spaces": list(args.spaces),
        "ct_cat_id": args.ct_cat_id,
        "celltype_axes": list(args.celltype_axes),
        "coarse_mapping_version": args.coarse_mapping_version,
        "primary_celltype_axis": primary_axis,
        "seed": int(args.seed),
        "min_views": int(args.min_views),
        "min_cells_per_group": int(args.min_cells_per_group),
    }, indent=2, sort_keys=True))

    # Find embedding files (hidden ._* files skipped)
    emb_files = sorted(
        p for p in embeddings_dir.glob("*.h5") if not p.name.startswith(".")
    )
    if not emb_files:
        raise FileNotFoundError(f"No slide embedding files found in {embeddings_dir}")

    # Check the cell-type vocabulary of every slide before the main loop
    print("Preflight: checking the cell-type vocabulary against every requested axis...")
    observed_labels: set = set()
    for emb_h5 in emb_files:
        observed_labels.update(
            load_celltype_map_from_h5ad(st_dir / f"{emb_h5.stem}.h5ad").values()
        )
    assert_vocabulary_covered(
        observed_labels, celltype_axes, where=f"{len(emb_files)} slides under {st_dir}"
    )

    # Detailed tables, streamed to parquet slide by slide
    per_cell_writer = StreamingParquetWriter(out_dir / "per_cell_metrics.parquet")
    per_view_writer = StreamingParquetWriter(
        out_dir / "per_view_metrics.parquet", enabled=(not args.no_per_view)
    )

    # Accumulators of the random-pair baseline, anisotropy and cell-type matrix (filled per slide)
    random_baseline_rows: List[Dict] = []
    anisotropy_sample_by_space: Dict[str, List[np.ndarray]] = {}
    celltype_sample_by_space: Dict[str, Dict[str, List[np.ndarray]]] = {}
    # For each (space, cell_type): lists of emb / cell_id / slide chunks.

    print(f"Found {len(emb_files)} embedding slide file(s) in: {embeddings_dir}")
    print(f"Saving results to: {out_dir}")

    # Process each slide
    for i, emb_h5 in enumerate(emb_files, start=1):
        
        slide_id = emb_h5.stem
        tissue = slide_to_tissue.get(slide_id, "UNKNOWN")
        slide_h5ad = st_dir / f"{slide_id}.h5ad"

        print(f"\n[{i}/{len(emb_files)}] Slide: {slide_id}")

        celltype_map_fine = load_celltype_map_from_h5ad(slide_h5ad)
        # per_cell["cell_type"] uses the primary (first requested) axis
        celltype_map_by_axis = {
            axis: apply_label_map_to_dict(celltype_map_fine, mapping)
            for axis, mapping in celltype_axes.items()
        }
        celltype_map = celltype_map_by_axis[primary_axis]
        if len(celltype_map) == 0:
            print("  No STHELAR cell rows / cell_type map found in .h5ad; cell_type will be UNKNOWN.")

        loaded = load_spaces_from_embeddings_h5(emb_h5, args.spaces)
        if loaded is None:
            print("  No patch-cell id/alpha table found in .h5; skipping slide.")
            continue

        cell_id, alpha, emb_by_space = loaded

        # Precompute the stable grouping once per slide: reuse for all spaces
        order = np.argsort(cell_id, kind="mergesort")
        cell_id_sorted = cell_id[order]
        change = np.r_[True, cell_id_sorted[1:] != cell_id_sorted[:-1], True]
        bounds = np.flatnonzero(change)
        starts = bounds[:-1]
        ends = bounds[1:]

        # For given slide, analyze each embedding space
        for space_ordinal, embedding_space in enumerate(args.spaces):
            emb = emb_by_space.get(embedding_space)
            if emb is None:
                print(f"  - space='{embedding_space}': not available, skipping.")
                continue

            if emb.shape[0] == 0:
                print(f"  - space='{embedding_space}': empty patch-cell table, skipping.")
                continue

            print(
                f"  - space='{embedding_space}': "
                f"N_patch_cell={emb.shape[0]}, dim={emb.shape[1]}"
            )

            # One RNG stream per (slide, space, diagnostic)
            def slide_stream(tag: int, _sid=slide_id, _s=space_ordinal) -> np.random.Generator:
                return stream_rng(stable_seed(_sid), _s, tag)

            df_cell, df_view = analyze_one_slide_space(
                slide_id=slide_id,
                tissue=tissue,
                embedding_space=embedding_space,
                cell_id=cell_id,
                alpha=alpha,
                emb=emb,
                celltype_map=celltype_map,
                min_views=args.min_views,
                save_per_view=(not args.no_per_view),
                order=order,
                starts=starts,
                ends=ends,
                compute_impostor_loo=(not args.no_impostor_loo),
                rng=slide_stream(SLIDE_STREAM_IMPOSTOR_LOO),
            )

            print(f"    -> cells with >= {args.min_views} views: {len(df_cell)}")

            # Secondary cell-type axis columns (per slide, as cell_id_int is slide-scoped)
            df_cell = attach_secondary_axis_columns(df_cell, celltype_map_by_axis, primary_axis)
            df_view = attach_secondary_axis_columns(df_view, celltype_map_by_axis, primary_axis)

            per_cell_writer.write(df_cell)
            per_view_writer.write(df_view)
            del df_cell, df_view

            # Random-pair baseline (per-slide).
            if not args.no_random_baseline:
                base = compute_random_pair_baseline(
                    emb=emb,
                    cell_id=cell_id,
                    n_pairs=int(args.n_random_pairs_per_slide),
                    rng=slide_stream(SLIDE_STREAM_RANDOM_BASELINE),
                )
                base.update({
                    "embedding_space": embedding_space,
                    "slide_id": slide_id,
                    "tissue": tissue,
                })
                random_baseline_rows.append(base)

            # Accumulate samples for the global anisotropy & celltype matrix.
            if not args.no_anisotropy:
                buf = anisotropy_sample_by_space.setdefault(embedding_space, [])
                # Take a per-slide sample to bound memory.
                n_take = min(int(args.anisotropy_sample_per_slide), emb.shape[0])
                if n_take > 0:
                    anis_rng = slide_stream(SLIDE_STREAM_ANISOTROPY)
                    idx = anis_rng.choice(emb.shape[0], size=n_take, replace=False) if n_take < emb.shape[0] else np.arange(emb.shape[0])
                    buf.append(emb[idx].astype(np.float32, copy=False))

            if not args.no_celltype_matrix:
                space_buf = celltype_sample_by_space.setdefault(embedding_space, {})
                # Rows are bucketed by their fine label (mapped to each axis later)
                ctype = np.fromiter(
                    (celltype_map_fine.get(int(c), "UNKNOWN") for c in cell_id),
                    dtype=object,
                    count=cell_id.shape[0],
                )
                # Uniform per-slide sample of up to args.celltype_sample_per_slide rows (per-type cap applied later)
                n_take = min(int(args.celltype_sample_per_slide), emb.shape[0])
                if n_take > 0:
                    ct_rng = slide_stream(SLIDE_STREAM_CELLTYPE_MATRIX)
                    idx = ct_rng.choice(emb.shape[0], size=n_take, replace=False) if n_take < emb.shape[0] else np.arange(emb.shape[0])
                    sel_emb = emb[idx].astype(np.float32, copy=False)
                    sel_cid = cell_id[idx].astype(np.int64, copy=False)
                    sel_ct = ctype[idx]
                    # Group by cell type within the slice (with the slide of each row)
                    for t in np.unique(sel_ct):
                        bucket = space_buf.setdefault(
                            str(t), {"emb": [], "cell_id": [], "slide": []}
                        )
                        mask = sel_ct == t
                        bucket["emb"].append(sel_emb[mask])
                        bucket["cell_id"].append(sel_cid[mask])
                        bucket["slide"].append(
                            np.full(int(mask.sum()), slide_id, dtype=object)
                        )

            # Free this space's matrix before loading the next one
            emb_by_space.pop(embedding_space, None)
            del emb

    # Close the streamed parquet tables
    per_cell_path = per_cell_writer.close()
    per_view_path = per_view_writer.close()
    if per_cell_path is None:
        raise RuntimeError("No cells with multiple views were found. Nothing to analyze.")
    print(f"\nSaved: {per_cell_path}")
    if per_view_path is not None:
        print(f"Saved: {per_view_path}")

    per_cell = pd.read_parquet(per_cell_path)

    # Summaries
    summary_global = summarize_per_cell(per_cell, ["embedding_space"])
    summary_by_slide = summarize_per_cell(per_cell, ["embedding_space", "slide_id"])
    summary_by_n_views = summarize_per_cell(per_cell, ["embedding_space", "n_views"])

    # One summary block per cell-type axis, with a `celltype_axis` column
    by_ct_blocks = []
    for axis in celltype_axes:
        col = "cell_type" if axis == primary_axis else f"cell_type_{axis}"
        if col not in per_cell.columns:
            continue
        # Keep only this axis's cell-type column, renamed to `cell_type`
        frame = per_cell.drop(
            columns=[c for c in per_cell.columns
                     if (c == "cell_type" or c.startswith("cell_type_")) and c != col]
        ).rename(columns={col: "cell_type"})
        block = summarize_per_cell(frame, ["embedding_space", "cell_type"])
        if block.empty:
            continue
        block = block[block["n_cells"] >= args.min_cells_per_group].copy()
        block.insert(0, "celltype_axis", axis)
        by_ct_blocks.append(block)
    summary_by_cell_type = (
        pd.concat(by_ct_blocks, axis=0, ignore_index=True) if by_ct_blocks else pd.DataFrame()
    )

    if "tissue" in per_cell.columns and (per_cell["tissue"] != "UNKNOWN").any():
        summary_by_tissue = summarize_per_cell(per_cell, ["embedding_space", "tissue"])
        if not summary_by_tissue.empty:
            summary_by_tissue = summary_by_tissue[summary_by_tissue["n_cells"] >= args.min_cells_per_group].copy()
    else:
        summary_by_tissue = pd.DataFrame()

    summary_global.to_csv(out_dir / "summary_global.csv", index=False)
    summary_by_slide.to_csv(out_dir / "summary_by_slide.csv", index=False)
    summary_by_n_views.to_csv(out_dir / "summary_by_n_views.csv", index=False)
    summary_by_cell_type.to_csv(out_dir / "summary_by_cell_type.csv", index=False)
    if not summary_by_tissue.empty:
        summary_by_tissue.to_csv(out_dir / "summary_by_tissue.csv", index=False)

    print(f"Saved: {out_dir / 'summary_global.csv'}")
    print(f"Saved: {out_dir / 'summary_by_slide.csv'}")
    print(f"Saved: {out_dir / 'summary_by_n_views.csv'}")
    print(f"Saved: {out_dir / 'summary_by_cell_type.csv'}")
    if not summary_by_tissue.empty:
        print(f"Saved: {out_dir / 'summary_by_tissue.csv'}")

    # Alpha analysis.
    # Read back only the columns used by the alpha analyses
    if per_view_path is not None:
        per_view = pd.read_parquet(
            per_view_path,
            columns=["embedding_space", "slide_id", "alpha_frac_max",
                     "loo_cos_prenorm", "loo_cos_raw"],
        )
    else:
        per_view = pd.DataFrame()
    if not args.no_per_view and not per_view.empty:
        alpha_bin_summary = build_alpha_bin_summary(per_view)
        alpha_bin_summary.to_csv(out_dir / "alpha_bin_summary.csv", index=False)
        print(f"Saved: {out_dir / 'alpha_bin_summary.csv'}")

        alpha_corr_global = correlation_table(
            per_view,
            group_cols=["embedding_space"],
            x_col="alpha_frac_max",
            y_col="loo_cos_prenorm",
            min_rows=args.min_rows_for_corr,
        )
        alpha_corr_by_slide = correlation_table(
            per_view,
            group_cols=["embedding_space", "slide_id"],
            x_col="alpha_frac_max",
            y_col="loo_cos_prenorm",
            min_rows=args.min_rows_for_corr,
        )

        alpha_corr_global.to_csv(out_dir / "alpha_correlation_global.csv", index=False)
        alpha_corr_by_slide.to_csv(out_dir / "alpha_correlation_by_slide.csv", index=False)

        print(f"Saved: {out_dir / 'alpha_correlation_global.csv'}")
        print(f"Saved: {out_dir / 'alpha_correlation_by_slide.csv'}")
    else:
        alpha_bin_summary = pd.DataFrame()

    # Discriminability and collapse diagnostics

    # 1. Random-pair baseline (per-slide + global)
    if not args.no_random_baseline and random_baseline_rows:
        df_baseline_slide = pd.DataFrame(random_baseline_rows)
        cols = [
            "embedding_space", "slide_id", "tissue",
            "n_rows", "n_pairs",
            "cos_mean", "cos_median", "cos_p10", "cos_p90",
        ]
        df_baseline_slide = df_baseline_slide[[c for c in cols if c in df_baseline_slide.columns]]
        df_baseline_slide.to_csv(out_dir / "random_pair_baseline_by_slide.csv", index=False)
        print(f"Saved: {out_dir / 'random_pair_baseline_by_slide.csv'}")

        global_rows = []
        for space, sub in df_baseline_slide.groupby("embedding_space", dropna=False):
            arr_mean = pd.to_numeric(sub["cos_mean"], errors="coerce").dropna().to_numpy(dtype=np.float64)
            global_rows.append({
                "embedding_space": space,
                "n_slides": int(len(sub)),
                "n_pairs_total": int(pd.to_numeric(sub["n_pairs"], errors="coerce").fillna(0).sum()),
                "cos_mean_of_slide_means": float(np.mean(arr_mean)) if arr_mean.size else np.nan,
                "cos_median_of_slide_medians": float(np.median(
                    pd.to_numeric(sub["cos_median"], errors="coerce").dropna().to_numpy(dtype=np.float64)
                )) if sub["cos_median"].notna().any() else np.nan,
                "cos_p10_of_slide_p10s": float(np.median(
                    pd.to_numeric(sub["cos_p10"], errors="coerce").dropna().to_numpy(dtype=np.float64)
                )) if sub["cos_p10"].notna().any() else np.nan,
                "cos_p90_of_slide_p90s": float(np.median(
                    pd.to_numeric(sub["cos_p90"], errors="coerce").dropna().to_numpy(dtype=np.float64)
                )) if sub["cos_p90"].notna().any() else np.nan,
            })
        df_baseline_global = pd.DataFrame(global_rows)
        df_baseline_global.to_csv(out_dir / "random_pair_baseline_global.csv", index=False)
        print(f"Saved: {out_dir / 'random_pair_baseline_global.csv'}")

    # 2. Anisotropy / effective rank (one row per embedding space)
    if not args.no_anisotropy and anisotropy_sample_by_space:
        anis_rows = []
        for space, chunks in anisotropy_sample_by_space.items():
            if not chunks:
                continue
            # Own stream, keyed on the space's ordinal in --spaces and a step tag
            global_rng = stream_rng(args.spaces.index(space), GLOBAL_STREAM_ANISOTROPY)
            big = np.concatenate(chunks, axis=0)
            # Cap to args.anisotropy_sample.
            if big.shape[0] > int(args.anisotropy_sample):
                idx = global_rng.choice(big.shape[0], size=int(args.anisotropy_sample), replace=False)
                big = big[idx]
            rec = compute_anisotropy(big)
            rec["embedding_space"] = space
            rec["mean_random_cosine"] = compute_mean_random_cosine(
                big, n_pairs=min(50_000, big.shape[0] * 5), rng=global_rng,
            )
            anis_rows.append(rec)
        if anis_rows:
            df_anis = pd.DataFrame(anis_rows)
            cols = [
                "embedding_space", "n_rows_sampled", "embedding_dim",
                "top1_var_fraction", "top5_var_fraction", "effective_rank",
                "mean_random_cosine",
            ]
            df_anis = df_anis[[c for c in cols if c in df_anis.columns]]
            df_anis.to_csv(out_dir / "representation_geometry.csv", index=False)
            print(f"Saved: {out_dir / 'representation_geometry.csv'}")

    # 3. Cell-type cosine matrix (long form + heatmap plots)
    celltype_matrix_df = pd.DataFrame()
    if not args.no_celltype_matrix and celltype_sample_by_space:
        matrix_rows = []
        for space, by_type in celltype_sample_by_space.items():
            if not by_type:
                continue
            # Own stream family per space
            space_ordinal = args.spaces.index(space)
            # Concatenate per-type chunks (per-type cap applied in compute_celltype_cosine_matrix)
            types = sorted(by_type.keys())
            type_emb: List[np.ndarray] = []
            type_cid: List[np.ndarray] = []
            type_lbl: List[np.ndarray] = []
            type_slide: List[np.ndarray] = []
            for t in types:
                if not by_type[t]["emb"]:
                    continue
                e_cat = np.concatenate(by_type[t]["emb"], axis=0)
                c_cat = np.concatenate(by_type[t]["cell_id"], axis=0)
                s_cat = np.concatenate(by_type[t]["slide"], axis=0)
                type_emb.append(e_cat)
                type_cid.append(c_cat)
                type_slide.append(s_cat)
                type_lbl.append(np.full(e_cat.shape[0], t, dtype=object))
            if not type_emb:
                continue
            big_emb = np.concatenate(type_emb, axis=0)
            big_cid = np.concatenate(type_cid, axis=0)
            big_lbl = np.concatenate(type_lbl, axis=0)
            big_slide = np.concatenate(type_slide, axis=0)
            # One matrix per cell-type axis, on the same sampled rows (fine labels mapped to each axis)
            for axis, mapping in celltype_axes.items():
                lbl_axis = apply_label_map(big_lbl, mapping)
                # Both pair scopes, as separate row blocks under a `pair_scope` column
                for scope in PAIR_SCOPES:
                    df_mat = compute_celltype_cosine_matrix(
                        emb=big_emb,
                        cell_id=big_cid,
                        celltype_per_row=lbl_axis,
                        min_cells_per_type=int(args.min_cells_per_group),
                        n_pairs_per_type_pair=int(args.n_pairs_per_type_pair),
                        # Cap stream keyed on (space, axis), pair stream on (space, axis, scope)
                        cap_rng=stream_rng(space_ordinal, GLOBAL_STREAM_CELLTYPE_MATRIX,
                                           MATRIX_STREAM_CAP, stable_seed(axis)),
                        pair_rng=stream_rng(space_ordinal, GLOBAL_STREAM_CELLTYPE_MATRIX,
                                            MATRIX_STREAM_PAIRS, stable_seed(axis),
                                            stable_seed(scope)),
                        max_rows_per_type=int(args.celltype_max_per_type),
                        slide_per_row=big_slide,
                        pair_scope=scope,
                    )
                    if df_mat.empty:
                        continue
                    df_mat.insert(0, "embedding_space", space)
                    df_mat.insert(1, "celltype_axis", axis)
                    matrix_rows.append(df_mat)

        if matrix_rows:
            celltype_matrix_df = pd.concat(matrix_rows, axis=0, ignore_index=True)
            celltype_matrix_df.to_csv(out_dir / "celltype_cosine_matrix.csv", index=False)
            print(f"Saved: {out_dir / 'celltype_cosine_matrix.csv'}")

    # Plots
    if not args.no_plots:
        plot_histograms(per_cell, plot_dir)
        plot_alpha_bin_summary(alpha_bin_summary, plot_dir)
        if not celltype_matrix_df.empty:
            plot_celltype_cosine_matrix(celltype_matrix_df, plot_dir)
        print(f"Saved plots to: {plot_dir}")

    # Console summary
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 220)

    key_cols = [
        "embedding_space",
        "n_cells",
        "pair_cos_mean_median",
        "pair_cos_min_median",
        "norm_cv_median",
        "pair_norm_ratio_mean_median",
        "resultant_length_alpha_median",
        "mean_loo_cos_prenorm_median",
        "merge_cos_alpha_raw_vs_alpha_prenorm_median",
        "merge_cos_alpha_raw_vs_unweighted_raw_median",
    ]
    present_key_cols = [c for c in key_cols if c in summary_global.columns]

    print("\n================ GLOBAL SUMMARY ================\n")
    print(summary_global[present_key_cols].to_string(index=False))

    if not args.no_per_view and not per_view.empty:
        alpha_corr_global_path = out_dir / "alpha_correlation_global.csv"
        print("\n============ ALPHA vs CONSENSUS ============\n")
        try:
            print(pd.read_csv(alpha_corr_global_path).to_string(index=False))
        except Exception:
            pass

    print("\nDone.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze similarity of H&E cell embeddings across overlapping patches for the same cell.")

    parser.add_argument("--training_dataset_id", type=str,
                        required=True,
                        help="Training dataset ID used for eval_st_{training_dataset_id}, e.g. ds_ct1_20x_224_o112_1")
    
    parser.add_argument("--training_num", type=str,
                        required=True,
                        help="Training number, e.g. '46' for training_46")
    
    parser.add_argument("--checkpoint_num", type=str,
                        required=True,
                        help="Checkpoint epoch number, e.g. '13' for checkpoint_epoch_13.pth")

    parser.add_argument("--ct_cat_id", type=str, default="ct_1",
                        help=("Cell-type category id of data/annots/annot_dicts_{ct_cat_id}/ (e.g. ct_1)."))
    parser.add_argument("--celltype_axes", type=str, nargs="+",
                        default=["fine", "coarse"], choices=["fine", "coarse"],
                        help=("Cell-type axes to report (the first one is the primary axis, used for the cell_type column)."))
    parser.add_argument("--coarse_mapping_version", type=str, default="v1",
                        help=("Version of the coarse cell-type mapping (required with 'coarse')."))
    parser.add_argument("--overwrite", action="store_true",
                        help=("Allow writing into an output folder produced by a different "
                              "A0_ANALYSIS_VERSION."))
    parser.add_argument("--spaces", type=str, nargs="+",
                        default=["he", "he_clip"],
                        choices=["he", "he_clip"],
                        help="Embedding spaces to analyze.")
    
    parser.add_argument("--min_views", type=int,
                        default=2,
                        help="Minimum number of patch views required for a cell to be analyzed.")
    
    parser.add_argument("--min_cells_per_group", type=int,
                        default=30,
                        help="Minimum number of cells required to keep a tissue/cell-type summary row.")
    
    parser.add_argument("--min_rows_for_corr", type=int,
                        default=100,
                        help="Minimum number of per-view rows required to compute a correlation.")

    parser.add_argument("--no_per_view",
                        action="store_true",
                        help="Do not save per-view metrics and alpha-based analyses.")

    parser.add_argument("--no_plots",
                        action="store_true",
                        help="Do not generate png plots.")

    # Reproducibility of the sampling-based diagnostics
    parser.add_argument("--seed", type=int,
                        default=42,
                        help="Random seed used for impostor LOO and random-pair samples (set_global_seed + rng).")

    # Random-pair baseline
    parser.add_argument("--n_random_pairs_per_slide", type=int,
                        default=100_000,
                        help="Number of random different-cell pairs per slide for the random-pair cosine baseline.")
    parser.add_argument("--no_random_baseline", action="store_true",
                        help="Disable the random-pair cosine baseline.")

    # Impostor LOO
    parser.add_argument("--no_impostor_loo", action="store_true",
                        help="Disable the per-cell impostor leave-one-out cosine.")

    # Anisotropy
    parser.add_argument("--anisotropy_sample", type=int,
                        default=200_000,
                        help="Total rows used for the SVD-based anisotropy diagnostic per embedding space.")
    parser.add_argument("--anisotropy_sample_per_slide", type=int,
                        default=20_000,
                        help="Per-slide sample cap accumulated into the anisotropy pool.")
    parser.add_argument("--no_anisotropy", action="store_true",
                        help="Disable the anisotropy / effective-rank diagnostic.")

    # Cell-type cosine matrix
    parser.add_argument("--n_pairs_per_type_pair", type=int,
                        default=5_000,
                        help="Number of random pairs per (cell_type, cell_type) cell of the K x K matrix.")
    parser.add_argument("--celltype_sample_per_slide", type=int,
                        default=10_000,
                        help="Per-slide sample cap accumulated into the per-celltype pools for the matrix.")
    parser.add_argument("--celltype_max_per_type", type=int,
                        default=100_000,
                        help="Maximum rows kept per cell type across slides before sampling pairs for the matrix.")
    parser.add_argument("--no_celltype_matrix", action="store_true",
                        help="Disable the cell-type pairwise cosine matrix.")

    args = parser.parse_args()
    main(args)