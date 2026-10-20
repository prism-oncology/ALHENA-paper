"""Task A.1 target data: gene tables, per-gene targets, raw counts and signature scores."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .common import (
    SlideRecord,
    resolve_eval_dataset_root,
)


def _part1_dir() -> Optional[Path]:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "_shared_st" / "st_data.py").exists():
            return parent
    return None


_PART1_DIR = _part1_dir()
if _PART1_DIR is not None and str(_PART1_DIR.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(_PART1_DIR.parent.parent.parent))

from src.evaluation.part1_ST._shared_st.st_data import (  # noqa: E402,F401
    SlideRowBundle,
    build_row_manifest_for_slide,
    build_split_specific_cell_embedding_index,
    load_cell_embeddings_via_main_assignment,
    load_membership_table,
    load_missing_cells_for_modelgroup,
    load_novae_patch_domains,
    load_patch_embeddings,
    load_patch_geometry,
    load_slide_records,
    load_split_specific_cell_embeddings,
    _obs_bool_array,
)


@dataclass
class CanonicalizedMatrix:
    genes: np.ndarray
    X: np.ndarray


@dataclass
class PredictionSlideBundle:
    slide_id: str
    subset: str
    source: str
    tissue: str
    row_keys: np.ndarray
    coords_xy: np.ndarray
    gene_names: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    obs_df: pd.DataFrame


def _collapse_duplicate_gene_columns(
    X: Any,
    genes: np.ndarray,
) -> Tuple[Any, np.ndarray]:
    """Sum the expression columns that share a gene symbol (output columns in np.unique order)."""
    uniq, inv = np.unique(genes, return_inverse=True)
    inv = np.asarray(inv).ravel()
    n_in = int(np.asarray(genes).shape[0])
    if sparse.issparse(X):
        Xc = X.tocsr()
        grouping = sparse.csr_matrix(
            (np.ones(n_in, dtype=Xc.dtype), (np.arange(n_in), inv)),
            shape=(n_in, len(uniq)),
        )
        X = (Xc @ grouping).tocsr()
    else:
        Xd = np.asarray(X, dtype=np.float32)
        grouping = sparse.csr_matrix(
            (np.ones(n_in, dtype=np.float32), (np.arange(n_in), inv)),
            shape=(n_in, len(uniq)),
        )
        X = np.asarray(Xd @ grouping.toarray(), dtype=np.float32)
    return X, uniq.astype(object)


def canonicalize_gene_table(
    *,
    adata_path: Path,
    slide_record: SlideRecord,
    gene_subset: Optional[Sequence[str]] = None,
    mouse_mapping_path: Optional[Path] = None,
) -> CanonicalizedMatrix:
    adata = ad.read_h5ad(adata_path)
    try:
        genes = np.asarray(adata.var_names.astype(str), dtype=object)
        X = adata.X
        if sparse.issparse(X):
            X = X.tocsr()
        else:
            X = np.asarray(X, dtype=np.float32)
        if slide_record.species == "Mus musculus":
            if mouse_mapping_path is None or not mouse_mapping_path.exists():
                raise FileNotFoundError(f"Mouse mapping file not found for slide '{slide_record.slide_id}': {mouse_mapping_path}")
            mp = pd.read_parquet(mouse_mapping_path)
            if "target_gene_name" not in mp.columns or "var_index" not in mp.columns:
                raise KeyError(f"Mouse mapping '{mouse_mapping_path}' must contain 'var_index' and 'target_gene_name'.")
            mp = mp.sort_values("var_index")
            target_raw = mp["target_gene_name"]
            if target_raw.shape[0] != genes.shape[0]:
                raise ValueError(f"Mouse mapping length mismatch for slide '{slide_record.slide_id}': {target_raw.shape[0]} vs {genes.shape[0]}.")
            mapped_mask = target_raw.notna().to_numpy(dtype=bool)
            mapped_vals = target_raw.loc[mapped_mask].astype(str).str.strip()
            nonempty_mask = (mapped_vals != "").to_numpy(dtype=bool)
            mapped_cols = np.flatnonzero(mapped_mask)[nonempty_mask]
            genes = mapped_vals.loc[nonempty_mask].to_numpy(dtype=object)
            if sparse.issparse(X):
                X = X[:, mapped_cols]
            else:
                X = np.asarray(X[:, mapped_cols], dtype=np.float32)
            X, genes = _collapse_duplicate_gene_columns(X, genes)

        if gene_subset is not None:
            gene_subset_arr = np.asarray(list(gene_subset), dtype=object)
            lookup = {g: i for i, g in enumerate(genes.tolist())}
            keep_cols = [lookup[g] for g in gene_subset_arr.tolist() if g in lookup]
            out_genes = gene_subset_arr[np.array([g in lookup for g in gene_subset_arr.tolist()], dtype=bool)]
            if sparse.issparse(X):
                X = X[:, keep_cols].toarray().astype(np.float32)
            else:
                X = np.asarray(X[:, keep_cols], dtype=np.float32)
            genes = out_genes
        else:
            if sparse.issparse(X):
                X = X.toarray().astype(np.float32)
            else:
                X = np.asarray(X, dtype=np.float32)

        return CanonicalizedMatrix(genes=np.asarray(genes, dtype=object), X=np.asarray(X, dtype=np.float32))
    finally:
        if getattr(adata, "file", None) is not None:
            try:
                adata.file.close()
            except Exception:
                pass


def apply_zscore(
    Y: np.ndarray,
    *,
    slide_id: str,
    gene_names: np.ndarray,
    zscore_stats: pd.DataFrame,
) -> np.ndarray:
    """Z-score each measured gene column of Y with the per-(slide, gene) stats (all-NaN columns are left as is)."""
    sub = zscore_stats.loc[zscore_stats["slide_id"].astype(str) == str(slide_id)]
    lut = {
        str(g): (float(m), float(s))
        for g, m, s in zip(sub["gene"], sub["mean"], sub["std"])
    }
    out = np.array(Y, dtype=np.float32, copy=True)
    for j, gene in enumerate(gene_names.tolist()):
        col = out[:, j]
        if not np.any(np.isfinite(col)):
            continue
        if str(gene) not in lut:
            raise KeyError(
                f"Missing frozen z-score stats for measured gene "
                f"(slide={slide_id}, gene={gene}). Re-run _2_fill_eval_settings.py."
            )
        mean_g, std_g = lut[str(gene)]
        out[:, j] = (col - mean_g) / max(std_g, 1.0e-8)
    return out


def _load_canonical_row_counts(
    *,
    adata_path: Path,
    slide_record: SlideRecord,
    ordered_row_keys: Sequence[str],
    mouse_mapping_path: Optional[Path] = None,
) -> Tuple[Any, np.ndarray]:
    """Raw counts (X, genes) of the given rows, after the mouse->human gene mapping (no gene subset, no normalization)."""
    adata = ad.read_h5ad(adata_path)
    try:
        obs_index = pd.Index(adata.obs_names.astype(str))
        row_pos = obs_index.get_indexer(pd.Index(list(ordered_row_keys), dtype=object))
        if np.any(row_pos < 0):
            missing = np.asarray(list(ordered_row_keys), dtype=object)[row_pos < 0][:10].tolist()
            raise KeyError(f"Some requested rows are missing from '{adata_path}': {missing}")

        row_view = adata[row_pos, :].copy()
        genes = np.asarray(row_view.var_names.astype(str), dtype=object)
        X = row_view.X
        if sparse.issparse(X):
            X = X.tocsr()
        else:
            X = np.asarray(X, dtype=np.float32)

        if slide_record.species == "Mus musculus":
            if mouse_mapping_path is None or not mouse_mapping_path.exists():
                raise FileNotFoundError(f"Mouse mapping file not found for slide '{slide_record.slide_id}': {mouse_mapping_path}")
            mp = pd.read_parquet(mouse_mapping_path).sort_values("var_index")
            target_raw = mp["target_gene_name"]
            if target_raw.shape[0] != genes.shape[0]:
                raise ValueError(f"Mouse mapping mismatch for slide '{slide_record.slide_id}'.")
            mapped_mask = target_raw.notna().to_numpy(dtype=bool)
            mapped_vals = target_raw.loc[mapped_mask].astype(str).str.strip()
            nonempty_mask = (mapped_vals != "").to_numpy(dtype=bool)
            mapped_cols = np.flatnonzero(mapped_mask)[nonempty_mask]
            genes = mapped_vals.loc[nonempty_mask].to_numpy(dtype=object)
            if sparse.issparse(X):
                X = X[:, mapped_cols]
            else:
                X = np.asarray(X[:, mapped_cols], dtype=np.float32)
            X, genes = _collapse_duplicate_gene_columns(X, genes)

        return X, np.asarray(genes, dtype=object)
    finally:
        if getattr(adata, "file", None) is not None:
            try:
                adata.file.close()
            except Exception:
                pass


def extract_targets_for_rows(
    *,
    adata_path: Path,
    slide_record: SlideRecord,
    level: str,
    ordered_row_keys: Sequence[str],
    ordered_gene_list: Sequence[str],
    target_transform: Mapping[str, Any],
    mouse_mapping_path: Optional[Path] = None,
    zscore_stats: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Target matrix of the given rows and genes: log1p(count / row total * target_sum), then z-scored per (slide, gene)
    and winsorized if zscore_stats is given. Returns (Y, detected): Y is NaN for unmeasured genes and zero-count rows,
    detected is True where the raw count is > 0.
    """
    X, genes = _load_canonical_row_counts(
        adata_path=adata_path,
        slide_record=slide_record,
        ordered_row_keys=ordered_row_keys,
        mouse_mapping_path=mouse_mapping_path,
    )
    gene_lookup = {g: i for i, g in enumerate(genes.tolist())}
    G = len(ordered_gene_list)
    Y = np.full((len(ordered_row_keys), G), np.nan, dtype=np.float32)
    detected = np.zeros((len(ordered_row_keys), G), dtype=bool)
    present_cols = [gene_lookup[g] for g in ordered_gene_list if g in gene_lookup]
    present_mask = np.array([g in gene_lookup for g in ordered_gene_list], dtype=bool)
    if np.any(present_mask):
        if sparse.issparse(X):
            row_totals = np.asarray(X.sum(axis=1)).reshape(-1).astype(np.float64)
            vals = X[:, present_cols].toarray().astype(np.float32)
        else:
            X_dense = np.asarray(X, dtype=np.float32)
            row_totals = X_dense.sum(axis=1, dtype=np.float64)
            vals = np.asarray(X_dense[:, present_cols], dtype=np.float32)
        detected[:, present_mask] = vals > 0
        target_sum = float(target_transform.get("target_sum", 1.0e4))
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(row_totals > 0.0, target_sum / row_totals, 0.0)
        e = np.log1p(vals.astype(np.float64) * scale[:, None]).astype(np.float32)
        # Units with zero total counts cannot be depth-normalized; treat as unmeasured.
        e[row_totals <= 0.0, :] = np.nan
        Y[:, present_mask] = e
    if zscore_stats is not None and bool(np.any(present_mask)):
        Y = apply_zscore(
            Y,
            slide_id=str(slide_record.slide_id),
            gene_names=np.asarray(ordered_gene_list, dtype=object),
            zscore_stats=zscore_stats,
        )
        Y = winsorize_zscored(Y, target_transform)
    return Y, detected


def winsorize_zscored(Y: np.ndarray, target_transform: Mapping[str, Any]) -> np.ndarray:
    """Clip a z-scored matrix to +/- target_transform.zscore_clip (NaN passes through)."""
    clip = target_transform.get("zscore_clip", None)
    if clip is None:
        raise KeyError(
            "target_transform.zscore_clip is required."
        )
    c = float(clip)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError(f"target_transform.zscore_clip must be finite and > 0, got {clip!r}.")
    return np.clip(np.asarray(Y), -c, c)


def extract_raw_counts_for_rows(
    *,
    adata_path: Path,
    slide_record: SlideRecord,
    ordered_row_keys: Sequence[str],
    mouse_mapping_path: Optional[Path] = None,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Raw count matrix (CSR) and gene names of the given rows, for the count-thinning noise ceiling."""
    X, genes = _load_canonical_row_counts(
        adata_path=adata_path,
        slide_record=slide_record,
        ordered_row_keys=ordered_row_keys,
        mouse_mapping_path=mouse_mapping_path,
    )
    if sparse.issparse(X):
        C = X.tocsr()
    else:
        C = sparse.csr_matrix(np.asarray(X, dtype=np.float32))
    return C, np.asarray(genes, dtype=object)


def build_signature_weight_matrix(
    gene_names: Sequence[str],
    targets_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Signed weight matrix W (n_genes x n_signatures) from signature_targets.parquet, signatures sorted by
    (signature_set, signature_id). Member genes absent from gene_names are dropped. Returns (W, sig_keys).
    """
    gene_index = {str(g): i for i, g in enumerate(gene_names)}
    sig_keys = sorted(
        {
            (str(s), str(i))
            for s, i in zip(targets_df["signature_set"], targets_df["signature_id"])
        }
    )
    sig_col = {k: j for j, k in enumerate(sig_keys)}
    W = np.zeros((len(gene_names), len(sig_keys)), dtype=np.float64)
    for set_name, sig_id, gene, weight in zip(
        targets_df["signature_set"].astype(str),
        targets_df["signature_id"].astype(str),
        targets_df["gene"].astype(str),
        targets_df["weight"].astype(float),
    ):
        gi = gene_index.get(str(gene))
        if gi is None:
            continue
        W[gi, sig_col[(str(set_name), str(sig_id))]] = float(weight)
    return W, sig_keys


def aggregate_signature_targets(
    Z: np.ndarray,
    finite_mask: np.ndarray,
    gene_names: Sequence[str],
    targets_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[Tuple[str, str]], np.ndarray]:
    """
    Per-signature scores y[l, s] = sum_g w[g] * Z[l, g] / sum_g |w[g]| over the member genes present and finite at l
    (NaN if none). Returns (Y_sig, sig_keys, n_present), n_present being the count of present finite members.
    """
    W, sig_keys = build_signature_weight_matrix(gene_names, targets_df)
    n_loc = int(np.asarray(Z).shape[0])
    if len(sig_keys) == 0:
        empty = np.zeros((n_loc, 0), dtype=np.float64)
        return empty, sig_keys, empty
    F = np.asarray(finite_mask, dtype=bool)
    Ff = F.astype(np.float64)
    Z0 = np.where(F, np.asarray(Z, dtype=np.float64), 0.0)
    absW = np.abs(W)
    member = (W != 0.0).astype(np.float64)
    num = Z0 @ W            # (n_loc, n_sig)
    den = Ff @ absW         # (n_loc, n_sig)  -- available abs-weight sum
    n_present = Ff @ member  # (n_loc, n_sig)
    with np.errstate(invalid="ignore", divide="ignore"):
        Y = np.where(den > 0.0, num / den, np.nan)
    return Y.astype(np.float64), sig_keys, n_present


def build_slide_gene_panel_manifest(training_dataset_id: str, slide_id: str) -> pd.DataFrame:
    slide_records = load_slide_records(training_dataset_id)
    rec = slide_records[slide_id]
    eval_root = resolve_eval_dataset_root(training_dataset_id)
    adata_path = eval_root / "st" / f"{slide_id}.h5ad"
    mouse_map = eval_root / "mouse_to_human_gene_mappings" / f"{slide_id}_mouse2human_mapping.parquet"
    mat = canonicalize_gene_table(adata_path=adata_path, slide_record=rec, gene_subset=None, mouse_mapping_path=mouse_map if mouse_map.exists() else None)
    return pd.DataFrame({"slide_id": slide_id, "gene": mat.genes.astype(object)})
