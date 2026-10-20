"""
Build the per-gene property table of the Task A.1 gene strata for the genes of a G_base panel, from TRAIN cells only:
cell-type specificity (median over slides of per-slide Tau) and spatial structure (median over slides of bias-corrected Moran's I).

Saved in <ROOT_HistoST> / eval / eval_datasets / st_eval_data / eval_st_<training_dataset_id> / gene_properties / <gene_property_id> /
    - gene_property_table.parquet: one row per G_base gene
    - config.json: parameters, paths and input G_base provenance
"""

import sys
from pathlib import Path
import argparse
import logging
import os
import json
import datetime as _dt
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse

logger = logging.getLogger(__name__)

# Add project root
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path


DEFAULT_CELL_ID_COL = "cell_id"
DEFAULT_CELL_TYPE_COL = "cell_type"
DEFAULT_GRAPH_KEY = "spatial_connectivities"
DEFAULT_CT_CAT_ID = "ct_1"
_NULL_LABELS = {"", "nan", "none", "na", "null", "<na>"}



# ---------------------------------------------------------------------------
# Shared utilities (cell-id handling, IO, dataset loading)
# ---------------------------------------------------------------------------

def _xenium_str_ids_to_int64(cell_id_strs: pd.Series) -> Optional[np.ndarray]:
    """
    Vectorized conversion of Xenium Explorer string cell IDs (e.g. 'aaaachba-1')
    into int64 IDs. Returns None if format does not match.
    """
    if cell_id_strs.empty:
        return np.empty((0,), dtype=np.int64)

    s = cell_id_strs.astype(str).to_numpy()
    b = s.astype("S")

    if not np.all(np.char.endswith(b, b"-1")):
        return None

    maxlen = b.dtype.itemsize
    raw = b.tobytes()
    chars = np.frombuffer(raw, dtype=np.uint8).reshape(b.shape[0], maxlen)[:, :8]
    digits = chars.astype(np.int16) - ord("a")

    if np.any(digits < 0) or np.any(digits > 15):
        return None

    powers = np.array([16**i for i in range(7, -1, -1)], dtype=np.int64)
    out = (digits.astype(np.int64) * powers).sum(axis=1) + 1
    return out



def _load_cell_ids_sets_npz(
    cell_ids_sets_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load cell_ids_sets.npz and return the (train_only, valid_only, test_only) dicts (only the TRAIN key is required)."""
    if not os.path.isfile(cell_ids_sets_path):
        raise FileNotFoundError(
            f"[ERROR] cell_ids_sets file not found: '{cell_ids_sets_path}'."
        )

    npz = np.load(cell_ids_sets_path, allow_pickle=True)

    def _get_dict(name: str, required: bool = False) -> Dict[str, np.ndarray]:
        if name not in npz.files:
            if required:
                raise KeyError(
                    f"[ERROR] '{name}' key not found in cell_ids_sets npz: "
                    f"'{cell_ids_sets_path}'. Available keys: {list(npz.files)}."
                )
            return {}
        obj = npz[name]
        try:
            return obj.item()
        except Exception:
            return dict(obj)

    return (
        _get_dict("train_only_by_slide", required=True),
        _get_dict("valid_only_by_slide"),
        _get_dict("test_only_by_slide"),
    )



def _standardize_cell_ids_for_isin(
    obs_cell_ids: pd.Series,
    target_cell_ids: np.ndarray,
) -> Tuple[pd.Series, np.ndarray]:
    """Make obs_cell_ids and target_cell_ids comparable for membership testing."""
    tgt_arr = np.asarray(target_cell_ids)

    obs_num = pd.to_numeric(obs_cell_ids, errors="coerce")
    if not obs_num.isna().any():
        return obs_num.astype(np.int64), tgt_arr.astype(np.int64, copy=False)

    if tgt_arr.dtype.kind in ("i", "u"):
        obs_int = _xenium_str_ids_to_int64(obs_cell_ids)
        if obs_int is not None:
            return (
                pd.Series(obs_int, index=obs_cell_ids.index, dtype=np.int64),
                tgt_arr.astype(np.int64, copy=False),
            )

    # Fallback: both sides non-numeric (e.g. Xenium-string on both); compare as strings.
    return obs_cell_ids.astype(str), tgt_arr.astype(str)



def _subset_adata_to_cell_ids(
    adata: AnnData,
    cell_ids: np.ndarray,
    cell_id_col: str = DEFAULT_CELL_ID_COL,
) -> AnnData:
    """Return adata restricted to obs[cell_id_col] in cell_ids. obsp graphs follow."""
    if cell_id_col not in adata.obs.columns:
        raise KeyError(
            f"[ERROR] '{cell_id_col}' not found in adata.obs columns: "
            f"{list(adata.obs.columns)}"
        )
    if cell_ids is None or len(cell_ids) == 0:
        return adata[:0].copy()

    obs_std, tgt_std = _standardize_cell_ids_for_isin(adata.obs[cell_id_col], cell_ids)
    mask = obs_std.isin(tgt_std).to_numpy()
    return adata[mask].copy()



def _load_slide_ids(training_folder: str) -> List[str]:
    slide_ids_path = os.path.join(training_folder, "slide_ids.txt")
    if not os.path.isfile(slide_ids_path):
        raise FileNotFoundError(f"[ERROR] slide_ids.txt not found: '{slide_ids_path}'")
    with open(slide_ids_path, "r") as f:
        slide_ids = [line.strip() for line in f if line.strip()]
    if len(slide_ids) == 0:
        raise ValueError(f"[ERROR] No slide IDs found in '{slide_ids_path}'.")
    return slide_ids



def _get_gene_names(adata: AnnData, gene_col: Optional[str]) -> np.ndarray:
    """Return adata.var_names (or adata.var[gene_col]); raises on duplicates."""
    if gene_col is None:
        gene_names = adata.var_names.astype(str).to_numpy()
    else:
        if gene_col not in adata.var.columns:
            raise KeyError(
                f"[ERROR] gene_col='{gene_col}' not found in adata.var.columns: "
                f"{list(adata.var.columns)}"
            )
        gene_names = adata.var[gene_col].astype(str).to_numpy()

    idx = pd.Index(gene_names)
    if idx.has_duplicates:
        duplicated = idx[idx.duplicated()].unique().tolist()
        raise ValueError(
            f"[ERROR] Gene names contain duplicates (showing up to 10): {duplicated[:10]}."
        )
    return gene_names



def _load_cid2ct_map(ct_cat_id: str, slide_id: str) -> Dict[str, str]:
    """
    Load {cell_id (str) -> cell_type} for one slide from
    cID2cT/{ct_cat_id}/cID2cT_{slide_id}.parquet. cell_id is the Xenium Explorer
    string id; unannotated cells appear with a null cell_type.
    """
    cid2ct_dir = get_full_path("STHELAR_HistoST", "cID2cT", ct_cat_id, as_str=True)
    parquet_path = os.path.join(cid2ct_dir, f"cID2cT_{slide_id}.parquet")
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(
            f"[ERROR] cID2cT parquet not found for slide '{slide_id}': "
            f"'{parquet_path}'. Check --ct_cat_id (got '{ct_cat_id}')."
        )

    df = pd.read_parquet(parquet_path, columns=["cell_id", "cell_type"])
    df["cell_id"] = df["cell_id"].astype(str)
    df["cell_type"] = df["cell_type"].astype("object")
    if df["cell_id"].duplicated().any():
        raise ValueError(
            f"[ERROR] Duplicate cell_id values found in cID2cT parquet '{parquet_path}'."
        )
    return dict(zip(df["cell_id"].to_numpy(), df["cell_type"].to_numpy()))



def _assert_nonnegative_counts(adata: AnnData, slide_id: str) -> None:
    """Sanity-check that adata.X holds raw non-negative integer counts."""
    x_min = adata.X.min()
    if sparse.issparse(adata.X):
        x_min = float(x_min)
    if x_min < 0:
        raise ValueError(
            f"[ERROR] Negative values found in adata.X for slide '{slide_id}'. "
            "Expected raw non-negative counts."
        )

    if sparse.issparse(adata.X):
        data = np.asarray(adata.X.data)
    else:
        data = np.asarray(adata.X).ravel()
    if data.size > 0 and not np.allclose(data, np.round(data), atol=1e-6):
        raise ValueError(
            f"[ERROR] adata.X for slide '{slide_id}' contains non-integer values; "
            "this script expects raw counts (it then applies normalize_total + log1p)."
        )



def _save_json_atomic(path: str, payload: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)



def _save_parquet_atomic(path: str, df: pd.DataFrame) -> None:
    tmp_path = path + ".tmp"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)



# ---------------------------------------------------------------------------
# Math kernels (Moran's I and Tau)
# ---------------------------------------------------------------------------

def _morans_i_all_genes(X, W) -> np.ndarray:
    """
    Per-gene raw Moran's I of X (cells x genes, depth-normalized log1p) with the sparse spatial weights W (cells x cells):
    I = (n / W_sum) * sum_ij w_ij z_i z_j / sum_i z_i^2, z = x - mean(x), computed without dense centering.
    Genes with zero variance get NaN.
    """
    n = X.shape[0]
    g = X.shape[1]
    out = np.full(g, np.nan, dtype=np.float64)
    W_sum = float(W.sum())
    if n < 3 or W_sum <= 0.0:
        return out

    Xc = X.tocsr().astype(np.float64)
    g_mean = np.asarray(Xc.sum(axis=0)).ravel() / float(n)
    WX = W @ Xc
    x_Wx = np.asarray(Xc.multiply(WX).sum(axis=0)).ravel()
    row_w = np.asarray(W.sum(axis=1)).ravel()
    col_w = np.asarray(W.sum(axis=0)).ravel()
    rw_x = np.asarray(Xc.T @ row_w).ravel()
    cw_x = np.asarray(Xc.T @ col_w).ravel()
    num = x_Wx - g_mean * cw_x - g_mean * rw_x + (g_mean ** 2) * W_sum
    sumsq = np.asarray(Xc.multiply(Xc).sum(axis=0)).ravel()
    denom = sumsq - float(n) * (g_mean ** 2)

    ok = denom > 1e-12
    out[ok] = (float(n) / W_sum) * (num[ok] / denom[ok])
    return out



def _ct_expression_sums(X, ct_labels: pd.Series) -> Dict[str, Tuple[np.ndarray, int]]:
    """
    Per cell type, sum X (cells x genes) over its cells. Returns
    {cell_type: (sum_vector, n_cells)} for every non-null cell type present.
    """
    Xc = X.tocsr().astype(np.float64)
    labels = ct_labels.astype("string")
    out: Dict[str, Tuple[np.ndarray, int]] = {}
    for ct in pd.unique(labels):
        if pd.isna(ct) or str(ct).strip().lower() in _NULL_LABELS:
            continue
        mask = (labels == ct).fillna(False).to_numpy(dtype=bool)
        n = int(mask.sum())
        if n == 0:
            continue
        s = np.asarray(Xc[mask].sum(axis=0)).ravel()
        out[str(ct)] = (s, n)
    return out



def _tau_from_ct_means(M: np.ndarray) -> np.ndarray:
    """
    Per-gene Tau cell-type specificity from a CT-mean matrix M (n_cell_types x n_genes), over the K >= 2 finite
    per-CT means: tau = sum_k (1 - x_k / max_k x_k) / (K - 1) (1: marker, 0: housekeeping; NaN if K < 2 or max <= 0).
    """
    n_genes = M.shape[1]
    out = np.full(n_genes, np.nan, dtype=np.float64)
    finite = np.isfinite(M)
    k_per_gene = finite.sum(axis=0).astype(np.int64)
    xmax = np.where(finite, M, -np.inf).max(axis=0)
    ok = (k_per_gene >= 2) & np.isfinite(xmax) & (xmax > 0.0)
    if np.any(ok):
        xhat = M[:, ok] / xmax[ok][None, :]
        contrib = np.where(np.isfinite(xhat), 1.0 - xhat, 0.0)
        out[ok] = contrib.sum(axis=0) / (k_per_gene[ok] - 1).astype(np.float64)
    return out



# ---------------------------------------------------------------------------
# G_base loader
# ---------------------------------------------------------------------------

def _load_g_base(
    training_dataset_id: str,
    G_base_id: str,
) -> Tuple[List[str], str, str]:
    """Read the input G_base: (genes deduplicated and sorted, scGPT_model_name of its config, G_base directory)."""
    g_base_dir = get_full_path(
        "eval", "eval_datasets", "st_eval_data",
        f"eval_st_{training_dataset_id}", "G_base", G_base_id, as_str=True,
    )

    genes_path = os.path.join(g_base_dir, "G_base_genes.json")
    if not os.path.isfile(genes_path):
        raise FileNotFoundError(
            f"[ERROR] G_base_genes.json not found at '{genes_path}'. "
            f"Did you run _11_Gbase.py for --G_base_id '{G_base_id}'?"
        )
    with open(genes_path, "r") as f:
        genes_payload = json.load(f)
    if "genes" not in genes_payload or not isinstance(genes_payload["genes"], list):
        raise ValueError(
            f"[ERROR] '{genes_path}' is missing a list under the 'genes' key."
        )
    raw_genes = [str(g) for g in genes_payload["genes"]]
    if len(raw_genes) == 0:
        raise ValueError(f"[ERROR] G_base '{G_base_id}' contains zero genes.")
    sorted_genes = sorted(set(raw_genes))
    if len(sorted_genes) != len(raw_genes):
        n_dup = len(raw_genes) - len(sorted_genes)
        logger.warning(
            "G_base '%s' has %d duplicate gene names; deduplicated to %d.",
            G_base_id, n_dup, len(sorted_genes),
        )

    config_path = os.path.join(g_base_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"[ERROR] G_base config.json not found at '{config_path}'."
        )
    with open(config_path, "r") as f:
        g_base_config = json.load(f)
    scgpt_model_name = g_base_config.get("scGPT_model_name")
    if not isinstance(scgpt_model_name, str) or not scgpt_model_name:
        raise ValueError(
            f"[ERROR] G_base config.json '{config_path}' is missing a non-empty "
            f"'scGPT_model_name' field."
        )

    return sorted_genes, scgpt_model_name, g_base_dir



# ---------------------------------------------------------------------------
# Per-slide stats
# ---------------------------------------------------------------------------

class SlideStats(NamedTuple):
    tau_per_gene: Optional[pd.Series]            # indexed by gene name (G_base subset)
    moran_corr_per_gene: Optional[pd.Series]     # indexed by gene name (G_base subset)
    n_cells_train: int
    panel_size: int                              # |G_base ∩ slide panel|
    typed_ratio: float



def _per_slide_stats(
    slide_id: str,
    adata_path: str,
    train_cell_ids: np.ndarray,
    g_base_set: set,
    *,
    ct_cat_id: str,
    cell_id_col: str,
    cell_type_col: str,
    graph_key: str,
    gene_col: Optional[str],
    target_sum: float,
    min_cells_per_ct: int,
    min_cells_moran: int,
) -> SlideStats:
    """
    Per-slide Tau and bias-corrected Moran's I of one TRAIN slide, on G_base ∩ slide panel. A score is None if it
    cannot be computed (Tau: < 2 cell types kept; Moran's I: < min_cells_moran TRAIN cells); panel_size == 0 if the
    slide has no G_base gene.
    """
    if not os.path.isfile(adata_path):
        raise FileNotFoundError(f"[ERROR] Missing AnnData file: '{adata_path}'")

    adata = sc.read_h5ad(adata_path)
    adata = _subset_adata_to_cell_ids(adata, train_cell_ids, cell_id_col=cell_id_col)
    n_cells_train = int(adata.n_obs)
    if n_cells_train == 0:
        raise ValueError(
            f"[ERROR] Slide '{slide_id}' has 0 cells after filtering to TRAIN cell "
            f"IDs. Check consistency between adata.obs['{cell_id_col}'] and "
            f"cell_ids_sets.npz."
        )

    _assert_nonnegative_counts(adata, slide_id=slide_id)

    if graph_key not in adata.obsp:
        raise KeyError(
            f"[ERROR] graph_key='{graph_key}' not in adata.obsp for slide "
            f"'{slide_id}': {list(adata.obsp.keys())}"
        )

    # Cell-type labels from cID2cT, joined on the Xenium string cell_id
    cid2ct = _load_cid2ct_map(ct_cat_id, slide_id)
    adata.obs[cell_type_col] = (
        adata.obs[cell_id_col].astype(str).map(cid2ct).astype("object")
    )
    # Typed cells: same null-label filter as _ct_expression_sums
    ct_series = adata.obs[cell_type_col]
    ct_str_lower = ct_series.astype("string").str.strip().str.lower()
    non_null_mask = ct_series.notna() & ~ct_str_lower.isin(_NULL_LABELS)
    n_typed = int(non_null_mask.sum())
    if n_typed == 0:
        raise ValueError(
            f"[ERROR] Slide '{slide_id}': no TRAIN cell received a cell type from "
            f"cID2cT (ct_cat_id='{ct_cat_id}'). Check cell_id consistency."
        )
    typed_ratio = float(n_typed) / float(n_cells_train)
    # Raise on a cell_id format mismatch, warn on smaller annotation gaps
    if typed_ratio < 0.5:
        raise ValueError(
            f"[ERROR] Slide '{slide_id}': only {typed_ratio:.1%} of TRAIN cells "
            f"received a cell-type label (n_typed={n_typed} / "
            f"n_train={n_cells_train}). Almost certainly a cell_id format mismatch "
            f"with cID2cT/{ct_cat_id}/."
        )
    if typed_ratio < 0.9:
        logger.warning(
            "Slide '%s': only %.1f%% of TRAIN cells typed; per-slide Tau computed "
            "on a biased subset.",
            slide_id, typed_ratio * 100.0,
        )

    # G_base ∩ slide panel.
    gene_names = _get_gene_names(adata, gene_col=gene_col)
    g_base_mask = np.fromiter(
        (g in g_base_set for g in gene_names),
        dtype=bool, count=len(gene_names),
    )
    panel_size = int(g_base_mask.sum())
    if panel_size == 0:
        return SlideStats(None, None, n_cells_train, 0, typed_ratio)
    gene_names_gb = gene_names[g_base_mask]

    # TRAIN-cell-induced Delaunay subgraph (only TRAIN-TRAIN edges of adata.obsp[graph_key])
    W = adata.obsp[graph_key]
    if not sparse.issparse(W):
        W = sparse.csr_matrix(np.asarray(W))
    W = W.tocsr()

    # Depth-normalize over the full panel, then restrict columns to G_base
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(np.asarray(X, dtype=np.float64))
    X_gb = X.tocsr()[:, g_base_mask]

    # Per-slide bias-corrected Moran's I, when the slide is large enough.
    if n_cells_train < min_cells_moran:
        logger.info(
            "Slide '%s' has %d TRAIN cells (< min_train_cells_per_slide_moran=%d); "
            "Moran's I skipped (slide still contributes per-slide Tau).",
            slide_id, n_cells_train, min_cells_moran,
        )
        moran_series: Optional[pd.Series] = None
    else:
        morans_i_raw = _morans_i_all_genes(X_gb, W)
        morans_i_corr = morans_i_raw + 1.0 / float(n_cells_train - 1)
        moran_series = pd.Series(
            morans_i_corr, index=pd.Index(gene_names_gb, name="gene"),
            dtype=np.float64,
        )

    # Per-slide Tau: per-(slide, CT) cell-count gate, then Tau on the CT-mean matrix.
    ct_sums = _ct_expression_sums(X_gb, adata.obs[cell_type_col])
    kept_means: List[np.ndarray] = [
        s_vec / float(n_ct)
        for _, (s_vec, n_ct) in ct_sums.items()
        if n_ct >= min_cells_per_ct
    ]
    if len(kept_means) >= 2:
        M_s = np.vstack(kept_means)
        tau_slide = _tau_from_ct_means(M_s)
        tau_series: Optional[pd.Series] = pd.Series(
            tau_slide, index=pd.Index(gene_names_gb, name="gene"),
            dtype=np.float64,
        )
    else:
        logger.info(
            "Slide '%s' has fewer than 2 cell types with >= %d TRAIN cells; "
            "per-slide Tau skipped (slide still contributes Moran's I).",
            slide_id, min_cells_per_ct,
        )
        tau_series = None

    return SlideStats(
        tau_per_gene=tau_series,
        moran_corr_per_gene=moran_series,
        n_cells_train=n_cells_train,
        panel_size=panel_size,
        typed_ratio=typed_ratio,
    )



# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def _aggregate_to_table(
    per_slide_tau: Dict[str, pd.Series],
    per_slide_moran: Dict[str, pd.Series],
    g_base_genes: List[str],
    min_train_slides: int,
) -> pd.DataFrame:
    """
    Combine per-slide series into the final per-gene table, indexed on the
    alphabetically sorted G_base panel, and apply the min-train-slides NaN-mask.
    """
    gene_index = pd.Index(g_base_genes, name="gene")

    tau_wide = pd.DataFrame(per_slide_tau).reindex(gene_index)
    moran_wide = pd.DataFrame(per_slide_moran).reindex(gene_index)

    tau_median = tau_wide.median(axis=1, skipna=True)
    moran_median = moran_wide.median(axis=1, skipna=True)
    n_slides_tau = tau_wide.notna().sum(axis=1).astype(np.int64)
    n_slides_moran = moran_wide.notna().sum(axis=1).astype(np.int64)

    tau_median = tau_median.where(n_slides_tau >= int(min_train_slides), other=np.nan)
    moran_median = moran_median.where(
        n_slides_moran >= int(min_train_slides), other=np.nan
    )

    return pd.DataFrame(
        {
            "gene": list(g_base_genes),
            "ct_specificity_score": tau_median.to_numpy(dtype=np.float64),
            "spatial_structure_score": moran_median.to_numpy(dtype=np.float64),
            "n_train_slides_tau": n_slides_tau.to_numpy(dtype=np.int64),
            "n_train_slides_moran": n_slides_moran.to_numpy(dtype=np.int64),
            "in_scgpt_vocab": np.ones(len(g_base_genes), dtype=bool),
        }
    )



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:

    # Argument floors
    if args.min_train_slides < 2:
        raise ValueError(
            "[ERROR] --min_train_slides must be >= 2."
        )
    if args.min_cells_per_cell_type < 10:
        raise ValueError(
            "[ERROR] --min_cells_per_cell_type must be >= 10."
        )
    if args.min_train_cells_per_slide_moran < 1:
        raise ValueError("[ERROR] --min_train_cells_per_slide_moran must be >= 1.")

    # Paths
    adata_folder = get_full_path("STHELAR_HistoST", "adata_graph", as_str=True)
    training_folder = get_full_path(
        "train", "training_datasets", args.dataset_id, args.training_dataset_id,
        as_str=True,
    )
    output_folder = get_full_path(
        "eval", "eval_datasets", "st_eval_data",
        f"eval_st_{args.training_dataset_id}", "gene_properties",
        args.gene_property_id, as_str=True,
    )

    # Refuse to overwrite an existing gene_property_id (unless --overwrite)
    main_output = os.path.join(output_folder, "gene_property_table.parquet")
    if os.path.exists(main_output) and not args.overwrite:
        mtime = _dt.datetime.fromtimestamp(
            os.path.getmtime(main_output)
        ).isoformat(timespec="seconds")
        raise FileExistsError(
            f"[ERROR] gene_property_table.parquet already exists at '{main_output}' "
            f"(mtime={mtime}). Pass --overwrite to replace it, or choose a new --gene_property_id."
        )

    os.makedirs(output_folder, exist_ok=True)

    g_base_genes, g_base_scgpt_model_name, g_base_dir = _load_g_base(
        args.training_dataset_id, args.G_base_id,
    )
    g_base_set = set(g_base_genes)

    cell_ids_sets_path = os.path.join(training_folder, "cell_ids_sets.npz")
    slide_ids = _load_slide_ids(training_folder)
    train_only_by_slide, _, _ = _load_cell_ids_sets_npz(cell_ids_sets_path)

    logger.info("dataset_id            = %s", args.dataset_id)
    logger.info("training_dataset_id   = %s", args.training_dataset_id)
    logger.info("G_base_id             = %s", args.G_base_id)
    logger.info("gene_property_id      = %s", args.gene_property_id)
    logger.info("ct_cat_id             = %s", args.ct_cat_id)
    logger.info(
        "scGPT_model_name      = %s  (forwarded from G_base)",
        g_base_scgpt_model_name,
    )
    logger.info("#slide_ids listed     = %d", len(slide_ids))
    logger.info("#G_base genes         = %d", len(g_base_genes))
    logger.info("output_folder         = %s", output_folder)

    per_slide_tau: Dict[str, pd.Series] = {}
    per_slide_moran: Dict[str, pd.Series] = {}
    used_slide_ids: List[str] = []
    panel_sizes: Dict[str, int] = {}
    typed_ratios: Dict[str, float] = {}

    for slide_id in slide_ids:
        logger.info("========== Processing slide: %s ==========", slide_id)

        train_cell_ids = np.asarray(
            train_only_by_slide.get(slide_id, np.array([], dtype=np.int64))
        )
        if train_cell_ids.size == 0:
            logger.info("Slide '%s' has no TRAIN cells. Skipping.", slide_id)
            continue

        adata_path = os.path.join(adata_folder, f"{slide_id}.h5ad")
        stats = _per_slide_stats(
            slide_id=slide_id,
            adata_path=adata_path,
            train_cell_ids=train_cell_ids,
            g_base_set=g_base_set,
            ct_cat_id=args.ct_cat_id,
            cell_id_col=args.cell_id_col,
            cell_type_col=args.cell_type_col,
            graph_key=args.graph_key,
            gene_col=args.gene_col,
            target_sum=args.target_sum,
            min_cells_per_ct=args.min_cells_per_cell_type,
            min_cells_moran=args.min_train_cells_per_slide_moran,
        )

        if stats.panel_size == 0:
            logger.warning(
                "Slide '%s' has 0 G_base genes in its panel. Skipping.", slide_id,
            )
            continue

        if stats.tau_per_gene is not None:
            per_slide_tau[slide_id] = stats.tau_per_gene
        if stats.moran_corr_per_gene is not None:
            per_slide_moran[slide_id] = stats.moran_corr_per_gene
        used_slide_ids.append(slide_id)
        panel_sizes[slide_id] = stats.panel_size
        typed_ratios[slide_id] = stats.typed_ratio

        logger.info(
            "train_cells=%d | G_base on slide=%d | typed=%.1f%% | tau_ok=%s | moran_ok=%s",
            stats.n_cells_train, stats.panel_size, stats.typed_ratio * 100.0,
            stats.tau_per_gene is not None, stats.moran_corr_per_gene is not None,
        )

    if len(used_slide_ids) == 0:
        raise RuntimeError(
            "[ERROR] No TRAIN slide contributed. Cannot build the gene-property table."
        )

    table = _aggregate_to_table(
        per_slide_tau, per_slide_moran, g_base_genes, args.min_train_slides,
    )

    config_payload = {
        "dataset_id": args.dataset_id,
        "training_dataset_id": args.training_dataset_id,
        "gene_property_id": args.gene_property_id,
        "G_base_id": args.G_base_id,
        "G_base_dir": g_base_dir,
        "scGPT_model_name": g_base_scgpt_model_name,
        "ct_cat_id": args.ct_cat_id,
        "adata_folder": adata_folder,
        "training_folder": training_folder,
        "cell_ids_sets_path": cell_ids_sets_path,
        "cell_id_col": args.cell_id_col,
        "cell_type_col": args.cell_type_col,
        "graph_key": args.graph_key,
        "gene_col": args.gene_col,
        "n_g_base_genes": len(g_base_genes),
        "n_slide_ids_listed": len(slide_ids),
        "n_training_slides_used": len(used_slide_ids),
        "training_slide_ids_used": used_slide_ids,
        "panel_sizes_per_slide": {sid: int(v) for sid, v in panel_sizes.items()},
        "typed_ratio_per_slide": {sid: float(v) for sid, v in typed_ratios.items()},
        "parameters": {
            "target_sum": args.target_sum,
            "min_cells_per_cell_type": args.min_cells_per_cell_type,
            "min_train_slides": args.min_train_slides,
            "min_train_cells_per_slide_moran": args.min_train_cells_per_slide_moran,
        },
        "recipe": {
            "genes": "restricted to the input G_base panel (vocab-filtered upstream by _11_Gbase.py)",
            "expression": "normalize_total + log1p on TRAIN cells only (depth-normalized log1p)",
            "ct_specificity_score": (
                "median across slides of per-slide Tau, computed on a per-(slide, CT) "
                "cell-count-gated CT-mean matrix; NaN if fewer than --min_train_slides "
                "finite per-slide Tau values are available for the gene"
            ),
            "spatial_structure_score": (
                "median across slides of per-slide bias-corrected Moran's I "
                "(I_corr = I_raw + 1/(n-1)) on the TRAIN-cell-induced Delaunay subgraph "
                f"adata.obsp['{args.graph_key}']; per-slide values kept only when the "
                "slide has at least --min_train_cells_per_slide_moran TRAIN cells; "
                "NaN if fewer than --min_train_slides finite per-slide I_corr values "
                "are available for the gene"
            ),
            "in_scgpt_vocab": (
                "always True (G_base genes are vocab-filtered upstream by _11_Gbase.py); "
                "read by protocols.py::persist_gene_strata"
            ),
            "cell_types": (
                f"read from cID2cT/{args.ct_cat_id}/cID2cT_<slide_id>.parquet "
                "(annotation source of truth)"
            ),
        },
    }

    _save_parquet_atomic(main_output, table)
    _save_json_atomic(os.path.join(output_folder, "config.json"), config_payload)

    n_with_ct = int(np.isfinite(table["ct_specificity_score"]).sum())
    n_with_sp = int(np.isfinite(table["spatial_structure_score"]).sum())
    logger.info("==================== DONE ====================")
    logger.info("Gene-property table: %s", main_output)
    logger.info(
        "#genes=%d  with_ct_specificity=%d  with_spatial_structure=%d  slides_used=%d",
        len(table), n_with_ct, n_with_sp, len(used_slide_ids),
    )




if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build the per-gene property table of the Task A.1 gene strata (G_base genes, TRAIN cells only)."
    )

    parser.add_argument("--dataset_id", type=str,
                        default="ds_ct1_20x_224_o112",
                        help="Dataset ID, e.g. ds_ct1_20x_224_o112.")

    parser.add_argument("--training_dataset_id", type=str,
                        default="ds_ct1_20x_224_o112_1",
                        help="Training dataset ID, e.g. ds_ct1_20x_224_o112_1.")

    parser.add_argument("--G_base_id", type=str,
                        default="G_base_1",
                        help="Identifier of the input G_base panel (built by _11_Gbase.py).")

    parser.add_argument("--gene_property_id", type=str,
                        default="gene_property_1",
                        help="Identifier of this gene-property table build.")

    parser.add_argument("--ct_cat_id", type=str,
                        default=DEFAULT_CT_CAT_ID,
                        help="Cell-type category id; selects cID2cT/<ct_cat_id>/ "
                        "(e.g. ct_1, consistent with the dataset_id).")

    parser.add_argument("--cell_id_col", type=str,
                        default=DEFAULT_CELL_ID_COL,
                        help="Column in adata.obs containing cell IDs.")

    parser.add_argument("--cell_type_col", type=str,
                        default=DEFAULT_CELL_TYPE_COL,
                        help="adata.obs column populated with the cell-type labels "
                        "loaded from cID2cT and used for the Tau index.")

    parser.add_argument("--graph_key", type=str,
                        default=DEFAULT_GRAPH_KEY,
                        help="Key in adata.obsp holding the spatial (Delaunay) graph "
                        "for Moran's I.")

    parser.add_argument("--gene_col", type=str,
                        default=None,
                        help="Optional column in adata.var with gene names. If "
                        "omitted, adata.var_names are used.")

    parser.add_argument("--target_sum", type=float,
                        default=1e4,
                        help="Target sum for scanpy normalize_total before log1p.")

    parser.add_argument("--min_cells_per_cell_type", type=int,
                        default=50,
                        help="Minimum TRAIN cells of a cell type on a slide for it to contribute a per-slide mean.")

    parser.add_argument("--min_train_slides", type=int,
                        default=3,
                        help="Minimum number of TRAIN slides with a finite per-slide "
                        "score for a gene; below this the aggregated score is NaN. "
                        "Applied independently to Tau and Moran's I.")

    parser.add_argument("--min_train_cells_per_slide_moran", type=int,
                        default=200,
                        help="Minimum TRAIN cells on a slide for it to contribute a per-slide Moran's I (Tau is not affected).")

    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing gene_property_id folder.")

    args = parser.parse_args()
    main(args)
