"""
Build the G_base gene panel for evaluation from TRAIN cells only: scGPT-vocab genes that pass the hard filters
(shared and supported on enough training slides), ranked by a weighted score; the top k_selected are kept.

Saved in <ROOT_HistoST> / eval / eval_datasets / st_eval_data / eval_st_<training_dataset_id> / G_base / <G_base_id> /
    - G_base_genes.json, G_base_genes.txt: selected genes
    - gene_scores.csv, selected_gene_scores.csv: gene metrics and scores
    - slide_summary.csv, per_slide_gene_metrics_long.csv: per-slide summary and gene metrics
    - config.json, summary.json: parameters and run summary
"""

import sys
from pathlib import Path
import argparse
import logging
import os
import json
import datetime as _dt
from typing import Dict, List, Optional, Sequence, Tuple
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
KEY_HIGHLY_VARIABLE = "highly_variable"



def _xenium_str_ids_to_int64(cell_id_strs: pd.Series) -> Optional[np.ndarray]:
    """
    Vectorized conversion of Xenium Explorer string cell IDs (e.g. 'aaaachba-1')
    into int64 IDs (e.g. 10001). Returns None if format does not match.
    """
    if cell_id_strs.empty:
        return np.empty((0,), dtype=np.int64)

    s = cell_id_strs.astype(str).to_numpy()
    b = s.astype("S")

    # Check the "-1" suffix on every row
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
    """
    Load output of cell_ids_sets.py (npz) and return:
      train_only_by_slide, valid_only_by_slide, test_only_by_slide
    """
    if not os.path.isfile(cell_ids_sets_path):
        raise FileNotFoundError(
            f"[ERROR] cell_ids_sets file not found: '{cell_ids_sets_path}'."
        )

    npz = np.load(cell_ids_sets_path, allow_pickle=True)

    def _get_dict(name: str) -> Dict[str, np.ndarray]:
        if name not in npz.files:
            return {}
        obj = npz[name]
        try:
            return obj.item()
        except Exception:
            return dict(obj)

    train_only = _get_dict("train_only_by_slide")
    valid_only = _get_dict("valid_only_by_slide")
    test_only = _get_dict("test_only_by_slide")
    return train_only, valid_only, test_only



def _standardize_cell_ids_for_isin(
    obs_cell_ids: pd.Series,
    target_cell_ids: np.ndarray,
) -> Tuple[pd.Series, np.ndarray]:
    """
    Make obs_cell_ids and target_cell_ids comparable for membership testing.
    Prefer int64 comparison when possible; otherwise fall back to string comparison.
    """
    tgt_arr = np.asarray(target_cell_ids)

    # Fast numeric case
    obs_num = pd.to_numeric(obs_cell_ids, errors="coerce")
    if not obs_num.isna().any():
        try:
            tgt_int = tgt_arr.astype(np.int64, copy=False)
        except Exception:
            tgt_int = tgt_arr.astype(np.int64)
        return obs_num.astype(np.int64), tgt_int

    # Xenium string obs + integer target
    if tgt_arr.dtype.kind in ("i", "u"):
        obs_int = _xenium_str_ids_to_int64(obs_cell_ids)
        if obs_int is not None:
            return (
                pd.Series(obs_int, index=obs_cell_ids.index, dtype=np.int64),
                tgt_arr.astype(np.int64, copy=False),
            )

    # Fallback: compare as strings
    return obs_cell_ids.astype(str), tgt_arr.astype(str)



def _subset_adata_to_cell_ids(
    adata: AnnData,
    cell_ids: np.ndarray,
    cell_id_col: str = DEFAULT_CELL_ID_COL,
) -> AnnData:
    """
    Return a copy of adata restricted to obs[cell_id_col] in cell_ids.
    """
    if cell_id_col not in adata.obs.columns:
        raise KeyError(f"[ERROR] '{cell_id_col}' not found in adata.obs columns: {list(adata.obs.columns)}")
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
    """
    Return the gene names to use.
    - If gene_col is None: use adata.var_names
    - Else: use adata.var[gene_col]
    """
    if gene_col is None:
        gene_names = adata.var_names.astype(str).to_numpy()
    else:
        if gene_col not in adata.var.columns:
            raise KeyError(f"[ERROR] gene_col='{gene_col}' not found in adata.var.columns: {list(adata.var.columns)}")
        gene_names = adata.var[gene_col].astype(str).to_numpy()

    idx = pd.Index(gene_names)
    if idx.has_duplicates:
        duplicated = idx[idx.duplicated()].unique().tolist()
        raise ValueError(f"[ERROR] Gene names contain duplicates (showing up to 10): {duplicated[:10]}.")
    return gene_names



def _load_scgpt_vocab(scgpt_model_name: str) -> Tuple[set, str]:
    """
    Load scGPT vocab.json from:
    get_full_path('train', 'pretrained_foundation_models', scgpt_model_name)
    """
    model_dir = get_full_path(
        "train", "pretrained_foundation_models", scgpt_model_name, as_str=True
    )
    vocab_path = os.path.join(model_dir, "vocab.json")
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"[ERROR] scGPT vocab.json not found: '{vocab_path}'")

    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    # Remove obvious special tokens if present
    vocab_genes = set()
    for tok in vocab.keys():
        if isinstance(tok, str) and tok.startswith("<") and tok.endswith(">"):
            continue
        vocab_genes.add(str(tok))

    return vocab_genes, model_dir



def _assert_nonnegative_counts(adata: AnnData, slide_id: str) -> None:
    """Check that adata.X holds raw counts (non-negative integers); normalize_total + log1p is applied afterwards."""
    x_min = adata.X.min()
    if sparse.issparse(adata.X):
        x_min = float(x_min)
    if x_min < 0:
        raise ValueError(
            f"[ERROR] Negative values found in adata.X for slide '{slide_id}'. "
            "Expected raw non-negative counts."
        )

    # Integer-likeness check on stored values
    if sparse.issparse(adata.X):
        data = np.asarray(adata.X.data)
    else:
        data = np.asarray(adata.X).ravel()
    if data.size > 0 and not np.allclose(data, np.round(data), atol=1e-6):
        raise ValueError(
            f"[ERROR] adata.X for slide '{slide_id}' contains non-integer values; "
            "this script expects raw counts (it then applies normalize_total + log1p). "
            "If adata.X was already normalized, rebuild adata_graph from raw counts."
        )



def _nonzero_rate_per_gene(X) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-gene:
    - nnz count across cells
    - nnz fraction across cells
    """
    if sparse.issparse(X):
        nnz = np.asarray(X.getnnz(axis=0)).ravel().astype(np.int64)
    else:
        nnz = np.asarray((np.asarray(X) != 0).sum(axis=0)).ravel().astype(np.int64)

    n_cells = X.shape[0]
    rates = nnz.astype(np.float64) / max(n_cells, 1)
    return nnz, rates



def _compute_hvg_topk_mask(
    adata: AnnData,
    hvg_top_k: int,
    hvg_flavor: str = "seurat",
) -> np.ndarray:
    """Boolean mask of the top-K HVGs of adata (adata.X normalized + log1p for flavor='seurat')."""
    n_vars = adata.n_vars
    mask = np.zeros(n_vars, dtype=bool)

    if hvg_top_k <= 0 or n_vars == 0:
        return mask
    if n_vars <= hvg_top_k:
        mask[:] = True
        return mask
    if adata.n_obs < 2:
        return mask

    try:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=min(hvg_top_k, n_vars),
            flavor=hvg_flavor,
            inplace=True,
            subset=False,
        )
        if KEY_HIGHLY_VARIABLE in adata.var.columns:
            return adata.var[KEY_HIGHLY_VARIABLE].to_numpy(dtype=bool)
    except Exception as e:
        logger.warning(
            "HVG computation failed with flavor='%s' (reason: %s). Falling back to "
            "variance ranking.",
            hvg_flavor, e,
        )

    # Fallback: top-K variance on current transformed matrix
    X = adata.X
    if sparse.issparse(X):
        means = np.asarray(X.mean(axis=0)).ravel()
        sq_means = np.asarray(X.power(2).mean(axis=0)).ravel()
    else:
        X_arr = np.asarray(X)
        means = X_arr.mean(axis=0)
        sq_means = np.square(X_arr).mean(axis=0)

    var = np.clip(sq_means - np.square(means), a_min=0.0, a_max=None)
    idx = np.argpartition(var, -hvg_top_k)[-hvg_top_k:]
    mask[idx] = True
    return mask



def _robust_dispersion_nonzero_only(
    X,
    q_low: float = 0.10,
    q_high: float = 0.90,
) -> np.ndarray:
    """Per-gene robust dispersion on the nonzero normalized/log1p values: quantile(q_high) - quantile(q_low)."""
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(f"[ERROR] Invalid quantiles: q_low={q_low}, q_high={q_high}. Need 0 <= q_low < q_high <= 1.")

    n_vars = X.shape[1]
    out = np.zeros(n_vars, dtype=np.float32)

    if sparse.issparse(X):
        X_csc = X.tocsc()
        indptr = X_csc.indptr
        data = X_csc.data
        for j in range(n_vars):
            vals = data[indptr[j] : indptr[j + 1]]
            if vals.size <= 1:
                out[j] = 0.0
            else:
                q1, q2 = np.quantile(vals, [q_low, q_high])
                out[j] = float(q2 - q1)
        return out

    X_arr = np.asarray(X)
    for j in range(n_vars):
        vals = X_arr[:, j]
        vals = vals[vals > 0]
        if vals.size <= 1:
            out[j] = 0.0
        else:
            q1, q2 = np.quantile(vals, [q_low, q_high])
            out[j] = float(q2 - q1)
    return out



def _padded_median(values: Sequence[float], total_length: int) -> float:
    """Median after zero-padding up to total_length (slides where the gene is not measured count as 0)."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size > total_length:
        raise ValueError(
            f"[ERROR] Cannot pad median: got {arr.size} values for total_length={total_length}."
        )
    if arr.size < total_length:
        arr = np.concatenate(
            [arr, np.zeros(total_length - arr.size, dtype=np.float32)], axis=0
        )
    if arr.size == 0:
        return 0.0
    return float(np.median(arr))



def _pct_rank_desc(series: pd.Series) -> pd.Series:
    """
    Percentile rank in [0,1], where larger input -> larger rank.
    """
    if len(series) == 1:
        return pd.Series([1.0], index=series.index, dtype=np.float64)
    return series.rank(method="average", ascending=True, pct=True).astype(np.float64)



def _save_json_atomic(path: str, payload: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)



def _save_text_atomic(path: str, lines: Sequence[str]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")
    os.replace(tmp_path, path)



def _save_csv_atomic(path: str, df: pd.DataFrame) -> None:
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)



def main(args: argparse.Namespace) -> None:
    
    # Check arguments
    if args.min_shared_slides < 1:
        raise ValueError("[ERROR] --min_shared_slides must be >= 1.")
    if args.min_support_slides < 1:
        raise ValueError("[ERROR] --min_support_slides must be >= 1.")
    if args.k_selected < 1:
        raise ValueError("[ERROR] --k_selected must be >= 1.")
    if not (0.0 <= args.min_nonzero_rate <= 1.0):
        raise ValueError("[ERROR] --min_nonzero_rate must be in [0, 1].")

    hvg_flavor = "seurat"  # need normalized + log1p data

    # Paths
    adata_folder = get_full_path("STHELAR_HistoST", "adata_graph", as_str=True)
    training_folder = get_full_path("train", "training_datasets", args.dataset_id, args.training_dataset_id, as_str=True)
    output_folder = get_full_path("eval","eval_datasets","st_eval_data",f"eval_st_{args.training_dataset_id}","G_base",args.G_base_id, as_str=True)

    # Refuse to overwrite an existing G_base_id (unless --overwrite)
    main_output = os.path.join(output_folder, "G_base_genes.json")
    if os.path.exists(main_output) and not args.overwrite:
        mtime = _dt.datetime.fromtimestamp(os.path.getmtime(main_output)).isoformat(timespec="seconds")
        raise FileExistsError(
            f"[ERROR] G_base_genes.json already exists at '{main_output}' "
            f"(mtime={mtime}). Pass --overwrite to replace it, or choose a new --G_base_id."
        )

    os.makedirs(output_folder, exist_ok=True)

    # Data loading
    cell_ids_sets_path = os.path.join(training_folder, "cell_ids_sets.npz")
    slide_ids = _load_slide_ids(training_folder)
    train_only_by_slide, _, _ = _load_cell_ids_sets_npz(cell_ids_sets_path)
    scgpt_vocab, scgpt_model_dir = _load_scgpt_vocab(args.scGPT_model_name)

    logger.info("dataset_id            = %s", args.dataset_id)
    logger.info("training_dataset_id   = %s", args.training_dataset_id)
    logger.info("G_base_id             = %s", args.G_base_id)
    logger.info("scGPT_model_name      = %s", args.scGPT_model_name)
    logger.info("#slide_ids listed     = %d", len(slide_ids))
    logger.info("scGPT vocab size      = %d", len(scgpt_vocab))
    logger.info("output_folder         = %s", output_folder)

    long_rows: List[pd.DataFrame] = []
    slide_summary_rows: List[dict] = []
    used_slide_ids: List[str] = []

    # Processing each slide
    for slide_id in slide_ids:
        logger.info("========== Processing slide: %s ==========", slide_id)

        # TRAIN cells only (slides without TRAIN cells are skipped)
        train_cell_ids = np.asarray(
            train_only_by_slide.get(slide_id, np.array([], dtype=np.int64))
        )
        if train_cell_ids.size == 0:
            logger.info(
                "Slide '%s' has no TRAIN cells in cell_ids_sets.npz. Skipping entirely.",
                slide_id,
            )
            slide_summary_rows.append(
                {
                    "slide_id": slide_id,
                    "used_for_G_base": 0,
                    "reason": "no_train_cells",
                    "n_cells_total": np.nan,
                    "n_cells_train": 0,
                    "n_genes_total": np.nan,
                    "n_vocab_genes_on_slide": 0,
                }
            )
            continue

        adata_path = os.path.join(adata_folder, f"{slide_id}.h5ad")
        if not os.path.isfile(adata_path):
            raise FileNotFoundError(f"[ERROR] Missing AnnData file: '{adata_path}'")

        adata = sc.read_h5ad(adata_path)
        n_cells_total = int(adata.n_obs)
        n_genes_total = int(adata.n_vars)

        adata = _subset_adata_to_cell_ids(adata, train_cell_ids, cell_id_col=args.cell_id_col)
        n_cells_train = int(adata.n_obs)

        if n_cells_train == 0:
            raise ValueError(
                f"[ERROR] Slide '{slide_id}' has 0 cells after filtering to TRAIN cell IDs. "
                f"Check consistency between adata.obs['{args.cell_id_col}'] and cell_ids_sets.npz."
            )

        _assert_nonnegative_counts(adata, slide_id=slide_id)

        gene_names = _get_gene_names(adata, gene_col=args.gene_col)
        vocab_mask = np.array([g in scgpt_vocab for g in gene_names], dtype=bool)
        n_vocab_genes = int(vocab_mask.sum())

        slide_summary_rows.append(
            {
                "slide_id": slide_id,
                "used_for_G_base": 1 if n_vocab_genes > 0 else 0,
                "reason": "ok" if n_vocab_genes > 0 else "no_scgpt_vocab_genes",
                "n_cells_total": n_cells_total,
                "n_cells_train": n_cells_train,
                "n_genes_total": n_genes_total,
                "n_vocab_genes_on_slide": n_vocab_genes,
            }
        )

        if n_vocab_genes == 0:
            logger.warning(
                "Slide '%s' has 0 genes in the selected scGPT vocab. Skipping this slide.",
                slide_id,
            )
            continue

        used_slide_ids.append(slide_id)

        # Metrics on raw counts (TRAIN cells only) --------
        X_raw_vocab = adata.X[:, vocab_mask]
        nnz, nonzero_rate = _nonzero_rate_per_gene(X_raw_vocab)
        any_nonzero = nnz > 0
        support = nonzero_rate >= args.min_nonzero_rate

        # Metrics on normalized/log1p data (TRAIN cells only) --------
        sc.pp.normalize_total(adata, target_sum=args.target_sum)
        sc.pp.log1p(adata)
        adata_vocab = adata[:, vocab_mask].copy()

        hvg_topk_mask = _compute_hvg_topk_mask(
            adata_vocab,
            hvg_top_k=args.hvg_top_k,
            hvg_flavor=hvg_flavor,
        )
        robust_dispersion = _robust_dispersion_nonzero_only(
            adata_vocab.X,
            q_low=args.dispersion_q_low,
            q_high=args.dispersion_q_high,
        )

        gene_names_vocab = gene_names[vocab_mask]

        # "measured_in_slide" = present in panel on that slide and in scGPT vocab
        # "any_nonzero" = at least one TRAIN cell expresses it
        # "support_slide" = TRAIN-cell nonzero rate >= threshold
        df_slide = pd.DataFrame(
            {
                "slide_id": slide_id,
                "gene": gene_names_vocab,
                "measured_in_slide": np.ones_like(any_nonzero, dtype=np.int8),
                "any_nonzero": any_nonzero.astype(np.int8),
                "support_slide": support.astype(np.int8),
                "hvg_topk_slide": hvg_topk_mask.astype(np.int8),
                "nonzero_rate": nonzero_rate.astype(np.float32),
                "robust_dispersion": robust_dispersion.astype(np.float32),
            }
        )
        long_rows.append(df_slide)

        logger.info(
            "TRAIN cells kept: %d / %d | genes on slide: %d | genes in scGPT vocab: %d",
            n_cells_train, n_cells_total, n_genes_total, n_vocab_genes,
        )
    
    # Then aggregate across slides and apply filters/ranking to select G_base genes

    if len(used_slide_ids) == 0:
        raise RuntimeError("[ERROR] No training slide with both TRAIN cells and scGPT-vocab genes was available. Cannot build G_base.")

    n_training_slides_used = len(used_slide_ids)
    logger.info("Number of training slides used for G_base: %d", n_training_slides_used)

    if args.min_shared_slides > n_training_slides_used:
        raise ValueError(f"[ERROR] --min_shared_slides={args.min_shared_slides} exceeds the number of training slides actually used ({n_training_slides_used}).")
    if args.min_support_slides > n_training_slides_used:
        raise ValueError(f"[ERROR] --min_support_slides={args.min_support_slides} exceeds the number of training slides actually used ({n_training_slides_used}).")

    long_df = pd.concat(long_rows, axis=0, ignore_index=True)
    group = long_df.groupby("gene", sort=False)

    summary_df = group.agg(
        n_shared_training_slides=("measured_in_slide", "sum"),
        n_nonzero_any_slides=("any_nonzero", "sum"),
        n_support_slides=("support_slide", "sum"),
        n_hvg_topk_slides=("hvg_topk_slide", "sum"),
    ).astype(np.int32)

    # Medians across all used training slides, with zero-padding for missing slides
    median_nonzero = {}
    median_dispersion = {}
    for gene, sub_df in group:
        median_nonzero[gene] = _padded_median(
            sub_df["nonzero_rate"].to_numpy(dtype=np.float32),
            total_length=n_training_slides_used,
        )
        median_dispersion[gene] = _padded_median(
            sub_df["robust_dispersion"].to_numpy(dtype=np.float32),
            total_length=n_training_slides_used,
        )

    summary_df["median_nonzero_rate_all_train_slides"] = pd.Series(median_nonzero)
    summary_df["median_robust_dispersion_all_train_slides"] = pd.Series(median_dispersion)

    summary_df["shared_training_slide_fraction"] = (
        summary_df["n_shared_training_slides"] / float(n_training_slides_used)
    )
    summary_df["support_slide_fraction"] = (
        summary_df["n_support_slides"] / float(n_training_slides_used)
    )
    summary_df["hvg_topk_slide_fraction"] = (
        summary_df["n_hvg_topk_slides"] / float(n_training_slides_used)
    )

    # ---------------- Hard filters ----------------
    candidates_df = summary_df.loc[
        (summary_df["n_shared_training_slides"] >= args.min_shared_slides)
        & (summary_df["n_support_slides"] >= args.min_support_slides)
    ].copy()

    if len(candidates_df) == 0:
        preview = summary_df.sort_values(
            by=[
                "n_shared_training_slides",
                "n_support_slides",
                "median_nonzero_rate_all_train_slides",
                "n_hvg_topk_slides",
            ],
            ascending=[False, False, False, False],
        ).head(20)

        preview_path = os.path.join(output_folder, "hard_filter_failed_preview.csv")
        _save_csv_atomic(
            preview_path,
            preview.reset_index().rename(columns={"index": "gene"}),
        )
        raise RuntimeError(
            "[ERROR] No gene passed the hard filters. "
            f"A preview of the best genes before filtering was saved to: {preview_path}"
        )

    # ---------------- Ranking / scoring ----------------
    # Weighted sum of percentile ranks
    candidates_df["rank_shared_slides"] = _pct_rank_desc(candidates_df["n_shared_training_slides"])
    candidates_df["rank_hvg_topk_slides"] = _pct_rank_desc(candidates_df["n_hvg_topk_slides"])
    candidates_df["rank_median_nonzero_rate"] = _pct_rank_desc(candidates_df["median_nonzero_rate_all_train_slides"])
    candidates_df["rank_median_robust_dispersion"] = _pct_rank_desc(candidates_df["median_robust_dispersion_all_train_slides"])

    candidates_df["score"] = (
        args.w_shared_slides * candidates_df["rank_shared_slides"]
        + args.w_hvg * candidates_df["rank_hvg_topk_slides"]
        + args.w_nonzero * candidates_df["rank_median_nonzero_rate"]
        + args.w_dispersion * candidates_df["rank_median_robust_dispersion"]
    )

    candidates_df = candidates_df.sort_values(
        by=[
            "score",
            "n_shared_training_slides",
            "n_hvg_topk_slides",
            "median_nonzero_rate_all_train_slides",
            "median_robust_dispersion_all_train_slides",
        ],
        ascending=[False, False, False, False, False],
    )

    n_select = min(args.k_selected, len(candidates_df))
    selected_genes = candidates_df.index[:n_select].tolist()

    # ---------------- Save outputs ----------------
    full_scores_df = summary_df.copy()
    full_scores_df["passed_hard_filters"] = 0
    full_scores_df["selected"] = 0

    for col in [
        "rank_shared_slides",
        "rank_hvg_topk_slides",
        "rank_median_nonzero_rate",
        "rank_median_robust_dispersion",
        "score",
    ]:
        full_scores_df[col] = np.nan

    full_scores_df.loc[candidates_df.index, "passed_hard_filters"] = 1
    full_scores_df.loc[
        candidates_df.index,
        [
            "rank_shared_slides",
            "rank_hvg_topk_slides",
            "rank_median_nonzero_rate",
            "rank_median_robust_dispersion",
            "score",
        ],
    ] = candidates_df[
        [
            "rank_shared_slides",
            "rank_hvg_topk_slides",
            "rank_median_nonzero_rate",
            "rank_median_robust_dispersion",
            "score",
        ]
    ]

    full_scores_df.loc[selected_genes, "selected"] = 1
    full_scores_df = (
        full_scores_df.reset_index()
        .rename(columns={"index": "gene"})
        .sort_values(
            by=["selected", "passed_hard_filters", "score", "n_shared_training_slides"],
            ascending=[False, False, False, False],
            na_position="last",
        )
    )

    slide_summary_df = pd.DataFrame(slide_summary_rows).sort_values("slide_id")
    selected_df = candidates_df.head(n_select).reset_index().rename(columns={"index": "gene"})

    config_payload = {
        "dataset_id": args.dataset_id,
        "training_dataset_id": args.training_dataset_id,
        "G_base_id": args.G_base_id,
        "scGPT_model_name": args.scGPT_model_name,
        "scGPT_model_dir": scgpt_model_dir,
        "adata_folder": adata_folder,
        "training_folder": training_folder,
        "cell_ids_sets_path": cell_ids_sets_path,
        "cell_id_col": args.cell_id_col,
        "gene_col": args.gene_col,
        "n_slide_ids_listed": len(slide_ids),
        "n_training_slides_used": n_training_slides_used,
        "training_slide_ids_used": used_slide_ids,
        "selection_recipe": {
            "hard_filters": {
                "in_scgpt_vocab": True,
                "min_shared_slides": args.min_shared_slides,
                "shared_slide_definition": "gene exists in adata.var (or adata.var[gene_col]) on the TRAIN-filtered slide and is in scGPT vocab",
                "min_nonzero_rate": args.min_nonzero_rate,
                "min_support_slides": args.min_support_slides,
                "support_slide_definition": "fraction of TRAIN cells with nonzero counts >= min_nonzero_rate",
            },
            "score_components": {
                "n_shared_training_slides": args.w_shared_slides,
                "n_hvg_topk_slides": args.w_hvg,
                "median_nonzero_rate_all_train_slides": args.w_nonzero,
                "median_robust_dispersion_all_train_slides": args.w_dispersion,
            },
            "hvg": {
                "hvg_top_k": args.hvg_top_k,
                "hvg_flavor": hvg_flavor,
                "normalization_target_sum": args.target_sum,
                "preprocessing": "normalize_total + log1p on TRAIN-only cells",
            },
            "robust_dispersion": {
                "definition": "q_high - q_low on nonzero log1p-normalized expression values only",
                "q_low": args.dispersion_q_low,
                "q_high": args.dispersion_q_high,
            },
            "k_selected": args.k_selected,
        },
    }

    summary_payload = {
        "dataset_id": args.dataset_id,
        "training_dataset_id": args.training_dataset_id,
        "G_base_id": args.G_base_id,
        "scGPT_model_name": args.scGPT_model_name,
        "n_slide_ids_listed": len(slide_ids),
        "n_training_slides_used": n_training_slides_used,
        "n_unique_vocab_genes_seen": int(summary_df.shape[0]),
        "n_candidate_genes_after_hard_filters": int(candidates_df.shape[0]),
        "n_selected_genes": int(len(selected_genes)),
        "selected_genes_preview_top20": selected_genes[:20],
    }

    selected_json = {
        "dataset_id": args.dataset_id,
        "training_dataset_id": args.training_dataset_id,
        "G_base_id": args.G_base_id,
        "scGPT_model_name": args.scGPT_model_name,
        "n_training_slides_used": n_training_slides_used,
        "genes": selected_genes,
    }

    _save_json_atomic(os.path.join(output_folder, "config.json"), config_payload)
    _save_json_atomic(os.path.join(output_folder, "summary.json"), summary_payload)
    _save_json_atomic(os.path.join(output_folder, "G_base_genes.json"), selected_json)
    _save_text_atomic(os.path.join(output_folder, "G_base_genes.txt"), selected_genes)
    _save_csv_atomic(os.path.join(output_folder, "gene_scores.csv"), full_scores_df)
    _save_csv_atomic(os.path.join(output_folder, "selected_gene_scores.csv"), selected_df)
    _save_csv_atomic(os.path.join(output_folder, "slide_summary.csv"), slide_summary_df)
    _save_csv_atomic(os.path.join(output_folder, "per_slide_gene_metrics_long.csv"), long_df)

    logger.info("==================== DONE ====================")
    logger.info("Saved G_base with %d genes", len(selected_genes))
    logger.info("Main file: %s", os.path.join(output_folder, "G_base_genes.json"))
    logger.info("Scores   : %s", os.path.join(output_folder, "gene_scores.csv"))
    logger.info("Config   : %s", os.path.join(output_folder, "config.json"))




if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser( description="Build the G_base gene list for ST evaluation from TRAIN cells only.")

    # Data & output

    parser.add_argument( "--dataset_id", type=str,
                        default="ds_ct1_20x_224_o112",
                        help="Dataset ID, e.g. ds_ct1_20x_224_o112")
    
    parser.add_argument("--training_dataset_id", type=str,
                        default="ds_ct1_20x_224_o112_1",
                        help="Training dataset ID, e.g. ds_ct1_20x_224_o112_1")
    
    parser.add_argument("--G_base_id", type=str,
                        default="G_base_1",
                        help="Identifier of the G_base selection run.")
    
    parser.add_argument("--scGPT_model_name", type=str,
                        default="scGPT_human",
                        choices=["scGPT_human", "scGPT_pancancer"],
                        help="Local scGPT vocab to use (keep it consistent across runs).")

    parser.add_argument("--cell_id_col", type=str,
                        default=DEFAULT_CELL_ID_COL,
                        help="Column in adata.obs containing cell IDs.")
    
    parser.add_argument("--gene_col", type=str,
                        default=None,
                        help="Optional column in adata.var containing gene names. If omitted, adata.var_names are used.")

    parser.add_argument("--k_selected", type=int,
                        default=256,
                        help="Final number of genes to keep in G_base.")
    
    # Selection filters

        # Hard filters

    parser.add_argument("--min_shared_slides", type=int,
                        default=5,
                        help="Hard filter: gene must be present in adata.var and in scGPT vocab on at least this many TRAIN slides.")
    
    parser.add_argument("--min_nonzero_rate", type=float,
                        default=0.05,
                        help="Per-slide support threshold: fraction of TRAIN cells with nonzero counts needed to count the slide as supportive.")
    
    parser.add_argument("--min_support_slides", type=int,
                        default=5,
                        help="Hard filter: gene must reach min_nonzero_rate on at least this many TRAIN slides.")
    
        # Scoring filters (for ranking candidates that pass hard filters)
    
    parser.add_argument("--hvg_top_k", type=int,
                        default=100,
                        help="Top-K HVGs counted per TRAIN slide for the HVG score component.")
    
    parser.add_argument("--target_sum", type=float,
                        default=1e4,
                        help="Target sum for scanpy normalize_total before HVG/dispersion scoring.")
    
    parser.add_argument("--dispersion_q_low", type=float,
                        default=0.10,
                        help="Lower quantile for robust dispersion on nonzero normalized/log1p values.")
    
    parser.add_argument("--dispersion_q_high", type=float,
                        default=0.90,
                        help="Upper quantile for robust dispersion on nonzero normalized/log1p values.")

    # Scoring weights for ranking candidates for final selection (used for scoring filters)
    
    parser.add_argument("--w_shared_slides", type=float,
                        default=2.0,
                        help="Weight of the number of shared training slides in the final score.")
    
    parser.add_argument("--w_hvg", type=float,
                        default=2.0,
                        help="Weight of the number of top-K HVG slides in the final score.")
    
    parser.add_argument("--w_nonzero", type=float,
                        default=1.0,
                          help="Weight of the median nonzero-rate component in the final score.")
    
    parser.add_argument("--w_dispersion", type=float,
                        default=1.0,
                        help="Weight of the median robust-dispersion component in the final score.")

    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing G_base_id folder.")

    args = parser.parse_args()
    main(args)