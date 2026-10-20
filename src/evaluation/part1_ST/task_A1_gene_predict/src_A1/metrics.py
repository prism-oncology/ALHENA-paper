from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness as _sk_trustworthiness
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors

_VAR_EPS = 1e-12

# Noise-ceiling normalization: metric / sqrt(clip(r, _CEILING_EPS, 1)), r = full-depth reliability (reliability.parquet)
_CEILING_EPS = 5e-2

# Neighborhood preservation: kNN overlap (k=15), trustworthiness / continuity (Venna-Kaski); the ceiling is
# the overlap between the two count-thinning halves of Y_true, floored at _NP_CEILING_EPS
_NP_K_LIST = (15,)
_NP_HEADLINE_K = 15
_NP_CEILING_EPS = 5e-2
# Max locations for trustworthiness/continuity (deterministic subsample; kNN overlap uses all locations)
_NP_TC_MAX_POINTS = 3000

# Slide-macro (genes within a slide, then slides) and gene-macro (slides within a gene, then genes) metrics
CORE_BENCHMARK_METRICS = [
    "slide_macro_spearman",
    "gene_macro_spearman",
    "slide_macro_pearson",
    "gene_macro_pearson",
    "slide_macro_mae",
    "slide_macro_rmse",
    "slide_macro_huber",
    "location_macro_profile_spearman",
    # Noise-ceiling-normalized correlation macros
    "slide_macro_spearman_ceiling_norm",
    "gene_macro_spearman_ceiling_norm",
    "slide_macro_pearson_ceiling_norm",
    "gene_macro_pearson_ceiling_norm",
    "location_macro_profile_spearman_ceiling_norm",
]

_NULL_DOMAIN_STRINGS = {"", "na", "nan", "none", "null", "<na>"}

HEST_CV_SUB_EVAL_CONFIGS = {
    "HEST_CV_CLOSE",
    "HEST_CV_MAGNIFICATION",
    "HEST_CV_SPECIES",
    "HEST_ORGANS",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _finite_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _assert_no_nonfinite_predictions(slide_id: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Raise if predictions are NaN/inf on measured targets."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    bad = np.isfinite(y_true) & ~np.isfinite(y_pred)
    if np.any(bad):
        n_bad = int(np.sum(bad))
        raise RuntimeError(
            f"Non-finite predictions detected for slide '{slide_id}' on {n_bad} measured values."
        )


def _df_col_mean(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return np.nan
    return _macro_mean(df[col].tolist())


def _ensure_columns(df: pd.DataFrame, defaults: Mapping[str, object]) -> pd.DataFrame:
    """Add the missing columns with default values."""
    out = df.copy()
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    return out


def _resolve_novae_domain_labels(obs_df: Optional[pd.DataFrame]) -> Optional[np.ndarray]:
    if obs_df is None or obs_df.empty:
        return None
    if "novae_domain" in obs_df.columns:
        s = obs_df["novae_domain"].astype("string")
        s = s.mask(s.isna() | s.str.strip().str.lower().isin(_NULL_DOMAIN_STRINGS), pd.NA)
        return s.to_numpy(dtype=object)
    domain_cols = sorted(
        c
        for c in obs_df.columns
        if c.startswith("novae_patch_domain_") and not c.startswith("novae_patch_domain_frac_")
    )
    if len(domain_cols) == 1:
        s = obs_df[domain_cols[0]].astype("string")
        s = s.mask(s.isna() | s.str.strip().str.lower().isin(_NULL_DOMAIN_STRINGS), pd.NA)
        return s.to_numpy(dtype=object)
    return None


def _valid_novae_label_mask(labels: np.ndarray) -> np.ndarray:
    s = pd.Series(labels, dtype="string")
    return (
        s.notna()
        & ~s.str.strip().str.lower().isin(_NULL_DOMAIN_STRINGS)
    ).to_numpy(dtype=bool)


def _safe_corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson correlation of two 1D arrays without warnings: NaN if x (target) is constant,
    0 if only y (prediction) is constant.
    """
    x, y = _finite_xy(x, y)
    if x.size < 2:
        return np.nan

    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)

    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))

    x_var = float(np.mean(np.square(xc), dtype=np.float64))
    y_var = float(np.mean(np.square(yc), dtype=np.float64))

    if x_var <= _VAR_EPS:
        return np.nan
    if y_var <= _VAR_EPS:
        return 0.0

    out = float(np.mean(xc * yc, dtype=np.float64) / np.sqrt(x_var * y_var))
    if not np.isfinite(out):
        return np.nan
    return float(np.clip(out, -1.0, 1.0))


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _finite_xy(x, y)
    if x.size < 2:
        return np.nan
    return _safe_corr_1d(
        rankdata(x, method="average"),
        rankdata(y, method="average"),
    )


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    return _safe_corr_1d(x, y)


def _corrcoef_columns_no_warning(X: np.ndarray) -> np.ndarray:
    """Column-wise correlation matrix of X (finite, non-constant columns only)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape={X.shape}.")
    if X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError(f"Need at least 2 rows and 2 columns, got shape={X.shape}.")

    Xc = X - np.mean(X, axis=0, keepdims=True)
    ss = np.sum(np.square(Xc), axis=0, dtype=np.float64)
    if np.any(ss <= _VAR_EPS):
        raise ValueError("Internal error: _corrcoef_columns_no_warning received a constant column.")

    corr = (Xc.T @ Xc) / np.sqrt(np.outer(ss, ss))
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _prepare_coexpression_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    min_rows: int,
    min_genes: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, int]]]:
    """Keep the finite gene columns with non-zero variance in both y_true and y_pred (coexpression preservation)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    if yt.shape != yp.shape:
        raise ValueError(f"y_true/y_pred shape mismatch: {yt.shape} vs {yp.shape}.")
    if yt.ndim != 2:
        raise ValueError(f"Expected 2D y_true/y_pred, got shape={yt.shape}.")

    info = {
        "n_coexpression_rows_initial": int(yt.shape[0]),
        "n_coexpression_genes_initial": int(yt.shape[1]),
    }

    finite_gene = np.all(np.isfinite(yt), axis=0) & np.all(np.isfinite(yp), axis=0)
    info["n_coexpression_genes_finite"] = int(np.sum(finite_gene))
    info["n_coexpression_genes_dropped_nonfinite"] = int(yt.shape[1] - np.sum(finite_gene))

    yt = yt[:, finite_gene]
    yp = yp[:, finite_gene]

    if yt.shape[0] < int(min_rows) or yt.shape[1] < int(min_genes):
        return None

    true_var = np.var(yt, axis=0, dtype=np.float64)
    pred_var = np.var(yp, axis=0, dtype=np.float64)
    variable_gene = (true_var > _VAR_EPS) & (pred_var > _VAR_EPS)

    info["n_coexpression_genes_dropped_low_variance_true"] = int(np.sum(true_var <= _VAR_EPS))
    info["n_coexpression_genes_dropped_low_variance_pred"] = int(np.sum(pred_var <= _VAR_EPS))
    info["n_coexpression_genes_used"] = int(np.sum(variable_gene))
    info["n_coexpression_rows_used"] = int(yt.shape[0])

    if info["n_coexpression_genes_used"] < int(min_genes):
        return None

    return yt[:, variable_gene], yp[:, variable_gene], info


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _finite_xy(x, y)
    if x.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(x - y), dtype=np.float64)))


def _mae(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _finite_xy(x, y)
    if x.size == 0:
        return np.nan
    return float(np.mean(np.abs(x - y), dtype=np.float64))


def _huber(x: np.ndarray, y: np.ndarray, delta: float) -> float:
    x, y = _finite_xy(x, y)
    if x.size == 0:
        return np.nan
    r = x - y
    a = np.abs(r)
    out = np.where(a <= delta, 0.5 * np.square(r), delta * (a - 0.5 * delta))
    return float(np.mean(out, dtype=np.float64))


def is_signature_basis(cfg: Mapping) -> bool:
    """True when the target columns are signatures (target_basis="signatures"), not genes."""
    return str(cfg["protocol"].get("target_basis", "genes")) == "signatures"


def _signature_target_cfg(cfg: Mapping, block: str) -> Mapping:
    return dict(cfg["protocol"].get("signature_target", {}).get(block, {}) or {})


def _valid_gene_pair_mask(y_true: np.ndarray, detected: np.ndarray, cfg: Mapping) -> np.ndarray:
    if is_signature_basis(cfg):
        scfg = _signature_target_cfg(cfg, "correlation_validity")
        min_n = int(scfg.get("min_samples_per_signature_slide", 30))
        min_unique = int(scfg.get("min_unique_target_values", 3))
        min_pos = 0  # no on/off state on an aggregate
    else:
        mcfg = cfg["protocol"]["metrics"]["correlation_validity"]
        min_n = int(mcfg["min_samples_per_gene_slide"])
        min_pos = int(mcfg["min_nonzero_samples_per_gene_slide"])
        min_unique = int(mcfg["min_unique_target_values"])
    G = y_true.shape[1]
    keep = np.zeros((G,), dtype=bool)
    for g in range(G):
        t = y_true[:, g]
        valid = np.isfinite(t)
        if int(np.sum(valid)) < min_n:
            continue
        # "nonzero" = the raw count was detected; the z-scored target can be negative.
        if min_pos > 0 and int(np.sum(detected[:, g][valid])) < min_pos:
            continue
        t = t[valid]
        if int(np.unique(t).size) < min_unique:
            continue
        if np.var(t) <= _VAR_EPS:
            continue
        keep[g] = True
    return keep


def _valid_magnitude_mask(y_true: np.ndarray, cfg: Mapping) -> np.ndarray:
    min_n = int(cfg["protocol"]["metrics"]["magnitude_validity"]["min_samples_per_gene_slide"])
    G = y_true.shape[1]
    keep = np.zeros((G,), dtype=bool)
    for g in range(G):
        t = y_true[:, g]
        valid = np.isfinite(t)
        keep[g] = int(np.sum(valid)) >= min_n
    return keep


def _location_profile_floors(cfg: Mapping, *, stratified: bool = False) -> Tuple[int, int]:
    """(min columns, min positive columns) for a per-location profile Spearman (stratified: on a subset of the columns)."""
    if is_signature_basis(cfg):
        scfg = _signature_target_cfg(cfg, "location_profile_validity")
        return (
            int(scfg.get("min_signatures_per_location", 5)),
            int(scfg.get("min_positive_signatures_per_location", 0)),
        )
    vcfg = cfg["protocol"]["metrics"]["location_profile_validity"]
    min_pos = int(vcfg["min_positive_genes_per_location"])
    if stratified:
        return int(vcfg.get("min_genes_per_location_stratified", 5)), min_pos
    return int(vcfg["min_genes_per_location"]), min_pos


def _location_profile_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    detected: np.ndarray,
    cfg: Mapping,
    *,
    stratified: bool = False,
) -> np.ndarray:
    min_g, min_pos = _location_profile_floors(cfg, stratified=stratified)
    out = np.full((y_true.shape[0],), np.nan, dtype=np.float32)
    for i in range(y_true.shape[0]):
        t = y_true[i]
        p = y_pred[i]
        valid = np.isfinite(t) & np.isfinite(p)
        if int(np.sum(valid)) < min_g:
            continue
        if min_pos > 0 and int(np.sum(detected[i][valid])) < min_pos:
            continue
        t = t[valid]
        p = p[valid]
        if np.var(t) <= _VAR_EPS:
            continue
        if np.var(p) <= _VAR_EPS:
            out[i] = 0.0
            continue
        out[i] = _safe_spearman(t, p)
    return out


def _macro_mean(vals: Sequence[float]) -> float:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else np.nan


def _finite_mean_with_counts(vals: Sequence[float]) -> Tuple[float, int, int]:
    arr = np.asarray(vals, dtype=np.float64).reshape(-1)
    valid = np.isfinite(arr)
    n_valid = int(np.sum(valid))
    n_total = int(arr.size)
    if n_valid == 0:
        return np.nan, 0, n_total
    return float(np.mean(arr[valid])), n_valid, n_total


def _empty_df(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns})


def _macro_eligibility_cfg(cfg: Mapping) -> Tuple[int, int]:
    """
    (min valid targets per slide, min valid slides per target) for the macro averages.
    Signature floors on the signature basis; 1 slide per target for STHELAR_TEST_ADAPTED (one slide per fold).
    """
    if is_signature_basis(cfg):
        scfg = _signature_target_cfg(cfg, "macro_aggregation_validity")
        min_valid_targets_per_slide = int(
            scfg.get("min_valid_signatures_per_slide_for_slide_macro", 1)
        )
        min_valid_slides_per_target = int(
            scfg.get("min_valid_slides_per_signature_for_signature_macro", 1)
        )
    else:
        mcfg = cfg["protocol"]["metrics"].get("macro_aggregation_validity", {})
        min_valid_targets_per_slide = int(
            mcfg.get("min_valid_genes_per_slide_for_slide_macro", 1)
        )
        min_valid_slides_per_target = int(
            mcfg.get("min_valid_slides_per_gene_for_gene_macro", 1)
        )
    if str(cfg["protocol"].get("sub_eval_configuration", "")) == "STHELAR_TEST_ADAPTED":
        min_valid_slides_per_target = min(min_valid_slides_per_target, 1)
    return min_valid_targets_per_slide, min_valid_slides_per_target


def _ceiling_normalize(metric: pd.Series, r: pd.Series) -> pd.Series:
    """Noise-ceiling-normalized score metric / sqrt(clip(r, _CEILING_EPS, 1)), NaN where r or metric is not finite."""
    m = pd.to_numeric(metric, errors="coerce").to_numpy(dtype=np.float64)
    rr = pd.to_numeric(r, errors="coerce").to_numpy(dtype=np.float64)
    denom = np.sqrt(np.clip(rr, _CEILING_EPS, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = m / denom
    out[~np.isfinite(rr) | ~np.isfinite(m)] = np.nan
    return pd.Series(out, index=metric.index)


def _attach_reliability(
    slide_gene_df: pd.DataFrame,
    loc_df: pd.DataFrame,
    reliability_df: Optional[pd.DataFrame],
    reliability_profile_df: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Join the count-thinning reliabilities to the per-(slide, gene) and per-location tables and add the
    *_ceiling_norm columns (Spearman / Pearson normalized by their own reliability; NaN without reliability).
    """
    sg = slide_gene_df.copy()
    if reliability_df is not None and not reliability_df.empty and not sg.empty:
        rel = reliability_df[["slide_id", "gene", "r_pearson", "r_spearman"]].copy()
        rel["slide_id"] = rel["slide_id"].astype(str)
        rel["gene"] = rel["gene"].astype(str)
        sg["slide_id"] = sg["slide_id"].astype(str)
        sg["gene"] = sg["gene"].astype(str)
        sg = sg.merge(rel, on=["slide_id", "gene"], how="left")
        sg["spearman_ceiling_norm"] = _ceiling_normalize(sg["spearman"], sg["r_spearman"])
        sg["pearson_ceiling_norm"] = _ceiling_normalize(sg["pearson"], sg["r_pearson"])

    lc = loc_df.copy()
    if reliability_profile_df is not None and not reliability_profile_df.empty and not lc.empty:
        relp = reliability_profile_df[["slide_id", "row_key", "r_profile_spearman"]].copy()
        relp["slide_id"] = relp["slide_id"].astype(str)
        relp["row_key"] = relp["row_key"].astype(str)
        lc["slide_id"] = lc["slide_id"].astype(str)
        lc["row_key"] = lc["row_key"].astype(str)
        lc = lc.merge(relp, on=["slide_id", "row_key"], how="left")
        lc["profile_spearman_ceiling_norm"] = _ceiling_normalize(
            lc["profile_spearman"], lc["r_profile_spearman"]
        )
    return sg, lc


def _compute_summary_from_tables(
    slide_gene_df: pd.DataFrame,
    loc_df: pd.DataFrame,
    auroc_df: pd.DataFrame,
    cfg: Mapping,
) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    min_valid_genes_per_slide, min_valid_slides_per_gene = _macro_eligibility_cfg(cfg)
    slide_group_cols = ["slide_id", "source", "subset", "tissue"]

    if slide_gene_df is None or slide_gene_df.empty:
        corr_df = pd.DataFrame(columns=slide_group_cols + ["gene", "spearman", "pearson", "corr_valid"])
        mag_df = pd.DataFrame(columns=slide_group_cols + ["gene", "mae", "rmse", "huber", "mag_valid"])
    else:
        corr_df = slide_gene_df.loc[slide_gene_df["corr_valid"]].copy()
        mag_df = slide_gene_df.loc[slide_gene_df["mag_valid"]].copy()

    loc_valid_df = (
        loc_df.loc[np.isfinite(loc_df["profile_spearman"])].copy()
        if (loc_df is not None and not loc_df.empty)
        else pd.DataFrame(columns=slide_group_cols + ["row_key", "profile_spearman"])
    )

    # correlations
    if not corr_df.empty:
        slide_cov = corr_df.groupby(slide_group_cols, dropna=False)["gene"].nunique().reset_index(name="n_valid_corr_genes")
        slide_cov["eligible_for_slide_macro_corr"] = slide_cov["n_valid_corr_genes"] >= min_valid_genes_per_slide
        gene_cov = corr_df.groupby("gene", dropna=False)["slide_id"].nunique().reset_index(name="n_valid_corr_slides")
        gene_cov["eligible_for_gene_macro_corr"] = gene_cov["n_valid_corr_slides"] >= min_valid_slides_per_gene

        slide_mean = corr_df.groupby(slide_group_cols, dropna=False)[["spearman", "pearson"]].mean().reset_index()
        slide_mean = slide_mean.merge(slide_cov, on=slide_group_cols, how="left")
        slide_mean = slide_mean.rename(columns={"spearman": "slide_macro_spearman", "pearson": "slide_macro_pearson"})

        gene_mean = corr_df.groupby("gene", dropna=False)[["spearman", "pearson"]].mean().reset_index()
        gene_mean = gene_mean.merge(gene_cov, on="gene", how="left")
        gene_mean = gene_mean.rename(columns={"spearman": "gene_macro_spearman", "pearson": "gene_macro_pearson"})

        slide_macro_spearman = float(slide_mean.loc[slide_mean["eligible_for_slide_macro_corr"], "slide_macro_spearman"].mean())
        slide_macro_pearson = float(slide_mean.loc[slide_mean["eligible_for_slide_macro_corr"], "slide_macro_pearson"].mean())
        gene_macro_spearman = float(gene_mean.loc[gene_mean["eligible_for_gene_macro_corr"], "gene_macro_spearman"].mean())
        gene_macro_pearson = float(gene_mean.loc[gene_mean["eligible_for_gene_macro_corr"], "gene_macro_pearson"].mean())
    else:
        slide_cov = pd.DataFrame(columns=slide_group_cols + ["n_valid_corr_genes", "eligible_for_slide_macro_corr"])
        gene_cov = pd.DataFrame(columns=["gene", "n_valid_corr_slides", "eligible_for_gene_macro_corr"])
        slide_mean = pd.DataFrame(columns=slide_group_cols + ["slide_macro_spearman", "slide_macro_pearson", "n_valid_corr_genes", "eligible_for_slide_macro_corr"])
        gene_mean = pd.DataFrame(columns=["gene", "gene_macro_spearman", "gene_macro_pearson", "n_valid_corr_slides", "eligible_for_gene_macro_corr"])
        slide_macro_spearman = np.nan
        slide_macro_pearson = np.nan
        gene_macro_spearman = np.nan
        gene_macro_pearson = np.nan

    # magnitude
    if not mag_df.empty:
        mag_slide_cov = mag_df.groupby(slide_group_cols, dropna=False)["gene"].nunique().reset_index(name="n_valid_mag_genes")
        mag_slide_cov["eligible_for_slide_macro_magnitude"] = mag_slide_cov["n_valid_mag_genes"] >= min_valid_genes_per_slide
        mag_gene_cov = mag_df.groupby("gene", dropna=False)["slide_id"].nunique().reset_index(name="n_valid_mag_slides")
        mag_gene_cov["eligible_for_gene_macro_magnitude"] = mag_gene_cov["n_valid_mag_slides"] >= min_valid_slides_per_gene

        mag_slide_mean = mag_df.groupby(slide_group_cols, dropna=False)[["mae", "rmse", "huber"]].mean().reset_index()
        mag_slide_mean = mag_slide_mean.merge(mag_slide_cov, on=slide_group_cols, how="left")
        mag_slide_mean = mag_slide_mean.rename(
            columns={"mae": "slide_macro_mae", "rmse": "slide_macro_rmse", "huber": "slide_macro_huber"}
        )
        mag_gene_mean = mag_df.groupby("gene", dropna=False)[["mae", "rmse", "huber"]].mean().reset_index()
        mag_gene_mean = mag_gene_mean.merge(mag_gene_cov, on="gene", how="left")
        mag_gene_mean = mag_gene_mean.rename(
            columns={"mae": "gene_macro_mae", "rmse": "gene_macro_rmse", "huber": "gene_macro_huber"}
        )

        slide_macro_mae = float(mag_slide_mean.loc[mag_slide_mean["eligible_for_slide_macro_magnitude"], "slide_macro_mae"].mean())
        slide_macro_rmse = float(mag_slide_mean.loc[mag_slide_mean["eligible_for_slide_macro_magnitude"], "slide_macro_rmse"].mean())
        slide_macro_huber = float(mag_slide_mean.loc[mag_slide_mean["eligible_for_slide_macro_magnitude"], "slide_macro_huber"].mean())
    else:
        mag_slide_cov = pd.DataFrame(columns=slide_group_cols + ["n_valid_mag_genes", "eligible_for_slide_macro_magnitude"])
        mag_gene_cov = pd.DataFrame(columns=["gene", "n_valid_mag_slides", "eligible_for_gene_macro_magnitude"])
        mag_slide_mean = pd.DataFrame(columns=slide_group_cols + ["slide_macro_mae", "slide_macro_rmse", "slide_macro_huber", "n_valid_mag_genes", "eligible_for_slide_macro_magnitude"])
        mag_gene_mean = pd.DataFrame(columns=["gene", "gene_macro_mae", "gene_macro_rmse", "gene_macro_huber", "n_valid_mag_slides", "eligible_for_gene_macro_magnitude"])
        slide_macro_mae = np.nan
        slide_macro_rmse = np.nan
        slide_macro_huber = np.nan

    # location profiles
    if not loc_valid_df.empty:
        loc_cov = loc_valid_df.groupby(slide_group_cols, dropna=False)["row_key"].size().reset_index(name="n_valid_locations")
        loc_slide_mean = loc_valid_df.groupby(slide_group_cols, dropna=False)["profile_spearman"].mean().reset_index()
        loc_slide_mean = loc_slide_mean.merge(loc_cov, on=slide_group_cols, how="left")
        loc_slide_mean = loc_slide_mean.rename(columns={"profile_spearman": "location_macro_profile_spearman"})
        location_macro_profile_spearman = float(loc_slide_mean["location_macro_profile_spearman"].mean())
    else:
        loc_cov = pd.DataFrame(columns=slide_group_cols + ["n_valid_locations"])
        loc_slide_mean = pd.DataFrame(columns=slide_group_cols + ["location_macro_profile_spearman", "n_valid_locations"])
        location_macro_profile_spearman = np.nan

    n_slides_used_for_slide_macro_corr = (
        int(slide_mean.loc[slide_mean["eligible_for_slide_macro_corr"], "slide_id"].nunique())
        if not slide_mean.empty else 0
    )
    n_genes_used_for_gene_macro_corr = (
        int(gene_mean.loc[gene_mean["eligible_for_gene_macro_corr"], "gene"].nunique())
        if not gene_mean.empty else 0
    )
    n_slides_used_for_slide_macro_magnitude = (
        int(mag_slide_mean.loc[mag_slide_mean["eligible_for_slide_macro_magnitude"], "slide_id"].nunique())
        if not mag_slide_mean.empty else 0
    )
    n_slides_used_for_location_macro = (
        int(loc_slide_mean["slide_id"].nunique())
        if not loc_slide_mean.empty else 0
    )

    summary = {
        "slide_macro_spearman": slide_macro_spearman,
        "gene_macro_spearman": gene_macro_spearman,
        "slide_macro_pearson": slide_macro_pearson,
        "gene_macro_pearson": gene_macro_pearson,
        "slide_macro_mae": slide_macro_mae,
        "slide_macro_rmse": slide_macro_rmse,
        "slide_macro_huber": slide_macro_huber,
        "location_macro_profile_spearman": location_macro_profile_spearman,
        "n_valid_slide_gene_pairs": int(corr_df.shape[0]),
        "n_valid_magnitude_pairs": int(mag_df.shape[0]),
        "n_valid_location_profiles": int(loc_valid_df.shape[0]),
        "n_valid_onoff_pairs": int(auroc_df.shape[0]),
        "n_slides_used_for_slide_macro_corr": n_slides_used_for_slide_macro_corr,
        "n_genes_used_for_gene_macro_corr": n_genes_used_for_gene_macro_corr,
        "n_slides_used_for_slide_macro_magnitude": n_slides_used_for_slide_macro_magnitude,
        "n_slides_used_for_location_macro": n_slides_used_for_location_macro,
    }

    # Noise-ceiling-normalized correlation macros (NaN without reliability)
    for base in ("spearman", "pearson"):
        cn_col = f"{base}_ceiling_norm"
        if (not corr_df.empty) and (cn_col in corr_df.columns):
            # Eligibility of the ceiling-norm macros: number of cells with a ceiling-normalized value
            cn_counts_slide = (
                corr_df.assign(_ok=corr_df[cn_col].notna())
                .groupby(slide_group_cols, dropna=False)["_ok"].sum().reset_index()
                .rename(columns={"_ok": f"n_valid_ceiling_genes_{base}"})
            )
            cn_counts_gene = (
                corr_df.assign(_ok=corr_df[cn_col].notna())
                .groupby("gene", dropna=False)["_ok"].sum().reset_index()
                .rename(columns={"_ok": f"n_valid_ceiling_slides_{base}"})
            )
            cn_slide = (
                corr_df.groupby(slide_group_cols, dropna=False)[cn_col].mean().reset_index()
                .merge(slide_cov, on=slide_group_cols, how="left")
                .merge(cn_counts_slide, on=slide_group_cols, how="left")
            )
            cn_gene = (
                corr_df.groupby("gene", dropna=False)[cn_col].mean().reset_index()
                .merge(gene_cov, on="gene", how="left")
                .merge(cn_counts_gene, on="gene", how="left")
            )
            cn_slide[f"eligible_for_slide_macro_ceiling_{base}"] = (
                cn_slide[f"n_valid_ceiling_genes_{base}"].fillna(0) >= min_valid_genes_per_slide
            )
            cn_gene[f"eligible_for_gene_macro_ceiling_{base}"] = (
                cn_gene[f"n_valid_ceiling_slides_{base}"].fillna(0) >= min_valid_slides_per_gene
            )
            summary[f"slide_macro_{base}_ceiling_norm"] = (
                float(cn_slide.loc[
                    cn_slide[f"eligible_for_slide_macro_ceiling_{base}"], cn_col
                ].mean())
                if not cn_slide.empty else np.nan
            )
            summary[f"gene_macro_{base}_ceiling_norm"] = (
                float(cn_gene.loc[
                    cn_gene[f"eligible_for_gene_macro_ceiling_{base}"], cn_col
                ].mean())
                if not cn_gene.empty else np.nan
            )
            summary[f"n_slides_used_for_slide_macro_ceiling_{base}"] = int(
                cn_slide[f"eligible_for_slide_macro_ceiling_{base}"].sum()
            )
            summary[f"n_genes_used_for_gene_macro_ceiling_{base}"] = int(
                cn_gene[f"eligible_for_gene_macro_ceiling_{base}"].sum()
            )
            # Cells at the ceiling floor and cells with a ceiling-normalized value above 1
            r_col = f"r_{base}"
            if r_col in corr_df.columns:
                r_vals = pd.to_numeric(corr_df[r_col], errors="coerce")
                summary[f"n_cells_at_ceiling_floor_{base}"] = int(
                    ((r_vals <= _CEILING_EPS) & r_vals.notna()).sum()
                )
                summary[f"frac_cells_at_ceiling_floor_{base}"] = (
                    float(((r_vals <= _CEILING_EPS) & r_vals.notna()).mean())
                    if len(r_vals) else np.nan
                )
            summary[f"n_cells_ceiling_norm_gt_1_{base}"] = int(
                (pd.to_numeric(corr_df[cn_col], errors="coerce") > 1.0).sum()
            )
            # Per-slide / per-gene ceiling-normalized values in the macro tables
            if not cn_slide.empty:
                slide_mean = slide_mean.merge(
                    cn_slide[slide_group_cols + [cn_col]].rename(
                        columns={cn_col: f"slide_macro_{base}_ceiling_norm"}),
                    on=slide_group_cols, how="left",
                )
            if not cn_gene.empty:
                gene_mean = gene_mean.merge(
                    cn_gene[["gene", cn_col]].rename(
                        columns={cn_col: f"gene_macro_{base}_ceiling_norm"}),
                    on="gene", how="left",
                )
        else:
            summary[f"slide_macro_{base}_ceiling_norm"] = np.nan
            summary[f"gene_macro_{base}_ceiling_norm"] = np.nan
            # Same keys as above
            summary[f"n_slides_used_for_slide_macro_ceiling_{base}"] = 0
            summary[f"n_genes_used_for_gene_macro_ceiling_{base}"] = 0
            summary[f"n_cells_at_ceiling_floor_{base}"] = 0
            summary[f"frac_cells_at_ceiling_floor_{base}"] = np.nan
            summary[f"n_cells_ceiling_norm_gt_1_{base}"] = 0

    loc_cn_col = "profile_spearman_ceiling_norm"
    if (not loc_valid_df.empty) and (loc_cn_col in loc_valid_df.columns):
        # Per slide: mean ceiling-normalized profile Spearman and number of locations with a value (eligibility)
        loc_cn = (
            loc_valid_df.assign(_ok=loc_valid_df[loc_cn_col].notna())
            .groupby(slide_group_cols, dropna=False)
            .agg(**{
                loc_cn_col: (loc_cn_col, "mean"),
                "n_valid_ceiling_locations": ("_ok", "sum"),
            })
            .reset_index()
        )
        loc_cn["eligible_for_location_macro_ceiling"] = (
            loc_cn["n_valid_ceiling_locations"].fillna(0) >= min_valid_genes_per_slide
        )
        summary["location_macro_profile_spearman_ceiling_norm"] = (
            float(loc_cn.loc[loc_cn["eligible_for_location_macro_ceiling"], loc_cn_col].mean())
            if not loc_cn.empty else np.nan
        )
        summary["n_slides_used_for_location_macro_ceiling"] = int(
            loc_cn["eligible_for_location_macro_ceiling"].sum()
        )
        # Per-slide values in the macro table (not gated)
        if not loc_cn.empty:
            loc_slide_mean = loc_slide_mean.merge(
                loc_cn[slide_group_cols + [loc_cn_col]].rename(
                    columns={loc_cn_col: "location_macro_profile_spearman_ceiling_norm"}),
                on=slide_group_cols, how="left",
            )
    else:
        summary["location_macro_profile_spearman_ceiling_norm"] = np.nan
        # Same key as above
        summary["n_slides_used_for_location_macro_ceiling"] = 0

    coverage = {
        "valid_genes_by_slide": slide_cov,
        "valid_slides_by_gene": gene_cov,
        "valid_mag_genes_by_slide": mag_slide_cov,
        "valid_mag_slides_by_gene": mag_gene_cov,
        "valid_locations_by_slide": loc_cov,
    }
    macro_tables = {
        "slide_macro_corr": slide_mean,
        "gene_macro_corr": gene_mean,
        "slide_macro_magnitude": mag_slide_mean,
        "gene_macro_magnitude": mag_gene_mean,
        "slide_macro_location_profile": loc_slide_mean,
    }
    return summary, coverage, macro_tables


def _subset_summary(
    y_true: np.ndarray, y_pred: np.ndarray, detected: np.ndarray, cfg: Mapping
) -> Dict[str, float]:
    corr_mask = _valid_gene_pair_mask(y_true, detected, cfg)
    mag_mask = _valid_magnitude_mask(y_true, cfg)
    spears, pears, maes, rmses = [], [], [], []
    for g in range(y_true.shape[1]):
        t = y_true[:, g]
        p = y_pred[:, g]
        valid = np.isfinite(t) & np.isfinite(p)
        tv = t[valid]
        pv = p[valid]
        if corr_mask[g]:
            spears.append(_safe_spearman(tv, pv))
            pears.append(_safe_pearson(tv, pv))
        if mag_mask[g]:
            maes.append(_mae(tv, pv))
            rmses.append(_rmse(tv, pv))
    return {
        "slide_macro_spearman": _macro_mean(spears),
        "slide_macro_pearson": _macro_mean(pears),
        "slide_macro_mae": _macro_mean(maes),
        "slide_macro_rmse": _macro_mean(rmses),
    }


# -----------------------------------------------------------------------------
# Level 1 benchmark metrics
# -----------------------------------------------------------------------------


def compute_benchmark_metrics(
    preds: Mapping[str, Mapping],
    cfg: Mapping,
    *,
    reliability_df: Optional[pd.DataFrame] = None,
    reliability_profile_df: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    per_slide_gene_rows: List[Dict] = []
    per_location_rows: List[Dict] = []
    auroc_rows: List[Dict] = []
    delta = float(cfg["protocol"]["metrics"]["huber_delta"])
    min_valid_genes_per_slide, min_valid_slides_per_gene = _macro_eligibility_cfg(cfg)
    auroc_min_pos = int(cfg["protocol"]["metrics"]["structure_validity"]["auroc_min_pos"])
    auroc_min_neg = int(cfg["protocol"]["metrics"]["structure_validity"]["auroc_min_neg"])
    onoff_applicable = not is_signature_basis(cfg)

    for slide_id, bundle in preds.items():
        y_true = np.asarray(bundle["y_true"], dtype=np.float32)
        y_pred = np.asarray(bundle["y_pred"], dtype=np.float32)
        _assert_no_nonfinite_predictions(slide_id, y_true, y_pred)
        gene_names = np.asarray(bundle["gene_names"], dtype=object)
        row_keys = np.asarray(bundle["row_keys"], dtype=object)
        detected = np.asarray(bundle["detected"], dtype=bool)


        corr_mask = _valid_gene_pair_mask(y_true, detected, cfg)
        mag_mask = _valid_magnitude_mask(y_true, cfg)

        for g, gene in enumerate(gene_names.tolist()):
            t = y_true[:, g]
            p = y_pred[:, g]
            valid = np.isfinite(t) & np.isfinite(p)
            corr_valid = bool(corr_mask[g])
            mag_valid = bool(mag_mask[g])
            if not (corr_valid or mag_valid):
                continue

            tv = t[valid]
            pv = p[valid]
            dv = detected[:, g][valid]
            n_pos = int(np.sum(dv))
            n_neg = int(tv.shape[0]) - n_pos
            per_slide_gene_rows.append(
                {
                    "slide_id": slide_id,
                    "gene": str(gene),
                    "source": bundle["source"],
                    "subset": bundle["subset"],
                    "tissue": bundle["tissue"],
                    "corr_valid": corr_valid,
                    "mag_valid": mag_valid,
                    "spearman": _safe_spearman(tv, pv) if corr_valid else np.nan,
                    "pearson": _safe_pearson(tv, pv) if corr_valid else np.nan,
                    "mae": _mae(tv, pv) if mag_valid else np.nan,
                    "rmse": _rmse(tv, pv) if mag_valid else np.nan,
                    "huber": _huber(tv, pv, delta=delta) if mag_valid else np.nan,
                    "n_samples": int(tv.shape[0]),
                    "n_positive": n_pos,
                    "n_negative": n_neg,
                }
            )

            # on/off metrics (gene basis only: a signature score has no on/off state)
            if onoff_applicable and n_pos >= auroc_min_pos and n_neg >= auroc_min_neg:
                y_bin = dv.astype(np.int32)
                try:
                    auroc = float(roc_auc_score(y_bin, pv))
                except Exception:
                    auroc = np.nan
                try:
                    auprc = float(average_precision_score(y_bin, pv))
                except Exception:
                    auprc = np.nan
                auroc_rows.append(
                    {
                        "slide_id": slide_id,
                        "gene": str(gene),
                        "source": bundle["source"],
                        "subset": bundle["subset"],
                        "tissue": bundle["tissue"],
                        "auroc": auroc,
                        "auprc": auprc,
                        "n_samples": int(tv.shape[0]),
                        "n_positive": n_pos,
                        "n_negative": n_neg,
                    }
                )

        loc_scores = _location_profile_scores(y_true, y_pred, detected, cfg)
        finite_true = np.isfinite(y_true)
        n_valid_genes_per_location = np.sum(finite_true, axis=1).astype(np.int32, copy=False)
        n_positive_genes_per_location = np.sum(finite_true & detected, axis=1).astype(np.int32, copy=False)
        for rk, score, nvg, npg in zip(row_keys.tolist(), loc_scores.tolist(), n_valid_genes_per_location.tolist(), n_positive_genes_per_location.tolist()):
            per_location_rows.append(
                {
                    "slide_id": slide_id,
                    "row_key": str(rk),
                    "profile_spearman": score,
                    "source": bundle["source"],
                    "subset": bundle["subset"],
                    "tissue": bundle["tissue"],
                    "n_valid_genes": int(nvg),
                    "n_positive_genes": int(npg),
                }
            )

    slide_gene_df = pd.DataFrame(per_slide_gene_rows)
    loc_df = pd.DataFrame(per_location_rows)
    auroc_df = pd.DataFrame(auroc_rows)

    # Join the count-thinning noise ceiling (inherited by the subset/tissue slices below)
    slide_gene_df, loc_df = _attach_reliability(
        slide_gene_df, loc_df, reliability_df, reliability_profile_df
    )

    benchmark_summary, coverage, macro_tables = _compute_summary_from_tables(
        slide_gene_df=slide_gene_df,
        loc_df=loc_df,
        auroc_df=auroc_df,
        cfg=cfg,
    )

    # Always save subset-level summaries when subsets exist
    per_subset_rows: List[Dict[str, object]] = []
    if not slide_gene_df.empty and "subset" in slide_gene_df.columns:
        subset_names = sorted(map(str, slide_gene_df["subset"].dropna().unique().tolist()))
        for subset in subset_names:
            subset_summary, _, _ = _compute_summary_from_tables(
                slide_gene_df=slide_gene_df.loc[slide_gene_df["subset"] == subset].copy() if not slide_gene_df.empty else pd.DataFrame(),
                loc_df=loc_df.loc[loc_df["subset"] == subset].copy() if not loc_df.empty else pd.DataFrame(),
                auroc_df=auroc_df.loc[auroc_df["subset"] == subset].copy() if not auroc_df.empty else pd.DataFrame(),
                cfg=cfg,
            )
            subset_summary = subset_summary[0] if isinstance(subset_summary, tuple) else subset_summary
            per_subset_rows.append({"subset": subset, **subset_summary})
    per_subset_benchmark = pd.DataFrame(per_subset_rows)

    # Tissue-level benchmark summaries.
    per_tissue_rows: List[Dict[str, object]] = []
    if not slide_gene_df.empty and "tissue" in slide_gene_df.columns:
        tissue_names = sorted(map(str, slide_gene_df["tissue"].dropna().unique().tolist()))
        for tissue in tissue_names:
            tissue_summary, _, _ = _compute_summary_from_tables(
                slide_gene_df=slide_gene_df.loc[slide_gene_df["tissue"].astype(str) == tissue].copy()
                if not slide_gene_df.empty else pd.DataFrame(),
                loc_df=loc_df.loc[loc_df["tissue"].astype(str) == tissue].copy()
                if not loc_df.empty else pd.DataFrame(),
                auroc_df=auroc_df.loc[auroc_df["tissue"].astype(str) == tissue].copy()
                if not auroc_df.empty else pd.DataFrame(),
                cfg=cfg,
            )
            per_tissue_rows.append({"tissue": tissue, **tissue_summary})
    per_tissue_benchmark = pd.DataFrame(per_tissue_rows)

    sub_cfg = str(cfg["protocol"].get("sub_eval_configuration", ""))
    if sub_cfg in HEST_CV_SUB_EVAL_CONFIGS:
        if per_tissue_benchmark.empty:
            for metric in CORE_BENCHMARK_METRICS:
                benchmark_summary[f"pooled_all_slides__{metric}"] = benchmark_summary.get(metric, np.nan)
                benchmark_summary[f"hest_tissue_macro__{metric}"] = np.nan
                benchmark_summary[f"hest_tissue_macro__n_valid_tissues__{metric}"] = 0
                benchmark_summary[f"hest_tissue_macro__n_total_tissues__{metric}"] = 0
                benchmark_summary[metric] = np.nan
        else:
            for metric in CORE_BENCHMARK_METRICS:
                if metric not in per_tissue_benchmark.columns:
                    continue
                tissue_macro_value, n_valid_tissue_scores, n_total_tissue_scores = _finite_mean_with_counts(
                    per_tissue_benchmark[metric].to_numpy(dtype=np.float64, copy=False)
                )
                benchmark_summary[f"pooled_all_slides__{metric}"] = benchmark_summary.get(metric, np.nan)
                benchmark_summary[f"hest_tissue_macro__{metric}"] = tissue_macro_value
                benchmark_summary[f"hest_tissue_macro__n_valid_tissues__{metric}"] = n_valid_tissue_scores
                benchmark_summary[f"hest_tissue_macro__n_total_tissues__{metric}"] = n_total_tissue_scores
                benchmark_summary[metric] = tissue_macro_value

    out = {
        "benchmark_summary": benchmark_summary,
        "per_slide_gene": slide_gene_df,
        "per_location": loc_df,
        "per_slide_onoff": auroc_df,
        "per_subset_benchmark": per_subset_benchmark,
        "per_tissue_benchmark": per_tissue_benchmark,
        "coverage": coverage,
        "macro_tables": macro_tables,
    }
    return out


def compute_slide_macro_spearman_score(preds: Mapping[str, Mapping], cfg: Mapping) -> Dict[str, float]:
    """Slide-macro Spearman only (same definition as compute_benchmark_metrics), used as the alpha-selection score."""
    min_valid_genes_per_slide, _ = _macro_eligibility_cfg(cfg)

    slide_spearman_values: List[float] = []
    n_valid_corr_pairs = 0

    for slide_id, bundle in preds.items():
        y_true = np.asarray(bundle["y_true"], dtype=np.float32)
        y_pred = np.asarray(bundle["y_pred"], dtype=np.float32)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Alpha-selection y_true/y_pred shape mismatch for slide '{slide_id}': "
                f"{y_true.shape} vs {y_pred.shape}."
            )

        _assert_no_nonfinite_predictions(str(slide_id), y_true, y_pred)

        detected = np.asarray(bundle["detected"], dtype=bool)
        corr_mask = _valid_gene_pair_mask(y_true, detected, cfg)

        spears: List[float] = []
        for g in np.flatnonzero(corr_mask).tolist():
            t = y_true[:, int(g)]
            p = y_pred[:, int(g)]
            valid = np.isfinite(t) & np.isfinite(p)
            rho = _safe_spearman(t[valid], p[valid])
            if np.isfinite(rho):
                spears.append(float(rho))

        n_corr = int(np.sum(corr_mask))
        n_valid_corr_pairs += n_corr

        if n_corr >= int(min_valid_genes_per_slide):
            slide_spearman_values.append(_macro_mean(spears))

    return {
        "loso_slide_macro_spearman": _macro_mean(slide_spearman_values),
        "loso_n_valid_corr_pairs": int(n_valid_corr_pairs),
        "loso_n_slides_used_for_slide_macro_corr": int(
            np.sum(np.isfinite(np.asarray(slide_spearman_values, dtype=np.float64)))
        ),
    }


# -----------------------------------------------------------------------------
# Level 2 diagnostics
# -----------------------------------------------------------------------------


def _profile_macro_for_gene_subset(
    preds: Mapping[str, Mapping], gene_subset: Sequence[str], cfg: Mapping
) -> Tuple[float, str, int]:
    """
    location_macro_profile_spearman restricted to a subset of genes (stratified profile floor).
    Returns (value, skip_reason, n_columns_used), skip_reason being "" when the value is finite.
    """
    min_cols, _ = _location_profile_floors(cfg, stratified=True)
    gset = {str(g) for g in gene_subset}
    n_subset = len(gset)
    if n_subset < max(2, min_cols):
        return (
            np.nan,
            f"stratum has {n_subset} target column(s); profile Spearman needs "
            f">= {max(2, min_cols)}",
            n_subset,
        )
    slide_means: List[float] = []
    n_cols_used = 0
    for _slide_id, bundle in preds.items():
        gene_names = np.asarray(bundle["gene_names"], dtype=object)
        cols = np.array([str(g) in gset for g in gene_names.tolist()], dtype=bool)
        if int(np.sum(cols)) < max(2, min_cols):
            continue
        n_cols_used = max(n_cols_used, int(np.sum(cols)))
        yt = np.asarray(bundle["y_true"], dtype=np.float32)[:, cols]
        yp = np.asarray(bundle["y_pred"], dtype=np.float32)[:, cols]
        det = np.asarray(bundle["detected"], dtype=bool)[:, cols]
        loc = _location_profile_scores(yt, yp, det, cfg, stratified=True)
        finite = loc[np.isfinite(loc)]
        if finite.size:
            slide_means.append(float(np.mean(finite)))
    value = _macro_mean(slide_means)
    if not np.isfinite(value):
        return (
            value,
            f"no slide yielded a finite profile Spearman on this stratum "
            f"(max {n_cols_used} column(s) present per slide, floor {max(2, min_cols)})",
            n_cols_used,
        )
    return value, "", n_cols_used


_TERCILE_LABELS = {-1: "no_score", 0: "low", 1: "mid", 2: "high"}


# Minimum number of genes for the localization metrics of a signature stratum
_MIN_SIGNATURE_GENES_LOCALIZATION = 1


def _min_genes_per_stratum(cfg: Mapping) -> int:
    """Minimum number of genes of a stratum with metrics (narrower strata are reported without metrics)."""
    return int(cfg["protocol"]["metrics"].get("min_genes_per_stratum", 6))


def _build_signature_strata_outputs(
    preds: Mapping[str, Mapping],
    gene_summary: pd.DataFrame,
    signature_strata: Optional[pd.DataFrame],
    cfg: Mapping,
) -> pd.DataFrame:
    """
    Per-signature localization and profile metrics, plus the any_signature (genes in at least one signature)
    and no_signature (genes in none) rows. Strata narrower than metrics.min_genes_per_stratum get no metrics.
    """
    cols = [
        "signature_id", "signature_category", "n_genes", "n_genes_in_teacher_vocab",
        "localization_spearman", "localization_spearman_ceiling_norm", "profile_spearman",
        "n_columns_used", "profile_skip_reason", "stratum_skip_reason",
    ]
    if (
        signature_strata is None
        or signature_strata.empty
        or gene_summary.empty
    ):
        return pd.DataFrame(columns=cols)

    ss = signature_strata.copy()
    ss["gene"] = ss["gene"].astype(str)
    ss["signature_id"] = ss["signature_id"].astype(str)
    if "signature_category" not in ss.columns:
        ss["signature_category"] = ""
    ss["signature_category"] = ss["signature_category"].astype(str)
    if "in_teacher_vocab" not in ss.columns:
        ss["in_teacher_vocab"] = True

    metric_cols = [
        c for c in (
            "gene", "gene_macro_spearman", "gene_macro_spearman_ceiling_norm",
        ) if c in gene_summary.columns
    ]
    gm = gene_summary[metric_cols].copy()
    gm["gene"] = gm["gene"].astype(str)
    merged = ss.merge(gm, on="gene", how="left")
    for col in ("gene_macro_spearman", "gene_macro_spearman_ceiling_norm"):
        if col not in merged.columns:
            merged[col] = np.nan

    def _row_for_subset(
        label: str, category: str, sub: pd.DataFrame,
    ) -> Dict[str, object]:
        n = int(sub.shape[0])
        n_in_vocab = (
            int(sub["in_teacher_vocab"].sum())
            if "in_teacher_vocab" in sub.columns
            else 0
        )
        min_stratum = _min_genes_per_stratum(cfg)
        if n < min_stratum:
            # Row kept without metrics
            return {
                "signature_id": label,
                "signature_category": category,
                "n_genes": n,
                "n_genes_in_teacher_vocab": n_in_vocab,
                "localization_spearman": np.nan,
                "localization_spearman_ceiling_norm": np.nan,
                "profile_spearman": np.nan,
                "n_columns_used": n,
                "profile_skip_reason": "",
                "stratum_skip_reason": (
                    f"n_genes={n} < min_genes_per_stratum={min_stratum} (untestable: "
                    "below the paired-test minimum)"
                ),
            }
        if n >= _MIN_SIGNATURE_GENES_LOCALIZATION:
            loc = _macro_mean(sub["gene_macro_spearman"].tolist())
            loc_norm = _macro_mean(sub["gene_macro_spearman_ceiling_norm"].tolist())
        else:
            loc = np.nan
            loc_norm = np.nan
        prof, prof_reason, n_cols = _profile_macro_for_gene_subset(
            preds, sub["gene"].tolist(), cfg
        )
        return {
            "signature_id": label,
            "signature_category": category,
            "n_genes": n,
            "n_genes_in_teacher_vocab": n_in_vocab,
            "localization_spearman": loc,
            "localization_spearman_ceiling_norm": loc_norm,
            "profile_spearman": prof,
            "n_columns_used": int(n_cols),
            "profile_skip_reason": prof_reason,
            "stratum_skip_reason": "",
        }

    rows: List[Dict[str, object]] = []
    for sig_id in sorted(merged["signature_id"].unique().tolist()):
        sub = merged.loc[merged["signature_id"] == sig_id]
        category = str(sub["signature_category"].iloc[0]) if not sub.empty else ""
        rows.append(_row_for_subset(sig_id, category, sub))

    # any_signature: deduplicate by gene so a multi-signature gene counts once.
    any_sub = merged.drop_duplicates(subset=["gene"]).copy()
    rows.append(_row_for_subset("any_signature", "aggregate", any_sub))

    # no_signature: eval genes not in any signature row.
    in_any = set(merged["gene"].astype(str).tolist())
    eval_genes = gene_summary["gene"].astype(str).tolist() if "gene" in gene_summary.columns else []
    no_sig_genes = [g for g in eval_genes if g not in in_any]
    if no_sig_genes:
        nb = gene_summary[gene_summary["gene"].astype(str).isin(no_sig_genes)].copy()
        nb["gene"] = nb["gene"].astype(str)
        nb["in_teacher_vocab"] = False
        for col in ("gene_macro_spearman", "gene_macro_spearman_ceiling_norm"):
            if col not in nb.columns:
                nb[col] = np.nan
        rows.append(_row_for_subset("no_signature", "background", nb))

    return pd.DataFrame(rows, columns=cols)


def _build_gene_strata_outputs(
    preds: Mapping[str, Mapping],
    gene_summary: pd.DataFrame,
    gene_strata: Optional[pd.DataFrame],
    cfg: Mapping,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Gene-property strata: per-axis tables of metrics per tercile (gene_property_strata) and
    per-gene metrics joined to the continuous scores (gene_strata_continuous).
    """
    cont_cols = [
        "gene", "gene_macro_spearman", "gene_macro_spearman_ceiling_norm", "gene_macro_pearson",
        "ct_specificity_score", "spatial_structure_score",
        "ct_specificity_tercile", "spatial_structure_tercile",
        "in_teacher_vocab", "in_scgpt_vocab",
    ]
    if gene_strata is None or gene_strata.empty or gene_summary.empty:
        return {}, pd.DataFrame(columns=cont_cols)

    gs = gene_strata.copy()
    gs["gene"] = gs["gene"].astype(str)
    metric_cols = [c for c in ["gene", "gene_macro_spearman", "gene_macro_pearson",
                               "gene_macro_spearman_ceiling_norm"] if c in gene_summary.columns]
    gm = gene_summary[metric_cols].copy()
    gm["gene"] = gm["gene"].astype(str)
    cont = gs.merge(gm, on="gene", how="left")
    for col in ("gene_macro_spearman", "gene_macro_pearson", "gene_macro_spearman_ceiling_norm"):
        if col not in cont.columns:
            cont[col] = np.nan

    strata: Dict[str, pd.DataFrame] = {}
    for axis_name, tcol, scol in [
        ("ct_specificity", "ct_specificity_tercile", "ct_specificity_score"),
        ("spatial_structure", "spatial_structure_tercile", "spatial_structure_score"),
    ]:
        if tcol not in cont.columns:
            continue
        rows: List[Dict[str, object]] = []
        min_stratum = _min_genes_per_stratum(cfg)
        for terc in sorted(pd.to_numeric(cont[tcol], errors="coerce").dropna().unique().tolist()):
            sub = cont.loc[cont[tcol] == terc]
            n_genes = int(sub.shape[0])
            score_finite = pd.to_numeric(sub[scol], errors="coerce").to_numpy(dtype=np.float64)
            score_finite = score_finite[np.isfinite(score_finite)]
            base = {
                "tercile": int(terc),
                "tercile_label": _TERCILE_LABELS.get(int(terc), str(int(terc))),
                "n_genes": n_genes,
                "n_genes_in_teacher_vocab": int(sub["in_teacher_vocab"].sum()) if "in_teacher_vocab" in sub.columns else 0,
                "score_min": float(np.min(score_finite)) if score_finite.size else np.nan,
                "score_max": float(np.max(score_finite)) if score_finite.size else np.nan,
            }
            if n_genes < min_stratum:
                # Row kept without metrics
                rows.append({
                    **base,
                    "localization_spearman": np.nan,
                    "localization_spearman_ceiling_norm": np.nan,
                    "profile_spearman": np.nan,
                    "n_columns_used": n_genes,
                    "profile_skip_reason": "",
                    "stratum_skip_reason": (
                        f"n_genes={n_genes} < min_genes_per_stratum={min_stratum} "
                        "(untestable: below the paired-test minimum)"
                    ),
                })
                continue
            prof, prof_reason, n_cols = _profile_macro_for_gene_subset(
                preds, sub["gene"].tolist(), cfg
            )
            rows.append({
                **base,
                "localization_spearman": _macro_mean(sub["gene_macro_spearman"].tolist()),
                "localization_spearman_ceiling_norm": _macro_mean(sub["gene_macro_spearman_ceiling_norm"].tolist()),
                "profile_spearman": prof,
                "n_columns_used": int(n_cols),
                "profile_skip_reason": prof_reason,
                "stratum_skip_reason": "",
            })
        strata[axis_name] = pd.DataFrame(rows)

    return strata, cont[cont_cols].copy()


def compute_diagnostics(
    preds: Mapping[str, Mapping],
    benchmark: Mapping[str, object],
    cfg: Mapping,
    gene_strata: Optional[pd.DataFrame] = None,
    signature_strata: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    slide_gene_df = benchmark["per_slide_gene"].copy()
    if slide_gene_df.empty and len(slide_gene_df.columns) == 0:
        slide_gene_df = pd.DataFrame(
            columns=[
                "slide_id", "gene", "source", "subset", "tissue",
                "corr_valid", "mag_valid",
                "spearman", "pearson", "mae", "rmse", "huber",
                "n_samples", "n_positive", "n_negative",
            ]
        )
    min_cells_ct = int(cfg["protocol"]["metrics"]["group_summaries"]["min_cells_per_slide_cell_type"])
    min_slides_tissue = int(cfg["protocol"]["metrics"]["group_summaries"]["min_slides_per_tissue"])
    slide_group_cols = ["slide_id", "source", "subset", "tissue"]
    macro_tables = benchmark.get("macro_tables", {})
    per_slide_corr = macro_tables.get("slide_macro_corr", pd.DataFrame())
    per_slide_mag = macro_tables.get("slide_macro_magnitude", pd.DataFrame())
    per_slide_loc = macro_tables.get("slide_macro_location_profile", pd.DataFrame())

    base_parts = []
    for df in [per_slide_corr, per_slide_mag, per_slide_loc]:
        if isinstance(df, pd.DataFrame) and not df.empty:
            base_parts.append(df[slide_group_cols].drop_duplicates())
    per_slide = pd.concat(base_parts, axis=0, ignore_index=True).drop_duplicates() if base_parts else pd.DataFrame(columns=slide_group_cols)

    for df, cols in [
        (per_slide_corr, ["slide_macro_spearman", "slide_macro_pearson",
                          # Per-slide ceiling-normalized values
                          "slide_macro_spearman_ceiling_norm", "slide_macro_pearson_ceiling_norm",
                          "n_valid_corr_genes", "eligible_for_slide_macro_corr"]),
        (per_slide_mag, ["slide_macro_mae", "slide_macro_rmse", "slide_macro_huber", "n_valid_mag_genes", "eligible_for_slide_macro_magnitude"]),
        (per_slide_loc, ["location_macro_profile_spearman",
                         # Per-slide ceiling-normalized value
                         "location_macro_profile_spearman_ceiling_norm",
                         "n_valid_locations"]),
    ]:
        if isinstance(df, pd.DataFrame) and not df.empty:
            # *_ceiling_norm columns are absent without reliability
            have = [c for c in cols if c in df.columns]
            if have:
                per_slide = per_slide.merge(df[slide_group_cols + have], on=slide_group_cols, how="left")

    if per_slide.empty:
        by_tissue = pd.DataFrame(columns=["tissue", "n_slides", "slide_macro_spearman", "slide_macro_pearson", "slide_macro_mae", "slide_macro_rmse", "slide_macro_huber", "location_macro_profile_spearman", "descriptive_only"])
    else:
        per_slide = _ensure_columns(
            per_slide,
            {
                "slide_macro_spearman": np.nan,
                "slide_macro_pearson": np.nan,
                "slide_macro_mae": np.nan,
                "slide_macro_rmse": np.nan,
                "slide_macro_huber": np.nan,
                "location_macro_profile_spearman": np.nan,
            },
        )
        by_tissue = (
            per_slide.groupby("tissue", dropna=False)
            .agg(
                n_slides=("slide_id", "nunique"),
                slide_macro_spearman=("slide_macro_spearman", "mean"),
                slide_macro_pearson=("slide_macro_pearson", "mean"),
                slide_macro_mae=("slide_macro_mae", "mean"),
                slide_macro_rmse=("slide_macro_rmse", "mean"),
                slide_macro_huber=("slide_macro_huber", "mean"),
                location_macro_profile_spearman=("location_macro_profile_spearman", "mean"),
            )
            .reset_index()
        )
        by_tissue["descriptive_only"] = by_tissue["n_slides"] < min_slides_tissue

    calibration_rows = []
    depth_rows = []
    cell_type_rows = []
    for slide_id, bundle in preds.items():
        yt = np.asarray(bundle["y_true"], dtype=np.float32)
        yp = np.asarray(bundle["y_pred"], dtype=np.float32)
        valid = np.isfinite(yt) & np.isfinite(yp)
        tv = yt[valid]
        pv = yp[valid]
        if tv.size:
            if tv.size >= 2 and np.var(tv) > 0:
                slope, intercept = np.polyfit(tv.astype(np.float64), pv.astype(np.float64), deg=1)
            else:
                slope, intercept = np.nan, np.nan
            calibration_rows.append(
                {
                    "slide_id": slide_id,
                    "source": bundle["source"],
                    "subset": bundle["subset"],
                    "tissue": bundle["tissue"],
                    "mean_error": float(np.mean(pv - tv)),
                    "median_error": float(np.median(pv - tv)),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "n_values": int(tv.size),
                }
            )
        det = np.asarray(bundle["detected"], dtype=bool)
        loc_scores = _location_profile_scores(yt, yp, det, cfg)
        # Depth proxy: number of detected genes per location
        row_depth = np.nansum(det.astype(np.float64), axis=1)
        depth_rows.append(
            {
                "slide_id": slide_id,
                "depth_vs_location_profile_spearman": _safe_spearman(row_depth, loc_scores),
                "depth_vs_location_profile_pearson": _safe_pearson(row_depth, loc_scores),
            }
        )
        cell_types = bundle.get("cell_types", None)
        if cell_types is not None:
            cell_types = np.asarray(cell_types, dtype=object)
            for ct in np.unique(cell_types):
                keep = cell_types == ct
                if int(np.sum(keep)) < min_cells_ct:
                    continue
                sub = _subset_summary(yt[keep], yp[keep], det[keep], cfg)
                cell_type_rows.append(
                    {
                        "slide_id": slide_id,
                        "cell_type": str(ct),
                        "n_cells": int(np.sum(keep)),
                        **sub,
                    }
                )

    # Gene-level diagnostics use the benchmark validity masks
    coverage = benchmark.get("coverage", {})
    corr_cov = coverage.get("valid_slides_by_gene", pd.DataFrame())
    mag_cov = coverage.get("valid_mag_slides_by_gene", pd.DataFrame())

    # Descriptive gene properties aggregated over measured slides
    gene_property_rows = []
    for slide_id, bundle in preds.items():
        yt = np.asarray(bundle["y_true"], dtype=np.float32)
        yp = np.asarray(bundle["y_pred"], dtype=np.float32)
        det = np.asarray(bundle["detected"], dtype=bool)
        support_df = bundle.get("train_gene_support", None)
        support_lut = None
        if support_df is not None and len(support_df) > 0:
            support_lut = support_df.set_index("gene", drop=False)
        for g, gene in enumerate(bundle["gene_names"].tolist()):
            t = yt[:, g]
            p = yp[:, g]
            valid = ~np.isnan(t)
            if int(np.sum(valid)) == 0:
                continue
            tv = t[valid]
            pv = p[valid]
            train_support_slides = np.nan
            train_support_rows = np.nan
            train_support_positive_rows = np.nan
            train_nonzero_rate = np.nan
            train_dispersion = np.nan
            if support_lut is not None and str(gene) in support_lut.index:
                srow = support_lut.loc[str(gene)]
                train_support_slides = float(srow["train_support_slides"])
                train_support_rows = float(srow["train_support_rows"])
                train_support_positive_rows = float(srow["train_support_positive_rows"])
                train_nonzero_rate = float(srow["train_nonzero_rate"])
                train_dispersion = float(srow["train_dispersion"])
            gene_property_rows.append(
                {
                    "slide_id": slide_id,
                    "gene": str(gene),
                    "train_support_slides": train_support_slides,
                    "train_support_rows": train_support_rows,
                    "train_support_positive_rows": train_support_positive_rows,
                    "train_nonzero_rate": train_nonzero_rate,
                    "train_dispersion": train_dispersion,
                    "test_nonzero_rate": float(np.mean(det[valid, g])),
                    "test_mean": float(np.mean(tv)),
                    "test_variance": float(np.var(tv)),
                    "mean_error": float(np.mean(pv - tv)),
                    "median_error": float(np.median(pv - tv)),
                }
            )

    gene_props_df = pd.DataFrame(gene_property_rows)
    if gene_props_df.empty:
        gene_props_summary = pd.DataFrame(
            columns=[
                "gene",
                "train_support_slides", "train_support_rows", "train_support_positive_rows",
                "train_nonzero_rate", "train_dispersion",
                "test_nonzero_rate", "test_mean", "test_variance",
                "mean_error", "median_error",
                "n_measured_slides",
            ]
        )
    else:
        gene_props_summary = (
            gene_props_df.groupby("gene", dropna=False)
            .agg(
                train_support_slides=("train_support_slides", "mean"),
                train_support_rows=("train_support_rows", "mean"),
                train_support_positive_rows=("train_support_positive_rows", "mean"),
                train_nonzero_rate=("train_nonzero_rate", "mean"),
                train_dispersion=("train_dispersion", "mean"),
                test_nonzero_rate=("test_nonzero_rate", "mean"),
                test_mean=("test_mean", "mean"),
                test_variance=("test_variance", "mean"),
                mean_error=("mean_error", "mean"),
                median_error=("median_error", "mean"),
                n_measured_slides=("slide_id", "nunique"),
            )
            .reset_index()
        )

    corr_gene = pd.DataFrame(columns=["gene", "gene_macro_spearman", "gene_macro_pearson", "gene_macro_spearman_ceiling_norm", "n_valid_corr_pairs"])
    mag_gene = pd.DataFrame(columns=["gene", "gene_macro_mae", "gene_macro_rmse", "gene_macro_huber", "n_valid_mag_pairs"])

    if not slide_gene_df.empty:
        corr_valid_df = slide_gene_df.loc[slide_gene_df["corr_valid"]].copy()
        if not corr_valid_df.empty:
            corr_agg = {
                "gene_macro_spearman": ("spearman", "mean"),
                "gene_macro_pearson": ("pearson", "mean"),
                "n_valid_corr_pairs": ("slide_id", "size"),
            }
            if "spearman_ceiling_norm" in corr_valid_df.columns:
                corr_agg["gene_macro_spearman_ceiling_norm"] = ("spearman_ceiling_norm", "mean")
            corr_gene = (
                corr_valid_df.groupby("gene", dropna=False)
                .agg(**corr_agg)
                .reset_index()
            )

        mag_valid_df = slide_gene_df.loc[slide_gene_df["mag_valid"]].copy()
        if not mag_valid_df.empty:
            mag_gene = (
                mag_valid_df.groupby("gene", dropna=False)
                .agg(
                    gene_macro_mae=("mae", "mean"),
                    gene_macro_rmse=("rmse", "mean"),
                    gene_macro_huber=("huber", "mean"),
                    n_valid_mag_pairs=("slide_id", "size"),
                )
                .reset_index()
            )


    onoff_df = benchmark.get("per_slide_onoff", pd.DataFrame())
    if onoff_df is None or onoff_df.empty:
        onoff_gene = pd.DataFrame(columns=["gene", "mean_auroc", "mean_auprc", "n_valid_onoff_slides"])
    else:
        onoff_gene = (
            onoff_df.groupby("gene", dropna=False)
            .agg(
                mean_auroc=("auroc", "mean"),
                mean_auprc=("auprc", "mean"),
                n_valid_onoff_slides=("slide_id", "nunique"),
            )
            .reset_index()
        )

    gene_parts = [df[["gene"]].copy() for df in [gene_props_summary, corr_gene, mag_gene, onoff_gene] if not df.empty]
    if gene_parts:
        gene_summary = pd.concat(gene_parts, axis=0, ignore_index=True).drop_duplicates(subset=["gene"], keep="first")
    else:
        gene_summary = pd.DataFrame(columns=["gene"])

    for df in [gene_props_summary, corr_gene, mag_gene, onoff_gene]:
        if not df.empty:
            gene_summary = gene_summary.merge(df, on="gene", how="left")

    if not corr_cov.empty:
        gene_summary = gene_summary.merge(
            corr_cov[["gene", "n_valid_corr_slides", "eligible_for_gene_macro_corr"]],
            on="gene",
            how="left",
        )
    else:
        gene_summary["n_valid_corr_slides"] = 0
        gene_summary["eligible_for_gene_macro_corr"] = False

    if not mag_cov.empty:
        gene_summary = gene_summary.merge(
            mag_cov[["gene", "n_valid_mag_slides", "eligible_for_gene_macro_magnitude"]],
            on="gene",
            how="left",
        )
    else:
        gene_summary["n_valid_mag_slides"] = 0
        gene_summary["eligible_for_gene_macro_magnitude"] = False

    for col in [
        "gene_macro_spearman", "gene_macro_pearson", "gene_macro_spearman_ceiling_norm",
        "gene_macro_mae", "gene_macro_rmse", "gene_macro_huber",
        "mean_auroc", "mean_auprc",
    ]:
        if col not in gene_summary.columns:
            gene_summary[col] = np.nan

    for col in [
        "n_valid_corr_pairs", "n_valid_corr_slides",
        "n_valid_mag_pairs", "n_valid_mag_slides",
        "n_valid_onoff_slides", "n_measured_slides",
    ]:
        if col not in gene_summary.columns:
            gene_summary[col] = 0

    for col in [
        "eligible_for_gene_macro_corr",
        "eligible_for_gene_macro_magnitude",
    ]:
        if col not in gene_summary.columns:
            gene_summary[col] = False
        gene_summary[col] = gene_summary[col].fillna(False).astype(bool)

    # Column aliases
    gene_summary["spearman"] = gene_summary["gene_macro_spearman"]
    gene_summary["pearson"] = gene_summary["gene_macro_pearson"]
    gene_summary["n_slides"] = gene_summary["n_measured_slides"]

    gene_summary = gene_summary.sort_values(
        ["eligible_for_gene_macro_corr", "gene_macro_spearman", "n_valid_corr_slides", "gene"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    # Gene-property strata: cell-type-specificity and spatial-structure terciles
    gene_property_strata, gene_strata_continuous = _build_gene_strata_outputs(
        preds, gene_summary, gene_strata, cfg
    )
    # External-signature strata
    gene_signature_strata = _build_signature_strata_outputs(
        preds, gene_summary, signature_strata, cfg
    )

    calibration_df = pd.DataFrame(calibration_rows)
    depth_df = pd.DataFrame(depth_rows)
    cell_type_df = pd.DataFrame(cell_type_rows)

    top_rank_source = gene_summary.loc[
        gene_summary["eligible_for_gene_macro_corr"] & np.isfinite(gene_summary["gene_macro_spearman"])
    ].copy()
    if top_rank_source.empty:
        top_rank_source = gene_summary.loc[np.isfinite(gene_summary["gene_macro_spearman"])].copy()
    if top_rank_source.empty:
        top_pred_genes = gene_summary.head(25).copy()
        bottom_pred_genes = gene_summary.tail(25).copy()
    else:
        top_pred_genes = top_rank_source.sort_values(
            ["gene_macro_spearman", "n_valid_corr_slides", "gene"],
            ascending=[False, False, True],
        ).head(25).copy()
        bottom_pred_genes = top_rank_source.sort_values(
            ["gene_macro_spearman", "n_valid_corr_slides", "gene"],
            ascending=[True, False, True],
        ).head(25).copy()

    diagnostics_summary = {
        "mean_auroc": _df_col_mean(onoff_df, "auroc"),
        "mean_auprc": _df_col_mean(onoff_df, "auprc"),
        "n_valid_onoff_pairs": int(onoff_df.shape[0]),
        # on/off metrics do not apply on the signature basis
        "onoff_applicable": not is_signature_basis(cfg),
        "onoff_not_applicable_reason": (
            ""
            if not is_signature_basis(cfg)
            else (
                "target_basis=signatures: a signature score is a weighted mean of "
                "z-scored members and carries no detection (on/off) state, so AUROC / "
                "AUPRC are undefined on this axis."
            )
        ),
        "mean_calibration_mean_error": _df_col_mean(calibration_df, "mean_error"),
        "mean_calibration_median_error": _df_col_mean(calibration_df, "median_error"),
        "mean_calibration_slope": _df_col_mean(calibration_df, "slope"),
        "mean_calibration_intercept": _df_col_mean(calibration_df, "intercept"),
        "mean_depth_vs_location_profile_spearman": _df_col_mean(depth_df, "depth_vs_location_profile_spearman"),
        "mean_depth_vs_location_profile_pearson": _df_col_mean(depth_df, "depth_vs_location_profile_pearson"),
        "n_cell_type_groups": int(cell_type_df.shape[0]),
        "n_eligible_slides_for_slide_macro_corr": int(per_slide_corr["eligible_for_slide_macro_corr"].sum()) if ("eligible_for_slide_macro_corr" in per_slide_corr.columns) else 0,
        "n_eligible_genes_for_gene_macro_corr": int(corr_cov["eligible_for_gene_macro_corr"].sum()) if ("eligible_for_gene_macro_corr" in corr_cov.columns) else 0,
        "n_eligible_genes_for_gene_macro_magnitude": int(mag_cov["eligible_for_gene_macro_magnitude"].sum()) if ("eligible_for_gene_macro_magnitude" in mag_cov.columns) else 0,
    }

    out = {
        "per_slide": per_slide,
        "by_tissue": by_tissue,
        "by_cell_type": cell_type_df,
        "calibration_by_slide": calibration_df,
        "depth_robustness": depth_df,
        "per_gene": gene_summary,
        "gene_property_strata": gene_property_strata,
        "gene_strata_continuous": gene_strata_continuous,
        "gene_signature_strata": gene_signature_strata,
        "top_pred_genes": top_pred_genes,
        "bottom_pred_genes": bottom_pred_genes,
        "diagnostics_summary": diagnostics_summary,
    }
    return out


# -----------------------------------------------------------------------------
# Level 3 structure analyses
# -----------------------------------------------------------------------------


def _flatten_upper_triangle(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]


def _knn_weights(coords_xy: np.ndarray, n_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(coords_xy)), metric="euclidean")
    nn.fit(coords_xy)
    dist, idx = nn.kneighbors(coords_xy)
    return dist[:, 1:], idx[:, 1:]


def _moran_i_from_graph(x: np.ndarray, dist: np.ndarray, idx: np.ndarray) -> float:
    """Moran's I of x on a precomputed kNN graph (see _moran_i)."""
    if len(x) < 3 or np.var(x) <= 0:
        return np.nan
    w = np.exp(-dist / max(np.median(dist), 1e-6))
    z = x - np.mean(x)
    N = len(x)
    W = np.sum(w)
    num_I = 0.0
    for i in range(N):
        num_I += float(np.sum(w[i] * z[i] * z[idx[i]]))
    den = float(np.sum(np.square(z)))
    return (N / W) * (num_I / den) if den > 0 and W > 0 else np.nan


def _moran_i(x: np.ndarray, coords_xy: np.ndarray, n_neighbors: int) -> float:
    """Moran's I of x on a distance-weighted kNN graph of coords_xy."""
    if len(x) < 3 or np.var(x) <= 0:
        return np.nan
    return _moran_i_from_graph(x, *_knn_weights(coords_xy, n_neighbors))


# -----------------------------------------------------------------------------
# Neighborhood preservation (Metric 1) + count-thinning self-consistency ceiling
# -----------------------------------------------------------------------------


def _knn_index_matrix(X: np.ndarray, k: int) -> Tuple[np.ndarray, int]:
    """k-nearest-neighbor index matrix (self excluded), Euclidean. Deterministic."""
    n = int(X.shape[0])
    k_eff = int(min(k, n - 1))
    if k_eff < 1:
        return np.empty((n, 0), dtype=np.int64), 0
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:].astype(np.int64, copy=False), k_eff


def _knn_overlap_fraction(idx_a: np.ndarray, idx_b: np.ndarray, k_eff: int) -> float:
    """Mean |kNN_a(i) ∩ kNN_b(i)| / k over rows -- the headline neighborhood overlap."""
    n = int(idx_a.shape[0])
    if n == 0 or k_eff <= 0:
        return np.nan
    shared = np.empty(n, dtype=np.float64)
    for i in range(n):
        shared[i] = np.intersect1d(idx_a[i], idx_b[i], assume_unique=True).size
    return float(np.mean(shared / float(k_eff)))


def _knn_overlap_for_pair(A: np.ndarray, B: np.ndarray, k: int) -> float:
    """Neighborhood overlap@k between two same-row representations A and B."""
    idx_a, ke_a = _knn_index_matrix(A, k)
    idx_b, ke_b = _knn_index_matrix(B, k)
    if ke_a < 1 or ke_a != ke_b:
        return np.nan
    return _knn_overlap_fraction(idx_a, idx_b, ke_a)


def _subsample_rows(n: int, max_points: int, seed: int) -> Optional[np.ndarray]:
    """Deterministic sorted row subsample, or None to keep all rows."""
    if n <= max_points:
        return None
    rng = np.random.default_rng(int(seed))
    sel = rng.choice(n, size=int(max_points), replace=False)
    sel.sort()
    return sel


def _trustworthiness_continuity(
    X_true: np.ndarray, X_pred: np.ndarray, k: int, *, max_points: int, seed: int
) -> Tuple[float, float, int]:
    """
    Venna-Kaski trustworthiness and continuity (trustworthiness with the spaces swapped), in [0, 1],
    on at most max_points rows (deterministic subsample). Returns (trust, cont, n_used).
    """
    n = int(X_true.shape[0])
    if n < k + 2:
        return np.nan, np.nan, n
    sel = _subsample_rows(n, max_points, seed)
    if sel is not None:
        X_true = X_true[sel]
        X_pred = X_pred[sel]
    m = int(X_true.shape[0])
    k_eff = int(min(k, m - 2))
    if k_eff < 1:
        return np.nan, np.nan, m
    try:
        trust = float(_sk_trustworthiness(X_true, X_pred, n_neighbors=k_eff, metric="euclidean"))
        cont = float(_sk_trustworthiness(X_pred, X_true, n_neighbors=k_eff, metric="euclidean"))
    except Exception:
        return np.nan, np.nan, m
    return trust, cont, m


def compute_structure_analyses(
    preds: Mapping[str, Mapping],
    cfg: Mapping,
    ceiling_halves: Optional[Mapping[str, Tuple[np.ndarray, np.ndarray]]] = None,
    embeddings: Optional[Mapping[str, np.ndarray]] = None,
    embedding_space: str = "",
    held_out: bool = False,
) -> Dict[str, object]:
    rows_coexp = []
    rows_spatial = []
    rows_cluster = []
    rows_within_slide_rank = []
    rows_np = []          # Metric 1: neighborhood preservation / trustworthiness
    rows_teacher = []     # Metric 2: teacher-niche recovery (encoder embedding vs Novae)

    level = str(cfg["protocol"].get("level", ""))
    cluster_k = int(cfg["protocol"]["structure"]["cluster_k"])
    pca_dim = int(cfg["protocol"]["structure"]["cluster_pca_dim"])
    n_neighbors = int(cfg["protocol"]["structure"]["moran_neighbors"])
    protocol_seed = int(cfg["protocol"].get("protocol_seed", 0))
    svcfg = cfg["protocol"]["metrics"]["structure_validity"]

    min_rows_for_coexpression = int(svcfg["min_rows_for_coexpression"])
    min_rows_for_clustering = int(svcfg["min_rows_for_clustering"])
    min_rows_for_moran = int(svcfg["min_rows_for_moran"])
    min_positive_locations_for_moran = int(svcfg["min_positive_locations_for_moran"])
    # Column floors depend on the target basis
    if is_signature_basis(cfg):
        sscfg = _signature_target_cfg(cfg, "structure_validity")
        min_genes_for_coexpression = int(sscfg.get("min_signatures_for_coexpression", 8))
        min_genes_for_clustering = int(sscfg.get("min_signatures_for_clustering", 12))
    else:
        min_genes_for_coexpression = int(svcfg["min_genes_for_coexpression"])
        min_genes_for_clustering = int(svcfg["min_genes_for_clustering"])
    # Per-slide reasons of the skipped column-gated analyses
    rows_skipped: List[Dict[str, object]] = []

    def _record_skip(
        slide_id: str, analysis: str, n_columns: int, floor: int, reason: str = "",
    ) -> None:
        rows_skipped.append(
            {
                "slide_id": str(slide_id),
                "analysis": str(analysis),
                "n_columns": int(n_columns),
                "min_columns_required": int(floor),
                "target_basis": "signatures" if is_signature_basis(cfg) else "genes",
                "reason": reason or (
                    f"{analysis} needs >= {floor} target columns; this slide has "
                    f"{n_columns}"
                ),
            }
        )

    def _coexpression_skip_reason(
        y_true: np.ndarray, y_pred: np.ndarray, min_rows: int, min_genes: int,
    ) -> Tuple[int, str]:
        """(n_columns, reason) for why _prepare_coexpression_inputs returned None (rows, finite columns or variance)."""
        yt = np.asarray(y_true, dtype=np.float64)
        yp = np.asarray(y_pred, dtype=np.float64)
        finite = np.all(np.isfinite(yt), axis=0) & np.all(np.isfinite(yp), axis=0)
        n_finite = int(np.sum(finite))
        if yt.shape[0] < int(min_rows):
            return n_finite, (
                f"coexpression needs >= {min_rows} rows; this slide has {yt.shape[0]}"
            )
        if n_finite < int(min_genes):
            return n_finite, (
                f"coexpression needs >= {min_genes} target columns finite in both truth "
                f"and prediction; this slide has {n_finite}"
            )
        tv = np.var(yt[:, finite], axis=0, dtype=np.float64)
        pv = np.var(yp[:, finite], axis=0, dtype=np.float64)
        n_var = int(np.sum((tv > _VAR_EPS) & (pv > _VAR_EPS)))
        return n_var, (
            f"coexpression needs >= {min_genes} target columns with non-zero variance in "
            f"both truth and prediction; {n_finite} were finite but only {n_var} varied "
            f"({int(np.sum(pv <= _VAR_EPS))} predicted columns were constant, "
            f"{int(np.sum(tv <= _VAR_EPS))} true columns were constant). A constant "
            "prediction usually means the probe landed on the alpha grid's boundary."
        )

    for slide_id, bundle in preds.items():
        y_true = np.asarray(bundle["y_true"], dtype=np.float32)
        y_pred = np.asarray(bundle["y_pred"], dtype=np.float32)
        _assert_no_nonfinite_predictions(slide_id, y_true, y_pred)
        valid_gene = np.any(np.isfinite(y_true), axis=0)
        y_true = y_true[:, valid_gene]
        y_pred = y_pred[:, valid_gene]
        detected = np.asarray(bundle["detected"], dtype=bool)[:, valid_gene]

        # Drop locations with no finite target (zero total counts), for all analyses below
        row_usable = np.any(np.isfinite(y_true), axis=1)
        n_rows_dropped = int(row_usable.size - int(np.sum(row_usable)))
        if n_rows_dropped:
            y_true = y_true[row_usable]
            y_pred = y_pred[row_usable]
            detected = detected[row_usable]

        if y_true.shape[1] == 0:
            rows_within_slide_rank.append({"slide_id": slide_id, "within_slide_rank_spearman": np.nan})
            continue

        valid_flat = np.isfinite(y_true) & np.isfinite(y_pred)
        rows_within_slide_rank.append(
            {
                "slide_id": slide_id,
                "within_slide_rank_spearman": _safe_spearman(y_true[valid_flat], y_pred[valid_flat]),
                # Number of zero-count locations removed above
                "n_locations_dropped_all_nan": n_rows_dropped,
                "n_locations_used": int(y_true.shape[0]),
            }
        )

        # coexpression preservation
        coexp_inputs = _prepare_coexpression_inputs(
            y_true,
            y_pred,
            min_rows=min_rows_for_coexpression,
            min_genes=min_genes_for_coexpression,
        )
        if coexp_inputs is not None:
            y_true_coexp, y_pred_coexp, coexp_info = coexp_inputs
            corr_true = _corrcoef_columns_no_warning(y_true_coexp)
            corr_pred = _corrcoef_columns_no_warning(y_pred_coexp)
            flat_true = _flatten_upper_triangle(corr_true)
            flat_pred = _flatten_upper_triangle(corr_pred)
            rows_coexp.append(
                {
                    "slide_id": slide_id,
                    "coexpression_preservation_pearson": _safe_pearson(flat_true, flat_pred),
                    "coexpression_preservation_spearman": _safe_spearman(flat_true, flat_pred),
                    **coexp_info,
                }
            )
        else:
            n_cols, why = _coexpression_skip_reason(
                y_true, y_pred, min_rows_for_coexpression, min_genes_for_coexpression
            )
            _record_skip(
                slide_id, "coexpression", n_cols, min_genes_for_coexpression, reason=why
            )

        # spatial autocorrelation preservation
        # Same row set as y_true/y_pred
        coords = np.asarray(bundle["coords_xy"], dtype=np.float64)
        if n_rows_dropped:
            coords = coords[row_usable]
        if y_true.shape[0] >= min_rows_for_moran:
            moran_true, moran_pred = [], []
            # kNN graphs cached by per-gene finite mask (one graph per slide in practice)
            moran_graph_cache: Dict[bytes, Tuple[np.ndarray, np.ndarray]] = {}
            for g in range(y_true.shape[1]):
                xt = y_true[:, g]
                xp = y_pred[:, g]

                finite = np.isfinite(xt) & np.isfinite(xp)
                if int(np.sum(finite)) < min_rows_for_moran:
                    continue
                det_g = detected[finite, g]
                xt = xt[finite]
                xp = xp[finite]
                coords_g = coords[finite]

                if int(np.sum(det_g)) < min_positive_locations_for_moran:
                    continue
                graph_key = finite.tobytes()
                graph = moran_graph_cache.get(graph_key)
                if graph is None:
                    graph = _knn_weights(coords_g, n_neighbors)
                    moran_graph_cache[graph_key] = graph
                moran_true.append(_moran_i_from_graph(xt, *graph))
                moran_pred.append(_moran_i_from_graph(xp, *graph))
            if moran_true:
                rows_spatial.append(
                    {
                        "slide_id": slide_id,
                        "moran_preservation_spearman": _safe_spearman(np.asarray(moran_true), np.asarray(moran_pred)),
                    }
                )

        # clustering agreement + external labels agreement
        cluster_gene_mask = np.all(np.isfinite(y_true), axis=0) & np.all(np.isfinite(y_pred), axis=0)
        if int(np.sum(cluster_gene_mask)) < min_genes_for_clustering:
            _record_skip(
                slide_id, "clustering", int(np.sum(cluster_gene_mask)), min_genes_for_clustering
            )
        if y_true.shape[0] >= min_rows_for_clustering and int(np.sum(cluster_gene_mask)) >= min_genes_for_clustering:
            X_true = y_true[:, cluster_gene_mask].astype(np.float32, copy=False)
            X_pred = y_pred[:, cluster_gene_mask].astype(np.float32, copy=False)
            d = min(pca_dim, X_true.shape[1], max(2, min(X_true.shape[0] - 1, pca_dim)))
            if d >= 2:
                z_true = PCA(n_components=d, random_state=0).fit_transform(X_true)
                z_pred = PCA(n_components=d, random_state=0).fit_transform(X_pred)
                k = min(cluster_k, max(2, int(np.sqrt(X_true.shape[0]))))
                c_true = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(z_true)
                c_pred = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(z_pred)
                row = {"slide_id": slide_id, "cluster_ami_true_vs_pred": float(adjusted_mutual_info_score(c_true, c_pred))}
                obs = bundle.get("obs_df", None)
                # obs rows aligned with the dropped-row space
                if obs is not None and n_rows_dropped:
                    obs = obs.iloc[np.flatnonzero(row_usable)]
                if obs is not None:
                    if level.startswith("patch"):
                        novae_labels = _resolve_novae_domain_labels(obs)
                        if novae_labels is not None:
                            valid_novae = _valid_novae_label_mask(novae_labels)
                            if (
                                int(np.sum(valid_novae)) >= min_rows_for_clustering
                                and pd.Series(novae_labels[valid_novae]).nunique(dropna=True) >= 2
                            ):
                                row["cluster_ami_pred_vs_novae"] = float(
                                    adjusted_mutual_info_score(
                                        c_pred[valid_novae],
                                        novae_labels[valid_novae].astype(str),
                                    )
                                )
                                row["n_rows_with_novae_domain"] = int(np.sum(valid_novae))

                    elif level == "cell" and "cell_type" in obs.columns:
                        cell_type = pd.Series(obs["cell_type"], dtype="string")
                        valid_cell_type = (
                            cell_type.notna()
                            & ~cell_type.str.strip().str.lower().isin(_NULL_DOMAIN_STRINGS)
                        ).to_numpy(dtype=bool)

                        if (
                            int(np.sum(valid_cell_type)) >= min_rows_for_clustering
                            and cell_type.loc[valid_cell_type].nunique(dropna=True) >= 2
                        ):
                            row["cluster_ami_pred_vs_cell_type"] = float(
                                adjusted_mutual_info_score(
                                    c_pred[valid_cell_type],
                                    cell_type.loc[valid_cell_type].astype(str).to_numpy(),
                                )
                            )
                            row["n_rows_with_cell_type"] = int(np.sum(valid_cell_type))

                rows_cluster.append(row)

        # --- Metric 1: neighborhood preservation / trustworthiness ---
        # kNN overlap of true vs predicted expression, normalized by the overlap of the two count-thinning halves
        np_gene_mask = np.all(np.isfinite(y_true), axis=0) & np.all(np.isfinite(y_pred), axis=0)
        if int(np.sum(np_gene_mask)) < min_genes_for_clustering:
            _record_skip(
                slide_id,
                "neighborhood_preservation",
                int(np.sum(np_gene_mask)),
                min_genes_for_clustering,
            )
        if (
            int(np.sum(np_gene_mask)) >= min_genes_for_clustering
            and y_true.shape[0] >= min_rows_for_clustering
        ):
            Xt_all = y_true[:, np_gene_mask].astype(np.float64, copy=False)
            Xp_all = y_pred[:, np_gene_mask].astype(np.float64, copy=False)
            row_ok = np.all(np.isfinite(Xt_all), axis=1) & np.all(np.isfinite(Xp_all), axis=1)
            Xt = Xt_all[row_ok]
            Xp = Xp_all[row_ok]
            n_np = int(Xt.shape[0])
            if n_np >= _NP_HEADLINE_K + 2:
                np_row: Dict[str, object] = {"slide_id": slide_id, "n_valid_locations": n_np}
                for k in _NP_K_LIST:
                    np_row[f"knn_overlap_k{k}"] = _knn_overlap_for_pair(Xt, Xp, k)
                np_row["knn_overlap"] = float(np_row.get(f"knn_overlap_k{_NP_HEADLINE_K}", np.nan))
                tr, co, n_used = _trustworthiness_continuity(
                    Xt, Xp, _NP_HEADLINE_K, max_points=_NP_TC_MAX_POINTS, seed=protocol_seed
                )
                np_row["trustworthiness"] = tr
                np_row["continuity"] = co
                np_row["n_used_trust_continuity"] = int(n_used)
                ceiling = np.nan
                if ceiling_halves is not None and slide_id in ceiling_halves:
                    # Apply the same row drop to the halves before row_ok
                    t1_full, t2_full = ceiling_halves[slide_id]
                    t1_full = np.asarray(t1_full)
                    t2_full = np.asarray(t2_full)
                    if n_rows_dropped:
                        t1_full = t1_full[row_usable]
                        t2_full = t2_full[row_usable]
                    t1m = t1_full[:, valid_gene][:, np_gene_mask][row_ok]
                    t2m = t2_full[:, valid_gene][:, np_gene_mask][row_ok]
                    cok = np.all(np.isfinite(t1m), axis=1) & np.all(np.isfinite(t2m), axis=1)
                    if int(np.sum(cok)) >= _NP_HEADLINE_K + 2:
                        ceiling = _knn_overlap_for_pair(t1m[cok], t2m[cok], _NP_HEADLINE_K)
                np_row["knn_overlap_ceiling"] = float(ceiling) if np.isfinite(ceiling) else np.nan
                np_row["ceiling_available"] = bool(np.isfinite(ceiling))
                ov = float(np_row["knn_overlap"])
                np_row["knn_overlap_ceiling_norm"] = (
                    ov / float(np.clip(ceiling, _NP_CEILING_EPS, 1.0))
                    if (np.isfinite(ov) and np.isfinite(ceiling))
                    else np.nan
                )
                rows_np.append(np_row)

        # --- Metric 2: teacher-niche recovery ---
        # AMI / ARI between the clustering of the encoder embedding and the Novae domains (skipped without them)
        if embeddings is not None and slide_id in embeddings and level.startswith("patch"):
            obs = bundle.get("obs_df", None)
            novae_labels = _resolve_novae_domain_labels(obs) if obs is not None else None
            emb = np.asarray(embeddings[slide_id], dtype=np.float64)
            if novae_labels is not None and emb.ndim == 2 and emb.shape[0] == novae_labels.shape[0]:
                valid_novae = _valid_novae_label_mask(novae_labels)
                E = emb[valid_novae]
                lab = novae_labels[valid_novae].astype(str)
                erow_ok = np.all(np.isfinite(E), axis=1)
                E = E[erow_ok]
                lab = lab[erow_ok]
                n_dom = int(pd.Series(lab).nunique()) if lab.size else 0
                if E.shape[0] >= min_rows_for_clustering and n_dom >= 2:
                    K = int(min(max(n_dom, 2), max(2, E.shape[0] - 1)))
                    d = min(pca_dim, E.shape[1], max(2, min(E.shape[0] - 1, pca_dim)))
                    Z = (
                        PCA(n_components=d, random_state=0).fit_transform(E)
                        if (d >= 2 and E.shape[1] > d)
                        else E
                    )
                    L_he = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(Z)
                    rows_teacher.append(
                        {
                            "slide_id": slide_id,
                            "teacher_niche_recovery_ami": float(adjusted_mutual_info_score(lab, L_he)),
                            "teacher_niche_recovery_ari": float(adjusted_rand_score(lab, L_he)),
                            "teacher_niche_K": int(K),
                            "n_rows_with_novae_domain": int(E.shape[0]),
                            "is_held_out": bool(held_out),
                            "sub_eval_configuration": str(cfg["protocol"].get("sub_eval_configuration", "")),
                            "embedding_space": str(embedding_space),
                        }
                    )

    coexp_df = pd.DataFrame(rows_coexp)
    spatial_df = pd.DataFrame(rows_spatial)
    cluster_df = pd.DataFrame(rows_cluster)
    rank_df = pd.DataFrame(rows_within_slide_rank)
    np_df = pd.DataFrame(rows_np)
    teacher_df = pd.DataFrame(rows_teacher)

    structure_summary = {
        "mean_coexpression_preservation_pearson": _df_col_mean(coexp_df, "coexpression_preservation_pearson"),
        "mean_coexpression_preservation_spearman": _df_col_mean(coexp_df, "coexpression_preservation_spearman"),
        "mean_moran_preservation_spearman": _df_col_mean(spatial_df, "moran_preservation_spearman"),
        "mean_cluster_ami_true_vs_pred": _df_col_mean(cluster_df, "cluster_ami_true_vs_pred"),
        "mean_cluster_ami_pred_vs_novae": _df_col_mean(cluster_df, "cluster_ami_pred_vs_novae"),
        "mean_cluster_ami_pred_vs_cell_type": _df_col_mean(cluster_df, "cluster_ami_pred_vs_cell_type"),
        "mean_within_slide_rank_spearman": _df_col_mean(rank_df, "within_slide_rank_spearman"),
        # Metric 1 (neighborhood preservation), macro over slides
        "mean_knn_overlap": _df_col_mean(np_df, "knn_overlap"),
        "mean_knn_overlap_ceiling_norm": _df_col_mean(np_df, "knn_overlap_ceiling_norm"),
        "mean_trustworthiness": _df_col_mean(np_df, "trustworthiness"),
        "mean_continuity": _df_col_mean(np_df, "continuity"),
        # Metric 2 (teacher-niche recovery), macro over slides
        "mean_teacher_niche_recovery_ami": _df_col_mean(teacher_df, "teacher_niche_recovery_ami"),
        "mean_teacher_niche_recovery_ari": _df_col_mean(teacher_df, "teacher_niche_recovery_ari"),
    }

    skipped_df = pd.DataFrame(
        rows_skipped,
        columns=[
            "slide_id", "analysis", "n_columns", "min_columns_required",
            "target_basis", "reason",
        ],
    )

    return {
        "coexpression": coexp_df,
        "spatial_autocorrelation": spatial_df,
        "clustering": cluster_df,
        "within_slide_rank": rank_df,
        "neighborhood_preservation": np_df,
        "teacher_niche_recovery": teacher_df,
        # Per-slide reasons of the skipped column-gated analyses
        "structure_skipped": skipped_df,
        "structure_summary": structure_summary,
    }
