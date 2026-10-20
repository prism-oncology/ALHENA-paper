"""Structure metrics of the Task A-0 UMAP families (kNN label purity, effective rank, top-1 variance fraction)."""
import hashlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Repo root, so `src.preclip_investigation.geometry` resolves when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preclip_investigation.geometry import effective_rank  # noqa: E402


logger = logging.getLogger(__name__)

EPS = 1e-12

# Reading direction of each label's purity (emitted per row)
HIGHER_IS_BETTER = "higher_is_better"
LOWER_IS_BETTER = "lower_is_better"

METRIC_DIRECTION: Dict[str, str] = {
    "slide_id": LOWER_IS_BETTER,
    "modality": LOWER_IS_BETTER,
    "tissue_HistoST": HIGHER_IS_BETTER,
    "source": LOWER_IS_BETTER,
    "subset": LOWER_IS_BETTER,
}

STATUS_OK = "ok"
STATUS_INSUFFICIENT = "insufficient_resolution"

# Rows of a family's matrix (emitted per row):
#   he_only   every row is an H&E embedding
#   st_only   every row is an ST teacher embedding
#   joint     the matrix is `concat([he_rows, st_rows])`
POPULATION_HE_ONLY = "he_only"
POPULATION_ST_ONLY = "st_only"
POPULATION_JOINT = "joint"

# Only `modality` is measured on a `joint` matrix
JOINT_LABEL_COLUMNS: Tuple[str, ...] = ("modality",)

# Labels whose kNN purity is computed without the same-group reference mask
UNMASKED_LABEL_COLUMNS = frozenset({"slide_id", "modality", "source", "subset"})


def slide_cohort_key(slide_ids: Sequence[object]) -> str:
    """Short, stable content key of the slide set a family's matrix was built from."""
    uniq = sorted({str(s) for s in slide_ids})
    if not uniq:
        return ""
    return hashlib.sha256("|".join(uniq).encode("utf-8")).hexdigest()[:12]


class MetricsUnresolvableError(RuntimeError):
    """Raised by `assert_any_metric_resolvable` when a config resolves no label at all."""


def direction_for(label_col: str) -> str:
    """Reading direction of a label's purity. Cell-type axes are all higher-is-better."""
    if label_col in METRIC_DIRECTION:
        return METRIC_DIRECTION[label_col]
    if label_col.startswith("cell_type"):
        return HIGHER_IS_BETTER
    return HIGHER_IS_BETTER


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """Row-L2 normalize so a dot product is the cosine. Zero rows are left at zero."""
    x = np.asarray(x, dtype=np.float32)
    n = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    return x / np.maximum(n, EPS)


def top1_var_fraction(x: np.ndarray, max_points: int = 20_000, seed: int = 0) -> float:
    """Variance share of the leading singular direction (uncentered, on the raw matrix, as in `_1`)."""
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] < 2:
        return float("nan")
    if x.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        x = x[rng.choice(x.shape[0], size=max_points, replace=False)]
    s = np.linalg.svd(x, compute_uv=False)
    total = float(np.sum(s * s))
    if total <= EPS:
        return float("nan")
    return float((s[0] * s[0]) / total)


def label_knn_purity(
    Xn: np.ndarray,
    labels: np.ndarray,
    *,
    group_ids: Optional[np.ndarray] = None,
    k: int = 15,
    chunk: int = 256,
) -> Dict[str, object]:
    """
    kNN label purity with an optional same-group reference mask (`group_ids`), plus its chance level.
    Returns per-label rows (purity, chance = label prevalence in the reference pool, excess_over_chance)
    and a macro summary.
    """
    Xn = np.asarray(Xn, dtype=np.float32)
    labels = np.asarray(labels, dtype=object).reshape(-1)
    n = int(Xn.shape[0])
    if labels.shape[0] != n:
        raise ValueError(f"labels ({labels.shape[0]}) and rows ({n}) disagree.")

    uniq = [str(u) for u in pd.unique(pd.Series(labels).astype(str))]
    per_label: List[Dict[str, object]] = []
    if n < k + 2 or len(uniq) < 2:
        return {"per_label": per_label, "macro": {}, "k": int(k), "n_points": n}

    lab_str = pd.Series(labels).astype(str).to_numpy()
    grp = None if group_ids is None else pd.Series(group_ids).astype(str).to_numpy()

    purities = np.full(n, np.nan, dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        sims = Xn[start:end] @ Xn.T                       # cosine, rows are unit-norm
        rows = np.arange(start, end)
        sims[np.arange(end - start), rows] = -np.inf      # never retrieve self
        if grp is not None:
            same_group = grp[rows][:, None] == grp[None, :]
            sims[same_group] = -np.inf
        k_eff = min(k, max(1, sims.shape[1] - 1))
        nbr = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]
        valid = np.isfinite(np.take_along_axis(sims, nbr, axis=1))
        nbr_lab = lab_str[nbr]
        match = (nbr_lab == lab_str[rows][:, None]) & valid
        denom = valid.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            purities[start:end] = np.where(denom > 0, match.sum(axis=1) / np.maximum(denom, 1), np.nan)

    for label in uniq:
        mask = lab_str == label
        vals = purities[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        # Chance = label share of the retrievable pool: (n_L - 1) / (n - 1) without group mask (self excluded),
        # else the query-weighted mean of the label share outside each group
        if grp is None:
            n_label = int(mask.sum())
            chance = float((n_label - 1) / (n - 1)) if n > 1 else float("nan")
        else:
            weights: List[float] = []
            shares: List[float] = []
            for g in pd.unique(pd.Series(grp[mask]).astype(str)):
                other = grp != g
                n_other = int(other.sum())
                if n_other == 0:
                    continue
                # Queries of this label that sit in group g -- their retrievable pool is `other`.
                n_queries_in_group = int((mask & (grp == g)).sum())
                if n_queries_in_group == 0:
                    continue
                shares.append(float((mask & other).sum() / n_other))
                weights.append(float(n_queries_in_group))
            if shares and sum(weights) > 0:
                chance = float(np.average(shares, weights=weights))
            else:
                # Same self-exclusion as the unmasked branch above.
                n_label = int(mask.sum())
                chance = float((n_label - 1) / (n - 1)) if n > 1 else float("nan")
        purity = float(np.mean(vals))
        per_label.append({
            "label": label,
            "n_query": int(vals.size),
            "purity": purity,
            "chance": chance,
            "excess_over_chance": purity - chance,
        })

    macro: Dict[str, object] = {}
    if per_label:
        macro = {
            "macro_purity": float(np.mean([r["purity"] for r in per_label])),
            "macro_chance": float(np.mean([r["chance"] for r in per_label])),
            "macro_excess_over_chance": float(np.mean([r["excess_over_chance"] for r in per_label])),
            "n_labels": len(per_label),
        }
    return {"per_label": per_label, "macro": macro, "k": int(k), "n_points": n}


def compute_umap_family_metrics(
    emb: np.ndarray,
    meta_df: pd.DataFrame,
    *,
    label_cols: Sequence[str],
    group_col: Optional[str] = "slide_id",
    k: int = 15,
    min_points_per_label: int = 30,
    seed: int = 0,
    population: str = POPULATION_HE_ONLY,
) -> Dict[str, object]:
    """
    Structure metrics for one UMAP family, on its raw embedding matrix.
    `population` (he_only / st_only / joint) is recorded with every row. A label column with fewer than
    2 labels reaching `min_points_per_label` points is recorded as `insufficient_resolution` (null values).
    """
    emb = np.asarray(emb, dtype=np.float32)
    # Slide cohort of this family (also recorded for an empty family)
    cohort_ids = (
        meta_df["slide_id"].astype(str).tolist() if "slide_id" in meta_df.columns else []
    )
    cohort = {
        "n_slides": int(pd.unique(pd.Series(cohort_ids, dtype=object)).size) if cohort_ids else 0,
        "slide_cohort_key": slide_cohort_key(cohort_ids),
    }
    if emb.ndim != 2 or emb.shape[0] == 0:
        return {"labels": {}, "global": dict(cohort), "k": int(k),
                "min_points_per_label": int(min_points_per_label),
                "population": str(population)}

    Xn = l2_normalize_rows(emb)
    out_labels: Dict[str, object] = {}

    for col in label_cols:
        if col not in meta_df.columns:
            continue
        series = meta_df[col].astype(str)
        counts = series.value_counts()
        eligible = counts[counts >= int(min_points_per_label)]
        entry: Dict[str, object] = {
            "direction": direction_for(col),
            "group_masked_by": group_col if (group_col and col not in UNMASKED_LABEL_COLUMNS) else None,
            "n_labels_total": int(counts.size),
            "n_labels_eligible": int(eligible.size),
            "min_points_per_label": int(min_points_per_label),
        }
        if eligible.size < 2:
            entry.update({
                "status": STATUS_INSUFFICIENT,
                "macro_purity": None,
                "macro_chance": None,
                "macro_excess_over_chance": None,
                "reason": (
                    f"only {int(eligible.size)} label(s) reach the {min_points_per_label}-point "
                    f"floor out of {int(counts.size)} present"
                ),
                "per_label": [],
            })
            out_labels[col] = entry
            continue

        keep = series.isin(eligible.index).to_numpy()
        # Same-group reference mask (not for UNMASKED_LABEL_COLUMNS)
        groups = None
        mask_disabled_reason = None
        if group_col and group_col in meta_df.columns and col not in UNMASKED_LABEL_COLUMNS:
            candidate = meta_df.loc[keep, group_col].to_numpy()
            # With a single group (e.g. single-slide configs) the mask would remove every reference:
            # measured unmasked, with the reason recorded
            if pd.unique(pd.Series(candidate).astype(str)).size >= 2:
                groups = candidate
            else:
                mask_disabled_reason = (
                    f"only one distinct {group_col} among the eligible points; measured unmasked "
                    "(within-group)"
                )
                entry["group_masked_by"] = None
                entry["mask_disabled_reason"] = mask_disabled_reason
        res = label_knn_purity(
            Xn[keep], series[keep].to_numpy(), group_ids=groups, k=k
        )
        entry.update({
            "status": STATUS_OK if res["macro"] else STATUS_INSUFFICIENT,
            "k": res["k"],
            "n_points": res["n_points"],
            "per_label": res["per_label"],
            **{key: res["macro"].get(key) for key in
               ("macro_purity", "macro_chance", "macro_excess_over_chance", "n_labels")},
        })
        out_labels[col] = entry

    return {
        "labels": out_labels,
        "global": {
            **cohort,
            "n_points": int(emb.shape[0]),
            "embedding_dim": int(emb.shape[1]),
            # Geometry metrics on the raw matrix (not the row-normalized Xn), as in `_1`
            "effective_rank": float(effective_rank(emb, seed=seed)),
            "top1_var_fraction": top1_var_fraction(emb, seed=seed),
        },
        "k": int(k),
        "min_points_per_label": int(min_points_per_label),
        "population": str(population),
    }


def structure_metrics_to_rows(
    metrics: Dict[str, object], *, config_name: str, family: str
) -> List[Dict[str, object]]:
    """Flatten one family's metrics into long rows for `umap_metrics.csv`."""
    rows: List[Dict[str, object]] = []
    glob = metrics.get("global", {}) or {}
    # `population` is written on every row
    population = metrics.get("population", POPULATION_HE_ONLY)
    for col, entry in (metrics.get("labels", {}) or {}).items():
        rows.append({
            "config_name": config_name,
            "family": family,
            "population": population,
            "label_column": col,
            "status": entry.get("status"),
            "direction": entry.get("direction"),
            "group_masked_by": entry.get("group_masked_by"),
            "mask_disabled_reason": entry.get("mask_disabled_reason"),
            "k": entry.get("k"),
            "n_points": entry.get("n_points"),
            "n_labels_eligible": entry.get("n_labels_eligible"),
            "n_labels_total": entry.get("n_labels_total"),
            "min_points_per_label": entry.get("min_points_per_label"),
            "macro_purity": entry.get("macro_purity"),
            "macro_chance": entry.get("macro_chance"),
            "macro_excess_over_chance": entry.get("macro_excess_over_chance"),
            "reason": entry.get("reason"),
            "effective_rank": glob.get("effective_rank"),
            "top1_var_fraction": glob.get("top1_var_fraction"),
            "embedding_dim": glob.get("embedding_dim"),
            # Slide cohort
            "n_slides": glob.get("n_slides"),
            "slide_cohort_key": glob.get("slide_cohort_key"),
        })
    return rows


def assert_any_metric_resolvable(
    rows: Sequence[Dict[str, object]], *, config_name: str, n_slides: int = 2
) -> None:
    """
    Raise MetricsUnresolvableError if a config produced label metrics but no resolvable one.
    Single-slide configs (`n_slides < 2`) only log it.
    """
    if not rows:
        return
    if any(r.get("status") == STATUS_OK for r in rows):
        return
    reasons = sorted({str(r.get("reason")) for r in rows if r.get("reason")})
    if int(n_slides) < 2:
        logger.info(
            "UMAP structure metrics for single-slide config '%s': no resolvable label "
            "(expected -- slide_id and tissue_HistoST are single-valued here). Reasons: %s",
            config_name,
            reasons,
        )
        return
    raise MetricsUnresolvableError(
        f"UMAP structure metrics for config '{config_name}' ({n_slides} slides): every label came "
        f"back '{STATUS_INSUFFICIENT}' at the current sampling budget (--Np / --Nc). Reasons: {reasons}. "
        "Raise the budget, lower --min_points_per_label, or drop the metrics for this config."
    )
