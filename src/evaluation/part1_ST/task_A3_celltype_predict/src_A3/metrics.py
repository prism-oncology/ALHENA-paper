"""Classification metrics for Task A.3 (pooled out-of-fold predictions)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

# sys.path setup (as in every src_A3 module)
from src_A3.common import ensure_imports

ensure_imports()

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Classification helpers
# -----------------------------------------------------------------------------


@dataclass
class ClassificationSummary:
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    per_class: pd.DataFrame


def labels_to_ids(values: Sequence[str], label_to_id: Mapping[str, int]) -> np.ndarray:
    out = np.full(len(values), -1, dtype=np.int32)
    for i, v in enumerate(values):
        out[i] = int(label_to_id.get(str(v), -1))
    return out


def confusion_matrix_fixed(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    ok = (y_true >= 0) & (y_pred >= 0)
    yt = y_true[ok]
    yp = y_pred[ok]
    if yt.size == 0:
        return cm
    np.add.at(cm, (yt, yp), 1)
    return cm


def classification_metrics_from_ids(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_order: Sequence[str],
    gt_present_only_for_macro: bool = True,
    macro_min_support: int = 0,
) -> ClassificationSummary:
    # macro_min_support: minimum support of a class to enter the macro average (0: support > 0)
    n_classes = len(label_order)
    cm = confusion_matrix_fixed(y_true, y_pred, n_classes=n_classes)
    support = cm.sum(axis=1)
    pred_support = cm.sum(axis=0)
    tp = np.diag(cm).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.divide(tp, pred_support, out=np.zeros_like(tp), where=pred_support > 0)
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        f1 = np.divide(2.0 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

    if gt_present_only_for_macro:
        macro_mask = support >= max(1, int(macro_min_support))
    else:
        macro_mask = np.ones(n_classes, dtype=bool)

    total = int(cm.sum())
    accuracy = float(tp.sum() / total) if total > 0 else np.nan
    macro_f1 = float(f1[macro_mask].mean()) if np.any(macro_mask) else np.nan
    balanced_accuracy = float(recall[macro_mask].mean()) if np.any(macro_mask) else np.nan

    per_class = pd.DataFrame(
        {
            "label": list(label_order),
            "support": support.astype(np.int64),
            "pred_support": pred_support.astype(np.int64),
            "precision": precision.astype(np.float64),
            "recall": recall.astype(np.float64),
            "f1": f1.astype(np.float64),
        }
    )
    return ClassificationSummary(
        metrics={
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_accuracy,
            "n_cells": float(total),
            "n_gt_present_classes": float(int(np.sum(macro_mask))),
        },
        confusion_matrix=cm,
        per_class=per_class,
    )


def predicted_labels_from_scores(class_scores: np.ndarray) -> np.ndarray:
    return np.argmax(class_scores, axis=1).astype(np.int32, copy=False)


# -----------------------------------------------------------------------------
# OvR AUROC, majority-class baseline, coarse aggregation
# -----------------------------------------------------------------------------


def per_class_auroc(y_true_ids: np.ndarray, proba: np.ndarray, *, n_classes: int) -> Dict[int, float]:
    """
    One-vs-rest AUROC per class {class id: AUROC}; NaN for a class without positives or negatives.
    Cells with y_true < 0 are ignored.
    """
    y_true_ids = np.asarray(y_true_ids).reshape(-1)
    proba = np.asarray(proba, dtype=np.float64)
    out: Dict[int, float] = {c: float("nan") for c in range(n_classes)}
    if proba.ndim != 2 or proba.shape[1] != n_classes:
        return out
    mask = y_true_ids >= 0
    yt = y_true_ids[mask]
    P = proba[mask]
    if yt.size < 2:
        return out
    for c in range(n_classes):
        y_bin = (yt == c).astype(np.int32)
        n_pos = int(y_bin.sum())
        n_neg = int(y_bin.size - n_pos)
        if n_pos == 0 or n_neg == 0:
            continue
        try:
            out[c] = float(roc_auc_score(y_bin, P[:, c]))
        except Exception:
            out[c] = float("nan")
    return out


def macro_auroc_ovr(
    y_true_ids: np.ndarray,
    proba: np.ndarray,
    *,
    n_classes: int,
    present_only: bool = True,
    min_present_classes: int = 2,
) -> float:
    """
    Macro one-vs-rest AUROC: mean of the defined per-class AUROCs (NaN if fewer than min_present_classes).
    present_only is unused (the average is always over the present classes).
    """
    per = per_class_auroc(y_true_ids, proba, n_classes=n_classes)
    finite = [v for v in per.values() if np.isfinite(v)]
    if len(finite) < int(min_present_classes):
        return float("nan")
    return float(np.mean(finite))


def majority_class_metrics(y_true_ids: np.ndarray, *, label_order: Sequence[str]) -> Dict[str, object]:
    """
    Metrics of the majority-class baseline (most frequent class of the pooled OOF ground truth predicted for
    every cell). Cells with y_true < 0 are ignored.
    """
    n_classes = len(label_order)
    out: Dict[str, object] = {
        "majority_class_index": -1,
        "majority_class_label": None,
        "majority_fraction": float("nan"),
        "accuracy": float("nan"),
        "macro_f1": float("nan"),
        "balanced_accuracy": float("nan"),
        "n_cells": 0.0,
    }
    y = np.asarray(y_true_ids).reshape(-1)
    y = y[y >= 0].astype(np.int64, copy=False)
    if y.size == 0:
        return out
    counts = np.bincount(y, minlength=n_classes)
    maj = int(np.argmax(counts))
    y_pred = np.full(y.shape, maj, dtype=np.int64)
    summ = classification_metrics_from_ids(y, y_pred, label_order, gt_present_only_for_macro=True)
    total = float(counts.sum())
    out.update(
        {
            "majority_class_index": maj,
            "majority_class_label": str(label_order[maj]),
            "majority_fraction": float(counts[maj] / total) if total > 0 else float("nan"),
            "accuracy": summ.metrics["accuracy"],
            "macro_f1": summ.metrics["macro_f1"],
            "balanced_accuracy": summ.metrics["balanced_accuracy"],
            "n_cells": summ.metrics["n_cells"],
        }
    )
    return out


def aggregate_proba_to_coarse(
    proba: np.ndarray,
    fine_label_space: object,
    coarse_label_space: object,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Sum the fine class probabilities [N, C_fine] into coarse classes [N, C_coarse] (fine -> coarse from
    fine_label_space.coarse_mapping). Returns (coarse_proba, fine -> coarse mapping).
    """
    proba = np.asarray(proba, dtype=np.float32)
    fine_labels = [str(x) for x in list(fine_label_space.labels)]
    coarse_labels = [str(x) for x in list(coarse_label_space.labels)]
    coarse_index = {str(k): int(v) for k, v in dict(coarse_label_space.label_to_index).items()}
    fine_to_coarse = getattr(fine_label_space, "coarse_mapping", None)
    if not fine_to_coarse:
        raise ValueError(
            "aggregate_proba_to_coarse requires fine_label_space.coarse_mapping "
            "(canonical fine label -> coarse label); got None/empty."
        )

    c_fine = len(fine_labels)
    c_coarse = len(coarse_labels)
    if proba.ndim != 2 or proba.shape[1] != c_fine:
        raise ValueError(
            f"proba has shape {proba.shape}; expected [N, {c_fine}] to match the fine label space."
        )

    col_map = np.full(c_fine, -1, dtype=np.int64)
    mapping: Dict[str, str] = {}
    for j, fl in enumerate(fine_labels):
        cl = fine_to_coarse.get(fl)
        if cl is None or str(cl) not in coarse_index:
            raise ValueError(
                f"Fine label {fl!r} has no coarse target in the coarse label space "
                f"(coarse_mapping value={cl!r})."
            )
        col_map[j] = coarse_index[str(cl)]
        mapping[fl] = str(cl)

    coarse_proba = np.zeros((proba.shape[0], c_coarse), dtype=np.float32)
    for j in range(c_fine):
        coarse_proba[:, int(col_map[j])] += proba[:, j]
    return coarse_proba, mapping


# -----------------------------------------------------------------------------
# The OOF metric entry.
# -----------------------------------------------------------------------------

# Per-slide columns (n_macro_classes / n_classes_below_macro_floor: classes in / out of the slide macro average)
_PER_SLIDE_COLUMNS = [
    "slide_id",
    "source",
    "subset",
    "tissue",
    "n_cells",
    "balanced_accuracy",
    "macro_f1",
    "accuracy",
    "macro_auroc",
    "n_macro_classes",
    "n_classes_below_macro_floor",
]
_FOLD_COVERAGE_COLUMNS = ["fold_name", "class", "in_train", "n_test_cells_true"]
# Per-(slide, class) F1 table: one row per class present in a slide
_PER_SLIDE_PER_CLASS_COLUMNS = [
    "slide_id",
    "source",
    "subset",
    "tissue",
    "n_cells",
    "class",
    "support",
    "f1",
]


def _label_granularity(cfg: object) -> str:
    """Read ``protocol.label_granularity`` from a Mapping or attribute-style cfg (default 'fine')."""
    prot: object
    if isinstance(cfg, Mapping):
        prot = cfg.get("protocol", cfg)
    else:
        prot = getattr(cfg, "protocol", cfg)
    if isinstance(prot, Mapping):
        return str(prot.get("label_granularity", "fine")).lower()
    return str(getattr(prot, "label_granularity", "fine")).lower()


def _auroc_min_present_classes(cfg: object) -> int:
    """protocol.metrics.auroc_min_present_classes (default 2, also for missing / invalid values)."""
    prot: object
    if isinstance(cfg, Mapping):
        prot = cfg.get("protocol", cfg)
    else:
        prot = getattr(cfg, "protocol", cfg)
    metrics_cfg = prot.get("metrics", {}) if isinstance(prot, Mapping) else getattr(prot, "metrics", {})
    try:
        val = int(dict(metrics_cfg).get("auroc_min_present_classes", 2))
    except (TypeError, ValueError):
        return 2
    return val if val >= 1 else 2


def _macro_min_class_support(cfg: object) -> int:
    """protocol.metrics.macro_min_class_support_per_slide (default 0 = off, also for missing / invalid values)."""
    prot: object
    if isinstance(cfg, Mapping):
        prot = cfg.get("protocol", cfg)
    else:
        prot = getattr(cfg, "protocol", cfg)
    metrics_cfg = prot.get("metrics", {}) if isinstance(prot, Mapping) else getattr(prot, "metrics", {})
    try:
        val = int(dict(metrics_cfg).get("macro_min_class_support_per_slide", 0))
    except (TypeError, ValueError):
        return 0
    return val if val >= 0 else 0


def _per_slide_classification_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    row_meta: pd.DataFrame,
    label_order: Sequence[str],
    *,
    covered: Optional[np.ndarray] = None,
    min_present_classes: int = 2,
    macro_min_support: int = 0,
) -> pd.DataFrame:
    """
    Per-slide classification metrics (optionally restricted to the covered cells). macro_min_support
    (metrics.macro_min_class_support_per_slide) applies to the per-slide macro averages only.
    """
    n_classes = len(label_order)
    if covered is not None:
        sel = np.flatnonzero(np.asarray(covered, dtype=bool))
        y_true = y_true[sel]
        y_pred = y_pred[sel]
        proba = proba[sel]
        row_meta = row_meta.iloc[sel].reset_index(drop=True)
    if len(y_true) == 0:
        return pd.DataFrame(columns=_PER_SLIDE_COLUMNS)

    tmp = row_meta.copy()
    tmp["_pos"] = np.arange(len(y_true), dtype=np.int64)
    rows = []
    for slide_id, g in tmp.groupby("slide_id", dropna=False, sort=True):
        ii = g["_pos"].to_numpy()
        yt = y_true[ii]
        yp = y_pred[ii]
        pr = proba[ii]
        summ = classification_metrics_from_ids(
            yt, yp, label_order,
            gt_present_only_for_macro=True,
            macro_min_support=macro_min_support,
        )
        # Number of classes present in the slide but below the macro support floor
        supports = summ.per_class["support"].to_numpy(dtype=np.int64)
        n_present = int((supports > 0).sum())
        n_macro = int(summ.metrics["n_gt_present_classes"])
        first = g.iloc[0]
        rows.append(
            {
                "slide_id": str(slide_id),
                "source": str(first["source"]),
                "subset": str(first["subset"]),
                "tissue": str(first["tissue"]),
                "n_cells": int(summ.metrics["n_cells"]),
                "balanced_accuracy": summ.metrics["balanced_accuracy"],
                "macro_f1": summ.metrics["macro_f1"],
                "accuracy": summ.metrics["accuracy"],
                "macro_auroc": macro_auroc_ovr(
                    yt, pr, n_classes=n_classes, min_present_classes=min_present_classes
                ),
                "n_macro_classes": n_macro,
                "n_classes_below_macro_floor": int(n_present - n_macro),
            }
        )
    return pd.DataFrame(rows, columns=_PER_SLIDE_COLUMNS)


def _per_slide_per_class_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    row_meta: pd.DataFrame,
    label_order: Sequence[str],
) -> pd.DataFrame:
    """Per-(slide, class) F1: one row per class present in the slide's ground truth (all cells)."""
    if len(y_true) == 0:
        return pd.DataFrame(columns=_PER_SLIDE_PER_CLASS_COLUMNS)
    tmp = row_meta.copy()
    tmp["_pos"] = np.arange(len(y_true), dtype=np.int64)
    rows = []
    for slide_id, g in tmp.groupby("slide_id", dropna=False, sort=True):
        ii = g["_pos"].to_numpy()
        yt = y_true[ii]
        yp = y_pred[ii]
        summ = classification_metrics_from_ids(yt, yp, label_order, gt_present_only_for_macro=True)
        first = g.iloc[0]
        n_cells = int(summ.metrics["n_cells"])
        present = summ.per_class.loc[summ.per_class["support"] > 0]
        for _, pc in present.iterrows():
            rows.append(
                {
                    "slide_id": str(slide_id),
                    "source": str(first["source"]),
                    "subset": str(first["subset"]),
                    "tissue": str(first["tissue"]),
                    "n_cells": n_cells,
                    "class": str(pc["label"]),
                    "support": int(pc["support"]),
                    "f1": float(pc["f1"]),
                }
            )
    return pd.DataFrame(rows, columns=_PER_SLIDE_PER_CLASS_COLUMNS)


def _fold_coverage_table(
    y_true: np.ndarray,
    covered: np.ndarray,
    row_fold: np.ndarray,
    label_order: Sequence[str],
) -> pd.DataFrame:
    """
    Per (fold, true class) coverage: in_train (class in the fold's train classes) and n_test_cells_true
    (OOF test cells of the class in the fold).
    """
    n = len(y_true)
    if n == 0:
        return pd.DataFrame(columns=_FOLD_COVERAGE_COLUMNS)
    df = pd.DataFrame(
        {
            "fold_name": np.asarray(row_fold, dtype=object).astype(str),
            "true_id": np.asarray(y_true, dtype=np.int64),
            "covered": np.asarray(covered, dtype=bool),
        }
    )
    df = df[df["true_id"] >= 0]
    if df.empty:
        return pd.DataFrame(columns=_FOLD_COVERAGE_COLUMNS)
    rows = []
    for (fn, tid), g in df.groupby(["fold_name", "true_id"], dropna=False, sort=True):
        rows.append(
            {
                "fold_name": str(fn),
                "class": str(label_order[int(tid)]),
                "in_train": bool(g["covered"].all()),
                "n_test_cells_true": int(len(g)),
            }
        )
    return pd.DataFrame(rows, columns=_FOLD_COVERAGE_COLUMNS)


def _classification_block(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    covered: np.ndarray,
    row_meta: pd.DataFrame,
    row_fold: np.ndarray,
    label_order: Sequence[str],
    min_present_classes: int = 2,
    macro_min_support: int = 0,
) -> Dict[str, object]:
    """
    One metric block (summary, per-slide, per-slide-per-class, per-class, confusion, fold coverage, restricted)
    from pooled OOF arrays aligned to label_order. Used for the fine and the coarse-by-aggregation metrics.
    """
    n_classes = len(label_order)
    cov = np.asarray(covered, dtype=bool)

    summ = classification_metrics_from_ids(y_true, y_pred, label_order, gt_present_only_for_macro=True)
    g_auroc = macro_auroc_ovr(
        y_true, proba, n_classes=n_classes, present_only=True, min_present_classes=min_present_classes
    )
    pc_auroc = per_class_auroc(y_true, proba, n_classes=n_classes)

    per_class_df = summ.per_class.copy()
    per_class_df["auroc"] = [pc_auroc.get(i, float("nan")) for i in range(n_classes)]

    # Weighted-F1 = support-weighted mean of the per-class F1 (as sklearn f1_score(average="weighted"))
    supp = per_class_df["support"].to_numpy(dtype=np.float64)
    f1v = per_class_df["f1"].to_numpy(dtype=np.float64)
    tot_supp = float(supp.sum())
    weighted_f1 = float(np.sum(supp * f1v) / tot_supp) if tot_supp > 0 else float("nan")

    if np.any(cov):
        summ_r = classification_metrics_from_ids(
            y_true[cov], y_pred[cov], label_order, gt_present_only_for_macro=True
        )
        bal_r = summ_r.metrics["balanced_accuracy"]
        mf1_r = summ_r.metrics["macro_f1"]
        n_r = summ_r.metrics["n_cells"]
    else:
        bal_r = float("nan")
        mf1_r = float("nan")
        n_r = 0.0

    maj = majority_class_metrics(y_true, label_order=label_order)

    benchmark_summary = {
        "balanced_accuracy": summ.metrics["balanced_accuracy"],
        "macro_f1": summ.metrics["macro_f1"],
        "macro_auroc": g_auroc,
        "accuracy": summ.metrics["accuracy"],
        "weighted_f1": weighted_f1,
        "n_cells": summ.metrics["n_cells"],
        "n_classes": float(n_classes),
        "n_gt_present_classes": summ.metrics["n_gt_present_classes"],
        "balanced_accuracy_restricted": bal_r,
        "macro_f1_restricted": mf1_r,
        "n_cells_restricted": float(n_r),
        "covered_fraction": float(np.mean(cov)) if cov.size else float("nan"),
        # Majority-class baseline (numeric values only)
        "majority_balanced_accuracy": maj["balanced_accuracy"],
        "majority_macro_f1": maj["macro_f1"],
        "majority_accuracy": maj["accuracy"],
        "majority_fraction": maj["majority_fraction"],
        "majority_class_index": float(maj["majority_class_index"]),
    }

    # The macro support floor applies to these two tables only
    per_slide = _per_slide_classification_table(
        y_true, y_pred, proba, row_meta, label_order, covered=None,
        min_present_classes=min_present_classes,
        macro_min_support=macro_min_support,
    )
    per_slide_restricted = _per_slide_classification_table(
        y_true, y_pred, proba, row_meta, label_order, covered=cov,
        min_present_classes=min_present_classes,
        macro_min_support=macro_min_support,
    )
    per_slide_per_class = _per_slide_per_class_table(y_true, y_pred, row_meta, label_order)
    fold_coverage = _fold_coverage_table(y_true, cov, row_fold, label_order)

    return {
        "benchmark_summary": benchmark_summary,
        "per_slide": per_slide,
        "per_slide_per_class": per_slide_per_class,
        "per_class": per_class_df,
        "confusion": {"labels": list(label_order), "matrix": summ.confusion_matrix},
        "fold_coverage": fold_coverage,
        "per_slide_restricted": per_slide_restricted,
    }


def compute_classification_metrics(
    preds: Mapping[str, Mapping],
    cfg: object,
    *,
    label_space: object,
    coarse_label_space: Optional[object] = None,
) -> Dict[str, object]:
    """
    Pooled out-of-fold classification metrics for Task A.3.

    preds: {slide_id -> bundle} (y_true, y_proba on the global class axis, true_class_covered, slide metadata).
    Returns {"benchmark_summary", "per_slide", "per_slide_per_class", "per_class", "confusion", "fold_coverage",
    "per_slide_restricted", "coarse"}. _restricted: covered cells only (true class in the fold's train set);
    coarse: coarse-by-aggregation metrics (fine eval with a coarse_label_space only, else None).
    """
    label_order = [str(x) for x in list(label_space.labels)]
    n_classes = len(label_order)
    min_present = _auroc_min_present_classes(cfg)
    macro_min_support = _macro_min_class_support(cfg)

    yt_list, yp_list, proba_list, cov_list = [], [], [], []
    slide_list, source_list, subset_list, tissue_list, fold_list = [], [], [], [], []

    for slide_id, b in preds.items():
        yt = np.asarray(b["y_true"], dtype=np.int64).reshape(-1)
        proba = np.asarray(b["y_proba"], dtype=np.float64)
        if proba.ndim != 2 or proba.shape[1] != n_classes:
            raise ValueError(
                f"Bundle '{slide_id}' y_proba has shape {proba.shape}; expected [N, {n_classes}] "
                "to match the global class vocabulary."
            )
        if proba.shape[0] != yt.shape[0]:
            raise ValueError(
                f"Bundle '{slide_id}' y_true/y_proba row mismatch: {yt.shape[0]} vs {proba.shape[0]}."
            )
        if not np.all(np.isfinite(proba)):
            raise ValueError(f"Bundle '{slide_id}' y_proba contains non-finite values.")
        if yt.size and (int(yt.min()) < 0 or int(yt.max()) >= n_classes):
            raise ValueError(
                f"Bundle '{slide_id}' y_true has ids outside [0, {n_classes}); "
                f"min={int(yt.min())}, max={int(yt.max())}."
            )
        cov = np.asarray(b["true_class_covered"], dtype=bool).reshape(-1)
        if cov.shape[0] != yt.shape[0]:
            raise ValueError(
                f"Bundle '{slide_id}' true_class_covered length {cov.shape[0]} != n_cells {yt.shape[0]}."
            )
        yp = predicted_labels_from_scores(proba).astype(np.int64)
        n = yt.shape[0]

        yt_list.append(yt)
        yp_list.append(yp)
        proba_list.append(proba)
        cov_list.append(cov)
        slide_list.append(np.full(n, str(slide_id), dtype=object))
        source_list.append(np.full(n, str(b.get("source", "")), dtype=object))
        subset_list.append(np.full(n, str(b.get("subset", "")), dtype=object))
        tissue_list.append(np.full(n, str(b.get("tissue", "")), dtype=object))
        # Per-row fold names (ADAPTED: 3 rotations merged per slide), else the scalar fold_name
        fold_names_b = b.get("fold_names")
        if fold_names_b is not None:
            fr = np.asarray(fold_names_b, dtype=object).reshape(-1)
            if fr.shape[0] != n:
                raise ValueError(
                    f"Bundle '{slide_id}' fold_names length {fr.shape[0]} != n_cells {n}."
                )
            fold_list.append(fr)
        else:
            fold_list.append(np.full(n, str(b.get("fold_name", "")), dtype=object))

    if yt_list:
        y_true = np.concatenate(yt_list)
        y_pred = np.concatenate(yp_list)
        proba = np.concatenate(proba_list, axis=0)
        covered = np.concatenate(cov_list)
        row_fold = np.concatenate(fold_list)
        row_meta = pd.DataFrame(
            {
                "slide_id": np.concatenate(slide_list),
                "source": np.concatenate(source_list),
                "subset": np.concatenate(subset_list),
                "tissue": np.concatenate(tissue_list),
            }
        )
    else:
        y_true = np.zeros(0, dtype=np.int64)
        y_pred = np.zeros(0, dtype=np.int64)
        proba = np.zeros((0, n_classes), dtype=np.float64)
        covered = np.zeros(0, dtype=bool)
        row_fold = np.zeros(0, dtype=object)
        row_meta = pd.DataFrame(columns=["slide_id", "source", "subset", "tissue"])

    out = _classification_block(
        y_true=y_true,
        y_pred=y_pred,
        proba=proba,
        covered=covered,
        row_meta=row_meta,
        row_fold=row_fold,
        label_order=label_order,
        min_present_classes=min_present,
        macro_min_support=macro_min_support,
    )

    # Coarse-by-aggregation metrics (fine eval only)
    out["coarse"] = None
    granularity = _label_granularity(cfg)
    if granularity == "fine" and coarse_label_space is not None and y_true.shape[0] > 0:
        coarse_mapping = getattr(label_space, "coarse_mapping", None)
        if not coarse_mapping:
            logger.warning(
                "label_granularity=fine but the fine label space has no coarse_mapping; "
                "skipping the coarse-by-aggregation mirror."
            )
        else:
            coarse_labels = [str(x) for x in list(coarse_label_space.labels)]
            coarse_index = {str(k): int(v) for k, v in dict(coarse_label_space.label_to_index).items()}
            fine_to_coarse_id = np.full(n_classes, -1, dtype=np.int64)
            for i, fl in enumerate(label_order):
                cl = coarse_mapping.get(fl)
                if cl is not None and str(cl) in coarse_index:
                    fine_to_coarse_id[i] = coarse_index[str(cl)]
            if np.any(fine_to_coarse_id < 0):
                missing = [label_order[i] for i in range(n_classes) if fine_to_coarse_id[i] < 0]
                raise ValueError(f"Fine labels have no coarse mapping target: {missing}")
            coarse_proba, _ = aggregate_proba_to_coarse(proba, label_space, coarse_label_space)
            coarse_true = fine_to_coarse_id[y_true]
            coarse_pred = predicted_labels_from_scores(coarse_proba).astype(np.int64)
            # Coarse coverage per fold: a cell is covered if a fine class of its coarse class was
            # covered in its fold
            covered_coarse_by_fold: Dict[str, set] = {}
            for fn_, fid_, cov_ in zip(row_fold.tolist(), y_true.tolist(), covered.tolist()):
                if cov_:
                    covered_coarse_by_fold.setdefault(str(fn_), set()).add(
                        int(fine_to_coarse_id[int(fid_)])
                    )
            coarse_covered = np.array(
                [
                    int(ct_) in covered_coarse_by_fold.get(str(fn_), set())
                    for ct_, fn_ in zip(coarse_true.tolist(), row_fold.tolist())
                ],
                dtype=bool,
            )
            out["coarse"] = _classification_block(
                y_true=coarse_true,
                y_pred=coarse_pred,
                proba=coarse_proba,
                covered=coarse_covered,
                row_meta=row_meta,
                row_fold=row_fold,
                label_order=coarse_labels,
                min_present_classes=min_present,
                macro_min_support=macro_min_support,
            )

    return out


def compute_primary_validation_metric(
    preds: Mapping[str, Mapping],
    cfg: object,
    label_space: object,
) -> float:
    """Pooled macro-F1 over the OOF bundle dict (cfg is unused)."""
    del cfg  # unused
    label_order = [str(x) for x in list(label_space.labels)]
    yt_list, yp_list = [], []
    for b in preds.values():
        proba = np.asarray(b["y_proba"], dtype=np.float64)
        yt_list.append(np.asarray(b["y_true"], dtype=np.int64).reshape(-1))
        yp_list.append(predicted_labels_from_scores(proba).astype(np.int64))
    if not yt_list:
        return float("nan")
    y_true = np.concatenate(yt_list)
    y_pred = np.concatenate(yp_list)
    summ = classification_metrics_from_ids(y_true, y_pred, label_order, gt_present_only_for_macro=True)
    return float(summ.metrics["macro_f1"])
