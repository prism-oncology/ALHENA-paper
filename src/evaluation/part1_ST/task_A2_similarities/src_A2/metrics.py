from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src_A2.common import entropy_from_probs


# macro_min_support (metrics.macro_min_class_support_per_slide) is applied to the three per-slide macros
# (headline, (k, tau) tuning objective, chance floor), never to the subset-pooled metrics.


def macro_min_class_support(cfg: Mapping[str, object]) -> int:
    """Read metrics.macro_min_class_support_per_slide (0 = off)."""
    try:
        val = int(dict(cfg.get("metrics", {}) or {}).get("macro_min_class_support_per_slide", 0))
    except (TypeError, ValueError):
        return 0
    return val if val >= 0 else 0


class UnscorableLabelLeakError(RuntimeError):
    """Raised when a classification-unscorable label appears as a truth or as an answer."""


def assert_unscorable_labels_absent(
    per_class: pd.DataFrame,
    unscorable_labels: Sequence[str],
    *,
    context: str,
) -> None:
    """Check that every unscorable label has support == 0 and pred_support == 0 in per_class (absent labels are skipped)."""
    wanted = {str(x) for x in (unscorable_labels or [])}
    if not wanted or per_class is None or per_class.empty:
        return
    if "label" not in per_class.columns:
        raise KeyError(
            f"assert_unscorable_labels_absent requires a 'label' column; got "
            f"{list(per_class.columns)} ({context})."
        )

    hit = per_class.loc[per_class["label"].astype(str).isin(wanted)]
    offenders = [
        (str(r["label"]), int(r.get("support", 0)), int(r.get("pred_support", 0)))
        for _, r in hit.iterrows()
        if int(r.get("support", 0)) != 0 or int(r.get("pred_support", 0)) != 0
    ]
    if not offenders:
        return

    detail = "; ".join(
        f"{lab!r}: support={sup:,}, pred_support={pred:,}" for lab, sup, pred in offenders
    )
    raise UnscorableLabelLeakError(
        f"[{context}] classification-unscorable label(s) leaked into the scored population: "
        f"{detail}.\n"
        "Every label of data.gt_mapped_unscorable_labels must have support == 0 and pred_support == 0.\n"
        "  support != 0      -> the GT-side scorable mask (gt_mapped_label_scorable) did not remove these cells.\n"
        "  pred_support != 0 -> the reference retrieval pool still holds rows with this label. Check that _2 "
        "applied references.drop_unscorable_reference_rows and that _3 read reference_audits/row_filters/{slide}.npz."
    )


@dataclass
class ClassificationSummary:
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    per_class: pd.DataFrame


@dataclass
class ExpressionSummary:
    metrics: Dict[str, float]
    per_gene: Optional[pd.DataFrame]
    per_cell: Optional[pd.DataFrame]




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
    # macro_min_support: minimum class support to enter the macro average (0 = any class with support > 0).
    # It only changes the macro denominator; use it for per-slide macros, not pooled summaries.
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



def majority_class_metrics(
    y_true_ids: np.ndarray,
    *,
    label_order: Sequence[str],
    macro_min_support: int = 0,
) -> Dict[str, object]:
    """
    Metrics of the majority-class baseline: the most frequent GT class (ids >= 0) predicted for every cell.
    Returns accuracy, macro-F1, balanced accuracy, n_gt_present_classes, and the majority class with its frequency.
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
        "n_gt_present_classes": 0.0,
    }
    y = np.asarray(y_true_ids).reshape(-1)
    y = y[y >= 0].astype(np.int64, copy=False)
    if y.size == 0:
        return out
    counts = np.bincount(y, minlength=n_classes)
    maj = int(np.argmax(counts))
    y_pred = np.full(y.shape, maj, dtype=np.int64)
    summ = classification_metrics_from_ids(
        y, y_pred, label_order, gt_present_only_for_macro=True, macro_min_support=macro_min_support
    )
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
            "n_gt_present_classes": summ.metrics["n_gt_present_classes"],
        }
    )
    return out


def class_scores_from_neighbors(
    neighbor_indices: np.ndarray,
    weights: np.ndarray,
    ref_label_ids: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    n, k = neighbor_indices.shape
    scores = np.zeros((n, n_classes), dtype=np.float32)
    label_ids = ref_label_ids[neighbor_indices].astype(np.int64, copy=False)
    flat_labels = label_ids.reshape(-1)
    flat_weights = weights.astype(np.float32, copy=False).reshape(-1)
    row_ids = np.repeat(np.arange(n, dtype=np.int64), k)
    ok = flat_labels >= 0
    np.add.at(scores, (row_ids[ok], flat_labels[ok]), flat_weights[ok])
    return scores



def predicted_labels_from_scores(class_scores: np.ndarray) -> np.ndarray:
    return np.argmax(class_scores, axis=1).astype(np.int32, copy=False)



def entropy_metrics_from_class_scores(class_scores: np.ndarray) -> np.ndarray:
    return entropy_from_probs(class_scores, axis=1).astype(np.float32)



def weighted_expression_prediction(
    neighbor_indices: np.ndarray,
    weights: np.ndarray,
    ref_expr: np.ndarray,
    query_batch_size: int = 512,
    gene_block_size: int = 512,
) -> np.ndarray:
    """Weighted average of the neighbors' reference expression (batched over queries and gene blocks)."""
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    ref_expr = np.asarray(ref_expr, dtype=np.float32)

    if neighbor_indices.ndim != 2 or weights.ndim != 2:
        raise ValueError("neighbor_indices and weights must both be 2D")
    if neighbor_indices.shape != weights.shape:
        raise ValueError(f"neighbor_indices and weights shapes differ: {neighbor_indices.shape} vs {weights.shape}")
    if neighbor_indices.shape[0] == 0:
        return np.empty((0, ref_expr.shape[1]), dtype=np.float32)

    out = np.empty((neighbor_indices.shape[0], ref_expr.shape[1]), dtype=np.float32)
    bs = max(1, int(query_batch_size))
    gb = max(1, int(gene_block_size))
    G = int(ref_expr.shape[1])
    for start in range(0, neighbor_indices.shape[0], bs):
        stop = min(start + bs, neighbor_indices.shape[0])
        idx = neighbor_indices[start:stop]
        w = weights[start:stop]
        for g0 in range(0, G, gb):
            g1 = min(g0 + gb, G)
            gathered = ref_expr[idx, g0:g1]  # [b, k, gb]
            out[start:stop, g0:g1] = np.einsum("bk,bkg->bg", w, gathered, optimize=True).astype(np.float32, copy=False)
    return out



def weighted_expression_prediction_h5(
    neighbor_indices: np.ndarray,
    weights: np.ndarray,
    cache_h5_path: str,
    query_batch_size: int = 512,
    gene_block_size: int = 512,
) -> np.ndarray:
    """
    Memory-safe weighted expression prediction from an on-disk HDF5 cache
    containing dataset 'ref_expr_log1p' with shape [Nref_subset, G].
    """
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)

    if neighbor_indices.ndim != 2 or weights.ndim != 2:
        raise ValueError("neighbor_indices and weights must both be 2D")
    if neighbor_indices.shape != weights.shape:
        raise ValueError(f"neighbor_indices and weights shapes differ: {neighbor_indices.shape} vs {weights.shape}")
    if neighbor_indices.shape[0] == 0:
        with h5py.File(cache_h5_path, "r") as f:
            G = int(f["ref_expr_log1p"].shape[1])
        return np.empty((0, G), dtype=np.float32)

    bs = max(1, int(query_batch_size))
    gb = max(1, int(gene_block_size))

    with h5py.File(cache_h5_path, "r") as f:
        dset = f["ref_expr_log1p"]
        G = int(dset.shape[1])
        out = np.empty((neighbor_indices.shape[0], G), dtype=np.float32)

        for start in range(0, neighbor_indices.shape[0], bs):
            stop = min(start + bs, neighbor_indices.shape[0])
            idx = neighbor_indices[start:stop]
            w = weights[start:stop]

            uniq, inv = np.unique(idx.reshape(-1), return_inverse=True)
            inv = inv.reshape(idx.shape)

            for g0 in range(0, G, gb):
                g1 = min(g0 + gb, G)
                ref_block = np.asarray(dset[uniq.tolist(), g0:g1], dtype=np.float32)  # [U, gb]
                gathered = ref_block[inv]  # [b, k, gb]
                out[start:stop, g0:g1] = np.einsum("bk,bkg->bg", w, gathered, optimize=True).astype(np.float32, copy=False)

    return out



def _paired_rowwise_pearson(a: np.ndarray, b: np.ndarray, variance_eps: float) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    am = a.mean(axis=1, keepdims=True)
    bm = b.mean(axis=1, keepdims=True)
    ac = a - am
    bc = b - bm
    av = np.mean(ac * ac, axis=1)
    bv = np.mean(bc * bc, axis=1)
    valid = (av >= variance_eps) & (bv >= variance_eps)
    corr = np.full(a.shape[0], np.nan, dtype=np.float32)
    denom = np.sqrt(av[valid] * bv[valid])
    denom = np.maximum(denom, np.finfo(np.float32).tiny)
    corr[valid] = np.mean(ac[valid] * bc[valid], axis=1) / denom
    return corr, valid



def _paired_rowwise_spearman(a: np.ndarray, b: np.ndarray, variance_eps: float) -> Tuple[np.ndarray, np.ndarray]:
    try:
        ra = rankdata(a, axis=1).astype(np.float32, copy=False)
        rb = rankdata(b, axis=1).astype(np.float32, copy=False)
    except TypeError:
        ra = np.apply_along_axis(rankdata, 1, a).astype(np.float32, copy=False)
        rb = np.apply_along_axis(rankdata, 1, b).astype(np.float32, copy=False)
    return _paired_rowwise_pearson(ra, rb, variance_eps=variance_eps)



def summarize_expression(
    pred: np.ndarray,
    gt: np.ndarray,
    gene_names: Sequence[str],
    variance_eps: float,
    min_valid_cells_for_cellwise_corr: int,
    min_valid_cells_for_genewise_corr: int,
    min_valid_genes_for_slide_summary: int,
    gene_report_names: Optional[Sequence[str]] = None,
    return_details: bool = False,
) -> ExpressionSummary:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt shapes differ: {pred.shape} vs {gt.shape}")
    if len(gene_names) != pred.shape[1]:
        raise ValueError(
            f"gene_names length mismatch: {len(gene_names)} vs n_genes={pred.shape[1]}"
        )
    if gene_report_names is not None and len(gene_report_names) != pred.shape[1]:
        raise ValueError(
            f"gene_report_names length mismatch: {len(gene_report_names)} vs n_genes={pred.shape[1]}"
        )
    if pred.shape[0] == 0:
        return ExpressionSummary(metrics={"n_cells": 0.0, "n_genes": float(pred.shape[1])}, per_gene=None, per_cell=None)

    mse_per_cell = np.mean((pred - gt) ** 2, axis=1)
    mae_per_cell = np.mean(np.abs(pred - gt), axis=1)
    pearson_cell, valid_p_cell = _paired_rowwise_pearson(pred, gt, variance_eps=variance_eps)
    spearman_cell, valid_s_cell = _paired_rowwise_spearman(pred, gt, variance_eps=variance_eps)

    pred_t = pred.T
    gt_t = gt.T
    pearson_gene, valid_p_gene = _paired_rowwise_pearson(pred_t, gt_t, variance_eps=variance_eps)
    spearman_gene, valid_s_gene = _paired_rowwise_spearman(pred_t, gt_t, variance_eps=variance_eps)

    metrics: Dict[str, float] = {
        "n_cells": float(pred.shape[0]),
        "n_genes": float(pred.shape[1]),
        "mean_mse": float(np.mean(mse_per_cell)),
        "mean_mae": float(np.mean(mae_per_cell)),
        "n_valid_cellwise_pearson": float(np.sum(valid_p_cell)),
        "n_valid_cellwise_spearman": float(np.sum(valid_s_cell)),
        "n_valid_genewise_pearson": float(np.sum(valid_p_gene)),
        "n_valid_genewise_spearman": float(np.sum(valid_s_gene)),
    }

    metrics["mean_cellwise_pearson"] = (
        float(np.nanmean(pearson_cell[valid_p_cell])) if np.sum(valid_p_cell) >= min_valid_cells_for_cellwise_corr else np.nan
    )
    metrics["mean_cellwise_spearman"] = (
        float(np.nanmean(spearman_cell[valid_s_cell])) if np.sum(valid_s_cell) >= min_valid_cells_for_cellwise_corr else np.nan
    )
    enough_cells_for_genewise = pred.shape[0] >= int(min_valid_cells_for_genewise_corr)
    metrics["gene_wise_pearson"] = (
        float(np.nanmean(pearson_gene[valid_p_gene]))
        if enough_cells_for_genewise and np.sum(valid_p_gene) >= min_valid_genes_for_slide_summary
        else np.nan
    )
    metrics["gene_wise_spearman"] = (
        float(np.nanmean(spearman_gene[valid_s_gene]))
        if enough_cells_for_genewise and np.sum(valid_s_gene) >= min_valid_genes_for_slide_summary
        else np.nan
    )

    per_cell = None
    per_gene = None
    if return_details:
        per_cell = pd.DataFrame(
            {
                "row_index": np.arange(pred.shape[0], dtype=np.int64),
                "mse": mse_per_cell.astype(np.float32),
                "mae": mae_per_cell.astype(np.float32),
                "pearson": pearson_cell.astype(np.float32),
                "spearman": spearman_cell.astype(np.float32),
                "valid_pearson": valid_p_cell.astype(np.int8),
                "valid_spearman": valid_s_cell.astype(np.int8),
            }
        )
        if gene_report_names is None:
            per_gene = pd.DataFrame(
                {
                    "gene": list(gene_names),
                    "pearson": pearson_gene.astype(np.float32),
                    "spearman": spearman_gene.astype(np.float32),
                    "valid_pearson": valid_p_gene.astype(np.int8),
                    "valid_spearman": valid_s_gene.astype(np.int8),
                }
            )
        else:
            per_gene = pd.DataFrame(
                {
                    "gene": list(gene_names),
                    "gene_name": list(gene_report_names),
                    "pearson": pearson_gene.astype(np.float32),
                    "spearman": spearman_gene.astype(np.float32),
                    "valid_pearson": valid_p_gene.astype(np.int8),
                    "valid_spearman": valid_s_gene.astype(np.int8),
                }
            )

    return ExpressionSummary(metrics=metrics, per_gene=per_gene, per_cell=per_cell)



def agreement_rate(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return np.nan
    return float(np.mean(a == b))
