"""Fit the Task A.1 ridge probe on one fold (alpha chosen by leave-one-train-group-out CV) and predict the test slides."""

import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .common import mkdir, save_json
from .metrics import compute_slide_macro_spearman_score
from .modeling import (
    accumulate_slide_moments,
    alpha_grid_from_config,
    apply_feature_geometry,
    assemble_gram,
    assemble_rhs,
    boundary_saturation,
    effective_dof,
    fold_feature_center,
    numerical_rank,
    resolve_feature_geometry,
    ridge_solve_all_alphas,
    safe_eigh,
    SlideMoments,
)

logger = logging.getLogger(__name__)


# Tolerance floor of the 1-SE rule (held-out scores closer than this are tied)
_TIE_ATOL = 1.0e-9


@dataclass
class SlideTensorData:
    slide_id: str
    subset: str
    source: str
    tissue: str
    X: np.ndarray           # [N, D]
    Y: np.ndarray           # [N, G], depth-normalized + per-(slide,gene) z-scored
                            # + winsorized; NaN where not measured
    row_keys: np.ndarray
    coords_xy: np.ndarray
    detected: Optional[np.ndarray] = None   # [N, G] bool, True where the raw count > 0
    cell_types: Optional[np.ndarray] = None
    obs_df: Optional[pd.DataFrame] = None
    train_gene_support: Optional[pd.DataFrame] = None

    def measured_gene_indices(self) -> np.ndarray:
        return np.flatnonzero(np.any(~np.isnan(self.Y), axis=0)).astype(np.int64)


@dataclass
class FoldData:
    fold_name: str
    train_slides: List[SlideTensorData]
    val_slides: List[SlideTensorData]
    test_slides: List[SlideTensorData]
    train_gene_names: np.ndarray
    test_gene_names: np.ndarray
    # Unit left out by alpha selection: "slide", or "stripe" for STHELAR_TEST_ADAPTED
    alpha_group: str = "slide"


@dataclass(frozen=True)
class EligibilityPattern:
    """One set of train slides, and the targets supervised on exactly that set."""

    pattern_id: int
    slide_idx: Tuple[int, ...]   # positions into fold_data.train_slides
    gene_idx: np.ndarray         # columns of the train-target axis


# -----------------------------------------------------------------------------
# Validity structure
# -----------------------------------------------------------------------------


def rank1_validity(Y: np.ndarray, slide_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Row and target validity (row_ok, gene_ok) of a slide's targets; raises if validity is not their outer product."""
    F = np.isfinite(np.asarray(Y))
    row_ok = F.any(axis=1)
    gene_ok = F.any(axis=0)
    if not np.array_equal(F, np.outer(row_ok, gene_ok)):
        bad = np.argwhere(F != np.outer(row_ok, gene_ok))[:10].tolist()
        raise RuntimeError(
            f"Slide '{slide_id}': target validity is not a rank-1 outer product of "
            f"(row_ok, gene_ok) (each target column must be fully valid or fully NaN). "
            f"first_mismatched_indices={bad}"
        )
    return row_ok, gene_ok


def eligibility_patterns(
    train_slides: Sequence[SlideTensorData],
    target_names: Sequence[str],
) -> List[EligibilityPattern]:
    """Group targets by the set of train slides that supervise them (targets of a group share one Gram)."""
    if not train_slides:
        raise RuntimeError("eligibility_patterns requires at least one train slide.")

    E = np.stack(
        [rank1_validity(s.Y, s.slide_id)[1] for s in train_slides], axis=0
    )  # [S, G]

    support = E.sum(axis=0)
    if np.any(support == 0):
        dead = [str(target_names[g]) for g in np.flatnonzero(support == 0)][:10]
        raise RuntimeError(
            f"{int(np.sum(support == 0))} train target(s) are eligible on zero train "
            f"slides and cannot be fit: {dead}. Inspect train_slide_gene_eligibility.csv and "
            f"protocol.gene_selection.min_train_slides_per_gene."
        )

    groups: Dict[bytes, List[int]] = {}
    for g in range(E.shape[1]):
        groups.setdefault(E[:, g].tobytes(), []).append(g)

    # sorted() so pattern_id is reproducible run to run.
    return [
        EligibilityPattern(
            pattern_id=i,
            slide_idx=tuple(np.flatnonzero(np.frombuffer(key, dtype=bool)).tolist()),
            gene_idx=np.asarray(genes, dtype=np.int64),
        )
        for i, (key, genes) in enumerate(sorted(groups.items()))
    ]


# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------


def _prediction_bundle(
    slide: SlideTensorData,
    target_names: Sequence[str],
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    # Predictions are NaN where the target is NaN
    out = np.asarray(y_pred, dtype=np.float32)
    out[np.isnan(slide.Y)] = np.nan
    bad = np.isfinite(slide.Y) & ~np.isfinite(out)
    if np.any(bad):
        idx = np.argwhere(bad)[:10].tolist()
        raise RuntimeError(
            f"Non-finite predictions for slide '{slide.slide_id}' where the target is "
            f"finite. first_bad_indices={idx}"
        )
    return {
        "slide_id": slide.slide_id,
        "source": slide.source,
        "subset": slide.subset,
        "tissue": slide.tissue,
        "row_keys": slide.row_keys,
        "coords_xy": slide.coords_xy,
        "gene_names": np.asarray(target_names, dtype=object),
        "y_true": slide.Y.astype(np.float32, copy=False),
        "y_pred": out,
        "detected": slide.detected,
        "cell_types": slide.cell_types,
        "obs_df": slide.obs_df.copy() if slide.obs_df is not None else None,
        "train_gene_support": slide.train_gene_support,
    }


def predict_slides(
    slides: Sequence[SlideTensorData],
    target_names: Sequence[str],
    W: np.ndarray,
    b: np.ndarray,
    geometry: str,
) -> Dict[str, Dict[str, Any]]:
    """Predict every slide with one dense matmul: Y_hat = geom(X) @ W + b."""
    preds: Dict[str, Dict[str, Any]] = {}
    for slide in slides:
        Xg = apply_feature_geometry(slide.X, geometry)
        preds[slide.slide_id] = _prediction_bundle(
            slide, target_names, Xg @ W + b[None, :]
        )
    return preds


# -----------------------------------------------------------------------------
# Alpha selection
# -----------------------------------------------------------------------------


def _build_moments(
    slides: Sequence[SlideTensorData],
    geometry: str,
) -> Tuple[List[SlideMoments], np.ndarray, List[np.ndarray]]:
    """Accumulate per-slide moments once. This is the only pass over X / Y."""
    geoms = [apply_feature_geometry(s.X, geometry) for s in slides]
    valid = [rank1_validity(s.Y, s.slide_id) for s in slides]
    mu0 = fold_feature_center(
        [Xg[row_ok].mean(axis=0) for Xg, (row_ok, _) in zip(geoms, valid)]
    )
    moments = [
        accumulate_slide_moments(
            slide_id=s.slide_id, X=Xg, Y=s.Y, row_ok=row_ok, gene_ok=gene_ok, mu0=mu0
        )
        for s, Xg, (row_ok, gene_ok) in zip(slides, geoms, valid)
    ]
    return moments, mu0, geoms


def _tau_for(moments: Sequence[SlideMoments], slide_idx: Sequence[int]) -> float:
    """trace(A)/D over a slide set -- the mean eigenvalue, used to scale alpha."""
    _, _, tau = assemble_gram(moments, slide_idx)
    return tau


def select_alpha(
    *,
    cfg: Mapping,
    slides: Sequence[SlideTensorData],
    moments: Sequence[SlideMoments],
    geoms: Sequence[np.ndarray],
    mu0: np.ndarray,
    patterns: Sequence[EligibilityPattern],
    target_names: Sequence[str],
    group_label: str,
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """
    Choose lambda by leave-one-train-group-out CV (group = slide, or stripe for the adapted protocol) on the
    slide-macro Spearman, with alpha = lambda * trace(A_fit)/D and the paired 1-SE rule (select_lambda_1se).
    Returns (lam_star, alpha_curve, per_group_scores).
    """
    lam_grid = alpha_grid_from_config(cfg)
    n_groups = len(slides)
    if n_groups < 2:
        raise RuntimeError(
            f"Alpha selection needs >= 2 train {group_label}s to leave one out, got "
            f"{n_groups}. Inspect the fold's split payload."
        )

    n_targets = len(target_names)
    scores = np.full((lam_grid.size, n_groups), np.nan, dtype=np.float64)
    rows: List[Dict[str, Any]] = []

    for si in range(n_groups):
        fit_idx = [j for j in range(n_groups) if j != si]
        tau = _tau_for(moments, fit_idx)

        # Targets fittable without slide si, keyed by the slide set left over.
        work: Dict[Tuple[int, ...], List[int]] = {}
        for p in patterns:
            if si not in p.slide_idx:
                continue  # si supervises nothing here -> nothing to score on it
            rest = tuple(x for x in p.slide_idx if x != si)
            if not rest:
                continue  # supported by si alone -> unfittable without it
            work.setdefault(rest, []).extend(p.gene_idx.tolist())

        if not work:
            logger.warning(
                "Held-out %s '%s' scores no target (every target it supervises is "
                "supported by it alone); it contributes no alpha signal.",
                group_label,
                slides[si].slide_id,
            )
            continue

        Xc = geoms[si] - mu0[None, :]
        yhat = [
            np.full((Xc.shape[0], n_targets), np.nan, dtype=np.float32)
            for _ in range(lam_grid.size)
        ]

        for slide_set, genes in work.items():
            gidx = np.asarray(sorted(genes), dtype=np.int64)
            A, xbar, _ = assemble_gram(moments, slide_set)
            evals, evecs = safe_eigh(A)
            B, ybar = assemble_rhs(moments, slide_set, gidx, xbar)
            Bt = evecs.T @ B
            Z = (Xc - xbar[None, :]) @ evecs   # one matmul, shared by every lambda
            for li, lam in enumerate(lam_grid):
                pred = Z @ (Bt / (evals + float(lam) * tau)[:, None]) + ybar[None, :]
                yhat[li][:, gidx] = pred.astype(np.float32, copy=False)
            del A, evals, evecs, Z, Bt

        # Mask the truth to the targets fittable without si
        scorable = np.zeros(n_targets, dtype=bool)
        scorable[np.asarray([g for gs in work.values() for g in gs], dtype=np.int64)] = True
        y_true = np.where(scorable[None, :], slides[si].Y, np.nan).astype(np.float32)

        for li in range(lam_grid.size):
            bundle = {
                slides[si].slide_id: _prediction_bundle(
                    _with_masked_truth(slides[si], y_true), target_names, yhat[li]
                )
            }
            score = compute_slide_macro_spearman_score(bundle, cfg)
            scores[li, si] = score["loso_slide_macro_spearman"]
        del yhat

    if not np.any(np.isfinite(scores)):
        raise RuntimeError(
            "Alpha selection produced no finite score on any lambda. The held-out "
            f"{group_label}s yielded no scorable target; inspect the fold's gene "
            "eligibility."
        )

    lam_star, mean_deficit, se_deficit = select_lambda_1se(scores, lam_grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_scores = np.nanmean(scores, axis=1)

    for li, lam in enumerate(lam_grid):
        rows.append(
            {
                "lambda": float(lam),
                "loso_slide_macro_spearman": float(mean_scores[li]),
                # Mean and SE of the per-group deficit vs the best lambda (used by the 1-SE rule)
                "mean_paired_deficit_vs_best": float(mean_deficit[li]),
                "se_paired_deficit_vs_best": float(se_deficit[li]),
                "within_1se_of_best": bool(
                    np.isfinite(mean_deficit[li])
                    and mean_deficit[li] >= -(se_deficit[li] if np.isfinite(se_deficit[li]) else 0.0)
                ),
                "n_groups_scored": int(np.sum(np.isfinite(scores[li]))),
                "selected": bool(float(lam) == lam_star),
            }
        )
    per_group = pd.DataFrame(
        scores,
        index=pd.Index([float(x) for x in lam_grid], name="lambda"),
        columns=[f"group__{slides[si].slide_id}__{si}" for si in range(n_groups)],
    ).reset_index()
    return lam_star, pd.DataFrame(rows), per_group


def _with_masked_truth(slide: SlideTensorData, y_true: np.ndarray) -> SlideTensorData:
    from dataclasses import replace

    return replace(slide, Y=y_true)


def select_lambda_1se(
    scores: np.ndarray,
    lam_grid: Sequence[float],
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    1-SE rule on the paired per-group deficit d_g = score_g(lam) - score_g(lam_best): selects the largest lambda
    whose mean deficit is within one SE (floored at _TIE_ATOL) of zero. scores is [n_lambda, n_group].
    Returns (lam_star, mean_deficit, se_deficit), the deficit arrays aligned to lam_grid.
    """
    S = np.asarray(scores, dtype=np.float64)
    grid = np.asarray([float(x) for x in lam_grid], dtype=np.float64)
    if S.ndim != 2 or S.shape[0] != grid.size:
        raise ValueError(
            f"scores must be [n_lambda, n_group] aligned to lam_grid; got {S.shape} "
            f"for {grid.size} lambdas."
        )

    with warnings.catch_warnings():
        # All-NaN rows (lambda scored on no group) give NaN
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_scores = np.nanmean(S, axis=1)
    if not np.any(np.isfinite(mean_scores)):
        raise RuntimeError("select_lambda_1se: no lambda has a finite mean score.")

    best_i = int(np.nanargmax(mean_scores))
    # Paired deficit against the best lambda, on the groups they share.
    deficit = S - S[best_i][None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_deficit = np.nanmean(deficit, axis=1)
        n_paired = np.sum(np.isfinite(deficit), axis=1)
        # sample standard deviation (ddof=1)
        sd = np.nanstd(deficit, axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        se_deficit = np.where(n_paired >= 2, sd / np.sqrt(np.maximum(n_paired, 1)), np.nan)

    # Admissible: mean deficit >= -max(SE, _TIE_ATOL) (the best lambda is always admissible)
    tol = np.maximum(np.where(np.isfinite(se_deficit), se_deficit, 0.0), _TIE_ATOL)
    admissible = np.isfinite(mean_deficit) & (mean_deficit >= -tol)
    lam_star = float(grid[int(np.max(np.flatnonzero(admissible)))])
    return lam_star, mean_deficit, se_deficit


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def fit_predict_one_fold(
    *,
    cfg: Mapping,
    fold_data: FoldData,
    space: str,
    fold_index: int,
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Fit the ridge probe on one fold and predict every test slide.
    Returns (artifacts, val_preds, test_preds): fit summary, {} (no validation split), {slide_id: prediction bundle}.
    """
    t0 = time.perf_counter()
    mkdir(output_dir)

    if not fold_data.train_slides:
        raise RuntimeError(f"Fold '{fold_data.fold_name}' has no train slides.")
    if not fold_data.test_slides:
        raise RuntimeError(f"Fold '{fold_data.fold_name}' has no test slides.")

    geometry = resolve_feature_geometry(cfg, space)
    target_names = np.asarray(fold_data.train_gene_names, dtype=object)
    n_targets = int(target_names.size)

    moments, mu0, geoms = _build_moments(fold_data.train_slides, geometry)
    patterns = eligibility_patterns(fold_data.train_slides, target_names)
    all_idx = list(range(len(fold_data.train_slides)))
    tau_full = _tau_for(moments, all_idx)

    group_label = str(fold_data.alpha_group)
    lam_star, alpha_curve, alpha_per_group = select_alpha(
        cfg=cfg,
        slides=fold_data.train_slides,
        moments=moments,
        geoms=geoms,
        mu0=mu0,
        patterns=patterns,
        target_names=target_names,
        group_label=group_label,
    )
    alpha_star = lam_star * tau_full

    lam_grid = alpha_grid_from_config(cfg)
    # Spectrum of the full-fold Gram
    _A_full, _, _ = assemble_gram(moments, all_idx)
    evals_full, _ = safe_eigh(_A_full)
    del _A_full
    at_boundary, boundary_kind = boundary_saturation(
        evals=evals_full, alpha=alpha_star, lam=lam_star, lam_grid=lam_grid,
    )
    if boundary_kind == "truncated":
        raise RuntimeError(
            f"Fold '{fold_data.fold_name}': the selected lambda ({lam_star:g}) is at an edge of "
            f"the grid [{lam_grid[0]:g}, {lam_grid[-1]:g}] without saturated capacity (effective_dof="
            f"{effective_dof(evals_full, alpha_star):.1f} of numerical_rank="
            f"{numerical_rank(evals_full)}). Widen protocol.ridge.alpha_grid and re-run."
        )
    if boundary_kind == "ols":
        logger.warning(
            "Fold %s: selected the smallest lambda of the grid (%g), where the probe is "
            "already OLS (effective_dof=%.1f == numerical_rank=%d). Recorded as "
            "lambda_at_grid_boundary=True.",
            fold_data.fold_name, lam_star, effective_dof(evals_full, alpha_star),
            numerical_rank(evals_full),
        )
    elif boundary_kind == "zero_capacity":
        logger.warning(
            "Fold %s: selected the largest lambda of the grid (%g), where the probe has "
            "almost no capacity (effective_dof=%.3f): no signal generalizes across held-out "
            "%ss (intercept-only predictor). Recorded as lambda_at_grid_boundary=True.",
            fold_data.fold_name, lam_star, effective_dof(evals_full, alpha_star),
            group_label,
        )

    # Final fit: one dense W over every pattern, so prediction is a single matmul.
    D = int(geoms[0].shape[1])
    W = np.zeros((D, n_targets), dtype=np.float64)
    b = np.zeros((n_targets,), dtype=np.float64)
    pattern_rows: List[Dict[str, Any]] = []

    for p in patterns:
        A, xbar, _ = assemble_gram(moments, p.slide_idx)
        evals, evecs = safe_eigh(A)
        B, ybar = assemble_rhs(moments, p.slide_idx, p.gene_idx, xbar)
        Wp, bp = ridge_solve_all_alphas(
            evals=evals, evecs=evecs, B=B, ybar=ybar, xbar=xbar, mu0=mu0,
            alphas=[alpha_star],
        )
        W[:, p.gene_idx] = Wp[0]
        b[p.gene_idx] = bp[0]
        pattern_rows.append(
            {
                "pattern_id": p.pattern_id,
                "n_slides": len(p.slide_idx),
                "slide_ids": "|".join(
                    fold_data.train_slides[j].slide_id for j in p.slide_idx
                ),
                "n_targets": int(p.gene_idx.size),
                "n_rows": int(sum(moments[j].n_rows for j in p.slide_idx)),
                "effective_dof": effective_dof(evals, alpha_star),
                "numerical_rank": numerical_rank(evals),
                "eig_max": float(evals.max()),
            }
        )
        del A, evals, evecs

    pattern_df = pd.DataFrame(pattern_rows)
    # Effective dof: mean over patterns weighted by their number of targets
    dof_w = np.average(
        pattern_df["effective_dof"].to_numpy(dtype=np.float64),
        weights=pattern_df["n_targets"].to_numpy(dtype=np.float64),
    )

    test_preds = predict_slides(fold_data.test_slides, target_names, W, b, geometry)

    artifacts: Dict[str, Any] = {
        "fold_name": str(fold_data.fold_name),
        "mode": "closed_form_ridge",
        "space": str(space),
        "feature_geometry": geometry,
        "input_dim": D,
        "n_targets": n_targets,
        "n_patterns": len(patterns),
        "n_train_slides": len(fold_data.train_slides),
        "n_train_rows": int(sum(m.n_rows for m in moments)),
        "alpha_group": group_label,
        "selected_lambda": float(lam_star),
        "selected_alpha": float(alpha_star),
        "tau_trace_over_d": float(tau_full),
        # "interior", or a saturated edge ("ols" / "zero_capacity"); "truncated" raises above
        "lambda_at_grid_boundary": bool(at_boundary),
        "lambda_boundary_kind": str(boundary_kind),
        "numerical_rank": int(numerical_rank(evals_full)),
        "effective_dof": float(dof_w),
        "effective_dof_pattern_min": float(pattern_df["effective_dof"].min()),
        "effective_dof_pattern_median": float(pattern_df["effective_dof"].median()),
        "effective_dof_pattern_max": float(pattern_df["effective_dof"].max()),
        # Best held-out score on the grid (optimistically biased, diagnostic only)
        "loso_best_score": float(alpha_curve["loso_slide_macro_spearman"].max()),
        "loso_score_at_selected": float(
            alpha_curve.loc[alpha_curve["selected"], "loso_slide_macro_spearman"].iloc[0]
        ),
        "alpha_selection_rule": "paired_1se",
        "fit_seconds": float(time.perf_counter() - t0),
    }

    alpha_curve.to_csv(output_dir / "alpha_selection.csv", index=False)
    # [n_lambda, n_group] score matrix used by the 1-SE rule
    alpha_per_group.to_csv(output_dir / "alpha_selection_per_group.csv", index=False)
    pattern_df.to_csv(output_dir / "ridge_patterns.csv", index=False)
    save_json(
        output_dir / "model_manifest.json",
        {
            **artifacts,
            "sample_weights": str(cfg["protocol"]["ridge"]["sample_weights"]),
            "fit_intercept": bool(cfg["protocol"]["ridge"]["fit_intercept"]),
            "dtype": "float64",
            "lambda_grid": [float(x) for x in lam_grid],
        },
    )
    logger.info(
        "Fold %s | lambda=%.4g alpha=%.4g effective_dof=%.1f/%d patterns=%d | %.1fs",
        fold_data.fold_name, lam_star, alpha_star, dof_w, D, len(patterns),
        artifacts["fit_seconds"],
    )
    return artifacts, {}, test_preds
