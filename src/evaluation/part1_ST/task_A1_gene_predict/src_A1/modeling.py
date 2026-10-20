"""Weighted ridge probe of Task A.1, solved in closed form from per-slide moments."""

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Eigenvalues below eig_max * D * eps are numerically indistinguishable from zero.
_RANK_EPS = float(np.finfo(np.float64).eps)


# -----------------------------------------------------------------------------
# Feature geometry
# -----------------------------------------------------------------------------


def _normalize_eval_space(space: str) -> str:
    space = str(space).strip().lower()
    if space not in {"he", "he_l2", "clip", "coords"}:
        raise ValueError(
            f"Unsupported encoder/evaluation space={space!r}. "
            "Expected one of {'he', 'he_l2', 'clip', 'coords'}."
        )
    return space


def _validate_feature_geometry(geometry: str) -> str:
    geometry = str(geometry).strip().lower()
    if geometry not in {"identity", "row_l2"}:
        raise ValueError(
            f"Unsupported feature geometry: {geometry!r}. Supported: {'identity', 'row_l2'}."
        )
    return geometry


def resolve_feature_geometry(config: Mapping, space: str) -> str:
    """Feature geometry of a space: row_l2 for clip and he_l2, identity otherwise."""
    space = _normalize_eval_space(space)
    return "row_l2" if space in {"clip", "he_l2"} else "identity"


def apply_feature_geometry(X: np.ndarray, geometry: str) -> np.ndarray:
    geometry = _validate_feature_geometry(geometry)
    Xf = np.asarray(X, dtype=np.float64)
    if geometry == "identity":
        return Xf
    norms = np.linalg.norm(Xf, axis=1, keepdims=True)
    # Zero rows stay at the origin (no division by zero)
    return Xf / np.maximum(norms, 1.0e-12)


# -----------------------------------------------------------------------------
# Per-slide moments
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SlideMoments:
    """Moments of one slide's kept rows (zero-depth rows dropped), in the fold-centered frame Xc = X - mu0."""

    slide_id: str
    n_rows: int          # kept rows; the w_i = 1/n_rows denominator
    M: np.ndarray        # [D,D]  (1/n) Xc^T Xc
    m: np.ndarray        # [D]    mean(Xc, axis=0)
    C: np.ndarray        # [D,G]  (1/n) Xc^T Y0        (Y0 = Y with NaN -> 0)
    ybar: np.ndarray     # [G]    mean(Y0, axis=0)
    gene_ok: np.ndarray  # [G]    bool; columns where this slide supervises


def fold_feature_center(feature_means: Sequence[np.ndarray]) -> np.ndarray:
    """Slide-uniform mean of the fold's slide feature means (mu0), used to center the features."""
    if len(feature_means) == 0:
        raise ValueError("fold_feature_center requires at least one slide.")
    stacked = np.stack([np.asarray(m, dtype=np.float64) for m in feature_means], axis=0)
    return stacked.mean(axis=0)


def accumulate_slide_moments(
    *,
    slide_id: str,
    X: np.ndarray,
    Y: np.ndarray,
    row_ok: np.ndarray,
    gene_ok: np.ndarray,
    mu0: np.ndarray,
) -> SlideMoments:
    """Accumulate one slide's moments from its kept rows."""
    Xk = np.asarray(X, dtype=np.float64)[np.asarray(row_ok, dtype=bool)]
    Yk = np.asarray(Y, dtype=np.float64)[np.asarray(row_ok, dtype=bool)]
    n = int(Xk.shape[0])
    if n == 0:
        raise RuntimeError(
            f"Slide '{slide_id}' has zero usable rows after dropping zero-depth units."
        )

    Xc = Xk - np.asarray(mu0, dtype=np.float64)[None, :]
    # Columns not supervised by this slide (NaN) are set to 0
    Y0 = np.where(np.isfinite(Yk), Yk, 0.0)

    return SlideMoments(
        slide_id=str(slide_id),
        n_rows=n,
        M=(Xc.T @ Xc) / float(n),
        m=Xc.mean(axis=0),
        C=(Xc.T @ Y0) / float(n),
        ybar=Y0.mean(axis=0),
        gene_ok=np.asarray(gene_ok, dtype=bool),
    )


# -----------------------------------------------------------------------------
# Slide-set systems
# -----------------------------------------------------------------------------


def assemble_gram(
    moments: Sequence[SlideMoments],
    slide_idx: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Weighted-centered Gram of a slide set T: A = sum_s M_s - |T| * xbar xbar^T, with xbar = (1/|T|) sum_s m_s.
    Returns (A, xbar, trace(A)/D).
    """
    idx = list(slide_idx)
    if not idx:
        raise ValueError("assemble_gram requires a non-empty slide set.")
    d = int(moments[idx[0]].M.shape[0])

    gram = np.zeros((d, d), dtype=np.float64)
    xsum = np.zeros((d,), dtype=np.float64)
    # Fixed summation order (bit-reproducible)
    for j in idx:
        gram += moments[j].M
        xsum += moments[j].m

    n_set = float(len(idx))
    xbar = xsum / n_set
    A = gram - n_set * np.outer(xbar, xbar)
    A = 0.5 * (A + A.T)  # accumulated fp error breaks exact symmetry
    return A, xbar, float(np.trace(A)) / float(d)


def assemble_rhs(
    moments: Sequence[SlideMoments],
    slide_idx: Sequence[int],
    gene_idx: np.ndarray,
    xbar: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """The weighted-centered RHS B [D, n_genes] and target means ybar [n_genes]."""
    idx = list(slide_idx)
    if not idx:
        raise ValueError("assemble_rhs requires a non-empty slide set.")
    g = np.asarray(gene_idx, dtype=np.int64)
    d = int(moments[idx[0]].C.shape[0])

    csum = np.zeros((d, g.size), dtype=np.float64)
    ysum = np.zeros((g.size,), dtype=np.float64)
    for j in idx:
        csum += moments[j].C[:, g]
        ysum += moments[j].ybar[g]

    n_set = float(len(idx))
    ybar = ysum / n_set
    B = csum - n_set * np.outer(xbar, ybar)
    return B, ybar


def safe_eigh(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition (evals, evecs), with evals floored at 0."""
    Af = np.asarray(A, dtype=np.float64)
    if not np.isfinite(Af).all():
        raise RuntimeError("Non-finite entries in the Gram matrix; cannot eigendecompose.")
    evals, evecs = np.linalg.eigh(0.5 * (Af + Af.T))
    evals = np.maximum(evals, 0.0)
    if not float(evals.sum()) > 0.0:
        raise RuntimeError(
            "Degenerate Gram matrix (all eigenvalues zero) -- the features are constant "
            "over this slide set. Inspect the encoder embeddings."
        )
    return evals, evecs


def numerical_rank(evals: np.ndarray) -> int:
    e = np.asarray(evals, dtype=np.float64)
    if e.size == 0:
        return 0
    return int(np.sum(e > float(e.max()) * float(e.size) * _RANK_EPS))


def effective_dof(evals: np.ndarray, alpha: float) -> float:
    """Effective degrees of freedom of the ridge, df(alpha) = sum_j e_j / (e_j + alpha)."""
    e = np.asarray(evals, dtype=np.float64)
    a = float(alpha)
    if not a > 0.0:
        raise ValueError(f"alpha must be positive, got {a!r}.")
    return float(np.sum(e / (e + a)))


# -----------------------------------------------------------------------------
# Solve
# -----------------------------------------------------------------------------


def ridge_solve_all_alphas(
    *,
    evals: np.ndarray,
    evecs: np.ndarray,
    B: np.ndarray,
    ybar: np.ndarray,
    xbar: np.ndarray,
    mu0: np.ndarray,
    alphas: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the weighted ridge for every alpha from one eigendecomposition (unpenalized intercept):
    w(alpha) = V diag(1/(e + alpha)) V^T B, b(alpha) = ybar - (mu0 + xbar) . w(alpha).
    Returns (W, b) with shapes [n_alpha, D, n_genes] and [n_alpha, n_genes].
    """
    e = np.asarray(evals, dtype=np.float64)
    V = np.asarray(evecs, dtype=np.float64)
    Bt = V.T @ np.asarray(B, dtype=np.float64)          # [D, n_genes]
    shift = np.asarray(mu0, dtype=np.float64) + np.asarray(xbar, dtype=np.float64)

    W = np.empty((len(alphas), V.shape[0], Bt.shape[1]), dtype=np.float64)
    b = np.empty((len(alphas), Bt.shape[1]), dtype=np.float64)
    for k, alpha in enumerate(alphas):
        a = float(alpha)
        if not a > 0.0:
            raise ValueError(f"alpha must be positive, got {a!r}.")
        W[k] = V @ (Bt / (e + a)[:, None])
        b[k] = np.asarray(ybar, dtype=np.float64) - shift @ W[k]
    return W, b


# An edge is saturated if the capacity left beyond it is below this fraction of the Gram rank
_SATURATION_CAPACITY_FRAC = 0.01


def boundary_saturation(
    *,
    evals: np.ndarray,
    alpha: float,
    lam: float,
    lam_grid: Sequence[float],
) -> Tuple[bool, str]:
    """
    Classify the selected lambda: (is_edge, kind), kind in {"interior", "ols", "zero_capacity", "truncated"}.
    An edge is "ols" / "zero_capacity" if going further along the grid cannot change the fit, else "truncated".
    """
    e = np.asarray(evals, dtype=np.float64)
    grid = [float(x) for x in lam_grid]
    lo, hi = grid[0], grid[-1]
    at_lo, at_hi = float(lam) == lo, float(lam) == hi
    if not (at_lo or at_hi):
        return False, "interior"

    dof = effective_dof(e, alpha)
    rank = float(numerical_rank(e))
    tol = _SATURATION_CAPACITY_FRAC * rank
    if at_lo and (rank - dof) <= tol:      # nothing left to gain by shrinking lambda
        return True, "ols"
    if at_hi and dof <= tol:               # nothing left to shed by raising lambda
        return True, "zero_capacity"
    return True, "truncated"


def alpha_grid_from_config(config: Mapping) -> np.ndarray:
    """The lambda grid; alpha = lambda * trace(A_fit)/D is applied by the caller."""
    cfg = config["protocol"]["ridge"]["alpha_grid"]
    lo = float(cfg["lambda_min"])
    hi = float(cfg["lambda_max"])
    n = int(cfg["n_points"])
    if not (0.0 < lo < hi):
        raise ValueError(f"Require 0 < lambda_min < lambda_max, got {lo!r} and {hi!r}.")
    if n < 2:
        raise ValueError(f"protocol.ridge.alpha_grid.n_points must be >= 2, got {n!r}.")
    return np.logspace(np.log10(lo), np.log10(hi), n)
