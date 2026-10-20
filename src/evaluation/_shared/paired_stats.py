"""Paired statistics (bootstrap CI, Wilcoxon signed-rank test, Benjamini-Hochberg FDR)."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Significance threshold: q <= SIG_ALPHA
SIG_ALPHA = 0.05

# Fewest non-zero paired differences for the Wilcoxon test (below: notes="insufficient_n", p = NaN)
MIN_N_FOR_WILCOXON = 6

# Fewest paired units for the bootstrap CI (below: CI = NaN)
DEFAULT_MIN_N_FOR_CI = 2


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PairedResult:
    """Outcome of one paired candidate-vs-baseline comparison.

    Fields:
        delta: mean of the direction-corrected per-unit deltas (positive = candidate better)
        ci_low, ci_high: percentile bootstrap CI on delta (NaN when n < 2)
        p_value: two-sided Wilcoxon signed-rank p-value (NaN when undefined)
        q_value: BH-adjusted p-value within the caller's family (NaN until apply_fdr(), or if p is NaN)
        fdr_applied: True once apply_fdr() has set q_value
        n: number of paired units after key intersection and NaN drop
        n_zero: number of pairs with a zero difference
        statistic: scipy Wilcoxon statistic (min(T+, T-) for the two-sided test)
        notes: "ok" | "insufficient_n" | "all_zero" | "all_nan" | "no_overlap" | "no_baseline" |
            "no_candidate", plus "; dropped=K" when pairs were dropped during key intersection
    """
    delta: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    p_value: float = float("nan")
    q_value: float = float("nan")
    fdr_applied: bool = False
    n: int = 0
    n_zero: int = 0
    statistic: float = float("nan")
    notes: str = "ok"


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _coerce_1d_float(arr) -> np.ndarray:
    if arr is None:
        return np.empty(0, dtype=float)
    a = np.asarray(arr, dtype=float).reshape(-1)
    return a


def _reject_duplicate_keys(keys: Sequence, name: str) -> None:
    """Raise if `keys` names the same paired unit more than once."""
    seen = set()
    dupes = []
    for k in keys:
        if k in seen:
            if k not in dupes:
                dupes.append(k)
        else:
            seen.add(k)
    if dupes:
        raise ValueError(
            f"_intersect_paired: {name} names {len(dupes)} unit(s) more than once "
            f"(first: {dupes[:3]!r}). Collapse to one row per unit before pairing "
            "(e.g. group-by mean on the key column)."
        )


def _intersect_paired(
    values_a,
    values_b,
    keys_a: Optional[Sequence] = None,
    keys_b: Optional[Sequence] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Align two arrays on shared keys (if given) or by position, and drop pairs with a NaN.

    Returns (a, b, n_dropped). Raises on unequal lengths without keys, and on duplicate keys.
    """
    a = _coerce_1d_float(values_a)
    b = _coerce_1d_float(values_b)
    if keys_a is None or keys_b is None:
        if a.size != b.size:
            raise ValueError(
                "_intersect_paired: positional pairing requires equal lengths, got "
                f"{a.size} and {b.size}. Pass keys_a and keys_b to pair on unit identity."
            )
        n_total_pairs = a.size
    else:
        ka = list(keys_a)
        kb = list(keys_b)
        if len(ka) != a.size or len(kb) != b.size:
            return np.empty(0, dtype=float), np.empty(0, dtype=float), max(a.size, b.size)
        _reject_duplicate_keys(ka, "keys_a")
        _reject_duplicate_keys(kb, "keys_b")
        idx_b = {k: i for i, k in enumerate(kb)}
        common_idx_a = [i for i, k in enumerate(ka) if k in idx_b]
        a_aligned = a[common_idx_a]
        b_aligned = np.fromiter((b[idx_b[ka[i]]] for i in common_idx_a), dtype=float, count=len(common_idx_a))
        n_total_pairs = max(len(ka), len(kb))
        a, b = a_aligned, b_aligned
    finite_mask = np.isfinite(a) & np.isfinite(b)
    n_dropped = int(n_total_pairs - int(finite_mask.sum()))
    return a[finite_mask], b[finite_mask], n_dropped


_WILCOXON_METHOD_KWARGS: Optional[dict] = None


def _wilcoxon_method_kwargs(fn) -> dict:
    """{"method": "auto"} if the installed scipy wilcoxon accepts it, else {} (memoized)."""
    global _WILCOXON_METHOD_KWARGS
    if _WILCOXON_METHOD_KWARGS is None:
        try:
            takes_method = "method" in inspect.signature(fn).parameters
        except (TypeError, ValueError):   # not introspectable
            takes_method = False
        _WILCOXON_METHOD_KWARGS = {"method": "auto"} if takes_method else {}
    return _WILCOXON_METHOD_KWARGS


def _signed_deltas(a: np.ndarray, b: np.ndarray, direction: str) -> np.ndarray:
    """Direction-corrected per-unit deltas: positive => candidate (a) is better."""
    if direction == "lower":
        return b - a
    return a - b


# -----------------------------------------------------------------------------
# Public primitives
# -----------------------------------------------------------------------------

def paired_bootstrap_delta_ci(
    deltas,
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    statistic: str = "mean",
    min_n_for_ci: int = DEFAULT_MIN_N_FOR_CI,
) -> Tuple[float, float, float, int]:
    """Percentile bootstrap CI on the mean (or median) of direction-corrected per-unit deltas.

    Returns (point, ci_low, ci_high, n); the CI is NaN below min_n_for_ci units.
    """
    d = _coerce_1d_float(deltas)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n < max(2, int(min_n_for_ci)):
        return (float(np.nanmean(d)) if n else float("nan"), float("nan"), float("nan"), n)
    if statistic not in ("mean", "median"):
        raise ValueError(f"Unsupported statistic: {statistic}")
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    idx = rng.integers(0, n, size=(int(n_resamples), n))
    samples = d[idx]
    boot = samples.mean(axis=1) if statistic == "mean" else np.median(samples, axis=1)
    alpha = (1.0 - float(level)) / 2.0
    lo, hi = np.quantile(boot, [alpha, 1.0 - alpha])
    point = float(d.mean()) if statistic == "mean" else float(np.median(d))
    return point, float(lo), float(hi), n


def wilcoxon_signed_rank(
    values_a,
    values_b,
    *,
    zero_method: str = "wilcox",
    alternative: str = "two-sided",
    min_n_for_wilcoxon: int = MIN_N_FOR_WILCOXON,
) -> Tuple[float, float, int, int, str]:
    """Two-sided Wilcoxon signed-rank test on aligned paired arrays (scipy, method="auto").

    Returns (statistic, p_value, n_pairs_used, n_zero, notes); never raises (NaN values with a note).
    """
    a = _coerce_1d_float(values_a)
    b = _coerce_1d_float(values_b)
    if a.size != b.size:
        return float("nan"), float("nan"), 0, 0, "size_mismatch"
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    n = int(a.size)
    if n == 0:
        return float("nan"), float("nan"), 0, 0, "all_nan"
    diff = a - b
    n_zero = int(np.sum(diff == 0))
    n_nonzero = n - n_zero
    if n_zero == n:
        return float("nan"), float("nan"), n, n_zero, "all_zero"
    if n_nonzero < int(min_n_for_wilcoxon):
        return float("nan"), float("nan"), n, n_zero, "insufficient_n"
    try:
        from scipy.stats import wilcoxon as _wilcoxon  # type: ignore
    except Exception as exc:
        logger.warning("scipy.stats.wilcoxon unavailable (%s); returning NaN", exc)
        return float("nan"), float("nan"), n, n_zero, "scipy_missing"
    try:
        res = _wilcoxon(
            a, b, zero_method=zero_method, alternative=alternative,
            **_wilcoxon_method_kwargs(_wilcoxon),
        )
        stat = float(getattr(res, "statistic", float("nan")))
        pval = float(getattr(res, "pvalue", float("nan")))
    except Exception as exc:
        logger.debug("wilcoxon failed (%s)", exc)
        return float("nan"), float("nan"), n, n_zero, "wilcoxon_error"
    return stat, pval, n, n_zero, "ok"


def benjamini_hochberg(
    p_values,
    *,
    alpha: float = SIG_ALPHA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Step-up Benjamini-Hochberg FDR. Returns (q_values, reject_mask).

    NaN p-values are excluded from the family and give NaN q-values. Uses
    scipy.stats.false_discovery_control (scipy >= 1.11) if available, else a numpy implementation.
    """
    p = _coerce_1d_float(p_values)
    n_total = p.size
    finite_mask = np.isfinite(p)
    q_out = np.full(n_total, np.nan, dtype=float)
    reject_out = np.zeros(n_total, dtype=bool)
    if not finite_mask.any():
        return q_out, reject_out
    p_finite = p[finite_mask]
    m = p_finite.size
    try:
        from scipy.stats import false_discovery_control  # type: ignore
        q_finite = np.asarray(false_discovery_control(p_finite, method="bh"), dtype=float)
    except Exception:
        order = np.argsort(p_finite, kind="mergesort")
        ranked = p_finite[order]
        ranks = np.arange(1, m + 1, dtype=float)
        raw_q = ranked * m / ranks
        # Step-up: enforce monotonic non-increasing from the largest p downward.
        cum_min = np.minimum.accumulate(raw_q[::-1])[::-1]
        q_sorted = np.minimum(cum_min, 1.0)
        q_finite = np.empty_like(q_sorted)
        q_finite[order] = q_sorted
    q_out[finite_mask] = q_finite
    reject_out[finite_mask] = q_finite <= float(alpha)
    return q_out, reject_out


# -----------------------------------------------------------------------------
# One-stop paired test + FDR application
# -----------------------------------------------------------------------------

def paired_test(
    values_a,
    values_b,
    metric: str,
    *,
    direction: str,
    keys_a: Optional[Sequence] = None,
    keys_b: Optional[Sequence] = None,
    n_bootstrap: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    min_n_for_wilcoxon: int = MIN_N_FOR_WILCOXON,
    min_n_for_ci: int = DEFAULT_MIN_N_FOR_CI,
) -> PairedResult:
    """Paired bootstrap CI + Wilcoxon test of candidate (a) vs baseline (b).

    Returns a PairedResult with q_value = NaN (set by apply_fdr). `metric` is not used.
    """
    if direction not in ("higher", "lower"):
        return PairedResult(notes=f"bad_direction:{direction}")
    a, b, n_dropped = _intersect_paired(values_a, values_b, keys_a=keys_a, keys_b=keys_b)
    extra = f"; dropped={n_dropped}" if n_dropped > 0 else ""
    if a.size == 0:
        return PairedResult(notes=("no_overlap" if (keys_a is not None and keys_b is not None) else "all_nan") + extra)
    deltas = _signed_deltas(a, b, direction)
    point, ci_low, ci_high, n_units = paired_bootstrap_delta_ci(
        deltas, n_resamples=n_bootstrap, level=level, seed=seed, statistic="mean",
        min_n_for_ci=min_n_for_ci,
    )
    stat, pval, n_pairs, n_zero, w_notes = wilcoxon_signed_rank(
        a, b, min_n_for_wilcoxon=min_n_for_wilcoxon,
    )
    note = w_notes if w_notes != "ok" else "ok"
    # Record a suppressed CI in notes
    if n_units and not np.isfinite(ci_low) and n_units < max(2, int(min_n_for_ci)):
        ci_note = f"insufficient_n_for_ci(n={n_units}<{int(min_n_for_ci)})"
        note = ci_note if note == "ok" else f"{note}; {ci_note}"
    return PairedResult(
        delta=float(point),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(pval),
        q_value=float("nan"),
        n=int(n_units),
        n_zero=int(n_zero),
        statistic=float(stat),
        notes=(note + extra) if extra else note,
    )


def apply_fdr(
    results: Sequence[PairedResult],
    *,
    alpha: float = SIG_ALPHA,
) -> List[PairedResult]:
    """Return a copy of the results with q_value set by BH within this family (fdr_applied=True).

    NaN p-values keep q = NaN. A single-element family gives q == p.
    """
    if not results:
        return []
    pvals = np.array([r.p_value for r in results], dtype=float)
    qvals, _ = benjamini_hochberg(pvals, alpha=alpha)
    out: List[PairedResult] = []
    for r, q in zip(results, qvals):
        out.append(replace(r, q_value=float(q), fdr_applied=True))
    return out
