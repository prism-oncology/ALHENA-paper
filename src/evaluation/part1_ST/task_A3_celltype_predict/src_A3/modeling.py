"""Feature pipeline and sklearn logistic-regression probe for Task A.3."""

from __future__ import annotations

import inspect
import logging
import warnings
from typing import TYPE_CHECKING, List, Mapping, Optional, Sequence, Tuple

# sys.path setup (as in every src_A3 module)
from src_A3.common import ensure_imports

ensure_imports()

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src_A3.labels import LabelSpace

logger = logging.getLogger(__name__)

# STD_EPS: min train-fold std of a kept feature column; ROW_L2_EPS: min row norm for row-L2
STD_EPS: float = 1e-12
ROW_L2_EPS: float = 1e-12

_VALID_GEOMETRY_STRATEGIES = ("identity", "row_l2")

# protocol.probe.multi_class is passed only if the installed sklearn accepts it (removed in sklearn 1.7)
_LOGREG_INIT_PARAMS = frozenset(inspect.signature(LogisticRegression).parameters)


def resolve_geometry_preprocessing(space: str) -> str:
    """Row-wise geometry of a space: 'identity' for he, coords and scconcept; 'row_l2' for he_l2 and clip."""
    space_norm = str(space).strip().lower()
    if space_norm in {"he", "coords", "scconcept"}:
        return "identity"
    if space_norm in {"he_l2", "clip"}:
        return "row_l2"
    raise ValueError(
        f"Unsupported evaluation space={space!r}. "
        "Expected one of {'he', 'he_l2', 'clip', 'coords', 'scconcept'}."
    )


class GeometryTransformer(BaseEstimator, TransformerMixin):
    """Row-wise geometry transformer: identity or row_l2 (X / clip(||X||_2, ROW_L2_EPS))."""

    def __init__(self, strategy: str = "identity") -> None:
        # sklearn convention: store params verbatim (no validation/conversion in __init__).
        self.strategy = strategy

    def _resolved_strategy(self) -> str:
        strat = str(self.strategy).strip().lower()
        if strat not in _VALID_GEOMETRY_STRATEGIES:
            raise ValueError(
                f"Unsupported geometry strategy {self.strategy!r}; "
                f"expected one of {list(_VALID_GEOMETRY_STRATEGIES)}."
            )
        return strat

    def fit(self, X, y=None) -> "GeometryTransformer":
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"GeometryTransformer expects a 2-D matrix; got shape {X.shape}.")
        self._resolved_strategy()  # validate the strategy
        self.n_features_in_ = int(X.shape[1])
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"GeometryTransformer expects a 2-D matrix; got shape {X.shape}.")
        if X.shape[1] != self.n_features_in_:
            raise RuntimeError(
                f"Feature dimension mismatch: got {X.shape[1]}, expected {self.n_features_in_}."
            )
        strategy = self._resolved_strategy()
        if strategy == "identity":
            out = np.asarray(X, dtype=np.float32)
        else:  # row_l2
            Xf = np.asarray(X, dtype=np.float32)
            # Per-row squared norm without materializing X**2 (cell-level matrices are large).
            sq = np.einsum("ij,ij->i", Xf, Xf)
            norms = np.sqrt(sq).reshape(-1, 1)
            out = np.asarray(Xf / np.clip(norms, ROW_L2_EPS, None), dtype=np.float32)
        if not np.isfinite(out).all():
            raise RuntimeError("Non-finite features after geometry preprocessing.")
        return out


class StandardizeDropConstant(BaseEstimator, TransformerMixin):
    """
    Column standardization fitted on the train fold (mean, std with ddof=0), dropping the columns with std < eps.
    Statistics in float64, output in float32. Raises on non-finite values.
    """

    def __init__(self, eps: float = STD_EPS) -> None:
        # sklearn convention: store the raw param; convert to float where used.
        self.eps = eps

    def fit(self, X, y=None) -> "StandardizeDropConstant":
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"StandardizeDropConstant expects a 2-D matrix; got shape {X.shape}.")
        # Plain reductions: a NaN makes the check below fail
        mean = X.mean(axis=0, dtype=np.float64)
        std = X.std(axis=0, dtype=np.float64, ddof=0)
        if not (np.isfinite(mean).all() and np.isfinite(std).all()):
            raise RuntimeError("Non-finite feature mean/std during StandardizeDropConstant.fit.")
        keep_mask = std >= float(self.eps)
        self.mean_ = mean
        self.std_ = std
        self.keep_mask_ = keep_mask
        self.n_features_in_ = int(X.shape[1])
        self.n_features_out_ = int(keep_mask.sum())
        self.dropped_feature_indices_ = np.flatnonzero(~keep_mask).astype(np.int64)
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, ("mean_", "std_", "keep_mask_"))
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"StandardizeDropConstant expects a 2-D matrix; got shape {X.shape}.")
        if X.shape[1] != self.n_features_in_:
            raise RuntimeError(
                f"Feature dimension mismatch: got {X.shape[1]}, expected {self.n_features_in_}."
            )
        keep = self.keep_mask_
        if not keep.any():
            # No column kept: empty matrix (the classifier then raises)
            return np.zeros((X.shape[0], 0), dtype=np.float32)
        sub = np.asarray(X[:, keep], dtype=np.float32)
        mean = self.mean_[keep].astype(np.float32)
        std = self.std_[keep].astype(np.float32)
        z = (sub - mean) / std
        if not np.isfinite(z).all():
            raise RuntimeError(
                "Non-finite standardized feature values during StandardizeDropConstant.transform."
            )
        return np.asarray(z, dtype=np.float32)


def _build_logistic_regression(
    protocol_cfg: Mapping,
    *,
    seed: int,
    C_override: Optional[float] = None,
    tol_override: Optional[float] = None,
) -> LogisticRegression:
    """
    Multinomial logistic-regression probe from protocol.probe (penalty='l2', random_state=seed).
    C_override (C selected by inner CV) and tol_override (inner-CV tolerance) replace probe.C and probe.tol.
    """
    probe = dict(protocol_cfg.get("probe", {}) or {})
    kwargs = {
        "penalty": "l2",
        "C": float(C_override) if C_override is not None else float(probe.get("C", 1.0)),
        "tol": float(tol_override) if tol_override is not None else float(probe.get("tol", 1e-4)),
        "class_weight": probe.get("class_weight", "balanced"),
        "solver": str(probe.get("solver", "lbfgs")),
        "max_iter": int(probe.get("max_iter", 2000)),
        "random_state": int(seed),
    }
    requested_multi_class = str(probe.get("multi_class", "multinomial"))
    if "multi_class" in _LOGREG_INIT_PARAMS:
        kwargs["multi_class"] = requested_multi_class
    elif requested_multi_class not in {"multinomial", "auto"}:
        # sklearn >= 1.7: lbfgs is multinomial only
        raise ValueError(
            f"protocol.probe.multi_class={requested_multi_class!r} is not supported by the "
            f"installed scikit-learn ({sklearn.__version__}), which provides only multinomial "
            "softmax for the lbfgs solver."
        )
    else:
        logger.debug(
            "scikit-learn %s has no `multi_class` argument; using its default multinomial "
            "softmax (equivalent to the frozen protocol.probe.multi_class=%r).",
            sklearn.__version__,
            requested_multi_class,
        )
    return LogisticRegression(**kwargs)


def build_probe_preprocessor(cfg: Mapping, space: str) -> Pipeline:
    """The C-independent part of build_probe_pipeline: geometry(space) -> standardization (cfg is unused)."""
    geometry = resolve_geometry_preprocessing(space)
    steps: List[Tuple[str, object]] = [
        ("geometry", GeometryTransformer(strategy=geometry)),
        ("standardize", StandardizeDropConstant(eps=STD_EPS)),
    ]
    return Pipeline(steps)


def build_probe_pipeline(
    cfg: Mapping,
    space: str,
    *,
    seed: int,
    C_override: Optional[float] = None,
) -> Pipeline:
    """
    Per-fold probe pipeline: geometry(space) -> StandardizeDropConstant -> LogisticRegression (fit on the
    train fold only). C_override replaces protocol.probe.C.
    """
    protocol_cfg = cfg.get("protocol", cfg)
    steps: List[Tuple[str, object]] = list(build_probe_preprocessor(cfg, space).steps)
    steps.append(
        ("classifier", _build_logistic_regression(protocol_cfg, seed=seed, C_override=C_override))
    )
    return Pipeline(steps)


def _fit_capture_convergence(estimator, X: np.ndarray, y: np.ndarray) -> bool:
    """Fit estimator while catching warnings; return False if a ConvergenceWarning was raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X, y)
    n_convergence_warnings = sum(
        1 for w in caught if issubclass(w.category, ConvergenceWarning)
    )
    if n_convergence_warnings:
        logger.warning(
            "LogisticRegression probe did not converge (%d ConvergenceWarning(s)); "
            "consider raising protocol.probe.max_iter if this recurs.",
            n_convergence_warnings,
        )
    return n_convergence_warnings == 0


def _predict_proba_list(
    estimator, X_test_list: Sequence[np.ndarray], n_local: int
) -> List[np.ndarray]:
    """`predict_proba` each test matrix, asserting the local class width."""
    proba_list: List[np.ndarray] = []
    for i, X_test in enumerate(X_test_list):
        X_test = np.asarray(X_test)
        if X_test.ndim != 2:
            raise ValueError(f"X_test_list[{i}] must be 2-D; got shape {X_test.shape}.")
        proba = np.asarray(estimator.predict_proba(X_test), dtype=np.float32)
        if proba.ndim != 2 or proba.shape[1] != n_local:
            raise RuntimeError(
                f"predict_proba returned shape {proba.shape}; expected [*, {n_local}]."
            )
        proba_list.append(proba)
    return proba_list


def _check_train_shapes(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2-D; got shape {X_train.shape}.")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train / y_train row mismatch: {X_train.shape[0]} vs {y_train.shape[0]}."
        )
    return X_train, y_train


def fit_predict_proba(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test_list: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], np.ndarray, bool]:
    """
    Fit the probe pipeline and predict the class probabilities of each test slide.
    Returns (proba_list, clf_classes, converged): proba_list[i] is [n_i, len(clf_classes)] float32 over the
    dense class ids of y_train (clf_classes, sorted); converged is False on a ConvergenceWarning.
    """
    X_train, y_train = _check_train_shapes(X_train, y_train)
    converged = _fit_capture_convergence(pipeline, X_train, y_train)
    clf_classes = np.asarray(pipeline.classes_)
    proba_list = _predict_proba_list(pipeline, X_test_list, int(clf_classes.shape[0]))
    return proba_list, clf_classes, converged


def fit_predict_proba_preprocessed(
    classifier,
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_test_list: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], np.ndarray, bool]:
    """fit_predict_proba for already preprocessed matrices (inner-CV C search)."""
    Z_train, y_train = _check_train_shapes(Z_train, y_train)
    converged = _fit_capture_convergence(classifier, Z_train, y_train)
    clf_classes = np.asarray(classifier.classes_)
    proba_list = _predict_proba_list(classifier, Z_test_list, int(clf_classes.shape[0]))
    return proba_list, clf_classes, converged


def align_proba_to_global(
    proba: np.ndarray,
    clf_classes: np.ndarray,
    label_space: "LabelSpace",
) -> np.ndarray:
    """
    Place a fold's class probabilities on the global class axis [n, C_global] (classes absent from the
    fold's train set get 0; rows are not renormalized).
    """
    proba = np.asarray(proba, dtype=np.float32)
    if proba.ndim != 2:
        raise ValueError(f"proba must be 2-D; got shape {proba.shape}.")
    clf_classes = np.asarray(clf_classes)
    if clf_classes.ndim != 1:
        raise ValueError(f"clf_classes must be 1-D; got shape {clf_classes.shape}.")
    if proba.shape[1] != clf_classes.shape[0]:
        raise ValueError(
            f"proba column count {proba.shape[1]} != len(clf_classes) {clf_classes.shape[0]}."
        )

    labels = getattr(label_space, "labels", None)
    if labels is None:
        raise TypeError("label_space must expose a `.labels` sequence (frozen global vocabulary).")
    n_global = int(len(labels))
    if n_global <= 0:
        raise ValueError("Global label vocabulary is empty.")

    out = np.zeros((proba.shape[0], n_global), dtype=np.float32)
    for local_col in range(clf_classes.shape[0]):
        gid = int(clf_classes[local_col])
        if gid < 0 or gid >= n_global:
            raise ValueError(
                f"Classifier class id {gid} is outside the global vocab range [0, {n_global})."
            )
        out[:, gid] = proba[:, local_col]
    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite probabilities after global-axis alignment.")
    return out
