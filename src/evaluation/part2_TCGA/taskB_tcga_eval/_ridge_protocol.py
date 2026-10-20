#!/usr/bin/env python3
"""Fold-safe ridge protocol for TCGA Task B encoder and baseline evaluation."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from _eval_common import (
    PREDICTION_REQUIRED_COLUMNS,
    BenchmarkPaths,
    FeatureBundle,
    collect_output_hashes,
    json_dump_compact,
    load_case_tile_diagnostics,
    load_final_case,
    load_folds,
    load_signature_eligibility,
    load_target_table,
    manifest_header,
    merge_case_metadata,
    merge_no_suffix,
    nonempty,
    numeric_or_nan,
    ordered,
    require_columns,
    safe_name,
    sha256_obj,
    sha256_path,
    target_hash_map,
    truthy,
    utc_now_iso,
    validation_gate,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
)

DEFAULT_ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
DEFAULT_INNER_FOLDS = 3
DEFAULT_INNER_SEED = 271828
DEFAULT_BOOTSTRAP_SEED = 314159
DEFAULT_BOOTSTRAP_REPS = 2000
STD_EPS = 1e-12

# Minimum outer-train case count of an evaluable outer fold (ridge probe and mean predictor)
MIN_OUTER_TRAIN_CASES = 3

DEFAULT_POOLING = "mean_std"
# Case-level pooling options (mean_std is the default)
VALID_POOLINGS: Tuple[str, ...] = ("mean", "mean_std", "mean_std_max")


@dataclass(frozen=True)
class RidgeProtocolConfig:
    alpha_grid: Tuple[float, ...] = tuple(DEFAULT_ALPHA_GRID)
    inner_folds: int = DEFAULT_INNER_FOLDS
    inner_seed: int = DEFAULT_INNER_SEED
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    bootstrap_valid_fraction_min: float = 0.95
    feature_std_eps: float = STD_EPS
    target_std_eps: float = STD_EPS
    pooling: str = DEFAULT_POOLING

    def __post_init__(self) -> None:
        if self.pooling not in VALID_POOLINGS:
            raise ValueError(
                f"Unknown pooling {self.pooling!r}; expected one of {VALID_POOLINGS}"
            )

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "alpha_grid": [float(a) for a in self.alpha_grid],
            "inner_folds": int(self.inner_folds),
            "inner_seed": int(self.inner_seed),
            "bootstrap_reps": int(self.bootstrap_reps),
            "bootstrap_seed": int(self.bootstrap_seed),
            "bootstrap_valid_fraction_min": float(self.bootstrap_valid_fraction_min),
            "primary_bootstrap_default": "case_block_within_project",
            "primary_bootstrap_reported": ["case_block_within_project", "cell_then_case"],
            "primary_bootstrap_hierarchical_seed": int(self.bootstrap_seed) + 1,
            "feature_std_eps": float(self.feature_std_eps),
            "target_std_eps": float(self.target_std_eps),
            "ridge": {"class": "sklearn.linear_model.Ridge", "fit_intercept": False, "solver": "svd"},
            "feature_preprocessing": "outer_train_mean_std_drop_std_lt_eps_then_apply_fixed_mask",
            "target_preprocessing": "outer_train_y_mean_std_then_inverse_predictions_to_frozen_target_scale",
            "alpha_selection": "minimize_inner_validation_mse_tiebreak_smallest_alpha",
            "alpha_selection_uses_inner_feature_refit": True,
            "alpha_selection_target_scale": "outer_train_y_mean_std",
            "inner_cv_split_seed_keys": ["project_id", "signature_name", "fold_id"],
            "inner_cv_split_seed_keys_explicitly_exclude": ["encoder_id", "baseline_id", "training_num", "checkpoint_num"],
            "target_recomputed_inside_cv": False,
        }
        if self.pooling != DEFAULT_POOLING:
            out["pooling"] = str(self.pooling)
        return out

    @property
    def sha256(self) -> str:
        return sha256_obj(self.as_dict())


@dataclass
class FoldTransformReport:
    input_numeric_columns: List[str]
    input_categorical_columns: List[str]
    numeric_impute_values: Dict[str, Optional[float]]
    dropped_all_missing_numeric_columns: List[str]
    categorical_levels: Dict[str, List[str]]
    raw_feature_names: List[str]
    standardized_feature_names: List[str]
    dropped_low_std_feature_names: List[str]
    dropped_low_std_count: int
    output_dim: int


class FoldFeatureTransformer:
    """
    Fold-local preprocessing fitted on the outer train: numeric columns median-imputed (all-missing ones dropped),
    categorical columns one-hot encoded (missing or unseen -> <UNK>), then standardized, dropping dimensions with std < eps.
    """

    def __init__(self, numeric_columns: Sequence[str], categorical_columns: Sequence[str], *, std_eps: float = STD_EPS) -> None:
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)
        self.std_eps = float(std_eps)
        self.numeric_keep_: List[str] = []
        self.numeric_median_: Dict[str, float] = {}
        self.dropped_all_missing_numeric_: List[str] = []
        self.cat_levels_: Dict[str, List[str]] = {}
        self.raw_feature_names_: List[str] = []
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.keep_mask_: Optional[np.ndarray] = None
        self.standardized_feature_names_: List[str] = []
        self.dropped_low_std_feature_names_: List[str] = []

    @staticmethod
    def _clean_cat(x: Any) -> str:
        if not nonempty(x):
            return "<UNK>"
        return str(x).strip()

    def _numeric_matrix_fit(self, df: pd.DataFrame) -> np.ndarray:
        cols: List[str] = []
        arrs: List[np.ndarray] = []
        for c in self.numeric_columns:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) if c in df.columns else np.full(len(df), np.nan)
            finite = np.isfinite(v)
            if not finite.any():
                self.dropped_all_missing_numeric_.append(c)
                continue
            med = float(np.nanmedian(v[finite]))
            v = np.where(np.isfinite(v), v, med)
            cols.append(c)
            self.numeric_median_[c] = med
            arrs.append(v.reshape(-1, 1))
        self.numeric_keep_ = cols
        if not arrs:
            return np.zeros((len(df), 0), dtype=float)
        return np.hstack(arrs).astype(float)

    def _numeric_matrix_transform(self, df: pd.DataFrame) -> np.ndarray:
        arrs: List[np.ndarray] = []
        for c in self.numeric_keep_:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) if c in df.columns else np.full(len(df), np.nan)
            med = float(self.numeric_median_[c])
            v = np.where(np.isfinite(v), v, med)
            arrs.append(v.reshape(-1, 1))
        if not arrs:
            return np.zeros((len(df), 0), dtype=float)
        return np.hstack(arrs).astype(float)

    def _categorical_matrix_fit(self, df: pd.DataFrame) -> np.ndarray:
        arrs: List[np.ndarray] = []
        names: List[str] = []
        for c in self.categorical_columns:
            vals = df[c].map(self._clean_cat) if c in df.columns else pd.Series(["<UNK>"] * len(df), index=df.index)
            cats = sorted(set(vals.astype(str).tolist()) | {"<UNK>"})
            self.cat_levels_[c] = cats
            mat = np.zeros((len(df), len(cats)), dtype=float)
            pos = {v: i for i, v in enumerate(cats)}
            for i, v in enumerate(vals.astype(str).tolist()):
                mat[i, pos.get(v, pos["<UNK>"])] = 1.0
            arrs.append(mat)
            names.extend([f"{c}={v}" for v in cats])
        if not arrs:
            return np.zeros((len(df), 0), dtype=float)
        self._cat_names_tmp = names  # type: ignore[attr-defined]
        return np.hstack(arrs).astype(float)

    def _categorical_matrix_transform(self, df: pd.DataFrame) -> np.ndarray:
        arrs: List[np.ndarray] = []
        for c in self.categorical_columns:
            cats = self.cat_levels_.get(c, ["<UNK>"])
            pos = {v: i for i, v in enumerate(cats)}
            vals = df[c].map(self._clean_cat) if c in df.columns else pd.Series(["<UNK>"] * len(df), index=df.index)
            mat = np.zeros((len(df), len(cats)), dtype=float)
            unk = pos.get("<UNK>", 0)
            for i, v in enumerate(vals.astype(str).tolist()):
                mat[i, pos.get(v, unk)] = 1.0
            arrs.append(mat)
        if not arrs:
            return np.zeros((len(df), 0), dtype=float)
        return np.hstack(arrs).astype(float)

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        x_num = self._numeric_matrix_fit(train_df)
        x_cat = self._categorical_matrix_fit(train_df)
        cat_names = getattr(self, "_cat_names_tmp", [])
        self.raw_feature_names_ = list(self.numeric_keep_) + list(cat_names)
        x = np.hstack([x_num, x_cat]).astype(float) if (x_num.shape[1] + x_cat.shape[1]) else np.zeros((len(train_df), 0), dtype=float)
        if x.shape[1] == 0:
            self.mean_ = np.zeros(0, dtype=float)
            self.std_ = np.ones(0, dtype=float)
            self.keep_mask_ = np.zeros(0, dtype=bool)
            self.standardized_feature_names_ = []
            self.dropped_low_std_feature_names_ = []
            return x
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0, ddof=0)
        if not np.isfinite(self.mean_).all() or not np.isfinite(self.std_).all():
            raise RuntimeError("Non-finite feature mean/std during fold preprocessing")
        self.keep_mask_ = self.std_ >= self.std_eps
        self.dropped_low_std_feature_names_ = [n for n, keep in zip(self.raw_feature_names_, self.keep_mask_) if not bool(keep)]
        self.standardized_feature_names_ = [n for n, keep in zip(self.raw_feature_names_, self.keep_mask_) if bool(keep)]
        if not self.keep_mask_.any():
            return np.zeros((len(train_df), 0), dtype=float)
        z = (x[:, self.keep_mask_] - self.mean_[self.keep_mask_]) / self.std_[self.keep_mask_]
        if not np.isfinite(z).all():
            raise RuntimeError("Non-finite standardized feature values")
        return z.astype(float)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None or self.std_ is None or self.keep_mask_ is None:
            raise RuntimeError("FoldFeatureTransformer must be fitted first")
        x_num = self._numeric_matrix_transform(df)
        x_cat = self._categorical_matrix_transform(df)
        x = np.hstack([x_num, x_cat]).astype(float) if (x_num.shape[1] + x_cat.shape[1]) else np.zeros((len(df), 0), dtype=float)
        if x.shape[1] != len(self.mean_):
            raise RuntimeError(f"Feature dimension mismatch after preprocessing: got {x.shape[1]}, expected {len(self.mean_)}")
        if not self.keep_mask_.any():
            return np.zeros((len(df), 0), dtype=float)
        z = (x[:, self.keep_mask_] - self.mean_[self.keep_mask_]) / self.std_[self.keep_mask_]
        if not np.isfinite(z).all():
            raise RuntimeError("Non-finite standardized feature values on transform")
        return z.astype(float)

    def report(self) -> FoldTransformReport:
        return FoldTransformReport(
            input_numeric_columns=list(self.numeric_columns),
            input_categorical_columns=list(self.categorical_columns),
            numeric_impute_values={c: self.numeric_median_.get(c) for c in self.numeric_columns},
            dropped_all_missing_numeric_columns=list(self.dropped_all_missing_numeric_),
            categorical_levels={k: list(v) for k, v in self.cat_levels_.items()},
            raw_feature_names=list(self.raw_feature_names_),
            standardized_feature_names=list(self.standardized_feature_names_),
            dropped_low_std_feature_names=list(self.dropped_low_std_feature_names_),
            dropped_low_std_count=int(len(self.dropped_low_std_feature_names_)),
            output_dim=int(len(self.standardized_feature_names_)),
        )


def _is_constant(x: np.ndarray) -> bool:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return True
    return bool(np.nanmax(x) - np.nanmin(x) <= 0.0)


def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None
    x = x[mask]
    y = y[mask]
    if _is_constant(x) or _is_constant(y):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return None
    rx = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
    return _pearson(rx, ry)


def compute_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, Any]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[mask]
    yp = yp[mask]
    if len(yt) == 0:
        return {"n_cases": 0, "spearman": None, "pearson": None, "r2": None, "mae": None, "rmse": None}
    resid = yp - yt
    denom = float(np.sum((yt - float(np.mean(yt))) ** 2))
    r2 = None if denom <= 0 else float(1.0 - np.sum(resid ** 2) / denom)
    return {
        "n_cases": int(len(yt)),
        "spearman": _spearman(yt, yp),
        "pearson": _pearson(yt, yp),
        "r2": r2,
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
    }


def _ridge_fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float) -> np.ndarray:
    model = Ridge(alpha=float(alpha), fit_intercept=False, solver="svd")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return np.asarray(pred, dtype=float)


def _stable_seed_offset(*parts: Any, modulus: int = 100_000) -> int:
    """Deterministic seed offset from a sha256 of the parts (not hash(), which is salted per process)."""
    h = sha256_obj([str(p) for p in parts])
    return int(str(h)[:12], 16) % int(modulus)


def _finite_mean_or_none(values: Any) -> Optional[float]:
    s = pd.to_numeric(pd.Series(list(values) if not isinstance(values, pd.Series) else values), errors="coerce")
    arr = s.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _select_alpha(
    train_df: pd.DataFrame,
    y_train_scaled: np.ndarray,
    *,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    cfg: RidgeProtocolConfig,
    rng_seed_offset: int = 0,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Select the ridge alpha minimizing the inner-CV mean squared error (ties: smallest alpha). Returns (alpha, rows)."""
    n = int(len(y_train_scaled))
    if n < 2:
        raise RuntimeError("Cannot select ridge alpha with fewer than two outer-train cases")
    n_splits = min(int(cfg.inner_folds), n)
    if n_splits < 2:
        raise RuntimeError("Inner CV requires at least two splits")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(cfg.inner_seed + rng_seed_offset))
    rows: List[Dict[str, Any]] = []
    best_alpha: Optional[float] = None
    best_mse: Optional[float] = None
    for alpha in cfg.alpha_grid:
        mses: List[float] = []
        for split_id, (tr, va) in enumerate(kf.split(train_df)):
            sub_train = train_df.iloc[tr]
            sub_val = train_df.iloc[va]
            # Feature preprocessing refit on the inner train split (target kept on the outer-train scale)
            sub_tf = FoldFeatureTransformer(numeric_columns, categorical_columns, std_eps=float(cfg.feature_std_eps))
            X_tr = sub_tf.fit_transform(sub_train)
            X_va = sub_tf.transform(sub_val)
            y_tr_raw = y_train_scaled[tr]  # outer-train target scale
            if X_tr.shape[1] == 0:
                pred = np.full(len(va), float(np.mean(y_tr_raw)), dtype=float)
            else:
                pred = _ridge_fit_predict(X_tr, y_tr_raw, X_va, float(alpha))
            mse = float(np.mean((pred - y_train_scaled[va]) ** 2))
            mses.append(mse)
            rows.append({"alpha": float(alpha), "inner_split": int(split_id), "inner_mse": mse})
        mean_mse = float(np.mean(mses))
        if best_mse is None or mean_mse < best_mse - 1e-15 or (abs(mean_mse - best_mse) <= 1e-15 and float(alpha) < float(best_alpha)):
            best_mse = mean_mse
            best_alpha = float(alpha)
    assert best_alpha is not None and best_mse is not None
    for r in rows:
        r["selected_alpha"] = best_alpha
        r["selected_alpha_mean_inner_mse"] = best_mse
    return best_alpha, rows


def _signature_metadata(elig_row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "support_category": str(elig_row.get("support_category", "")),
        "domain_slice": str(elig_row.get("domain_slice", "")),
        "signature_family": str(elig_row.get("signature_family", "")),
        "signature_source_name": str(elig_row.get("signature_source_name", elig_row.get("source_name", ""))),
        "signature_source_version": str(elig_row.get("signature_source_version", elig_row.get("source_version", ""))),
        "signature_set_hash": str(elig_row.get("signature_set_hash", "")),
    }


def _prepare_prediction_rows(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    fold_id: Any,
    project_id: str,
    signature_name: str,
    elig_row: Mapping[str, Any],
    run_info: Mapping[str, Any],
    target_table_hash: str,
    fold_assignment_hash: str,
    feature_file_hash: str,
    repeat_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    sig = _signature_metadata(elig_row)
    rows: List[Dict[str, Any]] = []
    for i, r in enumerate(df_test.to_dict("records")):
        row = {
            "case_id": str(r.get("case_id")),
            "case_uuid": str(r.get("case_uuid")),
            "project_id": str(project_id),
            "fold_id": str(fold_id),
            "y_true": float(y_true[i]),
            "y_pred": float(y_pred[i]),
            "n_tiles_eligible_before_cap": r.get("n_tiles_eligible_before_cap"),
            "n_tiles_used_after_cap": r.get("n_tiles_used_after_cap"),
            "n_slides_retained": r.get("n_slides_retained"),
            "match_tier_best_case": r.get("match_tier_best_case"),
            "biospecimen_match_level": r.get("biospecimen_match_level"),
            "official_primary_match": r.get("official_primary_match"),
            "encoder_id": str(run_info.get("encoder_id", run_info.get("baseline_id", ""))),
            "training_num": run_info.get("training_num", ""),
            "checkpoint_num": run_info.get("checkpoint_num", ""),
            "space": str(run_info.get("space", "")),
            "modelgroup_id": str(run_info.get("modelgroup_id", "")),
            "tile_feature_postproc": str(run_info.get("tile_feature_postproc", "")),
            "signature_name": str(signature_name),
            "target_table_hash": str(target_table_hash),
            "fold_assignment_hash": str(fold_assignment_hash),
            "feature_file_hash": str(feature_file_hash),
            "dominant_slide_tile_fraction_after_cap": r.get("dominant_slide_tile_fraction_after_cap"),
            "slide_tile_entropy_after_cap": r.get("slide_tile_entropy_after_cap"),
            "slide_tile_effective_n_after_cap": r.get("slide_tile_effective_n_after_cap"),
            "single_slide_case": r.get("single_slide_case"),
            "multi_slide_case": r.get("multi_slide_case"),
            "cap_applied": r.get("cap_applied"),
            **sig,
        }
        if repeat_id is not None:
            row["repeat_id"] = int(repeat_id)
        rows.append(row)
    return rows


def _prediction_filename(project_id: str, signature_name: str, repeat_id: Optional[int] = None) -> str:
    base = f"{safe_name(project_id)}__{safe_name(signature_name)}"
    if repeat_id is not None:
        base = f"repeat_{int(repeat_id):03d}__{base}"
    return base + ".parquet"


def _cell_metric_row(pred: pd.DataFrame, elig_row: Mapping[str, Any], *, repeat_id: Optional[int] = None) -> Dict[str, Any]:
    metrics = compute_metrics(pred["y_true"].to_numpy(dtype=float), pred["y_pred"].to_numpy(dtype=float))
    row = {
        "project_id": str(elig_row.get("project_id", pred["project_id"].iloc[0] if not pred.empty else "")),
        "signature_name": str(elig_row.get("signature_name", pred["signature_name"].iloc[0] if not pred.empty else "")),
        "eligible": bool(truthy(elig_row.get("eligible", True))),
        "primary_score_cell": bool(truthy(elig_row.get("primary_score_cell", False))),
        "domain_slice": str(elig_row.get("domain_slice", "")),
        "support_category": str(elig_row.get("support_category", "")),
        "signature_family": str(elig_row.get("signature_family", "")),
        "signature_source_name": str(elig_row.get("signature_source_name", elig_row.get("source_name", ""))),
        "signature_source_version": str(elig_row.get("signature_source_version", elig_row.get("source_version", ""))),
        "all_official_primary_match": bool(pred["official_primary_match"].map(truthy).all()) if "official_primary_match" in pred else False,
        "n_folds_predicted": int(pred["fold_id"].nunique()) if "fold_id" in pred else 0,
        **metrics,
    }
    if repeat_id is not None:
        row["repeat_id"] = int(repeat_id)
    return row


def _macro(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(group_cols) + ["n_cells", "macro_spearman", "macro_pearson", "macro_r2", "macro_mae", "macro_rmse"])
    rows = []
    for key, g in df.groupby(list(group_cols), dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {c: key[i] for i, c in enumerate(group_cols)}
        row.update({
            "n_cells": int(len(g)),
            "macro_spearman": _finite_mean_or_none(g["spearman"]),
            "macro_pearson": _finite_mean_or_none(g["pearson"]),
            "macro_r2": _finite_mean_or_none(g["r2"]),
            "macro_mae": _finite_mean_or_none(g["mae"]),
            "macro_rmse": _finite_mean_or_none(g["rmse"]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _primary_mask(cell_metrics: pd.DataFrame) -> pd.Series:
    if cell_metrics.empty:
        return pd.Series(False, index=cell_metrics.index)
    return (
        cell_metrics["eligible"].map(truthy)
        & cell_metrics["domain_slice"].astype(str).eq("in_domain")
        & cell_metrics["support_category"].astype(str).eq("supported")
        & cell_metrics["all_official_primary_match"].map(truthy)
    )


def _bootstrap_primary_case_block(
    predictions_by_cell: Mapping[Tuple[str, str], pd.DataFrame],
    primary_cells: Sequence[Tuple[str, str]],
    cfg: RidgeProtocolConfig,
) -> Dict[str, Any]:
    """
    Case-level block bootstrap of the primary macro Spearman: one case resample per project, shared by all
    primary cells of that project.
    """
    cells = list(primary_cells)
    if not cells:
        raise RuntimeError("No primary eligible cells for bootstrap")
    # Build case_id-aligned arrays per cell and group cells by project.
    by_project: Dict[str, List[Tuple[str, str]]] = {}
    cell_arrays: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    project_case_ids: Dict[str, List[str]] = {}
    for cell in cells:
        project_id = str(cell[0])
        df = predictions_by_cell[cell].sort_values("case_id").reset_index(drop=True)
        case_ids = df["case_id"].astype(str).tolist()
        cell_arrays[cell] = (df["y_true"].to_numpy(dtype=float), df["y_pred"].to_numpy(dtype=float))
        if project_id not in project_case_ids:
            project_case_ids[project_id] = case_ids
        elif case_ids != project_case_ids[project_id]:
            raise RuntimeError(
                f"Primary cells of project {project_id} do not share an identical case set; "
                "the case-block bootstrap requires aligned cases"
            )
        by_project.setdefault(project_id, []).append(cell)
    rng = np.random.default_rng(int(cfg.bootstrap_seed))
    macro_vals = np.full(int(cfg.bootstrap_reps), np.nan, dtype=float)
    cell_valid_counts = {cell: 0 for cell in cells}
    projects_sorted = sorted(by_project)
    for b in range(int(cfg.bootstrap_reps)):
        vals: List[float] = []
        for project_id in projects_sorted:
            n = len(project_case_ids[project_id])
            if n < 2:
                continue
            idx = rng.integers(0, n, size=n)
            for cell in by_project[project_id]:
                yt, yp = cell_arrays[cell]
                sp = _spearman(yt[idx], yp[idx])
                if sp is None or not math.isfinite(sp):
                    continue
                vals.append(float(sp))
                cell_valid_counts[cell] += 1
        if vals:
            macro_vals[b] = float(np.mean(vals))
    valid = np.isfinite(macro_vals)
    if not valid.any():
        return {
            "scheme": "case_block_within_project",
            "bootstrap_reps": int(cfg.bootstrap_reps),
            "bootstrap_seed": int(cfg.bootstrap_seed),
            "valid_fraction": 0.0,
            "ci95_low": None,
            "ci95_high": None,
            "cell_valid_fraction_min": 0.0,
            "warnings": ["all_bootstrap_replicates_invalid"],
        }
    cell_fracs = {f"{p}::{s}": float(v / max(1, int(cfg.bootstrap_reps))) for (p, s), v in cell_valid_counts.items()}
    min_frac = min(cell_fracs.values()) if cell_fracs else 0.0
    warnings: List[str] = []
    if min_frac < float(cfg.bootstrap_valid_fraction_min):
        warnings.append(f"cell_bootstrap_valid_fraction_below_threshold:min={min_frac:.6f},threshold={cfg.bootstrap_valid_fraction_min}")
    return {
        "scheme": "case_block_within_project",
        "bootstrap_reps": int(cfg.bootstrap_reps),
        "bootstrap_seed": int(cfg.bootstrap_seed),
        "valid_fraction": float(np.mean(valid)),
        "ci95_low": float(np.nanpercentile(macro_vals, 2.5)),
        "ci95_high": float(np.nanpercentile(macro_vals, 97.5)),
        "cell_valid_fraction_min": float(min_frac),
        "cell_valid_fraction_by_cell": cell_fracs,
        "warnings": warnings,
    }


def _bootstrap_primary_hierarchical(
    predictions_by_cell: Mapping[Tuple[str, str], pd.DataFrame],
    primary_cells: Sequence[Tuple[str, str]],
    cfg: RidgeProtocolConfig,
) -> Dict[str, Any]:
    cells = list(primary_cells)
    if not cells:
        return {
            "scheme": "cell_then_case",
            "bootstrap_reps": int(cfg.bootstrap_reps),
            "bootstrap_seed": int(cfg.bootstrap_seed) + 1,
            "valid_fraction": 0.0,
            "ci95_low": None,
            "ci95_high": None,
            "warnings": ["no_primary_cells"],
        }
    rng = np.random.default_rng(int(cfg.bootstrap_seed) + 1)
    macro_vals = np.full(int(cfg.bootstrap_reps), np.nan, dtype=float)
    n_cells = len(cells)
    for b in range(int(cfg.bootstrap_reps)):
        chosen = rng.integers(0, n_cells, size=n_cells)
        vals = []
        for ci in chosen:
            df = predictions_by_cell[cells[ci]]
            n = int(len(df))
            if n < 2:
                continue
            idx = rng.integers(0, n, size=n)
            sp = _spearman(df["y_true"].to_numpy(dtype=float)[idx], df["y_pred"].to_numpy(dtype=float)[idx])
            if sp is not None and math.isfinite(sp):
                vals.append(float(sp))
        if vals:
            macro_vals[b] = float(np.mean(vals))
    valid = np.isfinite(macro_vals)
    return {
        "scheme": "cell_then_case",
        "bootstrap_reps": int(cfg.bootstrap_reps),
        "bootstrap_seed": int(cfg.bootstrap_seed) + 1,
        "valid_fraction": float(np.mean(valid)) if valid.size else 0.0,
        "ci95_low": float(np.nanpercentile(macro_vals, 2.5)) if valid.any() else None,
        "ci95_high": float(np.nanpercentile(macro_vals, 97.5)) if valid.any() else None,
    }


def _secondary_scores(cell_metrics: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def add(name: str, mask: pd.Series) -> None:
        g = cell_metrics.loc[mask].copy()
        vals = pd.to_numeric(g.get("spearman", pd.Series(dtype=float)), errors="coerce")
        finite_mask = np.isfinite(vals.to_numpy(dtype=float)) if len(g) else np.zeros(0, dtype=bool)
        rows = g.to_dict("records")
        out[name] = {
            "n_cells": int(len(g)),
            "n_cells_with_defined_spearman": int(finite_mask.sum()),
            "n_cells_dropped_for_undefined_metric": int(len(g) - int(finite_mask.sum())),
            "macro_spearman": _finite_mean_or_none(vals),
            "cells_dropped_for_undefined_metric": [
                f"{r['project_id']}::{r['signature_name']}"
                for r, ok in zip(rows, finite_mask)
                if not bool(ok)
            ],
        }

    add("primary_in_domain_supported_official", _primary_mask(cell_metrics))
    add("in_domain_supported_plus_partial", cell_metrics["domain_slice"].astype(str).eq("in_domain") & cell_metrics["support_category"].astype(str).isin(["supported", "partial"]))
    add("in_domain_all_eligible", cell_metrics["domain_slice"].astype(str).eq("in_domain"))
    add("out_of_domain_all_eligible", cell_metrics["domain_slice"].astype(str).eq("out_of_domain"))
    add("all_projects_all_eligible", cell_metrics["eligible"].map(truthy))
    return out


def _write_metrics(
    *,
    output_root: Path,
    predictions_by_cell: Mapping[Tuple[str, str], pd.DataFrame],
    elig_by_cell: Mapping[Tuple[str, str], Mapping[str, Any]],
    cfg: RidgeProtocolConfig,
    overwrite: bool,
    repeat_id: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    cell_rows: List[Dict[str, Any]] = []
    for cell, pred in sorted(predictions_by_cell.items()):
        cell_rows.append(_cell_metric_row(pred, elig_by_cell[cell], repeat_id=repeat_id))
    cell_metrics = pd.DataFrame(cell_rows)
    write_csv_atomic(metrics_dir / "cell_metrics.csv", cell_metrics, overwrite=overwrite)
    project_macro = _macro(cell_metrics, ["project_id"] + (["repeat_id"] if repeat_id is not None else []))
    signature_macro = _macro(cell_metrics, ["signature_name"] + (["repeat_id"] if repeat_id is not None else []))
    write_csv_atomic(metrics_dir / "project_macro.csv", project_macro, overwrite=overwrite)
    write_csv_atomic(metrics_dir / "signature_macro.csv", signature_macro, overwrite=overwrite)
    primary = cell_metrics.loc[_primary_mask(cell_metrics)].copy()
    if primary.empty:
        raise RuntimeError("No primary-score eligible project × signature cells after metric computation")
    primary_vals = pd.to_numeric(primary["spearman"], errors="coerce")
    finite_mask = np.isfinite(primary_vals.to_numpy(dtype=float))
    n_primary_total = int(len(primary))
    n_primary_defined = int(finite_mask.sum())
    primary_observed = _finite_mean_or_none(primary_vals)
    primary_cells = [(str(r["project_id"]), str(r["signature_name"])) for r in primary.to_dict("records")]
    if int(cfg.bootstrap_reps) > 0:
        boot = _bootstrap_primary_case_block(predictions_by_cell, primary_cells, cfg)
        boot_hier = _bootstrap_primary_hierarchical(predictions_by_cell, primary_cells, cfg)
    else:
        boot = {
            "scheme": "case_block_within_project",
            "bootstrap_reps": int(cfg.bootstrap_reps),
            "bootstrap_seed": int(cfg.bootstrap_seed),
            "valid_fraction": None,
            "ci95_low": None,
            "ci95_high": None,
            "warnings": ["bootstrap_skipped_for_zero_reps"],
        }
        boot_hier = {
            "scheme": "cell_then_case",
            "bootstrap_reps": int(cfg.bootstrap_reps),
            "bootstrap_seed": int(cfg.bootstrap_seed) + 1,
            "valid_fraction": None,
            "ci95_low": None,
            "ci95_high": None,
            "warnings": ["bootstrap_skipped_for_zero_reps"],
        }
    summary = {
        "created_at_utc": utc_now_iso(),
        "primary_score_name": "macro_mean_spearman_in_domain_supported_official_primary",
        "primary_macro_spearman": primary_observed,
        "primary_n_cells": n_primary_total,
        "primary_n_cells_with_defined_spearman": n_primary_defined,
        "primary_n_cells_dropped_for_undefined_metric": int(n_primary_total - n_primary_defined),
        "primary_cells": [f"{p}::{s}" for p, s in primary_cells],
        "primary_cells_dropped_for_undefined_metric": [
            f"{r['project_id']}::{r['signature_name']}"
            for r, ok in zip(primary.to_dict("records"), finite_mask)
            if not bool(ok)
        ],
        "primary_bootstrap": boot,
        "primary_bootstrap_hierarchical": boot_hier,
        "secondary_scores": _secondary_scores(cell_metrics),
    }
    write_json_atomic(metrics_dir / "global_summary.json", summary, overwrite=overwrite)
    return cell_metrics, summary


def _fold_is_viable(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    project_id: str,
    signature: str,
    fold_id: Any,
) -> bool:
    """
    Outer-fold viability, shared by the ridge probe and the mean predictor: False (skip) if the outer test is empty;
    raises if the outer train has fewer than MIN_OUTER_TRAIN_CASES cases; else True.
    """
    if test.empty:
        return False
    if len(train) < MIN_OUTER_TRAIN_CASES:
        raise RuntimeError(
            f"Outer train too small for {project_id} × {signature}, fold={fold_id}: n_train={len(train)}"
        )
    return True


def evaluate_ridge_feature_bundle(
    *,
    output_root: Path,
    paths: BenchmarkPaths,
    feature_bundle: FeatureBundle,
    run_info: Mapping[str, Any],
    cfg: RidgeProtocolConfig,
    overwrite: bool = False,
    allow_unvalidated: bool = False,
    target_permutation_seed: Optional[int] = None,
    repeat_id: Optional[int] = None,
    progress_desc: Optional[str] = None,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    predictions_dir = output_root / "predictions"
    diagnostics_dir = output_root / "diagnostics"
    metrics_dir = output_root / "metrics"
    for d in [predictions_dir, diagnostics_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    validation = validation_gate(paths, allow_unvalidated=allow_unvalidated)
    final_case = load_final_case(paths)
    folds = load_folds(paths)
    diagnostics = load_case_tile_diagnostics(paths)
    case_meta = merge_case_metadata(final_case, diagnostics)
    eligibility = load_signature_eligibility(paths)
    fold_hash = sha256_path(paths.folds_dir / "fold_assignments.csv")
    target_hashes = target_hash_map(paths)

    feat = feature_bundle.features.copy()
    require_columns(feat, ["case_id", "case_uuid", "project_id"], "features")
    if feat["case_id"].astype(str).duplicated().any():
        dups = feat.loc[feat["case_id"].astype(str).duplicated(), "case_id"].head(20).tolist()
        raise RuntimeError(f"Duplicate case_id rows in features are forbidden: {dups}")
    final_cases = set(final_case["case_id"].astype(str))
    feat_cases = set(feat["case_id"].astype(str))
    if feat_cases != final_cases:
        missing = sorted(final_cases - feat_cases)[:20]
        extra = sorted(feat_cases - final_cases)[:20]
        raise RuntimeError(f"Feature cases must exactly match final eligible cases. Missing={missing}; extra={extra}")

    merged_base = feat.merge(folds[["case_id", "fold_id"]], on="case_id", how="left", validate="one_to_one")
    if merged_base["fold_id"].isna().any():
        raise RuntimeError("Some feature cases lack fold assignments")
    # Feature columns must not reuse case-metadata column names (merge_no_suffix raises on collisions)
    merged_base = merge_no_suffix(
        merged_base,
        case_meta.drop(columns=["case_uuid", "project_id"], errors="ignore"),
        on="case_id", how="left", validate="one_to_one",
        name="ridge merged_base <- case_meta",
    )
    merged_base = merged_base.sort_values(["project_id", "case_id"]).reset_index(drop=True)

    if target_permutation_seed is not None:
        perm_rng = np.random.default_rng(int(target_permutation_seed))
    else:
        perm_rng = None

    alpha_rows: List[Dict[str, Any]] = []
    transform_rows: List[Dict[str, Any]] = []
    fit_rows: List[Dict[str, Any]] = []
    predictions_by_cell: Dict[Tuple[str, str], pd.DataFrame] = {}
    elig_by_cell: Dict[Tuple[str, str], Mapping[str, Any]] = {}

    elig_records = eligibility.to_dict("records")
    elig_iter = tqdm(elig_records, desc=progress_desc, unit="sig") if progress_desc else elig_records
    for elig in elig_iter:
        project_id = str(elig["project_id"])
        signature = str(elig["signature_name"])
        target_table, target_path, target_hash, target_manifest = load_target_table(paths, project_id)
        if signature not in target_table.columns:
            raise RuntimeError(f"Signature {signature!r} not found in target table {target_path}")
        cell = (project_id, signature)
        base_p = merged_base.loc[merged_base["project_id"].astype(str).eq(project_id)].copy()
        if base_p.empty:
            raise RuntimeError(f"No features for project {project_id}")
        df = base_p.merge(target_table[["case_id", signature]], on="case_id", how="inner", validate="one_to_one")
        df = df.rename(columns={signature: "__target__"})
        df["__target__"] = pd.to_numeric(df["__target__"], errors="coerce")
        df = df.loc[np.isfinite(df["__target__"].to_numpy(dtype=float))].copy()
        if df.empty:
            raise RuntimeError(f"No finite targets for {project_id} × {signature}")
        if _is_constant(df["__target__"].to_numpy(dtype=float)):
            raise RuntimeError(f"Eligible target is constant for {project_id} × {signature}")

        pred_rows: List[Dict[str, Any]] = []
        folds_sorted = sorted(df["fold_id"].astype(str).unique(), key=lambda x: (int(x) if str(x).isdigit() else str(x)))
        for fold_id in folds_sorted:
            train = df.loc[~df["fold_id"].astype(str).eq(str(fold_id))].copy()
            test = df.loc[df["fold_id"].astype(str).eq(str(fold_id))].copy()
            if not _fold_is_viable(train, test, project_id=project_id, signature=signature, fold_id=fold_id):
                continue
            y_train_raw = train["__target__"].to_numpy(dtype=float)
            y_test_raw = test["__target__"].to_numpy(dtype=float)
            y_mean = float(np.mean(y_train_raw))
            y_std = float(np.std(y_train_raw, ddof=0))
            if not math.isfinite(y_std) or y_std < float(cfg.target_std_eps):
                raise RuntimeError(f"Outer-train target std too small for {project_id} × {signature}, fold={fold_id}")
            y_train = (y_train_raw - y_mean) / y_std
            if perm_rng is not None:
                y_train = perm_rng.permutation(y_train)

            transformer = FoldFeatureTransformer(feature_bundle.numeric_columns, feature_bundle.categorical_columns, std_eps=float(cfg.feature_std_eps))
            X_train = transformer.fit_transform(train)
            X_test = transformer.transform(test)
            tr_report = transformer.report()
            seed_offset = _stable_seed_offset(project_id, signature, fold_id)
            if X_train.shape[1] == 0:
                # No feature left: predict the outer-train mean (recorded with inner_split=-1)
                selected_alpha = float(cfg.alpha_grid[0])
                inner_rows = [{"alpha": selected_alpha, "inner_split": -1, "inner_mse": float(np.mean((y_train - np.mean(y_train)) ** 2)), "selected_alpha": selected_alpha, "selected_alpha_mean_inner_mse": float(np.mean((y_train - np.mean(y_train)) ** 2))}]
                y_pred_scaled = np.full(len(test), float(np.mean(y_train)), dtype=float)
            else:
                selected_alpha, inner_rows = _select_alpha(
                    train,
                    y_train,
                    numeric_columns=feature_bundle.numeric_columns,
                    categorical_columns=feature_bundle.categorical_columns,
                    cfg=cfg,
                    rng_seed_offset=seed_offset,
                )
                y_pred_scaled = _ridge_fit_predict(X_train, y_train, X_test, selected_alpha)
            y_pred = y_pred_scaled * y_std + y_mean
            if not np.isfinite(y_pred).all():
                raise RuntimeError(f"Non-finite predictions for {project_id} × {signature}, fold={fold_id}")

            for ir in inner_rows:
                alpha_rows.append({
                    "project_id": project_id,
                    "signature_name": signature,
                    "fold_id": str(fold_id),
                    "repeat_id": repeat_id if repeat_id is not None else pd.NA,
                    **ir,
                    "inner_seed": int(cfg.inner_seed + seed_offset),
                    "inner_seed_offset": int(seed_offset),
                    "n_outer_train": int(len(train)),
                    "n_outer_test": int(len(test)),
                    "target_permutation_seed": target_permutation_seed if target_permutation_seed is not None else pd.NA,
                })
            transform_rows.append({
                "project_id": project_id,
                "signature_name": signature,
                "fold_id": str(fold_id),
                "repeat_id": repeat_id if repeat_id is not None else pd.NA,
                "n_outer_train": int(len(train)),
                "n_outer_test": int(len(test)),
                "y_mean_outer_train": y_mean,
                "y_std_outer_train": y_std,
                "input_numeric_dim": int(len(feature_bundle.numeric_columns)),
                "input_categorical_dim": int(len(feature_bundle.categorical_columns)),
                "raw_feature_dim_after_encoding": int(len(tr_report.raw_feature_names)),
                "feature_dim_after_drop": int(tr_report.output_dim),
                "dropped_feature_count": int(tr_report.dropped_low_std_count + len(tr_report.dropped_all_missing_numeric_columns)),
                "dropped_all_missing_numeric_columns": json_dump_compact(tr_report.dropped_all_missing_numeric_columns),
                "dropped_low_std_feature_names_sha256": sha256_obj(tr_report.dropped_low_std_feature_names),
                "categorical_levels_sha256": sha256_obj(tr_report.categorical_levels),
            })
            fit_rows.append({
                "project_id": project_id,
                "signature_name": signature,
                "fold_id": str(fold_id),
                "repeat_id": repeat_id if repeat_id is not None else pd.NA,
                "selected_alpha": float(selected_alpha),
                "n_outer_train": int(len(train)),
                "n_outer_test": int(len(test)),
                "feature_dim_after_drop": int(X_train.shape[1]),
                "dropped_feature_count": int(tr_report.dropped_low_std_count + len(tr_report.dropped_all_missing_numeric_columns)),
                "target_permutation": bool(target_permutation_seed is not None),
            })
            feature_hash = feature_bundle.feature_hash_by_project.get(project_id) or feature_bundle.feature_table_hash or ""
            pred_rows.extend(_prepare_prediction_rows(
                test, y_test_raw, y_pred,
                fold_id=fold_id,
                project_id=project_id,
                signature_name=signature,
                elig_row=elig,
                run_info=run_info,
                target_table_hash=target_hash,
                fold_assignment_hash=fold_hash,
                feature_file_hash=str(feature_hash),
                repeat_id=repeat_id,
            ))
        pred = pd.DataFrame(pred_rows)
        if pred.empty:
            raise RuntimeError(f"No out-of-fold predictions produced for {project_id} × {signature}")
        pred = ordered(pred, PREDICTION_REQUIRED_COLUMNS + (["repeat_id"] if repeat_id is not None else []))
        pred_path = predictions_dir / _prediction_filename(project_id, signature, repeat_id=repeat_id)
        write_parquet_atomic(pred_path, pred, overwrite=overwrite)
        predictions_by_cell[cell] = pred
        elig_by_cell[cell] = elig

    alpha_df = pd.DataFrame(alpha_rows)
    transform_df = pd.DataFrame(transform_rows)
    fit_df = pd.DataFrame(fit_rows)
    write_csv_atomic(metrics_dir / "alpha_selected.csv", alpha_df, overwrite=overwrite)
    write_csv_atomic(diagnostics_dir / "fold_feature_transform_report.csv", transform_df, overwrite=overwrite)
    cell_metrics, summary = _write_metrics(
        output_root=output_root,
        predictions_by_cell=predictions_by_cell,
        elig_by_cell=elig_by_cell,
        cfg=cfg,
        overwrite=overwrite,
        repeat_id=repeat_id,
    )
    # Number of folds whose selected alpha is at a grid edge
    grid_min = float(min(cfg.alpha_grid))
    grid_max = float(max(cfg.alpha_grid))
    if not fit_df.empty:
        sel_alpha = pd.to_numeric(fit_df["selected_alpha"], errors="coerce").to_numpy(dtype=float)
        n_fit = int(len(sel_alpha))
        n_at_max = int(np.sum(sel_alpha == grid_max))
        n_at_min = int(np.sum(sel_alpha == grid_min))
    else:
        n_fit = n_at_max = n_at_min = 0
    alpha_grid_boundary = {
        "grid_min": grid_min,
        "grid_max": grid_max,
        "n_fit_folds": n_fit,
        "n_selected_at_grid_max": n_at_max,
        "n_selected_at_grid_min": n_at_min,
        "fraction_at_grid_max": float(n_at_max / n_fit) if n_fit else 0.0,
        "fraction_at_grid_min": float(n_at_min / n_fit) if n_fit else 0.0,
    }
    model_fit_summary = {
        "created_at_utc": utc_now_iso(),
        "run_info": dict(run_info),
        "ridge_protocol": cfg.as_dict(),
        "ridge_protocol_hash": cfg.sha256,
        "target_permutation_seed": target_permutation_seed,
        "repeat_id": repeat_id,
        "n_cells_fit": int(len(predictions_by_cell)),
        "n_prediction_rows": int(sum(len(x) for x in predictions_by_cell.values())),
        "n_fit_folds": int(len(fit_df)),
        "alpha_grid_boundary": alpha_grid_boundary,
        "feature_source_metadata": dict(feature_bundle.metadata),
        "feature_hash_by_project": dict(feature_bundle.feature_hash_by_project),
        "target_hashes": target_hashes,
        "fold_assignment_hash": fold_hash,
        "benchmark_validation": validation,
        "fit_rows": fit_df.to_dict("records"),
    }
    write_json_atomic(diagnostics_dir / "model_fit_summary.json", model_fit_summary, overwrite=overwrite)
    return {
        "status": "ok",
        "output_root": str(output_root),
        "n_cells_fit": int(len(predictions_by_cell)),
        "n_prediction_rows": int(sum(len(x) for x in predictions_by_cell.values())),
        "primary_macro_spearman": summary.get("primary_macro_spearman"),
        "primary_n_cells": summary.get("primary_n_cells"),
        "ridge_protocol_hash": cfg.sha256,
        "output_hashes": collect_output_hashes(output_root),
    }


def evaluate_mean_predictor(
    *,
    output_root: Path,
    paths: BenchmarkPaths,
    run_info: Mapping[str, Any],
    cfg: RidgeProtocolConfig,
    overwrite: bool = False,
    allow_unvalidated: bool = False,
    progress_desc: Optional[str] = None,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    predictions_dir = output_root / "predictions"
    diagnostics_dir = output_root / "diagnostics"
    metrics_dir = output_root / "metrics"
    for d in [predictions_dir, diagnostics_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    validation = validation_gate(paths, allow_unvalidated=allow_unvalidated)
    final_case = load_final_case(paths)
    folds = load_folds(paths)
    diagnostics = load_case_tile_diagnostics(paths)
    case_meta = merge_case_metadata(final_case, diagnostics)
    eligibility = load_signature_eligibility(paths)
    fold_hash = sha256_path(paths.folds_dir / "fold_assignments.csv")
    target_hashes = target_hash_map(paths)
    base = final_case[["case_id", "case_uuid", "project_id"]].merge(folds[["case_id", "fold_id"]], on="case_id", how="left", validate="one_to_one")
    base = base.merge(case_meta.drop(columns=["case_uuid", "project_id"], errors="ignore"), on="case_id", how="left", validate="one_to_one")

    predictions_by_cell: Dict[Tuple[str, str], pd.DataFrame] = {}
    elig_by_cell: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    fit_rows: List[Dict[str, Any]] = []
    elig_records = eligibility.to_dict("records")
    elig_iter = tqdm(elig_records, desc=progress_desc, unit="sig") if progress_desc else elig_records
    for elig in elig_iter:
        project_id = str(elig["project_id"])
        signature = str(elig["signature_name"])
        target_table, target_path, target_hash, target_manifest = load_target_table(paths, project_id)
        if signature not in target_table.columns:
            raise RuntimeError(f"Signature {signature!r} not found in {target_path}")
        df = base.loc[base["project_id"].astype(str).eq(project_id)].merge(
            target_table[["case_id", signature]], on="case_id", how="inner", validate="one_to_one"
        ).rename(columns={signature: "__target__"})
        df["__target__"] = pd.to_numeric(df["__target__"], errors="coerce")
        df = df.loc[np.isfinite(df["__target__"].to_numpy(dtype=float))].copy()
        # Same target checks as in evaluate_ridge_feature_bundle
        if df.empty:
            raise RuntimeError(f"No finite targets for {project_id} × {signature}")
        if _is_constant(df["__target__"].to_numpy(dtype=float)):
            raise RuntimeError(f"Eligible target is constant for {project_id} × {signature}")
        pred_rows: List[Dict[str, Any]] = []
        for fold_id in sorted(df["fold_id"].astype(str).unique(), key=lambda x: (int(x) if str(x).isdigit() else str(x))):
            train = df.loc[~df["fold_id"].astype(str).eq(str(fold_id))].copy()
            test = df.loc[df["fold_id"].astype(str).eq(str(fold_id))].copy()
            if not _fold_is_viable(train, test, project_id=project_id, signature=signature, fold_id=fold_id):
                continue
            y_mean = float(np.mean(train["__target__"].to_numpy(dtype=float)))
            y_test = test["__target__"].to_numpy(dtype=float)
            y_pred = np.full(len(test), y_mean, dtype=float)
            fit_rows.append({"project_id": project_id, "signature_name": signature, "fold_id": str(fold_id), "outer_train_mean": y_mean, "n_outer_train": int(len(train)), "n_outer_test": int(len(test))})
            pred_rows.extend(_prepare_prediction_rows(
                test, y_test, y_pred,
                fold_id=fold_id,
                project_id=project_id,
                signature_name=signature,
                elig_row=elig,
                run_info=run_info,
                target_table_hash=target_hash,
                fold_assignment_hash=fold_hash,
                feature_file_hash="mean_predictor_no_features",
            ))
        pred = pd.DataFrame(pred_rows)
        if pred.empty:
            raise RuntimeError(f"No out-of-fold predictions produced for {project_id} × {signature}")
        pred = ordered(pred, PREDICTION_REQUIRED_COLUMNS)
        write_parquet_atomic(predictions_dir / _prediction_filename(project_id, signature), pred, overwrite=overwrite)
        predictions_by_cell[(project_id, signature)] = pred
        elig_by_cell[(project_id, signature)] = elig

    # Empty alpha file (no alpha for the mean predictor); correlations are NA when predictions are constant
    write_csv_atomic(metrics_dir / "alpha_selected.csv", pd.DataFrame(columns=["project_id", "signature_name", "fold_id", "selected_alpha", "note"]), overwrite=overwrite)
    _, summary = _write_metrics(
        output_root=output_root,
        predictions_by_cell=predictions_by_cell,
        elig_by_cell=elig_by_cell,
        cfg=cfg,
        overwrite=overwrite,
    )
    model_fit_summary = {
        "created_at_utc": utc_now_iso(),
        "baseline_id": run_info.get("baseline_id", "mean_predictor"),
        "protocol": "outer_train_mean_on_frozen_target_scale",
        "correlation_metrics_forced_na": False,
        "correlation_policy": "computed_when_fold_specific_mean_predictions_are_not_constant",
        "ridge_protocol_reference": cfg.as_dict(),
        "ridge_protocol_hash": cfg.sha256,
        "target_hashes": target_hashes,
        "fold_assignment_hash": fold_hash,
        "benchmark_validation": validation,
        "fit_rows": fit_rows,
    }
    write_json_atomic(diagnostics_dir / "model_fit_summary.json", model_fit_summary, overwrite=overwrite)
    return {
        "status": "ok",
        "output_root": str(output_root),
        "n_cells_fit": int(len(predictions_by_cell)),
        "primary_macro_spearman": summary.get("primary_macro_spearman"),
        "ridge_protocol_hash": cfg.sha256,
        "output_hashes": collect_output_hashes(output_root),
    }
