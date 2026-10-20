"""Per-fold fit of the Task A.3 sklearn probe (multinomial logistic regression) and out-of-fold predictions."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# sys.path setup (as in every src_A3 module)
from src_A3.common import ensure_imports

ensure_imports()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src_A3.common import (  # noqa: E402
    A3_DEFAULT_PROTOCOL_SEED,
    PhaseTimer,
    head_type_from_config,
    mkdir,
    resolve_he_embeddings_dir,
    save_json,
    set_global_seeds,
)
from src_A3.data_loading import (  # noqa: E402
    build_split_specific_cell_embedding_index,
    centroid_parquet_path,
    load_cell_embeddings_via_main_assignment,
    load_scconcept_cell_embeddings,
    load_split_specific_cell_embeddings,
    resolve_main_assign_path,
    scconcept_slide_h5_path,
    slide_patch_h5_path,
)
from src_A3.labels import LabelSpace, map_cell_types_to_ids  # noqa: E402
from src_A3.metrics import classification_metrics_from_ids  # noqa: E402
from src_A3.modeling import (  # noqa: E402
    align_proba_to_global,
    _build_logistic_regression,
    build_probe_pipeline,
    build_probe_preprocessor,
    fit_predict_proba,
    fit_predict_proba_preprocessed,
)

# Task A.1 helpers: 1-SE selector and ridge capacity readout (used with lam = 1/C)
from src_A1.training import select_lambda_1se  # noqa: E402
from src_A1.modeling import boundary_saturation, effective_dof, numerical_rank  # noqa: E402

logger = logging.getLogger(__name__)

# Required columns of the scored row manifest
_REQUIRED_ROW_COLUMNS = ("row_key", "cell_id_int", "cell_type", "x", "y")

# Capacity readout: max train cells of the Gram subsample, floor of the mean Fisher weight wbar
_CAPACITY_MAX_GRAM_CELLS = 100_000
_WBAR_FLOOR = 1e-6


@dataclass
class A3SlideTensor:
    """Per-slide tensors for Task A.3 (cell level): embeddings X and dense class ids y."""

    slide_id: str
    subset: str
    source: str
    tissue: str
    X: np.ndarray            # [N, D] float32 embeddings (raw; geometry applied in the probe)
    y: np.ndarray            # [N] int32 dense global class ids (>= 0; scored rows only)
    row_keys: np.ndarray     # [N] object, globally unique across folds
    coords_xy: np.ndarray    # [N, 2] float32 nucleus centroids
    cell_types: np.ndarray   # [N] object canonical cell-type strings (diagnostics / coarse-agg)
    obs_df: Optional[pd.DataFrame] = None  # [N rows] or None


@dataclass
class A3FoldData:
    """One fold's train / val / test slide tensors and the global class names (val_slides is empty here)."""

    fold_name: str
    train_slides: List[A3SlideTensor]
    val_slides: List[A3SlideTensor]
    test_slides: List[A3SlideTensor]
    class_names: np.ndarray  # [C] object == global label_space.labels


def slide_tensor_from_rows_a3(
    *,
    cfg: Mapping,
    eval_dir: Path,
    slide_record,
    rows_sub_df: pd.DataFrame,
    space: str,
    training_num: str,
    checkpoint_num: str,
    label_space: LabelSpace,
    within_split_allowed_patch_ids: Optional[Sequence[str]] = None,
    slide_rows_all_df: Optional[pd.DataFrame] = None,
) -> A3SlideTensor:
    """
    Build the A3SlideTensor of slide_record restricted to rows_sub_df. X: coordinates, scConcept embeddings, or H&E
    cell embeddings (main assignment, or best patch among within_split_allowed_patch_ids for STHELAR_TEST_ADAPTED);
    y: dense class ids (raises on unscored rows). slide_rows_all_df is unused.
    """
    level = str(cfg["protocol"]["level"])
    if level != "cell":
        raise ValueError(f"Task A.3 supports only level='cell'; got level={level!r}.")

    if rows_sub_df is None or rows_sub_df.empty:
        raise RuntimeError(
            f"Cannot materialize slide tensor for slide '{slide_record.slide_id}': the requested "
            "row subset is empty. This indicates an inconsistent split, area stripe, or frozen "
            "row manifest."
        )

    missing_cols = [c for c in _REQUIRED_ROW_COLUMNS if c not in rows_sub_df.columns]
    if missing_cols:
        raise KeyError(
            f"rows_sub_df for slide '{slide_record.slide_id}' is missing required column(s) "
            f"{missing_cols}; got {list(rows_sub_df.columns)}."
        )

    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    dataset_id = str(cfg["identity"]["dataset_id"])

    cell_ids = rows_sub_df["cell_id_int"].to_numpy(dtype=np.int32)
    row_keys = rows_sub_df["row_key"].astype(str).to_numpy(dtype=object)

    space_norm = str(space).strip().lower()
    if space_norm == "coords":
        # Coordinates arm: nucleus (x, y) centroid
        X = rows_sub_df[["x", "y"]].to_numpy(dtype=np.float32)
    elif space_norm == "scconcept":
        # scConcept arm: per-cell scConcept embedding (cls_emb)
        scc_tid = str(cfg["protocol"].get("scconcept_training_id") or "").strip()
        if not scc_tid:
            raise ValueError(
                "space='scconcept' requires protocol.scconcept_training_id in the frozen eval_config; "
                "set it via `_1_setup.py --scconcept_training_id ...` before freezing this eval_id."
            )
        scc_h5 = scconcept_slide_h5_path(scc_tid, slide_record.slide_id)
        if not scc_h5.exists():
            raise FileNotFoundError(
                f"scConcept ceiling arm: missing per-cell embedding file for slide "
                f"'{slide_record.slide_id}': {scc_h5}. Build scConcept embeddings for every eval "
                "slide, or omit the scconcept_ceiling arm."
            )
        X = load_scconcept_cell_embeddings(scc_h5, cell_ids)
    else:
        # H&E encoder arm (he / he_l2 / clip): the precomputed per-cell H&E embedding.
        emb_dir = resolve_he_embeddings_dir(training_dataset_id, training_num, checkpoint_num)
        emb_h5 = emb_dir / f"{slide_record.slide_id}.h5"
        if within_split_allowed_patch_ids is None:
            # Default cell path: the precomputed "main" H&E embedding per cell.
            main_assign = resolve_main_assign_path(emb_dir, slide_record.slide_id)
            X = load_cell_embeddings_via_main_assignment(emb_h5, main_assign, cell_ids, space)
        else:
            # Within-slide (STHELAR_TEST_ADAPTED) path: re-select the best patch-cell row per cell,
            # restricted to the patches inside the allowed stripe.
            slide_patch_h5 = slide_patch_h5_path(training_dataset_id, slide_record.slide_id)
            centroid_parquet = centroid_parquet_path(
                dataset_id, training_dataset_id, slide_record.slide_id
            )
            split_index_df = build_split_specific_cell_embedding_index(
                emb_h5_path=emb_h5,
                allowed_patch_ids=within_split_allowed_patch_ids,
                slide_patch_h5=slide_patch_h5,
                centroid_parquet=centroid_parquet,
            )
            split_index_df = split_index_df.loc[split_index_df["cell_id_int"].isin(cell_ids)].copy()
            X = load_split_specific_cell_embeddings(
                emb_h5_path=emb_h5,
                split_index_df=split_index_df,
                ordered_cell_ids=cell_ids,
                space=space,
            )

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise RuntimeError(
            f"Cell embeddings for slide '{slide_record.slide_id}' must be 2-D; got shape {X.shape}."
        )
    if not np.isfinite(X).all():
        bad_rows = np.flatnonzero(~np.isfinite(X).all(axis=1))
        raise RuntimeError(
            f"Non-finite embeddings detected for slide '{slide_record.slide_id}'. "
            f"X.shape={X.shape}, n_bad_rows={bad_rows.size}, "
            f"first_bad_rows={bad_rows[:10].tolist()}"
        )

    cell_types = rows_sub_df["cell_type"].astype(str).to_numpy(dtype=object)
    y = map_cell_types_to_ids(cell_types, label_space)
    if not (y >= 0).all():
        bad = np.flatnonzero(y < 0)
        offending = sorted(set(np.asarray(cell_types)[bad].tolist()))[:10]
        raise RuntimeError(
            f"Unscored cell(s) leaked into the frozen scored rows for slide "
            f"'{slide_record.slide_id}': {bad.size} row(s) mapped to class id -1 under the frozen "
            f"{label_space.granularity} LabelSpace. Offending cell_type value(s): {offending}. "
            "The scored row manifest must contain only cells whose (coarse-mapped) canonical "
            "cell_type is in the frozen class vocabulary."
        )

    if not (X.shape[0] == y.shape[0] == row_keys.shape[0] == rows_sub_df.shape[0]):
        raise RuntimeError(
            f"Row-count mismatch for slide '{slide_record.slide_id}': "
            f"X={X.shape[0]}, y={y.shape[0]}, row_keys={row_keys.shape[0]}, "
            f"rows={rows_sub_df.shape[0]}."
        )

    coords_xy = rows_sub_df[["x", "y"]].to_numpy(dtype=np.float32)
    obs_df = rows_sub_df.reset_index(drop=True).copy()

    return A3SlideTensor(
        slide_id=str(slide_record.slide_id),
        subset=str(slide_record.subset),
        source=str(slide_record.source),
        tissue=str(slide_record.tissue),
        X=X,
        y=y.astype(np.int32, copy=False),
        row_keys=row_keys,
        coords_xy=coords_xy,
        cell_types=cell_types,
        obs_df=obs_df,
    )


def _stack_train_features(fold_data: A3FoldData) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate the fold's train-slide features / labels into `(X_train, y_train)`.

    Asserts a consistent feature dimension across train slides and a non-empty train pool.
    """
    if not fold_data.train_slides:
        raise RuntimeError(f"Fold '{fold_data.fold_name}' has no train slides.")

    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    expected_dim: Optional[int] = None
    for slide in fold_data.train_slides:
        Xs = np.asarray(slide.X, dtype=np.float32)
        ys = np.asarray(slide.y, dtype=np.int32)
        if Xs.ndim != 2:
            raise RuntimeError(
                f"Train slide '{slide.slide_id}' X must be 2-D; got shape {Xs.shape}."
            )
        if Xs.shape[0] != ys.shape[0]:
            raise RuntimeError(
                f"Train slide '{slide.slide_id}' row mismatch: X={Xs.shape[0]}, y={ys.shape[0]}."
            )
        if expected_dim is None:
            expected_dim = int(Xs.shape[1])
        elif int(Xs.shape[1]) != expected_dim:
            raise RuntimeError(
                f"All train slides in fold '{fold_data.fold_name}' must share the feature "
                f"dimension. Expected {expected_dim}, got {Xs.shape[1]} for slide "
                f"'{slide.slide_id}'."
            )
        x_parts.append(Xs)
        y_parts.append(ys)

    X_train = np.vstack(x_parts).astype(np.float32, copy=False)
    y_train = np.concatenate(y_parts).astype(np.int32, copy=False)
    if X_train.shape[0] == 0:
        raise RuntimeError(f"Fold '{fold_data.fold_name}' has zero training rows.")
    return X_train, y_train


def _c_grid_from_config(protocol_cfg: Mapping, space: Optional[str] = None) -> np.ndarray:
    """
    Log-spaced C grid of protocol.probe.C_grid {c_min, c_max, n_points}, or of
    protocol.probe.C_grid_overrides[space] if set (space=None: base grid).
    """
    probe = protocol_cfg.get("probe", {}) or {}
    g = dict(probe.get("C_grid", {}) or {})
    if space is not None:
        overrides = probe.get("C_grid_overrides", {}) or {}
        override = overrides.get(str(space).strip().lower())
        if override:
            g = dict(override)
    lo = float(g.get("c_min", 1e-6))
    hi = float(g.get("c_max", 1e0))
    n = int(g.get("n_points", 7))
    if not (0.0 < lo < hi):
        raise ValueError(f"protocol.probe.C_grid requires 0 < c_min < c_max, got {lo!r} and {hi!r}.")
    if n < 2:
        raise ValueError(f"protocol.probe.C_grid.n_points must be >= 2, got {n!r}.")
    return np.logspace(np.log10(lo), np.log10(hi), int(n))


def _partition_groups_into_blocks(
    groups: Sequence[A3SlideTensor], max_blocks: int
) -> List[List[int]]:
    """
    Partition the train groups into at most max_blocks disjoint blocks balanced in cell count (groups sorted by
    decreasing size, dealt in serpentine order). One block per group if max_blocks >= len(groups).
    """
    n_groups = len(groups)
    if n_groups == 0:
        raise ValueError("_partition_groups_into_blocks requires at least one group.")
    n_blocks = int(max_blocks) if max_blocks and int(max_blocks) > 0 else n_groups
    n_blocks = max(1, min(n_blocks, n_groups))
    if n_blocks >= n_groups:
        return [[i] for i in range(n_groups)]

    order = sorted(
        range(n_groups),
        key=lambda i: (-int(np.asarray(groups[i].y).shape[0]), str(groups[i].slide_id)),
    )
    blocks: List[List[int]] = [[] for _ in range(n_blocks)]
    for rank, gi in enumerate(order):
        pos, sweep = rank % n_blocks, rank // n_blocks
        blocks[pos if sweep % 2 == 0 else n_blocks - 1 - pos].append(gi)
    return [sorted(b) for b in blocks if b]


def _stack_groups_subsampled(
    groups: Sequence[A3SlideTensor],
    keep_idx: Sequence[int],
    *,
    max_cells: int,
    rng,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    np.vstack([groups[j].X for j in keep_idx])[sub] (and y) for a random subsample of max_cells rows, without
    building the full stack. No subsampling if max_cells is falsy or the pool is not larger.
    """
    if not keep_idx:
        raise ValueError("_stack_groups_subsampled requires at least one group to stack.")

    sizes = [int(np.asarray(groups[j].X).shape[0]) for j in keep_idx]
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    n_total = int(offsets[-1])

    if not max_cells or n_total <= int(max_cells):
        X = np.vstack([np.asarray(groups[j].X, dtype=np.float32) for j in keep_idx])
        y = np.concatenate([np.asarray(groups[j].y, dtype=np.int32) for j in keep_idx])
        return X, y

    sub = rng.choice(n_total, size=int(max_cells), replace=False)
    # Group owning each sampled row (empty groups own no rows)
    owner = np.searchsorted(offsets, sub, side="right") - 1
    local = sub - offsets[owner]

    dim = int(np.asarray(groups[keep_idx[0]].X).shape[1])
    X = np.empty((int(sub.size), dim), dtype=np.float32)
    y = np.empty(int(sub.size), dtype=np.int32)
    for k, j in enumerate(keep_idx):
        mask = owner == k
        if not mask.any():
            continue
        rows = local[mask]
        X[mask] = np.asarray(groups[j].X, dtype=np.float32)[rows]
        y[mask] = np.asarray(groups[j].y, dtype=np.int32)[rows]
    return X, y


def _select_probe_C(
    *,
    cfg: Mapping,
    space: str,
    groups: Sequence[A3SlideTensor],
    seed: int,
    label_order: Sequence[str],
    protocol_cfg: Mapping,
    timer: Optional[PhaseTimer] = None,
) -> Dict[str, Any]:
    """
    Select C per (encoder, fold) by leave-one-train-block-out CV (macro-F1 on each held-out block, 1-SE rule of
    select_lambda_1se on lam = 1/C; refits subsampled to inner_cv_max_train_cells). Falls back to
    protocol.probe.C with < 2 groups or no finite score.
    """
    probe = dict(protocol_cfg.get("probe", {}) or {})
    fixed_C = float(probe.get("C", 1.0))
    score_key = str(probe.get("inner_cv_score", "macro_f1")).strip().lower()
    if score_key not in {"macro_f1", "balanced_accuracy"}:
        raise ValueError(
            f"protocol.probe.inner_cv_score={score_key!r} must be 'macro_f1' or 'balanced_accuracy'."
        )
    C_grid = _c_grid_from_config(protocol_cfg, space)
    inner_tol = float(probe.get("inner_cv_tol", probe.get("tol", 1e-4)))
    n_groups = len(groups)
    fallback = {
        "selected_C": fixed_C,
        "lambda_star": float(1.0 / fixed_C),
        "C_grid": [float(c) for c in C_grid],
        "n_groups": int(n_groups),
        "n_blocks": 0,
        "blocks": None,
        "inner_cv_tol": inner_tol,
        "inner_cv_train_cells_per_fit": 0,
        "inner_cv_macro_f1_at_selected": float("nan"),
    }
    if n_groups < 2:
        fallback["status"] = "fallback_insufficient_groups"
        return fallback

    max_cells = int(probe.get("inner_cv_max_train_cells", 50000))
    blocks = _partition_groups_into_blocks(groups, int(probe.get("inner_cv_max_groups", 5) or 0))
    n_blocks = len(blocks)
    # Inner-CV subsample RNG seeded by (probe seed, protocol seed)
    protocol_seed = int(protocol_cfg.get("protocol_seed", A3_DEFAULT_PROTOCOL_SEED))
    rng = np.random.default_rng(np.asarray([int(seed), protocol_seed], dtype=np.int64))
    scores = np.full((len(C_grid), n_blocks), np.nan, dtype=np.float64)
    timer = PhaseTimer() if timer is None else timer
    inner_train_cells: List[int] = []

    for bi, block in enumerate(blocks):
        held_out = set(block)
        keep_idx = [j for j in range(n_groups) if j not in held_out]
        if not keep_idx:
            continue  # degenerate (single block): nothing to train on, leave the column NaN
        with timer.phase("inner_cv_stack"):
            X_tr, y_tr = _stack_groups_subsampled(
                groups, keep_idx, max_cells=max_cells, rng=rng
            )
            # Held-out block (not subsampled)
            X_val, y_val = _stack_groups_subsampled(groups, list(block), max_cells=0, rng=rng)
        inner_train_cells.append(int(X_tr.shape[0]))
        if np.unique(y_tr).size < 2:
            continue  # leave this block's column NaN; select_lambda_1se tolerates it

        # Preprocessing does not depend on C: fitted once per held-out block (a failure leaves the column NaN)
        try:
            with timer.phase("inner_cv_preprocess"):
                prep = build_probe_preprocessor(cfg, space)
                Z_tr = prep.fit_transform(X_tr)
                Z_val = prep.transform(X_val)
        except Exception as exc:
            logger.warning("inner-CV preprocessing failed (block %d): %s", bi, exc)
            continue

        for ci, C in enumerate(C_grid):
            # Selection refits use inner_cv_tol
            clf = _build_logistic_regression(
                protocol_cfg, seed=int(seed), C_override=float(C), tol_override=inner_tol
            )
            try:
                with timer.phase("inner_cv_fit"):
                    proba_list, clf_classes, _conv = fit_predict_proba_preprocessed(
                        clf, Z_tr, y_tr, [Z_val]
                    )
            except Exception as exc:  # a failed refit leaves the score NaN
                logger.warning("inner-CV refit failed (block %d, C=%.3g): %s", bi, float(C), exc)
                continue
            y_pred = np.asarray(clf_classes)[np.argmax(proba_list[0], axis=1)].astype(np.int32)
            with timer.phase("inner_cv_score"):
                summ = classification_metrics_from_ids(
                    y_val, y_pred, label_order, gt_present_only_for_macro=True
                )
            scores[ci, bi] = float(summ.metrics[score_key])

    # Mean inner-CV train size
    mean_inner_cells = int(round(float(np.mean(inner_train_cells)))) if inner_train_cells else 0
    block_provenance = {
        "n_blocks": int(n_blocks),
        "blocks": [[str(groups[j].slide_id) for j in block] for block in blocks],
        "inner_cv_tol": inner_tol,
        "inner_cv_train_cells_per_fit": mean_inner_cells,
    }

    if not np.isfinite(scores).any():
        fallback.update(block_provenance)
        fallback["status"] = "fallback_no_finite_cv_score"
        return fallback

    lam_grid = 1.0 / C_grid
    order = np.argsort(lam_grid)
    lam_sorted = lam_grid[order]
    scores_sorted = scores[order, :]
    lam_star, _mean_deficit, _se_deficit = select_lambda_1se(scores_sorted, lam_sorted)
    C_star = float(1.0 / lam_star)
    star_row = int(np.argmin(np.abs(lam_sorted - lam_star)))
    with np.errstate(invalid="ignore"):
        inner_at_sel = float(np.nanmean(scores_sorted[star_row]))
    return {
        "selected_C": C_star,
        "lambda_star": float(lam_star),
        "C_grid": [float(c) for c in C_grid],
        # n_groups: train groups (slides / stripes); n_blocks: held-out units of the C search
        "n_groups": int(n_groups),
        **block_provenance,
        "inner_cv_macro_f1_at_selected": inner_at_sel,
        "status": "selected",
    }


def _probe_capacity_readout(
    *,
    pipeline,
    X_train: np.ndarray,
    selected_C: float,
    lam_star: float,
    lam_grid_sorted: Sequence[float],
    rng,
) -> Dict[str, Any]:
    """
    Realized probe capacity: effective_dof of the standardized train Gram (subsample) at alpha_eff = 1/(C*·wbar),
    wbar = mean p(1-p), with numerical_rank and the grid-edge kind (boundary_saturation). NaN on failure.
    """
    out: Dict[str, Any] = {
        "effective_dof": float("nan"),
        "alpha_eff": float("nan"),
        "numerical_rank": -1,
        "lambda_boundary_kind": "unavailable",
        "wbar": float("nan"),
        "n_cells_for_capacity": 0,
        "capacity_status": "unavailable",
    }
    try:
        clf = pipeline.named_steps["classifier"]
        # Subsample the rows before the transform (row-wise steps, same result)
        X_cap = np.asarray(X_train)
        n_rows = int(X_cap.shape[0])
        if n_rows > _CAPACITY_MAX_GRAM_CELLS:
            sub = rng.choice(n_rows, size=_CAPACITY_MAX_GRAM_CELLS, replace=False)
            X_cap = X_cap[sub]
        Z = np.asarray(pipeline[:-1].transform(X_cap), dtype=np.float64)
        P = np.asarray(clf.predict_proba(Z), dtype=np.float64)
        wbar = max(float(np.mean(P * (1.0 - P))), _WBAR_FLOOR)
        evals = np.clip(np.linalg.eigvalsh(Z.T @ Z).astype(np.float64), 0.0, None)
        alpha_eff = 1.0 / (float(selected_C) * wbar)
        _is_edge, kind = boundary_saturation(
            evals=evals, alpha=alpha_eff, lam=float(lam_star),
            lam_grid=[float(x) for x in lam_grid_sorted],
        )
        out.update({
            "effective_dof": float(effective_dof(evals, alpha_eff)),
            "alpha_eff": float(alpha_eff),
            "numerical_rank": int(numerical_rank(evals)),
            "lambda_boundary_kind": str(kind),
            "wbar": float(wbar),
            "n_cells_for_capacity": int(Z.shape[0]),
            "capacity_status": "ok",
        })
    except Exception as exc:
        logger.warning("probe capacity readout failed: %s", exc)
        out["capacity_status"] = f"error:{type(exc).__name__}"
    return out


def fit_predict_one_fold(
    *,
    cfg: Mapping,
    fold_data: A3FoldData,
    space: str,
    seed: int,
    fold_index: int,
    label_space: LabelSpace,
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Fit the sklearn probe on one fold (C selected by inner CV or fixed) and predict each test slide on the global
    class axis. Writes model_manifest.json. Returns (artifacts, val_preds={}, test_preds={slide_id -> bundle}).
    """
    set_global_seeds(int(seed))
    mkdir(output_dir)
    timer = PhaseTimer()

    head_type = head_type_from_config(cfg)
    if head_type == "mlp_softmax":
        raise NotImplementedError(
            "protocol.head.type='mlp_softmax' is not implemented. Only 'linear_softmax' "
            "(sklearn multinomial LogisticRegression) is implemented."
        )
    if head_type != "linear_softmax":
        raise ValueError(
            f"Unsupported protocol.head.type={head_type!r} on the sklearn fold-fit path; "
            "expected 'linear_softmax'."
        )

    if not fold_data.test_slides:
        raise RuntimeError(f"Fold '{fold_data.fold_name}' has no test slides.")

    # The fold must use the global class vocabulary
    fold_class_names = np.asarray(fold_data.class_names, dtype=object)
    global_labels = np.asarray(list(label_space.labels), dtype=object)
    if not np.array_equal(fold_class_names, global_labels):
        raise RuntimeError(
            f"Fold '{fold_data.fold_name}' class_names do not match the frozen global LabelSpace. "
            f"fold={fold_class_names.tolist()} vs label_space={global_labels.tolist()}."
        )

    with timer.phase("stack_train"):
        X_train, y_train = _stack_train_features(fold_data)
    input_dim = int(X_train.shape[1])
    n_train = int(y_train.shape[0])

    train_class_ids = np.unique(y_train)
    train_class_set = set(int(c) for c in train_class_ids.tolist())
    if len(train_class_set) < 2:
        raise RuntimeError(
            f"Fold '{fold_data.fold_name}' train set contains fewer than 2 distinct classes "
            f"(present ids: {sorted(train_class_set)}); a multinomial LogisticRegression needs "
            ">= 2. Inspect the fold-class coverage / class-support thresholds."
        )

    # C selected per (encoder, fold) by inner CV, or fixed C (inner_cv_select False)
    protocol_cfg = cfg.get("protocol", cfg)
    inner_cv = bool((protocol_cfg.get("probe", {}) or {}).get("inner_cv_select", False))
    cap_rng = np.random.default_rng(np.asarray([int(seed), int(fold_index)], dtype=np.int64))
    if inner_cv:
        with timer.phase("select_C"):
            sel = _select_probe_C(
                cfg=cfg,
                space=space,
                groups=list(fold_data.train_slides),
                seed=int(seed),
                label_order=list(label_space.labels),
                protocol_cfg=protocol_cfg,
                timer=timer,
            )
    else:
        fixed_C = float((protocol_cfg.get("probe", {}) or {}).get("C", 1.0))
        sel = {
            "selected_C": fixed_C,
            "lambda_star": float(1.0 / fixed_C),
            "C_grid": None,
            "n_groups": len(fold_data.train_slides),
            "n_blocks": 0,
            "blocks": None,
            "inner_cv_tol": float("nan"),
            "inner_cv_train_cells_per_fit": 0,
            "inner_cv_macro_f1_at_selected": float("nan"),
            "status": "fixed",
        }
    selected_C = float(sel["selected_C"])

    # Final-fit train cells / inner-CV train cells per fit
    inner_cells = int(sel.get("inner_cv_train_cells_per_fit", 0) or 0)
    inner_cv_train_size_ratio = float(n_train / inner_cells) if inner_cells > 0 else float("nan")

    pipeline = build_probe_pipeline(cfg, space, seed=int(seed), C_override=selected_C)

    # Fit once on the train fold and predict every test slide
    X_test_list = [np.asarray(slide.X, dtype=np.float32) for slide in fold_data.test_slides]
    with timer.phase("final_fit"):
        proba_list, clf_classes, converged = fit_predict_proba(
            pipeline, X_train, y_train, X_test_list
        )

    standardize_step = pipeline.named_steps.get("standardize")
    n_features_kept = int(getattr(standardize_step, "n_features_out_", -1))
    classes_trained = [int(c) for c in np.asarray(clf_classes).tolist()]

    # Realized capacity at the selected C
    lam_grid_sorted = (
        sorted(1.0 / float(c) for c in sel["C_grid"])
        if sel.get("C_grid")
        else [float(sel["lambda_star"])]
    )
    with timer.phase("capacity_readout"):
        capacity = _probe_capacity_readout(
            pipeline=pipeline,
            X_train=X_train,
            selected_C=selected_C,
            lam_star=float(sel["lambda_star"]),
            lam_grid_sorted=lam_grid_sorted,
            rng=cap_rng,
        )
    # Warn (no error) when the selected C is at a grid edge
    if capacity["lambda_boundary_kind"] in {"ols", "zero_capacity", "truncated"}:
        logger.warning(
            "Fold '%s' (space=%s): selected C=%.4g sits at a grid edge of protocol.probe.C_grid "
            "(boundary=%s, effective_dof=%.2f); recorded in model_manifest.json.",
            fold_data.fold_name,
            space,
            selected_C,
            capacity["lambda_boundary_kind"],
            capacity["effective_dof"],
        )

    _t_bundles = time.perf_counter()
    test_preds: Dict[str, Any] = {}
    for i, slide in enumerate(fold_data.test_slides):
        y_proba = align_proba_to_global(proba_list[i], clf_classes, label_space)
        y_true = np.asarray(slide.y, dtype=np.int32)
        # true_class_covered: true class present in the fold's train set
        true_class_covered = np.isin(y_true, train_class_ids)
        if y_proba.shape[0] != y_true.shape[0]:
            raise RuntimeError(
                f"Prediction row mismatch for slide '{slide.slide_id}' in fold "
                f"'{fold_data.fold_name}': y_proba={y_proba.shape[0]}, y_true={y_true.shape[0]}."
            )
        test_preds[slide.slide_id] = {
            "slide_id": str(slide.slide_id),
            "source": str(slide.source),
            "subset": str(slide.subset),
            "tissue": str(slide.tissue),
            "row_keys": np.asarray(slide.row_keys, dtype=object),
            "coords_xy": np.asarray(slide.coords_xy, dtype=np.float32),
            "class_names": global_labels,
            "y_true": y_true,
            "y_proba": np.asarray(y_proba, dtype=np.float32),
            "true_class_covered": np.asarray(true_class_covered, dtype=bool),
            "cell_types": (None if slide.cell_types is None else np.asarray(slide.cell_types, dtype=object)),
            "obs_df": (slide.obs_df.copy() if slide.obs_df is not None else None),
            # Scalar fold name and per-row fold names (kept per cell by the ADAPTED merge)
            "fold_name": str(fold_data.fold_name),
            "fold_names": np.full(int(y_true.shape[0]), str(fold_data.fold_name), dtype=object),
        }
    timer.add("build_bundles", time.perf_counter() - _t_bundles)
    timer.add("fold_total", timer.seconds("stack_train") + timer.seconds("select_C")
              + timer.seconds("final_fit") + timer.seconds("capacity_readout")
              + timer.seconds("build_bundles"))

    save_json(
        output_dir / "model_manifest.json",
        {
            "fold_name": str(fold_data.fold_name),
            "mode": "sklearn_logreg",
            "head_type": head_type,
            "space": str(space),
            "seed": int(seed),
            "fold_index": int(fold_index),
            "input_dim": input_dim,
            "n_train": n_train,
            "n_classes_global": int(label_space.n_classes()),
            "n_features_kept": n_features_kept,
            "converged": bool(converged),
            "classes_trained": classes_trained,
            "classes_trained_labels": [
                str(label_space.labels[c]) for c in classes_trained
            ],
            "n_test_slides": len(fold_data.test_slides),
            # Selected C and realized capacity
            "probe_capacity": {
                "selected_C": selected_C,
                "C_selection_status": str(sel["status"]),
                "C_grid": sel.get("C_grid"),
                "n_inner_cv_groups": int(sel.get("n_groups", 0)),
                # Held-out blocks of the C search (train slides / stripes of each block)
                "n_inner_cv_blocks": int(sel.get("n_blocks", 0)),
                "inner_cv_blocks": sel.get("blocks"),
                "inner_cv_tol": sel.get("inner_cv_tol"),
                # Inner-CV vs final-fit train size
                "inner_cv_train_cells_per_fit": inner_cells,
                "deliverable_train_cells": n_train,
                "inner_cv_train_size_ratio": inner_cv_train_size_ratio,
                "inner_cv_macro_f1_at_selected": sel.get("inner_cv_macro_f1_at_selected"),
                "effective_dof": capacity["effective_dof"],
                "alpha_eff": capacity["alpha_eff"],
                "wbar": capacity["wbar"],
                "numerical_rank": capacity["numerical_rank"],
                "lambda_boundary_kind": capacity["lambda_boundary_kind"],
                "capacity_status": capacity["capacity_status"],
                "n_cells_for_capacity": capacity["n_cells_for_capacity"],
            },
            # Timings of this fold (select_C includes the inner_cv_* phases) and phase call counts
            "timing": {
                **timer.as_dict(),
                "phase_counts": timer.counts(),
            },
        },
    )

    if not converged:
        logger.warning(
            "Fold '%s' (seed=%s) probe did not converge; recorded in model_manifest.json.",
            fold_data.fold_name,
            seed,
        )

    artifacts: Dict[str, Any] = {
        "converged": bool(converged),
        "n_train": n_train,
        "train_history": [],
        "val_history": [],
        "best_step": 0,
        "n_features_kept": n_features_kept,
        "classes_trained": classes_trained,
        # Capacity fields (copied to fold_metrics.csv)
        "selected_C": selected_C,
        "C_selection_status": str(sel["status"]),
        "effective_dof": capacity["effective_dof"],
        "alpha_eff": capacity["alpha_eff"],
        "numerical_rank": capacity["numerical_rank"],
        "inner_cv_macro_f1_at_selected": sel.get("inner_cv_macro_f1_at_selected"),
        "n_inner_cv_groups": int(sel.get("n_groups", 0)),
        # n_inner_cv_blocks x len(C_grid) = number of inner-CV fits
        "n_inner_cv_blocks": int(sel.get("n_blocks", 0)),
        "inner_cv_train_cells_per_fit": inner_cells,
        "inner_cv_train_size_ratio": inner_cv_train_size_ratio,
        "lambda_boundary_kind": capacity["lambda_boundary_kind"],
        "probe_capacity_status": capacity["capacity_status"],
        # Timings (*_seconds; select_C_seconds includes the inner_cv_* phases) and inner-CV fit count
        **timer.as_dict(),
        "n_inner_cv_fits": int(timer.counts().get("inner_cv_fit", 0)),
    }
    val_preds: Dict[str, Any] = {}
    return artifacts, val_preds, test_preds


__all__ = [
    "A3SlideTensor",
    "A3FoldData",
    "slide_tensor_from_rows_a3",
    "fit_predict_one_fold",
]
