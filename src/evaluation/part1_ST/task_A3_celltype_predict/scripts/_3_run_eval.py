#!/usr/bin/env python
"""
Run the Task A.3 evaluation (cell-type linear-probe classification) of one arm (H&E encoder, coordinates
or scConcept embeddings) under an eval_id, for one or all probe seeds.

Saved in <ROOT_HistoST> / eval / A-3_eval_celltype_predictions / <eval_id> / <arm_dir> /
where <arm_dir> is HEencoder_t<training_num>e<checkpoint_num>_<space>, COORDS_BASELINE or SCCONCEPT_CEILING
    - runs/seed_<seed>/: per-fold outputs, fold_metrics.csv, metrics of the pooled out-of-fold predictions, timings
    - aggregate_across_seeds/: metrics aggregated across seeds
    - manifest.json: run record
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Add the task_A3 dir to sys.path so that 'src_A3' can be imported
# (ensure_imports() then adds the repo root and the task_A1 dir)
# --------------------------------------------------------------------------------------
task_A3_dir = Path(__file__).resolve().parents[1]
if str(task_A3_dir) not in sys.path:
    sys.path.insert(0, str(task_A3_dir))

from src_A3.common import ensure_imports  # noqa: E402

ensure_imports()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import h5py  # noqa: E402

from src_A3.common import (  # noqa: E402
    N_ADAPTED_ROTATIONS,
    VALID_A3_SPACES,
    PhaseTimer,
    adapted_test_stripe,
    adapted_train_stripes,
    assert_eval_config_matches_protocol_manifest,
    build_logger,
    dump_environment_report,
    ensure_project_import,
    head_type_from_config,
    infer_encoder_modelgroup,
    load_json,
    manifest_for_file,
    mkdir,
    normalize_modelgroup_name,
    read_a3_eval_config,
    resolve_a3_eval_task_root,
    resolve_encoder_path,
    resolve_he_embeddings_dir,
    rmtree_appledouble_safe,
    save_json,
    validate_a3_protocol_identity,
)
from src_A3.data_loading import (  # noqa: E402
    load_slide_records,
    resolve_main_assign_path,
    resolve_scconcept_cell_embeddings_dir,
    scconcept_slide_h5_path,
)
from src_A3.labels import (  # noqa: E402
    LabelSpace,
    build_label_space,
    derive_coarse_candidate_labels,
)
from src_A3.metrics import compute_classification_metrics  # noqa: E402
from src_A3.modeling import resolve_geometry_preprocessing  # noqa: E402
from src_A3.reporting import (  # noqa: E402
    aggregate_a3_seed_outputs,
    save_a3_metrics_outputs,
    save_a3_prediction_bundle,
    save_a3_seed_summary,
)
from src_A3.training import (  # noqa: E402
    A3FoldData,
    fit_predict_one_fold,
    slide_tensor_from_rows_a3,
)

logger = logging.getLogger(__name__)

# H5 dataset of the H&E cell embeddings per space (he / he_l2: backbone, clip: CLIP projection)
_CELL_EMB_DATASET = {"he": "patch_cell_emb_he", "he_l2": "patch_cell_emb_he", "clip": "patch_cell_emb_he_clip"}

_VALID_SPACES = VALID_A3_SPACES


# ======================================================================================
# Small helpers
# ======================================================================================
def _fingerprint_sequence(values: Sequence[object]) -> str:
    """Order-sensitive hash of an id sequence (key of the in-process tensor memo)."""
    vals = pd.Series(list(values), dtype="string")
    hvals = pd.util.hash_pandas_object(vals, index=False).to_numpy(dtype=np.uint64, copy=False)
    h = hashlib.sha1()
    h.update(np.asarray([len(vals)], dtype=np.int64).view(np.uint8))
    h.update(np.ascontiguousarray(hvals).view(np.uint8))
    return h.hexdigest()


def _load_label_space_from_vocab(vocab: Mapping[str, Any]) -> LabelSpace:
    """Rebuild the frozen LabelSpace from labels/class_vocabulary.json (LabelSpace.as_dict() format)."""
    coarse_mapping_raw = vocab.get("coarse_mapping", None)
    coarse_mapping = (
        None
        if coarse_mapping_raw is None
        else {str(k): str(v) for k, v in dict(coarse_mapping_raw).items()}
    )
    return LabelSpace(
        ct_cat_id=str(vocab["ct_cat_id"]),
        granularity=str(vocab["label_granularity"]),
        labels=[str(x) for x in vocab["labels"]],
        label_to_index={str(k): int(v) for k, v in dict(vocab["label_to_index"]).items()},
        coarse_mapping=coarse_mapping,
        dropped_labels=[str(x) for x in vocab.get("dropped_labels", [])],
        # Optional key (checked against the protocol in main())
        coarse_mapping_version=(
            str(vocab["coarse_mapping_version"])
            if vocab.get("coarse_mapping_version", None) is not None
            else None
        ),
    )


def _build_coarse_context(
    fine_label_space: LabelSpace, ct_cat_id: str, coarse_mapping_version: Optional[str]
) -> Tuple[LabelSpace, Optional[LabelSpace]]:
    """
    For a fine eval, return (fine LabelSpace with the fine -> coarse mapping, coarse LabelSpace) for the
    coarse-by-aggregation metrics. Returns (fine_label_space, None) if coarse_mapping_version is not set,
    if a fine class has no scorable coarse target, or on any error.
    """
    if not coarse_mapping_version:
        logger.info(
            "No protocol.coarse_mapping_version configured; this fine eval emits no "
            "coarse-by-aggregation mirror."
        )
        return fine_label_space, None
    try:
        coarse_candidates, canonical_to_coarse = derive_coarse_candidate_labels(
            ct_cat_id, coarse_mapping_version
        )
        fine_labels = list(fine_label_space.labels)
        coarse_mapping: Dict[str, str] = {}
        for fl in fine_labels:
            cl = canonical_to_coarse.get(fl)
            if cl is None:
                logger.warning(
                    "Coarse-by-aggregation context skipped: fine class %r has no scorable coarse "
                    "target under coarse_mapping_version=%r.",
                    fl,
                    coarse_mapping_version,
                )
                return fine_label_space, None
            coarse_mapping[fl] = str(cl)
        target_set = set(coarse_mapping.values())
        coarse_targets = [c for c in coarse_candidates if c in target_set]
        coarse_label_space = build_label_space(
            ct_cat_id=ct_cat_id,
            granularity="coarse",
            trainable_labels=coarse_targets,
            coarse_mapping_version=coarse_mapping_version,
        )
        enriched_fine = LabelSpace(
            ct_cat_id=fine_label_space.ct_cat_id,
            granularity=fine_label_space.granularity,
            labels=list(fine_label_space.labels),
            label_to_index=dict(fine_label_space.label_to_index),
            coarse_mapping=coarse_mapping,
            dropped_labels=list(fine_label_space.dropped_labels),
        )
        return enriched_fine, coarse_label_space
    except Exception as exc:  # noqa: BLE001 -- best-effort context only
        logger.warning(
            "Coarse-by-aggregation context unavailable (%s); the fine eval will emit no coarse mirror.",
            exc,
        )
        return fine_label_space, None


# ======================================================================================
# Preflight checks
# ======================================================================================
def _best_effort_modelgroup_from_embeddings(embeddings_dir: Path, expected_modelgroup: str) -> None:
    """Fallback modelgroup check from a JSON manifest of the embeddings dir (raises on mismatch, warns if none found)."""
    expected_group = normalize_modelgroup_name(expected_modelgroup)
    for name in ("manifest.json", "extraction_manifest.json", "embeddings_manifest.json"):
        path = embeddings_dir / name
        if not path.exists():
            continue
        try:
            meta = load_json(path)
        except Exception:  # noqa: BLE001 -- tolerate an unreadable/irrelevant manifest
            continue
        for key in ("modelgroup", "model_group", "eval_modelgroup"):
            if key in meta:
                actual_group = normalize_modelgroup_name(str(meta[key]))
                if actual_group != expected_group:
                    raise RuntimeError(
                        f"Embeddings-dir metadata reports modelgroup '{actual_group}', but the eval "
                        f"declares '{expected_group}' ({path})."
                    )
                logger.info("Embeddings-dir modelgroup metadata OK: %s (%s).", actual_group, path.name)
                return
    logger.warning(
        "No embeddings-dir modelgroup metadata found under %s; skipping the modelgroup check. "
        "Ensure the encoder belongs to eval modelgroup '%s'.",
        embeddings_dir,
        expected_group,
    )


def _check_encoder_modelgroup(
    encoder_path: Path, expected_modelgroup: str, embeddings_dir: Path
) -> None:
    """
    Check that the encoder's modelgroup matches the eval modelgroup (raises RuntimeError on mismatch).
    Falls back to the embeddings-dir metadata check if load_exported_he_encoder cannot be imported.
    """
    try:
        ensure_project_import()
        from src.evaluation.extract_encoders import load_exported_he_encoder
        import torch
    except Exception as exc:  # noqa: BLE001 -- degrade gracefully if the loader is unavailable
        logger.warning(
            "Strong modelgroup check unavailable (could not import load_exported_he_encoder: %s). "
            "Falling back to a best-effort embeddings-dir metadata check.",
            exc,
        )
        _best_effort_modelgroup_from_embeddings(embeddings_dir, expected_modelgroup)
        return

    enc = load_exported_he_encoder(str(encoder_path), device=torch.device("cpu"))
    expected_group = normalize_modelgroup_name(expected_modelgroup)
    actual_group = normalize_modelgroup_name(infer_encoder_modelgroup(enc))
    if actual_group != expected_group:
        backbone = str(getattr(enc, "backbone", "")).lower()
        resize_size = int(getattr(enc, "resize_size", 0))
        crop_size = int(getattr(enc, "crop_size", 0))
        raise RuntimeError(
            f"Encoder '{encoder_path.name}' is not compatible with eval modelgroup "
            f"'{expected_group}'. Detected modelgroup='{actual_group}' from backbone='{backbone}', "
            f"resize_size={resize_size}, crop_size={crop_size}."
        )
    logger.info("Encoder modelgroup OK: %s.", actual_group)


def _check_requested_space_available(
    embeddings_dir: Path,
    *,
    space: str,
    required_slide_ids: Sequence[str],
    require_main_assign: bool,
) -> None:
    """
    Check that the H5 cell-embedding dataset of space exists (2-D, non-zero width) for every required slide,
    and, if require_main_assign, that main_HEemb_cellID/<slide_id>.parquet exists. Raises RuntimeError otherwise.
    """
    required = sorted(dict.fromkeys(map(str, required_slide_ids)))
    if not required:
        raise RuntimeError("The protocol row manifest contains no slides to check.")
    dset_name = _CELL_EMB_DATASET[space]

    missing_files: List[str] = []
    missing_dataset_files: List[str] = []
    zero_width_files: List[str] = []
    bad_shape_files: List[str] = []
    missing_main_assign: List[str] = []

    for slide_id in required:
        if require_main_assign:
            main_assign = resolve_main_assign_path(embeddings_dir, slide_id)
            if not main_assign.exists():
                missing_main_assign.append(main_assign.name)
        sample = embeddings_dir / f"{slide_id}.h5"
        if not sample.exists():
            missing_files.append(sample.name)
            continue
        with h5py.File(sample, "r") as f:
            if dset_name not in f:
                missing_dataset_files.append(sample.name)
                continue
            shape = f[dset_name].shape
            if len(shape) != 2:
                bad_shape_files.append(f"{sample.name}:{tuple(shape)}")
                continue
            if shape[1] == 0:
                zero_width_files.append(sample.name)
                continue

    if (missing_files or missing_dataset_files or zero_width_files or bad_shape_files
            or missing_main_assign):
        msg = (
            f"Requested evaluation space='{space}' (cell level), but the per-cell feature source "
            f"is incomplete for the full protocol slide universe under '{embeddings_dir}'."
        )
        if missing_files:
            msg += f" Missing embedding files: {missing_files[:10]}."
        if missing_dataset_files:
            msg += f" Files missing the dataset: {missing_dataset_files[:5]}."
        if zero_width_files:
            msg += f" Files with zero-width dataset: {zero_width_files[:5]}."
        if bad_shape_files:
            msg += f" Files with invalid dataset shape: {bad_shape_files[:5]}."
        if missing_main_assign:
            msg += (
                f" Missing main-assignment parquets under 'main_HEemb_cellID/': "
                f"{missing_main_assign[:10]}."
            )
        raise RuntimeError(msg)


def _required_slide_ids(
    *,
    sub_eval_configuration: str,
    splits_manifest: Mapping[str, Any],
    area_manifest: Optional[Mapping[str, Any]],
    rows_df: pd.DataFrame,
) -> List[str]:
    """Slide IDs used at run time (train + test slides of the splits or area manifest; all row slides if empty)."""
    if sub_eval_configuration == "STHELAR_TEST_ADAPTED" and area_manifest is not None:
        ids = [str(e["slide_id"]) for e in area_manifest.get("slides", []) if "slide_id" in e]
        if ids:
            return sorted(dict.fromkeys(ids))

    stype = str(splits_manifest.get("type", ""))
    ids: List[str] = []
    if stype == "slide_cv":
        for fold in splits_manifest.get("folds", []):
            ids.extend(map(str, fold.get("train_slides", [])))
            ids.extend(map(str, fold.get("test_slides", [])))
    elif stype == "one_shot":
        ids.extend(map(str, splits_manifest.get("train_slides", [])))
        ids.extend(map(str, splits_manifest.get("test_slides", [])))
    elif stype == "within_slide_area":
        ids.extend(map(str, splits_manifest.get("slides", [])))

    if not ids:
        return sorted(rows_df["slide_id"].astype(str).unique().tolist())
    return sorted(dict.fromkeys(ids))


# ======================================================================================
# Fold construction (seed-independent) with a per-slide-tensor memo
# ======================================================================================
def _materialize_tensor(
    *,
    cfg: Mapping,
    eval_dir: Path,
    slide_record,
    rows_sub_df: pd.DataFrame,
    space: str,
    training_num: str,
    checkpoint_num: str,
    label_space: LabelSpace,
    memo: Dict[tuple, Any],
    within_split_allowed_patch_ids: Optional[Sequence[str]] = None,
    slide_rows_all_df: Optional[pd.DataFrame] = None,
) -> Any:
    """Build one A3SlideTensor, memoized (seed-independent, shared across folds and seeds)."""
    patch_scope = (
        None
        if within_split_allowed_patch_ids is None
        else _fingerprint_sequence(sorted(map(str, within_split_allowed_patch_ids)))
    )
    key = (
        str(slide_record.slide_id),
        str(space),
        str(training_num),
        str(checkpoint_num),
        _fingerprint_sequence(rows_sub_df["row_key"].astype(str).tolist()),
        patch_scope,
    )
    cached = memo.get(key)
    if cached is not None:
        return cached

    tensor = slide_tensor_from_rows_a3(
        cfg=cfg,
        eval_dir=eval_dir,
        slide_record=slide_record,
        rows_sub_df=rows_sub_df,
        space=space,
        training_num=training_num,
        checkpoint_num=checkpoint_num,
        label_space=label_space,
        within_split_allowed_patch_ids=within_split_allowed_patch_ids,
        slide_rows_all_df=slide_rows_all_df,
    )
    memo[key] = tensor
    return tensor


def _folds_slide_cv(
    *,
    cfg: Mapping,
    eval_dir: Path,
    rows_df: pd.DataFrame,
    splits_manifest: Mapping[str, Any],
    label_space: LabelSpace,
    space: str,
    training_num: str,
    checkpoint_num: str,
    slide_records: Mapping[str, Any],
    memo: Dict[tuple, Any],
) -> List[A3FoldData]:
    """Slide-level cross-validation folds (STHELAR_TRAIN_CV), one per splits manifest fold."""
    class_names = np.asarray(label_space.labels, dtype=object)
    slide_col = rows_df["slide_id"].astype(str)
    folds: List[A3FoldData] = []
    for fold in splits_manifest.get("folds", []):
        fold_id = int(fold["fold_id"])
        fold_name = f"fold_{fold_id:02d}"

        train_slides = []
        for sid in fold.get("train_slides", []):
            rows_sub = rows_df.loc[slide_col == str(sid)].copy()
            train_slides.append(
                _materialize_tensor(
                    cfg=cfg,
                    eval_dir=eval_dir,
                    slide_record=slide_records[str(sid)],
                    rows_sub_df=rows_sub,
                    space=space,
                    training_num=training_num,
                    checkpoint_num=checkpoint_num,
                    label_space=label_space,
                    memo=memo,
                    slide_rows_all_df=rows_sub,
                )
            )
        test_slides = []
        for sid in fold.get("test_slides", []):
            rows_sub = rows_df.loc[slide_col == str(sid)].copy()
            test_slides.append(
                _materialize_tensor(
                    cfg=cfg,
                    eval_dir=eval_dir,
                    slide_record=slide_records[str(sid)],
                    rows_sub_df=rows_sub,
                    space=space,
                    training_num=training_num,
                    checkpoint_num=checkpoint_num,
                    label_space=label_space,
                    memo=memo,
                    slide_rows_all_df=rows_sub,
                )
            )
        folds.append(
            A3FoldData(
                fold_name=fold_name,
                train_slides=train_slides,
                val_slides=[],
                test_slides=test_slides,
                class_names=class_names,
            )
        )
    if not folds:
        raise RuntimeError("STHELAR_TRAIN_CV splits_manifest has zero folds.")
    return folds


def _folds_one_shot(
    *,
    cfg: Mapping,
    eval_dir: Path,
    rows_df: pd.DataFrame,
    splits_manifest: Mapping[str, Any],
    label_space: LabelSpace,
    space: str,
    training_num: str,
    checkpoint_num: str,
    slide_records: Mapping[str, Any],
    memo: Dict[tuple, Any],
) -> List[A3FoldData]:
    """Single one-shot fold (STHELAR_TEST_ONE_SHOT): train on STHELAR_train slides, test on STHELAR_test slides."""
    class_names = np.asarray(label_space.labels, dtype=object)
    slide_col = rows_df["slide_id"].astype(str)

    def _tensors(slide_ids: Sequence[str]) -> List[Any]:
        out = []
        for sid in slide_ids:
            rows_sub = rows_df.loc[slide_col == str(sid)].copy()
            out.append(
                _materialize_tensor(
                    cfg=cfg,
                    eval_dir=eval_dir,
                    slide_record=slide_records[str(sid)],
                    rows_sub_df=rows_sub,
                    space=space,
                    training_num=training_num,
                    checkpoint_num=checkpoint_num,
                    label_space=label_space,
                    memo=memo,
                    slide_rows_all_df=rows_sub,
                )
            )
        return out

    train_slides = _tensors(splits_manifest.get("train_slides", []))
    test_slides = _tensors(splits_manifest.get("test_slides", []))
    if not train_slides or not test_slides:
        raise RuntimeError(
            f"STHELAR_TEST_ONE_SHOT split is degenerate: "
            f"{len(train_slides)} train slide(s), {len(test_slides)} test slide(s)."
        )
    return [
        A3FoldData(
            fold_name="oneshot",
            train_slides=train_slides,
            val_slides=[],
            test_slides=test_slides,
            class_names=class_names,
        )
    ]


def _folds_within_slide(
    *,
    cfg: Mapping,
    eval_dir: Path,
    rows_df: pd.DataFrame,
    area_manifest: Mapping[str, Any],
    label_space: LabelSpace,
    space: str,
    training_num: str,
    checkpoint_num: str,
    slide_records: Mapping[str, Any],
    memo: Dict[tuple, Any],
) -> List[A3FoldData]:
    """
    Within-slide area folds (STHELAR_TEST_ADAPTED): for each slide, ambiguous cells are excluded and each of
    the 3 stripe rotations trains on 2 stripes (one train tensor per stripe) and tests on the remaining one.
    """
    class_names = np.asarray(label_space.labels, dtype=object)
    folds: List[A3FoldData] = []

    for slide_entry in area_manifest.get("slides", []):
        sid = str(slide_entry["slide_id"])
        if "cell_manifest" not in slide_entry:
            raise KeyError(
                f"Area manifest entry for slide '{sid}' has no 'cell_manifest'; Task A.3 is "
                "cell-level and requires the per-slide cell-stripe parquet."
            )
        patch_df = pd.read_parquet(slide_entry["patch_manifest"])
        patch_df["patch_id"] = patch_df["patch_id"].astype(str)
        patch_stripe = dict(zip(patch_df["patch_id"], patch_df["stripe_id"].astype(int)))

        cell_df = pd.read_parquet(slide_entry["cell_manifest"])
        rows_slide = rows_df.loc[
            (rows_df["slide_id"].astype(str) == sid)
            & rows_df["cell_id_int"].isin(cell_df.loc[~cell_df["is_ambiguous"], "cell_id_int"])
        ].copy()

        for fold_idx in range(N_ADAPTED_ROTATIONS):
            train_stripes = adapted_train_stripes(fold_idx)
            test_stripe = adapted_test_stripe(fold_idx)

            # One train tensor per train stripe, with H&E rows taken only from that stripe's patches
            train_tensors = []
            for st in train_stripes:
                st_patch_ids = [p for p, s in patch_stripe.items() if int(s) == st]
                st_rows = rows_slide.loc[
                    rows_slide["cell_id_int"].isin(
                        cell_df.loc[cell_df["stripe_id"] == st, "cell_id_int"]
                    )
                ].copy()
                train_tensors.append(
                    _materialize_tensor(
                        cfg=cfg,
                        eval_dir=eval_dir,
                        slide_record=slide_records[sid],
                        rows_sub_df=st_rows,
                        space=space,
                        training_num=training_num,
                        checkpoint_num=checkpoint_num,
                        label_space=label_space,
                        memo=memo,
                        within_split_allowed_patch_ids=st_patch_ids,
                        slide_rows_all_df=rows_slide,
                    )
                )

            test_patch_ids = [p for p, s in patch_stripe.items() if int(s) == test_stripe]
            test_rows = rows_slide.loc[
                rows_slide["cell_id_int"].isin(
                    cell_df.loc[cell_df["stripe_id"] == test_stripe, "cell_id_int"]
                )
            ].copy()
            test_tensor = _materialize_tensor(
                cfg=cfg,
                eval_dir=eval_dir,
                slide_record=slide_records[sid],
                rows_sub_df=test_rows,
                space=space,
                training_num=training_num,
                checkpoint_num=checkpoint_num,
                label_space=label_space,
                memo=memo,
                within_split_allowed_patch_ids=test_patch_ids,
                slide_rows_all_df=rows_slide,
            )
            folds.append(
                A3FoldData(
                    fold_name=f"{sid}_fold_{fold_idx}",
                    train_slides=train_tensors,
                    val_slides=[],
                    test_slides=[test_tensor],
                    class_names=class_names,
                )
            )

    if not folds:
        raise RuntimeError("STHELAR_TEST_ADAPTED area manifest produced zero within-slide folds.")
    return folds


def _build_a3_folds(
    *,
    cfg: Mapping,
    eval_dir: Path,
    rows_df: pd.DataFrame,
    splits_manifest: Mapping[str, Any],
    area_manifest: Optional[Mapping[str, Any]],
    label_space: LabelSpace,
    space: str,
    training_num: str,
    checkpoint_num: str,
    slide_records: Mapping[str, Any],
    memo: Dict[tuple, Any],
) -> List[A3FoldData]:
    """Dispatch to the CV / one-shot / within-slide fold builder by the frozen splits type."""
    stype = str(splits_manifest.get("type", ""))
    if stype == "slide_cv":
        return _folds_slide_cv(
            cfg=cfg,
            eval_dir=eval_dir,
            rows_df=rows_df,
            splits_manifest=splits_manifest,
            label_space=label_space,
            space=space,
            training_num=training_num,
            checkpoint_num=checkpoint_num,
            slide_records=slide_records,
            memo=memo,
        )
    if stype == "one_shot":
        return _folds_one_shot(
            cfg=cfg,
            eval_dir=eval_dir,
            rows_df=rows_df,
            splits_manifest=splits_manifest,
            label_space=label_space,
            space=space,
            training_num=training_num,
            checkpoint_num=checkpoint_num,
            slide_records=slide_records,
            memo=memo,
        )
    if stype == "within_slide_area":
        if area_manifest is None:
            raise RuntimeError(
                "splits type 'within_slide_area' (STHELAR_TEST_ADAPTED) requires the frozen "
                "area_split_manifest.json, which was not loaded."
            )
        return _folds_within_slide(
            cfg=cfg,
            eval_dir=eval_dir,
            rows_df=rows_df,
            area_manifest=area_manifest,
            label_space=label_space,
            space=space,
            training_num=training_num,
            checkpoint_num=checkpoint_num,
            slide_records=slide_records,
            memo=memo,
        )
    raise ValueError(f"Unsupported splits_manifest type for Task A.3: {stype!r}.")


# ======================================================================================
# OOF merge (predictions on the global class axis)
# ======================================================================================
def _merge_a3_slide_bundle(existing: Mapping, new: Mapping) -> Dict[str, object]:
    """
    Merge two prediction bundles of the same slide (STHELAR_TEST_ADAPTED rotations): check that the class
    axes match and the row_keys are disjoint, concatenate the rows and sort them by row_keys.
    """
    if str(existing["slide_id"]) != str(new["slide_id"]):
        raise RuntimeError("Cannot merge A.3 predictions from different slides.")

    old_classes = np.asarray(existing["class_names"], dtype=object)
    new_classes = np.asarray(new["class_names"], dtype=object)
    if not np.array_equal(old_classes, new_classes):
        raise RuntimeError(
            f"OOF merge for slide '{existing['slide_id']}' found mismatched class axes; every "
            "A.3 bundle must carry the frozen global class vocabulary."
        )

    old_keys = np.asarray(existing["row_keys"], dtype=object)
    new_keys = np.asarray(new["row_keys"], dtype=object)
    overlap = np.intersect1d(old_keys, new_keys)
    if overlap.size > 0:
        raise RuntimeError(
            f"OOF merge for slide '{existing['slide_id']}' found duplicated row_keys "
            f"(each cell must be scored out-of-fold exactly once). Examples: {overlap[:10].tolist()}"
        )

    merged: Dict[str, object] = dict(existing)
    merged["row_keys"] = np.concatenate([old_keys, new_keys], axis=0)
    merged["coords_xy"] = np.concatenate(
        [np.asarray(existing["coords_xy"], dtype=np.float32), np.asarray(new["coords_xy"], dtype=np.float32)],
        axis=0,
    )
    merged["y_true"] = np.concatenate(
        [np.asarray(existing["y_true"], dtype=np.int32), np.asarray(new["y_true"], dtype=np.int32)],
        axis=0,
    )
    merged["y_proba"] = np.concatenate(
        [np.asarray(existing["y_proba"], dtype=np.float32), np.asarray(new["y_proba"], dtype=np.float32)],
        axis=0,
    )
    merged["true_class_covered"] = np.concatenate(
        [
            np.asarray(existing["true_class_covered"], dtype=bool),
            np.asarray(new["true_class_covered"], dtype=bool),
        ],
        axis=0,
    )
    # Per-row fold names (the scalar fold_name is broadcast if fold_names is missing)
    old_folds = existing.get("fold_names")
    if old_folds is None:
        old_folds = np.full(old_keys.shape[0], str(existing.get("fold_name", "")), dtype=object)
    new_folds = new.get("fold_names")
    if new_folds is None:
        new_folds = np.full(new_keys.shape[0], str(new.get("fold_name", "")), dtype=object)
    merged["fold_names"] = np.concatenate(
        [np.asarray(old_folds, dtype=object), np.asarray(new_folds, dtype=object)], axis=0
    )

    old_ct = existing.get("cell_types", None)
    new_ct = new.get("cell_types", None)
    if old_ct is None and new_ct is None:
        merged["cell_types"] = None
    elif old_ct is not None and new_ct is not None:
        merged["cell_types"] = np.concatenate(
            [np.asarray(old_ct, dtype=object), np.asarray(new_ct, dtype=object)], axis=0
        )
    else:
        raise RuntimeError(
            f"Inconsistent cell_types presence while merging slide '{existing['slide_id']}'."
        )

    old_obs = existing.get("obs_df", None)
    new_obs = new.get("obs_df", None)
    if old_obs is None and new_obs is None:
        merged["obs_df"] = None
    elif old_obs is not None and new_obs is not None:
        merged["obs_df"] = pd.concat([old_obs, new_obs], axis=0, ignore_index=True)
    else:
        raise RuntimeError(
            f"Inconsistent obs_df presence while merging slide '{existing['slide_id']}'."
        )

    order = np.argsort(np.asarray(merged["row_keys"]).astype(str))
    merged["row_keys"] = np.asarray(merged["row_keys"])[order]
    merged["coords_xy"] = np.asarray(merged["coords_xy"])[order]
    merged["y_true"] = np.asarray(merged["y_true"])[order]
    merged["y_proba"] = np.asarray(merged["y_proba"])[order]
    merged["true_class_covered"] = np.asarray(merged["true_class_covered"])[order]
    merged["fold_names"] = np.asarray(merged["fold_names"])[order]
    if merged["cell_types"] is not None:
        merged["cell_types"] = np.asarray(merged["cell_types"])[order]
    if merged["obs_df"] is not None:
        merged["obs_df"] = merged["obs_df"].iloc[order].reset_index(drop=True)

    if merged["y_proba"].shape[0] != merged["y_true"].shape[0]:
        raise RuntimeError(
            f"Merged y_proba/y_true row mismatch for slide '{existing['slide_id']}': "
            f"{merged['y_proba'].shape[0]} vs {merged['y_true'].shape[0]}."
        )
    return merged


def _merge_a3_prediction_dicts(dst: Dict[str, Mapping], src: Mapping[str, Mapping]) -> None:
    """Add one fold's test predictions to the pooled OOF predictions (keyed by slide_id; same slide -> merged)."""
    for slide_id, bundle in src.items():
        if slide_id not in dst:
            dst[slide_id] = bundle
        else:
            dst[slide_id] = _merge_a3_slide_bundle(dst[slide_id], bundle)


# ======================================================================================
# Per-seed run
# ======================================================================================
def _fold_summary_row(
    fold: A3FoldData,
    artifacts: Mapping[str, object],
    quick_metrics: Mapping[str, object],
    *,
    space: str,
    cfg: Mapping,
) -> Dict[str, object]:
    """One row of fold_metrics.csv: fold info, probe fit info and quick benchmark summary."""
    row: Dict[str, object] = {
        "fold_name": str(fold.fold_name),
        "head_type": head_type_from_config(cfg),
        "space": str(space),
        "converged": bool(artifacts.get("converged", True)),
        "n_train": int(artifacts.get("n_train", 0)),
        "n_features_kept": int(artifacts.get("n_features_kept", -1)),
        # Selected C and probe capacity info
        "selected_C": artifacts.get("selected_C"),
        "C_selection_status": artifacts.get("C_selection_status"),
        "effective_dof": artifacts.get("effective_dof"),
        "alpha_eff": artifacts.get("alpha_eff"),
        "numerical_rank": artifacts.get("numerical_rank"),
        "inner_cv_macro_f1_at_selected": artifacts.get("inner_cv_macro_f1_at_selected"),
        "n_inner_cv_groups": artifacts.get("n_inner_cv_groups"),
        "lambda_boundary_kind": artifacts.get("lambda_boundary_kind"),
        "probe_capacity_status": artifacts.get("probe_capacity_status"),
    }
    # Timings (*_seconds; select_C_seconds includes the inner_cv_* phases) and inner-CV fit count
    row.update({k: v for k, v in artifacts.items() if str(k).endswith("_seconds")})
    row["n_inner_cv_fits"] = artifacts.get("n_inner_cv_fits")
    row.update(dict(quick_metrics.get("benchmark_summary", {})))
    return row


def _run_seed(
    *,
    cfg: Mapping,
    encoder_dir: Path,
    folds: Sequence[A3FoldData],
    label_space: LabelSpace,
    metrics_label_space: LabelSpace,
    coarse_label_space: Optional[LabelSpace],
    training_num: str,
    checkpoint_num: str,
    space: str,
    seed: int,
    skip_predictions: bool = False,
) -> bool:
    """
    Run one probe seed over the prebuilt folds: fit and predict each fold, then compute the metrics of the
    pooled out-of-fold predictions. skip_predictions skips writing the prediction files. Returns True when done.
    """
    seed_dir = encoder_dir / "runs" / f"seed_{seed:05d}"
    if seed_dir.exists():
        rmtree_appledouble_safe(seed_dir)
    seed_dir = mkdir(seed_dir)
    seed_timer = PhaseTimer()
    seed_logger = build_logger(seed_dir / "log" / "logfile.log", name=f"a3_seed_{seed}")
    seed_logger.info(
        "Running Task A.3 eval_id=%s seed=%d encoder=%s (%d folds).",
        cfg["identity"]["eval_id"],
        seed,
        encoder_dir.name,
        len(folds),
    )

    merged_test_preds: Dict[str, Mapping] = {}
    fold_records: List[Dict[str, object]] = []

    for i, fold in enumerate(folds):
        fold_dir = mkdir(seed_dir / "folds" / f"fold_{i:02d}")
        seed_logger.info("Fitting fold %d/%d (%s).", i + 1, len(folds), fold.fold_name)

        with seed_timer.phase("fold_fit"):
            artifacts, _val_preds, test_preds = fit_predict_one_fold(
                cfg=cfg,
                fold_data=fold,
                space=space,
                seed=seed,
                fold_index=i,
                label_space=label_space,
                output_dir=fold_dir,
            )

        if not skip_predictions:
            with seed_timer.phase("fold_predictions_write"):
                save_a3_prediction_bundle(test_preds, fold_dir / "test_predictions")

        # Quick per-fold metrics (no coarse-by-aggregation metrics)
        with seed_timer.phase("fold_quick_metrics"):
            quick_metrics = compute_classification_metrics(
                test_preds, cfg, label_space=label_space, coarse_label_space=None
            )
            save_a3_metrics_outputs(quick_metrics, fold_dir / "quick_benchmark")
        fold_records.append(_fold_summary_row(fold, artifacts, quick_metrics, space=space, cfg=cfg))

        _merge_a3_prediction_dicts(merged_test_preds, test_preds)

    pd.DataFrame(fold_records).to_csv(seed_dir / "fold_metrics.csv", index=False)

    # Pooled out-of-fold predictions (every test fold concatenated per slide, global class axis).
    if not skip_predictions:
        with seed_timer.phase("pooled_predictions_write"):
            save_a3_prediction_bundle(merged_test_preds, seed_dir / "seed_predictions")

    with seed_timer.phase("pooled_metrics"):
        metrics = compute_classification_metrics(
            merged_test_preds,
            cfg,
            label_space=metrics_label_space,
            coarse_label_space=coarse_label_space,
        )
        save_a3_metrics_outputs(metrics, seed_dir / "benchmark")
        save_a3_seed_summary(seed_dir, metrics)

    # Timings of this seed (fold_phase_totals: sum of the per-fold phases)
    fold_seconds: Dict[str, float] = {}
    for rec in fold_records:
        for key, value in rec.items():
            if not str(key).endswith("_seconds"):
                continue
            if isinstance(value, (int, float)) and np.isfinite(value):
                fold_seconds[str(key)] = fold_seconds.get(str(key), 0.0) + float(value)
    n_inner_fits = int(sum(int(r.get("n_inner_cv_fits") or 0) for r in fold_records))
    save_json(
        seed_dir / "timing.json",
        {
            "seed": int(seed),
            "n_folds": len(folds),
            "skip_predictions": bool(skip_predictions),
            "seed_phases": seed_timer.as_dict(),
            "seed_phase_counts": seed_timer.counts(),
            "fold_phase_totals": {k: round(v, 3) for k, v in sorted(fold_seconds.items())},
            "n_inner_cv_fits_total": n_inner_fits,
            "note": (
                "Nested phases: `fold_fit` contains the whole per-fold fit, and `select_C` "
                "contains the `inner_cv_*` phases (a parent's time includes its children)."
            ),
        },
    )

    bench = metrics.get("benchmark_summary", {})
    seed_logger.info(
        "Seed %d complete: balanced_accuracy=%s macro_f1=%s macro_auroc=%s (n_cells=%s).",
        seed,
        bench.get("balanced_accuracy"),
        bench.get("macro_f1"),
        bench.get("macro_auroc"),
        bench.get("n_cells"),
    )
    seed_logger.info(
        "Seed %d wall-clock: fold_fit=%.1fs (select_C=%.1fs over %d inner-CV lbfgs fits, "
        "final_fit=%.1fs, capacity=%.1fs) | predictions_write=%.1fs | metrics=%.1fs.",
        seed,
        seed_timer.seconds("fold_fit"),
        fold_seconds.get("select_C_seconds", 0.0),
        n_inner_fits,
        fold_seconds.get("final_fit_seconds", 0.0),
        fold_seconds.get("capacity_readout_seconds", 0.0),
        seed_timer.seconds("fold_predictions_write")
        + seed_timer.seconds("pooled_predictions_write"),
        seed_timer.seconds("fold_quick_metrics") + seed_timer.seconds("pooled_metrics"),
    )
    return True


# ======================================================================================
# Encoder-level manifest
# ======================================================================================
def _save_a3_encoder_manifest(
    *,
    encoder_dir: Path,
    encoder_path: Optional[Path],
    embeddings_dir: Optional[Path],
    cfg: Mapping,
    run_eval_dir: Path,
    requested_seeds: Sequence[int],
    completed_seeds: Sequence[int],
    training_num: str,
    checkpoint_num: str,
    space: str,
    label_space: LabelSpace,
    kind: str = "he_encoder",
    skip_predictions: bool = False,
    run_timing: Optional[PhaseTimer] = None,
) -> None:
    """Write encoder_dir/manifest.json (run record). kind is the arm: he_encoder, coords_baseline or scconcept_ceiling."""
    protocol = cfg.get("protocol", {})
    payload = {
        "encoder_identity": {
            "kind": str(kind),
            "training_num": str(training_num),
            "checkpoint_num": str(checkpoint_num),
            "space": str(space),
            "scconcept_training_id": (
                str(protocol.get("scconcept_training_id"))
                if kind == "scconcept_ceiling" else None
            ),
        },
        "encoder_file": (manifest_for_file(encoder_path).__dict__ if encoder_path is not None else None),
        "embeddings_dir": (str(embeddings_dir) if embeddings_dir is not None else None),
        "protocol_manifest": str(run_eval_dir / "settings" / "protocol_manifest.json"),
        "run_eval_dir": str(run_eval_dir),
        "prediction_head": {
            "head_type": head_type_from_config(cfg),
            "mode": "sklearn_multinomial_logistic_regression",
            "probe": dict(protocol.get("probe", {})),
        },
        "input_feature_pipeline": {
            "raw_features": {
                "he_encoder": "he_encoder_cell_embeddings",
                "coords_baseline": "nucleus_xy_coordinates",
                "scconcept_ceiling": "scconcept_cls_emb",
            }.get(str(kind), "he_encoder_cell_embeddings"),
            "encoder_space": str(space),
            "geometry": resolve_geometry_preprocessing(space),
            "standardization": "standardize_drop_constant",
            "feature_pipeline": "geometry -> standardize_drop_constant -> logistic_regression",
        },
        "class_axis": {
            "granularity": str(label_space.granularity),
            "n_classes": int(label_space.n_classes()),
            "labels": list(label_space.labels),
        },
        "requested_probe_seeds": list(map(int, requested_seeds)),
        "completed_probe_seeds": list(map(int, completed_seeds)),
        "selection_metric": (
            "descriptive (per-slide macro_f1 / balanced_accuracy)"
        ),
        "wandb_mode": str(protocol.get("runtime", {}).get("wandb_mode", "offline")),
        # Optional outputs skipped by this run (not part of the protocol)
        "outputs": {
            "skip_predictions": bool(skip_predictions),
            "prediction_bundles_written": not bool(skip_predictions),
        },
        # Timings of the run (per-fold details in runs/seed_*/timing.json)
        "timing": (run_timing.as_dict() if run_timing is not None else {}),
        "environment": dump_environment_report(),
    }
    save_json(encoder_dir / "manifest.json", payload)


def _check_scconcept_available(
    embeddings_dir: Path, *, required_slide_ids: Sequence[str], scconcept_training_id: str
) -> None:
    """Check that the scConcept embeddings dir and the per-cell H5 of every required slide exist (raises otherwise)."""
    if not Path(embeddings_dir).exists():
        raise FileNotFoundError(
            f"scConcept ceiling arm: embeddings dir not found for scconcept_training_id="
            f"'{scconcept_training_id}': {embeddings_dir}. Build the scConcept per-cell embeddings first."
        )
    missing = [
        sid
        for sid in required_slide_ids
        if not scconcept_slide_h5_path(scconcept_training_id, sid).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"scConcept ceiling arm: missing per-cell embedding H5 for {len(missing)} required "
            f"slide(s) under scconcept_training_id='{scconcept_training_id}'. Examples: {missing[:10]}. "
            "Build scConcept embeddings for every eval slide, or omit the scconcept_ceiling arm."
        )


# ======================================================================================
# main
# ======================================================================================
def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    print(f"Starting Task A.3 evaluation with args: {args}")
    print("\n")

    # (1) Load + validate the frozen protocol.
    run_eval_dir = resolve_a3_eval_task_root() / args.eval_id
    cfg = read_a3_eval_config(run_eval_dir)
    protocol_manifest = load_json(run_eval_dir / "settings" / "protocol_manifest.json")
    assert_eval_config_matches_protocol_manifest(cfg, protocol_manifest)
    validate_a3_protocol_identity(cfg)  # NotImplementedError for COORDS / HEST sub-evals

    identity = cfg["identity"]
    protocol = cfg["protocol"]
    dataset_id = str(identity["dataset_id"])
    training_dataset_id = str(identity["training_dataset_id"])
    ct_cat_id = str(protocol["ct_cat_id"])
    granularity = str(protocol["label_granularity"])
    _coarse_version_raw = protocol.get("coarse_mapping_version", None)
    coarse_mapping_version = str(_coarse_version_raw).strip() if _coarse_version_raw else None
    sub_cfg = str(protocol["sub_eval_configuration"])

    # (2) Resolve the arm: H&E encoder, coordinates ("coords" space) or scConcept embeddings ("scconcept" space)
    arm = str(args.arm or "he_encoder").strip().lower()
    if arm not in {"he_encoder", "coords", "scconcept_ceiling"}:
        raise ValueError(f"--arm must be he_encoder | coords | scconcept_ceiling; got {arm!r}.")

    if arm == "he_encoder":
        if args.training_num is None or args.checkpoint_num is None or args.space is None:
            raise ValueError(
                "--arm he_encoder requires --training_num, --checkpoint_num and --space (he | he_l2 | clip)."
            )
        training_num = str(args.training_num)
        checkpoint_num = str(args.checkpoint_num)
        space = str(args.space)
        if space not in _VALID_SPACES:
            raise ValueError(f"--space must be one of {sorted(_VALID_SPACES)}; got {space!r}.")
        encoder_kind = "he_encoder"
        encoder_dir_name = f"HEencoder_t{training_num}e{checkpoint_num}_{space}"
    elif arm == "coords":
        training_num, checkpoint_num, space = "NA", "NA", "coords"
        encoder_kind = "coords_baseline"
        encoder_dir_name = "COORDS_BASELINE"
    else:  # scconcept_ceiling
        if not str(protocol.get("scconcept_training_id") or "").strip():
            raise ValueError(
                "--arm scconcept_ceiling requires protocol.scconcept_training_id in the frozen "
                "eval_config (set it via `_1_setup.py --scconcept_training_id ...`)."
            )
        training_num, checkpoint_num, space = "NA", "NA", "scconcept"
        encoder_kind = "scconcept_ceiling"
        encoder_dir_name = "SCCONCEPT_CEILING"

    if args.device is not None:
        logger.info(
            "--device=%s is unused (the A.3 probe is a CPU sklearn LogisticRegression).",
            args.device,
        )

    # (3) Load the frozen protocol artifacts.
    rows_df = pd.read_parquet(run_eval_dir / "settings" / "rows" / "all_rows.parquet")
    vocab = load_json(run_eval_dir / "settings" / "labels" / "class_vocabulary.json")
    label_space = _load_label_space_from_vocab(vocab)
    # The class vocabulary and the protocol must use the same coarse mapping version
    if granularity == "coarse":
        vocab_version = vocab.get("coarse_mapping_version", None)
        if vocab_version is None:
            raise RuntimeError(
                f"{run_eval_dir / 'settings' / 'labels' / 'class_vocabulary.json'} records no "
                "'coarse_mapping_version' but the protocol is coarse "
                f"(coarse_mapping_version={coarse_mapping_version!r}). Re-freeze the eval_id with "
                "`_2_fill_eval_settings.py`."
            )
        if str(vocab_version) != str(coarse_mapping_version):
            raise RuntimeError(
                "Frozen class vocabulary and protocol disagree on the coarse class axis: "
                f"class_vocabulary.json says {vocab_version!r}, protocol says "
                f"{coarse_mapping_version!r}. Re-freeze the eval_id."
            )
    splits_manifest = load_json(run_eval_dir / "settings" / "splits" / "splits_manifest.json")
    # Frozen per-fold train class coverage (true_class_covered is computed in fit_predict_one_fold)
    fold_class_coverage = load_json(
        run_eval_dir / "settings" / "labels" / "fold_class_coverage.json"
    )
    area_manifest: Optional[Mapping[str, Any]] = None
    if sub_cfg == "STHELAR_TEST_ADAPTED":
        area_manifest = load_json(
            run_eval_dir / "settings" / "area_splits" / "area_split_manifest.json"
        )
    logger.info(
        "Loaded protocol: %s / %s, %d %s classes, %d frozen fold(s), %d scored cell(s).",
        sub_cfg,
        granularity,
        int(label_space.n_classes()),
        granularity,
        len(fold_class_coverage),
        int(rows_df.shape[0]),
    )

    # (4) Resolve the arm's feature source; run the arm-appropriate preflight checks.
    required_slide_ids = _required_slide_ids(
        sub_eval_configuration=sub_cfg,
        splits_manifest=splits_manifest,
        area_manifest=area_manifest,
        rows_df=rows_df,
    )
    encoder_path: Optional[Path] = None
    embeddings_dir: Optional[Path] = None
    if arm == "he_encoder":
        encoder_path = resolve_encoder_path(training_num, checkpoint_num)
        embeddings_dir = resolve_he_embeddings_dir(training_dataset_id, training_num, checkpoint_num)
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
        if not embeddings_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
        _check_encoder_modelgroup(encoder_path, str(protocol["modelgroup"]), embeddings_dir)
        _check_requested_space_available(
            embeddings_dir,
            space=space,
            required_slide_ids=required_slide_ids,
            # ADAPTED reads patches/ + nucleus_centroids/ instead of the main-assignment parquet.
            require_main_assign=(sub_cfg != "STHELAR_TEST_ADAPTED"),
        )
    elif arm == "coords":
        # Coordinates are in the row manifest (no encoder or embeddings dir)
        pass
    else:  # scconcept_ceiling
        scc_tid = str(protocol.get("scconcept_training_id")).strip()
        embeddings_dir = resolve_scconcept_cell_embeddings_dir(scc_tid)
        _check_scconcept_available(
            embeddings_dir, required_slide_ids=required_slide_ids, scconcept_training_id=scc_tid
        )

    # (5) Class spaces for the metrics (fine eval: also coarse-by-aggregation)
    if granularity == "fine":
        metrics_label_space, coarse_label_space = _build_coarse_context(
            label_space, ct_cat_id, coarse_mapping_version
        )
    else:
        metrics_label_space, coarse_label_space = label_space, None

    # (6) Build the folds once (seed-independent; slide tensors memoized)
    slide_records = load_slide_records(training_dataset_id)
    tensor_memo: Dict[tuple, Any] = {}
    run_timer = PhaseTimer()
    with run_timer.phase("build_folds"):
        folds = _build_a3_folds(
            cfg=cfg,
            eval_dir=run_eval_dir,
            rows_df=rows_df,
            splits_manifest=splits_manifest,
            area_manifest=area_manifest,
            label_space=label_space,
            space=space,
            training_num=training_num,
            checkpoint_num=checkpoint_num,
            slide_records=slide_records,
            memo=tensor_memo,
        )
    logger.info(
        "Materialized %d fold(s) (%d unique slide tensors cached) in %.1fs.",
        len(folds),
        len(tensor_memo),
        run_timer.seconds("build_folds"),
    )

    # (7) Run each probe seed over the shared folds.
    requested_seeds = (
        list(map(int, protocol["probe_seeds"]))
        if args.probe_seed == "all"
        else [int(args.probe_seed)]
    )
    encoder_dir = mkdir(run_eval_dir / encoder_dir_name)

    for seed in requested_seeds:
        with run_timer.phase("run_seeds"):
            _run_seed(
                cfg=cfg,
                encoder_dir=encoder_dir,
                folds=folds,
                label_space=label_space,
                metrics_label_space=metrics_label_space,
                coarse_label_space=coarse_label_space,
                training_num=training_num,
                checkpoint_num=checkpoint_num,
                space=space,
                seed=seed,
                skip_predictions=bool(args.skip_predictions),
            )

    # (8) Aggregate across seeds + write the encoder manifest.
    aggregate_a3_seed_outputs(encoder_dir)
    runs_dir = encoder_dir / "runs"
    completed = sorted(
        int(p.name.split("_")[-1])
        for p in runs_dir.iterdir()
        if p.is_dir()
        and p.name.startswith("seed_")
        and (p / "benchmark" / "benchmark_summary.json").exists()
    )
    _save_a3_encoder_manifest(
        encoder_dir=encoder_dir,
        encoder_path=encoder_path,
        embeddings_dir=embeddings_dir,
        cfg=cfg,
        run_eval_dir=run_eval_dir,
        requested_seeds=requested_seeds,
        completed_seeds=completed,
        training_num=training_num,
        checkpoint_num=checkpoint_num,
        space=space,
        label_space=label_space,
        kind=encoder_kind,
        skip_predictions=bool(args.skip_predictions),
        run_timing=run_timer,
    )
    print(f"Completed Task A.3 {encoder_kind} evaluation in: {encoder_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the Task A.3 (cell-type linear-probe classification) evaluation for one arm "
            "and one or all probe seeds."
        )
    )

    parser.add_argument(
        "--eval_id",
        type=str,
        required=True,
        help="Evaluation ID, e.g. 'eval_1'.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="he_encoder",
        choices=["he_encoder", "coords", "scconcept_ceiling"],
        help=(
            "Arm to run: 'he_encoder' (needs --training_num, --checkpoint_num and --space), 'coords' "
            "(nucleus (x, y) coordinates) or 'scconcept_ceiling' (scConcept cls_emb, needs "
            "protocol.scconcept_training_id)."
        ),
    )
    parser.add_argument(
        "--training_num",
        type=str,
        default=None,
        help="Num from HistoST training_id (e.g. '46' for training_id=='training_46'). "
        "Use '-1' for the raw foundational baseline. Required only for --arm he_encoder.",
    )
    parser.add_argument(
        "--checkpoint_num",
        type=str,
        default=None,
        help="Checkpoint number to load (e.g. '13' for 'checkpoint_epoch_13.pth'). "
        "Use '-1' for the raw foundational baseline.",
    )
    parser.add_argument(
        "--space",
        type=str,
        default=None,
        choices=sorted(VALID_A3_SPACES),
        help=(
            "Embedding space: 'he' (backbone), 'he_l2' (backbone, row L2-normalized) or 'clip' "
            "(CLIP projection, row L2-normalized)."
        ),
    )
    parser.add_argument(
        "--probe_seed",
        type=str,
        default="all",
        help="'all' (every seed of protocol.probe_seeds) or one integer seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Unused (the probe is a CPU sklearn LogisticRegression).",
    )
    parser.add_argument(
        "--skip_predictions",
        action="store_true",
        help=(
            "Do not write the HDF5 prediction files (runs/seed_*/folds/*/test_predictions/ and "
            "runs/seed_*/seed_predictions/). Metrics are unchanged."
        ),
    )

    args = parser.parse_args()
    main(args)
