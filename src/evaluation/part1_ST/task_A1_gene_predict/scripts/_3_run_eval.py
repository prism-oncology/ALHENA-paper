#!/usr/bin/env python
"""
Run the Task A.1 evaluation of one encoder for a given eval_id: fit the ridge probe per fold,
predict the held-out slides, and compute metrics and reports.

Saved in <ROOT_HistoST> / eval / A-1_eval_st_predictions / <eval_id> / HEencoder_t<training_num>e<checkpoint_num>_<space> /
    - manifest.json: encoder, probe and output settings
    - run/: per-fold fits and predictions (folds/), benchmark/, diagnostics/, structure/, reports/
    - aggregate_across_folds/: per-fold benchmark and variability across folds
"""

import argparse
import hashlib
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import h5py
import re

project_root = Path(__file__).resolve().parents[1]
import sys
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src_A1.common import (
    HEST_CV_SUB_EVAL_CONFIGS,
    _ADAPTED_STRIPE_ROLES,
    assert_eval_config_matches_protocol_manifest,
    assert_reference_protocol_hashes_unchanged,
    build_logger,
    dump_environment_report,
    infer_encoder_modelgroup,
    load_json,
    load_yaml,
    manifest_for_file,
    mkdir,
    normalize_modelgroup_name,
    resolve_encoder_path,
    resolve_eval_dataset_root,
    resolve_eval_task_root,
    resolve_he_embeddings_dir,
    resolve_training_dataset_dir,
    rmtree_appledouble_safe,
    save_json,
    head_type_from_config,
    prediction_head_manifest_payload,
    protocol_eval_dir_from_manifest,
    validate_protocol_identity,
    SIGNATURE_AXIS_SEP,
)
from src_A1.data_loading import (
    aggregate_signature_targets,
    build_split_specific_cell_embedding_index,
    extract_targets_for_rows,
    load_cell_embeddings_via_main_assignment,
    load_patch_embeddings,
    load_slide_records,
    load_split_specific_cell_embeddings,
)
from src_A1.metrics import (
    compute_benchmark_metrics,
    compute_diagnostics,
    compute_structure_analyses,
)
from src_A1.protocols import load_thinned_halves
from src_A1.reporting import (
    save_benchmark_outputs,
    save_canonical_reports,
    save_diagnostics_outputs,
    save_prediction_bundle,
    save_run_summary,
    save_structure_outputs,
)
from src_A1.training import (
    FoldData,
    SlideTensorData,
    fit_predict_one_fold,
)
from src_A1.modeling import resolve_feature_geometry
from src_A1.wandb_utils import WandbRunHandle

from src_A1.common import ensure_project_import
ensure_project_import()
from src.evaluation.extract_encoders import load_exported_he_encoder


def _fingerprint_sequence(values: Sequence[object]) -> str:
    """Compact fingerprint of an id sequence (in-process cache key)."""
    vals = pd.Series(list(values), dtype="string")
    hvals = pd.util.hash_pandas_object(vals, index=False).to_numpy(dtype=np.uint64, copy=False)
    h = hashlib.sha1()
    h.update(np.asarray([len(vals)], dtype=np.int64).view(np.uint8))
    h.update(np.ascontiguousarray(hvals).view(np.uint8))
    return h.hexdigest()


def _bool_array_from_series(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).to_numpy(dtype=bool)
    
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        bad = x.notna() & ~x.isin([0, 1])
        if bool(bad.any()):
            raise ValueError(
                f"Boolean-like column {getattr(s, 'name', '<unnamed>')!r} "
                f"contains non-0/1 values. Examples: {s.loc[bad].head(10).tolist()}"
            )
        return x.fillna(0).to_numpy(dtype=np.float64) != 0.0

    ss = s.astype("string").str.strip().str.lower()
    true_values = {"true", "t", "1", "yes", "y"}
    false_values = {"false", "f", "0", "no", "n", "", "na", "nan", "none", "null", "<na>"}

    is_true = ss.isin(true_values).fillna(False)
    is_false = (ss.isin(false_values) | ss.isna()).fillna(False)
    bad = ~(is_true | is_false)
    if bool(bad.any()):
        raise ValueError(
            f"Boolean-like column {getattr(s, 'name', '<unnamed>')!r} "
            f"contains invalid string values. Examples: {s.loc[bad].head(10).tolist()}"
        )
    return is_true.to_numpy(dtype=bool)


def _assert_direct_outputs_have_fold_local_supervision(
    *,
    fold_name: str,
    train_gene_names: Sequence[str],
    train_slide_gene_eligibility_df: Optional[pd.DataFrame],
) -> None:
    """Check that every output gene of the fold is eligible on at least one train slide."""
    if train_slide_gene_eligibility_df is None:
        raise RuntimeError(
            f"Direct head fold '{fold_name}' has no train_slide_gene_eligibility_df. "
            "Cannot prove that every direct output column is supervised."
        )

    edf = train_slide_gene_eligibility_df.copy()
    required = {"gene", "eligible_slide_gene"}
    missing_cols = sorted(required - set(edf.columns))
    if missing_cols:
        raise KeyError(
            f"Direct head fold '{fold_name}' eligibility table is missing columns: "
            f"{missing_cols}"
        )

    edf["gene"] = edf["gene"].astype(str)
    edf["_eligible_bool"] = _bool_array_from_series(edf["eligible_slide_gene"])
    eligible = set(edf.loc[edf["_eligible_bool"], "gene"].astype(str).tolist())

    requested = [str(g) for g in train_gene_names]
    missing = sorted(set(requested) - eligible)
    if missing:
        raise RuntimeError(
            f"Unsafe direct-head gene payload for fold '{fold_name}': "
            f"{len(missing)} output gene(s) have no eligible train-stripe supervision. "
            f"Examples: {missing[:20]}. Re-run settings fill with fold-local direct gene payloads."
        )


def _mask_ineligible_train_slide_gene_pairs(
    *,
    Y: np.ndarray,
    slide_id: str,
    gene_list: Sequence[str],
    train_slide_gene_eligibility_df: Optional[pd.DataFrame],
) -> np.ndarray:
    """Set to NaN the targets of the (slide, gene) pairs that are not eligible (train slides only)."""
    if train_slide_gene_eligibility_df is None:
        return Y

    edf = train_slide_gene_eligibility_df.copy()
    if edf.empty:
        raise RuntimeError(
            f"Training slide-gene eligibility table is empty while materializing train slide '{slide_id}'."
        )

    required = {"slide_id", "gene", "eligible_slide_gene"}
    missing = sorted(required - set(edf.columns))
    if missing:
        raise KeyError(
            f"train_slide_gene_eligibility_df is missing required columns: {missing}"
        )

    edf["slide_id"] = edf["slide_id"].astype(str)
    edf["gene"] = edf["gene"].astype(str)
    edf["_eligible_bool"] = _bool_array_from_series(edf["eligible_slide_gene"])

    sub = edf.loc[edf["slide_id"].eq(str(slide_id))].copy()
    if sub.empty:
        raise RuntimeError(
            f"No persisted slide-gene eligibility rows found for training slide '{slide_id}'. "
            "This would mask all genes; check the fold gene payload."
        )

    eligible_genes_for_slide = set(
        sub.loc[sub["_eligible_bool"], "gene"].astype(str).tolist()
    )
    gene_ok = np.asarray(
        [str(g) in eligible_genes_for_slide for g in gene_list],
        dtype=bool,
    )

    out = np.asarray(Y, dtype=np.float32).copy()
    out[:, ~gene_ok] = np.nan

    if not np.any(np.isfinite(out)):
        raise RuntimeError(
            f"Eligibility masking removed all measured training targets for slide '{slide_id}'. "
            "Relax per-slide-gene thresholds or drop/inspect this slide."
        )
    return out


def _attach_fold_metadata_to_slide_tensor(
    base: SlideTensorData,
    *,
    train_gene_support_df: Optional[pd.DataFrame],
    train_slide_gene_eligibility_df: Optional[pd.DataFrame],
    gene_list: Sequence[str],
) -> SlideTensorData:
    Y = _mask_ineligible_train_slide_gene_pairs(
        Y=base.Y,
        slide_id=base.slide_id,
        gene_list=gene_list,
        train_slide_gene_eligibility_df=train_slide_gene_eligibility_df,
    )
    return replace(
        base,
        Y=Y,
        train_gene_support=(
            train_gene_support_df.copy()
            if train_gene_support_df is not None
            else None
        ),
    )


class SlideTensorCache:
    """In-process cache of SlideTensorData (the fold-specific support is attached outside the cache)."""

    def __init__(self) -> None:
        self._cache: Dict[tuple, SlideTensorData] = {}
        self.hits = 0
        self.misses = 0

    def make_key(
        self,
        *,
        slide_id: str,
        level: str,
        encoder_space: str,
        training_num: str,
        checkpoint_num: str,
        row_keys: Sequence[object],
        gene_list: Sequence[object],
        within_split_allowed_patch_ids: Optional[Sequence[str]],
        use_coords_only: bool,
    ) -> tuple:
        patch_scope = (
            "main_assignment"
            if within_split_allowed_patch_ids is None
            else _fingerprint_sequence(sorted(map(str, within_split_allowed_patch_ids)))
        )
        return (
            str(slide_id),
            str(level),
            str(encoder_space),
            str(training_num),
            str(checkpoint_num),
            bool(use_coords_only),
            _fingerprint_sequence(row_keys),
            _fingerprint_sequence(gene_list),
            patch_scope,
        )

    def get(self, key: tuple) -> Optional[SlideTensorData]:
        out = self._cache.get(key)
        if out is None:
            self.misses += 1
        else:
            self.hits += 1
        return out

    def put(self, key: tuple, value: SlideTensorData) -> None:
        self._cache[key] = value

    def attach_support(
        self,
        base: SlideTensorData,
        train_gene_support_df: Optional[pd.DataFrame],
        train_slide_gene_eligibility_df: Optional[pd.DataFrame],
        gene_list: Sequence[str],
    ) -> SlideTensorData:
        return _attach_fold_metadata_to_slide_tensor(
            base,
            train_gene_support_df=train_gene_support_df,
            train_slide_gene_eligibility_df=train_slide_gene_eligibility_df,
            gene_list=gene_list,
        )



def _check_encoder_modelgroup(encoder_path: Path, expected_modelgroup: str) -> None:
    enc = load_exported_he_encoder(str(encoder_path), device=torch.device("cpu"))
    expected_group = normalize_modelgroup_name(expected_modelgroup)
    actual_group = normalize_modelgroup_name(infer_encoder_modelgroup(enc))
    if actual_group != expected_group:
        backbone = str(getattr(enc, "backbone", "")).lower()
        resize_size = int(getattr(enc, "resize_size", 0))
        crop_size = int(getattr(enc, "crop_size", 0))
        raise RuntimeError(
            f"Encoder '{encoder_path.name}' is not compatible with eval modelgroup '{expected_group}'. "
            f"Detected modelgroup='{actual_group}' from backbone='{backbone}', resize_size={resize_size}, crop_size={crop_size}."
        )



def _check_requested_space_available(
    embeddings_dir: Path,
    *,
    level: str,
    space: str,
    sub_eval_configuration: str,
    required_slide_ids: Sequence[str],
) -> None:
    if sub_eval_configuration == "STHELAR_TRAIN_COORDS":
        return
    required_slide_ids = sorted(dict.fromkeys(map(str, required_slide_ids)))
    if not required_slide_ids:
        raise RuntimeError("The protocol row manifest contains no slides.")

    if level != "patch_all":
        raise ValueError(f"Task A.1 evaluates level='patch_all' only, got {level!r}.")
    dset_name = "patch_emb_he" if space in {"he", "he_l2"} else "patch_emb_he_clip"

    missing_files = []
    zero_width_files = []
    missing_dataset_files = []
    bad_shape_files = []

    for slide_id in required_slide_ids:
        sample = embeddings_dir / f"{slide_id}.h5"
        if not sample.exists():
            missing_files.append(sample.name)
            continue
        with h5py.File(sample, "r") as f:
            if dset_name not in f:
                missing_dataset_files.append(sample.name)
                continue
            if len(f[dset_name].shape) != 2:
                bad_shape_files.append(f"{sample.name}:{tuple(f[dset_name].shape)}")
                continue
            if f[dset_name].shape[1] == 0:
                zero_width_files.append(sample.name)
                continue

    if missing_files or missing_dataset_files or zero_width_files or bad_shape_files:
        msg = (
            f"Requested evaluation space='{space}' for level='{level}', but no compatible "
            f"'{dset_name}' dataset is available for the full protocol slide universe under '{embeddings_dir}'."
        )
        if missing_files:
            msg += f" Missing embedding files: {missing_files[:10]}."
        if missing_dataset_files:
            msg += f" Files missing the dataset: {missing_dataset_files[:5]}."
        if zero_width_files:
            msg += f" Files with zero-width dataset: {zero_width_files[:5]}."
        if bad_shape_files:
            msg += f" Files with invalid dataset shape: {bad_shape_files[:5]}."
        raise RuntimeError(msg)


def _load_rows_all(eval_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(eval_dir / "settings" / "rows" / "all_rows.parquet")


def _required_slide_ids_from_persisted_protocol(
    *,
    protocol_eval_dir: Path,
    cfg: Mapping,
    rows_df: pd.DataFrame,
) -> List[str]:
    """Slide IDs used by the runnable folds (for HEST CV, can be fewer than in all_rows.parquet)."""
    sub_cfg = str(cfg["protocol"]["sub_eval_configuration"])

    if sub_cfg == "STHELAR_TEST_ADAPTED":
        area_path = protocol_eval_dir / "settings" / "area_splits" / "area_split_manifest.json"
        if area_path.exists():
            area_manifest = load_json(area_path)
            slide_ids = [
                str(entry["slide_id"])
                for entry in area_manifest.get("slides", [])
                if "slide_id" in entry
            ]
            if slide_ids:
                return sorted(dict.fromkeys(slide_ids))

    splits_path = protocol_eval_dir / "settings" / "splits" / "splits_manifest.json"
    if not splits_path.exists():
        return sorted(rows_df["slide_id"].astype(str).unique().tolist())

    splits_manifest = load_json(splits_path)
    slide_ids: List[str] = []

    if splits_manifest.get("type") == "slide_cv":
        for fold in splits_manifest.get("folds", []):
            for key in ["train_slides", "val_slides", "test_slides"]:
                slide_ids.extend(map(str, fold.get(key, [])))
    elif splits_manifest.get("type") == "one_shot":
        for key in ["train_slides", "val_slides", "test_slides"]:
            slide_ids.extend(map(str, splits_manifest.get(key, [])))
    elif splits_manifest.get("type") == "within_slide_area":
        slide_ids.extend(map(str, splits_manifest.get("slides", [])))

    if not slide_ids:
        return sorted(rows_df["slide_id"].astype(str).unique().tolist())

    return sorted(dict.fromkeys(slide_ids))


def _validate_expected_measured_genes(
    *,
    slide_id: str,
    gene_list: Sequence[str],
    Y: np.ndarray,
    expected_measured_genes: Sequence[str],
) -> None:
    expected = sorted(dict.fromkeys(map(str, expected_measured_genes)))
    observed = sorted(
        str(gene_list[j])
        for j in np.flatnonzero(np.any(~np.isnan(Y), axis=0))
    )
    if observed == expected:
        return

    expected_set = set(expected)
    observed_set = set(observed)
    missing = sorted(expected_set - observed_set)[:10]
    unexpected = sorted(observed_set - expected_set)[:10]
    raise RuntimeError(
        f"Persisted eval gene panel mismatch for slide '{slide_id}'. "
        f"Expected {len(expected)} measured genes and observed {len(observed)} after target extraction. "
        f"Missing examples: {missing}. Unexpected examples: {unexpected}."
    )


def _normalized_coords_features(
    rows_sub_df: pd.DataFrame, ref_df: pd.DataFrame
) -> np.ndarray:
    """Features of the COORDS control: (x, y) min-max normalized to [0, 1] with the extent of ref_df (whole slide)."""
    xy = rows_sub_df[["x", "y"]].to_numpy(dtype=np.float32)
    x_all = ref_df["x"].to_numpy(dtype=np.float32)
    y_all = ref_df["y"].to_numpy(dtype=np.float32)
    x_min, y_min = float(np.min(x_all)), float(np.min(y_all))
    xden = max(float(np.max(x_all)) - x_min, 1e-6)
    yden = max(float(np.max(y_all)) - y_min, 1e-6)
    return np.stack(
        [(xy[:, 0] - x_min) / xden, (xy[:, 1] - y_min) / yden], axis=1
    ).astype(np.float32)


def _slide_tensor_from_rows(
    *,
    cfg: Mapping,
    eval_dir: Path,
    slide_record,
    rows_sub_df: pd.DataFrame,
    gene_list: Sequence[str],
    encoder_space: str,
    training_num: str,
    checkpoint_num: str,
    within_split_allowed_patch_ids: Optional[Sequence[str]] = None,
    slide_rows_all_df: Optional[pd.DataFrame] = None,
    train_gene_support_df: Optional[pd.DataFrame] = None,
    train_slide_gene_eligibility_df: Optional[pd.DataFrame] = None,
    expected_measured_genes: Optional[Sequence[str]] = None,
    zscore_stats: Optional[pd.DataFrame] = None,
    tensor_cache: Optional[SlideTensorCache] = None,
) -> SlideTensorData:
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    level = str(cfg["protocol"]["level"])

    if rows_sub_df.empty:
        raise RuntimeError(
            f"Cannot materialize slide tensor for slide '{slide_record.slide_id}': "
            "the requested row subset is empty. This indicates an inconsistent split, "
            "area stripe, row manifest, or cell-token membership artifact."
        )

    eval_dataset_root = resolve_eval_dataset_root(training_dataset_id)
    emb_dir = resolve_he_embeddings_dir(training_dataset_id, training_num, checkpoint_num)
    emb_h5 = emb_dir / f"{slide_record.slide_id}.h5"
    adata_path = eval_dataset_root / "st" / f"{slide_record.slide_id}.h5ad"
    mouse_map = eval_dataset_root / "mouse_to_human_gene_mappings" / f"{slide_record.slide_id}_mouse2human_mapping.parquet"
    row_keys = rows_sub_df["row_key"].astype(str).to_numpy(dtype=object)

    use_coords_only = str(cfg["protocol"]["sub_eval_configuration"]) == "STHELAR_TRAIN_COORDS"

    cache_key = None
    if tensor_cache is not None:
        cache_key = tensor_cache.make_key(
            slide_id=slide_record.slide_id,
            level=level,
            encoder_space=encoder_space,
            training_num=training_num,
            checkpoint_num=checkpoint_num,
            row_keys=row_keys,
            gene_list=gene_list,
            within_split_allowed_patch_ids=within_split_allowed_patch_ids,
            use_coords_only=use_coords_only,
        )
        cached = tensor_cache.get(cache_key)
        if cached is not None:
            if expected_measured_genes is not None:
                _validate_expected_measured_genes(
                    slide_id=slide_record.slide_id,
                    gene_list=gene_list,
                    Y=cached.Y,
                    expected_measured_genes=expected_measured_genes,
                )
            return tensor_cache.attach_support(
                cached,
                train_gene_support_df,
                train_slide_gene_eligibility_df,
                gene_list,
            )

    if use_coords_only:
        ref_df = rows_sub_df if slide_rows_all_df is None else slide_rows_all_df
        X = _normalized_coords_features(rows_sub_df, ref_df)
    elif level.startswith("patch"):
        patch_ids = rows_sub_df["patch_id"].astype(str).to_numpy(dtype=object)
        X = load_patch_embeddings(emb_h5, patch_ids, encoder_space)
    elif level == "cell":
        cell_ids = rows_sub_df["cell_id_int"].to_numpy(dtype=np.int32)
        if within_split_allowed_patch_ids is None:
            main_assign = emb_dir / "main_HEemb_cellID" / f"{slide_record.slide_id}.parquet"
            X = load_cell_embeddings_via_main_assignment(emb_h5, main_assign, cell_ids, encoder_space)
        else:
            training_dataset_dir = resolve_training_dataset_dir(str(cfg["identity"]["dataset_id"]), training_dataset_id)
            slide_patch_h5 = eval_dataset_root / "patches" / f"{slide_record.slide_id}.h5"
            centroid_parquet = training_dataset_dir / "nucleus_centroids" / f"nucleus_centroids_{slide_record.slide_id}.parquet"
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
                space=encoder_space,
            )
    else:
        raise ValueError(level)

    Y, detected = extract_targets_for_rows(
        adata_path=adata_path,
        slide_record=slide_record,
        level=level,
        ordered_row_keys=row_keys,
        ordered_gene_list=gene_list,
        target_transform=cfg["protocol"]["target_transform"],
        mouse_mapping_path=mouse_map if mouse_map.exists() else None,
        zscore_stats=zscore_stats,
    )
    if expected_measured_genes is not None:
        _validate_expected_measured_genes(
            slide_id=slide_record.slide_id,
            gene_list=gene_list,
            Y=Y,
            expected_measured_genes=expected_measured_genes,
        )

    # X must be finite; Y may contain NaN (unmeasured) but no inf
    if not np.isfinite(X).all():
        bad_rows = np.flatnonzero(~np.isfinite(X).all(axis=1))
        raise RuntimeError(
            f"Non-finite embeddings detected for slide '{slide_record.slide_id}'. "
            f"X.shape={X.shape}, bad_rows={len(bad_rows)}, "
            f"first_bad_rows={bad_rows[:10].tolist()}"
        )
    ok_y = np.isfinite(Y) | np.isnan(Y)
    if not np.all(ok_y):
        bad = np.argwhere(~ok_y)[:10].tolist()
        raise RuntimeError(
            f"Invalid targets detected for slide '{slide_record.slide_id}'. "
            f"Y.shape={Y.shape}, first_bad_indices={bad}"
        )

    cell_types = None
    if level == "cell" and "cell_type" in rows_sub_df.columns:
        ct = rows_sub_df["cell_type"].astype("string")
        ct = ct.fillna("unknown")
        ct = ct.str.strip()
        ct = ct.mask(ct.eq(""), "unknown")
        cell_types = ct.to_numpy(dtype=object)

        if cell_types.shape[0] != rows_sub_df.shape[0]:
            raise RuntimeError(
                f"cell_types length mismatch for slide '{slide_record.slide_id}': "
                f"{cell_types.shape[0]} vs {rows_sub_df.shape[0]}"
            )

    # Y is cached unmasked (fold-specific eligibility masking is applied on return)
    base = SlideTensorData(
        slide_id=slide_record.slide_id,
        subset=slide_record.subset,
        source=slide_record.source,
        tissue=slide_record.tissue,
        X=X.astype(np.float32, copy=False),
        Y=Y.astype(np.float32, copy=False),
        row_keys=row_keys,
        coords_xy=rows_sub_df[["x", "y"]].to_numpy(dtype=np.float32),
        detected=detected,
        cell_types=cell_types,
        obs_df=rows_sub_df.reset_index(drop=True).copy(),
        train_gene_support=None,
    )

    if tensor_cache is not None and cache_key is not None:
        tensor_cache.put(cache_key, base)

    return _attach_fold_metadata_to_slide_tensor(
        base,
        train_gene_support_df=train_gene_support_df,
        train_slide_gene_eligibility_df=train_slide_gene_eligibility_df,
        gene_list=gene_list,
    )



def _build_fold_data_slide_protocol(
    *,
    cfg: Mapping,
    eval_dir: Path,
    rows_df: pd.DataFrame,
    split_payload: Mapping,
    gene_payload: Mapping,
    training_num: str,
    checkpoint_num: str,
    encoder_space: str,
    zscore_stats: Optional[pd.DataFrame] = None,
    tensor_cache: Optional[SlideTensorCache] = None,
) -> FoldData:
    slide_records = load_slide_records(str(cfg["identity"]["training_dataset_id"]))
    train_gene_names = np.asarray(gene_payload.get("train_gene_names", gene_payload.get("trainable_genes", [])), dtype=object)

    # Test slides use the train-gene axis (NaN for unmeasured genes)
    model_test_gene_names = train_gene_names

    train_gene_support_df = None
    if "train_gene_support_csv" in gene_payload:
        train_gene_support_df = pd.read_csv(gene_payload["train_gene_support_csv"])
    train_slide_gene_eligibility_df = None
    if str(gene_payload.get("train_slide_gene_eligibility_csv", "") or ""):
        train_slide_gene_eligibility_df = pd.read_csv(gene_payload["train_slide_gene_eligibility_csv"])

    _assert_direct_outputs_have_fold_local_supervision(
        fold_name=str(split_payload.get("fold_name", f"fold_{split_payload.get('fold_id', 'oneshot')}")),
        train_gene_names=train_gene_names,
        train_slide_gene_eligibility_df=train_slide_gene_eligibility_df,
    )

    eval_genes_by_slide = {
        str(sid): [str(g) for g in genes]
        for sid, genes in gene_payload.get("eval_genes_by_slide", {}).items()
    }

    train_slides, test_slides = [], []
    for sid in split_payload.get("train_slides", []):
        rows_sub = rows_df.loc[rows_df["slide_id"] == sid].copy()
        train_slides.append(
            _slide_tensor_from_rows(
                cfg=cfg,
                eval_dir=eval_dir,
                slide_record=slide_records[sid],
                rows_sub_df=rows_sub,
                gene_list=train_gene_names,
                encoder_space=encoder_space,
                training_num=training_num,
                checkpoint_num=checkpoint_num,
                slide_rows_all_df=rows_df.loc[rows_df["slide_id"] == sid].copy(),
                train_gene_support_df=train_gene_support_df,
                train_slide_gene_eligibility_df=train_slide_gene_eligibility_df,
                zscore_stats=zscore_stats,
                tensor_cache=tensor_cache,
            )
        )
    for sid in split_payload.get("test_slides", []):
        rows_sub = rows_df.loc[rows_df["slide_id"] == sid].copy()
        test_slides.append(
            _slide_tensor_from_rows(
                cfg=cfg,
                eval_dir=eval_dir,
                slide_record=slide_records[sid],
                rows_sub_df=rows_sub,
                gene_list=model_test_gene_names,
                encoder_space=encoder_space,
                training_num=training_num,
                checkpoint_num=checkpoint_num,
                slide_rows_all_df=rows_df.loc[rows_df["slide_id"] == sid].copy(),
                train_gene_support_df=train_gene_support_df,
                expected_measured_genes=eval_genes_by_slide.get(str(sid)),
                zscore_stats=zscore_stats,
                tensor_cache=tensor_cache,
            )
        )
    return FoldData(
        fold_name=str(split_payload.get("fold_name", f"fold_{split_payload.get('fold_id', 'oneshot')}")),
        train_slides=train_slides,
        val_slides=[],
        test_slides=test_slides,
        train_gene_names=train_gene_names,
        test_gene_names=model_test_gene_names,
    )



# -----------------------------------------------------------------------------
# Signature target axis (target_basis="signatures")
# Member-gene targets aggregated (Sum w*z / Sum|w|) into per-signature columns, used as the "gene" axis
# -----------------------------------------------------------------------------

_SIG_NAME_SEP = SIGNATURE_AXIS_SEP


def _signature_axis_name(signature_set: str, signature_id: str) -> str:
    """Stable, parseable axis label for a signature (used as a 'gene' name)."""
    return f"{signature_set}{_SIG_NAME_SEP}{signature_id}"


def _signature_eligible_columns(
    slide_id: str,
    sig_keys: Sequence[Tuple[str, str]],
    slide_eligibility: Optional[pd.DataFrame],
) -> Optional[np.ndarray]:
    """Mask of the signatures eligible on this slide (None without eligibility table; absent signatures are ineligible)."""
    if slide_eligibility is None or slide_eligibility.empty:
        return None
    sub = slide_eligibility.loc[
        slide_eligibility["slide_id"].astype(str) == str(slide_id)
    ]
    ok = {
        (str(s), str(i))
        for s, i, e in zip(sub["signature_set"], sub["signature_id"], sub["eligible"])
        if bool(e)
    }
    return np.asarray([(str(a), str(b)) in ok for a, b in sig_keys], dtype=bool)


def _signaturize_fold_data(
    fold: FoldData,
    targets_df: pd.DataFrame,
    slide_eligibility: Optional[pd.DataFrame] = None,
) -> FoldData:
    """
    Turn a member-gene FoldData into a signature-axis FoldData (per-signature scores, signature names as axis).
    Signature columns not eligible on a slide (slide_eligibility) are set to NaN.
    """
    member = [str(g) for g in np.asarray(fold.train_gene_names, dtype=object).tolist()]
    axis_names: Optional[np.ndarray] = None

    def _convert(slides: Sequence[SlideTensorData]) -> List[SlideTensorData]:
        nonlocal axis_names
        out: List[SlideTensorData] = []
        for s in slides:
            F = np.isfinite(s.Y)
            Y_sig, sig_keys, n_present = aggregate_signature_targets(s.Y, F, member, targets_df)
            names = np.asarray(
                [_signature_axis_name(a, b) for a, b in sig_keys], dtype=object
            )
            det = n_present > 0
            keep = _signature_eligible_columns(s.slide_id, sig_keys, slide_eligibility)
            if keep is not None and not bool(np.all(keep)):
                Y_sig = np.array(Y_sig, dtype=np.float64, copy=True)
                Y_sig[:, ~keep] = np.nan
                det = np.array(det, dtype=bool, copy=True)
                det[:, ~keep] = False
            if axis_names is None:
                axis_names = names
            out.append(
                replace(
                    s,
                    Y=Y_sig.astype(np.float32, copy=False),
                    # Kept for shape compatibility (no on/off state on signatures)
                    detected=det,
                    train_gene_support=None,
                )
            )
        return out

    train = _convert(fold.train_slides)
    val = _convert(fold.val_slides)
    test = _convert(fold.test_slides)
    if axis_names is None:
        raise RuntimeError(
            f"Cannot signaturize fold '{fold.fold_name}': it has no slides."
        )
    return replace(
        fold,
        train_slides=train,
        val_slides=val,
        test_slides=test,
        train_gene_names=axis_names,
        test_gene_names=axis_names,
    )


def _signature_reliability_gene_axis(reliability_signature_df: pd.DataFrame) -> pd.DataFrame:
    """Signature reliability table in the per-(slide, gene) format of compute_benchmark_metrics (signature name as gene)."""
    df = reliability_signature_df.copy()
    df["gene"] = [
        _signature_axis_name(str(s), str(i))
        for s, i in zip(df["signature_set"], df["signature_id"])
    ]
    keep = [
        "slide_id", "gene", "r_pearson", "r_spearman",
        "r_pearson_halfsplit", "r_spearman_halfsplit", "n_locations",
    ]
    return df[[c for c in keep if c in df.columns]]


def _build_signature_fold_data(
    *,
    cfg: Mapping,
    eval_dir: Path,
    rows_df: pd.DataFrame,
    split_payload: Mapping,
    member_gene_names: Sequence[str],
    targets_df: pd.DataFrame,
    training_num: str,
    checkpoint_num: str,
    encoder_space: str,
    zscore_stats: Optional[pd.DataFrame] = None,
    tensor_cache: Optional[SlideTensorCache] = None,
    slide_eligibility: Optional[pd.DataFrame] = None,
) -> FoldData:
    """Build a fold on the signature axis: materialize per-member-gene bundles
    (shared covered member union across all folds), then aggregate to signatures."""
    slide_records = load_slide_records(str(cfg["identity"]["training_dataset_id"]))
    gene_list = np.asarray([str(g) for g in member_gene_names], dtype=object)

    def _slides_for(key: str) -> List[SlideTensorData]:
        out: List[SlideTensorData] = []
        for sid in split_payload.get(key, []):
            rows_sub = rows_df.loc[rows_df["slide_id"] == sid].copy()
            out.append(
                _slide_tensor_from_rows(
                    cfg=cfg,
                    eval_dir=eval_dir,
                    slide_record=slide_records[sid],
                    rows_sub_df=rows_sub,
                    gene_list=gene_list,
                    encoder_space=encoder_space,
                    training_num=training_num,
                    checkpoint_num=checkpoint_num,
                    slide_rows_all_df=rows_df.loc[rows_df["slide_id"] == sid].copy(),
                    train_gene_support_df=None,
                    train_slide_gene_eligibility_df=None,
                    expected_measured_genes=None,
                    zscore_stats=zscore_stats,
                    tensor_cache=tensor_cache,
                )
            )
        return out

    member_fold = FoldData(
        fold_name=str(split_payload.get("fold_name", f"fold_{split_payload.get('fold_id', 'oneshot')}")),
        train_slides=_slides_for("train_slides"),
        val_slides=_slides_for("val_slides"),
        test_slides=_slides_for("test_slides"),
        train_gene_names=gene_list,
        test_gene_names=gene_list,
    )
    return _signaturize_fold_data(member_fold, targets_df, slide_eligibility)


def _signature_folds_from_within_slide_protocol(
    *,
    cfg: Mapping,
    eval_dir: Path,
    rows_df: pd.DataFrame,
    member_gene_names: Sequence[str],
    targets_df: pd.DataFrame,
    training_num: str,
    checkpoint_num: str,
    encoder_space: str,
    zscore_stats: Optional[pd.DataFrame] = None,
    tensor_cache: Optional[SlideTensorCache] = None,
    slide_eligibility: Optional[pd.DataFrame] = None,
) -> List[FoldData]:
    """Signature-axis version of _folds_from_within_slide_protocol (STHELAR_TEST_ADAPTED), on the member-gene union."""
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    slide_records = load_slide_records(training_dataset_id)
    area_manifest = load_json(eval_dir / "settings" / "area_splits" / "area_split_manifest.json")
    gene_list = np.asarray([str(g) for g in member_gene_names], dtype=object)
    level = str(cfg["protocol"]["level"])
    folds: List[FoldData] = []

    for slide_entry in area_manifest["slides"]:
        sid = slide_entry["slide_id"]
        patch_df = pd.read_parquet(slide_entry["patch_manifest"])
        patch_df["patch_id"] = patch_df["patch_id"].astype(str)
        patch_stripe = dict(zip(patch_df["patch_id"], patch_df["stripe_id"].astype(int)))
        cell_df = pd.read_parquet(slide_entry["cell_manifest"]) if level == "cell" else None
        if level == "cell":
            rows_slide = rows_df.loc[
                (rows_df["slide_id"] == sid)
                & rows_df["cell_id_int"].isin(cell_df.loc[~cell_df["is_ambiguous"], "cell_id_int"])
            ].copy()
        else:
            rows_slide = rows_df.loc[
                (rows_df["slide_id"] == sid)
                & rows_df["patch_id"].astype(str).isin(
                    patch_df.loc[patch_df["stripe_id"] >= 0, "patch_id"].astype(str)
                )
            ].copy()
        slide_rows_full_df = rows_df.loc[rows_df["slide_id"] == sid].copy()

        for fold_idx, roles in enumerate(_ADAPTED_STRIPE_ROLES):
            stripes_by_role = {
                role: sorted(st for st, r in roles.items() if r == role)
                for role in ("train", "test")
            }

            def _mk(stripe_id: int) -> SlideTensorData:
                patch_ids = [pid for pid, st in patch_stripe.items() if int(st) == int(stripe_id)]
                if level == "cell":
                    rows_sub = rows_slide.loc[
                        rows_slide["cell_id_int"].isin(
                            cell_df.loc[cell_df["stripe_id"] == int(stripe_id), "cell_id_int"]
                        )
                    ].copy()
                    allowed = patch_ids
                    ref = rows_slide
                else:
                    rows_sub = rows_slide.loc[
                        rows_slide["patch_id"].astype(str).isin(patch_ids)
                    ].copy()
                    allowed = None
                    ref = slide_rows_full_df
                return _slide_tensor_from_rows(
                    cfg=cfg, eval_dir=eval_dir, slide_record=slide_records[sid],
                    rows_sub_df=rows_sub, gene_list=gene_list, encoder_space=encoder_space,
                    training_num=training_num, checkpoint_num=checkpoint_num,
                    within_split_allowed_patch_ids=allowed, slide_rows_all_df=ref,
                    train_gene_support_df=None, train_slide_gene_eligibility_df=None,
                    expected_measured_genes=None, zscore_stats=zscore_stats,
                    tensor_cache=tensor_cache,
                )

            member_fold = FoldData(
                fold_name=f"{sid}_fold_{fold_idx}",
                # One entry per train stripe (alpha-selection group)
                train_slides=[_mk(st) for st in stripes_by_role["train"]],
                val_slides=[],
                test_slides=[_mk(st) for st in stripes_by_role["test"]],
                train_gene_names=gene_list, test_gene_names=gene_list,
                alpha_group="stripe",
            )
            folds.append(
                _signaturize_fold_data(member_fold, targets_df, slide_eligibility)
            )
    return folds


def _folds_from_within_slide_protocol(
    *,
    cfg: Mapping,
    eval_dir: Path,
    rows_df: pd.DataFrame,
    training_num: str,
    checkpoint_num: str,
    encoder_space: str,
    zscore_stats: Optional[pd.DataFrame] = None,
    tensor_cache: Optional[SlideTensorCache] = None,
) -> List[FoldData]:
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    within_payload = load_json(eval_dir / "settings" / "genes" / "within_slide_genes.json")
    fold_payloads = {
        str(p["fold_name"]): p
        for p in within_payload.get("folds", [])
    }
    legacy_payload = None if fold_payloads else within_payload

    is_gbase = str(cfg["protocol"].get("G_base_id", "")).strip().lower() != "auto"

    if is_gbase and not fold_payloads:
        raise RuntimeError(
            "STHELAR_TEST_ADAPTED + G_base: within_slide_genes.json has no per-fold "
            "gene payloads. Re-run _2_fill_eval_settings.py or create a new eval_id."
        )

    def _materialize_gene_payload(payload: Mapping) -> Dict[str, object]:
        train_gene_names = np.asarray(
            payload.get("train_gene_names", payload.get("trainable_genes", [])),
            dtype=object,
        )
        test_gene_names = np.asarray(
            payload.get("test_gene_names", train_gene_names.tolist()),
            dtype=object,
        )
        if train_gene_names.size == 0 or test_gene_names.size == 0:
            raise RuntimeError(
                f"Within-slide gene payload for fold '{payload.get('fold_name', 'legacy')}' "
                "contains an empty train/test gene list."
            )
        model_test_gene_names = train_gene_names

        train_gene_support_df = None
        if str(payload.get("train_gene_support_csv", "") or ""):
            train_gene_support_df = pd.read_csv(payload["train_gene_support_csv"])
        train_slide_gene_eligibility_df = None
        if str(payload.get("train_slide_gene_eligibility_csv", "") or ""):
            train_slide_gene_eligibility_df = pd.read_csv(payload["train_slide_gene_eligibility_csv"])
        eval_genes_by_slide = {
            str(sid): [str(g) for g in genes]
            for sid, genes in payload.get("eval_genes_by_slide", {}).items()
        }
        return {
            "train_gene_names": train_gene_names,
            "test_gene_names": model_test_gene_names,
            "train_gene_support_df": train_gene_support_df,
            "train_slide_gene_eligibility_df": train_slide_gene_eligibility_df,
            "eval_genes_by_slide": eval_genes_by_slide,
        }

    slide_records = load_slide_records(training_dataset_id)
    area_manifest = load_json(eval_dir / "settings" / "area_splits" / "area_split_manifest.json")
    folds: List[FoldData] = []

    for slide_entry in area_manifest["slides"]:
        sid = slide_entry["slide_id"]
        patch_df = pd.read_parquet(slide_entry["patch_manifest"])
        patch_df["patch_id"] = patch_df["patch_id"].astype(str)
        patch_stripe = dict(zip(patch_df["patch_id"], patch_df["stripe_id"].astype(int)))
        if str(cfg["protocol"]["level"]) == "cell":
            cell_df = pd.read_parquet(slide_entry["cell_manifest"])
            rows_slide = rows_df.loc[(rows_df["slide_id"] == sid) & rows_df["cell_id_int"].isin(cell_df.loc[~cell_df["is_ambiguous"], "cell_id_int"])].copy()
        else:
            rows_slide = rows_df.loc[(rows_df["slide_id"] == sid) & rows_df["patch_id"].astype(str).isin(patch_df.loc[patch_df["stripe_id"] >= 0, "patch_id"].astype(str))].copy()

        for fold_idx, roles in enumerate(_ADAPTED_STRIPE_ROLES):
            train_stripes = sorted(st for st, r in roles.items() if r == "train")
            test_stripes = sorted(st for st, r in roles.items() if r == "test")
            test_patch_ids = [pid for pid, stripe in patch_stripe.items() if roles.get(int(stripe)) == "test"]

            fold_name = f"{sid}_fold_{fold_idx}"
            payload = fold_payloads.get(fold_name, legacy_payload)
            if payload is None:
                raise RuntimeError(f"Missing within-slide gene payload for fold_name='{fold_name}'.")
            g = _materialize_gene_payload(payload)

            train_gene_names = g["train_gene_names"]
            test_gene_names = g["test_gene_names"]
            train_gene_support_df = g["train_gene_support_df"]
            train_slide_gene_eligibility_df = g["train_slide_gene_eligibility_df"]
            eval_genes_by_slide = g["eval_genes_by_slide"]

            train_eligibility_this_fold = train_slide_gene_eligibility_df
            if train_eligibility_this_fold is not None and "fold_name" in train_eligibility_this_fold.columns:
                train_eligibility_this_fold = train_eligibility_this_fold.loc[
                    train_eligibility_this_fold["fold_name"].astype(str).eq(fold_name)
                ].copy()
                if train_eligibility_this_fold.empty:
                    raise RuntimeError(f"No within-slide train eligibility rows found for {fold_name}.")

            _assert_direct_outputs_have_fold_local_supervision(
                fold_name=fold_name,
                train_gene_names=train_gene_names,
                train_slide_gene_eligibility_df=train_eligibility_this_fold,
            )

            is_cell = str(cfg["protocol"]["level"]) == "cell"
            if is_cell:
                cell_df = pd.read_parquet(slide_entry["cell_manifest"])
            slide_rows_full_df = rows_df.loc[rows_df["slide_id"] == sid].copy()

            def _stripe_tensor(stripe_id: int, gene_names, *, is_train: bool) -> SlideTensorData:
                patch_ids = [pid for pid, st in patch_stripe.items() if int(st) == int(stripe_id)]
                if is_cell:
                    rows_sub = rows_slide.loc[
                        rows_slide["cell_id_int"].isin(
                            cell_df.loc[cell_df["stripe_id"] == int(stripe_id), "cell_id_int"]
                        )
                    ].copy()
                    allowed, ref = patch_ids, rows_slide
                else:
                    rows_sub = rows_slide.loc[
                        rows_slide["patch_id"].astype(str).isin(patch_ids)
                    ].copy()
                    allowed, ref = None, slide_rows_full_df
                return _slide_tensor_from_rows(
                    cfg=cfg, eval_dir=eval_dir, slide_record=slide_records[sid],
                    rows_sub_df=rows_sub, gene_list=gene_names, encoder_space=encoder_space,
                    training_num=training_num, checkpoint_num=checkpoint_num,
                    within_split_allowed_patch_ids=allowed, slide_rows_all_df=ref,
                    train_gene_support_df=train_gene_support_df,
                    train_slide_gene_eligibility_df=(train_eligibility_this_fold if is_train else None),
                    expected_measured_genes=eval_genes_by_slide.get(str(sid)),
                    zscore_stats=zscore_stats, tensor_cache=tensor_cache,
                )

            folds.append(
                FoldData(
                    fold_name=fold_name,
                    # One entry per train stripe: the stripe is the alpha-selection group.
                    train_slides=[_stripe_tensor(st, train_gene_names, is_train=True) for st in train_stripes],
                    val_slides=[],
                    test_slides=[_stripe_tensor(st, test_gene_names, is_train=False) for st in test_stripes],
                    train_gene_names=train_gene_names,
                    test_gene_names=test_gene_names,
                    alpha_group="stripe",
                )
            )
    return folds


def _merge_slide_prediction_bundle(existing: Mapping, new: Mapping) -> Dict[str, object]:
    if existing["slide_id"] != new["slide_id"]:
        raise RuntimeError("Cannot merge predictions from different slides.")

    old_genes = [str(g) for g in np.asarray(existing["gene_names"], dtype=object).tolist()]
    new_genes = [str(g) for g in np.asarray(new["gene_names"], dtype=object).tolist()]
    union_genes = sorted(dict.fromkeys(old_genes + new_genes))

    def _align(mat: np.ndarray, src_genes: Sequence[str]) -> np.ndarray:
        mat = np.asarray(mat, dtype=np.float32)
        out = np.full((mat.shape[0], len(union_genes)), np.nan, dtype=np.float32)
        dst_lut = {g: i for i, g in enumerate(union_genes)}
        src_lut = {str(g): i for i, g in enumerate(src_genes)}
        for g, src_j in src_lut.items():
            if g in dst_lut:
                out[:, dst_lut[g]] = mat[:, src_j]
        return out

    def _align_detected(mat: np.ndarray, src_genes: Sequence[str]) -> np.ndarray:
        # Same as _align, with False for the genes absent from a fold
        mat = np.asarray(mat, dtype=bool)
        out = np.zeros((mat.shape[0], len(union_genes)), dtype=bool)
        dst_lut = {g: i for i, g in enumerate(union_genes)}
        src_lut = {str(g): i for i, g in enumerate(src_genes)}
        for g, src_j in src_lut.items():
            if g in dst_lut:
                out[:, dst_lut[g]] = mat[:, src_j]
        return out

    old_keys = np.asarray(existing["row_keys"], dtype=object)
    new_keys = np.asarray(new["row_keys"], dtype=object)
    overlap = np.intersect1d(old_keys, new_keys)
    if overlap.size > 0:
        raise RuntimeError(
            f"OOF merge for slide '{existing['slide_id']}' found duplicated row_keys. "
            f"Examples: {overlap[:10].tolist()}"
        )

    merged = dict(existing)
    old_y_true = _align(np.asarray(existing["y_true"], dtype=np.float32), old_genes)
    old_y_pred = _align(np.asarray(existing["y_pred"], dtype=np.float32), old_genes)
    new_y_true = _align(np.asarray(new["y_true"], dtype=np.float32), new_genes)
    new_y_pred = _align(np.asarray(new["y_pred"], dtype=np.float32), new_genes)

    merged["gene_names"] = np.asarray(union_genes, dtype=object)
    merged["row_keys"] = np.concatenate([old_keys, new_keys], axis=0)
    merged["coords_xy"] = np.concatenate([np.asarray(existing["coords_xy"]), np.asarray(new["coords_xy"])] , axis=0)
    merged["y_true"] = np.concatenate([old_y_true, new_y_true], axis=0)
    merged["y_pred"] = np.concatenate([old_y_pred, new_y_pred], axis=0)

    # detected is aligned and concatenated like y_true/y_pred
    old_detected = _align_detected(np.asarray(existing["detected"], dtype=bool), old_genes)
    new_detected = _align_detected(np.asarray(new["detected"], dtype=bool), new_genes)
    merged["detected"] = np.concatenate([old_detected, new_detected], axis=0)

    old_ct = existing.get("cell_types", None)
    new_ct = new.get("cell_types", None)
    if old_ct is None and new_ct is None:
        merged["cell_types"] = None
    elif old_ct is not None and new_ct is not None:
        merged["cell_types"] = np.concatenate([np.asarray(old_ct, dtype=object), np.asarray(new_ct, dtype=object)], axis=0)
    else:
        raise RuntimeError(f"Inconsistent cell_types presence while merging slide '{existing['slide_id']}'.")

    old_obs = existing.get("obs_df", None)
    new_obs = new.get("obs_df", None)
    if old_obs is None and new_obs is None:
        merged["obs_df"] = None
    elif old_obs is not None and new_obs is not None:
        merged["obs_df"] = pd.concat([old_obs, new_obs], axis=0, ignore_index=True)
    else:
        raise RuntimeError(f"Inconsistent obs_df presence while merging slide '{existing['slide_id']}'.")

    # Train support (diagnostics only), merged across the folds of a slide
    old_support = existing.get("train_gene_support", None)
    new_support = new.get("train_gene_support", None)
    old_has_support = isinstance(old_support, pd.DataFrame) and not old_support.empty
    new_has_support = isinstance(new_support, pd.DataFrame) and not new_support.empty

    if old_has_support and new_has_support:
        sup = pd.concat([old_support.copy(), new_support.copy()], axis=0, ignore_index=True)
        if "gene" in sup.columns:
            sup["gene"] = sup["gene"].astype(str)
            support_counts = (
                sup.groupby("gene", dropna=False)
                .size()
                .reset_index(name="n_train_support_records_merged")
            )
            numeric_cols = [
                c for c in sup.columns
                if c != "gene" and pd.api.types.is_numeric_dtype(sup[c])
            ]
            other_cols = [
                c for c in sup.columns
                if c != "gene" and c not in numeric_cols
            ]
            agg_spec = {c: "mean" for c in numeric_cols}
            agg_spec.update({c: "first" for c in other_cols})
            sup = (
                sup.groupby("gene", dropna=False, as_index=False)
                .agg(agg_spec)
                .merge(support_counts, on="gene", how="left")
            )
        merged["train_gene_support"] = sup
    elif not old_has_support and not new_has_support:
        merged["train_gene_support"] = None
    else:
        # Partial support is dropped
        merged["train_gene_support"] = None

    order = np.argsort(merged["row_keys"].astype(str))
    merged["row_keys"] = merged["row_keys"][order]
    merged["coords_xy"] = merged["coords_xy"][order]
    merged["y_true"] = merged["y_true"][order]
    merged["y_pred"] = merged["y_pred"][order]
    merged["detected"] = merged["detected"][order]
    if merged["cell_types"] is not None:
        merged["cell_types"] = merged["cell_types"][order]
    if merged["obs_df"] is not None:
        merged["obs_df"] = merged["obs_df"].iloc[order].reset_index(drop=True)

    if merged["detected"].shape != merged["y_true"].shape:
        raise RuntimeError(
            f"Merged 'detected' shape {merged['detected'].shape} does not match "
            f"'y_true' shape {merged['y_true'].shape} for slide '{existing['slide_id']}'."
        )
    return merged



def _merge_prediction_dicts(dst: Dict[str, Mapping], src: Mapping[str, Mapping]) -> None:
    for sid, bundle in src.items():
        if sid not in dst:
            dst[sid] = bundle
        else:
            dst[sid] = _merge_slide_prediction_bundle(dst[sid], bundle)



def _numeric_only_dict(payload: Mapping[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in payload.items():
        if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v)):
            out[str(k)] = float(v)
    return out



def _flatten_summary(prefix: str, payload: Mapping[str, object]) -> Dict[str, float]:
    return {f"{prefix}{k}": float(v) for k, v in _numeric_only_dict(payload).items()}



def _aggregate_fold_outputs(encoder_dir: Path) -> None:
    """Summarize the per-fold benchmark over folds (mean, std, min, max) in aggregate_across_folds/."""
    run_dir = encoder_dir / "run"
    agg_dir = encoder_dir / "aggregate_across_folds"
    if agg_dir.exists():
        rmtree_appledouble_safe(agg_dir)
    agg_dir.mkdir(parents=True, exist_ok=True)

    path = run_dir / "fold_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty or "fold_name" not in df.columns:
        return

    df = df.sort_values("fold_name").reset_index(drop=True)
    df.to_csv(agg_dir / "per_fold_benchmark.csv", index=False)

    numeric_cols = [
        c for c in df.columns
        if c != "fold_name" and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        return

    long_df = df.melt(
        id_vars=["fold_name"],
        value_vars=numeric_cols,
        var_name="metric",
        value_name="value",
    )
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    n_folds_total = int(df["fold_name"].nunique())
    long_df = long_df.loc[np.isfinite(long_df["value"])].copy()

    fold_var = (
        long_df.groupby("metric", dropna=False)["value"]
        .agg(
            mean_over_folds="mean",
            fold_std="std",
            min_over_folds="min",
            max_over_folds="max",
            n_folds="count",
        )
        .reset_index()
    )
    # Add the metrics that are finite in no fold (n_folds = 0)
    missing = [m for m in numeric_cols if m not in set(fold_var["metric"].tolist())]
    if missing:
        fold_var = pd.concat(
            [
                fold_var,
                pd.DataFrame({
                    "metric": missing,
                    "mean_over_folds": np.nan,
                    "fold_std": np.nan,
                    "min_over_folds": np.nan,
                    "max_over_folds": np.nan,
                    "n_folds": 0,
                }),
            ],
            ignore_index=True,
        )
    fold_var = fold_var.sort_values("metric").reset_index(drop=True)
    fold_var["n_folds_total"] = n_folds_total
    fold_var["all_folds_nonfinite"] = fold_var["n_folds"] == 0
    fold_var["std_semantics"] = (
        "fold_std is the std across folds (the probe is deterministic and has no seed). "
        "n_folds counts folds where the metric was finite; n_folds == 0 means the metric "
        "was computed but never valid, not that it was skipped."
    )
    fold_var.to_csv(agg_dir / "fold_variability_benchmark.csv", index=False)



def _save_encoder_manifest(
    encoder_dir: Path,
    encoder_path: Optional[Path],
    embeddings_dir: Optional[Path],
    cfg: Mapping,
    protocol_eval_dir: Path,
    training_num: str,
    checkpoint_num: str,
    encoder_space: str,
    skip_reports: bool = False,
    skip_predictions: bool = False,
) -> None:
    payload = {
        "encoder_identity": {
            "kind": "coords_baseline" if encoder_path is None else "he_encoder",
            "training_num": str(training_num),
            "checkpoint_num": str(checkpoint_num),
            "space": str(encoder_space),
        },
        "encoder_file": (manifest_for_file(encoder_path).__dict__ if encoder_path is not None else None),
        "embeddings_dir": (str(embeddings_dir) if embeddings_dir is not None else None),
        "protocol_manifest": str(encoder_dir.parent / "settings" / "protocol_manifest.json"),
        "run_eval_dir": str(encoder_dir.parent),
        "protocol_eval_dir": str(protocol_eval_dir),
        "reference_eval_id": cfg.get("protocol", {}).get("reference_eval_id", ""),
        "control_kind": cfg.get("protocol", {}).get("control_kind", ""),
        "prediction_head": prediction_head_manifest_payload(cfg),
        "input_feature_pipeline": {
            "raw_features": "coords" if encoder_path is None else "he_encoder_embeddings",
            "encoder_space": str(encoder_space),
            # No dimensionality reduction (full-dimensional features)
            "input_reduction": "none",
            "feature_geometry": resolve_feature_geometry(cfg, encoder_space),
        },
        "probe": {
            "kind": "closed_form_weighted_ridge",
            "deterministic": True,
            "alpha_selection": dict(cfg["protocol"]["ridge"]["alpha_selection"]),
            "alpha_grid": dict(cfg["protocol"]["ridge"]["alpha_grid"]),
        },
        "wandb_mode": str(cfg.get("protocol", {}).get("runtime", {}).get("wandb_mode", "offline")),
        # Optional outputs skipped by this run (not part of the protocol)
        "outputs": {
            "skip_reports": bool(skip_reports),
            "skip_predictions": bool(skip_predictions),
            "qualitative_reports_written": not bool(skip_reports),
            "prediction_bundles_written": not bool(skip_predictions),
        },
        "environment": dump_environment_report(),
    }
    save_json(encoder_dir / "manifest.json", payload)



def _make_wandb_run(cfg: Mapping, run_dir: Path, encoder_dir: Path) -> WandbRunHandle:
    runtime = cfg.get("protocol", {}).get("runtime", {})
    enabled = bool(runtime.get("wandb_enabled", True))
    mode = str(runtime.get("wandb_mode", "offline"))
    project = str(runtime.get("wandb_project", "HistoST_A1_eval"))
    run_name = f"{cfg['identity']['eval_id']}__{encoder_dir.name}"
    return WandbRunHandle(
        enabled=enabled,
        mode=mode,
        run_dir=run_dir / "wandb",
        project=project,
        run_name=run_name,
        group=f"{cfg['identity']['eval_id']}",
        job_type="eval_probe",
        config=cfg,
        tags=[str(cfg["protocol"]["sub_eval_configuration"]), str(cfg["protocol"]["level"]), str(cfg["identity"]["training_dataset_id"])],
    )



# Sub-evals whose test slides were not used for CLIP training
_HELD_OUT_SUB_EVALS = {"STHELAR_TEST_ONE_SHOT", "STHELAR_TEST_ADAPTED"}


def _is_held_out_sub_eval(sub_eval_configuration: str) -> bool:
    """True for the sub-evals whose test slides were not used for CLIP training."""
    return str(sub_eval_configuration) in _HELD_OUT_SUB_EVALS


def _build_structure_aux_inputs(
    *,
    cfg: Mapping,
    protocol_eval_dir: Path,
    preds: Mapping[str, Mapping],
    encoder_space: str,
    training_num: str,
    checkpoint_num: str,
    logger,
) -> tuple[Dict[str, tuple], Dict[str, np.ndarray]]:
    """
    Per-slide inputs of the two manifold metrics: ceiling_halves[slide] = (t1, t2), the frozen count-thinning
    halves of Y_true (Metric 1 ceiling), and embeddings[slide], the encoder embedding or the coords (Metric 2).
    A missing halves file or row raises; other per-slide errors are logged and the slide is skipped.
    """
    ceiling_halves: Dict[str, tuple] = {}
    embeddings: Dict[str, np.ndarray] = {}
    sub_cfg = str(cfg["protocol"].get("sub_eval_configuration", ""))
    level = str(cfg["protocol"].get("level", ""))
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    use_coords_only = sub_cfg == "STHELAR_TRAIN_COORDS"
    is_patch = level.startswith("patch")
    noise_ceiling_dir = protocol_eval_dir / "settings" / "noise_ceiling"
    emb_dir = (
        resolve_he_embeddings_dir(training_dataset_id, training_num, checkpoint_num)
        if (is_patch and not use_coords_only)
        else None
    )
    for slide_id, bundle in preds.items():
        gene_names = [str(g) for g in np.asarray(bundle["gene_names"], dtype=object).tolist()]
        row_keys = [str(r) for r in np.asarray(bundle["row_keys"], dtype=object).tolist()]
        # Metric 1 ceiling: the frozen count-thinning halves, in y_true's metric space.
        try:
            ceiling_halves[str(slide_id)] = load_thinned_halves(
                noise_ceiling_dir, str(slide_id), row_keys, gene_names
            )
        except (FileNotFoundError, KeyError):
            # Missing halves file or row: stale settings, re-run _2_fill_eval_settings.py
            raise
        except Exception as e:  # noqa: BLE001 -- per-slide errors are logged
            logger.warning(
                "Neighborhood-preservation ceiling unavailable for slide '%s': %s", slide_id, e
            )
        # Metric 2 embedding: raw per-space encoder output (or coords for the baseline).
        try:
            if use_coords_only:
                # Min-max normalized coords (same features as the COORDS probe)
                xy = np.asarray(bundle["coords_xy"], dtype=np.float32)
                coords_df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})
                embeddings[str(slide_id)] = _normalized_coords_features(
                    coords_df, coords_df
                )
            elif is_patch and emb_dir is not None:
                obs = bundle.get("obs_df", None)
                if obs is not None and "patch_id" in obs.columns:
                    patch_ids = obs["patch_id"].astype(str).to_numpy(dtype=object)
                    embeddings[str(slide_id)] = load_patch_embeddings(
                        emb_dir / f"{slide_id}.h5", patch_ids, encoder_space
                    )
        except Exception as e:  # noqa: BLE001 -- best-effort embedding
            logger.warning("Teacher-niche embedding unavailable for slide '%s': %s", slide_id, e)
    return ceiling_halves, embeddings


def _compute_and_save_outputs(
    *,
    preds: Mapping[str, Mapping],
    cfg: Mapping,
    out_dir: Path,
    canonical_reports: bool = False,
    reliability_df: Optional[pd.DataFrame] = None,
    reliability_profile_df: Optional[pd.DataFrame] = None,
    gene_strata: Optional[pd.DataFrame] = None,
    signature_strata: Optional[pd.DataFrame] = None,
    ceiling_halves: Optional[Mapping[str, tuple]] = None,
    embeddings: Optional[Mapping[str, np.ndarray]] = None,
    embedding_space: str = "",
    held_out: bool = False,
) -> tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    benchmark = compute_benchmark_metrics(
        preds,
        cfg,
        reliability_df=reliability_df,
        reliability_profile_df=reliability_profile_df,
    )
    diagnostics = compute_diagnostics(
        preds, benchmark, cfg,
        gene_strata=gene_strata,
        signature_strata=signature_strata,
    )
    structure = compute_structure_analyses(
        preds,
        cfg,
        ceiling_halves=ceiling_halves,
        embeddings=embeddings,
        embedding_space=embedding_space,
        held_out=held_out,
    )
    save_benchmark_outputs(benchmark, out_dir / "benchmark")
    save_diagnostics_outputs(diagnostics, out_dir / "diagnostics")
    save_structure_outputs(structure, out_dir / "structure")
    if canonical_reports:
        save_canonical_reports(
            preds,
            benchmark,
            diagnostics,
            out_dir / "reports",
            cfg=cfg,
            n_slides=int(cfg["protocol"].get("reports", {}).get("n_slides", 3)),
            n_genes=int(cfg["protocol"].get("reports", {}).get("n_genes", 4)),
        )
    return benchmark, diagnostics, structure



def _assert_all_fold_embeddings_finite(folds) -> None:
    for i, fold in enumerate(folds):
        for split_name, slides in [
            ("train", fold.train_slides),
            ("val", fold.val_slides),
            ("test", fold.test_slides),
        ]:
            for s in slides:
                x = np.asarray(s.X, dtype=np.float32)
                if not np.isfinite(x).all():
                    raise RuntimeError(
                        f"Non-finite values found in {split_name} embeddings for "
                        f"fold_index={i}, fold_name='{fold.fold_name}', "
                        f"slide='{s.slide_id}', shape={x.shape}, "
                        f"n_nan={int(np.isnan(x).sum())}, "
                        f"n_inf={int(np.isinf(x).sum())}"
                    )



def _safe_dir_component(x: object) -> str:
    s = str(x).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:120] if len(s) > 120 else s



def _run_probe(
    *,
    cfg: Mapping,
    run_eval_dir: Path,
    protocol_eval_dir: Path,
    run_dir: Path,
    logger,
    wrun: WandbRunHandle,
    rows_df: pd.DataFrame,
    training_num: str,
    checkpoint_num: str,
    encoder_space: str,
    skip_reports: bool = False,
    skip_predictions: bool = False,
    tensor_cache: Optional[SlideTensorCache] = None,
) -> None:
    sub_cfg = str(cfg["protocol"]["sub_eval_configuration"])
    target_basis = str(cfg["protocol"].get("target_basis", "genes"))
    zscore_stats = pd.read_parquet(
        protocol_eval_dir / "settings" / "target_norm" / "zscore_stats.parquet"
    )
    noise_ceiling_dir = protocol_eval_dir / "settings" / "noise_ceiling"
    signature_targets_df: Optional[pd.DataFrame] = None
    signature_slide_eligibility: Optional[pd.DataFrame] = None
    if target_basis == "signatures":
        # Signature-axis ceilings (localization table with the signature as 'gene'); no gene strata
        reliability_df = _signature_reliability_gene_axis(
            pd.read_parquet(noise_ceiling_dir / "reliability_signature.parquet")
        )
        # Profile ceiling per (slide, location)
        reliability_profile_df = pd.read_parquet(
            noise_ceiling_dir / "reliability_profile_signature.parquet"
        )
        gene_strata = None
        signature_strata = None
        signature_targets_df = pd.read_parquet(
            protocol_eval_dir / "settings" / "genes" / "signature_targets.parquet"
        )
        # Per-(slide, signature) run-time mask (see _signaturize_fold_data).
        elig_path = (
            protocol_eval_dir / "settings" / "genes" / "signature_slide_eligibility.parquet"
        )
        if elig_path.exists():
            signature_slide_eligibility = pd.read_parquet(elig_path)
        else:
            logger.warning(
                "No signature_slide_eligibility.parquet under '%s': every signature is scored "
                "on every slide with at least one member. Re-run _2_fill_eval_settings.py to "
                "create the mask.",
                elig_path.parent,
            )
    else:
        reliability_df = pd.read_parquet(noise_ceiling_dir / "reliability.parquet")
        reliability_profile_df = pd.read_parquet(noise_ceiling_dir / "reliability_profile.parquet")
        gene_strata = pd.read_parquet(protocol_eval_dir / "settings" / "genes" / "gene_strata.parquet")
        signature_strata_path = protocol_eval_dir / "settings" / "genes" / "signature_strata.parquet"
        signature_strata = (
            pd.read_parquet(signature_strata_path)
            if signature_strata_path.exists()
            else None
        )
    if sub_cfg == "STHELAR_TEST_ADAPTED":
        if target_basis == "signatures":
            adapted_genes_manifest = load_json(
                protocol_eval_dir / "settings" / "genes" / "genes_manifest.json"
            )
            folds = _signature_folds_from_within_slide_protocol(
                cfg=cfg,
                eval_dir=protocol_eval_dir,
                rows_df=rows_df,
                member_gene_names=adapted_genes_manifest.get(
                    "train_gene_names", adapted_genes_manifest.get("trainable_genes", [])
                ),
                targets_df=signature_targets_df,
                training_num=training_num,
                checkpoint_num=checkpoint_num,
                encoder_space=encoder_space,
                zscore_stats=zscore_stats,
                tensor_cache=tensor_cache,
                slide_eligibility=signature_slide_eligibility,
            )
        else:
            folds = _folds_from_within_slide_protocol(
                cfg=cfg,
                eval_dir=protocol_eval_dir,
                rows_df=rows_df,
                training_num=training_num,
                checkpoint_num=checkpoint_num,
                encoder_space=encoder_space,
                zscore_stats=zscore_stats,
                tensor_cache=tensor_cache,
            )
        fold_specs = None
    else:
        splits_manifest = load_json(protocol_eval_dir / "settings" / "splits" / "splits_manifest.json")
        genes_manifest = load_json(protocol_eval_dir / "settings" / "genes" / "genes_manifest.json")
        if sub_cfg in HEST_CV_SUB_EVAL_CONFIGS and splits_manifest.get("type") == "slide_cv":
            if len(splits_manifest.get("folds", [])) == 0:
                raise RuntimeError(
                    f"{sub_cfg} has zero runnable HEST CV folds. "
                    "No supported tissues passed the current slide/group/gene thresholds. "
                    f"Inspect: {protocol_eval_dir / 'settings' / 'splits' / 'hest_cv_support_manifest.json'}"
                )
        folds = None
        fold_specs = []
        if splits_manifest["type"] == "slide_cv":
            if target_basis == "signatures":
                # Same signature manifest for every CV fold
                for fold in splits_manifest["folds"]:
                    fold_specs.append(
                        {"split_payload": fold, "gene_payload": genes_manifest}
                    )
            else:
                gene_by_fold = {int(f["fold_id"]): f for f in genes_manifest["folds"]}
                for fold in splits_manifest["folds"]:
                    if int(fold["fold_id"]) not in gene_by_fold:
                        raise RuntimeError(
                            f"Missing gene manifest for fold_id={fold['fold_id']} in genes_manifest.json."
                        )
                    fold_specs.append(
                        {
                            "split_payload": fold,
                            "gene_payload": gene_by_fold[int(fold["fold_id"])],
                        }
                    )
        else:
            fold_specs.append(
                {
                    "split_payload": splits_manifest,
                    "gene_payload": genes_manifest,
                }
            )

    merged_test_preds: Dict[str, Mapping] = {}
    fold_records = []
    if fold_specs is None:
        _assert_all_fold_embeddings_finite(folds)
        fold_iter = list(enumerate(folds))
    else:
        fold_iter = list(enumerate(fold_specs))

    for i, fold_or_spec in fold_iter:
        if fold_specs is None:
            fold = fold_or_spec
        else:
            t0 = time.perf_counter()
            split_payload = fold_or_spec["split_payload"]
            gene_payload = fold_or_spec["gene_payload"]
            logger.info("Materializing fold fold_%s", split_payload.get("fold_id", "oneshot"))
            if target_basis == "signatures":
                fold = _build_signature_fold_data(
                    cfg=cfg,
                    eval_dir=protocol_eval_dir,
                    rows_df=rows_df,
                    split_payload=split_payload,
                    member_gene_names=gene_payload.get(
                        "train_gene_names", gene_payload.get("trainable_genes", [])
                    ),
                    targets_df=signature_targets_df,
                    training_num=training_num,
                    checkpoint_num=checkpoint_num,
                    encoder_space=encoder_space,
                    zscore_stats=zscore_stats,
                    tensor_cache=tensor_cache,
                    slide_eligibility=signature_slide_eligibility,
                )
            else:
                fold = _build_fold_data_slide_protocol(
                    cfg=cfg,
                    eval_dir=protocol_eval_dir,
                    rows_df=rows_df,
                    split_payload=split_payload,
                    gene_payload=gene_payload,
                    training_num=training_num,
                    checkpoint_num=checkpoint_num,
                    encoder_space=encoder_space,
                    zscore_stats=zscore_stats,
                    tensor_cache=tensor_cache,
                )
            logger.info(
                "Materialized fold %s in %.1fs | tensor_cache hits=%d misses=%d",
                fold.fold_name,
                time.perf_counter() - t0,
                tensor_cache.hits if tensor_cache is not None else 0,
                tensor_cache.misses if tensor_cache is not None else 0,
            )

        if sub_cfg in HEST_CV_SUB_EVAL_CONFIGS:
            fold_dir = mkdir(run_dir / "folds" / f"{i:02d}_{_safe_dir_component(fold.fold_name)}")
        else:
            fold_dir = mkdir(run_dir / "folds" / f"fold_{i:02d}")

        _assert_all_fold_embeddings_finite([fold])

        logger.info("Fitting ridge probe on fold %s", fold.fold_name)
        artifacts, val_preds, test_preds = fit_predict_one_fold(
            cfg=cfg,
            fold_data=fold,
            space=encoder_space,
            fold_index=i,
            output_dir=fold_dir,
        )
        # Disk copy only (test_preds stays in memory)
        if not skip_predictions:
            save_prediction_bundle(test_preds, fold_dir / "test_predictions")

        bench_fold = compute_benchmark_metrics(
            test_preds,
            cfg,
            reliability_df=reliability_df,
            reliability_profile_df=reliability_profile_df,
        )
        save_benchmark_outputs(bench_fold, fold_dir / "fold_benchmark")
        fold_summary = {
            "fold_name": fold.fold_name,
            "head_type": head_type_from_config(cfg),
            "feature_geometry": artifacts["feature_geometry"],
            "input_dim": int(artifacts["input_dim"]),
            # Probe capacity of this fold
            "selected_lambda": float(artifacts["selected_lambda"]),
            "selected_alpha": float(artifacts["selected_alpha"]),
            # "interior", "ols" or "zero_capacity" (see modeling.boundary_saturation)
            "lambda_boundary_kind": str(artifacts["lambda_boundary_kind"]),
            "effective_dof": float(artifacts["effective_dof"]),
            "numerical_rank": int(artifacts["numerical_rank"]),
            "effective_dof_pattern_min": float(artifacts["effective_dof_pattern_min"]),
            "effective_dof_pattern_median": float(artifacts["effective_dof_pattern_median"]),
            "effective_dof_pattern_max": float(artifacts["effective_dof_pattern_max"]),
            "n_patterns": int(artifacts["n_patterns"]),
            "n_train_slides": int(artifacts["n_train_slides"]),
            "n_train_rows": int(artifacts["n_train_rows"]),
            "alpha_group": str(artifacts["alpha_group"]),
            "loso_best_score": float(artifacts["loso_best_score"]),
            # Held-out score at the selected lambda
            "loso_score_at_selected": float(artifacts["loso_score_at_selected"]),
            "fit_seconds": float(artifacts["fit_seconds"]),
            **bench_fold["benchmark_summary"],
        }
        fold_records.append(fold_summary)

        fold_test_prefix = f"fold/{fold.fold_name}/test"
        wrun.log(
            {
                f"{fold_test_prefix}/{k}": v
                for k, v in _numeric_only_dict(bench_fold["benchmark_summary"]).items()
            }
        )
        wrun.log(
            {
                f"fold/{fold.fold_name}/probe/selected_lambda": float(artifacts["selected_lambda"]),
                f"fold/{fold.fold_name}/probe/effective_dof": float(artifacts["effective_dof"]),
            }
        )

        _merge_prediction_dicts(merged_test_preds, test_preds)

    pd.DataFrame(fold_records).to_csv(run_dir / "fold_metrics.csv", index=False)
    # Optional disk copy of merged_test_preds
    if not skip_predictions:
        save_prediction_bundle(merged_test_preds, run_dir / "run_predictions")
    # Per-slide inputs of the manifold metrics (Metric 1 ceiling, Metric 2 embeddings)
    if target_basis == "signatures":
        # Not applicable on the signature axis
        logger.info(
            "target_basis=signatures: skipping the structure aux inputs (ceiling halves, "
            "teacher-niche embeddings); knn_overlap_ceiling_norm and teacher_niche_recovery_* "
            "are not computed."
        )
        ceiling_halves, structure_embeddings = None, None
    else:
        ceiling_halves, structure_embeddings = _build_structure_aux_inputs(
            cfg=cfg,
            protocol_eval_dir=protocol_eval_dir,
            preds=merged_test_preds,
            encoder_space=encoder_space,
            training_num=training_num,
            checkpoint_num=checkpoint_num,
            logger=logger,
        )
    benchmark, diagnostics, structure = _compute_and_save_outputs(
        preds=merged_test_preds,
        cfg=cfg,
        out_dir=run_dir,
        canonical_reports=not bool(skip_reports),
        reliability_df=reliability_df,
        reliability_profile_df=reliability_profile_df,
        gene_strata=gene_strata,
        signature_strata=signature_strata,
        ceiling_halves=ceiling_halves,
        embeddings=structure_embeddings,
        embedding_space=encoder_space,
        held_out=_is_held_out_sub_eval(sub_cfg),
    )
    save_run_summary(run_dir, benchmark, diagnostics, structure)
    wrun.summary_update(
        {f"run_benchmark/{k}": v for k, v in _numeric_only_dict(benchmark["benchmark_summary"]).items()}
    )



def run_probe(
    *,
    cfg: Mapping,
    run_eval_dir: Path,
    protocol_eval_dir: Path,
    encoder_dir: Path,
    training_num: str,
    checkpoint_num: str,
    encoder_space: str,
    rows_df: pd.DataFrame,
    skip_reports: bool = False,
    skip_predictions: bool = False,
    tensor_cache: Optional[SlideTensorCache] = None,
) -> None:
    run_dir = encoder_dir / "run"
    if run_dir.exists():
        rmtree_appledouble_safe(run_dir)
    run_dir = mkdir(run_dir)
    logger = build_logger(run_dir / "log" / "logfile.log", name="probe")
    wrun = _make_wandb_run(cfg, run_dir, encoder_dir)
    logger.info("Running eval_id=%s encoder=%s", cfg["identity"]["eval_id"], encoder_dir.name)
    try:
        _run_probe(
            cfg=cfg,
            run_eval_dir=run_eval_dir,
            protocol_eval_dir=protocol_eval_dir,
            run_dir=run_dir,
            logger=logger,
            wrun=wrun,
            rows_df=rows_df,
            training_num=training_num,
            checkpoint_num=checkpoint_num,
            encoder_space=encoder_space,
            skip_reports=skip_reports,
            skip_predictions=skip_predictions,
            tensor_cache=tensor_cache,
        )
    finally:
        wrun.finish()




def main(args: argparse.Namespace) -> None:

    print(f"Starting evaluation with args: {args}")
    print("\n")

    run_eval_dir = resolve_eval_task_root() / args.eval_id
    cfg = load_yaml(run_eval_dir / "settings" / "eval_config.yaml")
    protocol_manifest = load_json(run_eval_dir / "settings" / "protocol_manifest.json")
    assert_eval_config_matches_protocol_manifest(cfg, protocol_manifest)
    validate_protocol_identity(cfg)

    protocol_eval_dir = protocol_eval_dir_from_manifest(run_eval_dir, protocol_manifest)
    if protocol_eval_dir.resolve() != run_eval_dir.resolve():
        assert_reference_protocol_hashes_unchanged(protocol_manifest)

    rows_df = _load_rows_all(protocol_eval_dir)
    sub_cfg = str(cfg["protocol"]["sub_eval_configuration"])
    is_coords_baseline = sub_cfg == "STHELAR_TRAIN_COORDS"

    if is_coords_baseline:
        training_num = str(args.training_num) if args.training_num is not None else "NA"
        checkpoint_num = str(args.checkpoint_num) if args.checkpoint_num is not None else "NA"
        encoder_space = "coords"
    else:
        if args.training_num is None or args.checkpoint_num is None or args.space is None:
            raise ValueError("--training_num, --checkpoint_num and --space are required for non-coordinate baselines.")
        training_num = str(args.training_num)
        checkpoint_num = str(args.checkpoint_num)
        encoder_space = str(args.space)
        if encoder_space not in {"he", "he_l2", "clip"}:
            raise ValueError("space must be one of {'he','he_l2','clip'}")
    # Fail early on an unsupported space
    resolve_feature_geometry(cfg, encoder_space)

    encoder_path: Optional[Path] = None
    embeddings_dir: Optional[Path] = None
    if not is_coords_baseline:
        encoder_path = resolve_encoder_path(training_num, checkpoint_num)
        embeddings_dir = resolve_he_embeddings_dir(str(cfg["identity"]["training_dataset_id"]), training_num, checkpoint_num)
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder file not found: {encoder_path}")
        if not embeddings_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
        _check_encoder_modelgroup(encoder_path, str(cfg["protocol"]["modelgroup"]))
        required_slide_ids = _required_slide_ids_from_persisted_protocol(
            protocol_eval_dir=protocol_eval_dir,
            cfg=cfg,
            rows_df=rows_df,
        )
        _check_requested_space_available(
            embeddings_dir,
            level=str(cfg["protocol"]["level"]),
            space=encoder_space,
            sub_eval_configuration=sub_cfg,
            required_slide_ids=required_slide_ids,
        )

    encoder_dir_name = "COORDS_BASELINE" if is_coords_baseline else f"HEencoder_t{training_num}e{checkpoint_num}_{encoder_space}"
    encoder_dir = mkdir(run_eval_dir / encoder_dir_name)

    tensor_cache = SlideTensorCache()

    run_probe(
        cfg=cfg,
        run_eval_dir=run_eval_dir,
        protocol_eval_dir=protocol_eval_dir,
        encoder_dir=encoder_dir,
        training_num=training_num,
        checkpoint_num=checkpoint_num,
        encoder_space=encoder_space,
        rows_df=rows_df,
        skip_reports=bool(args.skip_reports),
        skip_predictions=bool(args.skip_predictions),
        tensor_cache=tensor_cache,
    )
    _aggregate_fold_outputs(encoder_dir)
    if not (encoder_dir / "run" / "benchmark" / "benchmark_summary.json").exists():
        raise RuntimeError(
            f"The probe run under {encoder_dir / 'run'} produced no benchmark summary."
        )
    _save_encoder_manifest(
        encoder_dir,
        encoder_path,
        embeddings_dir,
        cfg,
        protocol_eval_dir,
        training_num=training_num,
        checkpoint_num=checkpoint_num,
        encoder_space=encoder_space,
        skip_reports=bool(args.skip_reports),
        skip_predictions=bool(args.skip_predictions),
    )
    print(f"Completed encoder evaluation in: {encoder_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the Task A.1 evaluation for one H&E encoder and space."
        )
    )

    parser.add_argument("--eval_id", type=str,
                        required=True,
                        help="Evaluation ID, e.g. 'eval_1'")

    parser.add_argument("--training_num", type=str,
                        default=None,
                        help="Num from HistoST training_id (e.g. '46' for training_id=='training_46')")

    parser.add_argument("--checkpoint_num", type=str,
                        default=None,
                        help="Checkpoint number to load (e.g. '13' for 'checkpoint_epoch_13.pth')")

    parser.add_argument("--space", type=str,
                        default=None,
                        choices=["he", "he_l2", "clip"],
                        help=("Embedding space: 'he' (backbone), 'he_l2' (backbone, row-L2 normalized) "
                              "or 'clip' (CLIP projection, row-L2 normalized). Not needed for STHELAR_TRAIN_COORDS."))

    parser.add_argument("--skip_reports", action="store_true",
                        help=("Skip the qualitative reports (not part of the protocol)."))

    parser.add_argument("--skip_predictions", action="store_true",
                        help=("Do not write the HDF5 prediction files (run/folds/*/test_predictions/ and "
                              "run/run_predictions/); metrics are unchanged."))

    args = parser.parse_args()
    main(args)
