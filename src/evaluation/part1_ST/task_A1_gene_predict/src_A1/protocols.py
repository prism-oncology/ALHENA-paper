from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import anndata as ad
import numpy as np
import pandas as pd
import copy
import h5py
import json
import logging
from scipy import sparse

from .common import (
    AUTO_GENE_PANEL_SELECTION_SCOPE,
    DIRECT_HEAD_TYPES,
    CONTROL_KIND_BY_SUB_EVAL_CONFIG,
    CONTROL_SUB_EVAL_CONFIGS,
    HEST_CV_SUB_EVAL_CONFIGS,
    REFERENCE_CONTROL_EVAL_CONFIGURATION,
    REFERENCE_CONTROL_SUB_EVAL_CONFIGURATION,
    _ADAPTED_STRIPE_ROLES,
    assert_eval_config_matches_protocol_manifest,
    control_kind_for_sub_eval_configuration,
    stripes_for_role,
    protocol_identity_sha256,
    default_eval_config,
    dump_environment_report,
    load_json,
    load_yaml,
    mkdir,
    reference_protocol_file_manifests,
    reference_protocol_hash_aliases,
    read_metadata,
    resolve_G_base_dir,
    resolve_gene_property_dir,
    resolve_eval_signatures_dir,
    resolve_eval_signatures_INT_dir,
    load_signature_sets,
    SIGNATURE_AXIS_SEP,
    resolve_eval_dataset_root,
    resolve_metadata_csv,
    save_json,
    save_json_atomic,
    save_eval_config_with_comments,
    save_text,
    sha256_file,
    validate_protocol_identity,
    gene_panel_mode_from_G_base_id,
    head_type_from_config,
    prediction_head_manifest_payload,
    resolve_eval_task_root,
    utc_now,
)
from .data_loading import (
    _obs_bool_array,
    aggregate_signature_targets,
    build_row_manifest_for_slide,
    build_slide_gene_panel_manifest,
    canonicalize_gene_table,
    extract_raw_counts_for_rows,
    extract_targets_for_rows,
    load_membership_table,
    load_patch_geometry,
    load_slide_records,
    winsorize_zscored,
)
from .gene_embeddings import load_scgpt_vocab, scgpt_vocab_manifest
from .metrics import _safe_pearson, _safe_spearman, _VAR_EPS


logger = logging.getLogger(__name__)


# Minimum number of genes of a stratum (persist_gene_strata, persist_signature_strata);
# equal to paired_stats.MIN_N_FOR_WILCOXON
MIN_GENES_FOR_STRATA = 6


@dataclass
class ProtocolPaths:
    eval_dir: Path
    settings_dir: Path
    info_dir: Path
    splits_dir: Path
    rows_dir: Path
    genes_dir: Path
    area_dir: Path
    target_norm_dir: Path
    noise_ceiling_dir: Path


# -----------------------------------------------------------------------------
# Config bootstrap
# -----------------------------------------------------------------------------


def make_protocol_paths(eval_dir: Path) -> ProtocolPaths:
    return ProtocolPaths(
        eval_dir=eval_dir,
        settings_dir=mkdir(eval_dir / "settings"),
        info_dir=mkdir(eval_dir / "settings" / "informations"),
        splits_dir=mkdir(eval_dir / "settings" / "splits"),
        rows_dir=mkdir(eval_dir / "settings" / "rows"),
        genes_dir=mkdir(eval_dir / "settings" / "genes"),
        area_dir=mkdir(eval_dir / "settings" / "area_splits"),
        target_norm_dir=mkdir(eval_dir / "settings" / "target_norm"),
        noise_ceiling_dir=mkdir(eval_dir / "settings" / "noise_ceiling"),
    )


_CONTROL_INHERITED_IDENTITY_KEYS = [
    "dataset_id",
    "training_dataset_id",
]


_CONTROL_INHERITED_PROTOCOL_KEYS = [
    "level",
    "modelgroup",
    "G_base_id",
    "gene_property_id",
    "eval_signatures_id",
    "scgpt_model_name",
    "protocol_seed",
    "target_transform",
    "head",
    "ridge",
    "input_preprocessing",
    "metrics",
    "gene_selection",
    "cv",
]


def _stable_json(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), default=str)


def _load_reference_eval(reference_eval_id: str) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    if reference_eval_id is None or str(reference_eval_id).strip() == "":
        raise ValueError(
            "Control protocols require reference_eval_id pointing to an existing "
            "STHELAR_TRAIN/STHELAR_TRAIN_CV eval_id."
        )

    ref_dir = resolve_eval_task_root() / str(reference_eval_id)
    ref_cfg_path = ref_dir / "settings" / "eval_config.yaml"
    ref_manifest_path = ref_dir / "settings" / "protocol_manifest.json"

    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference eval_id does not exist: {reference_eval_id} -> {ref_dir}")
    if not ref_cfg_path.exists():
        raise FileNotFoundError(f"Reference eval_id is missing eval_config.yaml: {ref_cfg_path}")
    if not ref_manifest_path.exists():
        raise FileNotFoundError(
            f"Reference eval_id is not filled yet. Missing protocol_manifest.json: {ref_manifest_path}"
        )

    ref_cfg = load_yaml(ref_cfg_path)
    ref_manifest = load_json(ref_manifest_path)
    assert_eval_config_matches_protocol_manifest(ref_cfg, ref_manifest)

    ref_eval_conf = str(ref_cfg["protocol"]["eval_configuration"])
    ref_sub_conf = str(ref_cfg["protocol"]["sub_eval_configuration"])
    if ref_eval_conf != REFERENCE_CONTROL_EVAL_CONFIGURATION or ref_sub_conf != REFERENCE_CONTROL_SUB_EVAL_CONFIGURATION:
        raise RuntimeError(
            "Invalid reference_eval_id for child control. Expected reference to be "
            f"eval_configuration={REFERENCE_CONTROL_EVAL_CONFIGURATION}, "
            f"sub_eval_configuration={REFERENCE_CONTROL_SUB_EVAL_CONFIGURATION}; "
            f"got eval_configuration={ref_eval_conf}, sub_eval_configuration={ref_sub_conf}."
        )

    return ref_dir, ref_cfg, ref_manifest


def _assert_control_setup_args_match_reference(
    *,
    ref_cfg: Mapping[str, Any],
    dataset_id: str,
    training_dataset_id: str,
    level: str,
    G_base_id: str,
    gene_property_id: str,
    eval_signatures_id: Optional[str],
    modelgroup: str,
    scgpt_model_name: Optional[str],
    protocol_seed: Optional[int],
) -> None:
    checks = [
        ("identity.dataset_id", dataset_id, ref_cfg["identity"]["dataset_id"]),
        ("identity.training_dataset_id", training_dataset_id, ref_cfg["identity"]["training_dataset_id"]),
        ("protocol.level", level, ref_cfg["protocol"]["level"]),
        ("protocol.G_base_id", G_base_id, ref_cfg["protocol"]["G_base_id"]),
        ("protocol.gene_property_id", gene_property_id, ref_cfg["protocol"].get("gene_property_id", "")),
        ("protocol.eval_signatures_id", eval_signatures_id, ref_cfg["protocol"].get("eval_signatures_id", None)),
        ("protocol.modelgroup", modelgroup, ref_cfg["protocol"]["modelgroup"]),
    ]
    if scgpt_model_name is not None:
        checks.append(("protocol.scgpt_model_name", scgpt_model_name, ref_cfg["protocol"]["scgpt_model_name"]))
    if protocol_seed is not None:
        checks.append(("protocol.protocol_seed", int(protocol_seed), int(ref_cfg["protocol"]["protocol_seed"])))

    mismatches = []
    for key, child_value, ref_value in checks:
        if _stable_json(child_value) != _stable_json(ref_value):
            mismatches.append(f"{key}: child={child_value!r} reference={ref_value!r}")

    if mismatches:
        raise RuntimeError(
            "Control child setup arguments do not match the reference STHELAR_TRAIN_CV protocol. "
            f"Mismatches: {mismatches}"
        )


def _derive_control_cfg_from_reference(
    *,
    ref_cfg: Mapping[str, Any],
    eval_id: str,
    child_sub_eval_configuration: str,
    reference_eval_id: str,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(dict(ref_cfg))
    cfg["identity"]["eval_id"] = str(eval_id)
    cfg["identity"]["created_utc"] = utc_now()

    cfg["protocol"]["eval_configuration"] = REFERENCE_CONTROL_EVAL_CONFIGURATION
    cfg["protocol"]["sub_eval_configuration"] = str(child_sub_eval_configuration)
    cfg["protocol"]["reference_eval_id"] = str(reference_eval_id)
    cfg["protocol"]["control_kind"] = control_kind_for_sub_eval_configuration(child_sub_eval_configuration)

    validate_protocol_identity(cfg)
    return cfg


def _assert_child_config_compatible_with_reference(
    *,
    child_cfg: Mapping[str, Any],
    ref_cfg: Mapping[str, Any],
) -> None:
    child_sub = str(child_cfg["protocol"]["sub_eval_configuration"])
    if child_sub not in CONTROL_SUB_EVAL_CONFIGS:
        raise ValueError(f"Not a child control sub_eval_configuration: {child_sub}")

    mismatches: List[str] = []

    for key in _CONTROL_INHERITED_IDENTITY_KEYS:
        child_value = child_cfg["identity"].get(key)
        ref_value = ref_cfg["identity"].get(key)
        if _stable_json(child_value) != _stable_json(ref_value):
            mismatches.append(f"identity.{key}: child={child_value!r} reference={ref_value!r}")

    for key in _CONTROL_INHERITED_PROTOCOL_KEYS:
        child_value = child_cfg["protocol"].get(key)
        ref_value = ref_cfg["protocol"].get(key)
        if _stable_json(child_value) != _stable_json(ref_value):
            mismatches.append(f"protocol.{key}: child={child_value!r} reference={ref_value!r}")

    expected_ref_id = str(child_cfg["protocol"].get("reference_eval_id", ""))
    if not expected_ref_id:
        mismatches.append("protocol.reference_eval_id is missing")

    expected_kind = CONTROL_KIND_BY_SUB_EVAL_CONFIG[child_sub]
    actual_kind = str(child_cfg["protocol"].get("control_kind", ""))
    if actual_kind != expected_kind:
        mismatches.append(f"protocol.control_kind: child={actual_kind!r} expected={expected_kind!r}")

    if mismatches:
        raise RuntimeError(
            "Child control config is incompatible with its reference STHELAR_TRAIN_CV protocol. "
            f"Mismatches: {mismatches}"
        )


def _load_validated_reference_for_control(
    *,
    child_cfg: Mapping[str, Any],
    child_eval_dir: Path,
) -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    reference_eval_id = str(child_cfg["protocol"].get("reference_eval_id", ""))
    ref_dir, ref_cfg, ref_manifest = _load_reference_eval(reference_eval_id)

    if ref_dir.resolve() == Path(child_eval_dir).resolve():
        raise RuntimeError("A child control eval_id cannot reference itself.")

    _assert_child_config_compatible_with_reference(child_cfg=child_cfg, ref_cfg=ref_cfg)
    return ref_dir, ref_cfg, ref_manifest


def _build_reference_protocol_payload(
    *,
    child_cfg: Mapping[str, Any],
    reference_eval_id: str,
    reference_eval_dir: Path,
    reference_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    file_manifests = reference_protocol_file_manifests(reference_eval_dir)
    hash_aliases = reference_protocol_hash_aliases(file_manifests)
    child_sub = str(child_cfg["protocol"]["sub_eval_configuration"])

    payload: Dict[str, Any] = {
        "derived_from_reference": True,
        "reference_eval_id": str(reference_eval_id),
        "reference_eval_dir": str(reference_eval_dir),
        "reference_settings_dir": str(reference_eval_dir / "settings"),
        "reference_eval_configuration": str(reference_cfg["protocol"]["eval_configuration"]),
        "reference_sub_eval_configuration": str(reference_cfg["protocol"]["sub_eval_configuration"]),
        "child_sub_eval_configuration": child_sub,
        "control_kind": control_kind_for_sub_eval_configuration(child_sub),
        **hash_aliases,
        "reference_protocol_files": file_manifests,
    }
    return payload


def create_initial_eval_folder(
    *,
    eval_dir: Path,
    eval_id: str,
    dataset_id: str,
    training_dataset_id: str,
    eval_configuration: str,
    sub_eval_configuration: str,
    level: str,
    G_base_id: str,
    gene_property_id: str,
    eval_signatures_id: Optional[str],
    modelgroup: str,
    scgpt_model_name: Optional[str] = None,
    novae_training_id: Optional[str] = None,
    novae_domain_n: Optional[int] = None,
    protocol_seed: Optional[int] = None,
    reference_eval_id: Optional[str] = None,
    eval_signatures_INT_id: Optional[str] = None,
    target_basis: str = "genes",
    signature_sets: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if sub_eval_configuration in CONTROL_SUB_EVAL_CONFIGS:
        ref_dir, ref_cfg, _ = _load_reference_eval(str(reference_eval_id))
        # Control children cannot inherit target_basis="signatures" from their reference
        if str(ref_cfg["protocol"].get("target_basis", "genes")) == "signatures":
            raise NotImplementedError(
                f"Control protocol '{sub_eval_configuration}' does not support "
                f"target_basis='signatures' (inherited from reference "
                f"'{reference_eval_id}'). Signature-target mode is wired only for the "
                "primary sub-eval configurations, not the coordinate-only control."
            )
        _assert_control_setup_args_match_reference(
            ref_cfg=ref_cfg,
            dataset_id=dataset_id,
            training_dataset_id=training_dataset_id,
            level=level,
            G_base_id=G_base_id,
            gene_property_id=gene_property_id,
            eval_signatures_id=eval_signatures_id,
            modelgroup=modelgroup,
            scgpt_model_name=scgpt_model_name,
            protocol_seed=protocol_seed,
        )
        cfg = _derive_control_cfg_from_reference(
            ref_cfg=ref_cfg,
            eval_id=eval_id,
            child_sub_eval_configuration=sub_eval_configuration,
            reference_eval_id=str(reference_eval_id),
        )
    else:
        cfg = default_eval_config(
            eval_id=eval_id,
            dataset_id=dataset_id,
            training_dataset_id=training_dataset_id,
            eval_configuration=eval_configuration,
            sub_eval_configuration=sub_eval_configuration,
            level=level,
            G_base_id=G_base_id,
            gene_property_id=gene_property_id,
            eval_signatures_id=eval_signatures_id,
            eval_signatures_INT_id=eval_signatures_INT_id,
            target_basis=target_basis,
            signature_sets=signature_sets,
            modelgroup=modelgroup,
            scgpt_model_name=scgpt_model_name,
            novae_training_id=novae_training_id,
            novae_domain_n=novae_domain_n,
            protocol_seed=protocol_seed,
        )

        # Head and ridge settings come from the defaults
        validate_protocol_identity(cfg)

    paths = make_protocol_paths(eval_dir)
    save_eval_config_with_comments(paths.settings_dir / "eval_config.yaml", cfg)
    save_json(paths.settings_dir / "environment.json", dump_environment_report())
    save_text(paths.info_dir / "README.md", _settings_readme())
    build_initial_information_report(cfg=cfg, info_dir=paths.info_dir)
    return cfg


def _settings_readme() -> str:
    return (
        "# HistoST Task A.1 settings folder\n\n"
        "This folder stores the full persisted evaluation protocol for one eval_id.\n\n"
        "Every encoder run under the same eval_id uses the same splits, row masks,\n"
        "trainable gene sets, validity thresholds, head type and ridge settings.\n"
    )


def build_initial_information_report(*, cfg: Mapping[str, Any], info_dir: Path) -> None:
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    dataset_root = resolve_eval_dataset_root(training_dataset_id)
    meta = pd.read_csv(resolve_metadata_csv(training_dataset_id))
    slides = read_metadata(training_dataset_id)
    meta_rows = []
    for rec in slides:
        patch_h5 = dataset_root / "patches" / f"{rec.slide_id}.h5"
        st_h5ad = dataset_root / "st" / f"{rec.slide_id}.h5ad"
        n_patch_all_rows = np.nan
        if st_h5ad.exists():
            a = ad.read_h5ad(st_h5ad, backed="r")
            try:
                idx = a.obs_names.astype(str)
                # HEST spots are `_all`-like and are flagged by an `is_cell` column;
                # STHELAR marks them with an `_all` row suffix.
                if rec.source == "HEST":
                    if "is_cell" in a.obs.columns:
                        n_patch_all_rows = int(np.sum(~_obs_bool_array(a.obs["is_cell"], name="is_cell")))
                    else:
                        n_patch_all_rows = int(a.n_obs)
                else:
                    n_patch_all_rows = int(np.sum(idx.str.endswith("_all")))
            finally:
                if getattr(a, "file", None) is not None:
                    a.file.close()
        meta_rows.append(
            {
                "slide_id": rec.slide_id,
                "source": rec.source,
                "subset": rec.subset,
                "tissue": rec.tissue,
                "species": rec.species,
                "patch_h5_exists": int(patch_h5.exists()),
                "st_h5ad_exists": int(st_h5ad.exists()),
                "n_patch_all_rows": n_patch_all_rows,
            }
        )
    df = pd.DataFrame(meta_rows).sort_values(["source", "subset", "tissue", "slide_id"])
    df.to_csv(info_dir / "dataset_slide_summary.csv", index=False)
    by_subset = (
        df.groupby(["source", "subset", "tissue"], dropna=False)
        .agg(
            n_slides=("slide_id", "nunique"),
            patch_all_rows=("n_patch_all_rows", "sum"),
        )
        .reset_index()
    )
    by_subset.to_csv(info_dir / "dataset_summary_by_subset.csv", index=False)


# -----------------------------------------------------------------------------
# Row manifests
# -----------------------------------------------------------------------------

_ROW_MANIFEST_STRING_COLUMNS = frozenset(
    {
        "slide_id",
        "row_key",
        "source",
        "subset",
        "tissue",
        "species",
        "patient_raw",
        "patient_clean",
        "dataset_title",
        "split_group_id",
        "patch_id",
        "cell_type",
        "novae_domain",
    }
)

_ROW_MANIFEST_INT_COLUMNS = frozenset(
    {
        "cell_id_int",
    }
)

_ROW_MANIFEST_FLOAT_COLUMNS = frozenset(
    {
        "x",
        "y",
        "novae_domain_frac",
    }
)

_ROW_MANIFEST_BOOL_COLUMNS = frozenset(
    {
        "is_cell",
        "in_tissue",
    }
)

_BOOL_TRUE_STRINGS = frozenset({"true", "t", "1", "yes", "y"})
_BOOL_FALSE_STRINGS = frozenset({"false", "f", "0", "no", "n"})


def _is_missing_scalar(x: Any) -> bool:
    if x is None or x is pd.NA:
        return True
    try:
        is_na = pd.isna(x)
    except (TypeError, ValueError):
        return False
    if isinstance(is_na, (bool, np.bool_)):
        return bool(is_na)
    return False


def _boollike_scalar_or_none(x: Any) -> Any:
    """
    Return True/False/pd.NA for boolean-like scalar values.
    Return None when the value is non-missing but not boolean-like.
    """
    if _is_missing_scalar(x):
        return pd.NA
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)):
        xi = int(x)
        if xi in (0, 1):
            return bool(xi)
        return None
    if isinstance(x, (float, np.floating)):
        xf = float(x)
        if np.isfinite(xf) and xf in (0.0, 1.0):
            return bool(int(xf))
        return None

    s = str(x).strip().casefold()
    if s in _BOOL_TRUE_STRINGS:
        return True
    if s in _BOOL_FALSE_STRINGS:
        return False
    return None


def _series_is_boollike(s: pd.Series) -> bool:
    vals = s.to_numpy(dtype=object, copy=False)
    if len(vals) == 0:
        return True
    return all(_boollike_scalar_or_none(v) is not None for v in vals)


def _coerce_boollike_series(s: pd.Series) -> pd.Series:
    return pd.Series(
        [_boollike_scalar_or_none(v) for v in s.to_numpy(dtype=object, copy=False)],
        index=s.index,
        dtype="boolean",
    )


def _coerce_numeric_like_object_series(s: pd.Series) -> Optional[pd.Series]:
    """Numeric version of an object column if all non-missing values are numeric-like and not booleans, else None."""
    vals = s.to_numpy(dtype=object, copy=False)
    non_missing = [v for v in vals if not _is_missing_scalar(v)]
    if not non_missing:
        return None

    if any(isinstance(v, (bool, np.bool_)) for v in non_missing):
        return None

    numeric = pd.to_numeric(s, errors="coerce")
    valid_mask = pd.Series(
        [not _is_missing_scalar(v) for v in vals],
        index=s.index,
        dtype=bool,
    )
    if bool(numeric.loc[valid_mask].isna().any()):
        return None

    valid = numeric.loc[valid_mask].to_numpy(dtype=np.float64, copy=False)
    if valid.size and np.all(np.isfinite(valid)) and np.allclose(valid, np.round(valid)):
        return numeric.round().astype("Int64")
    return numeric.astype(np.float64)


def _coerce_row_manifest_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Cast the row manifest columns to Parquet-compatible types (one physical type per column)."""
    out = df.copy()

    for col in out.columns:
        s = out[col]

        if col in _ROW_MANIFEST_STRING_COLUMNS:
            out[col] = s.astype("string")
            continue

        if col in _ROW_MANIFEST_INT_COLUMNS:
            out[col] = pd.to_numeric(s, errors="coerce").astype("Int64")
            continue

        if col in _ROW_MANIFEST_FLOAT_COLUMNS:
            out[col] = pd.to_numeric(s, errors="coerce").astype(np.float64)
            continue

        if col in _ROW_MANIFEST_BOOL_COLUMNS or str(col).startswith(("is_", "has_")):
            if _series_is_boollike(s):
                out[col] = _coerce_boollike_series(s)
            else:
                out[col] = s.astype("string")
            continue

        if isinstance(s.dtype, pd.CategoricalDtype):
            out[col] = s.astype("string")
            continue

        if pd.api.types.is_object_dtype(s.dtype):
            numeric = _coerce_numeric_like_object_series(s)
            if numeric is not None:
                out[col] = numeric
            else:
                out[col] = s.astype("string")
            continue

    bad_object_cols = [
        str(c)
        for c in out.columns
        if pd.api.types.is_object_dtype(out[c].dtype)
    ]
    if bad_object_cols:
        raise RuntimeError(
            "Row manifest still contains object dtype columns after Parquet "
            f"schema canonicalization: {bad_object_cols[:20]}"
        )

    return out


def _protocol_slide_records(cfg: Mapping[str, Any]) -> List[Any]:
    """Slide records of the eval protocol."""
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    eval_conf = str(cfg["protocol"]["eval_configuration"])
    sub_conf = str(cfg["protocol"]["sub_eval_configuration"])
    records = read_metadata(training_dataset_id)
    available_hest_subsets = sorted(
        {str(r.subset) for r in records if str(r.source) == "HEST"}
    )

    out: List[Any] = []
    for rec in records:
        keep = False
        if eval_conf == "STHELAR_TRAIN":
            keep = (rec.source == "STHELAR") and (rec.subset == "STHELAR_train")
        elif eval_conf == "STHELAR_TEST":
            if sub_conf == "STHELAR_TEST_ONE_SHOT":
                keep = (rec.source == "STHELAR") and (rec.subset in {"STHELAR_train", "STHELAR_test"})
            elif sub_conf == "STHELAR_TEST_ADAPTED":
                keep = (rec.source == "STHELAR") and (rec.subset == "STHELAR_test")
            else:
                raise ValueError(f"Unsupported STHELAR_TEST sub-protocol: {sub_conf}")
        elif eval_conf == "HEST":
            if sub_conf not in HEST_CV_SUB_EVAL_CONFIGS:
                raise ValueError(f"Unsupported HEST sub-protocol: {sub_conf}")
            hest_cv_cfg = cfg["protocol"].get("hest_cv", {})
            subset_map = hest_cv_cfg.get("candidate_subsets", {})
            candidate_subsets = set(map(str, subset_map.get(sub_conf, [])))
            if not candidate_subsets:
                raise ValueError(f"No HEST candidate subsets configured for sub-protocol '{sub_conf}'.")
            missing_candidate_subsets = sorted(candidate_subsets - set(available_hest_subsets))
            if missing_candidate_subsets:
                raise ValueError(
                    f"HEST sub-protocol '{sub_conf}' requests metadata subset(s) that do not exist: "
                    f"{missing_candidate_subsets}. Available HEST subsets are: {available_hest_subsets}"
                )
            keep = (rec.source == "HEST") and (rec.subset in candidate_subsets)
        else:
            raise ValueError(f"Unsupported eval_configuration: {eval_conf}")

        if keep:
            out.append(rec)

    if not out:
        raise RuntimeError(
            f"No slides were selected for protocol "
            f"{eval_conf}/{sub_conf}."
        )
    return out


def _build_row_manifest_frame(cfg: Mapping[str, Any], rec: Any) -> pd.DataFrame:
    dataset_id = str(cfg["identity"]["dataset_id"])
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    level = str(cfg["protocol"]["level"])
    modelgroup = str(cfg["protocol"]["modelgroup"])
    structure_cfg = cfg["protocol"].get("structure", {})
    novae_training_id = structure_cfg.get("novae_training_id", None)
    novae_domain_n = structure_cfg.get("novae_domain_n", None)

    bundle = build_row_manifest_for_slide(
        dataset_id=dataset_id,
        training_dataset_id=training_dataset_id,
        level=level,
        modelgroup=modelgroup,
        slide_record=rec,
        novae_training_id=str(novae_training_id) if novae_training_id is not None else None,
        novae_domain_n=int(novae_domain_n) if novae_domain_n is not None else None,
    )

    df = bundle.obs_df.copy()
    df.insert(0, "row_key", bundle.row_keys)
    df.insert(0, "slide_id", rec.slide_id)
    df["source"] = rec.source
    df["subset"] = rec.subset
    df["tissue"] = rec.tissue
    df["species"] = rec.species
    df["patient_raw"] = rec.patient_raw
    df["patient_clean"] = rec.patient_clean
    df["dataset_title"] = rec.dataset_title
    df["split_group_id"] = rec.split_group_id
    if bundle.patch_ids is not None:
        df["patch_id"] = np.asarray(bundle.patch_ids, dtype=object)
    else:
        df["patch_id"] = None
    if bundle.cell_id_int is not None:
        df["cell_id_int"] = np.asarray(bundle.cell_id_int, dtype=np.int32)
    else:
        df["cell_id_int"] = np.nan

    out = df.reset_index(drop=True)
    if out.empty:
        raise RuntimeError(
            f"Protocol-relevant slide '{rec.slide_id}' produced zero eligible rows "
            f"for level='{level}'. This usually means the eval dataset or protocol "
            f"settings are inconsistent."
        )

    if bundle.row_filter_report is not None:
        out.attrs["row_filter_report"] = dict(bundle.row_filter_report)
    return out


def persist_row_manifests(cfg: Mapping[str, Any], rows_dir: Path) -> pd.DataFrame:

    slide_records = _protocol_slide_records(cfg)

    all_rows: List[pd.DataFrame] = []
    row_filter_reports: List[Dict[str, Any]] = []

    for rec in slide_records:
        out = _build_row_manifest_frame(cfg, rec)
        report = out.attrs.get("row_filter_report", None)
        if report is not None:
            row_filter_reports.append(dict(report))
        all_rows.append(out)

    all_df = pd.concat(all_rows, axis=0, ignore_index=True)

    # Canonicalize the column types on all slides at once (same types in every slide parquet)
    all_df = _coerce_row_manifest_for_parquet(all_df)

    for rec in slide_records:
        slide_df = all_df.loc[all_df["slide_id"].astype(str) == str(rec.slide_id)].copy()
        if slide_df.empty:
            raise RuntimeError(
                f"Internal row-manifest error: slide '{rec.slide_id}' disappeared "
                "during Parquet schema canonicalization."
            )
        slide_df.to_parquet(rows_dir / f"{rec.slide_id}.parquet", index=False)

    all_df.to_parquet(rows_dir / "all_rows.parquet", index=False)

    # Counts of the HEST rows present in the .h5ad but absent from the H&E patch geometry
    dropped_summary_path = rows_dir / "hest_unmatched_patch_rows_dropped.csv"
    if row_filter_reports:
        (
            pd.DataFrame(row_filter_reports)
            .sort_values(["slide_id", "level"])
            .to_csv(dropped_summary_path, index=False)
        )
    elif dropped_summary_path.exists():
        dropped_summary_path.unlink()

    return all_df


def _row_manifest_summary(rows_df: pd.DataFrame) -> pd.DataFrame:
    def _nunique_nonnull(series: pd.Series) -> int:
        s = pd.Series(series).dropna()
        return int(pd.Index(s).nunique()) if not s.empty else 0

    out = (
        rows_df.groupby(["slide_id", "source", "subset", "tissue", "species"], dropna=False)
        .agg(
            n_rows=("row_key", "size"),
            n_unique_patch_ids=("patch_id", _nunique_nonnull),
            n_unique_cell_ids=("cell_id_int", _nunique_nonnull),
        )
        .reset_index()
        .sort_values(["source", "subset", "tissue", "slide_id"])
        .reset_index(drop=True)
    )
    return out


# -----------------------------------------------------------------------------
# Split builders
# -----------------------------------------------------------------------------


def _slide_count_balance_table(rows_df: pd.DataFrame) -> pd.DataFrame:
    return (
        rows_df.groupby(["slide_id", "source", "subset", "tissue"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(["n_rows", "slide_id"], ascending=[False, True])
    )


def _safe_name_component(x: object) -> str:
    import re
    s = str(x).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:80] if len(s) > 80 else s


def _hest_candidate_subsets(cfg: Mapping[str, Any]) -> List[str]:
    sub_cfg = str(cfg["protocol"]["sub_eval_configuration"])
    hest_cfg = cfg["protocol"].get("hest_cv", {})
    subsets = hest_cfg.get("candidate_subsets", {}).get(sub_cfg, [])
    subsets = [str(x) for x in subsets]
    if not subsets:
        raise ValueError(f"No candidate HEST subsets configured for {sub_cfg}.")
    return subsets


def _hest_min_slides_per_tissue(cfg: Mapping[str, Any]) -> int:
    sub_cfg = str(cfg["protocol"]["sub_eval_configuration"])
    hest_cfg = cfg["protocol"].get("hest_cv", {})
    return int(hest_cfg.get("min_slides_per_tissue", {}).get(sub_cfg, 10))


def _hest_slide_stats(rows_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "slide_id",
        "source",
        "subset",
        "tissue",
        "species",
        "dataset_title",
        "patient_raw",
        "patient_clean",
        "split_group_id",
        "row_key",
    }
    missing = sorted(required - set(rows_df.columns))
    if missing:
        raise KeyError(f"HEST CV row manifest is missing required columns: {missing}")

    return (
        rows_df.groupby(
            [
                "slide_id",
                "source",
                "subset",
                "tissue",
                "species",
                "dataset_title",
                "patient_raw",
                "patient_clean",
                "split_group_id",
            ],
            dropna=False,
        )
        .agg(n_rows=("row_key", "size"))
        .reset_index()
        .sort_values(["tissue", "subset", "slide_id"])
        .reset_index(drop=True)
    )


def _group_stats_for_tissue(slide_stats_tissue: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid, gdf in slide_stats_tissue.groupby("split_group_id", sort=True, dropna=False):
        slide_ids = sorted(gdf["slide_id"].astype(str).unique().tolist())
        rows.append(
            {
                "split_group_id": str(gid),
                "slide_ids": slide_ids,
                "n_slides": int(len(slide_ids)),
                "n_rows": int(gdf["n_rows"].sum()),
                "subsets": sorted(gdf["subset"].astype(str).unique().tolist()),
                "dataset_titles": sorted(gdf["dataset_title"].astype(str).unique().tolist()),
                "has_valid_patient": bool(str(gid).startswith("patient::")),
            }
        )
    return pd.DataFrame(rows)


def _group_fold_balance_score(
    *,
    fold_n_slides: np.ndarray,
    fold_n_rows: np.ndarray,
    fold_n_groups: np.ndarray,
    total_slides: int,
    total_rows: int,
    total_groups: int,
    n_outer_folds: int,
) -> float:
    """
    Cost of an assignment of split groups to outer folds. Priority: no empty fold, then slide-count,
    row-count and group-count balance.
    """
    fold_n_slides = np.asarray(fold_n_slides, dtype=np.float64)
    fold_n_rows = np.asarray(fold_n_rows, dtype=np.float64)
    fold_n_groups = np.asarray(fold_n_groups, dtype=np.float64)

    empty_penalty = float(np.sum(fold_n_groups <= 0)) * 1_000_000.0

    target_slides = max(float(total_slides) / float(n_outer_folds), 1.0)
    target_rows = max(float(total_rows) / float(n_outer_folds), 1.0)
    target_groups = max(float(total_groups) / float(n_outer_folds), 1.0)

    slide_rel = (fold_n_slides - target_slides) / target_slides
    row_rel = (fold_n_rows - target_rows) / target_rows
    group_rel = (fold_n_groups - target_groups) / target_groups

    slide_loss = float(np.mean(np.square(slide_rel)) + np.max(np.abs(slide_rel)))
    row_loss = float(np.mean(np.square(row_rel)) + np.max(np.abs(row_rel)))
    group_loss = float(np.mean(np.square(group_rel)) + np.max(np.abs(group_rel)))

    return empty_penalty + 100.0 * slide_loss + 10.0 * row_loss + 1.0 * group_loss


def _assign_groups_to_outer_folds(
    group_stats: pd.DataFrame,
    *,
    n_outer_folds: int,
    seed: int,
) -> Dict[str, int]:
    """
    Assign split groups (patients, else slides) to HEST outer CV folds: one group per fold first, then greedy
    assignment and single-group moves minimizing _group_fold_balance_score. Returns {split_group_id: fold}.
    """
    if group_stats.empty:
        raise RuntimeError("Cannot split an empty group table.")

    n_outer_folds = int(n_outer_folds)
    if int(group_stats.shape[0]) < n_outer_folds:
        raise RuntimeError(
            f"n_split_groups={group_stats.shape[0]} < n_outer_folds={n_outer_folds}"
        )

    required = {"split_group_id", "n_slides", "n_rows"}
    missing = sorted(required - set(group_stats.columns))
    if missing:
        raise KeyError(f"group_stats is missing required columns: {missing}")

    rng = np.random.default_rng(seed)
    gdf = group_stats.copy().reset_index(drop=True)
    gdf["_seed_tie"] = rng.permutation(len(gdf))
    gdf = gdf.sort_values(
        ["n_slides", "n_rows", "_seed_tie", "split_group_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    group_ids = gdf["split_group_id"].astype(str).to_numpy(dtype=object)
    group_n_slides = gdf["n_slides"].to_numpy(dtype=np.int64, copy=False)
    group_n_rows = gdf["n_rows"].to_numpy(dtype=np.int64, copy=False)

    total_groups = int(len(gdf))
    total_slides = int(group_n_slides.sum())
    total_rows = int(group_n_rows.sum())

    fold_n_slides = np.zeros((n_outer_folds,), dtype=np.int64)
    fold_n_rows = np.zeros((n_outer_folds,), dtype=np.int64)
    fold_n_groups = np.zeros((n_outer_folds,), dtype=np.int64)
    assignment_arr = np.full((total_groups,), -1, dtype=np.int32)

    def score_current() -> float:
        return _group_fold_balance_score(
            fold_n_slides=fold_n_slides,
            fold_n_rows=fold_n_rows,
            fold_n_groups=fold_n_groups,
            total_slides=total_slides,
            total_rows=total_rows,
            total_groups=total_groups,
            n_outer_folds=n_outer_folds,
        )

    def apply_group_to_fold(group_idx: int, fold_idx: int, sign: int) -> None:
        fold_n_slides[fold_idx] += sign * int(group_n_slides[group_idx])
        fold_n_rows[fold_idx] += sign * int(group_n_rows[group_idx])
        fold_n_groups[fold_idx] += sign

    # One group per fold (no empty fold)
    for fold_idx, group_idx in enumerate(range(n_outer_folds)):
        assignment_arr[group_idx] = int(fold_idx)
        apply_group_to_fold(group_idx, fold_idx, +1)

    # Assign remaining groups by minimizing the global post-assignment score.
    for group_idx in range(n_outer_folds, total_groups):
        candidates: List[Tuple[float, int, int, int, int]] = []
        for fold_idx in range(n_outer_folds):
            apply_group_to_fold(group_idx, fold_idx, +1)
            sc = score_current()
            apply_group_to_fold(group_idx, fold_idx, -1)
            candidates.append(
                (
                    float(sc),
                    int(fold_n_slides[fold_idx]),
                    int(fold_n_rows[fold_idx]),
                    int(fold_n_groups[fold_idx]),
                    int(fold_idx),
                )
            )
        best_fold = min(candidates)[-1]
        assignment_arr[group_idx] = int(best_fold)
        apply_group_to_fold(group_idx, best_fold, +1)

    # Local refinement: move one group at a time when it improves the global
    # balance and does not empty the source fold.
    best_score = score_current()
    improved = True
    while improved:
        improved = False
        best_move: Optional[Tuple[int, int, int]] = None
        best_move_score = best_score

        for group_idx in range(total_groups):
            old_fold = int(assignment_arr[group_idx])
            if fold_n_groups[old_fold] <= 1:
                continue

            for new_fold in range(n_outer_folds):
                if new_fold == old_fold:
                    continue

                apply_group_to_fold(group_idx, old_fold, -1)
                apply_group_to_fold(group_idx, new_fold, +1)
                sc = score_current()
                apply_group_to_fold(group_idx, new_fold, -1)
                apply_group_to_fold(group_idx, old_fold, +1)

                if sc + 1e-12 < best_move_score:
                    best_move_score = float(sc)
                    best_move = (int(group_idx), int(old_fold), int(new_fold))

        if best_move is not None:
            group_idx, old_fold, new_fold = best_move
            apply_group_to_fold(group_idx, old_fold, -1)
            apply_group_to_fold(group_idx, new_fold, +1)
            assignment_arr[group_idx] = int(new_fold)
            best_score = best_move_score
            improved = True

    if np.any(assignment_arr < 0):
        raise RuntimeError("Internal error: at least one split group was not assigned to an outer fold.")
    if np.any(fold_n_groups <= 0) or np.any(fold_n_slides <= 0):
        raise RuntimeError(
            "Outer group CV produced an empty test fold after balanced assignment: "
            f"fold_n_groups={fold_n_groups.tolist()}, fold_n_slides={fold_n_slides.tolist()}"
        )

    return {
        str(group_ids[i]): int(assignment_arr[i])
        for i in range(total_groups)
    }


def _slides_for_groups(group_stats: pd.DataFrame, group_ids: Sequence[str]) -> List[str]:
    gids = set(map(str, group_ids))
    out: List[str] = []
    for row in group_stats.loc[group_stats["split_group_id"].astype(str).isin(gids)].itertuples(index=False):
        out.extend([str(x) for x in row.slide_ids])
    return sorted(dict.fromkeys(out))


def _build_hest_tissue_candidate_splits(
    *,
    cfg: Mapping[str, Any],
    tissue: str,
    slide_stats_tissue: pd.DataFrame,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    n_outer = int(cfg["protocol"]["cv"]["n_outer_folds"])

    group_stats = _group_stats_for_tissue(slide_stats_tissue)
    if int(group_stats.shape[0]) < n_outer:
        raise RuntimeError(f"n_split_groups={group_stats.shape[0]} < n_outer_folds={n_outer}")

    group_to_fold = _assign_groups_to_outer_folds(
        group_stats,
        n_outer_folds=n_outer,
        seed=seed,
    )

    folds: List[Dict[str, Any]] = []

    for tissue_fold_id in range(n_outer):
        test_groups = sorted([g for g, k in group_to_fold.items() if int(k) == tissue_fold_id])
        test_slides = _slides_for_groups(group_stats, test_groups)

        # All non-test split groups are train groups (no validation split)
        train_groups = sorted(
            set(group_stats["split_group_id"].astype(str).tolist()) - set(test_groups)
        )
        train_slides = _slides_for_groups(group_stats, train_groups)

        if not train_slides or not test_slides:
            raise RuntimeError(
                f"fold {tissue_fold_id}: empty split after group-aware split construction "
                f"train={len(train_slides)} test={len(test_slides)}"
            )
        if len(train_slides) < 2:
            raise RuntimeError(
                f"fold {tissue_fold_id}: {len(train_slides)} train slide(s); alpha selection "
                "needs >= 2 to leave one out."
            )

        if set(train_groups) & set(test_groups):
            raise RuntimeError(f"fold {tissue_fold_id}: split-group leakage detected.")

        train_summary = _summarize_slide_subset(slide_stats_tissue, train_slides)
        test_summary = _summarize_slide_subset(slide_stats_tissue, test_slides)

        folds.append(
            {
                "fold_id": -1,
                "fold_name": "",
                "tissue": str(tissue),
                "tissue_fold_id": int(tissue_fold_id),
                "train_slides": train_slides,
                "val_slides": [],
                "test_slides": test_slides,
                "train_split_groups": train_groups,
                "val_split_groups": [],
                "test_split_groups": test_groups,
                "train_row_count": int(train_summary["n_rows"]),
                "test_row_count": int(test_summary["n_rows"]),
            }
        )

    debug = {
        "n_split_groups": int(group_stats.shape[0]),
        "n_patient_groups": int(group_stats["has_valid_patient"].sum()),
        "n_slide_fallback_groups": int((~group_stats["has_valid_patient"]).sum()),
    }
    return folds, debug


def _eval_split_support_reason(
    *,
    cfg: Mapping[str, Any],
    rows_df: pd.DataFrame,
    slide_ids: Sequence[str],
    trainable_genes: Sequence[str],
    candidate_genes: Sequence[str],
    split_name: str,
) -> Tuple[bool, str]:
    if not slide_ids:
        return False, f"{split_name} split is empty"

    trainable_set = set(map(str, trainable_genes))
    support_df = _support_table_for_split(
        cfg=cfg,
        rows_df=rows_df,
        train_slides=slide_ids,
        candidate_genes=candidate_genes,
    )
    if support_df.empty:
        return False, f"{split_name} support table is empty"

    corr_cfg = cfg["protocol"]["metrics"]["correlation_validity"]
    min_samples = int(corr_cfg["min_samples_per_gene_slide"])
    min_pos = int(corr_cfg["min_nonzero_samples_per_gene_slide"])
    min_unique = int(corr_cfg["min_unique_target_values"])
    min_valid_genes_per_slide, _ = _macro_eligibility_cfg_for_protocol_reports(cfg)

    sub = support_df.loc[support_df["gene"].astype(str).isin(trainable_set)].copy()
    required_cols = {"n_rows", "n_positive", "n_unique_log1p", "target_var_log1p"}
    missing_cols = sorted(required_cols - set(sub.columns))
    if missing_cols:
        return False, (
            f"{split_name} support table is missing metric-validity columns: "
            f"{missing_cols}. Re-run _2_fill_eval_settings.py after updating support-table generation."
        )

    sub["valid_corr_candidate"] = (
        (sub["n_rows"].astype(int) >= min_samples)
        & (sub["n_positive"].astype(int) >= min_pos)
        & (sub["n_unique_log1p"].astype(int) >= min_unique)
        & (sub["target_var_log1p"].astype(float) > 1e-12)
    )

    per_slide = (
        sub.groupby("slide_id", dropna=False)["valid_corr_candidate"]
        .sum()
        .reindex(list(map(str, slide_ids)), fill_value=0)
    )

    bad = per_slide.loc[per_slide.astype(int) < min_valid_genes_per_slide]
    if not bad.empty:
        examples = {str(k): int(v) for k, v in bad.head(5).to_dict().items()}
        return (
            False,
            f"{split_name} metrics likely not computable: fewer than "
            f"{min_valid_genes_per_slide} valid correlation genes on slide(s) {examples}",
        )

    return True, ""


def _macro_eligibility_cfg_for_protocol_reports(cfg: Mapping[str, Any]) -> Tuple[int, int]:
    """(min valid genes per slide, min valid slides per gene) of the gene basis, used by _eval_split_support_reason."""
    mcfg = cfg["protocol"]["metrics"].get("macro_aggregation_validity", {})
    return (
        int(mcfg.get("min_valid_genes_per_slide_for_slide_macro", 1)),
        int(mcfg.get("min_valid_slides_per_gene_for_gene_macro", 1)),
    )


def _gene_payload_for_hest_fold(
    *,
    cfg: Mapping[str, Any],
    rows_df: pd.DataFrame,
    fold: Mapping[str, Any],
    candidate_genes: Sequence[str],
    candidate_panel_file: str = "",
    candidate_panel_key: str = "",
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    min_trainable = int(cfg["protocol"]["gene_selection"]["min_trainable_genes"])

    support_df = _support_table_for_split(
        cfg=cfg,
        rows_df=rows_df,
        train_slides=fold["train_slides"],
        candidate_genes=candidate_genes,
    )

    try:
        artifacts = _training_support_artifacts(cfg, support_df, fold["train_slides"])
    except RuntimeError as e:
        raise RuntimeError(f"fold {fold['tissue_fold_id']} has too few trainable genes: {e}")

    trainable = artifacts["trainable_genes"]
    support_summary = artifacts["support_summary"]
    eligibility_df = artifacts["eligibility_df"]
    slide_summary = artifacts["slide_summary"]

    if len(trainable) < min_trainable:
        raise RuntimeError(
            f"fold {fold['tissue_fold_id']} has only {len(trainable)} trainable genes, "
            f"required min_trainable_genes={min_trainable}"
        )

    _assert_alpha_groups_survived_drop(artifacts, f"fold {fold['tissue_fold_id']}")

    fold_out = dict(fold)
    fold_out["train_slides"] = artifacts["kept_train_slides"]
    fold_out["train_split_groups"] = _split_groups_for_slides(rows_df, fold_out["train_slides"])
    fold_out["test_split_groups"] = _split_groups_for_slides(rows_df, fold_out["test_slides"])

    train_summary = _summarize_slide_subset(_slide_count_balance_table(rows_df), fold_out["train_slides"])
    fold_out["train_row_count"] = int(train_summary["n_rows"])
    fold_out["train_tissue_counts"] = dict(train_summary["tissue_counts"])

    trainable_set = set(map(str, trainable))
    eval_by_slide: Dict[str, List[str]] = {}
    slide_panels: Dict[str, List[str]] = {}

    test_union: set[str] = set()

    for sid in list(fold_out["test_slides"]):
        panel_df = build_slide_gene_panel_manifest(training_dataset_id, sid)
        panel = set(panel_df["gene"].astype(str).tolist())
        measured = sorted(trainable_set & panel)
        if not measured:
            raise RuntimeError(f"test gene panel empty for slide '{sid}'")
        eval_by_slide[str(sid)] = measured
        slide_panels[str(sid)] = sorted(panel)
        test_union.update(measured)

    test_gene_names = sorted(test_union)

    if not test_gene_names:
        raise RuntimeError("test gene panel is empty")

    ok, reason = _eval_split_support_reason(
        cfg=cfg,
        rows_df=rows_df,
        slide_ids=fold_out["test_slides"],
        trainable_genes=trainable,
        candidate_genes=trainable,
        split_name="test",
    )
    if not ok:
        raise RuntimeError(reason)

    payload: Dict[str, Any] = {
        "fold_id": -1,
        "tissue": str(fold["tissue"]),
        "tissue_fold_id": int(fold["tissue_fold_id"]),
        "trainable_genes": trainable,
        "train_gene_names": trainable,
        "test_gene_names": test_gene_names,
        "train_gene_support_csv": "",
        "train_slide_gene_eligibility_csv": "",
        "train_slide_eligibility_summary_csv": "",
        "dropped_weak_train_slides": artifacts["dropped_train_slides"],
        "eval_genes_by_slide": eval_by_slide,
        "slide_panels": slide_panels,
        "candidate_gene_panel_mode": _gene_panel_mode(cfg),
        "candidate_panel_key": str(candidate_panel_key),
        "candidate_panel_file": str(candidate_panel_file),
        "n_candidate_genes": int(len(candidate_genes)),
    }

    debug = {
        "n_trainable_genes": int(len(trainable)),
        "n_test_gene_names": int(len(test_gene_names)),
        **artifacts["debug"],
    }
    return payload, support_summary, eligibility_df, slide_summary, debug, fold_out


def build_hest_tissue_cv_settings(
    *,
    cfg: Mapping[str, Any],
    paths: ProtocolPaths,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    sub_cfg = str(cfg["protocol"]["sub_eval_configuration"])
    if sub_cfg not in HEST_CV_SUB_EVAL_CONFIGS:
        raise ValueError(f"Not a HEST CV protocol: {sub_cfg}")

    # Signature targets: splits built without the per-fold gene gate (targets persisted afterwards)
    target_basis = str(cfg["protocol"].get("target_basis", "genes"))

    candidate_subsets = _hest_candidate_subsets(cfg)
    min_slides = _hest_min_slides_per_tissue(cfg)
    hest_cfg = dict(cfg["protocol"].get("hest_cv", {}) or {})
    n_outer = int(cfg["protocol"]["cv"]["n_outer_folds"])
    seed = int(cfg["protocol"]["protocol_seed"])

    rows_df = persist_row_manifests(cfg, paths.rows_dir)
    _validate_novae_row_manifest_contract(cfg, rows_df)
    _row_manifest_summary(rows_df).to_csv(paths.info_dir / "row_manifest_summary.csv", index=False)

    bad_source = sorted(set(rows_df["source"].astype(str).unique().tolist()) - {"HEST"})
    if bad_source:
        raise RuntimeError(f"HEST CV row manifest contains non-HEST sources: {bad_source}")
    bad_subset = sorted(set(rows_df["subset"].astype(str).unique().tolist()) - set(candidate_subsets))
    if bad_subset:
        raise RuntimeError(f"HEST CV row manifest contains unexpected HEST subsets: {bad_subset}")

    slide_stats = _hest_slide_stats(rows_df)
    support_rows: List[Dict[str, Any]] = []
    split_folds: List[Dict[str, Any]] = []
    gene_folds: List[Dict[str, Any]] = []
    fold_summary_rows: List[Dict[str, Any]] = []
    auto_panels: List[Dict[str, Any]] = []
    auto_diagnostics: List[pd.DataFrame] = []

    global_source_candidate_genes: Optional[List[str]] = None
    if _gene_panel_mode(cfg) != "auto":
        global_source_candidate_genes = _load_global_source_candidate_genes(cfg)

    global_fold_id = 0

    for tissue_idx, (tissue, tstats) in enumerate(slide_stats.groupby("tissue", sort=True, dropna=False)):
        tissue = str(tissue)
        tstats = tstats.copy().reset_index(drop=True)
        group_stats = _group_stats_for_tissue(tstats)

        support_row: Dict[str, Any] = {
            "sub_eval_configuration": sub_cfg,
            "candidate_subsets": ",".join(candidate_subsets),
            "tissue": tissue,
            "n_slides": int(tstats["slide_id"].nunique()),
            "n_split_groups": int(group_stats.shape[0]),
            "n_rows": int(tstats["n_rows"].sum()),
            "supported": False,
            "unsupported_reason": "",
            "n_outer_folds": int(n_outer),
            "trainable_gene_counts_per_fold": "",
        }

        if support_row["n_slides"] < min_slides:
            support_row["unsupported_reason"] = (
                f"n_slides={support_row['n_slides']} < min_slides_per_tissue={min_slides}"
            )
            support_rows.append(support_row)
            continue

        try:
            tissue_folds, split_debug = _build_hest_tissue_candidate_splits(
                cfg=cfg,
                tissue=tissue,
                slide_stats_tissue=tstats,
                seed=seed + 1000 * tissue_idx,
            )
        except RuntimeError as e:
            support_row["unsupported_reason"] = str(e)
            support_rows.append(support_row)
            continue

        if target_basis == "signatures":
            # No per-fold gene gate (the signature coverage gate is config-wide)
            for fold in tissue_folds:
                fold_id = int(global_fold_id)
                tissue_fold_id = int(fold["tissue_fold_id"])
                fold = dict(fold)
                fold["fold_id"] = fold_id
                fold["fold_name"] = f"{sub_cfg}__{_safe_name_component(tissue)}__fold_{tissue_fold_id:02d}"
                save_json(paths.splits_dir / f"fold_{fold_id:02d}.json", fold)
                split_folds.append(fold)
                fold_summary_rows.append({
                    "fold_id": fold_id,
                    "fold_name": fold["fold_name"],
                    "tissue": tissue,
                    "tissue_fold_id": tissue_fold_id,
                    "n_train_slides": int(len(fold["train_slides"])),
                    "n_test_slides": int(len(fold["test_slides"])),
                    "train_row_count": int(fold["train_row_count"]),
                    "test_row_count": int(fold["test_row_count"]),
                    "target_basis": "signatures",
                })
                global_fold_id += 1
            support_row["supported"] = True
            support_row["unsupported_reason"] = ""
            support_row["trainable_gene_counts_per_fold"] = "signatures(member_union)"
            support_rows.append(support_row)
            continue

        pending_auto_panel: Optional[Dict[str, Any]] = None
        pending_auto_diag: Optional[pd.DataFrame] = None

        if _gene_panel_mode(cfg) == "auto":
            tissue_rows = rows_df.loc[rows_df["tissue"].astype(str) == tissue].copy()
            auto_panel, auto_diag = _select_and_persist_auto_gene_panel(
                cfg=cfg,
                genes_dir=paths.genes_dir,
                rows_unit_df=tissue_rows,
                panel_type="tissue",
                panel_key=tissue,
            )
            pending_auto_panel = auto_panel
            pending_auto_diag = auto_diag
            candidate_genes = list(auto_panel["selected_genes"])
            candidate_panel_file = str(auto_panel["panel_file"])
            candidate_panel_key = str(auto_panel["panel_key"])
        else:
            candidate_genes = list(global_source_candidate_genes or [])
            candidate_panel_file = str(
                resolve_G_base_dir(
                    str(cfg["identity"]["training_dataset_id"]),
                    str(cfg["protocol"]["G_base_id"]),
                ) / "G_base_genes.json"
            )
            candidate_panel_key = str(cfg["protocol"]["G_base_id"])

        provisional_gene_payloads: List[Dict[str, Any]] = []
        provisional_support_tables: List[pd.DataFrame] = []
        provisional_eligibility_tables: List[pd.DataFrame] = []
        provisional_slide_eligibility_tables: List[pd.DataFrame] = []
        provisional_gene_debug: List[Dict[str, Any]] = []
        provisional_folds: List[Dict[str, Any]] = []

        failed_reason = ""
        for fold in tissue_folds:
            try:
                payload, support_summary, eligibility_df, slide_summary, gene_debug, fold_out = _gene_payload_for_hest_fold(
                    cfg=cfg,
                    rows_df=rows_df,
                    fold=fold,
                    candidate_genes=candidate_genes,
                    candidate_panel_file=candidate_panel_file,
                    candidate_panel_key=candidate_panel_key,
                )
                provisional_gene_payloads.append(payload)
                provisional_support_tables.append(support_summary)
                provisional_eligibility_tables.append(eligibility_df)
                provisional_slide_eligibility_tables.append(slide_summary)
                provisional_gene_debug.append(gene_debug)
                provisional_folds.append(fold_out)
            except RuntimeError as e:
                failed_reason = str(e)
                break

        if failed_reason:
            support_row["unsupported_reason"] = failed_reason
            support_rows.append(support_row)
            continue

        if pending_auto_panel is not None:
            auto_panels.append(pending_auto_panel)
        if pending_auto_diag is not None:
            auto_diagnostics.append(pending_auto_diag)

        trainable_counts: List[int] = []

        for local_idx, fold in enumerate(provisional_folds):
            fold_id = int(global_fold_id)
            tissue_fold_id = int(fold["tissue_fold_id"])
            fold_name = f"{sub_cfg}__{_safe_name_component(tissue)}__fold_{tissue_fold_id:02d}"

            fold = dict(fold)
            fold["fold_id"] = fold_id
            fold["fold_name"] = fold_name

            gene_payload = dict(provisional_gene_payloads[local_idx])
            gene_payload["fold_id"] = fold_id

            support_csv = paths.genes_dir / f"fold_{fold_id:02d}_train_gene_support.csv"
            provisional_support_tables[local_idx].to_csv(support_csv, index=False)
            gene_payload["train_gene_support_csv"] = str(support_csv)

            eligibility_csv = paths.genes_dir / f"fold_{fold_id:02d}_train_slide_gene_eligibility.csv"
            provisional_eligibility_tables[local_idx].to_csv(eligibility_csv, index=False)
            gene_payload["train_slide_gene_eligibility_csv"] = str(eligibility_csv)

            slide_eligibility_csv = paths.genes_dir / f"fold_{fold_id:02d}_train_slide_eligibility_summary.csv"
            provisional_slide_eligibility_tables[local_idx].to_csv(slide_eligibility_csv, index=False)
            gene_payload["train_slide_eligibility_summary_csv"] = str(slide_eligibility_csv)

            save_json(paths.splits_dir / f"fold_{fold_id:02d}.json", fold)
            save_json(paths.genes_dir / f"fold_{fold_id:02d}_genes.json", gene_payload)

            split_folds.append(fold)
            gene_folds.append(gene_payload)

            n_trainable = int(provisional_gene_debug[local_idx]["n_trainable_genes"])
            trainable_counts.append(n_trainable)

            fold_summary_rows.append(
                {
                    "fold_id": fold_id,
                    "fold_name": fold_name,
                    "tissue": tissue,
                    "tissue_fold_id": tissue_fold_id,
                    "n_train_slides": int(len(fold["train_slides"])),
                    "n_test_slides": int(len(fold["test_slides"])),
                    "n_train_split_groups": int(len(fold["train_split_groups"])),
                    "n_test_split_groups": int(len(fold["test_split_groups"])),
                    "train_row_count": int(fold["train_row_count"]),
                    "test_row_count": int(fold["test_row_count"]),
                    "n_trainable_genes": n_trainable,
                    "n_test_gene_names": int(provisional_gene_debug[local_idx]["n_test_gene_names"]),
                    "n_eligible_slide_gene_pairs": int(provisional_gene_debug[local_idx]["n_eligible_slide_gene_pairs"]),
                    "n_retained_eligible_slide_gene_pairs": int(provisional_gene_debug[local_idx]["n_retained_eligible_slide_gene_pairs"]),
                    "n_dropped_weak_train_slides": int(provisional_gene_debug[local_idx]["n_dropped_weak_train_slides"]),
                    "dropped_weak_train_slides": ",".join(map(str, gene_payload["dropped_weak_train_slides"])),
                }
            )

            global_fold_id += 1

        support_row["supported"] = True
        support_row["unsupported_reason"] = ""
        support_row["trainable_gene_counts_per_fold"] = ",".join(map(str, trainable_counts))
        support_rows.append(support_row)

    auto_manifest: Dict[str, Any] = {}
    auto_manifest_path = ""
    if target_basis != "signatures" and _gene_panel_mode(cfg) == "auto":
        auto_manifest = _persist_auto_gene_panels_manifest(
            cfg=cfg,
            genes_dir=paths.genes_dir,
            panels=auto_panels,
            diagnostics=auto_diagnostics,
        )
        auto_manifest_path = str(paths.genes_dir / "auto_gene_panels_manifest.json")

    splits_manifest: Dict[str, Any] = {
        "type": "slide_cv",
        "split_builder": "hest_tissue_patient_aware_cv_v1",
        "sub_eval_configuration": sub_cfg,
        "source": "HEST",
        "candidate_subsets": candidate_subsets,
        "n_outer_folds": int(n_outer),
        "min_slides_per_tissue": int(min_slides),
        "validation_policy": "none_alpha_by_leave_one_train_group_out",
        # Values from the config (checked by validate_protocol_identity)
        "primary_aggregation": str(hest_cfg.get("primary_aggregation", "tissue_macro")),
        "split_grouping": str(
            hest_cfg.get("split_grouping", "patient_if_available_else_slide")
        ),
        "folds": split_folds,
    }
    if target_basis == "signatures":
        # Placeholder, replaced by the genes_manifest of persist_signature_targets
        genes_manifest = {
            "protocol": sub_cfg,
            "target_basis": "signatures",
            "note": "signature target genes_manifest is written by persist_signature_targets",
        }
    else:
        genes_manifest = {
            "protocol": sub_cfg,
            "gene_panel_strategy": _gene_panel_strategy_payload(
                cfg,
                auto_manifest_path=auto_manifest_path,
            ),
            "folds": gene_folds,
        }
    support_manifest: Dict[str, Any] = {
        "enabled": True,
        "sub_eval_configuration": sub_cfg,
        "source": "HEST",
        "candidate_subsets": candidate_subsets,
        "min_slides_per_tissue": int(min_slides),
        "n_outer_folds": int(n_outer),
        "primary_aggregation": str(hest_cfg.get("primary_aggregation", "tissue_macro")),
        "split_grouping": str(
            hest_cfg.get("split_grouping", "patient_if_available_else_slide")
        ),
        "notes": [
            "HEST CV uses HEST-only training and testing.",
            "STHELAR slides are not used for train/test.",
            (
                "Gene panel strategy is auto: target-informed tissue-specific candidate panels."
                if _gene_panel_mode(cfg) == "auto"
                else "G_base remains STHELAR-derived."
            ),
            "Primary aggregation is macro over supported tissues.",
            "Splitting is patient-aware when valid patient labels exist, otherwise slide-level.",
            "There is no validation split: alpha is chosen by leave-one-train-group-out CV "
            "inside the train set.",
        ],
        "tissues": support_rows,
        "n_supported_tissues": int(sum(bool(r["supported"]) for r in support_rows)),
        "n_total_candidate_tissues": int(len(support_rows)),
        "n_runnable_folds": int(len(split_folds)),
    }

    # Warn when the tissue macro covers one tissue or a supported tissue is close to min_slides_per_tissue
    n_supported = int(support_manifest["n_supported_tissues"])
    supported_rows = [r for r in support_rows if bool(r["supported"])]
    margin = min(
        (int(r["n_slides"]) - int(min_slides) for r in supported_rows), default=0
    )
    support_manifest["tissue_macro_degenerate"] = bool(n_supported <= 1)
    support_manifest["min_supported_tissue_slide_margin"] = int(margin)
    support_manifest["supported_tissues"] = [str(r["tissue"]) for r in supported_rows]
    if n_supported <= 1:
        logger.warning(
            "%s: only %d tissue(s) pass min_slides_per_tissue=%d (%s); the tissue macro "
            "covers at most one tissue.",
            sub_cfg, n_supported, int(min_slides),
            ", ".join(f"{r['tissue']}={r['n_slides']}" for r in supported_rows) or "none",
        )
    elif margin <= 1:
        logger.warning(
            "%s: a supported tissue is within %d slide(s) of min_slides_per_tissue=%d. "
            "Per-tissue counts: %s",
            sub_cfg, margin, int(min_slides),
            ", ".join(f"{r['tissue']}={r['n_slides']}" for r in supported_rows),
        )

    save_json(paths.splits_dir / "splits_manifest.json", splits_manifest)
    save_json(paths.genes_dir / "genes_manifest.json", genes_manifest)
    save_json(paths.splits_dir / "hest_cv_support_manifest.json", support_manifest)

    support_df = pd.DataFrame(support_rows)
    support_df.to_csv(paths.info_dir / "hest_cv_tissue_support.csv", index=False)

    pd.DataFrame(fold_summary_rows).to_csv(
        paths.info_dir / "slide_cv_fold_summary.csv",
        index=False,
    )

    return rows_df, splits_manifest, genes_manifest, support_manifest


def _balanced_fold_capacities(n_items: int, n_folds: int) -> List[int]:
    """Fold sizes for n_items split into n_folds folds (sizes differ by at most 1)."""
    n_items = int(n_items)
    n_folds = int(n_folds)
    if n_folds <= 0:
        raise ValueError("n_folds must be > 0.")
    if n_items < n_folds:
        raise RuntimeError(f"Cannot partition {n_items} slides into {n_folds} non-empty folds.")
    base, rem = divmod(n_items, n_folds)
    return [base + (1 if k < rem else 0) for k in range(n_folds)]


def _fixed_cardinality_partition_score(
    tissue_codes: np.ndarray,
    n_rows: np.ndarray,
    assignment: np.ndarray,
    n_folds: int,
) -> float:
    """Cost of a fixed-size fold assignment: tissue balance (weight 10) and row-count balance (weight 1)."""
    assignment = np.asarray(assignment, dtype=np.int32)
    tissue_codes = np.asarray(tissue_codes, dtype=np.int32)
    n_rows = np.asarray(n_rows, dtype=np.float64)

    target_rows = float(np.sum(n_rows)) / float(n_folds)
    row_sums = np.bincount(assignment, weights=n_rows, minlength=n_folds).astype(np.float64)
    row_den = max(target_rows, 1.0)
    row_rel = (row_sums - target_rows) / row_den
    row_loss = float(np.mean(np.square(row_rel)) + np.max(np.abs(row_rel)))

    tissue_loss = 0.0
    n_tissues = int(np.max(tissue_codes)) + 1 if tissue_codes.size else 0
    for t in range(n_tissues):
        mask = tissue_codes == t
        target_t = float(np.sum(mask)) / float(n_folds)
        counts = np.bincount(assignment[mask], minlength=n_folds).astype(np.float64)
        tissue_den = max(target_t, 1.0)
        tissue_rel = (counts - target_t) / tissue_den
        tissue_loss += float(np.mean(np.square(tissue_rel)) + np.max(np.abs(tissue_rel)))

    return 10.0 * tissue_loss + 1.0 * row_loss


def _improve_fixed_cardinality_partition_by_swaps(
    tissue_codes: np.ndarray,
    n_rows: np.ndarray,
    assignment: np.ndarray,
    n_folds: int,
) -> Tuple[np.ndarray, float]:
    """Improve a fold assignment by pairwise swaps across folds (fold sizes unchanged)."""
    best = np.asarray(assignment, dtype=np.int32).copy()
    best_score = _fixed_cardinality_partition_score(tissue_codes, n_rows, best, n_folds)

    improved = True
    while improved:
        improved = False
        swap_pair = None
        swap_score = best_score

        for i in range(len(best) - 1):
            ai = int(best[i])
            for j in range(i + 1, len(best)):
                aj = int(best[j])
                if ai == aj:
                    continue
                trial = best.copy()
                trial[i], trial[j] = trial[j], trial[i]
                sc = _fixed_cardinality_partition_score(tissue_codes, n_rows, trial, n_folds)
                if sc + 1e-12 < swap_score:
                    swap_score = sc
                    swap_pair = (i, j)

        if swap_pair is not None:
            i, j = swap_pair
            best[i], best[j] = best[j], best[i]
            best_score = swap_score
            improved = True

    return best, float(best_score)


def _canonicalize_fold_labels(slide_ids: Sequence[str], assignment: np.ndarray, n_folds: int) -> np.ndarray:
    """
    Canonical renumbering of folds for stable persisted manifests.
    """
    slide_ids = [str(s) for s in slide_ids]
    assignment = np.asarray(assignment, dtype=np.int32)
    fold_keys = []
    for k in range(n_folds):
        members = tuple(sorted(slide_ids[i] for i in np.flatnonzero(assignment == k)))
        fold_keys.append((members, k))
    ordered_old = [old_k for _, old_k in sorted(fold_keys)]
    remap = {old_k: new_k for new_k, old_k in enumerate(ordered_old)}
    return np.asarray([remap[int(k)] for k in assignment], dtype=np.int32)


def build_fixed_cardinality_group_folds(
    slide_stats: pd.DataFrame,
    n_folds: int,
    seed: int,
) -> Dict[str, int]:
    """
    Build outer test folds with hard slide-cardinality constraints first,
    then optimize tissue balance and row-count balance inside that constraint.
    """
    df = slide_stats[["slide_id", "tissue", "n_rows"]].copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Cannot build folds from an empty slide pool.")

    n_items = int(len(df))
    capacities = np.asarray(_balanced_fold_capacities(n_items, n_folds), dtype=np.int32)
    n_rows = df["n_rows"].to_numpy(dtype=np.float64, copy=False)
    tissue_codes = pd.Categorical(df["tissue"].astype(str)).codes.astype(np.int32, copy=False)
    n_tissues = int(np.max(tissue_codes)) + 1 if tissue_codes.size else 0
    tissue_freq = np.bincount(tissue_codes, minlength=n_tissues).astype(np.int32, copy=False)
    target_rows = float(np.sum(n_rows)) / float(n_folds)

    master_rng = np.random.default_rng(seed)
    n_restarts = max(64, min(1024, 32 * n_items))

    best_assignment: Optional[np.ndarray] = None
    best_score = float("inf")

    for _ in range(n_restarts):
        tie = master_rng.permutation(n_items)
        # Rare tissues first, then larger slides first, then seeded tie-breaks
        order = np.lexsort((tie, -n_rows, tissue_freq[tissue_codes]))

        remaining = capacities.copy()
        fold_sizes = np.zeros(n_folds, dtype=np.int32)
        fold_rows = np.zeros(n_folds, dtype=np.float64)
        fold_tissue_counts = np.zeros((n_folds, n_tissues), dtype=np.int32)
        assignment = np.full(n_items, -1, dtype=np.int32)

        for idx in order:
            idx = int(idx)
            t = int(tissue_codes[idx])
            candidates = np.flatnonzero(remaining > 0)
            if candidates.size == 0:
                raise RuntimeError("Internal error: no remaining fold capacity during fixed-cardinality assignment.")

            tissue_target = float(tissue_freq[t]) / float(n_folds)
            tissue_den = max(tissue_target, 1.0)
            row_den = max(target_rows, 1.0)

            local_scores = []
            for k in candidates.tolist():
                tissue_after = float(fold_tissue_counts[k, t] + 1)
                row_after = float(fold_rows[k] + n_rows[idx])
                fill_after = float(fold_sizes[k] + 1) / float(max(capacities[k], 1))
                sc = (
                    10.0 * ((tissue_after - tissue_target) / tissue_den) ** 2
                    + 1.0 * ((row_after - target_rows) / row_den) ** 2
                    + 1e-6 * fill_after
                )
                local_scores.append(sc)

            local_scores_arr = np.asarray(local_scores, dtype=np.float64)
            best_local = np.min(local_scores_arr)
            keep = np.flatnonzero(np.isclose(local_scores_arr, best_local, rtol=0.0, atol=1e-12))
            chosen = int(candidates[int(master_rng.choice(keep))])

            assignment[idx] = chosen
            remaining[chosen] -= 1
            fold_sizes[chosen] += 1
            fold_rows[chosen] += float(n_rows[idx])
            fold_tissue_counts[chosen, t] += 1

        assignment, score = _improve_fixed_cardinality_partition_by_swaps(
            tissue_codes=tissue_codes,
            n_rows=n_rows,
            assignment=assignment,
            n_folds=n_folds,
        )
        if score + 1e-12 < best_score:
            best_score = score
            best_assignment = assignment.copy()

    if best_assignment is None:
        raise RuntimeError("Failed to construct a fixed-cardinality slide partition.")

    best_assignment = _canonicalize_fold_labels(
        slide_ids=df["slide_id"].astype(str).tolist(),
        assignment=best_assignment,
        n_folds=n_folds,
    )

    return {
        str(df.loc[i, "slide_id"]): int(best_assignment[i])
        for i in range(n_items)
    }


def _summarize_slide_subset(slide_stats: pd.DataFrame, slide_ids: Sequence[str]) -> Dict[str, Any]:
    sub = slide_stats.loc[slide_stats["slide_id"].isin(set(map(str, slide_ids)))].copy()
    tissue_counts = (
        sub["tissue"].astype(str).value_counts().sort_index().to_dict()
        if not sub.empty else {}
    )
    return {
        "n_slides": int(len(sub)),
        "n_rows": int(sub["n_rows"].sum()) if not sub.empty else 0,
        "tissue_counts": {str(k): int(v) for k, v in tissue_counts.items()},
    }


def _split_groups_for_slides(rows_df: pd.DataFrame, slide_ids: Sequence[str]) -> List[str]:
    slide_set = set(map(str, slide_ids))
    if not slide_set:
        return []
    sub = rows_df.loc[rows_df["slide_id"].astype(str).isin(slide_set)]
    if sub.empty:
        return []
    return sorted(sub["split_group_id"].astype(str).dropna().unique().tolist())


def build_slide_level_cv_splits(cfg: Mapping[str, Any], rows_df: pd.DataFrame, splits_dir: Path) -> Dict[str, Any]:
    slide_stats = _slide_count_balance_table(rows_df)
    seed = int(cfg["protocol"]["protocol_seed"])
    sub_cfg = str(cfg["protocol"]["sub_eval_configuration"])
    level = str(cfg["protocol"]["level"])

    if sub_cfg in {"STHELAR_TRAIN_CV", "STHELAR_TRAIN_COORDS"}:
        pool = slide_stats.loc[(slide_stats["source"] == "STHELAR") & (slide_stats["subset"] == "STHELAR_train")].copy()
        n_outer = int(cfg["protocol"]["cv"]["n_outer_folds"])
        if len(pool) < n_outer:
            raise RuntimeError(f"Cannot build {n_outer} outer folds with only {len(pool)} STHELAR_train slides.")

        slide_to_fold = build_fixed_cardinality_group_folds(pool, n_outer, seed)
        out: Dict[str, Any] = {
            "type": "slide_cv",
            "split_builder": "fixed_slide_cardinality_v1",
            "n_pool_slides": int(len(pool)),
            "n_outer_folds": int(n_outer),
            "outer_test_fold_slide_capacities": _balanced_fold_capacities(len(pool), n_outer),
            "balancing_strategy": {
                "primary": "fixed_slide_cardinality",
                "secondary": "tissue_balance",
                "tertiary": "row_count_balance",
            },
            # No validation split: every non-test slide is a train slide
            "validation_policy": "none_alpha_by_leave_one_train_group_out",
            "folds": [],
        }
        for fold_idx in range(n_outer):
            test_slides = sorted([s for s, k in slide_to_fold.items() if k == fold_idx])
            train_slides = sorted(pool.loc[~pool["slide_id"].isin(test_slides), "slide_id"].tolist())
            if len(train_slides) == 0 or len(test_slides) == 0:
                raise RuntimeError(
                    f"Invalid fold construction for fold {fold_idx}: "
                    f"train={len(train_slides)} test={len(test_slides)}."
                )
            if len(train_slides) < 2:
                raise RuntimeError(
                    f"Fold {fold_idx} has {len(train_slides)} train slide(s); alpha selection "
                    "needs >= 2 to leave one out."
                )

            train_summary = _summarize_slide_subset(pool, train_slides)
            test_summary = _summarize_slide_subset(pool, test_slides)

            fold = {
                "fold_id": fold_idx,
                "train_slides": train_slides,
                # No validation slides
                "val_slides": [],
                "test_slides": test_slides,
                "train_row_count": int(train_summary["n_rows"]),
                "test_row_count": int(test_summary["n_rows"]),
                "train_tissue_counts": dict(train_summary["tissue_counts"]),
                "test_tissue_counts": dict(test_summary["tissue_counts"]),
            }
            out["folds"].append(fold)
            save_json(splits_dir / f"fold_{fold_idx:02d}.json", fold)
        save_json(splits_dir / "splits_manifest.json", out)
        return out

    if sub_cfg == "STHELAR_TEST_ONE_SHOT":
        train_pool = slide_stats.loc[(slide_stats["source"] == "STHELAR") & (slide_stats["subset"] == "STHELAR_train")].copy()
        test_pool = slide_stats.loc[(slide_stats["source"] == "STHELAR") & (slide_stats["subset"] == "STHELAR_test")].copy()
        train_slides = sorted(train_pool["slide_id"].tolist())
        if len(train_slides) == 0 or len(test_pool) == 0:
            raise RuntimeError(
                f"Invalid STHELAR_TEST_ONE_SHOT split: train={len(train_slides)} test={len(test_pool)}."
            )
        if len(train_slides) < 2:
            raise RuntimeError(
                f"STHELAR_TEST_ONE_SHOT has {len(train_slides)} train slide(s); alpha selection "
                "needs >= 2 to leave one out."
            )
        manifest = {
            "type": "one_shot",
            "train_slides": train_slides,
            "val_slides": [],
            "test_slides": sorted(test_pool["slide_id"].tolist()),
            "validation_policy": "none_alpha_by_leave_one_train_group_out",
        }
        save_json(splits_dir / "splits_manifest.json", manifest)
        return manifest

    if sub_cfg == "STHELAR_TEST_ADAPTED":
        pool = slide_stats.loc[(slide_stats["source"] == "STHELAR") & (slide_stats["subset"] == "STHELAR_test")].copy()
        manifest = {
            "type": "within_slide_area",
            "slides": sorted(pool["slide_id"].tolist()),
            # Each fold: two train stripes and one test stripe (see common._ADAPTED_STRIPE_ROLES)
            "validation_policy": "none_alpha_by_leave_one_train_group_out",
        }
        save_json(splits_dir / "splits_manifest.json", manifest)
        return manifest

    raise ValueError(f"Unsupported sub_eval_configuration='{sub_cfg}'.")


# -----------------------------------------------------------------------------
# Area splits
# -----------------------------------------------------------------------------


def _partition_stripes(ext_min: float, ext_max: float) -> List[Tuple[float, float]]:
    span = ext_max - ext_min
    cuts = [ext_min, ext_min + span / 3.0, ext_min + 2.0 * span / 3.0, ext_max]
    return [(cuts[i], cuts[i + 1]) for i in range(3)]


def _assign_patch_stripe(geom: pd.DataFrame, axis: str, safe_border_px: float) -> np.ndarray:
    if axis not in {"x", "y"}:
        raise ValueError(axis)
    lo = geom[f"{axis}_min"].min()
    hi = geom[f"{axis}_max"].max()
    stripes = _partition_stripes(float(lo), float(hi))
    out = np.full((len(geom),), -1, dtype=np.int32)
    bbox_lo = geom[f"{axis}_min"].to_numpy(dtype=np.float64)
    bbox_hi = geom[f"{axis}_max"].to_numpy(dtype=np.float64)
    for k, (a, b) in enumerate(stripes):
        left = a if k == 0 else a + safe_border_px
        right = b if k == 2 else b - safe_border_px
        keep = (bbox_lo >= left) & (bbox_hi <= right)
        out[keep] = k
    return out


def _cell_stripe_from_membership(membership_df: pd.DataFrame, patch_to_stripe: Mapping[str, int]) -> pd.DataFrame:
    df = membership_df.copy()
    df["stripe_id"] = df["patch_id"].map(patch_to_stripe).fillna(-1).astype(np.int32)
    grouped = df.groupby("cell_id_int", sort=True)["stripe_id"].agg(list).reset_index()
    stripe_ids = []
    ambiguous = []
    for vals in grouped["stripe_id"]:
        vals_all = [int(v) for v in vals]
        uniq_nonneg = sorted({v for v in vals_all if v >= 0})
        has_outside_or_border_view = any(v < 0 for v in vals_all)
        # A cell gets a stripe only if all its patch views are inside that stripe (else ambiguous, dropped)
        if (not has_outside_or_border_view) and len(uniq_nonneg) == 1:
            stripe_ids.append(uniq_nonneg[0])
            ambiguous.append(False)
        else:
            stripe_ids.append(-1)
            ambiguous.append(True)
    grouped["stripe_id"] = np.asarray(stripe_ids, dtype=np.int32)
    grouped["is_ambiguous"] = np.asarray(ambiguous, dtype=bool)
    return grouped


def build_area_split_manifests(
    cfg: Mapping[str, Any],
    area_dir: Path,
    rows_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    dataset_id = str(cfg["identity"]["dataset_id"])
    level = str(cfg["protocol"]["level"])
    modelgroup = str(cfg["protocol"]["modelgroup"])
    safe_border_px = float(cfg["protocol"]["cv"]["safe_border_px"])
    slide_records = load_slide_records(training_dataset_id)
    eval_root = resolve_eval_dataset_root(training_dataset_id)

    manifest: Dict[str, Any] = {"enabled": True, "slides": []}
    for slide_id, rec in slide_records.items():
        if rec.source != "STHELAR" or rec.subset != "STHELAR_test":
            continue
        geom = load_patch_geometry(eval_root, slide_id)

        # Split feasibility is checked on the protocol rows (rows_df)
        rows_slide_df: Optional[pd.DataFrame] = None
        if rows_df is not None:
            rows_slide_df = rows_df.loc[
                rows_df["slide_id"].astype(str).eq(str(slide_id))
            ].copy()
            if rows_slide_df.empty:
                raise RuntimeError(
                    f"Area split requested for slide '{slide_id}', but the frozen "
                    "row manifest contains zero protocol rows for this slide."
                )

        assignments = {}
        scores = {}
        axis_debug = {}
        for axis in ("x", "y"):
            stripes = _assign_patch_stripe(geom, axis=axis, safe_border_px=safe_border_px)
            if level.startswith("patch"):
                if rows_slide_df is None:
                    counts = [int(np.sum(stripes == k)) for k in range(3)]
                else:
                    patch_to_stripe = dict(zip(geom["patch_id"].astype(str), stripes.tolist()))
                    row_stripes = (
                        rows_slide_df["patch_id"]
                        .astype(str)
                        .map(patch_to_stripe)
                        .fillna(-1)
                        .astype(np.int32)
                        .to_numpy()
                    )
                    counts = [int(np.sum(row_stripes == k)) for k in range(3)]
            else:
                membership = load_membership_table(eval_root, modelgroup, slide_id)
                patch_to_stripe = dict(zip(geom["patch_id"].astype(str), stripes.tolist()))
                cell_stripe = _cell_stripe_from_membership(membership, patch_to_stripe)
                if rows_slide_df is not None:
                    protocol_cell_ids = set(
                        pd.to_numeric(rows_slide_df["cell_id_int"], errors="coerce")
                        .dropna()
                        .astype(np.int64)
                        .tolist()
                    )
                    cell_stripe = cell_stripe.loc[
                        cell_stripe["cell_id_int"].astype(np.int64).isin(protocol_cell_ids)
                    ].copy()
                counts = [int(np.sum(cell_stripe["stripe_id"].to_numpy() == k)) for k in range(3)]
            retained = int(sum(counts))
            n_empty = int(sum(c == 0 for c in counts))
            balance = float(np.std(counts)) if retained > 0 else float("inf")
            # Prefer feasible splits first, then retain as many rows/cells as possible, then prefer better balance
            score = (n_empty, -retained, balance, axis)
            assignments[axis] = stripes
            scores[axis] = score
            axis_debug[axis] = {
                "counts": counts,
                "retained": retained,
                "n_empty_stripes": n_empty,
                "balance_std": balance,
            }
        axis_best = min(scores.keys(), key=lambda ax: scores[ax])
        if axis_debug[axis_best]["n_empty_stripes"] > 0:
            raise RuntimeError(
                f"Area split for slide '{slide_id}' is infeasible with safe_border_px={safe_border_px}. "
                f"Axis diagnostics: {axis_debug}"
            )
        patch_stripe = assignments[axis_best]
        patch_df = geom.copy()
        patch_df["stripe_id"] = patch_stripe
        patch_df.to_parquet(area_dir / f"{slide_id}_patch_stripes.parquet", index=False)

        cell_path = None
        if level == "cell":
            membership = load_membership_table(eval_root, modelgroup, slide_id)
            patch_to_stripe = dict(zip(patch_df["patch_id"].astype(str), patch_df["stripe_id"].astype(int)))
            cell_df = _cell_stripe_from_membership(membership, patch_to_stripe)
            cell_path = area_dir / f"{slide_id}_cell_stripes.parquet"
            cell_df.to_parquet(cell_path, index=False)
        entry = {
            "slide_id": slide_id,
            "chosen_axis": axis_best,
            "safe_border_px": safe_border_px,
            "axis_diagnostics": axis_debug,
            "patch_manifest": str(area_dir / f"{slide_id}_patch_stripes.parquet"),
        }
        if cell_path is not None:
            entry["cell_manifest"] = str(cell_path)
        manifest["slides"].append(entry)
    save_json(area_dir / "area_split_manifest.json", manifest)
    return manifest


def _within_slide_rows_for_area_entry(
    *,
    cfg: Mapping[str, Any],
    rows_df: pd.DataFrame,
    slide_entry: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    sid = str(slide_entry["slide_id"])
    patch_df = pd.read_parquet(slide_entry["patch_manifest"])
    patch_df["patch_id"] = patch_df["patch_id"].astype(str)

    if str(cfg["protocol"]["level"]) == "cell":
        cell_df = pd.read_parquet(slide_entry["cell_manifest"])
        rows_slide = rows_df.loc[
            (rows_df["slide_id"].astype(str) == sid)
            & rows_df["cell_id_int"].isin(
                cell_df.loc[~cell_df["is_ambiguous"], "cell_id_int"]
            )
        ].copy()
        return rows_slide, patch_df, cell_df

    rows_slide = rows_df.loc[
        (rows_df["slide_id"].astype(str) == sid)
        & rows_df["patch_id"].astype(str).isin(
            patch_df.loc[patch_df["stripe_id"] >= 0, "patch_id"].astype(str)
        )
    ].copy()
    return rows_slide, patch_df, None


def _rows_for_area_stripes(
    *,
    cfg: Mapping[str, Any],
    rows_slide: pd.DataFrame,
    patch_df: pd.DataFrame,
    cell_df: Optional[pd.DataFrame],
    stripe_ids: Sequence[int],
) -> pd.DataFrame:
    """Protocol rows of the given stripes."""
    ids = [int(s) for s in stripe_ids]
    if not ids:
        raise ValueError("_rows_for_area_stripes requires at least one stripe id.")
    level = str(cfg["protocol"]["level"])

    if level == "cell":
        if cell_df is None:
            raise RuntimeError("cell_df is required for cell-level area-stripe row selection.")
        stripe_cell_ids = set(
            pd.to_numeric(
                cell_df.loc[cell_df["stripe_id"].astype(int).isin(ids), "cell_id_int"],
                errors="coerce",
            )
            .dropna()
            .astype(np.int64)
            .tolist()
        )
        return rows_slide.loc[
            pd.to_numeric(rows_slide["cell_id_int"], errors="coerce").isin(stripe_cell_ids)
        ].copy()

    stripe_patch_ids = set(
        patch_df.loc[patch_df["stripe_id"].astype(int).isin(ids), "patch_id"]
        .astype(str)
        .tolist()
    )
    return rows_slide.loc[rows_slide["patch_id"].astype(str).isin(stripe_patch_ids)].copy()


def _assert_within_slide_eval_stripes_supported(
    *,
    cfg: Mapping[str, Any],
    rows_slide: pd.DataFrame,
    patch_df: pd.DataFrame,
    cell_df: Optional[pd.DataFrame],
    sid: str,
    fold_name: str,
    test_stripe: int,
    trainable_genes: Sequence[str],
) -> None:
    """Check that the test stripe of a fold has enough valid (gene, location) pairs for the metrics."""
    trainable_genes = [str(g) for g in trainable_genes]
    if not trainable_genes:
        raise RuntimeError(f"{fold_name}: trainable gene list is empty.")

    eval_rows = _rows_for_area_stripes(
        cfg=cfg,
        rows_slide=rows_slide,
        patch_df=patch_df,
        cell_df=cell_df,
        stripe_ids=[int(test_stripe)],
    )
    if eval_rows.empty:
        raise RuntimeError(
            f"{fold_name}: test stripe {int(test_stripe)} contains zero frozen protocol rows."
        )

    ok, reason = _eval_split_support_reason(
        cfg=cfg,
        rows_df=eval_rows,
        slide_ids=[sid],
        trainable_genes=trainable_genes,
        candidate_genes=trainable_genes,
        split_name=f"{fold_name} test",
    )
    if not ok:
        raise RuntimeError(
            f"{fold_name}: unsupported test stripe {int(test_stripe)}. {reason}"
        )


# -----------------------------------------------------------------------------
# Gene protocols
# -----------------------------------------------------------------------------


def _load_G_base_genes(training_dataset_id: str, G_base_id: str) -> List[str]:
    p = resolve_G_base_dir(training_dataset_id, G_base_id) / "G_base_genes.json"
    payload = load_json(p)
    genes = payload.get("genes", [])
    if not genes:
        raise RuntimeError(f"G_base file '{p}' contains no genes.")
    return [str(g) for g in genes]



def _gene_panel_mode(cfg: Mapping[str, Any]) -> str:
    return gene_panel_mode_from_G_base_id(cfg["protocol"]["G_base_id"])


def _is_direct_head_protocol(cfg: Mapping[str, Any]) -> bool:
    return head_type_from_config(cfg) in DIRECT_HEAD_TYPES


def _load_global_source_candidate_genes(cfg: Mapping[str, Any]) -> List[str]:
    if _gene_panel_mode(cfg) == "auto":
        raise RuntimeError(
            "Internal protocol error: G_base_id='auto' requires an explicit "
            "context-specific candidate_genes list. Auto panels must be frozen "
            "during _2_fill_eval_settings.py and must not be loaded implicitly."
        )
    return _load_G_base_genes(
        str(cfg["identity"]["training_dataset_id"]),
        str(cfg["protocol"]["G_base_id"]),
    )


def _dedupe_genes(genes: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for g in genes:
        s = str(g).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _auto_thresholds_for_panel(cfg: Mapping[str, Any], *, panel_type: str) -> Dict[str, Any]:
    gcfg = cfg["protocol"]["gene_selection"]
    if panel_type == "tissue":
        block = dict(gcfg.get("auto_hest", {}))
        defaults = {
            "min_slides_per_gene": 3,
            "min_rows_per_gene": 200,
            "min_positive_rows_per_gene": 30,
            "min_nonzero_rate": 0.001,
            "min_unique_values_per_gene": 3,
            "min_var_log1p": 1.0e-12,
        }
    elif panel_type == "slide":
        block = dict(gcfg.get("auto_sthelar_test_adapted", {}))
        defaults = {
            "min_rows_per_gene": 200,
            "min_positive_rows_per_gene": 30,
            "min_nonzero_rate": 0.001,
            "min_unique_values_per_gene": 3,
            "min_var_log1p": 1.0e-12,
        }
    else:
        raise ValueError(f"Unsupported auto panel_type={panel_type!r}.")

    out: Dict[str, Any] = {}
    for key, default in defaults.items():
        flat_key = f"auto_{key}"
        out[key] = block.get(key, gcfg.get(flat_key, default))
    return out


def _gene_panel_strategy_payload(cfg: Mapping[str, Any], *, auto_manifest_path: str = "") -> Dict[str, Any]:
    mode = _gene_panel_mode(cfg)
    G_base_id = str(cfg["protocol"]["G_base_id"])
    sub_cfg = str(cfg["protocol"]["sub_eval_configuration"])

    if mode != "auto":
        return {
            "mode": "G_base",
            "description": "source-fixed transfer benchmark",
            "target_informed": False,
            "G_base_id": G_base_id,
            "G_base_genes_json": str(
                resolve_G_base_dir(str(cfg["identity"]["training_dataset_id"]), G_base_id)
                / "G_base_genes.json"
            ),
        }

    selection_scope = AUTO_GENE_PANEL_SELECTION_SCOPE[sub_cfg]
    if selection_scope == "per_tissue":
        description = "target-informed diagnostic benchmark with one auto gene panel per HEST tissue"
    elif selection_scope == "per_slide":
        description = "target-informed diagnostic benchmark with one auto gene panel per STHELAR_test slide"
    else:
        raise ValueError(selection_scope)

    return {
        "mode": "auto",
        "description": description,
        "selection_scope": selection_scope,
        "target_informed": True,
        "auto_panels_manifest": str(auto_manifest_path),
    }


def _auto_panel_file_name(*, panel_type: str, panel_key: str) -> str:
    if panel_type == "tissue":
        return f"auto_panel__tissue_{_safe_name_component(panel_key)}.json"
    if panel_type == "slide":
        return f"auto_panel__slide_{_safe_name_component(panel_key)}.json"
    raise ValueError(panel_type)


def _scgpt_vocab_genes(cfg: Mapping[str, Any]) -> List[str]:
    return [str(g) for g in load_scgpt_vocab(str(cfg["protocol"]["scgpt_model_name"]))]


def _auto_gene_stats_for_slide_rows(
    *,
    cfg: Mapping[str, Any],
    rows_slide_df: pd.DataFrame,
    slide_id: str,
    scgpt_vocab_genes: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, set]]:
    """
    Compute target-expression statistics for one slide and a protocol-selected
    row subset. Raw counts are canonicalized before log1p.
    """
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    slide_records = load_slide_records(training_dataset_id)
    eval_root = resolve_eval_dataset_root(training_dataset_id)
    rec = slide_records[str(slide_id)]
    adata_path = eval_root / "st" / f"{slide_id}.h5ad"
    map_path = eval_root / "mouse_to_human_gene_mappings" / f"{slide_id}_mouse2human_mapping.parquet"

    mat = canonicalize_gene_table(
        adata_path=adata_path,
        slide_record=rec,
        gene_subset=scgpt_vocab_genes,
        mouse_mapping_path=map_path if map_path.exists() else None,
    )
    if mat.X.shape[1] == 0:
        return pd.DataFrame(), {}

    a_back = ad.read_h5ad(adata_path, backed="r")
    try:
        obs_index = pd.Index(a_back.obs_names.astype(str))
    finally:
        if getattr(a_back, "file", None) is not None:
            a_back.file.close()

    requested_row_keys = rows_slide_df["row_key"].astype(str).to_numpy(dtype=object)
    row_idx = obs_index.get_indexer(pd.Index(requested_row_keys, dtype=object))
    if np.any(row_idx < 0):
        missing = requested_row_keys[row_idx < 0][:10].tolist()
        raise KeyError(
            f"Some auto-panel row_keys are missing from '{adata_path}' for slide '{slide_id}'. "
            f"Examples: {missing}"
        )

    vals = np.asarray(mat.X[row_idx, :], dtype=np.float32)
    if vals.shape[0] == 0 or vals.shape[1] == 0:
        return pd.DataFrame(), {}

    vals_log1p = np.log1p(vals, dtype=np.float32)
    n_rows = int(vals.shape[0])
    n_positive = np.sum(vals > 0, axis=0, dtype=np.int64)
    sum_log1p = np.sum(vals_log1p, axis=0, dtype=np.float64)
    sum_sq_log1p = np.sum(np.square(vals_log1p, dtype=np.float64), axis=0, dtype=np.float64)
    mean_log1p = sum_log1p / max(float(n_rows), 1.0)
    var_log1p = np.maximum(
        sum_sq_log1p / max(float(n_rows), 1.0) - np.square(mean_log1p),
        0.0,
    )

    unique_values_by_gene: Dict[str, set] = {}
    for j, gene in enumerate(mat.genes.astype(object).tolist()):
        unique_values_by_gene[str(gene)] = set(np.unique(vals_log1p[:, j]).astype(float).tolist())

    rows = []
    for gene, n_pos, sum_lp, sum_sq_lp, mean_lp, var_lp in zip(
        mat.genes.astype(object).tolist(),
        n_positive.tolist(),
        sum_log1p.tolist(),
        sum_sq_log1p.tolist(),
        mean_log1p.tolist(),
        var_log1p.tolist(),
    ):
        rows.append(
            {
                "slide_id": str(slide_id),
                "gene": str(gene),
                "n_rows": int(n_rows),
                "n_positive": int(n_pos),
                "sum_log1p": float(sum_lp),
                "sum_sq_log1p": float(sum_sq_lp),
                "mean_log1p": float(mean_lp),
                "var_log1p": float(var_lp),
                "in_scgpt_vocab": True,
            }
        )
    return pd.DataFrame(rows), unique_values_by_gene


def _select_and_persist_auto_gene_panel(
    *,
    cfg: Mapping[str, Any],
    genes_dir: Path,
    rows_unit_df: pd.DataFrame,
    panel_type: str,
    panel_key: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Select and save one target-informed candidate gene panel (fold train-support filtering is applied later)."""
    rows_unit_df = rows_unit_df.copy()
    if rows_unit_df.empty:
        raise RuntimeError(f"Cannot build auto gene panel for {panel_type}={panel_key!r}: zero rows.")

    thresholds = _auto_thresholds_for_panel(cfg, panel_type=panel_type)
    auto_n = int(cfg["protocol"]["gene_selection"].get("auto_n_genes", 256))
    min_trainable = int(cfg["protocol"]["gene_selection"]["min_trainable_genes"])
    scgpt_vocab = _scgpt_vocab_genes(cfg)

    stat_frames: List[pd.DataFrame] = []
    unique_values_by_gene: Dict[str, set] = {}

    for sid, sdf in rows_unit_df.groupby("slide_id", sort=True, dropna=False):
        slide_stats, slide_uniques = _auto_gene_stats_for_slide_rows(
            cfg=cfg,
            rows_slide_df=sdf,
            slide_id=str(sid),
            scgpt_vocab_genes=scgpt_vocab,
        )
        if not slide_stats.empty:
            stat_frames.append(slide_stats)
        for gene, vals in slide_uniques.items():
            unique_values_by_gene.setdefault(str(gene), set()).update(vals)

    if not stat_frames:
        raise RuntimeError(
            f"Auto gene panel for {panel_type}={panel_key!r} has no scGPT-compatible "
            "canonical genes measured in the selected rows."
        )

    stats = pd.concat(stat_frames, axis=0, ignore_index=True)
    agg = (
        stats.groupby("gene", sort=True)
        .agg(
            n_slides_measured=("slide_id", "nunique"),
            n_rows=("n_rows", "sum"),
            n_positive_rows=("n_positive", "sum"),
            sum_log1p=("sum_log1p", "sum"),
            sum_sq_log1p=("sum_sq_log1p", "sum"),
            in_scgpt_vocab=("in_scgpt_vocab", "max"),
        )
        .reset_index()
    )
    rows_arr = np.asarray(agg["n_rows"], dtype=np.float64)
    mean = np.asarray(agg["sum_log1p"], dtype=np.float64) / np.clip(rows_arr, 1.0, None)
    var = (
        np.asarray(agg["sum_sq_log1p"], dtype=np.float64) / np.clip(rows_arr, 1.0, None)
        - np.square(mean)
    )
    agg["nonzero_rate"] = np.asarray(agg["n_positive_rows"], dtype=np.float64) / np.clip(rows_arr, 1.0, None)
    agg["mean_log1p"] = mean
    agg["var_log1p"] = np.clip(var, a_min=0.0, a_max=None)
    agg["n_unique_log1p"] = [
        int(len(unique_values_by_gene.get(str(g), set())))
        for g in agg["gene"].astype(str).tolist()
    ]
    agg["score"] = agg["var_log1p"].astype(float)

    keep = (
        agg["in_scgpt_vocab"].astype(bool)
        & (agg["n_rows"].astype(int) >= int(thresholds["min_rows_per_gene"]))
        & (agg["n_positive_rows"].astype(int) >= int(thresholds["min_positive_rows_per_gene"]))
        & (agg["nonzero_rate"].astype(float) >= float(thresholds.get("min_nonzero_rate", 0.0)))
        & (agg["n_unique_log1p"].astype(int) >= int(thresholds["min_unique_values_per_gene"]))
        & (agg["var_log1p"].astype(float) > float(thresholds["min_var_log1p"]))
    )
    if panel_type == "tissue":
        keep &= agg["n_slides_measured"].astype(int) >= int(thresholds["min_slides_per_gene"])

    ranked = (
        agg.loc[keep]
        .sort_values(
            ["score", "n_positive_rows", "n_rows", "n_slides_measured", "gene"],
            ascending=[False, False, False, False, True],
        )
        .reset_index(drop=True)
    )
    selected_genes = ranked["gene"].astype(str).head(auto_n).tolist()

    if len(selected_genes) < min_trainable:
        diag_path = genes_dir / "auto_gene_panel_diagnostics.csv"
        raise RuntimeError(
            f"Auto gene panel for {panel_type}={panel_key!r} has only {len(selected_genes)} genes "
            f"after filtering, below min_trainable_genes={min_trainable}. "
            f"Relax auto thresholds or inspect {diag_path}."
        )

    selected_set = set(selected_genes)
    diag = agg.copy()
    diag.insert(0, "panel_key", str(panel_key))
    diag.insert(0, "panel_type", str(panel_type))
    diag["tissue"] = str(panel_key) if panel_type == "tissue" else ""
    diag["slide_id"] = str(panel_key) if panel_type == "slide" else ""
    diag["selected"] = diag["gene"].astype(str).isin(selected_set)
    rank_lut = {g: i + 1 for i, g in enumerate(selected_genes)}
    diag["rank"] = diag["gene"].astype(str).map(rank_lut)
    diag["ranking_metric"] = "variance_log1p"

    panel_file = genes_dir / _auto_panel_file_name(panel_type=panel_type, panel_key=str(panel_key))
    panel_payload: Dict[str, Any] = {
        "panel_key": str(panel_key),
        "panel_type": str(panel_type),
        "selected_genes": selected_genes,
        "n_selected_genes": int(len(selected_genes)),
        "input_slide_ids": sorted(rows_unit_df["slide_id"].astype(str).unique().tolist()),
        "input_sources": sorted(rows_unit_df["source"].astype(str).unique().tolist()),
        "input_subsets": sorted(rows_unit_df["subset"].astype(str).unique().tolist()),
        "input_tissues": sorted(rows_unit_df["tissue"].astype(str).unique().tolist()),
        "thresholds": thresholds,
        "ranking_metric": "variance_log1p",
        "target_informed": True,
    }
    if panel_type == "tissue":
        panel_payload["tissue"] = str(panel_key)
    elif panel_type == "slide":
        panel_payload["slide_id"] = str(panel_key)

    save_json(panel_file, panel_payload)
    panel_payload["panel_file"] = str(panel_file)
    return panel_payload, diag


def _persist_auto_gene_panels_manifest(
    *,
    cfg: Mapping[str, Any],
    genes_dir: Path,
    panels: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[pd.DataFrame],
) -> Dict[str, Any]:
    sub_cfg = str(cfg["protocol"]["sub_eval_configuration"])
    selection_scope = AUTO_GENE_PANEL_SELECTION_SCOPE[sub_cfg]
    manifest_path = genes_dir / "auto_gene_panels_manifest.json"
    diagnostics_path = genes_dir / "auto_gene_panel_diagnostics.csv"

    if diagnostics:
        pd.concat(diagnostics, axis=0, ignore_index=True).to_csv(diagnostics_path, index=False)
    else:
        pd.DataFrame(columns=[
            "panel_type", "panel_key", "tissue", "slide_id", "gene", "selected",
            "rank", "score", "n_slides_measured", "n_rows", "n_positive_rows",
            "nonzero_rate", "mean_log1p", "var_log1p", "n_unique_log1p",
            "in_scgpt_vocab",
        ]).to_csv(diagnostics_path, index=False)

    panel_files = {
        str(p["panel_key"]): str(p["panel_file"])
        for p in panels
    }
    panel_type = "tissue" if selection_scope == "per_tissue" else "slide"
    thresholds = _auto_thresholds_for_panel(cfg, panel_type=panel_type)

    payload = {
        "strategy": "auto",
        "selection_scope": selection_scope,
        "target_informed": True,
        "eval_configuration": str(cfg["protocol"]["eval_configuration"]),
        "sub_eval_configuration": sub_cfg,
        "level": str(cfg["protocol"]["level"]),
        "auto_n_genes": int(cfg["protocol"]["gene_selection"].get("auto_n_genes", 256)),
        "ranking_metric": "variance_log1p",
        "thresholds": thresholds,
        "number_of_panels": int(len(panels)),
        "panel_files": panel_files,
        "diagnostics_csv": str(diagnostics_path),
        "scgpt_model_name": str(cfg["protocol"]["scgpt_model_name"]),
    }
    save_json(manifest_path, payload)
    return payload


def _row_keys_for_protocol(
    rows_df: pd.DataFrame,
    slide_ids: Sequence[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for sid in slide_ids:
        sub = rows_df.loc[rows_df["slide_id"] == sid].copy()
        out[sid] = sub["row_key"].astype(str).to_numpy(dtype=object)
    return out


def _support_table_for_split(
    *,
    cfg: Mapping[str, Any],
    rows_df: pd.DataFrame,
    train_slides: Sequence[str],
    candidate_genes: Optional[Sequence[str]] = None,
    row_filter_by_slide: Optional[Mapping[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    if candidate_genes is None:
        candidate_genes = _load_global_source_candidate_genes(cfg)
    candidate_genes = _dedupe_genes(candidate_genes)
    if not candidate_genes:
        raise RuntimeError("support_table_for_split received an empty candidate gene universe.")

    slide_records = load_slide_records(training_dataset_id)
    eval_root = resolve_eval_dataset_root(training_dataset_id)

    rows = []
    for sid in train_slides:
        slide_rows_df = rows_df.loc[rows_df["slide_id"] == sid].copy()
        if row_filter_by_slide is not None:
            keep_keys = set(map(str, row_filter_by_slide.get(sid, [])))
            slide_rows_df = slide_rows_df.loc[slide_rows_df["row_key"].astype(str).isin(keep_keys)].copy()
        if slide_rows_df.empty:
            continue
        rec = slide_records[sid]
        adata_path = eval_root / "st" / f"{sid}.h5ad"
        map_path = eval_root / "mouse_to_human_gene_mappings" / f"{sid}_mouse2human_mapping.parquet"
        mat = canonicalize_gene_table(
            adata_path=adata_path,
            slide_record=rec,
            gene_subset=candidate_genes,
            mouse_mapping_path=map_path if map_path.exists() else None,
        )

        a_back = ad.read_h5ad(adata_path, backed="r")
        try:
            obs_index = pd.Index(a_back.obs_names.astype(str))
        finally:
            if getattr(a_back, "file", None) is not None:
                a_back.file.close()

        requested_row_keys = slide_rows_df["row_key"].astype(str).to_numpy(dtype=object)
        row_idx = obs_index.get_indexer(pd.Index(requested_row_keys, dtype=object))
        if np.any(row_idx < 0):
            missing = requested_row_keys[row_idx < 0][:10].tolist()
            raise KeyError(
                f"Some persisted protocol row_keys are missing from '{adata_path}' for slide '{sid}'. "
                f"Examples: {missing}"
            )

        if mat.X.shape[1] == 0:
            continue

        vals = np.asarray(mat.X[row_idx, :], dtype=np.float32)
        vals_log1p = np.log1p(vals, dtype=np.float32)
        n_rows = int(vals.shape[0])
        n_positive = np.sum(vals > 0, axis=0, dtype=np.int64)
        mean_log1p = (
            np.mean(vals_log1p, axis=0, dtype=np.float64)
            if vals_log1p.size
            else np.zeros((vals.shape[1],), dtype=np.float64)
        )
        sum_log1p = np.sum(vals_log1p, axis=0, dtype=np.float64)
        sum_sq_log1p = np.sum(np.square(vals_log1p, dtype=np.float64), axis=0, dtype=np.float64)
        var_log1p = np.maximum(
            sum_sq_log1p / max(float(n_rows), 1.0) - np.square(mean_log1p),
            0.0,
        )
        n_unique_log1p = np.asarray(
            [np.unique(vals_log1p[:, j]).size for j in range(vals_log1p.shape[1])],
            dtype=np.int64,
        )

        for gene, n_pos, mean_lp, sum_lp, sum_sq_lp, var_lp, n_unique_lp in zip(
            mat.genes.astype(object).tolist(),
            n_positive.tolist(),
            mean_log1p.tolist(),
            sum_log1p.tolist(),
            sum_sq_log1p.tolist(),
            var_log1p.tolist(),
            n_unique_log1p.tolist(),
        ):
            rows.append(
                {
                    "slide_id": sid,
                    "gene": str(gene),
                    "n_rows": n_rows,
                    "n_positive": int(n_pos),
                    "measured": 1,
                    "mean_log1p": float(mean_lp),
                    "sum_log1p": float(sum_lp),
                    "sum_sq_log1p": float(sum_sq_lp),
                    "target_var_log1p": float(var_lp),
                    "n_unique_log1p": int(n_unique_lp),
                }
            )

    return pd.DataFrame(rows)


def _summarize_support_df(support_df: pd.DataFrame) -> pd.DataFrame:
    if support_df.empty:
        return pd.DataFrame(columns=[
            "gene",
            "train_support_slides",
            "train_support_rows",
            "train_support_positive_rows",
            "train_nonzero_rate",
            "train_mean_log1p",
            "train_dispersion",
        ])
    agg = (
        support_df.groupby("gene", sort=True)
        .agg(
            train_support_slides=("slide_id", "nunique"),
            train_support_rows=("n_rows", "sum"),
            train_support_positive_rows=("n_positive", "sum"),
            sum_log1p=("sum_log1p", "sum"),
            sum_sq_log1p=("sum_sq_log1p", "sum"),
        )
        .reset_index()
    )
    rows = np.asarray(agg["train_support_rows"], dtype=np.float64)
    mean = np.divide(np.asarray(agg["sum_log1p"], dtype=np.float64), np.clip(rows, 1.0, None))
    var = np.divide(np.asarray(agg["sum_sq_log1p"], dtype=np.float64), np.clip(rows, 1.0, None)) - np.square(mean)
    agg["train_nonzero_rate"] = np.divide(
        np.asarray(agg["train_support_positive_rows"], dtype=np.float64),
        np.clip(rows, 1.0, None),
    )
    agg["train_mean_log1p"] = mean
    agg["train_dispersion"] = np.clip(var, a_min=0.0, a_max=None)
    return agg[[
        "gene",
        "train_support_slides",
        "train_support_rows",
        "train_support_positive_rows",
        "train_nonzero_rate",
        "train_mean_log1p",
        "train_dispersion",
    ]].copy()


def _slide_gene_eligibility_from_support(
    cfg: Mapping[str, Any],
    support_df: pd.DataFrame,
) -> pd.DataFrame:
    """Mark each training (slide, gene) pair as eligible or not (gene-level thresholds are applied afterwards)."""
    base_cols = [
        "slide_id",
        "gene",
        "n_rows",
        "n_positive",
        "n_unique_log1p",
        "target_var_log1p",
    ]
    if support_df.empty:
        cols = list(dict.fromkeys(list(support_df.columns) + base_cols + ["eligible_slide_gene"]))
        return pd.DataFrame(columns=cols)

    missing = sorted(set(base_cols) - set(support_df.columns))
    if missing:
        raise KeyError(
            "Training support table is missing columns required for slide-gene "
            f"eligibility filtering: {missing}"
        )

    gcfg = cfg["protocol"]["gene_selection"]
    out = support_df.copy()
    out["slide_id"] = out["slide_id"].astype(str)
    out["gene"] = out["gene"].astype(str)

    for col in ["n_rows", "n_positive", "n_unique_log1p"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int64)
    out["target_var_log1p"] = pd.to_numeric(
        out["target_var_log1p"], errors="coerce"
    ).fillna(0.0).astype(np.float64)

    out["eligible_slide_gene"] = (
        (out["n_rows"] >= int(gcfg.get("min_train_rows_per_gene_slide", 1)))
        & (out["n_positive"] >= int(gcfg.get("min_train_positive_rows_per_gene_slide", 1)))
        & (out["n_unique_log1p"] >= int(gcfg.get("min_train_unique_values_per_gene_slide", 1)))
        & (out["target_var_log1p"] > float(gcfg.get("min_train_var_log1p_per_gene_slide", 0.0)))
    ).astype(bool)
    return out


def _eligible_support_from_support(
    cfg: Mapping[str, Any],
    support_df: pd.DataFrame,
) -> pd.DataFrame:
    eligibility_df = _slide_gene_eligibility_from_support(cfg, support_df)
    return eligibility_df.loc[eligibility_df["eligible_slide_gene"].astype(bool)].copy()


def _training_slide_eligibility_summary(
    cfg: Mapping[str, Any],
    eligibility_df: pd.DataFrame,
    train_slides: Sequence[str],
) -> pd.DataFrame:
    """Per-train-slide eligibility summary and weak-slide filter (n_train_rows_on_slide = row count of the slide)."""
    gcfg = cfg["protocol"]["gene_selection"]
    slide_ids = list(dict.fromkeys(map(str, train_slides)))
    out = pd.DataFrame({"slide_id": slide_ids})

    if eligibility_df.empty:
        out["n_train_rows_on_slide"] = 0
        out["n_slide_gene_pairs"] = 0
        out["n_eligible_slide_gene_pairs"] = 0
        out["n_eligible_genes_on_slide"] = 0
    else:
        edf = eligibility_df.copy()
        edf["slide_id"] = edf["slide_id"].astype(str)
        edf["gene"] = edf["gene"].astype(str)
        edf["eligible_slide_gene"] = edf["eligible_slide_gene"].astype(bool)

        rows_per_slide = (
            edf.groupby("slide_id", dropna=False)["n_rows"]
            .max()
            .rename("n_train_rows_on_slide")
            .reset_index()
        )
        pair_counts = (
            edf.groupby("slide_id", dropna=False)
            .agg(
                n_slide_gene_pairs=("gene", "size"),
                n_eligible_slide_gene_pairs=("eligible_slide_gene", "sum"),
            )
            .reset_index()
        )
        eligible_genes = (
            edf.loc[edf["eligible_slide_gene"]]
            .groupby("slide_id", dropna=False)["gene"]
            .nunique()
            .rename("n_eligible_genes_on_slide")
            .reset_index()
        )

        out = out.merge(rows_per_slide, on="slide_id", how="left")
        out = out.merge(pair_counts, on="slide_id", how="left")
        out = out.merge(eligible_genes, on="slide_id", how="left")
        for col in [
            "n_train_rows_on_slide",
            "n_slide_gene_pairs",
            "n_eligible_slide_gene_pairs",
            "n_eligible_genes_on_slide",
        ]:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int64)

    min_rows = int(gcfg.get("min_train_rows_per_slide", 0))
    min_genes = int(gcfg.get("min_train_eligible_genes_per_slide", 0))
    out["drop_weak_train_slide"] = (
        (out["n_train_rows_on_slide"] < min_rows)
        | (out["n_eligible_genes_on_slide"] < min_genes)
    ).astype(bool)

    reasons: List[str] = []
    for row in out.itertuples(index=False):
        r = []
        if int(row.n_train_rows_on_slide) < min_rows:
            r.append(f"n_train_rows_on_slide={int(row.n_train_rows_on_slide)} < {min_rows}")
        if int(row.n_eligible_genes_on_slide) < min_genes:
            r.append(f"n_eligible_genes_on_slide={int(row.n_eligible_genes_on_slide)} < {min_genes}")
        reasons.append("; ".join(r))
    out["train_slide_drop_reason"] = reasons
    out["keep_train_slide"] = ~out["drop_weak_train_slide"]
    return out


def _assert_alpha_groups_survived_drop(
    artifacts: Mapping[str, Any],
    where: str,
) -> None:
    """Check that at least 2 train slides remain after weak-slide filtering (slide-level protocols only)."""
    kept = [str(s) for s in artifacts["kept_train_slides"]]
    if len(kept) >= 2:
        return
    dropped = [str(s) for s in artifacts["dropped_train_slides"]]
    raise RuntimeError(
        f"{where}: weak-slide filtering left {len(kept)} train slide(s) ({kept}); alpha "
        f"selection needs >= 2 to leave one out. Dropped: {dropped}. Relax "
        "gene_selection.min_train_rows_per_slide / "
        "gene_selection.min_train_eligible_genes_per_slide, or inspect the support table."
    )


def _training_support_artifacts(
    cfg: Mapping[str, Any],
    support_df: pd.DataFrame,
    train_slides: Sequence[str],
) -> Dict[str, Any]:
    """
    Fold-level gene-selection artifacts: eligible (slide, gene) pairs, weak train slides dropped,
    trainable genes and support summary from the retained eligible pairs.
    """
    eligibility_df = _slide_gene_eligibility_from_support(cfg, support_df)
    slide_summary = _training_slide_eligibility_summary(cfg, eligibility_df, train_slides)

    slide_meta_cols = [
        "slide_id",
        "n_train_rows_on_slide",
        "n_eligible_genes_on_slide",
        "n_slide_gene_pairs",
        "n_eligible_slide_gene_pairs",
        "keep_train_slide",
        "drop_weak_train_slide",
        "train_slide_drop_reason",
    ]
    eligibility_df = eligibility_df.merge(
        slide_summary[slide_meta_cols],
        on="slide_id",
        how="left",
    )

    kept_train_slides = (
        slide_summary.loc[slide_summary["keep_train_slide"], "slide_id"]
        .astype(str)
        .tolist()
    )
    dropped_train_slides = (
        slide_summary.loc[slide_summary["drop_weak_train_slide"], "slide_id"]
        .astype(str)
        .tolist()
    )
    if not kept_train_slides:
        raise RuntimeError(
            "All training slides were dropped by weak-slide filtering. "
            "Relax gene_selection.min_train_rows_per_slide or "
            "gene_selection.min_train_eligible_genes_per_slide, or inspect the support table."
        )

    retained = eligibility_df.loc[
        eligibility_df["slide_id"].astype(str).isin(set(kept_train_slides))
    ].copy()
    trainable = _trainable_genes_from_support(cfg, retained)
    trainable_set = set(map(str, trainable))

    retained_eligible_trainable = retained.loc[
        retained["eligible_slide_gene"].astype(bool)
        & retained["gene"].astype(str).isin(trainable_set)
    ].copy()
    support_summary = _summarize_support_df(retained_eligible_trainable)

    debug = {
        "n_train_slides_before_drop": int(len(list(dict.fromkeys(map(str, train_slides))))),
        "n_train_slides_after_drop": int(len(kept_train_slides)),
        "n_dropped_weak_train_slides": int(len(dropped_train_slides)),
        "n_slide_gene_pairs": int(eligibility_df.shape[0]),
        "n_eligible_slide_gene_pairs": int(eligibility_df["eligible_slide_gene"].sum()) if not eligibility_df.empty else 0,
        "n_retained_eligible_slide_gene_pairs": int(retained_eligible_trainable.shape[0]),
        "n_trainable_genes": int(len(trainable)),
    }
    return {
        "trainable_genes": trainable,
        "support_summary": support_summary,
        "eligibility_df": eligibility_df,
        "slide_summary": slide_summary,
        "retained_support_df": retained,
        "kept_train_slides": kept_train_slides,
        "dropped_train_slides": dropped_train_slides,
        "debug": debug,
    }


def _single_unit_training_support_artifacts(
    cfg: Mapping[str, Any],
    support_df: pd.DataFrame,
    train_slides: Sequence[str],
) -> Dict[str, Any]:
    """Fold train support for STHELAR_TEST_ADAPTED + auto (single slide: min_train_slides_per_gene is not applied)."""
    eligibility_df = _slide_gene_eligibility_from_support(cfg, support_df)
    slide_summary = _training_slide_eligibility_summary(cfg, eligibility_df, train_slides)

    slide_meta_cols = [
        "slide_id",
        "n_train_rows_on_slide",
        "n_eligible_genes_on_slide",
        "n_slide_gene_pairs",
        "n_eligible_slide_gene_pairs",
        "keep_train_slide",
        "drop_weak_train_slide",
        "train_slide_drop_reason",
    ]
    eligibility_df = eligibility_df.merge(
        slide_summary[slide_meta_cols],
        on="slide_id",
        how="left",
    )

    kept_train_slides = (
        slide_summary.loc[slide_summary["keep_train_slide"], "slide_id"]
        .astype(str)
        .tolist()
    )
    dropped_train_slides = (
        slide_summary.loc[slide_summary["drop_weak_train_slide"], "slide_id"]
        .astype(str)
        .tolist()
    )
    if not kept_train_slides:
        raise RuntimeError(
            "The within-slide train stripe was dropped by weak-slide filtering. "
            "Relax gene_selection.min_train_rows_per_slide or "
            "gene_selection.min_train_eligible_genes_per_slide."
        )

    retained = eligibility_df.loc[
        eligibility_df["slide_id"].astype(str).isin(set(kept_train_slides))
    ].copy()
    retained_eligible = retained.loc[retained["eligible_slide_gene"].astype(bool)].copy()
    trainable = sorted(retained_eligible["gene"].astype(str).unique().tolist())

    min_trainable = int(cfg["protocol"]["gene_selection"]["min_trainable_genes"])
    if len(trainable) < min_trainable:
        raise RuntimeError(
            f"Too few trainable genes in within-slide fold after train-stripe filtering: "
            f"{len(trainable)} < min_trainable_genes={min_trainable}."
        )

    support_summary = _summarize_support_df(retained_eligible)
    debug = {
        "n_train_slides_before_drop": int(len(list(dict.fromkeys(map(str, train_slides))))),
        "n_train_slides_after_drop": int(len(kept_train_slides)),
        "n_dropped_weak_train_slides": int(len(dropped_train_slides)),
        "n_slide_gene_pairs": int(eligibility_df.shape[0]),
        "n_eligible_slide_gene_pairs": int(eligibility_df["eligible_slide_gene"].sum()) if not eligibility_df.empty else 0,
        "n_retained_eligible_slide_gene_pairs": int(retained_eligible.shape[0]),
        "n_trainable_genes": int(len(trainable)),
    }
    return {
        "trainable_genes": trainable,
        "support_summary": support_summary,
        "eligibility_df": eligibility_df,
        "slide_summary": slide_summary,
        "retained_support_df": retained,
        "kept_train_slides": kept_train_slides,
        "dropped_train_slides": dropped_train_slides,
        "debug": debug,
    }


def _trainable_genes_from_support(cfg: Mapping[str, Any], support_df: pd.DataFrame) -> List[str]:
    if support_df.empty:
        return []
    gcfg = cfg["protocol"]["gene_selection"]
    eligible = _eligible_support_from_support(cfg, support_df)
    if eligible.empty:
        raise RuntimeError(
            "Too few trainable genes after support filtering: 0 eligible slide-gene pairs."
        )
    agg = (
        eligible.groupby("gene", sort=True)
        .agg(
            n_train_slides=("slide_id", "nunique"),
            n_rows=("n_rows", "sum"),
            n_positive=("n_positive", "sum"),
        )
        .reset_index()
    )
    keep = agg.loc[
        (agg["n_train_slides"] >= int(gcfg["min_train_slides_per_gene"]))
        & (agg["n_rows"] >= int(gcfg["min_train_rows_per_gene"]))
        & (agg["n_positive"] >= int(gcfg["min_train_positive_rows_per_gene"]))
    ].copy()
    genes = keep["gene"].astype(str).tolist()
    if len(genes) < int(gcfg["min_trainable_genes"]):
        raise RuntimeError(
            "Too few trainable genes after eligible slide-gene filtering: "
            f"{len(genes)} < {gcfg['min_trainable_genes']}. "
            f"Eligible slide-gene pairs={int(eligible.shape[0])}; "
            "relax gene_selection thresholds or inspect the persisted "
            "train_slide_gene_eligibility.csv table."
        )
    return genes


def persist_gene_protocols(
    cfg: Mapping[str, Any],
    rows_df: pd.DataFrame,
    splits_manifest: Mapping[str, Any],
    genes_dir: Path,
    *,
    splits_dir: Optional[Path] = None,
    area_manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    sub_cfg = str(cfg["protocol"]["sub_eval_configuration"])
    seed = int(cfg["protocol"]["protocol_seed"])
    out: Dict[str, Any] = {"protocol": sub_cfg}

    if splits_manifest["type"] == "slide_cv":
        splits_manifest = copy.deepcopy(dict(splits_manifest))
        folds = []
        for fold in splits_manifest["folds"]:
            support_df = _support_table_for_split(
                cfg=cfg,
                rows_df=rows_df,
                train_slides=fold["train_slides"],
            )
            artifacts = _training_support_artifacts(cfg, support_df, fold["train_slides"])
            _assert_alpha_groups_survived_drop(
                artifacts, f"fold {int(fold['fold_id'])}"
            )

            if artifacts["dropped_train_slides"]:
                fold["train_slides"] = artifacts["kept_train_slides"]
                train_summary = _summarize_slide_subset(
                    _slide_count_balance_table(rows_df),
                    fold["train_slides"],
                )
                fold["train_row_count"] = int(train_summary["n_rows"])
                fold["train_tissue_counts"] = dict(train_summary["tissue_counts"])
                if splits_dir is not None:
                    save_json(splits_dir / f"fold_{int(fold['fold_id']):02d}.json", fold)

            support_summary = artifacts["support_summary"]
            support_csv = genes_dir / f"fold_{int(fold['fold_id']):02d}_train_gene_support.csv"
            support_summary.to_csv(support_csv, index=False)

            eligibility_csv = genes_dir / f"fold_{int(fold['fold_id']):02d}_train_slide_gene_eligibility.csv"
            artifacts["eligibility_df"].to_csv(eligibility_csv, index=False)

            slide_eligibility_csv = genes_dir / f"fold_{int(fold['fold_id']):02d}_train_slide_eligibility_summary.csv"
            artifacts["slide_summary"].to_csv(slide_eligibility_csv, index=False)

            trainable = artifacts["trainable_genes"]
            train_gene_names = sorted(trainable)
            test_gene_names = list(train_gene_names)
            eval_by_slide = {}
            slide_panels: Dict[str, List[str]] = {}
            for sid in fold["test_slides"]:
                panel_df = build_slide_gene_panel_manifest(training_dataset_id, sid)
                panel = set(panel_df["gene"].astype(str).tolist())
                slide_panels[sid] = sorted(panel)
                eval_by_slide[sid] = sorted([g for g in train_gene_names if g in panel])

                if not eval_by_slide[sid]:
                    raise RuntimeError(
                        f"Fold {fold['fold_id']} has an empty eval gene panel for slide '{sid}'."
                    )

            ok, reason = _eval_split_support_reason(
                cfg=cfg,
                rows_df=rows_df,
                slide_ids=fold["test_slides"],
                trainable_genes=test_gene_names,
                candidate_genes=test_gene_names,
                split_name=f"fold {fold['fold_id']} test",
            )
            if not ok:
                raise RuntimeError(reason)

            payload = {
                "fold_id": int(fold["fold_id"]),
                "fold_name": str(fold.get("fold_name", f"fold_{int(fold['fold_id']):02d}")),
                "trainable_genes": trainable,
                "train_gene_names": train_gene_names,
                "test_gene_names": test_gene_names,
                "train_gene_support_csv": str(support_csv),
                "train_slide_gene_eligibility_csv": str(eligibility_csv),
                "train_slide_eligibility_summary_csv": str(slide_eligibility_csv),
                "dropped_weak_train_slides": artifacts["dropped_train_slides"],
                "train_slides_after_drop": artifacts["kept_train_slides"],
                "eval_genes_by_slide": eval_by_slide,
                "slide_panels": slide_panels,
            }
            save_json(genes_dir / f"fold_{int(fold['fold_id']):02d}_genes.json", payload)
            folds.append(payload)
        out["folds"] = folds
        if splits_dir is not None:
            save_json(splits_dir / "splits_manifest.json", splits_manifest)
        save_json(genes_dir / "genes_manifest.json", out)
        return out

    if splits_manifest["type"] == "one_shot":
        splits_manifest = copy.deepcopy(dict(splits_manifest))
        support_df = _support_table_for_split(
            cfg=cfg,
            rows_df=rows_df,
            train_slides=splits_manifest["train_slides"],
        )
        artifacts = _training_support_artifacts(cfg, support_df, splits_manifest["train_slides"])
        _assert_alpha_groups_survived_drop(artifacts, "STHELAR_TEST_ONE_SHOT")
        if artifacts["dropped_train_slides"]:
            splits_manifest["train_slides"] = artifacts["kept_train_slides"]
            if splits_dir is not None:
                save_json(splits_dir / "splits_manifest.json", splits_manifest)

        support_summary = artifacts["support_summary"]
        support_csv = genes_dir / "oneshot_train_gene_support.csv"
        support_summary.to_csv(support_csv, index=False)

        eligibility_csv = genes_dir / "oneshot_train_slide_gene_eligibility.csv"
        artifacts["eligibility_df"].to_csv(eligibility_csv, index=False)

        slide_eligibility_csv = genes_dir / "oneshot_train_slide_eligibility_summary.csv"
        artifacts["slide_summary"].to_csv(slide_eligibility_csv, index=False)

        trainable = artifacts["trainable_genes"]
        eval_by_slide = {}
        for sid in splits_manifest["test_slides"]:
            panel_df = build_slide_gene_panel_manifest(training_dataset_id, sid)
            panel = set(panel_df["gene"].astype(str).tolist())
            eval_by_slide[sid] = sorted([g for g in trainable if g in panel])
            if not eval_by_slide[sid]:
                raise RuntimeError(
                    f"One-shot protocol has an empty eval gene panel for slide '{sid}'."
                )

        ok, reason = _eval_split_support_reason(
            cfg=cfg,
            rows_df=rows_df,
            slide_ids=splits_manifest["test_slides"],
            trainable_genes=trainable,
            candidate_genes=trainable,
            split_name="one-shot test",
        )
        if not ok:
            raise RuntimeError(reason)

        payload = {
            "fold_name": "oneshot",
            "trainable_genes": trainable,
            "train_gene_names": trainable,
            "test_gene_names": trainable,
            "train_gene_support_csv": str(support_csv),
            "train_slide_gene_eligibility_csv": str(eligibility_csv),
            "train_slide_eligibility_summary_csv": str(slide_eligibility_csv),
            "dropped_weak_train_slides": artifacts["dropped_train_slides"],
            "train_slides_after_drop": artifacts["kept_train_slides"],
            "eval_genes_by_slide": eval_by_slide,
        }
        save_json(genes_dir / "oneshot_genes.json", payload)
        out.update(payload)
        save_json(genes_dir / "genes_manifest.json", out)
        return out

    if splits_manifest["type"] == "within_slide_area":
        if _gene_panel_mode(cfg) == "auto":
            if area_manifest is None:
                raise RuntimeError("STHELAR_TEST_ADAPTED + auto requires a persisted area_manifest.")

            auto_panels: List[Dict[str, Any]] = []
            auto_diagnostics: List[pd.DataFrame] = []
            fold_payloads: List[Dict[str, Any]] = []

            for slide_entry in area_manifest.get("slides", []):
                sid = str(slide_entry["slide_id"])
                rows_slide, patch_df, cell_df = _within_slide_rows_for_area_entry(
                    cfg=cfg,
                    rows_df=rows_df,
                    slide_entry=slide_entry,
                )

                auto_panel, auto_diag = _select_and_persist_auto_gene_panel(
                    cfg=cfg,
                    genes_dir=genes_dir,
                    rows_unit_df=rows_slide,
                    panel_type="slide",
                    panel_key=sid,
                )
                auto_panels.append(auto_panel)
                auto_diagnostics.append(auto_diag)
                candidate_genes = list(auto_panel["selected_genes"])

                for fold_idx, roles in enumerate(_ADAPTED_STRIPE_ROLES):
                    train_stripes = stripes_for_role(roles, "train")
                    test_stripe = stripes_for_role(roles, "test")[0]
                    fold_name = f"{sid}_fold_{fold_idx}"
                    safe_fold = _safe_name_component(fold_name)

                    # Support on the union of the train stripes (one (slide, gene) support row)
                    train_rows = _rows_for_area_stripes(
                        cfg=cfg,
                        rows_slide=rows_slide,
                        patch_df=patch_df,
                        cell_df=cell_df,
                        stripe_ids=train_stripes,
                    )

                    area_support = _support_table_for_split(
                        cfg=cfg,
                        rows_df=train_rows,
                        train_slides=[sid],
                        candidate_genes=candidate_genes,
                    )
                    artifacts = _single_unit_training_support_artifacts(
                        cfg,
                        area_support,
                        [sid],
                    )

                    support_csv = genes_dir / f"within_slide_{safe_fold}_train_gene_support.csv"
                    artifacts["support_summary"].to_csv(support_csv, index=False)

                    eligibility_csv = genes_dir / f"within_slide_{safe_fold}_train_slide_gene_eligibility.csv"
                    artifacts["eligibility_df"].to_csv(eligibility_csv, index=False)

                    slide_eligibility_csv = genes_dir / f"within_slide_{safe_fold}_train_slide_eligibility_summary.csv"
                    artifacts["slide_summary"].to_csv(slide_eligibility_csv, index=False)

                    trainable = artifacts["trainable_genes"]

                    _assert_within_slide_eval_stripes_supported(
                        cfg=cfg,
                        rows_slide=rows_slide,
                        patch_df=patch_df,
                        cell_df=cell_df,
                        sid=sid,
                        fold_name=fold_name,
                        test_stripe=test_stripe,
                        trainable_genes=trainable,
                    )

                    payload = {
                        "fold_name": fold_name,
                        "slide_id": sid,
                        "area_fold_idx": int(fold_idx),
                        "train_stripes": [int(s) for s in train_stripes],
                        "test_stripe": int(test_stripe),
                        "gene_panel_mode": "auto",
                        "selection_scope": "per_slide",
                        "auto_panel_file": str(auto_panel["panel_file"]),
                        "candidate_panel_key": sid,
                        "n_candidate_genes": int(len(candidate_genes)),
                        "trainable_genes": trainable,
                        "train_gene_names": trainable,
                        "test_gene_names": trainable,
                        "train_gene_support_csv": str(support_csv),
                        "train_slide_gene_eligibility_csv": str(eligibility_csv),
                        "train_slide_eligibility_summary_csv": str(slide_eligibility_csv),
                        "dropped_weak_train_slides": artifacts["dropped_train_slides"],
                        "train_slides_after_drop": artifacts["kept_train_slides"],
                        "eval_genes_by_slide": {sid: trainable},
                    }
                    save_json(genes_dir / f"within_slide_{safe_fold}_genes.json", payload)
                    fold_payloads.append(payload)

            auto_manifest = _persist_auto_gene_panels_manifest(
                cfg=cfg,
                genes_dir=genes_dir,
                panels=auto_panels,
                diagnostics=auto_diagnostics,
            )
            auto_manifest_path = str(genes_dir / "auto_gene_panels_manifest.json")

            payload = {
                "protocol": sub_cfg,
                "gene_panel_mode": "auto",
                "selection_scope": "per_slide",
                "gene_panel_strategy": _gene_panel_strategy_payload(
                    cfg,
                    auto_manifest_path=auto_manifest_path,
                ),
                "auto_gene_panels_manifest": auto_manifest_path,
                "folds": fold_payloads,
            }
            save_json(genes_dir / "within_slide_genes.json", payload)
            out.update(payload)
            save_json(genes_dir / "genes_manifest.json", out)
            return out

        # same trainable gene universe from a separate STHELAR_train row universe only
        train_support_records = [
            rec
            for rec in read_metadata(training_dataset_id)
            if rec.source == "STHELAR" and rec.subset == "STHELAR_train"
        ]
        if not train_support_records:
            raise RuntimeError("within_slide_area gene protocol requires at least one STHELAR_train slide to define the trainable gene universe.")
        rows_train = pd.concat(
            [_build_row_manifest_frame(cfg, rec) for rec in train_support_records],
            axis=0,
            ignore_index=True,
        )
        support_df = _support_table_for_split(
            cfg=cfg,
            rows_df=rows_train,
            train_slides=sorted(rows_train["slide_id"].astype(str).unique().tolist()),
        )
        artifacts = _training_support_artifacts(
            cfg,
            support_df,
            sorted(rows_train["slide_id"].astype(str).unique().tolist()),
        )
        support_summary = artifacts["support_summary"]
        # Support tables of the trainable gene universe
        universe_support_csv = genes_dir / "within_slide_train_gene_support.csv"
        support_summary.to_csv(universe_support_csv, index=False)

        universe_eligibility_csv = genes_dir / "within_slide_train_universe_slide_gene_eligibility.csv"
        artifacts["eligibility_df"].to_csv(universe_eligibility_csv, index=False)

        universe_slide_eligibility_csv = genes_dir / "within_slide_train_universe_slide_eligibility_summary.csv"
        artifacts["slide_summary"].to_csv(universe_slide_eligibility_csv, index=False)

        trainable = artifacts["trainable_genes"]
        source_trainable = list(trainable)
 
        # ------------------------------------------------------------------
        # STHELAR_TEST_ADAPTED + G_base + direct head
        # Per fold: output genes = source-trainable genes eligible on the train stripes
        # ------------------------------------------------------------------
        if _is_direct_head_protocol(cfg):
            if area_manifest is None:
                raise RuntimeError(
                    "STHELAR_TEST_ADAPTED + G_base + direct head requires a persisted "
                    "area_manifest so fold-local supervised direct gene panels can be built."
                )

            candidate_panel_file = str(
                resolve_G_base_dir(
                    str(cfg["identity"]["training_dataset_id"]),
                    str(cfg["protocol"]["G_base_id"]),
                ) / "G_base_genes.json"
            )
            candidate_panel_key = str(cfg["protocol"]["G_base_id"])

            fold_payloads: List[Dict[str, Any]] = []

            for slide_entry in area_manifest.get("slides", []):
                sid = str(slide_entry["slide_id"])
                rows_slide, patch_df, cell_df = _within_slide_rows_for_area_entry(
                    cfg=cfg,
                    rows_df=rows_df,
                    slide_entry=slide_entry,
                )

                for fold_idx, roles in enumerate(_ADAPTED_STRIPE_ROLES):
                    train_stripes = stripes_for_role(roles, "train")
                    test_stripe = stripes_for_role(roles, "test")[0]

                    fold_name = f"{sid}_fold_{fold_idx}"
                    safe_fold = _safe_name_component(fold_name)

                    # Union of the train stripes
                    train_rows = _rows_for_area_stripes(
                        cfg=cfg,
                        rows_slide=rows_slide,
                        patch_df=patch_df,
                        cell_df=cell_df,
                        stripe_ids=train_stripes,
                    )

                    area_support = _support_table_for_split(
                        cfg=cfg,
                        rows_df=train_rows,
                        train_slides=[sid],
                        candidate_genes=source_trainable,
                    )
                    area_artifacts = _single_unit_training_support_artifacts(
                        cfg,
                        area_support,
                        [sid],
                    )

                    fold_trainable = area_artifacts["trainable_genes"]

                    _assert_within_slide_eval_stripes_supported(
                        cfg=cfg,
                        rows_slide=rows_slide,
                        patch_df=patch_df,
                        cell_df=cell_df,
                        sid=sid,
                        fold_name=fold_name,
                        test_stripe=test_stripe,
                        trainable_genes=fold_trainable,
                    )

                    support_csv = genes_dir / f"within_slide_{safe_fold}_train_gene_support.csv"
                    area_artifacts["support_summary"].to_csv(support_csv, index=False)

                    eligibility_csv = genes_dir / f"within_slide_{safe_fold}_train_slide_gene_eligibility.csv"
                    area_artifacts["eligibility_df"].to_csv(eligibility_csv, index=False)

                    slide_eligibility_csv = genes_dir / f"within_slide_{safe_fold}_train_slide_eligibility_summary.csv"
                    area_artifacts["slide_summary"].to_csv(slide_eligibility_csv, index=False)

                    fold_payload = {
                        "fold_name": fold_name,
                        "slide_id": sid,
                        "area_fold_idx": int(fold_idx),
                        "train_stripes": [int(s) for s in train_stripes],
                        "test_stripe": int(test_stripe),
                        "gene_panel_mode": "G_base",
                        "direct_head_fold_local_outputs": True,
                        "candidate_panel_key": candidate_panel_key,
                        "candidate_panel_file": candidate_panel_file,
                        "n_source_trainable_genes": int(len(source_trainable)),
                        "n_candidate_genes": int(len(source_trainable)),
                        "trainable_genes": fold_trainable,
                        "train_gene_names": fold_trainable,
                        "test_gene_names": fold_trainable,
                        "train_gene_support_csv": str(support_csv),
                        "train_slide_gene_eligibility_csv": str(eligibility_csv),
                        "train_slide_eligibility_summary_csv": str(slide_eligibility_csv),
                        "dropped_weak_train_slides": area_artifacts["dropped_train_slides"],
                        "train_slides_after_drop": area_artifacts["kept_train_slides"],
                        "eval_genes_by_slide": {sid: fold_trainable},
                    }
                    save_json(genes_dir / f"within_slide_{safe_fold}_genes.json", fold_payload)
                    fold_payloads.append(fold_payload)

            if not fold_payloads:
                raise RuntimeError(
                    "No fold-local direct gene payloads were created for "
                    "STHELAR_TEST_ADAPTED + G_base + direct head."
                )

            payload = {
                "protocol": sub_cfg,
                "gene_panel_mode": "G_base",
                "gene_panel_strategy": _gene_panel_strategy_payload(cfg),
                "direct_head_fold_local_outputs": True,
                "description": (
                    "G_base defines a source-fixed candidate pool. For direct heads, "
                    "each within-slide area fold restricts outputs/evaluation to genes "
                    "with target train-stripe supervision."
                ),
                "source_trainable_genes": source_trainable,
                "source_train_gene_support_csv": str(universe_support_csv),
                "source_train_universe_slide_gene_eligibility_csv": str(universe_eligibility_csv),
                "source_train_universe_slide_eligibility_summary_csv": str(universe_slide_eligibility_csv),
                "folds": fold_payloads,
            }
            save_json(genes_dir / "within_slide_genes.json", payload)
            out.update(payload)
            save_json(genes_dir / "genes_manifest.json", out)
            return out

    raise ValueError(f"Unsupported splits_manifest type='{splits_manifest['type']}'.")


# -----------------------------------------------------------------------------
# Novae domains in the row manifest
# -----------------------------------------------------------------------------


def _validate_novae_row_manifest_contract(cfg: Mapping[str, Any], rows_df: pd.DataFrame) -> None:
    """Check that the requested Novae domains are in the row manifest."""
    structure_cfg = cfg["protocol"].get("structure", {})
    novae_training_id = structure_cfg.get("novae_training_id", None)

    if novae_training_id is None or str(novae_training_id).strip() == "":
        return

    level = str(cfg["protocol"]["level"])
    if level != "patch_all":
        raise ValueError(
            f"protocol.structure.novae_training_id='{novae_training_id}' was set, "
            f"but Novae patch-domain labels attach only to patch rows. "
            f"Current level='{level}'."
        )

    st_mask = rows_df["source"].astype(str).eq("STHELAR")
    if not bool(st_mask.any()):
        return

    if "novae_domain" not in rows_df.columns:
        raise RuntimeError(
            f"Novae domains were requested with novae_training_id='{novae_training_id}', "
            "but the frozen row manifest has no 'novae_domain' column. "
            "This means labels were not attached during settings fill."
        )

    # Partial nulls are expected (unassigned Novae cells): raise only if every STHELAR row is null
    missing = rows_df.loc[st_mask, "novae_domain"].isna()
    if bool(missing.all()):
        raise RuntimeError(
            f"Novae domains were requested with novae_training_id='{novae_training_id}', "
            f"but all {int(st_mask.sum())} STHELAR rows have null Novae labels "
            "(labels not attached during settings fill)."
        )


# -----------------------------------------------------------------------------
# Manifest notes
# -----------------------------------------------------------------------------


def _protocol_contract_notes(cfg: Mapping[str, Any]) -> List[str]:
    notes = [
        "protocol_seed fixes the persisted benchmark definition: splits, row universes, and gene sets. It must remain identical across encoders within one eval_id.",
        "The probe is closed-form and deterministic, so it has no seed and there is exactly one run per (encoder, space). Probe-seed variance is zero by construction.",
        "Per-slide eval gene panels are persisted during settings fill and validated against measured targets at run time.",
        (
            "Training gene support is pair-aware: weak training (slide,gene) pairs are "
            "removed before counting trainable genes, trainable genes require enough "
            "eligible slides, and ineligible training pairs are masked to NaN at run time."
        ),
        (
            "Because the probe weights every training slide equally (w_i = 1/n_rows(slide)), "
            "weak training slides may be dropped if they have too few rows or too few "
            "eligible genes."
        ),
        "Genes are always filtered through the scGPT vocabulary during protocol construction for controlled comparisons and robust gene-name handling.",
        "The probe predicts the supervised train-gene panel directly with one unpenalized bias per gene. scGPT is used only to filter the gene vocabulary, never for prediction.",
        (
            "The probe is a closed-form weighted ridge, solved independently per target on "
            "the slides where that target is eligible. It is deterministic: no probe seeds, "
            "no validation split, no early stopping."
        ),
        (
            "Feature preprocessing is a single stage: raw encoder features -> feature "
            "geometry (he=identity, he_l2/clip=row_l2) -> probe. There is no dimensionality "
            "reduction; the probe consumes the full-dimensional backbone features."
        ),
        (
            "Probe capacity is set by one alpha per (encoder, fold), chosen by "
            "leave-one-train-group-out CV on train rows only and scored on "
            "slide_macro_spearman. alpha = lambda * trace(A)/D, so the grid is invariant to "
            "feature scale and comparable across he / he_l2 / clip. The realized capacity is "
            "reported per fold as effective_dof."
        ),
    ]
    if str(cfg["protocol"]["sub_eval_configuration"]) in {
        "STHELAR_TRAIN_CV",
        "STHELAR_TRAIN_COORDS",
    }:
        notes.append("For slide_cv protocols, outer test folds are built with fixed slide cardinality first (exactly equal when possible, otherwise as equal as mathematically possible), then optimized for tissue balance and row-count balance.")
    if str(cfg["protocol"]["sub_eval_configuration"]) in CONTROL_SUB_EVAL_CONFIGS:
        notes.append("This sub-protocol is a child control: it reuses the rows, splits, genes and protocol settings of reference_eval_id.")
    if str(cfg["protocol"]["sub_eval_configuration"]) in HEST_CV_SUB_EVAL_CONFIGS:
        notes.append("HEST CV uses HEST-only training and testing. STHELAR slides are not used for train/test.")
        notes.append("For HEST CV, the primary scalar benchmark metrics are tissue macros over supported tissues.")
        notes.append("HEST CV splitting is patient-aware when valid patient labels exist; otherwise slide_id is used as the split group.")
        if _gene_panel_mode(cfg) == "auto":
            notes.append("Gene panel strategy: auto. This is target-informed and selects one frozen candidate panel per HEST tissue.")
    if str(cfg["protocol"]["sub_eval_configuration"]) == "STHELAR_TEST_ADAPTED":
        if _gene_panel_mode(cfg) == "auto":
            notes.append("Gene panel strategy: auto. This is target-informed and selects one frozen candidate panel per STHELAR_test slide.")
        else:
            if _is_direct_head_protocol(cfg):
                notes.append(
                    "STHELAR_TEST_ADAPTED + G_base + direct head: G_base defines a "
                    "source-fixed candidate pool, but each direct head is built only "
                    "on genes with fold-local target train-stripe supervision. "
                    "This prevents untrained direct-output columns from being evaluated."
                )
            else:
                notes.append("STHELAR_TEST_ADAPTED evaluates only STHELAR_test rows, but its trainable gene universe is defined from a separate STHELAR_train support row universe.")
    return notes


# -----------------------------------------------------------------------------
# Target normalization statistics
# -----------------------------------------------------------------------------


def persist_target_norm_stats(
    cfg: Mapping[str, Any],
    rows_df: pd.DataFrame,
    genes_manifest: Mapping[str, Any],
    out_dir: Path,
    *,
    row_scope_record: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Save the per-(slide, gene) z-score stats (mean/std of the depth-normalized target) of the measured genes."""
    mkdir(out_dir)
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    target_transform = dict(cfg["protocol"].get("target_transform", {}))
    level = str(cfg["protocol"]["level"])
    slide_records = load_slide_records(training_dataset_id)
    eval_root = resolve_eval_dataset_root(training_dataset_id)

    # Union of every gene that can serve as a probe/target column.
    seen: set = set()
    gene_union: List[str] = []
    for payload in _iter_gene_payloads_for_reports(genes_manifest):
        for g in payload.get("train_gene_names", payload.get("trainable_genes", [])):
            if str(g) not in seen:
                seen.add(str(g))
                gene_union.append(str(g))
    gene_union = sorted(gene_union)
    if not gene_union:
        raise RuntimeError("persist_target_norm_stats: no trainable genes found in genes_manifest.")

    rows: List[Dict[str, Any]] = []
    for slide_id in sorted(rows_df["slide_id"].astype(str).unique().tolist()):
        rec = slide_records[slide_id]
        slide_rows = rows_df.loc[rows_df["slide_id"].astype(str) == slide_id]
        row_keys = slide_rows["row_key"].astype(str).to_numpy(dtype=object)
        adata_path = eval_root / "st" / f"{slide_id}.h5ad"
        mouse_map = eval_root / "mouse_to_human_gene_mappings" / f"{slide_id}_mouse2human_mapping.parquet"
        Y_stage1, detected = extract_targets_for_rows(
            adata_path=adata_path,
            slide_record=rec,
            level=level,
            ordered_row_keys=row_keys,
            ordered_gene_list=gene_union,
            target_transform=target_transform,
            mouse_mapping_path=mouse_map if mouse_map.exists() else None,
            zscore_stats=None,
        )
        for j, gene in enumerate(gene_union):
            col = Y_stage1[:, j]
            finite = np.isfinite(col)
            n = int(np.sum(finite))
            if n == 0:
                continue  # gene not measured on this slide
            vals = col[finite].astype(np.float64)
            rows.append(
                {
                    "slide_id": slide_id,
                    "gene": str(gene),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n_rows": n,
                    "n_nonzero": int(np.sum(detected[finite, j])),
                }
            )

    stats_df = pd.DataFrame(rows, columns=["slide_id", "gene", "mean", "std", "n_rows", "n_nonzero"])
    stats_path = out_dir / "zscore_stats.parquet"
    stats_df.to_parquet(stats_path, index=False)

    manifest = {
        "target_transform": target_transform,
        "stage1": "e = log1p(count / row_total * target_sum)",
        "stage2": "z = (e - mean) / std, per (slide, gene), using the stats in zscore_stats.parquet",
        "row_scope": "all_eval_rows_per_slide_train_plus_val_plus_test",
        "row_scope_rationale": (
            "The per-(slide,gene) mean and std are fitted over every eval row on the "
            "slide (probe-train, probe-val and probe-test together)."
        ),
        "n_slide_gene_pairs": int(stats_df.shape[0]),
        "n_slides": int(stats_df["slide_id"].nunique()) if not stats_df.empty else 0,
        "n_genes": int(stats_df["gene"].nunique()) if not stats_df.empty else 0,
        "n_degenerate_constant_pairs": int((stats_df["std"] <= 1.0e-8).sum()) if not stats_df.empty else 0,
        "zscore_stats_parquet": str(stats_path),
        "zscore_stats_sha256": sha256_file(stats_path),
    }
    # Row scope of the stats (STHELAR_TEST_ADAPTED: safe-border rows excluded)
    if row_scope_record:
        manifest.update(dict(row_scope_record))
    save_json(out_dir / "target_norm_manifest.json", manifest)
    return manifest


def _write_thinned_halves(
    path: Path,
    *,
    row_keys: np.ndarray,
    genes: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
) -> None:
    """Freeze one slide's count-thinned, z-scored, winsorized halves."""
    mkdir(path.parent)
    with h5py.File(path, "w") as f:
        f.create_dataset("row_keys", data=np.asarray(row_keys, dtype=object).astype("S"))
        f.create_dataset("genes", data=np.asarray(genes, dtype=object).astype("S"))
        f.create_dataset("t1", data=np.asarray(t1, dtype=np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("t2", data=np.asarray(t2, dtype=np.float32), compression="gzip", compression_opts=4)


def load_thinned_halves(
    noise_ceiling_dir: Path,
    slide_id: str,
    row_keys: Sequence[str],
    gene_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a slide's thinned halves aligned to the given rows and genes (NaN columns for genes not stored)."""
    path = Path(noise_ceiling_dir) / "thinned_halves" / f"{slide_id}.h5"
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen thinned halves not found for slide '{slide_id}': {path}. "
            "Re-run _2_fill_eval_settings.py for this eval_id."
        )
    with h5py.File(path, "r") as f:
        stored_rows = np.asarray([r.decode("utf-8") for r in f["row_keys"][:]], dtype=object)
        stored_genes = np.asarray([g.decode("utf-8") for g in f["genes"][:]], dtype=object)
        t1 = np.asarray(f["t1"][:], dtype=np.float64)
        t2 = np.asarray(f["t2"][:], dtype=np.float64)

    row_pos = {str(r): i for i, r in enumerate(stored_rows.tolist())}
    missing_rows = [str(r) for r in row_keys if str(r) not in row_pos]
    if missing_rows:
        raise KeyError(
            f"Slide '{slide_id}': {len(missing_rows)} requested row(s) are absent from the "
            f"frozen thinned halves (e.g. {missing_rows[:5]}). The row universe changed "
            "without a settings re-fill."
        )
    ridx = np.asarray([row_pos[str(r)] for r in row_keys], dtype=np.int64)

    gene_pos = {str(g): j for j, g in enumerate(stored_genes.tolist())}
    out1 = np.full((ridx.size, len(gene_names)), np.nan, dtype=np.float64)
    out2 = np.full((ridx.size, len(gene_names)), np.nan, dtype=np.float64)
    keep = [(j, gene_pos[str(g)]) for j, g in enumerate(gene_names) if str(g) in gene_pos]
    if keep:
        out_cols = [j for j, _ in keep]
        src_cols = [c for _, c in keep]
        out1[:, out_cols] = t1[np.ix_(ridx, src_cols)]
        out2[:, out_cols] = t2[np.ix_(ridx, src_cols)]
    return out1, out2


# A half whose spread is below this carries no rank or linear information.
_HALF_VAR_EPS = 1.0e-12

# _HALF_VAR_EPS must be >= metrics._VAR_EPS for a symmetric half-split reliability
assert _HALF_VAR_EPS >= _VAR_EPS, (
    f"_HALF_VAR_EPS ({_HALF_VAR_EPS}) must be >= metrics._VAR_EPS ({_VAR_EPS}); otherwise "
    "the count-thinning half-split reliability stops being symmetric in its two halves."
)


def _degenerate_half_pair(x: np.ndarray, y: np.ndarray) -> bool:
    """True when either count-thinned half is constant (correlation undefined, reliability set to NaN)."""
    xf = np.asarray(x, dtype=np.float64)
    yf = np.asarray(y, dtype=np.float64)
    if xf.size < 2 or yf.size < 2:
        return True
    return bool(
        float(np.var(xf)) <= _HALF_VAR_EPS or float(np.var(yf)) <= _HALF_VAR_EPS
    )


def _halfsplit_reliability(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, bool]:
    """(pearson, spearman, degenerate) reliability of one pair of halves."""
    if _degenerate_half_pair(x, y):
        return np.nan, np.nan, True
    return _safe_pearson(x, y), _safe_spearman(x, y), False


def _spearman_brown_full_depth(r_half: float) -> float:
    """
    Spearman-Brown step-up from half-depth to full-depth reliability: r_full = 2 r / (1 + r), capped at 1
    (0 for r <= 0, NaN for non-finite r).
    """
    r = float(r_half)
    if not np.isfinite(r):
        return np.nan
    if r <= 0.0:
        return 0.0
    return float(min(2.0 * r / (1.0 + r), 1.0))


def persist_noise_ceiling(
    cfg: Mapping[str, Any],
    rows_df: pd.DataFrame,
    genes_manifest: Mapping[str, Any],
    out_dir: Path,
    target_norm_dir: Path,
    *,
    row_scope_record: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Count-thinning noise ceiling: halves c1 ~ Binom(c, 0.5), c2 = c - c1, transformed like the target; saves their
    Spearman-Brown full-depth reliabilities per (slide, gene) and per (slide, location), and the halves.
    """
    mkdir(out_dir)
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    target_transform = dict(cfg["protocol"].get("target_transform", {}))
    target_sum = float(target_transform.get("target_sum", 1.0e4))
    protocol_seed = int(cfg["protocol"]["protocol_seed"])
    slide_records = load_slide_records(training_dataset_id)

    zstats = pd.read_parquet(Path(target_norm_dir) / "zscore_stats.parquet")
    stat_lookup: Dict[Tuple[str, str], Tuple[float, float]] = {
        (str(s), str(g)): (float(m), float(sd))
        for s, g, m, sd in zip(
            zstats["slide_id"], zstats["gene"], zstats["mean"], zstats["std"]
        )
    }
    eval_root = resolve_eval_dataset_root(training_dataset_id)

    # Union of every gene that can serve as a target column (as in persist_target_norm_stats)
    seen: set = set()
    gene_union: List[str] = []
    for payload in _iter_gene_payloads_for_reports(genes_manifest):
        for g in payload.get("train_gene_names", payload.get("trainable_genes", [])):
            if str(g) not in seen:
                seen.add(str(g))
                gene_union.append(str(g))
    gene_union = sorted(gene_union)
    if not gene_union:
        raise RuntimeError("persist_noise_ceiling: no trainable genes found in genes_manifest.")

    gene_rows: List[Dict[str, Any]] = []
    loc_rows: List[Dict[str, Any]] = []
    halves_dir = mkdir(out_dir / "thinned_halves")
    n_halves_slides = 0
    for slide_idx, slide_id in enumerate(sorted(rows_df["slide_id"].astype(str).unique().tolist())):
        rec = slide_records[slide_id]
        slide_rows = rows_df.loc[rows_df["slide_id"].astype(str) == slide_id]
        row_keys = slide_rows["row_key"].astype(str).to_numpy(dtype=object)
        adata_path = eval_root / "st" / f"{slide_id}.h5ad"
        mouse_map = eval_root / "mouse_to_human_gene_mappings" / f"{slide_id}_mouse2human_mapping.parquet"
        C, genes = extract_raw_counts_for_rows(
            adata_path=adata_path,
            slide_record=rec,
            ordered_row_keys=row_keys,
            mouse_mapping_path=mouse_map if mouse_map.exists() else None,
        )
        C = C.tocsr()
        counts = np.clip(np.rint(C.data), 0.0, None).astype(np.int64)
        rng = np.random.default_rng([protocol_seed, slide_idx])
        c1 = rng.binomial(counts, 0.5)
        c2 = counts - c1
        C1 = sparse.csr_matrix((c1.astype(np.float64), C.indices, C.indptr), shape=C.shape)
        C2 = sparse.csr_matrix((c2.astype(np.float64), C.indices, C.indptr), shape=C.shape)
        s1 = np.asarray(C1.sum(axis=1)).reshape(-1).astype(np.float64)
        s2 = np.asarray(C2.sum(axis=1)).reshape(-1).astype(np.float64)

        gene_lookup = {g: i for i, g in enumerate(genes.tolist())}
        present = [(g, gene_lookup[g]) for g in gene_union if g in gene_lookup]
        if not present:
            continue
        sub_genes = [g for g, _ in present]
        cols = [c for _, c in present]
        sub1 = C1[:, cols].toarray().astype(np.float64)
        sub2 = C2[:, cols].toarray().astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale1 = np.where(s1 > 0.0, target_sum / s1, 0.0)
            scale2 = np.where(s2 > 0.0, target_sum / s2, 0.0)
        t1 = np.log1p(sub1 * scale1[:, None])
        t2 = np.log1p(sub2 * scale2[:, None])
        # A unit whose half has zero counts cannot be depth-normalized.
        bad = (s1 <= 0.0) | (s2 <= 0.0)
        if np.any(bad):
            t1[bad, :] = np.nan
            t2[bad, :] = np.nan

        # Z-score (as in apply_zscore) and winsorize both halves like the target
        means = np.array(
            [stat_lookup.get((slide_id, g), (np.nan, np.nan))[0] for g in sub_genes]
        )
        stds = np.array(
            [stat_lookup.get((slide_id, g), (np.nan, np.nan))[1] for g in sub_genes]
        )
        denom_std = np.where(np.isfinite(stds), np.maximum(stds, 1e-8), np.nan)
        t1 = winsorize_zscored((t1 - means[None, :]) / denom_std[None, :], target_transform)
        t2 = winsorize_zscored((t2 - means[None, :]) / denom_std[None, :], target_transform)

        # Save the halves (used by the kNN-overlap ceiling of Metric 1)
        _write_thinned_halves(
            halves_dir / f"{slide_id}.h5",
            row_keys=row_keys,
            genes=np.asarray(sub_genes, dtype=object),
            t1=t1,
            t2=t2,
        )
        n_halves_slides += 1

        for k, gene in enumerate(sub_genes):
            x = t1[:, k]
            y = t2[:, k]
            m = np.isfinite(x) & np.isfinite(y)
            n = int(np.sum(m))
            if n >= 2:
                r_pear, r_spear, degenerate = _halfsplit_reliability(x[m], y[m])
            else:
                r_pear, r_spear, degenerate = np.nan, np.nan, True
            gene_rows.append(
                {
                    "slide_id": slide_id,
                    "gene": str(gene),
                    "r_pearson": _spearman_brown_full_depth(r_pear),
                    "r_spearman": _spearman_brown_full_depth(r_spear),
                    "r_pearson_halfsplit": r_pear,
                    "r_spearman_halfsplit": r_spear,
                    "n_locations": n,
                    "degenerate_half": bool(degenerate),
                }
            )
        for i, rk in enumerate(row_keys.tolist()):
            x = t1[i, :]
            y = t2[i, :]
            m = np.isfinite(x) & np.isfinite(y)
            n = int(np.sum(m))
            if n >= 2:
                r_prof_pear, r_prof_spear, degenerate = _halfsplit_reliability(x[m], y[m])
            else:
                r_prof_pear, r_prof_spear, degenerate = np.nan, np.nan, True
            loc_rows.append(
                {
                    "slide_id": slide_id,
                    "row_key": str(rk),
                    "r_profile_pearson": _spearman_brown_full_depth(r_prof_pear),
                    "r_profile_spearman": _spearman_brown_full_depth(r_prof_spear),
                    "r_profile_pearson_halfsplit": r_prof_pear,
                    "r_profile_spearman_halfsplit": r_prof_spear,
                    "n_genes": n,
                    "degenerate_half": bool(degenerate),
                }
            )

    # columns= also selects the saved keys
    reliability_df = pd.DataFrame(
        gene_rows,
        columns=[
            "slide_id", "gene", "r_pearson", "r_spearman",
            "r_pearson_halfsplit", "r_spearman_halfsplit", "n_locations",
            "degenerate_half",
        ],
    )
    reliability_profile_df = pd.DataFrame(
        loc_rows,
        columns=[
            "slide_id", "row_key", "r_profile_pearson", "r_profile_spearman",
            "r_profile_pearson_halfsplit", "r_profile_spearman_halfsplit", "n_genes",
            "degenerate_half",
        ],
    )
    reliability_path = out_dir / "reliability.parquet"
    reliability_profile_path = out_dir / "reliability_profile.parquet"
    reliability_df.to_parquet(reliability_path, index=False)
    reliability_profile_df.to_parquet(reliability_profile_path, index=False)

    def _finite_mean(s: pd.Series) -> float:
        v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        return float(np.mean(v)) if v.size else float("nan")

    manifest = {
        "method": "binomial_count_thinning",
        "thinning": "c1 ~ Binom(c, 0.5); c2 = c - c1; each half depth-normalized by its own full-panel total.",
        "half_transform": (
            "each half is then z-scored with the frozen per-(slide,gene) stats and winsorized "
            "to the target's zscore_clip (same metric space as the target)"
        ),
        "thinned_halves_dir": str(halves_dir),
        "thinned_halves_slides": int(n_halves_slides),
        "thinned_halves_note": (
            "Per-slide t1/t2 (rows x genes, float32) are frozen alongside the reliabilities. "
            "The correlation ceilings use r only; the kNN-overlap ceiling of Metric 1 uses "
            "the halves. Both ceilings share this thinning."
        ),
        "reliability_columns": (
            "r_* are the Spearman-Brown full-depth reliabilities r_full = 2 r / (1 + r); "
            "r_*_halfsplit are the raw half-vs-half count-thinning correlations r."
        ),
        "ceiling_norm": "encoder correlation metrics are reported as metric / sqrt(clip(r_full, eps, 1)).",
        "rng": "numpy default_rng seeded by [protocol_seed, slide_index]",
        "row_scope": "all_eval_rows_per_slide_train_plus_val_plus_test",
        "row_scope_rationale": (
            "The per-(slide,gene) reliability is computed over every eval row on the "
            "slide (probe-train, probe-val and probe-test together); the ceiling is the "
            "same for every encoder."
        ),
        "protocol_seed": protocol_seed,
        "target_sum": target_sum,
        "n_slide_gene_pairs": int(reliability_df.shape[0]),
        "n_locations": int(reliability_profile_df.shape[0]),
        "n_slides": int(reliability_df["slide_id"].nunique()) if not reliability_df.empty else 0,
        "n_genes": int(reliability_df["gene"].nunique()) if not reliability_df.empty else 0,
        "mean_r_spearman": _finite_mean(reliability_df["r_spearman"]) if not reliability_df.empty else float("nan"),
        "mean_r_spearman_halfsplit": (
            _finite_mean(reliability_df["r_spearman_halfsplit"]) if not reliability_df.empty else float("nan")
        ),
        "mean_r_profile_spearman": (
            _finite_mean(reliability_profile_df["r_profile_spearman"])
            if not reliability_profile_df.empty else float("nan")
        ),
        "mean_r_profile_spearman_halfsplit": (
            _finite_mean(reliability_profile_df["r_profile_spearman_halfsplit"])
            if not reliability_profile_df.empty else float("nan")
        ),
        "reliability_parquet": str(reliability_path),
        "reliability_sha256": sha256_file(reliability_path),
        "reliability_profile_parquet": str(reliability_profile_path),
        "reliability_profile_sha256": sha256_file(reliability_profile_path),
    }
    # Row scope of the ceiling (same as the z-score stats)
    if row_scope_record:
        manifest.update(dict(row_scope_record))
    save_json(out_dir / "noise_ceiling_manifest.json", manifest)
    return manifest


def _tercile_edges(scores: pd.Series) -> Tuple[float, float]:
    """Lower/upper tercile cut points of the finite values of `scores`."""
    v = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    q1, q2 = np.quantile(v, [1.0 / 3.0, 2.0 / 3.0])
    return (float(q1), float(q2))


def _assign_tercile(scores: pd.Series, edges: Tuple[float, float]) -> pd.Series:
    """Map scores to tercile labels 0/1/2 (low/mid/high); NaN scores get -1."""
    v = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=np.float64)
    q1, q2 = edges
    out = np.full(v.shape, -1, dtype=np.int64)
    finite = np.isfinite(v)
    if np.isfinite(q1) and np.isfinite(q2):
        out[finite & (v <= q1)] = 0
        out[finite & (v > q1) & (v <= q2)] = 1
        out[finite & (v > q2)] = 2
    return pd.Series(out, index=scores.index)


def persist_gene_strata(
    cfg: Mapping[str, Any],
    genes_manifest: Mapping[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Save per-gene strata: cell-type-specificity (Tau) and spatial-structure (Moran's I) scores from the gene-property
    table and their terciles over the genes of this eval_id (-1 for genes absent from the table).
    The axes are encoder-independent gene properties, not per-teacher attributions.
    """
    mkdir(out_dir)
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    gene_property_id = str(cfg["protocol"]["gene_property_id"])
    gene_property_dir = resolve_gene_property_dir(training_dataset_id, gene_property_id)
    table_path = gene_property_dir / "gene_property_table.parquet"
    if not table_path.exists():
        raise FileNotFoundError(
            f"Gene-property table not found: {table_path}. "
            f"Run _12_gene_properties.py with --gene_property_id {gene_property_id} first."
        )

    # The gene-property table and the eval must use the same scGPT model
    table_config_path = gene_property_dir / "config.json"
    table_scgpt_model_name = None
    if table_config_path.exists():
        table_config = load_json(table_config_path)
        table_scgpt_model_name = str(table_config.get("scGPT_model_name", "")) or None
        eval_scgpt_model_name = str(cfg["protocol"].get("scgpt_model_name", "") or "")
        if (
            table_scgpt_model_name is not None
            and eval_scgpt_model_name
            and table_scgpt_model_name != eval_scgpt_model_name
        ):
            raise RuntimeError(
                "scGPT model name mismatch between the frozen gene-property table and "
                "the eval config:\n"
                f"  gene_property_id          = {gene_property_id}\n"
                f"  table.scGPT_model_name    = '{table_scgpt_model_name}' (from {table_config_path})\n"
                f"  eval.scgpt_model_name     = '{eval_scgpt_model_name}'\n"
                "Either rebuild the gene-property table with the eval's scGPT model, "
                "or pick a gene_property_id whose table was built with the same vocab."
            )

    table = pd.read_parquet(table_path)
    lut = table.set_index(table["gene"].astype(str))

    seen: set = set()
    gene_union: List[str] = []
    for payload in _iter_gene_payloads_for_reports(genes_manifest):
        for g in payload.get("train_gene_names", payload.get("trainable_genes", [])):
            if str(g) not in seen:
                seen.add(str(g))
                gene_union.append(str(g))
    gene_union = sorted(gene_union)
    if not gene_union:
        raise RuntimeError("persist_gene_strata: no trainable genes found in genes_manifest.")

    rows: List[Dict[str, Any]] = []
    for g in gene_union:
        if g in lut.index:
            r = lut.loc[g]
            ct = r["ct_specificity_score"]
            sp = r["spatial_structure_score"]
            rows.append(
                {
                    "gene": g,
                    "ct_specificity_score": float(ct) if pd.notna(ct) else np.nan,
                    "spatial_structure_score": float(sp) if pd.notna(sp) else np.nan,
                    "in_scgpt_vocab": bool(r.get("in_scgpt_vocab", False)),
                    "in_teacher_vocab": True,
                }
            )
        else:
            rows.append(
                {
                    "gene": g,
                    "ct_specificity_score": np.nan,
                    "spatial_structure_score": np.nan,
                    "in_scgpt_vocab": False,
                    "in_teacher_vocab": False,
                }
            )
    strata = pd.DataFrame(
        rows, columns=["gene", "ct_specificity_score", "spatial_structure_score", "in_scgpt_vocab", "in_teacher_vocab"]
    )

    strata_path = out_dir / "gene_strata.parquet"

    n_with_ct = int(np.isfinite(strata["ct_specificity_score"]).sum())
    n_with_sp = int(np.isfinite(strata["spatial_structure_score"]).sum())
    if min(n_with_ct, n_with_sp) < MIN_GENES_FOR_STRATA:
        pd.DataFrame(
            columns=[
                "gene", "ct_specificity_score", "spatial_structure_score",
                "in_scgpt_vocab", "in_teacher_vocab",
                "ct_specificity_tercile", "spatial_structure_tercile",
            ]
        ).to_parquet(strata_path, index=False)
        manifest: Dict[str, Any] = {
            "skipped": True,
            "reason": (
                f"Fewer than MIN_GENES_FOR_STRATA={MIN_GENES_FOR_STRATA} eval-union "
                f"genes have a finite score on at least one teacher axis: "
                f"n_genes_with_ct_specificity={n_with_ct}, "
                f"n_genes_with_spatial_structure={n_with_sp}. "
                "Tercile edges are not computed."
            ),
            "gene_property_id": gene_property_id,
            "gene_property_table": str(table_path),
            "gene_property_table_sha256": sha256_file(table_path),
            "gene_property_scGPT_model_name": table_scgpt_model_name,
            "n_genes_in_eval_union": int(strata.shape[0]),
            "n_genes_in_teacher_vocab": int(strata["in_teacher_vocab"].sum()),
            "n_genes_with_ct_specificity": n_with_ct,
            "n_genes_with_spatial_structure": n_with_sp,
            "min_genes_for_strata": int(MIN_GENES_FOR_STRATA),
            "gene_strata_parquet": str(strata_path),
            "gene_strata_sha256": sha256_file(strata_path),
        }
        save_json(out_dir / "gene_strata_manifest.json", manifest)
        return manifest

    ct_edges = _tercile_edges(strata["ct_specificity_score"])
    sp_edges = _tercile_edges(strata["spatial_structure_score"])
    strata["ct_specificity_tercile"] = _assign_tercile(strata["ct_specificity_score"], ct_edges)
    strata["spatial_structure_tercile"] = _assign_tercile(strata["spatial_structure_score"], sp_edges)
    strata = strata.sort_values("gene").reset_index(drop=True)

    strata.to_parquet(strata_path, index=False)

    manifest = {
        "gene_property_id": gene_property_id,
        "gene_property_table": str(table_path),
        "gene_property_table_sha256": sha256_file(table_path),
        "gene_property_scGPT_model_name": table_scgpt_model_name,
        "tercile_scope": (
            "eval-local: tercile edges are computed over this eval_id's gene union "
            "(see ct_specificity_tercile_edges / spatial_structure_tercile_edges); "
            "they can differ between eval_ids."
        ),
        "strata_definition": {
            "ct_specificity": "Tau cell-type-specificity terciles (0/1/2 = low/mid/high); high = most cell-type-specific genes. Encoder-independent gene property, not a per-teacher attribution.",
            "spatial_structure": "Moran's I spatial-structure terciles (0/1/2 = low/mid/high); high = most spatially structured genes. Encoder-independent gene property, not a per-teacher attribution.",
            "rest": "ct_specificity_tercile == 0 AND spatial_structure_tercile == 0 (negative control).",
            "out_of_teacher_vocab": "gene absent from the STHELAR gene-property table; scores NaN, tercile -1.",
        },
        "ct_specificity_tercile_edges": list(ct_edges),
        "spatial_structure_tercile_edges": list(sp_edges),
        "n_genes": int(strata.shape[0]),
        "n_genes_in_teacher_vocab": int(strata["in_teacher_vocab"].sum()),
        "n_genes_with_ct_specificity": int(np.isfinite(strata["ct_specificity_score"]).sum()),
        "n_genes_with_spatial_structure": int(np.isfinite(strata["spatial_structure_score"]).sum()),
        "gene_strata_parquet": str(strata_path),
        "gene_strata_sha256": sha256_file(strata_path),
    }
    save_json(out_dir / "gene_strata_manifest.json", manifest)
    return manifest


def persist_signature_strata(
    cfg: Mapping[str, Any],
    genes_manifest: Mapping[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Save per-gene external-signature strata: membership of the eval genes in the signatures of eval_signatures_id
    (signatures with >= MIN_GENES_FOR_STRATA eval genes), and their overlap with the gene-property terciles.
    Skipped when eval_signatures_id is not set or no signature is kept.
    """
    mkdir(out_dir)
    strata_path = out_dir / "signature_strata.parquet"
    manifest_path = out_dir / "signature_strata_manifest.json"
    eval_columns = [
        "gene", "signature_id", "signature_category", "in_teacher_vocab",
    ]

    eval_signatures_id = cfg["protocol"].get("eval_signatures_id")
    if not eval_signatures_id:
        pd.DataFrame(columns=eval_columns).to_parquet(strata_path, index=False)
        manifest: Dict[str, Any] = {
            "skipped": True,
            "reason": "cfg['protocol']['eval_signatures_id'] is not set.",
            "signature_strata_parquet": str(strata_path),
        }
        save_json(manifest_path, manifest)
        return manifest

    eval_signatures_dir = resolve_eval_signatures_dir(str(eval_signatures_id))
    table_path = eval_signatures_dir / "eval_signatures_table.parquet"
    if not table_path.exists():
        raise FileNotFoundError(
            f"Eval-signatures table not found: {table_path}. "
            f"Run _13_eval_signatures.py with --eval_signatures_id {eval_signatures_id} first."
        )

    table_config_path = eval_signatures_dir / "config.json"
    table_config: Dict[str, Any] = {}
    if table_config_path.exists():
        table_config = load_json(table_config_path)

    table = pd.read_parquet(table_path)

    seen: set = set()
    gene_union: List[str] = []
    for payload in _iter_gene_payloads_for_reports(genes_manifest):
        for g in payload.get("train_gene_names", payload.get("trainable_genes", [])):
            if str(g) not in seen:
                seen.add(str(g))
                gene_union.append(str(g))
    gene_union = sorted(gene_union)
    if not gene_union:
        raise RuntimeError(
            "persist_signature_strata: no trainable genes found in genes_manifest."
        )
    gene_union_set = set(gene_union)

    # Per-signature category metadata (forwarded from the source artifact).
    signatures_meta: Dict[str, Dict[str, Any]] = {}
    for sig_id, meta in (table_config.get("signatures", {}) or {}).items():
        signatures_meta[str(sig_id)] = dict(meta) if isinstance(meta, Mapping) else {}

    # Long-format eval-local membership.
    sub = table[table["gene"].isin(gene_union_set)].copy()
    sub["signature_id"] = sub["signature_id"].astype(str)
    sub["gene"] = sub["gene"].astype(str)

    # Drop signatures with fewer than MIN_GENES_FOR_STRATA genes in the eval gene union
    counts_per_sig = sub.groupby("signature_id").size()
    kept_signatures = set(counts_per_sig.index[counts_per_sig >= MIN_GENES_FOR_STRATA].tolist())
    skipped_signatures: List[Dict[str, Any]] = []
    for sig_id, n_in_union in counts_per_sig.items():
        if sig_id not in kept_signatures:
            meta = signatures_meta.get(str(sig_id), {})
            skipped_signatures.append({
                "signature_id": str(sig_id),
                "signature_category": str(meta.get("category", "")),
                "n_genes_total": int(meta.get("n_genes_total", 0)),
                "n_genes_in_eval_gene_union": int(n_in_union),
                "reason": (
                    f"n_genes_in_eval_gene_union={int(n_in_union)} < "
                    f"MIN_GENES_FOR_STRATA={MIN_GENES_FOR_STRATA}"
                ),
            })
    # Signatures with no gene in the eval gene union are also recorded as skipped
    all_source_signatures = {str(s) for s in table["signature_id"].astype(str).unique().tolist()}
    for sig_id in sorted(all_source_signatures - set(counts_per_sig.index.astype(str).tolist())):
        meta = signatures_meta.get(sig_id, {})
        skipped_signatures.append({
            "signature_id": sig_id,
            "signature_category": str(meta.get("category", "")),
            "n_genes_total": int(meta.get("n_genes_total", 0)),
            "n_genes_in_eval_gene_union": 0,
            "reason": (
                f"n_genes_in_eval_gene_union=0 < "
                f"MIN_GENES_FOR_STRATA={MIN_GENES_FOR_STRATA}"
            ),
        })

    sub = sub[sub["signature_id"].isin(kept_signatures)].copy()

    if sub.empty:
        pd.DataFrame(columns=eval_columns).to_parquet(strata_path, index=False)
        manifest = {
            "skipped": True,
            "reason": (
                f"No signature has >= MIN_GENES_FOR_STRATA={MIN_GENES_FOR_STRATA} "
                "genes in the eval gene union."
            ),
            "eval_signatures_id": str(eval_signatures_id),
            "eval_signatures_table": str(table_path),
            "eval_signatures_table_sha256": sha256_file(table_path),
            "min_genes_for_strata": int(MIN_GENES_FOR_STRATA),
            "skipped_signatures": skipped_signatures,
            "signature_strata_parquet": str(strata_path),
        }
        save_json(manifest_path, manifest)
        return manifest

    sub["signature_category"] = sub["signature_id"].map(
        lambda s: str(signatures_meta.get(s, {}).get("category", ""))
    )
    sub["in_teacher_vocab"] = True  # by construction: gene is in eval gene union
    sub = sub[eval_columns].sort_values(["signature_id", "gene"]).reset_index(drop=True)
    sub.to_parquet(strata_path, index=False)

    # Independence diagnostic vs the existing teacher-aligned gene_strata.
    independence: Dict[str, Any] = {}
    gene_strata_path = out_dir / "gene_strata.parquet"
    if gene_strata_path.exists():
        try:
            from scipy.stats import mannwhitneyu
        except Exception:
            mannwhitneyu = None  # type: ignore[assignment]
        gs = pd.read_parquet(gene_strata_path)
        gs_idx = gs.set_index(gs["gene"].astype(str))
        per_sig: Dict[str, Any] = {}
        for sig_id in sorted(sub["signature_id"].unique().tolist()):
            in_sig_genes = sub.loc[sub["signature_id"] == sig_id, "gene"].tolist()
            in_sig_idx = gs_idx.index.isin(in_sig_genes)
            in_sig_rows = gs_idx[in_sig_idx]
            out_sig_rows = gs_idx[~in_sig_idx]

            def _tercile_counts(rows: pd.DataFrame, col: str) -> Dict[str, int]:
                if col not in rows.columns:
                    return {"low": 0, "mid": 0, "high": 0, "no_score": 0}
                v = pd.to_numeric(rows[col], errors="coerce").to_numpy(dtype=np.int64)
                return {
                    "low": int((v == 0).sum()),
                    "mid": int((v == 1).sum()),
                    "high": int((v == 2).sum()),
                    "no_score": int((v == -1).sum()),
                }

            def _mw_u(in_vals: np.ndarray, out_vals: np.ndarray) -> Dict[str, float]:
                in_finite = in_vals[np.isfinite(in_vals)]
                out_finite = out_vals[np.isfinite(out_vals)]
                if (
                    mannwhitneyu is None
                    or in_finite.size < 2
                    or out_finite.size < 2
                ):
                    return {"U": float("nan"), "p_value": float("nan")}
                try:
                    res = mannwhitneyu(
                        in_finite, out_finite, alternative="two-sided"
                    )
                    return {"U": float(res.statistic), "p_value": float(res.pvalue)}
                except Exception:
                    return {"U": float("nan"), "p_value": float("nan")}

            ct_in = pd.to_numeric(
                in_sig_rows.get("ct_specificity_score", pd.Series(dtype=float)),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            ct_out = pd.to_numeric(
                out_sig_rows.get("ct_specificity_score", pd.Series(dtype=float)),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            sp_in = pd.to_numeric(
                in_sig_rows.get("spatial_structure_score", pd.Series(dtype=float)),
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            sp_out = pd.to_numeric(
                out_sig_rows.get("spatial_structure_score", pd.Series(dtype=float)),
                errors="coerce",
            ).to_numpy(dtype=np.float64)

            per_sig[sig_id] = {
                "n_genes_in_eval_gene_union": int(in_sig_rows.shape[0]),
                "ct_specificity_tercile_counts": _tercile_counts(
                    in_sig_rows, "ct_specificity_tercile"
                ),
                "spatial_structure_tercile_counts": _tercile_counts(
                    in_sig_rows, "spatial_structure_tercile"
                ),
                "ct_specificity_mannwhitney_vs_rest": _mw_u(ct_in, ct_out),
                "spatial_structure_mannwhitney_vs_rest": _mw_u(sp_in, sp_out),
            }
        independence = {
            "per_signature": per_sig,
            "interpretation": (
                "For each signature, counts of in-signature genes falling into "
                "the low/mid/high tercile of the eval-local Tau and Moran's I "
                "score distributions (terciles defined in gene_strata_manifest.json), "
                "and a two-sided Mann-Whitney U test of the in-signature scores "
                "against the rest of the eval-gene union."
            ),
        }

    # Per-signature summary for the manifest.
    per_signature_summary: List[Dict[str, Any]] = []
    for sig_id in sorted(sub["signature_id"].unique().tolist()):
        meta = signatures_meta.get(sig_id, {})
        per_signature_summary.append({
            "signature_id": sig_id,
            "signature_category": str(meta.get("category", "")),
            "n_genes_total": int(meta.get("n_genes_total", 0)),
            "n_genes_in_eval_gene_union": int(
                (sub["signature_id"] == sig_id).sum()
            ),
            "citation": str(meta.get("citation", "")),
        })

    manifest = {
        "eval_signatures_id": str(eval_signatures_id),
        "eval_signatures_table": str(table_path),
        "eval_signatures_table_sha256": sha256_file(table_path),
        "membership_scope": (
            "eval-local: signature membership is restricted to this eval_id's "
            "gene union. |signature ∩ eval_gene_union| can differ across "
            "eval_ids (and across folds for HEST_CV_* / STHELAR_TEST_ADAPTED)."
        ),
        "strata_definition": {
            "per_signature": (
                "one stratum per signature_id; eval-local intersection of the "
                "signature gene list with the eval gene union."
            ),
            "any_signature": (
                "derived in metrics: union of all signature memberships "
                "(genes appearing in at least one signature)."
            ),
            "no_signature": (
                "derived in metrics: eval genes not in any signature - "
                "background control."
            ),
        },
        "min_genes_for_strata": int(MIN_GENES_FOR_STRATA),
        "n_signatures_with_eval_genes": int(sub["signature_id"].nunique()),
        "n_eval_genes_in_any_signature": int(sub["gene"].nunique()),
        "per_signature": per_signature_summary,
        "skipped_signatures": skipped_signatures,
        "independence_vs_teacher_axes": independence,
        "signature_strata_parquet": str(strata_path),
        "signature_strata_sha256": sha256_file(strata_path),
    }
    save_json(manifest_path, manifest)
    return manifest


# -----------------------------------------------------------------------------
# Signature target mode (target_basis="signatures")
# -----------------------------------------------------------------------------


def _config_signature_train_pool(splits_manifest: Mapping[str, Any]) -> List[str]:
    """Train slide pool of the config, over which the per-signature coverage gate is computed (same for all folds)."""
    t = str(splits_manifest.get("type", ""))
    if t == "slide_cv":
        pool: set = set()
        for fold in splits_manifest.get("folds", []):
            pool.update(map(str, fold.get("train_slides", [])))
        return sorted(pool)
    if t == "one_shot":
        return sorted(map(str, splits_manifest.get("train_slides", [])))
    if t == "within_slide_area":
        # STHELAR_TEST_ADAPTED: gate computed on the whole adapted slides
        return sorted(map(str, splits_manifest.get("slides", [])))
    raise NotImplementedError(
        f"target_basis='signatures' is not wired for splits type '{t}'. Supported: "
        "'slide_cv' (STHELAR_TRAIN_CV, HEST_CV_*), 'one_shot' (STHELAR_TEST_ONE_SHOT), "
        "'within_slide_area' (STHELAR_TEST_ADAPTED)."
    )


def _covered_member_genes(
    cfg: Mapping[str, Any],
    support_df: pd.DataFrame,
    *,
    single_unit: bool = False,
) -> pd.DataFrame:
    """
    Member genes passing the gene-level support thresholds (without min_trainable_genes, and without
    min_train_slides_per_gene if single_unit). Returns the per-gene aggregate with an is_covered column.
    """
    cols = ["gene", "n_train_slides", "n_rows", "n_positive", "is_covered"]
    if support_df.empty:
        return pd.DataFrame(columns=cols)
    gcfg = cfg["protocol"]["gene_selection"]
    eligible = _eligible_support_from_support(cfg, support_df)
    if eligible.empty:
        return pd.DataFrame(columns=cols)
    agg = (
        eligible.groupby("gene", sort=True)
        .agg(
            n_train_slides=("slide_id", "nunique"),
            n_rows=("n_rows", "sum"),
            n_positive=("n_positive", "sum"),
        )
        .reset_index()
    )
    agg["gene"] = agg["gene"].astype(str)
    min_slides = 1 if single_unit else int(gcfg["min_train_slides_per_gene"])
    agg["is_covered"] = (
        (agg["n_train_slides"] >= min_slides)
        & (agg["n_rows"] >= int(gcfg["min_train_rows_per_gene"]))
        & (agg["n_positive"] >= int(gcfg["min_train_positive_rows_per_gene"]))
    ).astype(bool)
    return agg[cols]


def _signature_fold_train_units(
    splits_manifest: Mapping[str, Any],
) -> List[Tuple[str, List[str]]]:
    """(fold_label, train_slides) units on which a kept signature must be fittable (per slide for within_slide_area)."""
    t = str(splits_manifest.get("type", ""))
    if t == "slide_cv":
        return [
            (
                str(fold.get("fold_name", f"fold_{int(fold['fold_id']):02d}")),
                [str(s) for s in fold.get("train_slides", [])],
            )
            for fold in splits_manifest.get("folds", [])
        ]
    if t == "one_shot":
        return [("one_shot", [str(s) for s in splits_manifest.get("train_slides", [])])]
    if t == "within_slide_area":
        return [(str(s), [str(s)]) for s in splits_manifest.get("slides", [])]
    raise NotImplementedError(
        f"target_basis='signatures' is not wired for splits type '{t}'."
    )


def _signature_slide_eligibility_table(
    sig: pd.DataFrame,
    eligibility_df: pd.DataFrame,
    all_slides: Sequence[str],
    *,
    min_covered_fraction: float,
    min_covered_genes: int,
) -> pd.DataFrame:
    """
    Per-(slide, signature) eligibility: enough members (count and fraction of the full membership) eligible on the
    slide. Ineligible signatures are set to NaN on that slide at run time.
    """
    slides = [str(s) for s in all_slides]
    members = (
        sig.groupby(["signature_set", "signature_id"], sort=True)["gene"]
        .apply(lambda s: sorted(set(map(str, s.tolist()))))
        .to_dict()
    )
    if eligibility_df.empty:
        eligible_by_slide: Dict[str, set] = {s: set() for s in slides}
    else:
        ok = eligibility_df.loc[eligibility_df["eligible_slide_gene"].astype(bool)]
        eligible_by_slide = {
            s: set(map(str, grp["gene"].astype(str).tolist()))
            for s, grp in ok.groupby(ok["slide_id"].astype(str), sort=False)
        }
        for s in slides:
            eligible_by_slide.setdefault(s, set())

    rows: List[Dict[str, Any]] = []
    for (set_name, sig_id), member_list in members.items():
        n_total = len(member_list)
        for sid in slides:
            avail = eligible_by_slide.get(sid, set())
            n_elig = sum(1 for g in member_list if g in avail)
            frac = (float(n_elig) / float(n_total)) if n_total else 0.0
            rows.append(
                {
                    "slide_id": str(sid),
                    "signature_set": str(set_name),
                    "signature_id": str(sig_id),
                    "n_members_total": int(n_total),
                    "n_members_eligible_on_slide": int(n_elig),
                    "coverage_fraction_on_slide": float(frac),
                    "eligible": bool(
                        n_elig >= int(min_covered_genes)
                        and frac >= float(min_covered_fraction)
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "slide_id", "signature_set", "signature_id", "n_members_total",
            "n_members_eligible_on_slide", "coverage_fraction_on_slide", "eligible",
        ],
    )


def _signature_weak_train_slides(
    cfg: Mapping[str, Any],
    support_df: pd.DataFrame,
    slide_elig: pd.DataFrame,
    train_slides: Sequence[str],
) -> Dict[str, Any]:
    """
    Signature-axis weak-train-slide filter: drops train slides with fewer than min_train_rows_per_slide rows or
    fewer than signature_target.min_train_eligible_signatures_per_slide eligible signatures.
    """
    gcfg = cfg["protocol"]["gene_selection"]
    st_cfg = dict(cfg["protocol"].get("signature_target", {}))
    min_rows = int(gcfg.get("min_train_rows_per_slide", 1))
    min_sigs = int(st_cfg.get("min_train_eligible_signatures_per_slide", 1))

    slide_ids = list(dict.fromkeys(map(str, train_slides)))
    rows_on_slide: Dict[str, int] = {}
    if not support_df.empty:
        grp = support_df.copy()
        grp["slide_id"] = grp["slide_id"].astype(str)
        rows_on_slide = grp.groupby("slide_id")["n_rows"].max().astype(int).to_dict()

    sigs_on_slide: Dict[str, int] = {}
    if not slide_elig.empty:
        ok = slide_elig.loc[slide_elig["eligible"].astype(bool)]
        sigs_on_slide = (
            ok.groupby(ok["slide_id"].astype(str)).size().astype(int).to_dict()
        )

    summary: List[Dict[str, Any]] = []
    for sid in slide_ids:
        n_rows = int(rows_on_slide.get(sid, 0))
        n_sigs = int(sigs_on_slide.get(sid, 0))
        reasons: List[str] = []
        if n_rows < min_rows:
            reasons.append(f"n_rows_on_slide={n_rows} < min_train_rows_per_slide={min_rows}")
        if n_sigs < min_sigs:
            reasons.append(
                f"n_eligible_signatures_on_slide={n_sigs} < "
                f"min_train_eligible_signatures_per_slide={min_sigs}"
            )
        summary.append(
            {
                "slide_id": sid,
                "n_train_rows_on_slide": n_rows,
                "n_eligible_signatures_on_slide": n_sigs,
                "keep_train_slide": not reasons,
                "drop_weak_train_slide": bool(reasons),
                "train_slide_drop_reason": "; ".join(reasons),
            }
        )
    summary_df = pd.DataFrame(summary)
    kept = [r["slide_id"] for r in summary if r["keep_train_slide"]]
    dropped = [r["slide_id"] for r in summary if r["drop_weak_train_slide"]]
    if not kept:
        raise RuntimeError(
            "Signature-axis weak-train-slide filter dropped every train slide "
            f"({dropped}). Inspect signature_slide_eligibility.parquet, or relax "
            "gene_selection.min_train_rows_per_slide / "
            "signature_target.min_train_eligible_signatures_per_slide."
        )
    return {
        "kept_train_slides": kept,
        "dropped_train_slides": dropped,
        "slide_summary": summary_df,
        "floors": {
            "min_train_rows_per_slide": min_rows,
            "min_train_eligible_signatures_per_slide": min_sigs,
        },
    }


def stripe_scoped_rows(
    cfg: Mapping[str, Any],
    rows_df: pd.DataFrame,
    area_manifest: Optional[Mapping[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Restrict rows_df to the stripe rows of STHELAR_TEST_ADAPTED (safe-border rows removed); pass-through for the
    other sub-configs. Returns (rows, scope_record).
    """
    sub_cfg = str(cfg["protocol"]["sub_eval_configuration"])
    if sub_cfg != "STHELAR_TEST_ADAPTED" or not area_manifest or not area_manifest.get("enabled"):
        return rows_df, {
            "row_scope": "all_eval_rows_per_slide",
            "row_scope_rationale": (
                "Every eval row on the slide (probe-train + probe-test), which is exactly "
                "the row set the protocol uses outside STHELAR_TEST_ADAPTED."
            ),
            "n_rows": int(rows_df.shape[0]),
            "n_rows_excluded": 0,
        }

    level = str(cfg["protocol"]["level"])
    keep = pd.Series(False, index=rows_df.index)
    for slide_entry in area_manifest.get("slides", []):
        sid = str(slide_entry["slide_id"])
        on_slide = rows_df["slide_id"].astype(str) == sid
        if level == "cell":
            cell_df = pd.read_parquet(slide_entry["cell_manifest"])
            in_stripe = set(
                cell_df.loc[cell_df["stripe_id"] >= 0, "cell_id_int"].tolist()
            )
            keep |= on_slide & rows_df["cell_id_int"].isin(in_stripe)
        else:
            patch_df = pd.read_parquet(slide_entry["patch_manifest"])
            in_stripe = set(
                patch_df.loc[patch_df["stripe_id"] >= 0, "patch_id"].astype(str).tolist()
            )
            keep |= on_slide & rows_df["patch_id"].astype(str).isin(in_stripe)

    scoped = rows_df.loc[keep].copy()
    if scoped.empty:
        raise RuntimeError(
            "STHELAR_TEST_ADAPTED: restricting to stripe rows (stripe_id >= 0) left no "
            "rows. Inspect settings/area_splits/area_split_manifest.json."
        )
    return scoped, {
        "row_scope": "adapted_stripe_rows_only",
        "row_scope_rationale": (
            "STHELAR_TEST_ADAPTED only: rows inside a stripe (stripe_id >= 0); safe-border "
            "patches (in no fold) are excluded. The noise ceiling is computed over all the "
            "stripes of the slide, not per stripe."
        ),
        "n_rows": int(scoped.shape[0]),
        "n_rows_excluded": int(rows_df.shape[0] - scoped.shape[0]),
    }


def _assert_signature_alpha_groups_survived_drop(
    kept: Sequence[str], dropped: Sequence[str], where: str
) -> None:
    """Signature-axis version of _assert_alpha_groups_survived_drop (at least 2 train slides per fold)."""
    if len(list(kept)) >= 2:
        return
    raise RuntimeError(
        f"{where}: signature-axis weak-slide filtering left {len(list(kept))} train "
        f"slide(s) ({list(kept)}); alpha selection needs >= 2 to leave one out. "
        f"Dropped: {list(dropped)}. Relax gene_selection.min_train_rows_per_slide / "
        "signature_target.min_train_eligible_signatures_per_slide, or inspect "
        "signature_slide_eligibility.parquet."
    )


def _apply_signature_weak_train_slide_filter(
    *,
    cfg: Mapping[str, Any],
    rows_df: pd.DataFrame,
    support_all: pd.DataFrame,
    elig_pooled: pd.DataFrame,
    splits_manifest: Dict[str, Any],
    splits_dir: Optional[Path],
) -> Dict[str, Any]:
    """Run the signature-axis weak-train-slide filter and rewrite the split files (splits_manifest is modified in place)."""
    t = str(splits_manifest.get("type", ""))
    balance = _slide_count_balance_table(rows_df)
    record: Dict[str, Any] = {"splits_type": t, "per_fold": []}

    if t == "slide_cv":
        for fold in splits_manifest.get("folds", []):
            train = [str(s) for s in fold.get("train_slides", [])]
            res = _signature_weak_train_slides(cfg, support_all, elig_pooled, train)
            label = str(fold.get("fold_name", f"fold_{int(fold['fold_id']):02d}"))
            _assert_signature_alpha_groups_survived_drop(
                res["kept_train_slides"], res["dropped_train_slides"], label
            )
            if res["dropped_train_slides"]:
                fold["train_slides"] = res["kept_train_slides"]
                summary = _summarize_slide_subset(balance, fold["train_slides"])
                fold["train_row_count"] = int(summary["n_rows"])
                fold["train_tissue_counts"] = dict(summary["tissue_counts"])
                if splits_dir is not None:
                    save_json(splits_dir / f"fold_{int(fold['fold_id']):02d}.json", fold)
            record["per_fold"].append({
                "fold": label,
                "kept_train_slides": res["kept_train_slides"],
                "dropped_train_slides": res["dropped_train_slides"],
                "floors": res["floors"],
            })
        if splits_dir is not None and any(
            f["dropped_train_slides"] for f in record["per_fold"]
        ):
            save_json(splits_dir / "splits_manifest.json", splits_manifest)
        return record

    if t == "one_shot":
        train = [str(s) for s in splits_manifest.get("train_slides", [])]
        res = _signature_weak_train_slides(cfg, support_all, elig_pooled, train)
        _assert_signature_alpha_groups_survived_drop(
            res["kept_train_slides"], res["dropped_train_slides"], "one_shot"
        )
        if res["dropped_train_slides"]:
            splits_manifest["train_slides"] = res["kept_train_slides"]
            if splits_dir is not None:
                save_json(splits_dir / "splits_manifest.json", splits_manifest)
        record["per_fold"].append({
            "fold": "one_shot",
            "kept_train_slides": res["kept_train_slides"],
            "dropped_train_slides": res["dropped_train_slides"],
            "floors": res["floors"],
        })
        return record

    if t == "within_slide_area":
        # A weak adapted slide cannot be dropped (it is the fold)
        slides = [str(s) for s in splits_manifest.get("slides", [])]
        res = _signature_weak_train_slides(cfg, support_all, elig_pooled, slides)
        if res["dropped_train_slides"]:
            raise RuntimeError(
                f"STHELAR_TEST_ADAPTED: adapted slide(s) {res['dropped_train_slides']} fail the "
                "signature-axis weak-slide floors (an adapted slide cannot be dropped from a fold). "
                "Inspect signature_slide_eligibility.parquet. Floors: " + str(res["floors"])
            )
        record["per_fold"].append({
            "fold": "within_slide_area",
            "kept_train_slides": res["kept_train_slides"],
            "dropped_train_slides": [],
            "floors": res["floors"],
        })
        return record

    raise NotImplementedError(
        f"Signature weak-train-slide filter is not wired for splits type '{t}'."
    )


def _signature_set_source_shas(cfg: Mapping[str, Any]) -> Dict[str, str]:
    """SHA256 of each resolved signature-set source table, for provenance."""
    protocol = cfg["protocol"]
    out: Dict[str, str] = {}
    for set_name in list(protocol.get("signature_sets", []) or []):
        if set_name == "external":
            sid = protocol.get("eval_signatures_id")
            if sid:
                p = resolve_eval_signatures_dir(str(sid)) / "eval_signatures_table.parquet"
                if p.exists():
                    out["external"] = sha256_file(p)
        elif set_name == "internal":
            sid = protocol.get("eval_signatures_INT_id")
            if sid:
                p = resolve_eval_signatures_INT_dir(str(sid)) / "eval_signatures_INT_table.parquet"
                if p.exists():
                    out["internal"] = sha256_file(p)
    return out


def persist_signature_targets(
    cfg: Mapping[str, Any],
    rows_df: pd.DataFrame,
    splits_manifest: Mapping[str, Any],
    genes_dir: Path,
    *,
    splits_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Save the signature targets (target_basis='signatures') after the pooled coverage gate, the weak-train-slide
    filter (rewrites the split files) and a fold-feasibility check. Writes in genes_dir: signature_targets.parquet,
    signature_slide_eligibility.parquet, signature_targets_manifest.json, genes_manifest.json (covered member genes).
    """
    sub_cfg = str(cfg["protocol"]["sub_eval_configuration"])
    st_cfg = dict(cfg["protocol"].get("signature_target", {}))
    gate = dict(st_cfg.get("coverage_gate", {}))
    min_frac = float(gate.get("min_covered_fraction", 0.5))
    min_genes = int(gate.get("min_covered_genes", 3))

    sig = load_signature_sets(cfg)
    if sig.empty:
        raise RuntimeError(
            "target_basis='signatures' but signature_sets resolved to no "
            "signatures. Set protocol.signature_sets and the corresponding "
            "eval_signatures_id / eval_signatures_INT_id."
        )
    sig = sig.copy()
    # Signature sets and ids must not contain the axis separator
    axis_offenders = sig.loc[
        sig["signature_set"].astype(str).str.contains(SIGNATURE_AXIS_SEP, regex=False)
        | sig["signature_id"].astype(str).str.contains(SIGNATURE_AXIS_SEP, regex=False),
        ["signature_set", "signature_id"],
    ].drop_duplicates()
    if not axis_offenders.empty:
        raise ValueError(
            f"signature_set / signature_id must not contain the reserved axis "
            f"separator '{SIGNATURE_AXIS_SEP}'. Offending entries: "
            f"{axis_offenders.to_dict('records')[:10]}"
        )
    sig["gene"] = sig["gene"].astype(str)
    member_union = sorted(sig["gene"].unique().tolist())

    train_pool = _config_signature_train_pool(splits_manifest)
    if not train_pool:
        raise RuntimeError("Signature coverage gate: empty config train-slide pool.")
    single_unit = str(splits_manifest.get("type", "")) == "within_slide_area"

    # Support over all eval slides (the pooled gate uses the train-pool subset)
    all_slides = sorted(rows_df["slide_id"].astype(str).unique().tolist())
    support_all = _support_table_for_split(
        cfg=cfg, rows_df=rows_df, train_slides=all_slides, candidate_genes=member_union,
    )
    eligibility_all = _slide_gene_eligibility_from_support(cfg, support_all)
    support_pool = (
        support_all.loc[support_all["slide_id"].astype(str).isin(set(train_pool))].copy()
        if not support_all.empty
        else support_all
    )

    # --- Gate 1: config-level pooled coverage (which members enter A) -----------------
    gene_support = _covered_member_genes(cfg, support_pool, single_unit=single_unit)
    covered_set = set(
        gene_support.loc[gene_support["is_covered"], "gene"].astype(str).tolist()
    ) if not gene_support.empty else set()

    # --- Per-(slide, signature) eligibility (full-membership denominator) -------------
    slide_elig = _signature_slide_eligibility_table(
        sig,
        eligibility_all,
        all_slides,
        min_covered_fraction=min_frac,
        min_covered_genes=min_genes,
    )
    max_slide_cov = (
        slide_elig.groupby(["signature_set", "signature_id"])["coverage_fraction_on_slide"]
        .max()
        .to_dict()
        if not slide_elig.empty
        else {}
    )

    # --- Gate 2: signature-axis weak-train-slide filter + frozen-split rewrite --------
    splits_manifest = copy.deepcopy(dict(splits_manifest))
    pooled_pass = {
        (str(s), str(i))
        for (s, i), grp in sig.groupby(["signature_set", "signature_id"], sort=True)
        for _n_cov in [sum(1 for g in set(grp["gene"].tolist()) if g in covered_set)]
        if _n_cov >= min_genes
        and (_n_cov / float(len(set(grp["gene"].tolist())) or 1)) >= min_frac
    }
    elig_pooled = (
        slide_elig.loc[
            slide_elig.apply(
                lambda r: (str(r["signature_set"]), str(r["signature_id"])) in pooled_pass,
                axis=1,
            )
        ].copy()
        if not slide_elig.empty
        else slide_elig
    )
    weak_filter = _apply_signature_weak_train_slide_filter(
        cfg=cfg,
        rows_df=rows_df,
        support_all=support_all,
        elig_pooled=elig_pooled,
        splits_manifest=splits_manifest,
        splits_dir=splits_dir,
    )

    # --- Gate 3: fold feasibility (>= 1 eligible train slide in every fold) -----------
    fold_units = _signature_fold_train_units(splits_manifest)
    eligible_slides_by_sig: Dict[Tuple[str, str], set] = {}
    if not slide_elig.empty:
        for (s, i), grp in slide_elig.loc[slide_elig["eligible"]].groupby(
            ["signature_set", "signature_id"], sort=True
        ):
            eligible_slides_by_sig[(str(s), str(i))] = set(
                grp["slide_id"].astype(str).tolist()
            )

    surviving_rows: List[Dict[str, Any]] = []
    per_signature: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for (set_name, sig_id), grp in sig.groupby(["signature_set", "signature_id"], sort=True):
        members = sorted(set(grp["gene"].tolist()))
        category = str(grp["signature_category"].iloc[0])
        covered = [g for g in members if g in covered_set]
        n_total = len(members)
        n_cov = len(covered)
        frac = (float(n_cov) / float(n_total)) if n_total else 0.0
        key = (str(set_name), str(sig_id))
        elig_slides = eligible_slides_by_sig.get(key, set())
        starved_folds = [
            label for label, train in fold_units if not (set(train) & elig_slides)
        ]
        record = {
            "signature_set": str(set_name),
            "signature_id": str(sig_id),
            "signature_category": category,
            "n_members_total": int(n_total),
            "n_members_covered": int(n_cov),
            "coverage_fraction": float(frac),
            "n_eligible_slides": int(len(elig_slides)),
            "max_slide_coverage_fraction": float(max_slide_cov.get(key, 0.0)),
            "eligible_slides": sorted(elig_slides),
        }
        reasons: List[str] = []
        if n_cov < min_genes:
            reasons.append(f"n_members_covered={n_cov} < min_covered_genes={min_genes}")
        if frac < min_frac:
            reasons.append(
                f"coverage_fraction={frac:.3f} < min_covered_fraction={min_frac}"
            )
        if not reasons and starved_folds:
            # Pooled gate passed but some fold has no eligible train slide
            reasons.append(
                f"eligible on {len(elig_slides)} slide(s); "
                f"max per-slide coverage={record['max_slide_coverage_fraction']:.3f} < "
                f"min_covered_fraction={min_frac}; no eligible train slide in fold(s) "
                f"{starved_folds[:5]}"
            )
        if reasons:
            record["reason"] = "; ".join(reasons)
            skipped.append(record)
            continue

        gsub = grp.loc[grp["gene"].isin(covered_set)]
        # A gene may appear once per signature; keep the largest |weight| if not.
        gsub = gsub.sort_values("weight", key=lambda s: s.abs(), ascending=False)
        gsub = gsub.drop_duplicates(subset=["gene"], keep="first")
        denom = float(gsub["weight"].abs().sum())
        record["denominator_abs_weight_sum"] = denom
        for _, r in gsub.iterrows():
            surviving_rows.append({
                "signature_set": str(set_name),
                "signature_id": str(sig_id),
                "signature_category": category,
                "gene": str(r["gene"]),
                "weight": float(r["weight"]),
            })
        per_signature.append(record)

    if not surviving_rows:
        raise RuntimeError(
            "No signature survived the coverage gate "
            f"(min_covered_fraction={min_frac}, min_covered_genes={min_genes}) on "
            f"'{sub_cfg}'. Relax the gate, add signature sets, or check the "
            "member-gene overlap with the config panel. Per-signature reasons are in "
            "signature_targets_manifest.json (skipped_signatures)."
        )

    targets_df = pd.DataFrame(
        surviving_rows,
        columns=["signature_set", "signature_id", "signature_category", "gene", "weight"],
    )
    targets_path = genes_dir / "signature_targets.parquet"
    mkdir(genes_dir)
    targets_df.to_parquet(targets_path, index=False)

    # The run-time mask, restricted to the surviving axis.
    surviving_keys = {(r["signature_set"], r["signature_id"]) for r in per_signature}
    slide_elig_out = (
        slide_elig.loc[
            slide_elig.apply(
                lambda r: (str(r["signature_set"]), str(r["signature_id"])) in surviving_keys,
                axis=1,
            )
        ].copy()
        if not slide_elig.empty
        else slide_elig
    )
    slide_elig_path = genes_dir / "signature_slide_eligibility.parquet"
    slide_elig_out.to_parquet(slide_elig_path, index=False)

    covered_union = sorted(targets_df["gene"].astype(str).unique().tolist())

    signature_ids = [
        {"signature_set": s["signature_set"], "signature_id": s["signature_id"]}
        for s in per_signature
    ]

    manifest = {
        "target_basis": "signatures",
        "sub_eval_configuration": sub_cfg,
        "signature_sets": list(cfg["protocol"].get("signature_sets", []) or []),
        "eval_signatures_id": cfg["protocol"].get("eval_signatures_id"),
        "eval_signatures_INT_id": cfg["protocol"].get("eval_signatures_INT_id"),
        "aggregation": str(st_cfg.get("aggregation", "weighted_zscore_mean")),
        "coverage_gate": {"min_covered_fraction": min_frac, "min_covered_genes": min_genes},
        "train_pool_slides": train_pool,
        "n_member_genes_total": int(len(member_union)),
        "n_member_genes_covered": int(len(covered_set)),
        "n_signatures_surviving": int(len(per_signature)),
        "n_signatures_skipped": int(len(skipped)),
        "surviving_signatures": per_signature,
        "skipped_signatures": skipped,
        "covered_member_union": covered_union,
        "source_table_sha256": _signature_set_source_shas(cfg),
        "signature_targets_parquet": str(targets_path),
        "signature_targets_sha256": sha256_file(targets_path),
        "slide_eligibility_parquet": str(slide_elig_path),
        "slide_eligibility_sha256": sha256_file(slide_elig_path),
        "slide_eligibility_gate": {
            "min_covered_fraction": min_frac,
            "min_covered_genes": min_genes,
            "denominator": "full_membership",
            "n_slides": int(len(all_slides)),
            "n_eligible_slide_signature_pairs": (
                int(slide_elig_out["eligible"].sum()) if not slide_elig_out.empty else 0
            ),
        },
        "weak_train_slide_filter": weak_filter,
        "single_unit_coverage": bool(single_unit),
        "notes": (
            "Signature-target axis. The gene panel is the covered member union "
            "(no top-256 auto-cap). Per-signature aggregate = Sum w*z / Sum|w| over "
            "covered, measured members (available-weight renormalized per location at "
            "run time). The coverage gate is applied pooled over the config train pool "
            "(which members enter A) and per (slide, signature) against full membership "
            "(which slides can supervise or score a signature). A signature also needs "
            "an eligible train slide in every fold."
        ),
    }
    save_json(genes_dir / "signature_targets_manifest.json", manifest)

    genes_manifest = {
        "protocol": sub_cfg,
        "target_basis": "signatures",
        "trainable_genes": covered_union,
        "train_gene_names": covered_union,
        "n_trainable_member_genes": int(len(covered_union)),
        "surviving_signatures": signature_ids,
        "signature_targets_parquet": str(targets_path),
        "signature_targets_manifest": str(genes_dir / "signature_targets_manifest.json"),
        "signature_slide_eligibility_parquet": str(slide_elig_path),
    }
    save_json(genes_dir / "genes_manifest.json", genes_manifest)
    return genes_manifest


def persist_signature_noise_ceiling(
    cfg: Mapping[str, Any],
    rows_df: pd.DataFrame,
    genes_manifest: Mapping[str, Any],
    out_dir: Path,
    target_norm_dir: Path,
    *,
    row_scope_record: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Save the noise ceiling of the signature axis: thinned halves (as in persist_noise_ceiling) aggregated through the
    signature weights, correlated per (slide, signature) across locations (reliability_signature.parquet) and per
    (slide, location) across signatures (reliability_profile_signature.parquet), stepped up with Spearman-Brown.
    """
    mkdir(out_dir)
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    target_transform = dict(cfg["protocol"].get("target_transform", {}))
    target_sum = float(target_transform.get("target_sum", 1.0e4))
    protocol_seed = int(cfg["protocol"]["protocol_seed"])
    slide_records = load_slide_records(training_dataset_id)
    eval_root = resolve_eval_dataset_root(training_dataset_id)

    # Member genes whose z-stats/reliability were frozen, and the frozen A matrix.
    member_genes = list(
        genes_manifest.get("train_gene_names", genes_manifest.get("trainable_genes", []))
    )
    targets_parquet = genes_manifest.get("signature_targets_parquet")
    if not targets_parquet or not Path(targets_parquet).is_file():
        raise RuntimeError(
            "persist_signature_noise_ceiling: signature_targets.parquet not found; "
            "persist_signature_targets must run first."
        )
    targets_df = pd.read_parquet(targets_parquet)

    # Frozen per-(slide,gene) z-score stats (written by persist_target_norm_stats).
    stats_path = Path(target_norm_dir) / "zscore_stats.parquet"
    if not stats_path.is_file():
        raise RuntimeError(
            f"persist_signature_noise_ceiling: frozen z-stats not found at "
            f"'{stats_path}'. persist_target_norm_stats must run first."
        )
    stats_df = pd.read_parquet(stats_path)
    stat_lookup: Dict[Tuple[str, str], Tuple[float, float]] = {
        (str(s), str(g)): (float(m), float(sd))
        for s, g, m, sd in zip(
            stats_df["slide_id"], stats_df["gene"], stats_df["mean"], stats_df["std"]
        )
    }

    rows: List[Dict[str, Any]] = []
    loc_rows: List[Dict[str, Any]] = []
    for slide_idx, slide_id in enumerate(sorted(rows_df["slide_id"].astype(str).unique().tolist())):
        rec = slide_records[slide_id]
        slide_rows = rows_df.loc[rows_df["slide_id"].astype(str) == slide_id]
        row_keys = slide_rows["row_key"].astype(str).to_numpy(dtype=object)
        adata_path = eval_root / "st" / f"{slide_id}.h5ad"
        mouse_map = eval_root / "mouse_to_human_gene_mappings" / f"{slide_id}_mouse2human_mapping.parquet"
        C, genes = extract_raw_counts_for_rows(
            adata_path=adata_path,
            slide_record=rec,
            ordered_row_keys=row_keys,
            mouse_mapping_path=mouse_map if mouse_map.exists() else None,
        )
        C = C.tocsr()
        counts = np.clip(np.rint(C.data), 0.0, None).astype(np.int64)
        rng = np.random.default_rng([protocol_seed, slide_idx])
        c1 = rng.binomial(counts, 0.5)
        c2 = counts - c1
        C1 = sparse.csr_matrix((c1.astype(np.float64), C.indices, C.indptr), shape=C.shape)
        C2 = sparse.csr_matrix((c2.astype(np.float64), C.indices, C.indptr), shape=C.shape)
        s1 = np.asarray(C1.sum(axis=1)).reshape(-1).astype(np.float64)
        s2 = np.asarray(C2.sum(axis=1)).reshape(-1).astype(np.float64)

        gene_lookup = {g: i for i, g in enumerate(genes.tolist())}
        present = [(g, gene_lookup[g]) for g in member_genes if g in gene_lookup]
        if not present:
            continue
        sub_genes = [g for g, _ in present]
        cols = [c for _, c in present]
        sub1 = C1[:, cols].toarray().astype(np.float64)
        sub2 = C2[:, cols].toarray().astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale1 = np.where(s1 > 0.0, target_sum / s1, 0.0)
            scale2 = np.where(s2 > 0.0, target_sum / s2, 0.0)
        t1 = np.log1p(sub1 * scale1[:, None])
        t2 = np.log1p(sub2 * scale2[:, None])
        bad = (s1 <= 0.0) | (s2 <= 0.0)
        if np.any(bad):
            t1[bad, :] = np.nan
            t2[bad, :] = np.nan

        # Z-score each half with the per-(slide, gene) stats
        means = np.array([stat_lookup.get((slide_id, g), (np.nan, np.nan))[0] for g in sub_genes])
        stds = np.array([stat_lookup.get((slide_id, g), (np.nan, np.nan))[1] for g in sub_genes])
        # As in apply_zscore: std clamped to 1e-8, genes without stats dropped
        denom_std = np.where(np.isfinite(stds), np.maximum(stds, 1e-8), np.nan)
        z1 = (t1 - means[None, :]) / denom_std[None, :]
        z2 = (t2 - means[None, :]) / denom_std[None, :]
        # Winsorize per gene before aggregating (as for the target)
        z1 = winsorize_zscored(z1, target_transform)
        z2 = winsorize_zscored(z2, target_transform)

        F1 = np.isfinite(z1)
        F2 = np.isfinite(z2)
        Y1, sig_keys, _ = aggregate_signature_targets(z1, F1, sub_genes, targets_df)
        Y2, _, _ = aggregate_signature_targets(z2, F2, sub_genes, targets_df)

        # Localization axis: per (slide, signature), correlate the two half-aggregates
        # across the slide's locations.
        for s_idx, (set_name, sig_id) in enumerate(sig_keys):
            x = Y1[:, s_idx]
            y = Y2[:, s_idx]
            m = np.isfinite(x) & np.isfinite(y)
            n = int(np.sum(m))
            if n >= 2:
                r_pear, r_spear, degenerate = _halfsplit_reliability(x[m], y[m])
            else:
                r_pear, r_spear, degenerate = np.nan, np.nan, True
            rows.append(
                {
                    "slide_id": slide_id,
                    "signature_set": str(set_name),
                    "signature_id": str(sig_id),
                    "r_pearson": _spearman_brown_full_depth(r_pear),
                    "r_spearman": _spearman_brown_full_depth(r_spear),
                    "r_pearson_halfsplit": r_pear,
                    "r_spearman_halfsplit": r_spear,
                    "n_locations": n,
                    "degenerate_half": bool(degenerate),
                }
            )

        # Profile axis: per (slide, location), across signatures (n_genes = number of signatures)
        for i, rk in enumerate(row_keys.tolist()):
            x = Y1[i, :]
            y = Y2[i, :]
            m = np.isfinite(x) & np.isfinite(y)
            n = int(np.sum(m))
            if n >= 2:
                r_prof_pear, r_prof_spear, degenerate = _halfsplit_reliability(x[m], y[m])
            else:
                r_prof_pear, r_prof_spear, degenerate = np.nan, np.nan, True
            loc_rows.append(
                {
                    "slide_id": slide_id,
                    "row_key": str(rk),
                    "r_profile_pearson": _spearman_brown_full_depth(r_prof_pear),
                    "r_profile_spearman": _spearman_brown_full_depth(r_prof_spear),
                    "r_profile_pearson_halfsplit": r_prof_pear,
                    "r_profile_spearman_halfsplit": r_prof_spear,
                    "n_genes": n,
                    "degenerate_half": bool(degenerate),
                }
            )

    # `columns=` is a selector here too -- see persist_noise_ceiling.
    reliability_df = pd.DataFrame(
        rows,
        columns=[
            "slide_id", "signature_set", "signature_id",
            "r_pearson", "r_spearman",
            "r_pearson_halfsplit", "r_spearman_halfsplit", "n_locations",
            "degenerate_half",
        ],
    )
    reliability_profile_df = pd.DataFrame(
        loc_rows,
        columns=[
            "slide_id", "row_key", "r_profile_pearson", "r_profile_spearman",
            "r_profile_pearson_halfsplit", "r_profile_spearman_halfsplit", "n_genes",
            "degenerate_half",
        ],
    )
    reliability_path = out_dir / "reliability_signature.parquet"
    reliability_profile_path = out_dir / "reliability_profile_signature.parquet"
    reliability_df.to_parquet(reliability_path, index=False)
    reliability_profile_df.to_parquet(reliability_profile_path, index=False)

    def _finite_mean(s: pd.Series) -> float:
        v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        return float(np.mean(v)) if v.size else float("nan")

    manifest = {
        "method": "binomial_count_thinning_on_aggregate",
        "thinning": (
            "c1 ~ Binom(c, 0.5); c2 = c - c1; each half depth-normalized by its own "
            "full-panel total, log1p, then z-scored with the frozen per-(slide,gene) "
            "stats, then aggregated through the frozen weight matrix A "
            "(Sum w*z / Sum|w|, available-weight renormalized)."
        ),
        "ceiling_norm": (
            "per-signature localization metrics are reported as metric / "
            "sqrt(clip(r_full, eps, 1)); r_full = 2 r / (1 + r) is the "
            "Spearman-Brown full-depth reliability of the aggregate. The profile axis "
            "is normalized the same way by reliability_profile_signature.parquet."
        ),
        "axes": (
            "reliability_signature.parquet is per (slide, signature): the two half-"
            "aggregates correlated across locations (localization axis). "
            "reliability_profile_signature.parquet is per (slide, location): the same "
            "two half-aggregates correlated across signatures (profile axis), used by "
            "location_macro_profile_spearman_ceiling_norm."
        ),
        "rationale": (
            "Recomputed on the aggregate (aggregation over member genes raises the "
            "reliability)."
        ),
        "rng": "numpy default_rng seeded by [protocol_seed, slide_index] (matches per-gene ceiling)",
        "protocol_seed": protocol_seed,
        "target_sum": target_sum,
        "n_slide_signature_pairs": int(reliability_df.shape[0]),
        "n_slides": int(reliability_df["slide_id"].nunique()) if not reliability_df.empty else 0,
        "n_signatures": (
            int(reliability_df[["signature_set", "signature_id"]].drop_duplicates().shape[0])
            if not reliability_df.empty else 0
        ),
        "n_locations": int(reliability_profile_df.shape[0]),
        "mean_r_spearman": _finite_mean(reliability_df["r_spearman"]) if not reliability_df.empty else float("nan"),
        "mean_r_spearman_halfsplit": (
            _finite_mean(reliability_df["r_spearman_halfsplit"]) if not reliability_df.empty else float("nan")
        ),
        "mean_r_profile_spearman": (
            _finite_mean(reliability_profile_df["r_profile_spearman"])
            if not reliability_profile_df.empty else float("nan")
        ),
        "mean_r_profile_spearman_halfsplit": (
            _finite_mean(reliability_profile_df["r_profile_spearman_halfsplit"])
            if not reliability_profile_df.empty else float("nan")
        ),
        "reliability_signature_parquet": str(reliability_path),
        "reliability_signature_sha256": sha256_file(reliability_path),
        "reliability_profile_signature_parquet": str(reliability_profile_path),
        "reliability_profile_signature_sha256": sha256_file(reliability_profile_path),
    }
    # Row scope of the ceiling (same as the z-score stats)
    if row_scope_record:
        manifest.update(dict(row_scope_record))
    save_json(out_dir / "reliability_signature_manifest.json", manifest)
    return manifest


# -----------------------------------------------------------------------------
# Full settings fill
# -----------------------------------------------------------------------------


def _assert_no_local_protocol_files_for_child(paths: ProtocolPaths) -> None:
    forbidden = [
        paths.rows_dir / "all_rows.parquet",
        paths.splits_dir / "splits_manifest.json",
        paths.genes_dir / "genes_manifest.json",
    ]
    existing = [str(p) for p in forbidden if p.exists()]
    if existing:
        raise RuntimeError(
            "Control child eval_id already contains local protocol files (child controls read "
            f"rows/splits/genes from their reference eval_id). Existing files: {existing}"
        )


def _fill_control_child_settings(
    *,
    cfg: Mapping[str, Any],
    paths: ProtocolPaths,
) -> Dict[str, Any]:
    """Fill the settings of a STHELAR_TRAIN_COORDS child (rows, splits and genes are read from the reference eval_id)."""
    _assert_no_local_protocol_files_for_child(paths)

    reference_eval_id = str(cfg["protocol"]["reference_eval_id"])
    ref_dir, ref_cfg, ref_protocol_manifest = _load_validated_reference_for_control(
        child_cfg=cfg,
        child_eval_dir=paths.eval_dir,
    )
    ref_settings = ref_dir / "settings"

    rows_df = pd.read_parquet(ref_settings / "rows" / "all_rows.parquet")
    splits_manifest = load_json(ref_settings / "splits" / "splits_manifest.json")
    genes_manifest = load_json(ref_settings / "genes" / "genes_manifest.json")
    scgpt_payload = load_json(ref_settings / "scgpt_manifest.json")

    area_manifest = {"enabled": False, "slides": []}
    save_json(paths.area_dir / "area_split_manifest.json", area_manifest)

    reference_payload = _build_reference_protocol_payload(
        child_cfg=cfg,
        reference_eval_id=reference_eval_id,
        reference_eval_dir=ref_dir,
        reference_cfg=ref_cfg,
    )
    save_json(paths.settings_dir / "reference_protocol.json", reference_payload)

    protocol_manifest = {
        "created_utc": cfg["identity"]["created_utc"],
        "identity": cfg["identity"],
        "protocol": cfg["protocol"],
        "eval_config_sha256": protocol_identity_sha256(cfg),
        "derived_from_reference": True,
        "run_eval_dir": str(paths.eval_dir),
        "protocol_eval_dir": str(ref_dir),
        "reference_protocol": reference_payload,
        "gene_panel_strategy": ref_protocol_manifest.get(
            "gene_panel_strategy",
            _gene_panel_strategy_payload(ref_cfg),
        ),
        "gene_canonicalization": ref_protocol_manifest.get("gene_canonicalization", {}),
        "prediction_head": prediction_head_manifest_payload(cfg),
        "scgpt": scgpt_payload,
        "splits_manifest": str(ref_settings / "splits" / "splits_manifest.json"),
        "genes_manifest": str(ref_settings / "genes" / "genes_manifest.json"),
        "target_norm_manifest": str(ref_settings / "target_norm" / "target_norm_manifest.json"),
        "noise_ceiling_manifest": str(ref_settings / "noise_ceiling" / "noise_ceiling_manifest.json"),
        "gene_strata_manifest": str(ref_settings / "genes" / "gene_strata_manifest.json"),
        "signature_strata_manifest": str(ref_settings / "genes" / "signature_strata_manifest.json"),
        "hest_cv_support_manifest": "",
        "area_split_manifest": str(ref_settings / "area_splits" / "area_split_manifest.json"),
        "row_manifest": str(ref_settings / "rows" / "all_rows.parquet"),
        "benchmark_contract_notes": [
            "This is a child control eval_id derived from a frozen STHELAR_TRAIN_CV reference protocol.",
            "Rows, splits, genes, test panels, metric settings, and probe hyperparameters are inherited from the reference.",
            "The reference eval_id is read-only and must not be modified after child creation.",
            "Reference file hashes are stored in settings/reference_protocol.json and checked before running.",
        ] + _protocol_contract_notes(cfg),
    }
    _row_manifest_summary(rows_df).to_csv(paths.info_dir / "row_manifest_summary.csv", index=False)
    _write_protocol_reports(
        cfg,
        rows_df,
        splits_manifest,
        genes_manifest,
        paths.info_dir,
    )

    # Written last and atomically (marks a completed fill)
    save_json_atomic(paths.settings_dir / "protocol_manifest.json", protocol_manifest)

    return protocol_manifest


def fill_settings_folder(eval_dir: Path) -> Dict[str, Any]:
    cfg = load_yaml(eval_dir / "settings" / "eval_config.yaml")
    validate_protocol_identity(cfg)
    paths = make_protocol_paths(eval_dir)

    sub_cfg = str(cfg["protocol"]["sub_eval_configuration"])

    if sub_cfg in CONTROL_SUB_EVAL_CONFIGS:
        return _fill_control_child_settings(cfg=cfg, paths=paths)


    target_basis = str(cfg["protocol"].get("target_basis", "genes"))
    hest_cv_support_manifest: Dict[str, Any] = {"enabled": False}

    if sub_cfg in HEST_CV_SUB_EVAL_CONFIGS:
        rows_df, splits_manifest, genes_manifest, hest_cv_support_manifest = build_hest_tissue_cv_settings(
            cfg=cfg,
            paths=paths,
        )
        if target_basis == "signatures":
            # Signature targets over the member-gene union
            genes_manifest = persist_signature_targets(
                cfg, rows_df, splits_manifest, paths.genes_dir,
                splits_dir=paths.splits_dir,
            )
        area_manifest = {"enabled": False, "slides": []}
        save_json(paths.area_dir / "area_split_manifest.json", area_manifest)
    else:
        rows_df = persist_row_manifests(cfg, paths.rows_dir)
        _validate_novae_row_manifest_contract(cfg, rows_df)
        _row_manifest_summary(rows_df).to_csv(paths.info_dir / "row_manifest_summary.csv", index=False)

        splits_manifest = build_slide_level_cv_splits(cfg, rows_df, paths.splits_dir)
        if sub_cfg == "STHELAR_TEST_ADAPTED":
            # Area stripes (same for both target bases)
            area_manifest = build_area_split_manifests(
                cfg,
                paths.area_dir,
                rows_df=rows_df,
            )
        else:
            area_manifest = {"enabled": False, "slides": []}
            save_json(paths.area_dir / "area_split_manifest.json", area_manifest)

        if target_basis == "signatures":
            genes_manifest = persist_signature_targets(
                cfg, rows_df, splits_manifest, paths.genes_dir,
                splits_dir=paths.splits_dir,
            )
        else:
            genes_manifest = persist_gene_protocols(
                cfg,
                rows_df,
                splits_manifest,
                paths.genes_dir,
                splits_dir=paths.splits_dir,
                area_manifest=area_manifest,
            )

    # STHELAR_TEST_ADAPTED: z-score stats and noise ceiling on the stripe rows only (safe-border rows dropped)
    stats_rows_df, row_scope_record = stripe_scoped_rows(cfg, rows_df, area_manifest)
    if row_scope_record["n_rows_excluded"]:
        logger.info(
            "%s: z-score stats and noise ceiling are frozen over %d stripe rows "
            "(%d safe-border rows excluded).",
            sub_cfg, row_scope_record["n_rows"], row_scope_record["n_rows_excluded"],
        )

    persist_target_norm_stats(
        cfg, stats_rows_df, genes_manifest, paths.target_norm_dir,
        row_scope_record=row_scope_record,
    )
    if target_basis == "signatures":
        # Signature axis: aggregate ceilings only (no per-gene ceiling, no gene/signature strata)
        persist_signature_noise_ceiling(
            cfg, stats_rows_df, genes_manifest, paths.noise_ceiling_dir,
            paths.target_norm_dir, row_scope_record=row_scope_record,
        )
    else:
        persist_noise_ceiling(
            cfg,
            stats_rows_df,
            genes_manifest,
            paths.noise_ceiling_dir,
            paths.target_norm_dir,
            row_scope_record=row_scope_record,
        )
        persist_gene_strata(cfg, genes_manifest, paths.genes_dir)
        persist_signature_strata(cfg, genes_manifest, paths.genes_dir)

    scgpt_payload = scgpt_vocab_manifest(str(cfg["protocol"]["scgpt_model_name"]))
    save_json(paths.settings_dir / "scgpt_manifest.json", scgpt_payload)

    protocol_manifest = {
        "created_utc": cfg["identity"]["created_utc"],
        "identity": cfg["identity"],
        "protocol": cfg["protocol"],
        "eval_config_sha256": protocol_identity_sha256(cfg),
        "gene_panel_strategy": _gene_panel_strategy_payload(
            cfg,
            auto_manifest_path=(
                str(paths.genes_dir / "auto_gene_panels_manifest.json")
                if _gene_panel_mode(cfg) == "auto"
                else ""
            ),
        ),
        "gene_canonicalization": {
            "case_normalization": "identity",
            "mouse_to_human_mapping": "per-slide parquet columns var_index -> target_gene_name",
            "drop_unmapped_mouse_genes": True,
            "duplicate_collapse": "sum raw counts by mapped human symbol before log1p",
        },
        "prediction_head": prediction_head_manifest_payload(cfg),
        "scgpt": scgpt_payload,
        "splits_manifest": str(paths.splits_dir / "splits_manifest.json"),
        "genes_manifest": str(paths.genes_dir / "genes_manifest.json"),
        "target_norm_manifest": str(paths.target_norm_dir / "target_norm_manifest.json"),
        "noise_ceiling_manifest": (
            "" if target_basis == "signatures"
            else str(paths.noise_ceiling_dir / "noise_ceiling_manifest.json")
        ),
        "target_basis": target_basis,
        "gene_strata_manifest": (
            "" if target_basis == "signatures"
            else str(paths.genes_dir / "gene_strata_manifest.json")
        ),
        "signature_strata_manifest": (
            "" if target_basis == "signatures"
            else str(paths.genes_dir / "signature_strata_manifest.json")
        ),
        "signature_targets_manifest": (
            str(paths.genes_dir / "signature_targets_manifest.json")
            if target_basis == "signatures" else ""
        ),
        "reliability_signature_manifest": (
            str(paths.noise_ceiling_dir / "reliability_signature_manifest.json")
            if target_basis == "signatures" else ""
        ),
        "hest_cv_support_manifest": (
            str(paths.splits_dir / "hest_cv_support_manifest.json")
            if hest_cv_support_manifest.get("enabled", False)
            else ""
        ),
        "area_split_manifest": str(paths.area_dir / "area_split_manifest.json"),
        "row_manifest": str(paths.rows_dir / "all_rows.parquet"),
        "benchmark_contract_notes": _protocol_contract_notes(cfg),
    }
    _write_protocol_reports(
        cfg,
        rows_df,
        splits_manifest,
        genes_manifest,
        paths.info_dir,
    )

    # protocol_manifest.json is written last and atomically: its presence marks a completed fill
    save_json_atomic(paths.settings_dir / "protocol_manifest.json", protocol_manifest)
    return protocol_manifest


# -----------------------------------------------------------------------------
# Human-readable reports
# -----------------------------------------------------------------------------

def _iter_gene_payloads_for_reports(genes_manifest: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    if "folds" in genes_manifest:
        return list(genes_manifest.get("folds", []))
    if "trainable_genes" in genes_manifest:
        return [genes_manifest]
    return []


def _write_gene_selection_diagnostics(
    *,
    cfg: Mapping[str, Any],
    genes_manifest: Mapping[str, Any],
    info_dir: Path,
) -> None:
    """Write the gene-selection diagnostics (from the gene payload CSVs) in info_dir."""
    if str(genes_manifest.get("target_basis", "genes")) == "signatures":
        # Not applicable on the signature axis
        return
    payloads = _iter_gene_payloads_for_reports(genes_manifest)
    summary_rows: List[Dict[str, Any]] = []
    eligibility_frames: List[pd.DataFrame] = []
    slide_frames: List[pd.DataFrame] = []

    min_trainable = int(cfg["protocol"]["gene_selection"]["min_trainable_genes"])

    for idx, payload in enumerate(payloads):
        fold_id = int(payload.get("fold_id", idx if len(payloads) > 1 else -1))
        fold_name = str(payload.get("fold_name", f"fold_{fold_id:02d}" if fold_id >= 0 else "global"))
        elig_path = str(payload.get("train_slide_gene_eligibility_csv", "") or "")
        slide_path = str(payload.get("train_slide_eligibility_summary_csv", "") or "")

        n_eligible_pairs = 0
        n_pairs = 0
        n_dropped = len(payload.get("dropped_weak_train_slides", []) or [])

        if elig_path:
            p = Path(elig_path)
            if p.exists():
                edf = pd.read_csv(p)
                edf.insert(0, "diagnostic_fold_name", fold_name)
                edf.insert(0, "diagnostic_fold_id", fold_id)
                eligibility_frames.append(edf)
                n_pairs = int(edf.shape[0])
                if "eligible_slide_gene" in edf.columns:
                    n_eligible_pairs = int(edf["eligible_slide_gene"].astype(bool).sum())

        if slide_path:
            p = Path(slide_path)
            if p.exists():
                sdf = pd.read_csv(p)
                sdf.insert(0, "diagnostic_fold_name", fold_name)
                sdf.insert(0, "diagnostic_fold_id", fold_id)
                slide_frames.append(sdf)
                if "drop_weak_train_slide" in sdf.columns:
                    n_dropped = int(sdf["drop_weak_train_slide"].astype(bool).sum())

        n_trainable = int(len(payload.get("trainable_genes", []) or []))
        if n_trainable < min_trainable:
            raise RuntimeError(
                f"Fold '{fold_name}' has only {n_trainable} trainable genes, "
                f"below min_trainable_genes={min_trainable}. "
                "Relax thresholds or inspect train_slide_gene_eligibility.csv."
            )

        summary_rows.append(
            {
                "fold_id": fold_id,
                "fold_name": fold_name,
                "n_trainable_genes": n_trainable,
                "n_train_gene_names": int(len(payload.get("train_gene_names", []) or [])),
                "n_test_gene_names": int(len(payload.get("test_gene_names", []) or [])),
                "n_slide_gene_pairs": n_pairs,
                "n_eligible_slide_gene_pairs": n_eligible_pairs,
                "n_dropped_weak_train_slides": n_dropped,
                "train_gene_support_csv": str(payload.get("train_gene_support_csv", "")),
                "train_slide_gene_eligibility_csv": elig_path,
                "train_slide_eligibility_summary_csv": slide_path,
            }
        )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            info_dir / "gene_selection_summary_by_fold.csv",
            index=False,
        )
    if eligibility_frames:
        pd.concat(eligibility_frames, axis=0, ignore_index=True).to_csv(
            info_dir / "train_slide_gene_eligibility_by_fold.csv",
            index=False,
        )
    if slide_frames:
        pd.concat(slide_frames, axis=0, ignore_index=True).to_csv(
            info_dir / "train_slide_eligibility_summary_by_fold.csv",
            index=False,
        )


def _write_protocol_reports(
    cfg: Mapping[str, Any],
    rows_df: pd.DataFrame,
    splits_manifest: Mapping[str, Any],
    genes_manifest: Mapping[str, Any],
    info_dir: Path,
) -> None:
    _write_gene_selection_diagnostics(
        cfg=cfg,
        genes_manifest=genes_manifest,
        info_dir=info_dir,
    )

    by_slide = (
        rows_df.groupby(["slide_id", "source", "subset", "tissue"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(["source", "subset", "tissue", "n_rows", "slide_id"], ascending=[True, True, True, False, True])
    )
    by_slide.to_csv(info_dir / "rows_by_slide.csv", index=False)

    by_subset = (
        rows_df.groupby(["source", "subset", "tissue"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(["source", "subset", "tissue"]) 
    )
    by_subset.to_csv(info_dir / "rows_by_subset.csv", index=False)

    notes = [
        f"eval_configuration: {cfg['protocol']['eval_configuration']}",
        f"sub_eval_configuration: {cfg['protocol']['sub_eval_configuration']}",
        f"level: {cfg['protocol']['level']}",
        f"modelgroup: {cfg['protocol']['modelgroup']}",
        f"head_type: {cfg['protocol'].get('head', {}).get('type', 'ridge')}",
        f"input_reduction: none (full-dimensional features)",
        f"gene_panel_strategy: {_gene_panel_mode(cfg)}",
        f"slide_count_total: {rows_df['slide_id'].astype(str).nunique()}",
        f"row_count_total: {len(rows_df)}",
    ]
    if _gene_panel_mode(cfg) == "auto":
        notes.extend(
            [
                "Gene panel strategy: auto.",
                "This is a target-informed diagnostic benchmark.",
                "Target expression from the selected evaluation unit was used to define frozen candidate gene panels before encoder comparison.",
                (
                    "For HEST, auto panels are selected per tissue."
                    if str(cfg["protocol"]["sub_eval_configuration"]) in HEST_CV_SUB_EVAL_CONFIGS
                    else "For STHELAR_TEST_ADAPTED, auto panels are selected per slide."
                ),
                "auto and G_base eval_ids use different gene panels.",
            ]
        )
    if splits_manifest["type"] == "slide_cv":
        notes.append(f"n_folds: {len(splits_manifest['folds'])}")
        if "split_builder" in splits_manifest:
            notes.append(f"split_builder: {splits_manifest['split_builder']}")
        if "outer_test_fold_slide_capacities" in splits_manifest:
            notes.append(f"outer_test_fold_slide_capacities: {splits_manifest['outer_test_fold_slide_capacities']}")
        if "balancing_strategy" in splits_manifest:
            strat = splits_manifest["balancing_strategy"]
            notes.append(
                "balancing_strategy: "
                f"primary={strat.get('primary')} "
                f"secondary={strat.get('secondary')} "
                f"tertiary={strat.get('tertiary')}"
            )

        fold_report_rows: List[Dict[str, Any]] = []
        for fold in splits_manifest["folds"]:
            line = (
                f"fold {fold['fold_id']}: "
                f"train={len(fold['train_slides'])} "
                f"test={len(fold['test_slides'])}"
            )
            if {"train_row_count", "test_row_count"}.issubset(set(fold.keys())):
                line += (
                    f" | train_rows={int(fold['train_row_count'])} "
                    f"test_rows={int(fold['test_row_count'])}"
                )
            notes.append(line)
            fold_report_rows.append(
                {
                    "fold_id": int(fold["fold_id"]),
                    "fold_name": str(fold.get("fold_name", f"fold_{fold['fold_id']}")),
                    "tissue": str(fold.get("tissue", "")),
                    "tissue_fold_id": int(fold.get("tissue_fold_id", -1)),
                    "n_train_slides": int(len(fold["train_slides"])),
                    "n_test_slides": int(len(fold["test_slides"])),
                    "n_train_split_groups": int(len(fold.get("train_split_groups", []))),
                    "n_test_split_groups": int(len(fold.get("test_split_groups", []))),
                    "train_row_count": int(fold.get("train_row_count", 0)),
                    "test_row_count": int(fold.get("test_row_count", 0)),
                }
            )
        if fold_report_rows:
            out_path = info_dir / "slide_cv_fold_summary.csv"
            if out_path.exists():
                pd.DataFrame(fold_report_rows).to_csv(
                    info_dir / "slide_cv_fold_summary_basic.csv",
                    index=False,
                )
            else:
                pd.DataFrame(fold_report_rows).to_csv(out_path, index=False)
    if "folds" in genes_manifest:
        for idx, fold in enumerate(genes_manifest["folds"]):
            fold_label = fold.get("fold_id", None)
            if fold_label is None:
                fold_label = fold.get("fold_name", idx)
            notes.append(
                f"fold {fold_label} "
                f"trainable_genes={len(fold.get('trainable_genes', []))}"
            )
    elif "trainable_genes" in genes_manifest:
        notes.append(f"trainable_genes={len(genes_manifest['trainable_genes'])}")
    if str(cfg["protocol"]["sub_eval_configuration"]) in HEST_CV_SUB_EVAL_CONFIGS:
        if _gene_panel_mode(cfg) == "auto":
            hest_gene_panel_note = ("  Gene panels are target-informed auto panels selected per HEST tissue.")
        else:
            hest_gene_panel_note = "  G_base remains STHELAR-derived."

        notes.extend(
            [
                "HEST CV interpretation:",
                "  HEST CV uses HEST-only training.",
                "  STHELAR slides are not used for train/val/test.",
                hest_gene_panel_note,
                "  Primary aggregation is tissue macro.",
                "  Splitting is patient-aware when valid patient labels exist, otherwise slide-level.",
            ]
        )
    save_text(info_dir / "protocol_summary.txt", "\n".join(notes) + "\n")

