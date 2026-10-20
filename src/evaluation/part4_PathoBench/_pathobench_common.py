"""Shared definitions for Task D (PathoBench): task registry, exclusions, probe protocol, paths and eval_id manifest."""

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.utils.constants import get_full_path

# --------------------------------------------------------------------------- platform shims


def install_appledouble_tolerant_fs_ops() -> None:
    """
    Patch shutil.rmtree and os.remove to ignore FileNotFoundError (macOS ._* files on exFAT, removed
    together with their file during the PathoBench cache cleanup). Idempotent.
    """
    if not getattr(shutil.rmtree, "_appledouble_tolerant", False):
        _orig_rmtree = shutil.rmtree

        def _rmtree(path, ignore_errors=False, onerror=None, **kwargs):  # type: ignore[no-untyped-def]
            if ignore_errors:
                return _orig_rmtree(path, ignore_errors=True, **kwargs)

            def _handler(func, failed_path, exc_info):
                if isinstance(exc_info[1], FileNotFoundError):
                    return  # phantom AppleDouble entry / already-deleted: benign
                if onerror is not None:
                    onerror(func, failed_path, exc_info)
                else:
                    raise exc_info[1]

            return _orig_rmtree(path, onerror=_handler, **kwargs)

        _rmtree._appledouble_tolerant = True  # type: ignore[attr-defined]
        shutil.rmtree = _rmtree  # type: ignore[assignment]

    if not getattr(os.remove, "_appledouble_tolerant", False):
        _orig_remove = os.remove

        def _remove(path, *args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                return _orig_remove(path, *args, **kwargs)
            except FileNotFoundError:
                return  # phantom AppleDouble sidecar / already-deleted: benign

        _remove._appledouble_tolerant = True  # type: ignore[attr-defined]
        os.remove = _remove  # type: ignore[assignment]


# --------------------------------------------------------------------------- versions

# eval_id manifest format version
TASKD_SCHEMA_VERSION = "3"
# Task registry version (change it with TASKD_REGISTRY; part of the resume fingerprint of _2_run_pathobench.py)
TASKD_REGISTRY_VERSION = "cptac_molecular_tme_v2"

# --------------------------------------------------------------------------- encoders

# Reference encoders: frozen UNI2-h baseline t-1e-1 (he only), SEAL t9001e0 (built by seal_comparison/build_seal_he_encoder.py)
FOUNDATION_BASELINE: Tuple[int, int] = (-1, -1)
SEAL_SENTINEL: Tuple[int, int] = (9001, 0)

# Encoder tag t<n>e<c> (negative numbers allowed)
ENCODER_TAG_RE = re.compile(r"t(-?\d+)e(-?\d+)")

# --------------------------------------------------------------------------- spaces / heads

SPACES: Tuple[str, ...] = ("he", "clip")          # backbone space, CLIP space
HEADS: Tuple[str, ...] = ("linprobe", "abmil")    # deterministic probe + trained MIL
DEFAULT_METRIC = "macro-ovr-auc"                  # binary tasks -> AUROC
DEFAULT_ABMIL_SEEDS: Tuple[int, ...] = (0, 1, 2)  # ABMIL seeds (score = mean over seeds)

# Linear probe hyperparameters (ExperimentFactory.linprobe kwargs of patho-bench 0.0.1), same for all encoders
LINPROBE_DEFAULTS: Dict[str, object] = {
    "cost": 0.5,             # fixed regularization
    "balanced": True,        # balanced class weights
}
# Trident slide encoder pooling the patch features for the linear probe (mean pooling, any feature dim)
LINPROBE_POOL_MODEL = "mean-uni_v2"

ABMIL_MODEL_NAME = "abmil"           # Trident trainable attention-MIL slide encoder (ABMILSlideEncoder)
# ABMIL architecture kwargs (Trident ABMILSlideEncoder); input_feature_dim is set at runtime
ABMIL_MODEL_KWARGS: Dict[str, object] = {
    "n_heads": 1,
    "head_dim": 512,
    "dropout": 0.25,
    "gated": False,
}
# ABMIL training hyperparameters (ExperimentFactory.finetune kwargs)
ABMIL_DEFAULTS: Dict[str, object] = {
    "base_learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "layer_decay": None,
    "gradient_accumulation": 1,
    "num_epochs": 20,
    "scheduler_type": "cosine",
    "optimizer_type": "AdamW",
    "balanced": True,
    "bag_size": 2048,
    "save_which_checkpoints": "last-1",
}

# False: each slide is pooled separately, then the slide vectors of a case are averaged
COMBINE_SLIDES_PER_PATIENT = False

# --------------------------------------------------------------------------- geometry (no-upsampling)

# Patch geometry (same as Task B)
TARGET_MAG = 20
PATCH_SIZE_PX = 224
TARGET_MPP_UM_PER_PX = 0.5
FOV_UM = 112.0
# Accepted slide MPP range: finer slides are downsampled, coarser slides (would need upsampling) are rejected
MAX_UPSAMPLE_TOL_FRAC = 0.01
MPP_MAX_UM_PER_PX = TARGET_MPP_UM_PER_PX * (1.0 + MAX_UPSAMPLE_TOL_FRAC)  # 0.505
MPP_MIN_UM_PER_PX = 0.20  # admit finer scans (e.g. 40x / 0.25) for clean downsampling


# --------------------------------------------------------------------------- task registry


@dataclass(frozen=True)
class TaskDTask:
    """One PathoBench slide-level task of Task D."""

    task_id: str      # PathoBench task id, e.g. "cptac_brca--TP53_mutation"
    family: str       # one of FAMILY_ORDER
    cohort: str       # PathoBench source cohort key, e.g. "cptac_brca", "cptac_ccrcc"
    label: str        # human-readable, e.g. "TP53 mutation (breast)"
    metric: str = DEFAULT_METRIC


FAMILY_MOLECULAR_STATUS = "molecular_status"
FAMILY_MOLECULAR_STATUS_EXT = "molecular_status_ext"
# PathoBench family name ("TME Characterization")
FAMILY_TME = "tme"
FAMILY_GRADING = "grading"
FAMILY_RECEPTOR = "receptor"
FAMILY_TREATMENT_RESPONSE = "treatment_response"

# Task D tasks (PathoBench ids from available_splits.yaml); change TASKD_REGISTRY_VERSION with this list
TASKD_REGISTRY: Tuple[TaskDTask, ...] = (
    # --- Molecular status (mutation; CPTAC) ---
    TaskDTask("cptac_brca--PIK3CA_mutation", FAMILY_MOLECULAR_STATUS, "cptac_brca", "PIK3CA mutation (breast)"),
    TaskDTask("cptac_brca--TP53_mutation", FAMILY_MOLECULAR_STATUS, "cptac_brca", "TP53 mutation (breast)"),
    TaskDTask("cptac_ccrcc--BAP1_mutation", FAMILY_MOLECULAR_STATUS, "cptac_ccrcc", "BAP1 mutation (ccRCC)"),
    TaskDTask("cptac_ccrcc--PBRM1_mutation", FAMILY_MOLECULAR_STATUS, "cptac_ccrcc", "PBRM1 mutation (ccRCC)"),
    TaskDTask("cptac_coad--KRAS_mutation", FAMILY_MOLECULAR_STATUS, "cptac_coad", "KRAS mutation (colon)"),
    TaskDTask("cptac_coad--TP53_mutation", FAMILY_MOLECULAR_STATUS, "cptac_coad", "TP53 mutation (colon)"),
    TaskDTask("cptac_luad--EGFR_mutation", FAMILY_MOLECULAR_STATUS, "cptac_luad", "EGFR mutation (lung AD)"),
    TaskDTask("cptac_luad--STK11_mutation", FAMILY_MOLECULAR_STATUS, "cptac_luad", "STK11 mutation (lung AD)"),
    TaskDTask("cptac_luad--TP53_mutation", FAMILY_MOLECULAR_STATUS, "cptac_luad", "TP53 mutation (lung AD)"),
    TaskDTask("cptac_hnsc--CASP8_mutation", FAMILY_MOLECULAR_STATUS, "cptac_hnsc", "CASP8 mutation (HNSC)"),
    # --- Molecular status, extended (mutation / genomic instability; same cohorts) ---
    TaskDTask("cptac_ccrcc--VHL_mutation", FAMILY_MOLECULAR_STATUS_EXT, "cptac_ccrcc", "VHL mutation (ccRCC)"),
    TaskDTask("cptac_coad--APC_mutation", FAMILY_MOLECULAR_STATUS_EXT, "cptac_coad", "APC mutation (colon)"),
    TaskDTask("cptac_coad--PIK3CA_mutation", FAMILY_MOLECULAR_STATUS_EXT, "cptac_coad", "PIK3CA mutation (colon)"),
    TaskDTask("cptac_coad--ACVR2A_mutation", FAMILY_MOLECULAR_STATUS_EXT, "cptac_coad", "ACVR2A mutation (colon)"),
    TaskDTask("cptac_coad--ARID1A_mutation", FAMILY_MOLECULAR_STATUS_EXT, "cptac_coad", "ARID1A mutation (colon)"),
    TaskDTask("cptac_coad--SETD1B_mutation", FAMILY_MOLECULAR_STATUS_EXT, "cptac_coad", "SETD1B mutation (colon)"),
    # MSI-high: genomic instability status (not a point mutation)
    TaskDTask("cptac_coad--MSI_H", FAMILY_MOLECULAR_STATUS_EXT, "cptac_coad", "MSI-high status (colon)"),
    TaskDTask("cptac_luad--KRAS_mutation", FAMILY_MOLECULAR_STATUS_EXT, "cptac_luad", "KRAS mutation (lung AD)"),
    # --- Immune status (3-class; one per cohort) ---
    TaskDTask("cptac_brca--Immune_class", FAMILY_TME, "cptac_brca", "Immune class, 3-class (breast)"),
    TaskDTask("cptac_ccrcc--Immune_class", FAMILY_TME, "cptac_ccrcc", "Immune class, 3-class (ccRCC)"),
    TaskDTask("cptac_coad--Immune_class", FAMILY_TME, "cptac_coad", "Immune class, 3-class (colon)"),
    TaskDTask("cptac_hnsc--Immune_class", FAMILY_TME, "cptac_hnsc", "Immune class, 3-class (HNSC)"),
    TaskDTask("cptac_luad--Immune_class", FAMILY_TME, "cptac_luad", "Immune class, 3-class (lung AD)"),
    # --- Histologic grading (3-class) ---
    TaskDTask("cptac_hnsc--Histologic_Grade", FAMILY_GRADING, "cptac_hnsc", "Histologic grade, 3-class (HNSC)"),
    # --- Not run (receptor / treatment_response): uncomment, add to FAMILY_ORDER and change TASKD_REGISTRY_VERSION ---
    # --- Receptor status (ER/PR/HER2; BCNB) ---
    # TaskDTask("bcnb--er", FAMILY_RECEPTOR, "bcnb", "ER status (breast)"),
    # TaskDTask("bcnb--pr", FAMILY_RECEPTOR, "bcnb", "PR status (breast)"),
    # TaskDTask("bcnb--her2", FAMILY_RECEPTOR, "bcnb", "HER2 status (breast)"),
    # --- Treatment response ---
    # TaskDTask("nadt--response", FAMILY_TREATMENT_RESPONSE, "nadt", "Neoadjuvant response (prostate)"),
    # TaskDTask("mbc_--Recist", FAMILY_TREATMENT_RESPONSE, "mbc_", "RECIST response (metastatic breast)"),
)

# Families of the registry tasks (display order)
FAMILY_ORDER: Tuple[str, ...] = (
    FAMILY_MOLECULAR_STATUS,
    FAMILY_MOLECULAR_STATUS_EXT,
    FAMILY_TME,
    FAMILY_GRADING,
)


def validate_registry(registry: Sequence[TaskDTask] = TASKD_REGISTRY) -> None:
    """Raise on a malformed registry (duplicate ids, unknown family, bad id)."""
    ids = [t.task_id for t in registry]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"Duplicate Task D task ids in registry: {dupes}")
    for t in registry:
        if t.family not in FAMILY_ORDER:
            raise ValueError(f"Task {t.task_id!r} has unknown family {t.family!r} (expected one of {FAMILY_ORDER}).")
        if "--" not in t.task_id:
            raise ValueError(f"Task id {t.task_id!r} is not a PathoBench `{{cohort}}--{{task}}` id.")


def cohorts_in_registry(registry: Sequence[TaskDTask] = TASKD_REGISTRY) -> List[str]:
    """Sorted unique cohort keys of the registry."""
    return sorted({t.cohort for t in registry})


def tasks_for_cohort(cohort: str, registry: Sequence[TaskDTask] = TASKD_REGISTRY) -> List[TaskDTask]:
    return [t for t in registry if t.cohort == cohort]


def tasks_for_family(family: str, registry: Sequence[TaskDTask] = TASKD_REGISTRY) -> List[TaskDTask]:
    return [t for t in registry if t.family == family]


def split_task_id(task_id: str) -> Tuple[str, str]:
    """Split a PathoBench task id "<source>--<task>" into (source, task)."""
    parts = str(task_id).split("--")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Task id {task_id!r} is not a PathoBench `{{source}}--{{task}}` id.")
    return parts[0], parts[1]


# --------------------------------------------------------------------------- catalog completeness

# PathoBench tasks on the registry cohorts that are not run, with the reason
DECLARED_EXCLUDED_TASKS: Dict[str, str] = {
    "cptac_ccrcc--OS": "survival (overall survival) — Task D is classification-only; no CoxNet head is wired",
    "cptac_hnsc--OS": "survival (overall survival) — Task D is classification-only; no CoxNet head is wired",
    "cptac_luad--OS": "survival (overall survival) — Task D is classification-only; no CoxNet head is wired",
}

# PathoBench task catalog {dataset: [task, ...]}, downloaded by SplitFactory.from_hf into pathobench_assets_root()
AVAILABLE_SPLITS_FILENAME = "available_splits.yaml"


def available_splits_path() -> Path:
    """Local copy of PathoBench's `available_splits.yaml` catalog (under the splits cache)."""
    return pathobench_assets_root() / AVAILABLE_SPLITS_FILENAME


def read_available_splits(path: Optional[Path] = None) -> Optional[Dict[str, List[str]]]:
    """Parse the catalog to {dataset: [task, ...]}, or None if it is not on disk."""
    import yaml

    p = Path(path) if path is not None else available_splits_path()
    if not p.exists():
        return None
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {str(ds): [str(t) for t in (tasks or [])] for ds, tasks in raw.items()}


@dataclass(frozen=True)
class CatalogAudit:
    """Registry vs PathoBench catalog, on the registry cohorts."""

    n_datasets: int                          # datasets in the catalog
    n_tasks: int                             # tasks in the catalog
    n_tasks_on_registry_cohorts: int         # of those, on the registry cohorts
    registered: Tuple[str, ...]              # run (== the registry)
    declared_excluded: Tuple[str, ...]       # on the registry cohorts, declared not run
    undeclared: Tuple[str, ...]              # on the registry cohorts, in neither
    not_in_catalog: Tuple[str, ...]          # registry ids the catalog does not carry
    off_cohort_datasets: Tuple[str, ...]     # datasets Task D holds no slides for (informational)

    @property
    def ok(self) -> bool:
        return not self.undeclared and not self.not_in_catalog


def audit_registry_against_catalog(
    catalog: Dict[str, List[str]],
    *,
    registry: Sequence[TaskDTask] = TASKD_REGISTRY,
    declared_excluded: Optional[Dict[str, str]] = None,
) -> CatalogAudit:
    """
    Compare the registry + DECLARED_EXCLUDED_TASKS with the catalog tasks on the registry cohorts.
    Failures: undeclared (catalog task neither run nor declared) and not_in_catalog (registry id absent from the catalog).
    """
    excluded = dict(DECLARED_EXCLUDED_TASKS if declared_excluded is None else declared_excluded)
    cohorts = set(cohorts_in_registry(registry))
    registered = {t.task_id for t in registry}

    catalog_ids = {f"{ds}--{task}" for ds, tasks in catalog.items() for task in tasks}
    on_cohorts = {tid for tid in catalog_ids if tid.split("--", 1)[0] in cohorts}

    return CatalogAudit(
        n_datasets=len(catalog),
        n_tasks=len(catalog_ids),
        n_tasks_on_registry_cohorts=len(on_cohorts),
        registered=tuple(sorted(registered)),
        declared_excluded=tuple(sorted(tid for tid in on_cohorts if tid in excluded)),
        undeclared=tuple(sorted(on_cohorts - registered - set(excluded))),
        not_in_catalog=tuple(sorted(registered - catalog_ids)),
        off_cohort_datasets=tuple(sorted(ds for ds in catalog if ds not in cohorts)),
    )


def validate_declared_exclusions(registry: Sequence[TaskDTask] = TASKD_REGISTRY) -> None:
    """Raise on a malformed DECLARED_EXCLUDED_TASKS (bad id, cohort not in the registry, task also in the registry, no reason)."""
    cohorts = set(cohorts_in_registry(registry))
    registered = {t.task_id for t in registry}
    for task_id, reason in DECLARED_EXCLUDED_TASKS.items():
        source, _task = split_task_id(task_id)     # raises on a non `{source}--{task}` id
        if source not in cohorts:
            raise ValueError(
                f"DECLARED_EXCLUDED_TASKS names {task_id!r}, whose cohort {source!r} is not a "
                f"registry cohort {sorted(cohorts)}."
            )
        if task_id in registered:
            raise ValueError(f"Task {task_id!r} is both in TASKD_REGISTRY and declared excluded.")
        if not str(reason).strip():
            raise ValueError(f"Declared exclusion {task_id!r} carries no reason.")


# --------------------------------------------------------------------------- known slide exclusions

# Version of KNOWN_EXCLUDED_SLIDES: slides whose IDC pixel data does not decode (excluded from Task D)
EXCLUSION_REGISTRY_VERSION = "idc_jp2_undecodable_v1"

# Only failure_reason allowed in KNOWN_EXCLUDED_SLIDES
ACCEPTED_EXCLUSION_REASONS: Tuple[str, ...] = ("unreadable_pixels",)

KNOWN_EXCLUDED_SLIDES: Dict[str, Dict[str, str]] = {
    "cptac_brca": {
        "11BR044-75adceb5-eea5-4cb1-864e-876a8f": "unreadable_pixels",
        "11BR054-b10c7b72-1d44-4668-a637-a536b2": "unreadable_pixels",
        "18BR003-436297af-feb7-4754-96b7-c5bfb5": "unreadable_pixels",
    },
    "cptac_coad": {
        "01CO022-7a0266b4-b07c-41dc-8ddd-350e1b": "unreadable_pixels",
        "11CO044-7e3d5ae0-3dcc-4207-8dd2-59ab4b": "unreadable_pixels",
    },
}


def excluded_slides(cohort: str) -> Dict[str, str]:
    """{slide_id: failure_reason} exclusions of one cohort (empty dict if none)."""
    return dict(KNOWN_EXCLUDED_SLIDES.get(str(cohort), {}))


def validate_exclusion_registry() -> None:
    """Raise on a malformed KNOWN_EXCLUDED_SLIDES (cohort not in the registry, reason not accepted)."""
    known = set(cohorts_in_registry())
    for cohort, slides in KNOWN_EXCLUDED_SLIDES.items():
        if cohort not in known:
            raise ValueError(
                f"Exclusion registry names cohort {cohort!r}, which is not in the task registry "
                f"{sorted(known)}."
            )
        for slide_id, reason in slides.items():
            if reason not in ACCEPTED_EXCLUSION_REASONS:
                raise ValueError(
                    f"Exclusion {cohort}/{slide_id} carries reason {reason!r}; only "
                    f"{ACCEPTED_EXCLUSION_REASONS} may be allowlisted."
                )


# --------------------------------------------------------------------------- encoder helpers


def encoder_tag(training_num: int, checkpoint_num: int) -> str:
    return f"t{int(training_num)}e{int(checkpoint_num)}"


def parse_encoder_tag(spec: str) -> Tuple[int, int]:
    """Parse a `t{tn}e{cn}` tag (optionally with a trailing space suffix like `t46e13he`)."""
    m = ENCODER_TAG_RE.search(str(spec))
    if not m:
        raise ValueError(f"Cannot parse an encoder tag `t{{tn}}e{{cn}}` from {spec!r}.")
    return int(m.group(1)), int(m.group(2))


def expects_clip(training_num: int, checkpoint_num: int) -> bool:
    """True if the bundle is expected to have a clip space (checkpoint >= 0; the foundation baseline is he only)."""
    return int(checkpoint_num) >= 0


def bundle_path(training_num: int, checkpoint_num: int) -> Path:
    """Exported encoder bundle path, by the repo-wide convention (see extract_encoders.main)."""
    return get_full_path(
        "train", "trainings", "trainings_HistoST", f"training_{int(training_num)}",
        "trained_encoders", f"HEencoder_t{int(training_num)}e{int(checkpoint_num)}.pt",
    )


def encoder_features_run_id(training_num: int, checkpoint_num: int) -> str:
    """Space-independent id of a Trident feature run (he and clip are subdirectories)."""
    return f"HEencoder_t{int(training_num)}e{int(checkpoint_num)}"


def encoder_probe_run_id(training_num: int, checkpoint_num: int, space: str) -> str:
    """Id of the PathoBench probe output of one (encoder, space)."""
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}, got {space!r}.")
    return f"HEencoder_t{int(training_num)}e{int(checkpoint_num)}_{space}"


# --------------------------------------------------------------------------- paths (ROOT_PATH)


def pathobench_eval_data_root(scratch: bool = False) -> Path:
    """
    Root of the Task D dataset tree eval/eval_datasets/pathobench_eval_data/ (WSIs, Trident features, PathoBench splits).
    scratch=True: under the $SCRATCH root (only used by cohort_wsi_dir).
    """
    return get_full_path("eval", "eval_datasets", "pathobench_eval_data", scratch=scratch)


def slide_store_root(scratch: bool = False) -> Path:
    """Per-cohort WSI store under the dataset root (built by _0_download_cohorts.py)."""
    return pathobench_eval_data_root(scratch=scratch) / "slide_store"


def cohort_wsi_dir(cohort: str, scratch: bool = False) -> Path:
    """Per-cohort WSI dir slide_store/<cohort>/wsi/ (scratch=True: under the $SCRATCH root)."""
    return slide_store_root(scratch=scratch) / str(cohort) / "wsi"


def cohort_manifest_dir(cohort: str) -> Path:
    """Per-cohort manifest dir slide_store/<cohort>/manifests/ (download_manifest.tsv, slide_inventory.tsv), never on $SCRATCH."""
    return slide_store_root() / str(cohort) / "manifests"


def slide_inventory_path(cohort: str) -> Path:
    """slide_store/<cohort>/manifests/slide_inventory.tsv, written by _1_extract_features.py and read by _2_run_pathobench.py."""
    return cohort_manifest_dir(cohort) / "slide_inventory.tsv"


# PathoBench cohort key -> IDC collection_id, when they differ
IDC_COLLECTION_OVERRIDES: Dict[str, str] = {"cptac_hnsc": "cptac_hnscc"}


def idc_collection_id(cohort: str) -> str:
    """IDC collection_id hosting the CPTAC pathology WSIs of a cohort."""
    return IDC_COLLECTION_OVERRIDES.get(str(cohort), str(cohort))


def slide_wsi_path(cohort: str, slide_id: str, scratch: bool = False) -> Path:
    """
    WSI entry path of a slide: slide_store/<cohort>/wsi/<slide_id>/<slide_id>.dcm (one instance of the
    DICOM series; OpenSlide reads the whole pyramid from the .dcm files of that folder).
    """
    return cohort_wsi_dir(cohort, scratch=scratch) / str(slide_id) / f"{slide_id}.dcm"


def pathobench_assets_root() -> Path:
    """Cache of PathoBench-shipped labels + CV splits (SplitFactory.from_hf downloads)."""
    return pathobench_eval_data_root() / "pathobench_assets"


def trident_features_root(cohort: str, training_num: int, checkpoint_num: int) -> Path:
    """Per-(cohort, encoder) Trident feature folder (he / clip subdirs), built by _1_extract_features.py."""
    return (
        pathobench_eval_data_root()
        / "trident_features"
        / str(cohort)
        / encoder_features_run_id(training_num, checkpoint_num)
    )


def pool_recipe_id(
    slide_ids: Sequence[str],
    *,
    pool_model: str = LINPROBE_POOL_MODEL,
    combine_slides_per_patient: bool = COMBINE_SLIDES_PER_PATIENT,
) -> str:
    """
    Key of a pooled slide-embedding cache: sha256(pool_model, combine_slides_per_patient, sorted slide ids)[:16]
    (a different slide set gives a new cache folder).
    """
    payload = "|".join(
        [str(pool_model), str(bool(combine_slides_per_patient)), *sorted(str(s) for s in slide_ids)]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def pooled_embeddings_root(
    cohort: str, training_num: int, checkpoint_num: int, space: str, recipe_id: str
) -> Path:
    """
    Pooled slide-embedding cache of the linear probe, shared by the tasks of a cohort:
    pooled_embeddings/<cohort>/HEencoder_t<n>e<c>/<space>/<recipe_id>/
    """
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}, got {space!r}.")
    return (
        pathobench_eval_data_root()
        / "pooled_embeddings"
        / str(cohort)
        / encoder_features_run_id(training_num, checkpoint_num)
        / str(space)
        / str(recipe_id)
    )


def cohort_tiling_dir(cohort: str) -> Path:
    """Per-cohort Trident tiling dir trident_features/<cohort>/_tiling/ (masks and patch coords shared by all encoders)."""
    return pathobench_eval_data_root() / "trident_features" / str(cohort) / "_tiling"


def taskd_root() -> Path:
    """Root of the Task D results tree, eval/D_eval_pathobench/."""
    return get_full_path("eval", "D_eval_pathobench")


def eval_root(eval_id: str) -> Path:
    """Per-eval_id results folder D_eval_pathobench/<eval_id>/."""
    return taskd_root() / str(eval_id)


def arms_dir(eval_id: str) -> Path:
    """Folder of the (encoder, space, head) run records of an eval_id: <eval_id>/arms/, one JSON per run."""
    return eval_root(eval_id) / "arms"


def arm_record_path(eval_id: str, training_num: int, checkpoint_num: int, space: str, head: str) -> Path:
    """Path of one arm's record: ``{eval_id}/arms/{tag}_{space}_{head}.json``."""
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}, got {space!r}.")
    if head not in HEADS:
        raise ValueError(f"head must be one of {HEADS}, got {head!r}.")
    return arms_dir(eval_id) / f"{encoder_tag(training_num, checkpoint_num)}_{space}_{head}.json"


def probe_output_root(eval_id: str, training_num: int, checkpoint_num: int, space: str, head: str) -> Path:
    """Per-(eval_id, encoder, space, head) PathoBench probe output folder."""
    if head not in HEADS:
        raise ValueError(f"head must be one of {HEADS}, got {head!r}.")
    return eval_root(eval_id) / encoder_probe_run_id(training_num, checkpoint_num, space) / str(head)


# --------------------------------------------------------------------------- filesystem helpers


def list_h5(dir_path: Path) -> List[Path]:
    """Sorted *.h5 files in dir_path, without macOS ._* files."""
    if not dir_path.exists():
        return []
    return sorted(p for p in dir_path.glob("*.h5") if not p.name.startswith("."))


# --------------------------------------------------------------------------- manifest


def build_taskd_protocol(
    *,
    registry: Sequence[TaskDTask] = TASKD_REGISTRY,
    abmil_seeds: Sequence[int] = DEFAULT_ABMIL_SEEDS,
) -> Dict[str, object]:
    """
    Protocol block of an eval_id: registry, exclusions, metric, probe hyperparameters, ABMIL seeds, geometry.
    Encoders, spaces and heads are not part of it (recorded in arms_dir).
    """
    validate_registry(registry)
    validate_exclusion_registry()
    validate_declared_exclusions(registry)
    return {
        "registry_version": TASKD_REGISTRY_VERSION,
        "exclusion_registry_version": EXCLUSION_REGISTRY_VERSION,
        "known_excluded_slides": {c: dict(s) for c, s in sorted(KNOWN_EXCLUDED_SLIDES.items())},
        # Excluded task ids only (not the reasons)
        "declared_excluded_tasks": sorted(DECLARED_EXCLUDED_TASKS),
        "abmil_seeds": [int(s) for s in abmil_seeds],
        "metric": DEFAULT_METRIC,
        "linprobe_defaults": dict(LINPROBE_DEFAULTS),
        "linprobe_pool_model": LINPROBE_POOL_MODEL,
        "abmil_defaults": dict(ABMIL_DEFAULTS),
        "abmil_model_name": ABMIL_MODEL_NAME,
        "abmil_model_kwargs": dict(ABMIL_MODEL_KWARGS),
        "combine_slides_per_patient": COMBINE_SLIDES_PER_PATIENT,
        "geometry": {
            "nominal_magnification": TARGET_MAG,
            "patch_size_px": PATCH_SIZE_PX,
            "nominal_mpp_um_per_px": TARGET_MPP_UM_PER_PX,
            "fov_um": FOV_UM,
            "mpp_min_um_per_px": MPP_MIN_UM_PER_PX,
            "mpp_max_um_per_px": MPP_MAX_UM_PER_PX,
            "max_upsample_tol_frac": MAX_UPSAMPLE_TOL_FRAC,
        },
        "families": list(FAMILY_ORDER),
        "cohorts": cohorts_in_registry(registry),
        "tasks": [
            {"task_id": t.task_id, "family": t.family, "cohort": t.cohort, "label": t.label, "metric": t.metric}
            for t in registry
        ],
    }


def protocol_fingerprint(protocol: Dict[str, object]) -> str:
    """Hash of a protocol block (checked by _2_run_pathobench.py for an existing eval_id)."""
    return hashlib.sha256(
        json.dumps(protocol, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def build_taskd_manifest(
    eval_id: str,
    *,
    registry: Sequence[TaskDTask] = TASKD_REGISTRY,
    abmil_seeds: Sequence[int] = DEFAULT_ABMIL_SEEDS,
    notes: str = "",
) -> Dict[str, object]:
    """eval_id manifest: schema version, eval_id, protocol block and its fingerprint."""
    protocol = build_taskd_protocol(registry=registry, abmil_seeds=abmil_seeds)
    return {
        "schema_version": TASKD_SCHEMA_VERSION,
        "eval_id": str(eval_id),
        "protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "notes": notes,
    }


# --------------------------------------------------------------------------- self-check


def _selfcheck() -> None:
    validate_registry()
    # Registry: cohorts, tasks, families
    assert cohorts_in_registry() == ["cptac_brca", "cptac_ccrcc", "cptac_coad", "cptac_hnsc", "cptac_luad"]
    assert len(TASKD_REGISTRY) == 24, len(TASKD_REGISTRY)
    assert FAMILY_ORDER == (FAMILY_MOLECULAR_STATUS, FAMILY_MOLECULAR_STATUS_EXT, FAMILY_TME, FAMILY_GRADING)
    assert all(t.family in FAMILY_ORDER for t in TASKD_REGISTRY)
    assert [len(tasks_for_family(f)) for f in FAMILY_ORDER] == [10, 8, 5, 1], \
        [(f, len(tasks_for_family(f))) for f in FAMILY_ORDER]
    # molecular_status tasks
    assert [t.task_id for t in tasks_for_family(FAMILY_MOLECULAR_STATUS)] == [
        "cptac_brca--PIK3CA_mutation", "cptac_brca--TP53_mutation",
        "cptac_ccrcc--BAP1_mutation", "cptac_ccrcc--PBRM1_mutation",
        "cptac_coad--KRAS_mutation", "cptac_coad--TP53_mutation",
        "cptac_luad--EGFR_mutation", "cptac_luad--STK11_mutation", "cptac_luad--TP53_mutation",
        "cptac_hnsc--CASP8_mutation",
    ]
    # Encoder tag parsing
    assert parse_encoder_tag("t46e13") == (46, 13)
    assert parse_encoder_tag("t-1e-1he") == (-1, -1)   # trailing space suffix, negative tag
    assert parse_encoder_tag("HEencoder_t9001e0_clip") == (9001, 0)
    assert split_task_id("cptac_brca--TP53_mutation") == ("cptac_brca", "TP53_mutation")
    assert split_task_id("mbc_--Recist") == ("mbc_", "Recist")   # trailing '_' in the source
    assert encoder_features_run_id(46, 13) == "HEencoder_t46e13"
    assert encoder_probe_run_id(46, 13, "clip") == "HEencoder_t46e13_clip"
    assert expects_clip(*SEAL_SENTINEL) and not expects_clip(*FOUNDATION_BASELINE)
    assert abs(MPP_MAX_UM_PER_PX - 0.505) < 1e-9
    man = build_taskd_manifest("eval_1", abmil_seeds=[0])
    prot = man["protocol"]
    assert man["schema_version"] == TASKD_SCHEMA_VERSION and man["eval_id"] == "eval_1"
    assert prot["cohorts"] == cohorts_in_registry() and len(prot["tasks"]) == len(TASKD_REGISTRY)
    # Encoders, spaces and heads are not in the manifest
    for arm_key in ("encoders", "spaces", "heads"):
        assert arm_key not in man and arm_key not in prot, f"{arm_key} must not be in the manifest."
    # The fingerprint depends on the protocol constants
    assert man["protocol_fingerprint"] == protocol_fingerprint(prot)
    assert build_taskd_manifest("eval_1", abmil_seeds=[0])["protocol_fingerprint"] == man["protocol_fingerprint"]
    assert build_taskd_manifest("eval_1", abmil_seeds=[0, 1, 2])["protocol_fingerprint"] != man["protocol_fingerprint"], \
        "abmil_seeds must change the fingerprint."
    _saved = ABMIL_DEFAULTS["num_epochs"]
    ABMIL_DEFAULTS["num_epochs"] = int(_saved) + 1
    try:
        assert protocol_fingerprint(build_taskd_protocol(abmil_seeds=[0])) != man["protocol_fingerprint"], \
            "editing ABMIL_DEFAULTS must change the fingerprint."
    finally:
        ABMIL_DEFAULTS["num_epochs"] = _saved
    # Run records: one file per (encoder, space, head)
    assert arms_dir("eval_1").name == "arms" and arms_dir("eval_1").parent == eval_root("eval_1")
    assert arm_record_path("eval_1", 46, 13, "he", "abmil").name == "t46e13_he_abmil.json"
    assert arm_record_path("eval_1", -1, -1, "he", "linprobe").name == "t-1e-1_he_linprobe.json"
    assert arm_record_path("eval_1", 46, 13, "he", "abmil") != arm_record_path("eval_1", 46, 13, "clip", "abmil")

    # IDC collection mapping: HNSC differs (cptac_hnsc -> cptac_hnscc), others are identity.
    assert idc_collection_id("cptac_hnsc") == "cptac_hnscc"
    assert idc_collection_id("cptac_luad") == "cptac_luad"

    # Slide-exclusion registry: well-formed, cohorts known, only `unreadable_pixels` allowlisted.
    validate_exclusion_registry()
    assert set(KNOWN_EXCLUDED_SLIDES) <= set(cohorts_in_registry())
    assert len(excluded_slides("cptac_brca")) == 3 and len(excluded_slides("cptac_coad")) == 2
    assert excluded_slides("cptac_ccrcc") == {}                       # unaffected cohort -> no exclusions
    assert set(excluded_slides("cptac_brca").values()) == {"unreadable_pixels"}
    assert excluded_slides("cptac_brca") is not KNOWN_EXCLUDED_SLIDES["cptac_brca"]  # copy, not the frozen dict
    prot_x = build_taskd_manifest("eval_x")["protocol"]
    assert prot_x["exclusion_registry_version"] == EXCLUSION_REGISTRY_VERSION
    assert prot_x["known_excluded_slides"]["cptac_coad"]["11CO044-7e3d5ae0-3dcc-4207-8dd2-59ab4b"] == "unreadable_pixels"
    assert prot_x["families"] == list(FAMILY_ORDER) and len(prot_x["tasks"]) == 24

    # --- catalog audit, on a fake catalog
    validate_declared_exclusions()
    assert prot_x["declared_excluded_tasks"] == sorted(DECLARED_EXCLUDED_TASKS), \
        "declared_excluded_tasks must be in the eval_id protocol."
    fake = {ds: [] for ds in cohorts_in_registry()}
    for t in TASKD_REGISTRY:
        fake[t.cohort].append(split_task_id(t.task_id)[1])
    for tid in DECLARED_EXCLUDED_TASKS:
        src, name = split_task_id(tid)
        fake[src].append(name)
    fake["some_other_cohort"] = ["whatever"]           # dataset without slides
    audit = audit_registry_against_catalog(fake)
    assert audit.ok and not audit.undeclared and not audit.not_in_catalog
    assert audit.n_tasks_on_registry_cohorts == 24 + len(DECLARED_EXCLUDED_TASKS)
    assert audit.declared_excluded == tuple(sorted(DECLARED_EXCLUDED_TASKS))
    assert audit.off_cohort_datasets == ("some_other_cohort",), audit.off_cohort_datasets
    # Catalog task neither run nor declared
    fake_undeclared = {**fake, "cptac_coad": [*fake["cptac_coad"], "BRAF_mutation"]}
    audit_bad = audit_registry_against_catalog(fake_undeclared)
    assert audit_bad.undeclared == ("cptac_coad--BRAF_mutation",) and not audit_bad.ok
    # Registry id missing from the catalog
    fake_gone = {**fake, "cptac_hnsc": [t for t in fake["cptac_hnsc"] if t != "CASP8_mutation"]}
    audit_gone = audit_registry_against_catalog(fake_gone)
    assert audit_gone.not_in_catalog == ("cptac_hnsc--CASP8_mutation",) and not audit_gone.ok
    # Invalid exclusions: task also in the registry, cohort not in the registry
    _saved_excl = dict(DECLARED_EXCLUDED_TASKS)
    for _bad_key, _why in (("cptac_brca--TP53_mutation", "both registered and declared-excluded"),
                           ("cptac_ucec--PTEN_mutation", "an exclusion on a non-registry cohort")):
        DECLARED_EXCLUDED_TASKS.clear()
        DECLARED_EXCLUDED_TASKS.update({**_saved_excl, _bad_key: "nope"})
        try:
            validate_declared_exclusions()
            raise AssertionError(f"{_why} must be fatal")
        except ValueError:
            pass
    DECLARED_EXCLUDED_TASKS.clear()
    DECLARED_EXCLUDED_TASKS.update(_saved_excl)
    validate_declared_exclusions()

    # pool_recipe_id: independent of the slide order, dependent on the slide set and recipe
    assert pool_recipe_id(["b", "a"]) == pool_recipe_id(["a", "b"])
    assert pool_recipe_id(["a", "b"]) != pool_recipe_id(["a"])
    assert pool_recipe_id(["a"]) != pool_recipe_id(["a"], pool_model="other-pooler")
    assert pool_recipe_id(["a"]) != pool_recipe_id(["a"], combine_slides_per_patient=True)
    assert len(pool_recipe_id(["a"])) == 16
    # The pooled cache is eval_id-agnostic (dataset tree) and per (cohort, encoder, space, recipe).
    rid = pool_recipe_id(["a", "b"])
    pooled = pooled_embeddings_root("cptac_brca", 46, 13, "he", rid)
    assert pooled.name == rid and pooled.parent.name == "he"
    assert pooled.parent.parent.name == "HEencoder_t46e13" and pooled.parent.parent.parent.name == "cptac_brca"
    assert str(pooled).startswith(str(pathobench_eval_data_root()))
    assert "D_eval_pathobench" not in str(pooled), "the pooled cache is a dataset artifact, not a per-eval_id result."
    assert pooled_embeddings_root("cptac_brca", 16, 31, "he", rid) != pooled_embeddings_root("cptac_brca", 16, 31, "clip", rid)

    # path helpers: canonical DICOM WSI entry (per-slide subdir) + shared per-cohort tiling dir.
    assert slide_inventory_path("cptac_hnsc").name == "slide_inventory.tsv"
    assert slide_inventory_path("cptac_hnsc").parent == cohort_manifest_dir("cptac_hnsc")
    assert slide_wsi_path("cptac_hnsc", "slide_0").name == "slide_0.dcm"
    assert slide_wsi_path("cptac_hnsc", "slide_0").parent.name == "slide_0"      # per-slide series subdir
    assert slide_wsi_path("cptac_hnsc", "slide_0").parent.parent.name == "wsi"
    assert cohort_tiling_dir("cptac_hnsc").name == "_tiling"
    assert cohort_tiling_dir("cptac_hnsc").parent.name == "cptac_hnsc"

    # scratch=True changes only the root of the WSI dir; everything else stays on the default root
    assert slide_wsi_path("cptac_hnsc", "slide_0", scratch=True).relative_to(pathobench_eval_data_root(scratch=True)) == \
           slide_wsi_path("cptac_hnsc", "slide_0").relative_to(pathobench_eval_data_root()), \
           "scratch must change only the root of a WSI path."
    assert cohort_wsi_dir("cptac_hnsc", scratch=True) == slide_wsi_path("cptac_hnsc", "s", scratch=True).parent.parent
    assert cohort_manifest_dir("cptac_hnsc").name == "manifests"
    assert cohort_manifest_dir("cptac_hnsc").parent.name == "cptac_hnsc"
    # Only where the scratch root differs
    if pathobench_eval_data_root(scratch=True) != pathobench_eval_data_root():
        assert cohort_wsi_dir("cptac_hnsc", scratch=True) != cohort_wsi_dir("cptac_hnsc")
        for stays_put in (cohort_manifest_dir("cptac_hnsc"), pathobench_assets_root(),
                          cohort_tiling_dir("cptac_hnsc"), trident_features_root("cptac_hnsc", 16, 31),
                          slide_inventory_path("cptac_hnsc"),
                          pooled_embeddings_root("cptac_hnsc", 16, 31, "he", "deadbeefdeadbeef")):
            assert not str(stays_put).startswith(str(pathobench_eval_data_root(scratch=True))), \
                f"{stays_put} must not be on $SCRATCH (only the WSI dir)."

    # list_h5 excludes ._* files and sorts
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for name in ("slide_b.h5", "slide_a.h5", "._slide_a.h5", "notes.txt"):
            (d / name).touch()
        assert [p.name for p in list_h5(d)] == ["slide_a.h5", "slide_b.h5"]
        assert list_h5(d / "does_not_exist") == []

    # os.remove ignores FileNotFoundError only; install is idempotent
    install_appledouble_tolerant_fs_ops()
    install_appledouble_tolerant_fs_ops()
    assert getattr(shutil.rmtree, "_appledouble_tolerant", False) and getattr(os.remove, "_appledouble_tolerant", False)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        real = d / "X.lock"
        real.touch()
        os.remove(real)                        # real removal still works
        assert not real.exists()
        os.remove(d / "._X.lock")              # phantom AppleDouble sidecar -> FNF swallowed, no raise
        try:
            os.remove(d)                       # a directory -> IsADirectoryError (not FNF) -> must raise
            still_raises = False
        except FileNotFoundError:
            still_raises = False
        except OSError:
            still_raises = True
        assert still_raises, "os.remove must still raise non-FileNotFoundError errors (only FNF is swallowed)."

    n_excluded = sum(len(s) for s in KNOWN_EXCLUDED_SLIDES.values())
    print(f"_pathobench_common self-check OK: {len(TASKD_REGISTRY)} tasks in "
          f"{len(FAMILY_ORDER)} families, {len(cohorts_in_registry())} cohorts, "
          f"registry={TASKD_REGISTRY_VERSION}, {len(DECLARED_EXCLUDED_TASKS)} declared-excluded tasks, "
          f"{n_excluded} excluded slides ({EXCLUSION_REGISTRY_VERSION}).")


# --------------------------------------------------------------------------- catalog audit CLI


def _print_audit() -> int:
    """Audit the registry against the PathoBench catalog on disk. Returns the exit code (non-zero if missing or incomplete)."""
    catalog = read_available_splits()
    if catalog is None:
        print(f"FAIL: no catalog at {available_splits_path()}.\n"
              "  It is written by `SplitFactory.from_hf` on the first split download, so run `_0` "
              "(or the split warm-up) once from a machine with internet, then re-audit.")
        return 2

    audit = audit_registry_against_catalog(catalog)
    print(f"PathoBench catalog: {audit.n_tasks} tasks across {audit.n_datasets} datasets "
          f"({available_splits_path()})")
    print(f"Task D cohorts ({len(cohorts_in_registry())}): {', '.join(cohorts_in_registry())}")
    print(f"  tasks available on those cohorts : {audit.n_tasks_on_registry_cohorts}")
    print(f"  registered (run)                 : {len(audit.registered)}")
    print(f"  declared excluded                : {len(audit.declared_excluded)}")
    for tid in audit.declared_excluded:
        print(f"      - {tid}: {DECLARED_EXCLUDED_TASKS.get(tid, '?')}")
    print(f"  UNDECLARED                       : {len(audit.undeclared) or '(none)'}")
    for tid in audit.undeclared:
        print(f"      ! {tid}")
    print(f"  registry ids absent from catalog : {len(audit.not_in_catalog) or '(none)'}")
    for tid in audit.not_in_catalog:
        print(f"      ! {tid}")
    print(f"  datasets Task D holds no slides for ({len(audit.off_cohort_datasets)}, informational): "
          f"{', '.join(audit.off_cohort_datasets)}")

    if audit.ok:
        print("OK: the registry is an explicitly declared subset of the catalog.")
        return 0
    print("FAIL: the registry is not an explicitly declared subset of the catalog. Either add the "
          "task(s) to TASKD_REGISTRY or record them in DECLARED_EXCLUDED_TASKS with a reason (and "
          "bump TASKD_REGISTRY_VERSION either way).")
    return 1


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Task D shared definitions: self-check / catalog audit.")
    parser.add_argument("--selfcheck", action="store_true",
                        help="Run the synthetic, ROOT_PATH-free self-check (the default when no flag is given).")
    parser.add_argument("--audit", action="store_true",
                        help="Check TASKD_REGISTRY + DECLARED_EXCLUDED_TASKS against the PathoBench "
                             "available_splits.yaml catalog (non-zero exit on any undeclared or missing task).")
    _args = parser.parse_args()
    if _args.audit:
        sys.exit(_print_audit())
    _selfcheck()
