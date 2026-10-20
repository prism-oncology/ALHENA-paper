"""Shared config, path and IO helpers for Task A.3 (re-exports the Task A.1 helpers)."""

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

import numpy as np
import yaml

# --------------------------------------------------------------------------------------
# sys.path: repo root (src.*), task_A1 dir (src_A1.*) and task_A3 dir (src_A3.*)
# --------------------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_TASK_A3_DIR = _THIS.parents[1]                       # .../task_A3_celltype_predict
_PART1_DIR = _TASK_A3_DIR.parent                      # .../part1_ST
_TASK_A1_DIR = _PART1_DIR / "task_A1_gene_predict"
_REPO_ROOT = _TASK_A3_DIR.parents[3]                  # .../HistoST

# Check the repo-root depth
assert (_REPO_ROOT / "src" / "utils" / "constants.py").exists(), (
    f"Repo-root discovery failed: expected 'src/utils/constants.py' under {_REPO_ROOT}. "
    "Check the src_A3/common.py directory depth (_TASK_A3_DIR.parents[3])."
)


def ensure_imports() -> None:
    """Idempotently place the repo root, the Task A.1 dir and the Task A.3 dir on sys.path."""
    for p in (str(_REPO_ROOT), str(_TASK_A1_DIR), str(_TASK_A3_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


ensure_imports()

# --------------------------------------------------------------------------------------
# Task A.1 helpers re-exported for Task A.3 (_infer_patch_overlap_px gives cv.safe_border_px)
# --------------------------------------------------------------------------------------
from src_A1.common import (  # noqa: E402
    mkdir,
    save_json,
    load_json,
    save_yaml,
    load_yaml,
    save_text,
    save_csv,
    write_jsonl,
    append_jsonl,
    sha256_bytes,
    sha256_file,
    ndarray_sha256,
    manifest_for_file,
    FileManifest,
    protocol_identity_payload,
    protocol_identity_sha256,
    assert_eval_config_matches_protocol_manifest,
    set_global_seeds,
    utc_now,
    maybe_git_commit,
    ensure_project_import,
    get_full_path,
    int_cell_id,
    read_metadata,
    SlideRecord,
    clean_patient_label,
    build_split_group_id,
    normalize_modelgroup_name,
    infer_encoder_modelgroup,
    build_logger,
    dump_environment_report,
    resolve_eval_dataset_root,
    resolve_he_embeddings_dir,
    resolve_encoder_path,
    resolve_training_dataset_dir,
    resolve_metadata_csv,
    rmtree_appledouble_safe,
    VALID_ENCODER_SPACES,
    EncoderSpec,
    _infer_patch_overlap_px,
)

# --------------------------------------------------------------------------------------
# Wall-clock instrumentation
# --------------------------------------------------------------------------------------


class PhaseTimer:
    """Accumulate wall-clock seconds under named phases (a nested phase is also counted in its parent)."""

    def __init__(self) -> None:
        self._seconds: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Time the enclosed block under ``name`` (accumulating across repeated entries)."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def add(self, name: str, seconds: float, *, count: int = 1) -> None:
        key = str(name)
        self._seconds[key] = self._seconds.get(key, 0.0) + float(seconds)
        self._counts[key] = self._counts.get(key, 0) + int(count)

    def merge(self, other: "PhaseTimer", *, prefix: str = "") -> None:
        """Fold another timer's phases into this one under an optional key prefix."""
        for key, seconds in other._seconds.items():
            self.add(f"{prefix}{key}", seconds, count=other._counts[key])

    def seconds(self, name: str) -> float:
        return float(self._seconds.get(str(name), 0.0))

    def as_dict(self, *, suffix: str = "_seconds", ndigits: int = 3) -> Dict[str, float]:
        """Phase seconds as a flat, CSV/JSON-friendly ``{f'{phase}{suffix}': value}`` map."""
        return {f"{k}{suffix}": round(v, ndigits) for k, v in sorted(self._seconds.items())}

    def counts(self) -> Dict[str, int]:
        """How many times each phase was entered (e.g. the realized inner-CV fit count)."""
        return {k: int(v) for k, v in sorted(self._counts.items())}

# --------------------------------------------------------------------------------------
# Task A.3 constants
# --------------------------------------------------------------------------------------
A3_TASK_ROOT_PARTS = ("eval", "A-3_eval_celltype_predictions")
A3_PACKAGE = "histost_a3"

# Default protocol.protocol_seed (seeds the frozen rows, splits and class axis)
A3_DEFAULT_PROTOCOL_SEED = 20260319

VALID_LEVELS = {"cell"}
VALID_A3_EVAL_CONFIGS = {"STHELAR_TRAIN", "STHELAR_TEST"}
VALID_A3_SUB_EVAL_CONFIGS = {
    "STHELAR_TRAIN_CV",
    "STHELAR_TEST_ONE_SHOT",
    "STHELAR_TEST_ADAPTED",
}
# Task A.1 sub-eval configs not implemented for Task A.3 (NotImplementedError)
DEFERRED_SUB_EVAL_CONFIGS = {
    "STHELAR_TRAIN_COORDS",
    "HEST_CV_CLOSE",
    "HEST_CV_MAGNIFICATION",
    "HEST_CV_SPECIES",
    "HEST_ORGANS",
}

VALID_A3_HEAD_TYPES = {"linear_softmax", "mlp_softmax"}
VALID_A3_LABEL_GRANULARITY = {"fine", "coarse"}
VALID_INPUT_PREPROCESSING_STRATEGIES = {"identity", "row_l2"}

# H&E embedding spaces of `_3_run_eval.py --space` (coords / scconcept are not H&E spaces)
VALID_A3_SPACES = {"he", "he_l2", "clip"}

# Allowed values of protocol.metrics.primary (both are always computed): slide_macro_f1 = mean over
# slides of the per-slide macro-F1, macro_f1 = macro-F1 over all pooled OOF cells
VALID_A3_PRIMARY_METRICS = {"slide_macro_f1", "macro_f1"}


# --------------------------------------------------------------------------------------
# Within-slide (STHELAR_TEST_ADAPTED) stripe rotations: each stripe is the test stripe of one
# rotation, the two others are its train stripes. Fold name: f"{slide_id}_fold_{fold_idx}".
# --------------------------------------------------------------------------------------
_ADAPTED_ROLE_ROTATIONS = (
    {0: "train", 1: "train", 2: "test"},
    {1: "train", 2: "train", 0: "test"},
    {0: "train", 2: "train", 1: "test"},
)
N_ADAPTED_ROTATIONS = len(_ADAPTED_ROLE_ROTATIONS)


def adapted_train_stripes(fold_idx: int) -> List[int]:
    """The two train stripe ids of within-slide rotation fold_idx (sorted)."""
    roles = _ADAPTED_ROLE_ROTATIONS[int(fold_idx)]
    return sorted(k for k, v in roles.items() if v == "train")


def adapted_test_stripe(fold_idx: int) -> int:
    """The single test stripe id of within-slide rotation ``fold_idx``."""
    roles = _ADAPTED_ROLE_ROTATIONS[int(fold_idx)]
    return next(k for k, v in roles.items() if v == "test")


def resolve_a3_eval_task_root() -> Path:
    """Task A.3 output root: <ROOT_HistoST> / eval / A-3_eval_celltype_predictions /"""
    return Path(get_full_path(*A3_TASK_ROOT_PARTS, as_str=True))


# --------------------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------------------
def head_type_from_config(cfg: Mapping[str, Any]) -> str:
    """protocol.head.type (lowercase, stripped); defaults to linear_softmax."""
    protocol = cfg.get("protocol", cfg)
    head_cfg = protocol.get("head", {})
    return str(head_cfg.get("type", "linear_softmax")).strip().lower()


# --------------------------------------------------------------------------------------
# Commented-YAML renderer helpers
# --------------------------------------------------------------------------------------
def _jsonify(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _indent_block(text: str, n_spaces: int) -> str:
    prefix = " " * int(n_spaces)
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def _yaml_block_for_key(key: str, value: Any, indent: int = 0) -> str:
    block = yaml.safe_dump(
        {key: _jsonify(value)},
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).rstrip()
    return _indent_block(block, indent) if indent > 0 else block


def _comment_lines(text: str, indent: int = 0) -> List[str]:
    prefix = " " * int(indent)
    lines: List[str] = []
    for raw in str(text).strip().splitlines():
        raw = raw.strip()
        if raw:
            lines.append(f"{prefix}# {raw}")
        else:
            lines.append(f"{prefix}#")
    return lines


def render_a3_eval_config_with_comments(cfg: Mapping[str, Any]) -> str:
    """Render eval_config.yaml with explanatory comments (same YAML content as cfg)."""
    identity = dict(cfg.get("identity", {}))
    protocol = dict(cfg.get("protocol", {}))

    lines: List[str] = []
    lines.extend(
        [
            "# HistoST Task A.3 evaluation config (cell-type linear-probe classification)",
            "#",
            "# This file can be edited before running `_2_fill_eval_settings.py`.",
            "# Do not modify it afterwards: `_3_run_eval.py` checks that it matches",
            "# settings/protocol_manifest.json.",
            "",
            "# ----------------------------",
            "# Eval identity",
            "# ----------------------------",
        ]
    )
    lines.append(_yaml_block_for_key("identity", identity))
    lines.append("")
    lines.extend(
        [
            "# ----------------------------",
            "# Protocol",
            "# ----------------------------",
            "protocol:",
        ]
    )

    def add_protocol_key(key: str, comment: str) -> None:
        if key not in protocol:
            return
        lines.extend(_comment_lines(comment, indent=2))
        lines.append(_yaml_block_for_key(key, protocol[key], indent=2))
        lines.append("")

    add_protocol_key(
        "eval_configuration",
        "High-level benchmark family. This fixes which slide pool is used:\n"
        "- STHELAR_TRAIN: same-slide-pool benchmark with held-out slides/folds\n"
        "- STHELAR_TEST: unseen STHELAR slides",
    )
    add_protocol_key(
        "sub_eval_configuration",
        "Specific protocol inside the family.\n"
        "The main switch controlling CV / one-shot / within-slide adaptation:\n"
        "- STHELAR_TRAIN_CV: slide-level cross-validation on STHELAR_train\n"
        "- STHELAR_TEST_ONE_SHOT: train on STHELAR_train, test on unseen STHELAR_test\n"
        "- STHELAR_TEST_ADAPTED: within-slide adaptation on STHELAR_test",
    )
    add_protocol_key(
        "level",
        "Supervision granularity. Task A.3 is cell-level only (level='cell').",
    )
    add_protocol_key(
        "modelgroup",
        "Geometric preprocessing group (resize/crop) of the encoders.\n"
        "All encoders evaluated under this eval_id must belong to this modelgroup.",
    )
    add_protocol_key(
        "ct_cat_id",
        "Cell-type category (labelling granularity) whose canonical annotation dicts define\n"
        "the class universe (data/annots/annot_dicts_{ct_cat_id}: cat2idx / coarse_ct_mapping).",
    )
    add_protocol_key(
        "label_granularity",
        "Class-axis resolution:\n"
        "- fine: canonical cell types\n"
        "- coarse: canonical types aggregated via coarse_ct_mapping_{coarse_mapping_version}.json\n"
        "fine and coarse are separate eval_ids.",
    )
    add_protocol_key(
        "coarse_mapping_version",
        "Which versioned coarse mapping defines the class axis:\n"
        "data/annots/annot_dicts_{ct_cat_id}/coarse_ct_mapping_{version}.json.\n"
        "- v1: 5 classes (T_NK, B_Plasma, Myeloid, Stromal_Vascular, Epithelial_Melanocyte)\n"
        "- v2: 3 classes (Immune, Stromal_Vascular, Epithelial_Melanocyte)\n"
        "Required when label_granularity is 'coarse'; an unversioned coarse_ct_mapping.json is\n"
        "not accepted. Each version is a separate benchmark (different admitted slides and\n"
        "scored cells).\n"
        "Under a 'fine' eval this key is optional and selects the mapping of the descriptive\n"
        "coarse-by-aggregation mirror; leave it null to emit no mirror.",
    )
    add_protocol_key(
        "protocol_seed",
        "Protocol randomness only. Fixes splits, the frozen class vocabulary and all other\n"
        "persisted benchmark definitions. It must not vary across encoders or probe seeds.",
    )
    add_protocol_key(
        "probe_seeds",
        "Optimization randomness only. The linear_softmax probe (lbfgs) is deterministic, so\n"
        "several seeds give identical numbers; the default is a single seed [11].",
    )
    add_protocol_key(
        "canonical_seed",
        "Canonical probe seed.",
    )
    add_protocol_key(
        "head",
        "Linear-probe classification head.\n"
        "- type: linear_softmax (sklearn multinomial LogisticRegression) | mlp_softmax (not implemented)\n"
        "- Canonical benchmark head: linear_softmax.",
    )
    add_protocol_key(
        "probe",
        "sklearn LogisticRegression hyper-parameters:\n"
        "- C: fixed inverse-L2-regularization strength (fallback; see inner_cv_select)\n"
        "- class_weight='balanced', solver='lbfgs', multi_class='multinomial'\n"
        "- tol: convergence tolerance of the final fit (1e-4 = sklearn default)\n"
        "- inner_cv_select: when true (default), C is chosen per (encoder, fold) by\n"
        "  leave-one-train-block-out CV over C_grid={c_min,c_max,n_points}, scored by\n"
        "  inner_cv_score (macro_f1). The selection fits use at most inner_cv_max_train_cells\n"
        "  cells (subsample); the final fit uses the full fold. selected_C and effective_dof\n"
        "  are reported per fold.\n"
        "- C_grid_overrides: optional per-space grid override (wider grid for the 2-d `coords` arm)\n"
        "- inner_cv_max_groups: max number of slide-disjoint blocks of train groups in the C\n"
        "  search (n_blocks x len(C_grid) fits; >= n_groups gives leave-one-group-out)\n"
        "- inner_cv_tol: convergence tolerance of the C selection fits",
    )
    add_protocol_key(
        "input_preprocessing",
        "Per-space geometry applied before standardization, protocol-fixed:\n"
        "- he: identity\n"
        "- he_l2: row-wise L2 normalization\n"
        "- clip: row-wise L2 normalization\n"
        "A StandardScaler (fit on train rows only, constant columns dropped) is always\n"
        "applied after this geometry.",
    )
    add_protocol_key(
        "slide_admission",
        "Freeze-time slide admission, applied between the row universe and the class vocabulary.\n"
        "A slide below either floor is excluded (no rows, no class support, no fold, no metrics).\n"
        "Admission depends on the slide only (encoder-independent).\n"
        "- min_scorable_cells_per_slide: minimum number of scorable cells of a slide\n"
        "- min_classes_per_slide: minimum number of classes present on a slide\n"
        "Both counted on the candidate class axis (before the train-pool support filter).\n"
        "0 disables a floor. min_classes_per_slide is relative to the class axis: '3' is\n"
        "'3 of 5' under coarse v1 and 'all 3' under v2.",
    )
    add_protocol_key(
        "fold_coverage",
        "Freeze-time fold-coverage gate, applied after the splits and the class vocabulary.\n"
        "Every frozen class must have >= min_train_cells_per_class_per_fold train cells in every\n"
        "fold. A class without train cells in a fold is zero-filled onto the global class axis\n"
        "at predict time.\n"
        "- min_train_cells_per_class_per_fold: 0 disables the gate.\n"
        "- declared_exceptions: {fold_name: reason}; violations in these folds are logged as\n"
        "  warnings instead of raising an error (the reason is mandatory).\n"
        "If the gate passes, macro_f1_restricted == macro_f1.",
    )
    add_protocol_key(
        "class_selection",
        "Rules defining trainable classes from the frozen candidate class universe.\n"
        "Cross-slide regimes (STHELAR_TRAIN_CV / STHELAR_TEST_ONE_SHOT): a class is trainable iff\n"
        "it has >= min_train_cells_per_class cells across >= min_train_slides_per_class train\n"
        "slides.\n"
        "Within-slide regime (STHELAR_TEST_ADAPTED): folds train and test on spatial stripes of\n"
        "the same slide; a class is trainable iff it has >= adapted_min_cells_per_slide cells\n"
        "within each of >= adapted_min_slides slides.\n"
        "In both cases, at least min_trainable_classes trainable classes are required.",
    )
    add_protocol_key(
        "cell_filter",
        "Cell-universe filters.\n"
        "- min_selected_alpha: optional main-assignment alpha gate (null = no gate, i.e. the\n"
        "  same cell universe as Task A.1).\n"
        "The unscorable labels are not configurable: module constant\n"
        "src_A3/labels.py::UNSCORABLE_LABELS = {'Unknown', 'nan', 'NaN', 'None', ''}.",
    )
    add_protocol_key(
        "scconcept_training_id",
        "scConcept teacher-ceiling arm: the scConcept training_id whose per-cell cls_emb are the\n"
        "ceiling arm's features (run via `_3_run_eval.py --arm scconcept_ceiling`; folder\n"
        "SCCONCEPT_CEILING). null = the ceiling arm is not configured for this eval_id (the\n"
        "coordinate floor + the H&E encoder arms still run). Set it only when the scConcept per-cell\n"
        "embeddings exist for every eval slide.",
    )
    add_protocol_key(
        "cv",
        "Fold-based-regime parameters (STHELAR_TRAIN_CV / STHELAR_TEST_ADAPTED):\n"
        "- n_outer_folds: number of CV folds (ADAPTED uses the 3 within-slide stripe rotations)\n"
        "- safe_border_px reflects patch overlap, to prevent leakage in within-slide splits.\n"
        "There is no validation split in any regime; C is selected per (encoder, fold) by inner\n"
        "CV on the train groups.",
    )
    add_protocol_key(
        "metrics",
        "Benchmark metric definitions.\n"
        "- primary: primary statistic of this benchmark (part of the protocol identity), one of\n"
        "  {'slide_macro_f1', 'macro_f1'}. 'slide_macro_f1' (default) is the per-slide macro-F1\n"
        "  averaged over slides; 'macro_f1' is the macro-F1 over all pooled out-of-fold cells.\n"
        "  Both are always reported.\n"
        "- macro_min_class_support_per_slide: minimum per-slide cell support for a class to enter\n"
        "  that slide's macro average (no cell is dropped; the pooled metrics and the\n"
        "  per-(slide, class) F1 rows are not affected). The per-slide class counts are recorded\n"
        "  as n_macro_classes / n_classes_below_macro_floor. 0 disables it.",
    )
    add_protocol_key(
        "runtime",
        "Execution settings. wandb_mode is recorded in the encoder manifest.",
    )

    # Remove trailing blank lines for cleanliness.
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def save_a3_eval_config_with_comments(path: Path, cfg: Mapping[str, Any]) -> None:
    mkdir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_a3_eval_config_with_comments(cfg))


def read_a3_eval_config(eval_dir: Path) -> Dict[str, Any]:
    return load_yaml(Path(eval_dir) / "settings" / "eval_config.yaml")


# --------------------------------------------------------------------------------------
# Config factory + validator.
# --------------------------------------------------------------------------------------
def default_a3_eval_config(
    *,
    eval_id: str,
    dataset_id: str,
    training_dataset_id: str,
    ct_cat_id: str,
    eval_configuration: str,
    sub_eval_configuration: str,
    modelgroup: str,
    label_granularity: str,
    coarse_mapping_version: Optional[str] = None,
    head_type: str = "linear_softmax",
    probe_C: float = 1.0,
    probe_max_iter: int = 2000,
    probe_c_min: float = 1e-6,
    probe_c_max: float = 1e0,
    probe_c_points: int = 7,
    min_selected_alpha: Optional[float] = None,
    scconcept_training_id: Optional[str] = None,
    protocol_seed: Optional[int] = A3_DEFAULT_PROTOCOL_SEED,
) -> Dict[str, Any]:
    """
    Build the frozen A.3 eval config {identity, protocol} (both hashed by protocol_identity_sha256) and validate it.
    probe_c_min / probe_c_max / probe_c_points set protocol.probe.C_grid (inner-CV search grid).
    """
    if protocol_seed is None:
        protocol_seed = A3_DEFAULT_PROTOCOL_SEED

    safe_border_px = _infer_patch_overlap_px(dataset_id, training_dataset_id)

    min_selected_alpha_clean = (
        None if min_selected_alpha is None else float(min_selected_alpha)
    )

    cfg: Dict[str, Any] = {
        "identity": {
            "eval_id": eval_id,
            "dataset_id": dataset_id,
            "training_dataset_id": training_dataset_id,
            "ct_cat_id": ct_cat_id,
            "created_utc": utc_now(),
            "package": A3_PACKAGE,
        },
        "protocol": {
            "eval_configuration": eval_configuration,
            "sub_eval_configuration": sub_eval_configuration,
            "level": "cell",
            "modelgroup": modelgroup,
            "ct_cat_id": ct_cat_id,
            "label_granularity": label_granularity,
            # Coarse mapping version (required for coarse; optional for fine: coarse-by-aggregation metrics)
            "coarse_mapping_version": (
                str(coarse_mapping_version).strip() if coarse_mapping_version else None
            ),
            "protocol_seed": int(protocol_seed),
            # Single probe seed by default (the linear_softmax lbfgs fit is deterministic)
            "probe_seeds": [11],
            "canonical_seed": 11,
            "head": {
                "type": str(head_type),
                # Used only by mlp_softmax (not implemented)
                "mlp_hidden_dim": 512,
                "mlp_dropout": 0.1,
            },
            "probe": {
                # Fixed C, used only if inner_cv_select is False or if there are < 2 train groups
                "C": float(probe_C),
                # Log-spaced grid of C, selected per (encoder, fold) by leave-one-train-block-out CV
                "C_grid": {
                    "c_min": float(probe_c_min),
                    "c_max": float(probe_c_max),
                    "n_points": int(probe_c_points),
                },
                # Per-space C grid overrides (coords: 2-d features, wider grid)
                "C_grid_overrides": {
                    "coords": {"c_min": 1e-6, "c_max": 1e3, "n_points": 10},
                },
                "max_iter": int(probe_max_iter),
                # Convergence tolerance of the final fit (sklearn default)
                "tol": 1e-4,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "multi_class": "multinomial",
                "inner_cv_select": True,
                "inner_cv_score": "macro_f1",
                # Convergence tolerance of the C selection fits
                "inner_cv_tol": 1e-3,
                # Max number of slide-disjoint blocks of train groups in the C search
                # (>= n_groups gives leave-one-group-out)
                "inner_cv_max_groups": 5,
                # Max train cells (random subsample) of the C selection fits; the final fit uses the full fold
                "inner_cv_max_train_cells": 50000,
            },
            "input_preprocessing": {
                "he": "identity",
                "he_l2": "row_l2",
                "clip": "row_l2",
            },
            # Slide admission (freeze time): a slide below either floor is excluded from the eval.
            # Counted on the candidate class axis; 0 disables a floor.
            "slide_admission": {
                "min_scorable_cells_per_slide": 1000,
                "min_classes_per_slide": 3,
            },
            # Fold-coverage gate (freeze time): every class needs >= min_train_cells_per_class_per_fold
            # train cells in every fold (0 disables; declared_exceptions: {fold_name: reason})
            "fold_coverage": {
                "min_train_cells_per_class_per_fold": 50,
                "declared_exceptions": {},
            },
            "class_selection": {
                "min_train_cells_per_class": 200,
                "min_train_slides_per_class": 3,
                "min_trainable_classes": 2,
                # STHELAR_TEST_ADAPTED only: a class is trainable if it has >= adapted_min_cells_per_slide
                # cells in each of >= adapted_min_slides slides
                "adapted_min_cells_per_slide": 200,
                "adapted_min_slides": 1,
            },
            "cell_filter": {
                # Unscorable labels: src_A3/labels.py::UNSCORABLE_LABELS (not configurable)
                "min_selected_alpha": min_selected_alpha_clean,
            },
            # scConcept training_id of the scconcept_ceiling arm (null = arm not configured)
            "scconcept_training_id": (str(scconcept_training_id) if scconcept_training_id else None),
            "cv": {
                "n_outer_folds": 5,
                "safe_border_px": int(safe_border_px),
            },
            "metrics": {
                # Primary statistic (slide_macro_f1 and macro_f1 are both reported)
                "primary": "slide_macro_f1",
                "auroc_min_present_classes": 2,
                # Minimum per-slide cell support for a class to enter that slide's macro average
                # (no cell is dropped; 0 disables)
                "macro_min_class_support_per_slide": 50,
                "group_summaries": {"min_cells_per_slide": 20},
            },
            # wandb_mode is recorded in the encoder manifest
            "runtime": {
                "wandb_mode": "offline",
            },
        },
    }

    # STHELAR_TEST_ADAPTED: one fold per stripe rotation
    if sub_eval_configuration == "STHELAR_TEST_ADAPTED":
        cfg["protocol"]["cv"]["n_outer_folds"] = N_ADAPTED_ROTATIONS

    # The cv block is only used by the fold-based regimes
    use_cv = sub_eval_configuration in {"STHELAR_TRAIN_CV", "STHELAR_TEST_ADAPTED"}
    if not use_cv:
        cfg["protocol"].pop("cv", None)

    validate_a3_protocol_identity(cfg)
    return cfg


def assert_setup_granularity_allowed(
    *,
    label_granularity: str,
    coarse_mapping_version: Optional[str],
    allow_retired_fine_axis: bool = False,
) -> None:
    """
    Class-axis checks of _1_setup.py (raises SystemExit): 'fine' requires allow_retired_fine_axis,
    'coarse' requires coarse_mapping_version.
    """
    granularity = str(label_granularity).strip().lower()
    if granularity == "fine" and not allow_retired_fine_axis:
        raise SystemExit(
            "--label_granularity fine is refused for Task A.3 (use '--label_granularity coarse "
            "--coarse_mapping_version v1'). Pass --allow_retired_fine_axis to run it anyway."
        )
    if granularity == "coarse" and not (coarse_mapping_version and str(coarse_mapping_version).strip()):
        raise SystemExit(
            "--coarse_mapping_version is required when --label_granularity coarse "
            "(e.g. 'v1' for 5 classes, 'v2' for 3 classes)."
        )


def validate_a3_protocol_identity(cfg: Mapping[str, Any]) -> None:
    """Validate the A.3 protocol fields (ValueError; NotImplementedError for the COORDS / HEST_* sub-eval configs)."""
    protocol = cfg["protocol"]
    level = str(protocol["level"])
    eval_configuration = str(protocol["eval_configuration"])
    sub_eval_configuration = str(protocol["sub_eval_configuration"])
    label_granularity = str(protocol.get("label_granularity", ""))
    ct_cat_id = str(protocol.get("ct_cat_id", "")).strip()
    head_type = head_type_from_config(cfg)

    if level not in VALID_LEVELS:
        raise ValueError(
            f"Invalid level='{level}'. Task A.3 supports only {sorted(VALID_LEVELS)}."
        )
    if eval_configuration not in VALID_A3_EVAL_CONFIGS:
        raise ValueError(
            f"Invalid eval_configuration='{eval_configuration}'. "
            f"Expected one of {sorted(VALID_A3_EVAL_CONFIGS)}."
        )
    if sub_eval_configuration in DEFERRED_SUB_EVAL_CONFIGS:
        raise NotImplementedError(
            f"sub_eval_configuration='{sub_eval_configuration}' is not implemented for Task A.3. "
            "Supported sub-eval configs are "
            f"{sorted(VALID_A3_SUB_EVAL_CONFIGS)}."
        )
    if sub_eval_configuration not in VALID_A3_SUB_EVAL_CONFIGS:
        raise ValueError(
            f"Invalid sub_eval_configuration='{sub_eval_configuration}'. "
            f"Expected one of {sorted(VALID_A3_SUB_EVAL_CONFIGS)}."
        )
    if eval_configuration not in sub_eval_configuration:
        raise ValueError(
            f"Mismatch between eval_configuration='{eval_configuration}' and "
            f"sub_eval_configuration='{sub_eval_configuration}' (name-prefix rule)."
        )

    if head_type not in VALID_A3_HEAD_TYPES:
        raise ValueError(
            f"Invalid protocol.head.type={head_type!r}. "
            f"Expected one of {sorted(VALID_A3_HEAD_TYPES)}."
        )

    if label_granularity not in VALID_A3_LABEL_GRANULARITY:
        raise ValueError(
            f"Invalid protocol.label_granularity={label_granularity!r}. "
            f"Expected one of {sorted(VALID_A3_LABEL_GRANULARITY)}."
        )

    # Lazy import (labels.py imports common.py)
    from src_A3.labels import normalize_coarse_mapping_version  # noqa: PLC0415

    coarse_mapping_version = protocol.get("coarse_mapping_version", None)
    if label_granularity == "coarse":
        if coarse_mapping_version is None or not str(coarse_mapping_version).strip():
            raise ValueError(
                "protocol.coarse_mapping_version is required when label_granularity='coarse' "
                "(e.g. 'v1' for 5 classes, 'v2' for 3 classes)."
            )
        normalize_coarse_mapping_version(coarse_mapping_version)
    elif coarse_mapping_version is not None and str(coarse_mapping_version).strip():
        # Optional under fine (coarse-by-aggregation metrics), still validated
        normalize_coarse_mapping_version(coarse_mapping_version)

    pp_cfg = protocol.get("input_preprocessing", {})
    expected_pp = {"he": "identity", "he_l2": "row_l2", "clip": "row_l2"}
    for space, expected in expected_pp.items():
        actual = str(pp_cfg.get(space, expected)).strip().lower()
        if actual not in VALID_INPUT_PREPROCESSING_STRATEGIES:
            raise ValueError(
                f"Unsupported protocol.input_preprocessing.{space}={actual!r}. "
                f"Expected one of {sorted(VALID_INPUT_PREPROCESSING_STRATEGIES)}."
            )
        if actual != expected:
            raise ValueError(
                "Task A.3 uses a fixed encoder-space preprocessing policy: "
                "protocol.input_preprocessing.he='identity', "
                "protocol.input_preprocessing.he_l2='row_l2' and "
                "protocol.input_preprocessing.clip='row_l2'. "
                f"Got {space}={actual!r}."
            )

    # Slide admission floors: >= 0 (0 disables)
    slide_admission = protocol.get("slide_admission", {})
    if not isinstance(slide_admission, Mapping):
        raise ValueError("protocol.slide_admission must be a mapping.")
    for key in ("min_scorable_cells_per_slide", "min_classes_per_slide"):
        raw = slide_admission.get(key, 0)
        if isinstance(raw, bool):
            raise ValueError(f"protocol.slide_admission.{key} must be an integer, got a bool.")
        try:
            val = int(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"protocol.slide_admission.{key} must be a non-negative integer (0 disables the "
                f"floor); got {raw!r}."
            )
        if val < 0:
            raise ValueError(
                f"protocol.slide_admission.{key} must be >= 0 (0 disables the floor), got {val}."
            )

    # Fold-coverage gate: >= 0 (0 disables)
    fold_coverage = protocol.get("fold_coverage", {})
    if not isinstance(fold_coverage, Mapping):
        raise ValueError("protocol.fold_coverage must be a mapping.")
    _fc_raw = fold_coverage.get("min_train_cells_per_class_per_fold", 0)
    if isinstance(_fc_raw, bool):
        raise ValueError(
            "protocol.fold_coverage.min_train_cells_per_class_per_fold must be an integer, "
            "got a bool."
        )
    try:
        _fc_val = int(_fc_raw)
    except (TypeError, ValueError):
        raise ValueError(
            "protocol.fold_coverage.min_train_cells_per_class_per_fold must be a non-negative "
            f"integer (0 disables the gate); got {_fc_raw!r}."
        )
    if _fc_val < 0:
        raise ValueError(
            "protocol.fold_coverage.min_train_cells_per_class_per_fold must be >= 0 (0 disables "
            f"the gate), got {_fc_val}."
        )
    _fc_exceptions = fold_coverage.get("declared_exceptions", {}) or {}
    if not isinstance(_fc_exceptions, Mapping):
        raise ValueError(
            "protocol.fold_coverage.declared_exceptions must be a mapping {fold_name: reason}."
        )
    # Each declared exception needs a non-empty reason
    _fc_empty = sorted(str(k) for k, v in _fc_exceptions.items() if not str(v).strip())
    if _fc_empty:
        raise ValueError(
            "protocol.fold_coverage.declared_exceptions entries need a non-empty written reason. "
            f"Missing a reason for: {_fc_empty}. State which class(es) the fold cannot train and "
            "why the fold is still worth scoring without them."
        )

    metrics_cfg = protocol.get("metrics", {})
    if not isinstance(metrics_cfg, Mapping):
        raise ValueError("protocol.metrics must be a mapping.")
    # metrics.primary must be one of VALID_A3_PRIMARY_METRICS
    _primary = metrics_cfg.get("primary", None)
    if _primary is None:
        raise ValueError(
            "protocol.metrics.primary is required: it names this benchmark's declared primary "
            f"statistic. Valid values: {sorted(VALID_A3_PRIMARY_METRICS)}."
        )
    if str(_primary) not in VALID_A3_PRIMARY_METRICS:
        raise ValueError(
            f"protocol.metrics.primary={_primary!r} is not a declarable primary. Valid values: "
            f"{sorted(VALID_A3_PRIMARY_METRICS)}."
        )
    _macro_floor = metrics_cfg.get("macro_min_class_support_per_slide", 0)
    if isinstance(_macro_floor, bool):
        raise ValueError(
            "protocol.metrics.macro_min_class_support_per_slide must be an integer, got a bool."
        )
    try:
        _macro_floor = int(_macro_floor)
    except (TypeError, ValueError):
        raise ValueError(
            "protocol.metrics.macro_min_class_support_per_slide must be a non-negative integer "
            f"(0 disables the floor); got {metrics_cfg.get('macro_min_class_support_per_slide')!r}."
        )
    if _macro_floor < 0:
        raise ValueError(
            "protocol.metrics.macro_min_class_support_per_slide must be >= 0 (0 disables the "
            f"floor), got {_macro_floor}."
        )

    class_selection = protocol.get("class_selection", {})
    for key in ("min_train_cells_per_class", "min_train_slides_per_class", "min_trainable_classes"):
        try:
            val = int(class_selection.get(key, 0))
        except (TypeError, ValueError):
            raise ValueError(f"protocol.class_selection.{key} must be a positive integer.")
        if val <= 0:
            raise ValueError(f"protocol.class_selection.{key} must be a positive integer, got {val}.")
    # Optional ADAPTED keys (defaults in fill_a3_settings_folder); must be positive integers
    for key, default in (("adapted_min_cells_per_slide", 200), ("adapted_min_slides", 1)):
        try:
            val = int(class_selection.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f"protocol.class_selection.{key} must be a positive integer.")
        if val <= 0:
            raise ValueError(f"protocol.class_selection.{key} must be a positive integer, got {val}.")

    probe = protocol.get("probe", {})
    if not isinstance(probe, Mapping):
        raise ValueError("protocol.probe must be a mapping.")
    if not isinstance(probe.get("inner_cv_select", False), bool):
        raise ValueError("protocol.probe.inner_cv_select must be a boolean.")
    inner_cv_score = str(probe.get("inner_cv_score", "macro_f1")).strip().lower()
    if inner_cv_score not in {"macro_f1", "balanced_accuracy"}:
        raise ValueError(
            f"protocol.probe.inner_cv_score={inner_cv_score!r} must be "
            "'macro_f1' or 'balanced_accuracy'."
        )
    try:
        max_cells = int(probe.get("inner_cv_max_train_cells", 50000))
    except (TypeError, ValueError):
        raise ValueError("protocol.probe.inner_cv_max_train_cells must be a positive integer.")
    if max_cells <= 0:
        raise ValueError(
            f"protocol.probe.inner_cv_max_train_cells must be a positive integer, got {max_cells}."
        )
    def _validate_c_grid(grid: Any, label: str) -> None:
        """Shared {c_min, c_max, n_points} check for `C_grid` and every `C_grid_overrides` entry."""
        if not isinstance(grid, Mapping):
            raise ValueError(f"{label} must be a mapping {{c_min, c_max, n_points}}.")
        try:
            c_min = float(grid.get("c_min"))
            c_max = float(grid.get("c_max"))
            n_points = int(grid.get("n_points"))
        except (TypeError, ValueError):
            raise ValueError(f"{label} requires numeric c_min, c_max and integer n_points.")
        if not (0.0 < c_min < c_max):
            raise ValueError(
                f"{label} requires 0 < c_min < c_max, got c_min={c_min}, c_max={c_max}."
            )
        if n_points < 2:
            raise ValueError(f"{label}.n_points must be >= 2, got {n_points}.")

    _validate_c_grid(probe.get("C_grid", {}), "protocol.probe.C_grid")

    # Optional per-space C grid overrides (keys must be known spaces)
    c_grid_overrides = probe.get("C_grid_overrides", {}) or {}
    if not isinstance(c_grid_overrides, Mapping):
        raise ValueError(
            "protocol.probe.C_grid_overrides must be a mapping {space: {c_min, c_max, n_points}}."
        )
    valid_grid_spaces = set(VALID_ENCODER_SPACES) | {"coords", "scconcept"}
    for space_key, grid in c_grid_overrides.items():
        if str(space_key) not in valid_grid_spaces:
            raise ValueError(
                f"protocol.probe.C_grid_overrides key {space_key!r} is not a known space; "
                f"expected one of {sorted(valid_grid_spaces)}."
            )
        _validate_c_grid(grid, f"protocol.probe.C_grid_overrides[{space_key!r}]")

    for key, default in (("tol", 1e-4), ("inner_cv_tol", 1e-3)):
        try:
            val = float(probe.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f"protocol.probe.{key} must be a positive float.")
        if not (val > 0.0):
            raise ValueError(f"protocol.probe.{key} must be a positive float, got {val}.")

    try:
        max_groups = int(probe.get("inner_cv_max_groups", 5))
    except (TypeError, ValueError):
        raise ValueError("protocol.probe.inner_cv_max_groups must be a positive integer.")
    if max_groups <= 0:
        raise ValueError(
            f"protocol.probe.inner_cv_max_groups must be a positive integer, got {max_groups}."
        )

    scc_tid = protocol.get("scconcept_training_id", None)
    if scc_tid is not None and (not isinstance(scc_tid, str) or not scc_tid.strip()):
        raise ValueError("protocol.scconcept_training_id must be null or a non-empty string.")

    if not ct_cat_id:
        raise ValueError("protocol.ct_cat_id must be a non-empty string.")
