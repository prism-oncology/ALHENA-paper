import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import numpy as np
import yaml

try:
    import torch
except Exception:  # no cover
    torch = None


def _find_project_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "src" / "utils" / "constants.py").exists():
            return parent
    return None


PROJECT_ROOT = _find_project_root()


# Task A.1 level: the H&E patch vs the full ST content of the patch
VALID_LEVELS = {"patch_all"}
VALID_EVAL_CONFIGS = {"STHELAR_TRAIN", "STHELAR_TEST", "HEST"}
VALID_SUB_EVAL_CONFIGS = {
    "STHELAR_TRAIN_CV",
    "STHELAR_TRAIN_COORDS",
    "STHELAR_TEST_ONE_SHOT",
    "STHELAR_TEST_ADAPTED",
    "HEST_CV_CLOSE",
    "HEST_CV_MAGNIFICATION",
    "HEST_CV_SPECIES",
    "HEST_ORGANS",
}
HEST_CV_SUB_EVAL_CONFIGS = {
    "HEST_CV_CLOSE",
    "HEST_CV_MAGNIFICATION",
    "HEST_CV_SPECIES",
    "HEST_ORGANS",
}

CONTROL_SUB_EVAL_CONFIGS = {
    "STHELAR_TRAIN_COORDS",
}

CONTROL_KIND_BY_SUB_EVAL_CONFIG = {
    "STHELAR_TRAIN_COORDS": "coords_baseline",
}

REFERENCE_CONTROL_EVAL_CONFIGURATION = "STHELAR_TRAIN"
REFERENCE_CONTROL_SUB_EVAL_CONFIGURATION = "STHELAR_TRAIN_CV"

VALID_ENCODER_SPACES = {"he", "he_l2", "clip"}
MEMBERSHIP_BACKBONES = ("uni2h", "virchow2", "hoptimus0", "hoptimus1")

# One prediction head: closed-form ridge
VALID_HEAD_TYPES = {"ridge"}
DIRECT_HEAD_TYPES = {"ridge"}
# Used by Task A.3 (Task A.1 does not reduce its features)
VALID_INPUT_REDUCTION_NAMES = {"none", "pca"}
VALID_INPUT_PREPROCESSING_STRATEGIES = {"identity", "row_l2"}


_PATIENT_NULL_STRINGS = {"", "na", "nan", "none", "null", "<na>"}

AUTO_GENE_PANEL_ALLOWED = {
    "HEST_CV_CLOSE",
    "HEST_CV_MAGNIFICATION",
    "HEST_CV_SPECIES",
    "HEST_ORGANS",
    "STHELAR_TEST_ADAPTED",
}

AUTO_GENE_PANEL_SELECTION_SCOPE = {
    "HEST_CV_CLOSE": "per_tissue",
    "HEST_CV_MAGNIFICATION": "per_tissue",
    "HEST_CV_SPECIES": "per_tissue",
    "HEST_ORGANS": "per_tissue",
    "STHELAR_TEST_ADAPTED": "per_slide",
}

# STHELAR_TEST_ADAPTED: roles of the three spatial stripes of a slide in each rotation
# (two train stripes, one test stripe); used by protocols.py and _3_run_eval.py
_ADAPTED_STRIPE_ROLES = (
    {0: "train", 1: "train", 2: "test"},
    {1: "train", 2: "train", 0: "test"},
    {2: "train", 0: "train", 1: "test"},
)


def stripes_for_role(roles: Mapping[int, str], role: str) -> List[int]:
    """Sorted stripe ids with the given role in one _ADAPTED_STRIPE_ROLES rotation."""
    return sorted(int(k) for k, v in roles.items() if str(v) == str(role))


def _protocol_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    return cfg.get("protocol", cfg)


def head_type_from_config(cfg: Mapping[str, Any]) -> str:
    protocol = _protocol_cfg(cfg)
    head_cfg = protocol.get("head", {})
    return str(head_cfg.get("type", "ridge")).strip().lower()


def is_direct_head_config(cfg: Mapping[str, Any]) -> bool:
    return head_type_from_config(cfg) in DIRECT_HEAD_TYPES


def input_reduction_from_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    protocol = _protocol_cfg(cfg)
    red = dict(protocol.get("input_reduction", {"name": "none"}))
    name = str(red.get("name", "none")).strip().lower()
    out: Dict[str, Any] = {"name": name}
    if name == "pca":
        out["n_components"] = int(red.get("n_components", 256))
    else:
        # n_components is ignored without PCA
        out["n_components"] = None
    return out


def prediction_head_manifest_payload(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    protocol = _protocol_cfg(cfg)
    ridge = dict(protocol.get("ridge", {}))
    return {
        "head_type": head_type_from_config(cfg),
        "mode": "closed_form_ridge",
        "scgpt_vocab_filtered_genes": True,
        "per_gene_bias": True,
        # No dimensionality reduction (full-dimensional features)
        "input_reduction": {"name": "none", "n_components_requested": None},
        "sample_weights": str(ridge.get("sample_weights", "slide_uniform")),
        "alpha_selection": dict(ridge.get("alpha_selection", {})),
        "alpha_grid": dict(ridge.get("alpha_grid", {})),
        "deterministic": True,
    }


def gene_panel_mode_from_G_base_id(G_base_id: Any) -> str:
    return "auto" if str(G_base_id).strip().lower() == "auto" else "G_base"


def validate_G_base_id_usage(
    *,
    eval_configuration: str,
    sub_eval_configuration: str,
    G_base_id: str,
) -> None:
    """
    Validate the gene-panel strategy: G_base_<n> (source-fixed), or auto (target-informed,
    allowed only for HEST_CV_* per tissue and STHELAR_TEST_ADAPTED per slide).
    """
    eval_configuration = str(eval_configuration)
    sub_eval_configuration = str(sub_eval_configuration)
    G_base_id = str(G_base_id).strip()

    if not G_base_id:
        raise ValueError("G_base_id must be non-empty. Use 'G_base_<n>' or 'auto'.")

    if gene_panel_mode_from_G_base_id(G_base_id) != "auto":
        return

    if sub_eval_configuration in AUTO_GENE_PANEL_ALLOWED:
        if eval_configuration == "HEST" and sub_eval_configuration in HEST_CV_SUB_EVAL_CONFIGS:
            return
        if eval_configuration == "STHELAR_TEST" and sub_eval_configuration == "STHELAR_TEST_ADAPTED":
            return

    if sub_eval_configuration == "STHELAR_TEST_ONE_SHOT":
        reason = (
            "STHELAR_TEST_ONE_SHOT tests on unseen STHELAR_test slides, so its genes "
            "cannot be chosen from STHELAR_test expression."
        )
    elif sub_eval_configuration in CONTROL_SUB_EVAL_CONFIGS:
        reason = (
            "control protocols reuse the gene panel of their reference protocol."
        )
    elif sub_eval_configuration.startswith("STHELAR_TRAIN"):
        reason = "STHELAR_TRAIN_* protocols must remain source-fixed."
    else:
        reason = "this protocol has no approved target-informed gene-panel unit."

    raise ValueError(
        "G_base_id='auto' is only allowed for HEST_CV_* and STHELAR_TEST_ADAPTED.\n"
        f"It is not allowed for {sub_eval_configuration} because {reason}\n"
        "Use --G_base_id G_base_<n> instead."
    )

def clean_patient_label(raw: Any) -> Optional[str]:
    """Normalized patient label (casefolded, whitespace collapsed), or None for missing/null-like labels."""
    if raw is None:
        return None
    s = re.sub(r"\s+", " ", str(raw).strip())
    if not s:
        return None
    low = s.casefold()
    if low in _PATIENT_NULL_STRINGS:
        return None
    return low


def _split_group_component(x: Any) -> str:
    s = re.sub(r"\s+", " ", str(x).strip())
    if not s:
        s = "unknown"
    return s.casefold().replace("::", "_")


def build_split_group_id(
    *,
    slide_id: str,
    source: str,
    subset: str,
    tissue: str,
    dataset_title: str,
    patient_clean: Optional[str],
) -> str:
    if patient_clean is not None:
        return "patient::" + "::".join(
            [
                _split_group_component(source),
                _split_group_component(tissue),
                _split_group_component(dataset_title),
                _split_group_component(patient_clean),
            ]
        )
    return f"slide::{str(slide_id)}"


def normalize_modelgroup_name(modelgroup: str) -> str:
    toks = [str(t).strip().lower() for t in str(modelgroup).split("_") if str(t).strip()]
    return "_".join(sorted(toks))


def infer_encoder_modelgroup(encoder: Any) -> str:
    """Modelgroup of an encoder, from its (resize_size, crop_size) preprocess (as in the dataset membership builder)."""
    ensure_project_import()
    from src.utils.models_settings import BACKBONE_TRANSFORM

    by_sig: Dict[tuple[int, int], List[str]] = {}
    for backbone in MEMBERSHIP_BACKBONES:
        if backbone not in BACKBONE_TRANSFORM:
            continue
        cfg = BACKBONE_TRANSFORM[backbone]
        sig = (int(cfg.get("resize_size", 0)), int(cfg.get("crop_size", 0)))
        by_sig.setdefault(sig, [])
        if backbone not in by_sig[sig]:
            by_sig[sig].append(backbone)
    sig_to_group = {
        sig: normalize_modelgroup_name("_".join(models))
        for sig, models in by_sig.items()
    }

    backbone = str(getattr(encoder, "backbone", "")).lower()
    resize_size = int(getattr(encoder, "resize_size", 0))
    crop_size = int(getattr(encoder, "crop_size", 0))
    if backbone in BACKBONE_TRANSFORM:
        resize_size = int(BACKBONE_TRANSFORM[backbone].get("resize_size", resize_size))
        crop_size = int(BACKBONE_TRANSFORM[backbone].get("crop_size", crop_size))

    sig = (resize_size, crop_size)
    return sig_to_group.get(sig, f"custom_r{resize_size}_c{crop_size}")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def maybe_git_commit() -> str | None:
    if PROJECT_ROOT is None:
        return None
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None

@dataclass
class EncoderSpec:
    training_num: str
    checkpoint_num: str
    space: str

    def folder_name(self) -> str:
        return f"HEencoder_t{self.training_num}e{self.checkpoint_num}_{self.space}"

    def as_dict(self) -> Dict[str, str]:
        return {"training_num": self.training_num, "checkpoint_num": self.checkpoint_num, "space": self.space}


@dataclass
class RunContext:
    eval_id: str
    dataset_id: str
    training_dataset_id: str
    eval_configuration: str
    sub_eval_configuration: str
    level: str
    modelgroup: str
    G_base_id: str
    scgpt_model_name: str
    protocol_seed: int = 20260319


@dataclass
class SlideRecord:
    slide_id: str
    source: str
    subset: str
    tissue: str
    species: str
    patient_raw: Optional[str] = None
    patient_clean: Optional[str] = None
    dataset_title: str = ""
    split_group_id: str = ""


@dataclass
class FileManifest:
    path: str
    sha256: str
    n_bytes: int


def ensure_project_import() -> None:
    if PROJECT_ROOT is None:
        return
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def get_full_path(*parts: str, as_str: bool = False):
    ensure_project_import()
    from src.utils.constants import get_full_path as _get_full_path

    return _get_full_path(*parts, as_str=as_str)


def int_cell_id(cell_id_str: str) -> int:
    """Canonical Xenium cell-id -> int encoder (see src/utils/cell_ids.py)."""
    ensure_project_import()
    from src.utils.cell_ids import int_cell_id as _int_cell_id

    return _int_cell_id(cell_id_str)


def resolve_eval_task_root() -> Path:
    return Path(get_full_path("eval", "A-1_eval_st_predictions", as_str=True))


def resolve_eval_dataset_root(training_dataset_id: str) -> Path:
    return Path(get_full_path("eval", "eval_datasets", "st_eval_data", f"eval_st_{training_dataset_id}", as_str=True))


def resolve_metadata_csv(training_dataset_id: str) -> Path:
    return resolve_eval_dataset_root(training_dataset_id) / f"metadata_eval_st_{training_dataset_id}.csv"


def resolve_G_base_dir(training_dataset_id: str, G_base_id: str) -> Path:
    return resolve_eval_dataset_root(training_dataset_id) / "G_base" / G_base_id


def resolve_gene_property_dir(training_dataset_id: str, gene_property_id: str) -> Path:
    return resolve_eval_dataset_root(training_dataset_id) / "gene_properties" / gene_property_id


def resolve_eval_signatures_dir(eval_signatures_id: str) -> Path:
    return Path(get_full_path("eval", "eval_datasets", "eval_signatures", eval_signatures_id, as_str=True))


def resolve_eval_signatures_INT_dir(eval_signatures_INT_id: str) -> Path:
    """Directory of an internal (data-driven) signature panel built by _14_eval_signatures_INT.py."""
    return Path(get_full_path("eval", "eval_datasets", "eval_signatures_INT", eval_signatures_INT_id, as_str=True))


# Separator of the signature target names "<signature_set><SEP><signature_id>" (target_basis="signatures");
# gene symbols and signature ids must not contain it
SIGNATURE_AXIS_SEP = "::"


# Signature sets usable in signature_sets: config id key, table file, weighted table (external sets get weight 1.0)
SIGNATURE_SET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "external": {
        "id_key": "eval_signatures_id",
        "table_file": "eval_signatures_table.parquet",
        "weighted": False,
    },
    "internal": {
        "id_key": "eval_signatures_INT_id",
        "table_file": "eval_signatures_INT_table.parquet",
        "weighted": True,
    },
}


def _resolve_signature_set_dir(set_name: str, set_id: str) -> Path:
    if set_name == "external":
        return resolve_eval_signatures_dir(set_id)
    if set_name == "internal":
        return resolve_eval_signatures_INT_dir(set_id)
    raise ValueError(f"[ERROR] Unknown signature set name '{set_name}'.")


def _read_signature_set(set_name: str, set_id: str):
    """
    Read one signature set as a long DataFrame [signature_set, signature_id, signature_category, gene, weight]
    (weight 1.0 if the table has no weight column; category from the set's config.json).
    """
    import pandas as pd  # lazy: common.py stays pandas-free at import time

    spec = SIGNATURE_SET_REGISTRY.get(set_name)
    if spec is None:
        raise ValueError(
            f"[ERROR] Unknown signature set '{set_name}'. Known sets: "
            f"{sorted(SIGNATURE_SET_REGISTRY)}."
        )
    set_dir = _resolve_signature_set_dir(set_name, set_id)
    table_path = set_dir / spec["table_file"]
    if not table_path.is_file():
        raise FileNotFoundError(
            f"[ERROR] Signature table for set '{set_name}' (id '{set_id}') not "
            f"found at '{table_path}'."
        )
    df = pd.read_parquet(table_path)
    if "signature_id" not in df.columns or "gene" not in df.columns:
        raise ValueError(
            f"[ERROR] Signature table '{table_path}' must have columns "
            f"'signature_id' and 'gene'; got {list(df.columns)}."
        )
    df = df.copy()
    df["signature_id"] = df["signature_id"].astype(str)
    df["gene"] = df["gene"].astype(str)
    if "weight" in df.columns:
        df["weight"] = df["weight"].astype(float)
    else:
        df["weight"] = 1.0

    # Per-signature category from config.json
    categories: Dict[str, str] = {}
    config_path = set_dir / "config.json"
    if config_path.is_file():
        with open(config_path, "r") as f:
            sig_meta = json.load(f).get("signatures", {})
        if isinstance(sig_meta, Mapping):
            for sid, meta in sig_meta.items():
                if isinstance(meta, Mapping):
                    categories[str(sid)] = str(meta.get("category", set_name))
    df["signature_category"] = df["signature_id"].map(
        lambda s: categories.get(str(s), set_name)
    )
    df["signature_set"] = set_name

    # Guard against duplicate (signature_id, gene) within a set.
    df = df.drop_duplicates(subset=["signature_id", "gene"], keep="first")
    return df[["signature_set", "signature_id", "signature_category", "gene", "weight"]]


def load_signature_sets(cfg: Mapping[str, Any]):
    """
    Concatenate the signature sets listed in cfg['protocol']['signature_sets'] into one long table
    [signature_set, signature_id, signature_category, gene, weight].
    """
    import pandas as pd

    protocol = cfg.get("protocol", {})
    set_names = list(protocol.get("signature_sets", []) or [])
    frames = []
    for set_name in set_names:
        spec = SIGNATURE_SET_REGISTRY.get(set_name)
        if spec is None:
            raise ValueError(
                f"[ERROR] signature_sets lists unknown set '{set_name}'. Known "
                f"sets: {sorted(SIGNATURE_SET_REGISTRY)}."
            )
        set_id = protocol.get(spec["id_key"])
        if not isinstance(set_id, str) or not set_id.strip():
            raise ValueError(
                f"[ERROR] signature_sets includes '{set_name}' but "
                f"protocol['{spec['id_key']}'] is unset. Provide the id or remove "
                f"the set from signature_sets."
            )
        frames.append(_read_signature_set(set_name, set_id.strip()))
    if not frames:
        return pd.DataFrame(
            columns=["signature_set", "signature_id", "signature_category", "gene", "weight"]
        )
    return pd.concat(frames, axis=0, ignore_index=True)


def resolve_training_dataset_dir(dataset_id: str, training_dataset_id: str) -> Path:
    return Path(get_full_path("train", "training_datasets", dataset_id, training_dataset_id, as_str=True))


def resolve_he_embeddings_dir(training_dataset_id: str, training_num: str, checkpoint_num: str) -> Path:
    return resolve_eval_dataset_root(training_dataset_id) / "he_embeddings" / f"HEembs_t{training_num}e{checkpoint_num}"


def resolve_encoder_path(training_num: str, checkpoint_num: str) -> Path:
    return Path(
        get_full_path(
            "train",
            "trainings",
            "trainings_HistoST",
            f"training_{training_num}",
            "trained_encoders",
            f"HEencoder_t{training_num}e{checkpoint_num}.pt",
            as_str=True,
        )
    )


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rmtree_appledouble_safe(path: Path) -> None:
    """shutil.rmtree that ignores macOS AppleDouble (._*) sidecars removed during the walk."""
    def _ignore_missing(func, p, exc):
        if not isinstance(exc, FileNotFoundError):
            raise exc

    shutil.rmtree(path, onexc=_ignore_missing)
    if path.exists():
        shutil.rmtree(path, onexc=_ignore_missing)
    if path.exists():
        raise RuntimeError(
            f"Could not fully remove {path}. Rename the directory aside and re-run."
        )


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


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
       json.dump(_jsonify(payload), f, indent=2, sort_keys=False)


def save_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """save_json through a temp file and a rename, so the file is either complete or absent."""
    mkdir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_jsonify(payload), f, indent=2, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        # No-op on success (os.replace consumed it); cleans up on failure.
        tmp.unlink(missing_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(payload), f, sort_keys=False, allow_unicode=True)


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


def render_eval_config_with_comments(cfg: Mapping[str, Any]) -> str:
    """Render eval_config.yaml with explanatory comments (same YAML content as cfg)."""
    identity = dict(cfg.get("identity", {}))
    protocol = dict(cfg.get("protocol", {}))

    lines: List[str] = []
    lines.extend(
        [
            "# HistoST Task A.1 evaluation config",
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
        "- STHELAR_TEST: unseen STHELAR slides\n"
        "- HEST: external OOD benchmark",
    )
    add_protocol_key(
        "sub_eval_configuration",
        "Specific protocol inside the high-level benchmark family.\n"
        "This is the main switch controlling CV / one-shot / controls.",
    )
    add_protocol_key(
        "reference_eval_id",
        "Only used by child control protocols.\n"
        "Must point to an existing STHELAR_TRAIN/STHELAR_TRAIN_CV eval_id whose frozen protocol is reused.",
    )
    add_protocol_key(
        "control_kind",
        "Only used by child control protocols.\n"
        "This describes what is changed relative to the frozen reference protocol.",
    )
    add_protocol_key(
        "level",
        "Supervision level. Task A.1 has one: patch_all (the H&E patch against the full ST\n"
        "content of that patch; the only level supported by HEST).",
    )
    add_protocol_key(
        "modelgroup",
        "Geometric preprocessing group (resize/crop) of the encoders.\n"
        "All encoders evaluated under this eval_id must belong to this modelgroup.",
    )
    add_protocol_key(
        "G_base_id",
        "Gene-panel strategy for Task A.1.\n"
        "- G_base_<n>: source-fixed transfer benchmark using a precomputed STHELAR/CLIP-training gene universe.\n"
        "- auto: target-informed diagnostic benchmark. Allowed only for HEST_CV_* and STHELAR_TEST_ADAPTED.",
    )
    add_protocol_key(
        "gene_property_id",
        "Frozen gene-property table (built by _12_gene_properties.py) used to stratify the\n"
        "panel into teacher-relevant gene strata: cell-type-specificity (Tau) and\n"
        "spatial-structure (Moran's I) terciles. Encoder- and eval_id-independent.",
    )
    add_protocol_key(
        "eval_signatures_id",
        "Frozen external gene-signature panel (built by _13_eval_signatures.py) used to\n"
        "stratify the panel by membership in biology-driven gene sets (MSigDB Hallmark,\n"
        "curated immune / niche / stromal). Independent of G_base_id and training_dataset_id;\n"
        "usable with source-fixed (G_base_<n>) and auto-panel (HEST_CV_* / STHELAR_TEST_ADAPTED)\n"
        "evals. Each signature is intersected with the eval-local trainable gene union\n"
        "at fill time; signatures with fewer than MIN_GENES_FOR_STRATA matching genes\n"
        "are skipped and recorded in the manifest.",
    )
    add_protocol_key(
        "eval_signatures_INT_id",
        "Frozen internal, data-driven gene-signature panel (built by\n"
        "_14_eval_signatures_INT.py): co-expression modules discovered among the\n"
        "G_base genes on CLIP-training cells only, packaged as named gene sets with\n"
        "eigengene/kME weights. Scoped to (G_base_id, training_dataset_id). Used when\n"
        "'internal' appears in signature_sets. Defaults to None.",
    )
    add_protocol_key(
        "target_basis",
        "Prediction target axis:\n"
        "- genes: per-gene A.1 (predict single-gene z-scores).\n"
        "- signatures: predict per-signature aggregate scores (Sum w*z over z-scored\n"
        "  member genes) for the sets in signature_sets, with a per-signature coverage\n"
        "  gate and a noise ceiling recomputed on the aggregate.",
    )
    add_protocol_key(
        "signature_sets",
        "Signature sets predicted when target_basis=signatures:\n"
        "- external: uses eval_signatures_id (biology-curated).\n"
        "- internal: uses eval_signatures_INT_id (data-driven co-expression modules).\n"
        "Both may be listed; they are gated together per config. Ignored when\n"
        "target_basis=genes.",
    )
    add_protocol_key(
        "signature_target",
        "Signature-target-mode parameters (used only when target_basis=signatures):\n"
        "- aggregation: weighted_zscore_mean (Sum w*z / Sum|w|, as in Task B).\n"
        "- coverage_gate: keep a signature only if enough member genes are\n"
        "  measured+supported in this config's data (min_covered_fraction and\n"
        "  min_covered_genes); the per-gene support filter lifted to the signature.\n"
        "- correlation_validity / macro_aggregation_validity: the per-gene validity\n"
        "  floors lifted to the signature axis.",
    )
    add_protocol_key(
        "scgpt_model_name",
        "Fixed scGPT source whose gene vocabulary filters the gene panel\n"
        "(drops control probes, undefined/blank genes, etc.).\n"
        "scGPT is used only for vocabulary filtering, not for prediction.",
    )
    add_protocol_key(
        "protocol_seed",
        "Seed of the protocol: splits, gene lists, held-out genes and the other persisted\n"
        "benchmark definitions. The probe is closed-form and has no seed.",
    )
    add_protocol_key(
        "target_transform",
        "Target preprocessing:\n"
        "- name: depth_norm_zscore_winsor\n"
        "- target_sum: library-size target for per-row depth normalization\n"
        "  (e = log1p(count / row_total * target_sum)).\n"
        "- zscore: per-(slide,gene) standardization using stats frozen in settings/target_norm/.\n"
        "- zscore_clip: winsorization limit of the z-scored target, in SD; the same clip is\n"
        "  applied to the target and to the noise ceiling.",
    )
    add_protocol_key(
        "head",
        "Prediction head for Task A.1.\n"
        "- type: ridge (the only value)\n"
        "- A closed-form weighted ridge, solved independently per target, with one\n"
        "  unpenalized bias per target. See the `ridge` block for its parameters.",
    )
    add_protocol_key(
        "ridge",
        "The probe: closed-form weighted ridge, deterministic (no seeds, no validation split,\n"
        "no early stopping). Each target is solved independently on the slides where it is eligible.\n"
        "- sample_weights: slide_uniform (w_i = 1/n_rows(slide(i))): every slide has the same weight.\n"
        "- fit_intercept: one unpenalized bias per target.\n"
        "- alpha_grid: alpha = lambda * trace(A)/D (lambda in units of the mean eigenvalue of the\n"
        "  Gram matrix), so the grid does not depend on the feature scale. A selected lambda at a\n"
        "  grid edge raises an error unless the capacity is saturated there (OLS or intercept-only,\n"
        "  modeling.boundary_saturation).\n"
        "- alpha_selection: one alpha per (encoder, fold), chosen by leave-one-train-group-out\n"
        "  CV on train data only, scored on slide_macro_spearman. The group is a slide, or a\n"
        "  stripe for STHELAR_TEST_ADAPTED (one train slide per fold). The selected lambda is the\n"
        "  largest one whose mean paired per-group deficit against the best lambda is within one\n"
        "  standard error of zero (1-SE rule, training.select_lambda_1se).\n"
        "  The realized capacity is reported per fold as effective_dof.",
    )
    add_protocol_key(
        "input_preprocessing",
        "Feature geometry of the probe input (one stage, no dimensionality reduction):\n"
        "- he space: identity\n"
        "- clip / he_l2 spaces: row-wise L2 normalization\n"
        "The probe uses the full-dimensional backbone features. The embedding width can differ\n"
        "between the encoders of one eval_id (e.g. 1536-d uni2h / hoptimus1, 2560-d virchow2);\n"
        "effective_dof is reported raw (not normalized by the width D).",
    )
    add_protocol_key(
        "metrics",
        "Benchmark metric definitions and validity thresholds.\n"
        "Primary metric is slide-macro Spearman over valid (slide, gene) pairs.",
    )
    add_protocol_key(
        "gene_selection",
        "Rules defining trainable genes from the frozen candidate gene universe.\n"
        "For G_base, the candidate universe is source-fixed.\n"
        "For auto, the candidate universe is target-informed and frozen during settings fill:\n"
        "- HEST: one auto panel per tissue\n"
        "- STHELAR_TEST_ADAPTED: one auto panel per slide\n"
        "Fold-specific train-support filtering is still applied after candidate-panel selection.",
    )
    add_protocol_key(
        "cv",
        "CV-specific parameters.\n"
        "safe_border_px should reflect patch overlap, to prevent leakage in within-slide splits.\n"
        "There is no validation split: alpha is chosen by leave-one-train-group-out CV inside\n"
        "the train set.",
    )
    add_protocol_key(
        "hest_cv",
        "HEST-only tissue-wise CV settings.\n"
        "HEST train/validation/test splits never use STHELAR slides. "
        "Only G_base remains STHELAR-derived.",
    )
    add_protocol_key(
        "structure",
        "Level-3 structure-analysis defaults (diagnostic analyses).",
    )
    add_protocol_key(
        "reports",
        "Level-4 qualitative report defaults.",
    )
    add_protocol_key(
        "runtime",
        "wandb logging settings (the ridge probe runs on CPU).",
    )

    # Remove final extra blank line for cleanliness
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def save_eval_config_with_comments(path: Path, cfg: Mapping[str, Any]) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_eval_config_with_comments(cfg))


def protocol_identity_payload(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Identity and protocol blocks of eval_config.yaml (frozen after _2_fill_eval_settings.py)."""
    return {
        "identity": _jsonify(dict(cfg.get("identity", {}))),
        "protocol": _jsonify(dict(cfg.get("protocol", {}))),
    }


def protocol_identity_sha256(cfg: Mapping[str, Any]) -> str:
    blob = json.dumps(
        protocol_identity_payload(cfg),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(blob)


def assert_eval_config_matches_protocol_manifest(
    cfg: Mapping[str, Any],
    protocol_manifest: Mapping[str, Any],
) -> None:
    actual = protocol_identity_payload(cfg)
    expected = {
        "identity": _jsonify(dict(protocol_manifest.get("identity", {}))),
        "protocol": _jsonify(dict(protocol_manifest.get("protocol", {}))),
    }
    if actual != expected:
        raise RuntimeError(
            "eval_config.yaml does not match the protocol saved by "
            "`_2_fill_eval_settings.py` (config modified after the settings fill). "
            "Create a new eval_id or rebuild the settings folder from scratch."
        )


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {} if data is None else data


def save_text(path: Path, text: str) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def manifest_for_file(path: Path) -> FileManifest:
    return FileManifest(path=str(path), sha256=sha256_file(path), n_bytes=path.stat().st_size)


def is_control_sub_eval_configuration(sub_eval_configuration: str) -> bool:
    return str(sub_eval_configuration) in CONTROL_SUB_EVAL_CONFIGS


def control_kind_for_sub_eval_configuration(sub_eval_configuration: str) -> str:
    sub_eval_configuration = str(sub_eval_configuration)
    if sub_eval_configuration not in CONTROL_KIND_BY_SUB_EVAL_CONFIG:
        raise ValueError(f"Not a control sub_eval_configuration: {sub_eval_configuration}")
    return CONTROL_KIND_BY_SUB_EVAL_CONFIG[sub_eval_configuration]


def reference_protocol_file_manifests(reference_eval_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Hashes of the reference protocol files used by child controls (paths relative to reference_eval_dir/settings)."""
    reference_eval_dir = Path(reference_eval_dir)
    settings_dir = reference_eval_dir / "settings"

    required = {
        "eval_config.yaml",
        "protocol_manifest.json",
        "rows/all_rows.parquet",
        "splits/splits_manifest.json",
        "genes/genes_manifest.json",
        "scgpt_manifest.json",
    }

    relpaths = set(required)
    for pattern in [
        "rows/*.parquet",
        "splits/*.json",
        "genes/fold_*_genes.json",
        "genes/fold_*_train_gene_support.csv",
        "genes/fold_*_train_slide_gene_eligibility.csv",
        "genes/fold_*_train_slide_eligibility_summary.csv",
        "genes/within_slide_genes.json",
        "genes/*train_slide_gene_eligibility.csv",
        "genes/*train_slide_eligibility_summary.csv",
        "genes/oneshot_genes.json",
        "target_norm/zscore_stats.parquet",
        "target_norm/target_norm_manifest.json",
    ]:
        for p in settings_dir.glob(pattern):
            # Skip hidden files (macOS AppleDouble ._* sidecars)
            if p.name.startswith("."):
                continue
            if p.is_file():
                relpaths.add(str(p.relative_to(settings_dir)))

    missing = sorted(rel for rel in required if not (settings_dir / rel).exists())
    if missing:
        raise FileNotFoundError(
            f"Reference eval_id is missing required protocol files under {settings_dir}: {missing}"
        )

    out: Dict[str, Dict[str, Any]] = {}
    for rel in sorted(relpaths):
        p = settings_dir / rel
        if not p.exists() or not p.is_file():
            continue
        out[rel] = manifest_for_file(p).__dict__
    return out


def reference_protocol_hash_aliases(file_manifests: Mapping[str, Mapping[str, Any]]) -> Dict[str, str]:
    """Hashes of the main reference protocol files, for the child reference_protocol.json."""
    aliases = {
        "reference_protocol_manifest_sha256": "protocol_manifest.json",
        "reference_eval_config_sha256": "eval_config.yaml",
        "reference_rows_manifest_sha256": "rows/all_rows.parquet",
        "reference_splits_manifest_sha256": "splits/splits_manifest.json",
        "reference_genes_manifest_sha256": "genes/genes_manifest.json",
        "reference_scgpt_manifest_sha256": "scgpt_manifest.json",
    }
    out: Dict[str, str] = {}
    for alias, rel in aliases.items():
        if rel not in file_manifests:
            raise FileNotFoundError(f"Missing reference protocol file required for alias {alias}: {rel}")
        out[alias] = str(file_manifests[rel]["sha256"])
    return out


def protocol_eval_dir_from_manifest(run_eval_dir: Path, protocol_manifest: Mapping[str, Any]) -> Path:
    """
    For normal eval_ids, protocol_eval_dir == run_eval_dir.
    For child controls, protocol_eval_dir is the frozen reference eval_id.
    """
    ref = protocol_manifest.get("reference_protocol", {})
    if bool(protocol_manifest.get("derived_from_reference", False)) or bool(ref.get("derived_from_reference", False)):
        if not ref:
            raise RuntimeError(
                "protocol_manifest.json has derived_from_reference=True but no reference_protocol block."
            )
        return Path(ref["reference_eval_dir"])
    return Path(run_eval_dir)


def assert_reference_protocol_hashes_unchanged(protocol_manifest: Mapping[str, Any]) -> None:
    """Check that the reference protocol files of a child control have not changed since its creation."""
    ref = protocol_manifest.get("reference_protocol", {})
    if not (bool(protocol_manifest.get("derived_from_reference", False)) or bool(ref.get("derived_from_reference", False))):
        return

    reference_eval_dir = Path(ref["reference_eval_dir"])
    stored_files = ref.get("reference_protocol_files", {})
    if not stored_files:
        raise RuntimeError(
            "Child protocol manifest is missing reference_protocol_files. "
            "Recreate the child eval_id."
        )

    current_files = reference_protocol_file_manifests(reference_eval_dir)

    mismatches: List[str] = []
    for rel, stored in stored_files.items():
        if rel not in current_files:
            mismatches.append(f"{rel}: missing now")
            continue
        old_sha = str(stored.get("sha256", ""))
        new_sha = str(current_files[rel].get("sha256", ""))
        old_n = int(stored.get("n_bytes", -1))
        new_n = int(current_files[rel].get("n_bytes", -1))
        if old_sha != new_sha or old_n != new_n:
            mismatches.append(
                f"{rel}: sha256 {old_sha[:12]} -> {new_sha[:12]}, n_bytes {old_n} -> {new_n}"
            )

    if mismatches:
        raise RuntimeError(
            "Reference protocol files changed since this child control eval_id was created. "
            "Recreate the child from the updated reference. "
            f"First mismatches: {mismatches[:10]}"
        )


def ndarray_sha256(arr: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(arr).view(np.uint8).tobytes())


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def build_logger(log_path: Path, name: str = "histost_eval") -> logging.Logger:
    mkdir(log_path.parent)
    logger = logging.getLogger(f"{name}_{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def flatten_dict(d: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(dict(rec), sort_keys=False) + "\n")


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    mkdir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), sort_keys=False) + "\n")


def read_metadata(training_dataset_id: str) -> List[SlideRecord]:
    import pandas as pd

    meta_path = resolve_metadata_csv(training_dataset_id)
    df = pd.read_csv(meta_path)
    needed = ["slide_id", "source", "subset", "species", "patient", "dataset_title"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Metadata '{meta_path}' is missing required columns: {missing}")

    tissue_col = "tissue_HistoST" if "tissue_HistoST" in df.columns else None
    if tissue_col is None:
        raise KeyError(f"Metadata '{meta_path}' must contain 'tissue_HistoST'.")

    records: List[SlideRecord] = []
    for row in df.itertuples(index=False):
        slide_id = str(getattr(row, "slide_id"))
        source = str(getattr(row, "source"))
        subset = str(getattr(row, "subset"))
        tissue = str(getattr(row, tissue_col))
        species = str(getattr(row, "species"))
        patient_raw_obj = getattr(row, "patient")
        patient_raw = None if pd.isna(patient_raw_obj) else str(patient_raw_obj)
        patient_clean = clean_patient_label(patient_raw_obj)
        dataset_title = str(getattr(row, "dataset_title"))

        split_group_id = build_split_group_id(
            slide_id=slide_id,
            source=source,
            subset=subset,
            tissue=tissue,
            dataset_title=dataset_title,
            patient_clean=patient_clean,
        )

        records.append(
            SlideRecord(
                slide_id=slide_id,
                source=source,
                subset=subset,
                tissue=tissue,
                species=species,
                patient_raw=patient_raw,
                patient_clean=patient_clean,
                dataset_title=dataset_title,
                split_group_id=split_group_id,
            )
        )
    return records


def read_eval_config(eval_dir: Path) -> Dict[str, Any]:
    return load_yaml(eval_dir / "settings" / "eval_config.yaml")


def validate_protocol_identity(cfg: Mapping[str, Any]) -> None:
    level = str(cfg["protocol"]["level"])
    eval_configuration = str(cfg["protocol"]["eval_configuration"])
    sub_eval_configuration = str(cfg["protocol"]["sub_eval_configuration"])
    G_base_id = str(cfg["protocol"].get("G_base_id", ""))
    head_type = head_type_from_config(cfg)

    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid level='{level}'. Expected one of {sorted(VALID_LEVELS)}.")
    if eval_configuration not in VALID_EVAL_CONFIGS:
        raise ValueError(f"Invalid eval_configuration='{eval_configuration}'.")
    if sub_eval_configuration not in VALID_SUB_EVAL_CONFIGS:
        raise ValueError(f"Invalid sub_eval_configuration='{sub_eval_configuration}'.")
    if eval_configuration == "HEST" and level != "patch_all":
        raise ValueError("HEST protocols only support level='patch_all'.")
    if eval_configuration not in sub_eval_configuration:
        raise ValueError(f"Mismatch between eval_configuration='{eval_configuration}' and sub_eval_configuration='{sub_eval_configuration}'.")

    if head_type not in VALID_HEAD_TYPES:
        raise ValueError(
            f"Invalid protocol.head.type={head_type!r}. "
            f"Expected one of {sorted(VALID_HEAD_TYPES)}."
        )

    if "input_reduction" in cfg["protocol"]:
        raise ValueError(
            "protocol.input_reduction is not supported by Task A.1. "
            "Remove the block and create a new eval_id."
        )

    target_transform = cfg["protocol"].get("target_transform", {})
    target_transform_name = str(target_transform.get("name", "")).strip().lower()
    if target_transform_name != "depth_norm_zscore_winsor":
        raise ValueError(
            "Task A.1 requires protocol.target_transform.name='depth_norm_zscore_winsor' "
            "(per-slide depth normalization, per-(slide,gene) z-score, then winsorization). "
            f"Got {target_transform_name!r}."
        )
    if float(target_transform.get("target_sum", 0.0)) <= 0.0:
        raise ValueError("protocol.target_transform.target_sum must be a positive number.")
    zclip = target_transform.get("zscore_clip", None)
    if zclip is None or not math.isfinite(float(zclip)) or float(zclip) <= 0.0:
        raise ValueError(
            "protocol.target_transform.zscore_clip must be finite and > 0. "
            f"Got {zclip!r}."
        )
    # zscore is declarative (z-scoring cannot be disabled): it must be True
    if target_transform.get("zscore", None) is not True:
        raise ValueError(
            "protocol.target_transform.zscore must be True (the pipeline has no "
            "non-z-scored path)."
        )

    # Signature aggregation is declarative (always Sum w*z / Sum|w|)
    if str(cfg["protocol"].get("target_basis", "genes")) == "signatures":
        aggregation = str(
            cfg["protocol"].get("signature_target", {}).get("aggregation", "")
        ).strip().lower()
        if aggregation != "weighted_zscore_mean":
            raise ValueError(
                "Task A.1 hardcodes protocol.signature_target.aggregation="
                f"'weighted_zscore_mean' (Sum w*z / Sum|w|); the config says "
                f"{aggregation!r}."
            )

    # HEST CV: values implemented by the split builder
    hest_cfg = cfg["protocol"].get("hest_cv", {}) or {}
    hest_expected = {
        "primary_aggregation": "tissue_macro",
        "split_grouping": "patient_if_available_else_slide",
    }
    for key, expected in hest_expected.items():
        if key in hest_cfg and str(hest_cfg[key]).strip().lower() != expected:
            raise ValueError(
                f"Task A.1 implements protocol.hest_cv.{key}={expected!r}; the config says "
                f"{hest_cfg[key]!r}."
            )

    ridge_cfg = cfg["protocol"].get("ridge", None)
    if not isinstance(ridge_cfg, Mapping):
        raise ValueError("protocol.ridge is required (the A.1 probe is a closed-form ridge).")
    grid = ridge_cfg.get("alpha_grid", {})
    lo, hi = float(grid.get("lambda_min", 0.0)), float(grid.get("lambda_max", 0.0))
    if not (0.0 < lo < hi):
        raise ValueError(
            f"Require 0 < protocol.ridge.alpha_grid.lambda_min < lambda_max, got {lo!r} and {hi!r}."
        )
    if int(grid.get("n_points", 0)) < 2:
        raise ValueError("protocol.ridge.alpha_grid.n_points must be >= 2.")
    # Declarative ridge keys (hardcoded in the probe): must match the implementation
    ridge_expected = {"sample_weights": "slide_uniform", "fit_intercept": True}
    for key, expected in ridge_expected.items():
        if key not in ridge_cfg:
            raise ValueError(f"protocol.ridge.{key} is required; expected {expected!r}.")
        if ridge_cfg[key] != expected:
            raise ValueError(
                f"Task A.1's probe hardcodes protocol.ridge.{key}={expected!r}; the config "
                f"says {ridge_cfg[key]!r}."
            )
    # Declarative too: the grid is always lambda * mean eigenvalue
    grid_param = str(grid.get("parameterization", "")).strip().lower()
    if grid_param != "lambda_times_mean_eigenvalue":
        raise ValueError(
            "Task A.1 hardcodes protocol.ridge.alpha_grid.parameterization="
            f"'lambda_times_mean_eigenvalue'; the config says {grid_param!r}."
        )

    sel_cfg = ridge_cfg.get("alpha_selection", {})
    if not isinstance(sel_cfg, Mapping):
        raise ValueError("protocol.ridge.alpha_selection is required.")
    sel_expected = {
        "scheme": "leave_one_train_group_out",
        "score": "slide_macro_spearman",
        "rule": "paired_1se",
    }
    for key, expected in sel_expected.items():
        actual = str(sel_cfg.get(key, "")).strip().lower()
        if actual != expected:
            raise ValueError(
                f"protocol.ridge.alpha_selection.{key} must be {expected!r} (the probe "
                f"hardcodes it). Got {actual!r}."
            )

    pp_cfg = cfg["protocol"].get("input_preprocessing", {})
    # Every space accepted by the runner, with its geometry (as in modeling.resolve_feature_geometry)
    expected_pp = {"he": "identity", "he_l2": "row_l2", "clip": "row_l2"}
    for space, expected in expected_pp.items():
        if space not in pp_cfg:
            raise ValueError(
                f"protocol.input_preprocessing.{space} is required; expected {expected!r}."
            )
        actual = str(pp_cfg[space]).strip().lower()
        if actual not in VALID_INPUT_PREPROCESSING_STRATEGIES:
            raise ValueError(
                f"Unsupported protocol.input_preprocessing.{space}={actual!r}. "
                f"Expected one of {sorted(VALID_INPUT_PREPROCESSING_STRATEGIES)}."
            )
        if actual != expected:
            raise ValueError(
                "Task A.1 uses a fixed encoder-space preprocessing policy: "
                + ", ".join(f"{k}={v!r}" for k, v in expected_pp.items())
                + f". Got {space}={actual!r}."
            )

    validate_G_base_id_usage(
        eval_configuration=eval_configuration,
        sub_eval_configuration=sub_eval_configuration,
        G_base_id=G_base_id,
    )

    gene_selection = cfg["protocol"].get("gene_selection", {})

    gene_selection_G_base_id = str(gene_selection.get("G_base_id", G_base_id))
    if gene_selection_G_base_id != G_base_id:
        raise ValueError(
            "protocol.G_base_id and protocol.gene_selection.G_base_id must match. "
            f"Got protocol.G_base_id={G_base_id!r}, "
            f"gene_selection.G_base_id={gene_selection_G_base_id!r}."
        )

    if sub_eval_configuration in CONTROL_SUB_EVAL_CONFIGS:
        if eval_configuration != REFERENCE_CONTROL_EVAL_CONFIGURATION:
            raise ValueError(
                f"{sub_eval_configuration} must use eval_configuration="
                f"'{REFERENCE_CONTROL_EVAL_CONFIGURATION}'."
            )
        ref_id = cfg["protocol"].get("reference_eval_id", None)
        if ref_id is None or str(ref_id).strip() == "":
            raise ValueError(
                f"{sub_eval_configuration} is a control protocol and requires reference_eval_id "
                f"pointing to an existing {REFERENCE_CONTROL_EVAL_CONFIGURATION}/"
                f"{REFERENCE_CONTROL_SUB_EVAL_CONFIGURATION} eval_id."
            )
        expected_kind = CONTROL_KIND_BY_SUB_EVAL_CONFIG[sub_eval_configuration]
        actual_kind = str(cfg["protocol"].get("control_kind", ""))
        if actual_kind != expected_kind:
            raise ValueError(
                f"{sub_eval_configuration} must have control_kind='{expected_kind}', got '{actual_kind}'."
            )


def _infer_patch_overlap_px(dataset_id: str, training_dataset_id: str) -> int:
    for s in (str(training_dataset_id), str(dataset_id)):
        m = re.search(r"(?:^|_)o(\d+)(?:_|$)", s)
        if m is not None:
            return int(m.group(1))
    return 112


def default_eval_config(
    *,
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
    scgpt_model_name: Optional[str] = "scGPT_human",
    protocol_seed: Optional[int] = 20260319,
    novae_training_id: Optional[str] = None,
    novae_domain_n: Optional[int] = None,
    reference_eval_id: Optional[str] = None,
    eval_signatures_INT_id: Optional[str] = None,
    target_basis: str = "genes",
    signature_sets: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if scgpt_model_name is None:
        scgpt_model_name = "scGPT_human"
    if target_basis not in ("genes", "signatures"):
        raise ValueError(
            f"[ERROR] target_basis must be 'genes' or 'signatures', got '{target_basis}'."
        )
    if protocol_seed is None:
        protocol_seed = 20260319

    safe_border_px = _infer_patch_overlap_px(dataset_id, training_dataset_id)

    novae_training_id_clean = None
    if novae_training_id is not None and str(novae_training_id).strip():
        novae_training_id_clean = str(novae_training_id).strip()
    novae_domain_n_clean = None if novae_domain_n is None else int(novae_domain_n)

    cfg: Dict[str, Any] = {
        "identity": {
            "eval_id": eval_id,
            "dataset_id": dataset_id,
            "training_dataset_id": training_dataset_id,
            "created_utc": utc_now(),
        },
        "protocol": {
            "eval_configuration": eval_configuration,
            "sub_eval_configuration": sub_eval_configuration,
            "level": level,
            "modelgroup": modelgroup,
            "G_base_id": G_base_id,
            "gene_property_id": gene_property_id,
            "eval_signatures_id": eval_signatures_id,
            "eval_signatures_INT_id": eval_signatures_INT_id,
            # Prediction target: "genes" or "signatures" (aggregate scores of the signatures in signature_sets)
            "target_basis": target_basis,
            "signature_sets": list(signature_sets) if signature_sets else [],
            "signature_target": {
                # Weighted mean of the z-scored member genes (Sum w*z / Sum|w|)
                "aggregation": "weighted_zscore_mean",
                # Per-signature coverage gate, applied over the config train pool (member genes kept)
                # and per (slide, signature) against full membership (where a signature is scored)
                "coverage_gate": {
                    "min_covered_fraction": 0.5,
                    "min_covered_genes": 3,
                },
                # Signature-axis validity floors (the metrics block applies to the gene basis)
                "correlation_validity": {
                    "min_samples_per_signature_slide": 30,
                    "min_unique_target_values": 3,
                    # No nonzero floor (a signature aggregate has no on/off state)
                },
                "macro_aggregation_validity": {
                    "min_valid_signatures_per_slide_for_slide_macro": 2,
                    "min_valid_slides_per_signature_for_signature_macro": 2,
                },
                # Profile axis (cross-signature Spearman per location); min_positive is 0 (no "detected" signatures)
                "location_profile_validity": {
                    "min_signatures_per_location": 5,
                    "min_positive_signatures_per_location": 0,
                },
                # Minimum number of signatures for the structure analyses
                "structure_validity": {
                    "min_signatures_for_coexpression": 8,
                    "min_signatures_for_clustering": 12,
                },
                # Weak-train-slide filter: minimum number of eligible signatures per train slide
                "min_train_eligible_signatures_per_slide": 1,
            },
            "scgpt_model_name": scgpt_model_name,
            "protocol_seed": int(protocol_seed),
            "target_transform": {
                "name": "depth_norm_zscore_winsor",
                "target_sum": 1.0e4,
                "zscore": True,
                "zscore_clip": 5.0,
            },
            "head": {"type": "ridge"},
            "ridge": {
                "sample_weights": "slide_uniform",
                "fit_intercept": True,
                "alpha_grid": {
                    "parameterization": "lambda_times_mean_eigenvalue",
                    "lambda_min": 1.0e-6,
                    "lambda_max": 1.0e3,
                    "n_points": 19,
                },
                "alpha_selection": {
                    "scheme": "leave_one_train_group_out",
                    "score": "slide_macro_spearman",
                    "rule": "paired_1se",
                },
            },
            "input_preprocessing": {
                "he": "identity",
                "he_l2": "row_l2",
                "clip": "row_l2",
            },
            "metrics": {
                "huber_delta": 1.0,
                "correlation_validity": {
                    "min_samples_per_gene_slide": 30,
                    "min_nonzero_samples_per_gene_slide": 10,
                    "min_unique_target_values": 3,
                },
                "magnitude_validity": {"min_samples_per_gene_slide": 30},
                "location_profile_validity": {
                    "min_genes_per_location": 20,
                    "min_positive_genes_per_location": 5,
                    # Floor for a profile Spearman on a subset of the genes (gene stratum, signature members)
                    "min_genes_per_location_stratified": 5,
                },
                "macro_aggregation_validity": {
                    "min_valid_genes_per_slide_for_slide_macro": 5,
                    "min_valid_slides_per_gene_for_gene_macro": 2,
                },
                # Strata with fewer genes get no metrics (= paired_stats.MIN_N_FOR_WILCOXON)
                "min_genes_per_stratum": 6,
                "group_summaries": {
                    "min_slides_per_tissue": 3,
                    # Only used at cell level (by_cell_type diagnostic), not reached with patch_all
                    "min_cells_per_slide_cell_type": 50,
                },
                "structure_validity": {
                    "min_rows_for_coexpression": 100,
                    "min_genes_for_coexpression": 20,
                    "min_rows_for_clustering": 100,
                    "min_genes_for_clustering": 30,
                    "min_rows_for_moran": 100,
                    "min_positive_locations_for_moran": 20,
                    "auroc_min_pos": 10,
                    "auroc_min_neg": 10,
                },
            },
            "gene_selection": {
                "G_base_id": G_base_id,
                "auto_n_genes": 256,
                "auto_hest": {
                    "min_slides_per_gene": 3,
                    "min_rows_per_gene": 200,
                    "min_positive_rows_per_gene": 30,
                    "min_nonzero_rate": 0.001,
                    "min_unique_values_per_gene": 3,
                    "min_var_log1p": 1.0e-12,
                },
                "auto_sthelar_test_adapted": {
                    "min_rows_per_gene": 200,
                    "min_positive_rows_per_gene": 30,
                    "min_nonzero_rate": 0.001,
                    "min_unique_values_per_gene": 3,
                    "min_var_log1p": 1.0e-12,
                },
                "min_train_slides_per_gene": 3,
                "min_train_rows_per_gene": 200,
                "min_train_positive_rows_per_gene": 30,
                "min_trainable_genes": 32,
                "min_train_rows_per_gene_slide": 30,
                "min_train_positive_rows_per_gene_slide": 5,
                "min_train_unique_values_per_gene_slide": 3,
                "min_train_var_log1p_per_gene_slide": 1.0e-12,
                "min_train_rows_per_slide": 100,
                "min_train_eligible_genes_per_slide": 10,
            },
            "cv": {
                "n_outer_folds": 5,
                "safe_border_px": int(safe_border_px),
            },
            "hest_cv": {
                "primary_aggregation": "tissue_macro",
                "split_grouping": "patient_if_available_else_slide",
                "candidate_subsets": {
                    "HEST_CV_CLOSE": ["other_technologies", "other_preservations"],
                    "HEST_CV_MAGNIFICATION": ["other_magnifications"],
                    "HEST_CV_SPECIES": ["other_species"],
                    "HEST_ORGANS": ["other_organs"],
                },
                "min_slides_per_tissue": {
                    "HEST_CV_CLOSE": 10,
                    "HEST_CV_MAGNIFICATION": 10,
                    "HEST_CV_SPECIES": 10,
                    "HEST_ORGANS": 10,
                },
            },
            "structure": {
                "moran_neighbors": 8,
                "cluster_k": 8,
                "cluster_pca_dim": 20,
                "novae_training_id": novae_training_id_clean,
                "novae_domain_n": novae_domain_n_clean,
            },
            "reports": {
                "n_slides": 3,
                "n_genes": 6,
                "he_mosaic_downsample": 8,
                "he_mosaic_max_megapixels": 24,
                "max_plot_points": 60000,
                "patch_alpha": 0.55,
                "zscore_clip": 5.0,
                "gene_value_percentile": 99.0,
                "residual_abs_percentile": 98.0,
            },
            "runtime": {
                "wandb_enabled": True,
                "wandb_mode": "offline",
                "wandb_project": "HistoST_A1_eval",
            },
        },
    }

    if sub_eval_configuration in CONTROL_SUB_EVAL_CONFIGS:
        cfg["protocol"]["reference_eval_id"] = str(reference_eval_id) if reference_eval_id is not None else ""
        cfg["protocol"]["control_kind"] = CONTROL_KIND_BY_SUB_EVAL_CONFIG[sub_eval_configuration]

    # Adapt per sub-protocol
    if sub_eval_configuration == "STHELAR_TEST_ADAPTED":
        cfg["protocol"]["cv"]["n_outer_folds"] = 3

    if sub_eval_configuration in HEST_CV_SUB_EVAL_CONFIGS:
        cfg["protocol"]["cv"]["n_outer_folds"] = 3

    # Keep only the sections relevant for this protocol
    use_cv = sub_eval_configuration in {
        "STHELAR_TRAIN_CV",
        "STHELAR_TRAIN_COORDS",
        "STHELAR_TEST_ADAPTED",
        "HEST_CV_CLOSE",
        "HEST_CV_MAGNIFICATION",
        "HEST_CV_SPECIES",
        "HEST_ORGANS",
    }
    use_hest_cv = sub_eval_configuration in HEST_CV_SUB_EVAL_CONFIGS

    if not use_cv:
        cfg["protocol"].pop("cv", None)
    if not use_hest_cv:
        cfg["protocol"].pop("hest_cv", None)

    validate_protocol_identity(cfg)
    return cfg


def dump_environment_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "utc_now": utc_now(),
    }
    git_commit = maybe_git_commit()
    if git_commit is not None:
        report["git_commit"] = git_commit
    if torch is not None:
        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["mps_available"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    return report
