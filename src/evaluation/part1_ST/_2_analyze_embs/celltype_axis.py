"""Cell-type label axes (fine, versioned coarse) and column names for the Task A-0 embedding analyses."""
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# Cell-type axes of the A-0 analyses (each one is reported separately)
VALID_A0_CELLTYPE_AXES = ("fine", "coarse")

# Accepted coarse_mapping_version strings (the version is part of a filename)
COARSE_MAPPING_VERSION_RE = re.compile(r"^v[0-9]+$")

# Separator of the wide per-(metric, axis, cell type) column names (cell types may contain "_")
AXIS_COLUMN_SEP = "__"

# Label of a cell with no row in the ct_cat_id annotation table (distinct from the `Unknown` cell type);
# mapped to itself on every axis
MISSING_ANNOTATION_LABELS: Tuple[str, ...] = ("UNKNOWN",)


# Labels excluded from every scored statistic: the `Unknown` cell type and the missing-annotation label
EXCLUDED_FROM_SCORED: Tuple[str, ...] = ("Unknown", "UNKNOWN")

# Suffix of the scored shuffled-baseline flavours (`_2`)
SCORED_FLAVOUR_SUFFIX = "scored"


def scored_flavour(flavour: str) -> str:
    """`sameCT` -> `sameCTscored`. The scored twin of a cell-type-defined shuffled flavour."""
    return f"{flavour}{SCORED_FLAVOUR_SUFFIX}"


def is_scored_flavour(flavour: str) -> bool:
    """True for a flavour minted by `scored_flavour`."""
    return str(flavour).endswith(SCORED_FLAVOUR_SUFFIX)


def scored_mean_column(axis: str) -> str:
    """
    Name of the per-axis column `cell_sim_mean_scored__{axis}`: mean over the scored rows (minuend of the scored gaps).
    """
    return f"cell_sim_mean_scored{AXIS_COLUMN_SEP}{axis}"


# Pair scope of the `_1` cell-type cosine matrix:
#   pooled        both members drawn from the cross-slide pool
#   within_slide  both members drawn from one slide
PAIR_SCOPE_POOLED = "pooled"
PAIR_SCOPE_WITHIN_SLIDE = "within_slide"
PAIR_SCOPES: Tuple[str, ...] = (PAIR_SCOPE_POOLED, PAIR_SCOPE_WITHIN_SLIDE)


# Headroom-normalised variant of the axis-free `any` shuffled gap (`_2`)
HEADROOM_NORM_GAP_COLUMN = "headroom_norm_gap_any"


def normalize_coarse_mapping_version(version: Any) -> str:
    """Validate + normalize a `coarse_mapping_version` string (e.g. 'v1'). Raises ValueError."""
    value = str(version).strip()
    if not COARSE_MAPPING_VERSION_RE.match(value):
        raise ValueError(
            f"Invalid coarse_mapping_version={version!r}. Expected a string matching "
            f"{COARSE_MAPPING_VERSION_RE.pattern} (e.g. 'v1')."
        )
    return value


def coarse_mapping_filename(version: Any) -> str:
    """Filename of the versioned coarse mapping for a `coarse_mapping_version`."""
    return f"coarse_ct_mapping_{normalize_coarse_mapping_version(version)}.json"


def _repo_root() -> Path:
    """Repo root (holds data/annots/)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "utils" / "constants.py").exists():
            return parent
    # Fallback: <repo>/src/evaluation/part1_ST/_2_analyze_embs/celltype_axis.py
    return here.parents[4]


def annot_dir(ct_cat_id: str) -> Path:
    """Directory of the cell-type annotation dicts of a ct_cat_id."""
    return _repo_root() / "data" / "annots" / f"annot_dicts_{ct_cat_id}"


def load_coarse_ct_mapping(ct_cat_id: str, version: Any) -> Dict[str, str]:
    """
    Load the fine -> coarse cell-type map of one (ct_cat_id, coarse_mapping_version).
    Raises on a missing file, an empty key or value, or conflicting duplicate keys.
    """
    directory = annot_dir(ct_cat_id)
    if not directory.is_dir():
        raise FileNotFoundError(f"Annotation-dict directory not found: {directory}")
    filename = coarse_mapping_filename(version)
    path = directory / filename
    if not path.is_file():
        available = sorted(
            p.name for p in directory.glob("coarse_ct_mapping_*.json") if not p.name.startswith(".")
        )
        raise FileNotFoundError(
            f"Missing coarse mapping '{filename}' under {directory}. "
            f"Available versions: {available or 'none'}."
        )

    payload: Any = json.loads(path.read_text())
    if not isinstance(payload, Mapping) or len(payload) == 0:
        raise ValueError(
            f"{path} must be a non-empty JSON object {{fine_label: coarse_label}}, "
            f"got: {type(payload).__name__}"
        )

    out: Dict[str, str] = {}
    for key, value in payload.items():
        fine = str(key).strip()
        coarse = str(value).strip()
        if not fine or not coarse:
            raise ValueError(f"{path} has an empty key or value: {key!r} -> {value!r}")
        if fine in out and out[fine] != coarse:
            raise ValueError(f"{path} has conflicting entries for {fine!r}: {out[fine]!r} vs {coarse!r}")
        out[fine] = coarse
    return out


def normalize_axes(axes: Optional[Iterable[Any]]) -> List[str]:
    """Validate a requested axis list (at least one axis), preserving order and dropping duplicates."""
    if axes is None:
        raise ValueError("celltype_axes must name at least one axis; got None.")
    seen: List[str] = []
    for raw in axes:
        value = str(raw).strip().lower()
        if value not in VALID_A0_CELLTYPE_AXES:
            raise ValueError(
                f"Invalid celltype axis={raw!r}. Expected one of {sorted(VALID_A0_CELLTYPE_AXES)}."
            )
        if value not in seen:
            seen.append(value)
    if not seen:
        raise ValueError("celltype_axes must name at least one axis; got an empty list.")
    return seen


def axis_token(granularity: Any, coarse_mapping_version: Any = None) -> str:
    """Token naming one axis: `fine`, or `coarse_v1` / `coarse_v2` / ..."""
    value = str(granularity).strip().lower()
    if value not in VALID_A0_CELLTYPE_AXES:
        raise ValueError(
            f"Invalid celltype axis={granularity!r}. Expected one of {sorted(VALID_A0_CELLTYPE_AXES)}."
        )
    if value == "fine":
        return "fine"
    if coarse_mapping_version is None or not str(coarse_mapping_version).strip():
        raise ValueError(
            "celltype axis 'coarse' requires an explicit coarse_mapping_version (e.g. 'v1')."
        )
    return f"coarse_{normalize_coarse_mapping_version(coarse_mapping_version)}"


def axis_column(metric: str, axis: str, cell_type: str) -> str:
    """Wide per-(metric, axis, cell type) column name: `{metric}__{axis}__{cell_type}`."""
    return f"{metric}{AXIS_COLUMN_SEP}{axis}{AXIS_COLUMN_SEP}{cell_type}"


def shuffled_ct_column(flavour: str, axis: Optional[str] = None) -> str:
    """
    Name of a per-axis shuffled-baseline column, e.g. `shuffled_cos_sameCT__coarse_v1_mean`.
    `axis=None` gives the unsuffixed name (`shuffled_cos_sameCT_mean`), used for the primary axis.
    """
    suffix = "" if axis is None else f"{AXIS_COLUMN_SEP}{axis}"
    return f"shuffled_cos_{flavour}{suffix}_mean"


def shuffled_gap_column(flavour: str, axis: Optional[str] = None) -> str:
    """Name of the matching `cell_sim_mean - shuffled_cos_*` gap column (axis suffix at the end)."""
    suffix = "" if axis is None else f"{AXIS_COLUMN_SEP}{axis}"
    return f"cell_sim_mean_minus_shuffled_{flavour}{suffix}"


def parse_shuffled_ct_column(column: str) -> Optional[Tuple[str, Optional[str]]]:
    """Inverse of `shuffled_ct_column`: `column -> (flavour, axis_or_None)`, else None."""
    prefix, suffix = "shuffled_cos_", "_mean"
    if not (column.startswith(prefix) and column.endswith(suffix)):
        return None
    remainder = column[len(prefix): -len(suffix)]
    if not remainder:
        return None
    if AXIS_COLUMN_SEP in remainder:
        flavour, axis = remainder.split(AXIS_COLUMN_SEP, 1)
        return flavour, axis
    return remainder, None


def parse_shuffled_gap_column(column: str) -> Optional[Tuple[str, Optional[str]]]:
    """Inverse of `shuffled_gap_column`: `column -> (flavour, axis_or_None)`, else None."""
    prefix = "cell_sim_mean_minus_shuffled_"
    if not column.startswith(prefix):
        return None
    remainder = column[len(prefix):]
    if not remainder:
        return None
    if AXIS_COLUMN_SEP in remainder:
        flavour, axis = remainder.split(AXIS_COLUMN_SEP, 1)
        return flavour, axis
    return remainder, None


def build_axes(
    ct_cat_id: str,
    axes: Optional[Iterable[Any]] = None,
    coarse_mapping_version: Any = None,
) -> Dict[str, Optional[Dict[str, str]]]:
    """
    Resolve a requested axis list to `{axis_token: mapping_or_None}` (None = identity, for `fine`), in order.
    `coarse_mapping_version` is required when `coarse` is requested.
    """
    requested = normalize_axes(axes if axes is not None else VALID_A0_CELLTYPE_AXES)
    out: Dict[str, Optional[Dict[str, str]]] = {}
    for granularity in requested:
        token = axis_token(granularity, coarse_mapping_version)
        if granularity == "fine":
            out[token] = None
            continue
        mapping = load_coarse_ct_mapping(ct_cat_id, coarse_mapping_version)
        out[token] = mapping
        logger.info(
            "A-0 cell-type axis '%s' for ct_cat_id=%s: %d fine labels -> %d classes %s.",
            token,
            ct_cat_id,
            len(mapping),
            len(set(mapping.values())),
            sorted(set(mapping.values())),
        )
    logger.info("A-0 cell-type axes resolved: %s", list(out))
    return out


LabelValues = Union[pd.Series, np.ndarray, list, tuple]


def _effective_mapping(mapping: Mapping[str, str]) -> Dict[str, str]:
    """Coarse mapping plus the missing-annotation labels mapped to themselves."""
    out = dict(mapping)
    for sentinel in MISSING_ANNOTATION_LABELS:
        out.setdefault(sentinel, sentinel)
    return out


def apply_label_map(values: LabelValues, mapping: Optional[Mapping[str, str]]) -> LabelValues:
    """
    Map observed cell-type labels through `mapping` (None = identity); raises on an unmapped label.
    A `pd.Series` in gives a `pd.Series` out (index preserved); anything else gives an object-dtype `np.ndarray`.
    """
    is_series = isinstance(values, pd.Series)
    series = values.astype(str).str.strip() if is_series else pd.Series(
        np.asarray(values, dtype=object).reshape(-1), dtype=object
    ).astype(str).str.strip()

    if mapping is None:
        return series if is_series else series.to_numpy(dtype=object)

    mapped = series.map(_effective_mapping(mapping))
    unmapped = mapped.isna()
    if bool(unmapped.any()):
        missing = sorted(set(series[unmapped].tolist()))
        raise ValueError(
            "Observed cell-type labels are absent from the coarse mapping: "
            f"{missing}. Every label of the ct_cat_id vocabulary must be a key of the versioned "
            "coarse_ct_mapping_{version}.json in use "
            f"(the only exceptions are the missing-annotation sentinels {list(MISSING_ANNOTATION_LABELS)})."
        )

    return mapped if is_series else mapped.to_numpy(dtype=object)


def apply_label_map_to_dict(
    cell_id_to_type: Mapping[int, str], mapping: Optional[Mapping[str, str]]
) -> Dict[int, str]:
    """Map a `{cell_id_int: cell_type}` dict through `mapping` (same checks as `apply_label_map`). Returns a new dict."""
    if mapping is None:
        return {int(k): str(v).strip() for k, v in cell_id_to_type.items()}
    effective = _effective_mapping(mapping)
    out: Dict[int, str] = {}
    missing = set()
    for key, value in cell_id_to_type.items():
        label = str(value).strip()
        target = effective.get(label)
        if target is None:
            missing.add(label)
            continue
        out[int(key)] = target
    if missing:
        raise ValueError(
            "Observed cell-type labels are absent from the coarse mapping: "
            f"{sorted(missing)}. Every label of the ct_cat_id vocabulary must be a key of the "
            "versioned coarse_ct_mapping_{version}.json in use "
            f"(the only exceptions are the missing-annotation sentinels {list(MISSING_ANNOTATION_LABELS)})."
        )
    return out


def assert_vocabulary_covered(
    observed_labels: Iterable[Any],
    axes: Mapping[str, Optional[Mapping[str, str]]],
    *,
    where: str = "",
) -> List[str]:
    """
    Check that every observed cell-type label is mapped on every requested axis (raises listing all
    missing (axis, label) pairs). Returns the sorted vocabulary checked.
    """
    vocabulary = sorted({str(v).strip() for v in observed_labels if str(v).strip()})
    if not vocabulary:
        return vocabulary

    problems: Dict[str, List[str]] = {}
    for axis, mapping in axes.items():
        if mapping is None:            # `fine` is the identity
            continue
        effective = _effective_mapping(mapping)
        missing = sorted(label for label in vocabulary if label not in effective)
        if missing:
            problems[axis] = missing

    if problems:
        detail = "; ".join(f"axis '{axis}': {labels}" for axis, labels in sorted(problems.items()))
        raise ValueError(
            f"Cell-type vocabulary is not covered by the requested axes{f' ({where})' if where else ''}. "
            f"{detail}. Every label of the ct_cat_id vocabulary must be a key of the versioned "
            "coarse_ct_mapping_{version}.json in use (the only exceptions are the "
            f"missing-annotation sentinels {list(MISSING_ANNOTATION_LABELS)})."
        )

    logger.info(
        "A-0 cell-type vocabulary preflight OK%s: %d label(s) covered on every axis %s.",
        f" ({where})" if where else "",
        len(vocabulary),
        list(axes),
    )
    return vocabulary


def axis_cell_types(
    unique_cell_types: Sequence[str], mapping: Optional[Mapping[str, str]]
) -> List[str]:
    """Map a cell-type vocabulary to one axis, preserving first-appearance order (raises on an unmapped label)."""
    if mapping is None:
        return [str(t).strip() for t in unique_cell_types]
    effective = _effective_mapping(mapping)
    out: List[str] = []
    missing = set()
    for raw in unique_cell_types:
        label = str(raw).strip()
        target = effective.get(label)
        if target is None:
            missing.add(label)
            continue
        if target not in out:
            out.append(target)
    if missing:
        raise ValueError(
            "Observed cell-type labels are absent from the coarse mapping: "
            f"{sorted(missing)}. Every label of the ct_cat_id vocabulary must be a key of the "
            "versioned coarse_ct_mapping_{version}.json in use "
            f"(the only exceptions are the missing-annotation sentinels {list(MISSING_ANNOTATION_LABELS)})."
        )
    return out
