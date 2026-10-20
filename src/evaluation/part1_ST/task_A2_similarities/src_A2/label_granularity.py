"""Task A.2 cell-type label granularity: fine labels, or fine labels collapsed with the versioned coarse mapping."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# Label granularities of the A-2 protocol
VALID_A2_LABEL_GRANULARITY = ("coarse", "fine")

# Accepted coarse_mapping_version strings (the version is used in a file name)
COARSE_MAPPING_VERSION_RE = re.compile(r"^v[0-9]+$")


def normalize_coarse_mapping_version(version: Any) -> str:
    """Validate + normalize a `coarse_mapping_version` string (e.g. 'v1'). Raises ValueError."""
    value = str(version).strip()
    if not COARSE_MAPPING_VERSION_RE.match(value):
        raise ValueError(
            f"Invalid coarse_mapping_version={version!r}. Expected a string matching "
            f"{COARSE_MAPPING_VERSION_RE.pattern} (e.g. 'v1' or 'v2')."
        )
    return value


def coarse_mapping_filename(version: Any) -> str:
    """Filename of the versioned coarse mapping for a `coarse_mapping_version`."""
    return f"coarse_ct_mapping_{normalize_coarse_mapping_version(version)}.json"


def _repo_root() -> Path:
    """Repo root that holds data/annots/ (found by walking up to src/utils/constants.py)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "utils" / "constants.py").exists():
            return parent
    # Fallback: fixed location of this file in the repo
    return here.parents[5]


def annot_dir(ct_cat_id: str) -> Path:
    """Directory of the cell-type annotation dicts of a ct_cat_id (in the repo)."""
    return _repo_root() / "data" / "annots" / f"annot_dicts_{ct_cat_id}"


def load_coarse_ct_mapping(ct_cat_id: str, version: Any) -> Dict[str, str]:
    """Load the fine -> coarse label map of one (ct_cat_id, coarse_mapping_version) (raises on conflicting duplicate keys)."""
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


def normalize_granularity(granularity: Any) -> str:
    """Normalize + validate a granularity string; raises ValueError on anything unsupported."""
    value = str(granularity).strip().lower()
    if value not in VALID_A2_LABEL_GRANULARITY:
        raise ValueError(
            f"Invalid label_granularity={granularity!r}. "
            f"Expected one of {sorted(VALID_A2_LABEL_GRANULARITY)}."
        )
    return value


def resolve_label_map(
    ct_cat_id: str, granularity: Any, coarse_mapping_version: Any = None
) -> Optional[Dict[str, str]]:
    """
    Label map of one (ct_cat_id, granularity, coarse_mapping_version): None for fine (labels used as they are),
    the fine -> coarse map for coarse (coarse_mapping_version required).
    """
    value = normalize_granularity(granularity)
    if value == "fine":
        return None
    if coarse_mapping_version is None or not str(coarse_mapping_version).strip():
        raise ValueError(
            "label_granularity='coarse' requires an explicit coarse_mapping_version "
            "(identity.coarse_mapping_version, e.g. 'v1')."
        )
    mapping = load_coarse_ct_mapping(ct_cat_id, coarse_mapping_version)
    logger.info(
        "Resolved A-2 label_granularity=coarse (coarse_mapping_version=%s) for ct_cat_id=%s: "
        "%d fine labels -> %d coarse labels %s.",
        normalize_coarse_mapping_version(coarse_mapping_version),
        ct_cat_id,
        len(mapping),
        len(set(mapping.values())),
        sorted(set(mapping.values())),
    )
    return mapping


LabelValues = Union[pd.Series, np.ndarray, list, tuple]


def apply_label_map(values: LabelValues, mapping: Optional[Mapping[str, str]]) -> LabelValues:
    """
    Collapse observed mapped cell-type labels through mapping (identity if None, after whitespace stripping).
    Raises on labels absent from the mapping. Returns a pd.Series for a Series input, else an object np.ndarray.
    """
    is_series = isinstance(values, pd.Series)
    series = values.astype(str).str.strip() if is_series else pd.Series(
        np.asarray(values, dtype=object).reshape(-1), dtype=object
    ).astype(str).str.strip()

    if mapping is None:
        return series if is_series else series.to_numpy(dtype=object)

    mapped = series.map(dict(mapping))
    unmapped = mapped.isna()
    if bool(unmapped.any()):
        missing = sorted(set(series[unmapped].tolist()))
        raise ValueError(
            "Observed mapped cell-type labels are absent from the coarse mapping: "
            f"{missing}. Every label of the ct_cat_id vocabulary must be a key of the versioned "
            "coarse_ct_mapping_{version}.json in use."
        )

    return mapped if is_series else mapped.to_numpy(dtype=object)


def collapse_label_counts(
    counts: Mapping[str, int], mapping: Optional[Mapping[str, str]]
) -> Dict[str, int]:
    """Collapse a {label: n_cells} count dict through mapping, summing the members of each coarse class (copy if mapping is None)."""
    if mapping is None:
        return {str(k): int(v) for k, v in counts.items()}

    out: Dict[str, int] = {}
    missing = sorted({str(k) for k in counts if str(k) not in mapping})
    if missing:
        raise ValueError(
            "Observed reference cell-type labels are absent from the coarse mapping: "
            f"{missing}. Every label of the ct_cat_id vocabulary must be a key of the versioned "
            "coarse_ct_mapping_{version}.json in use."
        )
    for label, n in counts.items():
        coarse = str(mapping[str(label)])
        out[coarse] = out.get(coarse, 0) + int(n)
    return dict(sorted(out.items()))
