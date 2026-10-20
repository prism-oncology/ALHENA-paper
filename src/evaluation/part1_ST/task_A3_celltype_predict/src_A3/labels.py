"""Label space, cell-type id mapping and class support for Task A.3."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src_A3.common import ensure_imports

ensure_imports()

import numpy as np
import pandas as pd

from src_A3.common import load_json


logger = logging.getLogger(__name__)


# Labels never scored ("Unknown" and stringified nulls)
UNSCORABLE_LABELS = {"Unknown", "nan", "NaN", "None", ""}

# Accepted coarse_mapping_version strings (used in a file name, so no path separators)
COARSE_MAPPING_VERSION_RE = re.compile(r"^v[0-9]+$")


def _repo_root() -> Path:
    """Repo root that holds data/annots/ (found by walking up to src/utils/constants.py)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "utils" / "constants.py").exists():
            return parent
    # Fallback for <repo>/src/evaluation/part1_ST/task_A3_celltype_predict/src_A3/labels.py
    return here.parents[5]


def ANNOT_DIR(ct_cat_id: str) -> Path:
    """Directory of the frozen cell-type annotation dicts for a ct_cat_id (in-repo metadata)."""
    return _repo_root() / "data" / "annots" / f"annot_dicts_{ct_cat_id}"


def load_annot_dicts(ct_cat_id: str) -> Dict[str, Any]:
    """Load the annotation dicts {cat2idx, cat2color} of a ct_cat_id (coarse mapping: see load_coarse_ct_mapping)."""
    annot_dir = ANNOT_DIR(ct_cat_id)
    if not annot_dir.is_dir():
        raise FileNotFoundError(f"Annotation-dict directory not found: {annot_dir}")
    filenames = {
        "cat2idx": "cat2idx.json",
        "cat2color": "cat2color.json",
    }
    out: Dict[str, Any] = {}
    for key, fname in filenames.items():
        path = annot_dir / fname
        if not path.is_file():
            raise FileNotFoundError(f"Missing annotation dict '{fname}' under {annot_dir}")
        out[key] = load_json(path)
    return out


def normalize_coarse_mapping_version(version: Any) -> str:
    """Validate + normalize a `coarse_mapping_version` string (e.g. 'v1'). Raises ValueError."""
    value = str(version).strip()
    if not COARSE_MAPPING_VERSION_RE.match(value):
        raise ValueError(
            f"Invalid coarse_mapping_version={version!r}. Expected a string matching "
            f"{COARSE_MAPPING_VERSION_RE.pattern} (e.g. 'v1', 'v2')."
        )
    return value


def coarse_mapping_filename(version: Any) -> str:
    """Filename of the versioned coarse mapping for a `coarse_mapping_version`."""
    return f"coarse_ct_mapping_{normalize_coarse_mapping_version(version)}.json"


def load_coarse_ct_mapping(ct_cat_id: str, version: Any) -> Dict[str, str]:
    """Load the canonical-fine -> coarse map of one (ct_cat_id, coarse_mapping_version); raises on conflicting entries."""
    annot_dir = ANNOT_DIR(ct_cat_id)
    if not annot_dir.is_dir():
        raise FileNotFoundError(f"Annotation-dict directory not found: {annot_dir}")
    fname = coarse_mapping_filename(version)
    path = annot_dir / fname
    if not path.is_file():
        available = sorted(p.name for p in annot_dir.glob("coarse_ct_mapping_*.json")
                           if not p.name.startswith("."))
        raise FileNotFoundError(
            f"Missing coarse mapping '{fname}' under {annot_dir}. "
            f"Available versions: {available or 'none'}."
        )
    payload = load_json(path)
    if not isinstance(payload, dict) or not payload:
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


def _is_unscorable(label: Any) -> bool:
    return str(label).strip() in UNSCORABLE_LABELS


def canonical_fine_labels(ct_cat_id: str) -> List[str]:
    """Canonical fine categories (cat2idx keys) in ascending index order (includes Unknown)."""
    cat2idx = load_annot_dicts(ct_cat_id)["cat2idx"]
    return [str(lab) for lab, _ in sorted(cat2idx.items(), key=lambda kv: int(kv[1]))]


@dataclass(frozen=True)
class LabelSpace:
    """
    Frozen, dense-reindexed class vocabulary for one (ct_cat_id, granularity).

    Fields:
      - ct_cat_id: cell-type category id (e.g. "ct_1").
      - granularity: "fine" or "coarse".
      - labels: scored class labels in dense-id order (index i == class id i).
      - label_to_index: {label -> dense id 0..C-1}.
      - coarse_mapping: canonical-fine -> coarse map (None for fine).
      - dropped_labels: candidate labels of this granularity that failed the support filter.
      - coarse_mapping_version: version of the coarse mapping ("v1", "v2", ...); None for fine.
    """

    ct_cat_id: str
    granularity: str
    labels: List[str]
    label_to_index: Dict[str, int]
    coarse_mapping: Optional[Dict[str, str]]
    dropped_labels: List[str]
    coarse_mapping_version: Optional[str] = None

    def n_classes(self) -> int:
        return len(self.labels)

    def as_dict(self) -> Dict[str, Any]:
        """Serialization for labels/class_vocabulary.json (coarse_mapping included to reload a coarse LabelSpace)."""
        return {
            "ct_cat_id": self.ct_cat_id,
            "label_granularity": self.granularity,
            "coarse_mapping_version": self.coarse_mapping_version,
            "labels": list(self.labels),
            "label_to_index": {str(k): int(v) for k, v in self.label_to_index.items()},
            "coarse_mapping": (
                None
                if self.coarse_mapping is None
                else {str(k): str(v) for k, v in self.coarse_mapping.items()}
            ),
            "dropped_labels": list(self.dropped_labels),
        }


def derive_fine_candidate_labels(ct_cat_id: str) -> List[str]:
    """Canonical fine labels (cat2idx order) minus the unscorable universe (drops Unknown)."""
    return [lab for lab in canonical_fine_labels(ct_cat_id) if not _is_unscorable(lab)]


def derive_coarse_candidate_labels(
    ct_cat_id: str, version: Any
) -> Tuple[List[str], Dict[str, str]]:
    """
    Coarse candidate labels (first-appearance order in cat2idx) and the canonical-fine -> coarse map for one
    coarse_mapping_version. Fine labels that are unscorable or map to an unscorable coarse label are dropped.
    """
    coarse_map_raw = load_coarse_ct_mapping(ct_cat_id, version)
    canonical_to_coarse: Dict[str, str] = {}
    coarse_labels: List[str] = []
    for fine in canonical_fine_labels(ct_cat_id):
        if _is_unscorable(fine):
            continue
        if fine not in coarse_map_raw:
            raise KeyError(
                f"{coarse_mapping_filename(version)} for ct_cat_id={ct_cat_id!r} is missing "
                f"canonical label {fine!r}. Every fine label must be a key of the mapping."
            )
        coarse = str(coarse_map_raw[fine]).strip()
        if _is_unscorable(coarse):
            continue
        canonical_to_coarse[fine] = coarse
        if coarse not in coarse_labels:
            coarse_labels.append(coarse)
    return coarse_labels, canonical_to_coarse


def _candidate_labels_for_granularity(
    ct_cat_id: str, granularity: str, coarse_mapping_version: Any = None
) -> Tuple[str, List[str], Optional[Dict[str, str]]]:
    """(normalized granularity, ordered candidate labels, canonical->coarse map or None).

    `coarse_mapping_version` is required for `coarse` and ignored for `fine`.
    """
    granularity = str(granularity).strip().lower()
    if granularity == "fine":
        return granularity, derive_fine_candidate_labels(ct_cat_id), None
    if granularity == "coarse":
        if coarse_mapping_version is None:
            raise ValueError(
                "granularity='coarse' requires an explicit coarse_mapping_version "
                "(protocol.coarse_mapping_version, e.g. 'v1')."
            )
        candidates, canonical_to_coarse = derive_coarse_candidate_labels(
            ct_cat_id, coarse_mapping_version
        )
        return granularity, candidates, canonical_to_coarse
    raise ValueError(f"Invalid granularity={granularity!r}; expected 'fine' or 'coarse'.")


def build_label_space(
    *,
    ct_cat_id: str,
    granularity: str,
    trainable_labels: Sequence[str],
    coarse_mapping_version: Any = None,
) -> LabelSpace:
    """
    Freeze the class vocabulary over trainable_labels (already support-filtered), in candidate order,
    dense-reindexed 0..C-1. coarse_mapping_version is required for coarse.
    """
    granularity, candidate_order, coarse_mapping = _candidate_labels_for_granularity(
        ct_cat_id, granularity, coarse_mapping_version
    )
    candidate_set = set(candidate_order)
    trainable_set = {str(x).strip() for x in trainable_labels}

    not_candidates = sorted(t for t in trainable_set if t not in candidate_set)
    if not_candidates:
        raise ValueError(
            f"trainable_labels contains labels absent from the {granularity} candidate "
            f"vocabulary for ct_cat_id={ct_cat_id!r}: {not_candidates}. "
            f"Candidates: {candidate_order}."
        )

    labels = [c for c in candidate_order if c in trainable_set]
    if not labels:
        raise ValueError(
            "build_label_space received no trainable labels intersecting the candidate vocabulary."
        )
    label_to_index = {lab: i for i, lab in enumerate(labels)}
    dropped_labels = [c for c in candidate_order if c not in trainable_set]

    logger.debug(
        "Built %s LabelSpace (ct_cat_id=%s): %d classes %s; dropped %s.",
        granularity,
        ct_cat_id,
        len(labels),
        labels,
        dropped_labels,
    )
    return LabelSpace(
        ct_cat_id=str(ct_cat_id),
        granularity=granularity,
        labels=labels,
        label_to_index=label_to_index,
        coarse_mapping=coarse_mapping,
        dropped_labels=dropped_labels,
        # Normalized version (None for fine)
        coarse_mapping_version=(
            normalize_coarse_mapping_version(coarse_mapping_version)
            if granularity == "coarse"
            else None
        ),
    )


def map_cell_types_to_ids(cell_types: np.ndarray, label_space: LabelSpace) -> np.ndarray:
    """
    Map canonical cell-type strings to the dense class ids of a LabelSpace (through coarse_mapping for coarse).
    Unscorable and out-of-vocabulary labels -> -1. Returns int32 [N].
    """
    arr = np.atleast_1d(np.asarray(cell_types, dtype=object))
    n = int(arr.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.int32)

    s = pd.Series(arr, dtype=object).astype(str).str.strip()
    scorable = ~s.isin(UNSCORABLE_LABELS)
    if label_space.coarse_mapping is not None:
        key = s.map(label_space.coarse_mapping)  # canonical fine -> coarse; OOV -> NaN
    else:
        key = s
    ids = key.map(label_space.label_to_index)  # dense id or NaN
    ids = ids.where(scorable, other=np.nan)  # unscorable -> NaN
    vals = ids.to_numpy(dtype="float64")
    vals = np.where(np.isfinite(vals), vals, -1.0)
    return vals.astype(np.int32)


def class_support_table(
    rows_df: pd.DataFrame,
    *,
    ct_cat_id: str,
    granularity: str,
    coarse_mapping_version: Any = None,
) -> pd.DataFrame:
    """
    Support of every candidate class over the given train-pool rows: columns [class, n_cells, n_slides]
    (0 if absent). Unscorable / out-of-vocab cells are ignored.
    """
    for col in ("cell_type", "slide_id"):
        if col not in rows_df.columns:
            raise KeyError(
                f"class_support_table requires column {col!r} in rows_df; got {list(rows_df.columns)}."
            )
    granularity, candidates, canonical_to_coarse = _candidate_labels_for_granularity(
        ct_cat_id, granularity, coarse_mapping_version
    )
    if granularity == "fine":
        canonical_to_class: Dict[str, str] = {c: c for c in candidates}
    else:
        canonical_to_class = dict(canonical_to_coarse or {})

    cell_type = rows_df["cell_type"].astype(str).str.strip()
    mapped_class = cell_type.map(canonical_to_class)  # candidate class or NaN
    full = pd.DataFrame(
        {
            "class": mapped_class.to_numpy(dtype=object),
            "slide_id": rows_df["slide_id"].astype(str).to_numpy(),
        }
    )
    df = full[full["class"].notna()]

    if len(df) == 0:
        n_cells_by = pd.Series(dtype="int64")
        n_slides_by = pd.Series(dtype="int64")
    else:
        grouped = df.groupby("class")
        n_cells_by = grouped.size()
        n_slides_by = grouped["slide_id"].nunique()

    records = [
        {
            "class": cls,
            "n_cells": int(n_cells_by.get(cls, 0)),
            "n_slides": int(n_slides_by.get(cls, 0)),
        }
        for cls in candidates
    ]
    return pd.DataFrame(records, columns=["class", "n_cells", "n_slides"])


def select_trainable_classes(
    support_df: pd.DataFrame,
    *,
    min_cells: int,
    min_slides: int,
    min_classes: int,
) -> List[str]:
    """
    Keep candidate classes meeting both support thresholds (n_cells and n_slides), preserving
    the support_df (candidate) order. Raise RuntimeError if fewer than min_classes survive.
    """
    keep = [
        str(row["class"])
        for _, row in support_df.iterrows()
        if int(row["n_cells"]) >= int(min_cells) and int(row["n_slides"]) >= int(min_slides)
    ]
    if len(keep) < int(min_classes):
        raise RuntimeError(
            f"Only {len(keep)} class(es) meet the support thresholds "
            f"(min_train_cells_per_class={min_cells}, min_train_slides_per_class={min_slides}); "
            f"need >= {min_classes}. Support table: {support_df.to_dict(orient='records')}"
        )
    logger.debug("Selected %d trainable class(es): %s", len(keep), keep)
    return keep


def class_per_slide_support_table(
    rows_df: pd.DataFrame,
    *,
    ct_cat_id: str,
    granularity: str,
    min_cells_per_slide: int,
    coarse_mapping_version: Any = None,
) -> pd.DataFrame:
    """
    Within-slide support of every candidate class (STHELAR_TEST_ADAPTED): columns [class, n_cells, n_slides,
    n_slides_meeting_floor (slides with >= min_cells_per_slide cells), max_cells_in_slide].
    """
    for col in ("cell_type", "slide_id"):
        if col not in rows_df.columns:
            raise KeyError(
                f"class_per_slide_support_table requires column {col!r} in rows_df; "
                f"got {list(rows_df.columns)}."
            )
    min_cells_per_slide = int(min_cells_per_slide)
    granularity, candidates, canonical_to_coarse = _candidate_labels_for_granularity(
        ct_cat_id, granularity, coarse_mapping_version
    )
    if granularity == "fine":
        canonical_to_class: Dict[str, str] = {c: c for c in candidates}
    else:
        canonical_to_class = dict(canonical_to_coarse or {})

    cell_type = rows_df["cell_type"].astype(str).str.strip()
    mapped_class = cell_type.map(canonical_to_class)  # candidate class or NaN
    full = pd.DataFrame(
        {
            "class": mapped_class.to_numpy(dtype=object),
            "slide_id": rows_df["slide_id"].astype(str).to_numpy(),
        }
    )
    df = full[full["class"].notna()]

    # Per (class, slide) cell counts -> per-class within-slide aggregates.
    if len(df) == 0:
        per_cs = pd.DataFrame({"class": [], "slide_id": [], "n": []})
    else:
        per_cs = df.groupby(["class", "slide_id"], sort=False).size().reset_index(name="n")

    records = []
    for cls in candidates:
        counts = per_cs.loc[per_cs["class"] == cls, "n"].to_numpy()
        records.append(
            {
                "class": cls,
                "n_cells": int(counts.sum()) if counts.size else 0,
                "n_slides": int(counts.size),
                "n_slides_meeting_floor": int((counts >= min_cells_per_slide).sum())
                if counts.size
                else 0,
                "max_cells_in_slide": int(counts.max()) if counts.size else 0,
            }
        )
    return pd.DataFrame(
        records,
        columns=["class", "n_cells", "n_slides", "n_slides_meeting_floor", "max_cells_in_slide"],
    )


def select_trainable_classes_within_slide(
    per_slide_support_df: pd.DataFrame,
    *,
    min_cells_per_slide: int,
    min_slides: int,
    min_classes: int,
) -> List[str]:
    """
    Within-slide (STHELAR_TEST_ADAPTED) class selection: keep the classes with >= min_cells_per_slide cells in
    >= min_slides slides, in candidate order. Raise RuntimeError if fewer than min_classes remain.
    """
    keep = [
        str(row["class"])
        for _, row in per_slide_support_df.iterrows()
        if int(row["n_slides_meeting_floor"]) >= int(min_slides)
    ]
    if len(keep) < int(min_classes):
        raise RuntimeError(
            f"Only {len(keep)} class(es) meet the ADAPTED within-slide support thresholds "
            f"(adapted_min_cells_per_slide={int(min_cells_per_slide)}, "
            f"adapted_min_slides={int(min_slides)}); need >= {int(min_classes)}. "
            f"Support table: {per_slide_support_df.to_dict(orient='records')}"
        )
    logger.debug("Selected %d ADAPTED trainable class(es): %s", len(keep), keep)
    return keep


def cat2color_hex(ct_cat_id: str) -> Dict[str, str]:
    """Canonical cell-type -> '#RRGGBB' hex from cat2color.json (RGB 0-255 triples)."""
    cat2color = load_annot_dicts(ct_cat_id)["cat2color"]
    out: Dict[str, str] = {}
    for lab, rgb in cat2color.items():
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        out[str(lab)] = "#{:02x}{:02x}{:02x}".format(r, g, b)
    return out
