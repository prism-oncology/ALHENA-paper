"""Protocol freeze for Task A.3: frozen settings of an eval_id (rows, class vocabulary, splits, class coverage)."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from src_A3.common import ensure_imports

ensure_imports()

# Imports after ensure_imports() (sys.path setup)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src_A3.common import (  # noqa: E402
    N_ADAPTED_ROTATIONS,
    adapted_train_stripes,
    mkdir,
    save_json,
    save_text,
    sha256_file,
    read_metadata,
    SlideRecord,
    build_split_group_id,
    dump_environment_report,
    protocol_identity_sha256,
    read_a3_eval_config,
    validate_a3_protocol_identity,
    default_a3_eval_config,
    save_a3_eval_config_with_comments,
    resolve_eval_dataset_root,
)
from src_A3.labels import (  # noqa: E402
    LabelSpace,
    ANNOT_DIR,
    _candidate_labels_for_granularity,
    build_label_space,
    coarse_mapping_filename,
    map_cell_types_to_ids,
    class_support_table,
    select_trainable_classes,
    class_per_slide_support_table,
    select_trainable_classes_within_slide,
)
from src_A3.data_loading import build_row_manifest_for_slide  # noqa: E402

# Split builders of Task A.1 (slide-level CV / one-shot / within-slide area)
from src_A1.protocols import (  # noqa: E402
    build_slide_level_cv_splits,
    build_area_split_manifests,
)


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# settings/ subdir layout
# -----------------------------------------------------------------------------
@dataclass
class A3ProtocolPaths:
    """Paths of the settings/ subdirs of one Task A.3 eval_id."""

    eval_dir: Path
    settings_dir: Path
    info_dir: Path
    splits_dir: Path
    rows_dir: Path
    area_dir: Path
    labels_dir: Path
    manifests_dir: Path


def make_a3_protocol_paths(eval_dir: Path) -> A3ProtocolPaths:
    """Create (mkdir) and return the frozen ``settings/`` subdir bundle for an eval_dir."""
    eval_dir = Path(eval_dir)
    return A3ProtocolPaths(
        eval_dir=eval_dir,
        settings_dir=mkdir(eval_dir / "settings"),
        info_dir=mkdir(eval_dir / "settings" / "informations"),
        splits_dir=mkdir(eval_dir / "settings" / "splits"),
        rows_dir=mkdir(eval_dir / "settings" / "rows"),
        area_dir=mkdir(eval_dir / "settings" / "area_splits"),
        labels_dir=mkdir(eval_dir / "settings" / "labels"),
        # Excluded slides (rows/, splits/ and labels/ hold admitted slides only)
        manifests_dir=mkdir(eval_dir / "settings" / "manifests"),
    )


# -----------------------------------------------------------------------------
# Initial eval folder (the _1 step)
# -----------------------------------------------------------------------------
def _settings_readme() -> str:
    return (
        "# HistoST Task A.3 settings folder\n\n"
        "This folder stores the full persisted evaluation protocol for one Task A.3 eval_id\n"
        "(cell-type linear-probe classification).\n\n"
        "Every encoder run under the same eval_id uses the same slide/area splits, per-cell row\n"
        "universe, class vocabulary and class-support thresholds, head type and per-space\n"
        "geometry. The scoring class axis (fine or coarse) is a protocol-identity field; fine\n"
        "and coarse are separate eval_ids.\n"
    )


def create_initial_a3_eval_folder(
    *,
    eval_dir: Path,
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
    probe_c_min: float = 1e-6,
    probe_c_max: float = 1e0,
    probe_c_points: int = 7,
    min_selected_alpha: Optional[float] = None,
    scconcept_training_id: Optional[str] = None,
    protocol_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Write the initial (editable) eval_config.yaml, environment.json and settings README of an eval_id.
    Returns the config dict.
    """
    cfg = default_a3_eval_config(
        eval_id=eval_id,
        dataset_id=dataset_id,
        training_dataset_id=training_dataset_id,
        ct_cat_id=ct_cat_id,
        eval_configuration=eval_configuration,
        sub_eval_configuration=sub_eval_configuration,
        modelgroup=modelgroup,
        label_granularity=label_granularity,
        coarse_mapping_version=coarse_mapping_version,
        head_type=head_type,
        probe_C=probe_C,
        probe_c_min=probe_c_min,
        probe_c_max=probe_c_max,
        probe_c_points=probe_c_points,
        min_selected_alpha=min_selected_alpha,
        scconcept_training_id=scconcept_training_id,
        protocol_seed=protocol_seed,
    )
    validate_a3_protocol_identity(cfg)

    paths = make_a3_protocol_paths(Path(eval_dir))
    save_a3_eval_config_with_comments(paths.settings_dir / "eval_config.yaml", cfg)
    save_json(paths.settings_dir / "environment.json", dump_environment_report())
    save_text(paths.info_dir / "README.md", _settings_readme())
    return cfg


# -----------------------------------------------------------------------------
# Protocol slide pool
# -----------------------------------------------------------------------------
def _a3_protocol_slide_records(cfg: Mapping[str, Any], records: List[SlideRecord]) -> List[SlideRecord]:
    """
    STHELAR slides of the protocol: STHELAR_train (STHELAR_TRAIN_CV), STHELAR_train + STHELAR_test
    (STHELAR_TEST_ONE_SHOT) or STHELAR_test (STHELAR_TEST_ADAPTED).
    """
    sub_conf = str(cfg["protocol"]["sub_eval_configuration"])

    out: List[SlideRecord] = []
    for rec in records:
        if str(rec.source) != "STHELAR":
            continue
        subset = str(rec.subset)
        if sub_conf == "STHELAR_TRAIN_CV":
            keep = subset == "STHELAR_train"
        elif sub_conf == "STHELAR_TEST_ONE_SHOT":
            keep = subset in {"STHELAR_train", "STHELAR_test"}
        elif sub_conf == "STHELAR_TEST_ADAPTED":
            keep = subset == "STHELAR_test"
        else:
            raise ValueError(
                f"Unsupported sub_eval_configuration for Task A.3: {sub_conf!r}."
            )
        if keep:
            out.append(rec)

    if not out:
        raise RuntimeError(
            f"No STHELAR slides were selected for Task A.3 protocol '{sub_conf}'. "
            "Check the training-dataset metadata (source/subset columns)."
        )
    return out


# -----------------------------------------------------------------------------
# Frozen per-cell row universe
# -----------------------------------------------------------------------------
# Columns of the frozen row frame (label_id is added to the scored rows later)
_A3_ROW_UNIVERSE_COLUMNS = [
    "row_key",
    "slide_id",
    "source",
    "subset",
    "tissue",
    "species",
    "patient_raw",
    "patient_clean",
    "dataset_title",
    "split_group_id",
    "cell_id_int",
    "x",
    "y",
    "cell_type",
]

# Parquet dtype of each row column (other columns are left as-is)
_A3_ROW_STRING_COLUMNS = frozenset(
    {
        "row_key",
        "slide_id",
        "source",
        "subset",
        "tissue",
        "species",
        "patient_raw",
        "patient_clean",
        "dataset_title",
        "split_group_id",
        "cell_type",
    }
)
_A3_ROW_INT_COLUMNS = frozenset({"cell_id_int"})
_A3_ROW_FLOAT_COLUMNS = frozenset({"x", "y"})


def _coerce_a3_rows_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast the row frame to Parquet-safe dtypes (strings -> string, cell_id_int -> Int64, x / y -> float64).
    Raises if an object column remains.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if col in _A3_ROW_STRING_COLUMNS:
            out[col] = s.astype("string")
        elif col in _A3_ROW_INT_COLUMNS:
            out[col] = pd.to_numeric(s, errors="coerce").astype("Int64")
        elif col in _A3_ROW_FLOAT_COLUMNS:
            out[col] = pd.to_numeric(s, errors="coerce").astype(np.float64)
        elif isinstance(s.dtype, pd.CategoricalDtype):
            out[col] = s.astype("string")
        elif pd.api.types.is_object_dtype(s.dtype):
            out[col] = s.astype("string")
        # otherwise (already a real numeric/bool dtype, e.g. int label_id) leave as-is.

    bad_object_cols = [str(c) for c in out.columns if pd.api.types.is_object_dtype(out[c].dtype)]
    if bad_object_cols:
        raise RuntimeError(
            "A.3 row manifest still contains object dtype columns after Parquet schema "
            f"canonicalization: {bad_object_cols[:20]}"
        )
    return out


def _build_a3_slide_row_frame(
    cfg: Mapping[str, Any], rec: SlideRecord
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Row frame of one protocol slide (cell level). Returns (frame, record of the cells dropped by the
    modelgroup token eligibility).
    """
    dataset_id = str(cfg["identity"]["dataset_id"])
    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    modelgroup = str(cfg["protocol"]["modelgroup"])

    bundle = build_row_manifest_for_slide(
        dataset_id=dataset_id,
        training_dataset_id=training_dataset_id,
        level="cell",
        modelgroup=modelgroup,
        slide_record=rec,
    )

    if bundle.cell_id_int is None:
        raise RuntimeError(
            f"Slide '{rec.slide_id}' produced no cell ids at level='cell'. "
            "The cell-level eval dataset for this modelgroup is likely missing."
        )
    if bundle.cell_types is None:
        raise RuntimeError(
            f"Slide '{rec.slide_id}' produced no cell_type strings; Task A.3 requires "
            "canonical cell-type labels on every cell."
        )

    row_keys = np.asarray(bundle.row_keys, dtype=object)
    coords = np.asarray(bundle.coords_xy, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise RuntimeError(
            f"Slide '{rec.slide_id}': coords_xy must have shape (N, 2), got {coords.shape}."
        )
    cell_id_int = np.asarray(bundle.cell_id_int)
    cell_types = np.asarray(bundle.cell_types, dtype=object)

    n = int(row_keys.shape[0])
    if not (coords.shape[0] == n == cell_id_int.shape[0] == cell_types.shape[0]):
        raise RuntimeError(
            f"Slide '{rec.slide_id}': inconsistent row-bundle lengths "
            f"(row_keys={row_keys.shape[0]}, coords={coords.shape[0]}, "
            f"cell_id_int={cell_id_int.shape[0]}, cell_types={cell_types.shape[0]})."
        )
    if n == 0:
        raise RuntimeError(
            f"Slide '{rec.slide_id}' produced zero eligible cells for the frozen row universe."
        )

    # Same value as rec.split_group_id (recomputed)
    split_group_id = build_split_group_id(
        slide_id=str(rec.slide_id),
        source=str(rec.source),
        subset=str(rec.subset),
        tissue=str(rec.tissue),
        dataset_title=str(rec.dataset_title),
        patient_clean=rec.patient_clean,
    )

    frame = pd.DataFrame(
        {
            "row_key": row_keys,
            "slide_id": str(rec.slide_id),
            "source": str(rec.source),
            "subset": str(rec.subset),
            "tissue": str(rec.tissue),
            "species": str(rec.species),
            "patient_raw": rec.patient_raw,
            "patient_clean": rec.patient_clean,
            "dataset_title": str(rec.dataset_title),
            "split_group_id": split_group_id,
            "cell_id_int": cell_id_int,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cell_type": cell_types,
        }
    )
    report = dict(bundle.row_filter_report or {})
    token_record = {
        "slide_id": str(rec.slide_id),
        "n_is_cell_rows": int(report.get("n_is_cell_rows", n)),
        "n_cells_missing_after_modelgroup_tokenization": int(
            report.get("n_cells_missing_after_modelgroup_tokenization", 0)
        ),
    }
    return frame[_A3_ROW_UNIVERSE_COLUMNS], token_record


def build_a3_row_universe(
    *,
    cfg: Mapping[str, Any],
    slide_records: List[SlideRecord],
    eval_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-cell row universe of the protocol slides (unscorable cells kept for rows/unknown_cells.parquet).
    Returns (rows_full, per-slide cells dropped by the modelgroup token eligibility). eval_dir is unused.
    """
    built = [_build_a3_slide_row_frame(cfg, rec) for rec in slide_records]
    rows_full = pd.concat([f for f, _ in built], axis=0, ignore_index=True)
    token_eligibility = pd.DataFrame(
        [rec for _, rec in built],
        columns=["slide_id", "n_is_cell_rows", "n_cells_missing_after_modelgroup_tokenization"],
    )

    # The main-assignment alpha gate is not implemented (min_selected_alpha must be None)
    min_selected_alpha = cfg["protocol"].get("cell_filter", {}).get("min_selected_alpha", None)
    if min_selected_alpha is not None:
        raise NotImplementedError(
            "cell_filter.min_selected_alpha is set, but the main-assignment alpha gate is not "
            "implemented in Task A.3 ('selected_alpha' is encoder-specific). Leave "
            "min_selected_alpha null."
        )

    return rows_full, token_eligibility


# -----------------------------------------------------------------------------
# Slide admission (freeze-time, before the class vocabulary)
# -----------------------------------------------------------------------------
def a3_slide_admission_counts(
    rows_full_df: pd.DataFrame,
    *,
    ct_cat_id: str,
    granularity: str,
    coarse_mapping_version: Optional[str],
    macro_floor: int = 0,
) -> pd.DataFrame:
    """
    Per-slide admission counts on the candidate class axis of granularity: slide_id, subset, tissue, n_cells,
    n_scorable_cells, n_classes_present, n_classes_at_macro_floor (classes with >= macro_floor cells).
    """
    _, candidates, canonical_to_coarse = _candidate_labels_for_granularity(
        ct_cat_id, granularity, coarse_mapping_version
    )
    canonical_to_class: Dict[str, str] = (
        {c: c for c in candidates} if canonical_to_coarse is None else dict(canonical_to_coarse)
    )
    macro_floor = max(0, int(macro_floor))

    work = pd.DataFrame(
        {
            "slide_id": rows_full_df["slide_id"].astype(str).to_numpy(),
            "subset": rows_full_df["subset"].astype(str).to_numpy(),
            "tissue": rows_full_df["tissue"].astype(str).to_numpy(),
            "cls": rows_full_df["cell_type"].astype(str).str.strip().map(canonical_to_class).to_numpy(
                dtype=object
            ),
        }
    )
    records: List[Dict[str, Any]] = []
    for slide_id, g in work.groupby("slide_id", sort=True):
        scorable = g.loc[g["cls"].notna()]
        per_class = scorable["cls"].value_counts()
        records.append(
            {
                "slide_id": str(slide_id),
                "subset": str(g.iloc[0]["subset"]),
                "tissue": str(g.iloc[0]["tissue"]),
                "n_cells": int(len(g)),
                "n_scorable_cells": int(len(scorable)),
                "n_classes_present": int(per_class.size),
                "n_classes_at_macro_floor": int((per_class >= macro_floor).sum())
                if macro_floor > 0
                else int(per_class.size),
            }
        )
    return pd.DataFrame(
        records,
        columns=[
            "slide_id", "subset", "tissue", "n_cells", "n_scorable_cells",
            "n_classes_present", "n_classes_at_macro_floor",
        ],
    )


def a3_slide_admission_decision(
    counts: pd.DataFrame,
    *,
    min_cells: int,
    min_classes: int,
    logger_: Optional[logging.Logger] = None,
) -> Tuple[set, List[Dict[str, Any]]]:
    """
    Slide admission: a slide is kept if n_scorable_cells >= min_cells and n_classes_present >= min_classes
    (<= 0 disables a floor). Returns (kept_slide_ids, excluded_records); raises if no slide is kept.
    """
    log = logger_ or logger
    if counts is None or counts.empty:
        raise ValueError("A.3 slide admission received an empty cohort.")
    counts = counts.sort_values(["n_scorable_cells", "slide_id"]).reset_index(drop=True)
    min_cells = int(min_cells)
    min_classes = int(min_classes)

    if min_cells <= 0 and min_classes <= 0:
        log.info(
            "A.3 slide admission is disabled (min_scorable_cells_per_slide=%d, "
            "min_classes_per_slide=%d); all %d slide(s) admitted.",
            min_cells, min_classes, int(counts.shape[0]),
        )
        return set(counts["slide_id"].astype(str)), []

    fails_cells = counts["n_scorable_cells"] < min_cells if min_cells > 0 else False
    fails_classes = counts["n_classes_present"] < min_classes if min_classes > 0 else False
    dropped_mask = (fails_cells | fails_classes) if (min_cells > 0 or min_classes > 0) else False
    dropped = counts.loc[dropped_mask]
    kept = counts.loc[~dropped_mask] if len(counts) else counts

    excluded: List[Dict[str, Any]] = []
    for _, r in dropped.iterrows():
        reasons: List[str] = []
        if min_cells > 0 and int(r["n_scorable_cells"]) < min_cells:
            reasons.append(
                f"{int(r['n_scorable_cells']):,} scorable cells < "
                f"min_scorable_cells_per_slide={min_cells:,}"
            )
        if min_classes > 0 and int(r["n_classes_present"]) < min_classes:
            reasons.append(
                f"{int(r['n_classes_present'])} scorable class(es) present < "
                f"min_classes_per_slide={min_classes}"
            )
        record = {
            "slide_id": str(r["slide_id"]),
            "subset": str(r["subset"]),
            "tissue": str(r["tissue"]),
            "n_cells": int(r["n_cells"]),
            "n_scorable_cells": int(r["n_scorable_cells"]),
            "n_classes_present": int(r["n_classes_present"]),
            "n_classes_at_macro_floor": int(r["n_classes_at_macro_floor"]),
            "min_scorable_cells_per_slide": min_cells,
            "min_classes_per_slide": min_classes,
            "reason": "; ".join(reasons),
        }
        excluded.append(record)
        log.warning(
            "Slide excluded from the A.3 cohort: '%s' (subset=%s, tissue=%s) -- %s.",
            record["slide_id"], record["subset"], record["tissue"], record["reason"],
        )

    if kept.empty:
        raise RuntimeError(
            f"A.3 slide admission (min_scorable_cells_per_slide={min_cells}, "
            f"min_classes_per_slide={min_classes}) excluded every slide of the cohort "
            f"(largest: {int(counts['n_scorable_cells'].max()):,} scorable cells, "
            f"{int(counts['n_classes_present'].max())} classes). Lower the floors."
        )

    smallest = kept.iloc[0]
    log.info(
        "A.3 slide admission (min_scorable_cells_per_slide=%s, min_classes_per_slide=%d): "
        "%d/%d slide(s) admitted, %d excluded. Smallest retained slide: '%s' with %s scorable "
        "cell(s) (%.2fx the floor) and %d classes. Largest excluded: %s.",
        f"{min_cells:,}", min_classes,
        int(kept.shape[0]), int(counts.shape[0]), int(dropped.shape[0]),
        str(smallest["slide_id"]), f"{int(smallest['n_scorable_cells']):,}",
        (float(smallest["n_scorable_cells"]) / float(min_cells)) if min_cells > 0 else float("inf"),
        int(smallest["n_classes_present"]),
        "none" if dropped.empty
        else f"'{dropped.iloc[-1]['slide_id']}' with {int(dropped.iloc[-1]['n_scorable_cells']):,}",
    )

    # Warn when fewer than min_classes_per_slide classes of an admitted slide reach the macro support floor
    if min_classes > 0:
        thin = kept.loc[kept["n_classes_at_macro_floor"] < min_classes]
        for _, r in thin.iterrows():
            log.warning(
                "Slide '%s' is admitted with %d scorable class(es) but only %d reach the per-slide "
                "macro support floor (min_classes_per_slide=%d).",
                str(r["slide_id"]), int(r["n_classes_present"]),
                int(r["n_classes_at_macro_floor"]), min_classes,
            )

    return set(kept["slide_id"].astype(str)), excluded


# -----------------------------------------------------------------------------
# Freeze the row manifests (scored vs unknown)
# -----------------------------------------------------------------------------
def freeze_a3_row_manifests(
    rows_full_df: pd.DataFrame,
    label_space: LabelSpace,
    paths: A3ProtocolPaths,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the row universe into scored cells (label_id >= 0) and unknown cells (unscorable, out-of-vocab or
    support-dropped classes); write rows/all_rows.parquet, rows/<slide_id>.parquet and rows/unknown_cells.parquet.
    Returns (rows_scored_df, unknown_df).
    """
    cell_types = rows_full_df["cell_type"].to_numpy()
    label_id = map_cell_types_to_ids(cell_types, label_space)  # int32 [N]; -1 for unscorable/OOV
    scored_mask = label_id >= 0

    rows_scored = rows_full_df.loc[scored_mask].reset_index(drop=True).copy()
    rows_scored["label_id"] = label_id[scored_mask].astype(np.int32)  # positional; both in order
    unknown = rows_full_df.loc[~scored_mask].reset_index(drop=True).copy()

    rows_scored = _coerce_a3_rows_for_parquet(rows_scored)
    unknown = _coerce_a3_rows_for_parquet(unknown)

    if rows_scored.empty:
        raise RuntimeError(
            "No scored cells remain after applying the frozen class vocabulary; "
            "the class-support thresholds or the label dictionaries are likely inconsistent."
        )

    rows_dir = paths.rows_dir
    kept_slide_ids = set(rows_scored["slide_id"].astype(str).unique().tolist())

    # Warn about per-slide row files left by a previous freeze (not deleted here)
    stale_rows = [
        p for p in sorted(rows_dir.glob("*.parquet"))
        if not p.name.startswith(".")
        and p.name not in {"all_rows.parquet", "unknown_cells.parquet"}
        and p.stem not in kept_slide_ids
    ]
    if stale_rows:
        logger.warning(
            "%d stale per-slide row manifest(s) from a previous freeze are still in %s and are "
            "not part of this freeze: %s. They are not read (_3 reads all_rows.parquet); delete "
            "them by hand.",
            len(stale_rows),
            rows_dir,
            [p.name for p in stale_rows],
        )

    rows_scored.to_parquet(rows_dir / "all_rows.parquet", index=False)
    for slide_id, slide_df in rows_scored.groupby(rows_scored["slide_id"].astype(str), sort=True):
        slide_df.to_parquet(rows_dir / f"{slide_id}.parquet", index=False)
    unknown.to_parquet(rows_dir / "unknown_cells.parquet", index=False)

    logger.info(
        "Froze A.3 row manifests: %d scored cells, %d unknown cells (%d classes).",
        int(rows_scored.shape[0]),
        int(unknown.shape[0]),
        int(label_space.n_classes()),
    )
    return rows_scored, unknown


# -----------------------------------------------------------------------------
# Class protocols (vocabulary / support / per-fold coverage)
# -----------------------------------------------------------------------------
def _a3_support_pool(rows_df: pd.DataFrame, sub_eval_configuration: str) -> pd.DataFrame:
    """Rows used for class support: STHELAR_train slides for STHELAR_TEST_ONE_SHOT, all protocol slides otherwise."""
    if str(sub_eval_configuration) == "STHELAR_TEST_ONE_SHOT":
        return rows_df.loc[rows_df["subset"].astype(str) == "STHELAR_train"]
    return rows_df


def _train_class_counts_from_rows(
    rows: pd.DataFrame, label_space: LabelSpace
) -> Dict[str, int]:
    """Train-cell count of every class of label_space in rows (dense-id order, 0 if absent)."""
    n_classes = int(label_space.n_classes())
    counts = {label_space.labels[i]: 0 for i in range(n_classes)}
    if rows is None or rows.empty or "label_id" not in rows.columns:
        return counts
    ids = pd.to_numeric(rows["label_id"], errors="coerce").dropna().astype(int)
    for label_id, n in ids.value_counts().items():
        i = int(label_id)
        if 0 <= i < n_classes:
            counts[label_space.labels[i]] = int(n)
    return counts


def _a3_fold_class_coverage(
    cfg: Mapping[str, Any],
    rows_scored_df: pd.DataFrame,
    splits_manifest: Mapping[str, Any],
    label_space: LabelSpace,
    paths: A3ProtocolPaths,
) -> Dict[str, Dict[str, int]]:
    """
    Per-fold train class coverage {fold_name -> {class -> n_train_cells}} (0 if absent). Fold names:
    fold_<fold_id:02d> (slide_cv), oneshot (one_shot), <slide_id>_fold_<fold_idx> (within_slide_area).
    """
    stype = str(splits_manifest.get("type", ""))
    coverage: Dict[str, Dict[str, int]] = {}
    slide_col = rows_scored_df["slide_id"].astype(str)

    if stype == "slide_cv":
        for fold in splits_manifest.get("folds", []):
            fold_id = int(fold["fold_id"])
            fold_name = f"fold_{fold_id:02d}"
            train_slides = {str(s) for s in fold.get("train_slides", [])}
            train_rows = rows_scored_df.loc[slide_col.isin(train_slides)]
            coverage[fold_name] = _train_class_counts_from_rows(train_rows, label_space)

    elif stype == "one_shot":
        train_slides = {str(s) for s in splits_manifest.get("train_slides", [])}
        train_rows = rows_scored_df.loc[slide_col.isin(train_slides)]
        coverage["oneshot"] = _train_class_counts_from_rows(train_rows, label_space)

    elif stype == "within_slide_area":
        n_folds = N_ADAPTED_ROTATIONS  # 3 stripe rotations, independent of cv.n_outer_folds
        for slide in splits_manifest.get("slides", []):
            slide = str(slide)
            slide_rows = rows_scored_df.loc[slide_col == slide]
            cell_path = paths.area_dir / f"{slide}_cell_stripes.parquet"
            # An unreadable stripe file raises (no fallback)
            try:
                cell_df = pd.read_parquet(cell_path)
                stripe_ids = pd.to_numeric(cell_df["stripe_id"], errors="coerce")
                slide_cell_int = pd.to_numeric(slide_rows["cell_id_int"], errors="coerce")
            except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
                raise RuntimeError(
                    f"ADAPTED per-stripe geometry is unreadable for slide '{slide}' "
                    f"({cell_path}): {exc}. Per-fold class coverage cannot be computed. "
                    "Rebuild the within-slide area splits for this slide."
                ) from exc
            for fold_idx in range(n_folds):
                # Train cells of the rotation: its two train stripes (as in _3_run_eval.py)
                train_cells = pd.to_numeric(
                    cell_df.loc[stripe_ids.isin(adapted_train_stripes(fold_idx)), "cell_id_int"],
                    errors="coerce",
                )
                train_rows = slide_rows.loc[slide_cell_int.isin(train_cells).to_numpy()]
                coverage[f"{slide}_fold_{fold_idx}"] = _train_class_counts_from_rows(
                    train_rows, label_space
                )

    else:
        raise ValueError(f"Unsupported splits_manifest type for class coverage: {stype!r}.")

    return coverage


class FoldCoverageError(RuntimeError):
    """Raised when a class has too few train cells in a fold (fold-coverage gate)."""


def a3_fold_coverage_decision(
    coverage: Mapping[str, Mapping[str, int]],
    *,
    min_cells: int,
    declared_exceptions: Optional[Mapping[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fold-coverage decision: a (fold, class) with fewer than min_cells train cells is a violation (min_cells <= 0
    disables the check). Returns (violations, declared), declared = violations in folds of declared_exceptions.
    """
    min_cells = int(min_cells)
    declared_exceptions = {str(k): str(v) for k, v in dict(declared_exceptions or {}).items()}
    violations: List[Dict[str, Any]] = []
    declared: List[Dict[str, Any]] = []
    if min_cells <= 0:
        return violations, declared

    for fold_name in sorted(coverage):
        counts = coverage[fold_name] or {}
        for class_name in sorted(counts):
            n = int(counts[class_name])
            if n >= min_cells:
                continue
            record: Dict[str, Any] = {
                "fold_name": str(fold_name),
                "class": str(class_name),
                "n_train_cells": n,
                "min_train_cells_per_class_per_fold": min_cells,
                "absent_from_train": bool(n == 0),
            }
            if str(fold_name) in declared_exceptions:
                record["reason"] = declared_exceptions[str(fold_name)]
                declared.append(record)
            else:
                violations.append(record)
    return violations, declared


def assert_a3_fold_coverage(
    coverage: Mapping[str, Mapping[str, int]],
    *,
    min_cells: int,
    declared_exceptions: Optional[Mapping[str, str]] = None,
    logger_: Optional[logging.Logger] = None,
) -> None:
    """
    Enforce the fold-coverage check: raises FoldCoverageError listing every (fold, class) violation; declared
    exceptions are only logged. Logs the thinnest passing (fold, class).
    """
    log = logger_ or logger
    min_cells = int(min_cells)
    if min_cells <= 0:
        log.warning(
            "A.3 fold-coverage gate is disabled (protocol.fold_coverage."
            "min_train_cells_per_class_per_fold=%d).",
            min_cells,
        )
        return

    violations, declared = a3_fold_coverage_decision(
        coverage, min_cells=min_cells, declared_exceptions=declared_exceptions
    )

    for rec in declared:
        log.warning(
            "Declared fold-coverage exception for fold '%s', class '%s': %d train cell(s) < %d. "
            "Reason: %s",
            rec["fold_name"], rec["class"], rec["n_train_cells"], min_cells, rec["reason"],
        )

    # Thinnest passing (fold, class)
    passing = [
        (fold, cls, int(n))
        for fold, counts in coverage.items()
        for cls, n in (counts or {}).items()
        if int(n) >= min_cells
    ]
    if passing:
        fold_m, cls_m, n_m = min(passing, key=lambda t: (t[2], t[0], t[1]))
        log.info(
            "A.3 fold-coverage gate (min_train_cells_per_class_per_fold=%d): %d fold(s) x %d "
            "class(es) checked, %d violation(s), %d declared. Thinnest passing cell: fold '%s' / "
            "class '%s' with %s train cell(s) (%.2fx the floor).",
            min_cells,
            len(coverage),
            max((len(c or {}) for c in coverage.values()), default=0),
            len(violations),
            len(declared),
            fold_m, cls_m, f"{n_m:,}", float(n_m) / float(min_cells),
        )

    if not violations:
        return

    absent = [r for r in violations if r["absent_from_train"]]
    thin = [r for r in violations if not r["absent_from_train"]]
    blocks = "".join(
        f"    fold '{r['fold_name']}'  class '{r['class']}'  {r['n_train_cells']:>9,} train cell(s)"
        + ("   <-- absent from this fold's train set\n" if r["absent_from_train"] else "\n")
        for r in violations
    )
    offending_folds = sorted({r["fold_name"] for r in violations})
    raise FoldCoverageError(
        f"A.3 fold-coverage gate violated by {len(violations)} (fold, class) cell(s) across "
        f"{len(offending_folds)} fold(s): {offending_folds}.\n"
        f"  {len(absent)} class(es) absent from a fold's train set, "
        f"{len(thin)} below min_train_cells_per_class_per_fold={min_cells}.\n\n"
        f"{blocks}\n"
        "Options:\n"
        "  (1) raise protocol.class_selection.min_train_slides_per_class (drops the class); or\n"
        "  (2) re-freeze with a different protocol.protocol_seed (only before any result exists); or\n"
        "  (3) declare the fold in the frozen eval_config.yaml:\n\n"
        "        protocol:\n"
        "          fold_coverage:\n"
        "            declared_exceptions:\n"
        + "".join(
            f'              {f}: "<why this fold is still worth scoring with '
            f'{", ".join(sorted(r["class"] for r in violations if r["fold_name"] == f))} '
            f'untrained>"\n'
            for f in offending_folds
        )
        + "\n"
    )


def persist_a3_class_protocols(
    *,
    cfg: Mapping[str, Any],
    rows_scored_df: pd.DataFrame,
    splits_manifest: Mapping[str, Any],
    label_space: LabelSpace,
    paths: A3ProtocolPaths,
) -> Dict[str, Any]:
    """
    Write settings/labels/ class_vocabulary.json, class_support.csv and fold_class_coverage.json, then run the
    fold-coverage check (FoldCoverageError). Returns the paths and a vocabulary summary.
    """
    ct_cat_id = str(cfg["protocol"]["ct_cat_id"])
    granularity = str(label_space.granularity)
    sub_eval = str(cfg["protocol"]["sub_eval_configuration"])
    coarse_version = cfg["protocol"].get("coarse_mapping_version", None)
    coarse_version = str(coarse_version).strip() if coarse_version else None

    # 1) Frozen vocabulary.
    class_vocab_path = paths.labels_dir / "class_vocabulary.json"
    save_json(class_vocab_path, label_space.as_dict())

    # 2) Class support of the frozen (trainable) classes on the train pool.
    support_pool = _a3_support_pool(rows_scored_df, sub_eval)
    support = class_support_table(
        support_pool,
        ct_cat_id=ct_cat_id,
        granularity=granularity,
        coarse_mapping_version=coarse_version,
    )
    support = support.set_index("class").reindex(label_space.labels).reset_index()
    support["label_id"] = support["class"].map(label_space.label_to_index).astype(int)
    support["n_cells"] = support["n_cells"].fillna(0).astype(int)
    support["n_slides"] = support["n_slides"].fillna(0).astype(int)
    support = support[["class", "label_id", "n_cells", "n_slides"]]
    class_support_path = paths.labels_dir / "class_support.csv"
    support.to_csv(class_support_path, index=False)

    # 3) Per-fold train class coverage (written before the check)
    coverage = _a3_fold_class_coverage(cfg, rows_scored_df, splits_manifest, label_space, paths)
    fold_coverage_path = paths.labels_dir / "fold_class_coverage.json"
    save_json(fold_coverage_path, coverage)

    # 4) Fold-coverage check
    fold_cov_cfg = cfg["protocol"].get("fold_coverage", {}) or {}
    assert_a3_fold_coverage(
        coverage,
        min_cells=int(fold_cov_cfg.get("min_train_cells_per_class_per_fold", 0)),
        declared_exceptions=fold_cov_cfg.get("declared_exceptions", {}) or {},
    )

    logger.info(
        "Froze A.3 class protocols: %d %s classes across %d fold(s).",
        int(label_space.n_classes()),
        granularity,
        len(coverage),
    )
    # Thinnest (fold, class) train-cell count
    all_counts = [int(n) for counts in coverage.values() for n in (counts or {}).values()]
    return {
        "class_vocabulary": str(class_vocab_path),
        "class_support": str(class_support_path),
        "fold_class_coverage": str(fold_coverage_path),
        "granularity": granularity,
        "n_classes": int(label_space.n_classes()),
        "labels": list(label_space.labels),
        "dropped_labels": list(label_space.dropped_labels),
        "min_train_cells_per_class_per_fold": int(
            (cfg["protocol"].get("fold_coverage", {}) or {}).get(
                "min_train_cells_per_class_per_fold", 0
            )
        ),
        "thinnest_fold_class_train_cells": (min(all_counts) if all_counts else None),
    }


# -----------------------------------------------------------------------------
# Informational summary
# -----------------------------------------------------------------------------
def _write_a3_row_summary(
    rows_scored_df: pd.DataFrame,
    unknown_df: pd.DataFrame,
    info_dir: Path,
) -> None:
    """Write a small per-slide scored/unknown cell-count summary into ``informations/``."""
    cols = ["slide_id", "source", "subset", "tissue"]
    frames: List[pd.DataFrame] = []
    for df, kind in ((rows_scored_df, "scored"), (unknown_df, "unknown")):
        if df is None or df.empty:
            continue
        sub = df[cols].copy()
        sub["kind"] = kind
        frames.append(sub)
    if not frames:
        return
    summary = (
        pd.concat(frames, axis=0, ignore_index=True)
        .groupby(cols + ["kind"], dropna=False)
        .size()
        .reset_index(name="n_cells")
    )
    summary.to_csv(info_dir / "row_manifest_summary.csv", index=False)


def _a3_protocol_contract_notes(cfg: Mapping[str, Any]) -> List[str]:
    """Notes stored in the protocol manifest."""
    granularity = str(cfg["protocol"]["label_granularity"])
    coarse_version = cfg["protocol"].get("coarse_mapping_version", None)
    min_alpha = cfg["protocol"].get("cell_filter", {}).get("min_selected_alpha", None)
    notes = [
        "Task A.3 metrics are descriptive.",
        "Cell-level supervision only; the scoring axis is the frozen global class vocabulary "
        "(dense 0..C-1).",
        "Per-fold classifier outputs are aligned to the global class vocabulary at predict time "
        "(0-fill for classes absent from a fold's train set); argmax can only pick a trained class.",
        "The freeze-time fold-coverage gate (protocol.fold_coverage) requires every frozen class "
        "to have >= min_train_cells_per_class_per_fold train cells in every fold. When it passes "
        "with no declared exception, true_class_covered is all-True and the _restricted metrics "
        "equal the unrestricted ones.",
        f"Label granularity '{granularity}' is a protocol-identity field; fine and coarse are "
        "separate eval_ids.",
        "obs['cell_type'] is already canonical (label2cat applied in preprocessing); A.3 applies "
        "the dense id map only (no second label2cat).",
    ]
    if coarse_version is not None:
        notes.append(
            f"Coarse class axis frozen at coarse_mapping_version='{coarse_version}' "
            f"({coarse_mapping_filename(coarse_version)})."
        )
    if min_alpha is None:
        notes.append("No main-assignment alpha gate: identical cell universe to Task A.1.")
    return notes


# -----------------------------------------------------------------------------
# The freeze orchestrator
# -----------------------------------------------------------------------------
def fill_a3_settings_folder(eval_dir: Path) -> Dict[str, Any]:
    """
    Fill the settings folder of an eval_id from settings/eval_config.yaml: slide pool, row universe, slide
    admission, class vocabulary, row manifests, splits, class protocols and protocol_manifest.json.
    Returns the protocol manifest dict.
    """
    eval_dir = Path(eval_dir)
    cfg = read_a3_eval_config(eval_dir)
    validate_a3_protocol_identity(cfg)
    paths = make_a3_protocol_paths(eval_dir)

    training_dataset_id = str(cfg["identity"]["training_dataset_id"])
    ct_cat_id = str(cfg["protocol"]["ct_cat_id"])
    granularity = str(cfg["protocol"]["label_granularity"])
    coarse_version = cfg["protocol"].get("coarse_mapping_version", None)
    coarse_version = str(coarse_version).strip() if coarse_version else None
    sub_eval = str(cfg["protocol"]["sub_eval_configuration"])

    # (1) Protocol slide pool.
    records = read_metadata(training_dataset_id)
    slide_records = _a3_protocol_slide_records(cfg, records)

    # (1b) Check that the modelgroup token-eligibility directory exists
    modelgroup = str(cfg["protocol"]["modelgroup"])
    missing_cells_dir = (
        resolve_eval_dataset_root(training_dataset_id)
        / f"cell_token_membership_missing_cells_{modelgroup}"
    )
    if not missing_cells_dir.exists():
        raise FileNotFoundError(
            f"Frozen structural-eligibility directory is missing for modelgroup='{modelgroup}': "
            f"{missing_cells_dir} (required for slide admission)."
        )

    # (2) Full row universe (Unknown retained for the diagnostics file).
    rows_full, token_eligibility = build_a3_row_universe(
        cfg=cfg, slide_records=slide_records, eval_dir=eval_dir
    )
    logger.info(
        "Modelgroup token-eligibility drop (applied before slide admission, modelgroup='%s'): "
        "%d of %d is_cell rows removed across %d slides.",
        modelgroup,
        int(token_eligibility["n_cells_missing_after_modelgroup_tokenization"].sum()),
        int(token_eligibility["n_is_cell_rows"].sum()),
        int(token_eligibility.shape[0]),
    )

    # (2b) Slide admission (before the class vocabulary)
    slide_admission = cfg["protocol"].get("slide_admission", {}) or {}
    admission_counts = a3_slide_admission_counts(
        rows_full,
        ct_cat_id=ct_cat_id,
        granularity=granularity,
        coarse_mapping_version=coarse_version,
        macro_floor=int(
            (cfg["protocol"].get("metrics", {}) or {}).get("macro_min_class_support_per_slide", 0)
        ),
    )
    kept_slide_ids, excluded_slides = a3_slide_admission_decision(
        admission_counts,
        min_cells=int(slide_admission.get("min_scorable_cells_per_slide", 0)),
        min_classes=int(slide_admission.get("min_classes_per_slide", 0)),
    )
    if excluded_slides:
        rows_full = rows_full.loc[
            rows_full["slide_id"].astype(str).isin(kept_slide_ids)
        ].reset_index(drop=True)
        slide_records = [r for r in slide_records if str(r.slide_id) in kept_slide_ids]
    # Admission counts and excluded slides (written even if no slide is excluded)
    admission_counts = admission_counts.merge(token_eligibility, on="slide_id", how="left")
    admission_counts.to_csv(paths.info_dir / "slide_admission_counts.csv", index=False)
    save_json(
        paths.manifests_dir / "excluded_slides.json",
        {
            "min_scorable_cells_per_slide": int(
                slide_admission.get("min_scorable_cells_per_slide", 0)
            ),
            "min_classes_per_slide": int(slide_admission.get("min_classes_per_slide", 0)),
            "label_granularity": granularity,
            "coarse_mapping_version": coarse_version,
            # Admission is counted on the modelgroup-token-eligible cells
            "admission_population": f"modelgroup_token_eligible_cells[{modelgroup}]",
            "n_cells_dropped_by_token_eligibility": int(
                token_eligibility["n_cells_missing_after_modelgroup_tokenization"].sum()
            ),
            "n_slides_total": int(admission_counts.shape[0]),
            "n_slides_admitted": int(len(kept_slide_ids)),
            "n_excluded": len(excluded_slides),
            "excluded": excluded_slides,
        },
    )

    # (3) Frozen scoring class vocabulary (support-filtered on the train pool, dense-reindexed).
    class_selection = cfg["protocol"]["class_selection"]
    min_cells = int(class_selection["min_train_cells_per_class"])
    min_slides = int(class_selection["min_train_slides_per_class"])
    min_classes = int(class_selection["min_trainable_classes"])
    # STHELAR_TEST_ADAPTED: a class is trainable if it has >= adapted_min_cells_per_slide cells in each of
    # >= adapted_min_slides slides (instead of min_train_slides_per_class)
    is_adapted = sub_eval == "STHELAR_TEST_ADAPTED"
    adapted_min_cells_per_slide = int(class_selection.get("adapted_min_cells_per_slide", min_cells))
    adapted_min_slides = int(class_selection.get("adapted_min_slides", 1))
    support_pool = _a3_support_pool(rows_full, sub_eval)

    def _select_trainable(granularity_x: str) -> List[str]:
        """Regime-aware trainable-class selection on ``support_pool`` for one granularity."""
        if is_adapted:
            per_slide_support = class_per_slide_support_table(
                support_pool,
                ct_cat_id=ct_cat_id,
                granularity=granularity_x,
                min_cells_per_slide=adapted_min_cells_per_slide,
                coarse_mapping_version=coarse_version,
            )
            return select_trainable_classes_within_slide(
                per_slide_support,
                min_cells_per_slide=adapted_min_cells_per_slide,
                min_slides=adapted_min_slides,
                min_classes=min_classes,
            )
        support = class_support_table(
            support_pool,
            ct_cat_id=ct_cat_id,
            granularity=granularity_x,
            coarse_mapping_version=coarse_version,
        )
        return select_trainable_classes(
            support, min_cells=min_cells, min_slides=min_slides, min_classes=min_classes
        )

    # Fine class vocabulary (scoring space of a fine eval; only a diagnostic for a coarse eval)
    label_space_fine: Optional[LabelSpace] = None
    try:
        fine_trainable = _select_trainable("fine")
        label_space_fine = build_label_space(
            ct_cat_id=ct_cat_id, granularity="fine", trainable_labels=fine_trainable
        )
    except (RuntimeError, ValueError):
        # Fine axis not buildable (re-raised only for a fine eval)
        if granularity == "fine":
            raise
        logger.warning(
            "Fine class vocabulary is degenerate under the class-support thresholds; continuing "
            "because the frozen scoring granularity is 'coarse'."
        )

    if granularity == "fine":
        scoring_space = label_space_fine
    else:
        coarse_trainable = _select_trainable("coarse")
        scoring_space = build_label_space(
            ct_cat_id=ct_cat_id,
            granularity="coarse",
            trainable_labels=coarse_trainable,
            coarse_mapping_version=coarse_version,
        )
        if label_space_fine is not None:
            logger.debug(
                "Coarse eval: fine diagnostic vocabulary has %d classes.",
                int(label_space_fine.n_classes()),
            )

    if scoring_space is None:
        raise RuntimeError(
            "Internal error: no frozen scoring class vocabulary was built for "
            f"granularity='{granularity}'."
        )

    # (4) Freeze row manifests (scored vs unknown; scored carry the dense label_id).
    rows_scored, unknown = freeze_a3_row_manifests(rows_full, scoring_space, paths)
    _write_a3_row_summary(rows_scored, unknown, paths.info_dir)

    # (5) Split manifest (Task A.1 builders); ADAPTED also freezes the stripe geometry
    splits_manifest = build_slide_level_cv_splits(cfg, rows_scored, paths.splits_dir)
    # Task A.3 validation policy (C selected by inner CV, or fixed C)
    inner_cv = bool((cfg["protocol"].get("probe", {}) or {}).get("inner_cv_select", False))
    splits_manifest["validation_policy"] = (
        "none_C_by_leave_one_train_group_out" if inner_cv else "none_fixed_C_no_capacity_selection"
    )
    save_json(paths.splits_dir / "splits_manifest.json", splits_manifest)
    if sub_eval == "STHELAR_TEST_ADAPTED":
        build_area_split_manifests(cfg, paths.area_dir, rows_scored)

    # (6) Class protocols (vocabulary / support / per-fold coverage).
    class_fragment = persist_a3_class_protocols(
        cfg=cfg,
        rows_scored_df=rows_scored,
        splits_manifest=splits_manifest,
        label_space=scoring_space,
        paths=paths,
    )

    # (7) Protocol manifest (label dict hashes: coarse_ct_mapping is null if no version is configured)
    label_dicts_sha256: Dict[str, Any] = {
        "cat2idx": sha256_file(ANNOT_DIR(ct_cat_id) / "cat2idx.json"),
        "coarse_ct_mapping": None,
    }
    if coarse_version is not None:
        coarse_fname = coarse_mapping_filename(coarse_version)
        label_dicts_sha256["coarse_ct_mapping"] = {
            "version": coarse_version,
            "filename": coarse_fname,
            "sha256": sha256_file(ANNOT_DIR(ct_cat_id) / coarse_fname),
        }
    protocol_manifest: Dict[str, Any] = {
        "created_utc": cfg["identity"]["created_utc"],
        "identity": cfg["identity"],
        "protocol": cfg["protocol"],
        # Value: protocol_identity_sha256(cfg)
        "eval_config_sha256": protocol_identity_sha256(cfg),
        "splits_manifest": str(paths.splits_dir / "splits_manifest.json"),
        "class_vocabulary": class_fragment["class_vocabulary"],
        "class_support": class_fragment["class_support"],
        "fold_class_coverage": class_fragment["fold_class_coverage"],
        # Fold-coverage floor and thinnest (fold, class) train-cell count
        "fold_coverage_min_train_cells_per_class_per_fold": class_fragment[
            "min_train_cells_per_class_per_fold"
        ],
        "thinnest_fold_class_train_cells": class_fragment["thinnest_fold_class_train_cells"],
        "area_split_manifest": (
            str(paths.area_dir / "area_split_manifest.json")
            if sub_eval == "STHELAR_TEST_ADAPTED"
            else ""
        ),
        "row_manifest": str(paths.rows_dir / "all_rows.parquet"),
        "unknown_cells": str(paths.rows_dir / "unknown_cells.parquet"),
        # Slide admission
        "excluded_slides": str(paths.manifests_dir / "excluded_slides.json"),
        "slide_admission_counts": str(paths.info_dir / "slide_admission_counts.csv"),
        "n_slides_total": int(admission_counts.shape[0]),
        "n_slides_admitted": int(len(kept_slide_ids)),
        "n_slides_excluded": len(excluded_slides),
        "slides_excluded": [str(e["slide_id"]) for e in excluded_slides],
        "label_dicts_sha256": label_dicts_sha256,
        "min_selected_alpha": cfg["protocol"].get("cell_filter", {}).get("min_selected_alpha", None),
        "benchmark_contract_notes": _a3_protocol_contract_notes(cfg),
    }
    save_json(paths.settings_dir / "protocol_manifest.json", protocol_manifest)

    logger.info(
        "Filled Task A.3 settings for eval_id='%s' (%s / %s%s, %d classes: %s).",
        str(cfg["identity"].get("eval_id", "")),
        sub_eval,
        granularity,
        "" if coarse_version is None else f" {coarse_version}",
        int(class_fragment["n_classes"]),
        class_fragment["labels"],
    )
    return protocol_manifest
