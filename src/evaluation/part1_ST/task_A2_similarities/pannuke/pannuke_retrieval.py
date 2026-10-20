"""PanNuke retrieval scoring: H&E queries, k-NN vote, per-tissue metrics vs chance floor, paired tests, diagnostics."""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Project imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src_A2.common import ensure_dir, entropy_from_probs, read_json, write_dataframe, write_json
from src_A2.embeddings import ensure_selected_main_rows, load_selected_he_embeddings
from src_A2.metrics import (
    assert_unscorable_labels_absent,
    class_scores_from_neighbors,
    classification_metrics_from_ids,
    entropy_metrics_from_class_scores,
    labels_to_ids,
    majority_class_metrics,
    predicted_labels_from_scores,
)
from src_A2.retrieval import l2_normalize, mapped_class_mass, retrieve_topk, weights_from_similarities
from pannuke import pannuke_embeddings as pe
from pannuke.pannuke_common import (
    COARSE_V2_SCORABLE_CLASSES,
    tissue_for_pseudo_slide_id,
)
from pannuke import pannuke_references as pr
from pannuke.pannuke_references import (
    LABEL_ORDER,
    LABEL_TO_INDEX,
    UNSCORABLE_LABELS,
    ReferencePool,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------
# Protocol values (from the eval_2 run manifest and config)
# ---------------------------------------------------------------------------------------------
# selected_k / selected_tau of eval_2/HEencoder_t46e13_clip/run_manifest.json

SELECTED_K: int = 30
SELECTED_TAU: float = 0.03

# `retrieval.Kmax`: retrieval at KMAX, then sliced to SELECTED_K (as in runner._evaluate_query_path)
KMAX: int = 200

SIMILARITY: str = "cosine_after_l2"
RETRIEVAL_BACKEND: str = "auto"

# `metrics.macro_min_class_support_per_slide` (per-tissue macro and chance floor only, not pooled values)
MACRO_MIN_CLASS_SUPPORT: int = 50

# `runtime.he_support.min_selected_alpha`. A cell is a usable query only above it.
MIN_SELECTED_ALPHA: float = 0.0

# `runtime.seed` (used by the paired bootstrap)
SEED: int = 13

# Scored populations and the headline one
POPULATIONS: Tuple[str, ...] = ("he_valid_all", "he_valid_covered")
HEADLINE_POPULATION: str = "he_valid_covered"

# Confidence thresholds of the abstention sweep (0.0 = no abstention)
CONFIDENCE_THRESHOLD_GRID: Tuple[float, ...] = (0.0,) + tuple(
    round(0.30 + 0.05 * i, 2) for i in range(14)
)

# BH correction families (named in paired_tests.csv)
FAMILY_PRIMARY: str = "primary_he_vs_floor"
FAMILY_SECONDARY: str = "secondary_he_vs_floor"
FAMILY_CROSS_RECIPE: str = "cross_recipe_agreement"

# Primary and secondary metrics of the paired tests
PRIMARY_METRIC: str = "macro_f1"
SECONDARY_METRICS: Tuple[str, ...] = ("accuracy", "balanced_accuracy", "weighted_f1")

# Expected number of cells embedded in both recipes (checked by _6_run_retrieval.py)
EXPECTED_PAIRED_CELLS: int = 135_327

SCORE_SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------

def recipe_results_dir(recipe_name: str) -> Path:
    """Results folder of one geometry recipe: <results_root>/eval_st_pannuke_<recipe>."""
    return pr.results_root() / f"eval_st_pannuke_{recipe_name}"


def checkpoint_results_dir(recipe_name: str, training_num: str, checkpoint_num: str) -> Path:
    """Per-encoder results folder: <recipe_results_dir>/HEencoder_t<n>e<c>_clip."""
    return recipe_results_dir(recipe_name) / f"HEencoder_t{training_num}e{checkpoint_num}_clip"


def cross_recipe_results_dir(training_num: str, checkpoint_num: str) -> Path:
    """Results folder of the native224 vs scale224 comparison."""
    return pr.results_root() / "cross_recipe" / f"HEencoder_t{training_num}e{checkpoint_num}_clip"


def predictions_path(checkpoint_dir: Path, slide_id: str) -> Path:
    return Path(checkpoint_dir) / "predictions" / f"{slide_id}.parquet"


def retrieval_path(checkpoint_dir: Path, slide_id: str) -> Path:
    return Path(checkpoint_dir) / "retrieval" / f"{slide_id}_topk{SELECTED_K}.npz"


def confusion_path(checkpoint_dir: Path, slide_id: str, population: str) -> Path:
    return Path(checkpoint_dir) / "metrics" / "confusion" / f"{slide_id}_{population}.npy"


def run_manifest_path(checkpoint_dir: Path) -> Path:
    return Path(checkpoint_dir) / "run_manifest.json"


# ---------------------------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringPreflight:
    """One recipe's prepared + embedded tree, validated and measured, before any retrieval."""

    recipe_name: str
    prepared_root: Path
    emb_dir: Path
    membership_group: str
    preflight: Any = field(repr=False)
    extract_manifest: Dict[str, Any] = field(repr=False)

    @property
    def slide_ids(self) -> Tuple[str, ...]:
        return self.preflight.slide_ids


#: Number of trailing path components compared between a recorded path and the local one
PATH_TAIL_PARTS: int = 2


def _relative_tail(path_string: str, n_parts: int = PATH_TAIL_PARTS) -> str:
    """The last n_parts components of a path (compared instead of the absolute path)."""
    return "/".join(Path(str(path_string)).parts[-int(n_parts):])


def _recipe_name_from_manifest(manifest: Mapping[str, Any], path: Path) -> str:
    """Recipe name from the geometry_recipe entry of extract_manifest.json (a dict with 'name', or a string)."""
    block = manifest.get("geometry_recipe")
    if isinstance(block, Mapping):
        name = str(block.get("name", "")).strip()
        if not name:
            raise ValueError(
                f"{path} has a geometry_recipe block with no 'name' key (keys: "
                f"{sorted(block)[:10]})."
            )
        return name
    if isinstance(block, str) and block.strip():
        return block.strip()
    raise ValueError(
        f"{path} has no usable geometry_recipe (got {type(block).__name__}). Expected the recipe "
        "block of prepare_manifest.json or a recipe name."
    )


def preflight_for_scoring(
    recipe_name: str,
    training_num: str,
    checkpoint_num: str,
    slides: Optional[Sequence[str]] = None,
    allow_partial_tree: bool = False,
    log: Optional[logging.Logger] = None,
) -> ScoringPreflight:
    """
    Check one recipe's prepared and embedded tree, with roots resolved locally: preflight_recipe of
    pannuke_embeddings, plus embedding width, main-row counts, one candidate patch per cell, selected_alpha > 0.
    """
    log = log or logger
    from pannuke.pannuke_common import prepared_root as _prepared_root

    prepared = _prepared_root(pr.data_root(), recipe_name)
    manifest = pe.read_prepare_manifest(prepared)
    membership_group = str((manifest.get("membership") or {}).get("group_name", ""))
    if not membership_group:
        raise ValueError(
            f"{prepared}/prepare_manifest.json declares no membership.group_name."
        )

    preflight = pe.preflight_recipe(
        prepared_root_path=prepared,
        recipe_name=recipe_name,
        membership_group=membership_group,
        slides=slides,
        allow_partial_tree=allow_partial_tree,
    )
    emb_dir = pe.he_embeddings_dir(prepared, training_num, checkpoint_num)
    if not emb_dir.is_dir():
        raise FileNotFoundError(
            f"No H&E embeddings for t{training_num}e{checkpoint_num} under {emb_dir}. Run "
            f"_5_compute_HEembs.py --geometry_recipe {recipe_name} --training_num "
            f"{training_num} --checkpoint_num {checkpoint_num} first."
        )

    extract_manifest_path = emb_dir / pe.EXTRACT_MANIFEST_NAME
    if not extract_manifest_path.is_file():
        raise FileNotFoundError(
            f"No {pe.EXTRACT_MANIFEST_NAME} under {emb_dir}."
        )
    extract_manifest = read_json(extract_manifest_path)

    derived_group = str((extract_manifest.get("encoder") or {}).get("membership_group_derived", ""))
    if derived_group != membership_group:
        raise ValueError(
            f"Membership-group disagreement for '{recipe_name}': prepare_manifest.json "
            f"says {membership_group!r}, extract_manifest.json derived {derived_group!r} "
            "from the encoder."
        )
    got_recipe = _recipe_name_from_manifest(extract_manifest, extract_manifest_path)
    if got_recipe != recipe_name:
        raise ValueError(
            f"{extract_manifest_path} describes recipe {got_recipe!r}, not {recipe_name!r}."
        )
    recorded_root = str((extract_manifest.get("prepared_tree") or {}).get("root", ""))
    if recorded_root and _relative_tail(recorded_root) != _relative_tail(str(prepared)):
        raise ValueError(
            f"The prepared-tree tail recorded in extract_manifest.json {_relative_tail(recorded_root)!r} does "
            f"not match the locally resolved {_relative_tail(str(prepared))!r} (recorded path: {recorded_root!r})."
        )
    log.info(
        "Recipe '%s': extract_manifest.json recorded root %r; scoring the locally resolved %s.",
        recipe_name,
        recorded_root,
        prepared,
    )

    for exp in preflight.slides:
        _assert_slide_embeddings_usable(emb_dir, exp)

    return ScoringPreflight(
        recipe_name=recipe_name,
        prepared_root=Path(prepared),
        emb_dir=Path(emb_dir),
        membership_group=membership_group,
        preflight=preflight,
        extract_manifest=extract_manifest,
    )


def _assert_slide_embeddings_usable(emb_dir: Path, exp: Any) -> None:
    """Check one slide's embedding h5 and main-row parquet before scoring. Raises on the first problem."""
    import h5py
    import pyarrow.parquet as pq

    h5_path = pe.embedding_h5_path(emb_dir, exp.slide_id)
    parquet_path = pe.main_rows_parquet_path(emb_dir, exp.slide_id)
    if not h5_path.is_file():
        raise FileNotFoundError(f"Missing embedding h5 for '{exp.slide_id}': {h5_path}")
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Missing selected-main-row parquet for '{exp.slide_id}': {parquet_path}."
        )

    with h5py.File(h5_path, "r") as f:
        if pe.STEP4_REQUIRED_H5_DATASET not in f:
            raise KeyError(
                f"'{exp.slide_id}': {h5_path} has no '{pe.STEP4_REQUIRED_H5_DATASET}'."
            )
        shape = f[pe.STEP4_REQUIRED_H5_DATASET].shape
        n_rows, width = int(shape[0]), int(shape[1])
    if width <= 0:
        raise RuntimeError(
            f"'{exp.slide_id}': {pe.STEP4_REQUIRED_H5_DATASET} has width 0 (encoder with no trained "
            "token projection)."
        )
    if n_rows != int(exp.n_expected_groups):
        raise RuntimeError(
            f"'{exp.slide_id}': {pe.STEP4_REQUIRED_H5_DATASET} holds {n_rows:,} rows but the prepared tree "
            f"declares {int(exp.n_expected_groups):,} token-eligible cells."
        )

    table = pq.read_table(
        parquet_path, columns=["cell_id_int", "selected_alpha", "n_candidate_patches"]
    )
    if table.num_rows != int(exp.n_expected_groups):
        raise RuntimeError(
            f"'{exp.slide_id}': {parquet_path.name} holds {table.num_rows:,} rows but the prepared tree "
            f"declares {int(exp.n_expected_groups):,} token-eligible cells."
        )
    alpha = table.column("selected_alpha").to_numpy(zero_copy_only=False).astype(np.float64)
    if not np.all(alpha > MIN_SELECTED_ALPHA):
        n_bad = int((~(alpha > MIN_SELECTED_ALPHA)).sum())
        raise RuntimeError(
            f"'{exp.slide_id}': {n_bad:,} cell(s) have selected_alpha <= {MIN_SELECTED_ALPHA}."
        )
    n_cand = table.column("n_candidate_patches").to_numpy(zero_copy_only=False).astype(np.int64)
    if n_cand.size and int(n_cand.max()) != 1:
        raise RuntimeError(
            f"'{exp.slide_id}': n_candidate_patches reaches {int(n_cand.max())}, not 1 (PanNuke "
            "tiles do not overlap)."
        )


# ---------------------------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------------------------

@dataclass
class SlideQueries:
    """One pseudo-slide's H&E cell embeddings, joined to its ground truth."""

    slide_id: str
    tissue_pannuke: str
    tissue_histost: str
    cells: pd.DataFrame = field(repr=False)
    z_he: np.ndarray = field(repr=False)
    he_valid: np.ndarray = field(repr=False)

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])


def load_slide_queries(
    preflight: ScoringPreflight,
    slide_id: str,
    checkpoint_dir: Path,
    training_num: str,
    checkpoint_num: str,
    tissue_histost_by_pannuke: Mapping[str, str],
) -> SlideQueries:
    """
    Load one pseudo-slide's H&E cell embeddings (ensure_selected_main_rows copies the main-row parquet into
    <checkpoint_dir>/selected_main_he_rows/) and join them to the ground truth on cell_id_int.
    """
    ground_truth = pd.read_parquet(pe.ground_truth_path(preflight.prepared_root, slide_id))
    selected_rows = ensure_selected_main_rows(
        Path(checkpoint_dir), preflight.prepared_root, training_num, checkpoint_num, slide_id
    )
    he = load_selected_he_embeddings(
        pe.embedding_h5_path(preflight.emb_dir, slide_id), selected_rows, require_clip=True
    )
    if he.table.empty:
        raise RuntimeError(f"'{slide_id}' has zero embedded cells.")

    embedded_ids = he.table["cell_id_int"].to_numpy(dtype=np.int64)
    gt_ids = ground_truth["cell_id_int"].to_numpy(dtype=np.int64)
    missing_from_gt = np.setdiff1d(embedded_ids, gt_ids)
    if missing_from_gt.size:
        raise RuntimeError(
            f"'{slide_id}': {missing_from_gt.size:,} embedded cell id(s) have no ground-truth row, "
            f"e.g. {missing_from_gt[:5].tolist()}."
        )

    # Both tables carry cell_id: checked equal, then the ground-truth copy is dropped
    cells = he.table.merge(
        ground_truth, on="cell_id_int", how="left", validate="one_to_one", suffixes=("", "_gt")
    )
    if cells.shape[0] != he.table.shape[0]:
        raise RuntimeError(f"'{slide_id}': the ground-truth join changed the row count.")
    if "cell_id_gt" in cells.columns:
        mismatch = cells["cell_id"].astype(str) != cells["cell_id_gt"].astype(str)
        if bool(mismatch.any()):
            raise RuntimeError(
                f"'{slide_id}': {int(mismatch.sum()):,} cell(s) whose `cell_id` differs between "
                "the selected-main-row parquet and the ground truth, e.g. "
                f"{cells.loc[mismatch, 'cell_id_int'].head(5).tolist()}."
            )
        cells = cells.drop(columns=["cell_id_gt"])

    tissue_pannuke = tissue_for_pseudo_slide_id(slide_id)
    observed_tissues = sorted(set(cells["tissue"].astype(str)))
    if observed_tissues != [tissue_pannuke]:
        raise RuntimeError(
            f"'{slide_id}': ground truth carries tissues {observed_tissues}, expected "
            f"['{tissue_pannuke}']."
        )
    observed_histost = sorted(set(cells["tissue_HistoST"].astype(str)))
    expected_histost = tissue_histost_by_pannuke[tissue_pannuke]
    if observed_histost != [expected_histost]:
        raise RuntimeError(
            f"'{slide_id}': ground truth carries tissue_HistoST {observed_histost}, expected "
            f"['{expected_histost}']."
        )

    from pannuke.pannuke_references import safe_embedding_validity

    z_he = np.ascontiguousarray(he.z_he, dtype=np.float32)
    alpha = cells["selected_alpha"].astype(float).to_numpy()
    he_valid = safe_embedding_validity(z_he) & (alpha > MIN_SELECTED_ALPHA)

    cells = cells.assign(
        slide_id=slide_id,
        tissue_pannuke=tissue_pannuke,
        he_valid=he_valid.astype(np.int8),
    )
    return SlideQueries(
        slide_id=slide_id,
        tissue_pannuke=tissue_pannuke,
        tissue_histost=expected_histost,
        cells=cells,
        z_he=z_he,
        he_valid=he_valid,
    )


# ---------------------------------------------------------------------------------------------
# Vote + score
# ---------------------------------------------------------------------------------------------

def mass_column(label: str) -> str:
    """Column name of one class's weighted neighbour mass in the per-cell prediction table."""
    return f"mass_{label}"


def weighted_f1_from_per_class(per_class: pd.DataFrame) -> float:
    """Support-weighted mean F1 over the GT-present classes (not floored), from a per-class table."""
    support = per_class["support"].to_numpy(dtype=np.float64)
    f1 = per_class["f1"].to_numpy(dtype=np.float64)
    total = float(support.sum())
    if total <= 0:
        return float("nan")
    return float((f1 * support).sum() / total)


def _majority_weighted_f1(y_true_ids: np.ndarray, floor: Mapping[str, Any]) -> float:
    """Support-weighted F1 of the majority-class floor, on the same cells and class axis."""
    y = np.asarray(y_true_ids).reshape(-1)
    y = y[y >= 0].astype(np.int64, copy=False)
    if y.size == 0:
        return float("nan")
    y_pred = np.full(y.shape, int(floor["majority_class_index"]), dtype=np.int64)
    summary = classification_metrics_from_ids(
        y, y_pred, label_order=LABEL_ORDER, macro_min_support=MACRO_MIN_CLASS_SUPPORT
    )
    return weighted_f1_from_per_class(summary.per_class)


@dataclass
class SlideScores:
    """Everything one pseudo-slide contributes to the outputs."""

    slide_id: str
    tissue_pannuke: str
    tissue_histost: str
    predictions: pd.DataFrame = field(repr=False)
    per_tissue: List[Dict[str, Any]] = field(default_factory=list)
    per_class: List[pd.DataFrame] = field(default_factory=list, repr=False)
    chance_floor: List[Dict[str, Any]] = field(default_factory=list)
    confusions: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    nn_idx: Optional[np.ndarray] = field(default=None, repr=False)
    nn_sim: Optional[np.ndarray] = field(default=None, repr=False)
    valid_rows: Optional[np.ndarray] = field(default=None, repr=False)


def score_slide(
    queries: SlideQueries,
    pool: ReferencePool,
    *,
    ref_z_norm: Optional[np.ndarray] = None,
    log: Optional[logging.Logger] = None,
) -> SlideScores:
    """
    Retrieve, vote and score one pseudo-slide (as runner._precompute_retrieval + runner._evaluate_query_path):
    retrieve at KMAX, keep k, softmax the similarities at tau, sum the weights per class, take the argmax.
    Each population is scored with its majority-class floor (same cells and class axis, checked).
    """
    log = log or logger
    cells = queries.cells
    n_cells = int(cells.shape[0])

    gt_labels = cells["coarse_v2_class"].astype(str).to_numpy()
    gt_label_ids = labels_to_ids(gt_labels, LABEL_TO_INDEX)
    if np.any(gt_label_ids < 0):
        missing = sorted(set(gt_labels[gt_label_ids < 0].tolist()))
        raise ValueError(
            f"'{queries.slide_id}': ground-truth labels absent from LABEL_ORDER "
            f"{list(LABEL_ORDER)}: {missing}."
        )
    gt_scorable = ~np.isin(gt_labels, np.asarray(UNSCORABLE_LABELS, dtype=object))
    covered = np.isin(gt_labels, np.unique(pool.labels))

    valid_rows = np.flatnonzero(queries.he_valid).astype(np.int64)
    if valid_rows.size == 0:
        raise RuntimeError(
            f"'{queries.slide_id}' has zero valid query cells; there is nothing to score."
        )

    if ref_z_norm is None:
        ref_z_norm = l2_normalize(pool.z_ref)

    log.info(
        "Retrieval slide=%s queries=%d ref=%d Kmax=%d score_pairs=%.3fB",
        queries.slide_id,
        int(valid_rows.size),
        int(ref_z_norm.shape[0]),
        KMAX,
        float(valid_rows.size * ref_z_norm.shape[0]) / 1.0e9,
    )
    retrieval = retrieve_topk(
        queries.z_he[valid_rows],
        ref_z_norm,
        K=KMAX,
        backend=RETRIEVAL_BACKEND,
        references_normalized=True,
    )

    # As in runner._evaluate_query_path
    k = min(int(SELECTED_K), int(retrieval.indices.shape[1]))
    nn_idx = retrieval.indices[:, :k]
    nn_sim = retrieval.similarities[:, :k]
    weights = weights_from_similarities(nn_sim, tau=float(SELECTED_TAU))
    class_scores = class_scores_from_neighbors(
        nn_idx, weights, pool.label_ids, n_classes=len(LABEL_ORDER)
    )
    pred_ids_valid = predicted_labels_from_scores(class_scores)
    pred_labels_valid = np.asarray(LABEL_ORDER, dtype=object)[pred_ids_valid]

    pred_label_ids = np.full(n_cells, -1, dtype=np.int32)
    pred_label_ids[valid_rows] = pred_ids_valid
    pred_labels = np.full(n_cells, "", dtype=object)
    pred_labels[valid_rows] = pred_labels_valid

    class_scores_full = np.zeros((n_cells, len(LABEL_ORDER)), dtype=np.float32)
    class_scores_full[valid_rows] = class_scores
    entropy_full = np.full(n_cells, np.nan, dtype=np.float32)
    entropy_full[valid_rows] = entropy_metrics_from_class_scores(class_scores)
    weight_entropy_full = np.full(n_cells, np.nan, dtype=np.float32)
    weight_entropy_full[valid_rows] = entropy_from_probs(weights, axis=1).astype(np.float32)

    gt_mass_valid, gt_in_topk_valid = mapped_class_mass(
        nn_idx, weights, pool.label_ids, gt_label_ids[valid_rows]
    )
    gt_mass_full = np.full(n_cells, np.nan, dtype=np.float32)
    gt_mass_full[valid_rows] = gt_mass_valid
    gt_in_topk_full = np.zeros(n_cells, dtype=np.int8)
    gt_in_topk_full[valid_rows] = gt_in_topk_valid.astype(np.int8)

    population_masks = {
        "he_valid_all": queries.he_valid & gt_scorable,
        "he_valid_covered": queries.he_valid & gt_scorable & covered,
    }

    scores = SlideScores(
        slide_id=queries.slide_id,
        tissue_pannuke=queries.tissue_pannuke,
        tissue_histost=queries.tissue_histost,
        predictions=pd.DataFrame(),
        nn_idx=nn_idx,
        nn_sim=nn_sim,
        valid_rows=valid_rows,
    )

    for population in POPULATIONS:
        rows = np.flatnonzero(population_masks[population])
        if rows.size == 0:
            raise RuntimeError(
                f"'{queries.slide_id}': population '{population}' is empty."
            )
        cls = classification_metrics_from_ids(
            gt_label_ids[rows],
            pred_label_ids[rows],
            label_order=LABEL_ORDER,
            macro_min_support=MACRO_MIN_CLASS_SUPPORT,
        )
        # Unscorable labels must be absent from the per-class table
        assert_unscorable_labels_absent(
            cls.per_class,
            UNSCORABLE_LABELS,
            context=f"slide={queries.slide_id} population={population}",
        )
        floor = majority_class_metrics(
            gt_label_ids[rows],
            label_order=LABEL_ORDER,
            macro_min_support=MACRO_MIN_CLASS_SUPPORT,
        )
        _assert_floor_shares_class_axis(queries.slide_id, population, cls.metrics, floor)
        floor_weighted_f1 = _majority_weighted_f1(gt_label_ids[rows], floor)

        support = cls.per_class["support"].to_numpy(dtype=np.int64)
        n_present_unfloored = int((support > 0).sum())
        n_macro_classes = int(cls.metrics["n_gt_present_classes"])

        scores.per_tissue.append(
            {
                "slide_id": queries.slide_id,
                "tissue_pannuke": queries.tissue_pannuke,
                "tissue_HistoST": queries.tissue_histost,
                "population": population,
                "is_headline_population": int(population == HEADLINE_POPULATION),
                "n_cells": int(cls.metrics["n_cells"]),
                "macro_f1": float(cls.metrics["macro_f1"]),
                "balanced_accuracy": float(cls.metrics["balanced_accuracy"]),
                "accuracy": float(cls.metrics["accuracy"]),
                "weighted_f1": weighted_f1_from_per_class(cls.per_class),
                "n_gt_present_classes": n_macro_classes,
                "n_gt_present_classes_unfloored": n_present_unfloored,
                "n_classes_below_macro_floor": n_present_unfloored - n_macro_classes,
                "macro_min_class_support": MACRO_MIN_CLASS_SUPPORT,
                "n_reference_cells": int(pool.n_rows),
                "reference_stem": pool.reference_stem,
            }
        )
        per_class = cls.per_class.copy()
        per_class.insert(0, "population", population)
        per_class.insert(0, "tissue_HistoST", queries.tissue_histost)
        per_class.insert(0, "tissue_pannuke", queries.tissue_pannuke)
        per_class.insert(0, "slide_id", queries.slide_id)
        scores.per_class.append(per_class)
        scores.confusions[population] = cls.confusion_matrix
        scores.chance_floor.append(
            {
                "slide_id": queries.slide_id,
                "tissue_pannuke": queries.tissue_pannuke,
                "tissue_HistoST": queries.tissue_histost,
                "population": population,
                "majority_class_label": floor["majority_class_label"],
                "majority_fraction": float(floor["majority_fraction"]),
                "n_cells": int(floor["n_cells"]),
                "macro_f1": float(floor["macro_f1"]),
                "balanced_accuracy": float(floor["balanced_accuracy"]),
                "accuracy": float(floor["accuracy"]),
                "weighted_f1": floor_weighted_f1,
                "n_gt_present_classes": int(floor["n_gt_present_classes"]),
                "macro_min_class_support": MACRO_MIN_CLASS_SUPPORT,
            }
        )

    predictions = pd.DataFrame(
        {
            "slide_id": queries.slide_id,
            "tissue_pannuke": queries.tissue_pannuke,
            "tissue_HistoST": queries.tissue_histost,
            "cell_id_int": cells["cell_id_int"].to_numpy(dtype=np.int64),
            "cell_id": cells["cell_id"].astype(str).to_numpy(),
            "patch_id": cells["patch_id"].astype(str).to_numpy(),
            "fold": cells["fold"].astype(str).to_numpy(),
            "pannuke_class": cells["pannuke_class"].astype(str).to_numpy(),
            "gt_label": gt_labels,
            "gt_scorable": gt_scorable.astype(np.int8),
            "covered": covered.astype(np.int8),
            "he_valid": queries.he_valid.astype(np.int8),
            "pred_label": pred_labels,
            "pred_label_id": pred_label_ids,
            "confidence": class_scores_full.max(axis=1).astype(np.float32),
            "class_score_entropy": entropy_full,
            "neighbor_weight_entropy": weight_entropy_full,
            "gt_class_mass": gt_mass_full,
            "gt_class_in_topk": gt_in_topk_full,
            "touches_border": cells["touches_border"].astype(bool).to_numpy().astype(np.int8),
            "nuc_px_membership": cells["nuc_px_membership"].to_numpy(),
            "selected_alpha": cells["selected_alpha"].astype(float).to_numpy(),
            "n_candidate_patches": cells["n_candidate_patches"].astype(int).to_numpy(),
        }
    )
    for i, label in enumerate(LABEL_ORDER):
        predictions[mass_column(label)] = class_scores_full[:, i]
    scores.predictions = predictions
    return scores


def _assert_floor_shares_class_axis(
    slide_id: str,
    population: str,
    headline: Mapping[str, float],
    floor: Mapping[str, Any],
) -> None:
    """Raise unless the chance floor and the headline have the same number of cells and of GT-present classes."""
    if int(floor["n_cells"]) != int(headline["n_cells"]):
        raise RuntimeError(
            f"slide={slide_id} population={population}: the chance floor was computed on "
            f"{int(floor['n_cells']):,} cells and the headline on {int(headline['n_cells']):,}."
        )
    if int(floor["n_gt_present_classes"]) != int(headline["n_gt_present_classes"]):
        raise RuntimeError(
            f"slide={slide_id} population={population}: the chance floor averages over "
            f"{int(floor['n_gt_present_classes'])} class(es) and the headline over "
            f"{int(headline['n_gt_present_classes'])}."
        )


# ---------------------------------------------------------------------------------------------
# Cohort aggregation
# ---------------------------------------------------------------------------------------------

COHORT_METRICS: Tuple[str, ...] = ("macro_f1", "balanced_accuracy", "accuracy", "weighted_f1")


def aggregate_cohort(
    per_tissue: pd.DataFrame,
    chance_floor: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cohort statistics of each population, for H&E and the majority-class floor: unweighted mean over the
    tissues (headline) and pooled over cells (one confusion matrix, macro_min_support=0).
    """
    rows: List[Dict[str, Any]] = []
    for population in POPULATIONS:
        sub = per_tissue.loc[per_tissue["population"] == population]
        floor_sub = chance_floor.loc[chance_floor["population"] == population]
        for metric in COHORT_METRICS:
            rows.append(
                {
                    "population": population,
                    "is_headline_population": int(population == HEADLINE_POPULATION),
                    "summary_scope": "unweighted_tissue_mean",
                    "metric": metric,
                    "he_value": float(sub[metric].mean()),
                    "floor_value": (
                        float(floor_sub[metric].mean()) if metric in floor_sub.columns else float("nan")
                    ),
                    "n_tissues": int(sub["slide_id"].nunique()),
                    "n_cells": int(sub["n_cells"].sum()),
                }
            )

        mask = (
            (predictions["he_valid"].to_numpy() == 1)
            & (predictions["gt_scorable"].to_numpy() == 1)
        )
        if population == "he_valid_covered":
            mask = mask & (predictions["covered"].to_numpy() == 1)
        pooled = predictions.loc[mask]
        y_true = labels_to_ids(pooled["gt_label"].astype(str).tolist(), LABEL_TO_INDEX)
        y_pred = labels_to_ids(pooled["pred_label"].astype(str).tolist(), LABEL_TO_INDEX)
        cls = classification_metrics_from_ids(y_true, y_pred, label_order=LABEL_ORDER)
        floor = majority_class_metrics(y_true, label_order=LABEL_ORDER)
        pooled_values = {
            "macro_f1": float(cls.metrics["macro_f1"]),
            "balanced_accuracy": float(cls.metrics["balanced_accuracy"]),
            "accuracy": float(cls.metrics["accuracy"]),
            "weighted_f1": weighted_f1_from_per_class(cls.per_class),
        }
        floor_values = {
            "macro_f1": float(floor["macro_f1"]),
            "balanced_accuracy": float(floor["balanced_accuracy"]),
            "accuracy": float(floor["accuracy"]),
            "weighted_f1": _majority_weighted_f1(y_true, floor),
        }
        for metric in COHORT_METRICS:
            rows.append(
                {
                    "population": population,
                    "is_headline_population": int(population == HEADLINE_POPULATION),
                    "summary_scope": "pooled_over_cells",
                    "metric": metric,
                    "he_value": pooled_values[metric],
                    "floor_value": floor_values[metric],
                    "n_tissues": int(pooled["slide_id"].nunique()),
                    "n_cells": int(cls.metrics["n_cells"]),
                }
            )
    out = pd.DataFrame(rows)
    out["he_minus_floor"] = out["he_value"] - out["floor_value"]
    return out


def assert_populations_agree(per_tissue: pd.DataFrame, log: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Report whether he_valid_all and he_valid_covered have the same cells on every tissue."""
    log = log or logger
    wide = per_tissue.pivot_table(index="slide_id", columns="population", values="n_cells")
    identical = bool((wide["he_valid_all"] == wide["he_valid_covered"]).all())
    record = {
        "populations_identical": identical,
        "n_cells_he_valid_all": int(wide["he_valid_all"].sum()),
        "n_cells_he_valid_covered": int(wide["he_valid_covered"].sum()),
        "tissues_where_they_differ": sorted(
            wide.index[wide["he_valid_all"] != wide["he_valid_covered"]].astype(str).tolist()
        ),
        "note": (
            "`all` and `covered` coincide on a tissue when all its scorable classes are covered "
            "by its reference."
        ),
    }
    if identical:
        log.info(
            "he_valid_all == he_valid_covered on all %d tissues (%d cells): every scorable class "
            "is covered by its reference.",
            int(wide.shape[0]),
            record["n_cells_he_valid_all"],
        )
    else:
        log.warning(
            "he_valid_all and he_valid_covered differ on %s: a GT class is not covered by the "
            "reference there.",
            record["tissues_where_they_differ"],
        )
    return record


# ---------------------------------------------------------------------------------------------
# Paired tests
# ---------------------------------------------------------------------------------------------

def _paired_row(
    result: Any, *, family: str, recipe: str, metric: str, population: str, label: str
) -> Dict[str, Any]:
    return {
        "family": family,
        "recipe": recipe,
        "population": population,
        "metric": metric,
        "comparison": label,
        "delta": result.delta,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "p_value": result.p_value,
        "q_value": result.q_value,
        "fdr_applied": bool(result.fdr_applied),
        "n_units": int(result.n),
        "n_zero": int(result.n_zero),
        "statistic": result.statistic,
        "notes": result.notes,
    }


def paired_he_vs_floor(
    per_tissue_by_recipe: Mapping[str, pd.DataFrame],
    floor_by_recipe: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Paired tests of H&E vs the majority-class floor over the tissues, with BH correction per family:
    primary (macro_f1) and secondary (accuracy, balanced accuracy, weighted F1), both recipes in each.
    """
    from src.evaluation._shared.paired_stats import apply_fdr, paired_test

    def _run(metrics: Sequence[str], family: str) -> List[Dict[str, Any]]:
        results, rows = [], []
        for recipe in sorted(per_tissue_by_recipe):
            he = per_tissue_by_recipe[recipe]
            fl = floor_by_recipe[recipe]
            he = he.loc[he["population"] == HEADLINE_POPULATION].sort_values("slide_id")
            fl = fl.loc[fl["population"] == HEADLINE_POPULATION].sort_values("slide_id")
            for metric in metrics:
                res = paired_test(
                    he[metric].to_numpy(dtype=float),
                    fl[metric].to_numpy(dtype=float),
                    metric=metric,
                    direction="higher",
                    keys_a=he["slide_id"].astype(str).tolist(),
                    keys_b=fl["slide_id"].astype(str).tolist(),
                    seed=SEED,
                )
                results.append(res)
                rows.append((family, recipe, metric))
        corrected = apply_fdr(results)
        return [
            _paired_row(
                res,
                family=fam,
                recipe=rec,
                metric=met,
                population=HEADLINE_POPULATION,
                label="he_vs_chance_floor",
            )
            for res, (fam, rec, met) in zip(corrected, rows)
        ]

    out = _run([PRIMARY_METRIC], FAMILY_PRIMARY) + _run(list(SECONDARY_METRICS), FAMILY_SECONDARY)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------
# Diagnostics (computed from the per-cell prediction table)
# ---------------------------------------------------------------------------------------------

def _scored_mask(predictions: pd.DataFrame, population: str) -> np.ndarray:
    mask = (predictions["he_valid"].to_numpy() == 1) & (predictions["gt_scorable"].to_numpy() == 1)
    if population == "he_valid_covered":
        mask = mask & (predictions["covered"].to_numpy() == 1)
    return mask


def confidence_sweep(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Accuracy, macro-F1 and per-class recall vs a threshold on the confidence (winning class mass), pooled
    over cells (macro_min_support=0), plus the unweighted tissue mean of accuracy.
    """
    rows: List[Dict[str, Any]] = []
    for population in POPULATIONS:
        scored = predictions.loc[_scored_mask(predictions, population)]
        n_total = int(scored.shape[0])
        y_true_all = labels_to_ids(scored["gt_label"].astype(str).tolist(), LABEL_TO_INDEX)
        y_pred_all = labels_to_ids(scored["pred_label"].astype(str).tolist(), LABEL_TO_INDEX)
        confidence = scored["confidence"].to_numpy(dtype=float)
        slides = scored["slide_id"].astype(str).to_numpy()

        for threshold in CONFIDENCE_THRESHOLD_GRID:
            keep = confidence >= float(threshold)
            n_keep = int(keep.sum())
            base = {
                "population": population,
                "is_headline_population": int(population == HEADLINE_POPULATION),
                "confidence_threshold": float(threshold),
                "n_cells_total": n_total,
                "n_cells_retained": n_keep,
                "retained_fraction": float(n_keep) / float(max(1, n_total)),
            }
            if n_keep == 0:
                rows.append({**base, "scope": "pooled", "label": "ALL", "metric_ok": 0})
                continue
            cls = classification_metrics_from_ids(
                y_true_all[keep], y_pred_all[keep], label_order=LABEL_ORDER
            )
            tissue_acc = [
                float(np.mean(y_true_all[keep & (slides == s)] == y_pred_all[keep & (slides == s)]))
                for s in sorted(set(slides.tolist()))
                if int((keep & (slides == s)).sum()) > 0
            ]
            rows.append(
                {
                    **base,
                    "scope": "pooled",
                    "label": "ALL",
                    "metric_ok": 1,
                    "accuracy": float(cls.metrics["accuracy"]),
                    "macro_f1": float(cls.metrics["macro_f1"]),
                    "balanced_accuracy": float(cls.metrics["balanced_accuracy"]),
                    "n_gt_present_classes": int(cls.metrics["n_gt_present_classes"]),
                    "unweighted_tissue_mean_accuracy": float(np.mean(tissue_acc)) if tissue_acc else float("nan"),
                    "n_tissues_contributing": len(tissue_acc),
                }
            )
            for _, row in cls.per_class.iterrows():
                if str(row["label"]) in set(UNSCORABLE_LABELS):
                    continue
                rows.append(
                    {
                        **base,
                        "scope": "per_class",
                        "label": str(row["label"]),
                        "metric_ok": 1,
                        "support": int(row["support"]),
                        "pred_support": int(row["pred_support"]),
                        "precision": float(row["precision"]),
                        "recall": float(row["recall"]),
                        "f1": float(row["f1"]),
                    }
                )
    return pd.DataFrame(rows)


def epithelial_by_source_class(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Recall of Epithelial_Melanocyte split by the PanNuke source class of the GT cell (Neoplastic or
    Epithelial), for the cohort and per tissue. Precision is undefined per source class (not reported).
    """
    target = "Epithelial_Melanocyte"
    rows: List[Dict[str, Any]] = []
    for population in POPULATIONS:
        scored = predictions.loc[_scored_mask(predictions, population)]
        scored = scored.loc[scored["gt_label"].astype(str) == target]
        for scope, frame in [("cohort", scored)] + [
            ("tissue", g) for _, g in scored.groupby("slide_id", sort=True)
        ]:
            if frame.empty:
                continue
            n_class_total = int(frame.shape[0])
            for source_class, sub in frame.groupby("pannuke_class", sort=True):
                hit = (sub["pred_label"].astype(str) == target).to_numpy()
                rows.append(
                    {
                        "population": population,
                        "is_headline_population": int(population == HEADLINE_POPULATION),
                        "scope": scope,
                        "slide_id": "COHORT" if scope == "cohort" else str(sub["slide_id"].iloc[0]),
                        "tissue_pannuke": "COHORT" if scope == "cohort" else str(sub["tissue_pannuke"].iloc[0]),
                        "gt_class": target,
                        "source_pannuke_class": str(source_class),
                        "n_source_cells": int(sub.shape[0]),
                        "source_fraction_of_gt_class": float(sub.shape[0]) / float(max(1, n_class_total)),
                        "recall": float(hit.mean()),
                        "n_correct": int(hit.sum()),
                        "precision_note": (
                            "undefined per source class: a prediction carries no source class"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def border_vs_interior(predictions: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and per-class recall of border-touching vs interior nuclei (pooled over cells), cohort and per tissue."""
    rows: List[Dict[str, Any]] = []
    for population in POPULATIONS:
        scored = predictions.loc[_scored_mask(predictions, population)]
        for scope, frame in [("cohort", scored)] + [
            ("tissue", g) for _, g in scored.groupby("slide_id", sort=True)
        ]:
            if frame.empty:
                continue
            n_scope = int(frame.shape[0])
            for border_flag, sub in frame.groupby("touches_border", sort=True):
                y_true = labels_to_ids(sub["gt_label"].astype(str).tolist(), LABEL_TO_INDEX)
                y_pred = labels_to_ids(sub["pred_label"].astype(str).tolist(), LABEL_TO_INDEX)
                cls = classification_metrics_from_ids(y_true, y_pred, label_order=LABEL_ORDER)
                base = {
                    "population": population,
                    "is_headline_population": int(population == HEADLINE_POPULATION),
                    "scope": scope,
                    "slide_id": "COHORT" if scope == "cohort" else str(sub["slide_id"].iloc[0]),
                    "tissue_pannuke": "COHORT" if scope == "cohort" else str(sub["tissue_pannuke"].iloc[0]),
                    "touches_border": int(border_flag),
                    "n_cells": int(sub.shape[0]),
                    "fraction_of_scope": float(sub.shape[0]) / float(max(1, n_scope)),
                }
                rows.append(
                    {
                        **base,
                        "label": "ALL",
                        "accuracy": float(cls.metrics["accuracy"]),
                        "macro_f1": float(cls.metrics["macro_f1"]),
                        "balanced_accuracy": float(cls.metrics["balanced_accuracy"]),
                    }
                )
                for _, row in cls.per_class.iterrows():
                    if str(row["label"]) in set(UNSCORABLE_LABELS):
                        continue
                    rows.append(
                        {
                            **base,
                            "label": str(row["label"]),
                            "support": int(row["support"]),
                            "precision": float(row["precision"]),
                            "recall": float(row["recall"]),
                            "f1": float(row["f1"]),
                        }
                    )
    return pd.DataFrame(rows)


def reference_priors(pools: Mapping[str, ReferencePool]) -> pd.DataFrame:
    """Class counts and fractions of each tissue's reference pool, with the margin over the coverage floor."""
    from pannuke.pannuke_references import MIN_REF_CELLS_PER_GT_CLASS

    rows: List[Dict[str, Any]] = []
    for tissue, pool in sorted(pools.items()):
        total = float(max(1, pool.n_rows))
        for label in COARSE_V2_SCORABLE_CLASSES:
            n = int(pool.label_counts.get(label, 0))
            rows.append(
                {
                    "tissue_HistoST": tissue,
                    "reference_stem": pool.reference_stem,
                    "n_reference_cells": int(pool.n_rows),
                    "label": label,
                    "n_cells": n,
                    "fraction": float(n) / total,
                    "min_ref_cells_per_gt_class": MIN_REF_CELLS_PER_GT_CLASS,
                    "margin_factor": float(n) / float(MIN_REF_CELLS_PER_GT_CLASS),
                    "clears_floor": int(n >= MIN_REF_CELLS_PER_GT_CLASS),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Cross-recipe pairing
# ---------------------------------------------------------------------------------------------

def cross_recipe_comparison(
    predictions_by_recipe: Mapping[str, pd.DataFrame],
    pairing: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare native224 and scale224 on the cells embedded in both: per-tissue metrics, paired test of
    macro_f1 and cell-level agreement of the predicted labels.
    """
    from src.evaluation._shared.paired_stats import apply_fdr, paired_test

    recipes = sorted(predictions_by_recipe)
    if len(recipes) != 2:
        raise ValueError(
            f"The cross-recipe comparison needs exactly two recipes, got {recipes}."
        )
    a, b = recipes
    ids_a = set(predictions_by_recipe[a]["cell_id_int"].to_numpy(dtype=np.int64).tolist())
    ids_b = set(predictions_by_recipe[b]["cell_id_int"].to_numpy(dtype=np.int64).tolist())
    shared = np.array(sorted(ids_a & ids_b), dtype=np.int64)
    declared = int(pairing.get("n_paired_cells", -1))
    if declared >= 0 and int(shared.size) != declared:
        raise RuntimeError(
            f"The intersection computed from the prediction tables ({shared.size:,} cells) "
            f"disagrees with `cross_recipe_pairing` ({declared:,})."
        )

    per_tissue_rows: List[Dict[str, Any]] = []
    values: Dict[str, Dict[str, Dict[str, float]]] = {a: {}, b: {}}
    for recipe in recipes:
        preds = predictions_by_recipe[recipe]
        preds = preds.loc[preds["cell_id_int"].isin(shared)]
        scored = preds.loc[_scored_mask(preds, HEADLINE_POPULATION)]
        for slide_id, sub in scored.groupby("slide_id", sort=True):
            y_true = labels_to_ids(sub["gt_label"].astype(str).tolist(), LABEL_TO_INDEX)
            y_pred = labels_to_ids(sub["pred_label"].astype(str).tolist(), LABEL_TO_INDEX)
            cls = classification_metrics_from_ids(
                y_true, y_pred, label_order=LABEL_ORDER, macro_min_support=MACRO_MIN_CLASS_SUPPORT
            )
            record = {
                "macro_f1": float(cls.metrics["macro_f1"]),
                "balanced_accuracy": float(cls.metrics["balanced_accuracy"]),
                "accuracy": float(cls.metrics["accuracy"]),
            }
            values[recipe][str(slide_id)] = record
            per_tissue_rows.append(
                {
                    "recipe": recipe,
                    "slide_id": str(slide_id),
                    "tissue_pannuke": str(sub["tissue_pannuke"].iloc[0]),
                    "tissue_HistoST": str(sub["tissue_HistoST"].iloc[0]),
                    "population": HEADLINE_POPULATION,
                    "cell_scope": "cross_recipe_intersection",
                    "n_cells": int(cls.metrics["n_cells"]),
                    "n_gt_present_classes": int(cls.metrics["n_gt_present_classes"]),
                    **record,
                }
            )

    slides = sorted(set(values[a]) & set(values[b]))
    agreement = _prediction_agreement(predictions_by_recipe, shared, a, b)
    result = paired_test(
        np.array([values[a][s][PRIMARY_METRIC] for s in slides], dtype=float),
        np.array([values[b][s][PRIMARY_METRIC] for s in slides], dtype=float),
        metric=PRIMARY_METRIC,
        direction="higher",
        keys_a=slides,
        keys_b=slides,
        seed=SEED,
    )
    (corrected,) = apply_fdr([result])
    paired_row = _paired_row(
        corrected,
        family=FAMILY_CROSS_RECIPE,
        recipe=f"{a}_minus_{b}",
        metric=PRIMARY_METRIC,
        population=HEADLINE_POPULATION,
        label=f"{a}_vs_{b}_on_intersection",
    )
    paired_row.update(
        {
            "n_paired_cells": int(shared.size),
            "n_embedded_" + a: int(len(ids_a)),
            "n_embedded_" + b: int(len(ids_b)),
            **agreement,
            "note": (
                "Computed on the intersection of the two embedded cell sets, a different "
                "population from the per-tissue headline."
            ),
        }
    )
    return pd.DataFrame(per_tissue_rows), pd.DataFrame([paired_row])


def _prediction_agreement(
    predictions_by_recipe: Mapping[str, pd.DataFrame],
    shared: np.ndarray,
    a: str,
    b: str,
) -> Dict[str, Any]:
    """Cell-level label agreement between the two recipes on the shared cells."""
    def _series(recipe: str) -> pd.Series:
        preds = predictions_by_recipe[recipe]
        preds = preds.loc[preds["cell_id_int"].isin(shared)]
        scored = preds.loc[_scored_mask(preds, HEADLINE_POPULATION)]
        return scored.set_index("cell_id_int")["pred_label"].astype(str)

    sa, sb = _series(a), _series(b)
    common = sa.index.intersection(sb.index)
    if common.size == 0:
        return {"cell_level_agreement": float("nan"), "n_agreement_cells": 0}
    agree = (sa.loc[common].to_numpy() == sb.loc[common].to_numpy())
    return {
        "cell_level_agreement": float(agree.mean()),
        "n_agreement_cells": int(common.size),
    }


# ---------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------

def build_run_manifest(
    *,
    recipe_name: str,
    training_num: str,
    checkpoint_num: str,
    checkpoint_dir: Path,
    scoring_preflight: ScoringPreflight,
    bundle_facts: Mapping[str, Any],
    pools: Mapping[str, ReferencePool],
    coverage_reports: Sequence[Any],
    population_agreement: Mapping[str, Any],
    per_tissue: pd.DataFrame,
    cohort: pd.DataFrame,
    device: str,
    argv: Sequence[str],
    started_at: str,
    finished_at: str,
    wall_seconds: float,
) -> Dict[str, Any]:
    """Content of run_manifest.json (settings, protocol, inputs, cohort, coverage, headline values)."""
    from pannuke.pannuke_common import git_head
    from pannuke.pannuke_references import protocol_record

    headline = per_tissue.loc[per_tissue["population"] == HEADLINE_POPULATION]
    return {
        "step": 4,
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "geometry_recipe": recipe_name,
        "checkpoint_dir": str(checkpoint_dir),
        "training_num": str(training_num),
        "checkpoint_num": str(checkpoint_num),
        "space": "clip",
        "argv": list(argv),
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": float(wall_seconds),
        "git_head": git_head(),
        "seed": SEED,
        "device": device,
        "environment": pe.environment_record()
        | {"required_conda_env_local": "histost_eval_a2", "required_conda_env_cluster": "histost_eval_a2"},
        "retrieval": {
            "selected_k": SELECTED_K,
            "selected_tau": SELECTED_TAU,
            "Kmax": KMAX,
            "similarity": SIMILARITY,
            "backend_requested": RETRIEVAL_BACKEND,
            "hyperparameters_source": (
                "Taken from the STHELAR run "
                "(eval/A-2_eval_st_predictions/eval_2/HEencoder_t46e13_clip/run_manifest.json: "
                "selected_k=30, selected_tau=0.03); no tuning on PanNuke."
            ),
        },
        "protocol": protocol_record()
        | {
            "macro_min_class_support_per_slide": MACRO_MIN_CLASS_SUPPORT,
            "min_selected_alpha": MIN_SELECTED_ALPHA,
            "populations": list(POPULATIONS),
            "headline_population": HEADLINE_POPULATION,
            "headline_population_note": (
                "The STHELAR headline population `matched_covered_he` requires the cell's GT "
                "scConcept embedding (a transcriptome), which PanNuke does not have. "
                "`he_valid_covered` is the same population without that condition."
            ),
            "primary_metric": PRIMARY_METRIC,
            "secondary_metrics": list(SECONDARY_METRICS),
            "confidence_threshold_grid": list(CONFIDENCE_THRESHOLD_GRID),
            "bh_families": [FAMILY_PRIMARY, FAMILY_SECONDARY, FAMILY_CROSS_RECIPE],
        },
        "st_cell_bundle": dict(bundle_facts),
        "he_encoder": {
            "checkpoint_path": str(
                pe.he_encoder_checkpoint_path(training_num, checkpoint_num)
            ),
            "step3_recorded": (scoring_preflight.extract_manifest.get("encoder") or {}),
            "note": (
                "`step3_recorded.checkpoint_path` is provenance only; the paths here are resolved "
                "with get_full_path."
            ),
        },
        "upstream": {
            "prepare_manifest": {
                "root_resolved_locally": str(scoring_preflight.prepared_root),
                "prepare_schema_version": scoring_preflight.preflight.manifest.get(
                    "prepare_schema_version"
                ),
                "is_full_run": scoring_preflight.preflight.manifest.get("is_full_run"),
                "git_head": scoring_preflight.preflight.manifest.get("git_head"),
                "membership_group": scoring_preflight.membership_group,
            },
            "extract_manifest": {
                "root_recorded": (scoring_preflight.extract_manifest.get("prepared_tree") or {}).get("root"),
                "root_recorded_tail": _relative_tail(
                    str((scoring_preflight.extract_manifest.get("prepared_tree") or {}).get("root", ""))
                ),
                "extract_schema_version": scoring_preflight.extract_manifest.get(
                    "extract_schema_version"
                ),
                "git_head": scoring_preflight.extract_manifest.get("git_head"),
                "cohort": scoring_preflight.extract_manifest.get("cohort"),
                "note": (
                    "Recorded roots are provenance only; only their root-independent tails are "
                    "compared."
                ),
            },
        },
        "cohort": {
            "n_tissues": int(headline["slide_id"].nunique()),
            "slides": sorted(headline["slide_id"].astype(str).unique().tolist()),
            "n_scored_cells": int(headline["n_cells"].sum()),
            "restricted_to_slides": scoring_preflight.preflight.restricted_to_slides,
            "is_partial": bool(scoring_preflight.preflight.is_partial),
        },
        "reference_pools": {t: p.audit for t, p in sorted(pools.items())},
        "coverage": {
            "contract": protocol_record()["coverage"],
            "any_violation": any(not r.ok for r in coverage_reports),
            "reports": [r.as_dict() for r in coverage_reports],
        },
        "population_agreement": dict(population_agreement),
        "realized_macro_denominators": [
            {
                "slide_id": str(row["slide_id"]),
                "population": str(row["population"]),
                "n_cells": int(row["n_cells"]),
                "n_gt_present_classes": int(row["n_gt_present_classes"]),
                "n_gt_present_classes_unfloored": int(row["n_gt_present_classes_unfloored"]),
                "n_classes_below_macro_floor": int(row["n_classes_below_macro_floor"]),
            }
            for _, row in per_tissue.iterrows()
        ],
        "headline": {
            row["metric"]: {"he": float(row["he_value"]), "floor": float(row["floor_value"])}
            for _, row in cohort.loc[
                (cohort["population"] == HEADLINE_POPULATION)
                & (cohort["summary_scope"] == "unweighted_tissue_mean")
            ].iterrows()
        },
        "reading_rules": [
            "Accuracy and macro-F1 are not interchangeable: the classes are imbalanced, so the "
            "majority-class floor is much higher in accuracy than in macro-F1. The headline "
            "statistic is macro-F1, as in A.2 on STHELAR.",
            "The unweighted tissue mean (headline, as in A.2) and the pooled-over-cells value are "
            "different statistics; the pooled value is weighted towards the largest tissues.",
            "The two geometry recipes are two builds of the same cohort; they are paired only on "
            "the intersection of their embedded cell sets.",
            "Per-tissue macro denominators differ only if a class falls below the 50-cell floor; "
            "`n_gt_present_classes` gives the realized denominator of each tissue.",
        ],
    }


# ---------------------------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------------------------

REQUIRED_OUTPUTS: Tuple[str, ...] = (
    "metrics/per_tissue.csv",
    "metrics/per_class.csv",
    "metrics/chance_floor.csv",
    "metrics/cohort_summary.csv",
    "metrics/paired_tests.csv",
    "metrics/confidence_sweep.csv",
    "diagnostics/reference_priors.csv",
    "diagnostics/epithelial_by_source_class.csv",
    "diagnostics/border_vs_interior.csv",
    "diagnostics/coverage_report.json",
    "run_manifest.json",
)


def verify_outputs(
    checkpoint_dir: Path,
    scoring_preflight: ScoringPreflight,
    log: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Check the outputs of one recipe on disk (files, slides, floor vs headline, k/tau, coverage). Writes nothing."""
    log = log or logger
    checkpoint_dir = Path(checkpoint_dir)
    missing = [rel for rel in REQUIRED_OUTPUTS if not (checkpoint_dir / rel).exists()]
    if missing:
        raise FileNotFoundError(
            f"{checkpoint_dir} is missing {len(missing)} required output(s): {missing}. Run "
            "--stages score first."
        )

    per_tissue = pd.read_csv(checkpoint_dir / "metrics" / "per_tissue.csv")
    chance_floor = pd.read_csv(checkpoint_dir / "metrics" / "chance_floor.csv")
    manifest = read_json(run_manifest_path(checkpoint_dir))

    expected_slides = set(scoring_preflight.slide_ids)
    got_slides = set(per_tissue["slide_id"].astype(str))
    if got_slides != expected_slides:
        raise RuntimeError(
            f"per_tissue.csv covers {sorted(got_slides)}, the tree holds {sorted(expected_slides)}."
        )

    merged = per_tissue.merge(
        chance_floor, on=["slide_id", "population"], suffixes=("_he", "_floor")
    )
    if merged.shape[0] != per_tissue.shape[0]:
        raise RuntimeError("per_tissue.csv and chance_floor.csv do not align one-to-one.")
    bad_cells = merged.loc[merged["n_cells_he"] != merged["n_cells_floor"]]
    if not bad_cells.empty:
        raise RuntimeError(
            "The chance floor and the headline sit on different cell counts for "
            f"{bad_cells[['slide_id', 'population']].to_dict('records')}."
        )
    bad_axis = merged.loc[
        merged["n_gt_present_classes_he"] != merged["n_gt_present_classes_floor"]
    ]
    if not bad_axis.empty:
        raise RuntimeError(
            "The chance floor and the headline average over different class axes for "
            f"{bad_axis[['slide_id', 'population']].to_dict('records')}."
        )

    for slide_id in sorted(expected_slides):
        for path in (
            predictions_path(checkpoint_dir, slide_id),
            retrieval_path(checkpoint_dir, slide_id),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing per-slide artifact: {path}")
        for population in POPULATIONS:
            cm_path = confusion_path(checkpoint_dir, slide_id, population)
            if not cm_path.exists():
                raise FileNotFoundError(f"Missing confusion matrix: {cm_path}")
            cm = np.load(cm_path)
            if cm.shape != (len(LABEL_ORDER), len(LABEL_ORDER)):
                raise RuntimeError(
                    f"{cm_path} has shape {cm.shape}, expected "
                    f"{(len(LABEL_ORDER), len(LABEL_ORDER))}."
                )

    if manifest.get("retrieval", {}).get("selected_k") != SELECTED_K:
        raise RuntimeError(
            f"{run_manifest_path(checkpoint_dir)} records selected_k="
            f"{manifest.get('retrieval', {}).get('selected_k')}, this code uses {SELECTED_K}."
        )
    if float(manifest.get("retrieval", {}).get("selected_tau", -1)) != float(SELECTED_TAU):
        raise RuntimeError(
            f"{run_manifest_path(checkpoint_dir)} records a different selected_tau than "
            f"{SELECTED_TAU}."
        )
    if manifest.get("coverage", {}).get("any_violation"):
        raise RuntimeError(
            "The recorded coverage report carries a violation (see diagnostics/coverage_report.json)."
        )

    record = {
        "checkpoint_dir": str(checkpoint_dir),
        "n_tissues": len(expected_slides),
        "outputs_present": list(REQUIRED_OUTPUTS),
        "floor_shares_class_axis": True,
        "all_passed": True,
    }
    log.info("Verified %d tissue(s) of outputs under %s.", len(expected_slides), checkpoint_dir)
    return record


# ---------------------------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------------------------

def write_slide_outputs(checkpoint_dir: Path, scores: SlideScores, save_retrieval: bool = True) -> None:
    """Persist one pseudo-slide's predictions, top-k neighbours and confusion matrices."""
    checkpoint_dir = Path(checkpoint_dir)
    write_dataframe(scores.predictions, predictions_path(checkpoint_dir, scores.slide_id))
    for population, cm in scores.confusions.items():
        path = confusion_path(checkpoint_dir, scores.slide_id, population)
        ensure_dir(path.parent)
        np.save(path, cm)
    if save_retrieval and scores.nn_idx is not None:
        path = retrieval_path(checkpoint_dir, scores.slide_id)
        ensure_dir(path.parent)
        # valid_rows: row indices of the queried cells in predictions/<slide>.parquet
        np.savez_compressed(
            path,
            valid_rows=np.asarray(scores.valid_rows, dtype=np.int64),
            cell_id_int=scores.predictions["cell_id_int"].to_numpy(dtype=np.int64)[scores.valid_rows],
            neighbor_reference_row=np.asarray(scores.nn_idx, dtype=np.int32),
            neighbor_cosine=np.asarray(scores.nn_sim, dtype=np.float32),
            k=np.int32(SELECTED_K),
            tau=np.float32(SELECTED_TAU),
        )


def write_coverage_report(checkpoint_dir: Path, reports: Sequence[Any]) -> None:
    """Persist the full per-tissue coverage accounting, contract verdict included."""
    from pannuke.pannuke_references import coverage_cfg

    write_json(
        Path(checkpoint_dir) / "diagnostics" / "coverage_report.json",
        {
            "stage": "step4_score",
            "contract": coverage_cfg()["references"]["coverage"],
            "any_violation": any(not r.ok for r in reports),
            "reports": [r.as_dict() for r in reports],
        },
    )
