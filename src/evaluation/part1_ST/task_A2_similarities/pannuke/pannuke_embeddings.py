#!/usr/bin/env python
"""Preflight checks, verification and manifest of the PanNuke H&E embedding extraction (_5_compute_HEembs.py)."""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Project imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pannuke.pannuke_common import (
    PANNUKE_TISSUE_TO_HISTOST,
    PanNukeVocabularyError,
    pseudo_slide_id_for_tissue,
    tissue_for_pseudo_slide_id,
)

EXTRACT_SCHEMA_VERSION: int = 1

# Supported prepare_schema_version values of prepare_manifest.json
SUPPORTED_PREPARE_SCHEMA_VERSIONS: Tuple[int, ...] = (1,)

# Folder names read by src_A2/embeddings.py (resolve_legacy_he_embedding_dir, ensure_selected_main_rows)
HE_EMBEDDINGS_DIRNAME: str = "he_embeddings"
MAIN_ROWS_DIRNAME: str = "main_HEemb_cellID"

EXTRACT_MANIFEST_NAME: str = "extract_manifest.json"
EXTRACT_LOG_NAME: str = "extract_log.txt"

# Expected UNI2-h geometry
EXPECTED_PATCH_HE_DIM: int = 1536
EXPECTED_TOKENS_PER_PATCH: int = 256
EXPECTED_TOKENS_PER_DIM: int = 16
EXPECTED_MODEL_INPUT_PX: int = 224

# h5 dataset and parquet columns read by src_A2/embeddings.py (load_selected_he_embeddings)
STEP4_REQUIRED_H5_DATASET: str = "patch_cell_emb_he_clip"
STEP4_REQUIRED_PARQUET_COLS: Tuple[str, ...] = (
    "cell_id_int",
    "selected_patch_cell_row_index",
    "selected_alpha",
)

# Output schema of `_9_main_HEemb_cellID.py`
MAIN_ROW_SCHEMA: Tuple[Tuple[str, str], ...] = (
    ("cell_id_int", "int32"),
    ("cell_id", "string"),
    ("selected_patch_cell_row_index", "int64"),
    ("selected_patch_index", "int32"),
    ("selected_patch_id", "string"),
    ("selected_alpha", "float"),
    ("selected_center_dist2", "double"),
    ("n_candidate_patches", "int32"),
)

# Datasets written by `_8` for a slide with cells
EXPECTED_H5_DATASETS: Tuple[str, ...] = (
    "patch_id",
    "patch_emb_he",
    "patch_emb_he_clip",
    "patch_cell_patch_index",
    "patch_cell_cell_id_int",
    "patch_cell_alpha",
    "patch_cell_emb_he",
    "patch_cell_emb_he_clip",
)

# Rows of patch_cell_emb_he checked for non-finite values (all rows with --full_scan)
EMB_SAMPLE_ROWS: int = 10_000

# Attributes written by `_5` into the h5 after a successful extraction (checked on resume)
COMPLETION_ATTRS: Tuple[str, ...] = (
    "pannuke_extract_complete",
    "pannuke_extract_schema_version",
    "pannuke_geometry_recipe",
    "pannuke_n_expected_groups",
)


# ---------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------

def checkpoint_tag(training_num: str, checkpoint_num: str) -> str:
    """The `HEembs_t{n}e{c}` folder token, as `src_A2/embeddings.py` builds it."""
    return f"HEembs_t{training_num}e{checkpoint_num}"


def he_embeddings_dir(prepared_root_path: Path, training_num: str, checkpoint_num: str) -> Path:
    """Output folder of the H&E embeddings: <prepared_root>/he_embeddings/HEembs_t<n>e<c>."""
    return (
        Path(prepared_root_path)
        / HE_EMBEDDINGS_DIRNAME
        / checkpoint_tag(training_num, checkpoint_num)
    )


def main_rows_dir(emb_dir: Path) -> Path:
    """Folder of the one-row-per-cell parquets written by `_9`."""
    return Path(emb_dir) / MAIN_ROWS_DIRNAME


def embedding_h5_path(emb_dir: Path, slide_id: str) -> Path:
    return Path(emb_dir) / f"{slide_id}.h5"


def main_rows_parquet_path(emb_dir: Path, slide_id: str) -> Path:
    return main_rows_dir(emb_dir) / f"{slide_id}.parquet"


def he_encoder_checkpoint_path(training_num: str, checkpoint_num: str) -> Path:
    """Exported H&E encoder bundle of a checkpoint."""
    from src_A2.common import src_import

    return src_import("get_full_path")(
        "train",
        "trainings",
        "trainings_HistoST",
        f"training_{training_num}",
        "trained_encoders",
        f"HEencoder_t{training_num}e{checkpoint_num}.pt",
    )


def metadata_csv_path(prepared_root_path: Path, recipe_name: str) -> Path:
    """Metadata CSV of a prepared tree (metadata_eval_st_pannuke_<recipe>.csv)."""
    return Path(prepared_root_path) / f"metadata_eval_st_pannuke_{recipe_name}.csv"


def patches_h5_path(prepared_root_path: Path, slide_id: str) -> Path:
    return Path(prepared_root_path) / "patches" / f"{slide_id}.h5"


def ground_truth_path(prepared_root_path: Path, slide_id: str) -> Path:
    return Path(prepared_root_path) / "ground_truth" / f"{slide_id}.parquet"


def centroids_path(prepared_root_path: Path, slide_id: str) -> Path:
    return Path(prepared_root_path) / "nucleus_centroids" / f"{slide_id}.parquet"


def missing_cells_path(prepared_root_path: Path, membership_group: str, slide_id: str) -> Path:
    return (
        Path(prepared_root_path)
        / f"cell_token_membership_missing_cells_{membership_group}"
        / f"{slide_id}.parquet"
    )


# ---------------------------------------------------------------------------------------------
# Expectations
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SlideExpectation:
    """Paths and expected counts of one pseudo-slide, before extraction."""

    slide_id: str
    patches_h5: Path
    membership_parquet: Path
    missing_parquet: Path
    ground_truth_parquet: Path
    centroids_parquet: Path
    n_patches: int
    stored_px: int
    n_ground_truth_cells: int          # ground_truth rows
    n_missing_cells: int               # missing-cell rows
    n_membership_rows: int
    n_expected_groups: int             # distinct cell_id_int of the membership parquet
    max_token_index: int


@dataclass(frozen=True)
class RecipePreflight:
    """One geometry recipe's tree, validated and measured."""

    recipe_name: str
    prepared_root: Path
    membership_group: str
    manifest: Dict[str, Any] = field(repr=False)
    slides: Tuple[SlideExpectation, ...] = field(repr=False)
    restricted_to_slides: Optional[Tuple[str, ...]] = None

    @property
    def slide_ids(self) -> Tuple[str, ...]:
        return tuple(s.slide_id for s in self.slides)

    @property
    def is_partial(self) -> bool:
        """True for a --slides run (subset of the pseudo-slides)."""
        return self.restricted_to_slides is not None

    def total(self, attr: str) -> int:
        return int(sum(getattr(s, attr) for s in self.slides))


def read_prepare_manifest(prepared_root_path: Path) -> Dict[str, Any]:
    """Load prepare_manifest.json and check its schema version."""
    path = Path(prepared_root_path) / "prepare_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No prepare_manifest.json under {prepared_root_path}. Run _4_prepare_patches.py first."
        )
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    version = manifest.get("prepare_schema_version")
    if version not in SUPPORTED_PREPARE_SCHEMA_VERSIONS:
        raise ValueError(
            f"{path} declares prepare_schema_version={version!r}; supported: "
            f"{list(SUPPORTED_PREPARE_SCHEMA_VERSIONS)}."
        )
    return manifest


def slide_ids_from_metadata(prepared_root_path: Path, recipe_name: str) -> List[str]:
    """Slide ids of the metadata CSV, checked to equal the stems of the patch h5 files (as in `_8.main()`)."""
    csv_path = metadata_csv_path(prepared_root_path, recipe_name)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "slide_id" not in df.columns:
        raise ValueError(f"'slide_id' column not found in {csv_path}")
    csv_ids = set(df["slide_id"].astype(str))

    patches_dir = Path(prepared_root_path) / "patches"
    if not patches_dir.is_dir():
        raise FileNotFoundError(f"Patches directory not found: {patches_dir}")
    glob_ids = {p.stem for p in sorted(patches_dir.glob("*.h5")) if not p.name.startswith(".")}
    if not glob_ids:
        raise FileNotFoundError(f"No patch .h5 files found in {patches_dir}")

    if csv_ids != glob_ids:
        only_csv = sorted(csv_ids - glob_ids)
        only_dir = sorted(glob_ids - csv_ids)
        raise ValueError(
            f"metadata CSV and patches/ disagree for recipe '{recipe_name}':\n"
            f"  in the CSV but with no patch h5: {only_csv}\n"
            f"  in patches/ but not in the CSV : {only_dir}"
        )
    return sorted(glob_ids)


def _parquet_num_rows(path: Path) -> int:
    """Row count from parquet footer metadata -- no row group is read."""
    return int(pq.ParquetFile(path).metadata.num_rows)


def _parquet_columns(path: Path) -> List[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def _require_columns(path: Path, required: Sequence[str]) -> None:
    have = set(_parquet_columns(path))
    missing = [c for c in required if c not in have]
    if missing:
        raise KeyError(f"{path} is missing column(s) {missing}; found {sorted(have)}.")


def _verdicts_by_slide(manifest: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """`admission.verdicts` is a list of per-slide dicts; key it by pseudo-slide id."""
    verdicts = (manifest.get("admission") or {}).get("verdicts") or []
    out: Dict[str, Dict[str, Any]] = {}
    for entry in verdicts:
        sid = entry.get("pseudo_slide_id")
        if sid:
            out[str(sid)] = dict(entry)
    return out


def expected_counts_for_slide(
    prepared_root_path: Path,
    membership_group: str,
    slide_id: str,
    stored_px: int,
    verdict: Optional[Mapping[str, Any]] = None,
) -> SlideExpectation:
    """
    Expected counts of one slide. Checks that distinct membership cells == ground_truth rows - missing
    rows == n_token_eligible_cells of the prepare_manifest verdict.
    """
    patches_h5 = patches_h5_path(prepared_root_path, slide_id)
    mem_path = (
        Path(prepared_root_path) / f"cell_token_membership_{membership_group}" / f"{slide_id}.parquet"
    )
    missing_path = missing_cells_path(prepared_root_path, membership_group, slide_id)
    gt_path = ground_truth_path(prepared_root_path, slide_id)
    cen_path = centroids_path(prepared_root_path, slide_id)

    # --- the patches h5 -----------------------------------------------------------------------
    if not patches_h5.exists():
        raise FileNotFoundError(f"{slide_id}: patches h5 not found at {patches_h5}.")
    with h5py.File(patches_h5, "r") as f:
        for key in ("img", "coords", "barcode"):
            if key not in f:
                raise KeyError(f"{patches_h5} has no '{key}' dataset (found {sorted(f.keys())}).")
        n_patches = int(f["img"].shape[0])
        img_shape = tuple(int(v) for v in f["img"].shape)
        if int(f["coords"].shape[0]) != n_patches or int(f["barcode"].shape[0]) != n_patches:
            raise ValueError(
                f"{slide_id}: img/coords/barcode disagree on length "
                f"({n_patches}/{f['coords'].shape[0]}/{f['barcode'].shape[0]})."
            )
    if n_patches == 0:
        raise ValueError(
            f"{slide_id}: the patches h5 holds zero patches."
        )
    if len(img_shape) != 4 or img_shape[1] != stored_px or img_shape[2] != stored_px:
        raise ValueError(
            f"{slide_id}: img has shape {img_shape}, expected (N, {stored_px}, {stored_px}, 3) for "
            f"this geometry recipe's stored_px."
        )

    # --- the membership parquet ---------------------------------------------------------------
    if not mem_path.exists():
        raise FileNotFoundError(
            f"Missing cell-token-membership parquet for '{slide_id}': {mem_path}"
        )
    _require_columns(mem_path, ("cell_id_int", "patch_id", "token_index", "nuc_px"))
    n_membership_rows = _parquet_num_rows(mem_path)
    if n_membership_rows == 0:
        raise ValueError(
            f"{slide_id}: membership parquet {mem_path} has 0 rows."
        )
    mem = pq.read_table(mem_path, columns=["cell_id_int", "token_index"])
    cell_ids = mem.column("cell_id_int").to_numpy(zero_copy_only=False)
    token_index = mem.column("token_index").to_numpy(zero_copy_only=False)
    n_expected_groups = int(np.unique(cell_ids).size)
    max_token_index = int(token_index.max())
    min_token_index = int(token_index.min())
    if min_token_index < 0 or max_token_index >= EXPECTED_TOKENS_PER_PATCH:
        raise ValueError(
            f"{slide_id}: membership token_index spans [{min_token_index}, {max_token_index}], "
            f"outside [0, {EXPECTED_TOKENS_PER_PATCH - 1}]."
        )

    # --- ground truth and the missing set ------------------------------------------------------
    for path, cols in (
        (missing_path, ("cell_id_int",)),
        (gt_path, ("cell_id_int", "nuc_px_membership")),
        (cen_path, ("cell_id", "x", "y")),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{slide_id}: expected {path} to exist.")
        _require_columns(path, cols)
    n_ground_truth = _parquet_num_rows(gt_path)
    n_missing = _parquet_num_rows(missing_path)
    n_centroids = _parquet_num_rows(cen_path)
    if n_centroids != n_ground_truth:
        raise ValueError(
            f"{slide_id}: nucleus_centroids has {n_centroids} rows against ground_truth's "
            f"{n_ground_truth}."
        )

    # --- count identity -----------------------------------------------------------------------
    d_value = None
    if verdict is not None and "n_token_eligible_cells" in verdict:
        d_value = int(verdict["n_token_eligible_cells"])
    if n_expected_groups != n_ground_truth - n_missing or (
        d_value is not None and d_value != n_expected_groups
    ):
        raise RuntimeError(
            f"{slide_id}: the token-eligible cell count disagrees across independent sources -- "
            f"membership parquet {n_expected_groups}, ground_truth minus missing = "
            f"{n_ground_truth} - {n_missing} = {n_ground_truth - n_missing}, "
            f"prepare_manifest verdict {d_value}."
        )

    return SlideExpectation(
        slide_id=slide_id,
        patches_h5=patches_h5,
        membership_parquet=mem_path,
        missing_parquet=missing_path,
        ground_truth_parquet=gt_path,
        centroids_parquet=cen_path,
        n_patches=n_patches,
        stored_px=stored_px,
        n_ground_truth_cells=n_ground_truth,
        n_missing_cells=n_missing,
        n_membership_rows=n_membership_rows,
        n_expected_groups=n_expected_groups,
        max_token_index=max_token_index,
    )


def preflight_recipe(
    prepared_root_path: Path,
    recipe_name: str,
    membership_group: str,
    slides: Optional[Sequence[str]] = None,
    allow_partial_tree: bool = False,
) -> RecipePreflight:
    """Check one prepared tree before any GPU work (raises on the first problem). Returns its RecipePreflight."""
    prepared_root_path = Path(prepared_root_path)
    if not prepared_root_path.is_dir():
        raise FileNotFoundError(
            f"Prepared PanNuke tree not found for recipe '{recipe_name}': {prepared_root_path}. "
            f"Run _4_prepare_patches.py --geometry_recipe {recipe_name} first."
        )

    manifest = read_prepare_manifest(prepared_root_path)

    got_recipe = (manifest.get("geometry_recipe") or {}).get("name")
    if got_recipe != recipe_name:
        raise ValueError(
            f"{prepared_root_path}/prepare_manifest.json describes recipe {got_recipe!r}, not "
            f"{recipe_name!r}. --data_dir is pointing at the wrong tree."
        )

    if not manifest.get("is_full_run", False) and not allow_partial_tree:
        raise RuntimeError(
            f"'{recipe_name}' was built with is_full_run=False (--tissues / --max_tiles subset). "
            "Rebuild it without those flags, or pass --allow_partial_tree."
        )

    declared_group = (manifest.get("membership") or {}).get("group_name")
    if declared_group != membership_group:
        raise ValueError(
            f"'{recipe_name}' was prepared for membership group {declared_group!r}, but the "
            f"encoder resolves to {membership_group!r}."
        )
    tokens_per_patch = int((manifest.get("membership") or {}).get("tokens_per_patch", 0))
    tokens_per_dim = int((manifest.get("membership") or {}).get("tokens_per_dim", 0))
    if (tokens_per_patch, tokens_per_dim) != (EXPECTED_TOKENS_PER_PATCH, EXPECTED_TOKENS_PER_DIM):
        raise ValueError(
            f"'{recipe_name}' declares a {tokens_per_dim}x{tokens_per_dim} token grid "
            f"({tokens_per_patch} tokens/patch); expected "
            f"{EXPECTED_TOKENS_PER_DIM}x{EXPECTED_TOKENS_PER_DIM} "
            f"({EXPECTED_TOKENS_PER_PATCH})."
        )

    membership_dirs = sorted(
        p.name
        for p in prepared_root_path.glob("cell_token_membership_*")
        if p.is_dir() and not p.name.startswith("cell_token_membership_missing_cells_")
    )
    if membership_dirs != [f"cell_token_membership_{membership_group}"]:
        raise ValueError(
            f"'{recipe_name}' holds membership folders {membership_dirs}, expected exactly "
            f"['cell_token_membership_{membership_group}']."
        )

    stored_px = int((manifest.get("geometry_recipe") or {}).get("stored_px", 0))
    if stored_px <= 0:
        raise ValueError(f"'{recipe_name}' declares a non-positive stored_px={stored_px}.")

    all_slide_ids = slide_ids_from_metadata(prepared_root_path, recipe_name)
    for sid in all_slide_ids:
        try:
            tissue_for_pseudo_slide_id(sid)
        except PanNukeVocabularyError as exc:
            raise ValueError(
                f"'{sid}' is not a PanNuke pseudo-slide id ({exc}); check --data_dir."
            ) from exc

    restricted: Optional[Tuple[str, ...]] = None
    if slides:
        requested = [str(s) for s in slides]
        unknown = [s for s in requested if s not in set(all_slide_ids)]
        if unknown:
            raise ValueError(
                f"--slides names {unknown}, which are not in '{recipe_name}'. Available: "
                f"{all_slide_ids}"
            )
        selected_ids = [s for s in all_slide_ids if s in set(requested)]
        restricted = tuple(selected_ids)
    else:
        selected_ids = all_slide_ids
        expected_n = len(PANNUKE_TISSUE_TO_HISTOST)
        if manifest.get("is_full_run", False) and len(selected_ids) != expected_n:
            raise ValueError(
                f"'{recipe_name}' is marked is_full_run but holds {len(selected_ids)} "
                f"pseudo-slides, not the {expected_n} retained tissues "
                f"({sorted(pseudo_slide_id_for_tissue(t) for t in PANNUKE_TISSUE_TO_HISTOST)})."
            )

    verdicts = _verdicts_by_slide(manifest)
    expectations = tuple(
        expected_counts_for_slide(
            prepared_root_path,
            membership_group,
            sid,
            stored_px=stored_px,
            verdict=verdicts.get(sid),
        )
        for sid in selected_ids
    )

    return RecipePreflight(
        recipe_name=recipe_name,
        prepared_root=prepared_root_path,
        membership_group=membership_group,
        manifest=manifest,
        slides=expectations,
        restricted_to_slides=restricted,
    )


def assert_encoder_matches_tree(
    encoder_facts: Mapping[str, Any],
    preflights: Sequence[RecipePreflight],
) -> None:
    """Raise if the encoder cannot produce the cell CLIP embeddings (token outputs, token CLIP head, 224 px input)."""
    tag = encoder_facts.get("checkpoint_tag", "<unknown>")

    if not encoder_facts.get("has_encode_with_tokens", False):
        raise RuntimeError(
            f"{tag} exports {encoder_facts.get('class_name')!r}, whose backbone has no token "
            "outputs (needed to pool the nuclei)."
        )

    if not encoder_facts.get("has_token_proj", False):
        raise RuntimeError(
            f"Encoder {tag} has no img_tok_proj (the token CLIP head); use a checkpoint trained with "
            "lambda_token > 0."
        )

    resize = int(encoder_facts.get("resize_size", 0))
    crop = int(encoder_facts.get("crop_size", 0))
    if resize != EXPECTED_MODEL_INPUT_PX or crop != EXPECTED_MODEL_INPUT_PX:
        raise RuntimeError(
            f"Encoder {tag} declares resize={resize}, crop={crop}; the membership token grid is "
            f"{EXPECTED_TOKENS_PER_DIM}x{EXPECTED_TOKENS_PER_DIM} at {EXPECTED_MODEL_INPUT_PX} px."
        )

    for pre in preflights:
        stored_px = int((pre.manifest.get("geometry_recipe") or {}).get("stored_px", 0))
        defer_resize = bool((pre.manifest.get("geometry_recipe") or {}).get("defer_image_resize"))
        if not defer_resize and stored_px != EXPECTED_MODEL_INPUT_PX:
            raise RuntimeError(
                f"'{pre.recipe_name}' pre-bakes its canvas (defer_image_resize=False) at "
                f"stored_px={stored_px}; expected {EXPECTED_MODEL_INPUT_PX}."
            )


def assert_ground_truth_identical_across_recipes(
    preflights: Sequence[RecipePreflight],
) -> None:
    """Raise unless the recipes have the same pseudo-slides, ground-truth cell counts and patch counts."""
    if len(preflights) < 2:
        return
    base = preflights[0]
    base_by_slide = {s.slide_id: s for s in base.slides}
    for other in preflights[1:]:
        for exp in other.slides:
            ref = base_by_slide.get(exp.slide_id)
            if ref is None:
                raise RuntimeError(
                    f"'{exp.slide_id}' is present in '{other.recipe_name}' but not in "
                    f"'{base.recipe_name}'. The recipes must cover the same pseudo-slides."
                )
            if ref.n_ground_truth_cells != exp.n_ground_truth_cells:
                raise RuntimeError(
                    f"{exp.slide_id}: ground-truth cell count differs between "
                    f"'{base.recipe_name}' ({ref.n_ground_truth_cells}) and "
                    f"'{other.recipe_name}' ({exp.n_ground_truth_cells})."
                )
            if ref.n_patches != exp.n_patches:
                raise RuntimeError(
                    f"{exp.slide_id}: patch count differs between '{base.recipe_name}' "
                    f"({ref.n_patches}) and '{other.recipe_name}' ({exp.n_patches})."
                )


# ---------------------------------------------------------------------------------------------
# Completion marker (resume safety)
# ---------------------------------------------------------------------------------------------

def mark_extract_complete(
    h5_path: Path,
    recipe_name: str,
    expectation: SlideExpectation,
    finished_at: str,
) -> None:
    """Write the completion attributes into an h5 fully written by `_8`."""
    with h5py.File(h5_path, "a") as f:
        f.attrs["pannuke_extract_complete"] = True
        f.attrs["pannuke_extract_schema_version"] = EXTRACT_SCHEMA_VERSION
        f.attrs["pannuke_geometry_recipe"] = recipe_name
        f.attrs["pannuke_n_expected_groups"] = int(expectation.n_expected_groups)
        f.attrs["pannuke_extract_finished_at"] = finished_at


def is_extract_complete(
    h5_path: Path,
    recipe_name: str,
    expectation: SlideExpectation,
) -> bool:
    """True for an h5 completed by this script, for this recipe, with the expected row count."""
    if not Path(h5_path).exists():
        return False
    try:
        with h5py.File(h5_path, "r") as f:
            if not bool(f.attrs.get("pannuke_extract_complete", False)):
                return False
            if int(f.attrs.get("pannuke_extract_schema_version", -1)) != EXTRACT_SCHEMA_VERSION:
                return False
            if str(f.attrs.get("pannuke_geometry_recipe", "")) != recipe_name:
                return False
            if int(f.attrs.get("pannuke_n_expected_groups", -1)) != expectation.n_expected_groups:
                return False
            if "patch_cell_cell_id_int" not in f:
                return False
            return int(f["patch_cell_cell_id_int"].shape[0]) == expectation.n_expected_groups
    except OSError:
        # Unreadable (e.g. truncated) file
        return False


# ---------------------------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------------------------

def _decode_str_array(values: np.ndarray) -> np.ndarray:
    """h5py returns vlen-utf8 as bytes; normalise to a str array."""
    flat = np.asarray(values).reshape(-1)
    return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in flat])


def _nonfinite_count(arr: np.ndarray) -> int:
    return int((~np.isfinite(arr)).sum())


def verify_embedding_h5(
    h5_path: Path,
    expectation: SlideExpectation,
    recipe_name: str,
    full_scan: bool = False,
) -> Dict[str, Any]:
    """
    Check the embedding h5 of one slide (datasets, shapes, cell ids, row order, finite values, alpha).
    Returns the measured values.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise RuntimeError(
            f"{recipe_name}/{expectation.slide_id}: extract_embeddings_for_slide returned without "
            f"creating {h5_path}."
        )

    record: Dict[str, Any] = {"slide_id": expectation.slide_id}
    with h5py.File(h5_path, "r") as f:
        missing = [k for k in EXPECTED_H5_DATASETS if k not in f]
        if missing:
            raise KeyError(
                f"{recipe_name}/{expectation.slide_id}: the embedding h5 is missing "
                f"{missing}; found {sorted(f.keys())}."
            )

        # Attributes of a cell-level extraction
        if not bool(f.attrs.get("has_patch_cells", False)):
            raise RuntimeError(
                f"{recipe_name}/{expectation.slide_id}: the h5 records has_patch_cells=False "
                "(patch-only output: `_8` did not find the membership parquet)."
            )
        # Compare only the folder and file names of the recorded membership path (the root may differ)
        recorded_membership = str(f.attrs.get("membership_parquet", ""))
        expected_membership = Path(expectation.membership_parquet)
        recorded_path = Path(recorded_membership) if recorded_membership else None
        if (
            recorded_path is None
            or recorded_path.name != expected_membership.name
            or recorded_path.parent.name != expected_membership.parent.name
        ):
            raise RuntimeError(
                f"{recipe_name}/{expectation.slide_id}: the h5 records "
                f"membership_parquet={recorded_membership!r}, whose membership folder + filename "
                f"({recorded_path.parent.name + '/' + recorded_path.name if recorded_path else '<empty>'}) "
                f"do not match this tree's "
                f"({expected_membership.parent.name}/{expected_membership.name})."
            )

        # Patch level.
        patch_ids = _decode_str_array(f["patch_id"][:])
        if patch_ids.size != expectation.n_patches:
            raise RuntimeError(
                f"{recipe_name}/{expectation.slide_id}: {patch_ids.size} patch ids written, "
                f"expected {expectation.n_patches}."
            )
        with h5py.File(expectation.patches_h5, "r") as fp:
            barcodes = _decode_str_array(fp["barcode"][:])
        if not np.array_equal(patch_ids, barcodes):
            raise RuntimeError(
                f"{recipe_name}/{expectation.slide_id}: the embedding h5's patch_id order does "
                "not match the barcode order of patches/{slide}.h5."
            )

        patch_he = f["patch_emb_he"]
        if tuple(patch_he.shape) != (expectation.n_patches, EXPECTED_PATCH_HE_DIM):
            raise RuntimeError(
                f"{recipe_name}/{expectation.slide_id}: patch_emb_he has shape "
                f"{tuple(patch_he.shape)}, expected "
                f"({expectation.n_patches}, {EXPECTED_PATCH_HE_DIM})."
            )
        patch_clip = f["patch_emb_he_clip"]
        if patch_clip.ndim != 2 or patch_clip.shape[0] != expectation.n_patches:
            raise RuntimeError(
                f"{recipe_name}/{expectation.slide_id}: patch_emb_he_clip has shape "
                f"{tuple(patch_clip.shape)}, expected ({expectation.n_patches}, D)."
            )

        # Patch-cell level (arrays read by A-2)
        n_groups = int(f["patch_cell_cell_id_int"].shape[0])
        if n_groups != expectation.n_expected_groups:
            raise RuntimeError(
                f"{recipe_name}/{expectation.slide_id}: {n_groups:,} patch-cell groups written, "
                f"expected {expectation.n_expected_groups:,} (= distinct cell_id_int in the "
                "membership parquet = ground_truth minus missing = prepare_manifest "
                "n_token_eligible_cells)."
            )
        for name in (
            "patch_cell_patch_index",
            "patch_cell_alpha",
            "patch_cell_emb_he",
            "patch_cell_emb_he_clip",
        ):
            if int(f[name].shape[0]) != n_groups:
                raise RuntimeError(
                    f"{recipe_name}/{expectation.slide_id}: {name} has "
                    f"{int(f[name].shape[0]):,} rows against patch_cell_cell_id_int's "
                    f"{n_groups:,}."
                )

        cell_he_dim = int(f["patch_cell_emb_he"].shape[1])
        cell_clip_dim = int(f["patch_cell_emb_he_clip"].shape[1])
        if cell_he_dim != EXPECTED_PATCH_HE_DIM:
            raise RuntimeError(
                f"{recipe_name}/{expectation.slide_id}: patch_cell_emb_he is {cell_he_dim}-d, "
                f"expected {EXPECTED_PATCH_HE_DIM}."
            )
        if cell_clip_dim <= 0:
            raise RuntimeError(
                f"{recipe_name}/{expectation.slide_id}: {STEP4_REQUIRED_H5_DATASET} has width "
                f"{cell_clip_dim} (the encoder has no trained img_tok_proj)."
            )

        grid_h = int(f.attrs.get("token_grid_h", -1))
        grid_w = int(f.attrs.get("token_grid_w", -1))
        if (grid_h, grid_w) != (EXPECTED_TOKENS_PER_DIM, EXPECTED_TOKENS_PER_DIM):
            raise RuntimeError(
                f"{recipe_name}/{expectation.slide_id}: token grid recorded as {grid_h}x{grid_w}, "
                f"expected {EXPECTED_TOKENS_PER_DIM}x{EXPECTED_TOKENS_PER_DIM}."
            )

        cell_id_int = np.asarray(f["patch_cell_cell_id_int"][:], dtype=np.int64)
        patch_index = np.asarray(f["patch_cell_patch_index"][:], dtype=np.int64)
        alpha = np.asarray(f["patch_cell_alpha"][:], dtype=np.float64)

        # Row order: (patch_index, cell_id_int) ascending
        if patch_index.size:
            if patch_index.min() < 0 or patch_index.max() >= expectation.n_patches:
                raise RuntimeError(
                    f"{recipe_name}/{expectation.slide_id}: patch_cell_patch_index spans "
                    f"[{patch_index.min()}, {patch_index.max()}], outside "
                    f"[0, {expectation.n_patches - 1}]."
                )
            order_ok = np.all(np.lexsort((cell_id_int, patch_index)) == np.arange(n_groups))
            if not order_ok:
                raise RuntimeError(
                    f"{recipe_name}/{expectation.slide_id}: patch-cell rows are not ordered by "
                    "(patch_index, cell_id_int)."
                )

        # Same cell-id set as the membership parquet
        mem_cells = pq.read_table(
            expectation.membership_parquet, columns=["cell_id_int"]
        ).column("cell_id_int").to_numpy(zero_copy_only=False)
        mem_unique = np.unique(np.asarray(mem_cells, dtype=np.int64))
        h5_unique = np.unique(cell_id_int)
        if not np.array_equal(mem_unique, h5_unique):
            only_h5 = np.setdiff1d(h5_unique, mem_unique)[:5].tolist()
            only_mem = np.setdiff1d(mem_unique, h5_unique)[:5].tolist()
            raise RuntimeError(
                f"{recipe_name}/{expectation.slide_id}: the embedding h5 and the membership "
                f"parquet disagree on which cells were pooled. In the h5 only: {only_h5}; in the "
                f"parquet only: {only_mem}."
            )
        if h5_unique.size != n_groups:
            raise RuntimeError(
                f"{recipe_name}/{expectation.slide_id}: {n_groups:,} rows carry only "
                f"{h5_unique.size:,} distinct cell ids (each PanNuke cell is in exactly one patch)."
            )

        # Finite values and non-zero rows (patch_cell_emb_he sampled unless --full_scan)
        clip = np.asarray(f["patch_cell_emb_he_clip"][:], dtype=np.float64)
        n_nonfinite = _nonfinite_count(clip)
        if n_nonfinite:
            raise RuntimeError(
                f"{recipe_name}/{expectation.slide_id}: {n_nonfinite:,} non-finite value(s) in "
                f"{STEP4_REQUIRED_H5_DATASET}."
            )
        clip_norms = np.linalg.norm(clip, axis=1)
        n_zero_rows = int((clip_norms == 0.0).sum())
        if n_zero_rows:
            raise RuntimeError(
                f"{recipe_name}/{expectation.slide_id}: {n_zero_rows:,} all-zero row(s) in "
                f"{STEP4_REQUIRED_H5_DATASET}."
            )

        patch_he_arr = np.asarray(patch_he[:], dtype=np.float64)
        n_nonfinite_patch = _nonfinite_count(patch_he_arr)
        if n_nonfinite_patch:
            raise RuntimeError(
                f"{recipe_name}/{expectation.slide_id}: {n_nonfinite_patch:,} non-finite value(s) "
                "in patch_emb_he."
            )

        if full_scan or n_groups <= EMB_SAMPLE_ROWS:
            he_rows = np.arange(n_groups)
        else:
            he_rows = np.unique(
                np.linspace(0, n_groups - 1, EMB_SAMPLE_ROWS).astype(np.int64)
            )
        cell_he_sample = np.asarray(f["patch_cell_emb_he"][he_rows], dtype=np.float64)
        n_nonfinite_he = _nonfinite_count(cell_he_sample)
        if n_nonfinite_he:
            raise RuntimeError(
                f"{recipe_name}/{expectation.slide_id}: {n_nonfinite_he:,} non-finite value(s) in "
                f"patch_cell_emb_he (over {he_rows.size:,} scanned rows)."
            )

        record.update(
            {
                "n_patches": expectation.n_patches,
                "n_groups_written": n_groups,
                "patch_he_dim": EXPECTED_PATCH_HE_DIM,
                "patch_cell_he_dim": cell_he_dim,
                "patch_cell_he_clip_dim": cell_clip_dim,
                "token_grid": [grid_h, grid_w],
                "emb_clip_l2": {
                    "min": float(clip_norms.min()),
                    "median": float(np.median(clip_norms)),
                    "max": float(clip_norms.max()),
                },
                "patch_cell_emb_he_rows_scanned": int(he_rows.size),
                # Membership path recorded by `_8` (provenance)
                "membership_parquet_recorded": recorded_membership,
            }
        )

    record["alpha"] = verify_alpha_against_ground_truth(
        h5_path, expectation, recipe_name, cell_id_int=cell_id_int, alpha=alpha
    )
    return record


def verify_alpha_against_ground_truth(
    h5_path: Path,
    expectation: SlideExpectation,
    recipe_name: str,
    cell_id_int: Optional[np.ndarray] = None,
    alpha: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Check that patch_cell_alpha equals ground_truth.nuc_px_membership exactly, for each cell
    (each PanNuke cell is in exactly one patch).
    """
    if cell_id_int is None or alpha is None:
        with h5py.File(h5_path, "r") as f:
            cell_id_int = np.asarray(f["patch_cell_cell_id_int"][:], dtype=np.int64)
            alpha = np.asarray(f["patch_cell_alpha"][:], dtype=np.float64)

    gt = pq.read_table(
        expectation.ground_truth_parquet, columns=["cell_id_int", "nuc_px_membership"]
    ).to_pandas()
    gt_map = pd.Series(
        gt["nuc_px_membership"].to_numpy(dtype=np.float64),
        index=gt["cell_id_int"].to_numpy(dtype=np.int64),
    )
    try:
        expected = gt_map.loc[cell_id_int].to_numpy(dtype=np.float64)
    except KeyError as exc:
        raise RuntimeError(
            f"{recipe_name}/{expectation.slide_id}: the embedding h5 carries cell ids absent from "
            f"ground_truth ({exc})."
        ) from exc

    diff = np.abs(alpha - expected)
    n_bad = int((diff > 0).sum())
    if n_bad:
        worst = int(np.argmax(diff))
        raise RuntimeError(
            f"{recipe_name}/{expectation.slide_id}: patch_cell_alpha disagrees with "
            f"ground_truth.nuc_px_membership for {n_bad:,} of {alpha.size:,} cell(s); largest "
            f"absolute difference {diff[worst]:.6g} on cell_id_int={int(cell_id_int[worst])} "
            f"(alpha={alpha[worst]:.6g}, ground truth={expected[worst]:.6g})."
        )
    return {
        "matches_ground_truth": True,
        "n_cells": int(alpha.size),
        "alpha_min": float(alpha.min()) if alpha.size else 0.0,
        "alpha_max": float(alpha.max()) if alpha.size else 0.0,
    }


def _arrow_type_name(t: Any) -> str:
    name = str(t)
    # Arrow spells a utf8 column 'string' or 'large_string' depending on the writer.
    return "string" if name in ("string", "large_string", "utf8", "large_utf8") else name


def verify_main_rows_parquet(
    parquet_path: Path,
    h5_path: Path,
    expectation: SlideExpectation,
    recipe_name: str,
) -> Dict[str, Any]:
    """
    Check the main-row parquet written by `_9.process_slide` for one slide (schema, one candidate
    patch per cell, consistency with the h5).
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise RuntimeError(
            f"{recipe_name}/{expectation.slide_id}: `_9.process_slide` wrote no parquet at "
            f"{parquet_path}."
        )

    pf = pq.ParquetFile(parquet_path)
    n_rows = int(pf.metadata.num_rows)
    if n_rows != expectation.n_expected_groups:
        raise RuntimeError(
            f"{recipe_name}/{expectation.slide_id}: the main-row parquet holds {n_rows:,} rows, "
            f"expected {expectation.n_expected_groups:,} (one per token-eligible cell)."
        )

    schema = pf.schema_arrow
    got = {n: _arrow_type_name(t) for n, t in zip(schema.names, schema.types)}
    problems = []
    for name, want in MAIN_ROW_SCHEMA:
        if name not in got:
            problems.append(f"missing column '{name}'")
        elif got[name] != want:
            problems.append(f"'{name}' is {got[name]}, expected {want}")
    extra = [n for n in got if n not in {c for c, _ in MAIN_ROW_SCHEMA}]
    if extra:
        problems.append(f"unexpected column(s) {extra}")
    if problems:
        raise ValueError(
            f"{recipe_name}/{expectation.slide_id}: the main-row parquet does not match "
            f"the `_9_main_HEemb_cellID.py` schema: {problems}."
        )

    table = pq.read_table(parquet_path).to_pandas()

    for col in STEP4_REQUIRED_PARQUET_COLS:
        if table[col].isna().any():
            raise ValueError(
                f"{recipe_name}/{expectation.slide_id}: column '{col}' holds NaN."
            )
    if table["cell_id_int"].duplicated().any():
        n_dup = int(table["cell_id_int"].duplicated().sum())
        raise ValueError(
            f"{recipe_name}/{expectation.slide_id}: {n_dup:,} duplicated cell_id_int in the "
            "main-row parquet (one row per cell expected)."
        )

    row_idx = table["selected_patch_cell_row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(row_idx), np.arange(n_rows, dtype=np.int64)):
        raise RuntimeError(
            f"{recipe_name}/{expectation.slide_id}: the selected row indices are not a permutation "
            f"of all {n_rows:,} patch-cell rows (a cell appears in more than one patch)."
        )

    n_cand = table["n_candidate_patches"].to_numpy(dtype=np.int64)
    n_bad = int((n_cand != 1).sum())
    if n_bad:
        raise RuntimeError(
            f"{recipe_name}/{expectation.slide_id}: {n_bad:,} of {n_cand.size:,} cells have "
            f"n_candidate_patches != 1 (max {int(n_cand.max())})."
        )

    # The parquet row indices must address the current h5 (stale after a re-extraction)
    with h5py.File(h5_path, "r") as f:
        h5_alpha = np.asarray(f["patch_cell_alpha"][:], dtype=np.float64)
        h5_cells = np.asarray(f["patch_cell_cell_id_int"][:], dtype=np.int64)
    sel_alpha = table["selected_alpha"].to_numpy(dtype=np.float64)
    if not np.allclose(sel_alpha, h5_alpha[row_idx], rtol=0.0, atol=0.0):
        n_off = int((sel_alpha != h5_alpha[row_idx]).sum())
        raise RuntimeError(
            f"{recipe_name}/{expectation.slide_id}: selected_alpha disagrees with the h5 it "
            f"indexes for {n_off:,} row(s). The parquet is stale relative to a re-extracted h5; "
            "re-run with --overwrite."
        )
    if not np.array_equal(
        np.sort(table["cell_id_int"].to_numpy(dtype=np.int64)), np.sort(h5_cells)
    ):
        raise RuntimeError(
            f"{recipe_name}/{expectation.slide_id}: the main-row parquet and the embedding h5 "
            "disagree on the set of cell ids."
        )

    return {
        "n_main_rows": n_rows,
        "n_candidate_patches_max": int(n_cand.max()),
        "n_candidate_patches_mean": float(n_cand.mean()),
    }


# ---------------------------------------------------------------------------------------------
# Cross-recipe pairing
# ---------------------------------------------------------------------------------------------

def _embedded_and_missing_ids(
    preflight: RecipePreflight, emb_dir: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(embedded, missing, ground_truth) cell-id unions over a recipe's slides."""
    embedded: List[np.ndarray] = []
    missing: List[np.ndarray] = []
    ground_truth: List[np.ndarray] = []
    for exp in preflight.slides:
        parquet = main_rows_parquet_path(emb_dir, exp.slide_id)
        embedded.append(
            pq.read_table(parquet, columns=["cell_id_int"])
            .column("cell_id_int")
            .to_numpy(zero_copy_only=False)
            .astype(np.int64)
        )
        missing.append(
            pq.read_table(exp.missing_parquet, columns=["cell_id_int"])
            .column("cell_id_int")
            .to_numpy(zero_copy_only=False)
            .astype(np.int64)
        )
        ground_truth.append(
            pq.read_table(exp.ground_truth_parquet, columns=["cell_id_int"])
            .column("cell_id_int")
            .to_numpy(zero_copy_only=False)
            .astype(np.int64)
        )
    cat = lambda xs: np.unique(np.concatenate(xs)) if xs else np.empty(0, dtype=np.int64)  # noqa: E731
    return cat(embedded), cat(missing), cat(ground_truth)


def cross_recipe_pairing(
    preflights_by_recipe: Mapping[str, RecipePreflight],
    emb_dirs_by_recipe: Mapping[str, Path],
) -> Dict[str, Any]:
    """
    Compare the cell sets of the two recipes: same ground truth, embedded + missing = ground truth for each,
    and symmetric difference of the embedded sets == symmetric difference of the missing sets.
    Returns the counts, including the number of cells embedded in both recipes (n_paired_cells).
    """
    names = sorted(preflights_by_recipe)
    if len(names) != 2:
        raise ValueError(
            f"The cross-recipe pairing needs exactly two recipes, got {names}."
        )
    a, b = names
    e_a, m_a, gt_a = _embedded_and_missing_ids(preflights_by_recipe[a], emb_dirs_by_recipe[a])
    e_b, m_b, gt_b = _embedded_and_missing_ids(preflights_by_recipe[b], emb_dirs_by_recipe[b])

    if not np.array_equal(gt_a, gt_b):
        raise RuntimeError(
            f"Ground-truth cell ids differ between '{a}' ({gt_a.size:,}) and '{b}' "
            f"({gt_b.size:,}); symmetric difference "
            f"{np.setxor1d(gt_a, gt_b).size:,}. The two recipes must describe the same cohort."
        )
    for name, embedded, missing, ground_truth in ((a, e_a, m_a, gt_a), (b, e_b, m_b, gt_b)):
        if np.intersect1d(embedded, missing).size:
            raise RuntimeError(
                f"'{name}': {np.intersect1d(embedded, missing).size:,} cell(s) are both embedded "
                "and listed as missing."
            )
        union = np.union1d(embedded, missing)
        if not np.array_equal(union, ground_truth):
            lost = np.setdiff1d(ground_truth, union)
            raise RuntimeError(
                f"'{name}': embedded ({embedded.size:,}) + missing ({missing.size:,}) = "
                f"{union.size:,} does not reconstruct the ground truth ({ground_truth.size:,}); "
                f"{lost.size:,} cell(s) are in neither set, e.g. {lost[:5].tolist()}."
            )

    e_sym = np.setxor1d(e_a, e_b)
    m_sym = np.setxor1d(m_a, m_b)
    if not np.array_equal(e_sym, m_sym):
        raise RuntimeError(
            "Cross-recipe pairing broken: the symmetric difference of the two embedded cell-id "
            f"sets ({e_sym.size:,} cells) is not the symmetric difference of the two missing sets "
            f"({m_sym.size:,})."
        )

    paired = np.intersect1d(e_a, e_b)
    return {
        "recipes": [a, b],
        "ground_truth_identical": True,
        "n_ground_truth_cells": int(gt_a.size),
        f"n_embedded_{a}": int(e_a.size),
        f"n_embedded_{b}": int(e_b.size),
        f"n_missing_{a}": int(m_a.size),
        f"n_missing_{b}": int(m_b.size),
        "n_missing_union": int(np.union1d(m_a, m_b).size),
        "n_missing_intersection": int(np.intersect1d(m_a, m_b).size),
        "n_paired_cells": int(paired.size),
        "embedded_symmetric_difference": int(e_sym.size),
        "missing_symmetric_difference": int(m_sym.size),
        "no_new_asymmetry": True,
        "note": (
            "The two embedded sets differ because the two geometries lose different nuclei at "
            "tokenization; their symmetric difference equals that of the two missing sets. "
            f"Paired native-vs-scale comparisons use the intersection ({paired.size:,} cells)."
        ),
    }


# ---------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------

def _file_info(path: Path, sha256: bool = False) -> Dict[str, Any]:
    path = Path(path)
    info: Dict[str, Any] = {"path": str(path), "sha256": None}
    if path.exists():
        stat = path.stat()
        info["size_bytes"] = int(stat.st_size)
        info["mtime"] = float(stat.st_mtime)
        if sha256:
            from src_A2.common import sha256_file

            info["sha256"] = sha256_file(path)
    else:
        info["size_bytes"] = None
        info["mtime"] = None
    return info


def build_extract_manifest(
    preflight: RecipePreflight,
    emb_dir: Path,
    encoder_facts: Mapping[str, Any],
    runtime: Mapping[str, Any],
    per_slide: Sequence[Mapping[str, Any]],
    verification: Mapping[str, Any],
    environment: Mapping[str, Any],
    argv: Sequence[str],
    seed: int,
    git_head_value: Optional[str],
    started_at: str,
    finished_at: str,
    wall_seconds: float,
    checksums: bool = False,
    cross_recipe: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Content of extract_manifest.json: inputs, encoder, runtime settings, per-slide counts and checks."""
    from pannuke.pannuke_upstream import he_embeddings_path, main_row_selector_path

    manifest = preflight.manifest
    geometry = dict(manifest.get("geometry_recipe") or {})
    n_embedded = int(sum(int(r["n_groups_written"]) for r in per_slide)) if per_slide else 0
    n_main_rows = int(sum(int(r.get("n_main_rows") or 0) for r in per_slide)) if per_slide else 0
    cand_max = max((int(r.get("n_candidate_patches_max") or 0) for r in per_slide), default=0)

    payload: Dict[str, Any] = {
        "extract_schema_version": EXTRACT_SCHEMA_VERSION,
        "dataset": "PanNuke",
        "step": "A2_pannuke_step3_he_embeddings",
        "is_full_run": bool(
            not preflight.is_partial
            and manifest.get("is_full_run", False)
            and set(runtime.get("stages_run", [])) >= {"extract", "main_rows", "verify"}
        ),
        "geometry_recipe": geometry,
        "prepared_tree": {
            "root": str(preflight.prepared_root),
            "prepare_schema_version": manifest.get("prepare_schema_version"),
            "prepare_is_full_run": manifest.get("is_full_run"),
            "prepare_git_head": manifest.get("git_head"),
            "membership_group": preflight.membership_group,
            "tokens_per_patch": (manifest.get("membership") or {}).get("tokens_per_patch"),
        },
        "encoder": dict(encoder_facts),
        "extraction": {
            "contract_source": str(he_embeddings_path()),
            "contract_function": "extract_embeddings_for_slide",
            "selector_source": str(main_row_selector_path()),
            "selector_function": "process_slide",
            "contract_imported_not_copied": True,
            "row_order": "sorted by (patch_index, cell_id_int); invariant under --batch_size",
            **dict(runtime),
        },
        "cohort": {
            "n_pseudo_slides": len(preflight.slides),
            "slides": list(preflight.slide_ids),
            "restricted_to_slides": (
                list(preflight.restricted_to_slides) if preflight.is_partial else None
            ),
            "n_patches": preflight.total("n_patches"),
            "n_ground_truth_cells": preflight.total("n_ground_truth_cells"),
            "n_missing_cells": preflight.total("n_missing_cells"),
            "n_embedded_cells": n_embedded,
            "n_selected_main_rows": n_main_rows,
            "n_candidate_patches_max": cand_max,
            "note": (
                "n_embedded = ground_truth - missing. The missing cells have no token support "
                "after the geometry transform of _4_prepare_patches.py, hence no embedding."
            ),
        },
        "per_slide": [dict(r) for r in per_slide],
        "verification": dict(verification),
        "cross_recipe": dict(cross_recipe) if cross_recipe else None,
        "downstream_contract": {
            "eval_data_root_for_step4": str(preflight.prepared_root),
            "he_embedding_dir_resolver": "src_A2/embeddings.py:resolve_legacy_he_embedding_dir",
            "expected_dir": str(emb_dir),
            "main_rows_resolver": "src_A2/embeddings.py:ensure_selected_main_rows",
            "expected_main_rows_dir": str(main_rows_dir(emb_dir)),
            "required_h5_dataset": STEP4_REQUIRED_H5_DATASET,
            "required_parquet_columns": list(STEP4_REQUIRED_PARQUET_COLS),
            "selected_center_dist2_is_meaningless": True,
            "selected_center_dist2_note": (
                "computed against the declared synthetic 1-D strip `coords` "
                "(coords_layout='synthetic_1d_strip'); never exercised, because every cell has "
                "n_candidate_patches == 1."
            ),
        },
        "environment": dict(environment),
        "git_head": git_head_value,
        "seed": int(seed),
        "argv": list(argv),
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": float(wall_seconds),
    }
    if checksums:
        for row in payload["per_slide"]:
            for key in ("embeddings_h5", "main_rows_parquet"):
                info = row.get(key)
                if isinstance(info, dict) and info.get("path"):
                    row[key] = _file_info(Path(info["path"]), sha256=True)
    return payload


def slide_file_info(emb_dir: Path, slide_id: str, checksums: bool = False) -> Dict[str, Any]:
    """Per-slide file record for the manifest: size + mtime, sha256 only on request."""
    return {
        "embeddings_h5": _file_info(embedding_h5_path(emb_dir, slide_id), sha256=checksums),
        "main_rows_parquet": _file_info(
            main_rows_parquet_path(emb_dir, slide_id), sha256=checksums
        ),
    }


def environment_record() -> Dict[str, Any]:
    """Versions and host, for the manifest. Torch is reported only if already imported."""
    import platform

    record: Dict[str, Any] = {
        "required_conda_env_local": "histost",
        "required_conda_env_cluster": "histost_env",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "h5py": h5py.__version__,
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cluster_name": os.environ.get("SLURM_CLUSTER_NAME"),
    }
    try:
        import pyarrow

        record["pyarrow"] = pyarrow.__version__
    except Exception:
        record["pyarrow"] = None
    torch_module = sys.modules.get("torch")
    record["torch"] = getattr(torch_module, "__version__", None)
    return record
