#!/usr/bin/env python
"""
Build the patches and cell-token-membership tables of the retained PanNuke tissues (one pseudo-slide per tissue)
for one geometry recipe, in the STHELAR eval dataset format read by _8_compute_HEembs.py.
Runs in the `process_slide` env.

Saved in <ROOT_HistoST> / eval / eval_datasets / pannuke_eval_data / prepared / eval_st_pannuke_<recipe> /
    - metadata_eval_st_pannuke_<recipe>.csv
    - patches/<pseudo_slide>.h5
    - cell_token_membership_<modelgroup>/<pseudo_slide>.parquet
    - cell_token_membership_missing_cells_<modelgroup>/<pseudo_slide>.parquet
    - ground_truth/<pseudo_slide>.parquet: class, areas and centroids of each nucleus
    - nucleus_centroids/<pseudo_slide>.parquet
    - overlays/<tissue>_<tile_id>.png: stored image with its nucleus mask
    - prepare_manifest.json, prepare_log.txt
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Project imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src_A2.common import configure_logger, ensure_dir, write_dataframe, write_json
from pannuke import pannuke_upstream as upstream
from pannuke.pannuke_common import (
    COARSE_V2_SCORABLE_CLASSES,
    DEFAULT_PAD_VALUE,
    DEFAULT_SCALE_IMAGE_INTERPOLATION,
    DEFAULT_SCALE_TISSUE_PX,
    DEFAULT_TILE_ID_STRIDE,
    GEOMETRY_RECIPE_NAMES,
    OVERLAP_LAST_WRITE_WINS,
    OVERLAP_POLICIES,
    PANNUKE_CLASS_NAMES,
    PANNUKE_LICENSE,
    PANNUKE_NOMINAL_MAGNIFICATION,
    PANNUKE_TILE_PX,
    PANNUKE_TISSUE_TO_HISTOST,
    admission_verdict,
    apply_recipe_to_image,
    apply_recipe_to_instance_map,
    assert_cell_id_budget,
    assert_known_class_indices,
    assert_known_tissues,
    build_geometry_recipe,
    composite_cell_id,
    composite_instance_map,
    git_head,
    is_scorable_class,
    map_pannuke_class_to_v2,
    normalize_tissue_name,
    pannuke_root,
    prepared_root,
    pseudo_slide_id_for_tissue,
)

logger = logging.getLogger(__name__)

PREPARE_SCHEMA_VERSION = 1

# Columns of the STHELAR metadata CSV (ST-specific fields are left blank for PanNuke)
METADATA_COLUMNS: Tuple[str, ...] = (
    "dataset_title", "slide_id", "image_filename", "organ", "disease_state", "oncotree_code",
    "species", "patient", "st_technology", "data_publication_date", "license", "study_link",
    "download_page_link1", "inter_spot_dist", "spot_diameter", "spots_under_tissue",
    "preservation_method", "nb_genes", "treatment_comment", "pixel_size_um_embedded",
    "pixel_size_um_estimated", "magnification", "fullres_px_width", "fullres_px_height",
    "tissue", "disease_comment", "subseries", "hest_version_added", "tissue_HistoST",
    "source", "see_in_training", "subset",
)

METADATA_SOURCE = "PANNUKE"
PANNUKE_STUDY_LINK = "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke"

# Maximum fraction of nuclei without token support (raises above it)
DEFAULT_MAX_LOST_CELL_FRACTION = 0.02

# Allowed band for the median of nuc_px / (area_px * model_area_scale)
NUC_PX_RATIO_BAND = (0.90, 1.20)


# ---------------------------------------------------------------------------------------------
# Phase 0 -- cohort and cell ids
# ---------------------------------------------------------------------------------------------

def load_stats_tables(char_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the per-tile and per-nucleus tables of _2_characterize_pannuke.py and check them."""
    tile_path = char_dir / "per_tile_stats.parquet"
    nucleus_path = char_dir / "per_nucleus_stats.parquet"
    for path in (tile_path, nucleus_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `_2_characterize_pannuke.py` first."
            )
    tiles = pd.read_parquet(tile_path)
    nuclei = pd.read_parquet(nucleus_path)

    if tiles["tile_id"].duplicated().any():
        raise ValueError(f"{tile_path} has duplicate tile_id values.")
    tile_index = tiles["tile_index"].to_numpy()
    if not np.array_equal(np.sort(tile_index), np.arange(len(tiles))):
        raise ValueError(
            f"{tile_path}: tile_index is not a contiguous 0..n-1 range (a partial characterisation run cannot be used)."
        )
    bad_shape = tiles[(tiles["image_height"] != PANNUKE_TILE_PX) |
                      (tiles["image_width"] != PANNUKE_TILE_PX)]
    if len(bad_shape):
        raise ValueError(
            f"{tile_path}: {len(bad_shape)} tile(s) are not {PANNUKE_TILE_PX}x{PANNUKE_TILE_PX}."
        )
    bad_mode = sorted(set(tiles["image_mode"]) - {"RGB"})
    if bad_mode:
        raise ValueError(f"{tile_path}: unexpected image modes {bad_mode}; expected RGB only.")

    assert_known_tissues(tiles["tissue"].unique(), context=str(tile_path))
    unknown_classes = sorted(set(nuclei["pannuke_class"]) - set(PANNUKE_CLASS_NAMES))
    if unknown_classes:
        raise ValueError(f"{nucleus_path}: unknown nucleus classes {unknown_classes}.")

    return tiles, nuclei


def resolve_retained_tissues(char_dir: Path) -> List[str]:
    """Retained tissues from reference_support_v2.csv, checked against PANNUKE_TISSUE_TO_HISTOST."""
    path = char_dir / "reference_support_v2.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `_3_check_reference_support.py` first."
        )
    table = pd.read_csv(path)
    retained = sorted(str(t) for t in table.loc[table["retained"].astype(bool), "pannuke_tissue"])
    declared = sorted(PANNUKE_TISSUE_TO_HISTOST)
    if retained != declared:
        raise ValueError(
            f"{path} retains {retained}, but PANNUKE_TISSUE_TO_HISTOST declares {declared}."
        )
    assert_known_tissues(retained, context=str(path))
    return retained


def build_cohort(
    tiles: pd.DataFrame,
    nuclei: pd.DataFrame,
    retained_tissues: Sequence[str],
    keep_empty_tiles: bool,
    stride: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select the retained tiles and scorable nuclei and assign each nucleus its cell_id_int
    (from the global tile_index, so ids are unique across pseudo-slides).
    """
    keep = set(str(t) for t in retained_tissues)

    nuclei_keep = nuclei[nuclei["scorable"].astype(bool) & nuclei["tissue"].isin(keep)].copy()
    if nuclei_keep.empty:
        raise ValueError("No scorable nuclei survive the tissue filter.")

    blank = nuclei_keep["coarse_v2_class"].astype(str).str.len() == 0
    if blank.any():
        raise ValueError(
            f"{int(blank.sum())} scorable nucleus row(s) have an empty coarse_v2_class."
        )
    expected = nuclei_keep["pannuke_class"].map(map_pannuke_class_to_v2)
    mismatch = expected != nuclei_keep["coarse_v2_class"]
    if mismatch.any():
        rows = nuclei_keep.loc[mismatch, ["pannuke_class", "coarse_v2_class"]].head(5)
        raise ValueError(
            f"{int(mismatch.sum())} nucleus row(s) whose coarse_v2_class disagrees with the class map. Examples:\n{rows}"
        )

    tiles_keep = tiles[tiles["tissue"].isin(keep)].copy()
    per_tile_retained = nuclei_keep.groupby("tile_id", sort=False).size()
    tiles_keep["n_retained_nuclei"] = (
        tiles_keep["tile_id"].map(per_tile_retained).fillna(0).astype(int)
    )
    disagree = tiles_keep["n_retained_nuclei"] != tiles_keep["n_scorable_instances"]
    if disagree.any():
        raise ValueError(
            f"{int(disagree.sum())} retained tile(s) whose scorable-nucleus count disagrees between "
            "per_tile_stats and per_nucleus_stats."
        )
    if not keep_empty_tiles:
        tiles_keep = tiles_keep[tiles_keep["n_retained_nuclei"] > 0].copy()

    # Keep only nuclei whose tile is kept
    nuclei_keep = nuclei_keep[nuclei_keep["tile_id"].isin(set(tiles_keep["tile_id"]))].copy()

    max_instances = int(nuclei["instance_ordinal"].max())
    budget = assert_cell_id_budget(len(tiles), max_instances, stride=stride)
    logger.info(
        "Cell-id budget: stride=%d, largest possible id=%d, headroom=%dx",
        budget["stride"], budget["largest_cell_id"], budget["headroom_factor"],
    )

    str_cell_id = _str_cell_id()
    int_cell_id = _int_cell_id()
    nuclei_keep["cell_id_int"] = np.fromiter(
        (composite_cell_id(int(t), int(k), stride=stride)
         for t, k in zip(nuclei_keep["tile_index"], nuclei_keep["instance_ordinal"])),
        dtype=np.int64, count=len(nuclei_keep),
    )
    if nuclei_keep["cell_id_int"].duplicated().any():
        raise ValueError(
            "Duplicate cell_id_int values."
        )
    nuclei_keep["cell_id"] = [str_cell_id(int(v)) for v in nuclei_keep["cell_id_int"]]
    # Check the str <-> int cell id round trip on a few ids
    for probe in (0, len(nuclei_keep) // 2, len(nuclei_keep) - 1):
        row = nuclei_keep.iloc[probe]
        if int_cell_id(str(row["cell_id"])) != int(row["cell_id_int"]):
            raise ValueError(
                f"cell id round trip failed for {row['cell_id']} / {row['cell_id_int']}."
            )
    nuclei_keep["cell_id_int"] = nuclei_keep["cell_id_int"].astype(np.int32)

    nuclei_keep["pseudo_slide_id"] = [pseudo_slide_id_for_tissue(t) for t in nuclei_keep["tissue"]]
    tiles_keep["pseudo_slide_id"] = [pseudo_slide_id_for_tissue(t) for t in tiles_keep["tissue"]]
    tiles_keep = tiles_keep.sort_values("tile_index").reset_index(drop=True)
    nuclei_keep = nuclei_keep.sort_values(["tile_index", "instance_ordinal"]).reset_index(drop=True)
    return tiles_keep, nuclei_keep


def _str_cell_id():
    from src.utils.cell_ids import str_cell_id

    return str_cell_id


def _int_cell_id():
    from src.utils.cell_ids import int_cell_id

    return int_cell_id


# ---------------------------------------------------------------------------------------------
# Phase 1 -- one scan over the raw folds
# ---------------------------------------------------------------------------------------------

class _SlideBuffer:
    """Per-pseudo-slide accumulator: images, coords, barcodes and the open membership writer."""

    def __init__(self, pseudo_slide_id: str, n_patches: int, stored_px: int):
        self.pseudo_slide_id = pseudo_slide_id
        self.n_patches = int(n_patches)
        self.stored_px = int(stored_px)
        self.imgs = np.empty((self.n_patches, self.stored_px, self.stored_px, 3), dtype=np.uint8)
        self.coords = np.empty((self.n_patches, 2), dtype=np.int32)
        self.barcodes: List[str] = []
        self.writer = None
        self.token_eligible: Set[int] = set()
        self.n_filled = 0

    def add_patch(self, patch_id: str, image: np.ndarray) -> None:
        if self.n_filled >= self.n_patches:
            raise RuntimeError(
                f"{self.pseudo_slide_id}: more patches than the {self.n_patches} planned."
            )
        self.imgs[self.n_filled] = image
        # Synthetic coords: patches side by side on a 1-D strip (stride = stored width)
        self.coords[self.n_filled, 0] = np.int32(self.n_filled * self.stored_px)
        self.coords[self.n_filled, 1] = np.int32(0)
        self.barcodes.append(patch_id)
        self.n_filled += 1


def _decode_tile_rgb(decoders: Dict[str, Any], tile: Dict[str, Any], tile_id: str) -> np.ndarray:
    image = decoders["_open_image"](decoders["_image_bytes"](tile["image"]))
    if image.mode != "RGB":
        raise ValueError(f"Tile '{tile_id}' has image mode {image.mode!r}, expected 'RGB'.")
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array.shape != (PANNUKE_TILE_PX, PANNUKE_TILE_PX, 3):
        raise ValueError(
            f"Tile '{tile_id}' decoded to shape {array.shape}, expected "
            f"({PANNUKE_TILE_PX}, {PANNUKE_TILE_PX}, 3)."
        )
    return array


def _decode_tile_masks(
    decoders: Dict[str, Any],
    tile: Dict[str, Any],
    tile_id: str,
    ordinals: np.ndarray,
    classes: Sequence[str],
    class_names: Sequence[str],
) -> List[np.ndarray]:
    """Decode the masks of the retained nuclei of a tile, checking each class against per_nucleus_stats."""
    instances = tile["instances"]
    categories = tile["categories"]
    if len(instances) != len(categories):
        raise ValueError(
            f"Tile '{tile_id}': {len(instances)} instance mask(s) but {len(categories)} categories."
        )
    assert_known_class_indices(categories, context=f"{tile_id} (raw codes)")

    masks: List[np.ndarray] = []
    for ordinal, expected_class in zip(ordinals, classes):
        idx = int(ordinal) - 1
        if not (0 <= idx < len(instances)):
            raise ValueError(
                f"Tile '{tile_id}': instance_ordinal {int(ordinal)} is out of range for "
                f"{len(instances)} instance(s)."
            )
        observed = class_names[int(categories[idx])]
        if observed != expected_class:
            raise ValueError(
                f"Tile '{tile_id}', instance {int(ordinal)}: the raw fold says '{observed}' where "
                f"per_nucleus_stats says '{expected_class}'."
            )
        if not is_scorable_class(observed):
            raise ValueError(
                f"Tile '{tile_id}', instance {int(ordinal)}: unscorable class '{observed}' in the retained cohort."
            )
        mask = np.asarray(
            decoders["_open_image"](decoders["_image_bytes"](instances[idx])).convert("1")
        ).astype(bool)
        if mask.shape != (PANNUKE_TILE_PX, PANNUKE_TILE_PX):
            raise ValueError(
                f"Tile '{tile_id}', instance {int(ordinal)}: mask shape {mask.shape}, expected "
                f"({PANNUKE_TILE_PX}, {PANNUKE_TILE_PX})."
            )
        masks.append(mask)
    return masks


def _membership_rows(
    mask_t: np.ndarray,
    token_ids_flat: np.ndarray,
    tokens_per_patch: int,
    patch_id: str,
    str_cell_id,
) -> Optional[pd.DataFrame]:
    """Cell-token-membership rows of one patch (same computation as _1_download_eval_data.py), or None if empty."""
    mask_flat = mask_t.ravel()
    valid = mask_flat > 0
    if not np.any(valid):
        return None

    cell_flat = mask_flat[valid].astype(np.int64, copy=False)
    tok_flat = token_ids_flat[valid].astype(np.int64, copy=False)

    key = cell_flat * np.int64(tokens_per_patch) + tok_flat
    uniq_key, counts = np.unique(key, return_counts=True)

    cell_ids = (uniq_key // tokens_per_patch).astype(np.int32, copy=False)
    tok_ids = (uniq_key % tokens_per_patch).astype(np.int32, copy=False)
    nuc_px = counts.astype(np.int32, copy=False)

    cell_id_str = np.fromiter(
        (str_cell_id(int(c)) for c in cell_ids), dtype=object, count=cell_ids.shape[0]
    )
    return pd.DataFrame(
        {
            "cell_id_int": cell_ids,
            "cell_id": cell_id_str,
            "patch_id": np.repeat(patch_id, cell_ids.shape[0]).astype(object),
            "token_index": tok_ids,
            "nuc_px": nuc_px,
        }
    )


def scan_and_build(
    raw_dir: Path,
    tiles_keep: pd.DataFrame,
    nuclei_keep: pd.DataFrame,
    recipe,
    group: Dict[str, Any],
    out_root: Path,
    overlap_policy: str,
    max_tiles: Optional[int],
    n_tiles_total: int,
    scan_total: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Scan all tiles of the raw folds (tile_index = scan order); for the retained ones, decode, composite,
    transform and tokenize the nuclei, buffer the patch and write its membership rows.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    contract = upstream.contract()
    decoders = upstream.decoders()
    str_cell_id = _str_cell_id()

    tokens_per_patch = int(group["tokens_per_patch"])
    token_ids_flat = contract.token_ids_flat_lut(
        upstream.EXPECTED_FINAL_HW, upstream.EXPECTED_FINAL_HW,
        int(group["tokens_per_dim"]), contract.vit_token_px,
    )

    parquets = sorted(p for p in raw_dir.rglob("*.parquet") if not p.name.startswith("."))
    if not parquets:
        raise FileNotFoundError(f"No fold parquet found under {raw_dir}.")
    class_names = decoders["read_hf_feature_names"](parquets[0])["categories"]
    if list(class_names) != list(PANNUKE_CLASS_NAMES):
        raise ValueError(
            f"The release's own class order {list(class_names)} differs from the declared "
            f"{list(PANNUKE_CLASS_NAMES)}."
        )

    tile_index_by_id = dict(zip(tiles_keep["tile_id"], tiles_keep["tile_index"]))
    slide_by_tile = dict(zip(tiles_keep["tile_id"], tiles_keep["pseudo_slide_id"]))
    retained_tile_ids = set(tiles_keep["tile_id"])
    expected_tile_index = dict(zip(tiles_keep["tile_id"], tiles_keep["tile_index"]))

    by_tile: Dict[str, Dict[str, np.ndarray]] = {}
    for tile_id, block in nuclei_keep.groupby("tile_id", sort=False):
        by_tile[tile_id] = {
            "row_pos": block.index.to_numpy(),
            "ordinals": block["instance_ordinal"].to_numpy(),
            "cell_ids": block["cell_id_int"].to_numpy(dtype=np.int64),
            "classes": block["pannuke_class"].to_numpy(),
        }

    counts = tiles_keep.groupby("pseudo_slide_id", sort=False).size()
    buffers = {
        psid: _SlideBuffer(psid, int(n), int(recipe.stored_px)) for psid, n in counts.items()
    }
    membership_dir = ensure_dir(out_root / upstream.membership_dirname(group["name"]))

    n_nuclei = len(nuclei_keep)
    realized_px_src = np.zeros(n_nuclei, dtype=np.int64)
    nuc_px_membership = np.zeros(n_nuclei, dtype=np.int64)
    centroid_y_src = np.full(n_nuclei, np.nan, dtype=np.float64)
    centroid_x_src = np.full(n_nuclei, np.nan, dtype=np.float64)

    stats = {
        "n_tiles_scanned": 0,
        "n_tiles_written": 0,
        "n_overlap_px": 0,
        "n_contested_px": 0,
        "n_tiles_with_overlap": 0,
        "nuc_px_ratios": [],
    }

    progress = tqdm(
        total=scan_total,
        desc="  scan tiles (decode -> composite -> tokenize)",
        unit="tile",
        disable=not show_progress,
        dynamic_ncols=True,
    )
    running_tile_index = 0
    for path in parquets:
        fold = path.stem.split("-")[0]
        progress.set_postfix_str(f"{fold}, written {stats['n_tiles_written']}", refresh=False)
        for tile in decoders["iter_tiles"](path, max_tiles=max_tiles):
            progress.update(1)
            tile_id = f"{fold}_{int(tile['row_group']):04d}_{int(tile['row_in_group']):05d}"
            stats["n_tiles_scanned"] += 1

            # The scan position must match the tile_index of per_tile_stats (cell ids depend on it)
            if tile_id in expected_tile_index and running_tile_index != int(expected_tile_index[tile_id]):
                raise ValueError(
                    f"Tile '{tile_id}' is scan position {running_tile_index} but per_tile_stats records "
                    f"tile_index={int(expected_tile_index[tile_id])}."
                )
            if tile_id not in retained_tile_ids:
                running_tile_index += 1
                continue

            entry = by_tile[tile_id]
            image = _decode_tile_rgb(decoders, tile, tile_id)
            masks = _decode_tile_masks(
                decoders, tile, tile_id, entry["ordinals"], entry["classes"], class_names
            )

            comp = composite_instance_map(
                masks, entry["cell_ids"].tolist(), (PANNUKE_TILE_PX, PANNUKE_TILE_PX),
                policy=overlap_policy,
            )
            rows = entry["row_pos"]
            realized_px_src[rows] = comp.realized_px
            centroid_y_src[rows] = comp.centroid_y
            centroid_x_src[rows] = comp.centroid_x
            stats["n_overlap_px"] += comp.n_overlap_px
            stats["n_contested_px"] += comp.n_contested_px
            stats["n_tiles_with_overlap"] += int(comp.n_overlap_px > 0)

            stored_img = apply_recipe_to_image(image, recipe)
            stored_map = apply_recipe_to_instance_map(comp.instance_map, recipe)
            if stored_img.shape[:2] != stored_map.shape:
                raise RuntimeError(
                    f"Tile '{tile_id}': image canvas {stored_img.shape[:2]} and mask canvas "
                    f"{stored_map.shape} disagree."
                )

            mask_t = contract.apply_membership_preprocess_to_mask(
                stored_map, resize_size=upstream.EXPECTED_FINAL_HW, crop_size=upstream.EXPECTED_FINAL_HW
            )
            if mask_t.shape != (upstream.EXPECTED_FINAL_HW, upstream.EXPECTED_FINAL_HW):
                raise RuntimeError(
                    f"Tile '{tile_id}': the membership preprocess produced {mask_t.shape}, expected "
                    f"({upstream.EXPECTED_FINAL_HW}, {upstream.EXPECTED_FINAL_HW})."
                )

            chunk = _membership_rows(mask_t, token_ids_flat, tokens_per_patch, tile_id, str_cell_id)
            psid = slide_by_tile[tile_id]
            buf = buffers[psid]
            if chunk is not None:
                if int(chunk["token_index"].max()) >= tokens_per_patch:
                    raise RuntimeError(
                        f"Tile '{tile_id}': token_index {int(chunk['token_index'].max())} is outside "
                        f"the {tokens_per_patch}-token grid."
                    )
                emitted = set(int(v) for v in chunk["cell_id_int"])
                unexpected = emitted - set(int(v) for v in entry["cell_ids"])
                if unexpected:
                    raise RuntimeError(
                        f"Tile '{tile_id}': membership emitted cell ids {sorted(unexpected)[:5]} that "
                        "do not belong to it."
                    )
                per_cell = chunk.groupby("cell_id_int", sort=False)["nuc_px"].sum()
                id_to_row = dict(zip(entry["cell_ids"].tolist(), rows.tolist()))
                for cid, px in per_cell.items():
                    nuc_px_membership[id_to_row[int(cid)]] = int(px)
                buf.token_eligible.update(emitted)

                table = pa.Table.from_pandas(
                    chunk, preserve_index=False, schema=contract.membership_schema
                )
                if buf.writer is None:
                    buf.writer = pq.ParquetWriter(
                        str(membership_dir / f"{psid}.parquet"),
                        contract.membership_schema,
                        compression="zstd",
                        use_dictionary=True,
                    )
                buf.writer.write_table(table)

            buf.add_patch(tile_id, stored_img)
            stats["n_tiles_written"] += 1
            running_tile_index += 1
            if stats["n_tiles_written"] % 50 == 0:
                progress.set_postfix_str(f"{fold}, written {stats['n_tiles_written']}", refresh=False)
    progress.close()

    if max_tiles is None and stats["n_tiles_scanned"] != n_tiles_total:
        raise ValueError(
            f"Scanned {stats['n_tiles_scanned']} tiles but per_tile_stats has {n_tiles_total}."
        )

    for psid, buf in buffers.items():
        if buf.n_filled != buf.n_patches:
            raise RuntimeError(
                f"{psid}: filled {buf.n_filled} of {buf.n_patches} planned patches."
            )

    nuclei_keep = nuclei_keep.copy()
    nuclei_keep["realized_px_src"] = realized_px_src
    nuclei_keep["nuc_px_membership"] = nuc_px_membership
    nuclei_keep["centroid_y_src256"] = centroid_y_src
    nuclei_keep["centroid_x_src256"] = centroid_x_src
    return {"buffers": buffers, "nuclei": nuclei_keep, "stats": stats, "membership_dir": membership_dir}


# ---------------------------------------------------------------------------------------------
# Phase 2 -- per-pseudo-slide close-out
# ---------------------------------------------------------------------------------------------

def finalize_pseudo_slide(
    psid: str,
    buf: _SlideBuffer,
    block: pd.DataFrame,
    recipe,
    group: Dict[str, Any],
    out_root: Path,
    allow_admission_shortfall: bool,
) -> Dict[str, Any]:
    """Close the writers, assert the membership/missing partition, and write every artifact."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    contract = upstream.contract()
    str_cell_id = _str_cell_id()

    if buf.writer is not None:
        buf.writer.close()
        buf.writer = None
    membership_path = out_root / upstream.membership_dirname(group["name"]) / f"{psid}.parquet"
    if not membership_path.exists():
        # Empty membership file when no cell has token support
        ensure_dir(membership_path.parent)
        pq.write_table(
            contract.membership_schema.empty_table(), str(membership_path), compression="zstd"
        )

    gt_ids = set(int(v) for v in block["cell_id_int"])
    eligible = set(int(v) for v in buf.token_eligible)
    missing = gt_ids - eligible
    stray = eligible - gt_ids
    if stray:
        raise RuntimeError(
            f"{psid}: {len(stray)} membership cell id(s) are absent from the ground truth, e.g. "
            f"{sorted(stray)[:5]}."
        )
    if len(eligible) + len(missing) != len(gt_ids):
        raise RuntimeError(
            f"{psid}: membership ({len(eligible)}) + missing ({len(missing)}) != ground truth "
            f"({len(gt_ids)})."
        )

    missing_dir = ensure_dir(out_root / upstream.missing_membership_dirname(group["name"]))
    missing_ids = np.array(sorted(missing), dtype=np.int32)
    if missing_ids.size:
        missing_df = pd.DataFrame(
            {
                "cell_id_int": missing_ids,
                "cell_id": np.fromiter(
                    (str_cell_id(int(c)) for c in missing_ids), dtype=object, count=missing_ids.size
                ),
            }
        )
        table = pa.Table.from_pandas(
            missing_df, preserve_index=False, schema=contract.missing_membership_schema
        )
    else:
        table = contract.missing_membership_schema.empty_table()
    pq.write_table(table, str(missing_dir / f"{psid}.parquet"), compression="zstd")

    barcodes_arr = np.array(buf.barcodes, dtype=object).reshape(-1, 1)
    contract.write_hest_h5(
        ensure_dir(out_root / "patches") / f"{psid}.h5", barcodes_arr, buf.coords, buf.imgs
    )

    # Rename on the block itself (the block is returned to main)
    block = block.rename(columns={"tile_id": "patch_id", "area_px": "area_px_src"}).copy()
    block["lost_at_flatten"] = block["realized_px_src"].to_numpy() == 0
    block["lost_at_membership"] = (~block["lost_at_flatten"]) & (
        block["nuc_px_membership"].to_numpy() == 0
    )
    block["area_px_stored"] = block["nuc_px_membership"].astype(np.int32)
    block["tissue_HistoST"] = block["tissue"].map(PANNUKE_TISSUE_TO_HISTOST)

    local_y = recipe.source_point_to_stored(block["centroid_y_src256"].to_numpy())
    local_x = recipe.source_point_to_stored(block["centroid_x_src256"].to_numpy())
    patch_slot = {pid: i for i, pid in enumerate(buf.barcodes)}
    slot = block["patch_id"].map(patch_slot).to_numpy(dtype=np.float64)
    block["centroid_y_local"] = local_y
    block["centroid_x_local"] = local_x
    block["centroid_y_slide"] = local_y
    block["centroid_x_slide"] = local_x + slot * float(recipe.stored_px)

    gt_columns = [
        "cell_id_int", "cell_id", "patch_id", "tile_index", "fold", "tissue", "tissue_HistoST",
        "pannuke_class", "coarse_v2_class", "area_px_src", "area_px_stored", "nuc_px_membership",
        "touches_border", "lost_at_flatten", "lost_at_membership",
        "centroid_y_local", "centroid_x_local", "centroid_y_src256", "centroid_x_src256",
        "centroid_y_slide", "centroid_x_slide",
    ]
    write_dataframe(block[gt_columns], ensure_dir(out_root / "ground_truth") / f"{psid}.parquet")

    # Same schema as the nucleus centroids read by _9_main_HEemb_cellID.py
    centroids = pd.DataFrame(
        {
            "cell_id": block["cell_id"].to_numpy(),
            "x": block["centroid_x_slide"].to_numpy(dtype=np.float64),
            "y": block["centroid_y_slide"].to_numpy(dtype=np.float64),
        }
    )
    write_dataframe(centroids, ensure_dir(out_root / "nucleus_centroids") / f"{psid}.parquet")

    eligible_block = block[block["cell_id_int"].isin(eligible)]
    class_counts = {
        c: int((eligible_block["coarse_v2_class"] == c).sum()) for c in COARSE_V2_SCORABLE_CLASSES
    }
    verdict = admission_verdict(class_counts)
    verdict["pseudo_slide_id"] = psid
    verdict["n_patches"] = int(buf.n_filled)
    verdict["n_ground_truth_cells"] = len(gt_ids)
    verdict["n_token_eligible_cells"] = len(eligible)
    verdict["n_missing_cells"] = len(missing)
    verdict["n_lost_at_flatten"] = int(block["lost_at_flatten"].sum())
    verdict["n_lost_at_membership"] = int(block["lost_at_membership"].sum())
    verdict["allow_admission_shortfall"] = bool(allow_admission_shortfall)
    return {"verdict": verdict, "block": block}


# ---------------------------------------------------------------------------------------------
# Phase 3 -- metadata, overlays, manifest
# ---------------------------------------------------------------------------------------------

def build_metadata_csv(tiles_keep: pd.DataFrame, recipe, out_root: Path) -> pd.DataFrame:
    """Metadata CSV (one row per pseudo-slide), with slide_id equal to the stems of patches/*.h5."""
    rows: List[Dict[str, Any]] = []
    for psid, block in tiles_keep.groupby("pseudo_slide_id", sort=True):
        tissue = str(block["tissue"].iloc[0])
        row = {c: "" for c in METADATA_COLUMNS}
        row.update(
            {
                "dataset_title": "PanNuke",
                "slide_id": psid,
                "organ": PANNUKE_TISSUE_TO_HISTOST[tissue],
                "species": "Homo sapiens",
                "license": PANNUKE_LICENSE,
                "study_link": PANNUKE_STUDY_LINK,
                "pixel_size_um_embedded": recipe.mpp_um_per_px,
                "magnification": PANNUKE_NOMINAL_MAGNIFICATION,
                "fullres_px_width": int(len(block) * recipe.stored_px),
                "fullres_px_height": int(recipe.stored_px),
                "tissue": tissue,
                "tissue_HistoST": PANNUKE_TISSUE_TO_HISTOST[tissue],
                "source": METADATA_SOURCE,
                "see_in_training": False,
                "subset": f"{METADATA_SOURCE}_pannuke_{recipe.name}",
            }
        )
        rows.append(row)
    metadata = pd.DataFrame(rows, columns=list(METADATA_COLUMNS))
    write_dataframe(metadata, out_root / f"metadata_eval_st_pannuke_{recipe.name}.csv")
    return metadata


def write_overlays(
    raw_dir: Path,
    tiles_keep: pd.DataFrame,
    nuclei_keep: pd.DataFrame,
    recipe,
    overlap_policy: str,
    out_dir: Path,
    n_overlays: int,
    show_progress: bool = True,
) -> List[str]:
    """Save the stored image of a sample of tiles, beside the same image with the stored nucleus mask on top."""
    if n_overlays <= 0:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decoders = upstream.decoders()
    tile_rows = tiles_keep.to_dict("records")
    # select_overlay_tiles returns tile_index values (not list positions)
    chosen_index = decoders["select_overlay_tiles"](tile_rows, n_overlays)
    by_index = {int(r["tile_index"]): r for r in tile_rows}
    unknown = [int(i) for i in chosen_index if int(i) not in by_index]
    if unknown:
        raise RuntimeError(
            f"Overlay selection returned tile_index values {unknown[:5]} that are not in the "
            "retained cohort."
        )
    chosen = [by_index[int(i)] for i in chosen_index]

    # Group by fold file ((row_group, row_in_group) is only unique within a file)
    chosen_by_path: Dict[Path, List[Dict[str, Any]]] = defaultdict(list)
    for row in chosen:
        path = raw_dir / "data" / str(row["source_file"])
        if not path.exists():
            matches = [p for p in raw_dir.rglob(str(row["source_file"])) if not p.name.startswith(".")]
            if not matches:
                raise FileNotFoundError(f"Cannot locate fold file {row['source_file']} under {raw_dir}.")
            path = matches[0]
        chosen_by_path[path].append(row)

    by_tile = {tid: block for tid, block in nuclei_keep.groupby("tile_id", sort=False)}
    written: List[str] = []
    overlay_bar = tqdm(
        total=len(chosen), desc="  render geometry overlays", unit="tile",
        disable=not show_progress, dynamic_ncols=True,
    )
    for path, rows in chosen_by_path.items():
        spec: Dict[int, List[int]] = defaultdict(list)
        for row in rows:
            spec[int(row["row_group"])].append(int(row["row_in_group"]))
        fetched = decoders["read_specific_tiles"](path, {k: sorted(v) for k, v in spec.items()})
        class_names = decoders["read_hf_feature_names"](path)["categories"]
        for row in rows:
            key = (int(row["row_group"]), int(row["row_in_group"]))
            if key not in fetched:
                raise RuntimeError(
                    f"Overlay tile {row['tile_id']} was requested from {path.name} but not returned."
                )
            tile_id = str(row["tile_id"])
            block = by_tile[tile_id]
            image = _decode_tile_rgb(decoders, fetched[key], tile_id)
            masks = _decode_tile_masks(
                decoders, fetched[key], tile_id,
                block["instance_ordinal"].to_numpy(), block["pannuke_class"].to_numpy(),
                class_names,
            )
            comp = composite_instance_map(
                masks, block["cell_id_int"].to_numpy(dtype=np.int64).tolist(),
                (PANNUKE_TILE_PX, PANNUKE_TILE_PX), policy=overlap_policy,
            )
            stored_img = apply_recipe_to_image(image, recipe)
            stored_map = apply_recipe_to_instance_map(comp.instance_map, recipe)

            fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
            axes[0].imshow(stored_img)
            axes[0].set_title(f"{tile_id} - stored image ({recipe.name})", fontsize=9)
            blend = stored_img.astype(np.float32) / 255.0
            hit = stored_map != 0
            blend[hit] = 0.45 * blend[hit] + 0.55 * np.array([1.0, 0.15, 0.15], dtype=np.float32)
            axes[1].imshow(blend)
            axes[1].set_title(
                f"mask on image - {int(hit.sum())} nucleus px of {len(masks)} nuclei", fontsize=9
            )
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            fig.tight_layout()
            dest = ensure_dir(out_dir) / f"{row['tissue']}_{tile_id}.png"
            fig.savefig(dest, dpi=110)
            plt.close(fig)
            written.append(dest.name)
            overlay_bar.update(1)
    overlay_bar.close()

    if len(written) != len(chosen):
        raise RuntimeError(
            f"Requested {len(chosen)} overlays but wrote {len(written)}."
        )
    return written


def _source_provenance(data_root: Path) -> Dict[str, Any]:
    path = Path(data_root) / "manifest.json"
    if not path.exists():
        return {"manifest_present": False}
    with path.open() as f:
        manifest = json.load(f)
    return {
        "manifest_present": True,
        "repo_id": manifest.get("repo_id"),
        "resolved_revision": manifest.get("resolved_revision"),
        "license": manifest.get("license"),
        "citations": manifest.get("citations"),
        "files": [
            {"path": rec.get("path"), "sha256": rec.get("sha256")}
            for rec in manifest.get("files", [])
        ],
    }


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------

def main(args: argparse.Namespace) -> int:
    data_root = Path(args.data_dir).expanduser().resolve() if args.data_dir else pannuke_root()
    recipe = build_geometry_recipe(
        args.geometry_recipe,
        scale_tissue_px=args.scale_tissue_px,
        image_interpolation=args.scale_image_interpolation,
        pad_value=args.pad_value,
    )
    out_root = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir else prepared_root(data_root, recipe.name)
    )
    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{out_root} already exists and is not empty. Pass --overwrite to rebuild it, or "
            "--out_dir to write elsewhere."
        )
    ensure_dir(out_root)
    # Log file attached to this module's logger
    configure_logger(out_root / "prepare_log.txt", name=logger.name)

    group = upstream.resolve_membership_group()
    geometry = {**recipe.as_dict(token_px=group["vit_token_px"])}
    logger.info("Geometry recipe '%s': %s", recipe.name, json.dumps(geometry, default=str)[:400])
    if not geometry["token_aligned"]:
        logger.warning(
            "Recipe '%s' leaves %d mixed tissue/pad tokens; about %.0f%% of nuclei touch a tile "
            "border and may pool a partly-blank token.",
            recipe.name, geometry["n_mixed_tokens"], 30.8,
        )
    logger.info("Membership group: %s (%d tokens/patch)", group["name"], group["tokens_per_patch"])

    char_dir = data_root / "characterization"
    raw_dir = data_root / "raw"
    tiles, nuclei = load_stats_tables(char_dir)
    retained_tissues = resolve_retained_tissues(char_dir)
    if args.tissues:
        requested = [normalize_tissue_name(t) for t in args.tissues]
        assert_known_tissues(requested, context="--tissues")
        unknown = sorted(set(requested) - set(retained_tissues))
        if unknown:
            raise ValueError(f"--tissues names non-retained tissue(s) {unknown}.")
        retained_tissues = sorted(requested)
    logger.info("Retained tissues (%d): %s", len(retained_tissues), retained_tissues)

    tiles_keep, nuclei_keep = build_cohort(
        tiles, nuclei, retained_tissues, args.keep_empty_tiles, DEFAULT_TILE_ID_STRIDE
    )
    n_zero_scorable = int(
        (tiles[tiles["tissue"].isin(set(retained_tissues))]["n_scorable_instances"] == 0).sum()
    )
    logger.info(
        "Cohort: %d tiles (%d zero-scorable %s), %d scorable nuclei, %d pseudo-slides",
        len(tiles_keep), n_zero_scorable,
        "kept" if args.keep_empty_tiles else "dropped",
        len(nuclei_keep), tiles_keep["pseudo_slide_id"].nunique(),
    )

    # Work estimate (all tiles are scanned, since cell ids depend on the scan order)
    per_fold = tiles.groupby("source_file", sort=True).size()
    scan_total = int(sum(
        min(args.max_tiles, int(n)) if args.max_tiles else int(n) for n in per_fold
    ))
    est_bytes = len(tiles_keep) * recipe.stored_px * recipe.stored_px * 3
    logger.info(
        "Work: scan %d tiles across %d fold file(s), decode %d of them, write %d patches "
        "(~%.2f GB uncompressed) into %s",
        scan_total, len(per_fold), len(tiles_keep), len(tiles_keep),
        est_bytes / 1e9, out_root,
    )

    built = scan_and_build(
        raw_dir, tiles_keep, nuclei_keep, recipe, group, out_root,
        args.overlap_policy, args.max_tiles, n_tiles_total=len(tiles),
        scan_total=scan_total, show_progress=not args.no_progress,
    )
    nuclei_out = built["nuclei"]

    verdicts: List[Dict[str, Any]] = []
    blocks: List[pd.DataFrame] = []
    # Progress bar over the pseudo-slide writes
    slide_bar = tqdm(
        sorted(nuclei_out["pseudo_slide_id"].unique()),
        desc="  write pseudo-slides (h5 + parquets)",
        unit="slide",
        disable=args.no_progress,
        dynamic_ncols=True,
    )
    grouped = dict(tuple(nuclei_out.groupby("pseudo_slide_id", sort=True)))
    for psid in slide_bar:
        slide_bar.set_postfix_str(psid, refresh=False)
        result = finalize_pseudo_slide(
            psid, built["buffers"][psid], grouped[psid], recipe, group, out_root,
            args.allow_admission_shortfall,
        )
        verdicts.append(result["verdict"])
        blocks.append(result["block"])
    slide_bar.close()
    finalized = pd.concat(blocks, ignore_index=True)

    metadata = build_metadata_csv(tiles_keep, recipe, out_root)
    patch_stems = {p.stem for p in (out_root / "patches").glob("*.h5") if not p.name.startswith("._")}
    if patch_stems != set(metadata["slide_id"]):
        raise RuntimeError(
            f"Metadata slide_id set {sorted(set(metadata['slide_id']))} differs from the patch "
            f"stems {sorted(patch_stems)}."
        )

    # --- checks: nucleus loss, nuc_px ratio, admission -------------------------------------
    n_lost = int(finalized["lost_at_flatten"].sum() + finalized["lost_at_membership"].sum())
    lost_fraction = n_lost / max(len(finalized), 1)
    lost_rows = finalized[finalized["lost_at_flatten"] | finalized["lost_at_membership"]]
    lost_by_class = lost_rows.groupby("coarse_v2_class", sort=True).size().to_dict()
    lost_area = lost_rows["area_px_src"].to_numpy(dtype=np.int64)
    lost_area_buckets = {
        label: int(((lost_area >= lo) & (lost_area < hi)).sum())
        for label, (lo, hi) in {
            "1": (1, 2), "2-3": (2, 4), "4-5": (4, 6), "6-10": (6, 11),
            "11-25": (11, 26), "26+": (26, 1 << 62),
        }.items()
    }
    logger.info(
        "Nucleus loss: %d of %d (%.4f%%) -- by class %s, largest lost area %s px^2",
        n_lost, len(finalized), 100 * lost_fraction, lost_by_class,
        int(lost_area.max()) if lost_area.size else "n/a",
    )
    if lost_fraction > args.max_lost_cell_fraction:
        raise RuntimeError(
            f"Lost {n_lost} of {len(finalized)} nuclei ({lost_fraction:.4%}), above "
            f"--max_lost_cell_fraction={args.max_lost_cell_fraction:.4%}."
        )

    supported = finalized[finalized["nuc_px_membership"] > 0]
    ratios = (
        supported["nuc_px_membership"].to_numpy(dtype=np.float64)
        / (supported["area_px_src"].to_numpy(dtype=np.float64) * recipe.model_area_scale)
    )
    ratio_median = float(np.median(ratios)) if ratios.size else float("nan")
    logger.info(
        "nuc_px / (area * scale^2): median %.4f, p05 %.4f, p95 %.4f over %d nuclei",
        ratio_median,
        float(np.percentile(ratios, 5)) if ratios.size else float("nan"),
        float(np.percentile(ratios, 95)) if ratios.size else float("nan"),
        ratios.size,
    )
    lo, hi = NUC_PX_RATIO_BAND
    if ratios.size and not (lo <= ratio_median <= hi):
        raise RuntimeError(
            f"Median nuc_px / (area * scale^2) is {ratio_median:.4f}, outside NUC_PX_RATIO_BAND "
            f"[{lo}, {hi}]."
        )

    admission_applicable = args.max_tiles is None
    failed = [v for v in verdicts if not v["passes"]]
    for v in verdicts:
        logger.info(
            "%-22s patches=%4d cells=%6d eligible=%6d missing=%3d classes>=%d: %d  margins=%s  %s",
            v["pseudo_slide_id"], v["n_patches"], v["n_ground_truth_cells"],
            v["n_token_eligible_cells"], v["n_missing_cells"],
            v["min_class_support_for_admission"], v["n_classes_at_support"],
            v["class_margins"], "OK" if v["passes"] else "SHORTFALL",
        )
    if failed and admission_applicable and not args.allow_admission_shortfall:
        detail = "\n".join(
            f"  {v['pseudo_slide_id']}: " + "; ".join(v["failures"]) for v in failed
        )
        raise RuntimeError(
            f"A-2 would not admit the following pseudo-slide(s):\n{detail}\n"
            "Pass --allow_admission_shortfall to write the tree anyway."
        )
    if failed and not admission_applicable:
        logger.warning(
            "Admission check not applicable on a --max_tiles run; %d pseudo-slide(s) fall short.",
            len(failed),
        )

    overlays = write_overlays(
        raw_dir, tiles_keep, nuclei_out, recipe, args.overlap_policy,
        out_root / "overlays", args.overlay_tiles, show_progress=not args.no_progress,
    )

    # --- manifest --------------------------------------------------------------------------
    per_tissue_class_before = (
        pd.crosstab(nuclei_out["tissue"], nuclei_out["coarse_v2_class"]).to_dict("index")
    )
    eligible_only = finalized[~(finalized["lost_at_flatten"] | finalized["lost_at_membership"])]
    per_tissue_class_after = (
        pd.crosstab(eligible_only["tissue"], eligible_only["coarse_v2_class"]).to_dict("index")
    )
    prostate_after = per_tissue_class_after.get("Prostate", {})
    manifest = {
        "prepare_schema_version": PREPARE_SCHEMA_VERSION,
        "dataset": "PanNuke",
        "is_full_run": bool(args.max_tiles is None and not args.tissues),
        "geometry_recipe": geometry,
        "membership": {
            "group_name": group["name"],
            "models": group["models"],
            "resize_size": group["resize_size"],
            "crop_size": group["crop_size"],
            "tokens_per_dim": group["tokens_per_dim"],
            "tokens_per_patch": group["tokens_per_patch"],
            "vit_token_px": group["vit_token_px"],
            "contract_source": str(upstream.eval_data_builder_path()),
            "contract_imported_not_copied": True,
        },
        "cell_id_scheme": {
            "formula": "tile_index * stride + instance_ordinal",
            "stride": DEFAULT_TILE_ID_STRIDE,
            "tile_index_scope": "global across all folds (scan order of _2_characterize_pannuke.py)",
            "max_cell_id_int": int(nuclei_out["cell_id_int"].max()),
            "int32_headroom_factor": int((2 ** 31) // max(int(nuclei_out["cell_id_int"].max()), 1)),
            "str_int_round_trip_verified": True,
        },
        "coords_layout": {
            "layout": "synthetic_1d_strip",
            "stride_px": int(recipe.stored_px),
            "note": "PanNuke tiles are non-contiguous visual fields; these coordinates are "
                    "declared synthetic and carry no spatial meaning.",
        },
        "overlap": {
            "policy": args.overlap_policy,
            "n_overlap_px": int(built["stats"]["n_overlap_px"]),
            "n_contested_px": int(built["stats"]["n_contested_px"]),
            "n_tiles_with_overlap": int(built["stats"]["n_tiles_with_overlap"]),
        },
        "cohort": {
            "retained_tissues": retained_tissues,
            "n_tiles_scanned": int(built["stats"]["n_tiles_scanned"]),
            "n_tiles_in_retained_tissues": int(len(tiles[tiles["tissue"].isin(set(retained_tissues))])),
            "n_tiles_zero_scorable": n_zero_scorable,
            "keep_empty_tiles": bool(args.keep_empty_tiles),
            "n_tiles_written": int(built["stats"]["n_tiles_written"]),
            "n_scorable_nuclei": int(len(nuclei_out)),
            "n_pseudo_slides": int(tiles_keep["pseudo_slide_id"].nunique()),
            "border_touching_fraction": float(nuclei_out["touches_border"].mean()),
            "per_tissue_class_before_membership": per_tissue_class_before,
            "per_tissue_class_after_membership": per_tissue_class_after,
            "prostate_immune_margin_cells": (
                int(prostate_after.get("Immune", 0)) - int(verdicts[0]["min_class_support_for_admission"])
                if prostate_after else None
            ),
        },
        "nucleus_loss": {
            "n_lost_total": n_lost,
            "lost_fraction": float(lost_fraction),
            "max_lost_cell_fraction": float(args.max_lost_cell_fraction),
            "n_lost_at_flatten": int(finalized["lost_at_flatten"].sum()),
            "n_lost_at_membership": int(finalized["lost_at_membership"].sum()),
            "lost_by_coarse_v2_class": {str(k): int(v) for k, v in lost_by_class.items()},
            "max_lost_area_px_src": int(lost_area.max()) if lost_area.size else None,
            "median_lost_area_px_src": float(np.median(lost_area)) if lost_area.size else None,
            "lost_by_source_area_bucket": lost_area_buckets,
        },
        "nuc_px_ratio": {
            "definition": "nuc_px_membership / (area_px_src * model_area_scale)",
            "median": ratio_median,
            "p05": float(np.percentile(ratios, 5)) if ratios.size else None,
            "p95": float(np.percentile(ratios, 95)) if ratios.size else None,
            "band": list(NUC_PX_RATIO_BAND),
            "note": "The image and the mask use different resampling filters, so this is a band "
                    "on the median rather than a per-cell equality.",
        },
        "admission": {
            "applicable": bool(admission_applicable),
            "allow_admission_shortfall": bool(args.allow_admission_shortfall),
            "n_pseudo_slides_short": len(failed),
            "verdicts": verdicts,
        },
        "overlays": overlays,
        "source": _source_provenance(data_root),
        "environment": {
            "required_conda_env": upstream.REQUIRED_CONDA_ENV,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "git_head": git_head(),
        "seed": int(args.seed),
        "argv": sys.argv,
    }
    write_json(out_root / "prepare_manifest.json", manifest)
    logger.info("Wrote %s", out_root / "prepare_manifest.json")
    logger.info(
        "DONE: %d pseudo-slides, %d patches, %d scorable nuclei -> %s",
        tiles_keep["pseudo_slide_id"].nunique(), len(tiles_keep), len(nuclei_out), out_root,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Build the PanNuke patches and cell-token-membership tables for one geometry recipe "
            "(runs in the `process_slide` env)."
        )
    )
    parser.add_argument(
        "--geometry_recipe", type=str, required=True, choices=list(GEOMETRY_RECIPE_NAMES),
        help=(
            "Geometry recipe: 'native224' stores the raw 256 px tile (resized to 224 by each backbone), "
            "'scale224' resizes the tile to --scale_tissue_px and pads it to 224."
        ),
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="PanNuke data root with raw/ and characterization/ (default: eval/eval_datasets/pannuke_eval_data).",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help=(
            "Output folder (default: <data_dir>/prepared/eval_st_pannuke_<recipe>/)."
        ),
    )
    parser.add_argument(
        "--tissues", type=str, nargs="*", default=None,
        help="Restrict to these PanNuke tissues (default: all retained tissues).",
    )
    parser.add_argument(
        "--max_tiles", type=int, default=None,
        help=(
            "Stop after this many tiles per fold (the admission check is then not applied)."
        ),
    )
    parser.add_argument(
        "--overlay_tiles", type=int, default=12,
        help="Number of image/mask overlay PNGs.",
    )
    parser.add_argument(
        "--overlap_policy", type=str, default=OVERLAP_LAST_WRITE_WINS, choices=list(OVERLAP_POLICIES),
        help=(
            "Pixels claimed by several nuclei: 'last_write_wins' (as in _2_characterize_pannuke.py) "
            "or 'drop_contested' (removed from all of them)."
        ),
    )
    parser.add_argument(
        "--scale_tissue_px", type=int, default=None,
        help=(
            f"scale224 only: px of the 224 canvas covered by the tile (default {DEFAULT_SCALE_TISSUE_PX}, "
            "token-aligned: no token mixes tissue and padding)."
        ),
    )
    parser.add_argument(
        "--scale_image_interpolation", type=str, default=None,
        choices=["nearest", "linear", "cubic", "area"],
        help=(
            f"scale224 only: image resampling filter (default '{DEFAULT_SCALE_IMAGE_INTERPOLATION}'); "
            "the mask always uses nearest."
        ),
    )
    parser.add_argument(
        "--pad_value", type=int, default=None,
        help=(
            f"scale224 only: uint8 padding value (default {DEFAULT_PAD_VALUE} = white)."
        ),
    )
    parser.add_argument(
        "--keep_empty_tiles", action="store_true",
        help=(
            "Keep the tiles with no scorable nucleus (dropped by default)."
        ),
    )
    parser.add_argument(
        "--max_lost_cell_fraction", type=float, default=DEFAULT_MAX_LOST_CELL_FRACTION,
        help=(
            "Raise if more than this fraction of nuclei have no token support."
        ),
    )
    parser.add_argument(
        "--allow_admission_shortfall", action="store_true",
        help=(
            "Write the tree even if a pseudo-slide is below the A-2 admission floors "
            "(ADMISSION_* in pannuke_common.py)."
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Rebuild into a non-empty output directory.",
    )
    parser.add_argument(
        "--no_progress", action="store_true",
        help=(
            "Disable the tqdm progress bars."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed recorded in the manifest (nothing in the script is random).",
    )
    sys.exit(main(parser.parse_args()))
