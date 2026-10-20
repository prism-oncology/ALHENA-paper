#!/usr/bin/env python
"""
Characterise the PanNuke dataset downloaded by _1_download_pannuke.py (vocabularies, counts, instance statistics,
resolution estimate, overlays) and check the counts against the published PanNuke figures.
Runs in the `histost_eval_a2` env.

Saved in <ROOT_HistoST> / eval / eval_datasets / pannuke_eval_data / characterization /
    - pannuke_characterization.json: vocabularies, counts, distributions, degeneracies
    - per_nucleus_stats.parquet: one row per nucleus
    - per_tile_stats.parquet: one row per tile
    - overlays/<tissue>_<tile_id>.png: sample tiles with their nuclei coloured by class
    - characterize.log
Markdown summary saved in <repo> / access_reports / A2_pannuke_characterization.md
"""

import argparse
import io
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Project imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src_A2.common import PROJECT_ROOT, configure_logger, ensure_dir, write_dataframe, write_json, write_text
from src_A2.common import src_import
from pannuke.pannuke_common import (
    COARSE_V2_SCORABLE_CLASSES,
    DEFAULT_TILE_ID_STRIDE,
    PANNUKE_CLASS_NAMES,
    PANNUKE_CLASS_TO_COARSE_V2,
    PANNUKE_LICENSE,
    PANNUKE_NOMINAL_MAGNIFICATION,
    PANNUKE_NOMINAL_MPP_UM_PER_PX,
    PANNUKE_RESOLUTION_CAVEAT,
    PANNUKE_TILE_PX,
    PANNUKE_TISSUE_TO_HISTOST,
    PANNUKE_TISSUES,
    PANNUKE_TISSUES_WITHOUT_REFERENCE,
    PANNUKE_UNSCORABLE_CLASSES,
    PUBLISHED_CLASS_COUNTS,
    PUBLISHED_FOLD_SIZES,
    PUBLISHED_N_NUCLEI,
    PUBLISHED_N_TILES,
    STHELAR_MPP_UM_PER_PX,
    assert_cell_id_budget,
    assert_known_class_indices,
    assert_known_tissues,
    bimodality_coefficient,
    estimate_mpp_from_area_ratio,
    instance_map_from_masks,
    is_scorable_class,
    map_pannuke_class_to_v2,
    normalize_tissue_name,
    nucleus_area_px,
    per_cell_nucleus_area_px,
    touches_border,
)

get_full_path = src_import("get_full_path")

logger = logging.getLogger(__name__)

CHARACTERIZATION_SCHEMA_VERSION = 1

DEFAULT_STHELAR_TRAINING_DATASET_ID = "ds_ct1_20x_224_o112_1"
DEFAULT_MEMBERSHIP_MODELGROUP = "hoptimus0_hoptimus1_uni2h_virchow2"


def pannuke_root() -> Path:
    return get_full_path("eval", "eval_datasets", "pannuke_eval_data")


# ---------------------------------------------------------------------------------------------
# Parquet reading -- HF Arrow encoding, without the `datasets` library
# ---------------------------------------------------------------------------------------------

def read_hf_feature_names(parquet_path: Path) -> Dict[str, List[str]]:
    """Read the ClassLabel names (tissue, categories) from the HF metadata of the parquet schema (raises if absent)."""
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(str(parquet_path)).schema_arrow
    meta = schema.metadata or {}
    raw = meta.get(b"huggingface")
    if raw is None:
        raise ValueError(
            f"{parquet_path} has no b'huggingface' schema metadata: the ClassLabel names of `tissue` "
            "and `categories` cannot be recovered."
        )
    info = json.loads(raw.decode("utf-8"))
    features = (info.get("info") or {}).get("features") or info.get("features") or {}

    def _names(node: Any, field: str) -> List[str]:
        if node is None:
            raise ValueError(f"Feature '{field}' missing from the huggingface metadata of {parquet_path}.")
        # `categories` is a Sequence(ClassLabel(...)), so the names sit one level down.
        if isinstance(node, dict) and "names" in node:
            return [str(n) for n in node["names"]]
        for key in ("feature", "_type"):
            if isinstance(node, dict) and isinstance(node.get(key), dict):
                inner = node[key]
                if "names" in inner:
                    return [str(n) for n in inner["names"]]
        if isinstance(node, list) and node and isinstance(node[0], dict) and "names" in node[0]:
            return [str(n) for n in node[0]["names"]]
        raise ValueError(
            f"Could not locate ClassLabel names for '{field}' in {parquet_path}; got node {node!r}."
        )

    return {
        "tissue": _names(features.get("tissue"), "tissue"),
        "categories": _names(features.get("categories"), "categories"),
    }


def _image_bytes(cell: Any) -> bytes:
    """Pull the encoded bytes out of an HF Image cell (a {'bytes':..., 'path':...} struct)."""
    if isinstance(cell, dict):
        data = cell.get("bytes")
        if data is None:
            raise ValueError(
                "HF Image cell has no inline 'bytes' (only a path to an external file)."
            )
        return data
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise ValueError(f"Unsupported image cell type {type(cell)!r}.")


def _open_image(data: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(data))


def iter_tiles(parquet_path: Path, max_tiles: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    """Yield the raw cells of each tile of one fold's parquet, one row group at a time."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet_path))
    emitted = 0
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=["image", "instances", "categories", "tissue"])
        batch = table.to_pylist()
        for row_in_group, row in enumerate(batch):
            yield {
                "source_file": parquet_path.name,
                "row_group": rg,
                "row_in_group": row_in_group,
                "image": row["image"],
                "instances": row["instances"],
                "categories": row["categories"],
                "tissue": row["tissue"],
            }
            emitted += 1
            if max_tiles is not None and emitted >= max_tiles:
                return


def read_specific_tiles(
    parquet_path: Path, wanted: Dict[int, List[int]]
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Read only the given {row_group: [row_in_group, ...]} rows, keyed by (row_group, row_in_group)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet_path))
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for rg, rows in sorted(wanted.items()):
        table = pf.read_row_group(rg, columns=["image", "instances", "categories", "tissue"])
        batch = table.to_pylist()
        for row_in_group in rows:
            if not (0 <= row_in_group < len(batch)):
                raise IndexError(
                    f"{parquet_path.name}: row {row_in_group} is outside row group {rg} ({len(batch)} rows)."
                )
            row = batch[row_in_group]
            out[(rg, row_in_group)] = {
                "source_file": parquet_path.name,
                "row_group": rg,
                "row_in_group": row_in_group,
                "image": row["image"],
                "instances": row["instances"],
                "categories": row["categories"],
                "tissue": row["tissue"],
            }
    return out


# ---------------------------------------------------------------------------------------------
# Per-tile characterisation
# ---------------------------------------------------------------------------------------------

def characterize_tile(
    tile: Dict[str, Any],
    tile_id: str,
    tile_index: int,
    fold: str,
    tissue_names: List[str],
    class_names: List[str],
    keep_arrays: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Measure one tile. Returns (tile_row, nucleus_rows, arrays_or_None).
    With keep_arrays, arrays holds the decoded RGB and the instance map (used for the overlays).
    """
    img = _open_image(_image_bytes(tile["image"]))
    img_w, img_h = img.size
    img_mode = img.mode

    tissue_code = int(tile["tissue"])
    if not (0 <= tissue_code < len(tissue_names)):
        raise ValueError(
            f"{tile_id}: tissue code {tissue_code} is outside the parquet's own ClassLabel list "
            f"of {len(tissue_names)} names."
        )
    tissue = normalize_tissue_name(tissue_names[tissue_code])

    categories = list(tile["categories"] or [])
    instances = list(tile["instances"] or [])
    if len(categories) != len(instances):
        raise ValueError(
            f"{tile_id}: {len(instances)} instance masks but {len(categories)} categories."
        )

    assert_known_class_indices(
        [c for c in categories if 0 <= int(c) < len(class_names)] or [0],
        context=f"{tile_id} (raw codes)",
    )
    for code in categories:
        if not (0 <= int(code) < len(class_names)):
            raise ValueError(
                f"{tile_id}: category code {code} is outside the parquet's own ClassLabel list of "
                f"{len(class_names)} names."
            )

    masks: List[np.ndarray] = []
    nucleus_rows: List[Dict[str, Any]] = []
    n_zero_area = 0
    n_border = 0
    for ordinal, (mask_cell, code) in enumerate(zip(instances, categories), start=1):
        m = np.asarray(_open_image(_image_bytes(mask_cell)).convert("1")).astype(bool)
        if m.shape != (img_h, img_w):
            raise ValueError(
                f"{tile_id}: instance {ordinal} has mask shape {m.shape} but the image is {(img_h, img_w)}."
            )
        masks.append(m)
        class_name = str(class_names[int(code)])
        area = nucleus_area_px(m)
        border = touches_border(m)
        if area == 0:
            n_zero_area += 1
        if border:
            n_border += 1
        nucleus_rows.append(
            {
                "fold": fold,
                "tile_id": tile_id,
                "tile_index": tile_index,
                "tissue": tissue,
                "instance_ordinal": ordinal,
                "pannuke_class": class_name,
                "scorable": is_scorable_class(class_name),
                "coarse_v2_class": (
                    map_pannuke_class_to_v2(class_name) if is_scorable_class(class_name) else ""
                ),
                "area_px": area,
                "touches_border": border,
            }
        )

    instance_map, n_overlap_px = instance_map_from_masks(masks, (img_h, img_w))

    tile_row = {
        "fold": fold,
        "tile_id": tile_id,
        "tile_index": tile_index,
        "tissue": tissue,
        # Row address in the parquet (used to re-read the overlay tiles)
        "source_file": str(tile.get("source_file", "")),
        "row_group": int(tile["row_group"]),
        "row_in_group": int(tile["row_in_group"]),
        "image_height": img_h,
        "image_width": img_w,
        "image_mode": img_mode,
        "n_instances": len(instances),
        "n_zero_area_instances": n_zero_area,
        "n_border_instances": n_border,
        "n_overlap_px": n_overlap_px,
        "n_scorable_instances": int(sum(1 for r in nucleus_rows if r["scorable"])),
    }

    arrays = None
    if keep_arrays:
        arrays = {
            "rgb": np.asarray(img.convert("RGB")),
            "instance_map": instance_map,
            "classes": [r["pannuke_class"] for r in nucleus_rows],
        }
    return tile_row, nucleus_rows, arrays


# ---------------------------------------------------------------------------------------------
# The STHELAR calibrator for the empirical resolution estimate
# ---------------------------------------------------------------------------------------------

def sthelar_median_nucleus_area_px(
    training_dataset_id: str, modelgroup: str, max_slides: int
) -> Dict[str, Any]:
    """
    Median nucleus area (px^2) in the STHELAR cell-token-membership tables (per cell: nuc_px summed within a
    patch, max over patches). Returns a status dict ("unavailable" with a reason if the tables are missing).
    """
    root = get_full_path(
        "eval", "eval_datasets", "st_eval_data", f"eval_st_{training_dataset_id}",
        f"cell_token_membership_{modelgroup}",
    )
    if not Path(root).is_dir():
        return {
            "status": "unavailable",
            "reason": f"cell-token-membership folder not found: {root}",
            "path": str(root),
        }
    files = sorted(p for p in Path(root).glob("*.parquet") if not p.name.startswith("."))
    if not files:
        return {
            "status": "unavailable",
            "reason": f"no per-slide parquet files under {root}",
            "path": str(root),
        }

    used = files[: max(1, int(max_slides))]
    medians: List[float] = []
    per_slide: List[Dict[str, Any]] = []
    all_areas: List[np.ndarray] = []
    for path in used:
        df = pd.read_parquet(path, columns=["cell_id_int", "patch_id", "nuc_px"])
        if df.empty:
            continue
        areas = per_cell_nucleus_area_px(df)
        if areas.size == 0:
            continue
        all_areas.append(areas)
        medians.append(float(np.median(areas)))
        per_slide.append(
            {"slide": path.stem, "n_cells": int(areas.size), "median_area_px": float(np.median(areas))}
        )

    if not all_areas:
        return {
            "status": "unavailable",
            "reason": f"every sampled membership parquet under {root} was empty",
            "path": str(root),
        }

    pooled = np.concatenate(all_areas)
    return {
        "status": "ok",
        "path": str(root),
        "training_dataset_id": training_dataset_id,
        "modelgroup": modelgroup,
        "n_slides_used": len(per_slide),
        "n_slides_available": len(files),
        "n_cells": int(pooled.size),
        "median_area_px_pooled": float(np.median(pooled)),
        "median_of_slide_medians_px": float(np.median(medians)),
        "mpp_um_per_px": STHELAR_MPP_UM_PER_PX,
        "per_slide": per_slide,
    }


# ---------------------------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------------------------

def _class_colors() -> Dict[str, Tuple[int, int, int]]:
    return {
        "Neoplastic": (228, 26, 28),
        "Inflammatory": (55, 126, 184),
        "Connective": (77, 175, 74),
        "Dead": (152, 78, 163),
        "Epithelial": (255, 127, 0),
    }


def write_overlay(arrays: Dict[str, Any], tile_row: Dict[str, Any], out_path: Path) -> None:
    """Save one tile beside a copy with its nuclei coloured by class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    rgb = arrays["rgb"]
    inst = arrays["instance_map"]
    classes = arrays["classes"]
    colors = _class_colors()

    overlay = rgb.astype(float).copy()
    present: List[str] = []
    for ordinal in range(1, int(inst.max()) + 1 if inst.size else 1):
        sel = inst == ordinal
        if not np.any(sel):
            continue
        cname = classes[ordinal - 1]
        present.append(cname)
        color = np.asarray(colors.get(cname, (128, 128, 128)), dtype=float)
        overlay[sel] = 0.45 * overlay[sel] + 0.55 * color

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2))
    axes[0].imshow(rgb.astype(np.uint8))
    axes[0].set_title(
        f"{tile_row['tissue']} | {tile_row['fold']} | {tile_row['tile_id']}", fontsize=8
    )
    axes[1].imshow(overlay.astype(np.uint8))
    axes[1].set_title(
        f"{tile_row['n_instances']} nuclei "
        f"({tile_row['n_border_instances']} touching a border)", fontsize=8,
    )
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [
        mpatches.Patch(color=np.asarray(colors[c]) / 255.0, label=c)
        for c in PANNUKE_CLASS_NAMES
        if c in set(present)
    ]
    if handles:
        axes[1].legend(handles=handles, loc="upper right", fontsize=6, framealpha=0.85)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def select_overlay_tiles(tile_rows: List[Dict[str, Any]], n: int) -> List[int]:
    """Deterministic sample of tile indices, spread over tissues, most nucleus-rich tiles first."""
    if n <= 0 or not tile_rows:
        return []
    by_tissue: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in tile_rows:
        by_tissue[row["tissue"]].append(row)
    for rows in by_tissue.values():
        rows.sort(key=lambda r: (-int(r["n_instances"]), int(r["tile_index"])))

    chosen: List[int] = []
    tissues = sorted(by_tissue)
    rank = 0
    while len(chosen) < n:
        added = False
        for tissue in tissues:
            rows = by_tissue[tissue]
            if rank < len(rows):
                chosen.append(int(rows[rank]["tile_index"]))
                added = True
                if len(chosen) >= n:
                    break
        if not added:
            break
        rank += 1
    return chosen


# ---------------------------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------------------------

def _describe(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {k: float("nan") for k in ("min", "p25", "median", "mean", "p75", "p95", "max")}
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def build_characterization(
    tiles: pd.DataFrame,
    nuclei: pd.DataFrame,
    observed_vocab: Dict[str, List[str]],
    manifest: Optional[Dict[str, Any]],
    sthelar: Dict[str, Any],
    is_full_run: bool,
    warnings: List[str],
) -> Dict[str, Any]:
    """Assemble the machine-readable characterisation."""
    retained_tissues = sorted(set(PANNUKE_TISSUE_TO_HISTOST) & set(tiles["tissue"].unique()))
    scorable = nuclei[nuclei["scorable"]]
    retained_nuclei = scorable[scorable["tissue"].isin(retained_tissues)]

    # --- counts ---------------------------------------------------------------------------
    per_class_counts = Counter(nuclei["pannuke_class"].tolist())
    gate = {
        "applicable": bool(is_full_run),
        "reason_if_not_applicable": (
            "" if is_full_run else "partial run (--folds and/or --max_tiles restricted the cohort)"
        ),
        "expected": {
            "n_tiles": PUBLISHED_N_TILES,
            "n_nuclei": PUBLISHED_N_NUCLEI,
            "class_counts": dict(PUBLISHED_CLASS_COUNTS),
            "fold_sizes": dict(PUBLISHED_FOLD_SIZES),
        },
        "observed": {
            "n_tiles": int(len(tiles)),
            "n_nuclei": int(len(nuclei)),
            "class_counts": {k: int(per_class_counts.get(k, 0)) for k in PANNUKE_CLASS_NAMES},
            "fold_sizes": {k: int(v) for k, v in tiles["fold"].value_counts().sort_index().items()},
        },
    }
    if is_full_run:
        gate["passed"] = bool(
            gate["observed"]["n_tiles"] == PUBLISHED_N_TILES
            and gate["observed"]["n_nuclei"] == PUBLISHED_N_NUCLEI
            and gate["observed"]["class_counts"] == dict(PUBLISHED_CLASS_COUNTS)
            and gate["observed"]["fold_sizes"] == dict(PUBLISHED_FOLD_SIZES)
        )

    # --- instance statistics ---------------------------------------------------------------
    areas_all = nuclei["area_px"].to_numpy()
    per_class_area = {
        cname: _describe(nuclei.loc[nuclei["pannuke_class"] == cname, "area_px"].to_numpy())
        for cname in PANNUKE_CLASS_NAMES
    }

    # --- resolution -----------------------------------------------------------------------
    resolution: Dict[str, Any] = {
        "nominal_magnification": PANNUKE_NOMINAL_MAGNIFICATION,
        "nominal_mpp_um_per_px": PANNUKE_NOMINAL_MPP_UM_PER_PX,
        "caveat": PANNUKE_RESOLUTION_CAVEAT,
        "per_tile_resolution_metadata_available": False,
        "per_tile_resolution_metadata_note": (
            "The parquet carries only image / instances / categories / tissue. There is no "
            "per-tile native-magnification field, so the upsampled-from-20x subset cannot be "
            "identified or filtered."
        ),
        "empirical_estimate": {"status": "not_computed"},
    }
    if sthelar.get("status") == "ok" and areas_all.size > 0:
        median_pannuke = float(np.median(areas_all))
        median_sthelar = float(sthelar["median_area_px_pooled"])
        resolution["empirical_estimate"] = {
            "status": "ok",
            "method": (
                "mpp_pannuke = mpp_sthelar * sqrt(median_area_sthelar / median_area_pannuke); "
                "nucleus area scales as (um/px)^-2 at fixed physical size"
            ),
            "median_nucleus_area_px_pannuke": median_pannuke,
            "median_nucleus_area_px_sthelar": median_sthelar,
            "sthelar_mpp_um_per_px": STHELAR_MPP_UM_PER_PX,
            "estimated_mpp_um_per_px": estimate_mpp_from_area_ratio(median_pannuke, median_sthelar),
            "implied_magnification_ratio_vs_sthelar": float(
                np.sqrt(median_pannuke / median_sthelar)
            ),
            "bimodality_coefficient_log_area": bimodality_coefficient(
                np.log(areas_all[areas_all > 0])
            ),
            "bimodality_note": (
                "Sarle's coefficient on log nucleus area. Above ~0.555 suggests a mixture (e.g. "
                "of 20x and 40x scans); nucleus size also varies by tissue and class."
            ),
            "caveat": (
                "An estimate, not a measurement: PanNuke is enriched in enlarged neoplastic "
                "nuclei and its tissue composition differs from STHELAR. 20x and 40x differ by "
                "4x in nucleus area."
            ),
        }
    else:
        resolution["empirical_estimate"] = {
            "status": "unavailable",
            "reason": sthelar.get("reason", "STHELAR calibration not available"),
        }

    # --- cell-id budget -------------------------------------------------------------------
    max_inst = int(tiles["n_instances"].max()) if len(tiles) else 0
    id_budget = assert_cell_id_budget(
        n_tiles=int(len(tiles)), max_instances_per_tile=max(max_inst, 1),
        stride=DEFAULT_TILE_ID_STRIDE,
    )

    per_tissue = (
        tiles.groupby("tissue")
        .agg(n_tiles=("tile_id", "size"), n_instances=("n_instances", "sum"))
        .reset_index()
        .sort_values("tissue")
    )
    per_tissue_class = (
        nuclei.groupby(["tissue", "pannuke_class"]).size().unstack(fill_value=0).sort_index()
    )

    coarse_counts = Counter(retained_nuclei["coarse_v2_class"].tolist())
    coarse_total = int(sum(coarse_counts.values()))

    return {
        "characterization_schema_version": CHARACTERIZATION_SCHEMA_VERSION,
        "dataset": "PanNuke",
        "source_manifest": {
            "present": manifest is not None,
            "resolved_revision": (manifest or {}).get("resolved_revision"),
            "repo_id": (manifest or {}).get("repo_id"),
            "license": (manifest or {}).get("license", PANNUKE_LICENSE),
        },
        "is_full_run": bool(is_full_run),
        "warnings": list(warnings),

        "geometry": {
            "declared_tile_px": PANNUKE_TILE_PX,
            "observed_image_shapes": sorted(
                {f"{int(h)}x{int(w)}" for h, w in zip(tiles["image_height"], tiles["image_width"])}
            ),
            "observed_image_modes": sorted(set(tiles["image_mode"].tolist())),
            "resolution": resolution,
        },
        "mask_encoding": {
            "representation": (
                "one binary mask per nucleus (HF `instances`, PIL mode '1') with a parallel "
                "per-nucleus class code (HF `categories`)"
            ),
            "instance_id_scope": (
                "no ids in the source; the 1-based ordinal within the tile is minted here, so "
                "ids are unique per tile by construction"
            ),
            "background_encoding": "0 in the flattened instance map; the source has no background channel",
            "n_tiles_with_overlapping_instance_px": int((tiles["n_overlap_px"] > 0).sum()),
            "total_overlapping_instance_px": int(tiles["n_overlap_px"].sum()),
            "overlap_note": (
                "Overlap is counted, not treated as an error: per-nucleus areas are computed from "
                "the individual masks. PanNuke masks are described as non-overlapping."
            ),
        },

        "class_vocabulary": {
            "declared": list(PANNUKE_CLASS_NAMES),
            "observed_in_data": list(observed_vocab["categories"]),
            "observed_present": sorted(set(nuclei["pannuke_class"].tolist())),
            "unscorable_dropped_two_sided": sorted(PANNUKE_UNSCORABLE_CLASSES),
            "map_to_coarse_v2": dict(PANNUKE_CLASS_TO_COARSE_V2),
        },
        "tissue_vocabulary": {
            "declared": list(PANNUKE_TISSUES),
            "observed_in_data_raw": list(observed_vocab["tissue"]),
            "observed_normalized": sorted(set(tiles["tissue"].tolist())),
            "map_to_tissue_histost": dict(PANNUKE_TISSUE_TO_HISTOST),
            "without_reference_excluded": list(PANNUKE_TISSUES_WITHOUT_REFERENCE),
            "spelling_note": (
                "HuggingFace ships 'Adrenal Gland' / 'Bile Duct' / 'Head & Neck' where the in-repo "
                "CellViT convention writes 'Adrenal_gland' / 'Bile-duct' / 'HeadNeck'. "
                "normalize_tissue_name() folds both onto the CellViT tokens."
            ),
        },

        "counts": {
            "whole_dataset": {
                "n_tiles": int(len(tiles)),
                "n_nuclei": int(len(nuclei)),
                "n_nuclei_scorable": int(len(scorable)),
                "per_class": {k: int(per_class_counts.get(k, 0)) for k in PANNUKE_CLASS_NAMES},
                "per_fold": {k: int(v) for k, v in tiles["fold"].value_counts().sort_index().items()},
            },
            "retained_tissues_only": {
                "tissues": retained_tissues,
                "n_tissues": len(retained_tissues),
                "n_tiles": int(tiles["tissue"].isin(retained_tissues).sum()),
                "n_nuclei": int(nuclei["tissue"].isin(retained_tissues).sum()),
                "n_nuclei_scorable": int(len(retained_nuclei)),
                "per_coarse_v2_class": {k: int(coarse_counts.get(k, 0)) for k in COARSE_V2_SCORABLE_CLASSES},
                "per_coarse_v2_share": {
                    k: (float(coarse_counts.get(k, 0)) / coarse_total if coarse_total else float("nan"))
                    for k in COARSE_V2_SCORABLE_CLASSES
                },
                "majority_class_floor_accuracy": (
                    float(max(coarse_counts.values()) / coarse_total) if coarse_total else float("nan")
                ),
            },
            "per_tissue": per_tissue.to_dict(orient="records"),
            "per_tissue_class": per_tissue_class.to_dict(orient="index"),
            "acceptance_gate": gate,
        },

        "instance_statistics": {
            "instances_per_tile": _describe(tiles["n_instances"].to_numpy()),
            "nucleus_area_px_all": _describe(areas_all),
            "nucleus_area_px_per_class": per_class_area,
            "border_touching_fraction": (
                float(nuclei["touches_border"].mean()) if len(nuclei) else float("nan")
            ),
            "border_touching_count": int(nuclei["touches_border"].sum()),
            "per_class_area_note": (
                "The per-class distribution shows whether a downsample would preferentially "
                "lose small, mostly lymphoid nuclei."
            ),
        },

        "degeneracies": {
            "n_tiles_zero_nuclei": int((tiles["n_instances"] == 0).sum()),
            "n_zero_area_instances": int(tiles["n_zero_area_instances"].sum()),
            "n_tiles_with_overlap": int((tiles["n_overlap_px"] > 0).sum()),
            "distinct_image_shapes": int(
                len({(int(h), int(w)) for h, w in zip(tiles["image_height"], tiles["image_width"])})
            ),
            "note": (
                "Measured, not filtered. Vocabulary violations, shape mismatches and duplicated "
                "class assignments raise during the scan instead of appearing here."
            ),
        },

        # Raw size (from manifest.json) and cell-id budget
        "footprint_projection": {
            "raw_bytes": int((manifest or {}).get("total_size_bytes", 0)),
            "id_budget": id_budget,
            "note": (
                "Raw download size and cell-id budget; the prepared artifacts "
                "(_4_prepare_patches.py) are not built here."
            ),
        },

        "sthelar_calibration": sthelar,
    }


def render_markdown(char: Dict[str, Any]) -> str:
    """Markdown summary of the characterisation."""
    c = char["counts"]
    g = c["acceptance_gate"]
    res = char["geometry"]["resolution"]
    emp = res.get("empirical_estimate", {})
    lines: List[str] = []
    a = lines.append

    a("# Task A-2 on PanNuke — characterisation")
    a("")
    a(f"Source revision: `{char['source_manifest'].get('resolved_revision')}` "
      f"(`{char['source_manifest'].get('repo_id')}`), license **{char['source_manifest'].get('license')}**.")
    a(f"Full run: **{char['is_full_run']}**. Warnings: **{len(char['warnings'])}**.")
    a("")

    a("## Acceptance gate (published counts)")
    a("")
    if not g["applicable"]:
        a(f"Not applicable — {g['reason_if_not_applicable']}.")
    else:
        a(f"**{'PASSED' if g.get('passed') else 'FAILED'}**")
    a("")
    a("| quantity | expected | observed |")
    a("|---|---|---|")
    a(f"| tiles | {g['expected']['n_tiles']} | {g['observed']['n_tiles']} |")
    a(f"| nuclei | {g['expected']['n_nuclei']} | {g['observed']['n_nuclei']} |")
    for cname in PANNUKE_CLASS_NAMES:
        a(f"| {cname} | {g['expected']['class_counts'][cname]} | {g['observed']['class_counts'][cname]} |")
    a("")

    a("## Resolution")
    a("")
    a(f"Nominal: **{res['nominal_magnification']}**, {res['nominal_mpp_um_per_px']} µm/px.")
    a("")
    a(f"> {res['caveat']}")
    a("")
    a(f"Per-tile resolution metadata available: **{res['per_tile_resolution_metadata_available']}**.")
    a("")
    if emp.get("status") == "ok":
        a(f"Empirical estimate against STHELAR ({emp['sthelar_mpp_um_per_px']} µm/px): "
          f"**{emp['estimated_mpp_um_per_px']:.4f} µm/px**")
        a("")
        a(f"- median nucleus area — PanNuke {emp['median_nucleus_area_px_pannuke']:.1f} px² · "
          f"STHELAR {emp['median_nucleus_area_px_sthelar']:.1f} px²")
        a(f"- bimodality coefficient (log area): **{emp['bimodality_coefficient_log_area']:.4f}** "
          "(> ~0.555 suggests a mixture)")
        a(f"- {emp['caveat']}")
    else:
        a(f"Empirical estimate: **unavailable** — {emp.get('reason')}")
    a("")

    a("## Counts")
    a("")
    w = c["whole_dataset"]
    r = c["retained_tissues_only"]
    a(f"Whole dataset: **{w['n_tiles']}** tiles, **{w['n_nuclei']}** nuclei "
      f"({w['n_nuclei_scorable']} scorable after dropping {sorted(PANNUKE_UNSCORABLE_CLASSES)}).")
    a("")
    a(f"Retained tissues ({r['n_tissues']}): {', '.join(r['tissues'])}")
    a("")
    a(f"Retained-only: **{r['n_tiles']}** tiles, **{r['n_nuclei_scorable']}** scorable nuclei.")
    a("")
    a("| coarse v2 class | n | share |")
    a("|---|---|---|")
    for k in COARSE_V2_SCORABLE_CLASSES:
        a(f"| {k} | {r['per_coarse_v2_class'][k]} | {r['per_coarse_v2_share'][k]:.3f} |")
    a("")
    a(f"Majority-class floor (accuracy): **{r['majority_class_floor_accuracy']:.3f}**.")
    a("")

    a("### Per tissue")
    a("")
    a("| tissue | tiles | nuclei | reference |")
    a("|---|---|---|---|")
    for row in c["per_tissue"]:
        ref = PANNUKE_TISSUE_TO_HISTOST.get(row["tissue"], "— (no exact-name reference)")
        a(f"| {row['tissue']} | {row['n_tiles']} | {row['n_instances']} | {ref} |")
    a("")

    a("## Instance statistics")
    a("")
    ist = char["instance_statistics"]
    a(f"Instances per tile: min {ist['instances_per_tile']['min']:.0f} · "
      f"median {ist['instances_per_tile']['median']:.0f} · max {ist['instances_per_tile']['max']:.0f}")
    a("")
    a(f"Border-touching nuclei: **{ist['border_touching_count']}** "
      f"({ist['border_touching_fraction']:.3f})")
    a("")
    a("| class | min | median | mean | p95 | max |")
    a("|---|---|---|---|---|---|")
    for cname in PANNUKE_CLASS_NAMES:
        d = ist["nucleus_area_px_per_class"][cname]
        a(f"| {cname} | {d['min']:.0f} | {d['median']:.0f} | {d['mean']:.1f} | {d['p95']:.0f} | {d['max']:.0f} |")
    a("")
    a(f"{ist['per_class_area_note']}")
    a("")

    a("## Degeneracies")
    a("")
    for k, v in char["degeneracies"].items():
        if k == "note":
            continue
        a(f"- `{k}`: **{v}**")
    a("")
    a(f"{char['degeneracies']['note']}")
    a("")

    a("## Vocabularies")
    a("")
    a(f"- classes observed in data: {char['class_vocabulary']['observed_in_data']}")
    a(f"- tissues observed in data (raw): {char['tissue_vocabulary']['observed_in_data_raw']}")
    a(f"- {char['tissue_vocabulary']['spelling_note']}")
    a("")

    a("## Cell-id budget")
    a("")
    for k, v in char["footprint_projection"]["id_budget"].items():
        a(f"- `{k}`: {v}")
    a("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

def main(args: argparse.Namespace) -> int:
    src_root = Path(args.data_dir).expanduser().resolve() if args.data_dir else pannuke_root()
    out_root = Path(args.out_dir).expanduser().resolve() if args.out_dir else src_root
    char_dir = out_root / "characterization"
    ensure_dir(char_dir)

    log = configure_logger(char_dir / "characterize.log", name="A2_pannuke_characterize")
    log.info("PanNuke characterisation. data_dir=%s out_dir=%s", src_root, out_root)

    raw_dir = src_root / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"No raw/ folder under {src_root}. Run _1_download_pannuke.py first."
        )
    parquets = sorted(p for p in raw_dir.rglob("*.parquet") if not p.name.startswith("."))
    if not parquets:
        raise FileNotFoundError(f"No parquet files under {raw_dir}.")

    manifest_path = src_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    warnings: List[str] = []
    if manifest is None:
        warnings.append(f"no manifest.json at {manifest_path}; provenance fields will be empty")

    selected = parquets
    if args.folds:
        wanted = set(args.folds)
        selected = [p for p in parquets if any(w in p.stem for w in wanted)]
        if not selected:
            raise FileNotFoundError(
                f"--folds {args.folds} matched none of {[p.stem for p in parquets]}."
            )
    log.info("Scanning %d parquet file(s): %s", len(selected), [p.name for p in selected])

    vocab = read_hf_feature_names(selected[0])
    log.info("ClassLabel names from parquet metadata: %d tissues, %d categories",
             len(vocab["tissue"]), len(vocab["categories"]))
    assert_known_tissues(vocab["tissue"], context=f"{selected[0].name} tissue ClassLabel")
    observed_class_names = [str(c).strip() for c in vocab["categories"]]
    if observed_class_names != list(PANNUKE_CLASS_NAMES):
        raise ValueError(
            "Observed PanNuke class ClassLabel names differ from the declared vocabulary.\n"
            f"  observed: {observed_class_names}\n  declared: {list(PANNUKE_CLASS_NAMES)}\n"
            "Update PANNUKE_CLASS_NAMES in pannuke_common.py if the release changed."
        )

    tile_rows: List[Dict[str, Any]] = []
    nucleus_rows: List[Dict[str, Any]] = []
    tile_index = 0

    for path in selected:
        fold = path.stem.split("-")[0]
        log.info("Fold %s: %s", fold, path.name)
        for tile in iter_tiles(path, max_tiles=args.max_tiles):
            tile_id = f"{fold}_{tile['row_group']:04d}_{tile['row_in_group']:05d}"
            trow, nrows, _ = characterize_tile(
                tile, tile_id=tile_id, tile_index=tile_index, fold=fold,
                tissue_names=vocab["tissue"], class_names=observed_class_names,
                keep_arrays=False,
            )
            tile_rows.append(trow)
            nucleus_rows.extend(nrows)
            tile_index += 1
            if tile_index % 500 == 0:
                log.info("  ... %d tiles, %d nuclei", tile_index, len(nucleus_rows))

    tiles = pd.DataFrame(tile_rows)
    nuclei = pd.DataFrame(nucleus_rows)
    log.info("Scanned %d tiles, %d nuclei.", len(tiles), len(nuclei))

    assert_known_tissues(tiles["tissue"].unique(), context="observed tile tissues")

    is_full_run = bool(args.max_tiles is None and not args.folds and len(selected) == 3)

    sthelar: Dict[str, Any]
    if args.sthelar_calibration:
        sthelar = sthelar_median_nucleus_area_px(
            args.sthelar_training_dataset_id, args.sthelar_modelgroup, args.sthelar_max_slides
        )
        if sthelar.get("status") != "ok":
            log.info("STHELAR calibration unavailable: %s", sthelar.get("reason"))
    else:
        sthelar = {"status": "disabled", "reason": "--no_sthelar_calibration was passed"}

    char = build_characterization(
        tiles=tiles, nuclei=nuclei, observed_vocab=vocab, manifest=manifest,
        sthelar=sthelar, is_full_run=is_full_run, warnings=warnings,
    )

    write_dataframe(nuclei, char_dir / "per_nucleus_stats.parquet")
    write_dataframe(tiles, char_dir / "per_tile_stats.parquet")
    write_json(char_dir / "pannuke_characterization.json", char)

    if args.overlay_tiles > 0 and tile_rows:
        # Second pass: re-read the selected tiles to render the overlays
        chosen = select_overlay_tiles(tile_rows, args.overlay_tiles)
        by_index = {int(r["tile_index"]): r for r in tile_rows}
        by_file: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        for idx in chosen:
            row = by_index[idx]
            by_file[row["source_file"]][int(row["row_group"])].append(int(row["row_in_group"]))

        path_by_name = {p.name: p for p in selected}
        written = 0
        for name, wanted in by_file.items():
            tiles_read = read_specific_tiles(path_by_name[name], dict(wanted))
            for idx in chosen:
                row = by_index[idx]
                if row["source_file"] != name:
                    continue
                tile = tiles_read[(int(row["row_group"]), int(row["row_in_group"]))]
                _, _, arrays = characterize_tile(
                    tile, tile_id=row["tile_id"], tile_index=idx, fold=row["fold"],
                    tissue_names=vocab["tissue"], class_names=observed_class_names,
                    keep_arrays=True,
                )
                write_overlay(
                    arrays, row,
                    char_dir / "overlays" / f"{row['tissue']}_{row['tile_id']}.png",
                )
                written += 1
        log.info("Wrote %d overlay PNG(s) to %s", written, char_dir / "overlays")
        if written != len(chosen):
            raise RuntimeError(
                f"Requested {len(chosen)} overlays but wrote {written}."
            )

    md_path = (
        Path(args.markdown_path).expanduser().resolve()
        if args.markdown_path
        else (PROJECT_ROOT or Path.cwd()) / "access_reports" / "A2_pannuke_characterization.md"
    )
    write_text(md_path, render_markdown(char))
    log.info("Markdown summary -> %s", md_path)

    gate = char["counts"]["acceptance_gate"]
    if gate["applicable"] and not gate.get("passed"):
        raise RuntimeError(
            "Acceptance gate failed: the observed counts do not reproduce the published PanNuke "
            f"figures.\n  expected: {gate['expected']}\n  observed: {gate['observed']}"
        )
    if gate["applicable"]:
        log.info("Acceptance gate passed: %d tiles / %d nuclei reproduce the published figures.",
                 gate["observed"]["n_tiles"], gate["observed"]["n_nuclei"])
    else:
        log.info("Acceptance gate not applicable: %s", gate["reason_if_not_applicable"])
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Characterise the downloaded PanNuke dataset: vocabularies, counts, instance "
            "statistics, degeneracies, an empirical resolution estimate and overlay PNGs."
        )
    )

    parser.add_argument("--data_dir", type=str, default=None,
                        help="Folder holding raw/ and manifest.json "
                             "(default: <ROOT_PATH>/eval/eval_datasets/pannuke_eval_data).")

    parser.add_argument("--out_dir", type=str, default=None,
                        help="Folder where characterization/ is written (default: --data_dir).")

    parser.add_argument("--folds", type=str, nargs="*", default=None,
                        help="Only scan the parquet files whose stem contains one of these tokens, e.g. fold1 "
                             "(default: all folds). Disables the acceptance gate.")

    parser.add_argument("--max_tiles", type=int, default=None,
                        help="Stop after this many tiles per fold (disables the acceptance gate).")

    parser.add_argument("--markdown_path", type=str, default=None,
                        help="Path of the Markdown summary "
                             "(default: <repo>/access_reports/A2_pannuke_characterization.md).")

    parser.add_argument("--overlay_tiles", type=int, default=12,
                        help="Number of overlay PNGs, spread over tissues (0 disables them).")

    parser.add_argument("--sthelar_calibration", dest="sthelar_calibration", action="store_true",
                        default=True,
                        help="Estimate the PanNuke um/px from its median nucleus area vs the STHELAR "
                             "cell-token-membership nuc_px (default: on).")

    parser.add_argument("--no_sthelar_calibration", dest="sthelar_calibration", action="store_false",
                        help="Skip the STHELAR resolution calibration.")

    parser.add_argument("--sthelar_training_dataset_id", type=str,
                        default=DEFAULT_STHELAR_TRAINING_DATASET_ID,
                        help="training_dataset_id whose cell-token-membership tables calibrate the "
                             "resolution estimate.")

    parser.add_argument("--sthelar_modelgroup", type=str, default=DEFAULT_MEMBERSHIP_MODELGROUP,
                        help="Model-group suffix of the cell-token-membership folder.")

    parser.add_argument("--sthelar_max_slides", type=int, default=5,
                        help="Number of STHELAR slides pooled for the median nucleus area.")

    args = parser.parse_args()
    sys.exit(main(args))
