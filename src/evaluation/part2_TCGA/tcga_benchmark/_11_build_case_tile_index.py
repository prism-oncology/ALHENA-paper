#!/usr/bin/env python3
"""
Build the case-level tile index of a TCGA Task B benchmark (per-case tile caps, minimum tiles per case,
per-project trim of the over-provisioned cases) and its final cohort.

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data / benchmarks / <benchmark_id> /
    - shared/tiles/: case_tile_index.parquet, case_tile_diagnostics.parquet, case_slide_contributions.parquet,
      tile_exclusion_log.csv, case_tile_index_manifest.json
    - shared/cohort/: final_case_manifest.csv, project_summary.json
    - reports/benchmark_build_report.md
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path

from _taskb_common import (
    collect_output_hashes,
    manifest_header,
    read_json,
    read_yaml,
    sha256_obj,
    sha256_path,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)

logger = logging.getLogger(__name__)

OFFICIAL_MATCH_LEVELS = {"exact_same_portion", "same_sample", "same_case_primary_tumor", "same_case_primary_tumor_unknown_sublevel"}

CASE_TILE_INDEX_COLUMNS = [
    "case_id",
    "case_uuid",
    "project_id",
    "slide_file_uuid",
    "slide_file_name",
    "tile_id",
    "tile_order_in_case",
    "top_left_x_l0",
    "top_left_y_l0",
    "center_x_l0",
    "center_y_l0",
    "read_window_px_l0_x",
    "read_window_px_l0_y",
    "source_mpp_x",
    "source_mpp_y",
    "tile_fov_um_effective",
    "n_tiles_eligible_before_cap",
    "n_tiles_used_after_cap",
    "n_slides_retained",
    "match_tier_best_case",
    "biospecimen_match_level",
    "official_primary_match",
    "tile_asset_id",
    "benchmark_id",
    "modelgroup_id",
]

CASE_SLIDE_CONTRIBUTIONS_COLUMNS = [
    "case_id",
    "case_uuid",
    "project_id",
    "slide_file_uuid",
    "slide_file_name",
    "n_tiles_eligible_before_cap_slide",
    "n_tiles_used_after_cap_slide",
    "slide_tile_fraction_before_cap",
    "slide_tile_fraction_after_cap",
    "mean_tissue_fraction_slide",
    "sum_tissue_area_um2_slide",
    "source_mpp_x",
    "source_mpp_y",
    "biospecimen_match_level",
    "match_tier_best_case",
    "official_primary_match",
    "technical_prefetch_status",
    "benchmark_id",
    "modelgroup_id",
    "tile_asset_id",
]

CASE_TILE_DIAGNOSTICS_COLUMNS = [
    "case_id",
    "case_uuid",
    "project_id",
    "n_slides_retained",
    "n_tiles_eligible_before_cap",
    "n_tiles_used_after_cap",
    "dominant_slide_file_uuid_before_cap",
    "dominant_slide_tile_fraction_before_cap",
    "dominant_slide_file_uuid_after_cap",
    "dominant_slide_tile_fraction_after_cap",
    "slide_tile_entropy_before_cap",
    "slide_tile_entropy_after_cap",
    "slide_tile_effective_n_before_cap",
    "slide_tile_effective_n_after_cap",
    "slide_tile_gini_before_cap",
    "slide_tile_gini_after_cap",
    "single_slide_case",
    "multi_slide_case",
    "dominant_slide_fraction_gt_0_50",
    "dominant_slide_fraction_gt_0_75",
    "dominant_slide_fraction_gt_0_90",
    "cap_applied",
    "benchmark_id",
    "modelgroup_id",
    "tile_asset_id",
]

FINAL_CASE_MANIFEST_COLUMNS = [
    "case_id",
    "case_uuid",
    "project_id",
    "rna_file_uuid",
    "rna_file_name",
    "n_slides_retained",
    "n_tiles_eligible_before_cap",
    "n_tiles_used_after_cap",
    "match_tier_best_case",
    "biospecimen_match_level_best_case",
    "official_primary_match",
    "final_case_eligible",
    "final_case_exclusion_reason",
    "tile_asset_id",
    "benchmark_id",
    "modelgroup_id",
]

TILE_EXCLUSION_COLUMNS = [
    "stage",
    "case_id",
    "case_uuid",
    "project_id",
    "slide_file_uuid",
    "reason",
    "n_tiles_affected",
    "detail",
    "benchmark_id",
    "modelgroup_id",
    "tile_asset_id",
]


def _load_mapping(path: Path) -> Dict[str, Any]:
    obj = read_json(path) if path.suffix.lower() == ".json" else read_yaml(path)
    if not isinstance(obj, Mapping):
        raise RuntimeError(f"Expected mapping at {path}")
    return dict(obj)


def _effective_case_cap(cohort_dir: Path) -> Tuple[Optional[int], Optional[bool], Path]:
    """Per-project case cap applied by _6_resolve_rna_and_slides.py, read from rna_slide_resolution_manifest.json.

    Returns (max_cases_per_project, case_cap_applied, manifest_path); case_cap_applied is None if unavailable.
    """
    manifest_path = cohort_dir / "rna_slide_resolution_manifest.json"
    if not manifest_path.exists():
        return None, None, manifest_path
    obj = read_json(manifest_path)
    if not isinstance(obj, Mapping):
        return None, None, manifest_path
    policy = obj.get("case_cap_policy")
    if not isinstance(policy, Mapping):
        policy = obj.get("summary") if isinstance(obj.get("summary"), Mapping) else {}
    applied = policy.get("case_cap_applied")
    cap = policy.get("max_cases_per_project")
    cap_int = int(cap) if cap is not None else None
    applied_bool = bool(applied) if applied is not None else None
    return cap_int, applied_bool, manifest_path


def _nonempty(x: Any) -> bool:
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    s = str(x).strip()
    return bool(s and s.lower() not in {"nan", "none", "null", "na", "n/a", "<na>"})


def _truthy(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if not _nonempty(x):
        return False
    return str(x).strip().lower() in {"true", "t", "1", "yes", "y"}


def _ordered(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = pd.NA
    extras = [c for c in out.columns if c not in columns]
    return out.loc[:, list(columns) + extras]


def _hash_downsample_keys(df: pd.DataFrame, *, benchmark_id: str, modelgroup_id: str) -> pd.Series:
    """Deterministic per-tile SHA256 keys used to downsample the tiles of a case."""
    required = [
        "case_id",
        "slide_file_uuid",
        "center_x_l0",
        "center_y_l0",
        "tile_fov_um_effective",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Cannot build downsample keys; missing columns: {missing}")
    if df.empty:
        return pd.Series([], index=df.index, dtype=object)

    case_ids = df["case_id"].astype(str).to_numpy(dtype=object, copy=False)
    slide_uuids = df["slide_file_uuid"].astype(str).to_numpy(dtype=object, copy=False)
    center_x = pd.to_numeric(df["center_x_l0"], errors="raise").to_numpy(dtype="float64", copy=False)
    center_y = pd.to_numeric(df["center_y_l0"], errors="raise").to_numpy(dtype="float64", copy=False)
    fovs = pd.to_numeric(df["tile_fov_um_effective"], errors="raise").to_numpy(dtype="float64", copy=False)

    keys = [
        sha256_obj({
            "case_id": case_id,
            "slide_file_uuid": slide_uuid,
            "center_x_l0": float(cx),
            "center_y_l0": float(cy),
            "tile_fov_um_effective": float(fov),
            "benchmark_id": benchmark_id,
            "modelgroup_id": modelgroup_id,
        })
        for case_id, slide_uuid, cx, cy, fov in zip(
            case_ids,
            slide_uuids,
            center_x,
            center_y,
            fovs,
        )
    ]
    return pd.Series(keys, index=df.index, dtype=object)


def _entropy(counts: Iterable[int]) -> float:
    arr = np.asarray([c for c in counts if c > 0], dtype=float)
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-(p * np.log(p)).sum())


def _effective_n(counts: Iterable[int]) -> float:
    return float(math.exp(_entropy(counts)))


def _gini(counts: Iterable[int]) -> float:
    arr = np.asarray([c for c in counts if c >= 0], dtype=float)
    if arr.size == 0 or arr.sum() == 0:
        return 0.0
    arr.sort()
    n = arr.size
    index = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(index * arr) / (n * np.sum(arr))) - ((n + 1.0) / n))


def _dominant(counts: Mapping[str, int]) -> Tuple[Optional[str], float]:
    if not counts:
        return None, 0.0
    total = sum(int(v) for v in counts.values())
    if total <= 0:
        return None, 0.0
    # Deterministic tie-break by slide UUID ascending.
    slide, val = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[0]
    return str(slide), float(int(val) / total)


def _resolve_tile_asset(data_root: Path, settings_dir: Path, explicit_tile_asset_id: Optional[str]) -> Tuple[str, Path, Path, Dict[str, Any]]:
    links_path = settings_dir / "benchmark_asset_links.json"
    if not links_path.exists():
        raise FileNotFoundError(links_path)
    links = read_json(links_path)
    if not isinstance(links, Mapping):
        raise RuntimeError("benchmark_asset_links.json must be a mapping")
    tile_asset_id = explicit_tile_asset_id or links.get("tile_asset_id")
    if not tile_asset_id:
        raise RuntimeError("tile_asset_id missing. Run _10_build_tile_asset.py first or pass --tile_asset_id.")
    tile_root = Path(links.get("tile_asset_root") or data_root / "processed_assets" / "tiles" / str(tile_asset_id))
    manifest_path = Path(links.get("tile_asset_manifest_path") or tile_root / "tile_asset_manifest.json")
    if not tile_root.exists():
        raise FileNotFoundError(tile_root)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    return str(tile_asset_id), tile_root, manifest_path, dict(links)


def _read_kept_tiles(tile_manifest: Mapping[str, Any], tile_asset_id: str, *, benchmark_id: str, modelgroup_id: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    # Per-slide tile manifests from the shared store (referenced_shared_tiles); benchmark ids are added to the rows
    ref_tiles = tile_manifest.get("referenced_shared_tiles")
    if not isinstance(ref_tiles, Mapping) or not ref_tiles:
        raise RuntimeError(
            "tile_asset_manifest.json has no referenced_shared_tiles map; "
            "re-run _10_build_tile_asset.py (shared tile store)."
        )
    frames: List[pd.DataFrame] = []
    exclusions: List[Dict[str, Any]] = []
    required = [
        "tile_id",
        "slide_file_uuid",
        "file_name",
        "case_id",
        "case_uuid",
        "project_id",
        "top_left_x_l0",
        "top_left_y_l0",
        "center_x_l0",
        "center_y_l0",
        "read_window_px_l0_x",
        "read_window_px_l0_y",
        "source_mpp_x",
        "source_mpp_y",
        "tile_fov_um_effective",
        "tissue_fraction",
        "keep_tile",
        "technical_prefetch_status",
    ]
    for uuid in tqdm(sorted(ref_tiles), desc="Reading tile manifests", unit="slide"):
        info = ref_tiles[uuid]
        mp = info.get("manifest_path") if isinstance(info, Mapping) else None
        if not mp:
            continue
        p = Path(str(mp))
        if not p.exists():
            raise FileNotFoundError(f"Shared tile manifest missing for slide {uuid}: {p}")
        df = pd.read_parquet(p)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(f"{p} missing required tile columns: {missing}")
        if df.empty:
            exclusions.append({
                "stage": "tile_manifest",
                "case_id": None,
                "case_uuid": None,
                "project_id": None,
                "slide_file_uuid": str(uuid),
                "reason": "empty_tile_manifest_or_failed_slide_validation",
                "n_tiles_affected": 0,
                "detail": str(p),
                "benchmark_id": benchmark_id,
                "modelgroup_id": modelgroup_id,
                "tile_asset_id": tile_asset_id,
            })
            continue
        df = df.loc[df["keep_tile"].map(_truthy)].copy()
        if df.empty:
            exclusions.append({
                "stage": "tile_qc",
                "case_id": None,
                "case_uuid": None,
                "project_id": None,
                "slide_file_uuid": str(uuid),
                "reason": "no_kept_tiles_after_qc",
                "n_tiles_affected": 0,
                "detail": str(p),
                "benchmark_id": benchmark_id,
                "modelgroup_id": modelgroup_id,
                "tile_asset_id": tile_asset_id,
            })
            continue
        df["tile_asset_id"] = tile_asset_id
        df["benchmark_id"] = benchmark_id
        df["modelgroup_id"] = modelgroup_id
        frames.append(df)
    if not frames:
        return pd.DataFrame(), exclusions
    out = pd.concat(frames, ignore_index=True)
    if out["tile_id"].duplicated().any():
        dups = out.loc[out["tile_id"].duplicated(), "tile_id"].head(20).tolist()
        raise RuntimeError(f"Duplicate tile_id values in tile asset: {dups}")
    return out, exclusions


def _load_slide_metadata(cohort_dir: Path) -> pd.DataFrame:
    p = cohort_dir / "slide_manifest.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    required = ["slide_file_uuid", "case_id", "case_uuid", "project_id", "biospecimen_match_level", "match_tier_best_case", "official_primary_match"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"slide_manifest.csv missing required columns: {missing}")
    if "slide_file_name" not in df.columns:
        df["slide_file_name"] = df.get("file_name", "")
    keep_cols = [
        "slide_file_uuid",
        "slide_file_name",
        "case_id",
        "case_uuid",
        "project_id",
        "biospecimen_match_level",
        "match_tier_best_case",
        "official_primary_match",
    ]
    # One row per slide
    meta = df.loc[:, keep_cols].copy()
    if meta["slide_file_uuid"].duplicated().any():
        dups = meta.loc[meta["slide_file_uuid"].duplicated(), "slide_file_uuid"].head(20).tolist()
        raise RuntimeError(f"slide_manifest.csv has duplicate slide_file_uuid rows: {dups}")
    return meta


def _case_level_select(
    tiles: pd.DataFrame,
    slide_meta: pd.DataFrame,
    *,
    min_tiles_per_case: int,
    max_tiles_per_case: int,
    benchmark_id: str,
    modelgroup_id: str,
    tile_asset_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if tiles.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(columns=TILE_EXCLUSION_COLUMNS)

    merged = tiles.merge(slide_meta, on=["slide_file_uuid", "case_id", "case_uuid", "project_id"], how="left", validate="many_to_one")
    missing_meta = merged["biospecimen_match_level"].isna() | ~merged["biospecimen_match_level"].astype(str).isin(OFFICIAL_MATCH_LEVELS) | ~merged["official_primary_match"].map(_truthy)
    exclusion_rows: List[Dict[str, Any]] = []
    if missing_meta.any():
        bad = merged.loc[missing_meta, ["case_id", "case_uuid", "project_id", "slide_file_uuid"]].drop_duplicates()
        for r in bad.to_dict("records"):
            exclusion_rows.append({
                "stage": "official_primary_filter",
                "case_id": r.get("case_id"),
                "case_uuid": r.get("case_uuid"),
                "project_id": r.get("project_id"),
                "slide_file_uuid": r.get("slide_file_uuid"),
                "reason": "slide_not_official_primary_at_any_admitted_tier",
                "n_tiles_affected": int((merged["slide_file_uuid"] == r.get("slide_file_uuid")).sum()),
                "detail": "biospecimen_match_level must be one of exact_same_portion/same_sample/same_case_primary_tumor[_unknown_sublevel] and official_primary_match true",
                "benchmark_id": benchmark_id,
                "modelgroup_id": modelgroup_id,
                "tile_asset_id": tile_asset_id,
            })
    merged = merged.loc[~missing_meta].copy()
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), _ordered(pd.DataFrame(exclusion_rows), TILE_EXCLUSION_COLUMNS)

    selected_frames: List[pd.DataFrame] = []
    diagnostics: List[Dict[str, Any]] = []
    contributions: List[Dict[str, Any]] = []

    n_cases = int(merged["case_id"].nunique())
    for case_id, case_df in tqdm(merged.groupby("case_id", sort=True), total=n_cases, desc="Selecting case tiles", unit="case"):
        case_df = case_df.copy()
        case_uuid = str(case_df["case_uuid"].iloc[0])
        project_id = str(case_df["project_id"].iloc[0])
        n_before = int(len(case_df))
        slide_counts_before = case_df.groupby("slide_file_uuid", sort=True).size().astype(int).to_dict()
        n_slides_before = int(len(slide_counts_before))
        dom_slide_before, dom_frac_before = _dominant(slide_counts_before)

        if n_before < min_tiles_per_case:
            exclusion_rows.append({
                "stage": "case_tile_minimum",
                "case_id": case_id,
                "case_uuid": case_uuid,
                "project_id": project_id,
                "slide_file_uuid": None,
                "reason": "case_below_min_tiles_per_case",
                "n_tiles_affected": n_before,
                "detail": json.dumps({"min_tiles_per_case": min_tiles_per_case, "n_tiles_eligible_before_cap": n_before}, sort_keys=True),
                "benchmark_id": benchmark_id,
                "modelgroup_id": modelgroup_id,
                "tile_asset_id": tile_asset_id,
            })
            continue

        case_df["downsample_key"] = _hash_downsample_keys(case_df, benchmark_id=benchmark_id, modelgroup_id=modelgroup_id)
        case_df = case_df.sort_values(["downsample_key", "slide_file_uuid", "center_x_l0", "center_y_l0", "tile_id"]).reset_index(drop=True)
        cap_applied = n_before > max_tiles_per_case
        selected = case_df.head(max_tiles_per_case).copy() if cap_applied else case_df.copy()
        dropped_n = n_before - len(selected)
        if dropped_n > 0:
            exclusion_rows.append({
                "stage": "case_tile_cap",
                "case_id": case_id,
                "case_uuid": case_uuid,
                "project_id": project_id,
                "slide_file_uuid": None,
                "reason": "deterministic_sha256_downsample_above_max_tiles_per_case",
                "n_tiles_affected": int(dropped_n),
                "detail": json.dumps({"max_tiles_per_case": max_tiles_per_case, "n_tiles_eligible_before_cap": n_before}, sort_keys=True),
                "benchmark_id": benchmark_id,
                "modelgroup_id": modelgroup_id,
                "tile_asset_id": tile_asset_id,
            })
        selected = selected.reset_index(drop=True)
        selected["tile_order_in_case"] = np.arange(len(selected), dtype=int)
        n_after = int(len(selected))
        slide_counts_after = selected.groupby("slide_file_uuid", sort=True).size().astype(int).to_dict()
        n_slides_after = int(len(slide_counts_after))
        dom_slide_after, dom_frac_after = _dominant(slide_counts_after)

        before_counts_arr = list(slide_counts_before.values())
        after_counts_arr = list(slide_counts_after.values())
        diagnostics.append({
            "case_id": case_id,
            "case_uuid": case_uuid,
            "project_id": project_id,
            "n_slides_retained": n_slides_after,
            "n_tiles_eligible_before_cap": n_before,
            "n_tiles_used_after_cap": n_after,
            "dominant_slide_file_uuid_before_cap": dom_slide_before,
            "dominant_slide_tile_fraction_before_cap": dom_frac_before,
            "dominant_slide_file_uuid_after_cap": dom_slide_after,
            "dominant_slide_tile_fraction_after_cap": dom_frac_after,
            "slide_tile_entropy_before_cap": _entropy(before_counts_arr),
            "slide_tile_entropy_after_cap": _entropy(after_counts_arr),
            "slide_tile_effective_n_before_cap": _effective_n(before_counts_arr),
            "slide_tile_effective_n_after_cap": _effective_n(after_counts_arr),
            "slide_tile_gini_before_cap": _gini(before_counts_arr),
            "slide_tile_gini_after_cap": _gini(after_counts_arr),
            "single_slide_case": bool(n_slides_after == 1),
            "multi_slide_case": bool(n_slides_after > 1),
            "dominant_slide_fraction_gt_0_50": bool(dom_frac_after > 0.50),
            "dominant_slide_fraction_gt_0_75": bool(dom_frac_after > 0.75),
            "dominant_slide_fraction_gt_0_90": bool(dom_frac_after > 0.90),
            "cap_applied": bool(cap_applied),
            "benchmark_id": benchmark_id,
            "modelgroup_id": modelgroup_id,
            "tile_asset_id": tile_asset_id,
        })

        for slide_uuid, g_before in case_df.groupby("slide_file_uuid", sort=True):
            g_after = selected.loc[selected["slide_file_uuid"].astype(str).eq(str(slide_uuid))]
            n_before_slide = int(len(g_before))
            n_after_slide = int(len(g_after))
            fov = float(g_before["tile_fov_um_effective"].iloc[0]) if len(g_before) else float("nan")
            tissue_after = (
                pd.to_numeric(g_after["tissue_fraction"], errors="coerce")
                if n_after_slide > 0
                else pd.Series(dtype=float)
            )
            tissue_area_um2 = float((tissue_after.fillna(0.0) * fov * fov).sum()) if n_after_slide > 0 else 0.0
            mean_tissue_after = float(tissue_after.mean()) if n_after_slide > 0 else float("nan")
            meta_row = g_before.iloc[0]
            contributions.append({
                "case_id": case_id,
                "case_uuid": case_uuid,
                "project_id": project_id,
                "slide_file_uuid": slide_uuid,
                "slide_file_name": meta_row.get("slide_file_name") or meta_row.get("file_name"),
                "n_tiles_eligible_before_cap_slide": n_before_slide,
                "n_tiles_used_after_cap_slide": n_after_slide,
                "slide_tile_fraction_before_cap": float(n_before_slide / n_before) if n_before else 0.0,
                "slide_tile_fraction_after_cap": float(n_after_slide / n_after) if n_after else 0.0,
                "mean_tissue_fraction_slide": mean_tissue_after,
                "sum_tissue_area_um2_slide": tissue_area_um2,
                "source_mpp_x": float(pd.to_numeric(g_before["source_mpp_x"], errors="coerce").median()),
                "source_mpp_y": float(pd.to_numeric(g_before["source_mpp_y"], errors="coerce").median()),
                "biospecimen_match_level": meta_row.get("biospecimen_match_level"),
                "match_tier_best_case": meta_row.get("match_tier_best_case"),
                "official_primary_match": bool(_truthy(meta_row.get("official_primary_match"))),
                "technical_prefetch_status": meta_row.get("technical_prefetch_status"),
                "benchmark_id": benchmark_id,
                "modelgroup_id": modelgroup_id,
                "tile_asset_id": tile_asset_id,
            })

        selected["n_tiles_eligible_before_cap"] = n_before
        selected["n_tiles_used_after_cap"] = n_after
        selected["n_slides_retained"] = n_slides_after
        selected_frames.append(selected)

    if selected_frames:
        selected_all = pd.concat(selected_frames, ignore_index=True)
        selected_all["slide_file_name"] = selected_all["slide_file_name"].where(selected_all["slide_file_name"].map(_nonempty), selected_all.get("file_name"))
        index_df = _ordered(selected_all, CASE_TILE_INDEX_COLUMNS)
        index_df = index_df.sort_values(["project_id", "case_id", "tile_order_in_case", "slide_file_uuid", "tile_id"]).reset_index(drop=True)
    else:
        index_df = pd.DataFrame(columns=CASE_TILE_INDEX_COLUMNS)
    diag_df = _ordered(pd.DataFrame(diagnostics), CASE_TILE_DIAGNOSTICS_COLUMNS)
    contrib_df = _ordered(pd.DataFrame(contributions), CASE_SLIDE_CONTRIBUTIONS_COLUMNS)
    excl_df = _ordered(pd.DataFrame(exclusion_rows), TILE_EXCLUSION_COLUMNS)
    return index_df, diag_df, contrib_df, excl_df


def _build_final_case_manifest(
    *,
    rna_manifest: pd.DataFrame,
    case_tile_index: pd.DataFrame,
    case_diag: pd.DataFrame,
    benchmark_id: str,
    modelgroup_id: str,
    tile_asset_id: str,
) -> pd.DataFrame:
    required_rna = ["case_id", "case_uuid", "project_id", "rna_file_uuid"]
    missing = [c for c in required_rna if c not in rna_manifest.columns]
    if missing:
        raise RuntimeError(f"rna_manifest.csv missing required columns: {missing}")
    if "rna_file_name" not in rna_manifest.columns:
        rna_manifest = rna_manifest.copy()
        rna_manifest["rna_file_name"] = rna_manifest.get("file_name", "")
    if rna_manifest["case_id"].duplicated().any():
        dups = rna_manifest.loc[rna_manifest["case_id"].duplicated(), "case_id"].head(20).tolist()
        raise RuntimeError(f"rna_manifest.csv must contain one RNA row per case; duplicates: {dups}")

    if case_tile_index.empty:
        out = rna_manifest[["case_id", "case_uuid", "project_id", "rna_file_uuid", "rna_file_name"]].copy()
        out["n_slides_retained"] = 0
        out["n_tiles_eligible_before_cap"] = 0
        out["n_tiles_used_after_cap"] = 0
        out["match_tier_best_case"] = pd.NA
        out["biospecimen_match_level_best_case"] = pd.NA
        out["official_primary_match"] = False
        out["final_case_eligible"] = False
        out["final_case_exclusion_reason"] = "no_retained_tiles"
        out["tile_asset_id"] = tile_asset_id
        out["benchmark_id"] = benchmark_id
        out["modelgroup_id"] = modelgroup_id
        return _ordered(out, FINAL_CASE_MANIFEST_COLUMNS)

    tile_case = case_tile_index.groupby(["case_id", "case_uuid", "project_id"], as_index=False, sort=True).agg(
        n_slides_retained=("slide_file_uuid", "nunique"),
        n_tiles_used_after_cap=("tile_id", "size"),
        n_tiles_eligible_before_cap=("n_tiles_eligible_before_cap", "first"),
        match_tier_best_case=("match_tier_best_case", "first"),
        biospecimen_match_level_best_case=("biospecimen_match_level", "first"),
        official_primary_match=("official_primary_match", lambda s: bool(all(_truthy(x) for x in s))),
    )
    out = rna_manifest[["case_id", "case_uuid", "project_id", "rna_file_uuid", "rna_file_name"]].merge(
        tile_case,
        on=["case_id", "case_uuid", "project_id"],
        how="left",
        validate="one_to_one",
    )
    out["n_slides_retained"] = out["n_slides_retained"].fillna(0).astype(int)
    out["n_tiles_eligible_before_cap"] = out["n_tiles_eligible_before_cap"].fillna(0).astype(int)
    out["n_tiles_used_after_cap"] = out["n_tiles_used_after_cap"].fillna(0).astype(int)
    out["official_primary_match"] = out["official_primary_match"].map(_truthy)
    out["final_case_eligible"] = (
        (out["n_slides_retained"] >= 1)
        & (out["n_tiles_used_after_cap"] > 0)
        & (out["biospecimen_match_level_best_case"].astype(str).isin(OFFICIAL_MATCH_LEVELS))
        & (out["official_primary_match"])
    )
    reasons = []
    for r in out.to_dict("records"):
        rs = []
        if int(r.get("n_slides_retained") or 0) < 1:
            rs.append("no_retained_diagnostic_primary_slide")
        if int(r.get("n_tiles_used_after_cap") or 0) <= 0:
            rs.append("no_tiles_used_after_cap")
        if str(r.get("biospecimen_match_level_best_case")) not in OFFICIAL_MATCH_LEVELS:
            rs.append("biospecimen_match_level_not_in_admitted_official_tiers")
        if not _truthy(r.get("official_primary_match")):
            rs.append("official_primary_match_not_true")
        reasons.append("" if not rs else ";".join(rs))
    out["final_case_exclusion_reason"] = reasons
    out["tile_asset_id"] = tile_asset_id
    out["benchmark_id"] = benchmark_id
    out["modelgroup_id"] = modelgroup_id
    return _ordered(out, FINAL_CASE_MANIFEST_COLUMNS)


def _project_summary(final_case_manifest: pd.DataFrame, *, min_official_project_cases: int) -> Dict[str, Any]:
    rows = []
    eligible = final_case_manifest.loc[final_case_manifest["final_case_eligible"].map(_truthy)].copy()
    for project_id, g in final_case_manifest.groupby("project_id", sort=True):
        ge = g.loc[g["final_case_eligible"].map(_truthy)]
        n = int(len(ge))
        rows.append({
            "project_id": project_id,
            "n_cases_total_after_rna_selection": int(len(g)),
            "n_final_cases": n,
            "official_project": bool(n >= min_official_project_cases),
            "official_project_min_cases": int(min_official_project_cases),
        })
    return {
        "n_cases_total_after_rna_selection": int(len(final_case_manifest)),
        "n_final_cases": int(len(eligible)),
        "n_projects_with_final_cases": int(eligible["project_id"].nunique()) if not eligible.empty else 0,
        "n_official_projects": int(sum(1 for r in rows if r["official_project"])),
        "official_project_min_cases": int(min_official_project_cases),
        "project_rows": rows,
    }


def _write_report(path: Path, summary: Mapping[str, Any], *, overwrite: bool) -> None:
    lines = [
        "# Task B benchmark build report: tile index and final cohort",
        "",
        f"- benchmark_id: `{summary.get('benchmark_id')}`",
        f"- modelgroup_id: `{summary.get('modelgroup_id')}`",
        f"- tile_asset_id: `{summary.get('tile_asset_id')}`",
        f"- final cases: `{summary.get('n_final_cases')}`",
        f"- official projects: `{summary.get('n_official_projects')}`",
        f"- min_official_project_cases: `{summary.get('min_official_project_cases')}` (CLI option, not stored in benchmark_config.yaml)",
        f"- max_cases_per_project (effective, from _6): `{summary.get('max_cases_per_project_effective')}`",
        f"- case_tile_index rows: `{summary.get('n_case_tile_index_rows')}`",
        f"- min_tiles_per_case: `{summary.get('min_tiles_per_case')}`",
        f"- max_tiles_per_case: `{summary.get('max_tiles_per_case')}`",
        f"- over-provision trim applied: `{summary.get('overprovision_trim_applied')}` "
        f"(surplus cases trimmed back to the per-project target: `{summary.get('n_trimmed_over_provision_surplus')}`)",
        "",
        "Primary tile selection pools all official-primary retained slides at case level and down-samples only by deterministic SHA256 order when over the cap.",
        "After tile QC, each project's eligible survivors are trimmed back to the per-project target (max_cases_per_project) by lowest case_selection_rank, undoing the _6 over-provision buffer.",
        "",
    ]
    write_text_atomic(path, "\n".join(lines), overwrite=overwrite)


def _update_asset_links(settings_dir: Path, update: Mapping[str, Any], *, force: bool, force_cascade: bool) -> Dict[str, Any]:
    path = settings_dir / "benchmark_asset_links.json"
    old = read_json(path)
    if not isinstance(old, Mapping):
        raise RuntimeError("benchmark_asset_links.json must be a mapping")
    old_dict = dict(old)
    conflicts = {}
    for k, v in update.items():
        if k in old_dict and old_dict[k] not in (v, None, ""):
            conflicts[k] = {"old": old_dict[k], "new": v}
    if conflicts and not (force and force_cascade):
        raise RuntimeError(f"Refusing to change existing benchmark links without --force --force_cascade: {conflicts}")
    merged = dict(old_dict)
    merged.update(update)
    old_hash = sha256_path(path)
    write_json_atomic(path, merged, overwrite=True)
    return {"path": str(path), "previous_sha256": old_hash, "new_sha256": sha256_path(path), "updated_keys": sorted(update)}


def main(args) -> int:

    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    settings_dir = benchmark_root / "settings"
    cohort_dir = benchmark_root / "shared" / "cohort"
    tiles_dir = benchmark_root / "shared" / "tiles"
    reports_dir = benchmark_root / "reports"
    if not settings_dir.exists():
        raise FileNotFoundError(settings_dir)
    if not cohort_dir.exists():
        raise FileNotFoundError(cohort_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        tiles_dir / "case_tile_index.parquet",
        tiles_dir / "case_tile_diagnostics.parquet",
        tiles_dir / "case_slide_contributions.parquet",
        tiles_dir / "tile_exclusion_log.csv",
        tiles_dir / "case_tile_index_manifest.json",
        cohort_dir / "final_case_manifest.csv",
        cohort_dir / "project_summary.json",
    ]
    if any(p.exists() for p in outputs) and not args.force:
        existing = [str(p) for p in outputs if p.exists()]
        raise FileExistsError(f"Case tile index/final cohort outputs already exist; use --force to rebuild. Existing: {existing[:8]}")
    if any(p.exists() for p in outputs) and args.force and not args.force_cascade:
        downstream = [benchmark_root / "shared" / "rna" / "target_tables", benchmark_root / "shared" / "folds"]
        existing_downstream = [p for p in downstream if p.exists()]
        if existing_downstream:
            raise RuntimeError(f"Refusing to rebuild case tile index while downstream target/fold assets exist: {existing_downstream}. Use --force_cascade if intentional.")

    cfg = _load_mapping(settings_dir / "benchmark_config.yaml")
    links_path = settings_dir / "benchmark_asset_links.json"
    modelgroup_id = str(cfg.get("modelgroup_id"))
    min_tiles = int(args.min_tiles_per_case if args.min_tiles_per_case is not None else cfg.get("min_tiles_per_case", 256))
    max_tiles = int(args.max_tiles_per_case if args.max_tiles_per_case is not None else cfg.get("max_tiles_per_case", 8192))
    if min_tiles <= 0 or max_tiles < min_tiles:
        raise RuntimeError("Invalid min/max tile configuration")

    # min_official_project_cases (not in benchmark_config.yaml) must not exceed the case cap applied by _6
    min_official_project_cases = int(args.min_official_project_cases)
    if min_official_project_cases < 0:
        raise RuntimeError("Invalid --min_official_project_cases: must be >= 0")
    effective_cap, cap_applied, cap_manifest_path = _effective_case_cap(cohort_dir)
    if cap_applied is True and effective_cap is not None and min_official_project_cases > effective_cap:
        raise RuntimeError(
            f"--min_official_project_cases={min_official_project_cases} exceeds the per-project case cap "
            f"(max_cases_per_project={effective_cap}) applied by _6_resolve_rna_and_slides.py (see {cap_manifest_path}). "
            f"Pass --min_official_project_cases <= {effective_cap}."
        )
    if cap_applied is None:
        logger.warning(
            "Could not read the per-project case cap from %s; the check "
            "min_official_project_cases (%d) <= cap is skipped.",
            cap_manifest_path,
            min_official_project_cases,
        )

    tile_asset_id, tile_root, tile_asset_manifest_path, asset_links = _resolve_tile_asset(data_root, settings_dir, args.tile_asset_id)
    tile_asset_manifest = read_json(tile_asset_manifest_path)
    if not isinstance(tile_asset_manifest, Mapping):
        raise RuntimeError("tile_asset_manifest.json must be a mapping")
    if tile_asset_manifest.get("tile_asset_id") != tile_asset_id:
        raise RuntimeError(f"tile_asset_id mismatch: links/CLI={tile_asset_id}, manifest={tile_asset_manifest.get('tile_asset_id')}")
    if str(tile_asset_manifest.get("benchmark_id")) != args.benchmark_id:
        raise RuntimeError("tile asset benchmark_id mismatch")
    if str(tile_asset_manifest.get("modelgroup_id")) != modelgroup_id:
        raise RuntimeError("tile asset modelgroup_id mismatch")
    if tile_asset_manifest.get("frozen_final") is not True or str(tile_asset_manifest.get("status", "frozen_final")) != "frozen_final":
        raise RuntimeError(
            "Refusing to build the final case tile index from a non-final/debug tile asset. "
            f"frozen_final={tile_asset_manifest.get('frozen_final')}, status={tile_asset_manifest.get('status')}"
        )

    rna_manifest_path = cohort_dir / "rna_manifest.csv"
    if not rna_manifest_path.exists():
        raise FileNotFoundError(rna_manifest_path)
    rna_manifest = pd.read_csv(rna_manifest_path, dtype=str, keep_default_na=False)
    slide_meta = _load_slide_metadata(cohort_dir)

    logger.info("Reading kept tiles for benchmark %s (tile asset %s) ...", args.benchmark_id, tile_asset_id)
    tiles, pre_exclusions = _read_kept_tiles(tile_asset_manifest, tile_asset_id, benchmark_id=args.benchmark_id, modelgroup_id=modelgroup_id)
    logger.info("Read %d kept tiles; running case-level tile selection (SHA256 down-sampling) ...", int(len(tiles)))
    index_df, diag_df, contrib_df, excl_df = _case_level_select(
        tiles,
        slide_meta,
        min_tiles_per_case=min_tiles,
        max_tiles_per_case=max_tiles,
        benchmark_id=args.benchmark_id,
        modelgroup_id=modelgroup_id,
        tile_asset_id=tile_asset_id,
    )
    if pre_exclusions:
        excl_df = pd.concat([_ordered(pd.DataFrame(pre_exclusions), TILE_EXCLUSION_COLUMNS), excl_df], ignore_index=True)
    excl_df = _ordered(excl_df, TILE_EXCLUSION_COLUMNS)

    final_case_manifest = _build_final_case_manifest(
        rna_manifest=rna_manifest,
        case_tile_index=index_df,
        case_diag=diag_df,
        benchmark_id=args.benchmark_id,
        modelgroup_id=modelgroup_id,
        tile_asset_id=tile_asset_id,
    )
    # Enforce the explicit min-tile criterion in final cohort manifest.
    too_few = final_case_manifest["n_tiles_used_after_cap"].astype(int) < min_tiles
    final_case_manifest.loc[too_few, "final_case_eligible"] = False
    final_case_manifest.loc[too_few & final_case_manifest["final_case_exclusion_reason"].astype(str).eq(""), "final_case_exclusion_reason"] = "case_below_min_tiles_per_case"
    final_case_manifest = _ordered(final_case_manifest, FINAL_CASE_MANIFEST_COLUMNS)

    # Over-provision trim: keep the max_cases_per_project eligible cases per project with the lowest
    # case_selection_rank (from rna_manifest.csv); the others become ineligible
    overprovision_trim: Dict[str, Any] = {"applied": False, "per_project_target_cap": effective_cap, "per_project": []}
    trim_rank_available = "case_selection_rank" in rna_manifest.columns
    if effective_cap is not None and trim_rank_available:
        rank_map = dict(zip(
            rna_manifest["case_id"].astype(str),
            pd.to_numeric(rna_manifest["case_selection_rank"], errors="coerce"),
        ))
        elig = final_case_manifest.loc[final_case_manifest["final_case_eligible"].map(_truthy), ["case_id", "project_id"]].copy()
        elig["case_id"] = elig["case_id"].astype(str)
        elig["rank"] = elig["case_id"].map(rank_map).fillna(np.inf)
        surplus_case_ids = set()
        per_project_trim: List[Dict[str, Any]] = []
        for project_id, g in elig.groupby("project_id", sort=True):
            n_before = int(len(g))
            n_keep = min(n_before, int(effective_cap))
            if n_before > int(effective_cap):
                surplus_case_ids.update(
                    g.sort_values(["rank", "case_id"]).iloc[int(effective_cap):]["case_id"].tolist()
                )
            per_project_trim.append({
                "project_id": str(project_id),
                "n_eligible_before_trim": n_before,
                "n_kept_after_trim": n_keep,
                "n_trimmed_surplus": n_before - n_keep,
            })
        if surplus_case_ids:
            mask = final_case_manifest["case_id"].astype(str).isin(surplus_case_ids)
            final_case_manifest.loc[mask, "final_case_eligible"] = False
            empty_reason = final_case_manifest["final_case_exclusion_reason"].astype(str).eq("")
            final_case_manifest.loc[mask & empty_reason, "final_case_exclusion_reason"] = "per_project_over_provision_trim_surplus"
        overprovision_trim = {
            "applied": True,
            "per_project_target_cap": int(effective_cap),
            "n_trimmed_surplus_total": int(len(surplus_case_ids)),
            "per_project": per_project_trim,
        }
    elif effective_cap is not None and not trim_rank_available:
        logger.warning(
            "rna_manifest.csv lacks case_selection_rank; skipping the over-provision trim "
            "(final per-project cohort may exceed target=%d if it was over-provisioned).",
            int(effective_cap),
        )

    project_summary = _project_summary(final_case_manifest, min_official_project_cases=min_official_project_cases)

    logger.info("Writing case tile index and final cohort outputs to %s ...", tiles_dir)
    write_parquet_atomic(tiles_dir / "case_tile_index.parquet", index_df, overwrite=args.force)
    write_parquet_atomic(tiles_dir / "case_tile_diagnostics.parquet", diag_df, overwrite=args.force)
    write_parquet_atomic(tiles_dir / "case_slide_contributions.parquet", contrib_df, overwrite=args.force)
    write_csv_atomic(tiles_dir / "tile_exclusion_log.csv", excl_df, overwrite=args.force)
    write_csv_atomic(cohort_dir / "final_case_manifest.csv", final_case_manifest, overwrite=args.force)
    project_summary_obj = {
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": modelgroup_id,
        "tile_asset_id": tile_asset_id,
        **project_summary,
    }
    write_json_atomic(cohort_dir / "project_summary.json", project_summary_obj, overwrite=args.force)

    link_update = _update_asset_links(settings_dir, {
        "tile_asset_id": tile_asset_id,
        "tile_asset_root": str(tile_root),
        "tile_asset_manifest_path": str(tile_asset_manifest_path),
        "case_tile_index_path": str(tiles_dir / "case_tile_index.parquet"),
        "case_tile_diagnostics_path": str(tiles_dir / "case_tile_diagnostics.parquet"),
        "case_slide_contributions_path": str(tiles_dir / "case_slide_contributions.parquet"),
        "final_case_manifest_path": str(cohort_dir / "final_case_manifest.csv"),
        "project_summary_path": str(cohort_dir / "project_summary.json"),
    }, force=bool(args.force), force_cascade=bool(args.force_cascade))

    summary = {
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": modelgroup_id,
        "tile_asset_id": tile_asset_id,
        "min_tiles_per_case": min_tiles,
        "max_tiles_per_case": max_tiles,
        "min_official_project_cases": min_official_project_cases,
        "max_cases_per_project_effective": effective_cap,
        "n_kept_tiles_before_case_filter": int(len(tiles)) if tiles is not None else 0,
        "n_case_tile_index_rows": int(len(index_df)),
        "n_cases_with_tiles_after_filter": int(index_df["case_id"].nunique()) if not index_df.empty else 0,
        "n_final_cases": int(final_case_manifest["final_case_eligible"].map(_truthy).sum()),
        "n_official_projects": int(project_summary_obj["n_official_projects"]),
        "n_case_exclusion_or_cap_log_rows": int(len(excl_df)),
        "overprovision_trim_applied": bool(overprovision_trim.get("applied")),
        "n_trimmed_over_provision_surplus": int(overprovision_trim.get("n_trimmed_surplus_total", 0) or 0),
    }

    manifest = manifest_header(Path(__file__).name, vars(args))
    manifest.update({
        "frozen_final": True,
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": modelgroup_id,
        "tile_asset_id": tile_asset_id,
        "input_paths": {
            "tile_asset_manifest": str(tile_asset_manifest_path),
            "tile_asset_root": str(tile_root),
            "rna_manifest_csv": str(rna_manifest_path),
            "slide_manifest_csv": str(cohort_dir / "slide_manifest.csv"),
            "benchmark_config": str(settings_dir / "benchmark_config.yaml"),
            "benchmark_asset_links": str(links_path),
        },
        "input_hashes": {
            "tile_asset_manifest": sha256_path(tile_asset_manifest_path),
            "rna_manifest_csv": sha256_path(rna_manifest_path),
            "slide_manifest_csv": sha256_path(cohort_dir / "slide_manifest.csv"),
            "benchmark_config": sha256_path(settings_dir / "benchmark_config.yaml"),
            "benchmark_asset_links_after_update": sha256_path(links_path),
        },
        "selection_policy": {
            "min_tiles_per_case": min_tiles,
            "max_tiles_per_case": max_tiles,
            "min_official_project_cases": min_official_project_cases,
            "min_official_project_cases_frozen_in_config": False,
            "max_cases_per_project_effective": effective_cap,
            "pooling": "pool all official-primary retained slides per case",
            "slide_balancing": "none",
            "downsample_key": ["case_id", "slide_file_uuid", "center_x_l0", "center_y_l0", "tile_fov_um_effective", "benchmark_id", "modelgroup_id"],
            "downsample_order": "ascending_sha256_hex",
            "runtime_random_sampling": False,
        },
        "summary": summary,
        "project_summary": project_summary_obj,
        "overprovision_trim": overprovision_trim,
        "asset_link_update": link_update,
        "output_hashes": collect_output_hashes(tiles_dir),
        "cohort_output_hashes": {k: v for k, v in collect_output_hashes(cohort_dir).items() if k in {"final_case_manifest.csv", "project_summary.json"}},
    })
    write_json_atomic(tiles_dir / "case_tile_index_manifest.json", manifest, overwrite=args.force)
    _write_report(reports_dir / "benchmark_build_report.md", summary, overwrite=args.force)

    if summary["n_final_cases"] <= 0:
        raise RuntimeError("No final eligible cases after tile selection")

    print(json.dumps({"status": "ok", **summary, "case_tile_index": str(tiles_dir / "case_tile_index.parquet")}, indent=2))
    return 0


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Build the frozen case-level tile index and the final TCGA Task B cohort.")

    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help="Benchmark ID whose tile manifests are aggregated into the case tile index.")

    ap.add_argument("--tile_asset_id", type=str,
                    default=None,
                    help="Tile asset ID to index. Defaults to the tile asset referenced by the benchmark.")

    ap.add_argument("--min_tiles_per_case", type=int,
                    default=None,
                    help="Minimum tiles required for a case to enter the final cohort. Defaults to the value in benchmark_config.yaml or 256 if not set there.")

    ap.add_argument("--max_tiles_per_case", type=int,
                    default=None,
                    help="Maximum tiles kept per case (the per-case tile cap). Defaults to the value in benchmark_config.yaml or 8192 if not set there.")

    ap.add_argument("--min_official_project_cases", type=int,
                    default=50,
                    help="Minimum final cases for a project to be official (used by _13_build_targets_and_folds.py); "
                         "must be <= the per-project case cap applied by _6_resolve_rna_and_slides.py.")

    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing case tile index and final cohort.")

    ap.add_argument("--force_cascade", action="store_true",
                    help="Required with --force if downstream targets/folds already exist.")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main(args))
