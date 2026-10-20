#!/usr/bin/env python3
"""Baseline feature builders for TCGA Task B."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from _eval_common import (
    BenchmarkPaths,
    FeatureBundle,
    coalesce_merge,
    load_case_slide_contributions,
    load_final_case,
    nonempty,
    numeric_or_nan,
    read_json,
    require_columns,
    sha256_obj,
    sha256_path,
    truthy,
    utc_now_iso,
)

# Tile manifest paths in the shared tile store (with a fallback outside the project tree)
try:  # pragma: no cover - exercised in the real project tree
    from _taskb_common import (  # type: ignore
        reroot_under_data_root,
        shared_tile_manifest_path,
    )
except Exception:  # pragma: no cover - lightweight fallback for syntax/tests only
    def shared_tile_manifest_path(data_root: Path, uuid: str, set_recipe_id: str) -> Path:
        return (
            Path(data_root) / "processed_assets" / "tiles" / "_store"
            / "by_uuid" / str(uuid) / str(set_recipe_id) / "manifest.parquet"
        )

    def reroot_under_data_root(data_root: Path, stored_path: Optional[str]) -> Optional[Path]:
        if not stored_path:
            return None
        data_root = Path(data_root)
        anchor = data_root.name
        parts = Path(str(stored_path)).parts
        if anchor not in parts:
            return None
        idx = len(parts) - 1 - parts[::-1].index(anchor)
        return data_root.joinpath(*parts[idx + 1:])


logger = logging.getLogger(__name__)


def _final_base(paths: BenchmarkPaths) -> pd.DataFrame:
    final = load_final_case(paths)
    keep = ["case_id", "case_uuid", "project_id", "n_tiles_eligible_before_cap", "n_tiles_used_after_cap", "n_slides_retained"]
    for c in keep:
        if c not in final.columns:
            final[c] = pd.NA
    return final.loc[:, keep].copy()


def build_tile_count_features(paths: BenchmarkPaths) -> FeatureBundle:
    df = _final_base(paths)
    for c in ["n_tiles_used_after_cap", "n_tiles_eligible_before_cap", "n_slides_retained"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["log1p_n_tiles_used_after_cap"] = np.log1p(df["n_tiles_used_after_cap"].fillna(0.0).to_numpy(dtype=float))
    df["log1p_n_tiles_eligible_before_cap"] = np.log1p(df["n_tiles_eligible_before_cap"].fillna(0.0).to_numpy(dtype=float))
    df["n_slides_retained_feature"] = df["n_slides_retained"].fillna(0.0).to_numpy(dtype=float)
    feature_cols = ["log1p_n_tiles_used_after_cap", "log1p_n_tiles_eligible_before_cap", "n_slides_retained_feature"]
    return FeatureBundle(
        features=df[["case_id", "case_uuid", "project_id"] + feature_cols],
        feature_columns=feature_cols,
        numeric_columns=feature_cols,
        categorical_columns=[],
        metadata={
            "baseline_id": "tile_count",
            "feature_definition": "log1p tile counts and retained slide count from frozen final_case_manifest.csv",
            "input_hashes": {"final_case_manifest": sha256_path(paths.cohort_dir / "final_case_manifest.csv")},
        },
        feature_hash_by_project={},
    )


def _resolve_tile_asset_root(paths: BenchmarkPaths) -> Path:
    root = paths.links.get("tile_asset_root")
    if root:
        return Path(str(root))
    tile_asset_id = paths.links.get("tile_asset_id") or paths.config.get("tile_asset_id")
    if not tile_asset_id:
        raise RuntimeError("Cannot resolve tile_asset_root from benchmark_asset_links.json")
    return paths.data_root / "processed_assets" / "tiles" / str(tile_asset_id)


def _resolve_shared_tile_manifest_paths(paths: BenchmarkPaths) -> List[Path]:
    """
    Per-slide tile manifest.parquet paths of the benchmark in the shared tile store (from the referenced_shared_tiles
    map of tile_asset_manifest.json). Returns [] if the map is absent; raises if a referenced manifest is missing.
    """
    tile_root = _resolve_tile_asset_root(paths)
    manifest_path = tile_root / "tile_asset_manifest.json"
    if not manifest_path.exists():
        return []
    manifest = read_json(manifest_path)
    ref_tiles = manifest.get("referenced_shared_tiles") if isinstance(manifest, Mapping) else None
    if not isinstance(ref_tiles, Mapping) or not ref_tiles:
        return []
    resolved: List[Path] = []
    for uuid in sorted(ref_tiles):
        info = ref_tiles[uuid]
        if not isinstance(info, Mapping):
            continue
        set_recipe_id = str(info.get("set_recipe_id") or "")
        candidates: List[Path] = []
        if set_recipe_id:
            candidates.append(shared_tile_manifest_path(paths.data_root, uuid, set_recipe_id))
        rerooted = reroot_under_data_root(paths.data_root, info.get("manifest_path"))
        if rerooted is not None:
            candidates.append(rerooted)
        stored = info.get("manifest_path")
        if stored:
            candidates.append(Path(str(stored)))
        chosen = next((c for c in candidates if c.exists()), None)
        if chosen is None:
            raise FileNotFoundError(
                f"Shared tile manifest missing for slide {uuid} "
                f"(set_recipe_id={set_recipe_id or '<unknown>'}); tried "
                f"{[str(c) for c in candidates]}. Re-run _10_build_tile_asset.py or "
                "restore the shared tile store."
            )
        resolved.append(chosen)
    return resolved


def build_tissue_area_features(paths: BenchmarkPaths) -> FeatureBundle:
    final = _final_base(paths)
    contrib = load_case_slide_contributions(paths)
    p = paths.tiles_dir / "case_tile_index.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    case_tile = pd.read_parquet(p)
    require_columns(case_tile, ["case_id", "tile_id", "tile_fov_um_effective"], "case_tile_index")
    tile_root = _resolve_tile_asset_root(paths)
    # Per-tile tissue_fraction from case_tile_index.parquet when present, else from the per-slide manifests
    if "tissue_fraction" in case_tile.columns:
        logger.info(
            "tissue_area: using per-tile tissue_fraction already present in "
            "case_tile_index.parquet (no per-slide manifest scan)"
        )
        ct = case_tile
    else:
        # Read tile_id + tissue_fraction from the shared per-slide manifests
        logger.info(
            "tissue_area: case_tile_index.parquet has no tissue_fraction column; "
            "scanning per-slide manifests"
        )
        manifest_paths = _resolve_shared_tile_manifest_paths(paths)
        used_ids = set(case_tile["tile_id"].astype(str).tolist())
        wanted_cols = ["tile_id", "tissue_fraction"]
        frames: List[pd.DataFrame] = []
        for mp in tqdm(manifest_paths, desc="tissue_area: scan slide manifests", unit="slide"):
            pf = pq.ParquetFile(mp)
            if not {"tile_id", "tissue_fraction"}.issubset(pf.schema_arrow.names):
                continue
            t = pf.read(columns=wanted_cols).to_pandas()
            t = t.loc[t["tile_id"].astype(str).isin(used_ids), wanted_cols].copy()
            if not t.empty:
                frames.append(t)
        if frames:
            tissue = pd.concat(frames, ignore_index=True).drop_duplicates("tile_id")
            ct = case_tile.merge(tissue, on="tile_id", how="left", validate="one_to_one")
        else:
            # No tile-level tissue_fraction: std is missing, sum/mean use case_slide_contributions below
            ct = pd.DataFrame(columns=["case_id", "tissue_fraction", "tile_fov_um_effective"])
    rows: List[Dict[str, Any]] = []
    for cid, g in ct.groupby("case_id", sort=True):
        tf = pd.to_numeric(g.get("tissue_fraction", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        fov = pd.to_numeric(g.get("tile_fov_um_effective", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(tf) & np.isfinite(fov)
        if finite.any():
            area = float(np.sum(tf[finite] * (fov[finite] ** 2)))
            mean_tf = float(np.mean(tf[finite]))
            std_tf = float(np.std(tf[finite], ddof=0))
            n_tiles = int(finite.sum())
        else:
            area = float("nan")
            mean_tf = float("nan")
            std_tf = float("nan")
            n_tiles = 0
        rows.append({"case_id": str(cid), "sum_tissue_area_um2": area, "mean_tissue_fraction": mean_tf, "std_tissue_fraction": std_tf, "n_tiles_with_tissue_fraction": n_tiles})
    feat = final.merge(pd.DataFrame(rows), on="case_id", how="left")
    if "sum_tissue_area_um2" not in feat.columns:
        feat["sum_tissue_area_um2"] = pd.NA
        feat["mean_tissue_fraction"] = pd.NA
        feat["std_tissue_fraction"] = pd.NA
        feat["n_tiles_with_tissue_fraction"] = 0
    # Fallback from case_slide_contributions for sum/mean when tile-level tissue_fraction is unavailable
    if feat["sum_tissue_area_um2"].isna().any() and "sum_tissue_area_um2_slide" in contrib.columns:
        agg = contrib.copy()
        agg["sum_tissue_area_um2_slide"] = pd.to_numeric(agg["sum_tissue_area_um2_slide"], errors="coerce")
        agg["n_tiles_used_after_cap_slide"] = pd.to_numeric(agg["n_tiles_used_after_cap_slide"], errors="coerce")
        if "mean_tissue_fraction_slide" in agg.columns:
            agg["mean_tissue_fraction_slide"] = pd.to_numeric(agg["mean_tissue_fraction_slide"], errors="coerce")
        fallback_rows = []
        for cid, g in agg.groupby("case_id", sort=True):
            n = g["n_tiles_used_after_cap_slide"].fillna(0).to_numpy(dtype=float)
            area = float(np.nansum(g["sum_tissue_area_um2_slide"].to_numpy(dtype=float)))
            if "mean_tissue_fraction_slide" in g.columns and np.nansum(n) > 0:
                mean_tf = float(np.nansum(g["mean_tissue_fraction_slide"].to_numpy(dtype=float) * n) / np.nansum(n))
            else:
                mean_tf = float("nan")
            fallback_rows.append({"case_id": str(cid), "sum_tissue_area_um2_fallback": area, "mean_tissue_fraction_fallback": mean_tf})
        fb = pd.DataFrame(fallback_rows)
        feat = feat.merge(fb, on="case_id", how="left")
        feat["sum_tissue_area_um2"] = feat["sum_tissue_area_um2"].fillna(feat["sum_tissue_area_um2_fallback"])
        feat["mean_tissue_fraction"] = feat["mean_tissue_fraction"].fillna(feat["mean_tissue_fraction_fallback"])
        feat = feat.drop(columns=["sum_tissue_area_um2_fallback", "mean_tissue_fraction_fallback"], errors="ignore")
    feature_cols = ["sum_tissue_area_um2", "mean_tissue_fraction", "std_tissue_fraction"]
    return FeatureBundle(
        features=feat[["case_id", "case_uuid", "project_id"] + feature_cols],
        feature_columns=feature_cols,
        numeric_columns=feature_cols,
        categorical_columns=[],
        metadata={
            "baseline_id": "tissue_area",
            "feature_definition": "sum tissue_fraction * tile_fov_um_effective^2, mean tissue_fraction, std tissue_fraction over frozen selected tiles",
            "fallback": "case_slide_contributions used for sum/mean if tile-level tissue_fraction unavailable",
            "input_hashes": {
                "case_tile_index": sha256_path(paths.tiles_dir / "case_tile_index.parquet"),
                "case_slide_contributions": sha256_path(paths.tiles_dir / "case_slide_contributions.parquet"),
                "tile_asset_manifest": sha256_path(tile_root / "tile_asset_manifest.json") if (tile_root / "tile_asset_manifest.json").exists() else None,
            },
        },
        feature_hash_by_project={},
    )


def _weighted_mode(values: Sequence[Any], weights: Sequence[float]) -> str:
    totals: Dict[str, float] = {}
    for v, w in zip(values, weights):
        s = str(v).strip() if nonempty(v) else "<UNK>"
        totals[s] = totals.get(s, 0.0) + float(w if math.isfinite(float(w)) else 0.0)
    if not totals:
        return "<UNK>"
    return sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def build_batch_stain_features(paths: BenchmarkPaths) -> FeatureBundle:
    final = load_final_case(paths)
    slide_manifest_path = paths.cohort_dir / "slide_manifest.csv"
    if not slide_manifest_path.exists():
        raise FileNotFoundError(slide_manifest_path)
    slide_manifest = pd.read_csv(slide_manifest_path, dtype=str, keep_default_na=False)
    contrib = load_case_slide_contributions(paths)
    tile_root = _resolve_tile_asset_root(paths)
    qc_path = tile_root / "slide_qc_summary.parquet"
    if qc_path.exists():
        qc = pd.read_parquet(qc_path)
    else:
        qc = pd.DataFrame(columns=["slide_file_uuid", "openslide_vendor"])
    require_columns(contrib, ["case_id", "slide_file_uuid", "n_tiles_used_after_cap_slide"], "case_slide_contributions")
    sm_cols = ["slide_file_uuid", "file_name", "stain_type", "unknown_stain", "unknown_ffpe", "match_tier_best_case", "biospecimen_match_level"]
    for c in sm_cols:
        if c not in slide_manifest.columns:
            slide_manifest[c] = pd.NA
    q_cols = ["slide_file_uuid", "openslide_vendor", "technical_prefetch_status", "full_openslide_mpp_x", "full_openslide_mpp_y"]
    for c in q_cols:
        if c not in qc.columns:
            qc[c] = pd.NA
    # slide_manifest / qc values replace the same columns of contrib
    x = coalesce_merge(
        contrib, slide_manifest[sm_cols].drop_duplicates("slide_file_uuid"),
        on="slide_file_uuid", how="left",
        overwrite=["match_tier_best_case", "biospecimen_match_level"],
    )
    x = coalesce_merge(
        x, qc[q_cols].drop_duplicates("slide_file_uuid"),
        on="slide_file_uuid", how="left",
        overwrite=["technical_prefetch_status"],
    )
    rows: List[Dict[str, Any]] = []
    for cid, g in x.groupby("case_id", sort=True):
        weights = pd.to_numeric(g["n_tiles_used_after_cap_slide"], errors="coerce").fillna(0).to_numpy(dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(g), dtype=float)
        case_id = str(cid)
        parts = case_id.split("-")
        tss = parts[1] if len(parts) > 1 else "<UNK>"
        file_exts = [Path(str(v)).suffix.lower() or "<UNK>" for v in g.get("file_name", pd.Series([""] * len(g))).tolist()]
        rows.append({
            "case_id": case_id,
            "tissue_source_site_code": tss,
            "slide_file_extension_mode": _weighted_mode(file_exts, weights),
            "stain_type_mode": _weighted_mode(g.get("stain_type", pd.Series(["<UNK>"] * len(g))).tolist(), weights),
            "unknown_stain_any": bool(g.get("unknown_stain", pd.Series([False] * len(g))).map(truthy).any()),
            "unknown_ffpe_any": bool(g.get("unknown_ffpe", pd.Series([False] * len(g))).map(truthy).any()),
            "match_tier_best_case_mode": _weighted_mode(g.get("match_tier_best_case", pd.Series(["<UNK>"] * len(g))).tolist(), weights),
            "biospecimen_match_level_mode": _weighted_mode(g.get("biospecimen_match_level", pd.Series(["<UNK>"] * len(g))).tolist(), weights),
            "openslide_vendor_mode": _weighted_mode(g.get("openslide_vendor", pd.Series(["<UNK>"] * len(g))).tolist(), weights),
            "technical_prefetch_status_mode": _weighted_mode(g.get("technical_prefetch_status", pd.Series(["<UNK>"] * len(g))).tolist(), weights),
            "mean_full_openslide_mpp_x": float(np.nanmean(pd.to_numeric(g.get("full_openslide_mpp_x", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float))) if len(g) else np.nan,
            "mean_full_openslide_mpp_y": float(np.nanmean(pd.to_numeric(g.get("full_openslide_mpp_y", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float))) if len(g) else np.nan,
        })
    df = final[["case_id", "case_uuid", "project_id"]].merge(pd.DataFrame(rows), on="case_id", how="left", validate="one_to_one")
    categorical = [
        "tissue_source_site_code", "slide_file_extension_mode", "stain_type_mode", "unknown_stain_any", "unknown_ffpe_any",
        "match_tier_best_case_mode", "biospecimen_match_level_mode", "openslide_vendor_mode", "technical_prefetch_status_mode",
    ]
    numeric = ["mean_full_openslide_mpp_x", "mean_full_openslide_mpp_y"]
    feature_cols = categorical + numeric
    return FeatureBundle(
        features=df[["case_id", "case_uuid", "project_id"] + feature_cols],
        feature_columns=feature_cols,
        numeric_columns=numeric,
        categorical_columns=categorical,
        metadata={
            "baseline_id": "batch_stain_diagnostic",
            "feature_definition": "site + scanner-MPP confound floor: tissue-source-site code, tile-contribution-weighted stain/vendor/prefetch categories, and mean scanner MPP (no morphology)",
            "input_hashes": {
                "slide_manifest": sha256_path(slide_manifest_path),
                "case_slide_contributions": sha256_path(paths.tiles_dir / "case_slide_contributions.parquet"),
                "slide_qc_summary": sha256_path(qc_path) if qc_path.exists() else None,
            },
        },
        feature_hash_by_project={},
    )


def build_random_feature_bundle(paths: BenchmarkPaths, *, feature_dim: int, seed: int) -> FeatureBundle:
    final = load_final_case(paths).sort_values(["project_id", "case_id"]).reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    x = rng.standard_normal((len(final), int(feature_dim))).astype(np.float32)
    cols = [f"random_feat_{i:06d}" for i in range(int(feature_dim))]
    df = pd.concat([final[["case_id", "case_uuid", "project_id"]].reset_index(drop=True), pd.DataFrame(x, columns=cols)], axis=1)
    return FeatureBundle(
        features=df,
        feature_columns=cols,
        numeric_columns=cols,
        categorical_columns=[],
        metadata={
            "baseline_id": "random_features",
            "random_seed": int(seed),
            "random_feature_dim": int(feature_dim),
            "case_order_sha256": sha256_obj(final["case_id"].astype(str).tolist()),
            "benchmark_id": paths.config.get("benchmark_id", ""),
        },
        feature_hash_by_project={},
    )
