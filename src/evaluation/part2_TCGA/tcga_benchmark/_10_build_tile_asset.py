#!/usr/bin/env python3
"""
Build the TCGA Task B tile asset from the downloaded slides: full-file OpenSlide MPP check, tiling at the
model-group geometry and tile QC (tissue, white, blur, HistoQC mask overlap). Slides run in parallel (--max_workers).

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data /
    - processed_assets/tiles/_store/by_uuid/<slide_file_uuid>/<set_recipe_id>/: manifest.parquet (tiles and QC),
      slide_qc.json, tile_set_record.json, shared by all benchmarks
    - processed_assets/tiles/<tile_asset_id>/: tile_asset_manifest.json, slide_qc_summary.parquet
    - benchmarks/<benchmark_id>/reports/tile_asset_build_report.md (tile keys added to settings/benchmark_asset_links.json)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import sys

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add this script's directory to sys.path (for `_taskb_common` in the worker processes)
_this_dir = Path(__file__).resolve().parent
if str(_this_dir) not in sys.path:
    sys.path.insert(0, str(_this_dir))

from src.utils.constants import get_full_path

from _taskb_common import (
    assert_modelgroup_no_upsampling,
    collect_output_hashes,
    composite_rgba_on_white,
    imaging_versions,
    manifest_header,
    read_json,
    read_yaml,
    resolve_histoqc_mask_use,
    sha256_obj,
    sha256_path,
    shared_tile_set_dir,
    shared_tile_set_is_complete,
    shared_tiles_store_root,
    software_versions,
    tile_set_recipe_id,
    utc_now_iso,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)

try:  # optional at import time; required in main
    import openslide  # type: ignore
except Exception:  # pragma: no cover
    openslide = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

# Default number of worker processes (all cores but 2)
DEFAULT_MAX_WORKERS = max(1, (os.cpu_count() or 4) - 2)

DEFAULT_QC_CONFIG: Dict[str, Any] = {
    "qc_version": "tcga_tile_qc_v1",
    "qc_tile_size_px": 224,
    "qc_resize_interpolation": "area",
    "tissue_mask": {
        "method": "hsv_saturation_value",
        "saturation_min": 20,
        "value_max": 245,
        "tissue_fraction_threshold": 0.50,
    },
    "white_flag": {
        "method": "rgb_white_fraction",
        "white_rgb_min": 235,
        "max_white_fraction": 0.80,
    },
    "blur_flag": {
        "method": "variance_of_laplacian",
        "grayscale_source": "qc_rgb_tile",
        "blur_threshold": 10.0,
    },
    # HistoQC: minimum fraction of the tile bounding box inside the HistoQC tissue-use mask
    "histoqc": {
        "histoqc_artifact_check_enabled": True,
        "histoqc_tissue_overlap_min": 0.70,
        "histoqc_overlap_method": "bbox_mean_in_mask_coords",
    },
}

TILE_MANIFEST_COLUMNS = [
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
    "tile_fov_um_read",
    "tile_fov_um_effective",
    "tile_stride_um",
    "tile_resample_factor",
    "tile_resample_direction",
    "tissue_fraction",
    "white_fraction",
    "white_flag",
    "blur_score",
    "blur_flag",
    "histoqc_tile_overlap",
    "histoqc_overlap_flag",
    "keep_tile",
    "qc_version",
    "tiling_version",
    "failure_reason",
    "technical_prefetch_status",
    "prefetch_mpp_x",
    "prefetch_mpp_y",
    "prefetch_range_supported",
    "prefetch_bytes_received",
    "prefetch_partial_sha256",
    "full_openslide_mpp_x",
    "full_openslide_mpp_y",
    "prefetch_full_mpp_consistency",
    "final_mpp_source",
]

SLIDE_QC_SUMMARY_COLUMNS = [
    "slide_file_uuid",
    "file_name",
    "case_id",
    "case_uuid",
    "project_id",
    "local_path",
    "slide_status",
    "failure_reason",
    "openslide_vendor",
    "slide_width_l0",
    "slide_height_l0",
    "full_openslide_mpp_x",
    "full_openslide_mpp_y",
    "read_window_px_l0_x",
    "read_window_px_l0_y",
    "stride_px_l0_x",
    "stride_px_l0_y",
    "resample_factor_x",
    "resample_factor_y",
    "resample_direction",
    "n_candidate_tiles",
    "n_candidates_skipped_outside_histoqc_mask",
    "n_read_failures",
    "n_keep_tiles",
    "histoqc_asset_id",
    "histoqc_mask_use_path",
    "histoqc_mask_use_sha256",
    "histoqc_mask_shape_h",
    "histoqc_mask_shape_w",
    "histoqc_scale_x",
    "histoqc_scale_y",
    "histoqc_mean_tile_overlap",
    "technical_prefetch_status",
    "prefetch_mpp_x",
    "prefetch_mpp_y",
    "prefetch_full_mpp_consistency",
    "prefetch_full_discrepancy_reason",
    "set_recipe_id",
    "histoqc_run_key",
    "benchmark_id",
    "modelgroup_id",
    "tile_asset_id",
]


def _load_mapping(path: Path) -> Dict[str, Any]:
    obj = read_json(path) if path.suffix.lower() == ".json" else read_yaml(path)
    if not isinstance(obj, Mapping):
        raise RuntimeError(f"Expected mapping at {path}")
    return dict(obj)


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


def _float_or_none(x: Any) -> Optional[float]:
    if not _nonempty(x):
        return None
    try:
        v = float(str(x).strip())
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _int_or_none(x: Any) -> Optional[int]:
    if not _nonempty(x):
        return None
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


def _ordered(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = pd.NA
    extras = [c for c in out.columns if c not in columns]
    return out.loc[:, list(columns) + extras]


def _empty_tile_df() -> pd.DataFrame:
    return pd.DataFrame(columns=TILE_MANIFEST_COLUMNS)


def _read_qc_config(path: Optional[Path]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_QC_CONFIG)
    if path is None:
        return cfg
    user = _load_mapping(path)
    # Merge one level deep
    for k, v in user.items():
        if isinstance(v, Mapping) and isinstance(cfg.get(k), Mapping):
            tmp = dict(cfg[k])
            tmp.update(v)
            cfg[k] = tmp
        else:
            cfg[k] = v
    return cfg


def _validate_modelgroup(mg: Mapping[str, Any]) -> None:
    required = [
        "modelgroup_id",
        "crop_size",
        "resize_size",
        "tile_size_px_encoder_input",
        "tile_fov_um_read",
        "tile_fov_um_effective",
        "tile_stride_um",
        "nominal_mpp_um_per_px",
        "rgba_policy",
        "rgb_policy",
        "icc_profile_policy",
        "qc_version",
        "mpp_policy",
        "tiling_version",
        "tiling_toolchain_version",
        "mpp_min",
        "mpp_max",
        "mpp_anisotropy_max",
        "mpp_prefetch_abs_tol",
    ]
    missing = [c for c in required if c not in mg]
    if missing:
        raise RuntimeError(f"modelgroup is missing required fields: {missing}")
    # MPP acceptance bounds (from the model group)
    mpp_min, mpp_max = float(mg["mpp_min"]), float(mg["mpp_max"])
    if not (0.0 < mpp_min < mpp_max):
        raise RuntimeError(f"model group requires 0 < mpp_min < mpp_max, got {mpp_min}, {mpp_max}")
    if float(mg["mpp_anisotropy_max"]) < 1.0:
        raise RuntimeError(f"model group mpp_anisotropy_max must be >= 1.0, got {mg['mpp_anisotropy_max']}")
    if float(mg["mpp_prefetch_abs_tol"]) < 0.0:
        raise RuntimeError(f"model group mpp_prefetch_abs_tol must be >= 0, got {mg['mpp_prefetch_abs_tol']}")
    # No upsampling: mpp_max must not exceed the target resolution (+ tolerance)
    assert_modelgroup_no_upsampling(mg)
    if not str(mg.get("tiling_toolchain_version") or "").strip():
        raise RuntimeError("model group tiling_toolchain_version must be a non-empty string")
    # tile_fov_um_effective / crop_size must equal nominal_mpp_um_per_px
    crop = int(mg["crop_size"])
    if crop > 0:
        expected_mpp = float(mg["tile_fov_um_effective"]) / float(crop)
        nominal = float(mg["nominal_mpp_um_per_px"])
        if abs(nominal - expected_mpp) > 1e-6 * max(1.0, expected_mpp):
            raise RuntimeError(
                f"model group inconsistent: nominal_mpp_um_per_px={nominal} != "
                f"tile_fov_um_effective/crop_size ({mg['tile_fov_um_effective']}/{crop}={expected_mpp})."
            )
    if str(mg.get("rgba_policy")) != "openslide_rgba_composite_white":
        raise RuntimeError("Only rgba_policy=openslide_rgba_composite_white is supported")
    if str(mg.get("rgb_policy")) != "convert_to_rgb":
        raise RuntimeError("Only rgb_policy=convert_to_rgb is supported")
    if str(mg.get("icc_profile_policy")) != "ignore_embedded_icc_use_openslide_returned_rgb":
        raise RuntimeError("Only icc_profile_policy=ignore_embedded_icc_use_openslide_returned_rgb is supported")
    if str(mg.get("mpp_policy")) != "openslide_mpp_required":
        raise RuntimeError("Only mpp_policy=openslide_mpp_required is supported")
    if bool(mg.get("objective_power_fallback_allowed", False)):
        raise RuntimeError("objective_power_fallback_allowed must be false for the primary protocol")


def _deterministic_tile_asset_id(
    *,
    benchmark_id: str,
    modelgroup_id: str,
    slide_manifest_hash: str,
    modelgroup_yaml_hash: str,
    qc_config_hash: str,
    tiling_version: str,
    histoqc_asset_id: Optional[str],
) -> str:
    # histoqc_asset_id is None only with --no_histoqc (debug)
    components = {
        "benchmark_id": benchmark_id,
        "modelgroup_id": modelgroup_id,
        "slide_manifest_hash": slide_manifest_hash,
        "modelgroup_yaml_hash": modelgroup_yaml_hash,
        "qc_config_hash": qc_config_hash,
        "tiling_version": tiling_version,
        "histoqc_asset_id": histoqc_asset_id or "<no_histoqc_debug>",
    }
    return f"{benchmark_id}__{modelgroup_id}__tiles__{sha256_obj(components)[:16]}"


def _mpp_from_openslide_props(props: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    keys_x = ["openslide.mpp-x", "MPP_X", "aperio.MPP"]
    keys_y = ["openslide.mpp-y", "MPP_Y", "aperio.MPP"]
    x = None
    y = None
    for k in keys_x:
        if k in props:
            x = _float_or_none(props.get(k))
            if x is not None:
                break
    for k in keys_y:
        if k in props:
            y = _float_or_none(props.get(k))
            if y is not None:
                break
    return x, y


def _validate_mpp(x: Optional[float], y: Optional[float], *, mpp_min: float, mpp_max: float, anisotropy_max: float) -> Tuple[bool, str]:
    if x is None or y is None:
        return False, "mpp_unresolved"
    if not (mpp_min <= x <= mpp_max and mpp_min <= y <= mpp_max):
        return False, f"mpp_out_of_range:{x},{y}"
    if max(x, y) / min(x, y) > anisotropy_max:
        return False, f"mpp_anisotropy_too_high:{x},{y}"
    return True, ""


def _prefetch_lookup(cohort_dir: Path, *, allow_missing_prefetch: bool) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    p = cohort_dir / "slide_prefetch_manifest.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if "slide_file_uuid" not in df.columns:
            raise RuntimeError(f"{p} missing slide_file_uuid")
        return {str(r["slide_file_uuid"]): dict(r) for r in df.to_dict("records")}, {"path": str(p), "sha256": sha256_path(p), "sentinel": False}
    if not allow_missing_prefetch:
        raise FileNotFoundError(f"Missing {p}; run _7_prefetch_slide_technical_metadata.py first or pass --allow_missing_prefetch")
    return {}, {"path": None, "sha256": None, "sentinel": True, "technical_prefetch_status": "not_run"}


def _sentinel_prefetch(uuid: str) -> Dict[str, Any]:
    return {
        "slide_file_uuid": uuid,
        "technical_prefetch_status": "not_run",
        "prefetch_mpp_x": None,
        "prefetch_mpp_y": None,
        "range_supported": None,
        "bytes_received": None,
        "partial_sha256": None,
        "prefetch_exclusion_reason": "prefetch_not_run",
    }


def _prefetch_consistency(prefetch_x: Optional[float], prefetch_y: Optional[float], full_x: Optional[float], full_y: Optional[float], *, tol: float) -> Tuple[str, str, Optional[float], Optional[float]]:
    if prefetch_x is None or prefetch_y is None:
        return "prefetch_missing_or_not_run", "prefetch_mpp_missing_or_not_run", None, None
    if full_x is None or full_y is None:
        return "full_mpp_missing", "full_openslide_mpp_missing", None, None
    dx = abs(prefetch_x - full_x)
    dy = abs(prefetch_y - full_y)
    if dx <= tol and dy <= tol:
        return "consistent", "", dx, dy
    return "discrepant", f"abs_diff_exceeds_{tol}:dx={dx},dy={dy}", dx, dy


def _tile_id(
    *,
    set_recipe_id: str,
    slide_file_uuid: str,
    x: int,
    y: int,
    wx: int,
    wy: int,
    tile_fov_um_effective: float,
    tiling_version: str,
) -> str:
    # Keyed on set_recipe_id: tile ids are the same in all benchmarks sharing the tile-set recipe
    obj = {
        "set_recipe_id": str(set_recipe_id),
        "slide_file_uuid": slide_file_uuid,
        "top_left_x_l0": int(x),
        "top_left_y_l0": int(y),
        "read_window_px_l0_x": int(wx),
        "read_window_px_l0_y": int(wy),
        "tile_fov_um_effective": float(tile_fov_um_effective),
        "tiling_version": str(tiling_version),
    }
    return sha256_obj(obj)


def _resample_factor_direction(read_window_px: int, encoder_input_px: int, *, tol: float = 0.01) -> Tuple[float, str]:
    """Resample factor read_window_px / encoder_input_px and its direction: "downsample" (> 1 + tol),
    "upsample" (< 1 - tol) or "none"."""
    if encoder_input_px <= 0:
        return float("nan"), "unknown"
    factor = float(read_window_px) / float(encoder_input_px)
    if factor > 1.0 + tol:
        direction = "downsample"
    elif factor < 1.0 - tol:
        direction = "upsample"
    else:
        direction = "none"
    return factor, direction


# RGBA composited on a white background (shared with _15_build_tile_rgb_cache.py)
_composite_rgba_on_white = composite_rgba_on_white


def _resize_for_qc(rgb: np.ndarray, size: int) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("opencv-python/cv2 is required for qc_resize_interpolation=area")
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def _resolve_histoqc_mask_path(
    mask_path_by_uuid: Mapping[str, Path],
    histoqc_asset_root: Optional[Path],
    slide_file_uuid: str,
    slide_stem: str,
) -> Optional[Path]:
    """Path of one slide's HistoQC mask: the pointer of the HistoQC asset manifest (shared store),
    else <histoqc_asset_root>/slide_masks/<uuid>/, else None."""
    mp = mask_path_by_uuid.get(slide_file_uuid)
    if mp is not None and Path(mp).exists():
        return Path(mp)
    if histoqc_asset_root is not None:
        legacy = resolve_histoqc_mask_use(histoqc_asset_root / "slide_masks" / slide_file_uuid, slide_stem)
        if legacy is not None:
            return legacy
    return None


def _load_histoqc_mask(
    mask_path: Optional[Path],
    slide_l0_shape: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float], Optional[Path]]:
    """Load a HistoQC tissue-use mask. Returns (binary uint8 mask, scale_x, scale_y, mask_path), where
    scale = mask size / slide L0 size (mask_x = round(l0_x * scale_x)); mask is None if not found."""
    if mask_path is None:
        return None, None, None, None
    if cv2 is None:
        raise RuntimeError("opencv-python/cv2 is required to load HistoQC masks")
    mask_path = Path(mask_path)
    if not mask_path.exists():
        return None, None, None, None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.size == 0:
        return None, None, None, mask_path
    mask = (mask > 0).astype(np.uint8)
    width_l0, height_l0 = int(slide_l0_shape[0]), int(slide_l0_shape[1])
    mask_h, mask_w = int(mask.shape[0]), int(mask.shape[1])
    if width_l0 <= 0 or height_l0 <= 0 or mask_h <= 0 or mask_w <= 0:
        return None, None, None, mask_path
    scale_x = float(mask_w) / float(width_l0)
    scale_y = float(mask_h) / float(height_l0)
    return mask, scale_x, scale_y, mask_path


def _center_in_mask(
    x: int, y: int, wx: int, wy: int,
    mask: np.ndarray, scale_x: float, scale_y: float,
) -> bool:
    """True iff the L0-pixel tile centered at (x+wx/2, y+wy/2) projects
    onto a non-zero mask pixel."""
    cx = int(round((x + wx / 2.0) * scale_x))
    cy = int(round((y + wy / 2.0) * scale_y))
    mh, mw = mask.shape[:2]
    cx = min(max(cx, 0), mw - 1)
    cy = min(max(cy, 0), mh - 1)
    return bool(mask[cy, cx])


def _tile_histoqc_overlap(
    x: int, y: int, wx: int, wy: int,
    mask: np.ndarray, scale_x: float, scale_y: float,
) -> float:
    """Fraction of the tile's L0 bounding box that lies inside the HistoQC
    mask, computed in mask coordinates."""
    mh, mw = mask.shape[:2]
    mx0 = int(round(x * scale_x))
    my0 = int(round(y * scale_y))
    mx1 = int(round((x + wx) * scale_x))
    my1 = int(round((y + wy) * scale_y))
    mx0 = min(max(mx0, 0), mw)
    my0 = min(max(my0, 0), mh)
    mx1 = min(max(mx1, 0), mw)
    my1 = min(max(my1, 0), mh)
    if mx1 <= mx0 or my1 <= my0:
        return 0.0
    submask = mask[my0:my1, mx0:mx1]
    return float(submask.mean())


def _qc_rgb_tile(
    rgb: np.ndarray,
    qc: Mapping[str, Any],
    *,
    histoqc_tile_overlap: Optional[float] = None,
) -> Dict[str, Any]:
    if cv2 is None:
        raise RuntimeError("opencv-python/cv2 is required for Task B tile QC")
    qc_size = int(qc.get("qc_tile_size_px", 224))
    rgb_qc = _resize_for_qc(rgb, qc_size)

    tissue_cfg = qc["tissue_mask"]
    hsv = cv2.cvtColor(rgb_qc, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    tissue_mask = (sat >= int(tissue_cfg.get("saturation_min", 20))) & (val <= int(tissue_cfg.get("value_max", 245)))
    tissue_fraction = float(np.mean(tissue_mask))

    white_cfg = qc["white_flag"]
    white_rgb_min = int(white_cfg.get("white_rgb_min", 235))
    white_fraction = float(np.mean(np.all(rgb_qc >= white_rgb_min, axis=2)))
    white_flag = bool(white_fraction > float(white_cfg.get("max_white_fraction", 0.80)))

    blur_cfg = qc["blur_flag"]
    gray = cv2.cvtColor(rgb_qc, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_flag = bool(blur_score < float(blur_cfg.get("blur_threshold", 10.0)))

    histoqc_cfg = qc.get("histoqc") or {}
    histoqc_enabled = bool(histoqc_cfg.get("histoqc_artifact_check_enabled", False))
    overlap_min = float(histoqc_cfg.get("histoqc_tissue_overlap_min", 0.70))
    if histoqc_enabled and histoqc_tile_overlap is None:
        # No overlap value (--no_histoqc debug path): not rejected
        overlap_value = 1.0
        overlap_flag = False
    elif histoqc_enabled:
        overlap_value = float(histoqc_tile_overlap)  # type: ignore[arg-type]
        overlap_flag = bool(overlap_value < overlap_min)
    else:
        overlap_value = float("nan")
        overlap_flag = False

    keep_tile = bool(
        tissue_fraction >= float(tissue_cfg.get("tissue_fraction_threshold", 0.50))
        and not white_flag
        and not blur_flag
        and not overlap_flag
    )
    reasons: List[str] = []
    if tissue_fraction < float(tissue_cfg.get("tissue_fraction_threshold", 0.50)):
        reasons.append("low_tissue_fraction")
    if white_flag:
        reasons.append("white_flag")
    if blur_flag:
        reasons.append("blur_flag")
    if overlap_flag:
        reasons.append("histoqc_overlap_below_threshold")
    return {
        "tissue_fraction": tissue_fraction,
        "white_fraction": white_fraction,
        "white_flag": white_flag,
        "blur_score": blur_score,
        "blur_flag": blur_flag,
        "histoqc_tile_overlap": overlap_value if histoqc_enabled else pd.NA,
        "histoqc_overlap_flag": overlap_flag if histoqc_enabled else pd.NA,
        "keep_tile": keep_tile,
        "failure_reason": "" if keep_tile else ";".join(reasons),
    }


def _iter_positions(width: int, height: int, wx: int, wy: int, sx: int, sy: int) -> Iterable[Tuple[int, int]]:
    if wx <= 0 or wy <= 0 or sx <= 0 or sy <= 0:
        return
    max_x = width - wx
    max_y = height - wy
    if max_x < 0 or max_y < 0:
        return
    y = 0
    while y <= max_y:
        x = 0
        while x <= max_x:
            yield x, y
            x += sx
        y += sy


def _load_download_manifest(cache_root: Path) -> pd.DataFrame:
    p = cache_root / "manifests" / "selected_slide_download_manifest.tsv"
    if not p.exists():
        raise FileNotFoundError(f"Run _8_download_selected_slides.py first; missing {p}")
    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    required = ["slide_file_uuid", "local_path", "download_status", "md5sum_expected", "md5sum_observed"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"download manifest missing required columns: {missing}")
    # skipped_existing_size_ok: blob present, accepted on file-size match (fast resume in _8).
    ok_statuses = {"downloaded_md5_ok", "skipped_existing_md5_ok", "skipped_existing_size_ok"}
    ok = df.loc[df["download_status"].isin(ok_statuses)].copy()
    if ok.empty:
        raise RuntimeError("No downloaded MD5-ok slides found in selected_slide_download_manifest.tsv")
    if ok["slide_file_uuid"].duplicated().any():
        dups = ok.loc[ok["slide_file_uuid"].duplicated(), "slide_file_uuid"].head(20).tolist()
        raise RuntimeError(f"Duplicate downloaded slide_file_uuid rows: {dups}")
    return ok


def _write_report(path: Path, summary: Mapping[str, Any], *, overwrite: bool) -> None:
    lines = [
        "# Tile asset build report",
        "",
        f"- benchmark_id: `{summary.get('benchmark_id')}`",
        f"- modelgroup_id: `{summary.get('modelgroup_id')}`",
        f"- tile_asset_id: `{summary.get('tile_asset_id')}`",
        f"- downloaded slides evaluated: `{summary.get('n_downloaded_slides')}`",
        f"- slides passed full validation: `{summary.get('n_slides_passed_full_validation')}`",
        f"- slides failed full validation: `{summary.get('n_slides_failed_full_validation')}`",
        f"- kept tiles: `{summary.get('n_keep_tiles')}`",
        f"- prefetch/full MPP discrepancies: `{summary.get('n_prefetch_full_mpp_discrepancies')}`",
        "",
        "## Resolution provenance (no-upsampling contract)",
        "",
        f"- slides at native 20x resolution: `{summary.get('n_slides_native_resolution')}`",
        f"- slides downsampled from a finer native resolution: `{summary.get('n_slides_downsampled')}`",
        f"- slides upsampled (must be 0 — coarser-than-target slides are rejected): `{summary.get('n_slides_upsampled')}`",
    ]
    per_project_resample = summary.get("per_project_resample") or {}
    if per_project_resample:
        lines.extend(["", "Per-project (native / downsampled / upsampled):"])
        for proj in sorted(per_project_resample):
            r = per_project_resample[proj]
            lines.append(
                f"- `{proj}`: {r.get('n_slides_native_resolution', 0)} / "
                f"{r.get('n_slides_downsampled', 0)} / {r.get('n_slides_upsampled', 0)}"
            )
    lines.extend([
        "",
        "Tiles are read at native resolution and resized to the model group's "
        "nominal_mpp_um_per_px; finer slides are downsampled, coarser slides are rejected. "
        "Final tile geometry uses full-file OpenSlide MPP only.",
        "",
    ])
    write_text_atomic(path, "\n".join(lines), overwrite=overwrite)


def _update_asset_links(settings_dir: Path, update: Mapping[str, Any], *, force: bool, force_cascade: bool) -> Dict[str, Any]:
    path = settings_dir / "benchmark_asset_links.json"
    if not path.exists():
        raise FileNotFoundError(path)
    old = read_json(path)
    if not isinstance(old, Mapping):
        raise RuntimeError("benchmark_asset_links.json must be a mapping")
    old_dict = dict(old)
    conflicts = {}
    for k, v in update.items():
        if k in old_dict and old_dict[k] not in (v, None, ""):
            conflicts[k] = {"old": old_dict[k], "new": v}
    if conflicts and not (force and force_cascade):
        raise RuntimeError(f"Refusing to change existing tile asset links without --force --force_cascade: {conflicts}")
    merged = dict(old_dict)
    merged.update(update)
    old_hash = sha256_path(path)
    # Add the tile asset keys to benchmark_asset_links.json
    write_json_atomic(path, merged, overwrite=True)
    return {"path": str(path), "previous_sha256": old_hash, "new_sha256": sha256_path(path), "updated_keys": sorted(update)}


def _process_slide(
    *,
    row: Mapping[str, Any],
    prefetch_row: Mapping[str, Any],
    slide: Any,
    qc_config: Mapping[str, Any],
    modelgroup_id: str,
    set_recipe_id: str,
    histoqc_run_key: Optional[str],
    modelgroup: Mapping[str, Any],
    mpp_min: float,
    mpp_max: float,
    mpp_anisotropy_max: float,
    mpp_prefetch_abs_tol: float,
    max_candidates_per_slide: Optional[int],
    histoqc_mask: Optional[np.ndarray],
    histoqc_scale_x: Optional[float],
    histoqc_scale_y: Optional[float],
    histoqc_mask_path: Optional[Path],
    show_tile_progress: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    uuid = str(row["slide_file_uuid"])
    file_name = str(row.get("file_name") or Path(str(row.get("local_path", ""))).name)
    local_path = str(row["local_path"])
    case_id = row.get("case_id")
    case_uuid = row.get("case_uuid")
    project_id = row.get("project_id")
    tiling_version = str(modelgroup.get("tiling_version"))
    qc_version = str(qc_config.get("qc_version"))
    tile_fov_um_read = float(modelgroup.get("tile_fov_um_read"))
    tile_fov_um_effective = float(modelgroup.get("tile_fov_um_effective"))
    tile_stride_um = float(modelgroup.get("tile_stride_um"))

    props = dict(slide.properties)
    width, height = map(int, slide.dimensions)
    full_mpp_x, full_mpp_y = _mpp_from_openslide_props(props)
    mpp_ok, mpp_failure = _validate_mpp(full_mpp_x, full_mpp_y, mpp_min=mpp_min, mpp_max=mpp_max, anisotropy_max=mpp_anisotropy_max)

    prefetch_status = str(prefetch_row.get("technical_prefetch_status") or "not_run")
    prefetch_x = _float_or_none(prefetch_row.get("prefetch_mpp_x"))
    prefetch_y = _float_or_none(prefetch_row.get("prefetch_mpp_y"))
    prefetch_consistency, prefetch_discrepancy, diff_x, diff_y = _prefetch_consistency(prefetch_x, prefetch_y, full_mpp_x, full_mpp_y, tol=mpp_prefetch_abs_tol)

    histoqc_enabled = bool((qc_config.get("histoqc") or {}).get("histoqc_artifact_check_enabled", False))
    histoqc_mask_sha = sha256_path(histoqc_mask_path) if histoqc_mask_path is not None else None
    base_qc: Dict[str, Any] = {
        "slide_file_uuid": uuid,
        "file_name": file_name,
        "case_id": case_id,
        "case_uuid": case_uuid,
        "project_id": project_id,
        "local_path": local_path,
        "slide_status": "started",
        "failure_reason": "",
        "openslide_vendor": props.get("openslide.vendor"),
        "openslide_properties": props,
        "slide_width_l0": width,
        "slide_height_l0": height,
        "full_openslide_mpp_x": full_mpp_x,
        "full_openslide_mpp_y": full_mpp_y,
        "read_window_px_l0_x": None,
        "read_window_px_l0_y": None,
        "stride_px_l0_x": None,
        "stride_px_l0_y": None,
        "n_candidate_tiles": 0,
        "n_candidates_skipped_outside_histoqc_mask": 0,
        "n_read_failures": 0,
        "n_keep_tiles": 0,
        "histoqc_run_key": histoqc_run_key,
        "histoqc_mask_use_path": str(histoqc_mask_path) if histoqc_mask_path is not None else None,
        "histoqc_mask_use_sha256": histoqc_mask_sha,
        "histoqc_mask_shape_h": int(histoqc_mask.shape[0]) if histoqc_mask is not None else None,
        "histoqc_mask_shape_w": int(histoqc_mask.shape[1]) if histoqc_mask is not None else None,
        "histoqc_scale_x": histoqc_scale_x,
        "histoqc_scale_y": histoqc_scale_y,
        "histoqc_mean_tile_overlap": None,
        "technical_prefetch_status": prefetch_status,
        "prefetch_mpp_x": prefetch_x,
        "prefetch_mpp_y": prefetch_y,
        "prefetch_range_supported": prefetch_row.get("range_supported"),
        "prefetch_bytes_received": prefetch_row.get("bytes_received"),
        "prefetch_partial_sha256": prefetch_row.get("partial_sha256"),
        "mpp_prefetch_full_abs_diff_x": diff_x,
        "mpp_prefetch_full_abs_diff_y": diff_y,
        "prefetch_full_mpp_consistency": prefetch_consistency,
        "prefetch_full_discrepancy_reason": prefetch_discrepancy,
        "set_recipe_id": set_recipe_id,
        "modelgroup_id": modelgroup_id,
        "final_mpp_source": "full_openslide",
    }

    if histoqc_enabled and histoqc_mask is None:
        base_qc.update({"slide_status": "failed_full_validation", "failure_reason": "histoqc_mask_missing"})
        return _empty_tile_df(), base_qc

    if not mpp_ok:
        base_qc.update({"slide_status": "failed_full_validation", "failure_reason": mpp_failure})
        return _empty_tile_df(), base_qc

    assert full_mpp_x is not None and full_mpp_y is not None
    wx = int(round(tile_fov_um_read / full_mpp_x))
    wy = int(round(tile_fov_um_read / full_mpp_y))
    sx = int(round(tile_stride_um / full_mpp_x))
    sy = int(round(tile_stride_um / full_mpp_y))
    # Resample direction to the encoder input size (tolerance: max_upsample_tol_frac, as for mpp_max)
    tile_px = int(modelgroup.get("tile_size_px_encoder_input", modelgroup.get("crop_size")))
    upsample_tol = float(modelgroup.get("max_upsample_tol_frac", 0.0))
    factor_x, _dir_x = _resample_factor_direction(wx, tile_px, tol=upsample_tol)
    factor_y, _dir_y = _resample_factor_direction(wy, tile_px, tol=upsample_tol)
    _mean_factor, resample_direction = _resample_factor_direction(int(round((wx + wy) / 2.0)), tile_px, tol=upsample_tol)
    base_qc.update({
        "read_window_px_l0_x": wx, "read_window_px_l0_y": wy, "stride_px_l0_x": sx, "stride_px_l0_y": sy,
        "resample_factor_x": factor_x, "resample_factor_y": factor_y, "resample_direction": resample_direction,
    })
    if wx <= 0 or wy <= 0 or sx <= 0 or sy <= 0:
        base_qc.update({"slide_status": "failed_full_validation", "failure_reason": "nonpositive_tile_geometry"})
        return _empty_tile_df(), base_qc
    if width < wx or height < wy:
        base_qc.update({"slide_status": "failed_full_validation", "failure_reason": "slide_smaller_than_read_window"})
        return _empty_tile_df(), base_qc

    records: List[Dict[str, Any]] = []
    overlaps: List[float] = []
    n_read_fail = 0
    n_candidates = 0
    n_skipped_outside_mask = 0
    # Number of grid positions (tqdm total)
    n_grid_positions = ((width - wx) // sx + 1) * ((height - wy) // sy + 1)
    for x, y in tqdm(
        _iter_positions(width, height, wx, wy, sx, sy),
        total=n_grid_positions,
        desc=f"  tiles {uuid[:8]}",
        unit="tile",
        leave=False,
        disable=not show_tile_progress,
    ):
        # Skip positions whose tile center is outside the HistoQC mask
        if histoqc_enabled and histoqc_mask is not None:
            if not _center_in_mask(x, y, wx, wy, histoqc_mask, float(histoqc_scale_x), float(histoqc_scale_y)):
                n_skipped_outside_mask += 1
                continue
        n_candidates += 1
        if max_candidates_per_slide is not None and n_candidates > max_candidates_per_slide:
            break
        center_x = x + wx / 2.0
        center_y = y + wy / 2.0
        tid = _tile_id(
            set_recipe_id=set_recipe_id,
            slide_file_uuid=uuid,
            x=x,
            y=y,
            wx=wx,
            wy=wy,
            tile_fov_um_effective=tile_fov_um_effective,
            tiling_version=tiling_version,
        )
        rec: Dict[str, Any] = {
            "tile_id": tid,
            "slide_file_uuid": uuid,
            "file_name": file_name,
            "case_id": case_id,
            "case_uuid": case_uuid,
            "project_id": project_id,
            "top_left_x_l0": int(x),
            "top_left_y_l0": int(y),
            "center_x_l0": float(center_x),
            "center_y_l0": float(center_y),
            "read_window_px_l0_x": int(wx),
            "read_window_px_l0_y": int(wy),
            "source_mpp_x": float(full_mpp_x),
            "source_mpp_y": float(full_mpp_y),
            "tile_fov_um_read": tile_fov_um_read,
            "tile_fov_um_effective": tile_fov_um_effective,
            "tile_stride_um": tile_stride_um,
            "tile_resample_factor": float(factor_x),
            "tile_resample_direction": resample_direction,
            "tissue_fraction": pd.NA,
            "white_fraction": pd.NA,
            "white_flag": pd.NA,
            "blur_score": pd.NA,
            "blur_flag": pd.NA,
            "histoqc_tile_overlap": pd.NA,
            "histoqc_overlap_flag": pd.NA,
            "keep_tile": False,
            "qc_version": qc_version,
            "tiling_version": tiling_version,
            "failure_reason": "",
            "technical_prefetch_status": prefetch_status,
            "prefetch_mpp_x": prefetch_x,
            "prefetch_mpp_y": prefetch_y,
            "prefetch_range_supported": prefetch_row.get("range_supported"),
            "prefetch_bytes_received": prefetch_row.get("bytes_received"),
            "prefetch_partial_sha256": prefetch_row.get("partial_sha256"),
            "full_openslide_mpp_x": float(full_mpp_x),
            "full_openslide_mpp_y": float(full_mpp_y),
            "prefetch_full_mpp_consistency": prefetch_consistency,
            "final_mpp_source": "full_openslide",
        }
        # Per-tile HistoQC overlap (recorded even on read failures for diagnostics).
        if histoqc_enabled and histoqc_mask is not None:
            overlap = _tile_histoqc_overlap(x, y, wx, wy, histoqc_mask, float(histoqc_scale_x), float(histoqc_scale_y))
        else:
            overlap = None
        try:
            rgba = slide.read_region((int(x), int(y)), 0, (int(wx), int(wy)))
            rgb_img = _composite_rgba_on_white(rgba)
            rgb = np.asarray(rgb_img, dtype=np.uint8)
            rec.update(_qc_rgb_tile(rgb, qc_config, histoqc_tile_overlap=overlap))
            if overlap is not None:
                overlaps.append(overlap)
        except Exception as e:
            n_read_fail += 1
            rec.update({"keep_tile": False, "failure_reason": f"read_or_qc_failed:{type(e).__name__}:{str(e)[:200]}"})
            if overlap is not None:
                rec["histoqc_tile_overlap"] = float(overlap)
                rec["histoqc_overlap_flag"] = bool(
                    float(overlap) < float((qc_config.get("histoqc") or {}).get("histoqc_tissue_overlap_min", 0.70))
                )
        records.append(rec)

    df = _ordered(pd.DataFrame(records), TILE_MANIFEST_COLUMNS)
    n_keep = int(df["keep_tile"].map(_truthy).sum()) if not df.empty else 0
    base_qc.update({
        "slide_status": "passed_full_validation" if n_keep > 0 else "failed_full_validation",
        "failure_reason": "" if n_keep > 0 else "no_tiles_passed_qc",
        "n_candidate_tiles": int(len(df)),
        "n_candidates_skipped_outside_histoqc_mask": int(n_skipped_outside_mask),
        "n_read_failures": int(n_read_fail),
        "n_keep_tiles": n_keep,
        "histoqc_mean_tile_overlap": float(np.mean(overlaps)) if overlaps else None,
    })
    if max_candidates_per_slide is not None and n_candidates > max_candidates_per_slide:
        base_qc["debug_truncated_after_max_candidates_per_slide"] = int(max_candidates_per_slide)
        base_qc["slide_status"] = "debug_truncated_passed_full_validation" if n_keep > 0 else "debug_truncated_failed_full_validation"
    return df, base_qc


def _publish_shared_tile_set(
    *,
    set_dir: Path,
    set_recipe_id: str,
    tile_df: pd.DataFrame,
    qc: Mapping[str, Any],
    shared: Mapping[str, Any],
    uuid: str,
) -> Dict[str, Any]:
    """Publish a per-slide tile set (manifest.parquet, slide_qc.json, tile_set_record.json) into the shared store
    atomically (temp dir + os.replace; an existing complete set is kept). Returns the pointer fields."""
    set_dir = Path(set_dir)
    tmp_dir = set_dir.parent / f".{set_recipe_id}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        tmp_manifest = tmp_dir / "manifest.parquet"
        tmp_slide_qc = tmp_dir / "slide_qc.json"
        tmp_record = tmp_dir / "tile_set_record.json"
        write_parquet_atomic(tmp_manifest, tile_df, overwrite=True)
        manifest_sha = sha256_path(tmp_manifest)
        # slide_qc.json holds the benchmark-agnostic QC dict
        write_json_atomic(tmp_slide_qc, dict(qc), overwrite=True)
        slide_qc_sha = sha256_path(tmp_slide_qc)
        record = {
            "status": "ok",
            "slide_file_uuid": uuid,
            "set_recipe_id": set_recipe_id,
            "set_recipe_components": dict(shared.get("set_recipe_components") or {}),
            "manifest_relpath": "manifest.parquet",
            "manifest_sha256": manifest_sha,
            "slide_qc_relpath": "slide_qc.json",
            "slide_qc_sha256": slide_qc_sha,
            "n_keep_tiles": int(qc.get("n_keep_tiles") or 0),
            "slide_status": qc.get("slide_status"),
            "created_utc": utc_now_iso(),
            "software_versions": software_versions(),
            "imaging_versions": imaging_versions(),
        }
        write_json_atomic(tmp_record, record, overwrite=True)
        set_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(tmp_dir, set_dir)
        except OSError:
            # Published by a concurrent worker: reuse it if complete
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if not shared_tile_set_is_complete(set_dir, expected_set_recipe_id=set_recipe_id):
                raise
            manifest_sha = sha256_path(set_dir / "manifest.parquet")
            slide_qc_sha = sha256_path(set_dir / "slide_qc.json")
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return {
        "reused_shared_tile_set": False,
        "set_dir": str(set_dir),
        "manifest_path": str(set_dir / "manifest.parquet"),
        "manifest_sha256": manifest_sha,
        "slide_qc_path": str(set_dir / "slide_qc.json"),
        "slide_qc_sha256": slide_qc_sha,
    }


def _process_one_slide(
    shared: Mapping[str, Any],
    row: Mapping[str, Any],
    prefetch_row: Mapping[str, Any],
    resolved_mask_path: Optional[Path],
    *,
    show_tile_progress: bool = False,
) -> Dict[str, Any]:
    """Build (or reuse) one slide's tile set in the shared store; run in a process pool.
    Returns the slide QC dict (with shared-store pointers on success)."""
    uuid = str(row["slide_file_uuid"])
    data_root = Path(shared["data_root"])
    modelgroup_id = shared["modelgroup_id"]
    set_recipe_id = shared["set_recipe_id"]
    histoqc_run_key = shared["histoqc_run_key"]
    force_recompute = bool(shared.get("force_recompute_shared_tiles", False))
    local_path = Path(str(row["local_path"]))

    set_dir = shared_tile_set_dir(data_root, uuid, set_recipe_id)
    manifest_path = set_dir / "manifest.parquet"
    slide_qc_path = set_dir / "slide_qc.json"

    def _failure_qc(reason: str, *, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        qc = {
            "slide_file_uuid": uuid,
            "file_name": row.get("file_name"),
            "case_id": row.get("case_id"),
            "case_uuid": row.get("case_uuid"),
            "project_id": row.get("project_id"),
            "local_path": str(local_path),
            "slide_status": "failed_full_validation",
            "failure_reason": reason,
            "technical_prefetch_status": prefetch_row.get("technical_prefetch_status", "not_run"),
            "set_recipe_id": set_recipe_id,
            "modelgroup_id": modelgroup_id,
            "histoqc_run_key": histoqc_run_key,
        }
        if extra:
            qc.update(dict(extra))
        return qc

    # Reuse a complete shared blob (cross-benchmark and cross-shard / resume).
    if not force_recompute and shared_tile_set_is_complete(set_dir, expected_set_recipe_id=set_recipe_id):
        try:
            loaded = read_json(slide_qc_path)
            qc = dict(loaded) if isinstance(loaded, Mapping) else _failure_qc("shared_slide_qc_unreadable")
        except Exception:
            qc = _failure_qc("shared_slide_qc_unreadable")
        qc.update({
            "reused_shared_tile_set": True,
            "set_dir": str(set_dir),
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_path(manifest_path),
            "slide_qc_path": str(slide_qc_path),
            "slide_qc_sha256": sha256_path(slide_qc_path),
        })
        return qc

    if not local_path.exists():
        return _failure_qc("downloaded_file_missing")

    try:
        slide = openslide.OpenSlide(str(local_path))
        try:
            # Load the slide's HistoQC mask (path resolved by the main process)
            if histoqc_run_key is not None:
                hq_mask, hq_scale_x, hq_scale_y, hq_mask_path = _load_histoqc_mask(
                    resolved_mask_path, tuple(slide.dimensions),
                )
            else:
                hq_mask, hq_scale_x, hq_scale_y, hq_mask_path = None, None, None, None
            tile_df, qc = _process_slide(
                row=row,
                prefetch_row=prefetch_row,
                slide=slide,
                qc_config=shared["qc_config"],
                modelgroup_id=modelgroup_id,
                set_recipe_id=set_recipe_id,
                histoqc_run_key=histoqc_run_key,
                modelgroup=shared["modelgroup"],
                mpp_min=float(shared["mpp_min"]),
                mpp_max=float(shared["mpp_max"]),
                mpp_anisotropy_max=float(shared["mpp_anisotropy_max"]),
                mpp_prefetch_abs_tol=float(shared["mpp_prefetch_abs_tol"]),
                max_candidates_per_slide=shared["max_candidates_per_slide"],
                histoqc_mask=hq_mask,
                histoqc_scale_x=hq_scale_x,
                histoqc_scale_y=hq_scale_y,
                histoqc_mask_path=hq_mask_path,
                show_tile_progress=show_tile_progress,
            )
        finally:
            slide.close()
        # Publish the tile set (also when no tile is kept)
        qc.update(_publish_shared_tile_set(
            set_dir=set_dir, set_recipe_id=set_recipe_id, tile_df=tile_df,
            qc=qc, shared=shared, uuid=uuid,
        ))
        return qc
    except Exception as e:
        return _failure_qc(
            f"openslide_or_tile_build_failed:{type(e).__name__}:{str(e)[:500]}",
            extra={"traceback": traceback.format_exc(limit=8)},
        )


def main(args) -> int:

    if openslide is None:
        raise RuntimeError("openslide-python is required for _10_build_tile_asset.py")
    if cv2 is None:
        raise RuntimeError("opencv-python/cv2 is required for deterministic Task B tile QC")
    if args.max_candidates_per_slide is not None and not args.force:
        raise RuntimeError("--max_candidates_per_slide is debug-only and requires --force")

    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    settings_dir = benchmark_root / "settings"
    cohort_dir = benchmark_root / "shared" / "cohort"
    reports_dir = benchmark_root / "reports"
    cache_root = data_root / "selected_slide_cache" / args.benchmark_id
    if not settings_dir.exists():
        raise FileNotFoundError(settings_dir)
    if not cohort_dir.exists():
        raise FileNotFoundError(cohort_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_mapping(settings_dir / "benchmark_config.yaml")
    links_path = settings_dir / "benchmark_asset_links.json"
    links = read_json(links_path) if links_path.exists() else {}
    if not isinstance(links, Mapping):
        raise RuntimeError("benchmark_asset_links.json must be a mapping when present")
    modelgroup_id = str(cfg.get("modelgroup_id") or links.get("modelgroup_id"))
    # Model group path rebuilt under the current root (not read from benchmark_asset_links.json)
    modelgroup_path = settings_dir / "modelgroups" / f"{modelgroup_id}.yaml"
    if not modelgroup_path.exists():
        raise FileNotFoundError(modelgroup_path)
    modelgroup = _load_mapping(modelgroup_path)
    _validate_modelgroup(modelgroup)
    if str(modelgroup.get("modelgroup_id")) != modelgroup_id:
        raise RuntimeError(f"modelgroup_id mismatch: config={modelgroup_id}, yaml={modelgroup.get('modelgroup_id')}")

    slide_manifest_path = cohort_dir / "slide_manifest.csv"
    if not slide_manifest_path.exists():
        raise FileNotFoundError(slide_manifest_path)
    slide_manifest_hash = sha256_path(slide_manifest_path)
    modelgroup_yaml_hash = sha256_path(modelgroup_path)
    qc_config = _read_qc_config(args.qc_config)
    if str(qc_config.get("qc_version")) != str(modelgroup.get("qc_version")):
        raise RuntimeError(f"QC version mismatch: modelgroup={modelgroup.get('qc_version')} qc_config={qc_config.get('qc_version')}")
    qc_config_hash = sha256_obj(qc_config)
    tiling_version = str(modelgroup.get("tiling_version"))

    # HistoQC asset (required; --no_histoqc for debugging gives a non-frozen asset)
    histoqc_asset_id: Optional[str] = None
    histoqc_asset_root: Optional[Path] = None
    histoqc_asset_manifest_path: Optional[Path] = None
    histoqc_asset_manifest_hash: Optional[str] = None
    histoqc_config_hash_recorded: Optional[str] = None
    histoqc_mask_path_by_uuid: Dict[str, Path] = {}
    # HistoQC run key (part of set_recipe_id; None with --no_histoqc)
    histoqc_run_key_value: Optional[str] = None
    if args.no_histoqc:
        if not args.force:
            raise RuntimeError("--no_histoqc is debug-only and requires --force")
        logger_msg = (
            "WARNING: --no_histoqc set; building the tile asset without HistoQC masks "
            "(non-frozen asset, no HistoQC overlap check)."
        )
        print(logger_msg)
        # Disable the HistoQC check in qc_config
        qc_config = dict(qc_config)
        qc_config["histoqc"] = dict(qc_config.get("histoqc") or {})
        qc_config["histoqc"]["histoqc_artifact_check_enabled"] = False
        qc_config_hash = sha256_obj(qc_config)
    else:
        histoqc_asset_id_raw = links.get("histoqc_asset_id")
        if not histoqc_asset_id_raw:
            raise RuntimeError(
                "Missing histoqc_asset_id in benchmark_asset_links.json. "
                "Run _9_run_histoqc.py first (or pass --no_histoqc for a debug build)."
            )
        histoqc_asset_id = str(histoqc_asset_id_raw)
        # HistoQC asset paths rebuilt under the current root from its id
        histoqc_asset_root = data_root / "processed_assets" / "histoqc" / histoqc_asset_id
        histoqc_asset_manifest_path = histoqc_asset_root / "histoqc_asset_manifest.json"
        if not histoqc_asset_root.exists() or not histoqc_asset_manifest_path.exists():
            raise FileNotFoundError(
                f"HistoQC asset missing: root={histoqc_asset_root}, "
                f"manifest={histoqc_asset_manifest_path}. Run _9_run_histoqc.py first."
            )
        # The HistoQC asset must be frozen_final
        ha_manifest = read_json(histoqc_asset_manifest_path)
        if not isinstance(ha_manifest, Mapping):
            raise RuntimeError(f"histoqc_asset_manifest.json must be a mapping: {histoqc_asset_manifest_path}")
        if ha_manifest.get("frozen_final") is not True or str(ha_manifest.get("status") or "") != "frozen_final":
            raise RuntimeError(
                f"HistoQC asset is not frozen_final (status={ha_manifest.get('status')!r}). "
                f"Re-run _9_run_histoqc.py without --limit_slides / --allow_partial."
            )
        if str(ha_manifest.get("benchmark_id") or "") != str(args.benchmark_id):
            raise RuntimeError(
                f"HistoQC asset benchmark_id={ha_manifest.get('benchmark_id')!r} != requested {args.benchmark_id!r}"
            )
        histoqc_asset_manifest_hash = sha256_path(histoqc_asset_manifest_path)
        histoqc_config_hash_recorded = ((ha_manifest.get("input_hashes") or {}).get("histoqc_config"))
        # Per-slide mask paths: referenced_shared_masks, else per_slide_records[*].mask_use_path
        ref_masks = ha_manifest.get("referenced_shared_masks")
        if isinstance(ref_masks, Mapping):
            for u, info in ref_masks.items():
                p = info.get("path") if isinstance(info, Mapping) else None
                if u and p:
                    histoqc_mask_path_by_uuid[str(u)] = Path(str(p))
                if histoqc_run_key_value is None and isinstance(info, Mapping) and info.get("run_key"):
                    histoqc_run_key_value = str(info.get("run_key"))
        for r in (ha_manifest.get("per_slide_records") or []):
            if not isinstance(r, Mapping):
                continue
            u = str(r.get("slide_file_uuid") or "")
            p = r.get("mask_use_path")
            if u and p and u not in histoqc_mask_path_by_uuid:
                histoqc_mask_path_by_uuid[u] = Path(str(p))
            if histoqc_run_key_value is None and r.get("histoqc_run_key"):
                histoqc_run_key_value = str(r.get("histoqc_run_key"))
        if not histoqc_run_key_value:
            raise RuntimeError(
                "Could not resolve histoqc_run_key from the HistoQC asset manifest "
                f"({histoqc_asset_manifest_path}); it keys the shared tile set."
            )

    tile_asset_id = _deterministic_tile_asset_id(
        benchmark_id=args.benchmark_id,
        modelgroup_id=modelgroup_id,
        slide_manifest_hash=slide_manifest_hash,
        modelgroup_yaml_hash=modelgroup_yaml_hash,
        qc_config_hash=qc_config_hash,
        tiling_version=tiling_version,
        histoqc_asset_id=histoqc_asset_id,
    )
    # set_recipe_id keys the shared per-slide tile sets (tile_asset_id: per-benchmark manifest)
    histoqc_run_key_for_recipe = "<no_histoqc_debug>" if args.no_histoqc else str(histoqc_run_key_value)
    set_recipe_id = tile_set_recipe_id(
        modelgroup_yaml_hash=modelgroup_yaml_hash,
        qc_config_hash=qc_config_hash,
        histoqc_run_key=histoqc_run_key_for_recipe,
    )
    set_recipe_components = {
        "modelgroup_yaml_hash": modelgroup_yaml_hash,
        "qc_config_hash": qc_config_hash,
        "histoqc_run_key": histoqc_run_key_for_recipe,
        "tiling_toolchain_version": str(modelgroup.get("tiling_toolchain_version")),
    }
    # MPP bounds from the model group (--mpp_* debug overrides require --force)
    mpp_overrides = {
        k: getattr(args, k)
        for k in ("mpp_min", "mpp_max", "mpp_anisotropy_max", "mpp_prefetch_abs_tol")
        if getattr(args, k) is not None
    }
    if mpp_overrides and not args.force:
        raise RuntimeError(f"--mpp_* overrides are debug-only and require --force: {sorted(mpp_overrides)}")
    mpp_min = float(mpp_overrides.get("mpp_min", modelgroup["mpp_min"]))
    mpp_max = float(mpp_overrides.get("mpp_max", modelgroup["mpp_max"]))
    mpp_anisotropy_max = float(mpp_overrides.get("mpp_anisotropy_max", modelgroup["mpp_anisotropy_max"]))
    mpp_prefetch_abs_tol = float(mpp_overrides.get("mpp_prefetch_abs_tol", modelgroup["mpp_prefetch_abs_tol"]))
    tile_root = data_root / "processed_assets" / "tiles" / tile_asset_id
    if args.force and args.resume:
        raise RuntimeError(
            "--force and --resume are mutually exclusive: --force wipes and rebuilds the "
            "tile asset from scratch, --resume continues an existing (partial) build."
        )
    if tile_root.exists() and not (args.force or args.resume):
        raise FileExistsError(
            f"Tile asset already exists; use --force to rebuild from scratch or --resume to continue: {tile_root}"
        )
    if tile_root.exists() and (args.force or args.resume) and not args.force_cascade:
        downstream = [benchmark_root / "shared" / "tiles", cohort_dir / "final_case_manifest.csv"]
        existing = [p for p in downstream if p.exists()]
        if existing:
            verb = "rebuild" if args.force else "resume"
            raise RuntimeError(f"Refusing to {verb} tile asset while downstream case index/cohort exists: {existing}. Use --force_cascade if intentional.")
    if tile_root.exists() and args.force:
        # Remove the per-benchmark dir only (shared tile sets: see --force_recompute_shared_tiles)
        shutil.rmtree(tile_root)
    tile_root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        # Remove the previous manifest (it would be included in output_hashes)
        stale_manifest = tile_root / "tile_asset_manifest.json"
        if stale_manifest.exists():
            stale_manifest.unlink()

    download_df = _load_download_manifest(cache_root)
    if args.limit_slides is not None:
        download_df = download_df.sort_values(["project_id", "case_id", "slide_file_uuid"]).head(args.limit_slides).copy()
    prefetch_map, prefetch_meta = _prefetch_lookup(cohort_dir, allow_missing_prefetch=bool(args.allow_missing_prefetch))

    slide_rows = download_df.sort_values(["project_id", "case_id", "slide_file_uuid"]).to_dict("records")

    # Per-run context passed to each slide worker (small and picklable)
    overwrite_outputs = bool(args.force or args.resume)
    shared_ctx: Dict[str, Any] = {
        "data_root": str(data_root),
        "modelgroup_id": modelgroup_id,
        "set_recipe_id": set_recipe_id,
        "set_recipe_components": dict(set_recipe_components),
        "histoqc_run_key": None if args.no_histoqc else str(histoqc_run_key_value),
        "modelgroup": dict(modelgroup),
        "qc_config": dict(qc_config),
        "mpp_min": mpp_min,
        "mpp_max": mpp_max,
        "mpp_anisotropy_max": mpp_anisotropy_max,
        "mpp_prefetch_abs_tol": mpp_prefetch_abs_tol,
        "max_candidates_per_slide": args.max_candidates_per_slide,
        "force_recompute_shared_tiles": bool(args.force_recompute_shared_tiles),
    }

    if int(args.max_workers) < 1:
        raise RuntimeError("--max_workers must be >= 1")
    max_workers = max(1, min(int(args.max_workers), len(slide_rows) or 1))

    print(f"[tile_asset] benchmark={args.benchmark_id} modelgroup={modelgroup_id} tile_asset_id={tile_asset_id}", flush=True)
    print(f"[tile_asset] output root: {tile_root}", flush=True)
    print(
        f"[tile_asset] processing {len(slide_rows)} downloaded slide(s) "
        f"with {max_workers} worker(s){' [resume]' if args.resume else ''}",
        flush=True,
    )

    # Slides to process, with their HistoQC mask path
    qc_by_uuid: Dict[str, Dict[str, Any]] = {}
    todo: List[Tuple[Dict[str, Any], Dict[str, Any], Optional[Path]]] = []
    n_resumed = 0
    # All slides are submitted; those already in the shared store are reused by the worker
    for row in slide_rows:
        uuid = str(row["slide_file_uuid"])
        prefetch_row = prefetch_map.get(uuid, _sentinel_prefetch(uuid))
        if histoqc_asset_id is not None:
            resolved_mask_path = _resolve_histoqc_mask_path(
                histoqc_mask_path_by_uuid, histoqc_asset_root, uuid, Path(str(row["local_path"])).stem,
            )
        else:
            resolved_mask_path = None
        todo.append((row, prefetch_row, resolved_mask_path))

    # Progress-bar counters
    passed_statuses = {"passed_full_validation", "debug_truncated_passed_full_validation"}
    pbar = tqdm(total=len(slide_rows), desc="Building tile asset", unit="slide")
    pbar.update(n_resumed)
    n_keep_running = sum(int(q.get("n_keep_tiles") or 0) for q in qc_by_uuid.values())
    n_fail_running = sum(1 for q in qc_by_uuid.values() if q.get("slide_status") not in passed_statuses)
    pbar.set_postfix_str(f"kept={n_keep_running} fail={n_fail_running}")

    def _account(slide_uuid: str, qc: Dict[str, Any]) -> None:
        nonlocal n_keep_running, n_fail_running
        qc_by_uuid[slide_uuid] = qc
        n_keep_running += int(qc.get("n_keep_tiles") or 0)
        if qc.get("slide_status") not in passed_statuses:
            n_fail_running += 1
        pbar.update(1)
        pbar.set_postfix_str(f"slide={slide_uuid[:8]} kept={n_keep_running} fail={n_fail_running}")

    if not todo:
        pass  # no slide to process
    elif max_workers == 1:
        # Serial path (with the per-slide tile progress bar)
        for row, prefetch_row, resolved_mask_path in todo:
            qc = _process_one_slide(shared_ctx, row, prefetch_row, resolved_mask_path, show_tile_progress=True)
            _account(str(row["slide_file_uuid"]), qc)
    else:
        # Parallel path: one slide per worker process (no per-tile progress bar)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            fut_to_uuid = {
                ex.submit(
                    _process_one_slide, shared_ctx, row, prefetch_row, resolved_mask_path,
                    show_tile_progress=False,
                ): str(row["slide_file_uuid"])
                for row, prefetch_row, resolved_mask_path in todo
            }
            for fut in as_completed(fut_to_uuid):
                _account(fut_to_uuid[fut], fut.result())
    pbar.close()

    # Aggregate in slide order (independent of completion order)
    all_qc: List[Dict[str, Any]] = [qc_by_uuid[str(r["slide_file_uuid"])] for r in slide_rows]
    # Add the benchmark ids to the QC dicts (for slide_qc_summary)
    for q in all_qc:
        q["benchmark_id"] = args.benchmark_id
        q["tile_asset_id"] = tile_asset_id
        q["histoqc_asset_id"] = histoqc_asset_id
    n_reused_shared = sum(1 for q in all_qc if q.get("reused_shared_tile_set"))
    # Pointers into the shared tile store for slides with a manifest (codec_recipe_id / h5_* are filled by _15)
    referenced_shared_tiles: Dict[str, Dict[str, Any]] = {
        str(q["slide_file_uuid"]): {
            "set_recipe_id": set_recipe_id,
            "codec_recipe_id": None,
            "manifest_path": str(q.get("manifest_path")),
            "manifest_sha256": q.get("manifest_sha256"),
            "slide_qc_path": q.get("slide_qc_path"),
            "slide_qc_sha256": q.get("slide_qc_sha256"),
            "h5_path": None,
            "h5_sha256": None,
        }
        for q in all_qc
        if q.get("slide_file_uuid") and q.get("manifest_path")
    }
    n_keep_total = sum(int(q.get("n_keep_tiles") or 0) for q in all_qc)
    failures: List[Dict[str, Any]] = [q for q in all_qc if q.get("slide_status") not in passed_statuses]
    warnings: List[Dict[str, Any]] = [
        {
            "slide_file_uuid": str(q.get("slide_file_uuid")),
            "warning": "prefetch_full_mpp_discrepancy_full_openslide_used",
            "reason": q.get("prefetch_full_discrepancy_reason"),
            "prefetch_mpp_x": q.get("prefetch_mpp_x"),
            "prefetch_mpp_y": q.get("prefetch_mpp_y"),
            "full_mpp_x": q.get("full_openslide_mpp_x"),
            "full_mpp_y": q.get("full_openslide_mpp_y"),
        }
        for q in all_qc
        if q.get("prefetch_full_mpp_consistency") == "discrepant"
    ]

    if not all_qc:
        raise RuntimeError("No slides were processed")
    qc_df = _ordered(pd.DataFrame(all_qc), SLIDE_QC_SUMMARY_COLUMNS)
    qc_summary_path = tile_root / "slide_qc_summary.parquet"
    write_parquet_atomic(qc_summary_path, qc_df, overwrite=overwrite_outputs)

    passed = qc_df["slide_status"].astype(str).isin(["passed_full_validation", "debug_truncated_passed_full_validation"])
    # Resample directions of the slides that passed validation
    passed_qc = qc_df.loc[passed]
    _rdir = passed_qc["resample_direction"].astype(str) if "resample_direction" in passed_qc.columns else pd.Series([], dtype=str)
    resample_counts = _rdir.value_counts().to_dict()
    n_keep_upsampled = (
        int(pd.to_numeric(passed_qc.loc[_rdir.eq("upsample"), "n_keep_tiles"], errors="coerce").fillna(0).sum())
        if "resample_direction" in passed_qc.columns else 0
    )
    # Resample directions per project
    per_project_resample: Dict[str, Dict[str, int]] = {}
    if (not passed_qc.empty) and {"project_id", "resample_direction"}.issubset(passed_qc.columns):
        for proj, pg in passed_qc.groupby(passed_qc["project_id"].astype(str)):
            d = pg["resample_direction"].astype(str).value_counts().to_dict()
            per_project_resample[str(proj)] = {
                "n_slides_native_resolution": int(d.get("none", 0)),
                "n_slides_downsampled": int(d.get("downsample", 0)),
                "n_slides_upsampled": int(d.get("upsample", 0)),
            }
    summary = {
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": modelgroup_id,
        "tile_asset_id": tile_asset_id,
        "n_downloaded_slides": int(len(qc_df)),
        "n_slides_passed_full_validation": int(passed.sum()),
        "n_slides_failed_full_validation": int((~passed).sum()),
        "n_keep_tiles": int(n_keep_total),
        "n_slides_upsampled": int(resample_counts.get("upsample", 0)),
        "n_slides_native_resolution": int(resample_counts.get("none", 0)),
        "n_slides_downsampled": int(resample_counts.get("downsample", 0)),
        "n_keep_tiles_from_upsampled_slides": int(n_keep_upsampled),
        "per_project_resample": per_project_resample,
        "n_prefetch_full_mpp_discrepancies": int(len(warnings)),
        "n_reused_shared_tile_sets": int(n_reused_shared),
        "debug_max_candidates_per_slide": args.max_candidates_per_slide,
    }

    asset_link_update = _update_asset_links(settings_dir, {
        "tile_asset_id": tile_asset_id,
        "tile_asset_root": str(tile_root),
        "tile_asset_manifest_path": str(tile_root / "tile_asset_manifest.json"),
        "tile_set_recipe_id": set_recipe_id,
        "tiles_shared_store_root": str(shared_tiles_store_root(data_root)),
        "case_tile_index_path": str(benchmark_root / "shared" / "tiles" / "case_tile_index.parquet"),
        "final_case_manifest_path": str(cohort_dir / "final_case_manifest.csv"),
    }, force=bool(args.force), force_cascade=bool(args.force_cascade))

    official_final_tile_asset = (
        args.limit_slides is None
        and args.max_candidates_per_slide is None
        and not args.no_histoqc
        and not mpp_overrides
        and histoqc_asset_id is not None
    )
    nonfinal_reasons = []
    if args.limit_slides is not None:
        nonfinal_reasons.append("limit_slides")
    if args.max_candidates_per_slide is not None:
        nonfinal_reasons.append("max_candidates_per_slide")
    if args.no_histoqc:
        nonfinal_reasons.append("no_histoqc")
    if mpp_overrides:
        nonfinal_reasons.append("mpp_override")
    manifest = manifest_header(Path(__file__).name, vars(args))
    manifest.update({
        "frozen_final": bool(official_final_tile_asset),
        "status": "frozen_final" if official_final_tile_asset else "debug_nonfinal",
        "nonfinal_reason": "" if official_final_tile_asset else ";".join(nonfinal_reasons) or "unknown",
        "tile_asset_id": tile_asset_id,
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": modelgroup_id,
        "tile_asset_root": str(tile_root),
        "tile_set_recipe_id": set_recipe_id,
        "tile_set_recipe_components": dict(set_recipe_components),
        "tiles_shared_store_root": str(shared_tiles_store_root(data_root)),
        "referenced_shared_tiles": referenced_shared_tiles,
        "histoqc_asset_id": histoqc_asset_id,
        "histoqc_asset_root": str(histoqc_asset_root) if histoqc_asset_root is not None else None,
        "input_paths": {
            "slide_manifest_csv": str(slide_manifest_path),
            "download_manifest_tsv": str(cache_root / "manifests" / "selected_slide_download_manifest.tsv"),
            "prefetch_manifest": prefetch_meta.get("path"),
            "modelgroup_yaml": str(modelgroup_path),
            "qc_config_override": str(args.qc_config) if args.qc_config else None,
            "benchmark_config": str(settings_dir / "benchmark_config.yaml"),
            "histoqc_asset_manifest": str(histoqc_asset_manifest_path) if histoqc_asset_manifest_path is not None else None,
        },
        "input_hashes": {
            "slide_manifest_csv": slide_manifest_hash,
            "download_manifest_tsv": sha256_path(cache_root / "manifests" / "selected_slide_download_manifest.tsv"),
            "prefetch_manifest": prefetch_meta.get("sha256"),
            "modelgroup_yaml": modelgroup_yaml_hash,
            "qc_config_hash": qc_config_hash,
            "benchmark_config": sha256_path(settings_dir / "benchmark_config.yaml"),
            "histoqc_asset_manifest": histoqc_asset_manifest_hash,
            "histoqc_config_hash": histoqc_config_hash_recorded,
        },
        "deterministic_tile_asset_id_components": {
            "benchmark_id": args.benchmark_id,
            "modelgroup_id": modelgroup_id,
            "slide_manifest_hash": slide_manifest_hash,
            "modelgroup_yaml_hash": modelgroup_yaml_hash,
            "qc_config_hash": qc_config_hash,
            "tiling_version": tiling_version,
            "histoqc_asset_id": histoqc_asset_id or "<no_histoqc_debug>",
        },
        "modelgroup": dict(modelgroup),
        "qc_config": qc_config,
        "mpp_policy": {
            "final_mpp_source": "full_openslide",
            "mpp_min": mpp_min,
            "mpp_max": mpp_max,
            "mpp_anisotropy_max_ratio": mpp_anisotropy_max,
            "mpp_source": "model_group_yaml" if not mpp_overrides else "cli_debug_override",
            "mpp_debug_overrides": dict(mpp_overrides),
            "objective_power_fallback_allowed": False,
            "prefetch_full_abs_tolerance_um": mpp_prefetch_abs_tol,
            "prefetch_not_run_sentinel": "not_run/nulls/prefetch_missing_or_not_run",
        },
        "color_policy": {
            "read_region": "OpenSlide read_region RGBA",
            "alpha": "composite alpha on white background",
            "rgb": "convert_to_rgb",
            "icc_profile_policy": "ignore_embedded_icc_use_openslide_returned_rgb",
            "stain_normalization": "none",
            "augmentation": "none",
        },
        "summary": summary,
        "prefetch_full_mpp_warnings": warnings,
        "failed_slide_qc_records": failures,
        "asset_link_update": asset_link_update,
        "output_hashes": collect_output_hashes(tile_root),
    })
    write_json_atomic(tile_root / "tile_asset_manifest.json", manifest, overwrite=overwrite_outputs)
    _write_report(reports_dir / "tile_asset_build_report.md", summary, overwrite=overwrite_outputs)

    if summary["n_slides_passed_full_validation"] <= 0:
        raise RuntimeError("No slides passed full OpenSlide/tile QC validation")
    print(json.dumps({"status": "ok", **summary, "tile_asset_manifest": str(tile_root / "tile_asset_manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Build the frozen TCGA Task B tile asset from downloaded diagnostic slides.")

    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help="Benchmark ID whose downloaded slides are tiled.")

    ap.add_argument("--qc_config", type=Path,
                    default=None,
                    help="Optional YAML/JSON file overriding the default tile QC configuration.")

    # Debug overrides of the model group MPP bounds (require --force; non-frozen asset)
    ap.add_argument("--mpp_min", type=float, default=None,
                    help="Debug override of the model group's minimum accepted microns-per-pixel (requires --force).")

    ap.add_argument("--mpp_max", type=float, default=None,
                    help="Debug override of the model group's maximum accepted microns-per-pixel (requires --force).")

    ap.add_argument("--mpp_anisotropy_max", type=float, default=None,
                    help="Debug override of the model group's maximum x/y MPP anisotropy ratio (requires --force).")

    ap.add_argument("--mpp_prefetch_abs_tol", type=float, default=None,
                    help="Debug override of the model group's prefetch-vs-full MPP tolerance (requires --force).")

    ap.add_argument("--allow_missing_prefetch", action="store_true",
                    help="Use explicit not_run prefetch sentinels.")

    ap.add_argument("--limit_slides", type=int,
                    default=None,
                    help="Debug only: process only the first N slides.")

    ap.add_argument("--max_candidates_per_slide", type=int,
                    default=None,
                    help="Debug only: cap candidate tiles per slide; invalid for a final benchmark.")

    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing tile asset.")

    ap.add_argument("--force_cascade", action="store_true",
                    help="Required with --force if downstream case tile indexes already exist.")

    ap.add_argument("--force_recompute_shared_tiles", action="store_true",
                    help=(
                        "Debug only: recompute and overwrite the shared per-slide tile sets in "
                        "processed_assets/tiles/_store/ even if they exist."
                    ))

    ap.add_argument("--no_histoqc", action="store_true",
                    help=(
                        "Debug only (requires --force): build the tile asset without HistoQC masks "
                        "(non-frozen asset, no HistoQC overlap check)."
                    ))

    ap.add_argument("--max_workers", type=int,
                    default=DEFAULT_MAX_WORKERS,
                    help=(
                        "Number of slides tiled in parallel (worker processes); 1 shows the per-slide tile progress "
                        f"bar. Default: {DEFAULT_MAX_WORKERS} (os.cpu_count()-2)."
                    ))

    ap.add_argument("--resume", action="store_true",
                    help=(
                        "Continue an interrupted build of the same tile_asset_id (slides already in the shared "
                        "store are reused). Mutually exclusive with --force."
                    ))

    args = ap.parse_args()
    raise SystemExit(main(args))
