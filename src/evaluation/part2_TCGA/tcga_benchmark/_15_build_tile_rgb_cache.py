#!/usr/bin/env python3
"""
Build the RGB tile cache of a TCGA Task B benchmark: all kept tiles of each final-cohort slide, encoded (jpeg by
default) in one HDF5 file per slide, shared by all benchmarks. Sharded runs are consolidated with --finalize_only.

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data / processed_assets / tiles /
    - _store/by_uuid/<slide_file_uuid>/<set_recipe_id>/pixels/<codec_recipe_id>/: tiles.h5, tile_pixels_record.json
    - <tile_asset_id>/tile_rgb_cache/: tile_rgb_cache_manifest.json, shard_status/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path

from _taskb_common import (
    _truthy_value,
    composite_rgba_on_white,
    imaging_versions,
    manifest_header,
    read_json,
    read_shard_records,
    read_yaml,
    sha256_obj,
    sha256_path,
    shard_status_path,
    shared_tile_pixels_dir,
    shared_tile_pixels_is_complete,
    software_versions,
    tile_codec_recipe_id,
    utc_now_iso,
    write_json_atomic,
)

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import openslide  # type: ignore
except Exception:  # pragma: no cover
    openslide = None  # type: ignore


DEFAULT_PNG_COMPRESSION = 6  # cv2 PNG level (0-9); only used when --cache_format png.
DEFAULT_CACHE_FORMAT = "jpeg"
DEFAULT_QUALITY = 95  # lossy quality for jpeg/webp

# `format` attr of each per-slide HDF5 (must match CACHE_FORMAT_LABELS_SUPPORTED in _0_extract_encoder_features.py)
CACHE_FORMAT_LABEL_PNG = "png_rgb_vlen_uint8_v1"        # lossless PNG
CACHE_FORMAT_LABEL_JPEG = "jpeg444_rgb_vlen_uint8_v1"   # lossy JPEG, no chroma subsampling
CACHE_FORMAT_LABEL_WEBP = "webp_rgb_vlen_uint8_v1"      # lossy WebP
_FORMAT_LABELS = {
    "png": CACHE_FORMAT_LABEL_PNG,
    "jpeg": CACHE_FORMAT_LABEL_JPEG,
    "webp": CACHE_FORMAT_LABEL_WEBP,
}


def _format_label(cache_format: str) -> str:
    """Provenance label for a codec (pure; no cv2 needed)."""
    try:
        return _FORMAT_LABELS[str(cache_format).lower()]
    except KeyError:
        raise RuntimeError(f"unsupported cache_format {cache_format!r} (choose png/jpeg/webp)")


def _encode_params(cache_format: str, *, quality: int, png_compression: int) -> Tuple[str, List[int], str]:
    """Return (cv2 imencode extension, params, provenance label) for a codec (jpeg with 4:4:4 chroma)."""
    fmt = str(cache_format).lower()
    if fmt == "png":
        return ".png", [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)], CACHE_FORMAT_LABEL_PNG
    if fmt == "jpeg":
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        sf = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
        sf444 = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444", None)
        if sf is not None and sf444 is not None:  # older cv2: default 4:2:0
            params += [int(sf), int(sf444)]
        return ".jpg", params, CACHE_FORMAT_LABEL_JPEG
    if fmt == "webp":
        return ".webp", [int(cv2.IMWRITE_WEBP_QUALITY), int(quality)], CACHE_FORMAT_LABEL_WEBP
    raise RuntimeError(f"unsupported cache_format {cache_format!r} (choose png/jpeg/webp)")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _resolve_tile_asset(data_root: Path, settings_dir: Path) -> Tuple[str, Path, Path]:
    links_path = settings_dir / "benchmark_asset_links.json"
    if not links_path.exists():
        raise FileNotFoundError(links_path)
    links = read_json(links_path)
    if not isinstance(links, Mapping):
        raise RuntimeError("benchmark_asset_links.json must be a mapping")
    tile_asset_id = str(links.get("tile_asset_id") or "")
    if not tile_asset_id:
        raise RuntimeError("tile_asset_id missing. Run _10_build_tile_asset.py first.")
    tile_root = Path(links.get("tile_asset_root") or data_root / "processed_assets" / "tiles" / tile_asset_id)
    if not tile_root.exists():
        raise FileNotFoundError(tile_root)
    set_recipe_id = str(links.get("tile_set_recipe_id") or "")
    if not set_recipe_id:
        raise RuntimeError("tile_set_recipe_id missing in benchmark_asset_links.json. Re-run _10_build_tile_asset.py.")
    return tile_asset_id, tile_root, set_recipe_id, links_path


def _resolve_codec(settings_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Tile codec from benchmark_config.yaml (tile_recipe.codec), with optional debug CLI overrides."""
    cfg_path = settings_dir / "benchmark_config.yaml"
    cfg = read_yaml(cfg_path) if cfg_path.exists() else {}
    codec = dict(((cfg or {}).get("tile_recipe") or {}).get("codec") or {})
    codec.setdefault("format", DEFAULT_CACHE_FORMAT)
    codec.setdefault("quality", DEFAULT_QUALITY)
    codec.setdefault("chroma", "444")
    codec.setdefault("png_level", DEFAULT_PNG_COMPRESSION)
    if args.cache_format is not None:
        codec["format"] = str(args.cache_format)
    if args.quality is not None:
        codec["quality"] = int(args.quality)
    if args.png_compression is not None:
        codec["png_level"] = int(args.png_compression)
    if getattr(args, "chroma", None) is not None:
        codec["chroma"] = str(args.chroma)
    codec["format"] = str(codec["format"]).lower()
    codec["quality"] = int(codec["quality"])
    codec["png_level"] = int(codec["png_level"])
    codec["chroma"] = str(codec["chroma"])
    if codec["format"] not in _FORMAT_LABELS:
        raise RuntimeError(f"unsupported tile_recipe codec format {codec['format']!r} (choose png/jpeg/webp)")
    return codec


def _load_case_tile_index(benchmark_root: Path) -> Tuple[pd.DataFrame, Path, Path]:
    tiles_dir = benchmark_root / "shared" / "tiles"
    idx_path = tiles_dir / "case_tile_index.parquet"
    mf_path = tiles_dir / "case_tile_index_manifest.json"
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    if not mf_path.exists():
        raise FileNotFoundError(mf_path)
    df = pd.read_parquet(idx_path)
    required = [
        "tile_id", "slide_file_uuid", "case_id", "project_id",
        "top_left_x_l0", "top_left_y_l0",
        "read_window_px_l0_x", "read_window_px_l0_y",
        "tile_asset_id",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"case_tile_index.parquet missing required columns: {missing}")
    return df, idx_path, mf_path


def _full_kept_tiles(manifest_path: Path) -> pd.DataFrame:
    """All kept (QC-passed) tiles of one slide, from its shared tile manifest (not only the case-capped selection)."""
    df = pd.read_parquet(manifest_path)
    need = ["tile_id", "top_left_x_l0", "top_left_y_l0", "read_window_px_l0_x", "read_window_px_l0_y", "keep_tile"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"{manifest_path} missing required tile columns: {missing}")
    return df.loc[df["keep_tile"].fillna(False).astype(bool)].copy()


def _load_download_manifest(cache_root: Path) -> pd.DataFrame:
    p = cache_root / "manifests" / "selected_slide_download_manifest.tsv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    if not {"slide_file_uuid", "local_path"}.issubset(df.columns):
        raise RuntimeError("download manifest missing local_path/slide_file_uuid")
    return df[["slide_file_uuid", "local_path"]].drop_duplicates("slide_file_uuid")


def _shard_filter(uuids: Sequence[str], *, shard_index: Optional[int], total_shards: Optional[int]) -> List[str]:
    if shard_index is None and total_shards is None:
        return list(uuids)
    if shard_index is None or total_shards is None:
        raise RuntimeError("--shard_index and --total_shards must be supplied together")
    if total_shards <= 0:
        raise RuntimeError("--total_shards must be > 0")
    if not (0 <= shard_index < total_shards):
        raise RuntimeError(f"--shard_index out of range: {shard_index} not in [0, {total_shards})")
    return list(uuids[shard_index::total_shards])


def _write_per_slide_h5(
    out_path: Path,
    *,
    n_tiles: int,
    tile_stream: Iterable[Tuple[str, bytes]],
    set_recipe_id: str,
    codec_recipe_id: str,
    tile_asset_id: str,
    slide_file_uuid: str,
    cache_format: str,
    quality: int,
    png_compression: int,
    chroma: str,
) -> int:
    """Write one per-slide HDF5 cache, streaming the encoded tiles; returns the number of tiles written (= n_tiles).
    Datasets: tile_ids (utf-8) and tiles_png (encoded RGB tiles, vlen uint8, any codec); attrs: set_recipe_id,
    codec_recipe_id, tile_asset_id, slide_file_uuid, n_tiles, format and codec settings."""
    if h5py is None:
        raise RuntimeError("h5py is required to write the tile RGB cache")
    if int(n_tiles) < 0:
        raise RuntimeError(f"n_tiles must be >= 0, got {n_tiles}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    vlen_bytes_dt = h5py.vlen_dtype(np.uint8)
    tile_ids: List[str] = []
    with h5py.File(out_path, "w", libver="latest") as f:
        ds = f.create_dataset("tiles_png", (int(n_tiles),), dtype=vlen_bytes_dt)
        n_written = 0
        for tile_id, blob in tile_stream:
            if n_written >= n_tiles:
                raise RuntimeError(f"tile_stream yielded more than the declared n_tiles={n_tiles}")
            ds[n_written] = np.frombuffer(blob, dtype=np.uint8)
            tile_ids.append(str(tile_id))
            n_written += 1
        if n_written != int(n_tiles):
            raise RuntimeError(f"tile_stream yielded {n_written} tiles but n_tiles={n_tiles} was declared")
        # Identity: (set_recipe_id, codec_recipe_id); tile_asset_id = benchmark that first built it
        f.create_dataset("tile_ids", data=np.asarray(tile_ids, dtype=object), dtype=str_dt)
        f.attrs["set_recipe_id"] = str(set_recipe_id)
        f.attrs["codec_recipe_id"] = str(codec_recipe_id)
        f.attrs["tile_asset_id"] = str(tile_asset_id)
        f.attrs["slide_file_uuid"] = str(slide_file_uuid)
        f.attrs["n_tiles"] = int(n_written)
        f.attrs["format"] = _format_label(cache_format)
        f.attrs["cache_format"] = str(cache_format).lower()
        f.attrs["quality"] = int(quality)
        f.attrs["chroma"] = str(chroma)
        f.attrs["built_by"] = "_15_build_tile_rgb_cache.py"
        f.attrs["built_utc"] = utc_now_iso()
        f.attrs["png_compression"] = int(png_compression)
    # fsync the file (limits write-back memory across slides; best-effort)
    try:
        with open(out_path, "rb+") as _fh:
            os.fsync(_fh.fileno())
    except OSError:
        pass
    return n_written


def _process_one_slide(
    *,
    slide_file_uuid: str,
    local_path: Path,
    tiles_for_slide: pd.DataFrame,
    data_root: Path,
    set_recipe_id: str,
    codec_recipe_id: str,
    codec_recipe: Mapping[str, Any],
    tile_asset_id: str,
    force: bool = False,
    verify_sha: bool = False,
) -> Dict[str, Any]:
    if openslide is None:
        raise RuntimeError("openslide-python is required for _15_build_tile_rgb_cache.py")
    if cv2 is None:
        raise RuntimeError("opencv-python/cv2 is required for tile encoding")
    cache_format = str(codec_recipe["format"])
    quality = int(codec_recipe["quality"])
    png_compression = int(codec_recipe["png_level"])
    chroma = str(codec_recipe["chroma"])
    imencode_ext, imencode_params, _ = _encode_params(cache_format, quality=quality, png_compression=png_compression)

    pixels_dir = shared_tile_pixels_dir(data_root, slide_file_uuid, set_recipe_id, codec_recipe_id)
    out_path = pixels_dir / "tiles.h5"

    # Reuse a complete shared pixel blob (file-size check, or sha256 with verify_sha)
    if not force and shared_tile_pixels_is_complete(
        pixels_dir,
        expected_set_recipe_id=set_recipe_id,
        expected_codec_recipe_id=codec_recipe_id,
        verify_sha=verify_sha,
    ):
        rec = read_json(pixels_dir / "tile_pixels_record.json")
        rec = dict(rec) if isinstance(rec, Mapping) else {}
        return {
            "slide_file_uuid": slide_file_uuid,
            "status": "reused_shared",
            "failure_reason": "",
            "out_path": str(out_path),
            "n_tiles": int(rec.get("n_tiles") or 0),
            "h5_sha256": str(rec.get("h5_sha256") or sha256_path(out_path)),
            "h5_bytes": int(out_path.stat().st_size),
            "elapsed_seconds": 0.0,
        }

    if not local_path.exists():
        return {
            "slide_file_uuid": slide_file_uuid,
            "status": "failed_no_downloaded_file",
            "failure_reason": f"missing local_path: {local_path}",
            "out_path": str(out_path),
            "n_tiles": 0,
            "elapsed_seconds": 0.0,
        }

    # Needed columns, sorted by position
    need_cols = [
        "tile_id", "top_left_x_l0", "top_left_y_l0",
        "read_window_px_l0_x", "read_window_px_l0_y",
    ]
    tiles_for_slide = (
        tiles_for_slide.sort_values(["top_left_y_l0", "top_left_x_l0", "tile_id"])[need_cols]
        .reset_index(drop=True)
    )
    n_tiles = int(len(tiles_for_slide))

    t0 = time.time()

    # Write tiles.h5 in a temp dir, then publish it atomically with os.replace (nothing is published if a tile
    # fails; a complete blob published concurrently is reused)
    pixels_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = pixels_dir.parent / f".{codec_recipe_id}.tmp.{os.getpid()}"
    slide = None
    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        slide = openslide.OpenSlide(str(local_path))

        def _tile_stream():
            for row in tiles_for_slide.itertuples(index=False):
                x = int(float(row.top_left_x_l0))
                y = int(float(row.top_left_y_l0))
                wx = int(float(row.read_window_px_l0_x))
                wy = int(float(row.read_window_px_l0_y))
                try:
                    rgba = slide.read_region((x, y), 0, (wx, wy))
                    rgb_pil = composite_rgba_on_white(rgba)
                    rgb = np.asarray(rgb_pil, dtype=np.uint8)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    ok, buf = cv2.imencode(imencode_ext, bgr, imencode_params)
                    if not ok:
                        raise RuntimeError("cv2.imencode returned False")
                except Exception as e:
                    # Abort the slide on the first tile error
                    raise RuntimeError(
                        f"Tile read/encode failed for slide={slide_file_uuid}, "
                        f"tile_id={row.tile_id!r}: {type(e).__name__}: {e}"
                    ) from e
                yield str(row.tile_id), bytes(buf.tobytes())

        tmp_h5 = tmp_dir / "tiles.h5"
        n_written = _write_per_slide_h5(
            tmp_h5,
            n_tiles=n_tiles,
            tile_stream=_tile_stream(),
            set_recipe_id=set_recipe_id,
            codec_recipe_id=codec_recipe_id,
            tile_asset_id=tile_asset_id,
            slide_file_uuid=slide_file_uuid,
            cache_format=cache_format,
            quality=quality,
            png_compression=png_compression,
            chroma=chroma,
        )
        h5_sha = sha256_path(tmp_h5)
        h5_bytes = int(tmp_h5.stat().st_size)
        record = {
            "status": "ok",
            "slide_file_uuid": slide_file_uuid,
            "set_recipe_id": set_recipe_id,
            "codec_recipe_id": codec_recipe_id,
            "codec_recipe": dict(codec_recipe),
            "h5_relpath": "tiles.h5",
            "h5_sha256": h5_sha,
            "h5_bytes": h5_bytes,
            "n_tiles": int(n_written),
            "created_utc": utc_now_iso(),
            "software_versions": software_versions(),
            "imaging_versions": imaging_versions(),
        }
        write_json_atomic(tmp_dir / "tile_pixels_record.json", record, overwrite=True)
        try:
            os.replace(tmp_dir, pixels_dir)
        except OSError:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if not shared_tile_pixels_is_complete(
                pixels_dir,
                expected_set_recipe_id=set_recipe_id,
                expected_codec_recipe_id=codec_recipe_id,
                verify_sha=verify_sha,
            ):
                raise
            h5_sha = sha256_path(pixels_dir / "tiles.h5")
            h5_bytes = int((pixels_dir / "tiles.h5").stat().st_size)
    finally:
        if slide is not None:
            slide.close()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = float(time.time() - t0)
    return {
        "slide_file_uuid": slide_file_uuid,
        "status": "ok",
        "failure_reason": "",
        "out_path": str(out_path),
        "n_tiles": int(n_written),
        "n_read_failures": 0,
        "h5_sha256": h5_sha,
        "h5_bytes": h5_bytes,
        "elapsed_seconds": elapsed,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _main_impl(args: argparse.Namespace) -> int:
    if openslide is None:
        raise RuntimeError("openslide-python is required")
    if cv2 is None:
        raise RuntimeError("opencv-python/cv2 is required (tile encode)")
    if h5py is None:
        raise RuntimeError("h5py is required")

    data_root = get_full_path("eval", "eval_datasets", "TCGA_eval_data")
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    settings_dir = benchmark_root / "settings"
    cache_root_slides = data_root / "selected_slide_cache" / args.benchmark_id
    if not settings_dir.exists():
        raise FileNotFoundError(settings_dir)

    tile_asset_id, tile_root, set_recipe_id, links_path = _resolve_tile_asset(data_root, settings_dir)
    # Resolve the declared codec + its recipe id; validate the codec early too.
    codec_recipe = _resolve_codec(settings_dir, args)
    codec_recipe_id = tile_codec_recipe_id(set_recipe_id=set_recipe_id, codec_recipe=codec_recipe)
    _encode_params(codec_recipe["format"], quality=codec_recipe["quality"], png_compression=codec_recipe["png_level"])
    cache_dir = tile_root / "tile_rgb_cache"
    (cache_dir / "shard_status").mkdir(parents=True, exist_ok=True)

    case_tile, idx_path, idx_manifest_path = _load_case_tile_index(benchmark_root)
    idx_manifest = read_json(idx_manifest_path) or {}
    if not isinstance(idx_manifest, Mapping):
        raise RuntimeError("case_tile_index_manifest.json must be a mapping")
    # The case tile index must be frozen
    if idx_manifest.get("frozen_final") is not True:
        raise RuntimeError(
            "Refusing to build the RGB cache against a non-frozen case_tile_index "
            f"(frozen_final={idx_manifest.get('frozen_final')!r}). Re-run "
            f"_11_build_case_tile_index.py without debug flags first."
        )
    if str(idx_manifest.get("tile_asset_id") or "") != tile_asset_id:
        raise RuntimeError(
            f"case_tile_index_manifest.tile_asset_id={idx_manifest.get('tile_asset_id')!r} "
            f"!= benchmark_asset_links.tile_asset_id={tile_asset_id!r}"
        )

    # Restrict to the final_case_eligible cases (case_tile_index.parquet also holds the over-provision surplus)
    cohort_dir = benchmark_root / "shared" / "cohort"
    final_case_path = cohort_dir / "final_case_manifest.csv"
    if not final_case_path.exists():
        raise FileNotFoundError(final_case_path)
    final_case = pd.read_csv(final_case_path, dtype=str, keep_default_na=False)
    if not {"final_case_eligible", "case_id"}.issubset(final_case.columns):
        raise RuntimeError("final_case_manifest.csv missing final_case_eligible/case_id columns")
    eligible_cases = set(
        final_case.loc[final_case["final_case_eligible"].map(_truthy_value), "case_id"].astype(str)
    )
    if not eligible_cases:
        raise RuntimeError("No final-eligible cases in final_case_manifest.csv; re-run _11_build_case_tile_index.py")
    n_index_slides = int(case_tile["slide_file_uuid"].astype(str).nunique())
    case_tile = case_tile.loc[case_tile["case_id"].astype(str).isin(eligible_cases)].copy()
    if case_tile.empty:
        raise RuntimeError("case_tile_index.parquet has no rows for final-eligible cases")
    n_eligible_slides = int(case_tile["slide_file_uuid"].astype(str).nunique())
    if n_eligible_slides < n_index_slides:
        print(
            f"INFO: restricting tile RGB cache to the final cohort: {n_eligible_slides} "
            f"eligible slides (skipping {n_index_slides - n_eligible_slides} over-provision "
            f"surplus slides of {n_index_slides} in case_tile_index).",
            file=sys.stderr,
        )

    # case_tile_index selects the slides; all kept tiles of each slide (shared tile manifest) are cached
    tile_asset_manifest_path = tile_root / "tile_asset_manifest.json"
    tile_asset_manifest = read_json(tile_asset_manifest_path) if tile_asset_manifest_path.exists() else {}
    ref_tiles = tile_asset_manifest.get("referenced_shared_tiles") if isinstance(tile_asset_manifest, Mapping) else None
    if not isinstance(ref_tiles, Mapping) or not ref_tiles:
        raise RuntimeError(
            f"tile_asset_manifest.json has no referenced_shared_tiles map ({tile_asset_manifest_path}); "
            "re-run _10_build_tile_asset.py."
        )
    manifest_path_by_uuid: Dict[str, str] = {
        str(u): str(info.get("manifest_path"))
        for u, info in ref_tiles.items()
        if isinstance(info, Mapping) and info.get("manifest_path")
    }
    # Local WSI paths (only needed for slides not cached yet)
    dl = _load_download_manifest(cache_root_slides)
    local_path_by_uuid: Dict[str, str] = {
        str(r["slide_file_uuid"]): str(r["local_path"]) for r in dl.to_dict("records")
    }

    used_uuids = case_tile["slide_file_uuid"].astype(str).unique().tolist()
    missing_manifest = [u for u in used_uuids if u not in manifest_path_by_uuid]
    if missing_manifest:
        raise RuntimeError(f"No shared tile manifest pointer for used slides: {missing_manifest[:20]}")
    uuids = sorted(used_uuids)
    shard_uuids = _shard_filter(uuids, shard_index=args.shard_index, total_shards=args.total_shards)

    # -------------------------------------------------------------------------
    # Finalize-only fast path
    # -------------------------------------------------------------------------
    if args.finalize_only:
        if args.total_shards is None:
            raise RuntimeError("--finalize_only requires --total_shards")
        rows, completed = read_shard_records(cache_dir, int(args.total_shards))
        missing_shards = [i for i in range(int(args.total_shards)) if i not in set(completed)]
        if missing_shards and not args.allow_partial:
            raise RuntimeError(
                f"Missing shard_status for shards: {missing_shards[:20]} (of {args.total_shards}). "
                f"Pass --allow_partial to write a non-frozen manifest."
            )
        return _write_cache_manifest(
            args=args,
            cache_dir=cache_dir,
            tile_asset_id=tile_asset_id,
            tile_root=tile_root,
            settings_dir=settings_dir,
            data_root=data_root,
            set_recipe_id=set_recipe_id,
            codec_recipe_id=codec_recipe_id,
            codec_recipe=codec_recipe,
            per_slide_records=rows,
            n_slides_in_case_tile_index=n_index_slides,
            n_total_slides=len(uuids),
            frozen=not missing_shards,
            missing_shards=missing_shards,
        )

    # -------------------------------------------------------------------------
    # Per-shard run
    # -------------------------------------------------------------------------
    per_slide: List[Dict[str, Any]] = []
    n_failed = 0
    n_skipped = 0
    # Slides already in the shared store are returned as reused_shared
    pbar = tqdm(shard_uuids, desc="Caching tile RGB", unit="slide")
    for uuid in pbar:
        try:
            # Tile manifest read inside the try (a read error becomes a failed_runtime record)
            tiles_for_slide = _full_kept_tiles(Path(manifest_path_by_uuid[uuid]))
            local_path = Path(local_path_by_uuid.get(uuid, ""))
            rec = _process_one_slide(
                slide_file_uuid=uuid,
                local_path=local_path,
                tiles_for_slide=tiles_for_slide,
                data_root=data_root,
                set_recipe_id=set_recipe_id,
                codec_recipe_id=codec_recipe_id,
                codec_recipe=codec_recipe,
                tile_asset_id=tile_asset_id,
                force=bool(args.force),
                verify_sha=bool(args.verify_cache_sha256),
            )
        except Exception as e:
            rec = {
                "slide_file_uuid": uuid,
                "status": "failed_runtime",
                "failure_reason": f"{type(e).__name__}: {e}",
                "out_path": str(shared_tile_pixels_dir(data_root, uuid, set_recipe_id, codec_recipe_id) / "tiles.h5"),
                "n_tiles": 0,
                "elapsed_seconds": 0.0,
            }
        per_slide.append(rec)
        status = rec.get("status")
        if status == "reused_shared":
            n_skipped += 1
        elif status != "ok":
            n_failed += 1
        pbar.set_postfix_str(f"reused={n_skipped} fail={n_failed}")

    if args.shard_index is not None:
        shard_obj = {
            "benchmark_id": args.benchmark_id,
            "tile_asset_id": tile_asset_id,
            "shard_index": int(args.shard_index),
            "total_shards": int(args.total_shards),
            "n_slides_in_shard": int(len(shard_uuids)),
            "n_failed_in_shard": int(n_failed),
            "n_skipped_existing_in_shard": int(n_skipped),
            "per_slide_records": per_slide,
            "finished_utc": utc_now_iso(),
        }
        write_json_atomic(
            shard_status_path(cache_dir, shard_index=int(args.shard_index), total_shards=int(args.total_shards)),
            shard_obj,
            overwrite=True,
        )
        print(json.dumps({
            "status": "shard_ok",
            "shard_index": int(args.shard_index),
            "total_shards": int(args.total_shards),
            "n_slides_in_shard": int(len(shard_uuids)),
            "n_failed_in_shard": int(n_failed),
        }, indent=2))
        return 0

    return _write_cache_manifest(
        args=args,
        cache_dir=cache_dir,
        tile_asset_id=tile_asset_id,
        tile_root=tile_root,
        settings_dir=settings_dir,
        data_root=data_root,
        set_recipe_id=set_recipe_id,
        codec_recipe_id=codec_recipe_id,
        codec_recipe=codec_recipe,
        per_slide_records=per_slide,
        n_slides_in_case_tile_index=n_index_slides,
        n_total_slides=len(uuids),
        frozen=True,
        missing_shards=[],
    )


def _write_cache_manifest(
    *,
    args: argparse.Namespace,
    cache_dir: Path,
    tile_asset_id: str,
    tile_root: Path,
    settings_dir: Path,
    data_root: Path,
    set_recipe_id: str,
    codec_recipe_id: str,
    codec_recipe: Mapping[str, Any],
    per_slide_records: Sequence[Mapping[str, Any]],
    n_slides_in_case_tile_index: int,
    n_total_slides: int,
    frozen: bool,
    missing_shards: Sequence[int],
) -> int:
    # Deduplicate per slide, last-good wins.
    by_uuid: Dict[str, Dict[str, Any]] = {}
    for r in per_slide_records:
        uuid = str(r.get("slide_file_uuid") or "")
        if not uuid:
            continue
        prev = by_uuid.get(uuid)
        if prev is None or (r.get("status") == "ok" and prev.get("status") != "ok"):
            by_uuid[uuid] = dict(r)

    consolidated = sorted(by_uuid.values(), key=lambda x: str(x.get("slide_file_uuid") or ""))
    n_ok = sum(1 for r in consolidated if r.get("status") == "ok")
    n_skipped = sum(1 for r in consolidated if r.get("status") in ("skipped_existing", "reused_shared"))
    n_failed = sum(1 for r in consolidated if str(r.get("status") or "").startswith("failed"))

    official_final = frozen and not missing_shards and (n_failed == 0)

    # Pointers into the shared pixel store (read by feature extraction, _14 and the prune/eviction tools)
    referenced_shared_tiles = {
        str(r["slide_file_uuid"]): {
            "set_recipe_id": set_recipe_id,
            "codec_recipe_id": codec_recipe_id,
            "h5_path": str(shared_tile_pixels_dir(data_root, str(r["slide_file_uuid"]), set_recipe_id, codec_recipe_id) / "tiles.h5"),
            "h5_sha256": r.get("h5_sha256"),
        }
        for r in consolidated
        if r.get("slide_file_uuid") and str(r.get("status") or "") in ("ok", "reused_shared")
    }

    manifest = manifest_header(Path(__file__).name, vars(args))
    manifest.update({
        "frozen_final": bool(official_final),
        "status": "frozen_final" if official_final else "debug_or_partial",
        "nonfinal_reason": (
            "" if official_final else ("missing_shards" if missing_shards else "n_failed>0")
        ),
        "benchmark_id": args.benchmark_id,
        "tile_asset_id": tile_asset_id,
        "tile_root": str(tile_root),
        "cache_dir": str(cache_dir),
        "set_recipe_id": set_recipe_id,
        "codec_recipe_id": codec_recipe_id,
        "codec_recipe": dict(codec_recipe),
        "referenced_shared_tiles": referenced_shared_tiles,
        # Codec of this run (each tiles.h5 also records its recipe ids)
        "format": _format_label(codec_recipe["format"]),
        "cache_format": str(codec_recipe["format"]).lower(),
        "quality": int(codec_recipe["quality"]),
        "png_compression": int(codec_recipe["png_level"]),
        "chroma": str(codec_recipe["chroma"]),
        # Slide counts (case_tile_index also holds the over-provision surplus, not cached)
        "n_slides_in_case_tile_index": int(n_slides_in_case_tile_index),
        "n_final_cohort_slides_to_cache": int(n_total_slides),
        "n_over_provision_surplus_slides_skipped": int(max(0, int(n_slides_in_case_tile_index) - int(n_total_slides))),
        "n_consolidated_slides": int(len(consolidated)),
        "n_ok": int(n_ok),
        "n_skipped_existing": int(n_skipped),
        "n_failed": int(n_failed),
        "total_bytes_on_disk": int(sum(int(r.get("h5_bytes") or 0) for r in consolidated)),
        "per_slide_records": consolidated,
        "missing_shards": list(missing_shards),
    })
    write_json_atomic(cache_dir / "tile_rgb_cache_manifest.json", manifest, overwrite=True)

    # Add the codec recipe id and cache manifest path to benchmark_asset_links.json
    links_path = settings_dir / "benchmark_asset_links.json"
    if links_path.exists():
        links = read_json(links_path)
        if isinstance(links, Mapping):
            links = dict(links)
            links["tile_codec_recipe_id"] = codec_recipe_id
            links["tile_rgb_cache_manifest_path"] = str(cache_dir / "tile_rgb_cache_manifest.json")
            write_json_atomic(links_path, links, overwrite=True)

    print(json.dumps({
        "status": "ok",
        "tile_asset_id": tile_asset_id,
        "cache_dir": str(cache_dir),
        "n_ok": n_ok,
        "n_failed": n_failed,
        "manifest": str(cache_dir / "tile_rgb_cache_manifest.json"),
    }, indent=2))
    if n_failed > 0:
        raise RuntimeError(
            f"{n_failed} slide cache builds failed. See tile_rgb_cache_manifest.json."
        )
    return 0


def main(args: argparse.Namespace) -> int:
    return _main_impl(args)


if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Build the per-slide HDF5 RGB tile cache of a TCGA Task B benchmark."
    )

    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help="Benchmark ID whose kept tiles are cached.")

    ap.add_argument("--shard_index", type=int,
                    default=None,
                    help="Optional 0-based shard index (shards by slide_file_uuid).")

    ap.add_argument("--total_shards", type=int,
                    default=None,
                    help="Total number of shards (must accompany --shard_index).")

    ap.add_argument("--finalize_only", action="store_true",
                    help="Skip per-slide work and only aggregate shard_status JSONs into the consolidated manifest.")

    ap.add_argument("--allow_partial", action="store_true",
                    help="With --finalize_only, allow consolidation even if some shards are missing (manifest non-frozen).")

    # Debug overrides of the codec declared in benchmark_config.tile_recipe.codec
    ap.add_argument("--cache_format", type=str,
                    choices=["jpeg", "webp", "png"],
                    default=None,
                    help="Debug override of benchmark_config.tile_recipe.codec.format (jpeg/webp/png).")

    ap.add_argument("--quality", type=int,
                    default=None,
                    help="Debug override of the declared lossy quality (1-100) for jpeg/webp. Ignored for png.")

    ap.add_argument("--png_compression", type=int,
                    default=None,
                    help="Debug override of the declared cv2 PNG compression level (0-9) for png.")

    ap.add_argument("--chroma", type=str,
                    default=None,
                    help="Debug override of the declared jpeg chroma subsampling tag (e.g. 444).")

    ap.add_argument("--force", action="store_true",
                    help="Recompute and overwrite the shared per-slide pixel blobs even if complete ones exist.")

    ap.add_argument("--verify_cache_sha256", action="store_true",
                    help="Check the sha256 of already-cached tiles.h5 files (default: file-size check only).")

    args = ap.parse_args()
    raise SystemExit(main(args))
