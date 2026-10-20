#!/usr/bin/env python3
"""
Delete the shared WSIs (selected_slide_cache/_store/slide_files/by_uuid/<uuid>/) of the slides of a TCGA Task B
benchmark that are fully covered by the tile RGB cache and not used by another benchmark (dry run unless --delete).

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data / selected_slide_cache / <benchmark_id> / manifests /
    - wsi_eviction_log_<utc>.json
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import pandas as pd

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path

from _taskb_common import (
    SHARED_SLIDE_STORE_DIRNAME,
    SHARED_TILES_STORE_DIRNAME,
    manifest_header,
    read_json,
    sha256_path,
    shared_store_root,
    shared_tile_h5_path,
    shared_tile_pixels_dir,
    shared_tile_pixels_is_complete,
    utc_now_iso,
    write_json_atomic,
)

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

logger = logging.getLogger(__name__)

DOWNLOAD_MANIFEST_RELPATH = ("manifests", "selected_slide_download_manifest.tsv")


def _dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _resolve_tile_ids(settings_dir: Path) -> Tuple[str, str, str]:
    links_path = settings_dir / "benchmark_asset_links.json"
    if not links_path.exists():
        raise FileNotFoundError(links_path)
    links = read_json(links_path)
    if not isinstance(links, Mapping):
        raise RuntimeError("benchmark_asset_links.json must be a mapping")
    tile_asset_id = str(links.get("tile_asset_id") or "")
    set_recipe_id = str(links.get("tile_set_recipe_id") or "")
    codec_recipe_id = str(links.get("tile_codec_recipe_id") or "")
    if not tile_asset_id or not set_recipe_id:
        raise RuntimeError("tile_asset_id / tile_set_recipe_id missing. Run _10_build_tile_asset.py first.")
    if not codec_recipe_id:
        raise RuntimeError("tile_codec_recipe_id missing. Run _15_build_tile_rgb_cache.py first.")
    return tile_asset_id, set_recipe_id, codec_recipe_id


def _needed_codecs_by_uuid(tiles_root: Path, set_recipe_id: str) -> Dict[str, Set[str]]:
    """uuid -> codec_recipe_ids referenced for this set_recipe_id by the tile RGB cache manifests of all benchmarks."""
    out: Dict[str, Set[str]] = {}
    if not tiles_root.exists():
        return out
    for entry in sorted(tiles_root.iterdir()):
        if not entry.is_dir() or entry.name == SHARED_TILES_STORE_DIRNAME:
            continue
        mp = entry / "tile_rgb_cache" / "tile_rgb_cache_manifest.json"
        if not mp.exists():
            continue
        try:
            manifest = read_json(mp)
        except Exception:
            continue
        ref_tiles = manifest.get("referenced_shared_tiles") if isinstance(manifest, Mapping) else None
        if not isinstance(ref_tiles, Mapping):
            continue
        for uuid, info in ref_tiles.items():
            if not isinstance(info, Mapping) or str(info.get("set_recipe_id") or "") != set_recipe_id:
                continue
            cid = str(info.get("codec_recipe_id") or "").strip()
            if cid:
                out.setdefault(str(uuid), set()).add(cid)
    return out


def _needed_tiles_by_uuid(benchmark_root: Path, tile_asset_id: str) -> Dict[str, Set[str]]:
    idx_path = benchmark_root / "shared" / "tiles" / "case_tile_index.parquet"
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    df = pd.read_parquet(idx_path, columns=["tile_id", "slide_file_uuid", "tile_asset_id"])
    if not df["tile_asset_id"].astype(str).eq(tile_asset_id).all():
        raise RuntimeError(
            "case_tile_index.parquet contains rows from a different tile_asset_id than "
            f"benchmark_asset_links ({tile_asset_id!r}) — refusing to evict."
        )
    by_uuid: Dict[str, Set[str]] = {}
    for uuid, tid in zip(df["slide_file_uuid"].astype(str), df["tile_id"].astype(str)):
        by_uuid.setdefault(uuid, set()).add(tid)
    return by_uuid


def _cached_tile_ids(h5_path: Path, *, expected_set_recipe_id: str, expected_codec_recipe_id: str) -> Set[str]:
    if h5py is None:
        raise RuntimeError("h5py is required to read the tile RGB cache")
    with h5py.File(h5_path, "r") as f:
        declared = (str(f.attrs.get("set_recipe_id", "")), str(f.attrs.get("codec_recipe_id", "")))
        if declared != (expected_set_recipe_id, expected_codec_recipe_id):
            raise RuntimeError(
                f"cache {h5_path} declares recipe {declared!r} != expected "
                f"{(expected_set_recipe_id, expected_codec_recipe_id)!r}"
            )
        ids = f["tile_ids"][:]
    return {(t.decode("utf-8") if isinstance(t, (bytes, bytearray)) else str(t)) for t in ids}


def _scan_uuid_benchmark_refs(cache_root: Path) -> Dict[str, Set[str]]:
    """uuid -> set of benchmark_ids whose download manifest references it."""
    refs: Dict[str, Set[str]] = {}
    if not cache_root.exists():
        return refs
    for entry in sorted(cache_root.iterdir()):
        if not entry.is_dir() or entry.name == SHARED_SLIDE_STORE_DIRNAME:
            continue
        manifest_path = entry.joinpath(*DOWNLOAD_MANIFEST_RELPATH)
        if not manifest_path.exists():
            continue
        df = pd.read_csv(manifest_path, sep="\t", dtype=str, keep_default_na=False)
        if "slide_file_uuid" not in df.columns:
            continue
        for u in df["slide_file_uuid"].astype(str):
            u = u.strip()
            if u:
                refs.setdefault(u, set()).add(entry.name)
    return refs


def _cache_manifest_sha_by_uuid(cache_manifest: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not cache_manifest:
        return out
    for rec in cache_manifest.get("per_slide_records", []) or []:
        u = str(rec.get("slide_file_uuid") or "").strip()
        sha = str(rec.get("h5_sha256") or "").strip()
        if u and sha:
            out[u] = sha
    return out


def main(args: argparse.Namespace) -> int:
    data_root = get_full_path("eval", "eval_datasets", "TCGA_eval_data")
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    settings_dir = benchmark_root / "settings"
    cache_root = data_root / "selected_slide_cache"
    store_root = shared_store_root(data_root)
    slide_by_uuid = store_root / "slide_files" / "by_uuid"
    if not settings_dir.exists():
        raise FileNotFoundError(settings_dir)

    tiles_root = data_root / "processed_assets" / "tiles"
    tile_asset_id, set_recipe_id, codec_recipe_id = _resolve_tile_ids(settings_dir)
    cache_dir = tiles_root / tile_asset_id / "tile_rgb_cache"
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"tile RGB cache dir {cache_dir} does not exist — build it with "
            f"_15_build_tile_rgb_cache.py before evicting WSIs."
        )

    needed_by_uuid = _needed_tiles_by_uuid(benchmark_root, tile_asset_id)

    cache_manifest = read_json(cache_dir / "tile_rgb_cache_manifest.json") if (cache_dir / "tile_rgb_cache_manifest.json").exists() else None
    cache_frozen = bool((cache_manifest or {}).get("frozen_final"))
    if not cache_frozen and not args.allow_unfrozen:
        raise RuntimeError(
            f"tile RGB cache for tile_asset_id={tile_asset_id!r} is not frozen_final "
            f"(manifest frozen_final={cache_frozen!r}). Finalize the cache with "
            f"_15_build_tile_rgb_cache.py, or pass --allow_unfrozen to override (discouraged)."
        )
    sha_by_uuid = _cache_manifest_sha_by_uuid(cache_manifest)

    uuid_refs = _scan_uuid_benchmark_refs(cache_root)
    # Codecs needed by any benchmark (all must be cached before a WSI is deleted)
    needed_codecs = _needed_codecs_by_uuid(tiles_root, set_recipe_id)

    records: List[Dict[str, Any]] = []
    for uuid in sorted(needed_by_uuid):
        needed = needed_by_uuid[uuid]
        # WSI dir and cache file paths, rebuilt from ids under the current data_root
        store_blob = slide_by_uuid / uuid
        this_h5 = shared_tile_h5_path(data_root, uuid, set_recipe_id, codec_recipe_id)
        codecs_required = set(needed_codecs.get(uuid, set())) | {codec_recipe_id}
        rec: Dict[str, Any] = {
            "slide_file_uuid": uuid,
            "n_tiles_needed": len(needed),
            "n_codecs_required": len(codecs_required),
            "store_blob": str(store_blob),
            "store_blob_bytes": _dir_size_bytes(store_blob) if store_blob.exists() else 0,
        }
        other_benchmarks = sorted(uuid_refs.get(uuid, set()) - {args.benchmark_id})

        if not store_blob.exists():
            rec.update(status="already_absent", reason="no WSI blob in store")
        elif other_benchmarks:
            rec.update(status="skip_shared", reason=f"referenced by other benchmark(s): {other_benchmarks}")
        else:
            incomplete = [
                c for c in sorted(codecs_required)
                if not shared_tile_pixels_is_complete(
                    shared_tile_pixels_dir(data_root, uuid, set_recipe_id, c),
                    expected_set_recipe_id=set_recipe_id, expected_codec_recipe_id=c,
                )
            ]
            if incomplete:
                rec.update(status="skip_codec_coverage", reason=f"pixel blob incomplete for codec(s): {incomplete}")
            elif not this_h5.exists():
                rec.update(status="skip_cache_missing", reason="no per-slide cache HDF5 for this codec")
            else:
                try:
                    cached = _cached_tile_ids(this_h5, expected_set_recipe_id=set_recipe_id, expected_codec_recipe_id=codec_recipe_id)
                    missing = needed - cached
                    if missing:
                        rec.update(status="skip_cache_incomplete",
                                   reason=f"{len(missing)} of {len(needed)} tiles not in cache")
                    elif args.verify_sha and sha_by_uuid.get(uuid) and sha256_path(this_h5) != sha_by_uuid[uuid]:
                        rec.update(status="skip_sha_mismatch", reason="cache HDF5 sha256 != manifest")
                    else:
                        rec.update(status="evictable", reason="")
                except Exception as e:  # corrupt / unreadable cache -> never evict
                    rec.update(status="skip_cache_error", reason=f"{type(e).__name__}: {e}")
        records.append(rec)

    evictable = [r for r in records if r["status"] == "evictable"]
    reclaimable_bytes = sum(int(r["store_blob_bytes"]) for r in evictable)

    deleted: List[Dict[str, Any]] = []
    if args.delete:
        if not evictable:
            logger.info("Nothing evictable; no slide blobs deleted.")
        for r in evictable:
            shutil.rmtree(r["store_blob"], ignore_errors=False)
            deleted.append(r)
            logger.info("Evicted WSI blob %s (%d bytes): %s",
                        r["slide_file_uuid"], r["store_blob_bytes"], r["store_blob"])

    by_status: Dict[str, int] = {}
    for r in records:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    summary = {
        "benchmark_id": args.benchmark_id,
        "tile_asset_id": tile_asset_id,
        "set_recipe_id": set_recipe_id,
        "codec_recipe_id": codec_recipe_id,
        "cache_dir": str(cache_dir),
        "cache_frozen_final": cache_frozen,
        "n_slides_in_benchmark": len(needed_by_uuid),
        "counts_by_status": by_status,
        "n_evictable": len(evictable),
        "reclaimable_bytes": reclaimable_bytes,
        "reclaimable_gib": round(reclaimable_bytes / (1024 ** 3), 3),
        "deleted": bool(args.delete),
        "n_deleted_blobs": len(deleted),
        "verify_sha": bool(args.verify_sha),
    }

    log_obj = manifest_header(Path(__file__).name, vars(args))
    log_obj.update({"summary": summary, "per_slide_records": records, "deleted_blobs": deleted})
    logs_dir = cache_root / args.benchmark_id / "manifests"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"wsi_eviction_log_{utc_now_iso().replace(':', '').replace('-', '')}.json"
    write_json_atomic(log_path, log_obj, overwrite=False)

    print(json.dumps({"status": "ok", **summary, "log": str(log_path)}, indent=2))
    return 0


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ap = argparse.ArgumentParser(
        description="Evict shared WSI bytes for slides fully baked into the Task B tile RGB cache."
    )

    ap.add_argument("--benchmark_id", type=str, required=True,
                    help="Benchmark whose cached slides' WSI bytes are candidates for eviction.")

    ap.add_argument("--delete", action="store_true",
                    help="Delete evictable WSI blobs. Without this flag the run is a dry-run that only reports them.")

    ap.add_argument("--verify_sha", action="store_true",
                    help="Also re-check each per-slide cache HDF5 sha256 against the cache manifest before evicting.")

    ap.add_argument("--allow_unfrozen", action="store_true",
                    help="Permit eviction even when the tile RGB cache manifest is not frozen_final (discouraged).")

    args = ap.parse_args()
    raise SystemExit(main(args))
