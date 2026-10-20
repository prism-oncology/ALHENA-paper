#!/usr/bin/env python3
"""
Find the tile-store blobs (processed_assets/tiles/_store/: tile sets and pixels) referenced by no benchmark tile or
RGB-cache manifest and delete them with --delete (dry run by default); --verify checks the referenced blobs.

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data / processed_assets / tiles / _store / _prune_logs /
    - prune_log_<utc>.json
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path

from _taskb_common import (
    SHARED_TILES_STORE_DIRNAME,
    manifest_header,
    read_json,
    sha256_path,
    shared_tile_h5_path,
    shared_tile_manifest_path,
    shared_tiles_store_root,
    utc_now_iso,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

SetPair = Tuple[str, str]            # (uuid, set_recipe_id)
PixelTriple = Tuple[str, str, str]   # (uuid, set_recipe_id, codec_recipe_id)


def _dir_size_bytes(path: Path) -> int:
    # Skip files that disappear during the walk (e.g. macOS ._* files)
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _resilient_rmtree(path, attempts: int = 4) -> None:
    """Recursively delete path; missing entries count as removed (e.g. macOS ._* files deleted with their
    primary file) and other OSErrors are retried `attempts` times before being raised."""
    path = Path(path)

    def _onexc(func, p, exc):            # shutil.rmtree onexc (Python 3.12+)
        if isinstance(exc, FileNotFoundError):
            return
        raise exc

    def _onerror(func, p, exc_info):     # shutil.rmtree onerror (Python < 3.12)
        if issubclass(exc_info[0], FileNotFoundError):
            return
        raise exc_info[1]

    last_exc = None
    for _ in range(max(1, attempts)):
        if not path.exists():
            return
        try:
            try:
                shutil.rmtree(path, onexc=_onexc)         # Python 3.12+
            except TypeError:
                shutil.rmtree(path, onerror=_onerror)     # Python < 3.12
            return
        except FileNotFoundError:
            return
        except OSError as e:
            last_exc = e
    if path.exists() and last_exc is not None:
        raise last_exc


def _scan_tile_manifests(
    tiles_root: Path,
) -> Tuple[Set[SetPair], Set[PixelTriple], Dict[SetPair, Dict[str, Any]], Dict[PixelTriple, Dict[str, Any]], List[Dict[str, Any]]]:
    """Referenced (uuid, set_recipe_id) pairs and (uuid, set, codec) triples in the tile_asset and tile_rgb_cache
    manifests of all benchmarks. Returns (referenced_sets, referenced_pixels, set_info, pixel_info, per_benchmark),
    where set_info / pixel_info hold the recorded sha256."""
    referenced_sets: Set[SetPair] = set()
    referenced_pixels: Set[PixelTriple] = set()
    set_info: Dict[SetPair, Dict[str, Any]] = {}
    pixel_info: Dict[PixelTriple, Dict[str, Any]] = {}
    per_benchmark: List[Dict[str, Any]] = []
    if not tiles_root.exists():
        return referenced_sets, referenced_pixels, set_info, pixel_info, per_benchmark
    for entry in sorted(tiles_root.iterdir()):
        if not entry.is_dir() or entry.name == SHARED_TILES_STORE_DIRNAME:
            continue
        manifests = [
            ("tile_asset_manifest", entry / "tile_asset_manifest.json"),
            ("tile_rgb_cache_manifest", entry / "tile_rgb_cache" / "tile_rgb_cache_manifest.json"),
        ]
        n_set_refs = 0
        n_pixel_refs = 0
        benchmark_id = None
        for kind, mp in manifests:
            if not mp.exists():
                continue
            try:
                manifest = read_json(mp)
            except Exception:
                logger.warning("Could not read %s; skipping.", mp)
                continue
            if not isinstance(manifest, Mapping):
                continue
            benchmark_id = benchmark_id or manifest.get("benchmark_id")
            ref_tiles = manifest.get("referenced_shared_tiles")
            if not isinstance(ref_tiles, Mapping):
                continue
            for uuid, info in ref_tiles.items():
                if not isinstance(info, Mapping):
                    continue
                uuid = str(uuid).strip()
                sid = str(info.get("set_recipe_id") or "").strip()
                if not uuid or not sid:
                    continue
                pair = (uuid, sid)
                referenced_sets.add(pair)
                n_set_refs += 1
                if info.get("manifest_sha256"):
                    set_info.setdefault(pair, {"sha256": info.get("manifest_sha256")})
                cid = str(info.get("codec_recipe_id") or "").strip()
                if cid:
                    triple = (uuid, sid, cid)
                    referenced_pixels.add(triple)
                    n_pixel_refs += 1
                    if info.get("h5_sha256"):
                        pixel_info.setdefault(triple, {"sha256": info.get("h5_sha256")})
        per_benchmark.append({
            "tile_asset_id": entry.name,
            "benchmark_id": benchmark_id,
            "n_referenced_sets": n_set_refs,
            "n_referenced_pixels": n_pixel_refs,
        })
    return referenced_sets, referenced_pixels, set_info, pixel_info, per_benchmark


def _store_blobs(by_uuid_root: Path) -> Tuple[Dict[SetPair, Path], Dict[PixelTriple, Path]]:
    """Enumerate every set-tier and pixel-tier blob present in the store."""
    set_blobs: Dict[SetPair, Path] = {}
    pixel_blobs: Dict[PixelTriple, Path] = {}
    if not by_uuid_root.exists():
        return set_blobs, pixel_blobs
    for uuid_dir in sorted(by_uuid_root.iterdir()):
        if not uuid_dir.is_dir() or uuid_dir.name.startswith("."):
            continue
        for set_dir in sorted(uuid_dir.iterdir()):
            if not set_dir.is_dir() or set_dir.name.startswith("."):
                continue
            set_blobs[(uuid_dir.name, set_dir.name)] = set_dir
            pixels_root = set_dir / "pixels"
            if pixels_root.is_dir():
                for codec_dir in sorted(pixels_root.iterdir()):
                    if codec_dir.is_dir() and not codec_dir.name.startswith("."):
                        pixel_blobs[(uuid_dir.name, set_dir.name, codec_dir.name)] = codec_dir
    return set_blobs, pixel_blobs


def main(args) -> int:
    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    tiles_root = data_root / "processed_assets" / "tiles"
    store_root = shared_tiles_store_root(data_root)
    by_uuid = store_root / "by_uuid"

    referenced_sets, referenced_pixels, set_info, pixel_info, per_benchmark = _scan_tile_manifests(tiles_root)
    set_blobs, pixel_blobs = _store_blobs(by_uuid)

    # Pixel orphans: triple referenced by no cache manifest.
    orphan_pixels = {t: p for t, p in pixel_blobs.items() if t not in referenced_pixels}
    # Set orphans: pair referenced by no manifest and with no referenced pixel beneath it.
    sets_with_ref_pixel = {(u, s) for (u, s, _c) in referenced_pixels}
    orphan_sets = {
        pair: p for pair, p in set_blobs.items()
        if pair not in referenced_sets and pair not in sets_with_ref_pixel
    }

    orphan_pixel_records = [
        {"slide_file_uuid": u, "set_recipe_id": s, "codec_recipe_id": c, "path": str(p), "bytes": _dir_size_bytes(p)}
        for (u, s, c), p in orphan_pixels.items()
    ]
    orphan_set_records = [
        {"slide_file_uuid": u, "set_recipe_id": s, "path": str(p), "bytes": _dir_size_bytes(p)}
        for (u, s), p in orphan_sets.items()
    ]
    reclaimable_bytes = sum(r["bytes"] for r in orphan_pixel_records) + sum(r["bytes"] for r in orphan_set_records)

    # Optional integrity check of referenced blobs against the manifests (never deletes).
    verify_mismatches: List[Dict[str, Any]] = []
    if args.verify:
        # Blob paths rebuilt from their ids under the current data_root
        for (u, s), info in set_info.items():
            expected = info.get("sha256")
            if not expected:
                continue
            fp = shared_tile_manifest_path(data_root, u, s)
            if not fp.exists():
                verify_mismatches.append({"tier": "set", "slide_file_uuid": u, "set_recipe_id": s, "path": str(fp), "reason": "missing"})
            elif sha256_path(fp) != expected:
                verify_mismatches.append({"tier": "set", "slide_file_uuid": u, "set_recipe_id": s, "path": str(fp), "expected_sha256": expected, "observed_sha256": sha256_path(fp)})
        for (u, s, c), info in pixel_info.items():
            expected = info.get("sha256")
            if not expected:
                continue
            fp = shared_tile_h5_path(data_root, u, s, c)
            if not fp.exists():
                verify_mismatches.append({"tier": "pixels", "slide_file_uuid": u, "set_recipe_id": s, "codec_recipe_id": c, "path": str(fp), "reason": "missing"})
            elif sha256_path(fp) != expected:
                verify_mismatches.append({"tier": "pixels", "slide_file_uuid": u, "set_recipe_id": s, "codec_recipe_id": c, "path": str(fp), "expected_sha256": expected, "observed_sha256": sha256_path(fp)})

    # Refuse --delete if no manifest was found (every blob would look orphaned)
    if args.delete and not per_benchmark and not args.allow_empty_reference:
        raise RuntimeError(
            "Refusing to --delete: no benchmark tile manifests were found under "
            f"{tiles_root}, so every blob looks orphaned. Re-run without --delete to inspect, "
            "or pass --allow_empty_reference if you really intend to empty the store."
        )

    deleted_pixels: List[Dict[str, Any]] = []
    deleted_sets: List[Dict[str, Any]] = []
    if args.delete:
        # Pixels first (they nest under set dirs), then orphaned set dirs whole.
        for rec in orphan_pixel_records:
            _resilient_rmtree(rec["path"])
            deleted_pixels.append(rec)
            logger.info("Deleted orphan pixel blob uuid=%s set=%s codec=%s (%d bytes): %s",
                        rec["slide_file_uuid"], rec["set_recipe_id"], rec["codec_recipe_id"], rec["bytes"], rec["path"])
        for rec in orphan_set_records:
            _resilient_rmtree(rec["path"])
            deleted_sets.append(rec)
            logger.info("Deleted orphan set blob uuid=%s set=%s (%d bytes): %s",
                        rec["slide_file_uuid"], rec["set_recipe_id"], rec["bytes"], rec["path"])
        # Remove now-empty pixels/ and per-uuid parent dirs.
        for rec in deleted_pixels:
            pixels_root = Path(rec["path"]).parent
            try:
                if pixels_root.is_dir() and not any(pixels_root.iterdir()):
                    pixels_root.rmdir()
            except OSError:
                pass
        parent_dirs = {Path(r["path"]).parent for r in deleted_sets} | {Path(r["path"]).parent.parent for r in deleted_pixels}
        for uuid_dir in sorted(parent_dirs):
            try:
                if uuid_dir.is_dir() and uuid_dir.name != "by_uuid" and not any(uuid_dir.iterdir()):
                    uuid_dir.rmdir()
            except OSError:
                pass

    summary = {
        "shared_store_root": str(store_root),
        "n_benchmarks_with_manifest": len(per_benchmark),
        "n_referenced_sets": len(referenced_sets),
        "n_referenced_pixels": len(referenced_pixels),
        "n_store_set_blobs": len(set_blobs),
        "n_store_pixel_blobs": len(pixel_blobs),
        "n_orphan_set_blobs": len(orphan_sets),
        "n_orphan_pixel_blobs": len(orphan_pixels),
        "reclaimable_bytes": reclaimable_bytes,
        "reclaimable_gib": round(reclaimable_bytes / (1024 ** 3), 3),
        "deleted": bool(args.delete),
        "n_deleted_set_blobs": len(deleted_sets),
        "n_deleted_pixel_blobs": len(deleted_pixels),
        "n_verify_mismatches": len(verify_mismatches),
    }

    log_obj = manifest_header(Path(__file__).name, vars(args))
    log_obj.update({
        "summary": summary,
        "per_benchmark": per_benchmark,
        "orphan_set_blobs": orphan_set_records,
        "orphan_pixel_blobs": orphan_pixel_records,
        "deleted_set_blobs": deleted_sets,
        "deleted_pixel_blobs": deleted_pixels,
        "verify_mismatches": verify_mismatches,
    })
    logs_dir = store_root / "_prune_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"prune_log_{utc_now_iso().replace(':', '').replace('-', '')}.json"
    write_json_atomic(log_path, log_obj, overwrite=False)

    if verify_mismatches and not args.allow_verify_mismatch:
        raise RuntimeError(f"{len(verify_mismatches)} referenced tile blob(s) failed SHA256 verification. See {log_path}")

    print(json.dumps({"status": "ok", **summary, "log": str(log_path)}, indent=2))
    return 0


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ap = argparse.ArgumentParser(description="Prune orphaned blobs from the shared tile store (processed_assets/tiles/_store).")

    ap.add_argument("--delete", action="store_true",
                    help="Delete orphaned tile blobs. Without this flag the run is a dry-run that only reports them.")

    ap.add_argument("--verify", action="store_true",
                    help="Re-check the SHA256 of referenced manifest.parquet / tiles.h5 against the benchmark manifests (never deletes).")

    ap.add_argument("--allow_empty_reference", action="store_true",
                    help="Permit --delete even when no benchmark tile manifests are found (would empty the store). Use with care.")

    ap.add_argument("--allow_verify_mismatch", action="store_true",
                    help="Exit 0 even if --verify finds SHA256 mismatches (still recorded in the log).")

    args = ap.parse_args()
    raise SystemExit(main(args))
