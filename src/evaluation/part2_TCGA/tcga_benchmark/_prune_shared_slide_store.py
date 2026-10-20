#!/usr/bin/env python3
"""
Find the blobs of the shared slide store (selected_slide_cache/_store/: WSIs and prefetch results) referenced by no
benchmark's download manifest and delete them with --delete (dry run by default); --verify checks the WSI MD5s.

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data / selected_slide_cache / _store / _prune_logs /
    - prune_log_<utc>.json
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import pandas as pd

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path

from _taskb_common import (
    SHARED_SLIDE_STORE_DIRNAME,
    manifest_header,
    md5_path,
    shared_store_root,
    utc_now_iso,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

DOWNLOAD_MANIFEST_RELPATH = ("manifests", "selected_slide_download_manifest.tsv")


def _dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _scan_manifests(cache_root: Path) -> Tuple[Set[str], Dict[str, str], List[Dict[str, Any]]]:
    """Union slide_file_uuids referenced by every benchmark's download manifest.

    Returns (referenced_uuids, expected_md5_by_uuid, per_benchmark_summaries).
    """
    referenced: Set[str] = set()
    md5_by_uuid: Dict[str, str] = {}
    per_benchmark: List[Dict[str, Any]] = []
    if not cache_root.exists():
        return referenced, md5_by_uuid, per_benchmark
    for entry in sorted(cache_root.iterdir()):
        if not entry.is_dir() or entry.name == SHARED_SLIDE_STORE_DIRNAME:
            continue
        manifest_path = entry.joinpath(*DOWNLOAD_MANIFEST_RELPATH)
        if not manifest_path.exists():
            continue
        df = pd.read_csv(manifest_path, sep="\t", dtype=str, keep_default_na=False)
        if "slide_file_uuid" not in df.columns:
            logger.warning("Manifest %s lacks slide_file_uuid; skipping.", manifest_path)
            continue
        uuids = {str(u).strip() for u in df["slide_file_uuid"] if str(u).strip()}
        referenced |= uuids
        # Prefer the expected (GDC-truth) md5 for --verify; fall back to observed.
        for col in ("md5sum_expected", "md5sum_observed"):
            if col in df.columns:
                for u, m in zip(df["slide_file_uuid"].astype(str), df[col].astype(str)):
                    u, m = u.strip(), m.strip()
                    if u and m and u not in md5_by_uuid:
                        md5_by_uuid[u] = m
        per_benchmark.append({"benchmark_id": entry.name, "manifest": str(manifest_path), "n_uuids": len(uuids)})
    return referenced, md5_by_uuid, per_benchmark


def _store_uuid_dirs(by_uuid_root: Path) -> Dict[str, Path]:
    if not by_uuid_root.exists():
        return {}
    return {p.name: p for p in sorted(by_uuid_root.iterdir()) if p.is_dir()}


def _orphans(uuid_dirs: Dict[str, Path], referenced: Set[str]) -> Dict[str, Path]:
    return {u: p for u, p in uuid_dirs.items() if u not in referenced}


def main(args) -> int:
    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    cache_root = data_root / "selected_slide_cache"
    store_root = shared_store_root(data_root)
    slide_by_uuid = store_root / "slide_files" / "by_uuid"
    prefetch_by_uuid = store_root / "prefetch" / "by_uuid"

    # Blobs found by walking the store under the current data_root, matched by uuid to the manifests
    referenced, md5_by_uuid, per_benchmark = _scan_manifests(cache_root)
    slide_dirs = _store_uuid_dirs(slide_by_uuid)
    prefetch_dirs = _store_uuid_dirs(prefetch_by_uuid)

    slide_orphans = _orphans(slide_dirs, referenced)
    prefetch_orphans = _orphans(prefetch_dirs, referenced)

    # (kind, uuid, path, bytes) for every orphan blob.
    orphan_records: List[Dict[str, Any]] = []
    for kind, orphans in (("slide_files", slide_orphans), ("prefetch", prefetch_orphans)):
        for u, p in orphans.items():
            orphan_records.append({"kind": kind, "slide_file_uuid": u, "path": str(p), "bytes": _dir_size_bytes(p)})
    reclaimable_bytes = sum(r["bytes"] for r in orphan_records)

    # Optional integrity check of referenced WSIs against the manifests (never deletes).
    verify_mismatches: List[Dict[str, Any]] = []
    if args.verify:
        for u, p in slide_dirs.items():
            expected = md5_by_uuid.get(u)
            if u not in referenced or not expected:
                continue
            for f in sorted(p.iterdir()):
                if not f.is_file():
                    continue
                observed = md5_path(f)
                if observed.lower() != expected.lower():
                    verify_mismatches.append({"slide_file_uuid": u, "path": str(f), "expected_md5": expected, "observed_md5": observed})

    # Refuse --delete if no manifest was found (every blob would look orphaned)
    if args.delete and not per_benchmark and not args.allow_empty_reference:
        raise RuntimeError(
            "Refusing to --delete: no benchmark download manifests were found under "
            f"{cache_root}, so every blob looks orphaned. Re-run without --delete to inspect, "
            "or pass --allow_empty_reference if you really intend to empty the store."
        )

    deleted: List[Dict[str, Any]] = []
    if args.delete:
        for rec in orphan_records:
            shutil.rmtree(rec["path"], ignore_errors=False)
            deleted.append(rec)
            logger.info("Deleted orphan %s blob %s (%d bytes): %s", rec["kind"], rec["slide_file_uuid"], rec["bytes"], rec["path"])

    summary = {
        "shared_store_root": str(store_root),
        "n_benchmarks_with_manifest": len(per_benchmark),
        "n_referenced_uuids": len(referenced),
        "n_store_slide_blobs": len(slide_dirs),
        "n_store_prefetch_blobs": len(prefetch_dirs),
        "n_orphan_slide_blobs": len(slide_orphans),
        "n_orphan_prefetch_blobs": len(prefetch_orphans),
        "reclaimable_bytes": reclaimable_bytes,
        "reclaimable_gib": round(reclaimable_bytes / (1024 ** 3), 3),
        "deleted": bool(args.delete),
        "n_deleted_blobs": len(deleted),
        "n_verify_mismatches": len(verify_mismatches),
    }

    log_obj = manifest_header(Path(__file__).name, vars(args))
    log_obj.update({
        "summary": summary,
        "per_benchmark": per_benchmark,
        "orphans": orphan_records,
        "deleted_blobs": deleted,
        "verify_mismatches": verify_mismatches,
    })
    logs_dir = store_root / "_prune_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"prune_log_{utc_now_iso().replace(':', '').replace('-', '')}.json"
    write_json_atomic(log_path, log_obj, overwrite=False)

    if verify_mismatches and not args.allow_verify_mismatch:
        raise RuntimeError(f"{len(verify_mismatches)} referenced WSI(s) failed MD5 verification. See {log_path}")

    print(json.dumps({"status": "ok", **summary, "log": str(log_path)}, indent=2))
    return 0


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ap = argparse.ArgumentParser(description="Prune orphaned blobs from the shared TCGA slide store (selected_slide_cache/_store).")

    ap.add_argument("--delete", action="store_true",
                    help="Delete orphaned blobs. Without this flag the run is a dry-run that only reports them.")

    ap.add_argument("--verify", action="store_true",
                    help="Re-check the MD5 of referenced WSIs against the benchmark manifests (never deletes).")

    ap.add_argument("--allow_empty_reference", action="store_true",
                    help="Permit --delete even when no benchmark manifests are found (would empty the store). Use with care.")

    ap.add_argument("--allow_verify_mismatch", action="store_true",
                    help="Exit 0 even if --verify finds MD5 mismatches (still recorded in the log).")

    args = ap.parse_args()
    raise SystemExit(main(args))
