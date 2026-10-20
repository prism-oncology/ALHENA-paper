#!/usr/bin/env python3
"""
Find the blobs of the shared HistoQC mask store (processed_assets/histoqc/_store/) referenced by no benchmark's
histoqc_asset_manifest.json and delete them with --delete (dry run by default); --verify checks the referenced masks.

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data / processed_assets / histoqc / _store / _prune_logs /
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
    SHARED_HISTOQC_STORE_DIRNAME,
    manifest_header,
    read_json,
    reroot_under_data_root,
    sha256_path,
    shared_histoqc_store_root,
    utc_now_iso,
    write_json_atomic,
)

logger = logging.getLogger(__name__)


def _dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _scan_histoqc_manifests(
    histoqc_root: Path,
) -> Tuple[Set[Tuple[str, str]], Dict[Tuple[str, str], Dict[str, Any]], List[Dict[str, Any]]]:
    """(uuid, run_key) pairs referenced by the HistoQC manifests of all benchmarks.
    Returns (referenced_pairs, {pair: {path, sha256}}, per-benchmark summaries)."""
    referenced: Set[Tuple[str, str]] = set()
    ref_info: Dict[Tuple[str, str], Dict[str, Any]] = {}
    per_benchmark: List[Dict[str, Any]] = []
    if not histoqc_root.exists():
        return referenced, ref_info, per_benchmark
    for entry in sorted(histoqc_root.iterdir()):
        if not entry.is_dir() or entry.name == SHARED_HISTOQC_STORE_DIRNAME:
            continue
        manifest_path = entry / "histoqc_asset_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = read_json(manifest_path)
        except Exception:
            logger.warning("Could not read %s; skipping.", manifest_path)
            continue
        ref_masks = manifest.get("referenced_shared_masks") if isinstance(manifest, Mapping) else None
        pairs: Set[Tuple[str, str]] = set()
        if isinstance(ref_masks, Mapping):
            for uuid, info in ref_masks.items():
                if not isinstance(info, Mapping):
                    continue
                run_key = str(info.get("run_key") or "").strip()
                uuid = str(uuid).strip()
                if not uuid or not run_key:
                    continue
                pair = (uuid, run_key)
                pairs.add(pair)
                ref_info.setdefault(pair, {"path": info.get("path"), "sha256": info.get("sha256")})
        referenced |= pairs
        per_benchmark.append({
            "histoqc_asset_id": entry.name,
            "benchmark_id": manifest.get("benchmark_id") if isinstance(manifest, Mapping) else None,
            "manifest": str(manifest_path),
            "n_referenced_pairs": len(pairs),
        })
    return referenced, ref_info, per_benchmark


def _store_pairs(by_uuid_root: Path) -> Dict[Tuple[str, str], Path]:
    """Map every (uuid, run_key) blob present in the store to its directory."""
    pairs: Dict[Tuple[str, str], Path] = {}
    if not by_uuid_root.exists():
        return pairs
    for uuid_dir in sorted(by_uuid_root.iterdir()):
        if not uuid_dir.is_dir():
            continue
        for run_dir in sorted(uuid_dir.iterdir()):
            if run_dir.is_dir() and not run_dir.name.startswith("."):
                pairs[(uuid_dir.name, run_dir.name)] = run_dir
    return pairs


def main(args) -> int:
    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    histoqc_root = data_root / "processed_assets" / "histoqc"
    store_root = shared_histoqc_store_root(data_root)
    by_uuid = store_root / "by_uuid"

    referenced, ref_info, per_benchmark = _scan_histoqc_manifests(histoqc_root)
    store = _store_pairs(by_uuid)
    orphans = {pair: path for pair, path in store.items() if pair not in referenced}

    orphan_records: List[Dict[str, Any]] = [
        {"slide_file_uuid": u, "run_key": rk, "path": str(p), "bytes": _dir_size_bytes(p)}
        for (u, rk), p in orphans.items()
    ]
    reclaimable_bytes = sum(r["bytes"] for r in orphan_records)

    # Optional integrity check of referenced masks against the manifests (never deletes).
    verify_mismatches: List[Dict[str, Any]] = []
    if args.verify:
        # Recorded mask path re-rooted under the current data_root (missing if not found)
        for pair, info in ref_info.items():
            expected = info.get("sha256")
            p = info.get("path")
            if not expected or not p:
                continue
            mask_path = reroot_under_data_root(data_root, p)
            if not mask_path or not mask_path.exists():
                verify_mismatches.append({"slide_file_uuid": pair[0], "run_key": pair[1], "path": str(mask_path) if mask_path else str(p), "reason": "missing"})
                continue
            observed = sha256_path(mask_path)
            if observed != expected:
                verify_mismatches.append({"slide_file_uuid": pair[0], "run_key": pair[1], "path": str(mask_path), "expected_sha256": expected, "observed_sha256": observed})

    # Refuse --delete if no manifest was found (every blob would look orphaned)
    if args.delete and not per_benchmark and not args.allow_empty_reference:
        raise RuntimeError(
            "Refusing to --delete: no benchmark HistoQC manifests were found under "
            f"{histoqc_root}, so every blob looks orphaned. Re-run without --delete to inspect, "
            "or pass --allow_empty_reference if you really intend to empty the store."
        )

    deleted: List[Dict[str, Any]] = []
    if args.delete:
        for rec in orphan_records:
            shutil.rmtree(rec["path"], ignore_errors=False)
            deleted.append(rec)
            logger.info("Deleted orphan mask blob uuid=%s run_key=%s (%d bytes): %s",
                        rec["slide_file_uuid"], rec["run_key"], rec["bytes"], rec["path"])
        # Remove now-empty per-uuid parent dirs.
        for uuid_dir in sorted({Path(r["path"]).parent for r in deleted}):
            try:
                if uuid_dir.is_dir() and not any(uuid_dir.iterdir()):
                    uuid_dir.rmdir()
            except OSError:
                pass

    summary = {
        "shared_store_root": str(store_root),
        "n_benchmarks_with_manifest": len(per_benchmark),
        "n_referenced_pairs": len(referenced),
        "n_store_blobs": len(store),
        "n_orphan_blobs": len(orphans),
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
        raise RuntimeError(f"{len(verify_mismatches)} referenced mask(s) failed SHA256 verification. See {log_path}")

    print(json.dumps({"status": "ok", **summary, "log": str(log_path)}, indent=2))
    return 0


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ap = argparse.ArgumentParser(description="Prune orphaned blobs from the shared HistoQC mask store (processed_assets/histoqc/_store).")

    ap.add_argument("--delete", action="store_true",
                    help="Delete orphaned mask blobs. Without this flag the run is a dry-run that only reports them.")

    ap.add_argument("--verify", action="store_true",
                    help="Re-check the SHA256 of referenced masks against the benchmark manifests (never deletes).")

    ap.add_argument("--allow_empty_reference", action="store_true",
                    help="Permit --delete even when no benchmark HistoQC manifests are found (would empty the store). Use with care.")

    ap.add_argument("--allow_verify_mismatch", action="store_true",
                    help="Exit 0 even if --verify finds SHA256 mismatches (still recorded in the log).")

    args = ap.parse_args()
    raise SystemExit(main(args))
