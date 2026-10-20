#!/usr/bin/env python3
"""
Run HistoQC on the downloaded slides of a TCGA Task B benchmark to get their tissue-use masks (used by
_10_build_tile_asset.py). Sharded runs (--shard_index/--total_shards) are consolidated with --finalize_only.

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data /
    - processed_assets/histoqc/_store/by_uuid/<slide_file_uuid>/<histoqc_run_key>/: HistoQC output and
      mask_record.json, shared by all benchmarks
    - processed_assets/histoqc/<histoqc_asset_id>/: histoqc_asset_manifest.json, histoqc_config_snapshot.ini,
      shard_status/
    - benchmarks/<benchmark_id>/reports/histoqc_build_report.md (histoqc keys added to settings/benchmark_asset_links.json)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path

from _taskb_common import (
    histoqc_run_key,
    manifest_header,
    read_json,
    read_shard_records,
    resolve_histoqc_mask_use,
    sha256_obj,
    sha256_path,
    shard_status_path,
    shared_histoqc_slide_dir,
    shared_histoqc_store_root,
    utc_now_iso,
    write_json_atomic,
    write_text_atomic,
)


DEFAULT_HISTOQC_CONFIG = (
    Path(__file__).resolve().parent / "histoqc_configs" / "tcga_taskb_v1.ini"
)
DEFAULT_HISTOQC_TIMEOUT_S = 30 * 60  # 30 minutes per slide


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _detect_histoqc_version() -> str:
    """Return the installed HistoQC version (package metadata, else `histoqc --version`), or 'unknown'."""
    try:
        import importlib.metadata as importlib_metadata
        v = importlib_metadata.version("histoqc").strip()
        if v:
            return v
    except Exception:
        pass
    # Fallback: `--version` stdout, only if rc == 0
    for cmd in (["histoqc", "--version"], [sys.executable, "-m", "histoqc", "--version"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip().splitlines()[0]
        except Exception:
            continue
    return "unknown"


def _load_download_manifest(cache_root: Path) -> pd.DataFrame:
    p = cache_root / "manifests" / "selected_slide_download_manifest.tsv"
    if not p.exists():
        raise FileNotFoundError(
            f"Run _8_download_selected_slides.py first; missing {p}"
        )
    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    required = ["slide_file_uuid", "local_path", "download_status"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"download manifest missing columns: {missing}")
    # skipped_existing_size_ok: blob present, accepted on file-size match (fast resume in _8).
    ok_statuses = {"downloaded_md5_ok", "skipped_existing_md5_ok", "skipped_existing_size_ok"}
    ok = df.loc[df["download_status"].isin(ok_statuses)].copy()
    if ok.empty:
        raise RuntimeError(
            "No downloaded MD5-ok slides found in selected_slide_download_manifest.tsv"
        )
    if ok["slide_file_uuid"].duplicated().any():
        dups = ok.loc[ok["slide_file_uuid"].duplicated(), "slide_file_uuid"].head(20).tolist()
        raise RuntimeError(f"Duplicate downloaded slide_file_uuid rows: {dups}")
    return ok.sort_values(["project_id", "case_id", "slide_file_uuid"], na_position="last").reset_index(drop=True)


def _deterministic_histoqc_asset_id(
    *,
    benchmark_id: str,
    slide_manifest_hash: str,
    histoqc_config_hash: str,
    histoqc_version: str,
) -> str:
    components = {
        "benchmark_id": benchmark_id,
        "slide_manifest_hash": slide_manifest_hash,
        "histoqc_config_hash": histoqc_config_hash,
        "histoqc_version": histoqc_version,
    }
    return f"{benchmark_id}__histoqc__{sha256_obj(components)[:16]}"


def _shared_mask_dir_is_complete(shared_dir: Path, slide_stem: str) -> Optional[Path]:
    """Return the mask_use path if shared_dir holds a complete HistoQC result (status ok, mask present and
    matching its recorded sha256), else None."""
    record_path = shared_dir / "mask_record.json"
    if not record_path.exists():
        return None
    try:
        rec = read_json(record_path)
    except Exception:
        return None
    if not isinstance(rec, Mapping) or rec.get("status") != "ok":
        return None
    mask = resolve_histoqc_mask_use(shared_dir, slide_stem)
    if mask is None:
        return None
    recorded_sha = rec.get("mask_use_sha256")
    actual_sha = sha256_path(mask)
    if recorded_sha and actual_sha != recorded_sha:
        # A record pointing to a macOS ._* file is rewritten to point to the real mask; any other mismatch is rejected
        recorded_relpath = str(rec.get("mask_use_relpath") or "")
        if not (recorded_relpath and Path(recorded_relpath).name.startswith(".")):
            return None
        healed = dict(rec)
        healed["mask_use_relpath"] = str(mask.relative_to(shared_dir))
        healed["mask_use_sha256"] = actual_sha
        write_json_atomic(record_path, healed, overwrite=True)
        print(
            f"[heal] rewrote stale AppleDouble mask_record for {shared_dir.name} "
            f"({recorded_relpath} -> {healed['mask_use_relpath']})",
            file=sys.stderr,
        )
    return mask


def _run_histoqc_on_one_slide(
    *,
    slide_path: Path,
    out_dir: Path,
    config_path: Path,
    timeout_s: int,
) -> Dict[str, Any]:
    """Run HistoQC on one slide with one thread (parallelism comes from sharding)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "histoqc",
        "-c", str(config_path),
        "-o", str(out_dir),
        "-n", "1",
        "--force",  # OK to overwrite within this slide's per-slide dir.
        str(slide_path),
    ]
    t0 = time.time()
    rec: Dict[str, Any] = {
        "command": cmd,
        "started_utc": utc_now_iso(),
    }
    # Private working dir per run (HistoQC writes error.log in the cwd, which concurrent runs would share)
    work_dir = tempfile.mkdtemp(prefix="histoqc_run_")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False,
            cwd=work_dir,
        )
        rec.update({
            "returncode": int(proc.returncode),
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
        })
    except subprocess.TimeoutExpired as e:
        rec.update({
            "returncode": -1,
            "stdout_tail": (e.stdout or "")[-4000:] if e.stdout else "",
            "stderr_tail": f"TIMEOUT after {timeout_s}s",
        })
    except Exception as e:
        rec.update({
            "returncode": -2,
            "stderr_tail": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=6),
        })
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    rec["elapsed_seconds"] = float(time.time() - t0)
    rec["finished_utc"] = utc_now_iso()
    return rec


def _obtain_shared_mask(
    *,
    data_root: Path,
    uuid: str,
    run_key: str,
    slide_path: Path,
    config_path: Path,
    config_hash: str,
    histoqc_version: str,
    timeout_s: int,
    force_recompute: bool,
) -> Dict[str, Any]:
    """Get the slide's HistoQC mask from the shared store, or run HistoQC in a temp dir and publish it atomically.

    Returns a dict with status ("reused_shared", "ok", "failed_histoqc"), shared_dir, mask_use_path,
    mask_use_sha256 and the HistoQC run_record (None on reuse).
    """
    shared_dir = shared_histoqc_slide_dir(data_root, uuid, run_key)
    slide_stem = slide_path.stem

    if not force_recompute:
        reuse_mask = _shared_mask_dir_is_complete(shared_dir, slide_stem)
        if reuse_mask is not None:
            return {
                "status": "reused_shared",
                "shared_dir": shared_dir,
                "mask_use_path": reuse_mask,
                "mask_use_sha256": sha256_path(reuse_mask),
                "run_record": None,
            }

    # Compute into a temp dir on the same filesystem, then atomically publish.
    shared_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = shared_dir.parent / f".{run_key}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    rec_run = _run_histoqc_on_one_slide(
        slide_path=slide_path, out_dir=tmp_dir, config_path=config_path, timeout_s=timeout_s,
    )
    mask = resolve_histoqc_mask_use(tmp_dir, slide_stem)
    ok = mask is not None and rec_run.get("returncode") == 0
    if not ok:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {
            "status": "failed_histoqc",
            "shared_dir": shared_dir,
            "mask_use_path": None,
            "mask_use_sha256": None,
            "run_record": rec_run,
        }

    # Write mask_record.json before publishing
    mask_sha = sha256_path(mask)
    write_json_atomic(tmp_dir / "mask_record.json", {
        "slide_file_uuid": uuid,
        "histoqc_run_key": run_key,
        "histoqc_config_hash": config_hash,
        "histoqc_version": histoqc_version,
        "mask_use_relpath": str(mask.relative_to(tmp_dir)),
        "mask_use_sha256": mask_sha,
        "status": "ok",
        "created_utc": utc_now_iso(),
        "command": rec_run.get("command"),
    }, overwrite=True)

    # Atomic publish (if another worker published it first, its blob is kept)
    published = shared_dir
    if force_recompute and shared_dir.exists():
        shutil.rmtree(shared_dir, ignore_errors=True)
    try:
        os.replace(tmp_dir, shared_dir)
    except OSError:
        if shared_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            raise
    pub_mask = resolve_histoqc_mask_use(published, slide_stem)
    return {
        "status": "ok",
        "shared_dir": published,
        "mask_use_path": pub_mask,
        "mask_use_sha256": sha256_path(pub_mask) if pub_mask is not None else mask_sha,
        "run_record": rec_run,
    }


def _shard_filter(df: pd.DataFrame, *, shard_index: Optional[int], total_shards: Optional[int]) -> pd.DataFrame:
    if shard_index is None and total_shards is None:
        return df
    if shard_index is None or total_shards is None:
        raise RuntimeError("--shard_index and --total_shards must be supplied together")
    if total_shards <= 0:
        raise RuntimeError("--total_shards must be > 0")
    if not (0 <= shard_index < total_shards):
        raise RuntimeError(f"--shard_index out of range: {shard_index} not in [0, {total_shards})")
    # Deterministic round-robin by sorted uuid position (df is pre-sorted by _load_download_manifest).
    return df.iloc[shard_index::total_shards].copy()


def _update_asset_links(
    settings_dir: Path,
    update: Mapping[str, Any],
    *,
    force: bool,
    force_cascade: bool,
) -> Dict[str, Any]:
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
        raise RuntimeError(
            f"Refusing to change existing HistoQC asset links without --force --force_cascade: {conflicts}"
        )
    merged = dict(old_dict)
    merged.update(update)
    old_hash = sha256_path(path)
    write_json_atomic(path, merged, overwrite=True)
    return {
        "path": str(path),
        "previous_sha256": old_hash,
        "new_sha256": sha256_path(path),
        "updated_keys": sorted(update),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _main_impl(args: argparse.Namespace) -> int:
    if args.timeout_seconds <= 0:
        raise RuntimeError("--timeout_seconds must be positive")

    data_root = get_full_path("eval", "eval_datasets", "TCGA_eval_data")
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    settings_dir = benchmark_root / "settings"
    cohort_dir = benchmark_root / "shared" / "cohort"
    cache_root = data_root / "selected_slide_cache" / args.benchmark_id
    for p in (settings_dir, cohort_dir, cache_root):
        if not p.exists():
            raise FileNotFoundError(p)

    slide_manifest_path = cohort_dir / "slide_manifest.csv"
    if not slide_manifest_path.exists():
        raise FileNotFoundError(slide_manifest_path)
    slide_manifest_hash = sha256_path(slide_manifest_path)

    histoqc_config = Path(args.histoqc_config) if args.histoqc_config else DEFAULT_HISTOQC_CONFIG
    if not histoqc_config.exists():
        raise FileNotFoundError(histoqc_config)
    histoqc_config_hash = sha256_path(histoqc_config)
    histoqc_version = _detect_histoqc_version()
    if histoqc_version == "unknown" and not args.allow_unknown_version:
        raise RuntimeError(
            "Could not detect the HistoQC version. Pass --allow_unknown_version to proceed anyway "
            "(the version is part of histoqc_asset_id)."
        )

    # Benchmark-agnostic content key for the per-slide masks in the shared store.
    run_key = histoqc_run_key(histoqc_config_hash, histoqc_version)

    histoqc_asset_id = _deterministic_histoqc_asset_id(
        benchmark_id=args.benchmark_id,
        slide_manifest_hash=slide_manifest_hash,
        histoqc_config_hash=histoqc_config_hash,
        histoqc_version=histoqc_version,
    )
    asset_root = data_root / "processed_assets" / "histoqc" / histoqc_asset_id

    if asset_root.exists() and not (args.force or args.shard_index is not None or args.finalize_only):
        raise FileExistsError(
            f"HistoQC asset already exists; use --force to rebuild, "
            f"or --shard_index/--total_shards to fill in shards, "
            f"or --finalize_only to write the consolidated manifest: {asset_root}"
        )
    if asset_root.exists() and args.force and args.shard_index is None and not args.finalize_only:
        # Rebuild the per-benchmark manifest dir (shared masks: see --force_recompute_shared_mask)
        shutil.rmtree(asset_root)
    asset_root.mkdir(parents=True, exist_ok=True)
    (asset_root / "shard_status").mkdir(parents=True, exist_ok=True)

    # Snapshot of the config
    config_snapshot = asset_root / "histoqc_config_snapshot.ini"
    if not config_snapshot.exists():
        shutil.copy2(histoqc_config, config_snapshot)

    download_df = _load_download_manifest(cache_root)
    if args.limit_slides is not None:
        if not args.force:
            raise RuntimeError("--limit_slides is debug-only and requires --force")
        download_df = download_df.head(int(args.limit_slides)).copy()

    # -------------------------------------------------------------------------
    # Finalize-only fast path: aggregate shard records into the final manifest.
    # -------------------------------------------------------------------------
    if args.finalize_only:
        if args.total_shards is None:
            raise RuntimeError("--finalize_only requires --total_shards")
        rows, completed = read_shard_records(asset_root, int(args.total_shards))
        missing_shards = [i for i in range(int(args.total_shards)) if i not in set(completed)]
        if missing_shards and not args.allow_partial:
            raise RuntimeError(
                f"Missing shard status files for shards: {missing_shards[:20]} "
                f"(of {args.total_shards}). Pass --allow_partial to write a non-frozen manifest."
            )
        return _write_consolidated_manifest_and_links(
            args=args,
            data_root=data_root,
            asset_root=asset_root,
            histoqc_asset_id=histoqc_asset_id,
            histoqc_run_key=run_key,
            histoqc_version=histoqc_version,
            histoqc_config=histoqc_config,
            histoqc_config_hash=histoqc_config_hash,
            slide_manifest_path=slide_manifest_path,
            slide_manifest_hash=slide_manifest_hash,
            settings_dir=settings_dir,
            per_slide_records=rows,
            full_n=int(len(download_df)),
            frozen=not missing_shards,
            missing_shards=missing_shards,
        )

    # -------------------------------------------------------------------------
    # Worker / single-process run
    # -------------------------------------------------------------------------
    shard_df = _shard_filter(
        download_df,
        shard_index=args.shard_index,
        total_shards=args.total_shards,
    )
    per_slide: List[Dict[str, Any]] = []
    n_failed = 0
    n_reused_shared = 0

    for row in shard_df.to_dict("records"):
        uuid = str(row["slide_file_uuid"])
        local_path = Path(str(row["local_path"]))
        if not local_path.exists():
            per_slide.append({
                "slide_file_uuid": uuid,
                "file_name": row.get("file_name"),
                "case_id": row.get("case_id"),
                "project_id": row.get("project_id"),
                "status": "failed_no_downloaded_file",
                "failure_reason": f"missing local_path: {local_path}",
                "histoqc_run_key": run_key,
                "elapsed_seconds": 0.0,
            })
            n_failed += 1
            continue

        res = _obtain_shared_mask(
            data_root=data_root,
            uuid=uuid,
            run_key=run_key,
            slide_path=local_path,
            config_path=histoqc_config,
            config_hash=histoqc_config_hash,
            histoqc_version=histoqc_version,
            timeout_s=int(args.timeout_seconds),
            force_recompute=bool(args.force_recompute_shared_mask),
        )
        status = str(res["status"])
        rec_run = res.get("run_record") or {}
        success = status in ("ok", "reused_shared")
        per_slide.append({
            "slide_file_uuid": uuid,
            "file_name": local_path.name,
            "case_id": row.get("case_id"),
            "project_id": row.get("project_id"),
            "status": status,
            "failure_reason": (
                "" if success else f"returncode={rec_run.get('returncode')}; mask_use={res.get('mask_use_path')}"
            ),
            "histoqc_run_key": run_key,
            "histoqc_output_dir": str(res["shared_dir"]),
            "mask_use_path": str(res["mask_use_path"]) if res.get("mask_use_path") is not None else None,
            "mask_use_sha256": res.get("mask_use_sha256"),
            "elapsed_seconds": rec_run.get("elapsed_seconds", 0.0),
            "returncode": rec_run.get("returncode"),
            "started_utc": rec_run.get("started_utc"),
            "finished_utc": rec_run.get("finished_utc"),
            "stdout_tail": rec_run.get("stdout_tail"),
            "stderr_tail": rec_run.get("stderr_tail"),
        })
        if not success:
            n_failed += 1
        elif status == "reused_shared":
            n_reused_shared += 1

    # Sharded mode: persist this shard's results and stop.
    if args.shard_index is not None:
        shard_obj = {
            "benchmark_id": args.benchmark_id,
            "histoqc_asset_id": histoqc_asset_id,
            "histoqc_run_key": run_key,
            "shard_index": int(args.shard_index),
            "total_shards": int(args.total_shards),
            "n_slides_in_shard": int(len(shard_df)),
            "n_failed_in_shard": int(n_failed),
            "n_reused_shared_in_shard": int(n_reused_shared),
            "per_slide_records": per_slide,
            "finished_utc": utc_now_iso(),
        }
        write_json_atomic(
            shard_status_path(asset_root, shard_index=int(args.shard_index), total_shards=int(args.total_shards)),
            shard_obj,
            overwrite=True,
        )
        print(json.dumps({
            "status": "shard_ok",
            "shard_index": int(args.shard_index),
            "total_shards": int(args.total_shards),
            "n_slides_in_shard": int(len(shard_df)),
            "n_failed_in_shard": int(n_failed),
        }, indent=2))
        return 0

    # Single-process mode: write the consolidated manifest immediately.
    return _write_consolidated_manifest_and_links(
        args=args,
        data_root=data_root,
        asset_root=asset_root,
        histoqc_asset_id=histoqc_asset_id,
        histoqc_run_key=run_key,
        histoqc_version=histoqc_version,
        histoqc_config=histoqc_config,
        histoqc_config_hash=histoqc_config_hash,
        slide_manifest_path=slide_manifest_path,
        slide_manifest_hash=slide_manifest_hash,
        settings_dir=settings_dir,
        per_slide_records=per_slide,
        full_n=int(len(download_df)),
        frozen=True,
        missing_shards=[],
    )


def _write_consolidated_manifest_and_links(
    *,
    args: argparse.Namespace,
    data_root: Path,
    asset_root: Path,
    histoqc_asset_id: str,
    histoqc_run_key: str,
    histoqc_version: str,
    histoqc_config: Path,
    histoqc_config_hash: str,
    slide_manifest_path: Path,
    slide_manifest_hash: str,
    settings_dir: Path,
    per_slide_records: Sequence[Mapping[str, Any]],
    full_n: int,
    frozen: bool,
    missing_shards: Sequence[int],
) -> int:
    # Deduplicate per-slide records by uuid, keeping the last successful run.
    by_uuid: Dict[str, Dict[str, Any]] = {}
    for r in per_slide_records:
        uuid = str(r.get("slide_file_uuid") or "")
        if not uuid:
            continue
        prev = by_uuid.get(uuid)
        if prev is None:
            by_uuid[uuid] = dict(r)
            continue
        # Keep the successful record if a previous shard rerun overwrote a failure
        # ("ok" = freshly computed, "reused_shared" = found in the shared store).
        r_success = str(r.get("status") or "") in ("ok", "reused_shared")
        prev_success = str(prev.get("status") or "") in ("ok", "reused_shared")
        if r_success or not prev_success:
            by_uuid[uuid] = dict(r)
    consolidated = sorted(by_uuid.values(), key=lambda x: (str(x.get("project_id") or ""), str(x.get("case_id") or ""), str(x.get("slide_file_uuid") or "")))

    n_ok = sum(1 for r in consolidated if r.get("status") == "ok")
    n_reused_shared = sum(1 for r in consolidated if r.get("status") == "reused_shared")
    n_failed = sum(1 for r in consolidated if str(r.get("status") or "").startswith("failed"))

    # Per-slide pointers into the shared mask store (read by _10, _14 and _prune_shared_histoqc_store.py)
    referenced_shared_masks = {
        str(r.get("slide_file_uuid")): {
            "run_key": r.get("histoqc_run_key") or histoqc_run_key,
            "path": r.get("mask_use_path"),
            "sha256": r.get("mask_use_sha256"),
        }
        for r in consolidated
        if str(r.get("status") or "") in ("ok", "reused_shared") and r.get("mask_use_path")
    }

    official_final = (
        frozen
        and not missing_shards
        and args.limit_slides is None
        and (n_failed == 0 or args.allow_histoqc_failures)
    )

    summary = {
        "benchmark_id": args.benchmark_id,
        "histoqc_asset_id": histoqc_asset_id,
        "histoqc_run_key": histoqc_run_key,
        "n_downloaded_slides": int(full_n),
        "n_slides_consolidated": int(len(consolidated)),
        "n_ok": int(n_ok),
        "n_reused_shared": int(n_reused_shared),
        "n_failed": int(n_failed),
        "frozen_final": bool(official_final),
        "missing_shards": list(missing_shards),
    }

    manifest = manifest_header(Path(__file__).name, vars(args))
    manifest.update({
        "frozen_final": bool(official_final),
        "status": "frozen_final" if official_final else "debug_or_partial",
        "nonfinal_reason": (
            ""
            if official_final
            else (
                "--limit_slides was used"
                if args.limit_slides is not None
                else ("--allow_partial / missing shards" if missing_shards else "n_failed>0 without --allow_histoqc_failures")
            )
        ),
        "histoqc_asset_id": histoqc_asset_id,
        "histoqc_run_key": histoqc_run_key,
        "shared_store_root": str(shared_histoqc_store_root(data_root)),
        "benchmark_id": args.benchmark_id,
        "input_paths": {
            "slide_manifest_csv": str(slide_manifest_path),
            "histoqc_config": str(histoqc_config),
            "histoqc_config_snapshot": str(asset_root / "histoqc_config_snapshot.ini"),
        },
        "input_hashes": {
            "slide_manifest_csv": slide_manifest_hash,
            "histoqc_config": histoqc_config_hash,
        },
        "histoqc_version": histoqc_version,
        "histoqc_module_list_in_config_hash": True,  # the whole config is hashed
        "deterministic_histoqc_asset_id_components": {
            "benchmark_id": args.benchmark_id,
            "slide_manifest_hash": slide_manifest_hash,
            "histoqc_config_hash": histoqc_config_hash,
            "histoqc_version": histoqc_version,
        },
        "histoqc_run_key_components": {
            "histoqc_config_hash": histoqc_config_hash,
            "histoqc_version": histoqc_version,
        },
        "per_slide_records": consolidated,
        "summary": summary,
        # Shared mask (path, sha256) of each slide
        "referenced_shared_masks": referenced_shared_masks,
    })
    write_json_atomic(asset_root / "histoqc_asset_manifest.json", manifest, overwrite=True)

    # Brief markdown report for humans.
    lines = [
        "# HistoQC asset build report",
        "",
        f"- benchmark_id: `{args.benchmark_id}`",
        f"- histoqc_asset_id: `{histoqc_asset_id}`",
        f"- histoqc_run_key: `{histoqc_run_key}`",
        f"- HistoQC version: `{histoqc_version}`",
        f"- slides ok (newly computed): `{n_ok}`",
        f"- slides reused from shared store: `{n_reused_shared}`",
        f"- slides failed: `{n_failed}`",
        f"- frozen_final: `{official_final}`",
        f"- missing shards: `{list(missing_shards)}`",
        "",
        "Per-slide records and stdout/stderr tails are in `histoqc_asset_manifest.json`.",
        "",
    ]
    reports_dir = (settings_dir.parent / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(reports_dir / "histoqc_build_report.md", "\n".join(lines), overwrite=True)

    if official_final:
        _update_asset_links(
            settings_dir,
            {
                "histoqc_asset_id": histoqc_asset_id,
                "histoqc_asset_root": str(asset_root),
                "histoqc_run_key": histoqc_run_key,
                "histoqc_shared_store_root": str(shared_histoqc_store_root(data_root)),
                "histoqc_config_path": str(histoqc_config),
                "histoqc_config_snapshot_path": str(asset_root / "histoqc_config_snapshot.ini"),
                "histoqc_asset_manifest_path": str(asset_root / "histoqc_asset_manifest.json"),
            },
            force=bool(args.force),
            force_cascade=bool(args.force_cascade),
        )
    else:
        print(
            f"WARNING: HistoQC asset is not frozen_final "
            f"(reason={manifest['nonfinal_reason']!r}); benchmark_asset_links.json was not updated.",
            file=sys.stderr,
        )

    if n_failed > 0 and not args.allow_histoqc_failures:
        raise RuntimeError(
            f"HistoQC failed on {n_failed} slide(s). See histoqc_asset_manifest.json. "
            f"Re-run failed shards individually or pass --allow_histoqc_failures."
        )

    print(json.dumps({
        "status": "ok",
        **summary,
        "histoqc_asset_manifest": str(asset_root / "histoqc_asset_manifest.json"),
    }, indent=2))
    return 0


def main(args: argparse.Namespace) -> int:
    return _main_impl(args)


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Run HistoQC on the downloaded slides of a TCGA Task B benchmark.")

    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help="Benchmark ID whose downloaded slides are QC'd with HistoQC.")

    ap.add_argument("--histoqc_config", type=Path,
                    default=None,
                    help=(
                        "Path to a HistoQC .ini config. Defaults to the bundled "
                        f"{DEFAULT_HISTOQC_CONFIG.relative_to(project_root)} "
                        "(SHA256 hashed into histoqc_asset_id)."
                    ))

    ap.add_argument("--shard_index", type=int,
                    default=None,
                    help="Optional 0-based shard index for SLURM array runs.")

    ap.add_argument("--total_shards", type=int,
                    default=None,
                    help="Total number of shards (must accompany --shard_index).")

    ap.add_argument("--finalize_only", action="store_true",
                    help="Skip per-slide HistoQC and only aggregate previously-written shard_status JSONs into the consolidated manifest.")

    ap.add_argument("--allow_partial", action="store_true",
                    help="With --finalize_only, allow consolidation even if some shard_status files are missing (manifest will be marked non-frozen).")

    ap.add_argument("--allow_histoqc_failures", action="store_true",
                    help="Do not fail the run if any slide HistoQC step returned non-zero or produced no mask.")

    ap.add_argument("--allow_unknown_version", action="store_true",
                    help="Allow running when the HistoQC version cannot be detected.")

    ap.add_argument("--timeout_seconds", type=int,
                    default=DEFAULT_HISTOQC_TIMEOUT_S,
                    help="Per-slide HistoQC timeout in seconds.")

    ap.add_argument("--limit_slides", type=int,
                    default=None,
                    help="Debug only: process only the first N downloaded slides; marks the asset non-frozen.")

    ap.add_argument("--force", action="store_true",
                    help="Rebuild the per-benchmark HistoQC manifest dir (shared masks are reused; see "
                         "--force_recompute_shared_mask).")

    ap.add_argument("--force_recompute_shared_mask", action="store_true",
                    help="Recompute and overwrite the shared HistoQC masks even if valid ones exist.")

    ap.add_argument("--force_cascade", action="store_true",
                    help="Required with --force if benchmark_asset_links.json already contains a different histoqc_asset_id.")

    args = ap.parse_args()
    raise SystemExit(main(args))
