#!/usr/bin/env python3
"""
Download the official-primary TCGA slides selected for Task B (gdc-client by default) and check their MD5.
Slides already present with the expected size (or MD5 with --reverify_existing_md5) are reused.

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data /
    - selected_slide_cache/_store/slide_files/by_uuid/<slide_file_uuid>/<file_name>: WSIs, shared by all benchmarks
    - selected_slide_cache/<benchmark_id>/manifests/: gdc_manifest.tsv, selected_slide_download_manifest.tsv,
      selected_slide_checksums.tsv, selected_slide_download_log.json
    - benchmarks/<benchmark_id>/reports/selected_slide_download_report.md
"""

import argparse
import json
import os
import re
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import pandas as pd
import sys
from tqdm import tqdm

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path

from _taskb_common import (
    DEFAULT_GDC_API_BASE,
    GDCClient,
    collect_output_hashes,
    gdc_client_version,
    hashes_path,
    manifest_header,
    read_json,
    read_yaml,
    run_gdc_client_download,
    sha256_path,
    shared_store_root,
    utc_now_iso,
    write_gdc_manifest,
    write_json_atomic,
    write_text_atomic,
    write_tsv_atomic,
)

OFFICIAL_MATCH_LEVELS = {"exact_same_portion", "same_sample", "same_case_primary_tumor", "same_case_primary_tumor_unknown_sublevel"}
ALLOW_BY_DEFAULT_PREFETCH_STATUSES = {
    "passed_prefetch",
    "inconclusive_prefetch",
    "range_not_supported",
    "prefetch_error_retryable",
    # Request errors are inconclusive about the slide (checked later by _10)
    "prefetch_error_final",
    "prefetch_not_run",
    "not_run",
}
CONCLUSIVE_FAILURE_STATUS = "failed_prefetch_conclusive"

DOWNLOAD_MANIFEST_COLUMNS = [
    "slide_file_uuid",
    "case_id",
    "case_uuid",
    "project_id",
    "slide_file_name",
    "file_name",
    "file_size_expected",
    "file_size_observed",
    "md5sum_expected",
    "md5sum_observed",
    "sha256_observed",
    "local_path",
    "download_status",
    "download_attempted",
    "download_error",
    "download_started_at_utc",
    "download_finished_at_utc",
    "gdc_api_base_url",
    "technical_prefetch_status",
    "eligible_for_full_download",
    "prefetch_mpp_x",
    "prefetch_mpp_y",
    "prefetch_exclusion_reason",
    "range_supported",
    "bytes_prefetched",
    "partial_sha256",
    "prefetch_deferred_to_full_validation",
    "override_prefetch_fail_used",
    "benchmark_id",
    "modelgroup_id",
]

CHECKSUM_COLUMNS = [
    "slide_file_uuid",
    "file_name",
    "local_path",
    "file_size_expected",
    "file_size_observed",
    "md5sum_expected",
    "md5sum_observed",
    "md5_ok",
    "sha256_observed",
    "checksum_status",
    "benchmark_id",
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


def _clean_string(x: Any) -> str:
    return str(x).strip() if _nonempty(x) else ""


def _safe_file_name(name: Any, uuid: str) -> str:
    raw = Path(_clean_string(name)).name
    if not raw:
        raw = f"{uuid}.svs"
    # Keep GDC filenames intact except for path separators/control characters.
    raw = re.sub(r"[\x00-\x1f\x7f/\\]+", "_", raw).strip(".")
    return raw or f"{uuid}.svs"


def _ordered(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = pd.NA
    extras = [c for c in out.columns if c not in columns]
    return out.loc[:, list(columns) + extras]


def _resolve_gdc_api_base(args: argparse.Namespace, source_root: Path) -> str:
    if args.gdc_api_base:
        return str(args.gdc_api_base).rstrip("/")
    source_config_path = source_root / "settings" / "source_config.yaml"
    if source_config_path.exists():
        cfg = _load_mapping(source_config_path)
        if cfg.get("gdc_api_base_url"):
            return str(cfg["gdc_api_base_url"]).rstrip("/")
    return DEFAULT_GDC_API_BASE


def _load_prefetch_or_sentinel(cohort_dir: Path, *, allow_missing_prefetch: bool) -> Tuple[pd.DataFrame, Path, Optional[str]]:
    prefetch_path = cohort_dir / "slide_prefetch_manifest.parquet"
    if prefetch_path.exists():
        df = pd.read_parquet(prefetch_path)
        return df, prefetch_path, sha256_path(prefetch_path)
    if not allow_missing_prefetch:
        raise FileNotFoundError(
            f"Missing {prefetch_path}. Run _7_prefetch_slide_technical_metadata.py first, "
            "or pass --allow_missing_prefetch to use explicit not_run sentinels."
        )
    slide_manifest_path = cohort_dir / "slide_manifest.csv"
    if not slide_manifest_path.exists():
        raise FileNotFoundError(slide_manifest_path)
    df = pd.read_csv(slide_manifest_path, dtype=str, keep_default_na=False)
    df["technical_prefetch_status"] = "not_run"
    df["eligible_for_full_download"] = True
    df["prefetch_mpp_x"] = pd.NA
    df["prefetch_mpp_y"] = pd.NA
    df["prefetch_exclusion_reason"] = "prefetch_not_run_by_user_override"
    df["range_supported"] = pd.NA
    df["bytes_received"] = pd.NA
    df["partial_sha256"] = pd.NA
    return df, slide_manifest_path, sha256_path(slide_manifest_path)


def _validate_prefetch_schema(df: pd.DataFrame) -> None:
    required = ["slide_file_uuid", "case_id", "case_uuid", "project_id", "file_name", "md5sum"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"slide_prefetch_manifest is missing required columns: {missing}")
    if df["slide_file_uuid"].astype(str).duplicated().any():
        dups = df.loc[df["slide_file_uuid"].astype(str).duplicated(), "slide_file_uuid"].astype(str).head(20).tolist()
        raise RuntimeError(f"Duplicate slide_file_uuid rows are forbidden: {dups}")


def _row_is_official_primary(row: Mapping[str, Any]) -> bool:
    official = row.get("official_primary_match")
    selected = row.get("selected_for_official_primary")
    level = _clean_string(row.get("biospecimen_match_level"))
    if _nonempty(official):
        return _truthy(official)
    if _nonempty(selected):
        return _truthy(selected)
    return level in OFFICIAL_MATCH_LEVELS


def _prefetch_decision(row: Mapping[str, Any], *, override_prefetch_fail: bool) -> Tuple[bool, str, bool]:
    status = _clean_string(row.get("technical_prefetch_status")) or "not_run"
    eligible = row.get("eligible_for_full_download")
    eligible_bool = True if not _nonempty(eligible) else _truthy(eligible)
    if status == CONCLUSIVE_FAILURE_STATUS or not eligible_bool:
        if override_prefetch_fail:
            return True, "override_prefetch_failure", True
        return False, "conclusive_prefetch_failure_or_ineligible", False
    if status in ALLOW_BY_DEFAULT_PREFETCH_STATUSES:
        return True, "allowed_by_prefetch_policy", False
    # Unknown statuses: downloaded only with --override_prefetch_fail
    if override_prefetch_fail:
        return True, f"override_unknown_prefetch_status:{status}", True
    return False, f"unknown_prefetch_status:{status}", False


def _write_report(path: Path, summary: Mapping[str, Any], *, overwrite: bool) -> None:
    lines = [
        "# Selected slide download report",
        "",
        f"- benchmark_id: `{summary.get('benchmark_id')}`",
        f"- modelgroup_id: `{summary.get('modelgroup_id')}`",
        f"- slides considered: `{summary.get('n_considered')}`",
        f"- slides selected for download: `{summary.get('n_selected_for_download')}`",
        f"- downloaded: `{summary.get('n_downloaded')}`",
        f"- existing size-ok (fast resume): `{summary.get('n_skipped_existing_size_ok')}`",
        f"- existing MD5-ok (full verify): `{summary.get('n_skipped_existing_md5_ok')}`",
        f"- skipped by policy: `{summary.get('n_skipped_by_policy')}`",
        f"- failed: `{summary.get('n_failed')}`",
        "",
        "Full OpenSlide validation is done by `_10_build_tile_asset.py`.",
        "",
    ]
    write_text_atomic(path, "\n".join(lines), overwrite=overwrite)


def _store_hygiene(selected_subset: Sequence[Mapping[str, Any]]) -> List[str]:
    """Remove the gdc-client leftovers (e.g. *.partial, logs/) from the by_uuid dirs of the given slides, keeping
    only the WSI. Returns the removed paths."""
    removed: List[str] = []
    for s in selected_subset:
        uuid_dir = Path(s["dest"]).parent
        keep = Path(s["dest"]).name
        if not uuid_dir.exists():
            continue
        for entry in sorted(uuid_dir.iterdir()):
            if entry.name == keep:
                continue
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed.append(str(entry))
            except Exception as e:
                removed.append(f"{entry} (failed to remove: {e})")
    return removed


def main(args) -> int:

    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    settings_dir = benchmark_root / "settings"
    cohort_dir = benchmark_root / "shared" / "cohort"
    reports_dir = benchmark_root / "reports"
    cache_root = data_root / "selected_slide_cache" / args.benchmark_id
    # WSIs are stored in the shared store; local_path in the per-benchmark manifest points into it
    slide_files_root = shared_store_root(data_root) / "slide_files" / "by_uuid"
    manifests_dir = cache_root / "manifests"
    if not settings_dir.exists():
        raise FileNotFoundError(settings_dir)
    if not cohort_dir.exists():
        raise FileNotFoundError(cohort_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    slide_files_root.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        manifests_dir / "selected_slide_download_manifest.tsv",
        manifests_dir / "selected_slide_checksums.tsv",
        manifests_dir / "selected_slide_download_log.json",
    ]
    if any(p.exists() for p in outputs) and not args.force:
        existing = [str(p) for p in outputs if p.exists()]
        raise FileExistsError(f"Download manifests already exist; use --force to rebuild. Existing: {existing}")
    downstream = [data_root / "processed_assets" / "tiles"]
    if args.force and any(p.exists() for p in downstream) and not args.force_cascade:
        raise RuntimeError(f"Refusing to rebuild slide download manifests while downstream tile assets exist: {downstream}. Use --force_cascade if intentional.")

    cfg = _load_mapping(settings_dir / "benchmark_config.yaml")
    links_path = settings_dir / "benchmark_asset_links.json"
    links = read_json(links_path) if links_path.exists() else {}
    if not isinstance(links, Mapping):
        raise RuntimeError("benchmark_asset_links.json must be a mapping when present")
    modelgroup_id = str(cfg.get("modelgroup_id") or links.get("modelgroup_id"))
    source_root = Path(links.get("source_release_root") or data_root / "source_releases" / str(cfg.get("tcga_source_id")))
    gdc_api_base = _resolve_gdc_api_base(args, source_root)

    prefetch_df, prefetch_input_path, prefetch_hash = _load_prefetch_or_sentinel(cohort_dir, allow_missing_prefetch=bool(args.allow_missing_prefetch))
    _validate_prefetch_schema(prefetch_df)
    if args.limit is not None:
        prefetch_df = prefetch_df.sort_values(["project_id", "case_id", "slide_file_uuid"]).head(args.limit).copy()

    client = GDCClient(base_url=gdc_api_base, timeout=int(args.timeout), retries=int(args.retries), backoff=float(args.backoff))
    try:
        gdc_status = client.status()
    except Exception as e:
        gdc_status = {"status_query_error": str(e)}

    rows: List[Dict[str, Any]] = []
    checksums: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []

    sorted_rows = prefetch_df.sort_values(["project_id", "case_id", "slide_file_uuid"]).to_dict("records")

    # ------------------------------------------------------------------
    # Phase A -- selection: one manifest_row per slide; slides to download are queued in `selected`
    # ------------------------------------------------------------------
    selected: List[Dict[str, Any]] = []
    for row in sorted_rows:
        uuid = _clean_string(row.get("slide_file_uuid"))
        file_name = _safe_file_name(row.get("slide_file_name") or row.get("file_name"), uuid)
        dest = slide_files_root / uuid / file_name
        expected_md5 = _clean_string(row.get("md5sum"))
        expected_size = int(float(row.get("file_size"))) if _nonempty(row.get("file_size")) else None
        prefetch_status = _clean_string(row.get("technical_prefetch_status")) or "not_run"
        allowed_prefetch, prefetch_reason, override_used = _prefetch_decision(row, override_prefetch_fail=bool(args.override_prefetch_fail))
        is_official = _row_is_official_primary(row)

        if prefetch_status == "inconclusive_prefetch" and not args.download_inconclusive_prefetch_slides:
            allowed_prefetch = False
            prefetch_reason = "inconclusive_prefetch_disabled_by_cli"
        if prefetch_status == "range_not_supported" and not args.download_range_not_supported_slides:
            allowed_prefetch = False
            prefetch_reason = "range_not_supported_download_disabled_by_cli"

        manifest_row: Dict[str, Any] = {
            "slide_file_uuid": uuid,
            "case_id": row.get("case_id"),
            "case_uuid": row.get("case_uuid"),
            "project_id": row.get("project_id"),
            "slide_file_name": file_name,
            "file_name": file_name,
            "file_size_expected": expected_size,
            "file_size_observed": pd.NA,
            "md5sum_expected": expected_md5,
            "md5sum_observed": pd.NA,
            "sha256_observed": pd.NA,
            "local_path": str(dest),
            "download_status": None,
            "download_attempted": False,
            "download_error": "",
            "download_started_at_utc": None,
            "download_finished_at_utc": None,
            "gdc_api_base_url": gdc_api_base,
            "technical_prefetch_status": prefetch_status,
            "eligible_for_full_download": row.get("eligible_for_full_download"),
            "prefetch_mpp_x": row.get("prefetch_mpp_x"),
            "prefetch_mpp_y": row.get("prefetch_mpp_y"),
            "prefetch_exclusion_reason": row.get("prefetch_exclusion_reason"),
            "range_supported": row.get("range_supported"),
            "bytes_prefetched": row.get("bytes_received"),
            "partial_sha256": row.get("partial_sha256"),
            "prefetch_deferred_to_full_validation": prefetch_status in {"inconclusive_prefetch", "range_not_supported", "prefetch_error_retryable", "prefetch_error_final", "not_run", "prefetch_not_run"},
            "override_prefetch_fail_used": bool(override_used),
            "benchmark_id": args.benchmark_id,
            "modelgroup_id": modelgroup_id,
        }
        rows.append(manifest_row)

        if not is_official:
            manifest_row.update({"download_status": "skipped_not_official_primary", "download_error": "row_not_marked_official_primary_in_cohort"})
            continue
        if not allowed_prefetch:
            manifest_row.update({"download_status": "skipped_by_prefetch_policy", "download_error": prefetch_reason})
            continue
        if not expected_md5:
            manifest_row.update({"download_status": "failed_missing_expected_md5", "download_error": "md5sum missing from manifest"})
            logs.append({"slide_file_uuid": uuid, "status": "failed_missing_expected_md5", "error": "md5sum missing"})
            continue

        selected.append({
            "uuid": uuid, "file_name": file_name, "dest": dest,
            "expected_md5": expected_md5, "expected_size": expected_size, "row": manifest_row,
        })

    # ------------------------------------------------------------------
    # Phase B/C -- check the slides already present: reused if valid, else re-downloaded with
    # --repair_corrupt_shared_blob (the run aborts otherwise)
    # ------------------------------------------------------------------
    # uuid -> {"mode": "size"|"md5", "md5", "sha256", "size"}: how a present slide was accepted
    pre_existing_ok: Dict[str, Dict[str, Any]] = {}
    to_download: List[Dict[str, Any]] = []
    repaired_corrupt: List[str] = []
    corrupt_blobs: List[Dict[str, str]] = []
    for s in tqdm(selected, desc="Checking existing slides", unit="slide"):
        dest = s["dest"]
        uuid = s["uuid"]
        expected = s["expected_md5"]
        expected_size = s["expected_size"]
        if not dest.exists():
            to_download.append(s)
            continue
        # Default: accept a present slide with the expected size (MD5 checked with --reverify_existing_md5 or unknown size)
        if not args.reverify_existing_md5 and expected_size is not None:
            observed_size = int(dest.stat().st_size)
            if observed_size == expected_size:
                pre_existing_ok[uuid] = {"mode": "size", "md5": None, "sha256": None, "size": observed_size}
                continue
            if args.repair_corrupt_shared_blob:
                dest.unlink()
                repaired_corrupt.append(uuid)
                to_download.append(s)
            else:
                corrupt_blobs.append({"slide_file_uuid": uuid, "path": str(dest),
                                      "observed": f"size={observed_size}", "expected": f"size={expected_size}"})
            continue
        # Strict / fallback path: full single-pass md5+sha256 (one read, both hashes).
        observed_md5, observed_sha = hashes_path(dest)
        observed_md5 = observed_md5.lower()
        if observed_md5 == expected.lower():
            pre_existing_ok[uuid] = {"mode": "md5", "md5": observed_md5, "sha256": observed_sha, "size": int(dest.stat().st_size)}
            continue
        if args.repair_corrupt_shared_blob:
            dest.unlink()
            repaired_corrupt.append(uuid)
            to_download.append(s)
        else:
            corrupt_blobs.append({"slide_file_uuid": uuid, "path": str(dest),
                                  "observed": f"md5={observed_md5}", "expected": f"md5={expected}"})
    if corrupt_blobs:
        listing = "\n".join(
            f"  {c['slide_file_uuid']}: {c['path']} (observed {c['observed']}, expected {c['expected']})"
            for c in corrupt_blobs[:20]
        )
        raise FileExistsError(
            f"{len(corrupt_blobs)} shared-store WSI(s) exist with the wrong MD5/size; "
            f"re-run with --repair_corrupt_shared_blob to re-download them.\n{listing}"
        )

    print(
        f"[download] benchmark '{args.benchmark_id}': {len(selected)} selected, "
        f"{len(to_download)} to transfer ({len(pre_existing_ok)} already present md5-ok), "
        f"backend={args.transfer_backend}, via {gdc_api_base}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Phase D/E/F -- write the GDC manifest of the missing slides and download them (gdc-client or python_stream)
    # ------------------------------------------------------------------
    gdc_manifest_path = manifests_dir / "gdc_manifest.tsv"
    # Always written (header only if nothing to download)
    write_gdc_manifest(
        gdc_manifest_path,
        [{"id": s["uuid"], "filename": s["file_name"], "md5": s["expected_md5"], "size": s["expected_size"]} for s in to_download],
        overwrite=True,
    )
    transfer_record: Dict[str, Any] = {"backend": args.transfer_backend, "n_files_to_transfer": len(to_download)}
    transfer_started = utc_now_iso()
    gdc_version: Optional[str] = None
    stream_errors: Dict[str, str] = {}

    if to_download:
        if args.transfer_backend == "gdc_client":
            # Check that gdc-client is installed
            gdc_version = gdc_client_version(args.gdc_client_bin)
            if gdc_version == "unknown" and shutil.which(args.gdc_client_bin) is None:
                raise RuntimeError(
                    f"gdc-client not found ('{args.gdc_client_bin}'). Install the GDC Data Transfer Tool "
                    "(https://gdc.cancer.gov/access-data/gdc-data-transfer-tool), pass its path with "
                    "--gdc_client_bin, or fall back to --transfer_backend python_stream."
                )
            rec = run_gdc_client_download(
                manifest_path=gdc_manifest_path,
                dest_dir=slide_files_root,
                bin_name=args.gdc_client_bin,
                n_processes=int(args.n_processes),
                retry_amount=int(args.retry_amount),
                wait_time=int(args.wait_time),
                gdc_api_base=gdc_api_base,
                extra_args=list(args.gdc_client_extra_args or []),
                cwd=manifests_dir,
                timeout=args.transfer_timeout,
                capture=bool(args.capture_gdc_output),
            )
            transfer_record.update(rec)
            transfer_record["gdc_client_version"] = gdc_version
            # Keep only the WSI in each by_uuid dir (checked by _prune_shared_slide_store.py --verify)
            transfer_record["hygiene_removed"] = _store_hygiene(to_download)
        else:
            # python_stream: one file at a time with the Python GDC client (files in to_download are absent)
            for s in to_download:
                try:
                    logs.append(client.download_file(s["uuid"], s["dest"], expected_md5=s["expected_md5"], overwrite=True))
                except Exception as e:
                    stream_errors[s["uuid"]] = str(e)
                    logs.append({
                        "file_uuid": s["uuid"], "destination": str(s["dest"]),
                        "status": "failed_download_or_checksum", "expected_md5": s["expected_md5"],
                        "error": str(e), "traceback": traceback.format_exc(limit=5),
                    })
            transfer_record["n_stream_errors"] = len(stream_errors)
    else:
        transfer_record.update({"status": "no_files_to_transfer", "note": "all selected slides already present md5-ok, or none selected"})
        if args.transfer_backend == "gdc_client":
            gdc_version = gdc_client_version(args.gdc_client_bin)
            transfer_record["gdc_client_version"] = gdc_version

    transfer_record["repaired_corrupt_uuids"] = repaired_corrupt
    transfer_record["n_pre_existing_md5_ok"] = len(pre_existing_ok)

    # ------------------------------------------------------------------
    # Phase H -- check the MD5 of the downloaded slides (OpenSlide/MPP checks are done by _10)
    # ------------------------------------------------------------------
    for s in tqdm(selected, desc="Verifying transferred slides", unit="slide"):
        row = s["row"]
        uuid = s["uuid"]
        dest = s["dest"]
        expected_md5 = s["expected_md5"]
        file_name = s["file_name"]
        expected_size = s["expected_size"]
        row["download_started_at_utc"] = transfer_started
        if uuid in pre_existing_ok:
            info = pre_existing_ok[uuid]
            observed_md5 = info["md5"]
            observed_sha = info["sha256"]
            observed_size = info["size"]
            row["download_attempted"] = False
            if info["mode"] == "size":
                # Accepted on file size (no md5/sha256 computed)
                status = "skipped_existing_size_ok"
                md5_ok_flag = pd.NA
                checksum_status = "size_ok_not_hashed"
            else:
                status = "skipped_existing_md5_ok"
                md5_ok_flag = True
                checksum_status = "md5_ok"
        else:
            row["download_attempted"] = True
            if not dest.exists():
                err = stream_errors.get(uuid) or "file not present after transfer (see transfer log)"
                row.update({"download_status": "failed_download_or_checksum", "download_error": err, "download_finished_at_utc": utc_now_iso()})
                logs.append({"file_uuid": uuid, "destination": str(dest), "status": "failed_download_or_checksum", "expected_md5": expected_md5, "error": err})
                continue
            observed_md5, observed_sha = hashes_path(dest)
            observed_md5 = observed_md5.lower()
            if observed_md5 != expected_md5.lower():
                # MD5 mismatch after download: remove the file
                try:
                    dest.unlink()
                except Exception:
                    pass
                err = f"md5 mismatch after transfer: expected {expected_md5}, observed {observed_md5} (corrupt download removed)"
                row.update({"download_status": "failed_download_or_checksum", "download_error": err, "download_finished_at_utc": utc_now_iso()})
                logs.append({"file_uuid": uuid, "destination": str(dest), "status": "failed_download_or_checksum", "expected_md5": expected_md5, "error": err})
                continue
            observed_size = int(dest.stat().st_size)
            status = "downloaded_md5_ok"
            md5_ok_flag = True
            checksum_status = "md5_ok"

        row.update({
            "download_status": status,
            "file_size_observed": observed_size,
            "md5sum_observed": observed_md5 if observed_md5 is not None else pd.NA,
            "sha256_observed": observed_sha if observed_sha is not None else pd.NA,
            "download_finished_at_utc": utc_now_iso(),
        })
        checksums.append({
            "slide_file_uuid": uuid,
            "file_name": file_name,
            "local_path": str(dest),
            "file_size_expected": expected_size,
            "file_size_observed": observed_size,
            "md5sum_expected": expected_md5,
            "md5sum_observed": observed_md5 if observed_md5 is not None else pd.NA,
            "md5_ok": md5_ok_flag,
            "sha256_observed": observed_sha if observed_sha is not None else pd.NA,
            "checksum_status": checksum_status,
            "benchmark_id": args.benchmark_id,
        })
        logs.append({
            "file_uuid": uuid,
            "destination": str(dest),
            "status": status,
            "expected_md5": expected_md5,
            "md5": observed_md5,
            "sha256": observed_sha,
            "bytes": observed_size,
            "start_time_utc": transfer_started,
            "end_time_utc": utc_now_iso(),
        })

    manifest_df = _ordered(pd.DataFrame(rows), DOWNLOAD_MANIFEST_COLUMNS)
    checksum_df = _ordered(pd.DataFrame(checksums), CHECKSUM_COLUMNS)
    status_counts = manifest_df["download_status"].value_counts(dropna=False).to_dict() if not manifest_df.empty else {}
    summary = {
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": modelgroup_id,
        "n_considered": int(len(prefetch_df)),
        "n_selected_for_download": int(manifest_df["download_status"].isin(["downloaded_md5_ok", "downloaded_no_expected_md5", "skipped_existing_md5_ok", "skipped_existing_size_ok"]).sum()) if not manifest_df.empty else 0,
        "n_downloaded": int((manifest_df["download_status"] == "downloaded_md5_ok").sum()) if not manifest_df.empty else 0,
        "n_skipped_existing_size_ok": int((manifest_df["download_status"] == "skipped_existing_size_ok").sum()) if not manifest_df.empty else 0,
        "n_skipped_existing_md5_ok": int((manifest_df["download_status"] == "skipped_existing_md5_ok").sum()) if not manifest_df.empty else 0,
        "n_skipped_by_policy": int((manifest_df["download_status"].astype(str).str.startswith("skipped_") & ~manifest_df["download_status"].isin(["skipped_existing_md5_ok", "skipped_existing_size_ok"])).sum()) if not manifest_df.empty else 0,
        "n_failed": int(manifest_df["download_status"].astype(str).str.startswith("failed_").sum()) if not manifest_df.empty else 0,
        "download_status_counts": {str(k): int(v) for k, v in status_counts.items()},
    }

    write_tsv_atomic(manifests_dir / "selected_slide_download_manifest.tsv", manifest_df, overwrite=args.force)
    write_tsv_atomic(manifests_dir / "selected_slide_checksums.tsv", checksum_df, overwrite=args.force)

    log_obj = manifest_header(Path(__file__).name, vars(args))
    log_obj.update({
        "benchmark_id": args.benchmark_id,
        "benchmark_root": str(benchmark_root),
        "selected_slide_cache_root": str(cache_root),
        "shared_slide_store_root": str(shared_store_root(data_root)),
        "gdc_api_base_url": gdc_api_base,
        "gdc_status_response": gdc_status,
        "gdc_client_version": gdc_version,
        # Transfer backend, command and outcome
        "transfer": transfer_record,
        "input_paths": {
            "prefetch_or_slide_manifest": str(prefetch_input_path),
            "benchmark_config": str(settings_dir / "benchmark_config.yaml"),
            "benchmark_asset_links": str(links_path) if links_path.exists() else None,
            "gdc_manifest": str(gdc_manifest_path) if gdc_manifest_path.exists() else None,
        },
        "input_hashes": {
            "prefetch_or_slide_manifest": prefetch_hash,
            "benchmark_config": sha256_path(settings_dir / "benchmark_config.yaml"),
            "benchmark_asset_links": sha256_path(links_path) if links_path.exists() else None,
            "gdc_manifest": sha256_path(gdc_manifest_path) if gdc_manifest_path.exists() else None,
        },
        "download_policy": {
            "transfer_backend": str(args.transfer_backend),
            "download_inconclusive_prefetch_slides": bool(args.download_inconclusive_prefetch_slides),
            "download_range_not_supported_slides": bool(args.download_range_not_supported_slides),
            "download_conclusive_prefetch_failures": bool(args.override_prefetch_fail),
            "reverify_existing_md5": bool(args.reverify_existing_md5),
            "skip_existing_file_rule": "full_md5_match" if args.reverify_existing_md5 else "size_equality_else_full_md5_when_size_unknown",
            "checksum_rule": "never_accept_md5_mismatch",
        },
        "summary": summary,
        "download_logs": logs,
        # Shared-store path of each slide
        "shared_store_paths": {str(r.get("slide_file_uuid")): str(r.get("local_path")) for r in rows if r.get("local_path")},
        "output_hashes": collect_output_hashes(cache_root),
    })
    write_json_atomic(manifests_dir / "selected_slide_download_log.json", log_obj, overwrite=args.force)
    _write_report(reports_dir / "selected_slide_download_report.md", summary, overwrite=args.force)

    if summary["n_failed"] > 0 and not args.allow_partial_downloads:
        raise RuntimeError(f"{summary['n_failed']} slide downloads failed. Logs written to {manifests_dir / 'selected_slide_download_log.json'}")

    print(json.dumps({"status": "ok", **summary, "download_manifest": str(manifests_dir / "selected_slide_download_manifest.tsv")}, indent=2))
    return 0


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Download the frozen official-primary TCGA slides selected for Task B.")

    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help="Benchmark ID whose selected slides are downloaded.")

    ap.add_argument("--gdc_api_base", type=str,
                    default=None,
                    help="Base URL for the GDC API. Defaults to the project default.")

    ap.add_argument("--timeout", type=int,
                    default=120,
                    help="HTTP request timeout in seconds (status query and python_stream backend).")

    ap.add_argument("--retries", type=int,
                    default=4,
                    help="Retries per request (status query and python_stream backend; gdc-client uses "
                         "--retry_amount).")

    ap.add_argument("--backoff", type=float,
                    default=1.5,
                    help="Exponential backoff factor for the python_stream fallback / status preflight.")

    # --- Transfer backend (gdc-client by default) -------------------------------------
    ap.add_argument("--transfer_backend", type=str,
                    choices=["gdc_client", "python_stream"],
                    default="gdc_client",
                    help="Download backend: 'gdc_client' (GDC Data Transfer Tool) or 'python_stream' "
                         "(single-stream Python client).")

    ap.add_argument("--gdc_client_bin", type=str,
                    default="gdc-client",
                    help="gdc-client executable name or path (gdc_client backend).")

    ap.add_argument("--n_processes", type=int,
                    default=8,
                    help="Number of gdc-client download processes (-n).")

    ap.add_argument("--retry_amount", type=int,
                    default=5,
                    help="gdc-client per-file retry count (--retry-amount) for transient errors.")

    ap.add_argument("--wait_time", type=int,
                    default=5,
                    help="gdc-client seconds to wait between retries (--wait-time).")

    ap.add_argument("--transfer_timeout", type=int,
                    default=None,
                    help="Timeout (seconds) of the whole gdc-client transfer (default: none).")

    ap.add_argument("--capture_gdc_output", action="store_true",
                    help="Record gdc-client stdout/stderr in the log instead of streaming it to the terminal.")

    ap.add_argument("--gdc_client_extra_args", type=str, nargs="*",
                    default=[],
                    help="Extra raw arguments appended to the gdc-client download command.")

    ap.add_argument("--download_inconclusive_prefetch_slides", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Download slides whose prefetch result was inconclusive (default true).")

    ap.add_argument("--download_range_not_supported_slides", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Download slides whose host did not support HTTP range requests during prefetch (default true).")

    ap.add_argument("--override_prefetch_fail", action="store_true",
                    help="Download conclusive prefetch failures; recorded in the manifest.")

    ap.add_argument("--allow_missing_prefetch", action="store_true",
                    help="Use explicit not_run prefetch sentinels from slide_manifest.csv.")

    ap.add_argument("--allow_partial_downloads", action="store_true",
                    help="Write logs and exit 0 even if some downloads fail.")

    ap.add_argument("--limit", type=int,
                    default=None,
                    help="Debug only: process only the first N sorted rows.")

    ap.add_argument("--force", action="store_true",
                    help="Regenerate the per-benchmark download manifests and report (shared-store WSIs are "
                         "not re-downloaded).")

    ap.add_argument("--force_cascade", action="store_true",
                    help="Required with --force if tile assets already exist.")

    ap.add_argument("--repair_corrupt_shared_blob", action="store_true",
                    help="Re-download shared-store WSIs whose MD5 or file size is wrong.")

    ap.add_argument("--reverify_existing_md5", action="store_true",
                    help="Check the MD5 of already-present WSIs instead of their file size (always done if "
                         "the size is unknown).")

    args = ap.parse_args()
    raise SystemExit(main(args))
