#!/usr/bin/env python3
"""
Build the RNA expression asset (log10(1 + FPKM-UQ) per resolved gene symbol) of a TCGA Task B benchmark: the
selected RNA TSVs (final_case_manifest.csv, else provisional_case_manifest.csv, else rna_manifest.csv cases)
are downloaded, checked (MD5) and parsed.

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data /
    - processed_assets/rna/<rna_asset_id>/: case_expression_log10_fpkmuq.parquet, gene_manifest.parquet,
      rna_parser_report.csv, multiple_rna_files_per_case_audit.csv, rna_asset_manifest.json
    - source_releases/<tcga_source_id>/raw_cache/: downloaded RNA TSVs (rna_files/by_uuid/) and download logs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd
from tqdm import tqdm

import sys

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path

from _taskb_common import (
    DEFAULT_GDC_API_BASE,
    GDCClient,
    collect_output_hashes,
    deterministic_rna_asset_id,
    keep_ordered_columns,
    load_alias_resolver,
    manifest_header,
    md5_path,
    parse_star_counts_expression,
    read_json,
    read_yaml,
    rna_eligibility_config,
    rna_parser_config,
    select_pre_download_eligible_rna_files,
    sha256_obj,
    sha256_path,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
    write_tsv_atomic,
    infer_date_from_alias_filename,
    utc_now_iso,
    version_ok_uuids,
)

EXPR_META_COLUMNS = [
    "case_id", "case_uuid", "project_id", "rna_file_uuid", "rna_file_name",
    "sample_id", "sample_submitter_id", "sample_type", "portion_id", "portion_submitter_id",
    "analyte_id", "analyte_submitter_id", "aliquot_id", "aliquot_submitter_id",
    "workflow_type", "workflow_version", "data_release", "version",
    "n_eligible_rna_files_for_case", "canonical_rna_selection_policy",
]

GENE_MANIFEST_COLUMNS = [
    "resolved_symbol", "n_files_detected", "n_contributing_rows_total", "n_duplicate_files",
    "contributing_ensembl_ids", "contributing_gene_names", "contributing_gene_types",
    "alias_resolution_statuses", "alias_map_sha256", "duplicate_symbol_policy",
]

CHECKSUM_COLUMNS = ["rna_file_uuid", "file_name", "path", "expected_md5", "observed_md5", "sha256", "status"]


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


def _load_mapping(path: Path) -> Dict[str, Any]:
    obj = read_json(path) if path.suffix.lower() == ".json" else read_yaml(path)
    if not isinstance(obj, Mapping):
        raise RuntimeError(f"Expected mapping at {path}")
    return dict(obj)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def optional_hash(path: Path | None) -> str | None:
    return sha256_path(path) if path is not None and path.exists() else None


def load_annotation_excluded_entities(source_root: Path) -> Set[str]:
    excl_path = source_root / "inventory" / "annotation_exclusion_log.csv"
    if not excl_path.exists():
        return set()
    df = pd.read_csv(excl_path, dtype=str, keep_default_na=False)
    if df.empty or "entity_id" not in df.columns:
        return set()
    return set(str(x).strip() for x in df["entity_id"].tolist() if _nonempty(x))


def _benchmark_paths(data_root: Path, benchmark_id: str) -> Dict[str, Path]:
    root = data_root / "benchmarks" / benchmark_id
    return {
        "benchmark_root": root,
        "settings_dir": root / "settings",
        "cohort_dir": root / "shared" / "cohort",
        "benchmark_config": root / "settings" / "benchmark_config.yaml",
        "benchmark_asset_links": root / "settings" / "benchmark_asset_links.json",
        "rna_manifest": root / "shared" / "cohort" / "rna_manifest.csv",
        "provisional_case_manifest": root / "shared" / "cohort" / "provisional_case_manifest.csv",
        "final_case_manifest": root / "shared" / "cohort" / "final_case_manifest.csv",
    }


def _resolve_source_root_and_asset_id(
    *,
    data_root: Path,
    tcga_source_id: Optional[str],
    rna_asset_id: Optional[str],
    benchmark_id: Optional[str],
) -> Tuple[Path, Optional[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {"benchmark_id": benchmark_id}
    if benchmark_id:
        paths = _benchmark_paths(data_root, benchmark_id)
        if not paths["benchmark_config"].exists():
            raise FileNotFoundError(paths["benchmark_config"])
        cfg = _load_mapping(paths["benchmark_config"])
        links = read_json(paths["benchmark_asset_links"]) if paths["benchmark_asset_links"].exists() else {}
        if not isinstance(links, Mapping):
            raise RuntimeError("benchmark_asset_links.json must be a mapping")
        source_id = str(tcga_source_id or cfg.get("tcga_source_id") or links.get("tcga_source_id"))
        source_root = Path(links.get("source_release_root") or data_root / "source_releases" / source_id)
        cfg_rna_asset_id = str(rna_asset_id or cfg.get("rna_asset_id") or links.get("rna_asset_id") or "") or None
        if rna_asset_id and cfg.get("rna_asset_id") and str(rna_asset_id) != str(cfg.get("rna_asset_id")):
            raise RuntimeError(f"--rna_asset_id {rna_asset_id!r} does not match benchmark_config.yaml rna_asset_id {cfg.get('rna_asset_id')!r}")
        meta.update({
            "benchmark_root": str(paths["benchmark_root"]),
            "benchmark_config_hash": optional_hash(paths["benchmark_config"]),
            "benchmark_asset_links_hash": optional_hash(paths["benchmark_asset_links"]),
            "configured_rna_asset_id": cfg_rna_asset_id,
        })
        return source_root, cfg_rna_asset_id, meta
    if not tcga_source_id:
        raise RuntimeError("--tcga_source_id is required when --benchmark_id is not supplied")
    return data_root / "source_releases" / tcga_source_id, rna_asset_id, meta


def _select_from_benchmark(data_root: Path, benchmark_id: str, explicit_manifest: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return the narrowest available benchmark RNA selection."""
    paths = _benchmark_paths(data_root, benchmark_id)
    cohort_dir = paths["cohort_dir"]
    if explicit_manifest is not None:
        df = _read_table(explicit_manifest)
        return df, {
            "selection_source": "explicit_rna_selection_manifest",
            "selection_path": str(explicit_manifest),
            "selection_hash": sha256_path(explicit_manifest),
            "selection_stage": "explicit_user_supplied",
        }

    rna_manifest_path = paths["rna_manifest"]
    if not rna_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {rna_manifest_path}. Run _6_resolve_rna_and_slides.py before building a benchmark-specific RNA asset."
        )
    rna_manifest = _read_table(rna_manifest_path)
    if "rna_file_uuid" not in rna_manifest.columns:
        raise RuntimeError(f"rna_manifest.csv missing rna_file_uuid: {rna_manifest_path}")

    final_path = paths["final_case_manifest"]
    if final_path.exists():
        final = _read_table(final_path)
        if "rna_file_uuid" not in final.columns:
            raise RuntimeError(f"final_case_manifest.csv missing rna_file_uuid: {final_path}")
        mask = final["final_case_eligible"].map(_truthy) if "final_case_eligible" in final.columns else pd.Series(True, index=final.index)
        final_sel = final.loc[mask].copy()
        selected_uuids = set(final_sel["rna_file_uuid"].dropna().astype(str))
        df = rna_manifest.loc[rna_manifest["rna_file_uuid"].astype(str).isin(selected_uuids)].copy()
        # Preserve the exact final case order and ensure no final cases are lost.
        order = final_sel[["case_id", "rna_file_uuid"]].drop_duplicates().reset_index(drop=True)
        order["__order__"] = range(len(order))
        df = df.merge(order, on=["case_id", "rna_file_uuid"], how="right", validate="one_to_one")
        missing = df.loc[df["project_id"].isna() if "project_id" in df.columns else df["rna_file_uuid"].isna(), ["case_id", "rna_file_uuid"]].to_dict("records")
        if missing:
            raise RuntimeError(f"Final RNA UUIDs missing from rna_manifest.csv. Examples: {missing[:5]}")
        df = df.sort_values("__order__").drop(columns=["__order__"]).reset_index(drop=True)
        return df, {
            "selection_source": "benchmark_final_case_manifest",
            "selection_path": str(final_path),
            "selection_hash": sha256_path(final_path),
            "rna_manifest_path": str(rna_manifest_path),
            "rna_manifest_hash": sha256_path(rna_manifest_path),
            "selection_stage": "final_case_manifest_after_tile_qc",
            "n_final_case_rows": int(len(final)),
            "n_selected_final_cases": int(len(final_sel)),
        }

    provisional_path = paths["provisional_case_manifest"]
    if provisional_path.exists():
        prov = _read_table(provisional_path)
        if "case_id" not in prov.columns:
            raise RuntimeError(f"provisional_case_manifest.csv missing case_id: {provisional_path}")
        if "official_primary_provisional" in prov.columns:
            mask = prov["official_primary_provisional"].map(_truthy)
        elif "n_official_primary_slides_selected" in prov.columns:
            mask = pd.to_numeric(prov["n_official_primary_slides_selected"], errors="coerce").fillna(0).gt(0)
        else:
            raise RuntimeError("provisional_case_manifest.csv has neither official_primary_provisional nor n_official_primary_slides_selected")
        keep_cases = set(prov.loc[mask, "case_id"].dropna().astype(str))
        df = rna_manifest.loc[rna_manifest["case_id"].astype(str).isin(keep_cases)].copy()
        return df.reset_index(drop=True), {
            "selection_source": "benchmark_provisional_case_manifest",
            "selection_path": str(provisional_path),
            "selection_hash": sha256_path(provisional_path),
            "rna_manifest_path": str(rna_manifest_path),
            "rna_manifest_hash": sha256_path(rna_manifest_path),
            "selection_stage": "provisional_official_primary_after_slide_metadata",
            "n_provisional_case_rows": int(len(prov)),
            "n_selected_provisional_cases": int(len(keep_cases)),
        }

    return rna_manifest.copy().reset_index(drop=True), {
        "selection_source": "benchmark_rna_manifest",
        "selection_path": str(rna_manifest_path),
        "selection_hash": sha256_path(rna_manifest_path),
        "selection_stage": "metadata_rna_manifest_without_slide_case_filter",
        "warning": "final/provisional case manifests were unavailable; this may parse RNA for cases later removed by slide filters",
    }


def _load_source_inventory_for_asset(source_root: Path, selected_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    inventory_dir = source_root / "inventory"
    candidate_path = inventory_dir / "candidate_rna_files.parquet"
    selected_path = inventory_dir / "selected_rna_files.parquet"
    if selected_df is not None:
        if "rna_file_uuid" not in selected_df.columns:
            raise RuntimeError("Benchmark RNA selection missing rna_file_uuid")
        if not candidate_path.exists():
            raise FileNotFoundError(candidate_path)
        inv = pd.read_parquet(candidate_path)
        uuids = selected_df["rna_file_uuid"].dropna().astype(str).tolist()
        wanted = set(uuids)
        out = inv.loc[inv["rna_file_uuid"].astype(str).isin(wanted)].copy()
        missing = sorted(wanted - set(out["rna_file_uuid"].astype(str)))
        if missing:
            raise RuntimeError(f"Selected RNA UUIDs missing from source candidate inventory. Examples: {missing[:10]}")
        order = pd.DataFrame({"rna_file_uuid": uuids, "__order__": range(len(uuids))}).drop_duplicates("rna_file_uuid")
        out = out.merge(order, on="rna_file_uuid", how="inner", validate="one_to_one").sort_values("__order__").drop(columns=["__order__"])
        # Carry benchmark-side selection annotations if present.
        for col in ["n_eligible_rna_files_for_case", "rna_selection_rank", "rna_selection_reason", "benchmark_id"]:
            if col in selected_df.columns and col not in out.columns:
                out = out.merge(selected_df[["rna_file_uuid", col]].drop_duplicates("rna_file_uuid"), on="rna_file_uuid", how="left")
        return out.reset_index(drop=True), {
            "inventory_source": "candidate_rna_files.parquet_filtered_by_benchmark_selection",
            "candidate_rna_files_hash": sha256_path(candidate_path),
            "n_inventory_rows_before_filter": int(len(inv)),
            "n_inventory_rows_after_filter": int(len(out)),
        }

    if selected_path.exists():
        return pd.read_parquet(selected_path), {"inventory_source": "selected_rna_files.parquet", "selected_rna_files_hash": sha256_path(selected_path)}
    if candidate_path.exists():
        return pd.read_parquet(candidate_path), {"inventory_source": "candidate_rna_files.parquet", "candidate_rna_files_hash": sha256_path(candidate_path)}
    raise FileNotFoundError(f"Missing RNA inventory under {inventory_dir}")


def _metadata_eligible_rna_files(source_root: Path, inv_df: pd.DataFrame, inventory_meta: Mapping[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    versions_path = source_root / "inventory" / "file_versions.parquet"
    if not versions_path.exists():
        raise FileNotFoundError(versions_path)
    versions = pd.read_parquet(versions_path)
    excluded_entities = load_annotation_excluded_entities(source_root)

    required = ["rna_file_uuid", "file_name", "case_id", "case_uuid", "project_id", "workflow_type", "sample_type", "access", "state", "parent_resolution_status"]
    missing = [c for c in required if c not in inv_df.columns]
    if missing:
        raise RuntimeError(f"RNA inventory missing columns: {missing}")
    if inv_df["rna_file_uuid"].duplicated().any():
        dups = inv_df.loc[inv_df["rna_file_uuid"].duplicated(), "rna_file_uuid"].astype(str).head(10).tolist()
        raise RuntimeError(f"Duplicate RNA file UUIDs are forbidden: {dups}")

    versioned = version_ok_uuids(versions, allow_low_confidence_fallback=False, require_latest=False)
    out = select_pre_download_eligible_rna_files(
        inv_df,
        excluded_entities=excluded_entities,
        versioned_uuids=versioned,
        require_versioned=True,
    )
    out = out.sort_values(["project_id", "case_id", "rna_file_uuid"]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("No eligible RNA files after applying frozen metadata/annotation/version rules")
    meta = dict(inventory_meta)
    meta.update({
        "n_version_acceptable_uuids": int(len(versioned)),
        "n_annotation_excluded_entities": int(len(excluded_entities)),
        "n_eligible_rows_for_asset_before_download": int(len(out)),
        "n_eligible_cases_for_asset_before_download": int(out["case_id"].nunique()),
    })
    return out, meta


def _download_set_name(benchmark_id: Optional[str], rna_asset_id: str) -> str:
    raw = benchmark_id or rna_asset_id or "source_release"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(raw))


def ensure_downloaded_rna_files(
    *,
    source_root: Path,
    rna_df: pd.DataFrame,
    gdc_api_base: str,
    overwrite: bool,
    download_missing: bool,
    download_set_name: str,
    limit: Optional[int] = None,
) -> Tuple[Set[str], Dict[str, Any]]:
    """Ensure selected RNA TSVs exist and return verified UUIDs plus manifest metadata."""
    rows = rna_df.sort_values("rna_file_uuid").to_dict("records")
    if limit is not None:
        rows = rows[:limit]
    client = GDCClient(gdc_api_base) if download_missing else None
    logs: List[Dict[str, Any]] = []
    checksums: List[Dict[str, Any]] = []
    verified: Set[str] = set()

    print(
        f"Downloading/verifying {len(rows)} RNA expression file(s) from GDC (one at a time; "
        "cached files are skipped).",
        flush=True,
    )
    for row in tqdm(rows, desc="Downloading/verifying RNA files", unit="file"):
        uuid = str(row["rna_file_uuid"])
        fname = str(row["file_name"])
        expected = str(row.get("md5sum") or "").strip().lower()
        dest = source_root / "raw_cache" / "rna_files" / "by_uuid" / uuid / fname
        if dest.exists():
            observed = md5_path(dest).lower()
            if expected and observed == expected:
                sha = sha256_path(dest)
                verified.add(uuid)
                log = {
                    "file_uuid": uuid,
                    "destination": str(dest),
                    "status": "skipped_existing_md5_ok",
                    "expected_md5": expected,
                    "md5": observed,
                    "sha256": sha,
                    "end_time_utc": utc_now_iso(),
                }
            elif not overwrite:
                raise RuntimeError(f"Existing RNA TSV has wrong MD5 and overwrite is disabled: {dest}")
            else:
                if not download_missing:
                    raise RuntimeError(f"RNA TSV missing/invalid and --no-download_missing_rna was set: {dest}")
                assert client is not None
                log = client.download_file(uuid, dest, expected_md5=expected or None, overwrite=True)
                observed = str(log.get("md5") or "").lower()
                sha = str(log.get("sha256") or "")
                if expected and observed == expected:
                    verified.add(uuid)
        else:
            if not download_missing:
                raise FileNotFoundError(f"RNA TSV missing and --no-download_missing_rna was set: {dest}")
            assert client is not None
            log = client.download_file(uuid, dest, expected_md5=expected or None, overwrite=overwrite)
            observed = str(log.get("md5") or "").lower()
            sha = str(log.get("sha256") or "")
            if expected and observed == expected:
                verified.add(uuid)
        logs.append(log)
        checksums.append({
            "rna_file_uuid": uuid,
            "file_name": fname,
            "path": str(dest),
            "expected_md5": expected,
            "observed_md5": str(log.get("md5") or "").lower(),
            "sha256": str(log.get("sha256") or ""),
            "status": str(log.get("status") or ""),
        })

    checksums_df = keep_ordered_columns(pd.DataFrame(checksums), CHECKSUM_COLUMNS)
    download_dir = source_root / "raw_cache" / "rna_download_sets" / download_set_name
    download_dir.mkdir(parents=True, exist_ok=True)
    checksum_path = download_dir / "checksums.tsv"
    log_path = download_dir / "download_log.json"
    write_tsv_atomic(checksum_path, checksums_df, overwrite=overwrite)
    write_json_atomic(log_path, {"downloads": logs, "created_at_utc": utc_now_iso(), "download_set_name": download_set_name}, overwrite=overwrite)

    failures = [r for r in checksums if r["rna_file_uuid"] not in verified]
    if failures:
        raise RuntimeError(f"RNA download/checksum verification failed for {len(failures)} files. Examples: {failures[:5]}")
    return verified, {
        "download_set_name": download_set_name,
        "checksum_path": str(checksum_path),
        "download_log_path": str(log_path),
        "checksums_hash": sha256_path(checksum_path),
        "download_log_hash": sha256_path(log_path),
        "n_requested_rna_files": int(len(rows)),
        "n_verified_downloaded_uuids": int(len(verified)),
        "download_missing_enabled": bool(download_missing),
        "download_limit": limit,
    }


def new_gene_accumulator_entry() -> Dict[str, Any]:
    """Empty gene-manifest accumulator entry of one resolved_symbol: contributor sets and counters."""
    return {
        "ensembl_ids": set(),
        "gene_names": set(),
        "gene_types": set(),
        "statuses": set(),
        "n_files_detected": 0,
        "n_contributing_rows_total": 0,
        "n_duplicate_files": 0,
    }


def fold_gene_manifest_rows(acc: Dict[str, Dict[str, Any]], gene_manifest_rows: Iterable[Mapping[str, Any]]) -> int:
    """Fold one file's per-symbol gene rows into the accumulator acc (one row per symbol and file).
    Returns the number of rows folded."""
    n = 0
    for gr in gene_manifest_rows:
        sym = gr["resolved_symbol"]
        a = acc.get(sym)
        if a is None:
            a = new_gene_accumulator_entry()
            acc[sym] = a
        a["ensembl_ids"].update(json.loads(gr["contributing_ensembl_ids"]))
        a["gene_names"].update(json.loads(gr["contributing_gene_names"]))
        a["gene_types"].update(json.loads(gr["contributing_gene_types"]))
        a["statuses"].update(json.loads(gr["alias_resolution_statuses"]))
        a["n_files_detected"] += 1
        nrows = int(gr["n_contributing_rows"])
        a["n_contributing_rows_total"] += nrows
        if nrows > 1:
            a["n_duplicate_files"] += 1
        n += 1
    return n


def aggregate_gene_manifest(acc: Mapping[str, Dict[str, Any]], alias_hash: str, duplicate_policy: str) -> pd.DataFrame:
    """Build the gene manifest (one row per resolved symbol, sorted) from the accumulator."""
    if not acc:
        return pd.DataFrame(columns=GENE_MANIFEST_COLUMNS)
    rows = []
    for sym in sorted(acc.keys()):
        a = acc[sym]
        rows.append({
            "resolved_symbol": sym,
            "n_files_detected": int(a["n_files_detected"]),
            "n_contributing_rows_total": int(a["n_contributing_rows_total"]),
            "n_duplicate_files": int(a["n_duplicate_files"]),
            "contributing_ensembl_ids": json.dumps(sorted(a["ensembl_ids"]), sort_keys=True),
            "contributing_gene_names": json.dumps(sorted(a["gene_names"]), sort_keys=True),
            "contributing_gene_types": json.dumps(sorted(a["gene_types"]), sort_keys=True),
            "alias_resolution_statuses": json.dumps(sorted(a["statuses"]), sort_keys=True),
            "alias_map_sha256": alias_hash,
            "duplicate_symbol_policy": duplicate_policy,
        })
    return keep_ordered_columns(pd.DataFrame(rows), GENE_MANIFEST_COLUMNS)


def build_components(
    source_root: Path,
    alias_map_path: Path,
    *,
    selection_meta: Mapping[str, Any],
    eligibility_meta: Mapping[str, Any],
    download_meta: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    source_config_path = source_root / "settings" / "source_config.yaml"
    annotation_policy_path = source_root / "settings" / "annotation_policy.yaml"
    if not source_config_path.exists():
        raise FileNotFoundError(source_config_path)
    if not annotation_policy_path.exists():
        raise FileNotFoundError(annotation_policy_path)
    alias_date = infer_date_from_alias_filename(alias_map_path)
    return {
        "tcga_source_id": source_root.name,
        "source_config_hash": sha256_path(source_config_path),
        "candidate_rna_files_hash": optional_hash(source_root / "inventory" / "candidate_rna_files.parquet"),
        "source_selected_rna_files_hash": optional_hash(source_root / "inventory" / "selected_rna_files.parquet"),
        "file_versions_hash": sha256_path(source_root / "inventory" / "file_versions.parquet"),
        "annotation_exclusion_log_hash": optional_hash(source_root / "inventory" / "annotation_exclusion_log.csv"),
        "benchmark_selection": dict(selection_meta),
        "eligibility_meta": dict(eligibility_meta),
        "rna_download_set": dict(download_meta or {}),
        "rna_eligibility_config_hash": sha256_obj(rna_eligibility_config()),
        "rna_parser_config_hash": sha256_obj(rna_parser_config()),
        "annotation_policy_hash": sha256_path(annotation_policy_path),
        "hgnc_alias_map_hash": sha256_path(alias_map_path),
        "alias_map_date": alias_date,
        "expression_transform": "log10(1 + FPKM-UQ)",
        "duplicate_resolved_symbol_rule": "sum_prelog_fpkm_uq_then_log10_1_plus_sum",
        "parser_software_version": rna_parser_config()["parser_software_version"],
    }


def _read_prior_deterministic_id(out_root: Path) -> Optional[str]:
    """deterministic_rna_asset_id_pre_download recorded by a previous build of this rna_asset_id, or None."""
    manifest_path = out_root / "rna_asset_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        prior = read_json(manifest_path)
    except Exception:
        return None
    if not isinstance(prior, Mapping):
        return None
    value = prior.get("deterministic_rna_asset_id_pre_download")
    return str(value) if value is not None else None


def main(args) -> int:

    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    source_root, configured_rna_asset_id, benchmark_link_meta = _resolve_source_root_and_asset_id(
        data_root=data_root,
        tcga_source_id=args.tcga_source_id,
        rna_asset_id=args.rna_asset_id,
        benchmark_id=args.benchmark_id,
    )
    if not source_root.exists():
        raise FileNotFoundError(source_root)

    alias_candidates = sorted((source_root / "settings").glob("gene_alias_map_hgnc_*.tsv"))
    if len(alias_candidates) != 1:
        raise RuntimeError(f"Expected exactly one frozen HGNC alias map in source settings, found {len(alias_candidates)}")
    alias_map_path = alias_candidates[0]

    selected_df: Optional[pd.DataFrame] = None
    selection_meta: Dict[str, Any] = {"selection_source": "source_release_inventory", "selection_stage": "source_pre_slide_selection_or_candidate_inventory"}
    if args.benchmark_id or args.rna_selection_manifest is not None:
        if not args.benchmark_id and args.rna_selection_manifest is None:
            raise RuntimeError("Internal argument error")
        if args.benchmark_id:
            selected_df, selection_meta = _select_from_benchmark(data_root, args.benchmark_id, args.rna_selection_manifest)
        else:
            selected_df = _read_table(args.rna_selection_manifest)  # type: ignore[arg-type]
            selection_meta = {
                "selection_source": "explicit_rna_selection_manifest",
                "selection_stage": "explicit_user_supplied",
                "selection_path": str(args.rna_selection_manifest),
                "selection_hash": sha256_path(args.rna_selection_manifest),  # type: ignore[arg-type]
            }
        if selected_df.empty:
            raise RuntimeError("RNA selection is empty; no RNA TSVs will be downloaded/parsed")

    inv_df, inventory_meta = _load_source_inventory_for_asset(source_root, selected_df)
    eligible, eligibility_meta = _metadata_eligible_rna_files(source_root, inv_df, inventory_meta)
    if args.download_limit is not None:
        eligible = eligible.sort_values(["project_id", "case_id", "rna_file_uuid"]).head(args.download_limit).reset_index(drop=True)

    preliminary_components = build_components(
        source_root,
        alias_map_path,
        selection_meta=selection_meta,
        eligibility_meta=eligibility_meta,
        download_meta=None,
    )
    derived_id = deterministic_rna_asset_id(source_root.name, preliminary_components)
    rna_asset_id = args.rna_asset_id or configured_rna_asset_id or derived_id
    if configured_rna_asset_id and str(rna_asset_id) != str(configured_rna_asset_id):
        raise RuntimeError(f"Resolved RNA asset ID {rna_asset_id!r} does not match benchmark-configured ID {configured_rna_asset_id!r}")

    out_root = data_root / "processed_assets" / "rna" / str(rna_asset_id)
    downstream = list((data_root / "benchmarks").glob("*/settings/benchmark_asset_links.json"))
    if out_root.exists() and not args.force:
        raise FileExistsError(f"RNA asset already exists: {out_root}. Use --force to rebuild.")
    if out_root.exists() and args.force and downstream and not args.force_cascade:
        raise RuntimeError("Refusing to rebuild RNA asset while benchmark asset links exist; pass --force_cascade if intentional.")

    # Compare the deterministic id with the one recorded by a previous build of the same rna_asset_id
    # (read before the manifest is overwritten)
    prior_pre_download_id = _read_prior_deterministic_id(out_root)
    if prior_pre_download_id is None:
        print(json.dumps({"info": "RNA asset content fingerprint: first build under this rna_asset_id; recording baseline for future drift checks", "rna_asset_id": str(rna_asset_id), "deterministic_id_pre_download": derived_id}, indent=2))
    elif str(prior_pre_download_id) != str(derived_id):
        print(json.dumps({"warning": "RNA asset content fingerprint drifted from the previous build of this rna_asset_id: the cohort / eligibility / config / alias-map inputs changed.", "rna_asset_id": str(rna_asset_id), "previous_deterministic_id_pre_download": str(prior_pre_download_id), "current_deterministic_id_pre_download": derived_id}, indent=2))
    else:
        print(json.dumps({"info": "RNA asset content fingerprint matches the previous build (inputs reproduced identically)", "rna_asset_id": str(rna_asset_id), "deterministic_id_pre_download": derived_id}, indent=2))

    out_root.mkdir(parents=True, exist_ok=True)

    download_meta: Dict[str, Any] = {"skipped_for_debug": bool(args.allow_metadata_only_source)}
    if args.allow_metadata_only_source:
        verified_uuids: Set[str] = set(eligible["rna_file_uuid"].astype(str))
    else:
        verified_uuids, download_meta = ensure_downloaded_rna_files(
            source_root=source_root,
            rna_df=eligible,
            gdc_api_base=args.gdc_api_base,
            overwrite=args.force,
            download_missing=args.download_missing_rna,
            download_set_name=_download_set_name(args.benchmark_id, str(rna_asset_id)),
            limit=None,
        )
        eligible = eligible[eligible["rna_file_uuid"].astype(str).isin(verified_uuids)].copy()
        if eligible.empty:
            raise RuntimeError("No verified downloaded RNA files remained for parsing")

    components = build_components(
        source_root,
        alias_map_path,
        selection_meta=selection_meta,
        eligibility_meta=eligibility_meta,
        download_meta=download_meta,
    )
    final_derived_id = deterministic_rna_asset_id(source_root.name, components)

    resolver = load_alias_resolver(alias_map_path, date_hint=components.get("alias_map_date"), source="source_release_frozen_HGNC_alias_map")

    expression_rows: List[Dict[str, Any]] = []
    parser_reports: List[Dict[str, Any]] = []
    # Gene manifest accumulated per resolved symbol, file by file
    gene_acc: Dict[str, Dict[str, Any]] = {}
    n_gene_rows_folded = 0
    for r in tqdm(eligible.to_dict("records"), desc="Parsing RNA files", unit="file"):
        uuid = str(r["rna_file_uuid"])
        fname = str(r["file_name"])
        path = source_root / "raw_cache" / "rna_files" / "by_uuid" / uuid / fname
        if not path.exists():
            if args.allow_metadata_only_source:
                continue
            raise FileNotFoundError(path)
        parsed = parse_star_counts_expression(path, resolver)
        meta = {c: r.get(c) for c in EXPR_META_COLUMNS if c in r}
        meta["rna_file_name"] = fname
        row = {**meta, **parsed.expression}
        expression_rows.append(row)
        rep = dict(parsed.parser_report)
        rep.update({
            "rna_file_uuid": uuid,
            "case_id": r.get("case_id"),
            "case_uuid": r.get("case_uuid"),
            "project_id": r.get("project_id"),
            "sha256": sha256_path(path),
            "alias_map_sha256": resolver.sha256,
        })
        parser_reports.append(rep)
        n_gene_rows_folded += fold_gene_manifest_rows(gene_acc, parsed.gene_manifest_rows)

    if not expression_rows:
        raise RuntimeError("No RNA files were parsed. Check selected cohort, downloads, and --allow_metadata_only_source.")
    expr_df = pd.DataFrame(expression_rows)
    meta_cols = [c for c in EXPR_META_COLUMNS if c in expr_df.columns]
    gene_cols = sorted([c for c in expr_df.columns if c not in meta_cols])
    expr_df = expr_df.loc[:, meta_cols + gene_cols]
    if expr_df["rna_file_uuid"].duplicated().any():
        raise RuntimeError("Duplicate RNA file UUIDs after parsing")
    if expr_df[gene_cols].fillna(0).sum(axis=1).eq(0).any():
        bad = expr_df.loc[expr_df[gene_cols].fillna(0).sum(axis=1).eq(0), "rna_file_uuid"].tolist()
        raise RuntimeError(f"All-zero parsed expression rows: {bad[:10]}")

    case_rna_counts = (
        expr_df.groupby(["project_id", "case_id"], dropna=False)["rna_file_uuid"]
        .nunique()
        .reset_index(name="n_rna_files")
        .sort_values(["project_id", "case_id"])
    )
    multi_case_ids = set(case_rna_counts.loc[case_rna_counts["n_rna_files"] > 1, "case_id"].dropna().astype(str))
    audit_cols = [
        c for c in [
            "project_id", "case_id", "case_uuid", "rna_file_uuid", "rna_file_name",
            "sample_type", "sample_id", "portion_id", "analyte_id", "aliquot_id",
            "workflow_type", "workflow_version", "data_release", "version",
            "n_eligible_rna_files_for_case", "canonical_rna_selection_policy",
        ]
        if c in expr_df.columns
    ]
    multi_rna_audit = expr_df.loc[expr_df["case_id"].astype(str).isin(multi_case_ids), audit_cols].sort_values(["project_id", "case_id", "rna_file_uuid"])

    print(f"Aggregating gene manifest over {n_gene_rows_folded} per-file gene rows "
          f"({len(gene_acc)} distinct resolved symbols) ...", flush=True)
    gene_manifest = aggregate_gene_manifest(gene_acc, resolver.sha256, rna_parser_config()["duplicate_resolved_symbol_rule"])
    parser_report_df = pd.DataFrame(parser_reports).sort_values(["project_id", "case_id", "rna_file_uuid"]).reset_index(drop=True)

    write_parquet_atomic(out_root / "case_expression_log10_fpkmuq.parquet", expr_df, overwrite=args.force)
    write_parquet_atomic(out_root / "gene_manifest.parquet", gene_manifest, overwrite=args.force)
    write_csv_atomic(out_root / "rna_parser_report.csv", parser_report_df, overwrite=args.force)
    write_csv_atomic(out_root / "multiple_rna_files_per_case_audit.csv", multi_rna_audit, overwrite=args.force)

    source_prov = read_json(source_root / "settings" / "provenance.json") if (source_root / "settings" / "provenance.json").exists() else {}
    source_policy = source_prov.get("efficient_source_release_policy", {}) if isinstance(source_prov, Mapping) else {}
    manifest = manifest_header(Path(__file__).name, vars(args))
    manifest.update({
        "rna_asset_id": str(rna_asset_id),
        "deterministic_rna_asset_id_pre_download": derived_id,
        "deterministic_rna_asset_id_with_download_set": final_derived_id,
        "rna_asset_id_matches_deterministic_with_download_set": str(rna_asset_id) == final_derived_id,
        "content_fingerprint_check": {
            "previous_build_deterministic_id_pre_download": prior_pre_download_id,
            "current_deterministic_id_pre_download": derived_id,
            "current_deterministic_id_with_download_set": final_derived_id,
            "verdict": (
                "first_build" if prior_pre_download_id is None
                else "reproduced" if str(prior_pre_download_id) == str(derived_id)
                else "drifted"
            ),
        },
        "tcga_source_id": source_root.name,
        "benchmark_id": args.benchmark_id,
        "source_root": str(source_root),
        "derivation_components": components,
        "source_release_rna_selection_policy": source_policy,
        "selection_meta": selection_meta,
        "eligibility_meta": eligibility_meta,
        "download_meta": download_meta,
        "alias_map_path": str(alias_map_path),
        "alias_map_sha256": resolver.sha256,
        "alias_map_date": resolver.date,
        "alias_map_source": resolver.source,
        "alias_resolution_policy": "frozen_HGNC_alias_previous_symbols_exact_case_insensitive_keep_unmapped_with_audit",
        "duplicate_symbol_policy": rna_parser_config()["duplicate_resolved_symbol_rule"],
        "expression_transform": "log10(1 + FPKM-UQ)",
        "missing_gene_policy": "missing_remains_missing_no_zero_fill",
        "expression_row_level": "rna_file; benchmark-specific builds contain only retained final/provisional cases depending on available cohort manifest",
        "case_level_target_policy": "Downstream target/fold code must use the final benchmark cohort and one row per retained case.",
        "rna_eligibility_config": rna_eligibility_config(),
        "rna_parser_config": rna_parser_config(),
        "n_eligible_rna_files": int(len(eligible)),
        "n_parsed_rna_files": int(len(expr_df)),
        "n_cases_with_expression": int(expr_df["case_id"].nunique()),
        "n_cases_with_multiple_rna_files": int(len(multi_case_ids)),
        "n_gene_columns": int(len(gene_cols)),
        "case_expression_hash": sha256_path(out_root / "case_expression_log10_fpkmuq.parquet"),
        "gene_manifest_hash": sha256_path(out_root / "gene_manifest.parquet"),
        "parser_report_hash": sha256_path(out_root / "rna_parser_report.csv"),
        "multiple_rna_files_per_case_audit_hash": sha256_path(out_root / "multiple_rna_files_per_case_audit.csv"),
        "source_release_provenance_hash": optional_hash(source_root / "settings" / "provenance.json"),
        "output_hashes": collect_output_hashes(out_root),
    })
    write_json_atomic(out_root / "rna_asset_manifest.json", manifest, overwrite=args.force)
    print(json.dumps({
        "status": "ok",
        "rna_asset_root": str(out_root),
        "rna_asset_id": str(rna_asset_id),
        "selection_stage": selection_meta.get("selection_stage"),
        "n_files": len(expr_df),
        "n_cases": int(expr_df["case_id"].nunique()),
        "n_genes": len(gene_cols),
    }, indent=2))
    return 0


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Build a deterministic RNA expression asset for the TCGA Task B benchmark.")

    ap.add_argument("--tcga_source_id", type=str,
                    default=None,
                    help="Source release ID (default: from the benchmark_config.yaml of --benchmark_id).")

    ap.add_argument("--benchmark_id", type=str,
                    default=None,
                    help="Build RNA only for the retained cohort of this benchmark.")

    ap.add_argument("--rna_selection_manifest", type=Path,
                    default=None,
                    help="Optional explicit CSV/TSV/Parquet RNA selection manifest with an rna_file_uuid column.")

    ap.add_argument("--rna_asset_id", type=str,
                    default=None,
                    help="Optional explicit RNA asset ID. With --benchmark_id, defaults to the benchmark_config.yaml rna_asset_id.")

    ap.add_argument("--gdc_api_base", type=str,
                    default=DEFAULT_GDC_API_BASE,
                    help="Base URL for the GDC API. Default: https://api.gdc.cancer.gov")

    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing RNA asset.")

    ap.add_argument("--force_cascade", action="store_true",
                    help="Required with --force when rebuilding with downstream benchmarks present.")

    ap.add_argument("--download_missing_rna", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Download selected RNA TSVs if they are missing from the local source cache (default true).")

    ap.add_argument("--download_limit", type=int,
                    default=None,
                    help="Debug only: parse/download at most N selected RNA files.")

    ap.add_argument("--allow_metadata_only_source", action="store_true",
                    help="Debug only: do not require downloaded RNA TSV files; outputs may be empty and are not official.")

    args = ap.parse_args()
    raise SystemExit(main(args))
