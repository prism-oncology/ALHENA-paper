#!/usr/bin/env python3
"""
Build a TCGA/GDC source release: query the GDC API for TCGA STAR-counts RNA-seq and diagnostic-slide metadata,
apply the metadata, annotation and file-version eligibility checks and select the RNA files per case
(metadata only by default: the RNA TSVs are downloaded by _12_build_rna_asset.py).

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data / source_releases / <tcga_source_id> /
    - settings/: configuration, GDC query payloads and mapping snapshots, HGNC alias map, provenance.json
    - inventory/: RNA, slide and annotation inventories and audit tables
    - raw_cache/: RNA download manifest and checksums (empty unless --download_rna)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import pandas as pd
from tqdm.auto import tqdm
import sys

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path

from _taskb_common import (
    ANNOTATION_COLUMNS,
    DEFAULT_GDC_API_BASE,
    FILE_VERSION_COLUMNS,
    RNA_INVENTORY_COLUMNS,
    SLIDE_INVENTORY_COLUMNS,
    GDCClient,
    annotate_exclusions,
    choose_canonical_rna_per_case,
    collect_output_hashes,
    copy_or_hardlink,
    default_annotation_policy,
    default_source_config,
    field_available,
    flatten_annotation_hit,
    flatten_rna_hit,
    flatten_slide_hit,
    gdc_and,
    gdc_eq,
    gdc_in,
    keep_ordered_columns,
    manifest_header,
    mapping_fields,
    pre_download_rna_eligibility_mask,
    rna_annotation_entity_ids,
    rna_annotation_excluded_mask,
    select_pre_download_eligible_rna_files,
    sha256_obj,
    sha256_path,
    utc_now_iso,
    version_ok_uuids,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
    write_tsv_atomic,
    write_yaml_atomic,
)

RNA_FIELDS_BASE = [
    "file_id", "file_name", "file_size", "md5sum", "version", "data_release", "data_category", "data_type",
    "data_format", "experimental_strategy", "access", "state", "created_datetime", "updated_datetime",
    "analysis.workflow_type", "analysis.workflow_version", "associated_entities.case_id", "associated_entities.entity_id",
    "associated_entities.entity_submitter_id", "associated_entities.entity_type", "cases.case_id", "cases.submitter_id",
    "cases.project.project_id", "cases.project.program.name", "cases.diagnoses.diagnosis_id",
    "cases.diagnoses.diagnosis_is_primary_disease", "cases.diagnoses.primary_diagnosis",
    "cases.diagnoses.tissue_or_organ_of_origin", "cases.samples.sample_id", "cases.samples.submitter_id",
    "cases.samples.sample_type", "cases.samples.tumor_descriptor", "cases.samples.tissue_type",
    "cases.samples.portions.portion_id", "cases.samples.portions.submitter_id", "cases.samples.portions.is_ffpe",
    "cases.samples.portions.analytes.analyte_id", "cases.samples.portions.analytes.submitter_id",
    "cases.samples.portions.analytes.analyte_type", "cases.samples.portions.analytes.aliquots.aliquot_id",
    "cases.samples.portions.analytes.aliquots.submitter_id",
]

SLIDE_FIELDS_BASE = [
    "file_id", "file_name", "file_size", "md5sum", "version", "data_release", "data_category", "data_type",
    "data_format", "experimental_strategy", "access", "state", "created_datetime", "updated_datetime",
    "associated_entities.case_id", "associated_entities.entity_id", "associated_entities.entity_submitter_id",
    "associated_entities.entity_type", "cases.case_id", "cases.submitter_id", "cases.project.project_id",
    "cases.samples.sample_id", "cases.samples.submitter_id", "cases.samples.sample_type",
    "cases.samples.tumor_descriptor", "cases.samples.tissue_type", "cases.samples.portions.portion_id",
    "cases.samples.portions.submitter_id", "cases.samples.portions.is_ffpe", "cases.samples.portions.slides.slide_id",
    "cases.samples.portions.slides.submitter_id", "cases.samples.portions.slides.section_location",
    "cases.samples.portions.slides.state", "cases.samples.portions.slides.percent_tumor_cells",
    "cases.samples.portions.slides.percent_normal_cells", "cases.samples.portions.slides.percent_necrosis",
    "cases.samples.portions.slides.percent_stromal_cells", "cases.samples.portions.slides.percent_lymphocyte_infiltration",
]

ANNOTATION_FIELDS = [
    "annotation_id", "case_id", "case_submitter_id", "entity_id", "entity_submitter_id", "entity_type",
    "classification", "category", "notes", "status", "state", "created_datetime", "updated_datetime",
]

CHECKSUM_COLUMNS = ["rna_file_uuid", "file_name", "path", "expected_md5", "observed_md5", "sha256", "status"]


def print_step(message: str) -> None:
    print(f"\n▶ {message}", flush=True)


def nonempty_strings(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "<na>"}:
            out.append(text)
    return sorted(dict.fromkeys(out))


def build_filters(project_ids: Sequence[str] | None = None, *, gdc_prefilter_primary_samples: bool = False) -> Dict[str, Dict[str, Any]]:
    rna_clauses: List[Mapping[str, Any]] = [
        gdc_eq("cases.project.program.name", "TCGA"),
        gdc_eq("files.data_category", "Transcriptome Profiling"),
        gdc_eq("files.data_type", "Gene Expression Quantification"),
        gdc_eq("files.experimental_strategy", "RNA-Seq"),
        gdc_eq("files.analysis.workflow_type", "STAR - Counts"),
        gdc_eq("files.data_format", "TSV"),
        gdc_eq("files.access", "open"),
        gdc_eq("files.state", "released"),
    ]
    slide_clauses: List[Mapping[str, Any]] = [
        gdc_eq("cases.project.program.name", "TCGA"),
        gdc_eq("files.data_category", "Biospecimen"),
        gdc_eq("files.data_type", "Slide Image"),
        gdc_eq("files.experimental_strategy", "Diagnostic Slide"),
        gdc_eq("files.access", "open"),
        gdc_eq("files.state", "released"),
    ]
    projects = nonempty_strings(project_ids or [])
    if projects:
        rna_clauses.append(gdc_in("cases.project.project_id", projects))
        slide_clauses.append(gdc_in("cases.project.project_id", projects))
    # Optional GDC-side prefilter (the sample type is checked again after parent resolution)
    if gdc_prefilter_primary_samples:
        rna_clauses.append(gdc_in("cases.samples.sample_type", ["Primary Tumor", "Primary Solid Tumor"]))
    return {"rna_files": gdc_and(*rna_clauses), "slide_files": gdc_and(*slide_clauses)}


def optionalize_fields(fields: Sequence[str], mapping: Mapping[str, Any], optional: Sequence[str]) -> List[str]:
    available = set(mapping_fields(mapping))
    out: List[str] = []
    missing_required: List[str] = []
    for f in fields:
        if f in optional and available and f not in available and f"files.{f}" not in available:
            continue
        out.append(f)
    for f in fields:
        if f not in optional and available and f not in available and f"files.{f}" not in available:
            if f.startswith(("file_", "data_", "access", "state", "version", "md5sum", "experimental_strategy")):
                missing_required.append(f)
    if missing_required:
        raise RuntimeError(f"Required fields appear absent from frozen GDC mapping: {missing_required}")
    return out


def write_payloads(
    settings_dir: Path,
    filters: Dict[str, Any],
    rna_fields: List[str],
    slide_fields: List[str],
    page_size: int,
    overwrite: bool,
    *,
    efficient_policy: Mapping[str, Any],
) -> None:
    payload_dir = settings_dir / "gdc_query_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(payload_dir / "rna_source_files.json", {
        "endpoint": "files", "filters": filters["rna_files"], "fields": rna_fields,
        "sort": "file_id:asc", "size": page_size,
        "note": "Broad source query; metadata-only. Download happens only after downstream-equivalent eligibility gates.",
    }, overwrite=overwrite)
    write_json_atomic(payload_dir / "slide_source_files.json", {
        "endpoint": "files", "filters": filters["slide_files"], "fields": slide_fields,
        "sort": "file_id:asc", "size": page_size,
        "note": "May be further restricted to selected RNA cases in chunked queries.",
    }, overwrite=overwrite)
    write_json_atomic(payload_dir / "annotations_for_pre_download_rna.json", {
        "endpoint": "annotations",
        "filters_template": {"op": "in", "content": {"field": "entity_id", "value": "<RNA-relevant entity UUID list>"}},
        "fields": ANNOTATION_FIELDS, "sort": "annotation_id:asc", "size": page_size,
    }, overwrite=overwrite)
    write_json_atomic(payload_dir / "efficient_source_release_policy.json", dict(efficient_policy), overwrite=overwrite)


def query_annotations(
    client: GDCClient,
    entity_ids: Sequence[str],
    fields: List[str],
    page_size: int,
    snapshot_dir: Path | None,
) -> Dict[str, Any]:
    if not entity_ids:
        return {"hits": [], "meta": {"n_hits": 0, "note": "no entity ids"}}
    all_hits: List[Dict[str, Any]] = []
    all_meta: List[Dict[str, Any]] = []
    chunk_size = 1000
    for i in tqdm(range(0, len(entity_ids), chunk_size), desc="Querying GDC annotations", unit="chunk"):
        chunk = list(entity_ids[i : i + chunk_size])
        hits, _, meta = client.search(
            "annotations",
            filters=gdc_in("entity_id", chunk),
            fields=fields,
            sort="annotation_id:asc",
            page_size=page_size,
            query_snapshot_dir=snapshot_dir,
        )
        all_hits.extend(hits)
        all_meta.append(meta)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for h in all_hits:
        aid = h.get("annotation_id") or h.get("id") or sha256_obj(h)
        if aid not in seen:
            seen.add(aid)
            uniq.append(h)
    return {"hits": uniq, "meta": {"chunks": all_meta, "n_hits": len(uniq), "queried_entity_count": len(entity_ids)}}


def query_slide_inventory(
    client: GDCClient,
    *,
    base_filter: Mapping[str, Any],
    slide_fields: List[str],
    page_size: int,
    snapshot_dir: Path | None,
    case_uuids: Sequence[str] | None,
    chunk_size: int,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    all_hits: List[Dict[str, Any]] = []
    all_meta: List[Dict[str, Any]] = []
    if case_uuids is None:
        hits, _, meta = client.search(
            "files", filters=base_filter, fields=slide_fields, sort="file_id:asc",
            page_size=page_size, query_snapshot_dir=snapshot_dir,
        )
        all_hits.extend(hits)
        all_meta.append(meta)
    else:
        cases = nonempty_strings(case_uuids)
        for i in tqdm(range(0, len(cases), chunk_size), desc="Querying slide metadata for selected RNA cases", unit="chunk"):
            chunk = cases[i : i + chunk_size]
            filt = gdc_and(base_filter, gdc_in("cases.case_id", chunk))
            hits, _, meta = client.search(
                "files", filters=filt, fields=slide_fields, sort="file_id:asc",
                page_size=page_size, query_snapshot_dir=snapshot_dir,
            )
            all_hits.extend(hits)
            all_meta.append(meta)
    dedup: Dict[str, Dict[str, Any]] = {}
    for hit in all_hits:
        fid = str(hit.get("file_id") or hit.get("id") or sha256_obj(hit))
        dedup.setdefault(fid, hit)
    df = keep_ordered_columns(pd.DataFrame([flatten_slide_hit(h) for h in dedup.values()]), SLIDE_INVENTORY_COLUMNS)
    meta = {
        "mode": "restricted_to_selected_rna_cases" if case_uuids is not None else "all_matching_diagnostic_slides",
        "case_uuid_count": len(nonempty_strings(case_uuids or [])) if case_uuids is not None else None,
        "n_hits_before_dedup": len(all_hits),
        "n_hits_after_dedup": len(df),
        "chunks": all_meta,
    }
    return df, meta


def project_inventory(rna_df: pd.DataFrame, slide_df: pd.DataFrame, selected_rna_df: pd.DataFrame) -> pd.DataFrame:
    projects = sorted(
        set(rna_df.get("project_id", pd.Series(dtype=str)).dropna().astype(str)) |
        set(slide_df.get("project_id", pd.Series(dtype=str)).dropna().astype(str)) |
        set(selected_rna_df.get("project_id", pd.Series(dtype=str)).dropna().astype(str))
    )
    rows = []
    for p in projects:
        r = rna_df[rna_df["project_id"] == p] if "project_id" in rna_df else pd.DataFrame()
        s = slide_df[slide_df["project_id"] == p] if "project_id" in slide_df else pd.DataFrame()
        sel = selected_rna_df[selected_rna_df["project_id"] == p] if "project_id" in selected_rna_df else pd.DataFrame()
        rows.append({
            "project_id": p,
            "n_candidate_rna_files": int(len(r)),
            "n_candidate_rna_cases": int(r["case_id"].nunique()) if "case_id" in r else 0,
            "n_selected_rna_files_for_download": int(len(sel)),
            "n_selected_rna_cases_for_download": int(sel["case_id"].nunique()) if "case_id" in sel else 0,
            "n_candidate_slide_files": int(len(s)),
            "n_candidate_slide_cases": int(s["case_id"].nunique()) if "case_id" in s else 0,
            "n_resolved_rna_parents": int((r.get("parent_resolution_status") == "resolved").sum()) if not r.empty else 0,
            "n_resolved_slide_parents": int((s.get("parent_resolution_status") == "resolved").sum()) if not s.empty else 0,
        })
    return pd.DataFrame(rows).sort_values("project_id").reset_index(drop=True)


def build_rna_selection_audit(
    rna_df: pd.DataFrame,
    *,
    excluded_entities: Iterable[Any],
    versioned_uuids: Iterable[Any],
    selected_uuids: Iterable[Any],
) -> pd.DataFrame:
    audit = rna_df.copy()
    audit["passes_metadata_eligibility"] = pre_download_rna_eligibility_mask(audit)
    audit["annotation_excluded"] = rna_annotation_excluded_mask(audit, excluded_entities)
    versioned = set(nonempty_strings(versioned_uuids))
    selected = set(nonempty_strings(selected_uuids))
    audit["version_acceptable"] = audit["rna_file_uuid"].astype(str).isin(versioned)
    audit["selected_for_rna_download"] = audit["rna_file_uuid"].astype(str).isin(selected)
    reasons: List[str] = []
    for r in audit.to_dict("records"):
        failed: List[str] = []
        if not r.get("passes_metadata_eligibility"):
            failed.append("metadata_not_eligible")
        if r.get("annotation_excluded"):
            failed.append("annotation_excluded")
        if not r.get("version_acceptable"):
            failed.append("version_not_acceptable_or_not_queried")
        if r.get("passes_metadata_eligibility") and not r.get("annotation_excluded") and r.get("version_acceptable") and not r.get("selected_for_rna_download"):
            failed.append("not_canonical_rna_for_case")
        reasons.append("selected" if r.get("selected_for_rna_download") else ";".join(failed))
    audit["rna_selection_reason"] = reasons
    cols = [
        "project_id", "case_id", "case_uuid", "rna_file_uuid", "file_name", "sample_type",
        "sample_submitter_id", "portion_submitter_id", "analyte_type", "aliquot_submitter_id",
        "is_ffpe", "workflow_type", "access", "state", "parent_resolution_status",
        "diagnosis_is_primary_disease", "passes_metadata_eligibility", "annotation_excluded",
        "version_acceptable", "selected_for_rna_download", "rna_selection_reason",
    ]
    return audit.loc[:, [c for c in cols if c in audit.columns]].sort_values(["project_id", "case_id", "rna_file_uuid"])


def download_rna_files(
    client: GDCClient,
    source_root: Path,
    rna_df: pd.DataFrame,
    *,
    overwrite: bool,
    limit: int | None = None,
) -> Dict[str, Any]:
    logs: List[Dict[str, Any]] = []
    checksums: List[Dict[str, Any]] = []
    rows = rna_df.sort_values("rna_file_uuid").to_dict("records")
    if limit is not None:
        rows = rows[:limit]
    for row in tqdm(rows, desc="Downloading selected RNA files", unit="file"):
        uuid = str(row["rna_file_uuid"])
        fname = str(row["file_name"])
        dest = source_root / "raw_cache" / "rna_files" / "by_uuid" / uuid / fname
        log = client.download_file(uuid, dest, expected_md5=row.get("md5sum"), overwrite=overwrite)
        logs.append(log)
        checksums.append({
            "rna_file_uuid": uuid,
            "file_name": fname,
            "path": str(dest),
            "expected_md5": row.get("md5sum"),
            "observed_md5": log.get("md5"),
            "sha256": log.get("sha256"),
            "status": log.get("status"),
        })
    checksums_df = keep_ordered_columns(pd.DataFrame(checksums), CHECKSUM_COLUMNS)
    write_tsv_atomic(source_root / "raw_cache" / "checksums.tsv", checksums_df, overwrite=overwrite)
    write_json_atomic(source_root / "raw_cache" / "download_log.json", {"downloads": logs, "created_at_utc": utc_now_iso()}, overwrite=overwrite)
    manifest_dir = source_root / "raw_cache" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_tsv_atomic(manifest_dir / "rna_download_manifest.tsv", checksums_df, overwrite=overwrite)
    return {
        "n_download_attempts": len(logs),
        "n_ok": int(sum("md5_ok" in str(l.get("status", "")) or str(l.get("status", "")).endswith("ok") for l in logs)),
        "download_scope": "selected_rna_files",
    }



def main(args) -> int:

    print_step(f"Building source release: {args.tcga_source_id}")
    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    source_root = data_root / "source_releases" / args.tcga_source_id
    settings_dir = source_root / "settings"
    inventory_dir = source_root / "inventory"
    snapshot_dir = inventory_dir / "query_snapshots" if args.save_query_snapshots else None

    if source_root.exists() and not args.force:
        raise FileExistsError(f"Source release already exists: {source_root}. Use --force to rebuild intentionally.")
    source_root.mkdir(parents=True, exist_ok=True)
    settings_dir.mkdir(parents=True, exist_ok=True)
    inventory_dir.mkdir(parents=True, exist_ok=True)
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    print_step("Preparing output directories and HGNC alias map")
    alias_date = args.alias_map_date
    if alias_date is None:
        m = re.search(r"(20\d{6}|19\d{6})", args.hgnc_alias_map.name)
        if not m:
            raise RuntimeError("--alias_map_date is required when the HGNC filename does not contain YYYYMMDD")
        alias_date = m.group(1)
    alias_dst = settings_dir / f"gene_alias_map_hgnc_{alias_date}.tsv"
    alias_src = data_root / "refs" / "hgnc" / args.hgnc_alias_map
    link_mode = copy_or_hardlink(alias_src, alias_dst, overwrite=args.force)
    alias_sha = sha256_path(alias_dst)
    write_json_atomic(settings_dir / f"gene_alias_map_hgnc_{alias_date}.sha256.json", {
        "path": str(alias_dst), "sha256": alias_sha, "alias_map_date": alias_date,
        "alias_map_source": args.alias_map_source, "copy_mode": link_mode,
    }, overwrite=args.force)

    print_step("Contacting GDC API and saving compact mapping snapshots")
    client = GDCClient(args.gdc_api_base)
    status = client.status()
    write_json_atomic(settings_dir / "gdc_api_status.json", status, overwrite=args.force)
    mapping_files = client.mapping("files")
    mapping_cases = client.mapping("cases")
    mapping_annotations = client.mapping("annotations")
    write_json_atomic(settings_dir / "gdc_mapping_snapshot_files.json", mapping_files, overwrite=args.force)
    write_json_atomic(settings_dir / "gdc_mapping_snapshot_cases.json", mapping_cases, overwrite=args.force)
    write_json_atomic(settings_dir / "gdc_mapping_snapshot_annotations.json", mapping_annotations, overwrite=args.force)

    filters = build_filters(args.tcga_projects, gdc_prefilter_primary_samples=args.gdc_prefilter_primary_samples)
    optional_file_fields = ["stain_type", "magnification", "analysis.workflow_version", "version", "data_release"]
    rna_fields = optionalize_fields(RNA_FIELDS_BASE, mapping_files, optional_file_fields)
    slide_fields = optionalize_fields(
        SLIDE_FIELDS_BASE
        + (["stain_type"] if field_available(mapping_files, "stain_type") or field_available(mapping_files, "files.stain_type") else [])
        + (["magnification"] if field_available(mapping_files, "magnification") or field_available(mapping_files, "files.magnification") else []),
        mapping_files,
        optional_file_fields,
    )

    efficient_policy = {
        "rna_download_default": "metadata_only_defer_rna_tsv_download_until_after_benchmark_slide_tile_filtering",
        "rna_per_case": args.rna_per_case,
        "download_scope_if_legacy_download_rna_is_used": args.download_scope,
        "slide_inventory_default": "restricted to selected RNA cases" if args.restrict_slides_to_selected_rna_cases else "all matching slides",
        "raw_query_snapshots_saved": bool(args.save_query_snapshots),
        "tcga_project_filter": nonempty_strings(args.tcga_projects),
        "gdc_prefilter_primary_samples": bool(args.gdc_prefilter_primary_samples),
    }
    write_payloads(settings_dir, filters, rna_fields, slide_fields, args.page_size, args.force, efficient_policy=efficient_policy)

    print_step("Writing source configuration and annotation policy")
    source_config = default_source_config(args.tcga_source_id, gdc_api_base=args.gdc_api_base, alias_map_sha256=alias_sha, alias_map_date=alias_date)
    source_config["stain_type_available"] = bool(field_available(mapping_files, "stain_type") or field_available(mapping_files, "files.stain_type"))
    source_config["magnification_available"] = bool(field_available(mapping_files, "magnification") or field_available(mapping_files, "files.magnification"))
    source_config["efficient_source_release_policy"] = efficient_policy
    write_yaml_atomic(settings_dir / "source_config.yaml", source_config, overwrite=args.force)
    write_yaml_atomic(settings_dir / "annotation_policy.yaml", default_annotation_policy(), overwrite=args.force)

    print_step("Querying GDC RNA file inventory metadata")
    rna_hits, _, rna_meta = client.search(
        "files", filters=filters["rna_files"], fields=rna_fields, sort="file_id:asc",
        page_size=args.page_size, query_snapshot_dir=snapshot_dir,
    )
    rna_df = keep_ordered_columns(pd.DataFrame([flatten_rna_hit(h) for h in rna_hits]), RNA_INVENTORY_COLUMNS)
    print(f"Found {len(rna_df)} candidate RNA metadata rows before local eligibility.", flush=True)
    if not rna_df.empty and rna_df["rna_file_uuid"].duplicated().any():
        raise RuntimeError("Duplicate RNA file UUIDs in flattened inventory")
    write_parquet_atomic(inventory_dir / "candidate_rna_files.parquet", rna_df, overwrite=args.force)

    print_step("Applying pre-download RNA metadata eligibility")
    metadata_eligible_rna = select_pre_download_eligible_rna_files(rna_df)
    write_parquet_atomic(inventory_dir / "metadata_eligible_rna_files.parquet", metadata_eligible_rna, overwrite=args.force)
    if metadata_eligible_rna.empty:
        raise RuntimeError("No RNA files passed metadata eligibility before annotation/version checks")
    print(f"Kept {len(metadata_eligible_rna)} RNA files across {metadata_eligible_rna['case_id'].nunique()} cases after metadata eligibility.", flush=True)

    print_step("Querying file-version metadata only for metadata-eligible RNA files")
    version_query_uuids = sorted(metadata_eligible_rna["rna_file_uuid"].dropna().astype(str).unique())
    version_rows, version_meta = client.file_versions(version_query_uuids)
    version_df = keep_ordered_columns(pd.DataFrame(version_rows), FILE_VERSION_COLUMNS)
    open_released = set(metadata_eligible_rna.query("access == 'open' and state == 'released'")["rna_file_uuid"].astype(str))
    if not version_df.empty:
        version_df["selected_file_is_released_open"] = version_df["queried_uuid"].astype(str).isin(open_released)
    write_parquet_atomic(inventory_dir / "file_versions.parquet", version_df, overwrite=args.force)
    write_json_atomic(settings_dir / "gdc_file_versions_query_manifest.json", {
        "created_at_utc": utc_now_iso(),
        "gdc_api_base_url": args.gdc_api_base,
        "queried_uuid_count": len(version_query_uuids),
        "query_scope": "metadata_eligible_rna_files_only",
        "query_manifest": version_meta,
        "response_schema_hash": sha256_obj(version_df.head(20).to_dict("records")),
    }, overwrite=args.force)
    versioned = version_ok_uuids(version_df, allow_low_confidence_fallback=False, require_latest=False)

    print_step("Querying GDC annotations only for metadata-eligible RNA entities")
    entity_ids = rna_annotation_entity_ids(metadata_eligible_rna)
    print(f"Collected {len(entity_ids)} RNA annotation entity IDs.", flush=True)
    ann = query_annotations(client, entity_ids, ANNOTATION_FIELDS, args.page_size, snapshot_dir)
    ann_df = keep_ordered_columns(pd.DataFrame([flatten_annotation_hit(h) for h in ann["hits"]]), ANNOTATION_COLUMNS)
    write_parquet_atomic(inventory_dir / "gdc_annotations.parquet", ann_df, overwrite=args.force)
    ann_audit, ann_excl = annotate_exclusions(ann_df, default_annotation_policy())
    write_csv_atomic(inventory_dir / "annotation_audit.csv", ann_audit, overwrite=args.force)
    write_csv_atomic(inventory_dir / "annotation_exclusion_log.csv", ann_excl, overwrite=args.force)
    excluded_entities = ann_excl.get("entity_id", pd.Series(dtype=str)).dropna().astype(str).tolist() if not ann_excl.empty else []

    print_step("Finalizing RNA files selected for download")
    selected_rna = select_pre_download_eligible_rna_files(
        metadata_eligible_rna,
        excluded_entities=excluded_entities,
        versioned_uuids=versioned,
        require_versioned=True,
    )
    if args.rna_per_case == "canonical":
        selected_rna = choose_canonical_rna_per_case(selected_rna)
    if selected_rna.empty:
        raise RuntimeError("No RNA files remained after annotation and file-version eligibility")
    write_parquet_atomic(inventory_dir / "selected_rna_files.parquet", selected_rna, overwrite=args.force)
    audit = build_rna_selection_audit(
        rna_df,
        excluded_entities=excluded_entities,
        versioned_uuids=versioned,
        selected_uuids=selected_rna["rna_file_uuid"].astype(str).tolist(),
    )
    write_csv_atomic(inventory_dir / "rna_selection_audit.csv", audit, overwrite=args.force)
    print(f"Selected {len(selected_rna)} pre-slide RNA candidate files across {selected_rna['case_id'].nunique()} cases. RNA TSV download is deferred by default.", flush=True)

    print_step("Querying slide metadata after RNA selection")
    if args.skip_slide_inventory:
        slide_df = keep_ordered_columns(pd.DataFrame(), SLIDE_INVENTORY_COLUMNS)
        slide_meta: Dict[str, Any] = {"mode": "skipped", "n_hits_after_dedup": 0}
    else:
        selected_case_uuids = selected_rna["case_uuid"].dropna().astype(str).unique().tolist() if args.restrict_slides_to_selected_rna_cases else None
        slide_df, slide_meta = query_slide_inventory(
            client,
            base_filter=filters["slide_files"],
            slide_fields=slide_fields,
            page_size=args.page_size,
            snapshot_dir=snapshot_dir,
            case_uuids=selected_case_uuids,
            chunk_size=args.slide_case_chunk_size,
        )
        if not slide_df.empty and slide_df["slide_file_uuid"].duplicated().any():
            raise RuntimeError("Duplicate slide file UUIDs in flattened inventory")
    write_parquet_atomic(inventory_dir / "candidate_slide_files.parquet", slide_df, overwrite=args.force)

    proj_df = project_inventory(rna_df, slide_df, selected_rna)
    write_parquet_atomic(inventory_dir / "project_inventory.parquet", proj_df, overwrite=args.force)

    if args.download_scope == "selected":
        download_df = selected_rna
    elif args.download_scope == "metadata_eligible":
        download_df = metadata_eligible_rna
    else:
        download_df = rna_df
    write_parquet_atomic(inventory_dir / "download_scope_rna_files.parquet", download_df, overwrite=args.force)

    download_meta: Dict[str, Any] = {
        "skipped": bool(args.skip_rna_download),
        "download_scope": args.download_scope,
        "official_reason": "RNA TSV download is deferred until the final benchmark cohort is known",
    }
    if not args.skip_rna_download:
        print_step("Downloading RNA files (--download_scope)")
        download_meta = download_rna_files(client, source_root, download_df, overwrite=args.force, limit=args.download_limit)
        download_meta["download_scope_argument"] = args.download_scope
        download_meta["download_limit"] = args.download_limit
        download_meta["legacy_source_release_download"] = True
    else:
        print_step("Skipping RNA file downloads; _12_build_rna_asset.py downloads only the benchmark-retained RNA files")
        empty_checksums = keep_ordered_columns(pd.DataFrame(), CHECKSUM_COLUMNS)
        write_tsv_atomic(source_root / "raw_cache" / "checksums.tsv", empty_checksums, overwrite=args.force)
        write_json_atomic(source_root / "raw_cache" / "download_log.json", {"downloads": [], "created_at_utc": utc_now_iso(), "skipped": True}, overwrite=args.force)
        manifest_dir = source_root / "raw_cache" / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        write_tsv_atomic(manifest_dir / "rna_download_manifest.tsv", empty_checksums, overwrite=args.force)

    print_step("Writing provenance and output hashes")
    provenance = manifest_header(Path(__file__).name, vars(args))
    provenance.update({
        "tcga_source_id": args.tcga_source_id,
        "gdc_api_base_url": args.gdc_api_base,
        "gdc_api_status_response_hash": sha256_obj(status),
        "source_config_hash": sha256_path(settings_dir / "source_config.yaml"),
        "alias_map_path": str(alias_dst),
        "alias_map_sha256": alias_sha,
        "alias_map_date": alias_date,
        "alias_map_source": args.alias_map_source,
        "alias_resolution_policy": source_config["hgnc_alias_map"]["alias_resolution_policy"],
        "duplicate_symbol_policy": source_config["hgnc_alias_map"]["duplicate_symbol_policy"],
        "efficient_source_release_policy": efficient_policy,
        "rna_query_meta": rna_meta,
        "slide_query_meta": slide_meta,
        "annotation_query_meta": ann["meta"],
        "file_versions_query_meta": version_meta,
        "rna_selection_counts": {
            "n_candidate_rna_files": int(len(rna_df)),
            "n_metadata_eligible_rna_files": int(len(metadata_eligible_rna)),
            "n_version_acceptable_rna_files": int(len(versioned)),
            "n_annotation_excluded_entities": int(len(set(excluded_entities))),
            "n_pre_slide_selected_rna_files": int(len(selected_rna)),
            "n_pre_slide_selected_rna_cases": int(selected_rna["case_id"].nunique()),
        },
        "rna_download_meta": download_meta,
        "output_hashes": collect_output_hashes(source_root),
    })
    write_json_atomic(settings_dir / "provenance.json", provenance, overwrite=args.force)
    print(json.dumps({
        "status": "ok",
        "source_root": str(source_root),
        "n_candidate_rna_files": len(rna_df),
        "n_pre_slide_selected_rna_files": len(selected_rna),
        "n_pre_slide_selected_rna_cases": int(selected_rna["case_id"].nunique()),
        "n_slide_metadata_files": len(slide_df),
    }, indent=2))
    return 0




if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Build a frozen TCGA/GDC source release for the TCGA Task B benchmark.")

    ap.add_argument("--tcga_source_id", type=str,
                    required=True,
                    help="Source release ID, e.g. gdc_tcga_YYYYMMDD_starcounts_dxslides_v1.")

    ap.add_argument("--hgnc_alias_map", type=Path,
                    required=True,
                    help="Filename of the HGNC alias TSV under refs/hgnc.")

    ap.add_argument("--alias_map_date", type=str,
                    default=None,
                    help="YYYYMMDD date of the alias map. Inferred from the filename if omitted.")

    ap.add_argument("--alias_map_source", type=str,
                    default="HGNC frozen TSV supplied by user",
                    help="Freeform text describing the source of the HGNC alias map.")

    ap.add_argument("--gdc_api_base", type=str,
                    default=DEFAULT_GDC_API_BASE,
                    help="Base URL for the GDC API. Default: https://api.gdc.cancer.gov")

    ap.add_argument("--page_size", type=int,
                    default=5000,
                    help="Page size for paginated GDC API queries.")

    ap.add_argument("--tcga_projects", type=str, nargs="*",
                    default=[],
                    help="TCGA project_ids to query, e.g. TCGA-BRCA TCGA-LUAD (default: all TCGA projects).")

    ap.add_argument("--gdc_prefilter_primary_samples", action="store_true",
                    help="Add a GDC-side sample_type prefilter (not recommended).")

    ap.add_argument("--rna_per_case", type=str, choices=["canonical", "all"],
                    default="canonical",
                    help="'canonical' keeps one deterministic eligible RNA file per case; 'all' preserves all eligible replicates.")

    ap.add_argument("--download_scope", type=str, choices=["selected", "metadata_eligible", "all_candidate"],
                    default="selected",
                    help="Debug only: scope of the RNA TSVs downloaded with --download_rna.")

    ap.add_argument("--skip_rna_download", dest="skip_rna_download", action="store_true",
                    default=True,
                    help="Build metadata only, without downloading RNA TSVs (the default).")

    ap.add_argument("--download_rna", dest="skip_rna_download", action="store_false",
                    help="Debug only: download RNA TSVs during source-release construction.")

    ap.add_argument("--download_limit", type=int,
                    default=None,
                    help="Debug only: download at most N RNA files from the chosen scope when --download_rna is used.")

    ap.add_argument("--skip_slide_inventory", action="store_true",
                    help="Write an empty slide inventory; useful for RNA-only debugging.")

    ap.add_argument("--restrict_slides_to_selected_rna_cases", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Query slide metadata only for cases with a selected RNA file (default true).")

    ap.add_argument("--slide_case_chunk_size", type=int,
                    default=500,
                    help="Number of cases per chunk when querying slide metadata.")

    ap.add_argument("--save_query_snapshots", action="store_true",
                    help="Save the raw GDC page JSONs (page hashes are always recorded).")

    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing source release at the same location if it exists.")

    args = ap.parse_args()
    raise SystemExit(main(args))
