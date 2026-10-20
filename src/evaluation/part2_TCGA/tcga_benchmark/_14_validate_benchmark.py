#!/usr/bin/env python3
"""
Validate a TCGA Task B benchmark before encoder evaluation: consistency of the final cohort, tiles, targets, folds,
signatures, source/RNA provenance, model-group geometry and slide metadata (exits with an error if a check fails).

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data / benchmarks / <benchmark_id> / reports /
    - benchmark_validation_report.json
    - benchmark_validation_report.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path

from _taskb_common import (
    FOLD_GROUPING_USED_VALUES,
    collect_output_hashes,
    deterministic_rna_asset_id,
    manifest_header,
    read_json,
    read_yaml,
    reroot_under_data_root,
    rna_eligibility_config,
    rna_parser_config,
    sha256_obj,
    sha256_path,
    shared_tile_h5_path,
    shared_tile_manifest_path,
    tissue_source_site,
    write_json_atomic,
    write_text_atomic,
)

# Required (minimum) columns of case_tile_index.parquet (extra columns allowed)
CASE_TILE_INDEX_COLUMNS = [
    "case_id", "case_uuid", "project_id", "slide_file_uuid", "slide_file_name", "tile_id", "tile_order_in_case",
    "top_left_x_l0", "top_left_y_l0", "center_x_l0", "center_y_l0", "read_window_px_l0_x", "read_window_px_l0_y",
    "source_mpp_x", "source_mpp_y", "tile_fov_um_effective", "n_tiles_eligible_before_cap", "n_tiles_used_after_cap",
    "n_slides_retained", "match_tier_best_case", "biospecimen_match_level", "official_primary_match", "tile_asset_id",
    "benchmark_id", "modelgroup_id",
]

FINAL_CASE_MANIFEST_COLUMNS = [
    "case_id", "case_uuid", "project_id", "rna_file_uuid", "rna_file_name", "n_slides_retained", "n_tiles_eligible_before_cap",
    "n_tiles_used_after_cap", "match_tier_best_case", "biospecimen_match_level_best_case", "official_primary_match",
    "final_case_eligible", "final_case_exclusion_reason", "tile_asset_id", "benchmark_id", "modelgroup_id",
]

CASE_TILE_DIAGNOSTICS_REQUIRED = [
    "case_id", "case_uuid", "project_id", "n_slides_retained", "n_tiles_eligible_before_cap", "n_tiles_used_after_cap",
    "dominant_slide_file_uuid_before_cap", "dominant_slide_tile_fraction_before_cap", "dominant_slide_file_uuid_after_cap",
    "dominant_slide_tile_fraction_after_cap", "slide_tile_entropy_before_cap", "slide_tile_entropy_after_cap",
    "slide_tile_effective_n_before_cap", "slide_tile_effective_n_after_cap", "slide_tile_gini_before_cap", "slide_tile_gini_after_cap",
    "single_slide_case", "multi_slide_case", "dominant_slide_fraction_gt_0_50", "dominant_slide_fraction_gt_0_75",
    "dominant_slide_fraction_gt_0_90", "cap_applied", "benchmark_id", "modelgroup_id", "tile_asset_id",
]

CASE_SLIDE_CONTRIBUTIONS_REQUIRED = [
    "case_id", "case_uuid", "project_id", "slide_file_uuid", "slide_file_name", "n_tiles_eligible_before_cap_slide",
    "n_tiles_used_after_cap_slide", "slide_tile_fraction_before_cap", "slide_tile_fraction_after_cap", "mean_tissue_fraction_slide",
    "sum_tissue_area_um2_slide", "source_mpp_x", "source_mpp_y", "biospecimen_match_level", "match_tier_best_case",
    "official_primary_match", "technical_prefetch_status", "benchmark_id", "modelgroup_id", "tile_asset_id",
]

FOLD_ASSIGNMENT_COLUMNS = ["case_id", "case_uuid", "project_id", "fold_id", "fold_seed", "benchmark_id"]
TARGET_META_COLUMNS = ["case_id", "case_uuid", "project_id"]

SIGNATURE_ELIGIBILITY_REQUIRED = [
    "project_id", "signature_name", "eligible", "eligibility_reason", "official_project", "domain_slice", "support_category",
    "primary_score_cell", "n_final_cases", "n_cases_with_target", "n_signature_genes", "n_genes_available_project",
    "n_genes_used", "coverage_fraction", "target_variance", "target_std", "target_min", "target_max",
    "fold_level_ridge_feasible", "fold_feasibility_reason", "score_method", "denominator_rule", "signature_family",
    "signature_source_name", "signature_source_version", "signature_set_id", "signature_set_hash", "target_table_hash",
    "final_case_manifest_hash", "fold_assignment_hash", "rna_asset_id", "rna_asset_hash", "alias_map_hash",
    "benchmark_id", "modelgroup_id", "fold_seed", "n_folds",
]

# Required (minimum) columns of slide_prefetch_manifest.parquet (extra columns allowed)
SLIDE_PREFETCH_COLUMNS = [
    "slide_file_uuid", "case_id", "case_uuid", "project_id", "slide_submitter_id", "biospecimen_match_level",
    "match_tier_best_case", "official_primary_match", "file_name", "slide_file_name", "file_size", "md5sum",
    "requested_range", "http_status", "accept_ranges_header", "content_range_header", "content_length_header", "etag",
    "last_modified", "bytes_received", "partial_sha256", "range_supported", "server_response_headers_hash",
    "request_timestamp", "retry_count", "failure_reason", "partial_parse_success", "partial_parse_engine", "partial_parse_error",
    "prefetch_openslide_readable", "prefetch_tiff_readable", "prefetch_mpp_x", "prefetch_mpp_y", "prefetch_objective_power",
    "prefetch_slide_width_l0", "prefetch_slide_height_l0", "prefetch_vendor", "prefetch_openslide_properties",
    "prefetch_image_description", "mpp_source", "mpp_confidence", "prefetch_mpp_valid_range", "prefetch_mpp_anisotropy_ok",
    "technical_prefetch_status", "prefetch_exclusion_reason", "eligible_for_full_download", "benchmark_id", "modelgroup_id",
]

OFFICIAL_MATCH_LEVELS = {"exact_same_portion", "same_sample", "same_case_primary_tumor", "same_case_primary_tumor_unknown_sublevel"}

# download_status values of _8 for slides present in the store (downloaded or already present)
SLIDE_BYTES_PRESENT_STATUSES = {
    "downloaded_md5_ok", "downloaded_no_expected_md5", "skipped_existing_md5_ok", "skipped_existing_size_ok",
}

# Cases with fewer tiles get a warning (degenerate std in the concat(mean, std) case feature)
LOW_TILE_WARN_THRESHOLD = 4


class Validator:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.checks: List[Dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: str = "", *, warning: bool = False) -> None:
        rec = {"name": name, "status": "passed" if ok else ("warning" if warning else "failed"), "detail": detail}
        self.checks.append(rec)
        if not ok:
            (self.warnings if warning else self.errors).append(f"{name}: {detail}")


def _load_mapping(path: Path) -> Dict[str, Any]:
    obj = read_json(path) if path.suffix.lower() == ".json" else read_yaml(path)
    if not isinstance(obj, Mapping):
        raise RuntimeError(f"Expected mapping at {path}")
    return dict(obj)


def _truthy(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    return str(x).strip().lower() in {"true", "t", "1", "yes", "y"}


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


# TCGA tissue-source-site code (shared with _13_build_targets_and_folds.py)
_tissue_source_site = tissue_source_site


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _require_exact_columns(v: Validator, df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    ok = list(df.columns) == list(columns)
    if not ok:
        missing = [c for c in columns if c not in df.columns]
        extra = [c for c in df.columns if c not in columns]
        wrong_order = not missing and not extra
        v.check(f"{name} schema exact", False, f"missing={missing}, extra={extra}, wrong_order={wrong_order}")
    else:
        v.check(f"{name} schema exact", True)


def _require_columns(v: Validator, df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    v.check(f"{name} required columns", not missing, f"missing={missing}" if missing else "")


def _signature_names(signature_set: Mapping[str, Any]) -> List[str]:
    sigs = signature_set.get("signatures")
    if not isinstance(sigs, list):
        raise RuntimeError("signature_set.signatures missing")
    return [str(s["signature_name"]) for s in sigs if isinstance(s, Mapping) and s.get("signature_name")]


def _domain_map(settings_dir: Path) -> Dict[str, str]:
    obj = read_json(settings_dir / "project_domain_map.json")
    labels = obj.get("project_domain_map") or obj.get("projects") or {}
    return {str(k): str(v) for k, v in labels.items()} if isinstance(labels, Mapping) else {}


def _official_projects(project_summary: Mapping[str, Any]) -> List[str]:
    rows = project_summary.get("project_rows") or []
    return sorted(str(r.get("project_id")) for r in rows if isinstance(r, Mapping) and bool(r.get("official_project")))


def _validate_hash_field(v: Validator, field_name: str, observed: str, expected: Optional[str]) -> None:
    v.check(field_name, bool(expected) and observed == expected, f"observed={observed}, expected={expected}")


def _safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _manifest_is_frozen(obj: Mapping[str, Any]) -> bool:
    return obj.get("frozen_final") is True or obj.get("status") == "frozen_final"


def _write_markdown(path: Path, report: Mapping[str, Any], *, overwrite: bool) -> None:
    lines = [
        "# Task B benchmark validation report",
        "",
        f"- status: `{report['status']}`",
        f"- benchmark_id: `{report['benchmark_id']}`",
        f"- errors: `{len(report.get('errors', []))}`",
        f"- warnings: `{len(report.get('warnings', []))}`",
        "",
        "## Failed checks",
        "",
    ]
    failed = [c for c in report["checks"] if c["status"] == "failed"]
    if not failed:
        lines.append("None.")
    else:
        for c in failed:
            lines.append(f"- **{c['name']}**: {c.get('detail', '')}")
    lines.extend(["", "## Warnings", ""])
    warns = [c for c in report["checks"] if c["status"] == "warning"]
    if not warns:
        lines.append("None.")
    else:
        for c in warns:
            lines.append(f"- **{c['name']}**: {c.get('detail', '')}")
    lines.extend(["", "## Summary", "", "```json", json.dumps(report.get("summary", {}), indent=2, sort_keys=True), "```", ""])
    write_text_atomic(path, "\n".join(lines), overwrite=overwrite)


def validate_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    v = Validator()
    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    settings_dir = benchmark_root / "settings"
    shared_dir = benchmark_root / "shared"
    cohort_dir = shared_dir / "cohort"
    tiles_dir = shared_dir / "tiles"
    rna_dir = shared_dir / "rna"
    folds_dir = shared_dir / "folds"
    reports_dir = benchmark_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    paths_required = [
        settings_dir / "benchmark_config.yaml",
        settings_dir / "benchmark_asset_links.json",
        settings_dir / "project_domain_map.json",
        settings_dir / "eval_signatures" / "signatures.gmt",
        settings_dir / "eval_signatures" / "sources.json",
        settings_dir / "signature_set_tcga_case_v1.json",
        settings_dir / "signature_support_training.json",
        settings_dir / "annotation_policy.yaml",
        settings_dir / "encoder_registry.json",
        cohort_dir / "rna_manifest.csv",
        cohort_dir / "slide_manifest.csv",
        cohort_dir / "final_case_manifest.csv",
        cohort_dir / "project_summary.json",
        tiles_dir / "case_tile_index.parquet",
        tiles_dir / "case_tile_index_manifest.json",
        tiles_dir / "case_tile_diagnostics.parquet",
        tiles_dir / "case_slide_contributions.parquet",
        rna_dir / "signature_eligibility.csv",
        rna_dir / "target_and_folds_manifest.json",
        folds_dir / "fold_assignments.csv",
    ]
    missing_initial = [str(p) for p in paths_required if not p.exists()]
    v.check("all required benchmark files exist", not missing_initial, f"missing={missing_initial}")
    if missing_initial:
        return _assemble_report(args, v, {}, benchmark_root, reports_dir)

    cfg = _load_mapping(settings_dir / "benchmark_config.yaml")
    links = _load_mapping(settings_dir / "benchmark_asset_links.json")
    modelgroup_id = str(cfg.get("modelgroup_id") or links.get("modelgroup_id"))
    rna_asset_id = str(cfg.get("rna_asset_id") or links.get("rna_asset_id"))
    tcga_source_id = str(cfg.get("tcga_source_id") or links.get("tcga_source_id"))
    min_tiles = int(cfg.get("min_tiles_per_case", 1))
    fold_count = int(cfg.get("fold_count", 5))
    fold_seed = int(cfg.get("fold_seed", 20260518))

    # Asset paths rebuilt under the current root (not read from benchmark_asset_links.json)
    source_root = data_root / "source_releases" / tcga_source_id
    rna_asset_root = data_root / "processed_assets" / "rna" / rna_asset_id
    tile_asset_id = str(links.get("tile_asset_id") or "")
    tile_asset_root = data_root / "processed_assets" / "tiles" / tile_asset_id
    modelgroup_path = settings_dir / "modelgroups" / f"{modelgroup_id}.yaml"
    source_prov = source_root / "settings" / "provenance.json"
    rna_asset_manifest_path = rna_asset_root / "rna_asset_manifest.json"
    tile_asset_manifest_path = tile_asset_root / "tile_asset_manifest.json"
    histoqc_asset_id = str(links.get("histoqc_asset_id") or "")
    histoqc_asset_root = (data_root / "processed_assets" / "histoqc" / histoqc_asset_id) if histoqc_asset_id else None
    histoqc_asset_manifest_path = (histoqc_asset_root / "histoqc_asset_manifest.json") if histoqc_asset_root is not None else None
    selected_cache = data_root / "selected_slide_cache" / args.benchmark_id
    prefetch_path = cohort_dir / "slide_prefetch_manifest.parquet"
    download_manifest_path = selected_cache / "manifests" / "selected_slide_download_manifest.tsv"

    required_paths = [
        ("source release provenance", source_prov),
        ("RNA asset manifest", rna_asset_manifest_path),
        ("tile asset manifest", tile_asset_manifest_path),
        ("modelgroup yaml", modelgroup_path),
        ("slide_prefetch_manifest", prefetch_path),
        ("selected_slide_download_manifest", download_manifest_path),
    ]
    if histoqc_asset_manifest_path is not None:
        required_paths.append(("HistoQC asset manifest", histoqc_asset_manifest_path))
    for name, path in required_paths:
        v.check(f"{name} exists", path.exists(), str(path))

    alias_hash = str(links.get("alias_map_sha256") or links.get("alias_map_hash") or "")
    # Alias map file name under the current settings_dir
    alias_path = (settings_dir / Path(str(links.get("alias_map_path"))).name) if links.get("alias_map_path") else None
    if alias_path is not None:
        v.check("HGNC alias map exists", alias_path.exists(), str(alias_path))
        if alias_path.exists():
            _validate_hash_field(v, "HGNC alias map hash matches links", sha256_path(alias_path), alias_hash)
    else:
        v.check("HGNC alias map path present", False, "benchmark_asset_links missing alias_map_path")

    rna_manifest_csv = _read_csv(cohort_dir / "rna_manifest.csv")
    slide_manifest = _read_csv(cohort_dir / "slide_manifest.csv")
    final_case = _read_csv(cohort_dir / "final_case_manifest.csv")
    case_tile = pd.read_parquet(tiles_dir / "case_tile_index.parquet")
    case_diag = pd.read_parquet(tiles_dir / "case_tile_diagnostics.parquet")
    case_contrib = pd.read_parquet(tiles_dir / "case_slide_contributions.parquet")
    folds = _read_csv(folds_dir / "fold_assignments.csv")
    sig_elig = _read_csv(rna_dir / "signature_eligibility.csv")
    project_summary = read_json(cohort_dir / "project_summary.json")
    signature_set = read_json(settings_dir / "signature_set_tcga_case_v1.json")
    support = read_json(settings_dir / "signature_support_training.json")
    target_manifest = read_json(rna_dir / "target_and_folds_manifest.json")
    case_tile_manifest = read_json(tiles_dir / "case_tile_index_manifest.json")
    modelgroup = _load_mapping(modelgroup_path) if modelgroup_path.exists() else {}

    _require_exact_columns(v, final_case, FINAL_CASE_MANIFEST_COLUMNS, "final_case_manifest.csv")
    # Required columns only (extra columns allowed)
    _require_columns(v, case_tile, CASE_TILE_INDEX_COLUMNS, "case_tile_index.parquet")
    _require_columns(v, case_diag, CASE_TILE_DIAGNOSTICS_REQUIRED, "case_tile_diagnostics.parquet")
    _require_columns(v, case_contrib, CASE_SLIDE_CONTRIBUTIONS_REQUIRED, "case_slide_contributions.parquet")
    _require_exact_columns(v, folds, FOLD_ASSIGNMENT_COLUMNS, "fold_assignments.csv")
    _require_columns(v, sig_elig, SIGNATURE_ELIGIBILITY_REQUIRED, "signature_eligibility.csv")

    final_eligible = final_case.loc[final_case["final_case_eligible"].map(_truthy)].copy()
    v.check("final cohort nonempty", not final_eligible.empty)
    v.check("no duplicate final case rows", not final_case["case_id"].duplicated().any())
    v.check("no duplicate final eligible case rows", not final_eligible["case_id"].duplicated().any())
    v.check("one selected RNA per final case", not rna_manifest_csv["case_id"].duplicated().any() and set(final_eligible["case_id"]).issubset(set(rna_manifest_csv["case_id"])))

    # Final cohort tile/match checks.
    final_cases = set(final_eligible["case_id"].astype(str))
    case_tile_cases = set(case_tile["case_id"].astype(str))
    v.check("case_tile_index covers every final case", final_cases.issubset(case_tile_cases), f"missing={sorted(final_cases - case_tile_cases)[:20]}")
    grouped_tile = case_tile.groupby("case_id", sort=False).agg(
        n_tiles=("tile_id", "size"), n_slides=("slide_file_uuid", "nunique"), official_primary_match=("official_primary_match", lambda s: all(_truthy(x) for x in s)),
    )
    bad_tile = [c for c in final_cases if c not in grouped_tile.index or int(grouped_tile.loc[c, "n_tiles"]) < min_tiles]
    v.check("all final cases have >= min_tiles_per_case", not bad_tile, f"bad_cases={bad_tile[:20]}, min_tiles={min_tiles}")
    low_tile_cases = sorted(
        c for c in final_cases
        if c in grouped_tile.index and int(grouped_tile.loc[c, "n_tiles"]) < LOW_TILE_WARN_THRESHOLD
    )
    v.check(
        "few-tile cases produce degenerate std-pooled features",
        not low_tile_cases,
        f"n_cases_with_lt_{LOW_TILE_WARN_THRESHOLD}_tiles={len(low_tile_cases)}, examples={low_tile_cases[:20]}",
        warning=True,
    )
    bad_slide = [c for c in final_cases if c not in grouped_tile.index or int(grouped_tile.loc[c, "n_slides"]) < 1]
    v.check("at least one retained slide per final case", not bad_slide, f"bad_cases={bad_slide[:20]}")
    v.check("official_primary_match true for all final cases", final_eligible["official_primary_match"].map(_truthy).all())
    v.check("biospecimen_match_level present for all final cases", final_eligible["biospecimen_match_level_best_case"].map(_nonempty).all())
    v.check("official-primary biospecimen policy respected", final_eligible["biospecimen_match_level_best_case"].astype(str).isin(OFFICIAL_MATCH_LEVELS).all())

    # Manifest counts must match the actual selected tile table after all caps.
    count_check = final_eligible.merge(
        grouped_tile[["n_tiles", "n_slides"]].reset_index(),
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    bad_manifest_counts = count_check.loc[
        (pd.to_numeric(count_check["n_tiles_used_after_cap"], errors="coerce").astype("Int64") != pd.to_numeric(count_check["n_tiles"], errors="coerce").astype("Int64"))
        | (pd.to_numeric(count_check["n_slides_retained"], errors="coerce").astype("Int64") != pd.to_numeric(count_check["n_slides"], errors="coerce").astype("Int64")),
        ["case_id", "n_tiles_used_after_cap", "n_tiles", "n_slides_retained", "n_slides"],
    ].head(20).to_dict("records")
    v.check("final_case_manifest tile/slide counts match case_tile_index", not bad_manifest_counts, f"bad={bad_manifest_counts}")

    row_counts = case_tile.merge(
        grouped_tile[["n_tiles", "n_slides"]].reset_index(),
        on="case_id",
        how="left",
        validate="many_to_one",
    )
    bad_row_counts = row_counts.loc[
        (pd.to_numeric(row_counts["n_tiles_used_after_cap"], errors="coerce").astype("Int64") != pd.to_numeric(row_counts["n_tiles"], errors="coerce").astype("Int64"))
        | (pd.to_numeric(row_counts["n_slides_retained"], errors="coerce").astype("Int64") != pd.to_numeric(row_counts["n_slides"], errors="coerce").astype("Int64")),
        ["case_id", "n_tiles_used_after_cap", "n_tiles", "n_slides_retained", "n_slides"],
    ].drop_duplicates().head(20).to_dict("records")
    v.check("case_tile_index repeated counts match selected tiles", not bad_row_counts, f"bad={bad_row_counts}")

    # Folds.
    fold_cases = set(folds["case_id"].astype(str))
    v.check("all final cases have fold assignments", final_cases == fold_cases, f"missing={sorted(final_cases - fold_cases)[:20]}, extra={sorted(fold_cases - final_cases)[:20]}")
    v.check("fold assignment case rows unique", not folds["case_id"].duplicated().any())
    if not folds.empty:
        fold_ids = sorted(pd.to_numeric(folds["fold_id"], errors="coerce").dropna().astype(int).unique().tolist())
        v.check("fold ids match configured fold_count", fold_ids == list(range(fold_count)), f"observed={fold_ids}, expected={list(range(fold_count))}")
        v.check("fold_seed matches benchmark_config", set(pd.to_numeric(folds["fold_seed"], errors="coerce").dropna().astype(int).tolist()) == {fold_seed})

    # TSS-grouped folds: each tissue-source-site must stay within a single fold.
    fold_protocol = target_manifest.get("fold_protocol") or {}
    per_project_grouping = fold_protocol.get("per_project_fold_grouping") or {}
    # Every per-project fold_grouping_used must be a known value
    if per_project_grouping:
        bad_grouping = sorted({
            str((v or {}).get("fold_grouping_used"))
            for v in per_project_grouping.values()
            if str((v or {}).get("fold_grouping_used")) not in set(FOLD_GROUPING_USED_VALUES)
        })
        v.check(
            "per-project fold_grouping_used values are all recognized",
            not bad_grouping,
            f"unrecognized fold_grouping_used={bad_grouping}; expected one of {list(FOLD_GROUPING_USED_VALUES)}",
        )
    if str(fold_protocol.get("fold_grouping") or "") == "tissue_source_site" and not folds.empty:
        folds_tss = folds.copy()
        folds_tss["__tss__"] = folds_tss["case_id"].astype(str).map(_tissue_source_site)
        tss_split: List[str] = []
        for project_id, g in folds_tss.groupby("project_id", sort=True):
            used = str((per_project_grouping.get(str(project_id)) or {}).get("fold_grouping_used"))
            if used != "tissue_source_site":
                continue
            spanning = g.groupby("__tss__")["fold_id"].nunique()
            tss_split.extend(f"{project_id}:{t}" for t in spanning.index[spanning > 1].tolist())
        v.check(
            "tissue-source-site grouped folds keep each site within one fold",
            not tss_split,
            f"sites_spanning_multiple_folds={tss_split[:20]}",
            warning=True,
        )

    # RNA / source / signature artifacts.
    source_ok = source_prov.exists()
    rna_ok = rna_asset_manifest_path.exists()
    tile_ok = tile_asset_manifest_path.exists()
    source_manifest = read_json(source_prov) if source_ok else {}
    rna_asset_manifest = read_json(rna_asset_manifest_path) if rna_ok else {}
    tile_asset_manifest = read_json(tile_asset_manifest_path) if tile_ok else {}
    v.check("source release provenance present", bool(source_manifest))
    v.check("RNA asset manifest present", bool(rna_asset_manifest))
    v.check("tile asset manifest frozen", _manifest_is_frozen(tile_asset_manifest), f"frozen_final/status={tile_asset_manifest.get('frozen_final')}/{tile_asset_manifest.get('status')}")

    # HistoQC checks
    v.check(
        "HistoQC asset registered in benchmark_asset_links.json",
        bool(histoqc_asset_id),
        "links.histoqc_asset_id is empty — run _9_run_histoqc.py",
    )
    histoqc_asset_manifest: Mapping[str, Any] = {}
    if histoqc_asset_id and histoqc_asset_manifest_path is not None and histoqc_asset_manifest_path.exists():
        histoqc_asset_manifest = read_json(histoqc_asset_manifest_path) or {}
        v.check(
            "HistoQC asset manifest frozen",
            _manifest_is_frozen(histoqc_asset_manifest),
            f"frozen_final/status={histoqc_asset_manifest.get('frozen_final')}/{histoqc_asset_manifest.get('status')}",
        )
        v.check(
            "HistoQC asset benchmark_id matches",
            str(histoqc_asset_manifest.get("benchmark_id") or "") == args.benchmark_id,
            f"histoqc={histoqc_asset_manifest.get('benchmark_id')!r}, benchmark={args.benchmark_id!r}",
        )
        v.check(
            "tile asset histoqc_asset_id matches on-disk HistoQC",
            (tile_asset_manifest.get("deterministic_tile_asset_id_components") or {}).get("histoqc_asset_id") == histoqc_asset_id,
            f"tile_asset_components={(tile_asset_manifest.get('deterministic_tile_asset_id_components') or {}).get('histoqc_asset_id')!r}, "
            f"links={histoqc_asset_id!r}",
        )
        v.check(
            "tile asset input_hashes records histoqc_config_hash",
            bool((tile_asset_manifest.get("input_hashes") or {}).get("histoqc_config_hash")),
            "missing tile_asset_manifest.input_hashes.histoqc_config_hash",
        )
        # histoqc_config_hash of the tile asset must match that of the HistoQC asset
        tile_recorded = (tile_asset_manifest.get("input_hashes") or {}).get("histoqc_config_hash")
        histoqc_recorded = (histoqc_asset_manifest.get("input_hashes") or {}).get("histoqc_config")
        v.check(
            "histoqc_config_hash consistent between tile asset and HistoQC asset",
            tile_recorded == histoqc_recorded and bool(tile_recorded),
            f"tile={tile_recorded!r}, histoqc={histoqc_recorded!r}",
        )
        # Every referenced shared HistoQC mask must exist and match its recorded sha256
        ref_masks = histoqc_asset_manifest.get("referenced_shared_masks") or {}
        if isinstance(ref_masks, Mapping) and ref_masks:
            missing_masks: List[str] = []
            mismatched_masks: List[str] = []
            for u, info in tqdm(ref_masks.items(), total=len(ref_masks), desc="Verifying HistoQC masks", unit="slide"):
                if not isinstance(info, Mapping):
                    continue
                # Recorded mask path re-rooted under the current data_root
                p = reroot_under_data_root(data_root, info.get("path"))
                sha = info.get("sha256")
                if not p or not Path(str(p)).exists():
                    missing_masks.append(str(u))
                    continue
                if sha and sha256_path(Path(str(p))) != sha:
                    mismatched_masks.append(str(u))
            v.check(
                "all referenced shared HistoQC masks exist",
                not missing_masks,
                f"missing for {missing_masks[:10]} (of {len(ref_masks)})",
            )
            v.check(
                "referenced shared HistoQC masks match recorded sha256",
                not mismatched_masks,
                f"sha mismatch for {mismatched_masks[:10]}",
            )
    # Shared tile store: one set recipe (and one codec recipe) per benchmark; every referenced blob must exist
    # and match its recorded sha256
    ref_tiles = tile_asset_manifest.get("referenced_shared_tiles") or {}
    if isinstance(ref_tiles, Mapping) and ref_tiles:
        set_ids = {str(i.get("set_recipe_id")) for i in ref_tiles.values() if isinstance(i, Mapping) and i.get("set_recipe_id")}
        v.check("all referenced tile sets share one set_recipe_id", len(set_ids) <= 1, f"set_ids={sorted(set_ids)}")
        v.check(
            "tile asset tile_set_recipe_id matches referenced sets",
            not set_ids or str(tile_asset_manifest.get("tile_set_recipe_id") or "") in set_ids,
            f"manifest={tile_asset_manifest.get('tile_set_recipe_id')!r}, referenced={sorted(set_ids)}",
        )
        missing_sets: List[str] = []
        mismatched_sets: List[str] = []
        for u, info in tqdm(ref_tiles.items(), total=len(ref_tiles), desc="Verifying tile manifests", unit="slide"):
            if not isinstance(info, Mapping):
                continue
            # manifest.parquet path rebuilt from its ids under the current root
            set_id = str(info.get("set_recipe_id") or "")
            p = shared_tile_manifest_path(data_root, str(u), set_id) if set_id else None
            sha = info.get("manifest_sha256")
            if not p or not Path(str(p)).exists():
                missing_sets.append(str(u))
            elif sha and sha256_path(Path(str(p))) != sha:
                mismatched_sets.append(str(u))
        v.check("all referenced shared tile manifests exist", not missing_sets, f"missing for {missing_sets[:10]} (of {len(ref_tiles)})")
        v.check("referenced shared tile manifests match recorded sha256", not mismatched_sets, f"sha mismatch for {mismatched_sets[:10]}")

    cache_mf_path = tile_asset_root / "tile_rgb_cache" / "tile_rgb_cache_manifest.json"  # under the current root
    if cache_mf_path.exists():
        cache_mf = read_json(cache_mf_path) or {}
        ref_pix = cache_mf.get("referenced_shared_tiles") or {}
        if isinstance(ref_pix, Mapping) and ref_pix:
            codec_ids = {str(i.get("codec_recipe_id")) for i in ref_pix.values() if isinstance(i, Mapping) and i.get("codec_recipe_id")}
            v.check("all referenced tile pixels share one codec_recipe_id", len(codec_ids) <= 1, f"codec_ids={sorted(codec_ids)}")
            missing_pix: List[str] = []
            mismatched_pix: List[str] = []
            for u, info in tqdm(ref_pix.items(), total=len(ref_pix), desc="Verifying tile pixels", unit="slide"):
                if not isinstance(info, Mapping):
                    continue
                # tiles.h5 path rebuilt from its ids under the current root
                set_id = str(info.get("set_recipe_id") or "")
                codec_id = str(info.get("codec_recipe_id") or "")
                p = shared_tile_h5_path(data_root, str(u), set_id, codec_id) if (set_id and codec_id) else None
                sha = info.get("h5_sha256")
                if not p or not Path(str(p)).exists():
                    missing_pix.append(str(u))
                elif sha and sha256_path(Path(str(p))) != sha:
                    mismatched_pix.append(str(u))
            v.check("all referenced shared tile pixel blobs exist", not missing_pix, f"missing for {missing_pix[:10]} (of {len(ref_pix)})")
            v.check("referenced shared tile pixel blobs match recorded sha256", not mismatched_pix, f"sha mismatch for {mismatched_pix[:10]}")

    v.check("signature set frozen", _manifest_is_frozen(signature_set), f"status={signature_set.get('status')}")
    v.check("signature support frozen", _manifest_is_frozen(support), f"status={support.get('status')}")
    v.check("final cohort/case tile index frozen", _manifest_is_frozen(case_tile_manifest), f"status={case_tile_manifest.get('status')}")
    v.check("targets/folds frozen", _manifest_is_frozen(target_manifest), f"status={target_manifest.get('status')}")

    if rna_asset_manifest:
        v.check("RNA duplicate-symbol rule is pre-log sum then log10", rna_asset_manifest.get("duplicate_symbol_policy") == rna_parser_config()["duplicate_resolved_symbol_rule"], str(rna_asset_manifest.get("duplicate_symbol_policy")))
        v.check("RNA asset source release matches benchmark", str(rna_asset_manifest.get("tcga_source_id")) == tcga_source_id, f"rna={rna_asset_manifest.get('tcga_source_id')}, benchmark={tcga_source_id}")
        v.check("RNA asset alias matches benchmark alias", rna_asset_manifest.get("alias_map_sha256") == alias_hash, f"rna={rna_asset_manifest.get('alias_map_sha256')}, benchmark={alias_hash}")
        # Recompute the deterministic RNA asset id (an explicit, configured rna_asset_id is allowed)
        if "derivation_components" in rna_asset_manifest and source_manifest:
            derived = deterministic_rna_asset_id(str(rna_asset_manifest.get("tcga_source_id")), rna_asset_manifest.get("derivation_components") or {})
            recorded_ids = {
                str(x) for x in [
                    rna_asset_manifest.get("deterministic_rna_asset_id"),
                    rna_asset_manifest.get("deterministic_rna_asset_id_pre_download"),
                    rna_asset_manifest.get("deterministic_rna_asset_id_with_download_set"),
                ] if x
            }
            configured_id = str(rna_asset_manifest.get("rna_asset_id"))
            id_is_deterministic = configured_id == derived or configured_id in recorded_ids or derived in recorded_ids
            explicit_recorded = (
                rna_asset_manifest.get("rna_asset_id_matches_deterministic") is False
                or rna_asset_manifest.get("rna_asset_id_matches_deterministic_with_download_set") is False
            )
            v.check(
                "RNA asset deterministic derivation recorded",
                id_is_deterministic or explicit_recorded,
                f"derived={derived}, configured={configured_id}, recorded={sorted(recorded_ids)}",
                warning=bool(explicit_recorded and not id_is_deterministic),
            )
    v.check("signature set and RNA asset use same alias map", signature_set.get("alias_map_sha256") == (rna_asset_manifest.get("alias_map_sha256") if rna_asset_manifest else alias_hash))
    v.check("signature denominator rule is available_resolved_abs_weight_sum", signature_set.get("denominator_rule") == "available_resolved_abs_weight_sum")
    panel_gmt_path = settings_dir / "eval_signatures" / "signatures.gmt"
    panel_sources_path = settings_dir / "eval_signatures" / "sources.json"
    v.check(
        "eval signatures GMT frozen and hashed",
        panel_gmt_path.exists() and sha256_path(panel_gmt_path) == signature_set.get("signatures_gmt_sha256"),
        str(panel_gmt_path),
    )
    v.check(
        "eval signatures sources.json frozen and hashed",
        panel_sources_path.exists() and sha256_path(panel_sources_path) == signature_set.get("signatures_sources_sha256"),
        str(panel_sources_path),
    )
    v.check("annotation policy frozen and hashed", (settings_dir / "annotation_policy.yaml").exists() and sha256_path(settings_dir / "annotation_policy.yaml") is not None)
    encoder_registry_path = settings_dir / "encoder_registry.json"
    try:
        encoder_registry = read_json(encoder_registry_path)
    except Exception as e:
        encoder_registry = None
        v.check("encoder registry complete", False, f"unreadable: {e}")
    else:
        v.check("encoder registry complete", isinstance(encoder_registry, (dict, list)) and bool(encoder_registry))

    # Target tables.
    sig_names = _signature_names(signature_set)
    official_projects = _official_projects(project_summary)
    target_dir = rna_dir / "target_tables"
    v.check("target table directory exists", target_dir.exists(), str(target_dir))
    table_hashes: Dict[str, str] = {}
    for project_id in official_projects:
        tpath = target_dir / f"{project_id}.parquet"
        mpath = target_dir / f"{project_id}.manifest.json"
        if not tpath.exists():
            v.check(f"target table exists {project_id}", False, str(tpath))
            continue
        t = pd.read_parquet(tpath)
        expected_cols = TARGET_META_COLUMNS + sig_names
        _require_exact_columns(v, t, expected_cols, f"target table {project_id}")
        target_cases = set(t["case_id"].astype(str))
        expected_cases = set(final_eligible.loc[final_eligible["project_id"].astype(str).eq(project_id), "case_id"].astype(str))
        v.check(f"target table matches final cases {project_id}", target_cases == expected_cases, f"missing={sorted(expected_cases-target_cases)[:10]}, extra={sorted(target_cases-expected_cases)[:10]}")
        for sig in sig_names:
            vals = pd.to_numeric(t[sig], errors="coerce").to_numpy(dtype=float)
            v.check(f"target finite/nonconstant {project_id}/{sig}", np.isfinite(vals).all() and float(np.var(vals, ddof=0)) > float(args.target_variance_min), f"var={float(np.nanvar(vals)) if len(vals) else None}", warning=True)
        table_hashes[project_id] = sha256_path(tpath)
        v.check(f"target table manifest exists {project_id}", mpath.exists(), str(mpath))
        if mpath.exists():
            tm = read_json(mpath)
            v.check(f"target table manifest frozen {project_id}", _manifest_is_frozen(tm))
            _validate_hash_field(v, f"target table hash manifest {project_id}", table_hashes[project_id], tm.get("target_table_hash"))
            v.check(f"target z-scoring manifest frozen-label construction {project_id}", (tm.get("target_construction") or {}).get("recomputed_inside_cv") is False and (tm.get("target_construction") or {}).get("fold_local_target_construction") is False)
    v.check("target tables use frozen signature set only", set(sig_elig["signature_name"].astype(str)).issubset(set(sig_names)))
    if "target_z_scoring_manifest" in target_manifest:
        tz = target_manifest["target_z_scoring_manifest"]
        v.check("no script recomputed target z-scores inside CV", tz.get("recomputed_inside_cv") is False and tz.get("fold_local_target_construction") is False)
    v.check("fold-level ridge feasibility checked", "fold_level_ridge_feasible" in sig_elig.columns and sig_elig["fold_level_ridge_feasible"].map(lambda x: str(x).lower() in {"true", "false", "1", "0"}).all())
    primary_count = int(sig_elig.get("primary_score_cell", pd.Series(dtype=str)).map(_truthy).sum()) if "primary_score_cell" in sig_elig else 0
    eligible_count = int(sig_elig.get("eligible", pd.Series(dtype=str)).map(_truthy).sum()) if "eligible" in sig_elig else 0
    v.check("primary eligible project × signature count > 0", primary_count > 0, f"primary_count={primary_count}, eligible_count={eligible_count}")

    # Tile physical scale / MPP.
    if modelgroup:
        expected_fov = float(modelgroup.get("tile_fov_um_effective", np.nan))
        if "tile_fov_um_effective" in case_tile.columns:
            observed = pd.to_numeric(case_tile["tile_fov_um_effective"], errors="coerce")
            v.check("tile FOV values match modelgroup", np.isfinite(observed).all() and np.allclose(observed.to_numpy(dtype=float), expected_fov, rtol=0, atol=1e-6), f"expected={expected_fov}")
        v.check("tile physical scale frozen", modelgroup.get("tiling_version") is not None and modelgroup.get("tile_stride_um") is not None)
    for col in ["source_mpp_x", "source_mpp_y"]:
        vals = pd.to_numeric(case_tile[col], errors="coerce")
        v.check(f"MPP values valid {col}", np.isfinite(vals).all() and (vals > 0).all() and vals.between(0.20, 1.20).all())

    # Slide policy / metadata.
    for col in ["unknown_stain", "unknown_ffpe"]:
        v.check(f"{col} present", col in slide_manifest.columns)
    for col in ["slide_file_uuid", "case_id", "case_uuid", "project_id", "biospecimen_match_level", "official_primary_match"]:
        v.check(f"slide_manifest has {col}", col in slide_manifest.columns)
    if {"slide_file_uuid", "official_primary_match", "biospecimen_match_level"}.issubset(slide_manifest.columns):
        retained_slides = set(case_tile["slide_file_uuid"].astype(str))
        sm = slide_manifest.loc[slide_manifest["slide_file_uuid"].astype(str).isin(retained_slides)].copy()
        v.check("diagnostic-vs-tissue slide policy respected", sm["official_primary_match"].map(_truthy).all() and sm["biospecimen_match_level"].astype(str).isin(OFFICIAL_MATCH_LEVELS).all())
    v.check("match_rna present for all final cases", final_eligible["rna_file_uuid"].map(_nonempty).all())

    # Prefetch/download/final OpenSlide metadata.
    if prefetch_path.exists():
        prefetch = pd.read_parquet(prefetch_path)
        # Required columns only (extra columns allowed)
        _require_columns(v, prefetch, SLIDE_PREFETCH_COLUMNS, "slide_prefetch_manifest.parquet")
        v.check("slide technical metadata frozen", not prefetch.empty and "technical_prefetch_status" in prefetch.columns)
        downloaded = pd.read_csv(download_manifest_path, sep="\t", dtype=str, keep_default_na=False) if download_manifest_path.exists() else pd.DataFrame()
        if not downloaded.empty:
            v.check("all downloaded slides have prefetch status recorded", downloaded["slide_file_uuid"].astype(str).isin(set(prefetch["slide_file_uuid"].astype(str))).all())
            # Use the technical_prefetch_status of _7 (drop the copy of the download manifest)
            merged_dl = downloaded.drop(columns=["technical_prefetch_status"], errors="ignore").merge(
                prefetch[["slide_file_uuid", "technical_prefetch_status"]], on="slide_file_uuid", how="left"
            )
            if "override_prefetch_fail_used" in merged_dl.columns:
                # Conclusive prefetch failures present in the store without a recorded override
                bytes_present = merged_dl.get("download_status", pd.Series("", index=merged_dl.index)).astype(str).isin(SLIDE_BYTES_PRESENT_STATUSES)
                bad = merged_dl.loc[bytes_present & merged_dl["technical_prefetch_status"].astype(str).eq("failed_prefetch_conclusive") & (~merged_dl["override_prefetch_fail_used"].map(_truthy))]
                v.check("conclusive prefetch failures were not downloaded unless override recorded", bad.empty, f"bad={bad['slide_file_uuid'].head(20).tolist() if not bad.empty else []}")
            inconclusive = prefetch.loc[prefetch["technical_prefetch_status"].astype(str).isin({"inconclusive_prefetch", "range_not_supported", "prefetch_error_retryable"})]
            v.check("inconclusive prefetch slides are allowed and audited", True, f"n_inconclusive={len(inconclusive)}", warning=True if len(inconclusive) else False)
    else:
        prefetch = pd.DataFrame()
    if tile_asset_manifest:
        summary = tile_asset_manifest.get("summary") or {}
        v.check("full OpenSlide MPP is final source for tile geometry", str(tile_asset_manifest.get("final_mpp_source") or summary.get("final_mpp_source") or "full_openslide") == "full_openslide")
        v.check("prefetch-vs-full MPP discrepancies are recorded", "n_prefetch_full_mpp_discrepancies" in summary or "prefetch_full_mpp_consistency" in case_tile.columns)
        # No upsampled slide
        v.check(
            "no upsampled slides baked into benchmark (no-upsampling contract)",
            int(summary.get("n_slides_upsampled", 0)) == 0,
            f"n_slides_upsampled={summary.get('n_slides_upsampled')}",
        )

    # Dominance diagnostics.
    diag_cases = set(case_diag["case_id"].astype(str)) if "case_id" in case_diag.columns else set()
    v.check("case_tile_diagnostics covers all final cases", final_cases.issubset(diag_cases), f"missing={sorted(final_cases-diag_cases)[:20]}")
    for col in ["dominant_slide_tile_fraction_after_cap", "slide_tile_entropy_after_cap", "slide_tile_effective_n_after_cap"]:
        if col in case_diag.columns:
            vals = case_diag.loc[case_diag["case_id"].astype(str).isin(final_cases), col]
            v.check(f"{col} present for all final cases", vals.map(_nonempty).all())
    v.check("case_slide_contributions exists", not case_contrib.empty)

    # Project/domain/support annotations.
    domains = _domain_map(settings_dir)
    v.check("project/domain annotations complete", all(p in domains for p in set(final_case["project_id"].astype(str))))
    support_names = set()
    by = support.get("signature_support_by_name") or support.get("signatures_by_name") or {}
    if isinstance(by, Mapping):
        support_names = set(map(str, by.keys()))
    rows = support.get("signatures") or support.get("signature_support") or []
    if isinstance(rows, list):
        support_names |= {str(r.get("signature_name")) for r in rows if isinstance(r, Mapping) and r.get("signature_name")}
    v.check("signature support annotations complete", set(sig_names).issubset(support_names), f"missing={sorted(set(sig_names)-support_names)}")

    summary = {
        "benchmark_root": str(benchmark_root),
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": modelgroup_id,
        "tcga_source_id": tcga_source_id,
        "rna_asset_id": rna_asset_id,
        "tile_asset_id": tile_asset_id,
        "n_final_cases": int(len(final_eligible)),
        "n_case_tile_rows": int(len(case_tile)),
        "n_projects_final": int(final_eligible["project_id"].nunique()) if not final_eligible.empty else 0,
        "n_official_projects": int(len(official_projects)),
        "n_signature_names": int(len(sig_names)),
        "n_eligible_project_signature_cells": eligible_count,
        "n_primary_project_signature_cells": primary_count,
        "fold_count": fold_count,
        "fold_seed": fold_seed,
    }
    return _assemble_report(args, v, summary, benchmark_root, reports_dir)


def _assemble_report(args: argparse.Namespace, v: Validator, summary: Mapping[str, Any], benchmark_root: Path, reports_dir: Path) -> Dict[str, Any]:
    report = manifest_header(Path(__file__).name, vars(args))
    report.update({
        "status": "passed" if not v.errors else "failed",
        "frozen_final": bool(not v.errors),
        "benchmark_id": args.benchmark_id,
        "benchmark_root": str(benchmark_root),
        "summary": dict(summary),
        "n_checks": int(len(v.checks)),
        "n_errors": int(len(v.errors)),
        "n_warnings": int(len(v.warnings)),
        "errors": v.errors,
        "warnings": v.warnings,
        "checks": v.checks,
        "output_hashes_at_validation_time": collect_output_hashes(benchmark_root / "shared") if benchmark_root.exists() else {},
    })
    return report


def main(args) -> int:

    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    reports_dir = benchmark_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "benchmark_validation_report.json"
    md_path = reports_dir / "benchmark_validation_report.md"
    if (json_path.exists() or md_path.exists()) and not args.force:
        raise FileExistsError(f"Validation report already exists. Use --force to overwrite: {json_path}")

    report = validate_benchmark(args)
    write_json_atomic(json_path, report, overwrite=args.force)
    _write_markdown(md_path, report, overwrite=args.force)
    print(json.dumps({"status": report["status"], "n_errors": report["n_errors"], "n_warnings": report["n_warnings"], "report": str(json_path)}, indent=2))
    if report["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Validate the frozen TCGA Task B benchmark before encoder evaluation.")

    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help="Benchmark ID to validate.")

    ap.add_argument("--target_variance_min", type=float,
                    default=1e-6,
                    help="Minimum variance a signature target must have to be considered valid.")

    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing validation reports.")

    args = ap.parse_args()
    raise SystemExit(main(args))
