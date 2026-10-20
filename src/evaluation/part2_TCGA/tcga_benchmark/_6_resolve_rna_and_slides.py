#!/usr/bin/env python3
"""
Resolve one RNA file per eligible TCGA case and its official-primary diagnostic slides (metadata only), then apply
the per-case slide cap and the per-project case cap (TSS-stratified round-robin, with an over-provision buffer).

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data / benchmarks / <benchmark_id> /
    - shared/cohort/: rna_manifest.csv, rna_resolution_audit.csv, provisional_case_manifest.csv, slide_manifest.csv,
      slide_resolution_audit.csv, exclusion_log.csv, rna_slide_resolution_manifest.json
    - reports/rna_slide_resolution_report.md
"""

import argparse
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
import sys

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path

from _taskb_common import (
    case_cap_order,
    collect_output_hashes,
    jsonish,
    manifest_header,
    parse_simple_version,
    project_tss_distribution,
    projected_fold_verdict,
    read_json,
    read_yaml,
    sha256_obj,
    sha256_path,
    stable_project_seed,
    tissue_source_site,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
    version_ok_uuids,
)

logger = logging.getLogger(__name__)

PRIMARY_SAMPLE_TYPES = {"Primary Tumor", "Primary Solid Tumor"}
OFFICIAL_MATCH_LEVELS = {"exact_same_portion", "same_sample", "same_case_primary_tumor", "same_case_primary_tumor_unknown_sublevel"}
EXPLORATORY_MATCH_LEVELS: Set[str] = set()
PATHOLOGY_PERCENT_COLUMNS = [
    "percent_tumor_cells",
    "percent_normal_cells",
    "percent_necrosis",
    "percent_stromal_cells",
    "percent_lymphocyte_infiltration",
]

SLIDE_MANIFEST_COLUMNS = [
    "case_id", "case_uuid", "project_id", "rna_file_uuid", "rna_file_name",
    "rna_sample_id", "rna_sample_submitter_id", "rna_portion_id", "rna_portion_submitter_id",
    "rna_analyte_id", "rna_analyte_submitter_id", "rna_aliquot_id", "rna_aliquot_submitter_id",
    "slide_file_uuid", "slide_file_name", "file_name", "file_size", "md5sum", "version", "data_release",
    "data_category", "data_type", "data_format", "experimental_strategy", "access", "state",
    "sample_id", "sample_submitter_id", "sample_type", "tumor_descriptor", "tissue_type",
    "portion_id", "portion_submitter_id", "is_ffpe", "slide_id", "slide_submitter_id",
    "section_location", "slide_state", "stain_type", "magnification",
    "biospecimen_match_level", "match_tier", "match_tier_best_case", "official_primary_match",
    "selected_for_official_primary", "unknown_stain", "unknown_ffpe",
    "case_level_metadata_fallback", "metadata_eligibility_status", "metadata_exclusion_reason",
    "annotation_excluded", "version_provenance_ok", "benchmark_id", "modelgroup_id",
] + PATHOLOGY_PERCENT_COLUMNS

RNA_MANIFEST_COLUMNS = [
    "case_id", "case_uuid", "project_id", "rna_file_uuid", "file_name", "rna_file_name", "file_size", "md5sum",
    "version", "data_release", "workflow_type", "workflow_version", "access", "state",
    "sample_id", "sample_submitter_id", "sample_type", "tumor_descriptor", "tissue_type",
    "portion_id", "portion_submitter_id", "is_ffpe", "analyte_id", "analyte_submitter_id",
    "analyte_type", "aliquot_id", "aliquot_submitter_id", "parent_resolution_status",
    "n_eligible_rna_files_for_case", "rna_selection_rank", "rna_selection_reason",
    "case_selection_rank", "benchmark_id",
]

CASE_MANIFEST_COLUMNS = [
    "case_id", "case_uuid", "project_id", "rna_file_uuid", "rna_file_name",
    "n_metadata_eligible_slides", "n_exact_same_portion_slides", "n_same_sample_slides",
    "n_same_case_primary_tumor_slides", "n_same_case_primary_tumor_unknown_sublevel_slides",
    "match_tier_best_case", "biospecimen_match_level_best_case", "official_primary_provisional",
    "n_slides_available_best_tier", "n_official_primary_slides_selected", "case_exclusion_reason",
    "benchmark_id", "modelgroup_id",
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


def _clean(x: Any) -> Optional[str]:
    return str(x).strip() if _nonempty(x) else None


def _as_bool_or_none(x: Any) -> Optional[bool]:
    if not _nonempty(x):
        return None
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return None


def _is_primary_sample(sample_type: Any) -> bool:
    if not _nonempty(sample_type):
        return True  # missing sample type: not excluded
    return str(sample_type).strip() in PRIMARY_SAMPLE_TYPES


def _is_primary_sample_strict(sample_type: Any) -> bool:
    return str(sample_type).strip() in PRIMARY_SAMPLE_TYPES if _nonempty(sample_type) else False


def _version_key(value: Any, width: int = 6, n: int = 6) -> Tuple[int, ...]:
    nums = tuple(parse_simple_version(value))[:n]
    return nums + tuple(0 for _ in range(max(0, n - len(nums))))


def _timestamp_key(value: Any) -> float:
    if not _nonempty(value):
        return float("-inf")
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return float("-inf")


def _truthy(x: Any) -> bool:
    b = _as_bool_or_none(x)
    if b is not None:
        return b
    return bool(x)


def _load_included_projects(settings_dir: Path) -> Set[str]:
    inc_path = settings_dir / "included_projects.json"
    if inc_path.exists():
        obj = read_json(inc_path)
        vals = obj.get("included_projects") if isinstance(obj, Mapping) else None
        if isinstance(vals, list) and vals:
            return set(str(x) for x in vals)
    domain_path = settings_dir / "project_domain_map.json"
    if not domain_path.exists():
        raise FileNotFoundError("Cannot resolve included projects; missing included_projects.json and project_domain_map.json")
    obj = read_json(domain_path)
    labels = obj.get("project_domain_map") or obj.get("projects") or {}
    return set(str(k) for k, v in labels.items() if v in {"in_domain", "out_of_domain"})


def _load_excluded_entities(source_root: Path) -> Set[str]:
    path = source_root / "inventory" / "annotation_exclusion_log.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "entity_id" not in df.columns:
        return set()
    return set(str(x).strip() for x in df["entity_id"].tolist() if _nonempty(x))


def _load_version_ok(source_root: Path) -> Set[str]:
    path = source_root / "inventory" / "file_versions.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    return version_ok_uuids(
        df,
        allow_low_confidence_fallback=False,
        require_latest=False,
    )


def _load_usable_rna_uuids(rna_asset_root: Path) -> Set[str]:
    expr_path = rna_asset_root / "case_expression_log10_fpkmuq.parquet"
    if not expr_path.exists():
        raise FileNotFoundError(expr_path)
    df = pd.read_parquet(expr_path, columns=None)
    if "rna_file_uuid" not in df.columns:
        raise RuntimeError(f"RNA asset expression table missing rna_file_uuid: {expr_path}")
    return set(str(x).strip() for x in df["rna_file_uuid"].tolist() if _nonempty(x))


def _entity_annotation_excluded(row: Mapping[str, Any], excluded_entities: Set[str], fields: Sequence[str]) -> bool:
    for f in fields:
        v = row.get(f)
        if _nonempty(v) and str(v).strip() in excluded_entities:
            return True
    return False


def _eligible_rna_flags(
    row: Mapping[str, Any],
    included_projects: Set[str],
    excluded_entities: Set[str],
    version_ok: Set[str],
    usable_rna_uuids: Optional[Set[str]] = None,
    *,
    require_parsed_rna_asset: bool = False,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    reasons: List[str] = []
    flags: Dict[str, Any] = {}
    uuid = str(row.get("rna_file_uuid") or "")
    flags["project_included"] = row.get("project_id") in included_projects
    flags["primary_tumor_sample"] = _is_primary_sample(row.get("sample_type"))
    flags["workflow_ok"] = str(row.get("workflow_type") or "") == "STAR - Counts"
    flags["released_open"] = str(row.get("access") or "") == "open" and str(row.get("state") or "") == "released"
    flags["parent_resolved"] = str(row.get("parent_resolution_status") or "") == "resolved"
    diag = row.get("diagnosis_is_primary_disease")
    flags["primary_diagnosis_ok"] = (not _nonempty(diag)) or str(diag).strip().lower() in {"true", "1", "yes"}
    flags["annotation_ok"] = not _entity_annotation_excluded(
        row,
        excluded_entities,
        ["case_uuid", "sample_id", "portion_id", "analyte_id", "aliquot_id", "rna_file_uuid"],
    )
    flags["file_version_ok"] = uuid in version_ok
    if require_parsed_rna_asset:
        flags["parsed_in_rna_asset"] = bool(usable_rna_uuids is not None and uuid in usable_rna_uuids)
    else:
        flags["parsed_in_rna_asset"] = "not_required_deferred_rna_asset_build"
    reason_map = {
        "project_included": "project_not_retained",
        "primary_tumor_sample": "not_primary_tumor_rna_sample",
        "workflow_ok": "unsupported_rna_workflow",
        "released_open": "rna_not_released_open",
        "parent_resolved": "rna_parent_not_unambiguously_resolved",
        "primary_diagnosis_ok": "diagnosis_is_not_primary_disease",
        "annotation_ok": "rna_or_parent_annotation_excluded",
        "file_version_ok": "rna_file_version_provenance_missing_or_not_open_released",
    }
    if require_parsed_rna_asset:
        reason_map["parsed_in_rna_asset"] = "rna_file_not_present_in_parsed_rna_asset"
    for k, reason in reason_map.items():
        if not flags[k]:
            reasons.append(reason)
    return not reasons, reasons, flags


def _rna_sort_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    workflow_rank = 0 if str(row.get("workflow_type") or "") == "STAR - Counts" else 9
    state_rank = 0 if str(row.get("state") or "") == "released" else 9
    # Descending version-ish keys are represented as negative components.
    workflow_version_desc = tuple(-x for x in _version_key(row.get("workflow_version")))
    data_release_desc = tuple(-x for x in _version_key(row.get("data_release")))
    version_desc = tuple(-x for x in _version_key(row.get("version")))
    updated_desc = -_timestamp_key(row.get("updated_datetime"))
    return (
        workflow_rank,
        workflow_version_desc,
        state_rank,
        data_release_desc,
        version_desc,
        updated_desc,
        str(row.get("file_name") or ""),
        str(row.get("rna_file_uuid") or ""),
    )


def _select_rna(
    rna_df: pd.DataFrame,
    included_projects: Set[str],
    excluded_entities: Set[str],
    version_ok: Set[str],
    usable_rna_uuids: Optional[Set[str]],
    benchmark_id: str,
    *,
    require_parsed_rna_asset: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit_rows: List[Dict[str, Any]] = []
    eligible_rows: List[Dict[str, Any]] = []
    exclusion_rows: List[Dict[str, Any]] = []
    for row in rna_df.to_dict("records"):
        eligible, reasons, flags = _eligible_rna_flags(
            row,
            included_projects,
            excluded_entities,
            version_ok,
            usable_rna_uuids,
            require_parsed_rna_asset=require_parsed_rna_asset,
        )
        out = dict(row)
        out.update({
            "rna_metadata_eligible": bool(eligible),
            "rna_exclusion_reasons": ";".join(reasons),
            "rna_eligibility_flags_json": jsonish(flags),
        })
        audit_rows.append(out)
        if eligible:
            eligible_rows.append(dict(row))
        elif _nonempty(row.get("case_id")):
            exclusion_rows.append({
                "stage": "rna_resolution",
                "entity_type": "rna_file",
                "entity_id": row.get("rna_file_uuid"),
                "case_id": row.get("case_id"),
                "case_uuid": row.get("case_uuid"),
                "project_id": row.get("project_id"),
                "reason": ";".join(reasons) or "rna_not_eligible",
                "detail": jsonish(flags),
                "benchmark_id": benchmark_id,
            })
    if not eligible_rows:
        raise RuntimeError("No eligible RNA files after frozen metadata rules")

    selected: List[Dict[str, Any]] = []
    eligible_df = pd.DataFrame(eligible_rows)
    for case_id, g in eligible_df.groupby("case_id", sort=True):
        records = g.to_dict("records")
        records_sorted = sorted(records, key=_rna_sort_key)
        chosen = dict(records_sorted[0])
        chosen["n_eligible_rna_files_for_case"] = int(len(records_sorted))
        chosen["rna_selection_rank"] = 1
        chosen["rna_selection_reason"] = "deterministic_rank_workflow_version_state_release_version_updated_file_name_uuid"
        selected.append(chosen)
        if len(records_sorted) > 1:
            for r in records_sorted[1:]:
                exclusion_rows.append({
                    "stage": "rna_resolution",
                    "entity_type": "rna_file",
                    "entity_id": r.get("rna_file_uuid"),
                    "case_id": r.get("case_id"),
                    "case_uuid": r.get("case_uuid"),
                    "project_id": r.get("project_id"),
                    "reason": "lower_ranked_competing_primary_tumor_rna_file_not_selected",
                    "detail": jsonish({"selected_rna_file_uuid": chosen.get("rna_file_uuid"), "n_eligible_rna_files_for_case": len(records_sorted)}),
                    "benchmark_id": benchmark_id,
                })

    selected_df = pd.DataFrame(selected).sort_values(["project_id", "case_id", "rna_file_uuid"]).reset_index(drop=True)
    selected_df["rna_file_name"] = selected_df.get("file_name")
    selected_df["benchmark_id"] = benchmark_id
    audit_df = pd.DataFrame(audit_rows).sort_values(["project_id", "case_id", "rna_file_uuid"]).reset_index(drop=True)
    excl_df = pd.DataFrame(exclusion_rows)
    return selected_df, audit_df, excl_df


# Case-cap priority order (shared with _11_build_case_tile_index.py)
_case_cap_order = case_cap_order


def _cap_cases_per_project(
    selected_rna: pd.DataFrame,
    *,
    target_cap: Optional[int],
    overprovision_frac: float,
    seed: int,
    method: str,
    benchmark_id: str,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """Cap the cases per project: keep the first ceil(target_cap * (1 + overprovision_frac)) cases in case_cap_order.

    Every case gets a 1-based case_selection_rank (used by _11 to trim back to target_cap); target_cap None or <= 0
    disables the cap. Returns (kept cases, exclusion-log rows, summary).
    """
    cap_enabled = target_cap is not None and int(target_cap) > 0
    if target_cap is not None and not cap_enabled:
        target_cap = None  # <= 0: no cap
    if method not in {"tss_stratified_round_robin", "random"}:
        raise RuntimeError(f"Unknown case_cap_method={method!r}")
    frac = float(overprovision_frac) if (overprovision_frac and overprovision_frac > 0) else 0.0
    n_select = int(math.ceil(int(target_cap) * (1.0 + frac))) if cap_enabled else None

    n_before_total = int(len(selected_rna))
    exclusion_rows: List[Dict[str, Any]] = []
    per_project_stats: List[Dict[str, Any]] = []
    kept_frames: List[pd.DataFrame] = []

    if not selected_rna.empty:
        for project_id, g in selected_rna.groupby("project_id", sort=True):
            g = g.reset_index(drop=True)
            n_before = int(len(g))
            n_tss_before = len({tissue_source_site(c) for c in g["case_id"].astype(str).tolist()})
            rng = np.random.default_rng(
                stable_project_seed(int(seed), str(project_id), namespace="case_cap_seed")
            )
            order = _case_cap_order(g, method=method, rng=rng)  # full priority permutation
            # 1-based selection rank of every case (used by the _11 trim)
            rank_by_pos = {int(pos): (i + 1) for i, pos in enumerate(order)}
            g = g.assign(case_selection_rank=[int(rank_by_pos[int(p)]) for p in range(n_before)])
            if (not cap_enabled) or n_before <= int(n_select):
                kept = g
            else:
                keep_pos = order[: int(n_select)]
                drop_pos = order[int(n_select):]
                kept = g.iloc[sorted(int(p) for p in keep_pos)].copy()
                dropped = g.iloc[sorted(int(p) for p in drop_pos)]
                for _, r in dropped.iterrows():
                    exclusion_rows.append({
                        "stage": "per_project_case_cap",
                        "entity_type": "case",
                        "entity_id": r.get("case_uuid"),
                        "case_id": r.get("case_id"),
                        "case_uuid": r.get("case_uuid"),
                        "project_id": project_id,
                        "reason": "per_project_case_cap_exceeded",
                        "detail": jsonish({
                            "per_project_target_cap": int(target_cap),
                            "per_project_download_cap": int(n_select),
                            "case_cap_overprovision_frac": frac,
                            "case_cap_method": method,
                            "case_cap_seed": int(seed),
                            "n_viable_cases_before_cap": n_before,
                            "case_selection_rank": int(r.get("case_selection_rank")),
                        }),
                        "benchmark_id": benchmark_id,
                    })
            n_after = int(len(kept))
            n_tss_after = len({tissue_source_site(c) for c in kept["case_id"].astype(str).tolist()})
            kept_frames.append(kept)
            per_project_stats.append({
                "project_id": str(project_id),
                "n_viable_cases_before_cap": n_before,
                "n_cases_selected_for_download": n_after,
                "n_cases_dropped_by_cap": n_before - n_after,
                "n_tss_before_cap": int(n_tss_before),
                "n_tss_after_cap": int(n_tss_after),
            })

    if kept_frames:
        capped = pd.concat(kept_frames, ignore_index=True).sort_values(
            ["project_id", "case_id", "rna_file_uuid"]
        ).reset_index(drop=True)
    else:
        capped = selected_rna.copy()
        if "case_selection_rank" not in capped.columns:
            capped["case_selection_rank"] = pd.Series(dtype="Int64")

    summary = {
        "case_cap_applied": bool(cap_enabled),
        # Final per-project target (read by _11 as the trim target)
        "max_cases_per_project": (int(target_cap) if cap_enabled else None),
        "per_project_target_cap": (int(target_cap) if cap_enabled else None),
        "per_project_download_cap": (int(n_select) if cap_enabled else None),
        "case_cap_overprovision_frac": frac,
        "case_cap_method": (method if cap_enabled else "disabled"),
        "case_cap_seed": int(seed),
        "n_cases_before_cap": n_before_total,
        "n_cases_selected_for_download": int(len(capped)),
        "n_cases_dropped_by_cap": int(n_before_total - len(capped)),
        "per_project": sorted(per_project_stats, key=lambda d: d["project_id"]),
    }
    return capped, exclusion_rows, summary


def _classify_stain(stain: Any, experimental_strategy: Any) -> Tuple[bool, bool, Optional[str]]:
    if not _nonempty(stain):
        if str(experimental_strategy or "") == "Diagnostic Slide":
            return True, True, None
        return False, True, "unknown_stain_not_allowed_for_non_diagnostic_slide"
    text = str(stain).strip()
    low = text.lower()
    excluded_terms = ["ihc", "immunohistochemistry", "special stain", "fluorescence"]
    if any(t in low for t in excluded_terms) or low.strip() == "if":
        return False, False, "stain_metadata_indicates_non_he"
    compact = low.replace(" ", "").replace("-", "").replace("_", "")
    if "h&e" in low or "hematoxylinandeosin" in compact or compact in {"he", "hestain", "hematoxylineosin"}:
        return True, False, None
    # Unknown stain values on diagnostic slides are accepted and flagged
    if low in {"unknown", "not reported", "not applicable", "na", "n/a"} and str(experimental_strategy or "") == "Diagnostic Slide":
        return True, True, None
    return False, False, "stain_not_he_compatible"


def _eligible_slide_flags(row: Mapping[str, Any], included_projects: Set[str], excluded_entities: Set[str]) -> Tuple[bool, List[str], Dict[str, Any]]:
    # No file-version check for slides (file_versions.parquet has RNA files only); state/access are used instead
    reasons: List[str] = []
    stain_ok, unknown_stain, stain_reason = _classify_stain(row.get("stain_type"), row.get("experimental_strategy"))
    ffpe = _as_bool_or_none(row.get("is_ffpe"))
    ffpe_ok = (ffpe is True) or (ffpe is None)
    parent_status = str(row.get("parent_resolution_status") or "")
    slide_state_text = str(row.get("slide_state") or "").strip().lower()
    slide_state_usable = not any(term in slide_state_text for term in ["unusable", "withdrawn", "redacted", "suppressed", "lost", "failed", "excluded"])
    flags = {
        "project_included": row.get("project_id") in included_projects,
        "slide_image": str(row.get("data_type") or "") == "Slide Image",
        "diagnostic_slide": str(row.get("experimental_strategy") or "") == "Diagnostic Slide",
        "released_open": str(row.get("access") or "") == "open" and str(row.get("state") or "") == "released",
        "primary_tumor_sample_when_available": _is_primary_sample(row.get("sample_type")),
        "stain_he_compatible": stain_ok,
        "unknown_stain": bool(unknown_stain),
        "ffpe_ok": ffpe_ok,
        "unknown_ffpe": bool(ffpe is None),
        "slide_state_usable": slide_state_usable,
        "annotation_ok": not _entity_annotation_excluded(row, excluded_entities, ["case_uuid", "sample_id", "portion_id", "slide_id", "slide_file_uuid"]),
        "parent_resolution_usable": parent_status in {"resolved", "case_level_fallback"},
    }
    reason_map = {
        "project_included": "project_not_retained",
        "slide_image": "not_slide_image",
        "diagnostic_slide": "not_diagnostic_slide",
        "released_open": "slide_not_released_open",
        "primary_tumor_sample_when_available": "not_primary_tumor_slide_sample",
        "stain_he_compatible": stain_reason or "stain_not_he_compatible",
        "ffpe_ok": "non_ffpe_portion_metadata",
        "slide_state_usable": "slide_metadata_marked_unusable",
        "annotation_ok": "slide_or_parent_annotation_excluded",
        "parent_resolution_usable": "slide_parent_not_usable",
    }
    for k, reason in reason_map.items():
        if k in {"unknown_stain", "unknown_ffpe"}:
            continue
        if not flags[k]:
            reasons.append(reason)
    return not reasons, reasons, flags


def _match_level(rna: Mapping[str, Any], slide: Mapping[str, Any]) -> str:
    rna_portion = _clean(rna.get("portion_id"))
    slide_portion = _clean(slide.get("portion_id"))
    if rna_portion and slide_portion and rna_portion == slide_portion:
        return "exact_same_portion"
    rna_sample = _clean(rna.get("sample_id"))
    slide_sample = _clean(slide.get("sample_id"))
    if rna_sample and slide_sample and rna_sample == slide_sample:
        return "same_sample"
    if _is_primary_sample_strict(slide.get("sample_type")):
        return "same_case_primary_tumor"
    return "same_case_primary_tumor_unknown_sublevel"


def _tier(level: str) -> int:
    if level == "exact_same_portion":
        return 0
    if level == "same_sample":
        return 1
    if level in ("same_case_primary_tumor", "same_case_primary_tumor_unknown_sublevel"):
        return 2
    return 9


def _resolve_slides(
    selected_rna: pd.DataFrame,
    slide_df: pd.DataFrame,
    *,
    included_projects: Set[str],
    excluded_entities: Set[str],
    benchmark_id: str,
    modelgroup_id: str,
    max_slides_per_case: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_by_case = {str(r["case_id"]): dict(r) for r in selected_rna.to_dict("records")}
    selected_cases = set(selected_by_case)
    audit_rows: List[Dict[str, Any]] = []
    eligible_match_rows: List[Dict[str, Any]] = []
    exclusion_rows: List[Dict[str, Any]] = []

    source_rows = slide_df.loc[slide_df["case_id"].astype(str).isin(selected_cases)].copy() if "case_id" in slide_df.columns else slide_df.iloc[0:0].copy()
    for row in source_rows.to_dict("records"):
        case_id = str(row.get("case_id"))
        rna = selected_by_case.get(case_id)
        if rna is None:
            continue
        eligible, reasons, flags = _eligible_slide_flags(row, included_projects, excluded_entities)
        level = _match_level(rna, row) if eligible else "not_applicable_metadata_ineligible"
        tier = _tier(level)
        base = dict(row)
        base.update({
            "rna_file_uuid": rna.get("rna_file_uuid"),
            "rna_file_name": rna.get("file_name"),
            "rna_sample_id": rna.get("sample_id"),
            "rna_sample_submitter_id": rna.get("sample_submitter_id"),
            "rna_portion_id": rna.get("portion_id"),
            "rna_portion_submitter_id": rna.get("portion_submitter_id"),
            "rna_analyte_id": rna.get("analyte_id"),
            "rna_analyte_submitter_id": rna.get("analyte_submitter_id"),
            "rna_aliquot_id": rna.get("aliquot_id"),
            "rna_aliquot_submitter_id": rna.get("aliquot_submitter_id"),
            "slide_file_name": row.get("file_name"),
            "biospecimen_match_level": level,
            "match_tier": tier,
            "official_primary_match": bool(level in OFFICIAL_MATCH_LEVELS),
            "unknown_stain": bool(flags.get("unknown_stain", False)),
            "unknown_ffpe": bool(flags.get("unknown_ffpe", False)),
            "metadata_eligibility_status": "eligible" if eligible else "excluded",
            "metadata_exclusion_reason": ";".join(reasons),
            "metadata_eligibility_flags_json": jsonish(flags),
            "annotation_excluded": not bool(flags.get("annotation_ok", True)),
            # released/open state (no file-version check for slides)
            "version_provenance_ok": bool(flags.get("released_open", False)),
            "selected_for_official_primary": False,
            "match_tier_best_case": None,
            "benchmark_id": benchmark_id,
            "modelgroup_id": modelgroup_id,
        })
        audit_rows.append(base)
        if eligible:
            eligible_match_rows.append(base)
        else:
            exclusion_rows.append({
                "stage": "slide_metadata_resolution",
                "entity_type": "slide_file",
                "entity_id": row.get("slide_file_uuid"),
                "case_id": row.get("case_id"),
                "case_uuid": row.get("case_uuid"),
                "project_id": row.get("project_id"),
                "reason": ";".join(reasons) or "slide_not_metadata_eligible",
                "detail": jsonish(flags),
                "benchmark_id": benchmark_id,
            })

    eligible_df = pd.DataFrame(eligible_match_rows)
    official_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    for rna in selected_rna.sort_values(["project_id", "case_id"]).to_dict("records"):
        case_id = str(rna["case_id"])
        g = eligible_df.loc[eligible_df["case_id"].astype(str).eq(case_id)].copy() if not eligible_df.empty else pd.DataFrame()
        n_by_level = {lvl: 0 for lvl in ["exact_same_portion", "same_sample", "same_case_primary_tumor", "same_case_primary_tumor_unknown_sublevel"]}
        best_tier: Optional[int] = None
        best_level: Optional[str] = None
        n_slides_available_best_tier = 0
        selected_case_rows: List[Dict[str, Any]] = []
        if not g.empty:
            for lvl, gg in g.groupby("biospecimen_match_level"):
                if lvl in n_by_level:
                    n_by_level[lvl] = int(len(gg))
            official_g = g.loc[g["biospecimen_match_level"].isin(OFFICIAL_MATCH_LEVELS)].copy()
            if not official_g.empty:
                # Prefer tighter tier per case (Tier 0 > 1 > 2); never mix tiers within a case.
                best_tier = int(official_g["match_tier"].min())
                tier_rows = official_g.loc[official_g["match_tier"].eq(best_tier)].copy()
                n_slides_available_best_tier = int(len(tier_rows))
                # Rank the slides of the case (biospecimen level, H&E-known, FFPE-known, 20x, larger file, uuid)
                # and keep the top max_slides_per_case
                level_priority = ["exact_same_portion", "same_sample", "same_case_primary_tumor", "same_case_primary_tumor_unknown_sublevel"]
                level_rank = {lvl: i for i, lvl in enumerate(level_priority)}
                tier_rows["_rank_level"] = tier_rows["biospecimen_match_level"].astype(str).map(lambda v: level_rank.get(v, 9))
                tier_rows["_rank_stain"] = tier_rows["unknown_stain"].map(_truthy).astype(int)
                tier_rows["_rank_ffpe"] = tier_rows["unknown_ffpe"].map(_truthy).astype(int)
                # Magnification rank (GDC metadata): 20x, then > 20x, then unknown, then < 20x
                _mag = pd.to_numeric(tier_rows.get("magnification"), errors="coerce")
                tier_rows["_rank_resolution"] = np.select(
                    [_mag.isna(), (_mag - 20.0).abs() < 1e-6, _mag > 20.0],
                    [2, 0, 1],
                    default=3,
                ).astype(int)
                tier_rows["_rank_size"] = -pd.to_numeric(tier_rows.get("file_size"), errors="coerce").fillna(0.0)
                tier_rows = tier_rows.sort_values(
                    ["_rank_level", "_rank_stain", "_rank_ffpe", "_rank_resolution", "_rank_size", "slide_file_uuid"]
                ).reset_index(drop=True)
                kept = tier_rows if (max_slides_per_case is None or int(max_slides_per_case) <= 0) else tier_rows.head(int(max_slides_per_case))
                kept = kept.drop(columns=["_rank_level", "_rank_stain", "_rank_ffpe", "_rank_resolution", "_rank_size"])
                best_level = str(kept["biospecimen_match_level"].iloc[0])
                selected_case_rows = kept.to_dict("records")
        official = bool(selected_case_rows)
        case_reason = "" if official else ("no_eligible_diagnostic_primary_tumor_slide_at_any_official_tier" if not g.empty else "no_metadata_eligible_diagnostic_slide")
        if not official:
            exclusion_rows.append({
                "stage": "slide_biospecimen_resolution",
                "entity_type": "case",
                "entity_id": rna.get("case_uuid"),
                "case_id": rna.get("case_id"),
                "case_uuid": rna.get("case_uuid"),
                "project_id": rna.get("project_id"),
                "reason": case_reason,
                "detail": jsonish(n_by_level),
                "benchmark_id": benchmark_id,
            })
        for sr in selected_case_rows:
            sr = dict(sr)
            sr["selected_for_official_primary"] = True
            sr["match_tier_best_case"] = best_tier
            official_rows.append(sr)
        case_rows.append({
            "case_id": rna.get("case_id"),
            "case_uuid": rna.get("case_uuid"),
            "project_id": rna.get("project_id"),
            "rna_file_uuid": rna.get("rna_file_uuid"),
            "rna_file_name": rna.get("file_name"),
            "n_metadata_eligible_slides": int(len(g)),
            "n_exact_same_portion_slides": n_by_level["exact_same_portion"],
            "n_same_sample_slides": n_by_level["same_sample"],
            "n_same_case_primary_tumor_slides": n_by_level["same_case_primary_tumor"],
            "n_same_case_primary_tumor_unknown_sublevel_slides": n_by_level["same_case_primary_tumor_unknown_sublevel"],
            "match_tier_best_case": best_tier,
            "biospecimen_match_level_best_case": best_level,
            "official_primary_provisional": official,
            "n_slides_available_best_tier": int(n_slides_available_best_tier),
            "n_official_primary_slides_selected": int(len(selected_case_rows)),
            "case_exclusion_reason": case_reason,
            "benchmark_id": benchmark_id,
            "modelgroup_id": modelgroup_id,
        })

    audit_df = pd.DataFrame(audit_rows)
    if not audit_df.empty:
        audit_df = audit_df.sort_values(["project_id", "case_id", "slide_file_uuid"]).reset_index(drop=True)
    slide_manifest = pd.DataFrame(official_rows)
    if not slide_manifest.empty:
        slide_manifest = slide_manifest.sort_values(["project_id", "case_id", "match_tier", "slide_file_uuid"]).reset_index(drop=True)
        # Keep both file_name and slide_file_name
        slide_manifest["file_name"] = slide_manifest.get("file_name", slide_manifest.get("slide_file_name"))
        slide_manifest["slide_file_name"] = slide_manifest.get("slide_file_name", slide_manifest.get("file_name"))
        for c in SLIDE_MANIFEST_COLUMNS:
            if c not in slide_manifest.columns:
                slide_manifest[c] = pd.NA
        slide_manifest = slide_manifest.loc[:, SLIDE_MANIFEST_COLUMNS]
    case_manifest = pd.DataFrame(case_rows)
    if not case_manifest.empty:
        case_manifest = case_manifest.sort_values(["project_id", "case_id"]).reset_index(drop=True)
        for c in CASE_MANIFEST_COLUMNS:
            if c not in case_manifest.columns:
                case_manifest[c] = pd.NA
        case_manifest = case_manifest.loc[:, CASE_MANIFEST_COLUMNS]
    excl_df = pd.DataFrame(exclusion_rows)
    return slide_manifest, audit_df, case_manifest, excl_df


def _ordered(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = pd.NA
    extras = [c for c in out.columns if c not in columns]
    return out.loc[:, list(columns) + extras]


def _write_report(path: Path, summary: Mapping[str, Any], *, overwrite: bool) -> None:
    lines = [
        "# RNA and slide metadata-resolution report",
        "",
        f"- benchmark_id: `{summary.get('benchmark_id')}`",
        f"- modelgroup_id: `{summary.get('modelgroup_id')}`",
        f"- eligible RNA cases (pre-cap): `{summary.get('n_eligible_rna_cases')}` "
        f"(without official-primary slide: `{summary.get('n_cases_without_official_primary_slide')}`, "
        f"viable: `{summary.get('n_viable_cases')}`)",
        f"- per-project case cap (final target): `{summary.get('per_project_target_cap')}` "
        f"(download cap `{summary.get('per_project_download_cap')}` @ over-provision "
        f"`{summary.get('case_cap_overprovision_frac')}`, method `{summary.get('case_cap_method')}`, "
        f"applied `{summary.get('case_cap_applied')}`)",
        f"- max slides per case: `{summary.get('max_slides_per_case')}`",
        f"- cases selected for download: `{summary.get('n_cases_selected_for_download')}` "
        f"(from `{summary.get('n_viable_cases_before_cap')}` viable; dropped by cap "
        f"`{summary.get('n_cases_dropped_by_cap')}`)",
        f"- tier 0 exact-same-portion cases: `{summary.get('n_exact_same_portion_cases')}`",
        f"- tier 1 same-sample cases: `{summary.get('n_same_sample_cases')}`",
        f"- tier 2 same-case-primary-tumor cases: `{summary.get('n_same_case_primary_tumor_cases')}`",
        f"- expected official-primary cases before tile QC: `{summary.get('expected_n_official_primary_before_tile_qc')}`",
        "",
        "This step is metadata-only; no OpenSlide/MPP/tile-read requirement was applied. The "
        "download cohort is over-provisioned; _11_build_case_tile_index.py trims tile-QC "
        "survivors back to the per-project target.",
        "",
    ]
    write_text_atomic(path, "\n".join(lines), overwrite=overwrite)


def main(args) -> int:

    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    settings_dir = benchmark_root / "settings"
    cohort_dir = benchmark_root / "shared" / "cohort"
    reports_dir = benchmark_root / "reports"
    if not settings_dir.exists():
        raise FileNotFoundError(settings_dir)
    cohort_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        cohort_dir / "rna_manifest.csv",
        cohort_dir / "rna_resolution_audit.csv",
        cohort_dir / "provisional_case_manifest.csv",
        cohort_dir / "slide_manifest.csv",
        cohort_dir / "slide_resolution_audit.csv",
        cohort_dir / "exclusion_log.csv",
        cohort_dir / "rna_slide_resolution_manifest.json",
    ]
    if any(p.exists() for p in outputs) and not args.force:
        existing = [str(p) for p in outputs if p.exists()]
        raise FileExistsError(f"Resolution outputs already exist; use --force to rebuild. Existing: {existing[:5]}")
    downstream = [data_root / "selected_slide_cache" / args.benchmark_id, data_root / "processed_assets" / "tiles"]
    if args.force and any(p.exists() for p in downstream) and not args.force_cascade:
        raise RuntimeError(f"Refusing to rebuild RNA/slide resolution while downstream assets exist: {downstream}. Use --force_cascade if intentional.")

    cfg = _load_mapping(settings_dir / "benchmark_config.yaml")
    links = read_json(settings_dir / "benchmark_asset_links.json")
    if not isinstance(links, Mapping):
        raise RuntimeError("benchmark_asset_links.json must be a mapping")
    modelgroup_id = str(cfg.get("modelgroup_id") or links.get("modelgroup_id"))
    # Case-cap settings: CLI value, else benchmark_config.yaml (max_cases_per_project None / <= 0: no cap)
    max_cases_per_project = args.max_cases_per_project if args.max_cases_per_project is not None else cfg.get("max_cases_per_project")
    max_cases_per_project = int(max_cases_per_project) if max_cases_per_project is not None else None
    case_cap_seed = int(args.case_cap_seed if args.case_cap_seed is not None else cfg.get("case_cap_seed", 20260518))
    case_cap_method = str(args.case_cap_method if args.case_cap_method is not None else cfg.get("case_cap_method", "tss_stratified_round_robin"))
    case_cap_overprovision_frac = float(args.case_cap_overprovision_frac if args.case_cap_overprovision_frac is not None else cfg.get("case_cap_overprovision_frac", 0.0))
    # Per-case slide cap (None / <= 0: all best-tier slides)
    max_slides_per_case = args.max_slides_per_case if args.max_slides_per_case is not None else cfg.get("max_slides_per_case", 1)
    max_slides_per_case = int(max_slides_per_case) if max_slides_per_case is not None else None
    # Used only for the projected-fold diagnostic (folds are built by _13)
    n_folds_cfg = int(cfg.get("fold_count", 5))
    degenerate_tss_fraction_cfg = float(cfg.get("degenerate_tss_fraction", 0.5))
    source_root = Path(links.get("source_release_root") or data_root / "source_releases" / str(cfg.get("tcga_source_id")))
    rna_asset_root = Path(links.get("rna_asset_root") or data_root / "processed_assets" / "rna" / str(cfg.get("rna_asset_id")))
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if args.require_parsed_rna_asset and not rna_asset_root.exists():
        raise FileNotFoundError(rna_asset_root)

    rna_inv_path = source_root / "inventory" / "candidate_rna_files.parquet"
    slide_inv_path = source_root / "inventory" / "candidate_slide_files.parquet"
    if not rna_inv_path.exists():
        raise FileNotFoundError(rna_inv_path)
    if not slide_inv_path.exists():
        raise FileNotFoundError(slide_inv_path)
    rna_df = pd.read_parquet(rna_inv_path)
    slide_df = pd.read_parquet(slide_inv_path)
    for name, df, key in [("candidate_rna_files", rna_df, "rna_file_uuid"), ("candidate_slide_files", slide_df, "slide_file_uuid")]:
        if key not in df.columns:
            raise RuntimeError(f"{name} missing required column {key}")
        if df[key].duplicated().any():
            dups = df.loc[df[key].duplicated(), key].astype(str).head(10).tolist()
            raise RuntimeError(f"{name} contains duplicate {key}: {dups}")

    included_projects = _load_included_projects(settings_dir)
    excluded_entities = _load_excluded_entities(source_root)
    version_ok = _load_version_ok(source_root)
    usable_rna_uuids = _load_usable_rna_uuids(rna_asset_root) if args.require_parsed_rna_asset else None

    # 1. RNA eligibility -> one canonical RNA row per eligible case (full eligible set).
    eligible_rna, rna_audit, rna_excl = _select_rna(
        rna_df,
        included_projects,
        excluded_entities,
        version_ok,
        usable_rna_uuids,
        args.benchmark_id,
        require_parsed_rna_asset=args.require_parsed_rna_asset,
    )
    # 2. Resolve the slides of all eligible cases (with the per-case slide cap), before the case cap
    slide_manifest, slide_audit, case_manifest_full, slide_excl = _resolve_slides(
        eligible_rna,
        slide_df,
        included_projects=included_projects,
        excluded_entities=excluded_entities,
        benchmark_id=args.benchmark_id,
        modelgroup_id=modelgroup_id,
        max_slides_per_case=max_slides_per_case,
    )
    # Full-eligible-set viability diagnostics (reported separately from the cohort below).
    n_eligible_rna_cases = int(case_manifest_full["case_id"].nunique()) if not case_manifest_full.empty else 0
    n_eligible_without_official = int((~case_manifest_full["official_primary_provisional"].map(_truthy)).sum()) if not case_manifest_full.empty else 0
    # 3. Viable cases: >= 1 official-primary slide
    viable_cases = set(
        case_manifest_full.loc[case_manifest_full["official_primary_provisional"].map(_truthy), "case_id"].astype(str)
    ) if not case_manifest_full.empty else set()
    viable_rna = eligible_rna.loc[eligible_rna["case_id"].astype(str).isin(viable_cases)].copy()
    # 4. Per-project case cap on the viable cases (download cohort: ceil(target * (1 + frac)) per project)
    selected_rna, cap_excl, case_cap_summary = _cap_cases_per_project(
        viable_rna,
        target_cap=max_cases_per_project,
        overprovision_frac=case_cap_overprovision_frac,
        seed=case_cap_seed,
        method=case_cap_method,
        benchmark_id=args.benchmark_id,
    )
    selected_rna = _ordered(selected_rna, RNA_MANIFEST_COLUMNS)
    # 5. Subset slide_manifest / provisional case_manifest to the capped download cohort.
    kept_cases = set(selected_rna["case_id"].astype(str)) if not selected_rna.empty else set()
    if not slide_manifest.empty:
        slide_manifest = slide_manifest.loc[slide_manifest["case_id"].astype(str).isin(kept_cases)].copy()
    case_manifest = case_manifest_full.loc[case_manifest_full["case_id"].astype(str).isin(kept_cases)].copy() if not case_manifest_full.empty else case_manifest_full
    # 6. Projected fold grouping per project (warning on a case-level fallback)
    projected_fold_structure: List[Dict[str, Any]] = []
    if not selected_rna.empty:
        for project_id, g in selected_rna.groupby("project_id", sort=True):
            case_ids_p = g["case_id"].astype(str).tolist()
            tss_counts = project_tss_distribution(case_ids_p)
            grouping, max_tss_fraction = projected_fold_verdict(
                len(case_ids_p), tss_counts, n_folds=n_folds_cfg, degenerate_frac=degenerate_tss_fraction_cfg
            )
            projected_fold_structure.append({
                "project_id": str(project_id),
                "n_cases_download_cohort": int(len(case_ids_p)),
                "n_tss": int(len(tss_counts)),
                "max_tss_fraction": float(max_tss_fraction),
                "projected_fold_grouping_used": grouping,
            })
            if grouping != "tissue_source_site":
                logger.warning(
                    "Projected fold fallback for project %s: %s (n_cases=%d, n_tss=%d, "
                    "max_tss_fraction=%.3f) on the download cohort (folds are built by _13).",
                    project_id, grouping, len(case_ids_p), len(tss_counts), max_tss_fraction,
                )
    if slide_manifest.empty:
        n_selected_cases = int(selected_rna["case_id"].nunique()) if not selected_rna.empty else 0
        n_audit_pairs = int(len(slide_audit))
        if slide_audit.empty:
            diag = f"no candidate slides for any of the {n_selected_cases} selected RNA case(s)"
        else:
            eligible_mask = slide_audit["metadata_eligibility_status"].eq("eligible")
            n_eligible = int(eligible_mask.sum())
            if n_eligible == 0:
                exploded = slide_audit.loc[~eligible_mask, "metadata_exclusion_reason"].fillna("").str.split(";").explode().str.strip()
                top_reasons = exploded.loc[exploded.ne("")].value_counts().head(5).to_dict()
                diag = (
                    f"all {n_audit_pairs} slide-case pair(s) for {n_selected_cases} selected RNA case(s) failed slide eligibility; "
                    f"top exclusion reasons: {top_reasons}"
                )
            else:
                level_counts = slide_audit.loc[eligible_mask, "biospecimen_match_level"].value_counts().to_dict()
                diag = (
                    f"{n_eligible} of {n_audit_pairs} slide-case pair(s) passed eligibility but none matched at an official tier; "
                    f"match-level breakdown: {level_counts}"
                )
        raise RuntimeError(
            "No official-primary provisional slides were selected at any biospecimen tier (0/1/2). "
            f"Diagnostic: {diag}. "
            "Do not relax matching further inside this benchmark_id; create a new benchmark_id if a looser policy is needed."
        )

    cap_excl_df = pd.DataFrame(cap_excl)
    excl_frames = [df for df in (rna_excl, cap_excl_df, slide_excl) if not df.empty]
    exclusion_log = (
        pd.concat(excl_frames, ignore_index=True)
        if excl_frames
        else pd.DataFrame(columns=["stage", "entity_type", "entity_id", "case_id", "case_uuid", "project_id", "reason", "detail", "benchmark_id"])
    )
    if not exclusion_log.empty:
        exclusion_log = exclusion_log.sort_values(["stage", "project_id", "case_id", "entity_type", "entity_id"], na_position="last").reset_index(drop=True)

    write_csv_atomic(cohort_dir / "rna_manifest.csv", selected_rna, overwrite=args.force)
    write_csv_atomic(cohort_dir / "rna_resolution_audit.csv", rna_audit, overwrite=args.force)
    write_csv_atomic(cohort_dir / "provisional_case_manifest.csv", case_manifest, overwrite=args.force)
    write_csv_atomic(cohort_dir / "slide_manifest.csv", slide_manifest, overwrite=args.force)
    write_csv_atomic(cohort_dir / "slide_resolution_audit.csv", slide_audit, overwrite=args.force)
    write_csv_atomic(cohort_dir / "exclusion_log.csv", exclusion_log, overwrite=args.force)

    n_exact_cases = int(case_manifest.loc[case_manifest["biospecimen_match_level_best_case"].eq("exact_same_portion"), "case_id"].nunique()) if not case_manifest.empty else 0
    n_same_sample_cases = int(case_manifest.loc[case_manifest["biospecimen_match_level_best_case"].eq("same_sample"), "case_id"].nunique()) if not case_manifest.empty else 0
    n_same_case_cases = int(case_manifest.loc[case_manifest["biospecimen_match_level_best_case"].isin(["same_case_primary_tumor", "same_case_primary_tumor_unknown_sublevel"]), "case_id"].nunique()) if not case_manifest.empty else 0
    summary = {
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": modelgroup_id,
        "n_candidate_rna_files": int(len(rna_df)),
        # Full eligible set, before the cap
        "n_eligible_rna_cases": int(n_eligible_rna_cases),
        "n_cases_without_official_primary_slide": int(n_eligible_without_official),
        "n_viable_cases": int(n_eligible_rna_cases - n_eligible_without_official),
        # Cohort selected for download (post viable-filter + per-project cap + over-provision).
        "n_selected_rna_cases": int(selected_rna["case_id"].nunique()) if not selected_rna.empty else 0,
        "case_cap_applied": case_cap_summary.get("case_cap_applied"),
        "max_cases_per_project": case_cap_summary.get("max_cases_per_project"),
        "per_project_target_cap": case_cap_summary.get("per_project_target_cap"),
        "per_project_download_cap": case_cap_summary.get("per_project_download_cap"),
        "case_cap_overprovision_frac": case_cap_summary.get("case_cap_overprovision_frac"),
        "case_cap_method": case_cap_summary.get("case_cap_method"),
        "max_slides_per_case": (int(max_slides_per_case) if max_slides_per_case is not None else None),
        "n_viable_cases_before_cap": case_cap_summary.get("n_cases_before_cap"),
        "n_cases_selected_for_download": case_cap_summary.get("n_cases_selected_for_download"),
        "n_cases_dropped_by_cap": case_cap_summary.get("n_cases_dropped_by_cap"),
        "n_candidate_slide_files": int(len(slide_df)),
        "n_slide_candidates_for_eligible_rna_cases": int(len(slide_audit)),
        "n_metadata_eligible_slide_case_pairs": int((slide_audit.get("metadata_eligibility_status", pd.Series(dtype=str)) == "eligible").sum()) if not slide_audit.empty else 0,
        "n_provisional_official_primary_slides": int(len(slide_manifest)),
        "expected_n_official_primary_before_tile_qc": int(case_manifest["official_primary_provisional"].map(_truthy).sum()) if not case_manifest.empty else 0,
        "n_exact_same_portion_cases": n_exact_cases,
        "n_same_sample_cases": n_same_sample_cases,
        "n_same_case_primary_tumor_cases": n_same_case_cases,
        "n_cases_with_ambiguous_or_unresolved_rna_parent": int((rna_audit.get("parent_resolution_status", pd.Series(dtype=str)).astype(str).isin(["ambiguous", "failed"])).sum()) if not rna_audit.empty else 0,
        "require_parsed_rna_asset": bool(args.require_parsed_rna_asset),
        "rna_asset_stage": "deferred_until_after_slide_tile_filtering" if not args.require_parsed_rna_asset else "prebuilt_required_legacy_mode",
        "metadata_only_invariant": "No parsed RNA TSV, OpenSlide, MPP, tile-read, download, or HTTP range prefetch checks were applied unless --require_parsed_rna_asset was set.",
    }

    manifest = manifest_header(Path(__file__).name, vars(args))
    manifest.update({
        "benchmark_id": args.benchmark_id,
        "benchmark_root": str(benchmark_root),
        "source_root": str(source_root),
        "rna_asset_root": str(rna_asset_root),
        "modelgroup_id": modelgroup_id,
        "included_projects": sorted(included_projects),
        "official_primary_biospecimen_match_levels": sorted(OFFICIAL_MATCH_LEVELS),
        "exploratory_only_biospecimen_match_levels": sorted(EXPLORATORY_MATCH_LEVELS),
        "multi_slide_policy_official_v4": "per case, pick the tightest available biospecimen tier (Tier 0 exact_same_portion > Tier 1 same_sample > Tier 2 same_case_primary_tumor[_unknown_sublevel]); never mix tiers within a case. Within that tier keep the top max_slides_per_case slides by the slide_selection_rank ordering below (default 1 slide/case), with a native-20x resolution preference (native_20x_then_finer_then_unknown_then_sub20x) as a GDC-magnification tie-break before file_size. The resolution gate is the OpenSlide-MPP no-upsampling check in _7/_10. Tier 2 is admitted as official primary; Tier 0/1 are preferred when they exist.",
        "max_slides_per_case": (int(max_slides_per_case) if max_slides_per_case is not None else None),
        "slide_selection_rank": ["biospecimen_level_priority", "he_stain_known_first", "ffpe_known_first", "native_20x_resolution_first", "file_size_desc", "slide_file_uuid_asc"],
        "rna_tie_break": ["workflow_rank", "workflow_version_desc", "file_state_rank", "data_release_desc", "version_desc", "updated_datetime_desc", "file_name_asc", "rna_file_uuid_asc"],
        "projected_fold_structure": {
            "note": "Pre-download projection of the per-project TSS-grouped fold outcome on the download cohort; the folds are built by _13_build_targets_and_folds.py on the final trimmed cohort.",
            "n_folds": int(n_folds_cfg),
            "degenerate_tss_fraction": float(degenerate_tss_fraction_cfg),
            "per_project": projected_fold_structure,
        },
        "metadata_eligibility_policies": {
            "rna": "retained project, primary tumor, STAR Counts, released/open, primary disease when available, annotation-clean, resolved parent, version provenance; parsed RNA asset deferred to _12_build_rna_asset.py",
            "slide": "retained project, Diagnostic Slide, released/open, primary tumor when available, H&E-compatible or allowed unknown stain, FFPE true or allowed missing, annotation-clean, usable parent; no slide file-version check (snapshot state/access used instead)",
            "unknown_stain_policy": cfg.get("unknown_stain_policy"),
            "unknown_ffpe_policy": cfg.get("unknown_ffpe_policy"),
        },
        "case_cap_policy": case_cap_summary,
        "summary": summary,
        "input_hashes": {
            "benchmark_config": sha256_path(settings_dir / "benchmark_config.yaml"),
            "benchmark_asset_links": sha256_path(settings_dir / "benchmark_asset_links.json"),
            "candidate_rna_files": sha256_path(rna_inv_path),
            "candidate_slide_files": sha256_path(slide_inv_path),
            "file_versions": sha256_path(source_root / "inventory" / "file_versions.parquet"),
            "annotation_exclusion_log": sha256_path(source_root / "inventory" / "annotation_exclusion_log.csv") if (source_root / "inventory" / "annotation_exclusion_log.csv").exists() else None,
            "case_expression_log10_fpkmuq": sha256_path(rna_asset_root / "case_expression_log10_fpkmuq.parquet") if args.require_parsed_rna_asset else None,
        },
        "output_hashes": collect_output_hashes(cohort_dir),
    })
    write_json_atomic(cohort_dir / "rna_slide_resolution_manifest.json", manifest, overwrite=args.force)
    _write_report(reports_dir / "rna_slide_resolution_report.md", summary, overwrite=args.force)

    print(json.dumps({"status": "ok", **summary, "cohort_dir": str(cohort_dir)}, indent=2))
    return 0


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Resolve one RNA file per TCGA case and the provisional official-primary slides.")

    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help="Benchmark ID whose provisional cohort is resolved.")

    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing resolved cohort.")

    ap.add_argument("--force_cascade", action="store_true",
                    help="Required with --force if a selected slide cache or tile assets already exist.")

    ap.add_argument("--require_parsed_rna_asset", action="store_true",
                    help="Require the RNA asset (built by _12_build_rna_asset.py) to exist and contain "
                         "each selected RNA UUID.")

    ap.add_argument("--max_cases_per_project", type=int, default=None,
                    help="Max cases per TCGA project (default: benchmark_config.yaml; <=0: no cap).")

    ap.add_argument("--case_cap_seed", type=int, default=None,
                    help="Seed of the per-project case-cap subsampling (default: benchmark_config.yaml).")

    ap.add_argument("--case_cap_method", type=str, default=None,
                    choices=["tss_stratified_round_robin", "random"],
                    help="Case-cap selection method: 'tss_stratified_round_robin' interleaves tissue-source sites, "
                         "'random' is a seeded subsample (default: benchmark_config.yaml).")

    ap.add_argument("--case_cap_overprovision_frac", type=float, default=None,
                    help="Over-provision fraction for the case cap: ceil(max_cases_per_project * (1 + this)) cases "
                         "per project are selected, then trimmed by _11_build_case_tile_index.py "
                         "(default: benchmark_config.yaml; 0: disabled).")

    ap.add_argument("--max_slides_per_case", type=int, default=None,
                    help="Max slides per case at the case's best biospecimen tier "
                         "(default: benchmark_config.yaml; <=0: all).")

    args = ap.parse_args()
    raise SystemExit(main(args))
