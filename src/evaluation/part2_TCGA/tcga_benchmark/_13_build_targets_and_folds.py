#!/usr/bin/env python3
"""
Build the Task B RNA signature targets (per-project z-scored signature scores of the final cases) and the
project-specific CV folds (grouped by tissue-source site, or by case if there are too few sites or one dominates).

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data / benchmarks / <benchmark_id> /
    - shared/rna/target_tables/<project_id>.parquet, <project_id>.manifest.json
    - shared/rna/signature_eligibility.csv, shared/rna/target_and_folds_manifest.json
    - shared/folds/fold_assignments.csv
    - reports/targets_and_folds_report.md
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
    collect_output_hashes,
    manifest_header,
    project_tss_distribution,
    projected_fold_verdict,
    read_json,
    read_yaml,
    sha256_path,
    stable_project_seed,
    tissue_source_site,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)

TARGET_META_COLUMNS = ["case_id", "case_uuid", "project_id"]
FOLD_ASSIGNMENT_COLUMNS = ["case_id", "case_uuid", "project_id", "fold_id", "fold_seed", "benchmark_id"]

SIGNATURE_ELIGIBILITY_COLUMNS = [
    "project_id",
    "signature_name",
    "eligible",
    "eligibility_reason",
    "official_project",
    "domain_slice",
    "support_category",
    "primary_score_cell",
    "n_final_cases",
    "n_cases_with_target",
    "n_signature_genes",
    "n_genes_available_project",
    "n_genes_used",
    "coverage_fraction",
    "target_variance",
    "target_std",
    "target_min",
    "target_max",
    "fold_level_ridge_feasible",
    "fold_feasibility_reason",
    "score_method",
    "denominator_rule",
    "signature_family",
    "signature_source_name",
    "signature_source_version",
    "signature_set_id",
    "signature_set_hash",
    "target_table_hash",
    "final_case_manifest_hash",
    "fold_assignment_hash",
    "rna_asset_id",
    "rna_asset_hash",
    "alias_map_hash",
    "benchmark_id",
    "modelgroup_id",
    "fold_seed",
    "n_folds",
]

REPORT_NAME = "targets_and_folds_report.md"


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


def _ordered(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = pd.NA
    extras = [c for c in out.columns if c not in columns]
    return out.loc[:, list(columns) + extras]


def _require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def _project_domain_map(settings_dir: Path) -> Dict[str, str]:
    path = settings_dir / "project_domain_map.json"
    if not path.exists():
        raise FileNotFoundError(path)
    obj = read_json(path)
    labels = obj.get("project_domain_map") or obj.get("projects") or {}
    if not isinstance(labels, Mapping) or not labels:
        raise RuntimeError("project_domain_map.json contains no project labels")
    return {str(k): str(v) for k, v in labels.items()}


def _official_projects(project_summary: Mapping[str, Any]) -> List[str]:
    rows = project_summary.get("project_rows")
    if not isinstance(rows, list):
        raise RuntimeError("project_summary.json missing project_rows")
    return sorted(str(r["project_id"]) for r in rows if bool(r.get("official_project")))


def _signature_rows(signature_set: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    sigs = signature_set.get("signatures")
    if not isinstance(sigs, list) or not sigs:
        raise RuntimeError("signature_set.signatures must be a non-empty list")
    out: List[Mapping[str, Any]] = []
    seen = set()
    for sig in sigs:
        if not isinstance(sig, Mapping):
            raise RuntimeError("signature_set.signatures contains a non-object entry")
        name = str(sig.get("signature_name") or "").strip()
        if not name:
            raise RuntimeError("signature without signature_name")
        if name in seen:
            raise RuntimeError(f"Duplicate signature_name: {name}")
        seen.add(name)
        if sig.get("denominator_rule") != "available_resolved_abs_weight_sum":
            raise RuntimeError(f"Signature {name} has invalid denominator_rule={sig.get('denominator_rule')!r}")
        weights = sig.get("gene_weights")
        genes = sig.get("genes_resolved")
        if not isinstance(weights, Mapping) or not isinstance(genes, list) or not genes:
            raise RuntimeError(f"Signature {name} has invalid genes_resolved/gene_weights")
        if set(map(str, genes)) != set(map(str, weights.keys())):
            raise RuntimeError(f"Signature {name} genes_resolved and gene_weights keys disagree")
        out.append(sig)
    return out


def _support_by_signature(support_obj: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    if support_obj.get("frozen_final") is not True and support_obj.get("status") != "frozen_final":
        raise RuntimeError("signature_support_training.json is not frozen_final")
    by = support_obj.get("signature_support_by_name") or support_obj.get("signatures_by_name")
    if isinstance(by, Mapping):
        return {str(k): v for k, v in by.items() if isinstance(v, Mapping)}
    rows = support_obj.get("signatures") or support_obj.get("signature_support") or []
    if isinstance(rows, list):
        return {str(r.get("signature_name")): r for r in rows if isinstance(r, Mapping) and r.get("signature_name")}
    return {}


# Per-project RNG seed (default namespace "fold_seed")
_stable_project_seed = stable_project_seed


# TCGA tissue-source-site code (shared with _14_validate_benchmark.py)
_tissue_source_site = tissue_source_site


def _case_level_fold_ids(n: int, rng: np.random.Generator, n_folds: int) -> np.ndarray:
    perm = rng.permutation(n)
    fold_ids = np.empty(n, dtype=int)
    for rank, idx in enumerate(perm):
        fold_ids[int(idx)] = int(rank % n_folds)
    return fold_ids


def _tss_grouped_fold_ids(case_ids: Sequence[str], rng: np.random.Generator, n_folds: int) -> np.ndarray:
    """Assign whole tissue-source-site groups to folds (greedy: largest group into the smallest fold)."""
    tss = [_tissue_source_site(c) for c in case_ids]
    groups: Dict[str, List[int]] = {}
    for i, t in enumerate(tss):
        groups.setdefault(t, []).append(i)
    group_names = list(groups)
    perm = rng.permutation(len(group_names))
    shuffled = [group_names[int(j)] for j in perm]
    # Stable sort by descending size; equal-size ties keep the shuffled order.
    shuffled.sort(key=lambda t: len(groups[t]), reverse=True)
    fold_sizes = [0] * n_folds
    fold_ids = np.empty(len(case_ids), dtype=int)
    for t in shuffled:
        target = min(range(n_folds), key=lambda f: (fold_sizes[f], f))
        for i in groups[t]:
            fold_ids[i] = target
        fold_sizes[target] += len(groups[t])
    return fold_ids


def _assign_project_folds(
    cases: pd.DataFrame, *, n_folds: int, fold_seed: int, benchmark_id: str, fold_grouping: str, degenerate_tss_fraction: float
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    _require_columns(cases, ["case_id", "case_uuid", "project_id"], "cases for fold assignment")
    rows: List[Dict[str, Any]] = []
    per_project: Dict[str, Dict[str, Any]] = {}
    for project_id, g in cases.groupby("project_id", sort=True):
        g = g.sort_values(["case_id", "case_uuid"]).reset_index(drop=True)
        n = len(g)
        if n < n_folds:
            raise RuntimeError(f"Project {project_id} has only {n} final cases, fewer than n_folds={n_folds}")
        rng = np.random.default_rng(_stable_project_seed(fold_seed, str(project_id)))
        case_ids = g["case_id"].astype(str).tolist()
        tss_counts = project_tss_distribution(case_ids)
        n_tss = len(tss_counts)
        if fold_grouping == "tissue_source_site":
                # Site-pure folds unless there are too few sites or one site dominates (as projected by _6)
            grouping_used, max_tss_fraction = projected_fold_verdict(
                n, tss_counts, n_folds=n_folds, degenerate_frac=degenerate_tss_fraction
            )
            if grouping_used == "tissue_source_site":
                fold_ids = _tss_grouped_fold_ids(case_ids, rng, n_folds)
                reason = "site-pure folds feasible; no single tissue-source-site exceeds the degeneracy threshold"
            elif grouping_used == "case_fallback_insufficient_tss":
                fold_ids = _case_level_fold_ids(n, rng, n_folds)
                reason = f"only {n_tss} distinct tissue-source-sites (< n_folds={n_folds}); using balanced case-level folds"
            else:  # case_fallback_degenerate_dominant_tss
                fold_ids = _case_level_fold_ids(n, rng, n_folds)
                reason = (
                    f"dominant site is {max_tss_fraction:.3f} of cases (> degenerate_tss_fraction="
                    f"{degenerate_tss_fraction}); using balanced case-level folds"
                )
        else:
            fold_ids = _case_level_fold_ids(n, rng, n_folds)
            grouping_used = "case"
            max_tss_fraction = (max(tss_counts.values()) / n) if (n > 0 and tss_counts) else 0.0
            reason = "fold_grouping='case' (explicit case-level folds requested in config)"
        fold_sizes = [int(x) for x in np.bincount(np.asarray(fold_ids, dtype=int), minlength=n_folds)]
        per_project[str(project_id)] = {
            "fold_grouping_used": grouping_used,
            "fold_grouping_reason": reason,
            "n_cases": int(n),
            "n_tissue_source_sites": int(n_tss),
            "max_tss_fraction": float(max_tss_fraction),
            "fold_sizes": fold_sizes,
        }
        for i, r in g.iterrows():
            rows.append({
                "case_id": str(r["case_id"]),
                "case_uuid": str(r["case_uuid"]),
                "project_id": str(project_id),
                "fold_id": int(fold_ids[int(i)]),
                "fold_seed": int(fold_seed),
                "benchmark_id": benchmark_id,
            })
    out = pd.DataFrame(rows).sort_values(["project_id", "fold_id", "case_id"]).reset_index(drop=True)
    return out.loc[:, FOLD_ASSIGNMENT_COLUMNS], per_project


def _numeric_matrix(df: pd.DataFrame, gene_cols: Sequence[str]) -> pd.DataFrame:
    if not gene_cols:
        raise RuntimeError("No gene columns available in RNA expression table")
    out = df.loc[:, list(gene_cols)].apply(pd.to_numeric, errors="coerce")
    return out


def _available_project_genes(expr: pd.DataFrame, *, near_zero_gene_std: float) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], Dict[str, Any]]:
    """Z-score the genes that are finite in all cases of the project and not near-constant.
    Returns (z, means, stds, kept genes, meta).
    """
    finite_all = np.isfinite(expr.to_numpy(dtype=float)).all(axis=0)
    finite_cols = [c for c, keep in zip(expr.columns, finite_all) if bool(keep)]
    finite = expr.loc[:, finite_cols]
    means = finite.mean(axis=0)
    stds = finite.std(axis=0, ddof=0)
    keep = stds > float(near_zero_gene_std)
    kept_cols = [str(c) for c in finite.columns[keep.to_numpy()]]
    if not kept_cols:
        raise RuntimeError("All RNA genes were removed by missingness/nonzero-variance filters")
    z = (finite.loc[:, kept_cols] - means.loc[kept_cols]) / stds.loc[kept_cols]
    meta = {
        "n_gene_columns_total": int(expr.shape[1]),
        "n_genes_finite_all_cases": int(len(finite_cols)),
        "n_genes_removed_for_any_missing_case": int(expr.shape[1] - len(finite_cols)),
        "n_genes_removed_near_zero_variance": int(len(finite_cols) - len(kept_cols)),
        "near_zero_gene_std_threshold": float(near_zero_gene_std),
        "std_ddof": 0,
    }
    return z, means.loc[kept_cols], stds.loc[kept_cols], kept_cols, meta


def _score_signature(z: pd.DataFrame, sig: Mapping[str, Any]) -> Tuple[np.ndarray, int, float, List[str]]:
    weights_map = {str(k): float(v) for k, v in dict(sig["gene_weights"]).items()}
    used = sorted([g for g in weights_map if g in z.columns])
    if not used:
        return np.full(len(z), np.nan, dtype=float), 0, 0.0, []
    weights = np.asarray([weights_map[g] for g in used], dtype=np.float64)
    denom = float(np.sum(np.abs(weights)))
    if not math.isfinite(denom) or denom <= 0.0:
        raise RuntimeError(f"Invalid denominator for signature {sig.get('signature_name')}")
    mat = z.loc[:, used].to_numpy(dtype=np.float64)
    values = mat @ weights / denom
    return values.astype(np.float64), int(len(used)), denom, used


def _fold_feasibility(y: pd.Series, folds: pd.DataFrame, *, n_folds: int, inner_cv_folds: int, var_min: float) -> Tuple[bool, str]:
    merged = pd.DataFrame({"case_id": y.index.astype(str), "y": y.to_numpy(dtype=float)}).merge(
        folds[["case_id", "fold_id"]], on="case_id", how="left", validate="one_to_one"
    )
    if merged["fold_id"].isna().any():
        return False, "missing_fold_assignment"
    if not np.isfinite(merged["y"].to_numpy(dtype=float)).all():
        return False, "nonfinite_target"
    reasons: List[str] = []
    for fold_id in range(n_folds):
        test = merged.loc[merged["fold_id"].astype(int).eq(fold_id), "y"].to_numpy(dtype=float)
        train = merged.loc[~merged["fold_id"].astype(int).eq(fold_id), "y"].to_numpy(dtype=float)
        if len(test) < 1:
            reasons.append(f"fold_{fold_id}_empty_test")
        if len(train) < inner_cv_folds:
            reasons.append(f"fold_{fold_id}_train_lt_inner_cv")
        if float(np.var(train, ddof=0)) <= var_min:
            reasons.append(f"fold_{fold_id}_train_target_variance_le_threshold")
    return (len(reasons) == 0), ";".join(reasons)


def _update_asset_links(settings_dir: Path, update: Mapping[str, Any], *, force: bool, force_cascade: bool) -> Dict[str, Any]:
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
        raise RuntimeError(f"Refusing to change benchmark links without --force --force_cascade: {conflicts}")
    merged = dict(old_dict)
    merged.update(update)
    old_hash = sha256_path(path)
    write_json_atomic(path, merged, overwrite=True)
    return {"path": str(path), "previous_sha256": old_hash, "new_sha256": sha256_path(path), "updated_keys": sorted(update)}


def _write_report(
    path: Path,
    summary: Mapping[str, Any],
    *,
    per_project_fold_grouping: Optional[Mapping[str, Mapping[str, Any]]] = None,
    overwrite: bool,
) -> None:
    lines = [
        "# Task B targets and folds report",
        "",
        f"- benchmark_id: `{summary.get('benchmark_id')}`",
        f"- modelgroup_id: `{summary.get('modelgroup_id')}`",
        f"- n official projects: `{summary.get('n_official_projects')}`",
        f"- n target tables: `{summary.get('n_target_tables')}`",
        f"- n fold-assigned cases: `{summary.get('n_fold_rows')}`",
        f"- fold grouping (requested): `{summary.get('fold_grouping')}`",
        f"- n eligible project × signature cells: `{summary.get('n_eligible_cells')}`",
        f"- n primary score cells: `{summary.get('n_primary_score_cells')}`",
        "",
    ]
    if per_project_fold_grouping:
        # Fold grouping used per project
        lines += [
            "## Per-project fold grouping used",
            "",
            "| project | grouping_used | n_cases | n_tss | max_tss_fraction | fold_sizes | reason |",
            "|---|---|---|---|---|---|---|",
        ]
        for project_id in sorted(per_project_fold_grouping):
            v = per_project_fold_grouping[project_id]
            lines.append(
                f"| `{project_id}` | `{v.get('fold_grouping_used')}` | {v.get('n_cases')} | "
                f"{v.get('n_tissue_source_sites')} | {float(v.get('max_tss_fraction', 0.0)):.3f} | "
                f"{v.get('fold_sizes')} | {v.get('fold_grouping_reason')} |"
            )
        lines.append("")
    lines += [
        "Targets are project-level gene-z-score signature labels computed once from the final benchmark cohort.",
        "RNA z-scores are not recomputed inside the ridge CV.",
        "",
    ]
    write_text_atomic(path, "\n".join(lines), overwrite=overwrite)


def main(args) -> int:

    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    settings_dir = benchmark_root / "settings"
    cohort_dir = benchmark_root / "shared" / "cohort"
    shared_rna_dir = benchmark_root / "shared" / "rna"
    target_dir = shared_rna_dir / "target_tables"
    folds_dir = benchmark_root / "shared" / "folds"
    reports_dir = benchmark_root / "reports"
    if not settings_dir.exists():
        raise FileNotFoundError(settings_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        shared_rna_dir / "signature_eligibility.csv",
        shared_rna_dir / "target_and_folds_manifest.json",
        folds_dir / "fold_assignments.csv",
    ]
    if (any(p.exists() for p in outputs) or any(target_dir.glob("*.parquet")) or any(target_dir.glob("*.manifest.json"))) and not args.force:
        raise FileExistsError("Target/fold outputs already exist. Use --force to rebuild.")
    downstream = [benchmark_root / "shared" / "features", benchmark_root / "reports" / "benchmark_validation_report.json"]
    if args.force and any(p.exists() for p in downstream) and not args.force_cascade:
        raise RuntimeError(f"Refusing to rebuild targets/folds while downstream artifacts exist: {downstream}. Use --force_cascade if intentional.")
    if args.force:
        # Remove the previous per-project target tables and manifests
        for old_target in list(target_dir.glob("*.parquet")) + list(target_dir.glob("*.manifest.json")):
            old_target.unlink()

    cfg = _load_mapping(settings_dir / "benchmark_config.yaml")
    links = _load_mapping(settings_dir / "benchmark_asset_links.json")
    modelgroup_id = str(cfg.get("modelgroup_id") or links.get("modelgroup_id"))
    rna_asset_id = str(cfg.get("rna_asset_id") or links.get("rna_asset_id"))
    fold_seed = int(args.fold_seed if args.fold_seed is not None else cfg.get("fold_seed", 20260518))
    n_folds = int(args.n_folds if args.n_folds is not None else cfg.get("fold_count", 5))
    # Minimum number of ridge inner CV folds (fold feasibility check)
    inner_cv_folds = int(cfg["ridge_inner_cv_folds_min"])
    fold_grouping = str(args.fold_grouping if args.fold_grouping is not None else cfg.get("fold_grouping", "tissue_source_site"))
    if fold_grouping not in {"tissue_source_site", "case"}:
        raise RuntimeError(f"fold_grouping must be 'tissue_source_site' or 'case', got {fold_grouping!r}")
    # Dominant-site degeneracy threshold for TSS-grouped folds (see _assign_project_folds).
    degenerate_tss_fraction = float(args.degenerate_tss_fraction if args.degenerate_tss_fraction is not None else cfg.get("degenerate_tss_fraction", 0.5))
    if not (0.0 < degenerate_tss_fraction <= 1.0):
        raise RuntimeError(f"degenerate_tss_fraction must be in (0, 1], got {degenerate_tss_fraction!r}")
    if n_folds < 2:
        raise RuntimeError("n_folds must be >= 2")

    alias_map_hash = str(links.get("alias_map_sha256") or links.get("alias_map_hash") or "")
    if not alias_map_hash:
        raise RuntimeError("benchmark_asset_links.json missing alias_map_sha256")

    rna_asset_root = Path(str(links.get("rna_asset_root") or data_root / "processed_assets" / "rna" / rna_asset_id))
    rna_manifest_path = rna_asset_root / "rna_asset_manifest.json"
    expr_path = rna_asset_root / "case_expression_log10_fpkmuq.parquet"
    if not rna_manifest_path.exists():
        raise FileNotFoundError(rna_manifest_path)
    if not expr_path.exists():
        raise FileNotFoundError(expr_path)
    rna_manifest = read_json(rna_manifest_path)
    if rna_manifest.get("alias_map_sha256") != alias_map_hash:
        raise RuntimeError("RNA asset alias_map_sha256 does not match benchmark alias map")

    signature_set_path = settings_dir / "signature_set_tcga_case_v1.json"
    signature_support_path = settings_dir / "signature_support_training.json"
    signature_set = read_json(signature_set_path)
    if signature_set.get("frozen_final") is not True and signature_set.get("status") != "frozen_final":
        raise RuntimeError("signature_set_tcga_case_v1.json is not frozen_final")
    if signature_set.get("alias_map_sha256") != alias_map_hash:
        raise RuntimeError("Signature set alias_map_sha256 does not match benchmark alias map")
    if signature_set.get("denominator_rule") != "available_resolved_abs_weight_sum":
        raise RuntimeError("Signature set denominator_rule must be available_resolved_abs_weight_sum")
    sig_rows = _signature_rows(signature_set)
    sig_names = [str(s["signature_name"]) for s in sig_rows]
    coverage_threshold = float(args.coverage_threshold if args.coverage_threshold is not None else signature_set.get("coverage_threshold", 0.70))
    minimum_n_genes_used = int(args.minimum_n_genes_used if args.minimum_n_genes_used is not None else signature_set.get("minimum_n_genes_used", 10))

    support = read_json(signature_support_path)
    support_by_name = _support_by_signature(support)
    domain_map = _project_domain_map(settings_dir)

    final_case_path = cohort_dir / "final_case_manifest.csv"
    selected_rna_path = cohort_dir / "rna_manifest.csv"
    project_summary_path = cohort_dir / "project_summary.json"
    if not final_case_path.exists():
        raise FileNotFoundError(final_case_path)
    if not selected_rna_path.exists():
        raise FileNotFoundError(selected_rna_path)
    if not project_summary_path.exists():
        raise FileNotFoundError(project_summary_path)
    final_case = pd.read_csv(final_case_path, dtype=str, keep_default_na=False)
    _require_columns(final_case, ["case_id", "case_uuid", "project_id", "final_case_eligible", "official_primary_match"], "final_case_manifest.csv")
    final_eligible = final_case.loc[final_case["final_case_eligible"].map(_truthy)].copy()
    if final_eligible.empty:
        raise RuntimeError("No final eligible cases")
    if final_eligible["case_id"].duplicated().any():
        raise RuntimeError("Duplicate final eligible case_id values")
    if not final_eligible["official_primary_match"].map(_truthy).all():
        raise RuntimeError("All final eligible cases must have official_primary_match == true")

    selected_rna = pd.read_csv(selected_rna_path, dtype=str, keep_default_na=False)
    _require_columns(
        selected_rna,
        ["case_id", "case_uuid", "project_id", "rna_file_uuid"],
        "shared/cohort/rna_manifest.csv",
    )
    if selected_rna["case_id"].duplicated().any():
        dups = selected_rna.loc[selected_rna["case_id"].duplicated(), "case_id"].head(20).tolist()
        raise RuntimeError(f"rna_manifest.csv must contain one selected RNA row per case; duplicates={dups}")

    expected_selected = final_eligible[["case_id", "case_uuid", "project_id", "rna_file_uuid"]].copy()
    expected_selected = expected_selected.merge(
        selected_rna[["case_id", "case_uuid", "project_id", "rna_file_uuid"]],
        on=["case_id", "case_uuid", "project_id", "rna_file_uuid"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if not expected_selected["_merge"].eq("both").all():
        bad = expected_selected.loc[~expected_selected["_merge"].eq("both"), "case_id"].head(20).tolist()
        raise RuntimeError(f"Final eligible cases are not all present in selected rna_manifest.csv: {bad}")

    project_summary = read_json(project_summary_path)
    official_projects = _official_projects(project_summary)
    if not official_projects:
        raise RuntimeError("No official projects in project_summary.json")
    official_set = set(official_projects)
    final_eligible = final_eligible.loc[final_eligible["project_id"].astype(str).isin(official_set)].copy()
    if final_eligible.empty:
        raise RuntimeError("No final eligible cases in official projects")

    expression_all = pd.read_parquet(expr_path)
    _require_columns(expression_all, ["case_id", "case_uuid", "project_id", "rna_file_uuid"], "case_expression_log10_fpkmuq.parquet")
    if expression_all["rna_file_uuid"].duplicated().any():
        dups = expression_all.loc[expression_all["rna_file_uuid"].duplicated(), "rna_file_uuid"].head(20).tolist()
        raise RuntimeError(f"case_expression_log10_fpkmuq.parquet contains duplicate RNA file UUID rows: {dups}")

    gene_cols = sorted([c for c in expression_all.columns if c not in set(["case_id", "case_uuid", "project_id", "rna_file_uuid", "rna_file_name", "sample_id", "sample_submitter_id", "sample_type", "portion_id", "portion_submitter_id", "analyte_id", "analyte_submitter_id", "aliquot_id", "aliquot_submitter_id", "workflow_type", "workflow_version", "data_release", "version"])])
    if not gene_cols:
        raise RuntimeError("No gene columns found in expression table")

    selected_expr_keys = final_eligible[["case_id", "case_uuid", "project_id", "rna_file_uuid"]].copy()
    expression = selected_expr_keys.merge(
        expression_all,
        on=["case_id", "case_uuid", "project_id", "rna_file_uuid"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_expr = expression.loc[~expression["_merge"].eq("both"), "case_id"].head(20).tolist()
    if missing_expr:
        raise RuntimeError(f"Expression table is missing selected RNA files for final cases: {missing_expr}")
    expression = expression.drop(columns=["_merge"])
    if expression["case_id"].duplicated().any():
        raise RuntimeError("Internal error: selected expression table has duplicate case_id rows after RNA UUID filtering")

    fold_assignments, per_project_fold_grouping = _assign_project_folds(
        final_eligible[["case_id", "case_uuid", "project_id"]],
        n_folds=n_folds, fold_seed=fold_seed, benchmark_id=args.benchmark_id, fold_grouping=fold_grouping,
        degenerate_tss_fraction=degenerate_tss_fraction,
    )
    write_csv_atomic(folds_dir / "fold_assignments.csv", fold_assignments, overwrite=args.force)
    fold_hash = sha256_path(folds_dir / "fold_assignments.csv")

    eligibility_rows: List[Dict[str, Any]] = []
    project_gene_reports: Dict[str, Any] = {}
    target_hash_by_project: Dict[str, str] = {}
    table_paths: Dict[str, str] = {}
    project_manifest_payloads: Dict[str, Dict[str, Any]] = {}

    for project_id in tqdm(official_projects, desc="Building targets/folds", unit="project"):
        cases_p = final_eligible.loc[final_eligible["project_id"].astype(str).eq(project_id)].copy()
        if cases_p.empty:
            continue
        expr_p = cases_p[["case_id", "case_uuid", "project_id"]].merge(
            expression,
            on=["case_id", "case_uuid", "project_id"],
            how="left",
            validate="one_to_one",
        )
        if expr_p["rna_file_uuid"].isna().any():
            missing = expr_p.loc[expr_p["rna_file_uuid"].isna(), "case_id"].head(20).tolist()
            raise RuntimeError(f"Expression table missing final cases for {project_id}: {missing}")
        expr_num = _numeric_matrix(expr_p, gene_cols)
        z, gene_means, gene_stds, available_genes, gene_report = _available_project_genes(expr_num, near_zero_gene_std=float(args.near_zero_gene_std))
        project_gene_reports[project_id] = gene_report
        z.index = expr_p["case_id"].astype(str).to_numpy()

        target_table = expr_p[["case_id", "case_uuid", "project_id"]].copy()
        for sig in sig_rows:
            name = str(sig["signature_name"])
            values, n_used, denom, used_genes = _score_signature(z, sig)
            target_table[name] = values.astype(np.float32)

            y = pd.Series(values, index=expr_p["case_id"].astype(str), name=name)
            finite = np.isfinite(y.to_numpy(dtype=float))
            n_cases_with_target = int(finite.sum())
            var = float(np.var(y.to_numpy(dtype=float)[finite], ddof=0)) if n_cases_with_target > 0 else float("nan")
            std = float(math.sqrt(var)) if math.isfinite(var) and var >= 0 else float("nan")
            cov = float(n_used / max(1, len(sig.get("genes_resolved", []))))
            folds_p = fold_assignments.loc[fold_assignments["project_id"].astype(str).eq(project_id)].copy()
            fold_ok, fold_reason = _fold_feasibility(y, folds_p, n_folds=n_folds, inner_cv_folds=inner_cv_folds, var_min=float(args.target_variance_min))
            support_row = support_by_name.get(name, {})
            support_category = str(support_row.get("support_category", "unknown"))
            domain_slice = str(domain_map.get(project_id, "unknown"))
            official_project = True
            reasons: List[str] = []
            if not official_project:
                reasons.append("not_official_project")
            if cov < coverage_threshold:
                reasons.append("coverage_below_threshold")
            if n_used < minimum_n_genes_used:
                reasons.append("n_genes_used_below_minimum")
            if not (math.isfinite(var) and var > float(args.target_variance_min)):
                reasons.append("target_variance_below_threshold")
            if n_cases_with_target < int(args.min_cases_with_target):
                reasons.append("n_cases_with_target_below_minimum")
            if not fold_ok:
                reasons.append("fold_level_ridge_not_feasible")
            eligible = len(reasons) == 0
            eligibility_rows.append({
                "project_id": project_id,
                "signature_name": name,
                "eligible": bool(eligible),
                "eligibility_reason": "" if eligible else ";".join(reasons),
                "official_project": bool(official_project),
                "domain_slice": domain_slice,
                "support_category": support_category,
                "primary_score_cell": bool(eligible and domain_slice == "in_domain" and support_category == "supported"),
                "n_final_cases": int(len(cases_p)),
                "n_cases_with_target": int(n_cases_with_target),
                "n_signature_genes": int(len(sig.get("genes_resolved", []))),
                "n_genes_available_project": int(len(available_genes)),
                "n_genes_used": int(n_used),
                "coverage_fraction": float(cov),
                "target_variance": float(var),
                "target_std": float(std),
                "target_min": float(np.nanmin(y.to_numpy(dtype=float))) if n_cases_with_target > 0 else np.nan,
                "target_max": float(np.nanmax(y.to_numpy(dtype=float))) if n_cases_with_target > 0 else np.nan,
                "fold_level_ridge_feasible": bool(fold_ok),
                "fold_feasibility_reason": fold_reason,
                "score_method": str(sig.get("score_method", signature_set.get("score_method"))),
                "denominator_rule": str(sig.get("denominator_rule", signature_set.get("denominator_rule"))),
                "signature_family": str(sig.get("signature_family", "")),
                "signature_source_name": str(sig.get("source_name", "")),
                "signature_source_version": str(sig.get("source_version", "")),
                "signature_set_id": str(signature_set.get("signature_set_id", "")),
                "signature_set_hash": sha256_path(signature_set_path),
                "target_table_hash": "PENDING",
                "final_case_manifest_hash": sha256_path(final_case_path),
                "fold_assignment_hash": fold_hash,
                "rna_asset_id": rna_asset_id,
                "rna_asset_hash": sha256_path(rna_manifest_path),
                "alias_map_hash": alias_map_hash,
                "benchmark_id": args.benchmark_id,
                "modelgroup_id": modelgroup_id,
                "fold_seed": int(fold_seed),
                "n_folds": int(n_folds),
            })

        target_table = target_table.loc[:, TARGET_META_COLUMNS + sig_names]
        # Deterministic case order
        target_table = target_table.sort_values(["case_id", "case_uuid"]).reset_index(drop=True)
        target_path = target_dir / f"{project_id}.parquet"
        write_parquet_atomic(target_path, target_table, overwrite=args.force)
        target_hash = sha256_path(target_path)
        target_hash_by_project[project_id] = target_hash
        table_paths[project_id] = str(target_path)
        per_project_elig = [r for r in eligibility_rows if r["project_id"] == project_id]
        for r in per_project_elig:
            r["target_table_hash"] = target_hash
        table_manifest = manifest_header(Path(__file__).name, vars(args))
        table_manifest.update({
            "frozen_final": True,
            "benchmark_id": args.benchmark_id,
            "project_id": project_id,
            "modelgroup_id": modelgroup_id,
            "rna_asset_id": rna_asset_id,
            "rna_asset_hash": sha256_path(rna_manifest_path),
            "case_expression_hash": sha256_path(expr_path),
            "signature_set_hash": sha256_path(signature_set_path),
            "signature_set_content_sha256": signature_set.get("signature_set_content_sha256"),
            "signature_eligibility_hash": None,
            "alias_map_hash": alias_map_hash,
            "final_case_manifest_hash": sha256_path(final_case_path),
            "target_table_hash": target_hash,
            "fold_assignment_hash": fold_hash,
            "target_table_schema": TARGET_META_COLUMNS + sig_names,
            "target_construction": {
                "protocol": "frozen_project_level_gene_zscores_computed_once_after_final_cohort",
                "gene_zscore_scope": "project_final_eligible_cases",
                "gene_std_ddof": 0,
                "near_zero_gene_std_threshold": float(args.near_zero_gene_std),
                "case_missing_gene_policy": "exclude_gene_project_wide_if_missing_in_any_final_case_to_keep_fixed_signature_denominator",
                "signature_score_formula": "sum(w_g * z_project_case_g) / sum(abs(w_g)) over project-available resolved genes",
                "denominator_rule": "available_resolved_abs_weight_sum",
                "recomputed_inside_cv": False,
                "fold_local_target_construction": False,
            },
            "project_gene_filter_report": gene_report,
            "n_cases": int(len(target_table)),
            "n_signatures": int(len(sig_names)),
        })
        project_manifest_payloads[project_id] = table_manifest

    if not eligibility_rows:
        raise RuntimeError("No target tables were produced")
    eligibility = pd.DataFrame(eligibility_rows)
    eligibility = _ordered(eligibility, SIGNATURE_ELIGIBILITY_COLUMNS)
    write_csv_atomic(shared_rna_dir / "signature_eligibility.csv", eligibility, overwrite=args.force)
    eligibility_hash = sha256_path(shared_rna_dir / "signature_eligibility.csv")

    # Per-project manifests (written after signature_eligibility.csv, whose hash they record)
    for project_id in sorted(project_manifest_payloads):
        mpath = target_dir / f"{project_id}.manifest.json"
        m = dict(project_manifest_payloads[project_id])
        m["signature_eligibility_hash"] = eligibility_hash
        write_json_atomic(mpath, m, overwrite=args.force)

    primary_count = int(eligibility["primary_score_cell"].map(_truthy).sum())
    eligible_count = int(eligibility["eligible"].map(_truthy).sum())
    if eligible_count <= 0:
        raise RuntimeError("No eligible project × signature cells after target construction")
    if primary_count <= 0:
        raise RuntimeError("No primary-score eligible cells (in_domain + supported + eligible)")

    link_update = _update_asset_links(settings_dir, {
        "target_tables_dir": str(target_dir),
        "signature_eligibility_path": str(shared_rna_dir / "signature_eligibility.csv"),
        "fold_assignments_path": str(folds_dir / "fold_assignments.csv"),
        "target_and_folds_manifest_path": str(shared_rna_dir / "target_and_folds_manifest.json"),
    }, force=bool(args.force), force_cascade=bool(args.force_cascade))

    summary = {
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": modelgroup_id,
        "rna_asset_id": rna_asset_id,
        "n_official_projects": int(len(target_hash_by_project)),
        "n_target_tables": int(len(target_hash_by_project)),
        "n_fold_rows": int(len(fold_assignments)),
        "n_eligible_cells": eligible_count,
        "n_primary_score_cells": primary_count,
        "n_signature_eligibility_rows": int(len(eligibility)),
        "n_folds": int(n_folds),
        "fold_seed": int(fold_seed),
        "fold_grouping": fold_grouping,
    }
    manifest = manifest_header(Path(__file__).name, vars(args))
    manifest.update({
        "frozen_final": True,
        "benchmark_id": args.benchmark_id,
        "modelgroup_id": modelgroup_id,
        "rna_asset_id": rna_asset_id,
        "target_construction_protocol": "frozen_project_level_gene_zscores_computed_once_after_final_cohort",
        "target_z_scoring_manifest": {
            "scope": "project_final_eligible_cases",
            "gene_std_ddof": 0,
            "near_zero_gene_std_threshold": float(args.near_zero_gene_std),
            "recomputed_inside_cv": False,
            "fold_local_target_construction": False,
            "case_missing_gene_policy": "exclude_gene_project_wide_if_missing_in_any_final_case_to_keep_fixed_signature_denominator",
        },
        "fold_protocol": {
            "n_folds": int(n_folds),
            "fold_seed": int(fold_seed),
            "fold_grouping": fold_grouping,
            "degenerate_tss_fraction": float(degenerate_tss_fraction),
            "assignment": (
                "project_specific_tissue_source_site_grouped_cv"
                if fold_grouping == "tissue_source_site"
                else "project_specific_shuffled_case_level_cv"
            ),
            "per_project_fold_grouping": per_project_fold_grouping,
            "n_projects_tss_grouped": int(sum(1 for v in per_project_fold_grouping.values() if v["fold_grouping_used"] == "tissue_source_site")),
            "n_projects_case_fallback_insufficient_tss": int(sum(1 for v in per_project_fold_grouping.values() if v["fold_grouping_used"] == "case_fallback_insufficient_tss")),
            "n_projects_case_fallback_degenerate_dominant_tss": int(sum(1 for v in per_project_fold_grouping.values() if v["fold_grouping_used"] == "case_fallback_degenerate_dominant_tss")),
            "n_projects_case_explicit": int(sum(1 for v in per_project_fold_grouping.values() if v["fold_grouping_used"] == "case")),
            "same_folds_for_every_encoder_and_baseline": True,
        },
        "eligibility_thresholds": {
            "coverage_threshold": float(coverage_threshold),
            "minimum_n_genes_used": int(minimum_n_genes_used),
            "target_variance_min": float(args.target_variance_min),
            "min_cases_with_target": int(args.min_cases_with_target),
            "inner_cv_folds_checked": int(inner_cv_folds),
        },
        "input_paths": {
            "case_expression": str(expr_path),
            "selected_rna_manifest": str(selected_rna_path),
            "rna_asset_manifest": str(rna_manifest_path),
            "final_case_manifest": str(final_case_path),
            "project_summary": str(project_summary_path),
            "signature_set": str(signature_set_path),
            "signature_support_training": str(signature_support_path),
            "project_domain_map": str(settings_dir / "project_domain_map.json"),
            "benchmark_config": str(settings_dir / "benchmark_config.yaml"),
        },
        "input_hashes": {
            "case_expression": sha256_path(expr_path),
            "selected_rna_manifest": sha256_path(selected_rna_path),
            "rna_asset_manifest": sha256_path(rna_manifest_path),
            "final_case_manifest": sha256_path(final_case_path),
            "project_summary": sha256_path(project_summary_path),
            "signature_set": sha256_path(signature_set_path),
            "signature_support_training": sha256_path(signature_support_path),
            "project_domain_map": sha256_path(settings_dir / "project_domain_map.json"),
            "benchmark_config": sha256_path(settings_dir / "benchmark_config.yaml"),
            "benchmark_asset_links_after_update": sha256_path(settings_dir / "benchmark_asset_links.json"),
        },
        "output_paths": {
            "target_tables_dir": str(target_dir),
            "signature_eligibility": str(shared_rna_dir / "signature_eligibility.csv"),
            "fold_assignments": str(folds_dir / "fold_assignments.csv"),
        },
        "target_table_hashes_by_project": target_hash_by_project,
        "target_table_paths_by_project": table_paths,
        "signature_eligibility_hash": eligibility_hash,
        "fold_assignment_hash": fold_hash,
        "summary": summary,
        "asset_link_update": link_update,
        "output_hashes": collect_output_hashes(shared_rna_dir),
        "fold_output_hashes": collect_output_hashes(folds_dir),
    })
    write_json_atomic(shared_rna_dir / "target_and_folds_manifest.json", manifest, overwrite=args.force)
    _write_report(reports_dir / REPORT_NAME, summary, per_project_fold_grouping=per_project_fold_grouping, overwrite=args.force)

    print(json.dumps({"status": "ok", **summary, "target_tables_dir": str(target_dir)}, indent=2))
    return 0


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Build the frozen TCGA Task B RNA signature targets and CV folds.")

    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help="Benchmark ID whose RNA targets and folds are built.")

    ap.add_argument("--n_folds", type=int,
                    default=None,
                    help="Number of cross-validation folds. Defaults to the benchmark configuration.")

    ap.add_argument("--fold_seed", type=int,
                    default=None,
                    help="Random seed for fold assignment. Defaults to the benchmark configuration.")

    ap.add_argument("--fold_grouping", type=str,
                    default=None, choices=["tissue_source_site", "case"],
                    help="CV fold grouping: 'tissue_source_site' (each site in one fold) or 'case' (random "
                         "case-level folds). Defaults to the benchmark configuration.")

    ap.add_argument("--degenerate_tss_fraction", type=float,
                    default=None,
                    help="Max share of a project's cases in one tissue-source site for TSS-grouped folds, above "
                         "which case-level folds are used, in (0, 1]. Defaults to the benchmark configuration.")

    ap.add_argument("--near_zero_gene_std", type=float,
                    default=1e-12,
                    help="Gene-expression standard deviation below which a gene is treated as constant.")

    ap.add_argument("--target_variance_min", type=float,
                    default=1e-6,
                    help="Minimum variance a signature target must have to be kept.")

    ap.add_argument("--min_cases_with_target", type=int,
                    default=25,
                    help="Minimum cases with a defined target for a signature to be eligible in a project.")

    ap.add_argument("--coverage_threshold", type=float,
                    default=None,
                    help="Minimum signature gene-coverage fraction. Defaults to the signature configuration.")

    ap.add_argument("--minimum_n_genes_used", type=int,
                    default=None,
                    help="Minimum genes used to score a signature. Defaults to the signature configuration.")

    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing target tables and folds.")

    ap.add_argument("--force_cascade", action="store_true",
                    help="Required with --force if downstream encoder features or ridge results already exist.")

    args = ap.parse_args()
    raise SystemExit(main(args))
