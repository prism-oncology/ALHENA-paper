#!/usr/bin/env python3
"""
Compute which Task B signatures are supported by the genes of the HistoST training slides (set == "train" in
infos_<training_dataset_id>.csv, genes from genes_training_<training_dataset_id>.json).

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data / benchmarks / <benchmark_id> /
    - settings/signature_support_training.json, settings/signature_support_training_manifest.json
    - reports/signature_support_training_report.md
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
import pandas as pd
import sys

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path

from _taskb_common import (
    collect_output_hashes,
    load_alias_resolver,
    manifest_header,
    read_json,
    read_yaml,
    sha256_obj,
    sha256_path,
    write_json_atomic,
)

DEFAULT_GENE_MIN_SLIDES = 5
DEFAULT_GENE_MIN_TISSUES = 2
DEFAULT_SIGNATURE_SUPPORTED_MIN_GENES = 10
DEFAULT_SIGNATURE_SUPPORTED_MIN_FRACTION = 0.60
DEFAULT_SIGNATURE_PARTIAL_MIN_GENES = 5
DEFAULT_SIGNATURE_PARTIAL_MIN_FRACTION = 0.30

ALIAS_POLICY = "frozen_HGNC_alias_previous_symbols_exact_case_insensitive_keep_unmapped_with_audit"


def _existing_state(path: Path) -> str:
    if not path.exists():
        return "absent"
    try:
        obj = read_json(path)
    except Exception:
        return "unreadable"
    if obj.get("frozen_final") is True or obj.get("status") == "frozen_final":
        return "frozen_final"
    if str(obj.get("status", "")).startswith("pending_build_by__5_build_signature_support_training"):
        return "pending_sentinel"
    return "nonfinal_existing"


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    obj = read_json(path) if path.suffix.lower() == ".json" else read_yaml(path)
    if not isinstance(obj, Mapping):
        raise RuntimeError(f"Expected mapping at {path}")
    return dict(obj)


def _resolve_training_paths(
    *,
    dataset_id: str,
    training_dataset_id: str,
    infos_path: Optional[Path],
    genes_path: Optional[Path],
) -> Tuple[Path, Path, Dict[str, Any]]:
    if infos_path is None:
        infos_path = get_full_path(
            "train",
            "training_datasets",
            dataset_id,
            training_dataset_id,
            "informations",
            f"infos_{training_dataset_id}.csv",
        )
        infos_source = "get_full_path_default"
    else:
        infos_path = Path(infos_path)
        infos_source = "cli_override"
    if genes_path is None:
        genes_path = get_full_path(
            "train",
            "training_datasets",
            dataset_id,
            training_dataset_id,
            f"genes_training_{training_dataset_id}.json",
        )
        genes_source = "get_full_path_default"
    else:
        genes_path = Path(genes_path)
        genes_source = "cli_override"
    return infos_path, genes_path, {"infos_path_source": infos_source, "genes_path_source": genes_source}


def _nonempty(x: Any) -> bool:
    if x is None:
        return False
    if pd.isna(x):
        return False
    s = str(x).strip()
    return bool(s and s.lower() not in {"nan", "none", "null", "na", "n/a", "<na>"})


def _as_clean_str(x: Any) -> str:
    return str(x).strip()


def _normalize_set_value(x: Any) -> str:
    return str(x).strip().lower()


def _load_train_infos(path: Path) -> Tuple[pd.DataFrame, Dict[str, str], List[str], List[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    infos = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = ["slide_id", "type", "set"]
    missing = [c for c in required if c not in infos.columns]
    if missing:
        raise RuntimeError(f"infos CSV is missing required columns {missing}. Columns={list(infos.columns)}")
    infos = infos.copy()
    infos["__set_norm__"] = infos["set"].map(_normalize_set_value)
    train = infos.loc[infos["__set_norm__"].eq("train")].copy()
    if train.empty:
        raise RuntimeError(f"No rows with set == 'train' in {path}")
    train = train.loc[train["slide_id"].map(_nonempty)].copy()
    if train.empty:
        raise RuntimeError("Training rows exist but none has a non-empty slide_id")
    train["slide_id"] = train["slide_id"].map(_as_clean_str)
    train["type"] = train["type"].map(_as_clean_str)
    slide_to_tissues: Dict[str, str] = {}
    conflicts: Dict[str, List[str]] = {}
    for slide_id, g in train.groupby("slide_id", sort=True):
        tissues = sorted({str(x).strip() for x in g["type"].tolist() if _nonempty(x)})
        if not tissues:
            raise RuntimeError(f"Training slide {slide_id!r} has no non-empty tissue type in infos CSV")
        if len(tissues) != 1:
            conflicts[slide_id] = tissues
        else:
            slide_to_tissues[slide_id] = tissues[0]
    if conflicts:
        sample = dict(list(conflicts.items())[:20])
        raise RuntimeError(f"A training slide_id maps to multiple tissue type values; refusing to infer. Examples: {sample}")
    train_slide_ids = sorted(slide_to_tissues)
    train_tissue_types = sorted(set(slide_to_tissues.values()))
    return train, slide_to_tissues, train_slide_ids, train_tissue_types


def _extract_genes_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        # Accept common wrappers of the slide_id -> genes mapping
        for key in ["genes_by_slide", "genes_training", "slide_genes", "data"]:
            if isinstance(obj.get(key), Mapping):
                return obj[key]
        return obj
    raise RuntimeError("genes_training JSON must be an object mapping slide_id to a list of genes, or a wrapper containing that mapping")


def _coerce_gene_list(value: Any, *, slide_id: str) -> List[str]:
    if isinstance(value, Mapping):
        for key in ["genes", "gene_symbols", "genes_training", "symbols"]:
            if key in value:
                value = value[key]
                break
    if isinstance(value, str):
        # Gene list given as a delimited string
        value = [g.strip() for g in value.replace(";", ",").replace("|", ",").split(",") if g.strip()]
    if not isinstance(value, list):
        raise RuntimeError(f"genes_training entry for slide_id={slide_id!r} must be a list of gene symbols")
    out = []
    for x in value:
        if _nonempty(x):
            out.append(str(x).strip())
    return out


def _load_train_genes(path: Path, train_slide_ids: Sequence[str], *, allow_missing_train_slide_genes: bool) -> Tuple[Dict[str, List[str]], int]:
    if not path.exists():
        raise FileNotFoundError(path)
    obj = read_json(path)
    raw_map = _extract_genes_mapping(obj)
    train_set = set(train_slide_ids)
    missing = sorted(train_set - set(str(k) for k in raw_map.keys()))
    if missing and not allow_missing_train_slide_genes:
        raise RuntimeError(
            "genes_training JSON is missing training slide IDs from infos CSV; this would undercount support. "
            f"Missing examples: {missing[:20]}"
        )
    genes_by_slide: Dict[str, List[str]] = {}
    ignored_nontrain = 0
    for raw_slide, raw_genes in raw_map.items():
        slide_id = str(raw_slide).strip()
        if slide_id not in train_set:
            ignored_nontrain += 1
            continue
        genes_by_slide[slide_id] = _coerce_gene_list(raw_genes, slide_id=slide_id)
    if not genes_by_slide:
        raise RuntimeError("No genes from genes_training JSON overlap training slide IDs from infos CSV")
    empty = sorted([sid for sid, genes in genes_by_slide.items() if not genes])
    if empty:
        raise RuntimeError(f"Training slides with empty gene lists are not allowed. Examples: {empty[:20]}")
    return genes_by_slide, ignored_nontrain


def _signature_list(signature_set: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    sigs = signature_set.get("signatures")
    if not isinstance(sigs, list) or not sigs:
        raise RuntimeError("signature_set.signatures must be a non-empty list")
    out: List[Mapping[str, Any]] = []
    for i, sig in enumerate(sigs):
        if not isinstance(sig, Mapping):
            raise RuntimeError(f"signature_set.signatures[{i}] is not an object")
        out.append(sig)
    return out


def _validate_signature_set(signature_set: Mapping[str, Any], alias_sha: str) -> None:
    if signature_set.get("frozen_final") is not True and signature_set.get("status") != "frozen_final":
        raise RuntimeError("signature_set_tcga_case_v1.json is not a frozen final signature set")
    found_alias = signature_set.get("alias_map_sha256") or signature_set.get("alias_map_hash")
    if found_alias != alias_sha:
        raise RuntimeError(f"Signature set alias hash mismatch: signature_set={found_alias} benchmark_alias={alias_sha}")
    for sig in _signature_list(signature_set):
        name = sig.get("signature_name")
        genes = sig.get("genes_resolved")
        weights = sig.get("gene_weights")
        if not name or not isinstance(genes, list) or not genes:
            raise RuntimeError(f"Invalid signature gene list for {name!r}")
        if not isinstance(weights, Mapping) or set(map(str, genes)) != set(map(str, weights.keys())):
            raise RuntimeError(f"Signature {name!r} has inconsistent genes_resolved/gene_weights")


def _resolve_training_gene_support(
    genes_by_slide: Mapping[str, List[str]],
    slide_to_tissue: Mapping[str, str],
    resolver: Any,
    *,
    gene_min_slides: int,
    gene_min_tissues: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    gene_to_slides: MutableMapping[str, set] = {}
    gene_to_tissues: MutableMapping[str, set] = {}
    alias_status_counts: MutableMapping[str, int] = {}
    duplicate_gene_events = 0
    per_slide_counts: Dict[str, Dict[str, int]] = {}

    train_slide_ids = sorted(slide_to_tissue)
    for slide_id in train_slide_ids:
        raw_genes = list(genes_by_slide.get(slide_id, []))
        resolved_seen_this_slide = set()
        status_seen: MutableMapping[str, int] = {}
        for raw in raw_genes:
            resolved, status = resolver.resolve(raw)
            alias_status_counts[status] = alias_status_counts.get(status, 0) + 1
            status_seen[status] = status_seen.get(status, 0) + 1
            if not resolved:
                continue
            resolved = str(resolved).strip()
            if not resolved:
                continue
            if resolved in resolved_seen_this_slide:
                duplicate_gene_events += 1
                continue
            resolved_seen_this_slide.add(resolved)
            gene_to_slides.setdefault(resolved, set()).add(slide_id)
            gene_to_tissues.setdefault(resolved, set()).add(slide_to_tissue[slide_id])
        per_slide_counts[slide_id] = {
            "n_original_genes": int(len(raw_genes)),
            "n_unique_resolved_genes": int(len(resolved_seen_this_slide)),
            "n_alias_or_case_resolved": int(status_seen.get("alias_or_case_resolved", 0)),
            "n_unmapped_kept_original": int(status_seen.get("unmapped_kept_original", 0)),
            "n_missing": int(status_seen.get("missing", 0)),
        }

    n_train_slides = len(train_slide_ids)
    support_by_gene: Dict[str, Dict[str, Any]] = {}
    for gene in sorted(gene_to_slides):
        slides = sorted(gene_to_slides[gene])
        tissues = sorted(gene_to_tissues.get(gene, set()))
        support_flag = len(slides) >= gene_min_slides and len(tissues) >= gene_min_tissues
        support_by_gene[gene] = {
            "resolved_symbol": gene,
            "n_slides_supported": int(len(slides)),
            "slide_coverage_fraction": float(len(slides) / n_train_slides) if n_train_slides else 0.0,
            "n_tissue_groups_supported": int(len(tissues)),
            "supported_slide_ids": slides,
            "supported_tissue_types": tissues,
            "support_flag": bool(support_flag),
        }
    audit = {
        "n_train_slides": int(n_train_slides),
        "n_training_genes_resolved_union": int(len(support_by_gene)),
        "alias_resolution_status_counts": dict(sorted(alias_status_counts.items())),
        "duplicate_resolved_gene_events_within_slide_dropped": int(duplicate_gene_events),
        "per_slide_gene_counts": per_slide_counts,
    }
    return support_by_gene, audit


def _categorize_signature(
    *,
    n_supported_genes: int,
    supported_fraction: float,
    supported_min_genes: int,
    supported_min_fraction: float,
    partial_min_genes: int,
    partial_min_fraction: float,
) -> str:
    if n_supported_genes >= supported_min_genes and supported_fraction >= supported_min_fraction:
        return "supported"
    if n_supported_genes >= partial_min_genes and supported_fraction >= partial_min_fraction:
        return "partial"
    return "unsupported"


def _build_signature_support(
    signature_set: Mapping[str, Any],
    gene_support: Mapping[str, Mapping[str, Any]],
    *,
    supported_min_genes: int,
    supported_min_fraction: float,
    partial_min_genes: int,
    partial_min_fraction: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    by_name: Dict[str, Dict[str, Any]] = {}
    for sig in _signature_list(signature_set):
        name = str(sig["signature_name"])
        genes = [str(g) for g in sig.get("genes_resolved", [])]
        if not genes:
            raise RuntimeError(f"Signature {name} has no resolved genes")
        supported_genes = [g for g in genes if bool((gene_support.get(g) or {}).get("support_flag", False))]
        present_genes = [g for g in genes if g in gene_support]
        unsupported_genes = [g for g in genes if g not in supported_genes]
        supported_fraction = len(supported_genes) / len(genes)
        present_fraction = len(present_genes) / len(genes)
        category = _categorize_signature(
            n_supported_genes=len(supported_genes),
            supported_fraction=supported_fraction,
            supported_min_genes=supported_min_genes,
            supported_min_fraction=supported_min_fraction,
            partial_min_genes=partial_min_genes,
            partial_min_fraction=partial_min_fraction,
        )
        gene_rows = []
        for g in genes:
            gs = gene_support.get(g)
            gene_rows.append({
                "resolved_symbol": g,
                "in_training_gene_universe": bool(gs is not None),
                "support_flag": bool(gs.get("support_flag", False)) if gs else False,
                "n_slides_supported": int(gs.get("n_slides_supported", 0)) if gs else 0,
                "n_tissue_groups_supported": int(gs.get("n_tissue_groups_supported", 0)) if gs else 0,
                "slide_coverage_fraction": float(gs.get("slide_coverage_fraction", 0.0)) if gs else 0.0,
            })
        row = {
            "signature_name": name,
            "support_category": category,
            "primary_score_eligible": bool(category == "supported"),
            "exploratory_only": bool(category != "supported"),
            "n_signature_genes": int(len(genes)),
            "n_training_present_genes": int(len(present_genes)),
            "training_present_fraction": float(present_fraction),
            "n_supported_genes": int(len(supported_genes)),
            "supported_fraction": float(supported_fraction),
            "supported_genes": sorted(supported_genes),
            "unsupported_or_below_threshold_genes": sorted(unsupported_genes),
            "per_gene_support": gene_rows,
            "threshold_rule": {
                "supported": {"n_supported_genes_min": supported_min_genes, "supported_fraction_min": supported_min_fraction},
                "partial": {"n_supported_genes_min": partial_min_genes, "supported_fraction_min": partial_min_fraction},
            },
        }
        rows.append(row)
        by_name[name] = row
    return rows, by_name


def _report_lines(artifact: Mapping[str, Any]) -> str:
    counts = artifact.get("signature_support_counts", {})
    return "\n".join([
        "# Signature training-support report",
        "",
        f"- benchmark_id: `{artifact.get('benchmark_id')}`",
        f"- dataset_id: `{artifact.get('dataset_id')}`",
        f"- training_dataset_id: `{artifact.get('training_dataset_id')}`",
        f"- train slides used: `{artifact.get('training_summary', {}).get('n_train_slides')}`",
        f"- training tissue groups: `{artifact.get('training_summary', {}).get('n_training_tissue_types')}`",
        f"- supported signatures: `{counts.get('supported', 0)}`",
        f"- partial signatures: `{counts.get('partial', 0)}`",
        f"- unsupported signatures: `{counts.get('unsupported', 0)}`",
        "",
        "Primary Task B scoring uses only signatures with `support_category == supported`.",
        "",
    ])


def main(args) -> int:

    if args.gene_min_slides < 1 or args.gene_min_tissues < 1:
        raise RuntimeError("Gene support thresholds must be positive")
    if not (0.0 <= args.signature_supported_min_fraction <= 1.0 and 0.0 <= args.signature_partial_min_fraction <= 1.0):
        raise RuntimeError("Signature support fractions must be in [0, 1]")

    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    settings_dir = benchmark_root / "settings"
    reports_dir = benchmark_root / "reports"
    if not settings_dir.exists():
        raise FileNotFoundError(settings_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = settings_dir / "benchmark_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = _load_yaml_or_json(cfg_path)
    dataset_id = args.dataset_id or cfg.get("dataset_id")
    training_dataset_id = args.training_dataset_id or cfg.get("training_dataset_id")
    if not dataset_id or not training_dataset_id:
        raise RuntimeError("dataset_id and training_dataset_id must be supplied by CLI or benchmark_config.yaml")
    dataset_id = str(dataset_id)
    training_dataset_id = str(training_dataset_id)

    output_path = settings_dir / "signature_support_training.json"
    manifest_path = settings_dir / "signature_support_training_manifest.json"
    existing_state = _existing_state(output_path)
    if existing_state == "frozen_final" and not args.force:
        raise FileExistsError(f"Frozen signature support artifact already exists: {output_path}. Use --force to rebuild.")
    if existing_state in {"nonfinal_existing", "unreadable"} and not args.force:
        raise FileExistsError(f"Non-pending signature support file already exists: {output_path}. Use --force if intentional.")
    downstream_targets = [
        benchmark_root / "shared" / "rna" / "target_tables",
        benchmark_root / "shared" / "rna" / "signature_eligibility.csv",
        benchmark_root / "shared" / "rna" / "target_and_folds_manifest.json",
        benchmark_root / "shared" / "folds",
    ]
    if args.force and any(p.exists() for p in downstream_targets) and not args.force_cascade:
        raise RuntimeError(f"Refusing to rebuild support artifact while downstream target/fold outputs exist: {downstream_targets}. Use --force_cascade if intentional.")
    overwrite_outputs = bool(args.force or existing_state == "pending_sentinel")

    alias_candidates = sorted(settings_dir.glob("gene_alias_map_hgnc_*.tsv"))
    if len(alias_candidates) != 1:
        raise RuntimeError(f"Expected exactly one frozen benchmark HGNC alias map; found {len(alias_candidates)}")
    alias_map_path = alias_candidates[0]
    resolver = load_alias_resolver(alias_map_path, source="benchmark_frozen_HGNC_alias_map")

    signature_set_path = settings_dir / "signature_set_tcga_case_v1.json"
    if not signature_set_path.exists():
        raise FileNotFoundError(signature_set_path)
    signature_set = read_json(signature_set_path)
    _validate_signature_set(signature_set, resolver.sha256)

    infos_path, genes_path, path_meta = _resolve_training_paths(
        dataset_id=dataset_id,
        training_dataset_id=training_dataset_id,
        infos_path=args.infos_path,
        genes_path=args.genes_path,
    )

    train_infos, slide_to_tissue, train_slide_ids, train_tissue_types = _load_train_infos(infos_path)
    genes_by_slide_raw, ignored_nontrain_slide_count = _load_train_genes(
        genes_path, train_slide_ids, allow_missing_train_slide_genes=args.allow_missing_train_slide_genes
    )

    gene_support, gene_audit = _resolve_training_gene_support(
        genes_by_slide_raw,
        slide_to_tissue,
        resolver,
        gene_min_slides=args.gene_min_slides,
        gene_min_tissues=args.gene_min_tissues,
    )
    signature_rows, signature_by_name = _build_signature_support(
        signature_set,
        gene_support,
        supported_min_genes=args.signature_supported_min_genes,
        supported_min_fraction=args.signature_supported_min_fraction,
        partial_min_genes=args.signature_partial_min_genes,
        partial_min_fraction=args.signature_partial_min_fraction,
    )

    counts = {"supported": 0, "partial": 0, "unsupported": 0}
    for row in signature_rows:
        counts[row["support_category"]] = counts.get(row["support_category"], 0) + 1

    support_thresholds = {
        "gene_support": {
            "n_slides_supported_min": int(args.gene_min_slides),
            "n_tissue_groups_supported_min": int(args.gene_min_tissues),
        },
        "signature_supported": {
            "n_supported_genes_min": int(args.signature_supported_min_genes),
            "supported_fraction_min": float(args.signature_supported_min_fraction),
        },
        "signature_partial": {
            "n_supported_genes_min": int(args.signature_partial_min_genes),
            "supported_fraction_min": float(args.signature_partial_min_fraction),
        },
    }

    artifact: Dict[str, Any] = {
        "status": "frozen_final",
        "frozen_final": True,
        "benchmark_id": args.benchmark_id,
        "dataset_id": dataset_id,
        "training_dataset_id": training_dataset_id,
        "created_at_utc": manifest_header(Path(__file__).name, vars(args))["created_at_utc"],
        "artifact_name": "signature_support_training",
        "artifact_version": "tcga_taskb_signature_training_support_v1",
        "alias_map_path": str(alias_map_path),
        "alias_map_sha256": resolver.sha256,
        "alias_map_hash": resolver.sha256,
        "alias_map_date": resolver.date,
        "alias_map_source": resolver.source,
        "alias_resolution_policy": ALIAS_POLICY,
        "signature_set_path": str(signature_set_path),
        "signature_set_sha256": sha256_path(signature_set_path),
        "signature_set_content_sha256": signature_set.get("signature_set_content_sha256"),
        "infos_path": str(infos_path),
        "infos_sha256": sha256_path(infos_path),
        "genes_path": str(genes_path),
        "genes_sha256": sha256_path(genes_path),
        "path_resolution": path_meta,
        "support_thresholds": support_thresholds,
        "training_summary": {
            "n_train_rows": int(len(train_infos)),
            "n_train_slides": int(len(train_slide_ids)),
            "n_training_tissue_types": int(len(train_tissue_types)),
            "training_slide_ids": train_slide_ids,
            "training_tissue_types": train_tissue_types,
            "ignored_nontrain_slide_ids_present_in_genes_json": int(ignored_nontrain_slide_count),
        },
        "gene_support": gene_support,
        "gene_support_audit": gene_audit,
        "signatures": signature_rows,
        "signatures_by_name": signature_by_name,
        "primary_signature_names": [r["signature_name"] for r in signature_rows if r["support_category"] == "supported"],
        "exploratory_signature_names": [r["signature_name"] for r in signature_rows if r["support_category"] != "supported"],
        "signature_support_counts": counts,
        "primary_scoring_rule": "use only signatures where support_category == 'supported' for primary Task B score",
        "partial_and_unsupported_rule": "partial and unsupported signatures may run but must be marked exploratory",
        "no_online_downloads_performed": True,
    }
    content_for_hash = dict(artifact)
    content_for_hash.pop("created_at_utc", None)
    artifact["signature_support_training_content_sha256"] = sha256_obj(content_for_hash)

    write_json_atomic(output_path, artifact, overwrite=overwrite_outputs)

    manifest = manifest_header(Path(__file__).name, vars(args))
    manifest.update({
        "benchmark_id": args.benchmark_id,
        "benchmark_root": str(benchmark_root),
        "dataset_id": dataset_id,
        "training_dataset_id": training_dataset_id,
        "infos_path": str(infos_path),
        "infos_sha256": sha256_path(infos_path),
        "genes_path": str(genes_path),
        "genes_sha256": sha256_path(genes_path),
        "signature_set_path": str(signature_set_path),
        "signature_set_hash": sha256_path(signature_set_path),
        "signature_set_content_sha256": signature_set.get("signature_set_content_sha256"),
        "alias_map_path": str(alias_map_path),
        "alias_map_hash": resolver.sha256,
        "alias_map_sha256": resolver.sha256,
        "alias_map_date": resolver.date,
        "alias_map_source": resolver.source,
        "alias_resolution_policy": ALIAS_POLICY,
        "duplicate_symbol_policy": "per_slide_duplicate_resolved_training_genes_count_once_for_support",
        "train_slide_ids_used": train_slide_ids,
        "train_tissue_types_used": train_tissue_types,
        "support_thresholds": support_thresholds,
        "n_training_genes_resolved_union": gene_audit["n_training_genes_resolved_union"],
        "signature_support_counts": counts,
        "signature_support_training_path": str(output_path),
        "signature_support_training_file_sha256": sha256_path(output_path),
        "signature_support_training_content_sha256": artifact["signature_support_training_content_sha256"],
        "output_hashes": collect_output_hashes(settings_dir),
    })
    write_json_atomic(manifest_path, manifest, overwrite=overwrite_outputs)

    # Human-readable report
    from _taskb_common import write_text_atomic
    write_text_atomic(reports_dir / "signature_support_training_report.md", _report_lines(artifact), overwrite=args.force or existing_state == "pending_sentinel")

    print(json.dumps({
        "status": "ok",
        "benchmark_id": args.benchmark_id,
        "signature_support_training_path": str(output_path),
        "n_train_slides": len(train_slide_ids),
        "n_training_genes_resolved_union": gene_audit["n_training_genes_resolved_union"],
        "signature_support_counts": counts,
    }, indent=2))
    return 0


if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Build the frozen HistoST-training gene support for TCGA Task B signatures.")

    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help="Benchmark ID whose signatures are scored against the training gene universe.")

    ap.add_argument("--dataset_id", type=str,
                    default=None,
                    help="Overrides the dataset_id read from benchmark_config.yaml.")

    ap.add_argument("--training_dataset_id", type=str,
                    default=None,
                    help="Overrides the training_dataset_id read from benchmark_config.yaml.")

    ap.add_argument("--infos_path", type=Path,
                    default=None,
                    help="Optional explicit path to infos_{training_dataset_id}.csv.")

    ap.add_argument("--genes_path", type=Path,
                    default=None,
                    help="Optional explicit path to genes_training_{training_dataset_id}.json.")

    ap.add_argument("--gene_min_slides", type=int,
                    default=DEFAULT_GENE_MIN_SLIDES,
                    help="Minimum number of train slides a gene must appear in to count as supported.")

    ap.add_argument("--gene_min_tissues", type=int,
                    default=DEFAULT_GENE_MIN_TISSUES,
                    help="Minimum number of distinct tissues a gene must appear in to count as supported.")

    ap.add_argument("--signature_supported_min_genes", type=int,
                    default=DEFAULT_SIGNATURE_SUPPORTED_MIN_GENES,
                    help="Minimum supported genes for a signature to be classified as supported.")

    ap.add_argument("--signature_supported_min_fraction", type=float,
                    default=DEFAULT_SIGNATURE_SUPPORTED_MIN_FRACTION,
                    help="Minimum supported-gene fraction for a signature to be classified as supported.")

    ap.add_argument("--signature_partial_min_genes", type=int,
                    default=DEFAULT_SIGNATURE_PARTIAL_MIN_GENES,
                    help="Minimum supported genes for a signature to be classified as partially supported.")

    ap.add_argument("--signature_partial_min_fraction", type=float,
                    default=DEFAULT_SIGNATURE_PARTIAL_MIN_FRACTION,
                    help="Minimum supported-gene fraction for a signature to be classified as partially supported.")

    ap.add_argument("--allow_missing_train_slide_genes", action="store_true",
                    help="Debug only: do not fail if the genes JSON lacks a train slide.")

    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing frozen signature support artifact.")

    ap.add_argument("--force_cascade", action="store_true",
                    help="Required with --force to overwrite a frozen support artifact if downstream target/fold outputs exist.")

    args = ap.parse_args()
    raise SystemExit(main(args))
