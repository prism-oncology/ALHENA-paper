#!/usr/bin/env python3
"""
Build the TCGA Task B signature set (prediction targets) from the gene-signature panel copied by _3_init_benchmark.py
into settings/eval_signatures/, then validate it against the panel and the HGNC alias map.

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data / benchmarks / <benchmark_id> / settings /
    - signature_set_tcga_case_v1.json: signature set
    - signature_build_manifest.json, signature_build_report.md
    - signature_validation_manifest.json, signature_validation_report.md (exits with an error if validation fails)
"""

import argparse
import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple
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
    sha256_obj,
    sha256_path,
    write_json_atomic,
    write_text_atomic,
)

REQUIRED_OUTPUT = "signature_set_tcga_case_v1.json"
DEFAULT_SIGNATURE_SET_ID = "signature_set_tcga_case_v1"

# Methodology constants
SCORE_METHOD = "weighted_project_zscore_mean"
DENOMINATOR_RULE = "available_resolved_abs_weight_sum"
COVERAGE_THRESHOLD = 0.70
# 2 admits short signatures (e.g. CYTOLYTIC_ACTIVITY = GZMA + PRF1)
MINIMUM_N_GENES_USED = 2
ALIAS_POLICY = "frozen_HGNC_alias_previous_symbols_exact_case_insensitive_keep_unmapped_with_audit"


# =============================================================================
# Build section
# =============================================================================


@dataclasses.dataclass(frozen=True)
class GeneRecord:
    signature_name: str
    gene_symbol: str
    weight: float
    source_order: int


@dataclasses.dataclass(frozen=True)
class SourceCandidate:
    signature_name: str
    records: Tuple[GeneRecord, ...]
    metadata: Dict[str, Any]
    content_hash: str


def _nonempty_str(x: Any) -> str | None:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    return s


def _float(x: Any, *, context: str) -> float:
    try:
        v = float(x)
    except Exception as e:
        raise RuntimeError(f"Non-numeric signature weight in {context}: {x!r}") from e
    if not math.isfinite(v):
        raise RuntimeError(f"Non-finite signature weight in {context}: {x!r}")
    return v


def _normalize_signature_name(x: Any) -> str:
    s = str(x).strip()
    return re.sub(r"\s+", "_", s).upper()


def _normalize_gene_key(x: Any) -> str:
    return re.sub(r"\s+", "", str(x).strip()).upper()


def _parse_gmt(gmt_path: Path) -> Dict[str, List[str]]:
    """Parse a GMT file into {signature_id: [gene, gene, ...]} (duplicate genes removed)."""
    if not gmt_path.is_file():
        raise FileNotFoundError(f"signatures.gmt not found: {gmt_path}")

    signatures: Dict[str, List[str]] = {}
    with gmt_path.open("r") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(
                    f"Malformed GMT line {line_no} in '{gmt_path}': expected "
                    f"at least 3 tab-separated fields, got {len(parts)}."
                )
            sig_id = parts[0].strip()
            if not sig_id:
                raise ValueError(f"Empty signature_id on line {line_no} of '{gmt_path}'.")
            if sig_id in signatures:
                raise ValueError(
                    f"Duplicate signature_id '{sig_id}' on line {line_no} of '{gmt_path}'."
                )
            genes = [g.strip() for g in parts[2:] if g.strip()]
            if not genes:
                raise ValueError(
                    f"Signature '{sig_id}' (line {line_no}) has zero genes."
                )
            seen: set = set()
            deduped: List[str] = []
            for g in genes:
                if g not in seen:
                    seen.add(g)
                    deduped.append(g)
            signatures[sig_id] = deduped
    if not signatures:
        raise ValueError(f"No signatures parsed from '{gmt_path}'.")
    return signatures


def _load_sources_json(sources_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load sources.json into {signature_id: metadata_dict}."""
    if not sources_path.is_file():
        raise FileNotFoundError(f"signatures sources.json not found: {sources_path}")
    payload = read_json(sources_path)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"'{sources_path}' must be a JSON object keyed by signature_id."
        )
    return {str(k): dict(v) if isinstance(v, Mapping) else {} for k, v in payload.items()}


def _normalized_gmt_required_signatures(gmt_path: Path) -> List[str]:
    """Return the normalized, de-duplicated list of signature ids from a GMT file."""
    gmt = _parse_gmt(gmt_path)
    required: List[str] = []
    seen_norm: Dict[str, str] = {}
    for sig_id in gmt.keys():
        norm = _normalize_signature_name(sig_id)
        if norm in seen_norm:
            raise RuntimeError(
                f"Two GMT entries normalize to the same signature_id "
                f"({seen_norm[norm]!r} and {sig_id!r}); refusing to silently merge."
            )
        seen_norm[norm] = sig_id
        required.append(norm)
    return required


def _signature_family(sig_id: str, src_meta: Mapping[str, Any]) -> str:
    """Pick a signature_family label.

    HALLMARK_ ids map to 'hallmark'; otherwise derive from sources.json category
    (lowercased, non-alnum → '_'), falling back to 'immune'.
    """
    if sig_id.startswith("HALLMARK_"):
        return "hallmark"
    category = _nonempty_str(src_meta.get("category"))
    if category:
        return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_") or "immune"
    return "immune"


def _source_metadata_from_sources_json(
    sig_id: str,
    src_meta: Mapping[str, Any],
    *,
    signatures_gmt_relpath: str,
    signatures_gmt_sha256: str,
) -> Dict[str, Any]:
    """Per-signature source metadata: GMT path and hash, citation/version/url from sources.json."""
    citation = _nonempty_str(src_meta.get("citation")) or ""
    version = (
        _nonempty_str(src_meta.get("msigdb_version"))
        or _nonempty_str(src_meta.get("version"))
    )
    if not version and citation:
        m = re.search(r"\b(19|20)\d{2}\b", citation)
        if m:
            version = m.group(0)
    url = (
        _nonempty_str(src_meta.get("msigdb_url"))
        or _nonempty_str(src_meta.get("url"))
        or _nonempty_str(src_meta.get("download_url"))
        or signatures_gmt_relpath
    )
    return {
        "source_name": sig_id,
        "source_version": version or "unspecified",
        "source_citation": citation or "unspecified",
        "source_url_or_path": url,
        "source_local_path": signatures_gmt_relpath,
        "source_file_hash": signatures_gmt_sha256,
        "source_file_sha256": signatures_gmt_sha256,
        "license_or_usage_note": _nonempty_str(
            src_meta.get("license_or_usage_note")
            or src_meta.get("license")
            or src_meta.get("usage_note")
        ),
        "gene_symbol_column": None,
        "weight_column": None,
        "signature_family": _signature_family(sig_id, src_meta),
    }


def _candidate_from_records(
    signature_name: str, records: List[GeneRecord], metadata: Dict[str, Any]
) -> SourceCandidate:
    if not records:
        raise RuntimeError(f"Empty gene list for {signature_name}")
    normalized = tuple(
        GeneRecord(signature_name, str(r.gene_symbol).strip(), float(r.weight), int(i))
        for i, r in enumerate(records)
        if _nonempty_str(r.gene_symbol)
    )
    content_hash = sha256_obj([
        {"gene_symbol": r.gene_symbol, "weight": r.weight, "source_order": r.source_order}
        for r in normalized
    ])
    return SourceCandidate(signature_name, normalized, metadata, content_hash)


def _build_signature(
    candidate: SourceCandidate,
    resolver: Any,
    *,
    coverage_threshold: float,
    minimum_n_genes_used: int,
    score_method: str,
) -> Dict[str, Any]:
    raw = list(candidate.records)
    if not raw:
        raise RuntimeError(f"Empty source for {candidate.signature_name}")
    for r in raw:
        if not _nonempty_str(r.gene_symbol):
            raise RuntimeError(f"Empty gene symbol in {candidate.signature_name}")
        if not math.isfinite(float(r.weight)):
            raise RuntimeError(f"Invalid weight in {candidate.signature_name}: {r}")

    is_weighted = any(abs(float(r.weight) - 1.0) > 1e-12 for r in raw) or bool(
        candidate.metadata.get("weight_column")
    )

    resolved_rows: List[Dict[str, Any]] = []
    alias_counts: Dict[str, int] = {}
    for r in raw:
        resolved, status = resolver.resolve(r.gene_symbol)
        alias_counts[status] = alias_counts.get(status, 0) + 1
        if resolved is None or not str(resolved).strip():
            continue
        resolved_rows.append({
            "source_order": int(r.source_order),
            "original_symbol": str(r.gene_symbol).strip(),
            "original_symbol_key": _normalize_gene_key(r.gene_symbol),
            "resolved_symbol": str(resolved).strip(),
            "alias_resolution_status": status,
            "weight": float(r.weight),
        })

    if not resolved_rows:
        raise RuntimeError(f"All genes failed alias resolution for {candidate.signature_name}")

    duplicate_original_keys = sorted({
        x["original_symbol_key"]
        for x in resolved_rows
        if sum(y["original_symbol_key"] == x["original_symbol_key"] for y in resolved_rows) > 1
    })
    by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for row in resolved_rows:
        by_symbol.setdefault(row["resolved_symbol"], []).append(row)

    final_weights: Dict[str, float] = {}
    duplicate_groups: Dict[str, List[Dict[str, Any]]] = {}
    zero_after_sum: List[str] = []
    for sym in sorted(by_symbol):
        rows = by_symbol[sym]
        if len(rows) > 1:
            duplicate_groups[sym] = rows
        if is_weighted:
            w = float(sum(float(r["weight"]) for r in rows))
        else:
            w = 1.0
        if abs(w) <= 0.0:
            zero_after_sum.append(sym)
            continue
        final_weights[sym] = w

    if not final_weights:
        raise RuntimeError(f"No nonzero final gene weights for {candidate.signature_name}")

    if is_weighted:
        theoretical = float(sum(abs(float(r.weight)) for r in raw if _nonempty_str(r.gene_symbol)))
        available = float(sum(abs(w) for w in final_weights.values()))
    else:
        theoretical = float(len({
            _normalize_gene_key(r.gene_symbol) for r in raw if _nonempty_str(r.gene_symbol)
        }))
        available = float(len(final_weights))
    if theoretical <= 0:
        raise RuntimeError(f"Invalid theoretical signature size for {candidate.signature_name}")
    coverage_fraction = available / theoretical
    n_genes_used = len(final_weights)
    eligible = coverage_fraction >= coverage_threshold and n_genes_used >= minimum_n_genes_used
    if not eligible:
        raise RuntimeError(
            f"Signature {candidate.signature_name} fails eligibility: "
            f"coverage={coverage_fraction:.4f} threshold={coverage_threshold:.4f}, "
            f"n_genes_used={n_genes_used}, minimum={minimum_n_genes_used}"
        )

    genes_original = []
    seen_orig: set = set()
    for r in raw:
        g = str(r.gene_symbol).strip()
        k = _normalize_gene_key(g)
        if g and k not in seen_orig:
            seen_orig.add(k)
            genes_original.append(g)

    family = candidate.metadata.get("signature_family")
    if not family:
        family = "hallmark" if candidate.signature_name.startswith("HALLMARK_") else "immune"

    gene_weights_sorted = {k: float(final_weights[k]) for k in sorted(final_weights)}
    ordered_weight_rows = [
        {"resolved_symbol": k, "weight": float(gene_weights_sorted[k])}
        for k in gene_weights_sorted
    ]
    original_rows_sorted = sorted(
        resolved_rows,
        key=lambda x: (int(x["source_order"]), x["original_symbol"], x["resolved_symbol"]),
    )

    return {
        "signature_name": candidate.signature_name,
        "signature_family": family,
        "score_method": score_method,
        "denominator_rule": DENOMINATOR_RULE,
        "genes_original": genes_original,
        "genes_original_with_weights": [
            {
                "source_order": int(x["source_order"]),
                "gene_symbol": x["original_symbol"],
                "resolved_symbol": x["resolved_symbol"],
                "alias_resolution_status": x["alias_resolution_status"],
                "weight": float(x["weight"]),
            }
            for x in original_rows_sorted
        ],
        "genes_resolved": sorted(gene_weights_sorted),
        "gene_weights": gene_weights_sorted,
        "gene_weights_ordered": ordered_weight_rows,
        "is_weighted_source": bool(is_weighted),
        "source_name": candidate.metadata["source_name"],
        "source_version": candidate.metadata["source_version"],
        "source_citation": candidate.metadata["source_citation"],
        "source_url_or_path": candidate.metadata["source_url_or_path"],
        "source_local_path": candidate.metadata["source_local_path"],
        "source_file_hash": candidate.metadata["source_file_hash"],
        "source_file_sha256": candidate.metadata["source_file_sha256"],
        "license_or_usage_note": candidate.metadata.get("license_or_usage_note"),
        "gene_symbol_column": candidate.metadata.get("gene_symbol_column"),
        "weight_column": candidate.metadata.get("weight_column"),
        "minimum_coverage_threshold": float(coverage_threshold),
        "minimum_n_genes_used": int(minimum_n_genes_used),
        "coverage_fraction": float(coverage_fraction),
        "sum_abs_weights_available": float(available),
        "sum_abs_weights_theoretical": float(theoretical),
        "n_genes_original": int(len(genes_original)),
        "n_source_rows": int(len(raw)),
        "n_genes_resolved": int(n_genes_used),
        "n_genes_used": int(n_genes_used),
        "alias_resolution_report": {
            "alias_map_path": resolver.source_path,
            "alias_map_hash": resolver.sha256,
            "alias_map_sha256": resolver.sha256,
            "alias_map_date": resolver.date,
            "alias_map_source": resolver.source,
            "alias_resolution_policy": ALIAS_POLICY,
            "status_counts": dict(sorted(alias_counts.items())),
            "n_unmapped_kept_original": int(alias_counts.get("unmapped_kept_original", 0)),
        },
        "duplicate_resolution_report": {
            "duplicate_resolution_policy": "weighted_sum_after_alias_resolution; unsigned_sources_collapse_to_unique_weight_1",
            "n_duplicate_original_gene_keys": int(len(duplicate_original_keys)),
            "duplicate_original_gene_keys": duplicate_original_keys,
            "n_duplicate_resolved_symbols": int(len(duplicate_groups)),
            "duplicate_resolved_symbols": sorted(duplicate_groups),
            "zero_weight_resolved_symbols_dropped_after_sum": zero_after_sum,
        },
        "alias_map_hash": resolver.sha256,
        "alias_map_sha256": resolver.sha256,
        "alias_map_date": resolver.date,
    }


def _existing_signature_state(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        obj = read_json(path)
    except Exception:
        return "unreadable"
    if obj.get("frozen_final") is True:
        return "frozen_final"
    return "nonfinal_existing"


def _check_downstream(
    settings_dir: Path, benchmark_root: Path, *, force: bool, force_cascade: bool
) -> None:
    if not force:
        return
    downstream: List[Path] = []
    support = settings_dir / "signature_support_training.json"
    if support.exists():
        try:
            support_obj = read_json(support)
            if support_obj.get("frozen_final") is True:
                downstream.append(support)
        except Exception:
            downstream.append(support)
    for p in [
        benchmark_root / "shared" / "rna" / "signature_eligibility.csv",
        benchmark_root / "shared" / "folds" / "fold_assignments.csv",
        benchmark_root / "shared" / "final_cohort_manifest.parquet",
    ]:
        if p.exists():
            downstream.append(p)
    tt = benchmark_root / "shared" / "rna" / "target_tables"
    if tt.exists() and any(tt.glob("*")):
        downstream.append(tt)
    if downstream and not force_cascade:
        raise RuntimeError(
            "Refusing to rebuild signature set because downstream frozen artifacts exist. "
            "Pass --force_cascade only if intentionally invalidating them: "
            + ", ".join(str(p) for p in downstream[:20])
        )


def _build_report(signature_set: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Task B signature set build report",
        "",
        f"- benchmark_id: `{signature_set['benchmark_id']}`",
        f"- signature_set_id: `{signature_set['signature_set_id']}`",
        f"- score_method: `{signature_set['score_method']}`",
        f"- denominator_rule: `{signature_set['denominator_rule']}`",
        f"- alias_map_sha256: `{signature_set['alias_map_sha256']}`",
        f"- signatures_gmt_sha256: `{signature_set['signatures_gmt_sha256']}`",
        f"- signatures_sources_sha256: `{signature_set['signatures_sources_sha256']}`",
        f"- signature_set_content_sha256: `{signature_set['signature_set_content_sha256']}`",
        "",
        "| signature | family | source version | n genes used | coverage | weighted |",
        "|---|---|---|---:|---:|---:|",
    ]
    for sig in signature_set["signatures"]:
        lines.append(
            f"| `{sig['signature_name']}` | {sig['signature_family']} | {sig['source_version']} | "
            f"{sig['n_genes_used']} | {sig['coverage_fraction']:.4f} | {sig['is_weighted_source']} |"
        )
    lines.extend([
        "",
        "All signature sources were read from the in-repo `data/eval_signatures/` panel, "
        "copied by `_3_init_benchmark.py` into `settings/eval_signatures/`. No online "
        "download was performed.",
        "",
        "## Output hashes",
        "",
    ])
    for rel, info in sorted(manifest.get("output_hashes", {}).items()):
        lines.append(f"- `{rel}`: `{info.get('sha256')}`")
    return "\n".join(lines) + "\n"


# =============================================================================
# Validation section
# =============================================================================
# Re-reads the written signature_set_tcga_case_v1.json and checks it against eval_signatures/ and the HGNC alias map


def _fail(errors: List[str], message: str) -> None:
    errors.append(message)


def _require_mapping(obj: Any, name: str, errors: List[str]) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        _fail(errors, f"{name} must be a JSON object")
        return {}
    return obj


def _get_signature_list(signature_set: Mapping[str, Any], errors: List[str]) -> List[Mapping[str, Any]]:
    sigs = signature_set.get("signatures")
    if not isinstance(sigs, list):
        _fail(errors, "signature_set.signatures must be a list")
        return []
    out: List[Mapping[str, Any]] = []
    for i, sig in enumerate(sigs):
        if not isinstance(sig, Mapping):
            _fail(errors, f"signatures[{i}] must be an object")
        else:
            out.append(sig)
    return out


def _check_content_hash(signature_set: Mapping[str, Any], errors: List[str]) -> None:
    stored = signature_set.get("signature_set_content_sha256")
    if not stored:
        _fail(errors, "signature_set_content_sha256 is missing")
        return
    tmp = dict(signature_set)
    tmp.pop("signature_set_content_sha256", None)
    tmp.pop("created_at_utc", None)
    observed = sha256_obj(tmp)
    if observed != stored:
        _fail(errors, f"signature_set_content_sha256 mismatch: observed={observed}, stored={stored}")


def _validate_source_file_hash(sig: Mapping[str, Any], settings_dir: Path, errors: List[str]) -> None:
    h = sig.get("source_file_hash") or sig.get("source_file_sha256")
    if not h:
        _fail(errors, f"{sig.get('signature_name')}: source_file_hash/source_file_sha256 is missing")
        return
    rel_or_url = sig.get("source_local_path") or sig.get("source_url_or_path")
    if not rel_or_url:
        _fail(errors, f"{sig.get('signature_name')}: source_url_or_path/source_local_path is missing")
        return
    text = str(rel_or_url)
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text):
        return
    p = Path(text)
    if not p.is_absolute():
        p = settings_dir / p
    if not p.exists() or not p.is_file():
        _fail(errors, f"{sig.get('signature_name')}: local source file is missing: {p}")
        return
    observed = sha256_path(p)
    if observed != h:
        _fail(errors, f"{sig.get('signature_name')}: source file hash mismatch for {p}: observed={observed}, stored={h}")


def _validate_signature(sig: Mapping[str, Any], *, coverage_threshold: float, minimum_n_genes_used: int, alias_hash: str, settings_dir: Path, errors: List[str]) -> Dict[str, Any]:
    name = sig.get("signature_name")
    if not name:
        _fail(errors, "A signature is missing signature_name")
        name = "<missing>"
    score_method = sig.get("score_method")
    if score_method != SCORE_METHOD:
        _fail(errors, f"{name}: score_method must be {SCORE_METHOD}, got {score_method}")
    if sig.get("denominator_rule") != DENOMINATOR_RULE:
        _fail(errors, f"{name}: denominator_rule must be {DENOMINATOR_RULE}")

    genes_original = sig.get("genes_original")
    genes_resolved = sig.get("genes_resolved")
    gene_weights = sig.get("gene_weights")
    if not isinstance(genes_original, list) or not genes_original:
        _fail(errors, f"{name}: genes_original must be a non-empty list")
    if not isinstance(genes_resolved, list) or not genes_resolved:
        _fail(errors, f"{name}: genes_resolved must be a non-empty list")
    if not isinstance(gene_weights, Mapping) or not gene_weights:
        _fail(errors, f"{name}: gene_weights must be a non-empty mapping")
        gene_weights = {}

    weights: Dict[str, float] = {}
    for g, w in gene_weights.items():
        gs = str(g).strip()
        if not gs:
            _fail(errors, f"{name}: empty resolved symbol in gene_weights")
            continue
        try:
            wf = float(w)
        except Exception:
            _fail(errors, f"{name}: non-numeric weight for {g}: {w!r}")
            continue
        if not math.isfinite(wf):
            _fail(errors, f"{name}: non-finite weight for {g}: {w!r}")
            continue
        if abs(wf) <= 0.0:
            _fail(errors, f"{name}: zero final weight for {g}")
        weights[gs] = wf

    if isinstance(genes_resolved, list):
        resolved_as_str = [str(x) for x in genes_resolved]
        if len(resolved_as_str) != len(set(resolved_as_str)):
            _fail(errors, f"{name}: genes_resolved contains duplicates")
        if set(resolved_as_str) != set(weights):
            _fail(errors, f"{name}: genes_resolved and gene_weights keys differ")

    if not sig.get("source_name"):
        _fail(errors, f"{name}: source_name is missing")
    if not sig.get("source_version"):
        _fail(errors, f"{name}: source_version is missing")
    if not sig.get("source_citation"):
        _fail(errors, f"{name}: source_citation is missing")
    if not (sig.get("source_file_hash") or sig.get("source_file_sha256")):
        _fail(errors, f"{name}: source file hash is missing")
    if "minimum_coverage_threshold" not in sig:
        _fail(errors, f"{name}: minimum_coverage_threshold is missing")
    else:
        try:
            mct = float(sig["minimum_coverage_threshold"])
            if abs(mct - coverage_threshold) > 1e-12:
                _fail(errors, f"{name}: minimum_coverage_threshold {mct} != config threshold {coverage_threshold}")
        except Exception:
            _fail(errors, f"{name}: minimum_coverage_threshold is not numeric")

    try:
        cov = float(sig.get("coverage_fraction"))
        if not math.isfinite(cov):
            raise ValueError
        if cov < coverage_threshold:
            _fail(errors, f"{name}: coverage_fraction {cov:.4f} < threshold {coverage_threshold:.4f}")
    except Exception:
        _fail(errors, f"{name}: coverage_fraction missing or nonnumeric")
        cov = float("nan")

    try:
        n_used = int(sig.get("n_genes_used"))
        if n_used < minimum_n_genes_used:
            _fail(errors, f"{name}: n_genes_used {n_used} < minimum {minimum_n_genes_used}")
        if gene_weights and n_used != len(gene_weights):
            _fail(errors, f"{name}: n_genes_used does not equal len(gene_weights)")
    except Exception:
        _fail(errors, f"{name}: n_genes_used missing or invalid")
        n_used = -1

    alias_report = _require_mapping(sig.get("alias_resolution_report"), f"{name}.alias_resolution_report", errors)
    duplicate_report = _require_mapping(sig.get("duplicate_resolution_report"), f"{name}.duplicate_resolution_report", errors)
    if alias_report:
        found_hash = sig.get("alias_map_sha256") or sig.get("alias_map_hash") or alias_report.get("alias_map_sha256") or alias_report.get("alias_map_hash")
        if found_hash != alias_hash:
            _fail(errors, f"{name}: alias map hash mismatch: {found_hash} != {alias_hash}")
        if not (sig.get("alias_map_date") or alias_report.get("alias_map_date")):
            _fail(errors, f"{name}: alias_map_date is missing")
    if duplicate_report and not duplicate_report.get("duplicate_resolution_policy"):
        _fail(errors, f"{name}: duplicate_resolution_policy is missing")

    ordered = sig.get("gene_weights_ordered")
    if ordered is not None:
        if not isinstance(ordered, list):
            _fail(errors, f"{name}: gene_weights_ordered must be a list when present")
        elif not all(isinstance(x, Mapping) for x in ordered):
            _fail(errors, f"{name}: gene_weights_ordered entries must all be objects")
        elif [str(x.get("resolved_symbol")) for x in ordered] != sorted(weights):
            _fail(errors, f"{name}: gene_weights_ordered is not sorted by resolved_symbol")
        else:
            for x in ordered:
                sym = str(x.get("resolved_symbol"))
                try:
                    ow = float(x.get("weight"))
                except Exception:
                    _fail(errors, f"{name}: invalid gene_weights_ordered weight for {sym}: {x.get('weight')!r}")
                    continue
                if sym in weights and abs(ow - weights[sym]) > 1e-12:
                    _fail(errors, f"{name}: ordered weight mismatch for {sym}: {ow} != {weights[sym]}")

    _validate_source_file_hash(sig, settings_dir, errors)
    return {
        "signature_name": str(name),
        "n_genes_used": n_used,
        "coverage_fraction": cov,
        "source_name": sig.get("source_name"),
        "source_version": sig.get("source_version"),
        "source_file_hash": sig.get("source_file_hash") or sig.get("source_file_sha256"),
    }


def _make_validation_report(status: str, benchmark_id: str, errors: List[str], warnings: List[str], summary: List[Dict[str, Any]], manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Task B signature set validation report",
        "",
        f"- status: `{status}`",
        f"- benchmark_id: `{benchmark_id}`",
        f"- signature_set_file_sha256: `{manifest.get('signature_set_file_sha256')}`",
        f"- alias_map_sha256: `{manifest.get('alias_map_sha256')}`",
        "",
    ]
    if errors:
        lines.append("## Errors")
        lines.append("")
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    lines.extend([
        "## Signature summary",
        "",
        "| signature | n genes used | coverage | source | version |",
        "|---|---:|---:|---|---|",
    ])
    for row in summary:
        cov = row.get("coverage_fraction")
        cov_s = "NA" if cov is None or not isinstance(cov, (int, float)) or not math.isfinite(float(cov)) else f"{float(cov):.4f}"
        lines.append(f"| `{row.get('signature_name')}` | {row.get('n_genes_used')} | {cov_s} | {row.get('source_name')} | {row.get('source_version')} |")
    return "\n".join(lines) + "\n"


def _run_validation(
    settings_dir: Path,
    benchmark_root: Path,
    benchmark_id: str,
    script_name: str,
    args_dict: Mapping[str, Any],
) -> Tuple[List[str], List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """Re-read and validate signature_set_tcga_case_v1.json.

    Returns (errors, warnings, per-signature summary, signature_validation_manifest dict).
    """
    errors: List[str] = []
    warnings: List[str] = []

    eval_signatures_dir = settings_dir / "eval_signatures"
    if not eval_signatures_dir.is_dir():
        raise FileNotFoundError(eval_signatures_dir)
    signatures_gmt_path = eval_signatures_dir / "signatures.gmt"
    signatures_sources_path = eval_signatures_dir / "sources.json"
    if not signatures_gmt_path.exists():
        raise FileNotFoundError(signatures_gmt_path)
    if not signatures_sources_path.exists():
        raise FileNotFoundError(signatures_sources_path)
    required = _normalized_gmt_required_signatures(signatures_gmt_path)
    coverage_threshold = COVERAGE_THRESHOLD
    minimum_n_genes_used = MINIMUM_N_GENES_USED

    signature_path = settings_dir / REQUIRED_OUTPUT
    if not signature_path.exists():
        raise FileNotFoundError(signature_path)
    signature_set = read_json(signature_path)
    if not isinstance(signature_set, Mapping):
        errors.append("signature_set_tcga_case_v1.json root must be an object")
        signature_set = {}

    if signature_set.get("frozen_final") is not True:
        errors.append("signature set is not frozen_final; the build step did not finalize the file")
    if signature_set.get("score_method") != SCORE_METHOD:
        errors.append(f"signature_set.score_method must be {SCORE_METHOD}")
    if signature_set.get("denominator_rule") != DENOMINATOR_RULE:
        errors.append(f"signature_set.denominator_rule must be {DENOMINATOR_RULE}")
    try:
        cov_ss = float(signature_set.get("coverage_threshold"))
        if abs(cov_ss - COVERAGE_THRESHOLD) > 1e-12:
            errors.append(
                f"signature_set.coverage_threshold {cov_ss} must equal {COVERAGE_THRESHOLD}"
            )
    except (TypeError, ValueError):
        errors.append("signature_set.coverage_threshold is missing or not numeric")
    try:
        mng_ss = int(signature_set.get("minimum_n_genes_used"))
        if mng_ss != MINIMUM_N_GENES_USED:
            errors.append(
                f"signature_set.minimum_n_genes_used {mng_ss} must equal {MINIMUM_N_GENES_USED}"
            )
    except (TypeError, ValueError):
        errors.append("signature_set.minimum_n_genes_used is missing or not numeric")
    ss_required = [
        _normalize_signature_name(x)
        for x in (signature_set.get("required_signatures") or [])
    ]
    if ss_required != required:
        errors.append(
            "signature_set.required_signatures does not match eval_signatures/signatures.gmt: "
            f"signature_set={ss_required}, gmt={required}"
        )
    if signature_set.get("signature_set_id") != "signature_set_tcga_case_v1":
        errors.append("signature_set_id must be signature_set_tcga_case_v1")
    _check_content_hash(signature_set, errors)

    alias_candidates = sorted(settings_dir.glob("gene_alias_map_hgnc_*.tsv"))
    if len(alias_candidates) != 1:
        errors.append(f"Expected exactly one benchmark alias map TSV, found {len(alias_candidates)}")
        alias_hash = None
        alias_path = None
    else:
        alias_path = alias_candidates[0]
        alias_hash = sha256_path(alias_path)
        found = signature_set.get("alias_map_sha256") or signature_set.get("alias_map_hash")
        if found != alias_hash:
            errors.append(f"Signature set alias hash mismatch: {found} != benchmark alias hash {alias_hash}")
        if not signature_set.get("alias_map_date"):
            errors.append("signature_set alias_map_date is missing")

    sigs = _get_signature_list(signature_set, errors)
    names = [str(s.get("signature_name")) for s in sigs if s.get("signature_name")]
    norm_names = [_normalize_signature_name(n) for n in names]
    if len(norm_names) != len(set(norm_names)):
        errors.append("Duplicate signature_name values in signature set")
    missing = sorted(set(required) - set(norm_names))
    extra = sorted(set(norm_names) - set(required))
    if missing:
        errors.append(f"Missing configured signatures: {missing}")
    if extra:
        errors.append(f"Extra signatures not configured: {extra}")

    root_names = signature_set.get("signature_names")
    if root_names is not None:
        root_norm = [_normalize_signature_name(x) for x in root_names]
        if root_norm != norm_names:
            errors.append("signature_names does not match signatures order")
    by_name = signature_set.get("signatures_by_name")
    if by_name is not None:
        if not isinstance(by_name, Mapping):
            errors.append("signatures_by_name must be a mapping")
        elif set(_normalize_signature_name(k) for k in by_name.keys()) != set(norm_names):
            errors.append("signatures_by_name keys do not match signatures")

    summary: List[Dict[str, Any]] = []
    for sig in sigs:
        summary.append(_validate_signature(
            sig,
            coverage_threshold=coverage_threshold,
            minimum_n_genes_used=minimum_n_genes_used,
            alias_hash=alias_hash or "<missing>",
            settings_dir=settings_dir,
            errors=errors,
        ))

    signatures_gmt_sha256 = sha256_path(signatures_gmt_path)
    signatures_sources_sha256 = sha256_path(signatures_sources_path)
    gmt_stored_sha = signature_set.get("signatures_gmt_sha256")
    sources_stored_sha = signature_set.get("signatures_sources_sha256")
    if not gmt_stored_sha:
        errors.append("signature_set.signatures_gmt_sha256 is missing")
    elif signatures_gmt_sha256 != gmt_stored_sha:
        errors.append(
            f"signatures.gmt hash mismatch: file={signatures_gmt_sha256}, "
            f"signature_set={gmt_stored_sha}"
        )
    if not sources_stored_sha:
        errors.append("signature_set.signatures_sources_sha256 is missing")
    elif signatures_sources_sha256 != sources_stored_sha:
        errors.append(
            f"sources.json hash mismatch: file={signatures_sources_sha256}, "
            f"signature_set={sources_stored_sha}"
        )

    manifest = manifest_header(script_name, args_dict)
    status = "ok" if not errors else "failed"
    manifest.update({
        "status": status,
        "benchmark_id": benchmark_id,
        "benchmark_root": str(benchmark_root),
        "settings_dir": str(settings_dir),
        "signature_set_path": str(signature_path),
        "signature_set_file_sha256": sha256_path(signature_path),
        "signature_set_content_sha256": signature_set.get("signature_set_content_sha256"),
        "signatures_gmt_path": str(signatures_gmt_path),
        "signatures_gmt_sha256": signatures_gmt_sha256,
        "signatures_sources_path": str(signatures_sources_path),
        "signatures_sources_sha256": signatures_sources_sha256,
        "required_signatures": required,
        "observed_signatures": norm_names,
        "coverage_threshold": coverage_threshold,
        "minimum_n_genes_used": minimum_n_genes_used,
        "alias_map_path": str(alias_path) if alias_hash else None,
        "alias_map_sha256": alias_hash,
        "summary": summary,
        "errors": errors,
        "warnings": warnings,
    })
    return errors, warnings, summary, manifest


# =============================================================================
# Main
# =============================================================================


def main(args) -> int:

    data_root = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data')
    benchmark_root = data_root / "benchmarks" / args.benchmark_id
    settings_dir = benchmark_root / "settings"
    if not settings_dir.exists():
        raise FileNotFoundError(settings_dir)

    output_path = settings_dir / REQUIRED_OUTPUT
    existing_state = _existing_signature_state(output_path)
    if existing_state == "frozen_final" and not args.force:
        raise FileExistsError(
            f"Frozen signature set already exists: {output_path}. Use --force to rebuild."
        )
    if existing_state in {"nonfinal_existing", "unreadable"} and not args.force:
        raise FileExistsError(
            f"Non-final signature set file already exists: {output_path}. "
            f"Use --force if intentional."
        )
    _check_downstream(settings_dir, benchmark_root, force=args.force, force_cascade=args.force_cascade)
    overwrite_outputs = bool(args.force)

    alias_candidates = sorted(settings_dir.glob("gene_alias_map_hgnc_*.tsv"))
    if len(alias_candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one benchmark HGNC alias map; found {len(alias_candidates)}"
        )
    alias_map_path = alias_candidates[0]
    resolver = load_alias_resolver(alias_map_path, source="benchmark_frozen_HGNC_alias_map")

    eval_signatures_dir = settings_dir / "eval_signatures"
    if not eval_signatures_dir.is_dir():
        raise FileNotFoundError(
            f"Frozen eval_signatures panel is required: {eval_signatures_dir}. "
            f"Run _3_init_benchmark.py first."
        )
    signatures_gmt_path = eval_signatures_dir / "signatures.gmt"
    signatures_sources_path = eval_signatures_dir / "sources.json"
    signatures_gmt_sha256 = sha256_path(signatures_gmt_path)
    signatures_sources_sha256 = sha256_path(signatures_sources_path)
    signatures_gmt_relpath = str(signatures_gmt_path.relative_to(settings_dir))

    gmt = _parse_gmt(signatures_gmt_path)
    sources_json = _load_sources_json(signatures_sources_path)

    # Required signatures: every signature in the GMT, in GMT order, normalized.
    required_signatures: List[str] = []
    seen_norm: Dict[str, str] = {}
    for sig_id in gmt.keys():
        norm = _normalize_signature_name(sig_id)
        if norm in seen_norm:
            raise RuntimeError(
                f"Two GMT entries normalize to the same signature_id "
                f"({seen_norm[norm]!r} and {sig_id!r}); refusing to silently merge."
            )
        seen_norm[norm] = sig_id
        required_signatures.append(norm)

    missing_in_sources = [s for s in gmt.keys() if s not in sources_json]
    if missing_in_sources:
        # Missing metadata is not fatal (placeholder source fields)
        pass

    signatures: List[Dict[str, Any]] = []
    for sig_id_norm in required_signatures:
        sig_id_raw = seen_norm[sig_id_norm]
        genes = gmt[sig_id_raw]
        src_meta = sources_json.get(sig_id_raw, {})
        metadata = _source_metadata_from_sources_json(
            sig_id_norm,
            src_meta,
            signatures_gmt_relpath=signatures_gmt_relpath,
            signatures_gmt_sha256=signatures_gmt_sha256,
        )
        records = [
            GeneRecord(sig_id_norm, g, 1.0, i) for i, g in enumerate(genes)
        ]
        candidate = _candidate_from_records(sig_id_norm, records, metadata)
        signatures.append(_build_signature(
            candidate,
            resolver,
            coverage_threshold=COVERAGE_THRESHOLD,
            minimum_n_genes_used=MINIMUM_N_GENES_USED,
            score_method=SCORE_METHOD,
        ))

    signatures_by_name = {s["signature_name"]: s for s in signatures}
    signature_set: Dict[str, Any] = {
        "status": "frozen_final",
        "frozen_final": True,
        "benchmark_id": args.benchmark_id,
        "signature_set_id": DEFAULT_SIGNATURE_SET_ID,
        "signature_set_version": "tcga_case_v1",
        "created_at_utc": None,
        "score_method": SCORE_METHOD,
        "denominator_rule": DENOMINATOR_RULE,
        "score_formula": "score = sum(w_g * z_g) / sum(abs(w_g))",
        "coverage_threshold": float(COVERAGE_THRESHOLD),
        "minimum_n_genes_used": int(MINIMUM_N_GENES_USED),
        "required_signatures": required_signatures,
        "signature_names": [s["signature_name"] for s in signatures],
        "signatures": signatures,
        "signatures_by_name": signatures_by_name,
        "alias_map_path": str(alias_map_path),
        "alias_map_hash": resolver.sha256,
        "alias_map_sha256": resolver.sha256,
        "alias_map_date": resolver.date,
        "alias_map_source": resolver.source,
        "alias_resolution_policy": ALIAS_POLICY,
        "duplicate_symbol_policy": "weighted_sum_after_alias_resolution; unsigned_sources_collapse_to_unique_weight_1",
        "signatures_gmt_path": str(signatures_gmt_path),
        "signatures_gmt_sha256": signatures_gmt_sha256,
        "signatures_sources_path": str(signatures_sources_path),
        "signatures_sources_sha256": signatures_sources_sha256,
        "signatures_missing_metadata_in_sources_json": missing_in_sources,
        "no_online_downloads_performed": True,
    }
    signature_set["created_at_utc"] = manifest_header(Path(__file__).name, vars(args))["created_at_utc"]
    content_for_hash = dict(signature_set)
    content_for_hash.pop("created_at_utc", None)
    content_hash = sha256_obj(content_for_hash)
    signature_set["signature_set_content_sha256"] = content_hash

    if sorted(signatures_by_name) != sorted(required_signatures):
        raise RuntimeError("Internal error: required signatures and built signatures differ")
    for sig in signatures:
        if sig["coverage_fraction"] < COVERAGE_THRESHOLD or sig["n_genes_used"] < MINIMUM_N_GENES_USED:
            raise RuntimeError(f"Internal eligibility check failed for {sig['signature_name']}")

    write_json_atomic(output_path, signature_set, overwrite=overwrite_outputs)

    # ---- Internal validation pass (re-reads what was just written) ----
    val_errors, val_warnings, val_summary, val_manifest = _run_validation(
        settings_dir=settings_dir,
        benchmark_root=benchmark_root,
        benchmark_id=args.benchmark_id,
        script_name=Path(__file__).name,
        args_dict=vars(args),
    )
    # Snapshot of the settings_dir output hashes
    val_manifest["output_hashes"] = collect_output_hashes(settings_dir)
    write_json_atomic(
        settings_dir / "signature_validation_manifest.json",
        val_manifest,
        overwrite=overwrite_outputs,
    )
    val_status = "ok" if not val_errors else "failed"
    val_report = _make_validation_report(
        val_status, args.benchmark_id, val_errors, val_warnings, val_summary, val_manifest
    )
    write_text_atomic(
        settings_dir / "signature_validation_report.md",
        val_report,
        overwrite=overwrite_outputs,
    )
    if val_errors:
        message = "Signature set validation failed:\n" + "\n".join(f"- {e}" for e in val_errors[:50])
        if len(val_errors) > 50:
            message += f"\n... {len(val_errors) - 50} more errors"
        raise RuntimeError(message)

    # ---- Build manifest and report (only written once validation passes) ----
    build_manifest = manifest_header(Path(__file__).name, vars(args))
    build_manifest.update({
        "benchmark_id": args.benchmark_id,
        "benchmark_root": str(benchmark_root),
        "settings_dir": str(settings_dir),
        "signature_set_path": str(output_path),
        "signature_set_content_sha256": content_hash,
        "signature_set_file_sha256": sha256_path(output_path),
        "signatures_gmt_path": str(signatures_gmt_path),
        "signatures_gmt_sha256": signatures_gmt_sha256,
        "signatures_sources_path": str(signatures_sources_path),
        "signatures_sources_sha256": signatures_sources_sha256,
        "alias_map_path": str(alias_map_path),
        "alias_map_sha256": resolver.sha256,
        "alias_map_date": resolver.date,
        "score_method": SCORE_METHOD,
        "denominator_rule": DENOMINATOR_RULE,
        "coverage_threshold": COVERAGE_THRESHOLD,
        "minimum_n_genes_used": MINIMUM_N_GENES_USED,
        "required_signatures": required_signatures,
        "built_signature_summary": [
            {
                "signature_name": s["signature_name"],
                "n_genes_used": s["n_genes_used"],
                "coverage_fraction": s["coverage_fraction"],
                "source_name": s["source_name"],
                "source_version": s["source_version"],
                "source_file_hash": s["source_file_hash"],
            }
            for s in signatures
        ],
        "output_hashes": collect_output_hashes(settings_dir),
    })
    write_json_atomic(
        settings_dir / "signature_build_manifest.json",
        build_manifest,
        overwrite=overwrite_outputs,
    )
    report = _build_report(signature_set, build_manifest)
    write_text_atomic(
        settings_dir / "signature_build_report.md",
        report,
        overwrite=overwrite_outputs,
    )

    print(json.dumps({
        "status": "ok",
        "benchmark_id": args.benchmark_id,
        "signature_set_path": str(output_path),
        "signature_set_file_sha256": sha256_path(output_path),
        "n_signatures": len(signatures),
        "validation_status": val_status,
    }, indent=2))
    return 0


if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="Build and validate the frozen TCGA Task B transcriptomic signature set "
                    "from settings/eval_signatures/{signatures.gmt, sources.json}.",
    )

    ap.add_argument("--benchmark_id", type=str,
                    required=True,
                    help="Benchmark ID whose settings/eval_signatures/ holds the frozen panel.")

    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing frozen signature set (and its build/validation outputs).")

    ap.add_argument("--force_cascade", action="store_true",
                    help="Required with --force if downstream frozen artifacts exist.")

    args = ap.parse_args()
    raise SystemExit(main(args))
