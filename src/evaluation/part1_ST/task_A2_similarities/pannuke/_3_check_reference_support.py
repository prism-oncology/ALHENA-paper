#!/usr/bin/env python
"""
Count the coarse-v2 classes of the scRNA-seq reference of each PanNuke tissue (exact-name match), and retain the
tissues whose classes all reach the coverage floor and whose scConcept reference embedding exists.
Runs in the `histost_eval_a2` env.

Saved in <ROOT_HistoST> / eval / eval_datasets / pannuke_eval_data / characterization /
    - reference_support_v2.csv: one row per tissue (class counts, A.2 audit counts, retained)
    - reference_support_v2.json: same rows, parameters and retained tissues
    - reference_support.log
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Project imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src_A2.common import PROJECT_ROOT, configure_logger, ensure_dir, write_dataframe, write_json
from src_A2.common import src_import
from src_A2.coverage import COVERAGE_DEFAULTS
from src_A2.label_granularity import apply_label_map, load_coarse_ct_mapping
from src_A2.references import (
    load_reference_mapping,
    read_obs_columns,
    resolve_reference_for_tissue,
)
from pannuke.pannuke_common import (
    COARSE_V2_SCORABLE_CLASSES,
    COARSE_V2_UNSCORABLE_LABEL,
    PANNUKE_TISSUE_TO_HISTOST,
    PANNUKE_TISSUES_WITHOUT_REFERENCE,
)

get_full_path = src_import("get_full_path")

logger = logging.getLogger(__name__)

REFERENCE_SUPPORT_SCHEMA_VERSION = 1

DEFAULT_CT_CAT_ID = "ct_1"
DEFAULT_COARSE_MAPPING_VERSION = "v2"
DEFAULT_SCCONCEPT_TRAINING_ID = "training_7"
DEFAULT_AUDIT_EVAL_ID = "eval_2"

# Minimum reference cells per class (A.2 coverage settings)
MIN_REF_CELLS_PER_CLASS = int(COVERAGE_DEFAULTS["min_ref_cells_per_gt_class"])

# Tissues whose smallest class is below this multiple of the floor are flagged as thin_margin
THIN_MARGIN_FACTOR = 2.0


def repo_ref_mapping_path() -> Path:
    """Path of data/eval_scRNAseq_ref/ref_tissue_files.json in the repo (not under ROOT_PATH)."""
    root = PROJECT_ROOT or Path(__file__).resolve().parents[5]
    return root / "data" / "eval_scRNAseq_ref" / "ref_tissue_files.json"


def resolve_repo_mapping_to_absolute(mapping: Dict[str, str], adatas_dir: Path) -> Dict[str, str]:
    """Turn the bare .h5ad file names of the repo mapping into absolute paths under adatas_dir."""
    resolved: Dict[str, str] = {}
    for tissue, value in mapping.items():
        p = Path(str(value))
        p = (adatas_dir / p).resolve() if not p.is_absolute() else p.resolve()
        resolved[str(tissue)] = str(p)
    return resolved


def load_shipped_audit_counts(eval_id: str) -> Dict[str, Any]:
    """
    Per-class reference counts after the full A.2 row filter, read from the A.2 reference audits of eval_id.
    Returns a status dict.
    """
    audit_dir = get_full_path(
        "eval", "A-2_eval_st_predictions", eval_id, "shared", "reference_audits"
    )
    if not Path(audit_dir).is_dir():
        return {"status": "unavailable", "reason": f"audit folder not found: {audit_dir}"}

    by_tissue: Dict[str, Dict[str, int]] = {}
    stems: Dict[str, str] = {}
    for path in sorted(Path(audit_dir).glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001 - a malformed sidecar is reported, not fatal
            logger.info("Skipping unreadable audit %s: %s", path.name, exc)
            continue
        if str(payload.get("label_granularity")) != "coarse":
            continue
        tissue = str(payload.get("tissue_HistoST", ""))
        counts = payload.get("reference_label_counts") or {}
        if tissue and counts:
            by_tissue[tissue] = {str(k): int(v) for k, v in counts.items()}
            stems[tissue] = str(payload.get("reference_stem", ""))
    if not by_tissue:
        return {"status": "unavailable", "reason": f"no coarse reference audits under {audit_dir}"}
    return {
        "status": "ok",
        "eval_id": eval_id,
        "path": str(audit_dir),
        "n_tissues": len(by_tissue),
        "counts_by_tissue": by_tissue,
        "stems_by_tissue": stems,
        "note": (
            "Realized counts after A.2's full reference row filter (mapped label present and "
            "non-empty expression), the coarse collapse and the unscorable-label drop."
        ),
    }


def audit_one_tissue(
    pannuke_tissue: str,
    histost_key: str,
    mapping: Dict[str, str],
    mapping_path: Path,
    coarse_map: Dict[str, str],
    mapped_label_column: str,
    scconcept_dir: Path,
) -> Dict[str, Any]:
    """Coarse-v2 class counts of one tissue's reference (from its obs) and scConcept embedding status."""
    spec = resolve_reference_for_tissue(histost_key, mapping, mapping_path)
    obs = read_obs_columns(spec.path, [mapped_label_column])
    mapped = apply_label_map(obs[mapped_label_column], coarse_map)
    counts_all = Counter(mapped.tolist() if hasattr(mapped, "tolist") else list(mapped))
    n_unscorable = int(counts_all.get(COARSE_V2_UNSCORABLE_LABEL, 0))
    per_class = {c: int(counts_all.get(c, 0)) for c in COARSE_V2_SCORABLE_CLASSES}

    embedding_path = scconcept_dir / f"{spec.stem}.h5"
    smallest = min(per_class.values()) if per_class else 0
    clears = all(v >= MIN_REF_CELLS_PER_CLASS for v in per_class.values())

    return {
        "pannuke_tissue": pannuke_tissue,
        "tissue_HistoST": histost_key,
        "reference_stem": spec.stem,
        "reference_h5ad": str(spec.path),
        "reference_h5ad_size_bytes": int(spec.path.stat().st_size),
        "n_rows_total": int(len(obs)),
        "n_rows_unscorable_dropped": n_unscorable,
        "n_rows_scorable_upper_bound": int(sum(per_class.values())),
        **{f"upper_bound_{c}": per_class[c] for c in COARSE_V2_SCORABLE_CLASSES},
        "smallest_class_upper_bound": int(smallest),
        "min_ref_cells_per_class": MIN_REF_CELLS_PER_CLASS,
        "clears_floor_on_upper_bound": bool(clears),
        "margin_factor_upper_bound": (
            float(smallest) / MIN_REF_CELLS_PER_CLASS if MIN_REF_CELLS_PER_CLASS else float("nan")
        ),
        "scconcept_embedding_path": str(embedding_path),
        "scconcept_embedding_present": bool(embedding_path.is_file()),
        "scconcept_embedding_size_bytes": (
            int(embedding_path.stat().st_size) if embedding_path.is_file() else 0
        ),
    }


def main(args: argparse.Namespace) -> int:
    out_root = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else get_full_path("eval", "eval_datasets", "pannuke_eval_data")
    )
    char_dir = Path(out_root) / "characterization"
    ensure_dir(char_dir)

    log = configure_logger(char_dir / "reference_support.log", name="A2_pannuke_reference_support")

    mapping_path = repo_ref_mapping_path()
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Reference mapping not found in the repo: {mapping_path}")
    adatas_dir = Path(get_full_path("eval", "eval_datasets", "scRNAseq_references", "adatas"))
    mapping = resolve_repo_mapping_to_absolute(load_reference_mapping(mapping_path), adatas_dir)
    log.info("Reference mapping: %s (%d keys) -> %s", mapping_path, len(mapping), adatas_dir)

    coarse_map = load_coarse_ct_mapping(args.ct_cat_id, args.coarse_mapping_version)
    mapped_label_column = f"{args.ct_cat_id}_cell_type"
    log.info(
        "Coarse map %s/%s -> %d keys; reference label column %s",
        args.ct_cat_id, args.coarse_mapping_version, len(coarse_map), mapped_label_column,
    )

    missing_keys = sorted(set(PANNUKE_TISSUE_TO_HISTOST.values()) - set(mapping))
    if missing_keys:
        raise KeyError(
            f"tissue_HistoST keys {missing_keys} of PANNUKE_TISSUE_TO_HISTOST are absent from {mapping_path}."
        )

    scconcept_dir = Path(
        get_full_path(
            "eval", "eval_datasets", "scRNAseq_references", args.raw_reference_embedding_subdir,
            args.scconcept_training_id,
        )
    )
    log.info("scConcept embedding folder: %s (exists=%s)", scconcept_dir, scconcept_dir.is_dir())

    shipped = load_shipped_audit_counts(args.audit_eval_id)
    log.info("Shipped audit join: %s", shipped.get("status"))

    rows: List[Dict[str, Any]] = []
    for pannuke_tissue in sorted(PANNUKE_TISSUE_TO_HISTOST):
        histost_key = PANNUKE_TISSUE_TO_HISTOST[pannuke_tissue]
        log.info("Auditing %s -> %s", pannuke_tissue, histost_key)
        row = audit_one_tissue(
            pannuke_tissue=pannuke_tissue,
            histost_key=histost_key,
            mapping=mapping,
            mapping_path=mapping_path,
            coarse_map=coarse_map,
            mapped_label_column=mapped_label_column,
            scconcept_dir=scconcept_dir,
        )

        realized = (shipped.get("counts_by_tissue") or {}).get(histost_key)
        row["shipped_audit_available"] = realized is not None
        for c in COARSE_V2_SCORABLE_CLASSES:
            row[f"realized_{c}"] = int(realized[c]) if realized and c in realized else -1
        if realized:
            ub = row["n_rows_scorable_upper_bound"]
            rz = int(sum(realized.values()))
            row["realized_total"] = rz
            row["filter_retention_ratio"] = float(rz) / ub if ub else float("nan")
            row["smallest_class_realized"] = int(min(realized.values()))
            row["clears_floor_on_realized"] = bool(
                all(int(v) >= MIN_REF_CELLS_PER_CLASS for v in realized.values())
                and set(realized) >= set(COARSE_V2_SCORABLE_CLASSES)
            )
        else:
            row["realized_total"] = -1
            row["filter_retention_ratio"] = float("nan")
            row["smallest_class_realized"] = -1
            row["clears_floor_on_realized"] = None

        thin = row["margin_factor_upper_bound"] < THIN_MARGIN_FACTOR
        row["thin_margin"] = bool(thin)
        row["retained"] = bool(
            row["clears_floor_on_upper_bound"] and row["scconcept_embedding_present"]
        )
        row["retained_reason"] = (
            "ok" if row["retained"]
            else (
                "class support below floor even on the upper bound"
                if not row["clears_floor_on_upper_bound"]
                else "scConcept reference embedding missing"
            )
        )
        rows.append(row)
        log.info(
            "  %-11s smallest class (upper bound) %8d  clears=%s  emb=%s  retained=%s",
            pannuke_tissue, row["smallest_class_upper_bound"],
            row["clears_floor_on_upper_bound"], row["scconcept_embedding_present"], row["retained"],
        )

    table = pd.DataFrame(rows).sort_values("pannuke_tissue").reset_index(drop=True)
    write_dataframe(table, char_dir / "reference_support_v2.csv")

    retained = table.loc[table["retained"], "pannuke_tissue"].tolist()
    thin = table.loc[table["retained"] & table["thin_margin"], "pannuke_tissue"].tolist()
    ratios = table.loc[table["filter_retention_ratio"].notna(), "filter_retention_ratio"]

    payload = {
        "reference_support_schema_version": REFERENCE_SUPPORT_SCHEMA_VERSION,
        "ct_cat_id": args.ct_cat_id,
        "coarse_mapping_version": args.coarse_mapping_version,
        "scorable_classes": list(COARSE_V2_SCORABLE_CLASSES),
        "min_ref_cells_per_class": MIN_REF_CELLS_PER_CLASS,
        "reference_mapping_path": str(mapping_path),
        "scconcept_training_id": args.scconcept_training_id,
        "scconcept_embedding_dir": str(scconcept_dir),
        "shipped_audit_join": {k: v for k, v in shipped.items() if k != "counts_by_tissue"},
        "counting_note": (
            "Counts are an upper bound: the coarse-v2 collapse and the two-sided Unknown drop are "
            "applied, but not A.2's non-empty-expression row filter. `filter_retention_ratio` "
            "gives the measured attrition on tissues that have an A.2 reference audit."
        ),
        "filter_retention_ratio_summary": (
            {
                "n_tissues": int(ratios.size),
                "min": float(ratios.min()),
                "median": float(ratios.median()),
                "max": float(ratios.max()),
            }
            if ratios.size
            else {"n_tissues": 0}
        ),
        "candidate_tissues": sorted(PANNUKE_TISSUE_TO_HISTOST),
        "excluded_no_exact_name_reference": list(PANNUKE_TISSUES_WITHOUT_REFERENCE),
        "retained_tissues": retained,
        "n_retained": len(retained),
        "thin_margin_tissues": thin,
        "thin_margin_factor": THIN_MARGIN_FACTOR,
        "rows": table.to_dict(orient="records"),
    }
    write_json(char_dir / "reference_support_v2.json", payload)

    # Retention ratio of the A.2 row filter on the audited references
    if ratios.size:
        unaudited = table.loc[~table["shipped_audit_available"], "pannuke_tissue"].tolist()
        if float(ratios.min()) == 1.0:
            log.info(
                "Filter retention is 1.000 on all %d audited references (unaudited tissues: %s).",
                int(ratios.size), unaudited or "none",
            )
        else:
            log.info(
                "Filter retention on audited references: min %.4f / median %.4f / max %.4f (unaudited tissues: %s).",
                float(ratios.min()), float(ratios.median()), float(ratios.max()),
                unaudited or "none",
            )

    log.info("Retained %d of %d candidate tissues: %s",
             len(retained), len(PANNUKE_TISSUE_TO_HISTOST), retained)
    if thin:
        log.info("Thin margin (< %.1fx the floor on an upper bound): %s", THIN_MARGIN_FACTOR, thin)
    dropped = table.loc[~table["retained"], "pannuke_tissue"].tolist()
    if dropped:
        log.info("Dropped: %s", dropped)
    log.info("Table -> %s", char_dir / "reference_support_v2.csv")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Count the coarse-v2 classes of the scRNA-seq reference of each PanNuke tissue and "
            "select the retained tissues."
        )
    )

    parser.add_argument("--out_dir", type=str, default=None,
                        help="Folder where characterization/ is written "
                             "(default: <ROOT_PATH>/eval/eval_datasets/pannuke_eval_data).")

    parser.add_argument("--ct_cat_id", type=str, default=DEFAULT_CT_CAT_ID,
                        help="Cell-type category id whose mapped label column is audited "
                             "({ct_cat_id}_cell_type on the reference side).")

    parser.add_argument("--coarse_mapping_version", type=str, default=DEFAULT_COARSE_MAPPING_VERSION,
                        help="Version of the coarse cell-type mapping.")

    parser.add_argument("--scconcept_training_id", type=str, default=DEFAULT_SCCONCEPT_TRAINING_ID,
                        help="scConcept training_id whose raw reference embeddings must exist for a "
                             "tissue to be retained.")

    parser.add_argument("--raw_reference_embedding_subdir", type=str, default="scConcept_embs",
                        help="Folder under eval/eval_datasets/scRNAseq_references/ holding the raw "
                             "scConcept reference embeddings.")

    parser.add_argument("--audit_eval_id", type=str, default=DEFAULT_AUDIT_EVAL_ID,
                        help="A.2 eval_id whose reference audits give the realized (fully filtered) "
                             "per-class counts (optional).")

    args = parser.parse_args()
    sys.exit(main(args))
