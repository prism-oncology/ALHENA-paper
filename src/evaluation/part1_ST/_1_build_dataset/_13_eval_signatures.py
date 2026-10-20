"""
Build the external gene-signature panel (MSigDB Hallmark and curated immune / niche / stromal gene sets)
used by the Task A.1 signature strata.

Saved in <ROOT_HistoST> / eval / eval_datasets / eval_signatures / <eval_signatures_id> /
    - eval_signatures_table.parquet: one row per (signature_id, gene)
    - config.json: parameters, source-file hashes and per-signature metadata
    - signatures.gmt, sources.json: copies of the source files
"""

import sys
from pathlib import Path
import argparse
import logging
import os
import json
import hashlib
import shutil
import datetime as _dt
from typing import Dict, List
import pandas as pd

logger = logging.getLogger(__name__)

# Add project root
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path


DEFAULT_SIGNATURES_GMT_PATH = str(
    project_root / "data" / "eval_signatures" / "signatures.gmt"
)
DEFAULT_SIGNATURES_SOURCES_PATH = str(
    project_root / "data" / "eval_signatures" / "sources.json"
)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _save_json_atomic(path: str, payload: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _save_parquet_atomic(path: str, df: pd.DataFrame) -> None:
    tmp_path = path + ".tmp"
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_file_atomic(src: str, dst: str) -> None:
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


# ---------------------------------------------------------------------------
# Source-file loaders
# ---------------------------------------------------------------------------

def _parse_gmt(gmt_path: str) -> Dict[str, List[str]]:
    """Parse a GMT file (tab-separated: signature_id, description, genes) into {signature_id: [genes]}."""
    if not os.path.isfile(gmt_path):
        raise FileNotFoundError(f"[ERROR] signatures.gmt not found: '{gmt_path}'.")

    signatures: Dict[str, List[str]] = {}
    with open(gmt_path, "r") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(
                    f"[ERROR] Malformed GMT line {line_no} in '{gmt_path}': "
                    f"expected at least 3 tab-separated fields (signature_id, "
                    f"description, gene), got {len(parts)}."
                )
            sig_id = parts[0].strip()
            if not sig_id:
                raise ValueError(
                    f"[ERROR] Empty signature_id on line {line_no} of '{gmt_path}'."
                )
            if sig_id in signatures:
                raise ValueError(
                    f"[ERROR] Duplicate signature_id '{sig_id}' on line {line_no} "
                    f"of '{gmt_path}'."
                )
            genes_raw = [g.strip() for g in parts[2:]]
            genes = [g for g in genes_raw if g]
            if len(genes) == 0:
                raise ValueError(
                    f"[ERROR] Signature '{sig_id}' (line {line_no}) has zero genes."
                )
            # De-duplicate within a signature while preserving first-seen order.
            seen: set = set()
            deduped: List[str] = []
            for g in genes:
                if g not in seen:
                    seen.add(g)
                    deduped.append(g)
            signatures[sig_id] = deduped
    if len(signatures) == 0:
        raise ValueError(f"[ERROR] No signatures parsed from '{gmt_path}'.")
    return signatures


def _load_sources_json(sources_path: str) -> Dict[str, dict]:
    """Load sources.json into {signature_id: metadata_dict}."""
    if not os.path.isfile(sources_path):
        raise FileNotFoundError(
            f"[ERROR] signatures sources.json not found: '{sources_path}'."
        )
    with open(sources_path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(
            f"[ERROR] '{sources_path}' must be a JSON object keyed by signature_id."
        )
    return payload


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _build_signatures_table(
    signatures: Dict[str, List[str]],
) -> pd.DataFrame:
    """Long-format table with one row per (signature_id, gene), in GMT order (gene symbols kept as-is)."""
    rows: List[Dict[str, object]] = []
    for sig_id, genes in signatures.items():
        for g in genes:
            rows.append({
                "signature_id": sig_id,
                "gene": g,
            })
    return pd.DataFrame(rows, columns=["signature_id", "gene"])


def _per_signature_summary(
    table: pd.DataFrame,
    sources: Dict[str, dict],
) -> Dict[str, dict]:
    """Per-signature metadata for config.json: sources.json metadata plus n_genes_total."""
    summary: Dict[str, dict] = {}
    for sig_id, grp in table.groupby("signature_id", sort=False):
        meta = dict(sources.get(sig_id, {}))
        meta["n_genes_total"] = int(grp.shape[0])
        summary[str(sig_id)] = meta
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:

    output_folder = get_full_path(
        "eval", "eval_datasets", "eval_signatures",
        args.eval_signatures_id, as_str=True,
    )

    main_output = os.path.join(output_folder, "eval_signatures_table.parquet")
    if os.path.exists(main_output) and not args.overwrite:
        mtime = _dt.datetime.fromtimestamp(
            os.path.getmtime(main_output)
        ).isoformat(timespec="seconds")
        raise FileExistsError(
            f"[ERROR] eval_signatures_table.parquet already exists at "
            f"'{main_output}' (mtime={mtime}). Pass --overwrite to replace it, "
            f"or choose a new --eval_signatures_id."
        )

    os.makedirs(output_folder, exist_ok=True)

    signatures = _parse_gmt(args.signatures_gmt_path)
    sources = _load_sources_json(args.signatures_sources_path)

    missing_in_sources = [s for s in signatures.keys() if s not in sources]
    if missing_in_sources:
        logger.warning(
            "Signatures present in GMT but missing from sources.json (metadata "
            "will be empty for these): %s",
            missing_in_sources,
        )

    logger.info("eval_signatures_id    = %s", args.eval_signatures_id)
    logger.info("signatures.gmt        = %s", args.signatures_gmt_path)
    logger.info("sources.json          = %s", args.signatures_sources_path)
    logger.info("#signatures parsed    = %d", len(signatures))
    logger.info("output_folder         = %s", output_folder)

    table = _build_signatures_table(signatures)
    summary = _per_signature_summary(table, sources)

    gmt_sha256 = _sha256_file(args.signatures_gmt_path)
    sources_sha256 = _sha256_file(args.signatures_sources_path)

    config_payload = {
        "eval_signatures_id": args.eval_signatures_id,
        "signatures_gmt_path": args.signatures_gmt_path,
        "signatures_sources_path": args.signatures_sources_path,
        "signatures_gmt_sha256": gmt_sha256,
        "signatures_sources_sha256": sources_sha256,
        "n_signatures": len(signatures),
        "signatures": summary,
        "recipe": {
            "scope": (
                "Global: this artifact is independent of any training_dataset_id "
                "or G_base_id. The downstream Task A.1 consumer "
                "(persist_signature_strata) intersects each signature with the "
                "eval-local trainable gene union at fill time."
            ),
            "genes": (
                "GMT-derived gene symbols, kept as-is from the source files "
                "(case_normalization='identity'). Outdated gene symbols (e.g. "
                "KAL1, KIAA0125, IGL@) are not remapped."
            ),
            "frozen_provenance": (
                "signatures.gmt and sources.json are copied into the output "
                "folder. The sha256 hashes in this config.json are those of the "
                "source files used."
            ),
        },
    }

    _save_parquet_atomic(main_output, table)
    _save_json_atomic(os.path.join(output_folder, "config.json"), config_payload)

    # Copy the source files into the output folder
    _copy_file_atomic(
        args.signatures_gmt_path, os.path.join(output_folder, "signatures.gmt")
    )
    _copy_file_atomic(
        args.signatures_sources_path,
        os.path.join(output_folder, "sources.json"),
    )

    n_rows = int(table.shape[0])
    logger.info("==================== DONE ====================")
    logger.info("eval-signatures table: %s", main_output)
    logger.info(
        "#signatures=%d  #(signature,gene)_rows=%d",
        len(signatures), n_rows,
    )




if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build the external gene-signature panel for the Task A.1 signature strata."
    )

    parser.add_argument("--eval_signatures_id", type=str,
                        default="eval_signatures_1",
                        help="Identifier of this eval-signatures build.")

    parser.add_argument("--signatures_gmt_path", type=str,
                        default=DEFAULT_SIGNATURES_GMT_PATH,
                        help="Path to signatures.gmt (default: data/eval_signatures/signatures.gmt).")

    parser.add_argument("--signatures_sources_path", type=str,
                        default=DEFAULT_SIGNATURES_SOURCES_PATH,
                        help="Path to sources.json (default: data/eval_signatures/sources.json).")

    parser.add_argument("--overwrite",
                        action="store_true",
                        help="Overwrite an existing eval_signatures_id folder.")

    args = parser.parse_args()
    main(args)
