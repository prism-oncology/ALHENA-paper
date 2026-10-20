#!/usr/bin/env python3
"""
Build a gene-symbol alias map from the HGNC complete-set TSV <input>: approved, alias and previous
symbols are mapped to the approved symbol, and symbols mapping to several genes are flagged as ambiguous.

Saved in <ROOT_HistoST> / eval / eval_datasets / TCGA_eval_data / refs / hgnc / <output> (next to <input>)
"""

import argparse
import hashlib
from pathlib import Path
import pandas as pd
import sys

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_hgnc_multi_value(value: str) -> list[str]:
    """
    HGNC multi-value fields are commonly pipe-separated.
    Example: A|B|C
    """
    if value is None:
        return []

    value = str(value).strip()

    if not value or value.lower() == "nan":
        return []

    return [x.strip() for x in value.split("|") if x.strip()]


def main(args) -> int:

    input_path = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data', 'refs', 'hgnc', args.input)
    output_path = get_full_path('eval', 'eval_datasets', 'TCGA_eval_data', 'refs', 'hgnc', args.output)

    df = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)

    required_columns = {
        "hgnc_id",
        "symbol",
        "name",
        "status",
        "alias_symbol",
        "prev_symbol",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required HGNC columns: {sorted(missing)}")

    rows = []

    for _, row in df.iterrows():
        status = row["status"].strip()
        approved_symbol = row["symbol"].strip()
        hgnc_id = row["hgnc_id"].strip()
        approved_name = row["name"].strip()

        if status and status != "Approved":
            continue

        if not approved_symbol:
            continue

        # 1. The approved symbol maps to itself.
        rows.append(
            {
                "input_symbol": approved_symbol,
                "approved_symbol": approved_symbol,
                "hgnc_id": hgnc_id,
                "approved_name": approved_name,
                "source_field": "symbol",
            }
        )

        # 2. HGNC alias symbols map to the approved symbol.
        for alias in split_hgnc_multi_value(row["alias_symbol"]):
            rows.append(
                {
                    "input_symbol": alias,
                    "approved_symbol": approved_symbol,
                    "hgnc_id": hgnc_id,
                    "approved_name": approved_name,
                    "source_field": "alias_symbol",
                }
            )

        # 3. Previous HGNC symbols map to the current approved symbol.
        for prev in split_hgnc_multi_value(row["prev_symbol"]):
            rows.append(
                {
                    "input_symbol": prev,
                    "approved_symbol": approved_symbol,
                    "hgnc_id": hgnc_id,
                    "approved_name": approved_name,
                    "source_field": "prev_symbol",
                }
            )

    alias_df = pd.DataFrame(rows)

    alias_df = alias_df.drop_duplicates(
        subset=[
            "input_symbol",
            "approved_symbol",
            "hgnc_id",
            "source_field",
        ]
    )

    # Mark ambiguous input symbols (mapped to more than one approved symbol)
    n_targets = alias_df.groupby("input_symbol")["approved_symbol"].nunique()
    ambiguous_symbols = set(n_targets[n_targets > 1].index)

    alias_df["is_ambiguous"] = alias_df["input_symbol"].isin(ambiguous_symbols)

    alias_df = alias_df[
        [
            "input_symbol",
            "approved_symbol",
            "hgnc_id",
            "approved_name",
            "source_field",
            "is_ambiguous",
        ]
    ].sort_values(
        [
            "input_symbol",
            "approved_symbol",
            "source_field",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    alias_df.to_csv(output_path, sep="\t", index=False)

    print(
        {
            "input": str(input_path),
            "output": str(output_path),
            "n_rows": int(len(alias_df)),
            "n_unique_input_symbols": int(alias_df["input_symbol"].nunique()),
            "n_ambiguous_input_symbols": int(alias_df.loc[alias_df["is_ambiguous"], "input_symbol"].nunique()),
            "source_sha256": sha256_path(input_path),
            "alias_map_sha256": sha256_path(output_path),
        }
    )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a gene alias map from HGNC data.")
    
    parser.add_argument("--input", type=Path,
                        required=True, 
                        help="Filename to the input HGNC complete set TSV file.")
    
    parser.add_argument("--output", type=Path,
                        required=True,
                        help="Filename to the output gene alias map TSV file.")

    args = parser.parse_args()
    raise SystemExit(main(args))