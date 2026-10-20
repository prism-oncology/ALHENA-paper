"""
Validate that the 'cell_id' strings in the AnnData obs can be safely converted to integers using the int_cell_id() function, 
and that the resulting integers fit within the np.int32 range. 
This is crucial as mask building relies on int32 cell IDs.
"""

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple, List
import numpy as np
import anndata as ad

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.cell_ids import int_cell_id



def validate_cell_id_string(cell_id_str: str) -> str | None:
    """
    Validate a Xenium-style cell_id string before passing it to int_cell_id().
    Returns None if valid; otherwise returns a short error message describing the issue.

    Checks:
      1) Must end with the exact suffix '-1' (because int_cell_id() slices off 2 chars).
      2) All characters before the suffix must be lowercase letters in 'a'..'p' inclusive.
    """
    if not cell_id_str.endswith("-1"):
        return "missing or incorrect '-1' suffix"
    core = cell_id_str[:-2]
    if len(core) == 0:
        return "no characters before '-1' suffix"
    for ch in core:
        if not ('a' <= ch <= 'p'):
            return f"invalid character '{ch}' (expected only 'a'..'p' before the suffix)"
    return None



def validate_all_cell_ids(cell_ids: Iterable[str], max_examples: int = 10) -> List[str]:
    """
    Validate a collection of cell_id strings. Returns a list of human-readable
    error examples (up to max_examples). An empty list means all are valid.
    """
    examples: List[str] = []
    for idx, s in enumerate(cell_ids):
        msg = validate_cell_id_string(s)
        if msg is not None:
            if len(examples) < max_examples:
                examples.append(f"[{idx}] '{s}': {msg}")
            else:
                # Collect enough for a helpful report; no need to scan further for examples
                # but continue full scan to know there are more errors.
                pass
    return examples



def convert_cell_ids_strict(cell_ids: Iterable[str]) -> np.ndarray:
    """
    Validate then convert using int_cell_id().
    Keeps Python int (arbitrary precision) until the explicit cast below.
    """
    # Validate first
    errors = validate_all_cell_ids(cell_ids)
    if errors:
        more = "..." if len(errors) == 10 else ""
        report = "\n".join(errors)
        raise ValueError(
            "Invalid cell_id strings detected BEFORE conversion. "
            "Examples (index -> problem):\n"
            f"{report}\n{more}"
        )
    # Convert using the exact function
    converted = [int_cell_id(cid) for cid in cell_ids]
    return np.asarray(converted, dtype=object)  # object dtype to avoid accidental wrap



def check_int32_safety(values: np.ndarray) -> Tuple[bool, int, int]:
    """
    Check whether all values fit into signed 32-bit int range.
    Returns (is_safe, min_value, max_value).
    """
    if values.dtype != object:
        values = values.astype(object)
    vmin = int(min(values))
    vmax = int(max(values))
    i32 = np.iinfo(np.int32)
    is_safe = (vmin >= i32.min) and (vmax <= i32.max)
    return is_safe, vmin, vmax



def process_file(folder: Path, slide_id: str, obs_column: str) -> int:
    """
    Load {folder}/{slide_id}.h5ad, validate obs[obs_column] format,
    convert with the int_cell_id(), attach as obs['cell_id_int'],
    cast to np.int32
    Raise an error if any step fails, or if values do not fit in np.int32.
    """
    
    # Check input file exists
    in_path = folder / f"{slide_id}.h5ad"
    if not in_path.exists():
        raise ValueError(f"Input file does not exist: {in_path}")

    # Load the AnnData
    print(f"* Loading: {in_path}")
    adata = ad.read_h5ad(in_path)

    # Check obs column exists
    if obs_column not in adata.obs:
        raise ValueError(f"Column '{obs_column}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")

    # Check for missing values
    series = adata.obs[obs_column]
    if series.isna().any():
        n_na = int(series.isna().sum())
        raise ValueError(f"Column '{obs_column}' contains {n_na} missing values (NA). All values must be non-missing.")

    # Ensure string type
    cell_ids = series.astype(str).tolist()

    # Validate and convert cell_id to int
    print("* Validating and converting cell_id...")
    try:
        int_values = convert_cell_ids_strict(cell_ids)
    except ValueError as e:
        raise ValueError(f"Validation/conversion failed for '{obs_column}': {str(e)}")

    # Safety check for np.int32
    is_safe, vmin, vmax = check_int32_safety(int_values)
    print(f"  - Converted {len(int_values)} IDs.")
    print(f"  - Value range: min={vmin}, max={vmax}")
    print(f"  - np.int32 range: [{np.iinfo(np.int32).min}, {np.iinfo(np.int32).max}]")
    print(f"  - Safe to cast to np.int32: {is_safe}")

    if not is_safe:
        raise ValueError("Converted integer values exceed np.int32 range. Cannot safely cast to int32 for mask building.")

    # Choose dtype for storage
    target_dtype = np.int32
    cast_values = np.asarray(int_values, dtype=target_dtype)

    # Extra sanity check: ensure no negative values after cast
    if cast_values.min(initial=0) < 0:
        raise ValueError("Negative values detected after casting to np.int32, which is unexpected for cell IDs.")

    print(f"-> Successfully validated and converted cell IDs for slide '{slide_id}'. All values fit in np.int32.")
    print("Done.")



def main(args) -> int:

    # Path
    folder = get_full_path('STHELAR_HistoST', 'adata_graph')
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Provided folder does not exist or is not a directory: {folder}")

    # Determine which slides to process
    if args.slide_id:
        slide_ids = [args.slide_id]
    else:
        # Default: process all slides present in the adata_graph folder
        h5ad_paths = [
            p for p in folder.glob("*.h5ad")
            if p.is_file() and not p.name.startswith(".") and not p.name.startswith("._")
        ]
        slide_ids = sorted(p.stem for p in h5ad_paths)
        if not slide_ids:
            raise ValueError(f"No valid .h5ad files found in folder: {folder}")
    print(f"{len(slide_ids)} slide(s) to process: {slide_ids}")

    # Perform the checkings
    for slide_id in slide_ids:
        print(f"\n=== PROCESSING: {slide_id} ===\n")
        process_file(
            folder=folder,
            slide_id=slide_id,
            obs_column=args.obs_column
        )



if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Validate Xenium cell_id strings, convert to integers, and verify int32 safety.")
    
    p.add_argument("--slide_id", type=str,
                   default=None,
                   help="Slide ID (if omitted, process all .h5ad slides found in the adata_graph folder).")
    
    p.add_argument("--obs_column", 
                   default="cell_id",
                   help="Name of the obs column containing Xenium cell IDs (default: 'cell_id').")
    
    args = p.parse_args()
    main(args)