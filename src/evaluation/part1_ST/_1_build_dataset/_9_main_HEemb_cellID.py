#!/usr/bin/env python
"""
Select one main H&E cell embedding per cell among its per-(patch, cell) embeddings (largest alpha, then closest
to the patch center), stored as a row index into patch_cell_emb_he / patch_cell_emb_he_clip.

Saved in <ROOT_HistoST> / eval / eval_datasets / st_eval_data / eval_st_<training_dataset_id> / he_embeddings / HEembs_t<training_num>e<checkpoint_num> / main_HEemb_cellID / <slide_id>.parquet
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple
import h5py
import numpy as np
import pandas as pd

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.cell_ids import int_cell_id


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def decode_h5_string_array(arr) -> np.ndarray:
    """
    Decode an HDF5 string dataset into a 1D numpy object array of Python strings.
    Handles both (N,) and (N,1) cases.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]

    out = []
    for x in arr:
        if isinstance(x, (bytes, bytearray, np.bytes_)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object)



def infer_patch_hw_from_img_shape(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """Patch (H, W) from an image dataset shape (N, H, W, 3) or (N, 3, H, W)."""
    if len(shape) != 4:
        raise ValueError(f"Expected 4D image dataset, got shape={shape}")

    if shape[-1] == 3:
        return int(shape[1]), int(shape[2])

    if shape[1] == 3:
        return int(shape[2]), int(shape[3])

    raise ValueError(
        f"Could not infer patch H/W from image dataset shape={shape}. "
        "Expected NHWC or NCHW with 3 channels."
    )



def empty_output_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id_int": pd.Series(dtype=np.int32),
            "cell_id": pd.Series(dtype=object),
            "selected_patch_cell_row_index": pd.Series(dtype=np.int64),
            "selected_patch_index": pd.Series(dtype=np.int32),
            "selected_patch_id": pd.Series(dtype=object),
            "selected_alpha": pd.Series(dtype=np.float32),
            "selected_center_dist2": pd.Series(dtype=np.float64),
            "n_candidate_patches": pd.Series(dtype=np.int32),
        }
    )



def write_output_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, compression="zstd")



def load_patch_centers_from_eval_h5(
    slide_patch_h5: Path,
    expected_patch_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Patch centers (x, y) from the eval slide patch file; the patch order is checked against expected_patch_ids."""
    if not slide_patch_h5.exists():
        raise FileNotFoundError(f"Missing eval patch file: {slide_patch_h5}")

    with h5py.File(slide_patch_h5, "r") as f:
        if "coords" not in f or "img" not in f or "barcode" not in f:
            raise KeyError(f"{slide_patch_h5} must contain 'coords', 'img', and 'barcode'.")

        coords = f["coords"][:]
        patch_ids_h5 = decode_h5_string_array(f["barcode"][:])
        patch_h, patch_w = infer_patch_hw_from_img_shape(f["img"].shape)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"'coords' in {slide_patch_h5} must have shape (N, 2), got {coords.shape}")

    if len(patch_ids_h5) != len(expected_patch_ids):
        raise ValueError(
            f"Patch count mismatch between eval patch file and embedding file for {slide_patch_h5.stem}: "
            f"{len(patch_ids_h5)} vs {len(expected_patch_ids)}"
        )

    if not np.array_equal(patch_ids_h5, expected_patch_ids):
        raise ValueError(f"Patch order mismatch between eval patch file and embedding file for slide '{slide_patch_h5.stem}'.")

    x_min = coords[:, 0].astype(np.float64, copy=False)
    y_min = coords[:, 1].astype(np.float64, copy=False)

    center_x = x_min + (float(patch_w) / 2.0)
    center_y = y_min + (float(patch_h) / 2.0)
    return center_x, center_y



def load_centroids(centroid_parquet: Path) -> pd.DataFrame:
    """Load the centroid parquet (cell_id, x, y) and add cell_id_int."""
    if not centroid_parquet.exists():
        raise FileNotFoundError(f"Missing centroid parquet: {centroid_parquet}")

    df = pd.read_parquet(centroid_parquet, columns=["cell_id", "x", "y"]).copy()
    if df.empty:
        raise ValueError(f"Centroid parquet is empty: {centroid_parquet}")

    df["cell_id"] = df["cell_id"].astype(str)
    df["x"] = df["x"].astype(np.float64)
    df["y"] = df["y"].astype(np.float64)

    df["cell_id_int"] = np.fromiter(
        (int_cell_id(cid) for cid in df["cell_id"].values),
        dtype=np.int32,
        count=len(df),
    )

    if df["cell_id_int"].duplicated().any():
        dup = df.loc[df["cell_id_int"].duplicated(), "cell_id"].head(10).tolist()
        raise ValueError(
            f"Duplicate cell_id_int values in centroid parquet '{centroid_parquet}'. "
            f"Examples: {dup}"
        )

    return df[["cell_id_int", "cell_id", "x", "y"]]



def select_main_rows(
    patch_cell_patch_index: np.ndarray,
    patch_cell_cell_id_int: np.ndarray,
    patch_cell_alpha: np.ndarray,
    patch_ids: np.ndarray,
    patch_center_x: np.ndarray,
    patch_center_y: np.ndarray,
    centroids_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select one patch-cell row per cell: max alpha, then min distance to the patch center, then min
    patch_index, then min row index. Returns one row per cell_id_int.
    """
    n_groups = int(patch_cell_patch_index.shape[0])
    if n_groups == 0:
        return empty_output_df()

    if not (patch_cell_patch_index.shape == patch_cell_cell_id_int.shape == patch_cell_alpha.shape):
        raise ValueError("patch_cell_patch_index, patch_cell_cell_id_int, and patch_cell_alpha must have identical shapes.")

    if len(patch_ids) != len(patch_center_x) or len(patch_ids) != len(patch_center_y):
        raise ValueError("Patch metadata arrays have inconsistent lengths.")

    if patch_cell_patch_index.min(initial=0) < 0:
        raise ValueError("patch_cell_patch_index contains negative values.")

    if patch_cell_patch_index.max(initial=-1) >= len(patch_ids):
        raise ValueError(
            "patch_cell_patch_index contains values outside the patch_id range."
        )

    # -----------------------------------------------------------------
    # Vectorized join: patch-cell rows  <->  cell centroids
    # -----------------------------------------------------------------
    cent_cell_id_int = centroids_df["cell_id_int"].to_numpy(dtype=np.int32, copy=False)
    cent_cell_id = centroids_df["cell_id"].to_numpy(dtype=object, copy=False)
    cent_x = centroids_df["x"].to_numpy(dtype=np.float64, copy=False)
    cent_y = centroids_df["y"].to_numpy(dtype=np.float64, copy=False)

    cent_order = np.argsort(cent_cell_id_int, kind="mergesort")
    cent_cell_id_int_s = cent_cell_id_int[cent_order]
    cent_cell_id_s = cent_cell_id[cent_order]
    cent_x_s = cent_x[cent_order]
    cent_y_s = cent_y[cent_order]

    pos = np.searchsorted(cent_cell_id_int_s, patch_cell_cell_id_int)
    ok = (pos < len(cent_cell_id_int_s)) & (cent_cell_id_int_s[pos] == patch_cell_cell_id_int)

    if not np.all(ok):
        missing = np.unique(patch_cell_cell_id_int[~ok])[:20].tolist()
        raise KeyError(
            "Some cell_id_int values present in patch_cell_* were not found in the centroid parquet. "
            f"Examples: {missing}"
        )

    cell_x = cent_x_s[pos]
    cell_y = cent_y_s[pos]

    # -----------------------------------------------------------------
    # Tie-break score: distance from centroid to patch center
    # -----------------------------------------------------------------
    pcx = patch_center_x[patch_cell_patch_index]
    pcy = patch_center_y[patch_cell_patch_index]

    dx = cell_x - pcx
    dy = cell_y - pcy
    dist2 = dx * dx + dy * dy  # float64

    # -----------------------------------------------------------------
    # Sort by cell_id_int, -alpha, dist2, patch_index, row index; keep the first row of each cell
    # -----------------------------------------------------------------
    row_index = np.arange(n_groups, dtype=np.int64)

    order = np.lexsort(
        (
            row_index,
            patch_cell_patch_index,
            dist2,
            -patch_cell_alpha.astype(np.float64, copy=False),
            patch_cell_cell_id_int,
        )
    )

    cell_sorted = patch_cell_cell_id_int[order]

    first = np.empty(cell_sorted.shape[0], dtype=bool)
    first[0] = True
    first[1:] = cell_sorted[1:] != cell_sorted[:-1]

    selected_rows = order[first]

    # Candidate counts per cell
    uniq_cells, counts = np.unique(cell_sorted, return_counts=True)

    # Sanity: selected rows are in the same sorted-cell order as uniq_cells
    selected_cells = patch_cell_cell_id_int[selected_rows]
    if not np.array_equal(selected_cells, uniq_cells):
        raise RuntimeError("Internal consistency error while selecting main patch rows.")

    # Recover string cell ids
    pos_sel = np.searchsorted(cent_cell_id_int_s, uniq_cells)
    selected_cell_ids = cent_cell_id_s[pos_sel]

    selected_patch_index = patch_cell_patch_index[selected_rows]
    selected_patch_id = patch_ids[selected_patch_index]
    selected_alpha = patch_cell_alpha[selected_rows]
    selected_dist2 = dist2[selected_rows]

    out = pd.DataFrame(
        {
            "cell_id_int": uniq_cells.astype(np.int32, copy=False),
            "cell_id": selected_cell_ids.astype(object, copy=False),
            "selected_patch_cell_row_index": selected_rows.astype(np.int64, copy=False),
            "selected_patch_index": selected_patch_index.astype(np.int32, copy=False),
            "selected_patch_id": selected_patch_id.astype(object, copy=False),
            "selected_alpha": selected_alpha.astype(np.float32, copy=False),
            "selected_center_dist2": selected_dist2.astype(np.float64, copy=False),
            "n_candidate_patches": counts.astype(np.int32, copy=False),
        }
    )

    return out



def process_slide(
    slide_id: str,
    emb_h5_path: Path,
    slide_patch_h5: Path,
    centroid_parquet: Path,
    out_parquet: Path,
    overwrite: bool = False,
) -> None:
    if out_parquet.exists() and not overwrite:
        print(f"  - Output already exists, skipping: {out_parquet.name}")
        return

    print(f"  - Reading embedding metadata: {emb_h5_path.name}")
    with h5py.File(emb_h5_path, "r") as f:
        required = [
            "patch_id",
            "patch_cell_patch_index",
            "patch_cell_cell_id_int",
            "patch_cell_alpha",
        ]
        missing = [k for k in required if k not in f]
        if missing:
            raise KeyError(f"{emb_h5_path} is missing datasets: {missing}")

        patch_cell_patch_index = f["patch_cell_patch_index"][:].astype(np.int32, copy=False)
        patch_cell_cell_id_int = f["patch_cell_cell_id_int"][:].astype(np.int32, copy=False)
        patch_cell_alpha = f["patch_cell_alpha"][:].astype(np.float32, copy=False)

        n_groups = int(patch_cell_patch_index.shape[0])

        if n_groups == 0:
            print("    No patch-cell groups in this slide. Skipping output.")
            return

        patch_ids = decode_h5_string_array(f["patch_id"][:])

    print(f"    patch-cell groups: {n_groups:,}")

    # Patch geometry from eval patch file
    patch_center_x, patch_center_y = load_patch_centers_from_eval_h5(
        slide_patch_h5=slide_patch_h5,
        expected_patch_ids=patch_ids,
    )

    # Cell centroids
    print(f"    Loading centroids: {centroid_parquet.name}")
    centroids_df = load_centroids(centroid_parquet)

    # Select main row per cell
    out_df = select_main_rows(
        patch_cell_patch_index=patch_cell_patch_index,
        patch_cell_cell_id_int=patch_cell_cell_id_int,
        patch_cell_alpha=patch_cell_alpha,
        patch_ids=patch_ids,
        patch_center_x=patch_center_x,
        patch_center_y=patch_center_y,
        centroids_df=centroids_df,
    )

    mean_candidates = (float(out_df["n_candidate_patches"].mean()) if len(out_df) > 0 else 0.0)

    print(
        f"    selected cells: {len(out_df):,} | "
        f"mean candidate patches per cell: {mean_candidates:.3f}"
    )

    write_output_parquet(out_df, out_parquet)
    print(f"    saved: {out_parquet}")



# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(args) -> None:
    
    # Paths
    eval_root = get_full_path("eval", "eval_datasets", "st_eval_data", f"eval_st_{args.training_dataset_id}")

    emb_root = eval_root / "he_embeddings" / f"HEembs_t{args.training_num}e{args.checkpoint_num}"
    patches_dir = eval_root / "patches"
    out_dir = emb_root / "main_HEemb_cellID"

    centroids_dir = get_full_path("train", "training_datasets", args.dataset_id, args.training_dataset_id, "nucleus_centroids")

    if not emb_root.exists():
        raise FileNotFoundError(f"Embedding directory not found: {emb_root}")
    if not patches_dir.exists():
        raise FileNotFoundError(f"Eval patches directory not found: {patches_dir}")
    if not centroids_dir.exists():
        raise FileNotFoundError(f"Centroids directory not found: {centroids_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Slide embedding files (macOS '._*' files skipped)
    slide_emb_files = sorted(
        p for p in emb_root.glob("*.h5") if not p.name.startswith(".")
    )
    if not slide_emb_files:
        raise FileNotFoundError(f"No slide embedding H5 files found in: {emb_root}")

    print(f"Embedding root : {emb_root}")
    print(f"Eval patches   : {patches_dir}")
    print(f"Centroids dir  : {centroids_dir}")
    print(f"Output dir     : {out_dir}")
    print(f"Slides found   : {len(slide_emb_files)}")

    # Process each slide
    for i, emb_h5_path in enumerate(slide_emb_files, start=1):
        slide_id = emb_h5_path.stem
        print(f"\n* Processing slide '{slide_id}' ({i}/{len(slide_emb_files)})")

        slide_patch_h5 = patches_dir / f"{slide_id}.h5"
        centroid_parquet = centroids_dir / f"nucleus_centroids_{slide_id}.parquet"
        out_parquet = out_dir / f"{slide_id}.parquet"

        process_slide(
            slide_id=slide_id,
            emb_h5_path=emb_h5_path,
            slide_patch_h5=slide_patch_h5,
            centroid_parquet=centroid_parquet,
            out_parquet=out_parquet,
            overwrite=args.overwrite,
        )

    print("\nDone.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select one main H&E cell embedding per cell from per-(patch, cell) embeddings.")

    parser.add_argument("--dataset_id", type=str,
                        required=True,
                        help="Dataset ID, e.g. ds_ct1_20x_224_o112")
    
    parser.add_argument("--training_dataset_id", type=str,
                        required=True,
                        help="Training dataset ID, e.g. ds_ct1_20x_224_o112_1")
    
    parser.add_argument("--training_num", type=str,
                        required=True,
                        help="Training number, e.g. 46 for 'training_46'")
    
    parser.add_argument("--checkpoint_num", type=str,
                        required=True,
                        help="Checkpoint number, e.g. 13 for 'checkpoint_epoch_13'")
   
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output parquet files.")

    args = parser.parse_args()
    main(args)