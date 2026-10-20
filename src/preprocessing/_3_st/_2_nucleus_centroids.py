"""
Compute nucleus centroids from SpatialData to get them in the correct coordinates system comparing to patch bbox (H&E space instead of ST space).
Saved in <ROOT_HistoST> / train / datasets / <dataset_id> / nucleus_centroids / nucleus_centroids_<slide_id>.parquet.

Format:
cell_id_str | x | y 
"""

import sys
from pathlib import Path
import argparse
import os
import spatialdata as sd
from typing import List

from sopa.utils import get_spatial_image, to_intrinsic

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path
from src.utils.sdata_utils import promote_20x_to_default



def load_slide_ids_from_txt(dataset_id_folder: Path, filename: str = "slide_ids.txt") -> List[str]:
    """Load slide IDs from <dataset_id_folder>/slide_ids.txt (one ID per line)."""
    path = Path(dataset_id_folder) / filename
    if not path.exists():
        raise FileNotFoundError(f"slide id file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        slide_ids = [line.strip() for line in f if line.strip()]
    if not slide_ids:
        raise ValueError(f"Slide id file is empty: {path}")
    return slide_ids



def process_slide(dataset_id: str, slide_id: str, resolution: int) -> None:

    print(f"\n====== Processing {slide_id} ======")

    # Paths
    sdata_folder = get_full_path('STHELAR_HistoST', 'sdata_final', as_str=True)
    output_folder = get_full_path('train', 'datasets', dataset_id, 'nucleus_centroids', as_str=True)

    sdata_path = os.path.join(sdata_folder, f"sdata_{slide_id}.zarr")
    output_path = os.path.join(output_folder, f"nucleus_centroids_{slide_id}.parquet")

    # Skip if already done
    if os.path.exists(output_path):
        print(f"* Already exists, skipping: {output_path}")
        return

    # Load spatial data
    print("\n* Loading SpatialData...")
    sdata = sd.read_zarr(sdata_path, selection=("shapes", "images"))
    print(sdata)

    # Promote 20x to default if needed
    print(f"\nUsing H&E resolution: {resolution}x")
    resolution_40 = get_spatial_image(sdata, "he").shape
    print(f"40x resolution: {resolution_40}")
    if resolution == 20:
        promote_20x_to_default(sdata, key="he")
        resolution_20 = get_spatial_image(sdata, "he").shape
        print(f"20x resolution: {resolution_20}")
        print(f"Check dividing 40x by 2 / Diff are : {resolution_40[1] // 2 - resolution_20[1]} and {resolution_40[2] // 2 - resolution_20[2]}")
    elif resolution != 40:
        raise ValueError("Unsupported H&E resolution. Choose 20 or 40.")
    print(f"\nNew sdata: {sdata}")

    # Extract nucleus boundaries in the correct spatial reference
    he = get_spatial_image(sdata, "he")  # Uses highest resolution scale (scale0)
    geo_df_nuclei = to_intrinsic(sdata, sdata.shapes['nucleus_boundaries'], he)

    # Compute centroids efficiently
    print("\n* Extracting nucleus centroids...")
    geo_df_nuclei["centroid"] = geo_df_nuclei.geometry.centroid
    geo_df_centroids = geo_df_nuclei[["centroid"]].reset_index().rename(columns={"index": "cell_id"})
    if geo_df_centroids["centroid"].isnull().any(): # Check there is no NaN in centroids
        raise ValueError("Some centroids are NaN. Check the nucleus boundaries data.")
    geo_df_centroids["x"] = geo_df_centroids["centroid"].x
    geo_df_centroids["y"] = geo_df_centroids["centroid"].y
    geo_df_centroids = geo_df_centroids.drop(columns=["centroid"])
    print(geo_df_centroids.head())

    # Save in Parquet format
    print("\n* Saving centroid data...")
    os.makedirs(output_folder, exist_ok=True)
    geo_df_centroids.attrs = {}  # Reset metadata to avoid JSON serialization issues
    geo_df_centroids.to_parquet(output_path, index=False)

    print("\nDone.")



def process_dataset(args) -> None:
    
    dataset_id = args.dataset_id
    resolution = args.resolution

    dataset_id_folder = get_full_path('train', 'datasets', dataset_id)
    slide_ids = load_slide_ids_from_txt(dataset_id_folder, filename="slide_ids.txt")

    print(f"\nFound {len(slide_ids)} slide_ids in {dataset_id_folder / 'slide_ids.txt'}")
    n_done, n_skip = 0, 0
    for slide_id in slide_ids:
        out_path = get_full_path('train', 'datasets', dataset_id, 'nucleus_centroids') / f"nucleus_centroids_{slide_id}.parquet"
        if out_path.exists():
            n_skip += 1
            continue
        process_slide(dataset_id=dataset_id, slide_id=slide_id, resolution=resolution)
        n_done += 1
    print(f"\nSummary: processed={n_done}, skipped(existing)={n_skip}, total={len(slide_ids)}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract nucleus centroids from SpatialData")
    
    parser.add_argument("--dataset_id", type=str,
                        default="ds_ct1_20x_224_o112",
                        help="Dataset_id, e.g. ds_ct1_20x_224_o112")
    
    parser.add_argument("--resolution", type=int, 
                        default=20,
                        help="Resolution for H&E images using 20 or 40x (choose 20 or 40)")

    args = parser.parse_args()
    process_dataset(args)