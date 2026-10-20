"""
Extract patch coordinates (bbox) in H&E space for each slide for each patch_id in parquet format.
Saved in <ROOT_HistoST> / train / datasets / <dataset_id> / patch_bboxes.parquet with all slides.
"""

import sys
from pathlib import Path
import os
import argparse
import pandas as pd

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path




def main(args):
    """Main function to process multiple slides and store patch bounding boxes in parquet format."""

    output_folder = get_full_path("train", "datasets", args.dataset_id, as_str=True)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "patch_bboxes.parquet")
        
    # Always overwrite the parquet
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # Build the patch_bboxes.parquet file
    csv_path = os.path.join(output_folder, "patch_metrics.csv")
    if os.path.exists(csv_path):
        final_df = pd.read_csv(csv_path)
        final_df = final_df[["slide_id", "patch_id", "x_min", "y_min", "x_max", "y_max"]]  # keep only the relevant columns
        print(final_df.head())
        print("Saving...")
        final_df.to_parquet(output_path, index=False)
        print("Done.")
    
    else:
        print(f"[ERROR] The file {csv_path} does not exist.")





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Store patch bounding boxes in parquet format.")
    
    parser.add_argument("--dataset_id", type=str, 
                        default="ds_ct1_20x_224_o112",
                        help="Dataset_id, e.g. ds_ct1_20x_224_o112")

    args = parser.parse_args()
    main(args)
