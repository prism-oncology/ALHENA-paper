"""
Add metrics from CellVit inference results to the patch_metrics.csv for given <dataset_id> and <run_number>.
"""

import argparse
import os
import json
import pandas as pd
from pathlib import Path
import sys

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path



def open_json_metrics(output_cellvit_folder):
    
    json_path = output_cellvit_folder / 'inference_results.json'
    with open(json_path, 'r') as file:
        metric_json_file = json.load(file)
    
    return metric_json_file




def build_df_metrics(metric_json_file):

    df_metrics = pd.DataFrame.from_dict(metric_json_file['image_metrics'], orient='index')

    df_metrics.reset_index(inplace=True)
    df_metrics.rename(columns={'index': 'patch_id'}, inplace=True)

    # Remove .png at the end of patch_id
    df_metrics['patch_id'] = df_metrics['patch_id'].str.replace('.png', '', regex=False)

    return df_metrics




def main(args):

    # Paths
    dataset_id_folder = get_full_path('train', 'datasets', args.dataset_id)
    cellvit_output_folder = get_full_path(dataset_id_folder, 'do_patch_eval', args.run_id)
    
    # Get metrics output from CellVit
    print("\n* Loading metrics...")
    metric_json_file = open_json_metrics(cellvit_output_folder)
    df_metrics = build_df_metrics(metric_json_file)
    print(df_metrics)

    # Check there is no NaN in the metrics
    if df_metrics.isnull().values.any():
        print("\nWarning: There are NaN values in the metrics. Please check the input data.")
    else:
        print("\nAll metrics are valid, no NaN values found.")

    # Load patch_metrics.csv
    print("\n* Loading patch_metrics.csv...")
    patch_metrics_path = dataset_id_folder / 'patch_metrics.csv'
    df_final = pd.read_csv(patch_metrics_path)
    print(df_final)

    # Add / update metrics only for the patches in this run
    print("\n* Updating metrics for new patches...")
    df_final.set_index("patch_id", inplace=True)
    df_metrics.set_index("patch_id", inplace=True)
    df_final.update(df_metrics) # overwrite existing metrics or add new ones
    new_cols = df_metrics.columns.difference(df_final.columns) # bring across any brand-new metric columns (if the JSON added more)
    if len(new_cols):
        df_final[new_cols] = df_metrics[new_cols]
    df_final.reset_index(inplace=True)
    print(df_final)

    # Check if patch_id is unique
    if df_final['patch_id'].is_unique:
        print("\nAll patch_ids are unique.")
    else:
        print("\nWarning: There are duplicate patch_ids in the data. Please check the input data.")

    # Save updated patch_metrics.csv
    print("\n* Saving updated patch_metrics.csv...")
    df_final.to_csv(patch_metrics_path, index=False)
    print("Done.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add metrics in patch_metrics.csv")

    parser.add_argument("--dataset_id", type=str, 
                        default="ds_ct1_20x_256_o112", 
                        help="Dataset ID to identify the patch_metrics.csv to update with CellViT results, e.g. ds_ct1_20x_256_o112")
    
    parser.add_argument("--run_id", type=str, 
                        default="run1", 
                        help="Run ID to identify the patch evaluation folder with pre-trained CellViT results, e.g. run1")

    args = parser.parse_args()
    main(args)
