"""
Given a training_dataset_id produced by make_folds.py 
(i.e. folder <ROOT_HistoST> / train / training_datasets / <dataset_id> / <training_dataset_id>)
(i.e., a folder that contains cell_count_train.csv / cell_count_valid.csv [/ cell_count_test.csv] and an "informations" subfolder with an infos_*.csv), 
this script:
1) Loads each per-set cell_count_*.csv in the folder.
2) Drops patches whose total #cells across all cell-type columns is < n_min_cells.
3) Writes filtered CSVs next to the originals with the suffix _{n_min_cells}.csv (e.g., cell_count_train_100.csv, cell_count_valid_100.csv, etc.)
4) Builds a new folder "informations_{n_min_cells}" replicating the plots that the original "informations" contains (tissues / slide_ids / cell_types) after filtering. 
For the infos_*.csv, it is filtered and saved as infos_*_{n_min_cells}.csv and used to produce the plots.
"""

import sys
from pathlib import Path
import argparse
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path



def find_cell_count_files(csv_dir: Path) -> Dict[str, Path]:
    """Return mapping {set_name: path} for exact cell_count_{train,valid,test}.csv files."""
    expected_files = {
        "train": csv_dir / "cell_count_train.csv",
        "valid": csv_dir / "cell_count_valid.csv",
        "test": csv_dir / "cell_count_test.csv",
    }
    mapping = {}
    for set_name, path in expected_files.items():
        if path.is_file():
            mapping[set_name] = path
    if not mapping:
        raise FileNotFoundError(
            f"No exact 'cell_count_train/valid/test.csv' files found under: {csv_dir}"
        )
    return mapping


def numeric_celltype_cols(df: pd.DataFrame) -> List[str]:
    """Return numeric columns that likely correspond to cell-type counts."""
    non_cell_cols = {"patch_id", "slide_set_scope", "set", "type"}
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in non_cell_cols]


def add_total_cells(df: pd.DataFrame, cell_cols: List[str], out_col: str = "_total_cells") -> pd.DataFrame:
    df = df.copy()
    df[out_col] = df[cell_cols].sum(axis=1)
    return df


def filter_cell_count_df(df: pd.DataFrame, n_min_cells: int) -> Tuple[pd.DataFrame, List[str]]:
    """Filter by total cells >= n_min_cells, return filtered df and kept patch IDs."""
    cell_cols = numeric_celltype_cols(df)
    if "patch_id" not in df.columns:
        raise KeyError("Expected column 'patch_id' in cell_count CSV.")
    df2 = add_total_cells(df, cell_cols)
    df2 = df2[df2["_total_cells"] >= n_min_cells].drop(columns=["_total_cells"])
    kept = df2["patch_id"].astype(str).tolist()
    return df2, kept


def read_infos_csv(csv_dir: Path) -> Optional[Path]:
    """Find infos_*.csv under 'informations' subfolder."""
    info_dir = csv_dir / "informations"
    if not info_dir.exists():
        return None
    candidates = sorted(info_dir.glob("infos_*.csv"))
    if not candidates:
        return None
    # If multiple exist, pick the first (usually there is only one).
    return candidates[0]




def create_plots_flexible(df: pd.DataFrame, cell_types_cols: List[str], save_dir: Path) -> None:
    """
    Replicates the original plotting as closely as possible:
      1) tissues_distribution.png (only if 'type' exists)
      2) slide_ids_distribution.png
      3) cell_types_distribution.png
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    # Ensure set has an order: train, valid, test, others...
    set_order = [s for s in ["train", "valid", "test"] if s in df["set"].unique().tolist()]
    other_sets = [s for s in df["set"].unique().tolist() if s not in set_order]
    df["set"] = pd.Categorical(df["set"], categories=set_order + other_sets, ordered=True)

    # 1) Tissues plot (only if 'type' column exists)
    if "type" in df.columns:
        fig1, axs = plt.subplots(1, 2, figsize=(20, 7))

        sns.countplot(data=df, x="type", hue="set", order=df["type"].unique(), ax=axs[0])
        axs[0].set_title("Patch count per tissue for each set")
        axs[0].set_xlabel("Tissue type")
        axs[0].set_ylabel("Patch count")
        axs[0].tick_params(axis="x", rotation=45)
        axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3)

        tissue_counts = df.groupby(["set", "type"], observed=True).size().reset_index(name="count")
        tissue_totals = tissue_counts.groupby("set", observed=True)["count"].transform("sum")
        tissue_counts["percent"] = (tissue_counts["count"] / tissue_totals) * 100
        pivot_tissue = tissue_counts.pivot(index="set", columns="type", values="percent").fillna(0)
        pivot_tissue.plot(kind="bar", stacked=True, ax=axs[1])
        axs[1].set_title("Percentage of tissue type for each set")
        axs[1].set_xlabel("set")
        axs[1].set_ylabel("Percentage of patches")
        axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)

        plt.tight_layout()
        plt.savefig(save_dir / "tissues_distribution.png")
        plt.close(fig1)

    # 2) Slide IDs plot (requires 'slide_id')
    if "slide_id" not in df.columns:
        df["slide_id"] = df["patch_id"].astype(str).str.rsplit("_", n=1).str[0]

    fig2, axs = plt.subplots(1, 2, figsize=(20, 7))
    sns.countplot(data=df, x="slide_id", hue="set", order=df["slide_id"].unique(), ax=axs[0])
    axs[0].set_title("Patch count per slide_id for each set")
    axs[0].set_xlabel("Slide ID")
    axs[0].set_ylabel("Patch Count")
    axs[0].tick_params(axis="x", rotation=45)
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=3)

    slide_counts = df.groupby(["set", "slide_id"], observed=True).size().reset_index(name="count")
    slide_totals = slide_counts.groupby("set", observed=True)["count"].transform("sum")
    slide_counts["percent"] = (slide_counts["count"] / slide_totals) * 100
    pivot_slide = slide_counts.pivot(index="set", columns="slide_id", values="percent").fillna(0)
    pivot_slide.plot(kind="bar", stacked=True, ax=axs[1])
    axs[1].set_title("Percentage for slide_id for each set")
    axs[1].set_xlabel("set")
    axs[1].set_ylabel("Percentage of patches")
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4)

    plt.tight_layout()
    plt.savefig(save_dir / "slide_ids_distribution.png")
    plt.close(fig2)

    # 3) Cell types plot (needs cell_types_cols)
    # Guard: ensure all requested cell type columns exist
    cell_types_cols = [c for c in cell_types_cols if c in df.columns]
    if cell_types_cols:
        fig3, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Sum counts per set
        cell_counts = df.groupby(["set"], observed=True)[cell_types_cols].sum().reset_index()
        ax1 = cell_counts.set_index("set")[cell_types_cols].plot(kind="bar", stacked=True, ax=axs[0], legend=False)
        axs[0].set_title("Cell types count for each set")
        axs[0].set_xlabel("set")
        axs[0].set_ylabel("Cell count")

        # Percentages per set
        totals = cell_counts[cell_types_cols].sum(axis=1).replace(0, np.nan)
        cell_perc = cell_counts[cell_types_cols].div(totals, axis=0) * 100
        cell_perc["set"] = cell_counts["set"]
        ax2 = cell_perc.set_index("set").plot(kind="bar", stacked=True, ax=axs[1], legend=False)
        axs[1].set_title("Cell types percentage for each set")
        axs[1].set_xlabel("set")
        axs[1].set_ylabel("Cell percentage")

        handles, labels = ax1.get_legend_handles_labels()
        fig3.legend(handles, labels, loc="lower center", ncol=4, title="Cell Types")
        plt.tight_layout(rect=[0, 0.18, 1, 1])
        plt.savefig(save_dir / "cell_types_distribution.png")
        plt.close(fig3)




def main(args):

    n_min_cells: int = args.n_min_cells
    csv_dir = get_full_path('train', 'training_datasets', args.dataset_id, args.training_dataset_id)

    if not csv_dir.exists():
        raise FileNotFoundError(f"Folder not found: {csv_dir}")

    print(f"\n[1/5] Scanning for per-set CSVs in: {csv_dir}")
    set_files = find_cell_count_files(csv_dir)
    print("Found:", {k: str(v.name) for k, v in set_files.items()})

    # Load, filter, and save per-set CSVs
    print(f"\n[2/5] Filtering each 'cell_count_*.csv' with threshold n_min_cells = {n_min_cells}")
    filtered_sets: Dict[str, pd.DataFrame] = {}
    kept_ids_by_set: Dict[str, List[str]] = {}
    example_cell_cols: Optional[List[str]] = None

    for set_name, path in set_files.items():
        print(f"  - {set_name}: reading {path.name}")
        df = pd.read_csv(path)
        if example_cell_cols is None:
            example_cell_cols = numeric_celltype_cols(df)
            print(f"    Detected {len(example_cell_cols)} cell-type columns: {example_cell_cols}")
        df_filt, kept_ids = filter_cell_count_df(df, n_min_cells=n_min_cells)
        filtered_sets[set_name] = df_filt
        kept_ids_by_set[set_name] = kept_ids

        # Save with suffix
        out_name = f"{path.stem}_{n_min_cells}.csv"
        out_path = path.with_name(out_name)
        df_filt.to_csv(out_path, index=False)
        removed = len(df) - len(df_filt)
        print(f"    Kept {len(df_filt)}/{len(df)} rows (removed {removed}). Wrote: {out_path.name}")

    # Aggregate all kept patch IDs for filtering infos
    all_kept_ids = set()
    for ids in kept_ids_by_set.values():
        all_kept_ids.update(ids)
    print(f"\nTotal unique kept patches across sets: {len(all_kept_ids)}")

    # Try to load infos_*.csv
    print("\n[3/5] Making new infos_*.csv under 'informations_{n_min_cells}'...")
    infos_path = read_infos_csv(csv_dir)
    print(f"Found infos file: {infos_path.name}")
    df_infos = pd.read_csv(infos_path)
    # Determine the patch ID column
    if "patch_id" not in df_infos.columns:
        raise KeyError(f"Expected column 'patch_id' in infos CSV: {infos_path.name}")

    # Filter infos both by kept ids
    df_infos_filt = df_infos[df_infos["patch_id"].isin(all_kept_ids)].copy()

    # Save filtered infos CSV
    info_out_dir = csv_dir / f"informations_{n_min_cells}"
    info_out_dir.mkdir(parents=True, exist_ok=True)

    # Build output name: infos_{dataset_id}_{n_min_cells}.csv
    # If original was "infos_ds_ct1_20x_224_o112_1.csv" -> "infos_ds_ct1_20x_224_o112_1_100.csv"
    info_out_name = infos_path.stem + f"_{n_min_cells}.csv"
    info_out_path = info_out_dir / f"{info_out_name}"
    df_infos_filt.to_csv(info_out_path, index=False)
    print(f"Wrote filtered infos CSV: {info_out_path.name} ({len(df_infos_filt)} rows)")

    # Create plots from filtered infos
    print(f"\n[5/5] Creating plots in {info_out_dir}")
    cell_types_cols = [c for c in (example_cell_cols or []) if c in df_infos_filt.columns]
    create_plots_flexible(df_infos_filt, cell_types_cols, info_out_dir)

    print("\nDone.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter per-set cell_count CSVs by a minimum total-cell threshold.")

    parser.add_argument("--dataset_id", type=str,
                        default="ds_ct1_20x_224_o112",
                        help="Dataset_id, e.g. ds_ct1_20x_224_o112")
    
    parser.add_argument("--training_dataset_id", type=str,
                        default="ds_ct1_20x_224_o112_1",
                        help="Training_dataset_id, e.g. ds_ct1_20x_224_o112_1")
    
    parser.add_argument("--n_min_cells", type=int, 
                        required=True, 
                        help="Minimum number of cells required to keep a patch - Look at the output of check_connectivity_in_patches.py to find the best threshold (recommanded between 5 and 10).")
    
    args = parser.parse_args()
    main(args)