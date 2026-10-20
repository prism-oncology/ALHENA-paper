"""
Make train, valid, and test splits for a given dataset_id -> output for a given training_dataset_id.
Using explicit slide-level valid/test + per-slide spatial area split 77/9/14 (train/valid/test, from the stripe cuts at 0.77 and 0.86 of the longer axis) for remaining slides with no overlap.

Saved in <ROOT_HistoST> / train / training_datasets / <dataset_id> / <training_dataset_id> / :
    - informations/ folder: contains few information/analysis about the splits
        - cell_count_train.csv, cell_count_valid.csv, cell_count_test.csv : the cell count for each set, with an additional 'slide_set_scope' column indicating whether the patch's slide is 'specific' to that set (i.e. appears only in that set) or 'shared' (appears in multiple sets)
- slide_ids.txt : the list of slide ids used for this training_dataset_id (filtered from the original dataset_id slide_ids if --slide_ids provided, or same as original if not provided)
"""

import sys
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import List, Optional

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path



def load_slide_ids_from_txt(dataset_id_folder: Path, filename: str = "slide_ids.txt") -> List[str]:
    """Load slide IDs from <dataset_id_folder>/slide_ids.txt (one ID per line)."""
    path = Path(dataset_id_folder) / filename
    if not path.exists():
        raise FileNotFoundError(f"--slide_ids not provided and slide id file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        slide_ids = [line.strip() for line in f if line.strip()]
    if not slide_ids:
        raise ValueError(f"Slide id file is empty: {path}")
    return slide_ids



def save_slide_ids_to_txt(slide_ids: List[str], out_folder: Path, filename: str = "slide_ids.txt") -> Path:
    """Save slide IDs (one per line) to <out_folder>/slide_ids.txt."""
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    path = out_folder / filename
    with open(path, "w", encoding="utf-8") as f:
        for sid in slide_ids:
            sid = str(sid).strip()
            if sid:
                f.write(sid + "\n")
    return path



def make_final_df(slide_ids, dataset_id_folder):

    # Load data
    print("- Loading data...")
    df_cell_count = pd.read_csv(dataset_id_folder / "cell_count.csv")
    df_types = pd.read_csv(dataset_id_folder / "types.csv")
    df_patch_metrics = pd.read_csv(dataset_id_folder / "patch_metrics.csv")

    # Merge df_cell_count with df_types
    print("- Merging df_cell_count with df_types...")
    df_final = pd.merge(df_cell_count, df_types, on="patch_id")

    # Merge with df_patch_metrics
    print("- Merging with df_patch_metrics...")
    df_final = pd.merge(df_final, df_patch_metrics, on="patch_id")

    # Add slide_id column
    df_final['slide_id'] = df_final['patch_id'].str.rsplit('_', n=1).str[0]

    # Keep only the chosen slide_ids
    print("Slide_ids before filtering:", df_final['slide_id'].unique())
    print(len(df_final['slide_id'].unique()))
    df_final_filtered = df_final[df_final['slide_id'].isin(slide_ids)].copy()
    print("Slide_ids after filtering:", df_final_filtered['slide_id'].unique())
    print(len(df_final_filtered['slide_id'].unique()))

    # Get the cell types columns
    cell_types_cols = list(df_cell_count.columns)
    cell_types_cols.remove('patch_id')

    return df_final_filtered, cell_types_cols, df_cell_count



def split_dataset_spatial_regions(
    df_final_filtered: pd.DataFrame,
    val_slide_ids: Optional[List[str]] = None,
    test_slide_ids: Optional[List[str]] = None,
    border_margin_px: float = 0.0,
    seed: int = 42,
):
    """
    Spatial, region-based split with explicit slide-level valid/test assignment.
    - All patches from slides in `val_slide_ids` => 'valid'
    - All patches from slides in `test_slide_ids` => 'test'
    - Remaining slides are split by non-overlapping spatial regions into
      ~77% 'train', ~9% 'valid', ~14% 'test' (by contiguous stripes along the longer axis,
      cut at 0.77 and 0.86 of its extent: 'train' is the first stripe, then 'valid', then 'test').
    - Patches whose centers lie within `border_margin_px` of a region boundary are dropped.
    """
    df = df_final_filtered.copy()
    val_slide_ids = set(val_slide_ids or [])
    test_slide_ids = set(test_slide_ids or [])

    overlap = val_slide_ids & test_slide_ids
    if overlap:
        raise ValueError(f"[WARN] slide_ids present in both val and test lists: {sorted(overlap)}.")

    # Initialize
    df['set'] = 'att'
    train_patch_ids, valid_patch_ids, test_patch_ids = [], [], []

    # Assign explicit slides first
    if val_slide_ids:
        m = df['slide_id'].isin(val_slide_ids)
        valid_patch_ids.extend(df.loc[m, 'patch_id'].tolist())
    if test_slide_ids:
        m = df['slide_id'].isin(test_slide_ids)
        test_patch_ids.extend(df.loc[m, 'patch_id'].tolist())

    remaining_slides = sorted(set(df['slide_id'].unique()) - val_slide_ids - test_slide_ids)
    print(f"Using {len(val_slide_ids)} slides for valid only with {len(valid_patch_ids)} patches, {len(test_slide_ids)} slides for test only with {len(test_patch_ids)} patches, and {len(remaining_slides)} remaining slides for spatial splitting.")
    rng = np.random.default_rng(seed)

    for sid in remaining_slides:

        print(f"\nProcessing slide_id: {sid}")
        d = df[df['slide_id'] == sid].copy()
        if d.empty:
            continue

        x0, x1 = d['x_min'].min(), d['x_max'].max()
        y0, y1 = d['y_min'].min(), d['y_max'].max()
        w, h = x1 - x0, y1 - y0

        if w >= h:
            # Vertical stripes along X
            t1 = x0 + 0.77 * w
            t2 = x0 + 0.86 * w
            centers = (d['x_min'] + d['x_max']) / 2.0

            assign_train = centers < t1
            assign_valid = (centers >= t1) & (centers < t2)
            assign_test  = centers >= t2

            print(f"  - w={w:.1f} >= h={h:.1f} => vertical stripes")
            print(f"  - t1={t1:.1f}, t2={t2:.1f}")
            print(f"  - train: {assign_train.sum()}, valid: {assign_valid.sum()}, test: {assign_test.sum()}")

            if border_margin_px > 0:
                near_t1 = np.abs(centers - t1) < border_margin_px
                near_t2 = np.abs(centers - t2) < border_margin_px
                assign_train = assign_train & (~near_t1)
                assign_valid = assign_valid & (~(near_t1 | near_t2))
                assign_test  = assign_test  & (~near_t2)
                print(f"  - after applying border_margin_px={border_margin_px}:")
                total_patches_slide = assign_train.sum() + assign_valid.sum() + assign_test.sum()
                print(f"    train: {assign_train.sum()} ({assign_train.sum() / total_patches_slide * 100:.1f}%), valid: {assign_valid.sum()} ({assign_valid.sum() / total_patches_slide * 100:.1f}%), test: {assign_test.sum()} ({assign_test.sum() / total_patches_slide * 100:.1f}%)")

        else:
            # Horizontal stripes along Y
            s1 = y0 + 0.77 * h
            s2 = y0 + 0.86 * h
            centers = (d['y_min'] + d['y_max']) / 2.0

            assign_train = centers < s1
            assign_valid = (centers >= s1) & (centers < s2)
            assign_test  = centers >= s2

            print(f"  - w={w:.1f} < h={h:.1f} => horizontal stripes")
            print(f"  - s1={s1:.1f}, s2={s2:.1f}")
            print(f"  - train: {assign_train.sum()}, valid: {assign_valid.sum()}, test: {assign_test.sum()}")

            if border_margin_px > 0:
                near_s1 = np.abs(centers - s1) < border_margin_px
                near_s2 = np.abs(centers - s2) < border_margin_px
                assign_train = assign_train & (~near_s1)
                assign_valid = assign_valid & (~(near_s1 | near_s2))
                assign_test  = assign_test  & (~near_s2)
                print(f"  - after applying border_margin_px={border_margin_px}:")
                total_patches_slide = assign_train.sum() + assign_valid.sum() + assign_test.sum()
                print(f"    train: {assign_train.sum()} ({assign_train.sum() / total_patches_slide * 100:.1f}%), valid: {assign_valid.sum()} ({assign_valid.sum() / total_patches_slide * 100:.1f}%), test: {assign_test.sum()} ({assign_test.sum() / total_patches_slide * 100:.1f}%)")

        train_patch_ids.extend(d.loc[assign_train, 'patch_id'].tolist())
        valid_patch_ids.extend(d.loc[assign_valid, 'patch_id'].tolist())
        test_patch_ids.extend(d.loc[assign_test,  'patch_id'].tolist())

    # Write back to df
    df_final_filtered['set'] = 'att'
    df_final_filtered.loc[df_final_filtered['patch_id'].isin(train_patch_ids), 'set'] = 'train'
    df_final_filtered.loc[df_final_filtered['patch_id'].isin(valid_patch_ids), 'set'] = 'valid'
    df_final_filtered.loc[df_final_filtered['patch_id'].isin(test_patch_ids),  'set'] = 'test'

    return df_final_filtered, train_patch_ids, valid_patch_ids, test_patch_ids




def create_plots(df_final_filtered, cell_types_cols, save_dir):

    df = df_final_filtered.copy()

    set_order = [s for s in ['train', 'valid', 'test'] if s in df['set'].unique().tolist()]
    if not set_order:
        set_order = sorted(df['set'].unique().tolist())
    df['set'] = pd.Categorical(df['set'], categories=set_order, ordered=True)

    # Figure 1: tissues
    print("- Making fig1...")
    fig1, axs = plt.subplots(1, 2, figsize=(20, 7))

    # Subplot 1: Patch count per tissue for each set
    sns.countplot(data=df, x="type", hue="set", order=df['type'].unique(), ax=axs[0])
    axs[0].set_title("Patch count per tissue for each set")
    axs[0].set_xlabel("Tissue type")
    axs[0].set_ylabel("Patch count")
    axs[0].tick_params(axis='x', rotation=90)
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=3)  # Legend below plot
    
    # Subplot 2: Percentage of tissue type for each set
    n_colors = len(df['type'].unique())
    hues = np.linspace(0, 1, n_colors, endpoint=False)  # Evenly distribute hues
    distinguishable_palette = [sns.hls_palette(1, l, s)[0] for l, s in zip(hues, [0.6] * n_colors)]  # Varying hue, constant saturation
    random.shuffle(distinguishable_palette) # Shuffle the palette

    tissue_counts = df.groupby(['set', 'type'], observed=True).size().reset_index(name='count')
    tissue_totals = tissue_counts.groupby('set', observed=True)['count'].transform('sum')
    tissue_counts['percent'] = (tissue_counts['count'] / tissue_totals) * 100
    pivot_tissue = tissue_counts.pivot(index="set", columns="type", values="percent").fillna(0)
    pivot_tissue.plot(kind="bar", stacked=True, ax=axs[1], color=distinguishable_palette)
    axs[1].set_title("Percentage of tissue type for each set")
    axs[1].set_xlabel("set")
    axs[1].set_ylabel("Percentage of patches")
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)  # Legend below plot

    plt.tight_layout()
    plt.savefig(f"{save_dir}/tissues_distribution.png")

    # Figure 2: slide_ids
    print("- Making fig2...")
    fig2, axs = plt.subplots(1, 2, figsize=(20, 7))

    # Subplot 1: Patch count per slide_id for each set
    sns.countplot(data=df, x="slide_id", hue="set", order=df['slide_id'].unique(), ax=axs[0])
    axs[0].set_title("Patch count per slide_id for each set")
    axs[0].set_xlabel("Slide ID")
    axs[0].set_ylabel("Patch Count")
    axs[0].tick_params(axis='x', rotation=90)
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.38), ncol=3)  # Legend below plot

    # Subplot 2: Percentage for slide_id for each set
    n_colors = len(df['slide_id'].unique())
    hues = np.linspace(0, 1, n_colors, endpoint=False)  # Evenly distribute hues
    distinguishable_palette = [sns.hls_palette(1, l, s)[0] for l, s in zip(hues, [0.6] * n_colors)]  # Varying hue, constant saturation
    random.shuffle(distinguishable_palette) # Shuffle the palette

    slide_counts = df.groupby(['set', 'slide_id'], observed=True).size().reset_index(name='count')
    slide_totals = slide_counts.groupby('set', observed=True)['count'].transform('sum')
    slide_counts['percent'] = (slide_counts['count'] / slide_totals) * 100
    pivot_slide = slide_counts.pivot(index="set", columns="slide_id", values="percent").fillna(0)
    pivot_slide.plot(kind="bar", stacked=True, ax=axs[1], color=distinguishable_palette)
    axs[1].set_title("Percentage for slide_id for each set")
    axs[1].set_xlabel("set")
    axs[1].set_ylabel("Percentage of patches")
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4)  # Legend below plot

    plt.tight_layout()
    plt.savefig(f"{save_dir}/slide_ids_distribution.png")

    # Figure 3: cell types
    print("- Making fig3...")
    fig3, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Cell types count for each set
    cell_counts = df.groupby(['set'], observed=True)[cell_types_cols].sum().reset_index()
    ax1 = cell_counts.set_index("set")[cell_types_cols].plot(kind="bar", stacked=True, ax=axs[0], colormap="tab20", legend=False)
    axs[0].set_title("Cell types count for each set")
    axs[0].set_xlabel("set")
    axs[0].set_ylabel("Cell count")

    # Subplot 2: Cell types percentage for each set
    cell_totals = cell_counts[cell_types_cols].sum(axis=1)
    cell_percentages = cell_counts[cell_types_cols].div(cell_totals, axis=0) * 100
    cell_percentages['set'] = cell_counts['set']
    
    ax2 = cell_percentages.set_index("set").plot(kind="bar", stacked=True, ax=axs[1], colormap="tab20", legend=False)
    axs[1].set_title("Cell types percentage for each set")
    axs[1].set_xlabel("set")
    axs[1].set_ylabel("Cell percentage")

    # Create a common legend from the first plot
    handles, labels = ax1.get_legend_handles_labels()
    fig3.legend(handles, labels, loc='lower center', ncol=4, title="Cell Types")

    # Adjust layout to leave space for the legend
    plt.tight_layout(rect=[0, 0.18, 1, 1])  # Leave space at the bottom for the legend
    plt.savefig(f"{save_dir}/cell_types_distribution.png")



def print_final_stats(df_final_filtered, exclusive_by_set):
    """Print final statistics for each set regarding shared vs specific slide patches"""

    tmp = df_final_filtered[df_final_filtered['set'].isin(['train', 'valid', 'test'])].copy()

    # Identify shared slides (present in >1 set)
    slide_to_sets = tmp.groupby('slide_id', observed=True)['set'].apply(lambda s: set(s))
    shared_slides = set(slide_to_sets[slide_to_sets.apply(lambda x: len(x) > 1)].index)

    # 1) Numbers and % per set for 'shared' slides only
    print("\n-- Patches per set (ONLY from SHARED slides) --")
    shared_df = tmp[tmp['slide_id'].isin(shared_slides)].copy()
    shared_total = len(shared_df)
    if shared_total == 0:
        print("No shared slides.")
    else:
        shared_counts = (
            shared_df['set']
            .value_counts()
            .reindex(['train', 'valid', 'test'])
            .fillna(0)
            .astype(int)
        )
        for s, n in shared_counts.items():
            pct = (n / shared_total * 100.0) if shared_total else 0.0
            print(f"{s:5s}: {n:6d} patches ({pct:5.1f}%)")
        print(f"TOTAL (shared only): {shared_total} patches")

    # 2) Numbers per set for 'specific' slides (exclusive to that set), with per-slide breakdown
    print("\n-- Patches per set from SPECIFIC slides (exclusive to that set) --")
    for s in ['train', 'valid', 'test']:
        spec_slides = exclusive_by_set.get(s, set())
        if not spec_slides:
            print(f"{s:5s}: 0 patches (no specific slides)")
            continue

        spec_df = tmp[(tmp['set'] == s) & (tmp['slide_id'].isin(spec_slides))]
        total_spec = len(spec_df)
        print(f"{s:5s}: {total_spec} patches from {len(spec_slides)} specific slide(s)")
        per_slide = (
            spec_df.groupby('slide_id', observed=True)
                   .size()
                   .sort_index()
        )
        for slide_id, n in per_slide.items():
            print(f"   - {slide_id}: {n} patches")




def main(args):

    # Create the output directories
    out_base = get_full_path('train', 'training_datasets', args.dataset_id, args.training_dataset_id, as_str=True)
    os.makedirs(out_base, exist_ok=True)
    save_info_dir = get_full_path('train', 'training_datasets', args.dataset_id, args.training_dataset_id, "informations", as_str=True)
    os.makedirs(save_info_dir, exist_ok=True)

    # Make the final DataFrame, keeping only the chosen slide_ids
    print("\n** Making the final DataFrame...")
    dataset_id_folder = get_full_path('train', 'datasets', args.dataset_id, as_str=True)
    
    # If slide_ids not provided, default to slide_ids.txt in the dataset folder
    slide_ids = args.slide_ids
    if slide_ids is None:
        slide_ids = load_slide_ids_from_txt(Path(dataset_id_folder), filename="slide_ids.txt")
    
    # De-duplicate (safer) while preserving order
    slide_ids = list(dict.fromkeys(slide_ids))
    # Re-save the effective slide ids list in the output folder (training_dataset_id folder)
    saved_path = save_slide_ids_to_txt(slide_ids, Path(out_base), filename="slide_ids.txt")
    print(f"Saved slide_ids.txt to: {saved_path}")

    df_final, cell_types_cols, df_cell_count = make_final_df(slide_ids, Path(dataset_id_folder))

    # Filter patches based on the provided metric ranges
    print("\n** Filtering patches based on provided metric intervals...")
    df_final_filtered = df_final.copy()
    if 'Dice' in df_final.columns and 'Jaccard' in df_final.columns and 'bPQ' in df_final.columns:
        for metric, sel in {
            "Dice":    args.dice_selection,
            "Jaccard": args.jaccard_selection,
            "bPQ":     args.bpq_selection,
        }.items():
            if sel is not None:
                vmin, vmax = map(float, sel)
                print(f"   - {metric}: keeping [{vmin}, {vmax}]")
                df_final_filtered = df_final_filtered[
                    (df_final_filtered[metric] >= vmin) &
                    (df_final_filtered[metric] <= vmax)
                ]
    else:
        print("   - Warning: One or more of the metrics 'Dice', 'Jaccard', 'bPQ' are not present in df_final columns. No filtering applied.")
    print(f"Kept {len(df_final_filtered)} patches over {len(df_final)} in total.")
    print(f"\nNumber of patches per slide_id BEFORE:\n {df_final.groupby('slide_id', observed=True).size()}")
    print(f"\nNumber of patches per slide_id AFTER:\n {df_final_filtered.groupby('slide_id', observed=True).size()}")

    # Split the dataset
    print("\n** Splitting the dataset into train / valid / test using spatial regions strategy and specific slide IDs...")
    print(f"   - Explicit VALID slides: {args.val_slide_ids}")
    print(f"   - Explicit TEST slides : {args.test_slide_ids}")
    print(f"   - Border margin (px)   : {args.border_margin_px}")
    df_final_filtered, train_patch_ids, valid_patch_ids, test_patch_ids = split_dataset_spatial_regions(
        df_final_filtered=df_final_filtered,
        val_slide_ids=args.val_slide_ids,
        test_slide_ids=args.test_slide_ids,
        border_margin_px=float(args.border_margin_px),
        seed=42,
    )
    print(f"\nSplitting results:\n {df_final_filtered.groupby(['slide_id', 'set'], observed=True).size()}")

    # Create plots
    print("\n** Creating plots...")
    create_plots(df_final_filtered, cell_types_cols, save_info_dir)

    # Save the final DataFrame
    print("\n** Saving the final DataFrame...")
    print(df_final_filtered.head())
    df_final_filtered.to_csv(f"{save_info_dir}/infos_{args.training_dataset_id}.csv", index=False)

    # Saving the cell count for each set (+ scope column)
    print("\n** Saving the cell count for each set (with slide scope)...")

    # Determine which slides are exclusive to a single set vs shared across sets
    tmp = df_final_filtered[df_final_filtered['set'].isin(['train', 'valid', 'test'])].copy()
    slide_to_sets = tmp.groupby('slide_id', observed=True)['set'].apply(lambda s: set(s))
    exclusive_by_set = {s: set(slide_to_sets[slide_to_sets.apply(lambda x: x == {s})].index) for s in ['train', 'valid', 'test'] if s in tmp['set'].unique()}

    def _save_with_scope(patch_ids, set_name, filename):
        if not patch_ids:
            return
        df_out = df_cell_count[df_cell_count['patch_id'].isin(patch_ids)].copy()
        slide_ids_series = df_out['patch_id'].str.rsplit('_', n=1).str[0]
        is_exclusive = slide_ids_series.isin(exclusive_by_set.get(set_name, set()))
        # 'specific' => patches from slides assigned exclusively to this set
        # 'shared'   => patches from slides that appear in multiple sets
        df_out['slide_set_scope'] = np.where(is_exclusive, 'specific', 'shared')
        df_out.to_csv(os.path.join(out_base, filename), index=False)
    
    _save_with_scope(train_patch_ids, 'train', 'cell_count_train.csv')
    print(f"Saved train set with {len(train_patch_ids)} patches")
    _save_with_scope(valid_patch_ids, 'valid', 'cell_count_valid.csv')
    print(f"Saved valid set with {len(valid_patch_ids)} patches")
    _save_with_scope(test_patch_ids,  'test',  'cell_count_test.csv')
    print(f"Saved test set with {len(test_patch_ids)} patches")

    print("\n** Final stats **")
    print_final_stats(df_final_filtered, exclusive_by_set)

    print("Done.")
  


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Make train/valid/test  splits")
    
    # 1. Input (<dataset_id>) and output (<training_dataset_id>)
    parser.add_argument("--dataset_id", type=str, default="ds_ct1_20x_224_o112", help="Dataset_id, e.g. ds_ct1_20x_224_o112")
    parser.add_argument("--training_dataset_id", type=str, default="ds_ct1_20x_224_o112_1", help="Training_dataset_id, e.g. ds_ct1_20x_224_o112_1")

    # 2. Data filtering arguments
    parser.add_argument("--slide_ids", type=str, nargs='+', default=None, help="List of slide ids to keep for the given dataset_id (If None, using slide_ids.txt in dataset_id folder)")
    parser.add_argument("--dice_selection", nargs=2, metavar=('MIN','MAX'), default=[0.0, 1.0], type=float, help="Interval to keep patches based on the Dice metric")
    parser.add_argument("--jaccard_selection", nargs=2, metavar=('MIN','MAX'), default=[0.0, 1.0], type=float, help="Interval to keep patches based on the Jaccard metric")
    parser.add_argument("--bpq_selection", nargs=2, metavar=('MIN','MAX'), default=[0.0, 1.0], type=float, help="Interval to keep patches based on the bPQ metric")

    # 3. Split strategy arguments (train/valid/test sets)
    parser.add_argument("--val_slide_ids", type=str, nargs='*', default=["breast_s3"], help="List of slide_ids to assign entirely to the VALIDATION set (only used with --split_strategy spatial_regions).")
    parser.add_argument("--test_slide_ids", type=str, nargs='*', default=["kidney_s0", "pancreatic_s2"], help="List of slide_ids to assign entirely to the TEST set (only used with --split_strategy spatial_regions).")
    parser.add_argument("--border_margin_px", type=float, default=112, help="(Optional) Margin in pixels around region boundaries to DROP patches (helps avoid overlap bleed due to patch overlap).")

    args = parser.parse_args()
    main(args)