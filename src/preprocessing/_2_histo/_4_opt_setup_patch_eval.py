"""
Prepare folder for patch evaluation using the PanNuke cat for pre-trained CellViT.

This script creates a new folder <ROOT_HistoST> / train / datasets / <dataset_id> / do_patch_eval / <run_number> with the following files:
- cell_count_test.csv and types.csv with fake information to match the expected format for CellViT
- config.yaml: with the configuration to run CellViT with SAM-H backbone and the same training settings as in the original CellViT paper 
- dataset_config.yaml: with the same content as in the original CellViT paper (with all tissue types and nuclei types)

Skip this if given dataset_id does not have 256x256 patches.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Optional

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.constants import get_full_path



def build_test_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Create `cell_count_test.csv` with all counts in the first category."""
    count_cols = [c for c in df.columns if c != "patch_id"]
    totals = df[count_cols].sum(axis=1).astype(int)

    new_df = pd.DataFrame(
        {
            "Image": df["patch_id"],  # Rename "patch_id" to "Image" to match expected format for CellViT
            "Neoplastic": totals,
            "Inflammatory": np.zeros(len(df), dtype=int),
            "Connective": np.zeros(len(df), dtype=int),
            "Dead": np.zeros(len(df), dtype=int),
            "Epithelial": np.zeros(len(df), dtype=int),
        }
    )

    # Add .png to the end of each patch_id
    new_df["Image"] = new_df["Image"].apply(lambda x: f"{x}.png")
    
    return new_df



def load_slide_ids_from_txt(input_dir: Path, filename: str = "slide_ids.txt") -> List[str]:
    """Load slide IDs from <input_dir>/slide_ids.txt (one ID per line)."""
    path = input_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"--slide_ids not provided and slide id file not found: {path}"
        )
    with open(path, "r", encoding="utf-8") as f:
        slide_ids = [line.strip() for line in f if line.strip()]
    if not slide_ids:
        raise ValueError(f"Slide id file is empty: {path}")
    return slide_ids



def main(dataset_id: str, resolution: int, slide_ids: Optional[List[str]], run_number: str) -> None:
    
    input_dir = get_full_path('train', 'datasets', dataset_id)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # If slide_ids not provided, default to slide_ids.txt in the dataset_id folder
    if slide_ids is None:
        slide_ids = load_slide_ids_from_txt(input_dir, filename="slide_ids.txt")

    # Load original CSVs
    cell_counts_path = input_dir / "cell_count.csv"
    types_path = input_dir / "types.csv"
    cell_counts_df = pd.read_csv(cell_counts_path)
    types_df = pd.read_csv(types_path)

    # Keep only the requested slide IDs
    slide_set = set(slide_ids)
    cell_counts_df = cell_counts_df[
        cell_counts_df["patch_id"].str.rsplit(pat="_", n=1).str[0].isin(slide_set)
    ].reset_index(drop=True)
    types_df = types_df[
        types_df["patch_id"].str.rsplit(pat="_", n=1).str[0].isin(slide_set)
    ].reset_index(drop=True)

    print(f"\nKeeping {len(cell_counts_df)} patches from {len(slide_set)} slides.")
    print(f"Same for types.csv: {len(types_df)} patches for {len(slide_set)} slides.")

    # Build outputs
    out_dir = input_dir / "do_patch_eval" / run_number
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fake cell_count_test.csv
    print("\n* Building fake cell_count_test.csv")
    test_df = build_test_counts(cell_counts_df)
    test_df.to_csv(out_dir / "cell_count_test.csv", index=False)

    # 2. config.yaml
    print("* Building config.yaml")
    eval_checkpoint = f"CellViT-SAM-H-x{resolution}.pth"
    dataset_path = f"/lustre/fswork/projects/rech/<group>/<login>/ST_HE/HistoST/train/datasets/{dataset_id}/do_patch_eval/{run_number}"
    yaml_text = f"""logging:
  mode: offline
  project: Check_align
  notes: CellViT-SAM-H
  log_comment: CellViT-SAM-H_check-align
  tags:
  - Check-align
  - SAM-H
  wandb_dir: None
  level: Debug
  group: CellViT-SAM-H
random_seed: 19
gpu: 0
data:
  dataset: PanNuke
  dataset_path: {dataset_path}
  train_folds:
  - train
  val_folds:
  - valid
  test_folds:
  - test
  num_nuclei_classes: 6
  num_tissue_classes: 19
model:
  backbone: SAM-H
  pretrained_encoder: ./models/pretrained/SAM/sam_vit_h.pth
  shared_skip_connections: true
loss:
  nuclei_binary_map:
    focaltverskyloss:
      loss_fn: FocalTverskyLoss
      weight: 1
    dice:
      loss_fn: dice_loss
      weight: 1
  hv_map:
    mse:
      loss_fn: mse_loss_maps
      weight: 2.5
    msge:
      loss_fn: msge_loss_maps
      weight: 8
  nuclei_type_map:
    bce:
      loss_fn: xentropy_loss
      weight: 0.5
    dice:
      loss_fn: dice_loss
      weight: 0.2
    mcfocaltverskyloss:
      loss_fn: MCFocalTverskyLoss
      weight: 0.5
      args:
        num_classes: 6
  tissue_types:
    ce:
      loss_fn: CrossEntropyLoss
      weight: 0.1
training:
  drop_rate: 0
  attn_drop_rate: 0.1
  drop_path_rate: 0.1
  batch_size: 16
  epochs: 130
  optimizer: AdamW
  early_stopping_patience: 130
  scheduler:
    scheduler_type: exponential
    hyperparameters:
      gamma: 0.85
  optimizer_hyperparameter:
    betas:
    - 0.85
    - 0.95
    lr: 0.0003
    weight_decay: 0.0001
  unfreeze_epoch: 25
  sampling_gamma: 0.85
  sampling_strategy: cell+tissue
  mixed_precision: true
transformations:
  randomrotate90:
    p: 0.5
  horizontalflip:
    p: 0.5
  verticalflip:
    p: 0.5
  downscale:
    p: 0.15
    scale: 0.5
  blur:
    p: 0.2
    blur_limit: 10
  gaussnoise:
    p: 0.25
    var_limit: 50
  colorjitter:
    p: 0.2
    scale_setting: 0.25
    scale_color: 0.1
  superpixels:
    p: 0.1
  zoomblur:
    p: 0.1
  randomsizedcrop:
    p: 0.1
  elastictransform:
    p: 0.2
  normalize:
    mean:
    - 0.5
    - 0.5
    - 0.5
    std:
    - 0.5
    - 0.5
    - 0.5
eval_checkpoint: {eval_checkpoint}
run_sweep: false
agent: null
dataset_config:
  tissue_types:
    Adrenal_gland: 0
    Bile-duct: 1
    Bladder: 2
    Breast: 3
    Cervix: 4
    Colon: 5
    Esophagus: 6
    HeadNeck: 7
    Kidney: 8
    Liver: 9
    Lung: 10
    Ovarian: 11
    Pancreatic: 12
    Prostate: 13
    Skin: 14
    Stomach: 15
    Testis: 16
    Thyroid: 17
    Uterus: 18
  nuclei_types:
    Background: 0
    Neoplastic: 1
    Inflammatory: 2
    Connective: 3
    Dead: 4
    Epithelial: 5

"""
    (out_dir / "config.yaml").write_text(yaml_text, encoding="utf-8")

    # 3. Fake types.csv with all rows set to 'Breast'
    print("* Building fake types.csv")
    types_df["type"] = "Breast"
    types_df = types_df.rename(columns={"patch_id": "img"})  # Rename "patch_id" column to "img" to match expected format for CellViT
    types_df["img"] = types_df["img"].apply(lambda x: f"{x}.png")  # Add .png to the end of each patch_id in the "img" column
    types_df.to_csv(out_dir / "types.csv", index=False)

    # 4. dataset_config.yaml
    print("* Building dataset_config.yaml")
    yaml_text = """tissue_types:
  "Adrenal_gland": 0
  "Bile-duct": 1
  "Bladder": 2
  "Breast": 3
  "Cervix": 4
  "Colon": 5
  "Esophagus": 6
  "HeadNeck": 7
  "Kidney": 8
  "Liver": 9
  "Lung": 10
  "Ovarian": 11
  "Pancreatic": 12
  "Prostate": 13
  "Skin": 14
  "Stomach": 15
  "Testis": 16
  "Thyroid": 17
  "Uterus": 18

nuclei_types:
  "Background": 0
  "Neoplastic": 1
  "Inflammatory": 2
  "Connective": 3
  "Dead": 4
  "Epithelial": 5
"""
    (out_dir / "dataset_config.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"Done.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create do_patch_eval folder with needed files to run pretrained CellViT.")

    parser.add_argument(
        "--dataset_id",
        type=str,
        default="ds_ct1_20x_256_o112",
        help="Dataset ID, e.g. ds_ct1_20x_256_o112",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=20,
        help="Resolution for H&E patches (choose 20 for 20x or 40 for 40x patches)",
    )
    parser.add_argument(
        "--slide_ids",
        nargs="+",
        default=None,
        help="Slide IDs to include (e.g. breast_s0 kidney_s2 liver_s0). If not provided, will look for slide_ids.txt in the dataset folder.",
    )
    parser.add_argument(
        "--run_number",
        type=str,
        required=True,
        help="Run identifier for the output folder (e.g. run2) - useful to add new slides later",
    )
    args = parser.parse_args()
    main(args.dataset_id, args.resolution, args.slide_ids, args.run_number)

## CellViT requires 256x256 patches. Using it on 224x224 patches would need code changes, so this optional step is skipped for such datasets.