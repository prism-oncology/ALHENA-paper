#!/bin/bash
#SBATCH --job-name=checkalign_ds_ct1_20x_256_o112         # job name
#SBATCH --output=./output/%x_%j.out     # output file (%j = job ID)
#SBATCH --error=./output/%x_%j.out      # error file (%j = job ID)
#SBATCH --constraint=h100               # request A100 or H100 GPUs (80 GB)
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks=1                      # number of tasks (processes)
#SBATCH --gres=gpu:1                    # number of GPUs per node
#SBATCH --cpus-per-task=24               # number of CPUs per task (sets the available memory; 24 for H100, 8 for A100)
#SBATCH --time=20:00:00                 # maximum run time (HH:MM:SS)
#SBATCH --hint=nomultithread            # disable hyperthreading
#SBATCH --account=<group>@h100              # A100 or H100 accounting
#SBATCH --mail-type=ALL                 # When to send email notifications
#SBATCH --mail-user=your.email@institution.org  # Your email address

# Clean inherited modules
module purge 

# Select modules compiled for A100 or H100
module load arch/h100 

# Load modules
# No module to load for cuda or cudnn because already included in the conda env
module load miniforge/24.11.3

# Deactivate inherited environments
conda deactivate 

# Activate conda environment
conda activate cellvit_env

# Run inference script
srun /linkhome/rech/<home_group>/<login>/CellViT_for_STHELAR/cell_segmentation/inference/inference_cellvit_experiment_pannuke.py \
    --run_dir /lustre/fswork/projects/rech/<group>/<login>/ST_HE/HistoST/train/datasets/ds_ct1_20x_256_o112/do_patch_eval/run1/ \
    --checkpoint_name CellViT-SAM-H-x20.pth \
    --gpu 0 \
    --magnification 20