#!/bin/bash
#SBATCH --job-name=histost_taskD_features_t-1e-1_0               # job name
#SBATCH --output=./output/%x_%j.out     # output file (%j = job ID)
#SBATCH --error=./output/%x_%j.out      # error file (%j = job ID)
#SBATCH --constraint=h100               # request A100 or H100 GPUs (80 GB)
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks=1                      # number of tasks (processes)
#SBATCH --ntasks-per-node=1             # number of tasks (processes) per node
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
module load miniforge/24.11.3

# Deactivate inherited environments
conda deactivate

# Activate conda environment
conda activate trident_env

# W&B offline mode
export WANDB_MODE=offline

# Trident tiling and H&E feature extraction for one encoder (all CPTAC cohorts)
# Run one job at a time (shared tiling directory), e.g. chained with: sbatch --dependency=singleton
# --max_workers must stay below --cpus-per-task
python /linkhome/rech/<home_group>/<login>/ST_HE/HistoST/src/evaluation/part4_PathoBench/_1_extract_features.py \
    --encoders t-1e-1 \
    --device cuda \
    --batch_limit 32 \
    --max_workers 16 \
    --wsi_on_scratch
