#!/bin/bash
#SBATCH --job-name=histost_e2_TCGAfeatures_t46e13         # job name
#SBATCH --output=./output/%x_%j.out     # output file (%j = job ID)
#SBATCH --error=./output/%x_%j.out      # error file (%j = job ID)
#SBATCH --constraint=h100               # request A100 or H100 GPUs (80 GB)
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks=1                      # number of tasks (processes)
#SBATCH --ntasks-per-node=1             # number of tasks (processes) per node
#SBATCH --gres=gpu:1                    # number of GPUs per node
#SBATCH --cpus-per-task=24               # number of CPUs per task (sets the available memory; 24 for H100, 8 for A100)
#SBATCH --time=16:00:00                 # maximum run time (HH:MM:SS)
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
module load cuda/12.8.0

# Deactivate inherited environments
conda deactivate

# Activate conda environment
conda activate histost_env

# Task B: H&E encoder tile and case features for the TCGA benchmark (WSIs read from $SCRATCH)
# --num_workers must stay below --cpus-per-task
srun /linkhome/rech/<home_group>/<login>/ST_HE/HistoST/src/evaluation/part2_TCGA/taskB_tcga_eval/_0_extract_encoder_features.py \
    --benchmark_id tcga_benchmark_1 \
    --training_num 46 \
    --checkpoint_num 13 \
    --batch_size 256 \
    --num_workers 22 \
    --use_tile_rgb_cache never \
    --wsi_on_scratch \
    --resume \
    --device cuda
