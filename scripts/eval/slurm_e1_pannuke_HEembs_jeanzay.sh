#!/bin/bash
#SBATCH --job-name=histost_e1_pannuke_HEembs_t46e13   # job name
#SBATCH --output=./output/%x_%j.out     # output file (%j = job ID)
#SBATCH --error=./output/%x_%j.out      # error file (%j = job ID)
#SBATCH --constraint=h100               # request A100 or H100 GPUs (80 GB)
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks=1                      # number of tasks (processes)
#SBATCH --ntasks-per-node=1             # number of tasks (processes) per node
#SBATCH --gres=gpu:1                    # number of GPUs per node
#SBATCH --cpus-per-task=24               # number of CPUs per task (sets the available memory; 24 for H100, 8 for A100)
#SBATCH --time=01:00:00                 # maximum run time (HH:MM:SS)
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

# PanNuke H&E patch and cell embeddings for both geometry recipes (native224, scale224)
srun python3 /linkhome/rech/<home_group>/<login>/ST_HE/HistoST/src/evaluation/part1_ST/task_A2_similarities/pannuke/_5_compute_HEembs.py \
    --geometry_recipe native224 scale224 \
    --training_num 46 \
    --checkpoint_num 13 \
    --batch_size 256 \
    --num_workers 22 \
    --device cuda \
    --dtype fp32
