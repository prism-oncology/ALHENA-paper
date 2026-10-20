#!/bin/bash
#SBATCH --job-name=histost_thunder_t46e13_he         # job name
#SBATCH --output=./output/%x_%j.out     # output file (%j = job ID)
#SBATCH --error=./output/%x_%j.out      # error file (%j = job ID)
#SBATCH --constraint=a100               # request A100 or H100 GPUs (80 GB)
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks=1                      # number of tasks (processes)
#SBATCH --ntasks-per-node=1             # number of tasks (processes) per node
#SBATCH --gres=gpu:1                    # number of GPUs per node
#SBATCH --cpus-per-task=8               # number of CPUs per task (sets the available memory; 24 for H100, 8 for A100)
#SBATCH --time=20:00:00                 # maximum run time (HH:MM:SS)
#SBATCH --hint=nomultithread            # disable hyperthreading
#SBATCH --account=<group>@a100              # A100 or H100 accounting
#SBATCH --mail-type=ALL                 # When to send email notifications
#SBATCH --mail-user=your.email@institution.org  # Your email address

# Clean inherited modules
module purge 

# Select modules compiled for A100 or H100
module load arch/a100 

# Load modules
module load miniforge/24.11.3

# Deactivate inherited environments
conda deactivate 

# Activate conda environment
conda activate thunder_env

# Set wandb offline
export WANDB_MODE=offline

# Run Thunder benchmark
python /linkhome/rech/<home_group>/<login>/ST_HE/HistoST/src/evaluation/part3_Thunder/_1_run_thunder_benchmark.py \
    --training_num 46 \
    --checkpoint_num 13 \
    --space he \
    --datasets all \
    --tasks all_non_vlm \
    --num_workers 7

    # To recompute a (dataset, task) pair, delete its _done/<dataset>__<task>.json marker
