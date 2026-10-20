#!/bin/bash
#SBATCH --job-name=histost_taskD_probe_t-1e-1                  # job name
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

# Deactivate inherited environments
conda deactivate

# Activate conda environment
conda activate pathobench_env

# Task D step 2: PathoBench linear probe on the Trident features
# Use --gpu 0 (not -1) under SLURM
python /linkhome/rech/<home_group>/<login>/ST_HE/HistoST/src/evaluation/part4_PathoBench/_2_run_pathobench.py \
    --eval_id eval_1 \
    --encoders t-1e-1 \
    --spaces he \
    --heads linprobe \
    --abmil_seeds 0 \
    --gpu 0 \
    --pathobench_version 0.0.1 \
    --trident_version 0.3.0
