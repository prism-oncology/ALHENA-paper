#!/bin/bash
#SBATCH --job-name=histost_lora_drift    # job name
#SBATCH --output=./output/%x_%j.out      # output file (%j = job ID)
#SBATCH --error=./output/%x_%j.out       # error file (%j = job ID)
#SBATCH --constraint=h100                # request H100 GPUs (80 GB)
#SBATCH --nodes=1                        # number of nodes
#SBATCH --ntasks=1                       # number of tasks (processes)
#SBATCH --ntasks-per-node=1              # number of tasks (processes) per node
#SBATCH --gres=gpu:1                     # number of GPUs per node (not used by lora_drift.py)
#SBATCH --cpus-per-task=8                # number of CPUs per task
#SBATCH --time=02:00:00                  # maximum run time (HH:MM:SS)
#SBATCH --hint=nomultithread             # disable hyperthreading
#SBATCH --account=<group>@h100               # H100 accounting
#SBATCH --mail-type=ALL                  # When to send email notifications
#SBATCH --mail-user=your.email@institution.org  # Your email address

# lora_drift.py only uses the CPU (it can also be run directly on a login node)

# Clean inherited modules
module purge

# Select modules compiled for H100
module load arch/h100

# Load modules
module load miniforge/24.11.3

# Deactivate inherited environments
conda deactivate

# Activate conda environment
conda activate histost_env

# LoRA drift and intruder dimensions of the H&E backbone vs frozen UNI2-h, for the listed trainings and epochs

srun python -u /linkhome/rech/<home_group>/<login>/ST_HE/HistoST/src/training/lora_drift.py \
    --training_ids training_41 training_40 training_42 training_45 training_44 training_43 training_46 \
    --epochs 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --intruder \
    --markdown
