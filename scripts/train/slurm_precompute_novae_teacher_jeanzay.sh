#!/bin/bash
#SBATCH --job-name=histost_precompute_novae_t4    # job name
#SBATCH --output=./output/%x_%j.out     # output file (%j = job ID)
#SBATCH --error=./output/%x_%j.out      # error file (%j = job ID)
#SBATCH --constraint=h100               # request H100 GPUs (80 GB)
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks=1                      # number of tasks (processes)
#SBATCH --ntasks-per-node=1             # number of tasks (processes) per node
#SBATCH --gres=gpu:1                    # number of GPUs per node
#SBATCH --cpus-per-task=24              # number of CPUs per task (sets the available memory; 24 for H100, 8 for A100)
#SBATCH --time=10:00:00                 # maximum run time (HH:MM:SS)
#SBATCH --hint=nomultithread            # disable hyperthreading
#SBATCH --account=<group>@h100              # H100 accounting
#SBATCH --mail-type=ALL                 # When to send email notifications
#SBATCH --mail-user=your.email@institution.org  # Your email address

# Clean inherited modules
module purge

# Select modules compiled for H100
module load arch/h100

# Load modules
module load miniforge/24.11.3
module load cuda/12.8.0

# Deactivate inherited environments
conda deactivate

# Activate conda environment (novae v1.1.1)
conda activate novae_env

# Precompute the frozen Novae teacher features (st_feat) of each patch
srun python -u /linkhome/rech/<home_group>/<login>/ST_HE/HistoST/src/dataset/precompute_novae_teacher.py \
    --novae_training_id training_4 \
    --dataset_id ds_ct1_20x_224_o112 \
    --training_dataset_id ds_ct1_20x_224_o112_1 \
    --folds train,valid,test \
    --device cuda
