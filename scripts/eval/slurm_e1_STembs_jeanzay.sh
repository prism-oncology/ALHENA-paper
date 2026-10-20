#!/bin/bash
#SBATCH --job-name=histost_e1_STembs_t46e13         # job name
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
module load cuda/12.8.0

# Deactivate inherited environments
conda deactivate 

# Activate conda environment
conda activate histost_env

# ST embeddings for the given training and checkpoint
srun /linkhome/rech/<home_group>/<login>/ST_HE/HistoST/src/evaluation/part1_ST/_1_build_dataset/_10_compute_STembs.py \
    --dataset_id ds_ct1_20x_224_o112 \
    --training_dataset_id ds_ct1_20x_224_o112_1 \
    --training_id training_46 \
    --checkpoint_num 13 \
    --device cuda \
    --graph_batch_size 256 \
    --cell_batch_size 8192 \
    --overwrite
    # Needs the Novae teacher cache of the test fold (precompute_novae_teacher.py --folds train,valid,test)