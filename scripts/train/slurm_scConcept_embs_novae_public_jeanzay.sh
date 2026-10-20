#!/bin/bash
#SBATCH --job-name=histost_scc_novae_public  # job name
#SBATCH --output=./output/%x_%j.out     # output file (%j = job ID)
#SBATCH --error=./output/%x_%j.out      # error file (%j = job ID)
#SBATCH --constraint=h100               # request H100 GPUs (80 GB)
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks=1                      # number of tasks (processes)
#SBATCH --ntasks-per-node=1             # number of tasks (processes) per node
#SBATCH --gres=gpu:1                    # number of GPUs per node
#SBATCH --cpus-per-task=24               # number of CPUs per task (sets the available memory; 24 for H100, 8 for A100)
#SBATCH --time=05:00:00                 # maximum run time (HH:MM:SS)
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

# Activate conda environment
conda activate sc_concept_env2

# Export environment variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# scConcept cell embeddings (obsm["X_scConcept"]) for the public Novae cohort slides
srun /linkhome/rech/<home_group>/<login>/ST_HE/HistoST/src/pretraining_ST/novae_pretraining_st/_1_scConcept_embs_novae_public.py \
    --cohort_id human_cellbased_nobrain_scc40M30M_1 \
    --scconcept_training_id training_7 \
    --gene_id_column scconcept_gene_id \
    --batch_size 256 \
    --num_workers 20
