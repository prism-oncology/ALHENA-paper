#!/bin/bash
#SBATCH --job-name=histost_scRNAseq_scConcept_v2_t7      # job name
#SBATCH --output=./output/%x_%j.out     # output file (%j = job ID)
#SBATCH --error=./output/%x_%j.out      # error file (%j = job ID)
#SBATCH --constraint=h100               # request A100 or H100 GPUs (80 GB)
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks=1                      # number of tasks (processes)
#SBATCH --ntasks-per-node=1             # number of tasks (processes) per node
#SBATCH --gres=gpu:1                    # number of GPUs per node
#SBATCH --cpus-per-task=24               # number of CPUs per task (sets the available memory; 24 for H100, 8 for A100)
#SBATCH --time=04:00:00                 # maximum run time (HH:MM:SS)
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
conda activate sc_concept_env2

# Export environment variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Cell embedding extraction for all data
srun /linkhome/rech/<home_group>/<login>/ST_HE/HistoST/src/evaluation/part1_ST/scRNAseq_ref/_2_v2_scConcept_embs_scRNAseq.py \
    --sc_concept_training_id training_7 \
    --ref_json /linkhome/rech/<home_group>/<login>/ST_HE/HistoST/data/eval_scRNAseq_ref/ref_tissue_files.json \
    --gene_id_column gene_ids \
    --batch_size 256 \
    --num_workers 23 \
    --cells_per_chunk 100000 \
    --dtype fp16 \
    --chunk_rows 4096 \
    --device cuda \
    --overwrite