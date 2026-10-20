#!/bin/bash
#SBATCH --job-name=histost_scConcept_t1         # job name
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
conda activate sc_concept_env

# Export environment variables
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Start running scConcept adaptation on train, and then cell embedding extraction for all data
srun /linkhome/rech/<home_group>/<login>/ST_HE/HistoST/src/pretraining_ST/scConcept_pretraining_st/v1_scConcept_cellEmb.py \
    --dataset_id ds_ct1_20x_224_o112 \
    --training_dataset_id ds_ct1_20x_224_o112_1 \
    --training_id_scConcept training_1 \
    --adapt_steps 50000 \
    --adapt_batch_size 128 \
    --max_train_cells_per_slide 100000 \
    --save_adapted_ckpt \
    --adapt_max_tokens 512 \
    --adapt_panel_selection random \
    --adapt_panel_size_min 256 \
    --adapt_gene_sampling_strategy top-nonzero \
    --adapt_loss_switch_step 0 \
    --extract_batch_size 256 \
    --extract_num_workers 20 \
    --no_cell_id_str
