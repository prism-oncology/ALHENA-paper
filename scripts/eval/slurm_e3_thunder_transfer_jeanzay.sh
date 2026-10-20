#!/bin/bash
#SBATCH --job-name=histost_thunder_transfer            # job name
#SBATCH --output=./output/%x_%j.out     # output file (%j = job ID)
#SBATCH --error=./output/%x_%j.out      # error file (%j = job ID)
#SBATCH --partition=prepost             # partition with access to $STORE (no GPU)
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks=1                      # number of tasks (processes)
#SBATCH --cpus-per-task=2               # number of CPUs per task
#SBATCH --time=04:00:00                 # maximum run time (HH:MM:SS)
#SBATCH --hint=nomultithread            # disable hyperthreading
#SBATCH --account=<group>@a100              # A100 accounting
#SBATCH --mail-type=ALL                 # When to send email notifications
#SBATCH --mail-user=your.email@institution.org  # Your email address

# Copy the Thunder (Task C) results of encoder t9001e0 from $WORK to $STORE

module purge   # no module needed for rsync

SRC="/lustre/fswork/projects/rech/<group>/<login>/ST_HE/HistoST/eval/C_eval_thunder/HEencoder_t9001e0_he"
DST="/lustre/fsstor/projects/rech/<group>/<login>/ST_HE/HistoST/eval/C_eval_thunder/"

echo "[$(date '+%F %T')] START rsync"
echo "  SRC = ${SRC}"
echo "  DST = ${DST}"

mkdir -p "${DST}"   # create the destination directory

rsync -avh --partial --info=progress2 --stats "${SRC}" "${DST}"
rc=$?

echo "[$(date '+%F %T')] END rsync (exit code: ${rc})"
exit ${rc}   # propagate rsync's exit code to the job status
