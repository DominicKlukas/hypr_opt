#!/bin/bash
#SBATCH --job-name=optuna_test
#SBATCH --output=logs/%x_%j_%a.out
#SBATCH --error=logs/%x_%j_%a.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=1-100
#SBATCH --nodes=1

set -euo pipefail

# Always start from project root
cd /project/6107222/klukasd
mkdir -p logs

# Metadata (very useful for debugging / SQL queries)
export WHERE="cluster"
export RUN_ID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

# Run
/project/6107222/klukasd/.venv/bin/python optuna_worker.py

