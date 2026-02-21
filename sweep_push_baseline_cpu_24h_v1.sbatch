#!/bin/bash
#SBATCH --account=def-cneary
#SBATCH --job-name=push_base_cpu_v1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=1-64%8

set -euo pipefail

source ~/project/.venv/bin/activate
module load mujoco

cd ~/project/hypr_opt
mkdir -p logs

# Force CPU-only execution
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export OPTUNA_STORAGE_URL='postgresql+psycopg2://postgres.xcmxfgzjnqowuceuqdny:OptunahmsTITAN@aws-0-us-west-2.pooler.supabase.com:6543/postgres?sslmode=require'
export WORKER_STARTUP_JITTER_MAX_S=20

export FETCH_MODEL=baseline
export FETCH_TASK=push
export OPTUNA_STUDY=fetch_push_baseline_v1_cpu_24h
export BASELINE_SWEEP_MODE=full
export N_TRIALS=1

python optuna_worker_fetch.py
