#!/bin/bash
#SBATCH --account=def-cneary
#SBATCH --job-name=slide_eq_cpu_v3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=1-64

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
export PYTHONPATH=~/project/equivariant-rl:$PYTHONPATH

export FETCH_MODEL=equivariant
export FETCH_TASK=slide
export OPTUNA_STUDY=fetch_slide_equivariant_v3_cpu_24h
export N_TRIALS=1

python optuna_worker_fetch.py
