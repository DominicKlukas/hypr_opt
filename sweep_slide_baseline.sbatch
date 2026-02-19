#!/bin/bash
#SBATCH --account=def-cneary
#SBATCH --job-name=slide_base_opt
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:h100:1
#SBATCH --array=1-8

set -euo pipefail

module load mujoco
source ~/project/.venv/bin/activate

cd ~/project/hypr_opt
mkdir -p logs

export OPTUNA_STORAGE_URL='postgresql+psycopg2://postgres.xcmxfgzjnqowuceuqdny:OptunahmsTITAN@aws-0-us-west-2.pooler.supabase.com:6543/postgres?sslmode=require'

export FETCH_MODEL=baseline
export FETCH_TASK=slide
export OPTUNA_STUDY=fetch_slide_baseline_v1
export N_TRIALS=5

python optuna_worker_fetch.py
