#!/usr/bin/env bash
#SBATCH --account=sarder-hubmap
#SBATCH --qos=sarder-hubmap
#SBATCH --job-name=cortex_med_svc
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch scripts/train_and_eval.sbatch SPLITS_CSV [MODEL_DIR]" >&2
  exit 2
fi

manifest=$1
model_dir=${2:-./models}
project_dir=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}

module load conda
conda activate "${SVC_CONDA_ENV:-rapids_svc}"
cd "$project_dir"
mkdir -p "$model_dir"

exec python -m svc.train \
  --manifest "$manifest" \
  --output-dir "$model_dir" \
  --n-components "${N_COMPONENTS:-1024}" \
  --C "${SVC_C:-2.0}" \
  --threshold "${SVC_THRESHOLD:-0.0}" \
  --cache-size "${SVC_CACHE_SIZE_MB:-4096}"
