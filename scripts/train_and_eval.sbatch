#!/bin/bash
#SBATCH --account=pinaki.sarder
#SBATCH --qos=pinaki.sarder
#SBATCH --job-name=cortex_med_svc
#SBATCH --output=/blue/pinaki.sarder/j.fermin/cortex_medullary_identification/logs/train_%j.out
#SBATCH --error=/blue/pinaki.sarder/j.fermin/cortex_medullary_identification/logs/train_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1

set -euo pipefail

module load conda
conda activate "${SVC_CONDA_ENV:-/blue/pinaki.sarder/j.fermin/conda/envs/rapids_svc}"

PROJECT_DIR="/blue/pinaki.sarder/j.fermin/cortex_medullary_identification"
cd "${PROJECT_DIR}"
mkdir -p models

python -m svc.train \
  --xlsx "${TRAIN_XLSX:-/blue/pinaki.sarder/j.fermin/DigitAb_Master/Segmentation/csv_files/case_split_70_15_151.xlsx}" \
  --feature-root "${TRAIN_FEATURE_ROOT:-/orange/pinaki.sarder/j.fermin/cort_med_wsi/UNI_V1_features}" \
  --output-dir "${MODEL_DIR:-${PROJECT_DIR}/models}" \
  --n-components "${N_COMPONENTS:-1024}" \
  --C "${SVC_C:-2.0}" \
  --threshold "${SVC_THRESHOLD:-0.0}" \
  --cache-size "${SVC_CACHE_SIZE_MB:-4096}"
