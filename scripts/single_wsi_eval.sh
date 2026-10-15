#!/bin/bash
#SBATCH --account=sarder-hubmap
#SBATCH --qos=sarder-hubmap
#SBATCH --job-name=cortmed_wsi
#SBATCH --output=/blue/pinaki.sarder/j.fermin/cortex_medullary_identification/logs/wsi_%j.out
#SBATCH --error=/blue/pinaki.sarder/j.fermin/cortex_medullary_identification/logs/wsi_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1

set -euo pipefail

if [[ -z "${WSI_PATH:-}" ]]; then
  echo "Set WSI_PATH when submitting this job, for example:" >&2
  echo "  sbatch --export=ALL,WSI_PATH=/data/slide.svs scripts/single_wsi_eval.sh" >&2
  exit 2
fi

PROJECT_DIR="/blue/pinaki.sarder/j.fermin/cortex_medullary_identification"
WSI_PATH="$(realpath "${WSI_PATH}")"
SAMPLE_NAME="$(basename "${WSI_PATH}")"
SAMPLE_NAME="${SAMPLE_NAME%.*}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/bulk_wsi_eval/results}"
PROCESSED_DIR="${OUTPUT_ROOT}/${SAMPLE_NAME}/trident_processed"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/models/svc_pipeline.pkl}"
MAGNIFICATION="${MAGNIFICATION:-40}"
PATCH_SIZE="${PATCH_SIZE:-512}"
SEGMENTER="${SEGMENTER:-hest}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-/home/j.fermin/.cache/huggingface/hub/models--MahmoodLab--uni/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0}"
TRIDENT_ENV="${TRIDENT_CONDA_ENV:-/blue/pinaki.sarder/j.fermin/conda/envs/trident}"
SVC_ENV="${SVC_CONDA_ENV:-/blue/pinaki.sarder/j.fermin/conda/envs/rapids_svc}"

mkdir -p "${PROJECT_DIR}/logs" "${PROCESSED_DIR}"
cd "${PROJECT_DIR}"
module load conda

# TRIDENT performs tissue segmentation, patch-coordinate generation, and UNI-v1 embedding.
conda activate "${TRIDENT_ENV}"
TRIDENT_ARGS=(
  single --
  --slide_path "${WSI_PATH}"
  --job_dir "${PROCESSED_DIR}"
  --segmenter "${SEGMENTER}"
  --patch_encoder uni_v1
  --mag "${MAGNIFICATION}"
  --patch_size "${PATCH_SIZE}"
  --overlap 0
  --gpu 0
)
if [[ -n "${ENCODER_CHECKPOINT}" ]]; then
  TRIDENT_ARGS+=(--patch_encoder_ckpt_path "${ENCODER_CHECKPOINT}")
fi
trident "${TRIDENT_ARGS[@]}"

# The SVC runtime converts TRIDENT HDF5 features into a compact PT bundle and predicts.
conda deactivate
conda activate "${SVC_ENV}"
python -m svc.single_wsi \
  --wsi "${WSI_PATH}" \
  --processed-dir "${PROCESSED_DIR}" \
  --model "${MODEL_PATH}" \
  --magnification "${MAGNIFICATION}" \
  --patch-size "${PATCH_SIZE}" \
  --threshold "${SVC_THRESHOLD:-0.0}" \
  --gaussian-sigma "${GAUSSIAN_SIGMA:-0.25}" \
  --refined-cell-size "${REFINED_CELL_SIZE:-128}" \
#   --skip-trident
