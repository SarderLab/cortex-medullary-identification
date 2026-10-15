#!/bin/bash
# Shared pipeline for the Slurm evaluation and inference entry points.

PROJECT_DIR="/blue/pinaki.sarder/j.fermin/cortex_medullary_identification"
if [[ -z "${WSI_PATH:-}" ]]; then
  echo "Set WSI_PATH to a .svs/.tif/.tiff slide before submitting this job." >&2
  exit 2
fi
if [[ "${SVC_WSI_MODE}" == "evaluation" && -z "${ANNOTATION_DIR:-${ANNOTATION_ROOT:-}}" ]]; then
  echo "Evaluation requires ANNOTATION_DIR (a sample folder or annotation root)." >&2
  exit 2
fi

WSI_PATH="$(realpath "${WSI_PATH}")"
SAMPLE_NAME="$(basename "${WSI_PATH}")"
SAMPLE_NAME="${SAMPLE_NAME%.*}"
ANNOTATION_INPUT="${ANNOTATION_DIR:-${ANNOTATION_ROOT:-}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/bulk_wsi_eval/results}"
PROCESSED_DIR="${OUTPUT_ROOT}/${SAMPLE_NAME}/trident_processed"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/models/svc_pipeline.pkl}"
OVERLAP_PERCENTAGE="${OVERLAP_PERCENTAGE:-0}"
SEGMENTER="${SEGMENTER:-hest}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-/home/j.fermin/.cache/huggingface/hub/models--MahmoodLab--uni/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0}"
TRIDENT_ENV="${TRIDENT_CONDA_ENV:-/blue/pinaki.sarder/j.fermin/conda/envs/trident}"
SVC_ENV="${SVC_CONDA_ENV:-/blue/pinaki.sarder/j.fermin/conda/envs/rapids_svc}"

mkdir -p "${PROJECT_DIR}/logs" "${PROCESSED_DIR}"
cd "${PROJECT_DIR}"
module load conda
conda activate "${TRIDENT_ENV}"

# Plan from WSI objective power and MPP. The patch covers 512 x 0.25 µm =
# 128 µm per side; a typical 20x / 0.5 µm slide therefore uses 256 px at 20x.
PATCH_PLAN="$(python -m svc.wsi_patch_plan "${WSI_PATH}" \
  --overlap-percentage "${OVERLAP_PERCENTAGE}")"
IFS=$'\t' read -r MAGNIFICATION PATCH_SIZE OVERLAP_PX ENCODER_IMG_SIZE MPP_X MPP_Y NATIVE_MAG EXPECTED_FOV_UM <<< "${PATCH_PLAN}"
printf 'WSI sampling plan: objective=%sx; mag=%sx; patch=%sx%s px; overlap=%s px (%s%%); MPP=%.5fx%.5f; encoder input=%sx%s; field=%.2f um\n' \
  "${NATIVE_MAG}" "${MAGNIFICATION}" "${PATCH_SIZE}" "${PATCH_SIZE}" \
  "${OVERLAP_PX}" "${OVERLAP_PERCENTAGE}" "${MPP_X}" "${MPP_Y}" \
  "${ENCODER_IMG_SIZE}" "${ENCODER_IMG_SIZE}" "${EXPECTED_FOV_UM}"

TRIDENT_ARGS=(
  single --
  --slide_path "${WSI_PATH}"
  --job_dir "${PROCESSED_DIR}"
  --segmenter "${SEGMENTER}"
  --patch_encoder uni_v1
  --patch_encoder_img_size "${ENCODER_IMG_SIZE}"
  --mag "${MAGNIFICATION}"
  --patch_size "${PATCH_SIZE}"
  --overlap "${OVERLAP_PX}"
  --gpu 0
  --batch_size "${TRIDENT_BATCH_SIZE:-1024}"
)
if [[ -n "${ENCODER_CHECKPOINT}" ]]; then
  TRIDENT_ARGS+=(--patch_encoder_ckpt_path "${ENCODER_CHECKPOINT}")
fi
trident "${TRIDENT_ARGS[@]}"

conda deactivate
conda activate "${SVC_ENV}"
PYTHON_ARGS=(
  -m svc.single_wsi
  --wsi "${WSI_PATH}"
  --processed-dir "${PROCESSED_DIR}"
  --model "${MODEL_PATH}"
  --magnification "${MAGNIFICATION}"
  --patch-size "${PATCH_SIZE}"
  --overlap-px "${OVERLAP_PX}"
  --encoder-img-size "${ENCODER_IMG_SIZE}"
  --mpp-x "${MPP_X}"
  --mpp-y "${MPP_Y}"
  --expected-field-of-view-um "${EXPECTED_FOV_UM}"
  --max-field-error-percent "${MAX_FIELD_ERROR_PERCENT:-10}"
  --threshold "${SVC_THRESHOLD:-0.0}"
  --gaussian-sigma "${GAUSSIAN_SIGMA:-0.25}"
  --refined-cell-size "${REFINED_CELL_SIZE:-128}"
  --skip-trident
)
if [[ "${SVC_WSI_MODE}" == "evaluation" ]]; then
  PYTHON_ARGS+=(--annotation-dir "${ANNOTATION_INPUT}")
fi
python "${PYTHON_ARGS[@]}"
