#!/bin/bash
# Submit labeled SVC evaluations; annotation subfolders match WSI stems.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 WSI_DIRECTORY ANNOTATION_ROOT [OUTPUT_ROOT]" >&2
  exit 2
fi

WSI_DIR="$(realpath "$1")"
ANNOTATION_ROOT="$(realpath "$2")"
OUTPUT_ROOT="${3:-${WSI_DIR%/}_cortex_medullary_eval}"
PROJECT_DIR="/blue/pinaki.sarder/j.fermin/cortex_medullary_identification"
SINGLE_JOB="${PROJECT_DIR}/scripts/single_wsi_eval.sh"

[[ -d "${WSI_DIR}" ]] || { echo "WSI directory does not exist: ${WSI_DIR}" >&2; exit 2; }
[[ -d "${ANNOTATION_ROOT}" ]] || { echo "Annotation directory does not exist: ${ANNOTATION_ROOT}" >&2; exit 2; }

mkdir -p "${OUTPUT_ROOT}/logs" "${PROJECT_DIR}/logs"
count=0
while IFS= read -r -d '' wsi; do
  ((count += 1))
  slide_name="$(basename "${wsi}")"
  slide_name="${slide_name%.*}"
  safe_name="${slide_name//[^[:alnum:]_-]/_}"
  sample_annotations="${ANNOTATION_ROOT}/${slide_name}"
  if [[ ! -f "${sample_annotations}/cortical_gt.json" && ! -f "${sample_annotations}/medullary_gt.json" ]]; then
    echo "Missing cortex/medullary GT for ${slide_name} under ${ANNOTATION_ROOT}" >&2
    exit 2
  fi
  echo "Submitting labeled evaluation for ${slide_name}"
  sbatch \
    --job-name="cortmed_eval_${safe_name:0:24}" \
    --export="ALL,WSI_PATH=${wsi},ANNOTATION_ROOT=${ANNOTATION_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}" \
    "${SINGLE_JOB}"
done < <(find "${WSI_DIR}" -type f \( \
  -iname '*.svs' -o -iname '*.tif' -o -iname '*.tiff' -o \
  -iname '*.ndpi' -o -iname '*.mrxs' -o -iname '*.scn' \
\) -print0 | sort -z)

if (( count == 0 )); then
  echo "No supported WSI files found under ${WSI_DIR}" >&2
  exit 1
fi
echo "Submitted ${count} labeled evaluation job(s); results: ${OUTPUT_ROOT}"
