#!/bin/bash
# Submit annotation-free SVC inference jobs for WSIs in a directory.

set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 WSI_DIRECTORY [OUTPUT_ROOT]" >&2
  exit 2
fi
WSI_DIR="$(realpath "$1")"
OUTPUT_ROOT="${2:-${WSI_DIR%/}_cortex_medullary_inference}"
PROJECT_DIR="/blue/pinaki.sarder/j.fermin/cortex_medullary_identification"
SINGLE_JOB="${PROJECT_DIR}/scripts/single_wsi_inference.sh"

[[ -d "${WSI_DIR}" ]] || { echo "WSI directory does not exist: ${WSI_DIR}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}/logs" "${PROJECT_DIR}/logs"

count=0
while IFS= read -r -d '' wsi; do
  ((count += 1))
  slide_name="$(basename "${wsi}")"
  slide_name="${slide_name%.*}"
  safe_name="${slide_name//[^[:alnum:]_-]/_}"
  echo "Submitting inference for ${slide_name}"
  sbatch \
    --job-name="cortmed_infer_${safe_name:0:24}" \
    --export="ALL,WSI_PATH=${wsi},OUTPUT_ROOT=${OUTPUT_ROOT}" \
    "${SINGLE_JOB}"
done < <(find "${WSI_DIR}" -type f \( \
  -iname '*.svs' -o -iname '*.tif' -o -iname '*.tiff' -o \
  -iname '*.ndpi' -o -iname '*.mrxs' -o -iname '*.scn' \
\) -print0 | sort -z)

if (( count == 0 )); then
  echo "No supported WSI files found under ${WSI_DIR}" >&2
  exit 1
fi
echo "Submitted ${count} inference job(s); outputs: ${OUTPUT_ROOT}"
