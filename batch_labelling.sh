#!/usr/bin/env bash
#SBATCH --account=sarder-hubmap
#SBATCH --qos=sarder-hubmap
#SBATCH --job-name=tile_labelling
#SBATCH --output=logs/tile_labelling_%j.out
#SBATCH --error=logs/tile_labelling_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

set -euo pipefail

if [[ -n ${SLURM_JOB_ID:-} ]]; then
  module load conda
  conda activate "${LABEL_CONDA_ENV:-trident}"
fi

script_dir=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}
label_python=${LABEL_PYTHON:-python}
export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -ge 2 && $1 != -* ]]; then
  annotation_dir=$1
  job_dir=$2
  shift 2
  exec "$label_python" -m svc.batch_labelling \
    --annotation-dir "$annotation_dir" --job-dir "$job_dir" "$@"
fi

exec "$label_python" -m svc.batch_labelling "$@"
