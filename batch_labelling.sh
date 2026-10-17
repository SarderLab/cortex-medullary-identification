#!/usr/bin/env bash
#SBATCH --account=sarder-hubmap
#SBATCH --qos=sarder-hubmap
#SBATCH --job-name=tile_labelling
#SBATCH --output=logs/tile_labelling_%j.out
#SBATCH --error=logs/tile_labelling_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail

if [[ -n ${SLURM_JOB_ID:-} ]]; then
  module load conda
  conda activate "${LABEL_CONDA_ENV:-${CORT_MED_CONDA_ENV:-/blue/pinaki.sarder/j.fermin/conda/envs/cort_med_gpu}}"
fi

# script_dir=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}
script_dir=${PROJECT_DIR:-/blue/pinaki.sarder/j.fermin/cortex_medullary_identification}
# label_python=${LABEL_PYTHON:-python}
export PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}"

echo $script_dir
echo $PYTHONPATH

if [[ $# -ge 2 && $1 != -* ]]; then
  annotation_dir=$1
  job_dir=$2
  shift 2
  exec python -m svc.batch_labelling \
    --save-gt-visualization --annotation-dir "$annotation_dir" --job-dir "$job_dir" "$@"
fi

exec python -m svc.batch_labelling "$@"
