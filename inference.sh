#!/usr/bin/env bash
#SBATCH --account=sarder-hubmap
#SBATCH --qos=sarder-hubmap
#SBATCH --job-name=svc_inference
#SBATCH --output=logs/svc_inference_%j.out
#SBATCH --error=logs/svc_inference_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load conda
conda activate "${CORT_MED_CONDA_ENV:-/blue/pinaki.sarder/j.fermin/conda/envs/cort_med_gpu}"
# --visualize saves raw and smoothed heatmaps plus thresholded class masks.
exec python -m scripts.inference --visualize "$@"


# sbatch inference.sh \
#   --xlsx "$SPLIT_XLSX" \
#   --h5-dir /orange/pinaki.sarder/j.fermin/cort_med_wsi/trident_processed_final/KPMP/40x_512px_384px_overlap \
#   --model "$MODEL" \
#   --split test \
#   --output-dir "$JOB_DIR/inference_jsons" \
#   --visualization-dir "$JOB_DIR/prediction_visualization"
