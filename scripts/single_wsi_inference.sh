#!/bin/bash
#SBATCH --account=pinaki.sarder
#SBATCH --qos=pinaki.sarder
#SBATCH --job-name=cortmed_infer
#SBATCH --output=/blue/pinaki.sarder/j.fermin/cortex_medullary_identification/logs/wsi_infer_%j.out
#SBATCH --error=/blue/pinaki.sarder/j.fermin/cortex_medullary_identification/logs/wsi_infer_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1

set -euo pipefail
SVC_WSI_MODE="inference"
source "$(dirname "$(realpath "$0")")/single_wsi_job_common.sh"
