#!/bin/bash
# Backwards-compatible name matching the documented labeled batch evaluation.
set -euo pipefail
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
exec bash "${SCRIPT_DIR}/batch_eval_wsi.sh" "$@"
