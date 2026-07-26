#!/usr/bin/env bash
# Train CUT-4DFlow.
#   conda activate cut4dflow
#   ./scripts/train.sh [configs/train.yaml]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"

CONFIG="${1:-configs/train.yaml}"
# Optional overrides (else the config's paths are used):
#   CMRX_TRAIN_ROOT  extracted TrainSet root (see Data/README.md)
#   CMRX_CACHE_DIR   precomputed .pt cache dir
#   CMRX_SAVE_DIR    checkpoint output dir
export CMRX_TRAIN_ROOT="${CMRX_TRAIN_ROOT:-$REPO/Data/TaskR1R2/TrainSet/Aorta}"

PYTHONPATH=. python -m src.training.train_cmrx --config "$CONFIG"
