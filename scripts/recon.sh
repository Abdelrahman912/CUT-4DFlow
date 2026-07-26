#!/usr/bin/env bash
# Reconstruct a ValidationSet with a trained checkpoint, writing the submission layout
# (COO .npz) + optional animations.
#   conda activate cut4dflow
#   CKPT=checkpoints/full_super_v2/best.ckpt ./scripts/recon.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"

CKPT="${CKPT:-checkpoints/full_super_v2/best.ckpt}"
VAL_ROOT="${VAL_ROOT:-$REPO/Data/TaskR1R2/ValidationSet/Aorta}"
OUT_DIR="${OUT_DIR:-outputs/recon}"
SUBPATH="${SUBPATH:-TaskR1R2/ValidationSet/Aorta}"
N="${N:--1}"; OVERLAP="${OVERLAP:-2}"; ANIM="${ANIM:-0}"

[ -f "$CKPT" ]     || { echo "no checkpoint: $CKPT"; exit 1; }
[ -d "$VAL_ROOT" ] || { echo "no data dir: $VAL_ROOT (see Data/README.md)"; exit 1; }

args=(--ckpt "$CKPT" --out-dir "$OUT_DIR" --val-root "$VAL_ROOT" --out-subpath "$SUBPATH" --n "$N" --overlap "$OVERLAP")
[ "$ANIM" = "1" ] && args+=(--anim)

PYTHONPATH=. python src/postprocess/batch_recon.py "${args[@]}"
