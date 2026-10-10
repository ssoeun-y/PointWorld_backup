#!/usr/bin/env bash
set -euo pipefail

cd /home/soenchun/cse/PointWorld_workspace/PointWorld

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/soenchun/cse/PointWorld_workspace/PointWorld
export WANDB_MODE=disabled

PY=/home/soenchun/cse/PointWorld_workspace/envs/pointworld-env/bin/python

ROOT=diagnostics/results/rigid_generation_ablation
mkdir -p "$ROOT"

echo "===== lambda 0 control ====="
$PY diagnostics/train_rigid_generation_ablation.py \
  --lambda-rigid 0 \
  --steps 5000 \
  --tag lam0 \
  --eval-pre \
  2>&1 | tee "$ROOT/lam0.log"

echo "===== lambda 3435.536 ====="
$PY diagnostics/train_rigid_generation_ablation.py \
  --lambda-rigid 3435.5363005687427 \
  --steps 5000 \
  --tag lam3436 \
  2>&1 | tee "$ROOT/lam3436.log"

echo "===== lambda 8588.841 ====="
$PY diagnostics/train_rigid_generation_ablation.py \
  --lambda-rigid 8588.840751421856 \
  --steps 5000 \
  --tag lam8589 \
  2>&1 | tee "$ROOT/lam8589.log"

echo "===== lambda 17177.682 ====="
$PY diagnostics/train_rigid_generation_ablation.py \
  --lambda-rigid 17177.681502843712 \
  --steps 5000 \
  --tag lam17178 \
  2>&1 | tee "$ROOT/lam17178.log"

echo "===== ALL ABLATIONS FINISHED ====="
