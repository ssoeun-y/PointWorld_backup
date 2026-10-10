#!/usr/bin/env bash
set -euo pipefail

BASE=/home/soenchun/cse/PointWorld_workspace
OUT="$BASE/PointWorld/diagnostics/results/droid_sam2_20261010"
PY="$BASE/envs/pointworld-env/bin/python"
DEPS="$BASE/third_party/sam2_python_deps"
REPO="$BASE/third_party/sam2_official"

mkdir -p "$DEPS"

if [ ! -f "$OUT/dependencies_installed.ok" ]; then
    "$PY" -m pip install \
      --disable-pip-version-check \
      --no-deps \
      --target "$DEPS" \
      hydra-core==1.3.2 \
      omegaconf==2.3.0 \
      antlr4-python3-runtime==4.9.3 \
      iopath==0.1.10 \
      portalocker==2.10.1

    touch "$OUT/dependencies_installed.ok"
fi

export PYTHONPATH="$DEPS:$REPO:$BASE/PointWorld"
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1

cd "$BASE/PointWorld"

"$PY" "$OUT/run_one_image.py"
