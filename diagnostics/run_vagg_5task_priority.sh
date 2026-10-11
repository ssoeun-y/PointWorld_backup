#!/usr/bin/env bash
cd /home/soenchun/cse/PointWorld_workspace/PointWorld || exit 1

PY=/home/soenchun/cse/PointWorld_workspace/envs/pointworld-env/bin/python
R="$PWD/diagnostics/results/vagg_20261011"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD"
export WANDB_MODE=disabled
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

for MODE in vagg speed_l2 ungated; do
    echo "START $MODE $(date)"
    "$PY" -u diagnostics/vagg_train_20261011.py \
        --variant "$MODE" \
        --steps 0 \
        --eval-clips 1000000 \
        --workers 2 \
        > "$R/eval_5task100_${MODE}.log" 2>&1

    RC=$?
    echo "EXIT_CODE=$RC" >> "$R/eval_5task100_${MODE}.log"
    echo "FINISH $MODE EXIT_CODE=$RC $(date)"
done
