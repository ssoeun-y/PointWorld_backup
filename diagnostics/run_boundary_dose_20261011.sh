#!/usr/bin/env bash
set -euo pipefail

cd /home/soenchun/cse/PointWorld_workspace/PointWorld

PERCENT="$1"
GPU="$2"
PY=/home/soenchun/cse/PointWorld_workspace/envs/pointworld-env/bin/python
SCRIPT="diagnostics/eval_acc_causal_geometry_dose${PERCENT}_20261011.py"
LOGDIR="diagnostics/results/acc_dose_runs_20261011"
mkdir -p "$LOGDIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

echo "SMOKE START dose${PERCENT} GPU${GPU}"

"$PY" "$SCRIPT" --task task-0000 --limit 1 \
    > "$LOGDIR/dose${PERCENT}_smoke.log" 2>&1 || {
    echo "SMOKE FAILED: inspect $LOGDIR/dose${PERCENT}_smoke.log"
    exit 1
}

echo "SMOKE PASS dose${PERCENT}"

for TASK in 0000 0001 0006 0008 0010; do
    echo "START dose${PERCENT} task-${TASK}"

    "$PY" "$SCRIPT" \
        --task "task-${TASK}" \
        --limit 0 \
        > "$LOGDIR/dose${PERCENT}_task${TASK}.log" 2>&1 || {
        echo "FAILED dose${PERCENT} task-${TASK}"
        exit 1
    }

    echo "FINISHED dose${PERCENT} task-${TASK}"
done

echo "DOSE${PERCENT} ALL TASKS COMPLETE"
