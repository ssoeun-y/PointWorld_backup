#!/usr/bin/env bash
set -euo pipefail

cd /home/soenchun/cse/PointWorld_workspace/PointWorld

export PYTHONPATH="$PWD:$PWD/diagnostics"
export CUDA_VISIBLE_DEVICES=""

PY=/home/soenchun/cse/PointWorld_workspace/envs/pointworld-env/bin/python
SCRIPT=diagnostics/eval_gt_world_comotion_524.py

for task in 0000 0001 0006 0008 0010; do
    echo "[START] task-$task $(date)"

    "$PY" -u "$SCRIPT" --task "$task"

    echo "[DONE] task-$task $(date)"
done

"$PY" "$SCRIPT" --analyze

echo "[ALL GT WORLD CO-MOTION VERIFIED]"
