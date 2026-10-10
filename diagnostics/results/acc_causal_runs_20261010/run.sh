#!/usr/bin/env bash
set -euo pipefail

GPU="$1"
cd /home/soenchun/cse/PointWorld_workspace/PointWorld

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1

PY=/home/soenchun/cse/PointWorld_workspace/envs/pointworld-env/bin/python
LOGROOT=diagnostics/results/acc_causal_runs_20261010

if [ "$GPU" = "2" ]; then
    MODES=(boundary)
    LIMIT=0
else
    MODES=(freq1 freq3)
    LIMIT=120
fi

for MODE in "${MODES[@]}"; do
    F="diagnostics/eval_acc_causal_geometry_${MODE}_20261010.py"
    RESULT="diagnostics/results/acc_causal_geometry_${MODE}_20261010"

    echo "SMOKE START $MODE"

    "$PY" "$F" --task task-0000 --smoke \
      > "$LOGROOT/${MODE}_smoke.log" 2>&1

    test -s "$RESULT/smoke_task-0000/mechanism_geometry_per_frame.csv"
    test -s "$RESULT/smoke_task-0000/mechanism_action_audit.csv"

    echo "SMOKE PASS $MODE"

    for TASK in task-0000 task-0001 task-0006 task-0008 task-0010; do
        echo "START $MODE $TASK limit=$LIMIT"

        "$PY" "$F" --task "$TASK" --limit "$LIMIT" \
          > "$LOGROOT/${MODE}_${TASK}.log" 2>&1

        test -s "$RESULT/$TASK/mechanism_geometry_per_frame.csv"
        test -s "$RESULT/$TASK/mechanism_action_audit.csv"

        echo "FINISHED $MODE $TASK"
    done

    echo "MODE COMPLETE $MODE"
done

echo "GPU${GPU} CAUSAL EXPERIMENTS COMPLETE"
