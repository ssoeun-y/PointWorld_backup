#!/usr/bin/env bash

cd /home/soenchun/cse/PointWorld_workspace/PointWorld

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/soenchun/cse/PointWorld_workspace/PointWorld
export PYTHONUNBUFFERED=1

PY=/home/soenchun/cse/PointWorld_workspace/envs/pointworld-env/bin/python
EVAL=diagnostics/eval_acc_boundary_rigid_support_v9_3_current10457.py

FAILED=0

for TASK in task-0000 task-0001 task-0006 task-0008 task-0010
do
    echo
    echo "============================================================"
    echo "START $TASK"
    echo "============================================================"

    if ! "$PY" "$EVAL" --task "$TASK"; then
        echo "FAILED: $TASK"
        FAILED=1
        break
    fi
done

if [ "$FAILED" -eq 0 ]; then
    echo
    echo "============================================================"
    echo "COMBINE 5 TASKS"
    echo "============================================================"

    "$PY" "$EVAL" --combine
fi

echo
echo "============================================================"
echo "V9.3 FULL RUN FINISHED | FAILED=$FAILED"
echo "============================================================"
