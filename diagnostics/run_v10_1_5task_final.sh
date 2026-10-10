#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/soenchun/cse/PointWorld_workspace/PointWorld
PY=/home/soenchun/cse/PointWorld_workspace/envs/pointworld-env/bin/python
EVAL=$ROOT/diagnostics/eval_acc_boundary_rigid_support_v10_1_current10457.py

LOGDIR=$ROOT/diagnostics/logs/v10_1_5task_final
mkdir -p "$LOGDIR"

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$ROOT
export PYTHONUNBUFFERED=1

cd "$ROOT"

echo "========================================"
echo "V10.1 FINAL 5-TASK EVALUATION"
echo "physical GPU = 1"
echo "expected total clips = 2683"
echo "start = $(date)"
echo "========================================"

TASKS=(
  task-0000
  task-0001
  task-0006
  task-0008
  task-0010
)

for TASK in "${TASKS[@]}"; do
    echo
    echo "========================================"
    echo "START $TASK : $(date)"
    echo "========================================"

    "$PY" "$EVAL" \
      --task "$TASK" \
      2>&1 | tee "$LOGDIR/${TASK}.log"

    echo
    echo "========================================"
    echo "DONE $TASK : $(date)"
    echo "========================================"
done

echo
echo "========================================"
echo "COMBINE 5 TASKS"
echo "========================================"

"$PY" "$EVAL" \
  --combine \
  2>&1 | tee "$LOGDIR/combine.log"

echo
echo "========================================"
echo "ALL FIVE TASKS + COMBINE COMPLETE"
echo "finish = $(date)"
echo "========================================"
