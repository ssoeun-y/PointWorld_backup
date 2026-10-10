#!/usr/bin/env bash
set -euo pipefail

cd /home/soenchun/cse/PointWorld_workspace/PointWorld

PY=/home/soenchun/cse/PointWorld_workspace/envs/pointworld-env/bin/python

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/home/soenchun/cse/PointWorld_workspace/PointWorld

EVAL=diagnostics/eval_acc_boundary_adaptive_contact_v6_current10457.py
LOGROOT=diagnostics/logs/adaptive_contact_v6

TASKS=(
  task-0000
  task-0001
  task-0006
  task-0008
  task-0010
)

echo "================================================"
echo "ADAPTIVE CONTACT V6 FULL 5-TASK"
echo "GPU = ${CUDA_VISIBLE_DEVICES}"
echo "START = $(date)"
echo "================================================"

for TASK in "${TASKS[@]}"
do
    echo
    echo "================================================"
    echo "START ${TASK} : $(date)"
    echo "================================================"

    "$PY" "$EVAL" \
      --task "$TASK" \
      2>&1 | tee \
      "${LOGROOT}/${TASK}.log"

    echo "DONE ${TASK} : $(date)"
done

echo
echo "================================================"
echo "COMBINE MPJPE + PA-MPJPE"
echo "================================================"

"$PY" "$EVAL" \
  --combine \
  2>&1 | tee \
  "${LOGROOT}/combine_metrics.log"

echo
echo "================================================"
echo "COMBINE V6 CONTACT / RELEASE DIAGNOSTICS"
echo "================================================"

"$PY" \
  diagnostics/combine_adaptive_contact_v6_diagnostics.py \
  2>&1 | tee \
  "${LOGROOT}/combine_v6_diagnostics.log"

echo
echo "================================================"
echo "ALL DONE : $(date)"
echo "================================================"
