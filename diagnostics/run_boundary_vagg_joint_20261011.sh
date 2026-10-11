#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/soenchun/cse/PointWorld_workspace/PointWorld
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=1

PY=/home/soenchun/cse/PointWorld_workspace/envs/pointworld-env/bin/python
R=diagnostics/results/boundary_vagg_joint_20261011
mkdir -p "$R"

echo "STARTING SMOKE TEST"
"$PY" diagnostics/eval_boundary_vagg_joint_20261011.py \
    --smoke > "$R/smoke.log" 2>&1

echo "SMOKE TEST PASSED"
echo "STARTING FULL 5-TASK EVALUATION"

"$PY" diagnostics/eval_boundary_vagg_joint_20261011.py \
    > "$R/full.log" 2>&1

echo "JOINT_EXPERIMENT_COMPLETE"
