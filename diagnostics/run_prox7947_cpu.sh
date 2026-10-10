#!/bin/bash
set -euo pipefail

cd /home/soenchun/cse/PointWorld_workspace/PointWorld

export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=/home/soenchun/cse/PointWorld_workspace/PointWorld
export PYTHONUNBUFFERED=1
export PW_FULL_PROX_TASKS=task-0000,task-0001,task-0006,task-0008,task-0010

mkdir -p \
diagnostics/results/scene_action_proximity_full7947 \
diagnostics/logs/scene_action_proximity_full7947

/home/soenchun/cse/PointWorld_workspace/envs/pointworld-env/bin/python \
diagnostics/eval_scene_action_proximity_full7947.py \
2>&1 | tee \
diagnostics/logs/scene_action_proximity_full7947/CPU_ALL5.log

touch diagnostics/results/scene_action_proximity_full7947/CPU_ALL5_DONE
