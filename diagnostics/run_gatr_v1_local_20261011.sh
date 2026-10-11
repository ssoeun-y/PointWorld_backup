#!/usr/bin/env bash
set -euo pipefail

cd /home/soenchun/cse/PointWorld_workspace/PointWorld

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

PY=/home/soenchun/cse/PointWorld_workspace/envs/pointworld-env/bin/python
SCRIPT=diagnostics/eval_gatr_v1_local_20261011.py
OUT=diagnostics/results/gatr_v1_local_20261011

mkdir -p "$OUT"

echo "GATR V1 SMOKE START"

"$PY" "$SCRIPT" --task task-0000 --limit 1 \
  > "$OUT/smoke.log" 2>&1 || {
    echo "SMOKE FAILED"
    tail -30 "$OUT/smoke.log"
    exit 1
  }

echo "GATR V1 SMOKE PASS"

for T in 0000 0001 0006 0008 0010; do
    echo "START GATR V1 task-$T"

    "$PY" "$SCRIPT" \
      --task "task-$T" \
      --limit 80 \
      > "$OUT/task_${T}.log" 2>&1 || {
        echo "FAILED task-$T"
        tail -30 "$OUT/task_${T}.log"
        exit 1
      }

    echo "FINISHED GATR V1 task-$T"

    # Abort if the first task produces no
    # meaningful local-action candidates.
    if [ "$T" = "0000" ]; then
      COUNT=$("$PY" - <<'PY'
import pandas as pd
p = (
 "diagnostics/results/"
 "acc_gatr_v1_local_eval_20261011/"
 "task-0000/mechanism_action_audit.csv"
)
d = pd.read_csv(p)
print(int(d.gatr_applied.astype(bool).sum()))
PY
)
      echo "TASK0000 LOCAL APPLIED=$COUNT"
      if [ "$COUNT" -lt 3 ]; then
          echo "EARLY STOP: insufficient feasible local retiming"
          exit 0
      fi
    fi
done

echo "GATR V1 ALL FIVE TASKS COMPLETE"
