#!/usr/bin/env bash
set -euo pipefail

V="$1"
GPU="$2"

cd /home/soenchun/cse/PointWorld_workspace/PointWorld

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1

PY=/home/soenchun/cse/PointWorld_workspace/envs/pointworld-env/bin/python

TASKS=(
    task-0000 task-0001 task-0006
    task-0008 task-0010
)

# Reduce simultaneous reading of the same task shards.
if [ "$V" = "2" ]; then
    TASKS=(
        task-0010 task-0008 task-0006
        task-0001 task-0000
    )
fi

for TASK in "${TASKS[@]}"; do
    echo "===== START v${V} ${TASK} ====="

    "$PY" \
      "diagnostics/iff_teacher_full_acc_v${V}_20261010.py" \
      --task "$TASK"

    # Confirm completed targets exactly match strict cohort.
    "$PY" - "$V" "$TASK" <<'PY'
import csv
import sys
from pathlib import Path

v, task = sys.argv[1:]
r = Path("diagnostics/results")

with (r/"cvpr_boundary_strict_20261010/boundary_valid_2677.csv").open() as f:
    expected = {
        row["clip"] for row in csv.DictReader(f)
        if row["task"] == task
    }

p = r/f"iff_teacher_full_acc_v{v}_20261010"/task/"per_clip_metrics.csv"

with p.open() as f:
    rows = list(csv.DictReader(f))

actual = [x["clip"] for x in rows]

assert len(actual) == len(set(actual)), "Duplicate clip"
assert set(actual) == expected, (
    task, len(actual), len(expected),
    len(expected-set(actual)), len(set(actual)-expected)
)

print(f"STRICT OUTPUT VERIFIED: {task} {len(actual)} clips", flush=True)
PY

    echo "===== FINISHED v${V} ${TASK} ====="
done

echo "IFF_FULL_STRICT_V${V}_COMPLETE"
