#!/usr/bin/env bash
set -euo pipefail

cd /home/soenchun/cse/PointWorld_workspace/PointWorld

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/soenchun/cse/PointWorld_workspace/PointWorld
export PYTHONUNBUFFERED=1

PY=/home/soenchun/cse/PointWorld_workspace/envs/pointworld-env/bin/python
S=diagnostics/eval_acc_boundary_phase_grasp_current10457.py
LOG=diagnostics/logs/phase_grasp_final

mkdir -p "$LOG"

for TASK in \
    task-0000 \
    task-0001 \
    task-0006 \
    task-0008 \
    task-0010
do
    echo "======================================"
    echo "===== $TASK ====="
    echo "======================================"

    "$PY" "$S" \
        --task "$TASK" \
        --no-resume \
        2>&1 | tee "$LOG/${TASK}.log"
done

echo "======================================"
echo "===== GOAL MPJPE 5-TASK COMBINE ====="
echo "======================================"

"$PY" "$S" --combine \
    2>&1 | tee "$LOG/combine.log"

echo "======================================"
echo "===== PAPER-MAIN PA 5-TASK COMBINE ==="
echo "======================================"

"$PY" - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(
    "diagnostics/results/"
    "acc_boundary_phase_grasp_current10457"
)

tasks = [
    "task-0000",
    "task-0001",
    "task-0006",
    "task-0008",
    "task-0010",
]

dfs = []

for task in tasks:
    f = (
        root
        / task
        / "per_clip_main_pa_metrics.csv"
    )

    if not f.exists():
        raise RuntimeError(
            f"missing: {f}"
        )

    d = pd.read_csv(f)

    print(
        task,
        "clips=",
        len(d),
    )

    dfs.append(d)

df = pd.concat(
    dfs,
    ignore_index=True,
)

if len(df) != 2683:
    raise RuntimeError(
        f"expected 2683 corrected clips, "
        f"got {len(df)}"
    )

if (
    df[
        ["task", "clip"]
    ]
    .drop_duplicates()
    .shape[0]
    != 2683
):
    raise RuntimeError(
        "duplicate/missing clip identity"
    )

out = (
    root
    / "combined_5task_main_pa_metrics.csv"
)

df.to_csv(
    out,
    index=False,
)

o = df[
    "base_pa_mean_mm"
].to_numpy(float)

r = df[
    "base_repeat_pa_mean_mm"
].to_numpy(float)

c = df[
    "corrected_pa_mean_mm"
].to_numpy(float)

good = (
    np.isfinite(o)
    & np.isfinite(c)
    & (o > 0)
)

paired = (
    c[good] / o[good] - 1.0
) * 100.0

print()
print("=" * 80)
print("5-TASK PAPER-MAIN PA-MPJPE vs GT0")
print("=" * 80)

print("clips =", len(df))
print(
    f"Original median  = "
    f"{np.median(o):.6f} mm"
)
print(
    f"Repeat median    = "
    f"{np.median(r):.6f} mm"
)
print(
    f"Corrected median = "
    f"{np.median(c):.6f} mm"
)
print(
    f"Aggregate change = "
    f"{(np.median(c)/np.median(o)-1)*100:+.3f}%"
)
print(
    f"Paired relative median = "
    f"{np.median(paired):+.3f}%"
)
print(
    f"Corrected lower = "
    f"{np.mean(c < o)*100:.2f}%"
)

print()
print("TASKWISE")

for task, d in df.groupby("task"):
    oo = d[
        "base_pa_mean_mm"
    ].to_numpy(float)

    cc = d[
        "corrected_pa_mean_mm"
    ].to_numpy(float)

    print(
        task,
        f"N={len(d)}",
        f"orig={np.median(oo):.6f}",
        f"corr={np.median(cc):.6f}",
        f"change="
        f"{(np.median(cc)/np.median(oo)-1)*100:+.3f}%",
        f"lower={np.mean(cc < oo)*100:.2f}%"
    )

print()
print("[SAVE]", out)
PY

