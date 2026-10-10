#!/usr/bin/env bash
set -euo pipefail

cd /home/soenchun/cse/PointWorld_workspace/PointWorld

export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=1

PY=/home/soenchun/cse/PointWorld_workspace/envs/pointworld-env/bin/python
PILOT=diagnostics/logs/grasp_pose_pilot.log

# Pilot 검증
if ! grep -q '\[RELATIVE_OK\]' "$PILOT" 2>/dev/null; then
    echo "[STOP] Pilot success not confirmed."
    tail -40 "$PILOT" 2>/dev/null || true
    exit 1
fi

if grep -Eq '\[ERROR\]|Traceback' "$PILOT"; then
    echo "[STOP] Pilot contains errors."
    exit 1
fi

# 5개 Task 순차 실행
for TASK in task-0000 task-0001 task-0006 task-0008 task-0010
do
    LOG="diagnostics/logs/grasp_relative_${TASK}.log"

    echo "[$(date)] START $TASK"

    "$PY" -u \
      diagnostics/eval_close_anchor_relative_replay_524.py \
      --task "$TASK" \
      > "$LOG" 2>&1

    if grep -Eq '\[ERROR\]|Traceback' "$LOG"; then
        echo "[FAILED] $TASK - check $LOG"
        exit 1
    fi

    echo "[$(date)] FINISHED $TASK"
done

# 결과 개수 검증
"$PY" - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("diagnostics/results")
old = base / "close_anchor_causal_5task_current10457"
new = base / "close_anchor_relative_replay_current524"

total = 0

for task in (
    "task-0000", "task-0001", "task-0006",
    "task-0008", "task-0010"
):
    a = pd.read_csv(old / task / "per_clip_causal_metrics.csv")
    b = pd.read_csv(new / task / "per_clip_causal_metrics.csv")
    f = pd.read_csv(new / task / "per_frame_relative_metrics.csv")

    expected = set(a["clip"].astype(str))
    obtained = set(b["clip"].astype(str))
    framed = set(f["clip"].astype(str))

    assert len(a) == len(expected)
    assert len(b) == len(obtained)
    assert expected == obtained == framed, task
    assert not f.duplicated(["clip", "frame", "condition"]).any()
    assert (f.groupby(["clip", "frame"])["condition"].nunique() == 4).all()

    print(f"[PASS] {task}: {len(obtained)}/{len(expected)} clips")
    total += len(obtained)

assert total == 524, total
print("[ALL VERIFIED] 524 / 524 clips")
PY

echo "[ALL TASKS COMPLETED]"
