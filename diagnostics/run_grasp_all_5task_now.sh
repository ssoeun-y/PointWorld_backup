#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/soenchun/cse/PointWorld_workspace/PointWorld

export PYTHONPATH="$PWD"

PY=/home/soenchun/cse/PointWorld_workspace/envs/pointworld-env/bin/python
SCRIPT=diagnostics/eval_close_anchor_relative_replay_524.py
OUT=diagnostics/results/close_anchor_relative_replay_current524
OLD=diagnostics/results/close_anchor_causal_5task_current10457

echo "[START] $(date)"

# 코드 기본 검사
"$PY" -m py_compile "$SCRIPT"
grep -q 'def _pose_local_points' "$SCRIPT" || {
    echo "[STOP] Pose-based correction missing"
    exit 1
}

# 이전 실패 실행 결과는 삭제하지 않고 별도 보존
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"

for task in task-0000 task-0001 task-0006 task-0008 task-0010; do
    if [ -d "$OUT/$task" ]; then
        mkdir -p "$OUT/_previous/$STAMP"
        mv "$OUT/$task" "$OUT/_previous/$STAMP/"
    fi
done

run_task() {
    local gpu="$1"
    local task="$2"
    local log="diagnostics/logs/grasp_full_${task}.log"

    echo "[START] $(date) $task GPU$gpu"

    if ! CUDA_VISIBLE_DEVICES="$gpu" \
        "$PY" -u "$SCRIPT" --task "$task" > "$log" 2>&1; then
        echo "[FAILED] $task process error: $log"
        return 1
    fi

    if grep -Eq '\[ERROR\]|Traceback|unexpected_exception' "$log"; then
        echo "[FAILED] $task inference error: $log"
        return 1
    fi

    # 실제 처리된 clip이 기존 524개 데이터와 일치하는지 검증
    if ! "$PY" - "$task" <<'PY'
import sys
import pandas as pd
from pathlib import Path

task = sys.argv[1]

old = Path(
    "diagnostics/results/close_anchor_causal_5task_current10457"
) / task

new = Path(
    "diagnostics/results/close_anchor_relative_replay_current524"
) / task

a = pd.read_csv(old / "per_clip_causal_metrics.csv")
b = pd.read_csv(new / "per_clip_causal_metrics.csv")
f = pd.read_csv(new / "per_frame_relative_metrics.csv")

expected = set(a["clip"].astype(str))
completed = set(b["clip"].astype(str))
framed = set(f["clip"].astype(str))

assert len(a) == len(expected)
assert len(b) == len(completed)
assert expected == completed == framed, task

assert not f.duplicated(
    ["clip", "frame", "condition"]
).any()

assert (
    f.groupby(["clip", "frame"])["condition"].nunique() == 4
).all()

assert (
    f.groupby(["clip", "condition"])["frame"].nunique() >= 2
).all()

print(f"[VERIFIED] {task}: {len(completed)} clips")
PY
    then
        echo "[FAILED] $task verification error"
        return 1
    fi

    echo "[DONE] $(date) $task"
}

worker() {
    local gpu="$1"
    shift

    for task in "$@"; do
        run_task "$gpu" "$task" || return 1
    done
}

# GPU1: 3 Tasks
worker 1 task-0000 task-0001 task-0006 &
PID1=$!

# GPU2: 2 Tasks
worker 2 task-0008 task-0010 &
PID2=$!

set +e
wait "$PID1"
RC1=$?
wait "$PID2"
RC2=$?
set -e

if [ "$RC1" -ne 0 ] || [ "$RC2" -ne 0 ]; then
    echo "[FAILED] At least one task failed. Review logs."
    exit 1
fi

"$PY" - <<'PY'
from pathlib import Path
import pandas as pd

root = Path(
    "diagnostics/results/close_anchor_relative_replay_current524"
)

counts = {}

for t in ("0000", "0001", "0006", "0008", "0010"):
    df = pd.read_csv(
        root / f"task-{t}" / "per_clip_causal_metrics.csv"
    )
    counts[t] = len(df)

assert counts == {
    "0000": 57,
    "0001": 144,
    "0006": 62,
    "0008": 111,
    "0010": 150,
}, counts

print("[ALL VERIFIED]", counts)
print("[TOTAL]", sum(counts.values()), "/ 524")
PY

echo "[ALL 5 TASKS COMPLETED] $(date)"
