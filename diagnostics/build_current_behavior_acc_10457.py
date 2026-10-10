from pathlib import Path
import numpy as np
import pandas as pd

from arguments import parse_args
from dataset_components.dataloader import build_dataloader
from diagnostics.run_established_action_metrics_5task import (
    established_metrics_for_clip,
)

TASK_DIRS = {
    "task-0000": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
    "task-0001": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0001",
    "task-0006": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0006",
    "task-0008": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0008",
    "task-0010": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0010",
}

EXPECTED = {
    "task-0000": 612,
    "task-0001": 2477,
    "task-0006": 2157,
    "task-0008": 2972,
    "task-0010": 2239,
}

OUT = Path(
    "diagnostics/results/current_behavior_established_acc_10457/"
    "per_sample_current_established_acc_10457.csv"
)


def norm_key(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, bytes):
        x = x.decode()
    return str(x)


rows = []

for task, data_dir in TASK_DIRS.items():
    print("\n" + "=" * 70)
    print("TASK =", task)
    print("=" * 70)

    args = parse_args(skip_command_line=True)

    args.seed = 42
    args.domains = ["behavior"]
    args.data_dirs = [data_dir]
    args.batch_size = 1
    args.num_workers = 0
    args.eval_num_workers = 0

    loader, info = build_dataloader(
        args=args,
        mode="test",
        rank=0,
        world_size=1,
        override_splits="test",
        force_resampled_eval=False,
    )

    seen = set()
    n = 0

    for batch in loader:
        key = norm_key(batch["__key__"])

        if key in seen:
            raise RuntimeError(f"duplicate clip: {key}")

        seen.add(key)

        right = batch["right_gripper_pose"][0].detach().cpu().numpy()
        left = batch["left_gripper_pose"][0].detach().cpu().numpy()

        m = established_metrics_for_clip(
            np.asarray(right, dtype=np.float64),
            np.asarray(left, dtype=np.float64),
        )

        native = float(m["rms_acceleration"])

        rows.append({
            "task": task,
            "clip": key,
            "dominant_hand": m["dominant_hand"],
            "rms_acceleration": native,
            "rms_acceleration_si_m_s2": native * 25.0,
        })

        n += 1

        if n % 250 == 0:
            print(f"{task}: {n}/{EXPECTED[task]}")

        if n >= EXPECTED[task]:
            break

    print(
        f"{task}: SEEN={len(seen)} "
        f"EXPECTED={EXPECTED[task]}"
    )

    if len(seen) != EXPECTED[task]:
        raise RuntimeError(
            f"{task}: expected {EXPECTED[task]}, got {len(seen)}"
        )


df = pd.DataFrame(rows)

assert len(df) == 10457
assert df["clip"].duplicated().sum() == 0
assert df.groupby("task").size().to_dict() == EXPECTED

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("\n" + "=" * 70)
print("CURRENT ACC 10457 READY")
print("ROWS =", len(df))
print(df.groupby("task").size().to_string())
print("SAVED =", OUT)
print("=" * 70)
