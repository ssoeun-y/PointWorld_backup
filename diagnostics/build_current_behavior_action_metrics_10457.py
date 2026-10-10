from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

from arguments import parse_args
from dataset_components.dataloader import build_dataloader

from diagnostics.run_established_action_metrics_5task import (
    established_metrics_for_clip,
)

SEED_SPLIT = 20260902

TASK_DIRS = {
    "task-0000":
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
    "task-0001":
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0001",
    "task-0006":
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0006",
    "task-0008":
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0008",
    "task-0010":
        "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0010",
}

EXPECTED = {
    "task-0000": 612,
    "task-0001": 2477,
    "task-0006": 2157,
    "task-0008": 2972,
    "task-0010": 2239,
}

EXPECTED_VAL = {
    "task-0000": 184,
    "task-0001": 743,
    "task-0006": 647,
    "task-0008": 892,
    "task-0010": 672,
}

OUTDIR = Path(
    "diagnostics/results/current_behavior_established_metrics_10457"
)

OUT_ALL = OUTDIR / "per_sample_established_metrics_10457.csv"
OUT_VAL = OUTDIR / "validation_split_3138.csv"

OLD_ACC = Path(
    "diagnostics/results/current_behavior_established_acc_10457/"
    "per_sample_current_established_acc_10457.csv"
)


def stable_number(text):
    h = hashlib.sha256(
        f"{SEED_SPLIT}|{text}".encode()
    ).digest()

    return int.from_bytes(
        h[:8], "big"
    ) / 2**64


def exact_split(df):
    result = pd.Series(
        index=df.index,
        dtype="object",
    )

    for task, g in df.groupby("task"):
        idx = sorted(
            g.index,
            key=lambda i:
                stable_number(
                    str(df.at[i, "clip"])
                ),
        )

        n_cal = round(
            len(idx) * 0.70
        )

        result.loc[
            idx[:n_cal]
        ] = "calibration"

        result.loc[
            idx[n_cal:]
        ] = "validation"

    return result


def norm_key(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    if isinstance(x, np.ndarray):
        if x.size == 1:
            x = x.reshape(-1)[0]

    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            x = x[0]

    if isinstance(x, bytes):
        x = x.decode("utf-8")

    return str(x)


rows = []

for task, root in TASK_DIRS.items():

    args = parse_args(
        skip_command_line=True
    )

    args.seed = 42
    args.domains = ["behavior"]
    args.data_dirs = [root]
    args.batch_size = 1
    args.num_workers = 0

    if hasattr(args, "eval_num_workers"):
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

    print()
    print("=" * 70)
    print(task, root)
    print("=" * 70)

    for batch in loader:

        key = norm_key(
            batch["__key__"]
        )

        if key in seen:
            raise RuntimeError(
                f"duplicate key: {key}"
            )

        seen.add(key)

        right = (
            batch[
                "right_gripper_pose"
            ][0]
            .detach()
            .cpu()
            .numpy()
        )

        left = (
            batch[
                "left_gripper_pose"
            ][0]
            .detach()
            .cpu()
            .numpy()
        )

        m = established_metrics_for_clip(
            np.asarray(
                right,
                dtype=np.float64,
            ),
            np.asarray(
                left,
                dtype=np.float64,
            ),
        )

        rows.append({
            "task":
                task,

            "clip":
                key,

            "dominant_hand":
                m["dominant_hand"],

            "rms_acceleration":
                float(
                    m[
                        "rms_acceleration"
                    ]
                ),

            "mean_squared_jerk":
                float(
                    m[
                        "mean_squared_jerk"
                    ]
                ),

            "rms_acceleration_si_m_s2":
                float(
                    m[
                        "rms_acceleration"
                    ]
                ) * 25.0,

            "mean_squared_jerk_si_m2_s6":
                float(
                    m[
                        "mean_squared_jerk"
                    ]
                ) * 15625.0,
        })

        if len(seen) % 500 == 0:
            print(
                task,
                len(seen),
                "/",
                EXPECTED[task],
            )

        if len(seen) >= EXPECTED[task]:
            break

    print(
        task,
        "SEEN =",
        len(seen),
        "EXPECTED =",
        EXPECTED[task],
    )

    if len(seen) != EXPECTED[task]:
        raise RuntimeError(
            f"{task}: "
            f"{len(seen)} != "
            f"{EXPECTED[task]}"
        )


df = pd.DataFrame(rows)

if len(df) != 10457:
    raise RuntimeError(
        f"TOTAL {len(df)} != 10457"
    )

if df["clip"].duplicated().any():
    raise RuntimeError(
        "duplicate clip in final table"
    )

if (
    df.groupby("task")
    .size()
    .to_dict()
    != EXPECTED
):
    raise RuntimeError(
        "task count mismatch"
    )


# ------------------------------------------------------------
# SAME exact 70/30 split as established code
# ------------------------------------------------------------

df["split"] = exact_split(df)

val = df[
    df["split"] == "validation"
].copy()

val_counts = (
    val.groupby("task")
    .size()
    .to_dict()
)

if val_counts != EXPECTED_VAL:
    raise RuntimeError(
        f"validation mismatch: "
        f"{val_counts} != "
        f"{EXPECTED_VAL}"
    )

if len(val) != 3138:
    raise RuntimeError(
        f"validation total "
        f"{len(val)} != 3138"
    )


# ------------------------------------------------------------
# Cross-check RMS Acc against the current 10457 table
# already generated earlier.
# ------------------------------------------------------------

if not OLD_ACC.exists():
    raise FileNotFoundError(
        OLD_ACC
    )

old = pd.read_csv(
    OLD_ACC
)

chk = df.merge(
    old[
        [
            "clip",
            "rms_acceleration",
            "rms_acceleration_si_m_s2",
        ]
    ],
    on="clip",
    suffixes=("_new", "_old"),
)

if len(chk) != 10457:
    raise RuntimeError(
        f"Acc audit overlap "
        f"{len(chk)} != 10457"
    )

native_diff = float(
    np.max(
        np.abs(
            chk[
                "rms_acceleration_new"
            ].to_numpy(float)
            -
            chk[
                "rms_acceleration_old"
            ].to_numpy(float)
        )
    )
)

si_diff = float(
    np.max(
        np.abs(
            chk[
                "rms_acceleration_si_m_s2_new"
            ].to_numpy(float)
            -
            chk[
                "rms_acceleration_si_m_s2_old"
            ].to_numpy(float)
        )
    )
)

if native_diff > 1e-12:
    raise RuntimeError(
        f"native Acc mismatch "
        f"{native_diff}"
    )

if si_diff > 1e-12:
    raise RuntimeError(
        f"SI Acc mismatch "
        f"{si_diff}"
    )


OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUT_ALL,
    index=False,
)

val.to_csv(
    OUT_VAL,
    index=False,
)


print()
print("=" * 78)
print("CURRENT 5-TASK ACTION METRICS READY")
print("=" * 78)

print("ALL =", len(df))
print(
    df.groupby("task")
    .size()
    .to_string()
)

print()
print("VALIDATION =", len(val))
print(
    val.groupby("task")
    .size()
    .to_string()
)

print()
print(
    "MAX ACC NATIVE DIFF =",
    f"{native_diff:.3e}",
)

print(
    "MAX ACC SI DIFF =",
    f"{si_diff:.3e}",
)

print()
print("ALL CSV =", OUT_ALL)
print("VAL CSV =", OUT_VAL)
print("=" * 78)
