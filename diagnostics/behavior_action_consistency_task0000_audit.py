import csv
import io
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(
    "/home/viglab/cse/PointWorld_workspace/datasets/behavior/wds_task0000"
)
OUT = Path(
    "diagnostics/results/action_consistency_task0000.csv"
)
OUT.parent.mkdir(parents=True, exist_ok=True)

EPS = 1e-12


def metrics(pose):
    xyz = np.asarray(pose, dtype=np.float64)[:, :3]

    # Per-step translational velocity in m / step.
    v = np.diff(xyz, axis=0)
    speed = np.linalg.norm(v, axis=1)

    path_length = float(speed.sum())
    net_disp = float(np.linalg.norm(xyz[-1] - xyz[0]))

    mean_speed = float(speed.mean())
    median_speed = float(np.median(speed))
    speed_std = float(speed.std())
    speed_cv = (
        speed_std / mean_speed
        if mean_speed > EPS else np.nan
    )

    # Vector acceleration / jerk.
    a = np.diff(v, axis=0)
    accel = np.linalg.norm(a, axis=1)

    j = np.diff(a, axis=0)
    jerk = np.linalg.norm(j, axis=1)

    # Direction change between consecutive velocity vectors.
    turn_angles = []
    reversal = []

    for x, y in zip(v[:-1], v[1:]):
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)

        if nx <= EPS or ny <= EPS:
            continue

        cos = np.clip(
            np.dot(x, y) / (nx * ny),
            -1.0,
            1.0,
        )
        angle = np.degrees(np.arccos(cos))
        turn_angles.append(angle)

        # Geometrically natural reversal criterion:
        # consecutive velocity vectors > 90 degrees apart.
        reversal.append(float(cos < 0.0))

    turn_angles = np.asarray(turn_angles, dtype=np.float64)
    reversal = np.asarray(reversal, dtype=np.float64)

    def stat(x, fn):
        if len(x) == 0:
            return np.nan
        return float(fn(x))

    return {
        "path_length_m": path_length,
        "net_displacement_m": net_disp,
        "path_efficiency": (
            net_disp / path_length
            if path_length > EPS else np.nan
        ),
        "path_inefficiency": (
            1.0 - net_disp / path_length
            if path_length > EPS else np.nan
        ),

        "speed_mean": mean_speed,
        "speed_median": median_speed,
        "speed_cv": speed_cv,
        "speed_max": float(speed.max()),

        "accel_median": stat(accel, np.median),
        "accel_p95": stat(
            accel, lambda x: np.percentile(x, 95)
        ),
        "accel_max": stat(accel, np.max),

        "jerk_median": stat(jerk, np.median),
        "jerk_p95": stat(
            jerk, lambda x: np.percentile(x, 95)
        ),
        "jerk_max": stat(jerk, np.max),

        "turn_median_deg": stat(turn_angles, np.median),
        "turn_p95_deg": stat(
            turn_angles, lambda x: np.percentile(x, 95)
        ),
        "turn_max_deg": stat(turn_angles, np.max),

        "reversal_fraction": stat(reversal, np.mean),
    }


rows = []

for split in ["train", "test"]:
    tar_files = sorted((ROOT / split).glob("*.tar"))

    for tar_path in tar_files:
        with tarfile.open(tar_path) as tf:
            members = {
                m.name: m
                for m in tf.getmembers()
                if m.isfile()
            }

            suffix = ".left_gripper_pose.npy"
            clips = sorted(
                name[:-len(suffix)]
                for name in members
                if name.endswith(suffix)
            )

            for clip in clips:
                poses = {}

                for hand in ["left", "right"]:
                    name = f"{clip}.{hand}_gripper_pose.npy"
                    if name not in members:
                        continue

                    f = tf.extractfile(members[name])
                    poses[hand] = np.load(
                        io.BytesIO(f.read())
                    )

                if not poses:
                    continue

                hand_metrics = {
                    h: metrics(p)
                    for h, p in poses.items()
                }

                # Avoid arbitrary movement threshold:
                # use whichever hand has the larger path length.
                dominant = max(
                    hand_metrics,
                    key=lambda h:
                    hand_metrics[h]["path_length_m"]
                )

                row = {
                    "split": split,
                    "clip": clip,
                    "dominant_hand": dominant,
                    "left_path_length_m":
                        hand_metrics.get(
                            "left", {}
                        ).get("path_length_m", np.nan),
                    "right_path_length_m":
                        hand_metrics.get(
                            "right", {}
                        ).get("path_length_m", np.nan),
                }

                row.update(hand_metrics[dominant])
                rows.append(row)


fieldnames = list(rows[0].keys())

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print("=" * 80)
print("ACTION CONSISTENCY AUDIT")
print("=" * 80)
print("N clips =", len(rows))
print("output =", OUT.resolve())

metrics_to_show = [
    "path_length_m",
    "path_inefficiency",
    "speed_cv",
    "accel_p95",
    "jerk_p95",
    "turn_p95_deg",
    "reversal_fraction",
]

print("\n=== DISTRIBUTIONS ===")
for key in metrics_to_show:
    x = np.array(
        [r[key] for r in rows],
        dtype=np.float64,
    )
    x = x[np.isfinite(x)]

    print(
        f"{key:22s}"
        f" median={np.median(x):.6g}"
        f" p90={np.percentile(x,90):.6g}"
        f" p95={np.percentile(x,95):.6g}"
        f" p99={np.percentile(x,99):.6g}"
        f" max={np.max(x):.6g}"
    )

rank_metrics = [
    "speed_cv",
    "accel_p95",
    "jerk_p95",
    "turn_p95_deg",
    "reversal_fraction",
    "path_inefficiency",
]

for key in rank_metrics:
    valid = [
        r for r in rows
        if np.isfinite(r[key])
    ]
    valid.sort(
        key=lambda r: r[key],
        reverse=True,
    )

    print(f"\n=== TOP 10: {key} ===")
    for i, r in enumerate(valid[:10], 1):
        print(
            f"{i:2d}. {r[key]:.6g} | "
            f"{r['split']:5s} | "
            f"{r['dominant_hand']:5s} | "
            f"{r['clip']}"
        )

print("\nSUCCESS = True")
