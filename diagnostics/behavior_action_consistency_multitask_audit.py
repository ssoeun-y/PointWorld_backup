import csv
import re
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(
    "/home/viglab/cse/PointWorld_workspace/PointWorld/"
    "diagnostics/data/pointworld_behavior_multitask_restored/"
    "behavior/flows"
)

TASKS = ["0006", "0008", "0010"]

OUT = Path(
    "diagnostics/results/"
    "action_consistency_tasks0006_0008_0010.csv"
)
OUT.parent.mkdir(parents=True, exist_ok=True)

EPS = 1e-12


def metrics(pose):
    xyz = np.asarray(pose, dtype=np.float64)[:, :3]

    # Translational displacement per timestep
    v = np.diff(xyz, axis=0)
    speed = np.linalg.norm(v, axis=1)

    path = float(speed.sum())
    disp = float(np.linalg.norm(xyz[-1] - xyz[0]))

    mean_speed = float(speed.mean())

    # Acceleration / jerk in per-step units
    a = np.diff(v, axis=0)
    accel = np.linalg.norm(a, axis=1)

    j = np.diff(a, axis=0)
    jerk = np.linalg.norm(j, axis=1)

    turns = []
    reversals = []

    for x, y in zip(v[:-1], v[1:]):
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)

        if nx <= EPS or ny <= EPS:
            continue

        c = np.clip(
            np.dot(x, y) / (nx * ny),
            -1.0,
            1.0,
        )

        turns.append(
            np.degrees(np.arccos(c))
        )
        reversals.append(
            float(c < 0.0)
        )

    turns = np.asarray(turns, dtype=float)
    reversals = np.asarray(reversals, dtype=float)

    def p95(x):
        return (
            float(np.percentile(x, 95))
            if len(x) else np.nan
        )

    return {
        "path_length_m": path,
        "net_displacement_m": disp,

        "path_inefficiency":
            (1.0 - disp / path)
            if path > EPS else np.nan,

        "speed_cv":
            float(speed.std() / mean_speed)
            if mean_speed > EPS else np.nan,

        "accel_p95": p95(accel),
        "jerk_p95": p95(jerk),
        "turn_p95_deg": p95(turns),

        "reversal_fraction":
            float(reversals.mean())
            if len(reversals) else np.nan,
    }


rows = []

for task in TASKS:
    task_dir = ROOT / f"task-{task}"
    h5s = sorted(task_dir.glob("episode_*.hdf5"))

    print(
        f"task-{task}: H5 episodes = {len(h5s)}"
    )

    for hp in h5s:
        episode = f"task-{task}_{hp.stem}"

        with h5py.File(hp, "r") as f:
            for clip_key in f.keys():

                if not re.fullmatch(
                    r"\d+:\d+",
                    clip_key
                ):
                    continue

                g = f[clip_key]

                if (
                    "left_gripper_pose" not in g
                    or "right_gripper_pose" not in g
                ):
                    continue

                hm = {
                    "left": metrics(
                        g["left_gripper_pose"][:]
                    ),
                    "right": metrics(
                        g["right_gripper_pose"][:]
                    ),
                }

                dominant = max(
                    hm,
                    key=lambda h:
                    hm[h]["path_length_m"]
                )

                start, end = map(
                    int,
                    clip_key.split(":")
                )

                r = {
                    "task": f"task-{task}",
                    "episode": episode,
                    "clip":
                        f"{episode}-{clip_key}",
                    "start": start,
                    "end": end,
                    "dominant_hand": dominant,
                }

                r.update(hm[dominant])
                rows.append(r)


with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)


KEYS = [
    "path_length_m",
    "speed_cv",
    "accel_p95",
    "jerk_p95",
    "turn_p95_deg",
    "reversal_fraction",
    "path_inefficiency",
]


def summarize(name, rr):
    print(
        f"\n=== {name} N={len(rr)} ==="
    )

    for key in KEYS:
        x = np.asarray(
            [r[key] for r in rr],
            dtype=float
        )
        x = x[np.isfinite(x)]

        print(
            f"{key:22s}"
            f" med={np.median(x):.6g}"
            f" p90={np.percentile(x,90):.6g}"
            f" p95={np.percentile(x,95):.6g}"
            f" p99={np.percentile(x,99):.6g}"
            f" max={np.max(x):.6g}"
        )


print("\n" + "=" * 95)
print("MULTI-TASK ACTION CONSISTENCY AUDIT")
print("=" * 95)
print("Total clips =", len(rows))
print("CSV =", OUT.resolve())


for task in TASKS:
    name = f"task-{task}"

    task_rows = [
        r for r in rows
        if r["task"] == name
    ]

    summarize(
        f"{name} ALL",
        task_rows
    )

    # Episode-start confound excluded
    valid = [
        r for r in task_rows
        if r["start"] > 0
    ]

    # Define high-motion relative to each task's
    # own action distribution.
    q75 = np.percentile(
        [r["path_length_m"] for r in valid],
        75
    )

    high = [
        r for r in valid
        if r["path_length_m"] > q75
    ]

    print(
        f"\n{name} HIGH-MOTION:"
        f" q75={q75:.6g}m"
        f" windows={len(high)}"
        f" episodes="
        f"{len(set(r['episode'] for r in high))}"
    )

    for key in [
        "speed_cv",
        "accel_p95",
        "jerk_p95",
        "turn_p95_deg",
        "reversal_fraction",
        "path_inefficiency",
    ]:

        # One strongest window per episode:
        # overlapping windows cannot dominate.
        best = {}

        for r in high:
            ep = r["episode"]

            if (
                ep not in best
                or r[key] > best[ep][key]
            ):
                best[ep] = r

        vals = np.asarray(
            [r[key] for r in best.values()],
            dtype=float
        )
        vals = vals[np.isfinite(vals)]

        print("\n" + "-" * 90)
        print(name, key)
        print(
            "episode-level:"
            f" median={np.median(vals):.6g}"
            f" p90={np.percentile(vals,90):.6g}"
            f" p95={np.percentile(vals,95):.6g}"
            f" p99={np.percentile(vals,99):.6g}"
            f" max={np.max(vals):.6g}"
        )

        ranked = sorted(
            best.values(),
            key=lambda r: r[key],
            reverse=True
        )

        print("TOP 5 UNIQUE EPISODES")

        for i, r in enumerate(
            ranked[:5], 1
        ):
            print(
                f"{i:2d}. "
                f"{r[key]:.6g} | "
                f"path={r['path_length_m']:.4f}m | "
                f"{r['dominant_hand']:5s} | "
                f"{r['clip']}"
            )


print("\nSUCCESS = True")
