import csv
import re
from pathlib import Path
import numpy as np

RESULTS = Path("diagnostics/results")

FILES = [
    RESULTS / "action_consistency_task0000.csv",
    RESULTS / "action_consistency_task0001.csv",
    RESULTS / "action_consistency_tasks0006_0008_0010.csv",
]

METRICS = [
    "speed_cv",
    "accel_p95",
    "jerk_p95",
    "turn_p95_deg",
    "reversal_fraction",
    "path_inefficiency",
]

TASKS = [
    "task-0000",
    "task-0001",
    "task-0006",
    "task-0008",
    "task-0010",
]

rows = []

for p in FILES:
    if not p.exists():
        raise FileNotFoundError(p)

    with p.open() as f:
        for r in csv.DictReader(f):

            clip = r["clip"]

            m = re.search(
                r"(task-\d+)_episode_(\d+)-(\d+):(\d+)$",
                clip
            )
            if not m:
                raise RuntimeError(f"Cannot parse: {clip}")

            task = m.group(1)
            episode = f"{task}_episode_{m.group(2)}"
            start = int(m.group(3))

            out = {
                "task": task,
                "episode": episode,
                "clip": clip,
                "start": start,
                "dominant_hand": r["dominant_hand"],
                "path_length_m": float(r["path_length_m"]),
            }

            for k in METRICS:
                out[k] = float(r[k])

            rows.append(out)

print("=" * 110)
print("5-TASK CROSS-TASK ACTION CONSISTENCY SUMMARY")
print("=" * 110)
print("total windows =", len(rows))

summary_rows = []

for task in TASKS:

    # episode-start artifact 제거
    tr = [
        r for r in rows
        if r["task"] == task and r["start"] > 0
    ]

    if not tr:
        print("NO ROWS:", task)
        continue

    # task별 motion distribution 기준 Q4
    q75 = float(np.percentile(
        [r["path_length_m"] for r in tr],
        75
    ))

    high = [
        r for r in tr
        if r["path_length_m"] > q75
    ]

    n_eps = len(set(r["episode"] for r in high))

    print("\n" + "=" * 110)
    print(
        f"{task} | START>0={len(tr)} | "
        f"Q4 path>{q75:.4f}m | "
        f"high-motion windows={len(high)} | "
        f"unique episodes={n_eps}"
    )

    for metric in METRICS:

        # overlapping windows가 결과를 지배하지 않도록
        # episode별 strongest window 1개
        best = {}

        for r in high:
            ep = r["episode"]

            if (
                ep not in best
                or r[metric] > best[ep][metric]
            ):
                best[ep] = r

        vals = np.asarray(
            [r[metric] for r in best.values()],
            dtype=float
        )
        vals = vals[np.isfinite(vals)]

        med = float(np.median(vals))
        p90 = float(np.percentile(vals, 90))
        p95 = float(np.percentile(vals, 95))
        p99 = float(np.percentile(vals, 99))
        mx = float(np.max(vals))

        print(
            f"{metric:21s} "
            f"med={med:9.5g} "
            f"p90={p90:9.5g} "
            f"p95={p95:9.5g} "
            f"p99={p99:9.5g} "
            f"max={mx:9.5g}"
        )

        summary_rows.append({
            "task": task,
            "q75_path_m": q75,
            "high_motion_unique_episodes": n_eps,
            "metric": metric,
            "median": med,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "max": mx,
        })

        ranked = sorted(
            best.values(),
            key=lambda r: r[metric],
            reverse=True
        )

        print("   TOP3:")
        for i, r in enumerate(ranked[:3], 1):
            print(
                f"     {i}. {r[metric]:.6g} | "
                f"path={r['path_length_m']:.4f}m | "
                f"{r['clip']}"
            )

# Compact cross-task P95 table
print("\n" + "=" * 110)
print("CROSS-TASK P95 TABLE")
print("=" * 110)

header = "metric".ljust(22)
for task in TASKS:
    header += task.replace("task-", "").rjust(12)
print(header)

for metric in METRICS:
    line = metric.ljust(22)

    for task in TASKS:
        found = [
            x for x in summary_rows
            if x["task"] == task and x["metric"] == metric
        ]

        if found:
            line += f"{found[0]['p95']:12.5g}"
        else:
            line += "NA".rjust(12)

    print(line)

OUT = RESULTS / "action_consistency_5task_episode_summary.csv"

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=list(summary_rows[0].keys())
    )
    w.writeheader()
    w.writerows(summary_rows)

print("\nSaved:", OUT.resolve())
print("SUCCESS = True")
