import csv, tarfile, re
from pathlib import Path
import numpy as np

CSV = Path(
    "diagnostics/results/"
    "action_consistency_tasks0006_0008_0010.csv"
)

WDS = Path(
    "diagnostics/data/"
    "pointworld_behavior_multitask_restored/"
    "behavior/wds"
)

OUT = Path(
    "diagnostics/results/"
    "real_action_state_link_candidates.csv"
)

PRIMARY = [
    "speed_cv",
    "jerk_p95",
    "path_inefficiency",
]

# ---------------------------------------------------------
# 1. Official WDS TEST sample universe
# ---------------------------------------------------------
test_keys = set()

for tp in sorted((WDS / "test").glob("*.tar")):
    with tarfile.open(tp) as tf:
        for m in tf.getmembers():
            if m.isfile() and "." in m.name:
                test_keys.add(m.name.split(".", 1)[0])

print("WDS test clips =", len(test_keys))

# ---------------------------------------------------------
# 2. Action audit rows
# ---------------------------------------------------------
rows = []

with CSV.open() as f:
    for r in csv.DictReader(f):

        if r["clip"] not in test_keys:
            continue

        m = re.search(
            r"(task-\d+)_episode_(\d+)-(\d+):(\d+)$",
            r["clip"]
        )

        if not m:
            continue

        r["episode"] = (
            f"{m.group(1)}_episode_{m.group(2)}"
        )
        r["start"] = int(m.group(3))

        for k in [
            "path_length_m",
            "net_displacement_m",
            "speed_cv",
            "accel_p95",
            "jerk_p95",
            "turn_p95_deg",
            "reversal_fraction",
            "path_inefficiency",
        ]:
            r[k] = float(r[k])

        rows.append(r)

print("matched action rows =", len(rows))

# episode-start artifact 제외
rows = [r for r in rows if r["start"] > 0]

selected = []

for task in ["task-0006", "task-0008", "task-0010"]:

    tr = [r for r in rows if r["task"] == task]

    # Same high-motion criterion as previous analysis,
    # now calculated within the official test population.
    q75_motion = np.percentile(
        [r["path_length_m"] for r in tr], 75
    )

    high = [
        r for r in tr
        if r["path_length_m"] > q75_motion
    ]

    print("\n" + "="*90)
    print(
        task,
        "test START>0 =", len(tr),
        "Q4 threshold =", q75_motion,
        "high-motion =", len(high),
        "episodes =", len(set(r["episode"] for r in high))
    )

    for metric in PRIMARY:

        vals = np.array(
            [r[metric] for r in high], float
        )

        p40, p60, p90 = np.percentile(
            vals, [40, 60, 90]
        )

        # Extreme = top 10%
        extreme = [
            r for r in high
            if r[metric] >= p90
        ]

        # Normal comparison pool = central 40–60%
        normal = [
            r for r in high
            if p40 <= r[metric] <= p60
        ]

        # one strongest extreme per episode
        ebest = {}
        for r in extreme:
            ep = r["episode"]
            if ep not in ebest or r[metric] > ebest[ep][metric]:
                ebest[ep] = r

        extreme = sorted(
            ebest.values(),
            key=lambda r: r[metric],
            reverse=True
        )

        print(
            f"\n{metric}: "
            f"p40={p40:.6g} "
            f"p60={p60:.6g} "
            f"p90={p90:.6g}"
        )
        print(
            "extreme episodes =", len(extreme),
            "| normal windows =", len(normal)
        )

        # Top 5 extreme, with nearest path-length normal control
        used_controls = set()

        for er in extreme[:5]:

            candidates = [
                nr for nr in normal
                if nr["episode"] != er["episode"]
                and nr["episode"] not in used_controls
                and nr["dominant_hand"] == er["dominant_hand"]
            ]

            if not candidates:
                continue

            # Match movement magnitude only.
            # We do not use any PointWorld prediction outcome here.
            ctrl = min(
                candidates,
                key=lambda nr:
                    abs(
                        nr["path_length_m"]
                        - er["path_length_m"]
                    )
            )

            used_controls.add(ctrl["episode"])

            for label, rr in [
                ("extreme", er),
                ("control", ctrl),
            ]:
                selected.append({
                    "task": task,
                    "metric": metric,
                    "group": label,
                    "clip": rr["clip"],
                    "episode": rr["episode"],
                    "dominant_hand": rr["dominant_hand"],
                    "path_length_m": rr["path_length_m"],
                    "metric_value": rr[metric],
                })

            print(
                "  EXT",
                er["clip"],
                f"{metric}={er[metric]:.6g}",
                f"path={er['path_length_m']:.4f}m"
            )
            print(
                "  CTL",
                ctrl["clip"],
                f"{metric}={ctrl[metric]:.6g}",
                f"path={ctrl['path_length_m']:.4f}m"
            )

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=list(selected[0].keys())
    )
    w.writeheader()
    w.writerows(selected)

print("\nselected rows =", len(selected))
print("saved =", OUT.resolve())
print("SUCCESS = True")
