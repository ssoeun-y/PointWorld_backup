import csv, re
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(
    "diagnostics/data/pointworld_behavior_task0001_restored/"
    "behavior/flows/task-0001"
)
OUT = Path("diagnostics/results/action_consistency_task0001.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

EPS = 1e-12

def metrics(pose):
    xyz = np.asarray(pose, dtype=np.float64)[:, :3]

    v = np.diff(xyz, axis=0)
    speed = np.linalg.norm(v, axis=1)

    path = float(speed.sum())
    disp = float(np.linalg.norm(xyz[-1] - xyz[0]))

    a = np.diff(v, axis=0)
    acc = np.linalg.norm(a, axis=1)

    j = np.diff(a, axis=0)
    jerk = np.linalg.norm(j, axis=1)

    turns, revs = [], []
    for x, y in zip(v[:-1], v[1:]):
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        if nx <= EPS or ny <= EPS:
            continue
        c = np.clip(np.dot(x,y)/(nx*ny), -1, 1)
        turns.append(np.degrees(np.arccos(c)))
        revs.append(float(c < 0))

    turns = np.asarray(turns)
    revs = np.asarray(revs)

    mean_speed = float(speed.mean())

    def med(x): return float(np.median(x)) if len(x) else np.nan
    def p95(x): return float(np.percentile(x,95)) if len(x) else np.nan

    return dict(
        path_length_m=path,
        net_displacement_m=disp,
        path_inefficiency=(1-disp/path) if path > EPS else np.nan,
        speed_cv=float(speed.std()/mean_speed) if mean_speed > EPS else np.nan,
        accel_p95=p95(acc),
        jerk_p95=p95(jerk),
        turn_p95_deg=p95(turns),
        reversal_fraction=float(revs.mean()) if len(revs) else np.nan,
    )

rows = []

for hp in sorted(ROOT.glob("episode_*.hdf5")):
    episode = "task-0001_" + hp.stem

    with h5py.File(hp, "r") as f:
        for clip in f.keys():
            if not re.fullmatch(r"\d+:\d+", clip):
                continue

            g = f[clip]
            if "left_gripper_pose" not in g or "right_gripper_pose" not in g:
                continue

            hm = {
                "left": metrics(g["left_gripper_pose"][:]),
                "right": metrics(g["right_gripper_pose"][:]),
            }

            dominant = max(
                hm,
                key=lambda h: hm[h]["path_length_m"]
            )

            start, end = map(int, clip.split(":"))

            r = dict(
                episode=episode,
                clip=f"{episode}-{clip}",
                start=start,
                end=end,
                dominant_hand=dominant,
            )
            r.update(hm[dominant])
            rows.append(r)

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print("="*90)
print("TASK-0001 ACTION AUDIT")
print("clips =", len(rows))
print("episodes =", len(set(r["episode"] for r in rows)))
print("CSV =", OUT.resolve())

KEYS = [
    "path_length_m",
    "speed_cv",
    "accel_p95",
    "jerk_p95",
    "turn_p95_deg",
    "reversal_fraction",
    "path_inefficiency",
]

def show(name, rr):
    print("\n===", name, "N =", len(rr), "===")
    for k in KEYS:
        x = np.asarray([r[k] for r in rr], float)
        x = x[np.isfinite(x)]
        print(
            f"{k:22s}"
            f" med={np.median(x):.6g}"
            f" p90={np.percentile(x,90):.6g}"
            f" p95={np.percentile(x,95):.6g}"
            f" p99={np.percentile(x,99):.6g}"
            f" max={np.max(x):.6g}"
        )

show("ALL", rows)
show("START>0", [r for r in rows if r["start"] > 0])

# START>0 only
valid = [r for r in rows if r["start"] > 0]

# High-motion quartile, data-driven
q75 = np.percentile(
    [r["path_length_m"] for r in valid], 75
)
high = [
    r for r in valid
    if r["path_length_m"] > q75
]

print("\nHIGH-MOTION Q4 threshold =", q75)
print("windows =", len(high))
print("unique episodes =", len(set(r["episode"] for r in high)))

for key in [
    "speed_cv",
    "accel_p95",
    "jerk_p95",
    "turn_p95_deg",
    "reversal_fraction",
    "path_inefficiency",
]:
    # strongest window per episode
    best = {}
    for r in high:
        ep = r["episode"]
        if ep not in best or r[key] > best[ep][key]:
            best[ep] = r

    vals = np.asarray([r[key] for r in best.values()], float)
    vals = vals[np.isfinite(vals)]

    print("\n" + "="*90)
    print(key)
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

    print("TOP 10 UNIQUE EPISODES")
    for i,r in enumerate(ranked[:10],1):
        print(
            f"{i:2d}. {r[key]:.6g} | "
            f"path={r['path_length_m']:.4f}m | "
            f"{r['dominant_hand']:5s} | "
            f"{r['clip']}"
        )

print("\nSUCCESS = True")
