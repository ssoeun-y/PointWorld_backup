from pathlib import Path
import numpy as np
import pandas as pd

BOUNDARY = 0.24580418
MIN_POINTS = 400
TASKS = ["task-0000", "task-0001", "task-0006", "task-0008", "task-0010"]

ROOT = Path("diagnostics/results/object_pa_mpjpe_acc_5task_current10457")
OUTDIR = Path("diagnostics/results/problem_failure_selection")
OUTDIR.mkdir(parents=True, exist_ok=True)

REQ = [
    "task",
    "clip",
    "object_id",
    "object_short_name",
    "frame",
    "valid_point_count",
    "rms_acceleration_si_m_s2",
    "pa_mpjpe_mm",
]

dfs = []
for task in TASKS:
    p = ROOT / task / "per_frame_metrics.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    miss = [c for c in REQ if c not in df.columns]
    if miss:
        raise RuntimeError(f"{p} missing columns: {miss}")
    dfs.append(df[REQ].copy())

raw = pd.concat(dfs, ignore_index=True)

# ------------------------------------------------------------
# 1) object-track level aggregation
#    one row per (task, clip, object_id, frame)
# ------------------------------------------------------------
objf = (
    raw.groupby(
        ["task", "clip", "object_id", "object_short_name", "frame"],
        as_index=False
    )
    .agg(
        pa_mpjpe_mm=("pa_mpjpe_mm", "mean"),
        valid_point_count=("valid_point_count", "mean"),
        rms_acceleration_si_m_s2=("rms_acceleration_si_m_s2", "mean"),
    )
)

# keep only full 10-frame tracks
count_df = (
    objf.groupby(["task", "clip", "object_id", "object_short_name"], as_index=False)
    .agg(
        n_frames=("frame", "nunique"),
        frames=("frame", lambda x: tuple(sorted(x.tolist()))),
    )
)

objf = objf.merge(
    count_df,
    on=["task", "clip", "object_id", "object_short_name"],
    how="left",
)

objf = objf[objf["n_frames"] == 10].copy()

# ------------------------------------------------------------
# 2) track summary
# ------------------------------------------------------------
def _fit_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])

def _monotonic_fraction(y):
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return np.nan
    d = np.diff(y)
    return float(np.mean(d > 0))

summ_rows = []
for keys, g in objf.groupby(["task", "clip", "object_id", "object_short_name"]):
    task, clip, oid, oname = keys
    g = g.sort_values("frame").reset_index(drop=True)

    frames = g["frame"].to_numpy()
    pa = g["pa_mpjpe_mm"].to_numpy()
    pts = g["valid_point_count"].to_numpy()
    acc = g["rms_acceleration_si_m_s2"].to_numpy()

    if tuple(frames.tolist()) != tuple(range(1, 11)):
        continue

    pa_start = float(pa[0])
    pa_mid = float(pa[4])   # frame 5
    pa_end = float(pa[-1])

    delta = pa_end - pa_start
    slope = _fit_slope(frames, pa)
    mono = _monotonic_fraction(pa)

    summ_rows.append({
        "task": task,
        "clip": clip,
        "object_id": int(oid),
        "object_short_name": oname,
        "n_frames": 10,
        "points_median": float(np.median(pts)),
        "points_mean": float(np.mean(pts)),
        "acc_median": float(np.median(acc)),
        "acc_mean": float(np.mean(acc)),
        "pa_t1": pa_start,
        "pa_t5": pa_mid,
        "pa_t10": pa_end,
        "pa_delta_1_10": float(delta),
        "pa_ratio_10_over_1": float(pa_end / max(pa_start, 1e-12)),
        "pa_slope": float(slope),
        "pa_median": float(np.median(pa)),
        "pa_mean": float(np.mean(pa)),
        "monotonic_fraction": float(mono),
    })

tracks = pd.DataFrame(summ_rows)
if len(tracks) == 0:
    raise RuntimeError("No full 10-frame object tracks found.")

# ------------------------------------------------------------
# 3) high-dynamics population only
# ------------------------------------------------------------
high = tracks[
    (tracks["acc_median"] > BOUNDARY) &
    (tracks["points_median"] >= MIN_POINTS) &
    (tracks["pa_delta_1_10"] > 0) &
    (tracks["pa_t10"] > tracks["pa_t1"]) &
    (tracks["monotonic_fraction"] >= 0.55)
].copy()

if len(high) == 0:
    raise RuntimeError("No high-dynamics candidate satisfies the filter.")

# ------------------------------------------------------------
# 4) representative target = near median degradation in the
#    HIGH-DYNAMICS population, not extreme best/worst
# ------------------------------------------------------------
target = {
    "acc_median": float(high["acc_median"].median()),
    "pa_delta_1_10": float(high["pa_delta_1_10"].median()),
    "pa_slope": float(high["pa_slope"].median()),
    "pa_t10": float(high["pa_t10"].median()),
    "points_median": float(high["points_median"].median()),
}

def robust_scale(s):
    x = np.asarray(s, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return max(1.4826 * mad, 1e-8)

scales = {
    k: robust_scale(high[k].values)
    for k in target.keys()
}

# score = closeness to population median + prefer cleaner monotonic growth
high["repr_score"] = (
    np.abs(high["acc_median"] - target["acc_median"]) / scales["acc_median"] +
    np.abs(high["pa_delta_1_10"] - target["pa_delta_1_10"]) / scales["pa_delta_1_10"] +
    np.abs(high["pa_slope"] - target["pa_slope"]) / scales["pa_slope"] +
    np.abs(high["pa_t10"] - target["pa_t10"]) / scales["pa_t10"] +
    0.25 * np.abs(high["points_median"] - target["points_median"]) / scales["points_median"] +
    0.75 * np.abs(high["monotonic_fraction"] - high["monotonic_fraction"].median()) / max(robust_scale(high["monotonic_fraction"]), 1e-8)
)

high = high.sort_values(
    ["repr_score", "points_median", "pa_delta_1_10"],
    ascending=[True, False, False]
).reset_index(drop=True)

# ------------------------------------------------------------
# 5) save outputs
# ------------------------------------------------------------
topk = high.head(50).copy()
top_csv = OUTDIR / "representative_highdyn_candidates_top50.csv"
topk.to_csv(top_csv, index=False)

choice = high.iloc[0].copy()
choice_csv = OUTDIR / "representative_highdyn_choice.csv"
pd.DataFrame([choice]).to_csv(choice_csv, index=False)

summary_txt = OUTDIR / "representative_highdyn_summary.txt"
with open(summary_txt, "w") as f:
    f.write("=" * 90 + "\n")
    f.write("REPRESENTATIVE HIGH-DYNAMICS FAILURE CASE SELECTION\n")
    f.write("=" * 90 + "\n")
    f.write(f"Boundary                = {BOUNDARY}\n")
    f.write(f"Minimum points median   = {MIN_POINTS}\n")
    f.write(f"All object-tracks       = {len(tracks)}\n")
    f.write(f"High-dynamics candidates= {len(high)}\n")
    f.write("\n")
    f.write("Population medians within HIGH-DYNAMICS subset\n")
    for k, v in target.items():
        f.write(f"  {k:18s}: {v:.6f}\n")
    f.write("\n")
    f.write("FIRST CHOICE\n")
    for k in [
        "task", "clip", "object_id", "object_short_name",
        "points_median", "acc_median",
        "pa_t1", "pa_t5", "pa_t10",
        "pa_delta_1_10", "pa_slope", "monotonic_fraction", "repr_score"
    ]:
        f.write(f"  {k:18s}: {choice[k]}\n")
    f.write("\n")
    f.write("Caption-ready line\n")
    f.write(
        f"  PA-MPJPE increases from {choice['pa_t1']:.2f} mm at t=1 "
        f"to {choice['pa_t10']:.2f} mm at t=10.\n"
    )
    f.write("=" * 90 + "\n")

print("=" * 90)
print("HIGH-DYNAMICS REPRESENTATIVE SELECTION COMPLETE")
print("=" * 90)
print(f"All object-tracks        : {len(tracks)}")
print(f"High-dynamics candidates : {len(high)}")
print("")
print("HIGH-DYNAMICS MEDIANS")
for k, v in target.items():
    print(f"{k:18s}: {v:.6f}")
print("")
print("FIRST CHOICE")
for k in [
    "task", "clip", "object_id", "object_short_name",
    "points_median", "acc_median",
    "pa_t1", "pa_t5", "pa_t10",
    "pa_delta_1_10", "pa_slope", "monotonic_fraction", "repr_score"
]:
    print(f"{k:18s}: {choice[k]}")
print("")
print("CAPTION:")
print(
    f"A rigid object gradually loses shape consistency during PointWorld rollout "
    f"under a high-dynamics action (representative clip selected near the median "
    f"PA-MPJPE degradation within the high-dynamics population; see Fig. X for "
    f"aggregate statistics across {len(high)} candidate object-tracks). "
    f"PA-MPJPE increases from {choice['pa_t1']:.2f} mm at t=1 to {choice['pa_t10']:.2f} mm at t=10."
)
print("")
print("SAVED:")
print(top_csv)
print(choice_csv)
print(summary_txt)
print("=" * 90)
