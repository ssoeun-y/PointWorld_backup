from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE = Path("diagnostics/results")
SRC = BASE / "close_anchor_relative_replay_current524"
OLD = BASE / "close_anchor_causal_5task_current10457"
OUT = BASE / "close_anchor_relative_analysis_current524"
OUT.mkdir(exist_ok=True)

TASKS = ("0000", "0001", "0006", "0008", "0010")
EXPECTED = (57, 144, 62, 111, 150)
B = 0.24580418

frames = []
originals = []

for t, expected in zip(TASKS, EXPECTED):
    task = f"task-{t}"
    f = pd.read_csv(SRC/task/"per_frame_relative_metrics.csv")
    o = pd.read_csv(OLD/task/"per_clip_causal_metrics.csv")

    assert o["clip"].nunique() == expected
    assert f["clip"].nunique() == expected
    assert not f.duplicated(["clip", "frame", "condition"]).any()
    assert set(f["condition"]) == {"GT", "Original", "Free", "Anchor"}
    assert (
        f.groupby(["clip", "frame"])["condition"].nunique() == 4
    ).all()

    frames.append(f)
    originals.append(o)

df = pd.concat(frames, ignore_index=True)
orig = pd.concat(originals, ignore_index=True)

assert len(orig) == 524
assert not orig.duplicated(["task", "clip"]).any()

records = []

for (task, clip, condition), g in df.groupby(
    ["task", "clip", "condition"], sort=False
):
    g = g.sort_values("frame")
    post = g[g["frames_since_close"] > 0]

    assert len(post) >= 1, (clip, condition)
    assert g.iloc[0]["frames_since_close"] == 0

    records.append({
        "task": task,
        "clip": clip,
        "condition": condition,
        "end_centroid_mm":
            float(g.iloc[-1]["relative_centroid_drift_mm"]),
        "mean_centroid_mm":
            float(post["relative_centroid_drift_mm"].mean()),
        "end_point_mm":
            float(g.iloc[-1]["relative_point_motion_median_mm"]),
        "mean_point_mm":
            float(post["relative_point_motion_median_mm"].mean()),
        "end_gap_mm":
            float(g.iloc[-1]["absolute_q10_gap_mm"]),
        "end_gt_error_mm":
            float(g.iloc[-1][
                "original_gt_relative_delta_error_mm"
            ]) if condition == "Original" else np.nan,
    })

clip_long = pd.DataFrame(records)

assert len(clip_long) == 524 * 4

wide = clip_long.pivot(
    index=["task", "clip"],
    columns="condition",
    values=[
        "end_centroid_mm",
        "mean_centroid_mm",
        "end_point_mm",
        "mean_point_mm",
        "end_gap_mm",
        "end_gt_error_mm",
    ],
)
wide.columns = [
    f"{metric}_{condition}"
    for metric, condition in wide.columns
]
wide = wide.reset_index()

a = orig[[
    "task", "clip", "original_acc_si",
    "free_acc_si", "anchor_acc_si",
    "gt_acquisition_min_gap_mm"
]]

wide = wide.merge(
    a, on=["task", "clip"], validate="one_to_one"
)

assert len(wide) == 524

wide["strict_boundary"] = (
    (wide["original_acc_si"] > B)
    & (wide["free_acc_si"] <= B)
    & (wide["anchor_acc_si"] <= B)
)

wide.to_csv(
    OUT/"per_clip_relative_drift.csv", index=False
)

rng = np.random.default_rng(42)

def comparison(data, metric):
    free = data[f"{metric}_Free"].to_numpy(dtype=float)
    anchor = data[f"{metric}_Anchor"].to_numpy(dtype=float)

    assert np.isfinite(free).all()
    assert np.isfinite(anchor).all()

    d = free - anchor
    nz = d[d != 0]

    p = (
        wilcoxon(nz, alternative="two-sided").pvalue
        if len(nz) else 1.0
    )

    # Clip-paired bootstrap CI: Free minus Anchor.
    boot = np.median(
        d[rng.integers(0, len(d), (3000, len(d)))],
        axis=1
    )
    lo, hi = np.quantile(boot, [0.025, 0.975])

    print(
        f"{metric}: "
        f"Free={np.median(free):.4f} mm | "
        f"Anchor={np.median(anchor):.4f} mm | "
        f"paired gain={np.median(d):+.4f} mm | "
        f"95%CI=[{lo:+.4f},{hi:+.4f}] | "
        f"Anchor lower={100*np.mean(d>0):.1f}% | "
        f"p={p:.4g}"
    )

print("\n===== GT / ORIGINAL REFERENCE =====")
for condition in ("GT", "Original"):
    print(
        condition,
        "end centroid relative motion median:",
        round(wide[f"end_centroid_mm_{condition}"].median(), 4),
        "mm"
    )

print(
    "Original vs GT endpoint relative-motion error:",
    round(wide["end_gt_error_mm_Original"].median(), 4),
    "mm"
)

for name, sub in (
    ("ALL MATCHED", wide),
    ("STRICT BOUNDARY", wide[wide["strict_boundary"]])
):
    print(f"\n===== {name}: N={len(sub)} =====")
    assert len(sub) > 0

    comparison(sub, "end_centroid_mm")
    comparison(sub, "mean_centroid_mm")
    comparison(sub, "end_point_mm")
    comparison(sub, "mean_point_mm")

print("\n===== TASKWISE PRIMARY METRIC =====")
for task, sub in wide.groupby("task"):
    d = (
        sub["end_centroid_mm_Free"]
        - sub["end_centroid_mm_Anchor"]
    )
    print(
        task,
        f"N={len(sub)}",
        f"paired gain={d.median():+.4f} mm",
        f"Anchor lower={100*(d>0).mean():.1f}%"
    )

print("\nSTRICT BOUNDARY N =", wide["strict_boundary"].sum())
print("OUTPUT =", OUT/"per_clip_relative_drift.csv")
print("[ANALYSIS COMPLETE]")
