from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest

BASE = Path("diagnostics/results")
OLD = BASE / "close_anchor_causal_5task_current10457"
NEW = BASE / "close_anchor_relative_replay_current524"
OUT = BASE / "gt_stable_grasp_analysis_current524"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "0000": 57, "0001": 144, "0006": 62,
    "0008": 111, "0010": 150
}
B = 0.24580418

# GT-based proxy criteria:
# initial contact proximity, sustained proximity,
# maximum GT relative displacement, endpoint point displacement,
# minimum point count, minimum frames, minimum coverage
TIERS = {
    "strict": (10, 15, 5, 10, 30, 3, 0.80),
    "moderate": (20, 25, 10, 15, 30, 3, 0.80),
    "permissive": (30, 40, 15, 25, 20, 3, 0.70),
}

METRICS = {
    "end_centroid": "relative_centroid_drift_mm",
    "mean_centroid": "relative_centroid_drift_mm",
    "end_point": "relative_point_motion_median_mm",
    "mean_point": "relative_point_motion_median_mm",
}

frame_parts = []
original_parts = []

for tid, expected in TASKS.items():
    task = f"task-{tid}"

    f = pd.read_csv(
        NEW / task / "per_frame_relative_metrics.csv"
    )
    o = pd.read_csv(
        OLD / task / "per_clip_causal_metrics.csv"
    )

    assert len(o) == expected
    assert o["clip"].nunique() == expected
    assert f["clip"].nunique() == expected

    assert not f.duplicated(
        ["task", "clip", "frame", "condition"]
    ).any()

    conditions_ok = f.groupby(
        ["task", "clip", "frame"]
    )["condition"].agg(
        lambda x: set(x) == {
            "GT", "Original", "Free", "Anchor"
        }
    )

    assert conditions_ok.all(), task

    frame_parts.append(f)
    original_parts.append(o)

frame = pd.concat(frame_parts, ignore_index=True)
old = pd.concat(original_parts, ignore_index=True)

assert len(old) == 524
assert not old.duplicated(["task", "clip"]).any()

# --------------------------------------------------
# Per-clip GT stability and four-condition motion
# --------------------------------------------------

rows = []

for (task, clip), group in frame.groupby(
    ["task", "clip"], sort=False
):
    gt = group[
        group["condition"] == "GT"
    ].sort_values("frame")

    assert len(gt) >= 2
    assert gt.iloc[0]["frames_since_close"] == 0

    if gt["n_common_points"].nunique() != 1:
        raise RuntimeError(
            f"{clip}: inconsistent common points"
        )

    row = {
        "task": task,
        "clip": clip,
        "scored_frames": len(gt),
        "common_points": int(
            gt["n_common_points"].iloc[0]
        ),
        "gt_initial_min_gap_replayed_mm":
            float(gt["absolute_min_gap_mm"].iloc[0]),
        "gt_max_min_gap_mm":
            float(gt["absolute_min_gap_mm"].max()),
        "gt_max_centroid_mm":
            float(gt["relative_centroid_drift_mm"].max()),
        "gt_end_centroid_mm":
            float(gt["relative_centroid_drift_mm"].iloc[-1]),
        "gt_end_point_mm":
            float(
                gt["relative_point_motion_median_mm"].iloc[-1]
            ),
    }

    for condition in (
        "GT", "Original", "Free", "Anchor"
    ):
        g = group[
            group["condition"] == condition
        ].sort_values("frame")

        assert np.array_equal(
            g["frame"].to_numpy(),
            gt["frame"].to_numpy()
        )

        for name, column in METRICS.items():
            if name.startswith("mean_"):
                value = g.loc[
                    g["frames_since_close"] > 0, column
                ].mean()
            else:
                value = g[column].iloc[-1]

            row[f"{name}_{condition}"] = float(value)

    rows.append(row)

df = pd.DataFrame(rows)

assert len(df) == 524
assert not df.duplicated(["task", "clip"]).any()

cols = [
    "task", "clip", "close_side", "closed_frames",
    "gt_acquisition_min_gap_mm",
    "original_acc_si", "free_acc_si", "anchor_acc_si"
]

df = df.merge(
    old[cols],
    on=["task", "clip"],
    validate="one_to_one"
)

assert len(df) == 524

# Independent replay consistency check
assert np.allclose(
    df["gt_initial_min_gap_replayed_mm"],
    df["gt_acquisition_min_gap_mm"],
    atol=1e-4,
    rtol=0,
), "GT acquisition gap mismatch"

df["coverage"] = (
    df["scored_frames"] / df["closed_frames"]
)
assert df["coverage"].le(1).all()

df["strict_boundary"] = (
    (df["original_acc_si"] > B)
    & (df["free_acc_si"] <= B)
    & (df["anchor_acc_si"] <= B)
)

assert int(df["strict_boundary"].sum()) == 64

# --------------------------------------------------
# GT-only candidate selection
# --------------------------------------------------

for tier, values in TIERS.items():
    near, persist, maxdrift, endpoint, npoints, nframes, coverage = values

    df[tier] = (
        (df["gt_acquisition_min_gap_mm"] <= near)
        & (df["gt_max_min_gap_mm"] <= persist)
        & (df["gt_max_centroid_mm"] <= maxdrift)
        & (df["gt_end_point_mm"] <= endpoint)
        & (df["common_points"] >= npoints)
        & (df["scored_frames"] >= nframes)
        & (df["coverage"] >= coverage)
    )

df.to_csv(
    OUT / "per_clip_gt_stability.csv", index=False
)

# --------------------------------------------------
# Paired statistics
# --------------------------------------------------

rng = np.random.default_rng(20261010)

def compare(name, sub, metric):
    n = len(sub)
    result = {
        "subset": name, "metric": metric, "N": n
    }

    if n == 0:
        return result

    values = {
        condition: sub[
            f"{metric}_{condition}"
        ].to_numpy(dtype=float)
        for condition in ("GT", "Original", "Free", "Anchor")
    }

    assert all(
        np.isfinite(v).all() for v in values.values()
    )

    d = values["Free"] - values["Anchor"]
    nz = d[d != 0]

    for condition, v in values.items():
        result[f"{condition}_median_mm"] = float(
            np.median(v)
        )

    result["paired_gain_mm"] = float(np.median(d))
    result["anchor_better_pct"] = float(
        100 * np.mean(d > 0)
    )

    result["sign_p"] = (
        float(binomtest(
            int(np.sum(nz > 0)), len(nz), 0.5
        ).pvalue)
        if len(nz) else 1.0
    )

    result["wilcoxon_p"] = (
        float(wilcoxon(nz).pvalue)
        if len(nz) else 1.0
    )

    if n >= 5:
        idx = rng.integers(0, n, (5000, n))
        bootstrap = np.median(d[idx], axis=1)
        lo, hi = np.quantile(
            bootstrap, [0.025, 0.975]
        )
        result["CI_low_mm"] = float(lo)
        result["CI_high_mm"] = float(hi)

    return result


results = []

# Main populations
for name, sub in [
    ("all_524", df),
    ("boundary_64", df[df["strict_boundary"]])
]:
    for metric in METRICS:
        results.append(compare(name, sub, metric))

# GT-selected populations
for tier in TIERS:
    selected = df[df[tier]]

    groups = {
        tier: selected,
        f"{tier}_boundary":
            selected[selected["strict_boundary"]],
        f"{tier}_single_arm":
            selected[selected["close_side"] != "both"],
    }

    for name, sub in groups.items():
        for metric in METRICS:
            results.append(compare(name, sub, metric))

summary = pd.DataFrame(results)
summary.to_csv(
    OUT / "paired_comparisons.csv", index=False
)

# Taskwise primary comparison
task_results = []

for tier in TIERS:
    for task, sub in df[df[tier]].groupby("task"):
        task_results.append(
            compare(f"{tier}_{task}", sub, "end_centroid")
        )

pd.DataFrame(task_results).to_csv(
    OUT / "taskwise_primary.csv", index=False
)

# --------------------------------------------------
# Frame-by-frame curves
# --------------------------------------------------

merged = frame.merge(
    df[[
        "task", "clip", "strict_boundary",
        *TIERS.keys()
    ]],
    on=["task", "clip"],
    validate="many_to_one"
)

curve_rows = []

for tier in TIERS:
    for boundary in ("all", "strict_boundary"):
        selected = merged[merged[tier]]

        if boundary == "strict_boundary":
            selected = selected[
                selected["strict_boundary"]
            ]

        for (condition, dt), group in selected.groupby(
            ["condition", "frames_since_close"]
        ):
            curve_rows.append({
                "tier": tier,
                "boundary": boundary,
                "condition": condition,
                "frames_since_close": int(dt),
                "n_clips": group["clip"].nunique(),
                "median_centroid_mm": float(
                    group["relative_centroid_drift_mm"].median()
                ),
                "median_min_gap_mm": float(
                    group["absolute_min_gap_mm"].median()
                ),
            })

pd.DataFrame(curve_rows).to_csv(
    OUT / "framewise_medians.csv", index=False
)

# --------------------------------------------------
# Report
# --------------------------------------------------

print("\n===== GT-STABLE GRASP CANDIDATES =====")
print("Overall: N=524")
print("Strict boundary: N=64")

for tier in TIERS:
    sub = df[df[tier]]

    print(f"\n===== {tier.upper()} =====")
    print(
        f"GT-stable candidates: {len(sub)}/524"
    )
    print(
        "Boundary-compliant:",
        int(sub["strict_boundary"].sum())
    )

    for name in (
        tier,
        f"{tier}_boundary",
        f"{tier}_single_arm"
    ):
        r = summary[
            (summary["subset"] == name)
            & (summary["metric"] == "end_centroid")
        ].iloc[0]

        n = int(r["N"])

        if n == 0:
            print(name, ": N=0")
            continue

        print(f"\n{name}: N={n}")
        print(
            "GT / Original / Free / Anchor:",
            *(f"{r[f'{c}_median_mm']:.4f}"
              for c in ("GT", "Original", "Free", "Anchor")),
            "mm"
        )
        print(
            f"Paired gain: {r['paired_gain_mm']:+.4f} mm"
        )
        print(
            f"Anchor better: {r['anchor_better_pct']:.1f}%"
        )
        print(
            f"Sign p={r['sign_p']:.5g}",
            f"Wilcoxon p={r['wilcoxon_p']:.5g}"
        )

        if n >= 5:
            print(
                "Bootstrap 95% CI:",
                f"[{r['CI_low_mm']:+.4f},",
                f"{r['CI_high_mm']:+.4f}] mm"
            )

print("\n===== TASKWISE STRICT CANDIDATES =====")

for r in task_results:
    if r["subset"].startswith("strict_"):
        print(
            r["subset"],
            f"N={r['N']}",
            f"paired gain={r['paired_gain_mm']:+.4f} mm"
        )

print("\n[ANALYSIS COMPLETE]")
print("Results:", OUT)
print(
    "Interpretation: GT-stable candidates,",
    "not independently verified physical grasps."
)
