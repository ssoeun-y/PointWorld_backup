from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, pearsonr

FRAME_CSV = Path(
    "diagnostics/results/framewise_acc_pa_5task_current10457/"
    "combined_one_row_per_clip_frame.csv"
)

ACC_CSV = Path(
    "diagnostics/results/current_behavior_established_acc_10457/"
    "per_sample_current_established_acc_10457.csv"
)

OUT = Path(
    "diagnostics/results/pa_innovation_future_growth_current10457"
)
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load exact current-10457 files.
# These column names were verified from the actual CSVs.
# ============================================================

d = pd.read_csv(FRAME_CSV)
a = pd.read_csv(ACC_CSV)

required_frame = {
    "task",
    "clip",
    "frame",
    "pa_mpjpe_mm",
}
required_acc = {
    "task",
    "clip",
    "rms_acceleration_si_m_s2",
}

assert required_frame.issubset(d.columns), d.columns.tolist()
assert required_acc.issubset(a.columns), a.columns.tolist()

d = d[
    ["task", "clip", "frame", "pa_mpjpe_mm"]
].copy()

a = a[
    ["task", "clip", "rms_acceleration_si_m_s2"]
].copy()

a = a.rename(
    columns={
        "rms_acceleration_si_m_s2": "rms_acc"
    }
)

d["frame"] = pd.to_numeric(d["frame"], errors="raise")
d["pa_mpjpe_mm"] = pd.to_numeric(
    d["pa_mpjpe_mm"],
    errors="raise"
)

a["rms_acc"] = pd.to_numeric(
    a["rms_acc"],
    errors="raise"
)


# ============================================================
# Population audit
# ============================================================

counts = (
    d.groupby(["task", "clip"])
    .size()
)

print("======================================================")
print("POPULATION AUDIT")
print("======================================================")
print("rows =", len(d))
print("clips =", len(counts))
print("rows/clip:")
print(counts.value_counts().sort_index().to_string())

assert len(d) == 79470, len(d)
assert len(counts) == 7947, len(counts)
assert (counts == 10).all()

frames_per_clip = (
    d.groupby(["task", "clip"])["frame"]
    .apply(lambda x: tuple(sorted(x.tolist())))
)

expected_frames = tuple(range(1, 11))

assert all(
    x == expected_frames
    for x in frames_per_clip
), "Unexpected frame indexing"

print("frame indexing = 1..10 for all 7,947 clips")


# ============================================================
# Remove the ordinary population rollout trend.
#
# We already know PA normally increases with future horizon.
# Therefore, for each task+frame:
#
#   excess PA =
#       clip PA - median PA of same task/same frame
#
# Positive excess:
#   this clip is worse than normal at the same rollout time.
# ============================================================

task_frame_median = (
    d.groupby(["task", "frame"])["pa_mpjpe_mm"]
    .median()
    .rename("task_frame_median_pa")
    .reset_index()
)

d = d.merge(
    task_frame_median,
    on=["task", "frame"],
    how="left",
    validate="many_to_one"
)

d["pa_excess"] = (
    d["pa_mpjpe_mm"]
    - d["task_frame_median_pa"]
)

d = d.merge(
    a,
    on=["task", "clip"],
    how="inner",
    validate="many_to_one"
)

assert d[["task", "clip"]].drop_duplicates().shape[0] == 7947


def residualize_rank(y, controls):
    y = np.asarray(y, dtype=float)
    yr = rankdata(y)

    X = [np.ones(len(y), dtype=float)]

    for c in controls:
        c = np.asarray(c, dtype=float)
        if np.std(c) > 1e-12:
            X.append(rankdata(c))

    X = np.column_stack(X)

    beta = np.linalg.lstsq(
        X,
        yr,
        rcond=None
    )[0]

    return yr - X @ beta


def analyze_checkpoint(checkpoint):
    """
    Example checkpoint=5:

      history = frames 1..4
      checkpoint = frame 5
      future = frames 6..10

    Fit expected excess-PA trajectory using ONLY history.

    innovation:
        actual PA excess at checkpoint
        minus expected checkpoint excess from history.

    Then ask whether that unexpected error predicts
    subsequent future deterioration.
    """

    rows = []

    for (task, clip), g in d.groupby(
        ["task", "clip"],
        sort=False
    ):
        g = g.sort_values("frame")

        hist = g[g["frame"] < checkpoint]
        cur = g[g["frame"] == checkpoint]
        fut = g[g["frame"] > checkpoint]

        if len(hist) < 3:
            continue
        if len(cur) != 1:
            continue
        if len(fut) < 3:
            continue

        xh = hist["frame"].to_numpy(float)
        yh = hist["pa_excess"].to_numpy(float)

        # Expected clip-specific trend BEFORE checkpoint.
        slope_hist, intercept_hist = np.polyfit(
            xh,
            yh,
            1
        )

        current_frame = float(
            cur["frame"].iloc[0]
        )
        current_excess = float(
            cur["pa_excess"].iloc[0]
        )

        expected_current = (
            intercept_hist
            + slope_hist * current_frame
        )

        # Unexpected new error at checkpoint.
        innovation = (
            current_excess
            - expected_current
        )

        xf = fut["frame"].to_numpy(float)
        yf = fut["pa_excess"].to_numpy(float)

        # 1) Does this clip continue getting relatively worse?
        future_slope = float(
            np.polyfit(
                xf,
                yf,
                1
            )[0]
        )

        # 2) How far does future deviate from the
        #    trajectory expected from history alone?
        expected_future = (
            intercept_hist
            + slope_hist * xf
        )

        future_deviation = (
            yf - expected_future
        )

        mean_future_deviation = float(
            np.mean(future_deviation)
        )

        late_n = min(3, len(future_deviation))
        late_future_deviation = float(
            np.mean(
                future_deviation[-late_n:]
            )
        )

        rows.append({
            "task": task,
            "clip": clip,
            "checkpoint": checkpoint,
            "rms_acc": float(
                g["rms_acc"].iloc[0]
            ),
            "history_mean_excess": float(
                np.mean(yh)
            ),
            "history_slope_excess":
                float(slope_hist),
            "checkpoint_excess":
                current_excess,
            "checkpoint_expected":
                float(expected_current),
            "checkpoint_innovation":
                float(innovation),
            "future_excess_slope":
                future_slope,
            "mean_future_deviation":
                mean_future_deviation,
            "late_future_deviation":
                late_future_deviation,
        })

    z = pd.DataFrame(rows)

    task_dummy = pd.get_dummies(
        z["task"],
        drop_first=True,
        dtype=float
    )

    task_controls = [
        task_dummy[c].to_numpy(float)
        for c in task_dummy.columns
    ]

    # Controls:
    # - task
    # - established trajectory-level RMS Acc
    # - how bad this clip already was in history
    # - its pre-checkpoint error growth slope
    controls = [
        z["rms_acc"].to_numpy(float),
        z["history_mean_excess"].to_numpy(float),
        z["history_slope_excess"].to_numpy(float),
    ] + task_controls

    x = z["checkpoint_innovation"].to_numpy(float)

    results = []

    for target in [
        "future_excess_slope",
        "mean_future_deviation",
        "late_future_deviation",
    ]:
        y = z[target].to_numpy(float)

        sr, sp = spearmanr(x, y)

        rx = residualize_rank(
            x,
            controls
        )
        ry = residualize_rank(
            y,
            controls
        )

        pr, pp = pearsonr(
            rx,
            ry
        )

        results.append({
            "checkpoint": checkpoint,
            "target": target,
            "N": len(z),
            "simple_spearman": sr,
            "simple_p": sp,
            "partial_rank_rho": pr,
            "partial_p": pp,
        })

    return z, pd.DataFrame(results)


# ============================================================
# Do NOT cherry-pick one frame.
# Main checkpoint=5 plus neighboring robustness checkpoints.
# ============================================================

all_features = []
all_results = []

for checkpoint in [4, 5, 6]:
    z, r = analyze_checkpoint(checkpoint)

    all_features.append(z)
    all_results.append(r)

features = pd.concat(
    all_features,
    ignore_index=True
)

results = pd.concat(
    all_results,
    ignore_index=True
)

features.to_csv(
    OUT / "per_clip_checkpoint_features.csv",
    index=False
)

results.to_csv(
    OUT / "checkpoint_future_growth_summary.csv",
    index=False
)


print()
print("======================================================")
print("CHECKPOINT INNOVATION -> FUTURE DETERIORATION")
print("======================================================")
print(
    "Controls: task + established RMS Acc + "
    "pre-checkpoint PA excess level + "
    "pre-checkpoint PA excess slope"
)
print()
print(
    results.to_string(index=False)
)


# ============================================================
# Main checkpoint=5 easy effect size.
# Within each task:
# bottom vs top quartile checkpoint innovation,
# compare subsequent late deviation.
# ============================================================

main = features[
    features["checkpoint"] == 5
].copy()

effect_rows = []

for task, g in main.groupby("task"):
    q25 = g["checkpoint_innovation"].quantile(0.25)
    q75 = g["checkpoint_innovation"].quantile(0.75)

    low = g[
        g["checkpoint_innovation"] <= q25
    ]
    high = g[
        g["checkpoint_innovation"] >= q75
    ]

    effect_rows.append({
        "task": task,
        "N_low": len(low),
        "N_high": len(high),
        "checkpoint_innovation_low_median":
            float(
                low["checkpoint_innovation"].median()
            ),
        "checkpoint_innovation_high_median":
            float(
                high["checkpoint_innovation"].median()
            ),
        "late_future_deviation_low_median":
            float(
                low["late_future_deviation"].median()
            ),
        "late_future_deviation_high_median":
            float(
                high["late_future_deviation"].median()
            ),
    })

effect = pd.DataFrame(effect_rows)

effect.to_csv(
    OUT / "checkpoint5_quartile_effect.csv",
    index=False
)

print()
print("======================================================")
print("CHECKPOINT=5 QUARTILE EFFECT")
print("======================================================")
print(
    effect.to_string(index=False)
)


# ============================================================
# Sign consistency across tasks at checkpoint=5.
# ============================================================

print()
print("======================================================")
print("TASKWISE CHECKPOINT=5")
print("innovation -> late future deviation")
print("======================================================")

task_rows = []

for task, g in main.groupby("task"):
    rho, p = spearmanr(
        g["checkpoint_innovation"],
        g["late_future_deviation"]
    )

    task_rows.append({
        "task": task,
        "N": len(g),
        "rho": rho,
        "p": p,
    })

task_df = pd.DataFrame(task_rows)

task_df.to_csv(
    OUT / "checkpoint5_taskwise.csv",
    index=False
)

print(task_df.to_string(index=False))

print()
print("Saved to:")
print(OUT)
