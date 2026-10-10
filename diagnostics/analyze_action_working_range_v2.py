#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr


MAIN = Path(
    "diagnostics/results/established_action_metrics_5task/"
    "per_sample_established_metrics_7987.csv"
)

RAW = Path(
    "diagnostics/results/raw_action_good_bad_7987/"
    "raw_action_profiles_7987.csv"
)

OUT = Path(
    "diagnostics/results/action_working_range_v2_7987"
)

SEED = 20260902
BOOTSTRAP = 300

EXPECTED = {
    "task-0000": 377,
    "task-0001": 2322,
    "task-0006": 1782,
    "task-0008": 1827,
    "task-0010": 1679,
}

REQ = [
    "clip",
    "task",
    "l2_moved_mm",
    "rms_acceleration",
    "mean_squared_jerk",
]


def stable_number(text):
    h = hashlib.sha256(f"{SEED}|{text}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def split_data(df):
    result = pd.Series(index=df.index, dtype="object")

    for task, g in df.groupby("task"):
        idx = sorted(
            g.index,
            key=lambda i: stable_number(str(df.at[i, "clip"]))
        )

        n_cal = round(len(idx) * 0.70)

        result.loc[idx[:n_cal]] = "calibration"
        result.loc[idx[n_cal:]] = "validation"

    return result


def task_dummies(task):
    return pd.get_dummies(
        task.astype(str),
        drop_first=True,
        dtype=float
    ).to_numpy(float)


def ols_bic(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    residual = y - X @ beta
    sse = max(float(residual @ residual), 1e-15)

    n, k = X.shape

    bic = n * np.log(sse / n) + k * np.log(n)

    return beta, bic


def prepare_xy(df, metric):
    x_raw = df[metric].to_numpy(float)

    positive = x_raw[x_raw > 0]

    if len(positive) == 0:
        raise RuntimeError(f"No positive values for {metric}")

    floor = positive.min() * 0.5

    # Dynamics scale is strongly skewed, so fit on log scale.
    x = np.log10(np.maximum(x_raw, floor))

    # Prediction error is also heavy-tailed.
    y = np.log1p(df["l2_moved_mm"].to_numpy(float))

    task = task_dummies(df["task"])

    return x_raw, x, y, task


def fit_models(df, metric):

    x_raw, x, y, task = prepare_xy(df, metric)

    # -----------------------------------------
    # Model 1: simple monotonic/linear trend
    # -----------------------------------------

    X_linear = np.column_stack([
        np.ones(len(df)),
        x,
        task
    ])

    beta_linear, bic_linear = ols_bic(y, X_linear)

    # -----------------------------------------
    # Model 2: smooth curvature
    # -----------------------------------------

    xc = x - x.mean()

    X_quad = np.column_stack([
        np.ones(len(df)),
        x,
        xc ** 2,
        task
    ])

    beta_quad, bic_quad = ols_bic(y, X_quad)

    # -----------------------------------------
    # Model 3: one candidate change point
    # -----------------------------------------

    candidate_taus = np.unique(
        np.quantile(
            x,
            np.linspace(0.20, 0.90, 71)
        )
    )

    best = None

    for tau in candidate_taus:

        hinge = np.maximum(0.0, x - tau)

        X_seg = np.column_stack([
            np.ones(len(df)),
            x,
            hinge,
            task
        ])

        beta, bic = ols_bic(y, X_seg)

        if best is None or bic < best["bic"]:
            best = {
                "tau": float(tau),
                "beta": beta,
                "bic": float(bic),
            }

    pre_slope = float(best["beta"][1])

    post_slope = float(
        best["beta"][1] +
        best["beta"][2]
    )

    return {
        "threshold_raw": float(10 ** best["tau"]),
        "bic_linear": float(bic_linear),
        "bic_quadratic": float(bic_quad),
        "bic_segmented": float(best["bic"]),

        # Positive = segmented model fits better.
        "delta_bic_seg_vs_linear":
            float(bic_linear - best["bic"]),

        "delta_bic_seg_vs_quadratic":
            float(bic_quad - best["bic"]),

        "slope_pre": pre_slope,
        "slope_post": post_slope,
        "slope_change":
            float(post_slope - pre_slope),
    }


def bootstrap_threshold(df, metric):

    rng = np.random.default_rng(SEED)

    groups = {
        task: g.index.to_numpy()
        for task, g in df.groupby("task")
    }

    thresholds = []

    for _ in range(BOOTSTRAP):

        sampled = np.concatenate([
            rng.choice(
                idx,
                size=len(idx),
                replace=True
            )
            for idx in groups.values()
        ])

        try:
            thresholds.append(
                fit_models(
                    df.loc[sampled],
                    metric
                )["threshold_raw"]
            )
        except Exception:
            pass

    if len(thresholds) < 50:
        return np.nan, np.nan, len(thresholds)

    lo, hi = np.quantile(
        thresholds,
        [0.025, 0.975]
    )

    return float(lo), float(hi), len(thresholds)


def cliff_and_p(high, low):

    high = np.asarray(high, float)
    low = np.asarray(low, float)

    if len(high) == 0 or len(low) == 0:
        return np.nan, np.nan

    result = mannwhitneyu(
        high,
        low,
        alternative="two-sided"
    )

    delta = (
        2.0 * result.statistic /
        (len(high) * len(low))
        - 1.0
    )

    return float(delta), float(result.pvalue)


def validate_threshold(df, metric, threshold):

    inside = df.loc[
        df[metric] <= threshold,
        "l2_moved_mm"
    ].to_numpy(float)

    outside = df.loc[
        df[metric] > threshold,
        "l2_moved_mm"
    ].to_numpy(float)

    cliff, p = cliff_and_p(
        outside,
        inside
    )

    med_in = (
        float(np.median(inside))
        if len(inside) else np.nan
    )

    med_out = (
        float(np.median(outside))
        if len(outside) else np.nan
    )

    return {
        "val_n_in": len(inside),
        "val_n_out": len(outside),

        "val_med_in_mm": med_in,
        "val_med_out_mm": med_out,

        "val_out_in_ratio":
            med_out / med_in
            if med_in > 0 else np.nan,

        "val_cliff_delta": cliff,
        "val_p": p,
    }


def binned_curve(df, metric, bins=20):

    d = df[
        [metric, "l2_moved_mm"]
    ].dropna().copy()

    d["bin"] = pd.qcut(
        d[metric].rank(method="first"),
        q=bins,
        labels=False
    )

    rng = np.random.default_rng(1234)

    rows = []

    for b, g in d.groupby("bin"):

        y = g["l2_moved_mm"].to_numpy(float)

        boot = []

        for _ in range(300):
            boot.append(
                np.median(
                    rng.choice(
                        y,
                        size=len(y),
                        replace=True
                    )
                )
            )

        rows.append({
            "bin": int(b),
            "N": len(g),

            "dynamics_median":
                float(g[metric].median()),

            "dynamics_p10":
                float(g[metric].quantile(.10)),

            "dynamics_p90":
                float(g[metric].quantile(.90)),

            "error_median_mm":
                float(np.median(y)),

            "error_ci_low_mm":
                float(np.quantile(boot, .025)),

            "error_ci_high_mm":
                float(np.quantile(boot, .975)),
        })

    return pd.DataFrame(rows)


def plot_main_curve(
    df,
    curve,
    metric,
    threshold,
    filename
):

    fig, ax = plt.subplots(
        figsize=(7.5, 5.3)
    )

    x = df[metric].to_numpy(float)
    y = df["l2_moved_mm"].to_numpy(float)

    positive = x[x > 0]
    floor = positive.min() * 0.5

    ax.scatter(
        np.maximum(x, floor),
        y,
        s=5,
        alpha=.05,
        rasterized=True
    )

    ax.errorbar(
        curve["dynamics_median"],
        curve["error_median_mm"],

        yerr=[
            curve["error_median_mm"]
            - curve["error_ci_low_mm"],

            curve["error_ci_high_mm"]
            - curve["error_median_mm"],
        ],

        marker="o",
        linewidth=1.6,
        capsize=2,
    )

    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=1.5,
        label=(
            "candidate boundary "
            f"{threshold:.5g}"
        ),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        metric +
        " (PointWorld discrete units)"
    )

    ax.set_ylabel(
        "Official moved-L2 (mm)"
    )

    ax.legend()
    ax.grid(alpha=.2)

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=220
    )

    plt.close(fig)


def plot_taskwise(
    df,
    metric,
    threshold,
    filename
):

    fig, ax = plt.subplots(
        figsize=(7.8, 5.5)
    )

    for task, g in df.groupby(
        "task",
        sort=True
    ):

        c = binned_curve(
            g,
            metric,
            bins=8
        )

        ax.plot(
            c["dynamics_median"],
            c["error_median_mm"],
            marker="o",
            linewidth=1.2,
            label=task
        )

    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=1.4
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        metric +
        " (PointWorld discrete units)"
    )

    ax.set_ylabel(
        "Median moved-L2 (mm)"
    )

    ax.legend(fontsize=8)
    ax.grid(alpha=.2)

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=220
    )

    plt.close(fig)


def plot_joint_heatmap(df):

    d = df.copy()

    # Normalize only for this visualization
    # so tasks with different raw ranges can be
    # compared on the same panel.

    d["acc_rank"] = (
        d.groupby("task")
        ["rms_acceleration"]
        .rank(pct=True)
    )

    d["msj_rank"] = (
        d.groupby("task")
        ["mean_squared_jerk"]
        .rank(pct=True)
    )

    d["task_error_median"] = (
        d.groupby("task")
        ["l2_moved_mm"]
        .transform("median")
    )

    d["error_ratio"] = (
        d["l2_moved_mm"] /
        d["task_error_median"]
    )

    edges = np.linspace(0, 1, 6)

    labels = [
        "0-20",
        "20-40",
        "40-60",
        "60-80",
        "80-100",
    ]

    d["acc_bin"] = pd.cut(
        d["acc_rank"],
        edges,
        labels=labels,
        include_lowest=True
    )

    d["msj_bin"] = pd.cut(
        d["msj_rank"],
        edges,
        labels=labels,
        include_lowest=True
    )

    values = d.pivot_table(
        index="msj_bin",
        columns="acc_bin",
        values="error_ratio",
        aggfunc="median",
        observed=False
    )

    counts = d.pivot_table(
        index="msj_bin",
        columns="acc_bin",
        values="clip",
        aggfunc="count",
        fill_value=0,
        observed=False
    )

    values.to_csv(
        OUT /
        "rms_msj_joint_heatmap_values.csv"
    )

    fig, ax = plt.subplots(
        figsize=(6.7, 5.6)
    )

    im = ax.imshow(
        values.to_numpy(float),
        origin="lower",
        aspect="auto"
    )

    ax.set_xticks(
        range(5),
        labels
    )

    ax.set_yticks(
        range(5),
        labels
    )

    ax.set_xlabel(
        "RMS acceleration within-task percentile"
    )

    ax.set_ylabel(
        "MSJ within-task percentile"
    )

    colorbar = fig.colorbar(
        im,
        ax=ax
    )

    colorbar.set_label(
        "Median moved-L2 / task median"
    )

    for i in range(5):
        for j in range(5):

            value = values.iloc[i, j]
            n = int(counts.iloc[i, j])

            text = (
                f"{value:.2f}x\nn={n}"
                if np.isfinite(value)
                else f"n={n}"
            )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7
            )

    fig.tight_layout()

    fig.savefig(
        OUT /
        "rms_msj_joint_heatmap.png",
        dpi=220
    )

    plt.close(fig)


def bh_fdr(p):

    p = np.asarray(p, float)

    q = np.full(
        len(p),
        np.nan
    )

    valid = np.isfinite(p)

    idx = np.where(valid)[0]

    if len(idx) == 0:
        return q

    order = idx[
        np.argsort(p[idx])
    ]

    m = len(order)

    adjusted = (
        p[order] *
        m /
        np.arange(1, m + 1)
    )

    adjusted = (
        np.minimum.accumulate(
            adjusted[::-1]
        )[::-1]
    )

    q[order] = np.minimum(
        adjusted,
        1.0
    )

    return q


def screen_other_action_factors(main):

    if not RAW.exists():
        return

    raw = pd.read_csv(RAW)

    keys = [
        c for c in ["task", "clip"]
        if c in raw.columns
        and c in main.columns
    ]

    if "clip" not in keys:
        return

    raw = raw.drop_duplicates(keys)

    merged = (
        main[
            ["task", "clip", "l2_moved_mm"]
        ]
        .merge(
            raw,
            on=keys,
            how="left",
            suffixes=("", "_raw")
        )
    )

    # Only inspect action-side variables.
    # Explicitly reject output / GT / scene fields.

    banned = (
        "l2",
        "error",
        "pred",
        "gt_",
        "scene",
        "visibility",
        "depth",
        "response",
        "quality",
        "score",
        "label",
    )

    allowed = (
        "rotation",
        "orient",
        "gripper",
        "path",
        "speed",
        "velocity",
        "accel",
        "jerk",
        "turn",
        "reversal",
        "disp",
        "range",
        "step",
    )

    candidates = []

    for c in merged.columns:

        name = c.lower()

        if c in {
            "task",
            "clip",
            "l2_moved_mm"
        }:
            continue

        if any(x in name for x in banned):
            continue

        if not any(x in name for x in allowed):
            continue

        numeric = pd.to_numeric(
            merged[c],
            errors="coerce"
        )

        if numeric.notna().sum() < 500:
            continue

        if numeric.nunique(
            dropna=True
        ) < 10:
            continue

        candidates.append(c)

    rows = []

    for c in sorted(set(candidates)):

        g = merged[
            ["task", "l2_moved_mm", c]
        ].copy()

        g[c] = pd.to_numeric(
            g[c],
            errors="coerce"
        )

        g = g.dropna()

        g["x_rank"] = (
            g.groupby("task")[c]
            .rank(pct=True)
        )

        g["error_rank"] = (
            g.groupby("task")
            ["l2_moved_mm"]
            .rank(pct=True)
        )

        rho, p = spearmanr(
            g["x_rank"],
            g["error_rank"]
        )

        low = g.loc[
            g["x_rank"] <= .25,
            "l2_moved_mm"
        ].to_numpy(float)

        high = g.loc[
            g["x_rank"] >= .75,
            "l2_moved_mm"
        ].to_numpy(float)

        cliff, mw_p = cliff_and_p(
            high,
            low
        )

        low_med = float(
            np.median(low)
        )

        high_med = float(
            np.median(high)
        )

        rows.append({
            "factor": c,
            "N": len(g),

            "within_task_rho": rho,
            "spearman_p": p,

            "low_q_error_mm":
                low_med,

            "high_q_error_mm":
                high_med,

            "high_low_ratio":
                high_med / low_med
                if low_med > 0
                else np.nan,

            "cliff_delta":
                cliff,

            "mw_p":
                mw_p,
        })

    if not rows:
        return

    result = pd.DataFrame(rows)

    result["spearman_fdr_q"] = bh_fdr(
        result["spearman_p"].values
    )

    result = result.sort_values(
        "within_task_rho",
        key=lambda x: x.abs(),
        ascending=False
    )

    result.to_csv(
        OUT /
        "exploratory_action_factor_screen.csv",
        index=False
    )

    top = result.head(15).iloc[::-1]

    fig, ax = plt.subplots(
        figsize=(8, 5.8)
    )

    ax.barh(
        top["factor"],
        top["within_task_rho"]
    )

    ax.axvline(
        0,
        linewidth=1
    )

    ax.set_xlabel(
        "Within-task Spearman rho with moved-L2"
    )

    ax.set_title(
        "Exploratory action-only factor screen"
    )

    fig.tight_layout()

    fig.savefig(
        OUT /
        "exploratory_action_factor_top15.png",
        dpi=220
    )

    plt.close(fig)


def verdict(row):

    # We deliberately do NOT force a threshold.
    #
    # BIC >= 10:
    # strong preference over simple linear.
    #
    # Segmented must also beat quadratic,
    # otherwise it may just be smooth curvature.

    segmented_vs_linear = (
        row["delta_bic_seg_vs_linear"]
        >= 10
    )

    segmented_vs_quadratic = (
        row["delta_bic_seg_vs_quadratic"]
        >= 6
    )

    gets_steeper = (
        row["slope_post"]
        > row["slope_pre"]
    )

    validation_direction = (
        row["val_med_out_mm"]
        >
        row["val_med_in_mm"]
        and
        row["val_cliff_delta"] > 0
    )

    if (
        segmented_vs_linear
        and
        segmented_vs_quadratic
        and
        gets_steeper
        and
        validation_direction
    ):
        return "BREAKPOINT_CANDIDATE"

    if (
        row["bic_quadratic"] + 10
        < row["bic_linear"]
    ):
        return (
            "SMOOTH_NONLINEAR_"
            "NO_UNIQUE_THRESHOLD"
        )

    if (
        segmented_vs_linear
        and
        not segmented_vs_quadratic
    ):
        return (
            "CURVED_RELATION_"
            "THRESHOLD_NOT_UNIQUE"
        )

    return (
        "NO_STRONG_THRESHOLD_EVIDENCE"
    )


# ============================================================
# MAIN
# ============================================================

OUT.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(MAIN)

missing = [
    c for c in REQ
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )

for c in [
    "l2_moved_mm",
    "rms_acceleration",
    "mean_squared_jerk",
]:
    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )

if df[REQ].isna().any().any():
    raise RuntimeError(
        "NaN exists in required columns"
    )

if len(df) != 7987:
    raise RuntimeError(
        f"Expected 7987, got {len(df)}"
    )

if df["clip"].duplicated().any():
    raise RuntimeError(
        "Duplicate clip IDs detected"
    )

counts = (
    df["task"]
    .value_counts()
    .sort_index()
    .to_dict()
)

if counts != EXPECTED:
    raise RuntimeError(
        f"Task-count mismatch: {counts}"
    )


# ------------------------------------------------------------
# 70 / 30 split:
# 70% discovers a candidate condition.
# 30% only tests it.
# ------------------------------------------------------------

df["split"] = split_data(df)

calibration = df[
    df["split"] == "calibration"
].copy()

validation = df[
    df["split"] == "validation"
].copy()


audit = {
    "N": len(df),

    "unique_clips":
        int(df["clip"].nunique()),

    "task_counts":
        counts,

    "split_counts":
        df["split"]
        .value_counts()
        .to_dict(),

    "rms_msj_spearman":
        float(
            spearmanr(
                df["rms_acceleration"],
                df["mean_squared_jerk"]
            ).statistic
        ),
}

(
    OUT / "audit.json"
).write_text(
    json.dumps(
        audit,
        indent=2
    )
)


# ------------------------------------------------------------
# Primary Acc / Jerk analyses
# ------------------------------------------------------------

metrics = [
    "rms_acceleration",
    "mean_squared_jerk",
]

rows = []
thresholds = {}

for metric in metrics:

    fit = fit_models(
        calibration,
        metric
    )

    ci_low, ci_high, n_boot = (
        bootstrap_threshold(
            calibration,
            metric
        )
    )

    val = validate_threshold(
        validation,
        metric,
        fit["threshold_raw"]
    )

    row = {
        "metric":
            metric,

        **fit,

        "bootstrap_ci_low":
            ci_low,

        "bootstrap_ci_high":
            ci_high,

        "bootstrap_success_n":
            n_boot,

        **val,
    }

    rows.append(row)

    thresholds[metric] = (
        fit["threshold_raw"]
    )

    curve = binned_curve(
        df,
        metric,
        bins=20
    )

    curve.to_csv(
        OUT /
        f"{metric}_20bin_curve.csv",
        index=False
    )

    plot_main_curve(
        df,
        curve,
        metric,
        fit["threshold_raw"],
        OUT /
        f"{metric}_error_curve.png"
    )

    plot_taskwise(
        df,
        metric,
        fit["threshold_raw"],
        OUT /
        f"{metric}_taskwise_curve.png"
    )


summary = pd.DataFrame(rows)

summary["verdict"] = (
    summary.apply(
        verdict,
        axis=1
    )
)

summary.to_csv(
    OUT /
    "working_range_summary.csv",
    index=False
)


# ------------------------------------------------------------
# Taskwise:
# Do NOT force one global boundary if tasks disagree.
# ------------------------------------------------------------

task_rows = []

for task, g in df.groupby(
    "task",
    sort=True
):

    for metric in metrics:

        fit = fit_models(
            g,
            metric
        )

        rho, p = spearmanr(
            g[metric],
            g["l2_moved_mm"]
        )

        task_rows.append({
            "task": task,
            "metric": metric,
            "N": len(g),

            "rho": rho,
            "p": p,

            **fit,
        })

taskwise = pd.DataFrame(
    task_rows
)

taskwise.to_csv(
    OUT /
    "taskwise_shape_diagnostics.csv",
    index=False
)


# ------------------------------------------------------------
# Candidate combined condition
#
# IMPORTANT:
# This is NOT called a final violation rule unless
# the range evidence survives the diagnostics above.
# ------------------------------------------------------------

tau_acc = thresholds[
    "rms_acceleration"
]

tau_jerk = thresholds[
    "mean_squared_jerk"
]


def candidate_group(row):

    high_acc = (
        row["rms_acceleration"]
        > tau_acc
    )

    high_jerk = (
        row["mean_squared_jerk"]
        > tau_jerk
    )

    if high_acc and high_jerk:
        return "both"

    if high_acc:
        return "acc_only"

    if high_jerk:
        return "jerk_only"

    return "in_candidate_range"


df["candidate_group"] = (
    df.apply(
        candidate_group,
        axis=1
    )
)

group_rows = []

for split in [
    "calibration",
    "validation",
    "all"
]:

    g0 = (
        df
        if split == "all"
        else df[df["split"] == split]
    )

    for group, g in g0.groupby(
        "candidate_group"
    ):

        group_rows.append({
            "split":
                split,

            "group":
                group,

            "N":
                len(g),

            "pct":
                100 * len(g) / len(g0),

            "median_moved_l2_mm":
                float(
                    g["l2_moved_mm"]
                    .median()
                ),
        })


pd.DataFrame(
    group_rows
).to_csv(
    OUT /
    "candidate_range_group_summary.csv",
    index=False
)


# ------------------------------------------------------------
# Acc x Jerk visualization
# ------------------------------------------------------------

plot_joint_heatmap(df)


# ------------------------------------------------------------
# Other possible Action factors
# Orientation / gripper / path / speed etc.
# Exploratory only.
# ------------------------------------------------------------

screen_other_action_factors(df)


# ------------------------------------------------------------
# Terminal summary
# ------------------------------------------------------------

print("\n")
print("=" * 90)
print("AUDIT")
print("=" * 90)
print(json.dumps(audit, indent=2))


print("\n")
print("=" * 90)
print("PRIMARY WORKING-RANGE DIAGNOSTIC")
print("=" * 90)

show_cols = [
    "metric",

    "threshold_raw",

    "bootstrap_ci_low",
    "bootstrap_ci_high",

    "delta_bic_seg_vs_linear",
    "delta_bic_seg_vs_quadratic",

    "slope_pre",
    "slope_post",

    "val_n_in",
    "val_n_out",

    "val_med_in_mm",
    "val_med_out_mm",

    "val_out_in_ratio",
    "val_cliff_delta",
    "val_p",

    "verdict",
]

print(
    summary[
        show_cols
    ].to_string(index=False)
)


print("\n")
print("=" * 90)
print("TASKWISE DIAGNOSTIC")
print("=" * 90)

print(
    taskwise[
        [
            "task",
            "metric",
            "N",
            "rho",

            "threshold_raw",

            "delta_bic_seg_vs_linear",
            "delta_bic_seg_vs_quadratic",

            "slope_pre",
            "slope_post",
        ]
    ].to_string(index=False)
)


print("\n")
print("=" * 90)
print("VALIDATION CANDIDATE GROUPS")
print("=" * 90)

groups = pd.read_csv(
    OUT /
    "candidate_range_group_summary.csv"
)

print(
    groups[
        groups["split"]
        == "validation"
    ].to_string(index=False)
)


extra = (
    OUT /
    "exploratory_action_factor_screen.csv"
)

if extra.exists():

    factors = pd.read_csv(extra)

    print("\n")
    print("=" * 90)
    print("OTHER ACTION FACTORS — EXPLORATORY TOP 15")
    print("=" * 90)

    print(
        factors.head(15)
        .to_string(index=False)
    )


print("\n")
print("RESULT_DIR =", OUT)
print("SUCCESS = True")
