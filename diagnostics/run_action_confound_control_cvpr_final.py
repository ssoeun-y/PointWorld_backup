#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, wilcoxon


METRIC = Path(
    "diagnostics/results/established_action_metrics_5task/"
    "per_sample_established_metrics_7987.csv"
)

GT = Path(
    "diagnostics/results/"
    "gt_motion_exact_5task_moved_7987.csv"
)

MOTION = Path(
    "diagnostics/results/action_confound_control_5task/"
    "per_sample_covariates_7987.csv"
)

INTERACTION = Path(
    "diagnostics/results/action_confound_control_5task/"
    "task0010_interaction_features_7987.csv"
)

OUTDIR = Path(
    "diagnostics/results/"
    "action_confound_control_cvpr_final"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N = 7987

EXPECTED_TASKS = {
    "task-0000": 377,
    "task-0001": 2322,
    "task-0006": 1782,
    "task-0008": 1827,
    "task-0010": 1679,
}

METRICS = [
    "rms_acceleration",
    "mean_squared_jerk",
]

# Controls chosen BEFORE looking at outcome.
# All represent task/interaction/motion difficulty;
# no prediction-error quantity is used for matching.
CONTROLS = [
    "gt_motion_mm",
    "active_path_length_m",
    "active_net_displacement_m",
    "dominant_gripper_range",
    "dominant_rotation_rad",
    "arm_balance",
    "base_path_m",
]

CALIPERS = [0.05, 0.10]
BOOT_N = 10000
SEED = 42


def rank01(s):
    return s.rank(
        method="average",
        pct=True,
    )


def residualize(y, X):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    X = np.column_stack([
        np.ones(len(X)),
        X,
    ])

    beta, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    return y - X @ beta


def pooled_smd(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)

    denom = np.sqrt(
        (va + vb) / 2.0
    )

    if denom <= 1e-12:
        return 0.0

    return float(
        (np.mean(a) - np.mean(b))
        / denom
    )


def bootstrap_median_ci(x, n_boot=BOOT_N):
    x = np.asarray(x, dtype=float)

    rng = np.random.default_rng(SEED)

    vals = np.empty(
        n_boot,
        dtype=float,
    )

    for i in range(n_boot):
        idx = rng.integers(
            0,
            len(x),
            size=len(x),
        )

        vals[i] = np.median(
            x[idx]
        )

    return (
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    )


def load_population():

    m = pd.read_csv(METRIC)

    assert len(m) == EXPECTED_N
    assert m["clip"].is_unique
    assert (
        m.groupby("task")
        .size()
        .to_dict()
        == EXPECTED_TASKS
    )

    gt = pd.read_csv(GT)[
        ["clip", "task", "gt_motion_mm"]
    ].copy()

    assert len(gt) == EXPECTED_N
    assert gt["clip"].is_unique

    motion = pd.read_csv(MOTION)[
        [
            "clip",
            "task_loaded",
            "active_path_length_m",
            "active_net_displacement_m",
        ]
    ].copy()

    interaction = pd.read_csv(INTERACTION)[
        [
            "clip",
            "task_loaded",
            "dominant_gripper_range",
            "dominant_rotation_rad",
            "arm_balance",
            "base_path_m",
        ]
    ].copy()

    motion = motion.rename(
        columns={
            "task_loaded":
            "task_motion"
        }
    )

    interaction = interaction.rename(
        columns={
            "task_loaded":
            "task_interaction"
        }
    )

    df = (
        m.merge(
            gt,
            on=["clip", "task"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            motion,
            on="clip",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            interaction,
            on="clip",
            how="inner",
            validate="one_to_one",
        )
    )

    assert len(df) == EXPECTED_N

    assert (
        df["task"]
        == df["task_motion"]
    ).all()

    assert (
        df["task"]
        == df["task_interaction"]
    ).all()

    df = df.drop(
        columns=[
            "task_motion",
            "task_interaction",
        ]
    )

    needed = (
        ["clip", "task", "l2_moved_mm"]
        + METRICS
        + CONTROLS
    )

    assert df[needed].notna().all().all()

    print("POPULATION AUDIT PASSED")
    print("N =", len(df))
    print(df.groupby("task").size())

    return df


def add_within_task_ranks(df):

    df = df.copy()

    rank_cols = (
        METRICS
        + ["l2_moved_mm"]
        + CONTROLS
    )

    for c in rank_cols:
        df[c + "__rank"] = (
            df.groupby("task")[c]
            .rank(
                method="average",
                pct=True,
            )
        )

    return df


def optimal_match_one_task(
    g,
    metric,
    caliper,
):

    lo_thr = g[metric].quantile(0.25)
    hi_thr = g[metric].quantile(0.75)

    low = (
        g[g[metric] <= lo_thr]
        .copy()
        .reset_index(drop=True)
    )

    high = (
        g[g[metric] >= hi_thr]
        .copy()
        .reset_index(drop=True)
    )

    rank_controls = [
        c + "__rank"
        for c in CONTROLS
    ]

    H = high[
        rank_controls
    ].to_numpy(float)

    L = low[
        rank_controls
    ].to_numpy(float)

    # Pairwise absolute rank differences.
    dif = np.abs(
        H[:, None, :]
        - L[None, :, :]
    )

    # Hard per-covariate caliper.
    feasible = (
        dif.max(axis=2)
        <= caliper
    )

    # Outcome-blind cost:
    # only control-variable distances.
    cost = np.sqrt(
        np.square(dif).sum(axis=2)
    )

    BIG = 1e6
    cost[~feasible] = BIG

    r, c = linear_sum_assignment(
        cost
    )

    keep = (
        cost[r, c] < BIG
    )

    r = r[keep]
    c = c[keep]

    rows = []

    for hi, li in zip(r, c):

        hr = high.iloc[hi]
        lr = low.iloc[li]

        row = {
            "task": hr["task"],
            "metric": metric,
            "caliper": caliper,

            "high_clip": hr["clip"],
            "low_clip": lr["clip"],

            "high_metric": hr[metric],
            "low_metric": lr[metric],

            "high_l2_moved_mm":
                hr["l2_moved_mm"],

            "low_l2_moved_mm":
                lr["l2_moved_mm"],
        }

        for cc in CONTROLS:
            row[
                "high_" + cc
            ] = hr[cc]

            row[
                "low_" + cc
            ] = lr[cc]

            row[
                "high_" + cc + "__rank"
            ] = hr[cc + "__rank"]

            row[
                "low_" + cc + "__rank"
            ] = lr[cc + "__rank"]

        rows.append(row)

    return pd.DataFrame(rows)


def match_all_tasks(
    df,
    metric,
    caliper,
):

    parts = []

    for task, g in df.groupby("task"):

        p = optimal_match_one_task(
            g,
            metric,
            caliper,
        )

        if len(p):
            parts.append(p)

    if not parts:
        return pd.DataFrame()

    return pd.concat(
        parts,
        ignore_index=True,
    )


def summarize_balance(pairs):

    rows = []

    for c in CONTROLS:

        h = pairs[
            "high_" + c + "__rank"
        ].to_numpy(float)

        l = pairs[
            "low_" + c + "__rank"
        ].to_numpy(float)

        rows.append({
            "covariate": c,
            "rank_smd": pooled_smd(
                h,
                l,
            ),
            "abs_rank_smd": abs(
                pooled_smd(h, l)
            ),
            "median_abs_rank_gap":
                float(
                    np.median(
                        np.abs(h - l)
                    )
                ),
        })

    return pd.DataFrame(rows)


def summarize_pairs(
    pairs,
    metric,
    caliper,
):

    high = pairs[
        "high_l2_moved_mm"
    ].to_numpy(float)

    low = pairs[
        "low_l2_moved_mm"
    ].to_numpy(float)

    delta = high - low

    paired_ratio = high / np.maximum(
        low,
        1e-12,
    )

    ci_lo, ci_hi = (
        bootstrap_median_ci(
            delta
        )
    )

    if np.allclose(delta, 0):
        p = 1.0
    else:
        p = float(
            wilcoxon(delta).pvalue
        )

    bal = summarize_balance(
        pairs
    )

    return {
        "metric": metric,
        "caliper": caliper,
        "N_pairs": len(pairs),

        "low_median_l2_mm":
            float(np.median(low)),

        "high_median_l2_mm":
            float(np.median(high)),

        "median_paired_delta_mm":
            float(np.median(delta)),

        "delta_ci95_low_mm":
            ci_lo,

        "delta_ci95_high_mm":
            ci_hi,

        "ratio_of_medians":
            float(
                np.median(high)
                / np.median(low)
            ),

        "median_paired_ratio":
            float(
                np.median(
                    paired_ratio
                )
            ),

        "high_worse_fraction":
            float(
                np.mean(delta > 0)
            ),

        "wilcoxon_p":
            p,

        "max_abs_rank_smd":
            float(
                bal[
                    "abs_rank_smd"
                ].max()
            ),

        "median_abs_rank_gap_max":
            float(
                bal[
                    "median_abs_rank_gap"
                ].max()
            ),
    }


def partial_rank_independent(
    df,
    metric,
):

    # IMPORTANT:
    # Do NOT control the other dynamics metric.
    # RMS/MSJ are strongly collinear and are
    # treated as independent physical axes.

    needed = [
        metric,
        "l2_moved_mm",
    ] + CONTROLS

    parts = []

    for _, g in df.groupby("task"):
        g = g[needed].copy()

        for c in needed:
            g[c] = rank01(
                g[c]
            )

        parts.append(g)

    d = pd.concat(
        parts,
        ignore_index=True,
    )

    X = d[
        CONTROLS
    ].to_numpy(float)

    rx = residualize(
        d[metric],
        X,
    )

    ry = residualize(
        d["l2_moved_mm"],
        X,
    )

    rho, p = pearsonr(
        rx,
        ry,
    )

    return {
        "scope":
            "ALL_WITHIN_TASK",

        "metric":
            metric,

        "N":
            len(d),

        "partial_rank_rho":
            float(rho),

        "p":
            float(p),
    }


def taskwise_partial(
    df,
    metric,
):

    rows = []

    for task, g in df.groupby("task"):

        needed = [
            metric,
            "l2_moved_mm",
        ] + CONTROLS

        d = g[needed].copy()

        for c in needed:
            d[c] = rank01(
                d[c]
            )

        X = d[
            CONTROLS
        ].to_numpy(float)

        rx = residualize(
            d[metric],
            X,
        )

        ry = residualize(
            d["l2_moved_mm"],
            X,
        )

        rho, p = pearsonr(
            rx,
            ry,
        )

        rows.append({
            "scope": task,
            "metric": metric,
            "N": len(d),
            "partial_rank_rho":
                float(rho),
            "p":
                float(p),
        })

    return rows


def main():

    df = load_population()
    df = add_within_task_ranks(df)

    summary_rows = []
    partial_rows = []

    for metric in METRICS:

        print()
        print("=" * 100)
        print("METRIC =", metric)
        print("=" * 100)

        for caliper in CALIPERS:

            pairs = match_all_tasks(
                df,
                metric,
                caliper,
            )

            if len(pairs) == 0:
                raise RuntimeError(
                    f"No pairs: "
                    f"{metric}, {caliper}"
                )

            tag = (
                f"{metric}"
                f"__caliper"
                f"{int(caliper*100):02d}"
            )

            pairs.to_csv(
                OUTDIR /
                f"matched_pairs__{tag}.csv",
                index=False,
            )

            bal = summarize_balance(
                pairs
            )

            bal.to_csv(
                OUTDIR /
                f"balance__{tag}.csv",
                index=False,
            )

            s = summarize_pairs(
                pairs,
                metric,
                caliper,
            )

            summary_rows.append(s)

            print()
            print(
                f"CALIPER = {caliper}"
            )
            print(
                pd.Series(s).to_string()
            )

            print()
            print("BALANCE")
            print(
                bal.to_string(
                    index=False
                )
            )

        partial_rows.append(
            partial_rank_independent(
                df,
                metric,
            )
        )

        partial_rows.extend(
            taskwise_partial(
                df,
                metric,
            )
        )

    summary = pd.DataFrame(
        summary_rows
    )

    partial = pd.DataFrame(
        partial_rows
    )

    summary.to_csv(
        OUTDIR /
        "matched_effect_summary.csv",
        index=False,
    )

    partial.to_csv(
        OUTDIR /
        "partial_rank_summary.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("FINAL MATCHED EFFECT SUMMARY")
    print("=" * 100)
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("INDEPENDENT PARTIAL-RANK SUMMARY")
    print("=" * 100)
    print(
        partial.to_string(
            index=False
        )
    )

    print()
    print("OUTPUT =", OUTDIR.resolve())
    print("SUCCESS = True")


if __name__ == "__main__":
    main()
