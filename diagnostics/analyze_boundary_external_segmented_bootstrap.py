from pathlib import Path
import numpy as np
import pandas as pd

B = 0.24580418
SEED = 42
N_BOOT = 2000
N_KNOT_BOOT = 500

ROOT = Path(
    "diagnostics/results/rigid_projection_repair_current10457"
)

REPAIR_CSV = ROOT / "combined_5task_repair_per_clip.csv"
ACC_CSV = Path(
    "diagnostics/results/current_behavior_established_acc_10457/"
    "per_sample_current_established_acc_10457.csv"
)

repair = pd.read_csv(REPAIR_CSV)
acc = pd.read_csv(ACC_CSV)

if "benefit_mm" not in repair.columns:
    repair["benefit_mm"] = (
        repair["raw_mp"] - repair["repaired_mp"]
    )

d = repair.merge(
    acc[
        ["task", "clip", "rms_acceleration_si_m_s2"]
    ],
    on=["task", "clip"],
    how="inner",
    validate="one_to_one",
)

assert len(d) == 7947, len(d)

d["z"] = (
    d["rms_acceleration_si_m_s2"] / B
)

TASKS = sorted(d["task"].unique())

# ---------------------------------------------------------
# Design:
# y = intercept + beta_pre*z
#     + beta_change*max(0, z-knot)
#     + task fixed effects
#
# pre slope  = beta_pre
# post slope = beta_pre + beta_change
# ---------------------------------------------------------

def design(df, knot=1.0):
    z = df["z"].to_numpy(float)

    cols = [
        np.ones(len(df)),
        z,
        np.maximum(0.0, z - knot),
    ]

    # task fixed effects, first task reference
    for task in TASKS[1:]:
        cols.append(
            (df["task"].to_numpy() == task).astype(float)
        )

    return np.column_stack(cols)


def fit_piecewise(df, metric, knot=1.0):
    X = design(df, knot)
    y = df[metric].to_numpy(float)

    beta, *_ = np.linalg.lstsq(
        X, y, rcond=None
    )

    pred = X @ beta
    sse = float(
        np.sum((y - pred) ** 2)
    )

    pre = float(beta[1])
    change = float(beta[2])
    post = pre + change

    return {
        "pre": pre,
        "post": post,
        "change": change,
        "sse": sse,
    }


def stratified_bootstrap(df, metric, knot, n_boot):
    rng = np.random.default_rng(SEED)

    by_task = {
        t: g.reset_index(drop=True)
        for t, g in df.groupby("task")
    }

    vals = []

    for _ in range(n_boot):
        pieces = []

        for task, g in by_task.items():
            idx = rng.integers(
                0, len(g), size=len(g)
            )
            pieces.append(
                g.iloc[idx]
            )

        b = pd.concat(
            pieces,
            ignore_index=True,
        )

        f = fit_piecewise(
            b, metric, knot
        )

        vals.append(
            [
                f["pre"],
                f["post"],
                f["change"],
            ]
        )

    return np.asarray(vals, float)


def ci(x):
    return (
        float(np.quantile(x, 0.025)),
        float(np.quantile(x, 0.975)),
    )


def boot_p_two_sided(x):
    pneg = np.mean(x <= 0)
    ppos = np.mean(x >= 0)

    return min(
        1.0,
        2.0 * min(pneg, ppos)
    )


# ---------------------------------------------------------
# PRIMARY: local region around established B
# Avoid high-Acc tail dominating the change-point test.
# ---------------------------------------------------------

local = d[
    (d["z"] >= 0.5)
    & (d["z"] <= 1.5)
].copy()

print("=" * 82)
print("ESTABLISHED BOUNDARY EXTERNAL VALIDATION")
print("=" * 82)
print("Fixed B =", B, "m/s^2")
print("Full N =", len(d))
print("Primary local window = [0.5B, 1.5B]")
print("Local N =", len(local))

for task, g in local.groupby("task"):
    print(task, "N =", len(g))


summary_rows = []

for metric in [
    "residual_mm",
    "benefit_mm",
]:
    print("\n" + "=" * 82)
    print("METRIC:", metric)
    print("=" * 82)

    f = fit_piecewise(
        local,
        metric,
        knot=1.0,
    )

    boot = stratified_bootstrap(
        local,
        metric,
        knot=1.0,
        n_boot=N_BOOT,
    )

    pre_ci = ci(boot[:, 0])
    post_ci = ci(boot[:, 1])
    change_ci = ci(boot[:, 2])
    pboot = boot_p_two_sided(
        boot[:, 2]
    )

    print(
        "Pre-B slope  = %.6f per 1B"
        % f["pre"]
    )
    print(
        "  95%% bootstrap CI = [%.6f, %.6f]"
        % pre_ci
    )

    print(
        "Post-B slope = %.6f per 1B"
        % f["post"]
    )
    print(
        "  95%% bootstrap CI = [%.6f, %.6f]"
        % post_ci
    )

    print(
        "Slope change = %.6f"
        % f["change"]
    )
    print(
        "  95%% bootstrap CI = [%.6f, %.6f]"
        % change_ci
    )
    print(
        "Bootstrap two-sided p(change=0) = %.6g"
        % pboot
    )

    print(
        "P(change > 0) = %.4f"
        % np.mean(boot[:, 2] > 0)
    )

    summary_rows.append({
        "metric": metric,
        "N": len(local),
        "pre_slope_per_B": f["pre"],
        "pre_ci_lo": pre_ci[0],
        "pre_ci_hi": pre_ci[1],
        "post_slope_per_B": f["post"],
        "post_ci_lo": post_ci[0],
        "post_ci_hi": post_ci[1],
        "slope_change": f["change"],
        "change_ci_lo": change_ci[0],
        "change_ci_hi": change_ci[1],
        "bootstrap_p": pboot,
        "prob_change_positive":
            np.mean(boot[:, 2] > 0),
    })


# ---------------------------------------------------------
# ROBUSTNESS: different local windows, same fixed B
# ---------------------------------------------------------

print("\n" + "=" * 82)
print("WINDOW ROBUSTNESS — FIXED B")
print("=" * 82)

window_rows = []

for lo, hi in [
    (0.4, 1.6),
    (0.5, 1.5),
    (0.5, 2.0),
    (0.8, 1.5),
]:
    w = d[
        (d["z"] >= lo)
        & (d["z"] <= hi)
    ]

    for metric in [
        "residual_mm",
        "benefit_mm",
    ]:
        f = fit_piecewise(
            w,
            metric,
            knot=1.0,
        )

        window_rows.append({
            "window": f"{lo:.1f}-{hi:.1f}B",
            "metric": metric,
            "N": len(w),
            "pre_slope": f["pre"],
            "post_slope": f["post"],
            "slope_change": f["change"],
        })

window_df = pd.DataFrame(
    window_rows
)

print(
    window_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ---------------------------------------------------------
# EXTERNAL KNOT PROFILE
#
# We are NOT redefining B.
# This asks whether residual/benefit independently prefer
# a hinge location near the already-established B.
# Search only local neighborhood [0.75B, 1.25B].
# ---------------------------------------------------------

print("\n" + "=" * 82)
print("EXTERNAL CHANGEPOINT PROFILE")
print("=" * 82)

knots = np.linspace(
    0.75,
    1.25,
    101,
)

knot_rows = []

for metric in [
    "residual_mm",
    "benefit_mm",
]:
    sses = np.array([
        fit_piecewise(
            local,
            metric,
            knot=float(k),
        )["sse"]
        for k in knots
    ])

    best_idx = int(
        np.argmin(sses)
    )

    best_k = float(
        knots[best_idx]
    )

    print(
        metric,
        "best external knot = %.3fB = %.6f m/s^2"
        % (
            best_k,
            best_k * B,
        )
    )

    # bootstrap knot localization
    rng = np.random.default_rng(
        SEED + (
            0 if metric == "residual_mm"
            else 10000
        )
    )

    by_task = {
        t: g.reset_index(drop=True)
        for t, g in local.groupby("task")
    }

    best_boot = []

    for _ in range(N_KNOT_BOOT):
        pieces = []

        for task, g in by_task.items():
            idx = rng.integers(
                0, len(g), size=len(g)
            )
            pieces.append(
                g.iloc[idx]
            )

        b = pd.concat(
            pieces,
            ignore_index=True,
        )

        boot_sses = np.array([
            fit_piecewise(
                b,
                metric,
                knot=float(k),
            )["sse"]
            for k in knots
        ])

        best_boot.append(
            float(
                knots[
                    int(np.argmin(boot_sses))
                ]
            )
        )

    best_boot = np.asarray(
        best_boot
    )

    k_ci = ci(best_boot)

    print(
        "  bootstrap knot 95%% CI = [%.3fB, %.3fB]"
        % k_ci
    )
    print(
        "  established B (=1.000B) inside CI:",
        bool(
            k_ci[0] <= 1.0 <= k_ci[1]
        )
    )

    knot_rows.append({
        "metric": metric,
        "best_knot_B_multiple": best_k,
        "best_knot_mps2": best_k * B,
        "knot_ci_lo_B": k_ci[0],
        "knot_ci_hi_B": k_ci[1],
        "established_B_in_CI":
            k_ci[0] <= 1.0 <= k_ci[1],
    })


# ---------------------------------------------------------
# TASKWISE direction at fixed B
# ---------------------------------------------------------

print("\n" + "=" * 82)
print("TASKWISE FIXED-B SLOPE CHANGE")
print("=" * 82)

task_rows = []

for task, g in local.groupby("task"):
    for metric in [
        "residual_mm",
        "benefit_mm",
    ]:
        # Single-task design: intercept + z + hinge
        z = g["z"].to_numpy(float)
        X = np.column_stack([
            np.ones(len(g)),
            z,
            np.maximum(0.0, z - 1.0),
        ])

        y = g[metric].to_numpy(float)

        beta, *_ = np.linalg.lstsq(
            X, y, rcond=None
        )

        pre = float(beta[1])
        change = float(beta[2])
        post = pre + change

        task_rows.append({
            "task": task,
            "metric": metric,
            "N": len(g),
            "pre_slope": pre,
            "post_slope": post,
            "slope_change": change,
        })

task_df = pd.DataFrame(
    task_rows
)

print(
    task_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

# save
pd.DataFrame(
    summary_rows
).to_csv(
    ROOT / "boundary_segmented_bootstrap_summary.csv",
    index=False,
)

window_df.to_csv(
    ROOT / "boundary_segmented_window_robustness.csv",
    index=False,
)

pd.DataFrame(
    knot_rows
).to_csv(
    ROOT / "boundary_external_knot_profile.csv",
    index=False,
)

task_df.to_csv(
    ROOT / "boundary_segmented_taskwise.csv",
    index=False,
)

print("\nSAVED =", ROOT)
