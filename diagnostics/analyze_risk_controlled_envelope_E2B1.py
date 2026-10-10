#!/usr/bin/env python3

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

SEED = 20260902
INNER_SALT = "E2B1-risk-calibration-v1"

ACC_BOUNDARY = 0.009832167363499603
MSJ_BOUNDARY = 8.152857428804615e-05

# User-selectable risk tolerances.
# These are NOT tuned to the result; report the full grid.
ALPHAS = [0.15, 0.20, 0.25]

# Family-wise confidence for threshold selection.
DELTA = 0.05

# Fixed finite candidate family.
COVERAGE_GRID = np.arange(0.10, 1.00, 0.05).tolist() + [1.00]

ROOT = Path(__file__).resolve().parents[1]

IN = (
    ROOT
    / "diagnostics/results/established_action_metrics_5task/"
      "per_sample_established_metrics_7987.csv"
)

OUTDIR = (
    ROOT
    / "diagnostics/results/"
      "risk_controlled_envelope_E2B1"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Existing outer 70/30 split -- reproduce EXACTLY.
# ============================================================

def stable_number(text, salt=""):
    h = hashlib.sha256(
        f"{SEED}|{salt}|{text}".encode()
    ).digest()

    return int.from_bytes(h[:8], "big") / 2**64


def outer_split(df):
    out = pd.Series(index=df.index, dtype="object")

    for task, g in df.groupby("task"):
        idx = sorted(
            g.index,
            key=lambda i:
                stable_number(str(df.at[i, "clip"]))
        )

        n_cal = round(0.70 * len(idx))

        out.loc[idx[:n_cal]] = "calibration"
        out.loc[idx[n_cal:]] = "validation"

    return out


# ============================================================
# Split ONLY outer calibration:
# proposal 70% / risk-calibration 30%, stratified by task.
#
# proposal:
#   - defines high-error threshold
#   - defines candidate score thresholds
#
# risk-calibration:
#   - estimates risk upper bounds
#   - selects threshold
# ============================================================

def inner_split(cal):
    out = pd.Series(index=cal.index, dtype="object")

    for task, g in cal.groupby("task"):
        idx = sorted(
            g.index,
            key=lambda i:
                stable_number(
                    str(cal.at[i, "clip"]),
                    salt=INNER_SALT
                )
        )

        n_prop = round(0.70 * len(idx))

        out.loc[idx[:n_prop]] = "proposal"
        out.loc[idx[n_prop:]] = "risk_calibration"

    return out


# ============================================================
# One-sided exact Clopper-Pearson upper bound.
# ============================================================

def cp_upper(k, n, tail_prob):
    if n <= 0:
        return np.nan

    if k >= n:
        return 1.0

    return float(
        beta.ppf(
            1.0 - tail_prob,
            k + 1,
            n - k
        )
    )


def subset_stats(df):
    if len(df) == 0:
        return {
            "n": 0,
            "risk": np.nan,
            "median_mm": np.nan,
            "p95_mm": np.nan,
        }

    err = df["l2_moved_mm"].to_numpy(float)

    return {
        "n": int(len(df)),
        "risk": float(df["high_risk"].mean()),
        "median_mm": float(np.median(err)),
        "p95_mm": float(np.quantile(err, 0.95)),
    }


# ============================================================
# Load + exact outer split.
# ============================================================

d = pd.read_csv(IN)

required = [
    "clip",
    "task",
    "rms_acceleration",
    "mean_squared_jerk",
    "l2_moved_mm",
]

missing = [c for c in required if c not in d.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

assert len(d) == 7987
assert d["clip"].nunique() == 7987

d["outer_split"] = outer_split(d)

cal = d[d["outer_split"] == "calibration"].copy()
val = d[d["outer_split"] == "validation"].copy()

assert len(cal) == 5590
assert len(val) == 2397

cal["inner_split"] = inner_split(cal)

prop = cal[cal["inner_split"] == "proposal"].copy()
rcal = cal[cal["inner_split"] == "risk_calibration"].copy()

assert len(prop) + len(rcal) == 5590


# ============================================================
# High-risk definition:
# task-specific proposal Q75 ONLY.
# ============================================================

risk_threshold = (
    prop.groupby("task")["l2_moved_mm"]
    .quantile(0.75)
    .to_dict()
)

for x in [prop, rcal, val]:
    x["high_risk"] = np.array(
        [
            int(err > risk_threshold[task])
            for task, err in zip(
                x["task"],
                x["l2_moved_mm"]
            )
        ],
        dtype=int
    )


# ============================================================
# Existing global boundary severity.
# ============================================================

for x in [prop, rcal, val]:
    x["boundary_severity"] = np.maximum(
        x["rms_acceleration"] / ACC_BOUNDARY,
        x["mean_squared_jerk"] / MSJ_BOUNDARY
    )


# ============================================================
# Candidate thresholds fixed from PROPOSAL only.
# ============================================================

candidates = []

for cov in COVERAGE_GRID:
    if cov < 1.0:
        lam = float(
            np.quantile(
                prop["boundary_severity"],
                cov
            )
        )
    else:
        lam = np.inf

    candidates.append((float(cov), lam))

M = len(candidates)

# Bonferroni -> simultaneous 95% family-wise confidence
# over the fixed candidate family.
per_candidate_tail = DELTA / M


# ============================================================
# Risk-calibration evaluation for every candidate.
# ============================================================

rows = []

for target_cov, lam in candidates:

    a = rcal[
        rcal["boundary_severity"] <= lam
    ].copy()

    n = len(a)
    k = int(a["high_risk"].sum())

    empirical_risk = (
        k / n if n > 0 else np.nan
    )

    upper = cp_upper(
        k,
        n,
        per_candidate_tail
    )

    rows.append({
        "target_proposal_coverage": target_cov,
        "lambda": lam,
        "riskcal_total_n": len(rcal),
        "riskcal_accepted_n": n,
        "riskcal_coverage":
            n / len(rcal),
        "riskcal_highrisk_count": k,
        "riskcal_empirical_risk":
            empirical_risk,
        "riskcal_simultaneous_upper":
            upper,
    })

cand = pd.DataFrame(rows)

cand.to_csv(
    OUTDIR / "E2B1_candidate_risk_bounds.csv",
    index=False
)


# ============================================================
# For each alpha:
# choose LARGEST lambda certified by risk-calibration UCB.
#
# Then report development validation result.
# ============================================================

selected_rows = []
task_rows = []

for alpha in ALPHAS:

    feasible = cand[
        cand["riskcal_simultaneous_upper"] <= alpha
    ].copy()

    if len(feasible) == 0:
        print()
        print(
            f"alpha={alpha:.2f}: "
            "NO CERTIFIED CANDIDATE"
        )

        selected_rows.append({
            "alpha": alpha,
            "status": "no_candidate",
        })

        continue

    # Largest acceptance set = largest lambda.
    chosen = feasible.sort_values(
        "lambda"
    ).iloc[-1]

    lam = float(chosen["lambda"])

    accepted = val[
        val["boundary_severity"] <= lam
    ].copy()

    vs = subset_stats(accepted)

    full_risk = float(
        val["high_risk"].mean()
    )

    reduction = (
        100.0
        * (1.0 - vs["risk"] / full_risk)
        if full_risk > 0
        else np.nan
    )

    selected_rows.append({
        "alpha": alpha,
        "status": "selected",
        "lambda": lam,

        "riskcal_coverage":
            chosen["riskcal_coverage"],

        "riskcal_empirical_risk":
            chosen["riskcal_empirical_risk"],

        "riskcal_simultaneous_upper":
            chosen["riskcal_simultaneous_upper"],

        "validation_total_n":
            len(val),

        "validation_accepted_n":
            vs["n"],

        "validation_coverage":
            vs["n"] / len(val),

        "validation_empirical_risk":
            vs["risk"],

        "validation_full_risk":
            full_risk,

        "validation_risk_reduction_pct":
            reduction,

        "validation_median_error_mm":
            vs["median_mm"],

        "validation_p95_error_mm":
            vs["p95_mm"],
    })

    for task, g in val.groupby("task"):

        ag = g[
            g["boundary_severity"] <= lam
        ]

        s = subset_stats(ag)

        full_task_risk = float(
            g["high_risk"].mean()
        )

        task_reduction = (
            100.0
            * (
                1.0
                - s["risk"] / full_task_risk
            )
            if (
                full_task_risk > 0
                and np.isfinite(s["risk"])
            )
            else np.nan
        )

        task_rows.append({
            "alpha": alpha,
            "lambda": lam,
            "task": task,
            "task_total_n": len(g),
            "accepted_n": s["n"],
            "coverage":
                s["n"] / len(g),
            "empirical_risk":
                s["risk"],
            "full_task_risk":
                full_task_risk,
            "risk_reduction_pct":
                task_reduction,
            "median_error_mm":
                s["median_mm"],
            "p95_error_mm":
                s["p95_mm"],
        })


selected = pd.DataFrame(selected_rows)
taskdf = pd.DataFrame(task_rows)

selected.to_csv(
    OUTDIR / "E2B1_selected_operating_points.csv",
    index=False
)

taskdf.to_csv(
    OUTDIR / "E2B1_selected_operating_points_by_task.csv",
    index=False
)


# ============================================================
# Print.
# ============================================================

print("=" * 110)
print("E2-B1 — RISK-CONTROLLED ENVELOPE FEASIBILITY")
print("=" * 110)

print("Total data             =", len(d))
print("Outer calibration      =", len(cal))
print("  proposal             =", len(prop))
print("  risk calibration     =", len(rcal))
print("Development validation =", len(val))

print()
print("Risk thresholds from PROPOSAL only:")
for task in sorted(risk_threshold):
    print(
        f"  {task}: "
        f"{risk_threshold[task]:.6f} mm"
    )

print()
print(
    f"Candidate family M={M}, "
    f"FWER delta={DELTA}, "
    f"per-candidate tail={per_candidate_tail:.8f}"
)

print()
print("=" * 110)
print("SELECTED OPERATING POINTS")
print("=" * 110)

cols = [
    "alpha",
    "status",
    "lambda",
    "riskcal_coverage",
    "riskcal_empirical_risk",
    "riskcal_simultaneous_upper",
    "validation_coverage",
    "validation_empirical_risk",
    "validation_risk_reduction_pct",
    "validation_median_error_mm",
    "validation_p95_error_mm",
]

present = [
    c for c in cols
    if c in selected.columns
]

print(
    selected[present].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

if len(taskdf):
    print()
    print("=" * 110)
    print("TASK-WISE DEVELOPMENT VALIDATION")
    print("=" * 110)

    print(
        taskdf[
            [
                "alpha",
                "task",
                "coverage",
                "empirical_risk",
                "full_task_risk",
                "risk_reduction_pct",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

print()
print("=" * 110)
print("INTERPRETATION")
print("=" * 110)
print(
    "This tests whether the existing boundary score can "
    "support a statistically risk-controlled operating point."
)
print(
    "Because the current global boundary constants were "
    "originally developed using the outer calibration set, "
    "this is FEASIBILITY / METHOD-DEVELOPMENT evidence only."
)
print(
    "A final guarantee requires nested/cross-fitted boundary "
    "fitting + independent risk calibration after the method "
    "is frozen."
)
print()
print("OUTPUT =", OUTDIR)
print("SUCCESS = True")
print("=" * 110)
