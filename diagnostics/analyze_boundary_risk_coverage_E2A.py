#!/usr/bin/env python3

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260902

ACC_BOUNDARY = 0.009832167363499603
MSJ_BOUNDARY = 8.152857428804615e-05

ROOT = Path(__file__).resolve().parents[1]

IN = (
    ROOT /
    "diagnostics/results/established_action_metrics_5task/"
    "per_sample_established_metrics_7987.csv"
)

OUTDIR = (
    ROOT /
    "diagnostics/results/"
    "risk_calibrated_envelope_E2A"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


# ==============================================================
# Exact existing task-controlled 70/30 split
# ==============================================================

def stable_number(text):
    h = hashlib.sha256(
        f"{SEED}|{text}".encode()
    ).digest()

    return int.from_bytes(
        h[:8], "big"
    ) / 2**64


def exact_split(df):
    out = pd.Series(
        index=df.index,
        dtype="object"
    )

    for task, g in df.groupby("task"):

        idx = sorted(
            g.index,
            key=lambda i:
                stable_number(
                    str(df.at[i, "clip"])
                )
        )

        n_cal = round(
            0.70 * len(idx)
        )

        out.loc[idx[:n_cal]] = "calibration"
        out.loc[idx[n_cal:]] = "validation"

    return out


# ==============================================================
# Basic statistics
# ==============================================================

def quant(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    return float(
        np.quantile(x, q)
    )


def evaluate_subset(df):
    n = len(df)

    if n == 0:
        return {
            "n": 0,
            "risk": np.nan,
            "median_error_mm": np.nan,
            "p90_error_mm": np.nan,
            "p95_error_mm": np.nan,
        }

    err = df[
        "l2_moved_mm"
    ].to_numpy(float)

    return {
        "n": n,
        "risk":
            float(
                df["high_risk"].mean()
            ),
        "median_error_mm":
            float(np.median(err)),
        "p90_error_mm":
            quant(err, 0.90),
        "p95_error_mm":
            quant(err, 0.95),
    }


# ==============================================================
# Load
# ==============================================================

d = pd.read_csv(IN)

assert len(d) == 7987
assert d["clip"].nunique() == 7987

required = [
    "clip",
    "task",
    "rms_acceleration",
    "mean_squared_jerk",
    "l2_moved_mm",
]

missing = [
    c for c in required
    if c not in d.columns
]

if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )

d["split"] = exact_split(d)

cal = d[
    d["split"] == "calibration"
].copy()

val = d[
    d["split"] == "validation"
].copy()

assert len(cal) == 5590
assert len(val) == 2397


# ==============================================================
# Prediction-risk definition
#
# Per-task high-error threshold estimated ONLY on calibration.
# High risk = moved-L2 > task calibration Q75.
# ==============================================================

risk_thresholds = (
    cal.groupby("task")[
        "l2_moved_mm"
    ]
    .quantile(0.75)
    .to_dict()
)

for x in [cal, val]:
    x["high_risk"] = np.asarray(
        [
            err > risk_thresholds[task]
            for task, err in zip(
                x["task"],
                x["l2_moved_mm"]
            )
        ],
        dtype=int
    )


# ==============================================================
# Existing global boundary severity
#
# score = 1 exactly at envelope boundary.
# score <= 1 = existing reliable side.
# ==============================================================

for x in [cal, val]:

    x["acc_severity"] = (
        x["rms_acceleration"]
        / ACC_BOUNDARY
    )

    x["msj_severity"] = (
        x["mean_squared_jerk"]
        / MSJ_BOUNDARY
    )

    x["boundary_severity"] = np.maximum(
        x["acc_severity"],
        x["msj_severity"]
    )


# ==============================================================
# 1. Existing fixed boundary itself
# ==============================================================

print("=" * 100)
print("E2-A — EXISTING FIXED BOUNDARY")
print("=" * 100)

for name, x in [
    ("CAL", cal),
    ("VAL", val),
]:

    inside = x[
        x["boundary_severity"] <= 1.0
    ]

    outside = x[
        x["boundary_severity"] > 1.0
    ]

    si = evaluate_subset(inside)
    so = evaluate_subset(outside)

    print()
    print(name)
    print(
        f"INSIDE : N={si['n']:4d} | "
        f"coverage={si['n']/len(x):.4f} | "
        f"high-risk={si['risk']:.4f} | "
        f"median={si['median_error_mm']:.4f} mm | "
        f"p95={si['p95_error_mm']:.4f} mm"
    )

    print(
        f"OUTSIDE: N={so['n']:4d} | "
        f"coverage={so['n']/len(x):.4f} | "
        f"high-risk={so['risk']:.4f} | "
        f"median={so['median_error_mm']:.4f} mm | "
        f"p95={so['p95_error_mm']:.4f} mm"
    )


# ==============================================================
# 2. Risk–coverage curve
#
# Thresholds are determined ONLY from calibration score
# quantiles, then frozen and evaluated on validation.
#
# Lower boundary_severity = predicted safer.
# ==============================================================

COVERAGE_TARGETS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

rows = []

for target_cov in COVERAGE_TARGETS:

    if target_cov < 1.0:
        threshold = float(
            np.quantile(
                cal["boundary_severity"],
                target_cov
            )
        )
    else:
        threshold = np.inf

    for split_name, x in [
        ("calibration", cal),
        ("validation", val),
    ]:

        accepted = x[
            x["boundary_severity"]
            <= threshold
        ]

        s = evaluate_subset(
            accepted
        )

        rows.append({
            "target_coverage":
                target_cov,

            "score_threshold":
                threshold,

            "split":
                split_name,

            "total_n":
                len(x),

            "accepted_n":
                s["n"],

            "actual_coverage":
                s["n"] / len(x),

            "high_risk_rate":
                s["risk"],

            "median_error_mm":
                s["median_error_mm"],

            "p90_error_mm":
                s["p90_error_mm"],

            "p95_error_mm":
                s["p95_error_mm"],
        })


curve = pd.DataFrame(rows)

curve.to_csv(
    OUTDIR /
    "E2A_boundary_risk_coverage.csv",
    index=False
)


# ==============================================================
# 3. Task-wise validation curve
# ==============================================================

task_rows = []

for target_cov in COVERAGE_TARGETS:

    if target_cov < 1.0:
        threshold = float(
            np.quantile(
                cal["boundary_severity"],
                target_cov
            )
        )
    else:
        threshold = np.inf

    for task, g in val.groupby(
        "task"
    ):

        accepted = g[
            g["boundary_severity"]
            <= threshold
        ]

        s = evaluate_subset(
            accepted
        )

        task_rows.append({
            "target_coverage":
                target_cov,

            "score_threshold":
                threshold,

            "task":
                task,

            "task_total_n":
                len(g),

            "accepted_n":
                s["n"],

            "actual_coverage":
                s["n"] / len(g),

            "high_risk_rate":
                s["risk"],

            "median_error_mm":
                s["median_error_mm"],

            "p95_error_mm":
                s["p95_error_mm"],
        })


task_curve = pd.DataFrame(
    task_rows
)

task_curve.to_csv(
    OUTDIR /
    "E2A_boundary_risk_coverage_by_task.csv",
    index=False
)


# ==============================================================
# Print held-out curve
# ==============================================================

v = curve[
    curve["split"]
    == "validation"
].copy()

print()
print("=" * 100)
print("E2-A — FROZEN VALIDATION RISK–COVERAGE")
print("=" * 100)

print(
    v[
        [
            "target_coverage",
            "actual_coverage",
            "score_threshold",
            "accepted_n",
            "high_risk_rate",
            "median_error_mm",
            "p95_error_mm",
        ]
    ].to_string(
        index=False,
        float_format=lambda z:
            f"{z:.6f}"
    )
)


# ==============================================================
# Monotonicity sanity
#
# As coverage increases, accepted set includes progressively
# riskier actions, so risk should broadly rise.
# ==============================================================

risk = v[
    "high_risk_rate"
].to_numpy(float)

finite = np.isfinite(risk)

if finite.sum() >= 2:
    diffs = np.diff(
        risk[finite]
    )

    nondecreasing_steps = (
        diffs >= -1e-12
    ).mean()

else:
    nondecreasing_steps = np.nan


full_val_risk = float(
    val["high_risk"].mean()
)

risk_50 = float(
    v.loc[
        np.isclose(
            v["target_coverage"],
            0.50
        ),
        "high_risk_rate"
    ].iloc[0]
)

risk_80 = float(
    v.loc[
        np.isclose(
            v["target_coverage"],
            0.80
        ),
        "high_risk_rate"
    ].iloc[0]
)


print()
print("=" * 100)
print("E2-A — SUMMARY")
print("=" * 100)

print(
    "Full validation high-risk rate =",
    f"{full_val_risk:.6f}"
)

print(
    "50% target coverage risk       =",
    f"{risk_50:.6f}"
)

print(
    "80% target coverage risk       =",
    f"{risk_80:.6f}"
)

print(
    "Risk reduction @50% coverage   =",
    f"{100*(1-risk_50/full_val_risk):.2f}%"
)

print(
    "Risk reduction @80% coverage   =",
    f"{100*(1-risk_80/full_val_risk):.2f}%"
)

print(
    "Nondecreasing risk steps       =",
    f"{nondecreasing_steps:.3f}"
)

print()
print(
    "NOTE: This is method-development evidence, "
    "NOT a final conformal guarantee."
)

print(
    "Final guarantee requires a frozen score and "
    "independent/nested risk calibration."
)

print()
print("OUTPUT =", OUTDIR)
print("SUCCESS = True")
