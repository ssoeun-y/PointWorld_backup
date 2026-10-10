#!/usr/bin/env python3

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

SEED = 20260902

TA = 0.009832167363499603
TJ = 8.152857428804615e-05

ROOT = Path(__file__).resolve().parents[1]

IN = (
    ROOT /
    "diagnostics/results/established_action_metrics_5task/"
    "per_sample_established_metrics_7987.csv"
)

OUTDIR = (
    ROOT /
    "diagnostics/results/"
    "joint_reliable_action_envelope_E1"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


def stable_number(text):
    h = hashlib.sha256(
        f"{SEED}|{text}".encode()
    ).digest()

    return int.from_bytes(
        h[:8], "big"
    ) / 2**64


def exact_split(df):
    split = pd.Series(
        index=df.index,
        dtype="object",
    )

    for task, g in df.groupby("task"):

        idx = sorted(
            g.index,
            key=lambda i:
                stable_number(
                    str(df.at[i, "clip"])
                ),
        )

        n_cal = round(
            len(idx) * 0.70
        )

        split.loc[
            idx[:n_cal]
        ] = "calibration"

        split.loc[
            idx[n_cal:]
        ] = "validation"

    return split


def safe_log10(x):
    x = np.asarray(
        x,
        dtype=float,
    )

    positive = x[
        np.isfinite(x)
        & (x > 0)
    ]

    floor = (
        positive.min() * 0.5
    )

    return np.log10(
        np.maximum(
            x,
            floor,
        )
    )


def build_features(
    df,
    kind,
    mu=None,
    sd=None,
):

    a = safe_log10(
        df["rms_acceleration"]
    )

    j = safe_log10(
        df["mean_squared_jerk"]
    )

    raw = np.column_stack([
        a,
        j,
    ])

    if mu is None:
        mu = raw.mean(axis=0)
        sd = raw.std(
            axis=0,
            ddof=1,
        )

    z = (
        raw - mu
    ) / sd

    za = z[:, 0]
    zj = z[:, 1]

    if kind == "acc":
        X = za[:, None]

    elif kind == "msj":
        X = zj[:, None]

    elif kind == "joint_linear":
        X = np.column_stack([
            za,
            zj,
        ])

    elif kind == "joint_interaction":
        X = np.column_stack([
            za,
            zj,
            za * zj,
        ])

    else:
        raise ValueError(kind)

    return X, mu, sd


d = pd.read_csv(IN)

assert len(d) == 7987
assert d["clip"].nunique() == 7987

d["split"] = exact_split(d)

cal = d[
    d["split"]
    == "calibration"
].copy()

val = d[
    d["split"]
    == "validation"
].copy()

assert len(cal) == 5590
assert len(val) == 2397


# ============================================================
# High-risk outcome:
# task-specific calibration Q75 moved-L2.
#
# Threshold NEVER estimated from validation.
# ============================================================

risk_threshold = (
    cal.groupby("task")[
        "l2_moved_mm"
    ]
    .quantile(0.75)
    .to_dict()
)

for df in [cal, val]:

    df["high_risk"] = [
        int(
            err >
            risk_threshold[task]
        )
        for task, err
        in zip(
            df["task"],
            df["l2_moved_mm"],
        )
    ]


print("CAL =", len(cal))
print("VAL =", len(val))
print()
print("Calibration-fixed high-risk thresholds:")
for k, v in risk_threshold.items():
    print(k, v)


# ============================================================
# Existing fixed global boundary baseline
# ============================================================

val["fixed_boundary_score"] = np.maximum(
    val["rms_acceleration"] / TA,
    val["mean_squared_jerk"] / TJ,
)

val["fixed_boundary_outside"] = (
    (
        val["rms_acceleration"]
        > TA
    )
    |
    (
        val["mean_squared_jerk"]
        > TJ
    )
).astype(int)


rows = []

ycal = cal[
    "high_risk"
].to_numpy(int)

yval = val[
    "high_risk"
].to_numpy(int)


# Continuous fixed-boundary severity.
fixed_score = (
    val[
        "fixed_boundary_score"
    ].to_numpy(float)
)

rows.append({
    "model":
        "fixed_boundary_severity",

    "validation_auc":
        roc_auc_score(
            yval,
            fixed_score,
        ),

    "validation_ap":
        average_precision_score(
            yval,
            fixed_score,
        ),

    "validation_brier":
        np.nan,
})


# ============================================================
# Joint envelope candidates
# ============================================================

predictions = pd.DataFrame({
    "clip": val["clip"].values,
    "task": val["task"].values,
    "l2_moved_mm":
        val["l2_moved_mm"].values,
    "high_risk": yval,
    "fixed_boundary_score":
        fixed_score,
    "fixed_boundary_outside":
        val[
            "fixed_boundary_outside"
        ].values,
})


for kind in [
    "acc",
    "msj",
    "joint_linear",
    "joint_interaction",
]:

    Xc, mu, sd = build_features(
        cal,
        kind,
    )

    Xv, _, _ = build_features(
        val,
        kind,
        mu=mu,
        sd=sd,
    )

    # Task controls included identically in calibration/validation.
    task_levels = sorted(
        cal["task"]
        .astype(str)
        .unique()
    )

    Ctask = pd.get_dummies(
        pd.Categorical(
            cal["task"],
            categories=task_levels,
        ),
        drop_first=True,
        dtype=float,
    ).to_numpy()

    Vtask = pd.get_dummies(
        pd.Categorical(
            val["task"],
            categories=task_levels,
        ),
        drop_first=True,
        dtype=float,
    ).to_numpy()

    Xc = np.column_stack([
        Xc,
        Ctask,
    ])

    Xv = np.column_stack([
        Xv,
        Vtask,
    ])

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=5000,
        random_state=SEED,
    )

    clf.fit(
        Xc,
        ycal,
    )

    p = clf.predict_proba(
        Xv
    )[:, 1]

    predictions[
        f"risk_{kind}"
    ] = p

    rows.append({
        "model": kind,

        "validation_auc":
            roc_auc_score(
                yval,
                p,
            ),

        "validation_ap":
            average_precision_score(
                yval,
                p,
            ),

        "validation_brier":
            brier_score_loss(
                yval,
                p,
            ),
    })


summary = pd.DataFrame(
    rows
)

summary.to_csv(
    OUTDIR /
    "E1_joint_envelope_summary.csv",
    index=False,
)

predictions.to_csv(
    OUTDIR /
    "E1_validation_predictions.csv",
    index=False,
)


print()
print("=" * 90)
print("E1 — RELIABILITY PREDICTION ON FROZEN HELD-OUT 30%")
print("=" * 90)

print(
    summary
    .sort_values(
        "validation_auc",
        ascending=False,
    )
    .to_string(
        index=False
    )
)

print()
print("=" * 90)
print("Interpretation target:")
print(
    "Does joint Acc+MSJ risk outperform "
    "the existing fixed-boundary severity?"
)
print("OUTPUT =", OUTDIR)
print("SUCCESS = True")
