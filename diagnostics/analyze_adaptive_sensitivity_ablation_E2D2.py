#!/usr/bin/env python3

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

SEED = 20260902
NFOLDS = 5
BOOT = 2000

ROOT = Path(__file__).resolve().parents[1]

IN = (
    ROOT
    / "diagnostics/results/adaptive_reliable_envelope_E2D/"
      "E2D_equal_delta_5task_2397_merged.csv"
)

OUTDIR = (
    ROOT
    / "diagnostics/results/adaptive_reliable_envelope_E2D/"
      "E2D2_sensitivity_ablation"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


def stable_number(text, salt):
    h = hashlib.sha256(
        f"{SEED}|{salt}|{text}".encode()
    ).digest()
    return int.from_bytes(h[:8], "big") / 2**64


d = pd.read_csv(IN)

# ------------------------------------------------------------
# Fair population: both perturbations matched.
# ------------------------------------------------------------

q = d[
    (d["acc_matched"] == 1)
    &
    (d["jerk_matched"] == 1)
].copy()

q = q.dropna(
    subset=[
        "boundary_severity",
        "acc_sensitivity_all_mm",
        "jerk_sensitivity_all_mm",
        "high_risk",
    ]
).copy()

assert len(q) == 1669


# ------------------------------------------------------------
# Transform features.
# ------------------------------------------------------------

q["log_boundary"] = np.log1p(
    q["boundary_severity"]
)

q["log_acc_sens"] = np.log1p(
    np.maximum(
        q["acc_sensitivity_all_mm"],
        0
    )
)

q["log_jerk_sens"] = np.log1p(
    np.maximum(
        q["jerk_sensitivity_all_mm"],
        0
    )
)


FEATURES = {
    "boundary_plus_acc": [
        "log_boundary",
        "log_acc_sens",
    ],

    "boundary_plus_jerk": [
        "log_boundary",
        "log_jerk_sens",
    ],

    "boundary_plus_both": [
        "log_boundary",
        "log_acc_sens",
        "log_jerk_sens",
    ],
}


# ------------------------------------------------------------
# Same deterministic task × risk balanced folds.
# ------------------------------------------------------------

q["fold"] = -1

for (task, risk), g in q.groupby(
    ["task", "high_risk"]
):
    idx = sorted(
        g.index,
        key=lambda i:
            stable_number(
                str(q.at[i, "clip"]),
                f"E2D2-{task}-{risk}"
            )
    )

    for pos, i in enumerate(idx):
        q.at[i, "fold"] = (
            pos % NFOLDS
        )

assert (q["fold"] >= 0).all()


y = q["high_risk"].to_numpy(int)

oof = pd.DataFrame({
    "clip": q["clip"].values,
    "task": q["task"].values,
    "high_risk": y,
    "fixed_boundary":
        q["boundary_severity"].to_numpy(float),
})

for name in FEATURES:
    oof[name] = np.nan


# ------------------------------------------------------------
# OOF fits.
# ------------------------------------------------------------

for fold in range(NFOLDS):

    tr = q["fold"] != fold
    te = q["fold"] == fold

    ytr = q.loc[
        tr,
        "high_risk"
    ].to_numpy(int)

    for name, cols in FEATURES.items():

        Xtr = q.loc[
            tr,
            cols
        ].to_numpy(float)

        Xte = q.loc[
            te,
            cols
        ].to_numpy(float)

        scaler = StandardScaler()

        Xtr = scaler.fit_transform(
            Xtr
        )

        Xte = scaler.transform(
            Xte
        )

        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=5000,
            random_state=SEED,
        )

        clf.fit(
            Xtr,
            ytr
        )

        oof.loc[
            te.values,
            name
        ] = clf.predict_proba(
            Xte
        )[:, 1]


assert oof[list(FEATURES)].notna().all().all()


# ------------------------------------------------------------
# AUC / AP.
# ------------------------------------------------------------

names = [
    "fixed_boundary",
    "boundary_plus_acc",
    "boundary_plus_jerk",
    "boundary_plus_both",
]

rows = []

for name in names:

    s = oof[name].to_numpy(float)

    rows.append({
        "model": name,
        "n": len(oof),
        "auc":
            roc_auc_score(
                y,
                s
            ),
        "ap":
            average_precision_score(
                y,
                s
            ),
    })

summary = pd.DataFrame(rows)


# ------------------------------------------------------------
# Risk @ 80% OOF coverage.
# ------------------------------------------------------------

risk_rows = []

full_risk = float(
    y.mean()
)

for name in names:

    s = oof[name].to_numpy(float)

    lam = float(
        np.quantile(
            s,
            0.80
        )
    )

    accept = (
        s <= lam
    )

    risk = float(
        y[accept].mean()
    )

    risk_rows.append({
        "model": name,
        "coverage":
            float(accept.mean()),
        "accepted_n":
            int(accept.sum()),
        "risk": risk,
        "risk_reduction_pct":
            100.0
            * (
                1.0
                - risk / full_risk
            ),
    })

riskdf = pd.DataFrame(
    risk_rows
)


# ------------------------------------------------------------
# Paired task-stratified bootstrap vs fixed.
# ------------------------------------------------------------

rng = np.random.default_rng(
    SEED
)

task_indices = {
    task:
        np.flatnonzero(
            oof["task"].to_numpy()
            == task
        )
    for task in sorted(
        oof["task"].unique()
    )
}

fixed = oof[
    "fixed_boundary"
].to_numpy(float)

boot_rows = []

for candidate in names[1:]:

    cand = oof[
        candidate
    ].to_numpy(float)

    dauc = []
    dap = []

    for _ in range(BOOT):

        idx = np.concatenate([
            rng.choice(
                ids,
                size=len(ids),
                replace=True
            )
            for ids
            in task_indices.values()
        ])

        yb = y[idx]

        if len(
            np.unique(yb)
        ) < 2:
            continue

        dauc.append(
            roc_auc_score(
                yb,
                cand[idx]
            )
            -
            roc_auc_score(
                yb,
                fixed[idx]
            )
        )

        dap.append(
            average_precision_score(
                yb,
                cand[idx]
            )
            -
            average_precision_score(
                yb,
                fixed[idx]
            )
        )

    dauc = np.asarray(dauc)
    dap = np.asarray(dap)

    obs_auc = (
        summary.loc[
            summary["model"]
            == candidate,
            "auc"
        ].iloc[0]
        -
        summary.loc[
            summary["model"]
            == "fixed_boundary",
            "auc"
        ].iloc[0]
    )

    obs_ap = (
        summary.loc[
            summary["model"]
            == candidate,
            "ap"
        ].iloc[0]
        -
        summary.loc[
            summary["model"]
            == "fixed_boundary",
            "ap"
        ].iloc[0]
    )

    boot_rows.append({
        "model":
            candidate,

        "delta_auc":
            obs_auc,

        "delta_auc_ci025":
            float(
                np.quantile(
                    dauc,
                    0.025
                )
            ),

        "delta_auc_ci975":
            float(
                np.quantile(
                    dauc,
                    0.975
                )
            ),

        "delta_ap":
            obs_ap,

        "delta_ap_ci025":
            float(
                np.quantile(
                    dap,
                    0.025
                )
            ),

        "delta_ap_ci975":
            float(
                np.quantile(
                    dap,
                    0.975
                )
            ),
    })


bootdf = pd.DataFrame(
    boot_rows
)


# ------------------------------------------------------------
# Task-wise comparison.
# ------------------------------------------------------------

task_rows = []

for task, g in oof.groupby(
    "task"
):

    yt = g[
        "high_risk"
    ].to_numpy(int)

    for name in names:

        s = g[name].to_numpy(float)

        task_rows.append({
            "task": task,
            "model": name,
            "n": len(g),
            "auc":
                roc_auc_score(
                    yt,
                    s
                ),
            "ap":
                average_precision_score(
                    yt,
                    s
                ),
        })


taskdf = pd.DataFrame(
    task_rows
)


# ------------------------------------------------------------
# Save + print.
# ------------------------------------------------------------

summary.to_csv(
    OUTDIR /
    "E2D2_summary.csv",
    index=False
)

riskdf.to_csv(
    OUTDIR /
    "E2D2_risk80.csv",
    index=False
)

bootdf.to_csv(
    OUTDIR /
    "E2D2_bootstrap.csv",
    index=False
)

taskdf.to_csv(
    OUTDIR /
    "E2D2_by_task.csv",
    index=False
)

oof.to_csv(
    OUTDIR /
    "E2D2_oof_predictions.csv",
    index=False
)


print("=" * 105)
print("E2-D2 — SENSITIVITY ABLATION")
print("=" * 105)

print(
    summary.sort_values(
        "auc",
        ascending=False
    ).to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)

print()
print("=" * 105)
print("RISK @ 80% COVERAGE")
print("=" * 105)

print(
    riskdf.sort_values(
        "risk"
    ).to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)

print()
print("=" * 105)
print("BOOTSTRAP VS FIXED")
print("=" * 105)

print(
    bootdf.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)

print()
print("=" * 105)
print("TASK-WISE")
print("=" * 105)

print(
    taskdf.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)

print()
print("OUTPUT =", OUTDIR)
print("SUCCESS = True")
