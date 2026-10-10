#!/usr/bin/env python3

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

SEED = 20260902
NFOLDS = 5

ACC_BOUNDARY = 0.009832167363499603
MSJ_BOUNDARY = 8.152857428804615e-05

ROOT = Path(__file__).resolve().parents[1]

BDIR = (
    ROOT
    / "diagnostics/results/equal_physical_delta_global_heldout"
)

EST = (
    ROOT
    / "diagnostics/results/established_action_metrics_5task/"
      "per_sample_established_metrics_7987.csv"
)

OUTDIR = (
    ROOT
    / "diagnostics/results/adaptive_reliable_envelope_E2D"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

BFILES = [
    BDIR / "global_heldout_task0000__equal_delta.csv",
    BDIR / "global_heldout_task0001__equal_delta.csv",
    BDIR / "global_heldout_task0006_0008_0010__equal_delta.csv",
]

EXPECTED = {
    "task-0000": 113,
    "task-0001": 697,
    "task-0006": 535,
    "task-0008": 548,
    "task-0010": 504,
}


# =====================================================================
# Exact existing outer 70/30 split
# =====================================================================

def stable_number(text, salt=""):
    # IMPORTANT:
    # Reproduce the original fixed split EXACTLY when no salt is used:
    #     f"{SEED}|{clip}"
    # Salt is used only for NEW internal CV folds.
    if salt:
        key = f"{SEED}|{salt}|{text}"
    else:
        key = f"{SEED}|{text}"

    h = hashlib.sha256(
        key.encode()
    ).digest()

    return int.from_bytes(h[:8], "big") / 2**64


def exact_split(df):
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


# =====================================================================
# Load all B outputs
# =====================================================================

frames = []

for p in BFILES:
    if not p.exists():
        raise FileNotFoundError(p)

    q = pd.read_csv(p)

    print(
        "B FILE:",
        p.name,
        "| rows =", len(q),
        "| unique =", q["clip"].nunique()
    )

    q["source_file"] = p.name
    frames.append(q)

b = pd.concat(frames, ignore_index=True)

assert len(b) == 2397, len(b)
assert b["clip"].nunique() == 2397
assert not b["clip"].duplicated().any()


# =====================================================================
# Load established table and recover TRUE task from clip identity
# =====================================================================

e = pd.read_csv(EST)

assert len(e) == 7987
assert e["clip"].nunique() == 7987

e["split"] = exact_split(e)

ev = e[
    e["split"] == "validation"
].copy()

assert len(ev) == 2397

need = [
    "clip",
    "task",
    "rms_acceleration",
    "mean_squared_jerk",
    "l2_moved_mm",
]

m = b.merge(
    ev[need],
    on="clip",
    how="left",
    validate="one_to_one",
    suffixes=("_b_saved", "")
)

assert len(m) == 2397
assert m["task"].notna().all()


print()
print("=" * 110)
print("TRUE TASK RECOVERY")
print("=" * 110)

counts = (
    m["task"]
    .value_counts()
    .sort_index()
)

print(counts.to_string())

for task, n in EXPECTED.items():
    got = int(counts.get(task, 0))
    assert got == n, (task, got, n)


# =====================================================================
# Exact Action-metric identity check
# =====================================================================

acc_diff = np.abs(
    m["original_action_acc"].to_numpy(float)
    - m["rms_acceleration"].to_numpy(float)
)

msj_diff = np.abs(
    m["original_action_msj"].to_numpy(float)
    - m["mean_squared_jerk"].to_numpy(float)
)

print()
print("=" * 110)
print("ACTION METRIC IDENTITY")
print("=" * 110)
print("ACC max abs diff =", acc_diff.max())
print("MSJ max abs diff =", msj_diff.max())

assert acc_diff.max() <= 1e-12
assert msj_diff.max() <= 1e-12


# =====================================================================
# Risk outcome:
# SAME outer-calibration task-specific Q75 definition used in E2.
# Thresholds come from original 5590 outer calibration, not B data.
# =====================================================================

cal = e[
    e["split"] == "calibration"
].copy()

risk_thr = (
    cal.groupby("task")["l2_moved_mm"]
    .quantile(0.75)
    .to_dict()
)

m["high_risk"] = np.asarray(
    [
        int(err > risk_thr[task])
        for task, err in zip(
            m["task"],
            m["l2_moved_mm"]
        )
    ],
    dtype=int
)

m["boundary_severity"] = np.maximum(
    m["rms_acceleration"] / ACC_BOUNDARY,
    m["mean_squared_jerk"] / MSJ_BOUNDARY
)

m["boundary_outside"] = (
    m["boundary_severity"] > 1.0
).astype(int)


# =====================================================================
# Save corrected full B aggregate immediately
# =====================================================================

AGG = OUTDIR / "E2D_equal_delta_5task_2397_merged.csv"
m.to_csv(AGG, index=False)


# =====================================================================
# B mechanism:
# matched perturbations only.
#
# IMPORTANT:
# sensitivity_*_all is treated as DEVELOPMENT/MECHANISM evidence only,
# NOT final deployable inference-time score.
# =====================================================================

mechanism_rows = []

for direction, matched_col, sens_col, outside_col in [
    (
        "acc",
        "acc_matched",
        "acc_sensitivity_all_mm",
        "acc_outside",
    ),
    (
        "jerk",
        "jerk_matched",
        "jerk_sensitivity_all_mm",
        "msj_outside",
    ),
]:

    q = m[
        pd.to_numeric(
            m[matched_col],
            errors="coerce"
        ).fillna(0).astype(int) == 1
    ].copy()

    for scope, g in [
        ("ALL_5TASK", q),
        *[
            (task, q[q["task"] == task])
            for task in EXPECTED
        ],
    ]:

        if len(g) == 0:
            continue

        inside = g[
            g[outside_col] == 0
        ][sens_col].dropna().to_numpy(float)

        outside = g[
            g[outside_col] == 1
        ][sens_col].dropna().to_numpy(float)

        if len(inside) and len(outside):
            ratio = (
                np.median(outside)
                / np.median(inside)
            )

            u = mannwhitneyu(
                outside,
                inside,
                alternative="two-sided"
            )

            pval = float(u.pvalue)

        else:
            ratio = np.nan
            pval = np.nan

        rho, rp = spearmanr(
            g["boundary_severity"],
            g[sens_col],
            nan_policy="omit"
        )

        mechanism_rows.append({
            "direction": direction,
            "scope": scope,
            "n_matched": len(g),
            "inside_n": len(inside),
            "outside_n": len(outside),
            "inside_sensitivity_median_mm":
                np.median(inside) if len(inside) else np.nan,
            "outside_sensitivity_median_mm":
                np.median(outside) if len(outside) else np.nan,
            "outside_inside_ratio":
                ratio,
            "mannwhitney_p":
                pval,
            "severity_sensitivity_rho":
                rho,
            "severity_sensitivity_p":
                rp,
        })


mechanism = pd.DataFrame(mechanism_rows)

mechanism.to_csv(
    OUTDIR / "E2D_B_mechanism_summary.csv",
    index=False
)


# =====================================================================
# Fair adaptive-boundary development comparison.
#
# Restrict ALL fitted models to BOTH-matched population so that
# sensitivity model is not advantaged by a different sample set.
# =====================================================================

both = m[
    (
        pd.to_numeric(
            m["acc_matched"],
            errors="coerce"
        ).fillna(0).astype(int) == 1
    )
    &
    (
        pd.to_numeric(
            m["jerk_matched"],
            errors="coerce"
        ).fillna(0).astype(int) == 1
    )
].copy()

both = both.dropna(
    subset=[
        "boundary_severity",
        "acc_sensitivity_all_mm",
        "jerk_sensitivity_all_mm",
        "high_risk",
    ]
).copy()

print()
print("=" * 110)
print("BOTH-MATCHED FAIR POPULATION")
print("=" * 110)
print("N =", len(both))
print(both["task"].value_counts().sort_index().to_string())


# =====================================================================
# Deterministic task × risk-balanced hash folds
# =====================================================================

both["fold"] = -1

for (task, risk), g in both.groupby(
    ["task", "high_risk"]
):
    idx = sorted(
        g.index,
        key=lambda i:
            stable_number(
                str(both.at[i, "clip"]),
                salt=f"E2D-fold-{task}-{risk}"
            )
    )

    for pos, i in enumerate(idx):
        both.at[i, "fold"] = pos % NFOLDS

assert (both["fold"] >= 0).all()


# =====================================================================
# Feature families.
#
# fixed_boundary = no fitted model
# sensitivity_only = two directional local model responses
# adaptive = global dynamics prior + local model response
# =====================================================================

both["log_boundary"] = np.log1p(
    both["boundary_severity"]
)

both["log_acc_sens"] = np.log1p(
    np.maximum(
        both["acc_sensitivity_all_mm"],
        0
    )
)

both["log_jerk_sens"] = np.log1p(
    np.maximum(
        both["jerk_sensitivity_all_mm"],
        0
    )
)

FEATURES = {
    "sensitivity_only": [
        "log_acc_sens",
        "log_jerk_sens",
    ],
    "adaptive_boundary_plus_sensitivity": [
        "log_boundary",
        "log_acc_sens",
        "log_jerk_sens",
    ],
}


y = both["high_risk"].to_numpy(int)

oof = pd.DataFrame({
    "clip": both["clip"].values,
    "task": both["task"].values,
    "fold": both["fold"].values,
    "high_risk": y,
    "fixed_boundary":
        both["boundary_severity"].to_numpy(float),
})

for name in FEATURES:
    oof[name] = np.nan


for fold in range(NFOLDS):

    tr = both["fold"] != fold
    te = both["fold"] == fold

    ytr = both.loc[
        tr,
        "high_risk"
    ].to_numpy(int)

    for name, cols in FEATURES.items():

        Xtr = both.loc[
            tr,
            cols
        ].to_numpy(float)

        Xte = both.loc[
            te,
            cols
        ].to_numpy(float)

        scaler = StandardScaler()

        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=5000,
            random_state=SEED,
        )

        clf.fit(Xtr, ytr)

        oof.loc[
            te.values,
            name
        ] = clf.predict_proba(
            Xte
        )[:, 1]


assert oof[list(FEATURES)].notna().all().all()


# =====================================================================
# OOF discrimination
# =====================================================================

score_rows = []

for name in [
    "fixed_boundary",
    "sensitivity_only",
    "adaptive_boundary_plus_sensitivity",
]:

    s = oof[name].to_numpy(float)

    score_rows.append({
        "model": name,
        "n": len(oof),
        "oof_auc":
            roc_auc_score(y, s),
        "oof_ap":
            average_precision_score(y, s),
    })


scores = pd.DataFrame(score_rows)


# =====================================================================
# OOF Risk @ 80% coverage.
#
# Since fitted scores are out-of-fold, select the lowest-risk-score 80%
# by the pooled OOF score only for DEVELOPMENT comparison.
# Not a conformal guarantee.
# =====================================================================

risk80_rows = []

for name in [
    "fixed_boundary",
    "sensitivity_only",
    "adaptive_boundary_plus_sensitivity",
]:

    s = oof[name].to_numpy(float)

    lam = float(
        np.quantile(s, 0.80)
    )

    accept = s <= lam

    full_risk = float(y.mean())
    selected_risk = float(
        y[accept].mean()
    )

    risk80_rows.append({
        "model": name,
        "threshold": lam,
        "coverage": float(accept.mean()),
        "accepted_n": int(accept.sum()),
        "full_risk": full_risk,
        "selected_risk": selected_risk,
        "risk_reduction_pct":
            100.0
            * (
                1.0
                - selected_risk / full_risk
            ),
    })


risk80 = pd.DataFrame(risk80_rows)


# =====================================================================
# Paired bootstrap of OOF score differences vs fixed.
# Task-stratified resampling.
# =====================================================================

rng = np.random.default_rng(SEED)

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

BOOT = 2000

boot_rows = []

for candidate in [
    "sensitivity_only",
    "adaptive_boundary_plus_sensitivity",
]:

    dauc = []
    dap = []

    fixed = oof[
        "fixed_boundary"
    ].to_numpy(float)

    cand = oof[
        candidate
    ].to_numpy(float)

    for _ in range(BOOT):

        idx = np.concatenate([
            rng.choice(
                ids,
                size=len(ids),
                replace=True
            )
            for ids in task_indices.values()
        ])

        yb = y[idx]

        if len(np.unique(yb)) < 2:
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

    observed_auc = (
        scores.loc[
            scores["model"] == candidate,
            "oof_auc"
        ].iloc[0]
        -
        scores.loc[
            scores["model"] == "fixed_boundary",
            "oof_auc"
        ].iloc[0]
    )

    observed_ap = (
        scores.loc[
            scores["model"] == candidate,
            "oof_ap"
        ].iloc[0]
        -
        scores.loc[
            scores["model"] == "fixed_boundary",
            "oof_ap"
        ].iloc[0]
    )

    boot_rows.append({
        "model": candidate,
        "delta_auc_vs_fixed":
            observed_auc,
        "delta_auc_ci025":
            np.quantile(dauc, 0.025),
        "delta_auc_ci975":
            np.quantile(dauc, 0.975),
        "delta_ap_vs_fixed":
            observed_ap,
        "delta_ap_ci025":
            np.quantile(dap, 0.025),
        "delta_ap_ci975":
            np.quantile(dap, 0.975),
    })


bootdf = pd.DataFrame(boot_rows)


# =====================================================================
# Task-wise adaptive result
# =====================================================================

task_rows = []

for task, g in oof.groupby("task"):

    yt = g["high_risk"].to_numpy(int)

    for name in [
        "fixed_boundary",
        "sensitivity_only",
        "adaptive_boundary_plus_sensitivity",
    ]:

        st = g[name].to_numpy(float)

        if len(np.unique(yt)) >= 2:
            auc = roc_auc_score(
                yt,
                st
            )
            ap = average_precision_score(
                yt,
                st
            )
        else:
            auc = np.nan
            ap = np.nan

        task_rows.append({
            "task": task,
            "model": name,
            "n": len(g),
            "auc": auc,
            "ap": ap,
        })


taskdf = pd.DataFrame(task_rows)


# =====================================================================
# Save
# =====================================================================

scores.to_csv(
    OUTDIR / "E2D_oof_reliability_scores.csv",
    index=False
)

risk80.to_csv(
    OUTDIR / "E2D_oof_risk_at_80coverage.csv",
    index=False
)

bootdf.to_csv(
    OUTDIR / "E2D_bootstrap_vs_fixed.csv",
    index=False
)

taskdf.to_csv(
    OUTDIR / "E2D_oof_by_task.csv",
    index=False
)

oof.to_csv(
    OUTDIR / "E2D_oof_predictions.csv",
    index=False
)


# =====================================================================
# Print
# =====================================================================

print()
print("=" * 110)
print("B — 5-TASK EQUAL-DELTA MECHANISM")
print("=" * 110)

print(
    mechanism.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6g}"
    )
)

print()
print("=" * 110)
print("E2-D — FAIR OOF RELIABILITY COMPARISON")
print("=" * 110)

print(
    scores.sort_values(
        "oof_auc",
        ascending=False
    ).to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)

print()
print("=" * 110)
print("E2-D — RISK @ 80% OOF COVERAGE")
print("=" * 110)

print(
    risk80.sort_values(
        "selected_risk"
    ).to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)

print()
print("=" * 110)
print("E2-D — BOOTSTRAP INCREMENTAL VALUE VS FIXED")
print("=" * 110)

print(
    bootdf.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)

print()
print("=" * 110)
print("E2-D — TASK-WISE OOF")
print("=" * 110)

print(
    taskdf.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)

print()
print("=" * 110)
print("INTERPRETATION RULE")
print("=" * 110)
print(
    "Promote adaptive/model-sensitive boundary ONLY if "
    "boundary+sensitivity improves fixed boundary on OOF AUC/AP "
    "and bootstrap evidence supports genuine incremental value."
)
print(
    "B sensitivity_all is development/mechanism evidence only. "
    "If it survives, next run computes a GT-mask-free deployable "
    "local-sensitivity score with new PointWorld inference."
)
print()
print("SAVED AGGREGATE =", AGG)
print("OUTPUT =", OUTDIR)
print("SUCCESS = True")
print("=" * 110)
