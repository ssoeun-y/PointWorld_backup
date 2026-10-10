from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 20260902
NFOLDS = 5
BOOT = 2000

ROOT = Path(__file__).resolve().parents[1]

PROX = (
    ROOT
    / "diagnostics/results/scene_action_proximity_probe/"
      "scene_action_proximity_5task_2397.csv"
)

REF = (
    ROOT
    / "diagnostics/results/adaptive_reliable_envelope_E2D/"
      "E2D_equal_delta_5task_2397_merged.csv"
)

OUTDIR = (
    ROOT
    / "diagnostics/results/context_incremental_fair_oof"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


def stable_number(text, salt):
    h = hashlib.sha256(
        f"{SEED}|{salt}|{text}".encode()
    ).digest()
    return int.from_bytes(h[:8], "big") / 2**64


# ============================================================
# Load exact fair population
# ============================================================

d = pd.read_csv(PROX)

ref = (
    pd.read_csv(REF)[
        [
            "clip",
            "boundary_severity",
            "high_risk",
        ]
    ]
    .drop_duplicates("clip")
)

q = (
    d[d["acc_matched"] == 1]
    .merge(
        ref,
        on="clip",
        how="left",
        validate="one_to_one",
    )
)

q = q.dropna(
    subset=[
        "scene_action_dist_p05_native",
        "boundary_severity",
        "high_risk",
    ]
).copy()

assert len(q) == 2067, len(q)

expected = {
    "task-0000": 95,
    "task-0001": 612,
    "task-0006": 467,
    "task-0008": 486,
    "task-0010": 407,
}

assert (
    q["task"]
    .value_counts()
    .sort_index()
    .to_dict()
    == expected
)

# ============================================================
# Features
# ============================================================

q["log_boundary"] = np.log1p(
    q["boundary_severity"].astype(float)
)

q["closeness"] = -np.log(
    np.maximum(
        q["scene_action_dist_p05_native"].astype(float),
        1e-8,
    )
)

q["boundary_x_closeness"] = (
    q["log_boundary"]
    * q["closeness"]
)

BOUNDARY_FEATURES = [
    "log_boundary",
]

CONTEXT_FEATURES = [
    "log_boundary",
    "closeness",
    "boundary_x_closeness",
]

# ============================================================
# EXACT SAME deterministic task × risk folds
# for both candidate models
# ============================================================

q["fold"] = -1

for (task, risk), g in q.groupby(
    ["task", "high_risk"]
):
    idx = sorted(
        g.index,
        key=lambda i: stable_number(
            str(q.at[i, "clip"]),
            f"FAIRCTX-{task}-{risk}",
        ),
    )

    for pos, i in enumerate(idx):
        q.at[i, "fold"] = pos % NFOLDS

assert (q["fold"] >= 0).all()

# ============================================================
# PER-TASK OOF:
# both models trained under IDENTICAL task/fold conditions
# ============================================================

oof = pd.DataFrame({
    "clip": q["clip"].values,
    "task": q["task"].values,
    "high_risk":
        q["high_risk"].to_numpy(int),

    "task_boundary_oof":
        np.nan,

    "task_context_oof":
        np.nan,
})

coef_rows = []

for task in sorted(q["task"].unique()):

    task_mask = q["task"] == task

    for fold in range(NFOLDS):

        tr = (
            task_mask
            & (q["fold"] != fold)
        )

        te = (
            task_mask
            & (q["fold"] == fold)
        )

        if te.sum() == 0:
            continue

        ytr = q.loc[
            tr,
            "high_risk",
        ].to_numpy(int)

        # --------------------------------------------
        # Model 1: task-specific boundary only
        # --------------------------------------------

        Xtr_b = q.loc[
            tr,
            BOUNDARY_FEATURES,
        ].to_numpy(float)

        Xte_b = q.loc[
            te,
            BOUNDARY_FEATURES,
        ].to_numpy(float)

        sb = StandardScaler()

        Xtr_b = sb.fit_transform(Xtr_b)
        Xte_b = sb.transform(Xte_b)

        mb = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=5000,
            random_state=SEED,
        )

        mb.fit(
            Xtr_b,
            ytr,
        )

        pb = mb.predict_proba(
            Xte_b
        )[:, 1]

        oof.loc[
            te.values,
            "task_boundary_oof",
        ] = pb

        # --------------------------------------------
        # Model 2: SAME task/fold +
        # boundary + proximity + interaction
        # --------------------------------------------

        Xtr_c = q.loc[
            tr,
            CONTEXT_FEATURES,
        ].to_numpy(float)

        Xte_c = q.loc[
            te,
            CONTEXT_FEATURES,
        ].to_numpy(float)

        sc = StandardScaler()

        Xtr_c = sc.fit_transform(Xtr_c)
        Xte_c = sc.transform(Xte_c)

        mc = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=5000,
            random_state=SEED,
        )

        mc.fit(
            Xtr_c,
            ytr,
        )

        pc = mc.predict_proba(
            Xte_c
        )[:, 1]

        oof.loc[
            te.values,
            "task_context_oof",
        ] = pc

        coef_rows.append({
            "task": task,
            "fold": fold,
            "boundary_coef":
                float(mc.coef_[0, 0]),
            "closeness_coef":
                float(mc.coef_[0, 1]),
            "interaction_coef":
                float(mc.coef_[0, 2]),
        })

assert (
    oof[
        [
            "task_boundary_oof",
            "task_context_oof",
        ]
    ]
    .notna()
    .all()
    .all()
)

y = oof["high_risk"].to_numpy(int)

# ============================================================
# Overall OOF AUC / AP
# ============================================================

rows = []

for name in [
    "task_boundary_oof",
    "task_context_oof",
]:

    s = oof[name].to_numpy(float)

    rows.append({
        "model": name,
        "n": len(oof),
        "auc":
            roc_auc_score(
                y,
                s,
            ),
        "ap":
            average_precision_score(
                y,
                s,
            ),
    })

summary = pd.DataFrame(rows)

# ============================================================
# Risk @80%
# ============================================================

full_risk = float(y.mean())

risk_rows = []

for name in [
    "task_boundary_oof",
    "task_context_oof",
]:

    s = oof[name].to_numpy(float)

    lam = float(
        np.quantile(
            s,
            0.80,
        )
    )

    accept = s <= lam

    risk = float(
        y[accept].mean()
    )

    risk_rows.append({
        "model": name,
        "coverage":
            float(accept.mean()),
        "accepted_n":
            int(accept.sum()),
        "risk":
            risk,
        "risk_reduction_pct":
            100.0
            * (
                1.0
                - risk / full_risk
            ),
    })

riskdf = pd.DataFrame(risk_rows)

# ============================================================
# Task-wise fair comparison
# ============================================================

task_rows = []

for task, g in oof.groupby("task"):

    yt = g["high_risk"].to_numpy(int)

    for name in [
        "task_boundary_oof",
        "task_context_oof",
    ]:

        s = g[name].to_numpy(float)

        task_rows.append({
            "task": task,
            "model": name,
            "n": len(g),
            "auc":
                roc_auc_score(
                    yt,
                    s,
                ),
            "ap":
                average_precision_score(
                    yt,
                    s,
                ),
        })

taskdf = pd.DataFrame(task_rows)

# ============================================================
# Paired task-stratified bootstrap:
# context - task-specific boundary
# ============================================================

rng = np.random.default_rng(SEED)

task_idx = {
    task:
        np.flatnonzero(
            oof["task"].to_numpy()
            == task
        )
    for task in sorted(
        oof["task"].unique()
    )
}

base = oof[
    "task_boundary_oof"
].to_numpy(float)

cand = oof[
    "task_context_oof"
].to_numpy(float)

dauc = []
dap = []

for _ in range(BOOT):

    idx = np.concatenate([
        rng.choice(
            task_idx[t],
            size=len(task_idx[t]),
            replace=True,
        )
        for t in task_idx
    ])

    yy = y[idx]

    if len(np.unique(yy)) < 2:
        continue

    dauc.append(
        roc_auc_score(
            yy,
            cand[idx],
        )
        -
        roc_auc_score(
            yy,
            base[idx],
        )
    )

    dap.append(
        average_precision_score(
            yy,
            cand[idx],
        )
        -
        average_precision_score(
            yy,
            base[idx],
        )
    )

boot = pd.DataFrame([{
    "comparison":
        "task_context_minus_task_boundary",

    "delta_auc":
        float(np.mean(dauc)),

    "delta_auc_ci025":
        float(
            np.quantile(
                dauc,
                .025,
            )
        ),

    "delta_auc_ci975":
        float(
            np.quantile(
                dauc,
                .975,
            )
        ),

    "delta_ap":
        float(np.mean(dap)),

    "delta_ap_ci025":
        float(
            np.quantile(
                dap,
                .025,
            )
        ),

    "delta_ap_ci975":
        float(
            np.quantile(
                dap,
                .975,
            )
        ),
}])

# ============================================================
# Coefficient stability
# ============================================================

coefdf = pd.DataFrame(
    coef_rows
)

coefsummary = (
    coefdf
    .groupby("task")
    .agg(
        closeness_mean=(
            "closeness_coef",
            "mean"
        ),
        interaction_mean=(
            "interaction_coef",
            "mean"
        ),
        interaction_min=(
            "interaction_coef",
            "min"
        ),
        interaction_max=(
            "interaction_coef",
            "max"
        ),
        interaction_positive_folds=(
            "interaction_coef",
            lambda x:
                int((x > 0).sum())
        ),
    )
    .reset_index()
)

# ============================================================
# Save
# ============================================================

summary.to_csv(
    OUTDIR / "summary.csv",
    index=False,
)

riskdf.to_csv(
    OUTDIR / "risk80.csv",
    index=False,
)

taskdf.to_csv(
    OUTDIR / "by_task.csv",
    index=False,
)

boot.to_csv(
    OUTDIR / "bootstrap.csv",
    index=False,
)

coefdf.to_csv(
    OUTDIR / "fold_coefficients.csv",
    index=False,
)

coefsummary.to_csv(
    OUTDIR / "coefficient_summary.csv",
    index=False,
)

oof.to_csv(
    OUTDIR / "oof_predictions.csv",
    index=False,
)

# ============================================================
# Print
# ============================================================

print("\n" + "=" * 100)
print("FAIR INCREMENTAL CONTEXT TEST")
print("SAME TASK / SAME FOLD / SAME OOF")
print("=" * 100)
print(summary.to_string(index=False))

print("\nRISK @ 80%")
print(riskdf.to_string(index=False))

print("\nBOOTSTRAP: CONTEXT - TASK BOUNDARY")
print(boot.to_string(index=False))

print("\nTASK-WISE")
print(taskdf.to_string(index=False))

print("\nCOEFFICIENT STABILITY")
print(coefsummary.to_string(index=False))

print("\nSUCCESS = True")
print("OUTPUT =", OUTDIR)
