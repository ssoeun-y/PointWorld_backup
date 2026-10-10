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
    / "diagnostics/results/context_interaction_oof_5task"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


def stable_number(text, salt):
    h = hashlib.sha256(
        f"{SEED}|{salt}|{text}".encode()
    ).digest()
    return int.from_bytes(h[:8], "big") / 2**64


# ============================================================
# Load exact same fair population
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
# Pre-fixed features
# ============================================================

q["log_boundary"] = np.log1p(
    q["boundary_severity"].astype(float)
)

# lower distance = closer
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

FEATURES = [
    "log_boundary",
    "closeness",
    "boundary_x_closeness",
]

# ============================================================
# Same deterministic task × risk balanced folds
# ============================================================

q["fold"] = -1

for (task, risk), g in q.groupby(
    ["task", "high_risk"]
):
    idx = sorted(
        g.index,
        key=lambda i: stable_number(
            str(q.at[i, "clip"]),
            f"CTXINT-{task}-{risk}",
        ),
    )

    for pos, i in enumerate(idx):
        q.at[i, "fold"] = pos % NFOLDS

assert (q["fold"] >= 0).all()

# ============================================================
# PER-TASK OOF
#
# Important:
# each task learns its own context relationship.
# No sample is predicted by a model trained on itself.
# ============================================================

oof = pd.DataFrame({
    "clip": q["clip"].values,
    "task": q["task"].values,
    "high_risk":
        q["high_risk"].to_numpy(int),
    "fixed_boundary":
        q["boundary_severity"].to_numpy(float),
    "context_interaction_oof":
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

        Xtr = q.loc[
            tr,
            FEATURES,
        ].to_numpy(float)

        Xte = q.loc[
            te,
            FEATURES,
        ].to_numpy(float)

        ytr = q.loc[
            tr,
            "high_risk",
        ].to_numpy(int)

        scaler = StandardScaler()

        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=5000,
            random_state=SEED,
        )

        clf.fit(
            Xtr,
            ytr,
        )

        pred = clf.predict_proba(
            Xte
        )[:, 1]

        oof.loc[
            te.values,
            "context_interaction_oof",
        ] = pred

        coef_rows.append({
            "task": task,
            "fold": fold,
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),

            "coef_boundary_std":
                float(clf.coef_[0, 0]),

            "coef_closeness_std":
                float(clf.coef_[0, 1]),

            "coef_interaction_std":
                float(clf.coef_[0, 2]),
        })

assert (
    oof["context_interaction_oof"]
    .notna()
    .all()
)

coefdf = pd.DataFrame(coef_rows)

# ============================================================
# Overall OOF performance
# ============================================================

y = oof["high_risk"].to_numpy(int)

summary_rows = []

for name in [
    "fixed_boundary",
    "context_interaction_oof",
]:

    s = oof[name].to_numpy(float)

    summary_rows.append({
        "model": name,
        "n": len(oof),
        "auc":
            roc_auc_score(y, s),
        "ap":
            average_precision_score(y, s),
    })

summary = pd.DataFrame(summary_rows)

# ============================================================
# Risk @80%
# ============================================================

full_risk = float(y.mean())

risk_rows = []

for name in [
    "fixed_boundary",
    "context_interaction_oof",
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
        "risk": risk,
        "risk_reduction_pct":
            100.0 * (
                1.0
                - risk / full_risk
            ),
    })

riskdf = pd.DataFrame(risk_rows)

# ============================================================
# Task-wise OOF
# ============================================================

task_rows = []

for task, g in oof.groupby("task"):

    yt = g["high_risk"].to_numpy(int)

    for name in [
        "fixed_boundary",
        "context_interaction_oof",
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
# Interaction coefficient stability across folds
# ============================================================

coef_summary = (
    coefdf
    .groupby("task")
    .agg(
        interaction_mean=(
            "coef_interaction_std",
            "mean"
        ),
        interaction_min=(
            "coef_interaction_std",
            "min"
        ),
        interaction_max=(
            "coef_interaction_std",
            "max"
        ),
        closeness_mean=(
            "coef_closeness_std",
            "mean"
        ),
        positive_interaction_folds=(
            "coef_interaction_std",
            lambda x:
                int((x > 0).sum())
        ),
    )
    .reset_index()
)

# ============================================================
# Paired task-stratified bootstrap vs fixed
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

fixed = oof[
    "fixed_boundary"
].to_numpy(float)

cand = oof[
    "context_interaction_oof"
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
            fixed[idx],
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
            fixed[idx],
        )
    )

boot = pd.DataFrame([{
    "model":
        "context_interaction_oof",

    "delta_auc":
        float(np.mean(dauc)),

    "delta_auc_ci025":
        float(
            np.quantile(
                dauc,
                0.025,
            )
        ),

    "delta_auc_ci975":
        float(
            np.quantile(
                dauc,
                0.975,
            )
        ),

    "delta_ap":
        float(np.mean(dap)),

    "delta_ap_ci025":
        float(
            np.quantile(
                dap,
                0.025,
            )
        ),

    "delta_ap_ci975":
        float(
            np.quantile(
                dap,
                0.975,
            )
        ),
}])

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

coefdf.to_csv(
    OUTDIR / "fold_coefficients.csv",
    index=False,
)

coef_summary.to_csv(
    OUTDIR / "interaction_summary.csv",
    index=False,
)

boot.to_csv(
    OUTDIR / "bootstrap.csv",
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
print("CONTEXT INTERACTION — 5-TASK OOF")
print("=" * 100)
print(summary.to_string(index=False))

print("\nRISK @ 80%")
print(riskdf.to_string(index=False))

print("\nBOOTSTRAP VS FIXED")
print(boot.to_string(index=False))

print("\nTASK-WISE OOF")
print(taskdf.to_string(index=False))

print("\nINTERACTION COEFFICIENT STABILITY")
print(coef_summary.to_string(index=False))

print("\nSUCCESS = True")
print("OUTPUT =", OUTDIR)
