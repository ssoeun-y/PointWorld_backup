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

IN = (
    ROOT
    / "diagnostics/results/scene_action_proximity_probe/"
      "scene_action_proximity_5task_2397.csv"
)

REF = (
    ROOT
    / "diagnostics/results/adaptive_reliable_envelope_E2D/"
      "E2D_equal_delta_5task_2397_merged.csv"
)

OUT = (
    ROOT
    / "diagnostics/results/scene_action_proximity_probe"
)

d = pd.read_csv(IN)

# Fair population = Acc-matched only
q = d[d["acc_matched"] == 1].copy()

ref = pd.read_csv(REF)[
    ["clip", "boundary_severity", "high_risk"]
].drop_duplicates("clip")

q = q.merge(
    ref,
    on="clip",
    how="left",
    validate="one_to_one",
)

q = q.dropna(
    subset=[
        "scene_action_dist_p05_native",
        "boundary_severity",
        "high_risk",
    ]
).copy()

assert len(q) == 2067, len(q)

print("N =", len(q))
print(
    "TASKS =",
    q["task"].value_counts().sort_index().to_dict()
)

# ------------------------------------------------------------
# Pre-fixed context model
# boundary + scene-action closeness + interaction
# ------------------------------------------------------------
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
    q["log_boundary"] * q["closeness"]
)

FEATURES = [
    "log_boundary",
    "closeness",
    "boundary_x_closeness",
]

def stable_number(text, salt):
    h = hashlib.sha256(
        f"{SEED}|{salt}|{text}".encode()
    ).digest()
    return int.from_bytes(h[:8], "big") / 2**64

# deterministic task × risk balanced folds
q["fold"] = -1

for (task, risk), g in q.groupby(
    ["task", "high_risk"]
):
    idx = sorted(
        g.index,
        key=lambda i: stable_number(
            str(q.at[i, "clip"]),
            f"PROX-{task}-{risk}",
        ),
    )

    for pos, i in enumerate(idx):
        q.at[i, "fold"] = pos % NFOLDS

assert (q["fold"] >= 0).all()

y = q["high_risk"].to_numpy(int)

oof = pd.DataFrame({
    "clip": q["clip"].values,
    "task": q["task"].values,
    "high_risk": y,
    "fixed_boundary":
        q["boundary_severity"].to_numpy(float),
    "context_adaptive": np.nan,
})

for fold in range(NFOLDS):

    tr = q["fold"] != fold
    te = q["fold"] == fold

    Xtr = q.loc[tr, FEATURES].to_numpy(float)
    Xte = q.loc[te, FEATURES].to_numpy(float)
    ytr = q.loc[tr, "high_risk"].to_numpy(int)

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
        "context_adaptive",
    ] = clf.predict_proba(Xte)[:, 1]

assert oof["context_adaptive"].notna().all()

# AUC / AP
rows = []

for name in [
    "fixed_boundary",
    "context_adaptive",
]:
    score = oof[name].to_numpy(float)

    rows.append({
        "model": name,
        "n": len(oof),
        "auc": roc_auc_score(y, score),
        "ap": average_precision_score(y, score),
    })

summary = pd.DataFrame(rows)

# Risk @80%
full_risk = float(y.mean())
risk_rows = []

for name in [
    "fixed_boundary",
    "context_adaptive",
]:
    score = oof[name].to_numpy(float)

    lam = float(np.quantile(score, 0.80))
    accept = score <= lam
    risk = float(y[accept].mean())

    risk_rows.append({
        "model": name,
        "coverage": float(accept.mean()),
        "accepted_n": int(accept.sum()),
        "risk": risk,
        "risk_reduction_pct":
            100.0 * (1.0 - risk / full_risk),
    })

riskdf = pd.DataFrame(risk_rows)

# Task-wise
task_rows = []

for task, g in oof.groupby("task"):

    yt = g["high_risk"].to_numpy(int)

    for name in [
        "fixed_boundary",
        "context_adaptive",
    ]:
        score = g[name].to_numpy(float)

        task_rows.append({
            "task": task,
            "model": name,
            "n": len(g),
            "auc": roc_auc_score(yt, score),
            "ap": average_precision_score(yt, score),
        })

taskdf = pd.DataFrame(task_rows)

# Paired task-stratified bootstrap
rng = np.random.default_rng(SEED)

task_idx = {
    t: np.flatnonzero(
        oof["task"].to_numpy() == t
    )
    for t in sorted(oof["task"].unique())
}

fixed = oof["fixed_boundary"].to_numpy(float)
cand = oof["context_adaptive"].to_numpy(float)

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
        roc_auc_score(yy, cand[idx])
        - roc_auc_score(yy, fixed[idx])
    )

    dap.append(
        average_precision_score(yy, cand[idx])
        - average_precision_score(yy, fixed[idx])
    )

boot = pd.DataFrame([{
    "model": "context_adaptive",
    "delta_auc": float(np.mean(dauc)),
    "delta_auc_ci025": float(np.quantile(dauc, .025)),
    "delta_auc_ci975": float(np.quantile(dauc, .975)),
    "delta_ap": float(np.mean(dap)),
    "delta_ap_ci025": float(np.quantile(dap, .025)),
    "delta_ap_ci975": float(np.quantile(dap, .975)),
}])

summary.to_csv(OUT / "proximity_summary.csv", index=False)
riskdf.to_csv(OUT / "proximity_risk80.csv", index=False)
taskdf.to_csv(OUT / "proximity_by_task.csv", index=False)
boot.to_csv(OUT / "proximity_bootstrap.csv", index=False)
oof.to_csv(OUT / "proximity_oof.csv", index=False)

print("\n" + "=" * 90)
print("SCENE-ACTION PROXIMITY QUICK PROBE")
print("=" * 90)
print(summary.to_string(index=False))

print("\nRISK @ 80%")
print(riskdf.to_string(index=False))

print("\nBOOTSTRAP VS FIXED")
print(boot.to_string(index=False))

print("\nTASK-WISE")
print(taskdf.to_string(index=False))

print("\nSUCCESS = True")
print("OUTPUT =", OUT)
