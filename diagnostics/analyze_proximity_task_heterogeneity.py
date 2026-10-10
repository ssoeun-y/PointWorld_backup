from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

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

OUT = (
    ROOT
    / "diagnostics/results/scene_action_proximity_probe/"
      "proximity_task_heterogeneity.csv"
)

d = pd.read_csv(PROX)
r = pd.read_csv(REF)[
    ["clip", "boundary_severity", "high_risk"]
].drop_duplicates("clip")

q = (
    d[d["acc_matched"] == 1]
    .merge(r, on="clip", how="left", validate="one_to_one")
)

q = q.dropna(
    subset=[
        "scene_action_dist_p05_native",
        "boundary_severity",
        "high_risk",
    ]
).copy()

assert len(q) == 2067

# lower distance = closer
q["closeness"] = -np.log(
    np.maximum(
        q["scene_action_dist_p05_native"].astype(float),
        1e-8,
    )
)

q["log_boundary"] = np.log1p(
    q["boundary_severity"].astype(float)
)

rows = []

for task, g in q.groupby("task"):

    y = g["high_risk"].to_numpy(int)

    # Standardized within-task features
    Xb = g[["log_boundary"]].to_numpy(float)
    Xc = g[["log_boundary", "closeness"]].to_numpy(float)

    sb = StandardScaler()
    sc = StandardScaler()

    Xb = sb.fit_transform(Xb)
    Xc = sc.fit_transform(Xc)

    mb = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=5000,
        random_state=20260902,
    ).fit(Xb, y)

    mc = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=5000,
        random_state=20260902,
    ).fit(Xc, y)

    pb = mb.predict_proba(Xb)[:, 1]
    pc = mc.predict_proba(Xc)[:, 1]

    # proximity quartiles, defined within each task
    gg = g.copy()
    gg["prox_q"] = pd.qcut(
        gg["closeness"],
        4,
        labels=["Q1_far", "Q2", "Q3", "Q4_close"],
        duplicates="drop",
    )

    qrisk = (
        gg.groupby("prox_q", observed=True)["high_risk"]
        .mean()
        .to_dict()
    )

    rows.append({
        "task": task,
        "n": len(g),

        "median_dist_p05":
            float(g["scene_action_dist_p05_native"].median()),

        "fixed_auc":
            float(roc_auc_score(y, pb)),
        "boundary_plus_prox_auc":
            float(roc_auc_score(y, pc)),
        "delta_auc":
            float(roc_auc_score(y, pc) - roc_auc_score(y, pb)),

        "fixed_ap":
            float(average_precision_score(y, pb)),
        "boundary_plus_prox_ap":
            float(average_precision_score(y, pc)),
        "delta_ap":
            float(
                average_precision_score(y, pc)
                - average_precision_score(y, pb)
            ),

        # coefficient after within-task standardization
        "proximity_coef":
            float(mc.coef_[0, 1]),

        "risk_far_Q1":
            float(qrisk.get("Q1_far", np.nan)),
        "risk_Q2":
            float(qrisk.get("Q2", np.nan)),
        "risk_Q3":
            float(qrisk.get("Q3", np.nan)),
        "risk_close_Q4":
            float(qrisk.get("Q4_close", np.nan)),
        "close_vs_far_risk_diff":
            float(
                qrisk.get("Q4_close", np.nan)
                - qrisk.get("Q1_far", np.nan)
            ),
    })

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)

print("\n" + "="*110)
print("PROXIMITY TASK HETEROGENEITY")
print("="*110)
print(res.to_string(index=False))

print("\nInterpretation guide:")
print("proximity_coef > 0 : closer scene-action geometry => higher risk after controlling boundary")
print("close_vs_far_risk_diff > 0 : closest quartile has more high-risk actions than farthest quartile")
print("delta_auc / delta_ap > 0 : proximity adds predictive value within that task")
print("\nOUTPUT =", OUT)
