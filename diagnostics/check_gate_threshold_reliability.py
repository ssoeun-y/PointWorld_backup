from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model"
)
GROOT = ROOT / "uncertainty_gate_5task"
RROOT = GROOT / "robustness"
OUT = ROOT / "reliability_router_final"
OUT.mkdir(parents=True, exist_ok=True)

REJECT_RATES = [0.05, 0.10, 0.20, 0.30]
FAILURE_RATES = [0.10, 0.20]   # 실제 error 상위 10%, 20%

# ============================================================
# Helper
# ============================================================

def evaluate(df, score_col, group_cols, setting, feature, base_pair="OOF"):
    rows = []

    # failure threshold는 task별로 잡음
    for fail_rate in FAILURE_RATES:

        d = df.copy()

        d["failure"] = (
            d.groupby(group_cols)["true_error_mm"]
            .transform(
                lambda x: x >= x.quantile(1.0 - fail_rate)
            )
            .astype(int)
        )

        prevalence = d["failure"].mean()

        for reject_rate in REJECT_RATES:

            rejected_parts = []
            retained_parts = []

            # 각 task/held-out task 내부에서 predicted risk 상위 x% reject
            for _, g in d.groupby(group_cols):

                g = g.sort_values(
                    score_col,
                    ascending=False
                )

                k = max(
                    1,
                    int(round(reject_rate * len(g)))
                )

                rejected_parts.append(g.iloc[:k])
                retained_parts.append(g.iloc[k:])

            rejected = pd.concat(rejected_parts)
            retained = pd.concat(retained_parts)

            tp = int(rejected["failure"].sum())
            fp = int(len(rejected) - tp)

            all_fail = int(d["failure"].sum())
            all_nonfail = int(len(d) - all_fail)

            fn = all_fail - tp
            tn = all_nonfail - fp

            precision = tp / (tp + fp) if (tp + fp) else np.nan
            recall = tp / all_fail if all_fail else np.nan
            fpr = fp / all_nonfail if all_nonfail else np.nan
            fnr = fn / all_fail if all_fail else np.nan

            rows.append({
                "setting": setting,
                "base_pair": base_pair,
                "features": feature,
                "failure_definition":
                    f"top_{int(fail_rate*100)}pct_true_error",
                "failure_prevalence": prevalence,
                "reject_rate": reject_rate,
                "precision_of_reject": precision,
                "failure_recall": recall,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "precision_enrichment_vs_random":
                    precision / prevalence,
                "overall_mean_mm":
                    d["true_error_mm"].mean(),
                "rejected_mean_mm":
                    rejected["true_error_mm"].mean(),
                "retained_mean_mm":
                    retained["true_error_mm"].mean(),
                "overall_p95_mm":
                    d["true_error_mm"].quantile(0.95),
                "retained_p95_mm":
                    retained["true_error_mm"].quantile(0.95),
            })

    return rows


# ============================================================
# 1. Episode-disjoint OOF
# ============================================================

oof = pd.read_csv(
    GROOT / "gate_oof_predictions.csv"
)

oof["task"] = oof["task"].astype(str).str.zfill(4)

all_rows = []

for feat in ["Action-only", "Scene+Action"]:

    g = oof[
        (oof["model"] == "RandomForest")
        & (oof["features"] == feat)
    ].copy()

    g = g.rename(
        columns={
            "base_error_mm": "true_error_mm",
            "predicted_risk": "risk_score",
        }
    )

    all_rows += evaluate(
        g,
        score_col="risk_score",
        group_cols=["task"],
        setting="Episode-disjoint OOF",
        feature=feat,
    )


# ============================================================
# 2. STRICT LOTO — completely unseen task
# ============================================================

loto = pd.read_csv(
    RROOT / "loto_action_vs_sceneaction_predictions.csv"
)

loto["task"] = loto["task"].astype(str).str.zfill(4)

for base in ["HL", "HR"]:

    b = loto[
        loto["base_pair"] == base
    ].copy()

    for feat, col in [
        ("Action-only", "pred_Action-only"),
        ("Scene+Action", "pred_Scene+Action"),
    ]:

        g = b[
            [
                "task",
                "true_error_mm",
                col,
            ]
        ].copy()

        g = g.rename(
            columns={col: "risk_score"}
        )

        all_rows += evaluate(
            g,
            score_col="risk_score",
            group_cols=["task"],
            setting="Strict LOTO unseen-task",
            feature=feat,
            base_pair=base,
        )


# ============================================================
# 3. Risk calibration bins — does higher predicted risk
#    actually correspond to higher true error?
# ============================================================

cal_rows = []

g = oof[
    (oof["model"] == "RandomForest")
    & (oof["features"] == "Action-only")
].copy()

# percentile rank within task avoids task-scale confound
g["risk_percentile"] = (
    g.groupby("task")["predicted_risk"]
    .rank(pct=True)
)

g["risk_decile"] = pd.cut(
    g["risk_percentile"],
    bins=np.linspace(0, 1, 11),
    labels=[f"D{i}" for i in range(1, 11)],
    include_lowest=True,
)

for decile, x in g.groupby("risk_decile", observed=True):

    cal_rows.append({
        "risk_decile": decile,
        "N": len(x),
        "predicted_risk_mean":
            x["predicted_risk"].mean(),
        "true_error_mean_mm":
            x["base_error_mm"].mean(),
        "true_error_median_mm":
            x["base_error_mm"].median(),
        "true_error_p95_mm":
            x["base_error_mm"].quantile(0.95),
    })


summary = pd.DataFrame(all_rows)
cal = pd.DataFrame(cal_rows)

summary.to_csv(
    OUT / "gate_threshold_reliability.csv",
    index=False,
)

cal.to_csv(
    OUT / "gate_risk_calibration_deciles.csv",
    index=False,
)


# ============================================================
# PRINT MAIN NUMBERS
# ============================================================

print("=" * 100)
print("ACTION-ONLY — STRICT LOTO, TRUE FAILURE = TOP 20% ERROR")
print("=" * 100)

show = summary[
    (summary["setting"] == "Strict LOTO unseen-task")
    & (summary["features"] == "Action-only")
    & (
        summary["failure_definition"]
        == "top_20pct_true_error"
    )
]

print(
    show[
        [
            "base_pair",
            "reject_rate",
            "precision_of_reject",
            "failure_recall",
            "false_positive_rate",
            "false_negative_rate",
            "precision_enrichment_vs_random",
            "retained_mean_mm",
            "retained_p95_mm",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 100)
print("ACTION-ONLY RISK DECILES")
print("=" * 100)

print(cal.to_string(index=False))

print("\nOUTPUT =", OUT)
print("SUCCESS = True")
