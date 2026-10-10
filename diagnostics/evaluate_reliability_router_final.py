from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
)

ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model"
)

GATE_ROOT = ROOT / "uncertainty_gate_5task"
OUT = ROOT / "reliability_router_final"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["0000", "0001", "0006", "0008", "0010"]
QUERY_RATES = [0.10, 0.20, 0.30]
REJECT_RATES = [0.00, 0.10, 0.20, 0.30, 0.50]

# ============================================================
# LOAD EXISTING DATA
# ============================================================

data = pd.read_csv(
    GATE_ROOT / "gate_analysis_input.csv"
)

data["task"] = (
    data["task"].astype(str).str.zfill(4)
)

data["group_id"] = (
    data["task"].astype(str)
    + "::"
    + data["episode"].astype(str)
)

# HLR
hlr_frames = []

for task in TASKS:

    f = (
        ROOT /
        f"fixed_camera_pair_task{task}" /
        f"task{task}_pair_HLR__per_sample.csv"
    )

    h = pd.read_csv(f)

    metric = (
        "behavior/l2_moved/mean"
        if "behavior/l2_moved/mean" in h.columns
        else "l2_moved/mean"
    )

    h = h[["clip", metric]].copy()
    h["task"] = task
    h["hlr_error_mm"] = h[metric] * 1000.0

    hlr_frames.append(
        h[["task", "clip", "hlr_error_mm"]]
    )

hlr = pd.concat(
    hlr_frames,
    ignore_index=True,
)

data = data.merge(
    hlr,
    on=["task", "clip"],
    validate="one_to_one",
)

assert len(data) == 7988

# IMPORTANT:
# signed benefit. Negative values are retained.
data["signed_benefit_mm"] = (
    data["base_error_mm"]
    - data["hlr_error_mm"]
)

scene_cols = sorted([
    c for c in data.columns
    if c.startswith("scene_")
])

action_cols = sorted([
    c for c in data.columns
    if c.startswith("action_")
])

FEATURES = {
    "Scene-only": scene_cols,
    "Action-only": action_cols,
    "Scene+Action": scene_cols + action_cols,
}

# Reuse EXACT previous OOF folds.
fold_ids = sorted(
    data["fold"].astype(int).unique()
)

folds = []

for f in fold_ids:
    te = np.where(
        data["fold"].astype(int).to_numpy() == f
    )[0]

    tr = np.where(
        data["fold"].astype(int).to_numpy() != f
    )[0]

    assert set(
        data.iloc[tr]["group_id"]
    ).isdisjoint(
        set(data.iloc[te]["group_id"])
    )

    folds.append((tr, te))


def rf():
    return RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )


# ============================================================
# PART A
# FAILURE DETECTION / SELECTIVE PREDICTION
#
# Can the router know when NOT to trust PointWorld?
# ============================================================

existing = pd.read_csv(
    GATE_ROOT / "gate_oof_predictions.csv"
)

existing["task"] = (
    existing["task"].astype(str).str.zfill(4)
)

existing = existing[
    existing["model"] == "RandomForest"
].copy()

risk_summary = []
risk_curves = []

print("=" * 100)
print("A. FAILURE DETECTION / RISK ROUTING")
print("=" * 100)

for feat in [
    "Scene-only",
    "Action-only",
    "Scene+Action",
]:

    g = existing[
        existing["features"] == feat
    ].copy()

    # top 20% error WITHIN EACH TASK = failure
    g["failure"] = (
        g.groupby("task")["base_error_mm"]
        .transform(
            lambda x: x >= x.quantile(0.80)
        )
        .astype(int)
    )

    auc = roc_auc_score(
        g["failure"],
        g["predicted_risk"]
    )

    ap = average_precision_score(
        g["failure"],
        g["predicted_risk"]
    )

    rho = spearmanr(
        g["base_error_mm"],
        g["predicted_risk"]
    ).statistic

    risk_summary.append({
        "features": feat,
        "Spearman_rho": rho,
        "failure_AUROC": auc,
        "failure_AUPRC": ap,
    })

    print(
        f"{feat:12s} | "
        f"rho={rho:+.4f} | "
        f"AUROC={auc:.4f} | "
        f"AUPRC={ap:.4f}"
    )

    # risk-coverage:
    # reject high predicted-risk predictions.
    order = np.argsort(
        g["predicted_risk"].to_numpy()
    )

    err = g["base_error_mm"].to_numpy()

    for reject in REJECT_RATES:

        keep_n = int(
            round((1.0 - reject) * len(g))
        )

        keep = order[:keep_n]

        retained_error = err[keep].mean()
        retained_p95 = np.quantile(
            err[keep],
            0.95,
        )

        risk_curves.append({
            "features": feat,
            "reject_rate": reject,
            "coverage": 1.0 - reject,
            "retained_moved_l2_mm":
                retained_error,
            "retained_p95_mm":
                retained_p95,
        })


# ============================================================
# PART B
# VALUE OF ADDITIONAL OBSERVATION
#
# Predict SIGNED 2-view -> HLR benefit directly.
# ============================================================

y = data["signed_benefit_mm"].to_numpy()

benefit_summary = []
taskwise_summary = []
predictions = {}

print("\n" + "=" * 100)
print("B. SIGNED BENEFIT PREDICTION — POOLED OOF")
print("=" * 100)

for feat, cols in FEATURES.items():

    X = (
        data[cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    pred = np.full(
        len(data),
        np.nan,
    )

    for tr, te in folds:

        model = rf()

        model.fit(
            X.iloc[tr],
            y[tr],
        )

        pred[te] = model.predict(
            X.iloc[te]
        )

    predictions[feat] = pred

    rho, pv = spearmanr(
        y,
        pred,
    )

    mae = mean_absolute_error(
        y,
        pred,
    )

    r2 = r2_score(
        y,
        pred,
    )

    benefit_summary.append({
        "features": feat,
        "N": len(data),
        "Spearman_rho": rho,
        "Spearman_p": pv,
        "MAE_mm": mae,
        "R2": r2,
    })

    print(
        f"{feat:12s} | "
        f"rho={rho:+.4f} | "
        f"R2={r2:+.4f} | "
        f"MAE={mae:.4f}"
    )

    # TASKWISE
    for task in TASKS:

        idx = (
            data["task"].to_numpy()
            == task
        )

        rt, pt = spearmanr(
            y[idx],
            pred[idx],
        )

        taskwise_summary.append({
            "task": task,
            "features": feat,
            "N": int(idx.sum()),
            "Spearman_rho": rt,
            "Spearman_p": pt,
            "mean_true_signed_benefit_mm":
                y[idx].mean(),
        })


print("\n" + "=" * 100)
print("TASKWISE SIGNED-BENEFIT RHO")
print("=" * 100)

taskwise_df = pd.DataFrame(
    taskwise_summary
)

print(
    taskwise_df.pivot(
        index="task",
        columns="features",
        values="Spearman_rho",
    ).to_string()
)


# ============================================================
# PART C
# ACTUAL QUERY-RATE CURVE
#
# Top predicted-benefit samples ACTUALLY switch to HLR.
# ============================================================

base = data["base_error_mm"].to_numpy()
h3 = data["hlr_error_mm"].to_numpy()
benefit = data["signed_benefit_mm"].to_numpy()

base_mean = base.mean()

query_rows = []

def query_curve(method, score):

    order = np.argsort(
        -np.asarray(score)
    )

    for rate in QUERY_RATES:

        k = int(
            round(rate * len(data))
        )

        idx = order[:k]

        final = base.copy()

        # REAL HLR result.
        # No oracle min() fallback.
        final[idx] = h3[idx]

        gain = (
            100.0
            * (base_mean - final.mean())
            / base_mean
        )

        positive = np.mean(
            benefit[idx] > 0
        )

        query_rows.append({
            "method": method,
            "query_rate": rate,
            "final_moved_l2_mm":
                final.mean(),
            "improvement_pct": gain,
            "selected_positive_fraction":
                positive,
            "mean_selected_signed_benefit_mm":
                benefit[idx].mean(),
        })


# Learned benefit gates
for feat in FEATURES:
    query_curve(
        f"Benefit Gate: {feat}",
        predictions[feat],
    )

# Existing Action risk gate
risk_gate = existing[
    existing["features"] == "Action-only"
].copy()

risk_gate = (
    risk_gate[
        ["task", "clip", "predicted_risk"]
    ]
    .merge(
        data[["task", "clip"]],
        on=["task", "clip"],
        validate="one_to_one",
    )
)

# Restore exact data ordering
risk_lookup = existing[
    existing["features"] == "Action-only"
].set_index(
    ["task", "clip"]
)["predicted_risk"]

risk_scores = np.array([
    risk_lookup.loc[(r.task, r.clip)]
    for r in data.itertuples()
])

query_curve(
    "Old Action-Risk Gate",
    risk_scores,
)

# Benefit oracle
query_curve(
    "Benefit Oracle",
    benefit,
)

# Random expected
mean_benefit = benefit.mean()

for rate in QUERY_RATES:

    expected_final = (
        base_mean
        - rate * mean_benefit
    )

    query_rows.append({
        "method": "Random Expected",
        "query_rate": rate,
        "final_moved_l2_mm":
            expected_final,
        "improvement_pct":
            100.0
            * (base_mean - expected_final)
            / base_mean,
        "selected_positive_fraction":
            np.mean(benefit > 0),
        "mean_selected_signed_benefit_mm":
            mean_benefit,
    })


query_df = pd.DataFrame(
    query_rows
)

print("\n" + "=" * 100)
print("C. ACTUAL 2-VIEW -> HLR QUERY CURVE")
print("=" * 100)

print(
    query_df[
        [
            "method",
            "query_rate",
            "final_moved_l2_mm",
            "improvement_pct",
            "selected_positive_fraction",
        ]
    ]
    .sort_values(
        ["query_rate", "improvement_pct"],
        ascending=[True, False],
    )
    .to_string(index=False)
)


# ============================================================
# PART D
# STRICT LEAVE-ONE-TASK-OUT BENEFIT GENERALIZATION
#
# Use HL and HR separately to avoid held-out-task
# baseline-selection leakage.
# ============================================================

loto_rows = []
loto_query_rows = []

print("\n" + "=" * 100)
print("D. STRICT LOTO SIGNED-BENEFIT GENERALIZATION")
print("=" * 100)

for base_pair in ["HL", "HR"]:

    base_col = f"err_{base_pair}_mm"

    print(f"\nBASE = {base_pair}")

    for heldout in TASKS:

        train = (
            data["task"] != heldout
        )

        test = (
            data["task"] == heldout
        )

        y_train = (
            data.loc[train, base_col].to_numpy()
            - data.loc[train, "hlr_error_mm"].to_numpy()
        )

        y_test = (
            data.loc[test, base_col].to_numpy()
            - data.loc[test, "hlr_error_mm"].to_numpy()
        )

        base_test = data.loc[
            test,
            base_col
        ].to_numpy()

        h3_test = data.loc[
            test,
            "hlr_error_mm"
        ].to_numpy()

        for feat, cols in FEATURES.items():

            Xtr = (
                data.loc[train, cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            Xte = (
                data.loc[test, cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            model = rf()
            model.fit(
                Xtr,
                y_train,
            )

            pred = model.predict(
                Xte
            )

            rho, pv = spearmanr(
                y_test,
                pred,
            )

            loto_rows.append({
                "base_pair": base_pair,
                "heldout_task": heldout,
                "features": feat,
                "N": int(test.sum()),
                "Spearman_rho": rho,
                "Spearman_p": pv,
                "mean_signed_benefit_mm":
                    y_test.mean(),
            })

            print(
                f"heldout={heldout} | "
                f"{feat:12s} | "
                f"rho={rho:+.4f}"
            )

            order = np.argsort(-pred)

            for rate in QUERY_RATES:

                k = max(
                    1,
                    int(
                        round(
                            rate
                            * len(base_test)
                        )
                    )
                )

                idx = order[:k]

                final = base_test.copy()
                final[idx] = h3_test[idx]

                gain = (
                    100.0
                    * (
                        base_test.mean()
                        - final.mean()
                    )
                    / base_test.mean()
                )

                random_gain = (
                    100.0
                    * rate
                    * y_test.mean()
                    / base_test.mean()
                )

                loto_query_rows.append({
                    "base_pair": base_pair,
                    "heldout_task": heldout,
                    "features": feat,
                    "query_rate": rate,
                    "gate_gain_pct": gain,
                    "random_expected_gain_pct":
                        random_gain,
                    "advantage_over_random_pctpt":
                        gain - random_gain,
                })


# ============================================================
# SAVE + COMPACT FINAL SUMMARY
# ============================================================

risk_summary_df = pd.DataFrame(
    risk_summary
)

risk_curves_df = pd.DataFrame(
    risk_curves
)

benefit_summary_df = pd.DataFrame(
    benefit_summary
)

loto_df = pd.DataFrame(
    loto_rows
)

loto_query_df = pd.DataFrame(
    loto_query_rows
)

risk_summary_df.to_csv(
    OUT / "failure_detection_summary.csv",
    index=False,
)

risk_curves_df.to_csv(
    OUT / "risk_coverage_curve.csv",
    index=False,
)

benefit_summary_df.to_csv(
    OUT / "benefit_prediction_summary.csv",
    index=False,
)

taskwise_df.to_csv(
    OUT / "benefit_prediction_taskwise.csv",
    index=False,
)

query_df.to_csv(
    OUT / "benefit_query_curve.csv",
    index=False,
)

loto_df.to_csv(
    OUT / "benefit_loto_summary.csv",
    index=False,
)

loto_query_df.to_csv(
    OUT / "benefit_loto_query_curve.csv",
    index=False,
)

print("\n" + "=" * 100)
print("LOTO MACRO BENEFIT PREDICTION")
print("=" * 100)

print(
    loto_df.groupby(
        ["base_pair", "features"]
    )["Spearman_rho"]
    .mean()
    .to_string()
)

print("\n" + "=" * 100)
print("LOTO QUERY 30%")
print("=" * 100)

print(
    loto_query_df[
        loto_query_df["query_rate"] == 0.30
    ]
    .groupby(
        ["base_pair", "features"]
    )[
        [
            "gate_gain_pct",
            "random_expected_gain_pct",
            "advantage_over_random_pctpt",
        ]
    ]
    .mean()
    .to_string()
)

print("\nOUTPUT =", OUT)
print("SUCCESS = True")
