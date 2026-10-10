from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model"
)

OUT = ROOT / "uncertainty_gate_5task"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["0000", "0001", "0006", "0008", "0010"]

EXPECTED = {
    "0000": 383,
    "0001": 2312,
    "0006": 1855,
    "0008": 1824,
    "0010": 1614,
}

# ============================================================
# LOAD DATA
# ============================================================

frames = []

print("=" * 100)
print("LOAD / AUDIT")
print("=" * 100)

for task in TASKS:

    d = ROOT / f"fixed_camera_pair_task{task}"

    hl = pd.read_csv(
        d / f"task{task}_pair_HL__per_sample.csv"
    )

    hr = pd.read_csv(
        d / f"task{task}_pair_HR__per_sample.csv"
    )

    feat = pd.read_csv(
        d / f"task{task}_head_scene_action_query_features.csv"
    )

    metric = (
        "behavior/l2_moved/mean"
        if "behavior/l2_moved/mean" in hl.columns
        else "l2_moved/mean"
    )

    pair = (
        hl[["clip", metric]]
        .rename(columns={metric: "err_HL"})
        .merge(
            hr[["clip", metric]].rename(
                columns={metric: "err_HR"}
            ),
            on="clip",
            validate="one_to_one",
        )
    )

    pair["err_HL_mm"] = pair["err_HL"] * 1000.0
    pair["err_HR_mm"] = pair["err_HR"] * 1000.0

    pair = pair.dropna(
        subset=["err_HL_mm", "err_HR_mm"]
    )

    feat = feat.copy()
    feat["task"] = task

    df = pair.merge(
        feat,
        on="clip",
        how="inner",
        validate="one_to_one",
    )

    df["task"] = task

    print(
        f"task-{task}: "
        f"N={len(df)} | "
        f"expected={EXPECTED[task]} | "
        f"{'OK' if len(df) == EXPECTED[task] else 'MISMATCH'}"
    )

    frames.append(df)

data = pd.concat(frames, ignore_index=True)

assert len(data) == sum(EXPECTED.values())

print("-" * 100)
print("POOLED N =", len(data))

# Make episode identity unambiguously task-specific.
data["group_id"] = (
    data["task"].astype(str)
    + "::"
    + data["episode"].astype(str)
)

print("GROUPS =", data["group_id"].nunique())

# ============================================================
# FEATURES
# ============================================================

scene_cols = sorted([
    c for c in data.columns
    if c.startswith("scene_")
])

action_cols = sorted([
    c for c in data.columns
    if c.startswith("action_")
])

feature_sets = {
    "Scene-only": scene_cols,
    "Action-only": action_cols,
    "Scene+Action": scene_cols + action_cols,
}

print("Scene features =", len(scene_cols))
print("Action features =", len(action_cols))

# ============================================================
# EPISODE-DISJOINT, TASK-BALANCED FOLDS
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

folds = list(
    cv.split(
        np.zeros((len(data), 1)),
        data["task"],
        groups=data["group_id"],
    )
)

# ============================================================
# DEFINE CURRENT / ALTERNATIVE VIEW WITHOUT TEST LEAKAGE
#
# Per task, current fixed pair is selected from TRAIN fold only.
# The other pair is treated as the alternative queried observation.
# ============================================================

data["base_error_mm"] = np.nan
data["alt_error_mm"] = np.nan
data["fold"] = -1
data["base_pair"] = ""

for fold_id, (tr, te) in enumerate(folds):

    train = data.iloc[tr]
    test = data.iloc[te]

    assert set(train["group_id"]).isdisjoint(
        set(test["group_id"])
    )

    data.loc[data.index[te], "fold"] = fold_id

    for task in TASKS:

        tr_task = train[train["task"] == task]

        te_mask_local = (
            test["task"].to_numpy() == task
        )

        te_indices = test.index[te_mask_local]

        if len(te_indices) == 0:
            continue

        mean_hl = tr_task["err_HL_mm"].mean()
        mean_hr = tr_task["err_HR_mm"].mean()

        if mean_hl <= mean_hr:
            base_pair = "HL"
            base = data.loc[te_indices, "err_HL_mm"]
            alt = data.loc[te_indices, "err_HR_mm"]
        else:
            base_pair = "HR"
            base = data.loc[te_indices, "err_HR_mm"]
            alt = data.loc[te_indices, "err_HL_mm"]

        data.loc[te_indices, "base_pair"] = base_pair
        data.loc[te_indices, "base_error_mm"] = base
        data.loc[te_indices, "alt_error_mm"] = alt

assert data["base_error_mm"].notna().all()
assert data["alt_error_mm"].notna().all()

# If alternative is better, this is the maximum recoverable benefit
# from querying the other observation.
data["query_benefit_mm"] = np.maximum(
    data["base_error_mm"]
    - data["alt_error_mm"],
    0.0
)

# Paired oracle after having access to the alternative observation.
data["queried_oracle_error_mm"] = np.minimum(
    data["base_error_mm"],
    data["alt_error_mm"],
)

base_mean = data["base_error_mm"].mean()
oracle_mean = data["queried_oracle_error_mm"].mean()

print("\n" + "=" * 100)
print("OBSERVATION-INTERVENTION HEADROOM")
print("=" * 100)

print(f"Fixed current-view error : {base_mean:.4f} mm")
print(f"Full paired oracle error : {oracle_mean:.4f} mm")
print(
    "Maximum recoverable gain:",
    f"{base_mean - oracle_mean:.4f} mm "
    f"({100*(base_mean-oracle_mean)/base_mean:.3f}%)"
)

benefit_positive = (
    data["query_benefit_mm"] > 0
).mean()

print(
    "Alternative view helps:",
    f"{benefit_positive*100:.2f}% of samples"
)

rho_risk_benefit, p_risk_benefit = spearmanr(
    data["base_error_mm"],
    data["query_benefit_mm"],
)

print(
    "True current error vs query benefit:",
    f"rho={rho_risk_benefit:+.4f}, "
    f"p={p_risk_benefit:.3e}"
)

# ============================================================
# ERROR / UNCERTAINTY GATE
#
# This is NOT HL-vs-HR classification.
# Target = scalar current prediction error.
# ============================================================

models = {
    "Ridge": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]),
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    ),
}

summary = []
pred_tables = []

y = data["base_error_mm"].to_numpy()

print("\n" + "=" * 100)
print("UNCERTAINTY / ERROR GATE")
print("=" * 100)

for model_name, make_model in models.items():

    for feature_name, cols in feature_sets.items():

        X = (
            data[cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        pred = np.full(len(data), np.nan)

        for tr, te in folds:

            model = make_model()

            model.fit(
                X.iloc[tr],
                y[tr],
            )

            pred[te] = model.predict(
                X.iloc[te]
            )

        mae = mean_absolute_error(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        r2 = r2_score(y, pred)
        rho, p = spearmanr(y, pred)

        print(
            f"{model_name:14s} | "
            f"{feature_name:12s} | "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"R2={r2:+.4f} | "
            f"rho={rho:+.4f} | "
            f"p={p:.3e}"
        )

        summary.append({
            "model": model_name,
            "features": feature_name,
            "MAE_mm": mae,
            "RMSE_mm": rmse,
            "R2": r2,
            "Spearman_rho": rho,
            "Spearman_p": p,
        })

        temp = data[
            [
                "clip",
                "episode",
                "task",
                "group_id",
                "fold",
                "base_pair",
                "base_error_mm",
                "alt_error_mm",
                "query_benefit_mm",
            ]
        ].copy()

        temp["model"] = model_name
        temp["features"] = feature_name
        temp["predicted_risk"] = pred

        pred_tables.append(temp)

# ============================================================
# QUERY-RATE CURVES
#
# For queried samples:
#   use min(base, alt)
#
# This intentionally gives the query stage an ORACLE chooser.
# Why?
#   We are isolating "WHEN TO QUERY" first.
#   Previous weak L/R selector is not allowed to contaminate gate test.
#
# Curves:
#   Random
#   True-Risk Oracle
#   Benefit Oracle
#   Predicted Gate
# ============================================================

RATES = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]

curves = []

def evaluate_ranking(score, name, model="", features=""):

    n = len(data)

    order = np.argsort(-np.asarray(score))

    base = data["base_error_mm"].to_numpy()
    queried = data["queried_oracle_error_mm"].to_numpy()
    benefit = data["query_benefit_mm"].to_numpy()

    for rate in RATES:

        k = int(round(rate * n))

        final = base.copy()

        if k > 0:
            idx = order[:k]
            final[idx] = queried[idx]

            hit_rate = np.mean(
                benefit[idx] > 0
            )

            avg_benefit_queried = np.mean(
                benefit[idx]
            )
        else:
            hit_rate = np.nan
            avg_benefit_queried = np.nan

        curves.append({
            "ranking": name,
            "model": model,
            "features": features,
            "query_rate": rate,
            "query_count": k,
            "post_query_error_mm": final.mean(),
            "relative_improvement_pct":
                100.0 * (base.mean() - final.mean())
                / base.mean(),
            "queried_hit_rate": hit_rate,
            "avg_benefit_queried_mm":
                avg_benefit_queried,
        })


# True error = perfect uncertainty gate.
evaluate_ranking(
    data["base_error_mm"].to_numpy(),
    "True-Risk Oracle",
)

# True benefit = absolute upper bound on which samples to query.
evaluate_ranking(
    data["query_benefit_mm"].to_numpy(),
    "Benefit Oracle",
)

# Expected random-query curve.
mean_benefit = data["query_benefit_mm"].mean()

for rate in RATES:

    expected_error = (
        base_mean
        - rate * mean_benefit
    )

    curves.append({
        "ranking": "Random Expected",
        "model": "",
        "features": "",
        "query_rate": rate,
        "query_count": int(round(rate * len(data))),
        "post_query_error_mm": expected_error,
        "relative_improvement_pct":
            100.0 * (base_mean - expected_error)
            / base_mean,
        "queried_hit_rate": benefit_positive,
        "avg_benefit_queried_mm": mean_benefit,
    })

# Learned gates.
pred_df = pd.concat(
    pred_tables,
    ignore_index=True,
)

for (model_name, feature_name), g in pred_df.groupby(
    ["model", "features"],
    sort=False,
):

    # restore original row order
    g = g.reset_index(drop=True)

    evaluate_ranking(
        g["predicted_risk"].to_numpy(),
        "Predicted Gate",
        model=model_name,
        features=feature_name,
    )

curves_df = pd.DataFrame(curves)

# ============================================================
# TRUE ERROR QUINTILES -> QUERY BENEFIT
#
# Directly answers:
# "Are high-error samples actually helped more by another view?"
# ============================================================

data["risk_quintile"] = pd.qcut(
    data["base_error_mm"],
    5,
    labels=[
        "Q1-low",
        "Q2",
        "Q3",
        "Q4",
        "Q5-high",
    ],
)

risk_benefit = (
    data.groupby(
        "risk_quintile",
        observed=True,
    )
    .agg(
        N=("clip", "size"),
        base_error_mm=("base_error_mm", "mean"),
        mean_query_benefit_mm=("query_benefit_mm", "mean"),
        median_query_benefit_mm=("query_benefit_mm", "median"),
        query_helps_fraction=(
            "query_benefit_mm",
            lambda x: np.mean(x > 0),
        ),
    )
    .reset_index()
)

print("\n" + "=" * 100)
print("TRUE ERROR -> QUERY BENEFIT BY QUINTILE")
print("=" * 100)

print(risk_benefit.to_string(index=False))

# ============================================================
# PRINT QUERY CURVE: RF ONLY FOR QUICK READING
# ============================================================

print("\n" + "=" * 100)
print("QUERY-RATE CURVE — RANDOM FOREST GATES")
print("=" * 100)

show = curves_df[
    (
        curves_df["ranking"].isin([
            "Random Expected",
            "True-Risk Oracle",
            "Benefit Oracle",
        ])
    )
    |
    (
        (curves_df["ranking"] == "Predicted Gate")
        &
        (curves_df["model"] == "RandomForest")
    )
].copy()

print(
    show[
        [
            "ranking",
            "features",
            "query_rate",
            "post_query_error_mm",
            "relative_improvement_pct",
            "queried_hit_rate",
        ]
    ].to_string(index=False)
)

# ============================================================
# SAVE
# ============================================================

pd.DataFrame(summary).to_csv(
    OUT / "gate_regression_summary.csv",
    index=False,
)

pred_df.to_csv(
    OUT / "gate_oof_predictions.csv",
    index=False,
)

curves_df.to_csv(
    OUT / "query_rate_curves.csv",
    index=False,
)

risk_benefit.to_csv(
    OUT / "true_error_vs_query_benefit_quintiles.csv",
    index=False,
)

data.to_csv(
    OUT / "gate_analysis_input.csv",
    index=False,
)

print("\nOUTPUT =", OUT)
print("SUCCESS = True")
