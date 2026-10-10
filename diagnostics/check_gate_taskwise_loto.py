from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model/"
    "uncertainty_gate_5task"
)
OUT = ROOT / "robustness"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["0000", "0001", "0006", "0008", "0010"]
RATES = [0.10, 0.20, 0.30]

# ============================================================
# 1. TASKWISE BREAKDOWN OF EXISTING EPISODE-DISJOINT OOF GATE
# ============================================================

oof = pd.read_csv(ROOT / "gate_oof_predictions.csv")

# Primary model from Stage 1
oof = oof[oof["model"] == "RandomForest"].copy()

taskwise_rows = []
taskwise_curves = []

print("=" * 100)
print("TASKWISE OOF — RANDOM FOREST")
print("=" * 100)

for feat in ["Scene-only", "Action-only", "Scene+Action"]:

    sub = oof[oof["features"] == feat].copy()

    for task in TASKS:

        g = sub[sub["task"].astype(str).str.zfill(4) == task].copy()

        y = g["base_error_mm"].to_numpy()
        p = g["predicted_risk"].to_numpy()
        benefit = g["query_benefit_mm"].to_numpy()

        rho, pv = spearmanr(y, p)
        mae = mean_absolute_error(y, p)
        r2 = r2_score(y, p)

        taskwise_rows.append({
            "task": task,
            "features": feat,
            "N": len(g),
            "MAE_mm": mae,
            "R2": r2,
            "Spearman_rho": rho,
            "Spearman_p": pv,
        })

        print(
            f"task-{task} | {feat:12s} | "
            f"N={len(g):4d} | "
            f"rho={rho:+.4f} | R2={r2:+.4f} | MAE={mae:.3f}"
        )

        order = np.argsort(-p)
        base_mean = y.mean()

        for rate in RATES:
            k = max(1, int(round(rate * len(g))))
            idx = order[:k]

            # Gate-only feasibility:
            # queried sample receives oracle better of current/alternative.
            final = y.copy()
            final[idx] = y[idx] - benefit[idx]

            learned_gain = 100 * (base_mean - final.mean()) / base_mean

            random_expected = (
                100 * rate * benefit.mean() / base_mean
            )

            taskwise_curves.append({
                "task": task,
                "features": feat,
                "query_rate": rate,
                "learned_gate_gain_pct": learned_gain,
                "random_expected_gain_pct": random_expected,
                "advantage_over_random_pctpt":
                    learned_gain - random_expected,
            })

print()

# ============================================================
# 2. STRICT LEAVE-ONE-TASK-OUT
#
# Train risk predictor on 4 tasks.
# Test on completely unseen fifth task.
#
# Run separately:
#   baseline = HL, query alternative HR
#   baseline = HR, query alternative HL
#
# This avoids choosing the base view using held-out task data.
# ============================================================

data = pd.read_csv(ROOT / "gate_analysis_input.csv")

data["task"] = data["task"].astype(str).str.zfill(4)

scene_cols = sorted(
    c for c in data.columns if c.startswith("scene_")
)
action_cols = sorted(
    c for c in data.columns if c.startswith("action_")
)

feature_sets = {
    "Scene-only": scene_cols,
    "Action-only": action_cols,
    "Scene+Action": scene_cols + action_cols,
}

loto_rows = []
loto_curves = []

print("=" * 100)
print("STRICT LEAVE-ONE-TASK-OUT")
print("=" * 100)

for base_pair in ["HL", "HR"]:

    base_col = f"err_{base_pair}_mm"
    alt_pair = "HR" if base_pair == "HL" else "HL"
    alt_col = f"err_{alt_pair}_mm"

    print(f"\nBASE={base_pair} -> ALTERNATIVE={alt_pair}")

    for heldout in TASKS:

        train = data[data["task"] != heldout].copy()
        test = data[data["task"] == heldout].copy()

        y_train = train[base_col].to_numpy()
        y_test = test[base_col].to_numpy()

        alt_test = test[alt_col].to_numpy()

        benefit = np.maximum(
            y_test - alt_test,
            0.0
        )

        for feat_name, cols in feature_sets.items():

            Xtr = (
                train[cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            Xte = (
                test[cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            model = RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=5,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            )

            model.fit(Xtr, y_train)
            pred = model.predict(Xte)

            rho, pv = spearmanr(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)

            loto_rows.append({
                "heldout_task": heldout,
                "base_pair": base_pair,
                "features": feat_name,
                "N": len(test),
                "MAE_mm": mae,
                "R2": r2,
                "Spearman_rho": rho,
                "Spearman_p": pv,
            })

            print(
                f"heldout={heldout} | "
                f"{feat_name:12s} | "
                f"rho={rho:+.4f} | "
                f"R2={r2:+.4f} | "
                f"MAE={mae:.3f}"
            )

            order = np.argsort(-pred)
            base_mean = y_test.mean()

            for rate in RATES:
                k = max(
                    1,
                    int(round(rate * len(test)))
                )

                idx = order[:k]

                final = y_test.copy()

                # Gate-only feasibility / oracle downstream selector.
                final[idx] = (
                    y_test[idx] - benefit[idx]
                )

                gate_gain = (
                    100
                    * (base_mean - final.mean())
                    / base_mean
                )

                random_gain = (
                    100
                    * rate
                    * benefit.mean()
                    / base_mean
                )

                loto_curves.append({
                    "heldout_task": heldout,
                    "base_pair": base_pair,
                    "features": feat_name,
                    "query_rate": rate,
                    "gate_gain_pct": gate_gain,
                    "random_expected_gain_pct": random_gain,
                    "advantage_over_random_pctpt":
                        gate_gain - random_gain,
                })

# ============================================================
# SUMMARIES
# ============================================================

taskwise_df = pd.DataFrame(taskwise_rows)
taskwise_curve_df = pd.DataFrame(taskwise_curves)
loto_df = pd.DataFrame(loto_rows)
loto_curve_df = pd.DataFrame(loto_curves)

print("\n" + "=" * 100)
print("TASKWISE MACRO SUMMARY")
print("=" * 100)

print(
    taskwise_df.groupby("features")[
        ["Spearman_rho", "R2"]
    ].mean().to_string()
)

print("\n" + "=" * 100)
print("LOTO MACRO SUMMARY")
print("=" * 100)

print(
    loto_df.groupby(
        ["base_pair", "features"]
    )[["Spearman_rho", "R2"]]
    .mean()
    .to_string()
)

print("\n" + "=" * 100)
print("LOTO QUERY-RATE SUMMARY")
print("=" * 100)

print(
    loto_curve_df.groupby(
        ["base_pair", "features", "query_rate"]
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

taskwise_df.to_csv(
    OUT / "taskwise_gate_metrics.csv",
    index=False,
)

taskwise_curve_df.to_csv(
    OUT / "taskwise_query_curves.csv",
    index=False,
)

loto_df.to_csv(
    OUT / "loto_gate_metrics.csv",
    index=False,
)

loto_curve_df.to_csv(
    OUT / "loto_query_curves.csv",
    index=False,
)

print("\nOUTPUT =", OUT)
print("SUCCESS = True")
