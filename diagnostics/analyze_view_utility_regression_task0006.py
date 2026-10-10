from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model/"
    "fixed_camera_pair_task0006"
)

PAIR_CSV = ROOT / "task0006_three_pair_merged_analysis.csv"
FEATURE_CSV = ROOT / "task0006_head_scene_action_query_features.csv"

OUT_RESULT = ROOT / "task0006_view_utility_regression.csv"
OUT_OOF = ROOT / "task0006_view_utility_regression_oof.csv"

# ============================================================
# Load
# ============================================================

pair = pd.read_csv(PAIR_CSV)
feat = pd.read_csv(FEATURE_CSV)

# Only need both official PointWorld errors.
pair = pair.dropna(
    subset=["err_HL_mm", "err_HR_mm"]
).copy()

df = pair.merge(
    feat,
    on="clip",
    how="inner",
    validate="one_to_one",
)

# Utility:
# delta > 0  => HR error > HL error => LEFT query is better
# delta < 0  => RIGHT query is better
df["delta_view_mm"] = (
    df["err_HR_mm"] -
    df["err_HL_mm"]
)

print("=" * 100)
print("VIEW UTILITY REGRESSION")
print("=" * 100)

print("N =", len(df))
print("EPISODES =", df["episode"].nunique())

print(
    "Delta median =",
    f"{df['delta_view_mm'].median():.4f} mm"
)

print(
    "Delta mean =",
    f"{df['delta_view_mm'].mean():.4f} mm"
)

print(
    "|Delta| median =",
    f"{df['delta_view_mm'].abs().median():.4f} mm"
)

print(
    "|Delta| mean =",
    f"{df['delta_view_mm'].abs().mean():.4f} mm"
)

# ============================================================
# Feature sets
# ============================================================

scene_cols = [
    c for c in df.columns
    if c.startswith("scene_")
]

action_cols = [
    c for c in df.columns
    if c.startswith("action_")
]

feature_sets = {
    "Scene-only": scene_cols,
    "Action-only": action_cols,
    "Scene+Action": scene_cols + action_cols,
}

print("\nScene features =", len(scene_cols))
print("Action features =", len(action_cols))

# ============================================================
# Models
#
# No hyperparameter sweep.
# One simple linear model + one generic nonlinear model.
# ============================================================

models = {
    "Ridge": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ]),

    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    ),
}

y = df["delta_view_mm"].to_numpy()
groups = df["episode"].to_numpy()

# Episode-disjoint folds.
cv = GroupKFold(n_splits=5)
folds = list(
    cv.split(
        np.zeros((len(df), 1)),
        y,
        groups,
    )
)

results = []
oof_tables = []

# ============================================================
# Mean-delta baseline
# ============================================================

baseline_pred = np.zeros(len(df), dtype=float)
baseline_selected = np.zeros(len(df), dtype=float)
baseline_fixed = np.zeros(len(df), dtype=float)
baseline_oracle = np.zeros(len(df), dtype=float)

for fold_id, (tr, te) in enumerate(folds, start=1):

    train_delta_mean = float(
        np.mean(y[tr])
    )

    baseline_pred[te] = train_delta_mean

    train_HL = df.iloc[tr]["err_HL_mm"].mean()
    train_HR = df.iloc[tr]["err_HR_mm"].mean()

    # Fixed pair chosen only on training data.
    fixed_query = (
        "L"
        if train_HL < train_HR
        else "R"
    )

    test_df = df.iloc[te]

    # Mean-delta predictor decision
    if train_delta_mean > 0:
        baseline_selected[te] = test_df["err_HL_mm"].to_numpy()
    else:
        baseline_selected[te] = test_df["err_HR_mm"].to_numpy()

    if fixed_query == "L":
        baseline_fixed[te] = test_df["err_HL_mm"].to_numpy()
    else:
        baseline_fixed[te] = test_df["err_HR_mm"].to_numpy()

    baseline_oracle[te] = np.minimum(
        test_df["err_HL_mm"].to_numpy(),
        test_df["err_HR_mm"].to_numpy(),
    )

baseline_mae = mean_absolute_error(y, baseline_pred)
baseline_rmse = np.sqrt(
    mean_squared_error(y, baseline_pred)
)

rho, rho_p = spearmanr(
    y,
    baseline_pred
)

print("\n" + "=" * 100)
print("TRAIN-MEAN BASELINE")
print("=" * 100)

print(f"MAE = {baseline_mae:.4f} mm")
print(f"RMSE = {baseline_rmse:.4f} mm")
print(f"R2 = {r2_score(y, baseline_pred):.4f}")
print(f"Spearman = {rho}")

print(
    "Selected moved-L2 =",
    f"{baseline_selected.mean():.4f} mm"
)

print(
    "Fixed moved-L2 =",
    f"{baseline_fixed.mean():.4f} mm"
)

print(
    "Oracle moved-L2 =",
    f"{baseline_oracle.mean():.4f} mm"
)

# ============================================================
# Models
# ============================================================

for model_name, factory in models.items():

    for feature_name, cols in feature_sets.items():

        X = (
            df[cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy()
        )

        pred_oof = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        selected_error = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        fixed_error = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        oracle_error = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        print("\n" + "=" * 100)
        print(model_name, "|", feature_name)
        print("=" * 100)

        for fold_id, (tr, te) in enumerate(
            folds,
            start=1,
        ):

            model = factory()

            model.fit(
                X[tr],
                y[tr],
            )

            pred = model.predict(
                X[te]
            )

            pred_oof[te] = pred

            test_df = df.iloc[te]
            train_df = df.iloc[tr]

            # Predicted utility sign chooses query.
            # predicted delta > 0 => LEFT is better.
            choose_left = pred > 0

            selected = np.where(
                choose_left,
                test_df["err_HL_mm"].to_numpy(),
                test_df["err_HR_mm"].to_numpy(),
            )

            selected_error[te] = selected

            # Best fixed query estimated only from training fold.
            train_HL = train_df["err_HL_mm"].mean()
            train_HR = train_df["err_HR_mm"].mean()

            fixed_query = (
                "L"
                if train_HL < train_HR
                else "R"
            )

            if fixed_query == "L":
                fixed = test_df["err_HL_mm"].to_numpy()
            else:
                fixed = test_df["err_HR_mm"].to_numpy()

            fixed_error[te] = fixed

            oracle_error[te] = np.minimum(
                test_df["err_HL_mm"].to_numpy(),
                test_df["err_HR_mm"].to_numpy(),
            )

            fold_mae = mean_absolute_error(
                y[te],
                pred,
            )

            fold_rho = spearmanr(
                y[te],
                pred,
            ).statistic

            print(
                f"Fold {fold_id}: "
                f"N={len(te)} | "
                f"MAE={fold_mae:.4f} mm | "
                f"Spearman={fold_rho:.3f} | "
                f"selected={selected.mean():.4f} mm | "
                f"fixed={fixed.mean():.4f} mm"
            )

        # ----------------------------------------------------
        # Standard regression metrics
        # ----------------------------------------------------

        mae = mean_absolute_error(
            y,
            pred_oof,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y,
                pred_oof,
            )
        )

        r2 = r2_score(
            y,
            pred_oof,
        )

        rho, rho_p = spearmanr(
            y,
            pred_oof,
        )

        # Direction accuracy is descriptive:
        # Did model choose the lower-error query?
        direction_acc = np.mean(
            np.sign(pred_oof)
            ==
            np.sign(y)
        )

        selected_mean = float(
            selected_error.mean()
        )

        fixed_mean = float(
            fixed_error.mean()
        )

        oracle_mean = float(
            oracle_error.mean()
        )

        gain_vs_fixed = (
            100.0
            * (
                fixed_mean
                - selected_mean
            )
            / fixed_mean
        )

        oracle_gap = (
            100.0
            * (
                selected_mean
                - oracle_mean
            )
            / oracle_mean
        )

        print("\nFINAL:")
        print(f"MAE = {mae:.4f} mm")
        print(f"RMSE = {rmse:.4f} mm")
        print(f"R2 = {r2:.4f}")

        print(
            f"Spearman rho = {rho:.4f}, "
            f"p = {rho_p:.4g}"
        )

        print(
            "Correct query direction =",
            f"{direction_acc*100:.2f}%"
        )

        print(
            "Official moved-L2 selected =",
            f"{selected_mean:.4f} mm"
        )

        print(
            "Official moved-L2 fixed =",
            f"{fixed_mean:.4f} mm"
        )

        print(
            "Official moved-L2 oracle =",
            f"{oracle_mean:.4f} mm"
        )

        print(
            "Improvement vs fixed =",
            f"{gain_vs_fixed:.3f}%"
        )

        print(
            "Gap to oracle =",
            f"{oracle_gap:.3f}%"
        )

        results.append({
            "model": model_name,
            "features": feature_name,
            "MAE_mm": mae,
            "RMSE_mm": rmse,
            "R2": r2,
            "Spearman_rho": rho,
            "Spearman_p": rho_p,
            "direction_accuracy": direction_acc,
            "selected_moved_l2_mm": selected_mean,
            "fixed_moved_l2_mm": fixed_mean,
            "oracle_moved_l2_mm": oracle_mean,
            "improvement_vs_fixed_pct": gain_vs_fixed,
            "oracle_gap_pct": oracle_gap,
        })

        temp = pd.DataFrame({
            "clip": df["clip"],
            "episode": df["episode"],
            "model": model_name,
            "features": feature_name,
            "true_delta_mm": y,
            "pred_delta_mm": pred_oof,
            "selected_error_mm": selected_error,
            "fixed_error_mm": fixed_error,
            "oracle_error_mm": oracle_error,
        })

        oof_tables.append(temp)

# ============================================================
# Save
# ============================================================

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUT_RESULT,
    index=False,
)

pd.concat(
    oof_tables,
    ignore_index=True,
).to_csv(
    OUT_OOF,
    index=False,
)

print("\n" + "=" * 100)
print("FINAL COMPARISON")
print("=" * 100)

print(
    result_df.to_string(
        index=False
    )
)

print("\nRESULT =", OUT_RESULT)
print("OOF =", OUT_OOF)
print("SUCCESS = True")
