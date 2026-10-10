from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model"
)

TASKS = ["0000", "0001", "0006", "0008", "0010"]

EXPECTED_VALID = {
    "0000": 383,
    "0001": 2312,
    "0006": 1855,
    "0008": 1824,
    "0010": 1614,
}

frames = []

# ============================================================
# LOAD ALL FIVE TASKS
# ============================================================

print("=" * 100)
print("5-TASK FEATURE / LABEL AUDIT")
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
    ).copy()

    # Feature files are already extracted only for valid HL/HR clips.
    feat["task"] = task

    df = pair.merge(
        feat,
        on="clip",
        how="inner",
        validate="one_to_one",
    )

    df["task"] = task

    df["query_label"] = np.where(
        df["err_HL_mm"] < df["err_HR_mm"],
        "L",
        "R",
    )

    # Positive = LEFT is better.
    df["delta_view_mm"] = (
        df["err_HR_mm"]
        - df["err_HL_mm"]
    )

    print(
        f"task-{task}: "
        f"pair-valid={len(pair)} | "
        f"features={len(feat)} | "
        f"merged={len(df)} | "
        f"expected={EXPECTED_VALID[task]} | "
        f"{'OK' if len(df)==EXPECTED_VALID[task] else 'MISMATCH'}"
    )

    frames.append(df)

data = pd.concat(
    frames,
    ignore_index=True,
)

print("-" * 100)
print("POOLED N =", len(data))
print("EXPECTED =", sum(EXPECTED_VALID.values()))

assert len(data) == sum(EXPECTED_VALID.values())

print("EPISODES =", data["episode"].nunique())
print(
    "LEFT best =",
    f"{(data.query_label=='L').mean()*100:.2f}%"
)
print(
    "RIGHT best =",
    f"{(data.query_label=='R').mean()*100:.2f}%"
)

# ============================================================
# FEATURE SETS
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

print("\nScene features =", len(scene_cols))
print("Action features =", len(action_cols))

# ============================================================
# SAME EPISODE-DISJOINT FOLDS FOR ALL METHODS
# ============================================================

y_cls = data["query_label"]
y_reg = data["delta_view_mm"].to_numpy()
groups = data["episode"]

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

folds = list(
    cv.split(
        np.zeros((len(data), 1)),
        y_cls,
        groups,
    )
)

# ============================================================
# FAIR FIXED BASELINE
#
# Fixed LEFT/RIGHT is selected PER TASK using training fold only.
# This is stronger/fairer than one global fixed camera.
# ============================================================

def fold_taskwise_fixed(train_idx, test_idx):

    train = data.iloc[train_idx]
    test = data.iloc[test_idx]

    fixed = np.full(len(test), np.nan)

    for task in TASKS:

        tr = train[train["task"] == task]
        mask = (
            test["task"].to_numpy()
            == task
        )

        if not mask.any():
            continue

        choose = (
            "L"
            if tr["err_HL_mm"].mean()
            < tr["err_HR_mm"].mean()
            else "R"
        )

        fixed[mask] = (
            test.loc[mask, "err_HL_mm"]
            if choose == "L"
            else test.loc[mask, "err_HR_mm"]
        )

    return fixed


# ============================================================
# CLASSIFICATION
# ============================================================

classifier_factories = {
    "LogisticRegression": lambda: Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]),

    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    ),
}

cls_results = []
cls_oof_tables = []

print("\n" + "=" * 100)
print("CLASSIFICATION: WHICH QUERY IS BETTER?")
print("=" * 100)

for model_name, factory in classifier_factories.items():

    for feature_name, cols in feature_sets.items():

        X = (
            data[cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        pred_all = np.empty(
            len(data),
            dtype=object,
        )

        selected_all = np.full(
            len(data),
            np.nan,
        )

        fixed_all = np.full(
            len(data),
            np.nan,
        )

        oracle_all = np.minimum(
            data["err_HL_mm"].to_numpy(),
            data["err_HR_mm"].to_numpy(),
        )

        for fold, (tr, te) in enumerate(
            folds,
            start=1,
        ):

            # No episode leakage.
            assert set(groups.iloc[tr]).isdisjoint(
                set(groups.iloc[te])
            )

            model = factory()

            model.fit(
                X.iloc[tr],
                y_cls.iloc[tr],
            )

            pred = model.predict(
                X.iloc[te]
            )

            pred_all[te] = pred

            test = data.iloc[te]

            selected_all[te] = np.where(
                pred == "L",
                test["err_HL_mm"],
                test["err_HR_mm"],
            )

            fixed_all[te] = fold_taskwise_fixed(
                tr,
                te,
            )

        acc = accuracy_score(
            y_cls,
            pred_all,
        )

        bacc = balanced_accuracy_score(
            y_cls,
            pred_all,
        )

        f1 = f1_score(
            y_cls,
            pred_all,
            average="macro",
        )

        selected_mean = float(
            np.mean(selected_all)
        )

        fixed_mean = float(
            np.mean(fixed_all)
        )

        oracle_mean = float(
            np.mean(oracle_all)
        )

        gain = (
            100.0
            * (fixed_mean - selected_mean)
            / fixed_mean
        )

        print(
            f"{model_name:18s} | "
            f"{feature_name:12s} | "
            f"acc={acc*100:5.2f}% | "
            f"bacc={bacc*100:5.2f}% | "
            f"selected={selected_mean:.4f} mm | "
            f"fixed={fixed_mean:.4f} mm | "
            f"gain={gain:+.3f}%"
        )

        cls_results.append({
            "model": model_name,
            "features": feature_name,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "macro_f1": f1,
            "selected_moved_l2_mm": selected_mean,
            "fixed_moved_l2_mm": fixed_mean,
            "oracle_moved_l2_mm": oracle_mean,
            "improvement_vs_fixed_pct": gain,
        })

        temp = data[
            ["clip", "episode", "task"]
        ].copy()

        temp["model"] = model_name
        temp["features"] = feature_name
        temp["true_query"] = y_cls.to_numpy()
        temp["pred_query"] = pred_all
        temp["selected_error_mm"] = selected_all
        temp["fixed_error_mm"] = fixed_all
        temp["oracle_error_mm"] = oracle_all

        cls_oof_tables.append(temp)


# ============================================================
# CONTINUOUS VIEW UTILITY REGRESSION
# ============================================================

regressor_factories = {
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

reg_results = []
reg_oof_tables = []

print("\n" + "=" * 100)
print("REGRESSION: PREDICT HR - HL ERROR DIFFERENCE")
print("=" * 100)

for model_name, factory in regressor_factories.items():

    for feature_name, cols in feature_sets.items():

        X = (
            data[cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        pred_all = np.full(
            len(data),
            np.nan,
        )

        selected_all = np.full(
            len(data),
            np.nan,
        )

        fixed_all = np.full(
            len(data),
            np.nan,
        )

        oracle_all = np.minimum(
            data["err_HL_mm"].to_numpy(),
            data["err_HR_mm"].to_numpy(),
        )

        for tr, te in folds:

            model = factory()

            model.fit(
                X.iloc[tr],
                y_reg[tr],
            )

            pred = model.predict(
                X.iloc[te]
            )

            pred_all[te] = pred

            test = data.iloc[te]

            # delta > 0 => HR error larger => choose LEFT.
            selected_all[te] = np.where(
                pred > 0,
                test["err_HL_mm"],
                test["err_HR_mm"],
            )

            fixed_all[te] = fold_taskwise_fixed(
                tr,
                te,
            )

        mae = mean_absolute_error(
            y_reg,
            pred_all,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_reg,
                pred_all,
            )
        )

        r2 = r2_score(
            y_reg,
            pred_all,
        )

        rho, p = spearmanr(
            y_reg,
            pred_all,
        )

        direction_acc = np.mean(
            np.sign(y_reg)
            == np.sign(pred_all)
        )

        selected_mean = float(
            np.mean(selected_all)
        )

        fixed_mean = float(
            np.mean(fixed_all)
        )

        oracle_mean = float(
            np.mean(oracle_all)
        )

        gain = (
            100.0
            * (fixed_mean - selected_mean)
            / fixed_mean
        )

        print(
            f"{model_name:18s} | "
            f"{feature_name:12s} | "
            f"rho={rho:+.4f} | "
            f"dir={direction_acc*100:5.2f}% | "
            f"selected={selected_mean:.4f} mm | "
            f"gain={gain:+.3f}%"
        )

        reg_results.append({
            "model": model_name,
            "features": feature_name,
            "MAE_mm": mae,
            "RMSE_mm": rmse,
            "R2": r2,
            "Spearman_rho": rho,
            "Spearman_p": p,
            "direction_accuracy": direction_acc,
            "selected_moved_l2_mm": selected_mean,
            "fixed_moved_l2_mm": fixed_mean,
            "oracle_moved_l2_mm": oracle_mean,
            "improvement_vs_fixed_pct": gain,
        })

        temp = data[
            ["clip", "episode", "task"]
        ].copy()

        temp["model"] = model_name
        temp["features"] = feature_name
        temp["true_delta_mm"] = y_reg
        temp["pred_delta_mm"] = pred_all
        temp["selected_error_mm"] = selected_all
        temp["fixed_error_mm"] = fixed_all
        temp["oracle_error_mm"] = oracle_all

        reg_oof_tables.append(temp)


# ============================================================
# TASKWISE REPORT FROM OOF
# ============================================================

best_cls = pd.DataFrame(cls_results).sort_values(
    "selected_moved_l2_mm"
).iloc[0]

best_reg = pd.DataFrame(reg_results).sort_values(
    "selected_moved_l2_mm"
).iloc[0]

print("\n" + "=" * 100)
print("BEST POOLED METHODS")
print("=" * 100)

print(
    "Classification:",
    best_cls["model"],
    "|",
    best_cls["features"],
    "| moved-L2 =",
    f"{best_cls['selected_moved_l2_mm']:.4f} mm",
    "| gain =",
    f"{best_cls['improvement_vs_fixed_pct']:+.3f}%"
)

print(
    "Regression:",
    best_reg["model"],
    "|",
    best_reg["features"],
    "| moved-L2 =",
    f"{best_reg['selected_moved_l2_mm']:.4f} mm",
    "| gain =",
    f"{best_reg['improvement_vs_fixed_pct']:+.3f}%"
)

# ============================================================
# SAVE
# ============================================================

OUT = ROOT / "pooled_5task_scene_action"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

pd.DataFrame(cls_results).to_csv(
    OUT / "classification_summary.csv",
    index=False,
)

pd.DataFrame(reg_results).to_csv(
    OUT / "regression_summary.csv",
    index=False,
)

pd.concat(
    cls_oof_tables,
    ignore_index=True,
).to_csv(
    OUT / "classification_oof.csv",
    index=False,
)

pd.concat(
    reg_oof_tables,
    ignore_index=True,
).to_csv(
    OUT / "regression_oof.csv",
    index=False,
)

data.to_csv(
    OUT / "pooled_input_7988.csv",
    index=False,
)

print("\nOUTPUT =", OUT)
print("SUCCESS = True")
