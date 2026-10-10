from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model"
)
GROOT = ROOT / "uncertainty_gate_5task"
OUT = ROOT / "reliability_router_final"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["0000", "0001", "0006", "0008", "0010"]
RATES = [0.10, 0.20, 0.30]

# ============================================================
# 1. LOAD CURRENT DATA
# ============================================================

data = pd.read_csv(GROOT / "gate_analysis_input.csv")
data["task"] = data["task"].astype(str).str.zfill(4)

# HLR actual errors
hs = []

for t in TASKS:
    p = (
        ROOT /
        f"fixed_camera_pair_task{t}" /
        f"task{t}_pair_HLR__per_sample.csv"
    )

    h = pd.read_csv(p)

    metric = (
        "behavior/l2_moved/mean"
        if "behavior/l2_moved/mean" in h.columns
        else "l2_moved/mean"
    )

    h = h[["clip", metric]].copy()
    h["task"] = t
    h["err_HLR_mm"] = h[metric] * 1000.0

    hs.append(
        h[["task", "clip", "err_HLR_mm"]]
    )

hlr = pd.concat(hs, ignore_index=True)

data = data.merge(
    hlr,
    on=["task", "clip"],
    validate="one_to_one",
)

assert len(data) == 7988

# Existing OOF Action-risk score
risk = pd.read_csv(
    GROOT / "gate_oof_predictions.csv"
)

risk["task"] = risk["task"].astype(str).str.zfill(4)

risk = risk[
    (risk["model"] == "RandomForest")
    & (risk["features"] == "Action-only")
][["task", "clip", "predicted_risk"]]

data = data.merge(
    risk,
    on=["task", "clip"],
    validate="one_to_one",
)

# ============================================================
# 2. FEATURES
# ============================================================

scene_cols = sorted(
    c for c in data.columns
    if c.startswith("scene_")
)

action_cols = sorted(
    c for c in data.columns
    if c.startswith("action_")
)

FEATURES = {
    "Scene-only": scene_cols,
    "Action-only": action_cols,
    "Scene+Action": scene_cols + action_cols,
}

TARGETS = {
    "HL": "err_HL_mm",
    "HR": "err_HR_mm",
    "HLR": "err_HLR_mm",
}

def make_rf():
    return RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )

# Existing episode-disjoint folds
fold_ids = sorted(data["fold"].astype(int).unique())

# ============================================================
# 3. OOF PREDICT EACH CANDIDATE'S ERROR
# ============================================================

candidate_predictions = {}

print("=" * 100)
print("GT-FREE CANDIDATE ERROR PREDICTION")
print("=" * 100)

for feat_name, cols in FEATURES.items():

    X = (
        data[cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    pred = {
        name: np.full(len(data), np.nan)
        for name in TARGETS
    }

    for fold in fold_ids:

        te = np.where(
            data["fold"].astype(int).to_numpy()
            == fold
        )[0]

        tr = np.where(
            data["fold"].astype(int).to_numpy()
            != fold
        )[0]

        for candidate, target_col in TARGETS.items():

            model = make_rf()

            model.fit(
                X.iloc[tr],
                data.iloc[tr][target_col].to_numpy(),
            )

            pred[candidate][te] = model.predict(
                X.iloc[te]
            )

    candidate_predictions[feat_name] = pred

    for candidate, target_col in TARGETS.items():

        rho = spearmanr(
            data[target_col],
            pred[candidate],
        ).statistic

        print(
            f"{feat_name:12s} | "
            f"{candidate:3s} error prediction rho={rho:+.4f}"
        )


# ============================================================
# 4. HIGH-RISK SELECTIVE RECOVERY
#
# Low-risk: keep original 2-view baseline.
# High-risk: choose candidate with lowest PREDICTED error.
#
# TEST TIME DOES NOT USE GT.
# ============================================================

base = data["base_error_mm"].to_numpy()

actual = np.stack(
    [
        data["err_HL_mm"].to_numpy(),
        data["err_HR_mm"].to_numpy(),
        data["err_HLR_mm"].to_numpy(),
    ],
    axis=1,
)

candidate_names = np.array(
    ["HL", "HR", "HLR"]
)

risk_score = data["predicted_risk"].to_numpy()

rows = []
task_rows = []

print("\n" + "=" * 100)
print("HIGH-RISK SELECTIVE RECOVERY")
print("=" * 100)

for feat_name, pred_dict in candidate_predictions.items():

    pred_matrix = np.stack(
        [
            pred_dict["HL"],
            pred_dict["HR"],
            pred_dict["HLR"],
        ],
        axis=1,
    )

    # GT-free candidate choice
    selected_candidate = np.argmin(
        pred_matrix,
        axis=1,
    )

    selected_actual_error = actual[
        np.arange(len(data)),
        selected_candidate,
    ]

    # Oracle candidate for analysis only
    oracle_candidate = np.argmin(
        actual,
        axis=1,
    )

    candidate_accuracy = np.mean(
        selected_candidate == oracle_candidate
    )

    print(
        f"\n[{feat_name}] "
        f"oracle-candidate accuracy = "
        f"{candidate_accuracy*100:.2f}%"
    )

    order = np.argsort(-risk_score)

    for rate in RATES:

        k = int(round(rate * len(data)))
        idx = order[:k]

        # ----------------------------------------------------
        # GT-free selector
        # ----------------------------------------------------
        final = base.copy()
        final[idx] = selected_actual_error[idx]

        selector_gain = (
            100.0
            * (base.mean() - final.mean())
            / base.mean()
        )

        # ----------------------------------------------------
        # Oracle upper bound on same high-risk subset
        # ----------------------------------------------------
        oracle_final = base.copy()
        oracle_final[idx] = np.min(
            actual[idx],
            axis=1,
        )

        oracle_gain = (
            100.0
            * (base.mean() - oracle_final.mean())
            / base.mean()
        )

        recovery_fraction = (
            selector_gain / oracle_gain
            if oracle_gain > 0
            else np.nan
        )

        switch_rate = np.mean(
            selected_actual_error[idx]
            != base[idx]
        )

        helps = np.mean(
            selected_actual_error[idx]
            < base[idx]
        )

        print(
            f"rate={rate:.1f} | "
            f"selector gain={selector_gain:+.3f}% | "
            f"oracle={oracle_gain:+.3f}% | "
            f"oracle recovered={recovery_fraction*100:.1f}% | "
            f"actually helps={helps*100:.1f}%"
        )

        rows.append({
            "features": feat_name,
            "risk_rate": rate,
            "selector_gain_pct": selector_gain,
            "oracle_gain_pct": oracle_gain,
            "oracle_recovered_fraction":
                recovery_fraction,
            "selected_candidate_accuracy":
                candidate_accuracy,
            "selected_helps_fraction": helps,
            "switch_fraction": switch_rate,
        })

        # ----------------------------------------------------
        # TASKWISE
        # ----------------------------------------------------
        for task in TASKS:

            task_mask = (
                data["task"].to_numpy()
                == task
            )

            # high-risk top rate WITHIN each task
            task_idx_all = np.where(task_mask)[0]

            local_order = task_idx_all[
                np.argsort(
                    -risk_score[task_idx_all]
                )
            ]

            kt = max(
                1,
                int(round(
                    rate * len(task_idx_all)
                ))
            )

            tidx = local_order[:kt]

            task_base = base[task_idx_all].mean()

            # final error for whole task
            task_final = base[
                task_idx_all
            ].copy()

            loc = {
                global_idx: local_i
                for local_i, global_idx
                in enumerate(task_idx_all)
            }

            for gi in tidx:
                task_final[loc[gi]] = (
                    selected_actual_error[gi]
                )

            task_gain = (
                100.0
                * (task_base - task_final.mean())
                / task_base
            )

            task_rows.append({
                "task": task,
                "features": feat_name,
                "risk_rate": rate,
                "gain_pct": task_gain,
            })


# ============================================================
# 5. SAVE / SUMMARIZE
# ============================================================

result = pd.DataFrame(rows)
task_result = pd.DataFrame(task_rows)

result.to_csv(
    OUT / "highrisk_gtfree_selector_summary.csv",
    index=False,
)

task_result.to_csv(
    OUT / "highrisk_gtfree_selector_taskwise.csv",
    index=False,
)

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    result[
        [
            "features",
            "risk_rate",
            "selector_gain_pct",
            "oracle_gain_pct",
            "oracle_recovered_fraction",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 100)
print("TASKWISE @ 20% HIGH-RISK")
print("=" * 100)

print(
    task_result[
        task_result["risk_rate"] == 0.20
    ].pivot(
        index="task",
        columns="features",
        values="gain_pct",
    ).to_string()
)

print("\nOUTPUT =", OUT)
print("SUCCESS = True")
