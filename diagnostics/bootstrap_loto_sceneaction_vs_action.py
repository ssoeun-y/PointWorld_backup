from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model/"
    "uncertainty_gate_5task"
)
OUT = ROOT / "robustness"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["0000", "0001", "0006", "0008", "0010"]
N_BOOT = 10000
SEED = 42

data = pd.read_csv(ROOT / "gate_analysis_input.csv")
data["task"] = data["task"].astype(str).str.zfill(4)
data["group_id"] = (
    data["task"].astype(str) + "::" + data["episode"].astype(str)
)

scene_cols = sorted(
    c for c in data.columns if c.startswith("scene_")
)
action_cols = sorted(
    c for c in data.columns if c.startswith("action_")
)

feature_sets = {
    "Action-only": action_cols,
    "Scene+Action": scene_cols + action_cols,
}

all_preds = []

print("=" * 100)
print("GENERATE STRICT LOTO PREDICTIONS")
print("=" * 100)

for base_pair in ["HL", "HR"]:
    target_col = f"err_{base_pair}_mm"

    for heldout in TASKS:
        train = data[data["task"] != heldout].copy()
        test = data[data["task"] == heldout].copy()

        y_train = train[target_col].to_numpy()
        y_test = test[target_col].to_numpy()

        temp = test[
            ["clip", "episode", "task", "group_id"]
        ].copy()
        temp["base_pair"] = base_pair
        temp["true_error_mm"] = y_test

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

            temp[f"pred_{feat_name}"] = pred

        rho_a = spearmanr(
            temp["true_error_mm"],
            temp["pred_Action-only"]
        ).statistic

        rho_sa = spearmanr(
            temp["true_error_mm"],
            temp["pred_Scene+Action"]
        ).statistic

        print(
            f"base={base_pair} | heldout={heldout} | "
            f"Action={rho_a:+.4f} | "
            f"Scene+Action={rho_sa:+.4f} | "
            f"delta={rho_sa-rho_a:+.4f}"
        )

        all_preds.append(temp)

pred = pd.concat(all_preds, ignore_index=True)
pred.to_csv(
    OUT / "loto_action_vs_sceneaction_predictions.csv",
    index=False,
)

# ------------------------------------------------------------------
# Stratified paired cluster bootstrap:
# resample EPISODES independently within each held-out task.
# Same resampled observations are used for both models.
# ------------------------------------------------------------------

rng = np.random.default_rng(SEED)

rows = []

for base_pair in ["HL", "HR"]:

    d = pred[pred["base_pair"] == base_pair].copy()

    # Original macro metric:
    # compute rho separately per heldout task, then average across 5 tasks.
    orig = {}

    for model in ["Action-only", "Scene+Action"]:
        rhos = []

        for task in TASKS:
            g = d[d["task"] == task]

            rhos.append(
                spearmanr(
                    g["true_error_mm"],
                    g[f"pred_{model}"]
                ).statistic
            )

        orig[model] = np.mean(rhos)

    orig_delta = (
        orig["Scene+Action"] - orig["Action-only"]
    )

    boot_delta_rho = []
    boot_delta_mae = []

    for b in range(N_BOOT):

        task_rho_a = []
        task_rho_sa = []

        mae_a_all = []
        mae_sa_all = []

        for task in TASKS:

            g = d[d["task"] == task].copy()

            episodes = g["group_id"].unique()

            sampled_eps = rng.choice(
                episodes,
                size=len(episodes),
                replace=True,
            )

            pieces = []

            for i, ep in enumerate(sampled_eps):
                x = g[g["group_id"] == ep].copy()

                # make duplicated sampled episodes distinct clusters
                x["_boot_cluster"] = i
                pieces.append(x)

            bs = pd.concat(
                pieces,
                ignore_index=True
            )

            rho_a = spearmanr(
                bs["true_error_mm"],
                bs["pred_Action-only"]
            ).statistic

            rho_sa = spearmanr(
                bs["true_error_mm"],
                bs["pred_Scene+Action"]
            ).statistic

            if np.isfinite(rho_a):
                task_rho_a.append(rho_a)

            if np.isfinite(rho_sa):
                task_rho_sa.append(rho_sa)

            mae_a_all.append(
                np.abs(
                    bs["true_error_mm"]
                    - bs["pred_Action-only"]
                ).to_numpy()
            )

            mae_sa_all.append(
                np.abs(
                    bs["true_error_mm"]
                    - bs["pred_Scene+Action"]
                ).to_numpy()
            )

        if len(task_rho_a) == 5 and len(task_rho_sa) == 5:
            boot_delta_rho.append(
                np.mean(task_rho_sa)
                - np.mean(task_rho_a)
            )

        mae_a = np.concatenate(mae_a_all).mean()
        mae_sa = np.concatenate(mae_sa_all).mean()

        # Negative means Scene+Action has lower/better MAE.
        boot_delta_mae.append(
            mae_sa - mae_a
        )

    boot_delta_rho = np.asarray(boot_delta_rho)
    boot_delta_mae = np.asarray(boot_delta_mae)

    ci_rho = np.percentile(
        boot_delta_rho,
        [2.5, 50, 97.5]
    )

    ci_mae = np.percentile(
        boot_delta_mae,
        [2.5, 50, 97.5]
    )

    # Two-sided bootstrap sign-style p-value around zero.
    p_rho = 2 * min(
        np.mean(boot_delta_rho <= 0),
        np.mean(boot_delta_rho >= 0),
    )
    p_rho = min(p_rho, 1.0)

    p_mae = 2 * min(
        np.mean(boot_delta_mae <= 0),
        np.mean(boot_delta_mae >= 0),
    )
    p_mae = min(p_mae, 1.0)

    rows.append({
        "base_pair": base_pair,
        "rho_action_macro": orig["Action-only"],
        "rho_scene_action_macro": orig["Scene+Action"],
        "delta_rho_sceneaction_minus_action": orig_delta,
        "delta_rho_ci_low": ci_rho[0],
        "delta_rho_ci_median": ci_rho[1],
        "delta_rho_ci_high": ci_rho[2],
        "delta_rho_bootstrap_p": p_rho,
        "delta_mae_sceneaction_minus_action_ci_low": ci_mae[0],
        "delta_mae_sceneaction_minus_action_ci_median": ci_mae[1],
        "delta_mae_sceneaction_minus_action_ci_high": ci_mae[2],
        "delta_mae_bootstrap_p": p_mae,
    })

    print("\n" + "=" * 100)
    print(f"BOOTSTRAP RESULT — BASE {base_pair}")
    print("=" * 100)

    print(
        f"Action-only macro rho    = {orig['Action-only']:.4f}"
    )
    print(
        f"Scene+Action macro rho   = {orig['Scene+Action']:.4f}"
    )
    print(
        f"Delta rho (SA - A)       = {orig_delta:+.4f}"
    )
    print(
        f"95% CI                   = "
        f"[{ci_rho[0]:+.4f}, {ci_rho[2]:+.4f}]"
    )
    print(
        f"bootstrap p              = {p_rho:.6g}"
    )

    print(
        f"\nDelta MAE (SA - A) median = "
        f"{ci_mae[1]:+.4f} mm"
    )
    print(
        f"95% CI                    = "
        f"[{ci_mae[0]:+.4f}, {ci_mae[2]:+.4f}]"
    )
    print(
        f"bootstrap p               = {p_mae:.6g}"
    )

summary = pd.DataFrame(rows)

summary.to_csv(
    OUT / "bootstrap_sceneaction_vs_action_summary.csv",
    index=False,
)

print("\n" + "=" * 100)
print("FINAL SUMMARY")
print("=" * 100)
print(summary.to_string(index=False))

print("\nOUTPUT =", OUT)
print("SUCCESS = True")
