from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model/"
    "uncertainty_gate_5task"
)
OUT = Path(
    "diagnostics/results/action_conditioned_active_world_model/"
    "reliability_router_final"
)
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["0000", "0001", "0006", "0008", "0010"]

df = pd.read_csv(ROOT / "gate_analysis_input.csv")
df["task"] = df["task"].astype(str).str.zfill(4)

action_cols = sorted(
    c for c in df.columns
    if c.startswith("action_")
)

X = (
    df[action_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

rows = []

print("=" * 100)
print("DEPLOYABLE ACTION-RISK GATE")
print("3 tasks train -> 1 calibration -> 1 completely unseen test")
print("=" * 100)

for base_pair in ["HL", "HR"]:

    target = f"err_{base_pair}_mm"

    print(f"\nBASE = {base_pair}")

    for test_task in TASKS:

        remaining = [
            t for t in TASKS
            if t != test_task
        ]

        for calib_task in remaining:

            train_tasks = [
                t for t in remaining
                if t != calib_task
            ]

            tr = df["task"].isin(train_tasks)
            ca = df["task"].eq(calib_task)
            te = df["task"].eq(test_task)

            y_train = df.loc[tr, target].to_numpy()
            y_calib = df.loc[ca, target].to_numpy()
            y_test  = df.loc[te, target].to_numpy()

            # -------------------------------------------------
            # "True failure" threshold is defined ONLY
            # from training tasks.
            # Top 20%-level error in training distribution.
            # -------------------------------------------------
            failure_threshold_mm = np.quantile(
                y_train, 0.80
            )

            model = RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=5,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            )

            model.fit(
                X.loc[tr],
                y_train
            )

            pred_calib = model.predict(
                X.loc[ca]
            )

            pred_test = model.predict(
                X.loc[te]
            )

            # -------------------------------------------------
            # Gate threshold fixed ONLY using calibration task.
            # Highest predicted-risk 20% of calibration set.
            # -------------------------------------------------
            risk_threshold = np.quantile(
                pred_calib, 0.80
            )

            reject = pred_test >= risk_threshold

            # IMPORTANT:
            # Test failure uses TRAIN-derived absolute threshold.
            # No top-20%-of-test information is used.
            failure = y_test >= failure_threshold_mm

            tp = np.sum(reject & failure)
            fp = np.sum(reject & ~failure)
            fn = np.sum(~reject & failure)
            tn = np.sum(~reject & ~failure)

            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0 else np.nan
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0 else np.nan
            )

            fpr = (
                fp / (fp + tn)
                if (fp + tn) > 0 else np.nan
            )

            actual_reject_rate = reject.mean()

            baseline_mean = y_test.mean()
            baseline_p95 = np.quantile(
                y_test, 0.95
            )

            retained = y_test[~reject]

            retained_mean = retained.mean()
            retained_p95 = np.quantile(
                retained, 0.95
            )

            mean_reduction = (
                100 * (
                    baseline_mean
                    - retained_mean
                ) / baseline_mean
            )

            p95_reduction = (
                100 * (
                    baseline_p95
                    - retained_p95
                ) / baseline_p95
            )

            rows.append({
                "base_pair": base_pair,
                "test_task": test_task,
                "calib_task": calib_task,
                "train_tasks": "+".join(train_tasks),

                "failure_threshold_mm":
                    failure_threshold_mm,
                "risk_threshold":
                    risk_threshold,

                "actual_test_failure_rate":
                    failure.mean(),
                "actual_test_reject_rate":
                    actual_reject_rate,

                "precision":
                    precision,
                "recall":
                    recall,
                "false_positive_rate":
                    fpr,

                "baseline_mean_mm":
                    baseline_mean,
                "retained_mean_mm":
                    retained_mean,
                "mean_reduction_pct":
                    mean_reduction,

                "baseline_p95_mm":
                    baseline_p95,
                "retained_p95_mm":
                    retained_p95,
                "p95_reduction_pct":
                    p95_reduction,
            })

            print(
                f"test={test_task} | "
                f"calib={calib_task} | "
                f"reject={actual_reject_rate*100:5.1f}% | "
                f"precision={precision*100:5.1f}% | "
                f"recall={recall*100:5.1f}% | "
                f"mean↓={mean_reduction:5.1f}% | "
                f"p95↓={p95_reduction:5.1f}%"
            )

res = pd.DataFrame(rows)

res.to_csv(
    OUT / "deployable_gate_threshold_results.csv",
    index=False
)

print("\n" + "=" * 100)
print("MACRO SUMMARY")
print("=" * 100)

print(
    res.groupby("base_pair")[
        [
            "actual_test_reject_rate",
            "precision",
            "recall",
            "false_positive_rate",
            "mean_reduction_pct",
            "p95_reduction_pct",
        ]
    ].mean().to_string()
)

print("\n" + "=" * 100)
print("PER UNSEEN TASK")
print("=" * 100)

print(
    res.groupby(
        ["base_pair", "test_task"]
    )[
        [
            "actual_test_reject_rate",
            "precision",
            "recall",
            "mean_reduction_pct",
            "p95_reduction_pct",
        ]
    ].mean().to_string()
)

print("\nOUTPUT =", OUT)
print("SUCCESS = True")
