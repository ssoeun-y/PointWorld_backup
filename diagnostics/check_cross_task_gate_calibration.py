from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

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
PROB_THRESHOLDS = [0.30, 0.50, 0.70]

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


def safe_auroc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def safe_auprc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, p)


def ece_score(y, p, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            m = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            m = (p >= edges[i]) & (p < edges[i + 1])

        if m.sum() == 0:
            continue

        conf = p[m].mean()
        acc = y[m].mean()

        ece += (m.sum() / len(y)) * abs(conf - acc)

    return ece


rows = []
prob_rows = []

print("=" * 105)
print("CROSS-TASK PROBABILITY CALIBRATION")
print("3 tasks train -> 1 calibration task -> 1 completely unseen test task")
print("TRUE FAILURE THRESHOLD DEFINED ONLY FROM TRAIN TASKS")
print("=" * 105)

for base_pair in ["HL", "HR"]:

    target = f"err_{base_pair}_mm"

    print(f"\n{'='*105}")
    print("BASE =", base_pair)
    print("=" * 105)

    for test_task in TASKS:

        remaining = [t for t in TASKS if t != test_task]

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
            y_test = df.loc[te, target].to_numpy()

            # --------------------------------------------------
            # Ground-truth failure definition:
            # fixed exclusively from TRAIN distribution.
            # --------------------------------------------------
            failure_threshold_mm = np.quantile(y_train, 0.80)

            failure_calib = (
                y_calib >= failure_threshold_mm
            ).astype(int)

            failure_test = (
                y_test >= failure_threshold_mm
            ).astype(int)

            # --------------------------------------------------
            # Action-only error predictor
            # --------------------------------------------------
            model = RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=5,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42,
            )

            model.fit(
                X.loc[tr],
                y_train
            )

            score_calib = model.predict(
                X.loc[ca]
            )

            score_test = model.predict(
                X.loc[te]
            )

            # --------------------------------------------------
            # Platt-style probability calibration.
            #
            # Learns:
            # predicted moved-L2 risk
            #        ->
            # probability of TRAIN-defined failure
            #
            # Uses calibration task ONLY.
            # --------------------------------------------------
            if len(np.unique(failure_calib)) < 2:
                raise RuntimeError(
                    f"Calibration task {calib_task} has only "
                    f"one failure class."
                )

            calibrator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                )
            )

            calibrator.fit(
                score_calib.reshape(-1, 1),
                failure_calib
            )

            prob_test = calibrator.predict_proba(
                score_test.reshape(-1, 1)
            )[:, 1]

            actual_failure_rate = failure_test.mean()
            predicted_failure_rate = prob_test.mean()

            auroc = safe_auroc(
                failure_test,
                prob_test
            )

            auprc = safe_auprc(
                failure_test,
                prob_test
            )

            brier = brier_score_loss(
                failure_test,
                prob_test
            )

            ece = ece_score(
                failure_test,
                prob_test,
                n_bins=10,
            )

            prob_rows.append({
                "base_pair": base_pair,
                "test_task": test_task,
                "calib_task": calib_task,
                "train_tasks": "+".join(train_tasks),

                "failure_threshold_mm":
                    failure_threshold_mm,

                "actual_failure_rate":
                    actual_failure_rate,

                "predicted_failure_prob_mean":
                    predicted_failure_rate,

                "prevalence_gap":
                    predicted_failure_rate
                    - actual_failure_rate,

                "auroc":
                    auroc,

                "auprc":
                    auprc,

                "brier":
                    brier,

                "ece":
                    ece,
            })

            # --------------------------------------------------
            # Fixed probability thresholds.
            #
            # IMPORTANT:
            # No quantile/top-X% information from test task.
            # --------------------------------------------------
            for pthr in PROB_THRESHOLDS:

                reject = prob_test >= pthr

                tp = np.sum(
                    reject & (failure_test == 1)
                )
                fp = np.sum(
                    reject & (failure_test == 0)
                )
                fn = np.sum(
                    (~reject) & (failure_test == 1)
                )
                tn = np.sum(
                    (~reject) & (failure_test == 0)
                )

                precision = (
                    tp / (tp + fp)
                    if (tp + fp) > 0
                    else np.nan
                )

                recall = (
                    tp / (tp + fn)
                    if (tp + fn) > 0
                    else np.nan
                )

                fpr = (
                    fp / (fp + tn)
                    if (fp + tn) > 0
                    else np.nan
                )

                retained = y_test[~reject]

                base_mean = y_test.mean()
                base_p95 = np.quantile(
                    y_test,
                    0.95
                )

                if len(retained) > 0:

                    retained_mean = retained.mean()
                    retained_p95 = np.quantile(
                        retained,
                        0.95
                    )

                    mean_reduction = (
                        100.0
                        * (base_mean - retained_mean)
                        / base_mean
                    )

                    p95_reduction = (
                        100.0
                        * (base_p95 - retained_p95)
                        / base_p95
                    )

                else:
                    retained_mean = np.nan
                    retained_p95 = np.nan
                    mean_reduction = np.nan
                    p95_reduction = np.nan

                rows.append({
                    "base_pair": base_pair,
                    "test_task": test_task,
                    "calib_task": calib_task,
                    "train_tasks":
                        "+".join(train_tasks),

                    "prob_threshold":
                        pthr,

                    "actual_failure_rate":
                        actual_failure_rate,

                    "predicted_failure_prob_mean":
                        predicted_failure_rate,

                    "reject_rate":
                        reject.mean(),

                    "precision":
                        precision,

                    "recall":
                        recall,

                    "false_positive_rate":
                        fpr,

                    "baseline_mean_mm":
                        base_mean,

                    "retained_mean_mm":
                        retained_mean,

                    "mean_reduction_pct":
                        mean_reduction,

                    "baseline_p95_mm":
                        base_p95,

                    "retained_p95_mm":
                        retained_p95,

                    "p95_reduction_pct":
                        p95_reduction,
                })

            print(
                f"test={test_task} | "
                f"calib={calib_task} | "
                f"true failure={actual_failure_rate*100:5.1f}% | "
                f"pred prob={predicted_failure_rate*100:5.1f}% | "
                f"AUROC={auroc:.3f} | "
                f"AUPRC={auprc:.3f} | "
                f"ECE={ece:.3f}"
            )


prob_df = pd.DataFrame(prob_rows)
gate_df = pd.DataFrame(rows)

prob_df.to_csv(
    OUT / "cross_task_probability_calibration.csv",
    index=False
)

gate_df.to_csv(
    OUT / "cross_task_calibrated_gate_thresholds.csv",
    index=False
)


print("\n" + "=" * 105)
print("PROBABILITY CALIBRATION — MACRO")
print("=" * 105)

print(
    prob_df.groupby("base_pair")[
        [
            "actual_failure_rate",
            "predicted_failure_prob_mean",
            "prevalence_gap",
            "auroc",
            "auprc",
            "brier",
            "ece",
        ]
    ].mean().to_string()
)


print("\n" + "=" * 105)
print("FIXED p>=0.50 GATE — MACRO")
print("=" * 105)

p50 = gate_df[
    np.isclose(
        gate_df["prob_threshold"],
        0.50
    )
]

print(
    p50.groupby("base_pair")[
        [
            "actual_failure_rate",
            "reject_rate",
            "precision",
            "recall",
            "false_positive_rate",
            "mean_reduction_pct",
            "p95_reduction_pct",
        ]
    ].mean().to_string()
)


print("\n" + "=" * 105)
print("FIXED p>=0.50 GATE — PER UNSEEN TASK")
print("=" * 105)

print(
    p50.groupby(
        ["base_pair", "test_task"]
    )[
        [
            "actual_failure_rate",
            "reject_rate",
            "precision",
            "recall",
            "mean_reduction_pct",
            "p95_reduction_pct",
        ]
    ].mean().to_string()
)


print("\n" + "=" * 105)
print("THRESHOLD SWEEP")
print("=" * 105)

print(
    gate_df.groupby(
        ["base_pair", "prob_threshold"]
    )[
        [
            "reject_rate",
            "precision",
            "recall",
            "mean_reduction_pct",
            "p95_reduction_pct",
        ]
    ].mean().to_string()
)


print("\nOUTPUT =", OUT)
print("SUCCESS = True")
