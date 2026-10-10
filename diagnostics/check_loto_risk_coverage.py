from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model/"
    "uncertainty_gate_5task/robustness"
)

FILE = ROOT / "loto_action_vs_sceneaction_predictions.csv"
OUT = ROOT / "loto_risk_coverage.csv"

TASKS = ["0000", "0001", "0006", "0008", "0010"]
REJECT_RATES = [0.0, 0.1, 0.2, 0.3, 0.5]

df = pd.read_csv(FILE)
df["task"] = df["task"].astype(str).str.zfill(4)

rows = []

print("=" * 100)
print("STRICT LOTO RISK-COVERAGE")
print("=" * 100)

for base in ["HL", "HR"]:

    print(f"\nBASE = {base}")

    dbase = df[df["base_pair"] == base].copy()

    for feat, pred_col in [
        ("Action-only", "pred_Action-only"),
        ("Scene+Action", "pred_Scene+Action"),
    ]:

        print(f"\n[{feat}]")

        for task in TASKS:

            g = dbase[
                dbase["task"] == task
            ].copy()

            y = g["true_error_mm"].to_numpy()
            score = g[pred_col].to_numpy()

            # low predicted risk first = retained predictions
            order = np.argsort(score)

            base_mean = y.mean()
            base_p95 = np.quantile(y, 0.95)

            for reject in REJECT_RATES:

                keep_n = max(
                    1,
                    int(round(
                        (1.0 - reject) * len(g)
                    ))
                )

                keep = order[:keep_n]
                retained = y[keep]

                mean_err = retained.mean()
                p95_err = np.quantile(
                    retained, 0.95
                )

                rows.append({
                    "base_pair": base,
                    "task": task,
                    "features": feat,
                    "reject_rate": reject,
                    "coverage": 1.0 - reject,
                    "N_total": len(g),
                    "N_retained": len(retained),
                    "baseline_mean_mm": base_mean,
                    "retained_mean_mm": mean_err,
                    "mean_reduction_pct":
                        100.0 * (
                            base_mean - mean_err
                        ) / base_mean,
                    "baseline_p95_mm": base_p95,
                    "retained_p95_mm": p95_err,
                    "p95_reduction_pct":
                        100.0 * (
                            base_p95 - p95_err
                        ) / base_p95,
                })

        temp = pd.DataFrame(rows)

        x = temp[
            (temp["base_pair"] == base) &
            (temp["features"] == feat)
        ]

        macro = (
            x.groupby("reject_rate")[
                [
                    "mean_reduction_pct",
                    "p95_reduction_pct",
                ]
            ]
            .mean()
        )

        print("\nMacro across 5 unseen tasks:")
        print(macro.to_string())

result = pd.DataFrame(rows)
result.to_csv(OUT, index=False)

print("\n" + "=" * 100)
print("20% REJECT — TASKWISE")
print("=" * 100)

print(
    result[
        result["reject_rate"] == 0.2
    ][
        [
            "base_pair",
            "task",
            "features",
            "mean_reduction_pct",
            "p95_reduction_pct",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 100)
print("20% REJECT — MACRO")
print("=" * 100)

print(
    result[
        result["reject_rate"] == 0.2
    ]
    .groupby(
        ["base_pair", "features"]
    )[
        [
            "mean_reduction_pct",
            "p95_reduction_pct",
        ]
    ]
    .mean()
    .to_string()
)

print("\nOUTPUT =", OUT)
print("SUCCESS = True")
