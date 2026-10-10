from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

METRIC = Path(
    "diagnostics/results/established_action_metrics_5task/"
    "per_sample_established_metrics_7987.csv"
)
COV = Path(
    "diagnostics/results/action_confound_control_5task/"
    "per_sample_covariates_7987.csv"
)

OUT = Path(
    "diagnostics/results/action_confound_control_5task/"
    "task0010_boundary_summary.csv"
)

def summarize(g, col):
    x = g[col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()

    q05, q25, q50, q75, q95 = np.percentile(x, [5,25,50,75,95])

    return {
        "median": q50,
        "iqr": q75 - q25,
        "p05": q05,
        "p95": q95,
        "p05_p95_range": q95 - q05,
        "mean": np.mean(x),
        "std": np.std(x),
    }

def main():
    m = pd.read_csv(METRIC)
    c = pd.read_csv(COV)

    df = m.merge(c, on="clip", how="inner", validate="one_to_one")

    assert len(df) == 7987

    cols = [
        "rms_acceleration",
        "mean_squared_jerk",
        "active_path_length_m",
        "active_net_displacement_m",
        "gt_moved_motion_mm",
        "l2_moved_mm",
    ]

    rows = []

    for task, g in df.groupby("task"):
        for col in cols:
            s = summarize(g, col)

            rows.append({
                "task": task,
                "variable": col,
                "N": len(g),
                **s,
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print("\n=== TASKWISE DISTRIBUTION / RANGE ===")
    print(
        out.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}"
        )
    )

    print("\n=== ACTION METRIC -> ERROR CORRELATION ===")
    for task, g in df.groupby("task"):
        print(f"\n[{task}] N={len(g)}")

        for metric in ["rms_acceleration", "mean_squared_jerk"]:
            d = g[[metric, "l2_moved_mm"]].dropna()
            rho, p = spearmanr(d[metric], d["l2_moved_mm"])

            print(
                f"{metric:20s} "
                f"rho={rho:+.6f}  p={p:.6g}"
            )

    # Compare 0010 range against median range of the other 4 tasks
    print("\n=== TASK-0010 RANGE RELATIVE TO OTHER TASKS ===")

    for col in cols:
        tmp = out[out["variable"] == col].set_index("task")

        r0010 = tmp.loc["task-0010", "p05_p95_range"]
        others = tmp.drop(index="task-0010")["p05_p95_range"]

        ratio = r0010 / others.median()

        print(
            f"{col:28s} "
            f"0010_range={r0010:.6g}  "
            f"other_median_range={others.median():.6g}  "
            f"ratio={ratio:.3f}x"
        )

    print("\nOUTPUT =", OUT.resolve())

if __name__ == "__main__":
    main()
