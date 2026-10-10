from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr


CSV = Path("diagnostics/results/established_action_metrics_5task/per_sample_established_metrics_7987.csv")
OUT = Path("diagnostics/results/established_action_metrics_5task/figures")
OUT.mkdir(parents=True, exist_ok=True)


def add_prediction_groups(df, q=0.25):
    parts = []
    for task, g in df.groupby("task"):
        g = g.copy()
        lo = g["l2_moved_mm"].quantile(q)
        hi = g["l2_moved_mm"].quantile(1 - q)
        g["prediction_group"] = "MID"
        g.loc[g["l2_moved_mm"] <= lo, "prediction_group"] = "LOW_ERROR_PREDICTION"
        g.loc[g["l2_moved_mm"] >= hi, "prediction_group"] = "HIGH_ERROR_PREDICTION"
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def cliffs_delta_from_u(u, n1, n2):
    return 2.0 * float(u) / (n1 * n2) - 1.0


def summarize_group_difference(df, metric):
    low = df[df["prediction_group"] == "LOW_ERROR_PREDICTION"][metric].dropna().to_numpy()
    high = df[df["prediction_group"] == "HIGH_ERROR_PREDICTION"][metric].dropna().to_numpy()

    u, p = mannwhitneyu(high, low, alternative="two-sided")
    rho, rho_p = spearmanr(df[metric], df["l2_moved_mm"], nan_policy="omit")

    return {
        "metric": metric,
        "N_low_error": len(low),
        "N_high_error": len(high),
        "low_error_median": float(np.median(low)),
        "high_error_median": float(np.median(high)),
        "high_minus_low": float(np.median(high) - np.median(low)),
        "high_over_low": float(np.median(high) / np.median(low)) if np.median(low) != 0 else np.nan,
        "mannwhitney_p": float(p),
        "cliffs_delta": float(cliffs_delta_from_u(u, len(high), len(low))),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
    }


def save_histogram(df, metric, title, out_name, log_x=False):
    low = df[df["prediction_group"] == "LOW_ERROR_PREDICTION"][metric].dropna().to_numpy()
    high = df[df["prediction_group"] == "HIGH_ERROR_PREDICTION"][metric].dropna().to_numpy()

    allv = np.concatenate([low, high])
    if log_x:
        positive = allv[allv > 0]
        lo = np.percentile(positive, 1)
        hi = np.percentile(positive, 99)
        bins = np.logspace(np.log10(lo), np.log10(hi), 40)
    else:
        lo = np.percentile(allv, 1)
        hi = np.percentile(allv, 99)
        bins = np.linspace(lo, hi, 40)

    plt.figure(figsize=(6, 4.5))
    plt.hist(low, bins=bins, density=True, alpha=0.5, label="Low-error prediction")
    plt.hist(high, bins=bins, density=True, alpha=0.5, label="High-error prediction")
    if log_x:
        plt.xscale("log")
    plt.xlabel(metric)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / out_name, dpi=250)
    plt.close()


def save_scatter(df, metric, title, out_name, log_x=False):
    x = df[metric].to_numpy()
    y = df["l2_moved_mm"].to_numpy()

    plt.figure(figsize=(6, 4.5))
    plt.scatter(x, y, alpha=0.18, s=8)
    if log_x:
        plt.xscale("log")
    plt.xlabel(metric)
    plt.ylabel("PointWorld official moved-L2 (mm)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT / out_name, dpi=250)
    plt.close()


def main():
    df = pd.read_csv(CSV)
    df = add_prediction_groups(df, q=0.25)

    stats = pd.DataFrame([
        summarize_group_difference(df, "rms_acceleration"),
        summarize_group_difference(df, "mean_squared_jerk"),
    ])
    stats.to_csv(OUT / "main_group_stats_q25.csv", index=False)

    grouped = df[df["prediction_group"].isin(["LOW_ERROR_PREDICTION", "HIGH_ERROR_PREDICTION"])].copy()
    grouped.to_csv(OUT / "prediction_groups_q25_for_figures.csv", index=False)

    save_histogram(
        grouped,
        metric="rms_acceleration",
        title="RMS Acceleration Distribution by Prediction Group",
        out_name="rms_acceleration_distribution_q25.png",
        log_x=False,
    )

    save_histogram(
        grouped,
        metric="mean_squared_jerk",
        title="Mean Squared Jerk Distribution by Prediction Group",
        out_name="mean_squared_jerk_distribution_q25.png",
        log_x=True,
    )

    save_scatter(
        df,
        metric="rms_acceleration",
        title="RMS Acceleration vs PointWorld moved-L2",
        out_name="rms_acceleration_vs_moved_l2.png",
        log_x=False,
    )

    save_scatter(
        df,
        metric="mean_squared_jerk",
        title="Mean Squared Jerk vs PointWorld moved-L2",
        out_name="mean_squared_jerk_vs_moved_l2.png",
        log_x=True,
    )

    print("=== MAIN STATS ===")
    print(stats.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print()
    print("OUTPUT =", OUT.resolve())


if __name__ == "__main__":
    main()
