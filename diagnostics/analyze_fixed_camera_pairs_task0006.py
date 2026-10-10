from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model/"
    "fixed_camera_pair_task0006"
)

FILES = {
    "HL": ROOT / "task0006_pair_HL__per_sample.csv",
    "HR": ROOT / "task0006_pair_HR__per_sample.csv",
    "LR": ROOT / "task0006_pair_LR__per_sample.csv",
}

METRIC = "behavior/l2_moved/mean"

# ------------------------------------------------------------
# Load + align exactly the same clips
# ------------------------------------------------------------
dfs = {}

for pair, path in FILES.items():
    df = pd.read_csv(path)

    if METRIC not in df.columns:
        # fallback
        metric = "l2_moved/mean"
    else:
        metric = METRIC

    dfs[pair] = (
        df[["clip", metric]]
        .rename(columns={metric: f"err_{pair}"})
    )

merged = dfs["HL"]

for pair in ["HR", "LR"]:
    merged = merged.merge(
        dfs[pair],
        on="clip",
        how="inner",
        validate="one_to_one",
    )

print("=" * 100)
print("TASK-0006 FIXED CAMERA PAIR ANALYSIS")
print("=" * 100)
print("MATCHED CLIPS =", len(merged))

assert len(merged) == 2157, (
    f"Expected 2157 matched clips, got {len(merged)}"
)

# metres -> mm
for pair in ["HL", "HR", "LR"]:
    merged[f"err_{pair}_mm"] = merged[f"err_{pair}"] * 1000.0

E = merged[[
    "err_HL_mm",
    "err_HR_mm",
    "err_LR_mm",
]]

# ------------------------------------------------------------
# Per-sample best pair
# ------------------------------------------------------------
merged["best_pair"] = (
    E.idxmin(axis=1)
    .str.replace("err_", "", regex=False)
    .str.replace("_mm", "", regex=False)
)

merged["best_error_mm"] = E.min(axis=1)
merged["worst_error_mm"] = E.max(axis=1)
merged["pair_spread_mm"] = (
    merged["worst_error_mm"] -
    merged["best_error_mm"]
)

# second best
arr = np.sort(E.to_numpy(), axis=1)
merged["best_margin_mm"] = arr[:, 1] - arr[:, 0]

print("\n===== OVERALL ERROR =====")

for pair in ["HL", "HR", "LR"]:
    x = merged[f"err_{pair}_mm"]

    print(
        f"{pair}: "
        f"median={x.median():.4f} mm | "
        f"mean={x.mean():.4f} mm"
    )

# ------------------------------------------------------------
# Winner distribution
# ------------------------------------------------------------
print("\n===== SAMPLE-WISE BEST PAIR =====")

counts = (
    merged["best_pair"]
    .value_counts()
    .reindex(["HL", "HR", "LR"], fill_value=0)
)

for pair, n in counts.items():
    print(
        f"{pair}: {n} / {len(merged)} "
        f"({100*n/len(merged):.2f}%)"
    )

p = counts.to_numpy(dtype=float)
p /= p.sum()

entropy = -np.sum(
    np.where(p > 0, p * np.log2(p), 0)
)

print(f"Best-pair entropy = {entropy:.4f} bits")
print("Maximum possible for 3 pairs =", np.log2(3))

# ------------------------------------------------------------
# How much does pair choice matter?
# ------------------------------------------------------------
print("\n===== VIEW-PAIR SENSITIVITY =====")

print(
    "Median best-vs-worst spread = "
    f"{merged['pair_spread_mm'].median():.4f} mm"
)

print(
    "Mean best-vs-worst spread = "
    f"{merged['pair_spread_mm'].mean():.4f} mm"
)

print(
    "Median winner margin over 2nd best = "
    f"{merged['best_margin_mm'].median():.4f} mm"
)

print(
    "Mean winner margin over 2nd best = "
    f"{merged['best_margin_mm'].mean():.4f} mm"
)

# ------------------------------------------------------------
# Oracle pair vs best global fixed pair
# Best global pair is chosen ONLY from the three aggregate means.
# This is descriptive, not a learned method.
# ------------------------------------------------------------
means = {
    pair: merged[f"err_{pair}_mm"].mean()
    for pair in ["HL", "HR", "LR"]
}

fixed_best = min(means, key=means.get)

merged["fixed_best_error_mm"] = merged[
    f"err_{fixed_best}_mm"
]

merged["oracle_gain_mm"] = (
    merged["fixed_best_error_mm"]
    - merged["best_error_mm"]
)

den = merged["fixed_best_error_mm"].replace(0, np.nan)

merged["oracle_gain_pct"] = (
    100.0 *
    merged["oracle_gain_mm"] /
    den
)

print("\n===== FIXED PAIR vs PER-SAMPLE ORACLE =====")
print("Best GLOBAL fixed pair =", fixed_best)

print(
    f"Fixed {fixed_best}: "
    f"median={merged['fixed_best_error_mm'].median():.4f} mm | "
    f"mean={merged['fixed_best_error_mm'].mean():.4f} mm"
)

print(
    "Per-sample oracle: "
    f"median={merged['best_error_mm'].median():.4f} mm | "
    f"mean={merged['best_error_mm'].mean():.4f} mm"
)

print(
    "Oracle gain: "
    f"median={merged['oracle_gain_mm'].median():.4f} mm | "
    f"mean={merged['oracle_gain_mm'].mean():.4f} mm"
)

print(
    "Oracle relative gain: "
    f"median={merged['oracle_gain_pct'].median():.2f}% | "
    f"mean={merged['oracle_gain_pct'].mean():.2f}%"
)

print(
    "Oracle beats fixed pair on "
    f"{(merged['oracle_gain_mm'] > 0).mean()*100:.2f}% "
    "of clips"
)

# ------------------------------------------------------------
# Paired statistical tests
# ------------------------------------------------------------
print("\n===== PAIRED STATISTICS =====")

stat, p_friedman = friedmanchisquare(
    merged["err_HL_mm"],
    merged["err_HR_mm"],
    merged["err_LR_mm"],
)

print(
    f"Friedman: statistic={stat:.6g}, "
    f"p={p_friedman:.6g}"
)

for a, b in [
    ("HL", "HR"),
    ("HL", "LR"),
    ("HR", "LR"),
]:
    x = merged[f"err_{a}_mm"]
    y = merged[f"err_{b}_mm"]

    try:
        w, pval = wilcoxon(
            x,
            y,
            zero_method="wilcox",
            alternative="two-sided",
        )
    except ValueError:
        w, pval = np.nan, np.nan

    better_a = (x < y).mean() * 100
    better_b = (y < x).mean() * 100
    equal = (x == y).mean() * 100

    print(
        f"{a} vs {b}: "
        f"{a} lower={better_a:.2f}% | "
        f"{b} lower={better_b:.2f}% | "
        f"equal={equal:.2f}% | "
        f"Wilcoxon p={pval:.6g}"
    )

# ------------------------------------------------------------
# Save canonical merged table
# ------------------------------------------------------------
OUT = ROOT / "task0006_three_pair_merged_analysis.csv"

merged.to_csv(OUT, index=False)

print("\nOUTPUT =", OUT)
print("SUCCESS = True")
