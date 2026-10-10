import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = "diagnostics/results/behavior_radio_rigidity_rank.csv"
OUTDIR = "diagnostics/results/behavior_radio_rigidity_distribution"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(CSV)

assert len(df) > 0
assert "pred_mean_mm" in df.columns
assert "sample_key" in df.columns

x = df["pred_mean_mm"].astype(float).values

mean = float(np.mean(x))
median = float(np.median(x))
p75 = float(np.quantile(x, 0.75))
p90 = float(np.quantile(x, 0.90))
p95 = float(np.quantile(x, 0.95))
mx = float(np.max(x))
mn = float(np.min(x))

thr5 = float(np.mean(x > 5.0) * 100.0)
thr10 = float(np.mean(x > 10.0) * 100.0)
thr20 = float(np.mean(x > 20.0) * 100.0)
thr40 = float(np.mean(x > 40.0) * 100.0)

sample_00470 = "task-0000_episode_00000470-375:386"
sample_02530 = "task-0000_episode_00002530-300:311"

v00470 = df.loc[df["sample_key"] == sample_00470, "pred_mean_mm"]
v02530 = df.loc[df["sample_key"] == sample_02530, "pred_mean_mm"]

v00470 = float(v00470.iloc[0]) if len(v00470) else None
v02530 = float(v02530.iloc[0]) if len(v02530) else None

# summary csv
summary = pd.DataFrame([{
    "n_radio": len(df),
    "mean_mm": mean,
    "median_mm": median,
    "p75_mm": p75,
    "p90_mm": p90,
    "p95_mm": p95,
    "min_mm": mn,
    "max_mm": mx,
    "frac_gt_5mm_percent": thr5,
    "frac_gt_10mm_percent": thr10,
    "frac_gt_20mm_percent": thr20,
    "frac_gt_40mm_percent": thr40,
    "sample_00470_pred_mean_mm": v00470,
    "sample_02530_pred_mean_mm": v02530,
}])

summary_csv = os.path.join(OUTDIR, "radio_rigidity_distribution_summary.csv")
summary.to_csv(summary_csv, index=False)

print("=" * 100)
print("N_RADIO =", len(df))
print("mean_mm   =", mean)
print("median_mm =", median)
print("p75_mm    =", p75)
print("p90_mm    =", p90)
print("p95_mm    =", p95)
print("min_mm    =", mn)
print("max_mm    =", mx)

print("\nTHRESHOLD FRACTIONS")
print("> 5mm  =", thr5, "%")
print(">10mm  =", thr10, "%")
print(">20mm  =", thr20, "%")
print(">40mm  =", thr40, "%")

print("\nSPECIAL SAMPLES")
print("00470 pred_mean_mm =", v00470)
print("02530 pred_mean_mm =", v02530)

# histogram
plt.figure(figsize=(9, 6))
plt.hist(x, bins=40)

plt.axvline(median, linestyle="--", linewidth=2, label=f"median={median:.2f} mm")
plt.axvline(p90, linestyle="--", linewidth=2, label=f"p90={p90:.2f} mm")
plt.axvline(p95, linestyle="--", linewidth=2, label=f"p95={p95:.2f} mm")

if v00470 is not None:
    plt.axvline(v00470, linestyle="-", linewidth=2, label=f"00470={v00470:.2f} mm")
if v02530 is not None:
    plt.axvline(v02530, linestyle="-", linewidth=2, label=f"02530={v02530:.2f} mm")

plt.xlabel("Pred rigid residual mean (mm)")
plt.ylabel("Number of radio samples")
plt.title("Distribution of radio non-rigid prediction residuals")
plt.legend()

hist_png = os.path.join(OUTDIR, "radio_rigidity_distribution_hist.png")
plt.tight_layout()
plt.savefig(hist_png, dpi=200, bbox_inches="tight")
plt.close()

print("\nSUMMARY_CSV =", summary_csv)
print("HIST_PNG =", hist_png)
print("SUMMARY_EXISTS =", os.path.exists(summary_csv))
print("HIST_EXISTS =", os.path.exists(hist_png))
print("RADIO_RIGIDITY_DISTRIBUTION_SUCCESS = True")
