import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

R = "diagnostics/results"
OUT = f"{R}/problem_dose_response"
df = pd.read_csv(f"{R}/action_prediction_sensitivity_all7987/prediction_shift_with_action_metrics_7987.csv",
                 usecols=["clip", "task", "rms_acceleration", "l2_moved_mm"]).dropna()
df = df.drop_duplicates("clip").reset_index(drop=True)
print("N =", len(df)); print(df.groupby("task").size().to_string())
r, p = spearmanr(df["rms_acceleration"], df["l2_moved_mm"])
print(f"pooled spearman acc vs moved-L2: rho={r:+.4f} p={p:.2e}")
for t, g in df.groupby("task"):
    print(f"   {t}: rho={spearmanr(g['rms_acceleration'], g['l2_moved_mm'])[0]:+.4f} N={len(g)}")

rng = np.random.default_rng(0)
def med_ci(v, n=400):
    v = np.asarray(v, float)
    m = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return np.median(v), np.percentile(m, 2.5), np.percentile(m, 97.5)

df["bin"] = pd.qcut(df["rms_acceleration"].rank(method="first"), 10, labels=False)
df["wbin"] = np.minimum((df.groupby("task")["rms_acceleration"].rank(pct=True) * 10).astype(int), 9)
df["l2_rel"] = df["l2_moved_mm"] / df.groupby("task")["l2_moved_mm"].transform("median")

fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=200)
rows = []
for a, binc, col, ttl, yl in [
    (ax[0], "bin", "l2_moved_mm", "(a) pooled: acc decile", "median moved-L2 (mm)"),
    (ax[1], "wbin", "l2_rel", "(b) within-task acc decile", "moved-L2 / task median")]:
    xs, ms, los, his = [], [], [], []
    for b in range(10):
        v = df.loc[df[binc] == b, col].values
        m, lo, hi = med_ci(v)
        xs.append(b + 1); ms.append(m); los.append(lo); his.append(hi)
        rows.append((ttl, b + 1, len(v), m, lo, hi))
    a.plot(xs, ms, "-o", color="#3a76b8"); a.fill_between(xs, los, his, color="#3a76b8", alpha=0.15)
    a.set_title(ttl); a.set_xlabel("RMS acceleration decile (1 = low, 10 = high)"); a.set_ylabel(yl)
    a.set_xticks(range(1, 11))
    for s in ("top", "right"): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/dose_response_l2_only_7987.png")
T = pd.DataFrame(rows, columns=["panel", "bin", "n", "med", "lo", "hi"])
T.to_csv(f"{OUT}/dose_response_l2_only_7987.csv", index=False)
print(T.round(3).to_string(index=False))
print("saved:", f"{OUT}/dose_response_l2_only_7987.png")
