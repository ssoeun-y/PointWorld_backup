import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

R = "diagnostics/results"
OUT = f"{R}/problem_dose_response"
A = pd.read_csv(f"{R}/action_prediction_sensitivity_all7987/prediction_shift_with_action_metrics_7987.csv",
                usecols=["clip", "task", "rms_acceleration", "l2_moved_mm"])
B = pd.read_csv(f"{R}/acc_only_boundary_pa_final_current10457/combined_5task_per_clip_metrics.csv",
                usecols=["clip", "original_acc_si_m_s2", "base_pa_mean_mm"])
print("A rows", len(A), "dup clip", A["clip"].duplicated().sum(), "| B rows", len(B), "dup clip", B["clip"].duplicated().sum())
A = A.drop_duplicates("clip"); B = B.drop_duplicates("clip")
df = A.merge(B, on="clip", how="inner").dropna().reset_index(drop=True)
print("joined N =", len(df)); print(df.groupby("task").size().to_string())
rho_acc = spearmanr(df["rms_acceleration"], df["original_acc_si_m_s2"])[0]
print(f"sanity: spearman(rms_acceleration, original_acc_si_m_s2) = {rho_acc:.4f}")

for name, col in [("moved-L2", "l2_moved_mm"), ("PA-MPJPE", "base_pa_mean_mm")]:
    r, p = spearmanr(df["rms_acceleration"], df[col])
    print(f"pooled spearman acc vs {name}: rho={r:+.4f} p={p:.2e}")
    for t, g in df.groupby("task"):
        print(f"   {t}: rho={spearmanr(g['rms_acceleration'], g[col])[0]:+.4f} N={len(g)}")

rng = np.random.default_rng(0)
def med_ci(v, n=300):
    v = np.asarray(v, float)
    m = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return np.median(v), np.percentile(m, 2.5), np.percentile(m, 97.5)

df["bin"] = pd.qcut(df["rms_acceleration"].rank(method="first"), 10, labels=False)
df["pct"] = df.groupby("task")["rms_acceleration"].rank(pct=True)
df["wbin"] = np.minimum((df["pct"] * 10).astype(int), 9)
df["l2_rel"] = df["l2_moved_mm"] / df.groupby("task")["l2_moved_mm"].transform("median")
df["pa_rel"] = df["base_pa_mean_mm"] / df.groupby("task")["base_pa_mean_mm"].transform("median")

def curve(binc, col, rel_first):
    rows = []
    for b in range(10):
        v = df.loc[df[binc] == b, col].values
        rows.append((b + 1, len(v), *med_ci(v)))
    c = pd.DataFrame(rows, columns=["bin", "n", "med", "lo", "hi"])
    if rel_first:
        f = c["med"].iloc[0]; c[["med", "lo", "hi"]] = c[["med", "lo", "hi"]] / f
    return c

fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=200)
tabs = []
for a, binc, cols, rel, ttl, yl in [
    (ax[0], "bin", ("l2_moved_mm", "base_pa_mean_mm"), True,
     "(a) pooled: acc decile", "median, relative to lowest decile"),
    (ax[1], "wbin", ("l2_rel", "pa_rel"), False,
     "(b) within-task acc decile", "median, relative to task median")]:
    for col, lab, colr in [(cols[0], "moved-L2", "#3a76b8"), (cols[1], "PA-MPJPE", "#d95f3b")]:
        c = curve(binc, col, rel)
        a.plot(c["bin"], c["med"], "-o", color=colr, label=lab)
        a.fill_between(c["bin"], c["lo"], c["hi"], color=colr, alpha=0.15)
        c["panel"], c["metric"] = ttl, lab; tabs.append(c)
    a.set_title(ttl); a.set_xlabel("RMS acceleration decile (1 = low, 10 = high)")
    a.set_ylabel(yl); a.set_xticks(range(1, 11)); a.legend(frameon=False)
    for s in ("top", "right"): a.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/dose_response_l2_vs_pa.png")
T = pd.concat(tabs); T.to_csv(f"{OUT}/dose_response_table.csv", index=False)
print(T.round(3).to_string(index=False))
print("saved:", f"{OUT}/dose_response_l2_vs_pa.png")
