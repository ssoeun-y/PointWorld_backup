import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

R = "diagnostics/results"
OUT = f"{R}/problem_dose_response"
P = pd.read_csv(f"{R}/object_pa_mpjpe_acc_5task_current10457/analysis_5task/merged_per_clip_metrics_5task_clean.csv",
                usecols=["task", "clip", "rms_acceleration_si_m_s2", "mpjpe_objectmean_mm", "pa_mpjpe_objectmean_mm"])
L = pd.read_csv(f"{R}/action_prediction_sensitivity_all7987/prediction_shift_with_action_metrics_7987.csv",
                usecols=["clip", "rms_acceleration", "l2_moved_mm"])
P = P.drop_duplicates("clip"); L = L.drop_duplicates("clip")
df = P.merge(L, on="clip", how="inner").dropna().reset_index(drop=True)
print(f"PA table {len(P)} | L2 table {len(L)} | joined N = {len(df)}")
print(df.groupby("task").size().to_string())
print("acc sanity spearman:", round(spearmanr(df["rms_acceleration_si_m_s2"], df["rms_acceleration"])[0], 4))

df = df.rename(columns={"rms_acceleration_si_m_s2": "acc", "mpjpe_objectmean_mm": "mpjpe", "pa_mpjpe_objectmean_mm": "pa"})
df["ratio"] = df["pa"] / df["mpjpe"]
for name, col in [("moved-L2", "l2_moved_mm"), ("MPJPE", "mpjpe"), ("PA-MPJPE", "pa"), ("PA/MPJPE", "ratio")]:
    r, p = spearmanr(df["acc"], df[col])
    print(f"pooled spearman acc vs {name}: rho={r:+.4f} p={p:.2e}")
    for t, g in df.groupby("task"):
        print(f"   {t}: rho={spearmanr(g['acc'], g[col])[0]:+.4f} N={len(g)}")

rng = np.random.default_rng(0)
def med_ci(v, n=400):
    v = np.asarray(v, float)
    m = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return np.median(v), np.percentile(m, 2.5), np.percentile(m, 97.5)

df["bin"] = pd.qcut(df["acc"].rank(method="first"), 10, labels=False)
df["wbin"] = np.minimum((df.groupby("task")["acc"].rank(pct=True) * 10).astype(int), 9)
for c in ["l2_moved_mm", "mpjpe", "pa"]:
    df[c + "_rel"] = df[c] / df.groupby("task")[c].transform("median")

def curve(binc, col, norm_first):
    rows = []
    for b in range(10):
        v = df.loc[df[binc] == b, col].values
        rows.append((b + 1, len(v), *med_ci(v)))
    c = pd.DataFrame(rows, columns=["bin", "n", "med", "lo", "hi"])
    if norm_first:
        f = c["med"].iloc[0]; c[["med", "lo", "hi"]] = c[["med", "lo", "hi"]] / f
    return c

COL = {"moved-L2": "#3a76b8", "MPJPE": "#999999", "PA-MPJPE": "#d95f3b"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4), dpi=200)
tabs = []
for a, binc, cols, nf, ttl, yl in [
    (ax[0], "bin", {"moved-L2": "l2_moved_mm", "MPJPE": "mpjpe", "PA-MPJPE": "pa"}, True,
     "(a) pooled: acc decile", "median / lowest-decile median"),
    (ax[1], "wbin", {"moved-L2": "l2_moved_mm_rel", "MPJPE": "mpjpe_rel", "PA-MPJPE": "pa_rel"}, False,
     "(b) within-task acc decile", "median / task median")]:
    for lab, col in cols.items():
        c = curve(binc, col, nf)
        a.plot(c["bin"], c["med"], "-o", ms=4, color=COL[lab], label=lab)
        a.fill_between(c["bin"], c["lo"], c["hi"], color=COL[lab], alpha=0.15)
        c["panel"], c["metric"] = ttl, lab; tabs.append(c)
    a.set_title(ttl); a.set_ylabel(yl); a.legend(frameon=False)
c = curve("bin", "ratio", False)
ax[2].plot(c["bin"], c["med"], "-o", ms=4, color="#2a9d5c")
ax[2].fill_between(c["bin"], c["lo"], c["hi"], color="#2a9d5c", alpha=0.15)
ax[2].set_title("(c) PA / MPJPE (error left after alignment)"); ax[2].set_ylabel("median ratio")
c["panel"], c["metric"] = "(c)", "PA/MPJPE"; tabs.append(c)
for a in ax:
    a.set_xlabel("RMS acceleration decile (1 = low, 10 = high)"); a.set_xticks(range(1, 11))
    for s in ("top", "right"): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/dose_response_final.png")
T = pd.concat(tabs); T.to_csv(f"{OUT}/dose_response_final.csv", index=False)
print(T.round(3).to_string(index=False))
print("saved:", f"{OUT}/dose_response_final.png")

f = f"{R}/object_pa_mpjpe_acc_5task_current10457/task-0001/per_frame_metrics.csv"
print("\nper_frame_metrics columns:", pd.read_csv(f, nrows=2).columns.tolist())
