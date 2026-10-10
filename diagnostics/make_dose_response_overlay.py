import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

R = "diagnostics/results"
OUT = f"{R}/problem_dose_response"
ROOT = f"{R}/object_pa_mpjpe_acc_5task_current10457"

P = pd.read_csv(f"{ROOT}/analysis_5task/merged_per_clip_metrics_5task_clean.csv",
                usecols=["task", "clip", "rms_acceleration_si_m_s2", "mpjpe_objectmean_mm", "pa_mpjpe_objectmean_mm"]
                ).drop_duplicates("clip").dropna()
P = P.rename(columns={"rms_acceleration_si_m_s2": "acc", "mpjpe_objectmean_mm": "mpjpe", "pa_mpjpe_objectmean_mm": "pa"})
P["ratio"] = P["pa"] / P["mpjpe"]
L = pd.read_csv(f"{R}/action_prediction_sensitivity_all7987/prediction_shift_with_action_metrics_7987.csv",
                usecols=["clip", "task", "rms_acceleration", "l2_moved_mm"]).drop_duplicates("clip").dropna()
L = L.rename(columns={"rms_acceleration": "acc"})
print(f"PA/MPJPE table N={len(P)} | moved-L2 table N={len(L)}")

F = []
for t in ["task-0000", "task-0001", "task-0006", "task-0008", "task-0010"]:
    d = pd.read_csv(f"{ROOT}/{t}/per_frame_metrics.csv",
                    usecols=["task", "clip", "frame", "rms_acceleration_si_m_s2", "fitted_scale"])
    F.append(d[d["frame"] == 10])
F = pd.concat(F).dropna().rename(columns={"rms_acceleration_si_m_s2": "acc"})
F["abslogs"] = np.abs(np.log(F["fitted_scale"]))
print(f"per_frame (frame=10) rows N={len(F)} | fitted_scale median={F['fitted_scale'].median():.4f} "
      f"p5={F['fitted_scale'].quantile(0.05):.4f} p95={F['fitted_scale'].quantile(0.95):.4f}")

rng = np.random.default_rng(0)
def med_ci(v, n=400):
    v = np.asarray(v, float)
    m = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return np.median(v), np.percentile(m, 2.5), np.percentile(m, 97.5)

def add_bins(d):
    d["bin"] = pd.qcut(d["acc"].rank(method="first"), 10, labels=False)
    d["wbin"] = np.minimum((d.groupby("task")["acc"].rank(pct=True) * 10).astype(int), 9)
    return d
P, L, F = add_bins(P), add_bins(L), add_bins(F)
for c in ["mpjpe", "pa"]:
    P[c + "_rel"] = P[c] / P.groupby("task")[c].transform("median")
L["l2_rel"] = L["l2_moved_mm"] / L.groupby("task")["l2_moved_mm"].transform("median")

def curve(d, binc, col, norm_first=False):
    rows = []
    for b in range(10):
        v = d.loc[d[binc] == b, col].values
        rows.append((b + 1, len(v), *med_ci(v)))
    c = pd.DataFrame(rows, columns=["bin", "n", "med", "lo", "hi"])
    if norm_first:
        f = c["med"].iloc[0]; c[["med", "lo", "hi"]] = c[["med", "lo", "hi"]] / f
    return c

COL = {"moved-L2": "#3a76b8", "MPJPE": "#999999", "PA-MPJPE": "#d95f3b"}
fig, ax = plt.subplots(2, 2, figsize=(11, 8), dpi=200)
tabs = []
def draw(a, items, ttl, yl):
    for lab, d, binc, col, nf in items:
        c = curve(d, binc, col, nf)
        a.plot(c["bin"], c["med"], "-o", ms=4, color=COL[lab], label=lab)
        a.fill_between(c["bin"], c["lo"], c["hi"], color=COL[lab], alpha=0.15)
        c["panel"], c["metric"] = ttl, lab; tabs.append(c)
    a.set_title(ttl); a.set_ylabel(yl); a.legend(frameon=False)

draw(ax[0, 0], [("moved-L2", L, "bin", "l2_moved_mm", True), ("MPJPE", P, "bin", "mpjpe", True),
                ("PA-MPJPE", P, "bin", "pa", True)],
     "(a) pooled acc decile", "median / lowest-decile median")
draw(ax[0, 1], [("moved-L2", L, "wbin", "l2_rel", False), ("MPJPE", P, "wbin", "mpjpe_rel", False),
                ("PA-MPJPE", P, "wbin", "pa_rel", False)],
     "(b) within-task acc decile", "median / task median")
c = curve(P, "bin", "ratio")
ax[1, 0].plot(c["bin"], c["med"], "-o", ms=4, color="#2a9d5c")
ax[1, 0].fill_between(c["bin"], c["lo"], c["hi"], color="#2a9d5c", alpha=0.15)
ax[1, 0].set_title("(c) PA / MPJPE (error left after alignment)"); ax[1, 0].set_ylabel("median ratio")
c["panel"], c["metric"] = "(c)", "PA/MPJPE"; tabs.append(c)
c = curve(F, "bin", "abslogs")
ax[1, 1].plot(c["bin"], c["med"], "-o", ms=4, color="#8e44ad")
ax[1, 1].fill_between(c["bin"], c["lo"], c["hi"], color="#8e44ad", alpha=0.15)
ax[1, 1].set_title("(d) |log fitted_scale| at frame 10"); ax[1, 1].set_ylabel("median |log s| (0 = no scale change)")
c["panel"], c["metric"] = "(d)", "abslogs"; tabs.append(c)
for a in ax.ravel():
    a.set_xlabel("RMS acceleration decile (1 = low, 10 = high)"); a.set_xticks(range(1, 11))
    for s in ("top", "right"): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/dose_response_overlay.png")
T = pd.concat(tabs); T.to_csv(f"{OUT}/dose_response_overlay.csv", index=False)

print("\n== Spearman(acc, metric) ==")
for name, d, col in [("moved-L2 (7987)", L, "l2_moved_mm"), ("MPJPE (7947)", P, "mpjpe"),
                     ("PA-MPJPE (7947)", P, "pa"), ("PA/MPJPE (7947)", P, "ratio"),
                     ("|log fitted_scale| (frame10)", F, "abslogs")]:
    r, p = spearmanr(d["acc"], d[col])
    print(f"{name}: pooled rho={r:+.4f} p={p:.2e} | " +
          " ".join(f"{t[-4:]}:{spearmanr(g['acc'], g[col])[0]:+.2f}" for t, g in d.groupby("task")))
print("\ndecile1 -> decile10 (median ratio):")
for name, d, col in [("moved-L2", L, "l2_moved_mm"), ("MPJPE", P, "mpjpe"), ("PA-MPJPE", P, "pa")]:
    c = curve(d, "bin", col); print(f"  {name}: x{c['med'].iloc[-1] / c['med'].iloc[0]:.2f}")
print("\n(c) ratio by decile:", curve(P, "bin", "ratio")["med"].round(3).tolist())
print("(d) |log s| by decile:", curve(F, "bin", "abslogs")["med"].round(4).tolist())
print("saved:", f"{OUT}/dose_response_overlay.png")
