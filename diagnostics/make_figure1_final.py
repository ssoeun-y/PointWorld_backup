import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "diagnostics/results"
OUT = f"{R}/problem_dose_response"
ROOT = f"{R}/object_pa_mpjpe_acc_5task_current10457"
rng = np.random.default_rng(0)
BLUE, RED, GRAY = "#3a76b8", "#d95f3b", "#999999"

def boot(v, fn, n=400):
    v = np.asarray(v, float)
    s = fn(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return fn(v), np.percentile(s, 2.5), np.percentile(s, 97.5)

# ---------- (a) acc decile dose-response ----------
P = pd.read_csv(f"{ROOT}/analysis_5task/merged_per_clip_metrics_5task_clean.csv",
                usecols=["clip", "rms_acceleration_si_m_s2", "pa_mpjpe_objectmean_mm"]).drop_duplicates("clip").dropna()
P = P.rename(columns={"rms_acceleration_si_m_s2": "acc", "pa_mpjpe_objectmean_mm": "pa"})
L = pd.read_csv(f"{R}/action_prediction_sensitivity_all7987/prediction_shift_with_action_metrics_7987.csv",
                usecols=["clip", "rms_acceleration", "l2_moved_mm"]).drop_duplicates("clip").dropna()
L = L.rename(columns={"rms_acceleration": "acc"})
for d in (P, L):
    d["bin"] = pd.qcut(d["acc"].rank(method="first"), 10, labels=False)

def dec_curve(d, col):
    rows = [boot(d.loc[d["bin"] == b, col].values, np.median) for b in range(10)]
    c = np.array(rows); return c / c[0, 0]

cL, cP = dec_curve(L, "l2_moved_mm"), dec_curve(P, "pa")

# ---------- (b) rollout accumulation (per_frame, frames 1..10) ----------
F = pd.concat([pd.read_csv(f"{ROOT}/{t}/per_frame_metrics.csv",
               usecols=["task", "clip", "frame", "rms_acceleration_si_m_s2", "pa_mpjpe_mm"])
               for t in ["task-0000", "task-0001", "task-0006", "task-0008", "task-0010"]])
F = F[(F["frame"] >= 1) & (F["frame"] <= 10)].dropna()
cf = F.groupby(["task", "clip", "frame"], as_index=False).agg(
    acc=("rms_acceleration_si_m_s2", "mean"), pa=("pa_mpjpe_mm", "mean"))
ca = cf.groupby(["task", "clip"], as_index=False)["acc"].mean()
ca["pct"] = ca.groupby("task")["acc"].rank(pct=True)
ca["grp"] = np.where(ca["pct"] <= 0.25, "low", np.where(ca["pct"] > 0.75, "high", "mid"))
cf = cf.merge(ca[["task", "clip", "grp"]], on=["task", "clip"])
print("clips  low:", (ca.grp == "low").sum(), " high:", (ca.grp == "high").sum(), " frames:", sorted(cf.frame.unique().astype(int)))

def roll(g):
    rows = [boot(cf.loc[(cf.grp == g) & (cf.frame == f), "pa"].values, np.mean) for f in range(1, 11)]
    return np.array(rows)
rl, rh = roll("low"), roll("high")

# ---------- plot ----------
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9), dpi=300)
x = np.arange(1, 11)
for c, col, lab in [(cL, BLUE, "moved-L2"), (cP, RED, "PA-MPJPE")]:
    ax[0].plot(x, c[:, 0], "-o", ms=3.5, color=col, label=lab)
    ax[0].fill_between(x, c[:, 1], c[:, 2], color=col, alpha=0.15)
ax[0].set_xticks(x); ax[0].set_xlabel("Action RMS acceleration decile (low → high)")
ax[0].set_ylabel("Error / lowest-decile error")
ax[0].set_title("(a) Higher dynamics → larger error", fontsize=9, fontweight="bold")
ax[0].legend(frameon=False, loc="upper left")
ax[0].text(0.97, 0.05, f"D10 vs D1:  L2 ×{cL[-1,0]:.1f},  PA ×{cP[-1,0]:.1f}",
           transform=ax[0].transAxes, ha="right", fontsize=8)
for c, col, lab in [(rl, BLUE, "Low dynamics"), (rh, RED, "High dynamics")]:
    ax[1].plot(x, c[:, 0], "-o", ms=3.5, color=col, label=lab)
    ax[1].fill_between(x, c[:, 1], c[:, 2], color=col, alpha=0.15)
ax[1].set_xticks(x); ax[1].set_xlabel("Rollout frame"); ax[1].set_ylabel("PA-MPJPE (mm)")
ax[1].set_title("(b) Shape error accumulates over time", fontsize=9, fontweight="bold")
ax[1].legend(frameon=False, loc="upper left")
ax[1].text(0.97, 0.05, f"frame 10: {rl[-1,0]:.2f} → {rh[-1,0]:.2f} mm  (×{rh[-1,0]/rl[-1,0]:.1f})",
           transform=ax[1].transAxes, ha="right", fontsize=8)
for a in ax:
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUT}/figure1_final.png"); plt.savefig(f"{OUT}/figure1_final.pdf")
print(f"(a) D10/D1  moved-L2 x{cL[-1,0]:.2f} | PA x{cP[-1,0]:.2f}")
print(f"(b) frame10 low={rl[-1,0]:.3f} high={rh[-1,0]:.3f} ratio x{rh[-1,0]/rl[-1,0]:.2f}")
print("saved:", f"{OUT}/figure1_final.png / .pdf")
