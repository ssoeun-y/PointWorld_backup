from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

B = 0.24580418

ROOT = Path("diagnostics/results/rigid_projection_repair_current10457")
REPAIR_CSV = ROOT / "combined_5task_repair_per_clip.csv"
ACC_CSV = Path(
    "diagnostics/results/current_behavior_established_acc_10457/"
    "per_sample_current_established_acc_10457.csv"
)

repair = pd.read_csv(REPAIR_CSV)
acc = pd.read_csv(ACC_CSV)

need_r = {
    "task", "clip", "raw_mp", "repaired_mp",
    "residual_mm", "delta_mm", "rel_pct"
}
need_a = {
    "task", "clip", "rms_acceleration_si_m_s2"
}

assert need_r.issubset(repair.columns), repair.columns.tolist()
assert need_a.issubset(acc.columns), acc.columns.tolist()

if "benefit_mm" not in repair.columns:
    repair["benefit_mm"] = repair["raw_mp"] - repair["repaired_mp"]

d = repair.merge(
    acc[["task", "clip", "rms_acceleration_si_m_s2"]],
    on=["task", "clip"],
    how="inner",
    validate="one_to_one",
)

if len(d) != 7947:
    raise RuntimeError(f"Expected 7947 clips, got {len(d)}")

A = d["rms_acceleration_si_m_s2"]
d["above_B"] = A > B
d["acc_multiple_B"] = A / B

print("=" * 78)
print("POPULATION")
print("=" * 78)
print("N =", len(d))
print("Boundary B =", B, "m/s^2")
print("Below/equal B =", int((~d["above_B"]).sum()))
print("Above B       =", int(d["above_B"].sum()))
print("Above B %%    = %.2f%%" % (100*d["above_B"].mean()))

# ------------------------------------------------------------
# B below vs above
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("BELOW B vs ABOVE B")
print("=" * 78)

rows = []

for name, g in [
    ("<=B", d[~d["above_B"]]),
    (">B", d[d["above_B"]]),
]:
    rows.append({
        "group": name,
        "N": len(g),
        "acc_median": g["rms_acceleration_si_m_s2"].median(),
        "residual_median_mm": g["residual_mm"].median(),
        "benefit_median_mm": g["benefit_mm"].median(),
        "paired_rel_median_pct": g["rel_pct"].median(),
        "repair_lower_pct": 100*(g["benefit_mm"] > 0).mean(),
        "raw_mpjpe_median_mm": g["raw_mp"].median(),
    })

tab = pd.DataFrame(rows)
print(tab.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

lo = d[~d["above_B"]]
hi = d[d["above_B"]]

print("\nRatios (>B / <=B)")
print(
    "Residual median ratio = %.3fx"
    % (hi["residual_mm"].median() / lo["residual_mm"].median())
)
print(
    "Repair-benefit median ratio = %.3fx"
    % (hi["benefit_mm"].median() / lo["benefit_mm"].median())
)

for metric in ["residual_mm", "benefit_mm", "rel_pct"]:
    u = mannwhitneyu(
        hi[metric],
        lo[metric],
        alternative="two-sided"
    )
    print(
        f"{metric}: Mann-Whitney p = {u.pvalue:.6e}"
    )

# ------------------------------------------------------------
# Dose response by multiples of B
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("DOSE RESPONSE BY B MULTIPLE")
print("=" * 78)

edges = [-np.inf, 0.8, 1.0, 1.2, 1.5, 2.0, np.inf]
labels = [
    "<0.8B",
    "0.8-1.0B",
    "1.0-1.2B",
    "1.2-1.5B",
    "1.5-2.0B",
    ">=2.0B",
]

d["B_bin"] = pd.cut(
    d["acc_multiple_B"],
    bins=edges,
    labels=labels,
    right=False,
)

dose_rows = []

for label in labels:
    g = d[d["B_bin"] == label]
    if len(g) == 0:
        continue

    dose_rows.append({
        "bin": label,
        "N": len(g),
        "acc_med": g["rms_acceleration_si_m_s2"].median(),
        "residual_med_mm": g["residual_mm"].median(),
        "benefit_med_mm": g["benefit_mm"].median(),
        "paired_rel_med_pct": g["rel_pct"].median(),
        "repair_lower_pct": 100*(g["benefit_mm"] > 0).mean(),
        "raw_mpjpe_med_mm": g["raw_mp"].median(),
    })

dose = pd.DataFrame(dose_rows)
print(dose.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# ------------------------------------------------------------
# Correlations
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("ACC -> RESIDUAL / REPAIR BENEFIT")
print("=" * 78)

for metric in ["residual_mm", "benefit_mm"]:
    rho, p = spearmanr(
        d["rms_acceleration_si_m_s2"],
        d[metric]
    )
    print(
        f"Pooled Acc vs {metric}: "
        f"rho={rho:.6f}, p={p:.6e}"
    )

print("\nTaskwise:")
for task, g in d.groupby("task"):
    r1, p1 = spearmanr(
        g["rms_acceleration_si_m_s2"],
        g["residual_mm"]
    )
    r2, p2 = spearmanr(
        g["rms_acceleration_si_m_s2"],
        g["benefit_mm"]
    )
    print(
        task,
        f"N={len(g)}",
        f"Acc-residual rho={r1:.4f}",
        f"Acc-benefit rho={r2:.4f}",
    )

# Within-task rank control
d["acc_rank_task"] = d.groupby("task")[
    "rms_acceleration_si_m_s2"
].rank(pct=True)

for metric in ["residual_mm", "benefit_mm"]:
    col = f"{metric}_rank_task"
    d[col] = d.groupby("task")[metric].rank(pct=True)
    rho, p = spearmanr(d["acc_rank_task"], d[col])
    print(
        f"Within-task rank Acc vs {metric}: "
        f"rho={rho:.6f}, p={p:.6e}"
    )

# ------------------------------------------------------------
# Can B selectively repair instead of repairing everything?
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("SELECTIVE REPAIR USING BOUNDARY")
print("=" * 78)

always = d["repaired_mp"].to_numpy()
raw = d["raw_mp"].to_numpy()

selective = np.where(
    d["above_B"],
    d["repaired_mp"],
    d["raw_mp"],
)

raw_mean = raw.mean()
always_mean = always.mean()
sel_mean = selective.mean()

always_gain = raw_mean - always_mean
sel_gain = raw_mean - sel_mean

print("Raw mean MPJPE       = %.6f mm" % raw_mean)
print("Always-repair mean   = %.6f mm" % always_mean)
print("B-selective mean     = %.6f mm" % sel_mean)
print("Always repair gain   = %.6f mm" % always_gain)
print("B-selective gain     = %.6f mm" % sel_gain)

if always_gain > 0:
    print(
        "Net gain captured by B = %.2f%%"
        % (100 * sel_gain / always_gain)
    )

positive_gain = np.clip(d["benefit_mm"].to_numpy(), 0, None)
positive_above = positive_gain[d["above_B"].to_numpy()].sum()

if positive_gain.sum() > 0:
    print(
        "Positive repair-benefit mass above B = %.2f%%"
        % (100 * positive_above / positive_gain.sum())
    )

print(
    "Only %.2f%% of clips would be repaired"
    % (100*d["above_B"].mean())
)

# Threshold sweep
print("\nThreshold sweep:")
sweep = []

for mult in [0.8, 1.0, 1.2, 1.5, 2.0]:
    sel = A >= mult*B
    pred = np.where(sel, d["repaired_mp"], d["raw_mp"])
    gain = raw_mean - pred.mean()

    sweep.append({
        "threshold": f"{mult:.1f}B",
        "selected_pct": 100*sel.mean(),
        "mean_gain_mm": gain,
        "gain_capture_pct":
            100*gain/always_gain if always_gain > 0 else np.nan,
        "selected_repair_success_pct":
            100*(d.loc[sel, "benefit_mm"] > 0).mean(),
    })

sweep = pd.DataFrame(sweep)
print(sweep.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# save
tab.to_csv(ROOT / "boundary_below_above_summary.csv", index=False)
dose.to_csv(ROOT / "boundary_dose_response_summary.csv", index=False)
sweep.to_csv(ROOT / "boundary_selective_repair_sweep.csv", index=False)
d.to_csv(ROOT / "boundary_repair_joined_7947.csv", index=False)

print("\nSAVED =", ROOT)
