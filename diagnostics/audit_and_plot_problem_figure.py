from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# FIXED CURRENT EXPERIMENT
# ============================================================

TASKS = [
    "task-0000",
    "task-0001",
    "task-0006",
    "task-0008",
    "task-0010",
]

ACC_CSV = Path(
    "diagnostics/results/current_behavior_established_acc_10457/"
    "per_sample_current_established_acc_10457.csv"
)

PA_ROOT = Path(
    "diagnostics/results/object_pa_mpjpe_acc_5task_current10457"
)

OUT_ROOT = Path(
    "diagnostics/results/problem_figure_final_audited"
)

OUT_ROOT.mkdir(parents=True, exist_ok=True)

EXPECTED_N = 7947
EXPECTED_RHO = 0.420267
EXPECTED_LOW = 1.02432
EXPECTED_HIGH = 2.38449

RNG = np.random.default_rng(42)


def detect(df, names):
    table = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in table:
            return table[name.lower()]
    return None


# ============================================================
# 1. RMS ACC
# ============================================================

acc = pd.read_csv(ACC_CSV)

clip_c = detect(acc, ["clip", "key", "__key__", "sample"])
task_c = detect(acc, ["task"])
acc_c = detect(
    acc,
    [
        "rms_acceleration_si_m_s2",
        "rms_acc_si_m_s2",
        "rms_acceleration_si",
        "rms_acc",
    ],
)

if clip_c is None or acc_c is None:
    raise RuntimeError(
        f"ACC columns not recognized:\n{list(acc.columns)}"
    )

A = pd.DataFrame({
    "clip": acc[clip_c].astype(str),
    "rms_acc": pd.to_numeric(
        acc[acc_c],
        errors="coerce",
    ),
})

if task_c is not None:
    A["task"] = acc[task_c].astype(str)
else:
    A["task"] = (
        A["clip"]
        .str.extract(r"(task-\d{4})")[0]
    )

A = (
    A.dropna(subset=["task", "clip", "rms_acc"])
     .drop_duplicates(["task", "clip"])
)


# ============================================================
# 2. FRAME-WISE PA
# ============================================================

frames = []

for task in TASKS:

    f = PA_ROOT / task / "per_frame_metrics.csv"

    if not f.exists():
        raise RuntimeError(f"Missing file: {f}")

    d = pd.read_csv(f)

    cc = detect(d, ["clip", "key", "__key__", "sample"])
    fc = detect(d, ["frame", "t", "step"])
    pc = detect(d, ["pa_mpjpe_mm", "pa_mpjpe", "pa_mm"])

    if cc is None or fc is None or pc is None:
        raise RuntimeError(
            f"\nCould not identify columns in {f}\n"
            f"{list(d.columns)}"
        )

    z = pd.DataFrame({
        "task": task,
        "clip": d[cc].astype(str),
        "frame": pd.to_numeric(
            d[fc],
            errors="coerce",
        ),
        "pa": pd.to_numeric(
            d[pc],
            errors="coerce",
        ),
    })

    z = z.dropna()

    frames.append(z)

F = pd.concat(
    frames,
    ignore_index=True,
)

F["frame"] = F["frame"].astype(int)


# ============================================================
# 3. CLIP-LEVEL PA
# ============================================================

P = (
    F.groupby(["task", "clip"], as_index=False)
     .agg(
         pa_mean=("pa", "mean"),
         n_frames=("frame", "nunique"),
     )
)

D = P.merge(
    A,
    on=["task", "clip"],
    how="inner",
)

D = D[
    np.isfinite(D["rms_acc"])
    & np.isfinite(D["pa_mean"])
].copy()


# ============================================================
# 4. HARD AUDIT: ESTABLISHED RESULTS
# ============================================================

rho = D["rms_acc"].corr(
    D["pa_mean"],
    method="spearman",
)

print()
print("=" * 88)
print("AUDIT 1 — CURRENT POPULATION / MAIN CORRELATION")
print("=" * 88)
print("N clips       :", len(D))
print("Spearman rho  :", rho)
print("Frame rows    :", len(F))
print(
    "Frames/clip   :",
    sorted(D["n_frames"].unique().tolist())
)

if len(D) != EXPECTED_N:
    raise RuntimeError(
        f"N FAIL: expected {EXPECTED_N}, got {len(D)}"
    )

if abs(rho - EXPECTED_RHO) > 0.005:
    raise RuntimeError(
        f"RHO FAIL: expected ~{EXPECTED_RHO}, got {rho}"
    )

if not (D["n_frames"] == 10).all():
    raise RuntimeError(
        "FRAME COUNT FAIL: not every clip has 10 frames."
    )


# ============================================================
# 5. DEFINE LOW / HIGH INSIDE EACH TASK
# ============================================================

groups = []
task_clip_stats = []

for task in TASKS:

    d = D[D["task"] == task].copy()

    q25 = d["rms_acc"].quantile(0.25)
    q75 = d["rms_acc"].quantile(0.75)

    low = d[d["rms_acc"] <= q25].copy()
    high = d[d["rms_acc"] >= q75].copy()

    overlap = set(low["clip"]) & set(high["clip"])

    if overlap:
        raise RuntimeError(
            f"{task}: LOW/HIGH overlap = {len(overlap)}"
        )

    task_clip_stats.append({
        "task": task,
        "q25_acc": q25,
        "q75_acc": q75,
        "n_low": len(low),
        "n_high": len(high),
        "low_clip_pa": low["pa_mean"].median(),
        "high_clip_pa": high["pa_mean"].median(),
    })

    for clip in low["clip"]:
        groups.append({
            "task": task,
            "clip": clip,
            "group": "Low dynamics",
        })

    for clip in high["clip"]:
        groups.append({
            "task": task,
            "clip": clip,
            "group": "High dynamics",
        })

G = pd.DataFrame(groups)
TASK_CLIP = pd.DataFrame(task_clip_stats)


# ============================================================
# 6. REPRODUCE ESTABLISHED LOW25 / HIGH25 NUMBERS
# ============================================================

DG = D.merge(
    G,
    on=["task", "clip"],
    how="inner",
)

low_clip = DG[
    DG["group"] == "Low dynamics"
]["pa_mean"]

high_clip = DG[
    DG["group"] == "High dynamics"
]["pa_mean"]

low_med = float(low_clip.median())
high_med = float(high_clip.median())
ratio = high_med / low_med

print()
print("=" * 88)
print("AUDIT 2 — ESTABLISHED LOW25 / HIGH25 CLIP-LEVEL RESULT")
print("=" * 88)
print("Low  median :", low_med)
print("High median :", high_med)
print("Ratio       :", ratio)
print()
print(TASK_CLIP.to_string(index=False))

if abs(low_med - EXPECTED_LOW) > 0.005:
    raise RuntimeError(
        f"LOW FAIL: expected ~{EXPECTED_LOW}, got {low_med}"
    )

if abs(high_med - EXPECTED_HIGH) > 0.005:
    raise RuntimeError(
        f"HIGH FAIL: expected ~{EXPECTED_HIGH}, got {high_med}"
    )


# ============================================================
# 7. NEW ANALYSIS FOR PANEL (b)
#
# IMPORTANT:
# This is genuinely recomputed from the low/high clip groups.
# No old pooled rollout curve is reused.
# ============================================================

FG = F.merge(
    G,
    on=["task", "clip"],
    how="inner",
)

# Verify exactly 10 rows per selected clip
selected_counts = (
    FG.groupby(["task", "clip", "group"])
      .size()
)

if not (selected_counts == 10).all():
    raise RuntimeError(
        "Selected low/high clips do not all have exactly 10 frame rows."
    )

frame_values = sorted(FG["frame"].unique().tolist())

print()
print("=" * 88)
print("AUDIT 3 — LOW/HIGH FRAME-WISE ROLLOUT")
print("=" * 88)
print("Frames:", frame_values)


def median_curve(df):
    return (
        df.groupby("frame", as_index=False)
          .agg(
              median_pa=("pa", "median"),
              n=("clip", "nunique"),
          )
          .sort_values("frame")
    )


LOW_CURVE = median_curve(
    FG[FG["group"] == "Low dynamics"]
)

HIGH_CURVE = median_curve(
    FG[FG["group"] == "High dynamics"]
)


# ============================================================
# 8. CLIP-WISE SLOPES
# ============================================================

slope_rows = []

for (task, clip, group), d in FG.groupby(
    ["task", "clip", "group"]
):
    d = d.sort_values("frame")

    if len(d) != 10:
        continue

    slope = np.polyfit(
        d["frame"].to_numpy(dtype=float),
        d["pa"].to_numpy(dtype=float),
        1,
    )[0]

    slope_rows.append({
        "task": task,
        "clip": clip,
        "group": group,
        "slope_mm_per_frame": slope,
        "frame_first": int(d["frame"].iloc[0]),
        "frame_last": int(d["frame"].iloc[-1]),
        "pa_first": float(d["pa"].iloc[0]),
        "pa_last": float(d["pa"].iloc[-1]),
        "delta_first_last": float(
            d["pa"].iloc[-1] - d["pa"].iloc[0]
        ),
    })

SLOPES = pd.DataFrame(slope_rows)


# ============================================================
# 9. POOLED FRAME RESULTS
# ============================================================

f0 = frame_values[0]
f1 = frame_values[-1]

low_first = float(
    LOW_CURVE.loc[
        LOW_CURVE["frame"] == f0,
        "median_pa"
    ].iloc[0]
)

low_last = float(
    LOW_CURVE.loc[
        LOW_CURVE["frame"] == f1,
        "median_pa"
    ].iloc[0]
)

high_first = float(
    HIGH_CURVE.loc[
        HIGH_CURVE["frame"] == f0,
        "median_pa"
    ].iloc[0]
)

high_last = float(
    HIGH_CURVE.loc[
        HIGH_CURVE["frame"] == f1,
        "median_pa"
    ].iloc[0]
)

low_growth = low_last - low_first
high_growth = high_last - high_first

low_slope = float(
    SLOPES.loc[
        SLOPES["group"] == "Low dynamics",
        "slope_mm_per_frame"
    ].median()
)

high_slope = float(
    SLOPES.loc[
        SLOPES["group"] == "High dynamics",
        "slope_mm_per_frame"
    ].median()
)

print()
print("POOLED LOW")
print(" n clips       =", G[G["group"] == "Low dynamics"]["clip"].nunique())
print(f" frame {f0}      =", low_first)
print(f" frame {f1}     =", low_last)
print(" first→last Δ  =", low_growth)
print(" median slope  =", low_slope, "mm/frame")

print()
print("POOLED HIGH")
print(" n clips       =", G[G["group"] == "High dynamics"]["clip"].nunique())
print(f" frame {f0}      =", high_first)
print(f" frame {f1}     =", high_last)
print(" first→last Δ  =", high_growth)
print(" median slope  =", high_slope, "mm/frame")


# ============================================================
# 10. TASK-WISE FRAME / SLOPE AUDIT
# ============================================================

task_rollout_rows = []

for task in TASKS:

    for group in ["Low dynamics", "High dynamics"]:

        q = FG[
            (FG["task"] == task)
            & (FG["group"] == group)
        ]

        curve = median_curve(q)

        first = float(
            curve.loc[
                curve["frame"] == f0,
                "median_pa"
            ].iloc[0]
        )

        last = float(
            curve.loc[
                curve["frame"] == f1,
                "median_pa"
            ].iloc[0]
        )

        slopes = SLOPES[
            (SLOPES["task"] == task)
            & (SLOPES["group"] == group)
        ]

        task_rollout_rows.append({
            "task": task,
            "group": group,
            "n_clips": q["clip"].nunique(),
            "frame1_median": first,
            "frame10_median": last,
            "growth": last - first,
            "median_clip_slope": slopes[
                "slope_mm_per_frame"
            ].median(),
        })

TASK_ROLLOUT = pd.DataFrame(task_rollout_rows)

print()
print("=" * 88)
print("AUDIT 4 — TASK-WISE LOW/HIGH ROLLOUT")
print("=" * 88)
print(TASK_ROLLOUT.to_string(index=False))


# ============================================================
# 11. TASK-WISE COMPARISON TABLE
# ============================================================

comparison = []

for task in TASKS:

    lo = TASK_ROLLOUT[
        (TASK_ROLLOUT["task"] == task)
        & (TASK_ROLLOUT["group"] == "Low dynamics")
    ].iloc[0]

    hi = TASK_ROLLOUT[
        (TASK_ROLLOUT["task"] == task)
        & (TASK_ROLLOUT["group"] == "High dynamics")
    ].iloc[0]

    comparison.append({
        "task": task,
        "low_growth": lo["growth"],
        "high_growth": hi["growth"],
        "high_minus_low_growth":
            hi["growth"] - lo["growth"],
        "low_slope": lo["median_clip_slope"],
        "high_slope": hi["median_clip_slope"],
        "high_minus_low_slope":
            hi["median_clip_slope"]
            - lo["median_clip_slope"],
    })

COMP = pd.DataFrame(comparison)

print()
print("=" * 88)
print("AUDIT 5 — DOES HIGH DYNAMICS GROW FASTER?")
print("=" * 88)
print(COMP.to_string(index=False))

n_growth_positive = int(
    (COMP["high_minus_low_growth"] > 0).sum()
)

n_slope_positive = int(
    (COMP["high_minus_low_slope"] > 0).sum()
)

print()
print(
    "Tasks with High > Low endpoint growth:",
    f"{n_growth_positive}/5"
)

print(
    "Tasks with High > Low median slope:",
    f"{n_slope_positive}/5"
)


# ============================================================
# 12. CLUSTER BOOTSTRAP CURVES
#     Bootstrap clips, not individual frame rows.
# ============================================================

def bootstrap_curve(group_name, B=1000):

    q = FG[
        FG["group"] == group_name
    ].copy()

    pivot = q.pivot(
        index=["task", "clip"],
        columns="frame",
        values="pa",
    )

    pivot = pivot[frame_values].dropna()

    M = pivot.to_numpy(dtype=float)

    med = np.median(M, axis=0)

    boots = np.empty(
        (B, len(frame_values)),
        dtype=float,
    )

    for b in range(B):

        idx = RNG.integers(
            0,
            len(M),
            size=len(M),
        )

        boots[b] = np.median(
            M[idx],
            axis=0,
        )

    lo = np.quantile(
        boots,
        0.025,
        axis=0,
    )

    hi = np.quantile(
        boots,
        0.975,
        axis=0,
    )

    return med, lo, hi


low_curve, low_ci_lo, low_ci_hi = bootstrap_curve(
    "Low dynamics"
)

high_curve, high_ci_lo, high_ci_hi = bootstrap_curve(
    "High dynamics"
)


# ============================================================
# 13. SAVE AUDIT TABLES BEFORE FIGURE
# ============================================================

TASK_CLIP.to_csv(
    OUT_ROOT / "task_clip_level_low_high.csv",
    index=False,
)

TASK_ROLLOUT.to_csv(
    OUT_ROOT / "task_rollout_low_high.csv",
    index=False,
)

COMP.to_csv(
    OUT_ROOT / "task_rollout_low_vs_high_comparison.csv",
    index=False,
)

SLOPES.to_csv(
    OUT_ROOT / "per_clip_rollout_slopes.csv",
    index=False,
)


# ============================================================
# 14. WRITE HUMAN-READABLE AUDIT REPORT
# ============================================================

report = f"""
POINTWORLD PROBLEM FIGURE — AUDIT REPORT
========================================

ESTABLISHED CHECK
-----------------
N clips                  = {len(D)}
Spearman RMSAcc vs PA    = {rho:.9f}

Clip-level Low25 median  = {low_med:.9f} mm
Clip-level High25 median = {high_med:.9f} mm
High / Low               = {ratio:.9f} x

NEW LOW/HIGH ROLLOUT ANALYSIS
-----------------------------
Low dynamics:
  frame {f0} median       = {low_first:.9f} mm
  frame {f1} median      = {low_last:.9f} mm
  first-last growth     = {low_growth:.9f} mm
  median clip slope     = {low_slope:.9f} mm/frame

High dynamics:
  frame {f0} median       = {high_first:.9f} mm
  frame {f1} median      = {high_last:.9f} mm
  first-last growth     = {high_growth:.9f} mm
  median clip slope     = {high_slope:.9f} mm/frame

TASK CONSISTENCY
----------------
High > Low endpoint growth = {n_growth_positive}/5 tasks
High > Low median slope     = {n_slope_positive}/5 tasks
"""

(OUT_ROOT / "AUDIT_REPORT.txt").write_text(report)

print(report)


# ============================================================
# 15. FINAL FIGURE
#
# IMPORTANT:
# Panel (a): established clip-level Low25/High25 analysis.
# Panel (b): NEW frame-wise Low/High analysis.
# Numbers are NEVER mixed.
# ============================================================

C_LOW = "#3B78B8"
C_HIGH = "#D95F3A"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.2, 2.8),
    constrained_layout=True,
)


# ------------------------------------------------------------
# PANEL A
# ------------------------------------------------------------

ax = axes[0]

x = np.arange(len(TASK_CLIP))
w = 0.34

ax.bar(
    x - w / 2,
    TASK_CLIP["low_clip_pa"],
    width=w,
    color=C_LOW,
    label="Low dynamics",
)

ax.bar(
    x + w / 2,
    TASK_CLIP["high_clip_pa"],
    width=w,
    color=C_HIGH,
    label="High dynamics",
)

ax.set_xticks(x)

ax.set_xticklabels([
    t.replace("task-", "")
    for t in TASK_CLIP["task"]
])

ax.set_xlabel("BEHAVIOR task")
ax.set_ylabel("PA-MPJPE (mm) ↓")

ax.set_title(
    "(a) High-dynamics actions exhibit larger rigid-shape error",
    loc="left",
    fontweight="bold",
)

ax.legend(
    frameon=False,
    loc="upper right",
)

ax.grid(
    axis="y",
    alpha=0.15,
    linewidth=0.6,
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ONLY clip-level numbers belong here
ax.text(
    0.03,
    0.96,
    f"Pooled median: {low_med:.2f} vs. {high_med:.2f} mm "
    f"({ratio:.2f}×)",
    transform=ax.transAxes,
    va="top",
    fontsize=7.7,
)


# ------------------------------------------------------------
# PANEL B
# ------------------------------------------------------------

ax = axes[1]

fr = np.asarray(frame_values)

ax.fill_between(
    fr,
    low_ci_lo,
    low_ci_hi,
    color=C_LOW,
    alpha=0.13,
    linewidth=0,
)

ax.fill_between(
    fr,
    high_ci_lo,
    high_ci_hi,
    color=C_HIGH,
    alpha=0.13,
    linewidth=0,
)

ax.plot(
    fr,
    low_curve,
    color=C_LOW,
    linewidth=2.1,
    marker="o",
    markersize=3.6,
    label="Low dynamics",
)

ax.plot(
    fr,
    high_curve,
    color=C_HIGH,
    linewidth=2.1,
    marker="o",
    markersize=3.6,
    label="High dynamics",
)

ax.set_xlabel("Rollout frame")
ax.set_ylabel("PA-MPJPE (mm) ↓")

ax.set_title(
    "(b) Rollout error by action-dynamics group",
    loc="left",
    fontweight="bold",
)

ax.legend(
    frameon=False,
    loc="upper left",
)

ax.set_xticks(fr)

ax.grid(
    axis="y",
    alpha=0.15,
    linewidth=0.6,
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ONLY actual frame-wise curve numbers belong here
ax.text(
    0.98,
    0.05,
    (
        f"Low: {low_first:.2f} → {low_last:.2f} mm\n"
        f"High: {high_first:.2f} → {high_last:.2f} mm"
    ),
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
)


# ============================================================
# SAVE
# ============================================================

PDF = OUT_ROOT / "figure_problem_final_audited.pdf"
PNG = OUT_ROOT / "figure_problem_final_audited.png"
SVG = OUT_ROOT / "figure_problem_final_audited.svg"

fig.savefig(
    PDF,
    bbox_inches="tight",
)

fig.savefig(
    PNG,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    SVG,
    bbox_inches="tight",
)

print()
print("=" * 88)
print("AUDIT PASSED")
print("=" * 88)
print("PDF :", PDF)
print("PNG :", PNG)
print("SVG :", SVG)
print("REPORT:", OUT_ROOT / "AUDIT_REPORT.txt")
print("=" * 88)
