from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
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

OUTDIR = Path(
    "diagnostics/results/final_problem_figure_audited"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

EXPECTED_N = 7947
EXPECTED_CF = 79470
EXPECTED_RHO = 0.420267
EXPECTED_LOW = 1.02432
EXPECTED_HIGH = 2.38449

C_LOW = "#3B78B8"
C_HIGH = "#D95F3A"

RNG = np.random.default_rng(42)


def detect(df, names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


# ============================================================
# 1. LOAD RMS ACC
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
        f"Cannot identify ACC columns: {list(acc.columns)}"
    )

A = pd.DataFrame({
    "clip": acc[clip_c].astype(str),
    "rms_acc": pd.to_numeric(acc[acc_c], errors="coerce"),
})

if task_c is not None:
    A["task"] = acc[task_c].astype(str)
else:
    A["task"] = A["clip"].str.extract(
        r"(task-\d{4})"
    )[0]

A = (
    A.dropna()
     .drop_duplicates(["task", "clip"])
)


# ============================================================
# 2. LOAD RAW OBJECT-LEVEL PA ROWS
# ============================================================

parts = []

for task in TASKS:
    f = PA_ROOT / task / "per_frame_metrics.csv"
    d = pd.read_csv(f)

    required = [
        "clip",
        "frame",
        "pa_mpjpe_mm",
    ]

    for c in required:
        if c not in d.columns:
            raise RuntimeError(
                f"{f}: missing {c}"
            )

    z = pd.DataFrame({
        "task": task,
        "clip": d["clip"].astype(str),
        "frame": pd.to_numeric(
            d["frame"],
            errors="coerce",
        ),
        "pa": pd.to_numeric(
            d["pa_mpjpe_mm"],
            errors="coerce",
        ),
    }).dropna()

    parts.append(z)

RAW = pd.concat(
    parts,
    ignore_index=True,
)

RAW["frame"] = RAW["frame"].astype(int)


# ============================================================
# 3. IMPORTANT:
# OBJECT ROWS -> ONE PA VALUE PER CLIP-FRAME
# ============================================================

CF = (
    RAW.groupby(
        ["task", "clip", "frame"],
        as_index=False,
    )
    .agg(
        pa=("pa", "mean"),
        n_objects=("pa", "size"),
    )
)

if len(CF) != EXPECTED_CF:
    raise RuntimeError(
        f"clip-frame count mismatch: "
        f"{len(CF)} != {EXPECTED_CF}"
    )

frame_count = (
    CF.groupby(["task", "clip"])["frame"]
      .nunique()
)

if not frame_count.eq(10).all():
    raise RuntimeError(
        "Not every clip has exactly 10 rollout frames."
    )


# ============================================================
# 4. CLIP-LEVEL PA
# ============================================================

P = (
    CF.groupby(
        ["task", "clip"],
        as_index=False,
    )
    .agg(
        pa_mean=("pa", "mean"),
    )
)

D = P.merge(
    A,
    on=["task", "clip"],
    how="inner",
)

if len(D) != EXPECTED_N:
    raise RuntimeError(
        f"N mismatch: {len(D)} != {EXPECTED_N}"
    )

rho = D["rms_acc"].corr(
    D["pa_mean"],
    method="spearman",
)

if abs(rho - EXPECTED_RHO) > 0.005:
    raise RuntimeError(
        f"rho mismatch: {rho}"
    )


# ============================================================
# 5. LOW/HIGH = WITHIN-TASK RMS ACC QUARTILES
# ============================================================

groups = []
task_bar_rows = []

for task in TASKS:

    d = D[D["task"] == task].copy()

    q25 = d["rms_acc"].quantile(0.25)
    q75 = d["rms_acc"].quantile(0.75)

    low = d[d["rms_acc"] <= q25].copy()
    high = d[d["rms_acc"] >= q75].copy()

    task_bar_rows.append({
        "task": task.replace("task-", ""),
        "low_pa": low["pa_mean"].median(),
        "high_pa": high["pa_mean"].median(),
        "n_low": len(low),
        "n_high": len(high),
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
BAR = pd.DataFrame(task_bar_rows)


# ============================================================
# 6. REPRODUCE ESTABLISHED LOW/HIGH NUMBERS
# ============================================================

DG = D.merge(
    G,
    on=["task", "clip"],
    how="inner",
)

low_med = float(
    DG.loc[
        DG["group"] == "Low dynamics",
        "pa_mean"
    ].median()
)

high_med = float(
    DG.loc[
        DG["group"] == "High dynamics",
        "pa_mean"
    ].median()
)

ratio = high_med / low_med

if abs(low_med - EXPECTED_LOW) > 0.005:
    raise RuntimeError(
        f"Low median mismatch: {low_med}"
    )

if abs(high_med - EXPECTED_HIGH) > 0.005:
    raise RuntimeError(
        f"High median mismatch: {high_med}"
    )


# ============================================================
# 7. LOW/HIGH FRAME-WISE ROLLOUT
# ============================================================

FG = CF.merge(
    G,
    on=["task", "clip"],
    how="inner",
)

frames = sorted(
    FG["frame"].unique()
)

if len(frames) != 10:
    raise RuntimeError(
        f"Expected 10 frames, got {frames}"
    )


def pooled_curve(group):
    q = FG[
        FG["group"] == group
    ]

    return (
        q.groupby("frame", as_index=False)
         .agg(
             median_pa=("pa", "median"),
             n_clips=("clip", "nunique"),
         )
         .sort_values("frame")
    )


LOW = pooled_curve("Low dynamics")
HIGH = pooled_curve("High dynamics")


# ============================================================
# 8. BOOTSTRAP 95% CI OVER CLIPS
# ============================================================

def bootstrap_curve(group, B=1000):

    q = FG[
        FG["group"] == group
    ]

    pivot = (
        q.pivot(
            index=["task", "clip"],
            columns="frame",
            values="pa",
        )
        .reindex(columns=frames)
        .dropna()
    )

    M = pivot.to_numpy(dtype=float)

    observed = np.median(
        M,
        axis=0,
    )

    boots = np.empty(
        (B, len(frames))
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

    return observed, lo, hi


low_curve, low_lo, low_hi = bootstrap_curve(
    "Low dynamics"
)

high_curve, high_lo, high_hi = bootstrap_curve(
    "High dynamics"
)


# ============================================================
# 9. ACTUAL FRAME-1 / FRAME-10 VALUES
# ============================================================

f_first = frames[0]
f_last = frames[-1]

low_first = float(
    LOW.loc[
        LOW["frame"] == f_first,
        "median_pa"
    ].iloc[0]
)

low_last = float(
    LOW.loc[
        LOW["frame"] == f_last,
        "median_pa"
    ].iloc[0]
)

high_first = float(
    HIGH.loc[
        HIGH["frame"] == f_first,
        "median_pa"
    ].iloc[0]
)

high_last = float(
    HIGH.loc[
        HIGH["frame"] == f_last,
        "median_pa"
    ].iloc[0]
)

low_growth = low_last - low_first
high_growth = high_last - high_first


# ============================================================
# 10. CLIP-WISE SLOPE AUDIT
# ============================================================

slope_rows = []

for (task, clip, group), d in FG.groupby(
    ["task", "clip", "group"]
):

    d = d.sort_values("frame")

    x = d["frame"].to_numpy(dtype=float)
    y = d["pa"].to_numpy(dtype=float)

    slope = np.polyfit(
        x,
        y,
        1,
    )[0]

    slope_rows.append({
        "task": task,
        "clip": clip,
        "group": group,
        "slope": slope,
    })

S = pd.DataFrame(slope_rows)

low_slope = float(
    S.loc[
        S["group"] == "Low dynamics",
        "slope"
    ].median()
)

high_slope = float(
    S.loc[
        S["group"] == "High dynamics",
        "slope"
    ].median()
)


# ============================================================
# 11. TASK-WISE ROLLOUT AUDIT
# ============================================================

task_rows = []

for task in TASKS:

    for group in [
        "Low dynamics",
        "High dynamics",
    ]:

        q = FG[
            (FG["task"] == task)
            & (FG["group"] == group)
        ]

        curve = (
            q.groupby("frame", as_index=False)
             .agg(pa=("pa", "median"))
             .sort_values("frame")
        )

        first = float(
            curve.loc[
                curve["frame"] == f_first,
                "pa"
            ].iloc[0]
        )

        last = float(
            curve.loc[
                curve["frame"] == f_last,
                "pa"
            ].iloc[0]
        )

        slope_med = float(
            S.loc[
                (S["task"] == task)
                & (S["group"] == group),
                "slope"
            ].median()
        )

        task_rows.append({
            "task": task,
            "group": group,
            "frame_first": first,
            "frame_last": last,
            "growth": last - first,
            "median_slope": slope_med,
        })

TASK_ROLLOUT = pd.DataFrame(task_rows)

print()
print("=" * 90)
print("FINAL AUDIT")
print("=" * 90)

print("RAW object rows        =", len(RAW))
print("Unique clip-frame rows =", len(CF))
print("N clips                =", len(D))
print("Spearman rho           =", rho)

print()
print(
    f"Clip-level Low/High     = "
    f"{low_med:.6f} -> {high_med:.6f} mm"
)
print(
    f"High / Low              = "
    f"{ratio:.6f}x"
)

print()
print(
    f"Rollout Low  frame {f_first}->{f_last} = "
    f"{low_first:.6f} -> {low_last:.6f} mm "
    f"(delta={low_growth:.6f})"
)

print(
    f"Rollout High frame {f_first}->{f_last} = "
    f"{high_first:.6f} -> {high_last:.6f} mm "
    f"(delta={high_growth:.6f})"
)

print()
print(
    f"Median clip slope Low   = "
    f"{low_slope:.6f} mm/frame"
)

print(
    f"Median clip slope High  = "
    f"{high_slope:.6f} mm/frame"
)

print()
print("TASK-WISE:")
print(
    TASK_ROLLOUT.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE NUMERIC AUDIT
# ============================================================

BAR.to_csv(
    OUTDIR / "task_low_high_clip_level.csv",
    index=False,
)

LOW.to_csv(
    OUTDIR / "low_rollout_curve.csv",
    index=False,
)

HIGH.to_csv(
    OUTDIR / "high_rollout_curve.csv",
    index=False,
)

TASK_ROLLOUT.to_csv(
    OUTDIR / "taskwise_rollout_audit.csv",
    index=False,
)

S.to_csv(
    OUTDIR / "per_clip_rollout_slopes.csv",
    index=False,
)


# ============================================================
# 13. FINAL PAPER FIGURE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.15, 2.75),
)

# ============================================================
# PANEL A
# ============================================================

ax = axes[0]

x = np.arange(
    len(BAR)
)

w = 0.34

ax.bar(
    x - w / 2,
    BAR["low_pa"],
    width=w,
    color=C_LOW,
    label="Low dynamics",
)

ax.bar(
    x + w / 2,
    BAR["high_pa"],
    width=w,
    color=C_HIGH,
    label="High dynamics",
)

ax.set_xticks(x)
ax.set_xticklabels(
    BAR["task"]
)

ax.set_xlabel(
    "BEHAVIOR task"
)

ax.set_ylabel(
    "PA-MPJPE (mm) ↓"
)

ax.set_title(
    "(a) High-dynamics actions exhibit larger rigid-shape error",
    loc="left",
    fontweight="bold",
)

ax.text(
    0.03,
    0.96,
    (
        f"Pooled median: "
        f"{low_med:.2f} vs. {high_med:.2f} mm "
        f"({ratio:.2f}×)"
    ),
    transform=ax.transAxes,
    va="top",
    fontsize=7.5,
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


# ============================================================
# PANEL B
# ============================================================

ax = axes[1]

fr = np.asarray(
    frames
)

ax.fill_between(
    fr,
    low_lo,
    low_hi,
    color=C_LOW,
    alpha=0.14,
    linewidth=0,
)

ax.fill_between(
    fr,
    high_lo,
    high_hi,
    color=C_HIGH,
    alpha=0.14,
    linewidth=0,
)

ax.plot(
    fr,
    low_curve,
    color=C_LOW,
    linewidth=2.2,
    marker="o",
    markersize=3.8,
    label="Low dynamics",
)

ax.plot(
    fr,
    high_curve,
    color=C_HIGH,
    linewidth=2.2,
    marker="o",
    markersize=3.8,
    label="High dynamics",
)

ax.set_xticks(fr)

ax.set_xlabel(
    "Rollout frame"
)

ax.set_ylabel(
    "PA-MPJPE (mm) ↓"
)

ax.set_title(
    "(b) Rollout PA-MPJPE by action-dynamics group",
    loc="left",
    fontweight="bold",
)

ax.legend(
    frameon=False,
    loc="upper left",
)

# IMPORTANT:
# these are actual panel-(b) frame-wise values,
# NOT the 1.02 -> 2.38 clip-level numbers.
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
    fontsize=7.8,
)

ax.grid(
    axis="y",
    alpha=0.15,
    linewidth=0.6,
)

ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout(
    w_pad=2.0
)

PNG = OUTDIR / "figure1_problem_final.png"
PDF = OUTDIR / "figure1_problem_final.pdf"
SVG = OUTDIR / "figure1_problem_final.svg"

fig.savefig(
    PNG,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    PDF,
    bbox_inches="tight",
)

fig.savefig(
    SVG,
    bbox_inches="tight",
)

print()
print("=" * 90)
print("SAVED")
print("=" * 90)
print(PNG)
print(PDF)
print(SVG)
print("=" * 90)
