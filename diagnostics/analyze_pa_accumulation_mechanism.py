#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]

FRAME_CSV = (
    ROOT
    / "diagnostics/results/"
      "framewise_acc_pa_5task_current10457/"
      "combined_one_row_per_clip_frame.csv"
)

ACC_CSV = (
    ROOT
    / "diagnostics/results/"
      "current_behavior_established_acc_10457/"
      "per_sample_current_established_acc_10457.csv"
)

OUT = (
    ROOT
    / "diagnostics/results/"
      "pa_accumulation_mechanism_current10457"
)

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load frame-wise PA + local Acc
# ============================================================

df = pd.read_csv(FRAME_CSV)

required = {
    "task",
    "clip",
    "frame",
    "local_acc_mps2",
    "pa_mpjpe_mm",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing frame columns: {missing}"
    )

df = df.sort_values(
    ["task", "clip", "frame"]
).copy()


# ============================================================
# Build history features
#
# IMPORTANT:
# past2/past3/past5 exclude the CURRENT frame.
# Therefore:
#   PA(t) vs Acc history before t
# ============================================================

parts = []

for (task, clip), g in df.groupby(
    ["task", "clip"],
    sort=False,
):

    g = g.copy().sort_values("frame")

    past = g[
        "local_acc_mps2"
    ].shift(1)

    g["past2_acc"] = (
        past
        .rolling(
            window=2,
            min_periods=1,
        )
        .mean()
    )

    g["past3_acc"] = (
        past
        .rolling(
            window=3,
            min_periods=1,
        )
        .mean()
    )

    g["past5_acc"] = (
        past
        .rolling(
            window=5,
            min_periods=1,
        )
        .mean()
    )

    # generic prediction-horizon coordinate
    n = len(g)

    g["horizon_norm"] = np.linspace(
        0.0,
        1.0,
        n,
    )

    parts.append(g)

df = pd.concat(
    parts,
    ignore_index=True,
)


# ============================================================
# A. Does PA accumulate with prediction horizon?
# ============================================================

clip_rows = []

for (task, clip), g in df.groupby(
    ["task", "clip"],
    sort=False,
):

    g = g.sort_values("frame")

    if len(g) < 5:
        continue

    x = g["frame"].to_numpy(
        dtype=np.float64
    )

    y = g["pa_mpjpe_mm"].to_numpy(
        dtype=np.float64
    )

    # linear PA growth per prediction frame
    slope = np.polyfit(
        x,
        y,
        deg=1,
    )[0]

    rho_t, _ = spearmanr(
        x,
        y,
    )

    # interpretable early vs late comparison
    k = min(3, len(g) // 2)

    early = float(
        g.iloc[:k][
            "pa_mpjpe_mm"
        ].median()
    )

    late = float(
        g.iloc[-k:][
            "pa_mpjpe_mm"
        ].median()
    )

    clip_rows.append({
        "task": task,
        "clip": clip,
        "n_frames": len(g),
        "pa_slope_mm_per_frame":
            float(slope),
        "rho_horizon_pa":
            float(rho_t),
        "early_pa_mm":
            early,
        "late_pa_mm":
            late,
        "late_minus_early_mm":
            late - early,
        "late_over_early":
            late / early
            if early > 0
            else np.nan,
    })


clip = pd.DataFrame(
    clip_rows
)


# ============================================================
# B. Join established clip-level RMS Acc
# ============================================================

acc = pd.read_csv(
    ACC_CSV
)

clip_col = None

for c in [
    "clip",
    "key",
    "__key__",
]:
    if c in acc.columns:
        clip_col = c
        break

if clip_col is None:
    raise RuntimeError(
        f"Cannot find clip column. "
        f"Columns={list(acc.columns)}"
    )


# Find established RMS Acc column robustly.
numeric_cols = [
    c for c in acc.columns
    if pd.api.types.is_numeric_dtype(
        acc[c]
    )
]

preferred = [
    "rms_acceleration_si",
    "rms_acc_si",
    "rms_acc_mps2",
    "rms_acceleration",
    "rms_acc",
]

acc_col = None

for c in preferred:
    if c in acc.columns:
        acc_col = c
        break

if acc_col is None:

    candidates = [
        c for c in numeric_cols
        if (
            "acc" in c.lower()
            and "rms" in c.lower()
        )
    ]

    if len(candidates) == 1:
        acc_col = candidates[0]

    elif len(candidates) > 1:

        # Prefer SI-labelled column if present.
        si = [
            c for c in candidates
            if "si" in c.lower()
        ]

        acc_col = (
            si[0]
            if si
            else candidates[0]
        )

if acc_col is None:
    raise RuntimeError(
        "Cannot automatically identify "
        f"RMS Acc column. Columns={list(acc.columns)}"
    )


acc_small = (
    acc[
        [
            clip_col,
            acc_col,
        ]
    ]
    .drop_duplicates(
        subset=[clip_col]
    )
    .rename(
        columns={
            clip_col: "clip",
            acc_col:
                "established_rms_acc",
        }
    )
)

clip = clip.merge(
    acc_small,
    on="clip",
    how="left",
)

matched = clip[
    np.isfinite(
        clip[
            "established_rms_acc"
        ]
    )
].copy()


# ============================================================
# C. Does higher trajectory RMS Acc mean
#    stronger PA accumulation?
# ============================================================

rho_acc_slope, p_acc_slope = (
    spearmanr(
        matched[
            "established_rms_acc"
        ],
        matched[
            "pa_slope_mm_per_frame"
        ],
    )
)

rho_acc_lategain, p_acc_lategain = (
    spearmanr(
        matched[
            "established_rms_acc"
        ],
        matched[
            "late_minus_early_mm"
        ],
    )
)


# ============================================================
# D. Does PREVIOUS Acc explain current PA?
#
# Compare:
#   current Acc
#   past 2 frames
#   past 3 frames
#   past 5 frames
#
# Main summary = within-clip Spearman.
# ============================================================

features = [
    (
        "current",
        "local_acc_mps2",
    ),
    (
        "past2",
        "past2_acc",
    ),
    (
        "past3",
        "past3_acc",
    ),
    (
        "past5",
        "past5_acc",
    ),
]


hist_rows = []

for (
    task,
    clip_id,
), g in df.groupby(
    ["task", "clip"],
    sort=False,
):

    for name, col in features:

        h = g[
            [
                col,
                "pa_mpjpe_mm",
            ]
        ].dropna()

        if len(h) < 5:
            continue

        if (
            h[col].nunique() < 3
            or
            h[
                "pa_mpjpe_mm"
            ].nunique() < 3
        ):
            continue

        rho, _ = spearmanr(
            h[col],
            h[
                "pa_mpjpe_mm"
            ],
        )

        if not np.isfinite(rho):
            continue

        hist_rows.append({
            "task": task,
            "clip": clip_id,
            "feature": name,
            "rho": float(rho),
        })


hist = pd.DataFrame(
    hist_rows
)


# ============================================================
# E. Time-controlled pooled rank relation
#
# Accumulated history can mechanically correlate with time.
# So remove generic prediction-horizon trend first.
# ============================================================

def pooled_time_controlled_rank_corr(
    data,
    feature,
):

    xx = []
    yy = []

    for clip_id, g in data.groupby(
        "clip",
        sort=False,
    ):

        h = g[
            [
                feature,
                "pa_mpjpe_mm",
                "frame",
            ]
        ].dropna().copy()

        if len(h) < 5:
            continue

        if (
            h[feature].nunique() < 3
            or
            h[
                "pa_mpjpe_mm"
            ].nunique() < 3
        ):
            continue

        # Spearman-style ranks within each clip
        xr = h[feature].rank(
            pct=True
        ).to_numpy()

        yr = h[
            "pa_mpjpe_mm"
        ].rank(
            pct=True
        ).to_numpy()

        tr = h["frame"].rank(
            pct=True
        ).to_numpy()

        # remove linear horizon effect from both ranks
        X = np.column_stack([
            np.ones(len(tr)),
            tr,
        ])

        bx = np.linalg.lstsq(
            X,
            xr,
            rcond=None,
        )[0]

        by = np.linalg.lstsq(
            X,
            yr,
            rcond=None,
        )[0]

        rx = xr - X @ bx
        ry = yr - X @ by

        xx.extend(rx.tolist())
        yy.extend(ry.tolist())

    if len(xx) < 10:
        return np.nan, np.nan, 0

    rho, p = spearmanr(
        xx,
        yy,
    )

    return (
        float(rho),
        float(p),
        len(xx),
    )


time_controlled = {}

for name, col in features:

    time_controlled[name] = (
        pooled_time_controlled_rank_corr(
            df,
            col,
        )
    )


# ============================================================
# Save
# ============================================================

df.to_csv(
    OUT / "frame_features.csv",
    index=False,
)

clip.to_csv(
    OUT / "per_clip_accumulation.csv",
    index=False,
)

hist.to_csv(
    OUT / "per_clip_history_correlations.csv",
    index=False,
)


# ============================================================
# Report
# ============================================================

print("=" * 90)
print("PA-MPJPE ACCUMULATION MECHANISM")
print("=" * 90)

print(
    f"Frame rows = {len(df)}"
)

print(
    f"Clips = {len(clip)}"
)

print(
    f"Matched with established RMS Acc = "
    f"{len(matched)}"
)

print(
    f"Established RMS Acc column used = "
    f"{acc_col}"
)

print()


# ------------------------------------------------------------
# A. Rollout accumulation
# ------------------------------------------------------------

print(
    "A. DOES PA-MPJPE GROW WITH PREDICTION HORIZON?"
)

print(
    "median PA slope = "
    f"{clip['pa_slope_mm_per_frame'].median():+.6f} "
    "mm/frame"
)

print(
    "clips with positive PA slope = "
    f"{(clip['pa_slope_mm_per_frame'] > 0).mean()*100:.1f}%"
)

print(
    "median within-clip horizon↔PA rho = "
    f"{clip['rho_horizon_pa'].median():+.6f}"
)

print(
    "early PA median = "
    f"{clip['early_pa_mm'].median():.6f} mm"
)

print(
    "late PA median = "
    f"{clip['late_pa_mm'].median():.6f} mm"
)

print(
    "late/early = "
    f"{clip['late_pa_mm'].median()/clip['early_pa_mm'].median():.3f}x"
)

try:
    w = wilcoxon(
        clip["late_pa_mm"],
        clip["early_pa_mm"],
        alternative="greater",
    )

    print(
        "late > early Wilcoxon p = "
        f"{w.pvalue:.3e}"
    )

except Exception:
    pass

print()


# ------------------------------------------------------------
# B. High RMS Acc -> steeper accumulation?
# ------------------------------------------------------------

print(
    "B. DOES HIGH TRAJECTORY RMS Acc HAVE STRONGER PA ACCUMULATION?"
)

print(
    "RMS Acc vs PA slope: "
    f"rho={rho_acc_slope:+.6f}, "
    f"p={p_acc_slope:.3e}"
)

print(
    "RMS Acc vs late-early PA increase: "
    f"rho={rho_acc_lategain:+.6f}, "
    f"p={p_acc_lategain:.3e}"
)

print()


# ------------------------------------------------------------
# C. Current vs previous Acc history
# ------------------------------------------------------------

print(
    "C. WHICH ACC SIGNAL TRACKS CURRENT PA-MPJPE?"
)

for name, _ in features:

    h = hist[
        hist["feature"] == name
    ]

    med = h["rho"].median()

    pos = (
        h["rho"] > 0
    ).mean() * 100.0

    tc_rho, tc_p, tc_n = (
        time_controlled[name]
    )

    print(
        f"{name:8s} | "
        f"median within-clip rho={med:+.6f} | "
        f"positive={pos:.1f}% | "
        f"time-controlled pooled rho="
        f"{tc_rho:+.6f}, "
        f"p={tc_p:.3e}, "
        f"N={tc_n}"
    )


print()
print("=" * 90)
print("HOW TO INTERPRET")
print("=" * 90)

print(
    "1) A positive horizon trend means predicted rigid shape "
    "degrades as rollout proceeds."
)

print(
    "2) Positive RMS-Acc vs PA-slope correlation means "
    "high-Acc trajectories accumulate shape error faster."
)

print(
    "3) If past2/past3/past5 are stronger than current Acc, "
    "the data support a temporal-history / accumulation mechanism."
)

print(
    "4) If these are absent, do NOT claim accumulation as the cause."
)

print()
print(
    "SAVED =",
    OUT,
)
