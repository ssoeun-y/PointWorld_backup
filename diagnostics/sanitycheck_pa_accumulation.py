#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

FRAME = (
    ROOT
    / "diagnostics/results/"
      "framewise_acc_pa_5task_current10457/"
      "combined_one_row_per_clip_frame.csv"
)

ACC = (
    ROOT
    / "diagnostics/results/"
      "current_behavior_established_acc_10457/"
      "per_sample_current_established_acc_10457.csv"
)

OUT = (
    ROOT
    / "diagnostics/results/"
      "pa_accumulation_mechanism_current10457/"
      "SANITY_CHECK.txt"
)

OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FRAME).sort_values(
    ["task", "clip", "frame"]
)

acc = pd.read_csv(ACC)

# ------------------------------------------------------------
# Identify clip / RMS Acc columns
# ------------------------------------------------------------
clip_col = next(
    c for c in ["clip", "key", "__key__"]
    if c in acc.columns
)

acc_candidates = [
    c for c in acc.columns
    if "rms" in c.lower()
    and "acc" in c.lower()
]

if not acc_candidates:
    raise RuntimeError(
        f"No RMS Acc column: {list(acc.columns)}"
    )

acc_col = (
    next(
        (
            c for c in acc_candidates
            if "si" in c.lower()
        ),
        acc_candidates[0],
    )
)

acc_map = (
    acc[[clip_col, acc_col]]
    .drop_duplicates(clip_col)
    .rename(
        columns={
            clip_col: "clip",
            acc_col: "rms_acc",
        }
    )
)


# ============================================================
# 1. Exact horizon behavior per clip
# ============================================================

rows = []

for (task, clip), g in df.groupby(
    ["task", "clip"],
    sort=False,
):
    g = g.sort_values("frame")

    x = g["frame"].to_numpy(float)
    y = g["pa_mpjpe_mm"].to_numpy(float)

    rho, _ = spearmanr(x, y)

    dy = np.diff(y)

    slope = np.polyfit(
        x,
        y,
        1,
    )[0]

    rows.append({
        "task": task,
        "clip": clip,
        "rho": rho,
        "slope": slope,
        "all_consecutive_increase":
            int(np.all(dy > 0)),
        "fraction_positive_steps":
            float(np.mean(dy > 0)),
        "early_pa":
            float(np.median(y[:3])),
        "late_pa":
            float(np.median(y[-3:])),
    })

c = pd.DataFrame(rows)

c = c.merge(
    acc_map,
    on="clip",
    how="left",
)


# ============================================================
# 2. Frame-index population curve
# ============================================================

frame_summary = (
    df.groupby("frame")["pa_mpjpe_mm"]
    .agg(
        median="median",
        mean="mean",
        count="size",
    )
    .reset_index()
)


# ============================================================
# 3. Shuffle sanity:
#    Shuffle PA order WITHIN each clip.
#
#    If our implementation itself forces positive horizon rho,
#    shuffled data would also remain positive.
# ============================================================

rng = np.random.default_rng(42)

shuffle_medians = []

for rep in range(200):

    rhos = []

    for _, g in df.groupby(
        ["task", "clip"],
        sort=False,
    ):
        x = g["frame"].to_numpy(float)

        y = g[
            "pa_mpjpe_mm"
        ].to_numpy(float).copy()

        rng.shuffle(y)

        rho, _ = spearmanr(x, y)

        if np.isfinite(rho):
            rhos.append(rho)

    shuffle_medians.append(
        np.median(rhos)
    )

shuffle_medians = np.asarray(
    shuffle_medians
)


# ============================================================
# 4. High RMS Acc -> steeper PA growth,
#    controlling simple task + baseline PA confounds.
#
# Rank residualization:
#   rank(Acc) and rank(slope)
# remove:
#   task fixed effects + early PA rank
# ============================================================

m = c.dropna(
    subset=[
        "rms_acc",
        "slope",
        "early_pa",
    ]
).copy()

m["acc_rank"] = (
    m["rms_acc"]
    .rank(pct=True)
)

m["slope_rank"] = (
    m["slope"]
    .rank(pct=True)
)

m["early_rank"] = (
    m["early_pa"]
    .rank(pct=True)
)

task_dummies = pd.get_dummies(
    m["task"],
    drop_first=True,
    dtype=float,
)

X = np.column_stack([
    np.ones(len(m)),
    m["early_rank"].to_numpy(float),
    task_dummies.to_numpy(float),
])

ba = np.linalg.lstsq(
    X,
    m["acc_rank"].to_numpy(float),
    rcond=None,
)[0]

bs = np.linalg.lstsq(
    X,
    m["slope_rank"].to_numpy(float),
    rcond=None,
)[0]

acc_resid = (
    m["acc_rank"].to_numpy(float)
    - X @ ba
)

slope_resid = (
    m["slope_rank"].to_numpy(float)
    - X @ bs
)

partial_rho, partial_p = spearmanr(
    acc_resid,
    slope_resid,
)


# ============================================================
# 5. Within-task Acc vs slope
# ============================================================

task_results = []

for task, g in c.dropna(
    subset=["rms_acc", "slope"]
).groupby("task"):

    rho, p = spearmanr(
        g["rms_acc"],
        g["slope"],
    )

    task_results.append(
        (task, len(g), rho, p)
    )


# ============================================================
# 6. Print representative clips
# ============================================================

sample_keys = (
    c.sample(
        n=min(5, len(c)),
        random_state=42,
    )["clip"]
    .tolist()
)


lines = []

def P(s=""):
    print(s)
    lines.append(str(s))


P("=" * 88)
P("PA ACCUMULATION SANITY CHECK")
P("=" * 88)

P(f"N clips = {len(c)}")
P()

P("1. HORIZON MONOTONICITY")
P(
    "median horizon↔PA rho = "
    f"{c['rho'].median():+.6f}"
)
P(
    "rho exactly +1.0 = "
    f"{np.isclose(c['rho'], 1.0).mean()*100:.1f}%"
)
P(
    "all consecutive PA steps increase = "
    f"{c['all_consecutive_increase'].mean()*100:.1f}%"
)
P(
    "median fraction of increasing steps = "
    f"{c['fraction_positive_steps'].median()*100:.1f}%"
)

P()
P("rho quantiles:")
P(
    c["rho"]
    .quantile(
        [0, .05, .25, .5, .75, .95, 1]
    )
    .to_string()
)

P()
P("2. POPULATION PA BY FUTURE FRAME")
P(frame_summary.to_string(index=False))

P()
P("3. SHUFFLE NULL")
P(
    "observed median rho = "
    f"{c['rho'].median():+.6f}"
)
P(
    "shuffle median-rho mean = "
    f"{shuffle_medians.mean():+.6f}"
)
P(
    "shuffle median-rho 95% range = "
    f"[{np.quantile(shuffle_medians,.025):+.6f}, "
    f"{np.quantile(shuffle_medians,.975):+.6f}]"
)

P()
P("4. RMS Acc -> PA GROWTH AFTER SIMPLE CONTROLS")
P(
    "partial rank correlation "
    "(control task + early PA) = "
    f"{partial_rho:+.6f}, "
    f"p={partial_p:.3e}"
)

P()
P("5. TASKWISE RMS Acc -> PA SLOPE")

for task, n, rho, p in task_results:
    P(
        f"{task}: N={n}, "
        f"rho={rho:+.6f}, "
        f"p={p:.3e}"
    )

P()
P("6. EXAMPLE PA TRAJECTORIES")

for key in sample_keys:

    g = df[
        df["clip"] == key
    ].sort_values("frame")

    vals = " ".join(
        f"{x:.3f}"
        for x in g[
            "pa_mpjpe_mm"
        ].to_numpy()
    )

    P(f"{key}")
    P(f"  PA(mm): {vals}")

P()
P("=" * 88)

OUT.write_text(
    "\n".join(lines) + "\n"
)

P(f"SAVED = {OUT}")
