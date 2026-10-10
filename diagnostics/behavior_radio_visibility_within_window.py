#!/usr/bin/env python3

from pathlib import Path
import math

import numpy as np
import pandas as pd


INFILE = Path(
    "diagnostics/results/"
    "behavior_radio_visibility_observational/"
    "radio554_visibility_per_point.csv"
)

OUTDIR = Path(
    "diagnostics/results/"
    "behavior_radio_visibility_within_window"
)

df = pd.read_csv(INFILE)

assert df["sample_key"].nunique() == 554
assert set(df["visible_views"].unique()).issubset({0, 1, 2})

rows = []

for key, g in df.groupby("sample_key", sort=True):

    g0 = g[g["visible_views"] == 0]
    gv = g[g["visible_views"] >= 1]

    # Paired comparison is only defined when BOTH groups
    # exist in the SAME window.
    if len(g0) == 0 or len(gv) == 0:
        continue

    l2_0 = float(g0["t10_l2_mm"].mean())
    l2_visible = float(gv["t10_l2_mm"].mean())

    rows.append({
        "sample_key": key,
        "n_view0": len(g0),
        "n_visible": len(gv),
        "view0_mean_l2_mm": l2_0,
        "visible_mean_l2_mm": l2_visible,
        "gap_view0_minus_visible_mm":
            l2_0 - l2_visible,
    })


paired = pd.DataFrame(rows)

assert len(paired) > 0

gap = paired["gap_view0_minus_visible_mm"].to_numpy()

n = len(gap)
worse = int((gap > 0).sum())
better = int((gap < 0).sum())
ties = int((gap == 0).sum())

# Exact one-sided sign test:
# H0: P(view0 worse) = 0.5
# No magnitude threshold is introduced.
non_tie = worse + better

if non_tie > 0:
    p_one_sided = sum(
        math.comb(non_tie, k)
        for k in range(worse, non_tie + 1)
    ) / (2 ** non_tie)
else:
    p_one_sided = float("nan")


OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

out_csv = (
    OUTDIR
    / "radio554_within_window_view0_vs_visible.csv"
)

paired.to_csv(
    out_csv,
    index=False,
)


print("=" * 72)
print("WITHIN-WINDOW VISIBILITY COMPARISON")
print("=" * 72)

print("total radio windows =", 554)
print("paired windows      =", n)
print()

print(
    "view0 worse than visible =",
    f"{worse}/{non_tie}"
    if non_tie else "NA",
)

print(
    "view0 better than visible =",
    f"{better}/{non_tie}"
    if non_tie else "NA",
)

print("ties =", ties)

print()
print(
    "median gap [view0-visible] mm =",
    float(np.median(gap)),
)

print(
    "mean gap [view0-visible] mm =",
    float(np.mean(gap)),
)

print(
    "one-sided exact sign-test p =",
    p_one_sided,
)

print()
print("SAVED =", out_csv)
print("NO NEW INFERENCE = True")
