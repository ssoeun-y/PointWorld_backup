#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]

DIR = (
    ROOT /
    "diagnostics/results/"
    "goal_preserved_global_boundary_whole_heldout_pointcloud_structure"
)

FILES = [
    DIR / "whole_heldout_task0000__whole_heldout_pointcloud_structure.csv",
    DIR / "whole_heldout_task0001__whole_heldout_pointcloud_structure.csv",
    DIR / "whole_heldout_task0006_0008_0010__whole_heldout_pointcloud_structure.csv",
]

OUT = DIR / "whole_heldout_5task_aggregate.csv"

for p in FILES:
    if not p.exists():
        raise FileNotFoundError(p)

frames = []

for p in FILES:
    d = pd.read_csv(p)
    print(
        p.name,
        "| rows =", len(d),
        "| unique clips =",
        d["clip"].nunique()
    )
    frames.append(d)

d = pd.concat(
    frames,
    ignore_index=True
)

if d["clip"].duplicated().any():
    dup = d.loc[
        d["clip"].duplicated(False),
        "clip"
    ]
    raise RuntimeError(
        "Duplicate clips:\n"
        + dup.to_string(index=False)
    )

assert len(d) == 2397, len(d)
assert d["clip"].nunique() == 2397

d.to_csv(
    OUT,
    index=False
)

print()
print("=" * 105)
print("A — GLOBAL WHOLE-HELDOUT 5-TASK AGGREGATE")
print("=" * 105)

print("found =", len(d))

if "task" in d.columns:
    print()
    print("Rows by task:")
    print(
        d["task"]
        .value_counts()
        .sort_index()
        .to_string()
    )

METRICS = [
    "rigidity_mean_mm",
    "rigidity_p95_mm",
    "final_rigidity_mean_mm",
    "jitter_mean_mm",
    "jitter_p95_mm",
]

summary = []

for metric in METRICS:

    # Support either original_/corrected_ naming pattern.
    candidates = [
        (
            f"original_{metric}",
            f"corrected_{metric}"
        ),
        (
            f"{metric}_original",
            f"{metric}_corrected"
        ),
    ]

    pair = None

    for oc, cc in candidates:
        if oc in d.columns and cc in d.columns:
            pair = (oc, cc)
            break

    if pair is None:
        print()
        print(
            metric,
            ": columns not found"
        )
        continue

    oc, cc = pair

    x = pd.to_numeric(
        d[oc],
        errors="coerce"
    ).to_numpy(float)

    y = pd.to_numeric(
        d[cc],
        errors="coerce"
    ).to_numpy(float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x > 0)
    )

    x = x[valid]
    y = y[valid]

    ratio = y / x

    median_paired_ratio = float(
        np.median(ratio)
    )

    improvement = (
        1.0 - median_paired_ratio
    ) * 100.0

    lower_frac = float(
        np.mean(y < x)
    )

    try:
        w = wilcoxon(
            y,
            x,
            zero_method="wilcox",
            alternative="two-sided"
        )
        pval = float(w.pvalue)
    except ValueError:
        pval = np.nan

    row = {
        "metric": metric,
        "N": len(x),
        "original_median": float(np.median(x)),
        "corrected_median": float(np.median(y)),
        "median_paired_ratio":
            median_paired_ratio,
        "paired_improvement_pct":
            improvement,
        "corrected_lower_fraction":
            lower_frac,
        "wilcoxon_p":
            pval,
    }

    summary.append(row)

    print()
    print(metric)
    print("  N =", len(x))
    print(
        "  original median =",
        np.median(x)
    )
    print(
        "  corrected median =",
        np.median(y)
    )
    print(
        "  median paired ratio =",
        median_paired_ratio
    )
    print(
        "  paired improvement =",
        f"{improvement:.6f}%"
    )
    print(
        "  corrected lower fraction =",
        lower_frac
    )
    print(
        "  Wilcoxon p =",
        pval
    )

pd.DataFrame(summary).to_csv(
    DIR /
    "whole_heldout_5task_metric_summary.csv",
    index=False
)

print()
print("=" * 105)
print("saved =", OUT)
print("SUCCESS = True")
print("=" * 105)
