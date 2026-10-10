#!/usr/bin/env python3

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


INFILE = Path(
    "diagnostics/results/"
    "behavior_allobject_visibility_within_object/"
    "all2314_object_visibility.csv"
)

OUTDIR = Path(
    "diagnostics/results/"
    "behavior_visibility_episode_object_audit"
)


df = pd.read_csv(INFILE)

assert len(df) == 2314
assert df["key"].nunique() == 578

# Robust bool parsing.
if df["is_moved_object"].dtype != bool:
    df["is_moved_object"] = (
        df["is_moved_object"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

# Example:
# task-0000_episode_00002530-300:311
# -> task-0000_episode_00002530
df["episode_key"] = (
    df["key"]
    .astype(str)
    .str.rsplit("-", n=1)
    .str[0]
)

# Only same-object comparisons for which BOTH
# 0-view and visible points existed.
paired = df[
    df["gap_view0_minus_visible_mm"].notna()
].copy()

assert len(paired) > 0


def exact_one_sided_sign_p(worse, better):
    """
    Exact sign test.
    H0: P(gap > 0) = 0.5
    No magnitude threshold.
    """
    n = worse + better

    if n == 0:
        return float("nan")

    return float(
        sum(
            math.comb(n, k)
            for k in range(worse, n + 1)
        )
        / (2 ** n)
    )


def make_episode_object_units(src, regime):
    """
    Collapse overlapping windows so each
    (episode, object) receives one unit of weight.

    The primary unit gap is simply the mean of the
    already-defined per-window gaps.
    Median is also stored descriptively.
    """
    units = (
        src.groupby(
            [
                "episode_key",
                "object_id",
                "object_name",
            ],
            as_index=False,
        )
        .agg(
            n_windows=(
                "key",
                "size",
            ),
            gap_mean_mm=(
                "gap_view0_minus_visible_mm",
                "mean",
            ),
            gap_median_mm=(
                "gap_view0_minus_visible_mm",
                "median",
            ),
            view0_l2_mean_mm=(
                "view0_t10_l2_mm",
                "mean",
            ),
            visible_l2_mean_mm=(
                "visible_t10_l2_mm",
                "mean",
            ),
            frac_view0_mean=(
                "frac_view0",
                "mean",
            ),
        )
    )

    units["regime"] = regime

    return units


all_units = make_episode_object_units(
    paired,
    "all",
)

moved_units = make_episode_object_units(
    paired[
        paired["is_moved_object"]
    ],
    "moved",
)

nonmoved_units = make_episode_object_units(
    paired[
        ~paired["is_moved_object"]
    ],
    "non_moved",
)


def summarize_units(name, units):

    gap = units[
        "gap_mean_mm"
    ].to_numpy(float)

    worse = int(
        (gap > 0).sum()
    )

    better = int(
        (gap < 0).sum()
    )

    ties = int(
        (gap == 0).sum()
    )

    non_tie = worse + better

    p = exact_one_sided_sign_p(
        worse,
        better,
    )

    result = {
        "name":
            name,

        "episode_object_units":
            int(len(units)),

        "unique_episodes":
            int(
                units[
                    "episode_key"
                ].nunique()
            ),

        "worse":
            worse,

        "better":
            better,

        "ties":
            ties,

        "non_tie":
            non_tie,

        "worse_fraction":
            (
                worse / non_tie
                if non_tie
                else float("nan")
            ),

        "median_gap_mean_mm":
            (
                float(
                    np.median(gap)
                )
                if len(gap)
                else float("nan")
            ),

        "mean_gap_mean_mm":
            (
                float(
                    np.mean(gap)
                )
                if len(gap)
                else float("nan")
            ),

        "one_sided_exact_sign_p":
            p,
    }

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    print(
        "episode-object units =",
        result[
            "episode_object_units"
        ],
    )

    print(
        "unique episodes      =",
        result[
            "unique_episodes"
        ],
    )

    print(
        "view0 worse =",
        f"{worse}/{non_tie}"
        if non_tie
        else "NA",
    )

    print(
        "view0 better =",
        f"{better}/{non_tie}"
        if non_tie
        else "NA",
    )

    print(
        "ties =",
        ties,
    )

    print(
        "worse fraction =",
        result[
            "worse_fraction"
        ],
    )

    print(
        "median episode-object gap [mm] =",
        result[
            "median_gap_mean_mm"
        ],
    )

    print(
        "mean episode-object gap [mm] =",
        result[
            "mean_gap_mean_mm"
        ],
    )

    print(
        "one-sided exact sign-test p =",
        p,
    )

    return result


summary = {}

summary["all"] = summarize_units(
    "ALL: EPISODE-OBJECT LEVEL",
    all_units,
)

summary["moved"] = summarize_units(
    "MOVED: EPISODE-OBJECT LEVEL",
    moved_units,
)

summary["non_moved"] = summarize_units(
    "NON-MOVED: EPISODE-OBJECT LEVEL",
    nonmoved_units,
)


# ============================================================
# Secondary robustness check:
# give EACH EPISODE only one vote in moved-object analysis.
#
# This removes both:
#   1) overlapping-window multiplicity
#   2) multiple moved objects within one episode
# ============================================================

moved_episode = (
    moved_units.groupby(
        "episode_key",
        as_index=False,
    )
    .agg(
        n_moved_objects=(
            "object_name",
            "size",
        ),
        gap_mean_mm=(
            "gap_mean_mm",
            "mean",
        ),
    )
)

ep_gap = moved_episode[
    "gap_mean_mm"
].to_numpy(float)

ep_worse = int(
    (ep_gap > 0).sum()
)

ep_better = int(
    (ep_gap < 0).sum()
)

ep_ties = int(
    (ep_gap == 0).sum()
)

ep_non_tie = (
    ep_worse + ep_better
)

ep_p = exact_one_sided_sign_p(
    ep_worse,
    ep_better,
)

summary["moved_episode_level"] = {
    "episodes":
        int(len(moved_episode)),

    "worse":
        ep_worse,

    "better":
        ep_better,

    "ties":
        ep_ties,

    "worse_fraction":
        (
            ep_worse / ep_non_tie
            if ep_non_tie
            else float("nan")
        ),

    "median_gap_mm":
        (
            float(
                np.median(ep_gap)
            )
            if len(ep_gap)
            else float("nan")
        ),

    "mean_gap_mm":
        (
            float(
                np.mean(ep_gap)
            )
            if len(ep_gap)
            else float("nan")
        ),

    "one_sided_exact_sign_p":
        ep_p,
}


print()
print("=" * 72)
print("MOVED: ONE VOTE PER EPISODE")
print("=" * 72)

print(
    "episodes =",
    len(moved_episode),
)

print(
    "view0 worse =",
    f"{ep_worse}/{ep_non_tie}"
    if ep_non_tie
    else "NA",
)

print(
    "view0 better =",
    f"{ep_better}/{ep_non_tie}"
    if ep_non_tie
    else "NA",
)

print(
    "ties =",
    ep_ties,
)

print(
    "worse fraction =",
    (
        ep_worse / ep_non_tie
        if ep_non_tie
        else float("nan")
    ),
)

print(
    "median episode gap [mm] =",
    float(
        np.median(ep_gap)
    ),
)

print(
    "mean episode gap [mm] =",
    float(
        np.mean(ep_gap)
    ),
)

print(
    "one-sided exact sign-test p =",
    ep_p,
)


# ============================================================
# Save report-ready artifacts.
# ============================================================

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

all_units.to_csv(
    OUTDIR
    / "episode_object_all.csv",
    index=False,
)

moved_units.to_csv(
    OUTDIR
    / "episode_object_moved.csv",
    index=False,
)

nonmoved_units.to_csv(
    OUTDIR
    / "episode_object_nonmoved.csv",
    index=False,
)

moved_episode.to_csv(
    OUTDIR
    / "episode_moved_one_vote.csv",
    index=False,
)

with open(
    OUTDIR / "summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print("=" * 72)
print("AUDIT")
print("=" * 72)

print(
    "source windows =",
    len(df),
)

print(
    "paired source windows =",
    len(paired),
)

print(
    "NO NEW INFERENCE = True"
)

print(
    "NO NEW VISIBILITY DEFINITION = True"
)

print(
    "NO MAGNITUDE THRESHOLD = True"
)

print(
    "SAVED =",
    OUTDIR,
)
