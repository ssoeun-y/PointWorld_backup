#!/usr/bin/env python3

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
    "behavior_visibility_moved_overlap_audit"
)

df = pd.read_csv(INFILE)

assert len(df) == 2314
assert df["key"].nunique() == 578

# Same robust bool interpretation as previous audit.
if df["is_moved_object"].dtype != bool:
    df["is_moved_object"] = (
        df["is_moved_object"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

df["episode_key"] = (
    df["key"]
    .astype(str)
    .str.rsplit("-", n=1)
    .str[0]
)

# Same paired population used in previous episode audit.
paired = df[
    df["gap_view0_minus_visible_mm"].notna()
].copy()

# ------------------------------------------------------------
# One row per episode-object.
# Check whether moved status changes across overlapping windows.
# ------------------------------------------------------------
rows = []

for (
    episode_key,
    object_id,
    object_name,
), g in paired.groupby(
    [
        "episode_key",
        "object_id",
        "object_name",
    ],
    sort=True,
):

    moved = g["is_moved_object"].to_numpy(bool)

    n_moved = int(moved.sum())
    n_nonmoved = int((~moved).sum())

    if n_moved > 0 and n_nonmoved > 0:
        status = "mixed"
    elif n_moved > 0:
        status = "always_moved"
    else:
        status = "always_nonmoved"

    rows.append({
        "episode_key": episode_key,
        "object_id": int(object_id),
        "object_name": object_name,

        "n_paired_windows": len(g),
        "n_moved_windows": n_moved,
        "n_nonmoved_windows": n_nonmoved,
        "status": status,

        "mean_gap_all_windows_mm":
            float(
                g[
                    "gap_view0_minus_visible_mm"
                ].mean()
            ),

        "mean_gap_moved_windows_mm":
            (
                float(
                    g.loc[
                        g["is_moved_object"],
                        "gap_view0_minus_visible_mm",
                    ].mean()
                )
                if n_moved > 0
                else np.nan
            ),

        "mean_gap_nonmoved_windows_mm":
            (
                float(
                    g.loc[
                        ~g["is_moved_object"],
                        "gap_view0_minus_visible_mm",
                    ].mean()
                )
                if n_nonmoved > 0
                else np.nan
            ),
    })


audit = pd.DataFrame(rows)

assert len(audit) == 770

n_always_moved = int(
    (audit["status"] == "always_moved").sum()
)

n_always_nonmoved = int(
    (audit["status"] == "always_nonmoved").sum()
)

n_mixed = int(
    (audit["status"] == "mixed").sum()
)

# Reproduce previous subset unit counts.
moved_unit_count = (
    n_always_moved + n_mixed
)

nonmoved_unit_count = (
    n_always_nonmoved + n_mixed
)

assert moved_unit_count == 157, moved_unit_count
assert nonmoved_unit_count == 661, nonmoved_unit_count
assert (
    n_always_moved
    + n_always_nonmoved
    + n_mixed
    == 770
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

audit.to_csv(
    OUTDIR / "episode_object_moved_status_audit.csv",
    index=False,
)

mixed = audit[
    audit["status"] == "mixed"
].copy()

mixed.to_csv(
    OUTDIR / "mixed_moved_status_episode_objects.csv",
    index=False,
)


print("=" * 72)
print("MOVED / NON-MOVED OVERLAP AUDIT")
print("=" * 72)

print("paired episode-object units =", len(audit))
print()

print("always moved     =", n_always_moved)
print("always non-moved =", n_always_nonmoved)
print("mixed status     =", n_mixed)

print()
print(
    "reconstructed moved units =",
    moved_unit_count,
)

print(
    "reconstructed non-moved units =",
    nonmoved_unit_count,
)

print(
    "overlap (= mixed) =",
    n_mixed,
)

print()
print(
    "157 + 661 - 770 =",
    157 + 661 - 770,
)

print(
    "OVERLAP_ACCOUNTED_EXACTLY =",
    bool(
        n_mixed
        == (157 + 661 - 770)
    ),
)

print()
print("=== MIXED OBJECT GAP SUMMARY ===")

if len(mixed):
    print(
        "mixed mean gap, moved windows [mm] =",
        float(
            mixed[
                "mean_gap_moved_windows_mm"
            ].mean()
        ),
    )

    print(
        "mixed mean gap, non-moved windows [mm] =",
        float(
            mixed[
                "mean_gap_nonmoved_windows_mm"
            ].mean()
        ),
    )

    print(
        "mixed median gap, moved windows [mm] =",
        float(
            mixed[
                "mean_gap_moved_windows_mm"
            ].median()
        ),
    )

    print(
        "mixed median gap, non-moved windows [mm] =",
        float(
            mixed[
                "mean_gap_nonmoved_windows_mm"
            ].median()
        ),
    )

print()
print(
    "SAVED =",
    OUTDIR / "episode_object_moved_status_audit.csv",
)

print(
    "SAVED =",
    OUTDIR / "mixed_moved_status_episode_objects.csv",
)

print("NO NEW INFERENCE = True")
print("NO NEW THRESHOLD = True")
