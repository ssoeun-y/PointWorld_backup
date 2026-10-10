#!/usr/bin/env python3

from pathlib import Path
import math
import numpy as np
import pandas as pd

SOURCE = Path(
    "diagnostics/results/"
    "behavior_allobject_visibility_within_object/"
    "all2314_object_visibility.csv"
)

OUTDIR = Path(
    "diagnostics/results/"
    "behavior_visibility_mixed_state_pair"
)

df = pd.read_csv(SOURCE)

if df["is_moved_object"].dtype != bool:
    df["is_moved_object"] = (
        df["is_moved_object"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

df["episode_key"] = (
    df["key"].astype(str)
    .str.rsplit("-", n=1)
    .str[0]
)

# Same paired population already used:
# each window must contain both 0-view and visible points.
df = df[
    df["gap_view0_minus_visible_mm"].notna()
].copy()

rows = []

for (ep, oid, name), g in df.groupby(
    ["episode_key", "object_id", "object_name"],
    sort=True,
):
    gm = g[g["is_moved_object"]]
    gs = g[~g["is_moved_object"]]

    # Mixed episode-object only.
    if len(gm) == 0 or len(gs) == 0:
        continue

    moved_gap = float(
        gm["gap_view0_minus_visible_mm"].mean()
    )
    nonmoved_gap = float(
        gs["gap_view0_minus_visible_mm"].mean()
    )

    rows.append({
        "episode_key": ep,
        "object_id": int(oid),
        "object_name": name,
        "n_moved_windows": len(gm),
        "n_nonmoved_windows": len(gs),
        "moved_gap_mm": moved_gap,
        "nonmoved_gap_mm": nonmoved_gap,
        "moved_minus_nonmoved_gap_mm":
            moved_gap - nonmoved_gap,
    })

out = pd.DataFrame(rows)

# Previous audit established exactly 48 mixed units.
assert len(out) == 48, len(out)

delta = out[
    "moved_minus_nonmoved_gap_mm"
].to_numpy(float)

positive = int((delta > 0).sum())
negative = int((delta < 0).sum())
ties = int((delta == 0).sum())

n = positive + negative

# Same exact sign-test form already used previously.
p = sum(
    math.comb(n, k)
    for k in range(positive, n + 1)
) / (2 ** n)

OUTDIR.mkdir(parents=True, exist_ok=True)

out.to_csv(
    OUTDIR / "mixed48_moved_vs_nonmoved.csv",
    index=False,
)

print("=" * 72)
print("MIXED 48: SAME OBJECT, MOVED VS NON-MOVED")
print("=" * 72)

print("episode-object units =", len(out))
print(
    "visibility penalty larger when moved =",
    f"{positive}/{n}",
)
print(
    "visibility penalty larger when non-moved =",
    f"{negative}/{n}",
)
print("ties =", ties)

print(
    "median [moved - nonmoved] gap mm =",
    float(np.median(delta)),
)
print(
    "mean [moved - nonmoved] gap mm =",
    float(np.mean(delta)),
)
print(
    "one-sided exact sign-test p =",
    p,
)

print()
print("NO NEW INFERENCE = True")
print("NO NEW VISIBILITY DEFINITION = True")
print("NO NEW MAGNITUDE THRESHOLD = True")
