from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

R = Path("diagnostics/results")
W = R / "gt_world_comotion_current524"
S = R / "close_anchor_relative_replay_current524"
OUT = W / "grasp_A_B_per_clip.csv"

parts = []
for t in ("0000", "0001", "0006", "0008", "0010"):
    f = pd.read_csv(
        S / f"task-{t}" / "per_frame_relative_metrics.csv"
    )
    parts.append(f)

f = pd.concat(parts, ignore_index=True)
w = pd.read_csv(W / "per_clip_gt_world_comotion.csv")

needed = {
    "task", "clip", "condition", "frame",
    "frames_since_close", "absolute_q10_gap_mm",
    "original_gt_relative_delta_error_mm"
}
assert needed <= set(f.columns), sorted(needed - set(f.columns))
assert len(w) == 524
assert not w.duplicated(["task", "clip"]).any()
assert not f.duplicated(
    ["task", "clip", "frame", "condition"]
).any()

# A: GT vs Original grasp-time proximity discrepancy
close = f[
    (f["frames_since_close"] == 0)
    & f["condition"].isin(["GT", "Original"])
]

a = close.pivot(
    index=["task", "clip"],
    columns="condition",
    values="absolute_q10_gap_mm"
)

assert len(a) == 524
assert {"GT", "Original"} <= set(a.columns)

a = (
    (a["Original"] - a["GT"]).abs()
    .rename("A_initial_gap_difference_mm")
    .reset_index()
)

# B: original prediction's post-grasp relative-motion
# error against GT, at the final scored frame.
original = f[f["condition"] == "Original"].copy()

original = original.sort_values("frame")

end = original.groupby(
    ["task", "clip"], as_index=False
).tail(1)

b = end[[
    "task", "clip",
    "original_gt_relative_delta_error_mm"
]].rename(columns={
    "original_gt_relative_delta_error_mm":
    "B_end_relative_motion_error_mm"
})

assert len(b) == 524

d = w.merge(a, on=["task", "clip"], validate="one_to_one")
d = d.merge(b, on=["task", "clip"], validate="one_to_one")
assert len(d) == 524

for c in ("A_initial_gap_difference_mm",
          "B_end_relative_motion_error_mm"):
    assert np.isfinite(d[c]).all(), c
    assert (d[c] >= 0).all(), c

def flag(name):
    v = d[name]
    if pd.api.types.is_bool_dtype(v):
        return v
    return v.astype(str).str.lower().eq("true")

valid = flag("gt_world_measurement_valid")
moving = flag("moving_10mm")
boundary = flag("strict_boundary")

groups = {
    "all_valid": valid,
    "moderate_transport": valid & moving & flag("moderate"),
    "permissive_transport": valid & moving & flag("permissive"),
    "transport_boundary":
        valid & moving & flag("permissive") & boundary,
}

print("\n===== GRASP ERROR A/B DIAGNOSTIC =====")

for name, mask in groups.items():
    q = d[mask]
    n = len(q)
    print(f"\n{name}: N={n}")

    if n < 3:
        print("Insufficient observations")
        continue

    A = q["A_initial_gap_difference_mm"].to_numpy()
    B = q["B_end_relative_motion_error_mm"].to_numpy()

    rho, p = spearmanr(A, B)

    gain = (
        q["end_centroid_mm_Free"]
        - q["end_centroid_mm_Anchor"]
    ).to_numpy()

    nonzero = gain[gain != 0]
    pw = wilcoxon(nonzero).pvalue if len(nonzero) else 1.0

    print(f"A initial gap discrepancy: {np.median(A):.4f} mm")
    print(f"B future relative-motion error: {np.median(B):.4f} mm")
    print(f"A-B Spearman rho: {rho:+.4f}, p={p:.4g}")
    print(f"Anchor paired motion gain: {np.median(gain):+.4f} mm")
    print(f"Anchor lower: {100*np.mean(gain>0):.1f}%")
    print(f"Anchor Wilcoxon p: {pw:.4g}")

d.to_csv(OUT, index=False)

print("\n[COMPLETE]", OUT)
print(
    "A is a grasp-proximity proxy; "
    "B is GT-referenced relative-motion error. "
    "Different metrics must not be treated as causal contributions."
)
