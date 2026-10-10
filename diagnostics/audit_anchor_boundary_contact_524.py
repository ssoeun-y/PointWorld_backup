from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(
    "diagnostics/results/close_anchor_causal_5task_current10457"
)
OUT = Path(
    "diagnostics/results/anchor_boundary_contact_audit_524"
)
B = 0.24580418
TOL = 1e-6

files = sorted(ROOT.glob("task-*/per_clip_causal_metrics.csv"))

assert len(files) == 5, f"Expected 5 task files, got {len(files)}"

df = pd.concat(
    [
        pd.read_csv(f).assign(task=f.parent.name)
        for f in files
    ],
    ignore_index=True
)

assert len(df) == 524, f"Unexpected population: {len(df)}"

print("\n=== ACC COLUMNS ===")
acc_cols = [c for c in df if "acc" in c.lower()]
print("\n".join(acc_cols))

# Compare only SI-valued acceleration columns against B.
needed = [
    "original_acc_si",
    "free_acc_si",
    "anchor_acc_si",
    "gt_acquisition_min_gap_mm",
]

missing = [c for c in needed if c not in df]
if missing:
    raise RuntimeError(
        f"Missing verified columns: {missing}. "
        "Inspect source acceleration conversion before proceeding."
    )

for c in needed:
    df[c] = pd.to_numeric(df[c], errors="raise")
    assert np.isfinite(df[c]).all(), c

# ---------------------------------------------------
# 1. Confirm the NPZ-to-SI unit relationship
# ---------------------------------------------------
phase_files = sorted(
    ROOT.glob("task-*/matched_phases/*.npz")
)
assert len(phase_files) == 524

raw_acc = []

for p in phase_files:
    with np.load(p, allow_pickle=False) as z:
        raw_acc.append([
            float(z["original_acc"]),
            float(z["free_acc"]),
            float(z["anchor_acc"]),
        ])

raw_acc = np.asarray(raw_acc)

print("\n=== UNIT AUDIT ===")
print("NPZ original median:", np.median(raw_acc[:, 0]))
print("CSV original SI median:", df.original_acc_si.median())

ratio = (
    df.original_acc_si.median()
    / np.median(raw_acc[:, 0])
)
print("CSV SI / NPZ ratio:", ratio)

if not (
    np.isclose(ratio, 1.0, rtol=1e-3)
    or np.isclose(ratio, 25.0, rtol=1e-3)
):
    raise RuntimeError(
        "Unexpected acceleration scale. "
        "Verify source implementation before boundary comparison."
    )

# ---------------------------------------------------
# 2. Boundary feasibility using SI units
# ---------------------------------------------------
original = df.original_acc_si
free = df.free_acc_si
anchor = df.anchor_acc_si

high = original > B + TOL
free_ok = free <= B + TOL
anchor_ok = anchor <= B + TOL

df["original_high"] = high
df["both_boundary_ok"] = free_ok & anchor_ok

print("\n=== BOUNDARY FEASIBILITY (SI) ===")
print("Boundary:", B, "m/s^2")
print("Total:", len(df))
print("Original high:", int(high.sum()))
print("Free <= B:", int(free_ok.sum()))
print("Anchor <= B:", int(anchor_ok.sum()))
print(
    "Original high AND both corrected <= B:",
    int((high & free_ok & anchor_ok).sum())
)
print("Original SI median:", original.median())
print("Free SI median:", free.median())
print("Anchor SI median:", anchor.median())
print(
    "Max Free-Anchor SI difference:",
    (free-anchor).abs().max()
)

# ---------------------------------------------------
# 3. Compare existing matched-pair drift metrics
# ---------------------------------------------------
metrics = {
    "Original gap deviation":
        (
            "free_close_gap_absdiff_from_original_mm",
            "anchor_close_gap_absdiff_from_original_mm"
        ),
    "Original drift deviation":
        (
            "free_postclose_drift_absdiff_from_original_mm",
            "anchor_postclose_drift_absdiff_from_original_mm"
        ),
    "Absolute final drift":
        (
            "free_postclose_drift_q10_mm",
            "anchor_postclose_drift_q10_mm"
        ),
    "Maximum gap increase":
        (
            "free_max_gap_increase_mm",
            "anchor_max_gap_increase_mm"
        ),
    "Mean absolute gap change":
        (
            "free_mean_abs_gap_change_mm",
            "anchor_mean_abs_gap_change_mm"
        ),
}

def compare(sub, group_name):
    rows = []

    for name, (fcol, acol) in metrics.items():
        if len(sub) == 0:
            continue

        x = sub[fcol].to_numpy(dtype=float)
        y = sub[acol].to_numpy(dtype=float)

        if name == "Absolute final drift":
            x = np.abs(x)
            y = np.abs(y)

        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        if len(x) < 2:
            continue

        gain = x - y

        if np.allclose(gain, 0, atol=1e-12, rtol=0):
            pvalue = 1.0
        else:
            pvalue = wilcoxon(gain).pvalue

        rows.append({
            "group": group_name,
            "metric": name,
            "N": len(x),
            "Free_median_mm": np.median(x),
            "Anchor_median_mm": np.median(y),
            "paired_gain_median_mm": np.median(gain),
            "anchor_better_pct":
                100 * np.mean(gain > 1e-8),
            "wilcoxon_p": pvalue
        })

    return rows

print("\n=== GT GRASP PROXIMITY ===")

gt_min = df.gt_acquisition_min_gap_mm

print("GT minimum gap median:", gt_min.median(), "mm")

groups = {
    "All_524": np.ones(len(df), dtype=bool),
    "GT_min_le_5mm": gt_min <= 5,
    "GT_min_le_10mm": gt_min <= 10,
    "GT_min_le_20mm": gt_min <= 20,
    "GT_min_gt_20mm": gt_min > 20,
    "Original_high_boundary_feasible":
        high & free_ok & anchor_ok,
    "GT_le_10mm_AND_boundary_feasible":
        (gt_min <= 10) & high & free_ok & anchor_ok,
}

all_rows = []

for name, mask in groups.items():
    sub = df.loc[mask]
    print("\n=== GROUP:", name, "N=", len(sub), "===")

    result = compare(sub, name)
    all_rows.extend(result)

    for r in result:
        print(
            f"{r['metric']}: "
            f"Free={r['Free_median_mm']:.4f}, "
            f"Anchor={r['Anchor_median_mm']:.4f}, "
            f"paired gain={r['paired_gain_median_mm']:.4f}, "
            f"better={r['anchor_better_pct']:.1f}%, "
            f"p={r['wilcoxon_p']:.4g}"
        )

OUT.mkdir(parents=True, exist_ok=True)

pd.DataFrame(all_rows).to_csv(
    OUT / "boundary_contact_paired_results.csv",
    index=False
)

df.to_csv(
    OUT / "audited_population.csv",
    index=False
)

print("\nSaved:", OUT)
print("No PointWorld inference executed.")
print("Subgroup tests are exploratory, not causal proof of contact.")
