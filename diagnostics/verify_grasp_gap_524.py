#!/usr/bin/env python3
"""Read-only validation of existing N=524 grasp-anchor results."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    "diagnostics/results/close_anchor_causal_5task_current10457"
)
OUT = Path(
    "diagnostics/results/grasp_gap_verification_current524"
)

TASKS = ("0000", "0001", "0006", "0008", "0010")
CONDS = ("original", "free", "anchor")

# 1. Load exactly the existing five-task population
dfs = []

for task in TASKS:
    path = ROOT / f"task-{task}" / "per_clip_causal_metrics.csv"

    if not path.is_file():
        raise FileNotFoundError(path)

    d = pd.read_csv(path)
    d["_source_task"] = task
    dfs.append(d)

df = pd.concat(dfs, ignore_index=True)

assert len(df) == 524, f"Expected 524, got {len(df)}"

print("\n=== POPULATION ===")
print("Total:", len(df))
print(df.groupby("_source_task").size())

# 2. Verify data columns
suffixes = [
    "close_gap_q10_mm",
    "end_gap_q10_mm",
    "postclose_drift_q10_mm",
    "max_gap_increase_mm",
    "mean_abs_gap_change_mm",
]

required = [
    f"{c}_{s}" for c in CONDS for s in suffixes
]

required += [
    "gt_acquisition_min_gap_mm",
    "gt_acquisition_q10_gap_mm",
    "free_close_gap_absdiff_from_original_mm",
    "anchor_close_gap_absdiff_from_original_mm",
    "free_postclose_drift_absdiff_from_original_mm",
    "anchor_postclose_drift_absdiff_from_original_mm",
]

for col in required:
    if col not in df:
        raise ValueError(f"Missing column: {col}")

    df[col] = pd.to_numeric(df[col], errors="raise")

    if not np.isfinite(df[col]).all():
        raise ValueError(f"Non-finite values: {col}")

# 3. Independently recompute previously reported deviations
print("\n=== REPRODUCIBILITY AUDIT ===")

pairs = [
    (
        "close_gap_q10_mm",
        "close_gap_absdiff_from_original_mm",
    ),
    (
        "postclose_drift_q10_mm",
        "postclose_drift_absdiff_from_original_mm",
    ),
]

for metric, reported in pairs:
    for cond in ("free", "anchor"):
        recomputed = (
            df[f"{cond}_{metric}"]
            - df[f"original_{metric}"]
        ).abs()

        saved = df[f"{cond}_{reported}"]

        err = (recomputed - saved).abs().max()

        print(cond, reported, "max error:", err)

        if err > 1e-5:
            raise ValueError(
                f"Metric mismatch: {cond}_{reported}"
            )

# 4. Absolute-gap and drift summary
print("\n=== ABSOLUTE GAP / DRIFT (mm) ===")

rows = []

for metric in suffixes:
    row = {"metric": metric}

    print("\n", metric)

    for cond in CONDS:
        value = df[f"{cond}_{metric}"].median()
        row[cond] = value
        print(f"  {cond:9s}: {value:.6f}")

    rows.append(row)

summary = pd.DataFrame(rows)

# 5. Ground-truth acquisition gap
print("\n=== GT ACQUISITION GAP ===")

for col in [
    "gt_acquisition_min_gap_mm",
    "gt_acquisition_q10_gap_mm",
]:
    values = df[col]

    print("\n", col)
    print("median:", values.median())
    print("p25:", values.quantile(0.25))
    print("p75:", values.quantile(0.75))
    print("<= 5mm:", (values <= 5).mean() * 100, "%")
    print("<=10mm:", (values <= 10).mean() * 100, "%")

print(
    "\nNOTE: GT acquisition gap is not automatically "
    "the same-frame predicted grasp gap."
)

# 6. Paired anchor improvements
print("\n=== PAIRED ANCHOR EFFECT ===")

comparison = []

for name, suffix in [
    ("grasp_gap", "close_gap_absdiff_from_original_mm"),
    ("postclose_drift", "postclose_drift_absdiff_from_original_mm"),
]:
    free = df[f"free_{suffix}"].to_numpy()
    anchor = df[f"anchor_{suffix}"].to_numpy()

    improvement = free - anchor

    better = int(np.sum(improvement > 1e-8))
    worse = int(np.sum(improvement < -1e-8))
    ties = len(improvement) - better - worse

    print("\n", name)
    print("Free median:", np.median(free))
    print("Anchor median:", np.median(anchor))
    print("Anchor better:", better, "/", len(df))
    print("Ties:", ties)
    print("Worse:", worse)
    print("Paired median gain:", np.median(improvement))

    try:
        from scipy.stats import wilcoxon
        print("Wilcoxon p:", wilcoxon(improvement).pvalue)
    except ImportError:
        print("Scipy unavailable; p-value skipped")

    comparison.append({
        "metric": name,
        "free_median_mm": np.median(free),
        "anchor_median_mm": np.median(anchor),
        "anchor_better_n": better,
        "paired_median_gain_mm": np.median(improvement),
    })

# 7. Recheck stored matched-dose phases
print("\n=== GRASP ANCHOR FAIRNESS ===")

phase_files = sorted(
    ROOT.glob("task-*/matched_phases/*.npz")
)

print("Phase files:", len(phase_files))

if len(phase_files) != 524:
    raise ValueError("Expected 524 phase files")

dose_errors = []
anchor_errors = []
final_errors = []

for path in phase_files:
    with np.load(path, allow_pickle=False) as data:
        t_close = int(data["t_close"])

        free_acc = float(data["free_acc"])
        anchor_acc = float(data["anchor_acc"])
        anchor_phase = np.asarray(
            data["anchor_phase"], dtype=float
        )

        if not (0 <= t_close < len(anchor_phase)):
            raise ValueError(f"Invalid t_close: {path}")

        dose_errors.append(abs(free_acc - anchor_acc))
        anchor_errors.append(
            abs(anchor_phase[t_close] - t_close)
        )
        final_errors.append(
            abs(anchor_phase[-1] - 10.0)
        )

print("Max RMS Acc difference:", max(dose_errors))
print("Max grasp phase error:", max(anchor_errors))
print("Max final phase error:", max(final_errors))

# 8. Save results
OUT.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    OUT / "absolute_gap_summary.csv",
    index=False,
)

pd.DataFrame(comparison).to_csv(
    OUT / "paired_anchor_effect.csv",
    index=False,
)

print("\n=== SAVED ===")
print(OUT)

print("\n=== REMAINING CHECKS ===")
print("Predicted minimum gap at t_close: NOT in summary CSV")
print("Full frame-by-frame gap: NOT in summary CSV")
print("Verify original source definition and prediction caches.")
print("No PointWorld inference was executed.")
