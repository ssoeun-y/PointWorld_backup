#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import csv

import numpy as np
from scipy.stats import spearmanr


LOCAL_CSV = Path(
    "diagnostics/results/"
    "droid_local_coherence_p95gate_full62/"
    "per_sequence_summary.csv"
)

GT_CSV = Path(
    "diagnostics/results/"
    "droid_gt_jump_qc_full62/"
    "per_sequence_timestep.csv"
)


def main():
    # -------------------------------------------------
    # Local-coherence result: exactly one row / sequence.
    # -------------------------------------------------
    with LOCAL_CSV.open(newline="") as f:
        local_rows = list(csv.DictReader(f))

    if len(local_rows) != 62:
        raise RuntimeError(
            f"expected 62 local rows, found {len(local_rows)}"
        )

    local = {
        r["file"]: r
        for r in local_rows
    }

    if len(local) != 62:
        raise RuntimeError(
            "duplicate local-coherence sequence keys"
        )

    # -------------------------------------------------
    # GT-jump QC: aggregate 10 timesteps to sequence level.
    # No sequence is excluded.
    # -------------------------------------------------
    by_file = defaultdict(list)

    with GT_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            value = float(
                r["gt_disp_p95_mm"]
            )

            if np.isfinite(value):
                by_file[r["file"]].append(value)

    if len(by_file) != 62:
        raise RuntimeError(
            f"expected 62 GT-QC sequences, found {len(by_file)}"
        )

    joined = []

    for file in sorted(local):
        if file not in by_file:
            raise RuntimeError(
                f"missing GT QC for {file}"
            )

        gt_vals = np.asarray(
            by_file[file],
            dtype=np.float64,
        )

        # Sequence-level GT discontinuity descriptors.
        gt_median_p95 = float(
            np.median(gt_vals)
        )

        gt_p90_p95 = float(
            np.quantile(
                gt_vals,
                0.90,
            )
        )

        relvel_mean = float(
            local[file]["relvel_gap_mean_mm"]
        )

        relvel_median = float(
            local[file]["relvel_gap_median_mm"]
        )

        positive_steps = float(
            local[file]["positive_step_fraction"]
        )

        joined.append({
            "file": file,
            "gt_median_timestep_p95_mm": gt_median_p95,
            "gt_p90_timestep_p95_mm": gt_p90_p95,
            "relvel_gap_mean_mm": relvel_mean,
            "relvel_gap_median_mm": relvel_median,
            "positive_step_fraction": positive_steps,
        })

    if len(joined) != 62:
        raise RuntimeError(
            f"expected 62 joined rows, found {len(joined)}"
        )

    # -------------------------------------------------
    # Continuous association: no arbitrary exclusion threshold.
    # -------------------------------------------------
    for gt_metric in [
        "gt_median_timestep_p95_mm",
        "gt_p90_timestep_p95_mm",
    ]:
        x = np.asarray([
            r[gt_metric]
            for r in joined
        ])

        print("=" * 80)
        print("GT_METRIC =", gt_metric)

        for outcome in [
            "relvel_gap_mean_mm",
            "relvel_gap_median_mm",
            "positive_step_fraction",
        ]:
            y = np.asarray([
                r[outcome]
                for r in joined
            ])

            rho, p = spearmanr(
                x,
                y,
            )

            print(
                outcome,
                "spearman_rho =",
                float(rho),
                "p =",
                float(p),
            )

    # -------------------------------------------------
    # Descriptive quartiles only.
    # These do NOT exclude any sequence.
    # -------------------------------------------------
    gt = np.asarray([
        r["gt_median_timestep_p95_mm"]
        for r in joined
    ])

    rel = np.asarray([
        r["relvel_gap_mean_mm"]
        for r in joined
    ])

    edges = np.quantile(
        gt,
        [0.25, 0.50, 0.75],
    )

    bins = np.digitize(
        gt,
        edges,
        right=True,
    )

    print("=" * 80)
    print("GT-JUMP QUARTILES")

    for q in range(4):
        mask = bins == q

        print(
            f"Q{q + 1}",
            "n =", int(mask.sum()),
            "gt_median_mm =", float(np.median(gt[mask])),
            "relvel_gap_mean_mm =", float(np.mean(rel[mask])),
            "relvel_gap_median_mm =", float(np.median(rel[mask])),
            "relvel_positive_fraction =", float(np.mean(rel[mask] > 0)),
        )

    print("=" * 80)
    print("joined_sequences =", len(joined))
    print("SUCCESS =", len(joined) == 62)


if __name__ == "__main__":
    main()
