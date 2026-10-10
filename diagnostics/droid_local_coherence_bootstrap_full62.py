#!/usr/bin/env python3

from pathlib import Path
import csv

import numpy as np


CSV_PATH = Path(
    "diagnostics/results/"
    "droid_local_coherence_p95gate_full62/"
    "per_sequence_summary.csv"
)

N_BOOT = 10000
SEED = 42


def main():
    with CSV_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 62:
        raise RuntimeError(
            f"expected 62 rows, found {len(rows)}"
        )

    rel_mean = np.asarray([
        float(r["relvel_gap_mean_mm"])
        for r in rows
    ], dtype=np.float64)

    rel_median = np.asarray([
        float(r["relvel_gap_median_mm"])
        for r in rows
    ], dtype=np.float64)

    positive_step = np.asarray([
        float(r["positive_step_fraction"])
        for r in rows
    ], dtype=np.float64)

    rng = np.random.default_rng(SEED)

    boot_mean = np.empty(N_BOOT)
    boot_median = np.empty(N_BOOT)
    boot_positive_seq = np.empty(N_BOOT)
    boot_positive_step = np.empty(N_BOOT)

    n = len(rows)

    for b in range(N_BOOT):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        x_mean = rel_mean[idx]
        x_median = rel_median[idx]
        x_step = positive_step[idx]

        boot_mean[b] = np.mean(x_mean)
        boot_median[b] = np.median(x_median)
        boot_positive_seq[b] = np.mean(
            x_mean > 0
        )
        boot_positive_step[b] = np.mean(
            x_step > 0.5
        )

    print("=" * 80)
    print("DROID P95-GATED LOCAL COHERENCE BOOTSTRAP")
    print("=" * 80)

    print(
        "relvel_gap_mean_mm =",
        np.mean(rel_mean),
    )
    print(
        "relvel_gap_mean_bootstrap_ci95 =",
        np.quantile(
            boot_mean,
            [0.025, 0.975],
        ).tolist(),
    )

    print(
        "relvel_gap_median_mm =",
        np.median(rel_median),
    )
    print(
        "relvel_gap_median_bootstrap_ci95 =",
        np.quantile(
            boot_median,
            [0.025, 0.975],
        ).tolist(),
    )

    print(
        "positive_sequence_fraction =",
        np.mean(rel_mean > 0),
    )
    print(
        "positive_sequence_fraction_ci95 =",
        np.quantile(
            boot_positive_seq,
            [0.025, 0.975],
        ).tolist(),
    )

    print(
        "positive_step_gt_half_fraction =",
        np.mean(positive_step > 0.5),
    )
    print(
        "positive_step_gt_half_fraction_ci95 =",
        np.quantile(
            boot_positive_step,
            [0.025, 0.975],
        ).tolist(),
    )

    print(
        "processed_sequences =",
        n,
    )
    print(
        "bootstrap_resamples =",
        N_BOOT,
    )
    print(
        "SUCCESS =",
        n == 62,
    )


if __name__ == "__main__":
    main()
