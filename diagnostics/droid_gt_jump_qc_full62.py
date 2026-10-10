#!/usr/bin/env python3

from pathlib import Path
import csv

import numpy as np
import torch


ROOT = Path(
    "diagnostics/results/"
    "prediction_consistency_droid_unfiltered_full_20260804_135952/"
    "raw_sequences"
)

OUT_DIR = Path(
    "diagnostics/results/"
    "droid_gt_jump_qc_full62"
)

OUT_CSV = OUT_DIR / "per_sequence_timestep.csv"


def main():
    files = sorted(ROOT.glob("*.pt"))

    if len(files) != 62:
        raise RuntimeError(
            f"expected 62 files, found {len(files)}"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    processed_sequences = 0

    for p in files:
        x = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )

        gt_all = x[
            "gt_scene_flows"
        ].float()

        valid_all = x[
            "official_valid_mask"
        ].bool()

        moved_all = x[
            "scene_moved_mask"
        ].bool()

        # Same point population used by the
        # DROID local-coherence analysis:
        # final official-valid + moved.
        idx = torch.where(
            valid_all[-1] & moved_all[-1]
        )[0]

        n_population = int(
            idx.numel()
        )

        if n_population == 0:
            raise RuntimeError(
                f"{p.name}: zero final-valid moved points"
            )

        gt = gt_all[:, idx]
        valid = valid_all[:, idx]

        for t in range(1, gt.shape[0]):
            pair_valid = (
                valid[t - 1]
                & valid[t]
            )

            ids = torch.where(
                pair_valid
            )[0]

            n_valid = int(
                ids.numel()
            )

            if n_valid == 0:
                row = {
                    "file": p.name,
                    "key": x["key"],
                    "timestep": t,
                    "population_n": n_population,
                    "valid_pair_n": 0,
                    "valid_pair_fraction": 0.0,
                    "gt_disp_median_mm": np.nan,
                    "gt_disp_p90_mm": np.nan,
                    "gt_disp_p95_mm": np.nan,
                    "gt_disp_p99_mm": np.nan,
                    "gt_disp_max_mm": np.nan,
                }

            else:
                disp = torch.linalg.vector_norm(
                    gt[t, ids]
                    - gt[t - 1, ids],
                    dim=-1,
                ).cpu().numpy()

                row = {
                    "file": p.name,
                    "key": x["key"],
                    "timestep": t,
                    "population_n": n_population,
                    "valid_pair_n": n_valid,
                    "valid_pair_fraction": (
                        n_valid
                        / n_population
                    ),
                    "gt_disp_median_mm": (
                        np.median(disp)
                        * 1000
                    ),
                    "gt_disp_p90_mm": (
                        np.quantile(
                            disp,
                            0.90,
                        )
                        * 1000
                    ),
                    "gt_disp_p95_mm": (
                        np.quantile(
                            disp,
                            0.95,
                        )
                        * 1000
                    ),
                    "gt_disp_p99_mm": (
                        np.quantile(
                            disp,
                            0.99,
                        )
                        * 1000
                    ),
                    "gt_disp_max_mm": (
                        np.max(disp)
                        * 1000
                    ),
                }

            rows.append(row)

        processed_sequences += 1

    if len(rows) != 62 * 10:
        raise RuntimeError(
            f"expected 620 rows, found {len(rows)}"
        )

    fieldnames = list(
        rows[0].keys()
    )

    with OUT_CSV.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print("DROID GT JUMP QC FULL62")
    print("=" * 80)
    print(
        "processed_sequences =",
        processed_sequences,
    )
    print(
        "sequence_timestep_rows =",
        len(rows),
    )
    print(
        "output_csv =",
        OUT_CSV,
    )

    # Descriptive distribution only.
    # No exclusion threshold is chosen here.
    for metric in [
        "valid_pair_fraction",
        "gt_disp_median_mm",
        "gt_disp_p90_mm",
        "gt_disp_p95_mm",
        "gt_disp_p99_mm",
        "gt_disp_max_mm",
    ]:
        arr = np.asarray([
            r[metric]
            for r in rows
            if np.isfinite(r[metric])
        ], dtype=np.float64)

        print("-" * 80)
        print("metric =", metric)
        print(
            "median =",
            np.median(arr),
        )
        print(
            "p90 =",
            np.quantile(
                arr,
                0.90,
            ),
        )
        print(
            "p95 =",
            np.quantile(
                arr,
                0.95,
            ),
        )
        print(
            "p99 =",
            np.quantile(
                arr,
                0.99,
            ),
        )
        print(
            "max =",
            np.max(arr),
        )

    print("-" * 80)
    print(
        "SUCCESS =",
        processed_sequences == 62
        and len(rows) == 620
        and OUT_CSV.exists(),
    )


if __name__ == "__main__":
    main()
