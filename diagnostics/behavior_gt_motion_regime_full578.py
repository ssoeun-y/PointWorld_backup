#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


TRAJ_DIR = Path(
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237"
)

DEBRIS_CSV = Path(
    "diagnostics/results/"
    "behavior_object_debris_geometry_full578_20260810_124340/"
    "per_point_debris.csv"
)

GEOM_CSV = Path(
    "diagnostics/results/"
    "behavior_cluster_relative_dynamics_full578_20260810_140244/"
    "per_object_timestep_dynamics.csv"
)

MATCHED_CSV = Path(
    "diagnostics/results/"
    "behavior_cluster_gt_matched_control_full578_20260811_034030/"
    "matched_control.csv"
)

OUT_ROOT = Path("diagnostics/results")

EPS = 1e-8


def vec_norm(x):
    return np.linalg.norm(x, axis=-1)


def main():
    debris = pd.read_csv(DEBRIS_CSV)
    geom = pd.read_csv(GEOM_CSV)
    matched = pd.read_csv(MATCHED_CSV)

    # Q95 = primary cluster definition used so far.
    geom = geom[
        np.isclose(geom["quantile"], 0.95)
    ].copy()

    vel_match = matched[
        np.isclose(matched["quantile"], 0.95)
        & (matched["metric"] == "velocity")
    ][
        [
            "key",
            "object_id",
            "timestep",
            "matched_error_gap_m",
        ]
    ].copy()

    acc_match = matched[
        np.isclose(matched["quantile"], 0.95)
        & (matched["metric"] == "acceleration")
    ][
        [
            "key",
            "object_id",
            "timestep",
            "matched_error_gap_m",
        ]
    ].copy()

    vel_match = vel_match.rename(
        columns={
            "matched_error_gap_m":
                "matched_relvel_error_gap_m"
        }
    )

    acc_match = acc_match.rename(
        columns={
            "matched_error_gap_m":
                "matched_relacc_error_gap_m"
        }
    )

    key_to_path = {}

    for p in TRAJ_DIR.glob("*.npz"):
        with np.load(p, allow_pickle=False) as z:
            key = str(z["sample_key"])

        if key in key_to_path:
            raise RuntimeError(
                f"duplicate trajectory key: {key}"
            )

        key_to_path[key] = p

    assert len(key_to_path) == 578

    groups = debris.groupby(
        ["key", "object_id"],
        sort=True,
    )

    rows = []
    processed = 0

    for (key, object_id), g in groups:
        key = str(key)
        object_id = int(object_id)

        g = g.sort_values(
            "global_point_index"
        )

        idx = g[
            "global_point_index"
        ].to_numpy(dtype=np.int64)

        with np.load(
            key_to_path[key],
            allow_pickle=False,
        ) as z:
            gt_all = z["gt_scene_flows"]
            ids = z["scene_object_id"]
            valid_all = z["official_valid_mask"]

            expected = np.where(
                (ids[0] == object_id)
                & valid_all[-1]
            )[0]

            if not np.array_equal(
                idx,
                expected,
            ):
                raise RuntimeError(
                    f"point population mismatch: "
                    f"{key} object={object_id}"
                )

            gt = gt_all[:, idx].astype(
                np.float64
            )
            valid = valid_all[:, idx].astype(
                bool
            )

        T = gt.shape[0]
        assert T == 11

        for t in range(1, T):
            # ----------------------------------
            # GT point speed: t-1 -> t
            # ----------------------------------
            pair_valid = (
                valid[t - 1]
                & valid[t]
            )

            speed = vec_norm(
                gt[t] - gt[t - 1]
            )

            speed_valid = speed[pair_valid]

            if len(speed_valid):
                speed_mean = float(
                    np.mean(speed_valid)
                )
                speed_median = float(
                    np.median(speed_valid)
                )
            else:
                speed_mean = np.nan
                speed_median = np.nan

            # ----------------------------------
            # GT acceleration + direction change
            # ----------------------------------
            if t >= 2:
                triple_valid = (
                    valid[t - 2]
                    & valid[t - 1]
                    & valid[t]
                )

                v_prev = (
                    gt[t - 1]
                    - gt[t - 2]
                )

                v_cur = (
                    gt[t]
                    - gt[t - 1]
                )

                acc = v_cur - v_prev
                acc_mag = vec_norm(acc)

                acc_valid = acc_mag[
                    triple_valid
                ]

                if len(acc_valid):
                    acc_mean = float(
                        np.mean(acc_valid)
                    )
                    acc_median = float(
                        np.median(acc_valid)
                    )
                else:
                    acc_mean = np.nan
                    acc_median = np.nan

                prev_mag = vec_norm(v_prev)
                cur_mag = vec_norm(v_cur)

                dir_valid = (
                    triple_valid
                    & (prev_mag > EPS)
                    & (cur_mag > EPS)
                )

                if np.any(dir_valid):
                    a = v_prev[dir_valid]
                    b = v_cur[dir_valid]

                    denom = (
                        vec_norm(a)
                        * vec_norm(b)
                    )

                    cosine = (
                        np.sum(a * b, axis=-1)
                        / denom
                    )

                    cosine = np.clip(
                        cosine,
                        -1.0,
                        1.0,
                    )

                    angles = np.degrees(
                        np.arccos(cosine)
                    )

                    dir_mean_deg = float(
                        np.mean(angles)
                    )
                    dir_median_deg = float(
                        np.median(angles)
                    )
                else:
                    dir_mean_deg = np.nan
                    dir_median_deg = np.nan
            else:
                acc_mean = np.nan
                acc_median = np.nan
                dir_mean_deg = np.nan
                dir_median_deg = np.nan

            rows.append({
                "key": key,
                "object_id": object_id,
                "timestep": t,

                "gt_speed_valid_n":
                    int(pair_valid.sum()),
                "gt_speed_mean_m":
                    speed_mean,
                "gt_speed_median_m":
                    speed_median,

                "gt_acc_mean_m":
                    acc_mean,
                "gt_acc_median_m":
                    acc_median,

                "gt_direction_change_mean_deg":
                    dir_mean_deg,
                "gt_direction_change_median_deg":
                    dir_median_deg,
            })

        processed += 1

        if processed % 50 == 0:
            print(
                "processed_objects =",
                processed,
            )

    regime = pd.DataFrame(rows)

    assert processed == 377
    assert len(regime) == 377 * 10

    # Q95 geometry outcome.
    geom_keep = geom[
        [
            "key",
            "object_id",
            "timestep",
            "geometry_gap_m",
            "boundary_geometry_delta_median_m",
            "boundary_stretched_fraction",
        ]
    ].copy()

    out = regime.merge(
        geom_keep,
        on=["key", "object_id", "timestep"],
        how="left",
        validate="one_to_one",
    )

    out = out.merge(
        vel_match,
        on=["key", "object_id", "timestep"],
        how="left",
        validate="one_to_one",
    )

    out = out.merge(
        acc_match,
        on=["key", "object_id", "timestep"],
        how="left",
        validate="one_to_one",
    )

    assert len(out) == 3770

    out_dir = (
        OUT_ROOT
        / (
            "behavior_gt_motion_regime_full578_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    out.to_csv(
        out_dir / "per_object_timestep_motion_regime.csv",
        index=False,
    )

    print()
    print("=" * 72)
    print("GT MOTION REGIME BUILD COMPLETE")
    print("processed_objects =", processed)
    print("output_rows =", len(out))
    print(
        "finite GT speed =",
        int(
            np.isfinite(
                out["gt_speed_median_m"]
            ).sum()
        ),
    )
    print(
        "finite GT acceleration =",
        int(
            np.isfinite(
                out["gt_acc_median_m"]
            ).sum()
        ),
    )
    print(
        "finite GT direction change =",
        int(
            np.isfinite(
                out[
                    "gt_direction_change_median_deg"
                ]
            ).sum()
        ),
    )
    print(
        "finite geometry gap =",
        int(
            np.isfinite(
                out["geometry_gap_m"]
            ).sum()
        ),
    )
    print(
        "finite matched velocity gap =",
        int(
            np.isfinite(
                out[
                    "matched_relvel_error_gap_m"
                ]
            ).sum()
        ),
    )
    print("output_dir =", out_dir)
    print("SUCCESS = True")
    print("=" * 72)


if __name__ == "__main__":
    main()
