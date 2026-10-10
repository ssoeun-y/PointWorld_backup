#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd


TEMPORAL_PATH = Path(
    "diagnostics/results/"
    "behavior_cluster_temporal_order_full578_20260811/"
    "per_object_temporal_order.csv"
)

MATCHED_PATH = Path(
    "diagnostics/results/"
    "behavior_cluster_gt_matched_control_full578_20260811_034030/"
    "matched_control.csv"
)

OUT_DIR = Path(
    "diagnostics/results/"
    "behavior_cluster_preonset_spike_full578_20260811"
)


def make_key(key, object_id, q, metric):
    return (
        str(key),
        int(object_id),
        round(float(q), 3),
        str(metric),
    )


def main():
    temporal = pd.read_csv(TEMPORAL_PATH)
    matched = pd.read_csv(MATCHED_PATH)

    assert len(temporal) == 6786
    assert len(matched) == 19227

    # Build one dynamics time series per
    # object / quantile / metric.
    series_map = {}

    grouped = matched.groupby(
        ["key", "object_id", "quantile", "metric"],
        sort=False,
    )

    for (key, object_id, q, metric), g in grouped:
        g = g.sort_values("timestep")

        k = make_key(
            key,
            object_id,
            q,
            metric,
        )

        if k in series_map:
            raise RuntimeError(
                f"duplicate series: {k}"
            )

        series_map[k] = g[
            ["timestep", "matched_error_gap_m"]
        ].copy()

    rows = []

    for _, r in temporal.iterrows():
        k = make_key(
            r["key"],
            r["object_id"],
            r["quantile"],
            r["metric"],
        )

        if k not in series_map:
            raise RuntimeError(
                f"missing matched series: {k}"
            )

        s = series_map[k].copy()

        s = s[
            np.isfinite(
                s["matched_error_gap_m"]
            )
        ]

        onset_t = r["geometry_onset_t"]
        prior_t = r["strict_prior_t"]
        prior_gap = r[
            "strict_prior_error_gap_m"
        ]

        out = dict(r)

        # Default: unavailable.
        out.update({
            "all_other_n": 0,
            "all_other_median_gap_m": np.nan,
            "prior_minus_all_other_median_m": np.nan,

            "pre_onset_baseline_n": 0,
            "pre_onset_baseline_median_gap_m": np.nan,
            "prior_minus_pre_onset_median_m": np.nan,

            "prior_percentile_among_object_timesteps": np.nan,
        })

        if not (
            np.isfinite(onset_t)
            and np.isfinite(prior_t)
            and np.isfinite(prior_gap)
        ):
            rows.append(out)
            continue

        onset_t = int(onset_t)
        prior_t = int(prior_t)
        prior_gap = float(prior_gap)

        # Verify temporal CSV and matched CSV
        # refer to exactly the same prior value.
        prior_match = s[
            s["timestep"] == prior_t
        ]

        if len(prior_match) != 1:
            raise RuntimeError(
                f"prior timestep mismatch: {k}, "
                f"prior_t={prior_t}"
            )

        prior_from_series = float(
            prior_match.iloc[0][
                "matched_error_gap_m"
            ]
        )

        if not np.isclose(
            prior_gap,
            prior_from_series,
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"prior gap mismatch: {k}"
            )

        # -------------------------------------------------
        # Control A:
        # Compare t-1 against ALL other valid timesteps,
        # excluding t-1 itself and the geometry onset t.
        # -------------------------------------------------
        other = s[
            (s["timestep"] != prior_t)
            & (s["timestep"] != onset_t)
        ][
            "matched_error_gap_m"
        ].to_numpy(dtype=float)

        if len(other) > 0:
            other_med = float(
                np.median(other)
            )

            out[
                "all_other_n"
            ] = int(len(other))

            out[
                "all_other_median_gap_m"
            ] = other_med

            out[
                "prior_minus_all_other_median_m"
            ] = (
                prior_gap - other_med
            )

        # -------------------------------------------------
        # Control B:
        # Stricter temporal baseline:
        # only timesteps STRICTLY EARLIER than t-1.
        # This asks whether error rises immediately before
        # geometry onset relative to its own earlier history.
        # -------------------------------------------------
        pre = s[
            s["timestep"] < prior_t
        ][
            "matched_error_gap_m"
        ].to_numpy(dtype=float)

        if len(pre) > 0:
            pre_med = float(
                np.median(pre)
            )

            out[
                "pre_onset_baseline_n"
            ] = int(len(pre))

            out[
                "pre_onset_baseline_median_gap_m"
            ] = pre_med

            out[
                "prior_minus_pre_onset_median_m"
            ] = (
                prior_gap - pre_med
            )

        # Rank of t-1 error within the object's
        # complete valid dynamics time series.
        all_vals = s[
            "matched_error_gap_m"
        ].to_numpy(dtype=float)

        out[
            "prior_percentile_among_object_timesteps"
        ] = float(
            np.mean(
                all_vals <= prior_gap
            )
        )

        rows.append(out)

    result = pd.DataFrame(rows)

    assert len(result) == len(temporal)

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUT_DIR / "per_object_preonset_spike.csv",
        index=False,
    )

    print("=" * 72)
    print("PRE-ONSET SPIKE BUILD COMPLETE")
    print("temporal rows =", len(temporal))
    print("matched rows =", len(matched))
    print("output rows =", len(result))
    print(
        "all-other available =",
        int(
            np.isfinite(
                result[
                    "prior_minus_all_other_median_m"
                ]
            ).sum()
        ),
    )
    print(
        "strict pre-onset available =",
        int(
            np.isfinite(
                result[
                    "prior_minus_pre_onset_median_m"
                ]
            ).sum()
        ),
    )
    print("output_dir =", OUT_DIR)
    print("SUCCESS = True")
    print("=" * 72)


if __name__ == "__main__":
    main()
