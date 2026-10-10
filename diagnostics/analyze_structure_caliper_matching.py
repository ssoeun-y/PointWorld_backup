#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

IN = Path(
    "diagnostics/results/"
    "natural_action_local_structure_hq_lq.csv"
)

OUT_SUM = Path(
    "diagnostics/results/"
    "natural_action_structure_caliper_summary.csv"
)

SEED = 42
N_BOOT = 5000

MATCH_COLS = [
    "gt_motion_mm",
    "path_length_m",
    "net_displacement_m",
]

METRICS = [
    "local_struct_mean_mm",
    "local_struct_median_mm",
    "local_struct_p95_mm",
]


def add_task_percentiles(df):
    """
    Convert each matching covariate to within-task percentile rank.
    This puts GT motion/path/net displacement on the same 0~1 scale.
    """
    out = df.copy()

    for c in MATCH_COLS:
        out[c + "_pct"] = (
            out.groupby("task")[c]
            .rank(method="average", pct=True)
        )

    return out


def greedy_caliper_match(d, caliper):
    """
    Outcome-blind 1:1 matching.

    A pair is eligible only when ALL three covariate percentile
    differences are <= caliper.

    Among eligible pairs, greedily take the smallest Euclidean
    distance in percentile space without replacement.
    """
    h = d[d.quality_group == "HQ"].copy().reset_index(drop=True)
    l = d[d.quality_group == "LQ"].copy().reset_index(drop=True)

    pct_cols = [c + "_pct" for c in MATCH_COLS]

    h = h.dropna(subset=pct_cols)
    l = l.dropna(subset=pct_cols)

    H = h[pct_cols].to_numpy(float)
    L = l[pct_cols].to_numpy(float)

    candidates = []

    for li in range(len(l)):
        diffs = np.abs(H - L[li])

        eligible = np.all(diffs <= caliper, axis=1)

        for hi in np.where(eligible)[0]:
            dist = float(np.linalg.norm(H[hi] - L[li]))
            candidates.append((dist, li, int(hi)))

    candidates.sort(key=lambda x: x[0])

    used_h = set()
    used_l = set()
    pairs = []

    for dist, li, hi in candidates:
        if li in used_l or hi in used_h:
            continue

        used_l.add(li)
        used_h.add(hi)

        hr = h.iloc[hi]
        lr = l.iloc[li]

        row = {
            "task": lr["task"],
            "hq_clip": hr["clip"],
            "lq_clip": lr["clip"],
            "match_distance": dist,
        }

        for c in MATCH_COLS:
            row["hq_" + c] = hr[c]
            row["lq_" + c] = lr[c]
            row["delta_" + c] = lr[c] - hr[c]

            row["hq_" + c + "_pct"] = hr[c + "_pct"]
            row["lq_" + c + "_pct"] = lr[c + "_pct"]
            row["abs_pct_diff_" + c] = abs(
                lr[c + "_pct"] - hr[c + "_pct"]
            )

        # Metrics are NOT used for matching.
        for m in METRICS:
            row["hq_" + m] = hr[m]
            row["lq_" + m] = lr[m]

            if pd.notna(hr[m]) and pd.notna(lr[m]):
                row["delta_" + m] = lr[m] - hr[m]
            else:
                row["delta_" + m] = np.nan

        pairs.append(row)

    return pd.DataFrame(pairs)


def bootstrap_paired_median(delta):
    delta = np.asarray(delta, float)
    delta = delta[np.isfinite(delta)]

    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT)

    for i in range(N_BOOT):
        x = rng.choice(delta, len(delta), replace=True)
        boots[i] = np.median(x)

    return (
        float(np.median(delta)),
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )


def summarize_pairs(pairs, caliper):
    rows = []

    for scope, d in [
        ("ALL", pairs),
        *[
            (task, pairs[pairs.task == task])
            for task in sorted(pairs.task.unique())
        ],
    ]:
        for m in METRICS:
            valid = d[
                d["hq_" + m].notna()
                & d["lq_" + m].notna()
            ].copy()

            if len(valid) == 0:
                continue

            h = valid["hq_" + m].to_numpy(float)
            l = valid["lq_" + m].to_numpy(float)
            delta = l - h

            med_delta, lo, hi = bootstrap_paired_median(delta)

            try:
                _, p = wilcoxon(
                    delta,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
            except ValueError:
                p = np.nan

            rows.append({
                "caliper": caliper,
                "scope": scope,
                "metric": m,
                "matched_valid_n": len(valid),
                "hq_median": float(np.median(h)),
                "lq_median": float(np.median(l)),
                "paired_median_delta": med_delta,
                "lq_over_hq_ratio":
                    float(np.median(l) / np.median(h))
                    if np.median(h) != 0 else np.nan,
                "lq_worse_fraction":
                    float(np.mean(delta > 0)),
                "wilcoxon_p": float(p),
                "bootstrap_ci95_low": lo,
                "bootstrap_ci95_high": hi,
            })

    return rows


def main():
    df = pd.read_csv(IN)

    df = df[
        df.quality_group.isin(["HQ", "LQ"])
    ].copy()

    df = add_task_percentiles(df)

    all_summary = []

    for caliper in [0.10, 0.05]:

        matched_parts = []

        for task in sorted(df.task.unique()):
            d = df[df.task == task]

            p = greedy_caliper_match(d, caliper)

            if len(p):
                matched_parts.append(p)

            print(
                f"caliper={caliper:.2f} "
                f"{task} pairs={len(p)}"
            )

        matched = pd.concat(
            matched_parts,
            ignore_index=True,
        )

        out_pairs = Path(
            "diagnostics/results/"
            f"natural_action_structure_"
            f"caliper_{int(caliper*100):02d}_pairs.csv"
        )
        matched.to_csv(out_pairs, index=False)

        print(
            f"\nCALIPER {int(caliper*100)}% "
            f"TOTAL PAIRS = {len(matched)}"
        )

        print("MATCH QUALITY")
        for c in MATCH_COLS:
            x = matched["delta_" + c].to_numpy(float)
            pct = matched["abs_pct_diff_" + c].to_numpy(float)

            print(
                f"{c:20s} "
                f"median_abs_raw={np.median(np.abs(x)):.6f} | "
                f"median_abs_pct={np.median(pct):.4f} | "
                f"max_abs_pct={np.max(pct):.4f}"
            )

        summary = summarize_pairs(matched, caliper)
        all_summary.extend(summary)

        S = pd.DataFrame(summary)

        print("\nOVERALL STRUCTURE RESULT")
        show = S[S.scope == "ALL"][
            [
                "metric",
                "matched_valid_n",
                "hq_median",
                "lq_median",
                "paired_median_delta",
                "lq_over_hq_ratio",
                "lq_worse_fraction",
                "wilcoxon_p",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
            ]
        ]

        print(show.to_string(index=False))
        print("-" * 100)

    result = pd.DataFrame(all_summary)
    result.to_csv(OUT_SUM, index=False)

    print("\nsaved =", OUT_SUM)
    print("SUCCESS = True")


if __name__ == "__main__":
    main()
