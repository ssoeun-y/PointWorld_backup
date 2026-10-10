#!/usr/bin/env python3

import numpy as np
import pandas as pd
from pathlib import Path

from scipy.stats import mannwhitneyu


IN_CSV = Path(
    "diagnostics/results/"
    "natural_action_local_structure_hq_lq.csv"
)

OUT_SUMMARY = Path(
    "diagnostics/results/"
    "natural_action_local_structure_hq_lq_summary.csv"
)

OUT_MATCHED = Path(
    "diagnostics/results/"
    "natural_action_local_structure_hq_lq_matched_pairs.csv"
)

SEED = 42
N_BOOT = 5000

METRICS = [
    "local_struct_mean_mm",
    "local_struct_median_mm",
    "local_struct_p95_mm",
]

MATCH_COLS = [
    "gt_motion_mm",
    "path_length_m",
    "net_displacement_m",
]


def bootstrap_median_diff(hq, lq, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)

    hq = np.asarray(hq, dtype=float)
    lq = np.asarray(lq, dtype=float)

    hq = hq[np.isfinite(hq)]
    lq = lq[np.isfinite(lq)]

    diffs = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        hs = rng.choice(hq, size=len(hq), replace=True)
        ls = rng.choice(lq, size=len(lq), replace=True)
        diffs[i] = np.median(ls) - np.median(hs)

    return (
        float(np.median(diffs)),
        float(np.quantile(diffs, 0.025)),
        float(np.quantile(diffs, 0.975)),
    )


def robust_standardize(x):
    x = np.asarray(x, dtype=float)

    med = np.nanmedian(x)
    q25, q75 = np.nanquantile(x, [0.25, 0.75])
    iqr = q75 - q25

    if iqr <= 0:
        iqr = np.nanstd(x)

    if not np.isfinite(iqr) or iqr <= 0:
        iqr = 1.0

    return (x - med) / iqr


def match_hq_lq(task_df):
    """
    1:1 nearest-neighbor matching within task.

    Matching variables:
      GT object motion
      robot path length
      robot net displacement

    Matching is outcome-blind:
      local structure metrics are NOT used.
    """

    hq = task_df[
        task_df["quality_group"] == "HQ"
    ].copy()

    lq = task_df[
        task_df["quality_group"] == "LQ"
    ].copy()

    hq = hq.dropna(subset=MATCH_COLS)
    lq = lq.dropna(subset=MATCH_COLS)

    if len(hq) == 0 or len(lq) == 0:
        return pd.DataFrame()

    both = pd.concat([hq[MATCH_COLS], lq[MATCH_COLS]])

    standardized = {}

    for c in MATCH_COLS:
        z = robust_standardize(both[c].values)
        standardized[c] = z

    Z = pd.DataFrame(
        standardized,
        index=both.index,
    )

    X_hq = Z.loc[hq.index, MATCH_COLS].values
    X_lq = Z.loc[lq.index, MATCH_COLS].values

    # Greedy 1:1 nearest-neighbor matching without sklearn.
    # Pairwise distance is only over the 3 matching covariates.
    dist_matrix = np.linalg.norm(
        X_lq[:, None, :] - X_hq[None, :, :],
        axis=2,
    )

    nearest_order = np.argsort(dist_matrix, axis=1)

    used_hq = set()
    pairs = []

    order = np.argsort(
        dist_matrix[np.arange(len(X_lq)), nearest_order[:, 0]]
    )

    for li in order:
        chosen = None

        for hi in nearest_order[li]:
            hi = int(hi)

            if hi not in used_hq:
                chosen = hi
                dist = float(dist_matrix[li, hi])
                break

        if chosen is None:
            continue

        used_hq.add(chosen)

        hrow = hq.iloc[chosen]
        lrow = lq.iloc[li]

        row = {
            "task": str(lrow["task"]),
            "hq_clip": hrow["clip"],
            "lq_clip": lrow["clip"],
            "match_distance": dist,
        }

        for c in MATCH_COLS:
            row[f"hq_{c}"] = float(hrow[c])
            row[f"lq_{c}"] = float(lrow[c])
            row[f"delta_{c}"] = (
                float(lrow[c]) - float(hrow[c])
            )

        for m in METRICS:
            row[f"hq_{m}"] = float(hrow[m])
            row[f"lq_{m}"] = float(lrow[m])
            row[f"delta_{m}"] = (
                float(lrow[m]) - float(hrow[m])
            )

        pairs.append(row)

    return pd.DataFrame(pairs)


def summarize_raw(df, scope, task_name):
    rows = []

    for metric in METRICS:
        d = df.dropna(subset=[metric])

        hq = d.loc[
            d.quality_group == "HQ",
            metric,
        ].values

        lq = d.loc[
            d.quality_group == "LQ",
            metric,
        ].values

        if len(hq) == 0 or len(lq) == 0:
            continue

        stat, p = mannwhitneyu(
            lq,
            hq,
            alternative="two-sided",
        )

        boot_med, ci_lo, ci_hi = bootstrap_median_diff(
            hq,
            lq,
        )

        rows.append(
            {
                "analysis": "raw",
                "scope": scope,
                "task": task_name,
                "metric": metric,
                "hq_n": len(hq),
                "lq_n": len(lq),
                "hq_median": float(np.median(hq)),
                "lq_median": float(np.median(lq)),
                "median_delta_lq_minus_hq":
                    float(np.median(lq) - np.median(hq)),
                "ratio_lq_over_hq":
                    float(np.median(lq) / np.median(hq))
                    if np.median(hq) != 0
                    else np.nan,
                "mannwhitney_p": float(p),
                "bootstrap_delta_median": boot_med,
                "bootstrap_ci95_low": ci_lo,
                "bootstrap_ci95_high": ci_hi,
            }
        )

    return rows


def summarize_matched(pairs, scope, task_name):
    rows = []

    if len(pairs) == 0:
        return rows

    for metric in METRICS:
        delta = pairs[
            f"delta_{metric}"
        ].dropna().values

        hq = pairs[
            f"hq_{metric}"
        ].dropna().values

        lq = pairs[
            f"lq_{metric}"
        ].dropna().values

        if len(delta) == 0:
            continue

        rng = np.random.default_rng(SEED)

        boot = np.empty(N_BOOT, dtype=float)

        for i in range(N_BOOT):
            s = rng.choice(
                delta,
                size=len(delta),
                replace=True,
            )
            boot[i] = np.median(s)

        rows.append(
            {
                "analysis": "matched",
                "scope": scope,
                "task": task_name,
                "metric": metric,
                "hq_n": len(hq),
                "lq_n": len(lq),
                "hq_median": float(np.median(hq)),
                "lq_median": float(np.median(lq)),
                "median_delta_lq_minus_hq":
                    float(np.median(delta)),
                "ratio_lq_over_hq":
                    float(np.median(lq) / np.median(hq))
                    if np.median(hq) != 0
                    else np.nan,
                "mannwhitney_p": np.nan,
                "bootstrap_delta_median":
                    float(np.median(boot)),
                "bootstrap_ci95_low":
                    float(np.quantile(boot, 0.025)),
                "bootstrap_ci95_high":
                    float(np.quantile(boot, 0.975)),
            }
        )

    return rows


def main():
    df = pd.read_csv(IN_CSV)

    required = (
        ["clip", "task", "quality_group"]
        + METRICS
        + MATCH_COLS
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"missing columns: {missing}"
        )

    df = df[
        df["quality_group"].isin(["HQ", "LQ"])
    ].copy()

    print("=" * 80)
    print("INPUT")
    print("=" * 80)
    print("N =", len(df))
    print(df.groupby(["task", "quality_group"]).size())

    summary = []

    # ------------------------------------------------------------
    # Raw overall
    # ------------------------------------------------------------
    summary.extend(
        summarize_raw(
            df,
            scope="overall",
            task_name="ALL",
        )
    )

    # ------------------------------------------------------------
    # Raw taskwise
    # ------------------------------------------------------------
    for task in sorted(df.task.unique()):
        summary.extend(
            summarize_raw(
                df[df.task == task],
                scope="taskwise",
                task_name=task,
            )
        )

    # ------------------------------------------------------------
    # Outcome-blind within-task matching
    # ------------------------------------------------------------
    matched_all = []

    for task in sorted(df.task.unique()):
        pairs = match_hq_lq(
            df[df.task == task]
        )

        print(
            f"[match] {task}: N={len(pairs)}"
        )

        if len(pairs):
            matched_all.append(pairs)

            summary.extend(
                summarize_matched(
                    pairs,
                    scope="taskwise",
                    task_name=task,
                )
            )

    if not matched_all:
        raise RuntimeError("no matched pairs")

    matched = pd.concat(
        matched_all,
        ignore_index=True,
    )

    summary.extend(
        summarize_matched(
            matched,
            scope="overall",
            task_name="ALL",
        )
    )

    summary_df = pd.DataFrame(summary)

    OUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    matched.to_csv(
        OUT_MATCHED,
        index=False,
    )

    # ------------------------------------------------------------
    # Print only the important result
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RAW HQ vs LQ")
    print("=" * 80)

    raw = summary_df[
        (summary_df.analysis == "raw")
        & (summary_df.scope == "overall")
    ]

    print(
        raw[
            [
                "metric",
                "hq_n",
                "lq_n",
                "hq_median",
                "lq_median",
                "median_delta_lq_minus_hq",
                "ratio_lq_over_hq",
                "mannwhitney_p",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("MATCHED HQ vs LQ")
    print("=" * 80)

    mat = summary_df[
        (summary_df.analysis == "matched")
        & (summary_df.scope == "overall")
    ]

    print(
        mat[
            [
                "metric",
                "hq_n",
                "lq_n",
                "hq_median",
                "lq_median",
                "median_delta_lq_minus_hq",
                "ratio_lq_over_hq",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("MATCH QUALITY")
    print("=" * 80)

    print(
        "pairs =",
        len(matched),
    )

    for c in MATCH_COLS:
        x = matched[f"delta_{c}"]

        print(
            c,
            "median_delta=",
            float(np.median(x)),
            "median_abs_delta=",
            float(np.median(np.abs(x))),
        )

    print("\nsaved =", OUT_SUMMARY)
    print("saved =", OUT_MATCHED)
    print("SUCCESS = True")


if __name__ == "__main__":
    main()
