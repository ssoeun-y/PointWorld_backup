#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


SEED = 42
BOOT = 5000

STRUCT_FILES = {
    "task0000": (
        "diagnostics/results/"
        "object_structure_full_task0000_result.csv",
        56,
    ),
    "task0001": (
        "diagnostics/results/"
        "object_structure_full_task0001_result.csv",
        429,
    ),
    "multi": (
        "diagnostics/results/"
        "object_structure_full_multi_result.csv",
        1092,
    ),
}

PAIR_FILES = {
    "10pct": (
        "diagnostics/results/"
        "temporal_quality_5task_matched_10pct_pairs.csv"
    ),
    "5pct": (
        "diagnostics/results/"
        "temporal_quality_5task_matched_5pct_pairs.csv"
    ),
}

# GT-rigidity sensitivity.
RIGID_THRESHOLDS_MM = [
    0.5,
    1.0,
    2.0,
]

# Object-structure outcomes.
METRICS = [
    "pred_se3_mean_mm",
    "pred_se3_p95_mm",
    "pred_pair_p95_mm",

    # GT baseline removed.
    "excess_se3_mean_mm",
    "excess_pair_p95_mm",
]

# These two are the most interpretable main diagnostics.
MAIN_METRICS = [
    "excess_se3_mean_mm",
    "excess_pair_p95_mm",
]


def bootstrap_median_ci(delta):
    x = np.asarray(delta, dtype=float)
    x = x[np.isfinite(x)]

    rng = np.random.default_rng(SEED)
    n = len(x)

    vals = np.empty(BOOT)

    for b in range(BOOT):
        idx = rng.integers(0, n, size=n)
        vals[b] = np.median(x[idx])

    return (
        float(np.quantile(vals, 0.025)),
        float(np.quantile(vals, 0.975)),
    )


def summarize_pair_metric(q, metric):

    h = q[f"hq_{metric}"].to_numpy(float)
    l = q[f"lq_{metric}"].to_numpy(float)

    valid = np.isfinite(h) & np.isfinite(l)

    h = h[valid]
    l = l[valid]

    delta = l - h

    lo, hi = bootstrap_median_ci(delta)

    try:
        w = wilcoxon(
            l,
            h,
            alternative="two-sided",
            zero_method="wilcox",
        )
        pval = float(w.pvalue)
    except Exception:
        pval = np.nan

    hm = float(np.median(h))
    lm = float(np.median(l))

    return {
        "N_pairs": len(h),
        "HQ_median": hm,
        "LQ_median": lm,
        "LQ_over_HQ":
            lm / hm if hm != 0 else np.nan,
        "paired_median_delta":
            float(np.median(delta)),
        "LQ_worse_fraction":
            float(np.mean(delta > 0)),
        "bootstrap95_low": lo,
        "bootstrap95_high": hi,
        "wilcoxon_p": pval,
    }


# ============================================================
# 1. LOAD THE THREE COMPLETED STRUCTURE RUNS
# ============================================================

print("=" * 78)
print("1. LOAD OBJECT-STRUCTURE RESULTS")
print("=" * 78)

parts = []

for tag, (path, expected) in STRUCT_FILES.items():

    p = Path(path)

    if not p.exists():
        print("MISSING:", path)
        print(
            "\nSTOP safely: run the missing object evaluation first."
        )
        sys.exit(2)

    x = pd.read_csv(p)

    print(
        tag,
        "rows =", len(x),
        "expected =", expected,
    )

    if len(x) != expected:
        print(
            "BAD ROW COUNT:",
            tag,
            len(x),
            expected,
        )
        sys.exit(3)

    if not x["clip"].is_unique:
        print("DUPLICATE CLIPS:", tag)
        sys.exit(4)

    parts.append(x)


obj = pd.concat(
    parts,
    ignore_index=True,
)

print("\nTOTAL =", len(obj))
print("UNIQUE =", obj["clip"].nunique())

if len(obj) != 1577:
    print("EXPECTED 1577")
    sys.exit(5)

if obj["clip"].nunique() != 1577:
    print("EXPECTED 1577 UNIQUE")
    sys.exit(6)


print("\nSTATUS:")
print(obj["status"].value_counts(dropna=False))

print("\nSTATUS BY TASK:")
print(
    pd.crosstab(
        obj["task"],
        obj["status"],
    )
)

obj.to_csv(
    "diagnostics/results/"
    "object_structure_full_5task_1577.csv",
    index=False,
)

print(
    "\nCOMBINED_OBJECT_RESULT_OK=True"
)


# ============================================================
# 2. CREATE HQ / LQ LOOKUPS
# ============================================================

keep_cols = [
    "clip",
    "task",
    "status",
    "object_id",
    "object_name",
    "moved_valid_count",
    "object_points_t0",
    "n_final_valid",

    "gt_se3_mean_mm",
    "gt_se3_p95_mm",
    "gt_pair_p95_mm",

    *METRICS,
]

for c in keep_cols:
    if c not in obj.columns:
        print("MISSING COLUMN:", c)
        sys.exit(7)


H = obj[keep_cols].rename(
    columns={
        c: f"hq_{c}"
        for c in keep_cols
        if c not in ["clip"]
    }
).rename(
    columns={"clip": "hq_clip"}
)

L = obj[keep_cols].rename(
    columns={
        c: f"lq_{c}"
        for c in keep_cols
        if c not in ["clip"]
    }
).rename(
    columns={"clip": "lq_clip"}
)


summary_rows = []
task_rows = []


# ============================================================
# 3. MATCHED HQ / LQ OBJECT ANALYSIS
# ============================================================

for caliper, pair_path in PAIR_FILES.items():

    print("\n" + "#" * 78)
    print("MATCHING =", caliper)
    print("#" * 78)

    pairs = pd.read_csv(pair_path)

    q = pairs.merge(
        H,
        on="hq_clip",
        how="left",
        validate="one_to_one",
    )

    q = q.merge(
        L,
        on="lq_clip",
        how="left",
        validate="one_to_one",
    )

    if len(q) != len(pairs):
        print("PAIR MERGE COUNT FAILURE")
        sys.exit(8)

    # Check object results exist.
    missing_h = q["hq_status"].isna()
    missing_l = q["lq_status"].isna()

    print("ORIGINAL PAIRS =", len(q))
    print("MISSING HQ OBJECT =", int(missing_h.sum()))
    print("MISSING LQ OBJECT =", int(missing_l.sum()))

    # Both sides must have enough object points.
    ok = q[
        (q["hq_status"] == "ok")
        & (q["lq_status"] == "ok")
    ].copy()

    print("BOTH OBJECTS OK =", len(ok))

    print("\n===== BEFORE RIGID FILTER: OBJECT SUPPORT =====")

    for c in [
        "n_final_valid",
        "object_points_t0",
        "moved_valid_count",
        "gt_se3_mean_mm",
        "gt_pair_p95_mm",
    ]:

        h = ok[f"hq_{c}"].to_numpy(float)
        l = ok[f"lq_{c}"].to_numpy(float)

        print(
            c,
            "| HQ median =",
            float(np.median(h)),
            "| LQ median =",
            float(np.median(l)),
            "| paired median delta =",
            float(np.median(l - h)),
        )

    # --------------------------------------------------------
    # GT-rigid sensitivity
    # BOTH members of pair must be rigid.
    # --------------------------------------------------------

    for threshold in RIGID_THRESHOLDS_MM:

        r = ok[
            (ok["hq_gt_se3_mean_mm"] <= threshold)
            & (ok["lq_gt_se3_mean_mm"] <= threshold)
        ].copy()

        print("\n" + "=" * 78)
        print(
            "CALIPER =", caliper,
            "| GT-RIGID SE3 MEAN <=",
            threshold,
            "mm"
        )
        print("=" * 78)

        print("RIGID PAIRS =", len(r))

        print("\nRIGID PAIRS BY TASK:")
        print(
            r["task"]
            .value_counts()
            .sort_index()
        )

        if len(r) < 10:
            print("TOO FEW PAIRS; SKIP")
            continue

        # --------------------------------------------
        # Overall structure results
        # --------------------------------------------

        for metric in METRICS:

            stat = summarize_pair_metric(
                r,
                metric,
            )

            row = {
                "caliper": caliper,
                "gt_rigid_threshold_mm":
                    threshold,
                "metric": metric,
                **stat,
            }

            summary_rows.append(row)

            print("\n", metric)

            for k, v in stat.items():
                print(k, "=", v)

        # --------------------------------------------
        # Taskwise direction for MAIN diagnostics
        # --------------------------------------------

        for metric in MAIN_METRICS:

            print(
                "\n===== TASKWISE:",
                metric,
                "====="
            )

            for task, z in r.groupby("task"):

                h = z[
                    f"hq_{metric}"
                ].to_numpy(float)

                l = z[
                    f"lq_{metric}"
                ].to_numpy(float)

                valid = (
                    np.isfinite(h)
                    & np.isfinite(l)
                )

                h = h[valid]
                l = l[valid]

                if len(h) == 0:
                    continue

                delta = l - h

                hm = float(
                    np.median(h)
                )

                lm = float(
                    np.median(l)
                )

                tr = {
                    "caliper": caliper,
                    "gt_rigid_threshold_mm":
                        threshold,
                    "metric": metric,
                    "task": task,
                    "N_pairs": len(h),
                    "HQ_median": hm,
                    "LQ_median": lm,
                    "ratio":
                        lm / hm
                        if hm != 0
                        else np.nan,
                    "paired_median_delta":
                        float(
                            np.median(delta)
                        ),
                    "LQ_worse_fraction":
                        float(
                            np.mean(delta > 0)
                        ),
                }

                task_rows.append(tr)

                print(tr)


# ============================================================
# 4. SAVE
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

taskwise = pd.DataFrame(
    task_rows
)

summary_path = (
    "diagnostics/results/"
    "temporal_quality_object_structure_"
    "matched_summary.csv"
)

task_path = (
    "diagnostics/results/"
    "temporal_quality_object_structure_"
    "matched_taskwise.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

taskwise.to_csv(
    task_path,
    index=False,
)


print("\n" + "=" * 78)
print("FINAL OBJECT-LEVEL SUMMARY")
print("=" * 78)

main = summary[
    summary["metric"].isin(
        MAIN_METRICS
    )
]

print(
    main.to_string(index=False)
)

print("\nSAVED =", summary_path)
print("SAVED =", task_path)

print(
    "\nOBJECT_HQ_LQ_ANALYSIS_SUCCESS=True"
)
