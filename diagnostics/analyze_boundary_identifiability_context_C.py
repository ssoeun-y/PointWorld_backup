#!/usr/bin/env python3

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# ============================================================
# FIXED PROTOCOL -- DO NOT RETUNE AFTER LOOKING AT RESULTS
# ============================================================

SEED = 20260902
BOOTSTRAPS = 500

ROOT = Path(__file__).resolve().parents[1]

BEHAVIOR_CSV = (
    ROOT
    / "diagnostics/results/established_action_metrics_5task/"
      "per_sample_established_metrics_7987.csv"
)

DROID_CSV = (
    ROOT
    / "diagnostics/results/droid_established_dynamics_1679/"
      "timestamp_aware_si/droid_eef_timestamp_aware_si_1679.csv"
)

OUTDIR = (
    ROOT
    / "diagnostics/results/"
      "boundary_identifiability_context_C"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

# Same pre-specified boundary evidence rules used in BEHAVIOR.
BIC_LINEAR_MIN = 10.0
BIC_QUAD_MIN = 6.0


# ============================================================
# Utilities
# ============================================================

def stable_number(text):
    h = hashlib.sha256(
        f"{SEED}|{text}".encode()
    ).digest()

    return int.from_bytes(
        h[:8], "big"
    ) / 2**64


def exact_group_split(df, group_col):
    """
    Exact deterministic 70/30 split independently inside each
    task/site. Same seed and SHA256 rule already used.
    """
    result = pd.Series(
        index=df.index,
        dtype="object",
    )

    for group, g in df.groupby(group_col):
        idx = sorted(
            g.index,
            key=lambda i:
                stable_number(
                    str(df.at[i, "clip"])
                ),
        )

        n_cal = round(
            len(idx) * 0.70
        )

        result.loc[
            idx[:n_cal]
        ] = "calibration"

        result.loc[
            idx[n_cal:]
        ] = "validation"

    return result


def nuisance_dummies(s):
    return pd.get_dummies(
        s.astype(str),
        drop_first=True,
        dtype=float,
    ).to_numpy(float)


def ols_bic(y, X):
    beta, *_ = np.linalg.lstsq(
        X, y, rcond=None
    )

    residual = y - X @ beta

    sse = max(
        float(residual @ residual),
        1e-15,
    )

    n, k = X.shape

    bic = (
        n * np.log(sse / n)
        + k * np.log(n)
    )

    return beta, float(bic)


def cliffs_delta(x, y):
    """
    x = outside
    y = inside
    Positive => outside errors larger.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0 or len(y) == 0:
        return np.nan

    more = 0
    less = 0

    for i in range(0, len(x), 500):
        xx = x[i:i+500, None]
        more += np.sum(
            xx > y[None, :]
        )
        less += np.sum(
            xx < y[None, :]
        )

    return float(
        (more - less)
        / (len(x) * len(y))
    )


def fit_models(
    data,
    metric,
    error_col,
    nuisance_col=None,
):
    """
    Same functional form as BEHAVIOR working-range analysis:
      x = log10(dynamics)
      y = log1p(moved prediction error)

    Compare:
      linear
      quadratic
      one-breakpoint segmented

    Candidate breakpoint grid:
      q20 ... q90, 71 points
    """
    x_raw = (
        data[metric]
        .to_numpy(float)
    )

    y_raw = (
        data[error_col]
        .to_numpy(float)
    )

    valid = (
        np.isfinite(x_raw)
        & np.isfinite(y_raw)
    )

    x_raw = x_raw[valid]
    y_raw = y_raw[valid]

    if len(x_raw) < 10:
        return None

    positive = x_raw[x_raw > 0]

    if len(positive) == 0:
        return None

    floor = positive.min() * 0.5

    x = np.log10(
        np.maximum(
            x_raw,
            floor,
        )
    )

    y = np.log1p(y_raw)

    if nuisance_col is not None:
        nuisance_series = (
            data.loc[
                data.index[valid],
                nuisance_col
            ]
        )

        nuisance = nuisance_dummies(
            nuisance_series
        )
    else:
        nuisance = np.empty(
            (len(x), 0),
            dtype=float,
        )

    # Linear
    X_linear = np.column_stack([
        np.ones(len(x)),
        x,
        nuisance,
    ])

    _, bic_linear = ols_bic(
        y,
        X_linear,
    )

    # Quadratic
    xc = x - x.mean()

    X_quad = np.column_stack([
        np.ones(len(x)),
        x,
        xc ** 2,
        nuisance,
    ])

    _, bic_quad = ols_bic(
        y,
        X_quad,
    )

    # Segmented
    candidate_taus = np.unique(
        np.quantile(
            x,
            np.linspace(
                0.20,
                0.90,
                71,
            ),
        )
    )

    best = None

    for tau in candidate_taus:
        hinge = np.maximum(
            0.0,
            x - tau,
        )

        X_seg = np.column_stack([
            np.ones(len(x)),
            x,
            hinge,
            nuisance,
        ])

        beta, bic = ols_bic(
            y,
            X_seg,
        )

        if (
            best is None
            or bic < best["bic"]
        ):
            best = {
                "tau": float(tau),
                "beta": beta,
                "bic": float(bic),
            }

    if best is None:
        return None

    pre = float(
        best["beta"][1]
    )

    post = float(
        best["beta"][1]
        + best["beta"][2]
    )

    delta_linear = float(
        bic_linear - best["bic"]
    )

    delta_quad = float(
        bic_quad - best["bic"]
    )

    calibration_support = bool(
        delta_linear >= BIC_LINEAR_MIN
        and delta_quad >= BIC_QUAD_MIN
        and post > pre
    )

    return {
        "threshold":
            float(10 ** best["tau"]),

        "bic_linear":
            bic_linear,

        "bic_quadratic":
            bic_quad,

        "bic_segmented":
            best["bic"],

        "delta_bic_vs_linear":
            delta_linear,

        "delta_bic_vs_quadratic":
            delta_quad,

        "slope_pre":
            pre,

        "slope_post":
            post,

        "slope_change":
            float(post - pre),

        "calibration_support":
            calibration_support,

        "n_fit":
            int(len(x)),
    }


def validate_boundary(
    val,
    metric,
    error_col,
    threshold,
):
    x = val[metric].to_numpy(float)
    y = val[error_col].to_numpy(float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    inside = y[
        x <= threshold
    ]

    outside = y[
        x > threshold
    ]

    med_in = (
        float(np.median(inside))
        if len(inside)
        else np.nan
    )

    med_out = (
        float(np.median(outside))
        if len(outside)
        else np.nan
    )

    if (
        len(inside)
        and len(outside)
    ):
        p = float(
            mannwhitneyu(
                outside,
                inside,
                alternative="two-sided",
            ).pvalue
        )

        cliff = cliffs_delta(
            outside,
            inside,
        )

    else:
        p = np.nan
        cliff = np.nan

    direction = bool(
        np.isfinite(med_in)
        and np.isfinite(med_out)
        and med_out > med_in
        and np.isfinite(cliff)
        and cliff > 0
    )

    return {
        "val_inside_n":
            int(len(inside)),

        "val_outside_n":
            int(len(outside)),

        "val_med_inside_mm":
            med_in,

        "val_med_outside_mm":
            med_out,

        "val_out_in_ratio":
            (
                float(med_out / med_in)
                if (
                    np.isfinite(med_in)
                    and med_in > 0
                )
                else np.nan
            ),

        "val_cliff_delta":
            cliff,

        "val_mw_p":
            p,

        "validation_direction":
            direction,
    }


# ============================================================
# Stratified bootstrap
#
# IMPORTANT:
# Only calibration data are resampled.
# Validation is NEVER used in bootstrap fitting.
# ============================================================

def bootstrap_fit(
    cal,
    metric,
    error_col,
    group_col,
    nuisance_col,
    n_boot=BOOTSTRAPS,
):

    rng = np.random.default_rng(
        SEED
    )

    groups = {
        g:
            d.index.to_numpy()
        for g, d
        in cal.groupby(group_col)
    }

    rows = []

    for b in range(n_boot):

        sampled_parts = []

        for g, idx in groups.items():
            sampled = rng.choice(
                idx,
                size=len(idx),
                replace=True,
            )

            sampled_parts.append(
                cal.loc[sampled]
            )

        boot = pd.concat(
            sampled_parts,
            ignore_index=True,
        )

        fit = fit_models(
            boot,
            metric,
            error_col,
            nuisance_col=nuisance_col,
        )

        if fit is None:
            continue

        rows.append({
            "bootstrap":
                b,

            "threshold":
                fit["threshold"],

            "delta_bic_vs_linear":
                fit["delta_bic_vs_linear"],

            "delta_bic_vs_quadratic":
                fit["delta_bic_vs_quadratic"],

            "slope_pre":
                fit["slope_pre"],

            "slope_post":
                fit["slope_post"],

            "calibration_support":
                fit["calibration_support"],
        })

    return pd.DataFrame(rows)


def summarize_bootstrap(bdf):
    if len(bdf) == 0:
        return {
            "bootstrap_n": 0,
            "bootstrap_support_rate": np.nan,
            "threshold_boot_median": np.nan,
            "threshold_boot_p025": np.nan,
            "threshold_boot_p975": np.nan,
            "threshold_ci_width": np.nan,
            "threshold_relative_ci_width": np.nan,
        }

    q025, q50, q975 = np.quantile(
        bdf["threshold"].to_numpy(float),
        [0.025, 0.50, 0.975],
    )

    width = float(
        q975 - q025
    )

    relative_width = (
        float(width / q50)
        if q50 > 0
        else np.nan
    )

    return {
        "bootstrap_n":
            int(len(bdf)),

        "bootstrap_support_rate":
            float(
                bdf[
                    "calibration_support"
                ].mean()
            ),

        "threshold_boot_median":
            float(q50),

        "threshold_boot_p025":
            float(q025),

        "threshold_boot_p975":
            float(q975),

        "threshold_ci_width":
            width,

        "threshold_relative_ci_width":
            relative_width,
    }


# ============================================================
# Load data
# ============================================================

behavior = pd.read_csv(
    BEHAVIOR_CSV
)

droid = pd.read_csv(
    DROID_CSV
)

assert len(behavior) == 7987
assert len(droid) == 1679
assert droid["clip"].nunique() == 1679

# Error-column compatibility
if "l2_moved_mm" in behavior.columns:
    BEH_ERROR = "l2_moved_mm"
elif "filtered_l2_moved_mm" in behavior.columns:
    BEH_ERROR = "filtered_l2_moved_mm"
else:
    raise RuntimeError(
        "BEHAVIOR moved-L2 column not found"
    )

DROID_ERROR = "filtered_l2_moved_mm"

# Native BEHAVIOR established metrics.
BEH_METRICS = {
    "acc":
        "rms_acceleration",
    "msj":
        "mean_squared_jerk",
}

# Timestamp-aware SI DROID EEF metrics.
DROID_METRICS = {
    "acc":
        "rms_acceleration_si_m_s2",
    "msj":
        "mean_squared_jerk_si_m2_s6",
}


# ============================================================
# Exact frozen splits
# ============================================================

behavior["split"] = exact_group_split(
    behavior,
    "task",
)

droid["split"] = exact_group_split(
    droid,
    "site",
)

beh_cal = behavior[
    behavior["split"]
    == "calibration"
].copy()

beh_val = behavior[
    behavior["split"]
    == "validation"
].copy()

dr_cal = droid[
    droid["split"]
    == "calibration"
].copy()

dr_val = droid[
    droid["split"]
    == "validation"
].copy()

assert len(beh_cal) == 5590
assert len(beh_val) == 2397

assert len(dr_cal) == 1176
assert len(dr_val) == 503


# ============================================================
# C1 -- Global identifiability:
# BEHAVIOR vs DROID
# ============================================================

global_rows = []

for dataset_name in [
    "BEHAVIOR",
    "DROID",
]:

    if dataset_name == "BEHAVIOR":
        cal = beh_cal
        val = beh_val
        metric_map = BEH_METRICS
        error_col = BEH_ERROR
        nuisance_col = "task"
        group_col = "task"

    else:
        cal = dr_cal
        val = dr_val
        metric_map = DROID_METRICS
        error_col = DROID_ERROR
        nuisance_col = "site"
        group_col = "site"

    for short_name, metric in metric_map.items():

        fit = fit_models(
            cal,
            metric,
            error_col,
            nuisance_col=nuisance_col,
        )

        validation = validate_boundary(
            val,
            metric,
            error_col,
            fit["threshold"],
        )

        final_valid = bool(
            fit["calibration_support"]
            and validation[
                "validation_direction"
            ]
        )

        boot = bootstrap_fit(
            cal,
            metric,
            error_col,
            group_col=group_col,
            nuisance_col=nuisance_col,
        )

        boot.to_csv(
            OUTDIR
            / (
                f"C1_{dataset_name.lower()}_"
                f"{short_name}_bootstrap.csv"
            ),
            index=False,
        )

        bs = summarize_bootstrap(
            boot
        )

        global_rows.append({
            "analysis":
                "C1_GLOBAL",

            "dataset":
                dataset_name,

            "metric":
                short_name,

            "metric_column":
                metric,

            "calibration_n":
                len(cal),

            "validation_n":
                len(val),

            **fit,
            **validation,
            **bs,

            "boundary_valid":
                final_valid,
        })

global_df = pd.DataFrame(
    global_rows
)

global_df.to_csv(
    OUTDIR
    / "C1_global_identifiability.csv",
    index=False,
)


# ============================================================
# C2 -- DROID site-specific local boundary analysis
#
# No site dummy inside a single site.
# Use the already-frozen calibration/validation membership.
# ALL sites are retained in output.
# ============================================================

site_rows = []

for site, full_site in droid.groupby(
    "site",
    sort=True,
):

    site_cal = full_site[
        full_site["split"]
        == "calibration"
    ].copy()

    site_val = full_site[
        full_site["split"]
        == "validation"
    ].copy()

    for short_name, metric in DROID_METRICS.items():

        row = {
            "analysis":
                "C2_SITE",

            "site":
                site,

            "metric":
                short_name,

            "total_n":
                len(full_site),

            "calibration_n":
                len(site_cal),

            "validation_n":
                len(site_val),
        }

        # Don't silently exclude small sites.
        # If fitting is impossible, report it explicitly.
        if (
            len(site_cal) < 10
            or len(site_val) < 2
        ):
            row.update({
                "fit_status":
                    "INSUFFICIENT_N",

                "boundary_valid":
                    False,
            })

            site_rows.append(row)
            continue

        fit = fit_models(
            site_cal,
            metric,
            DROID_ERROR,
            nuisance_col=None,
        )

        if fit is None:
            row.update({
                "fit_status":
                    "FIT_FAILED",

                "boundary_valid":
                    False,
            })

            site_rows.append(row)
            continue

        validation = validate_boundary(
            site_val,
            metric,
            DROID_ERROR,
            fit["threshold"],
        )

        final_valid = bool(
            fit["calibration_support"]
            and validation[
                "validation_direction"
            ]
        )

        row.update({
            "fit_status":
                "OK",

            **fit,
            **validation,

            "boundary_valid":
                final_valid,
        })

        site_rows.append(row)

site_df = pd.DataFrame(
    site_rows
)

site_df.to_csv(
    OUTDIR
    / "C2_droid_site_boundaries.csv",
    index=False,
)


# ============================================================
# C3 -- Site-specific bootstrap stability
#
# Only sites for which fitting is numerically possible.
# Still no result-based N threshold / cherry-picking:
# every possible site is attempted.
# ============================================================

site_boot_summary = []
site_boot_all = []

rng_master = np.random.default_rng(
    SEED
)

for site, full_site in droid.groupby(
    "site",
    sort=True,
):

    site_cal = full_site[
        full_site["split"]
        == "calibration"
    ].copy()

    if len(site_cal) < 10:
        continue

    for short_name, metric in DROID_METRICS.items():

        rows = []

        for b in range(BOOTSTRAPS):

            # Resample only within this site's calibration set.
            idx = rng_master.choice(
                site_cal.index.to_numpy(),
                size=len(site_cal),
                replace=True,
            )

            boot = site_cal.loc[
                idx
            ].reset_index(drop=True)

            fit = fit_models(
                boot,
                metric,
                DROID_ERROR,
                nuisance_col=None,
            )

            if fit is None:
                continue

            rows.append({
                "site":
                    site,

                "metric":
                    short_name,

                "bootstrap":
                    b,

                "threshold":
                    fit["threshold"],

                "delta_bic_vs_linear":
                    fit[
                        "delta_bic_vs_linear"
                    ],

                "delta_bic_vs_quadratic":
                    fit[
                        "delta_bic_vs_quadratic"
                    ],

                "slope_pre":
                    fit["slope_pre"],

                "slope_post":
                    fit["slope_post"],

                "calibration_support":
                    fit[
                        "calibration_support"
                    ],
            })

        bdf = pd.DataFrame(
            rows
        )

        if len(bdf):
            site_boot_all.append(
                bdf
            )

        summary = summarize_bootstrap(
            bdf
        )

        site_boot_summary.append({
            "site":
                site,

            "metric":
                short_name,

            "calibration_n":
                len(site_cal),

            **summary,
        })


site_boot_summary_df = pd.DataFrame(
    site_boot_summary
)

site_boot_summary_df.to_csv(
    OUTDIR
    / "C3_droid_site_bootstrap_summary.csv",
    index=False,
)

if site_boot_all:
    pd.concat(
        site_boot_all,
        ignore_index=True,
    ).to_csv(
        OUTDIR
        / "C3_droid_site_bootstrap_all.csv",
        index=False,
    )


# ============================================================
# Human-readable summary
# ============================================================

print()
print("=" * 100)
print("C1 — GLOBAL BOUNDARY IDENTIFIABILITY")
print("=" * 100)

cols = [
    "dataset",
    "metric",
    "threshold",
    "delta_bic_vs_linear",
    "delta_bic_vs_quadratic",
    "slope_pre",
    "slope_post",
    "val_out_in_ratio",
    "val_cliff_delta",
    "val_mw_p",
    "bootstrap_support_rate",
    "threshold_relative_ci_width",
    "boundary_valid",
]

print(
    global_df[cols]
    .to_string(index=False)
)

print()
print("=" * 100)
print("C2 — DROID SITE-SPECIFIC BOUNDARIES")
print("=" * 100)

show = [
    c for c in [
        "site",
        "metric",
        "total_n",
        "calibration_n",
        "validation_n",
        "fit_status",
        "threshold",
        "delta_bic_vs_linear",
        "delta_bic_vs_quadratic",
        "slope_pre",
        "slope_post",
        "val_out_in_ratio",
        "val_cliff_delta",
        "val_mw_p",
        "boundary_valid",
    ]
    if c in site_df.columns
]

print(
    site_df[show]
    .to_string(index=False)
)

print()
print("=" * 100)
print("C3 — DROID SITE BOOTSTRAP STABILITY")
print("=" * 100)

print(
    site_boot_summary_df
    .sort_values([
        "site",
        "metric",
    ])
    .to_string(index=False)
)

print()
print("=" * 100)
print("OUTPUT DIR =", OUTDIR)
print("SUCCESS = True")
print("=" * 100)
