#!/usr/bin/env python3

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEED = 20260902

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
      "boundary_identifiability_context_C/C4_failure_mechanism"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Fixed utilities
# ============================================================

def stable_number(text):
    h = hashlib.sha256(
        f"{SEED}|{text}".encode()
    ).digest()

    return int.from_bytes(
        h[:8], "big"
    ) / 2**64


def exact_group_split(df, group_col):
    result = pd.Series(
        index=df.index,
        dtype="object",
    )

    for _, g in df.groupby(group_col):

        idx = sorted(
            g.index,
            key=lambda i:
                stable_number(
                    str(df.at[i, "clip"])
                )
        )

        n_cal = round(
            len(idx) * 0.70
        )

        result.loc[idx[:n_cal]] = "calibration"
        result.loc[idx[n_cal:]] = "validation"

    return result


def dummies(s):
    return pd.get_dummies(
        s.astype(str),
        drop_first=True,
        dtype=float,
    ).to_numpy(float)


def ols_fit(y, X):
    beta, *_ = np.linalg.lstsq(
        X, y, rcond=None
    )

    pred = X @ beta
    resid = y - pred

    sse = max(
        float(resid @ resid),
        1e-15,
    )

    n, k = X.shape

    bic = (
        n * np.log(sse / n)
        + k * np.log(n)
    )

    sst = max(
        float(
            ((y - y.mean()) ** 2).sum()
        ),
        1e-15,
    )

    r2 = 1.0 - sse / sst

    return {
        "beta": beta,
        "pred": pred,
        "resid": resid,
        "sse": sse,
        "bic": float(bic),
        "r2": float(r2),
    }


# ============================================================
# Same 3-model comparison as C1
# ============================================================

def model_shape(data, metric, error_col, nuisance_col):

    x_raw = data[metric].to_numpy(float)
    y_raw = data[error_col].to_numpy(float)

    valid = (
        np.isfinite(x_raw)
        & np.isfinite(y_raw)
    )

    data = data.loc[valid].copy()
    x_raw = x_raw[valid]
    y_raw = y_raw[valid]

    positive = x_raw[x_raw > 0]
    floor = positive.min() * 0.5

    x = np.log10(
        np.maximum(x_raw, floor)
    )

    y = np.log1p(y_raw)

    nuisance = dummies(
        data[nuisance_col]
    )

    # nuisance-only
    X0 = np.column_stack([
        np.ones(len(x)),
        nuisance,
    ])

    f0 = ols_fit(y, X0)

    # linear
    X1 = np.column_stack([
        np.ones(len(x)),
        x,
        nuisance,
    ])

    f1 = ols_fit(y, X1)

    # quadratic
    xc = x - x.mean()

    X2 = np.column_stack([
        np.ones(len(x)),
        x,
        xc ** 2,
        nuisance,
    ])

    f2 = ols_fit(y, X2)

    # segmented
    taus = np.unique(
        np.quantile(
            x,
            np.linspace(0.20, 0.90, 71)
        )
    )

    best = None

    for tau in taus:

        hinge = np.maximum(
            0.0,
            x - tau
        )

        Xs = np.column_stack([
            np.ones(len(x)),
            x,
            hinge,
            nuisance,
        ])

        fs = ols_fit(y, Xs)

        if (
            best is None
            or fs["bic"] < best["bic"]
        ):
            best = {
                **fs,
                "tau": float(tau),
            }

    pre = float(
        best["beta"][1]
    )

    post = float(
        best["beta"][1]
        + best["beta"][2]
    )

    return {
        "n": len(x),

        "r2_nuisance_only": f0["r2"],
        "r2_linear": f1["r2"],
        "delta_r2_dynamics":
            f1["r2"] - f0["r2"],

        "bic_linear": f1["bic"],
        "bic_quadratic": f2["bic"],
        "bic_segmented": best["bic"],

        "delta_bic_seg_linear":
            f1["bic"] - best["bic"],

        "delta_bic_seg_quad":
            f2["bic"] - best["bic"],

        "threshold":
            float(10 ** best["tau"]),

        "slope_pre": pre,
        "slope_post": post,

        "residual_sd_logerror":
            float(
                np.std(
                    f1["resid"],
                    ddof=1
                )
            ),

        "residual_mad_logerror":
            float(
                np.median(
                    np.abs(
                        f1["resid"]
                        - np.median(
                            f1["resid"]
                        )
                    )
                )
            ),
    }


# ============================================================
# Distribution / coverage statistics
# ============================================================

def range_stats(x):

    x = np.asarray(x, dtype=float)
    x = x[
        np.isfinite(x)
        & (x > 0)
    ]

    q = np.quantile(
        x,
        [
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )

    (
        q01,q05,q10,q25,q50,
        q75,q90,q95,q99
    ) = q

    lx = np.log10(x)

    return {
        "n": len(x),

        "p01": q01,
        "p05": q05,
        "p10": q10,
        "p25": q25,
        "median": q50,
        "p75": q75,
        "p90": q90,
        "p95": q95,
        "p99": q99,

        "q95_q05_ratio":
            q95 / q05,

        "q99_median_ratio":
            q99 / q50,

        "iqr_over_median":
            (q75 - q25) / q50,

        # Unit-independent multiplicative spread.
        "log10_p95_p05_width":
            np.log10(q95)
            - np.log10(q05),

        "log10_sd":
            float(
                np.std(lx, ddof=1)
            ),
    }


# ============================================================
# Equal-population dynamics bins:
# does error rise sharply or smoothly?
# ============================================================

def bin_summary(
    data,
    metric,
    error_col,
    bins=10,
):

    d = data[
        [metric, error_col]
    ].dropna().copy()

    # rank-based bins avoid failures from repeated values
    ranks = (
        d[metric]
        .rank(
            method="first",
            pct=True
        )
    )

    d["bin"] = np.minimum(
        (ranks * bins).astype(int),
        bins - 1
    )

    rows = []

    for b, g in d.groupby("bin"):

        err = g[
            error_col
        ].to_numpy(float)

        dyn = g[
            metric
        ].to_numpy(float)

        rows.append({
            "bin": int(b) + 1,
            "n": len(g),

            "dynamics_median":
                float(np.median(dyn)),

            "error_median_mm":
                float(np.median(err)),

            "error_p25_mm":
                float(
                    np.quantile(
                        err, 0.25
                    )
                ),

            "error_p75_mm":
                float(
                    np.quantile(
                        err, 0.75
                    )
                ),

            "error_iqr_mm":
                float(
                    np.quantile(
                        err, 0.75
                    )
                    - np.quantile(
                        err, 0.25
                    )
                ),

            "error_cv":
                float(
                    np.std(err, ddof=1)
                    / np.mean(err)
                )
                if np.mean(err) > 0
                else np.nan,
        })

    return pd.DataFrame(rows)


# ============================================================
# Load and align physical metrics
# ============================================================

beh = pd.read_csv(
    BEHAVIOR_CSV
)

dro = pd.read_csv(
    DROID_CSV
)

assert len(beh) == 7987
assert len(dro) == 1679

if "l2_moved_mm" in beh.columns:
    BEH_ERR = "l2_moved_mm"
elif "filtered_l2_moved_mm" in beh.columns:
    BEH_ERR = "filtered_l2_moved_mm"
else:
    raise RuntimeError(
        "BEHAVIOR moved error column missing"
    )

DROID_ERR = "filtered_l2_moved_mm"


# ------------------------------------------------------------
# BEHAVIOR fixed dt = 0.2 s:
#
# acceleration SI = native / dt^2 = native * 25
# MSJ SI          = native / dt^6 = native * 15625
#
# This is only a unit conversion.
# No metric or sample definition is changed.
# ------------------------------------------------------------

beh["acc_si"] = (
    beh["rms_acceleration"]
    * 25.0
)

beh["msj_si"] = (
    beh["mean_squared_jerk"]
    * 15625.0
)

dro["acc_si"] = (
    dro[
        "rms_acceleration_si_m_s2"
    ]
)

dro["msj_si"] = (
    dro[
        "mean_squared_jerk_si_m2_s6"
    ]
)


# ============================================================
# Frozen splits
# ============================================================

beh["split"] = exact_group_split(
    beh,
    "task"
)

dro["split"] = exact_group_split(
    dro,
    "site"
)

assert (
    beh["split"]
    .value_counts()
    .to_dict()
    == {
        "calibration": 5590,
        "validation": 2397,
    }
)

assert (
    dro["split"]
    .value_counts()
    .to_dict()
    == {
        "calibration": 1176,
        "validation": 503,
    }
)


# ============================================================
# C4-A: dynamics range / coverage
# Use complete predefined population descriptively.
# ============================================================

range_rows = []

for dataset, df in [
    ("BEHAVIOR", beh),
    ("DROID", dro),
]:

    for metric in [
        "acc_si",
        "msj_si",
    ]:

        range_rows.append({
            "dataset": dataset,
            "metric": metric,
            **range_stats(
                df[metric]
            ),
        })

range_df = pd.DataFrame(
    range_rows
)

range_df.to_csv(
    OUTDIR
    / "C4A_dynamics_range.csv",
    index=False,
)


# ============================================================
# C4-B: error variance / signal strength
# Fit CALIBRATION only.
# ============================================================

shape_rows = []

for dataset in [
    "BEHAVIOR",
    "DROID",
]:

    if dataset == "BEHAVIOR":

        cal = beh[
            beh["split"]
            == "calibration"
        ]

        nuisance = "task"
        err = BEH_ERR

    else:

        cal = dro[
            dro["split"]
            == "calibration"
        ]

        nuisance = "site"
        err = DROID_ERR

    for metric in [
        "acc_si",
        "msj_si",
    ]:

        r = model_shape(
            cal,
            metric,
            err,
            nuisance,
        )

        shape_rows.append({
            "dataset": dataset,
            "metric": metric,
            **r,
        })


shape_df = pd.DataFrame(
    shape_rows
)

shape_df.to_csv(
    OUTDIR
    / "C4B_signal_noise_shape.csv",
    index=False,
)


# ============================================================
# C4-C: decile response curves
#
# Calibration + validation written separately.
# Validation is descriptive only; it does not choose thresholds.
# ============================================================

all_bins = []

for dataset in [
    "BEHAVIOR",
    "DROID",
]:

    if dataset == "BEHAVIOR":
        df = beh
        err = BEH_ERR
    else:
        df = dro
        err = DROID_ERR

    for split_name in [
        "calibration",
        "validation",
    ]:

        part = df[
            df["split"]
            == split_name
        ]

        for metric in [
            "acc_si",
            "msj_si",
        ]:

            b = bin_summary(
                part,
                metric,
                err,
                bins=10,
            )

            b.insert(
                0,
                "split",
                split_name,
            )

            b.insert(
                0,
                "metric",
                metric,
            )

            b.insert(
                0,
                "dataset",
                dataset,
            )

            all_bins.append(b)

bins_df = pd.concat(
    all_bins,
    ignore_index=True,
)

bins_df.to_csv(
    OUTDIR
    / "C4C_decile_response.csv",
    index=False,
)


# ============================================================
# C4-D: continuous monotonic association
# descriptive raw + within-context rank control
# ============================================================

assoc_rows = []

for dataset in [
    "BEHAVIOR",
    "DROID",
]:

    if dataset == "BEHAVIOR":
        df = beh.copy()
        context = "task"
        err = BEH_ERR
    else:
        df = dro.copy()
        context = "site"
        err = DROID_ERR

    # Use full fixed population descriptively.
    for metric in [
        "acc_si",
        "msj_si",
    ]:

        sub = df[
            [metric, err, context]
        ].dropna().copy()

        rho_raw, p_raw = spearmanr(
            sub[metric],
            sub[err],
        )

        sub["metric_rank_context"] = (
            sub.groupby(context)[metric]
            .rank(pct=True)
        )

        sub["error_rank_context"] = (
            sub.groupby(context)[err]
            .rank(pct=True)
        )

        rho_ctx, p_ctx = spearmanr(
            sub["metric_rank_context"],
            sub["error_rank_context"],
        )

        assoc_rows.append({
            "dataset": dataset,
            "metric": metric,
            "n": len(sub),

            "raw_spearman_rho":
                float(rho_raw),

            "raw_spearman_p":
                float(p_raw),

            "within_context_spearman_rho":
                float(rho_ctx),

            "within_context_spearman_p":
                float(p_ctx),
        })


assoc_df = pd.DataFrame(
    assoc_rows
)

assoc_df.to_csv(
    OUTDIR
    / "C4D_continuous_association.csv",
    index=False,
)


# ============================================================
# Print compact paper-oriented summary
# ============================================================

print()
print("=" * 100)
print("C4-A — DYNAMICS RANGE / COVERAGE")
print("=" * 100)

print(
    range_df[
        [
            "dataset",
            "metric",
            "n",
            "median",
            "p05",
            "p95",
            "q95_q05_ratio",
            "q99_median_ratio",
            "iqr_over_median",
            "log10_p95_p05_width",
        ]
    ].to_string(index=False)
)


print()
print("=" * 100)
print("C4-B — SIGNAL / NOISE / RELATIONSHIP SHAPE")
print("=" * 100)

print(
    shape_df[
        [
            "dataset",
            "metric",
            "n",
            "r2_nuisance_only",
            "r2_linear",
            "delta_r2_dynamics",
            "residual_sd_logerror",
            "delta_bic_seg_linear",
            "delta_bic_seg_quad",
            "threshold",
            "slope_pre",
            "slope_post",
        ]
    ].to_string(index=False)
)


print()
print("=" * 100)
print("C4-C — VALIDATION DECILE MEDIAN ERRORS")
print("=" * 100)

for dataset in [
    "BEHAVIOR",
    "DROID",
]:

    for metric in [
        "acc_si",
        "msj_si",
    ]:

        z = bins_df[
            (bins_df["dataset"] == dataset)
            & (bins_df["metric"] == metric)
            & (bins_df["split"] == "validation")
        ]

        vals = [
            round(float(x), 4)
            for x in
            z["error_median_mm"]
            .to_numpy()
        ]

        print(
            dataset,
            metric,
            "Q1..Q10 error medians =",
            vals
        )


print()
print("=" * 100)
print("C4-D — CONTINUOUS ASSOCIATION")
print("=" * 100)

print(
    assoc_df.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("OUTPUT DIR =", OUTDIR)
print("SUCCESS = True")
print("=" * 100)
