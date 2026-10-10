#!/usr/bin/env python3

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260902

ROOT = Path(__file__).resolve().parents[1]

IN = (
    ROOT /
    "diagnostics/results/boundary_identifiability_context_C/"
    "C6_droid_action_scene_coupling/"
    "C6_droid_action_scene_dynamics_1679.csv"
)

OUTDIR = (
    ROOT /
    "diagnostics/results/boundary_identifiability_context_C/"
    "C7_context_reliability"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Fixed site-stratified 70/30 split
# ============================================================

def stable_number(text):
    h = hashlib.sha256(
        f"{SEED}|{text}".encode()
    ).digest()

    return int.from_bytes(
        h[:8], "big"
    ) / 2**64


def exact_split(df):

    split = pd.Series(
        index=df.index,
        dtype="object"
    )

    for site, g in df.groupby("site"):

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

        split.loc[
            idx[:n_cal]
        ] = "calibration"

        split.loc[
            idx[n_cal:]
        ] = "validation"

    return split


# ============================================================
# Linear model helpers
# ============================================================

def design_base(df, site_levels):

    # Fixed site encoding learned from calibration.
    cat = pd.Categorical(
        df["site"],
        categories=site_levels
    )

    site = pd.get_dummies(
        cat,
        drop_first=True,
        dtype=float
    ).to_numpy(float)

    path = np.log1p(
        df["eef_path_length_m"]
        .to_numpy(float)
    )[:, None]

    return np.column_stack([
        np.ones(len(df)),
        site,
        path,
    ])


def safe_log(x):
    x = np.asarray(
        x,
        dtype=float
    )

    positive = x[
        np.isfinite(x)
        & (x > 0)
    ]

    floor = (
        positive.min() * 0.5
        if len(positive)
        else 1e-12
    )

    return np.log10(
        np.maximum(
            x,
            floor
        )
    )


def fit_ols(X, y):

    beta, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None
    )

    pred = X @ beta

    resid = y - pred

    sse = max(
        float(
            resid @ resid
        ),
        1e-15
    )

    n, k = X.shape

    bic = (
        n * np.log(sse / n)
        + k * np.log(n)
    )

    return beta, float(bic)


def eval_model(
    beta,
    X,
    y
):

    pred = X @ beta

    sse = float(
        np.sum(
            (y - pred) ** 2
        )
    )

    sst = float(
        np.sum(
            (y - np.mean(y)) ** 2
        )
    )

    r2 = (
        1.0 - sse / sst
        if sst > 0
        else np.nan
    )

    mae = float(
        np.mean(
            np.abs(
                y - pred
            )
        )
    )

    return r2, mae


# ============================================================
# Load
# ============================================================

d = pd.read_csv(IN)

assert len(d) == 1679
assert d["clip"].nunique() == 1679

d["split"] = exact_split(d)

cal = d[
    d["split"] == "calibration"
].copy()

val = d[
    d["split"] == "validation"
].copy()

assert len(cal) == 1176
assert len(val) == 503

site_levels = sorted(
    cal["site"]
    .astype(str)
    .unique()
)

print("calibration =", len(cal))
print("validation  =", len(val))


# ============================================================
# Pre-specified two analyses
# ============================================================

PAIRS = [
    (
        "ACC",
        "rms_acceleration_si_m_s2",
        "scene_acc_rms_si_m_s2",
    ),
    (
        "MSJ",
        "mean_squared_jerk_si_m2_s6",
        "scene_msj_si_m2_s6",
    ),
]

rows = []

for label, action_col, scene_col in PAIRS:

    need = [
        "filtered_l2_moved_mm",
        "eef_path_length_m",
        action_col,
        scene_col,
        "site",
    ]

    c = cal[
        need
    ].dropna().copy()

    v = val[
        need
    ].dropna().copy()

    # --------------------------------------------------------
    # Outcome
    # --------------------------------------------------------

    yc = np.log1p(
        c[
            "filtered_l2_moved_mm"
        ].to_numpy(float)
    )

    yv = np.log1p(
        v[
            "filtered_l2_moved_mm"
        ].to_numpy(float)
    )

    # --------------------------------------------------------
    # Predictors
    # --------------------------------------------------------

    X0c = design_base(
        c,
        site_levels
    )

    X0v = design_base(
        v,
        site_levels
    )

    action_c = safe_log(
        c[action_col]
    )[:, None]

    action_v = safe_log(
        v[action_col]
    )[:, None]

    scene_c = safe_log(
        c[scene_col]
    )[:, None]

    scene_v = safe_log(
        v[scene_col]
    )[:, None]

    # Standardize ONLY using calibration statistics.
    ac_mu = action_c.mean()
    ac_sd = action_c.std(ddof=1)

    sc_mu = scene_c.mean()
    sc_sd = scene_c.std(ddof=1)

    action_c = (
        action_c - ac_mu
    ) / ac_sd

    action_v = (
        action_v - ac_mu
    ) / ac_sd

    scene_c = (
        scene_c - sc_mu
    ) / sc_sd

    scene_v = (
        scene_v - sc_mu
    ) / sc_sd

    interaction_c = (
        action_c * scene_c
    )

    interaction_v = (
        action_v * scene_v
    )

    # --------------------------------------------------------
    # M0: context only
    # --------------------------------------------------------

    M0c = X0c
    M0v = X0v

    # --------------------------------------------------------
    # M1: context + Action dynamics
    # --------------------------------------------------------

    M1c = np.column_stack([
        X0c,
        action_c,
    ])

    M1v = np.column_stack([
        X0v,
        action_v,
    ])

    # --------------------------------------------------------
    # M2: context + Action + actual Scene dynamics
    # Mechanism model, NOT inference-time deployable.
    # --------------------------------------------------------

    M2c = np.column_stack([
        X0c,
        action_c,
        scene_c,
    ])

    M2v = np.column_stack([
        X0v,
        action_v,
        scene_v,
    ])

    # --------------------------------------------------------
    # M3: + Action x Scene coupling interaction
    # --------------------------------------------------------

    M3c = np.column_stack([
        X0c,
        action_c,
        scene_c,
        interaction_c,
    ])

    M3v = np.column_stack([
        X0v,
        action_v,
        scene_v,
        interaction_v,
    ])

    models = [
        ("context_only", M0c, M0v),
        ("+action", M1c, M1v),
        ("+scene", M2c, M2v),
        ("+action_scene_interaction", M3c, M3v),
    ]

    fitted = {}

    for name, Xc, Xv in models:

        beta, bic = fit_ols(
            Xc,
            yc
        )

        cal_r2, cal_mae = (
            eval_model(
                beta,
                Xc,
                yc
            )
        )

        val_r2, val_mae = (
            eval_model(
                beta,
                Xv,
                yv
            )
        )

        fitted[name] = {
            "bic": bic,
            "cal_r2": cal_r2,
            "val_r2": val_r2,
            "val_mae": val_mae,
        }

        rows.append({
            "metric": label,
            "model": name,
            "calibration_n": len(c),
            "validation_n": len(v),
            "calibration_bic": bic,
            "calibration_r2": cal_r2,
            "validation_r2": val_r2,
            "validation_mae_logerror":
                val_mae,
        })

    print()
    print("=" * 90)
    print(label)
    print("=" * 90)

    for name in [
        "context_only",
        "+action",
        "+scene",
        "+action_scene_interaction",
    ]:
        x = fitted[name]

        print(
            f"{name:28s} | "
            f"BIC={x['bic']:.3f} | "
            f"val R2={x['val_r2']:.5f} | "
            f"val MAE={x['val_mae']:.5f}"
        )

    print()
    print(
        "Action incremental val R2 =",
        fitted["+action"]["val_r2"]
        - fitted["context_only"]["val_r2"]
    )

    print(
        "Scene incremental val R2 =",
        fitted["+scene"]["val_r2"]
        - fitted["+action"]["val_r2"]
    )

    print(
        "Interaction incremental val R2 =",
        fitted[
            "+action_scene_interaction"
        ]["val_r2"]
        - fitted["+scene"]["val_r2"]
    )


out = pd.DataFrame(rows)

out.to_csv(
    OUTDIR /
    "C7_context_reliability_models.csv",
    index=False
)

print()
print("=" * 90)
print("OUTPUT =", OUTDIR)
print("SUCCESS = True")
print("=" * 90)
