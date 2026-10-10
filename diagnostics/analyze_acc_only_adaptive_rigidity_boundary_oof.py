from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

# ============================================================
# CONTRACT
# ============================================================

SEED = 20260921
NFOLDS = 5

# Established SI RMS-Acceleration boundary
B_FIXED = 0.24580418  # m/s^2

ROOT = Path("diagnostics/results")

REPAIR = (
    ROOT
    / "rigid_projection_repair_current10457"
    / "combined_5task_repair_per_clip.csv"
)

ACC = (
    ROOT
    / "current_behavior_established_acc_10457"
    / "per_sample_current_established_acc_10457.csv"
)

PROX = (
    ROOT
    / "scene_action_proximity_probe"
    / "scene_action_proximity_5task_2397.csv"
)

OUTDIR = (
    ROOT
    / "adaptive_rigidity_boundary_oof"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


def stable_number(text, salt):
    h = hashlib.sha256(
        f"{SEED}|{salt}|{text}".encode()
    ).digest()

    return (
        int.from_bytes(h[:8], "big")
        / 2**64
    )


# ============================================================
# LOAD
# ============================================================

r = pd.read_csv(REPAIR)
a = pd.read_csv(ACC)
p = pd.read_csv(PROX)

need_r = [
    "clip",
    "task",
    "raw_mp",
    "repaired_mp",
    "residual_mm",
]

need_a = [
    "clip",
    "task",
    "rms_acceleration_si_m_s2",
]

need_p = [
    "clip",
    "task",
    "acc_matched",
    "scene_action_dist_p05_native",
]

for c in need_r:
    assert c in r.columns, c

for c in need_a:
    assert c in a.columns, c

for c in need_p:
    assert c in p.columns, c


# Only context clips for which the previous deployable
# context experiment had valid matched measurements.
p = p[
    p["acc_matched"] == 1
][need_p].copy()

p = p.dropna(
    subset=[
        "scene_action_dist_p05_native",
    ]
)

p = p.drop_duplicates(
    ["task", "clip"]
)

d = (
    r[need_r]
    .merge(
        a[need_a],
        on=["task", "clip"],
        how="inner",
        validate="one_to_one",
    )
    .merge(
        p,
        on=["task", "clip"],
        how="inner",
        validate="one_to_one",
    )
)

if len(d) < 1000:
    raise RuntimeError(
        f"Unexpectedly small overlap: {len(d)}"
    )

d["benefit_mm"] = (
    d["raw_mp"]
    - d["repaired_mp"]
)

# ============================================================
# DEPLOYABLE INPUTS
#
# z:
#   action severity relative to established fixed B
#
# closeness:
#   initial scene/action geometric proximity
#
# NO MSJ.
# NO future GT as a MODEL INPUT.
#
# residual is used only as TRAINING TARGET.
# OOF prediction ensures each evaluated clip was not used
# to train its own model.
# ============================================================

acc_si = d[
    "rms_acceleration_si_m_s2"
].to_numpy(float)

dist = d[
    "scene_action_dist_p05_native"
].to_numpy(float)

d["z"] = np.log(
    np.maximum(acc_si, 1e-8)
    / B_FIXED
)

d["closeness"] = -np.log(
    np.maximum(dist, 1e-8)
)

# Continuous physical distortion target
d["target_log_residual"] = np.log1p(
    d["residual_mm"].astype(float)
)


# ============================================================
# DETERMINISTIC 5-FOLD OOF WITHIN EACH TASK
# ============================================================

d["fold"] = -1

for task, g in d.groupby("task"):

    idx = sorted(
        g.index,
        key=lambda i: stable_number(
            str(d.at[i, "clip"]),
            f"RIGID-OOF-{task}",
        ),
    )

    for pos, i in enumerate(idx):
        d.at[i, "fold"] = (
            pos % NFOLDS
        )

assert (d["fold"] >= 0).all()


# ============================================================
# MODEL:
#
# log(1 + rigid residual)
#     = A(context) + beta_acc * log(Acc / B)
#
# Context shifts the allowed Acc threshold.
#
# We intentionally DO NOT use z*context interaction here:
# this keeps the learned boundary monotonic and interpretable.
# ============================================================

d["oof_pred_log_residual"] = np.nan

# Analytical form of each fold model:
# prediction = A_i + S_i * z
d["context_intercept_A"] = np.nan
d["acc_slope_S"] = np.nan
d["ridge_alpha"] = np.nan

ALPHAS = np.array(
    [0.01, 0.1, 1.0, 10.0, 100.0],
    dtype=float,
)

for task in sorted(
    d["task"].unique()
):

    tm = (
        d["task"] == task
    )

    for fold in range(NFOLDS):

        tr = (
            tm
            & (d["fold"] != fold)
        )

        te = (
            tm
            & (d["fold"] == fold)
        )

        if tr.sum() < 20:
            raise RuntimeError(
                f"Too few train samples "
                f"{task} fold={fold}: "
                f"{tr.sum()}"
            )

        # z + context
        Xtr = d.loc[
            tr,
            ["z", "closeness"],
        ].to_numpy(float)

        Xte = d.loc[
            te,
            ["z", "closeness"],
        ].to_numpy(float)

        ytr = d.loc[
            tr,
            "target_log_residual",
        ].to_numpy(float)

        scaler = StandardScaler()

        Xtr_s = scaler.fit_transform(
            Xtr
        )

        Xte_s = scaler.transform(
            Xte
        )

        reg = RidgeCV(
            alphas=ALPHAS,
        )

        reg.fit(
            Xtr_s,
            ytr,
        )

        pred = reg.predict(
            Xte_s
        )

        d.loc[
            te,
            "oof_pred_log_residual",
        ] = pred

        beta = np.asarray(
            reg.coef_,
            dtype=float,
        )

        means = scaler.mean_
        scales = scaler.scale_

        c = d.loc[
            te,
            "closeness",
        ].to_numpy(float)

        # prediction = A(c) + S*z
        S = (
            beta[0]
            / scales[0]
        )

        A = (
            float(reg.intercept_)
            - beta[0]
              * means[0]
              / scales[0]
            + beta[1]
              * (
                  c - means[1]
              )
              / scales[1]
        )

        d.loc[
            te,
            "context_intercept_A",
        ] = A

        d.loc[
            te,
            "acc_slope_S",
        ] = S

        d.loc[
            te,
            "ridge_alpha",
        ] = float(
            reg.alpha_
        )

        # Exact algebra audit
        pred2 = (
            A
            + S
            * d.loc[
                te,
                "z",
            ].to_numpy(float)
        )

        err = float(
            np.max(
                np.abs(
                    pred
                    - pred2
                )
            )
        )

        if err > 1e-9:
            raise RuntimeError(
                f"Analytical prediction mismatch: "
                f"{task} fold={fold} "
                f"err={err}"
            )


assert d[
    [
        "oof_pred_log_residual",
        "context_intercept_A",
        "acc_slope_S",
    ]
].notna().all().all()


# ============================================================
# OOF QUALITY
# ============================================================

rho, pval = spearmanr(
    d["oof_pred_log_residual"],
    d["residual_mm"],
)

print(
    "=" * 84
)
print(
    "ACC-ONLY CONTEXT-ADAPTIVE RIGIDITY BOUNDARY"
)
print(
    "=" * 84
)

print(
    "OVERLAP N =",
    len(d),
)

print(
    "Fixed B =",
    B_FIXED,
    "m/s^2",
)

print(
    "OOF predicted residual vs actual residual:"
)
print(
    "Spearman rho = %.6f"
    % rho
)
print(
    "p = %.6e"
    % pval
)


# ============================================================
# FIXED B GATE
# ============================================================

d["fixed_gate"] = (
    d[
        "rms_acceleration_si_m_s2"
    ]
    > B_FIXED
).astype(int)

K = int(
    d["fixed_gate"].sum()
)

print()
print(
    "Fixed-B selected K =",
    K,
    "/",
    len(d),
    "(%.2f%%)"
    % (
        100.0
        * K
        / len(d)
    ),
)


# ============================================================
# ADAPTIVE GATE — SAME EXACT BUDGET
#
# Highest cross-fitted predicted rigid distortion first.
# ============================================================

ranked = d.sort_values(
    [
        "oof_pred_log_residual",
        "clip",
    ],
    ascending=[
        False,
        True,
    ],
    kind="mergesort",
)

selected = set(
    ranked.iloc[:K][
        "clip"
    ].tolist()
)

d["adaptive_gate"] = (
    d["clip"]
    .isin(selected)
    .astype(int)
)

assert int(
    d["adaptive_gate"].sum()
) == K


# score threshold halfway between selected/rejected
if 0 < K < len(ranked):

    score_hi = float(
        ranked.iloc[
            K - 1
        ][
            "oof_pred_log_residual"
        ]
    )

    score_lo = float(
        ranked.iloc[
            K
        ][
            "oof_pred_log_residual"
        ]
    )

    SCORE_TAU = (
        score_hi
        + score_lo
    ) / 2.0

else:
    raise RuntimeError(
        "Invalid K"
    )


# ============================================================
# EXPLICIT PER-CLIP B_i
#
# tau = A(context) + S*log(B_i/B_FIXED)
#
# therefore:
#
# B_i = B_FIXED *
#       exp((tau - A(context)) / S)
# ============================================================

positive_slope = (
    d["acc_slope_S"] > 1e-8
)

d["adaptive_B_si_m_s2"] = np.nan

d.loc[
    positive_slope,
    "adaptive_B_si_m_s2",
] = (
    B_FIXED
    * np.exp(
        (
            SCORE_TAU
            - d.loc[
                positive_slope,
                "context_intercept_A",
            ]
        )
        / d.loc[
            positive_slope,
            "acc_slope_S",
        ]
    )
)

finite_B = (
    positive_slope
    & np.isfinite(
        d[
            "adaptive_B_si_m_s2"
        ]
    )
)

# Mathematical gate implied by explicit B_i.
d["adaptive_boundary_gate"] = (
    d[
        "rms_acceleration_si_m_s2"
    ]
    > d[
        "adaptive_B_si_m_s2"
    ]
).astype(int)

agreement = float(
    (
        d.loc[
            finite_B,
            "adaptive_boundary_gate",
        ]
        ==
        d.loc[
            finite_B,
            "adaptive_gate",
        ]
    ).mean()
)


# ============================================================
# SELECTION QUALITY
# ============================================================

def describe_gate(name, col):

    g = d[
        d[col] == 1
    ]

    return {
        "gate": name,
        "N": len(g),

        "median_acc":
            g[
                "rms_acceleration_si_m_s2"
            ].median(),

        "median_rigid_residual_mm":
            g[
                "residual_mm"
            ].median(),

        "mean_rigid_residual_mm":
            g[
                "residual_mm"
            ].mean(),

        "median_repair_benefit_mm":
            g[
                "benefit_mm"
            ].median(),

        "mean_repair_benefit_mm":
            g[
                "benefit_mm"
            ].mean(),

        "repair_success_pct":
            100.0
            * (
                g[
                    "benefit_mm"
                ]
                > 0
            ).mean(),
    }


sel_summary = pd.DataFrame(
    [
        describe_gate(
            "FIXED_B",
            "fixed_gate",
        ),
        describe_gate(
            "ADAPTIVE_B",
            "adaptive_gate",
        ),
    ]
)

print()
print(
    "=" * 84
)
print(
    "SAME-BUDGET SELECTION QUALITY"
)
print(
    "=" * 84
)

print(
    sel_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# HOW MUCH RIGID DISTORTION DOES EACH GATE CAPTURE?
# ============================================================

total_residual_mass = float(
    d["residual_mm"].sum()
)

fixed_residual_mass = float(
    d.loc[
        d["fixed_gate"] == 1,
        "residual_mm",
    ].sum()
)

adaptive_residual_mass = float(
    d.loc[
        d["adaptive_gate"] == 1,
        "residual_mm",
    ].sum()
)

print()
print(
    "=" * 84
)
print(
    "RIGID DISTORTION CAPTURE"
)
print(
    "=" * 84
)

print(
    "Fixed-B captured residual mass = "
    "%.2f%%"
    % (
        100.0
        * fixed_residual_mass
        / total_residual_mass
    )
)

print(
    "Adaptive-B captured residual mass = "
    "%.2f%%"
    % (
        100.0
        * adaptive_residual_mass
        / total_residual_mass
    )
)

print(
    "Adaptive vs fixed relative capture = "
    "%+.2f%%"
    % (
        100.0
        * (
            adaptive_residual_mass
            / fixed_residual_mass
            - 1.0
        )
    )
)


# ============================================================
# ACTUAL MPJPE AFTER SELECTIVE RIGID REPAIR
#
# If gate=1:
#   use repaired MPJPE
#
# else:
#   retain raw PointWorld MPJPE
# ============================================================

raw = d[
    "raw_mp"
].to_numpy(float)

always = d[
    "repaired_mp"
].to_numpy(float)

fixed_policy = np.where(
    d["fixed_gate"].to_numpy(bool),
    always,
    raw,
)

adaptive_policy = np.where(
    d["adaptive_gate"].to_numpy(bool),
    always,
    raw,
)

raw_mean = float(
    np.mean(raw)
)

always_mean = float(
    np.mean(always)
)

fixed_mean = float(
    np.mean(fixed_policy)
)

adaptive_mean = float(
    np.mean(adaptive_policy)
)

always_gain = (
    raw_mean
    - always_mean
)

fixed_gain = (
    raw_mean
    - fixed_mean
)

adaptive_gain = (
    raw_mean
    - adaptive_mean
)

print()
print(
    "=" * 84
)
print(
    "SELECTIVE REPAIR → MPJPE"
)
print(
    "=" * 84
)

print(
    "Raw mean MPJPE       = %.6f mm"
    % raw_mean
)

print(
    "Always repair        = %.6f mm"
    % always_mean
)

print(
    "Fixed-B selective    = %.6f mm"
    % fixed_mean
)

print(
    "Adaptive-B selective = %.6f mm"
    % adaptive_mean
)

print()

print(
    "Fixed gain = %.6f mm"
    % fixed_gain
)

print(
    "Adaptive gain = %.6f mm"
    % adaptive_gain
)

print(
    "Adaptive gain vs fixed = %+.2f%%"
    % (
        100.0
        * (
            adaptive_gain
            / fixed_gain
            - 1.0
        )
        if fixed_gain > 0
        else np.nan
    )
)

print(
    "Fixed captures %.2f%% "
    "of always-repair gain"
    % (
        100.0
        * fixed_gain
        / always_gain
    )
)

print(
    "Adaptive captures %.2f%% "
    "of always-repair gain"
    % (
        100.0
        * adaptive_gain
        / always_gain
    )
)


# ============================================================
# EXPLICIT ADAPTIVE BOUNDARY AUDIT
# ============================================================

print()
print(
    "=" * 84
)
print(
    "PER-CLIP ADAPTIVE B_i AUDIT"
)
print(
    "=" * 84
)

print(
    "Positive Acc slope = %.2f%%"
    % (
        100.0
        * positive_slope.mean()
    )
)

print(
    "Finite adaptive B = %.2f%%"
    % (
        100.0
        * finite_B.mean()
    )
)

print(
    "Adaptive gate == (Acc > B_i) "
    "on valid clips = %.2f%%"
    % (
        100.0
        * agreement
    )
)

if finite_B.any():

    vals = d.loc[
        finite_B,
        "adaptive_B_si_m_s2",
    ]

    print(
        "B_i q10 = %.6f"
        % vals.quantile(0.10)
    )

    print(
        "B_i q25 = %.6f"
        % vals.quantile(0.25)
    )

    print(
        "B_i median = %.6f"
        % vals.median()
    )

    print(
        "B_i q75 = %.6f"
        % vals.quantile(0.75)
    )

    print(
        "B_i q90 = %.6f"
        % vals.quantile(0.90)
    )


# ============================================================
# TASKWISE
# ============================================================

print()
print(
    "=" * 84
)
print(
    "TASKWISE FIXED vs ADAPTIVE"
)
print(
    "=" * 84
)

task_rows = []

for task, g in d.groupby(
    "task"
):

    for name, col in [
        (
            "FIXED",
            "fixed_gate",
        ),
        (
            "ADAPTIVE",
            "adaptive_gate",
        ),
    ]:

        s = g[
            g[col] == 1
        ]

        task_rows.append(
            {
                "task": task,
                "gate": name,
                "selected_n":
                    len(s),

                "selected_pct":
                    100.0
                    * len(s)
                    / len(g),

                "median_residual_mm":
                    s[
                        "residual_mm"
                    ].median()
                    if len(s)
                    else np.nan,

                "median_benefit_mm":
                    s[
                        "benefit_mm"
                    ].median()
                    if len(s)
                    else np.nan,
            }
        )

task_df = pd.DataFrame(
    task_rows
)

print(
    task_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ============================================================
# SAVE
# ============================================================

d.to_csv(
    OUTDIR
    / "adaptive_rigidity_boundary_oof_per_clip.csv",
    index=False,
)

sel_summary.to_csv(
    OUTDIR
    / "adaptive_vs_fixed_selection_summary.csv",
    index=False,
)

task_df.to_csv(
    OUTDIR
    / "adaptive_vs_fixed_taskwise.csv",
    index=False,
)

print()
print(
    "SAVED =",
    OUTDIR,
)
