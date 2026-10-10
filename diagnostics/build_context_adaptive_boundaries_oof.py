from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SEED = 20260902
NFOLDS = 5

TA = 0.009832167363499603
TJ = 8.152857428804615e-05

ROOT = Path(__file__).resolve().parents[1]

PROX = (
    ROOT
    / "diagnostics/results/scene_action_proximity_probe/"
      "scene_action_proximity_5task_2397.csv"
)

REF = (
    ROOT
    / "diagnostics/results/adaptive_reliable_envelope_E2D/"
      "E2D_equal_delta_5task_2397_merged.csv"
)

OLD_OOF = (
    ROOT
    / "diagnostics/results/context_incremental_fair_oof/"
      "oof_predictions.csv"
)

OUTDIR = (
    ROOT
    / "diagnostics/results/context_adaptive_boundary"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "context_adaptive_boundary_2067.csv"


def stable_number(text, salt):
    h = hashlib.sha256(
        f"{SEED}|{salt}|{text}".encode()
    ).digest()
    return int.from_bytes(h[:8], "big") / 2**64


# ============================================================
# Exact same N=2067 population
# ============================================================

d = pd.read_csv(PROX)

ref = (
    pd.read_csv(REF)[
        ["clip", "boundary_severity", "high_risk"]
    ]
    .drop_duplicates("clip")
)

q = (
    d[d["acc_matched"] == 1]
    .merge(
        ref,
        on="clip",
        how="left",
        validate="one_to_one",
    )
)

q = q.dropna(
    subset=[
        "scene_action_dist_p05_native",
        "boundary_severity",
        "high_risk",
        "original_action_acc",
        "original_action_msj",
    ]
).copy()

assert len(q) == 2067

# ------------------------------------------------------------
# Same features
# ------------------------------------------------------------

q["log_boundary"] = np.log1p(
    q["boundary_severity"].astype(float)
)

q["closeness"] = -np.log(
    np.maximum(
        q["scene_action_dist_p05_native"].astype(float),
        1e-8,
    )
)

q["boundary_x_closeness"] = (
    q["log_boundary"]
    * q["closeness"]
)

FEATURES = [
    "log_boundary",
    "closeness",
    "boundary_x_closeness",
]

# ------------------------------------------------------------
# EXACT same folds as fair incremental experiment
# ------------------------------------------------------------

q["fold"] = -1

for (task, risk), g in q.groupby(
    ["task", "high_risk"]
):
    idx = sorted(
        g.index,
        key=lambda i: stable_number(
            str(q.at[i, "clip"]),
            f"FAIRCTX-{task}-{risk}",
        ),
    )

    for pos, i in enumerate(idx):
        q.at[i, "fold"] = pos % NFOLDS

assert (q["fold"] >= 0).all()

q["context_score_recomputed"] = np.nan
q["severity_slope_B"] = np.nan
q["severity_intercept_A"] = np.nan

# ============================================================
# Re-fit exact OOF context model and recover analytical
# p(context risk | severity, proximity)
#
# For fixed closeness c:
# logit(p) = A + B * log1p(severity)
# ============================================================

for task in sorted(q["task"].unique()):

    tm = q["task"] == task

    for fold in range(NFOLDS):

        tr = tm & (q["fold"] != fold)
        te = tm & (q["fold"] == fold)

        Xtr = q.loc[tr, FEATURES].to_numpy(float)
        Xte = q.loc[te, FEATURES].to_numpy(float)
        ytr = q.loc[tr, "high_risk"].to_numpy(int)

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=5000,
            random_state=SEED,
        )

        clf.fit(Xtr_s, ytr)

        prob = clf.predict_proba(Xte_s)[:, 1]

        q.loc[
            te,
            "context_score_recomputed",
        ] = prob

        beta = clf.coef_[0]
        intercept = float(clf.intercept_[0])

        means = scaler.mean_
        scales = scaler.scale_

        c = q.loc[
            te,
            "closeness",
        ].to_numpy(float)

        # logit(p) = A + B*z
        # z = log1p(severity)
        B = (
            beta[0] / scales[0]
            +
            beta[2] * c / scales[2]
        )

        A = (
            intercept
            - beta[0] * means[0] / scales[0]
            + beta[1] * (c - means[1]) / scales[1]
            - beta[2] * means[2] / scales[2]
        )

        q.loc[te, "severity_slope_B"] = B
        q.loc[te, "severity_intercept_A"] = A


assert q[
    [
        "context_score_recomputed",
        "severity_slope_B",
        "severity_intercept_A",
    ]
].notna().all().all()

# ============================================================
# Exact reproduction audit against saved OOF predictions
# ============================================================

old = pd.read_csv(OLD_OOF)[
    ["clip", "task_context_oof"]
]

audit = q[
    ["clip", "context_score_recomputed"]
].merge(
    old,
    on="clip",
    how="left",
    validate="one_to_one",
)

maxdiff = float(
    np.max(
        np.abs(
            audit["context_score_recomputed"]
            - audit["task_context_oof"]
        )
    )
)

print("MAX OOF SCORE REPRODUCTION DIFF =", maxdiff)

assert maxdiff < 1e-10, maxdiff

# ============================================================
# Same correction budget as fixed global envelope
# ============================================================

q["fixed_gate"] = (
    (q["original_action_acc"] > TA)
    |
    (q["original_action_msj"] > TJ)
).astype(int)

K = int(q["fixed_gate"].sum())
assert K == 949, K

ranked = q.sort_values(
    ["context_score_recomputed", "clip"],
    ascending=[False, True],
    kind="mergesort",
)

selected_clips = set(
    ranked.iloc[:K]["clip"]
)

q["context_gate"] = (
    q["clip"].isin(selected_clips)
).astype(int)

assert int(q["context_gate"].sum()) == K

# Cutoff halfway between last selected / first rejected.
s_hi = float(
    ranked.iloc[K - 1]["context_score_recomputed"]
)

s_lo = float(
    ranked.iloc[K]["context_score_recomputed"]
)

risk_cutoff = (s_hi + s_lo) / 2.0

assert 0 < risk_cutoff < 1

logit_cutoff = np.log(
    risk_cutoff / (1.0 - risk_cutoff)
)

# ============================================================
# Adaptive severity boundary
# ============================================================

q["original_severity_explicit"] = np.maximum(
    q["original_action_acc"] / TA,
    q["original_action_msj"] / TJ,
)

# Only meaningful when risk increases with severity.
q["monotonic_valid"] = (
    q["severity_slope_B"] > 0
)

q["adaptive_log_severity_boundary"] = np.nan
q["adaptive_severity_boundary"] = np.nan

valid = q["monotonic_valid"]

q.loc[
    valid,
    "adaptive_log_severity_boundary",
] = (
    (
        logit_cutoff
        - q.loc[valid, "severity_intercept_A"]
    )
    /
    q.loc[valid, "severity_slope_B"]
)

q.loc[
    valid,
    "adaptive_severity_boundary",
] = np.expm1(
    q.loc[
        valid,
        "adaptive_log_severity_boundary",
    ]
)

# No post-hoc clipping.
q["adaptive_boundary_positive"] = (
    q["adaptive_severity_boundary"] > 0
)

q["adaptive_requires_reduction"] = (
    (q["context_gate"] == 1)
    &
    q["monotonic_valid"]
    &
    q["adaptive_boundary_positive"]
    &
    (
        q["adaptive_severity_boundary"]
        < q["original_severity_explicit"]
    )
)

# Convert severity boundary back to the two established axes.
q["adaptive_tau_acc"] = (
    TA * q["adaptive_severity_boundary"]
)

q["adaptive_tau_msj"] = (
    TJ * q["adaptive_severity_boundary"]
)

q["required_severity_ratio"] = (
    q["adaptive_severity_boundary"]
    / q["original_severity_explicit"]
)

# Gate groups for reuse accounting
q["gate_group"] = np.select(
    [
        (q["fixed_gate"] == 1) & (q["context_gate"] == 1),
        (q["fixed_gate"] == 1) & (q["context_gate"] == 0),
        (q["fixed_gate"] == 0) & (q["context_gate"] == 1),
    ],
    [
        "BOTH",
        "FIXED_ONLY",
        "CONTEXT_ONLY",
    ],
    default="NEITHER",
)

# ============================================================
# Audit
# ============================================================

print("\n" + "=" * 100)
print("CONTEXT-ADAPTIVE BOUNDARY AUDIT")
print("=" * 100)

print("N =", len(q))
print("FIXED BUDGET =", K)
print("CONTEXT BUDGET =", int(q["context_gate"].sum()))
print("RISK CUTOFF =", risk_cutoff)

print("\nGATE GROUPS:")
print(q["gate_group"].value_counts().to_string())

print("\nMONOTONIC VALID:")
print(q["monotonic_valid"].value_counts().to_string())

sel = q[q["context_gate"] == 1].copy()

print("\nSELECTED CONTEXT TARGETS =", len(sel))
print(
    "monotonic valid =",
    int(sel["monotonic_valid"].sum()),
)
print(
    "positive adaptive boundary =",
    int(sel["adaptive_boundary_positive"].sum()),
)
print(
    "requires actual reduction =",
    int(sel["adaptive_requires_reduction"].sum()),
)

good = sel[
    sel["adaptive_requires_reduction"]
].copy()

if len(good):

    print("\nADAPTIVE SEVERITY BOUNDARY:")
    print(
        good["adaptive_severity_boundary"]
        .describe(
            percentiles=[.05, .25, .5, .75, .95]
        )
        .to_string()
    )

    print("\nREQUIRED TARGET / ORIGINAL SEVERITY:")
    print(
        good["required_severity_ratio"]
        .describe(
            percentiles=[.05, .25, .5, .75, .95]
        )
        .to_string()
    )

    print("\nBY GATE GROUP:")
    print(
        good.groupby("gate_group")
        .agg(
            n=("clip", "size"),
            median_original_severity=(
                "original_severity_explicit",
                "median",
            ),
            median_adaptive_boundary=(
                "adaptive_severity_boundary",
                "median",
            ),
            median_required_ratio=(
                "required_severity_ratio",
                "median",
            ),
        )
        .to_string()
    )

    print("\nBY TASK:")
    print(
        good.groupby("task")
        .agg(
            n=("clip", "size"),
            median_adaptive_boundary=(
                "adaptive_severity_boundary",
                "median",
            ),
            median_required_ratio=(
                "required_severity_ratio",
                "median",
            ),
        )
        .to_string()
    )

q.to_csv(OUT, index=False)

print("\nOUTPUT =", OUT)
print("SUCCESS = True")
