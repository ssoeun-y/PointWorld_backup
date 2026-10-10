from pathlib import Path
import csv
import json
import numpy as np
from scipy.spatial.distance import pdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(
    "diagnostics/results/"
    "behavior_task0001_soda_can_gt_top5_structural/captures"
)

OUT = Path(
    "diagnostics/results/"
    "behavior_soda_can_identity_sanity"
)
OUT.mkdir(parents=True, exist_ok=True)


CASES = [
    (
        1,
        ROOT / "rank1_task-0001_episode_00012190-460_471.npz"
    ),
    (
        3,
        ROOT / "rank3_task-0001_episode_00012100-455_466.npz"
    ),
    (
        4,
        ROOT / "rank4_task-0001_episode_00012150-415_426.npz"
    ),
    (
        5,
        ROOT / "rank5_task-0001_episode_00012210-480_491.npz"
    ),
]


# ============================================================
# EXACT SAME STRUCTURAL METRICS
# ============================================================

def pair_change_mm(x0, xt):
    d0 = pdist(x0.astype(np.float64))
    dt = pdist(xt.astype(np.float64))

    return np.abs(dt - d0) * 1000.0


def pair_p95_mm(x0, xt):
    d = pair_change_mm(x0, xt)

    if len(d) == 0:
        return 0.0

    return float(
        np.percentile(d, 95)
    )


def rigid_residual_mm(x0, xt):
    """
    Same Kabsch best-fit rigid transform
    used in structural analysis.
    """

    x0 = x0.astype(np.float64)
    xt = xt.astype(np.float64)

    c0 = x0.mean(axis=0)
    ct = xt.mean(axis=0)

    X = x0 - c0
    Y = xt - ct

    H = X.T @ Y

    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    aligned = X @ R.T + ct

    return (
        np.linalg.norm(
            aligned - xt,
            axis=-1,
        )
        * 1000.0
    )


def se3_p95_mm(x0, xt):
    r = rigid_residual_mm(
        x0,
        xt,
    )

    return float(
        np.percentile(r, 95)
    )


# ============================================================
# AUDIT
# ============================================================

print("=" * 78)
print("IDENTITY SANITY TEST")
print("=" * 78)

rows = []

for rank, path in CASES:

    print()
    print("#" * 78)
    print(f"RANK {rank}")
    print("#" * 78)

    print("file =", path)
    print("exists =", path.exists())

    if not path.exists():
        raise RuntimeError(
            f"Missing capture: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as z:

        gt = z["gt"].copy()
        pred = z["pred"].copy()

    print("shape =", gt.shape)

    if gt.shape != pred.shape:
        raise RuntimeError(
            f"GT/PRED shape mismatch rank {rank}"
        )

    # --------------------------------------------------------
    # Normal model prediction
    # --------------------------------------------------------

    pred_pair = pair_p95_mm(
        pred[0],
        pred[-1],
    )

    pred_se3 = se3_p95_mm(
        pred[0],
        pred[-1],
    )

    # --------------------------------------------------------
    # IDENTITY CONTROL:
    #
    # Pretend prediction is exactly GT.
    #
    # IMPORTANT:
    # We run the SAME t0 -> t10 structural metric.
    # Therefore result should equal GT's tiny
    # numerical rigid residual, not literally bitwise zero.
    # --------------------------------------------------------

    identity_pred = gt.copy()

    identity_pair = pair_p95_mm(
        identity_pred[0],
        identity_pred[-1],
    )

    identity_se3 = se3_p95_mm(
        identity_pred[0],
        identity_pred[-1],
    )

    # Direct GT computation is an additional
    # exact consistency check.
    direct_gt_pair = pair_p95_mm(
        gt[0],
        gt[-1],
    )

    direct_gt_se3 = se3_p95_mm(
        gt[0],
        gt[-1],
    )

    identity_pair_exact = (
        identity_pair
        == direct_gt_pair
    )

    identity_se3_exact = (
        identity_se3
        == direct_gt_se3
    )

    print(
        "GT direct pair P95      =",
        direct_gt_pair,
        "mm",
    )

    print(
        "IDENTITY pair P95       =",
        identity_pair,
        "mm",
    )

    print(
        "MODEL PRED pair P95     =",
        pred_pair,
        "mm",
    )

    print()

    print(
        "GT direct SE3 P95       =",
        direct_gt_se3,
        "mm",
    )

    print(
        "IDENTITY SE3 P95        =",
        identity_se3,
        "mm",
    )

    print(
        "MODEL PRED SE3 P95      =",
        pred_se3,
        "mm",
    )

    print()

    print(
        "IDENTITY_PAIR_EXACT_GT =",
        identity_pair_exact,
    )

    print(
        "IDENTITY_SE3_EXACT_GT =",
        identity_se3_exact,
    )

    # Conservative sanity threshold:
    # GT here should be far below 0.01 mm.
    identity_near_zero = (
        identity_pair < 0.01
        and identity_se3 < 0.01
    )

    print(
        "IDENTITY_NEAR_ZERO =",
        identity_near_zero,
    )

    ratio_pair = (
        pred_pair
        / max(identity_pair, 1e-12)
    )

    ratio_se3 = (
        pred_se3
        / max(identity_se3, 1e-12)
    )

    print(
        "PRED_vs_IDENTITY_pair_ratio =",
        ratio_pair,
    )

    print(
        "PRED_vs_IDENTITY_se3_ratio =",
        ratio_se3,
    )

    rows.append({
        "rank": rank,
        "n_points": gt.shape[1],

        "identity_pair_p95_mm":
            identity_pair,

        "model_pred_pair_p95_mm":
            pred_pair,

        "identity_se3_p95_mm":
            identity_se3,

        "model_pred_se3_p95_mm":
            pred_se3,

        "identity_pair_exact_gt":
            identity_pair_exact,

        "identity_se3_exact_gt":
            identity_se3_exact,

        "identity_near_zero":
            identity_near_zero,

        "pred_identity_pair_ratio":
            ratio_pair,

        "pred_identity_se3_ratio":
            ratio_se3,
    })


# ============================================================
# GLOBAL AUDIT
# ============================================================

all_pair_exact = all(
    r["identity_pair_exact_gt"]
    for r in rows
)

all_se3_exact = all(
    r["identity_se3_exact_gt"]
    for r in rows
)

all_near_zero = all(
    r["identity_near_zero"]
    for r in rows
)


print()
print("=" * 78)
print("GLOBAL RESULT")
print("=" * 78)

print(
    "ALL_IDENTITY_PAIR_EXACT_GT =",
    all_pair_exact,
)

print(
    "ALL_IDENTITY_SE3_EXACT_GT =",
    all_se3_exact,
)

print(
    "ALL_IDENTITY_NEAR_ZERO =",
    all_near_zero,
)


# ============================================================
# CSV
# ============================================================

csv_path = OUT / (
    "soda_can_identity_sanity.csv"
)

with open(
    csv_path,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# SIMPLE PROFESSOR-FRIENDLY FIGURE
# ============================================================

ranks = np.array([
    r["rank"]
    for r in rows
])

identity = np.array([
    r["identity_se3_p95_mm"]
    for r in rows
])

model = np.array([
    r["model_pred_se3_p95_mm"]
    for r in rows
])

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    ranks,
    identity,
    marker="o",
    label="Identity control (Pred = GT)",
)

ax.plot(
    ranks,
    model,
    marker="o",
    label="PointWorld prediction",
)

ax.set_yscale("log")

ax.set_xticks(ranks)

ax.set_xlabel(
    "Episode-disjoint soda-can case"
)

ax.set_ylabel(
    "Best-fit SE(3) residual P95 (mm, log scale)"
)

ax.set_title(
    "Identity sanity check: "
    "same structural metric pipeline"
)

ax.grid(
    alpha=0.25,
    which="both",
)

ax.legend()

fig.tight_layout()

plot_path = OUT / (
    "PROFESSOR_identity_vs_prediction_se3.png"
)

fig.savefig(
    plot_path,
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FINAL
# ============================================================

summary = {
    "n_cases": len(rows),

    "all_identity_pair_exact_gt":
        all_pair_exact,

    "all_identity_se3_exact_gt":
        all_se3_exact,

    "all_identity_near_zero":
        all_near_zero,

    "rows": rows,
}

with open(
    OUT / "summary.json",
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


print()
print("CSV =", csv_path)
print("PLOT =", plot_path)

print()
print("=" * 78)
print("FINAL")
print("=" * 78)

print(
    "ALL_IDENTITY_PAIR_EXACT_GT =",
    all_pair_exact,
)

print(
    "ALL_IDENTITY_SE3_EXACT_GT =",
    all_se3_exact,
)

print(
    "ALL_IDENTITY_NEAR_ZERO =",
    all_near_zero,
)

print(
    "NO MODEL INFERENCE"
)

print(
    "NO MBNSF"
)

print(
    "IDENTITY_SANITY_SUCCESS =",
    (
        all_pair_exact
        and all_se3_exact
        and all_near_zero
    ),
)
