from pathlib import Path
import csv
import json
import random

import numpy as np
import torch
from scipy.spatial.distance import pdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arguments import parse_args
from evaluation.tester import Tester


# ============================================================
# CONFIG
# ============================================================

IDENTITY_CSV = Path(
    "diagnostics/results/"
    "behavior_soda_can_identity_sanity/"
    "soda_can_identity_sanity.csv"
)

ROBUST_ROOT = Path(
    "diagnostics/results/"
    "behavior_task0001_soda_can_multiseed_robustness"
)

CASE_CSV = (
    ROBUST_ROOT
    / "soda_can_multiseed_case_summary.csv"
)

RUNS_CSV = (
    ROBUST_ROOT
    / "soda_can_multiseed_all_runs.csv"
)

CAPTURE = Path(
    "diagnostics/results/"
    "behavior_task0001_soda_can_gt_top5_structural/"
    "captures/"
    "rank1_task-0001_episode_00012190-460_471.npz"
)

OUT = Path(
    "diagnostics/results/"
    "behavior_soda_can_final_visuals"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


TARGET_KEY = (
    "task-0001_episode_00012190-460:471"
)

SEEDS = [42, 43, 44, 45, 46]
REPEATS = 2


# Explicit visualization semantics.
GT_COLOR = "#1f77b4"
PRED_COLOR = "#d62728"
GRAY = "0.50"


# ============================================================
# BASIC HELPERS
# ============================================================

def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def pair_p95_mm(x0, xt):
    d0 = pdist(
        x0.astype(np.float64)
    )

    dt = pdist(
        xt.astype(np.float64)
    )

    d = np.abs(
        dt - d0
    ) * 1000.0

    if len(d) == 0:
        return 0.0

    return float(
        np.percentile(
            d,
            95,
        )
    )


def kabsch_align(
    source,
    target,
):
    """
    Best rigid alignment:
        source -> target

    Used for rigid-motion removal.
    """

    source = source.astype(
        np.float64
    )

    target = target.astype(
        np.float64
    )

    cs = source.mean(axis=0)
    ct = target.mean(axis=0)

    X = source - cs
    Y = target - ct

    H = X.T @ Y

    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    aligned = (
        X @ R.T
        + ct
    )

    return aligned


def se3_residual_mm(
    x0,
    xt,
):
    """
    Structural metric:
    after best rigid transform,
    what residual remains?
    """

    aligned = kabsch_align(
        x0,
        xt,
    )

    return (
        np.linalg.norm(
            aligned
            - xt.astype(np.float64),
            axis=-1,
        )
        * 1000.0
    )


def se3_p95_mm(
    x0,
    xt,
):
    return float(
        np.percentile(
            se3_residual_mm(
                x0,
                xt,
            ),
            95,
        )
    )


# ============================================================
# LOAD TABLES
# ============================================================

print("=" * 80)
print("1. SOURCE FILE AUDIT")
print("=" * 80)

required = [
    IDENTITY_CSV,
    CASE_CSV,
    RUNS_CSV,
    CAPTURE,
]

for p in required:
    print(
        p,
        "exists =",
        p.exists(),
    )

if not all(
    p.exists()
    for p in required
):
    raise RuntimeError(
        "One or more required source files are missing."
    )


identity_rows = {}

with open(
    IDENTITY_CSV,
    newline="",
) as f:

    for r in csv.DictReader(f):

        identity_rows[
            int(r["rank"])
        ] = r


case_rows = {}

with open(
    CASE_CSV,
    newline="",
) as f:

    for r in csv.DictReader(f):

        case_rows[
            int(r["rank"])
        ] = r


run_rows = []

with open(
    RUNS_CSV,
    newline="",
) as f:

    for r in csv.DictReader(f):
        run_rows.append(r)


OBS_RANKS = [1, 3, 4, 5]

print(
    "identity ranks =",
    sorted(identity_rows),
)

print(
    "robust ranks =",
    sorted(case_rows),
)


# ============================================================
# FIGURE 1:
# IDENTITY SANITY
# ============================================================

print()
print("=" * 80)
print("2. IDENTITY SANITY FIGURE")
print("=" * 80)

identity_se3 = np.array([
    float(
        identity_rows[r][
            "identity_se3_p95_mm"
        ]
    )
    for r in OBS_RANKS
])

example_pred_se3 = np.array([
    float(
        identity_rows[r][
            "model_pred_se3_p95_mm"
        ]
    )
    for r in OBS_RANKS
])


fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)

ax.plot(
    OBS_RANKS,
    identity_se3,
    marker="o",
    linewidth=2,
    color=GT_COLOR,
    label="Identity control (Pred = GT)",
)

ax.plot(
    OBS_RANKS,
    example_pred_se3,
    marker="o",
    linewidth=2,
    color=PRED_COLOR,
    label="Example PointWorld forward",
)

ax.set_yscale("log")

ax.set_xticks(
    OBS_RANKS
)

ax.set_xlabel(
    "Episode-disjoint soda-can case"
)

ax.set_ylabel(
    "Best-fit SE(3) residual P95 (mm, log scale)"
)

ax.set_title(
    "Identity sanity check\n"
    "Same point correspondence and same structural metric"
)

ax.grid(
    alpha=0.25,
    which="both",
)

ax.legend()

fig.text(
    0.5,
    0.01,
    (
        "Pred = GT stays at numerical-zero scale; "
        "mm-scale residual appears only for PointWorld predictions."
    ),
    ha="center",
    fontsize=9,
)

fig.tight_layout(
    rect=[
        0,
        0.04,
        1,
        1,
    ]
)

identity_png = (
    OUT
    / "01_IDENTITY_SANITY.png"
)

fig.savefig(
    identity_png,
    dpi=240,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "saved =",
    identity_png,
)


# ============================================================
# FIGURE 2:
# MULTI-SEED ROBUSTNESS
# ============================================================

print()
print("=" * 80)
print("3. MULTI-SEED ROBUSTNESS FIGURE")
print("=" * 80)

gt = np.array([
    float(
        case_rows[r][
            "gt_se3_p95_t10_mm"
        ]
    )
    for r in OBS_RANKS
])

pred_min = np.array([
    float(
        case_rows[r][
            "pred_se3_min_mm"
        ]
    )
    for r in OBS_RANKS
])

pred_med = np.array([
    float(
        case_rows[r][
            "pred_se3_median_mm"
        ]
    )
    for r in OBS_RANKS
])

pred_max = np.array([
    float(
        case_rows[r][
            "pred_se3_max_mm"
        ]
    )
    for r in OBS_RANKS
])


lower = (
    pred_med - pred_min
)

upper = (
    pred_max - pred_med
)


fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)

ax.errorbar(
    OBS_RANKS,
    pred_med,
    yerr=np.vstack([
        lower,
        upper,
    ]),
    fmt="o",
    markersize=7,
    capsize=6,
    linewidth=2,
    color=PRED_COLOR,
    label=(
        "PointWorld "
        "(median, min–max)"
    ),
)

ax.plot(
    OBS_RANKS,
    gt,
    marker="o",
    linewidth=2,
    color=GT_COLOR,
    label="GT",
)

ax.set_yscale("log")

ax.set_xticks(
    OBS_RANKS
)

ax.set_xlabel(
    "Episode-disjoint GT-motion rank"
)

ax.set_ylabel(
    "Best-fit SE(3) residual P95 (mm, log scale)"
)

ax.set_title(
    "Rigid-object structural error is robust to stochastic inference\n"
    "5 seeds × 2 repeats per observable episode"
)

ax.grid(
    alpha=0.25,
    which="both",
)

ax.legend()

fig.text(
    0.5,
    0.01,
    (
        "Rank 2 is excluded here because official preprocessing "
        "contained zero soda-can points (N/A)."
    ),
    ha="center",
    fontsize=9,
)

fig.tight_layout(
    rect=[
        0,
        0.04,
        1,
        1,
    ]
)

robust_png = (
    OUT
    / "02_MULTISEED_ROBUSTNESS.png"
)

fig.savefig(
    robust_png,
    dpi=240,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "saved =",
    robust_png,
)


# ============================================================
# RANK-1 EXACT OBJECT MAPPING
# ============================================================

print()
print("=" * 80)
print("4. LOAD RANK-1 EXACT CAN INDICES")
print("=" * 80)

with np.load(
    CAPTURE,
    allow_pickle=False,
) as z:

    can_idx = (
        z["scene_indices"]
        .copy()
    )

    saved_gt = (
        z["gt"]
        .copy()
    )

print(
    "n_can_points =",
    len(can_idx),
)

print(
    "saved_gt_shape =",
    saved_gt.shape,
)


# ============================================================
# FIND TARGET MULTI-SEED MEDIAN
# ============================================================

target_median = float(
    case_rows[1][
        "pred_se3_median_mm"
    ]
)

print(
    "target multi-seed median SE3 =",
    target_median,
    "mm",
)


# ============================================================
# OFFICIAL POINTWORLD
# RERUN RANK1 ONLY
# ============================================================

print()
print("=" * 80)
print("5. REGENERATE A MEDIAN-NEAR REPRESENTATIVE RUN")
print("=" * 80)

args = parse_args()

tester = Tester(args)

dl, _ = tester._build_eval_loader(
    "test",
    enable_mask=False,
)

tester.model.eval()


target_cpu = None

for batch in dl:

    key = str(
        batch["__key__"][0]
    )

    if key == TARGET_KEY:

        target_cpu = {
            k: (
                v.detach().cpu().clone()
                if isinstance(
                    v,
                    torch.Tensor,
                )
                else v
            )
            for k, v in batch.items()
        }

        break


if target_cpu is None:
    raise RuntimeError(
        f"Target not found: {TARGET_KEY}"
    )


gt_full = (
    target_cpu[
        "gt_scene_flows"
    ][0]
    .numpy()
)

gt_can_now = (
    gt_full[
        :,
        can_idx,
        :
    ]
)

GT_CAPTURE_EXACT = (
    np.array_equal(
        gt_can_now,
        saved_gt,
    )
)

print(
    "GT_CAPTURE_EXACT =",
    GT_CAPTURE_EXACT,
)

if not GT_CAPTURE_EXACT:
    raise RuntimeError(
        "GT mapping changed."
    )


target_gpu = {
    k: (
        v.to(
            tester.device,
            non_blocking=True,
        )
        if isinstance(
            v,
            torch.Tensor,
        )
        else v
    )
    for k, v in target_cpu.items()
}


candidates = []

for seed in SEEDS:

    for rep in range(
        REPEATS
    ):

        reset_rng(seed)

        b = {
            k: (
                v.clone()
                if isinstance(
                    v,
                    torch.Tensor,
                )
                else v
            )
            for k, v
            in target_gpu.items()
        }

        with torch.inference_mode():

            out = tester.model(
                b,
                training=False,
            )

            _, loss_dict = (
                tester.model.loss_fn(
                    out,
                    b,
                    training=False,
                )
            )


        pred_full = (
            out[
                "scene_flows"
            ][0]
            .detach()
            .cpu()
            .numpy()
        )

        gt_run = (
            b[
                "gt_scene_flows"
            ][0]
            .detach()
            .cpu()
            .numpy()
        )

        pred_can = (
            pred_full[
                :,
                can_idx,
                :
            ]
        )

        gt_can = (
            gt_run[
                :,
                can_idx,
                :
            ]
        )


        if not np.array_equal(
            gt_can,
            saved_gt,
        ):
            raise RuntimeError(
                "GT changed during rerun."
            )


        # Official L2 reproduction.
        context = (
            b[
                "scene_context_mask"
            ]
        )

        exists = (
            b[
                "scene_exists"
            ]
        )

        supervised = (
            b[
                "scene_supervised_mask"
            ]
        )

        if context.ndim == 4:
            context = (
                context.squeeze(-1)
            )

        if exists.ndim == 4:
            exists = (
                exists.squeeze(-1)
            )

        if supervised.ndim == 4:
            supervised = (
                supervised.squeeze(-1)
            )

        valid = (
            (~context.bool())
            & exists.bool()
            & supervised.bool()
        )

        manual_l2 = (
            torch.linalg.vector_norm(
                out[
                    "scene_flows"
                ]
                - b[
                    "gt_scene_flows"
                ],
                dim=-1,
            )[valid]
            .mean()
            .item()
        )

        official_l2 = float(
            loss_dict[
                "l2/mean"
            ]
        )

        l2_ok = bool(
            np.isclose(
                manual_l2,
                official_l2,
                rtol=0.0,
                atol=1e-8,
            )
        )

        if not l2_ok:
            raise RuntimeError(
                "Official L2 mismatch."
            )


        se3_t10 = (
            se3_p95_mm(
                pred_can[0],
                pred_can[-1],
            )
        )

        pair_t10 = (
            pair_p95_mm(
                pred_can[0],
                pred_can[-1],
            )
        )

        distance_to_median = abs(
            se3_t10
            - target_median
        )


        candidates.append({
            "seed":
                seed,

            "repeat":
                rep,

            "se3_t10":
                se3_t10,

            "pair_t10":
                pair_t10,

            "distance_to_median":
                distance_to_median,

            "pred":
                pred_can.copy(),

            "gt":
                gt_can.copy(),

            "official_l2":
                official_l2,
        })


        print(
            f"seed={seed} "
            f"rep={rep} | "
            f"SE3={se3_t10:.3f} mm | "
            f"pair={pair_t10:.3f} mm | "
            f"|SE3-median|="
            f"{distance_to_median:.3f}"
        )


representative = min(
    candidates,
    key=lambda x:
        x[
            "distance_to_median"
        ],
)


REP_SEED = (
    representative[
        "seed"
    ]
)

REP_REPEAT = (
    representative[
        "repeat"
    ]
)

REP_SE3 = (
    representative[
        "se3_t10"
    ]
)

REP_PAIR = (
    representative[
        "pair_t10"
    ]
)


print()
print(
    "REPRESENTATIVE_SEED =",
    REP_SEED,
)

print(
    "REPRESENTATIVE_REPEAT =",
    REP_REPEAT,
)

print(
    "REPRESENTATIVE_SE3_P95_MM =",
    REP_SE3,
)

print(
    "TARGET_MULTI_SEED_MEDIAN_MM =",
    target_median,
)

print(
    "DISTANCE_TO_MEDIAN_MM =",
    representative[
        "distance_to_median"
    ],
)


rep_npz = (
    OUT
    / "03_REPRESENTATIVE_RANK1_RUN.npz"
)

np.savez_compressed(
    rep_npz,

    gt=representative[
        "gt"
    ],

    pred=representative[
        "pred"
    ],

    scene_indices=can_idx,

    seed=np.array(
        REP_SEED
    ),

    repeat=np.array(
        REP_REPEAT
    ),

    se3_t10_mm=np.array(
        REP_SE3
    ),

    pair_t10_mm=np.array(
        REP_PAIR
    ),

    target_multiseed_median_mm=np.array(
        target_median
    ),
)


# ============================================================
# CANONICAL CAN FRAME
# ============================================================

def build_can_frame(
    gt0,
):
    """
    Visualization-only PCA frame.
    """

    gt0 = gt0.astype(
        np.float64
    )

    center = (
        gt0.mean(axis=0)
    )

    X = (
        gt0 - center
    )

    cov = (
        X.T @ X
        / max(
            len(X) - 1,
            1,
        )
    )

    eigvals, eigvecs = (
        np.linalg.eigh(cov)
    )

    projected = (
        X @ eigvecs
    )

    spans = []

    for j in range(3):

        spans.append(
            np.percentile(
                projected[:, j],
                98,
            )
            -
            np.percentile(
                projected[:, j],
                2,
            )
        )

    height_idx = int(
        np.argmax(spans)
    )

    radial_idx = [
        x
        for x in range(3)
        if x != height_idx
    ]

    basis = np.column_stack([
        eigvecs[
            :,
            radial_idx[0]
        ],

        eigvecs[
            :,
            radial_idx[1]
        ],

        eigvecs[
            :,
            height_idx
        ],
    ])

    if np.linalg.det(
        basis
    ) < 0:

        basis[:, 0] *= -1

    return (
        center,
        basis,
    )


def canonicalize(
    points,
    center,
    basis,
):
    return (
        points.astype(
            np.float64
        )
        - center
    ) @ basis


def fit_cylinder(
    gt0_can,
):

    xy_center = np.median(
        gt0_can[:, :2],
        axis=0,
    )

    radius_values = (
        np.linalg.norm(
            gt0_can[:, :2]
            - xy_center[
                None,
                :
            ],
            axis=-1,
        )
    )

    radius = float(
        np.percentile(
            radius_values,
            95,
        )
    )

    z0 = float(
        np.percentile(
            gt0_can[:, 2],
            2,
        )
    )

    z1 = float(
        np.percentile(
            gt0_can[:, 2],
            98,
        )
    )

    zc = (
        0.5
        * (z0 + z1)
    )

    return {
        "xy":
            xy_center,

        "r":
            radius,

        "z0":
            z0,

        "z1":
            z1,

        "zc":
            zc,
    }


def shift_visual(
    p,
    cyl,
):
    return (
        p
        - np.array([
            cyl["xy"][0],
            cyl["xy"][1],
            cyl["zc"],
        ])
    )


def draw_cylinder(
    ax,
    cyl,
):
    r = cyl["r"]

    z0 = (
        cyl["z0"]
        - cyl["zc"]
    )

    z1 = (
        cyl["z1"]
        - cyl["zc"]
    )

    theta = np.linspace(
        0,
        2 * np.pi,
        64,
    )

    zs = np.linspace(
        z0,
        z1,
        8,
    )

    TH, ZZ = np.meshgrid(
        theta,
        zs,
    )

    XX = (
        r * np.cos(TH)
    )

    YY = (
        r * np.sin(TH)
    )

    # Very light translucent body.
    ax.plot_surface(
        XX,
        YY,
        ZZ,
        color="gray",
        alpha=0.025,
        linewidth=0,
        shade=False,
    )

    ax.plot_wireframe(
        XX,
        YY,
        ZZ,
        rstride=1,
        cstride=8,
        color=GRAY,
        alpha=0.27,
        linewidth=0.5,
    )

    for z in [z0, z1]:

        ax.plot(
            r * np.cos(theta),
            r * np.sin(theta),
            np.full_like(
                theta,
                z,
            ),
            color=GRAY,
            alpha=0.70,
            linewidth=1.0,
        )


def set_axes(
    ax,
    cyl,
    points,
):

    radial = max(
        cyl["r"] * 1.55,
        float(
            np.max(
                np.abs(
                    points[:, :2]
                )
            )
        ) * 1.10,
    )

    half_h = max(
        (
            cyl["z1"]
            - cyl["z0"]
        ) * 0.60,
        float(
            np.max(
                np.abs(
                    points[:, 2]
                )
            )
        ) * 1.10,
    )

    ax.set_xlim(
        -radial,
        radial,
    )

    ax.set_ylim(
        -radial,
        radial,
    )

    ax.set_zlim(
        -half_h,
        half_h,
    )

    ax.set_box_aspect([
        2 * radial,
        2 * radial,
        2 * half_h,
    ])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


# ============================================================
# FIGURE 3:
# REPRESENTATIVE t0 -> t10 DEFORMATION
# ============================================================

print()
print("=" * 80)
print("6. BUILD t0 -> t10 DEFORMATION VISUAL")
print("=" * 80)

gt_rep = (
    representative[
        "gt"
    ]
)

pred_rep = (
    representative[
        "pred"
    ]
)


center0, basis0 = (
    build_can_frame(
        gt_rep[0]
    )
)

gt0_can = canonicalize(
    gt_rep[0],
    center0,
    basis0,
)

cyl = fit_cylinder(
    gt0_can
)


gt_frames = []
pred_frames = []

se3_curve = []
pair_curve = []


for t in range(
    gt_rep.shape[0]
):

    # Remove the real GT rigid motion.
    gt_aligned = kabsch_align(
        gt_rep[t],
        gt_rep[0],
    )

    # Remove ALL possible rigid motion
    # from the PointWorld prediction.
    pred_aligned = kabsch_align(
        pred_rep[t],
        gt_rep[0],
    )

    gt_can = canonicalize(
        gt_aligned,
        center0,
        basis0,
    )

    pred_can = canonicalize(
        pred_aligned,
        center0,
        basis0,
    )

    gt_can = shift_visual(
        gt_can,
        cyl,
    )

    pred_can = shift_visual(
        pred_can,
        cyl,
    )

    gt_frames.append(
        gt_can
    )

    pred_frames.append(
        pred_can
    )

    se3_curve.append(
        se3_p95_mm(
            pred_rep[0],
            pred_rep[t],
        )
    )

    pair_curve.append(
        pair_p95_mm(
            pred_rep[0],
            pred_rep[t],
        )
    )


se3_curve = np.asarray(
    se3_curve
)

pair_curve = np.asarray(
    pair_curve
)


peak_t = int(
    np.argmax(
        se3_curve
    )
)

peak_se3 = float(
    se3_curve[
        peak_t
    ]
)

final_se3 = float(
    se3_curve[-1]
)

recovery = (
    max(
        0.0,
        peak_se3
        - final_se3,
    )
    / peak_se3
    if peak_se3 > 0
    else 0.0
)


fig = plt.figure(
    figsize=(16, 11)
)

fig.suptitle(
    (
        "PointWorld soda-can deformation over time — representative run\n"
        f"Rank 1 | seed={REP_SEED}, repeat={REP_REPEAT} | "
        f"t10 SE(3) P95={REP_SE3:.2f} mm | "
        f"multi-seed median={target_median:.2f} mm\n"
        "All rigid translation/rotation removed before visualization"
    ),
    fontsize=14,
)


for t in range(11):

    ax = fig.add_subplot(
        3,
        4,
        t + 1,
        projection="3d",
    )

    gt_t = (
        gt_frames[t]
    )

    pred_t = (
        pred_frames[t]
    )

    draw_cylinder(
        ax,
        cyl,
    )

    ax.scatter(
        gt_t[:, 0],
        gt_t[:, 1],
        gt_t[:, 2],
        s=10,
        color=GT_COLOR,
        alpha=0.35,
    )

    ax.scatter(
        pred_t[:, 0],
        pred_t[:, 1],
        pred_t[:, 2],
        s=24,
        color=PRED_COLOR,
        alpha=0.92,
    )

    # Show the largest correspondence residuals.
    residual = (
        np.linalg.norm(
            pred_t
            - gt_t,
            axis=-1,
        )
    )

    top = np.argsort(
        residual
    )[
        -min(
            6,
            len(residual),
        ):
    ]

    for i in top:

        ax.plot(
            [
                gt_t[i, 0],
                pred_t[i, 0],
            ],
            [
                gt_t[i, 1],
                pred_t[i, 1],
            ],
            [
                gt_t[i, 2],
                pred_t[i, 2],
            ],
            color="0.65",
            linewidth=0.7,
            alpha=0.60,
        )

    all_pts = np.concatenate([
        gt_t,
        pred_t,
    ])

    set_axes(
        ax,
        cyl,
        all_pts,
    )

    ax.view_init(
        elev=18,
        azim=-60,
    )

    ax.set_title(
        (
            f"t{t}\n"
            f"SE3 P95 "
            f"{se3_curve[t]:.1f} mm"
        ),
        fontsize=9,
    )


# Explanation panel.
ax = fig.add_subplot(
    3,
    4,
    12,
)

ax.axis("off")

ax.text(
    0.03,
    0.86,
    "Blue: rigid GT",
    color=GT_COLOR,
    fontsize=11,
)

ax.text(
    0.03,
    0.74,
    "Red: PointWorld",
    color=PRED_COLOR,
    fontsize=11,
)

ax.text(
    0.03,
    0.61,
    (
        "Gray cylinder:\n"
        "GT-fitted visual guide only"
    ),
    color=GRAY,
    fontsize=10,
)

ax.text(
    0.03,
    0.38,
    (
        f"Peak deformation:\n"
        f"{peak_se3:.2f} mm @ t{peak_t}\n\n"
        f"Final:\n"
        f"{final_se3:.2f} mm\n\n"
        f"Recovery:\n"
        f"{100*recovery:.1f}%"
    ),
    fontsize=10,
)


fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.93,
    ]
)

deformation_png = (
    OUT
    / "03_RANK1_DEFORMATION_T0_T10.png"
)

fig.savefig(
    deformation_png,
    dpi=240,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "saved =",
    deformation_png,
)


# ============================================================
# FIGURE 4:
# CURRENT TOP-5 STATUS
# 4 MEASURED + RANK2 N/A
# ============================================================

print()
print("=" * 80)
print("7. BUILD CURRENT TOP-5 STATUS")
print("=" * 80)


fig, ax = plt.subplots(
    figsize=(9, 5.8)
)


y_for_rank = {
    1: 5,
    2: 4,
    3: 3,
    4: 2,
    5: 1,
}


for r in OBS_RANKS:

    y = y_for_rank[r]

    s = case_rows[r]

    g = float(
        s[
            "gt_se3_p95_t10_mm"
        ]
    )

    lo = float(
        s[
            "pred_se3_min_mm"
        ]
    )

    med = float(
        s[
            "pred_se3_median_mm"
        ]
    )

    hi = float(
        s[
            "pred_se3_max_mm"
        ]
    )

    ax.errorbar(
        med,
        y,
        xerr=[
            [med - lo],
            [hi - med],
        ],
        fmt="o",
        capsize=5,
        markersize=7,
        color=PRED_COLOR,
    )

    ax.scatter(
        [g],
        [y],
        s=50,
        color=GT_COLOR,
        zorder=5,
    )


# Rank 2 = genuinely N/A.
ax.text(
    2e-3,
    y_for_rank[2],
    (
        "N/A — zero soda-can points "
        "after official preprocessing"
    ),
    va="center",
    fontsize=10,
)


ax.set_xscale("log")

ax.set_xlim(
    3e-5,
    30,
)

ax.set_ylim(
    0.4,
    5.6,
)

ax.set_yticks([
    5,
    4,
    3,
    2,
    1,
])

ax.set_yticklabels([
    "Rank 1",
    "Rank 2",
    "Rank 3",
    "Rank 4",
    "Rank 5",
])

ax.set_xlabel(
    "Best-fit SE(3) residual P95 (mm, log scale)"
)

ax.set_title(
    "GT-selected soda-can Top-5 status\n"
    "4 measurable cases + 1 N/A (not yet final 5/5)"
)

ax.grid(
    axis="x",
    alpha=0.25,
    which="both",
)

# Dummy legend marks.
ax.scatter(
    [],
    [],
    color=GT_COLOR,
    label="GT",
)

ax.errorbar(
    [],
    [],
    xerr=[],
    fmt="o",
    color=PRED_COLOR,
    label=(
        "PointWorld median "
        "(min–max over 10 runs)"
    ),
)

ax.legend(
    loc="lower right",
)

status_png = (
    OUT
    / "04_TOP5_STATUS_4_MEASURED_1_NA.png"
)

fig.tight_layout()

fig.savefig(
    status_png,
    dpi=240,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "saved =",
    status_png,
)


# ============================================================
# TEXT REPORT
# ============================================================

max_gt = max(
    float(
        case_rows[r][
            "gt_se3_p95_t10_mm"
        ]
    )
    for r in OBS_RANKS
)

min_pred = min(
    float(
        case_rows[r][
            "pred_se3_min_mm"
        ]
    )
    for r in OBS_RANKS
)


report = f"""
PointWorld Soda-Can Structural Integrity — Current Evidence

[1] Identity sanity
- 4/4 observable cases:
  Pred = GT control exactly reproduces the direct GT structural metric.
- Identity SE(3) residual is approximately 1e-4 mm.
- Therefore the structural metric does not turn a rigid GT trajectory
  into mm-scale deformation by itself.

[2] Multi-seed robustness
- 4 episode-disjoint observable cases.
- 5 seeds x 2 repeats = 40 PointWorld forwards.
- Worst GT SE(3) P95 among these cases:
  {max_gt:.9f} mm
- Best PointWorld SE(3) P95 among all tested cases/runs:
  {min_pred:.6f} mm
- The structural separation therefore remains even under the
  most favorable tested stochastic PointWorld forward.

[3] Representative deformation visualization
- Rank 1 was regenerated across 10 forwards.
- Multi-seed target median:
  {target_median:.6f} mm
- Selected representative:
  seed={REP_SEED}, repeat={REP_REPEAT}
- Exact selected-run t10 SE(3) P95:
  {REP_SE3:.6f} mm
- Peak structural deformation:
  {peak_se3:.6f} mm at timestep {peak_t}
- Final structural deformation:
  {final_se3:.6f} mm
- Peak-to-final recovery:
  {100*recovery:.2f} %

Interpretation:
PointWorld does not merely place the soda can at an incorrect global pose.
After allowing the prediction the best possible rigid translation and
rotation, the predicted object points still cannot be explained by one
rigid-body transform. The remaining error is therefore an internal
object-structure / rigidity-consistency error.

Current limitation:
- Only 4 of the original GT-selected Top-5 cases are measurable.
- Rank 2 contains zero soda-can points after official PointWorld
  preprocessing and is N/A, not success and not failure.
- These samples were selected for large GT motion, so this experiment
  establishes a repeatable failure mode, not population prevalence.

NEXT:
Replace Rank 2 outcome-blindly using the next episode-disjoint,
GT-ranked soda-can sample that has measurable PointWorld can points.
Only after that should the 5/5 final summary be produced.
""".strip()


report_path = (
    OUT
    / "REPORT_SUMMARY.txt"
)

report_path.write_text(
    report
)

print(
    "saved =",
    report_path,
)


# ============================================================
# MACHINE SUMMARY
# ============================================================

summary = {
    "identity_sanity_png":
        str(identity_png),

    "multiseed_robustness_png":
        str(robust_png),

    "deformation_png":
        str(deformation_png),

    "top5_status_png":
        str(status_png),

    "report":
        str(report_path),

    "representative_seed":
        REP_SEED,

    "representative_repeat":
        REP_REPEAT,

    "representative_se3_p95_mm":
        REP_SE3,

    "target_multiseed_median_mm":
        target_median,

    "representative_distance_to_median_mm":
        representative[
            "distance_to_median"
        ],

    "peak_timestep":
        peak_t,

    "peak_se3_p95_mm":
        peak_se3,

    "final_se3_p95_mm":
        final_se3,

    "recovery_fraction":
        recovery,

    "observable_cases":
        4,

    "rank2_status":
        "N/A_ZERO_CAN_POINTS",

    "true_5_of_5_final_ready":
        False,
}


with open(
    OUT
    / "summary.json",
    "w",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("FINAL")
print("=" * 80)

print(
    "GT_CAPTURE_EXACT =",
    GT_CAPTURE_EXACT,
)

print(
    "REPRESENTATIVE_SEED =",
    REP_SEED,
)

print(
    "REPRESENTATIVE_REPEAT =",
    REP_REPEAT,
)

print(
    "REPRESENTATIVE_SE3_P95_MM =",
    REP_SE3,
)

print(
    "TARGET_MULTI_SEED_MEDIAN_MM =",
    target_median,
)

print(
    "PEAK_TIMESTEP =",
    peak_t,
)

print(
    "PEAK_SE3_P95_MM =",
    peak_se3,
)

print(
    "FINAL_SE3_P95_MM =",
    final_se3,
)

print(
    "RECOVERY_FRACTION =",
    recovery,
)

print()
print(
    "IDENTITY =",
    identity_png,
)

print(
    "ROBUSTNESS =",
    robust_png,
)

print(
    "DEFORMATION =",
    deformation_png,
)

print(
    "TOP5_STATUS =",
    status_png,
)

print(
    "REPORT =",
    report_path,
)

print()
print(
    "TRUE_5_OF_5_FINAL_READY = False"
)

print(
    "NEXT_STEP = "
    "OUTCOME_BLIND_RANK2_REPLACEMENT"
)

print(
    "NO MBNSF"
)

print(
    "NO OFFICIAL FILE MODIFICATION"
)

print(
    "GPU_PHYSICAL = 1"
)

print(
    "FINAL_VISUALS_BUILD_SUCCESS = True"
)
