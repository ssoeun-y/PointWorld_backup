from pathlib import Path
import csv
import json
import numpy as np

from scipy.spatial.distance import pdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

STRUCT_ROOT = Path(
    "diagnostics/results/"
    "behavior_task0001_soda_can_gt_top5_structural"
)

CAPTURE_DIR = STRUCT_ROOT / "captures"

ROBUST_ROOT = Path(
    "diagnostics/results/"
    "behavior_task0001_soda_can_multiseed_robustness"
)

CASE_CSV = (
    ROBUST_ROOT
    / "soda_can_multiseed_case_summary.csv"
)

TIME_NPZ = (
    ROBUST_ROOT
    / "multiseed_time_curves.npz"
)

OUT = Path(
    "diagnostics/results/"
    "behavior_soda_can_professor_visuals"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FOUR OBSERVABLE CASES
# ============================================================

CASES = [
    {
        "rank": 1,
        "can_id": 114,
        "key":
            "task-0001_episode_00012190-460:471",
        "capture":
            "rank1_task-0001_episode_00012190-460_471.npz",
    },
    {
        "rank": 3,
        "can_id": 113,
        "key":
            "task-0001_episode_00012100-455:466",
        "capture":
            "rank3_task-0001_episode_00012100-455_466.npz",
    },
    {
        "rank": 4,
        "can_id": 113,
        "key":
            "task-0001_episode_00012150-415:426",
        "capture":
            "rank4_task-0001_episode_00012150-415_426.npz",
    },
    {
        "rank": 5,
        "can_id": 114,
        "key":
            "task-0001_episode_00012210-480:491",
        "capture":
            "rank5_task-0001_episode_00012210-480_491.npz",
    },
]


# ============================================================
# COLORS
# ============================================================

GT_COLOR = "#1f77b4"
PRED_COLOR = "#d62728"
REF_COLOR = "0.45"
LINE_COLOR = "0.55"


# ============================================================
# METRIC HELPERS
# ============================================================

def pair_p95_mm(x0, xt):

    d0 = pdist(
        x0.astype(np.float64)
    )

    dt = pdist(
        xt.astype(np.float64)
    )

    change = (
        np.abs(dt - d0)
        * 1000.0
    )

    if len(change) == 0:
        return 0.0

    return float(
        np.percentile(
            change,
            95,
        )
    )


def kabsch_transform(
    source,
    target,
):
    """
    Returns R,t such that

        source @ R.T + t

    best aligns source -> target.

    Rigid transform only.
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

    t = (
        ct
        - cs @ R.T
    )

    return R, t


def apply_transform(
    points,
    R,
    t,
):
    return (
        points.astype(np.float64)
        @ R.T
        + t
    )


def se3_residual_mm(
    source,
    target,
):
    """
    Residual after the BEST possible
    rigid transform.

    This is the actual structural metric.
    """

    R, t = kabsch_transform(
        source,
        target,
    )

    aligned = apply_transform(
        source,
        R,
        t,
    )

    residual = (
        np.linalg.norm(
            aligned
            - target.astype(
                np.float64
            ),
            axis=-1,
        )
        * 1000.0
    )

    return residual


def se3_p95_mm(
    source,
    target,
):
    return float(
        np.percentile(
            se3_residual_mm(
                source,
                target,
            ),
            95,
        )
    )


# ============================================================
# GT-DERIVED CANONICAL FRAME
# ============================================================

def canonical_can_frame(
    gt0,
):
    """
    Visualization only.

    PCA on GT t0.

    We choose the PCA axis having the
    largest robust spatial span as the
    approximate can-height axis.
    """

    gt0 = gt0.astype(
        np.float64
    )

    center = gt0.mean(
        axis=0
    )

    X = gt0 - center

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

    # Three PCA axes.
    projected = (
        X @ eigvecs
    )

    spans = []

    for j in range(3):

        lo = np.percentile(
            projected[:, j],
            2,
        )

        hi = np.percentile(
            projected[:, j],
            98,
        )

        spans.append(
            hi - lo
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

    # Right-handed frame.
    if np.linalg.det(
        basis
    ) < 0:

        basis[:, 0] *= -1

    return center, basis


def to_canonical(
    points,
    center,
    basis,
):
    return (
        points.astype(np.float64)
        - center
    ) @ basis


# ============================================================
# GT-FITTED CYLINDER
# ============================================================

def fit_visual_cylinder(
    gt0_canonical,
):
    """
    VISUALIZATION ONLY.

    Approximate cylinder from GT points.
    """

    xy_center = np.median(
        gt0_canonical[:, :2],
        axis=0,
    )

    radial = np.linalg.norm(
        gt0_canonical[:, :2]
        - xy_center[None, :],
        axis=-1,
    )

    radius = float(
        np.percentile(
            radial,
            95,
        )
    )

    z_lo = float(
        np.percentile(
            gt0_canonical[:, 2],
            2,
        )
    )

    z_hi = float(
        np.percentile(
            gt0_canonical[:, 2],
            98,
        )
    )

    z_center = (
        0.5
        * (z_lo + z_hi)
    )

    height = (
        z_hi - z_lo
    )

    return {
        "xy_center":
            xy_center,

        "radius":
            radius,

        "z_lo":
            z_lo,

        "z_hi":
            z_hi,

        "z_center":
            z_center,

        "height":
            height,

        "height_radius_ratio":
            (
                height
                / max(
                    radius,
                    1e-12,
                )
            ),
    }


def normalize_visual_frame(
    points,
    cylinder,
):
    """
    Put cylinder center at origin.
    """

    shift = np.array([
        cylinder[
            "xy_center"
        ][0],

        cylinder[
            "xy_center"
        ][1],

        cylinder[
            "z_center"
        ],
    ])

    return (
        points
        - shift
    )


def draw_cylinder(
    ax,
    cylinder,
):

    r = cylinder[
        "radius"
    ]

    z0 = (
        cylinder["z_lo"]
        - cylinder[
            "z_center"
        ]
    )

    z1 = (
        cylinder["z_hi"]
        - cylinder[
            "z_center"
        ]
    )

    theta = np.linspace(
        0,
        2 * np.pi,
        50,
    )

    zvals = np.linspace(
        z0,
        z1,
        7,
    )

    TH, ZZ = np.meshgrid(
        theta,
        zvals,
    )

    XX = (
        r * np.cos(TH)
    )

    YY = (
        r * np.sin(TH)
    )

    ax.plot_wireframe(
        XX,
        YY,
        ZZ,
        rstride=1,
        cstride=5,
        color=REF_COLOR,
        alpha=0.25,
        linewidth=0.6,
    )

    # top / bottom rings
    for zz in [z0, z1]:

        ax.plot(
            r * np.cos(theta),
            r * np.sin(theta),
            np.full_like(
                theta,
                zz,
            ),
            color=REF_COLOR,
            alpha=0.55,
            linewidth=1.0,
        )


def set_can_axes(
    ax,
    cylinder,
    extra_points=None,
):

    r = cylinder[
        "radius"
    ]

    half_h = (
        cylinder[
            "height"
        ]
        / 2.0
    )

    radial_extent = (
        r * 1.55
    )

    height_extent = (
        half_h * 1.20
    )

    if extra_points is not None:

        radial_from_data = float(
            np.max(
                np.abs(
                    extra_points[
                        :,
                        :2
                    ]
                )
            )
        )

        height_from_data = float(
            np.max(
                np.abs(
                    extra_points[
                        :,
                        2
                    ]
                )
            )
        )

        radial_extent = max(
            radial_extent,
            radial_from_data * 1.10,
        )

        height_extent = max(
            height_extent,
            height_from_data * 1.10,
        )

    ax.set_xlim(
        -radial_extent,
        radial_extent,
    )

    ax.set_ylim(
        -radial_extent,
        radial_extent,
    )

    ax.set_zlim(
        -height_extent,
        height_extent,
    )

    ax.set_box_aspect(
        (
            2 * radial_extent,
            2 * radial_extent,
            2 * height_extent,
        )
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


# ============================================================
# TEMPORAL VISUALIZATION
# ============================================================

def canonical_prediction_at_t(
    gt,
    pred,
    t,
    center0,
    basis0,
    cylinder,
):
    """
    IMPORTANT:

    1. Remove prediction global rigid error:
       pred_t -> gt_t via best-fit SE(3).

    2. Remove actual GT rigid motion:
       gt_t -> gt_0.

    3. Express result in GT-t0 canonical frame.

    Therefore what remains visually is
    NON-RIGID deformation.
    """

    gt0 = gt[0]

    gtt = gt[t]
    predt = pred[t]

    # Prediction -> GT at same timestep.
    Rp, tp = kabsch_transform(
        predt,
        gtt,
    )

    pred_to_gt = (
        apply_transform(
            predt,
            Rp,
            tp,
        )
    )

    # GT(t) -> GT(0).
    Rg, tg = kabsch_transform(
        gtt,
        gt0,
    )

    gt_to_ref = (
        apply_transform(
            gtt,
            Rg,
            tg,
        )
    )

    pred_to_ref = (
        apply_transform(
            pred_to_gt,
            Rg,
            tg,
        )
    )

    gt_can = to_canonical(
        gt_to_ref,
        center0,
        basis0,
    )

    pred_can = to_canonical(
        pred_to_ref,
        center0,
        basis0,
    )

    gt_can = (
        normalize_visual_frame(
            gt_can,
            cylinder,
        )
    )

    pred_can = (
        normalize_visual_frame(
            pred_can,
            cylinder,
        )
    )

    return (
        gt_can,
        pred_can,
    )


# ============================================================
# LOAD MULTI-SEED SUMMARY
# ============================================================

print("=" * 80)
print("1. LOAD MULTI-SEED RESULTS")
print("=" * 80)

case_summary = {}

with open(
    CASE_CSV,
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        rank = int(
            row["rank"]
        )

        case_summary[
            rank
        ] = row


print(
    "case_summary ranks =",
    sorted(
        case_summary
    ),
)


# ============================================================
# LOAD MULTI-SEED TIME CURVES
# ============================================================

time_data = np.load(
    TIME_NPZ,
    allow_pickle=False,
)


# ============================================================
# CASE PROCESSING
# ============================================================

visual_records = {}

for case in CASES:

    rank = case["rank"]

    path = (
        CAPTURE_DIR
        / case["capture"]
    )

    print()
    print("=" * 80)
    print(
        f"2. RANK {rank}"
    )
    print("=" * 80)

    print(
        "capture =",
        path,
    )

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

    n = gt.shape[1]

    print(
        "n_points =",
        n,
    )

    print(
        "T =",
        gt.shape[0],
    )

    # --------------------------------
    # Fixed GT-derived reference
    # --------------------------------

    center0, basis0 = (
        canonical_can_frame(
            gt[0]
        )
    )

    gt0_can = (
        to_canonical(
            gt[0],
            center0,
            basis0,
        )
    )

    cylinder = (
        fit_visual_cylinder(
            gt0_can
        )
    )

    print(
        "cylinder radius mm =",
        cylinder[
            "radius"
        ] * 1000.0,
    )

    print(
        "cylinder height mm =",
        cylinder[
            "height"
        ] * 1000.0,
    )

    print(
        "height/radius =",
        cylinder[
            "height_radius_ratio"
        ],
    )

    # --------------------------------
    # Generate all 11 canonical frames
    # --------------------------------

    gt_frames = []
    pred_frames = []

    se3_curve = []
    pair_curve = []

    for t in range(
        gt.shape[0]
    ):

        gt_can, pred_can = (
            canonical_prediction_at_t(
                gt,
                pred,
                t,
                center0,
                basis0,
                cylinder,
            )
        )

        gt_frames.append(
            gt_can
        )

        pred_frames.append(
            pred_can
        )

        se3_curve.append(
            se3_p95_mm(
                pred[t],
                gt[t],
            )
        )

        pair_curve.append(
            pair_p95_mm(
                pred[0],
                pred[t],
            )
        )


    gt_frames = np.asarray(
        gt_frames
    )

    pred_frames = np.asarray(
        pred_frames
    )

    se3_curve = np.asarray(
        se3_curve
    )

    pair_curve = np.asarray(
        pair_curve
    )

    # --------------------------------
    # VISUAL A:
    # 11-frame deformation sequence
    # --------------------------------

    fig = plt.figure(
        figsize=(16, 11)
    )

    fig.suptitle(
        (
            f"Soda can rank {rank} — "
            f"non-rigid deformation over time\n"
            f"{case['key']} | "
            f"n={n} points\n"
            "Prediction rigid motion removed; "
            "GT-fitted cylinder = visual reference only"
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

        draw_cylinder(
            ax,
            cylinder,
        )

        pred_t = (
            pred_frames[t]
        )

        gt_t = (
            gt_frames[t]
        )

        ax.scatter(
            pred_t[:, 0],
            pred_t[:, 1],
            pred_t[:, 2],
            s=18,
            color=PRED_COLOR,
            alpha=0.90,
        )

        # Small GT reference points.
        ax.scatter(
            gt_t[:, 0],
            gt_t[:, 1],
            gt_t[:, 2],
            s=6,
            color=GT_COLOR,
            alpha=0.35,
        )

        all_pts = np.concatenate([
            pred_t,
            gt_t,
        ])

        set_can_axes(
            ax,
            cylinder,
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


    # Final empty panel = legend/explanation
    ax_text = fig.add_subplot(
        3,
        4,
        12,
    )

    ax_text.axis("off")

    ax_text.text(
        0.05,
        0.80,
        "Blue: rigid GT reference",
        color=GT_COLOR,
        fontsize=11,
    )

    ax_text.text(
        0.05,
        0.66,
        "Red: PointWorld prediction",
        color=PRED_COLOR,
        fontsize=11,
    )

    ax_text.text(
        0.05,
        0.52,
        "Gray cylinder: approximate\nGT-fitted can shape",
        color=REF_COLOR,
        fontsize=11,
    )

    peak_t = int(
        np.argmax(
            se3_curve
        )
    )

    peak = float(
        se3_curve[
            peak_t
        ]
    )

    final = float(
        se3_curve[-1]
    )

    recovery = (
        max(
            0.0,
            peak - final,
        )
        / peak
        if peak > 0
        else 0.0
    )

    ax_text.text(
        0.05,
        0.30,
        (
            f"Peak: {peak:.2f} mm @ t{peak_t}\n"
            f"Final: {final:.2f} mm\n"
            f"Recovery: {recovery*100:.1f}%"
        ),
        fontsize=11,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.93,
        ]
    )

    montage_path = OUT / (
        f"rank{rank}_"
        "soda_can_deformation_t0_t10.png"
    )

    fig.savefig(
        montage_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


    # --------------------------------
    # VISUAL B:
    # t10 direct structural overlay
    # --------------------------------

    gt10 = (
        gt_frames[-1]
    )

    pred10 = (
        pred_frames[-1]
    )

    residual = (
        np.linalg.norm(
            pred10
            - gt10,
            axis=-1,
        )
        * 1000.0
    )

    # Draw only largest residuals
    # to keep the figure readable.
    n_lines = min(
        15,
        n,
    )

    top_idx = np.argsort(
        residual
    )[-n_lines:]


    fig = plt.figure(
        figsize=(8, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    draw_cylinder(
        ax,
        cylinder,
    )

    ax.scatter(
        gt10[:, 0],
        gt10[:, 1],
        gt10[:, 2],
        s=24,
        color=GT_COLOR,
        label="GT",
    )

    ax.scatter(
        pred10[:, 0],
        pred10[:, 1],
        pred10[:, 2],
        s=28,
        color=PRED_COLOR,
        label=(
            "Prediction "
            "(best rigid-aligned)"
        ),
    )


    for i in top_idx:

        ax.plot(
            [
                gt10[i, 0],
                pred10[i, 0],
            ],
            [
                gt10[i, 1],
                pred10[i, 1],
            ],
            [
                gt10[i, 2],
                pred10[i, 2],
            ],
            color=LINE_COLOR,
            alpha=0.55,
            linewidth=0.8,
        )


    set_can_axes(
        ax,
        cylinder,
        np.concatenate([
            gt10,
            pred10,
        ]),
    )

    ax.view_init(
        elev=18,
        azim=-60,
    )

    multi = case_summary[
        rank
    ]

    pred_med = float(
        multi[
            "pred_se3_median_mm"
        ]
    )

    pred_min = float(
        multi[
            "pred_se3_min_mm"
        ]
    )

    pred_max = float(
        multi[
            "pred_se3_max_mm"
        ]
    )

    gt_se3 = float(
        multi[
            "gt_se3_p95_t10_mm"
        ]
    )

    ax.set_title(
        (
            f"Soda can rank {rank} — t10\n"
            f"GT SE3 P95: "
            f"{gt_se3:.6f} mm | "
            f"Prediction multi-seed: "
            f"{pred_med:.2f} mm "
            f"[{pred_min:.2f}, "
            f"{pred_max:.2f}]"
        ),
        fontsize=11,
    )

    ax.legend(
        loc="upper right",
    )

    fig.text(
        0.5,
        0.02,
        (
            "Cylinder is an approximate "
            "GT-fitted visual guide only; "
            "all metrics use raw PointWorld points."
        ),
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=[
            0,
            0.05,
            1,
            1,
        ]
    )

    overlay_path = OUT / (
        f"rank{rank}_"
        "soda_can_t10_rigid_aligned_overlay.png"
    )

    fig.savefig(
        overlay_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


    visual_records[
        rank
    ] = {
        "rank":
            rank,

        "key":
            case["key"],

        "n_points":
            n,

        "gt":
            gt,

        "pred":
            pred,

        "gt_frames":
            gt_frames,

        "pred_frames":
            pred_frames,

        "se3_curve":
            se3_curve,

        "pair_curve":
            pair_curve,

        "cylinder":
            cylinder,

        "montage":
            str(
                montage_path
            ),

        "overlay":
            str(
                overlay_path
            ),
    }

    print(
        "MONTAGE =",
        montage_path,
    )

    print(
        "OVERLAY =",
        overlay_path,
    )


# ============================================================
# PROFESSOR SUMMARY:
# FOUR CASES AT t10
# ============================================================

print()
print("=" * 80)
print("3. BUILD PROFESSOR TOP-4 SUMMARY")
print("=" * 80)

fig = plt.figure(
    figsize=(13, 12)
)

fig.suptitle(
    (
        "PointWorld rigid-object integrity failure "
        "on GT-selected soda cans\n"
        "Prediction rigid motion removed; "
        "cylinder references are visualization only"
    ),
    fontsize=15,
)


for plot_idx, case in enumerate(
    CASES,
    1,
):

    rank = case[
        "rank"
    ]

    rec = visual_records[
        rank
    ]

    gt10 = rec[
        "gt_frames"
    ][-1]

    pred10 = rec[
        "pred_frames"
    ][-1]

    cylinder = rec[
        "cylinder"
    ]

    ax = fig.add_subplot(
        2,
        2,
        plot_idx,
        projection="3d",
    )

    draw_cylinder(
        ax,
        cylinder,
    )

    ax.scatter(
        gt10[:, 0],
        gt10[:, 1],
        gt10[:, 2],
        s=17,
        color=GT_COLOR,
        alpha=0.8,
        label="GT",
    )

    ax.scatter(
        pred10[:, 0],
        pred10[:, 1],
        pred10[:, 2],
        s=22,
        color=PRED_COLOR,
        alpha=0.9,
        label="Prediction",
    )

    set_can_axes(
        ax,
        cylinder,
        np.concatenate([
            gt10,
            pred10,
        ]),
    )

    ax.view_init(
        elev=18,
        azim=-60,
    )

    s = case_summary[
        rank
    ]

    gt_se3 = float(
        s[
            "gt_se3_p95_t10_mm"
        ]
    )

    med = float(
        s[
            "pred_se3_median_mm"
        ]
    )

    lo = float(
        s[
            "pred_se3_min_mm"
        ]
    )

    hi = float(
        s[
            "pred_se3_max_mm"
        ]
    )

    frac = float(
        s[
            "pred_se3_median_frac_diameter"
        ]
    ) * 100.0

    ax.set_title(
        (
            f"Rank {rank} | "
            f"n={rec['n_points']}\n"
            f"GT: {gt_se3:.4f} mm\n"
            f"Pred: {med:.1f} mm "
            f"[{lo:.1f}–{hi:.1f}] "
            f"({frac:.1f}% diameter)"
        ),
        fontsize=10,
    )


handles, labels = (
    fig.axes[0].get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
)

fig.text(
    0.5,
    0.035,
    (
        "Blue = GT. Red = PointWorld prediction after best-fit rigid alignment. "
        "Persistent separation therefore cannot be explained by object translation or rotation."
    ),
    ha="center",
    fontsize=10,
)

fig.tight_layout(
    rect=[
        0,
        0.07,
        1,
        0.94,
    ]
)

summary_path = OUT / (
    "PROFESSOR_top4_"
    "soda_can_rigid_failure_t10.png"
)

fig.savefig(
    summary_path,
    dpi=240,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "PROFESSOR_SUMMARY =",
    summary_path,
)


# ============================================================
# PROFESSOR TEMPORAL MULTI-SEED SUMMARY
# ============================================================

print()
print("=" * 80)
print("4. BUILD MULTI-SEED TEMPORAL SUMMARY")
print("=" * 80)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 9),
)

axes = axes.ravel()

for ax, case in zip(
    axes,
    CASES,
):

    rank = case[
        "rank"
    ]

    pred_curves = (
        time_data[
            f"rank{rank}_pred_se3"
        ]
    )

    gt_curve = (
        time_data[
            f"rank{rank}_gt_se3"
        ]
    )

    med = np.median(
        pred_curves,
        axis=0,
    )

    lo = np.min(
        pred_curves,
        axis=0,
    )

    hi = np.max(
        pred_curves,
        axis=0,
    )

    t = np.arange(
        len(med)
    )

    ax.plot(
        t,
        med,
        marker="o",
        color=PRED_COLOR,
        label=(
            "Prediction median"
        ),
    )

    ax.fill_between(
        t,
        lo,
        hi,
        color=PRED_COLOR,
        alpha=0.16,
        label=(
            "Prediction min-max"
        ),
    )

    ax.plot(
        t,
        gt_curve,
        marker="o",
        color=GT_COLOR,
        label="GT",
    )

    peak_t = int(
        np.argmax(
            med
        )
    )

    peak = float(
        med[
            peak_t
        ]
    )

    final = float(
        med[-1]
    )

    recovery = (
        max(
            0.0,
            peak - final,
        )
        / peak
        if peak > 0
        else 0.0
    )

    ax.set_title(
        (
            f"Rank {rank} | "
            f"peak t{peak_t}: "
            f"{peak:.1f} mm | "
            f"final: {final:.1f} mm | "
            f"recovery: "
            f"{recovery*100:.0f}%"
        ),
        fontsize=10,
    )

    ax.set_xlabel(
        "Future timestep"
    )

    ax.set_ylabel(
        "Best-fit SE(3) residual P95 (mm)"
    )

    ax.set_xticks(
        np.arange(11)
    )

    ax.grid(
        alpha=0.25,
    )


handles, labels = (
    axes[0].get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
)

fig.suptitle(
    (
        "Soda-can structural deformation over time "
        "across 5 seeds × 2 repeats\n"
        "Shaded region = min–max across stochastic PointWorld forwards"
    ),
    fontsize=14,
)

fig.tight_layout(
    rect=[
        0,
        0.07,
        1,
        0.93,
    ]
)

temporal_path = OUT / (
    "PROFESSOR_top4_"
    "multiseed_deformation_over_time.png"
)

fig.savefig(
    temporal_path,
    dpi=220,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "PROFESSOR_TEMPORAL =",
    temporal_path,
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {}

for rank, rec in (
    visual_records.items()
):

    s = case_summary[
        rank
    ]

    summary[
        f"rank{rank}"
    ] = {
        "sample_key":
            rec["key"],

        "n_points":
            rec["n_points"],

        "cylinder_radius_mm":
            rec[
                "cylinder"
            ][
                "radius"
            ] * 1000.0,

        "cylinder_height_mm":
            rec[
                "cylinder"
            ][
                "height"
            ] * 1000.0,

        "single_run_se3_p95_t10_mm":
            float(
                rec[
                    "se3_curve"
                ][-1]
            ),

        "multiseed_se3_min_mm":
            float(
                s[
                    "pred_se3_min_mm"
                ]
            ),

        "multiseed_se3_median_mm":
            float(
                s[
                    "pred_se3_median_mm"
                ]
            ),

        "multiseed_se3_max_mm":
            float(
                s[
                    "pred_se3_max_mm"
                ]
            ),

        "gt_se3_p95_t10_mm":
            float(
                s[
                    "gt_se3_p95_t10_mm"
                ]
            ),

        "montage":
            rec[
                "montage"
            ],

        "overlay":
            rec[
                "overlay"
            ],
    }


with open(
    OUT
    / "professor_visual_summary.json",
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
    "N_CASES_VISUALIZED =",
    len(
        visual_records
    ),
)

print(
    "PROFESSOR_SUMMARY =",
    summary_path,
)

print(
    "PROFESSOR_TEMPORAL =",
    temporal_path,
)

for rank in sorted(
    visual_records
):

    print(
        f"RANK{rank}_MONTAGE =",
        visual_records[
            rank
        ][
            "montage"
        ],
    )


print(
    "CYLINDER_REFERENCE_METRIC_USE = False"
)

print(
    "BEST_FIT_RIGID_ALIGNMENT_VISUAL_ONLY = True"
)

print(
    "NO MODEL RERUN"
)

print(
    "NO MBNSF"
)

print(
    "PROFESSOR_VISUALIZATION_SUCCESS = True"
)
