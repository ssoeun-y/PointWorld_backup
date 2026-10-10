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

SEEDS = [42, 43, 44, 45, 46]
REPEATS_PER_SEED = 2

RESULT_ROOT = Path(
    "diagnostics/results/"
    "behavior_task0001_soda_can_gt_top5_structural"
)

CAPTURE_DIR = RESULT_ROOT / "captures"

OUT = Path(
    "diagnostics/results/"
    "behavior_task0001_soda_can_multiseed_robustness"
)

OUT.mkdir(parents=True, exist_ok=True)


# Rank 2 had zero can points after official preprocessing.
# Therefore it is explicitly N/A, not counted as success/failure.
CASES = [
    {
        "rank": 1,
        "key": "task-0001_episode_00012190-460:471",
        "capture":
            "rank1_task-0001_episode_00012190-460_471.npz",
    },
    {
        "rank": 3,
        "key": "task-0001_episode_00012100-455:466",
        "capture":
            "rank3_task-0001_episode_00012100-455_466.npz",
    },
    {
        "rank": 4,
        "key": "task-0001_episode_00012150-415:426",
        "capture":
            "rank4_task-0001_episode_00012150-415_426.npz",
    },
    {
        "rank": 5,
        "key": "task-0001_episode_00012210-480:491",
        "capture":
            "rank5_task-0001_episode_00012210-480_491.npz",
    },
]


# ============================================================
# HELPERS
# ============================================================

def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def summary(x):
    x = np.asarray(x, dtype=np.float64)

    return {
        "min": float(np.min(x)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "max": float(np.max(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def pair_change_mm(x0, xt):
    d0 = pdist(
        x0.astype(np.float64)
    )

    dt = pdist(
        xt.astype(np.float64)
    )

    return np.abs(dt - d0) * 1000.0


def pair_p95_mm(x0, xt):
    d = pair_change_mm(
        x0,
        xt,
    )

    if len(d) == 0:
        return 0.0

    return float(
        np.percentile(
            d,
            95,
        )
    )


def rigid_residual_mm(x0, xt):
    """
    Best-fit rigid SE(3) x0 -> xt.
    What remains cannot be explained
    by one rigid-body transform.
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

    residual = np.linalg.norm(
        aligned - xt,
        axis=-1,
    )

    return residual * 1000.0


def se3_p95_mm(x0, xt):
    r = rigid_residual_mm(
        x0,
        xt,
    )

    return float(
        np.percentile(
            r,
            95,
        )
    )


def curve_metrics(gt, pred):
    """
    t0-reference structure error for every timestep.
    """

    T = gt.shape[0]

    gt_pair = []
    pred_pair = []

    gt_se3 = []
    pred_se3 = []

    for t in range(T):

        gt_pair.append(
            pair_p95_mm(
                gt[0],
                gt[t],
            )
        )

        pred_pair.append(
            pair_p95_mm(
                pred[0],
                pred[t],
            )
        )

        gt_se3.append(
            se3_p95_mm(
                gt[0],
                gt[t],
            )
        )

        pred_se3.append(
            se3_p95_mm(
                pred[0],
                pred[t],
            )
        )

    return (
        np.asarray(gt_pair),
        np.asarray(pred_pair),
        np.asarray(gt_se3),
        np.asarray(pred_se3),
    )


def recovery_metrics(curve):
    """
    Quantifies:
    deform -> peak -> potentially recover.

    recovery_fraction:
      0   = no drop after peak
      1   = final returned all the way to zero
    """

    curve = np.asarray(
        curve,
        dtype=np.float64,
    )

    t_peak = int(
        np.argmax(curve)
    )

    peak = float(
        curve[t_peak]
    )

    final = float(
        curve[-1]
    )

    drop = max(
        0.0,
        peak - final,
    )

    recovery_fraction = (
        drop / peak
        if peak > 0
        else 0.0
    )

    return {
        "peak_timestep": t_peak,
        "peak_mm": peak,
        "final_mm": final,
        "peak_to_final_drop_mm": drop,
        "recovery_fraction": recovery_fraction,
    }


# ============================================================
# LOAD CAPTURE AUDITS
# ============================================================

print("=" * 78)
print("1. LOAD PREVIOUS EXACT OBJECT MAPPINGS")
print("=" * 78)

case_data = {}

for c in CASES:

    path = (
        CAPTURE_DIR
        / c["capture"]
    )

    print()
    print(
        "rank =",
        c["rank"],
    )

    print(
        "capture =",
        path,
    )

    if not path.exists():
        raise RuntimeError(
            f"Capture missing: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as z:

        idx = z[
            "scene_indices"
        ].copy()

        gt_can = z[
            "gt"
        ].copy()

    if len(idx) < 4:
        raise RuntimeError(
            f"Too few saved can points "
            f"for rank {c['rank']}"
        )

    diameter_mm = float(
        pdist(
            gt_can[0].astype(
                np.float64
            )
        ).max()
        * 1000.0
    )

    gt_pair_t10 = (
        pair_p95_mm(
            gt_can[0],
            gt_can[-1],
        )
    )

    gt_se3_t10 = (
        se3_p95_mm(
            gt_can[0],
            gt_can[-1],
        )
    )

    print(
        "n_can_points =",
        len(idx),
    )

    print(
        "diameter_mm =",
        diameter_mm,
    )

    print(
        "GT pair P95 t10 =",
        gt_pair_t10,
    )

    print(
        "GT SE3 P95 t10 =",
        gt_se3_t10,
    )

    case_data[
        c["key"]
    ] = {
        **c,
        "idx": idx,
        "saved_gt": gt_can,
        "diameter_mm":
            diameter_mm,
        "gt_pair_t10":
            gt_pair_t10,
        "gt_se3_t10":
            gt_se3_t10,
    }


# ============================================================
# OFFICIAL MODEL + OFFICIAL TEST LOADER
# ============================================================

args = parse_args()

print()
print("=" * 78)
print("2. INITIALIZE OFFICIAL POINTWORLD")
print("=" * 78)

tester = Tester(args)

dl, _ = tester._build_eval_loader(
    "test",
    enable_mask=False,
)

tester.model.eval()

print(
    "OFFICIAL_MODEL_READY = True"
)


# ============================================================
# CACHE EXACT CPU BATCHES
# ============================================================

print()
print("=" * 78)
print("3. FIND FOUR OBSERVABLE CASES")
print("=" * 78)

batches = {}

for batch in dl:

    key = str(
        batch["__key__"][0]
    )

    if key not in case_data:
        continue

    batches[key] = {
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

    print(
        "FOUND =",
        key,
    )


expected = set(
    case_data.keys()
)

print()
print(
    "BATCH_SET_EXACT =",
    set(batches.keys())
    == expected,
)

if set(batches.keys()) != expected:
    raise RuntimeError(
        "Missing observable Top-5 batches"
    )


# ============================================================
# RUN MULTI-SEED ROBUSTNESS
# ============================================================

print()
print("=" * 78)
print("4. MULTI-SEED ROBUSTNESS")
print("=" * 78)

run_rows = []

time_curves = {}

for key in sorted(
    case_data,
    key=lambda k:
        case_data[k]["rank"],
):

    c = case_data[key]

    rank = c["rank"]

    idx = c["idx"]

    saved_gt = c[
        "saved_gt"
    ]

    diameter_mm = c[
        "diameter_mm"
    ]

    print()
    print("#" * 78)
    print(
        f"RANK {rank} | {key}"
    )
    print("#" * 78)

    cpu_batch = batches[key]

    # One exact GT check before any predictions.
    gt_current = (
        cpu_batch[
            "gt_scene_flows"
        ][0]
        .numpy()
    )

    gt_current_can = (
        gt_current[
            :,
            idx,
            :
        ]
    )

    gt_exact = np.array_equal(
        gt_current_can,
        saved_gt,
    )

    print(
        "GT_CAPTURE_EXACT =",
        gt_exact,
    )

    if not gt_exact:
        raise RuntimeError(
            f"GT changed: {key}"
        )

    # Move one immutable copy to GPU.
    gpu_batch = {
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
        for k, v in cpu_batch.items()
    }


    case_curves = []

    for seed in SEEDS:

        for repeat in range(
            REPEATS_PER_SEED
        ):

            reset_rng(seed)

            # Fresh tensor clones for each forward.
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
                in gpu_batch.items()
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

            pred_all = (
                out["scene_flows"][0]
                .detach()
                .cpu()
                .numpy()
            )

            gt_all = (
                b["gt_scene_flows"][0]
                .detach()
                .cpu()
                .numpy()
            )

            # Official-valid mask.
            context = (
                b["scene_context_mask"]
            )

            exists = (
                b["scene_exists"]
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

            l2_all = (
                torch.linalg.vector_norm(
                    out["scene_flows"]
                    - b["gt_scene_flows"],
                    dim=-1,
                )
            )

            manual_l2 = float(
                l2_all[
                    valid
                ].mean().item()
            )

            official_l2 = float(
                loss_dict[
                    "l2/mean"
                ]
            )

            l2_exact = bool(
                np.isclose(
                    manual_l2,
                    official_l2,
                    rtol=0.0,
                    atol=1e-8,
                )
            )

            if not l2_exact:
                raise RuntimeError(
                    f"L2 reproduction "
                    f"failed: {key}"
                )


            pred_can = (
                pred_all[
                    :,
                    idx,
                    :
                ]
            )

            gt_can = (
                gt_all[
                    :,
                    idx,
                    :
                ]
            )

            if not np.array_equal(
                gt_can,
                saved_gt,
            ):
                raise RuntimeError(
                    f"GT mismatch during "
                    f"run: {key}"
                )


            (
                gt_pair_curve,
                pred_pair_curve,
                gt_se3_curve,
                pred_se3_curve,
            ) = curve_metrics(
                gt_can,
                pred_can,
            )


            pair_t10 = float(
                pred_pair_curve[-1]
            )

            se3_t10 = float(
                pred_se3_curve[-1]
            )


            can_l2_t10 = (
                np.linalg.norm(
                    pred_can[-1]
                    - gt_can[-1],
                    axis=-1,
                )
                * 1000.0
            )

            can_l2_mean = float(
                can_l2_t10.mean()
            )


            rec = recovery_metrics(
                pred_se3_curve
            )


            row = {
                "rank": rank,
                "sample_key": key,

                "seed": seed,
                "repeat": repeat,

                "n_can_points":
                    len(idx),

                "object_diameter_mm":
                    diameter_mm,

                "gt_pair_p95_t10_mm":
                    float(
                        gt_pair_curve[-1]
                    ),

                "gt_se3_p95_t10_mm":
                    float(
                        gt_se3_curve[-1]
                    ),

                "pred_pair_p95_t10_mm":
                    pair_t10,

                "pred_se3_p95_t10_mm":
                    se3_t10,

                # Scale-independent descriptors.
                "pred_pair_p95_frac_diameter":
                    (
                        pair_t10
                        / diameter_mm
                    ),

                "pred_se3_p95_frac_diameter":
                    (
                        se3_t10
                        / diameter_mm
                    ),

                "can_t10_l2_mean_mm":
                    can_l2_mean,

                "scene_official_l2_m":
                    official_l2,

                "peak_se3_timestep":
                    rec[
                        "peak_timestep"
                    ],

                "peak_se3_p95_mm":
                    rec[
                        "peak_mm"
                    ],

                "final_se3_p95_mm":
                    rec[
                        "final_mm"
                    ],

                "peak_to_final_drop_mm":
                    rec[
                        "peak_to_final_drop_mm"
                    ],

                "recovery_fraction":
                    rec[
                        "recovery_fraction"
                    ],

                "official_l2_reproduced":
                    l2_exact,
            }

            run_rows.append(
                row
            )

            case_curves.append({
                "seed": seed,
                "repeat": repeat,
                "pred_pair":
                    pred_pair_curve,
                "pred_se3":
                    pred_se3_curve,
                "gt_pair":
                    gt_pair_curve,
                "gt_se3":
                    gt_se3_curve,
            })


            print(
                f"seed={seed} "
                f"rep={repeat} | "
                f"pair={pair_t10:.3f} mm | "
                f"SE3={se3_t10:.3f} mm | "
                f"peak={rec['peak_mm']:.3f} "
                f"@t{rec['peak_timestep']} | "
                f"recovery="
                f"{rec['recovery_fraction']:.3f}"
            )


    time_curves[key] = (
        case_curves
    )

    torch.cuda.empty_cache()


# ============================================================
# SAVE ALL 40 RUNS
# ============================================================

runs_csv = OUT / (
    "soda_can_multiseed_all_runs.csv"
)

with open(
    runs_csv,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            run_rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        run_rows
    )


# ============================================================
# CASE-LEVEL AGGREGATION
# ============================================================

print()
print("=" * 78)
print("5. CASE-LEVEL ROBUSTNESS")
print("=" * 78)

case_rows = []

for key in sorted(
    case_data,
    key=lambda k:
        case_data[k]["rank"],
):

    c = case_data[key]

    rows = [
        r
        for r in run_rows
        if r["sample_key"] == key
    ]

    pair_values = [
        r[
            "pred_pair_p95_t10_mm"
        ]
        for r in rows
    ]

    se3_values = [
        r[
            "pred_se3_p95_t10_mm"
        ]
        for r in rows
    ]

    pair_norm = [
        r[
            "pred_pair_p95_frac_diameter"
        ]
        for r in rows
    ]

    se3_norm = [
        r[
            "pred_se3_p95_frac_diameter"
        ]
        for r in rows
    ]

    l2_values = [
        r[
            "can_t10_l2_mean_mm"
        ]
        for r in rows
    ]

    recovery = [
        r[
            "recovery_fraction"
        ]
        for r in rows
    ]

    pair_s = summary(
        pair_values
    )

    se3_s = summary(
        se3_values
    )

    l2_s = summary(
        l2_values
    )

    rec_s = summary(
        recovery
    )

    pair_norm_s = summary(
        pair_norm
    )

    se3_norm_s = summary(
        se3_norm
    )

    cr = {
        "rank":
            c["rank"],

        "sample_key":
            key,

        "n_can_points":
            len(c["idx"]),

        "object_diameter_mm":
            c["diameter_mm"],

        "gt_pair_p95_t10_mm":
            c["gt_pair_t10"],

        "gt_se3_p95_t10_mm":
            c["gt_se3_t10"],

        "pred_pair_min_mm":
            pair_s["min"],

        "pred_pair_median_mm":
            pair_s["median"],

        "pred_pair_max_mm":
            pair_s["max"],

        "pred_se3_min_mm":
            se3_s["min"],

        "pred_se3_median_mm":
            se3_s["median"],

        "pred_se3_max_mm":
            se3_s["max"],

        "pred_pair_min_frac_diameter":
            pair_norm_s["min"],

        "pred_pair_median_frac_diameter":
            pair_norm_s["median"],

        "pred_se3_min_frac_diameter":
            se3_norm_s["min"],

        "pred_se3_median_frac_diameter":
            se3_norm_s["median"],

        "l2_median_mm":
            l2_s["median"],

        "recovery_fraction_median":
            rec_s["median"],

        "recovery_fraction_max":
            rec_s["max"],
    }

    case_rows.append(
        cr
    )


    print()
    print(
        f'RANK {c["rank"]}'
    )

    print(
        f' GT pair P95 = '
        f'{c["gt_pair_t10"]:.6f} mm'
    )

    print(
        f' PRED pair P95 '
        f'min/median/max = '
        f'{pair_s["min"]:.3f} / '
        f'{pair_s["median"]:.3f} / '
        f'{pair_s["max"]:.3f} mm'
    )

    print(
        f' GT SE3 P95 = '
        f'{c["gt_se3_t10"]:.6f} mm'
    )

    print(
        f' PRED SE3 P95 '
        f'min/median/max = '
        f'{se3_s["min"]:.3f} / '
        f'{se3_s["median"]:.3f} / '
        f'{se3_s["max"]:.3f} mm'
    )

    print(
        f' PRED SE3 median / '
        f'object diameter = '
        f'{100.0 * se3_norm_s["median"]:.2f}%'
    )

    print(
        f' recovery fraction '
        f'median/max = '
        f'{rec_s["median"]:.3f} / '
        f'{rec_s["max"]:.3f}'
    )


cases_csv = OUT / (
    "soda_can_multiseed_case_summary.csv"
)

with open(
    cases_csv,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            case_rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        case_rows
    )


# ============================================================
# GLOBAL ROBUSTNESS TEST
# ============================================================

max_gt_pair = max(
    r[
        "gt_pair_p95_t10_mm"
    ]
    for r in case_rows
)

max_gt_se3 = max(
    r[
        "gt_se3_p95_t10_mm"
    ]
    for r in case_rows
)

min_pred_pair_across_cases = min(
    r[
        "pred_pair_min_mm"
    ]
    for r in case_rows
)

min_pred_se3_across_cases = min(
    r[
        "pred_se3_min_mm"
    ]
    for r in case_rows
)


print()
print("=" * 78)
print("6. CONSERVATIVE CROSS-CASE RESULT")
print("=" * 78)

print(
    "MAX_GT_PAIR_P95_MM =",
    max_gt_pair,
)

print(
    "MIN_PRED_PAIR_P95_ACROSS_ALL_CASES_AND_RUNS_MM =",
    min_pred_pair_across_cases,
)

print(
    "MAX_GT_SE3_P95_MM =",
    max_gt_se3,
)

print(
    "MIN_PRED_SE3_P95_ACROSS_ALL_CASES_AND_RUNS_MM =",
    min_pred_se3_across_cases,
)


# ============================================================
# FIGURE A:
# GT vs prediction ranges per case
# ============================================================

ranks = np.asarray([
    r["rank"]
    for r in case_rows
])

pred_med = np.asarray([
    r["pred_se3_median_mm"]
    for r in case_rows
])

pred_min = np.asarray([
    r["pred_se3_min_mm"]
    for r in case_rows
])

pred_max = np.asarray([
    r["pred_se3_max_mm"]
    for r in case_rows
])

gt_value = np.asarray([
    r["gt_se3_p95_t10_mm"]
    for r in case_rows
])

lower = pred_med - pred_min
upper = pred_max - pred_med

fig, ax = plt.subplots(
    figsize=(8, 5),
)

ax.errorbar(
    ranks,
    pred_med,
    yerr=np.vstack([
        lower,
        upper,
    ]),
    fmt="o",
    capsize=5,
    label=(
        "Prediction "
        "(median, min-max over runs)"
    ),
)

ax.plot(
    ranks,
    gt_value,
    marker="o",
    label="GT",
)

ax.set_yscale("log")

ax.set_xticks(
    ranks
)

ax.set_xlabel(
    "Episode-disjoint GT-motion rank"
)

ax.set_ylabel(
    "Best-fit SE(3) residual P95 "
    "(mm, log scale)"
)

ax.set_title(
    "Soda can rigid-structure failure "
    "across seeds"
)

ax.grid(
    alpha=0.25,
    which="both",
)

ax.legend()

fig.tight_layout()

robust_png = OUT / (
    "multiseed_se3_robustness.png"
)

fig.savefig(
    robust_png,
    dpi=200,
)

plt.close(fig)


# ============================================================
# FIGURE B:
# timewise SE3 median + range
# ============================================================

for key in sorted(
    case_data,
    key=lambda k:
        case_data[k]["rank"],
):

    c = case_data[key]

    curves = np.stack([
        x["pred_se3"]
        for x in time_curves[key]
    ])

    median_curve = np.median(
        curves,
        axis=0,
    )

    min_curve = np.min(
        curves,
        axis=0,
    )

    max_curve = np.max(
        curves,
        axis=0,
    )

    gt_curve = (
        time_curves[key][0][
            "gt_se3"
        ]
    )

    t = np.arange(
        len(median_curve)
    )

    fig, ax = plt.subplots(
        figsize=(8, 5),
    )

    ax.plot(
        t,
        median_curve,
        marker="o",
        label="Prediction median",
    )

    ax.fill_between(
        t,
        min_curve,
        max_curve,
        alpha=0.2,
        label="Prediction min-max",
    )

    ax.plot(
        t,
        gt_curve,
        marker="o",
        label="GT",
    )

    ax.set_xlabel(
        "Future timestep"
    )

    ax.set_ylabel(
        "Best-fit SE(3) residual P95 (mm)"
    )

    ax.set_title(
        f'Rank {c["rank"]}: '
        'deformation and recovery over time'
    )

    ax.grid(
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUT
        / (
            f'rank{c["rank"]}_'
            'multiseed_se3_over_time.png'
        ),
        dpi=200,
    )

    plt.close(fig)


# ============================================================
# SAVE TIME CURVES
# ============================================================

time_npz = {}

for key, curves in (
    time_curves.items()
):

    rank = case_data[key][
        "rank"
    ]

    time_npz[
        f"rank{rank}_pred_se3"
    ] = np.stack([
        x["pred_se3"]
        for x in curves
    ])

    time_npz[
        f"rank{rank}_pred_pair"
    ] = np.stack([
        x["pred_pair"]
        for x in curves
    ])

    time_npz[
        f"rank{rank}_gt_se3"
    ] = curves[0][
        "gt_se3"
    ]

    time_npz[
        f"rank{rank}_gt_pair"
    ] = curves[0][
        "gt_pair"
    ]

np.savez_compressed(
    OUT / "multiseed_time_curves.npz",
    **time_npz,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary_json = {
    "seeds":
        SEEDS,

    "repeats_per_seed":
        REPEATS_PER_SEED,

    "n_observable_cases":
        len(case_rows),

    "n_total_forwards":
        len(run_rows),

    "rank2_status":
        "N/A: zero soda-can points after "
        "official PointWorld preprocessing",

    "max_gt_pair_p95_mm":
        max_gt_pair,

    "min_pred_pair_p95_across_all_cases_and_runs_mm":
        min_pred_pair_across_cases,

    "max_gt_se3_p95_mm":
        max_gt_se3,

    "min_pred_se3_p95_across_all_cases_and_runs_mm":
        min_pred_se3_across_cases,

    "case_summaries":
        case_rows,
}

with open(
    OUT / "summary.json",
    "w",
) as f:

    json.dump(
        summary_json,
        f,
        indent=2,
    )


print()
print("=" * 78)
print("FINAL")
print("=" * 78)

print(
    "OBSERVABLE_CASES =",
    len(case_rows),
)

print(
    "TOTAL_FORWARDS =",
    len(run_rows),
)

print(
    "ALL_OFFICIAL_L2_REPRODUCED =",
    all(
        r[
            "official_l2_reproduced"
        ]
        for r in run_rows
    ),
)

print(
    "RANK2_STATUS = N/A_ZERO_CAN_POINTS"
)

print(
    "RUNS_CSV =",
    runs_csv,
)

print(
    "CASES_CSV =",
    cases_csv,
)

print(
    "ROBUSTNESS_PLOT =",
    robust_png,
)

print(
    "TIME_CURVES =",
    OUT
    / "multiseed_time_curves.npz",
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
    "MULTISEED_ROBUSTNESS_SUCCESS = True"
)
