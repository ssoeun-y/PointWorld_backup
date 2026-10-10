#!/usr/bin/env python3

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import diagnostics.eval_acc_only_pa_intervention_5task as old


# ============================================================
# CONFIG
# ============================================================

TASK = "task-0000"
N_TARGET = 30

# Use clearly high-Acc clips for the first feasibility test.
BOUNDARY_SI = 0.24580418
MIN_ACC_MULTIPLE = 1.50

# Immediate phase-speed change:
# derivative at the beginning ~= 0.90
SLOW_FRACTION = 0.10

MIN_POINTS = 8
SEED = 42

ROOT = Path(__file__).resolve().parents[1]

CORR_CSV = (
    ROOT
    / "diagnostics/results/"
      "acc_only_boundary_correction_current10457/"
      "correction_results.csv"
)

OUT = (
    ROOT
    / "diagnostics/results/"
      "online_rigidity_reflex_poc"
)

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def make_slow10_phase(T):
    """
    Fixed-horizon proxy for an online immediate ~10% slowdown.

    phi'(0) = 1-a = 0.9
    phi'(1) = 1+a = 1.1

    During the first half, progress is slower than baseline.
    Endpoint remains exactly fixed:
      phi(0)=0, phi(1)=1.

    In a receding-horizon controller we would execute only
    the immediate portion, observe/replan, and not blindly
    execute the later catch-up portion.
    """

    u = np.linspace(
        0.0,
        1.0,
        T,
        dtype=np.float64,
    )

    a = SLOW_FRACTION

    phi = (
        u
        - a * u * (1.0 - u)
    )

    phi[0] = 0.0
    phi[-1] = 1.0

    if np.any(
        np.diff(phi) <= 0
    ):
        raise RuntimeError(
            "slow10 phase is not monotonic"
        )

    return phi


def rigidity_score(
    predicted,
    initial_shape,
):
    """
    Future-GT-free rigidity score.

    Compare pairwise point distances in predicted object
    against its initial observed geometry.

    Lower = more internally rigid.
    """

    p = np.asarray(
        predicted,
        dtype=np.float64,
    )

    x0 = np.asarray(
        initial_shape,
        dtype=np.float64,
    )

    n = len(x0)

    if n < MIN_POINTS:
        return np.nan

    tri = np.triu_indices(
        n,
        k=1,
    )

    d0 = np.linalg.norm(
        x0[:, None, :]
        - x0[None, :, :],
        axis=-1,
    )[tri]

    dp = np.linalg.norm(
        p[:, None, :]
        - p[None, :, :],
        axis=-1,
    )[tri]

    keep = (
        np.isfinite(d0)
        & np.isfinite(dp)
        & (d0 > 1e-4)
    )

    if keep.sum() == 0:
        return np.nan

    relative_change = (
        np.abs(
            dp[keep] - d0[keep]
        )
        /
        (
            d0[keep] + 1e-8
        )
    )

    return float(
        np.mean(relative_change)
    )


# ============================================================
# Select 30 high-Acc clips
# ============================================================

df = pd.read_csv(CORR_CSV)

success = (
    df["success"]
    .astype(str)
    .str.lower()
    .isin(["true", "1"])
)

pool = df[
    success
    & (df["task"] == TASK)
].copy()

pool["acc_multiple"] = (
    pool["original_acc_si"]
    / BOUNDARY_SI
)

pool = (
    pool[
        pool["acc_multiple"]
        >= MIN_ACC_MULTIPLE
    ]
    .sort_values(
        "original_acc_si",
        ascending=False,
    )
    .head(N_TARGET)
)

target_keys = set(
    pool["clip"].astype(str)
)

if len(target_keys) < N_TARGET:
    raise RuntimeError(
        f"Need {N_TARGET} clips, "
        f"found {len(target_keys)}"
    )

acc_lookup = dict(
    zip(
        pool["clip"].astype(str),
        pool["original_acc_si"],
    )
)

print("=" * 80)
print("ONLINE RIGIDITY REFLEX PoC")
print("TASK =", TASK)
print("N =", len(target_keys))
print("Protocol = original vs immediate slow10")
print("No absolute rigidity threshold")
print("=" * 80)


# ============================================================
# Frozen PointWorld
# ============================================================

args = old.parse_args(
    skip_command_line=True
)

args.seed = SEED
args.model_path = old.MODEL_PATH
args.norm_stats_path = old.NORM_STATS
args.domains = ["behavior"]
args.data_dirs = [
    old.TASK_DIRS[TASK]
]

args.batch_size = 1
args.num_workers = 0
args.eval_num_workers = 0
args.eval_num_batches = -1
args.device = "cuda"
args.eval_skip_viz = True

tester = old.Tester(args)
tester.model.eval()

device = tester.device
domain = "behavior"

urdf_path = (
    old.resolve_robot_urdf(
        domain
    )
)

robot_sampler = (
    old.TorchRobotSampler(
        urdf_path=urdf_path,
        gripper_only=(
            old.RELEASE_GRIPPER_ONLY
        ),
        device="cpu",
    )
)

force_single_arm = False

if getattr(
    args,
    "_eval_override_active",
    False,
):
    fs = getattr(
        args,
        "eval_force_single_arm_domains",
        set(),
    )

    force_single_arm = (
        domain in fs
    )


# Preserve object IDs through official pipeline.
old.omap.pmod.grid_sample_transform = (
    old.omap.grid_sample_with_object_ids
)

old.omap.pmod.gather = (
    old.omap.gather_with_object_ids
)

old.reset_seed(SEED)


# ============================================================
# Official WDS
# ============================================================

shard_paths = (
    old.dlmod.gather_shard_paths(
        old.TASK_DIRS[TASK],
        "test",
        rank=tester.rank,
    )
)

random.Random(
    args.seed + tester.rank
).shuffle(shard_paths)

dataset = old.wds.WebDataset(
    shard_paths,
    shardshuffle=False,
    detshuffle=True,
    resampled=False,
    nodesplitter=(
        old.wds.split_by_node
        if tester.world_size > 1
        else None
    ),
    seed=args.seed + tester.rank,
    handler=old.wds.warn_and_continue,
)


rows = []
found = set()

LABELS = [
    "original",
    "slow10",
]


try:

    with torch.inference_mode():

        for raw in dataset:

            raw_key = str(
                raw.get("__key__", "")
            )

            if raw_key not in target_keys:
                continue

            decoded = old.decode_data(
                raw,
                domain=domain,
            )

            key = str(
                decoded.get(
                    "__key__",
                    raw_key,
                )
            )

            built = (
                old.omap.build_flow_sample_with_object_ids(
                    decoded,
                    domain=domain,
                    robot_sampler=robot_sampler,
                    max_robot_points=(
                        args.max_robot_points
                    ),
                    deterministic=True,
                    seed=args.seed + tester.rank,
                    force_single_arm=force_single_arm,
                )
            )

            T = int(
                built[
                    "robot_flows"
                ].shape[0]
            )

            base_phase = np.linspace(
                0.0,
                1.0,
                T,
                dtype=np.float64,
            )

            slow_phase = (
                make_slow10_phase(T)
            )

            base = old.src.apply_phase(
                copy.deepcopy(built),
                base_phase,
            )

            slow = old.src.apply_phase(
                copy.deepcopy(built),
                slow_phase,
            )

            # -----------------------------------------------
            # Hard endpoint audit
            # -----------------------------------------------

            for ti in [0, -1]:

                if not np.allclose(
                    base[
                        "robot_flows"
                    ][ti],
                    slow[
                        "robot_flows"
                    ][ti],
                    atol=1e-7,
                    rtol=0,
                ):
                    raise RuntimeError(
                        f"{key}: endpoint changed"
                    )

            variants = [
                ("original", base),
                ("slow10", slow),
            ]

            released = []

            for label, sample in variants:

                sample = old.release_process(
                    sample,
                    args,
                    domain,
                    tester.rank,
                )

                released.append(
                    (
                        label,
                        sample,
                    )
                )

            batch_cpu = (
                old.custom_collate_fn(
                    [
                        x[1]
                        for x in released
                    ],
                    args=args,
                )
            )

            # Exact scene/object correspondence audit.
            for field in [
                "scene_flows",
                "scene_exists",
                "scene_context_mask",
                "scene_supervised_mask",
                "scene_moved_mask",
                "scene_object_id",
                "gt_scene_flows",
            ]:

                a = batch_cpu[field][0]
                b = batch_cpu[field][1]

                if not old.same_tensor(a, b):
                    raise RuntimeError(
                        f"{key}: "
                        f"{field} changed"
                    )

            batch_gpu = old.to_device(
                batch_cpu,
                device,
            )

            # =================================================
            # Forward 1:
            # Action decision using ONLY rigidity score.
            # =================================================

            out_score = tester.model(
                batch_gpu,
                training=False,
            )

            pred_score = (
                out_score[
                    "scene_flows"
                ]
                .detach()
                .cpu()
                .numpy()
            )

            # =================================================
            # Forward 2:
            # Independent PA-MPJPE evaluation.
            # =================================================

            out_eval = tester.model(
                batch_gpu,
                training=False,
            )

            pred_eval = (
                out_eval[
                    "scene_flows"
                ]
                .detach()
                .cpu()
                .numpy()
            )


            labels = (
                old.mask_array(
                    batch_cpu[
                        "scene_object_id"
                    ]
                )[0]
            )

            context = (
                old.mask_array(
                    batch_cpu[
                        "scene_context_mask"
                    ]
                )[0]
                .astype(bool)
            )

            exists = (
                old.mask_array(
                    batch_cpu[
                        "scene_exists"
                    ]
                )[0]
                .astype(bool)
            )

            supervised = (
                old.mask_array(
                    batch_cpu[
                        "scene_supervised_mask"
                    ]
                )[0]
                .astype(bool)
            )

            moved = (
                old.mask_array(
                    batch_cpu[
                        "scene_moved_mask"
                    ]
                )[0]
                .astype(bool)
            )

            valid = (
                (~context)
                & exists
                & supervised
            )

            gt = (
                old.arr(
                    batch_cpu[
                        "gt_scene_flows"
                    ]
                )[0]
            )

            table = (
                old.omap.OBJECT_TABLE[key]
            )


            # -------------------------------------------------
            # Because this is an online-immediate-control PoC,
            # only evaluate the early half of the prediction
            # horizon, where slow10 is actually slower.
            # -------------------------------------------------

            T_scene = gt.shape[0]

            eval_end = max(
                2,
                int(
                    np.ceil(
                        0.5 * T_scene
                    )
                ),
            )

            per_object = []

            for oid, mesh_name in table.items():

                oid = int(oid)

                object_mask = (
                    labels == oid
                )

                moved_support = int(
                    (
                        object_mask
                        & moved
                        & valid
                    ).sum()
                )

                if moved_support <= 0:
                    continue

                score_original = []
                score_slow = []

                pa_original = []
                pa_slow = []

                for t in range(
                    1,
                    eval_end,
                ):

                    m = (
                        object_mask[t]
                        & valid[t]
                        & object_mask[0]
                        & exists[0]
                    )

                    if int(m.sum()) < MIN_POINTS:
                        continue

                    x0 = gt[
                        0,
                        m,
                    ]

                    centered = (
                        x0
                        - x0.mean(
                            axis=0,
                            keepdims=True,
                        )
                    )

                    if (
                        np.linalg.matrix_rank(
                            centered,
                            tol=1e-8,
                        )
                        < 2
                    ):
                        continue

                    r0 = rigidity_score(
                        pred_score[
                            0,
                            t,
                            m,
                        ],
                        x0,
                    )

                    r1 = rigidity_score(
                        pred_score[
                            1,
                            t,
                            m,
                        ],
                        x0,
                    )

                    if (
                        not np.isfinite(r0)
                        or not np.isfinite(r1)
                    ):
                        continue

                    try:

                        pa0, _ = (
                            old.p_mpjpe_single(
                                pred_eval[
                                    0,
                                    t,
                                    m,
                                ],
                                x0,
                            )
                        )

                        pa1, _ = (
                            old.p_mpjpe_single(
                                pred_eval[
                                    1,
                                    t,
                                    m,
                                ],
                                x0,
                            )
                        )

                    except Exception:
                        continue

                    score_original.append(r0)
                    score_slow.append(r1)

                    pa_original.append(
                        float(pa0) * 1000.0
                    )

                    pa_slow.append(
                        float(pa1) * 1000.0
                    )


                if not score_original:
                    continue


                per_object.append({
                    "object":
                        old.short_name(
                            mesh_name
                        ),

                    "original_score":
                        float(
                            np.mean(
                                score_original
                            )
                        ),

                    "slow10_score":
                        float(
                            np.mean(
                                score_slow
                            )
                        ),

                    "original_pa":
                        float(
                            np.mean(
                                pa_original
                            )
                        ),

                    "slow10_pa":
                        float(
                            np.mean(
                                pa_slow
                            )
                        ),
                })


            if not per_object:
                continue


            original_score = float(
                np.mean([
                    x[
                        "original_score"
                    ]
                    for x in per_object
                ])
            )

            slow10_score = float(
                np.mean([
                    x[
                        "slow10_score"
                    ]
                    for x in per_object
                ])
            )

            original_pa = float(
                np.mean([
                    x[
                        "original_pa"
                    ]
                    for x in per_object
                ])
            )

            slow10_pa = float(
                np.mean([
                    x[
                        "slow10_pa"
                    ]
                    for x in per_object
                ])
            )


            # -----------------------------------------------
            # ZERO-THRESHOLD decision:
            #
            # Slow down ONLY if rigidity score improves.
            # -----------------------------------------------

            trigger_slow = (
                slow10_score
                < original_score
            )

            if trigger_slow:
                selected = "slow10"
                selected_pa = slow10_pa
            else:
                selected = "original"
                selected_pa = original_pa


            delta_score = (
                slow10_score
                - original_score
            )

            delta_pa = (
                slow10_pa
                - original_pa
            )

            row = {
                "task":
                    TASK,

                "clip":
                    key,

                "original_acc_si":
                    float(
                        acc_lookup[key]
                    ),

                "acc_multiple":
                    float(
                        acc_lookup[key]
                        / BOUNDARY_SI
                    ),

                "original_score":
                    original_score,

                "slow10_score":
                    slow10_score,

                "delta_score":
                    delta_score,

                "original_pa_mm":
                    original_pa,

                "slow10_pa_mm":
                    slow10_pa,

                "delta_pa_mm":
                    delta_pa,

                "trigger_slow":
                    int(
                        trigger_slow
                    ),

                "selected":
                    selected,

                "selected_pa_mm":
                    selected_pa,

                # Does the sign of GT-free score change
                # correctly predict the sign of PA change?
                "score_sign_correct":
                    int(
                        (
                            delta_score < 0
                            and delta_pa < 0
                        )
                        or
                        (
                            delta_score >= 0
                            and delta_pa >= 0
                        )
                    ),
            }

            rows.append(row)
            found.add(key)

            print(
                f"[EVAL] "
                f"{len(found)}/{N_TARGET} "
                f"{key} | "
                f"dR={delta_score:+.6f} "
                f"dPA={delta_pa:+.4f}mm "
                f"select={selected}"
            )

            if len(found) >= N_TARGET:
                break


finally:

    old.omap.pmod.grid_sample_transform = (
        old.omap.ORIG_GRID_SAMPLE
    )

    old.omap.pmod.gather = (
        old.omap.ORIG_GATHER
    )


# ============================================================
# Analysis
# ============================================================

res = pd.DataFrame(rows)

res.to_csv(
    OUT / "per_clip.csv",
    index=False,
)

if len(res) != N_TARGET:
    raise RuntimeError(
        f"Expected {N_TARGET}, "
        f"evaluated {len(res)}"
    )


rho, p = spearmanr(
    res["delta_score"],
    res["delta_pa_mm"],
)

triggered = res[
    res["trigger_slow"] == 1
]

print()
print("=" * 80)
print("ONLINE RIGIDITY REFLEX PoC RESULT")
print("=" * 80)

print(
    "N =",
    len(res),
)

print()

print(
    "Original PA median = %.6f mm"
    % res[
        "original_pa_mm"
    ].median()
)

print(
    "Selected PA median = %.6f mm"
    % res[
        "selected_pa_mm"
    ].median()
)

print()

print(
    "Slow triggered = %d / %d"
    % (
        len(triggered),
        len(res),
    )
)

if len(triggered):

    print(
        "When triggered, slow10 actually "
        "lowered PA = %.1f %%"
        % (
            (
                triggered[
                    "delta_pa_mm"
                ] < 0
            ).mean()
            * 100.0
        )
    )

    print(
        "Triggered clips PA delta median "
        "= %.6f mm"
        % triggered[
            "delta_pa_mm"
        ].median()
    )

print()

print(
    "Score-change sign correctly predicted "
    "PA-change sign = %.1f %%"
    % (
        res[
            "score_sign_correct"
        ].mean()
        * 100.0
    )
)

print(
    "delta rigidity score vs delta PA: "
    f"rho={rho:.4f}, p={p:.3e}"
)

print()
print(
    "SAVED =",
    OUT,
)

print("=" * 80)
