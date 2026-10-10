#!/usr/bin/env python3

import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import diagnostics.build_acc_only_boundary_correction_current10457 as corr
import diagnostics.eval_acc_only_pa_intervention_5task as old


# ============================================================
# PoC protocol
# ============================================================

TASK = "task-0000"
N_TARGET = 30

# Only clearly high-Acc clips for this first feasibility test.
MIN_BASE_MULTIPLE = 1.50

B = corr.ACC_BOUNDARY_NATIVE

TARGETS = {
    "boundary": 1.00 * B,
    "acc90":    0.90 * B,
    "acc80":    0.80 * B,
}

ROOT = Path(__file__).resolve().parents[1]

CORR_ROOT = (
    ROOT
    / "diagnostics/results/"
      "acc_only_boundary_correction_current10457"
)

CORR_CSV = (
    CORR_ROOT
    / "correction_results.csv"
)

PHASE_DIR = (
    CORR_ROOT
    / "corrected_actions"
)

OUT = (
    ROOT
    / "diagnostics/results/"
      "rigidity_guided_action_poc"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

MIN_POINTS = 8
SEED = 42


# ============================================================
# Helpers
# ============================================================

def safe_name(key):
    return (
        key
        .replace("/", "_")
        .replace(":", "__")
    )


def pairwise_rigid_score(
    predicted,
    reference,
):
    """
    GT-FUTURE-FREE score.

    reference:
        initial rigid-object shape only.

    Measures how much internal pairwise distances change.
    Translation / rotation do not affect this score.

    Lower = more rigidly consistent prediction.
    """

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    n = len(reference)

    if n < MIN_POINTS:
        return np.nan

    tri = np.triu_indices(
        n,
        k=1,
    )

    d0 = np.linalg.norm(
        reference[:, None, :]
        - reference[None, :, :],
        axis=-1,
    )[tri]

    dp = np.linalg.norm(
        predicted[:, None, :]
        - predicted[None, :, :],
        axis=-1,
    )[tri]

    # Avoid essentially coincident point pairs.
    keep = (
        np.isfinite(d0)
        & np.isfinite(dp)
        & (d0 > 1e-4)
    )

    if keep.sum() == 0:
        return np.nan

    rel = (
        np.abs(
            dp[keep]
            - d0[keep]
        )
        /
        (
            d0[keep]
            + 1e-8
        )
    )

    return float(
        np.mean(rel)
    )


# ============================================================
# 1. Build multiple Acc-feasible Action candidates
#    using the SAME established Acc implementation.
# ============================================================

df = pd.read_csv(
    CORR_CSV
)

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

pool["base_multiple"] = (
    pool["original_acc_si"]
    /
    corr.ACC_BOUNDARY_SI
)

pool = pool[
    pool["base_multiple"]
    >= MIN_BASE_MULTIPLE
].copy()

candidate_keys = set(
    pool["clip"].astype(str)
)

print("=" * 80)
print("RIGIDITY-GUIDED ACTION PoC")
print("TASK =", TASK)
print("High-Acc pool =", len(candidate_keys))
print("Need full feasible =", N_TARGET)
print(
    "Candidate Acc targets =",
    {
        k: v * 25.0
        for k, v in TARGETS.items()
    },
)
print("=" * 80)


# Official dataloader:
# this is the SAME route used to establish current 10,457 Acc values.

pw = corr.parse_args(
    skip_command_line=True
)

pw.seed = SEED
pw.domains = ["behavior"]
pw.data_dirs = [
    corr.TASK_DIRS[TASK]
]
pw.batch_size = 1
pw.num_workers = 0
pw.eval_num_workers = 0

loader, _ = corr.build_dataloader(
    args=pw,
    mode="test",
    rank=0,
    world_size=1,
    override_splits="test",
    force_resampled_eval=False,
)

phase_candidates = {}

for batch in loader:

    key = corr.normalize_key(
        batch["__key__"]
    )

    if key not in candidate_keys:
        continue

    right = (
        batch[
            "right_gripper_pose"
        ][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    left = (
        batch[
            "left_gripper_pose"
        ][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    original = corr.acc_only_metric(
        right,
        left,
    )

    # --------------------------------------------------------
    # Existing standard B correction:
    # preserve exact baseline used in previous experiment.
    # --------------------------------------------------------

    phase_file = (
        PHASE_DIR
        / f"{safe_name(key)}.npz"
    )

    if not phase_file.exists():
        continue

    z = np.load(
        phase_file,
        allow_pickle=True,
    )

    if not np.isclose(
        float(z["original_acc"]),
        original["rms_acceleration"],
        atol=1e-10,
        rtol=1e-8,
    ):
        raise RuntimeError(
            f"{key}: original Acc audit failed"
        )

    phase_B = np.asarray(
        z["phase"],
        dtype=np.float64,
    )

    acc_B = float(
        z["corrected_acc"]
    )

    # --------------------------------------------------------
    # Stronger correction candidates: 0.9B, 0.8B.
    # --------------------------------------------------------

    result90 = (
        corr.optimize_phase_acc_only(
            right,
            left,
            TARGETS["acc90"],
        )
    )

    if not result90["success"]:
        continue

    result80 = (
        corr.optimize_phase_acc_only(
            right,
            left,
            TARGETS["acc80"],
        )
    )

    if not result80["success"]:
        continue

    phase_candidates[key] = {
        "original_acc":
            float(
                original[
                    "rms_acceleration"
                ]
            ),

        "boundary": {
            "phase":
                phase_B,
            "acc":
                acc_B,
        },

        "acc90": {
            "phase":
                np.asarray(
                    result90["phase"],
                    dtype=np.float64,
                ),
            "acc":
                float(
                    result90[
                        "corrected_acc"
                    ]
                ),
        },

        "acc80": {
            "phase":
                np.asarray(
                    result80["phase"],
                    dtype=np.float64,
                ),
            "acc":
                float(
                    result80[
                        "corrected_acc"
                    ]
                ),
        },
    }

    print(
        f"[CANDIDATE] "
        f"{len(phase_candidates)}/{N_TARGET} "
        f"{key} | "
        f"base={original['rms_acceleration']*25:.3f} "
        f"B={acc_B*25:.3f} "
        f"0.9B={result90['corrected_acc']*25:.3f} "
        f"0.8B={result80['corrected_acc']*25:.3f}"
    )

    if (
        len(phase_candidates)
        >= N_TARGET
    ):
        break


if len(phase_candidates) < N_TARGET:
    raise RuntimeError(
        f"Only {len(phase_candidates)} "
        f"fully-feasible candidates; "
        f"need {N_TARGET}"
    )


target_keys = set(
    phase_candidates
)

print()
print(
    "FULL-FEASIBLE PoC CLIPS =",
    len(target_keys),
)


# ============================================================
# 2. Frozen PointWorld
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


# Object-ID propagation only.
old.omap.pmod.grid_sample_transform = (
    old.omap.grid_sample_with_object_ids
)

old.omap.pmod.gather = (
    old.omap.gather_with_object_ids
)

old.reset_seed(SEED)


# ============================================================
# 3. One-pass restored WDS
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
).shuffle(
    shard_paths
)

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


clip_rows = []
candidate_rows = []

found = set()

LABELS = [
    "base",
    "boundary",
    "acc90",
    "acc80",
]

SELECTABLE = [
    "boundary",
    "acc90",
    "acc80",
]


try:

    with torch.inference_mode():

        for raw in dataset:

            raw_key = str(
                raw.get(
                    "__key__",
                    "",
                )
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

            meta = phase_candidates[
                key
            ]

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

            phases = {
                "base":
                    np.linspace(
                        0.0,
                        1.0,
                        T,
                    ),
            }

            for label in SELECTABLE:

                p = (
                    meta[label][
                        "phase"
                    ]
                )

                if len(p) != T:
                    raise RuntimeError(
                        f"{key}: phase/T mismatch"
                    )

                phases[label] = (
                    p
                    / float(T - 1)
                )

            # ------------------------------------------------
            # Build all Action variants.
            # ------------------------------------------------

            variants = []

            for label in LABELS:

                v = old.src.apply_phase(
                    copy.deepcopy(
                        built
                    ),
                    phases[label],
                )

                variants.append(
                    (
                        label,
                        v,
                    )
                )

            # Endpoint preservation audit.
            ref = variants[0][1]

            for label, v in variants[1:]:

                for ti in [
                    0,
                    -1,
                ]:

                    if not np.allclose(
                        ref[
                            "robot_flows"
                        ][ti],
                        v[
                            "robot_flows"
                        ][ti],
                        atol=1e-7,
                        rtol=0,
                    ):
                        raise RuntimeError(
                            f"{key}: "
                            f"{label} endpoint changed"
                        )

            released = []

            for label, sample in variants:

                sample = (
                    old.release_process(
                        sample,
                        args,
                        domain,
                        tester.rank,
                    )
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

            # Scene / object correspondence audit.
            for field in [
                "scene_flows",
                "scene_exists",
                "scene_context_mask",
                "scene_supervised_mask",
                "scene_moved_mask",
                "scene_object_id",
                "gt_scene_flows",
            ]:

                r = batch_cpu[
                    field
                ][0]

                for vi in range(
                    1,
                    len(LABELS),
                ):

                    if not old.same_tensor(
                        r,
                        batch_cpu[
                            field
                        ][vi],
                    ):
                        raise RuntimeError(
                            f"{key}: "
                            f"{field} changed"
                        )

            batch_gpu = (
                old.to_device(
                    batch_cpu,
                    device,
                )
            )

            # =================================================
            # IMPORTANT:
            #
            # Forward #1 = choose Action using ONLY
            #              predicted rigidity score.
            #
            # Forward #2 = independent prediction used
            #              for PA-MPJPE evaluation.
            #
            # This reduces "lucky stochastic sample" bias.
            # =================================================

            out_select = tester.model(
                batch_gpu,
                training=False,
            )

            pred_select = (
                out_select[
                    "scene_flows"
                ]
                .detach()
                .cpu()
                .numpy()
            )

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
                old.omap.OBJECT_TABLE[
                    key
                ]
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


                scores = {
                    x: []
                    for x in LABELS
                }

                pas = {
                    x: []
                    for x in LABELS
                }


                for t in range(
                    1,
                    gt.shape[0],
                ):

                    m = (
                        object_mask[t]
                        & valid[t]
                        & object_mask[0]
                        & exists[0]
                    )

                    n = int(
                        m.sum()
                    )

                    if n < MIN_POINTS:
                        continue

                    # Initial rigid shape.
                    # No future GT is used for Action selection.
                    ref0 = gt[
                        0,
                        m,
                    ]

                    centered = (
                        ref0
                        - ref0.mean(
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


                    frame_ok = True

                    for vi, label in enumerate(
                        LABELS
                    ):

                        score = (
                            pairwise_rigid_score(
                                pred_select[
                                    vi,
                                    t,
                                    m,
                                ],
                                ref0,
                            )
                        )

                        if not np.isfinite(
                            score
                        ):
                            frame_ok = False
                            break

                        try:
                            pa, _ = (
                                old.p_mpjpe_single(
                                    pred_eval[
                                        vi,
                                        t,
                                        m,
                                    ],
                                    ref0,
                                )
                            )
                        except Exception:
                            frame_ok = False
                            break

                        scores[
                            label
                        ].append(
                            score
                        )

                        pas[
                            label
                        ].append(
                            float(pa)
                            * 1000.0
                        )

                    if not frame_ok:
                        for label in LABELS:
                            if scores[label]:
                                scores[label].pop()
                            if pas[label]:
                                pas[label].pop()


                if (
                    not scores[
                        "boundary"
                    ]
                ):
                    continue


                obj = {
                    "object":
                        old.short_name(
                            mesh_name
                        )
                }

                for label in LABELS:

                    obj[
                        f"{label}_score"
                    ] = float(
                        np.mean(
                            scores[label]
                        )
                    )

                    obj[
                        f"{label}_pa"
                    ] = float(
                        np.mean(
                            pas[label]
                        )
                    )

                per_object.append(
                    obj
                )


            if not per_object:
                continue


            # =================================================
            # Clip-level score / PA
            # =================================================

            clip_score = {}

            clip_pa = {}

            for label in LABELS:

                clip_score[label] = float(
                    np.mean([
                        x[
                            f"{label}_score"
                        ]
                        for x in per_object
                    ])
                )

                clip_pa[label] = float(
                    np.mean([
                        x[
                            f"{label}_pa"
                        ]
                        for x in per_object
                    ])
                )


            # Action selection uses ONLY predicted rigidity score.
            selected = min(
                SELECTABLE,
                key=lambda x:
                    clip_score[x],
            )

            # Oracle is analysis only:
            # how much room was theoretically available.
            oracle = min(
                SELECTABLE,
                key=lambda x:
                    clip_pa[x],
            )


            row = {
                "task": TASK,
                "clip": key,

                "base_acc_si":
                    meta[
                        "original_acc"
                    ] * 25.0,

                "base_multiple":
                    meta[
                        "original_acc"
                    ] / B,

                "selected":
                    selected,

                "oracle":
                    oracle,

                "base_pa_mm":
                    clip_pa[
                        "base"
                    ],

                "boundary_pa_mm":
                    clip_pa[
                        "boundary"
                    ],

                "selected_pa_mm":
                    clip_pa[
                        selected
                    ],

                "oracle_pa_mm":
                    clip_pa[
                        oracle
                    ],

                "selected_delta_vs_boundary_mm":
                    (
                        clip_pa[
                            selected
                        ]
                        -
                        clip_pa[
                            "boundary"
                        ]
                    ),

                "selected_lower_than_boundary":
                    int(
                        clip_pa[
                            selected
                        ]
                        <
                        clip_pa[
                            "boundary"
                        ]
                    ),

                "selection_equals_oracle":
                    int(
                        selected == oracle
                    ),

                "boundary_score":
                    clip_score[
                        "boundary"
                    ],

                "selected_score":
                    clip_score[
                        selected
                    ],
            }


            for label in SELECTABLE:

                row[
                    f"{label}_score"
                ] = clip_score[
                    label
                ]

                row[
                    f"{label}_pa_mm"
                ] = clip_pa[
                    label
                ]

                candidate_rows.append({
                    "clip":
                        key,
                    "candidate":
                        label,
                    "rigidity_score":
                        clip_score[
                            label
                        ],
                    "pa_mm":
                        clip_pa[
                            label
                        ],
                })


            clip_rows.append(
                row
            )

            found.add(
                key
            )

            print(
                f"[EVAL] "
                f"{len(found)}/"
                f"{len(target_keys)} "
                f"{key} | "
                f"select={selected} "
                f"B_PA={clip_pa['boundary']:.4f} "
                f"SEL_PA={clip_pa[selected]:.4f}"
            )

            if len(found) == len(
                target_keys
            ):
                break


finally:

    old.omap.pmod.grid_sample_transform = (
        old.omap.ORIG_GRID_SAMPLE
    )

    old.omap.pmod.gather = (
        old.omap.ORIG_GATHER
    )


# ============================================================
# Final PoC analysis
# ============================================================

res = pd.DataFrame(
    clip_rows
)

cand = pd.DataFrame(
    candidate_rows
)

res.to_csv(
    OUT / "per_clip_poc.csv",
    index=False,
)

cand.to_csv(
    OUT / "candidate_score_pa.csv",
    index=False,
)


if len(res) != N_TARGET:
    raise RuntimeError(
        f"Expected {N_TARGET} "
        f"evaluated clips, got {len(res)}"
    )


rho, p = spearmanr(
    cand[
        "rigidity_score"
    ],
    cand[
        "pa_mm"
    ],
)


print()
print("=" * 80)
print("RIGIDITY-GUIDED ACTION PoC RESULT")
print("=" * 80)

print(
    "N =",
    len(res),
)

print()

print(
    "Original PA median       = %.6f mm"
    % res[
        "base_pa_mm"
    ].median()
)

print(
    "Boundary PA median       = %.6f mm"
    % res[
        "boundary_pa_mm"
    ].median()
)

print(
    "Rigidity-selected median = %.6f mm"
    % res[
        "selected_pa_mm"
    ].median()
)

print(
    "Oracle-best median       = %.6f mm"
    % res[
        "oracle_pa_mm"
    ].median()
)

print()

print(
    "Selected vs Boundary "
    "paired delta median = %.6f mm"
    % res[
        "selected_delta_vs_boundary_mm"
    ].median()
)

print(
    "Selected PA lower than "
    "Boundary = %.1f %%"
    % (
        res[
            "selected_lower_than_boundary"
        ].mean()
        * 100.0
    )
)

print(
    "Selection == PA oracle = %.1f %%"
    % (
        res[
            "selection_equals_oracle"
        ].mean()
        * 100.0
    )
)

print()

print(
    "Rigidity score vs "
    "independent PA: "
    f"rho={rho:.4f}, p={p:.3e}"
)

print()

print(
    "Selected Action counts:"
)

print(
    res[
        "selected"
    ]
    .value_counts()
    .to_string()
)

print()
print(
    "SAVED =",
    OUT,
)

print("=" * 80)
