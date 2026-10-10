#!/usr/bin/env python3

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import diagnostics.eval_acc_only_pa_intervention_5task as old


# ============================================================
# CONFIG
# ============================================================

BOUNDARY_SI = 0.24580418
MIN_ACC_MULTIPLE = 1.00
SLOW_FRACTION = 0.10

MIN_POINTS = 8
SEED = 42

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

OUT_ROOT = (
    ROOT
    / "diagnostics/results/"
      "online_rigidity_reflex_4way_5task"
)


def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        required=True,
        choices=[
            "task-0000",
            "task-0001",
            "task-0006",
            "task-0008",
            "task-0010",
        ],
    )
    return p.parse_args()


def safe_name(key):
    return (
        key
        .replace("/", "_")
        .replace(":", "__")
    )


def make_slow10_phase(T):
    """
    Immediate ~10% slowdown proxy.

    phi'(0) ~= 0.9.
    First half progresses slower than original.
    Endpoint stays exactly fixed.

    In online receding-horizon use, only the immediate
    segment would be executed before observing/replanning.
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

    if np.any(np.diff(phi) <= 0):
        raise RuntimeError(
            "slow10 phase is not monotonic"
        )

    return phi


def rigidity_score(predicted, initial_shape):
    """
    Future-GT-free internal rigidity score.

    Initial observed rigid shape vs predicted shape.
    Translation and rotation do not affect pair distances.

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

    tri = np.triu_indices(n, k=1)

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

    rel = (
        np.abs(dp[keep] - d0[keep])
        /
        (d0[keep] + 1e-8)
    )

    return float(np.mean(rel))


def main():

    cli = parse_cli()
    task = cli.task

    out_dir = OUT_ROOT / task
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Population:
    # correction-successful clips AND original Acc >= 1.5 B
    # ========================================================

    df = pd.read_csv(CORR_CSV)

    success = (
        df["success"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    pool = df[
        success
        & (df["task"] == task)
    ].copy()

    pool["acc_multiple"] = (
        pool["original_acc_si"]
        / BOUNDARY_SI
    )

    pool = pool[
        pool["acc_multiple"]
        >= MIN_ACC_MULTIPLE
    ].copy()

    target_keys = set(
        pool["clip"].astype(str)
    )

    acc_lookup = dict(
        zip(
            pool["clip"].astype(str),
            pool["original_acc_si"],
        )
    )

    if len(target_keys) == 0:
        raise RuntimeError(
            f"{task}: no eligible clips"
        )

    print("=" * 80)
    print("ONLINE RIGIDITY REFLEX 4-WAY")
    print("TASK =", task)
    print("ELIGIBLE =", len(target_keys))
    print("Acc criterion >= established boundary")
    print(
        "Variants = Original / Always-slow10 / "
        "Rigidity-reflex / Acc-boundary"
    )
    print("=" * 80)

    # ========================================================
    # Frozen PointWorld
    # ========================================================

    args = old.parse_args(
        skip_command_line=True
    )

    args.seed = SEED
    args.model_path = old.MODEL_PATH
    args.norm_stats_path = old.NORM_STATS
    args.domains = ["behavior"]
    args.data_dirs = [
        old.TASK_DIRS[task]
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
        old.resolve_robot_urdf(domain)
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

    old.omap.pmod.grid_sample_transform = (
        old.omap.grid_sample_with_object_ids
    )

    old.omap.pmod.gather = (
        old.omap.gather_with_object_ids
    )

    old.reset_seed(SEED)

    # ========================================================
    # Official restored WDS
    # ========================================================

    shard_paths = (
        old.dlmod.gather_shard_paths(
            old.TASK_DIRS[task],
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

    labels_order = [
        "original",
        "slow10",
        "boundary",
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

                # ============================================
                # Existing B-boundary phase
                # ============================================

                phase_file = (
                    PHASE_DIR
                    / f"{safe_name(key)}.npz"
                )

                if not phase_file.exists():
                    raise RuntimeError(
                        f"{key}: missing boundary phase"
                    )

                z = np.load(
                    phase_file,
                    allow_pickle=True,
                )

                saved_original_si = (
                    float(z["original_acc"])
                    * 25.0
                )

                if not np.isclose(
                    saved_original_si,
                    float(acc_lookup[key]),
                    atol=1e-7,
                    rtol=1e-5,
                ):
                    raise RuntimeError(
                        f"{key}: Acc population mismatch | "
                        f"csv={acc_lookup[key]} "
                        f"npz={saved_original_si}"
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

                original_phase = np.linspace(
                    0.0,
                    1.0,
                    T,
                    dtype=np.float64,
                )

                slow_phase = (
                    make_slow10_phase(T)
                )

                boundary_phase_raw = (
                    np.asarray(
                        z["phase"],
                        dtype=np.float64,
                    )
                )

                if len(boundary_phase_raw) != T:
                    raise RuntimeError(
                        f"{key}: boundary phase "
                        f"len={len(boundary_phase_raw)} "
                        f"T={T}"
                    )

                boundary_phase = (
                    boundary_phase_raw
                    / float(T - 1)
                )

                phases = {
                    "original":
                        original_phase,
                    "slow10":
                        slow_phase,
                    "boundary":
                        boundary_phase,
                }

                variants = []

                for label in labels_order:

                    sample = (
                        old.src.apply_phase(
                            copy.deepcopy(built),
                            phases[label],
                        )
                    )

                    variants.append(
                        (label, sample)
                    )

                # ============================================
                # Endpoint audit
                # ============================================

                ref = variants[0][1]

                for label, sample in variants[1:]:

                    for ti in [0, -1]:

                        if not np.allclose(
                            ref[
                                "robot_flows"
                            ][ti],
                            sample[
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

                    sample = old.release_process(
                        sample,
                        args,
                        domain,
                        tester.rank,
                    )

                    released.append(
                        (label, sample)
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

                # ============================================
                # Exact scene/object correspondence audit
                # ============================================

                for field in [
                    "scene_flows",
                    "scene_exists",
                    "scene_context_mask",
                    "scene_supervised_mask",
                    "scene_moved_mask",
                    "scene_object_id",
                    "gt_scene_flows",
                ]:

                    ref_field = (
                        batch_cpu[field][0]
                    )

                    for vi in range(
                        1,
                        len(labels_order),
                    ):

                        if not old.same_tensor(
                            ref_field,
                            batch_cpu[field][vi],
                        ):
                            raise RuntimeError(
                                f"{key}: "
                                f"{field} changed"
                            )

                batch_gpu = old.to_device(
                    batch_cpu,
                    device,
                )

                # ============================================
                # FORWARD #1
                #
                # Decision forward.
                # Reflex uses ONLY:
                #   R(original) vs R(slow10)
                #
                # No future geometry GT used for the decision.
                # ============================================

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

                # ============================================
                # FORWARD #2
                #
                # Independent PA evaluation.
                # ============================================

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

                object_ids = (
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

                # ============================================
                # Online comparison:
                #
                # First half only, because slow10 is
                # deliberately slower in this immediate region.
                # ============================================

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
                        object_ids == oid
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

                    score_dict = {
                        label: []
                        for label in labels_order
                    }

                    pa_dict = {
                        label: []
                        for label in labels_order
                    }

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

                        frame_ok = True

                        for vi, label in enumerate(
                            labels_order
                        ):

                            rs = rigidity_score(
                                pred_score[
                                    vi,
                                    t,
                                    m,
                                ],
                                x0,
                            )

                            if not np.isfinite(rs):
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
                                        x0,
                                    )
                                )
                            except Exception:
                                frame_ok = False
                                break

                            score_dict[
                                label
                            ].append(rs)

                            pa_dict[
                                label
                            ].append(
                                float(pa)
                                * 1000.0
                            )

                        if not frame_ok:

                            for label in labels_order:

                                if score_dict[label]:
                                    score_dict[
                                        label
                                    ].pop()

                                if pa_dict[label]:
                                    pa_dict[
                                        label
                                    ].pop()

                    if not score_dict[
                        "original"
                    ]:
                        continue

                    obj = {
                        "object":
                            old.short_name(
                                mesh_name
                            )
                    }

                    for label in labels_order:

                        obj[
                            f"{label}_score"
                        ] = float(
                            np.mean(
                                score_dict[label]
                            )
                        )

                        obj[
                            f"{label}_pa"
                        ] = float(
                            np.mean(
                                pa_dict[label]
                            )
                        )

                    per_object.append(obj)

                if not per_object:
                    continue

                # ============================================
                # Clip-level values
                # ============================================

                clip_score = {}

                clip_pa = {}

                for label in labels_order:

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

                delta_score = (
                    clip_score["slow10"]
                    - clip_score["original"]
                )

                delta_pa = (
                    clip_pa["slow10"]
                    - clip_pa["original"]
                )

                # ============================================
                # ZERO-THRESHOLD reflex
                #
                # Slow only if predicted rigidity improves.
                # ============================================

                trigger_slow = (
                    delta_score < 0
                )

                if trigger_slow:
                    reflex_pa = (
                        clip_pa["slow10"]
                    )
                    selected = "slow10"
                else:
                    reflex_pa = (
                        clip_pa["original"]
                    )
                    selected = "original"

                oracle_label = min(
                    ["original", "slow10"],
                    key=lambda x:
                        clip_pa[x],
                )

                oracle_pa = (
                    clip_pa[
                        oracle_label
                    ]
                )

                score_sign_correct = int(
                    (
                        delta_score < 0
                        and delta_pa < 0
                    )
                    or
                    (
                        delta_score >= 0
                        and delta_pa >= 0
                    )
                )

                row = {
                    "task":
                        task,

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
                        clip_score[
                            "original"
                        ],

                    "slow10_score":
                        clip_score[
                            "slow10"
                        ],

                    "boundary_score":
                        clip_score[
                            "boundary"
                        ],

                    "delta_score":
                        delta_score,

                    "original_pa_mm":
                        clip_pa[
                            "original"
                        ],

                    "slow10_pa_mm":
                        clip_pa[
                            "slow10"
                        ],

                    "boundary_pa_mm":
                        clip_pa[
                            "boundary"
                        ],

                    "delta_pa_slow_mm":
                        delta_pa,

                    "trigger_slow":
                        int(
                            trigger_slow
                        ),

                    "reflex_selected":
                        selected,

                    "reflex_pa_mm":
                        reflex_pa,

                    "oracle_label":
                        oracle_label,

                    "oracle_pa_mm":
                        oracle_pa,

                    "selection_equals_oracle":
                        int(
                            selected
                            == oracle_label
                        ),

                    "score_sign_correct":
                        score_sign_correct,

                    "slow_lower_than_original":
                        int(
                            clip_pa["slow10"]
                            <
                            clip_pa["original"]
                        ),

                    "boundary_lower_than_original":
                        int(
                            clip_pa["boundary"]
                            <
                            clip_pa["original"]
                        ),

                    "reflex_lower_than_original":
                        int(
                            reflex_pa
                            <
                            clip_pa["original"]
                        ),

                    "reflex_lower_than_slow":
                        int(
                            reflex_pa
                            <
                            clip_pa["slow10"]
                        ),

                    "reflex_lower_than_boundary":
                        int(
                            reflex_pa
                            <
                            clip_pa["boundary"]
                        ),
                }

                rows.append(row)
                found.add(key)

                print(
                    f"[EVAL] "
                    f"{len(found)}/"
                    f"{len(target_keys)} "
                    f"{key} | "
                    f"dR={delta_score:+.6f} "
                    f"dPA={delta_pa:+.4f}mm "
                    f"reflex={selected} | "
                    f"PA "
                    f"O={clip_pa['original']:.4f} "
                    f"S={clip_pa['slow10']:.4f} "
                    f"R={reflex_pa:.4f} "
                    f"B={clip_pa['boundary']:.4f}"
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

    res = pd.DataFrame(rows)

    out_csv = (
        out_dir
        / "per_clip.csv"
    )

    res.to_csv(
        out_csv,
        index=False,
    )

    print()
    print("=" * 80)
    print("TASK COMPLETE")
    print("TASK =", task)
    print("TARGET =", len(target_keys))
    print("EVALUATED =", len(res))

    if len(res):

        print(
            "Original median = %.6f"
            % res[
                "original_pa_mm"
            ].median()
        )

        print(
            "Always slow median = %.6f"
            % res[
                "slow10_pa_mm"
            ].median()
        )

        print(
            "Reflex median = %.6f"
            % res[
                "reflex_pa_mm"
            ].median()
        )

        print(
            "Boundary median = %.6f"
            % res[
                "boundary_pa_mm"
            ].median()
        )

        print(
            "Reflex selection==oracle = %.1f%%"
            % (
                res[
                    "selection_equals_oracle"
                ].mean()
                * 100
            )
        )

        print(
            "Score sign correct = %.1f%%"
            % (
                res[
                    "score_sign_correct"
                ].mean()
                * 100
            )
        )

    print("SAVED =", out_csv)
    print("=" * 80)


if __name__ == "__main__":
    main()
