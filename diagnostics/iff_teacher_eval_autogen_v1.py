#!/usr/bin/env python3

import os
import copy
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Reuse the already validated intervention evaluator helpers.
import diagnostics.eval_acc_only_pa_intervention_5task as old


ROOT = Path(__file__).resolve().parents[1]

CORR_ROOT = (
    ROOT
    / "diagnostics/results/"
      "acc_only_boundary_correction_current10457"
)

CORR_CSV = CORR_ROOT / "correction_results.csv"
PHASE_DIR = CORR_ROOT / "corrected_actions"

OUT_ROOT = (
    ROOT
    / "diagnostics/results/"
      "iff_teacher_probe_v1_20261010"
)

EXPECTED_SUCCESS = 2683
MIN_POINTS = 8
SEED = 42


def successful_corrections(task):

    d = pd.read_csv(CORR_CSV)

    success = (
        d["success"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    all_success = d[success].copy()

    if len(all_success) != EXPECTED_SUCCESS:
        raise RuntimeError(
            f"Expected {EXPECTED_SUCCESS} successful corrections, "
            f"got {len(all_success)}"
        )

    q = all_success[
        all_success["task"].astype(str) == task
    ].copy()

    q["clip"] = q["clip"].astype(str)

    if q["clip"].duplicated().any():
        raise RuntimeError("duplicate correction clip")

    return q.set_index("clip")


def load_phase(key, expected_task):

    safe = (
        key
        .replace("/", "_")
        .replace(":", "__")
    )

    path = PHASE_DIR / f"{safe}.npz"

    if not path.exists():
        raise RuntimeError(
            f"{key}: correction phase file missing: {path}"
        )

    z = np.load(
        path,
        allow_pickle=True,
    )

    saved_key = str(z["clip"].item())
    saved_task = str(z["task"].item())

    if saved_key != key:
        raise RuntimeError(
            f"{key}: saved clip mismatch {saved_key}"
        )

    if saved_task != expected_task:
        raise RuntimeError(
            f"{key}: saved task mismatch {saved_task}"
        )

    return {
        "phase":
            np.asarray(
                z["phase"],
                dtype=np.float64,
            ),
        "original_acc":
            float(z["original_acc"].item()),
        "corrected_acc":
            float(z["corrected_acc"].item()),
        "boundary":
            float(z["acc_boundary"].item()),
    }


def save_csv(path, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(old.TASK_DIRS.keys()),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    cli = parser.parse_args()

    task = cli.task
    corr = successful_corrections(task)

    target_keys = set(
        corr.index.astype(str)
    )

    if cli.smoke:
        target_keys = set(
            sorted(target_keys)[:5]
        )

    if not target_keys:
        raise RuntimeError(
            f"{task}: no successful correction targets"
        )

    print("=" * 80)
    print("ACC-ONLY BOUNDARY CORRECTION PA-MPJPE")
    print("TASK =", task)
    print("TARGETS =", len(target_keys))
    print("VARIANTS = Original + Original repeat + Corrected")
    print("JERK/MSJ = NOT USED")
    print("=" * 80)

    # ---------------------------------------------------------
    # Frozen PointWorld
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Official robot sampler
    # ---------------------------------------------------------

    urdf_path = old.resolve_robot_urdf(
        domain
    )

    robot_sampler = old.TorchRobotSampler(
        urdf_path=urdf_path,
        gripper_only=old.RELEASE_GRIPPER_ONLY,
        device="cpu",
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

    # ---------------------------------------------------------
    # Object-ID hooks:
    # coordinates / GT themselves are unchanged.
    # ---------------------------------------------------------

    old.omap.pmod.grid_sample_transform = (
        old.omap.grid_sample_with_object_ids
    )

    old.omap.pmod.gather = (
        old.omap.gather_with_object_ids
    )

    old.reset_seed(SEED)

    # ---------------------------------------------------------
    # Original restored WDS: one pass
    # ---------------------------------------------------------

    shard_paths = old.dlmod.gather_shard_paths(
        old.TASK_DIRS[task],
        "test",
        rank=tester.rank,
    )

    if not shard_paths:
        raise RuntimeError(
            f"{task}: no test shards"
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

    found = set()

    object_rows = []
    clip_rows = []
    status_rows = []

    forwards = 0

    try:

        with torch.inference_mode():

            for raw in dataset:

                raw_key = str(
                    raw.get("__key__", "")
                )

                # Early filter.
                if raw_key not in target_keys:
                    continue

                if raw_key in found:
                    raise RuntimeError(
                        f"duplicate target {raw_key}"
                    )

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

                if key != raw_key:
                    raise RuntimeError(
                        f"{raw_key}: decoded key={key}"
                    )

                meta = load_phase(
                    key,
                    task,
                )
                from diagnostics.iff_counterfactual_probe_20261010 import perturb_meta_phase
                # Phase perturbation is applied after official build.

                # -------------------------------------------------
                # Build official PointWorld input ONCE.
                # -------------------------------------------------

                built = (
                    old.omap.build_flow_sample_with_object_ids(
                        decoded,
                        domain=domain,
                        robot_sampler=robot_sampler,
                        max_robot_points=args.max_robot_points,
                        deterministic=True,
                        seed=args.seed + tester.rank,
                        force_single_arm=force_single_arm,
                    )
                )
                meta = perturb_meta_phase(meta, built, key, 1)

                T = int(
                    built["robot_flows"].shape[0]
                )

                phase_frames = meta["phase"]

                if len(phase_frames) != T:
                    raise RuntimeError(
                        f"{key}: phase length "
                        f"{len(phase_frames)} != {T}"
                    )

                base_phase = np.linspace(
                    0.0,
                    1.0,
                    T,
                )

                corrected_phase = (
                    phase_frames
                    / float(T - 1)
                )

                # -------------------------------------------------
                # Original / corrected Action.
                # -------------------------------------------------

                base_unreleased = (
                    old.src.apply_phase(
                        copy.deepcopy(built),
                        base_phase,
                    )
                )

                corrected_unreleased = (
                    old.src.apply_phase(
                        copy.deepcopy(built),
                        corrected_phase,
                    )
                )

                # Hard start/end audit.
                for ti, name in [
                    (0, "START"),
                    (-1, "END"),
                ]:

                    if not np.allclose(
                        base_unreleased["robot_flows"][ti],
                        corrected_unreleased["robot_flows"][ti],
                        atol=1e-7,
                        rtol=0,
                    ):
                        raise RuntimeError(
                            f"{key}: {name} changed"
                        )

                variants = [
                    (
                        "base",
                        copy.deepcopy(
                            base_unreleased
                        ),
                    ),
                    (
                        "base_repeat",
                        copy.deepcopy(
                            base_unreleased
                        ),
                    ),
                    (
                        "corrected",
                        copy.deepcopy(
                            corrected_unreleased
                        ),
                    ),
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

                # Scene / GT must remain identical.
                for field in [
                    "scene_flows",
                    "scene_exists",
                    "scene_context_mask",
                    "scene_supervised_mask",
                    "scene_moved_mask",
                    "scene_object_id",
                    "gt_scene_flows",
                ]:

                    if field not in batch_cpu:
                        raise RuntimeError(
                            f"{key}: missing {field}"
                        )

                    ref = batch_cpu[field][0]

                    for vi in [1, 2]:

                        if not old.same_tensor(
                            ref,
                            batch_cpu[field][vi],
                        ):
                            raise RuntimeError(
                                f"{key}: {field} changed "
                                f"for variant {vi}"
                            )

                batch_gpu = old.to_device(
                    batch_cpu,
                    device,
                )

                # -------------------------------------------------
                # ONE GPU forward:
                # original + repeat + corrected
                # -------------------------------------------------

                outputs = tester.model(
                    batch_gpu,
                    training=False,
                )

                pred = (
                    outputs["scene_flows"]
                    .detach()
                    .cpu()
                    .numpy()
                )

                forwards += 1

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

                if key not in old.omap.OBJECT_TABLE:
                    raise RuntimeError(
                        f"{key}: object table missing"
                    )

                table = (
                    old.omap.OBJECT_TABLE[key]
                )

                eligible_objects = []

                # =================================================
                # Rigid object PA-MPJPE
                #
                # GT0 = initial rigid shape.
                # Therefore corrected Action does NOT need a new
                # counterfactual future GT.
                # =================================================

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

                    vals = {
                        "base": [],
                        "base_repeat": [],
                        "corrected": [],
                    }

                    gt_checks = []

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

                        n = int(m.sum())

                        if n < MIN_POINTS:
                            continue

                        gt0 = gt[
                            0,
                            m,
                        ]

                        centered = (
                            gt0
                            - gt0.mean(
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

                        try:

                            gt_pa, _ = (
                                old.p_mpjpe_single(
                                    gt[t, m],
                                    gt0,
                                )
                            )

                        except Exception:
                            continue

                        frame_vals = {}

                        ok = True

                        for vi, label in enumerate(
                            [
                                "base",
                                "base_repeat",
                                "corrected",
                            ]
                        ):

                            try:

                                pa, _ = (
                                    old.p_mpjpe_single(
                                        pred[
                                            vi,
                                            t,
                                            m,
                                        ],
                                        gt0,
                                    )
                                )

                            except Exception:
                                ok = False
                                break

                            frame_vals[label] = (
                                float(pa)
                                * 1000.0
                            )

                        if not ok:
                            continue

                        gt_checks.append(
                            float(gt_pa)
                            * 1000.0
                        )

                        for label in vals:
                            vals[label].append(
                                frame_vals[label]
                            )

                    if not gt_checks:
                        continue

                    if any(
                        len(vals[x])
                        != len(gt_checks)
                        for x in vals
                    ):
                        continue

                    row = {
                        "task": task,
                        "clip": key,
                        "object_id": oid,
                        "object_full_name":
                            str(mesh_name),
                        "object_short_name":
                            old.short_name(
                                mesh_name
                            ),

                        "eligible_future_frames":
                            len(gt_checks),

                        "original_acc_si_m_s2":
                            meta[
                                "original_acc"
                            ] * 25.0,

                        "corrected_acc_si_m_s2":
                            meta[
                                "corrected_acc"
                            ] * 25.0,

                        "boundary_si_m_s2":
                            meta[
                                "boundary"
                            ] * 25.0,

                        "gt_shape_check_pa_mean_mm":
                            float(
                                np.mean(gt_checks)
                            ),

                        "base_pa_mean_mm":
                            float(
                                np.mean(
                                    vals["base"]
                                )
                            ),

                        "base_repeat_pa_mean_mm":
                            float(
                                np.mean(
                                    vals[
                                        "base_repeat"
                                    ]
                                )
                            ),

                        "corrected_pa_mean_mm":
                            float(
                                np.mean(
                                    vals[
                                        "corrected"
                                    ]
                                )
                            ),
                    }

                    row[
                        "base_repeat_absdiff_mm"
                    ] = abs(
                        row[
                            "base_repeat_pa_mean_mm"
                        ]
                        -
                        row[
                            "base_pa_mean_mm"
                        ]
                    )

                    row[
                        "corrected_delta_pa_mm"
                    ] = (
                        row[
                            "corrected_pa_mean_mm"
                        ]
                        -
                        row[
                            "base_pa_mean_mm"
                        ]
                    )

                    row[
                        "corrected_pa_ratio"
                    ] = (
                        row[
                            "corrected_pa_mean_mm"
                        ]
                        /
                        row[
                            "base_pa_mean_mm"
                        ]
                        if row[
                            "base_pa_mean_mm"
                        ] > 0
                        else np.nan
                    )

                    object_rows.append(
                        row
                    )

                    eligible_objects.append(
                        row
                    )

                if eligible_objects:

                    base = float(
                        np.mean([
                            r[
                                "base_pa_mean_mm"
                            ]
                            for r
                            in eligible_objects
                        ])
                    )

                    repeat = float(
                        np.mean([
                            r[
                                "base_repeat_pa_mean_mm"
                            ]
                            for r
                            in eligible_objects
                        ])
                    )

                    corrected = float(
                        np.mean([
                            r[
                                "corrected_pa_mean_mm"
                            ]
                            for r
                            in eligible_objects
                        ])
                    )

                    clip_rows.append({
                        "task": task,
                        "clip": key,

                        "eligible_objects":
                            len(
                                eligible_objects
                            ),

                        "original_acc_si_m_s2":
                            meta[
                                "original_acc"
                            ] * 25.0,

                        "corrected_acc_si_m_s2":
                            meta[
                                "corrected_acc"
                            ] * 25.0,

                        "base_pa_mean_mm":
                            base,

                        "base_repeat_pa_mean_mm":
                            repeat,

                        "corrected_pa_mean_mm":
                            corrected,

                        "base_repeat_absdiff_mm":
                            abs(
                                repeat
                                - base
                            ),

                        "corrected_delta_pa_mm":
                            corrected
                            - base,

                        "corrected_pa_ratio":
                            (
                                corrected
                                / base
                                if base > 0
                                else np.nan
                            ),

                        "corrected_lower":
                            int(
                                corrected
                                < base
                            ),
                    })

                    status = "OK"

                else:
                    status = (
                        "NO_ELIGIBLE_OBJECT"
                    )

                status_rows.append({
                    "task": task,
                    "clip": key,
                    "status": status,
                    "eligible_objects":
                        len(
                            eligible_objects
                        ),
                })

                found.add(key)

                nfound = len(found)

                if (
                    nfound <= 5
                    or nfound % 50 == 0
                    or nfound
                    == len(target_keys)
                ):
                    print(
                        f"[{task}] "
                        f"{nfound}/"
                        f"{len(target_keys)} "
                        f"objects="
                        f"{len(eligible_objects)} "
                        f"{key}"
                    )

                if (
                    len(found)
                    == len(target_keys)
                ):
                    break

        missing = (
            target_keys
            - found
        )

        if missing:
            raise RuntimeError(
                f"{task}: missing "
                f"{len(missing)} targets"
            )

        outdir = (
            OUT_ROOT
            / (
                f"smoke_{task}"
                if cli.smoke
                else task
            )
        )

        outdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_csv(
            outdir
            / "per_object_metrics.csv",
            object_rows,
        )

        save_csv(
            outdir
            / "per_clip_metrics.csv",
            clip_rows,
        )

        save_csv(
            outdir
            / "clip_status.csv",
            status_rows,
        )

        c = pd.DataFrame(
            clip_rows
        )

        print()
        print("=" * 80)
        print("FINAL AUDIT")
        print("TASK =", task)
        print("TARGETS =", len(target_keys))
        print("FOUND =", len(found))
        print("MODEL FORWARDS =", forwards)
        print("ELIGIBLE CLIPS =", len(c))

        if len(c):

            print(
                "ORIGINAL PA median mm =",
                c[
                    "base_pa_mean_mm"
                ].median(),
            )

            print(
                "CORRECTED PA median mm =",
                c[
                    "corrected_pa_mean_mm"
                ].median(),
            )

            print(
                "DELTA PA median mm =",
                c[
                    "corrected_delta_pa_mm"
                ].median(),
            )

            print(
                "CORRECTED LOWER fraction =",
                c[
                    "corrected_lower"
                ].mean(),
            )

            print(
                "BASE-REPEAT noise median mm =",
                c[
                    "base_repeat_absdiff_mm"
                ].median(),
            )

        print("SAVED =", outdir)
        print("=" * 80)

    finally:

        old.omap.pmod.grid_sample_transform = (
            old.omap.ORIG_GRID_SAMPLE
        )

        old.omap.pmod.gather = (
            old.omap.ORIG_GATHER
        )


if __name__ == "__main__":
    main()
