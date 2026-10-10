#!/usr/bin/env python3

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from arguments import parse_args
from evaluation.tester import Tester

import dataset_components.dataloader as dlmod
import dataset_components.pipeline as pipeline_mod

from dataset_components.robot import gather_features
from dataset_components.constants import (
    RELEASE_CONTEXT_HORIZON,
)

from diagnostics import eval_equal_physical_delta_global_heldout as src


# ============================================================
# IMPORTANT
#
# Exact fast replacement for the current experiment:
#   PW_CONDITION=original
#
# Experimental definitions are NOT changed:
#   - same held-out clips
#   - same Global Fixed Acc/MSJ boundaries
#   - same 5% requested physical delta
#   - same GRID_N
#   - same MATCH_TOL
#   - same perturbation family
#   - same model
#   - same validity masks
#
# Only duplicated WDS / official preprocessing is removed.
# ============================================================

if src.CONDITION != "original":
    raise RuntimeError(
        "This one-pass implementation is currently validated "
        "for PW_CONDITION=original only."
    )


# ============================================================
# Per-clip perturbation phases.
# Search is done exactly once from the same raw gripper poses
# used by the existing 3-loader implementation.
# ============================================================

PHASE_BY_KEY = {
    "acc": {},
    "jerk": {},
}


def save_meta(kind, key, r):
    src.EQUAL_DELTA_META[kind][key] = {
        "actual_delta":
            float(r["actual_delta"]),

        "relative_match_error":
            float(r["relative_match_error"]),

        "matched":
            bool(r["matched"]),

        "collateral_delta":
            float(r["collateral_delta"]),

        "amp_frames":
            float(r["amp_frames"]),
    }

    PHASE_BY_KEY[kind][key] = (
        np.asarray(r["phase"])
        .copy()
    )


# ============================================================
# ONE build_flow_sample call per raw WDS sample.
#
# Existing version:
#   WDS -> build(base)
#   WDS -> build(acc)
#   WDS -> build(jerk)
#
# Here:
#   WDS -> OFFICIAL_BUILD once
#       -> search both perturbations
#       -> return base sample only
#
# The final Acc/Jerk features are reconstructed after the
# shared deterministic test preprocessing.
# ============================================================

def make_onepass_builder():

    def builder(
        sample,
        domain,
        robot_sampler,
        max_robot_points,
        deterministic=False,
        seed=None,
        force_single_arm=False,
    ):

        raw_key = str(
            sample.get("__key__", "")
        )

        metric_sample = None

        if raw_key in src.TARGET_KEYS:

            if (
                "right_gripper_pose" not in sample
                or
                "left_gripper_pose" not in sample
            ):
                raise RuntimeError(
                    f"{raw_key}: raw gripper pose missing"
                )

            # Exact same Action metric source as old code.
            metric_sample = {
                "right_gripper_pose":
                    np.asarray(
                        sample[
                            "right_gripper_pose"
                        ]
                    ).copy(),

                "left_gripper_pose":
                    np.asarray(
                        sample[
                            "left_gripper_pose"
                        ]
                    ).copy(),
            }

        # Official robot construction ONCE.
        out = src.old.OFFICIAL_BUILD(
            sample,
            domain=domain,
            robot_sampler=robot_sampler,
            max_robot_points=max_robot_points,
            deterministic=deterministic,
            seed=seed,
            force_single_arm=force_single_arm,
        )

        key = str(
            out.get("__key__", "")
        )

        if key not in src.TARGET_KEYS:
            return out

        if metric_sample is None:
            raise RuntimeError(
                f"{key}: raw metric sample missing"
            )

        if key != raw_key:
            raise RuntimeError(
                f"{key}: raw/built key mismatch "
                f"(raw={raw_key})"
            )

        T = out[
            "robot_flows"
        ].shape[0]

        bp = src.base_phase(
            key,
            T,
        )

        # Current experiment must be the original Action.
        # Fail rather than silently changing semantics.
        identity = np.linspace(
            0.0,
            1.0,
            T,
        )

        if not np.allclose(
            bp,
            identity,
            atol=1e-12,
            rtol=0,
        ):
            raise RuntimeError(
                f"{key}: CONDITION=original did not "
                "produce identity base phase"
            )

        # ----------------------------------------------------
        # Equal ΔAcceleration
        # ----------------------------------------------------

        ra = src.search_equal_delta(
            metric_sample,
            bp,
            target_metric=
                "rms_acceleration",
            collateral_metric=
                "mean_squared_jerk",
            requested_delta=
                src.TARGET_DELTA_ACC,
        )

        save_meta(
            "acc",
            key,
            ra,
        )

        # ----------------------------------------------------
        # Equal ΔMSJ
        # ----------------------------------------------------

        rj = src.search_equal_delta(
            metric_sample,
            bp,
            target_metric=
                "mean_squared_jerk",
            collateral_metric=
                "rms_acceleration",
            requested_delta=
                src.TARGET_DELTA_MSJ,
        )

        save_meta(
            "jerk",
            key,
            rj,
        )

        # base == original for this run.
        return src.apply_phase(
            out,
            bp,
        )

    return builder


# ============================================================
# The normal release gather() drops scene_colors,
# scene_normals and robot_normals after feature construction.
#
# Keep these three AUXILIARY arrays so Acc/Jerk
# scene_features / robot_features can be regenerated from
# the SAME already-preprocessed scene without reading WDS
# again.
#
# They are removed before model forward.
# ============================================================

ORIGINAL_GATHER = pipeline_mod.gather

AUX_FIELDS = (
    "scene_colors",
    "scene_normals",
    "robot_normals",
)


def gather_keep_aux(
    sample,
    fields,
    domain,
):

    result = ORIGINAL_GATHER(
        sample,
        fields,
        domain,
    )

    for name in AUX_FIELDS:
        if name in sample:
            result[name] = sample[name]

    return result


# ============================================================
# Batch/sample conversion
# ============================================================

def batch_to_numpy_sample(
    batch,
    key,
):

    sample = {
        "__key__": key,
    }

    for name, value in batch.items():

        if isinstance(
            value,
            torch.Tensor,
        ):

            x = (
                value.detach()
                .cpu()
            )

            if (
                x.ndim >= 1
                and
                x.shape[0] == 1
            ):
                x = x[0]

            sample[name] = (
                x.numpy()
                .copy()
            )

        elif (
            isinstance(
                value,
                (list, tuple),
            )
            and
            len(value) == 1
        ):
            sample[name] = value[0]

        else:
            sample[name] = copy.deepcopy(
                value
            )

    sample["__key__"] = key

    return sample


def tensor_with_batch_dim(
    array,
    reference,
):

    t = torch.as_tensor(
        np.asarray(array)
    )

    if isinstance(
        reference,
        torch.Tensor,
    ):
        t = t.to(
            dtype=reference.dtype
        )

    return t.unsqueeze(0)


# ============================================================
# Build Acc or Jerk variant FROM SAME processed sample.
#
# Important:
# gather_features() is rerun because PointWorld uses:
#   robot flows / velocity / acceleration
# and scene dist-to-robot / gripper state features.
# ============================================================

VARIANT_REPLACE_FIELDS = (
    "robot_flows",
    "robot_features",
    "scene_features",
    "right_gripper_open",
    "left_gripper_open",
    "gripper_positions",
    "robot_normals",
)


def make_variant_batch(
    b0,
    key,
    phase,
    args,
):

    sample = batch_to_numpy_sample(
        b0,
        key,
    )

    sample = src.apply_phase(
        sample,
        phase,
    )

    # Recompute EXACT official Action-dependent features.
    sample = gather_features(
        sample,
        robot_features=args.robot_features,
        scene_features=args.scene_features,
        random_context_mode=
            "fixed",
        context_horizon=
            RELEASE_CONTEXT_HORIZON,
        has_bimanual_robot=True,
        domain="behavior",
    )

    out = {}

    for k, v in b0.items():

        # AUX fields are diagnostic construction fields only.
        if k in AUX_FIELDS:
            continue

        if (
            k in VARIANT_REPLACE_FIELDS
            and
            k in sample
            and
            isinstance(v, torch.Tensor)
        ):
            out[k] = tensor_with_batch_dim(
                sample[k],
                v,
            )

        else:
            out[k] = (
                v.clone()
                if isinstance(
                    v,
                    torch.Tensor,
                )
                else copy.deepcopy(v)
            )

    return out


def strip_aux(
    batch,
):
    return {
        k: v
        for k, v in batch.items()
        if k not in AUX_FIELDS
    }


# ============================================================
# Main — ONE dataloader only
# ============================================================

def main():

    args = parse_args()

    tester = Tester(args)

    tester.model.eval()

    # Preserve auxiliary feature sources through ONE release
    # pipeline pass.
    pipeline_mod.gather = (
        gather_keep_aux
    )

    # ONE builder / ONE dataloader.
    dlmod.build_flow_sample = (
        make_onepass_builder()
    )

    src.old.reset()

    loader, _ = dlmod.build_dataloader(
        tester.args,
        mode="test",
        rank=tester.rank,
        world_size=tester.world_size,
        override_splits="test",
    )

    # Restore module globals after dataset construction.
    dlmod.build_flow_sample = (
        src.old.OFFICIAL_BUILD
    )

    rows = []
    found = set()

    with torch.no_grad():

        for b0 in loader:

            key = str(
                b0["__key__"][0]
            )

            if key not in src.TARGET_KEYS:
                continue

            if key in found:
                continue

            acc_meta = (
                src.EQUAL_DELTA_META[
                    "acc"
                ].get(key)
            )

            jerk_meta = (
                src.EQUAL_DELTA_META[
                    "jerk"
                ].get(key)
            )

            if acc_meta is None:
                raise RuntimeError(
                    f"{key}: missing equal-ΔAcc metadata"
                )

            if jerk_meta is None:
                raise RuntimeError(
                    f"{key}: missing equal-ΔMSJ metadata"
                )

            if key not in PHASE_BY_KEY["acc"]:
                raise RuntimeError(
                    f"{key}: missing Acc phase"
                )

            if key not in PHASE_BY_KEY["jerk"]:
                raise RuntimeError(
                    f"{key}: missing Jerk phase"
                )

            # ------------------------------------------------
            # Same processed scene; only Action variant differs.
            # ------------------------------------------------

            ba = make_variant_batch(
                b0,
                key,
                PHASE_BY_KEY[
                    "acc"
                ][key],
                tester.args,
            )

            bj = make_variant_batch(
                b0,
                key,
                PHASE_BY_KEY[
                    "jerk"
                ][key],
                tester.args,
            )

            b0_model = strip_aux(
                b0
            )

            # Hard invariant: GT is shared exactly.
            for other in (
                ba,
                bj,
            ):
                if not torch.allclose(
                    b0_model[
                        "gt_scene_flows"
                    ],
                    other[
                        "gt_scene_flows"
                    ],
                    atol=1e-7,
                    rtol=0,
                ):
                    raise RuntimeError(
                        f"{key}: GT changed"
                    )

            # ------------------------------------------------
            # Metadata — exact same definitions as 3-pass.
            # ------------------------------------------------

            acc_actual = (
                acc_meta[
                    "actual_delta"
                ]
            )

            acc_relerr = (
                acc_meta[
                    "relative_match_error"
                ]
            )

            acc_matched = (
                acc_meta[
                    "matched"
                ]
            )

            acc_collateral = (
                acc_meta[
                    "collateral_delta"
                ]
            )

            acc_amp = (
                acc_meta[
                    "amp_frames"
                ]
            )

            jerk_actual = (
                jerk_meta[
                    "actual_delta"
                ]
            )

            jerk_relerr = (
                jerk_meta[
                    "relative_match_error"
                ]
            )

            jerk_matched = (
                jerk_meta[
                    "matched"
                ]
            )

            jerk_collateral = (
                jerk_meta[
                    "collateral_delta"
                ]
            )

            jerk_amp = (
                jerk_meta[
                    "amp_frames"
                ]
            )

            g0 = src.to_device(
                b0_model,
                tester.device,
            )

            ga = src.to_device(
                ba,
                tester.device,
            )

            gj = src.to_device(
                bj,
                tester.device,
            )

            # Keep model-forward semantics unchanged.
            src.old.reset()

            p0 = tester.model(
                g0,
                training=False,
            )["scene_flows"]

            src.old.reset()

            pa = tester.model(
                ga,
                training=False,
            )["scene_flows"]

            src.old.reset()

            pj = tester.model(
                gj,
                training=False,
            )["scene_flows"]

            sa = torch.linalg.vector_norm(
                pa - p0,
                dim=-1,
            )

            sj = torch.linalg.vector_norm(
                pj - p0,
                dim=-1,
            )

            exists = src.mask3(
                g0,
                "scene_exists",
            )

            context = src.mask3(
                g0,
                "scene_context_mask",
            )

            supervised = src.mask3(
                g0,
                "scene_supervised_mask",
            )

            moved = src.mask3(
                g0,
                "scene_moved_mask",
            )

            valid = (
                (~context)
                & exists
                & supervised
            )

            vm = (
                valid
                & moved
            )

            action_acc = float(
                src.METRIC_BY_KEY[
                    key
                ][
                    "rms_acceleration"
                ]
            )

            action_msj = float(
                src.METRIC_BY_KEY[
                    key
                ][
                    "mean_squared_jerk"
                ]
            )

            row = {
                "clip":
                    key,

                "task":
                    src.TASK,

                "condition":
                    src.CONDITION,

                "original_action_acc":
                    action_acc,

                "original_action_msj":
                    action_msj,

                "acc_outside":
                    bool(
                        action_acc
                        > src.TA
                    ),

                "msj_outside":
                    bool(
                        action_msj
                        > src.TJ
                    ),

                "acc_boundary_excess":
                    float(
                        action_acc
                        / src.TA
                        - 1.0
                    ),

                "msj_boundary_excess":
                    float(
                        action_msj
                        / src.TJ
                        - 1.0
                    ),

                "requested_delta_acc":
                    src.TARGET_DELTA_ACC,

                "actual_delta_acc":
                    acc_actual,

                "acc_match_relerr":
                    acc_relerr,

                "acc_matched":
                    acc_matched,

                "acc_collateral_delta_msj":
                    acc_collateral,

                "acc_perturb_frames":
                    acc_amp,

                "requested_delta_msj":
                    src.TARGET_DELTA_MSJ,

                "actual_delta_msj":
                    jerk_actual,

                "jerk_match_relerr":
                    jerk_relerr,

                "jerk_matched":
                    jerk_matched,

                "jerk_collateral_delta_acc":
                    jerk_collateral,

                "jerk_perturb_frames":
                    jerk_amp,

                "acc_sensitivity_all_mm":
                    1000.0
                    * src.masked_mean(
                        sa,
                        valid,
                    ),

                "acc_sensitivity_moved_mm":
                    1000.0
                    * src.masked_mean(
                        sa,
                        vm,
                    ),

                "acc_sensitivity_t10_moved_mm":
                    1000.0
                    * src.masked_mean(
                        sa[:, -1],
                        vm[:, -1],
                    ),

                "jerk_sensitivity_all_mm":
                    1000.0
                    * src.masked_mean(
                        sj,
                        valid,
                    ),

                "jerk_sensitivity_moved_mm":
                    1000.0
                    * src.masked_mean(
                        sj,
                        vm,
                    ),

                "jerk_sensitivity_t10_moved_mm":
                    1000.0
                    * src.masked_mean(
                        sj[:, -1],
                        vm[:, -1],
                    ),
            }

            rows.append(row)

            found.add(key)

            print(
                f"[{len(found)}/"
                f"{len(src.TARGET_KEYS)}] "
                f"{key} | "
                f"cond={src.CONDITION} | "
                f"Acc match="
                f"{acc_relerr:.3f} "
                f"sens="
                f"{row['acc_sensitivity_moved_mm']:.3f} | "
                f"Jerk match="
                f"{jerk_relerr:.3f} "
                f"sens="
                f"{row['jerk_sensitivity_moved_mm']:.3f}"
            )

            # No reason to continue scanning after every
            # pre-specified target has been evaluated.
            if (
                len(found)
                == len(
                    src.TARGET_KEYS
                )
            ):
                break

    outdir = (
        src.ROOT
        / "diagnostics/results/"
          "equal_physical_delta_global_heldout"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = (
        outdir
        / (
            f"{src.TAG}"
            "__equal_delta.csv"
        )
    )

    pd.DataFrame(
        rows
    ).to_csv(
        out,
        index=False,
    )

    print()
    print("=" * 90)
    print("ONE-PASS EQUAL PHYSICAL DELTA RESULT")
    print("=" * 90)
    print("TASK =", src.TASK)
    print("CONDITION =", src.CONDITION)
    print("TARGETS =", len(src.TARGET_KEYS))
    print("FOUND =", len(found))
    print(
        "requested ΔAcc =",
        src.TARGET_DELTA_ACC,
    )
    print(
        "requested ΔMSJ =",
        src.TARGET_DELTA_MSJ,
    )
    print("OUTPUT =", out)
    print("SUCCESS = True")


if __name__ == "__main__":
    main()
