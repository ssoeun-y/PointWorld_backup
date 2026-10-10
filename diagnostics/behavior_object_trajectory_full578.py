#!/usr/bin/env python3

import csv
import json
import runpy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from evaluation.tester import Tester

import dataset_components.dataloader as dlmod
import dataset_components.decoders as decmod
import dataset_components.pipeline as pmod

import diagnostics.behavior_object_mapping_smoke as objmap




def mask3(batch, name):
    x = batch[name].bool()
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    assert x.ndim == 3, (name, x.shape)
    return x


def mean_or_nan(x):
    if x.numel() == 0:
        return float("nan")
    return float(x.float().mean().detach().cpu())


def get_key(batch):
    key = batch["__key__"]
    if isinstance(key, (list, tuple)):
        key = key[0]
    if isinstance(key, bytes):
        key = key.decode("utf-8")
    return str(key)


@torch.no_grad()
def multiclip_object_smoke(
    self,
    dl,
    split,
    apply_confidence_mask=False,
):
    self.model.eval()

    rows = []
    sequence_rows = []
    debris_point_rows = []

    max_batches = 578
    processed = 0
    all_official_l2_exact = True

    trajectory_out_dir = (
        Path("diagnostics/results")
        / (
            "behavior_object_trajectory_npz_full578_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    )
    trajectory_out_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    for batch_index, batch in enumerate(dl):
        if processed >= max_batches:
            break

        key = get_key(batch)


        if "scene_object_id" not in batch:
            raise RuntimeError(
                f"[{key}] scene_object_id missing"
            )

        object_ids = batch["scene_object_id"]

        if isinstance(object_ids, torch.Tensor):
            object_ids_np = (
                object_ids.detach().cpu().numpy()
            )
        else:
            object_ids_np = np.asarray(object_ids)

        assert object_ids_np.ndim == 3, (
            key,
            object_ids_np.shape,
        )
        assert object_ids_np.shape[0] == 1

        table = objmap.OBJECT_TABLE.get(key)

        if table is None:
            raise RuntimeError(
                f"[{key}] OBJECT_TABLE missing"
            )

        # Object ID must stay identical through all 11 timesteps.
        temporal_id_exact = np.array_equal(
            object_ids_np[0],
            np.broadcast_to(
                object_ids_np[0, 0:1],
                object_ids_np[0].shape,
            ),
        )

        if not temporal_id_exact:
            raise RuntimeError(
                f"[{key}] object ID changed over time"
            )

        # Don't send diagnostic object ID to model.
        batch_model = {
            k: (
                v.to(
                    self.device,
                    non_blocking=True,
                )
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
            if k != "scene_object_id"
        }

        outputs = self.model(
            batch_model,
            training=False,
        )

        _, official_dict = self.model.loss_fn(
            outputs,
            batch_model,
            training=False,
        )

        pred = outputs["scene_flows"]
        gt = batch_model["gt_scene_flows"]

        assert pred.shape == gt.shape
        assert pred.shape[0] == 1
        assert pred.shape[-1] == 3

        B, T, N, _ = pred.shape

        if object_ids_np.shape[2] != N:
            raise RuntimeError(
                f"[{key}] label N="
                f"{object_ids_np.shape[2]} "
                f"!= prediction N={N}"
            )

        context = mask3(
            batch_model,
            "scene_context_mask",
        )
        exists = mask3(
            batch_model,
            "scene_exists",
        )
        supervised = mask3(
            batch_model,
            "scene_supervised_mask",
        )
        moved = mask3(
            batch_model,
            "scene_moved_mask",
        )

        # Official evaluation-valid mask.
        valid = (
            (~context)
            & exists
            & supervised
        )

        if not bool(valid.any()):
            raise RuntimeError(
                f"[{key}] no official-valid points"
            )

        # Independent official-L2 reproduction.
        l2_all = torch.linalg.vector_norm(
            pred - gt,
            dim=-1,
        )

        official_l2 = float(
            official_dict["l2/mean"]
        )
        recomputed_l2 = float(
            l2_all[valid]
            .mean()
            .detach()
            .cpu()
        )

        official_l2_exact = (
            abs(
                official_l2
                - recomputed_l2
            )
            <= 1e-6
        )

        if not official_l2_exact:
            raise RuntimeError(
                f"[{key}] official L2 mismatch: "
                f"official={official_l2:.12f}, "
                f"recomputed={recomputed_l2:.12f}"
            )

        # T,N must exactly agree with diagnostic labels.
        assert object_ids_np.shape == (
            1,
            T,
            N,
        ), (
            key,
            object_ids_np.shape,
            pred.shape,
        )

        safe_key = (
            key
            .replace("/", "_")
            .replace(":", "_")
            .replace("+", "_")
        )

        np.savez_compressed(
            trajectory_out_dir
            / f"{safe_key}.npz",

            sample_key=np.asarray(key),

            # Stable global sampled-point index.
            point_index=np.arange(
                N,
                dtype=np.int64,
            ),

            # Full t0~t10 trajectories.
            gt_scene_flows=(
                gt[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            ),
            pred_scene_flows=(
                pred[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            ),

            # Object identity for every timestep.
            scene_object_id=(
                object_ids_np[0]
                .astype(np.int32)
            ),

            # Official masks.
            official_valid_mask=(
                valid[0]
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            ),
            scene_context_mask=(
                context[0]
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            ),
            scene_exists=(
                exists[0]
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            ),
            scene_supervised_mask=(
                supervised[0]
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            ),
            scene_moved_mask=(
                moved[0]
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            ),

            official_l2=np.asarray(
                official_l2,
                dtype=np.float64,
            ),
            recomputed_l2=np.asarray(
                recomputed_l2,
                dtype=np.float64,
            ),
            official_l2_exact=np.asarray(
                official_l2_exact,
            ),
            temporal_object_id_exact=np.asarray(
                temporal_id_exact,
            ),
        )

        valid = (
            (~context)
            & exists
            & supervised
        )

        l2 = torch.linalg.vector_norm(
            pred - gt,
            dim=-1,
        )

        official_l2 = float(
            official_dict["l2/mean"]
        )

        recomputed_l2 = mean_or_nan(
            l2[valid]
        )

        l2_difference = abs(
            official_l2 - recomputed_l2
        )

        official_exact = (
            l2_difference <= 1e-6
        )

        if not official_exact:
            all_official_l2_exact = False
            raise RuntimeError(
                f"[{key}] official L2 mismatch: "
                f"{official_l2} vs "
                f"{recomputed_l2}"
            )

        # Derivatives.
        pred_v = (
            pred[:, 1:]
            - pred[:, :-1]
        )
        gt_v = (
            gt[:, 1:]
            - gt[:, :-1]
        )

        velocity_error = torch.linalg.vector_norm(
            pred_v - gt_v,
            dim=-1,
        )

        velocity_valid = (
            valid[:, 1:]
            & valid[:, :-1]
        )

        pred_a = (
            pred_v[:, 1:]
            - pred_v[:, :-1]
        )
        gt_a = (
            gt_v[:, 1:]
            - gt_v[:, :-1]
        )

        acceleration_error = (
            torch.linalg.vector_norm(
                pred_a - gt_a,
                dim=-1,
            )
        )

        acceleration_valid = (
            valid[:, 2:]
            & valid[:, 1:-1]
            & valid[:, :-2]
        )

        gt_motion = torch.linalg.vector_norm(
            gt_v,
            dim=-1,
        )

        pred_motion = torch.linalg.vector_norm(
            pred_v,
            dim=-1,
        )

        # Exact same direction diagnostic used in
        # behavior_object_direction_timing_full578.py.
        direction_speed_threshold = 1e-3

        direction_valid_base = (
            velocity_valid
            & (gt_motion >= direction_speed_threshold)
            & (pred_motion >= direction_speed_threshold)
        )

        direction_dot = (
            gt_v * pred_v
        ).sum(dim=-1)

        direction_denom = (
            gt_motion * pred_motion
        )

        cosine_similarity = torch.clamp(
            direction_dot
            / torch.clamp(
                direction_denom,
                min=1e-12,
            ),
            min=-1.0,
            max=1.0,
        )

        angular_error_deg = torch.rad2deg(
            torch.acos(
                cosine_similarity
            )
        )

        # Final timestep.
        final_t = T - 1

        # Released PointWorld configuration uses
        # gripper-only robot_flows.
        if "robot_flows" not in batch_model:
            raise RuntimeError(
                f"[{key}] robot_flows missing"
            )

        robot_flows = batch_model[
            "robot_flows"
        ]

        if (
            robot_flows.ndim != 4
            or robot_flows.shape[0] != B
            or robot_flows.shape[1] != T
            or robot_flows.shape[-1] != 3
        ):
            raise RuntimeError(
                f"[{key}] robot_flows shape = "
                f"{tuple(robot_flows.shape)}"
            )

        gripper_final = robot_flows[
            0,
            final_t,
        ]

        if gripper_final.shape[0] == 0:
            raise RuntimeError(
                f"[{key}] no gripper points"
            )

        # Distance from every GT scene point at t=10
        # to its nearest gripper-surface point.
        final_scene_to_gripper = (
            torch.cdist(
                gt[
                    0,
                    final_t,
                ].unsqueeze(0),
                gripper_final.unsqueeze(0),
                p=2,
            )
            .squeeze(0)
            .min(dim=1)
            .values
        )

        if final_scene_to_gripper.shape != (N,):
            raise RuntimeError(
                f"[{key}] scene-gripper distance "
                f"shape = "
                f"{tuple(final_scene_to_gripper.shape)}"
            )

        if not torch.isfinite(
            final_scene_to_gripper
        ).all():
            raise RuntimeError(
                f"[{key}] non-finite gripper distance"
            )

        ids_t0 = object_ids_np[0, 0]
        unique_ids = np.unique(ids_t0)

        object_summary = []

        for object_id in unique_ids:
            object_id = int(object_id)

            object_name = table.get(
                object_id,
                f"UNKNOWN_ID_{object_id}",
            )

            point_mask_np = (
                ids_t0 == object_id
            )

            point_count = int(
                point_mask_np.sum()
            )

            point_mask = torch.as_tensor(
                point_mask_np,
                dtype=torch.bool,
                device=pred.device,
            )

            object_mask = (
                point_mask[None, None, :]
                .expand(B, T, N)
            )

            object_valid = (
                valid
                & object_mask
            )

            object_moved_valid = (
                valid
                & moved
                & object_mask
            )

            object_velocity_valid = (
                velocity_valid
                & object_mask[:, 1:]
            )

            object_acceleration_valid = (
                acceleration_valid
                & object_mask[:, 2:]
            )

            valid_count = int(
                object_valid.sum().item()
            )

            moved_valid_count = int(
                object_moved_valid.sum().item()
            )

            # ---------------------------------------------
            # t=10 object analysis:
            # highest-L2 5% vs remaining valid points.
            # ---------------------------------------------
            final_object_valid = (
                valid[
                    0,
                    final_t,
                ]
                & object_mask[
                    0,
                    final_t,
                ]
            )

            final_valid_count = int(
                final_object_valid.sum().item()
            )

            final_top5_count = 0

            final_top5_gripper_distance_mean_m = (
                float("nan")
            )
            final_other_gripper_distance_mean_m = (
                float("nan")
            )
            final_top5_minus_other_gripper_distance_mean_m = (
                float("nan")
            )
            final_distance_l2_spearman = (
                float("nan")
            )

            # GT-motion confound diagnostics.
            # final GT motion = displacement t=9 -> t=10.
            final_motion_valid_count = 0

            final_top5_gt_motion_mean_m = (
                float("nan")
            )
            final_other_gt_motion_mean_m = (
                float("nan")
            )
            final_top5_minus_other_gt_motion_mean_m = (
                float("nan")
            )

            final_distance_gt_motion_spearman = (
                float("nan")
            )
            final_gt_motion_l2_spearman = (
                float("nan")
            )

            # Partial Spearman:
            # distance <-> L2 after rank-controlling
            # GT motion magnitude.
            final_distance_l2_partial_spearman_gt_motion = (
                float("nan")
            )

            final_top5_far25_fraction = (
                float("nan")
            )
            final_top5_near25_fraction = (
                float("nan")
            )

            if final_valid_count >= 2:
                final_idx = torch.nonzero(
                    final_object_valid,
                    as_tuple=False,
                ).squeeze(1)

                final_error = l2[
                    0,
                    final_t,
                    final_idx,
                ]

                final_distance = (
                    final_scene_to_gripper[
                        final_idx
                    ]
                )

                k_top5 = max(
                    1,
                    int(
                        np.ceil(
                            0.05
                            * final_valid_count
                        )
                    ),
                )

                # Guarantee a non-empty comparison group.
                k_top5 = min(
                    k_top5,
                    final_valid_count - 1,
                )

                top_local_idx = torch.topk(
                    final_error,
                    k=k_top5,
                    largest=True,
                    sorted=True,
                ).indices

                top_local_mask = torch.zeros(
                    final_valid_count,
                    dtype=torch.bool,
                    device=final_distance.device,
                )

                top_local_mask[
                    top_local_idx
                ] = True

                other_local_mask = (
                    ~top_local_mask
                )

                top_distance = final_distance[
                    top_local_mask
                ]

                other_distance = final_distance[
                    other_local_mask
                ]

                final_top5_count = int(
                    top_local_mask.sum().item()
                )

                final_top5_gripper_distance_mean_m = (
                    float(
                        top_distance
                        .mean()
                        .detach()
                        .cpu()
                    )
                )

                final_other_gripper_distance_mean_m = (
                    float(
                        other_distance
                        .mean()
                        .detach()
                        .cpu()
                    )
                )

                final_top5_minus_other_gripper_distance_mean_m = (
                    final_top5_gripper_distance_mean_m
                    - final_other_gripper_distance_mean_m
                )

                distance_np = (
                    final_distance
                    .detach()
                    .cpu()
                    .numpy()
                )

                error_np = (
                    final_error
                    .detach()
                    .cpu()
                    .numpy()
                )

                if (
                    np.unique(
                        distance_np
                    ).size > 1
                    and np.unique(
                        error_np
                    ).size > 1
                ):
                    final_distance_l2_spearman = (
                        float(
                            spearmanr(
                                distance_np,
                                error_np,
                            )[0]
                        )
                    )

                # -----------------------------------------
                # Motion-confound check.
                #
                # l2[t=10] requires valid t=10.
                # GT motion t=9->10 additionally requires
                # the same point to be valid at t=9.
                # -----------------------------------------
                motion_local_valid = valid[
                    0,
                    final_t - 1,
                    final_idx,
                ]

                motion_idx = final_idx[
                    motion_local_valid
                ]

                final_motion_valid_count = int(
                    motion_idx.numel()
                )

                if final_motion_valid_count >= 3:
                    motion_error = l2[
                        0,
                        final_t,
                        motion_idx,
                    ]

                    motion_distance = (
                        final_scene_to_gripper[
                            motion_idx
                        ]
                    )

                    # gt_motion[:, j] corresponds to
                    # displacement j -> j+1.
                    motion_gt = gt_motion[
                        0,
                        final_t - 1,
                        motion_idx,
                    ]

                    k_motion_top5 = max(
                        1,
                        int(
                            np.ceil(
                                0.05
                                * final_motion_valid_count
                            )
                        ),
                    )

                    k_motion_top5 = min(
                        k_motion_top5,
                        final_motion_valid_count - 1,
                    )

                    motion_top_local = torch.topk(
                        motion_error,
                        k=k_motion_top5,
                        largest=True,
                        sorted=True,
                    ).indices

                    motion_top_mask = torch.zeros(
                        final_motion_valid_count,
                        dtype=torch.bool,
                        device=motion_gt.device,
                    )

                    motion_top_mask[
                        motion_top_local
                    ] = True

                    motion_other_mask = (
                        ~motion_top_mask
                    )

                    top_gt_motion = motion_gt[
                        motion_top_mask
                    ]

                    other_gt_motion = motion_gt[
                        motion_other_mask
                    ]

                    final_top5_gt_motion_mean_m = float(
                        top_gt_motion
                        .mean()
                        .detach()
                        .cpu()
                    )

                    final_other_gt_motion_mean_m = float(
                        other_gt_motion
                        .mean()
                        .detach()
                        .cpu()
                    )

                    final_top5_minus_other_gt_motion_mean_m = (
                        final_top5_gt_motion_mean_m
                        - final_other_gt_motion_mean_m
                    )

                    md_np = (
                        motion_distance
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )

                    me_np = (
                        motion_error
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )

                    mg_np = (
                        motion_gt
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )

                    if (
                        np.unique(md_np).size > 1
                        and np.unique(mg_np).size > 1
                    ):
                        final_distance_gt_motion_spearman = (
                            float(
                                spearmanr(
                                    md_np,
                                    mg_np,
                                )[0]
                            )
                        )

                    if (
                        np.unique(mg_np).size > 1
                        and np.unique(me_np).size > 1
                    ):
                        final_gt_motion_l2_spearman = (
                            float(
                                spearmanr(
                                    mg_np,
                                    me_np,
                                )[0]
                            )
                        )

                    # -------------------------------------
                    # Partial Spearman(distance, L2 |
                    #                  GT motion)
                    #
                    # Rank-transform all variables, regress
                    # distance-rank and error-rank separately
                    # on motion-rank, then correlate residuals.
                    # -------------------------------------
                    if (
                        np.unique(md_np).size > 1
                        and np.unique(me_np).size > 1
                        and np.unique(mg_np).size > 1
                    ):
                        rd = rankdata(
                            md_np,
                            method="average",
                        )

                        re = rankdata(
                            me_np,
                            method="average",
                        )

                        rm = rankdata(
                            mg_np,
                            method="average",
                        )

                        design = np.column_stack([
                            np.ones_like(rm),
                            rm,
                        ])

                        beta_d = np.linalg.lstsq(
                            design,
                            rd,
                            rcond=None,
                        )[0]

                        beta_e = np.linalg.lstsq(
                            design,
                            re,
                            rcond=None,
                        )[0]

                        residual_d = (
                            rd
                            - design @ beta_d
                        )

                        residual_e = (
                            re
                            - design @ beta_e
                        )

                        denom = (
                            np.linalg.norm(
                                residual_d
                            )
                            * np.linalg.norm(
                                residual_e
                            )
                        )

                        if denom > 0:
                            final_distance_l2_partial_spearman_gt_motion = (
                                float(
                                    np.dot(
                                        residual_d,
                                        residual_e,
                                    )
                                    / denom
                                )
                            )

                q25 = torch.quantile(
                    final_distance,
                    0.25,
                )

                q75 = torch.quantile(
                    final_distance,
                    0.75,
                )

                final_top5_near25_fraction = (
                    float(
                        (
                            top_distance
                            <= q25
                        )
                        .float()
                        .mean()
                        .detach()
                        .cpu()
                    )
                )

                final_top5_far25_fraction = (
                    float(
                        (
                            top_distance
                            >= q75
                        )
                        .float()
                        .mean()
                        .detach()
                        .cpu()
                    )
                )

            # =================================================
            # LOCAL POINT-LEVEL NEAR vs FAR DIRECTION ANALYSIS
            #
            # Same object, final transition t=9 -> t=10.
            # Only points satisfying the SAME direction-valid
            # definition used by the previous direction experiment.
            #
            # Near/Far groups are exact non-overlapping quartiles
            # based on final GT-point -> gripper-surface distance.
            # =================================================
            final_direction_valid = (
                direction_valid_base[
                    0,
                    final_t - 1,
                ]
                & point_mask
            )

            final_direction_valid_count = int(
                final_direction_valid.sum().item()
            )

            final_near25_count = 0
            final_far25_count = 0

            final_near25_l2_mean_m = float("nan")
            final_far25_l2_mean_m = float("nan")
            final_far25_minus_near25_l2_mean_m = float("nan")

            final_near25_direction_angle_mean_deg = float("nan")
            final_far25_direction_angle_mean_deg = float("nan")
            final_far25_minus_near25_direction_angle_mean_deg = (
                float("nan")
            )

            final_near25_direction_cosine_mean = float("nan")
            final_far25_direction_cosine_mean = float("nan")

            final_near25_opposite_direction_rate = float("nan")
            final_far25_opposite_direction_rate = float("nan")

            final_near25_gt_motion_mean_m = float("nan")
            final_far25_gt_motion_mean_m = float("nan")
            final_far25_minus_near25_gt_motion_mean_m = float("nan")

            final_distance_direction_angle_spearman = float("nan")

            final_distance_direction_angle_partial_spearman_gt_motion = (
                float("nan")
            )

            # Need at least 4 points to form disjoint
            # nearest/farthest quartiles.
            if final_direction_valid_count >= 4:
                dir_idx = torch.nonzero(
                    final_direction_valid,
                    as_tuple=False,
                ).squeeze(1)

                dir_distance = (
                    final_scene_to_gripper[
                        dir_idx
                    ]
                )

                dir_l2 = l2[
                    0,
                    final_t,
                    dir_idx,
                ]

                dir_angle = angular_error_deg[
                    0,
                    final_t - 1,
                    dir_idx,
                ]

                dir_cosine = cosine_similarity[
                    0,
                    final_t - 1,
                    dir_idx,
                ]

                dir_gt_motion = gt_motion[
                    0,
                    final_t - 1,
                    dir_idx,
                ]

                quartile_count = max(
                    1,
                    int(
                        np.floor(
                            0.25
                            * final_direction_valid_count
                        )
                    ),
                )

                distance_order = torch.argsort(
                    dir_distance,
                    descending=False,
                )

                near_local_idx = distance_order[
                    :quartile_count
                ]

                far_local_idx = distance_order[
                    -quartile_count:
                ]

                near_distance = dir_distance[
                    near_local_idx
                ]
                far_distance = dir_distance[
                    far_local_idx
                ]

                # Explicit separation audit:
                # max near distance must not exceed min far distance.
                if (
                    near_distance.max()
                    > far_distance.min()
                ):
                    raise RuntimeError(
                        f"[{key}] near/far distance ordering failed"
                    )

                near_l2 = dir_l2[
                    near_local_idx
                ]
                far_l2 = dir_l2[
                    far_local_idx
                ]

                near_angle = dir_angle[
                    near_local_idx
                ]
                far_angle = dir_angle[
                    far_local_idx
                ]

                near_cosine = dir_cosine[
                    near_local_idx
                ]
                far_cosine = dir_cosine[
                    far_local_idx
                ]

                near_gt_motion = dir_gt_motion[
                    near_local_idx
                ]
                far_gt_motion = dir_gt_motion[
                    far_local_idx
                ]

                final_near25_count = int(
                    near_local_idx.numel()
                )
                final_far25_count = int(
                    far_local_idx.numel()
                )

                final_near25_l2_mean_m = float(
                    near_l2.mean().detach().cpu()
                )
                final_far25_l2_mean_m = float(
                    far_l2.mean().detach().cpu()
                )
                final_far25_minus_near25_l2_mean_m = (
                    final_far25_l2_mean_m
                    - final_near25_l2_mean_m
                )

                final_near25_direction_angle_mean_deg = float(
                    near_angle.mean().detach().cpu()
                )
                final_far25_direction_angle_mean_deg = float(
                    far_angle.mean().detach().cpu()
                )
                final_far25_minus_near25_direction_angle_mean_deg = (
                    final_far25_direction_angle_mean_deg
                    - final_near25_direction_angle_mean_deg
                )

                final_near25_direction_cosine_mean = float(
                    near_cosine.mean().detach().cpu()
                )
                final_far25_direction_cosine_mean = float(
                    far_cosine.mean().detach().cpu()
                )

                final_near25_opposite_direction_rate = float(
                    (
                        near_cosine < 0
                    )
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                )

                final_far25_opposite_direction_rate = float(
                    (
                        far_cosine < 0
                    )
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                )

                final_near25_gt_motion_mean_m = float(
                    near_gt_motion.mean().detach().cpu()
                )
                final_far25_gt_motion_mean_m = float(
                    far_gt_motion.mean().detach().cpu()
                )
                final_far25_minus_near25_gt_motion_mean_m = (
                    final_far25_gt_motion_mean_m
                    - final_near25_gt_motion_mean_m
                )

                dist_np = (
                    dir_distance
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

                angle_np = (
                    dir_angle
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

                motion_np = (
                    dir_gt_motion
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

                if (
                    np.unique(dist_np).size > 1
                    and np.unique(angle_np).size > 1
                ):
                    final_distance_direction_angle_spearman = float(
                        spearmanr(
                            dist_np,
                            angle_np,
                        )[0]
                    )

                # Partial Spearman(distance, direction-angle | GT motion)
                if (
                    np.unique(dist_np).size > 1
                    and np.unique(angle_np).size > 1
                    and np.unique(motion_np).size > 1
                ):
                    rd = rankdata(
                        dist_np,
                        method="average",
                    )
                    ra = rankdata(
                        angle_np,
                        method="average",
                    )
                    rm = rankdata(
                        motion_np,
                        method="average",
                    )

                    design = np.column_stack([
                        np.ones_like(rm),
                        rm,
                    ])

                    beta_d = np.linalg.lstsq(
                        design,
                        rd,
                        rcond=None,
                    )[0]

                    beta_a = np.linalg.lstsq(
                        design,
                        ra,
                        rcond=None,
                    )[0]

                    residual_d = (
                        rd - design @ beta_d
                    )

                    residual_a = (
                        ra - design @ beta_a
                    )

                    denom = (
                        np.linalg.norm(
                            residual_d
                        )
                        * np.linalg.norm(
                            residual_a
                        )
                    )

                    if denom > 0:
                        final_distance_direction_angle_partial_spearman_gt_motion = (
                            float(
                                np.dot(
                                    residual_d,
                                    residual_a,
                                )
                                / denom
                            )
                        )

            # =================================================
            # INDEPENDENT DEBRIS-LIKE GEOMETRY ANALYSIS
            #
            # IMPORTANT:
            # Debris is NOT defined using L2.
            #
            # For each moved object:
            #   1) use final-official-valid object points,
            #   2) construct k-NN from GT coordinates at t=0,
            #   3) keep those neighbor identities fixed,
            #   4) compare local edge lengths in Pred(t=10)
            #      against GT(t=10).
            #
            # If one predicted point drifts away from the local
            # object surface, its fixed-neighbor edge lengths
            # become inconsistent with GT.
            # =================================================
            debris_valid_count = 0
            debris_knn_k = 0
            debris_top5_count = 0

            debris_score_mean_m = float("nan")
            debris_outward_stretch_mean_m = float("nan")

            debris_score_l2_spearman = float("nan")
            debris_score_gripper_distance_spearman = float("nan")
            debris_score_direction_angle_spearman = float("nan")

            debris_top5_l2_overlap_fraction = float("nan")
            debris_top5_l2_jaccard = float("nan")

            debris_top5_l2_mean_m = float("nan")
            debris_other_l2_mean_m = float("nan")
            debris_top5_minus_other_l2_mean_m = float("nan")

            debris_top5_gripper_distance_mean_m = float("nan")
            debris_other_gripper_distance_mean_m = float("nan")
            debris_top5_minus_other_gripper_distance_mean_m = (
                float("nan")
            )

            debris_top5_direction_angle_mean_deg = float("nan")
            debris_other_direction_angle_mean_deg = float("nan")
            debris_top5_minus_other_direction_angle_mean_deg = (
                float("nan")
            )

            if (
                moved_valid_count > 0
                and final_valid_count >= 4
            ):
                debris_idx = torch.nonzero(
                    final_object_valid,
                    as_tuple=False,
                ).squeeze(1)

                n_debris = int(
                    debris_idx.numel()
                )

                debris_valid_count = n_debris

                # k=8 where possible; for small objects use all
                # remaining points as potential neighbors.
                debris_knn_k = min(
                    8,
                    n_debris - 1,
                )

                gt_t0_obj = gt[
                    0,
                    0,
                    debris_idx,
                ]

                gt_final_obj = gt[
                    0,
                    final_t,
                    debris_idx,
                ]

                pred_final_obj = pred[
                    0,
                    final_t,
                    debris_idx,
                ]

                # Fixed neighborhood defined ONLY from GT t=0.
                d0 = torch.cdist(
                    gt_t0_obj.unsqueeze(0),
                    gt_t0_obj.unsqueeze(0),
                    p=2,
                ).squeeze(0)

                d0.fill_diagonal_(
                    float("inf")
                )

                knn_local = torch.topk(
                    d0,
                    k=debris_knn_k,
                    dim=1,
                    largest=False,
                ).indices

                center_local = torch.arange(
                    n_debris,
                    device=pred.device,
                )[:, None]

                gt_neighbor_final = (
                    gt_final_obj[
                        knn_local
                    ]
                )

                pred_neighbor_final = (
                    pred_final_obj[
                        knn_local
                    ]
                )

                gt_edge_length = torch.linalg.vector_norm(
                    gt_final_obj[
                        center_local[:, 0]
                    ][:, None, :]
                    - gt_neighbor_final,
                    dim=-1,
                )

                pred_edge_length = torch.linalg.vector_norm(
                    pred_final_obj[
                        center_local[:, 0]
                    ][:, None, :]
                    - pred_neighbor_final,
                    dim=-1,
                )

                edge_delta = (
                    pred_edge_length
                    - gt_edge_length
                )

                # Independent local shape-break score.
                debris_score = (
                    torch.abs(
                        edge_delta
                    ).mean(dim=1)
                )

                # Positive edge stretch is especially relevant
                # to visually detached / debris-like points.
                debris_outward_stretch = (
                    torch.clamp(
                        edge_delta,
                        min=0.0,
                    ).mean(dim=1)
                )

                if (
                    not torch.isfinite(
                        debris_score
                    ).all()
                ):
                    raise RuntimeError(
                        f"[{key}] non-finite debris score"
                    )

                debris_l2 = l2[
                    0,
                    final_t,
                    debris_idx,
                ]

                debris_distance = (
                    final_scene_to_gripper[
                        debris_idx
                    ]
                )

                debris_gt_motion = gt_motion[
                    0,
                    final_t - 1,
                    debris_idx,
                ]

                debris_direction_valid = (
                    direction_valid_base[
                        0,
                        final_t - 1,
                        debris_idx,
                    ]
                )

                debris_direction_angle = (
                    angular_error_deg[
                        0,
                        final_t - 1,
                        debris_idx,
                    ]
                )

                debris_direction_cosine = (
                    cosine_similarity[
                        0,
                        final_t - 1,
                        debris_idx,
                    ]
                )

                debris_score_mean_m = float(
                    debris_score.mean()
                    .detach()
                    .cpu()
                )

                debris_outward_stretch_mean_m = float(
                    debris_outward_stretch.mean()
                    .detach()
                    .cpu()
                )

                # Exact top 5% by geometry-break score.
                debris_top5_count = max(
                    1,
                    int(
                        np.ceil(
                            0.05
                            * n_debris
                        )
                    ),
                )

                debris_order = torch.argsort(
                    debris_score,
                    descending=True,
                )

                debris_top_local = debris_order[
                    :debris_top5_count
                ]

                debris_top_mask = torch.zeros(
                    n_debris,
                    dtype=torch.bool,
                    device=pred.device,
                )

                debris_top_mask[
                    debris_top_local
                ] = True

                debris_other_mask = (
                    ~debris_top_mask
                )

                # Independent high-L2 top 5% for overlap test.
                l2_order = torch.argsort(
                    debris_l2,
                    descending=True,
                )

                l2_top_local = l2_order[
                    :debris_top5_count
                ]

                l2_top_mask = torch.zeros(
                    n_debris,
                    dtype=torch.bool,
                    device=pred.device,
                )

                l2_top_mask[
                    l2_top_local
                ] = True

                overlap_count = int(
                    (
                        debris_top_mask
                        & l2_top_mask
                    ).sum().item()
                )

                union_count = int(
                    (
                        debris_top_mask
                        | l2_top_mask
                    ).sum().item()
                )

                debris_top5_l2_overlap_fraction = (
                    overlap_count
                    / debris_top5_count
                )

                debris_top5_l2_jaccard = (
                    overlap_count
                    / union_count
                    if union_count > 0
                    else float("nan")
                )

                debris_top5_l2_mean_m = mean_or_nan(
                    debris_l2[
                        debris_top_mask
                    ]
                )

                debris_other_l2_mean_m = mean_or_nan(
                    debris_l2[
                        debris_other_mask
                    ]
                )

                debris_top5_minus_other_l2_mean_m = (
                    debris_top5_l2_mean_m
                    - debris_other_l2_mean_m
                )

                debris_top5_gripper_distance_mean_m = mean_or_nan(
                    debris_distance[
                        debris_top_mask
                    ]
                )

                debris_other_gripper_distance_mean_m = mean_or_nan(
                    debris_distance[
                        debris_other_mask
                    ]
                )

                debris_top5_minus_other_gripper_distance_mean_m = (
                    debris_top5_gripper_distance_mean_m
                    - debris_other_gripper_distance_mean_m
                )

                debris_top_direction_valid = (
                    debris_top_mask
                    & debris_direction_valid
                )

                debris_other_direction_valid = (
                    debris_other_mask
                    & debris_direction_valid
                )

                if (
                    debris_top_direction_valid.any()
                    and debris_other_direction_valid.any()
                ):
                    debris_top5_direction_angle_mean_deg = (
                        mean_or_nan(
                            debris_direction_angle[
                                debris_top_direction_valid
                            ]
                        )
                    )

                    debris_other_direction_angle_mean_deg = (
                        mean_or_nan(
                            debris_direction_angle[
                                debris_other_direction_valid
                            ]
                        )
                    )

                    debris_top5_minus_other_direction_angle_mean_deg = (
                        debris_top5_direction_angle_mean_deg
                        - debris_other_direction_angle_mean_deg
                    )

                # -------------------------------------------------
                # Correlations.
                # -------------------------------------------------
                score_np = (
                    debris_score.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

                l2_np = (
                    debris_l2.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

                distance_np = (
                    debris_distance.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

                if (
                    np.unique(score_np).size > 1
                    and np.unique(l2_np).size > 1
                ):
                    debris_score_l2_spearman = float(
                        spearmanr(
                            score_np,
                            l2_np,
                        )[0]
                    )

                if (
                    np.unique(score_np).size > 1
                    and np.unique(distance_np).size > 1
                ):
                    debris_score_gripper_distance_spearman = float(
                        spearmanr(
                            score_np,
                            distance_np,
                        )[0]
                    )

                dir_mask_np = (
                    debris_direction_valid
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(bool)
                )

                if int(
                    dir_mask_np.sum()
                ) >= 2:
                    angle_np = (
                        debris_direction_angle.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )

                    if (
                        np.unique(
                            score_np[
                                dir_mask_np
                            ]
                        ).size > 1
                        and np.unique(
                            angle_np[
                                dir_mask_np
                            ]
                        ).size > 1
                    ):
                        debris_score_direction_angle_spearman = float(
                            spearmanr(
                                score_np[
                                    dir_mask_np
                                ],
                                angle_np[
                                    dir_mask_np
                                ],
                            )[0]
                        )

                # Save point identities + final coordinates so
                # candidate debris points can be visualized later
                # WITHOUT defining them from L2.
                for local_i in range(
                    n_debris
                ):
                    global_i = int(
                        debris_idx[
                            local_i
                        ].item()
                    )

                    dir_ok = bool(
                        debris_direction_valid[
                            local_i
                        ].item()
                    )

                    debris_point_rows.append({
                        "sequence_index":
                            processed,
                        "key":
                            key,
                        "object_id":
                            object_id,
                        "object_name":
                            object_name,
                        "global_point_index":
                            global_i,
                        "debris_score_m":
                            float(
                                debris_score[
                                    local_i
                                ].detach().cpu()
                            ),
                        "outward_stretch_m":
                            float(
                                debris_outward_stretch[
                                    local_i
                                ].detach().cpu()
                            ),
                        "final_l2_m":
                            float(
                                debris_l2[
                                    local_i
                                ].detach().cpu()
                            ),
                        "final_gripper_distance_m":
                            float(
                                debris_distance[
                                    local_i
                                ].detach().cpu()
                            ),
                        "final_gt_motion_m":
                            float(
                                debris_gt_motion[
                                    local_i
                                ].detach().cpu()
                            ),
                        "direction_valid":
                            dir_ok,
                        "direction_angle_deg":
                            (
                                float(
                                    debris_direction_angle[
                                        local_i
                                    ].detach().cpu()
                                )
                                if dir_ok
                                else float("nan")
                            ),
                        "direction_cosine":
                            (
                                float(
                                    debris_direction_cosine[
                                        local_i
                                    ].detach().cpu()
                                )
                                if dir_ok
                                else float("nan")
                            ),
                        "is_debris_top5":
                            bool(
                                debris_top_mask[
                                    local_i
                                ].item()
                            ),
                        "is_l2_top5":
                            bool(
                                l2_top_mask[
                                    local_i
                                ].item()
                            ),
                        "gt_final_x":
                            float(
                                gt_final_obj[
                                    local_i, 0
                                ].detach().cpu()
                            ),
                        "gt_final_y":
                            float(
                                gt_final_obj[
                                    local_i, 1
                                ].detach().cpu()
                            ),
                        "gt_final_z":
                            float(
                                gt_final_obj[
                                    local_i, 2
                                ].detach().cpu()
                            ),
                        "pred_final_x":
                            float(
                                pred_final_obj[
                                    local_i, 0
                                ].detach().cpu()
                            ),
                        "pred_final_y":
                            float(
                                pred_final_obj[
                                    local_i, 1
                                ].detach().cpu()
                            ),
                        "pred_final_z":
                            float(
                                pred_final_obj[
                                    local_i, 2
                                ].detach().cpu()
                            ),
                    })

            moved_fraction = (
                moved_valid_count / valid_count
                if valid_count > 0
                else float("nan")
            )

            row = {
                "sequence_index": processed,
                "key": key,
                "object_id": object_id,
                "object_name": object_name,
                "point_count_t0": point_count,
                "valid_count": valid_count,
                "moved_valid_count":
                    moved_valid_count,
                "moved_fraction":
                    moved_fraction,

                "is_moved_object":
                    moved_valid_count > 0,

                "final_valid_count":
                    final_valid_count,

                "final_top5_count":
                    final_top5_count,

                "final_top5_gripper_distance_mean_m":
                    final_top5_gripper_distance_mean_m,

                "final_other_gripper_distance_mean_m":
                    final_other_gripper_distance_mean_m,

                "final_top5_minus_other_gripper_distance_mean_m":
                    final_top5_minus_other_gripper_distance_mean_m,

                "final_distance_l2_spearman":
                    final_distance_l2_spearman,

                "final_motion_valid_count":
                    final_motion_valid_count,

                "final_top5_gt_motion_mean_m":
                    final_top5_gt_motion_mean_m,

                "final_other_gt_motion_mean_m":
                    final_other_gt_motion_mean_m,

                "final_top5_minus_other_gt_motion_mean_m":
                    final_top5_minus_other_gt_motion_mean_m,

                "final_distance_gt_motion_spearman":
                    final_distance_gt_motion_spearman,

                "final_gt_motion_l2_spearman":
                    final_gt_motion_l2_spearman,

                "final_distance_l2_partial_spearman_gt_motion":
                    final_distance_l2_partial_spearman_gt_motion,

                # Independent debris/local-geometry analysis
                "debris_valid_count":
                    debris_valid_count,
                "debris_knn_k":
                    debris_knn_k,
                "debris_top5_count":
                    debris_top5_count,

                "debris_score_mean_m":
                    debris_score_mean_m,
                "debris_outward_stretch_mean_m":
                    debris_outward_stretch_mean_m,

                "debris_score_l2_spearman":
                    debris_score_l2_spearman,
                "debris_score_gripper_distance_spearman":
                    debris_score_gripper_distance_spearman,
                "debris_score_direction_angle_spearman":
                    debris_score_direction_angle_spearman,

                "debris_top5_l2_overlap_fraction":
                    debris_top5_l2_overlap_fraction,
                "debris_top5_l2_jaccard":
                    debris_top5_l2_jaccard,

                "debris_top5_l2_mean_m":
                    debris_top5_l2_mean_m,
                "debris_other_l2_mean_m":
                    debris_other_l2_mean_m,
                "debris_top5_minus_other_l2_mean_m":
                    debris_top5_minus_other_l2_mean_m,

                "debris_top5_gripper_distance_mean_m":
                    debris_top5_gripper_distance_mean_m,
                "debris_other_gripper_distance_mean_m":
                    debris_other_gripper_distance_mean_m,
                "debris_top5_minus_other_gripper_distance_mean_m":
                    debris_top5_minus_other_gripper_distance_mean_m,

                "debris_top5_direction_angle_mean_deg":
                    debris_top5_direction_angle_mean_deg,
                "debris_other_direction_angle_mean_deg":
                    debris_other_direction_angle_mean_deg,
                "debris_top5_minus_other_direction_angle_mean_deg":
                    debris_top5_minus_other_direction_angle_mean_deg,

                # Point-level near vs far direction analysis
                "final_direction_valid_count":
                    final_direction_valid_count,

                "final_near25_count":
                    final_near25_count,

                "final_far25_count":
                    final_far25_count,

                "final_near25_l2_mean_m":
                    final_near25_l2_mean_m,

                "final_far25_l2_mean_m":
                    final_far25_l2_mean_m,

                "final_far25_minus_near25_l2_mean_m":
                    final_far25_minus_near25_l2_mean_m,

                "final_near25_direction_angle_mean_deg":
                    final_near25_direction_angle_mean_deg,

                "final_far25_direction_angle_mean_deg":
                    final_far25_direction_angle_mean_deg,

                "final_far25_minus_near25_direction_angle_mean_deg":
                    final_far25_minus_near25_direction_angle_mean_deg,

                "final_near25_direction_cosine_mean":
                    final_near25_direction_cosine_mean,

                "final_far25_direction_cosine_mean":
                    final_far25_direction_cosine_mean,

                "final_near25_opposite_direction_rate":
                    final_near25_opposite_direction_rate,

                "final_far25_opposite_direction_rate":
                    final_far25_opposite_direction_rate,

                "final_near25_gt_motion_mean_m":
                    final_near25_gt_motion_mean_m,

                "final_far25_gt_motion_mean_m":
                    final_far25_gt_motion_mean_m,

                "final_far25_minus_near25_gt_motion_mean_m":
                    final_far25_minus_near25_gt_motion_mean_m,

                "final_distance_direction_angle_spearman":
                    final_distance_direction_angle_spearman,

                "final_distance_direction_angle_partial_spearman_gt_motion":
                    final_distance_direction_angle_partial_spearman_gt_motion,

                "final_top5_far25_fraction":
                    final_top5_far25_fraction,

                "final_top5_near25_fraction":
                    final_top5_near25_fraction,

                "l2_mean_m":
                    mean_or_nan(
                        l2[object_valid]
                    ),
                "moved_l2_mean_m":
                    mean_or_nan(
                        l2[
                            object_moved_valid
                        ]
                    ),
                "velocity_error_mean_m_per_timestep":
                    mean_or_nan(
                        velocity_error[
                            object_velocity_valid
                        ]
                    ),
                "acceleration_error_mean_m_per_timestep2":
                    mean_or_nan(
                        acceleration_error[
                            object_acceleration_valid
                        ]
                    ),
                "gt_motion_mean_m_per_timestep":
                    mean_or_nan(
                        gt_motion[
                            object_velocity_valid
                        ]
                    ),
                "pred_motion_mean_m_per_timestep":
                    mean_or_nan(
                        pred_motion[
                            object_velocity_valid
                        ]
                    ),
            }

            rows.append(row)
            object_summary.append(row)

        # Which object contains the most officially-moved
        # point-timesteps in this sample?
        ranked = sorted(
            object_summary,
            key=lambda r:
                r["moved_valid_count"],
            reverse=True,
        )

        dominant = (
            ranked[0]
            if ranked
            else None
        )

        sequence_rows.append({
            "sequence_index": processed,
            "key": key,
            "final_scene_points": N,
            "num_objects": len(unique_ids),
            "official_l2": official_l2,
            "recomputed_l2":
                recomputed_l2,
            "l2_difference":
                l2_difference,
            "official_l2_exact":
                official_exact,
            "temporal_object_id_exact":
                temporal_id_exact,
            "dominant_moved_object":
                dominant["object_name"]
                if dominant is not None
                else "",
            "dominant_moved_valid_count":
                dominant[
                    "moved_valid_count"
                ]
                if dominant is not None
                else 0,
        })

        print()
        print("=" * 72)
        print(
            f"[{processed:02d}] {key}"
        )
        print(
            "official_l2_exact =",
            official_exact,
        )
        print(
            "num_objects =",
            len(unique_ids),
        )

        print("Top objects by moved count:")

        for r in ranked[:5]:
            print(
                "  ",
                r["object_name"],
                "| pts =",
                r["point_count_t0"],
                "| moved =",
                r["moved_valid_count"],
                "| moved_frac =",
                f'{r["moved_fraction"]:.4f}',
                "| L2 =",
                f'{r["l2_mean_m"]:.6f}',
                "| moved_L2 =",
                f'{r["moved_l2_mean_m"]:.6f}',
            )

        processed += 1

    if processed == 0:
        raise RuntimeError(
            "No samples processed"
        )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    out_dir = Path(
        "diagnostics/results",
        f"behavior_object_trajectory_full578_{stamp}",
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    with open(
        out_dir / "per_object.csv",
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

    with open(
        out_dir / "per_sequence.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                sequence_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            sequence_rows
        )

    if debris_point_rows:
        with open(
            out_dir / "per_point_debris.csv",
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    debris_point_rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(
                debris_point_rows
            )

    summary = {
        "processed_sequences":
            processed,
        "object_rows":
            len(rows),
        "debris_point_rows":
            len(debris_point_rows),
        "official_l2_reproduced_for_all":
            all_official_l2_exact,
        "temporal_object_ids_exact_for_all":
            all(
                r[
                    "temporal_object_id_exact"
                ]
                for r in sequence_rows
            ),
        "scope":
            "BEHAVIOR full 578 test samples; "
            "official PointWorld inference; "
            "exact object-ID sidecar mapping",
    }

    with open(
        out_dir / "summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    success = (
        processed == 578
        and all_official_l2_exact
        and summary[
            "temporal_object_ids_exact_for_all"
        ]
    )

    print()
    print("=" * 72)
    print("MULTICLIP OBJECT SMOKE DONE")
    print(
        "processed_sequences =",
        processed,
    )
    print(
        "object_rows =",
        len(rows),
    )
    print(
        "official_l2_reproduced_for_all =",
        all_official_l2_exact,
    )
    print(
        "temporal_object_ids_exact_for_all =",
        summary[
            "temporal_object_ids_exact_for_all"
        ],
    )
    print(
        "output_dir =",
        out_dir,
    )
    print("SUCCESS =", success)
    print("=" * 72)

    if not success:
        raise RuntimeError(
            "BEHAVIOR multi-object smoke FAILED"
        )

    return {
        f"diagnostic/{split}/processed_sequences":
            processed,
    }


# Runtime-only diagnostic patches.
dlmod.build_flow_sample = (
    objmap.build_flow_sample_with_object_ids
)
decmod.build_flow_sample = (
    objmap.build_flow_sample_with_object_ids
)

pmod.grid_sample_transform = (
    objmap.grid_sample_with_object_ids
)
pmod.gather = (
    objmap.gather_with_object_ids
)

Tester._accumulate_metrics = (
    multiclip_object_smoke
)

runpy.run_path(
    "eval.py",
    run_name="__main__",
)
