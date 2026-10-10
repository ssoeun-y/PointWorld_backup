#!/usr/bin/env python3

import runpy
from pathlib import Path
import numpy as np
import torch

from evaluation.tester import Tester

import dataset_components.dataloader as dlmod
import dataset_components.decoders as decmod
import dataset_components.pipeline as pmod

import diagnostics.behavior_object_mapping_smoke as objmap


TARGET_OBJECT = "radio_89"
TARGET_KEY = "task-0000_episode_00000010-175:186"


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


@torch.no_grad()
def radio_prediction_smoke(
    self,
    dl,
    split,
    apply_confidence_mask=False,
):
    self.model.eval()

    found = False

    for batch_index, batch in enumerate(dl):
        key_raw = batch["__key__"]

        if isinstance(key_raw, (list, tuple)):
            key = str(key_raw[0])
        else:
            key = str(key_raw)

        if key != TARGET_KEY:
            continue

        found = True

        print("=" * 72)
        print("TARGET SAMPLE FOUND")
        print("key =", key)
        print("=" * 72)

        # Keep object IDs on CPU before moving tensors.
        object_ids = batch["scene_object_id"]

        if isinstance(object_ids, torch.Tensor):
            object_ids_cpu = object_ids.detach().cpu().numpy()
        else:
            object_ids_cpu = np.asarray(object_ids)

        # expected: (B,T,N)
        assert object_ids_cpu.ndim == 3, object_ids_cpu.shape
        assert object_ids_cpu.shape[0] == 1

        table = objmap.OBJECT_TABLE.get(key)

        if table is None:
            print("DEBUG requested key =", repr(key))
            print("DEBUG OBJECT_TABLE size =", len(objmap.OBJECT_TABLE))
            print("DEBUG OBJECT_TABLE keys =", [repr(k) for k in objmap.OBJECT_TABLE.keys()])
            raise RuntimeError(
                f"No object table found for sample {key}"
            )

        radio_ids = [
            object_id
            for object_id, name in table.items()
            if TARGET_OBJECT in name
        ]

        if len(radio_ids) == 0:
            raise RuntimeError(
                f"{TARGET_OBJECT} not present in {key}"
            )

        object_ids_t0 = object_ids_cpu[0, 0]

        radio_point_mask_np = np.isin(
            object_ids_t0,
            np.asarray(radio_ids),
        )

        radio_point_count = int(radio_point_mask_np.sum())

        if radio_point_count == 0:
            raise RuntimeError(
                f"{TARGET_OBJECT} has zero surviving points"
            )

        # Move official tensors to model device.
        batch = {
            k: v.to(self.device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
            for k, v in batch.items()
            if k != "scene_object_id"
        }

        outputs = self.model(
            batch,
            training=False,
        )

        _, official_dict = self.model.loss_fn(
            outputs,
            batch,
            training=False,
        )

        pred = outputs["scene_flows"]
        gt = batch["gt_scene_flows"]

        assert pred.shape == gt.shape
        assert pred.ndim == 4
        assert pred.shape[-1] == 3
        assert pred.shape[0] == 1

        B, T, N, _ = pred.shape

        if N != object_ids_cpu.shape[2]:
            raise RuntimeError(
                "Prediction/object mapping point-count mismatch: "
                f"pred N={N}, labels N={object_ids_cpu.shape[2]}"
            )

        # ---------------------------------------------------------
        # Official validity + moved mask
        # ---------------------------------------------------------
        context = mask3(
            batch,
            "scene_context_mask",
        )
        exists = mask3(
            batch,
            "scene_exists",
        )
        supervised = mask3(
            batch,
            "scene_supervised_mask",
        )
        moved = mask3(
            batch,
            "scene_moved_mask",
        )

        valid = (
            (~context)
            & exists
            & supervised
        )

        # ---------------------------------------------------------
        # Official L2 reproduction
        # ---------------------------------------------------------
        l2 = torch.linalg.vector_norm(
            pred - gt,
            dim=-1,
        )

        recomputed_l2 = mean_or_nan(
            l2[valid]
        )

        official_l2 = float(
            official_dict["l2/mean"]
        )

        l2_diff = abs(
            recomputed_l2 - official_l2
        )

        official_l2_exact = (
            l2_diff <= 1e-6
        )

        # ---------------------------------------------------------
        # Object mask
        # ---------------------------------------------------------
        radio_point_mask = torch.as_tensor(
            radio_point_mask_np,
            dtype=torch.bool,
            device=pred.device,
        )

        radio_mask = (
            radio_point_mask[None, None, :]
            .expand(B, T, N)
        )

        radio_valid = (
            valid & radio_mask
        )

        radio_moved_valid = (
            valid
            & moved
            & radio_mask
        )

        other_moved_valid = (
            valid
            & moved
            & (~radio_mask)
        )

        # ---------------------------------------------------------
        # Per-point position error
        # ---------------------------------------------------------
        radio_l2 = mean_or_nan(
            l2[radio_valid]
        )

        radio_moved_l2 = mean_or_nan(
            l2[radio_moved_valid]
        )

        other_moved_l2 = mean_or_nan(
            l2[other_moved_valid]
        )

        # ---------------------------------------------------------
        # Velocity error
        # units: meter / timestep
        # ---------------------------------------------------------
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

        radio_velocity_mask = (
            radio_mask[:, 1:]
            & velocity_valid
        )

        radio_velocity_error = mean_or_nan(
            velocity_error[
                radio_velocity_mask
            ]
        )

        # Ground-truth motion magnitude
        gt_motion_magnitude = torch.linalg.vector_norm(
            gt_v,
            dim=-1,
        )

        radio_gt_motion_magnitude = mean_or_nan(
            gt_motion_magnitude[
                radio_velocity_mask
            ]
        )

        # Prediction motion magnitude too, useful sanity check
        pred_motion_magnitude = torch.linalg.vector_norm(
            pred_v,
            dim=-1,
        )

        radio_pred_motion_magnitude = mean_or_nan(
            pred_motion_magnitude[
                radio_velocity_mask
            ]
        )

        # ---------------------------------------------------------
        # Acceleration error
        # units: meter / timestep^2
        # ---------------------------------------------------------
        pred_a = (
            pred_v[:, 1:]
            - pred_v[:, :-1]
        )

        gt_a = (
            gt_v[:, 1:]
            - gt_v[:, :-1]
        )

        acceleration_error = torch.linalg.vector_norm(
            pred_a - gt_a,
            dim=-1,
        )

        acceleration_valid = (
            valid[:, 2:]
            & valid[:, 1:-1]
            & valid[:, :-2]
        )

        radio_acceleration_mask = (
            radio_mask[:, 2:]
            & acceleration_valid
        )

        radio_acceleration_error = mean_or_nan(
            acceleration_error[
                radio_acceleration_mask
            ]
        )

        # ---------------------------------------------------------
        # POINT-LEVEL DEBRIS DATA
        # ---------------------------------------------------------

        # Same final scene-point index must preserve object identity
        # across every timestep.
        object_ids_t0 = object_ids_cpu[0, 0]

        temporal_object_ids_exact = bool(
            np.array_equal(
                object_ids_cpu[0],
                np.broadcast_to(
                    object_ids_t0[None, :],
                    object_ids_cpu[0].shape,
                ),
            )
        )

        if not temporal_object_ids_exact:
            raise RuntimeError(
                "Temporal object-ID alignment FAILED"
            )

        radio_scene_indices = np.flatnonzero(
            radio_point_mask_np
        ).astype(np.int64)

        if len(radio_scene_indices) != radio_point_count:
            raise RuntimeError(
                "radio point index count mismatch"
            )

        # Shapes:
        # xyz:   (T, P, 3)
        # masks: (T, P)
        radio_gt_xyz = gt[0][
            :, radio_point_mask, :
        ]

        radio_pred_xyz = pred[0][
            :, radio_point_mask, :
        ]

        radio_l2_matrix = l2[0][
            :, radio_point_mask
        ]

        radio_valid_matrix = valid[0][
            :, radio_point_mask
        ]

        radio_moved_matrix = moved[0][
            :, radio_point_mask
        ]

        # ---------------------------------------------------------
        # Radio GT point <-> gripper surface proximity
        #
        # robot_flows is gripper-only in the released PointWorld
        # configuration (RELEASE_GRIPPER_ONLY=True).
        #
        # Same timestep, same transformed scene coordinate frame.
        # Result shape: (T, P_radio)
        # ---------------------------------------------------------
        if "robot_flows" not in batch:
            raise RuntimeError(
                "robot_flows missing from batch"
            )

        robot_flows = batch["robot_flows"]

        if robot_flows.ndim != 4:
            raise RuntimeError(
                f"Unexpected robot_flows shape: {robot_flows.shape}"
            )

        if robot_flows.shape[0] != 1:
            raise RuntimeError(
                f"Expected batch size 1, got {robot_flows.shape[0]}"
            )

        if robot_flows.shape[1] != T:
            raise RuntimeError(
                "robot/scene timestep mismatch: "
                f"robot T={robot_flows.shape[1]}, scene T={T}"
            )

        gripper_xyz = robot_flows[0]  # (T, Nr, 3)

        if gripper_xyz.shape[-1] != 3:
            raise RuntimeError(
                f"Unexpected gripper XYZ shape: {gripper_xyz.shape}"
            )

        # For every timestep:
        # each radio GT point -> nearest gripper surface point.
        radio_gripper_distance = torch.empty(
            (T, radio_point_count),
            dtype=gt.dtype,
            device=gt.device,
        )

        for t_dist in range(T):
            distances = torch.cdist(
                radio_gt_xyz[t_dist].unsqueeze(0),
                gripper_xyz[t_dist].unsqueeze(0),
                p=2,
            ).squeeze(0)  # (P_radio, N_gripper)

            radio_gripper_distance[t_dist] = (
                distances.min(dim=1).values
            )

        # Per persistent radio point:
        # closest it ever gets to the gripper over the sequence.
        min_gripper_distance_per_point = (
            radio_gripper_distance.min(dim=0).values
        )

        expected_xyz_shape = (
            T,
            radio_point_count,
            3,
        )

        if tuple(radio_gt_xyz.shape) != expected_xyz_shape:
            raise RuntimeError(
                "radio GT shape mismatch: "
                f"{tuple(radio_gt_xyz.shape)}"
            )

        if radio_pred_xyz.shape != radio_gt_xyz.shape:
            raise RuntimeError(
                "radio Pred/GT shape mismatch"
            )

        if tuple(radio_l2_matrix.shape) != (
            T,
            radio_point_count,
        ):
            raise RuntimeError(
                "radio L2 shape mismatch"
            )

        # ---------------------------------------------------------
        # mean / max / final L2 per persistent point
        # Valid timesteps only.
        # ---------------------------------------------------------
        valid_counts = radio_valid_matrix.sum(dim=0)
        has_valid = valid_counts > 0

        mean_l2_per_point = torch.full(
            (radio_point_count,),
            float("nan"),
            dtype=l2.dtype,
            device=l2.device,
        )

        valid_l2_sum = torch.where(
            radio_valid_matrix,
            radio_l2_matrix,
            torch.zeros_like(radio_l2_matrix),
        ).sum(dim=0)

        mean_l2_per_point[has_valid] = (
            valid_l2_sum[has_valid]
            / valid_counts[has_valid].to(l2.dtype)
        )

        max_l2_per_point = torch.full_like(
            mean_l2_per_point,
            float("nan"),
        )

        max_source = radio_l2_matrix.masked_fill(
            ~radio_valid_matrix,
            float("-inf"),
        )

        max_values = max_source.max(dim=0).values

        max_l2_per_point[has_valid] = (
            max_values[has_valid]
        )

        final_l2_per_point = torch.full_like(
            mean_l2_per_point,
            float("nan"),
        )

        final_valid = radio_valid_matrix[-1]

        final_l2_per_point[final_valid] = (
            radio_l2_matrix[-1, final_valid]
        )

        # ---------------------------------------------------------
        # Exact top 1%, 5%, 10%
        # ceil: 206 points -> 3 / 11 / 21
        # ---------------------------------------------------------
        def top_fraction_mask(values, fraction):
            finite_idx = torch.nonzero(
                torch.isfinite(values),
                as_tuple=False,
            ).squeeze(1)

            n = int(finite_idx.numel())

            result = torch.zeros(
                values.shape,
                dtype=torch.bool,
                device=values.device,
            )

            if n == 0:
                return result

            k = max(
                1,
                int(np.ceil(n * fraction)),
            )

            chosen_local = torch.topk(
                values[finite_idx],
                k=k,
                largest=True,
                sorted=True,
            ).indices

            result[
                finite_idx[chosen_local]
            ] = True

            return result

        top_mean_1 = top_fraction_mask(
            mean_l2_per_point, 0.01
        )
        top_mean_5 = top_fraction_mask(
            mean_l2_per_point, 0.05
        )
        top_mean_10 = top_fraction_mask(
            mean_l2_per_point, 0.10
        )

        top_max_1 = top_fraction_mask(
            max_l2_per_point, 0.01
        )
        top_max_5 = top_fraction_mask(
            max_l2_per_point, 0.05
        )
        top_max_10 = top_fraction_mask(
            max_l2_per_point, 0.10
        )

        top_final_1 = top_fraction_mask(
            final_l2_per_point, 0.01
        )
        top_final_5 = top_fraction_mask(
            final_l2_per_point, 0.05
        )
        top_final_10 = top_fraction_mask(
            final_l2_per_point, 0.10
        )

        # ---------------------------------------------------------
        # Save complete point-level data
        # ---------------------------------------------------------
        output_dir = Path(
            "diagnostics/results/"
            "behavior_radio_debris_visual_smoke"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        safe_key = key.replace(":", "_")

        npz_path = output_dir / (
            f"{safe_key}_{TARGET_OBJECT}_point_errors.npz"
        )

        np.savez_compressed(
            npz_path,

            sample_key=np.asarray(key),
            object_name=np.asarray(TARGET_OBJECT),

            scene_point_index=radio_scene_indices,

            gt_xyz=radio_gt_xyz.detach().cpu().numpy(),
            pred_xyz=radio_pred_xyz.detach().cpu().numpy(),

            l2=radio_l2_matrix.detach().cpu().numpy(),

            valid=radio_valid_matrix.detach().cpu().numpy(),
            moved=radio_moved_matrix.detach().cpu().numpy(),

            gripper_distance=(
                radio_gripper_distance
                .detach()
                .cpu()
                .numpy()
            ),

            min_gripper_distance=(
                min_gripper_distance_per_point
                .detach()
                .cpu()
                .numpy()
            ),

            mean_l2=mean_l2_per_point.detach().cpu().numpy(),
            max_l2=max_l2_per_point.detach().cpu().numpy(),
            final_l2=final_l2_per_point.detach().cpu().numpy(),

            top_mean_1pct=top_mean_1.detach().cpu().numpy(),
            top_mean_5pct=top_mean_5.detach().cpu().numpy(),
            top_mean_10pct=top_mean_10.detach().cpu().numpy(),

            top_max_1pct=top_max_1.detach().cpu().numpy(),
            top_max_5pct=top_max_5.detach().cpu().numpy(),
            top_max_10pct=top_max_10.detach().cpu().numpy(),

            top_final_1pct=top_final_1.detach().cpu().numpy(),
            top_final_5pct=top_final_5.detach().cpu().numpy(),
            top_final_10pct=top_final_10.detach().cpu().numpy(),
        )

        print()
        print("===== POINT-LEVEL DEBRIS DATA =====")

        print(
            "temporal_object_ids_exact =",
            temporal_object_ids_exact,
        )

        print(
            "radio_point_trajectory_shape =",
            tuple(radio_gt_xyz.shape),
        )

        print(
            "robot_flows_shape =",
            tuple(robot_flows.shape),
        )

        print(
            "radio_gripper_distance_shape =",
            tuple(radio_gripper_distance.shape),
        )

        print(
            "min_radio_gripper_distance_m =",
            float(
                min_gripper_distance_per_point.min()
            ),
        )

        print(
            "top_mean 1/5/10 =",
            int(top_mean_1.sum()),
            int(top_mean_5.sum()),
            int(top_mean_10.sum()),
        )

        print(
            "top_max 1/5/10 =",
            int(top_max_1.sum()),
            int(top_max_5.sum()),
            int(top_max_10.sum()),
        )

        print(
            "top_final 1/5/10 =",
            int(top_final_1.sum()),
            int(top_final_5.sum()),
            int(top_final_10.sum()),
        )

        print(
            "NPZ =",
            npz_path,
        )

        # ---------------------------------------------------------
        # Per-timestep radio L2
        # Future t only; context timestep naturally has no valid pts.
        # ---------------------------------------------------------
        print()
        print("===== PER-TIMESTEP RADIO L2 =====")

        for t in range(T):
            m = radio_valid[:, t]

            count = int(m.sum().item())

            value = (
                mean_or_nan(l2[:, t][m])
                if count > 0
                else float("nan")
            )

            print(
                f"t={t:02d}  "
                f"valid_points={count:4d}  "
                f"radio_l2={value:.8f}"
            )

        # ---------------------------------------------------------
        # Results
        # ---------------------------------------------------------
        print()
        print("===== OBJECT MAPPING =====")
        print("radio_ids =", radio_ids)
        print(
            "radio_final_point_count =",
            radio_point_count,
        )
        print(
            "final_scene_point_count =",
            N,
        )

        print()
        print("===== OFFICIAL L2 CHECK =====")
        print(
            "official_l2 =",
            official_l2,
        )
        print(
            "recomputed_l2 =",
            recomputed_l2,
        )
        print(
            "l2_difference =",
            l2_diff,
        )
        print(
            "official_l2_reproduced =",
            official_l2_exact,
        )

        print()
        print("===== RADIO_89 RESULTS =====")
        print(
            "radio_valid_count =",
            int(radio_valid.sum()),
        )
        print(
            "radio_moved_valid_count =",
            int(radio_moved_valid.sum()),
        )
        print(
            "other_moved_valid_count =",
            int(other_moved_valid.sum()),
        )

        print(
            "radio_l2 [m] =",
            radio_l2,
        )
        print(
            "radio_moved_l2 [m] =",
            radio_moved_l2,
        )
        print(
            "other_moved_l2 [m] =",
            other_moved_l2,
        )

        print(
            "radio_velocity_error [m/timestep] =",
            radio_velocity_error,
        )
        print(
            "radio_acceleration_error [m/timestep^2] =",
            radio_acceleration_error,
        )

        print(
            "radio_gt_motion_magnitude [m/timestep] =",
            radio_gt_motion_magnitude,
        )
        print(
            "radio_pred_motion_magnitude [m/timestep] =",
            radio_pred_motion_magnitude,
        )

        success = (
            key == TARGET_KEY
            and official_l2_exact
            and temporal_object_ids_exact
            and radio_point_count == 206
            and T == 11
            and tuple(radio_gt_xyz.shape) == (11, 206, 3)
            and tuple(radio_pred_xyz.shape) == (11, 206, 3)
            and tuple(radio_gripper_distance.shape) == (11, 206)
            and torch.isfinite(radio_gripper_distance).all()
            and int(radio_valid.sum()) > 0
            and npz_path.is_file()
        )

        print()
        print("SUCCESS =", success)

        if not success:
            raise RuntimeError(
                "BEHAVIOR radio prediction smoke FAILED"
            )

        # Stop after exact target sample.
        return {
            f"diagnostic/{split}/radio_smoke": 1,
        }

    if not found:
        raise RuntimeError(
            f"Target sample not found: {TARGET_KEY}"
        )


# -------------------------------------------------------------
# Runtime-only patches.
# No official source file modification.
# -------------------------------------------------------------
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
    radio_prediction_smoke
)

runpy.run_path(
    "eval.py",
    run_name="__main__",
)
