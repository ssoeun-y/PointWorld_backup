#!/usr/bin/env python3

import runpy
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
            official_l2_exact
            and radio_point_count == 206
            and int(radio_valid.sum()) > 0
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
