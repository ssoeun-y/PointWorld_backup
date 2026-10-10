#!/usr/bin/env python3

import csv
import json
import runpy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

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

    max_batches = 578
    processed = 0
    all_official_l2_exact = True

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

        # ----------------------------------------------------
        # Direction diagnostics.
        # Primary direction threshold:
        # both GT and prediction must move >= 1 mm/timestep.
        # This is a diagnostic threshold, not an official metric.
        # ----------------------------------------------------
        direction_speed_threshold = 1e-3

        direction_valid_base = (
            velocity_valid
            & (gt_motion >= direction_speed_threshold)
            & (pred_motion >= direction_speed_threshold)
        )

        dot = (gt_v * pred_v).sum(dim=-1)
        denom = gt_motion * pred_motion

        cosine_similarity = torch.clamp(
            dot / torch.clamp(denom, min=1e-12),
            min=-1.0,
            max=1.0,
        )

        angular_error_deg = torch.rad2deg(
            torch.acos(cosine_similarity)
        )

        magnitude_abs_error = torch.abs(
            pred_motion - gt_motion
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

            object_direction_valid = (
                direction_valid_base
                & object_mask[:, 1:]
            )

            valid_count = int(
                object_valid.sum().item()
            )

            moved_valid_count = int(
                object_moved_valid.sum().item()
            )

            moved_fraction = (
                moved_valid_count / valid_count
                if valid_count > 0
                else float("nan")
            )

            # ------------------------------------------------
            # Object-level temporal motion curve.
            #
            # For each of the 10 transitions, average motion over
            # valid points belonging to this object.
            # Then compare the GT and predicted temporal profiles.
            # ------------------------------------------------
            gt_curve = []
            pred_curve = []

            for step in range(T - 1):
                step_mask = object_velocity_valid[:, step]

                if step_mask.any():
                    gt_curve.append(
                        float(
                            gt_motion[:, step][step_mask]
                            .float().mean().detach().cpu()
                        )
                    )
                    pred_curve.append(
                        float(
                            pred_motion[:, step][step_mask]
                            .float().mean().detach().cpu()
                        )
                    )
                else:
                    gt_curve.append(float("nan"))
                    pred_curve.append(float("nan"))

            gt_curve_np = np.asarray(gt_curve, dtype=np.float64)
            pred_curve_np = np.asarray(pred_curve, dtype=np.float64)

            timing_ok = (
                np.isfinite(gt_curve_np)
                & np.isfinite(pred_curve_np)
            )

            if timing_ok.sum() >= 2:
                gt_t = gt_curve_np[timing_ok]
                pred_t = pred_curve_np[timing_ok]

                timing_motion_mae = float(
                    np.mean(np.abs(pred_t - gt_t))
                )

                # Pearson correlation is defined only when both
                # temporal curves have non-zero variance.
                if (
                    np.std(gt_t) > 0
                    and np.std(pred_t) > 0
                ):
                    timing_corr = float(
                        np.corrcoef(gt_t, pred_t)[0, 1]
                    )
                else:
                    timing_corr = float("nan")

                valid_steps = np.flatnonzero(timing_ok)

                gt_peak_step = int(
                    valid_steps[np.argmax(gt_t)] + 1
                )
                pred_peak_step = int(
                    valid_steps[np.argmax(pred_t)] + 1
                )

                peak_step_abs_error = abs(
                    pred_peak_step - gt_peak_step
                )
            else:
                timing_motion_mae = float("nan")
                timing_corr = float("nan")
                gt_peak_step = -1
                pred_peak_step = -1
                peak_step_abs_error = -1

            direction_count = int(
                object_direction_valid.sum().item()
            )

            if direction_count > 0:
                direction_cosine_mean = mean_or_nan(
                    cosine_similarity[
                        object_direction_valid
                    ]
                )

                direction_angle_mean_deg = mean_or_nan(
                    angular_error_deg[
                        object_direction_valid
                    ]
                )

                opposite_direction_rate = float(
                    (
                        cosine_similarity[
                            object_direction_valid
                        ] < 0
                    ).float().mean().detach().cpu()
                )

                angle_over_45_rate = float(
                    (
                        angular_error_deg[
                            object_direction_valid
                        ] > 45.0
                    ).float().mean().detach().cpu()
                )
            else:
                direction_cosine_mean = float("nan")
                direction_angle_mean_deg = float("nan")
                opposite_direction_rate = float("nan")
                angle_over_45_rate = float("nan")

            object_magnitude_abs_error = mean_or_nan(
                magnitude_abs_error[
                    object_velocity_valid
                ]
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

                # Direction diagnostics
                "direction_valid_count":
                    direction_count,
                "direction_cosine_mean":
                    direction_cosine_mean,
                "direction_angle_mean_deg":
                    direction_angle_mean_deg,
                "direction_angle_over_45_rate":
                    angle_over_45_rate,
                "opposite_direction_rate":
                    opposite_direction_rate,

                # Magnitude/timing diagnostics
                "motion_magnitude_abs_error_mean_m_per_timestep":
                    object_magnitude_abs_error,
                "timing_motion_curve_mae_m_per_timestep":
                    timing_motion_mae,
                "timing_motion_curve_corr":
                    timing_corr,
                "gt_peak_motion_step":
                    gt_peak_step,
                "pred_peak_motion_step":
                    pred_peak_step,
                "peak_motion_step_abs_error":
                    peak_step_abs_error,
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
        f"behavior_object_direction_timing_full578_{stamp}",
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

    summary = {
        "processed_sequences":
            processed,
        "object_rows":
            len(rows),
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
