#!/usr/bin/env python3

import os
import random
import runpy
from pathlib import Path

import numpy as np
import torch

import dataset_components.dataloader as dlmod
import dataset_components.pipeline as pmod
import dataset_components.transforms as tmod
import diagnostics.behavior_object_mapping_smoke as omap
from evaluation.tester import Tester


FORCED_CAMERAS = [
    x.strip()
    for x in os.environ["PW_FORCED_CAMERAS"].split(",")
    if x.strip()
]

if len(FORCED_CAMERAS) != 2 or len(set(FORCED_CAMERAS)) != 2:
    raise RuntimeError(
        f"PW_FORCED_CAMERAS must contain exactly 2 distinct cameras; "
        f"got {FORCED_CAMERAS}"
    )
ACTION_VARIANT = os.environ.get(
    "PW_ACTION_VARIANT", "baseline"
)

VALID_ACTION_VARIANTS = {
    "baseline",
    "smooth",
    "speed_osc",
    "accel_spike",
    "reversal",
}


ACTION_CORRECTION = os.environ.get(
    "PW_ACTION_CORRECTION", "none"
)

VALID_CORRECTIONS = {
    "none",
    "quality_fix",
}

if ACTION_CORRECTION not in VALID_CORRECTIONS:
    raise RuntimeError(
        f"Unknown PW_ACTION_CORRECTION={ACTION_CORRECTION}"
    )

if ACTION_VARIANT not in VALID_ACTION_VARIANTS:
    raise RuntimeError(
        f"Unknown PW_ACTION_VARIANT={ACTION_VARIANT}; "
        f"expected {sorted(VALID_ACTION_VARIANTS)}"
    )


def make_time_map(T, variant):
    t = np.linspace(0.0, 1.0, T, dtype=np.float64)

    if variant == "baseline":
        s = t.copy()

    elif variant == "smooth":
        # Minimum-jerk time parameterization.
        s = 10*t**3 - 15*t**4 + 6*t**5

    elif variant == "speed_osc":
        # Monotonic speed oscillation; same path/endpoints.
        s = t + 0.08 * np.sin(2.0 * np.pi * t)

    elif variant == "accel_spike":
        # Strong but monotonic mid-trajectory speed burst.
        u = np.linspace(0.0, 1.0, 2001)
        v = 1.0 + 4.0 * np.exp(
            -0.5 * ((u - 0.55) / 0.07) ** 2
        )

        du = np.diff(u)
        integ = np.concatenate([
            [0.0],
            np.cumsum(0.5 * (v[:-1] + v[1:]) * du),
        ])
        integ /= integ[-1]

        s = np.interp(t, u, integ)

    elif variant == "reversal":
        # Brief backtrack along the SAME point-flow path.
        kt = np.array(
            [0.0, 0.20, 0.40, 0.55, 0.65, 0.75, 1.0]
        )
        ks = np.array(
            [0.0, 0.20, 0.45, 0.60, 0.45, 0.70, 1.0]
        )
        s = np.interp(t, kt, ks)

    else:
        raise AssertionError(variant)

    if abs(float(s[0])) > 1e-12:
        raise RuntimeError("action map does not preserve t0")

    if abs(float(s[-1]) - 1.0) > 1e-12:
        raise RuntimeError("action map does not preserve final state")

    return s.astype(np.float64)


def quality_correct_time_map(s, strength=0.50, passes=2):
    s = np.asarray(s, dtype=np.float64).copy()

    if len(s) < 3:
        return s

    corrected = s.copy()

    # Temporal low-pass.
    for _ in range(passes):
        old = corrected.copy()

        corrected[1:-1] = (
            0.25 * old[:-2]
            + 0.50 * old[1:-1]
            + 0.25 * old[2:]
        )

    # Remove backwards temporal progress.
    corrected = np.maximum.accumulate(corrected)

    # Normalize endpoints.
    corrected -= corrected[0]

    if corrected[-1] <= 1e-12:
        raise RuntimeError("invalid corrected time map")

    corrected /= corrected[-1]

    # Gentle regularization toward uniform progress.
    t = np.linspace(
        0.0,
        1.0,
        len(corrected),
        dtype=np.float64,
    )

    corrected = (
        (1.0 - strength) * corrected
        + strength * t
    )

    corrected[0] = 0.0
    corrected[-1] = 1.0

    if np.any(np.diff(corrected) < -1e-12):
        raise RuntimeError(
            "corrected action still contains backward progress"
        )

    return corrected


def resample_time(arr, s, normalize_vectors=False):
    arr = np.asarray(arr)

    T = arr.shape[0]

    if T != len(s):
        raise RuntimeError(
            f"time length mismatch: {T} vs {len(s)}"
        )

    u = np.linspace(0.0, 1.0, T)

    original_shape = arr.shape
    flat = arr.astype(np.float64).reshape(T, -1)

    out = np.empty_like(flat)

    for j in range(flat.shape[1]):
        out[:, j] = np.interp(s, u, flat[:, j])

    out = out.reshape(original_shape).astype(arr.dtype)

    if normalize_vectors:
        norm = np.linalg.norm(out, axis=-1, keepdims=True)
        out = out / np.maximum(norm, 1e-12)

    return out


TARGET_KEY = os.environ.get(
    "PW_TARGET_KEY",
    "task-0000_episode_00002530-300:311",
)
OBJECT_ID = int(os.environ.get("PW_OBJECT_ID", "2"))
SAVE_DIR = Path(
    os.environ.get(
        "PW_SAVE_DIR",
        "diagnostics/results/view_condition_smoke",
    )
)
REPEATS = int(os.environ.get("PW_REPEATS", "3"))
SEED = int(os.environ.get("PW_SEED", "42"))

# sample key -> {camera_name: raw t0 object-point count}
PAIR_RAW_OBJECT_COUNTS = {}

SAVE_DIR.mkdir(parents=True, exist_ok=True)


def reset_rng(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Same logic as official sample_cameras(), except:
# selected_prefixes = [FORCED_CAMERA]
# ------------------------------------------------------------------
def force_camera_pair(
    sample,
    min_num_cameras=1,
    max_num_cameras=None,
    deterministic=False,
    seed=None,
):
    camera_prefixes = set()

    for key in sample.keys():
        if "__" not in key and "_scene_flows" in key and key != "scene_flows":
            prefix = key.split("_scene_")[0]
            camera_prefixes.add(prefix)

    camera_prefixes = sorted(camera_prefixes)

    missing = [c for c in FORCED_CAMERAS if c not in camera_prefixes]
    if missing:
        raise RuntimeError(
            f"Requested cameras missing: {missing}. "
            f"Available cameras = {camera_prefixes}"
        )

    # Official checkpoint contract requires exactly 2 evaluation cameras.
    # We only control WHICH two available cameras are selected.
    selected_prefixes = list(FORCED_CAMERAS)
    prefix_mapping = {
        old_prefix: f"cam{i}"
        for i, old_prefix in enumerate(selected_prefixes)
    }

    # Diagnostic only: how many points from the target object were
    # actually produced by each selected camera before camera merging.
    sample_key = omap.normalize_key(sample.get("__key__", ""))

    raw_counts = {}

    for camera in selected_prefixes:
        label_key = f"{camera}_scene_object_id"

        if label_key not in sample:
            raise RuntimeError(
                f"[{sample_key}] missing {label_key}; "
                "validated object-ID hook did not run before sample_cameras"
            )

        labels = np.asarray(sample[label_key])

        if labels.ndim != 2:
            raise RuntimeError(
                f"[{sample_key}] unexpected {label_key} shape: {labels.shape}"
            )

        raw_counts[camera] = int(
            np.count_nonzero(labels[0] == OBJECT_ID)
        )

    # Carry diagnostic metadata inside the sample itself.
    # This survives DataLoader worker/process boundaries.
    sample["diag_raw_object_counts"] = np.asarray(
        [raw_counts[c] for c in selected_prefixes],
        dtype=np.int64,
    )

    # Find all scene attributes belonging to selected camera.
    # Diagnostic-only provenance.
    # Parallel labels only; official scene/model inputs are unchanged.
    for source_id, camera in enumerate(selected_prefixes):
        flow_key = f"{camera}_scene_flows"

        if flow_key not in sample:
            raise RuntimeError(f"missing {flow_key}")

        T_uid, N_uid = sample[flow_key].shape[:2]

        sample[f"{camera}_scene_point_uid"] = np.broadcast_to(
            np.arange(N_uid, dtype=np.int64)[None, :],
            (T_uid, N_uid),
        ).copy()

        sample[f"{camera}_scene_camera_source"] = np.full(
            (T_uid, N_uid),
            source_id,
            dtype=np.int16,
        )

    scene_attributes = set()

    for key in list(sample.keys()):
        for prefix in selected_prefixes:
            if key.startswith(f"{prefix}_scene_"):
                attribute = key[len(prefix) + 1:]
                scene_attributes.add(attribute)

    # Create unprefixed scene tensors.
    for attr in scene_attributes:
        sample[attr] = None

    for old_prefix in selected_prefixes:
        for attr in scene_attributes:
            prefixed_key = f"{old_prefix}_{attr}"

            if prefixed_key not in sample:
                raise RuntimeError(
                    f"Missing expected key: {prefixed_key}"
                )

            if sample[attr] is None:
                sample[attr] = sample[prefixed_key]
            else:
                sample[attr] = np.concatenate(
                    [sample[attr], sample[prefixed_key]],
                    axis=1,
                )

    # Copy selected RGB-D/camera calibration to cam0.
    image_suffixes = [
        "_initial_rgb",
        "_initial_depth",
        "_intrinsic",
        "_extrinsic",
    ]

    for old_prefix in selected_prefixes:
        new_prefix = prefix_mapping[old_prefix]

        for suffix in image_suffixes:
            old_key = f"{old_prefix}{suffix}"
            new_key = f"{new_prefix}{suffix}"

            if old_key in sample:
                sample[new_key] = sample[old_key]

    # Remove original per-camera scene/image keys exactly at diagnostic level.
    keys_to_remove = []

    for key in list(sample.keys()):
        for prefix in camera_prefixes:
            if key.startswith(f"{prefix}_scene_"):
                keys_to_remove.append(key)

    for key in list(sample.keys()):
        for prefix in camera_prefixes:
            for suffix in image_suffixes:
                if key.startswith(f"{prefix}{suffix}"):
                    keys_to_remove.append(key)

    for key in set(keys_to_remove):
        sample.pop(key, None)

    assert "scene_flows" in sample
    assert sample["scene_flows"].shape[1] > 0
    assert "robot_flows" in sample
    assert sample["robot_flows"].shape[1] > 0

    return sample


# IMPORTANT:
# dataloader.py imported sample_cameras directly,
# so patch the symbol actually used by build_dataset().
# Existing validated diagnostic augmentation:
# attaches parallel object IDs without modifying official scene points.
def build_flow_sample_with_action_variant(
    sample,
    domain,
    robot_sampler,
    max_robot_points,
    deterministic=False,
    seed=None,
    force_single_arm=False,
):
    out = omap.build_flow_sample_with_object_ids(
        sample,
        domain=domain,
        robot_sampler=robot_sampler,
        max_robot_points=max_robot_points,
        deterministic=deterministic,
        seed=seed,
        force_single_arm=force_single_arm,
    )

    T = out["robot_flows"].shape[0]

    raw_time_map = make_time_map(
        T,
        ACTION_VARIANT,
    )

    if (
        ACTION_CORRECTION == "quality_fix"
        and ACTION_VARIANT != "baseline"
    ):
        time_map = quality_correct_time_map(
            raw_time_map
        )
    else:
        time_map = raw_time_map.copy()

    # PointWorld action coordinates.
    out["robot_flows"] = resample_time(
        out["robot_flows"],
        time_map,
    )

    # Robot orientation feature must follow the same timing.
    if "robot_normals" in out:
        out["robot_normals"] = resample_time(
            out["robot_normals"],
            time_map,
            normalize_vectors=True,
        )

    # Gripper state is also part of official robot/scene features.
    for key in [
        "right_gripper_open",
        "left_gripper_open",
        "gripper_positions",
    ]:
        if key in out:
            arr = np.asarray(out[key])
            if arr.ndim >= 1 and arr.shape[0] == T:
                out[key] = resample_time(arr, time_map)

    # Fail-fast: every variant must preserve identical endpoints.
    # Interior timesteps are the ONLY intended action change.
    return out


dlmod.build_flow_sample = build_flow_sample_with_action_variant
def grid_sample_with_provenance(sample, grid_size, mode):
    if "scene_point_uid" not in sample:
        raise RuntimeError("scene_point_uid missing BEFORE grid")

    if "scene_camera_source" not in sample:
        raise RuntimeError("scene_camera_source missing BEFORE grid")

    idx = omap.official_grid_indices(
        sample["scene_flows"][0],
        grid_size,
        mode,
    )

    uid_after = sample["scene_point_uid"][:, idx].copy()
    src_after = sample["scene_camera_source"][:, idx].copy()

    out = omap.grid_sample_with_object_ids(
        sample,
        grid_size,
        mode,
    )

    if out["scene_flows"].shape[1] != len(idx):
        raise RuntimeError(
            f"grid size mismatch: "
            f"{out['scene_flows'].shape[1]} vs {len(idx)}"
        )

    out["scene_point_uid"] = uid_after
    out["scene_camera_source"] = src_after

    return out


pmod.grid_sample_transform = grid_sample_with_provenance


def gather_with_view_diag(sample, fields, domain):
    out = omap.gather_with_object_ids(sample, fields, domain)

    if "diag_raw_object_counts" in sample:
        out["diag_raw_object_counts"] = sample["diag_raw_object_counts"]

    if "scene_point_uid" not in sample:
        raise RuntimeError("scene_point_uid missing BEFORE final gather")

    if "scene_camera_source" not in sample:
        raise RuntimeError("scene_camera_source missing BEFORE final gather")

    out["scene_point_uid"] = sample["scene_point_uid"]
    out["scene_camera_source"] = sample["scene_camera_source"]

    return out


pmod.gather = gather_with_view_diag

# ------------------------------------------------------------------
# Controlled coordinate frame for VIEW-ONLY diagnostic.
#
# Official center_shift uses mean(scene + robot), so changing cameras
# also changes the processed robot/action representation.
#
# Here the translation anchor depends ONLY on robot points.
# Therefore camera choice cannot change the coordinate anchor.
# This is diagnostic-only; official tracked code/checkpoint untouched.
# ------------------------------------------------------------------
def center_shift_robot_anchor(sample):
    dtype = sample["scene_flows"].dtype

    shift = sample["robot_flows"][0].mean(axis=0).astype(dtype)

    sample["scene_flows"] -= shift
    sample["robot_flows"] -= shift

    for key in list(sample.keys()):
        if key.endswith("gripper_pose"):
            sample[key][:, :3] -= shift

    for key in list(sample.keys()):
        if key.endswith("_extrinsic"):
            extrinsic = sample[key]

            if extrinsic.ndim == 2 and extrinsic.shape == (4, 4):
                extrinsic[:3, 3] = (
                    extrinsic[:3, 3]
                    + extrinsic[:3, :3] @ shift
                )

            elif extrinsic.ndim == 3:
                extrinsic[:, :3, 3] = (
                    extrinsic[:, :3, 3]
                    + np.einsum(
                        "nij,j->ni",
                        extrinsic[:, :3, :3],
                        shift,
                    )
                )

    if "__shift_amount__" in sample:
        sample["__shift_amount__"] -= shift
    else:
        sample["__shift_amount__"] = -shift.copy()

    return sample


pmod.center_shift = tmod.center_shift

# Controlled camera-pair selection.
dlmod.sample_cameras = force_camera_pair


def mask3(batch, name):
    x = batch[name].bool()

    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    assert x.ndim == 3, (name, x.shape)
    return x


def mean_masked(x, mask):
    vals = x[mask]

    if vals.numel() == 0:
        return float("nan")

    return float(vals.float().mean().detach().cpu())


@torch.no_grad()
def view_condition_capture(
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

        print("=" * 78)
        print("VIEW CONDITION SMOKE")
        print("target key    =", key)
        print("forced cameras =", FORCED_CAMERAS)
        print("object id     =", OBJECT_ID)
        print("action variant =", ACTION_VARIANT)
        print("action correction =", ACTION_CORRECTION)
        print("=" * 78)

        # --------------------------------------------------------------
        # Keep CPU copy for reproducibility/action audit.
        # --------------------------------------------------------------
        for required in ["scene_point_uid", "scene_camera_source"]:
            if required not in batch:
                raise RuntimeError(
                    f"{required} missing from FINAL batch; "
                    "provenance propagation failed before saving"
                )

        print("PROVENANCE FINAL BATCH OK")
        print("scene_point_uid shape =", tuple(batch["scene_point_uid"].shape))
        print("scene_camera_source shape =", tuple(batch["scene_camera_source"].shape))

        cpu_capture = {}

        for name in [
            "robot_flows",
            "robot_features",
            "robot_exists",
            "joint_positions",
            "__shift_amount__",
            "scene_object_id",
            "scene_point_uid",
            "scene_camera_source",
            "scene_visibility",
            "scene_exists",
            "scene_context_mask",
            "scene_supervised_mask",
            "scene_moved_mask",
            "gt_scene_flows",
            "scene_flows",
        ]:
            if name in batch and isinstance(batch[name], torch.Tensor):
                cpu_capture[name] = batch[name].detach().cpu().clone()

        for name in [
            "cam0_initial_rgb",
            "cam1_initial_rgb",
            "cam0_initial_depth",
            "cam1_initial_depth",
            "cam0_intrinsic",
            "cam1_intrinsic",
            "cam0_extrinsic",
            "cam1_extrinsic",
        ]:
            if name in batch and isinstance(batch[name], torch.Tensor):
                cpu_capture[name] = (
                    batch[name].detach().cpu().clone()
                )

        # --------------------------------------------------------------
        # Object visibility in THIS selected view.
        # --------------------------------------------------------------
        if "scene_object_id" not in batch:
            raise RuntimeError(
                "scene_object_id missing; stop before interpretation."
            )

        object_ids_cpu = batch["scene_object_id"]

        if object_ids_cpu.ndim == 4 and object_ids_cpu.shape[-1] == 1:
            object_ids_cpu = object_ids_cpu.squeeze(-1)

        assert object_ids_cpu.ndim == 3, object_ids_cpu.shape

        scene_exists_cpu = mask3(batch, "scene_exists")

        obj0 = object_ids_cpu[:, 0] == OBJECT_ID
        scene0 = scene_exists_cpu[:, 0]
        obj_exists0 = obj0 & scene0

        n_obj = int(obj_exists0.sum())
        n_scene = int(scene0.sum())

        support_ratio = (
            n_obj / n_scene if n_scene > 0 else float("nan")
        )

        raw = batch.get("diag_raw_object_counts", None)

        if raw is None:
            raise RuntimeError(
                "diag_raw_object_counts missing from final batch"
            )

        if isinstance(raw, torch.Tensor):
            raw = raw.detach().cpu().numpy()

        raw = np.asarray(raw).reshape(-1)

        if raw.size != len(FORCED_CAMERAS):
            raise RuntimeError(
                f"unexpected raw-count shape: {raw.shape}, "
                f"cameras={FORCED_CAMERAS}"
            )

        raw_counts = {
            cam: int(raw[i])
            for i, cam in enumerate(FORCED_CAMERAS)
        }

        print("raw object points by camera =", raw_counts)
        print(
            f"post-pipeline object support t0 = "
            f"{n_obj}/{n_scene} "
            f"({100.0 * support_ratio:.2f}%)"
        )

        # --------------------------------------------------------------
        # Move batch to GPU.
        # --------------------------------------------------------------
        batch_gpu = {
            k: (
                v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }

        gt = batch_gpu["gt_scene_flows"]

        exists = mask3(batch_gpu, "scene_exists")
        context = mask3(batch_gpu, "scene_context_mask")
        supervised = mask3(batch_gpu, "scene_supervised_mask")

        official_valid = (~context) & exists & supervised

        object_ids = batch_gpu["scene_object_id"]
        if object_ids.ndim == 4 and object_ids.shape[-1] == 1:
            object_ids = object_ids.squeeze(-1)

        obj_mask = object_ids == OBJECT_ID
        obj_valid = official_valid & obj_mask

        moved = None
        if "scene_moved_mask" in batch_gpu:
            moved = mask3(batch_gpu, "scene_moved_mask")

        # --------------------------------------------------------------
        # Repeat forward because GPU inference itself has some variation.
        # --------------------------------------------------------------
        if ACTION_VARIANT != "baseline":
            print(
                "NOTE: GT L2 below is reference-only; "
                "it is NOT counterfactual accuracy."
            )

        repeat_stats = []
        first_pred = None

        for r in range(REPEATS):

            reset_rng(SEED)

            outputs = self.model(batch_gpu, training=False)
            pred = outputs["scene_flows"]

            if first_pred is None:
                first_pred = pred.detach().cpu().clone()

            err = torch.linalg.vector_norm(
                pred - gt,
                dim=-1,
            )

            final_obj = obj_valid[:, -1]
            final_all = official_valid[:, -1]

            stats = {
                "repeat": r,
                "all_t10_l2_m": mean_masked(
                    err[:, -1],
                    final_all,
                ),
                "object_t10_l2_m": mean_masked(
                    err[:, -1],
                    final_obj,
                ),
            }

            if moved is not None:
                stats["object_moved_t10_l2_m"] = mean_masked(
                    err[:, -1],
                    final_obj & moved[:, -1],
                )

            repeat_stats.append(stats)

            print(
                f"repeat {r}: "
                f"object t10 L2 = "
                f"{1000.0 * stats['object_t10_l2_m']:.3f} mm | "
                f"all t10 L2 = "
                f"{1000.0 * stats['all_t10_l2_m']:.3f} mm"
            )

        obj_l2 = np.array(
            [x["object_t10_l2_m"] for x in repeat_stats],
            dtype=np.float64,
        )

        print("-" * 78)
        print(
            "object t10 L2 mean =",
            f"{1000.0 * np.nanmean(obj_l2):.3f} mm",
        )
        print(
            "object t10 L2 range =",
            f"{1000.0 * (np.nanmax(obj_l2) - np.nanmin(obj_l2)):.3f} mm",
        )

        pair_name = "__".join(
            c.replace("/", "_") for c in FORCED_CAMERAS
        )
        suffix = ACTION_VARIANT

        if ACTION_CORRECTION != "none":
            suffix += f"__{ACTION_CORRECTION}"

        out_path = SAVE_DIR / (
            f"{pair_name}__{suffix}.pt"
        )

        torch.save(
            {
                "target_key": TARGET_KEY,
                "forced_cameras": list(FORCED_CAMERAS),
                "action_variant": ACTION_VARIANT,
                "action_correction": ACTION_CORRECTION,
                "action_time_map_raw": make_time_map(
                    batch["robot_flows"].shape[1],
                    ACTION_VARIANT,
                ),
                "action_time_map_used": (
                    quality_correct_time_map(
                        make_time_map(
                            batch["robot_flows"].shape[1],
                            ACTION_VARIANT,
                        )
                    )
                    if (
                        ACTION_CORRECTION == "quality_fix"
                        and ACTION_VARIANT != "baseline"
                    )
                    else make_time_map(
                        batch["robot_flows"].shape[1],
                        ACTION_VARIANT,
                    )
                ),
                "object_id": OBJECT_ID,
                "object_points_t0": n_obj,
                "scene_points_t0": n_scene,
                "object_support_ratio_t0": support_ratio,
                "raw_object_points_by_camera": raw_counts,
                "repeat_stats": repeat_stats,
                "first_prediction": first_pred,
                **cpu_capture,
            },
            out_path,
        )

        print("saved =", out_path)
        print("=" * 78)

        break

    if not found:
        raise RuntimeError(
            f"Target sample not found: {TARGET_KEY}"
        )

    return {
        f"diagnostic/{split}/view_condition_capture": 1.0,
    }


Tester._accumulate_metrics = view_condition_capture

reset_rng(SEED)

runpy.run_path(
    "eval.py",
    run_name="__main__",
)
