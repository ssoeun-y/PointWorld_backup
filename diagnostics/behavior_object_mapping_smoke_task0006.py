#!/usr/bin/env python3

import copy
import numpy as np
import torch

from arguments import parse_args
import dataset_components.dataloader as dlmod
import dataset_components.pipeline as pmod
import dataset_components.transforms as tmod


DATA_DIR = (
    "/home/soenchun/cse/PointWorld_workspace/"
    "datasets/behavior/wds_task0006"
)

# ---------------------------------------------------------------------
# Official references: NEVER modify these objects.
# ---------------------------------------------------------------------
ORIG_BUILD_FLOW_SAMPLE = dlmod.build_flow_sample
ORIG_GRID_SAMPLE = pmod.grid_sample_transform
ORIG_GATHER = pmod.gather

# sample key -> {integer object id: mesh name}
OBJECT_TABLE = {}


def make_args():
    """
    Match diagnostics/gt_consistency_official.py exactly for BEHAVIOR eval.
    """
    args = parse_args(skip_command_line=True)

    args.domains = ["behavior"]
    args.data_dirs = [DATA_DIR]

    args.batch_size = 1
    args.num_workers = 0
    args.eval_num_workers = 0

    args.seed = 42
    args.device = "cpu"
    args.deterministic_data = True

    return args


def build_official_loader():
    """
    Unmodified official PointWorld test dataloader.
    """
    args = make_args()

    loader, info = dlmod.build_dataloader(
        args,
        mode="test",
        rank=0,
        world_size=1,
        override_splits="test",
    )

    if loader is None:
        raise RuntimeError("official test dataloader creation failed")

    return loader, info


def normalize_key(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    if isinstance(x, np.ndarray):
        if x.size == 1:
            x = x.reshape(-1)[0]

    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            x = x[0]

    if isinstance(x, bytes):
        x = x.decode("utf-8")

    return str(x)


# ---------------------------------------------------------------------
# Diagnostic augmentation #1:
# Attach per-point mesh/object IDs BEFORE official sample_cameras().
#
# IMPORTANT:
# scene points themselves are produced entirely by original
# build_flow_sample(). We only create a parallel integer array.
# ---------------------------------------------------------------------
def build_flow_sample_with_object_ids(
    sample,
    domain,
    robot_sampler,
    max_robot_points,
    deterministic=False,
    seed=None,
    force_single_arm=False,
):
    out = ORIG_BUILD_FLOW_SAMPLE(
        sample,
        domain=domain,
        robot_sampler=robot_sampler,
        max_robot_points=max_robot_points,
        deterministic=deterministic,
        seed=seed,
        force_single_arm=force_single_arm,
    )

    if "behavior" not in domain:
        return out

    sample_key = normalize_key(
        out.get("__key__", sample.get("__key__", ""))
    )

    # Collect mesh names across all cameras.
    camera_point_dicts = {}

    suffix = "_local_scene_points"

    for key, value in sample.items():
        if key.endswith(suffix) and isinstance(value, dict):
            camera = key[:-len(suffix)]
            camera_point_dicts[camera] = value

    if not camera_point_dicts:
        raise RuntimeError(
            f"[{sample_key}] no *_local_scene_points dictionaries found"
        )

    # Stable diagnostic-only mesh IDs.
    all_mesh_names = sorted({
        mesh_name
        for point_dict in camera_point_dicts.values()
        for mesh_name in point_dict.keys()
    })

    name_to_id = {
        name: idx
        for idx, name in enumerate(all_mesh_names)
    }

    OBJECT_TABLE[sample_key] = {
        idx: name
        for name, idx in name_to_id.items()
    }

    # SimDataDecoder uses:
    # mesh_names = list(local_points.keys())
    # and concatenates meshes in exactly that order.
    for camera, local_points in camera_point_dicts.items():
        flow_key = f"{camera}_scene_flows"

        if flow_key not in out:
            continue

        mesh_names = list(local_points.keys())

        ids = []

        for mesh_name in mesh_names:
            n_points = int(local_points[mesh_name].shape[0])

            ids.append(
                np.full(
                    n_points,
                    name_to_id[mesh_name],
                    dtype=np.int32,
                )
            )

        if len(ids) == 0:
            ids = np.empty((0,), dtype=np.int32)
        else:
            ids = np.concatenate(ids, axis=0)

        T, N = out[flow_key].shape[:2]

        if ids.shape[0] != N:
            raise RuntimeError(
                f"[{sample_key}] {camera}: "
                f"object-id count {ids.shape[0]} != scene point count {N}"
            )

        # (T,N), same identity at every timestep.
        out[f"{camera}_scene_object_id"] = np.broadcast_to(
            ids[None, :],
            (T, N),
        ).copy()

    return out


# ---------------------------------------------------------------------
# Exact reproduction of OFFICIAL grid select_indices().
#
# We do NOT use this to alter scene points.
# ORIG_GRID_SAMPLE still performs the actual PointWorld downsampling.
# This only computes the same selected indices for the parallel labels.
# ---------------------------------------------------------------------
def official_grid_indices(points_t0, grid_size, mode):
    if points_t0.shape[0] == 0:
        return np.array([], dtype=np.int64)

    scaled = points_t0 / grid_size
    grid_coord = np.floor(scaled).astype(np.int64)

    min_coord = grid_coord.min(axis=0)
    grid_coord -= min_coord

    keys = tmod.fnv_hash_vec_nb(grid_coord)

    idx_sort = np.argsort(keys)
    keys_sorted = keys[idx_sort]

    _, inv, counts = np.unique(
        keys_sorted,
        return_inverse=True,
        return_counts=True,
    )

    cumsum_counts = np.cumsum(
        np.insert(counts, 0, 0)[:-1]
    )

    if mode == "train":
        offsets = (
            np.random.randint(
                0,
                counts.max(),
                size=counts.shape,
            )
            % counts
        )
    else:
        offsets = np.zeros_like(
            counts,
            dtype=np.int64,
        )

    idx_picked = cumsum_counts + offsets
    idx_selected = idx_sort[idx_picked]

    return idx_selected


def grid_sample_with_object_ids(sample, grid_size, mode):
    labels = sample.get("scene_object_id", None)

    if labels is not None:
        N_before = sample["scene_flows"].shape[1]

        if labels.shape[1] != N_before:
            raise RuntimeError(
                "pre-grid object-id / scene point mismatch: "
                f"{labels.shape} vs {sample['scene_flows'].shape}"
            )

        idx_scene = official_grid_indices(
            sample["scene_flows"][0],
            grid_size=grid_size,
            mode=mode,
        )

        labels_after = labels[:, idx_scene].copy()

    # Actual scene downsampling is still the untouched official function.
    result = ORIG_GRID_SAMPLE(
        sample,
        grid_size=grid_size,
        mode=mode,
    )

    if labels is not None:
        if result["scene_flows"].shape[1] != labels_after.shape[1]:
            raise RuntimeError(
                "post-grid point count mismatch: "
                f"{result['scene_flows'].shape[1]} "
                f"vs {labels_after.shape[1]}"
            )

        result["scene_object_id"] = labels_after

    return result


# ---------------------------------------------------------------------
# Keep diagnostic label through official final gather().
# This changes ONLY the returned diagnostic payload.
# ---------------------------------------------------------------------
def gather_with_object_ids(sample, fields, domain):
    result = ORIG_GATHER(
        sample,
        fields=fields,
        domain=domain,
    )

    if "scene_object_id" in sample:
        result["scene_object_id"] = sample["scene_object_id"]

    return result


def tensor_exact(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.as_tensor(b)

    return (
        a.shape == b.shape
        and torch.equal(a.cpu(), b.cpu())
    )


def main():
    print("===== BASELINE: UNMODIFIED OFFICIAL LOADER =====")

    baseline_loader, baseline_info = build_official_loader()
    baseline_batch = next(iter(baseline_loader))

    baseline_key = normalize_key(
        baseline_batch["__key__"]
    )

    print("baseline_key =", baseline_key)
    print(
        "baseline scene_flows =",
        tuple(baseline_batch["scene_flows"].shape),
    )
    print(
        "baseline gt_scene_flows =",
        tuple(baseline_batch["gt_scene_flows"].shape),
    )

    # Save in RAM only so second loader cannot mutate reference objects.
    baseline_scene = (
        baseline_batch["scene_flows"]
        .detach()
        .cpu()
        .clone()
    )

    baseline_gt = (
        baseline_batch["gt_scene_flows"]
        .detach()
        .cpu()
        .clone()
    )

    baseline_context = (
        baseline_batch["scene_context_mask"]
        .detach()
        .cpu()
        .clone()
    )

    # -------------------------------------------------------------
    # Runtime monkeypatch ONLY.
    # No source file modification.
    # -------------------------------------------------------------
    dlmod.build_flow_sample = build_flow_sample_with_object_ids
    pmod.grid_sample_transform = grid_sample_with_object_ids
    pmod.gather = gather_with_object_ids

    try:
        print()
        print("===== DIAGNOSTIC LOADER + OBJECT IDS =====")

        mapped_loader, mapped_info = build_official_loader()
        mapped_batch = next(iter(mapped_loader))

    finally:
        # Restore Python module symbols even inside this process.
        dlmod.build_flow_sample = ORIG_BUILD_FLOW_SAMPLE
        pmod.grid_sample_transform = ORIG_GRID_SAMPLE
        pmod.gather = ORIG_GATHER

    mapped_key = normalize_key(
        mapped_batch["__key__"]
    )

    print("mapped_key =", mapped_key)

    labels = mapped_batch["scene_object_id"]

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # batch size 1
    if labels.ndim != 3 or labels.shape[0] != 1:
        raise RuntimeError(
            f"scene_object_id expected (1,T,N), got {labels.shape}"
        )

    labels = labels[0]

    mapped_scene = mapped_batch["scene_flows"]
    mapped_gt = mapped_batch["gt_scene_flows"]

    N_scene = int(mapped_scene.shape[2])
    N_labels = int(labels.shape[1])

    print(
        "mapped scene_flows =",
        tuple(mapped_scene.shape),
    )
    print(
        "mapped gt_scene_flows =",
        tuple(mapped_gt.shape),
    )
    print(
        "scene_object_id =",
        tuple(labels.shape),
    )

    # -------------------------------------------------------------
    # Core checks
    # -------------------------------------------------------------
    key_match = baseline_key == mapped_key

    scene_exact = tensor_exact(
        baseline_scene,
        mapped_scene,
    )

    gt_exact = tensor_exact(
        baseline_gt,
        mapped_gt,
    )

    context_exact = tensor_exact(
        baseline_context,
        mapped_batch["scene_context_mask"],
    )

    label_count_match = (
        N_scene == N_labels
    )

    # Object identity must remain exactly constant through all timesteps.
    temporal_id_exact = np.array_equal(
        labels,
        np.broadcast_to(
            labels[0:1],
            labels.shape,
        ),
    )

    table = OBJECT_TABLE.get(mapped_key, {})

    radio_ids = [
        object_id
        for object_id, name in table.items()
        if "easter_egg_179" in name
    ]

    radio_mask = np.isin(
        labels[0],
        np.asarray(radio_ids, dtype=np.int32),
    )

    radio_count = int(radio_mask.sum())

    print()
    print("===== OBJECT CHECK =====")
    print("num_object_meshes =", len(table))
    print("radio_ids =", radio_ids)
    print("radio_final_point_count =", radio_count)

    for object_id in radio_ids:
        print(
            f"radio mesh [{object_id}] =",
            table[object_id],
        )

    print()
    print("===== EXACTNESS CHECKS =====")
    print("sample_key_match =", key_match)
    print("scene_flows_exact =", scene_exact)
    print("gt_scene_flows_exact =", gt_exact)
    print("scene_context_mask_exact =", context_exact)
    print("final_point_count_match =", label_count_match)
    print("temporal_object_id_exact =", temporal_id_exact)
    print("radio_present =", radio_count > 0)

    success = all([
        key_match,
        scene_exact,
        gt_exact,
        context_exact,
        label_count_match,
        temporal_id_exact,
        radio_count > 0,
    ])

    print()
    print("SUCCESS =", success)

    if not success:
        raise RuntimeError(
            "Object mapping smoke FAILED. "
            "Do not use this mapping for analysis."
        )


if __name__ == "__main__":
    main()
