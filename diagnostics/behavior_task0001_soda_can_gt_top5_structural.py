from pathlib import Path
import json
import csv
import numpy as np
import torch
import webdataset as wds

from arguments import parse_args
from evaluation.tester import Tester

from dataset_components.decoders import (
    decode_data,
    build_flow_sample,
)
from dataset_components.cameras import sample_cameras
from dataset_components.robot import (
    canonicalize_gripper_keys_and_flags,
)
from dataset_components.constants import RELEASE_GRIPPER_ONLY
from dataset_components.transforms import (
    center_shift,
    filter_within_bounds,
    grid_sample_transform,
    enforce_max_num_points,
    normalize_colors,
    make_gt_copy,
    assert_camera_payload_resolution,
)
from dataset_components.utils import (
    _stable_int_hash,
    fnv_hash_vec_nb,
)

from robot_sampler import RobotSampler as TorchRobotSampler
from utils import resolve_robot_urdf

from scipy.spatial.distance import pdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

WDS_ROOT = Path(
    "diagnostics/data/"
    "task0001_soda_can_gt_top5_wds"
)

SELECTION_PATH = Path(
    "diagnostics/data/"
    "task0001_soda_can_gt_top5_selection.json"
)

OUT = Path(
    "diagnostics/results/"
    "behavior_task0001_soda_can_gt_top5_structural"
)
OUT.mkdir(parents=True, exist_ok=True)

args = parse_args()


# ============================================================
# BASIC HELPERS
# ============================================================

def to_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def cpu_np(x):
    return x.detach().cpu().numpy()


def summarize(x):
    x = np.asarray(x, dtype=np.float64)

    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def pair_change_mm(x0, xt):
    d0 = pdist(x0.astype(np.float64))
    dt = pdist(xt.astype(np.float64))

    return np.abs(dt - d0) * 1000.0


def rigid_fit_residual_mm(x0, xt):
    """
    Best rigid transform x0 -> xt.
    Residual = non-rigid component.
    """
    x0 = x0.astype(np.float64)
    xt = xt.astype(np.float64)

    c0 = x0.mean(axis=0)
    ct = xt.mean(axis=0)

    X = x0 - c0
    Y = xt - ct

    H = X.T @ Y

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    aligned = X @ R.T + ct

    residual = np.linalg.norm(
        aligned - xt,
        axis=1,
    )

    return residual * 1000.0


def rigid_align(source, target):
    """
    Align source -> target with best-fit rigid SE(3).
    Used ONLY for visualization.
    """
    source = source.astype(np.float64)
    target = target.astype(np.float64)

    cs = source.mean(axis=0)
    ct = target.mean(axis=0)

    X = source - cs
    Y = target - ct

    H = X.T @ Y

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    aligned = X @ R.T + ct

    return aligned


# ============================================================
# EXACT TEST VOXEL SELECTION
# ============================================================

def test_grid_indices(points_t0, grid_size):

    if len(points_t0) == 0:
        return np.array([], dtype=np.int64)

    scaled = points_t0 / grid_size

    grid_coord = np.floor(
        scaled
    ).astype(np.int64)

    min_coord = grid_coord.min(axis=0)
    grid_coord -= min_coord

    keys = fnv_hash_vec_nb(grid_coord)

    idx_sort = np.argsort(keys)
    keys_sorted = keys[idx_sort]

    _, _, counts = np.unique(
        keys_sorted,
        return_inverse=True,
        return_counts=True,
    )

    cumsum = np.cumsum(
        np.insert(
            counts,
            0,
            0,
        )[:-1]
    )

    idx_selected = idx_sort[cumsum]

    return idx_selected


# ============================================================
# EXACT CAMERA SELECTION AUDIT
# ============================================================

def expected_selected_cameras(
    sample,
    min_num,
    max_num,
    seed,
):
    camera_prefixes = sorted({
        key.split("_scene_")[0]
        for key in sample.keys()
        if "__" not in key
        and "_scene_flows" in key
        and key != "scene_flows"
    })

    ncam = len(camera_prefixes)

    lo = min(
        max(int(min_num), 1),
        ncam,
    )

    if max_num is None:
        hi = ncam
    else:
        hi = min(
            max(int(max_num), 1),
            ncam,
        )

    hi = max(hi, lo)

    key = str(
        sample.get("__key__", "")
    )

    rs = np.random.RandomState(
        _stable_int_hash(
            seed,
            key,
            "cam",
        )
    )

    num = rs.randint(
        lo,
        hi + 1,
    )

    selected = rs.choice(
        camera_prefixes,
        size=num,
        replace=False,
    )

    return list(selected)


# ============================================================
# REPLAY OFFICIAL PREPROCESSING + OBJECT MASK
# ============================================================

robot_sampler = TorchRobotSampler(
    urdf_path=resolve_robot_urdf(
        "behavior"
    ),
    gripper_only=RELEASE_GRIPPER_ONLY,
    device="cpu",
)


def reconstruct_object_mapping(
    raw,
    can_id,
):
    TARGET = str(raw["__key__"])
    CAN_TOKEN = f"can_of_soda_{can_id}"

    decoded = decode_data(
        raw,
        domain="behavior",
    )

    T = decoded[
        "joint_positions"
    ].shape[0]

    camera_can_masks = {}

    for key, mesh_dict in decoded.items():

        suffix = "_local_scene_points"

        if not key.endswith(suffix):
            continue

        cam = key[:-len(suffix)]

        labels = []

        for mesh_name, pts in mesh_dict.items():

            n = int(
                np.asarray(pts).shape[0]
            )

            is_can = (
                CAN_TOKEN
                in str(mesh_name)
            )

            labels.append(
                np.full(
                    n,
                    is_can,
                    dtype=bool,
                )
            )

        if labels:
            camera_can_masks[
                cam
            ] = np.concatenate(
                labels
            )

    if not camera_can_masks:
        raise RuntimeError(
            f"No camera mesh masks: {TARGET}"
        )

    if not any(
        m.any()
        for m in camera_can_masks.values()
    ):
        raise RuntimeError(
            f"{CAN_TOKEN} not found: {TARGET}"
        )

    sample = build_flow_sample(
        decoded,
        domain="behavior",
        robot_sampler=robot_sampler,
        max_robot_points=args.max_robot_points,
        deterministic=True,
        seed=args.seed,
        force_single_arm=False,
    )

    for cam, mask1d in (
        camera_can_masks.items()
    ):

        flow_key = (
            f"{cam}_scene_flows"
        )

        if flow_key not in sample:
            continue

        n_flow = (
            sample[flow_key].shape[1]
        )

        if n_flow != len(mask1d):
            raise RuntimeError(
                f"{TARGET}: "
                f"{cam}: flow={n_flow} "
                f"mask={len(mask1d)}"
            )

        sample[
            f"{cam}_scene_object_mask"
        ] = np.repeat(
            mask1d[None, :],
            T,
            axis=0,
        )

    selected_cameras = (
        expected_selected_cameras(
            sample,
            args.eval_min_num_cameras,
            args.eval_max_num_cameras,
            args.seed,
        )
    )

    sample = sample_cameras(
        sample,
        min_num_cameras=(
            args.eval_min_num_cameras
        ),
        max_num_cameras=(
            args.eval_max_num_cameras
        ),
        deterministic=True,
        seed=args.seed,
    )

    if "scene_object_mask" not in sample:
        raise RuntimeError(
            f"Object mask missing after "
            f"camera sampling: {TARGET}"
        )

    sample = (
        canonicalize_gripper_keys_and_flags(
            sample
        )
    )

    n_camera = (
        sample["scene_flows"].shape[1]
    )

    can_camera = int(
        sample["scene_object_mask"][
            0
        ].sum()
    )

    # Exact evaluator preprocessing.
    sample = center_shift(sample)

    sample = filter_within_bounds(
        sample
    )

    sample = (
        assert_camera_payload_resolution(
            sample,
            expected_hw=(180, 320),
        )
    )

    n_bounds = (
        sample["scene_flows"].shape[1]
    )

    can_bounds = int(
        sample["scene_object_mask"][
            0
        ].sum()
    )

    obj_before_grid = (
        sample[
            "scene_object_mask"
        ].copy()
    )

    idx_grid = test_grid_indices(
        sample["scene_flows"][0],
        args.grid_size,
    )

    sample = grid_sample_transform(
        sample,
        grid_size=args.grid_size,
        mode="test",
    )

    sample[
        "scene_object_mask"
    ] = obj_before_grid[
        :,
        idx_grid,
    ]

    n_grid = (
        sample["scene_flows"].shape[1]
    )

    can_grid = int(
        sample["scene_object_mask"][
            0
        ].sum()
    )

    sample = enforce_max_num_points(
        sample,
        max_scene_points=(
            args.max_scene_points
        ),
        deterministic=True,
        seed=args.seed,
    )

    n_final = (
        sample["scene_flows"].shape[1]
    )

    can_final = int(
        sample["scene_object_mask"][
            0
        ].sum()
    )

    sample = center_shift(sample)
    sample = normalize_colors(sample)
    sample = make_gt_copy(sample)

    return {
        "sample": sample,
        "selected_cameras":
            selected_cameras,

        "n_after_camera":
            n_camera,
        "can_after_camera":
            can_camera,

        "n_after_bounds":
            n_bounds,
        "can_after_bounds":
            can_bounds,

        "n_after_grid":
            n_grid,
        "can_after_grid":
            can_grid,

        "n_final":
            n_final,
        "can_final":
            can_final,
    }


# ============================================================
# CYLINDER VISUAL REFERENCE
# ============================================================

def canonical_can_frame(gt_points):
    """
    Build a GT-derived canonical visualization frame.

    IMPORTANT:
    Used ONLY for visualization.
    Never used in pair-distance / SE3 metrics.
    """
    x = gt_points.astype(
        np.float64
    )

    center = x.mean(axis=0)

    X = x - center

    cov = (
        X.T @ X
        / max(len(X) - 1, 1)
    )

    eigvals, eigvecs = (
        np.linalg.eigh(cov)
    )

    # Largest principal component:
    # approximate can height axis.
    order = np.argsort(
        eigvals
    )[::-1]

    height_axis = (
        eigvecs[:, order[0]]
    )

    radial_axis_1 = (
        eigvecs[:, order[1]]
    )

    radial_axis_2 = (
        eigvecs[:, order[2]]
    )

    basis = np.column_stack([
        radial_axis_1,
        radial_axis_2,
        height_axis,
    ])

    # Right-handed frame.
    if np.linalg.det(basis) < 0:
        basis[:, 0] *= -1

    return center, basis


def cylinder_reference(
    gt_canonical,
):
    """
    Robust approximate cylinder dimensions
    from GT visible can points.

    VISUALIZATION ONLY.
    """
    xy_center = np.median(
        gt_canonical[:, :2],
        axis=0,
    )

    radial = np.linalg.norm(
        gt_canonical[:, :2]
        - xy_center[None, :],
        axis=1,
    )

    radius = float(
        np.percentile(
            radial,
            95,
        )
    )

    z_lo = float(
        np.percentile(
            gt_canonical[:, 2],
            2,
        )
    )

    z_hi = float(
        np.percentile(
            gt_canonical[:, 2],
            98,
        )
    )

    z_mid = 0.5 * (
        z_lo + z_hi
    )

    return {
        "xy_center": xy_center,
        "radius": radius,
        "z_lo": z_lo,
        "z_hi": z_hi,
        "z_mid": z_mid,
    }


def draw_cylinder_wireframe(
    ax,
    ref,
):
    """
    Approximate can cylinder reference.
    """
    radius = ref["radius"]

    z_lo = (
        ref["z_lo"]
        - ref["z_mid"]
    )

    z_hi = (
        ref["z_hi"]
        - ref["z_mid"]
    )

    theta = np.linspace(
        0,
        2 * np.pi,
        48,
    )

    z = np.linspace(
        z_lo,
        z_hi,
        7,
    )

    TH, ZZ = np.meshgrid(
        theta,
        z,
    )

    XX = radius * np.cos(TH)
    YY = radius * np.sin(TH)

    ax.plot_wireframe(
        XX,
        YY,
        ZZ,
        rstride=1,
        cstride=4,
        linewidth=0.6,
        alpha=0.28,
    )

    # Top / bottom rings.
    for zz in [z_lo, z_hi]:
        ax.plot(
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.full_like(
                theta,
                zz,
            ),
            linewidth=1.0,
            alpha=0.45,
        )


def set_equal_3d(
    ax,
    points,
    ref,
):
    radius = ref["radius"]

    half_h = (
        ref["z_hi"]
        - ref["z_lo"]
    ) / 2.0

    extent = max(
        radius * 1.35,
        half_h * 1.15,
        np.max(
            np.abs(points)
        ) * 1.1,
    )

    if extent <= 0:
        extent = 0.05

    ax.set_xlim(
        -extent,
        extent,
    )

    ax.set_ylim(
        -extent,
        extent,
    )

    ax.set_zlim(
        -extent,
        extent,
    )

    ax.set_box_aspect(
        (1, 1, 1)
    )


def save_can_visualization(
    rank,
    sample_key,
    can_id,
    gt10,
    pred10,
    n_points,
):
    """
    Prediction is rigid-aligned to GT only
    for the visual figure.

    Metric values remain raw/non-aligned.
    """

    # Remove global rigid mismatch.
    pred_aligned = rigid_align(
        pred10,
        gt10,
    )

    center, basis = (
        canonical_can_frame(
            gt10
        )
    )

    gt_can = (
        (gt10 - center)
        @ basis
    )

    pred_can = (
        (pred_aligned - center)
        @ basis
    )

    ref = cylinder_reference(
        gt_can
    )

    shift = np.array([
        ref["xy_center"][0],
        ref["xy_center"][1],
        ref["z_mid"],
    ])

    gt_plot = gt_can - shift
    pred_plot = pred_can - shift

    all_pts = np.concatenate([
        gt_plot,
        pred_plot,
    ])

    fig = plt.figure(
        figsize=(12, 6)
    )

    fig.suptitle(
        f"Rank {rank} | Soda can {can_id} | "
        f"n={n_points}\n"
        "GT-fitted cylinder reference "
        "(visualization only)"
    )

    ax1 = fig.add_subplot(
        121,
        projection="3d",
    )

    draw_cylinder_wireframe(
        ax1,
        ref,
    )

    ax1.scatter(
        gt_plot[:, 0],
        gt_plot[:, 1],
        gt_plot[:, 2],
        s=22,
        depthshade=True,
        label="GT points",
    )

    ax1.set_title(
        "GT t10"
    )

    ax1.set_xlabel(
        "Can radial axis 1 (m)"
    )
    ax1.set_ylabel(
        "Can radial axis 2 (m)"
    )
    ax1.set_zlabel(
        "Can height axis (m)"
    )

    ax1.legend()

    set_equal_3d(
        ax1,
        all_pts,
        ref,
    )


    ax2 = fig.add_subplot(
        122,
        projection="3d",
    )

    draw_cylinder_wireframe(
        ax2,
        ref,
    )

    ax2.scatter(
        pred_plot[:, 0],
        pred_plot[:, 1],
        pred_plot[:, 2],
        s=22,
        depthshade=True,
        label=(
            "Prediction "
            "(rigid-aligned)"
        ),
    )

    ax2.set_title(
        "Prediction t10\n"
        "after best-fit rigid alignment"
    )

    ax2.set_xlabel(
        "Can radial axis 1 (m)"
    )
    ax2.set_ylabel(
        "Can radial axis 2 (m)"
    )
    ax2.set_zlabel(
        "Can height axis (m)"
    )

    ax2.legend()

    set_equal_3d(
        ax2,
        all_pts,
        ref,
    )

    fig.tight_layout()

    safe_key = (
        sample_key
        .replace(":", "_")
        .replace("/", "_")
    )

    path = OUT / (
        f"rank{rank}_"
        f"{safe_key}_"
        f"can{can_id}_"
        "cylinder_reference.png"
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


# ============================================================
# LOAD SELECTION
# ============================================================

selection = json.loads(
    SELECTION_PATH.read_text()
)

targets = selection["targets"]

target_info = {
    x["sample_key"]: x
    for x in targets
}

expected_keys = set(
    target_info.keys()
)

print("=" * 72)
print("1. SELECTION")
print("=" * 72)

print(
    "selection_rule =",
    selection[
        "selection_rule"
    ],
)

print(
    "prediction_outcome_used =",
    selection[
        "prediction_outcome_used_for_selection"
    ],
)

print(
    "n_targets =",
    len(expected_keys),
)

for x in targets:
    print(
        f' rank={x["rank"]} '
        f'key={x["sample_key"]} '
        f'can={x["can_id"]}'
    )

if len(expected_keys) != 5:
    raise RuntimeError(
        "Expected exactly five targets"
    )


# ============================================================
# LOAD RAW WDS TARGETS
# ============================================================

print()
print("=" * 72)
print("2. RAW WDS AUDIT")
print("=" * 72)

test_tars = sorted(
    (WDS_ROOT / "test").glob(
        "*.tar"
    )
)

if not test_tars:
    raise RuntimeError(
        "No test WDS tar"
    )

raw_by_key = {}

for raw in wds.WebDataset(
    [str(p) for p in test_tars],
    shardshuffle=False,
):
    key = str(
        raw.get("__key__", "")
    )

    if key in expected_keys:
        raw_by_key[key] = raw

print(
    "raw_target_count =",
    len(raw_by_key),
)

print(
    "RAW_TARGET_SET_EXACT =",
    set(raw_by_key)
    == expected_keys,
)

if set(raw_by_key) != expected_keys:
    raise RuntimeError(
        "Raw WDS target mismatch"
    )


# ============================================================
# OFFICIAL POINTWORLD MODEL
# ============================================================

print()
print("=" * 72)
print("3. INITIALIZE OFFICIAL POINTWORLD")
print("=" * 72)

tester = Tester(args)

dl, info = tester._build_eval_loader(
    "test",
    enable_mask=False,
)

tester.model.eval()

print(
    "OFFICIAL_TEST_LOADER_READY = True"
)


# ============================================================
# RUN FIVE TARGETS
# ============================================================

results = []
seen = set()

capture_dir = OUT / "captures"
capture_dir.mkdir(
    parents=True,
    exist_ok=True,
)

for batch_idx, batch in enumerate(dl):

    keys = [
        str(x)
        for x in batch["__key__"]
    ]

    if len(keys) != 1:
        raise RuntimeError(
            "Use batch_size=1"
        )

    key = keys[0]

    if key not in expected_keys:
        continue

    info_t = target_info[key]

    rank = int(
        info_t["rank"]
    )

    can_id = int(
        info_t["can_id"]
    )

    print()
    print("=" * 72)
    print(
        f"TARGET rank={rank}: {key}"
    )
    print("=" * 72)

    # --------------------------------
    # Object mapping replay
    # --------------------------------

    mapping = (
        reconstruct_object_mapping(
            raw_by_key[key],
            can_id,
        )
    )

    replay = mapping["sample"]

    # --------------------------------
    # Official model forward
    # --------------------------------

    batch = {
        k: (
            v.to(
                tester.device,
                non_blocking=True,
            )
            if isinstance(
                v,
                torch.Tensor,
            )
            else v
        )
        for k, v in batch.items()
    }

    with torch.inference_mode():

        outputs = tester.model(
            batch,
            training=False,
        )

        total_loss, loss_dict = (
            tester.model.loss_fn(
                outputs,
                batch,
                training=False,
            )
        )

    pred = outputs[
        "scene_flows"
    ]

    gt = batch[
        "gt_scene_flows"
    ]

    context = batch[
        "scene_context_mask"
    ]

    exists = batch[
        "scene_exists"
    ]

    supervised = batch[
        "scene_supervised_mask"
    ]

    if context.ndim == 4:
        context = context.squeeze(-1)

    if exists.ndim == 4:
        exists = exists.squeeze(-1)

    if supervised.ndim == 4:
        supervised = supervised.squeeze(
            -1
        )

    official_valid = (
        (~context.bool())
        & exists.bool()
        & supervised.bool()
    )

    l2 = torch.linalg.vector_norm(
        pred - gt,
        dim=-1,
    )

    manual_l2 = to_float(
        l2[
            official_valid
        ].mean()
    )

    official_l2 = to_float(
        loss_dict["l2/mean"]
    )

    l2_reproduced = bool(
        np.isclose(
            manual_l2,
            official_l2,
            rtol=0.0,
            atol=1e-8,
        )
    )

    print(
        "OFFICIAL_L2_REPRODUCED =",
        l2_reproduced,
    )

    if not l2_reproduced:
        raise RuntimeError(
            f"L2 mismatch: {key}"
        )

    # --------------------------------
    # Exact GT replay audit
    # --------------------------------

    gt_np = cpu_np(
        gt[0]
    )

    pred_np = cpu_np(
        pred[0]
    )

    valid_np = cpu_np(
        official_valid[0]
    ).astype(bool)

    replay_gt = replay[
        "gt_scene_flows"
    ]

    shape_match = (
        replay_gt.shape
        == gt_np.shape
    )

    if shape_match:

        replay_exact = (
            np.array_equal(
                replay_gt,
                gt_np,
            )
        )

        replay_max_diff = float(
            np.max(
                np.abs(
                    replay_gt.astype(
                        np.float64
                    )
                    - gt_np.astype(
                        np.float64
                    )
                )
            )
        )

    else:

        replay_exact = False
        replay_max_diff = np.inf

    print(
        "GT_REPLAY_EXACT =",
        replay_exact,
    )

    print(
        "GT_REPLAY_MAX_DIFF =",
        replay_max_diff,
    )

    if not replay_exact:
        raise RuntimeError(
            f"Unsafe object mapping: {key}"
        )

    # --------------------------------
    # Soda can point indices
    # --------------------------------

    can_mask = (
        replay[
            "scene_object_mask"
        ][0].astype(bool)
    )

    final_valid = valid_np[-1]

    can_final = (
        can_mask
        & final_valid
    )

    idx = np.flatnonzero(
        can_final
    )

    n = len(idx)

    print(
        "scene_points =",
        len(can_mask),
    )

    print(
        "can_before_valid =",
        int(can_mask.sum()),
    )

    print(
        "can_final_valid =",
        n,
    )

    if n < 4:
        raise RuntimeError(
            f"Too few can points: "
            f"{key}, n={n}"
        )

    n_pairs = (
        n * (n - 1) // 2
    )

    gt_can = gt_np[:, idx, :]
    pred_can = pred_np[:, idx, :]

    t0_max_mm = float(
        np.max(
            np.linalg.norm(
                pred_can[0]
                - gt_can[0],
                axis=-1,
            )
        )
        * 1000.0
    )

    print(
        "pred_vs_gt_t0_max_mm =",
        t0_max_mm,
    )

    # --------------------------------
    # Structural metrics
    # --------------------------------

    gt_pair = pair_change_mm(
        gt_can[0],
        gt_can[-1],
    )

    pred_pair = pair_change_mm(
        pred_can[0],
        pred_can[-1],
    )

    gt_pair_s = summarize(
        gt_pair
    )

    pred_pair_s = summarize(
        pred_pair
    )

    gt_se3 = (
        rigid_fit_residual_mm(
            gt_can[0],
            gt_can[-1],
        )
    )

    pred_se3 = (
        rigid_fit_residual_mm(
            pred_can[0],
            pred_can[-1],
        )
    )

    gt_se3_s = summarize(
        gt_se3
    )

    pred_se3_s = summarize(
        pred_se3
    )

    can_l2_t10 = (
        np.linalg.norm(
            pred_can[-1]
            - gt_can[-1],
            axis=-1,
        )
        * 1000.0
    )

    can_l2_s = summarize(
        can_l2_t10
    )

    # Initial object diameter:
    # useful descriptive scale only.
    initial_pair_dist = pdist(
        gt_can[0].astype(
            np.float64
        )
    ) * 1000.0

    object_diameter_mm = float(
        initial_pair_dist.max()
    )

    normalized_pair_p95 = (
        pred_pair_s["p95"]
        / max(
            object_diameter_mm,
            1e-12,
        )
    )

    # --------------------------------
    # Timewise metrics
    # --------------------------------

    T = gt_can.shape[0]

    gt_pair_p95_time = []
    pred_pair_p95_time = []

    gt_se3_p95_time = []
    pred_se3_p95_time = []

    for t in range(T):

        pg = pair_change_mm(
            gt_can[0],
            gt_can[t],
        )

        pp = pair_change_mm(
            pred_can[0],
            pred_can[t],
        )

        rg = rigid_fit_residual_mm(
            gt_can[0],
            gt_can[t],
        )

        rp = rigid_fit_residual_mm(
            pred_can[0],
            pred_can[t],
        )

        gt_pair_p95_time.append(
            float(
                np.percentile(
                    pg,
                    95,
                )
            )
            if len(pg)
            else 0.0
        )

        pred_pair_p95_time.append(
            float(
                np.percentile(
                    pp,
                    95,
                )
            )
            if len(pp)
            else 0.0
        )

        gt_se3_p95_time.append(
            float(
                np.percentile(
                    rg,
                    95,
                )
            )
        )

        pred_se3_p95_time.append(
            float(
                np.percentile(
                    rp,
                    95,
                )
            )
        )

    # --------------------------------
    # Save per-sample capture
    # --------------------------------

    safe_key = (
        key
        .replace(":", "_")
        .replace("/", "_")
    )

    np.savez_compressed(
        capture_dir
        / f"rank{rank}_{safe_key}.npz",

        sample_key=np.array(
            key
        ),

        can_id=np.array(
            can_id
        ),

        scene_indices=idx,

        gt=gt_can,
        pred=pred_can,

        official_valid=valid_np,

        gt_pair_change_t10_mm=(
            gt_pair
        ),

        pred_pair_change_t10_mm=(
            pred_pair
        ),

        gt_se3_t10_mm=(
            gt_se3
        ),

        pred_se3_t10_mm=(
            pred_se3
        ),

        can_l2_t10_mm=(
            can_l2_t10
        ),

        gt_pair_p95_time=np.asarray(
            gt_pair_p95_time
        ),

        pred_pair_p95_time=np.asarray(
            pred_pair_p95_time
        ),

        gt_se3_p95_time=np.asarray(
            gt_se3_p95_time
        ),

        pred_se3_p95_time=np.asarray(
            pred_se3_p95_time
        ),
    )

    # --------------------------------
    # CAN-LOOKING visualization
    # --------------------------------

    viz_path = save_can_visualization(
        rank=rank,
        sample_key=key,
        can_id=can_id,
        gt10=gt_can[-1],
        pred10=pred_can[-1],
        n_points=n,
    )

    # --------------------------------
    # Console
    # --------------------------------

    print()
    print(
        "GT pair P95    =",
        gt_pair_s["p95"],
        "mm",
    )

    print(
        "PRED pair P95  =",
        pred_pair_s["p95"],
        "mm",
    )

    print(
        "GT SE3 P95     =",
        gt_se3_s["p95"],
        "mm",
    )

    print(
        "PRED SE3 P95   =",
        pred_se3_s["p95"],
        "mm",
    )

    print(
        "CAN L2 mean    =",
        can_l2_s["mean"],
        "mm",
    )

    print(
        "visualization  =",
        viz_path,
    )

    row = {
        "rank": rank,
        "sample_key": key,
        "can_id": can_id,

        "n_can_points": n,
        "n_unique_pairs": n_pairs,

        "official_l2_scene_m":
            official_l2,

        "object_diameter_mm":
            object_diameter_mm,

        "gt_pair_median_mm":
            gt_pair_s["median"],

        "gt_pair_p95_mm":
            gt_pair_s["p95"],

        "gt_pair_max_mm":
            gt_pair_s["max"],

        "pred_pair_median_mm":
            pred_pair_s["median"],

        "pred_pair_p95_mm":
            pred_pair_s["p95"],

        "pred_pair_max_mm":
            pred_pair_s["max"],

        "gt_se3_median_mm":
            gt_se3_s["median"],

        "gt_se3_p95_mm":
            gt_se3_s["p95"],

        "gt_se3_max_mm":
            gt_se3_s["max"],

        "pred_se3_median_mm":
            pred_se3_s["median"],

        "pred_se3_p95_mm":
            pred_se3_s["p95"],

        "pred_se3_max_mm":
            pred_se3_s["max"],

        "can_t10_l2_mean_mm":
            can_l2_s["mean"],

        "can_t10_l2_median_mm":
            can_l2_s["median"],

        "can_t10_l2_p95_mm":
            can_l2_s["p95"],

        "normalized_pred_pair_p95":
            normalized_pair_p95,

        "gt_replay_exact":
            replay_exact,

        "official_l2_reproduced":
            l2_reproduced,

        "pred_vs_gt_t0_max_mm":
            t0_max_mm,

        "can_after_camera":
            mapping[
                "can_after_camera"
            ],

        "can_after_bounds":
            mapping[
                "can_after_bounds"
            ],

        "can_after_grid":
            mapping[
                "can_after_grid"
            ],
    }

    # Keep time curves internally
    row["_pred_pair_time"] = (
        pred_pair_p95_time
    )

    row["_gt_pair_time"] = (
        gt_pair_p95_time
    )

    row["_pred_se3_time"] = (
        pred_se3_p95_time
    )

    row["_gt_se3_time"] = (
        gt_se3_p95_time
    )

    results.append(row)
    seen.add(key)


# ============================================================
# GLOBAL AUDIT
# ============================================================

print()
print("=" * 72)
print("4. GLOBAL AUDIT")
print("=" * 72)

print(
    "seen_n =",
    len(seen),
)

print(
    "EXPECTED_TARGET_SET_EXACT =",
    seen == expected_keys,
)

if seen != expected_keys:
    print(
        "missing =",
        sorted(
            expected_keys - seen
        ),
    )
    raise RuntimeError(
        "Not all Top-5 targets processed"
    )

if len(results) != 5:
    raise RuntimeError(
        f"Expected 5 result rows, "
        f"got {len(results)}"
    )

results.sort(
    key=lambda x: x["rank"]
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = OUT / (
    "soda_can_gt_top5_"
    "structural_results.csv"
)

public_keys = [
    k
    for k in results[0].keys()
    if not k.startswith("_")
]

with open(
    csv_path,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=public_keys,
    )

    writer.writeheader()

    for r in results:

        writer.writerow({
            k: r[k]
            for k in public_keys
        })


# ============================================================
# AGGREGATE FIGURE 1:
# PAIR DISTANCE P95
# ============================================================

ranks = np.array([
    r["rank"]
    for r in results
])

gt_pair_p95 = np.array([
    r["gt_pair_p95_mm"]
    for r in results
])

pred_pair_p95 = np.array([
    r["pred_pair_p95_mm"]
    for r in results
])

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    ranks,
    gt_pair_p95,
    marker="o",
    label="GT",
)

ax.plot(
    ranks,
    pred_pair_p95,
    marker="o",
    label="Prediction",
)

ax.set_yscale("log")

ax.set_xticks(ranks)

ax.set_xlabel(
    "GT-motion rank "
    "(episode-disjoint)"
)

ax.set_ylabel(
    "t0→t10 pair-distance "
    "change P95 (mm, log scale)"
)

ax.set_title(
    "Soda can Top-5: "
    "rigid-structure consistency"
)

ax.grid(
    alpha=0.25,
    which="both",
)

ax.legend()

fig.tight_layout()

pair_summary_png = OUT / (
    "top5_pair_distance_p95_t10.png"
)

fig.savefig(
    pair_summary_png,
    dpi=200,
)

plt.close(fig)


# ============================================================
# AGGREGATE FIGURE 2:
# SE3 P95
# ============================================================

gt_se3_p95 = np.array([
    r["gt_se3_p95_mm"]
    for r in results
])

pred_se3_p95 = np.array([
    r["pred_se3_p95_mm"]
    for r in results
])

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    ranks,
    gt_se3_p95,
    marker="o",
    label="GT",
)

ax.plot(
    ranks,
    pred_se3_p95,
    marker="o",
    label="Prediction",
)

ax.set_yscale("log")

ax.set_xticks(ranks)

ax.set_xlabel(
    "GT-motion rank "
    "(episode-disjoint)"
)

ax.set_ylabel(
    "Best-fit SE(3) residual "
    "P95 (mm, log scale)"
)

ax.set_title(
    "Soda can Top-5: "
    "non-rigid residual"
)

ax.grid(
    alpha=0.25,
    which="both",
)

ax.legend()

fig.tight_layout()

se3_summary_png = OUT / (
    "top5_se3_p95_t10.png"
)

fig.savefig(
    se3_summary_png,
    dpi=200,
)

plt.close(fig)


# ============================================================
# AGGREGATE FIGURE 3:
# TIMEWISE PRED SE3
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

for r in results:

    y = np.asarray(
        r["_pred_se3_time"]
    )

    ax.plot(
        np.arange(len(y)),
        y,
        marker="o",
        label=(
            f'Rank {r["rank"]} '
            f'(can {r["can_id"]})'
        ),
    )

ax.set_xlabel(
    "Future timestep"
)

ax.set_ylabel(
    "Prediction best-fit "
    "SE(3) residual P95 (mm)"
)

ax.set_title(
    "GT-selected soda cans: "
    "structural error over time"
)

ax.grid(alpha=0.25)

ax.legend(
    fontsize=8,
)

fig.tight_layout()

time_png = OUT / (
    "top5_pred_se3_p95_over_time.png"
)

fig.savefig(
    time_png,
    dpi=200,
)

plt.close(fig)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "selection_rule":
        selection[
            "selection_rule"
        ],

    "prediction_outcome_used_for_selection":
        False,

    "n_cases":
        5,

    "pred_pair_p95_mm_per_case": [
        r["pred_pair_p95_mm"]
        for r in results
    ],

    "pred_se3_p95_mm_per_case": [
        r["pred_se3_p95_mm"]
        for r in results
    ],

    "can_l2_mean_mm_per_case": [
        r["can_t10_l2_mean_mm"]
        for r in results
    ],

    "median_case_pred_pair_p95_mm":
        float(
            np.median(
                pred_pair_p95
            )
        ),

    "median_case_pred_se3_p95_mm":
        float(
            np.median(
                pred_se3_p95
            )
        ),

    "max_gt_pair_p95_mm":
        float(
            np.max(
                gt_pair_p95
            )
        ),

    "max_gt_se3_p95_mm":
        float(
            np.max(
                gt_se3_p95
            )
        ),
}

summary_path = OUT / "summary.json"

summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


# ============================================================
# CONSOLE TABLE
# ============================================================

print()
print("=" * 72)
print("5. TOP-5 RESULTS")
print("=" * 72)

for r in results:

    print(
        f'RANK {r["rank"]} | '
        f'can={r["can_id"]} | '
        f'n={r["n_can_points"]} | '
        f'GT pair P95='
        f'{r["gt_pair_p95_mm"]:.6f} mm | '
        f'PRED pair P95='
        f'{r["pred_pair_p95_mm"]:.3f} mm | '
        f'PRED SE3 P95='
        f'{r["pred_se3_p95_mm"]:.3f} mm | '
        f'L2 mean='
        f'{r["can_t10_l2_mean_mm"]:.1f} mm'
    )

print()
print(
    "median case PRED pair P95 =",
    summary[
        "median_case_pred_pair_p95_mm"
    ],
    "mm",
)

print(
    "median case PRED SE3 P95 =",
    summary[
        "median_case_pred_se3_p95_mm"
    ],
    "mm",
)

print(
    "max GT pair P95 =",
    summary[
        "max_gt_pair_p95_mm"
    ],
    "mm",
)

print(
    "max GT SE3 P95 =",
    summary[
        "max_gt_se3_p95_mm"
    ],
    "mm",
)

print()
print(
    "CSV =",
    csv_path,
)

print(
    "PAIR SUMMARY =",
    pair_summary_png,
)

print(
    "SE3 SUMMARY =",
    se3_summary_png,
)

print(
    "TIME SUMMARY =",
    time_png,
)

print()
print("=" * 72)
print("FINAL")
print("=" * 72)

print(
    "EXPECTED_TARGET_SET_EXACT = True"
)

print(
    "ALL_GT_REPLAY_EXACT =",
    all(
        r["gt_replay_exact"]
        for r in results
    ),
)

print(
    "ALL_OFFICIAL_L2_REPRODUCED =",
    all(
        r["official_l2_reproduced"]
        for r in results
    ),
)

print(
    "CYLINDER_REFERENCE_USED_FOR_METRIC = False"
)

print(
    "CYLINDER_REFERENCE_VISUAL_ONLY = True"
)

print("NO MBNSF")
print("GPU_PHYSICAL = 1")
print(
    "TOP5_STRUCTURAL_ANALYSIS_SUCCESS = True"
)
