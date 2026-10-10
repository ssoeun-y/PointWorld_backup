from pathlib import Path
import json
import numpy as np
import webdataset as wds

from arguments import parse_args

from dataset_components.decoders import (
    decode_data,
    build_flow_sample,
)
from dataset_components.cameras import sample_cameras
from dataset_components.robot import canonicalize_gripper_keys_and_flags
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
# FIXED EXPERIMENT
# ============================================================

TARGET = "task-0001_episode_00012190-460:471"
CAN_TOKEN = "can_of_soda_114"

WDS_ROOT = Path(
    "diagnostics/data/"
    "task0001_soda_can_top1_wds_v2"
)

CAPTURE = Path(
    "diagnostics/results/"
    "behavior_task0001_soda_can_capture_top1/"
    "task-0001_episode_00012190-460_471_pointworld_output.npz"
)

OUT = Path(
    "diagnostics/results/"
    "behavior_task0001_soda_can_structural_top1"
)
OUT.mkdir(parents=True, exist_ok=True)

args = parse_args()


# ============================================================
# HELPERS
# ============================================================

def test_grid_indices(points_t0, grid_size):
    """
    Exact release test-mode selection logic:
    one FIRST point per voxel.
    """
    if points_t0.shape[0] == 0:
        return np.array([], dtype=np.int64)

    scaled = points_t0 / grid_size
    grid_coord = np.floor(scaled).astype(np.int64)

    min_coord = grid_coord.min(axis=0)
    grid_coord -= min_coord

    keys = fnv_hash_vec_nb(grid_coord)

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

    offsets = np.zeros_like(counts, dtype=np.int64)

    idx_picked = cumsum_counts + offsets
    idx_selected = idx_sort[idx_picked]

    return idx_selected


def expected_selected_cameras(sample, min_num, max_num, seed):
    """
    Exact deterministic camera-selection RNG logic.
    """
    camera_prefixes = sorted({
        key.split("_scene_")[0]
        for key in sample.keys()
        if "__" not in key
        and "_scene_flows" in key
        and key != "scene_flows"
    })

    ncam = len(camera_prefixes)

    lo = min(max(int(min_num), 1), ncam)

    if max_num is None:
        hi = ncam
    else:
        hi = min(max(int(max_num), 1), ncam)

    hi = max(hi, lo)

    key = str(sample.get("__key__", ""))
    rs = np.random.RandomState(
        _stable_int_hash(seed, key, "cam")
    )

    num = rs.randint(lo, hi + 1)

    selected = rs.choice(
        camera_prefixes,
        size=num,
        replace=False,
    )

    return list(selected)


def pair_change_mm(x0, xt):
    d0 = pdist(x0.astype(np.float64))
    dt = pdist(xt.astype(np.float64))
    return np.abs(dt - d0) * 1000.0


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


def rigid_fit_residual_mm(x0, xt):
    """
    Best-fit rigid transform x0 -> xt via Kabsch.
    Residual measures non-rigid deformation.
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


# ============================================================
# 1. LOAD EXACT RAW TARGET FROM OFFICIAL WDS
# ============================================================

print("=" * 72)
print("1. RAW TARGET")
print("=" * 72)

tars = sorted((WDS_ROOT / "test").glob("*.tar"))
assert tars, "No test tar"

raw = None

for sample in wds.WebDataset(
    [str(x) for x in tars],
    shardshuffle=False,
):
    if str(sample.get("__key__", "")) == TARGET:
        raw = sample
        break

assert raw is not None, TARGET

print("TARGET_RAW_EXACT = True")
print("raw_key =", raw["__key__"])


# ============================================================
# 2. OFFICIAL DECODE
# ============================================================

decoded = decode_data(raw, domain="behavior")

assert str(decoded["__key__"]) == TARGET

T = decoded["joint_positions"].shape[0]

print("T =", T)


# ============================================================
# 3. BUILD OBJECT LABELS FROM MESH DICTS
#    BEFORE ANY SCENE CONCATENATION
# ============================================================

print()
print("=" * 72)
print("2. CAN MESH INVENTORY")
print("=" * 72)

camera_can_masks = {}
mesh_report = {}

for key, mesh_dict in decoded.items():

    suffix = "_local_scene_points"

    if not key.endswith(suffix):
        continue

    cam = key[:-len(suffix)]

    labels = []
    report = []

    for mesh_name, pts in mesh_dict.items():

        n = int(np.asarray(pts).shape[0])
        is_can = CAN_TOKEN in str(mesh_name)

        labels.append(
            np.full(
                n,
                is_can,
                dtype=bool,
            )
        )

        if is_can:
            report.append(
                (str(mesh_name), n)
            )

    if not labels:
        continue

    camera_can_masks[cam] = np.concatenate(labels)
    mesh_report[cam] = report

    print(
        f"{cam}: total={len(camera_can_masks[cam])}, "
        f"can={int(camera_can_masks[cam].sum())}"
    )

    for name, n in report:
        print("   CAN_MESH =", name)
        print("   N_LOCAL  =", n)


assert camera_can_masks
assert any(mask.any() for mask in camera_can_masks.values())

print("CAN_MESH_FOUND = True")


# ============================================================
# 4. OFFICIAL build_flow_sample
# ============================================================

print()
print("=" * 72)
print("3. OFFICIAL FLOW CONSTRUCTION")
print("=" * 72)

robot_sampler = TorchRobotSampler(
    urdf_path=resolve_robot_urdf("behavior"),
    gripper_only=RELEASE_GRIPPER_ONLY,
    device="cpu",
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

assert str(sample["__key__"]) == TARGET


# Attach diagnostic object masks using EXACT mesh concatenation order.
for cam, mask1d in camera_can_masks.items():

    flow_key = f"{cam}_scene_flows"

    if flow_key not in sample:
        continue

    n_flow = sample[flow_key].shape[1]

    assert n_flow == len(mask1d), (
        cam,
        n_flow,
        len(mask1d),
    )

    sample[f"{cam}_scene_object_mask"] = np.repeat(
        mask1d[None, :],
        T,
        axis=0,
    )


# ============================================================
# 5. EXACT DETERMINISTIC CAMERA SELECTION
# ============================================================

selected_cameras = expected_selected_cameras(
    sample,
    args.eval_min_num_cameras,
    args.eval_max_num_cameras,
    args.seed,
)

print("eval_min_num_cameras =", args.eval_min_num_cameras)
print("eval_max_num_cameras =", args.eval_max_num_cameras)
print("SELECTED_CAMERAS =", selected_cameras)

sample = sample_cameras(
    sample,
    min_num_cameras=args.eval_min_num_cameras,
    max_num_cameras=args.eval_max_num_cameras,
    deterministic=True,
    seed=args.seed,
)

assert "scene_object_mask" in sample

sample = canonicalize_gripper_keys_and_flags(sample)

n_before = sample["scene_flows"].shape[1]
can_before = int(sample["scene_object_mask"][0].sum())

print("scene N after cameras =", n_before)
print("can N after cameras   =", can_before)


# ============================================================
# 6. EXACT RELEASE TEST PREPROCESSING
# ============================================================

print()
print("=" * 72)
print("4. RELEASE TEST POINT SAMPLING")
print("=" * 72)

# release pipeline:
# center_shift -> bounds -> resolution -> grid -> max-points
sample = center_shift(sample)

sample = filter_within_bounds(sample)

sample = assert_camera_payload_resolution(
    sample,
    expected_hw=(180, 320),
)

n_after_bounds = sample["scene_flows"].shape[1]
can_after_bounds = int(
    sample["scene_object_mask"][0].sum()
)

print("after bounds N     =", n_after_bounds)
print("after bounds can N =", can_after_bounds)


# grid_sample_transform does not know our diagnostic mask,
# so compute its exact selected index and apply the same index.
obj_before_grid = sample["scene_object_mask"].copy()

idx_grid = test_grid_indices(
    sample["scene_flows"][0],
    args.grid_size,
)

sample = grid_sample_transform(
    sample,
    grid_size=args.grid_size,
    mode="test",
)

sample["scene_object_mask"] = (
    obj_before_grid[:, idx_grid]
)

print("grid_size =", args.grid_size)
print("after grid N =", sample["scene_flows"].shape[1])
print(
    "after grid can N =",
    int(sample["scene_object_mask"][0].sum()),
)


# Same deterministic max-scene-point selection as evaluator.
sample = enforce_max_num_points(
    sample,
    max_scene_points=args.max_scene_points,
    deterministic=True,
    seed=args.seed,
)

print("max_scene_points =", args.max_scene_points)
print("after max N =", sample["scene_flows"].shape[1])
print(
    "after max can N =",
    int(sample["scene_object_mask"][0].sum()),
)

# Remaining coordinate preprocessing.
sample = center_shift(sample)
sample = normalize_colors(sample)
sample = make_gt_copy(sample)


# ============================================================
# 7. CRITICAL CROSS-CHECK AGAINST ACTUAL CAPTURE
# ============================================================

print()
print("=" * 72)
print("5. EXACT GT ALIGNMENT AUDIT")
print("=" * 72)

with np.load(CAPTURE, allow_pickle=False) as z:
    gt_capture = z["gt_scene_flows"].copy()
    pred_capture = z["pred_scene_flows"].copy()
    official_valid = z["official_valid"].astype(bool).copy()
    l2_all = z["l2"].copy()

gt_replay = sample["gt_scene_flows"]

print("replay shape  =", gt_replay.shape)
print("capture shape =", gt_capture.shape)

shape_match = gt_replay.shape == gt_capture.shape

print("SHAPE_MATCH =", shape_match)

if shape_match:
    diff = np.abs(
        gt_replay.astype(np.float64)
        - gt_capture.astype(np.float64)
    )

    max_diff = float(diff.max())
    mean_diff = float(diff.mean())

    exact = np.array_equal(
        gt_replay,
        gt_capture,
    )

    close = np.allclose(
        gt_replay,
        gt_capture,
        rtol=0.0,
        atol=1e-6,
    )

    print("GT_REPLAY_EXACT =", exact)
    print("GT_REPLAY_MAX_DIFF =", repr(max_diff))
    print("GT_REPLAY_MEAN_DIFF =", repr(mean_diff))
    print("GT_REPLAY_ALLCLOSE_1E6 =", close)

else:
    close = False


# ============================================================
# FAIL-SAFE:
# DO NOT INTERPRET OBJECT METRICS IF MAPPING DOES NOT MATCH
# ============================================================

if not shape_match or not close:

    print()
    print("=" * 72)
    print("STOP")
    print("=" * 72)

    print("OBJECT_MAPPING_SAFE = False")
    print("STRUCTURAL_METRIC_NOT_COMPUTED")
    print("NO MODEL RERUN")
    print("NO MBNSF")

else:

    print("OBJECT_MAPPING_SAFE = True")

    # ========================================================
    # 8. EXACT SODA-CAN POINT SET
    # ========================================================

    print()
    print("=" * 72)
    print("6. SODA CAN POINT SET")
    print("=" * 72)

    can_mask = sample["scene_object_mask"][0].astype(bool)

    # Same convention used in earlier rigid-object geometry:
    # select final official-valid object points,
    # then use SAME point IDs at t0 and t10.
    final_valid = official_valid[-1]

    can_final = can_mask & final_valid

    idx = np.flatnonzero(can_final)

    print("total PointWorld scene points =", len(can_mask))
    print("can points before validity    =", int(can_mask.sum()))
    print("can final-official-valid      =", len(idx))

    if len(idx) < 4:
        print("ENOUGH_CAN_POINTS = False")

    else:

        print("ENOUGH_CAN_POINTS = True")

        gt = gt_capture[:, idx, :]
        pred = pred_capture[:, idx, :]

        n = len(idx)
        n_pairs = n * (n - 1) // 2

        print("n_can_points =", n)
        print("n_unique_pairs =", n_pairs)

        # t0 model output should coincide with input geometry.
        t0_model_gt_mm = np.linalg.norm(
            pred[0] - gt[0],
            axis=-1,
        ) * 1000.0

        print(
            "pred_vs_gt_t0_max_mm =",
            float(t0_model_gt_mm.max()),
        )


        # ====================================================
        # 9. PAIR-DISTANCE RIGIDITY
        # ====================================================

        print()
        print("=" * 72)
        print("7. t0 -> t10 PAIR-DISTANCE CHANGE")
        print("=" * 72)

        gt_pair = pair_change_mm(
            gt[0],
            gt[-1],
        )

        pred_pair = pair_change_mm(
            pred[0],
            pred[-1],
        )

        gt_pair_s = summarize(gt_pair)
        pred_pair_s = summarize(pred_pair)

        print("GT:")
        for k, v in gt_pair_s.items():
            print(f"  {k:8s} = {v:.9f} mm")

        print()
        print("PRED:")
        for k, v in pred_pair_s.items():
            print(f"  {k:8s} = {v:.6f} mm")


        # ====================================================
        # 10. BEST-FIT SE(3) RESIDUAL
        # ====================================================

        print()
        print("=" * 72)
        print("8. BEST-FIT RIGID SE3 RESIDUAL @ t10")
        print("=" * 72)

        gt_se3 = rigid_fit_residual_mm(
            gt[0],
            gt[-1],
        )

        pred_se3 = rigid_fit_residual_mm(
            pred[0],
            pred[-1],
        )

        gt_se3_s = summarize(gt_se3)
        pred_se3_s = summarize(pred_se3)

        print("GT:")
        for k, v in gt_se3_s.items():
            print(f"  {k:8s} = {v:.9f} mm")

        print()
        print("PRED:")
        for k, v in pred_se3_s.items():
            print(f"  {k:8s} = {v:.6f} mm")


        # ====================================================
        # 11. ORDINARY t10 L2 FOR CONTEXT
        # ====================================================

        can_l2_t10_mm = (
            np.linalg.norm(
                pred[-1] - gt[-1],
                axis=-1,
            )
            * 1000.0
        )

        l2_s = summarize(can_l2_t10_mm)

        print()
        print("=" * 72)
        print("9. CAN t10 POSITION L2")
        print("=" * 72)

        for k, v in l2_s.items():
            print(f"  {k:8s} = {v:.6f} mm")


        # ====================================================
        # 12. TIMEWISE STRUCTURE
        # ====================================================

        pair_gt_p95 = []
        pair_pred_p95 = []

        se3_gt_p95 = []
        se3_pred_p95 = []

        for t in range(T):

            pg = pair_change_mm(
                gt[0],
                gt[t],
            )
            pp = pair_change_mm(
                pred[0],
                pred[t],
            )

            pair_gt_p95.append(
                np.percentile(pg, 95)
                if len(pg) else 0.0
            )
            pair_pred_p95.append(
                np.percentile(pp, 95)
                if len(pp) else 0.0
            )

            rg = rigid_fit_residual_mm(
                gt[0],
                gt[t],
            )
            rp = rigid_fit_residual_mm(
                pred[0],
                pred[t],
            )

            se3_gt_p95.append(
                np.percentile(rg, 95)
            )
            se3_pred_p95.append(
                np.percentile(rp, 95)
            )


        # ====================================================
        # 13. SAVE REAL-DATA VISUALIZATIONS
        # ====================================================

        timesteps = np.arange(T)

        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.plot(
            timesteps,
            pair_gt_p95,
            marker="o",
            label="GT",
        )

        ax.plot(
            timesteps,
            pair_pred_p95,
            marker="o",
            label="Prediction",
        )

        ax.set_xlabel("Future timestep")
        ax.set_ylabel(
            "Pair-distance change P95 (mm)"
        )
        ax.set_title(
            "Soda can rigid-structure consistency"
        )
        ax.grid(alpha=0.25)
        ax.legend()

        fig.tight_layout()

        pair_png = OUT / (
            "soda_can_pair_distance_p95_over_time.png"
        )

        fig.savefig(
            pair_png,
            dpi=180,
        )

        plt.close(fig)


        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.plot(
            timesteps,
            se3_gt_p95,
            marker="o",
            label="GT",
        )

        ax.plot(
            timesteps,
            se3_pred_p95,
            marker="o",
            label="Prediction",
        )

        ax.set_xlabel("Future timestep")
        ax.set_ylabel(
            "Best-fit SE(3) residual P95 (mm)"
        )
        ax.set_title(
            "Soda can non-rigid residual"
        )
        ax.grid(alpha=0.25)
        ax.legend()

        fig.tight_layout()

        se3_png = OUT / (
            "soda_can_se3_p95_over_time.png"
        )

        fig.savefig(
            se3_png,
            dpi=180,
        )

        plt.close(fig)


        # Shape-only t10 visualization:
        # independently center GT and Pred to remove translation.
        gt10c = gt[-1] - gt[-1].mean(axis=0)
        pr10c = pred[-1] - pred[-1].mean(axis=0)

        fig = plt.figure(figsize=(10, 5))

        ax1 = fig.add_subplot(
            121,
            projection="3d",
        )

        ax1.scatter(
            gt10c[:, 0],
            gt10c[:, 1],
            gt10c[:, 2],
            s=12,
        )

        ax1.set_title("GT t10 (centroid-centered)")


        ax2 = fig.add_subplot(
            122,
            projection="3d",
        )

        ax2.scatter(
            pr10c[:, 0],
            pr10c[:, 1],
            pr10c[:, 2],
            s=12,
        )

        ax2.set_title(
            "Prediction t10 (centroid-centered)"
        )

        fig.tight_layout()

        shape_png = OUT / (
            "soda_can_gt_pred_t10_centered.png"
        )

        fig.savefig(
            shape_png,
            dpi=180,
        )

        plt.close(fig)


        # ====================================================
        # 14. SAVE NUMERICAL RESULT
        # ====================================================

        np.savez_compressed(
            OUT / "soda_can_top1_structural_metrics.npz",

            scene_indices=idx,

            gt=gt,
            pred=pred,

            gt_pair_change_t10_mm=gt_pair,
            pred_pair_change_t10_mm=pred_pair,

            gt_se3_residual_t10_mm=gt_se3,
            pred_se3_residual_t10_mm=pred_se3,

            can_l2_t10_mm=can_l2_t10_mm,

            pair_gt_p95_over_time=np.asarray(
                pair_gt_p95
            ),
            pair_pred_p95_over_time=np.asarray(
                pair_pred_p95
            ),

            se3_gt_p95_over_time=np.asarray(
                se3_gt_p95
            ),
            se3_pred_p95_over_time=np.asarray(
                se3_pred_p95
            ),
        )


        summary = {
            "target": TARGET,
            "object": CAN_TOKEN,
            "selected_cameras": selected_cameras,
            "n_can_points": n,
            "n_unique_pairs": n_pairs,

            "gt_pair_change_t10_mm": gt_pair_s,
            "pred_pair_change_t10_mm": pred_pair_s,

            "gt_se3_t10_mm": gt_se3_s,
            "pred_se3_t10_mm": pred_se3_s,

            "can_l2_t10_mm": l2_s,
        }

        with open(
            OUT / "summary.json",
            "w",
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
            )

        print()
        print("PAIR_PLOT =", pair_png)
        print("SE3_PLOT  =", se3_png)
        print("SHAPE_PLOT =", shape_png)

        print()
        print("=" * 72)
        print("FINAL")
        print("=" * 72)

        print("TARGET_EXACT = True")
        print("OBJECT_MAPPING_SAFE = True")
        print("NO MODEL RERUN")
        print("NO MBNSF")
        print("STRUCTURAL_ANALYSIS_SUCCESS = True")
