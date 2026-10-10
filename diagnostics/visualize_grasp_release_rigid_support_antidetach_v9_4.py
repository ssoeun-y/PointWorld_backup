from pathlib import Path
import argparse
import copy
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arguments import parse_args
from evaluation.tester import Tester
import dataset_components.dataloader as dlmod

import diagnostics.behavior_object_mapping_smoke_task0006 as omap
from diagnostics.behavior_object_mpjpe_pa_smoke_task0006 import (
    MODEL_PATH,
    NORM_STATS,
)

TASK_DIRS = {
    "task-0000": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
    "task-0001": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0001",
    "task-0006": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0006",
    "task-0008": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0008",
    "task-0010": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0010",
}

CORR_ROOT = Path(
    "diagnostics/results/"
    "acc_only_boundary_correction_current10457/"
    "corrected_actions"
)

OUTDIR = Path(
    "diagnostics/results/"
    "grasp_release_adaptive_contact_v6_original_pa"
)

def reset_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def normalize_key(x):
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, bytes):
        x = x.decode()
    return str(x)

def resample(a, phase, normalize=False):
    a = np.asarray(a)
    T = a.shape[0]

    p = np.clip(
        np.asarray(phase, dtype=float),
        0.0,
        T - 1,
    )

    lo = np.floor(p).astype(int)
    hi = np.minimum(lo + 1, T - 1)

    w = p - lo
    shape = (T,) + (1,) * (a.ndim - 1)
    w = w.reshape(shape)

    out = (
        (1.0 - w) * a[lo]
        + w * a[hi]
    )

    if normalize:
        n = np.linalg.norm(
            out,
            axis=-1,
            keepdims=True,
        )
        out = out / np.maximum(n, 1e-12)

    return out.astype(a.dtype, copy=False)

def mask3(batch, name):
    x = batch[name][0]

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    x = np.asarray(x).astype(bool)

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    return x

def to_device(batch, device):
    return {
        k: (
            v.to(device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in batch.items()
    }

parser = argparse.ArgumentParser()
parser.add_argument("--npz", default=None)
parser.add_argument("--t", type=int, default=10)
parser.add_argument("--object_id", type=int, default=None)

args_local = parser.parse_args()

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ============================================================
# Choose corrected clip
# ============================================================

if args_local.npz is None:
    files = sorted(CORR_ROOT.glob("*.npz"))

    if not files:
        raise RuntimeError(
            f"No corrected action NPZ: {CORR_ROOT}"
        )

    npz_path = files[0]

else:
    npz_path = Path(args_local.npz)

z = np.load(
    npz_path,
    allow_pickle=True,
)

TARGET = str(
    z["clip"].item()
)

TASK = str(
    z["task"].item()
)

PHASE = np.asarray(
    z["phase"],
    dtype=np.float64,
)

print("=" * 80)
print("TARGET =", TARGET)
print("TASK   =", TASK)
print("PHASE  =", np.round(PHASE, 3))
print("=" * 80)

# ============================================================
# Object-ID mapping
# ============================================================

OBJECT_BUILD = (
    omap.build_flow_sample_with_object_ids
)

# Diagnostic-only sidecar.
# PointWorld RobotSampler.compute_points() concatenates sampled points
# mesh-by-mesh without changing their point order.
TARGET_ROBOT_LINK_LABELS_PREGRID = None
TARGET_ROBOT_LINK_LABELS = None

def grid_sample_with_object_ids_and_robot_links(
    sample,
    grid_size,
    mode,
):
    global TARGET_ROBOT_LINK_LABELS

    key_now = normalize_key(
        sample.get("__key__", "")
    )

    robot_labels_after = None

    if (
        key_now == TARGET
        and TARGET_ROBOT_LINK_LABELS_PREGRID is not None
        and "robot_flows" in sample
    ):
        pre = np.asarray(
            TARGET_ROBOT_LINK_LABELS_PREGRID,
            dtype=object,
        )

        n_before = int(
            sample["robot_flows"].shape[1]
        )

        if len(pre) != n_before:
            raise RuntimeError(
                "PRE-grid robot-label mismatch: "
                f"{len(pre)} vs {n_before}"
            )

        idx_robot = omap.official_grid_indices(
            sample["robot_flows"][0],
            grid_size=grid_size,
            mode=mode,
        )

        robot_labels_after = pre[
            idx_robot
        ].copy()

    # Actual PointWorld downsampling remains untouched.
    result = omap.grid_sample_with_object_ids(
        sample,
        grid_size=grid_size,
        mode=mode,
    )

    if robot_labels_after is not None:
        n_after = int(
            result["robot_flows"].shape[1]
        )

        if len(robot_labels_after) != n_after:
            raise RuntimeError(
                "POST-grid robot-label mismatch: "
                f"{len(robot_labels_after)} vs {n_after}"
            )

        TARGET_ROBOT_LINK_LABELS = (
            robot_labels_after
        )

        print(
            "[ROBOT_LINK_GRID] "
            f"before={n_before} "
            f"after={n_after}",
            flush=True,
        )

    return result


omap.pmod.grid_sample_transform = (
    grid_sample_with_object_ids_and_robot_links
)

omap.pmod.gather = (
    omap.gather_with_object_ids
)

def corrected_build(
    sample,
    domain,
    robot_sampler,
    max_robot_points,
    deterministic=False,
    seed=None,
    force_single_arm=False,
):

    out = OBJECT_BUILD(
        sample,
        domain=domain,
        robot_sampler=robot_sampler,
        max_robot_points=max_robot_points,
        deterministic=deterministic,
        seed=seed,
        force_single_arm=force_single_arm,
    )

    key = normalize_key(
        out.get("__key__", "")
    )

    # --------------------------------------------------------
    # Diagnostic-only robot point -> URDF link sidecar.
    # Does NOT modify PointWorld/model inputs.
    # --------------------------------------------------------
    if key == TARGET:
        global TARGET_ROBOT_LINK_LABELS_PREGRID

        link_labels = []

        for mesh_name, pts in robot_sampler._presampled_points.items():
            n_pts = int(pts.shape[0])
            link_name = str(
                robot_sampler._mesh_to_link.get(
                    mesh_name,
                    "",
                )
            )
            link_labels.extend(
                [link_name] * n_pts
            )

        link_labels = np.asarray(
            link_labels,
            dtype=object,
        )

        if len(link_labels) != out["robot_flows"].shape[1]:
            raise RuntimeError(
                "Robot-link sidecar mismatch: "
                f"{len(link_labels)} vs "
                f"{out['robot_flows'].shape[1]}"
            )

        TARGET_ROBOT_LINK_LABELS_PREGRID = link_labels

    if key != TARGET:
        return out

    original = out[
        "robot_flows"
    ].copy()

    out["robot_flows"] = resample(
        out["robot_flows"],
        PHASE,
    )

    # start/end must be exactly preserved
    assert np.allclose(
        original[0],
        out["robot_flows"][0],
        atol=1e-7,
        rtol=0,
    )

    assert np.allclose(
        original[-1],
        out["robot_flows"][-1],
        atol=1e-7,
        rtol=0,
    )

    if "robot_normals" in out:
        out["robot_normals"] = resample(
            out["robot_normals"],
            PHASE,
            normalize=True,
        )

    # Binary action events:
    # zero-order hold under the optimized phase map.
    for name in [
        "right_gripper_open",
        "left_gripper_open",
    ]:

        if name not in out:
            continue

        a = np.asarray(
            out[
                name
            ]
        )

        if (
            a.ndim >= 1
            and
            a.shape[0] == len(PHASE)
        ):

            p = np.clip(
                np.asarray(
                    PHASE,
                    dtype=np.float64,
                ),
                0.0,
                float(
                    len(PHASE) - 1
                ),
            )

            # State changes when optimized phase reaches the
            # corresponding original action sample.
            idx = np.floor(
                p + 1e-12
            ).astype(
                np.int64
            )

            mapped = a[
                idx
            ].copy()

            mapped[
                0
            ] = a[
                0
            ]

            mapped[
                -1
            ] = a[
                -1
            ]

            out[
                name
            ] = mapped

    # Continuous gripper quantity remains interpolated.
    if "gripper_positions" in out:

        a = np.asarray(
            out[
                "gripper_positions"
            ]
        )

        if (
            a.ndim >= 1
            and
            a.shape[0] == len(PHASE)
        ):
            out[
                "gripper_positions"
            ] = resample(
                a,
                PHASE,
            )

    return out


# ============================================================
# FAST TARGET-ONLY RAW WDS FILTER
#
# Filter by raw WebDataset __key__ BEFORE decode_data /
# build_flow_sample / camera sampling / transforms.
# Process-local monkeypatch only; official source is untouched.
# ============================================================

import dataset_components.dataloader as _pw_dl

_ORIGINAL_WEBDATASET = _pw_dl.wds.WebDataset

def _target_only_webdataset(*args, **kwargs):

    ds = _ORIGINAL_WEBDATASET(
        *args,
        **kwargs,
    )

    def _keep_target_raw(sample):

        key = sample.get(
            "__key__",
            "",
        )

        if isinstance(key, bytes):
            key = key.decode(
                "utf-8",
                errors="replace",
            )

        return (
            normalize_key(key)
            == TARGET
        )

    return ds.select(
        _keep_target_raw
    )

_pw_dl.wds.WebDataset = (
    _target_only_webdataset
)

print(
    "[FAST_TARGET] raw WDS filter installed:",
    TARGET,
)

# ============================================================
# PointWorld
# ============================================================

pw = parse_args(
    skip_command_line=True
)

pw.model_path = MODEL_PATH
pw.norm_stats_path = NORM_STATS

pw.domains = [
    "behavior"
]

pw.data_dirs = [
    TASK_DIRS[TASK]
]

pw.batch_size = 1
pw.num_workers = 0
pw.eval_num_workers = 0
pw.eval_num_batches = -1
pw.device = "cuda"
pw.eval_skip_viz = True

tester = Tester(pw)

tester.model.eval()

# ============================================================
# Original loader
# ============================================================

dlmod.build_flow_sample = OBJECT_BUILD

reset_seed(42)

base_loader, _ = (
    tester._build_eval_loader(
        "test",
        enable_mask=False,
    )
)

# ============================================================
# Corrected loader
# ============================================================

dlmod.build_flow_sample = corrected_build

reset_seed(42)

corr_loader, _ = (
    tester._build_eval_loader(
        "test",
        enable_mask=False,
    )
)

found = False

for b0, b1 in zip(
    base_loader,
    corr_loader,
):

    k0 = normalize_key(
        b0["__key__"]
    )

    k1 = normalize_key(
        b1["__key__"]
    )

    if k0 != k1:
        raise RuntimeError(
            f"loader mismatch: {k0} != {k1}"
        )

    if k0 != TARGET:
        continue

    found = True

    if not torch.equal(
        b0["scene_object_id"],
        b1["scene_object_id"],
    ):
        raise RuntimeError(
            "Object IDs changed."
        )

    g0 = to_device(
        b0,
        tester.device,
    )

    g1 = to_device(
        b1,
        tester.device,
    )

    # Same stochastic seed
    reset_seed(12345)

    with torch.no_grad():
        pred_original = (
            tester.model(
                g0,
                training=False,
            )["scene_flows"][0]
            .detach()
            .cpu()
            .numpy()
        )

    reset_seed(12345)

    with torch.no_grad():
        pred_corrected = (
            tester.model(
                g1,
                training=False,
            )["scene_flows"][0]
            .detach()
            .cpu()
            .numpy()
        )

    object_id = (
        b0["scene_object_id"][0]
        .detach()
        .cpu()
        .numpy()
    )

    context = mask3(
        b0,
        "scene_context_mask",
    )

    exists = mask3(
        b0,
        "scene_exists",
    )

    supervised = mask3(
        b0,
        "scene_supervised_mask",
    )

    moved = mask3(
        b0,
        "scene_moved_mask",
    )

    valid = (
        (~context)
        & exists
        & supervised
    )

    # ========================================================
    # SELF-RELEASE RIGID-SUPPORT V9.4
    # ========================================================

    from grasp_release_rigid_support_antidetach_v9_4 import (
        apply_rigid_support_antidetach_repair_v9_4,
    )

    from datetime import datetime
    from pathlib import Path
    import pandas as pd

    V6_TAG = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    V6_OUT = Path(
        "diagnostics/results/"
        "grasp_release_rigid_support_antidetach_v9_4_visuals"
    )

    V6_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pred_repaired, v4_diag = (
        apply_rigid_support_antidetach_repair_v9_4(
            pred_corrected,
            b1["robot_flows"],
            valid,

            right_gripper_open=(
                b1["right_gripper_open"]
                if "right_gripper_open" in b1
                else None
            ),

            left_gripper_open=(
                b1["left_gripper_open"]
                if "left_gripper_open" in b1
                else None
            ),

            right_gripper_pose=(
                b1["right_gripper_pose"]
                if "right_gripper_pose" in b1
                else None
            ),

            left_gripper_pose=(
                b1["left_gripper_pose"]
                if "left_gripper_pose" in b1
                else None
            ),

            sigma_geom_m=0.015,
            sigma_motion_m=0.015,
        )
    )

    print()
    print(
        "========== SELF-RELEASE RIGID-SUPPORT V9.4 =========="
    )

    print(
        "robot points by side =",
        v4_diag[
            "side_robot_point_counts"
        ],
    )

    for mi, mem in enumerate(
        v4_diag["memories"]
    ):
        print(
            f"memory[{mi}]"
            f" side={mem['side']}"
            f" ref={mem['ref_t']}"
            f" evidence={mem['evidence_t']}"
            f" end={mem['end_t']}"
            f" geom_sum={mem['geom_weight_sum']:.6f}"
            f" joint_sum={mem['joint_weight_sum']:.6f}"
            f" effective_support={mem['effective_support']:.3f}"
            f" max_weight={mem['max_weight']:.6f}"
            f" motion_med={mem['median_motion_error_mm']:.3f}mm"
        )

    frame_df_v4 = pd.DataFrame(
        v4_diag["frame_rows"]
    )

    print()
    print(
        frame_df_v4.to_string(
            index=False
        )
    )

    safe_target_v4 = (
        str(TARGET)
        .replace("/", "_")
        .replace(":", "__")
    )

    csv_v4 = (
        V6_OUT
        / (
            f"{safe_target_v4}"
            "__adaptive_contact_v6__"
            f"{V6_TAG}"
            "__frames.csv"
        )
    )

    frame_df_v4.to_csv(
        csv_v4,
        index=False,
    )

    print(
        "V6 FRAME CSV =",
        csv_v4,
    )

    # ========================================================
    # V6 SCENE-CANDIDATE / HAND-OBJECT GIF
    #
    # Visualization only:
    # - object ID is NOT used by the repair method
    # - target point indices are fixed once at t=0
    # - do NOT require moved[t] at every frame
    # - same camera / xyz limits for all panels and frames
    # ========================================================

    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from PIL import Image
    from pathlib import Path

    def _gif_np(x):
        if torch.is_tensor(x):
            x = (
                x.detach()
                .cpu()
                .numpy()
            )
        else:
            x = np.asarray(x)

        # remove batch dimension only
        if (
            x.ndim >= 3
            and
            x.shape[0] == 1
        ):
            x = x[0]

        return np.asarray(x)

    po = _gif_np(
        pred_original
    )
    pc = _gif_np(
        pred_corrected
    )
    pr = _gif_np(
        pred_repaired
    )

    ro = _gif_np(
        b0["robot_flows"]
    )
    rc = _gif_np(
        b1["robot_flows"]
    )

    gt_all = _gif_np(
        b0["gt_scene_flows"]
    )

    # --------------------------------------------------------
    # Fixed target object index.
    #
    # Do NOT use moved[t] here.
    # Point correspondence is fixed across rollout.
    # --------------------------------------------------------

    if args_local.object_id is None:

        counts = []

        for oid0 in sorted(
            np.unique(
                object_id[0]
            ).tolist()
        ):
            oid0 = int(oid0)

            if oid0 <= 0:
                continue

            mm = (
                (object_id[0] == oid0)
                & exists[0]
            )

            n0 = int(
                mm.sum()
            )

            if n0 >= 8:
                counts.append(
                    (n0, oid0)
                )

        if not counts:
            raise RuntimeError(
                "No >=8-point object exists at t=0."
            )

        counts.sort(
            reverse=True
        )

        oid = counts[0][1]

    else:
        oid = int(
            args_local.object_id
        )

    obj_idx = np.where(
        (object_id[0] == oid)
        & exists[0]
    )[0]

    if len(obj_idx) < 8:
        raise RuntimeError(
            f"object_id={oid}: only "
            f"{len(obj_idx)} t0 points"
        )

    print()
    print(
        "GIF TARGET OBJECT ID =",
        oid,
        "FIXED POINTS =",
        len(obj_idx),
    )

    # ========================================================
    # V9.4 GOAL MPJPE + BILATERAL FINGER GRASP AUDIT
    # ========================================================

    t_goal_audit = min(
        10,
        po.shape[0] - 1,
        pc.shape[0] - 1,
        pr.shape[0] - 1,
        gt_all.shape[0] - 1,
    )

    goal_mask = (
        (object_id[t_goal_audit] == oid)
        & valid[t_goal_audit]
        & (object_id[0] == oid)
        & exists[0]
    )

    n_goal = int(goal_mask.sum())

    if n_goal < 8:
        raise RuntimeError(
            f"Goal audit has only {n_goal} points"
        )

    P_original = po[
        t_goal_audit,
        goal_mask,
    ]
    P_boundary = pc[
        t_goal_audit,
        goal_mask,
    ]
    P_v93 = pr[
        t_goal_audit,
        goal_mask,
    ]

    goal_boundary_mm = float(
        np.linalg.norm(
            P_boundary - P_original,
            axis=1,
        ).mean()
        * 1000.0
    )

    goal_v93_mm = float(
        np.linalg.norm(
            P_v93 - P_original,
            axis=1,
        ).mean()
        * 1000.0
    )

    repair_from_boundary_mm = float(
        np.linalg.norm(
            P_v93 - P_boundary,
            axis=1,
        ).mean()
        * 1000.0
    )

    print()
    print("=" * 72)
    print("TARGET GOAL MPJPE AUDIT")
    print("=" * 72)
    print(
        f"[GOAL_MPJPE] frame={t_goal_audit} "
        f"points={n_goal}"
    )
    print(
        "[GOAL_MPJPE] "
        f"Boundary-only vs Original prediction = "
        f"{goal_boundary_mm:.6f} mm"
    )
    print(
        "[GOAL_MPJPE] "
        f"Boundary+V9.4 vs Original prediction = "
        f"{goal_v93_mm:.6f} mm"
    )
    print(
        "[GOAL_MPJPE] "
        f"V9.4 repair shift from Boundary = "
        f"{repair_from_boundary_mm:.6f} mm"
    )

    # --------------------------------------------------------
    # Bilateral finger diagnostic.
    #
    # Original GT + Original robot:
    #   strongest available evidence for whether the dataset
    #   interaction is actually a bilateral grasp.
    #
    # Counterfactual Boundary/V9.4 has no future GT.
    # --------------------------------------------------------

    if TARGET_ROBOT_LINK_LABELS is None:
        raise RuntimeError(
            "TARGET_ROBOT_LINK_LABELS was not captured"
        )

    robot_link_labels = np.asarray(
        TARGET_ROBOT_LINK_LABELS,
        dtype=str,
    )

    if not (
        len(robot_link_labels)
        == ro.shape[1]
        == rc.shape[1]
    ):
        raise RuntimeError(
            "Robot link labels do not align with "
            "original/corrected robot points"
        )

    finger_links = sorted(
        set(
            x
            for x in robot_link_labels.tolist()
            if "finger" in x.lower()
        )
    )

    print()
    print("=" * 72)
    print("URDF FINGER LINKS IN CURRENT ROBOT FLOW")
    print("=" * 72)

    for x in finger_links:
        print("[FINGER_LINK]", x)

    # Adaptive contact scale = median nearest-neighbor spacing
    # of the target rigid object at GT t0.
    from scipy.spatial import cKDTree as _AuditKDTree

    x0_audit = gt_all[
        0,
        obj_idx,
    ]

    _tree = _AuditKDTree(
        x0_audit
    )

    _d, _ = _tree.query(
        x0_audit,
        k=2,
    )

    local_scale_m = float(
        np.median(
            _d[:, 1]
        )
    )

    print(
        f"[GRASP_SCALE] object median NN = "
        f"{local_scale_m * 1000.0:.3f} mm"
    )

    def _pair_gap(obj, finger):
        if len(obj) == 0 or len(finger) == 0:
            return float("nan")

        d = np.linalg.norm(
            obj[:, None, :]
            - finger[None, :, :],
            axis=2,
        )

        return float(
            np.min(d)
        )

    def _between_fingers(
        obj,
        finger1,
        finger2,
    ):
        if (
            len(obj) == 0
            or len(finger1) == 0
            or len(finger2) == 0
        ):
            return False

        c1 = finger1.mean(
            axis=0
        )
        c2 = finger2.mean(
            axis=0
        )
        co = obj.mean(
            axis=0
        )

        axis = c2 - c1
        den = float(
            np.dot(
                axis,
                axis,
            )
        )

        if den < 1e-12:
            return False

        alpha = float(
            np.dot(
                co - c1,
                axis,
            )
            / den
        )

        return (
            0.0
            <= alpha
            <= 1.0
        )

    def _variant_grasp(
        obj,
        robot,
        m1,
        m2,
        closed,
    ):
        f1 = robot[m1]
        f2 = robot[m2]

        g1 = _pair_gap(
            obj,
            f1,
        )
        g2 = _pair_gap(
            obj,
            f2,
        )

        between = _between_fingers(
            obj,
            f1,
            f2,
        )

        bilateral = bool(
            closed
            and
            between
            and
            np.isfinite(g1)
            and
            np.isfinite(g2)
            and
            g1 <= local_scale_m
            and
            g2 <= local_scale_m
        )

        return (
            g1 * 1000.0,
            g2 * 1000.0,
            between,
            bilateral,
        )

    print()
    print("=" * 72)
    print("BILATERAL GRASP AUDIT")
    print(
        "criterion: CLOSED + object centroid between fingers "
        "+ both finger gaps <= object median-NN scale"
    )
    print("=" * 72)

    lower_labels = np.char.lower(
        robot_link_labels
    )

    for side in [
        "right",
        "left",
    ]:

        # R1Pro URDF naming convention:
        # *_gripper_finger_link1 / *_gripper_finger_link2
        m1 = np.array([
            (
                side in x
                and
                "finger" in x
                and
                (
                    "link1" in x
                    or
                    "finger1" in x
                )
            )
            for x in lower_labels
        ])

        m2 = np.array([
            (
                side in x
                and
                "finger" in x
                and
                (
                    "link2" in x
                    or
                    "finger2" in x
                )
            )
            for x in lower_labels
        ])

        if (
            not np.any(m1)
            or
            not np.any(m2)
        ):
            print(
                f"[BILATERAL_WARN] side={side} "
                f"finger1_points={int(m1.sum())} "
                f"finger2_points={int(m2.sum())}"
            )
            continue

        print(
            f"[BILATERAL_SIDE] side={side} "
            f"finger1_points={int(m1.sum())} "
            f"finger2_points={int(m2.sum())}"
        )

        orig_open = _gif_np(
            b0[
                f"{side}_gripper_open"
            ]
        ).reshape(-1)

        corr_open = _gif_np(
            b1[
                f"{side}_gripper_open"
            ]
        ).reshape(-1)

        for tt in range(
            1,
            t_goal_audit + 1,
        ):

            obj_valid_idx = obj_idx[
                valid[
                    tt,
                    obj_idx,
                ]
            ]

            if len(
                obj_valid_idx
            ) < 1:
                continue

            gt_obj = gt_all[
                tt,
                obj_valid_idx,
            ]

            original_obj = po[
                tt,
                obj_valid_idx,
            ]

            boundary_obj = pc[
                tt,
                obj_valid_idx,
            ]

            v93_obj = pr[
                tt,
                obj_valid_idx,
            ]

            closed_original = not bool(
                orig_open[
                    tt
                ]
            )

            closed_corrected = not bool(
                corr_open[
                    tt
                ]
            )

            gt_audit = _variant_grasp(
                gt_obj,
                ro[tt],
                m1,
                m2,
                closed_original,
            )

            original_audit = _variant_grasp(
                original_obj,
                ro[tt],
                m1,
                m2,
                closed_original,
            )

            boundary_audit = _variant_grasp(
                boundary_obj,
                rc[tt],
                m1,
                m2,
                closed_corrected,
            )

            v93_audit = _variant_grasp(
                v93_obj,
                rc[tt],
                m1,
                m2,
                closed_corrected,
            )

            def _fmt(x):
                g1, g2, bt, bi = x

                return (
                    f"g1={g1:.2f},"
                    f"g2={g2:.2f},"
                    f"between={int(bt)},"
                    f"bilateral={int(bi)}"
                )

            print(
                f"[BILATERAL] "
                f"frame={tt} side={side} "
                f"closed_orig={int(closed_original)} "
                f"closed_corr={int(closed_corrected)} | "
                f"GT={_fmt(gt_audit)} | "
                f"OriginalPred={_fmt(original_audit)} | "
                f"Boundary={_fmt(boundary_audit)} | "
                f"V9.4={_fmt(v93_audit)}"
            )

    # --------------------------------------------------------
    # Frames 1 -> 10
    # --------------------------------------------------------

    Tmax = min(
        10,
        po.shape[0] - 1,
        pc.shape[0] - 1,
        pr.shape[0] - 1,
        ro.shape[0] - 1,
        rc.shape[0] - 1,
    )

    frames_v6gif = list(
        range(
            1,
            Tmax + 1,
        )
    )

    if len(frames_v6gif) != 10:
        print(
            "WARNING: expected 10 future frames,"
            " got",
            frames_v6gif,
        )

    # --------------------------------------------------------
    # Split sampled gripper surface points into R/L.
    #
    # Visualization follows the same geometric convention:
    # assign t0 robot samples to nearest gripper center.
    # --------------------------------------------------------

    rp = (
        _gif_np(
            b1["right_gripper_pose"]
        )
        if "right_gripper_pose" in b1
        else None
    )

    lp = (
        _gif_np(
            b1["left_gripper_pose"]
        )
        if "left_gripper_pose" in b1
        else None
    )

    nr = rc.shape[1]

    if (
        rp is not None
        and
        lp is not None
    ):

        cr = np.asarray(
            rp[0, :3],
            dtype=np.float64,
        )

        cl = np.asarray(
            lp[0, :3],
            dtype=np.float64,
        )

        dr = np.linalg.norm(
            rc[0] - cr[None, :],
            axis=1,
        )

        dl = np.linalg.norm(
            rc[0] - cl[None, :],
            axis=1,
        )

        right_robot_mask = (
            dr <= dl
        )

        left_robot_mask = (
            ~right_robot_mask
        )

    else:

        right_robot_mask = np.ones(
            nr,
            dtype=bool,
        )

        left_robot_mask = np.zeros(
            nr,
            dtype=bool,
        )

    print(
        "GIF robot points:",
        "right=",
        int(
            right_robot_mask.sum()
        ),
        "left=",
        int(
            left_robot_mask.sum()
        ),
    )

    # --------------------------------------------------------
    # Fixed local scene candidates.
    #
    # Use initial scene positions only to choose visual context.
    # This is VISUALIZATION ONLY.
    # --------------------------------------------------------

    initial_xyz = np.asarray(
        gt_all[0],
        dtype=np.float64,
    )

    # Visualization-only initial world context.
    # t=0 is the PointWorld context frame, so do NOT apply ~context[0].
    # Target-object points are removed below.
    scene_initial_mask = (
        exists[0]
    )

    scene_initial_idx = np.where(
        scene_initial_mask
    )[0]

    # Don't draw the target twice.
    target_set = set(
        obj_idx.tolist()
    )

    scene_initial_idx = np.asarray(
        [
            i
            for i in scene_initial_idx
            if int(i) not in target_set
        ],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Focus geometry over ALL frames:
    # object + hands, across all three predictions.
    # Gives fixed axes for the complete GIF.
    # --------------------------------------------------------

    focus_parts = []

    for tt in frames_v6gif:

        focus_parts.extend(
            [
                po[tt, obj_idx],
                pc[tt, obj_idx],
                pr[tt, obj_idx],
                ro[tt],
                rc[tt],
            ]
        )

    focus = np.concatenate(
        focus_parts,
        axis=0,
    )

    focus = focus[
        np.all(
            np.isfinite(focus),
            axis=1,
        )
    ]

    if len(focus) == 0:
        raise RuntimeError(
            "No finite focus points."
        )

    lo = focus.min(
        axis=0
    )

    hi = focus.max(
        axis=0
    )

    span_xyz = (
        hi - lo
    )

    span = float(
        np.max(
            span_xyz
        )
    )

    if span <= 0:
        span = 1e-3

    # scene view: slightly wider
    scene_margin = (
        0.35 * span
    )

    scene_lo = (
        lo - scene_margin
    )

    scene_hi = (
        hi + scene_margin
    )

    # zoom view: object/hand focused
    zoom_margin = (
        0.12 * span
    )

    zoom_lo = (
        lo - zoom_margin
    )

    zoom_hi = (
        hi + zoom_margin
    )

    # ========================================================
    # STATIC LOCAL WORLD CONTEXT
    #
    # Visualization only.
    #
    # Use t=0 scene points as a fixed world reference so that
    # apparent floating / detachment can actually be judged.
    #
    # Select the nearest local scene points to the target object's
    # initial position.  No object label enters the V6 method.
    # ========================================================

    target0 = np.asarray(
        initial_xyz[obj_idx],
        dtype=np.float64,
    )

    target0 = target0[
        np.all(
            np.isfinite(target0),
            axis=1,
        )
    ]

    if len(target0) == 0:
        raise RuntimeError(
            "No finite target points at t=0."
        )

    target0_center = (
        target0.mean(axis=0)
    )

    s0 = np.asarray(
        initial_xyz[
            scene_initial_idx
        ],
        dtype=np.float64,
    )

    good_s0 = np.all(
        np.isfinite(s0),
        axis=1,
    )

    candidate_idx = (
        scene_initial_idx[
            good_s0
        ]
    )

    candidate_xyz = (
        s0[
            good_s0
        ]
    )

    if len(candidate_xyz) == 0:
        raise RuntimeError(
            "No finite static scene context points."
        )

    dist_to_target0 = np.linalg.norm(
        candidate_xyz
        - target0_center[None, :],
        axis=1,
    )

    # Visualization-only adaptive count:
    # enough context to show nearby table / support geometry
    # without making the manipulated object microscopic.
    K_scene = min(
        len(candidate_xyz),
        max(
            500,
            3 * len(obj_idx),
        ),
    )

    order = np.argsort(
        dist_to_target0
    )[:K_scene]

    scene_idx = (
        candidate_idx[
            order
        ]
    )

    static_scene_xyz = np.asarray(
        initial_xyz[
            scene_idx
        ],
        dtype=np.float64,
    )

    print(
        "STATIC SCENE CONTEXT POINTS =",
        len(scene_idx),
    )

    # --------------------------------------------------------
    # Scene-view limits contain:
    #   static t0 context
    #   all future object predictions
    #   both hands
    #
    # Zoom view remains object/hand focused.
    # --------------------------------------------------------

    scene_focus = np.concatenate(
        [
            focus,
            static_scene_xyz,
        ],
        axis=0,
    )

    scene_focus = scene_focus[
        np.all(
            np.isfinite(scene_focus),
            axis=1,
        )
    ]

    scene_lo = (
        scene_focus.min(axis=0)
    )

    scene_hi = (
        scene_focus.max(axis=0)
    )

    scene_span = float(
        np.max(
            scene_hi - scene_lo
        )
    )

    if scene_span <= 0:
        scene_span = 1e-3

    scene_margin = (
        0.08 * scene_span
    )

    scene_lo = (
        scene_lo - scene_margin
    )

    scene_hi = (
        scene_hi + scene_margin
    )

    # --------------------------------------------------------
    # Output folder -- unique version, no overwrite.
    # --------------------------------------------------------

    gif_dir = (
        V6_OUT
        / (
            f"{safe_target_v4}"
            f"__oid{oid}"
            "__scene_context_v9_4__"
            f"{V6_TAG}"
        )
    )

    gif_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    scene_frame_dir = (
        gif_dir
        / "scene_frames"
    )

    zoom_frame_dir = (
        gif_dir
        / "zoom_frames"
    )

    scene_frame_dir.mkdir()
    zoom_frame_dir.mkdir()

    # Save the exact V6 frame diagnostic used by graph.
    metrics_csv = (
        gif_dir
        / "frame_metrics.csv"
    )

    frame_df_v4.to_csv(
        metrics_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Axis helper
    # --------------------------------------------------------

    def _set_equal_fixed(
        ax,
        lim_lo,
        lim_hi,
    ):

        center = (
            lim_lo + lim_hi
        ) / 2.0

        side = float(
            np.max(
                lim_hi - lim_lo
            )
        )

        if side <= 0:
            side = 1e-3

        half = (
            side / 2.0
        )

        ax.set_xlim(
            center[0] - half,
            center[0] + half,
        )

        ax.set_ylim(
            center[1] - half,
            center[1] + half,
        )

        ax.set_zlim(
            center[2] - half,
            center[2] + half,
        )

        ax.set_box_aspect(
            (1, 1, 1)
        )

        # fixed camera
        ax.view_init(
            elev=24,
            azim=-62,
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.grid(False)

    # --------------------------------------------------------
    # separation lookup from the exact V6 diagnostic
    # --------------------------------------------------------

    def _sep_at(
        tt,
        col,
    ):

        if (
            "frame"
            not in frame_df_v4.columns
            or
            col
            not in frame_df_v4.columns
        ):
            return np.nan

        z = frame_df_v4[
            frame_df_v4["frame"]
            == tt
        ]

        if len(z) != 1:
            return np.nan

        return float(
            z.iloc[0][col]
        )

    # --------------------------------------------------------
    # Actual nearest hand-object gap for VISUALIZATION.
    #
    # This is different from "separation excess":
    #   - min_gap_mm = actual nearest Euclidean gap now
    #   - separation excess = increase relative to contact reference
    # --------------------------------------------------------

    def _nearest_pair(
        obj,
        robot,
    ):

        if (
            len(obj) == 0
            or
            len(robot) == 0
        ):
            return (
                np.nan,
                None,
                None,
            )

        d = (
            obj[:, None, :]
            - robot[None, :, :]
        )

        d2 = np.sum(
            d * d,
            axis=2,
        )

        flat = int(
            np.argmin(d2)
        )

        oi, ri = np.unravel_index(
            flat,
            d2.shape,
        )

        dist_mm = float(
            np.sqrt(
                d2[oi, ri]
            )
            * 1000.0
        )

        return (
            dist_mm,
            obj[oi],
            robot[ri],
        )

    # --------------------------------------------------------
    # One 3-panel frame
    # --------------------------------------------------------

    def _render_triptych(
        tt,
        out_png,
        include_scene,
    ):

        fig = plt.figure(
            figsize=(15.5, 5.4)
        )

        before_sep = _sep_at(
            tt,
            "separation_excess_before_mm",
        )

        after_sep = _sep_at(
            tt,
            "separation_excess_after_mm",
        )

        panels = [
            (
                "Original Action",
                po,
                ro,
                np.nan,
            ),
            (
                "Boundary-only",
                pc,
                rc,
                before_sep,
            ),
            (
                "Boundary + V9.4",
                pr,
                rc,
                after_sep,
            ),
        ]

        for jj, (
            title,
            pred_xyz,
            robot_xyz,
            sep_val,
        ) in enumerate(
            panels,
            start=1,
        ):

            ax = fig.add_subplot(
                1,
                3,
                jj,
                projection="3d",
            )

            # faint local scene
            if include_scene:

                # Fixed t=0 world reference.
                q = static_scene_xyz

                good = np.all(
                    np.isfinite(q),
                    axis=1,
                )

                q = q[good]

                ax.scatter(
                    q[:, 0],
                    q[:, 1],
                    q[:, 2],
                    s=2.5,
                    c="#9fb6cc",
                    alpha=0.28,
                    linewidths=0,
                    depthshade=False,
                    label="static t0 scene",
                )

            # target object
            obj = pred_xyz[
                tt,
                obj_idx,
            ]

            good_obj = np.all(
                np.isfinite(obj),
                axis=1,
            )

            obj = obj[
                good_obj
            ]

            ax.scatter(
                obj[:, 0],
                obj[:, 1],
                obj[:, 2],
                s=15,
                c="#d62728",
                alpha=0.95,
                linewidths=0,
                depthshade=False,
                label=f"target object (id={oid})",
            )

            # right hand
            rr = robot_xyz[
                tt,
                right_robot_mask,
            ]

            good_rr = np.all(
                np.isfinite(rr),
                axis=1,
            )

            rr = rr[
                good_rr
            ]

            if len(rr):

                ax.scatter(
                    rr[:, 0],
                    rr[:, 1],
                    rr[:, 2],
                    s=10,
                    c="#ff8c00",
                    alpha=0.9,
                    linewidths=0,
                    depthshade=False,
                    label="right gripper",
                )

            # left hand
            ll = robot_xyz[
                tt,
                left_robot_mask,
            ]

            good_ll = np.all(
                np.isfinite(ll),
                axis=1,
            )

            ll = ll[
                good_ll
            ]

            if len(ll):

                ax.scatter(
                    ll[:, 0],
                    ll[:, 1],
                    ll[:, 2],
                    s=10,
                    c="#2ca02c",
                    alpha=0.9,
                    linewidths=0,
                    depthshade=False,
                    label="left gripper",
                )

            # --------------------------------------------
            # Actual nearest hand-object pair
            # --------------------------------------------

            robot_parts = []

            if len(rr):
                robot_parts.append(
                    rr
                )

            if len(ll):
                robot_parts.append(
                    ll
                )

            if robot_parts:
                robot_vis = np.concatenate(
                    robot_parts,
                    axis=0,
                )
            else:
                robot_vis = np.empty(
                    (0, 3),
                    dtype=np.float64,
                )

            (
                min_gap_mm,
                obj_near,
                hand_near,
            ) = _nearest_pair(
                obj,
                robot_vis,
            )

            print(
                f"[ACTUAL_GAP] frame={tt} variant={title} "
                f"min_gap_mm={min_gap_mm:.6f}",
                flush=True,
            )

            if (
                obj_near is not None
                and
                hand_near is not None
            ):

                ax.plot(
                    [
                        obj_near[0],
                        hand_near[0],
                    ],
                    [
                        obj_near[1],
                        hand_near[1],
                    ],
                    [
                        obj_near[2],
                        hand_near[2],
                    ],
                    linewidth=2.2,
                    c="black",
                    alpha=0.85,
                )

                ax.scatter(
                    [
                        obj_near[0],
                        hand_near[0],
                    ],
                    [
                        obj_near[1],
                        hand_near[1],
                    ],
                    [
                        obj_near[2],
                        hand_near[2],
                    ],
                    s=28,
                    c="black",
                    depthshade=False,
                )

            ttl = (
                f"{title}\n"
                f"frame {tt}/10"
            )

            if np.isfinite(
                min_gap_mm
            ):
                ttl += (
                    "\n"
                    f"actual min gap = "
                    f"{min_gap_mm:.2f} mm"
                )

            if np.isfinite(
                sep_val
            ):
                ttl += (
                    " | "
                    f"excess = "
                    f"{sep_val:.2f} mm"
                )

            ax.set_title(
                ttl,
                fontsize=10,
            )

            if include_scene:
                _set_equal_fixed(
                    ax,
                    scene_lo,
                    scene_hi,
                )
            else:
                _set_equal_fixed(
                    ax,
                    zoom_lo,
                    zoom_hi,
                )

            if jj == 1:
                ax.legend(
                    loc="upper left",
                    fontsize=7,
                    frameon=True,
                )

        fig.suptitle(
            (
                f"{safe_target_v4} | "
                f"object id={oid} | "
                "fixed point correspondence"
            ),
            fontsize=12,
        )

        fig.tight_layout(
            rect=[
                0,
                0,
                1,
                0.94,
            ]
        )

        fig.savefig(
            out_png,
            dpi=155,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    # --------------------------------------------------------
    # Render all 10 frames.
    # --------------------------------------------------------

    scene_pngs = []
    zoom_pngs = []

    for tt in frames_v6gif:

        scene_png = (
            scene_frame_dir
            / f"frame_{tt:02d}.png"
        )

        zoom_png = (
            zoom_frame_dir
            / f"frame_{tt:02d}.png"
        )

        print(
            f"[GIF] rendering frame {tt}/10"
        )

        _render_triptych(
            tt,
            scene_png,
            include_scene=True,
        )

        _render_triptych(
            tt,
            zoom_png,
            include_scene=False,
        )

        scene_pngs.append(
            scene_png
        )

        zoom_pngs.append(
            zoom_png
        )

    # --------------------------------------------------------
    # GIF writer
    # --------------------------------------------------------

    def _write_gif(
        pngs,
        out_gif,
    ):

        imgs = [
            Image.open(p)
            .convert("RGB")
            for p in pngs
        ]

        if not imgs:
            raise RuntimeError(
                "No GIF frames."
            )

        # hold final frame a little longer
        imgs2 = (
            imgs
            + [
                imgs[-1],
                imgs[-1],
            ]
        )

        imgs2[0].save(
            out_gif,
            save_all=True,
            append_images=imgs2[1:],
            duration=450,
            loop=0,
        )

        for im in imgs:
            im.close()

    scene_gif = (
        gif_dir
        / "scene_context_original_boundary_v9_4.gif"
    )

    zoom_gif = (
        gif_dir
        / "zoom_hand_object_original_boundary_v9_4.gif"
    )

    _write_gif(
        scene_pngs,
        scene_gif,
    )

    _write_gif(
        zoom_pngs,
        zoom_gif,
    )

    # --------------------------------------------------------
    # Separate graph:
    # exact V6 separation diagnostic vs frame.
    # --------------------------------------------------------

    graph_png = (
        gif_dir
        / "separation_vs_frame.png"
    )

    if (
        "frame"
        in frame_df_v4.columns
        and
        "separation_excess_before_mm"
        in frame_df_v4.columns
        and
        "separation_excess_after_mm"
        in frame_df_v4.columns
    ):

        gd = frame_df_v4[
            frame_df_v4["frame"]
            .isin(
                frames_v6gif
            )
        ].copy()

        fig, ax = plt.subplots(
            figsize=(7.2, 4.8)
        )

        ax.plot(
            gd["frame"],
            gd[
                "separation_excess_before_mm"
            ],
            marker="o",
            linewidth=2,
            label="Boundary-only",
        )

        ax.plot(
            gd["frame"],
            gd[
                "separation_excess_after_mm"
            ],
            marker="o",
            linewidth=2,
            label="Boundary + V9.4",
        )

        ax.set_xlabel(
            "Rollout frame"
        )

        ax.set_ylabel(
            "Hand-object separation excess (mm)"
        )

        ax.set_title(
            f"{safe_target_v4} | object id={oid}"
        )

        ax.set_xticks(
            frames_v6gif
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            graph_png,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    print()
    print(
        "==========================================="
    )
    print(
        "V9.4 GIF VISUALIZATION COMPLETE"
    )
    print(
        "==========================================="
    )
    print(
        "OUTPUT DIR =",
        gif_dir,
    )
    print(
        "SCENE GIF  =",
        scene_gif,
    )
    print(
        "ZOOM GIF   =",
        zoom_gif,
    )
    print(
        "GRAPH      =",
        graph_png,
    )
    print(
        "METRICS    =",
        metrics_csv,
    )

    # Stop before the legacy per-frame PA visualization.
    raise SystemExit(0)

    # ========================================================
    # PROCRUSTES PROCESS VISUALIZATION
    #
    # Gray = initial rigid reference
    # Blue = Original Action prediction
    # Red  = Corrected Action prediction
    # ========================================================

    reference_xyz = (
        b0["gt_scene_flows"][0, 0, idx]
        .detach()
        .cpu()
        .numpy()
    )

    # ========================================================
    # MPJPE + PA-MPJPE
    # Object ID enters only for evaluation.
    # ========================================================

    def _pa_v4(
        pred_xyz,
        ref_xyz,
    ):

        pred_xyz = np.asarray(
            pred_xyz,
            dtype=np.float64,
        )

        ref_xyz = np.asarray(
            ref_xyz,
            dtype=np.float64,
        )

        pc = (
            pred_xyz
            - pred_xyz.mean(
                axis=0,
                keepdims=True,
            )
        )

        rc = (
            ref_xyz
            - ref_xyz.mean(
                axis=0,
                keepdims=True,
            )
        )

        pn = np.linalg.norm(pc)
        rn = np.linalg.norm(rc)

        if pn < 1e-12 or rn < 1e-12:
            return np.nan

        ps = (
            pc
            * (
                rn / pn
            )
        )

        H = ps.T @ rc

        U, _, Vt = np.linalg.svd(H)

        R = U @ Vt

        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        aligned = ps @ R

        return float(
            np.linalg.norm(
                aligned - rc,
                axis=1,
            ).mean()
            * 1000.0
        )

    goal_boundary_v4 = float(
        np.linalg.norm(
            corrected_xyz
            - original_xyz,
            axis=1,
        ).mean()
        * 1000.0
    )

    goal_repaired_v4 = float(
        np.linalg.norm(
            repaired_xyz
            - original_xyz,
            axis=1,
        ).mean()
        * 1000.0
    )

    repair_delta_v4 = float(
        np.linalg.norm(
            repaired_xyz
            - corrected_xyz,
            axis=1,
        ).mean()
        * 1000.0
    )

    pa_boundary_v4 = _pa_v4(
        corrected_xyz,
        reference_xyz,
    )

    pa_repaired_v4 = _pa_v4(
        repaired_xyz,
        reference_xyz,
    )

    row_v4 = next(
        (
            r
            for r in v4_diag[
                "frame_rows"
            ]
            if int(
                r["frame"]
            ) == int(t)
        ),
        None,
    )

    print()
    print(
        "========== V6 OBJECT EVALUATION =========="
    )

    print(
        "Goal MPJPE original->boundary = "
        f"{goal_boundary_v4:.6f} mm"
    )

    print(
        "Goal MPJPE original->repair   = "
        f"{goal_repaired_v4:.6f} mm"
    )

    print(
        "Repair delta boundary->repair = "
        f"{repair_delta_v4:.6f} mm"
    )

    print(
        "PA-MPJPE boundary = "
        f"{pa_boundary_v4:.6f} mm"
    )

    print(
        "PA-MPJPE repaired = "
        f"{pa_repaired_v4:.6f} mm"
    )

    if row_v4 is not None:

        print(
            "Carried residual = "
            f"{row_v4['carried_residual_mean_mm']:.6f} mm"
        )

        print(
            "Residual frame change = "
            f"{row_v4['residual_change_mean_mm']:.6f} mm"
        )

    print(
        "=========================================="
    )

    # ========================================================
    # UNIQUE V6 visualization
    #
    # 1 Boundary-only
    # 2 Stateful V6
    # 3 residual magnitude/change over time
    # ========================================================

    robot_v4 = (
        b1["robot_flows"][0, t]
        .detach()
        .cpu()
        .numpy()
    )

    fig_v4 = plt.figure(
        figsize=(18, 6)
    )

    ax1 = fig_v4.add_subplot(
        131,
        projection="3d",
    )

    ax2 = fig_v4.add_subplot(
        132,
        projection="3d",
    )

    ax3 = fig_v4.add_subplot(
        133
    )

    vis = np.concatenate(
        [
            corrected_xyz,
            repaired_xyz,
            robot_v4,
        ],
        axis=0,
    )

    mn = vis.min(axis=0)
    mx = vis.max(axis=0)

    ctr = (
        mn + mx
    ) / 2.0

    rad = max(
        float(
            np.max(mx - mn)
            / 2.0
        ),
        1e-3,
    )

    for ax, xyz, title in [
        (
            ax1,
            corrected_xyz,
            "Boundary-only",
        ),
        (
            ax2,
            repaired_xyz,
            "Boundary + Adaptive Contact V6",
        ),
    ]:

        ax.scatter(
            robot_v4[:, 0],
            robot_v4[:, 1],
            robot_v4[:, 2],
            s=10,
            c="black",
            alpha=0.45,
            label="Gripper",
        )

        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=18,
            alpha=0.85,
            label="Object prediction",
        )

        ax.set_xlim(
            ctr[0]-rad,
            ctr[0]+rad,
        )

        ax.set_ylim(
            ctr[1]-rad,
            ctr[1]+rad,
        )

        ax.set_zlim(
            ctr[2]-rad,
            ctr[2]+rad,
        )

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title(title)
        ax.legend()

    frames_v4 = np.asarray(
        [
            r["frame"]
            for r in v4_diag[
                "frame_rows"
            ]
        ]
    )

    residual_v4 = np.asarray(
        [
            r[
                "carried_residual_mean_mm"
            ]
            for r in v4_diag[
                "frame_rows"
            ]
        ],
        dtype=np.float64,
    )

    change_v4 = np.asarray(
        [
            r[
                "residual_change_mean_mm"
            ]
            for r in v4_diag[
                "frame_rows"
            ]
        ],
        dtype=np.float64,
    )

    ax3.plot(
        frames_v4,
        residual_v4,
        marker="o",
        label="Carried correction",
    )

    ax3.plot(
        frames_v4,
        change_v4,
        marker="o",
        label="Frame-to-frame correction change",
    )

    for r in v4_diag[
        "frame_rows"
    ]:
        if r[
            "release_frame"
        ]:
            ax3.axvline(
                r["frame"],
                linestyle="--",
                alpha=0.5,
            )

    ax3.set_xlabel(
        "Frame"
    )

    ax3.set_ylabel(
        "Residual [mm]"
    )

    ax3.set_title(
        "State continuity after release"
    )

    ax3.grid(
        True,
        alpha=0.25,
    )

    ax3.legend()

    fig_v4.suptitle(
        f"{TARGET} | object={oid} | frame={t}\n"
        f"Goal MPJPE "
        f"{goal_boundary_v4:.3f}"
        f" -> {goal_repaired_v4:.3f} mm | "
        f"PA-MPJPE "
        f"{pa_boundary_v4:.3f}"
        f" -> {pa_repaired_v4:.3f} mm"
    )

    fig_v4.tight_layout()

    fig_path_v4 = (
        V6_OUT
        / (
            f"{safe_target_v4}"
            f"__t{t}"
            "__adaptive_contact_v6"
            f"__{V6_TAG}.png"
        )
    )

    fig_v4.savefig(
        fig_path_v4,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig_v4
    )

    print(
        "V6 FIGURE =",
        fig_path_v4,
    )

    def center_points(x):
        mu = x.mean(axis=0, keepdims=True)
        return x - mu, mu

    def scale_to_reference(x_centered, ref_centered):
        nx = np.linalg.norm(x_centered)
        nr = np.linalg.norm(ref_centered)

        if nx < 1e-12 or nr < 1e-12:
            raise RuntimeError("Degenerate point cloud")

        return x_centered * (nr / nx)

    def rotate_to_reference(x, ref):
        """
        Proper Kabsch rotation.
        x/ref are already centered and scaled.
        """
        H = x.T @ ref

        U, S, Vt = np.linalg.svd(H)

        R = U @ Vt

        # No reflection
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt

        return x @ R

    # --------------------------------------------------------
    # STEP 0: RAW
    # --------------------------------------------------------
    ref_raw = reference_xyz.copy()
    org_raw = original_xyz.copy()
    cor_raw = corrected_xyz.copy()

    # --------------------------------------------------------
    # STEP 1: TRANSLATION / CENTERING
    # Put all clouds at the same centroid.
    # --------------------------------------------------------
    ref_center, ref_mu = center_points(ref_raw)
    org_center, _ = center_points(org_raw)
    cor_center, _ = center_points(cor_raw)

    # Display around reference centroid
    ref_center_vis = ref_center + ref_mu
    org_center_vis = org_center + ref_mu
    cor_center_vis = cor_center + ref_mu

    # --------------------------------------------------------
    # STEP 2: SCALE
    # Match overall point-cloud size to reference.
    # --------------------------------------------------------
    org_scale = scale_to_reference(
        org_center,
        ref_center,
    )

    cor_scale = scale_to_reference(
        cor_center,
        ref_center,
    )

    ref_scale_vis = ref_center + ref_mu
    org_scale_vis = org_scale + ref_mu
    cor_scale_vis = cor_scale + ref_mu

    # --------------------------------------------------------
    # STEP 3: ROTATION
    # Kabsch rotation -> final aligned shapes
    # --------------------------------------------------------
    org_rot = rotate_to_reference(
        org_scale,
        ref_center,
    )

    cor_rot = rotate_to_reference(
        cor_scale,
        ref_center,
    )

    ref_final = ref_center + ref_mu
    org_final = org_rot + ref_mu
    cor_final = cor_rot + ref_mu

    # Residual after alignment
    org_pa_mm = (
        np.linalg.norm(
            org_final - ref_final,
            axis=-1,
        ).mean()
        * 1000.0
    )

    cor_pa_mm = (
        np.linalg.norm(
            cor_final - ref_final,
            axis=-1,
        ).mean()
        * 1000.0
    )

    print()
    print("Original aligned residual =", org_pa_mm, "mm")
    print("Corrected aligned residual =", cor_pa_mm, "mm")

    # ========================================================
    # DRAW PROCESS + SCENE CONTEXT
    # ========================================================

    # --------------------------------------------------------
    # Real object/mesh name
    # OBJECT_TABLE: object_id -> full BEHAVIOR mesh name
    # ex) World__scene_0__can_of_soda_114__base_link__visuals
    # --------------------------------------------------------
    table = omap.OBJECT_TABLE.get(TARGET, {})
    object_full_name = str(
        table.get(oid, f"object_{oid}")
    )

    parts = object_full_name.split("__")
    object_name = object_full_name

    for i, part in enumerate(parts):
        if (
            part.startswith("scene_")
            and i + 1 < len(parts)
        ):
            object_name = parts[i + 1]
            break

    # --------------------------------------------------------
    # Full initial scene
    # --------------------------------------------------------
    full_gt = (
        b0["gt_scene_flows"][0]
        .detach()
        .cpu()
        .numpy()
    )

    scene_mask0 = (
        exists[0]
        & (~context[0])
    )

    scene0 = full_gt[0][scene_mask0]

    if len(scene0) == 0:
        scene0 = full_gt[0][exists[0]]

    if len(scene0) == 0:
        scene0 = full_gt[0]

    # --------------------------------------------------------
    # Nearby scene context around target object
    # --------------------------------------------------------
    object_center = ref_raw.mean(axis=0)

    dist = np.linalg.norm(
        scene0 - object_center[None, :],
        axis=1,
    )

    # show nearest scene points so object identity/context is visible
    n_context = min(
        2500,
        len(scene0),
    )

    nearest_idx = np.argsort(dist)[:n_context]
    scene_near = scene0[nearest_idx]

    def equal_axis(ax, clouds, pad=1.15):
        clouds = [
            x for x in clouds
            if x is not None and len(x) > 0
        ]

        q = np.concatenate(
            clouds,
            axis=0,
        )

        lo = q.min(axis=0)
        hi = q.max(axis=0)

        center = (lo + hi) / 2.0

        radius = max(
            (hi - lo).max() / 2.0,
            0.005,
        ) * pad

        ax.set_xlim(
            center[0] - radius,
            center[0] + radius,
        )
        ax.set_ylim(
            center[1] - radius,
            center[1] + radius,
        )
        ax.set_zlim(
            center[2] - radius,
            center[2] + radius,
        )

        ax.set_box_aspect((1, 1, 1))
        ax.view_init(
            elev=22,
            azim=-60,
        )

    fig = plt.figure(
        figsize=(22, 12)
    )

    # ========================================================
    # 1. TARGET OBJECT + LOCAL SCENE CONTEXT
    # ========================================================
    ax = fig.add_subplot(
        2, 3, 1,
        projection="3d",
    )

    ax.scatter(
        scene_near[:, 0],
        scene_near[:, 1],
        scene_near[:, 2],
        s=4,
        color="gray",
        alpha=0.10,
    )

    ax.scatter(
        ref_raw[:, 0],
        ref_raw[:, 1],
        ref_raw[:, 2],
        s=22,
        color="orange",
        alpha=0.95,
        label=object_name,
    )

    ax.set_title(
        f"1. Target object in scene\n"
        f"{object_name}",
        fontsize=14,
    )

    ax.legend(fontsize=10)

    equal_axis(
        ax,
        [scene_near, ref_raw],
        pad=1.08,
    )

    # ========================================================
    # 2. RAW SEGMENTED OBJECT
    # ========================================================
    ax = fig.add_subplot(
        2, 3, 2,
        projection="3d",
    )

    ax.scatter(
        ref_raw[:,0],
        ref_raw[:,1],
        ref_raw[:,2],
        s=10,
        color="gray",
        alpha=0.22,
        label="Initial reference",
    )

    ax.scatter(
        org_raw[:,0],
        org_raw[:,1],
        org_raw[:,2],
        s=14,
        color="blue",
        alpha=0.65,
        label="Original Action",
    )

    ax.scatter(
        cor_raw[:,0],
        cor_raw[:,1],
        cor_raw[:,2],
        s=14,
        color="red",
        alpha=0.55,
        label="Corrected Action",
    )

    ax.set_title(
        "2. Segmented object\nRaw prediction",
        fontsize=14,
    )

    ax.legend(fontsize=9)

    equal_axis(
        ax,
        [ref_raw, org_raw, cor_raw],
    )

    # ========================================================
    # 3. TRANSLATION
    # ========================================================
    ax = fig.add_subplot(
        2, 3, 3,
        projection="3d",
    )

    ax.scatter(
        ref_center_vis[:,0],
        ref_center_vis[:,1],
        ref_center_vis[:,2],
        s=9,
        color="gray",
        alpha=0.20,
    )

    ax.scatter(
        org_center_vis[:,0],
        org_center_vis[:,1],
        org_center_vis[:,2],
        s=14,
        color="blue",
        alpha=0.65,
    )

    ax.scatter(
        cor_center_vis[:,0],
        cor_center_vis[:,1],
        cor_center_vis[:,2],
        s=14,
        color="red",
        alpha=0.50,
    )

    ax.set_title(
        "3. Translation removed\nSame center",
        fontsize=14,
    )

    equal_axis(
        ax,
        [
            ref_center_vis,
            org_center_vis,
            cor_center_vis,
        ],
    )

    # ========================================================
    # 4. SCALE
    # ========================================================
    ax = fig.add_subplot(
        2, 3, 4,
        projection="3d",
    )

    ax.scatter(
        ref_scale_vis[:,0],
        ref_scale_vis[:,1],
        ref_scale_vis[:,2],
        s=9,
        color="gray",
        alpha=0.20,
    )

    ax.scatter(
        org_scale_vis[:,0],
        org_scale_vis[:,1],
        org_scale_vis[:,2],
        s=14,
        color="blue",
        alpha=0.65,
    )

    ax.scatter(
        cor_scale_vis[:,0],
        cor_scale_vis[:,1],
        cor_scale_vis[:,2],
        s=14,
        color="red",
        alpha=0.50,
    )

    ax.set_title(
        "4. Scale normalized\nSame overall size",
        fontsize=14,
    )

    equal_axis(
        ax,
        [
            ref_scale_vis,
            org_scale_vis,
            cor_scale_vis,
        ],
    )

    # ========================================================
    # 5. ROTATION / FINAL PA
    # Different markers so overlap is visible
    # ========================================================
    ax = fig.add_subplot(
        2, 3, 5,
        projection="3d",
    )

    ax.scatter(
        ref_final[:,0],
        ref_final[:,1],
        ref_final[:,2],
        s=8,
        color="gray",
        alpha=0.16,
        label="Initial reference",
    )

    ax.scatter(
        org_final[:,0],
        org_final[:,1],
        org_final[:,2],
        s=20,
        marker="x",
        color="blue",
        alpha=0.85,
        label="Original Action",
    )

    ax.scatter(
        cor_final[:,0],
        cor_final[:,1],
        cor_final[:,2],
        s=24,
        marker="o",
        facecolors="none",
        edgecolors="red",
        linewidths=0.8,
        alpha=0.80,
        label="Corrected Action",
    )

    ax.set_title(
        "5. Rotation aligned\nFinal PA alignment",
        fontsize=14,
    )

    ax.legend(fontsize=9)

    equal_axis(
        ax,
        [
            ref_final,
            org_final,
            cor_final,
        ],
    )

    # ========================================================
    # 6. DIRECT ORIGINAL vs CORRECTED
    # ========================================================
    ax = fig.add_subplot(
        2, 3, 6,
        projection="3d",
    )

    ax.scatter(
        org_final[:,0],
        org_final[:,1],
        org_final[:,2],
        s=22,
        marker="x",
        color="blue",
        alpha=0.90,
        label="Original Action",
    )

    ax.scatter(
        cor_final[:,0],
        cor_final[:,1],
        cor_final[:,2],
        s=28,
        marker="o",
        facecolors="none",
        edgecolors="red",
        linewidths=0.9,
        alpha=0.85,
        label="Corrected Action",
    )

    ax.set_title(
        "6. Original vs Corrected\n"
        "Blue × = Original   Red ○ = Corrected",
        fontsize=14,
    )

    ax.legend(fontsize=9)

    equal_axis(
        ax,
        [
            org_final,
            cor_final,
        ],
    )

    ax.text2D(
        0.02,
        -0.13,
        (
            f"Original PA residual:  {org_pa_mm:.3f} mm\n"
            f"Corrected PA residual: {cor_pa_mm:.3f} mm"
        ),
        transform=ax.transAxes,
        fontsize=11,
    )

    for ax in fig.axes:
        if hasattr(ax, "set_xlabel"):
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

    fig.suptitle(
        f"{TARGET}\n"
        f"Object: {object_name}  |  frame={t}  |  object_id={oid}",
        fontsize=17,
        y=0.985,
    )

    plt.tight_layout(
        rect=[0.01, 0.04, 0.99, 0.92],
        h_pad=4.0,
        w_pad=2.5,
    )

    safe = (
        TARGET
        .replace("/", "_")
        .replace(":", "__")
    )

    out = (
        OUTDIR /
        f"{safe}__t{t}__object_context_pa_process_v2.png"
    )

    fig.savefig(
        out,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("OBJECT FULL NAME =", object_full_name)
    print("OBJECT NAME      =", object_name)
    print("OBJECT ID        =", oid)
    print("POINTS           =", len(ref_raw))
    print("Original PA      =", org_pa_mm, "mm")
    print("Corrected PA     =", cor_pa_mm, "mm")
    print("SAVED            =", out)
    print("DONE")

    break

if not found:
    raise RuntimeError(
        f"Target clip not found: {TARGET}"
    )
