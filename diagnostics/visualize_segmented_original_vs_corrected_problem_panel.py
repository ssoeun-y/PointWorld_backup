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
    "segmented_original_vs_corrected"
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

omap.pmod.grid_sample_transform = (
    omap.grid_sample_with_object_ids
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

    for name in [
        "right_gripper_open",
        "left_gripper_open",
        "gripper_positions",
    ]:

        if name not in out:
            continue

        a = np.asarray(out[name])

        if (
            a.ndim >= 1
            and
            a.shape[0] == len(PHASE)
        ):
            out[name] = resample(
                a,
                PHASE,
            )

    return out

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

    t = args_local.t

    # ========================================================
    # Find eligible moving rigid objects
    # ========================================================

    candidates = []

    for oid in sorted(
        np.unique(
            object_id[t]
        ).tolist()
    ):

        oid = int(oid)

        if oid <= 0:
            continue

        m = (
            (object_id[t] == oid)
            & valid[t]
            & moved[t]
            & (object_id[0] == oid)
            & exists[0]
        )

        n = int(m.sum())

        if n >= 8:
            candidates.append(
                (n, oid)
            )

    if not candidates:
        raise RuntimeError(
            f"No eligible moving object at t={t}"
        )

    candidates.sort(
        reverse=True
    )

    if args_local.object_id is None:
        oid = candidates[0][1]
    else:
        oid = args_local.object_id

    m = (
        (object_id[t] == oid)
        & valid[t]
        & moved[t]
        & (object_id[0] == oid)
        & exists[0]
    )

    idx = np.where(m)[0]

    original_xyz = (
        pred_original[t, idx]
    )

    corrected_xyz = (
        pred_corrected[t, idx]
    )

    print(
        "OBJECT ID =",
        oid,
        "POINTS =",
        len(idx),
    )

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

    # ========================================================
    # PAPER VISUALIZATION
    # Metric / PA above remains unchanged.
    # Only visualization below is replaced.
    # ========================================================

    from matplotlib.patches import Rectangle

    def _np0(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim > 0 and x.shape[0] == 1:
            x = x[0]
        return x

    def _fps(points, k=40):
        p = np.asarray(points, dtype=np.float64)
        n = len(p)

        if n <= k:
            return np.arange(n, dtype=int)

        c = p.mean(axis=0)
        first = int(np.argmax(np.linalg.norm(p - c, axis=1)))

        selected = [first]
        d2 = np.sum((p - p[first]) ** 2, axis=1)

        for _ in range(1, k):
            j = int(np.argmax(d2))
            selected.append(j)

            new_d2 = np.sum((p - p[j]) ** 2, axis=1)
            d2 = np.minimum(d2, new_d2)

        return np.asarray(selected, dtype=int)

    def _pca_basis(x):
        xc = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(xc, full_matrices=False)
        return vt[:2].T

    def _p2(x, basis):
        return np.asarray(x) @ basis

    def _kabsch(src, ref):
        h = src.T @ ref
        u, _, vt = np.linalg.svd(h)
        r = u @ vt

        if np.linalg.det(r) < 0:
            vt[-1] *= -1
            r = u @ vt

        return r



    def _nearest_unique(points, targets, allowed=None):
        """
        Pick REAL PointWorld point indices nearest to canonical shape targets.
        No synthetic landmark coordinates are returned.
        """
        pts = np.asarray(points, dtype=np.float64)

        if allowed is None:
            allowed = np.arange(len(pts))
        else:
            allowed = np.asarray(allowed, dtype=int)

        used = set()
        result = []

        for t in np.asarray(targets):
            d = np.linalg.norm(
                pts[allowed] - t[None],
                axis=1,
            )

            for k in np.argsort(d):
                j = int(allowed[k])
                if j not in used:
                    result.append(j)
                    used.add(j)
                    break

        return result


    def _radio_graph(points):
        """
        Extract a sparse radio-like landmark graph from REAL reference points.

        Body:
          corners + a few edge points

        Antenna:
          base -> middle -> tip

        Returns:
          landmark indices
          graph edges between landmark indices
        """
        pts = np.asarray(points, dtype=np.float64).copy()

        # PCA already gives major/minor axes.
        # Orient minor axis so largest protrusion points upward.
        x = pts[:, 0]
        y = pts[:, 1]

        q10, q90 = np.percentile(y, [10, 90])

        pos_tail = y.max() - q90
        neg_tail = q10 - y.min()

        sy = 1.0 if pos_tail >= neg_tail else -1.0
        pts[:, 1] *= sy

        x = pts[:, 0]
        y = pts[:, 1]

        # Upper protrusion ≈ antenna.
        top = y >= np.percentile(y, 88)

        # Put antenna on visual right for readability.
        if top.sum() >= 2:
            sx = (
                1.0
                if np.mean(x[top]) >= np.median(x)
                else -1.0
            )
        else:
            sx = 1.0

        pts[:, 0] *= sx

        x = pts[:, 0]
        y = pts[:, 1]

        # ----------------------------------------------------
        # BODY: robust rectangle-like bulk, ignoring sparse top tail
        # ----------------------------------------------------
        body_mask = y <= np.percentile(y, 84)

        body_ids = np.where(body_mask)[0]

        if len(body_ids) < 8:
            body_ids = np.arange(len(pts))

        xb = x[body_ids]
        yb = y[body_ids]

        xl, xr = np.percentile(xb, [6, 94])
        yb0, yt = np.percentile(yb, [7, 93])

        w = xr - xl
        h = yt - yb0

        # 12 canonical positions around radio body.
        # Nearest REAL object point is selected for each.
        body_targets = np.array([
            [xl,             yb0],
            [xl + 0.33*w,    yb0],
            [xl + 0.67*w,    yb0],
            [xr,             yb0],

            [xr,             yb0 + 0.50*h],
            [xr,             yt],

            [xl + 0.67*w,    yt],
            [xl + 0.33*w,    yt],
            [xl,             yt],
            [xl,             yb0 + 0.50*h],
        ])

        body = _nearest_unique(
            pts,
            body_targets,
            allowed=body_ids,
        )

        # Body graph follows the perimeter.
        edges = [
            (i, (i + 1) % len(body))
            for i in range(len(body))
        ]

        landmark_ids = list(body)

        # ----------------------------------------------------
        # ANTENNA: detect real protruding points above body
        # ----------------------------------------------------
        antenna_ids = np.where(
            y > yt + 0.05 * max(h, 1e-8)
        )[0]

        if len(antenna_ids) >= 3:
            xa = x[antenna_ids]
            ya = y[antenna_ids]

            # antenna lateral position
            ax = np.median(xa)

            antenna_targets = np.array([
                [ax, yt],
                [ax, yt + 0.45 * (ya.max() - yt)],
                [ax, ya.max()],
            ])

            ant = _nearest_unique(
                pts,
                antenna_targets,
                allowed=np.unique(
                    np.concatenate([
                        antenna_ids,
                        body_ids,
                    ])
                ),
            )

            # make sure antenna points are unique
            ant = [
                j for j in ant
                if j not in landmark_ids
            ]

            if len(ant) >= 2:
                base_offset = len(landmark_ids)

                # connect closest body landmark to antenna base
                base_pt = pts[ant[0]]

                nearest_body_local = int(
                    np.argmin(
                        np.linalg.norm(
                            pts[np.asarray(body)] - base_pt,
                            axis=1,
                        )
                    )
                )

                edges.append(
                    (nearest_body_local, base_offset)
                )

                for k in range(len(ant) - 1):
                    edges.append(
                        (
                            base_offset + k,
                            base_offset + k + 1,
                        )
                    )

                landmark_ids.extend(ant)

        # Return orientation signs too so every stage uses same visual orientation.
        return (
            np.asarray(landmark_ids, dtype=int),
            edges,
            sx,
            sy,
        )


    def _draw_radio_graph(
        ax,
        ref,
        org,
        cor,
        landmark_ids,
        edges,
        title,
    ):
        """
        NO dense scatter.
        Only sparse shape-defining landmarks and graph edges.
        """
        ri = ref[landmark_ids]
        oi = org[landmark_ids]
        ci = cor[landmark_ids]

        # graph lines
        for a, b in edges:
            ax.plot(
                [ri[a,0], ri[b,0]],
                [ri[a,1], ri[b,1]],
                color="black",
                lw=1.3,
                alpha=0.65,
                zorder=1,
            )

            ax.plot(
                [oi[a,0], oi[b,0]],
                [oi[a,1], oi[b,1]],
                color="blue",
                lw=2.0,
                alpha=0.90,
                zorder=2,
            )

            ax.plot(
                [ci[a,0], ci[b,0]],
                [ci[a,1], ci[b,1]],
                color="red",
                lw=2.0,
                alpha=0.90,
                zorder=3,
            )

        # ONLY meaningful landmarks
        ax.scatter(
            ri[:,0], ri[:,1],
            s=24,
            c="black",
            zorder=4,
            label="GT0 (reference)",
        )

        ax.scatter(
            oi[:,0], oi[:,1],
            s=40,
            marker="x",
            c="blue",
            linewidths=1.6,
            zorder=5,
            label="Original",
        )

        ax.scatter(
            ci[:,0], ci[:,1],
            s=42,
            marker="o",
            facecolors="none",
            edgecolors="red",
            linewidths=1.6,
            zorder=6,
            label="Corrected",
        )

        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")


    def _draw(
        ax,
        ref,
        org,
        cor,
        landmarks,
        title,
        connections=False,
    ):
        # kept same call signature used by existing panels
        _draw_radio_graph(
            ax,
            ref,
            org,
            cor,
            RADIO_LANDMARKS,
            RADIO_EDGES,
            title,
        )

    def _project(points, K, E, H, W):
        p = np.asarray(points, dtype=np.float64)

        ph = np.concatenate(
            [p, np.ones((len(p), 1))],
            axis=1,
        )

        # PointWorld extrinsic: transformed world -> camera
        cam = (E @ ph.T).T[:, :3]

        pix = (K @ cam.T).T
        z = pix[:, 2]

        ok = np.isfinite(z) & (z > 1e-6)

        uv = np.zeros((len(p), 2), dtype=float)
        uv[ok] = pix[ok, :2] / z[ok, None]

        inside = (
            ok
            & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )

        return uv, inside

    # --------------------------------------------------------
    # FULL object for visualization.
    # Metric still uses original idx/ref_raw/org_raw/cor_raw above.
    # --------------------------------------------------------

    full_mask = (
        (object_id[0] == oid)
        & (object_id[t] == oid)
        & exists[0]
        & exists[t]
    )

    full_idx = np.where(full_mask)[0]

    if len(full_idx) < 8:
        full_idx = idx.copy()

    ref_full = (
        b0["gt_scene_flows"][0, 0, full_idx]
        .detach().cpu().numpy()
    )

    org_full = pred_original[t, full_idx]
    cor_full = pred_corrected[t, full_idx]

    # --------------------------------------------------------
    # PA display transforms estimated using the SAME metric points
    # --------------------------------------------------------

    ref_mu2 = ref_raw.mean(axis=0, keepdims=True)
    org_mu2 = org_raw.mean(axis=0, keepdims=True)
    cor_mu2 = cor_raw.mean(axis=0, keepdims=True)

    ref_c2 = ref_raw - ref_mu2
    org_c2 = org_raw - org_mu2
    cor_c2 = cor_raw - cor_mu2

    ref_full_c = ref_full - ref_mu2
    org_full_c = org_full - org_mu2
    cor_full_c = cor_full - cor_mu2

    nr = max(np.linalg.norm(ref_c2), 1e-12)
    no = max(np.linalg.norm(org_c2), 1e-12)
    nc = max(np.linalg.norm(cor_c2), 1e-12)

    so = nr / no
    sc = nr / nc

    ref_s2 = ref_c2
    org_s2 = org_c2 * so
    cor_s2 = cor_c2 * sc

    ref_full_s = ref_full_c
    org_full_s = org_full_c * so
    cor_full_s = cor_full_c * sc

    Ro = _kabsch(org_s2, ref_s2)
    Rc = _kabsch(cor_s2, ref_s2)

    ref_full_r = ref_full_s
    org_full_r = org_full_s @ Ro
    cor_full_r = cor_full_s @ Rc

    # --------------------------------------------------------
    # PCA is visualization ONLY. Metric remains full 3D above.
    # --------------------------------------------------------

    basis = _pca_basis(ref_c2)

    raw_ref2 = _p2(ref_full - ref_mu2, basis)
    raw_org2 = _p2(org_full - ref_mu2, basis)
    raw_cor2 = _p2(cor_full - ref_mu2, basis)

    cen_ref2 = _p2(ref_full_c, basis)
    cen_org2 = _p2(org_full_c, basis)
    cen_cor2 = _p2(cor_full_c, basis)

    scl_ref2 = _p2(ref_full_s, basis)
    scl_org2 = _p2(org_full_s, basis)
    scl_cor2 = _p2(cor_full_s, basis)

    rot_ref2 = _p2(ref_full_r, basis)
    rot_org2 = _p2(org_full_r, basis)
    rot_cor2 = _p2(cor_full_r, basis)

    # Sparse radio-shaped graph from REAL corresponding points.
    RADIO_LANDMARKS, RADIO_EDGES, RADIO_SX, RADIO_SY = _radio_graph(
        raw_ref2
    )

    # Apply identical display orientation to every PA stage.
    for _arr in [
        raw_ref2, raw_org2, raw_cor2,
        cen_ref2, cen_org2, cen_cor2,
        scl_ref2, scl_org2, scl_cor2,
        rot_ref2, rot_org2, rot_cor2,
    ]:
        _arr[:, 0] *= RADIO_SX
        _arr[:, 1] *= RADIO_SY

    landmarks = RADIO_LANDMARKS

    # --------------------------------------------------------
    # Object name
    # --------------------------------------------------------

    if (
        TARGET == "task-0000_episode_00001310-260:271"
        and int(oid) == 2
    ):
        vis_object_name = "radio_89"
    else:
        vis_object_name = f"object_{oid}"

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------




    # ========================================================
    # SIMPLE PROBLEM PANEL EXPORT
    #
    # Exports:
    #   - context image (small RGB/object-context crop)
    #   - original rollout panel for current t
    #   - corrected rollout panel for current t
    #
    # This is intentionally simple for intro/abstract figure.
    # ========================================================
    from pathlib import Path
    from PIL import Image
    import matplotlib.colors as mcolors

    def _equal_3d(ax, arrays, pad=1.08):
        pts = np.concatenate([np.asarray(a) for a in arrays], axis=0)
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        ctr = 0.5 * (lo + hi)
        rad = 0.5 * np.max(hi - lo)
        rad = max(rad, 1e-8) * pad
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
        ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
        ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
        ax.set_box_aspect((1, 1, 1))

    def _depth_norm(pts):
        z = np.asarray(pts)[:, 2]
        if float(z.max() - z.min()) < 1e-12:
            return np.ones_like(z) * 0.5
        return (z - z.min()) / (z.max() - z.min())

    def _vis_transform(points, basis, origin):
        pts = np.asarray(points) - origin
        return pts @ basis

    def _draw_gt_pred(ax, ref_pts, pred_pts, pred_kind="original", title=None):
        ref_pts = np.asarray(ref_pts)
        pred_pts = np.asarray(pred_pts)

        rz = _depth_norm(ref_pts)
        ref_cols = np.stack([
            0.70 + 0.18 * rz,
            0.70 + 0.18 * rz,
            0.70 + 0.18 * rz,
            0.18 + 0.45 * rz
        ], axis=1)

        ax.scatter(
            ref_pts[:, 0], ref_pts[:, 1], ref_pts[:, 2],
            s=5.0 + 8.0 * rz,
            c=ref_cols,
            linewidths=0,
            depthshade=False,
            label="Reference",
        )

        pz = _depth_norm(pred_pts)
        if pred_kind == "original":
            rgba = np.stack([
                np.full(len(pred_pts), 0.85),
                np.full(len(pred_pts), 0.36),
                np.full(len(pred_pts), 0.22),
                0.28 + 0.60 * pz,
            ], axis=1)
            ax.scatter(
                pred_pts[:, 0], pred_pts[:, 1], pred_pts[:, 2],
                s=7.0 + 10.0 * pz,
                c=rgba,
                linewidths=0,
                depthshade=False,
                label="Original",
            )
        else:
            edge_rgba = np.stack([
                np.full(len(pred_pts), 0.23),
                np.full(len(pred_pts), 0.47),
                np.full(len(pred_pts), 0.72),
                0.35 + 0.55 * pz,
            ], axis=1)
            ax.scatter(
                pred_pts[:, 0], pred_pts[:, 1], pred_pts[:, 2],
                s=12.0 + 14.0 * pz,
                facecolors="none",
                edgecolors=edge_rgba,
                linewidths=0.9,
                depthshade=False,
                label="Corrected",
            )

        if title is not None:
            ax.set_title(title, fontsize=10, pad=4)

        ax.set_axis_off()

    # --------------------------------------------------------
    # Build one stable visualization basis from reference
    # final aligned shapes are enough for intro figure
    # --------------------------------------------------------
    ref0 = ref_full_r.copy()
    ref0_center = ref0.mean(axis=0, keepdims=True)
    ref0_zero = ref0 - ref0_center

    _, _, vt = np.linalg.svd(ref0_zero, full_matrices=False)
    VIS_BASIS = vt.T
    if np.linalg.det(VIS_BASIS) < 0:
        VIS_BASIS[:, 2] *= -1

    ref_vis = _vis_transform(ref_full_r, VIS_BASIS, ref0_center)
    org_vis = _vis_transform(org_full_r, VIS_BASIS, ref0_center)
    cor_vis = _vis_transform(cor_full_r, VIS_BASIS, ref0_center)

    VIEW = (20, -62)

    safe_local = TARGET.replace(":", "__")
    out_root = Path("diagnostics/results/problem_intro_panels")
    out_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1) context crop
    # Use any earlier figure with RGB panel if available.
    # Only save once per clip.
    # --------------------------------------------------------
    context_out = out_root / f"{safe_local}__context.png"
    if not context_out.exists():
        context_candidates = [
            Path("diagnostics/results/segmented_original_vs_corrected") / f"{safe_local}__t4__paper_procrustes.png",
            Path("diagnostics/results/segmented_original_vs_corrected") / f"{safe_local}__t4__shape_procrustes.png",
            Path("diagnostics/results/segmented_original_vs_corrected") / f"{safe_local}__t4__object_context_pa_process_v2.png",
        ]

        found = None
        for cc in context_candidates:
            if cc.exists():
                found = cc
                break

        if found is not None:
            im = Image.open(found).convert("RGB")
            W, H = im.size

            # crop top-left panel region
            x0 = int(0.02 * W)
            x1 = int(0.34 * W)
            y0 = int(0.03 * H)
            y1 = int(0.48 * H)

            crop = im.crop((x0, y0, x1, y1))
            crop.save(context_out)
        else:
            # make blank placeholder
            fig_ctx = plt.figure(figsize=(3.0, 2.4))
            ax_ctx = fig_ctx.add_subplot(111)
            ax_ctx.axis("off")
            ax_ctx.text(
                0.5, 0.55,
                f"{TARGET}\nobject: radio_89",
                ha="center", va="center", fontsize=11
            )
            ax_ctx.text(
                0.5, 0.35,
                "Context image not found",
                ha="center", va="center", fontsize=9, color="0.4"
            )
            fig_ctx.savefig(context_out, dpi=220, bbox_inches="tight")
            plt.close(fig_ctx)

    # --------------------------------------------------------
    # 2) original panel
    # --------------------------------------------------------
    fig1 = plt.figure(figsize=(2.7, 2.2))
    ax1 = fig1.add_subplot(111, projection="3d")
    _draw_gt_pred(
        ax1,
        ref_vis,
        org_vis,
        pred_kind="original",
        title=None,
    )
    _equal_3d(ax1, [ref_vis, org_vis], pad=1.10)
    ax1.view_init(elev=VIEW[0], azim=VIEW[1])

    org_out = out_root / f"{safe_local}__t{t}__original.png"
    fig1.savefig(org_out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig1)

    # --------------------------------------------------------
    # 3) corrected panel
    # --------------------------------------------------------
    fig2 = plt.figure(figsize=(2.7, 2.2))
    ax2 = fig2.add_subplot(111, projection="3d")
    _draw_gt_pred(
        ax2,
        ref_vis,
        cor_vis,
        pred_kind="corrected",
        title=None,
    )
    _equal_3d(ax2, [ref_vis, cor_vis], pad=1.10)
    ax2.view_init(elev=VIEW[0], azim=VIEW[1])

    cor_out = out_root / f"{safe_local}__t{t}__corrected.png"
    fig2.savefig(cor_out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    print("=" * 80)
    print("SAVED SIMPLE PANELS")
    print("CONTEXT   =", context_out)
    print("ORIGINAL  =", org_out)
    print("CORRECTED =", cor_out)
    print("=" * 80)
    safe = (
        TARGET
        .replace(":", "__")
        .replace("/", "_")
    )

    outpath = (
        OUTDIR
        / f"{safe}__t{t}__problem_panel_export.png"
    )

    fig.savefig(
        outpath,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("=" * 80)
    print("SAVED        =", outpath)
    print("OBJECT       =", vis_object_name)
    print("OBJECT ID    =", oid)
    print("FULL POINTS  =", len(ref_full))
    print("LANDMARKS    =", len(landmarks))
    print("Original PA  =", org_pa_mm, "mm")
    print("Corrected PA =", cor_pa_mm, "mm")
    print("=" * 80)

    break
if not found:
    raise RuntimeError(
        f"Target clip not found: {TARGET}"
    )
