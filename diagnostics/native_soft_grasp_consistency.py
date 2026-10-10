#!/usr/bin/env python3

import numpy as np
import transform_utils


def _as_open(x):
    """
    PointWorld gripper_open -> [T] float.
    open=1, closed=0.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.clip(x, 0.0, 1.0)


def _pose_mats(pose):
    """
    Official PointWorld pose convention:
    [x, y, z, qx, qy, qz, qw] -> [T,4,4]
    """
    pose = np.asarray(pose, dtype=np.float64)
    H = transform_utils.convert_pose_quat2mat(pose)
    return np.asarray(H, dtype=np.float64)


def _soft_weight_from_official_norm(d, mean, var):
    """
    No distance threshold.

    PointWorld's raw dist2robot is standardized with its
    training-set normalization statistics.

    We only map that official z score monotonically to [0,1]:
        w = sigmoid(-z)

    Smaller dist2robot -> larger interaction weight.
    No clip-specific fitting.
    """
    d = np.asarray(d, dtype=np.float64)

    sigma = float(np.sqrt(max(float(var), 1e-12)))
    z = (d - float(mean)) / sigma
    z = np.clip(z, -30.0, 30.0)

    return 1.0 / (1.0 + np.exp(z))


def _first_max_closed_frame(open_seq):
    """
    No grasp-distance threshold.

    g = 1 - gripper_open.
    Pick the first frame at which closure is maximal.

    If the gripper is fully open for the whole clip,
    there is no constraint.
    """
    g = 1.0 - _as_open(open_seq)

    if float(np.max(g)) <= 0.0:
        return None

    return int(np.argmax(g))


def _hand_targets(reference_pred, H_ref, H_run, t0):
    """
    Fix each scene point in the hand frame at grasp anchor t0.

        q_i = H_ref(t0)^-1 p_ref(i,t0)
        target(i,t) = H_run(t) q_i

    No Kabsch, no object pose fitting, no rigid-shape projection.
    """
    reference_pred = np.asarray(reference_pred, dtype=np.float64)

    R0 = H_ref[t0, :3, :3]
    x0 = H_ref[t0, :3, 3]

    p0 = reference_pred[t0]

    q = (p0 - x0[None, :]) @ R0

    targets = np.empty_like(reference_pred)

    for t in range(len(reference_pred)):
        Rt = H_run[t, :3, :3]
        xt = H_run[t, :3, 3]

        targets[t] = q @ Rt.T + xt[None, :]

    return targets


def apply_native_soft_grasp_consistency(
    *,
    reference_pred,
    run_pred,
    dist2robot_reference,
    dist_mean,
    dist_var,
    right_pose_reference,
    right_pose_run,
    left_pose_reference,
    left_pose_run,
    right_open_reference,
    right_open_run,
    left_open_reference,
    left_open_run,
    active_sides,
    exists=None,
):
    """
    PointWorld-native soft grasp consistency.

    METHOD INPUTS:
      - PointWorld prediction
      - PointWorld native dist2robot
      - PointWorld native gripper_open
      - PointWorld gripper poses
      - official normalization stats

    NO:
      - object id
      - object mask
      - DINO clustering
      - co-motion
      - grasp-distance threshold
      - Kabsch
      - rigid-shape projection
      - clip-specific parameter fitting

    reference_pred : Original PointWorld prediction [T,N,3]
    run_pred       : Corrected PointWorld prediction [T,N,3]

    dist2robot_reference : Original PointWorld dist2robot [N,T]
    dist_mean / dist_var : official scene normalization stats [T]
    """

    reference_pred = np.asarray(
        reference_pred,
        dtype=np.float64,
    )

    out = np.asarray(
        run_pred,
        dtype=np.float64,
    ).copy()

    dref = np.asarray(
        dist2robot_reference,
        dtype=np.float64,
    )

    T, N, _ = out.shape

    if dref.shape != (N, T):
        raise RuntimeError(
            f"dist2robot shape expected {(N,T)}, got {dref.shape}"
        )

    dist_mean = np.asarray(
        dist_mean,
        dtype=np.float64,
    ).reshape(-1)

    dist_var = np.asarray(
        dist_var,
        dtype=np.float64,
    ).reshape(-1)

    if len(dist_mean) != T or len(dist_var) != T:
        raise RuntimeError(
            f"norm stats expected T={T}, "
            f"got mean={len(dist_mean)}, var={len(dist_var)}"
        )

    if exists is None:
        exists = np.ones((T, N), dtype=bool)
    else:
        exists = np.asarray(exists, dtype=bool)

        if exists.shape != (T, N):
            raise RuntimeError(
                f"exists shape expected {(T,N)}, got {exists.shape}"
            )

    side_data = {
        "right": {
            "pose_ref": right_pose_reference,
            "pose_run": right_pose_run,
            "open_ref": right_open_reference,
            "open_run": right_open_run,
        },
        "left": {
            "pose_ref": left_pose_reference,
            "pose_run": left_pose_run,
            "open_ref": left_open_reference,
            "open_run": left_open_run,
        },
    }

    active_sides = [
        s for s in active_sides
        if s in ("right", "left")
    ]

    if not active_sides:
        return out.astype(np.float32), []

    # ------------------------------------------------------------
    # Build one fixed interaction weight per active hand.
    #
    # IMPORTANT:
    # PointWorld dist2robot itself is object-agnostic and combined
    # across the robot points.  We do NOT introduce an object mask.
    #
    # When two hands are present, the same native interaction weight
    # is softly assigned to the nearer hand frame.  No hard cutoff.
    # ------------------------------------------------------------

    records = []

    raw_alpha = []
    targets_all = []

    for side in active_sides:

        D = side_data[side]

        if (
            D["pose_ref"] is None
            or D["pose_run"] is None
            or D["open_ref"] is None
            or D["open_run"] is None
        ):
            continue

        open_ref = _as_open(D["open_ref"])
        open_run = _as_open(D["open_run"])

        t0 = _first_max_closed_frame(open_ref)

        if t0 is None:
            records.append({
                "side": side,
                "applied": False,
                "reason": "always_open",
            })
            continue

        H_ref = _pose_mats(D["pose_ref"])
        H_run = _pose_mats(D["pose_run"])

        # Fixed once, from ORIGINAL / reference PointWorld dist2robot.
        w = _soft_weight_from_official_norm(
            dref[:, t0],
            dist_mean[t0],
            dist_var[t0],
        )

        # Only points that actually exist at anchor can be anchored.
        w = w * exists[t0].astype(np.float64)

        # Continuous PointWorld gripper gate.
        # closed -> 1, open -> 0.
        gate = 1.0 - open_run

        targets = _hand_targets(
            reference_pred,
            H_ref,
            H_run,
            t0,
        )

        alpha = gate[:, None] * w[None, :]

        raw_alpha.append(alpha)
        targets_all.append(targets)

        records.append({
            "side": side,
            "applied": True,
            "anchor_frame": int(t0),
            "weight_mean": float(np.mean(w)),
            "weight_median": float(np.median(w)),
            "weight_p90": float(np.quantile(w, 0.90)),
            "weight_max": float(np.max(w)),
            # diagnostic only; not used by the method
            "n_weight_gt_050": int(np.sum(w > 0.50)),
            "n_weight_gt_075": int(np.sum(w > 0.75)),
        })

    if not raw_alpha:
        return out.astype(np.float32), records

    A = np.stack(raw_alpha, axis=0)       # [H,T,N]
    TARGET = np.stack(targets_all, axis=0) # [H,T,N,3]

    # Convex-combination safeguard.
    # If both hands assign weight to the same point, rescale their
    # total contribution so it never exceeds 1.
    sum_a = np.sum(A, axis=0)             # [T,N]
    scale = np.maximum(sum_a, 1.0)

    A = A / scale[None, :, :]
    sum_a = np.sum(A, axis=0)

    # Never modify nonexistent points.
    sum_a = sum_a * exists.astype(np.float64)
    A = A * exists[None, :, :].astype(np.float64)

    corrected = (
        (1.0 - sum_a)[..., None] * out
        +
        np.sum(
            A[..., None] * TARGET,
            axis=0,
        )
    )

    return corrected.astype(np.float32), records
