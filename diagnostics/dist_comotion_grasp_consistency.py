#!/usr/bin/env python3

import numpy as np
import transform_utils


EPS = 1e-8


def _as_open(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.clip(x, 0.0, 1.0)


def _pose_mats(pose):
    pose = np.asarray(pose, dtype=np.float64)
    return np.asarray(
        transform_utils.convert_pose_quat2mat(pose),
        dtype=np.float64,
    )


def _first_max_closed_frame(open_seq):
    """
    gripper_open:
        open   -> 1
        closed -> 0

    No hand-object distance threshold.
    """
    gate = 1.0 - _as_open(open_seq)

    if float(np.max(gate)) <= 0.0:
        return None

    return int(np.argmax(gate))


def _distance_weight(raw_d, mean, var):
    """
    Official raw dist2robot -> official normalization scale
    -> monotonic [0,1].

    No clip-specific distance threshold.
    """
    raw_d = np.asarray(raw_d, dtype=np.float64)

    sigma = np.sqrt(max(float(var), 1e-12))
    z = (raw_d - float(mean)) / sigma
    z = np.clip(z, -30.0, 30.0)

    return 1.0 / (1.0 + np.exp(z))


def _hand_targets(reference_pred, H_ref, H_run, t0):
    """
    q_i = H_ref(t0)^-1 p_i(t0)

    Returns:
      target_ref: point if carried by ORIGINAL hand
      target_run: point if carried by CORRECTED hand
    """
    p0 = np.asarray(reference_pred[t0], dtype=np.float64)

    R0 = H_ref[t0, :3, :3]
    x0 = H_ref[t0, :3, 3]

    q = (p0 - x0[None, :]) @ R0

    T = len(reference_pred)

    target_ref = np.empty_like(reference_pred, dtype=np.float64)
    target_run = np.empty_like(reference_pred, dtype=np.float64)

    for t in range(T):
        Rr = H_ref[t, :3, :3]
        xr = H_ref[t, :3, 3]

        Rc = H_run[t, :3, :3]
        xc = H_run[t, :3, 3]

        target_ref[t] = q @ Rr.T + xr[None, :]
        target_run[t] = q @ Rc.T + xc[None, :]

    return target_ref, target_run


def _comotion_weight(
    reference_pred,
    target_ref,
    open_ref,
    t0,
):
    """
    Parameter-free continuous co-motion score.

    Compare:
      Original PointWorld predicted motion
        vs
      motion expected if the point were hand-attached.

    Uses only frames where the gripper is closed, via a soft gate.

    score =
        positive cosine trajectory similarity
        *
        motion-magnitude agreement

    Both terms are in [0,1].
    """
    p = np.asarray(reference_pred, dtype=np.float64)
    h = np.asarray(target_ref, dtype=np.float64)

    T, N, _ = p.shape

    gate = 1.0 - _as_open(open_ref)

    # Only use t0 and later.
    time_w = gate.copy()
    time_w[:t0] = 0.0

    # Anchor displacement.
    dp = p - p[t0][None, :, :]
    dh = h - h[t0][None, :, :]

    # Soft temporal weighting. No frame threshold.
    sw = np.sqrt(time_w)[:, None, None]

    dpw = dp * sw
    dhw = dh * sw

    # Flatten trajectory per point: [N, T*3]
    A = np.transpose(dpw, (1, 0, 2)).reshape(N, -1)
    B = np.transpose(dhw, (1, 0, 2)).reshape(N, -1)

    dot = np.sum(A * B, axis=1)

    na = np.linalg.norm(A, axis=1)
    nb = np.linalg.norm(B, axis=1)

    cosine = dot / (na * nb + EPS)
    cosine = np.clip(cosine, 0.0, 1.0)

    # Parameter-free magnitude similarity:
    # 1 when motion magnitudes agree, 0 when very different.
    magnitude_match = (
        2.0 * np.minimum(na, nb)
        / (na + nb + EPS)
    )

    magnitude_match = np.clip(
        magnitude_match,
        0.0,
        1.0,
    )

    score = cosine * magnitude_match

    # Static/undefined motion naturally gets ~0.
    score[(na < EPS) | (nb < EPS)] = 0.0

    return score


def apply_dist_comotion_grasp_consistency(
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
    DISTANCE × CO-MOTION soft grasp consistency.

    NO:
      - object IDs
      - object masks
      - semantic segmentation
      - Kabsch
      - explicit rigid projection
      - per-clip threshold tuning

    Weight is fixed once per grasp/hand:
        w = w_dist * w_comotion

    Corrected PointWorld output remains the base:
        p_new = (1-alpha) p_corr + alpha p_hand_target
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
            f"dist2robot expected {(N,T)}, got {dref.shape}"
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
            f"bad dist norm stats: "
            f"mean={len(dist_mean)}, var={len(dist_var)}, T={T}"
        )

    if exists is None:
        exists = np.ones((T, N), dtype=bool)
    else:
        exists = np.asarray(exists, dtype=bool)

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

    all_alpha = []
    all_targets = []
    audit = []

    for side in active_sides:

        if side not in side_data:
            continue

        D = side_data[side]

        if any(
            x is None
            for x in [
                D["pose_ref"],
                D["pose_run"],
                D["open_ref"],
                D["open_run"],
            ]
        ):
            continue

        open_ref = _as_open(D["open_ref"])
        open_run = _as_open(D["open_run"])

        t0 = _first_max_closed_frame(open_ref)

        if t0 is None:
            audit.append({
                "side": side,
                "applied": False,
                "reason": "always_open",
            })
            continue

        H_ref = _pose_mats(D["pose_ref"])
        H_run = _pose_mats(D["pose_run"])

        target_ref, target_run = _hand_targets(
            reference_pred,
            H_ref,
            H_run,
            t0,
        )

        # ------------------------------------------------------
        # 1. Native PointWorld robot proximity
        # ------------------------------------------------------
        w_dist = _distance_weight(
            dref[:, t0],
            dist_mean[t0],
            dist_var[t0],
        )

        # ------------------------------------------------------
        # 2. Original PointWorld co-motion with hand-frame target
        # ------------------------------------------------------
        w_motion = _comotion_weight(
            reference_pred,
            target_ref,
            open_ref,
            t0,
        )

        # Fixed point weight.
        w = w_dist * w_motion

        # Anchor must exist.
        w *= exists[t0].astype(np.float64)

        # Corrected gripper state controls when constraint is active.
        gate_run = 1.0 - open_run

        alpha = gate_run[:, None] * w[None, :]

        all_alpha.append(alpha)
        all_targets.append(target_run)

        audit.append({
            "side": side,
            "applied": True,
            "anchor_frame": int(t0),

            "dist_mean": float(np.mean(w_dist)),
            "dist_median": float(np.median(w_dist)),

            "motion_mean": float(np.mean(w_motion)),
            "motion_median": float(np.median(w_motion)),
            "motion_p90": float(np.quantile(w_motion, .90)),

            "final_weight_mean": float(np.mean(w)),
            "final_weight_median": float(np.median(w)),
            "final_weight_p90": float(np.quantile(w, .90)),
            "final_weight_max": float(np.max(w)),

            # diagnostics only
            "n_weight_gt_025": int(np.sum(w > .25)),
            "n_weight_gt_050": int(np.sum(w > .50)),
        })

    if not all_alpha:
        return out.astype(np.float32), audit

    A = np.stack(all_alpha, axis=0)
    TARGET = np.stack(all_targets, axis=0)

    # If both hands influence one point, keep total convex weight <= 1.
    sum_a = np.sum(A, axis=0)

    scale = np.maximum(sum_a, 1.0)

    A = A / scale[None, :, :]
    sum_a = np.sum(A, axis=0)

    A *= exists[None, :, :].astype(np.float64)
    sum_a *= exists.astype(np.float64)

    corrected = (
        (1.0 - sum_a)[..., None] * out
        +
        np.sum(
            A[..., None] * TARGET,
            axis=0,
        )
    )

    return corrected.astype(np.float32), audit
