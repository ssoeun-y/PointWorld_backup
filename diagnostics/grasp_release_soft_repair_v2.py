import numpy as np
from scipy.spatial import cKDTree


def _np(x):
    if x is None:
        return None

    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    return np.asarray(x)


def _squeeze_batch(x):
    x = _np(x)

    if x is None:
        return None

    if x.ndim >= 3 and x.shape[0] == 1:
        x = x[0]

    return x


def _closed_segments(open_seq, start_t=1):
    """
    True  = gripper open
    False = gripper closed

    Returns inclusive closed intervals [(start, end), ...].
    """
    open_seq = np.asarray(open_seq).reshape(-1).astype(bool)
    closed = ~open_seq

    T = len(closed)
    segments = []

    start = None

    for t in range(start_t, T):

        if closed[t] and start is None:
            start = t

        if start is not None:
            is_last = (t == T - 1)

            if (not closed[t]) or is_last:
                end = t if (closed[t] and is_last) else (t - 1)

                if end >= start:
                    segments.append(
                        (int(start), int(end))
                    )

                start = None

    return segments


def _effective_support(w, eps=1e-12):
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))

    if s2 <= eps:
        return 0.0

    return (s1 * s1) / s2


def apply_grasp_release_soft_repair(
    pred_corrected,
    robot_flows,
    valid,
    *,
    right_gripper_open=None,
    left_gripper_open=None,
    right_gripper_pose=None,
    left_gripper_pose=None,
    sigma_m=0.015,
):
    """
    PointWorld test-time hand-object separation repair.

    Method input:
      - corrected PointWorld prediction
      - corrected official robot/gripper flows
      - gripper open/closed state
      - t=0 gripper pose ONLY for left/right robot-point assignment

    Important:
      - NO object ID
      - NO hard contact threshold
      - NO PA optimization
      - NO action modification
      - contact weight is continuous Gaussian distance weight

    Each closed segment gets a fresh soft contact memory.
    Opening the corresponding hand terminates that memory.
    """

    pred = _squeeze_batch(pred_corrected).astype(
        np.float64,
        copy=False,
    )

    robot = _squeeze_batch(robot_flows).astype(
        np.float64,
        copy=False,
    )

    valid = _squeeze_batch(valid).astype(bool)

    if pred.ndim != 3:
        raise RuntimeError(
            f"pred shape must be [T,N,3], got {pred.shape}"
        )

    if robot.ndim != 3:
        raise RuntimeError(
            f"robot shape must be [T,Nr,3], got {robot.shape}"
        )

    T, N, _ = pred.shape

    if robot.shape[0] != T:
        raise RuntimeError(
            f"T mismatch pred={T}, robot={robot.shape[0]}"
        )

    if valid.shape != (T, N):
        raise RuntimeError(
            f"valid shape mismatch: {valid.shape}, expected {(T,N)}"
        )

    side_data = {}

    for side, open_raw, pose_raw in [
        (
            "right",
            right_gripper_open,
            right_gripper_pose,
        ),
        (
            "left",
            left_gripper_open,
            left_gripper_pose,
        ),
    ]:
        if open_raw is None or pose_raw is None:
            continue

        open_seq = _squeeze_batch(
            open_raw
        ).reshape(T, -1)[:, 0].astype(bool)

        pose = _squeeze_batch(
            pose_raw
        ).reshape(T, 7)

        side_data[side] = {
            "open": open_seq,
            # Only t=0 pose is used.
            # Boundary keeps the exact start endpoint.
            "center0": pose[0, :3].astype(np.float64),
        }

    if not side_data:
        raise RuntimeError(
            "No usable gripper open/pose fields."
        )

    # ========================================================
    # Assign each sampled robot point to left/right ONCE at t=0.
    #
    # We only use t=0 pose translations for this bookkeeping.
    # Future gripper_pose is NOT used by the repair.
    # ========================================================

    side_names = list(side_data.keys())

    if len(side_names) == 1:
        robot_side = np.zeros(
            robot.shape[1],
            dtype=np.int64,
        )

    else:
        centers = np.stack(
            [
                side_data[s]["center0"]
                for s in side_names
            ],
            axis=0,
        )

        d = np.linalg.norm(
            robot[0, :, None, :]
            - centers[None, :, :],
            axis=-1,
        )

        robot_side = np.argmin(
            d,
            axis=1,
        )

    side_robot_indices = {}

    for si, side in enumerate(side_names):
        ridx = np.flatnonzero(
            robot_side == si
        )

        if len(ridx) == 0:
            continue

        side_robot_indices[side] = ridx

    # ========================================================
    # Build one soft contact memory per CLOSED segment.
    # ========================================================

    memories = []
    eps = 1e-12

    for side in side_names:

        if side not in side_robot_indices:
            continue

        ridx = side_robot_indices[side]
        open_seq = side_data[side]["open"]

        segments = _closed_segments(
            open_seq,
            start_t=1,
        )

        for seg_start, seg_end in segments:

            ref_t = int(seg_start)

            robot_ref = robot[
                ref_t,
                ridx,
            ]

            tree = cKDTree(
                robot_ref
            )

            d_ref, local_nn = tree.query(
                pred[ref_t],
                k=1,
            )

            anchor_robot_idx = ridx[
                local_nn
            ]

            # Continuous soft weight.
            # NO radius cut.
            w = np.exp(
                -0.5
                * (
                    d_ref
                    / float(sigma_m)
                ) ** 2
            )

            # Invalid scene points have zero influence.
            w = (
                w
                * valid[ref_t].astype(np.float64)
            )

            memories.append(
                {
                    "side": side,
                    "ref_t": ref_t,
                    "end_t": int(seg_end),
                    "d_ref": d_ref,
                    "robot_idx": anchor_robot_idx,
                    "w": w,
                    "weight_sum": float(
                        np.sum(w)
                    ),
                    "effective_support": float(
                        _effective_support(w)
                    ),
                    "max_weight": float(
                        np.max(w)
                    ),
                }
            )

    repaired = pred.copy()

    # ========================================================
    # Apply minimal radial separation correction.
    #
    # Each frame starts from the frozen boundary prediction.
    # Multiple active hands are combined without double-pulling.
    # ========================================================

    for t in range(1, T):

        active = [
            m
            for m in memories
            if (
                m["ref_t"] < t
                and t <= m["end_t"]
            )
        ]

        if not active:
            continue

        base = pred[t]

        sum_delta = np.zeros_like(
            base,
            dtype=np.float64,
        )

        sum_w = np.zeros(
            N,
            dtype=np.float64,
        )

        for m in active:

            anchor = robot[
                t,
                m["robot_idx"],
            ]

            vec = (
                base - anchor
            )

            d_now = np.linalg.norm(
                vec,
                axis=1,
            )

            excess = np.maximum(
                d_now - m["d_ref"],
                0.0,
            )

            unit = (
                vec
                / np.maximum(
                    d_now[:, None],
                    eps,
                )
            )

            raw_correction = (
                -excess[:, None]
                * unit
            )

            wt = (
                m["w"]
                * valid[t].astype(np.float64)
            )

            sum_delta += (
                wt[:, None]
                * raw_correction
            )

            sum_w += wt

        # One hand:
        #   delta = w * correction
        #
        # If two hands overlap strongly:
        #   normalize instead of summing >1 correction strengths.
        denom = np.maximum(
            1.0,
            sum_w,
        )

        repaired[t] = (
            base
            + (
                sum_delta
                / denom[:, None]
            )
        )

    # ========================================================
    # Continuous separation diagnostic.
    #
    # Weighted EXCESS separation relative to contact memory:
    #
    # sum_i w_i [d_i(t)-d_i(ref)]+ / sum_i w_i
    #
    # Not a threshold metric.
    # ========================================================

    frame_rows = []

    for t in range(1, T):

        active = [
            m
            for m in memories
            if (
                m["ref_t"] <= t
                and t <= m["end_t"]
            )
        ]

        num_b = 0.0
        num_r = 0.0
        den = 0.0

        for m in active:

            anchor = robot[
                t,
                m["robot_idx"],
            ]

            d_boundary = np.linalg.norm(
                pred[t] - anchor,
                axis=1,
            )

            d_repaired = np.linalg.norm(
                repaired[t] - anchor,
                axis=1,
            )

            e_boundary = np.maximum(
                d_boundary - m["d_ref"],
                0.0,
            )

            e_repaired = np.maximum(
                d_repaired - m["d_ref"],
                0.0,
            )

            wt = (
                m["w"]
                * valid[t].astype(np.float64)
            )

            num_b += float(
                np.sum(
                    wt * e_boundary
                )
            )

            num_r += float(
                np.sum(
                    wt * e_repaired
                )
            )

            den += float(
                np.sum(wt)
            )

        if den > eps:
            sep_b_mm = (
                num_b / den
            ) * 1000.0

            sep_r_mm = (
                num_r / den
            ) * 1000.0
        else:
            sep_b_mm = np.nan
            sep_r_mm = np.nan

        valid_t = valid[t]

        if np.any(valid_t):
            repair_delta_mm = float(
                np.linalg.norm(
                    repaired[t, valid_t]
                    - pred[t, valid_t],
                    axis=1,
                ).mean()
                * 1000.0
            )
        else:
            repair_delta_mm = np.nan

        frame_rows.append(
            {
                "frame": int(t),
                "active_memories": int(
                    len(active)
                ),
                "separation_excess_boundary_mm":
                    float(sep_b_mm),
                "separation_excess_repaired_mm":
                    float(sep_r_mm),
                "repair_delta_scene_mean_mm":
                    float(repair_delta_mm),
            }
        )

    side_counts = {
        side: int(
            len(side_robot_indices.get(side, []))
        )
        for side in side_names
    }

    return repaired, {
        "memories": memories,
        "frame_rows": frame_rows,
        "side_robot_point_counts": side_counts,
        "sigma_m": float(sigma_m),
    }
