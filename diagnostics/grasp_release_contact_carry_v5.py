import numpy as np
from scipy.spatial import cKDTree


def _np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _squeeze(x):
    x = _np(x)
    if x is None:
        return None
    if x.ndim >= 3 and x.shape[0] == 1:
        x = x[0]
    return x


def _closed_segments(open_seq, start_t=1):
    open_seq = np.asarray(open_seq).reshape(-1).astype(bool)
    closed = ~open_seq

    out = []
    start = None

    for t in range(start_t, len(closed)):
        if closed[t] and start is None:
            start = t

        if start is not None:
            last = (t == len(closed) - 1)

            if (not closed[t]) or last:
                end = t if (closed[t] and last) else t - 1

                if end >= start:
                    out.append((int(start), int(end)))

                start = None

    return out


def _effective_support(w, eps=1e-12):
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))

    if s2 <= eps:
        return 0.0

    return (s1 * s1) / s2


def apply_contact_carry_repair_v5(
    pred_corrected,
    robot_flows,
    valid,
    *,
    right_gripper_open=None,
    left_gripper_open=None,
    right_gripper_pose=None,
    left_gripper_pose=None,
    sigma_geom_m=0.015,
    sigma_motion_m=None,   # kept only for visualizer compatibility
):
    """
    V5

    grasp:
      each frame is corrected DIRECTLY from current PW prediction.
      no recursive correction accumulation.

    release:
      freeze the LAST residual once and carry it forward.
      no snap-back to PW.
      no repeated integration.

    no object ID
    no hard contact threshold
    no PA optimization
    """

    pred = _squeeze(pred_corrected).astype(
        np.float64,
        copy=True,
    )

    robot = _squeeze(robot_flows).astype(
        np.float64,
        copy=False,
    )

    valid = _squeeze(valid).astype(bool)

    T, N, _ = pred.shape
    eps = 1e-12

    if robot.shape[0] != T:
        raise RuntimeError(
            f"T mismatch pred={T}, robot={robot.shape[0]}"
        )

    if valid.shape != (T, N):
        raise RuntimeError(
            f"valid={valid.shape}, expected={(T, N)}"
        )

    # ========================================================
    # Gripper states
    # ========================================================

    sides = {}

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

        open_seq = _squeeze(
            open_raw
        ).reshape(T, -1)[:, 0].astype(bool)

        pose = _squeeze(
            pose_raw
        ).reshape(T, 7)

        sides[side] = {
            "open": open_seq,
            "center0": pose[0, :3].astype(np.float64),
        }

    if not sides:
        raise RuntimeError(
            "No usable gripper state."
        )

    side_names = list(sides.keys())

    # ========================================================
    # Assign official robot-flow points to each hand at t=0
    # ========================================================

    if len(side_names) == 1:

        robot_side = np.zeros(
            robot.shape[1],
            dtype=np.int64,
        )

    else:

        centers = np.stack(
            [
                sides[s]["center0"]
                for s in side_names
            ],
            axis=0,
        )

        d_side = np.linalg.norm(
            robot[0, :, None, :]
            - centers[None, :, :],
            axis=-1,
        )

        robot_side = np.argmin(
            d_side,
            axis=1,
        )

    side_robot_idx = {}

    for si, side in enumerate(side_names):
        ridx = np.flatnonzero(
            robot_side == si
        )

        if len(ridx):
            side_robot_idx[side] = ridx

    # ========================================================
    # Closed-gripper segments
    # ========================================================

    segments = []

    for side in side_names:

        if side not in side_robot_idx:
            continue

        for ref_t, end_t in _closed_segments(
            sides[side]["open"],
            start_t=1,
        ):
            segments.append(
                {
                    "side": side,
                    "ref_t": int(ref_t),
                    "end_t": int(end_t),
                    "robot_side_idx":
                        side_robot_idx[side],
                    "initialized": False,
                }
            )

    repaired = pred.copy()

    # This residual changes ONLY when all active contact ends.
    frozen_release_residual = np.zeros(
        (N, 3),
        dtype=np.float64,
    )

    previous_total_residual = np.zeros(
        (N, 3),
        dtype=np.float64,
    )

    had_active_previous_frame = False

    memories = []
    frame_rows = []

    # ========================================================
    # Sequential evaluation
    # ========================================================

    for t in range(1, T):

        # Memories that were initialized previously and
        # are actively constraining this frame.
        active = [
            seg
            for seg in segments
            if (
                seg["initialized"]
                and seg["ref_t"] < t
                and t <= seg["end_t"]
            )
        ]

        # ----------------------------------------------------
        # RELEASE:
        #
        # Previous frame had active contact,
        # current frame has none.
        #
        # Freeze last residual ONCE.
        # ----------------------------------------------------

        release_frame = (
            had_active_previous_frame
            and len(active) == 0
        )

        if release_frame:
            frozen_release_residual = (
                previous_total_residual.copy()
            )

        # ----------------------------------------------------
        # Baseline for this frame:
        #
        # raw PointWorld prediction
        # + residual saved from the last completed grasp.
        #
        # IMPORTANT:
        # active contact residual from the previous frame is
        # NOT recursively added here.
        # ----------------------------------------------------

        base = (
            pred[t]
            + frozen_release_residual
        )

        repaired[t] = base.copy()

        # ----------------------------------------------------
        # Active grasp correction
        #
        # Minimal RADIAL separation repair only.
        # Freshly recomputed from this frame's base.
        # ----------------------------------------------------

        if active:

            sum_delta = np.zeros_like(
                base,
                dtype=np.float64,
            )

            sum_w = np.zeros(
                N,
                dtype=np.float64,
            )

            for seg in active:

                anchor = robot[
                    t,
                    seg["robot_idx"],
                ]

                vec = (
                    base - anchor
                )

                d_now = np.linalg.norm(
                    vec,
                    axis=1,
                )

                excess = np.maximum(
                    d_now - seg["d_ref"],
                    0.0,
                )

                unit = (
                    vec
                    / np.maximum(
                        d_now[:, None],
                        eps,
                    )
                )

                raw_delta = (
                    -excess[:, None]
                    * unit
                )

                wt = (
                    seg["w"]
                    * valid[t].astype(
                        np.float64
                    )
                )

                sum_delta += (
                    wt[:, None]
                    * raw_delta
                )

                sum_w += wt

            # prevents overlapping hands from amplifying
            # beyond one full correction.
            denom = np.maximum(
                1.0,
                sum_w,
            )

            repaired[t] = (
                base
                + sum_delta
                / denom[:, None]
            )

        # ----------------------------------------------------
        # Initialize NEW grasp memory AFTER existing active
        # corrections have been applied.
        #
        # This matters when second hand closes while first
        # hand is already manipulating the object.
        # ----------------------------------------------------

        for seg in segments:

            if (
                seg["ref_t"] == t
                and not seg["initialized"]
            ):

                ridx = seg[
                    "robot_side_idx"
                ]

                robot_ref = robot[
                    t,
                    ridx,
                ]

                tree = cKDTree(
                    robot_ref
                )

                d_ref, nn_local = tree.query(
                    repaired[t],
                    k=1,
                )

                robot_idx = ridx[
                    nn_local
                ]

                w = np.exp(
                    -0.5
                    * (
                        d_ref
                        / float(
                            sigma_geom_m
                        )
                    ) ** 2
                )

                w *= valid[t].astype(
                    np.float64
                )

                seg["d_ref"] = d_ref
                seg["robot_idx"] = robot_idx
                seg["w"] = w
                seg["initialized"] = True

                memories.append(
                    {
                        "side":
                            seg["side"],

                        "ref_t":
                            int(seg["ref_t"]),

                        "evidence_t":
                            int(seg["ref_t"]),

                        "end_t":
                            int(seg["end_t"]),

                        "geom_weight_sum":
                            float(
                                np.sum(w)
                            ),

                        # no fake co-motion term in V5
                        "joint_weight_sum":
                            float(
                                np.sum(w)
                            ),

                        "effective_support":
                            float(
                                _effective_support(
                                    w
                                )
                            ),

                        "max_weight":
                            float(
                                np.max(w)
                            ),

                        "median_motion_error_mm":
                            float("nan"),
                    }
                )

        # ====================================================
        # Diagnostics
        # ====================================================

        total_residual = (
            repaired[t]
            - pred[t]
        )

        vm = valid[t]

        if np.any(vm):

            residual_mean_mm = float(
                np.linalg.norm(
                    total_residual[vm],
                    axis=1,
                ).mean()
                * 1000.0
            )

            residual_change_mm = float(
                np.linalg.norm(
                    (
                        total_residual
                        - previous_total_residual
                    )[vm],
                    axis=1,
                ).mean()
                * 1000.0
            )

        else:
            residual_mean_mm = np.nan
            residual_change_mm = np.nan

        num_before = 0.0
        num_after = 0.0
        den = 0.0

        for seg in active:

            anchor = robot[
                t,
                seg["robot_idx"],
            ]

            d_before = np.linalg.norm(
                base - anchor,
                axis=1,
            )

            d_after = np.linalg.norm(
                repaired[t] - anchor,
                axis=1,
            )

            ex_before = np.maximum(
                d_before
                - seg["d_ref"],
                0.0,
            )

            ex_after = np.maximum(
                d_after
                - seg["d_ref"],
                0.0,
            )

            wt = (
                seg["w"]
                * valid[t].astype(
                    np.float64
                )
            )

            num_before += float(
                np.sum(
                    wt * ex_before
                )
            )

            num_after += float(
                np.sum(
                    wt * ex_after
                )
            )

            den += float(
                np.sum(wt)
            )

        if den > eps:

            sep_before_mm = (
                num_before / den
            ) * 1000.0

            sep_after_mm = (
                num_after / den
            ) * 1000.0

        else:

            sep_before_mm = np.nan
            sep_after_mm = np.nan

        frame_rows.append(
            {
                "frame":
                    int(t),

                "active_memories":
                    int(len(active)),

                "release_frame":
                    bool(release_frame),

                "separation_excess_before_mm":
                    float(sep_before_mm),

                "separation_excess_after_mm":
                    float(sep_after_mm),

                "carried_residual_mean_mm":
                    float(residual_mean_mm),

                "residual_change_mean_mm":
                    float(residual_change_mm),
            }
        )

        previous_total_residual = (
            total_residual.copy()
        )

        had_active_previous_frame = (
            len(active) > 0
        )

    return repaired, {
        "memories": memories,
        "frame_rows": frame_rows,

        "side_robot_point_counts":
            {
                side: int(
                    len(
                        side_robot_idx.get(
                            side,
                            [],
                        )
                    )
                )
                for side in side_names
            },

        "sigma_geom_m":
            float(sigma_geom_m),
    }
