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
    True  = open
    False = closed
    """
    open_seq = np.asarray(
        open_seq
    ).reshape(-1).astype(bool)

    closed = ~open_seq

    segments = []
    start = None

    for t in range(start_t, len(closed)):

        if closed[t] and start is None:
            start = t

        if start is not None:

            is_last = (
                t == len(closed) - 1
            )

            if (not closed[t]) or is_last:

                end = (
                    t
                    if closed[t] and is_last
                    else t - 1
                )

                if end >= start:
                    segments.append(
                        (int(start), int(end))
                    )

                start = None

    return segments


def _effective_support(w, eps=1e-12):
    w = np.asarray(
        w,
        dtype=np.float64,
    )

    s1 = float(w.sum())
    s2 = float((w * w).sum())

    if s2 <= eps:
        return 0.0

    return (
        s1 * s1
    ) / s2


def apply_stateful_geomotion_repair_v4(
    pred_corrected,
    robot_flows,
    valid,
    *,
    right_gripper_open=None,
    left_gripper_open=None,
    right_gripper_pose=None,
    left_gripper_pose=None,
    sigma_geom_m=0.015,
    sigma_motion_m=0.015,
):
    """
    Stateful hand-object separation repair.

    ------------------------------------------------------------
    Contact activation:
      gripper closed
      x geometry soft evidence
      x co-motion soft evidence

    NO hard contact distance threshold.
    NO object ID.
    NO PA optimization.
    NO action modification.

    ------------------------------------------------------------
    Critical V4 change:

    During grasp:
        repaired_t = PW_t + residual_t

    After release:
        contact force stops,
        BUT residual does NOT reset to zero.

        repaired_t = PW_t + carried_residual

    Therefore there is no snap-back to the unmodified
    boundary PointWorld trajectory at release.

    If another grasp begins later, its new contact memory is
    initialized from the CURRENT repaired state.
    ------------------------------------------------------------
    """

    pred = _squeeze_batch(
        pred_corrected
    ).astype(
        np.float64,
        copy=True,
    )

    robot = _squeeze_batch(
        robot_flows
    ).astype(
        np.float64,
        copy=False,
    )

    valid = _squeeze_batch(
        valid
    ).astype(bool)

    if pred.ndim != 3:
        raise RuntimeError(
            f"pred must be [T,N,3], got {pred.shape}"
        )

    if robot.ndim != 3:
        raise RuntimeError(
            f"robot must be [T,Nr,3], got {robot.shape}"
        )

    T, N, _ = pred.shape

    if robot.shape[0] != T:
        raise RuntimeError(
            f"T mismatch pred={T} robot={robot.shape[0]}"
        )

    if valid.shape != (T, N):
        raise RuntimeError(
            f"valid={valid.shape}, expected={(T, N)}"
        )

    eps = 1e-12

    # ========================================================
    # Hands
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

        if (
            open_raw is None
            or pose_raw is None
        ):
            continue

        open_seq = _squeeze_batch(
            open_raw
        ).reshape(
            T, -1
        )[:, 0].astype(bool)

        pose = _squeeze_batch(
            pose_raw
        ).reshape(
            T, 7
        )

        sides[side] = {
            "open": open_seq,

            # Only t=0 center is used for assigning
            # official robot points to left/right.
            "center0": pose[
                0, :3
            ].astype(np.float64),
        }

    if not sides:
        raise RuntimeError(
            "No usable gripper state."
        )

    side_names = list(
        sides.keys()
    )

    # ========================================================
    # Assign official robot points to each hand ONCE at t=0
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
            robot[
                0, :, None, :
            ]
            - centers[
                None, :, :
            ],
            axis=-1,
        )

        robot_side = np.argmin(
            d_side,
            axis=1,
        )

    side_robot_idx = {}

    for si, side in enumerate(
        side_names
    ):

        ridx = np.flatnonzero(
            robot_side == si
        )

        if len(ridx) > 0:
            side_robot_idx[
                side
            ] = ridx

    # ========================================================
    # Segment definitions
    #
    # We delay actual contact-memory construction until runtime,
    # because a later grasp should use CURRENT repaired state.
    # ========================================================

    segments = []

    for side in side_names:

        if side not in side_robot_idx:
            continue

        for ref_t, end_t in _closed_segments(
            sides[side]["open"],
            start_t=1,
        ):

            evidence_t = min(
                ref_t + 1,
                end_t,
            )

            segments.append(
                {
                    "side": side,
                    "ref_t": int(ref_t),
                    "evidence_t":
                        int(evidence_t),
                    "end_t": int(end_t),
                    "robot_side_idx":
                        side_robot_idx[side],
                    "initialized":
                        False,
                    "ready":
                        False,
                }
            )

    # ========================================================
    # Sequential stateful rollout
    # ========================================================

    repaired = pred.copy()

    # Residual carried between frames.
    carried_residual = np.zeros(
        (N, 3),
        dtype=np.float64,
    )

    memories = []

    frame_rows = []

    previous_residual = (
        carried_residual.copy()
    )

    for t in range(1, T):

        # ----------------------------------------------------
        # First propagate previous corrected state.
        #
        # This is THE important release-continuity step.
        # ----------------------------------------------------

        state_before_contact = (
            pred[t]
            + carried_residual
        )

        repaired[t] = (
            state_before_contact
        )

        # ----------------------------------------------------
        # Initialize any new grasp memory at this frame
        # using CURRENT repaired state, not raw PW state.
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

                anchor_idx = ridx[
                    nn_local
                ]

                w_geom = np.exp(
                    -0.5
                    * (
                        d_ref
                        / float(
                            sigma_geom_m
                        )
                    ) ** 2
                )

                w_geom *= valid[
                    t
                ].astype(
                    np.float64
                )

                seg[
                    "d_ref"
                ] = d_ref

                seg[
                    "robot_idx"
                ] = anchor_idx

                seg[
                    "w_geom"
                ] = w_geom

                seg[
                    "ref_scene"
                ] = repaired[
                    t
                ].copy()

                seg[
                    "initialized"
                ] = True

        # ----------------------------------------------------
        # At evidence frame, calculate co-motion gate.
        #
        # Evidence is measured from the FROZEN PointWorld
        # trajectory to avoid self-reinforcing repair.
        # ----------------------------------------------------

        for seg in segments:

            if (
                seg["initialized"]
                and not seg["ready"]
                and seg[
                    "evidence_t"
                ] == t
            ):

                ref_t = seg[
                    "ref_t"
                ]

                anchor_idx = seg[
                    "robot_idx"
                ]

                if t > ref_t:

                    scene_step = (
                        pred[t]
                        - pred[ref_t]
                    )

                    robot_step = (
                        robot[
                            t,
                            anchor_idx,
                        ]
                        - robot[
                            ref_t,
                            anchor_idx,
                        ]
                    )

                    motion_error = (
                        np.linalg.norm(
                            scene_step
                            - robot_step,
                            axis=1,
                        )
                    )

                    w_motion = np.exp(
                        -0.5
                        * (
                            motion_error
                            / float(
                                sigma_motion_m
                            )
                        ) ** 2
                    )

                else:

                    motion_error = (
                        np.zeros(
                            N,
                            dtype=np.float64,
                        )
                    )

                    w_motion = (
                        np.ones(
                            N,
                            dtype=np.float64,
                        )
                    )

                w = (
                    seg["w_geom"]
                    * w_motion
                    * valid[t].astype(
                        np.float64
                    )
                )

                seg[
                    "motion_error"
                ] = motion_error

                seg[
                    "w_motion"
                ] = w_motion

                seg[
                    "w"
                ] = w

                seg[
                    "ready"
                ] = True

                mem = {
                    "side":
                        seg["side"],

                    "ref_t":
                        int(
                            seg["ref_t"]
                        ),

                    "evidence_t":
                        int(
                            seg[
                                "evidence_t"
                            ]
                        ),

                    "end_t":
                        int(
                            seg["end_t"]
                        ),

                    "geom_weight_sum":
                        float(
                            np.sum(
                                seg[
                                    "w_geom"
                                ]
                            )
                        ),

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
                        float(
                            np.median(
                                motion_error
                            )
                            * 1000.0
                        ),
                }

                memories.append(
                    mem
                )

        # ----------------------------------------------------
        # Active contact memories at this frame
        # ----------------------------------------------------

        active = [
            seg
            for seg in segments
            if (
                seg["ready"]
                and seg[
                    "evidence_t"
                ] <= t
                <= seg[
                    "end_t"
                ]
            )
        ]

        if active:

            # Start from already state-propagated position.
            base_state = (
                repaired[t].copy()
            )

            sum_delta = (
                np.zeros_like(
                    base_state
                )
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
                    base_state
                    - anchor
                )

                d_now = (
                    np.linalg.norm(
                        vec,
                        axis=1,
                    )
                )

                excess = np.maximum(
                    d_now
                    - seg["d_ref"],
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

            # Multiple hands cannot amplify
            # correction past one full correction.
            denom = np.maximum(
                1.0,
                sum_w,
            )

            repaired[t] = (
                base_state
                + sum_delta
                / denom[:, None]
            )

        # ----------------------------------------------------
        # STATE UPDATE
        #
        # Whatever correction exists at this frame becomes
        # the starting state for the next frame.
        #
        # At release this residual remains instead of
        # snapping back to zero.
        # ----------------------------------------------------

        carried_residual = (
            repaired[t]
            - pred[t]
        )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        valid_t = valid[t]

        if np.any(valid_t):

            residual_mean_mm = float(
                np.linalg.norm(
                    carried_residual[
                        valid_t
                    ],
                    axis=1,
                ).mean()
                * 1000.0
            )

            residual_change_mm = float(
                np.linalg.norm(
                    (
                        carried_residual
                        - previous_residual
                    )[
                        valid_t
                    ],
                    axis=1,
                ).mean()
                * 1000.0
            )

        else:
            residual_mean_mm = np.nan
            residual_change_mm = np.nan

        # Continuous active-contact separation metric.
        num_b = 0.0
        num_r = 0.0
        den = 0.0

        for seg in active:

            anchor = robot[
                t,
                seg["robot_idx"],
            ]

            d_b = np.linalg.norm(
                (
                    pred[t]
                    + previous_residual
                )
                - anchor,
                axis=1,
            )

            d_r = np.linalg.norm(
                repaired[t]
                - anchor,
                axis=1,
            )

            ex_b = np.maximum(
                d_b
                - seg["d_ref"],
                0.0,
            )

            ex_r = np.maximum(
                d_r
                - seg["d_ref"],
                0.0,
            )

            wt = (
                seg["w"]
                * valid[t].astype(
                    np.float64
                )
            )

            num_b += float(
                np.sum(
                    wt * ex_b
                )
            )

            num_r += float(
                np.sum(
                    wt * ex_r
                )
            )

            den += float(
                np.sum(wt)
            )

        if den > eps:

            sep_before_mm = (
                num_b / den
            ) * 1000.0

            sep_after_mm = (
                num_r / den
            ) * 1000.0

        else:

            sep_before_mm = np.nan
            sep_after_mm = np.nan

        # Is this frame immediately after any release?
        release_frame = any(
            (
                seg["end_t"] + 1
                == t
            )
            for seg in segments
        )

        frame_rows.append(
            {
                "frame":
                    int(t),

                "active_memories":
                    int(
                        len(active)
                    ),

                "release_frame":
                    bool(
                        release_frame
                    ),

                "separation_excess_before_mm":
                    float(
                        sep_before_mm
                    ),

                "separation_excess_after_mm":
                    float(
                        sep_after_mm
                    ),

                "carried_residual_mean_mm":
                    float(
                        residual_mean_mm
                    ),

                "residual_change_mean_mm":
                    float(
                        residual_change_mm
                    ),
            }
        )

        previous_residual = (
            carried_residual.copy()
        )

    return repaired, {
        "memories":
            memories,

        "frame_rows":
            frame_rows,

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
            float(
                sigma_geom_m
            ),

        "sigma_motion_m":
            float(
                sigma_motion_m
            ),
    }
