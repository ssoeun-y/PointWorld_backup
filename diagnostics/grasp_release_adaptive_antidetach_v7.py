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
    """
    True  = gripper open
    False = gripper closed
    """

    open_seq = np.asarray(
        open_seq
    ).reshape(-1).astype(bool)

    closed = ~open_seq

    segments = []
    start = None

    for t in range(
        start_t,
        len(closed),
    ):

        if closed[t] and start is None:
            start = t

        if start is not None:

            last = (
                t == len(closed) - 1
            )

            if (not closed[t]) or last:

                end = (
                    t
                    if closed[t] and last
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

    s1 = float(
        np.sum(w)
    )

    s2 = float(
        np.sum(w * w)
    )

    if s2 <= eps:
        return 0.0

    return (
        s1 * s1
        / s2
    )


def _local_scene_scale(points, valid):
    """
    Parameter-free local geometric scale.

    For every valid scene point:
        scale_i =
            distance to its nearest OTHER valid scene point.

    cKDTree query k=2:
        neighbor 0 = itself
        neighbor 1 = nearest other point

    Degenerate zero-distance entries are replaced only by the
    median positive spacing from the SAME input scene.
    No fixed metric distance is introduced.
    """

    valid = np.asarray(
        valid
    ).astype(bool)

    valid_idx = np.flatnonzero(
        valid
    )

    if len(valid_idx) < 3:
        raise RuntimeError(
            "Need at least 3 valid scene points "
            "for adaptive local scale."
        )

    p = np.asarray(
        points[
            valid_idx
        ],
        dtype=np.float64,
    )

    tree = cKDTree(p)

    dist, _ = tree.query(
        p,
        k=2,
    )

    nn = dist[:, 1]

    positive = nn[
        np.isfinite(nn)
        & (nn > 1e-12)
    ]

    if len(positive) == 0:
        raise RuntimeError(
            "Could not derive scene point spacing."
        )

    fallback = float(
        np.median(
            positive
        )
    )

    nn = np.where(
        np.isfinite(nn)
        & (nn > 1e-12),
        nn,
        fallback,
    )

    scale = np.full(
        len(points),
        np.nan,
        dtype=np.float64,
    )

    scale[
        valid_idx
    ] = nn

    return scale, {
        "median_mm":
            float(
                np.median(nn)
                * 1000.0
            ),

        "p25_mm":
            float(
                np.quantile(
                    nn,
                    0.25,
                )
                * 1000.0
            ),

        "p75_mm":
            float(
                np.quantile(
                    nn,
                    0.75,
                )
                * 1000.0
            ),
    }


def apply_adaptive_antidetach_repair_v7(
    pred_corrected,
    robot_flows,
    valid,
    *,
    right_gripper_open=None,
    left_gripper_open=None,
    right_gripper_pose=None,
    left_gripper_pose=None,

    # Legacy kwargs accepted ONLY so the existing visualizer
    # can call this module without breaking.
    # They are intentionally NOT USED in V6.
    sigma_geom_m=None,
    sigma_motion_m=None,
):
    """
    V6 adaptive contact repair.

    Contact score for scene point i:

        w_i = exp(
            -0.5 * (d_hand_i / s_local_i)^2
        )

    where:
        d_hand_i
            = distance to gripper surface

        s_local_i
            = nearest-neighbor spacing of scene point i

    Therefore:
        - no fixed contact radius
        - no fixed contact sigma
        - no object ID
        - no semantic label
        - no PA optimization

    Gripper-open state determines whether a hand can be active.
    Geometry determines continuously how strong contact is.

    During grasp:
        every frame starts from current PW prediction
        + already released residuals.

        Active contact corrections are freshly recomputed.
        They are NOT recursively accumulated.

    At release:
        the released hand's last residual contribution is
        transferred once into persistent carry.

    This prevents release snap-back without recurrent drift.
    """

    pred = _squeeze(
        pred_corrected
    ).astype(
        np.float64,
        copy=True,
    )

    robot = _squeeze(
        robot_flows
    ).astype(
        np.float64,
        copy=False,
    )

    valid = _squeeze(
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
            f"T mismatch: pred={T}, robot={robot.shape[0]}"
        )

    if valid.shape != (T, N):
        raise RuntimeError(
            f"valid={valid.shape}, expected={(T,N)}"
        )

    eps = 1e-12

    # ========================================================
    # Available hands
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

        open_seq = _squeeze(
            open_raw
        ).reshape(
            T, -1
        )[:, 0].astype(bool)

        pose = _squeeze(
            pose_raw
        ).reshape(
            T, 7
        )

        sides[
            side
        ] = {
            "open":
                open_seq,

            # Only used once to split official
            # gripper-surface samples into hands.
            "center0":
                pose[
                    0, :3
                ].astype(
                    np.float64
                ),
        }

    if not sides:
        raise RuntimeError(
            "No usable gripper information."
        )

    side_names = list(
        sides.keys()
    )

    # ========================================================
    # Assign official robot surface points to L/R ONCE.
    # ========================================================

    if len(side_names) == 1:

        robot_side = np.zeros(
            robot.shape[1],
            dtype=np.int64,
        )

    else:

        centers = np.stack(
            [
                sides[
                    side
                ][
                    "center0"
                ]
                for side
                in side_names
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
    # Gripper-closed segments.
    # ========================================================

    segments = []

    for side in side_names:

        if side not in side_robot_idx:
            continue

        for ref_t, end_t in _closed_segments(
            sides[
                side
            ][
                "open"
            ],
            start_t=1,
        ):

            segments.append(
                {
                    "id":
                        len(segments),

                    "side":
                        side,

                    "ref_t":
                        int(ref_t),

                    "end_t":
                        int(end_t),

                    "robot_side_idx":
                        side_robot_idx[
                            side
                        ],

                    "initialized":
                        False,
                }
            )

    repaired = pred.copy()

    # Residuals from COMPLETED contact episodes.
    persistent_carry = np.zeros(
        (N, 3),
        dtype=np.float64,
    )

    # Last contribution of each currently-active memory.
    last_contribution = {
        seg["id"]:
            np.zeros(
                (N, 3),
                dtype=np.float64,
            )
        for seg in segments
    }

    memories = []
    frame_rows = []

    previous_total_residual = np.zeros(
        (N, 3),
        dtype=np.float64,
    )

    # ========================================================
    # Sequential post-hoc repair.
    # ========================================================

    for t in range(
        1,
        T,
    ):

        # ----------------------------------------------------
        # Per-hand RELEASE carry.
        #
        # A memory active at t-1 but ending before t
        # contributes its final correction ONCE to persistent
        # carry.
        # ----------------------------------------------------

        released_now = []

        for seg in segments:

            sid = seg["id"]

            if (
                seg["initialized"]
                and seg["end_t"] + 1 == t
            ):

                persistent_carry += (
                    last_contribution[
                        sid
                    ]
                )

                last_contribution[
                    sid
                ][...] = 0.0

                released_now.append(
                    sid
                )

        # ----------------------------------------------------
        # Base state:
        #
        # current frozen PW prediction
        # + residual from previously completed grasps.
        # ----------------------------------------------------

        base = (
            pred[t]
            + persistent_carry
        )

        repaired[t] = (
            base.copy()
        )

        # ----------------------------------------------------
        # Memories already available and active this frame.
        # ----------------------------------------------------

        active = [
            seg
            for seg in segments
            if (
                seg[
                    "initialized"
                ]
                and
                seg[
                    "ref_t"
                ] < t
                <= seg[
                    "end_t"
                ]
            )
        ]

        if active:

            # -----------------------------------------------
            # Compute unnormalized evidence for every active
            # hand first.
            # -----------------------------------------------

            raw_data = []

            sum_w = np.zeros(
                N,
                dtype=np.float64,
            )

            for seg in active:

                anchor = robot[
                    t,
                    seg[
                        "robot_idx"
                    ],
                ]

                vec = (
                    base
                    - anchor
                )

                d_now = np.linalg.norm(
                    vec,
                    axis=1,
                )

                excess = np.maximum(
                    d_now
                    - seg[
                        "d_ref"
                    ],
                    0.0,
                )

                unit = (
                    vec
                    / np.maximum(
                        d_now[
                            :, None
                        ],
                        eps,
                    )
                )

                raw_delta = (
                    -excess[
                        :, None
                    ]
                    * unit
                )

                # -------------------------------------------
                # V7: self-normalized anti-detachment weight.
                #
                # V6 weakness:
                # sparse object / robot sampling can make the
                # absolute peak contact weight small, causing
                # severe under-correction.
                #
                # V7 preserves the relative spatial ranking of
                # the V6 adaptive kernel but normalizes its peak
                # to 1 within each grasp memory.
                #
                # IMPORTANT:
                # raw_delta is non-zero ONLY when
                # d_now > d_ref, so this does NOT create contact
                # that did not exist at the reference state.
                # It only resists additional separation.
                #
                # No fixed mm threshold.
                # No new sigma.
                # No object ID.
                # -------------------------------------------

                base_w = seg[
                    "w"
                ]

                peak_w = max(
                    float(
                        np.max(
                            base_w
                        )
                    ),
                    eps,
                )

                repair_w = np.clip(
                    base_w
                    / peak_w,
                    0.0,
                    1.0,
                )

                wt = (
                    repair_w
                    * valid[
                        t
                    ].astype(
                        np.float64
                    )
                )

                raw_data.append(
                    (
                        seg,
                        raw_delta,
                        wt,
                    )
                )

                sum_w += wt

            # -----------------------------------------------
            # Soft hand responsibility.
            #
            # If only one hand is geometrically plausible:
            #   it gets its ordinary soft weight.
            #
            # If two hands overlap:
            #   they compete instead of double-pulling.
            # -----------------------------------------------

            denom = np.maximum(
                1.0,
                sum_w,
            )

            total_active_delta = np.zeros(
                (N, 3),
                dtype=np.float64,
            )

            for (
                seg,
                raw_delta,
                wt,
            ) in raw_data:

                alpha = (
                    wt
                    / denom
                )

                contribution = (
                    alpha[
                        :, None
                    ]
                    * raw_delta
                )

                total_active_delta += (
                    contribution
                )

                last_contribution[
                    seg[
                        "id"
                    ]
                ] = (
                    contribution.copy()
                )

            repaired[t] = (
                base
                + total_active_delta
            )

        # ----------------------------------------------------
        # Initialize any new grasp memory at CURRENT repaired
        # state.
        #
        # This allows a second hand to close while another
        # contact is already active.
        # ----------------------------------------------------

        for seg in segments:

            if (
                seg[
                    "ref_t"
                ] == t
                and not seg[
                    "initialized"
                ]
            ):

                ridx = seg[
                    "robot_side_idx"
                ]

                # Parameter-free local scene resolution.
                local_scale, scale_stats = (
                    _local_scene_scale(
                        repaired[t],
                        valid[t],
                    )
                )

                robot_ref = robot[
                    t,
                    ridx,
                ]

                robot_tree = cKDTree(
                    robot_ref
                )

                d_ref, nn_local = (
                    robot_tree.query(
                        repaired[t],
                        k=1,
                    )
                )

                robot_idx = ridx[
                    nn_local
                ]

                # -------------------------------------------
                # Adaptive contact kernel:
                #
                # distance / local scene spacing
                #
                # No fixed mm radius.
                # No fixed sigma.
                # -------------------------------------------

                w = np.zeros(
                    N,
                    dtype=np.float64,
                )

                vm = valid[t]

                ratio = (
                    d_ref[
                        vm
                    ]
                    / local_scale[
                        vm
                    ]
                )

                w[
                    vm
                ] = np.exp(
                    -0.5
                    * ratio
                    * ratio
                )

                valid_d = d_ref[
                    vm
                ]

                nearest_gap_mm = float(
                    np.min(
                        valid_d
                    )
                    * 1000.0
                )

                max_contact_weight = float(
                    np.max(
                        w[
                            vm
                        ]
                    )
                )

                seg[
                    "d_ref"
                ] = d_ref

                seg[
                    "robot_idx"
                ] = robot_idx

                seg[
                    "w"
                ] = w

                seg[
                    "local_scale"
                ] = local_scale

                seg[
                    "initialized"
                ] = True

                memories.append(
                    {
                        "id":
                            int(
                                seg["id"]
                            ),

                        "side":
                            seg["side"],

                        "ref_t":
                            int(
                                seg["ref_t"]
                            ),

                        # kept for visualizer compatibility
                        "evidence_t":
                            int(
                                seg["ref_t"]
                            ),

                        "end_t":
                            int(
                                seg["end_t"]
                            ),

                        "geom_weight_sum":
                            float(
                                np.sum(w)
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
                            max_contact_weight,

                        # No motion-gate in V6.
                        "median_motion_error_mm":
                            float("nan"),

                        "nearest_gap_mm":
                            nearest_gap_mm,

                        "local_scale_median_mm":
                            scale_stats[
                                "median_mm"
                            ],

                        "local_scale_p25_mm":
                            scale_stats[
                                "p25_mm"
                            ],

                        "local_scale_p75_mm":
                            scale_stats[
                                "p75_mm"
                            ],
                    }
                )

        # ====================================================
        # Continuous separation diagnostic
        # ====================================================

        active_after_init = [
            seg
            for seg in segments
            if (
                seg[
                    "initialized"
                ]
                and
                seg[
                    "ref_t"
                ] < t
                <= seg[
                    "end_t"
                ]
            )
        ]

        num_before = 0.0
        num_after = 0.0
        den = 0.0

        for seg in active_after_init:

            anchor = robot[
                t,
                seg[
                    "robot_idx"
                ],
            ]

            d_before = np.linalg.norm(
                base
                - anchor,
                axis=1,
            )

            d_after = np.linalg.norm(
                repaired[t]
                - anchor,
                axis=1,
            )

            ex_before = np.maximum(
                d_before
                - seg[
                    "d_ref"
                ],
                0.0,
            )

            ex_after = np.maximum(
                d_after
                - seg[
                    "d_ref"
                ],
                0.0,
            )

            wt = (
                seg[
                    "w"
                ]
                * valid[
                    t
                ].astype(
                    np.float64
                )
            )

            num_before += float(
                np.sum(
                    wt
                    * ex_before
                )
            )

            num_after += float(
                np.sum(
                    wt
                    * ex_after
                )
            )

            den += float(
                np.sum(wt)
            )

        if den > eps:

            sep_before_mm = (
                num_before
                / den
                * 1000.0
            )

            sep_after_mm = (
                num_after
                / den
                * 1000.0
            )

        else:

            sep_before_mm = np.nan
            sep_after_mm = np.nan

        total_residual = (
            repaired[t]
            - pred[t]
        )

        vm = valid[t]

        if np.any(vm):

            residual_mean_mm = float(
                np.linalg.norm(
                    total_residual[
                        vm
                    ],
                    axis=1,
                ).mean()
                * 1000.0
            )

            residual_change_mm = float(
                np.linalg.norm(
                    (
                        total_residual
                        - previous_total_residual
                    )[
                        vm
                    ],
                    axis=1,
                ).mean()
                * 1000.0
            )

        else:

            residual_mean_mm = np.nan
            residual_change_mm = np.nan

        frame_rows.append(
            {
                "frame":
                    int(t),

                "active_memories":
                    int(
                        len(
                            active_after_init
                        )
                    ),

                "released_memories":
                    int(
                        len(
                            released_now
                        )
                    ),

                "release_frame":
                    bool(
                        len(
                            released_now
                        ) > 0
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

        previous_total_residual = (
            total_residual.copy()
        )

    return repaired, {
        "memories":
            memories,

        "frame_rows":
            frame_rows,

        "side_robot_point_counts":
            {
                side:
                    int(
                        len(
                            side_robot_idx.get(
                                side,
                                [],
                            )
                        )
                    )
                for side
                in side_names
            },

        # Explicitly show that V6 has no fixed metric
        # contact scale.
        "contact_scale":
            "per-point nearest-neighbor scene spacing",
    }
