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
    w = np.asarray(w, dtype=np.float64)

    s1 = float(w.sum())
    s2 = float((w * w).sum())

    if s2 <= eps:
        return 0.0

    return (s1 * s1) / s2


def apply_geomotion_soft_repair_v3(
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
    Automatic grasp/contact-aware PointWorld repair.

    Evidence for each scene point:

        w_geom =
            exp(-d_hand^2 / (2*sigma_geom^2))

        w_motion =
            exp(-||Delta_scene - Delta_hand||^2
                / (2*sigma_motion^2))

        w_contact = w_geom * w_motion

    NO hard contact threshold.
    NO object labels.
    NO PA loss.
    NO action modification.

    Opening hand -> automatic OFF.
    Closing without geometric/co-motion evidence ->
    weights approach zero -> effectively OFF.
    """

    pred = _squeeze_batch(
        pred_corrected
    ).astype(np.float64)

    robot = _squeeze_batch(
        robot_flows
    ).astype(np.float64)

    valid = _squeeze_batch(
        valid
    ).astype(bool)

    T, N, _ = pred.shape

    if robot.shape[0] != T:
        raise RuntimeError(
            f"T mismatch: pred={T}, robot={robot.shape[0]}"
        )

    if valid.shape != (T, N):
        raise RuntimeError(
            f"valid={valid.shape}, expected={(T,N)}"
        )

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
        if open_raw is None or pose_raw is None:
            continue

        open_seq = _squeeze_batch(
            open_raw
        ).reshape(T, -1)[:, 0].astype(bool)

        pose = _squeeze_batch(
            pose_raw
        ).reshape(T, 7)

        sides[side] = {
            "open": open_seq,
            "center0": pose[
                0, :3
            ].astype(np.float64),
        }

    if not sides:
        raise RuntimeError(
            "No gripper side information available."
        )

    side_names = list(
        sides.keys()
    )

    # ========================================================
    # Assign official robot-flow points to hands ONCE.
    #
    # Uses only t=0 gripper pose.
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

        dist_side = np.linalg.norm(
            robot[
                0, :, None, :
            ]
            - centers[
                None, :, :
            ],
            axis=-1,
        )

        robot_side = np.argmin(
            dist_side,
            axis=1,
        )

    side_robot_idx = {}

    for si, side in enumerate(
        side_names
    ):
        ridx = np.flatnonzero(
            robot_side == si
        )

        if len(ridx):
            side_robot_idx[
                side
            ] = ridx

    # ========================================================
    # Build soft grasp memories.
    # ========================================================

    memories = []

    for side in side_names:

        if side not in side_robot_idx:
            continue

        ridx = side_robot_idx[
            side
        ]

        segments = _closed_segments(
            sides[side]["open"],
            start_t=1,
        )

        for start_t, end_t in segments:

            ref_t = int(start_t)

            evidence_t = min(
                ref_t + 1,
                int(end_t),
            )

            # -----------------------------------------------
            # GEOMETRY evidence
            # -----------------------------------------------

            robot_ref = robot[
                ref_t,
                ridx,
            ]

            tree = cKDTree(
                robot_ref
            )

            d_ref, nn_local = tree.query(
                pred[ref_t],
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

            # -----------------------------------------------
            # CO-MOTION evidence
            #
            # Frozen PointWorld prediction only.
            # No repair has happened yet here.
            # -----------------------------------------------

            if evidence_t > ref_t:

                scene_step = (
                    pred[evidence_t]
                    - pred[ref_t]
                )

                robot_step = (
                    robot[
                        evidence_t,
                        anchor_idx,
                    ]
                    - robot[
                        ref_t,
                        anchor_idx,
                    ]
                )

                motion_error = np.linalg.norm(
                    scene_step
                    - robot_step,
                    axis=1,
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
                motion_error = np.zeros(
                    N,
                    dtype=np.float64,
                )

                w_motion = np.ones(
                    N,
                    dtype=np.float64,
                )

            w = (
                w_geom
                * w_motion
                * valid[
                    ref_t
                ].astype(np.float64)
            )

            if evidence_t > ref_t:
                w *= valid[
                    evidence_t
                ].astype(np.float64)

            memories.append(
                {
                    "side": side,
                    "ref_t": ref_t,
                    "evidence_t":
                        int(evidence_t),
                    "end_t": int(end_t),

                    "d_ref": d_ref,
                    "robot_idx":
                        anchor_idx,

                    "w_geom":
                        w_geom,

                    "w_motion":
                        w_motion,

                    "w":
                        w,

                    "geom_weight_sum":
                        float(
                            w_geom.sum()
                        ),

                    "joint_weight_sum":
                        float(
                            w.sum()
                        ),

                    "effective_support":
                        float(
                            _effective_support(
                                w
                            )
                        ),

                    "max_weight":
                        float(
                            w.max()
                        ),

                    "median_motion_error_mm":
                        float(
                            np.median(
                                motion_error
                            )
                            * 1000.0
                        ),
                }
            )

    # ========================================================
    # Minimal radial separation repair.
    #
    # Starts only when geometry + co-motion evidence exists.
    # Since weights are continuous, no threshold is needed.
    # ========================================================

    repaired = pred.copy()

    eps = 1e-12

    for t in range(1, T):

        active = [
            m
            for m in memories
            if (
                m["evidence_t"] <= t
                <= m["end_t"]
            )
        ]

        if not active:
            continue

        base = pred[t]

        sum_delta = np.zeros_like(
            base
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
                d_now
                - m["d_ref"],
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
                m["w"]
                * valid[t].astype(
                    np.float64
                )
            )

            sum_delta += (
                wt[:, None]
                * raw_delta
            )

            sum_w += wt

        # Do not let overlapping hands amplify
        # total correction above one full correction.
        denom = np.maximum(
            1.0,
            sum_w,
        )

        repaired[t] = (
            base
            + sum_delta
            / denom[:, None]
        )

    # ========================================================
    # Framewise diagnostic
    # ========================================================

    frame_rows = []

    for t in range(1, T):

        active = [
            m
            for m in memories
            if (
                m["evidence_t"] <= t
                <= m["end_t"]
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

            d_b = np.linalg.norm(
                pred[t] - anchor,
                axis=1,
            )

            d_r = np.linalg.norm(
                repaired[t] - anchor,
                axis=1,
            )

            ex_b = np.maximum(
                d_b - m["d_ref"],
                0.0,
            )

            ex_r = np.maximum(
                d_r - m["d_ref"],
                0.0,
            )

            wt = (
                m["w"]
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
            sep_b = (
                num_b / den
            ) * 1000.0

            sep_r = (
                num_r / den
            ) * 1000.0
        else:
            sep_b = np.nan
            sep_r = np.nan

        vm = valid[t]

        if np.any(vm):
            delta_scene = float(
                np.linalg.norm(
                    repaired[t, vm]
                    - pred[t, vm],
                    axis=1,
                ).mean()
                * 1000.0
            )
        else:
            delta_scene = np.nan

        frame_rows.append(
            {
                "frame": int(t),

                "active_memories":
                    int(len(active)),

                "separation_excess_boundary_mm":
                    float(sep_b),

                "separation_excess_repaired_mm":
                    float(sep_r),

                "repair_delta_scene_mean_mm":
                    float(delta_scene),
            }
        )

    return repaired, {
        "memories": memories,
        "frame_rows": frame_rows,
        "side_robot_point_counts": {
            s: int(
                len(
                    side_robot_idx.get(
                        s, []
                    )
                )
            )
            for s in side_names
        },
        "sigma_geom_m":
            float(sigma_geom_m),
        "sigma_motion_m":
            float(sigma_motion_m),
    }
