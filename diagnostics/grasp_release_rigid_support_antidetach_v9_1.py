import numpy as np
from scipy.spatial import cKDTree

from grasp_release_handframe_antidetach_v8 import (
    _squeeze,
    _local_scene_scale,
    _closed_segments,
)


def _rot_xyzw(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)

    x, y, z, w = q / n

    return np.asarray([
        [
            1.0 - 2.0*(y*y + z*z),
            2.0*(x*y - z*w),
            2.0*(x*z + y*w),
        ],
        [
            2.0*(x*y + z*w),
            1.0 - 2.0*(x*x + z*z),
            2.0*(y*z - x*w),
        ],
        [
            2.0*(x*z - y*w),
            2.0*(y*z + x*w),
            1.0 - 2.0*(x*x + y*y),
        ],
    ], dtype=np.float64)


def _exact_otsu(x):
    """
    Parameter-free 1-D two-class split.
    No histogram bin count / fixed threshold.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.sort(x[np.isfinite(x)])

    if len(x) == 0:
        return np.nan

    if len(x) == 1 or x[0] == x[-1]:
        return float(x[0])

    cs = np.cumsum(x)
    n = len(x)

    k = np.arange(1, n, dtype=np.float64)

    mu0 = cs[:-1] / k
    mu1 = (cs[-1] - cs[:-1]) / (n - k)

    between = (
        k
        * (n - k)
        * (mu0 - mu1) ** 2
    )

    # Splits between identical values are invalid.
    between[x[:-1] >= x[1:]] = -np.inf

    j = int(np.argmax(between))

    return float(
        0.5 * (
            x[j]
            + x[j + 1]
        )
    )


def apply_rigid_support_antidetach_repair_v9_1(
    pred_corrected,
    robot_flows,
    valid,
    *,
    right_gripper_open=None,
    left_gripper_open=None,
    right_gripper_pose=None,
    left_gripper_pose=None,

    # legacy compatibility only
    sigma_geom_m=None,
    sigma_motion_m=None,
):
    """
    V9.1: seed-connected rigid-support anti-detachment repair.

    No:
      - object ID
      - semantic label
      - GT future
      - Original-action future target
      - fixed mm contact threshold

    Pipeline:
      1. scan each closed-hand segment
      2. acquire strongest adaptive geometric contact
      3. reject world-linked/static proximity
      4. infer rigid support from contact seeds
      5. project contact seed toward hand surface
      6. carry support rigidly in the gripper frame
      7. freeze final residual after release
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

        if open_raw is None or pose_raw is None:
            continue

        open_seq = (
            _squeeze(open_raw)
            .reshape(T, -1)[:, 0]
            .astype(bool)
        )

        pose = (
            _squeeze(pose_raw)
            .reshape(T, 7)
            .astype(np.float64)
        )

        sides[side] = {
            "open": open_seq,
            "pose": pose,
            "center0": pose[0, :3],
        }

    if not sides:
        raise RuntimeError(
            "No usable gripper information."
        )

    side_names = list(
        sides.keys()
    )

    # ========================================================
    # Split official robot surface samples L/R once.
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
    # Build closed segments.
    # ========================================================

    segments = []

    for side in side_names:

        if side not in side_robot_idx:
            continue

        for close_t, end_t in _closed_segments(
            sides[side]["open"],
            start_t=1,
        ):

            segments.append({
                "id": len(segments),
                "side": side,
                "close_t": int(close_t),
                "end_t": int(end_t),
                "robot_side_idx":
                    side_robot_idx[side],

                "accepted": False,
            })

    # ========================================================
    # PRECOMPUTE CONTACT / GRASP / RIGID SUPPORT
    #
    # This is post-hoc: whole Boundary rollout is available.
    # ========================================================

    memories = []
    rejected = []

    for seg in segments:

        side = seg["side"]
        ridx = seg["robot_side_idx"]

        close_t = seg["close_t"]
        end_t = seg["end_t"]

        # ----------------------------------------------------
        # 1. Contact acquisition:
        #    maximize adaptive contact evidence.
        # ----------------------------------------------------

        candidates = []

        for t in range(
            close_t,
            end_t + 1,
        ):

            vm = valid[t]

            if not np.any(vm):
                continue

            local_scale, _ = (
                _local_scene_scale(
                    pred[t],
                    vm,
                )
            )

            tree = cKDTree(
                robot[t, ridx]
            )

            d, nn_local = tree.query(
                pred[t],
                k=1,
            )

            w = np.zeros(
                N,
                dtype=np.float64,
            )

            ratio = (
                d[vm]
                /
                np.maximum(
                    local_scale[vm],
                    eps,
                )
            )

            w[vm] = np.exp(
                -0.5
                * ratio
                * ratio
            )

            candidates.append({
                "t": t,
                "sum_w": float(
                    np.sum(w)
                ),
                "w": w,
                "d": d,
                "nn_local": nn_local,
                "local_scale": local_scale,
            })

        if not candidates:
            rejected.append({
                "side": side,
                "reason": "no_valid_contact_candidate",
            })
            continue

        best = max(
            candidates,
            key=lambda x: x["sum_w"],
        )

        acq_t = int(
            best["t"]
        )

        w_acq = best["w"]
        d_acq = best["d"]

        local_scale = best[
            "local_scale"
        ]

        vm = valid[
            acq_t
        ]

        wv = w_acq[
            vm
        ]

        eff = float(
            np.sum(wv) ** 2
            /
            max(
                np.sum(wv * wv),
                eps,
            )
        )

        k_seed = max(
            1,
            int(np.ceil(eff)),
        )

        valid_idx = np.flatnonzero(
            vm
        )

        order = np.argsort(
            w_acq[valid_idx]
        )[::-1]

        seed_idx = valid_idx[
            order[:k_seed]
        ]

        # ----------------------------------------------------
        # 2. GRASP GATE:
        #
        # Compare seed stability in hand frame vs world frame.
        # ----------------------------------------------------

        pose_seq = sides[
            side
        ]["pose"]

        pose0 = pose_seq[
            acq_t
        ]

        c0 = pose0[:3]

        R0 = _rot_xyzw(
            pose0[3:7]
        )

        seed_ref = pred[
            acq_t,
            seed_idx,
        ]

        seed_local_ref = (
            seed_ref
            - c0[None, :]
        ) @ R0

        scale_seed = np.maximum(
            local_scale[
                seed_idx
            ],
            eps,
        )

        hand_errors = []
        world_errors = []

        for t in range(
            acq_t + 1,
            end_t + 1,
        ):

            seed_t = pred[
                t,
                seed_idx,
            ]

            pose_t = pose_seq[
                t
            ]

            ct = pose_t[:3]

            Rt = _rot_xyzw(
                pose_t[3:7]
            )

            seed_local_t = (
                seed_t
                - ct[None, :]
            ) @ Rt

            hand_errors.extend(
                (
                    np.linalg.norm(
                        seed_local_t
                        - seed_local_ref,
                        axis=1,
                    )
                    / scale_seed
                ).tolist()
            )

            world_errors.extend(
                (
                    np.linalg.norm(
                        seed_t
                        - seed_ref,
                        axis=1,
                    )
                    / scale_seed
                ).tolist()
            )

        if not hand_errors:
            rejected.append({
                "side": side,
                "acq_t": acq_t,
                "reason": "no_future_grasp_evidence",
            })
            continue

        E_hand = float(
            np.median(
                hand_errors
            )
        )

        E_world = float(
            np.median(
                world_errors
            )
        )

        if not (
            np.isfinite(E_hand)
            and
            np.isfinite(E_world)
            and
            E_hand < E_world
        ):
            rejected.append({
                "side": side,
                "acq_t": acq_t,
                "E_hand": E_hand,
                "E_world": E_world,
                "reason": "world_linked",
            })
            continue

        # ----------------------------------------------------
        # 3. RIGID-SUPPORT SCORE
        #
        # A point belongs to the contacted rigid component if
        # its pairwise distance to contact seeds is stable.
        # ----------------------------------------------------

        p0 = pred[
            acq_t
        ]

        seed0 = p0[
            seed_idx
        ]

        dist0 = np.linalg.norm(
            p0[:, None, :]
            - seed0[None, :, :],
            axis=2,
        )

        err_stack = []

        for t in range(
            acq_t + 1,
            end_t + 1,
        ):

            pt = pred[t]

            seed_t = pt[
                seed_idx
            ]

            dist_t = np.linalg.norm(
                pt[:, None, :]
                - seed_t[None, :, :],
                axis=2,
            )

            e = np.abs(
                dist_t
                - dist0
            )

            e = (
                e
                /
                np.maximum(
                    local_scale[:, None],
                    eps,
                )
            )

            good_t = (
                vm
                & valid[t]
            )

            e[
                ~good_t,
                :
            ] = np.nan

            err_stack.append(
                e
            )

        E = np.stack(
            err_stack,
            axis=0,
        )

        with np.errstate(
            invalid="ignore",
        ):
            rigid_error = np.nanmedian(
                E,
                axis=(0, 2),
            )

        rigid_score = np.zeros(
            N,
            dtype=np.float64,
        )

        good = (
            vm
            & np.isfinite(
                rigid_error
            )
        )

        rigid_score[
            good
        ] = np.exp(
            -0.5
            * rigid_error[
                good
            ] ** 2
        )

        rigid_score[
            seed_idx
        ] = 1.0

        # ----------------------------------------------------
        # 4. RIGID-SUPPORT ISOLATION
        #
        # Stage A:
        #   Exact Otsu splits high/low rigid-affinity points.
        #
        # Stage B:
        #   Among HIGH-rigidity points only, construct an
        #   adaptive spatial graph.
        #
        # Each scene point i owns a local-resolution ball
        # with radius s_i = nearest-neighbor scene spacing.
        #
        # Two candidates are connected iff their local
        # resolution neighborhoods overlap:
        #
        #       ||p_i - p_j|| <= s_i + s_j
        #
        # No fixed mm radius.
        # No manually selected graph radius.
        #
        # Finally keep ONLY graph components containing
        # actual hand-contact seeds.
        # ----------------------------------------------------

        vals = rigid_score[
            good
        ]

        tau = _exact_otsu(
            vals
        )

        high_mask = (
            good
            &
            (
                rigid_score
                >= tau
            )
        )

        # Contact seeds must always survive.
        high_mask[
            seed_idx
        ] = True

        candidate_idx = np.flatnonzero(
            high_mask
        )

        if len(candidate_idx) == 0:
            rejected.append({
                "side":
                    side,
                "acq_t":
                    acq_t,
                "reason":
                    "no_rigid_support_candidate",
            })
            continue

        candidate_xyz = p0[
            candidate_idx
        ]

        candidate_scale = np.maximum(
            local_scale[
                candidate_idx
            ],
            eps,
        )

        # ----------------------------------------------------
        # Union-Find over adaptive spatial neighborhoods.
        # ----------------------------------------------------

        M = len(
            candidate_idx
        )

        parent = np.arange(
            M,
            dtype=np.int64,
        )

        rank = np.zeros(
            M,
            dtype=np.int64,
        )

        def _find(a):
            while parent[a] != a:
                parent[a] = parent[
                    parent[a]
                ]
                a = parent[a]
            return a

        def _union(a, b):
            ra = _find(a)
            rb = _find(b)

            if ra == rb:
                return

            if rank[ra] < rank[rb]:
                parent[ra] = rb

            elif rank[ra] > rank[rb]:
                parent[rb] = ra

            else:
                parent[rb] = ra
                rank[ra] += 1

        tree_support = cKDTree(
            candidate_xyz
        )

        # Maximum candidate local scale is used ONLY to
        # retrieve possible overlapping balls efficiently.
        # Actual edge test below remains exactly s_i + s_j.
        max_scale = float(
            np.max(
                candidate_scale
            )
        )

        for i in range(M):

            search_radius = (
                candidate_scale[i]
                + max_scale
            )

            neigh = (
                tree_support.query_ball_point(
                    candidate_xyz[i],
                    r=search_radius,
                )
            )

            for j in neigh:

                if j <= i:
                    continue

                dij = float(
                    np.linalg.norm(
                        candidate_xyz[i]
                        - candidate_xyz[j]
                    )
                )

                if (
                    dij
                    <=
                    candidate_scale[i]
                    + candidate_scale[j]
                    + 1e-12
                ):
                    _union(
                        i,
                        j,
                    )

        # ----------------------------------------------------
        # Find connected components containing contact seeds.
        # ----------------------------------------------------

        global_to_local = {
            int(g):
                int(i)
            for i, g
            in enumerate(
                candidate_idx
            )
        }

        seed_roots = set()

        for g in seed_idx:

            g = int(g)

            if g not in global_to_local:
                continue

            seed_roots.add(
                _find(
                    global_to_local[g]
                )
            )

        support_mask = np.zeros(
            N,
            dtype=bool,
        )

        for i, g in enumerate(
            candidate_idx
        ):

            if _find(i) in seed_roots:
                support_mask[
                    g
                ] = True

        # Safety: contact seeds are always included.
        support_mask[
            seed_idx
        ] = True

        # HARD component membership.
        #
        # This is the key V9.1 change:
        # background low-affinity points no longer receive
        # small positive soft weights.
        support_w = (
            support_mask.astype(
                np.float64
            )
        )

        high_candidate_count = int(
            high_mask.sum()
        )

        connected_support_count = int(
            support_mask.sum()
        )

        # ----------------------------------------------------
        # 5. CONTACT PROJECTION
        #
        # Shift the inferred rigid support as ONE body so the
        # strongest contact seed reaches the gripper surface.
        #
        # No fixed target gap.
        # ----------------------------------------------------

        robot_tree = cKDTree(
            robot[
                acq_t,
                ridx,
            ]
        )

        seed_gap, seed_nn = (
            robot_tree.query(
                seed_ref,
                k=1,
            )
        )

        j = int(
            np.argmin(
                seed_gap
            )
        )

        anchor_contact = robot[
            acq_t,
            ridx[
                seed_nn[j]
            ],
        ]

        contact_shift = (
            anchor_contact
            - seed_ref[j]
        )

        projected_ref = (
            p0
            + contact_shift[
                None, :
            ]
        )

        # One rigid hand-frame geometry for all support.
        local_ref = (
            projected_ref
            - c0[
                None, :
            ]
        ) @ R0

        # ----------------------------------------------------
        # 6. Parameter-free temporal ramp:
        #    cumulative hand path / total hand path.
        # ----------------------------------------------------

        path = np.zeros(
            T,
            dtype=np.float64,
        )

        total_path = 0.0

        for t in range(
            acq_t + 1,
            end_t + 1,
        ):

            step = float(
                np.linalg.norm(
                    pose_seq[t, :3]
                    - pose_seq[t - 1, :3]
                )
            )

            total_path += step
            path[t] = total_path

        if total_path > eps:

            path[
                acq_t:end_t + 1
            ] /= total_path

        else:

            denom_t = max(
                end_t - acq_t,
                1,
            )

            for t in range(
                acq_t,
                end_t + 1,
            ):
                path[t] = (
                    t - acq_t
                ) / denom_t

        seg.update({
            "accepted": True,
            "acq_t": acq_t,
            "seed_idx": seed_idx,
            "support_w": support_w,
            "rigid_score": rigid_score,
            "support_tau": tau,
            "E_hand": E_hand,
            "E_world": E_world,
            "local_ref": local_ref,
            "path_fraction": path,
            "contact_shift": contact_shift,
        })

        memories.append({
            "id": int(seg["id"]),
            "side": side,

            # Visualizer compatibility.
            "ref_t": acq_t,
            "evidence_t": acq_t,
            "end_t": end_t,

            "closed_ref_t":
                int(close_t),

            "geom_weight_sum":
                float(
                    np.sum(w_acq)
                ),

            "joint_weight_sum":
                float(
                    np.sum(
                        support_w
                    )
                ),

            "effective_support":
                float(eff),

            "max_weight":
                float(
                    np.max(
                        w_acq
                    )
                ),

            "median_motion_error_mm":
                float("nan"),

            "grasp_E_hand":
                E_hand,

            "grasp_E_world":
                E_world,

            "seed_count":
                int(
                    len(seed_idx)
                ),

            "support_points_soft":
                float(
                    np.sum(
                        support_w
                    )
                ),

            "high_rigid_candidates":
                int(
                    high_candidate_count
                ),

            "connected_support_points":
                int(
                    connected_support_count
                ),

            # Diagnostic only.
            # Method itself never consumes object IDs.
            "support_indices":
                np.flatnonzero(
                    support_mask
                ).astype(
                    np.int64
                ).tolist(),

            "support_points_ge_05":
                int(
                    np.sum(
                        support_w >= 0.5
                    )
                ),

            "support_tau":
                float(tau),

            "contact_projection_mm":
                float(
                    np.linalg.norm(
                        contact_shift
                    )
                    * 1000.0
                ),
        })

    # ========================================================
    # APPLY REPAIR
    # ========================================================

    repaired = pred.copy()

    persistent_carry = np.zeros(
        (N, 3),
        dtype=np.float64,
    )

    last_contribution = {
        seg["id"]:
            np.zeros(
                (N, 3),
                dtype=np.float64,
            )
        for seg in segments
    }

    previous_total_residual = np.zeros(
        (N, 3),
        dtype=np.float64,
    )

    frame_rows = []

    for t in range(
        1,
        T,
    ):

        released_now = []

        for seg in segments:

            if (
                seg["accepted"]
                and
                seg["end_t"] + 1 == t
            ):

                persistent_carry += (
                    last_contribution[
                        seg["id"]
                    ]
                )

                last_contribution[
                    seg["id"]
                ][...] = 0.0

                released_now.append(
                    seg["id"]
                )

        base = (
            pred[t]
            + persistent_carry
        )

        repaired[t] = (
            base.copy()
        )

        active = [
            seg
            for seg in segments
            if (
                seg["accepted"]
                and
                seg["acq_t"] < t
                <= seg["end_t"]
            )
        ]

        if active:

            total_delta = np.zeros(
                (N, 3),
                dtype=np.float64,
            )

            sum_w = np.zeros(
                N,
                dtype=np.float64,
            )

            raw = []

            for seg in active:

                side = seg[
                    "side"
                ]

                pose_t = sides[
                    side
                ]["pose"][t]

                ct = pose_t[
                    :3
                ]

                Rt = _rot_xyzw(
                    pose_t[
                        3:7
                    ]
                )

                desired = (
                    ct[
                        None, :
                    ]
                    +
                    seg[
                        "local_ref"
                    ]
                    @ Rt.T
                )

                alpha = float(
                    seg[
                        "path_fraction"
                    ][t]
                )

                raw_delta = (
                    alpha
                    * (
                        desired
                        - base
                    )
                )

                wt = (
                    seg[
                        "support_w"
                    ]
                    * valid[t].astype(
                        np.float64
                    )
                )

                raw.append(
                    (
                        seg,
                        raw_delta,
                        wt,
                    )
                )

                sum_w += wt

            denom = np.maximum(
                1.0,
                sum_w,
            )

            for (
                seg,
                raw_delta,
                wt,
            ) in raw:

                contribution = (
                    wt[:, None]
                    / denom[:, None]
                    * raw_delta
                )

                total_delta += (
                    contribution
                )

                last_contribution[
                    seg["id"]
                ] = (
                    contribution.copy()
                )

            repaired[t] = (
                base
                + total_delta
            )

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

        # Method-internal old separation metric is intentionally
        # not reused: V9's objective is rigid support attachment.
        frame_rows.append({
            "frame":
                int(t),

            "active_memories":
                int(
                    len(active)
                ),

            "released_memories":
                int(
                    len(released_now)
                ),

            "release_frame":
                bool(
                    len(released_now)
                    > 0
                ),

            "separation_excess_before_mm":
                float("nan"),

            "separation_excess_after_mm":
                float("nan"),

            "carried_residual_mean_mm":
                residual_mean_mm,

            "residual_change_mean_mm":
                residual_change_mm,
        })

        previous_total_residual = (
            total_residual.copy()
        )

    return repaired, {
        "memories":
            memories,

        "rejected_segments":
            rejected,

        "frame_rows":
            frame_rows,

        "side_robot_point_counts": {
            side:
                int(
                    len(
                        side_robot_idx.get(
                            side,
                            [],
                        )
                    )
                )
            for side in side_names
        },
    }
