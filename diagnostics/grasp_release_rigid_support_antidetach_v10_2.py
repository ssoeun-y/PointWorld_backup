import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as _V98Rotation

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



def _fit_robot_surface_se3(A, B):
    """
    Proper rigid transform A -> B from corresponding
    official robot surface samples.

    Returns:
        ca, cb, R

    Row-vector application:
        Y = (X - ca) @ R.T + cb
    """

    A = np.asarray(
        A,
        dtype=np.float64,
    )

    B = np.asarray(
        B,
        dtype=np.float64,
    )

    if (
        A.ndim != 2
        or
        B.ndim != 2
        or
        A.shape != B.shape
        or
        A.shape[1] != 3
    ):
        raise RuntimeError(
            f"bad robot SE3 shapes: {A.shape}, {B.shape}"
        )

    ca = A.mean(
        axis=0
    )

    cb = B.mean(
        axis=0
    )

    X = A - ca
    Y = B - cb

    H = X.T @ Y

    U, _, Vt = np.linalg.svd(
        H
    )

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    return ca, cb, R


def _apply_robot_surface_se3(
    X,
    A,
    B,
):
    ca, cb, R = (
        _fit_robot_surface_se3(
            A,
            B,
        )
    )

    return (
        (
            np.asarray(
                X,
                dtype=np.float64,
            )
            - ca[None, :]
        )
        @ R.T
        + cb[None, :]
    )


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


def apply_rigid_support_antidetach_repair_v10_2(
    pred_corrected,
    robot_flows,
    valid,
    *,
    robot_link_labels=None,
    right_gripper_open=None,
    left_gripper_open=None,
    right_gripper_pose=None,
    left_gripper_pose=None,

    # legacy compatibility only
    sigma_geom_m=None,
    sigma_motion_m=None,
):
    """
    V9.2: phase-consistent self-release rigid-support repair.

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
        # 2. GRASP GATE — corrected robot-surface SE(3)
        #
        # Compare:
        #
        # H_hand:
        #   contact seeds follow corrected robot surface SE(3)
        #
        # H_world:
        #   seeds remain closer to their acquisition world pose
        #
        # No object ID.
        # No original future object trajectory.
        # ----------------------------------------------------

        seed_ref = pred[
            acq_t,
            seed_idx,
        ]

        scale_seed = np.maximum(
            local_scale[
                seed_idx
            ],
            eps,
        )

        hand_errors = []
        world_errors = []

        robot_ref_gate = robot[
            acq_t,
            ridx,
        ]

        for t in range(
            acq_t + 1,
            end_t + 1,
        ):

            good_seed = valid[
                t,
                seed_idx
            ]

            if not np.any(
                good_seed
            ):
                continue

            ids_local = np.flatnonzero(
                good_seed
            )

            seed_t = pred[
                t,
                seed_idx[
                    ids_local
                ],
            ]

            seed_ref_t = seed_ref[
                ids_local
            ]

            scale_t = scale_seed[
                ids_local
            ]

            hand_expected = (
                _apply_robot_surface_se3(
                    seed_ref_t,
                    robot_ref_gate,
                    robot[
                        t,
                        ridx,
                    ],
                )
            )

            hand_errors.extend(
                (
                    np.linalg.norm(
                        seed_t
                        - hand_expected,
                        axis=1,
                    )
                    /
                    scale_t
                ).tolist()
            )

            world_errors.extend(
                (
                    np.linalg.norm(
                        seed_t
                        - seed_ref_t,
                        axis=1,
                    )
                    /
                    scale_t
                ).tolist()
            )

        if not hand_errors:
            rejected.append({
                "side":
                    side,
                "acq_t":
                    acq_t,
                "reason":
                    "no_future_grasp_evidence",
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
                "side":
                    side,
                "acq_t":
                    acq_t,
                "E_hand":
                    E_hand,
                "E_world":
                    E_world,
                "reason":
                    "world_linked",
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
        # V9.2 compatibility field only.
        # Actual rigid transport below is computed directly
        # from corrected official robot-surface SE(3).
        local_ref = (
            projected_ref.copy()
        )

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
                np.mean(
                    np.linalg.norm(
                        robot[
                            t,
                            ridx,
                        ]
                        -
                        robot[
                            t - 1,
                            ridx,
                        ],
                        axis=1,
                    )
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


        # ====================================================
        # V10.2: PHASE-SYNCHRONIZED RELEASE
        #
        # Grasp/support inference is unchanged.
        # Release follows the retimed discrete gripper command.
        #
        # `end_t` comes from the closed segment built from
        # gripper_open, which is phase-retimed with ZOH.
        #
        # No prediction-derived early release.
        # No original future object trajectory.
        # No final-goal anchor.
        # ====================================================

        auto_release_t = int(end_t)

        # Compatibility diagnostics.
        step_advantage = {}

        # Positive because accepted grasp already satisfies
        # E_hand < E_world. Used only for carrier ranking.
        carrier_score = float(
            E_world - E_hand
        )

        seg[
            "auto_release_t"
        ] = int(
            auto_release_t
        )

        seg[
            "step_advantage"
        ] = step_advantage

        seg[
            "carrier_score"
        ] = carrier_score

        # IMPORTANT:
        # `path` was already computed above using the complete
        # retimed closed interval [acq_t, end_t].
        seg[
            "path_fraction"
        ] = path

        memories.append({
            "id": int(seg["id"]),
            "side": side,

            # Visualizer compatibility.
            "ref_t": acq_t,
            "evidence_t": acq_t,
            "end_t":
                int(
                    seg["auto_release_t"]
                ),

            "command_end_t":
                int(
                    end_t
                ),

            "auto_release_t":
                int(
                    seg["auto_release_t"]
                ),

            "carrier_score":
                float(
                    seg["carrier_score"]
                ),

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
    # MERGE OVERLAPPING HAND MEMORIES INTO ONE RIGID BODY
    # ========================================================

    accepted_segments = [
        seg
        for seg in segments
        if (
            seg.get(
                "accepted",
                False,
            )
            and
            seg.get(
                "auto_release_t",
                seg.get(
                    "acq_t",
                    0,
                )
            )
            >
            seg.get(
                "acq_t",
                0,
            )
        )
    ]

    K = len(
        accepted_segments
    )

    gp = np.arange(
        K,
        dtype=np.int64,
    )

    def _gf(i):
        while gp[i] != i:
            gp[i] = gp[
                gp[i]
            ]
            i = gp[i]
        return i

    def _gu(i, j):
        a = _gf(i)
        b = _gf(j)

        if a != b:
            gp[b] = a

    for i in range(K):

        mi = (
            accepted_segments[i][
                "support_w"
            ]
            > 0.0
        )

        for j in range(
            i + 1,
            K,
        ):

            mj = (
                accepted_segments[j][
                    "support_w"
                ]
                > 0.0
            )

            if np.any(
                mi & mj
            ):
                _gu(
                    i,
                    j,
                )

    grouped = {}

    for i, seg in enumerate(
        accepted_segments
    ):

        root = _gf(i)

        grouped.setdefault(
            root,
            [],
        ).append(
            seg
        )

    rigid_groups = []

    for gid, members in enumerate(
        grouped.values()
    ):

        gmask = np.zeros(
            N,
            dtype=bool,
        )

        for seg in members:

            gmask |= (
                seg[
                    "support_w"
                ]
                > 0.0
            )

        # For every possible carrier, define the SAME rigid
        # group's acquisition geometry.
        for seg in members:

            at = int(
                seg[
                    "acq_t"
                ]
            )

            # =================================================
            # V10.2: GRASP-GEOMETRY ADAPTIVE TRANSPORT
            #
            # Existing V9.3 grasp/contact gate already established
            # that this rigid component is hand-linked.
            #
            # Here we ONLY decide how strongly to attach it:
            #
            #   enclosed by the two fingers
            #       -> V9.3 contact-projected transport
            #
            #   one-sided / handoff-like contact
            #       -> V9.4 acquisition-offset-preserving transport
            #
            # No object ID.
            # No future GT.
            # No distance threshold.
            # =================================================

            group_ref = (
                pred[
                    at
                ].copy()
            )

            transport_mode = (
                "offset_preserve"
            )

            pinch_alpha = float(
                "nan"
            )

            if robot_link_labels is not None:

                labels_v95 = np.asarray(
                    robot_link_labels,
                    dtype=str,
                ).reshape(-1)

                if (
                    len(labels_v95)
                    != robot.shape[1]
                ):
                    raise RuntimeError(
                        "V10.2 robot-link label mismatch: "
                        f"{len(labels_v95)} vs "
                        f"{robot.shape[1]}"
                    )

                labels_lower = np.char.lower(
                    labels_v95
                )

                side_v95 = str(
                    seg["side"]
                ).lower()

                finger1_mask = np.array([
                    (
                        side_v95 in x
                        and
                        "finger" in x
                        and
                        (
                            "link1" in x
                            or
                            "finger1" in x
                        )
                    )
                    for x in labels_lower
                ])

                finger2_mask = np.array([
                    (
                        side_v95 in x
                        and
                        "finger" in x
                        and
                        (
                            "link2" in x
                            or
                            "finger2" in x
                        )
                    )
                    for x in labels_lower
                ])

                support_at = (
                    gmask
                    &
                    valid[
                        at
                    ]
                )

                if (
                    np.any(
                        support_at
                    )
                    and
                    np.any(
                        finger1_mask
                    )
                    and
                    np.any(
                        finger2_mask
                    )
                ):

                    c_obj = (
                        pred[
                            at,
                            support_at,
                        ]
                        .mean(
                            axis=0
                        )
                    )

                    c_f1 = (
                        robot[
                            at,
                            finger1_mask,
                        ]
                        .mean(
                            axis=0
                        )
                    )

                    c_f2 = (
                        robot[
                            at,
                            finger2_mask,
                        ]
                        .mean(
                            axis=0
                        )
                    )

                    finger_axis = (
                        c_f2
                        -
                        c_f1
                    )

                    axis_sq = float(
                        np.dot(
                            finger_axis,
                            finger_axis,
                        )
                    )

                    if axis_sq > 1e-12:

                        pinch_alpha = float(
                            np.dot(
                                c_obj
                                -
                                c_f1,
                                finger_axis,
                            )
                            /
                            axis_sq
                        )

                        # Existing grasp gate already provides
                        # contact evidence.
                        #
                        # This check only asks:
                        # is the contacted rigid body geometrically
                        # enclosed between the two fingers?
                        if (
                            0.0
                            <=
                            pinch_alpha
                            <=
                            1.0
                        ):

                            transport_mode = (
                                "enclosed_contact"
                            )

            if (
                transport_mode
                ==
                "enclosed_contact"
            ):

                # V9.3 behavior:
                # project rigid body toward the contacted
                # robot surface, then transport rigidly.
                group_ref[
                    gmask
                ] += (
                    seg[
                        "contact_shift"
                    ][
                        None, :
                    ]
                )

            # Else:
            # V9.4 behavior.
            # Keep the actual acquisition hand-object offset.

            seg[
                "transport_mode"
            ] = transport_mode

            seg[
                "pinch_alpha"
            ] = pinch_alpha

            seg[
                "group_ref_world"
            ] = group_ref

            print(
                "[V10.2_TRANSPORT] "
                f"side={seg['side']} "
                f"acq={at} "
                f"mode={transport_mode} "
                f"pinch_alpha={pinch_alpha:.6f} "
                f"support={int(np.sum(gmask))}",
                flush=True,
            )

            seg[
                "group_id"
            ] = gid

        rigid_groups.append({
            "id":
                gid,

            "members":
                members,

            "support_mask":
                gmask,
        })

    # Add grouping diagnostics to memory rows.
    memory_by_id = {
        int(m["id"]):
            m
        for m in memories
    }

    for group in rigid_groups:

        for seg in group[
            "members"
        ]:

            mid = int(
                seg["id"]
            )

            if mid in memory_by_id:

                memory_by_id[
                    mid
                ][
                    "group_id"
                ] = int(
                    group["id"]
                )

                memory_by_id[
                    mid
                ][
                    "group_support_points"
                ] = int(
                    np.sum(
                        group[
                            "support_mask"
                        ]
                    )
                )

    # ========================================================
    # APPLY V9.2 REPAIR
    # ========================================================

    repaired = pred.copy()

    persistent_carry = np.zeros(
        (N, 3),
        dtype=np.float64,
    )

    last_group_contribution = {
        group["id"]:
            np.zeros(
                (N, 3),
                dtype=np.float64,
            )
        for group in rigid_groups
    }

    group_was_active = {
        group["id"]:
            False
        for group in rigid_groups
    }

    # ========================================================
    # V10.2 SELECTIVE STATE
    #
    # enclosed_contact:
    #   strong two-finger grasp evidence.
    #   Use the existing rigid V9.9 correction.
    #
    # offset_preserve:
    #   one-sided / handoff-like interaction.
    #   Repair only after meaningful detachment evidence.
    # ========================================================

    group_repair_active = {
        group["id"]:
            False
        for group in rigid_groups
    }

    group_activation_alpha = {
        group["id"]:
            None
        for group in rigid_groups
    }

    group_gate_carrier_id = {
        group["id"]:
            None
        for group in rigid_groups
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

        # ----------------------------------------------------
        # Determine active group members first.
        # ----------------------------------------------------

        active_map = {}

        for group in rigid_groups:

            members = [
                seg
                for seg in group[
                    "members"
                ]
                if (
                    seg[
                        "acq_t"
                    ]
                    < t
                    <= seg[
                        "auto_release_t"
                    ]
                )
            ]

            active_map[
                group["id"]
            ] = members

        released_now = []

        # ----------------------------------------------------
        # When NO hand remains linked to a rigid group,
        # stop following the hand.
        #
        # Keep the last residual once, so there is no
        # artificial snap-back to Boundary prediction.
        # ----------------------------------------------------

        for group in rigid_groups:

            gid = group[
                "id"
            ]

            active_now = bool(
                active_map[
                    gid
                ]
            )

            if (
                group_was_active[
                    gid
                ]
                and
                not active_now
            ):

                persistent_carry += (
                    last_group_contribution[
                        gid
                    ]
                )

                last_group_contribution[
                    gid
                ][...] = 0.0

                released_now.append(
                    gid
                )

        base = (
            pred[
                t
            ]
            +
            persistent_carry
        )

        repaired[
            t
        ] = base.copy()

        total_delta = np.zeros(
            (N, 3),
            dtype=np.float64,
        )

        active_memory_count = 0

        # ----------------------------------------------------
        # ONE transform per rigid group.
        # ----------------------------------------------------

        for group in rigid_groups:

            gid = group[
                "id"
            ]

            members = active_map[
                gid
            ]

            if not members:

                group_was_active[
                    gid
                ] = False

                continue

            active_memory_count += len(
                members
            )

            # Choose the hand whose attachment interval has
            # the strongest cumulative evidence.
            carrier = max(
                members,
                key=lambda seg:
                    (
                        float(
                            seg.get(
                                "carrier_score",
                                0.0,
                            )
                        ),
                        -int(
                            seg[
                                "acq_t"
                            ]
                        ),
                    ),
            )

            at = int(
                carrier[
                    "acq_t"
                ]
            )

            ridx = carrier[
                "robot_side_idx"
            ]

            transport_mode = carrier.get(
                "transport_mode",
                "offset_preserve",
            )

            # =============================================
            # V10.2 MINIMAL RIGID TRANSPORT
            #
            # Grasp geometry has already selected the reference:
            #
            # enclosed_contact:
            #   contact-projected rigid reference
            #
            # offset_preserve:
            #   acquisition-offset rigid reference
            #
            # Unlike V9.8, DO NOT fully lock the object to the
            # gripper pose. Move the selected reference using
            # the observed robot-surface rigid SE(3).
            #
            # The strict rigid update below remains unchanged,
            # so the supported object cannot stretch/shear.
            # =============================================

            desired = (
                _apply_robot_surface_se3(
                    carrier[
                        "group_ref_world"
                    ],
                    robot[
                        at,
                        ridx,
                    ],
                    robot[
                        t,
                        ridx,
                    ],
                )
            )

            alpha = float(
                carrier[
                    "path_fraction"
                ][
                    t
                ]
            )

            ids = (
                group[
                    "support_mask"
                ]
                &
                valid[
                    t
                ]
            )

            # =============================================
            # V10.2 GENERAL INTERACTION GATE
            #
            # IMPORTANT:
            #
            # This is NOT Can-vs-Radio logic.
            #
            # The branch is determined only from the grasp
            # geometry already estimated above.
            #
            # enclosed_contact:
            #   object lies between the two fingers and has
            #   already passed the grasp/contact gate.
            #   -> strong grasp evidence
            #   -> keep V9.9 rigid correction.
            #
            # offset_preserve:
            #   one-sided / handoff-like relation.
            #   -> intervene only when the hand has moved by
            #      more than the local geometric resolution
            #      AND the object prediction stays closer to
            #      its old world position than to the
            #      hand-carried position.
            #
            # No GT.
            # No object ID.
            # No object class.
            # No fixed millimeter threshold.
            # =============================================

            carrier_id = int(
                carrier[
                    "id"
                ]
            )

            if (
                group_gate_carrier_id[
                    gid
                ]
                !=
                carrier_id
            ):

                group_gate_carrier_id[
                    gid
                ] = carrier_id

                group_repair_active[
                    gid
                ] = False

                group_activation_alpha[
                    gid
                ] = None

            local_geom_scale = float(
                "nan"
            )

            hand_motion = float(
                "nan"
            )

            e_hand = float(
                "nan"
            )

            e_world = float(
                "nan"
            )

            trigger_now = False

            # -------------------------------------------------
            # A. Strong enclosed grasp:
            #    use V9.9 correction directly.
            # -------------------------------------------------

            if (
                transport_mode
                ==
                "enclosed_contact"
            ):

                group_repair_active[
                    gid
                ] = True

                # V10.2 HARD GRASP LOCK
                #
                # enclosed_contact has already supplied strong
                # two-finger grasp evidence. Apply the complete
                # rigid hand-carried transform rather than only
                # a path-fraction interpolation.
                alpha_apply = 1.0

                gate_reason = (
                    "enclosed_hard_lock"
                )

            # -------------------------------------------------
            # B. One-sided / handoff:
            #    selective anti-detachment gate.
            # -------------------------------------------------

            else:

                gate_ids = (
                    ids
                    &
                    valid[
                        at
                    ]
                )

                if (
                    int(
                        np.sum(
                            gate_ids
                        )
                    )
                    >= 3
                ):

                    world_ref = np.asarray(
                        pred[
                            at
                        ],
                        dtype=np.float64,
                    )

                    Wg = np.asarray(
                        world_ref[
                            gate_ids
                        ],
                        dtype=np.float64,
                    )

                    # -----------------------------------------
                    # Local geometric resolution:
                    # median nearest-neighbor spacing of this
                    # rigid support at acquisition.
                    #
                    # This replaces an arbitrary fixed-mm
                    # motion threshold.
                    # -----------------------------------------

                    tree_scale = cKDTree(
                        Wg
                    )

                    nn_dist = (
                        tree_scale.query(
                            Wg,
                            k=2,
                        )[0][
                            :,
                            1
                        ]
                    )

                    nn_dist = nn_dist[
                        np.isfinite(
                            nn_dist
                        )
                        &
                        (
                            nn_dist
                            >
                            1.0e-12
                        )
                    ]

                    if len(
                        nn_dist
                    ) > 0:

                        local_geom_scale = float(
                            np.median(
                                nn_dist
                            )
                        )

                    hand_ref = (
                        _apply_robot_surface_se3(
                            world_ref,
                            robot[
                                at,
                                ridx,
                            ],
                            robot[
                                t,
                                ridx,
                            ],
                        )
                    )

                    Xg = np.asarray(
                        base[
                            gate_ids
                        ],
                        dtype=np.float64,
                    )

                    Hg = np.asarray(
                        hand_ref[
                            gate_ids
                        ],
                        dtype=np.float64,
                    )

                    hand_motion = float(
                        np.median(
                            np.linalg.norm(
                                Hg
                                -
                                Wg,
                                axis=1,
                            )
                        )
                    )

                    e_hand = float(
                        np.median(
                            np.linalg.norm(
                                Xg
                                -
                                Hg,
                                axis=1,
                            )
                        )
                    )

                    e_world = float(
                        np.median(
                            np.linalg.norm(
                                Xg
                                -
                                Wg,
                                axis=1,
                            )
                        )
                    )

                    trigger_now = bool(
                        np.isfinite(
                            local_geom_scale
                        )
                        and
                        hand_motion
                        >
                        local_geom_scale
                        and
                        e_world
                        <
                        e_hand
                    )

                if (
                    trigger_now
                    and
                    not group_repair_active[
                        gid
                    ]
                ):

                    group_repair_active[
                        gid
                    ] = True

                    group_activation_alpha[
                        gid
                    ] = float(
                        alpha
                    )

                if group_repair_active[
                    gid
                ]:

                    a0 = float(
                        group_activation_alpha[
                            gid
                        ]
                    )

                    if (
                        a0
                        <
                        1.0
                        -
                        1.0e-12
                    ):

                        u = float(
                            np.clip(
                                (
                                    alpha
                                    -
                                    a0
                                )
                                /
                                (
                                    1.0
                                    -
                                    a0
                                ),
                                0.0,
                                1.0,
                            )
                        )

                        # Smoothstep:
                        # no instantaneous pull at activation.
                        alpha_apply = float(
                            u
                            *
                            u
                            *
                            (
                                3.0
                                -
                                2.0
                                *
                                u
                            )
                        )

                    else:

                        alpha_apply = 0.0

                else:

                    alpha_apply = 0.0

                gate_reason = (
                    "offset_detach"
                    if trigger_now
                    else
                    "offset_preserve"
                )

            print(
                "[V10.2_GATE] "
                f"frame={t} "
                f"group={gid} "
                f"side={carrier['side']} "
                f"mode={transport_mode} "
                f"reason={gate_reason} "
                f"local_scale_mm="
                f"{1000.0 * local_geom_scale:.6f} "
                f"hand_motion_mm="
                f"{1000.0 * hand_motion:.6f} "
                f"E_hand_mm="
                f"{1000.0 * e_hand:.6f} "
                f"E_world_mm="
                f"{1000.0 * e_world:.6f} "
                f"trigger={int(trigger_now)} "
                f"active="
                f"{int(group_repair_active[gid])} "
                f"alpha_raw={alpha:.6f} "
                f"alpha_apply={alpha_apply:.6f}",
                flush=True,
            )

            contribution = np.zeros(
                (N, 3),
                dtype=np.float64,
            )

            # =============================================
            # V10.2 STRICT RIGID UPDATE — ALL MODES
            #
            # Same mathematical rule for:
            #   enclosed_contact
            #   offset_preserve
            #
            # The target differs, but the object update is
            # always ONE SE(3), never per-point pulling.
            # =============================================

            if np.any(
                ids
            ):

                X = np.asarray(
                    base[
                        ids
                    ],
                    dtype=np.float64,
                )

                Y = np.asarray(
                    desired[
                        ids
                    ],
                    dtype=np.float64,
                )

                if len(
                    X
                ) >= 3:

                    ca, cb, R_full = (
                        _fit_robot_surface_se3(
                            X,
                            Y,
                        )
                    )

                    rotvec = (
                        _V98Rotation
                        .from_matrix(
                            R_full
                        )
                        .as_rotvec()
                    )

                    R_alpha = (
                        _V98Rotation
                        .from_rotvec(
                            alpha_apply
                            *
                            rotvec
                        )
                        .as_matrix()
                    )

                    # Interpolate rigid-body centroid motion.
                    c_alpha = (
                        (1.0 - alpha_apply)
                        *
                        ca
                        +
                        alpha_apply
                        *
                        cb
                    )

                    X_rigid = (
                        (
                            X
                            -
                            ca[
                                None, :
                            ]
                        )
                        @
                        R_alpha.T
                        +
                        c_alpha[
                            None, :
                        ]
                    )

                    contribution[
                        ids
                    ] = (
                        X_rigid
                        -
                        X
                    )

                else:

                    # Degenerate component:
                    # one translation only.
                    # Still cannot deform shape.
                    delta_center = (
                        Y.mean(
                            axis=0
                        )
                        -
                        X.mean(
                            axis=0
                        )
                    )

                    contribution[
                        ids
                    ] = (
                        alpha_apply
                        *
                        delta_center[
                            None, :
                        ]
                    )

            total_delta += (
                contribution
            )

            last_group_contribution[
                gid
            ] = (
                contribution.copy()
            )

            group_was_active[
                gid
            ] = True

        repaired[
            t
        ] = (
            base
            +
            total_delta
        )

        total_residual = (
            repaired[
                t
            ]
            -
            pred[
                t
            ]
        )

        vm = valid[
            t
        ]

        if np.any(
            vm
        ):

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
                        -
                        previous_total_residual
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

        frame_rows.append({
            "frame":
                int(t),

            "active_memories":
                int(
                    active_memory_count
                ),

            "released_memories":
                int(
                    len(
                        released_now
                    )
                ),

            "release_frame":
                bool(
                    released_now
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

        "rigid_groups": [
            {
                "id":
                    int(g["id"]),
                "support_points":
                    int(
                        np.sum(
                            g["support_mask"]
                        )
                    ),
                "member_ids":
                    [
                        int(x["id"])
                        for x in g["members"]
                    ],
            }
            for g in rigid_groups
        ],

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
