import copy
import numpy as np

from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


# PointWorld official grid size = 0.015 m.
GRID_M = 0.015

# Automatic geometric grasp detection.
CONTACT_M = 1.0 * GRID_M      # 15 mm: definite close contact
NEAR_M = 2.0 * GRID_M         # 30 mm: persistence neighborhood
RELEASE_M = 3.0 * GRID_M      # 45 mm: clear separation
PERSIST_FRAMES = 2


# ============================================================
# KINEMATIC RETIMING
# ============================================================

def _retime_linear(a, phase_norm):
    a = np.asarray(a)
    T = a.shape[0]

    x0 = np.arange(T, dtype=np.float64)
    x1 = np.asarray(phase_norm, dtype=np.float64) * float(T - 1)

    flat = a.reshape(T, -1).astype(np.float64)
    out = np.empty_like(flat)

    for j in range(flat.shape[1]):
        out[:, j] = np.interp(
            x1,
            x0,
            flat[:, j],
        )

    return out.reshape(a.shape).astype(a.dtype)


def _retime_pose7(a, phase_norm):
    a = np.asarray(a)

    if a.ndim != 2 or a.shape[1] != 7:
        raise RuntimeError(
            f"expected (T,7) pose, got {a.shape}"
        )

    T = a.shape[0]

    x0 = np.arange(T, dtype=np.float64)
    x1 = np.asarray(phase_norm, dtype=np.float64) * float(T - 1)

    out = np.empty_like(a)

    for d in range(3):
        out[:, d] = np.interp(
            x1,
            x0,
            a[:, d].astype(np.float64),
        )

    rot = Rotation.from_quat(
        a[:, 3:7].astype(np.float64)
    )

    out[:, 3:7] = Slerp(
        x0,
        rot,
    )(x1).as_quat().astype(a.dtype)

    return out


def build_retimed_kinematics(
    decoded,
    corrected_phase,
    old,
    domain,
    robot_sampler,
    max_robot_points,
    deterministic,
    seed,
    force_single_arm,
    reference_built=None,
    key="",
):
    """
    Final action implementation:
      temporal retiming is applied to underlying robot kinematics,
      then PointWorld's official robot builder regenerates robot_flows.
    """

    d = copy.deepcopy(decoded)

    d["joint_positions"] = _retime_linear(
        decoded["joint_positions"],
        corrected_phase,
    )

    d["base_pose"] = _retime_pose7(
        decoded["base_pose"],
        corrected_phase,
    )

    for name in [
        "right_gripper_pose",
        "left_gripper_pose",
    ]:
        if name in decoded:
            d[name] = _retime_pose7(
                decoded[name],
                corrected_phase,
            )

    for name in [
        "right_gripper_open",
        "left_gripper_open",
        "gripper_positions",
    ]:
        if name in decoded:
            a = np.asarray(decoded[name])

            if a.ndim >= 1 and a.shape[0] == len(corrected_phase):
                d[name] = _retime_linear(
                    a,
                    corrected_phase,
                )

    rebuilt = old.omap.build_flow_sample_with_object_ids(
        d,
        domain=domain,
        robot_sampler=robot_sampler,
        max_robot_points=max_robot_points,
        deterministic=deterministic,
        seed=seed,
        force_single_arm=force_single_arm,
    )

    if reference_built is not None:

        rb = np.asarray(
            reference_built["robot_flows"]
        )

        rc = np.asarray(
            rebuilt["robot_flows"]
        )

        start_err = float(
            np.max(
                np.abs(
                    rc[0] - rb[0]
                )
            )
        )

        end_err = float(
            np.max(
                np.abs(
                    rc[-1] - rb[-1]
                )
            )
        )

        print(
            f"[KINEMATIC REBUILD] {key} "
            f"start={start_err:.3e} "
            f"end={end_err:.3e}",
            flush=True,
        )

        if start_err > 1e-5 or end_err > 1e-5:
            raise RuntimeError(
                f"{key}: robot start/end changed "
                f"{start_err:.3e}/{end_err:.3e}"
            )

    return rebuilt


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def _split_hands(
    robot_points,
    right_center,
    left_center,
):
    """
    RELEASE_GRIPPER_ONLY=True:
    assign each sampled gripper surface point to the exact
    RIGHT or LEFT gripper center.
    """

    G = np.asarray(robot_points)

    dr = np.linalg.norm(
        G - right_center[None, :],
        axis=1,
    )

    dl = np.linalg.norm(
        G - left_center[None, :],
        axis=1,
    )

    RH = G[dr <= dl]
    LH = G[dl < dr]

    return RH, LH


def _min_distance(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.inf

    return float(
        np.min(
            cKDTree(B).query(
                A,
                k=1,
            )[0]
        )
    )


def _pose_apply(points, pose_from, pose_to):
    """
    Preserve a rigid object's pose relative to one gripper:
        T_to * inv(T_from) * P
    """

    R0 = Rotation.from_quat(
        pose_from[3:7]
    ).as_matrix()

    R1 = Rotation.from_quat(
        pose_to[3:7]
    ).as_matrix()

    t0 = pose_from[:3]
    t1 = pose_to[:3]

    local = (
        R0.T
        @ (
            points - t0[None, :]
        ).T
    ).T

    return (
        R1 @ local.T
    ).T + t1[None, :]


def _pose_frame_points(pose, scale=0.05):
    """
    Four landmarks for a gripper SE(3) frame:
    origin + local xyz axes.
    Used for automatic two-hand rigid fitting.
    """

    R = Rotation.from_quat(
        pose[3:7]
    ).as_matrix()

    t = pose[:3]

    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, scale],
        ],
        dtype=np.float64,
    )

    return (
        R @ local.T
    ).T + t[None, :]


def _rigid_fit(A, B):
    """
    Rigid Kabsch transform mapping A -> B.
    No scale.
    """

    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)

    ca = A.mean(axis=0)
    cb = B.mean(axis=0)

    X = A - ca
    Y = B - cb

    U, _, Vt = np.linalg.svd(
        X.T @ Y
    )

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = cb - R @ ca

    return R, t


def _apply_rt(P, R, t):
    return (
        R @ P.T
    ).T + t[None, :]


def _bimanual_apply(
    points,
    right_from,
    left_from,
    right_to,
    left_to,
):
    """
    A single rigid object transform that best satisfies BOTH
    gripper frames.
    """

    A = np.concatenate(
        [
            _pose_frame_points(right_from),
            _pose_frame_points(left_from),
        ],
        axis=0,
    )

    B = np.concatenate(
        [
            _pose_frame_points(right_to),
            _pose_frame_points(left_to),
        ],
        axis=0,
    )

    R, t = _rigid_fit(A, B)

    return _apply_rt(
        points,
        R,
        t,
    )


# ============================================================
# AUTOMATIC GRASP DETECTION
# ============================================================

def _detect_grasp_interval(dist):
    """
    Automatic:
      - contact must reach <= 15 mm
      - must have nearby support over >=2 frames
      - release occurs only after >45 mm for 2 consecutive frames
    """

    d = np.asarray(dist, np.float64)
    T = len(d)

    anchor = None

    for t in range(T):

        if not np.isfinite(d[t]):
            continue

        if d[t] > CONTACT_M:
            continue

        lo = max(0, t - 1)
        hi = min(T, t + 2)

        near_count = int(
            np.sum(
                d[lo:hi] <= NEAR_M
            )
        )

        if near_count >= PERSIST_FRAMES:
            anchor = t
            break

    if anchor is None:
        return None, None

    release = T

    for t in range(anchor + 1, T - 1):

        if (
            d[t] > RELEASE_M
            and d[t + 1] > RELEASE_M
        ):
            release = t
            break

    return int(anchor), int(release)


def _state_name(left_on, right_on):
    if left_on and right_on:
        return "bimanual"

    if left_on:
        return "left"

    if right_on:
        return "right"

    return "free"


# ============================================================
# FINAL OBJECT-AGNOSTIC HARD GRASP CONSTRAINT
# ============================================================

def apply_grasp_consistency(
    pred,
    robot,
    labels,
    exists,
    valid,
    right_pose,
    left_pose,
    eligible_object_ids,
    corrected_phase,
    min_points=8,
):
    """
    Inputs
    ------
    pred:
        (3,T,N,3)
        [Original, Original-repeat, Corrected]

    robot:
        (3,T,Nr,3)
        RELEASE_GRIPPER_ONLY robot surface points.

    right_pose / left_pose:
        (3,T,7)
        Exact original/repeat/corrected gripper poses.

    labels:
        (3,T,N)

    valid / exists:
        (3,T,N)

    corrected_phase:
        normalized phase used for corrected action.

    Behavior
    --------
    1. Detect grasp automatically from ORIGINAL rollout.
    2. Automatically select left / right / bimanual.
    3. Apply hard rigid coupling only while grasp is active.
    4. On release, hand control is removed and PointWorld free
       motion is retained with a rigid continuity bridge.
    """

    out = np.asarray(
        pred
    ).copy()

    robot = np.asarray(robot)
    labels = np.asarray(labels)
    exists = np.asarray(exists).astype(bool)
    valid = np.asarray(valid).astype(bool)

    right_pose = np.asarray(right_pose)
    left_pose = np.asarray(left_pose)

    if out.ndim != 4 or out.shape[0] != 3:
        raise RuntimeError(
            f"bad pred shape {out.shape}"
        )

    if right_pose.shape[:2] != out.shape[:2]:
        raise RuntimeError(
            f"right pose shape {right_pose.shape} "
            f"vs pred {out.shape}"
        )

    if left_pose.shape[:2] != out.shape[:2]:
        raise RuntimeError(
            f"left pose shape {left_pose.shape} "
            f"vs pred {out.shape}"
        )

    T = out.shape[1]

    phase = np.asarray(
        corrected_phase,
        dtype=np.float64,
    ).reshape(-1)

    if len(phase) != T:
        raise RuntimeError(
            f"phase={len(phase)}, T={T}"
        )

    # Which ORIGINAL phase does each corrected frame correspond to?
    src_t = np.rint(
        np.clip(
            phase,
            0.0,
            1.0,
        )
        * float(T - 1)
    ).astype(int)

    audits = []

    for oid in eligible_object_ids:

        oid = int(oid)

        # Stable point identities belonging to this rigid object.
        object_idx = (
            (labels[0, 0] == oid)
            & exists[0, 0]
        )

        if int(object_idx.sum()) < min_points:
            continue

        d_left = np.full(
            T,
            np.inf,
            dtype=np.float64,
        )

        d_right = np.full(
            T,
            np.inf,
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # ORIGINAL rollout: exact per-hand distance.
        # ----------------------------------------------------
        for t in range(T):

            m = (
                object_idx
                & valid[0, t]
                & exists[0, t]
            )

            if int(m.sum()) < min_points:
                continue

            O = out[
                0,
                t,
                m,
                :,
            ]

            RH, LH = _split_hands(
                robot[0, t],
                right_pose[0, t, :3],
                left_pose[0, t, :3],
            )

            d_right[t] = _min_distance(
                O,
                RH,
            )

            d_left[t] = _min_distance(
                O,
                LH,
            )

        L_anchor, L_release = _detect_grasp_interval(
            d_left
        )

        R_anchor, R_release = _detect_grasp_interval(
            d_right
        )

        left_active_orig = np.zeros(
            T,
            dtype=bool,
        )

        right_active_orig = np.zeros(
            T,
            dtype=bool,
        )

        if L_anchor is not None:
            left_active_orig[
                L_anchor:L_release
            ] = True

        if R_anchor is not None:
            right_active_orig[
                R_anchor:R_release
            ] = True

        # No grasp -> leave PointWorld prediction untouched.
        if (
            not np.any(left_active_orig)
            and not np.any(right_active_orig)
        ):
            audits.append(
                {
                    "object_id": oid,
                    "mode": "free",
                    "left_anchor": -1,
                    "right_anchor": -1,
                    "left_release": -1,
                    "right_release": -1,
                    "applied": False,
                }
            )
            continue

        # Map original grasp state onto retimed corrected frames.
        left_active = left_active_orig[
            src_t
        ]

        right_active = right_active_orig[
            src_t
        ]

        any_bimanual = bool(
            np.any(
                left_active
                & right_active
            )
        )

        if any_bimanual:
            overall_mode = "bimanual"

        elif np.any(left_active) and np.any(right_active):
            overall_mode = "sequential"

        elif np.any(left_active):
            overall_mode = "left"

        else:
            overall_mode = "right"

        # ----------------------------------------------------
        # Sequential constrained rollout.
        # ----------------------------------------------------
        previous_state = "free"
        previous_fixed = None

        segment_anchor_obj = None
        anchor_R_pose = None
        anchor_L_pose = None

        # Free-motion continuity transform after release.
        bridge_R = np.eye(3)
        bridge_t = np.zeros(3)
        bridge_active = False

        for t in range(T):

            state = _state_name(
                bool(left_active[t]),
                bool(right_active[t]),
            )

            # Raw corrected object.
            raw_obj = out[
                2,
                t,
                object_idx,
                :,
            ].copy()

            # ------------------------------------------------
            # ENTER / CHANGE GRASP STATE
            # ------------------------------------------------
            if state != "free":

                if state != previous_state:

                    source = int(
                        src_t[t]
                    )

                    source_obj = out[
                        0,
                        source,
                        object_idx,
                        :,
                    ].copy()

                    # At initial grasp / re-grasp:
                    # establish object pose directly from the ORIGINAL
                    # grasp relation at the corresponding path phase.
                    if state == "left":

                        current_fixed = _pose_apply(
                            source_obj,
                            left_pose[
                                0,
                                source,
                            ],
                            left_pose[
                                2,
                                t,
                            ],
                        )

                    elif state == "right":

                        current_fixed = _pose_apply(
                            source_obj,
                            right_pose[
                                0,
                                source,
                            ],
                            right_pose[
                                2,
                                t,
                            ],
                        )

                    else:

                        current_fixed = _bimanual_apply(
                            source_obj,
                            right_pose[
                                0,
                                source,
                            ],
                            left_pose[
                                0,
                                source,
                            ],
                            right_pose[
                                2,
                                t,
                            ],
                            left_pose[
                                2,
                                t,
                            ],
                        )

                    segment_anchor_obj = (
                        current_fixed.copy()
                    )

                    anchor_R_pose = (
                        right_pose[
                            2,
                            t,
                        ].copy()
                    )

                    anchor_L_pose = (
                        left_pose[
                            2,
                            t,
                        ].copy()
                    )

                    bridge_active = False

                else:
                    # --------------------------------------------
                    # CONTINUE SAME GRASP STATE
                    # --------------------------------------------
                    if state == "left":

                        current_fixed = _pose_apply(
                            segment_anchor_obj,
                            anchor_L_pose,
                            left_pose[
                                2,
                                t,
                            ],
                        )

                    elif state == "right":

                        current_fixed = _pose_apply(
                            segment_anchor_obj,
                            anchor_R_pose,
                            right_pose[
                                2,
                                t,
                            ],
                        )

                    else:

                        current_fixed = _bimanual_apply(
                            segment_anchor_obj,
                            anchor_R_pose,
                            anchor_L_pose,
                            right_pose[
                                2,
                                t,
                            ],
                            left_pose[
                                2,
                                t,
                            ],
                        )

                out[
                    2,
                    t,
                    object_idx,
                    :,
                ] = current_fixed

                previous_fixed = (
                    current_fixed.copy()
                )

            # ------------------------------------------------
            # RELEASE / FREE MOTION
            # ------------------------------------------------
            else:

                if (
                    previous_state != "free"
                    and previous_fixed is not None
                ):
                    # Align the first free PointWorld prediction to the
                    # constrained release pose, then keep PointWorld's
                    # subsequent relative free motion.
                    Rb, tb = _rigid_fit(
                        raw_obj,
                        previous_fixed,
                    )

                    bridge_R = Rb
                    bridge_t = tb
                    bridge_active = True

                if bridge_active:

                    free_obj = _apply_rt(
                        raw_obj,
                        bridge_R,
                        bridge_t,
                    )

                    out[
                        2,
                        t,
                        object_idx,
                        :,
                    ] = free_obj

                    previous_fixed = (
                        free_obj.copy()
                    )

                else:
                    previous_fixed = (
                        raw_obj.copy()
                    )

            previous_state = state

        audits.append(
            {
                "object_id": oid,
                "mode": overall_mode,
                "left_anchor": (
                    -1
                    if L_anchor is None
                    else int(L_anchor + 1)
                ),
                "right_anchor": (
                    -1
                    if R_anchor is None
                    else int(R_anchor + 1)
                ),
                "left_release": (
                    -1
                    if L_anchor is None
                    or L_release >= T
                    else int(L_release + 1)
                ),
                "right_release": (
                    -1
                    if R_anchor is None
                    or R_release >= T
                    else int(R_release + 1)
                ),
                "original_left_min_mm": (
                    float(np.min(d_left)) * 1000.0
                    if np.isfinite(d_left).any()
                    else np.nan
                ),
                "original_right_min_mm": (
                    float(np.min(d_right)) * 1000.0
                    if np.isfinite(d_right).any()
                    else np.nan
                ),
                "applied": True,
            }
        )

    return out, audits
