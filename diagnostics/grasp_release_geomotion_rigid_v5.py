import numpy as np
from scipy.spatial import cKDTree

from grasp_release_geomotion_stateful_v4 import (
    _squeeze_batch,
    _closed_segments,
    _effective_support,
)


def apply_rigid_geomotion_repair_v5(
    pred_corrected,
    robot_flows,
    valid,
    *,
    right_gripper_open=None,
    left_gripper_open=None,
    right_gripper_pose=None,
    left_gripper_pose=None,
    sigma_geom_m=0.015,
    sigma_motion_m=None,  # ignored, kept for call compatibility
    rel_motion_sigma=0.30,
    step_floor_m=0.003,
    evidence_min_robot_step_m=0.010,
    ramp_full_m=0.020,
    sigma_support_m=0.015,
):
    pred = _squeeze_batch(pred_corrected).astype(np.float64, copy=True)
    robot = _squeeze_batch(robot_flows).astype(np.float64, copy=False)
    valid = _squeeze_batch(valid).astype(bool)
    T, N, _ = pred.shape
    eps = 1e-12

    sides = {}
    for side, open_raw, pose_raw in [
        ("right", right_gripper_open, right_gripper_pose),
        ("left", left_gripper_open, left_gripper_pose),
    ]:
        if open_raw is None or pose_raw is None:
            continue
        open_seq = _squeeze_batch(open_raw).reshape(T, -1)[:, 0].astype(bool)
        pose = _squeeze_batch(pose_raw).reshape(T, 7)
        sides[side] = {
            "open": open_seq,
            "center0": pose[0, :3].astype(np.float64),
        }
    if not sides:
        raise RuntimeError("No usable gripper state.")
    side_names = list(sides.keys())

    if len(side_names) == 1:
        robot_side = np.zeros(robot.shape[1], dtype=np.int64)
    else:
        centers = np.stack([sides[s]["center0"] for s in side_names], axis=0)
        d_side = np.linalg.norm(
            robot[0, :, None, :] - centers[None, :, :], axis=-1
        )
        robot_side = np.argmin(d_side, axis=1)

    side_robot_idx = {}
    for si, side in enumerate(side_names):
        ridx = np.flatnonzero(robot_side == si)
        if len(ridx) > 0:
            side_robot_idx[side] = ridx

    segments = []
    for side in side_names:
        if side not in side_robot_idx:
            continue
        ridx = side_robot_idx[side]
        for ref_t, end_t in _closed_segments(sides[side]["open"], start_t=1):
            evidence_t = end_t
            for tt in range(ref_t + 1, end_t + 1):
                step = np.linalg.norm(
                    robot[tt, ridx] - robot[ref_t, ridx], axis=1
                ).mean()
                if step >= evidence_min_robot_step_m:
                    evidence_t = tt
                    break
            segments.append(
                {
                    "side": side,
                    "ref_t": int(ref_t),
                    "evidence_t": int(evidence_t),
                    "end_t": int(end_t),
                    "robot_side_idx": ridx,
                    "initialized": False,
                    "ready": False,
                }
            )

    repaired = pred.copy()
    carried = np.zeros((N, 3), dtype=np.float64)
    prev = carried.copy()
    memories = []
    rows = []

    for t in range(1, T):
        repaired[t] = pred[t] + carried

        for seg in segments:
            if seg["ref_t"] == t and not seg["initialized"]:
                ridx = seg["robot_side_idx"]
                tree = cKDTree(robot[t, ridx])
                d_ref, nn = tree.query(repaired[t], k=1)
                seg["d_ref"] = d_ref
                seg["robot_idx"] = ridx[nn]
                seg["w_geom"] = (
                    np.exp(-0.5 * (d_ref / float(sigma_geom_m)) ** 2)
                    * valid[t].astype(np.float64)
                )
                seg["initialized"] = True

        for seg in segments:
            if (
                seg["initialized"]
                and not seg["ready"]
                and seg["evidence_t"] == t
            ):
                a = seg["robot_idx"]
                r0 = seg["ref_t"]
                scene_step = pred[t] - pred[r0]
                robot_step = robot[t, a] - robot[r0, a]
                err = np.linalg.norm(scene_step - robot_step, axis=1)
                rel = err / np.maximum(
                    np.linalg.norm(robot_step, axis=1), step_floor_m
                )
                w_motion = np.exp(-0.5 * (rel / rel_motion_sigma) ** 2)
                w = seg["w_geom"] * w_motion * valid[t].astype(np.float64)
                seg["w"] = w
                seg["ready"] = True
                memories.append(
                    {
                        "side": seg["side"],
                        "ref_t": int(seg["ref_t"]),
                        "evidence_t": int(seg["evidence_t"]),
                        "end_t": int(seg["end_t"]),
                        "geom_weight_sum": float(seg["w_geom"].sum()),
                        "joint_weight_sum": float(w.sum()),
                        "effective_support": float(_effective_support(w)),
                        "max_weight": float(w.max()),
                        "median_motion_error_mm": float(
                            np.median(err) * 1000.0
                        ),
                    }
                )

        before_state = repaired[t].copy()

        active = [
            s
            for s in segments
            if s["ready"] and s["evidence_t"] <= t <= s["end_t"]
        ]

        if active:
            num = np.zeros((N, 3), dtype=np.float64)
            den = np.zeros(N, dtype=np.float64)

            for seg in active:
                a = seg["robot_idx"]
                w = seg["w"]
                ws = float(w.sum())
                if ws <= eps:
                    continue

                vec = before_state - robot[t, a]
                d = np.linalg.norm(vec, axis=1)
                excess = np.maximum(d - seg["d_ref"], 0.0)
                raw = -excess[:, None] * vec / np.maximum(d[:, None], eps)

                # ONE translation per grasp (shape preserved)
                delta = (w[:, None] * raw).sum(axis=0) / ws

                # carrier: points that move together with the contact cluster
                step_scene = pred[t] - pred[seg["ref_t"]]
                cluster_step = (w[:, None] * step_scene).sum(axis=0) / ws
                ramp = min(
                    1.0, float(np.linalg.norm(cluster_step)) / ramp_full_m
                )
                dev = np.linalg.norm(step_scene - cluster_step[None], axis=1)
                c = (
                    np.exp(-0.5 * (dev / sigma_support_m) ** 2)
                    * valid[t].astype(np.float64)
                    * ramp
                )

                num += c[:, None] * delta[None, :]
                den += c

            repaired[t] = before_state + num / np.maximum(1.0, den)[:, None]

        carried = repaired[t] - pred[t]

        vt = valid[t]
        if np.any(vt):
            res_mean = float(np.linalg.norm(carried[vt], axis=1).mean() * 1000.0)
            res_chg = float(
                np.linalg.norm((carried - prev)[vt], axis=1).mean() * 1000.0
            )
        else:
            res_mean = np.nan
            res_chg = np.nan

        nb = nr = ds = 0.0
        for seg in active:
            a = seg["robot_idx"]
            wt = seg["w"] * vt.astype(np.float64)
            ex_b = np.maximum(
                np.linalg.norm(before_state - robot[t, a], axis=1) - seg["d_ref"],
                0.0,
            )
            ex_r = np.maximum(
                np.linalg.norm(repaired[t] - robot[t, a], axis=1) - seg["d_ref"],
                0.0,
            )
            nb += float((wt * ex_b).sum())
            nr += float((wt * ex_r).sum())
            ds += float(wt.sum())

        if ds > eps:
            sep_b = nb / ds * 1000.0
            sep_r = nr / ds * 1000.0
        else:
            sep_b = np.nan
            sep_r = np.nan

        rows.append(
            {
                "frame": int(t),
                "active_memories": int(len(active)),
                "release_frame": bool(
                    any(s["end_t"] + 1 == t for s in segments)
                ),
                "separation_excess_before_mm": float(sep_b),
                "separation_excess_after_mm": float(sep_r),
                "carried_residual_mean_mm": float(res_mean),
                "residual_change_mean_mm": float(res_chg),
            }
        )
        prev = carried.copy()

    return repaired, {
        "memories": memories,
        "frame_rows": rows,
        "side_robot_point_counts": {
            s: int(len(side_robot_idx.get(s, []))) for s in side_names
        },
        "sigma_geom_m": float(sigma_geom_m),
        "rel_motion_sigma": float(rel_motion_sigma),
    }
