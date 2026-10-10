#!/usr/bin/env python3
import ast
from pathlib import Path

P = Path("diagnostics/eval_close_anchor_relative_replay_524.py")
src = P.read_text(encoding="utf-8")

tree = ast.parse(src)
nodes = {
    n.name: n
    for n in tree.body
    if isinstance(n, ast.FunctionDef)
}

for name in ("_relative_motion", "_export_relative_frames"):
    assert name in nodes, f"Missing function: {name}"

new_motion = '''
def _pose_local_points(P, pose):
    from scipy.spatial.transform import Rotation

    P = np.asarray(P, dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64).reshape(7)

    if P.ndim != 2 or P.shape[1] != 3:
        raise RuntimeError("Invalid object points")

    if not np.isfinite(P).all() or not np.isfinite(pose).all():
        raise RuntimeError("Nonfinite pose or points")

    q = pose[3:]
    norm = np.linalg.norm(q)

    if norm < 0.95 or norm > 1.05:
        raise RuntimeError(f"Invalid quaternion norm: {norm}")

    rot = Rotation.from_quat(q / norm)

    return rot.inv().apply(P - pose[:3]), abs(norm - 1.0)


def _gripper_pose7(sample, t, arm):
    field = f"{arm}_gripper_pose"

    if field not in sample:
        raise RuntimeError(f"Missing {field}")

    poses = np.asarray(
        arr(sample[field]), dtype=np.float64
    ).reshape(-1, 7)

    if not 0 <= t < len(poses):
        raise RuntimeError("Pose timestep out of range")

    return poses[t]


def _relative_motion(P0, Pt, pose0, poset):
    P0 = np.asarray(P0, dtype=np.float64)
    Pt = np.asarray(Pt, dtype=np.float64)

    if P0.shape != Pt.shape or len(P0) < MIN_POINTS:
        raise RuntimeError("Object correspondence mismatch")

    rel0, e0 = _pose_local_points(P0, pose0)
    relt, et = _pose_local_points(Pt, poset)

    delta = relt - rel0
    centroid_delta = delta.mean(axis=0)

    return {
        "centroid_mm": float(
            np.linalg.norm(centroid_delta) * 1000.0
        ),
        "point_median_mm": float(
            np.median(np.linalg.norm(delta, axis=1)) * 1000.0
        ),
        "delta_xyz_m": centroid_delta,
        "quaternion_norm_error": max(e0, et),
    }


def _choose_reference_arm(
    close_side, released, gt, t_close, initial_mask
):
    if close_side in ("left", "right"):
        return close_side

    if close_side != "both":
        raise RuntimeError(f"Unknown close side: {close_side}")

    candidates = []

    for arm in ("right", "left"):
        idx = active_robot_indices(released["Original"], arm)

        gp = robot_points(
            released["Original"], t_close, idx
        )

        if len(gp) == 0:
            continue

        gmin, _, _ = gap_stats_mm(
            gt[t_close, initial_mask], gp
        )

        if np.isfinite(gmin):
            candidates.append((gmin, arm))

    if not candidates:
        raise RuntimeError("Cannot select reference arm")

    candidates.sort()
    return candidates[0][1]
'''

new_export = '''
def _export_relative_frames(
    task_out, key, oid, t_close, closed_frames,
    released, side_idx, pred, gt,
    labels, valid, exists, row, expected_row
):
    import csv

    assert int(row["object_id"]) == int(expected_row["object_id"])
    assert int(row["t_close"]) == int(expected_row["t_close"])
    assert str(row["close_side"]) == str(expected_row["close_side"])

    masks = {}
    good_times = []

    for t in closed_frames:
        m = (
            (labels[t] == oid)
            & valid[t]
            & (labels[0] == oid)
            & exists[0]
        )
        masks[t] = m

        if int(m.sum()) >= MIN_POINTS:
            good_times.append(t)

    if len(good_times) < 2:
        raise RuntimeError("Insufficient valid closed frames")

    if good_times[0] != t_close:
        raise RuntimeError("Grasp frame not valid")

    # Use the same object point identities at every evaluated frame.
    common = masks[good_times[0]].copy()
    for t in good_times[1:]:
        common &= masks[t]

    if int(common.sum()) < MIN_POINTS:
        raise RuntimeError(
            f"Insufficient common object points: {common.sum()}"
        )

    reference_arm = _choose_reference_arm(
        str(row["close_side"]),
        released,
        gt,
        t_close,
        common,
    )

    # Verify that the saved Anchor preserves the grasp pose.
    po = _gripper_pose7(
        released["Original"], t_close, reference_arm
    )
    pa = _gripper_pose7(
        released["Anchor"], t_close, reference_arm
    )

    if np.linalg.norm(po[:3] - pa[:3]) > 1e-7:
        raise RuntimeError("Anchor grasp translation changed")

    qo = po[3:] / np.linalg.norm(po[3:])
    qa = pa[3:] / np.linalg.norm(pa[3:])

    if 1.0 - abs(np.dot(qo, qa)) > 1e-6:
        raise RuntimeError("Anchor grasp orientation changed")

    records = []
    first_q10 = {}
    last_q10 = {}

    for t in good_times:
        frame_results = {}

        for condition in ("GT", "Original", "Free", "Anchor"):
            owner = "Original" if condition == "GT" else condition

            P = gt if condition == "GT" else pred[condition]

            # Preserve the ORIGINAL gap definition:
            # nearest distance from object points to selected robot surface.
            Ht = robot_points(
                released[owner], t, side_idx[owner]
            )

            gap_min, gap_q10, _ = gap_stats_mm(
                P[t, masks[t]], Ht
            )

            pose0 = _gripper_pose7(
                released[owner], t_close, reference_arm
            )
            poset = _gripper_pose7(
                released[owner], t, reference_arm
            )

            stats = _relative_motion(
                P[t_close, common],
                P[t, common],
                pose0,
                poset,
            )

            frame_results[condition] = stats

            records.append({
                "task": row["task"],
                "clip": key,
                "object_id": int(oid),
                "reference_arm": reference_arm,
                "t_close": int(t_close),
                "frame": int(t),
                "frames_since_close": int(t - t_close),
                "condition": condition,
                "n_full_points": int(masks[t].sum()),
                "n_common_points": int(common.sum()),
                "absolute_min_gap_mm": gap_min,
                "absolute_q10_gap_mm": gap_q10,
                "relative_centroid_drift_mm": stats["centroid_mm"],
                "relative_point_motion_median_mm":
                    stats["point_median_mm"],
                "quaternion_norm_error":
                    stats["quaternion_norm_error"],
                "relative_delta_x_m": stats["delta_xyz_m"][0],
                "relative_delta_y_m": stats["delta_xyz_m"][1],
                "relative_delta_z_m": stats["delta_xyz_m"][2],
                "original_gt_relative_delta_error_mm":
                    float("nan"),
            })

            if condition != "GT":
                first_q10.setdefault(condition, gap_q10)
                last_q10[condition] = gap_q10

        # Counterfactual GT does not exist for Free/Anchor.
        gt_delta = frame_results["GT"]["delta_xyz_m"]
        orig_delta = frame_results["Original"]["delta_xyz_m"]

        records[-3][
            "original_gt_relative_delta_error_mm"
        ] = float(
            np.linalg.norm(orig_delta - gt_delta) * 1000.0
        )

    # Existing gap values must be reproducible.
    for condition, prefix in (
        ("Original", "original"),
        ("Free", "free"),
        ("Anchor", "anchor"),
    ):
        for suffix, value in (
            ("close", first_q10[condition]),
            ("end", last_q10[condition]),
        ):
            expected = float(
                row[f"{prefix}_{suffix}_gap_q10_mm"]
            )

            error = abs(value - expected)

            if error > 1e-5:
                raise RuntimeError(
                    f"{key}: {condition} {suffix} "
                    f"gap replay mismatch {error:.6f} mm"
                )

    # Frame t_close must have zero relative displacement.
    for rec in records:
        if rec["frame"] == t_close:
            if rec["relative_centroid_drift_mm"] > 1e-5:
                raise RuntimeError("Nonzero grasp-frame drift")

    output = (
        Path(task_out) / "per_frame_relative_metrics.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0])
        )

        if output.stat().st_size == 0:
            writer.writeheader()

        writer.writerows(records)

    print(
        f"[RELATIVE_OK] {key} "
        f"frames={len(good_times)} "
        f"common_points={int(common.sum())} "
        f"reference_arm={reference_arm}",
        flush=True,
    )
'''

# Replace exact AST function ranges, not arbitrary substrings.
lines = src.splitlines(keepends=True)
changes = [
    (nodes["_relative_motion"], new_motion),
    (nodes["_export_relative_frames"], new_export),
]

for node, replacement in sorted(
    changes, key=lambda x: x[0].lineno, reverse=True
):
    lines[node.lineno-1:node.end_lineno] = [
        replacement + "\n"
    ]

patched = "".join(lines)

old_limit = (
    "if a.limit and matched - len(done) >= a.limit:"
)
new_limit = "if a.limit and target_seen >= a.limit:"

assert patched.count(old_limit) == 1
patched = patched.replace(old_limit, new_limit, 1)

ast.parse(patched)

P.write_text(patched, encoding="utf-8")

print("[PATCH OK] Gripper-pose local coordinates")
print("[PATCH OK] Stable object-point correspondence")
print("[PATCH OK] Pilot attempt limit")
print("[PATCH OK] Cached phases unchanged")
