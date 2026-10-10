from pathlib import Path
import ast

SRC = Path("diagnostics/eval_close_anchor_causal_5task.py")
DST = Path("diagnostics/eval_close_anchor_relative_replay_524.py")

EXPECTED = "05484826dfef74cbe278a3974179a5a16705d35d"

import subprocess
commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()
assert commit == EXPECTED, f"Unexpected commit: {commit}"

source = SRC.read_text(encoding="utf-8")

helper = r'''
# ============================================================
# Fixed-phase N=524 relative-motion evaluation
# ============================================================

_REPLAY_ROOT = Path(
    "diagnostics/results/close_anchor_causal_5task_current10457"
)
_REPLAY_OUT = Path(
    "diagnostics/results/close_anchor_relative_replay_current524"
)


def _relative_motion(P0, Pt, H0, Ht):
    P0, Pt, H0, Ht = [
        np.asarray(x, dtype=np.float64)
        for x in (P0, Pt, H0, Ht)
    ]

    if P0.shape != Pt.shape or len(P0) < 3:
        raise RuntimeError("Object correspondence mismatch")

    if H0.shape != Ht.shape or len(H0) < 3:
        raise RuntimeError("Gripper correspondence mismatch")

    if not all(
        np.isfinite(x).all()
        for x in (P0, Pt, H0, Ht)
    ):
        raise RuntimeError("Nonfinite coordinates")

    # Rigid alignment of corresponding gripper surface points.
    A = H0 - H0.mean(axis=0)
    B = Ht - Ht.mean(axis=0)

    if np.linalg.matrix_rank(A, tol=1e-8) < 2:
        raise RuntimeError("Degenerate gripper geometry")

    U, S, Vt = np.linalg.svd(A.T @ B)

    D = np.diag([
        1.0,
        1.0,
        np.sign(np.linalg.det(U @ Vt))
    ])

    R = U @ D @ Vt

    residual = np.sqrt(
        np.mean(np.sum((A @ R - B) ** 2, axis=1))
    ) * 1000.0

    # Fail rather than silently using an invalid rigid frame.
    if residual > 2.0:
        raise RuntimeError(
            f"Gripper alignment residual {residual:.4f} mm"
        )

    rel0 = P0 - H0.mean(axis=0)
    relt = (Pt - Ht.mean(axis=0)) @ R.T

    delta = relt - rel0

    centroid_delta = delta.mean(axis=0)

    return {
        "centroid_mm": float(
            np.linalg.norm(centroid_delta) * 1000
        ),
        "point_median_mm": float(
            np.median(np.linalg.norm(delta, axis=1)) * 1000
        ),
        "delta_xyz_m": centroid_delta,
        "gripper_residual_mm": float(residual),
    }


def _export_relative_frames(
    task_out, key, oid, t_close, closed_frames,
    released, side_idx, pred, gt,
    labels, valid, exists, row, expected_row
):
    import csv

    assert int(row["object_id"]) == int(expected_row["object_id"])
    assert int(t_close) == int(expected_row["t_close"])
    assert str(row["close_side"]) == str(expected_row["close_side"])
    assert int(row["closed_frames"]) == len(closed_frames)

    records = []
    first_q10 = {}
    last_q10 = {}

    for t in closed_frames:

        # Original evaluator's object mask.
        full = (
            (labels[t] == oid)
            & valid[t]
            & (labels[0] == oid)
            & exists[0]
        )

        # Require common point identities across two frames.
        initial = (
            (labels[t_close] == oid)
            & valid[t_close]
        )

        paired = full & initial

        if int(full.sum()) < MIN_POINTS:
            continue

        if int(paired.sum()) < MIN_POINTS:
            raise RuntimeError(
                f"{key}: insufficient paired points at {t}"
            )

        gt_delta = None

        for condition in (
            "GT", "Original", "Free", "Anchor"
        ):

            owner = (
                "Original" if condition == "GT"
                else condition
            )

            indices = side_idx[owner]

            H0 = robot_points(
                released[owner], t_close, indices
            )

            Ht = robot_points(
                released[owner], t, indices
            )

            P = (
                gt if condition == "GT"
                else pred[condition]
            )

            gap_min, gap_q10, _ = gap_stats_mm(
                P[t, full], Ht
            )

            stats = _relative_motion(
                P[t_close, paired],
                P[t, paired],
                H0,
                Ht,
            )

            record = {
                "task": row["task"],
                "clip": key,
                "object_id": int(oid),
                "t_close": int(t_close),
                "frame": int(t),
                "frames_since_close": int(t - t_close),
                "condition": condition,
                "n_full_points": int(full.sum()),
                "n_paired_points": int(paired.sum()),
                "absolute_min_gap_mm": gap_min,
                "absolute_q10_gap_mm": gap_q10,
                "relative_centroid_drift_mm": stats["centroid_mm"],
                "relative_point_motion_median_mm": stats["point_median_mm"],
                "gripper_rigid_fit_rms_mm":
                    stats["gripper_residual_mm"],
                "relative_delta_x_m": stats["delta_xyz_m"][0],
                "relative_delta_y_m": stats["delta_xyz_m"][1],
                "relative_delta_z_m": stats["delta_xyz_m"][2],
                "original_gt_relative_delta_error_mm":
                    float("nan"),
            }

            if condition == "GT":
                gt_delta = stats["delta_xyz_m"]

            elif condition == "Original":
                record[
                    "original_gt_relative_delta_error_mm"
                ] = float(
                    np.linalg.norm(
                        stats["delta_xyz_m"] - gt_delta
                    ) * 1000
                )

            records.append(record)

            if condition != "GT":
                first_q10.setdefault(condition, gap_q10)
                last_q10[condition] = gap_q10

    if not records:
        raise RuntimeError(f"{key}: no evaluated frames")

    # Reproduce the source evaluator's absolute gap exactly.
    for condition, prefix in [
        ("Original", "original"),
        ("Free", "free"),
        ("Anchor", "anchor"),
    ]:
        for name, actual in [
            ("close", first_q10[condition]),
            ("end", last_q10[condition]),
        ]:
            expected = float(
                row[f"{prefix}_{name}_gap_q10_mm"]
            )

            if abs(actual - expected) > 1e-5:
                raise RuntimeError(
                    f"{key}: {condition} {name} gap mismatch"
                )

    output = (
        Path(task_out) / "per_frame_relative_metrics.csv"
    )

    with output.open(
        "a", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0])
        )

        if output.stat().st_size == 0:
            writer.writeheader()

        writer.writerows(records)
'''

def replace_once(before, after):
    global source
    count = source.count(before)
    if count != 1:
        raise RuntimeError(
            f"Patch anchor count={count}: {before[:60]}"
        )
    source = source.replace(before, after, 1)


replace_once(
    "def main():\n",
    helper + "\ndef main():\n",
)

replace_once(
    "    a = parser.parse_args()\n",
    '    parser.add_argument("--limit", type=int, default=0)\n'
    "    a = parser.parse_args()\n",
)

replace_once(
    "    task_out = (\n",
    '''    global OUT_ROOT
    OUT_ROOT = _REPLAY_OUT

    replay_csv = (
        _REPLAY_ROOT / task / "per_clip_causal_metrics.csv"
    )
    replay_df = pd.read_csv(replay_csv)

    if replay_df["clip"].duplicated().any():
        raise RuntimeError("Duplicate cached clips")

    replay_by_key = {
        str(r["clip"]): r
        for _, r in replay_df.iterrows()
    }

    replay_keys = set(replay_by_key)
    assert replay_keys

    print("[REPLAY] cached clips:", len(replay_keys))

    task_out = (
''',
)

replace_once(
    "    for raw in dataset:\n",
    '''    for raw in dataset:
        if a.limit and matched - len(done) >= a.limit:
            break
''',
)

replace_once(
    "            if key not in target_keys:\n",
    '''            if key not in replay_keys:
                continue

            if key not in target_keys:
''',
)

# Replace only the three optimization function calls.
# Retain all original phase audits, sampling, and inference.

replacements = {
    ("amin", "find_anchor_min_acc"): '''
            cache_path = (
                _REPLAY_ROOT / task / "matched_phases"
                / (safe_name(key) + ".npz")
            )

            with np.load(cache_path, allow_pickle=False) as z:
                assert str(z["clip"].item()) == key
                assert int(z["t_close"]) == t_close
                assert str(z["close_side"].item()) == close_side

                saved_original_acc = float(z["original_acc"])
                saved_target_acc = float(z["target_acc"])
                saved_free_acc = float(z["free_acc"])
                saved_anchor_acc = float(z["anchor_acc"])

                saved_free_phase = np.asarray(
                    z["free_phase"], np.float64
                ).copy()

                saved_anchor_phase = np.asarray(
                    z["anchor_phase"], np.float64
                ).copy()

            cache_row = replay_by_key[key]

            if not np.isclose(
                original_acc, saved_original_acc,
                atol=1e-8, rtol=1e-6
            ):
                raise RuntimeError("Original Acc mismatch")

            for phase in (saved_free_phase, saved_anchor_phase):
                if phase.shape != identity.shape:
                    raise RuntimeError("Phase shape mismatch")
                if not np.isfinite(phase).all():
                    raise RuntimeError("Nonfinite phase")
                if np.min(np.diff(phase)) < -1e-8:
                    raise RuntimeError("Nonmonotonic phase")
                if abs(phase[0]) > 1e-8:
                    raise RuntimeError("Initial pose changed")
                if abs(phase[-1] - (T-1)) > 1e-8:
                    raise RuntimeError("Final pose changed")

            if abs(saved_anchor_phase[t_close] - t_close) > 1e-8:
                raise RuntimeError("Grasp pose not fixed")

            amin = {
                "success": True,
                "acc": float(
                    cache_row["anchor_min_acc_si"]
                ) / 25.0,
                "phase": saved_anchor_phase,
            }
''',
    ("free", "solve_target_acc"): '''
            if not np.isclose(
                target_acc, saved_target_acc,
                atol=1e-8, rtol=1e-6
            ):
                raise RuntimeError("Target Acc mismatch")

            free = {
                "success": True,
                "acc": saved_free_acc,
                "phase": saved_free_phase,
            }
''',
    ("anchor", "solve_target_acc"): '''
            anchor = {
                "success": True,
                "acc": saved_anchor_acc,
                "phase": saved_anchor_phase,
            }
''',
}

tree = ast.parse(source)
changes = []

for node in ast.walk(tree):
    if not isinstance(node, ast.Assign):
        continue
    if len(node.targets) != 1:
        continue

    target = node.targets[0]
    value = node.value

    if isinstance(target, ast.Name) and isinstance(value, ast.Call):
        if isinstance(value.func, ast.Name):
            key = (target.id, value.func.id)
            if key in replacements:
                changes.append(
                    (node.lineno, node.end_lineno, key)
                )

assert len(changes) == 3, changes
assert set(k for _, _, k in changes) == set(replacements)

lines = source.splitlines(keepends=True)

for start, end, key in sorted(changes, reverse=True):
    lines[start-1:end] = [replacements[key]]

source = "".join(lines)

replace_once(
    "            rows.append(row)\n",
    '''            _export_relative_frames(
                task_out=task_out,
                key=key,
                oid=oid,
                t_close=t_close,
                closed_frames=closed_frames,
                released=released,
                side_idx=side_idx,
                pred=pred,
                gt=gt,
                labels=labels,
                valid=valid,
                exists=exists,
                row=row,
                expected_row=replay_by_key[key],
            )

            rows.append(row)
''',
)

ast.parse(source, filename=str(DST))

if DST.exists():
    raise RuntimeError(
        f"{DST} already exists; refusing to overwrite"
    )

DST.write_text(source, encoding="utf-8")

print("[OK] New evaluator:", DST)
print("[OK] Original evaluator untouched")
print("[OK] Old matched phases reused")
print("[OK] No new action optimization")
