from pathlib import Path
import re

D = Path("diagnostics")
source = D / "eval_acc_only_boundary_pa_final.py"
original = source.read_text()

helper = '''
from diagnostics.build_acc_only_boundary_correction_current10457 import (
    acc_at_phase, poses_at_phase,
)

def _cause_pose(built, side, T):
    v = built.get(f"{side}_gripper_pose")
    if v is None:
        a = np.zeros((T, 7), dtype=np.float64)
        a[:, 6] = 1.0
        return a
    a = np.asarray(v, dtype=np.float64)
    assert a.shape == (T, 7)
    return a

def _cause_motion(right, left, phase):
    phase = np.asarray(phase, dtype=np.float64)
    rp, lp = poses_at_phase(right, left, phase)
    a = float(acc_at_phase(
        right, left, phase
    )["rms_acceleration"])
    velocities = np.concatenate([
        np.diff(rp[:, :3], axis=0),
        np.diff(lp[:, :3], axis=0),
    ], axis=0) * 25.0
    v = float(np.sqrt(np.mean(
        np.sum(velocities ** 2, axis=1)
    )))
    return a, v

def _cause_pair(right, left, T):
    u = np.linspace(0.0, 1.0, T)
    base = np.arange(T, dtype=np.float64)
    groups = {}
    for k, amps in (
        (1, np.linspace(0.12, 0.65, 12)),
        (3, np.linspace(0.04, 0.20, 12)),
    ):
        candidates = []
        for amp in amps:
            p = base + amp * np.sin(2 * np.pi * k * u)
            p[0], p[-1] = 0.0, float(T - 1)
            if np.min(np.diff(p)) < 0.10:
                continue
            a, v = _cause_motion(right, left, p)
            candidates.append((p, a, v))
        if not candidates:
            raise RuntimeError("No monotone temporal controls")
        groups[k] = candidates

    choices = []
    for p1, a1, v1 in groups[1]:
        for p3, a3, v3 in groups[3]:
            speed_gap = abs(v1 - v3) / max(
                (v1 + v3) / 2, 1e-9
            )
            acc_gap = abs(a1 - a3) / max(
                min(a1, a3), 1e-9
            )
            qualified = speed_gap <= 0.03 and acc_gap >= 0.20
            score = speed_gap - 0.025 * min(acc_gap, 2.0)
            choices.append((
                not qualified, score,
                speed_gap, acc_gap,
                p1, p3,
            ))

    choices.sort(key=lambda x: (x[0], x[1]))
    _, _, speed_gap, acc_gap, p1, p3 = choices[0]
    return p1, p3, float(speed_gap), float(acc_gap)

def _cause_geometry(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert x.shape == y.shape
    assert len(x) >= 8
    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)

    U, _, Vt = np.linalg.svd(xc.T @ yc)
    M = np.eye(3)
    M[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ M @ Vt
    aligned = xc @ R

    se3 = np.mean(np.linalg.norm(aligned - yc, axis=1)) * 1000
    den = np.linalg.norm(xc)
    scale = np.linalg.norm(yc) / max(den, 1e-12)

    ix = np.unique(np.linspace(
        0, len(x) - 1, min(48, len(x))
    ).astype(int))
    xx, yy = x[ix], y[ix]
    i, j = np.triu_indices(len(ix), k=1)
    d0 = np.linalg.norm(xx[i] - xx[j], axis=1)
    d1 = np.linalg.norm(yy[i] - yy[j], axis=1)
    pair = np.mean(np.abs(d1 - d0)) * 1000

    return float(se3), float(pair), float(abs(np.log(
        max(scale, 1e-12)
    )))
'''

phase_patch = '''
                _right = _cause_pose(built, "right", T)
                _left = _cause_pose(built, "left", T)
                _identity = np.arange(T, dtype=np.float64)
                _saved_acc = float(meta["corrected_acc"])

                _pair_speed_gap = np.nan
                _pair_acc_gap = np.nan

                if CAUSE_MODE in ("freq1", "freq3"):
                    _p1, _p3, _pair_speed_gap, _pair_acc_gap = (
                        _cause_pair(_right, _left, T)
                    )
                    _chosen = (
                        _p1 if CAUSE_MODE == "freq1" else _p3
                    )
                    corrected_phase = _chosen / float(T - 1)

                _chosen_phase = corrected_phase * float(T - 1)
                _base_acc, _base_speed = _cause_motion(
                    _right, _left, _identity
                )
                _new_acc, _new_speed = _cause_motion(
                    _right, _left, _chosen_phase
                )
                _max_shift = float(np.max(
                    np.abs(_chosen_phase - _identity)
                ))

                # Original evaluator output now reports the
                # actual action metrics passed through this path.
                meta["original_acc"] = _base_acc
                meta["corrected_acc"] = _new_acc

                audit_rows.append({
                    "task": task,
                    "clip": key,
                    "mode": CAUSE_MODE,
                    "original_acc_si": _base_acc * 25.0,
                    "modified_acc_si": _new_acc * 25.0,
                    "original_speed_rms": _base_speed,
                    "modified_speed_rms": _new_speed,
                    "max_phase_shift_frames": _max_shift,
                    "saved_corrected_acc_si": _saved_acc * 25.0,
                    "saved_runtime_absdiff_si": (
                        abs(_new_acc - _saved_acc) * 25.0
                        if CAUSE_MODE == "boundary" else np.nan
                    ),
                    "strict_acc_valid": bool(
                        _new_acc <= 0.009832167363499603
                    ),
                    "matched_speed_pair_gap": _pair_speed_gap,
                    "matched_speed_pair_acc_gap": _pair_acc_gap,
                    "matched_speed_pair_valid": bool(
                        _pair_speed_gap <= 0.03
                        and _pair_acc_gap >= 0.20
                    ),
                })
'''

geometry_patch = '''
                        for _vi, _label in enumerate(
                            ("base", "base_repeat", "corrected")
                        ):
                            _se3, _pair, _scale = _cause_geometry(
                                gt0, pred[_vi, t, m]
                            )
                            geom_rows.append({
                                "task": task,
                                "clip": key,
                                "object_id": oid,
                                "frame": t,
                                "variant": _label,
                                "se3_residual_mm": _se3,
                                "pairwise_error_mm": _pair,
                                "abs_log_radius_scale": _scale,
                                "pa_mpjpe_mm": frame_vals[_label],
                            })
'''

for mode in ("boundary", "freq1", "freq3"):
    s = original

    old_root = "acc_only_boundary_pa_final_current10457"
    new_root = f"acc_causal_geometry_{mode}_20261010"
    assert s.count(old_root) == 1, (
        mode, "Output path not unique"
    )
    s = s.replace(old_root, new_root)

    anchor = "def main():"
    assert s.count(anchor) == 1
    s = s.replace(
        anchor,
        helper +
        f'CAUSE_MODE = "{mode}"\n\n'
        + anchor +
        "\n    geom_rows = []\n    audit_rows = []",
    )

    assert s.count("    cli = parser.parse_args()") == 1
    s = s.replace(
        "    cli = parser.parse_args()",
        '    parser.add_argument("--limit", type=int, default=0)\n'
        "    cli = parser.parse_args()",
    )

    strict_anchor = (
        "    target_keys = set(\n"
        "        corr.index.astype(str)\n"
        "    )"
    )
    assert s.count(strict_anchor) == 1
    s = s.replace(strict_anchor, '''
    strict = pd.read_csv(
        "diagnostics/results/cvpr_boundary_strict_20261010/"
        "boundary_valid_2677.csv"
    )
    assert len(strict) == 2677
    assert not strict[["task", "clip"]].duplicated().any()

    target_keys = set(
        strict.loc[
            strict["task"].astype(str) == task, "clip"
        ].astype(str)
    )
    assert target_keys.issubset(
        set(corr.index.astype(str))
    )
''')

    assert s.count("    if not target_keys:") == 1
    s = s.replace(
        "    if not target_keys:",
        '''    if cli.limit > 0:
        target_keys = set(
            sorted(target_keys)[:cli.limit]
        )

    if not target_keys:''',
    )

    marker = (
        "                # Original / corrected Action."
    )
    assert s.count(marker) == 1
    s = s.replace(marker, phase_patch + "\n" + marker)

    marker = (
        "                        if not ok:\n"
        "                            continue"
    )
    assert s.count(marker) == 1
    s = s.replace(
        marker,
        marker + "\n" + geometry_patch,
    )

    marker = "        c = pd.DataFrame("
    assert s.count(marker) == 1
    s = s.replace(
        marker,
        '''        save_csv(
            outdir / "mechanism_geometry_per_frame.csv",
            geom_rows,
        )
        save_csv(
            outdir / "mechanism_action_audit.csv",
            audit_rows,
        )

''' + marker,
    )

    dst = D / f"eval_acc_causal_geometry_{mode}_20261010.py"
    compile(s, str(dst), "exec")
    if dst.exists():
        assert dst.read_text() == s, (
            "Existing generated file differs:", str(dst)
        )
    else:
        dst.write_text(s)

    print("READY", mode, dst)

print("THREE_CAUSAL_EVALUATORS_READY")
