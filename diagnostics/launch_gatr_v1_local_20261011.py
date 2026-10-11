from pathlib import Path
import json
import ast
import py_compile
import numpy as np
import pandas as pd

ROOT = Path("diagnostics")
RES = ROOT / "results"
OUT = RES / "gatr_v1_local_20261011"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------
# 1. Derive temporal geometric risk from Fixed
# Boundary predictions. No future GT is used.
# ---------------------------------------------------
rows = []

for task in ("0000", "0001", "0006", "0008", "0010"):
    p = (
        RES / "acc_causal_geometry_boundary_20261010"
        / f"task-{task}"
        / "mechanism_geometry_per_frame.csv"
    )
    d = pd.read_csv(p)
    rows.append(
        d[[
            "task", "clip", "frame", "variant",
            "pairwise_error_mm"
        ]]
    )

g = pd.concat(rows, ignore_index=True)

p = g.pivot_table(
    index=["task", "clip", "frame"],
    columns="variant",
    values="pairwise_error_mm",
    aggfunc="mean"
)

required = {"base", "base_repeat", "corrected"}
assert required <= set(p.columns)
p = p.dropna(subset=list(required))

risk_map = {}

for (task, clip), z in p.groupby(level=[0, 1]):
    z = z.droplevel([0, 1]).sort_index()

    if list(z.index) != list(range(1, 11)):
        continue

    fixed = z["corrected"].to_numpy(float)
    repeat_noise = abs(
        z["base"].to_numpy(float)
        - z["base_repeat"].to_numpy(float)
    )

    # New geometric deformation accumulated at
    # each future frame, above a repeat-noise margin.
    growth = np.diff(fixed, prepend=fixed[0])
    risk = np.maximum(
        growth - 2.0 * repeat_noise, 0.0
    )
    risk[0] = 0.0

    risk_map[f"{task}||{clip}"] = risk.tolist()

assert len(risk_map) >= 2600, len(risk_map)

(OUT / "temporal_geometry_risk.json").write_text(
    json.dumps(risk_map)
)

print("GEOMETRY RISK READY:", len(risk_map))

# ---------------------------------------------------
# 2. Upgrade verified Boundary evaluator.
# ---------------------------------------------------
src = (
    ROOT / "eval_acc_causal_geometry_boundary_20261010.py"
).read_text()

phase_anchor = (
    "_chosen_phase = corrected_phase * float(T - 1)"
)
base_anchor = "base_unreleased = ("
audit_anchor = '"max_phase_shift_frames": _max_shift,'

assert src.count("def main():") == 1
assert src.count(phase_anchor) == 1
assert src.count(base_anchor) == 1
assert src.count(audit_anchor) == 1

helper = '''
_GATR_RISK = None

def _gatr_v1_local(task, clip, right, left, fixed_norm, T):
    import json
    import numpy as np
    from pathlib import Path

    global _GATR_RISK

    if _GATR_RISK is None:
        path = Path(
            "diagnostics/results/gatr_v1_local_20261011/"
            "temporal_geometry_risk.json"
        )
        _GATR_RISK = json.loads(path.read_text())

    risk = np.asarray(
        _GATR_RISK.get(f"{task}||{clip}", []),
        dtype=np.float64
    )

    fixed = np.asarray(
        fixed_norm, dtype=np.float64
    ) * (T - 1)

    fixed_acc, fixed_speed = _cause_motion(
        right, left, fixed
    )

    info = {
        "applied": False,
        "risk_peak_mm": (
            float(np.max(risk)) if len(risk) else 0.0
        ),
        "fixed_acc_si": fixed_acc * 25.0,
        "fixed_speed_rms": fixed_speed,
        "weighted_acc_gain_pct": 0.0,
        "local_phase_change_frames": 0.0,
        "candidates_tested": 0,
        "feasible_candidates": 0,
    }

    if (
        T != 11
        or len(risk) != 10
        or not np.all(np.isfinite(risk))
        or np.max(risk) <= 1e-4
    ):
        return fixed_norm.copy(), info

    # Weight the action acceleration at frames
    # where fixed prediction's rigid geometry
    # becomes newly unstable.
    weights = (
        1.0 + 6.0 * risk[1:10]
        / max(np.max(risk), 1e-9)
    )

    def local_energy(phase_frames):
        rp, lp = poses_at_phase(
            right, left, phase_frames
        )
        ar = np.diff(
            rp[:, :3], n=2, axis=0
        ) * (25.0 ** 2)
        al = np.diff(
            lp[:, :3], n=2, axis=0
        ) * (25.0 ** 2)

        local_squared = 0.5 * (
            np.sum(ar ** 2, axis=1)
            + np.sum(al ** 2, axis=1)
        )
        return float(
            np.average(
                local_squared, weights=weights
            )
        )

    original_energy = local_energy(fixed)
    if original_energy <= 1e-12:
        return fixed_norm.copy(), info

    # Localized candidate centers: choose up to
    # three high-risk interior moments.
    centers = np.argsort(
        -risk[1:9]
    )[:3] + 2

    u = np.arange(T, dtype=float)
    best_phase = fixed.copy()
    best_objective = original_energy

    for center in centers:
        for width in (0.8, 1.5):
            bump = np.exp(
                -0.5 * ((u-center)/width)**2
            )
            bump[0] = bump[-1] = 0.0

            for amplitude in (
                -0.32, -0.20, -0.10, -0.05,
                 0.05,  0.10,  0.20,  0.32
            ):
                candidate = fixed + amplitude*bump
                candidate[0] = 0.0
                candidate[-1] = float(T-1)

                info["candidates_tested"] += 1

                if np.min(np.diff(candidate)) <= 0.10:
                    continue

                acc, speed = _cause_motion(
                    right, left, candidate
                )

                # Same established native-unit Boundary;
                # tolerance corresponds to 1e-6 SI.
                if acc > (
                    0.009832167363499603
                    + 1e-6 / 25.0
                ):
                    continue

                speed_gap = abs(
                    speed-fixed_speed
                ) / max(
                    0.5 * (speed+fixed_speed),
                    1e-9
                )

                if speed_gap > 0.03:
                    continue

                shift = float(np.max(
                    abs(candidate-fixed)
                ))

                if shift > 0.35:
                    continue

                info["feasible_candidates"] += 1

                energy = local_energy(candidate)

                # Penalize unnecessarily large phase
                # modifications within the trust region.
                objective = (
                    energy
                    + 0.005 * original_energy
                    * (shift/0.35)**2
                )

                if objective < 0.99 * best_objective:
                    best_objective = objective
                    best_phase = candidate.copy()

    if not np.array_equal(best_phase, fixed):
        info["applied"] = True
        info["weighted_acc_gain_pct"] = (
            100.0
            * (original_energy-best_objective)
            / original_energy
        )
        info["local_phase_change_frames"] = float(
            np.max(abs(best_phase-fixed))
        )

    return best_phase/float(T-1), info

'''

src = src.replace(
    "def main():",
    helper + "\ndef main():",
    1
)

patch = '''
                _gatr_fixed_phase = np.asarray(
                    corrected_phase, dtype=np.float64
                ).copy()

                corrected_phase, _gatr_info = _gatr_v1_local(
                    task, key, _right, _left,
                    _gatr_fixed_phase, T
                )

                _gatr_fixed_acc, _gatr_fixed_speed = (
                    _cause_motion(
                        _right, _left,
                        _gatr_fixed_phase * float(T - 1)
                    )
                )

                _chosen_phase = (
                    corrected_phase * float(T - 1)
                )
'''

src = src.replace(phase_anchor, patch, 1)

# IMPORTANT: in this experiment, "base" must be
# the FIXED BOUNDARY, not the original action.
# One forward: Fixed + Fixed repeat + GATR v1.
src = src.replace(
    base_anchor,
    '''base_phase = _gatr_fixed_phase.copy()
                meta["original_acc"] = _gatr_fixed_acc

                base_unreleased = (''',
    1
)

src = src.replace(
    audit_anchor,
    audit_anchor + '''
                    "gatr_applied": _gatr_info["applied"],
                    "gatr_risk_peak_mm": _gatr_info["risk_peak_mm"],
                    "gatr_fixed_acc_si": _gatr_info["fixed_acc_si"],
                    "gatr_fixed_speed_rms": _gatr_info["fixed_speed_rms"],
                    "gatr_weighted_acc_gain_pct":
                        _gatr_info["weighted_acc_gain_pct"],
                    "gatr_local_phase_change_frames":
                        _gatr_info["local_phase_change_frames"],
                    "gatr_candidates_tested":
                        _gatr_info["candidates_tested"],
                    "gatr_feasible_candidates":
                        _gatr_info["feasible_candidates"],
                    "gatr_boundary_valid": bool(
                        _new_acc * 25.0
                        <= 0.24580418408749 + 1e-6
                    ),
''',
    1
)

old_output = "acc_causal_geometry_boundary_20261010"
assert old_output in src
src = src.replace(
    old_output,
    "acc_gatr_v1_local_eval_20261011"
)

dest = ROOT / "eval_gatr_v1_local_20261011.py"
ast.parse(src)
dest.write_text(src)
py_compile.compile(str(dest), doraise=True)

print("GATR V1 EVALUATOR READY:", dest)
print("BASE VARIANT = FIXED BOUNDARY")
print("CORRECTED VARIANT = GATR LOCAL")
