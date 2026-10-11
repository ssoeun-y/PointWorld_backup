from pathlib import Path
import re
import ast
import py_compile

ROOT = Path("diagnostics")
source_path = ROOT / "eval_acc_causal_geometry_boundary_20261010.py"
source = source_path.read_text()

# Only modify the phase that is passed to PointWorld.
anchor = "_chosen_phase = corrected_phase * float(T - 1)"
assert source.count(anchor) == 1, (
    "Phase location changed; stopping to avoid invalid evaluation"
)

# Output directory is isolated for every dose.
output_patterns = [
    "acc_causal_geometry_boundary_20261010",
    "acc_causal_geometry_{CAUSE_MODE}_20261010",
]

def output_patch(s, name):
    hits = sum(s.count(x) for x in output_patterns)
    assert hits > 0, (
        "Cannot identify evaluator output directory safely. "
        "No script will be created."
    )
    for old in output_patterns:
        s = s.replace(old, f"acc_causal_geometry_{name}_20261011")
    return s

for percent in (25, 50, 75):
    alpha = percent / 100.0
    name = f"dose{percent}"
    dst = ROOT / f"eval_acc_causal_geometry_{name}_20261011.py"

    s = output_patch(source, name)

    replacement = f"""
                # Dose-response: interpolate temporal phase,
                # keeping endpoints and spatial action path.
                _dose_alpha = {alpha!r}
                _dose_target_phase = corrected_phase.copy()
                corrected_phase = (
                    (1.0 - _dose_alpha) * base_phase
                    + _dose_alpha * _dose_target_phase
                )
                assert len(corrected_phase) == T
                assert abs(corrected_phase[0]) < 1e-10
                assert abs(corrected_phase[-1] - 1.0) < 1e-10
                assert np.all(np.diff(corrected_phase) > 0)
                _chosen_phase = corrected_phase * float(T - 1)
"""

    s = s.replace(anchor, replacement, 1)
    ast.parse(s)
    dst.write_text(s)
    py_compile.compile(str(dst), doraise=True)
    print(f"CREATED: {dst} (phase dose={percent}%)")

print("ALL DOSE SCRIPTS READY")
