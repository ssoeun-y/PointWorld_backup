from pathlib import Path
import ast
import hashlib
import os
import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO / "diagnostics/results/iff_offline_online_20261010"
BOUNDARY_NATIVE = 0.009832167363499603
TOL_NATIVE = 1e-6 / 25.0


def perturb_meta_phase(meta, built, key, variant):
    from diagnostics.build_acc_only_boundary_correction_current10457 import (
        acc_at_phase,
    )

    p0 = np.asarray(meta["phase"], dtype=np.float64).copy()
    T = len(p0)

    assert T >= 5
    assert abs(p0[0]) < 1e-8
    assert abs(p0[-1] - (T - 1)) < 1e-8

    # Match official PointWorld gripper filtering.
    # Missing gripper is filled with a stationary identity pose,
    # exactly as in canonicalize_gripper_keys_and_flags.
    def selected_pose(side):
        value = built.get(f"{side}_gripper_pose")
        if value is None:
            pose = np.zeros((T, 7), dtype=np.float64)
            pose[:, 6] = 1.0
            return pose

        pose = np.asarray(value, dtype=np.float64)
        assert pose.shape == (T, 7), (
            key, side, pose.shape
        )
        return pose

    right = selected_pose("right")
    left = selected_pose("left")

    assert len(right) == len(left) == T, (
        right.shape, left.shape, T
    )

    tau = float(meta["boundary"])
    assert abs(tau - BOUNDARY_NATIVE) < 1e-8

    def acc(p):
        return float(
            acc_at_phase(right, left, p)["rms_acceleration"]
        )

    a0 = acc(p0)

    if "corrected_acc" in meta:
        stored = float(meta["corrected_acc"])
        assert abs(a0 - stored) <= TOL_NATIVE, (
            key, "Official acceleration mismatch",
            a0, stored
        )

    assert a0 <= tau + TOL_NATIVE, (
        key, "Original corrected phase violates strict boundary",
        a0, tau
    )

    u = np.linspace(0, 1, T)
    envelope = np.sin(np.pi * u)

    waves = {
        1: np.sin(2 * np.pi * u),
        2: -np.sin(2 * np.pi * u),
        3: np.sin(4 * np.pi * u),
    }
    assert variant in waves

    # Counterfactual local timing variations.
    target = p0 + 0.24 * envelope * waves[variant]
    target[0] = 0.0
    target[-1] = float(T - 1)

    def expand(x):
        return np.r_[
            0.0,
            np.asarray(x, dtype=float),
            float(T - 1),
        ]

    def objective(x):
        p = expand(x)
        return float(
            np.mean((p - target) ** 2)
            + 1e-4 * np.mean(
                (np.diff(p) - np.diff(p0)) ** 2
            )
        )

    # Slightly inside the allowed numerical tolerance.
    tau_safe = tau + 0.5 * TOL_NATIVE

    constraints = [
        {
            "type": "ineq",
            "fun": lambda x:
                np.diff(expand(x)) - 1e-4,
        },
        {
            "type": "ineq",
            "fun": lambda x:
                tau_safe - acc(expand(x)),
        },
    ]

    res = minimize(
        objective,
        p0[1:-1],
        method="SLSQP",
        constraints=constraints,
        options={
            "maxiter": 55,
            "ftol": 1e-10,
            "disp": False,
        },
    )

    p1 = expand(res.x)
    a1 = acc(p1)

    accepted = bool(
        np.isfinite(a1)
        and a1 <= tau + TOL_NATIVE
        and np.min(np.diff(p1)) >= 1e-4 - 1e-7
        and np.max(np.abs(p1 - p0)) >= 0.005
    )

    if not accepted:
        p1 = p0.copy()
        a1 = a0

    # Keep per-clip candidate and feasibility audit.
    out = EXP_ROOT / f"v{variant}_phases"
    out.mkdir(parents=True, exist_ok=True)

    h = hashlib.sha256(key.encode()).hexdigest()[:20]

    np.savez_compressed(
        out / f"{h}.npz",
        clip=key,
        variant=variant,
        original_phase=p0,
        candidate_phase=p1,
        orig_acc_native=a0,
        candidate_acc_native=a1,
        accepted=accepted,
        optimizer_success=bool(res.success),
        phase_delta_max=float(
            np.max(np.abs(p1 - p0))
        ),
    )

    print(
        "IFF_PHASE",
        key,
        "variant", variant,
        "acc_SI", a1 * 25,
        "accepted", accepted,
        "max_phase_delta",
        np.max(np.abs(p1 - p0)),
        flush=True,
    )

    new_meta = dict(meta)
    new_meta["phase"] = p1
    new_meta["corrected_acc"] = a1

    return new_meta


def build():
    src = (
        REPO
        / "diagnostics/eval_acc_only_boundary_pa_final.py"
    )
    source = src.read_text()
    tree = ast.parse(source)

    # Locate the exact existing phase-load operation.
    nodes = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "meta"
            for t in n.targets
        )
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "load_phase"
    ]

    if len(nodes) != 1:
        raise RuntimeError(
            f"Expected 1 meta=load_phase node, "
            f"found {len(nodes)}"
        )

    if "acc_only_boundary_pa_final_current10457" not in source:
        raise RuntimeError(
            "Original evaluation output root not found. "
            "Aborting to protect existing results."
        )

    n = nodes[0]
    lines = source.splitlines(keepends=True)
    line = lines[n.lineno - 1]
    indent = line[:len(line) - len(line.lstrip())]

    for variant in (1, 2, 3):
        injection = (
            f"{indent}from diagnostics."
            f"iff_counterfactual_probe_20261010 "
            f"import perturb_meta_phase\n"
            f"{indent}meta = perturb_meta_phase("
            f"meta, decoded, key, {variant})\n"
        )

        modified = lines.copy()
        modified.insert(n.end_lineno, injection)

        out = "".join(modified).replace(
            "acc_only_boundary_pa_final_current10457",
            f"iff_teacher_probe_v{variant}_20261010",
        )

        dest = (
            REPO
            / f"diagnostics/iff_teacher_eval_autogen_v{variant}.py"
        )

        compile(out, str(dest), "exec")
        dest.write_text(out)

        print("GENERATED:", dest)


if __name__ == "__main__":
    import subprocess

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        cwd=REPO,
    ).strip()

    assert commit == (
        "05484826dfef74cbe278a3974179a5a16705d35d"
    ), commit

    build()
