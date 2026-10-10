from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    "diagnostics/results/action_conditioned_active_world_model/"
    "uncertainty_gate_5task"
)

INPUT = ROOT / "gate_analysis_input.csv"

# Same setting used throughout current reliability analysis.
SEED = 42
HIGH_RISK_Q = 0.80

# Candidate temporal warps.
# gamma=1.0 is the original trajectory.
GAMMAS = np.array([
    0.60, 0.70, 0.80, 0.90,
    1.00,
    1.10, 1.20, 1.30, 1.40,
], dtype=np.float64)

PERTURB_LIMITS_MM = [10, 20, 30, 50]

# ============================================================
# LOAD ORIGINAL FEATURE IMPLEMENTATION SAFELY
# ============================================================

# Do NOT import extract_head_scene_action_features.py directly:
# that script has top-level execution / cache-exit side effects.
#
# Instead, parse its source and execute ONLY the exact original
# feature functions + DT assignment.

import ast

FEATURE_SOURCE = Path(
    "diagnostics/extract_head_scene_action_features.py"
)

src = FEATURE_SOURCE.read_text()
tree = ast.parse(src)

KEEP_FUNCS = {
    "rms",
    "p95",
    "quaternion_step_angles",
    "trajectory_features",
    "gripper_features",
    "joint_features",
}

selected_nodes = []

for node in tree.body:

    # Preserve DT exactly as defined in original source.
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        names = []

        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.append(node.target.id)

        if "DT" in names:
            selected_nodes.append(node)

    # Preserve exact original feature functions.
    if isinstance(node, ast.FunctionDef):
        if node.name in KEEP_FUNCS:
            selected_nodes.append(node)

module = ast.Module(
    body=selected_nodes,
    type_ignores=[],
)

ast.fix_missing_locations(module)

feature_ns = {
    "np": np,
}

exec(
    compile(
        module,
        str(FEATURE_SOURCE),
        "exec",
    ),
    feature_ns,
)

DT = feature_ns["DT"]
rms = feature_ns["rms"]
p95 = feature_ns["p95"]
quaternion_step_angles = feature_ns["quaternion_step_angles"]
trajectory_features = feature_ns["trajectory_features"]
gripper_features = feature_ns["gripper_features"]
joint_features = feature_ns["joint_features"]

print(
    "Loaded exact original feature definitions safely | "
    f"DT={DT}"
)

# ============================================================
# HELPERS
# ============================================================

def slerp(q0, q1, u):
    """
    Minimal quaternion SLERP.
    Assumes quaternion ordering is unchanged from source data.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)

    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)

    dot = float(np.dot(q0, q1))

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / max(np.linalg.norm(q), 1e-12)

    theta = np.arccos(dot)
    s = np.sin(theta)

    return (
        np.sin((1.0-u)*theta) / s * q0
        + np.sin(u*theta) / s * q1
    )


def interp_pose(pose, src_phase):
    """
    Resample a pose trajectory at arbitrary normalized phases.
    Endpoint is exactly preserved.
    """
    pose = np.asarray(pose, dtype=np.float64)
    T = len(pose)

    old = np.linspace(0.0, 1.0, T)
    out = np.zeros_like(pose, dtype=np.float64)

    # XYZ
    for j in range(3):
        out[:, j] = np.interp(
            src_phase,
            old,
            pose[:, j],
        )

    # quaternion
    for k, x in enumerate(src_phase):
        if x <= 0:
            out[k, 3:7] = pose[0, 3:7]
            continue
        if x >= 1:
            out[k, 3:7] = pose[-1, 3:7]
            continue

        z = x * (T - 1)
        i = min(int(np.floor(z)), T - 2)
        u = z - i

        out[k, 3:7] = slerp(
            pose[i, 3:7],
            pose[i+1, 3:7],
            u,
        )

    # hard endpoint preservation
    out[0] = pose[0]
    out[-1] = pose[-1]

    return out


def interp_array(x, src_phase):
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    old = np.linspace(0.0, 1.0, T)

    if x.ndim == 1:
        out = np.interp(src_phase, old, x)
    else:
        out = np.stack([
            np.interp(src_phase, old, x[:, j])
            for j in range(x.shape[1])
        ], axis=1)

    out[0] = x[0]
    out[-1] = x[-1]
    return out


def interp_gripper(x, src_phase):
    """
    Nearest-neighbour resampling because gripper is discrete.
    Endpoint preserved exactly.
    """
    x = np.asarray(x).reshape(-1)
    T = len(x)

    idx = np.rint(
        src_phase * (T - 1)
    ).astype(int)

    idx = np.clip(idx, 0, T-1)
    out = x[idx].copy()

    out[0] = x[0]
    out[-1] = x[-1]

    return out


def make_retimed(sample, gamma):
    """
    Pure temporal retiming.

    gamma < 1 and > 1 redistribute progress through the
    SAME trajectory while keeping start/end exactly fixed.
    """
    T = len(sample["left_pose"])

    phase = np.linspace(0.0, 1.0, T)

    # Warp sampling phase.
    src_phase = phase ** gamma
    src_phase[0] = 0.0
    src_phase[-1] = 1.0

    return {
        "left_pose":
            interp_pose(sample["left_pose"], src_phase),

        "right_pose":
            interp_pose(sample["right_pose"], src_phase),

        "left_open":
            interp_gripper(sample["left_open"], src_phase),

        "right_open":
            interp_gripper(sample["right_open"], src_phase),

        "joints":
            interp_array(sample["joints"], src_phase),
    }


def action_features(sample):
    feat = {}

    feat.update(
        trajectory_features(
            "left",
            sample["left_pose"],
        )
    )

    feat.update(
        trajectory_features(
            "right",
            sample["right_pose"],
        )
    )

    feat.update(
        gripper_features(
            "left",
            sample["left_open"],
        )
    )

    feat.update(
        gripper_features(
            "right",
            sample["right_open"],
        )
    )

    feat.update(
        joint_features(
            sample["joints"],
        )
    )

    return feat


def endpoint_error(a, b):
    errs = []

    for key in ["left_pose", "right_pose"]:
        errs.append(
            np.max(
                np.abs(
                    np.asarray(a[key])[[0, -1]]
                    - np.asarray(b[key])[[0, -1]]
                )
            )
        )

    errs.append(
        np.max(
            np.abs(
                np.asarray(a["joints"])[[0, -1]]
                - np.asarray(b["joints"])[[0, -1]]
            )
        )
    )

    return float(max(errs))


# ============================================================
# LOCATE ACTION TRAJECTORIES
# ============================================================

FLOW_ROOT = Path(
    "diagnostics/data/"
    "pointworld_behavior_multitask_restored/"
    "behavior/flows"
)

import h5py


def read_clip_from_hdf5(task, episode, clip_key):
    """
    Read only required action arrays.
    """
    candidates = list(
        (FLOW_ROOT / f"task-{task}").glob(
            f"*{episode}*.hdf5"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            f"No HDF5 for task={task}, episode={episode}"
        )

    p = candidates[0]

    with h5py.File(p, "r") as f:
        g = f[str(clip_key)]

        def find_dataset(names):
            for n in names:
                if n in g:
                    return np.asarray(g[n])

            # recursive suffix fallback
            hits = []

            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    for n in names:
                        if name.endswith(n):
                            hits.append(name)

            g.visititems(visitor)

            if not hits:
                raise KeyError(
                    f"{names} not found in {p}:{clip_key}"
                )

            return np.asarray(g[hits[0]])

        return {
            "left_pose": find_dataset([
                "left_gripper_pose",
                "robot/left_gripper_pose",
            ]),
            "right_pose": find_dataset([
                "right_gripper_pose",
                "robot/right_gripper_pose",
            ]),
            "left_open": find_dataset([
                "left_gripper_open",
                "robot/left_gripper_open",
            ]),
            "right_open": find_dataset([
                "right_gripper_open",
                "robot/right_gripper_open",
            ]),
            "joints": find_dataset([
                "joint_positions",
                "robot/joint_positions",
            ]),
        }


# ============================================================
# LOAD GATE DATA
# ============================================================

df = pd.read_csv(INPUT)
df["task"] = df["task"].astype(str).str.zfill(4)

action_cols = sorted([
    c for c in df.columns
    if c.startswith("action_")
])

print("=" * 90)
print("RELIABILITY-GUIDED ACTION REFINEMENT — FEASIBILITY POC")
print("=" * 90)
print("rows =", len(df))
print("action features =", len(action_cols))


# ============================================================
# IDENTIFY SAMPLE-ID COLUMNS
# ============================================================

print("\nAvailable ID-like columns:")
id_like = [
    c for c in df.columns
    if any(
        k in c.lower()
        for k in [
            "episode", "clip", "window",
            "sample", "key"
        ]
    )
]
print(id_like)

episode_col = next(
    (
        c for c in [
            "episode",
            "episode_id",
            "episode_idx",
            "episode_name",
        ]
        if c in df.columns
    ),
    None
)

clip_col = next(
    (
        c for c in [
            "clip",
            "clip_key",
            "window",
            "window_key",
            "sample_key",
            "key",
        ]
        if c in df.columns
    ),
    None
)

if episode_col is None or clip_col is None:
    print("\nSTOP: Need exact episode/clip mapping columns.")
    print("Columns:")
    print(df.columns.tolist())
    raise SystemExit(2)


# ============================================================
# EPISODE-DISJOINT OOF PREDICTIONS
# ============================================================

df["predicted_error_mm"] = np.nan

for fold in sorted(df["fold"].astype(int).unique()):
    tr = df["fold"].astype(int) != fold
    te = ~tr

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=SEED,
    )

    model.fit(
        df.loc[tr, action_cols],
        df.loc[tr, "base_error_mm"],
    )

    df.loc[te, "predicted_error_mm"] = model.predict(
        df.loc[te, action_cols]
    )


# ============================================================
# TRUE HIGH-RISK SET
# Used ONLY to define the evaluation population.
# Candidate selection NEVER sees GT.
# ============================================================

true_threshold = np.quantile(
    df["base_error_mm"],
    HIGH_RISK_Q,
)

high = df[
    df["base_error_mm"] >= true_threshold
].copy()

print(
    f"\nTrue high-risk evaluation set: "
    f"{len(high)}/{len(df)} "
    f"({100*len(high)/len(df):.1f}%)"
)
print(
    f"pooled 80th-percentile moved-L2 threshold = "
    f"{true_threshold:.3f} mm"
)


# ============================================================
# FIT DEPLOYMENT MODEL
# For this feasibility test only.
# It never uses candidate GT.
# ============================================================

gate = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=SEED,
)

gate.fit(
    df[action_cols],
    df["base_error_mm"],
)

pred_failure_boundary = float(
    np.quantile(
        gate.predict(df[action_cols]),
        HIGH_RISK_Q,
    )
)

print(
    f"Predicted-risk 80th percentile boundary = "
    f"{pred_failure_boundary:.3f} mm"
)


# ============================================================
# REFINEMENT
# ============================================================

rows = []
skipped = 0

for n, (_, r) in enumerate(high.iterrows(), 1):

    task = str(r["task"]).zfill(4)

    episode_raw = str(r[episode_col])
    clip_raw = str(r[clip_col])

    # CSV:
    # task-0000_episode_00000030
    # task-0000_episode_00000030-150:161
    #
    # HDF5:
    # episode_00000030.hdf5
    # group key: 150:161
    episode = episode_raw.split("_episode_")[-1]
    episode = f"episode_{episode}"

    clip_key = clip_raw.rsplit("-", 1)[-1]

    try:
        original = read_clip_from_hdf5(
            task,
            episode,
            clip_key,
        )

        # Recompute original features using EXACT feature code.
        f0 = action_features(original)

        X0 = pd.DataFrame(
            [[f0[c] for c in action_cols]],
            columns=action_cols,
        )

        original_pred = float(
            gate.predict(X0)[0]
        )

        candidates = []

        for gamma in GAMMAS:
            cand = make_retimed(
                original,
                float(gamma),
            )

            feat = action_features(cand)

            X = pd.DataFrame(
                [[feat[c] for c in action_cols]],
                columns=action_cols,
            )

            pred = float(
                gate.predict(X)[0]
            )

            ep_err = endpoint_error(
                original,
                cand,
            )

            # Internal Cartesian perturbation.
            left_delta = np.linalg.norm(
                cand["left_pose"][:, :3]
                - original["left_pose"][:, :3],
                axis=1,
            )

            right_delta = np.linalg.norm(
                cand["right_pose"][:, :3]
                - original["right_pose"][:, :3],
                axis=1,
            )

            perturb_mm = float(
                1000.0 * max(
                    left_delta.max(),
                    right_delta.max(),
                )
            )

            candidates.append({
                "gamma": float(gamma),
                "pred": pred,
                "endpoint_error": ep_err,
                "perturb_mm": perturb_mm,
            })

        # Candidate chosen ONLY using predicted error.
        best = min(
            candidates,
            key=lambda x: x["pred"],
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # First enforce perturbation limit, THEN choose
        # lowest-risk candidate within that feasible set.
        #
        # gamma=1.0 is original trajectory (0 mm perturbation),
        # so every constraint always has at least one candidate.
        # ----------------------------------------------------
        constrained = {}

        for lim in PERTURB_LIMITS_MM:
            feasible = [
                c for c in candidates
                if c["perturb_mm"] <= lim + 1e-9
            ]

            constrained[lim] = min(
                feasible,
                key=lambda x: x["pred"],
            )

        row = {
            "task": task,
            "episode": episode,
            "clip_key": clip_key,

            "true_base_error_mm":
                float(r["base_error_mm"]),

            "original_predicted_error_mm":
                original_pred,

            "best_predicted_error_mm":
                best["pred"],

            "predicted_reduction_mm":
                original_pred - best["pred"],

            "predicted_reduction_pct":
                100.0 * (
                    original_pred - best["pred"]
                ) / max(original_pred, 1e-12),

            "best_gamma":
                best["gamma"],

            "max_cartesian_perturb_mm":
                best["perturb_mm"],

            "endpoint_error":
                best["endpoint_error"],

            "original_above_boundary":
                int(
                    original_pred
                    >= pred_failure_boundary
                ),

            "refined_below_boundary":
                int(
                    original_pred
                    >= pred_failure_boundary
                    and best["pred"]
                    < pred_failure_boundary
                ),
        }

        # Save true constrained-selection result for every limit.
        for lim in PERTURB_LIMITS_MM:
            cbest = constrained[lim]

            row[
                f"c{lim}_best_predicted_error_mm"
            ] = cbest["pred"]

            row[
                f"c{lim}_predicted_reduction_pct"
            ] = (
                100.0
                * (original_pred - cbest["pred"])
                / max(original_pred, 1e-12)
            )

            row[
                f"c{lim}_perturb_mm"
            ] = cbest["perturb_mm"]

            row[
                f"c{lim}_best_gamma"
            ] = cbest["gamma"]

            row[
                f"c{lim}_risk_reduced"
            ] = int(
                cbest["pred"]
                < original_pred - 1e-12
            )

            row[
                f"c{lim}_crossed_boundary"
            ] = int(
                original_pred >= pred_failure_boundary
                and cbest["pred"] < pred_failure_boundary
            )

        rows.append(row)

    except Exception as e:
        skipped += 1

        if skipped <= 10:
            print(
                f"SKIP task={task} "
                f"episode={episode} "
                f"clip={clip_key}: {e}"
            )

    if n % 100 == 0:
        print(
            f"processed {n}/{len(high)} "
            f"| success={len(rows)} "
            f"| skipped={skipped}"
        )


res = pd.DataFrame(rows)

if len(res) == 0:
    raise RuntimeError(
        "No samples were matched to HDF5. "
        "Inspect ID mapping before proceeding."
    )


# ============================================================
# REPORT
# ============================================================

improved = (
    res["best_predicted_error_mm"]
    < res["original_predicted_error_mm"] - 1e-12
)

eligible = res[
    res["original_above_boundary"] == 1
]

crossed = (
    eligible["refined_below_boundary"].mean()
    if len(eligible)
    else np.nan
)

print("\n" + "=" * 90)
print("FEASIBILITY RESULT")
print("=" * 90)

print(
    f"Evaluated high-risk actions = {len(res)}"
)
print(
    f"Skipped = {skipped}"
)

print(
    f"\nPredicted risk reduced = "
    f"{100*improved.mean():.2f}%"
)

print(
    f"Median original predicted error = "
    f"{res['original_predicted_error_mm'].median():.3f} mm"
)

print(
    f"Median refined predicted error  = "
    f"{res['best_predicted_error_mm'].median():.3f} mm"
)

print(
    f"Median per-sample reduction = "
    f"{res['predicted_reduction_pct'].median():.2f}%"
)

print(
    f"\nOriginally above predicted failure boundary = "
    f"{len(eligible)}"
)

print(
    f"Moved below boundary after refinement = "
    f"{100*crossed:.2f}%"
)

print(
    f"\nMedian max Cartesian perturbation = "
    f"{res['max_cartesian_perturb_mm'].median():.3f} mm"
)

print(
    f"P95 max Cartesian perturbation = "
    f"{res['max_cartesian_perturb_mm'].quantile(.95):.3f} mm"
)

print(
    f"Maximum endpoint error = "
    f"{res['endpoint_error'].max():.3e}"
)

print("\n" + "=" * 90)
print("TRUE CONSTRAINED SELECTION")
print("=" * 90)

for lim in PERTURB_LIMITS_MM:

    pred_col = f"c{lim}_best_predicted_error_mm"
    red_col = f"c{lim}_predicted_reduction_pct"
    pert_col = f"c{lim}_perturb_mm"
    imp_col = f"c{lim}_risk_reduced"
    cross_col = f"c{lim}_crossed_boundary"

    eligible_lim = res[
        res["original_above_boundary"] == 1
    ]

    crossed_lim = (
        eligible_lim[cross_col].mean()
        if len(eligible_lim)
        else np.nan
    )

    print(
        f"\nMAX PERTURB <= {lim} mm"
        f"\nRisk reduced = "
        f"{100*res[imp_col].mean():.2f}%"
        f"\nMedian original predicted error = "
        f"{res['original_predicted_error_mm'].median():.3f} mm"
        f"\nMedian constrained predicted error = "
        f"{res[pred_col].median():.3f} mm"
        f"\nMedian per-sample reduction = "
        f"{res[red_col].median():.2f}%"
        f"\nMoved below failure boundary = "
        f"{100*crossed_lim:.2f}%"
        f"\nMedian selected perturbation = "
        f"{res[pert_col].median():.2f} mm"
        f"\nP95 selected perturbation = "
        f"{res[pert_col].quantile(.95):.2f} mm"
    )

    print("Taskwise:")
    print(
        res.groupby("task").agg(
            N=("task", "size"),
            risk_reduced=(imp_col, "mean"),
            reduction_pct=(red_col, "median"),
            selected_perturb_mm=(pert_col, "median"),
        ).assign(
            risk_reduced_pct=lambda x:
                100*x["risk_reduced"]
        )[
            [
                "N",
                "risk_reduced_pct",
                "reduction_pct",
                "selected_perturb_mm",
            ]
        ].to_string()
    )

print("\nBest gamma distribution:")
print(
    res["best_gamma"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
    .to_string()
)


# ============================================================
# TASKWISE
# ============================================================

print("\n" + "=" * 90)
print("TASKWISE")
print("=" * 90)

taskwise = (
    res.groupby("task")
    .agg(
        N=("task", "size"),
        original_pred_mm=(
            "original_predicted_error_mm",
            "median",
        ),
        refined_pred_mm=(
            "best_predicted_error_mm",
            "median",
        ),
        median_reduction_pct=(
            "predicted_reduction_pct",
            "median",
        ),
        median_perturb_mm=(
            "max_cartesian_perturb_mm",
            "median",
        ),
    )
)

print(taskwise.to_string())


# ============================================================
# SAVE
# ============================================================

OUT = Path(
    "diagnostics/results/action_conditioned_active_world_model/"
    "reliability_guided_action_refinement_poc"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

res.to_csv(
    OUT / "per_sample.csv",
    index=False,
)

taskwise.to_csv(
    OUT / "taskwise.csv"
)

print("\nOUTPUT =", OUT)
print("SUCCESS = True")
