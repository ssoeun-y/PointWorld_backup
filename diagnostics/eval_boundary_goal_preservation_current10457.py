#!/usr/bin/env python3

import os
import copy
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------
# Reuse the exact helper stack already used by the working
# Acc-only boundary intervention evaluator.
# ---------------------------------------------------------------------
import diagnostics.eval_acc_only_pa_intervention_5task as old


ROOT = Path("/home/soenchun/cse/PointWorld_workspace/PointWorld")

CORR_ROOT = (
    ROOT
    / "diagnostics/results/acc_only_boundary_correction_current10457"
)
CORR_CSV = CORR_ROOT / "correction_results.csv"
PHASE_DIR = CORR_ROOT / "corrected_actions"

# Existing final PA evaluator output.
# We reuse its object eligibility instead of redefining which object is valid.
PA_ROOT = (
    ROOT
    / "diagnostics/results/acc_only_boundary_pa_final_current10457"
)

OUT_ROOT = (
    ROOT
    / "diagnostics/results/acc_only_boundary_goal_preservation_current10457"
)

TASKS = [
    "task-0000",
    "task-0001",
    "task-0006",
    "task-0008",
    "task-0010",
]

EXPECTED_SUCCESS = {
    "task-0000": 145,
    "task-0001": 653,
    "task-0006": 552,
    "task-0008": 709,
    "task-0010": 624,
}

EXPECTED_TOTAL_SUCCESS = 2683

MIN_POINTS = 8
SEED = 42


# =====================================================================
# Utility
# =====================================================================

def mask3(batch, name):
    x = batch[name].bool()

    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.ndim != 3:
        raise RuntimeError(
            f"{name}: bad shape {tuple(x.shape)}"
        )

    return x


def scalar3(batch, name):
    x = batch[name]

    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.ndim != 3:
        raise RuntimeError(
            f"{name}: bad shape {tuple(x.shape)}"
        )

    return x



def scalar_str(x):
    if isinstance(x, np.ndarray):
        if x.size == 1:
            x = x.reshape(-1)[0]
        else:
            return str(x.tolist())

    if isinstance(x, bytes):
        return x.decode("utf-8")

    return str(x)


def raw_key(raw):
    k = raw["__key__"]
    if isinstance(k, bytes):
        k = k.decode("utf-8")
    return str(k)


def reset_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_clip_name(key):
    # This is the filename convention already used by corrected_actions.
    return str(key).replace("/", "_").replace(":", "__")


# =====================================================================
# Successful correction metadata
# =====================================================================

def successful_corrections(task):
    if not CORR_CSV.exists():
        raise FileNotFoundError(CORR_CSV)

    df = pd.read_csv(CORR_CSV)

    if "task" not in df.columns or "clip" not in df.columns:
        raise RuntimeError(
            f"{CORR_CSV} must contain task and clip columns. "
            f"Found: {list(df.columns)}"
        )

    if "success" not in df.columns:
        raise RuntimeError(
            f"{CORR_CSV} does not contain 'success'. "
            f"Found: {list(df.columns)}"
        )

    success = (
        df["success"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    all_ok = df.loc[success].copy()

    if len(all_ok) != EXPECTED_TOTAL_SUCCESS:
        raise RuntimeError(
            f"Successful correction population mismatch: "
            f"expected {EXPECTED_TOTAL_SUCCESS}, got {len(all_ok)}"
        )

    task_df = all_ok[all_ok["task"].astype(str) == task].copy()

    expected = EXPECTED_SUCCESS[task]
    if len(task_df) != expected:
        raise RuntimeError(
            f"{task}: expected {expected} successful corrections, "
            f"got {len(task_df)}"
        )

    if task_df["clip"].duplicated().any():
        dup = task_df.loc[
            task_df["clip"].duplicated(), "clip"
        ].head().tolist()
        raise RuntimeError(f"Duplicate clips in correction CSV: {dup}")

    return task_df.set_index("clip", drop=False)


def load_phase(key, expected_task):
    safe = safe_clip_name(key)
    path = PHASE_DIR / f"{safe}.npz"

    if not path.exists():
        # Fallback search only if exact sanitized path is absent.
        matches = list(PHASE_DIR.glob(f"{safe}*.npz"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Could not uniquely locate corrected-action NPZ for {key}. "
                f"Exact={path}, matches={len(matches)}"
            )
        path = matches[0]

    z = np.load(path, allow_pickle=True)

    required = ["clip", "task", "phase"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise RuntimeError(
            f"{path} missing keys {missing}; has {z.files}"
        )

    saved_clip = scalar_str(z["clip"])
    saved_task = scalar_str(z["task"])

    if saved_clip != key:
        raise RuntimeError(
            f"NPZ clip mismatch: requested={key}, saved={saved_clip}, "
            f"path={path}"
        )

    if saved_task != expected_task:
        raise RuntimeError(
            f"NPZ task mismatch: requested={expected_task}, "
            f"saved={saved_task}, path={path}"
        )

    out = {
        "path": str(path),
        "phase": np.asarray(z["phase"], dtype=np.float64).reshape(-1),
    }

    for name in [
        "original_acc",
        "corrected_acc",
        "boundary",
    ]:
        if name in z.files:
            out[name] = float(
                np.asarray(z[name]).reshape(-1)[0]
            )
        else:
            out[name] = np.nan

    return out


# =====================================================================
# Reuse exact object eligibility from the already-completed PA evaluator
# =====================================================================

def load_existing_pa_objects(task):
    path = PA_ROOT / task / "per_object_metrics.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Existing PA object output not found: {path}"
        )

    df = pd.read_csv(path)

    required = ["clip", "object_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{path} missing {missing}; columns={list(df.columns)}"
        )

    # Exactly one eligibility definition:
    # reuse objects that the existing final boundary PA evaluator accepted.
    keep_cols = ["clip", "object_id"]

    for c in ["object_full_name", "object_short_name"]:
        if c in df.columns:
            keep_cols.append(c)

    out = df[keep_cols].copy()
    out["clip"] = out["clip"].astype(str)
    out["object_id"] = out["object_id"].astype(int)

    out = out.drop_duplicates(
        subset=["clip", "object_id"]
    ).reset_index(drop=True)

    by_clip = {}
    for clip, g in out.groupby("clip", sort=False):
        objects = []

        for _, r in g.iterrows():
            objects.append({
                "object_id": int(r["object_id"]),
                "object_full_name": (
                    str(r["object_full_name"])
                    if "object_full_name" in r
                    and pd.notna(r["object_full_name"])
                    else ""
                ),
                "object_short_name": (
                    str(r["object_short_name"])
                    if "object_short_name" in r
                    and pd.notna(r["object_short_name"])
                    else ""
                ),
            })

        by_clip[str(clip)] = objects

    return by_clip


# =====================================================================
# Metrics
# =====================================================================

def alignment_metrics(reference, comparison):
    """
    reference:
        Original-action predicted final object points, (N,3)

    comparison:
        Corrected/base-repeat predicted final object points, (N,3)

    Both use exactly the same PointWorld point indices.

    Returns:
      raw_mpjpe_mm:
          Mean corresponding-point distance before alignment.

      centroid_shift_mm:
          Difference between predicted object centers.

      rigid_residual_mm:
          Residual after optimal R+t alignment, NO scale.

      pa_mpjpe_mm:
          Residual after optimal scale+R+t similarity alignment.
    """

    Y = np.asarray(reference, dtype=np.float64)
    X = np.asarray(comparison, dtype=np.float64)

    if X.shape != Y.shape:
        raise RuntimeError(
            f"Point shape mismatch: {X.shape} vs {Y.shape}"
        )

    if X.ndim != 2 or X.shape[1] != 3:
        raise RuntimeError(f"Expected Nx3 points, got {X.shape}")

    raw_mpjpe_mm = float(
        np.linalg.norm(X - Y, axis=1).mean() * 1000.0
    )

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)

    centroid_shift_mm = float(
        np.linalg.norm(mu_x - mu_y) * 1000.0
    )

    Xc = X - mu_x
    Yc = Y - mu_y

    # row-vector Kabsch:
    # Xc @ R ~= Yc
    H = Xc.T @ Yc
    U, S, Vt = np.linalg.svd(H)

    D = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1.0

    R = U @ D @ Vt

    X_rigid = Xc @ R + mu_y

    rigid_residual_mm = float(
        np.linalg.norm(X_rigid - Y, axis=1).mean() * 1000.0
    )

    denom = float(np.sum(Xc ** 2))

    if denom < 1e-12:
        pa_mpjpe_mm = np.nan
    else:
        signed_s = S * np.diag(D)
        scale = float(np.sum(signed_s) / denom)

        X_pa = scale * (Xc @ R) + mu_y

        pa_mpjpe_mm = float(
            np.linalg.norm(X_pa - Y, axis=1).mean() * 1000.0
        )

    return {
        "raw_mpjpe_mm": raw_mpjpe_mm,
        "centroid_shift_mm": centroid_shift_mm,
        "rigid_residual_mm": rigid_residual_mm,
        "pa_mpjpe_mm": pa_mpjpe_mm,
    }


# =====================================================================
# PointWorld setup
# =====================================================================

def setup_object_id_hooks():
    """
    Use the already-audited object-ID wrappers from the current
    diagnostics stack.

    They do NOT alter PointWorld point coordinates.
    They only propagate the parallel scene_object_id labels.
    """

    import dataset_components.dataloader as dlmod
    import dataset_components.pipeline as pmod

    dlmod.build_flow_sample = old.omap.build_flow_sample_with_object_ids
    pmod.grid_sample_transform = old.omap.grid_sample_with_object_ids
    pmod.gather = old.omap.gather_with_object_ids

    # Some diagnostic versions also import build_flow_sample through
    # decoders. Patch it only if that symbol exists.
    try:
        import dataset_components.decoders as decmod
        if hasattr(decmod, "build_flow_sample"):
            decmod.build_flow_sample = (
                old.omap.build_flow_sample_with_object_ids
            )
    except Exception:
        pass


def make_tester(task):
    if task not in TASKS:
        raise ValueError(task)

    args = old.parse_args(skip_command_line=True)

    # Reuse the constants from the already-working intervention evaluator.
    args.model_path = old.MODEL_PATH
    args.norm_stats_path = old.NORM_STATS

    args.domains = ["behavior"]
    args.data_dirs = [old.TASK_DIRS[task]]

    args.batch_size = 1
    args.num_workers = 0
    args.eval_num_workers = 0
    args.eval_num_batches = -1

    args.eval_skip_viz = True
    args.disable_compile = True

    args.seed = SEED
    args.device = "cuda"

    # Current official evaluation camera contract.
    args.eval_min_num_cameras = 2
    args.eval_max_num_cameras = 2

    tester = old.Tester(args)
    tester.model.eval()

    return args, tester


def make_robot_sampler(domain="behavior"):
    urdf_path = old.resolve_robot_urdf(domain)

    robot_sampler = old.TorchRobotSampler(
        urdf_path=urdf_path,
        gripper_only=old.RELEASE_GRIPPER_ONLY,
        device="cpu",
    )

    # BEHAVIOR evaluation is bimanual in the current protocol.
    force_single_arm = False

    return robot_sampler, force_single_arm


# =====================================================================
# One task
# =====================================================================

def run_task(task, resume=True):
    print("=" * 80)
    print(f"[TASK] {task}")
    print(f"[GPU ] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print("=" * 80)

    reset_seed(SEED)
    setup_object_id_hooks()

    task_meta = successful_corrections(task)
    eligible_by_clip = load_existing_pa_objects(task)

    target_clips = set(task_meta.index.astype(str))

    missing_pa = sorted(
        clip for clip in target_clips
        if clip not in eligible_by_clip
    )

    if missing_pa:
        raise RuntimeError(
            f"{task}: {len(missing_pa)} corrected clips are missing from "
            f"existing PA object output. Examples={missing_pa[:5]}"
        )

    task_out = OUT_ROOT / task
    task_out.mkdir(parents=True, exist_ok=True)

    object_csv = task_out / "per_object_endpoint_goal_metrics.csv"
    clip_csv = task_out / "per_clip_endpoint_goal_metrics.csv"

    old_rows = pd.DataFrame()
    done_clips = set()

    if resume and object_csv.exists():
        old_rows = pd.read_csv(object_csv)
        if "clip" in old_rows.columns:
            done_clips = set(
                old_rows["clip"].astype(str).unique()
            )
        print(f"[RESUME] already completed clips={len(done_clips)}")

    args, tester = make_tester(task)

    domain = "behavior"
    robot_sampler, force_single_arm = make_robot_sampler(domain)

    data_dir = old.TASK_DIRS[task]

    shard_paths = old.dlmod.gather_shard_paths(
        data_dir,
        "test",
        rank=tester.rank,
    )

    if not shard_paths:
        raise RuntimeError(f"No test shards found in {data_dir}")

    shard_paths = list(shard_paths)

    rng = random.Random(SEED + tester.rank)
    rng.shuffle(shard_paths)

    dataset = old.wds.WebDataset(
        shard_paths,
        shardshuffle=False,
        detshuffle=True,
        resampled=False,
        nodesplitter=(
            old.wds.split_by_node
            if tester.world_size > 1
            else None
        ),
        seed=SEED + tester.rank,
        handler=old.wds.warn_and_continue,
    )

    rows = []
    found_target = set(done_clips)

    with torch.no_grad():
        for raw in dataset:
            key = raw_key(raw)

            if key not in target_clips:
                continue

            if key in done_clips:
                continue

            meta = load_phase(key, task)

            decoded = old.decode_data(
                raw,
                domain=domain,
            )

            # Build ONCE from the official raw sample, then derive all
            # variants from this exact same built sample.
            built = old.omap.build_flow_sample_with_object_ids(
                decoded,
                domain=domain,
                robot_sampler=robot_sampler,
                max_robot_points=args.max_robot_points,
                deterministic=True,
                seed=SEED + tester.rank,
                force_single_arm=force_single_arm,
            )

            T = int(built["robot_flows"].shape[0])

            phase_frames = np.asarray(
                meta["phase"],
                dtype=np.float64,
            ).reshape(-1)

            if len(phase_frames) != T:
                raise RuntimeError(
                    f"{key}: phase len={len(phase_frames)}, T={T}"
                )

            base_phase = np.linspace(
                0.0,
                1.0,
                T,
                dtype=np.float64,
            )

            corrected_phase = (
                phase_frames / float(T - 1)
            )

            if not np.isclose(corrected_phase[0], 0.0):
                raise RuntimeError(
                    f"{key}: corrected phase start != 0: "
                    f"{corrected_phase[0]}"
                )

            if not np.isclose(corrected_phase[-1], 1.0):
                raise RuntimeError(
                    f"{key}: corrected phase end != 1: "
                    f"{corrected_phase[-1]}"
                )

            base_unreleased = old.src.apply_phase(
                copy.deepcopy(built),
                base_phase,
            )

            corr_unreleased = old.src.apply_phase(
                copy.deepcopy(built),
                corrected_phase,
            )

            variants = [
                ("base", copy.deepcopy(base_unreleased)),
                ("base_repeat", copy.deepcopy(base_unreleased)),
                ("corrected", copy.deepcopy(corr_unreleased)),
            ]

            released = []

            for name, sample in variants:
                r = old.release_process(
                    sample,
                    args,
                    domain,
                    tester.rank,
                )

                if r is None:
                    raise RuntimeError(
                        f"{key}: release_process returned None "
                        f"for variant={name}"
                    )

                released.append((name, r))

            batch_cpu = old.custom_collate_fn(
                [x[1] for x in released],
                args=args,
            )

            required_fields = [
                "scene_flows",
                "scene_exists",
                "scene_context_mask",
                "scene_supervised_mask",
                "scene_object_id",
            ]

            for field in required_fields:
                if field not in batch_cpu:
                    raise RuntimeError(
                        f"{key}: batch missing {field}; "
                        f"keys={list(batch_cpu.keys())}"
                    )

            # Critical audit:
            # scene/object support must be identical across
            # base, base_repeat, corrected.
            for field in required_fields:
                x = batch_cpu[field]

                if not isinstance(x, torch.Tensor):
                    continue

                if x.shape[0] != 3:
                    raise RuntimeError(
                        f"{key}: {field} first dimension "
                        f"expected 3 variants, got {tuple(x.shape)}"
                    )

                if not torch.equal(x[0], x[1]):
                    raise RuntimeError(
                        f"{key}: {field} differs between "
                        f"base and base_repeat"
                    )

                if not torch.equal(x[0], x[2]):
                    raise RuntimeError(
                        f"{key}: {field} differs between "
                        f"base and corrected"
                    )

            batch_gpu = old.to_device(
                batch_cpu,
                tester.device,
            )

            outputs = tester.model(
                batch_gpu,
                training=False,
            )

            if "scene_flows" not in outputs:
                raise RuntimeError(
                    f"{key}: model output missing scene_flows"
                )

            pred = (
                outputs["scene_flows"]
                .detach()
                .cpu()
                .numpy()
            )

            # Expected:
            # [base, base_repeat, corrected, T, Ns, 3]
            if pred.ndim != 4 or pred.shape[0] != 3:
                raise RuntimeError(
                    f"{key}: unexpected prediction shape {pred.shape}"
                )

            labels = (
                scalar3(
                    batch_cpu,
                    "scene_object_id",
                )
                .detach()
                .cpu()
                .numpy()
            )

            exists = (
                mask3(
                    batch_cpu,
                    "scene_exists",
                )
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )

            context = (
                mask3(
                    batch_cpu,
                    "scene_context_mask",
                )
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )

            supervised = (
                mask3(
                    batch_cpu,
                    "scene_supervised_mask",
                )
                .detach()
                .cpu()
                .numpy()
                .astype(bool)
            )

            valid = (
                (~context)
                & exists
                & supervised
            )

            if not (
                labels.shape
                == exists.shape
                == context.shape
                == supervised.shape
                == valid.shape
            ):
                raise RuntimeError(
                    f"{key}: tensor shape audit failed | "
                    f"labels={labels.shape}, "
                    f"exists={exists.shape}, "
                    f"context={context.shape}, "
                    f"supervised={supervised.shape}, "
                    f"valid={valid.shape}"
                )

            t_goal = pred.shape[1] - 1

            clip_rows = []

            # EXACT object IDs already accepted by the final PA evaluator.
            for obj_meta in eligible_by_clip[key]:
                oid = int(obj_meta["object_id"])

                object_mask = (
                    labels[0] == oid
                )

                # Keep the same support semantics used in the current
                # boundary PA evaluator.
                m = (
                    object_mask[t_goal]
                    & valid[0, t_goal]
                    & object_mask[0]
                    & exists[0, 0]
                )

                n = int(m.sum())

                if n < MIN_POINTS:
                    print(
                        f"[WARN] {key} oid={oid}: "
                        f"final goal support {n} < {MIN_POINTS}; skip"
                    )
                    continue

                P_base = pred[0, t_goal, m, :]
                P_repeat = pred[1, t_goal, m, :]
                P_corr = pred[2, t_goal, m, :]

                corr_m = alignment_metrics(
                    reference=P_base,
                    comparison=P_corr,
                )

                repeat_m = alignment_metrics(
                    reference=P_base,
                    comparison=P_repeat,
                )

                row = {
                    "task": task,
                    "clip": key,
                    "object_id": oid,
                    "object_full_name": obj_meta[
                        "object_full_name"
                    ],
                    "object_short_name": obj_meta[
                        "object_short_name"
                    ],
                    "frame": int(t_goal + 1),
                    "n_goal_points": n,

                    "original_acc_mps2": meta[
                        "original_acc"
                    ],
                    "corrected_acc_mps2": meta[
                        "corrected_acc"
                    ],
                    "boundary_mps2": meta[
                        "boundary"
                    ],

                    # -------------------------------------------------
                    # PRIMARY GOAL-PRESERVATION METRICS
                    # Original prediction vs Corrected prediction
                    # -------------------------------------------------
                    "goal_mpjpe_mm":
                        corr_m["raw_mpjpe_mm"],

                    "goal_centroid_shift_mm":
                        corr_m["centroid_shift_mm"],

                    # diagnostic decomposition
                    "goal_rigid_residual_mm":
                        corr_m["rigid_residual_mm"],

                    "goal_pa_mpjpe_mm":
                        corr_m["pa_mpjpe_mm"],

                    # -------------------------------------------------
                    # STOCHASTIC REPEAT CONTROL
                    # Original prediction vs Original repeat
                    # -------------------------------------------------
                    "repeat_mpjpe_mm":
                        repeat_m["raw_mpjpe_mm"],

                    "repeat_centroid_shift_mm":
                        repeat_m["centroid_shift_mm"],

                    "repeat_rigid_residual_mm":
                        repeat_m["rigid_residual_mm"],

                    "repeat_pa_mpjpe_mm":
                        repeat_m["pa_mpjpe_mm"],

                    # Difference above same-input repeat noise.
                    "goal_minus_repeat_mpjpe_mm":
                        corr_m["raw_mpjpe_mm"]
                        - repeat_m["raw_mpjpe_mm"],

                    "goal_minus_repeat_centroid_mm":
                        corr_m["centroid_shift_mm"]
                        - repeat_m["centroid_shift_mm"],
                }

                clip_rows.append(row)

            if not clip_rows:
                raise RuntimeError(
                    f"{key}: no eligible endpoint object rows produced"
                )

            rows.extend(clip_rows)
            found_target.add(key)

            print(
                f"[{task}] "
                f"{len(found_target):4d}/{len(target_clips):4d} "
                f"{key} | "
                f"objects={len(clip_rows)} | "
                f"goalMPJPE="
                f"{np.mean([r['goal_mpjpe_mm'] for r in clip_rows]):.4f} mm | "
                f"repeat="
                f"{np.mean([r['repeat_mpjpe_mm'] for r in clip_rows]):.4f} mm"
            )

            # Incremental save after every completed clip.
            cur = pd.DataFrame(rows)

            if not old_rows.empty:
                save_df = pd.concat(
                    [old_rows, cur],
                    ignore_index=True,
                )
            else:
                save_df = cur

            save_df.to_csv(
                object_csv,
                index=False,
            )

    missing = sorted(target_clips - found_target)

    if missing:
        raise RuntimeError(
            f"{task}: WDS scan ended with {len(missing)} "
            f"successful correction clips missing. "
            f"Examples={missing[:10]}"
        )

    object_df = pd.read_csv(object_csv)

    # Ensure one row/object.
    object_df = (
        object_df
        .drop_duplicates(
            subset=["task", "clip", "object_id"],
            keep="last",
        )
        .sort_values(
            ["task", "clip", "object_id"]
        )
        .reset_index(drop=True)
    )

    object_df.to_csv(
        object_csv,
        index=False,
    )

    # Clip-level object mean:
    # same aggregation philosophy as current PA analyses.
    clip_df = (
        object_df
        .groupby(
            ["task", "clip"],
            as_index=False,
        )
        .agg(
            goal_mpjpe_mm=(
                "goal_mpjpe_mm",
                "mean",
            ),
            goal_mpjpe_max_mm=(
                "goal_mpjpe_mm",
                "max",
            ),

            goal_centroid_shift_mm=(
                "goal_centroid_shift_mm",
                "mean",
            ),
            goal_centroid_shift_max_mm=(
                "goal_centroid_shift_mm",
                "max",
            ),

            goal_rigid_residual_mm=(
                "goal_rigid_residual_mm",
                "mean",
            ),
            goal_pa_mpjpe_mm=(
                "goal_pa_mpjpe_mm",
                "mean",
            ),

            repeat_mpjpe_mm=(
                "repeat_mpjpe_mm",
                "mean",
            ),
            repeat_centroid_shift_mm=(
                "repeat_centroid_shift_mm",
                "mean",
            ),

            repeat_pa_mpjpe_mm=(
                "repeat_pa_mpjpe_mm",
                "mean",
            ),

            goal_minus_repeat_mpjpe_mm=(
                "goal_minus_repeat_mpjpe_mm",
                "mean",
            ),

            n_rigid_objects=(
                "object_id",
                "nunique",
            ),
            n_goal_points=(
                "n_goal_points",
                "sum",
            ),

            original_acc_mps2=(
                "original_acc_mps2",
                "first",
            ),
            corrected_acc_mps2=(
                "corrected_acc_mps2",
                "first",
            ),
            boundary_mps2=(
                "boundary_mps2",
                "first",
            ),
        )
    )

    clip_df.to_csv(
        clip_csv,
        index=False,
    )

    if len(clip_df) != EXPECTED_SUCCESS[task]:
        raise RuntimeError(
            f"{task}: final clip count expected "
            f"{EXPECTED_SUCCESS[task]}, got {len(clip_df)}"
        )

    print()
    print("=" * 80)
    print(f"[DONE] {task}")
    print(f"[OBJECT ROWS] {len(object_df)}")
    print(f"[CLIPS      ] {len(clip_df)}")
    print()
    print(
        "[GOAL MPJPE] "
        f"median={clip_df['goal_mpjpe_mm'].median():.6f} mm | "
        f"mean={clip_df['goal_mpjpe_mm'].mean():.6f} mm | "
        f"p90={clip_df['goal_mpjpe_mm'].quantile(.90):.6f} mm | "
        f"p95={clip_df['goal_mpjpe_mm'].quantile(.95):.6f} mm"
    )
    print(
        "[CENTROID ] "
        f"median={clip_df['goal_centroid_shift_mm'].median():.6f} mm | "
        f"p90={clip_df['goal_centroid_shift_mm'].quantile(.90):.6f} mm | "
        f"p95={clip_df['goal_centroid_shift_mm'].quantile(.95):.6f} mm"
    )
    print(
        "[REPEAT   ] "
        f"median raw MPJPE="
        f"{clip_df['repeat_mpjpe_mm'].median():.6f} mm"
    )
    print(
        "[EXCESS   ] "
        f"median goal-repeat="
        f"{clip_df['goal_minus_repeat_mpjpe_mm'].median():.6f} mm"
    )
    print(f"[SAVE] {object_csv}")
    print(f"[SAVE] {clip_csv}")
    print("=" * 80)


# =====================================================================
# Combine all 5 tasks
# =====================================================================

def combine_all():
    dfs = []

    for task in TASKS:
        path = (
            OUT_ROOT
            / task
            / "per_clip_endpoint_goal_metrics.csv"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        expected = EXPECTED_SUCCESS[task]

        if len(df) != expected:
            raise RuntimeError(
                f"{task}: expected {expected}, got {len(df)}"
            )

        dfs.append(df)

    all_df = pd.concat(
        dfs,
        ignore_index=True,
    )

    if len(all_df) != EXPECTED_TOTAL_SUCCESS:
        raise RuntimeError(
            f"Combined expected {EXPECTED_TOTAL_SUCCESS}, "
            f"got {len(all_df)}"
        )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_csv = (
        OUT_ROOT
        / "combined_5task_endpoint_goal_metrics.csv"
    )

    all_df.to_csv(
        out_csv,
        index=False,
    )

    # -------------------------------------------------------------
    # Text summary
    # -------------------------------------------------------------
    summary_path = (
        OUT_ROOT / "FINAL_GOAL_ANALYSIS.txt"
    )

    def stat_line(df, col):
        x = df[col].dropna()
        return (
            f"N={len(x)} "
            f"median={x.median():.6f} "
            f"mean={x.mean():.6f} "
            f"p75={x.quantile(.75):.6f} "
            f"p90={x.quantile(.90):.6f} "
            f"p95={x.quantile(.95):.6f}"
        )

    lines = []

    lines.append(
        "POINTWORLD ACC-BOUNDARY GOAL PRESERVATION"
    )
    lines.append("=" * 72)
    lines.append(
        f"Combined successful corrected clips: {len(all_df)}"
    )
    lines.append("")

    lines.append(
        "PRIMARY: Original-prediction vs Corrected-prediction "
        "final endpoint MPJPE [mm]"
    )
    lines.append(
        stat_line(all_df, "goal_mpjpe_mm")
    )
    lines.append("")

    lines.append(
        "Final object centroid shift [mm]"
    )
    lines.append(
        stat_line(all_df, "goal_centroid_shift_mm")
    )
    lines.append("")

    lines.append(
        "Original vs Original-repeat endpoint MPJPE "
        "(stochastic baseline) [mm]"
    )
    lines.append(
        stat_line(all_df, "repeat_mpjpe_mm")
    )
    lines.append("")

    lines.append(
        "Correction endpoint MPJPE minus repeat endpoint MPJPE [mm]"
    )
    lines.append(
        stat_line(all_df, "goal_minus_repeat_mpjpe_mm")
    )
    lines.append("")

    lines.append(
        "Original vs Corrected PA-MPJPE after similarity alignment [mm]"
    )
    lines.append(
        stat_line(all_df, "goal_pa_mpjpe_mm")
    )
    lines.append("")

    lines.append("TASKWISE MEDIANS")
    lines.append("-" * 72)

    task_summary = (
        all_df
        .groupby("task")[
            [
                "goal_mpjpe_mm",
                "goal_centroid_shift_mm",
                "repeat_mpjpe_mm",
                "goal_minus_repeat_mpjpe_mm",
                "goal_pa_mpjpe_mm",
            ]
        ]
        .median()
    )

    lines.append(
        task_summary.to_string()
    )
    lines.append("")

    lines.append(
        "INTERPRETATION:"
    )
    lines.append(
        "- goal_mpjpe_mm: how far the corrected-action predicted "
        "final rigid-object points move away from the original-action "
        "predicted final object goal."
    )
    lines.append(
        "- goal_centroid_shift_mm: translation of the predicted final "
        "object center."
    )
    lines.append(
        "- repeat_mpjpe_mm: same-input stochastic inference baseline."
    )
    lines.append(
        "- goal_pa_mpjpe_mm: residual shape difference after removing "
        "translation/rotation/uniform-scale."
    )
    lines.append(
        "- This is predicted-goal preservation, NOT simulator GT "
        "task-success evaluation."
    )

    summary_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))
    print()
    print(f"[SAVE] {out_csv}")
    print(f"[SAVE] {summary_path}")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices=TASKS,
        default=None,
    )

    parser.add_argument(
        "--combine",
        action="store_true",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
    )

    args = parser.parse_args()

    if args.combine:
        combine_all()
        return

    if args.task is None:
        raise SystemExit(
            "Use --task task-XXXX or --combine"
        )

    run_task(
        args.task,
        resume=(not args.no_resume),
    )


if __name__ == "__main__":
    main()
