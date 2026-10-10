from pathlib import Path
from collections import Counter
import argparse
import random
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from arguments import parse_args
from evaluation.tester import Tester

import diagnostics.behavior_object_mapping_smoke_task0006 as omap
from diagnostics.behavior_object_mpjpe_pa_smoke_task0006 import (
    p_mpjpe_single,
    bool_mask,
    MODEL_PATH,
    NORM_STATS,
)


TASK_DIRS = {
    "task-0000": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0000",
    "task-0001": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0001",
    "task-0006": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0006",
    "task-0008": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0008",
    "task-0010": "/home/soenchun/cse/PointWorld_workspace/datasets/behavior/wds_task0010",
}

EXPECTED_TARGETS = {
    "task-0000": 612,
    "task-0001": 2477,
    "task-0006": 2157,
    "task-0008": 2972,
    "task-0010": 2239,
}

ACC_CSV = (
    "diagnostics/results/current_behavior_established_acc_10457/"
    "per_sample_current_established_acc_10457.csv"
)

OUT_ROOT = Path(
    "diagnostics/results/rigid_projection_repair_current10457"
)

MIN_POINTS = 8
EVAL_SEED = 42

# Representative qualitative examples per task.
# Metric computation still uses ALL eligible points.
VIS_LEVELS = [
    ("LOW", 0.10),
    ("MID", 0.50),
    ("HIGH", 0.90),
]

# Complete graph is drawn among displayed points.
# If an object has more points, deterministic farthest-point
# subsampling is used ONLY for visualization.
VIS_MAX_POINTS = 30


def parse_local_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tasks",
        required=True,
        type=str,
        help=(
            "Comma-separated tasks. Example: "
            "task-0006 or task-0006,task-0008"
        ),
    )

    return parser.parse_args()


def short_object_name(full_name):
    parts = str(full_name).split("__")

    for i, p in enumerate(parts):
        if p.startswith("scene_") and i + 1 < len(parts):
            return parts[i + 1]

    return str(full_name)


def reset_eval_seed(seed=EVAL_SEED):
    """
    Reset once at the beginning of EACH task.

    We do NOT reset per sample.
    Each clip still receives exactly one PointWorld forward.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_behavior_data_dir(tester, data_dir):
    domains = [str(x) for x in tester.args.domains]

    if "behavior" not in domains:
        raise RuntimeError(
            f"behavior not found in tester.args.domains: {domains}"
        )

    idx = domains.index("behavior")
    tester.args.data_dirs[idx] = data_dir

    if hasattr(tester.args, "eval_domain_whitelist"):
        tester.args.eval_domain_whitelist = ["behavior"]


def write_csv(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )




def rigid_project_initial_to_prediction(
    initial_shape,
    predicted_shape,
):
    """
    Project PointWorld prediction onto the nearest rigid transform
    of the INITIAL observed object shape.

    Uses rotation + translation only.
    NO scale.
    NO future GT.

    initial_shape:   (N,3)
    predicted_shape: (N,3)
    """

    x = np.asarray(
        initial_shape,
        dtype=np.float64,
    )

    y = np.asarray(
        predicted_shape,
        dtype=np.float64,
    )

    if x.shape != y.shape:
        raise ValueError(
            f"shape mismatch: {x.shape} vs {y.shape}"
        )

    if len(x) < MIN_POINTS:
        raise ValueError("too few points")

    mx = x.mean(
        axis=0,
        keepdims=True,
    )

    my = y.mean(
        axis=0,
        keepdims=True,
    )

    xc = x - mx
    yc = y - my

    H = xc.T @ yc

    U, _, Vt = np.linalg.svd(H)

    R = U @ Vt

    # Proper rotation only. No reflection.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = U @ Vt

    repaired = (
        xc @ R
        + my
    )

    return repaired



def choose_visualization_clips(acc_df, selected_tasks):
    """
    Choose representative low / middle / high Action RMS-Acceleration
    clips BEFORE inference, so visualization uses the same single
    inference already required for the quantitative experiment.
    """
    out = {}

    for task in selected_tasks:
        d = (
            acc_df[
                acc_df["task"] == task
            ]
            .copy()
            .reset_index(drop=True)
        )

        vals = d[
            "rms_acceleration_si_m_s2"
        ].to_numpy(float)

        task_map = {}

        for level, q in VIS_LEVELS:
            target_value = float(
                np.quantile(vals, q)
            )

            idx = int(
                np.argmin(
                    np.abs(
                        vals - target_value
                    )
                )
            )

            clip = str(
                d.iloc[idx]["clip"]
            )

            actual_acc = float(
                d.iloc[idx][
                    "rms_acceleration_si_m_s2"
                ]
            )

            task_map[clip] = {
                "level": level,
                "quantile": q,
                "target_acc": target_value,
                "actual_acc": actual_acc,
            }

        out[task] = task_map

    return out


def safe_filename(text):
    text = str(text)

    keep = []

    for ch in text:
        if (
            ch.isalnum()
            or ch in "-_"
        ):
            keep.append(ch)
        else:
            keep.append("-")

    result = "".join(keep)

    while "--" in result:
        result = result.replace(
            "--",
            "-",
        )

    return result.strip("-")


def farthest_point_display_indices(points, max_points):
    """
    Visualization-only deterministic farthest-point sampling.
    Keeps the displayed points spread over the object's geometry.
    """
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    n = len(points)

    if n <= max_points:
        return np.arange(
            n,
            dtype=np.int64,
        )

    centroid = points.mean(
        axis=0,
        keepdims=True,
    )

    first = int(
        np.argmax(
            np.linalg.norm(
                points - centroid,
                axis=1,
            )
        )
    )

    selected = [first]

    min_dist = np.linalg.norm(
        points - points[first],
        axis=1,
    )

    while len(selected) < max_points:
        nxt = int(
            np.argmax(
                min_dist
            )
        )

        selected.append(nxt)

        d = np.linalg.norm(
            points - points[nxt],
            axis=1,
        )

        min_dist = np.minimum(
            min_dist,
            d,
        )

        min_dist[
            np.asarray(
                selected,
                dtype=np.int64,
            )
        ] = -1.0

    return np.asarray(
        selected,
        dtype=np.int64,
    )


def procrustes_steps(predicted, target):
    """
    EXACT same convention as p_mpjpe_single().

    target    = X
    predicted = Y

    Fits:
        aligned = scale * (predicted @ R) + translation

    with one uniform scale + proper rotation + translation.
    """

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    # ------------------------------------------------------------
    # EXACT COPY OF p_mpjpe_single() ALIGNMENT CONVENTION
    # ------------------------------------------------------------
    muX = np.mean(
        target,
        axis=0,
        keepdims=True,
    )

    muY = np.mean(
        predicted,
        axis=0,
        keepdims=True,
    )

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(
        np.sum(X0 ** 2)
    )

    normY = np.sqrt(
        np.sum(Y0 ** 2)
    )

    if normX <= 0 or normY <= 0:
        raise RuntimeError(
            "Degenerate point geometry for PA-MPJPE"
        )

    X0 = X0 / normX
    Y0 = Y0 / normY

    H = X0.T @ Y0

    U, singular_values, Vt = (
        np.linalg.svd(H)
    )

    V = Vt.T
    R = V @ U.T

    sign_detR = np.sign(
        np.linalg.det(R)
    )

    if sign_detR == 0:
        sign_detR = 1.0

    V[:, -1] *= sign_detR
    singular_values[-1] *= sign_detR

    R = V @ U.T

    tr = np.sum(
        singular_values
    )

    scale = (
        tr
        * normX
        / normY
    )

    translation = (
        muX
        - scale
        * (muY @ R)
    )

    aligned = (
        scale
        * (predicted @ R)
        + translation
    )

    # ------------------------------------------------------------
    # Visualization decomposition.
    #
    # These three explanatory stages compose EXACTLY to `aligned`.
    #
    # 1. scale prediction around its own centroid
    # 2. translate its centroid onto GT centroid
    # 3. rotate around GT centroid
    # ------------------------------------------------------------
    scaled = (
        scale
        * (predicted - muY)
        + muY
    )

    shifted = (
        scaled
        - muY
        + muX
    )

    rotated = (
        (shifted - muX)
        @ R
        + muX
    )

    decomposition_error = float(
        np.max(
            np.abs(
                rotated - aligned
            )
        )
    )

    return {
        "scale":
            float(scale),

        "rotation":
            R,

        "translation":
            translation,

        "scaled":
            scaled,

        "shifted":
            shifted,

        "aligned":
            aligned,

        "decomposition_error":
            decomposition_error,
    }


def draw_complete_structure(
    ax,
    gt,
    pred,
    title,
):
    """
    GT points are red and connected to every other displayed GT point.
    Pred points are blue and connected to every other displayed Pred point.
    Same-index GT<->Pred correspondence is gray dashed.
    """
    n = len(gt)

    # Every displayed point connected to every other
    # point of the SAME color.
    for i in range(n):
        for j in range(i + 1, n):

            ax.plot(
                [
                    gt[i, 0],
                    gt[j, 0],
                ],
                [
                    gt[i, 1],
                    gt[j, 1],
                ],
                [
                    gt[i, 2],
                    gt[j, 2],
                ],
                c="red",
                linewidth=0.65,
                alpha=0.22,
            )

            ax.plot(
                [
                    pred[i, 0],
                    pred[j, 0],
                ],
                [
                    pred[i, 1],
                    pred[j, 1],
                ],
                [
                    pred[i, 2],
                    pred[j, 2],
                ],
                c="blue",
                linewidth=0.65,
                alpha=0.22,
            )

    # Corresponding physical point.
    for i in range(n):
        ax.plot(
            [
                gt[i, 0],
                pred[i, 0],
            ],
            [
                gt[i, 1],
                pred[i, 1],
            ],
            [
                gt[i, 2],
                pred[i, 2],
            ],
            c="gray",
            linewidth=0.7,
            alpha=0.28,
            linestyle="--",
        )

    ax.scatter(
        gt[:, 0],
        gt[:, 1],
        gt[:, 2],
        c="red",
        s=30,
        label="GT",
        depthshade=False,
    )

    ax.scatter(
        pred[:, 0],
        pred[:, 1],
        pred[:, 2],
        c="blue",
        s=30,
        label="PointWorld Pred",
        depthshade=False,
    )

    ax.set_title(
        title,
        fontsize=10,
    )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")


def save_procrustes_visualization(
    output_dir,
    task,
    clip,
    level,
    object_full_name,
    object_short_name,
    frame,
    pred_t,
    gt_t,
    rms_acc,
    mpjpe_mm,
    pa_mpjpe_mm,
    fitted_scale_metric,
):
    """
    Save one 4-panel visualization using the SAME prediction
    already produced for quantitative evaluation.
    No additional model inference occurs here.
    """
    pred_t = np.asarray(
        pred_t,
        dtype=np.float64,
    )

    gt_t = np.asarray(
        gt_t,
        dtype=np.float64,
    )

    proc = procrustes_steps(
        pred_t,
        gt_t,
    )

    aligned = proc["aligned"]

    pa_check_mm = float(
        np.mean(
            np.linalg.norm(
                aligned - gt_t,
                axis=-1,
            )
        )
        * 1000.0
    )

    # Strong audit: visualization alignment must reproduce
    # the recorded PA-MPJPE.
    if not np.isclose(
        pa_check_mm,
        pa_mpjpe_mm,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Visualization PA mismatch: "
            f"metric={pa_mpjpe_mm:.9f} mm "
            f"visual={pa_check_mm:.9f} mm"
        )

    if not np.isclose(
        proc["scale"],
        fitted_scale_metric,
        rtol=1e-5,
        atol=1e-7,
    ):
        raise RuntimeError(
            "Visualization scale mismatch: "
            f"metric={fitted_scale_metric:.9f} "
            f"visual={proc['scale']:.9f}"
        )

    # Visualization-only subset.
    display_idx = (
        farthest_point_display_indices(
            gt_t,
            VIS_MAX_POINTS,
        )
    )

    gt_display = gt_t[
        display_idx
    ]

    pred_display = pred_t[
        display_idx
    ]

    scaled_display = proc[
        "scaled"
    ][display_idx]

    shifted_display = proc[
        "shifted"
    ][display_idx]

    aligned_display = proc[
        "aligned"
    ][display_idx]

    stages = [
        (
            pred_display,
            "1. Before alignment",
        ),
        (
            scaled_display,
            "2. Scale",
        ),
        (
            shifted_display,
            "3. Translation",
        ),
        (
            aligned_display,
            "4. Rotation / Final PA",
        ),
    ]

    # Common axes across all four panels.
    all_points = np.concatenate(
        [
            gt_display,
            pred_display,
            scaled_display,
            shifted_display,
            aligned_display,
        ],
        axis=0,
    )

    mins = all_points.min(
        axis=0
    )

    maxs = all_points.max(
        axis=0
    )

    center = (
        mins + maxs
    ) / 2.0

    span = float(
        np.max(
            maxs - mins
        )
    )

    if span <= 0:
        span = 1e-3

    half = (
        span * 0.58
    )

    fig = plt.figure(
        figsize=(18, 5.7)
    )

    axes = []

    for i, (
        pred_stage,
        stage_title,
    ) in enumerate(
        stages,
        start=1,
    ):

        ax = fig.add_subplot(
            1,
            4,
            i,
            projection="3d",
        )

        axes.append(ax)

        draw_complete_structure(
            ax,
            gt_display,
            pred_stage,
            stage_title,
        )

        ax.set_xlim(
            center[0] - half,
            center[0] + half,
        )

        ax.set_ylim(
            center[1] - half,
            center[1] + half,
        )

        ax.set_zlim(
            center[2] - half,
            center[2] + half,
        )

        try:
            ax.set_box_aspect(
                (1, 1, 1)
            )
        except Exception:
            pass

    axes[0].legend(
        loc="upper left",
        fontsize=8,
    )

    main_title = (
        f"{task} | {level} Acc | "
        f"{object_short_name} | frame {frame}\n"
        f"N(metric)={len(gt_t)}, "
        f"N(display)={len(display_idx)} | "
        f"RMS Acc={rms_acc:.6f} m/s² | "
        f"MPJPE={mpjpe_mm:.3f} mm | "
        f"PA-MPJPE={pa_mpjpe_mm:.3f} mm | "
        f"s={proc['scale']:.4f}"
    )

    fig.suptitle(
        main_title,
        fontsize=12,
        y=0.98,
    )

    wrapped_object = textwrap.fill(
        object_full_name,
        width=140,
    )

    fig.text(
        0.5,
        0.015,
        "Rigid mesh: "
        + wrapped_object,
        ha="center",
        va="bottom",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[
            0.0,
            0.055,
            1.0,
            0.91,
        ]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{level}__"
        f"{safe_filename(clip)}__"
        f"{safe_filename(object_short_name)}__"
        f"frame{frame:02d}__procrustes.png"
    )

    path = (
        output_dir
        / filename
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return {
        "path": str(path),
        "display_point_count":
            int(len(display_idx)),
        "metric_point_count":
            int(len(gt_t)),
        "pa_check_mm":
            pa_check_mm,
        "visualization_scale":
            float(proc["scale"]),
        "decomposition_error":
            float(
                proc[
                    "decomposition_error"
                ]
            ),
    }


def main():
    cli = parse_local_args()

    selected_tasks = [
        x.strip()
        for x in cli.tasks.split(",")
        if x.strip()
    ]

    if not selected_tasks:
        raise RuntimeError("No task selected.")

    invalid = [
        t
        for t in selected_tasks
        if t not in TASK_DIRS
    ]

    if invalid:
        raise RuntimeError(
            f"Unknown task(s): {invalid}\n"
            f"Allowed: {list(TASK_DIRS)}"
        )

    if len(selected_tasks) != len(set(selected_tasks)):
        raise RuntimeError(
            f"Duplicate task supplied: {selected_tasks}"
        )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =============================================================
    # Established Action RMS Acceleration
    # =============================================================
    acc = pd.read_csv(ACC_CSV)

    assert len(acc) == 10457
    assert acc["clip"].duplicated().sum() == 0

    counts = (
        acc.groupby("task")
        .size()
        .to_dict()
    )

    assert counts == EXPECTED_TARGETS

    # SI audit: BEHAVIOR dt = 0.2 sec => Acc x25.
    si_err = np.max(
        np.abs(
            acc["rms_acceleration_si_m_s2"].to_numpy(float)
            - acc["rms_acceleration"].to_numpy(float) * 25.0
        )
    )

    assert si_err < 1e-12

    acc_lookup = {
        str(r.clip):
        float(r.rms_acceleration_si_m_s2)
        for r in acc.itertuples()
    }

    targets_by_task = {
        task: set(
            acc.loc[
                acc["task"] == task,
                "clip",
            ].astype(str)
        )
        for task in selected_tasks
    }

    # =============================================================
    # Representative visualization clips are fixed BEFORE inference.
    # No result-based cherry-picking.
    # =============================================================
    vis_clips_by_task = choose_visualization_clips(
        acc,
        selected_tasks,
    )

    print()
    print("===== FIXED VISUALIZATION CLIPS =====")

    for task in selected_tasks:
        print(task)

        for clip, meta in vis_clips_by_task[task].items():
            print(
                " ",
                meta["level"],
                "clip=",
                clip,
                "RMS_Acc=",
                f'{meta["actual_acc"]:.6f}',
                "m/s^2",
            )

    # =============================================================
    # Runtime-only rigid-object identity propagation.
    #
    # These hooks carry mesh ID in parallel with the official points.
    # They do NOT modify PointWorld scene coordinates or GT.
    # =============================================================
    omap.dlmod.build_flow_sample = (
        omap.build_flow_sample_with_object_ids
    )

    omap.pmod.grid_sample_transform = (
        omap.grid_sample_with_object_ids
    )

    omap.pmod.gather = (
        omap.gather_with_object_ids
    )

    try:
        # ---------------------------------------------------------
        # Instantiate PointWorld ONCE for this process.
        # ---------------------------------------------------------
        first_task = selected_tasks[0]

        args = parse_args(
            skip_command_line=True
        )

        args.model_path = MODEL_PATH
        args.domains = ["behavior"]
        args.data_dirs = [
            TASK_DIRS[first_task]
        ]
        args.norm_stats_path = NORM_STATS

        args.batch_size = 1
        args.num_workers = 0
        args.eval_num_workers = 0
        args.eval_num_batches = -1
        args.device = "cuda"
        args.eval_skip_viz = True

        tester = Tester(args)
        tester.model.eval()

        print()
        print("=" * 78)
        print("BEHAVIOR RIGID-OBJECT MPJPE / PA-MPJPE")
        print("SELECTED TASKS =", selected_tasks)
        print("MIN POINTS     =", MIN_POINTS)
        print("EVAL SEED      =", EVAL_SEED)
        print("FORWARD        = exactly 1 per target clip")
        print("=" * 78)

        # =========================================================
        # TASK LOOP
        # =========================================================
        for task in selected_tasks:

            data_dir = TASK_DIRS[task]
            task_targets = targets_by_task[task]

            if (
                len(task_targets)
                != EXPECTED_TARGETS[task]
            ):
                raise RuntimeError(
                    f"{task}: target count mismatch: "
                    f"{len(task_targets)} vs "
                    f"{EXPECTED_TARGETS[task]}"
                )

            task_out = (
                OUT_ROOT
                / task
            )

            task_out.mkdir(
                parents=True,
                exist_ok=True,
            )

            # IMPORTANT:
            # Each task is treated as an independent official-style
            # evaluation run. Reset ONCE here, never per sample.
            reset_eval_seed(EVAL_SEED)

            print()
            print("=" * 78)
            print("TASK          =", task)
            print("DATA          =", data_dir)
            print(
                "TARGET CLIPS  =",
                len(task_targets),
            )
            print(
                "SEED RESET    =",
                EVAL_SEED,
                "(once for this task)",
            )
            print("=" * 78)

            set_behavior_data_dir(
                tester,
                data_dir,
            )

            loader, _ = (
                tester._build_eval_loader(
                    "test",
                    enable_mask=False,
                )
            )

            # Task-specific accumulators.
            frame_rows = []
            object_rows = []
            clip_rows = []
            status_rows = []
            visualization_rows = []

            vis_selection = (
                vis_clips_by_task[task]
            )

            seen_targets = set()
            object_counter = Counter()

            # =====================================================
            # SAMPLE LOOP
            # =====================================================
            for loader_idx, batch_cpu in enumerate(loader):

                key = omap.normalize_key(
                    batch_cpu["__key__"]
                )

                # Evaluate the current official 10,457 test population.
                if key not in task_targets:
                    continue

                if key in seen_targets:
                    raise RuntimeError(
                        f"Duplicate target clip encountered: {key}"
                    )

                seen_targets.add(key)

                batch = {
                    k: (
                        v.to(
                            tester.device,
                            non_blocking=True,
                        )
                        if isinstance(
                            v,
                            torch.Tensor,
                        )
                        else v
                    )
                    for k, v
                    in batch_cpu.items()
                }

                # =================================================
                # EXACTLY ONE PointWorld inference for this clip.
                # =================================================
                with torch.no_grad():
                    outputs = tester.model(
                        batch,
                        training=False,
                    )

                # -------------------------------------------------
                # Object identity
                # -------------------------------------------------
                labels = batch[
                    "scene_object_id"
                ]

                if isinstance(
                    labels,
                    torch.Tensor,
                ):
                    labels_np = (
                        labels
                        .detach()
                        .cpu()
                        .numpy()[0]
                    )
                else:
                    labels_np = (
                        np.asarray(labels)[0]
                    )

                if key not in omap.OBJECT_TABLE:
                    raise RuntimeError(
                        f"No object table for {key}"
                    )

                table = (
                    omap.OBJECT_TABLE[key]
                )

                # -------------------------------------------------
                # Official PointWorld masks
                # -------------------------------------------------
                context = bool_mask(
                    batch[
                        "scene_context_mask"
                    ]
                )

                exists = bool_mask(
                    batch[
                        "scene_exists"
                    ]
                )

                supervised = bool_mask(
                    batch[
                        "scene_supervised_mask"
                    ]
                )

                moved = bool_mask(
                    batch[
                        "scene_moved_mask"
                    ]
                )

                official_valid = (
                    (~context)
                    & exists
                    & supervised
                )

                official_valid_np = (
                    official_valid[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(bool)
                )

                moved_np = (
                    moved[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(bool)
                )

                # -------------------------------------------------
                # PointWorld prediction and same-time GT
                # -------------------------------------------------
                pred = (
                    outputs[
                        "scene_flows"
                    ][0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                gt = (
                    batch[
                        "gt_scene_flows"
                    ][0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                # Initial-frame existence mask used ONLY for
                # correspondence-preserving rigid repair.
                exists_np_for_repair = (
                    exists[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(bool)
                )

                eligible_objects_this_clip = []

                # =================================================
                # RIGID OBJECT LOOP
                # =================================================
                for oid, mesh_name in table.items():

                    oid = int(oid)

                    object_mask_np = (
                        labels_np == oid
                    )

                    # ---------------------------------------------
                    # Identify MOVING rigid meshes.
                    #
                    # IMPORTANT:
                    # scene_moved_mask is used only for deciding
                    # whether the rigid mesh moved.
                    #
                    # MPJPE/PA-MPJPE below uses ALL valid points
                    # of that selected rigid object.
                    # ---------------------------------------------
                    moved_support = int(
                        (
                            object_mask_np
                            & moved_np
                            & official_valid_np
                        ).sum()
                    )

                    if moved_support <= 0:
                        continue

                    full_name = str(
                        mesh_name
                    )

                    short_name = (
                        short_object_name(
                            full_name
                        )
                    )

                    # Number of retained object points
                    # after official PointWorld sampling.
                    final_object_points = int(
                        object_mask_np[0].sum()
                    )

                    obj_frame_rows = []

                    # =============================================
                    # FUTURE FRAME LOOP
                    # =============================================
                    for t in range(
                        gt.shape[0]
                    ):

                        m = (
                            object_mask_np[t]
                            & official_valid_np[t]
                        )

                        n = int(
                            m.sum()
                        )

                        if n < MIN_POINTS:
                            continue

                        pred_t = pred[
                            t,
                            m,
                        ]

                        gt_t = gt[
                            t,
                            m,
                        ]

                        centered_gt = (
                            gt_t
                            - gt_t.mean(
                                axis=0,
                                keepdims=True,
                            )
                        )

                        gt_rank = int(
                            np.linalg.matrix_rank(
                                centered_gt,
                                tol=1e-8,
                            )
                        )

                        # At least non-collinear geometry.
                        if gt_rank < 2:
                            continue

                        # =========================================
                        # MPJPE
                        # Pred(t) vs same-time GT(t)
                        # =========================================
                        mpjpe_m = float(
                            np.mean(
                                np.linalg.norm(
                                    pred_t
                                    - gt_t,
                                    axis=-1,
                                )
                            )
                        )

                        # =========================================
                        # PA-MPJPE
                        # 3D similarity Procrustes:
                        # scale + proper rotation + translation
                        # =========================================
                        try:
                            (
                                pa_m,
                                fitted_scale,
                            ) = p_mpjpe_single(
                                pred_t,
                                gt_t,
                            )

                        except Exception as exc:
                            print(
                                "[PA FAIL]",
                                task,
                                key,
                                full_name,
                                f"t={t}",
                                repr(exc),
                            )
                            continue

                        # =========================================
                        # RIGID PROJECTION REPAIR
                        #
                        # Same physical point indices must exist
                        # at t=0 and t.
                        #
                        # Repair itself uses:
                        #   initial observed shape gt[0]
                        #   PointWorld prediction pred[t]
                        #
                        # It does NOT use future gt[t].
                        # =========================================

                        repair_mask = (
                            object_mask_np[t]
                            & official_valid_np[t]
                            & object_mask_np[0]
                            & exists_np_for_repair[0]
                        )

                        repair_point_count = int(
                            repair_mask.sum()
                        )

                        repair_raw_mpjpe_mm = np.nan
                        rigid_repair_mpjpe_mm = np.nan
                        rigid_residual_mm = np.nan
                        repair_delta_mpjpe_mm = np.nan
                        repair_relative_change_pct = np.nan

                        if (
                            repair_point_count
                            >= MIN_POINTS
                        ):
                            x0 = gt[
                                0,
                                repair_mask,
                            ]

                            pred_r = pred[
                                t,
                                repair_mask,
                            ]

                            gt_r = gt[
                                t,
                                repair_mask,
                            ]

                            x0_centered = (
                                x0
                                - x0.mean(
                                    axis=0,
                                    keepdims=True,
                                )
                            )

                            if (
                                np.linalg.matrix_rank(
                                    x0_centered,
                                    tol=1e-8,
                                )
                                >= 2
                            ):
                                repaired_r = (
                                    rigid_project_initial_to_prediction(
                                        x0,
                                        pred_r,
                                    )
                                )

                                raw_m = float(
                                    np.mean(
                                        np.linalg.norm(
                                            pred_r
                                            - gt_r,
                                            axis=-1,
                                        )
                                    )
                                )

                                repaired_m = float(
                                    np.mean(
                                        np.linalg.norm(
                                            repaired_r
                                            - gt_r,
                                            axis=-1,
                                        )
                                    )
                                )

                                residual_m = float(
                                    np.mean(
                                        np.linalg.norm(
                                            pred_r
                                            - repaired_r,
                                            axis=-1,
                                        )
                                    )
                                )

                                repair_raw_mpjpe_mm = (
                                    raw_m * 1000.0
                                )

                                rigid_repair_mpjpe_mm = (
                                    repaired_m * 1000.0
                                )

                                rigid_residual_mm = (
                                    residual_m * 1000.0
                                )

                                repair_delta_mpjpe_mm = (
                                    rigid_repair_mpjpe_mm
                                    - repair_raw_mpjpe_mm
                                )

                                if (
                                    repair_raw_mpjpe_mm
                                    > 1e-12
                                ):
                                    repair_relative_change_pct = (
                                        100.0
                                        * repair_delta_mpjpe_mm
                                        / repair_raw_mpjpe_mm
                                    )


                        row = {
                            "task":
                                task,

                            "clip":
                                key,

                            "object_id":
                                oid,

                            "object_full_name":
                                full_name,

                            "object_short_name":
                                short_name,

                            "frame":
                                int(t),

                            "valid_point_count":
                                n,

                            "gt_geometry_rank":
                                gt_rank,

                            "official_moved_support_pointframes":
                                moved_support,

                            "rms_acceleration_si_m_s2":
                                acc_lookup[key],

                            "mpjpe_mm":
                                mpjpe_m
                                * 1000.0,

                            "pa_mpjpe_mm":
                                float(pa_m)
                                * 1000.0,

                            "fitted_scale":
                                float(
                                    fitted_scale
                                ),

                            "repair_point_count":
                                repair_point_count,

                            "repair_raw_mpjpe_mm":
                                repair_raw_mpjpe_mm,

                            "rigid_repair_mpjpe_mm":
                                rigid_repair_mpjpe_mm,

                            "rigid_residual_mm":
                                rigid_residual_mm,

                            "repair_delta_mpjpe_mm":
                                repair_delta_mpjpe_mm,

                            "repair_relative_change_pct":
                                repair_relative_change_pct,
                        }

                        frame_rows.append(
                            row
                        )

                        obj_frame_rows.append(
                            row
                        )

                    # No eligible PA frame for this mesh.
                    if not obj_frame_rows:
                        continue

                    object_counter[
                        short_name
                    ] += 1

                    pa_vals = np.asarray(
                        [
                            r[
                                "pa_mpjpe_mm"
                            ]
                            for r
                            in obj_frame_rows
                        ],
                        dtype=float,
                    )

                    mp_vals = np.asarray(
                        [
                            r[
                                "mpjpe_mm"
                            ]
                            for r
                            in obj_frame_rows
                        ],
                        dtype=float,
                    )

                    scales = np.asarray(
                        [
                            r[
                                "fitted_scale"
                            ]
                            for r
                            in obj_frame_rows
                        ],
                        dtype=float,
                    )

                    object_row = {
                        "task":
                            task,

                        "clip":
                            key,

                        "object_id":
                            oid,

                        "object_full_name":
                            full_name,

                        "object_short_name":
                            short_name,

                        "final_object_point_count":
                            final_object_points,

                        "official_moved_support_pointframes":
                            moved_support,

                        "eligible_future_frames":
                            len(
                                obj_frame_rows
                            ),

                        "rms_acceleration_si_m_s2":
                            acc_lookup[key],

                        "mpjpe_mean_mm":
                            float(
                                np.mean(
                                    mp_vals
                                )
                            ),

                        "mpjpe_median_mm":
                            float(
                                np.median(
                                    mp_vals
                                )
                            ),

                        "pa_mpjpe_mean_mm":
                            float(
                                np.mean(
                                    pa_vals
                                )
                            ),

                        "pa_mpjpe_median_mm":
                            float(
                                np.median(
                                    pa_vals
                                )
                            ),

                        "fitted_scale_mean":
                            float(
                                np.mean(
                                    scales
                                )
                            ),

                        "fitted_scale_final":
                            float(
                                scales[-1]
                            ),
                    }

                    object_rows.append(
                        object_row
                    )

                    eligible_objects_this_clip.append(
                        object_row
                    )

                    # =============================================
                    # QUALITATIVE VISUALIZATION
                    #
                    # Uses the SAME single inference above.
                    # Only fixed LOW/MID/HIGH Acc clips are saved.
                    # =============================================
                    if key in vis_selection:

                        vis_meta = (
                            vis_selection[key]
                        )

                        # Use latest eligible future frame.
                        vis_metric = (
                            obj_frame_rows[-1]
                        )

                        vis_t = int(
                            vis_metric["frame"]
                        )

                        vis_mask = (
                            object_mask_np[vis_t]
                            & official_valid_np[vis_t]
                        )

                        vis_pred = pred[
                            vis_t,
                            vis_mask,
                        ]

                        vis_gt = gt[
                            vis_t,
                            vis_mask,
                        ]

                        vis_result = (
                            save_procrustes_visualization(
                                output_dir=(
                                    task_out
                                    / "visualizations"
                                ),
                                task=task,
                                clip=key,
                                level=vis_meta[
                                    "level"
                                ],
                                object_full_name=
                                    full_name,
                                object_short_name=
                                    short_name,
                                frame=vis_t,
                                pred_t=vis_pred,
                                gt_t=vis_gt,
                                rms_acc=
                                    acc_lookup[
                                        key
                                    ],
                                mpjpe_mm=
                                    float(
                                        vis_metric[
                                            "mpjpe_mm"
                                        ]
                                    ),
                                pa_mpjpe_mm=
                                    float(
                                        vis_metric[
                                            "pa_mpjpe_mm"
                                        ]
                                    ),
                                fitted_scale_metric=
                                    float(
                                        vis_metric[
                                            "fitted_scale"
                                        ]
                                    ),
                            )
                        )

                        visualization_rows.append(
                            {
                                "task":
                                    task,
                                "level":
                                    vis_meta[
                                        "level"
                                    ],
                                "clip":
                                    key,
                                "object_id":
                                    oid,
                                "object_full_name":
                                    full_name,
                                "object_short_name":
                                    short_name,
                                "frame":
                                    vis_t,
                                "rms_acceleration_si_m_s2":
                                    acc_lookup[
                                        key
                                    ],
                                "metric_point_count":
                                    vis_result[
                                        "metric_point_count"
                                    ],
                                "display_point_count":
                                    vis_result[
                                        "display_point_count"
                                    ],
                                "mpjpe_mm":
                                    float(
                                        vis_metric[
                                            "mpjpe_mm"
                                        ]
                                    ),
                                "pa_mpjpe_mm":
                                    float(
                                        vis_metric[
                                            "pa_mpjpe_mm"
                                        ]
                                    ),
                                "fitted_scale":
                                    float(
                                        vis_metric[
                                            "fitted_scale"
                                        ]
                                    ),
                                "pa_visual_check_mm":
                                    vis_result[
                                        "pa_check_mm"
                                    ],
                                "decomposition_error":
                                    vis_result[
                                        "decomposition_error"
                                    ],
                                "png_path":
                                    vis_result[
                                        "path"
                                    ],
                            }
                        )

                        print(
                            "[VIS SAVED]",
                            task,
                            vis_meta[
                                "level"
                            ],
                            "|",
                            key,
                            "|",
                            short_name,
                            "| frame",
                            vis_t,
                            "| N=",
                            len(vis_gt),
                            "|",
                            vis_result[
                                "path"
                            ],
                        )

                # =================================================
                # CLIP-LEVEL AGGREGATION
                # =================================================
                if eligible_objects_this_clip:

                    names = [
                        x[
                            "object_short_name"
                        ]
                        for x
                        in eligible_objects_this_clip
                    ]

                    pa_obj = np.asarray(
                        [
                            x[
                                "pa_mpjpe_mean_mm"
                            ]
                            for x
                            in eligible_objects_this_clip
                        ],
                        dtype=float,
                    )

                    mp_obj = np.asarray(
                        [
                            x[
                                "mpjpe_mean_mm"
                            ]
                            for x
                            in eligible_objects_this_clip
                        ],
                        dtype=float,
                    )

                    clip_rows.append(
                        {
                            "task":
                                task,

                            "clip":
                                key,

                            "rms_acceleration_si_m_s2":
                                acc_lookup[key],

                            "eligible_moving_objects":
                                len(
                                    eligible_objects_this_clip
                                ),

                            "object_short_names":
                                "|".join(
                                    names
                                ),

                            "mpjpe_objectmean_mm":
                                float(
                                    np.mean(
                                        mp_obj
                                    )
                                ),

                            "pa_mpjpe_objectmean_mm":
                                float(
                                    np.mean(
                                        pa_obj
                                    )
                                ),

                            "pa_mpjpe_objectmedian_mm":
                                float(
                                    np.median(
                                        pa_obj
                                    )
                                ),
                        }
                    )

                    status = "OK"

                else:
                    status = (
                        "NO_ELIGIBLE_MOVING_RIGID_OBJECT"
                    )

                status_rows.append(
                    {
                        "task":
                            task,

                        "clip":
                            key,

                        "rms_acceleration_si_m_s2":
                            acc_lookup[key],

                        "status":
                            status,

                        "eligible_moving_objects":
                            len(
                                eligible_objects_this_clip
                            ),
                    }
                )

                # -------------------------------------------------
                # Progress + safe snapshots
                # -------------------------------------------------
                nseen = len(
                    seen_targets
                )

                if (
                    nseen % 100 == 0
                    or nseen
                    == len(task_targets)
                ):
                    print(
                        f"[{task}] "
                        f"{nseen}/"
                        f"{len(task_targets)} "
                        f"target clips inferred"
                    )

                    write_csv(
                        task_out
                        / "per_frame_metrics.csv",
                        frame_rows,
                    )

                    write_csv(
                        task_out
                        / "per_object_metrics.csv",
                        object_rows,
                    )

                    write_csv(
                        task_out
                        / "per_clip_metrics.csv",
                        clip_rows,
                    )

                    write_csv(
                        task_out
                        / "clip_status.csv",
                        status_rows,
                    )

                    write_csv(
                        task_out
                        / "visualization_manifest.csv",
                        visualization_rows,
                    )

            # =====================================================
            # END-OF-TASK AUDIT
            # =====================================================
            missing = (
                task_targets
                - seen_targets
            )

            write_csv(
                task_out
                / "per_frame_metrics.csv",
                frame_rows,
            )

            write_csv(
                task_out
                / "per_object_metrics.csv",
                object_rows,
            )

            write_csv(
                task_out
                / "per_clip_metrics.csv",
                clip_rows,
            )

            write_csv(
                task_out
                / "clip_status.csv",
                status_rows,
            )

            write_csv(
                task_out
                / "visualization_manifest.csv",
                visualization_rows,
            )

            print()
            print("-" * 78)
            print(
                f"[{task}] TARGET SEEN = "
                f"{len(seen_targets)}/"
                f"{len(task_targets)}"
            )

            print(
                f"[{task}] MISSING TARGETS = "
                f"{len(missing)}"
            )

            if missing:
                print(
                    "FIRST MISSING =",
                    sorted(missing)[:10],
                )

            status_df = pd.DataFrame(
                status_rows
            )

            obj_df = pd.DataFrame(
                object_rows
            )

            clip_df = pd.DataFrame(
                clip_rows
            )

            if len(status_df):
                print()
                print(
                    f"[{task}] CLIP STATUS"
                )
                print(
                    status_df[
                        "status"
                    ]
                    .value_counts()
                    .to_string()
                )

            if len(obj_df):
                print()
                print(
                    f"[{task}] OBJECT NAME COUNTS"
                )

                print(
                    obj_df[
                        "object_short_name"
                    ]
                    .value_counts()
                    .to_string()
                )

            print()
            print(
                f"[{task}] ELIGIBLE CLIPS =",
                len(clip_df),
            )

            print(
                f"[{task}] ELIGIBLE OBJECTS =",
                len(obj_df),
            )

            print()
            print(
                f"[{task}] VISUALIZATIONS =",
                len(
                    visualization_rows
                ),
            )

            selected_vis_clips = set(
                vis_selection.keys()
            )

            completed_vis_clips = set(
                r["clip"]
                for r
                in visualization_rows
            )

            missing_vis = (
                selected_vis_clips
                - completed_vis_clips
            )

            print(
                f"[{task}] FIXED VIS CLIPS =",
                sorted(
                    selected_vis_clips
                ),
            )

            print(
                f"[{task}] VIS CLIPS WITH PNG =",
                sorted(
                    completed_vis_clips
                ),
            )

            print(
                f"[{task}] VIS CLIPS WITHOUT ELIGIBLE OBJECT =",
                sorted(
                    missing_vis
                ),
            )

            print()
            print(
                f"[{task}] SAVED TO:"
            )
            print(task_out)
            print("-" * 78)

            if missing:
                raise RuntimeError(
                    f"{task}: "
                    f"{len(missing)} target clips "
                    f"were not found."
                )

        print()
        print("=" * 78)
        print("SELECTED TASKS COMPLETE")
        print("TASKS =", selected_tasks)
        print("OUTPUT ROOT =", OUT_ROOT)
        print("=" * 78)

    finally:
        # Restore original PointWorld runtime functions.
        omap.dlmod.build_flow_sample = (
            omap.ORIG_BUILD_FLOW_SAMPLE
        )

        omap.pmod.grid_sample_transform = (
            omap.ORIG_GRID_SAMPLE
        )

        omap.pmod.gather = (
            omap.ORIG_GATHER
        )


if __name__ == "__main__":
    main()
