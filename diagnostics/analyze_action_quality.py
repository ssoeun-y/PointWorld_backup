#!/usr/bin/env python3

from pathlib import Path
import csv
import numpy as np
import torch

ROOTS = {
    "02530": Path("diagnostics/results/action_condition_02530"),
    "00470": Path("diagnostics/results/action_condition_00470"),
    "02690": Path("diagnostics/results/action_condition_02690"),
}

PAIR = "camera_head__camera_left"

VARIANTS = [
    "baseline",
    "smooth",
    "speed_osc",
    "accel_spike",
    "reversal",
]

OBJECT_ID = 2

OUT_CSV = Path(
    "diagnostics/results/action_quality_summary.csv"
)


def squeeze_btn(x):
    x = torch.as_tensor(x)

    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    return x


def masked_values(x, mask):
    vals = x[mask]
    return vals.float()


def safe_mean(x):
    if x.numel() == 0:
        return float("nan")
    return float(x.mean())


def safe_p95(x):
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x, 0.95))


def robot_quality(d):
    """
    robot_flows are processed robot point trajectories.
    We use per-step finite differences, so units are:
      velocity-like: mm / step
      acceleration-like: mm / step^2
      jerk-like: mm / step^3
    No assumption about physical dt is made.
    """
    x = torch.as_tensor(d["robot_flows"]).float()
    exists = squeeze_btn(d["robot_exists"]).bool()

    assert x.ndim == 4
    assert exists.ndim == 3

    # --------------------------------------------------
    # Velocity / per-step displacement
    # --------------------------------------------------
    v = x[:, 1:] - x[:, :-1]

    valid_v = (
        exists[:, 1:]
        & exists[:, :-1]
    )

    speed = torch.linalg.vector_norm(v, dim=-1)

    speed_vals = masked_values(speed, valid_v)

    # Temporal speed profile:
    # average speed across valid robot points at each interval.
    temporal_speed = []

    for t in range(speed.shape[1]):
        vals = speed[:, t][valid_v[:, t]]

        if vals.numel() > 0:
            temporal_speed.append(float(vals.mean()))

    temporal_speed = np.asarray(
        temporal_speed,
        dtype=np.float64,
    )

    speed_mean = (
        float(temporal_speed.mean())
        if temporal_speed.size
        else float("nan")
    )

    speed_cv = (
        float(temporal_speed.std() / temporal_speed.mean())
        if temporal_speed.size
        and temporal_speed.mean() > 1e-12
        else float("nan")
    )

    # --------------------------------------------------
    # Acceleration
    # --------------------------------------------------
    a = v[:, 1:] - v[:, :-1]

    valid_a = (
        valid_v[:, 1:]
        & valid_v[:, :-1]
    )

    acc_mag = torch.linalg.vector_norm(a, dim=-1)
    acc_vals = masked_values(acc_mag, valid_a)

    # --------------------------------------------------
    # Jerk
    # --------------------------------------------------
    j = a[:, 1:] - a[:, :-1]

    valid_j = (
        valid_a[:, 1:]
        & valid_a[:, :-1]
    )

    jerk_mag = torch.linalg.vector_norm(j, dim=-1)
    jerk_vals = masked_values(jerk_mag, valid_j)

    # --------------------------------------------------
    # Direction change / reversal
    # Natural reversal definition:
    # consecutive velocity vectors have negative cosine.
    # --------------------------------------------------
    v_prev = v[:, :-1]
    v_next = v[:, 1:]

    n_prev = torch.linalg.vector_norm(
        v_prev,
        dim=-1,
    )
    n_next = torch.linalg.vector_norm(
        v_next,
        dim=-1,
    )

    direction_valid = (
        valid_a
        & (n_prev > 1e-9)
        & (n_next > 1e-9)
    )

    cosine = (
        (v_prev * v_next).sum(dim=-1)
        / (n_prev * n_next).clamp_min(1e-12)
    )

    cosine = cosine.clamp(-1.0, 1.0)

    cos_vals = cosine[direction_valid]

    if cos_vals.numel():
        angle_deg = (
            torch.acos(cos_vals)
            * 180.0
            / np.pi
        )

        mean_turn_deg = float(angle_deg.mean())
        reversal_fraction = float(
            (cos_vals < 0).float().mean()
        )
    else:
        mean_turn_deg = float("nan")
        reversal_fraction = float("nan")

    # --------------------------------------------------
    # Path efficiency
    # endpoint displacement / total traveled distance.
    # 1 = no unnecessary backtracking.
    # --------------------------------------------------
    path_length = (
        speed * valid_v.float()
    ).sum(dim=1)

    endpoint = torch.linalg.vector_norm(
        x[:, -1] - x[:, 0],
        dim=-1,
    )

    valid_path = (
        exists[:, 0]
        & exists[:, -1]
        & (path_length > 1e-9)
    )

    efficiency = (
        endpoint[valid_path]
        / path_length[valid_path]
    )

    return {
        "mean_speed_mm_step":
            safe_mean(speed_vals) * 1000.0,

        "speed_cv":
            speed_cv,

        "mean_acc_mm_step2":
            safe_mean(acc_vals) * 1000.0,

        "p95_acc_mm_step2":
            safe_p95(acc_vals) * 1000.0,

        "mean_jerk_mm_step3":
            safe_mean(jerk_vals) * 1000.0,

        "p95_jerk_mm_step3":
            safe_p95(jerk_vals) * 1000.0,

        "mean_turn_deg":
            mean_turn_deg,

        "reversal_fraction":
            reversal_fraction,

        "path_efficiency":
            safe_mean(efficiency),
    }


def prediction_response(base, x):
    pred_b = torch.as_tensor(
        base["first_prediction"]
    ).float()

    pred_x = torch.as_tensor(
        x["first_prediction"]
    ).float()

    obj = (
        squeeze_btn(base["scene_object_id"]).long()
        == OBJECT_ID
    )

    exists = squeeze_btn(
        base["scene_exists"]
    ).bool()

    obj_mask = obj & exists

    delta = torch.linalg.vector_norm(
        pred_x - pred_b,
        dim=-1,
    )

    final_obj = obj_mask[:, -1]

    obj_t10 = float(
        delta[:, -1][final_obj].mean()
    )

    obj_all = float(
        delta[obj_mask].mean()
    )

    all_t10 = float(
        delta[:, -1][exists[:, -1]].mean()
    )

    rb = torch.as_tensor(
        base["robot_flows"]
    ).float()

    rx = torch.as_tensor(
        x["robot_flows"]
    ).float()

    input_delta = float(
        torch.linalg.vector_norm(
            rx - rb,
            dim=-1,
        ).mean()
    )

    gain = (
        obj_all / input_delta
        if input_delta > 1e-12
        else 0.0
    )

    return {
        "input_delta_mm":
            input_delta * 1000.0,

        "pred_obj_t10_mm":
            obj_t10 * 1000.0,

        "pred_obj_allT_mm":
            obj_all * 1000.0,

        "pred_scene_t10_mm":
            all_t10 * 1000.0,

        "response_gain":
            gain,
    }


rows = []

for sample, root in ROOTS.items():

    data = {}

    for variant in VARIANTS:
        f = root / f"{PAIR}__{variant}.pt"

        if not f.exists():
            raise FileNotFoundError(f)

        data[variant] = torch.load(
            f,
            map_location="cpu",
            weights_only=False,
        )

    base = data["baseline"]

    # --------------------------------------------------
    # Re-check controls before using results.
    # --------------------------------------------------
    for variant in VARIANTS[1:]:
        x = data[variant]

        for k in [
            "scene_flows",
            "scene_object_id",
            "scene_exists",
            "cam0_initial_rgb",
            "cam1_initial_rgb",
            "cam0_initial_depth",
            "cam1_initial_depth",
        ]:
            if not torch.equal(base[k], x[k]):
                raise RuntimeError(
                    f"{sample}/{variant}: "
                    f"control changed: {k}"
                )

        rb = base["robot_flows"]
        rx = x["robot_flows"]

        if not torch.equal(rb[:, 0], rx[:, 0]):
            raise RuntimeError(
                f"{sample}/{variant}: robot t0 changed"
            )

        if not torch.equal(rb[:, -1], rx[:, -1]):
            raise RuntimeError(
                f"{sample}/{variant}: robot final changed"
            )

    quality = {
        v: robot_quality(data[v])
        for v in VARIANTS
    }

    q0 = quality["baseline"]

    for variant in VARIANTS:

        q = quality[variant]

        if variant == "baseline":
            response = {
                "input_delta_mm": 0.0,
                "pred_obj_t10_mm": 0.0,
                "pred_obj_allT_mm": 0.0,
                "pred_scene_t10_mm": 0.0,
                "response_gain": 0.0,
            }
        else:
            response = prediction_response(
                base,
                data[variant],
            )

        row = {
            "sample": sample,
            "variant": variant,
            **q,
            **response,

            "delta_speed_cv":
                q["speed_cv"] - q0["speed_cv"],

            "delta_mean_acc_mm_step2":
                q["mean_acc_mm_step2"]
                - q0["mean_acc_mm_step2"],

            "delta_mean_jerk_mm_step3":
                q["mean_jerk_mm_step3"]
                - q0["mean_jerk_mm_step3"],

            "delta_mean_turn_deg":
                q["mean_turn_deg"]
                - q0["mean_turn_deg"],

            "delta_reversal_fraction":
                q["reversal_fraction"]
                - q0["reversal_fraction"],

            "delta_path_inefficiency":
                (
                    (1.0 - q["path_efficiency"])
                    - (1.0 - q0["path_efficiency"])
                ),
        }

        rows.append(row)


OUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# Compact human-readable output
# ============================================================
print("=" * 112)
print("ACTION QUALITY vs OBJECT PREDICTION RESPONSE")
print("=" * 112)

for sample in ROOTS:

    print(f"\n################ SAMPLE {sample} ################")

    print(
        f"{'variant':12s} "
        f"{'speedCV':>8s} "
        f"{'acc':>9s} "
        f"{'jerk':>9s} "
        f"{'turn°':>8s} "
        f"{'rev%':>7s} "
        f"{'eff':>7s} "
        f"{'objΔallT':>11s} "
        f"{'gain':>7s}"
    )

    for r in rows:
        if r["sample"] != sample:
            continue

        print(
            f"{r['variant']:12s} "
            f"{r['speed_cv']:8.3f} "
            f"{r['mean_acc_mm_step2']:9.2f} "
            f"{r['mean_jerk_mm_step3']:9.2f} "
            f"{r['mean_turn_deg']:8.2f} "
            f"{100*r['reversal_fraction']:7.2f} "
            f"{r['path_efficiency']:7.3f} "
            f"{r['pred_obj_allT_mm']:11.2f} "
            f"{r['response_gain']:7.3f}"
        )


# ============================================================
# Exploratory correlations across 12 perturbation conditions.
# NOT a significance/generalization claim.
# ============================================================
pert = [
    r for r in rows
    if r["variant"] != "baseline"
]

try:
    from scipy.stats import spearmanr

    candidates = [
        "delta_speed_cv",
        "delta_mean_acc_mm_step2",
        "delta_mean_jerk_mm_step3",
        "delta_mean_turn_deg",
        "delta_reversal_fraction",
        "delta_path_inefficiency",
    ]

    print("\n" + "=" * 112)
    print("EXPLORATORY SPEARMAN vs object prediction Δ all-T")
    print("(12 perturbation conditions only; descriptive, not final statistics)")
    print("=" * 112)

    y = np.asarray([
        r["pred_obj_allT_mm"]
        for r in pert
    ])

    for metric in candidates:
        x = np.asarray([
            r[metric]
            for r in pert
        ])

        rho, pval = spearmanr(x, y)

        print(
            f"{metric:32s} "
            f"rho={rho:+.3f}  p={pval:.4g}"
        )

except Exception as e:
    print(
        "\nSpearman skipped:",
        repr(e),
    )

print("\nsaved =", OUT_CSV)
