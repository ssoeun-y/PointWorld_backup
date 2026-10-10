#!/usr/bin/env python3

from pathlib import Path
import csv
import torch
import numpy as np

ROOTS = {
    "02530": Path("diagnostics/results/action_condition_02530"),
    "00470": Path("diagnostics/results/action_condition_00470"),
    "02690": Path("diagnostics/results/action_condition_02690"),
}

PAIR = "camera_head__camera_left"
OBJECT_ID = 2

VARIANTS = [
    "smooth",
    "speed_osc",
    "accel_spike",
    "reversal",
]

OUT = Path(
    "diagnostics/results/action_object_view_coupling.csv"
)


def btn(x):
    x = torch.as_tensor(x)

    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.ndim != 3:
        raise RuntimeError(f"expected B,T,N; got {x.shape}")

    return x


def mean_masked(x, m):
    y = x[m]

    if y.numel() == 0:
        return float("nan")

    return float(y.float().mean())


rows = []

for tag, root in ROOTS.items():

    baseline = torch.load(
        root / f"{PAIR}__baseline.pt",
        map_location="cpu",
        weights_only=False,
    )

    pred0 = torch.as_tensor(
        baseline["first_prediction"]
    ).float()

    obj = btn(
        baseline["scene_object_id"]
    ).long()

    src = btn(
        baseline["scene_camera_source"]
    ).long()

    exists = btn(
        baseline["scene_exists"]
    ).bool()

    context = btn(
        baseline["scene_context_mask"]
    ).bool()

    supervised = btn(
        baseline["scene_supervised_mask"]
    ).bool()

    # Same mask philosophy as official evaluation:
    valid = (
        (~context)
        & exists
        & supervised
    )

    target = valid & (obj == OBJECT_ID)
    rest = valid & (obj != OBJECT_ID)

    # Fixed view order:
    # source 0 = camera_head
    # source 1 = camera_left
    target_head = target & (src == 0)
    target_left = target & (src == 1)

    for variant in VARIANTS:

        d = torch.load(
            root / f"{PAIR}__{variant}.pt",
            map_location="cpu",
            weights_only=False,
        )

        # --------------------------------------------------
        # FAIL-FAST CONTROLS
        # --------------------------------------------------
        for k in [
            "scene_flows",
            "scene_object_id",
            "scene_exists",
            "scene_camera_source",
            "cam0_initial_rgb",
            "cam1_initial_rgb",
            "cam0_initial_depth",
            "cam1_initial_depth",
        ]:
            if not torch.equal(baseline[k], d[k]):
                raise RuntimeError(
                    f"{tag}/{variant}: control changed: {k}"
                )

        rb = baseline["robot_flows"]
        rx = d["robot_flows"]

        if not torch.equal(rb[:,0], rx[:,0]):
            raise RuntimeError(
                f"{tag}/{variant}: robot t0 changed"
            )

        if not torch.equal(rb[:,-1], rx[:,-1]):
            raise RuntimeError(
                f"{tag}/{variant}: robot t10 changed"
            )

        # --------------------------------------------------
        # Prediction response to action change
        # --------------------------------------------------
        pred = torch.as_tensor(
            d["first_prediction"]
        ).float()

        delta = torch.linalg.vector_norm(
            pred - pred0,
            dim=-1,
        )

        obj_all = mean_masked(delta, target)
        rest_all = mean_masked(delta, rest)

        obj_t10 = mean_masked(
            delta[:,-1],
            target[:,-1],
        )

        rest_t10 = mean_masked(
            delta[:,-1],
            rest[:,-1],
        )

        head_obj = mean_masked(
            delta,
            target_head,
        )

        left_obj = mean_masked(
            delta,
            target_left,
        )

        # --------------------------------------------------
        # Response concentration:
        # Is the target object receiving more response than
        # expected purely from its point count?
        # --------------------------------------------------
        valid_values = delta[valid]
        obj_values = delta[target]

        response_share = (
            float(obj_values.sum() / valid_values.sum())
            if valid_values.sum() > 0
            else float("nan")
        )

        point_share = (
            float(target.sum() / valid.sum())
            if valid.sum() > 0
            else float("nan")
        )

        concentration = (
            response_share / point_share
            if point_share > 0
            else float("nan")
        )

        obj_vs_rest = (
            obj_all / rest_all
            if rest_all > 1e-12
            else float("nan")
        )

        # View-source asymmetry on target object.
        source_ratio = (
            max(head_obj, left_obj)
            / max(min(head_obj, left_obj), 1e-12)
            if not (
                np.isnan(head_obj)
                or np.isnan(left_obj)
            )
            else float("nan")
        )

        # Input action perturbation magnitude.
        action_delta = torch.linalg.vector_norm(
            torch.as_tensor(rx).float()
            - torch.as_tensor(rb).float(),
            dim=-1,
        ).mean()

        rows.append({
            "sample": tag,
            "variant": variant,

            "input_action_delta_mm":
                float(action_delta) * 1000,

            "target_pred_delta_allT_mm":
                obj_all * 1000,

            "rest_pred_delta_allT_mm":
                rest_all * 1000,

            "target_vs_rest_ratio":
                obj_vs_rest,

            "target_pred_delta_t10_mm":
                obj_t10 * 1000,

            "rest_pred_delta_t10_mm":
                rest_t10 * 1000,

            "target_point_share":
                point_share,

            "target_response_share":
                response_share,

            "target_concentration_index":
                concentration,

            "head_object_delta_mm":
                head_obj * 1000,

            "left_object_delta_mm":
                left_obj * 1000,

            "view_source_ratio":
                source_ratio,

            "n_target_valid":
                int(target.sum()),

            "n_target_head":
                int(target_head.sum()),

            "n_target_left":
                int(target_left.sum()),
        })


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUT.open("w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(rows)


print("=" * 126)
print("ACTION -> OBJECT / VIEW COUPLING")
print("=" * 126)

print(
    f"{'sample':7s} "
    f"{'variant':12s} "
    f"{'inputΔ':>8s} "
    f"{'objΔ':>8s} "
    f"{'restΔ':>8s} "
    f"{'obj/rest':>9s} "
    f"{'conc':>7s} "
    f"{'headObj':>9s} "
    f"{'leftObj':>9s} "
    f"{'viewRatio':>9s}"
)

for r in rows:

    print(
        f"{r['sample']:7s} "
        f"{r['variant']:12s} "
        f"{r['input_action_delta_mm']:8.2f} "
        f"{r['target_pred_delta_allT_mm']:8.2f} "
        f"{r['rest_pred_delta_allT_mm']:8.2f} "
        f"{r['target_vs_rest_ratio']:9.2f} "
        f"{r['target_concentration_index']:7.2f} "
        f"{r['head_object_delta_mm']:9.2f} "
        f"{r['left_object_delta_mm']:9.2f} "
        f"{r['view_source_ratio']:9.2f}"
    )


print("\n" + "=" * 126)
print("AGGREGATE")
print("=" * 126)

ratios = np.asarray([
    r["target_vs_rest_ratio"]
    for r in rows
])

conc = np.asarray([
    r["target_concentration_index"]
    for r in rows
])

view_ratio = np.asarray([
    r["view_source_ratio"]
    for r in rows
])

print(
    "target/rest ratio median =",
    f"{np.nanmedian(ratios):.3f}"
)

print(
    "target/rest ratio > 1 =",
    f"{np.sum(ratios > 1)}/{len(ratios)}"
)

print(
    "target concentration median =",
    f"{np.nanmedian(conc):.3f}"
)

print(
    "concentration > 1 =",
    f"{np.sum(conc > 1)}/{len(conc)}"
)

print(
    "view-source ratio median =",
    f"{np.nanmedian(view_ratio):.3f}"
)

print("\nsaved =", OUT)
