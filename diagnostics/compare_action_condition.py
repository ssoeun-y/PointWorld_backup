#!/usr/bin/env python3

from pathlib import Path
import torch

ROOT = Path("diagnostics/results/action_condition_02530")
PAIR = "camera_head__camera_left"

variants = [
    "baseline",
    "smooth",
    "speed_osc",
    "accel_spike",
    "reversal",
]

D = {
    v: torch.load(
        ROOT / f"{PAIR}__{v}.pt",
        map_location="cpu",
        weights_only=False,
    )
    for v in variants
}

B = D["baseline"]
OBJECT_ID = 2


def btn(x):
    x = torch.as_tensor(x)
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x


print("=" * 78)
print("FIXED-VIEW ACTION CONDITIONING")
print("=" * 78)

# ---------------------------------------------------------
# Controls
# ---------------------------------------------------------
for v in variants[1:]:
    X = D[v]

    print(f"\n--- {v} CONTROL AUDIT ---")

    # Entire scene/GT input must remain exact.
    for k in [
        "scene_flows",
        "scene_object_id",
        "scene_exists",
    ]:
        print(
            f"{k:20s}",
            "exact=",
            torch.equal(B[k], X[k]),
        )

    # Image/depth must remain exact when saved.
    for k in [
        "cam0_initial_rgb",
        "cam1_initial_rgb",
        "cam0_initial_depth",
        "cam1_initial_depth",
    ]:
        if k in B and k in X:
            print(
                f"{k:20s}",
                "exact=",
                torch.equal(B[k], X[k]),
            )

    rb = B["robot_flows"].float()
    rx = X["robot_flows"].float()

    t0 = (rb[:, 0] - rx[:, 0]).abs().max().item()
    tN = (rb[:, -1] - rx[:, -1]).abs().max().item()

    action_delta = torch.linalg.vector_norm(
        rx - rb,
        dim=-1,
    ).mean().item()

    print(f"robot t0 maxdiff      = {t0:.10g} m")
    print(f"robot t10 maxdiff     = {tN:.10g} m")
    print(
        "interior action delta = "
        f"{action_delta*1000:.3f} mm"
    )

# ---------------------------------------------------------
# Prediction response
# ---------------------------------------------------------
obj = btn(B["scene_object_id"]).long() == OBJECT_ID
exists = btn(B["scene_exists"]).bool()

obj_mask = obj & exists

predB = torch.as_tensor(B["first_prediction"]).float()

print("\n" + "=" * 78)
print("PREDICTION RESPONSE vs BASELINE")
print("=" * 78)

for v in variants[1:]:
    X = D[v]
    predX = torch.as_tensor(X["first_prediction"]).float()

    if predX.shape != predB.shape:
        raise RuntimeError(
            f"{v}: prediction shape mismatch"
        )

    delta = torch.linalg.vector_norm(
        predX - predB,
        dim=-1,
    )

    final_obj = obj_mask[:, -1]

    obj_t10 = delta[:, -1][final_obj].mean().item()
    obj_all = delta[obj_mask].mean().item()

    all_t10 = delta[:, -1][
        exists[:, -1]
    ].mean().item()

    rb = B["robot_flows"].float()
    rx = X["robot_flows"].float()

    action_delta = torch.linalg.vector_norm(
        rx - rb,
        dim=-1,
    ).mean().item()

    gain = (
        obj_all / action_delta
        if action_delta > 0
        else float("nan")
    )

    print(f"\n{v}")
    print(
        f"  input robot-flow delta mean = "
        f"{action_delta*1000:.3f} mm"
    )
    print(
        f"  object prediction Δ t10     = "
        f"{obj_t10*1000:.3f} mm"
    )
    print(
        f"  object prediction Δ all-T   = "
        f"{obj_all*1000:.3f} mm"
    )
    print(
        f"  all-scene prediction Δ t10  = "
        f"{all_t10*1000:.3f} mm"
    )
    print(
        f"  output/input response gain   = "
        f"{gain:.3f}"
    )

print("=" * 78)
