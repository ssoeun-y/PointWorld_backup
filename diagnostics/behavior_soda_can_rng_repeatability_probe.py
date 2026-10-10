from pathlib import Path
import random
import numpy as np
import torch
from scipy.spatial.distance import pdist

from arguments import parse_args
from evaluation.tester import Tester


TARGET = "task-0001_episode_00012190-460:471"

CAPTURE = Path(
    "diagnostics/results/"
    "behavior_task0001_soda_can_gt_top5_structural/captures/"
    "rank1_task-0001_episode_00012190-460_471.npz"
)


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pair_p95_mm(x):
    d0 = pdist(x[0].astype(np.float64))
    d10 = pdist(x[-1].astype(np.float64))

    return float(
        np.percentile(
            np.abs(d10 - d0) * 1000.0,
            95,
        )
    )


def point_delta_mm(a, b):
    return (
        np.linalg.norm(
            a.astype(np.float64)
            - b.astype(np.float64),
            axis=-1,
        )
        * 1000.0
    )


args = parse_args()

print("=" * 72)
print("1. LOAD OFFICIAL POINTWORLD")
print("=" * 72)

tester = Tester(args)

dl, _ = tester._build_eval_loader(
    "test",
    enable_mask=False,
)

tester.model.eval()


print()
print("=" * 72)
print("2. FIND EXACT RANK-1 SAMPLE")
print("=" * 72)

target_batch = None

for batch in dl:

    key = str(batch["__key__"][0])

    print("loader key =", key)

    if key == TARGET:
        target_batch = batch
        break


if target_batch is None:
    raise RuntimeError(
        f"Target not found: {TARGET}"
    )

print("TARGET_EXACT = True")


target_batch = {
    k: (
        v.to(
            tester.device,
            non_blocking=True,
        )
        if isinstance(v, torch.Tensor)
        else v
    )
    for k, v in target_batch.items()
}


print()
print("=" * 72)
print("3. LOAD EXACT CAN INDICES")
print("=" * 72)

with np.load(
    CAPTURE,
    allow_pickle=False,
) as z:

    idx = z["scene_indices"].copy()

    old_gt_can = z["gt"].copy()


print("n_can_points =", len(idx))


gt_now = (
    target_batch[
        "gt_scene_flows"
    ][0]
    .detach()
    .cpu()
    .numpy()
)

gt_can = gt_now[:, idx, :]

print(
    "GT_CAPTURE_EXACT =",
    np.array_equal(
        gt_can,
        old_gt_can,
    ),
)

if not np.array_equal(
    gt_can,
    old_gt_can,
):
    raise RuntimeError(
        "GT changed; stop"
    )


def run_one(seed):

    reset_rng(seed)

    # Clone tensors so all forwards see
    # exactly the same input payload.
    b = {
        k: (
            v.clone()
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in target_batch.items()
    }

    with torch.inference_mode():

        out = tester.model(
            b,
            training=False,
        )

    pred = (
        out["scene_flows"][0]
        .detach()
        .cpu()
        .numpy()
    )

    return pred[:, idx, :]


print()
print("=" * 72)
print("4. CONTROLLED FORWARDS")
print("=" * 72)

print("RUN A: seed 42")
pred_a = run_one(42)

print("RUN B: seed 42")
pred_b = run_one(42)

print("RUN C: seed 43")
pred_c = run_one(43)


print()
print("=" * 72)
print("5. SAME-SEED REPEATABILITY")
print("=" * 72)

ab = point_delta_mm(
    pred_a,
    pred_b,
)

print(
    "SAME_SEED_PRED_EXACT =",
    np.array_equal(
        pred_a,
        pred_b,
    ),
)

print(
    "same_seed_delta_mean_mm =",
    float(ab.mean()),
)

print(
    "same_seed_delta_p95_mm =",
    float(
        np.percentile(
            ab,
            95,
        )
    ),
)

print(
    "same_seed_delta_max_mm =",
    float(ab.max()),
)


print()
print("=" * 72)
print("6. DIFFERENT-SEED EFFECT")
print("=" * 72)

ac = point_delta_mm(
    pred_a,
    pred_c,
)

print(
    "different_seed_delta_mean_mm =",
    float(ac.mean()),
)

print(
    "different_seed_delta_p95_mm =",
    float(
        np.percentile(
            ac,
            95,
        )
    ),
)

print(
    "different_seed_delta_max_mm =",
    float(ac.max()),
)


print()
print("=" * 72)
print("7. STRUCTURAL RESULT")
print("=" * 72)

gt_p95 = pair_p95_mm(
    gt_can
)

a_p95 = pair_p95_mm(
    pred_a
)

b_p95 = pair_p95_mm(
    pred_b
)

c_p95 = pair_p95_mm(
    pred_c
)

print(
    "GT_pair_P95_mm =",
    gt_p95,
)

print(
    "seed42_runA_pair_P95_mm =",
    a_p95,
)

print(
    "seed42_runB_pair_P95_mm =",
    b_p95,
)

print(
    "seed43_pair_P95_mm =",
    c_p95,
)


print()
print("=" * 72)
print("FINAL")
print("=" * 72)

same_seed_close = np.allclose(
    pred_a,
    pred_b,
    rtol=0.0,
    atol=1e-7,
)

different_seed_effect = bool(
    np.max(ac) > 1e-3
)

print(
    "SAME_SEED_ALLCLOSE_1E7 =",
    same_seed_close,
)

print(
    "DIFFERENT_SEED_EFFECT_PRESENT =",
    different_seed_effect,
)

if (
    same_seed_close
    and different_seed_effect
):
    print(
        "SERIALIZATION_RNG_DIAGNOSIS_CONFIRMED = True"
    )
else:
    print(
        "SERIALIZATION_RNG_DIAGNOSIS_CONFIRMED = False"
    )

print("NO OFFICIAL FILE MODIFICATION")
print("GPU_PHYSICAL = 1")
print("RNG_PROBE_DONE = True")
