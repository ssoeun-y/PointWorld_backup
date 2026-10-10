from pathlib import Path
import json
import numpy as np
import torch

from arguments import parse_args
from evaluation.tester import Tester


TARGET = "task-0001_episode_00012190-460:471"

OUT_DIR = Path(
    "diagnostics/results/"
    "behavior_task0001_soda_can_capture_top1"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUT_DIR / "task-0001_episode_00012190-460_471_pointworld_output.npz"

# Successful official eval from immediately before this diagnostic run.
REF_METRICS = Path(
    "eval_logs/20260812_214425-None-split=test-seed=42/metrics.json"
)


def cpu_np(x):
    return x.detach().cpu().numpy()


args = parse_args()

print("=" * 72)
print("1. INITIALIZE OFFICIAL TESTER")
print("=" * 72)

tester = Tester(args)

print()
print("=" * 72)
print("2. BUILD OFFICIAL TEST DATALOADER")
print("=" * 72)

dl, info = tester._build_eval_loader("test", enable_mask=False)

print("loader_info =", info)

batch = next(iter(dl))

keys = [str(x) for x in batch["__key__"]]
domains = [str(x) for x in batch["__domain__"]]

print("batch_keys =", keys)
print("domains    =", domains)

target_exact = (
    len(keys) == 1
    and keys[0] == TARGET
)

print("TARGET_EXACT =", target_exact)

if not target_exact:
    raise RuntimeError(
        f"Expected exact target {TARGET}, got {keys}"
    )


print()
print("=" * 72)
print("3. OFFICIAL POINTWORLD FORWARD")
print("=" * 72)

batch = {
    k: (
        v.to(tester.device, non_blocking=True)
        if isinstance(v, torch.Tensor)
        else v
    )
    for k, v in batch.items()
}

tester.model.eval()

with torch.no_grad():
    outputs = tester.model(batch, training=False)
    total_loss, loss_dict = tester.model.loss_fn(
        outputs,
        batch,
        training=False,
    )


print("pred scene_flows shape =", tuple(outputs["scene_flows"].shape))
print("gt   scene_flows shape =", tuple(batch["gt_scene_flows"].shape))

print()
print("loss_dict l2/mean =", loss_dict.get("l2/mean"))
print("loss_dict behavior/l2/mean =", loss_dict.get("behavior/l2/mean"))


print()
print("=" * 72)
print("4. OFFICIAL MASK + L2 REPRODUCTION")
print("=" * 72)

pred = outputs["scene_flows"]
gt = batch["gt_scene_flows"]

context = batch["scene_context_mask"]
exists = batch["scene_exists"]
supervised = batch["scene_supervised_mask"]

# Release tensors may be (B,T,N,1).
if context.ndim == 4:
    context = context.squeeze(-1)
if exists.ndim == 4:
    exists = exists.squeeze(-1)
if supervised.ndim == 4:
    supervised = supervised.squeeze(-1)

official_valid = (~context.bool()) & exists.bool() & supervised.bool()

l2 = torch.linalg.vector_norm(
    pred - gt,
    dim=-1,
)

manual_l2 = float(l2[official_valid].mean().item())
official_loss_l2 = float(loss_dict["l2/mean"])

print("official_valid_count =", int(official_valid.sum().item()))
print("manual_l2            =", repr(manual_l2))
print("loss_dict_l2          =", repr(official_loss_l2))
print(
    "manual_vs_loss_absdiff =",
    repr(abs(manual_l2 - official_loss_l2)),
)

manual_exact = np.isclose(
    manual_l2,
    official_loss_l2,
    rtol=0.0,
    atol=1e-8,
)

print("OFFICIAL_L2_REPRODUCED =", bool(manual_exact))


print()
print("=" * 72)
print("5. COMPARE TO PREVIOUS OFFICIAL eval.py METRICS")
print("=" * 72)

reference_metric = None
reference_diff = None

if REF_METRICS.exists():
    metrics = json.loads(REF_METRICS.read_text())

    metric_key = "full_eval/test/l2/mean"

    print("reference_metrics =", REF_METRICS)
    print("reference_key_exists =", metric_key in metrics)

    if metric_key in metrics:
        reference_metric = float(metrics[metric_key])
        reference_diff = abs(
            reference_metric - official_loss_l2
        )

        print("reference_eval_l2 =", repr(reference_metric))
        print("capture_l2        =", repr(official_loss_l2))
        print("reference_absdiff =", repr(reference_diff))
        print(
            "REFERENCE_EVAL_L2_MATCH =",
            bool(
                np.isclose(
                    reference_metric,
                    official_loss_l2,
                    rtol=0.0,
                    atol=1e-8,
                )
            ),
        )
else:
    print("reference metrics file missing:", REF_METRICS)


print()
print("=" * 72)
print("6. SAVE RAW POINTWORLD OUTPUT")
print("=" * 72)

save_dict = {
    "sample_key": np.array(keys[0]),

    # Model output / GT
    "pred_scene_flows": cpu_np(pred[0]),
    "gt_scene_flows": cpu_np(gt[0]),

    # Actual scene input seen by PointWorld
    "input_scene_flows": cpu_np(batch["scene_flows"][0]),

    # Absolute per-point error
    "l2": cpu_np(l2[0]),

    # Official masks
    "official_valid": cpu_np(official_valid[0]),
    "scene_exists": cpu_np(exists[0]),
    "scene_context_mask": cpu_np(context[0]),
    "scene_supervised_mask": cpu_np(supervised[0]),

    # Prediction representation
    "pred_scene_relative": cpu_np(outputs["scene_relative"][0]),
    "pred_scene_relative_norm": cpu_np(outputs["scene_relative_norm"][0]),
    "confidence": cpu_np(outputs["confidence"][0]),

    # Scalar audits
    "manual_l2_mean": np.array(manual_l2),
    "loss_dict_l2_mean": np.array(official_loss_l2),
}

# Save these if present, without assuming shape.
for name in [
    "scene_moved_mask",
    "scene_static_mask",
    "point_weights",
    "scene_colors",
    "scene_normals",
]:
    if name in batch and isinstance(batch[name], torch.Tensor):
        save_dict[name] = cpu_np(batch[name][0])

np.savez_compressed(
    OUT_NPZ,
    **save_dict,
)

print("saved =", OUT_NPZ)
print("saved_bytes =", OUT_NPZ.stat().st_size)

print()
print("NPZ KEYS / SHAPES:")
with np.load(OUT_NPZ, allow_pickle=False) as z:
    for k in z.files:
        v = z[k]
        print(f"  {k:28s} shape={v.shape} dtype={v.dtype}")

print()
print("=" * 72)
print("FINAL")
print("=" * 72)

print("TARGET_EXACT =", target_exact)
print("OFFICIAL_L2_REPRODUCED =", bool(manual_exact))

if reference_metric is not None:
    print(
        "REFERENCE_EVAL_L2_MATCH =",
        bool(
            np.isclose(
                reference_metric,
                official_loss_l2,
                rtol=0.0,
                atol=1e-8,
            )
        ),
    )

print("NO MBNSF")
print("NO OFFICIAL FILE MODIFICATION")
print("CAPTURE_SUCCESS = True")
