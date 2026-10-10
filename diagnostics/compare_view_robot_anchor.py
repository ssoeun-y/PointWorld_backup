#!/usr/bin/env python3

from pathlib import Path
import torch

ROOT = Path("diagnostics/results/view_condition_robot_anchor")

A = torch.load(
    ROOT / "camera_head__camera_left.pt",
    map_location="cpu",
    weights_only=False,
)
B = torch.load(
    ROOT / "camera_head__camera_right.pt",
    map_location="cpu",
    weights_only=False,
)

OBJECT_ID = 2


def as_btn(x, name):
    x = torch.as_tensor(x)

    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.ndim != 3:
        raise RuntimeError(f"{name}: expected (B,T,N), got {tuple(x.shape)}")

    return x


def unpack(d, tag):
    uid = as_btn(d["scene_point_uid"], f"{tag}:uid").long()
    src = as_btn(d["scene_camera_source"], f"{tag}:src").long()
    obj = as_btn(d["scene_object_id"], f"{tag}:obj").long()

    exists = as_btn(d["scene_exists"], f"{tag}:exists").bool()
    context = as_btn(d["scene_context_mask"], f"{tag}:context").bool()
    supervised = as_btn(
        d["scene_supervised_mask"], f"{tag}:supervised"
    ).bool()

    gt = torch.as_tensor(d["gt_scene_flows"])
    pred = torch.as_tensor(d["first_prediction"])

    N = gt.shape[2]

    for name, x in [
        ("uid", uid),
        ("src", src),
        ("obj", obj),
        ("exists", exists),
        ("context", context),
        ("supervised", supervised),
    ]:
        if x.shape[2] != N:
            raise RuntimeError(
                f"{tag}:{name} N={x.shape[2]} but GT N={N}"
            )

    if pred.shape != gt.shape:
        raise RuntimeError(
            f"{tag}: prediction/GT mismatch "
            f"{tuple(pred.shape)} vs {tuple(gt.shape)}"
        )

    # Official validity + target object + camera_head origin.
    # camera_head is source 0 in BOTH experiments.
    valid = (
        (~context)
        & exists
        & supervised
        & (obj == OBJECT_ID)
        & (src == 0)
    )

    return uid, valid, gt, pred


uidA, validA, gtA, predA = unpack(A, "LEFT")
uidB, validB, gtB, predB = unpack(B, "RIGHT")

# final timestep
idxA = torch.where(validA[0, -1])[0]
idxB = torch.where(validB[0, -1])[0]

mapA = {int(uidA[0, -1, i]): int(i) for i in idxA}
mapB = {int(uidB[0, -1, i]): int(i) for i in idxB}

if len(mapA) != len(idxA):
    raise RuntimeError("LEFT: duplicate camera_head UID")

if len(mapB) != len(idxB):
    raise RuntimeError("RIGHT: duplicate camera_head UID")

common = sorted(set(mapA) & set(mapB))

if not common:
    raise RuntimeError("No common camera_head object UIDs")

ia = torch.tensor([mapA[u] for u in common], dtype=torch.long)
ib = torch.tensor([mapB[u] for u in common], dtype=torch.long)

# Same physical-point trajectory sanity.
ga = gtA[0, :, ia, :]
gb = gtB[0, :, ib, :]

# Remove any pair-dependent global translation before sanity comparison.
ga_rel = ga - ga[0:1]
gb_rel = gb - gb[0:1]

gt_rel_max = (ga_rel - gb_rel).abs().max().item()

# Exact same physical head-origin point IDs, separately evaluated
# against each condition's own official GT coordinates.
errA = torch.linalg.vector_norm(
    predA[0, -1, ia] - gtA[0, -1, ia],
    dim=-1,
)
errB = torch.linalg.vector_norm(
    predB[0, -1, ib] - gtB[0, -1, ib],
    dim=-1,
)

delta = errB - errA

print("=" * 72)
print("COMMON camera_head OBJECT POINT COMPARISON")
print("=" * 72)

print("LEFT  valid head-object points =", len(mapA))
print("RIGHT valid head-object points =", len(mapB))
print("COMMON identical raw UIDs     =", len(common))

print(
    "GT relative-trajectory max diff =",
    f"{gt_rel_max * 1000:.6f} mm",
)

print("-" * 72)

print(
    "HEAD+LEFT  common-point L2 mean   =",
    f"{errA.mean().item() * 1000:.3f} mm",
)
print(
    "HEAD+RIGHT common-point L2 mean   =",
    f"{errB.mean().item() * 1000:.3f} mm",
)

print(
    "RIGHT - LEFT paired mean delta    =",
    f"{delta.mean().item() * 1000:+.3f} mm",
)
print(
    "RIGHT - LEFT paired median delta  =",
    f"{delta.median().item() * 1000:+.3f} mm",
)

print(
    "fraction RIGHT worse than LEFT    =",
    f"{(delta > 0).float().mean().item() * 100:.2f}%",
)

print("=" * 72)
