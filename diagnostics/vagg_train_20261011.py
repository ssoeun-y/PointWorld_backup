#!/usr/bin/env python3
import argparse
import copy
import math
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn


def mask(x, shape, name):
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"{name} missing")
    if tuple(x.shape) == (*shape, 1):
        x = x.squeeze(-1)
    if tuple(x.shape) != shape:
        raise ValueError(
            f"{name}: {tuple(x.shape)} != {shape}"
        )
    return x.bool()


def valid_points(d):
    B, T, N, _ = d["gt_scene_flows"].shape
    s = (B, T, N)
    return (
        mask(d["scene_exists"], s, "exists")
        & mask(d["scene_supervised_mask"], s, "supervised")
        & ~mask(d["scene_context_mask"], s, "context")
    )


def clock(d):
    r = d["robot_flows"].float()
    B, T, N, _ = r.shape
    valid = mask(
        d["robot_exists"], (B, T, N), "robot_exists"
    )
    dx = (r[:, 1:] - r[:, :-1]).norm(dim=-1)
    both = valid[:, 1:] & valid[:, :-1]
    speed = (
        (dx * both).sum(-1)
        / both.sum(-1).clamp_min(1)
    )
    ds = torch.cat(
        [torch.zeros_like(speed[:, :1]),
         speed[:, 1:] - speed[:, :-1]], 1
    )
    center = (
        (r * valid.unsqueeze(-1)).sum(-2)
        / valid.sum(-1, keepdim=True).clamp_min(1)
    )
    progress = (
        center[:, 1:] - center[:, :1]
    ).norm(dim=-1)

    condition = torch.stack(
        [speed / .02, ds / .02, progress / .20],
        dim=-1
    ).clamp(-5, 5)
    return condition.detach(), speed.detach()


class VAGGHead(nn.Module):
    def __init__(self, original, dim, T=10, gated=True):
        super().__init__()
        self.base = original
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.point = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 24)
        )
        self.cond = nn.Sequential(
            nn.Linear(3, 32),
            nn.SiLU(),
            nn.Linear(32, 8)
        )
        self.static = nn.Parameter(
            torch.randn(T, 8) * .02
        )
        self.gated = gated
        self.condition = None
        self.base_output = None
        self.T = T

        nn.init.zeros_(self.point[-1].weight)
        nn.init.zeros_(self.point[-1].bias)

    def forward(self, x):
        B, N, D = x.shape
        base = self.base(x)
        self.base_output = base.detach()

        assert base.shape == (B, N, 3 * self.T)

        basis = self.point(x.float()).view(B, N, 8, 3)

        if self.gated:
            assert self.condition is not None
            assert self.condition.shape == (B, self.T, 3)
            time_code = self.cond(
                self.condition.float()
            )
        else:
            time_code = self.static.unsqueeze(0).expand(
                B, -1, -1
            )

        residual = torch.einsum(
            "bnkc,btk->bntc", basis, time_code
        ) / math.sqrt(8)

        return base + (
            .15 * residual.tanh()
        ).reshape(B, N, self.T * 3).to(base.dtype)



def pa_fit(pred, gt):
    # SVD and Procrustes alignment must remain FP32 under BF16 training.
    with torch.autocast(device_type="cuda", enabled=False):
        p = pred.float()
        g = gt.float()

        with torch.no_grad():
            a = p.detach() - p.detach().mean(0)
            b = g.detach() - g.detach().mean(0)

            U, S, Vh = torch.linalg.svd(
                a.T @ b, full_matrices=False
            )

            det = torch.det(U @ Vh)
            sign = torch.where(
                det < 0,
                det.new_tensor(-1.0),
                det.new_tensor(1.0)
            )
            D = torch.diag(torch.stack([
                sign.new_ones(()),
                sign.new_ones(()),
                sign
            ]))

            R = U @ D @ Vh
            scale = (
                ((a @ R) * b).sum()
                / a.square().sum().clamp_min(1e-10)
            ).clamp(0.25, 4.0)

        aligned = (
            (p - p.mean(0)) @ R * scale
            + g.mean(0)
        )
        return (aligned - g).norm(dim=-1).mean()


def rigid_patches(d, seed=117):
    gt = d["gt_scene_flows"]
    B, T, N, _ = gt.shape
    valid = valid_points(d)
    patches = []

    for b in range(B):
        eligible = torch.where(
            valid[b, 1:].all(0)
        )[0]
        if len(eligible) < 32:
            continue

        rng = torch.Generator(
            device="cpu"
        ).manual_seed(seed + b)

        ids = torch.randperm(
            len(eligible), generator=rng
        )[:min(len(eligible), 256)].to(
            eligible.device
        )
        eligible = eligible[ids]
        coords = d["scene_flows"][
            b, 0, eligible
        ].float()

        chosen = 0
        anchors = torch.randperm(
            len(eligible), generator=rng
        )[:min(len(eligible), 30)]

        for k in anchors:
            if chosen >= 3:
                break

            distance = (
                coords - coords[int(k)]
            ).norm(dim=-1)
            nearest = distance.topk(
                24, largest=False
            ).indices

            if (
                distance[nearest].max() > .08
                or distance[nearest].max() < .008
            ):
                continue

            sub = eligible[nearest]
            d0 = torch.pdist(gt[b, 0, sub].float())
            d10 = torch.pdist(gt[b, -1, sub].float())

            if (
                (d0 - d10).abs().quantile(.95)
                > .003
            ):
                continue

            patches.append((b, sub))
            chosen += 1

    return patches


def patch_loss(pred, gt, patches):
    losses = []
    for b, sub in patches:
        for t in (3, 6, 10):
            if t < pred.shape[1]:
                losses.append(
                    pa_fit(
                        pred[b, t, sub],
                        gt[b, t, sub]
                    )
                )

    if not losses:
        return pred.sum() * 0., 0

    return torch.stack(losses).mean(), len(losses)



def _object_groups(d, max_groups=None, max_points=None):
    ids = d.get("scene_object_id")
    if ids is None:
        raise RuntimeError("OBJECT IDS MISSING")

    B, T, N, _ = d["gt_scene_flows"].shape
    if tuple(ids.shape) != (B, T, N, 1):
        raise RuntimeError(f"BAD OBJECT ID SHAPE: {tuple(ids.shape)}")

    valid = valid_points(d)
    gt = d["gt_scene_flows"]
    groups = []

    for b in range(B):
        support = valid[b, 1:].all(0)

        for oid in torch.unique(ids[b, 0, support, 0]):
            if int(oid) == 0:
                continue

            idx = torch.where(
                support & (ids[b, 0, :, 0] == oid)
            )[0]

            if len(idx) < 12:
                continue

            # GT rigidity verification uses at most 96 sampled points.
            check = idx
            if len(check) > 96:
                take = torch.linspace(
                    0, len(check)-1, 96,
                    device=check.device
                ).long()
                check = check[take]

            with torch.no_grad(), torch.autocast(
                device_type="cuda", enabled=False
            ):
                xyz = gt[b, 0, check].float()
                centered = xyz - xyz.mean(0)
                rank = int(torch.linalg.matrix_rank(
                    centered.T @ centered
                ))

                d0 = torch.pdist(gt[b, 0, check].float())
                d1 = torch.pdist(gt[b, -1, check].float())
                rigidity_gap = float(
                    (d0-d1).abs().quantile(.95)
                )

            if rank < 2 or rigidity_gap > .003:
                continue

            if max_points and len(idx) > max_points:
                take = torch.linspace(
                    0, len(idx)-1, max_points,
                    device=idx.device
                ).long()
                idx = idx[take]

            groups.append((b, int(oid), idx))

            if max_groups and len(groups) >= max_groups:
                return groups

    return groups


def _object_pa_train_loss(pred, d):
    gt = d["gt_scene_flows"]
    losses = []

    for b, oid, idx in _object_groups(
        d, max_groups=8, max_points=96
    ):
        for frame in (3, 6, 10):
            if frame < pred.shape[1]:
                losses.append(
                    pa_fit(pred[b, frame, idx], gt[b, frame, idx])
                )

    if not losses:
        return pred.sum() * 0.

    return torch.stack(losses).mean()


@torch.no_grad()
def _full_pa_mm(pred, gt):
    with torch.autocast(device_type="cuda", enabled=False):
        p = pred.float()
        g = gt.float()

        pc = p - p.mean(0)
        gc = g - g.mean(0)

        U, S, Vh = torch.linalg.svd(
            pc.T @ gc, full_matrices=False
        )

        sign = torch.where(
            torch.det(U @ Vh) < 0,
            S.new_tensor(-1.),
            S.new_tensor(1.)
        )

        D = torch.diag(torch.stack([
            S.new_ones(()),
            S.new_ones(()),
            sign
        ]))

        R = U @ D @ Vh
        scale = (
            ((pc @ R) * gc).sum()
            / pc.square().sum().clamp_min(1e-12)
        )

        aligned = (pc @ R) * scale
        return float((aligned-gc).norm(dim=-1).mean()) * 1000.


def _task_from_clip(clip):
    import re
    match = re.search(
        r'(?:task[-_])?(0000|0001|0006|0008|0010)(?:_|-|$)',
        str(clip)
    )
    if match is None:
        raise RuntimeError("UNKNOWN TASK: " + str(clip))
    return match.group(1)


@torch.no_grad()
def _evaluate_objects(base, new, d, clip):
    gt = d["gt_scene_flows"]
    rows = []

    for b, oid, idx in _object_groups(d):
        gt_end = gt[b, -1, idx].float()
        old_end = base[b, -1, idx].float()
        new_end = new[b, -1, idx].float()

        old_endpoint = float(
            (old_end.mean(0)-gt_end.mean(0)).norm()
        ) * 1000.
        new_endpoint = float(
            (new_end.mean(0)-gt_end.mean(0)).norm()
        ) * 1000.

        for t in range(1, gt.shape[1]):
            g = gt[b,t,idx].float()
            p = base[b,t,idx].float()
            q = new[b,t,idx].float()

            rows.append({
                "task": _task_from_clip(clip),
                "clip": str(clip),
                "object_id": oid,
                "frame": t,
                "n_points": len(idx),
                "base_pa_mpjpe_mm": _full_pa_mm(p,g),
                "vagg_pa_mpjpe_mm": _full_pa_mm(q,g),
                "base_gt_l2_mm": float(
                    (p-g).norm(dim=-1).mean()
                ) * 1000.,
                "vagg_gt_l2_mm": float(
                    (q-g).norm(dim=-1).mean()
                ) * 1000.,
                "base_endpoint_mm": (
                    old_endpoint if t == gt.shape[1]-1 else float("nan")
                ),
                "vagg_endpoint_mm": (
                    new_endpoint if t == gt.shape[1]-1 else float("nan")
                )
            })

    return rows


def _save_object_metrics(rows, out):
    import pandas as pd

    d = pd.DataFrame(rows)

    if d.empty:
        raise RuntimeError("OBJECT PA EVALUATION RETURNED ZERO ROWS")

    expected = {"0000","0001","0006","0008","0010"}
    found = set(d["task"].unique())

    d.to_csv(out/"object_pa_per_frame.csv", index=False)
    if found != expected:
        raise RuntimeError(
            f"INCOMPLETE FIVE TASKS: found={found}"
        )

    d.to_csv(out/"object_pa_per_frame.csv",index=False)

    clip = d.groupby(
        ["task","clip"],as_index=False
    ).agg(
        base_pa_mpjpe_mm=("base_pa_mpjpe_mm","mean"),
        vagg_pa_mpjpe_mm=("vagg_pa_mpjpe_mm","mean"),
        base_gt_l2_mm=("base_gt_l2_mm","mean"),
        vagg_gt_l2_mm=("vagg_gt_l2_mm","mean")
    )

    endpoint = d[d.frame==d.frame.max()].groupby(
        ["task","clip"],as_index=False
    ).agg(
        base_endpoint_mm=("base_endpoint_mm","mean"),
        vagg_endpoint_mm=("vagg_endpoint_mm","mean")
    )

    clip = clip.merge(
        endpoint,on=["task","clip"],
        how="left",validate="one_to_one"
    )
    clip.to_csv(out/"object_pa_per_clip.csv",index=False)

    summary=[]

    for task in sorted(expected)+["ALL"]:
        a = clip if task=="ALL" else clip[clip.task==task]
        record={"task":task,"clips":len(a)}

        for metric in [
            "pa_mpjpe_mm","gt_l2_mm","endpoint_mm"
        ]:
            gain = a["base_"+metric]-a["vagg_"+metric]
            record[metric+"_base_median"]=float(
                a["base_"+metric].median()
            )
            record[metric+"_new_median"]=float(
                a["vagg_"+metric].median()
            )
            record[metric+"_gain_median"]=float(
                gain.median()
            )
            record[metric+"_improved_pct"]=float(
                100*(gain>0).mean()
            )

        summary.append(record)
        print("OBJECT_RESULT",record,flush=True)

    pd.DataFrame(summary).to_csv(
        out/"object_pa_5task_summary.csv",index=False
    )
    print("FIVE_TASK_OBJECT_PA_COMPLETE",flush=True)


def train_loss(pred, d, patches, variant):
    gt = d["gt_scene_flows"].float()
    valid = valid_points(d)

    _, speed = clock(d)
    speed = torch.cat(
        [torch.zeros_like(speed[:, :1]), speed], 1
    )
    weight = (
        1 + .5 * (speed / .02).clamp(max=3)
    ).unsqueeze(-1) * valid

    error = (pred.float() - gt).norm(dim=-1)

    l2 = (
        (error * weight).sum()
        / weight.sum().clamp_min(1)
    )

    last = valid[:, -1]
    endpoint = (
        (error[:, -1] * last).sum()
        / last.sum().clamp_min(1)
    )

    if variant == "speed_l2":
        return l2

    pa, count = patch_loss(pred, gt, patches)
    object_pa = _object_pa_train_loss(pred, d)
    return l2 + .30 * endpoint + .15 * pa + .55 * object_pa


def evaluate(pred, gt, d, patches):
    valid = valid_points(d)
    error = (pred.float() - gt.float()).norm(dim=-1)

    l2 = (
        (error * valid).sum()
        / valid.sum().clamp_min(1)
    )
    last = valid[:, -1]
    endpoint = (
        (error[:, -1] * last).sum()
        / last.sum().clamp_min(1)
    )
    pa, count = patch_loss(pred, gt, patches)

    return [
        float(l2) * 1000,
        float(endpoint) * 1000,
        float(pa) * 1000,
        count
    ]


def run(cfg):
    os.environ["WANDB_MODE"] = "disabled"

    from arguments import parse_args
    from training.trainer import Trainer
    from dataset_components.dataloader import build_dataloader

    # VAGG_EVAL_ONLY_V2
    from diagnostics.vagg_object_hooks_20261011 import install as install_ids
    install_ids()
    print("OBJECT_HOOK_ACTIVE", flush=True)

    root = (
        Path("diagnostics/results/vagg_20261011")
        / cfg.variant
    )
    root.mkdir(parents=True, exist_ok=True)

    ckpt = Path(
        "pretrained_checkpoints/"
        "large-droid+behavior/model-best.pt"
    )

    # FIVE_TASK_WDS_RESTART_20261011
    tasks = ("0000", "0001", "0006", "0008", "0010")
    data_dirs = {
        task: Path(
            "/home/soenchun/cse/PointWorld_workspace/datasets/behavior"
        ).joinpath("wds_task" + task).resolve()
        for task in tasks
    }
    data_dir = data_dirs["0000"]
    assert ckpt.is_file(), "CHECKPOINT NOT FOUND"
    for task, directory in data_dirs.items():
        assert (directory / "train").is_dir(), f"{task} TRAIN MISSING"
        assert (directory / "test").is_dir(), f"{task} TEST MISSING"
    print("TRAIN_TASKS", list(data_dirs), flush=True)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    args = parse_args(skip_command_line=True)
    args.domains = ["droid", "behavior"]
    args.data_dirs = [
        "/home/soenchun/cse/PointWorld_workspace/datasets/PointWorld-DROID-subset_restored/droid/wds",
        str(data_dir)
    ]
    args.model_path = str(ckpt)
    args.seed = 42
    args.batch_size = 1
    args.num_workers = cfg.workers
    args.eval_num_workers = cfg.workers
    args.eval_min_num_cameras = 2
    args.eval_max_num_cameras = 2
    args.train_min_num_cameras = 2
    args.train_max_num_cameras = 2
    args.max_scene_points = 3000
    args.max_robot_points = 500
    args.device = "cuda"
    args.log_dir = str(root / "state")
    args.exp_name = f"vagg_{cfg.variant}_seed42"

    print("LOAD FROZEN POINTWORLD", flush=True)
    trainer = Trainer(args, inference_only=True)
    net = trainer.model

    for p in net.parameters():
        p.requires_grad_(False)

    head = VAGGHead(
        net.dynamics_predictor.dynamics_head,
        net.dynamics_predictor.in_c,
        10,
        gated=(cfg.variant != "ungated")
    ).to(trainer.device)

    net.dynamics_predictor.dynamics_head = head

    if cfg.steps == 0:
        checkpoint = root / "adapter_2000.pt"
        assert checkpoint.is_file(), f"MISSING ADAPTER: {checkpoint}"

        saved = torch.load(
            checkpoint, map_location="cpu", weights_only=True
        )
        assert saved["variant"] == cfg.variant
        assert saved["step"] == 2000

        result = head.load_state_dict(saved["adapter"], strict=False)
        assert not result.unexpected_keys, result
        assert all(
            name.startswith("base.") for name in result.missing_keys
        ), result

        print("EVAL_ONLY_ADAPTER_LOADED", cfg.variant, flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=2e-4
    )


    # Train from all FIVE independent BEHAVIOR task datasets.
    loader_args = copy.copy(trainer.args)
    loader_args.domains = ["behavior"] * len(tasks)
    loader_args.data_dirs = [str(data_dirs[t]) for t in tasks]
    loader_args._eval_override_active = False

    train_loader, _ = build_dataloader(
        loader_args, "train", override_splits="train"
    )
    assert train_loader is not None, "FIVE TASK TRAIN LOADER FAILED"

    # Evaluate each task separately. The official multi-dataset
    # deterministic RoundRobin can terminate at its shortest dataset.
    def five_task_test_batches():
        for task in tasks:
            eval_args = copy.copy(loader_args)
            eval_args.domains = ["behavior"]
            eval_args.data_dirs = [str(data_dirs[task])]
            loader, _ = build_dataloader(
                eval_args, "test", override_splits="test"
            )
            if loader is None:
                raise RuntimeError(f"{task} TEST LOADER MISSING")

            count = 0
            for batch in loader:
                keys = batch["__key__"]
                for key in keys:
                    if _task_from_clip(key) != task:
                        raise RuntimeError(
                            f"TEST TASK MISROUTED: {task} {key}"
                        )
                count += len(keys)
                yield batch
                # FIVE_TASK_BALANCED_100_20261011
                if cfg.steps == 0 and count >= 100:
                    break

            print(
                "TEST_TASK_FINISHED", task,
                "clips", count, flush=True
            )
            if count == 0:
                raise RuntimeError(f"EMPTY TEST TASK {task}")

    test_loader = five_task_test_batches()

    iterator = iter(train_loader)

    for step in range(1, cfg.steps + 1):
        start = time.time()

        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

        batch = {
            k: v.to(trainer.device)
            if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        net.eval()
        head.train()
        head.condition, _ = clock(batch)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_bf16_supported()
        ):
            output = net(batch, training=False)
            patches = rigid_patches(batch, seed=step)

            loss = train_loss(
                output["scene_flows"],
                batch, patches, cfg.variant
            )

        if not torch.isfinite(loss):
            raise RuntimeError("NONFINITE TRAIN LOSS")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in head.parameters() if p.requires_grad],
            1.
        )
        optimizer.step()

        print(
            "TRAIN", cfg.variant,
            "step", step,
            "loss", round(float(loss.detach()), 6),
            "seconds", round(time.time()-start, 2),
            flush=True
        )

        if step % 20 == 0 or step == cfg.steps:
            torch.save(
                {
                    "adapter": {
                        k: v.detach().cpu()
                        for k, v in head.state_dict().items()
                        if not k.startswith("base.")
                    },
                    "step": step,
                    "variant": cfg.variant
                },
                root / f"adapter_{step:04d}.pt"
            )

    print("BEGIN TEST SPLIT EVALUATION", flush=True)

    net.eval()
    head.eval()
    rows = []
    object_rows = []

    with torch.no_grad():
        for batch in test_loader:

            if cfg.steps == 0:
                if "scene_object_id" not in batch:
                    raise RuntimeError(
                        "OBJECT_ID_PIPELINE_MISSING: "
                        + str(sorted(batch.keys()))
                    )
            if len(rows) >= cfg.eval_clips:
                break

            batch = {
                k: v.to(trainer.device)
                if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            head.condition, _ = clock(batch)

            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_bf16_supported()
            ):
                output = net(batch, training=False)

            B, T, N, _ = output["scene_flows"].shape

            base_norm = head.base_output.view(
                B, N, 10, 3
            ).permute(0, 2, 1, 3)

            base_norm = torch.cat(
                [torch.zeros_like(base_norm[:, :1]),
                 base_norm], dim=1
            )

            base = (
                batch["scene_flows"][:, 0:1]
                + net.unnormalize(base_norm)
            )

            for b in range(B):
                one = {
                    k: (
                        v[b:b+1]
                        if isinstance(v, torch.Tensor)
                        and v.ndim > 0
                        and v.shape[0] == B
                        else [v[b]]
                        if isinstance(v, list)
                        and len(v) == B
                        else v
                    )
                    for k, v in batch.items()
                }

                patches = rigid_patches(
                    one, seed=10000+len(rows)
                )
                gt = one["gt_scene_flows"]

                bs = evaluate(
                    base[b:b+1], gt, one, patches
                )
                vs = evaluate(
                    output["scene_flows"][b:b+1],
                    gt, one, patches
                )

                row = {
                    "clip": str(
                        one.get("__key__", [len(rows)])[0]
                    ),
                    "base_l2_mm": bs[0],
                    "vagg_l2_mm": vs[0],
                    "base_endpoint_mm": bs[1],
                    "vagg_endpoint_mm": vs[1],
                    "base_patch_pa_mm": bs[2],
                    "vagg_patch_pa_mm": vs[2],
                    "patch_frames": vs[3]
                }
                object_rows.extend(_evaluate_objects(
                    base[b:b+1],
                    output['scene_flows'][b:b+1],
                    one,row['clip']
                ))
                rows.append(row)
                # VAGG_PARTIAL_PA_SAVE_20261011
                if cfg.steps == 0 and len(rows) % 500 == 0:
                    import pandas as pd
                    pd.DataFrame(object_rows).to_csv(
                        root / "object_pa_partial_frames.csv",
                        index=False
                    )
                    pd.DataFrame(rows).to_csv(
                        root / "test_paired_partial.csv",
                        index=False
                    )
                    print(
                        "PARTIAL_PA_SAVED",
                        cfg.variant, len(rows),
                        len(object_rows),
                        flush=True
                    )


                print(
                    "EVAL", cfg.variant,
                    len(rows), row, flush=True
                )

                if len(rows) >= cfg.eval_clips:
                    break

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(root / "test_paired.csv", index=False)

    if cfg.steps == 0 and cfg.eval_clips == 1:
        print(
            "OBJECT_ID_PREFLIGHT_PASS",
            "clips", len(rows),
            "eligible_object_frames", len(object_rows),
            flush=True
        )
    else:
        _save_object_metrics(object_rows,root)


    for metric in ("l2_mm", "endpoint_mm", "patch_pa_mm"):
        diff = (
            df["base_" + metric]
            - df["vagg_" + metric]
        )
        print(
            "RESULT", metric,
            "median_gain_mm", float(diff.median()),
            "improved_pct",
            float((diff > 0).mean() * 100),
            flush=True
        )

    print(
        "NOTE: patch PA is not official object PA-MPJPE",
        flush=True
    )
    print("ALL_COMPLETE", cfg.variant, flush=True)


def self_test():
    torch.manual_seed(42)
    old = nn.Linear(16, 30)
    head = VAGGHead(old, 16)
    head.condition = torch.randn(2, 10, 3)
    x = torch.randn(2, 30, 16)
    y = head(x)

    assert torch.equal(y, old(x))
    y.square().mean().backward()
    assert head.point[-1].weight.grad is not None

    p = torch.randn(32, 3)
    assert pa_fit(p, p * 1.3 + 4) < 1e-5

    print("SELF_TEST_PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        choices=["vagg", "speed_l2", "ungated"],
        default="vagg"
    )
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--eval-clips", type=int, default=40)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--self-test", action="store_true")
    cfg = ap.parse_args()

    if cfg.self_test:
        self_test()
    else:
        try:
            run(cfg)
        except Exception as exc:
            print("VAGG_FAILED", repr(exc), flush=True)
            traceback.print_exc()
            raise
