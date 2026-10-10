#!/usr/bin/env python3

import runpy
import torch
from evaluation.tester import Tester

TARGET = "task-0010_episode_00100090-1025:1036"

def mask3(batch, name):
    x = batch[name].bool()
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x

@torch.no_grad()
def capture(self, dl, split, apply_confidence_mask=False):

    self.model.eval()

    for batch in dl:

        keys = [str(x) for x in batch["__key__"]]

        if TARGET not in keys:
            continue

        i = keys.index(TARGET)

        batch_gpu = {
            k: (
                v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }

        # -------------------------------
        # BEFORE MODEL FORWARD
        # -------------------------------
        input0_before = (
            batch_gpu["scene_flows"][i, 0]
            .detach().clone()
        )

        gt0_before = (
            batch_gpu["gt_scene_flows"][i, 0]
            .detach().clone()
        )

        gt_rel_before = (
            batch_gpu["gt_scene_flows_relative"][i]
            .detach().clone()
        )

        gt_abs_before = (
            batch_gpu["gt_scene_flows"][i]
            .detach().clone()
        )

        print(
            "GT_REL_CONTRACT_BEFORE_MAX_MM =",
            torch.linalg.norm(
                gt_rel_before
                - (gt_abs_before - gt_abs_before[0:1]),
                dim=-1
            ).max().item() * 1000
        )

        print("===== BEFORE FORWARD =====")
        print(
            "input_t0_vs_gt_t0_max_m =",
            torch.linalg.norm(
                input0_before - gt0_before,
                dim=-1
            ).max().item()
        )

        outputs = self.model(
            batch_gpu,
            training=False,
        )

        print(
            "GT_ABS_MUTATION_MAX_MM =",
            torch.linalg.norm(
                batch_gpu["gt_scene_flows"][i]
                - gt_abs_before,
                dim=-1
            ).max().item() * 1000
        )

        # -------------------------------
        # AFTER MODEL FORWARD
        # -------------------------------
        input0_after = batch_gpu["scene_flows"][i, 0]

        print("\n===== AFTER FORWARD =====")
        print(
            "input_t0_vs_gt_t0_max_m =",
            torch.linalg.norm(
                input0_after - gt0_before,
                dim=-1
            ).max().item()
        )

        print(
            "input_t0_mutation_max_m =",
            torch.linalg.norm(
                input0_after - input0_before,
                dim=-1
            ).max().item()
        )

        pred_abs = outputs["scene_flows"][i]
        pred_rel = outputs["scene_relative"][i]

        gt_abs = batch_gpu["gt_scene_flows"][i]
        gt_rel = batch_gpu["gt_scene_flows_relative"][i]

        # official helper가 forward 중 변했는지도 확인
        print(
            "gt_relative_mutation_max_m =",
            torch.linalg.norm(
                gt_rel - gt_rel_before,
                dim=-1
            ).max().item()
        )

        exists = mask3(batch_gpu, "scene_exists")
        context = mask3(batch_gpu, "scene_context_mask")
        supervised = mask3(batch_gpu, "scene_supervised_mask")
        moved = mask3(batch_gpu, "scene_moved_mask")

        valid = (~context) & exists & supervised

        t = 10
        m = valid[i, t] & moved[i, t]

        print("\n===== T10 MOVED SUPPORT =====")
        print("n_moved =", int(m.sum()))

        abs_err = torch.linalg.norm(
            pred_abs[t][m] - gt_abs[t][m],
            dim=-1
        )

        rel_err = torch.linalg.norm(
            pred_rel[t][m] - gt_rel[t][m],
            dim=-1
        )

        print(
            "absolute_EPE_mm =",
            abs_err.mean().item() * 1000
        )

        print(
            "relative_EPE_mm =",
            rel_err.mean().item() * 1000
        )

        print(
            "EPE_difference_mm =",
            (abs_err.mean() - rel_err.mean()).item() * 1000
        )

        # algebraic identity 직접 확인
        identity = (
            (pred_abs[t][m] - gt_abs[t][m])
            -
            (pred_rel[t][m] - gt_rel[t][m])
        )

        print(
            "error_vector_identity_max_mm =",
            torch.linalg.norm(
                identity, dim=-1
            ).max().item() * 1000
        )

        print("\nSUCCESS = True")

        return {
            f"diagnostic/{split}/t0_contract_audit": 1.0
        }

    raise RuntimeError("TARGET not found")


Tester._accumulate_metrics = capture

runpy.run_path(
    "eval.py",
    run_name="__main__",
)
