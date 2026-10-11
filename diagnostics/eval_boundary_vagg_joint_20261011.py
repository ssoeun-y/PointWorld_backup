#!/usr/bin/env python3
"""Matched 4-way *shape-only* intervention audit (not counterfactual GT trajectory).
Boundary original/retimed robot actions crossed with PointWorld/VAGG generation.
Reuses validated boundary WDS builder/releaser and VAGG mesh-level PA metric.
"""
import argparse
import copy
import os
import random
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diagnostics import eval_acc_only_pa_intervention_5task as old
from diagnostics import eval_acc_only_boundary_pa_final as boundary
from diagnostics import vagg_train_20261011 as vagg

TASKS = ("0000", "0001", "0006", "0008", "0010")
OUT = Path("diagnostics/results/boundary_vagg_joint_20261011")
VROOT = Path("diagnostics/results/vagg_20261011/vagg")


def batch_slice(d, b):
    return {k: (v[b:b+1] if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == 2
                else [v[b]] if isinstance(v, list) and len(v) == 2 else v)
            for k, v in d.items()}



def align_object_ids(batch, raw):
    """Propagate mesh IDs through release sampling via geometric correspondence.
    Never truncate or resize IDs by position; reject an unverifiable mapping.
    """
    from scipy.spatial import cKDTree

    def to_np(v):
        return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

    gt = to_np(batch["gt_scene_flows"])
    ids = to_np(batch["scene_object_id"])
    if ids.ndim == 4 and ids.shape[-1] == 1:
        ids = ids[..., 0]
    B, T, N, _ = gt.shape
    if ids.shape == (B, T, N):
        return
    raw_ids = to_np(raw["scene_object_id"])
    if raw_ids.ndim == 3 and raw_ids.shape[-1] == 1:
        raw_ids = raw_ids[..., 0]
    if raw_ids.shape != (T, ids.shape[-1]):
        raise RuntimeError(f"UNMAPPABLE RAW ID: raw={raw_ids.shape}, batched={ids.shape}, GT={gt.shape}")

    coords = None
    for name in ("gt_scene_flows", "scene_flows"):
        x = raw.get(name)
        if x is not None:
            x = to_np(x)
            if x.shape == (T, raw_ids.shape[1], 3):
                coords = x
                break
    if coords is None:
        raise RuntimeError("NO RAW SCENE COORDINATES TO ALIGN IDS")

    exists = to_np(batch["scene_exists"])[0]
    if exists.ndim == 3 and exists.shape[-1] == 1:
        exists = exists[..., 0]
    if exists.shape != (T, N):
        raise RuntimeError(f"SCENE EXISTS SHAPE INVALID: {exists.shape}")
    aligned = np.full((T, N), -1, dtype=raw_ids.dtype)
    for t in range(T):
        src = coords[t]
        target = gt[0, t]
        finite_src = np.isfinite(src).all(axis=-1)
        valid = (exists[t] > 0) & np.isfinite(target).all(axis=-1)
        if not valid.any():
            continue
        tree = cKDTree(src[finite_src])
        distance, ix = tree.query(target[valid], k=1)
        if not np.isfinite(distance).all() or np.max(distance) > 1e-4:
            raise RuntimeError(
                f"OBJECT ID COORDINATE MISMATCH frame={t} max_distance_m={np.max(distance):.6g} "
                "(refusing approximate object label assignment)")
        raw_ix = np.flatnonzero(finite_src)[ix]
        aligned[t, valid] = raw_ids[t, raw_ix]

    new_ids = torch.as_tensor(aligned, dtype=batch["scene_object_id"].dtype,
                              device=batch["scene_object_id"].device)
    batch["scene_object_id"] = new_ids.unsqueeze(0).expand(B, -1, -1).clone()
    if batch["scene_object_id"].shape != (B, T, N):
        raise RuntimeError("ALIGNED OBJECT ID SHAPE ERROR")
    print("OBJECT_IDS_ALIGNED", tuple(ids.shape), "->", tuple(batch["scene_object_id"].shape), flush=True)

def check_same_scene(d):
    for k in ("scene_flows", "gt_scene_flows", "scene_exists",
              "scene_context_mask", "scene_supervised_mask", "scene_moved_mask",
              "scene_object_id"):
        if k not in d:
            raise RuntimeError(f"MISSING FIELD {k}")
        x = d[k]
        if not isinstance(x, torch.Tensor) or x.shape[0] != 2:
            raise RuntimeError(f"BAD SHAPE {k}: {getattr(x,'shape',None)}")
        same = (torch.allclose(x[0],x[1],atol=0,rtol=0,equal_nan=True)
                if x.is_floating_point() else torch.equal(x[0],x[1]))
        if not same:
            raise RuntimeError(f"SCENE/GT NOT MATCHED: {k} (invalid paired comparison)")


def load_model(device):
    args = old.parse_args(skip_command_line=True)
    args.seed = 42
    args.model_path = old.MODEL_PATH
    args.norm_stats_path = old.NORM_STATS
    args.domains = ["behavior"]
    args.data_dirs = [old.TASK_DIRS["task-0000"]]
    args.batch_size = 2
    args.num_workers = 0
    args.eval_num_workers = 0
    args.eval_num_batches = -1
    args.device = "cuda"
    args.eval_skip_viz = True
    args.eval_min_num_cameras = 2
    args.eval_max_num_cameras = 2
    args.max_scene_points = 3000
    args.max_robot_points = 500
    print("INITIALIZING FROZEN POINTWORLD", flush=True)
    tester = old.Tester(args)
    net = tester.model
    net.eval()
    head = vagg.VAGGHead(net.dynamics_predictor.dynamics_head,
                         net.dynamics_predictor.in_c, 10, gated=True).to(tester.device)
    net.dynamics_predictor.dynamics_head = head
    p = VROOT / "adapter_2000.pt"
    saved = torch.load(p, map_location="cpu", weights_only=True)
    assert saved["variant"] == "vagg" and saved["step"] == 2000
    loaded = head.load_state_dict(saved["adapter"], strict=False)
    assert not loaded.unexpected_keys, loaded
    assert all(k.startswith("base.") for k in loaded.missing_keys), loaded
    net.eval(); head.eval()
    old.omap.pmod.grid_sample_transform = old.omap.grid_sample_with_object_ids
    old.omap.pmod.gather = old.omap.gather_with_object_ids
    old.reset_seed(42)
    print("CHECKPOINT_LOADED", p, "DEVICE", tester.device, flush=True)
    return args, tester, net, head


def summarize(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "object_frames.csv", index=False)
    metrics = ["pointworld", "boundary", "vagg", "combined"]
    cl = df.groupby(["task", "clip"], as_index=False)[[m+"_pa_mm" for m in metrics]].mean()
    cl.to_csv(OUT / "per_clip.csv", index=False)
    rng = np.random.default_rng(42)
    report = []
    for t in list(TASKS) + ["ALL"]:
        q = cl if t == "ALL" else cl[cl.task == t]
        if q.empty: continue
        rec = {"task": t, "clips": len(q)}
        for m in metrics:
            rec[m+"_median_pa_mm"] = float(q[m+"_pa_mm"].median())
        for left, right, tag in [
            ("pointworld", "boundary", "boundary_gain"),
            ("pointworld", "vagg", "vagg_gain"),
            ("boundary", "combined", "combined_over_boundary"),
            ("vagg", "combined", "combined_over_vagg")]:
            gain = q[left+"_pa_mm"] - q[right+"_pa_mm"]
            rec[tag+"_paired_median_mm"] = float(gain.median())
            rec[tag+"_better_pct"] = float((gain > 0).mean()*100)
        denom = rec["boundary_median_pa_mm"]
        rec["combined_vs_boundary_median_reduction_pct"] = float(
            100*(denom-rec["combined_median_pa_mm"])/denom) if denom else float("nan")
        episodes = q["clip"].str.extract(r"^(task-\d{4}_episode_\d+)")[0].fillna(q["clip"])
        groups = [g.to_numpy() for _,g in (q.boundary_pa_mm-q.combined_pa_mm).groupby(episodes)]
        rec["episodes"] = len(groups)
        if len(groups) > 1:
            bs = []
            for _ in range(1000):
                sel = rng.integers(0,len(groups),len(groups))
                bs.append(np.median(np.concatenate([groups[i] for i in sel])))
            rec["combined_gain_boot_ci_low_mm"],rec["combined_gain_boot_ci_high_mm"] = [float(x) for x in np.percentile(bs,[2.5,97.5])]
        report.append(rec)
        print("RESULT", rec, flush=True)
    pd.DataFrame(report).to_csv(OUT / "summary.csv",index=False)
    print("SAVED", OUT / "summary.csv", "CLIPS", len(cl), flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-per-task",type=int,default=0)
    cfg=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    all_vagg = pd.read_csv(VROOT/"object_pa_per_clip.csv")
    assert len(all_vagg) >= 500
    v_keys = set(all_vagg["clip"].astype(str))
    targets={}
    for task in TASKS:
        eligible = boundary.successful_corrections("task-"+task)
        targets[task]=set(eligible.index.astype(str)) & v_keys
    print("MATCHED_SUCCESS_TARGETS", {k:len(v) for k,v in targets.items()},flush=True)
    assert all(targets.values()), "MISSING TASK MATCH"
    args, tester, net, head = load_model(os.environ.get("CUDA_VISIBLE_DEVICES",""))
    rows=[]
    robot_sampler = old.TorchRobotSampler(
        urdf_path=old.resolve_robot_urdf("behavior"),
        gripper_only=old.RELEASE_GRIPPER_ONLY,device="cpu")
    for task in TASKS:
        if cfg.smoke and task != "0000": break
        found=set()
        shard_paths=old.dlmod.gather_shard_paths(old.TASK_DIRS["task-"+task],"test",rank=tester.rank)
        assert shard_paths, f"NO SHARDS {task}"
        random.Random(42+tester.rank).shuffle(shard_paths)
        dataset=old.wds.WebDataset(shard_paths,shardshuffle=False,detshuffle=True,
            resampled=False,nodesplitter=(old.wds.split_by_node if tester.world_size>1 else None),
            seed=42+tester.rank,handler=old.wds.warn_and_continue)
        print("BEGIN_TASK",task,"TARGETS",len(targets[task]),flush=True)
        with torch.inference_mode():
            for raw in dataset:
                key=str(raw.get("__key__",""))
                if key not in targets[task]: continue
                if key in found: raise RuntimeError(f"DUPLICATE {key}")
                decoded=old.decode_data(raw,domain="behavior")
                assert str(decoded.get("__key__",key))==key
                built=old.omap.build_flow_sample_with_object_ids(
                    decoded,domain="behavior",robot_sampler=robot_sampler,
                    max_robot_points=args.max_robot_points,
                    deterministic=True,seed=42+tester.rank,force_single_arm=False)
                T=int(built["robot_flows"].shape[0])
                phase=boundary.load_phase(key,"task-"+task)["phase"]
                assert len(phase)==T and abs(float(phase[0]))<1e-7 and abs(float(phase[-1])-(T-1))<1e-7
                base=old.src.apply_phase(copy.deepcopy(built),np.linspace(0,1,T))
                corrected=old.src.apply_phase(copy.deepcopy(built),phase/(T-1))
                for ind in (0,-1):
                    assert np.allclose(base["robot_flows"][ind], corrected["robot_flows"][ind],rtol=0,atol=1e-7)
                # Identical sampling RNG for original and corrected actions.
                st_py, st_np, st_torch = random.getstate(), np.random.get_state(), torch.get_rng_state()
                base_rel=old.release_process(copy.deepcopy(base),args,"behavior",tester.rank)
                random.setstate(st_py); np.random.set_state(st_np); torch.set_rng_state(st_torch)
                corr_rel=old.release_process(copy.deepcopy(corrected),args,"behavior",tester.rank)
                d=old.custom_collate_fn([base_rel,corr_rel],args=args)
                align_object_ids(d,built)
                check_same_scene(d)
                d=old.to_device(d,tester.device)
                head.condition,_=vagg.clock(d)
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                    out=net(d,training=False)
                B,TF,N,_=out["scene_flows"].shape
                assert B==2 and TF==T
                bn=head.base_output.reshape(2,N,T-1,3).permute(0,2,1,3)
                bn=torch.cat([torch.zeros_like(bn[:,:1]),bn],dim=1)
                orig=d["scene_flows"][:,0:1]+net.unnormalize(bn)
                mod=out["scene_flows"]
                r0=vagg._evaluate_objects(orig[:1],mod[:1],batch_slice(d,0),key)
                r1=vagg._evaluate_objects(orig[1:],mod[1:],batch_slice(d,1),key)
                ix=lambda z:(int(z["object_id"]),int(z["frame"]),int(z["n_points"]))
                m0={ix(z):z for z in r0};m1={ix(z):z for z in r1}
                assert m0 and m0.keys()==m1.keys(),f"OBJECT MATCH FAIL {key}: {len(m0)} != {len(m1)}"
                for ik in m0:
                    a,b=m0[ik],m1[ik]
                    rows.append({"task":task,"clip":key,"object_id":ik[0],"frame":ik[1],
                        "n_points":ik[2],"pointworld_pa_mm":a["base_pa_mpjpe_mm"],
                        "boundary_pa_mm":b["base_pa_mpjpe_mm"],
                        "vagg_pa_mm":a["vagg_pa_mpjpe_mm"],
                        "combined_pa_mm":b["vagg_pa_mpjpe_mm"]})
                found.add(key)
                print("CLIP_DONE",task,len(found),"/",len(targets[task]),key,flush=True)
                if cfg.smoke or (cfg.max_per_task and len(found)>=cfg.max_per_task): break
                if len(found)==len(targets[task]):break
        print("TASK_FINISHED",task,"clips",len(found),"missing",len(targets[task]-found),flush=True)
        summarize(rows)
        if not cfg.smoke and not cfg.max_per_task and targets[task]-found:
            raise RuntimeError(f"MISSING MATCHED CLIPS {task}: {len(targets[task]-found)}")
    print("JOINT_EVAL_COMPLETE",flush=True)

if __name__=="__main__":
    try: main()
    except Exception:
        print("JOINT_EVAL_FAILED",flush=True)
        traceback.print_exc()
        raise
