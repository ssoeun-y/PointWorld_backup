import os, sys, hashlib, csv
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.argv = [
    "eval.py",
    "--model_path=pretrained_checkpoints/large-droid+behavior/model-best.pt",
    "--domains=droid",
    "--data_dirs=/home/soenchun/cse/PointWorld_workspace/datasets/PointWorld-DROID-subset_restored/droid/wds",
    "--batch_size=1",
    "--num_workers=0",
    "--eval_num_workers=0",
    "--eval_num_batches=-1",
    "--eval_skip_viz=true",
]

from arguments import parse_args
from evaluation.tester import Tester

OUT = ROOT / "diagnostics/results/droid_object_eval_cache_20261010"
OUT.mkdir(parents=True, exist_ok=True)

args = parse_args()
args.eval_min_num_cameras = 2
args.eval_max_num_cameras = 2

tester = Tester(args)
tester.model.eval()
loader, _ = tester._build_eval_loader("test", enable_mask=False)

def array(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def mask(x, T, N):
    a = array(x[0]).astype(bool)
    if a.shape == (T, N, 1):
        a = a[..., 0]
    assert a.shape == (T, N), a.shape
    return a

manifest = OUT / "manifest.csv"
written = 0
skipped = 0
seen = set()

with manifest.open("a", newline="") as f:
    writer = csv.writer(f)
    if manifest.stat().st_size == 0:
        writer.writerow(["clip", "file", "n_points", "valid_future_points"])

    with torch.no_grad():
        for batch in loader:
            key = str(batch["__key__"][0])
            assert key not in seen, f"Duplicate clip: {key}"
            seen.add(key)

            digest = hashlib.sha256(key.encode()).hexdigest()[:24]
            dest = OUT / f"{digest}.npz"

            if dest.exists():
                skipped += 1
                continue

            gpu_batch = {
                k: v.to(tester.device, non_blocking=True)
                if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            outputs = tester.model(gpu_batch, training=False)
            pred = array(outputs["scene_flows"][0]).astype(np.float32)
            gt = array(batch["gt_scene_flows"][0]).astype(np.float32)

            assert pred.shape == gt.shape
            T, N, D = gt.shape
            assert T == 11 and D == 3, (key, gt.shape)
            assert np.isfinite(pred).all() and np.isfinite(gt).all()

            valid = (
                ~mask(batch["scene_context_mask"], T, N)
                & mask(batch["scene_exists"], T, N)
                & mask(batch["scene_supervised_mask"], T, N)
            )

            save = {
                "clip": np.array(key),
                "pred_xyz": pred,
                "gt_xyz": gt,
                "valid": valid,
                "moved": mask(batch["scene_moved_mask"], T, N),
                "visibility": mask(batch["scene_visibility"], T, N),
                "depth_valid": mask(batch["scene_depth_valid_mask"], T, N),
                "robot_xyz": array(batch["robot_flows"][0]).astype(np.float32),
                "gripper_pose": array(batch["right_gripper_pose"][0]).astype(np.float32),
                "gripper_open": array(batch["right_gripper_open"][0]).astype(np.float32),
            }

            for k, v in batch.items():
                if k.startswith(("cam0_", "cam1_")) and k.endswith(
                    ("_initial_rgb", "_intrinsic", "_extrinsic")
                ) and isinstance(v, torch.Tensor):
                    save[k] = array(v[0])

            tmp = OUT / f"{digest}.part.npz"
            np.savez_compressed(tmp, **save)
            os.replace(tmp, dest)

            writer.writerow([key, dest.name, N, int(valid.sum())])
            f.flush()
            written += 1

            if written % 10 == 0:
                print(
                    f"DROID_CACHE saved={written} skipped={skipped} "
                    f"seen={len(seen)} points={N}",
                    flush=True,
                )

            del outputs, gpu_batch

assert len(seen) == 1679, f"Expected 1679, got {len(seen)}"
print("DROID_CACHE_COMPLETE", len(seen), flush=True)
