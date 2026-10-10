import os
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from arguments import parse_args
from evaluation.tester import Tester

import dataset_components.dataloader as dl
from dataset_components.cameras import sample_cameras as official_sample_cameras


PAIR = os.environ.get("PW_CAMERA_PAIR", "").upper()

PAIRS = {
    "HL": ("camera_head", "camera_left"),
    "HR": ("camera_head", "camera_right"),
    "LR": ("camera_left", "camera_right"),
}

if PAIR not in PAIRS:
    raise RuntimeError(
        "PW_CAMERA_PAIR must be one of: HL, HR, LR"
    )

WANTED = PAIRS[PAIR]

OUTDIR = Path(
    "diagnostics/results/"
    "action_conditioned_active_world_model/"
    "fixed_camera_pair_task0008"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Fixed 2-camera selector
# ============================================================
def fixed_pair_selector(
    sample,
    min_num_cameras=1,
    max_num_cameras=None,
    deterministic=False,
    seed=None,
):
    prefixes = sorted({
        key.split("_scene_")[0]
        for key in sample.keys()
        if "__" not in key
        and "_scene_flows" in key
        and key != "scene_flows"
    })

    missing = [
        p for p in WANTED
        if p not in prefixes
    ]

    if missing:
        raise RuntimeError(
            f"{sample.get('__key__')} missing cameras "
            f"{missing}; available={prefixes}"
        )

    drop = set(prefixes) - set(WANTED)

    # Remove the third camera before calling the official sampler.
    for key in list(sample.keys()):
        if any(
            key.startswith(prefix + "_")
            for prefix in drop
        ):
            sample.pop(key, None)

    # Exactly two remain.
    # Official PointWorld code performs scene concatenation and
    # standardizes them to cam0/cam1.
    return official_sample_cameras(
        sample,
        min_num_cameras=2,
        max_num_cameras=2,
        deterministic=True,
        seed=seed,
    )


# build_dataloader imported sample_cameras into this module,
# so replace only its diagnostics-run global.
dl.sample_cameras = fixed_pair_selector


# ============================================================
# Tester that also preserves sample-level official metrics
# ============================================================
class PerSampleTester(Tester):

    @torch.no_grad()
    def _accumulate_metrics(
        self,
        dataloader,
        split,
        apply_confidence_mask=False,
    ):
        if apply_confidence_mask:
            raise RuntimeError(
                "This BEHAVIOR diagnostic should not use "
                "a DROID confidence mask."
            )

        self.model.eval()

        rows = []
        values = defaultdict(list)

        total = (
            self.args.eval_num_batches
            if self.args.eval_num_batches > 0
            else None
        )

        iterator = tqdm(
            dataloader,
            desc=f"Eval [{PAIR}] [{split}]",
            total=total,
        )

        for i, batch in enumerate(iterator):

            if total is not None and i >= total:
                break

            batch = {
                k: (
                    v.to(
                        self.device,
                        non_blocking=True,
                    )
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in batch.items()
            }

            outputs = self.model(
                batch,
                training=False,
            )

            _, loss_dict = self.model.loss_fn(
                outputs,
                batch,
                training=False,
            )

            keys = [
                str(x)
                for x in batch["__key__"]
            ]

            if len(keys) != 1:
                raise RuntimeError(
                    "Use --batch_size 1 for this diagnostic"
                )

            row = {
                "clip": keys[0],
                "camera_pair": PAIR,
                "camera_1": WANTED[0],
                "camera_2": WANTED[1],
            }

            # Keep useful metadata if collate retained it.
            for name in [
                "__has_gripper_state_change__",
                "__gripper_moving__",
                "__robot_nonbase_moving__",
                "__has_left_gripper_finger_collision__",
                "__has_right_gripper_finger_collision__",
                "__left_min_distance_to_moving_objects__",
                "__right_min_distance_to_moving_objects__",
            ]:
                if name not in batch:
                    continue

                v = batch[name]

                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        row[name] = v.item()
                elif isinstance(v, (list, tuple)):
                    if len(v) == 1:
                        row[name] = v[0]
                elif isinstance(v, (bool, int, float, str)):
                    row[name] = v

            for metric_name, v in loss_dict.items():

                val = None

                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        x = float(v.item())
                        if math.isfinite(x):
                            val = x

                elif isinstance(v, (int, float)):
                    x = float(v)
                    if math.isfinite(x):
                        val = x

                if val is None:
                    continue

                row[metric_name] = val
                values[metric_name].append(val)

            rows.append(row)

        df = pd.DataFrame(rows)

        out = (
            OUTDIR
            / f"task0008_pair_{PAIR}"
              f"__per_sample.csv"
        )

        df.to_csv(
            out,
            index=False,
        )

        print("\n" + "=" * 100)
        print("FIXED CAMERA PAIR RESULT")
        print("=" * 100)
        print("PAIR =", PAIR, WANTED)
        print("ROWS =", len(df))
        print("OUTPUT =", out)

        interesting = [
            c for c in df.columns
            if "l2_moved" in c
            or "filtered_l2" in c
        ]

        print("\nERROR COLUMNS:")
        for c in interesting:
            print(
                c,
                "| median =",
                df[c].median(),
                "| mean =",
                df[c].mean(),
            )

        print("SUCCESS = True")

        # Tester.run_evaluation expects an aggregate dict.
        agg = {}

        for metric_name, vals in values.items():
            if vals:
                agg[
                    f"full_eval/{split}/{metric_name}"
                ] = float(np.mean(vals))

        return agg


if __name__ == "__main__":
    print("=" * 100)
    print(
        "BEHAVIOR FIXED CAMERA PAIR =",
        PAIR,
        WANTED,
    )
    print("=" * 100)

    args = parse_args()

    tester = PerSampleTester(args)
    tester.run_evaluation()
