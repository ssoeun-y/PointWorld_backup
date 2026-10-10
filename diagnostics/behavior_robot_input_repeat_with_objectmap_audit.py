#!/usr/bin/env python3

import random
import numpy as np
import torch

from arguments import parse_args
from evaluation.tester import Tester

import dataset_components.dataloader as dlmod
import dataset_components.decoders as decmod
import dataset_components.pipeline as pmod
import diagnostics.behavior_object_mapping_smoke as objmap

# Same runtime-only object-ID preservation path used by camera-pair experiment.
dlmod.build_flow_sample = objmap.build_flow_sample_with_object_ids
decmod.build_flow_sample = objmap.build_flow_sample_with_object_ids
pmod.grid_sample_transform = objmap.grid_sample_with_object_ids
pmod.gather = objmap.gather_with_object_ids


KEY = "task-0000_episode_00002530-300:311"


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_key(batch):
    k = batch["__key__"]
    if isinstance(k, (list, tuple)):
        k = k[0]
    if isinstance(k, bytes):
        k = k.decode()
    return str(k)


args = parse_args()
assert args.num_workers == 0

tester = Tester(args)

captures = []

for rep in range(3):

    # Same already-validated RNG reset protocol.
    reset_rng(args.seed)

    loader, _ = tester._build_eval_loader(
        "test",
        enable_mask=False,
    )

    target = None

    for batch in loader:
        if get_key(batch) == KEY:
            target = batch
            break

    assert target is not None

    captures.append({
        name:
            target[name].detach().cpu().numpy().copy()
        for name in [
            "robot_flows",
            "robot_features",
            "robot_exists",
        ]
    })


print("=" * 72)
print("ROBOT INPUT REPEATABILITY AUDIT")
print("=" * 72)

for name in captures[0]:

    print()
    print(name)

    for i in [1, 2]:

        a = captures[0][name]
        b = captures[i][name]

        print(
            f"repeat0 vs repeat{i}:",
            "exact =",
            np.array_equal(a, b),
            "max_abs_diff =",
            float(
                np.max(
                    np.abs(
                        a.astype(float)
                        - b.astype(float)
                    )
                )
            )
            if a.dtype != bool
            else "NA",
        )

print()
print("CAMERA INTERVENTION = False")
print("MODEL INFERENCE = False")
print("OFFICIAL CODE MODIFIED = False")
