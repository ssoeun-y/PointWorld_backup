#!/usr/bin/env python3

import itertools
import numpy as np

from arguments import parse_args
from evaluation.tester import Tester

import dataset_components.dataloader as dlmod
from dataset_components.utils import _stable_int_hash


TARGET_KEY = "task-0000_episode_00002530-300:311"

official_sample_cameras = dlmod.sample_cameras

audit = {}


def key_string(sample):
    k = sample.get("__key__", "")
    if isinstance(k, bytes):
        k = k.decode()
    return str(k)


def audited_sample_cameras(
    sample,
    min_num_cameras=1,
    max_num_cameras=None,
    deterministic=False,
    seed=None,
):
    key = key_string(sample)

    if key == TARGET_KEY:

        # EXACT same discovery rule as official sample_cameras().
        camera_prefixes = set()

        for k in sample.keys():
            if (
                "__" not in k
                and "_scene_flows" in k
                and k != "scene_flows"
            ):
                prefix = k.split("_scene_")[0]
                camera_prefixes.add(prefix)

        camera_prefixes = sorted(
            list(camera_prefixes)
        )

        assert len(camera_prefixes) > 0

        num_cameras = len(camera_prefixes)

        min_cam = (
            1
            if min_num_cameras is None
            else min_num_cameras
        )

        max_cam = (
            num_cameras
            if max_num_cameras is None
            else max_num_cameras
        )

        # EXACT official clipping logic.
        min_cam = min(
            max(int(min_cam), 1),
            num_cameras,
        )

        max_cam = min(
            max(int(max_cam), 1),
            num_cameras,
        )

        max_cam = max(
            max_cam,
            min_cam,
        )

        # Reproduce official deterministic selection
        # WITHOUT changing the actual official call.
        if deterministic:
            base = (
                0
                if seed is None
                else int(seed)
            )

            rs = np.random.RandomState(
                _stable_int_hash(
                    base,
                    key,
                    "cam",
                )
            )

            n_selected = rs.randint(
                min_cam,
                max_cam + 1,
            )

            selected = list(
                rs.choice(
                    camera_prefixes,
                    size=n_selected,
                    replace=False,
                )
            )
        else:
            selected = None

        audit["available"] = camera_prefixes
        audit["min_cam"] = min_cam
        audit["max_cam"] = max_cam
        audit["selected"] = selected

    # Actual pipeline still uses the untouched official function.
    return official_sample_cameras(
        sample,
        min_num_cameras=min_num_cameras,
        max_num_cameras=max_num_cameras,
        deterministic=deterministic,
        seed=seed,
    )


# Runtime diagnostic hook only.
dlmod.sample_cameras = audited_sample_cameras


args = parse_args()
tester = Tester(args)

loader, _ = tester._build_eval_loader(
    "test",
    enable_mask=False,
)


found = False

for batch in loader:

    k = batch["__key__"]

    if isinstance(k, (list, tuple)):
        k = k[0]

    if isinstance(k, bytes):
        k = k.decode()

    if str(k) == TARGET_KEY:
        found = True
        break


# Restore module symbol.
dlmod.sample_cameras = official_sample_cameras

assert found
assert "available" in audit


pairs = list(
    itertools.combinations(
        audit["available"],
        2,
    )
)


print("=" * 72)
print("OFFICIAL PRE-SAMPLING CAMERA INVENTORY")
print("=" * 72)

print("sample =", TARGET_KEY)
print(
    "available cameras =",
    audit["available"],
)
print(
    "num available cameras =",
    len(audit["available"]),
)

print()
print(
    "checkpoint eval range =",
    audit["min_cam"],
    audit["max_cam"],
)

print(
    "official deterministic pair "
    "(seed=runtime seed) =",
    audit["selected"],
)

print()
print(
    "all possible 2-camera pairs =",
    len(pairs),
)

for i, p in enumerate(pairs):
    print(i, p)

print()
print("MODEL INFERENCE = False")
print("OFFICIAL CODE MODIFIED = False")
print("RUNTIME HOOK ONLY = True")
