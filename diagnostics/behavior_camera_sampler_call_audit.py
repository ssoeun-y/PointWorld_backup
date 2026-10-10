#!/usr/bin/env python3

import dataset_components.dataloader as dlmod
import dataset_components.cameras as cammod

from arguments import parse_args
from evaluation.tester import Tester


TARGET = "task-0000_episode_00002530-300:311"

orig_dl = dlmod.sample_cameras
orig_cam = cammod.sample_cameras

state = {
    "calls": 0,
    "records": [],
}


def keystr(x):
    if isinstance(x, bytes):
        return x.decode()
    return str(x)


def wrapper(
    sample,
    min_num_cameras=1,
    max_num_cameras=None,
    deterministic=False,
    seed=None,
):
    state["calls"] += 1

    key = keystr(
        sample.get("__key__", "")
    )

    prefixes = sorted({
        k.split("_scene_")[0]
        for k in sample.keys()
        if (
            "__" not in k
            and "_scene_flows" in k
            and k != "scene_flows"
        )
    })

    state["records"].append(
        (key, prefixes)
    )

    return orig_dl(
        sample,
        min_num_cameras=min_num_cameras,
        max_num_cameras=max_num_cameras,
        deterministic=deterministic,
        seed=seed,
    )


# Patch BOTH references for audit.
dlmod.sample_cameras = wrapper
cammod.sample_cameras = wrapper


args = parse_args()
tester = Tester(args)

loader, _ = tester._build_eval_loader(
    "test",
    enable_mask=False,
)

post_target_found = False

for batch in loader:
    k = batch["__key__"]

    if isinstance(k, (list, tuple)):
        k = k[0]
    if isinstance(k, bytes):
        k = k.decode()

    if str(k) == TARGET:
        post_target_found = True
        break


# restore
dlmod.sample_cameras = orig_dl
cammod.sample_cameras = orig_cam


print("=" * 80)
print("CAMERA SAMPLER CALL AUDIT")
print("=" * 80)

print("wrapper calls =", state["calls"])
print("post-pipeline target found =", post_target_found)

print()
print("first 10 pre-sampling records:")
for r in state["records"][:10]:
    print(r)

print()
print("records containing episode_00002530:")
matches = [
    r for r in state["records"]
    if "00002530" in r[0]
]

for r in matches:
    print(r)

print()
print("num matching records =", len(matches))
print("NO MODEL INFERENCE = True")
print("OFFICIAL FILES MODIFIED = False")
