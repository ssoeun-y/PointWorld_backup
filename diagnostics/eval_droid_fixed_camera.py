import os

from arguments import parse_args
from evaluation.tester import Tester

import dataset_components.dataloader as dl
from dataset_components.cameras import sample_cameras as official_sample_cameras


MODE = os.environ.get("PW_CAMERA_MODE", "").upper()

if MODE not in {"A", "B", "AB"}:
    raise RuntimeError(
        "Set PW_CAMERA_MODE to A, B, or AB"
    )


def fixed_camera_selector(
    sample,
    min_num_cameras=1,
    max_num_cameras=None,
    deterministic=False,
    seed=None,
):
    prefixes = sorted({
        k.split("_scene_")[0]
        for k in sample.keys()
        if "__" not in k
        and "_scene_flows" in k
        and k != "scene_flows"
    })

    if len(prefixes) < 2:
        raise RuntimeError(
            f"Expected >=2 cameras, got {prefixes}"
        )

    if MODE == "A":
        keep = {prefixes[0]}
    elif MODE == "B":
        keep = {prefixes[1]}
    else:
        keep = set(prefixes[:2])

    drop = set(prefixes) - keep

    image_suffixes = (
        "_initial_rgb",
        "_initial_depth",
        "_intrinsic",
        "_extrinsic",
    )

    for k in list(sample.keys()):
        remove = False

        for p in drop:
            if k.startswith(p + "_scene_"):
                remove = True
                break

            if any(
                k == p + suffix
                for suffix in image_suffixes
            ):
                remove = True
                break

        if remove:
            sample.pop(k, None)

    n = len(keep)

    return official_sample_cameras(
        sample,
        min_num_cameras=n,
        max_num_cameras=n,
        deterministic=True,
        seed=seed,
    )


# build_dataset()가 참조하는 global symbol만 diagnostics run에서 교체.
# 원본 dataset_components/cameras.py는 수정하지 않는다.
dl.sample_cameras = fixed_camera_selector


if __name__ == "__main__":
    print("=" * 90)
    print("FIXED DROID CAMERA CONDITION =", MODE)
    print("=" * 90)

    args = parse_args()
    tester = Tester(args)
    tester.run_evaluation()
