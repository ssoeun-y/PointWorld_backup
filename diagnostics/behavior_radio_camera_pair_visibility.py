#!/usr/bin/env python3

from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch

from arguments import parse_args
from evaluation.tester import Tester

import dataset_components.dataloader as dlmod
import dataset_components.cameras as cammod
import dataset_components.decoders as decmod
import dataset_components.pipeline as pmod

import diagnostics.behavior_object_mapping_smoke as objmap

from dataset_components.utils import _stable_int_hash
from diagnostics.droid_visibility_quickcheck import get_visibility_count


TARGET_KEY = "task-0000_episode_00002530-300:311"

CAMERAS = [
    "camera_head",
    "camera_left",
    "camera_right",
]

CONDITIONS = [
    {
        "name": "head_left",
        "camera_seed": 0,
        "expected": ("camera_head", "camera_left"),
    },
    {
        "name": "head_right",
        "camera_seed": 2,
        "expected": ("camera_head", "camera_right"),
    },
    {
        "name": "left_right",
        "camera_seed": 1,
        "expected": ("camera_left", "camera_right"),
    },
]

INVENTORY = Path(
    "diagnostics/results/"
    "behavior_object_distance_motion_full578_20260810_114615/"
    "per_object.csv"
)

OUTDIR = Path(
    "diagnostics/results/"
    "behavior_radio_camera_pair_visibility"
)


# ------------------------------------------------------------
# Existing validated object-ID preservation path.
# ------------------------------------------------------------

orig_dl_build = dlmod.build_flow_sample
orig_dec_build = decmod.build_flow_sample
orig_grid = pmod.grid_sample_transform
orig_gather = pmod.gather

dlmod.build_flow_sample = objmap.build_flow_sample_with_object_ids
decmod.build_flow_sample = objmap.build_flow_sample_with_object_ids
pmod.grid_sample_transform = objmap.grid_sample_with_object_ids
pmod.gather = objmap.gather_with_object_ids


official_sample_cameras = dlmod.sample_cameras
official_cam_sample_cameras = cammod.sample_cameras


def keystr(x):
    if isinstance(x, bytes):
        x = x.decode()
    return str(x)


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Existing inventory determines radio object ID.
# ------------------------------------------------------------

inv = pd.read_csv(INVENTORY)

radio_row = inv[
    (inv["key"] == TARGET_KEY)
    &
    inv["object_name"].str.contains(
        "__radio_",
        regex=False,
    )
]

assert len(radio_row) == 1

RADIO_ID = int(
    radio_row.iloc[0]["object_id"]
)

print("radio object_id =", RADIO_ID)


args = parse_args()

# Important: _build_eval_loader internally uses args.num_workers.
assert args.num_workers == 0, (
    "Run with --num_workers=0 for runtime camera hook audit."
)

tester = Tester(args)
tester.model.eval()


results = []
reference_robot = None


for condition in CONDITIONS:

    reset_rng(args.seed)   # global/model/robot sampling 고정

    camera_seed = int(
        condition["camera_seed"]
    )

    expected = tuple(
        sorted(
            condition["expected"]
        )
    )

    state = {
        "target_seen": False,
        "selected": None,
    }


    def camera_wrapper(
        sample,
        min_num_cameras=1,
        max_num_cameras=None,
        deterministic=False,
        seed=None,
    ):
        key = keystr(
            sample.get("__key__", "")
        )

        if key == TARGET_KEY:

            assert deterministic is True

            prefixes = sorted({
                k.split("_scene_")[0]
                for k in sample.keys()
                if (
                    "__" not in k
                    and "_scene_flows" in k
                    and k != "scene_flows"
                )
            })

            assert prefixes == CAMERAS, prefixes

            # Reproduce exact official camera selection.
            rs = np.random.RandomState(
                _stable_int_hash(
                    camera_seed,
                    key,
                    "cam",
                )
            )

            n = rs.randint(
                min_num_cameras,
                max_num_cameras + 1,
            )

            selected = list(
                rs.choice(
                    prefixes,
                    size=n,
                    replace=False,
                )
            )

            assert tuple(
                sorted(selected)
            ) == expected, (
                condition,
                selected,
            )

            state["target_seen"] = True
            state["selected"] = selected

            # Only camera sampler seed changes.
            # The official implementation is still used.
            return official_sample_cameras(
                sample,
                min_num_cameras=min_num_cameras,
                max_num_cameras=max_num_cameras,
                deterministic=deterministic,
                seed=camera_seed,
            )

        return official_sample_cameras(
            sample,
            min_num_cameras=min_num_cameras,
            max_num_cameras=max_num_cameras,
            deterministic=deterministic,
            seed=seed,
        )


    dlmod.sample_cameras = camera_wrapper
    cammod.sample_cameras = camera_wrapper


    loader, _ = tester._build_eval_loader(
        "test",
        enable_mask=False,
    )

    selected_batch = None

    for batch in loader:

        k = batch["__key__"]

        if isinstance(k, (list, tuple)):
            k = k[0]

        if isinstance(k, bytes):
            k = k.decode()

        if str(k) == TARGET_KEY:
            selected_batch = batch
            break


    assert selected_batch is not None
    assert state["target_seen"]
    assert state["selected"] is not None


    # --------------------------------------------------------
    # Action / robot conditioning must remain identical.
    # --------------------------------------------------------

    robot_snapshot = {}

    for name in [
        "robot_flows",
        "robot_features",
        "robot_exists",
    ]:
        x = selected_batch[name]

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)

        robot_snapshot[name] = x.copy()


    if reference_robot is None:

        reference_robot = robot_snapshot

    else:

        for name in reference_robot:

            assert np.array_equal(
                reference_robot[name],
                robot_snapshot[name],
            ), (
                condition["name"],
                name,
            )


    # --------------------------------------------------------
    # Radio point population at initial observation.
    # --------------------------------------------------------

    obj = selected_batch[
        "scene_object_id"
    ]

    exists = selected_batch[
        "scene_exists"
    ]

    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()

    if isinstance(exists, torch.Tensor):
        exists = exists.detach().cpu().numpy()

    obj = np.asarray(obj)
    exists = np.asarray(exists).astype(bool)

    assert obj.ndim == 3
    assert exists.ndim == 3
    assert obj.shape == exists.shape

    radio_mask = (
        (obj[0, 0] == RADIO_ID)
        &
        exists[0, 0]
    )

    n_radio = int(
        radio_mask.sum()
    )

    assert n_radio > 0


    # --------------------------------------------------------
    # Existing validated copy of official visibility logic.
    # No dynamics inference.
    # --------------------------------------------------------

    batch_gpu = {
        k: (
            v.to(
                tester.device,
                non_blocking=True,
            )
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in selected_batch.items()
        if k != "scene_object_id"
    }

    with torch.no_grad():

        vis = (
            get_visibility_count(
                tester.model,
                batch_gpu,
            )[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )


    assert len(vis) == obj.shape[-1]
    assert np.isin(vis, [0, 1, 2]).all()

    rv = vis[
        radio_mask
    ]

    n0 = int(
        (rv == 0).sum()
    )

    n1 = int(
        (rv == 1).sum()
    )

    n2 = int(
        (rv == 2).sum()
    )

    results.append({
        "condition":
            condition["name"],

        "camera_seed":
            camera_seed,

        "selected_order":
            "|".join(
                state["selected"]
            ),

        "camera_pair":
            "|".join(
                sorted(
                    state["selected"]
                )
            ),

        "radio_points":
            n_radio,

        "view0":
            n0,

        "view1":
            n1,

        "view2":
            n2,

        "frac_view0":
            n0 / n_radio,

        "frac_view1":
            n1 / n_radio,

        "frac_view2":
            n2 / n_radio,

        "mean_visible_cameras":
            float(
                rv.mean()
            ),
    })


# ------------------------------------------------------------
# Restore runtime symbols.
# ------------------------------------------------------------

dlmod.sample_cameras = official_sample_cameras
cammod.sample_cameras = official_cam_sample_cameras

dlmod.build_flow_sample = orig_dl_build
decmod.build_flow_sample = orig_dec_build
pmod.grid_sample_transform = orig_grid
pmod.gather = orig_gather


df = pd.DataFrame(results)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTDIR / "radio_camera_pair_visibility.csv",
    index=False,
)


print()
print("=" * 80)
print("RADIO CAMERA-PAIR VISIBILITY")
print("=" * 80)

print(
    df.to_string(
        index=False
    )
)

print()
print(
    "ACTION_ROBOT_INPUTS_IDENTICAL = True"
)

print(
    "NO DYNAMICS MODEL INFERENCE = True"
)

print(
    "CAMERA_COUNT_ALWAYS_2 = True"
)

print(
    "OFFICIAL sample_cameras USED = True"
)

print(
    "OFFICIAL FILES MODIFIED = False"
)

print(
    "SAVED =",
    OUTDIR / "radio_camera_pair_visibility.csv",
)
