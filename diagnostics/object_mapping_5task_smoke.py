#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch

from arguments import parse_args

import dataset_components.dataloader as dlmod
import dataset_components.pipeline as pmod

import diagnostics.behavior_object_mapping_smoke as objmap


PAIR = (
    "diagnostics/results/"
    "temporal_quality_5task_matched_10pct_pairs.csv"
)

ROOTS = {
    "task-0000":
        "/home/soenchun/cse/PointWorld_workspace/"
        "datasets/behavior/wds_task0000",

    "task-0001":
        "/home/soenchun/cse/PointWorld_workspace/PointWorld/"
        "diagnostics/data/"
        "pointworld_behavior_task0001_restored/behavior/wds",

    "task-0006":
        "/home/soenchun/cse/PointWorld_workspace/PointWorld/"
        "diagnostics/data/"
        "pointworld_behavior_multitask_restored/behavior/wds",

    "task-0008":
        "/home/soenchun/cse/PointWorld_workspace/PointWorld/"
        "diagnostics/data/"
        "pointworld_behavior_multitask_restored/behavior/wds",

    "task-0010":
        "/home/soenchun/cse/PointWorld_workspace/PointWorld/"
        "diagnostics/data/"
        "pointworld_behavior_multitask_restored/behavior/wds",
}


# ------------------------------------------------------------
# Pick one REAL matched clip from each task.
# ------------------------------------------------------------
pairs = pd.read_csv(PAIR)

clips = pd.concat([
    pairs[["task", "hq_clip"]]
        .rename(columns={"hq_clip": "clip"}),
    pairs[["task", "lq_clip"]]
        .rename(columns={"lq_clip": "clip"}),
])

TARGETS = {}

for task in ROOTS:
    x = clips.loc[
        clips["task"] == task,
        "clip"
    ].astype(str)

    if len(x) == 0:
        raise RuntimeError(
            f"No matched clip for {task}"
        )

    TARGETS[task] = sorted(x.unique())[0]


print("TARGETS:")
for k, v in TARGETS.items():
    print(k, "->", v)


def make_args(data_dir):
    args = parse_args(
        skip_command_line=True
    )

    args.domains = ["behavior"]
    args.data_dirs = [data_dir]

    args.batch_size = 1
    args.num_workers = 0
    args.eval_num_workers = 0

    args.seed = 42
    args.device = "cpu"
    args.deterministic_data = True

    return args


def get_key(batch):
    x = batch["__key__"]

    if isinstance(x, (list, tuple)):
        x = x[0]

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    if isinstance(x, np.ndarray):
        x = x.reshape(-1)[0]

    if isinstance(x, bytes):
        x = x.decode()

    return str(x)


# ------------------------------------------------------------
# Runtime diagnostic patches ONLY.
# Same validated object-ID path.
# ------------------------------------------------------------
orig_dl = dlmod.build_flow_sample
orig_grid = pmod.grid_sample_transform
orig_gather = pmod.gather

dlmod.build_flow_sample = (
    objmap.build_flow_sample_with_object_ids
)
pmod.grid_sample_transform = (
    objmap.grid_sample_with_object_ids
)
pmod.gather = (
    objmap.gather_with_object_ids
)


results = []

try:

    for task, target in TARGETS.items():

        print()
        print("=" * 72)
        print(task)
        print("TARGET =", target)
        print("ROOT =", ROOTS[task])
        print("=" * 72)

        args = make_args(
            ROOTS[task]
        )

        loader, info = dlmod.build_dataloader(
            args,
            mode="test",
            rank=0,
            world_size=1,
            override_splits="test",
        )

        found = False

        for batch in loader:

            key = get_key(batch)

            if key != target:
                continue

            found = True

            if "scene_object_id" not in batch:
                raise RuntimeError(
                    f"{task}: scene_object_id missing"
                )

            ids = batch[
                "scene_object_id"
            ]

            if isinstance(ids, torch.Tensor):
                ids = (
                    ids.detach()
                    .cpu()
                    .numpy()
                )

            if ids.ndim == 4 and ids.shape[-1] == 1:
                ids = ids.squeeze(-1)

            if ids.ndim != 3:
                raise RuntimeError(
                    f"{task}: bad object ID shape "
                    f"{ids.shape}"
                )

            labels = ids[0]

            temporal_exact = (
                np.array_equal(
                    labels,
                    np.broadcast_to(
                        labels[0:1],
                        labels.shape,
                    ),
                )
            )

            scene = batch[
                "scene_flows"
            ]

            N_scene = int(
                scene.shape[2]
            )

            N_labels = int(
                labels.shape[1]
            )

            count_exact = (
                N_scene == N_labels
            )

            table = objmap.OBJECT_TABLE.get(
                key, {}
            )

            unique_ids = np.unique(
                labels[0]
            )

            table_ids = set(
                int(x)
                for x in table.keys()
            )

            label_ids = set(
                int(x)
                for x in unique_ids
            )

            ids_in_table = (
                label_ids <= table_ids
            )

            print(
                "scene_shape =",
                tuple(scene.shape),
            )
            print(
                "object_id_shape =",
                labels.shape,
            )
            print(
                "num_object_meshes =",
                len(table),
            )
            print(
                "num_unique_final_ids =",
                len(unique_ids),
            )
            print(
                "point_count_exact =",
                count_exact,
            )
            print(
                "temporal_object_id_exact =",
                temporal_exact,
            )
            print(
                "all_labels_have_mesh_name =",
                ids_in_table,
            )

            ok = all([
                count_exact,
                temporal_exact,
                ids_in_table,
                len(table) > 0,
            ])

            print(
                "TASK_OBJECT_MAPPING_OK =",
                ok,
            )

            results.append({
                "task": task,
                "clip": key,
                "num_objects": len(table),
                "N_scene": N_scene,
                "temporal_exact":
                    temporal_exact,
                "ids_in_table":
                    ids_in_table,
                "success": ok,
            })

            break

        if not found:
            raise RuntimeError(
                f"{task}: target clip "
                f"{target} not found"
            )

finally:

    dlmod.build_flow_sample = orig_dl
    pmod.grid_sample_transform = orig_grid
    pmod.gather = orig_gather


r = pd.DataFrame(results)

print()
print("=" * 72)
print("FINAL 5-TASK OBJECT MAPPING")
print("=" * 72)
print(r.to_string(index=False))

success = (
    len(r) == 5
    and r["success"].all()
)

print()
print(
    "FIVE_TASK_OBJECT_MAPPING_OK =",
    success,
)

if not success:
    print(
        "STOP: do not run object-level "
        "evaluation yet."
    )

print(
    "OBJECT_MAPPING_SMOKE_DONE=True"
)
