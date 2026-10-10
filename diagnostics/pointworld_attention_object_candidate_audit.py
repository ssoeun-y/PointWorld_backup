#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from arguments import parse_args
from evaluation.tester import Tester
from ptv3.ptv3 import prepare_flash_attn_dtype


KEY = "task-0000_episode_00002530-300:311"

TARGET_LAYER = (
    "dynamics_predictor.predictor_model."
    "enc.enc0.block0.attn"
)

TRAJ = Path(
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237/"
    + KEY.replace(":", "_")
    + ".npz"
)

INVENTORY = Path(
    "diagnostics/results/"
    "behavior_object_distance_motion_full578_20260810_114615/"
    "per_object.csv"
)

OUTDIR = Path(
    "diagnostics/results/"
    "pointworld_attention_object_candidate_audit"
)


def get_key(batch):
    x = batch["__key__"]

    if isinstance(x, (list, tuple)):
        x = x[0]

    if isinstance(x, bytes):
        x = x.decode()

    return str(x)


# ============================================================
# Existing validated object identity
# ============================================================

with np.load(
    TRAJ,
    allow_pickle=True,
) as z:

    object_ids = np.asarray(
        z["scene_object_id"]
    )

    gt = np.asarray(
        z["gt_scene_flows"]
    )


while (
    object_ids.ndim > 2
    and object_ids.shape[0] == 1
):
    object_ids = object_ids[0]

while (
    gt.ndim > 3
    and gt.shape[0] == 1
):
    gt = gt[0]


if object_ids.ndim == 2:
    obj0 = object_ids[0]
elif object_ids.ndim == 1:
    obj0 = object_ids
else:
    raise RuntimeError(
        f"unexpected object id shape "
        f"{object_ids.shape}"
    )


assert gt.ndim == 3
gt0 = gt[0]


inv = pd.read_csv(INVENTORY)

sample_inv = inv[
    inv["key"] == KEY
].copy()

name_map = {
    int(r.object_id):
        str(r.object_name)
    for _, r in sample_inv.iterrows()
}

radio_rows = sample_inv[
    sample_inv[
        "object_name"
    ].str.contains(
        "__radio_",
        regex=False,
    )
]

assert len(radio_rows) == 1

radio_id = int(
    radio_rows.iloc[0]["object_id"]
)


# ============================================================
# Model
# ============================================================

args = parse_args()

tester = Tester(args)
model = tester.model
model.eval()

module = dict(
    model.named_modules()
)[TARGET_LAYER]

loader, _ = tester._build_eval_loader(
    "test",
    enable_mask=False,
)

selected = None

for batch in loader:

    if get_key(batch) == KEY:
        selected = batch
        break


assert selected is not None


# ============================================================
# Exact scene alignment BEFORE attention analysis
# ============================================================

scene_coord0 = (
    selected["scene_flows"][0, 0]
    .detach()
    .cpu()
    .numpy()
)

scene_exists0 = (
    selected["scene_exists"][0, 0]
    .detach()
    .cpu()
    .numpy()
    .astype(bool)
)

assert scene_coord0.shape == gt0.shape

print(
    "EXACT_STORED_SCENE_ALIGNMENT =",
    np.array_equal(
        scene_coord0,
        gt0,
    ),
)

assert np.array_equal(
    scene_coord0,
    gt0,
)

valid_obj0 = obj0[
    scene_exists0
]

valid_scene_xyz = scene_coord0[
    scene_exists0
]


# ============================================================
# Attention hook
# ============================================================

capture = {"calls": 0}


def hook(mod, inputs):

    capture["calls"] += 1

    point = inputs[0]

    H = mod.num_heads
    C = mod.channels
    D = C // H

    is_robot = point["is_robot"].bool()

    n_scene = int(
        (~is_robot).sum().item()
    )

    n_robot = int(
        is_robot.sum().item()
    )

    assert n_scene == len(
        valid_obj0
    )

    # enc0.block0 is before downsampling:
    # scene token order must match exact valid scene order.
    point_scene_coord = (
        point.coord[
            ~is_robot
        ]
        .detach()
        .cpu()
        .numpy()
    )

    assert np.array_equal(
        point_scene_coord,
        valid_scene_xyz,
    )

    # Original token label:
    # scene -> actual object id
    # robot -> -1
    token_obj = torch.full(
        (len(is_robot),),
        -1,
        dtype=torch.long,
        device=point.feat.device,
    )

    token_obj[
        ~is_robot
    ] = torch.as_tensor(
        valid_obj0,
        dtype=torch.long,
        device=point.feat.device,
    )

    unique_objects = sorted(
        int(x)
        for x in np.unique(
            valid_obj0
        )
    )

    pad, unpad, cu = (
        mod.get_padding_and_inverse(
            point
        )
    )

    order = (
        point.serialized_order[
            mod.order_index
        ][pad]
    )

    inverse = (
        unpad[
            point.serialized_inverse[
                mod.order_index
            ]
        ]
    )

    qkv = (
        mod.qkv(point.feat)[order]
        .reshape(
            -1,
            3,
            H,
            D,
        )
    )

    qkv = prepare_flash_attn_dtype(
        qkv
    )

    # Which serialized queries survive official unpadding?
    real_query = torch.zeros(
        len(order),
        dtype=torch.bool,
        device=order.device,
    )

    real_query[
        inverse
    ] = True


    attention_sum = {
        oid: torch.zeros(
            H,
            dtype=torch.float32,
            device=point.feat.device,
        )
        for oid in unique_objects
    }

    candidate_sum = {
        oid: 0.0
        for oid in unique_objects
    }


    total_robot_queries = 0

    # Also independently accumulate total scene quantities
    # for a cross-check.
    scene_attention_sum = torch.zeros(
        H,
        dtype=torch.float32,
        device=point.feat.device,
    )

    scene_candidate_sum = 0.0


    cu_np = (
        cu.detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )


    for p in range(
        len(cu_np) - 1
    ):

        s = int(
            cu_np[p]
        )

        e = int(
            cu_np[p + 1]
        )

        q = qkv[
            s:e,
            0,
        ].float()

        k = qkv[
            s:e,
            1,
        ].float()

        score = torch.einsum(
            "ihd,jhd->hij",
            q,
            k,
        )

        score = (
            score
            * mod.scale
        )

        attn = torch.softmax(
            score,
            dim=-1,
        )

        original_idx = (
            order[s:e]
        )

        local_robot = (
            is_robot[
                original_idx
            ]
        )

        local_real = (
            real_query[
                s:e
            ]
        )

        real_robot_q = (
            local_real
            & local_robot
        )

        nq = int(
            real_robot_q
            .sum()
            .item()
        )

        if nq == 0:
            continue


        total_robot_queries += nq

        key_obj = (
            token_obj[
                original_idx
            ]
        )

        # ----------------------------------------------------
        # Overall scene candidate / attention mass
        # ----------------------------------------------------
        scene_key = (
            key_obj >= 0
        )

        scene_candidate_frac = float(
            scene_key
            .float()
            .mean()
            .item()
        )

        scene_candidate_sum += (
            nq
            * scene_candidate_frac
        )

        scene_attention_sum += (
            attn[
                :,
                real_robot_q,
                :,
            ][
                :,
                :,
                scene_key,
            ]
            .sum(
                dim=(1, 2)
            )
        )


        # ----------------------------------------------------
        # Per-object candidate baseline and actual attention
        # ----------------------------------------------------
        for oid in unique_objects:

            object_key = (
                key_obj == oid
            )

            candidate_frac = float(
                object_key
                .float()
                .mean()
                .item()
            )

            candidate_sum[
                oid
            ] += (
                nq
                * candidate_frac
            )

            if object_key.any():

                attention_sum[
                    oid
                ] += (
                    attn[
                        :,
                        real_robot_q,
                        :,
                    ][
                        :,
                        :,
                        object_key,
                    ]
                    .sum(
                        dim=(1, 2)
                    )
                )


    assert total_robot_queries > 0


    rows = []

    for oid in unique_objects:

        candidate = (
            candidate_sum[
                oid
            ]
            / total_robot_queries
        )

        attention = (
            attention_sum[
                oid
            ]
            / float(
                total_robot_queries
            )
        )

        n_points = int(
            (
                valid_obj0
                == oid
            ).sum()
        )

        for h in range(H):

            a = float(
                attention[h].item()
            )

            rows.append({
                "head":
                    h,

                "object_id":
                    oid,

                "object_name":
                    name_map.get(
                        oid,
                        f"object_{oid}",
                    ),

                "n_points":
                    n_points,

                "candidate_fraction":
                    candidate,

                "attention_mass":
                    a,

                "attention_minus_candidate":
                    (
                        a
                        - candidate
                    ),
            })


    capture["df"] = pd.DataFrame(
        rows
    )

    capture[
        "total_robot_queries"
    ] = total_robot_queries

    capture[
        "scene_candidate_fraction"
    ] = (
        scene_candidate_sum
        / total_robot_queries
    )

    capture[
        "scene_attention_mass"
    ] = (
        scene_attention_sum
        / float(
            total_robot_queries
        )
    ).detach().cpu().numpy()

    capture["H"] = H


handle = (
    module.register_forward_pre_hook(
        hook
    )
)


batch_gpu = {
    k: (
        v.to(
            tester.device,
            non_blocking=True,
        )
        if isinstance(
            v,
            torch.Tensor,
        )
        else v
    )
    for k, v in selected.items()
}


with torch.no_grad():

    _ = model(
        batch_gpu,
        training=False,
    )


handle.remove()

assert capture["calls"] == 1


# ============================================================
# Cross-checks
# ============================================================

df = capture["df"]

per_head_attention_sum = (
    df.groupby("head")[
        "attention_mass"
    ].sum()
)

per_head_candidate_sum = (
    df.groupby("head")[
        "candidate_fraction"
    ].sum()
)


print()
print("=" * 80)
print("ROBOT QUERY -> OBJECT CANDIDATE AUDIT")
print("=" * 80)

print(
    "sample =",
    KEY,
)

print(
    "real robot queries =",
    capture[
        "total_robot_queries"
    ],
)

print(
    "objects =",
    df[
        "object_id"
    ].nunique(),
)

print()
print(
    "direct scene candidate fraction =",
    capture[
        "scene_candidate_fraction"
    ],
)

print(
    "sum object candidate fractions by head ="
)

print(
    per_head_candidate_sum
    .to_numpy()
)

print()
print(
    "direct robot->scene attention by head ="
)

print(
    capture[
        "scene_attention_mass"
    ]
)

print(
    "sum object attention masses by head ="
)

print(
    per_head_attention_sum
    .to_numpy()
)

print()
print(
    "candidate reconstruction abs diff ="
)

print(
    np.abs(
        per_head_candidate_sum
        .to_numpy()
        - capture[
            "scene_candidate_fraction"
        ]
    )
)

print(
    "attention reconstruction abs diff ="
)

print(
    np.abs(
        per_head_attention_sum
        .to_numpy()
        - capture[
            "scene_attention_mass"
        ]
    )
)


# ============================================================
# Radio head-by-head
# ============================================================

radio = (
    df[
        df[
            "object_id"
        ] == radio_id
    ]
    .sort_values("head")
)


print()
print("=" * 80)
print("RADIO: ROBOT QUERY -> RADIO")
print("=" * 80)

print(
    radio[
        [
            "head",
            "n_points",
            "candidate_fraction",
            "attention_mass",
            "attention_minus_candidate",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Ranking for the heads previously identified
# ============================================================

for h in [0, 3, 7]:

    sub = (
        df[
            df["head"] == h
        ]
        .sort_values(
            "attention_minus_candidate",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print()
    print("=" * 80)
    print(
        f"HEAD {h}: SELECTIVITY RANKING"
    )
    print("=" * 80)

    print(
        sub[
            [
                "object_id",
                "object_name",
                "n_points",
                "candidate_fraction",
                "attention_mass",
                "attention_minus_candidate",
            ]
        ].to_string(
            index=False
        )
    )

    radio_rank = int(
        sub.index[
            sub[
                "object_id"
            ] == radio_id
        ][0]
    ) + 1

    print()
    print(
        "radio selectivity rank =",
        radio_rank,
        "/",
        len(sub),
    )


OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTDIR
    / "robot_query_object_candidate_audit.csv",
    index=False,
)


print()
print(
    "SAVED =",
    OUTDIR
    / "robot_query_object_candidate_audit.csv",
)

print("NO NEW MODEL INTERVENTION = True")
print("NO NEW THRESHOLD = True")
print("OFFICIAL_CODE_MODIFIED = False")
