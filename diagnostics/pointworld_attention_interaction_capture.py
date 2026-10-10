#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import torch

from arguments import parse_args
from evaluation.tester import Tester
from ptv3.ptv3 import prepare_flash_attn_dtype


TARGET_KEY = "task-0000_episode_00002530-300:311"
TARGET_LAYER = (
    "dynamics_predictor.predictor_model."
    "enc.enc0.block0.attn"
)

OUTDIR = Path(
    "diagnostics/results/"
    "pointworld_attention_interaction_overlay"
)


def get_key(batch):
    x = batch["__key__"]

    if isinstance(x, (list, tuple)):
        x = x[0]

    if isinstance(x, bytes):
        x = x.decode()

    return str(x)


args = parse_args()

tester = Tester(args)
model = tester.model
model.eval()

module = dict(
    model.named_modules()
)[TARGET_LAYER]

capture = {"calls": 0}


def hook(mod, inputs):

    capture["calls"] += 1

    point = inputs[0]

    H = mod.num_heads
    C = mod.channels
    D = C // H

    pad, unpad, cu = (
        mod.get_padding_and_inverse(point)
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
        .reshape(-1, 3, H, D)
    )

    qkv = prepare_flash_attn_dtype(
        qkv
    )

    is_robot = point["is_robot"].bool()

    N = len(is_robot)

    # ------------------------------------------------------
    # Which serialized query positions correspond to
    # real PointWorld outputs rather than padding copies?
    # ------------------------------------------------------
    real_query = torch.zeros(
        len(order),
        dtype=torch.bool,
        device=order.device,
    )

    real_query[inverse] = True

    # Scene-query -> robot-key attention mass.
    scene_to_robot = torch.full(
        (N, H),
        float("nan"),
        dtype=torch.float32,
        device=point.feat.device,
    )

    candidate_robot_fraction = torch.full(
        (N,),
        float("nan"),
        dtype=torch.float32,
        device=point.feat.device,
    )

    # ------------------------------------------------------
    # Incoming attention FROM all REAL robot queries TO each
    # original token.
    #
    # Summing over key duplicates produced by official padding
    # is required because those copies really participate as
    # keys in FlashAttention.
    # ------------------------------------------------------
    incoming_from_robot_sum = torch.zeros(
        (N, H),
        dtype=torch.float32,
        device=point.feat.device,
    )

    total_real_robot_queries = 0

    cu_np = (
        cu.detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    for p in range(len(cu_np) - 1):

        s = int(cu_np[p])
        e = int(cu_np[p + 1])

        q = qkv[s:e, 0].float()
        k = qkv[s:e, 1].float()

        score = torch.einsum(
            "ihd,jhd->hij",
            q,
            k,
        )

        score = score * mod.scale

        attn = torch.softmax(
            score,
            dim=-1,
        )

        original_idx = order[s:e]

        local_is_robot = (
            is_robot[original_idx]
        )

        local_real = (
            real_query[s:e]
        )

        # ----------------------------------------------
        # A) REAL scene query -> robot keys
        # ----------------------------------------------
        if local_is_robot.any():

            mass_to_robot = (
                attn[
                    :,
                    :,
                    local_is_robot,
                ]
                .sum(dim=-1)
            )
        else:

            mass_to_robot = torch.zeros(
                (H, e - s),
                device=attn.device,
            )

        candidate_frac = (
            local_is_robot
            .float()
            .mean()
        )

        real_local_idx = torch.where(
            local_real
        )[0]

        for li in real_local_idx:

            oi = int(
                original_idx[li].item()
            )

            scene_to_robot[oi] = (
                mass_to_robot[:, li]
            )

            candidate_robot_fraction[oi] = (
                candidate_frac
            )

        # ----------------------------------------------
        # B) REAL robot queries -> scene/key locations
        # ----------------------------------------------
        real_robot_queries = (
            local_real
            & local_is_robot
        )

        nq = int(
            real_robot_queries.sum().item()
        )

        if nq > 0:

            # [H, Qrobot, K] -> [H,K]
            incoming = (
                attn[
                    :,
                    real_robot_queries,
                    :,
                ]
                .sum(dim=1)
            )

            # Key positions can contain duplicates because
            # of official padding. index_add correctly merges
            # their actual attention contribution.
            incoming_from_robot_sum.index_add_(
                0,
                original_idx,
                incoming.transpose(0, 1),
            )

            total_real_robot_queries += nq


    assert total_real_robot_queries > 0

    incoming_from_robot_mean = (
        incoming_from_robot_sum
        / float(total_real_robot_queries)
    )

    # All original queries must have a reconstructed
    # scene->robot value.
    assert torch.isfinite(
        scene_to_robot
    ).all()

    assert torch.isfinite(
        candidate_robot_fraction
    ).all()

    capture["coord"] = (
        point.coord
        .detach()
        .cpu()
        .numpy()
    )

    capture["is_robot"] = (
        is_robot
        .detach()
        .cpu()
        .numpy()
    )

    capture["scene_to_robot"] = (
        scene_to_robot
        .detach()
        .cpu()
        .numpy()
    )

    capture["candidate_robot_fraction"] = (
        candidate_robot_fraction
        .detach()
        .cpu()
        .numpy()
    )

    capture["incoming_from_robot"] = (
        incoming_from_robot_mean
        .detach()
        .cpu()
        .numpy()
    )

    capture["total_real_robot_queries"] = (
        total_real_robot_queries
    )

    capture["heads"] = H


handle = module.register_forward_pre_hook(
    hook
)

loader, _ = tester._build_eval_loader(
    "test",
    enable_mask=False,
)

selected = None

for batch in loader:

    if get_key(batch) == TARGET_KEY:
        selected = batch
        break


assert selected is not None

T = int(
    selected["robot_flows"].shape[1]
)

Nr = int(
    selected["robot_flows"].shape[2]
)

robot_exists = (
    selected["robot_exists"]
    .detach()
    .cpu()
    .numpy()
    .astype(bool)
)

batch_gpu = {
    k: (
        v.to(
            tester.device,
            non_blocking=True,
        )
        if isinstance(v, torch.Tensor)
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

is_robot = capture["is_robot"]

# Direct cross-check:
# sum of incoming scene-key mass =
# average robot-query -> scene mass.
incoming = capture[
    "incoming_from_robot"
]

incoming_scene_mass = (
    incoming[
        ~is_robot
    ].sum(axis=0)
)

robot_query_scene_direct = (
    1.0
    - capture[
        "scene_to_robot"
    ][is_robot].mean(axis=0)
)

print("=" * 80)
print("INTERACTION ATTENTION CAPTURE")
print("=" * 80)

print(
    "sample =",
    TARGET_KEY,
)

print(
    "scene tokens =",
    int((~is_robot).sum()),
)

print(
    "robot tokens =",
    int(is_robot.sum()),
)

print(
    "real robot queries =",
    capture[
        "total_real_robot_queries"
    ],
)

print()
print(
    "incoming scene mass by head =",
    incoming_scene_mass,
)

print(
    "direct robot->scene by head =",
    robot_query_scene_direct,
)

print(
    "absolute difference =",
    np.abs(
        incoming_scene_mass
        - robot_query_scene_direct
    ),
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

np.savez_compressed(
    OUTDIR / "interaction_attention_capture.npz",

    key=np.array(TARGET_KEY),

    coord=capture["coord"],

    is_robot=is_robot,

    scene_to_robot=(
        capture[
            "scene_to_robot"
        ]
    ),

    candidate_robot_fraction=(
        capture[
            "candidate_robot_fraction"
        ]
    ),

    incoming_from_robot=(
        capture[
            "incoming_from_robot"
        ]
    ),

    T=np.array(T),
    Nr=np.array(Nr),

    robot_exists=robot_exists,
)

print()
print(
    "SAVED =",
    OUTDIR
    / "interaction_attention_capture.npz",
)

print("OFFICIAL_CODE_MODIFIED = False")
print("MODEL_OUTPUT_MODIFIED = False")
