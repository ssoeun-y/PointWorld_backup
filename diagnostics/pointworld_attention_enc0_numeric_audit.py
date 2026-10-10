#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from arguments import parse_args
from evaluation.tester import Tester
from ptv3.ptv3 import prepare_flash_attn_dtype

from flash_attn.flash_attn_interface import (
    flash_attn_varlen_qkvpacked_func,
)


TARGET_LAYER = (
    "dynamics_predictor.predictor_model."
    "enc.enc0.block0.attn"
)

# Already heavily validated interaction sample.
# This is only the attention extraction smoke sample,
# not the final population.
TARGET_KEY = "task-0000_episode_00002530-300:311"

OUTDIR = Path(
    "diagnostics/results/"
    "pointworld_attention_enc0_smoke"
)


def get_key(batch):
    x = batch["__key__"]

    if isinstance(x, (list, tuple)):
        x = x[0]

    if isinstance(x, bytes):
        x = x.decode("utf-8")

    return str(x)


args = parse_args()

tester = Tester(args)
model = tester.model
model.eval()

modules = dict(model.named_modules())

assert TARGET_LAYER in modules, TARGET_LAYER
attn_module = modules[TARGET_LAYER]

capture = {
    "called": 0,
}


def attention_pre_hook(module, inputs):

    capture["called"] += 1

    point = inputs[0]

    H = module.num_heads
    K = module.patch_size
    C = module.channels
    D = C // H

    # ----------------------------------------------------------
    # EXACT SAME serialization / padding path as official
    # SerializedAttention.forward()
    # ----------------------------------------------------------
    pad, unpad, cu_seqlens = (
        module.get_padding_and_inverse(point)
    )

    order = (
        point.serialized_order[
            module.order_index
        ][pad]
    )

    inverse = (
        unpad[
            point.serialized_inverse[
                module.order_index
            ]
        ]
    )

    qkv = (
        module.qkv(point.feat)[order]
        .reshape(-1, 3, H, D)
    )

    qkv_prepared = (
        prepare_flash_attn_dtype(qkv)
    )

    # Original point identities BEFORE serialization.
    is_robot = point["is_robot"].bool()

    coord = point.coord

    N = int(point.feat.shape[0])

    assert is_robot.numel() == N
    assert coord.shape[0] == N
    assert inverse.numel() == N

    # A padded query position is "real" iff it is used
    # by the official inverse operation.
    real_query = torch.zeros(
        len(order),
        dtype=torch.bool,
        device=order.device,
    )

    real_query[inverse] = True

    # Per-original-point results.
    robot_mass_by_head = torch.full(
        (N, H),
        float("nan"),
        dtype=torch.float32,
        device=point.feat.device,
    )

    robot_candidate_fraction = torch.full(
        (N,),
        float("nan"),
        dtype=torch.float32,
        device=point.feat.device,
    )

    patch_index_per_point = torch.full(
        (N,),
        -1,
        dtype=torch.long,
        device=point.feat.device,
    )

    manual_outputs = []

    cu = (
        cu_seqlens
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    # ----------------------------------------------------------
    # Reconstruct:
    # softmax(Q K^T * official_scale)
    #
    # Do it patch-by-patch using the exact serialized patches
    # used by PointWorld.
    # ----------------------------------------------------------
    for patch_idx in range(len(cu) - 1):

        start = int(cu[patch_idx])
        end = int(cu[patch_idx + 1])

        q_native = qkv_prepared[start:end, 0]
        k_native = qkv_prepared[start:end, 1]
        v_native = qkv_prepared[start:end, 2]

        # Main reconstruction in FP32 for numerically stable
        # QK^T + softmax.
        q = q_native.float()
        k = k_native.float()
        v = v_native.float()

        # q,k: [L,H,D]
        # scores: [H,L,L]
        scores = torch.einsum(
            "ihd,jhd->hij",
            q,
            k,
        )

        scores = scores * module.scale

        attn = torch.softmax(
            scores,
            dim=-1,
        )

        # Manual attention output.
        # [H,L,L] x [L,H,D] -> [L,H,D]
        manual_out = torch.einsum(
            "hij,jhd->ihd",
            attn,
            v,
        )

        manual_outputs.append(
            manual_out
        )

        original_indices = (
            order[start:end]
        )

        key_is_robot = (
            is_robot[
                original_indices
            ]
        )

        # Candidate composition actually seen by attention,
        # including official padding semantics.
        candidate_robot_frac = (
            key_is_robot
            .float()
            .mean()
        )

        if key_is_robot.any():
            robot_mass = (
                attn[
                    :,
                    :,
                    key_is_robot,
                ]
                .sum(dim=-1)
            )
        else:
            robot_mass = torch.zeros(
                (H, end - start),
                device=attn.device,
                dtype=attn.dtype,
            )

        # Only save query rows that survive official unpadding.
        local_real = (
            real_query[start:end]
        )

        real_local_idx = torch.where(
            local_real
        )[0]

        for local_idx in real_local_idx:

            original_idx = int(
                original_indices[
                    local_idx
                ].item()
            )

            robot_mass_by_head[
                original_idx
            ] = robot_mass[
                :,
                local_idx,
            ]

            robot_candidate_fraction[
                original_idx
            ] = candidate_robot_frac

            patch_index_per_point[
                original_idx
            ] = patch_idx

    manual_output = torch.cat(
        manual_outputs,
        dim=0,
    )

    assert torch.isfinite(
        robot_mass_by_head
    ).all()

    assert torch.isfinite(
        robot_candidate_fraction
    ).all()

    # ----------------------------------------------------------
    # Numerical cross-check only.
    #
    # No pass/fail threshold:
    # compare our mathematically reconstructed attention output
    # to the actual FlashAttention output used by PointWorld.
    # ----------------------------------------------------------
    flash_output = (
        flash_attn_varlen_qkvpacked_func(
            qkv_prepared,
            cu_seqlens,
            max_seqlen=module.patch_size,
            dropout_p=0.0,
            softmax_scale=module.scale,
        )
        .float()
    )

    absdiff = (
        manual_output
        - flash_output
    ).abs()

    # ----------------------------------------------------------
    # Save everything needed for later visualization.
    # ----------------------------------------------------------
    capture["H"] = H
    capture["K"] = K
    capture["C"] = C

    capture["N"] = N

    capture["num_patches"] = (
        len(cu) - 1
    )

    capture["is_robot"] = (
        is_robot
        .detach()
        .cpu()
        .numpy()
    )

    capture["coord"] = (
        coord
        .detach()
        .cpu()
        .numpy()
    )

    capture["robot_mass_by_head"] = (
        robot_mass_by_head
        .detach()
        .cpu()
        .numpy()
    )

    capture[
        "robot_candidate_fraction"
    ] = (
        robot_candidate_fraction
        .detach()
        .cpu()
        .numpy()
    )

    capture["patch_index"] = (
        patch_index_per_point
        .detach()
        .cpu()
        .numpy()
    )

    absdiff_flat = absdiff.flatten()

    capture["manual_flash_max_absdiff"] = float(
        absdiff_flat.max().item()
    )
    capture["manual_flash_mean_absdiff"] = float(
        absdiff_flat.mean().item()
    )
    capture["manual_flash_p50_absdiff"] = float(
        torch.quantile(absdiff_flat, 0.50).item()
    )
    capture["manual_flash_p95_absdiff"] = float(
        torch.quantile(absdiff_flat, 0.95).item()
    )
    capture["manual_flash_p99_absdiff"] = float(
        torch.quantile(absdiff_flat, 0.99).item()
    )

    flash_abs = flash_output.abs().flatten()

    capture["flash_abs_mean"] = float(
        flash_abs.mean().item()
    )
    capture["flash_abs_p95"] = float(
        torch.quantile(flash_abs, 0.95).item()
    )
    capture["flash_abs_max"] = float(
        flash_abs.max().item()
    )

    manual_abs = manual_output.abs().flatten()

    capture["manual_abs_mean"] = float(
        manual_abs.mean().item()
    )
    capture["manual_abs_p95"] = float(
        torch.quantile(manual_abs, 0.95).item()
    )
    capture["manual_abs_max"] = float(
        manual_abs.max().item()
    )

    capture["qkv_prepared_dtype"] = str(
        qkv_prepared.dtype
    )


handle = attn_module.register_forward_pre_hook(
    attention_pre_hook
)


# --------------------------------------------------------------
# Find the known interaction sample.
# No inference is run on earlier samples.
# --------------------------------------------------------------
loader, _ = tester._build_eval_loader(
    "test",
    enable_mask=False,
)

selected = None

for batch in loader:

    key = get_key(batch)

    if key == TARGET_KEY:
        selected = batch
        break


assert selected is not None, TARGET_KEY

key = get_key(selected)

# Input sizes BEFORE exists filtering.
Ns = int(
    selected[
        "scene_flows"
    ].shape[2]
)

T = int(
    selected[
        "robot_flows"
    ].shape[1]
)

Nr = int(
    selected[
        "robot_flows"
    ].shape[2]
)

scene_exists0 = (
    selected[
        "scene_exists"
    ][:, 0]
)

robot_exists = (
    selected[
        "robot_exists"
    ]
)

n_scene_valid = int(
    scene_exists0.sum().item()
)

n_robot_valid = int(
    robot_exists.sum().item()
)


batch_model = {
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
        batch_model,
        training=False,
    )


handle.remove()


# --------------------------------------------------------------
# Structural validations only.
# --------------------------------------------------------------
assert capture["called"] == 1

assert capture["N"] == (
    n_scene_valid
    + n_robot_valid
)

is_robot = capture["is_robot"]

assert int(is_robot.sum()) == n_robot_valid
assert int((~is_robot).sum()) == n_scene_valid


robot_mass = capture[
    "robot_mass_by_head"
]

candidate_robot_frac = capture[
    "robot_candidate_fraction"
]

coord = capture["coord"]

patch_idx = capture[
    "patch_index"
]


# --------------------------------------------------------------
# HEAD-LEVEL SUMMARY
# --------------------------------------------------------------
scene_mask = ~is_robot
robot_mask = is_robot

rows = []

for h in range(capture["H"]):

    scene_to_robot = (
        robot_mass[
            scene_mask,
            h,
        ]
    )

    robot_to_scene = (
        1.0
        - robot_mass[
            robot_mask,
            h,
        ]
    )

    rows.append({
        "head": h,

        "scene_query_to_robot_attention_mean":
            float(
                scene_to_robot.mean()
            ),

        "scene_query_robot_candidate_fraction_mean":
            float(
                candidate_robot_frac[
                    scene_mask
                ].mean()
            ),

        "scene_query_robot_attention_minus_candidate":
            float(
                (
                    scene_to_robot
                    - candidate_robot_frac[
                        scene_mask
                    ]
                ).mean()
            ),

        "robot_query_to_scene_attention_mean":
            float(
                robot_to_scene.mean()
            ),

        "robot_query_scene_candidate_fraction_mean":
            float(
                (
                    1.0
                    - candidate_robot_frac[
                        robot_mask
                    ]
                ).mean()
            ),

        "robot_query_scene_attention_minus_candidate":
            float(
                (
                    robot_to_scene
                    - (
                        1.0
                        - candidate_robot_frac[
                            robot_mask
                        ]
                    )
                ).mean()
            ),
    })


head_df = pd.DataFrame(rows)


# --------------------------------------------------------------
# SCENE-POINT MAP
# Each point gets mean attention over heads directed to
# robot/action candidate tokens.
# --------------------------------------------------------------
scene_robot_mass_mean = (
    robot_mass[
        scene_mask
    ].mean(axis=1)
)

scene_candidate_robot = (
    candidate_robot_frac[
        scene_mask
    ]
)

scene_coords = coord[
    scene_mask
]

scene_patches = patch_idx[
    scene_mask
]

scene_df = pd.DataFrame({
    "x": scene_coords[:, 0],
    "y": scene_coords[:, 1],
    "z": scene_coords[:, 2],

    "patch_index":
        scene_patches,

    "robot_attention_mean":
        scene_robot_mass_mean,

    "robot_candidate_fraction":
        scene_candidate_robot,

    "attention_minus_candidate":
        (
            scene_robot_mass_mean
            - scene_candidate_robot
        ),
})


OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

head_df.to_csv(
    OUTDIR / "head_summary.csv",
    index=False,
)

scene_df.to_csv(
    OUTDIR / "scene_point_robot_attention.csv",
    index=False,
)

np.savez_compressed(
    OUTDIR / "attention_capture.npz",

    key=np.array(key),

    coord=coord,

    is_robot=is_robot,

    patch_index=patch_idx,

    robot_mass_by_head=robot_mass,

    robot_candidate_fraction=(
        candidate_robot_frac
    ),
)


print("=" * 80)
print("POINTWORLD ENC0.BLOCK0 ATTENTION SMOKE")
print("=" * 80)

print("sample =", key)

print()
print("Ns =", Ns)
print("T  =", T)
print("Nr =", Nr)

print()
print(
    "valid scene tokens =",
    n_scene_valid,
)

print(
    "valid robot/action tokens =",
    n_robot_valid,
)

print(
    "total PTv3 tokens =",
    capture["N"],
)

print()
print(
    "attention heads =",
    capture["H"],
)

print(
    "patch size =",
    capture["K"],
)

print(
    "serialized patches =",
    capture["num_patches"],
)

print()
print(
    "manual-vs-Flash max abs diff =",
    capture[
        "manual_flash_max_absdiff"
    ],
)

print(
    "qkv prepared dtype =",
    capture["qkv_prepared_dtype"],
)

print(
    "manual-vs-Flash mean abs diff =",
    capture["manual_flash_mean_absdiff"],
)
print(
    "manual-vs-Flash p50 abs diff =",
    capture["manual_flash_p50_absdiff"],
)
print(
    "manual-vs-Flash p95 abs diff =",
    capture["manual_flash_p95_absdiff"],
)
print(
    "manual-vs-Flash p99 abs diff =",
    capture["manual_flash_p99_absdiff"],
)
print(
    "manual-vs-Flash max abs diff =",
    capture["manual_flash_max_absdiff"],
)

print()
print(
    "Flash |output| mean/p95/max =",
    capture["flash_abs_mean"],
    capture["flash_abs_p95"],
    capture["flash_abs_max"],
)
print(
    "Manual |output| mean/p95/max =",
    capture["manual_abs_mean"],
    capture["manual_abs_p95"],
    capture["manual_abs_max"],
)

print()
print("=== HEAD SUMMARY ===")
print(
    head_df.to_string(
        index=False
    )
)

print()
print("=== SCENE POINT SUMMARY ===")

print(
    "mean robot attention =",
    float(
        scene_df[
            "robot_attention_mean"
        ].mean()
    ),
)

print(
    "mean robot candidate fraction =",
    float(
        scene_df[
            "robot_candidate_fraction"
        ].mean()
    ),
)

print(
    "mean attention-minus-candidate =",
    float(
        scene_df[
            "attention_minus_candidate"
        ].mean()
    ),
)

print()
print(
    "SAVED =",
    OUTDIR,
)

print(
    "MODEL_OUTPUT_MODIFIED = False"
)

print(
    "OFFICIAL_CODE_MODIFIED = False"
)

print(
    "ATTENTION_RECONSTRUCTION = "
    "softmax(QK^T * official_scale)"
)
