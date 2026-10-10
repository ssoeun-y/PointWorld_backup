#!/usr/bin/env python3

import inspect

from arguments import parse_args
from evaluation.tester import Tester
from ptv3.ptv3 import SerializedAttention
from flash_attn.flash_attn_interface import (
    flash_attn_varlen_qkvpacked_func,
)


args = parse_args()
tester = Tester(args)
model = tester.model

print("=" * 80)
print("POINTWORLD ATTENTION STRUCTURE AUDIT")
print("=" * 80)

print("\n[1] FlashAttention API")
try:
    print(
        "signature =",
        inspect.signature(
            flash_attn_varlen_qkvpacked_func
        ),
    )
except Exception as e:
    print("signature unavailable:", repr(e))

print("\n[2] SerializedAttention modules")

rows = []

for name, module in model.named_modules():
    if isinstance(module, SerializedAttention):
        row = {
            "name": name,
            "channels": module.channels,
            "num_heads": module.num_heads,
            "patch_size": module.patch_size,
            "order_index": module.order_index,
            "scale": module.scale,
        }
        rows.append(row)

        print(
            f"{len(rows)-1:02d} | "
            f"{name} | "
            f"C={module.channels} | "
            f"H={module.num_heads} | "
            f"K={module.patch_size} | "
            f"order_index={module.order_index}"
        )

print()
print("num attention modules =", len(rows))

print("\n[3] Core conditioning modules")

targets = [
    "scene_feature_encoder",
    "robot_proj",
    "time_embed",
    "robot_type_emb",
    "dynamics_predictor",
]

for target in targets:
    found = [
        n for n, _ in model.named_modules()
        if target in n
    ]

    if target == "robot_type_emb":
        found = [
            n for n, _ in model.named_parameters()
            if target in n
        ]

    print(target, "=", found[:20])

print("\n[4] Global robot-conditioning parameters")

for name, p in model.named_parameters():
    if (
        "robot_film_gamma" in name
        or "robot_film_beta" in name
        or "skip_film_gamma" in name
        or "skip_film_beta" in name
    ):
        print(
            name,
            "shape=",
            tuple(p.shape),
        )

print()
print("AUDIT_ONLY = True")
print("MODEL_WEIGHTS_MODIFIED = False")
