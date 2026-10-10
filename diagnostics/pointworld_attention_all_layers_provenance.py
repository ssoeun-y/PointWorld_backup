#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from arguments import parse_args
from evaluation.tester import Tester
from ptv3.ptv3 import SerializedAttention, prepare_flash_attn_dtype


TARGET_KEY = "task-0000_episode_00002530-300:311"

OUTDIR = Path(
    "diagnostics/results/"
    "pointworld_attention_all_layers_provenance"
)

FIRST_LAYER_REFERENCE = Path(
    "diagnostics/results/"
    "pointworld_attention_enc0_smoke/"
    "head_summary.csv"
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


# ============================================================
# Locate all actual SerializedAttention blocks.
# ============================================================

attn_modules = [
    (name, module)
    for name, module in model.named_modules()
    if isinstance(module, SerializedAttention)
]

print(
    "registered attention modules =",
    len(attn_modules),
)


# ============================================================
# Exact provenance reconstruction.
#
# For every current token:
#   robot_count = number of ORIGINAL robot tokens beneath it
#   total_count = number of ORIGINAL tokens beneath it
#
# robot_fraction = robot_count / total_count
#
# No binary threshold is introduced.
# ============================================================

provenance_cache = {}


def get_provenance_counts(point):

    cache_key = id(point)

    if cache_key in provenance_cache:
        return provenance_cache[cache_key]

    if "is_robot" in point.keys():

        robot_count = (
            point["is_robot"]
            .float()
        )

        total_count = torch.ones_like(
            robot_count
        )

    elif (
        "pooling_parent" in point.keys()
        and
        "pooling_inverse" in point.keys()
    ):

        parent = point["pooling_parent"]
        inverse = point["pooling_inverse"].long()

        parent_robot, parent_total = (
            get_provenance_counts(parent)
        )

        n_child = int(
            point.feat.shape[0]
        )

        robot_count = torch.zeros(
            n_child,
            dtype=torch.float32,
            device=point.feat.device,
        )

        total_count = torch.zeros(
            n_child,
            dtype=torch.float32,
            device=point.feat.device,
        )

        robot_count.scatter_add_(
            0,
            inverse,
            parent_robot,
        )

        total_count.scatter_add_(
            0,
            inverse,
            parent_total,
        )

        assert (
            total_count > 0
        ).all()

    else:
        raise RuntimeError(
            "Cannot reconstruct scene/robot provenance "
            "for this Point object."
        )

    provenance_cache[cache_key] = (
        robot_count,
        total_count,
    )

    return robot_count, total_count


head_rows = []
layer_rows = []
call_count = {name: 0 for name, _ in attn_modules}


def make_hook(layer_index, layer_name):

    def hook(module, inputs):

        call_count[layer_name] += 1

        point = inputs[0]

        H = module.num_heads
        C = module.channels
        D = C // H

        robot_count, total_count = (
            get_provenance_counts(point)
        )

        robot_fraction = (
            robot_count
            / total_count
        )

        scene_fraction = (
            1.0
            - robot_fraction
        )

        N = int(
            point.feat.shape[0]
        )

        assert len(robot_fraction) == N

        pure_scene = int(
            (robot_fraction == 0).sum().item()
        )

        pure_robot = int(
            (robot_fraction == 1).sum().item()
        )

        mixed = int(
            (
                (robot_fraction > 0)
                & (robot_fraction < 1)
            ).sum().item()
        )

        assert (
            pure_scene
            + pure_robot
            + mixed
            == N
        )


        # ----------------------------------------------------
        # Exact official serialization / padding.
        # ----------------------------------------------------

        pad, unpad, cu = (
            module.get_padding_and_inverse(
                point
            )
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
            module.qkv(
                point.feat
            )[order]
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


        # Only query rows used by official unpadding.
        real_query = torch.zeros(
            len(order),
            dtype=torch.bool,
            device=order.device,
        )

        real_query[inverse] = True


        # ----------------------------------------------------
        # Provenance-weighted quantities.
        #
        # For query q:
        #
        # attention-to-robot =
        #   sum_k attention(q,k) * robot_fraction(k)
        #
        # No thresholding of mixed pooled tokens.
        # ----------------------------------------------------

        scene_query_weight_sum = 0.0
        robot_query_weight_sum = 0.0

        scene_to_robot_attn_sum = torch.zeros(
            H,
            dtype=torch.float64,
            device=point.feat.device,
        )

        scene_robot_candidate_sum = 0.0

        robot_to_scene_attn_sum = torch.zeros(
            H,
            dtype=torch.float64,
            device=point.feat.device,
        )

        robot_scene_candidate_sum = 0.0


        cu_np = (
            cu.detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )


        for p in range(
            len(cu_np) - 1
        ):

            s = int(cu_np[p])
            e = int(cu_np[p + 1])

            q = qkv[
                s:e,
                0,
            ].float()

            k = qkv[
                s:e,
                1,
            ].float()

            scores = torch.einsum(
                "ihd,jhd->hij",
                q,
                k,
            )

            scores = (
                scores
                * module.scale
            )

            attn = torch.softmax(
                scores,
                dim=-1,
            )


            original_idx = order[s:e]

            key_robot_fraction = (
                robot_fraction[
                    original_idx
                ]
            )

            key_scene_fraction = (
                1.0
                - key_robot_fraction
            )


            # [H,L]
            attention_to_robot = (
                attn
                * key_robot_fraction[
                    None,
                    None,
                    :
                ]
            ).sum(dim=-1)

            attention_to_scene = (
                attn
                * key_scene_fraction[
                    None,
                    None,
                    :
                ]
            ).sum(dim=-1)


            candidate_robot = float(
                key_robot_fraction
                .mean()
                .item()
            )

            candidate_scene = float(
                key_scene_fraction
                .mean()
                .item()
            )


            real_local = (
                real_query[s:e]
            )

            real_idx = torch.where(
                real_local
            )[0]

            original_real_idx = (
                original_idx[
                    real_idx
                ]
            )

            query_robot_fraction = (
                robot_fraction[
                    original_real_idx
                ]
            ).double()

            query_scene_fraction = (
                1.0
                - query_robot_fraction
            )


            scene_weight = float(
                query_scene_fraction
                .sum()
                .item()
            )

            robot_weight = float(
                query_robot_fraction
                .sum()
                .item()
            )


            if scene_weight > 0:

                values = (
                    attention_to_robot[
                        :,
                        real_idx,
                    ]
                    .double()
                )

                scene_to_robot_attn_sum += (
                    values
                    * query_scene_fraction[
                        None,
                        :
                    ]
                ).sum(dim=1)

                scene_robot_candidate_sum += (
                    scene_weight
                    * candidate_robot
                )

                scene_query_weight_sum += (
                    scene_weight
                )


            if robot_weight > 0:

                values = (
                    attention_to_scene[
                        :,
                        real_idx,
                    ]
                    .double()
                )

                robot_to_scene_attn_sum += (
                    values
                    * query_robot_fraction[
                        None,
                        :
                    ]
                ).sum(dim=1)

                robot_scene_candidate_sum += (
                    robot_weight
                    * candidate_scene
                )

                robot_query_weight_sum += (
                    robot_weight
                )


        assert scene_query_weight_sum > 0
        assert robot_query_weight_sum > 0


        scene_to_robot = (
            scene_to_robot_attn_sum
            / scene_query_weight_sum
        )

        scene_robot_candidate = (
            scene_robot_candidate_sum
            / scene_query_weight_sum
        )


        robot_to_scene = (
            robot_to_scene_attn_sum
            / robot_query_weight_sum
        )

        robot_scene_candidate = (
            robot_scene_candidate_sum
            / robot_query_weight_sum
        )


        for h in range(H):

            head_rows.append({
                "layer_index":
                    layer_index,

                "layer":
                    layer_name,

                "channels":
                    C,

                "heads":
                    H,

                "head":
                    h,

                "tokens":
                    N,

                "pure_scene_tokens":
                    pure_scene,

                "pure_robot_tokens":
                    pure_robot,

                "mixed_tokens":
                    mixed,

                "scene_query_to_robot_attention":
                    float(
                        scene_to_robot[
                            h
                        ].item()
                    ),

                "scene_query_robot_candidate":
                    float(
                        scene_robot_candidate
                    ),

                "scene_to_robot_selectivity":
                    float(
                        scene_to_robot[
                            h
                        ].item()
                        - scene_robot_candidate
                    ),

                "robot_query_to_scene_attention":
                    float(
                        robot_to_scene[
                            h
                        ].item()
                    ),

                "robot_query_scene_candidate":
                    float(
                        robot_scene_candidate
                    ),

                "robot_to_scene_selectivity":
                    float(
                        robot_to_scene[
                            h
                        ].item()
                        - robot_scene_candidate
                    ),
            })


        layer_rows.append({
            "layer_index":
                layer_index,

            "layer":
                layer_name,

            "channels":
                C,

            "heads":
                H,

            "tokens":
                N,

            "pure_scene_tokens":
                pure_scene,

            "pure_robot_tokens":
                pure_robot,

            "mixed_tokens":
                mixed,

            "mean_robot_provenance":
                float(
                    robot_fraction
                    .mean()
                    .item()
                ),

            "min_robot_provenance":
                float(
                    robot_fraction
                    .min()
                    .item()
                ),

            "max_robot_provenance":
                float(
                    robot_fraction
                    .max()
                    .item()
                ),
        })

    return hook


handles = []

for idx, (name, module) in enumerate(
    attn_modules
):

    handles.append(
        module.register_forward_pre_hook(
            make_hook(
                idx,
                name,
            )
        )
    )


# ============================================================
# One validated sample.
# ============================================================

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


for h in handles:
    h.remove()


# ============================================================
# Call audit.
# ============================================================

called = {
    k: v
    for k, v in call_count.items()
    if v > 0
}

print()
print("=" * 80)
print("ATTENTION CALL AUDIT")
print("=" * 80)

print(
    "registered layers =",
    len(attn_modules),
)

print(
    "called layers =",
    len(called),
)

print(
    "call counts =",
    sorted(
        set(called.values())
    ),
)


head_df = pd.DataFrame(
    head_rows
)

layer_df = (
    pd.DataFrame(
        layer_rows
    )
    .drop_duplicates(
        subset=["layer_index"]
    )
    .sort_values(
        "layer_index"
    )
)


OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

head_df.to_csv(
    OUTDIR
    / "all_layer_head_attention.csv",
    index=False,
)

layer_df.to_csv(
    OUTDIR
    / "all_layer_token_provenance.csv",
    index=False,
)


# ============================================================
# First-layer reproduction against previous exact binary
# diagnostic.
# ============================================================

first_name = (
    "dynamics_predictor.predictor_model."
    "enc.enc0.block0.attn"
)

current_first = (
    head_df[
        head_df["layer"]
        == first_name
    ]
    .sort_values("head")
)


print()
print("=" * 80)
print("FIRST-LAYER PROVENANCE CHECK")
print("=" * 80)

print(
    current_first[
        [
            "head",
            "scene_query_to_robot_attention",
            "scene_query_robot_candidate",
            "scene_to_robot_selectivity",
            "robot_query_to_scene_attention",
            "robot_query_scene_candidate",
            "robot_to_scene_selectivity",
        ]
    ].to_string(
        index=False
    )
)


if FIRST_LAYER_REFERENCE.exists():

    ref = (
        pd.read_csv(
            FIRST_LAYER_REFERENCE
        )
        .sort_values("head")
    )

    comparisons = {
        "scene_query_to_robot_attention":
            "scene_query_to_robot_attention_mean",

        "scene_query_robot_candidate":
            "scene_query_robot_candidate_fraction_mean",

        "scene_to_robot_selectivity":
            "scene_query_robot_attention_minus_candidate",

        "robot_query_to_scene_attention":
            "robot_query_to_scene_attention_mean",

        "robot_query_scene_candidate":
            "robot_query_scene_candidate_fraction_mean",

        "robot_to_scene_selectivity":
            "robot_query_scene_attention_minus_candidate",
    }

    print()
    print(
        "First-layer differences vs previous smoke:"
    )

    for current_col, ref_col in (
        comparisons.items()
    ):

        d = np.abs(
            current_first[
                current_col
            ].to_numpy()
            -
            ref[
                ref_col
            ].to_numpy()
        )

        print(
            current_col,
            "max_abs_diff =",
            float(d.max()),
        )


# ============================================================
# Descriptive rankings only.
# No threshold.
# ============================================================

print()
print("=" * 80)
print("TOKEN PROVENANCE BY LAYER")
print("=" * 80)

print(
    layer_df[
        [
            "layer_index",
            "layer",
            "tokens",
            "pure_scene_tokens",
            "pure_robot_tokens",
            "mixed_tokens",
            "mean_robot_provenance",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 80)
print("TOP SCENE -> ROBOT SELECTIVE HEADS")
print("=" * 80)

print(
    head_df.sort_values(
        "scene_to_robot_selectivity",
        ascending=False,
    )[
        [
            "layer_index",
            "layer",
            "head",
            "tokens",
            "scene_query_to_robot_attention",
            "scene_query_robot_candidate",
            "scene_to_robot_selectivity",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print()
print("=" * 80)
print("TOP ROBOT -> SCENE SELECTIVE HEADS")
print("=" * 80)

print(
    head_df.sort_values(
        "robot_to_scene_selectivity",
        ascending=False,
    )[
        [
            "layer_index",
            "layer",
            "head",
            "tokens",
            "robot_query_to_scene_attention",
            "robot_query_scene_candidate",
            "robot_to_scene_selectivity",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print()
print(
    "SAVED =",
    OUTDIR,
)

print(
    "NO NEW MODEL INTERVENTION = True"
)

print(
    "NO BINARY THRESHOLD FOR POOLED TOKENS = True"
)

print(
    "PROVENANCE_FROM_POOLING_PARENT_INVERSE = True"
)

print(
    "OFFICIAL_CODE_MODIFIED = False"
)
