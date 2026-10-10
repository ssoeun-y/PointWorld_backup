#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from arguments import parse_args
from evaluation.tester import Tester

import dataset_components.dataloader as dlmod
import dataset_components.decoders as decmod
import dataset_components.pipeline as pmod

import diagnostics.behavior_object_mapping_smoke as objmap

# Already validated copy of official PointWorld visibility decision.
from diagnostics.droid_visibility_quickcheck import (
    get_visibility_count,
    squeeze_bool,
)


INVENTORY_CSV = Path(
    "diagnostics/results/"
    "behavior_object_distance_motion_full578_20260810_114615/"
    "per_object.csv"
)

NPZ_ROOT = Path(
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237"
)

OUTDIR = Path(
    "diagnostics/results/"
    "behavior_allobject_visibility_within_object"
)


# Existing validated runtime-only object-ID preservation path.
dlmod.build_flow_sample = objmap.build_flow_sample_with_object_ids
decmod.build_flow_sample = objmap.build_flow_sample_with_object_ids
pmod.grid_sample_transform = objmap.grid_sample_with_object_ids
pmod.gather = objmap.gather_with_object_ids


def get_key(batch):
    x = batch["__key__"]

    if isinstance(x, (list, tuple)):
        x = x[0]

    if isinstance(x, bytes):
        x = x.decode("utf-8")

    return str(x)


args = parse_args()

inventory = pd.read_csv(INVENTORY_CSV)

assert len(inventory) == 2314
assert inventory["key"].nunique() == 578

# One row per (sequence, object).
assert not inventory.duplicated(
    ["key", "object_id"]
).any()

by_key = {
    key: g.copy()
    for key, g in inventory.groupby(
        "key",
        sort=False,
    )
}


tester = Tester(args)
tester.model.eval()

dl, _ = tester._build_eval_loader(
    "test",
    enable_mask=False,
)

rows = []
found_keys = set()


with torch.no_grad():

    for batch in dl:

        key = get_key(batch)

        if key not in by_key:
            continue

        # ---------------------------------------------------------
        # 1. Current object IDs from raw BEHAVIOR sample.
        # ---------------------------------------------------------
        object_ids = batch["scene_object_id"]

        if isinstance(object_ids, torch.Tensor):
            object_ids_np = (
                object_ids
                .detach()
                .cpu()
                .numpy()
            )
        else:
            object_ids_np = np.asarray(
                object_ids
            )

        assert object_ids_np.ndim == 3
        assert object_ids_np.shape[0] == 1

        # ---------------------------------------------------------
        # 2. Move official inputs to GPU.
        #    No PointWorld dynamics forward.
        # ---------------------------------------------------------
        batch_model = {
            k: (
                v.to(
                    tester.device,
                    non_blocking=True,
                )
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
            if k != "scene_object_id"
        }

        context = squeeze_bool(
            batch_model[
                "scene_context_mask"
            ]
        )

        exists = squeeze_bool(
            batch_model[
                "scene_exists"
            ]
        )

        supervised = squeeze_bool(
            batch_model[
                "scene_supervised_mask"
            ]
        )

        current_valid = (
            (~context)
            & exists
            & supervised
        )[0].detach().cpu().numpy()

        # ---------------------------------------------------------
        # 3. Load already-validated stored prediction / GT.
        # ---------------------------------------------------------
        npz_path = (
            NPZ_ROOT
            / (key.replace(":", "_") + ".npz")
        )

        assert npz_path.exists(), npz_path

        with np.load(
            npz_path,
            allow_pickle=True,
        ) as z:

            assert bool(
                z[
                    "official_l2_exact"
                ].item()
            )

            assert bool(
                z[
                    "temporal_object_id_exact"
                ].item()
            )

            old_obj = z[
                "scene_object_id"
            ].copy()

            old_valid = z[
                "official_valid_mask"
            ].copy()

            gt = z[
                "gt_scene_flows"
            ].astype(np.float64)

            pred = z[
                "pred_scene_flows"
            ].astype(np.float64)

        # Current raw sample must be exactly aligned with
        # the previously validated prediction capture.
        assert np.array_equal(
            object_ids_np[0],
            old_obj,
        ), key

        assert np.array_equal(
            current_valid,
            old_valid,
        ), key

        assert gt.shape == pred.shape

        # ---------------------------------------------------------
        # 4. Official initial-observation visibility.
        #    0 / 1 / 2 cameras contributing visual conditioning.
        # ---------------------------------------------------------
        vis0 = (
            get_visibility_count(
                tester.model,
                batch_model,
            )[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        assert vis0.shape[0] == gt.shape[1]

        # Stored prediction t10 point-wise L2.
        t10_l2_mm = (
            np.linalg.norm(
                pred[-1] - gt[-1],
                axis=-1,
            )
            * 1000.0
        )

        # ---------------------------------------------------------
        # 5. Every object in this sequence.
        # ---------------------------------------------------------
        for _, objrow in by_key[key].iterrows():

            object_id = int(
                objrow["object_id"]
            )

            object_name = str(
                objrow["object_name"]
            )

            expected_final_n = int(
                objrow[
                    "final_valid_count"
                ]
            )

            # SAME population convention:
            # object identity at t0 + final-valid at t10.
            obj_mask = (
                (
                    object_ids_np[
                        0, 0
                    ]
                    == object_id
                )
                & current_valid[-1]
            )

            idx = np.where(
                obj_mask
            )[0]

            assert len(idx) == expected_final_n, (
                key,
                object_id,
                len(idx),
                expected_final_n,
            )

            if len(idx) == 0:
                continue

            vv = vis0[idx]
            ee = t10_l2_mm[idx]

            assert np.isin(
                vv,
                [0, 1, 2],
            ).all()

            m0 = vv == 0
            m1 = vv == 1
            m2 = vv == 2
            m_visible = vv >= 1

            n0 = int(m0.sum())
            n1 = int(m1.sum())
            n2 = int(m2.sum())
            nv = int(
                m_visible.sum()
            )

            def mean_or_nan(mask):
                if not mask.any():
                    return np.nan

                return float(
                    ee[mask].mean()
                )

            view0_l2 = mean_or_nan(m0)
            visible_l2 = mean_or_nan(
                m_visible
            )

            # Only defined when this SAME object has
            # both invisible and visible points.
            if n0 > 0 and nv > 0:
                gap = (
                    view0_l2
                    - visible_l2
                )
            else:
                gap = np.nan

            rows.append({
                "key":
                    key,

                "object_id":
                    object_id,

                "object_name":
                    object_name,

                "is_moved_object":
                    bool(
                        objrow[
                            "is_moved_object"
                        ]
                    ),

                "final_valid_count":
                    len(idx),

                "n_view0":
                    n0,

                "n_view1":
                    n1,

                "n_view2":
                    n2,

                "n_visible":
                    nv,

                "frac_view0":
                    n0 / len(idx),

                "frac_view1":
                    n1 / len(idx),

                "frac_view2":
                    n2 / len(idx),

                "mean_visible_views":
                    float(vv.mean()),

                "object_t10_l2_mm":
                    float(
                        ee.mean()
                    ),

                "view0_t10_l2_mm":
                    view0_l2,

                "visible_t10_l2_mm":
                    visible_l2,

                "gap_view0_minus_visible_mm":
                    gap,
            })

        found_keys.add(key)

        if len(found_keys) % 50 == 0:
            print(
                f"processed "
                f"{len(found_keys)}/578"
            )


assert len(found_keys) == 578

result = pd.DataFrame(rows)

assert len(result) == 2314, len(result)


OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

out_csv = (
    OUTDIR
    / "all2314_object_visibility.csv"
)

result.to_csv(
    out_csv,
    index=False,
)


# -------------------------------------------------------------
# Descriptive within-object summary.
# No magnitude threshold.
# -------------------------------------------------------------
def summarize(name, df):

    paired = df[
        df[
            "gap_view0_minus_visible_mm"
        ].notna()
    ].copy()

    gap = paired[
        "gap_view0_minus_visible_mm"
    ].to_numpy()

    worse = int(
        (gap > 0).sum()
    )

    better = int(
        (gap < 0).sum()
    )

    ties = int(
        (gap == 0).sum()
    )

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    print(
        "object instances =",
        len(df),
    )

    print(
        "paired instances =",
        len(paired),
    )

    print(
        "view0 worse =",
        f"{worse}/{len(paired)}"
        if len(paired)
        else "NA",
    )

    print(
        "view0 better =",
        f"{better}/{len(paired)}"
        if len(paired)
        else "NA",
    )

    print(
        "ties =",
        ties,
    )

    if len(paired):

        print(
            "median gap "
            "[view0-visible] mm =",
            float(
                np.median(gap)
            ),
        )

        print(
            "mean gap "
            "[view0-visible] mm =",
            float(
                np.mean(gap)
            ),
        )

        print(
            "mean frac_view0 =",
            float(
                paired[
                    "frac_view0"
                ].mean()
            ),
        )


summarize(
    "ALL OBJECTS",
    result,
)

summarize(
    "MOVED OBJECTS",
    result[
        result[
            "is_moved_object"
        ]
    ],
)

summarize(
    "NON-MOVED OBJECTS",
    result[
        ~result[
            "is_moved_object"
        ]
    ],
)


print()
print("SAVED =", out_csv)
print("ALL_578_EXACT_ALIGNMENT = True")
print("NO NEW MODEL INFERENCE = True")
