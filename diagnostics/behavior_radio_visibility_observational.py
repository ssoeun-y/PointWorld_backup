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

# Already validated official-visibility copy.
from diagnostics.droid_visibility_quickcheck import (
    get_visibility_count,
    squeeze_bool,
)


RADIO_CSV = Path(
    "diagnostics/results/"
    "behavior_radio_full554_mbnsf_vs_l2/"
    "radio_full554_mbnsf_vs_l2.csv"
)

NPZ_ROOT = Path(
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237"
)

OUTDIR = Path(
    "diagnostics/results/"
    "behavior_radio_visibility_observational"
)


# Existing validated runtime-only object-ID preservation.
dlmod.build_flow_sample = objmap.build_flow_sample_with_object_ids
decmod.build_flow_sample = objmap.build_flow_sample_with_object_ids
pmod.grid_sample_transform = objmap.grid_sample_with_object_ids
pmod.gather = objmap.gather_with_object_ids


args = parse_args()

radio = pd.read_csv(RADIO_CSV)

assert len(radio) == 554
assert radio["sample_key"].nunique() == 554

radio = radio.set_index("sample_key")

tester = Tester(args)
tester.model.eval()

dl, _ = tester._build_eval_loader(
    "test",
    enable_mask=False,
)

sample_rows = []
point_rows = []

found = set()


with torch.no_grad():

    for batch in dl:

        key_raw = batch["__key__"]

        if isinstance(key_raw, (list, tuple)):
            key = str(key_raw[0])
        else:
            key = str(key_raw)

        if key not in radio.index:
            continue

        row = radio.loc[key]

        object_id = int(row["object_id"])
        expected_n = int(row["n_points"])

        # ---------------------------------------------------------
        # Keep current object IDs on CPU.
        # ---------------------------------------------------------
        object_ids = batch["scene_object_id"]

        if isinstance(object_ids, torch.Tensor):
            object_ids_np = (
                object_ids.detach().cpu().numpy()
            )
        else:
            object_ids_np = np.asarray(object_ids)

        assert object_ids_np.ndim == 3
        assert object_ids_np.shape[0] == 1

        # ---------------------------------------------------------
        # Move official input tensors to GPU.
        # No PointWorld dynamics forward is performed.
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
            batch_model["scene_context_mask"]
        )
        exists = squeeze_bool(
            batch_model["scene_exists"]
        )
        supervised = squeeze_bool(
            batch_model["scene_supervised_mask"]
        )

        valid = (
            (~context)
            & exists
            & supervised
        )

        current_valid_np = (
            valid[0]
            .detach()
            .cpu()
            .numpy()
        )

        # ---------------------------------------------------------
        # Load the exact already-validated prediction/GT capture.
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
                z["official_l2_exact"].item()
            )

            assert bool(
                z["temporal_object_id_exact"].item()
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

        # Exact current/raw vs previous-capture alignment.
        assert np.array_equal(
            object_ids_np[0],
            old_obj,
        ), key

        assert np.array_equal(
            current_valid_np,
            old_valid,
        ), key

        assert gt.shape == pred.shape

        # ---------------------------------------------------------
        # EXACT SAME radio population used by previous 554 analysis:
        # object ID at t0 AND final-valid at t10.
        # ---------------------------------------------------------
        radio_mask = (
            (object_ids_np[0, 0] == object_id)
            & current_valid_np[-1]
        )

        global_idx = np.where(
            radio_mask
        )[0]

        assert len(global_idx) == expected_n, (
            key,
            len(global_idx),
            expected_n,
        )

        # ---------------------------------------------------------
        # Official PointWorld visibility decision copied from
        # droid_visibility_quickcheck.py.
        #
        # Visibility is from the initial observation.
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

        radio_vis = vis0[global_idx]

        assert np.isin(
            radio_vis,
            [0, 1, 2],
        ).all()

        # ---------------------------------------------------------
        # Existing stored prediction: t10 L2 per radio point.
        # ---------------------------------------------------------
        point_l2_mm = (
            np.linalg.norm(
                pred[-1, global_idx]
                - gt[-1, global_idx],
                axis=-1,
            )
            * 1000.0
        )

        radio_t10_l2_mm = float(
            point_l2_mm.mean()
        )

        # Must reproduce the already-saved 554 radio t10 result.
        old_radio_t10_l2_mm = float(
            row["radio_t10_l2_mm"]
        )

        assert np.isclose(
            radio_t10_l2_mm,
            old_radio_t10_l2_mm,
            rtol=0.0,
            atol=1e-9,
        ), (
            key,
            radio_t10_l2_mm,
            old_radio_t10_l2_mm,
        )

        n0 = int(
            (radio_vis == 0).sum()
        )
        n1 = int(
            (radio_vis == 1).sum()
        )
        n2 = int(
            (radio_vis == 2).sum()
        )

        assert n0 + n1 + n2 == expected_n

        def mean_for_view(v):
            m = radio_vis == v
            if not m.any():
                return np.nan
            return float(
                point_l2_mm[m].mean()
            )

        sample_rows.append({
            "sample_key": key,
            "object_id": object_id,
            "n_points": expected_n,

            "n_view0": n0,
            "n_view1": n1,
            "n_view2": n2,

            "frac_view0": n0 / expected_n,
            "frac_view1": n1 / expected_n,
            "frac_view2": n2 / expected_n,

            "mean_visible_views":
                float(radio_vis.mean()),

            "radio_t10_l2_mm":
                radio_t10_l2_mm,

            "view0_t10_l2_mm":
                mean_for_view(0),

            "view1_t10_l2_mm":
                mean_for_view(1),

            "view2_t10_l2_mm":
                mean_for_view(2),
        })

        for gi, vv, ee in zip(
            global_idx,
            radio_vis,
            point_l2_mm,
        ):
            point_rows.append({
                "sample_key": key,
                "object_id": object_id,
                "global_idx": int(gi),
                "visible_views": int(vv),
                "t10_l2_mm": float(ee),
            })

        found.add(key)

        if len(found) % 50 == 0:
            print(
                f"processed {len(found)}/554"
            )


assert len(found) == 554, (
    len(found),
    sorted(
        set(radio.index) - found
    )[:20],
)

sample_df = pd.DataFrame(
    sample_rows
).sort_values(
    "sample_key"
)

point_df = pd.DataFrame(
    point_rows
).sort_values(
    ["sample_key", "global_idx"]
)

assert len(sample_df) == 554


OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)

sample_path = (
    OUTDIR
    / "radio554_visibility_per_sample.csv"
)

point_path = (
    OUTDIR
    / "radio554_visibility_per_point.csv"
)

sample_df.to_csv(
    sample_path,
    index=False,
)

point_df.to_csv(
    point_path,
    index=False,
)


print()
print("=" * 72)
print("RADIO VISIBILITY AUDIT")
print("=" * 72)

print("samples =", len(sample_df))
print("points  =", len(point_df))

for v in [0, 1, 2]:

    q = point_df[
        point_df["visible_views"] == v
    ]

    print(
        f"views={v}: "
        f"n={len(q)}, "
        f"mean_t10_L2="
        f"{q['t10_l2_mm'].mean():.6f} mm"
    )

print()
print(
    "mean frac_view0 =",
    sample_df["frac_view0"].mean(),
)

print(
    "mean frac_view1 =",
    sample_df["frac_view1"].mean(),
)

print(
    "mean frac_view2 =",
    sample_df["frac_view2"].mean(),
)

print()
print("SAVED =", sample_path)
print("SAVED =", point_path)
print("ALL_554_EXACT_ALIGNMENT = True")
print("NO NEW MODEL INFERENCE = True")
