#!/usr/bin/env python3

import numpy as np

from diagnostics.behavior_object_mapping_smoke_task0006 import (
    ORIG_BUILD_FLOW_SAMPLE,
    ORIG_GRID_SAMPLE,
    ORIG_GATHER,
    build_flow_sample_with_object_ids,
    official_grid_indices,
    gather_with_object_ids,
)


def grid_sample_with_object_ids_exact(sample, grid_size, mode):
    """
    Run the OFFICIAL grid sampler exactly once.

    For train mode, replay the RNG state only to recover the exact
    scene indices selected by the official sampler, then restore the
    post-official RNG state so this diagnostic label does not change
    PointWorld's stochastic pipeline.
    """

    labels = sample.get("scene_object_id", None)

    if labels is None:
        return ORIG_GRID_SAMPLE(
            sample,
            grid_size=grid_size,
            mode=mode,
        )

    if labels.ndim != 2:
        raise RuntimeError(
            f"scene_object_id expected (T,N), got {labels.shape}"
        )

    N_before = sample["scene_flows"].shape[1]

    if labels.shape[1] != N_before:
        raise RuntimeError(
            "pre-grid object-id mismatch: "
            f"labels={labels.shape}, "
            f"scene={sample['scene_flows'].shape}"
        )

    # Save PRE-grid arrays because official sampler mutates sample in place.
    points_t0_before = (
        sample["scene_flows"][0]
        .copy()
    )
    labels_before = labels.copy()

    # State immediately before the official sampler.
    rng_before = np.random.get_state()

    # ----------------------------------------------------------
    # The actual PointWorld downsampling.
    # This is called EXACTLY ONCE.
    # ----------------------------------------------------------
    result = ORIG_GRID_SAMPLE(
        sample,
        grid_size=grid_size,
        mode=mode,
    )

    # State after BOTH official scene + robot grid selections.
    rng_after = np.random.get_state()

    # ----------------------------------------------------------
    # Replay ONLY to recover the exact scene indices.
    # Then restore rng_after so there is zero RNG side effect.
    # ----------------------------------------------------------
    np.random.set_state(rng_before)

    try:
        idx_scene = official_grid_indices(
            points_t0_before,
            grid_size=grid_size,
            mode=mode,
        )
    finally:
        np.random.set_state(rng_after)

    labels_after = (
        labels_before[:, idx_scene]
        .copy()
    )

    if labels_after.shape[1] != result["scene_flows"].shape[1]:
        raise RuntimeError(
            "post-grid count mismatch: "
            f"labels={labels_after.shape}, "
            f"scene={result['scene_flows'].shape}"
        )

    result["scene_object_id"] = labels_after

    return result
