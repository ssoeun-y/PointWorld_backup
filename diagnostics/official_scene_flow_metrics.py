#!/usr/bin/env python3

import numpy as np


def evaluate_3d_hpl_official(sf_pred, sf_gt):
    """
    Exact metric definition from the official HPLFlowNet CVPR 2019
    evaluation_utils.py::evaluate_3d.

    Args
    ----
    sf_pred : (N, 3), meters
    sf_gt   : (N, 3), meters

    Returns
    -------
    EPE3D, Acc3DS, Acc3DR, Outliers3D
    """

    sf_pred = np.asarray(sf_pred, dtype=np.float64)
    sf_gt = np.asarray(sf_gt, dtype=np.float64)

    assert sf_pred.ndim == 2 and sf_pred.shape[1] == 3
    assert sf_gt.shape == sf_pred.shape

    l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)

    EPE3D = l2_norm.mean()

    sf_norm = np.linalg.norm(sf_gt, axis=-1)

    # IMPORTANT:
    # official HPLFlowNet uses +1e-4 exactly.
    relative_err = l2_norm / (sf_norm + 1e-4)

    acc3d_strict = np.logical_or(
        l2_norm < 0.05,
        relative_err < 0.05,
    ).mean()

    acc3d_relax = np.logical_or(
        l2_norm < 0.10,
        relative_err < 0.10,
    ).mean()

    outlier = np.logical_or(
        l2_norm > 0.30,
        relative_err > 0.10,
    ).mean()

    return EPE3D, acc3d_strict, acc3d_relax, outlier


def pointworld_absolute_to_scene_flow(pred_xyz, gt_xyz):
    """
    Convert PointWorld absolute XYZ trajectories to scene-flow vectors.

    PointWorld:
      pred_xyz[t] = predicted absolute scene coordinates
      gt_xyz[t]   = ground-truth absolute scene coordinates

    HPLFlowNet metric expects displacement / scene-flow vectors.
    Therefore:
      flow(t) = xyz(t) - xyz(t0)
    """

    pred_xyz = np.asarray(pred_xyz, dtype=np.float64)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64)

    assert pred_xyz.shape == gt_xyz.shape
    assert pred_xyz.ndim == 3
    assert pred_xyz.shape[-1] == 3

    pred_flow = pred_xyz - pred_xyz[0:1]
    gt_flow = gt_xyz - gt_xyz[0:1]

    return pred_flow, gt_flow


def pointworld_official_valid_moved_mask(
    scene_context_mask,
    scene_exists,
    scene_supervised_mask,
    scene_moved_mask,
):
    """
    Follow PointWorld official evaluation contract.
    """

    context = np.asarray(scene_context_mask).astype(bool)
    exists = np.asarray(scene_exists).astype(bool)
    supervised = np.asarray(scene_supervised_mask).astype(bool)
    moved = np.asarray(scene_moved_mask).astype(bool)

    context = np.squeeze(context)
    exists = np.squeeze(exists)
    supervised = np.squeeze(supervised)
    moved = np.squeeze(moved)

    valid = (~context) & exists & supervised
    valid_moved = valid & moved

    return valid_moved


def evaluate_pointworld_sample(
    pred_xyz,
    gt_xyz,
    scene_context_mask,
    scene_exists,
    scene_supervised_mask,
    scene_moved_mask,
):
    """
    Per-sample PointWorld moved-point scene-flow evaluation.

    Returns metrics pooled over official valid+moved
    points across predicted timesteps.
    """

    pred_xyz = np.asarray(pred_xyz)
    gt_xyz = np.asarray(gt_xyz)

    pred_flow, gt_flow = pointworld_absolute_to_scene_flow(
        pred_xyz,
        gt_xyz,
    )

    valid_moved = pointworld_official_valid_moved_mask(
        scene_context_mask,
        scene_exists,
        scene_supervised_mask,
        scene_moved_mask,
    )

    # Ignore t0 because displacement is exactly zero by construction.
    pred_flow = pred_flow[1:]
    gt_flow = gt_flow[1:]
    valid_moved = valid_moved[1:]

    sf_pred = pred_flow[valid_moved]
    sf_gt = gt_flow[valid_moved]

    if len(sf_pred) == 0:
        return None

    epe, acc_s, acc_r, outlier = evaluate_3d_hpl_official(
        sf_pred,
        sf_gt,
    )

    return {
        "n_eval_points": int(len(sf_pred)),
        "EPE3D_m": float(epe),
        "EPE3D_mm": float(epe * 1000.0),
        "Acc3DS": float(acc_s),
        "Acc3DR": float(acc_r),
        "Outliers3D": float(outlier),
    }
