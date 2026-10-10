import os
import sys
import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------------
# EXACT external official repository
# MBNSF commit audited separately:
# 192e483b8050500711cbf3d89a5424f23f29d7c6
# ------------------------------------------------------------------
MBNSF_ROOT = "/home/viglab/cse/MBNSF_official"
sys.path.insert(0, MBNSF_ROOT)

from utils.sc_utils import spatial_consistency_loss

RANK_CSV = "diagnostics/results/behavior_radio_rigidity_rank.csv"

NPZ_ROOT = (
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237"
)

PREV_CSV = (
    "diagnostics/results/"
    "behavior_radio_top9_published_isometry/"
    "top9_published_isometry.csv"
)

OUTDIR = (
    "diagnostics/results/"
    "behavior_radio_top9_mbnsf_official"
)
os.makedirs(OUTDIR, exist_ok=True)

# MBNSF actual default experiment setting:
# scene_flow_estimation/mbnsf.py --sc_dist_thresh default=0.03
D_THR = 0.03

rank_df = pd.read_csv(RANK_CSV).head(9).copy()

if os.path.exists(PREV_CSV):
    prev_df = pd.read_csv(PREV_CSV)
else:
    prev_df = None

rows = []

for rank_idx, row in rank_df.iterrows():

    key = str(row["sample_key"])
    object_id = int(row["object_id"])

    path = os.path.join(
        NPZ_ROOT,
        key.replace(":", "_") + ".npz"
    )

    assert os.path.exists(path), path

    d = np.load(path, allow_pickle=True)

    assert bool(d["official_l2_exact"].item())
    assert bool(d["temporal_object_id_exact"].item())

    gt = d["gt_scene_flows"]
    pred = d["pred_scene_flows"]
    obj = d["scene_object_id"]
    valid = d["official_valid_mask"]

    # Exactly the same final-valid object point IDs
    mask = (obj[0] == object_id) & valid[-1]

    x0 = gt[0, mask]
    xgt = gt[-1, mask]
    xpred = pred[-1, mask]

    n = int(mask.sum())

    assert n == int(row["n_points"])
    assert n < 3000  # guarantees no official random subsampling

    # Match official MBNSF data precision:
    # their scene flow code converts point clouds via .float()
    src = torch.from_numpy(x0).float().unsqueeze(0)
    tgt_gt = torch.from_numpy(xgt).float().unsqueeze(0)
    tgt_pred = torch.from_numpy(xpred).float().unsqueeze(0)

    # --------------------------------------------------------------
    # NO reimplementation:
    # directly call official MBNSF function
    # --------------------------------------------------------------
    with torch.no_grad():
        gt_sc_loss = spatial_consistency_loss(
            src,
            tgt_gt,
            d_thre=D_THR
        )

        pred_sc_loss = spatial_consistency_loss(
            src,
            tgt_pred,
            d_thre=D_THR
        )

    gt_loss = float(gt_sc_loss.item())
    pred_loss = float(pred_sc_loss.item())

    # This is only the exact inverse transformation of official loss:
    # official loss = -log(sc_score)
    # therefore score = exp(-loss)
    gt_score = float(np.exp(-gt_loss))
    pred_score = float(np.exp(-pred_loss))

    rows.append({
        "rank": rank_idx + 1,
        "sample_key": key,
        "object_id": object_id,
        "n_points": n,

        "mbnsf_gt_sc_loss": gt_loss,
        "mbnsf_pred_sc_loss": pred_loss,

        "mbnsf_gt_sc_score": gt_score,
        "mbnsf_pred_sc_score": pred_score,

        # Existing diagnostic: CROSS-CHECK ONLY
        "se3_residual_mean_mm":
            float(row["pred_mean_mm"]),
    })


out = pd.DataFrame(rows)

# Compare against previous hand implementation ONLY as audit.
# It is NOT used as the primary result.
if prev_df is not None:

    merged = out.merge(
        prev_df[
            [
                "sample_key",
                "gt_isometry_eq10",
                "pred_isometry_eq10"
            ]
        ],
        on="sample_key",
        how="left"
    )

    merged["gt_score_absdiff_vs_previous"] = np.abs(
        merged["mbnsf_gt_sc_score"] -
        merged["gt_isometry_eq10"]
    )

    merged["pred_score_absdiff_vs_previous"] = np.abs(
        merged["mbnsf_pred_sc_score"] -
        merged["pred_isometry_eq10"]
    )

    out = merged


csv_path = os.path.join(
    OUTDIR,
    "top9_mbnsf_official.csv"
)

out.to_csv(csv_path, index=False)


def spearman(a, b):
    ra = pd.Series(a).rank(method="average").to_numpy(float)
    rb = pd.Series(b).rank(method="average").to_numpy(float)
    return float(np.corrcoef(ra, rb)[0, 1])


rho_score_se3 = spearman(
    out["mbnsf_pred_sc_score"],
    out["se3_residual_mean_mm"]
)

rho_loss_se3 = spearman(
    out["mbnsf_pred_sc_loss"],
    out["se3_residual_mean_mm"]
)

print("=" * 120)
print("MBNSF OFFICIAL IMPLEMENTATION CROSS-CHECK")
print("official repo =", MBNSF_ROOT)
print("d_thre =", D_THR)
print("=" * 120)

for _, r in out.iterrows():

    print(
        f"rank={int(r['rank']):2d} "
        f"n={int(r['n_points']):4d} | "
        f"GT loss={r['mbnsf_gt_sc_loss']:.9f} "
        f"GT score={r['mbnsf_gt_sc_score']:.9f} | "
        f"Pred loss={r['mbnsf_pred_sc_loss']:.6f} "
        f"Pred score={r['mbnsf_pred_sc_score']:.6f} | "
        f"SE3={r['se3_residual_mean_mm']:.3f}mm"
    )

print("\n" + "=" * 120)
print("INDEPENDENT DIRECTION CHECK")
print(
    "Spearman(MBNSF score, SE3 residual) =",
    rho_score_se3,
    "(expected negative)"
)
print(
    "Spearman(MBNSF loss, SE3 residual)  =",
    rho_loss_se3,
    "(expected positive)"
)

if prev_df is not None:
    print("\nIMPLEMENTATION AUDIT")
    print(
        "max |official GT score - previous Eq10| =",
        out["gt_score_absdiff_vs_previous"].max()
    )
    print(
        "max |official Pred score - previous Eq10| =",
        out["pred_score_absdiff_vs_previous"].max()
    )

print("\nCSV =", csv_path)
print("CSV_EXISTS =", os.path.exists(csv_path))
print("MBNSF_OFFICIAL_TOP9_SUCCESS = True")
