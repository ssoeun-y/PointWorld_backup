import os
import sys
import numpy as np
import pandas as pd
import torch

# ================================================================
# OFFICIAL MBNSF implementation
# audited commit:
# 192e483b8050500711cbf3d89a5424f23f29d7c6
# ================================================================
MBNSF_ROOT = "/home/viglab/cse/MBNSF_official"
sys.path.insert(0, MBNSF_ROOT)

from utils.sc_utils import spatial_consistency_loss

RANK_CSV = "diagnostics/results/behavior_radio_rigidity_rank.csv"

NPZ_ROOT = (
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237"
)

OUTDIR = (
    "diagnostics/results/"
    "behavior_radio_full554_mbnsf_vs_l2"
)
os.makedirs(OUTDIR, exist_ok=True)

D_THR = 0.03  # official MBNSF experiment default

inventory = pd.read_csv(RANK_CSV).copy()

# Use old file ONLY as inventory.
# SE3 values are not used anywhere in this analysis.
inventory = inventory[
    ["sample_key", "object_id", "n_points"]
].drop_duplicates("sample_key")

inventory = inventory.sort_values("sample_key").reset_index(drop=True)

print("INVENTORY =", len(inventory))

rows = []

for k, row in inventory.iterrows():

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

    # Same final-valid radio point identities
    mask = (obj[0] == object_id) & valid[-1]

    x0 = gt[0, mask]
    xgt = gt[-1, mask]
    xpred = pred[-1, mask]

    n = int(mask.sum())

    assert n == int(row["n_points"])
    assert n < 3000  # no MBNSF random subsampling

    # ------------------------------------------------------------
    # 1. ACTUAL stored official PointWorld sequence-level L2
    # ------------------------------------------------------------
    official_scene_l2_mm = (
        float(d["official_l2"].item()) * 1000.0
    )

    # ------------------------------------------------------------
    # 2. Time-aligned positional control:
    #    same radio points, same final timestep as MBNSF rigidity
    #    NOT called an official metric.
    # ------------------------------------------------------------
    radio_t10_l2_mm = float(
        np.linalg.norm(
            xpred.astype(np.float64) -
            xgt.astype(np.float64),
            axis=1
        ).mean() * 1000.0
    )

    # ------------------------------------------------------------
    # 3. OFFICIAL MBNSF spatial consistency implementation
    # ------------------------------------------------------------
    src = torch.from_numpy(x0).float().unsqueeze(0)
    tgt_gt = torch.from_numpy(xgt).float().unsqueeze(0)
    tgt_pred = torch.from_numpy(xpred).float().unsqueeze(0)

    with torch.no_grad():

        gt_loss = float(
            spatial_consistency_loss(
                src,
                tgt_gt,
                d_thre=D_THR
            ).item()
        )

        pred_loss = float(
            spatial_consistency_loss(
                src,
                tgt_pred,
                d_thre=D_THR
            ).item()
        )

    # Exact inverse of official loss:
    # loss = -log(score)
    gt_score = float(np.exp(-gt_loss))
    pred_score = float(np.exp(-pred_loss))

    rows.append({
        "sample_key": key,
        "object_id": object_id,
        "n_points": n,

        "official_scene_l2_mm":
            official_scene_l2_mm,

        "radio_t10_l2_mm":
            radio_t10_l2_mm,

        "mbnsf_gt_sc_loss":
            gt_loss,

        "mbnsf_pred_sc_loss":
            pred_loss,

        "mbnsf_gt_sc_score":
            gt_score,

        "mbnsf_pred_sc_score":
            pred_score,
    })

    if (k + 1) % 50 == 0:
        print(
            f"processed {k+1}/{len(inventory)}"
        )


full = pd.DataFrame(rows)

assert len(full) == len(inventory)

full_csv = os.path.join(
    OUTDIR,
    "radio_full554_mbnsf_vs_l2.csv"
)

full.to_csv(full_csv, index=False)


def spearman(a, b):

    ra = pd.Series(a).rank(
        method="average"
    ).to_numpy(float)

    rb = pd.Series(b).rank(
        method="average"
    ).to_numpy(float)

    return float(
        np.corrcoef(ra, rb)[0, 1]
    )


# ================================================================
# Correlation: descriptive only
# ================================================================

rho_official_score = spearman(
    full["official_scene_l2_mm"],
    full["mbnsf_pred_sc_score"]
)

rho_t10_score = spearman(
    full["radio_t10_l2_mm"],
    full["mbnsf_pred_sc_score"]
)

rho_official_loss = spearman(
    full["official_scene_l2_mm"],
    full["mbnsf_pred_sc_loss"]
)

rho_t10_loss = spearman(
    full["radio_t10_l2_mm"],
    full["mbnsf_pred_sc_loss"]
)


# ================================================================
# Closest-L2 pair audit
#
# IMPORTANT:
# Pairs are selected ONLY by positional L2 ordering.
# MBNSF score is never used to select pairs.
# No arbitrary "same-L2" threshold yet.
# ================================================================

def make_adjacent_pairs(df, l2_col):

    s = df.sort_values(
        l2_col
    ).reset_index(drop=True)

    pairs = []

    for i in range(len(s) - 1):

        a = s.iloc[i]
        b = s.iloc[i + 1]

        pairs.append({
            "sample_a": a["sample_key"],
            "sample_b": b["sample_key"],

            "l2_a_mm":
                float(a[l2_col]),

            "l2_b_mm":
                float(b[l2_col]),

            "l2_gap_mm":
                abs(
                    float(a[l2_col]) -
                    float(b[l2_col])
                ),

            "mbnsf_score_a":
                float(a["mbnsf_pred_sc_score"]),

            "mbnsf_score_b":
                float(b["mbnsf_pred_sc_score"]),

            "mbnsf_score_gap":
                abs(
                    float(a["mbnsf_pred_sc_score"]) -
                    float(b["mbnsf_pred_sc_score"])
                ),

            "mbnsf_loss_a":
                float(a["mbnsf_pred_sc_loss"]),

            "mbnsf_loss_b":
                float(b["mbnsf_pred_sc_loss"]),

            "mbnsf_loss_gap":
                abs(
                    float(a["mbnsf_pred_sc_loss"]) -
                    float(b["mbnsf_pred_sc_loss"])
                ),

            "n_points_a":
                int(a["n_points"]),

            "n_points_b":
                int(b["n_points"]),
        })

    out = pd.DataFrame(pairs)

    # Selection/ranking ONLY by L2 gap
    out = out.sort_values(
        "l2_gap_mm"
    ).reset_index(drop=True)

    return out


pairs_official = make_adjacent_pairs(
    full,
    "official_scene_l2_mm"
)

pairs_t10 = make_adjacent_pairs(
    full,
    "radio_t10_l2_mm"
)


official_pairs_csv = os.path.join(
    OUTDIR,
    "closest_pairs_official_scene_l2.csv"
)

t10_pairs_csv = os.path.join(
    OUTDIR,
    "closest_pairs_radio_t10_l2.csv"
)

pairs_official.to_csv(
    official_pairs_csv,
    index=False
)

pairs_t10.to_csv(
    t10_pairs_csv,
    index=False
)


print("\n" + "=" * 120)
print("FULL 554 OFFICIAL MBNSF vs POSITIONAL L2")
print("=" * 120)

print("N =", len(full))

print("\nGT MBNSF SCORE")
print(
    "min    =",
    full["mbnsf_gt_sc_score"].min()
)
print(
    "median =",
    full["mbnsf_gt_sc_score"].median()
)
print(
    "max    =",
    full["mbnsf_gt_sc_score"].max()
)

print("\nPRED MBNSF SCORE")
print(
    full["mbnsf_pred_sc_score"]
    .describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    )
)

print("\nCORRELATION")
print(
    "Spearman(official scene L2, MBNSF score) =",
    rho_official_score
)
print(
    "Spearman(official scene L2, MBNSF loss)  =",
    rho_official_loss
)
print(
    "Spearman(radio t10 L2, MBNSF score)      =",
    rho_t10_score
)
print(
    "Spearman(radio t10 L2, MBNSF loss)       =",
    rho_t10_loss
)


def print_closest(title, p, n=15):

    print("\n" + "=" * 120)
    print(title)
    print(
        "Pairs below are ordered ONLY by L2 gap."
    )
    print("=" * 120)

    for i, r in p.head(n).iterrows():

        print(
            f"{i+1:2d}. "
            f"L2 gap={r['l2_gap_mm']:.6f} mm | "
            f"MBNSF score gap={r['mbnsf_score_gap']:.6f} | "
            f"A score={r['mbnsf_score_a']:.6f}, "
            f"B score={r['mbnsf_score_b']:.6f} | "
            f"n=({int(r['n_points_a'])},"
            f"{int(r['n_points_b'])})"
        )

        print(
            "    A =",
            r["sample_a"]
        )
        print(
            "    B =",
            r["sample_b"]
        )


print_closest(
    "15 CLOSEST PAIRS BY ACTUAL OFFICIAL SCENE L2",
    pairs_official
)

print_closest(
    "15 CLOSEST PAIRS BY RADIO t10 POSITIONAL L2",
    pairs_t10
)


print("\nFILES")
print(full_csv)
print(official_pairs_csv)
print(t10_pairs_csv)

print("\nFULL_CSV_EXISTS =",
      os.path.exists(full_csv))

print("OFFICIAL_PAIR_CSV_EXISTS =",
      os.path.exists(official_pairs_csv))

print("T10_PAIR_CSV_EXISTS =",
      os.path.exists(t10_pairs_csv))

print("MBNSF_FULL554_L2_AUDIT_SUCCESS = True")
