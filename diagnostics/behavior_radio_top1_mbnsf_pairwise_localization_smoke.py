import os
import sys
import numpy as np
import pandas as pd
import torch

# ================================================================
# FIXED / PRE-AUDITED SETTINGS
# ================================================================
MBNSF_ROOT = "/home/viglab/cse/MBNSF_official"
D_THRE = 0.03
MAX_POINTS = 3000

FULL_CSV = (
    "diagnostics/results/"
    "behavior_radio_full554_mbnsf_vs_l2/"
    "radio_full554_mbnsf_vs_l2.csv"
)

NPZ_ROOT = (
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237"
)

OUTDIR = (
    "diagnostics/results/"
    "behavior_radio_top1_mbnsf_pairwise_localization_smoke"
)
os.makedirs(OUTDIR, exist_ok=True)

# Load the exact official MBNSF file directly.
# This avoids a namespace collision with PointWorld's own "utils".
import importlib.util

SC_UTILS_PATH = os.path.join(
    MBNSF_ROOT,
    "utils",
    "sc_utils.py",
)

assert os.path.exists(SC_UTILS_PATH), SC_UTILS_PATH

spec = importlib.util.spec_from_file_location(
    "mbnsf_official_sc_utils",
    SC_UTILS_PATH,
)
mbnsf_sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mbnsf_sc)

spatial_consistency_loss = (
    mbnsf_sc.spatial_consistency_loss
)
spatial_consistency_score = (
    mbnsf_sc.spatial_consistency_score
)
power_iteration = (
    mbnsf_sc.power_iteration
)

# ================================================================
# 1. SELECT TOP-1 SEVERE CASE OBJECTIVELY
#    By already-computed official MBNSF score only.
# ================================================================
df = pd.read_csv(FULL_CSV)

assert len(df) == 554

row = df.loc[
    df["mbnsf_pred_sc_score"].idxmin()
]

key = str(row["sample_key"])
object_id = int(row["object_id"])
old_pred_score = float(row["mbnsf_pred_sc_score"])

print("=== SELECTED TOP-1 ===")
print("sample_key =", key)
print("object_id  =", object_id)
print("old MBNSF pred score =", old_pred_score)

# Audit against previously known severe case.
EXPECTED_KEY = "task-0000_episode_00002530-300:311"
assert key == EXPECTED_KEY, (key, EXPECTED_KEY)

# ================================================================
# 2. LOAD EXACT SAME POINT POPULATION
# ================================================================
path = os.path.join(
    NPZ_ROOT,
    key.replace(":", "_") + ".npz"
)
assert os.path.exists(path), path

d = np.load(path, allow_pickle=True)

assert bool(d["official_l2_exact"].item())
assert bool(d["temporal_object_id_exact"].item())

gt_np = d["gt_scene_flows"].astype(np.float64)
pred_np = d["pred_scene_flows"].astype(np.float64)

obj = d["scene_object_id"]
valid = d["official_valid_mask"]

assert gt_np.shape == pred_np.shape
assert gt_np.shape[0] == 11

# EXACT SAME fixed population as previous t10 MBNSF analysis.
mask = (
    (obj[0] == object_id)
    & valid[-1]
)

global_idx = np.where(mask)[0]

x0_np = gt_np[0, mask]
gt10_np = gt_np[-1, mask]
pred10_np = pred_np[-1, mask]

n = len(x0_np)

print()
print("=== POINT POPULATION ===")
print("n_points =", n)
print("max_points =", MAX_POINTS)
print("random subsampling needed =", n > MAX_POINTS)

assert n < MAX_POINTS

# ================================================================
# 3. EXACT OFFICIAL MBNSF SCORE REPRODUCTION
# ================================================================
# EXACTLY match previous full554 MBNSF implementation:
# torch.from_numpy(...).float().unsqueeze(0)
x0 = torch.from_numpy(x0_np).float().unsqueeze(0)
gt10 = torch.from_numpy(gt10_np).float().unsqueeze(0)
pred10 = torch.from_numpy(pred10_np).float().unsqueeze(0)

def official_mbnsf(src, tgt):
    loss = spatial_consistency_loss(
        src,
        tgt,
        d_thre=D_THRE,
        max_points=MAX_POINTS,
    )
    loss_value = float(loss.item())

    # EXACTLY same score reconstruction as previous full554.
    score_from_loss = float(
        np.exp(-loss_value)
    )

    # Reconstruct exact official adjacency formula for localization.
    src_dist = torch.norm(
        src[:, :, None, :] - src[:, None, :, :],
        dim=-1,
    )

    tgt_dist = torch.norm(
        tgt[:, :, None, :] - tgt[:, None, :, :],
        dim=-1,
    )

    cross_dist = torch.abs(
        src_dist - tgt_dist
    )

    adj = torch.clamp(
        1.0
        - cross_dist ** 2 / D_THRE ** 2,
        min=0,
    )

    eig = power_iteration(adj)
    score_direct = spatial_consistency_score(
        adj,
        eig,
    )

    return {
        "loss": loss_value,
        "score_from_loss": score_from_loss,
        "score_direct": float(score_direct.item()),
        "cross_dist": cross_dist[0].cpu().numpy(),
        "adj": adj[0].cpu().numpy(),
    }

gt_res = official_mbnsf(
    x0,
    gt10,
)

pred_res = official_mbnsf(
    x0,
    pred10,
)

print()
print("=== OFFICIAL MBNSF REPRODUCTION ===")
print("GT score =", gt_res["score_direct"])
print("Pred score =", pred_res["score_direct"])
print("Old pred score =", old_pred_score)
print(
    "Pred score abs diff =",
    abs(
        pred_res["score_direct"]
        - old_pred_score
    )
)

print(
    "Pred direct-vs-loss score diff =",
    abs(
        pred_res["score_direct"]
        - pred_res["score_from_loss"]
    )
)

print(
    "GT direct-vs-loss score diff =",
    abs(
        gt_res["score_direct"]
        - gt_res["score_from_loss"]
    )
)

# The historical full554 CSV used exp(-loss), therefore THIS is
# the exact reproduction criterion.
print(
    "Old-vs-loss-reconstructed pred diff =",
    abs(
        pred_res["score_from_loss"]
        - old_pred_score
    )
)

assert abs(
    pred_res["score_from_loss"]
    - old_pred_score
) < 1e-12

# ================================================================
# 4. UNIQUE PAIRWISE STRUCTURAL BREAK AUDIT
#
# No new threshold:
# "zero compatibility" follows DIRECTLY from official MBNSF:
# cross_dist >= d_thre -> adj = 0.
# ================================================================
ii, jj = np.triu_indices(
    n,
    k=1,
)

def pair_stats(name, cross, adj):

    cd = cross[ii, jj]
    aa = adj[ii, jj]

    print()
    print(f"=== {name} UNIQUE PAIR STATS ===")
    print("n_pairs =", len(cd))

    for q in [
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]:
        print(
            f"cross_dist q{int(q*100):02d} mm =",
            float(np.quantile(cd, q) * 1000.0)
        )

    print(
        "cross_dist max mm =",
        float(cd.max() * 1000.0)
    )

    print(
        "mean compatibility =",
        float(aa.mean())
    )

    print(
        "zero-compatibility pairs =",
        int((aa == 0).sum())
    )

    print(
        "zero-compatibility fraction =",
        float((aa == 0).mean())
    )

    print(
        "pairs with distance change >= 30mm =",
        int((cd >= D_THRE).sum())
    )

    # These MUST be identical by official equation.
    assert np.array_equal(
        aa == 0,
        cd >= D_THRE,
    )

    return cd, aa

gt_cd, gt_adj = pair_stats(
    "GT t0 -> t10",
    gt_res["cross_dist"],
    gt_res["adj"],
)

pred_cd, pred_adj = pair_stats(
    "PRED t0 -> t10",
    pred_res["cross_dist"],
    pred_res["adj"],
)

# ================================================================
# 5. TOP BROKEN PAIRS
#
# Ranking only, no new failure threshold.
# ================================================================
order = np.argsort(
    -pred_cd
)

top_rows = []

for rank, p in enumerate(
    order[:50],
    start=1,
):
    i = int(ii[p])
    j = int(jj[p])

    top_rows.append({
        "rank": rank,
        "local_i": i,
        "local_j": j,
        "global_i": int(global_idx[i]),
        "global_j": int(global_idx[j]),

        "pred_pair_distance_change_mm":
            float(pred_cd[p] * 1000.0),

        "pred_mbnsf_compatibility":
            float(pred_adj[p]),

        "gt_pair_distance_change_mm":
            float(
                gt_res["cross_dist"][i, j]
                * 1000.0
            ),

        "x0_i_x": float(x0_np[i, 0]),
        "x0_i_y": float(x0_np[i, 1]),
        "x0_i_z": float(x0_np[i, 2]),

        "x0_j_x": float(x0_np[j, 0]),
        "x0_j_y": float(x0_np[j, 1]),
        "x0_j_z": float(x0_np[j, 2]),
    })

top_pairs = pd.DataFrame(top_rows)

# ================================================================
# 6. PER-POINT DIAGNOSTIC LOCALIZATION
#
# IMPORTANT:
# This is NOT an official MBNSF metric.
# It only aggregates the official pairwise adjacency per point
# to help visualize where incompatibility is concentrated.
# ================================================================
point_rows = []

for i in range(n):

    others = np.arange(n) != i

    cd_i = pred_res["cross_dist"][
        i, others
    ]

    adj_i = pred_res["adj"][
        i, others
    ]

    gt_cd_i = gt_res["cross_dist"][
        i, others
    ]

    point_rows.append({
        "local_index": i,
        "global_point_index":
            int(global_idx[i]),

        "mean_pred_pair_change_mm":
            float(cd_i.mean() * 1000.0),

        "median_pred_pair_change_mm":
            float(np.median(cd_i) * 1000.0),

        "max_pred_pair_change_mm":
            float(cd_i.max() * 1000.0),

        "mean_mbnsf_incompatibility":
            float((1.0 - adj_i).mean()),

        "zero_compatibility_fraction":
            float((adj_i == 0).mean()),

        "mean_gt_pair_change_mm":
            float(gt_cd_i.mean() * 1000.0),

        "x0_x": float(x0_np[i, 0]),
        "x0_y": float(x0_np[i, 1]),
        "x0_z": float(x0_np[i, 2]),

        "pred10_x": float(pred10_np[i, 0]),
        "pred10_y": float(pred10_np[i, 1]),
        "pred10_z": float(pred10_np[i, 2]),

        "gt10_x": float(gt10_np[i, 0]),
        "gt10_y": float(gt10_np[i, 1]),
        "gt10_z": float(gt10_np[i, 2]),
    })

points = pd.DataFrame(point_rows)

points = points.sort_values(
    "mean_mbnsf_incompatibility",
    ascending=False,
).reset_index(drop=True)

# ================================================================
# 7. SAVE EXACT MATRICES FOR NEXT SPATIAL STEP
# ================================================================
top_csv = os.path.join(
    OUTDIR,
    "top50_broken_pairs.csv"
)

point_csv = os.path.join(
    OUTDIR,
    "per_point_mbnsf_localization_diagnostic.csv"
)

npz_out = os.path.join(
    OUTDIR,
    "top1_mbnsf_pairwise_matrices.npz"
)

top_pairs.to_csv(
    top_csv,
    index=False,
)

points.to_csv(
    point_csv,
    index=False,
)

np.savez_compressed(
    npz_out,
    sample_key=np.array(key),
    object_id=np.array(object_id),
    global_idx=global_idx,

    x0=x0_np,
    gt10=gt10_np,
    pred10=pred10_np,

    gt_cross_dist=gt_res["cross_dist"],
    pred_cross_dist=pred_res["cross_dist"],

    gt_adj=gt_res["adj"],
    pred_adj=pred_res["adj"],

    d_thre=np.array(D_THRE),
)

print()
print("=== MOST STRUCTURALLY INCOMPATIBLE POINTS ===")
print(
    points[
        [
            "local_index",
            "global_point_index",
            "mean_pred_pair_change_mm",
            "max_pred_pair_change_mm",
            "mean_mbnsf_incompatibility",
            "zero_compatibility_fraction",
            "mean_gt_pair_change_mm",
        ]
    ].head(15).to_string(index=False)
)

print()
print("=== TOP 10 BROKEN PAIRS ===")
print(
    top_pairs[
        [
            "rank",
            "local_i",
            "local_j",
            "pred_pair_distance_change_mm",
            "pred_mbnsf_compatibility",
            "gt_pair_distance_change_mm",
        ]
    ].head(10).to_string(index=False)
)

print()
print("TOP_PAIRS_CSV =", top_csv)
print("POINT_CSV     =", point_csv)
print("MATRICES_NPZ  =", npz_out)

print()
print("IMPORTANT:")
print(
    "Per-point localization is a diagnostic aggregation "
    "of the OFFICIAL MBNSF adjacency, not a new official metric."
)

print()
print(
    "TOP1_MBNSF_PAIRWISE_LOCALIZATION_SMOKE_SUCCESS = True"
)
