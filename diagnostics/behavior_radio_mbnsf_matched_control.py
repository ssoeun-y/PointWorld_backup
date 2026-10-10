import os
import re
import numpy as np
import pandas as pd

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
    "behavior_radio_mbnsf_matched_control"
)
os.makedirs(OUTDIR, exist_ok=True)


def episode_id_from_key(key):
    m = re.search(r"episode_(\d+)", key)
    assert m is not None, key
    return m.group(1)


def gt_motion_controls(x0, x1):
    """
    GT motion difficulty controls only.

    translation:
        centroid displacement

    rotation:
        best-fit rigid rotation angle by standard Kabsch alignment

    Kabsch is NOT used as the primary rigidity metric here.
    """

    c0 = x0.mean(axis=0)
    c1 = x1.mean(axis=0)

    translation_mm = float(
        np.linalg.norm(c1 - c0) * 1000.0
    )

    X = x0 - c0
    Y = x1 - c1

    H = X.T @ Y

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    cos_angle = (
        np.trace(R) - 1.0
    ) / 2.0

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0
    )

    rotation_deg = float(
        np.degrees(
            np.arccos(cos_angle)
        )
    )

    # Audit only: GT should already be extremely rigid.
    pred_rigid = X @ R.T

    gt_rigid_residual_mm = float(
        np.linalg.norm(
            pred_rigid - Y,
            axis=1
        ).mean() * 1000.0
    )

    return (
        translation_mm,
        rotation_deg,
        gt_rigid_residual_mm
    )


df = pd.read_csv(FULL_CSV).copy()

assert len(df) == 554

control_rows = []

for i, row in df.iterrows():

    key = str(row["sample_key"])
    object_id = int(row["object_id"])

    path = os.path.join(
        NPZ_ROOT,
        key.replace(":", "_") + ".npz"
    )

    assert os.path.exists(path), path

    d = np.load(
        path,
        allow_pickle=True
    )

    assert bool(
        d["official_l2_exact"].item()
    )

    assert bool(
        d["temporal_object_id_exact"].item()
    )

    gt = d["gt_scene_flows"].astype(
        np.float64
    )

    obj = d["scene_object_id"]
    valid = d["official_valid_mask"]

    mask = (
        (obj[0] == object_id)
        & valid[-1]
    )

    x0 = gt[0, mask]
    x1 = gt[-1, mask]

    n = len(x0)

    assert n == int(row["n_points"])

    (
        gt_translation_mm,
        gt_rotation_deg,
        gt_rigid_residual_mm
    ) = gt_motion_controls(
        x0,
        x1
    )

    control_rows.append({
        "sample_key":
            key,

        "episode_id":
            episode_id_from_key(key),

        "n_points":
            n,

        # SAME endpoint as MBNSF rigidity
        "radio_t10_l2_mm":
            float(row["radio_t10_l2_mm"]),

        "gt_translation_mm":
            gt_translation_mm,

        "gt_rotation_deg":
            gt_rotation_deg,

        # audit only
        "gt_rigid_residual_mm":
            gt_rigid_residual_mm,

        # OUTCOME -- never used for matching
        "mbnsf_pred_sc_score":
            float(row["mbnsf_pred_sc_score"]),

        "mbnsf_pred_sc_loss":
            float(row["mbnsf_pred_sc_loss"]),
    })


data = pd.DataFrame(control_rows)

controls_csv = os.path.join(
    OUTDIR,
    "radio554_matching_controls.csv"
)

data.to_csv(
    controls_csv,
    index=False
)


# ================================================================
# MATCHING VARIABLES
#
# MBNSF is deliberately NOT included here.
# ================================================================

MATCH_COLS = [
    "radio_t10_l2_mm",
    "gt_translation_mm",
    "gt_rotation_deg",
    "n_points",
]


# Standard nearest-neighbour matching idea:
# put covariates on comparable scales with ordinary z-standardization.
X = data[MATCH_COLS].to_numpy(
    dtype=np.float64
)

mu = X.mean(axis=0)
sd = X.std(axis=0, ddof=1)

assert np.all(sd > 0)

Z = (X - mu) / sd


# ================================================================
# Build every cross-episode candidate pair.
# IMPORTANT:
# score/loss is NEVER consulted here.
# ================================================================

candidates = []

N = len(data)

for i in range(N):

    epi = data.iloc[i]["episode_id"]

    for j in range(i + 1, N):

        epj = data.iloc[j]["episode_id"]

        # prevent within-episode pseudo-matching
        if epi == epj:
            continue

        dist = float(
            np.linalg.norm(
                Z[i] - Z[j]
            )
        )

        candidates.append(
            (dist, i, j)
        )


# Best covariate matches first.
# This ordering is outcome-blind.
candidates.sort(
    key=lambda x: x[0]
)


# ================================================================
# Episode-disjoint greedy matching
#
# One episode may appear in AT MOST ONE accepted pair.
# Again MBNSF outcome is not used.
# ================================================================

used_episodes = set()
accepted = []

for dist, i, j in candidates:

    epi = data.iloc[i]["episode_id"]
    epj = data.iloc[j]["episode_id"]

    if epi in used_episodes:
        continue

    if epj in used_episodes:
        continue

    accepted.append(
        (dist, i, j)
    )

    used_episodes.add(epi)
    used_episodes.add(epj)


pairs = []

for rank, (dist, i, j) in enumerate(
    accepted,
    start=1
):

    a = data.iloc[i]
    b = data.iloc[j]

    pairs.append({
        "match_rank":
            rank,

        "match_distance_z":
            dist,

        "sample_a":
            a["sample_key"],

        "sample_b":
            b["sample_key"],

        "episode_a":
            a["episode_id"],

        "episode_b":
            b["episode_id"],

        # --------------------------------------------------------
        # CONTROL GAPS
        # --------------------------------------------------------

        "t10_l2_a_mm":
            a["radio_t10_l2_mm"],

        "t10_l2_b_mm":
            b["radio_t10_l2_mm"],

        "t10_l2_gap_mm":
            abs(
                a["radio_t10_l2_mm"]
                -
                b["radio_t10_l2_mm"]
            ),

        "gt_translation_a_mm":
            a["gt_translation_mm"],

        "gt_translation_b_mm":
            b["gt_translation_mm"],

        "gt_translation_gap_mm":
            abs(
                a["gt_translation_mm"]
                -
                b["gt_translation_mm"]
            ),

        "gt_rotation_a_deg":
            a["gt_rotation_deg"],

        "gt_rotation_b_deg":
            b["gt_rotation_deg"],

        "gt_rotation_gap_deg":
            abs(
                a["gt_rotation_deg"]
                -
                b["gt_rotation_deg"]
            ),

        "n_points_a":
            int(a["n_points"]),

        "n_points_b":
            int(b["n_points"]),

        "n_points_gap":
            abs(
                int(a["n_points"])
                -
                int(b["n_points"])
            ),

        # --------------------------------------------------------
        # OUTCOME -- revealed ONLY after matching
        # --------------------------------------------------------

        "mbnsf_score_a":
            a["mbnsf_pred_sc_score"],

        "mbnsf_score_b":
            b["mbnsf_pred_sc_score"],

        "mbnsf_score_gap":
            abs(
                a["mbnsf_pred_sc_score"]
                -
                b["mbnsf_pred_sc_score"]
            ),

        "mbnsf_loss_a":
            a["mbnsf_pred_sc_loss"],

        "mbnsf_loss_b":
            b["mbnsf_pred_sc_loss"],

        "mbnsf_loss_gap":
            abs(
                a["mbnsf_pred_sc_loss"]
                -
                b["mbnsf_pred_sc_loss"]
            ),
    })


pairs = pd.DataFrame(pairs)

pairs_csv = os.path.join(
    OUTDIR,
    "episode_disjoint_matched_pairs.csv"
)

pairs.to_csv(
    pairs_csv,
    index=False
)


# ================================================================
# REPORT
# ================================================================

print("=" * 120)
print("RADIO MBNSF MATCHED CONTROL")
print("=" * 120)

print("SAMPLES =", len(data))
print(
    "UNIQUE_EPISODES =",
    data["episode_id"].nunique()
)
print(
    "MATCHED_PAIRS =",
    len(pairs)
)

print("\nMATCHING VARIABLES ONLY:")
for c in MATCH_COLS:
    print(" -", c)

print("\nOUTCOME EXCLUDED FROM MATCHING:")
print(" - mbnsf_pred_sc_score")
print(" - mbnsf_pred_sc_loss")


print("\n" + "=" * 120)
print("GT RIGIDITY AUDIT")
print("=" * 120)

print(
    data["gt_rigid_residual_mm"]
    .describe(
        percentiles=[
            0.5, 0.9, 0.95, 0.99
        ]
    )
)


print("\n" + "=" * 120)
print("MATCH QUALITY")
print("=" * 120)

for c in [
    "match_distance_z",
    "t10_l2_gap_mm",
    "gt_translation_gap_mm",
    "gt_rotation_gap_deg",
    "n_points_gap",
]:

    print("\n", c)

    print(
        pairs[c].describe(
            percentiles=[
                0.25,
                0.5,
                0.75,
                0.9
            ]
        )
    )


print("\n" + "=" * 120)
print("MBNSF SCORE GAP AFTER OUTCOME-BLIND MATCHING")
print("=" * 120)

print(
    pairs["mbnsf_score_gap"]
    .describe(
        percentiles=[
            0.25,
            0.5,
            0.75,
            0.9,
            0.95
        ]
    )
)


print("\n" + "=" * 120)
print("20 BEST-MATCHED PAIRS")
print("ORDERED ONLY BY CONTROL MATCH DISTANCE")
print("=" * 120)

for _, r in pairs.head(20).iterrows():

    print(
        f"match={int(r['match_rank']):2d} | "
        f"dist={r['match_distance_z']:.4f} | "
        f"L2 gap={r['t10_l2_gap_mm']:.4f}mm | "
        f"GT trans gap={r['gt_translation_gap_mm']:.4f}mm | "
        f"GT rot gap={r['gt_rotation_gap_deg']:.4f}deg | "
        f"n gap={int(r['n_points_gap'])} | "
        f"MBNSF gap={r['mbnsf_score_gap']:.6f}"
    )

    print(
        f"    A score={r['mbnsf_score_a']:.6f} "
        f"{r['sample_a']}"
    )

    print(
        f"    B score={r['mbnsf_score_b']:.6f} "
        f"{r['sample_b']}"
    )


# Also summarize the objectively closest quartile.
# Quartile boundary comes from the empirical matching-distance
# distribution, not from MBNSF outcomes.
q25 = pairs[
    "match_distance_z"
].quantile(0.25)

best_q = pairs[
    pairs["match_distance_z"] <= q25
]

print("\n" + "=" * 120)
print("CLOSEST 25% OF OUTCOME-BLIND MATCHES")
print("=" * 120)

print(
    "N =",
    len(best_q)
)

print(
    "match-distance boundary =",
    q25
)

print("\nCONTROL GAPS")

for c in [
    "t10_l2_gap_mm",
    "gt_translation_gap_mm",
    "gt_rotation_gap_deg",
    "n_points_gap",
]:

    print(
        c,
        "median =",
        best_q[c].median()
    )


print("\nMBNSF SCORE GAP")

print(
    best_q["mbnsf_score_gap"]
    .describe(
        percentiles=[
            0.25,
            0.5,
            0.75,
            0.9
        ]
    )
)


print("\nFILES")
print(controls_csv)
print(pairs_csv)

print(
    "\nCONTROLS_CSV_EXISTS =",
    os.path.exists(controls_csv)
)

print(
    "PAIRS_CSV_EXISTS =",
    os.path.exists(pairs_csv)
)

print(
    "MATCHED_CONTROL_SUCCESS = True"
)
