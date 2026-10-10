import os
import numpy as np
import pandas as pd

RANK_CSV = "diagnostics/results/behavior_radio_rigidity_rank.csv"

NPZ_ROOT = (
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237"
)

OUTDIR = (
    "diagnostics/results/"
    "behavior_radio_top9_published_isometry"
)
os.makedirs(OUTDIR, exist_ok=True)

# Multi-Body Neural Scene Flow, Eq. (5)
# Paper-selected value in Sec. 4.5
D_THR = 0.03  # meters

TOP_N = 9
POWER_ITERATIONS = 10

rank_df = pd.read_csv(RANK_CSV).head(TOP_N).copy()


def pairwise_distance_matrix(x):
    diff = x[:, None, :] - x[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def mbnsf_affinity(source, projected, d_thr=0.03):
    """
    Multi-Body Neural Scene Flow Eq. (5)-(7):
      A_ij = [1 - (d_ij - d_hat_ij)^2 / d_thr^2]_+
    """
    d = pairwise_distance_matrix(source)
    d_hat = pairwise_distance_matrix(projected)

    A = 1.0 - ((d - d_hat) ** 2) / (d_thr ** 2)
    A = np.maximum(A, 0.0)

    return A


def mbnsf_score_eq8(A):
    """
    Multi-Body Neural Scene Flow Eq. (8):
      s(C,F) = (1/n^2) 1^T A 1
    = mean of all entries in A
    """
    return float(A.mean())


def mbnsf_score_eq10(A, n_iter=10):
    """
    Multi-Body Neural Scene Flow Eq. (10),
    with power iteration from supplementary Eq. (14).
    """
    n = A.shape[0]

    v = np.ones(n, dtype=np.float64)

    for _ in range(n_iter):
        Av = A @ v
        norm = np.linalg.norm(Av)

        if norm == 0:
            return 0.0

        v = Av / norm

    score = float((v.T @ A @ v) / n)

    return score


rows = []

for idx, row in rank_df.iterrows():

    key = str(row["sample_key"])
    object_id = int(row["object_id"])

    path = os.path.join(
        NPZ_ROOT,
        key.replace(":", "_") + ".npz"
    )

    assert os.path.exists(path), path

    d = np.load(path, allow_pickle=True)

    assert d["official_l2_exact"].item()
    assert d["temporal_object_id_exact"].item()

    gt = d["gt_scene_flows"].astype(np.float64)
    pred = d["pred_scene_flows"].astype(np.float64)
    obj = d["scene_object_id"]
    valid = d["official_valid_mask"]

    # Same final-valid radio point identities
    mask = (obj[0] == object_id) & valid[-1]

    x0 = gt[0, mask]
    xgt = gt[-1, mask]
    xpred = pred[-1, mask]

    n = len(x0)

    assert n == int(row["n_points"])

    # Published isometry score:
    # source -> GT final
    A_gt = mbnsf_affinity(
        x0,
        xgt,
        d_thr=D_THR
    )

    # source -> PointWorld predicted final
    A_pred = mbnsf_affinity(
        x0,
        xpred,
        d_thr=D_THR
    )

    gt_eq8 = mbnsf_score_eq8(A_gt)
    pred_eq8 = mbnsf_score_eq8(A_pred)

    gt_eq10 = mbnsf_score_eq10(
        A_gt,
        POWER_ITERATIONS
    )

    pred_eq10 = mbnsf_score_eq10(
        A_pred,
        POWER_ITERATIONS
    )

    rows.append({
        "rank": idx + 1,
        "sample_key": key,
        "n_points": n,

        "gt_isometry_eq8": gt_eq8,
        "pred_isometry_eq8": pred_eq8,

        "gt_isometry_eq10": gt_eq10,
        "pred_isometry_eq10": pred_eq10,

        # Existing SE(3) diagnostic, cross-check only
        "se3_residual_mean_mm":
            float(row["pred_mean_mm"]),
    })


out = pd.DataFrame(rows)

csv_path = os.path.join(
    OUTDIR,
    "top9_published_isometry.csv"
)

out.to_csv(csv_path, index=False)

print("=" * 120)
print("PUBLISHED ISOMETRY SMOKE")
print("Source: Multi-Body Neural Scene Flow Eq. (5), (8), (10)")
print("D_THR =", D_THR, "m")
print("POWER_ITERATIONS =", POWER_ITERATIONS)
print("=" * 120)

for _, r in out.iterrows():

    print(
        f"rank={int(r['rank']):2d} "
        f"n={int(r['n_points']):4d} | "
        f"GT Eq8={r['gt_isometry_eq8']:.6f} "
        f"Pred Eq8={r['pred_isometry_eq8']:.6f} | "
        f"GT Eq10={r['gt_isometry_eq10']:.6f} "
        f"Pred Eq10={r['pred_isometry_eq10']:.6f} | "
        f"SE3={r['se3_residual_mean_mm']:.3f}mm"
    )


# Cross-check only:
# Published isometry should DECREASE as SE3 residual increases.
def spearman(a, b):
    ra = pd.Series(a).rank(method="average").to_numpy(float)
    rb = pd.Series(b).rank(method="average").to_numpy(float)

    return float(np.corrcoef(ra, rb)[0, 1])


rho_eq8 = spearman(
    out["pred_isometry_eq8"],
    out["se3_residual_mean_mm"]
)

rho_eq10 = spearman(
    out["pred_isometry_eq10"],
    out["se3_residual_mean_mm"]
)

print("\n" + "=" * 120)
print("CROSS-CHECK WITH EXISTING SE3 DIAGNOSTIC")
print("Spearman(Eq8 score, SE3 residual)  =", rho_eq8)
print("Spearman(Eq10 score, SE3 residual) =", rho_eq10)
print("Expected direction = NEGATIVE")

print("\nCSV =", csv_path)
print("CSV_EXISTS =", os.path.exists(csv_path))
print("PUBLISHED_ISOMETRY_SMOKE_SUCCESS = True")
