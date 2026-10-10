from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist, pdist

ROOT = Path(
    "diagnostics/results/"
    "behavior_task0001_soda_can_gt_top5_structural/captures"
)

CASES = [
    (1, ROOT / "rank1_task-0001_episode_00012190-460_471.npz"),
    (3, ROOT / "rank3_task-0001_episode_00012100-455_466.npz"),
    (4, ROOT / "rank4_task-0001_episode_00012150-415_426.npz"),
    (5, ROOT / "rank5_task-0001_episode_00012210-480_491.npz"),
]


def kabsch_residual_mm(x0, xt):
    x0 = x0.astype(np.float64)
    xt = xt.astype(np.float64)

    c0 = x0.mean(0)
    ct = xt.mean(0)

    X = x0 - c0
    Y = xt - ct

    U, _, Vt = np.linalg.svd(X.T @ Y)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    aligned = X @ R.T + ct

    return np.linalg.norm(
        aligned - xt,
        axis=1
    ) * 1000.0


def pair_change_p95_mm(x0, xt):
    if len(x0) < 2:
        return np.nan

    d0 = pdist(x0.astype(np.float64))
    dt = pdist(xt.astype(np.float64))

    return float(
        np.percentile(
            np.abs(dt - d0) * 1000.0,
            95
        )
    )


print("=" * 86)
print("QUICK DENSE/CORE-LIKE vs SPARSE/EDGE-LIKE CHECK")
print("Classification uses GT t0 only; prediction outcome is NOT used.")
print("=" * 86)

all_dense_res = []
all_sparse_res = []

for rank, path in CASES:

    with np.load(path, allow_pickle=False) as z:
        gt = z["gt"].copy()
        pred = z["pred"].copy()

    x0 = gt[0]
    n = len(x0)

    # --------------------------------------------------------
    # GT-t0-only local sparsity score
    # Higher = fewer nearby points = more edge/sparse-like
    # --------------------------------------------------------
    D = cdist(
        x0.astype(np.float64),
        x0.astype(np.float64)
    )

    np.fill_diagonal(D, np.inf)

    k = min(4, n - 1)

    knn = np.partition(
        D,
        kth=k-1,
        axis=1
    )[:, :k]

    sparsity = knn.mean(axis=1)

    # Median split: robust even for Rank5 n=16.
    threshold = np.median(sparsity)

    dense = sparsity <= threshold
    sparse = sparsity > threshold

    # --------------------------------------------------------
    # Fit ONE rigid transform on the whole object,
    # then inspect residual location.
    # --------------------------------------------------------
    residual = kabsch_residual_mm(
        pred[0],
        pred[-1]
    )

    dense_res = residual[dense]
    sparse_res = residual[sparse]

    # --------------------------------------------------------
    # Structural distortion restricted to each subset.
    # --------------------------------------------------------
    dense_pair = pair_change_p95_mm(
        pred[0][dense],
        pred[-1][dense]
    )

    sparse_pair = pair_change_p95_mm(
        pred[0][sparse],
        pred[-1][sparse]
    )

    # GT controls for identical subsets.
    dense_gt_pair = pair_change_p95_mm(
        gt[0][dense],
        gt[-1][dense]
    )

    sparse_gt_pair = pair_change_p95_mm(
        gt[0][sparse],
        gt[-1][sparse]
    )

    print()
    print("#" * 86)
    print(f"RANK {rank} | n={n} | dense={dense.sum()} | sparse={sparse.sum()}")
    print("#" * 86)

    print(
        "Dense/core-like SE3 residual median/P95 = "
        f"{np.median(dense_res):.3f} / "
        f"{np.percentile(dense_res,95):.3f} mm"
    )

    print(
        "Sparse/edge-like SE3 residual median/P95 = "
        f"{np.median(sparse_res):.3f} / "
        f"{np.percentile(sparse_res,95):.3f} mm"
    )

    print()

    print(
        "GT dense-dense pair P95   =",
        f"{dense_gt_pair:.6f} mm"
    )

    print(
        "PRED dense-dense pair P95 =",
        f"{dense_pair:.3f} mm"
    )

    print(
        "GT sparse-sparse pair P95   =",
        f"{sparse_gt_pair:.6f} mm"
    )

    print(
        "PRED sparse-sparse pair P95 =",
        f"{sparse_pair:.3f} mm"
    )

    all_dense_res.extend(dense_res.tolist())
    all_sparse_res.extend(sparse_res.tolist())


all_dense_res = np.asarray(all_dense_res)
all_sparse_res = np.asarray(all_sparse_res)

print()
print("=" * 86)
print("AGGREGATE")
print("=" * 86)

print(
    "Dense/core-like residual median/P95 =",
    f"{np.median(all_dense_res):.3f} / "
    f"{np.percentile(all_dense_res,95):.3f} mm"
)

print(
    "Sparse/edge-like residual median/P95 =",
    f"{np.median(all_sparse_res):.3f} / "
    f"{np.percentile(all_sparse_res,95):.3f} mm"
)

print()
print(
    "IMPORTANT: this is a fast local-density proxy, "
    "not a definitive mesh-boundary classification."
)

print("EDGE_VS_CORE_QUICK_DONE = True")
