from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist, pdist


ROOT = Path(
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237"
)

# Previously established episode-disjoint severe radio Top-5.
CASES = [
    (
        1,
        "task-0000_episode_00002530-300:311",
        ROOT / "task-0000_episode_00002530-300_311.npz",
    ),
    (
        2,
        "task-0000_episode_00000010-230:241",
        ROOT / "task-0000_episode_00000010-230_241.npz",
    ),
    (
        3,
        "task-0000_episode_00000970-220:231",
        ROOT / "task-0000_episode_00000970-220_231.npz",
    ),
    (
        4,
        "task-0000_episode_00000870-330:341",
        ROOT / "task-0000_episode_00000870-330_341.npz",
    ),
    (
        5,
        "task-0000_episode_00000220-335:346",
        ROOT / "task-0000_episode_00000220-335_346.npz",
    ),
]

RADIO_OBJECT_ID = 2


def kabsch_residual_mm(x0, xt):
    x0 = x0.astype(np.float64)
    xt = xt.astype(np.float64)

    c0 = x0.mean(axis=0)
    ct = xt.mean(axis=0)

    X = x0 - c0
    Y = xt - ct

    U, _, Vt = np.linalg.svd(X.T @ Y)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    aligned = X @ R.T + ct

    return (
        np.linalg.norm(
            aligned - xt,
            axis=-1,
        )
        * 1000.0
    )


def pair_p95_mm(x0, xt):
    if len(x0) < 2:
        return np.nan

    d0 = pdist(
        x0.astype(np.float64)
    )

    dt = pdist(
        xt.astype(np.float64)
    )

    return float(
        np.percentile(
            np.abs(dt - d0) * 1000.0,
            95,
        )
    )


print("=" * 90)
print("RADIO: DENSE/CORE-LIKE vs SPARSE/EDGE-LIKE QUICK CHECK")
print("Subset definition uses GT t0 geometry ONLY.")
print("Prediction outcome is NOT used to define dense/sparse subsets.")
print("=" * 90)

all_dense_res = []
all_sparse_res = []

processed = 0


for rank, key, path in CASES:

    print()
    print("#" * 90)
    print(f"SEVERE RANK {rank} | {key}")
    print("#" * 90)

    print("file =", path)
    print("exists =", path.exists())

    if not path.exists():
        raise RuntimeError(
            f"Missing trajectory NPZ: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as z:

        required = {
            "scene_object_id",
            "official_valid_mask",
            "gt_scene_flows",
            "pred_scene_flows",
        }

        print(
            "npz_keys =",
            sorted(z.files),
        )

        missing = (
            required - set(z.files)
        )

        if missing:
            raise RuntimeError(
                f"{path}: missing {sorted(missing)}"
            )

        ids = z[
            "scene_object_id"
        ].copy()

        valid = z[
            "official_valid_mask"
        ].astype(bool).copy()

        gt_all = z[
            "gt_scene_flows"
        ].copy()

        pred_all = z[
            "pred_scene_flows"
        ].copy()


    # --------------------------------------------------------
    # Exact same radio population convention used previously:
    #
    # object identity at t0
    # +
    # final official-valid at t10
    # --------------------------------------------------------

    if ids.shape != valid.shape:
        raise RuntimeError(
            f"ID/valid shape mismatch: "
            f"{ids.shape} vs {valid.shape}"
        )

    if gt_all.shape != pred_all.shape:
        raise RuntimeError(
            "GT/PRED shape mismatch"
        )

    if gt_all.shape[:2] != ids.shape:
        raise RuntimeError(
            "Trajectory/object-ID shape mismatch"
        )

    idx = np.flatnonzero(
        (ids[0] == RADIO_OBJECT_ID)
        & valid[-1]
    )

    gt = gt_all[:, idx, :]
    pred = pred_all[:, idx, :]

    n = len(idx)

    print(
        "radio_final_valid_points =",
        n,
    )

    if n < 8:
        raise RuntimeError(
            f"Too few radio points: {n}"
        )


    # --------------------------------------------------------
    # Strong correspondence sanity.
    # PointWorld prediction starts at GT t0.
    # --------------------------------------------------------

    t0_diff_mm = (
        np.linalg.norm(
            pred[0].astype(np.float64)
            - gt[0].astype(np.float64),
            axis=-1,
        )
        * 1000.0
    )

    print(
        "pred_vs_gt_t0_max_mm =",
        float(t0_diff_mm.max()),
    )


    # --------------------------------------------------------
    # GT-t0 local-density proxy.
    #
    # dense/core-like:
    #   locally closer neighbors
    #
    # sparse/edge-like:
    #   locally farther neighbors
    #
    # IMPORTANT:
    # This is NOT true mesh-boundary classification.
    # --------------------------------------------------------

    x0 = gt[0].astype(
        np.float64
    )

    D = cdist(
        x0,
        x0,
    )

    np.fill_diagonal(
        D,
        np.inf,
    )

    k = min(
        4,
        n - 1,
    )

    knn = np.partition(
        D,
        kth=k - 1,
        axis=1,
    )[:, :k]

    sparsity = (
        knn.mean(axis=1)
    )

    threshold = float(
        np.median(sparsity)
    )

    dense = (
        sparsity <= threshold
    )

    sparse = (
        sparsity > threshold
    )


    print(
        "dense/core-like n =",
        int(dense.sum()),
    )

    print(
        "sparse/edge-like n =",
        int(sparse.sum()),
    )

    print(
        "GT-only sparsity median mm =",
        threshold * 1000.0,
    )


    # --------------------------------------------------------
    # ONE rigid transform fitted to the WHOLE predicted radio.
    #
    # Then inspect where residual points lie.
    # --------------------------------------------------------

    full_pred_residual = (
        kabsch_residual_mm(
            pred[0],
            pred[-1],
        )
    )

    dense_res = (
        full_pred_residual[
            dense
        ]
    )

    sparse_res = (
        full_pred_residual[
            sparse
        ]
    )


    # --------------------------------------------------------
    # Pair-distance structure restricted within each subset.
    # --------------------------------------------------------

    pred_dense_pair = (
        pair_p95_mm(
            pred[0][dense],
            pred[-1][dense],
        )
    )

    pred_sparse_pair = (
        pair_p95_mm(
            pred[0][sparse],
            pred[-1][sparse],
        )
    )

    gt_dense_pair = (
        pair_p95_mm(
            gt[0][dense],
            gt[-1][dense],
        )
    )

    gt_sparse_pair = (
        pair_p95_mm(
            gt[0][sparse],
            gt[-1][sparse],
        )
    )


    print()
    print(
        "Dense/core-like SE3 residual "
        "median/P95 = "
        f"{np.median(dense_res):.3f} / "
        f"{np.percentile(dense_res, 95):.3f} mm"
    )

    print(
        "Sparse/edge-like SE3 residual "
        "median/P95 = "
        f"{np.median(sparse_res):.3f} / "
        f"{np.percentile(sparse_res, 95):.3f} mm"
    )

    print()

    print(
        "GT dense-dense pair P95   =",
        f"{gt_dense_pair:.6f} mm",
    )

    print(
        "PRED dense-dense pair P95 =",
        f"{pred_dense_pair:.3f} mm",
    )

    print(
        "GT sparse-sparse pair P95   =",
        f"{gt_sparse_pair:.6f} mm",
    )

    print(
        "PRED sparse-sparse pair P95 =",
        f"{pred_sparse_pair:.3f} mm",
    )


    all_dense_res.extend(
        dense_res.tolist()
    )

    all_sparse_res.extend(
        sparse_res.tolist()
    )

    processed += 1


# ============================================================
# AGGREGATE
# ============================================================

all_dense_res = np.asarray(
    all_dense_res,
    dtype=np.float64,
)

all_sparse_res = np.asarray(
    all_sparse_res,
    dtype=np.float64,
)


print()
print("=" * 90)
print("AGGREGATE TOP-5 RADIO")
print("=" * 90)

print(
    "Dense/core-like residual "
    "median/P95 =",
    f"{np.median(all_dense_res):.3f} / "
    f"{np.percentile(all_dense_res, 95):.3f} mm",
)

print(
    "Sparse/edge-like residual "
    "median/P95 =",
    f"{np.median(all_sparse_res):.3f} / "
    f"{np.percentile(all_sparse_res, 95):.3f} mm",
)

print()
print(
    "PROCESSED_CASES =",
    processed,
)

print(
    "EXPECTED_CASES =",
    len(CASES),
)

print(
    "ALL_TOP5_PROCESSED =",
    processed == len(CASES),
)

print()
print(
    "IMPORTANT: dense/core-like vs sparse/edge-like "
    "is a GT-t0 local-density proxy, "
    "NOT definitive mesh-boundary classification."
)

print("NO MODEL INFERENCE")
print("NO MBNSF")
print("RADIO_EDGE_VS_CORE_QUICK_DONE = True")
