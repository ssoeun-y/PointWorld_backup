import os
import glob
import numpy as np
import pandas as pd

NPZ_ROOT = (
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237"
)

OBJECT_CSV = (
    "diagnostics/results/"
    "behavior_object_full578_20260810_080555/"
    "per_object.csv"
)

OUT = (
    "diagnostics/results/"
    "behavior_radio_rigidity_rank.csv"
)

df = pd.read_csv(OBJECT_CSV)

print("=" * 100)
print("CSV_COLUMNS =", list(df.columns))

# ------------------------------------------------------------
# 필요한 column 자동 탐색
# ------------------------------------------------------------
key_candidates = ["sample_key", "key"]
name_candidates = ["object_name", "name"]
id_candidates = ["object_id"]

key_col = next((x for x in key_candidates if x in df.columns), None)
name_col = next((x for x in name_candidates if x in df.columns), None)
id_col = next((x for x in id_candidates if x in df.columns), None)

assert key_col is not None, "sample key column not found"
assert name_col is not None, "object name column not found"
assert id_col is not None, "object id column not found"

print("KEY_COLUMN =", key_col)
print("NAME_COLUMN =", name_col)
print("ID_COLUMN =", id_col)

radio_df = df[df[name_col].astype(str).str.contains("radio_89", case=False, na=False)].copy()

print("RADIO_ROWS_IN_CSV =", len(radio_df))
assert len(radio_df) > 0


def rigid_residual(src, dst):
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)

    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)

    X = src - src_c
    Y = dst - dst_c

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_c - src_c @ R.T
    fit = src @ R.T + t

    return np.linalg.norm(fit - dst, axis=1)


rows = []
missing = 0
skipped = 0

for _, r in radio_df.iterrows():
    sample_key = str(r[key_col])
    object_id = int(r[id_col])

    # ":" -> "_"인 NPZ filename
    fname = sample_key.replace(":", "_") + ".npz"
    path = os.path.join(NPZ_ROOT, fname)

    if not os.path.exists(path):
        missing += 1
        continue

    d = np.load(path, allow_pickle=True)

    gt = d["gt_scene_flows"]
    pred = d["pred_scene_flows"]
    obj = d["scene_object_id"]
    valid = d["official_valid_mask"]

    # 기존 분석과 동일: t0 object identity + final official-valid
    mask = (obj[0] == object_id) & valid[-1]

    n = int(mask.sum())

    # 너무 적은 point는 ranking에서 제외
    if n < 20:
        skipped += 1
        continue

    x0 = gt[0, mask]
    xgt = gt[-1, mask]
    xpred = pred[-1, mask]

    gt_res = rigid_residual(x0, xgt)
    pred_res = rigid_residual(x0, xpred)

    rows.append({
        "sample_key": sample_key,
        "object_id": object_id,
        "n_points": n,

        "gt_mean_mm": gt_res.mean() * 1000,
        "gt_p95_mm": np.quantile(gt_res, 0.95) * 1000,
        "gt_max_mm": gt_res.max() * 1000,

        "pred_mean_mm": pred_res.mean() * 1000,
        "pred_median_mm": np.median(pred_res) * 1000,
        "pred_p95_mm": np.quantile(pred_res, 0.95) * 1000,
        "pred_max_mm": pred_res.max() * 1000,

        "excess_mean_mm": (pred_res.mean() - gt_res.mean()) * 1000,
    })


out = pd.DataFrame(rows)

assert len(out) > 0

# GT 자체가 rigid한 샘플에서 Pred deformation 큰 순
out = out.sort_values(
    ["excess_mean_mm", "pred_p95_mm"],
    ascending=False
).reset_index(drop=True)

out.to_csv(OUT, index=False)

print("\n" + "=" * 100)
print("PROCESSED_RADIO =", len(out))
print("MISSING_NPZ =", missing)
print("SKIPPED_LT20PTS =", skipped)

print("\nTOP 15 RADIO BY NON-RIGID PREDICTION")
print("=" * 100)

cols = [
    "sample_key",
    "object_id",
    "n_points",
    "gt_mean_mm",
    "pred_mean_mm",
    "pred_p95_mm",
    "pred_max_mm",
    "excess_mean_mm",
]

print(
    out[cols]
    .head(15)
    .to_string(index=False)
)

print("\n00470 CURRENT SAMPLE")
print("=" * 100)

cur = out[
    out["sample_key"].astype(str).str.contains(
        "00000470-375:386",
        regex=False
    )
]

print(cur[cols].to_string(index=False))

if len(cur):
    rank = int(cur.index[0]) + 1
    print("00470_RANK =", rank, "/", len(out))

print("\nOUTPUT =", OUT)
print("OUTPUT_EXISTS =", os.path.exists(OUT))
print("RADIO_RIGIDITY_RANK_SUCCESS = True")
