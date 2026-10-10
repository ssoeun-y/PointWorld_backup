import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from tqdm import tqdm

BASE = Path(
    "diagnostics/results/"
    "droid_temporal_quality_final_1679.csv"
)
OUT = Path(
    "diagnostics/results/"
    "droid_action_denoise_diagnostic.csv"
)

sys.argv = [
    "eval.py",
    "--model_path=pretrained_checkpoints/large-droid+behavior/model-best.pt",
    "--domains=droid",
    "--data_dirs=/home/soenchun/cse/PointWorld_workspace/datasets/PointWorld-DROID-subset_restored/droid/wds",
    "--batch_size=1",
    "--eval_num_batches=-1",
    "--eval_skip_viz=true",
]

from arguments import parse_args
from evaluation.tester import Tester

EPS = 1e-12
TEMP = [
    "speed_cv",
    "accel_p95",
    "jerk_p95",
    "reversal_fraction",
]


# ----------------------------------------------------------
# EXACT SAME temporal primitives as BEHAVIOR
# ----------------------------------------------------------
def metrics_xyz(xyz):
    xyz = np.asarray(xyz, np.float64)

    v = np.diff(xyz, axis=0)
    speed = np.linalg.norm(v, axis=1)

    mean_speed = float(speed.mean())

    a = np.diff(v, axis=0)
    accel = np.linalg.norm(a, axis=1)

    j = np.diff(a, axis=0)
    jerk = np.linalg.norm(j, axis=1)

    rev = []

    for x, y in zip(v[:-1], v[1:]):
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)

        if nx <= EPS or ny <= EPS:
            continue

        c = np.clip(
            np.dot(x, y) / (nx * ny),
            -1.0, 1.0
        )
        rev.append(float(c < 0.0))

    def p95(x):
        return (
            float(np.percentile(x, 95))
            if len(x) else np.nan
        )

    return {
        "speed_cv":
            float(speed.std() / mean_speed)
            if mean_speed > EPS else np.nan,
        "accel_p95": p95(accel),
        "jerk_p95": p95(jerk),
        "reversal_fraction":
            float(np.mean(rev))
            if len(rev) else np.nan,
    }


# ----------------------------------------------------------
# Mild smoothing methods
# Endpoints are restored exactly.
# ----------------------------------------------------------
def sg5(xyz):
    y = savgol_filter(
        xyz,
        window_length=5,
        polyorder=2,
        axis=0,
        mode="interp",
    )
    y[0] = xyz[0]
    y[-1] = xyz[-1]
    return y


def ma3(xyz):
    y = xyz.copy()

    y[1:-1] = (
        0.25 * xyz[:-2]
        + 0.50 * xyz[1:-1]
        + 0.25 * xyz[2:]
    )

    y[0] = xyz[0]
    y[-1] = xyz[-1]
    return y


# ----------------------------------------------------------
# Load original official outcome
# ----------------------------------------------------------
base = pd.read_csv(BASE)

assert len(base) == 1679
assert base["clip"].nunique() == 1679

outcome = dict(zip(
    base["clip"],
    base["filtered_l2_moved_mm"]
))

episode = dict(zip(
    base["clip"],
    base["episode"]
))


# ----------------------------------------------------------
# Official PointWorld loader: pose extraction only
# NO model inference, NO modification of WDS
# ----------------------------------------------------------
args = parse_args()
tester = Tester(args)

dl, _ = tester._build_eval_loader(
    "test",
    enable_mask=False
)

rows = []

for batch in tqdm(
    dl,
    total=1679,
    desc="DROID pose denoise"
):
    key = str(batch["__key__"][0])

    pose = batch["right_gripper_pose"][0]

    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()

    xyz = np.asarray(pose, np.float64)[:, :3]

    variants = {
        "raw": xyz,
        "sg5": sg5(xyz),
        "ma3": ma3(xyz),
    }

    r = {
        "clip": key,
        "episode": episode[key],
        "filtered_l2_moved_mm": outcome[key],
    }

    for name, xx in variants.items():
        m = metrics_xyz(xx)

        for k, v in m.items():
            r[f"{name}_{k}"] = v

        # how much the smoothing actually moved the action points
        r[f"{name}_shift_mm"] = float(
            np.mean(
                np.linalg.norm(
                    xx - xyz,
                    axis=1
                )
            ) * 1000.0
        )

    rows.append(r)


df = pd.DataFrame(rows)

assert len(df) == 1679
assert df["clip"].nunique() == 1679


# ----------------------------------------------------------
# Temporal composite and outcome analysis
# ----------------------------------------------------------
summaries = []

for method in ["raw", "sg5", "ma3"]:

    pct_cols = []

    for c in TEMP:
        src = f"{method}_{c}"
        dst = f"{method}_{c}_pct"

        df[dst] = df[src].rank(
            method="average",
            pct=True
        )
        pct_cols.append(dst)

    raw_score = df[pct_cols].mean(axis=1)

    score = raw_score.rank(
        method="average",
        pct=True
    )

    df[f"{method}_temporal_score"] = score

    hq = score <= .25
    lq = score >= .75

    H = df.loc[
        hq,
        "filtered_l2_moved_mm"
    ]

    L = df.loc[
        lq,
        "filtered_l2_moved_mm"
    ]

    rho, p = spearmanr(
        score,
        df["filtered_l2_moved_mm"]
    )

    # episode-level
    ep = pd.DataFrame({
        "episode": df["episode"],
        "score": score,
        "err": df["filtered_l2_moved_mm"],
    }).groupby(
        "episode",
        as_index=False
    ).median(numeric_only=True)

    ep["q"] = ep["score"].rank(
        method="average",
        pct=True
    )

    EH = ep.loc[
        ep["q"] <= .25,
        "err"
    ]

    EL = ep.loc[
        ep["q"] >= .75,
        "err"
    ]

    erho, epval = spearmanr(
        ep["q"],
        ep["err"]
    )

    print("\n" + "=" * 88)
    print(method.upper())
    print("=" * 88)

    print(
        f"HQ={H.median():.6f} mm | "
        f"LQ={L.median():.6f} mm"
    )
    print(
        f"LQ/HQ={L.median()/H.median():.4f}x | "
        f"increase={(L.median()/H.median()-1)*100:.2f}%"
    )
    print(
        f"sample rho={rho:+.4f}, p={p:.3e}"
    )

    print(
        f"episode HQ={EH.median():.6f} | "
        f"LQ={EL.median():.6f} | "
        f"ratio={EL.median()/EH.median():.4f}x"
    )
    print(
        f"episode rho={erho:+.4f}, "
        f"p={epval:.3e}"
    )

    print("\nPRIMITIVES")

    for c in TEMP:
        rr, pp = spearmanr(
            df[f"{method}_{c}"],
            df["filtered_l2_moved_mm"],
            nan_policy="omit",
        )
        print(
            f"{c:20s} "
            f"rho={rr:+.4f} "
            f"p={pp:.3e}"
        )

    print(
        f"\nmean smoothing shift = "
        f"{df[f'{method}_shift_mm'].median():.4f} mm "
        f"(median over clips)"
    )

    summaries.append({
        "method": method,
        "hq_mm": H.median(),
        "lq_mm": L.median(),
        "ratio": L.median()/H.median(),
        "increase_pct":
            (L.median()/H.median()-1)*100,
        "rho": rho,
        "episode_ratio":
            EL.median()/EH.median(),
        "episode_rho": erho,
        "median_shift_mm":
            df[f"{method}_shift_mm"].median(),
    })


df.to_csv(OUT, index=False)

print("\n" + "=" * 88)
print("SUMMARY")
print("=" * 88)
print(
    pd.DataFrame(summaries)
    .to_string(index=False)
)

print("\nCSV =", OUT)
print("SUCCESS = True")
