from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SAMPLE_KEY = "task-0000_episode_00002530-300:311"

TRAJ_PATH = Path(
    "diagnostics/results/"
    "behavior_object_trajectory_npz_full578_20260810_135237/"
    + SAMPLE_KEY.replace(":", "_")
    + ".npz"
)

LOC_ROOT = Path(
    "diagnostics/results/"
    "behavior_radio_top1_mbnsf_pairwise_localization_smoke"
)

LOC_PATH = LOC_ROOT / "top1_mbnsf_pairwise_matrices.npz"

OUT = LOC_ROOT / "top1_radio_fullobject_vs_fixed425_t0.png"


# ============================================================
# 1. Load existing audited data
# ============================================================

d = np.load(TRAJ_PATH, allow_pickle=True)
loc = np.load(LOC_PATH, allow_pickle=False)

assert str(d["sample_key"].item()) == SAMPLE_KEY
assert str(loc["sample_key"].item()) == SAMPLE_KEY
assert bool(d["official_l2_exact"].item())
assert bool(d["temporal_object_id_exact"].item())

object_id = int(loc["object_id"].item())

gt = d["gt_scene_flows"].astype(np.float64)
obj = d["scene_object_id"]
valid = d["official_valid_mask"]

assert gt.shape == (11, 7875, 3)


# ============================================================
# 2. Two populations
#
# ALL OBJECT:
# every t0 scene point labelled object_id=2
#
# FIXED 425:
# exact population used by MBNSF analysis
# ============================================================

all_object_mask = (
    obj[0] == object_id
)

fixed_mask = (
    all_object_mask
    & valid[-1]
)

all_idx = np.flatnonzero(all_object_mask)
fixed_idx = np.flatnonzero(fixed_mask)

assert len(fixed_idx) == 425
assert np.array_equal(
    fixed_idx,
    loc["global_idx"],
)

all_xyz = gt[0, all_object_mask]
fixed_xyz = gt[0, fixed_mask]

assert np.array_equal(
    fixed_xyz,
    loc["x0"].astype(np.float64),
)


# ============================================================
# 3. Coverage numbers
# ============================================================

n_all = len(all_xyz)
n_fixed = len(fixed_xyz)

coverage = (
    n_fixed / n_all
    if n_all > 0
    else np.nan
)

print("=" * 80)
print("TOP-1 RADIO OBJECT-COVERAGE SANITY CHECK")
print("=" * 80)

print("sample_key =", SAMPLE_KEY)
print("object_id =", object_id)

print()
print("all t0 object-labelled points =", n_all)
print("fixed MBNSF points =", n_fixed)
print("fixed / all ratio =", coverage)

print()
print("fixed_idx exact =", np.array_equal(
    fixed_idx,
    loc["global_idx"],
))

print(
    "fixed x0 exact =",
    np.array_equal(
        fixed_xyz,
        loc["x0"].astype(np.float64),
    ),
)


# ============================================================
# 4. Same bounds/view for all panels
# ============================================================

mins = all_xyz.min(axis=0)
maxs = all_xyz.max(axis=0)

center = (
    mins + maxs
) / 2.0

span = float(
    np.max(maxs - mins)
)

half = max(
    span * 0.55,
    1e-6,
)

bounds = [
    (
        center[a] - half,
        center[a] + half,
    )
    for a in range(3)
]


def style(ax):
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_zlim(*bounds[2])

    ax.set_box_aspect((1, 1, 1))

    ax.view_init(
        elev=22,
        azim=-58,
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")


# ============================================================
# 5. Visualization
# ============================================================

fig = plt.figure(
    figsize=(15.5, 5.2)
)


# Panel 1: full object-labelled population
ax = fig.add_subplot(
    1, 3, 1,
    projection="3d",
)

ax.scatter(
    all_xyz[:, 0],
    all_xyz[:, 1],
    all_xyz[:, 2],
    s=13,
    alpha=0.8,
)

ax.set_title(
    f"All object_id={object_id} points at t0\n"
    f"n={n_all}"
)

style(ax)


# Panel 2: overlay
ax = fig.add_subplot(
    1, 3, 2,
    projection="3d",
)

ax.scatter(
    all_xyz[:, 0],
    all_xyz[:, 1],
    all_xyz[:, 2],
    s=10,
    alpha=0.15,
    label="All object-labelled points",
)

ax.scatter(
    fixed_xyz[:, 0],
    fixed_xyz[:, 1],
    fixed_xyz[:, 2],
    s=18,
    alpha=0.95,
    label="Fixed final-valid 425",
)

ax.set_title(
    "Coverage overlay"
)

ax.legend(
    loc="best",
    fontsize=8,
)

style(ax)


# Panel 3: exact MBNSF population
ax = fig.add_subplot(
    1, 3, 3,
    projection="3d",
)

ax.scatter(
    fixed_xyz[:, 0],
    fixed_xyz[:, 1],
    fixed_xyz[:, 2],
    s=18,
    alpha=0.95,
)

ax.set_title(
    f"Exact MBNSF population\n"
    f"425 / {n_all} = {coverage:.1%}"
)

style(ax)


fig.suptitle(
    "Top-1 radio sanity check: full object-labelled point cloud vs MBNSF subset",
    fontsize=15,
)

fig.tight_layout()

fig.savefig(
    OUT,
    dpi=250,
    bbox_inches="tight",
)

plt.close(fig)

print()
print("saved =", OUT)
print("NO INFERENCE")
print("NO NEW METRIC")
print("SUCCESS = True")
print("=" * 80)
