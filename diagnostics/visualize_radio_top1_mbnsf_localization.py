from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(
    "diagnostics/results/"
    "behavior_radio_top1_mbnsf_pairwise_localization_smoke"
)

CSV_PATH = ROOT / "per_point_mbnsf_localization_diagnostic.csv"

OUT_WORLD = ROOT / "top1_radio_mbnsf_spatial_world_coordinates.png"
OUT_CENTERED = ROOT / "top1_radio_mbnsf_spatial_centroid_centered.png"

COLOR_COL = "mean_mbnsf_incompatibility"

VIEW_ELEV = 22
VIEW_AZIM = -58


# ---------------------------------------------------------------------
# 1. Load existing result only
# ---------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

required = [
    "local_index",
    "global_point_index",
    "mean_mbnsf_incompatibility",
    "zero_compatibility_fraction",
    "x0_x", "x0_y", "x0_z",
    "gt10_x", "gt10_y", "gt10_z",
    "pred10_x", "pred10_y", "pred10_z",
]

missing = [c for c in required if c not in df.columns]
assert not missing, f"Missing columns: {missing}"

# stable point-ID order
df = df.sort_values("local_index").reset_index(drop=True)


# ---------------------------------------------------------------------
# 2. Audit before visualization
# ---------------------------------------------------------------------
print("=" * 80)
print("TOP-1 MBNSF SPATIAL VISUALIZATION AUDIT")
print("=" * 80)

print("csv =", CSV_PATH)
print("n_rows =", len(df))
print("unique local_index =", df["local_index"].nunique())
print("unique global_point_index =", df["global_point_index"].nunique())

assert len(df) == 425, f"Expected 425 points, got {len(df)}"
assert df["local_index"].nunique() == 425
assert df["global_point_index"].nunique() == 425

expected_local = np.arange(425)
actual_local = np.sort(df["local_index"].to_numpy())
assert np.array_equal(actual_local, expected_local), (
    "local_index is not exactly 0..424"
)

assert not df[required].isna().any().any(), "NaN detected"

print(
    "mean_mbnsf_incompatibility range =",
    float(df[COLOR_COL].min()),
    "to",
    float(df[COLOR_COL].max()),
)

print(
    "zero_compatibility_fraction range =",
    float(df["zero_compatibility_fraction"].min()),
    "to",
    float(df["zero_compatibility_fraction"].max()),
)


# ---------------------------------------------------------------------
# 3. Coordinate arrays
# ---------------------------------------------------------------------
xyz_t0 = df[["x0_x", "x0_y", "x0_z"]].to_numpy(dtype=np.float64)

xyz_gt10 = df[
    ["gt10_x", "gt10_y", "gt10_z"]
].to_numpy(dtype=np.float64)

xyz_pred10 = df[
    ["pred10_x", "pred10_y", "pred10_z"]
].to_numpy(dtype=np.float64)

colors = df[COLOR_COL].to_numpy(dtype=np.float64)

vmin = float(colors.min())
vmax = float(colors.max())

print("t0 centroid    =", xyz_t0.mean(axis=0))
print("GT t10 centroid =", xyz_gt10.mean(axis=0))
print("Pred t10 centroid =", xyz_pred10.mean(axis=0))


# ---------------------------------------------------------------------
# Helper: equal XYZ bounds
# ---------------------------------------------------------------------
def equal_bounds(arrays, center_mode=False):
    all_xyz = np.concatenate(arrays, axis=0)

    mins = all_xyz.min(axis=0)
    maxs = all_xyz.max(axis=0)

    if center_mode:
        center = np.zeros(3)
    else:
        center = (mins + maxs) / 2.0

    span = float(np.max(maxs - mins))

    if span <= 0:
        span = 1.0

    half = span * 0.55

    return (
        (center[0] - half, center[0] + half),
        (center[1] - half, center[1] + half),
        (center[2] - half, center[2] + half),
    )


def style_axis(ax, bounds):
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_zlim(*bounds[2])

    ax.set_box_aspect((1, 1, 1))

    ax.view_init(
        elev=VIEW_ELEV,
        azim=VIEW_AZIM,
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.grid(True, alpha=0.20)


def make_figure(arrays, titles, output_path, centered=False):
    bounds = equal_bounds(
        arrays,
        center_mode=centered,
    )

    fig = plt.figure(figsize=(16, 5.6))

    axes = [
        fig.add_subplot(1, 3, i + 1, projection="3d")
        for i in range(3)
    ]

    scatter = None

    for ax, xyz, title in zip(axes, arrays, titles):

        # plot low-incompatibility first, high-incompatibility last
        # so severe points are not hidden behind ordinary points.
        order = np.argsort(colors)

        scatter = ax.scatter(
            xyz[order, 0],
            xyz[order, 1],
            xyz[order, 2],
            c=colors[order],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            s=18,
            alpha=0.95,
            linewidths=0,
            depthshade=False,
        )

        ax.set_title(
            title,
            fontsize=15,
            fontweight="bold",
            pad=14,
        )

        style_axis(ax, bounds)

    cbar = fig.colorbar(
        scatter,
        ax=axes,
        fraction=0.025,
        pad=0.035,
    )

    cbar.set_label(
        "Mean MBNSF-derived incompatibility",
        rotation=90,
        labelpad=14,
    )

    if centered:
        subtitle = (
            "Same 425 point IDs · centroid translation removed independently "
            "for display only · no rotation / scaling"
        )
    else:
        subtitle = (
            "Same 425 point IDs · raw world coordinates · no alignment"
        )

    fig.suptitle(
        "Top-1 severe radio — spatial localization of structural incompatibility",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )

    fig.text(
        0.5,
        0.02,
        subtitle,
        ha="center",
        fontsize=10,
    )

    fig.subplots_adjust(
        left=0.03,
        right=0.91,
        bottom=0.10,
        top=0.88,
        wspace=0.08,
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("saved =", output_path)


# ---------------------------------------------------------------------
# 4. RAW WORLD-COORDINATE FIGURE
# ---------------------------------------------------------------------
make_figure(
    arrays=[
        xyz_t0,
        xyz_gt10,
        xyz_pred10,
    ],
    titles=[
        "t0",
        "GT t10",
        "Pred t10",
    ],
    output_path=OUT_WORLD,
    centered=False,
)


# ---------------------------------------------------------------------
# 5. CENTROID-CENTERED STRUCTURAL FIGURE
#
# Only global translation is removed independently.
# Internal geometry, orientation, scale, and point IDs are untouched.
# ---------------------------------------------------------------------
xyz_t0_c = xyz_t0 - xyz_t0.mean(axis=0, keepdims=True)
xyz_gt10_c = xyz_gt10 - xyz_gt10.mean(axis=0, keepdims=True)
xyz_pred10_c = xyz_pred10 - xyz_pred10.mean(axis=0, keepdims=True)

make_figure(
    arrays=[
        xyz_t0_c,
        xyz_gt10_c,
        xyz_pred10_c,
    ],
    titles=[
        "t0",
        "GT t10",
        "Pred t10",
    ],
    output_path=OUT_CENTERED,
    centered=True,
)


print("=" * 80)
print("SUCCESS = True")
print("No inference performed.")
print("No new metric computed.")
print("Source = existing per-point MBNSF localization CSV only.")
print("=" * 80)
