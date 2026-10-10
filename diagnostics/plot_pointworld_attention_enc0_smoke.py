#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INFILE = Path(
    "diagnostics/results/"
    "pointworld_attention_enc0_smoke/"
    "scene_point_robot_attention.csv"
)

OUTDIR = Path(
    "diagnostics/results/"
    "pointworld_attention_enc0_smoke"
)

df = pd.read_csv(INFILE)

x = df["x"].to_numpy()
y = df["y"].to_numpy()
z = df["z"].to_numpy()

attn = df["robot_attention_mean"].to_numpy()
delta = df["attention_minus_candidate"].to_numpy()


# ------------------------------------------------------------
# 1. Raw robot/action attention
# ------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    x, y, z,
    c=attn,
    s=3,
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(
    "enc0.block0: Scene → Robot/Action Attention"
)

fig.colorbar(
    sc,
    ax=ax,
    label="Robot/action attention mass",
    shrink=0.7,
)

fig.tight_layout()
fig.savefig(
    OUTDIR / "scene_to_robot_attention_3d.png",
    dpi=200,
)
plt.close(fig)


# ------------------------------------------------------------
# 2. Attention corrected for candidate composition
# ------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(
    x, y, z,
    c=delta,
    s=3,
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(
    "enc0.block0: Attention − Robot Candidate Fraction"
)

fig.colorbar(
    sc,
    ax=ax,
    label="Attention minus candidate fraction",
    shrink=0.7,
)

fig.tight_layout()
fig.savefig(
    OUTDIR / "scene_to_robot_attention_minus_candidate_3d.png",
    dpi=200,
)
plt.close(fig)


print(
    "SAVED =",
    OUTDIR / "scene_to_robot_attention_3d.png",
)

print(
    "SAVED =",
    OUTDIR / "scene_to_robot_attention_minus_candidate_3d.png",
)

print("NO NEW INFERENCE = True")
