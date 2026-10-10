#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def trim_whitespace(im, white_thresh=245, pad=20):
    im = im.convert("RGBA")
    arr = np.array(im)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3] > 0

    # "not almost-white" pixels
    mask = alpha & (rgb < white_thresh).any(axis=2)
    if not mask.any():
        return im

    ys, xs = np.where(mask)
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, im.width)
    y1 = min(int(ys.max()) + pad + 1, im.height)
    return im.crop((x0, y0, x1, y1))


def load_original_object_crop(radio_png: Path):
    """
    Crop only the left-most '(a) Full segmented radio GT reference' object area
    from radio_breakage_final.png, then trim whitespace.
    """
    im = Image.open(radio_png).convert("RGBA")
    W, H = im.size

    # Tuned for the current 6859x1792 wide figure.
    # Keeps the object itself, avoids most titles/text.
    crop = im.crop((
        int(0.035 * W),   # left
        int(0.18 * H),    # top
        int(0.245 * W),   # right
        int(0.88 * H),    # bottom
    ))
    crop = trim_whitespace(crop, white_thresh=245, pad=25)
    return crop


def find_time_step_images(results_dir: Path):
    """
    Find t*_prediction.png under problem_failure_qualitative* folders.
    """
    found = []
    pat = re.compile(r"t(\d+)_prediction\.png$")
    for p in results_dir.rglob("t*_prediction.png"):
        if "problem_failure_qualitative" not in str(p.parent):
            continue
        m = pat.search(p.name)
        if m:
            t = int(m.group(1))
            found.append((t, p))

    found = sorted(found, key=lambda x: x[0])

    # Prefer canonical frames if they exist
    preferred = [1, 3, 5, 10]
    preferred_found = [(t, p) for (t, p) in found if t in preferred]
    if preferred_found:
        return preferred_found

    return found


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
repo_root = Path(__file__).resolve().parents[1]
results_dir = repo_root / "diagnostics" / "results"

radio_png = results_dir / "radio_breakage_visualization" / "radio_breakage_final.png"
if not radio_png.exists():
    raise FileNotFoundError(f"Missing: {radio_png}")

time_imgs = find_time_step_images(results_dir)
if len(time_imgs) < 2:
    raise RuntimeError(
        "Could not find enough t*_prediction.png images under diagnostics/results/problem_failure_qualitative*"
    )

# load images
orig_obj = load_original_object_crop(radio_png)
rollout = []
for t, p in time_imgs:
    im = Image.open(p).convert("RGBA")
    im = trim_whitespace(im, white_thresh=245, pad=15)
    rollout.append((t, im, p))

print("Using rollout snapshots:")
for t, _, p in rollout:
    print(f"  t={t}: {p}")

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
n = len(rollout)

fig = plt.figure(figsize=(4.2 + 2.65 * n, 6.2), dpi=300)
outer = fig.add_gridspec(
    1, 2,
    width_ratios=[1.05, 2.55],
    left=0.03, right=0.99, top=0.89, bottom=0.15, wspace=0.06
)

# Left panel: original rigid object
axL = fig.add_subplot(outer[0, 0])
axL.imshow(orig_obj)
axL.axis("off")
axL.set_title("(a)  Original rigid object", loc="left", fontsize=24, fontweight="bold", pad=10)

# Right panel: rollout deterioration
right = outer[0, 1].subgridspec(1, n, wspace=0.08)
axes = []
for i, (t, im, _) in enumerate(rollout):
    ax = fig.add_subplot(right[0, i])
    ax.imshow(im)
    ax.axis("off")
    ax.set_title(rf"$t_{{{t}}}$", fontsize=24, pad=10)
    axes.append(ax)

# Right title
x0 = axes[0].get_position().x0
x1 = axes[-1].get_position().x1
xc = 0.5 * (x0 + x1)
fig.text(
    xc, 0.92,
    "(b)  Predicted object shape over rollout time",
    ha="center", va="center",
    fontsize=24, fontweight="bold"
)

# Vertical dashed separator
sep_x = axL.get_position().x1 + 0.01
fig.add_artist(Line2D(
    [sep_x, sep_x], [0.13, 0.90],
    transform=fig.transFigure,
    linestyle="--", linewidth=1.6, color="0.55"
))

# Arrows between rollout panels
for i in range(n - 1):
    p1 = axes[i].get_position()
    p2 = axes[i + 1].get_position()
    xa = 0.5 * (p1.x1 + p2.x0)
    ya = 0.5 * (p1.y0 + p1.y1)
    fig.text(xa, ya, "→", ha="center", va="center", fontsize=46, color="0.60")

# Bottom message
fig.text(
    xc, 0.07,
    "Rigid-object shape consistency deteriorates over rollout time.",
    ha="center", va="center",
    fontsize=20, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef1f6", edgecolor="none")
)

out_dir = results_dir / "problem_intro_panels"
out_dir.mkdir(parents=True, exist_ok=True)

png_path = out_dir / "problem_failure_simple.png"
pdf_path = out_dir / "problem_failure_simple.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("\nSaved:")
print(f"PNG = {png_path}")
print(f"PDF = {pdf_path}")
