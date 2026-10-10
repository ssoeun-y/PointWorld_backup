#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "diagnostics" / "results"
OUT_DIR = RESULTS / "problem_intro_panels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RADIO_FINAL = RESULTS / "radio_breakage_visualization" / "radio_breakage_final.png"

PNG_OUT = OUT_DIR / "figure1_problem_breakage.png"
PDF_OUT = OUT_DIR / "figure1_problem_breakage.pdf"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def trim_whitespace(im, white_thresh=245, pad=15):
    im = im.convert("RGBA")
    arr = np.array(im)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3] > 0
    mask = alpha & (rgb < white_thresh).any(axis=2)
    if not mask.any():
        return im
    ys, xs = np.where(mask)
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, im.width)
    y1 = min(int(ys.max()) + pad + 1, im.height)
    return im.crop((x0, y0, x1, y1))


def load_original_object_crop(path: Path):
    """
    Crop the left-most GT rigid object region from radio_breakage_final.png.
    Tuned for the current wide layout.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    im = Image.open(path).convert("RGBA")
    W, H = im.size

    crop_box = (
        int(0.035 * W),   # left
        int(0.20 * H),    # top
        int(0.235 * W),   # right
        int(0.86 * H),    # bottom
    )
    crop = im.crop(crop_box)
    crop = trim_whitespace(crop, white_thresh=245, pad=20)
    return crop


def find_best_qual_folder(results_root: Path):
    """
    Find a problem_failure_qualitative* folder that contains t*_prediction.png.
    Prefer the one with the most frames.
    """
    cand = []
    for d in results_root.glob("problem_failure_qualitative*"):
        if not d.is_dir():
            continue
        imgs = list(d.glob("t*_prediction.png"))
        if imgs:
            cand.append((len(imgs), d))
    if not cand:
        raise RuntimeError("No problem_failure_qualitative* folder with t*_prediction.png found.")
    cand.sort(key=lambda x: (-x[0], str(x[1])))
    return cand[0][1]


def load_rollout_images(folder: Path):
    pat = re.compile(r"t(\d+)_prediction\.png$")
    items = []
    for p in folder.glob("t*_prediction.png"):
        m = pat.match(p.name)
        if m:
            t = int(m.group(1))
            items.append((t, p))
    if not items:
        raise RuntimeError(f"No t*_prediction.png found in {folder}")
    items = sorted(items, key=lambda x: x[0])

    # keep all found frames, but if too many, subsample a few representative ones
    if len(items) > 4:
        keep_t = [items[0][0], items[len(items)//3][0], items[(2*len(items))//3][0], items[-1][0]]
        keep_t = sorted(set(keep_t))
        items = [(t, p) for (t, p) in items if t in keep_t]

    loaded = []
    for t, p in items:
        im = Image.open(p).convert("RGBA")
        im = trim_whitespace(im, white_thresh=245, pad=12)
        loaded.append((t, im, p))
    return loaded


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
orig_obj = load_original_object_crop(RADIO_FINAL)
qual_dir = find_best_qual_folder(RESULTS)
rollout = load_rollout_images(qual_dir)

print("Using qualitative folder:", qual_dir)
for t, _, p in rollout:
    print(f"  frame t={t}: {p}")

n = len(rollout)

fig = plt.figure(figsize=(4.2 + 2.35 * n, 5.2), dpi=300)
outer = fig.add_gridspec(
    1, 2,
    width_ratios=[1.0, 2.55],
    left=0.03, right=0.99, top=0.88, bottom=0.16, wspace=0.06
)

# ------------------------------------------------------------
# Left panel: original rigid object
# ------------------------------------------------------------
axL = fig.add_subplot(outer[0, 0])
axL.imshow(orig_obj)
axL.axis("off")
axL.set_title("(a)  Original rigid object", loc="left", fontsize=22, fontweight="bold", pad=8)

# ------------------------------------------------------------
# Right panel: predicted rigid-object shape over rollout time
# ------------------------------------------------------------
right = outer[0, 1].subgridspec(1, n, wspace=0.06)
axes = []

for i, (t, im, _) in enumerate(rollout):
    ax = fig.add_subplot(right[0, i])
    ax.imshow(im)
    ax.axis("off")
    ax.set_title(rf"$t_{{{t}}}$", fontsize=22, pad=6)
    axes.append(ax)

x0 = axes[0].get_position().x0
x1 = axes[-1].get_position().x1
xc = 0.5 * (x0 + x1)

fig.text(
    xc, 0.91,
    "(b)  Predicted rigid-object shape under a high-dynamics action",
    ha="center", va="center",
    fontsize=22, fontweight="bold"
)

# vertical separator
sep_x = axL.get_position().x1 + 0.008
fig.add_artist(Line2D(
    [sep_x, sep_x], [0.14, 0.89],
    transform=fig.transFigure,
    linestyle="--", linewidth=1.5, color="0.55"
))

# arrows
for i in range(n - 1):
    p1 = axes[i].get_position()
    p2 = axes[i + 1].get_position()
    xa = 0.5 * (p1.x1 + p2.x0)
    ya = 0.5 * (p1.y0 + p1.y1)
    fig.text(xa, ya, "→", ha="center", va="center", fontsize=42, color="0.6")

# bottom message
fig.text(
    xc, 0.07,
    "Rigid-object shape consistency deteriorates over rollout time.",
    ha="center", va="center",
    fontsize=18, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef1f6", edgecolor="none")
)

fig.savefig(PNG_OUT, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(PDF_OUT, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("\n====================================================")
print("DONE")
print("====================================================")
print(f"PNG = {PNG_OUT}")
print(f"PDF = {PDF_OUT}")
print("====================================================")
