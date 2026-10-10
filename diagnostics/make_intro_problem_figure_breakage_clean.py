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
PNG_OUT = OUT_DIR / "figure1_problem_breakage_clean.png"
PDF_OUT = OUT_DIR / "figure1_problem_breakage_clean.pdf"

# ------------------------------------------------------------
# Plot style
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
})

# ------------------------------------------------------------
# Image helpers
# ------------------------------------------------------------
def white_to_transparent(im, white_thresh=245):
    im = im.convert("RGBA")
    arr = np.array(im)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    white = (rgb[:, :, 0] >= white_thresh) & (rgb[:, :, 1] >= white_thresh) & (rgb[:, :, 2] >= white_thresh)
    alpha[white] = 0
    arr[:, :, 3] = alpha
    return Image.fromarray(arr)

def trim_by_mask(im, mask, pad=10):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return im
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, im.width)
    y1 = min(int(ys.max()) + pad + 1, im.height)
    return im.crop((x0, y0, x1, y1))

def find_content_mask(im, white_thresh=245, black_thresh=70):
    """
    Keep colored / gray geometry, remove white background and most black text.
    """
    arr = np.array(im.convert("RGBA"))
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3] > 0

    not_white = (rgb < white_thresh).any(axis=2)
    not_black_text = ~((rgb[:, :, 0] < black_thresh) &
                       (rgb[:, :, 1] < black_thresh) &
                       (rgb[:, :, 2] < black_thresh))

    mask = alpha & not_white & not_black_text
    return mask

def crop_original_object_from_radio_final(path: Path):
    """
    Crop only the left-most GT object from radio_breakage_final.png.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")

    im = Image.open(path).convert("RGBA")
    W, H = im.size

    # tuned crop for current radio_breakage_final layout
    crop = im.crop((
        int(0.03 * W),   # left
        int(0.18 * H),   # top
        int(0.23 * W),   # right
        int(0.84 * H),   # bottom
    ))

    mask = find_content_mask(crop)
    crop = trim_by_mask(crop, mask, pad=20)
    crop = white_to_transparent(crop)
    return crop

def choose_qual_dir(results_root: Path):
    preferred = results_root / "problem_failure_qualitative_final"
    if preferred.exists():
        return preferred

    cands = []
    for d in results_root.glob("problem_failure_qualitative*"):
        if d.is_dir():
            imgs = list(d.glob("t*_prediction.png"))
            if imgs:
                cands.append((len(imgs), d))
    if not cands:
        raise RuntimeError("No problem_failure_qualitative* directory with t*_prediction.png found.")
    cands.sort(key=lambda x: (-x[0], str(x[1])))
    return cands[0][1]

def pick_frames(img_dir: Path, desired=(1, 5, 10)):
    pat = re.compile(r"t(\d+)_prediction\.png$")
    found = {}
    for p in img_dir.glob("t*_prediction.png"):
        m = pat.match(p.name)
        if m:
            found[int(m.group(1))] = p

    if not found:
        raise RuntimeError(f"No t*_prediction.png found in {img_dir}")

    picked = []
    existing = sorted(found.keys())
    for t in desired:
        if t in found:
            picked.append((t, found[t]))
        else:
            nearest = min(existing, key=lambda x: abs(x - t))
            picked.append((nearest, found[nearest]))

    # remove accidental duplicates while preserving order
    out = []
    seen = set()
    for t, p in picked:
        if t not in seen:
            out.append((t, p))
            seen.add(t)
    return out

def crop_prediction_object(path: Path):
    """
    Crop only the geometric object region from a t*_prediction.png.
    Removes most embedded text and whitespace.
    """
    im = Image.open(path).convert("RGBA")
    arr = np.array(im)
    H, W = arr.shape[:2]

    # first pass: remove background + black text
    mask = find_content_mask(im)

    # If bottom captions remain, suppress bottom 22% area from bbox estimation
    mask_for_bbox = mask.copy()
    mask_for_bbox[int(0.78 * H):, :] = False

    cropped = trim_by_mask(im, mask_for_bbox, pad=18)

    # second pass trim
    mask2 = find_content_mask(cropped)
    cropped = trim_by_mask(cropped, mask2, pad=15)
    cropped = white_to_transparent(cropped)
    return cropped

# ------------------------------------------------------------
# Load assets
# ------------------------------------------------------------
orig = crop_original_object_from_radio_final(RADIO_FINAL)
qual_dir = choose_qual_dir(RESULTS)
frames = pick_frames(qual_dir, desired=(1, 5, 10))
rollout = [(t, crop_prediction_object(p), p) for t, p in frames]

print("Using qualitative folder:", qual_dir)
for t, _, p in rollout:
    print(f"  t={t}: {p}")

# ------------------------------------------------------------
# Build figure
# ------------------------------------------------------------
n = len(rollout)

fig = plt.figure(figsize=(4.4 + 2.7 * n, 5.5), dpi=300)
outer = fig.add_gridspec(
    1, 2,
    width_ratios=[1.0, 2.65],
    left=0.03, right=0.99, top=0.88, bottom=0.17, wspace=0.04
)

# Left panel
axL = fig.add_subplot(outer[0, 0])
axL.imshow(orig)
axL.axis("off")
axL.set_title("(a) Original rigid object", loc="left", pad=8)

# Right panel grid
right = outer[0, 1].subgridspec(1, n, wspace=0.10)
axes = []
for i, (t, im, _) in enumerate(rollout):
    ax = fig.add_subplot(right[0, i])
    ax.imshow(im)
    ax.axis("off")
    ax.set_title(rf"$t_{{{t}}}$", fontsize=18, pad=8)
    axes.append(ax)

# Right panel title
x0 = axes[0].get_position().x0
x1 = axes[-1].get_position().x1
xc = 0.5 * (x0 + x1)

fig.text(
    xc, 0.91,
    "(b) Severe rigid-object shape breakage under a high-dynamics action",
    ha="center", va="center",
    fontsize=18, fontweight="bold"
)

# Vertical separator
sep_x = axL.get_position().x1 + 0.008
fig.add_artist(Line2D(
    [sep_x, sep_x], [0.15, 0.88],
    transform=fig.transFigure,
    linestyle="--", linewidth=1.3, color="0.55"
))

# arrows
for i in range(n - 1):
    p1 = axes[i].get_position()
    p2 = axes[i + 1].get_position()
    xa = 0.5 * (p1.x1 + p2.x0)
    ya = 0.50 * (p1.y0 + p1.y1)
    fig.text(xa, ya, "→", ha="center", va="center", fontsize=34, color="0.55")

# Bottom message
fig.text(
    xc, 0.08,
    "The predicted rigid-object shape progressively collapses and scatters over rollout time.",
    ha="center", va="center",
    fontsize=15, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.42", facecolor="#eef1f6", edgecolor="none")
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
