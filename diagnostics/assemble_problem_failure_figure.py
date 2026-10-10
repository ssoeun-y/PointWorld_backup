import argparse
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--context", type=str, required=True)
parser.add_argument("--t1", type=str, required=True)
parser.add_argument("--t5", type=str, required=True)
parser.add_argument("--t10", type=str, required=True)
parser.add_argument("--title", type=str,
                    default="Point clouds lose rigid-shape consistency as objects move fast")
parser.add_argument("--subtitle", type=str,
                    default="A representative high-dynamics rollout selected near the median PA-MPJPE degradation of the high-dynamics population.")
parser.add_argument("--caption", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

context = Path(args.context)
t1 = Path(args.t1)
t5 = Path(args.t5)
t10 = Path(args.t10)

for p in [context, t1, t5, t10]:
    if not p.exists():
        raise FileNotFoundError(p)

fig = plt.figure(figsize=(13.5, 4.7))
gs = fig.add_gridspec(
    1, 4,
    width_ratios=[1.20, 1.0, 1.0, 1.0],
    left=0.035, right=0.985,
    top=0.80, bottom=0.20,
    wspace=0.08,
)

paths = [context, t1, t5, t10]
titles = ["Context", "t = 1", "t = 5", "t = 10"]
axes = []

for i, (p, ttl) in enumerate(zip(paths, titles)):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(Image.open(p).convert("RGB"))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(ttl, fontsize=12, pad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    axes.append(ax)

# red highlight on final failure panel
for sp in axes[-1].spines.values():
    sp.set_visible(True)
    sp.set_linewidth(3.0)
    sp.set_edgecolor("#D95F3A")

# subtle cue below final panel
axes[-1].text(
    0.5, -0.07,
    "shape collapse",
    transform=axes[-1].transAxes,
    ha="center", va="top",
    fontsize=10, color="#D95F3A"
)

fig.text(
    0.5, 0.92,
    args.title,
    ha="center", va="center",
    fontsize=18, weight="bold"
)

fig.text(
    0.5, 0.865,
    args.subtitle,
    ha="center", va="center",
    fontsize=10.5, color="0.25"
)

fig.text(
    0.5, 0.08,
    args.caption,
    ha="center", va="center",
    fontsize=10
)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
plt.close(fig)

print("=" * 80)
print("FINAL QUALITATIVE FAILURE FIGURE SAVED")
print(out)
print(out.with_suffix(".pdf"))
print("=" * 80)
