#!/usr/bin/env bash
set -euo pipefail

BASE=/home/soenchun/cse/PointWorld_workspace
REPO="$BASE/third_party/sam2_official"
WEIGHT="$BASE/third_party/checkpoints/sam2.1_hiera_tiny.pt"
PY="$BASE/envs/pointworld-env/bin/python"

mkdir -p "$BASE/third_party/checkpoints"

if [ ! -d "$REPO/.git" ]; then
    git clone --depth 1 \
      https://github.com/facebookresearch/sam2.git "$REPO"
fi

echo "SAM2 COMMIT:"
git -C "$REPO" rev-parse HEAD

if [ ! -s "$WEIGHT" ]; then
    curl -fL --retry 3 --connect-timeout 15 \
      -o "$WEIGHT.part" \
      https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
    mv "$WEIGHT.part" "$WEIGHT"
fi

"$PY" - <<'PY'
from pathlib import Path
import torch, importlib.util

p = Path("/home/soenchun/cse/PointWorld_workspace/third_party/checkpoints/sam2.1_hiera_tiny.pt")
ckpt = torch.load(p, map_location="cpu", weights_only=True)
assert "model" in ckpt
print("CHECKPOINT VALID:", round(p.stat().st_size/1048576, 1), "MiB")

for name in ("hydra", "omegaconf", "iopath", "cv2"):
    print(name, bool(importlib.util.find_spec(name)))

print("SAM2 ASSETS READY")
PY
