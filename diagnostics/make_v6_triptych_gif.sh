#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <name> <npz_path> <object_id> [gpu_id]"
  exit 1
fi

NAME="$1"
NPZ="$2"
OBJECT_ID="$3"
GPU_ID="${4:-1}"

cd /home/soenchun/cse/PointWorld_workspace/PointWorld

PY=/home/soenchun/cse/PointWorld_workspace/envs/pointworld-env/bin/python
VIS=diagnostics/visualize_grasp_release_adaptive_contact_v6_fasttarget.py
VIS_OUT=diagnostics/results/grasp_release_adaptive_contact_v6_visuals
FRAME_DIR="diagnostics/results/gif_frames/${NAME}"
GIF_OUT="diagnostics/results/gif_frames/${NAME}__adaptive_contact_v6.gif"
LOG_DIR="diagnostics/results/gif_frames/${NAME}_logs"

mkdir -p "$FRAME_DIR" "$LOG_DIR"

KEY=$(basename "$NPZ" .npz)

echo "================================================"
echo "NAME   = $NAME"
echo "NPZ    = $NPZ"
echo "OBJECT = $OBJECT_ID"
echo "GPU    = $GPU_ID"
echo "================================================"

for T in $(seq 1 10)
do
  echo
  echo "[RUN] frame t=$T"

  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PYTHONPATH=/home/soenchun/cse/PointWorld_workspace/PointWorld \
  "$PY" "$VIS" \
    --npz "$NPZ" \
    --t "$T" \
    --object_id "$OBJECT_ID" \
    > "${LOG_DIR}/t${T}.log" 2>&1

  LATEST=$(
    ls -t "${VIS_OUT}/${KEY}__t${T}__adaptive_contact_v6__"*.png \
    2>/dev/null | head -n 1
  )

  if [ -z "${LATEST:-}" ]; then
    echo "ERROR: could not find rendered PNG for t=$T"
    echo "Check log: ${LOG_DIR}/t${T}.log"
    exit 1
  fi

  cp "$LATEST" "${FRAME_DIR}/frame_$(printf '%02d' "$T").png"
  echo "[OK] ${FRAME_DIR}/frame_$(printf '%02d' "$T").png"
done

"$PY" - <<PY
from pathlib import Path
from PIL import Image

frame_dir = Path("$FRAME_DIR")
gif_out = Path("$GIF_OUT")

frames = []
for p in sorted(frame_dir.glob("frame_*.png")):
    img = Image.open(p).convert("RGB")
    frames.append(img)

if len(frames) != 10:
    raise RuntimeError(f"Expected 10 frames, got {len(frames)}")

frames[0].save(
    gif_out,
    save_all=True,
    append_images=frames[1:],
    duration=450,   # ms per frame
    loop=0,
)

print("GIF_SAVED =", gif_out)
PY

echo
echo "DONE"
echo "FRAME_DIR = $FRAME_DIR"
echo "GIF_OUT   = $GIF_OUT"
