#!/usr/bin/env bash
set -euo pipefail

MODEL="/mnt/hdd/sunjaeyoon/cse/PointWorld_workspace/checkpoints/pointworld/small-droid/model-best.pt"
DATA_ROOT="$(pwd)/diagnostics/data/visibility_single_sample_wds_20260805"
RESULT_ROOT="$(pwd)/diagnostics/results/visibility_multiclip_identity_20260805"

mkdir -p "${RESULT_ROOT}"

GPU_ID=$(
  nvidia-smi \
    --query-gpu=index,memory.used \
    --format=csv,noheader,nounits |
  sort -t, -k2,2n |
  head -n 1 |
  cut -d, -f1 |
  tr -d ' '
)

echo "Selected GPU: ${GPU_ID}"
nvidia-smi -i "${GPU_ID}"

for CLIP_ID in clip01 clip02 clip03 clip04 clip05; do
  DATA_DIR="${DATA_ROOT}/${CLIP_ID}"
  RUN_ID="multiclip_${CLIP_ID}_identity_20260805"

  INTERNAL_DIR="diagnostics/results/action_perturb_${RUN_ID}"
  FINAL_DIR="${RESULT_ROOT}/${CLIP_ID}"
  LOG_PATH="${RESULT_ROOT}/${CLIP_ID}.log"

  if [[ -f "${FINAL_DIR}/identity.pt" && -f "${FINAL_DIR}/identity.json" ]]; then
    echo "[SKIP] ${CLIP_ID}"
    continue
  fi

  if [[ -e "${INTERNAL_DIR}" ]]; then
    echo "[ERROR] Unexpected existing directory: ${INTERNAL_DIR}"
    exit 1
  fi

  echo
  echo "============================================================"
  echo "[IDENTITY] ${CLIP_ID}"
  echo "DATA: ${DATA_DIR}"
  echo "GPU: ${GPU_ID}"
  echo "============================================================"

  ACTION_PERTURB_RUN="${RUN_ID}" \
  ACTION_PERTURB_JOINT="0" \
  ACTION_PERTURB_AMPLITUDE="0.0" \
  PYTHONPATH="$(pwd)" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONUNBUFFERED=1 \
  python diagnostics/action_perturb_capture.py \
    --model_path "${MODEL}" \
    --domains=droid \
    --data_dirs="${DATA_DIR}" \
    --ptv3_size=small \
    --batch_size=1 \
    --num_workers=0 \
    --eval_num_workers=0 \
    --eval_num_batches=1 \
    --allow_missing_confidence_mask=true \
    --eval_skip_viz=true \
    --disable_compile=true \
    --seed=42 \
    2>&1 | tee "${LOG_PATH}"

  mkdir -p "${FINAL_DIR}"

  mv "${INTERNAL_DIR}/perturb.pt" \
     "${FINAL_DIR}/identity.pt"

  mv "${INTERNAL_DIR}/perturb.json" \
     "${FINAL_DIR}/identity.json"

  rmdir "${INTERNAL_DIR}"

  echo "[DONE] ${CLIP_ID}"
done
