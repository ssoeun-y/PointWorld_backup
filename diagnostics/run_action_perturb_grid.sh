#!/usr/bin/env bash
set -u

MODEL="/mnt/hdd/sunjaeyoon/cse/PointWorld_workspace/checkpoints/pointworld/small-droid/model-best.pt"
DATA="/mnt/hdd/sunjaeyoon/cse/PointWorld_workspace/datasets/droid/wds"

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

for JOINT_INDEX in 0 1 2 3 4 5 6; do
  JOINT_NUMBER=$((JOINT_INDEX + 1))

  for AMPLITUDE in 0.15 -0.15; do
    if [[ "${AMPLITUDE}" == "0.15" ]]; then
      SIGN_LABEL="plus015"
    else
      SIGN_LABEL="minus015"
    fi

    RUN_ID="joint${JOINT_NUMBER}_${SIGN_LABEL}_20260805"
    OUT_DIR="diagnostics/results/action_perturb_${RUN_ID}"
    LOG_PATH="diagnostics/results/action_perturb_${RUN_ID}.log"

    if [[ -f "${OUT_DIR}/perturb.pt" && -f "${OUT_DIR}/perturb.json" ]]; then
      echo "[SKIP] Existing result: ${RUN_ID}"
      continue
    fi

    echo
    echo "============================================================"
    echo "[RUN] ${RUN_ID}"
    echo "Joint index: ${JOINT_INDEX}"
    echo "Amplitude: ${AMPLITUDE}"
    echo "GPU: ${GPU_ID}"
    echo "============================================================"

    ACTION_PERTURB_RUN="${RUN_ID}" \
    ACTION_PERTURB_JOINT="${JOINT_INDEX}" \
    ACTION_PERTURB_AMPLITUDE="${AMPLITUDE}" \
    PYTHONPATH="$(pwd)" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONUNBUFFERED=1 \
    python diagnostics/action_perturb_capture.py \
      --model_path "${MODEL}" \
      --domains=droid \
      --data_dirs="${DATA}" \
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

    STATUS=${PIPESTATUS[0]}

    if [[ "${STATUS}" -ne 0 ]]; then
      echo "[FAILED] ${RUN_ID}, exit=${STATUS}"
    else
      echo "[DONE] ${RUN_ID}"
    fi
  done
done

echo
echo "===== COMPLETED FILE COUNT ====="
find diagnostics/results \
  -maxdepth 2 \
  -path "*/action_perturb_joint*_20260805/perturb.json" \
  -type f |
sort |
wc -l
