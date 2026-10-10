#!/usr/bin/env bash
set -euo pipefail

MODEL="/mnt/hdd/sunjaeyoon/cse/PointWorld_workspace/checkpoints/pointworld/small-droid/model-best.pt"
DATA_ROOT="$(pwd)/diagnostics/data/visibility_single_sample_wds_20260805"
RESULT_ROOT="$(pwd)/diagnostics/results/visibility_multiclip_perturbations_20260805"

mkdir -p "${RESULT_ROOT}"

GPU_ID=4

echo "Selected GPU: ${GPU_ID}"
nvidia-smi -i "${GPU_ID}"

TOTAL=70
COMPLETED=0

for CLIP_ID in clip01 clip02 clip03 clip04 clip05; do
  DATA_DIR="${DATA_ROOT}/${CLIP_ID}"

  for JOINT_INDEX in 0 1 2 3 4 5 6; do
    JOINT_NUMBER=$((JOINT_INDEX + 1))

    for AMPLITUDE in 0.15 -0.15; do
      if [[ "${AMPLITUDE}" == "0.15" ]]; then
        SIGN_LABEL="plus015"
      else
        SIGN_LABEL="minus015"
      fi

      ACTION_ID="joint${JOINT_NUMBER}_${SIGN_LABEL}"
      RUN_ID="visibility_${CLIP_ID}_${ACTION_ID}_20260805"

      INTERNAL_DIR="diagnostics/results/action_perturb_${RUN_ID}"
      FINAL_DIR="${RESULT_ROOT}/${CLIP_ID}/${ACTION_ID}"
      LOG_PATH="${RESULT_ROOT}/${CLIP_ID}/${ACTION_ID}.log"

      mkdir -p "${RESULT_ROOT}/${CLIP_ID}"

      if [[ -f "${FINAL_DIR}/perturb.pt" &&
            -f "${FINAL_DIR}/perturb.json" ]]; then
        COMPLETED=$((COMPLETED + 1))
        echo "[SKIP ${COMPLETED}/${TOTAL}] ${CLIP_ID} ${ACTION_ID}"
        continue
      fi

      if [[ -e "${INTERNAL_DIR}" ]]; then
        echo "[ERROR] Unexpected internal directory:"
        echo "${INTERNAL_DIR}"
        exit 1
      fi

      if [[ -e "${FINAL_DIR}" ]]; then
        echo "[ERROR] Incomplete final directory:"
        echo "${FINAL_DIR}"
        exit 1
      fi

      echo
      echo "============================================================"
      echo "[RUN $((COMPLETED + 1))/${TOTAL}]"
      echo "Clip: ${CLIP_ID}"
      echo "Action: ${ACTION_ID}"
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

      if [[ ! -f "${INTERNAL_DIR}/perturb.pt" ||
            ! -f "${INTERNAL_DIR}/perturb.json" ]]; then
        echo "[ERROR] Missing captured output: ${RUN_ID}"
        exit 1
      fi

      mkdir -p "${FINAL_DIR}"

      mv "${INTERNAL_DIR}/perturb.pt" \
         "${FINAL_DIR}/perturb.pt"

      mv "${INTERNAL_DIR}/perturb.json" \
         "${FINAL_DIR}/perturb.json"

      rmdir "${INTERNAL_DIR}"

      COMPLETED=$((COMPLETED + 1))
      echo "[DONE ${COMPLETED}/${TOTAL}] ${CLIP_ID} ${ACTION_ID}"
    done
  done
done

echo
echo "============================================================"
echo "MULTICLIP PERTURBATION RUN COMPLETE"
echo "completed=${COMPLETED}"
echo "expected=${TOTAL}"
echo "============================================================"
