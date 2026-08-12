#!/usr/bin/env bash
set -euo pipefail

# Evaluate the retrained two-channel PCD checkpoints on RTX 3090.

PROJECT_ROOT="${PROJECT_ROOT:-/linyuanping/dzs/codes/TSTTA}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints/pcd}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/pcd}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/pcd}"
DATA_BASE_DIR="${DATA_BASE_DIR:-${PROJECT_ROOT}/data}"
GPU_ID="${GPU_ID:-0}"

MODELS=(
    "FreTSPCD"
    "PatchTSTPCD"
    "DLinearPCD"
    "iTransformerPCD"
    "MICNPCD"
    "OLSPCD"
)
PRED_LENS=(24 48 96 192)
DIRECTIONS=("455:10" "10:455")

DATASET="eVED"
LABEL_LEN=12
PATCH_LEN=8
STRIDE=4
LR="1e-4"
EPOCHS=30
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED=0

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

python -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable.")
print(f"[CUDA] device={torch.cuda.get_device_name(0)}")
print(f"[CUDA] torch={torch.__version__}, cuda={torch.version.cuda}")
'

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"
SUMMARY_FILE="${RESULT_ROOT}/summary.tsv"
printf "model\tpred_len\ttrain_id\ttest_id\tmetrics\n" > "${SUMMARY_FILE}"

cd "${PROJECT_ROOT}"

for MODEL in "${MODELS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
        for DIRECTION in "${DIRECTIONS[@]}"; do
            IFS=: read -r TRAIN_ID TEST_ID <<< "${DIRECTION}"
            TRAIN_IDS="['${TRAIN_ID}']"
            TEST_IDS="['${TEST_ID}']"
            VAL_IDS="${TRAIN_IDS}"

            EXP_NAME="${DATASET}_${PRED_LEN}_${LR}_ep_${EPOCHS}_${TRAIN_ID}_to_${TEST_ID}"
            CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${MODEL}/${EXP_NAME}"
            CHECKPOINT_FILE="${CHECKPOINT_DIR}/checkpoint_best.pth"
            RESULT_DIR="${RESULT_ROOT}/${MODEL}/${DATASET}_P${PRED_LEN}_${TRAIN_ID}_to_${TEST_ID}"
            LOG_FILE="${LOG_ROOT}/${MODEL}_${DATASET}_P${PRED_LEN}_${TRAIN_ID}_to_${TEST_ID}.log"

            if [[ ! -s "${CHECKPOINT_FILE}" ]]; then
                echo "[ERROR] checkpoint not found: ${CHECKPOINT_FILE}" >&2
                exit 1
            fi
            mkdir -p "${RESULT_DIR}"

            echo "================================================================"
            echo "[NEW/GPU] model=${MODEL}, pred_len=${PRED_LEN}, ${TRAIN_ID}->${TEST_ID}, GPU=${GPU_ID}"
            echo "[NEW/GPU] checkpoint=${CHECKPOINT_DIR}"
            echo "[NEW/GPU] result=${RESULT_DIR}"

            python main.py \
                SEED "${SEED}" \
                DATA.BASE_DIR "${DATA_BASE_DIR}" \
                DATA.NAME "${DATASET}" \
                DATA.SEQ_LEN "${PRED_LEN}" \
                DATA.PRED_LEN "${PRED_LEN}" \
                DATA.LABEL_LEN "${LABEL_LEN}" \
                DATA.MIN_TEST_LEN 300 \
                DATA.TRAIN_VEHICLE_IDS "${TRAIN_IDS}" \
                DATA.VAL_VEHICLE_IDS "${VAL_IDS}" \
                DATA.TEST_VEHICLE_IDS "${TEST_IDS}" \
                MODEL.NAME "${MODEL}" \
                MODEL.seq_len "${PRED_LEN}" \
                MODEL.pred_len "${PRED_LEN}" \
                MODEL.label_len "${LABEL_LEN}" \
                MODEL.patch_len "${PATCH_LEN}" \
                MODEL.stride "${STRIDE}" \
                NORM_MODULE.ENABLE False \
                TRAIN.ENABLE False \
                TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
                TEST.ENABLE True \
                TEST.BATCH_SIZE "${BATCH_SIZE}" \
                TEST.SHUFFLE False \
                TEST.DROP_LAST False \
                DATA_LOADER.NUM_WORKERS "${NUM_WORKERS}" \
                TTA.ENABLE False \
                TTA.DOMAIN_SHIFT False \
                RESULT_DIR "${RESULT_DIR}" \
                2>&1 | tee "${LOG_FILE}"

            if [[ ! -s "${RESULT_DIR}/test.txt" ]]; then
                echo "[ERROR] result was not created: ${RESULT_DIR}/test.txt" >&2
                exit 1
            fi
            METRICS="$(tr '\n' ' ' < "${RESULT_DIR}/test.txt")"
            printf "%s\t%s\t%s\t%s\t%s\n" \
                "${MODEL}" "${PRED_LEN}" "${TRAIN_ID}" "${TEST_ID}" "${METRICS}" \
                >> "${SUMMARY_FILE}"
        done
    done
done

echo "================================================================"
echo "[PASS] New GPU checkpoint evaluation completed."
echo "[PASS] Summary: ${SUMMARY_FILE}"