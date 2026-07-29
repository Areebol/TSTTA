#!/usr/bin/env bash
set -euo pipefail

# Full CUDA training for all PCD forecasting models on eVED.
# The experiment settings follow scripts/0329/run_train_eved_455_2_10.sh,
# but run sequentially on one CUDA GPU to avoid oversubscribing RTX 3090 memory.
#
# Usage:
#   bash scripts/0728/run_train_all_pcd_gpu.sh
#   GPU_ID=1 BATCH_SIZE=32 bash scripts/0728/run_train_all_pcd_gpu.sh

MODELS=(
    "FreTSPCD"
    "PatchTSTPCD"
    "DLinearPCD"
    "iTransformerPCD"
    "MICNPCD"
    "OLSPCD"
)
PRED_LENS=(24 48 96 192)

DATASET="eVED"
LABEL_LEN=12
PATCH_LEN=8
STRIDE=4

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-1e-4}"
GPU_ID="${GPU_ID:-2,3}"

TRAIN_IDS="${TRAIN_IDS:-['455']}"
VAL_IDS="${VAL_IDS:-${TRAIN_IDS}}"
TEST_IDS="${TEST_IDS:-['10']}"

TRAIN_IDS_CLEAN="$(echo "${TRAIN_IDS}" | tr -d "[]'\" ")"
TEST_IDS_CLEAN="$(echo "${TEST_IDS}" | tr -d "[]'\" ")"

RUN_TAG="${RUN_TAG:-0729_full_gpu_pcd}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints/${RUN_TAG}}"
RESULT_ROOT="${RESULT_ROOT:-./results/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-./logs/${RUN_TAG}}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

python -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Install a CUDA-enabled PyTorch build.")
print(f"[CUDA] device={torch.cuda.get_device_name(0)}")
print(f"[CUDA] torch={torch.__version__}, cuda={torch.version.cuda}")
'

mkdir -p "${CHECKPOINT_ROOT}" "${RESULT_ROOT}" "${LOG_ROOT}"

for MODEL in "${MODELS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
        EXP_NAME="${DATASET}_${PRED_LEN}_${LR}_ep_${EPOCHS}_${TRAIN_IDS_CLEAN}_to_${TEST_IDS_CLEAN}"
        CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${MODEL}/${EXP_NAME}"
        RESULT_DIR="${RESULT_ROOT}/${MODEL}/${EXP_NAME}"
        LOG_FILE="${LOG_ROOT}/${MODEL}_${EXP_NAME}.log"

        mkdir -p "${CHECKPOINT_DIR}" "${RESULT_DIR}"

        echo "================================================================"
        echo "[TRAIN] model=${MODEL}, pred_len=${PRED_LEN}, GPU=${GPU_ID}"
        echo "[TRAIN] vehicles=${TRAIN_IDS} -> ${TEST_IDS}"
        echo "[TRAIN] epochs=${EPOCHS}, batch=${BATCH_SIZE}, lr=${LR}"
        echo "[TRAIN] checkpoint=${CHECKPOINT_DIR}"
        echo "[TRAIN] log=${LOG_FILE}"

        python main.py \
            DATA.NAME "${DATASET}" \
            DATA.SEQ_LEN "${PRED_LEN}" \
            DATA.PRED_LEN "${PRED_LEN}" \
            DATA.LABEL_LEN "${LABEL_LEN}" \
            DATA.TRAIN_VEHICLE_IDS "${TRAIN_IDS}" \
            DATA.VAL_VEHICLE_IDS "${VAL_IDS}" \
            DATA.TEST_VEHICLE_IDS "${TEST_IDS}" \
            MODEL.NAME "${MODEL}" \
            MODEL.seq_len "${PRED_LEN}" \
            MODEL.pred_len "${PRED_LEN}" \
            MODEL.label_len "${LABEL_LEN}" \
            MODEL.patch_len "${PATCH_LEN}" \
            MODEL.stride "${STRIDE}" \
            TRAIN.ENABLE True \
            TRAIN.BATCH_SIZE "${BATCH_SIZE}" \
            VAL.BATCH_SIZE "${BATCH_SIZE}" \
            TEST.BATCH_SIZE "${BATCH_SIZE}" \
            DATA_LOADER.NUM_WORKERS "${NUM_WORKERS}" \
            SOLVER.MAX_EPOCH "${EPOCHS}" \
            SOLVER.BASE_LR "${LR}" \
            TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
            TRAIN.FINETUNE False \
            RESULT_DIR "${RESULT_DIR}" \
            TEST.ENABLE False \
            TTA.ENABLE False \
            2>&1 | tee "${LOG_FILE}"

        CHECKPOINT_FILE="${CHECKPOINT_DIR}/checkpoint_best.pth"
        if [[ ! -s "${CHECKPOINT_FILE}" ]]; then
            echo "[ERROR] checkpoint was not created: ${CHECKPOINT_FILE}" >&2
            exit 1
        fi

        CHECKPOINT_SIZE="$(du --apparent-size -h "${CHECKPOINT_FILE}" | cut -f1)"
        echo "[PASS] ${MODEL} P${PRED_LEN}: checkpoint=${CHECKPOINT_SIZE}"
    done
done

echo "================================================================"
echo "[PASS] All full PCD CUDA training jobs completed."
