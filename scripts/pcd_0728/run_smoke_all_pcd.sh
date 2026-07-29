#!/usr/bin/env bash
set -euo pipefail

# Minimal end-to-end CUDA smoke test for all PCD forecasting models.
# It runs one P24 training epoch plus validation, sequentially on one GPU.
#
# Usage:
#   bash scripts/0728/run_smoke_all_pcd_gpu.sh
#   GPU_ID=1 BATCH_SIZE=32 bash scripts/0728/run_smoke_all_pcd_gpu.sh

GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"

DATASET="eVED"
PRED_LEN=24
LABEL_LEN=12
EPOCHS=1
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-2}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-2}"
PATCH_LEN=8
STRIDE=4

TRAIN_IDS="${TRAIN_IDS:-['455']}"
TEST_IDS="${TEST_IDS:-['10']}"
VAL_IDS="${VAL_IDS:-${TRAIN_IDS}}"

MODELS=(
    "FreTSPCD"
    "PatchTSTPCD"
    "DLinearPCD"
    "iTransformerPCD"
    "MICNPCD"
    "OLSPCD"
)

RUN_ROOT="${RUN_ROOT:-./checkpoints/0728_smoke_gpu_pcd}"
RESULT_ROOT="${RESULT_ROOT:-./results/0728_smoke_gpu_pcd}"
LOG_ROOT="${LOG_ROOT:-./logs/0728_smoke_gpu_pcd}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

python -c '
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Install a CUDA-enabled PyTorch build.")
print(f"[CUDA] device={torch.cuda.get_device_name(0)}")
print(f"[CUDA] torch={torch.__version__}, cuda={torch.version.cuda}")
'

mkdir -p "${RUN_ROOT}" "${RESULT_ROOT}" "${LOG_ROOT}"

for MODEL in "${MODELS[@]}"; do
    CHECKPOINT_DIR="${RUN_ROOT}/${MODEL}/${DATASET}_P${PRED_LEN}_target2"
    RESULT_DIR="${RESULT_ROOT}/${MODEL}/${DATASET}_P${PRED_LEN}_target2"
    LOG_FILE="${LOG_ROOT}/${MODEL}_${DATASET}_P${PRED_LEN}.log"

    mkdir -p "${CHECKPOINT_DIR}" "${RESULT_DIR}"

    echo "================================================================"
    echo "[SMOKE] model=${MODEL}, GPU=${GPU_ID}, batch=${BATCH_SIZE}"
    echo "[SMOKE] checkpoint=${CHECKPOINT_DIR}"
    echo "[SMOKE] log=${LOG_FILE}"

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
        TRAIN.EVAL_PERIOD 1 \
        TRAIN.PRINT_FREQ 1 \
        TRAIN.MAX_BATCHES "${MAX_TRAIN_BATCHES}" \
        VAL.BATCH_SIZE "${BATCH_SIZE}" \
        VAL.MAX_BATCHES "${MAX_VAL_BATCHES}" \
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
    if [[ ! -f "${CHECKPOINT_FILE}" ]]; then
        echo "[ERROR] checkpoint was not created: ${CHECKPOINT_FILE}" >&2
        exit 1
    fi

    CHECKPOINT_SIZE="$(du --apparent-size -h "${CHECKPOINT_FILE}" | cut -f1)"
    echo "[PASS] ${MODEL}: checkpoint=${CHECKPOINT_SIZE}"
done

echo "================================================================"
echo "[PASS] All PCD CUDA smoke tests completed."
