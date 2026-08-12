#!/usr/bin/env bash
set -euo pipefail

# Deterministic closed-form OLSPCD fitting on eVED.
# OLSPCD does not use epochs or gradient updates; the ep_30/lr tags below are
# retained only so existing evaluation scripts can discover the checkpoints.
#
# Examples:
#   bash scripts/0802/train_olspcd_repro_gpu.sh
#   RUN_TAG=0803_olspcd_repro_run2 PRED_LENS="24" \
#     DIRECTIONS="455:10" bash scripts/0802/train_olspcd_repro_gpu.sh

PROJECT_ROOT="${PROJECT_ROOT:-/linyuanping/dzs/codes/TSTTA}"
RUN_TAG="${RUN_TAG:-0803_olspcd_repro_run1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints/${RUN_TAG}/OLSPCD}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/${RUN_TAG}/OLSPCD}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${RUN_TAG}/OLSPCD}"

PRED_LENS_TEXT="${PRED_LENS:-24 48 96 192}"
DIRECTIONS_TEXT="${DIRECTIONS:-455:10 10:455}"
read -r -a PRED_LENS_ARRAY <<< "${PRED_LENS_TEXT}"
read -r -a DIRECTIONS_ARRAY <<< "${DIRECTIONS_TEXT}"

GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ALPHA="${ALPHA:-1e-6}"
OLSPCD_SOLVER="${OLSPCD_SOLVER:-ridge}"
OLSPCD_SVD_RCOND="${OLSPCD_SVD_RCOND:-0.0}"
# YACS preserves scalar types strictly. MODEL.alpha is declared as float, so
# normalize integer-looking environment values such as ALPHA=1 to ALPHA=1.0.
if [[ "${ALPHA}" =~ ^[+-]?[0-9]+$ ]]; then
    ALPHA="${ALPHA}.0"
fi
if [[ "${OLSPCD_SVD_RCOND}" =~ ^[+-]?[0-9]+$ ]]; then
    OLSPCD_SVD_RCOND="${OLSPCD_SVD_RCOND}.0"
fi
export OLSPCD_SOLVER
export OLSPCD_SVD_RCOND

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
mkdir -p "${CHECKPOINT_ROOT}" "${RESULT_ROOT}" "${LOG_ROOT}"

python -c '
import torch
print(f"[ENV] torch={torch.__version__}")
print(f"[ENV] cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[ENV] cuda_device={torch.cuda.get_device_name(0)}")
print(f"[ENV] torch_threads={torch.get_num_threads()}")
'

for PRED_LEN in "${PRED_LENS_ARRAY[@]}"; do
    for DIRECTION in "${DIRECTIONS_ARRAY[@]}"; do
        IFS=: read -r TRAIN_ID TEST_ID <<< "${DIRECTION}"

        TRAIN_IDS="['${TRAIN_ID}']"
        TEST_IDS="['${TEST_ID}']"
        EXP_NAME="eVED_${PRED_LEN}_1e-4_ep_30_${TRAIN_ID}_to_${TEST_ID}"
        CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${EXP_NAME}"
        RESULT_DIR="${RESULT_ROOT}/${EXP_NAME}"
        LOG_FILE="${LOG_ROOT}/${EXP_NAME}.log"

        mkdir -p "${CHECKPOINT_DIR}" "${RESULT_DIR}"

        echo "================================================================"
        echo "[SOLVE] pred_len=${PRED_LEN}, ${TRAIN_ID}->${TEST_ID}"
        echo "[SOLVE] solver=${OLSPCD_SOLVER}, alpha=${ALPHA}, svd_rcond=${OLSPCD_SVD_RCOND}"
        echo "[SOLVE] batch_size=${BATCH_SIZE}"
        echo "[SOLVE] checkpoint=${CHECKPOINT_DIR}/checkpoint_best.pth"

        python main.py \
            SEED 0 \
            DATA.NAME eVED \
            DATA.SEQ_LEN "${PRED_LEN}" \
            DATA.PRED_LEN "${PRED_LEN}" \
            DATA.LABEL_LEN 12 \
            DATA.TRAIN_VEHICLE_IDS "${TRAIN_IDS}" \
            DATA.VAL_VEHICLE_IDS "${TRAIN_IDS}" \
            DATA.TEST_VEHICLE_IDS "${TEST_IDS}" \
            MODEL.NAME OLSPCD \
            MODEL.seq_len "${PRED_LEN}" \
            MODEL.pred_len "${PRED_LEN}" \
            MODEL.label_len 12 \
            MODEL.instance_norm True \
            MODEL.alpha "${ALPHA}" \
            TRAIN.ENABLE True \
            TRAIN.SHUFFLE False \
            TRAIN.DROP_LAST False \
            TRAIN.BATCH_SIZE "${BATCH_SIZE}" \
            DATA_LOADER.NUM_WORKERS 0 \
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

        echo "[PASS] ${CHECKPOINT_FILE}"
    done
done

echo "================================================================"
echo "[PASS] OLSPCD closed-form fitting completed."
