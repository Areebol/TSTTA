#!/usr/bin/env bash
set -euo pipefail

# Evaluate only the fixed OLSPCD checkpoints trained on 2026-08-02.
#
# Usage:
#   bash scripts/0802/eval_olspcd_fixed_gpu.sh
#   bash scripts/0802/eval_olspcd_fixed_gpu.sh /absolute/checkpoint/root
#
# Optional overrides:
#   GPU_ID=1 BATCH_SIZE=128 bash scripts/0802/eval_olspcd_fixed_gpu.sh

PROJECT_ROOT="/linyuanping/dzs/codes/TSTTA"

# Use the first positional argument when supplied. Otherwise use the exact
# fixed checkpoint directory. This deliberately avoids inheriting a stale or
# empty CHECKPOINT_ROOT environment variable from an earlier experiment.
CHECKPOINT_ROOT="${1:-/linyuanping/dzs/codes/TSTTA/checkpoints/0802_olspcd_fixed/OLSPCD}"

RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/results/0802_olspcd_fixed/OLSPCD}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/0802_olspcd_fixed/OLSPCD}"
DATA_BASE_DIR="${PROJECT_ROOT}/data"

PRED_LENS_TEXT="${PRED_LENS:-24 48 96 192}"
DIRECTIONS_TEXT="${DIRECTIONS:-455:10 10:455}"
read -r -a PRED_LENS_ARRAY <<< "${PRED_LENS_TEXT}"
read -r -a DIRECTIONS_ARRAY <<< "${DIRECTIONS_TEXT}"

DATASET="eVED"
MODEL="OLSPCD"
LABEL_LEN=12
LR_TAG="1e-4"
EPOCH_TAG=30
SEED=0

BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GPU_ID="${GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"

printf '[CONFIG] checkpoint_root=%q\n' "${CHECKPOINT_ROOT}"

if [[ -z "${CHECKPOINT_ROOT}" ]]; then
    echo "[ERROR] checkpoint root resolved to an empty string" >&2
    exit 1
fi

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
    printf '[ERROR] checkpoint root does not exist: %q\n' "${CHECKPOINT_ROOT}" >&2
    exit 1
fi

mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"

SUMMARY_FILE="${RESULT_ROOT}/summary.tsv"
printf "model\tpred_len\ttrain_id\ttest_id\tcheckpoint\tmetrics\n" > "${SUMMARY_FILE}"

echo "================================================================"
echo "[CONFIG] project=${PROJECT_ROOT}"
printf '[CONFIG] checkpoint_root=%q\n' "${CHECKPOINT_ROOT}"
echo "[CONFIG] result_root=${RESULT_ROOT}"
echo "[CONFIG] gpu=${GPU_ID}"

python -c '
import torch
print(f"[DEVICE] torch={torch.__version__}")
print(f"[DEVICE] cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[DEVICE] cuda_device={torch.cuda.get_device_name(0)}")
'

for PRED_LEN in "${PRED_LENS_ARRAY[@]}"; do
    for DIRECTION in "${DIRECTIONS_ARRAY[@]}"; do
        IFS=: read -r TRAIN_ID TEST_ID <<< "${DIRECTION}"

        TRAIN_IDS="['${TRAIN_ID}']"
        VAL_IDS="['${TRAIN_ID}']"
        TEST_IDS="['${TEST_ID}']"

        EXP_NAME="${DATASET}_${PRED_LEN}_${LR_TAG}_ep_${EPOCH_TAG}_${TRAIN_ID}_to_${TEST_ID}"
        CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${EXP_NAME}"
        CHECKPOINT_FILE="${CHECKPOINT_DIR}/checkpoint_best.pth"

        RESULT_DIR="${RESULT_ROOT}/${DATASET}_P${PRED_LEN}_${TRAIN_ID}_to_${TEST_ID}"
        LOG_FILE="${LOG_ROOT}/${MODEL}_${DATASET}_P${PRED_LEN}_${TRAIN_ID}_to_${TEST_ID}.log"

        if [[ ! -s "${CHECKPOINT_FILE}" ]]; then
            echo "[ERROR] checkpoint not found: ${CHECKPOINT_FILE}" >&2
            exit 1
        fi

        mkdir -p "${RESULT_DIR}"

        echo "================================================================"
        echo "[EVAL] model=${MODEL}, pred_len=${PRED_LEN}, ${TRAIN_ID}->${TEST_ID}"
        echo "[EVAL] checkpoint=${CHECKPOINT_FILE}"
        echo "[EVAL] result=${RESULT_DIR}"
        echo "[EVAL] log=${LOG_FILE}"

        # Fail early when the CUDA project still contains the previous
        # OLSPCD implementation (seq_len inputs) instead of the legacy-
        # compatible compact implementation (seq_len + 1 inputs).
        "${PYTHON_BIN:-python}" - "${CHECKPOINT_FILE}" "${PRED_LEN}" <<'PY'
import sys
import torch

from config import get_cfg_defaults
from datasets.build import update_cfg_from_dataset
from models.OLSPCD import Model

checkpoint_path = sys.argv[1]
pred_len = int(sys.argv[2])

cfg = get_cfg_defaults()
update_cfg_from_dataset(cfg, "eVED")
cfg.MODEL.seq_len = pred_len
cfg.MODEL.pred_len = pred_len
cfg.MODEL.instance_norm = True

model = Model(cfg.MODEL)
model_shape = tuple(model.linear.linear_fusion.weight.shape)

try:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

state = checkpoint.get("model_state", checkpoint)
checkpoint_shape = tuple(state["linear.linear_fusion.weight"].shape)

print(f"[SHAPE] model={model_shape}, checkpoint={checkpoint_shape}")
if model_shape != checkpoint_shape:
    raise SystemExit(
        "[ERROR] OLSPCD code/checkpoint shape mismatch. "
        "The compact legacy-compatible model must use "
        "20 * (pred_len + 1) input features. Sync models/OLSPCD.py."
    )
PY

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
            MODEL.instance_norm True \
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

        if ! grep -Fq "Debug: model_path = ${CHECKPOINT_FILE}" "${LOG_FILE}"; then
            echo "[ERROR] evaluation did not report the expected checkpoint path:" >&2
            echo "        ${CHECKPOINT_FILE}" >&2
            exit 1
        fi

        METRICS="$(tr '\n' ' ' < "${RESULT_DIR}/test.txt")"
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${MODEL}" \
            "${PRED_LEN}" \
            "${TRAIN_ID}" \
            "${TEST_ID}" \
            "${CHECKPOINT_FILE}" \
            "${METRICS}" \
            >> "${SUMMARY_FILE}"

        echo "[PASS] ${MODEL} P${PRED_LEN} ${TRAIN_ID}->${TEST_ID}: ${METRICS}"
    done
done

echo "================================================================"
echo "[PASS] All fixed OLSPCD evaluations completed."
echo "[PASS] Summary: ${SUMMARY_FILE}"
