#!/bin/bash
set -euo pipefail

MODEL="PatchTST"
PAIRS=("ETTh1:ETTh2" "ETTh2:ETTh1" "ETTm1:ETTm2" "ETTm2:ETTm1")
PRED_LENS=(96 192 336 720)
DISTILL_MODES=("mean" "query_weighted")
SEEDS=(0)
OFFLINE_LR="1e-2"
ONLINE_LR="1e-3"

for PAIR in "${PAIRS[@]}"; do
  SOURCE="${PAIR%%:*}"
  TARGET="${PAIR##*:}"
  for PRED_LEN in "${PRED_LENS[@]}"; do
    for DISTILL_MODE in "${DISTILL_MODES[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        CHECKPOINT_DIR="./checkpoints/${MODEL}/${SOURCE}_${PRED_LEN}"
        RESULT_DIR="./results/tpa_0727/transfer/${SOURCE}_to_${TARGET}_${PRED_LEN}"
        python main.py \
          SEED "${SEED}" \
          DATA.NAME "${SOURCE}" \
          DATA.PRED_LEN "${PRED_LEN}" \
          DATA.DOMAIN_SHIFT_TARGET "${TARGET}" \
          MODEL.NAME "${MODEL}" \
          MODEL.pred_len "${PRED_LEN}" \
          TRAIN.ENABLE False \
          TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
          TEST.ENABLE False \
          TTA.ENABLE True \
          TTA.DOMAIN_SHIFT True \
          TTA.METHOD TPA \
          TTA.SOLVER.BASE_LR "${OFFLINE_LR}" \
          TTA.DUAL.PRETRAIN_EPOCHS 1 \
          TTA.DUAL.PAAS True \
          TTA.DUAL.ADJUST_PRED True \
          TTA.DUAL.CALI_NAME TPAPrototypeAdapter \
          TTA.DUAL.LOSS_NAME CoBA_Loss \
          TTA.DUAL.CALI_INPUT_ENABLE False \
          TTA.DUAL.CALI_OUTPUT_ENABLE True \
          TTA.DUAL.COBA_ONLINE_ENABLED True \
          TTA.DUAL.COBA_ONLINE_LR "${ONLINE_LR}" \
          TTA.TPA.N_SOURCE 16 \
          TTA.TPA.N_ONLINE 16 \
          TTA.TPA.ANCHOR_CAPACITY 64 \
          TTA.TPA.DISTILL_MODE "${DISTILL_MODE}" \
          TTA.VISUALIZE False \
          RESULT_DIR "${RESULT_DIR}"
      done
    done
  done
done
