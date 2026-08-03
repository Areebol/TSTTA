#!/bin/bash
set -euo pipefail

MODEL="PatchTSTPCD"
DATASET="eVED"
TRAIN_IDS="['10']"
TEST_IDS="['455']"
VAL_IDS="${TRAIN_IDS}"
PRED_LENS=(24 48 96 192)
DISTILL_MODES=("mean" "query_weighted")
SEEDS=(0)
OFFLINE_LR="1e-2"
ONLINE_LR="5e-3"

for PRED_LEN in "${PRED_LENS[@]}"; do
  for DISTILL_MODE in "${DISTILL_MODES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      CHECKPOINT_DIR="./checkpoints/0327/${MODEL}/${DATASET}_${PRED_LEN}_1e-4_ep_30_10_2_455/"
      RESULT_DIR="./results/tpa_0727/eved/10_to_455/${PRED_LEN}"
      python main.py \
        SEED "${SEED}" \
        DATA.NAME "${DATASET}" \
        DATA.SEQ_LEN "${PRED_LEN}" \
        DATA.PRED_LEN "${PRED_LEN}" \
        DATA.LABEL_LEN 12 \
        DATA.DOMAIN_SHIFT_TARGET "${DATASET}" \
        DATA.TRAIN_VEHICLE_IDS "${TRAIN_IDS}" \
        DATA.VAL_VEHICLE_IDS "${VAL_IDS}" \
        DATA.TEST_VEHICLE_IDS "${TEST_IDS}" \
        MODEL.NAME "${MODEL}" \
        MODEL.seq_len "${PRED_LEN}" \
        MODEL.pred_len "${PRED_LEN}" \
        MODEL.label_len 12 \
        MODEL.patch_len 8 \
        MODEL.stride 4 \
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
