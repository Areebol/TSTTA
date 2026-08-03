#!/bin/bash
set -euo pipefail

# Edit these arrays to launch a larger grid. Both distillation variants run by
# default so their results remain directly comparable.
MODELS=("PatchTST")
DATASETS=("ETTm1")
PRED_LENS=(96)
DISTILL_MODES=("mean" "query_weighted")
SEEDS=(0)
OFFLINE_LR="1e-2"
ONLINE_LR="1e-3"

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
      for DISTILL_MODE in "${DISTILL_MODES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
          CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}"
          RESULT_DIR="./results/tpa_0727/regular/${MODEL}/${DATASET}_${PRED_LEN}"
          python main.py \
            SEED "${SEED}" \
            DATA.NAME "${DATASET}" \
            DATA.PRED_LEN "${PRED_LEN}" \
            MODEL.NAME "${MODEL}" \
            MODEL.pred_len "${PRED_LEN}" \
            TRAIN.ENABLE False \
            TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
            TEST.ENABLE False \
            TTA.ENABLE True \
            TTA.DOMAIN_SHIFT False \
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
done
