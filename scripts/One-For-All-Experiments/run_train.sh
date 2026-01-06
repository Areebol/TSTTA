#!/bin/bash
MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
# DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
# PRED_LENS=(96 192 336 720)
# DATASETS=("weather")
DATASETS=("ETTh1")
PRED_LENS=(96 192 336 720)
for NAME in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for PRED_LEN in "${PRED_LENS[@]}"; do
            python main.py \
                DATA.NAME ${DATASET} \
                DATA.PRED_LEN ${PRED_LEN} \
                MODEL.NAME ${NAME} \
                MODEL.pred_len ${PRED_LEN} \
                TRAIN.ENABLE True \
                TRAIN.CHECKPOINT_DIR checkpoints/${NAME}/${DATASET}_${PRED_LEN}/ \
                TTA.ENABLE False \
                TEST.ENABLE False
        done
    done
done
