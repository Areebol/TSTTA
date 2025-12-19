#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
# DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
# PRED_LENS=(96 192 336 720)
# MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
MODELS=("LSTM")
DATASETS=("ETTh1")
PRED_LENS=(96)

parallel -j 4 --delay 1 '
    GPU=$(( ({#} - 1) % 8 ))
    SEED=0
    RESULT_DIR="./results/NONE/"

    echo "Job {#}: MODEL={1} DATASET={2} PRED={3} -> Running on GPU $GPU"
    
    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        SEED ${SEED} \
        DATA.NAME {2} \
        DATA.PRED_LEN {3} \
        MODEL.NAME {1} \
        MODEL.pred_len {3} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR checkpoints/{1}/{2}_{3}/ \
        TEST.ENABLE True \
        TTA.ENABLE False \
        RESULT_DIR ${RESULT_DIR} \
        
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}"