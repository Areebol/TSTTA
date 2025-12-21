#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
MODELS=("DLinear" "FreTS")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
PRED_LENS=(96 192 336 720)

parallel -j 4 --delay 0 '
    GPU=7
    SEED=0
    BASE_LR=0.001
    WEIGHT_DECAY=0.0001
    GATING_INIT=0.01
    RESULT_DIR="./results/TAFAS/"

    echo "Job {#}: MODEL={1} DATASET={2} PRED={3} -> Running on GPU $GPU"
    
    CUDA_VISIBLE_DEVICES=$GPU python main.py \
        SEED ${SEED} \
        DATA.NAME {2} \
        DATA.PRED_LEN {3} \
        MODEL.NAME {1} \
        MODEL.pred_len {3} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR checkpoints/{1}/{2}_{3}/ \
        TEST.ENABLE False \
        TTA.ENABLE True \
        TTA.SOLVER.BASE_LR ${BASE_LR} \
        TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
        TTA.TAFAS.GATING_INIT ${GATING_INIT} \
        RESULT_DIR ${RESULT_DIR} \
        TTA.METHOD TAFAS
        
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}"