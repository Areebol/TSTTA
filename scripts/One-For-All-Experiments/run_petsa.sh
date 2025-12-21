#!/bin/bash
GPUS=(0 5 6 7)
NUM_GPUS=${#GPUS[@]}

# MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
# DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
MODELS=("DLinear" "FreTS")
DATASETS=("ETTm1" "ETTm2" "exchange_rate" "weather")
PRED_LENS=(96)
PRED_LENS=(96 192 336 720)

parallel -j 32 --delay 0 '
    IDX=$(( ({#} - 1) % '"${NUM_GPUS}"' ))
    GPU='"${GPUS}"'[$IDX]
    SEED=0
    RESULT_DIR="./results/PETSA/"
    BASE_LR=0.001
    WEIGHT_DECAY=0.0001
    LOW_RANK=16
    LOSS_ALPHA=0.1
    GATING_INIT=0.01

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
        TTA.PETSA.GATING_INIT ${GATING_INIT} \
        TTA.PETSA.RANK ${LOW_RANK} \
        TTA.PETSA.LOSS_ALPHA ${LOSS_ALPHA} \
        RESULT_DIR ${RESULT_DIR} \
        TTA.METHOD PETSA
        
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}"