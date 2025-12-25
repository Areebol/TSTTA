#!/bin/bash

GPUS=(0 1 2 3 4 5 6 7)
NGPU=${#GPUS[@]}
GPU_STR="${GPUS[*]}"
export GPU_STR
JOBS_PER_GPU=16
TOTAL_JOBS=$((NGPU * JOBS_PER_GPU))

MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
PRED_LENS=(96 192 336 720)

# MODELS=("DLinear")
# DATASETS=("ETTh1")
# PRED_LENS=(96)

parallel --lb -j ${TOTAL_JOBS} '
    gpu_array=($GPU_STR)
    slot_idx=$(( ({%} - 1) % '"${NGPU}"' ))
    GPU_ID=${gpu_array[$slot_idx]}

    SEED=0
    BASE_LR=0.001
    WEIGHT_DECAY=0.0001
    GATING_INIT=0.01
    RESULT_DIR="./results/TAFAS/"

    echo "MODEL={1} DATASET={2} PRED={3} -> Running on GPU $GPU_ID (Slot $slot_idx)"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
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
        TTA.DUAL.GATING_INIT ${GATING_INIT} \
        TTA.DUAL.CALI_NAME tafas_GCM \
        TTA.DUAL.LOSS_NAME MSE \
        RESULT_DIR ${RESULT_DIR} \
        TTA.METHOD Dual-tta
        
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}"