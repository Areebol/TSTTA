#!/bin/bash

############################################
# 实验参数
############################################
# MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
MODELS=("PatchTSTPCD")
DATASETS=("eVED")
# PRED_LENS=(96 192 336 720)
# PRED_LENS=(192)
PRED_LENS=(24)
BASE_LRS=(1e-4)
TTA_LRS=(1e-2 1e-3 1e-4 1e-5)
# TTA_LRS=(1e-3)

# TRAIN_IDS="['455']"
# TEST_IDS="['10']"
# VAL_IDS=${TRAIN_IDS}
# # Clean TRAIN_IDS for directory naming: remove [, ], ', " and spaces
# TRAIN_IDS_CLEAN=$(echo "${TRAIN_IDS}" | tr -d "[]'\" ")

# export TRAIN_IDS
# export TEST_IDS
# export VAL_IDS
# export TRAIN_IDS_CLEAN

NPUS=(4 5 6 7)          # 可用的 NPU ID
# NPUS=(0 1 2 3 4 5 6 7)  # 可用的 NPU ID
NNPU=${#NPUS[@]}        # NPU 数量

PER_NPU=1               # 每个 NPU 并行任务数
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

parallel --lb -j ${TOTAL_JOBS} '
    npu_array=($NPU_STR)

    # parallel 的任务槽位 → 映射为某个 NPU ID
    slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
    NPU_ID=${npu_array[$slot_idx]}

    SEED=0
    MODEL={1} 
    DATASET={2}
    PRED_LEN={3}
    BASE_LR={4}
    TTA_LR={5}
    CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}_${BASE_LR}"
    
    RESULT_DIR="./results/NONE/"

    echo "Job slot {%}: NPU=${NPU_ID}  MODEL={1}  DATASET={2}  PRED_LEN={3} BASE_LR={4} TTA_LR={5}"

    # export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}
    export CUDA_VISIBLE_DEVICES=${NPU_ID}
    NORM_MODULE_ENABLE=False
    NORM_MODULE_NAME="RevIN"

    python main.py \
        SEED ${SEED} \
        DATA.NAME ${DATASET} \
        DATA.SEQ_LEN ${PRED_LEN} \
        DATA.PRED_LEN ${PRED_LEN} \
        DATA.LABEL_LEN 12 \
        DATA.MIN_TEST_LEN 300 \
        MODEL.NAME ${MODEL} \
        MODEL.pred_len ${PRED_LEN} \
        MODEL.seq_len ${PRED_LEN} \
        MODEL.label_len 12 \
        MODEL.patch_len 8 \
        MODEL.stride 4 \
        NORM_MODULE.ENABLE ${NORM_MODULE_ENABLE} \
        NORM_MODULE.NAME ${NORM_MODULE_NAME} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
        TEST.ENABLE False \
        TTA.ENABLE True \
        TTA.SOLVER.BASE_LR ${TTA_LR} \
        TTA.DOMAIN_SHIFT False \
        RESULT_DIR ${RESULT_DIR}
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${BASE_LRS[@]}" ::: "${TTA_LRS[@]}"