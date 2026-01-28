#!/bin/bash

############################################
# 实验参数
############################################
MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
MODELS=("FreTS")
DATASETS=("eVED")
TARGETS=("eVED")
# PRED_LENS=(96 192 336 720)
# PRED_LENS=(192)
PRED_LENS=(24)
LRS=(5e-2 1e-2 5e-3 1e-3 1e-4)
TRAIN_IDS="['455']"
TEST_IDS="['10']"
VAL_IDS=${TRAIN_IDS}
# Clean TRAIN_IDS for directory naming: remove [, ], ', " and spaces
TRAIN_IDS_CLEAN=$(echo "${TRAIN_IDS}" | tr -d "[]'\" ")

export TRAIN_IDS
export TEST_IDS
export VAL_IDS
export TRAIN_IDS_CLEAN

NPUS=(7)          # 可用的 NPU ID
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
    RESULT_DIR="./results/NONE/"
    BASE_LR={5}

    echo "Job slot {%}: NPU=${NPU_ID}  MODEL={1}  DATASET={2}  PRED={3}  TARGET={4} BASE_LR={5}"

    # export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}
    export CUDA_VISIBLE_DEVICES=${NPU_ID}
    NORM_MODULE_ENABLE=False
    NORM_MODULE_NAME="RevIN"

    python main.py \
        SEED ${SEED} \
        DATA.NAME {2} \
        DATA.SEQ_LEN {3} \
        DATA.PRED_LEN {3} \
        DATA.LABEL_LEN 12 \
        DATA.DOMAIN_SHIFT_TARGET {4} \
        DATA.MIN_TEST_LEN 300 \
        DATA.TRAIN_VEHICLE_IDS "${TRAIN_IDS}" \
        DATA.VAL_VEHICLE_IDS "${VAL_IDS}" \
        DATA.TEST_VEHICLE_IDS "${TEST_IDS}" \
        MODEL.NAME {1} \
        MODEL.pred_len {3} \
        MODEL.seq_len {3} \
        MODEL.label_len 12 \
        MODEL.patch_len 8 \
        MODEL.stride 4 \
        NORM_MODULE.ENABLE ${NORM_MODULE_ENABLE} \
        NORM_MODULE.NAME ${NORM_MODULE_NAME} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR checkpoints/{1}/{2}_{3}_ids${TRAIN_IDS_CLEAN}/ \
        TEST.ENABLE False \
        TTA.ENABLE True \
        TTA.DOMAIN_SHIFT False \
        TTA.SOLVER.BASE_LR ${BASE_LR} \
        RESULT_DIR ${RESULT_DIR} \
        TTA.METHOD PETSA
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${TARGETS[@]}" ::: "${LRS[@]}"