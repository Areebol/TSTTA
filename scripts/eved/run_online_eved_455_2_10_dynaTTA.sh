#!/bin/bash

############################################
# 实验参数
############################################

MODELS=("PatchTSTPCD" "DLinearPCD" "FreTSPCD" "iTransformerPCD" "OLSPCD" "MICNPCD" ) 
DATASETS=("eVED")
TTA_METHODS=("DynaTTA")
# TTA_METHODS=("TAFAS" "PETSA")
PRED_LENS=(24)
BASE_LRS=(1e-4)
TTA_LRS=(1e-2 1e-3 1e-4 1e-5)


# TRAIN_IDS="['10']"
# TEST_IDS="['455']"

TRAIN_IDS="['455']"
TEST_IDS="['10']"
VAL_IDS=${TRAIN_IDS}
# Clean TRAIN_IDS for directory naming: remove [, ], ', " and spaces
TRAIN_IDS_CLEAN=$(echo "${TRAIN_IDS}" | tr -d "[]'\" ")

export TRAIN_IDS
export TEST_IDS
export VAL_IDS
export TRAIN_IDS_CLEAN

NPUS=(4 5 6 7)          # 可用的 NPU ID
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
    TTA_METHOD={6}

    CHECKPOINT_DIR="./checkpoints/0325/${MODEL}/${DATASET}_${PRED_LEN}_${BASE_LR}_ep_30_455_2_10"
    RESULT_DIR="./0327/results/${TTA_METHOD}/${MODEL}/"

    echo "Job slot {%}: NPU=${NPU_ID}  MODEL={1} TTA_METHOD={6}  DATASET={2}  PRED_LEN={3} BASE_LR={4} TTA_LR={5}"
    export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}
    # export CUDA_VISIBLE_DEVICES=${NPU_ID}


    NORM_MODULE_ENABLE=False
    NORM_MODULE_NAME="RevIN"

    python main.py \
        SEED ${SEED} \
        DATA.NAME ${DATASET} \
        DATA.SEQ_LEN ${PRED_LEN} \
        DATA.PRED_LEN ${PRED_LEN} \
        DATA.LABEL_LEN 12 \
        DATA.MIN_TEST_LEN 300 \
        DATA.N_VAR 2 \
        MODEL.enc_in 2 \
        MODEL.dec_in 2 \
        MODEL.c_out 2 \
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
        TEST.ENABLE True \
        TTA.ENABLE True \
        TTA.SOLVER.BASE_LR ${TTA_LR} \
        TTA.DOMAIN_SHIFT False \
        TTA.METHOD ${TTA_METHOD} \
        RESULT_DIR ${RESULT_DIR}
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${BASE_LRS[@]}" ::: "${TTA_LRS[@]}" ::: "${TTA_METHODS[@]}"