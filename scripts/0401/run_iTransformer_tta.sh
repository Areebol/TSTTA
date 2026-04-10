#!/bin/bash

############################################
# 实验参数
############################################

# MODELS=("PatchTSTPCD" "DLinearPCD" "FreTSPCD" "iTransformerPCD" "OLSPCD" "MICNPCD" ) 
# MODELS=("PatchTSTPCD" "DLinearPCD" "FreTSPCD" )
MODELS=( "iTransformerPCD")
# DATASETS=("eVED")
TTA_METHODS=("TAFAS" "PETSA" "DynaTTA")
# TTA_METHODS=("TAFAS" )
# TTA_METHODS=("DynaTTA")
# TTA_METHODS=("TAFAS" "PETSA")
# PRED_LENS=(24)
# PRED_LENS=(48 96 192)
PRED_LENS=(24 48 96 192)
BASE_LRS=(1e-4)
TTA_LRS=(1e-2 1e-3 1e-4 1e-5)


# TRAIN_IDS="['10']"
# TEST_IDS="['455']"

PAIRS=("eVED:eVED")
TRAIN_IDS="['455']"
TEST_IDS="['10']"

# TRAIN_IDS="['10']"
# TEST_IDS="['455']"

VAL_IDS=${TRAIN_IDS}
# Clean TRAIN_IDS for directory naming: remove [, ], ', " and spaces
TRAIN_IDS_CLEAN=$(echo "${TRAIN_IDS}" | tr -d "[]'\" ")

export TRAIN_IDS
export TEST_IDS
export VAL_IDS
export TRAIN_IDS_CLEAN

NPUS=(0 1 )          # 可用的 NPU ID
NNPU=${#NPUS[@]}        # NPU 数量

PER_NPU=4              # 每个 NPU 并行任务数
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
    PAIR={2}
    PRED_LEN={3}
    BASE_LR={4}
    TTA_LR={5}
    TTA_METHOD={6}

    DATASET=$(echo $PAIR | cut -d: -f1)
    TARGET=$(echo $PAIR | cut -d: -f2)

    CHECKPOINT_DIR="./checkpoints/0401/${MODEL}/${DATASET}_${PRED_LEN}_${BASE_LR}_ep_60_455_2_10"
    RESULT_DIR="./results/0401/eVED/455_2_10/${TTA_METHOD}/${MODEL}/"

    echo "Job slot {%}: NPU=${NPU_ID} | MODEL={1} | TTA={6} |  ${DATASET} -> ${TARGET} | PRED_LEN={3} BASE_LR={4} TTA_LR={5}"
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
        DATA.DOMAIN_SHIFT_TARGET ${TARGET} \
        DATA.MIN_TEST_LEN 300 \
        DATA.TRAIN_VEHICLE_IDS "${TRAIN_IDS}" \
        DATA.VAL_VEHICLE_IDS "${VAL_IDS}" \
        DATA.TEST_VEHICLE_IDS "${TEST_IDS}" \
        MODEL.NAME ${MODEL} \
        MODEL.pred_len ${PRED_LEN} \
        MODEL.seq_len ${PRED_LEN} \
        MODEL.label_len 12 \
        NORM_MODULE.ENABLE ${NORM_MODULE_ENABLE} \
        NORM_MODULE.NAME ${NORM_MODULE_NAME} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
        TEST.ENABLE True \
        TTA.ENABLE True \
        TTA.SOLVER.BASE_LR ${TTA_LR} \
        TTA.DOMAIN_SHIFT True \
        TTA.METHOD ${TTA_METHOD} \
        RESULT_DIR ${RESULT_DIR}
' ::: "${MODELS[@]}" ::: "${PAIRS[@]}" ::: "${PRED_LENS[@]}" ::: "${BASE_LRS[@]}" ::: "${TTA_LRS[@]}" ::: "${TTA_METHODS[@]}" 