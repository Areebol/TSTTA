#!/bin/bash

############################################
# 实验参数
############################################
# MODELS=("PatchTSTPCD" "DLinearPCD" "FreTSPCD" "iTransformerPCD" "OLSPCD" "MICNPCD" )
# MODELS=("FreTSPCD")
MODELS=("LSTMPCI")
# MODELS=("OLSPCD")
DATASETS=("eVED")
PRED_LENS=(192) # 24
# PRED_LENS=(48 96 192)1
LABEL_LENS=(12)
# PATCH_LENS=(16) # 8
PATCH_LENS=(8)
# STRIDES=(8) # 4
STRIDES=(4)
# EPOCHS=(30)
EPOCHS=(30)
BATCH_SIZES=(64)
LRS=(1e-4)

# 核心ID参数
TRAIN_IDS="['455']"
TEST_IDS="['10']"
VAL_IDS=${TRAIN_IDS}
# Clean TRAIN_IDS for directory naming
TRAIN_IDS_CLEAN=$(echo "${TRAIN_IDS}" | tr -d "[]'\" ")

export TRAIN_IDS
export TEST_IDS
export VAL_IDS
export TRAIN_IDS_CLEAN

NPUS=(0 1 3 4)          # 可用的 NPU ID
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

    MODEL={1}
    DATASET={2}
    PRED_LEN={3}
    LABEL_LEN={4}
    patch_len={5}
    stride={6}
    EPOCH={7}
    BATCH_SIZE={8}
    LR={9}

    CHECKPOINT_DIR="./checkpoints/0327/${MODEL}/${DATASET}_${PRED_LEN}_${LR}_ep_${EPOCH}_455_2_10"
    mkdir -p ${CHECKPOINT_DIR}

    echo "Job slot {%}: NPU=${NPU_ID} | MODEL=${MODEL} | ${TRAIN_IDS} -> ${TEST_IDS} | PRED_LEN=${PRED_LEN} | LR=${LR}"
    echo "Checkpoint dir: ${CHECKPOINT_DIR}"

    # NPU 核心环境变量（昇腾标准）
    export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}

    python main.py \
        DATA.NAME ${DATASET} \
        DATA.SEQ_LEN ${PRED_LEN} \
        DATA.PRED_LEN ${PRED_LEN} \
        DATA.LABEL_LEN ${LABEL_LEN} \
        DATA.TRAIN_VEHICLE_IDS ${TRAIN_IDS} \
        DATA.VAL_VEHICLE_IDS ${VAL_IDS} \
        DATA.TEST_VEHICLE_IDS ${TEST_IDS} \
        MODEL.NAME ${MODEL} \
        MODEL.pred_len ${PRED_LEN} \
        MODEL.label_len ${LABEL_LEN} \
        MODEL.seq_len ${PRED_LEN} \
        MODEL.patch_len ${patch_len} \
        MODEL.stride ${stride} \
        TRAIN.ENABLE False \
        SOLVER.MAX_EPOCH ${EPOCH} \
        TRAIN.BATCH_SIZE ${BATCH_SIZE} \
        SOLVER.BASE_LR ${LR} \
        TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
        TRAIN.FINETUNE False \
        TEST.ENABLE True \
        TTA.ENABLE False
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${LABEL_LENS[@]}" ::: "${PATCH_LENS[@]}" ::: "${STRIDES[@]}" ::: "${EPOCHS[@]}" ::: "${BATCH_SIZES[@]}" ::: "${LRS[@]}"