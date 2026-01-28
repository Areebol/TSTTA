#!/bin/bash
# MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" )
MODELS=("PatchTST")
# DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
# DATASETS=("ETTm2")
DATASETS=("eVED")
# PRED_LENS=(96 192 336 720)
# PRED_LENS=(24)
PRED_LENS=(96)
LR=(1e-3)
# TRAIN_IDS="['455']"
# TEST_IDS="['10']"
TRAIN_IDS="['10']"
TEST_IDS="['455']"
VAL_IDS=${TRAIN_IDS}
# Clean TRAIN_IDS for directory naming: remove [, ], ', " and spaces
TRAIN_IDS_CLEAN=$(echo "${TRAIN_IDS}" | tr -d "[]'\" ")

export TRAIN_IDS
export TEST_IDS
export VAL_IDS
export TRAIN_IDS_CLEAN

# NPUS=(7)          # 可用的 NPU ID
NPUS=(0 1 2 3 4 5 6 7)  # 可用的 NPU ID
NNPU=${#NPUS[@]}        # NPU 数量

PER_NPU=1               # 每个 NPU 并行任务数
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

parallel --lb -j ${TOTAL_JOBS} '
    npu_array=($NPU_STR)

    # slot_idx：parallel 的任务槽位 → 映射到某个 NPU
    slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
    NPU_ID=${npu_array[$slot_idx]}

    echo "Job slot {%}: Running on NPU ${NPU_ID} (MODEL={1}, DATASET={2}, PRED={3})"
    # export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}
    export CUDA_VISIBLE_DEVICES=${NPU_ID}

    NORM_MODULE_ENABLE=False
    NORM_MODULE_NAME="RevIN"
    LR=1e-3
    patch_len=16
    stride=8

    python main.py \
        DATA.NAME {2} \
        DATA.SEQ_LEN {3} \
        DATA.PRED_LEN {3} \
        DATA.LABEL_LEN 12 \
        DATA.TRAIN_VEHICLE_IDS "${TRAIN_IDS}" \
        DATA.VAL_VEHICLE_IDS "${VAL_IDS}" \
        DATA.TEST_VEHICLE_IDS "${TEST_IDS}" \
        MODEL.NAME {1} \
        MODEL.pred_len {3} \
        MODEL.label_len 12 \
        MODEL.seq_len {3} \
        MODEL.c_out 2  \
        MODEL.patch_len ${patch_len} \
        MODEL.stride ${stride} \
        TRAIN.ENABLE True \
        SOLVER.BASE_LR ${LR} \
        TRAIN.CHECKPOINT_DIR checkpoints/{1}/{2}_{3}_ids${TRAIN_IDS_CLEAN}/ \
        NORM_MODULE.ENABLE ${NORM_MODULE_ENABLE} \
        NORM_MODULE.NAME ${NORM_MODULE_NAME} \
        TTA.ENABLE False \
        TEST.ENABLE False
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}"


# for NAME in "${MODELS[@]}"; do
#     for DATASET in "${DATASETS[@]}"; do
#         for PRED_LEN in "${PRED_LENS[@]}"; do
#             python main.py \
#                 DATA.NAME ${DATASET} \
#                 DATA.PRED_LEN ${PRED_LEN} \
#                 MODEL.NAME ${NAME} \
#                 MODEL.pred_len ${PRED_LEN} \
#                 TRAIN.ENABLE True \
#                 TRAIN.CHECKPOINT_DIR checkpoints/${NAME}/${DATASET}_${PRED_LEN}/ \
#                 TTA.ENABLE False \
#                 TEST.ENABLE False
#         done
#     done
# done
