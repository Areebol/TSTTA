#!/bin/bash
# MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
MODELS=("PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
PRED_LENS=(96 192 336 720)
MODELS=("MICN")
DATASETS=("ETTm2")
PRED_LENS=(720)

NPUS=(1 2 3)
NNPU=${#NPUS[@]}   # NPU 数量

PER_NPU=8

TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

parallel --lb -j ${TOTAL_JOBS} '
    npu_array=($NPU_STR)

    # slot_idx：parallel 的任务槽位 → 映射到某个 NPU
    slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
    NPU_ID=${npu_array[$slot_idx]}

    echo "Job slot {%}: Running on NPU ${NPU_ID} (MODEL={1}, DATASET={2}, PRED={3})"
    export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}

    python main.py \
        DATA.NAME {2} \
        DATA.PRED_LEN {3} \
        MODEL.NAME {1} \
        MODEL.pred_len {3} \
        TRAIN.ENABLE True \
        TRAIN.CHECKPOINT_DIR checkpoints/{1}/{2}_{3}/ \
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