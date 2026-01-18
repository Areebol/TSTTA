#!/bin/bash

############################################
# 实验参数
############################################
# MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
MODELS=("DLinear")
DATASETS=("ETTh1")
TARGETS=("ETTh2")
PRED_LENS=(96 192 336 720)
# PRED_LENS=(192)

# NPUS=(3)          # 可用的 NPU ID
NPUS=(0 1 2 3 4 5 6 7)  # 可用的 NPU ID
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

    echo "Job slot {%}: NPU=${NPU_ID}  MODEL={1}  DATASET={2}  PRED={3}  TARGET={4}"

    # export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}
    export CUDA_VISIBLE_DEVICES=${NPU_ID}

    python main.py \
        SEED ${SEED} \
        DATA.NAME {2} \
        DATA.PRED_LEN {3} \
        DATA.DOMAIN_SHIFT_TARGET {4} \
        MODEL.NAME {1} \
        MODEL.pred_len {3} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR checkpoints/{1}/{2}_{3}/ \
        TEST.ENABLE True \
        TTA.ENABLE False \
        TTA.DOMAIN_SHIFT True \
        RESULT_DIR ${RESULT_DIR}
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${TARGETS[@]}"
