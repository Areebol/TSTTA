#!/bin/bash

############################################
# 实验参数
############################################
MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
# MODELS=("DLinear")
# DATASETS=("ETTh1")
# TARGETS=("ETTh2")

# 固定迁移对: Source:Target
PAIRS=("ETTh1:ETTh2" "ETTh2:ETTh1" "ETTm1:ETTm2" "ETTm2:ETTm1")

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

    MODEL={1}
    
    # Parse Dataset Pair
    PAIR={2}
    DATASET=$(echo $PAIR | cut -d: -f1)
    TARGET=$(echo $PAIR | cut -d: -f2)

    PRED_LEN={3}

    echo "Job slot {%}: NPU=${NPU_ID}  MODEL=${MODEL}  DATASET=${DATASET}  PRED=${PRED_LEN}  TARGET=${TARGET}"

    # export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}
    export CUDA_VISIBLE_DEVICES=${NPU_ID}

    python main.py \
        SEED ${SEED} \
        DATA.NAME ${DATASET} \
        DATA.PRED_LEN ${PRED_LEN} \
        DATA.DOMAIN_SHIFT_TARGET ${TARGET} \
        DATA_LOADER.NUM_WORKERS 4 \
        MODEL.NAME ${MODEL} \
        MODEL.pred_len ${PRED_LEN} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/ \
        TEST.ENABLE True \
        TTA.ENABLE False \
        TTA.DOMAIN_SHIFT True \
        RESULT_DIR ${RESULT_DIR}
' ::: "${MODELS[@]}" ::: "${PAIRS[@]}" ::: "${PRED_LENS[@]}"
