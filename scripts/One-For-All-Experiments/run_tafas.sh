#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0

# MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
# DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
MODELS=("MICN")

# PRED_LENS=(96 192 336 720)
DATASETS=("ETTm2")
# DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
TARGETS=("ETTh1")
# MODELS=("DLinear")
PRED_LENS=(720)


# parallel -j 8 --delay 0 '
#     GPU=7
#     SEED=0
#     BASE_LR=0.001
#     WEIGHT_DECAY=0.0001
#     GATING_INIT=0.01
#     RESULT_DIR="./results/TAFAS/"

#     echo "Job {#}: MODEL={1} DATASET={2} PRED={3} -> Running on GPU $GPU"
    
#     CUDA_VISIBLE_DEVICES=$GPU python main.py \
#         SEED ${SEED} \
#         DATA.NAME {2} \
#         DATA.DOMAIN_SHIFT_TARGET {4} \
#         DATA.PRED_LEN {3} \
#         MODEL.NAME {1} \
#         MODEL.pred_len {3} \
#         TRAIN.ENABLE False \
#         TRAIN.CHECKPOINT_DIR checkpoints/{1}/{2}_{3}/ \
#         TEST.ENABLE False \
#         TTA.ENABLE True \
#         TTA.DOMAIN_SHIFT True \
#         TTA.SOLVER.BASE_LR ${BASE_LR} \
#         TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
#         TTA.TAFAS.GATING_INIT ${GATING_INIT} \
#         RESULT_DIR ${RESULT_DIR} \
#         TTA.METHOD TAFAS
        
# ' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${TARGETS[@]}"



NPUS=(0 1 2 3)          # 可用的 NPU ID
NNPU=${#NPUS[@]}        # NPU 数量

PER_NPU=8               # 每个 NPU 并行任务数
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

parallel --lb -j ${TOTAL_JOBS} '
    npu_array=($NPU_STR)

    # parallel 的任务槽位 → 映射为某个 NPU ID
    slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
    NPU_ID=${npu_array[$slot_idx]}

    SEED=0
    BASE_LR=0.001
    WEIGHT_DECAY=0.0001
    GATING_INIT=0.01
    RESULT_DIR="./results/TAFAS/"

    echo "Job slot {%}: NPU=${NPU_ID}  MODEL={1}  DATASET={2}  PRED={3}  TARGET={4}"

    export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}

    python main.py \
        SEED ${SEED} \
        DATA.NAME {2} \
        DATA.DOMAIN_SHIFT_TARGET {4} \
        DATA.PRED_LEN {3} \
        MODEL.NAME {1} \
        MODEL.pred_len {3} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR checkpoints/{1}/{2}_{3}/ \
        TEST.ENABLE False \
        TTA.ENABLE True \
        TTA.DOMAIN_SHIFT False \
        TTA.SOLVER.BASE_LR ${BASE_LR} \
        TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
        TTA.TAFAS.GATING_INIT ${GATING_INIT} \
        RESULT_DIR ${RESULT_DIR} \
        TTA.METHOD TAFAS
        
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${TARGETS[@]}"