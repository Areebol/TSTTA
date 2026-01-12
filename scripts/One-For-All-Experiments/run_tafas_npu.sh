#!/bin/bash
NPUS=(0 1 2 3)          # Available NPU IDs
NNPU=${#NPUS[@]}        # Number of NPUs

PER_NPU=4               # Parallel jobs per NPU
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
PRED_LENS=(96 192 336 720)

MODELS=("PatchTST")
DATASETS=("ETTh1")
TARGETS=("ETTh2")
# DATASETS=("ETTm1")
# PRED_LENS=(192)
LRS=(0.005 0.003 0.002 0.001 0.0005 0.0001)

parallel --lb -j ${TOTAL_JOBS} '
    npu_array=($NPU_STR)

    # Map parallel job slot to NPU ID
    slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
    NPU_ID=${npu_array[$slot_idx]}

    SEED=0
    WEIGHT_DECAY=0.0001
    GATING_INIT=0.01
    RESULT_DIR="./results/TAFAS/"

    DATASET={2}
    BASE_LR={5}
    echo "DATASET: ${DATASET}"

    echo "Job {%}: MODEL={1} DATASET={2} PRED={3} TARGET={4} LR={5} -> Running on NPU $NPU_ID"
    
    export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}

    python main.py \
        SEED ${SEED} \
        DATA.NAME {2} \
        DATA.PRED_LEN {3} \
        DATA.DOMAIN_SHIFT_TARGET {4} \
        MODEL.NAME {1} \
        MODEL.pred_len {3} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR checkpoints/{1}/{2}_{3}/ \
        TEST.ENABLE False \
        TTA.ENABLE True \
        TTA.DOMAIN_SHIFT True \
        TTA.SOLVER.BASE_LR ${BASE_LR} \
        TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
        TTA.TAFAS.GATING_INIT ${GATING_INIT} \
        RESULT_DIR ${RESULT_DIR} \
        TTA.METHOD TAFAS
        
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${TARGETS[@]}"  ::: "${LRS[@]}"