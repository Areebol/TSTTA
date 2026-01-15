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


MODELS=("DLinear")
# PRED_LENS=(192)
DATASETS=("ETTm2")
TARGETS=("ETTm1")
# LRS=(0.005 0.003 0.002 0.001 0.0005 0.0001)
LRS=(0.001)

parallel --lb -j ${TOTAL_JOBS} '
    npu_array=($NPU_STR)

    # Map parallel job slot to NPU ID
    slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
    NPU_ID=${npu_array[$slot_idx]}

    SEED=0
    RESULT_DIR="./results/PETSA/"
    BASE_LR={5}
    WEIGHT_DECAY=0.0001
    LOW_RANK=16
    LOSS_ALPHA=0.1
    GATING_INIT=0.01

    echo "Job {%}: MODEL={1} DATASET={2} PRED={3} TARGET={4} LR=${BASE_LR} -> Running on NPU $NPU_ID"
    
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
        TEST.ENABLE True \
        TTA.ENABLE True \
        TTA.DOMAIN_SHIFT True \
        TTA.SOLVER.BASE_LR ${BASE_LR} \
        TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
        TTA.PETSA.GATING_INIT ${GATING_INIT} \
        TTA.PETSA.RANK ${LOW_RANK} \
        TTA.PETSA.LOSS_ALPHA ${LOSS_ALPHA} \
        RESULT_DIR ${RESULT_DIR} \
        TTA.METHOD PETSA
        
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${TARGETS[@]}" ::: "${LRS[@]}"
