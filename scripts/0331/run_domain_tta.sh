#!/bin/bash
# NPUS=(0 1 2 3 4 5 6 7)          # Available NPU IDs
NPUS=(0 1 2 3)          # Available NPU IDs
NNPU=${#NPUS[@]}        # Number of NPUs

PER_NPU=4               # Parallel jobs per NPU
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

MODELS=( "iTransformer"  "MICN" )
DATASETS=("exchange_rate")
PRED_LENS=(96 192 336 720)
# PRED_LENS=(720)
DATASETS=("exchange_rate")
# MODELS=("FreTS")
# DATASETS=("ETTm1")
TARGETS=("exchange_rate")
# DATASETS=("ETTm1")
# PRED_LENS=(192)
TTA_LRS=(1e-2 1e-3 1e-4 1e-5)
# LRS=(0.005 0.003 0.001 0.0005)
TTA_METHODS=("TAFAS" "PETSA" "DynaTTA")
SEEDS=(0 1 2 3 4)


parallel --lb -j ${TOTAL_JOBS} '
    npu_array=($NPU_STR)

    # Map parallel job slot to NPU ID
    slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
    NPU_ID=${npu_array[$slot_idx]}

    WEIGHT_DECAY=0.0001
    GATING_INIT=0.01
    RESULT_DIR="./results/0401/Domain_TTA/SEED_${SEED}/${TTA_METHOD}/${MODEL}/"

    DATASET={2}
    BASE_LR={4}
    TTA_METHOD={5}
    SEED={6}

    echo "Job {%}: MODEL={1} DATASET={2} PRED={3} LR={4} METHOD={5} -> Running on NPU $NPU_ID"

    export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}
    # export CUDA_VISIBLE_DEVICES=${NPU_ID}

    python main.py \
        SEED ${SEED} \
        DATA.NAME {2} \
        DATA.PRED_LEN {3} \
        MODEL.NAME {1} \
        MODEL.pred_len {3} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR checkpoints/{1}/{2}_{3}/ \
        TEST.ENABLE True \
        TTA.ENABLE True \
        TTA.DOMAIN_SHIFT False \
        TTA.SOLVER.BASE_LR ${BASE_LR} \
        TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
        TTA.TAFAS.GATING_INIT ${GATING_INIT} \
        RESULT_DIR ${RESULT_DIR} \
        TTA.METHOD ${TTA_METHOD}
        
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}"  ::: "${TTA_LRS[@]}" ::: "${TTA_METHODS[@]}" ::: "${SEEDS[@]}"