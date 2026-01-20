#!/bin/bash
NPUS=(0 1 2 3 4 5 6 7)          # Available NPU IDs
NNPU=${#NPUS[@]}        # Number of NPUs

PER_NPU=1               # Parallel jobs per NPU
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
PRED_LENS=(96 192 336 720)


# MODELS=("DLinear")
# PRED_LENS=(192)
# DATASETS=("ETTm1")
# TARGETS=("ETTm2")

# 固定迁移对: Source:Target
PAIRS=("ETTh1:ETTh2" "ETTh2:ETTh1" "ETTm1:ETTm2" "ETTm2:ETTm1")

# LRS=(0.005 0.003 0.002 0.001 0.0005 0.0001)
LRS=(0.005 0.003 0.001 0.0005 0.0001)

parallel --lb -j ${TOTAL_JOBS} '
    npu_array=($NPU_STR)

    # Map parallel job slot to NPU ID
    slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
    NPU_ID=${npu_array[$slot_idx]}

    SEED=0
    RESULT_DIR="./results/PETSA/"
    
    MODEL={1}
  
    # Parse Dataset Pair
    PAIR={2}
    DATASET=$(echo $PAIR | cut -d: -f1)
    TARGET=$(echo $PAIR | cut -d: -f2)

    PRED_LEN={3}
    BASE_LR={4}

    WEIGHT_DECAY=0.0001
    LOW_RANK=16
    LOSS_ALPHA=0.1
    GATING_INIT=0.01

    echo "Job {%}: MODEL=${MODEL} DATASET=${DATASET} PRED=${PRED_LEN} TARGET=${TARGET} LR=${BASE_LR} -> Running on NPU $NPU_ID"
    
    # export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}
    export CUDA_VISIBLE_DEVICES=${NPU_ID}

    python main.py \
        SEED ${SEED} \
        DATA.NAME ${DATASET} \
        DATA.PRED_LEN ${PRED_LEN} \
        DATA.DOMAIN_SHIFT_TARGET ${TARGET} \
        MODEL.NAME ${MODEL} \
        MODEL.pred_len ${PRED_LEN} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/ \
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
        
' ::: "${MODELS[@]}" ::: "${PAIRS[@]}" ::: "${PRED_LENS[@]}" ::: "${LRS[@]}"
