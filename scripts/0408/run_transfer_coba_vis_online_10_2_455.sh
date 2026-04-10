#!/bin/bash
# NPUS=(0 1 2 3 4 5 6 7)          # Available NPU IDs
NPUS=(0 1 2 3)          # Available NPU IDs
NNPU=${#NPUS[@]}        # Number of NPUs

PER_NPU=4               # Parallel jobs per NPU
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

# MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
MODELS=("PatchTSTPCD")

# 固定迁移对: Source:Target
# PAIRS=("ETTh1:ETTh2" "ETTh2:ETTh1" "ETTm1:ETTm2" "ETTm2:ETTm1" )
# PAIRS=("ETTm2:ETTm1")
# PAIRS=("ETTm2:ETTm1" "ETTh1:ETTh2")
# PAIRS=("ETTh2:ETTh1")
# PAIRS=("ETTm2:ETTm2")
# PAIRS=("ETTh1:ETTh2")
PAIRS=("eVED:eVED")

# TRAIN_IDS="['10']"
# TEST_IDS="['455']"

TRAIN_IDS="['10']"
TEST_IDS="['455']"

VAL_IDS=${TRAIN_IDS}
TRAIN_IDS_CLEAN=$(echo "${TRAIN_IDS}" | tr -d "[]'\" ")

export TRAIN_IDS
export TEST_IDS
export VAL_IDS

# PRED_LENS=(96 192 336 720)
PRED_LENS=(24 48 96 192)
# BASE_NUMS=(1 2 4 8 16 32 64 128 256)
BASE_NUMS=(32)

# LRS=(1e-1 5e-2 3e-2 1e-2)
# LRS=(1e-1 5e-2 3e-2 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5)
# LRS=(1e-1 5e-2 3e-2 1e-2 5e-3 3e-3 1e-3)
# LRS=(0.0001)
# LRS=(0.001 0.005 0.01 0.05 0.1)
# LRS=(0.0001 0.0005)
# LRS=(0.0001)
# LRS=(1e-5 1e-4 1e-3 1e-2 1e-1)
OFFLINE_LRS=(1e-3)
ONLINE_LRS=(5e-2)
# SEEDS=(0 1 2 3 4)
# SEEDS=(0 1 2)
SEEDS=(0)

LAMBDA_KEYS=(1.0)
# LAMBDA_BUDGETS=(0.01 0.1 0.0 1.0 10.0)
LAMBDA_BUDGETS=(1.0)
# LAMBDA_ORTHOS=(0.0 0.01 0.1 1.0 10.0 100.0)
LAMBDA_ORTHOS=(1.0)
# LAMBDA_KEYS=(0.0001 0.001 0.01 0.1 1.0 10.0 100.0)
QUERY_TYPES=("time-CI")

parallel --lb -j ${TOTAL_JOBS} '
  npu_array=($NPU_STR)
  
  # Map parallel job slot to NPU ID
  slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
  NPU_ID=${npu_array[$slot_idx]}
  
  export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}
  # export CUDA_VISIBLE_DEVICES=${NPU_ID}

  MODEL={1}
  
  # Parse Dataset Pair
  PAIR={2}
  DATASET=$(echo $PAIR | cut -d: -f1)
  TARGET=$(echo $PAIR | cut -d: -f2)

  PRED_LEN={3}
  OFFLINE_LR={4}
  LAMBDA_ORTHO={5}
  N_BASES={6}
  QUERY_TYPE={7}
  SEED={8}
  ONLINE_LR={9}

  CHECKPOINT_DIR="./checkpoints/0327/${MODEL}/${DATASET}_${PRED_LEN}_1e-4_ep_30_455_2_10/"

  RESULT_DIR="./results/ablation_lambda/"
  mkdir -p "${RESULT_DIR}"
  
  echo "Running experiment: ${MODEL} | ${TRAIN_IDS} -> ${TEST_IDS} | Len: ${PRED_LEN} | OFFLINE_LR: ${OFFLINE_LR} | BASE_NUMS: ${N_BASES} | SEED: ${SEED} | ONLINE_LR: ${ONLINE_LR} | LAMBDA_ORTHO: ${LAMBDA_ORTHO} | NPU_ID: ${NPU_ID}"

  python main.py \
    SEED ${SEED} \
    DATA.NAME ${DATASET} \
    DATA.SEQ_LEN ${PRED_LEN} \
    DATA.PRED_LEN ${PRED_LEN} \
    DATA.LABEL_LEN 12 \
    DATA.DOMAIN_SHIFT_TARGET ${TARGET} \
    DATA.TRAIN_VEHICLE_IDS "${TRAIN_IDS}" \
    DATA.VAL_VEHICLE_IDS "${VAL_IDS}" \
    DATA.TEST_VEHICLE_IDS "${TEST_IDS}" \
    MODEL.NAME ${MODEL} \
    MODEL.pred_len ${PRED_LEN} \
    MODEL.seq_len ${PRED_LEN} \
    MODEL.label_len 12 \
    MODEL.patch_len 8 \
    MODEL.stride 4 \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
    TEST.ENABLE False \
    TTA.ENABLE True \
    TTA.DOMAIN_SHIFT True \
    TTA.METHOD 'COBA_Vis' \
    TTA.DUAL.BATCH_SIZE 64 \
    TTA.DUAL.GATING_INIT 0.01 \
    TTA.SOLVER.BASE_LR ${OFFLINE_LR} \
    TTA.DUAL.PRETRAIN_EPOCHS 1 \
    TTA.DUAL.PAAS True \
    TTA.DUAL.ADJUST_PRED True \
    TTA.DUAL.CALI_NAME CoBA_TF_Adapter \
    TTA.DUAL.LOSS_NAME CoBA_Loss \
    TTA.DUAL.QUERY_TYPE ${QUERY_TYPE} \
    TTA.DUAL.GCM_N_BASES ${N_BASES} \
    TTA.DUAL.LAMBDA_ORTHO ${LAMBDA_ORTHO} \
    TTA.DUAL.COBA_ONLINE_LR ${ONLINE_LR} \
    TTA.DUAL.CALI_INPUT_ENABLE False \
    TTA.DUAL.CALI_OUTPUT_ENABLE True \
    TTA.DUAL.COBA_ONLINE_ENABLED True \
    TTA.VISUALIZE True \
    RESULT_DIR ${RESULT_DIR}

' ::: "${MODELS[@]}" ::: "${PAIRS[@]}" ::: "${PRED_LENS[@]}" ::: "${OFFLINE_LRS[@]}" ::: "${LAMBDA_ORTHOS[@]}" ::: "${BASE_NUMS[@]}" ::: "${QUERY_TYPES[@]}" ::: "${SEEDS[@]}" ::: "${ONLINE_LRS[@]}"