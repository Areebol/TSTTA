#!/bin/bash
# NPUS=(0 1 2 3 4 5 6 7)          # Available NPU IDs
NPUS=(7)
NNPU=${#NPUS[@]}        # Number of NPUs

PER_NPU=1               # Parallel jobs per NPU
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

# MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
MODELS=("PatchTST")

# 固定迁移对: Source:Target
# PAIRS=("ETTh1:ETTh2" "ETTh2:ETTh1" "ETTm1:ETTm2" "ETTm2:ETTm1")
# PAIRS=("ETTm2:ETTm1")
PAIRS=("eVED:eVED")

# PRED_LENS=(96 192 336 720)
BASE_NUMS=(32)

# LRS=(1e-1 5e-2 3e-2 1e-2)
# LRS=(1e-1 5e-2 3e-2 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5)
# LRS=(2e-1 1e-1 1e-2 1e-3 1e-4)
# LRS=(1.0 0.5 0.3 0.2)
LRS=(1e-1)

LAMBDA_ORTHO=(1e-2)
QUERY_TYPES=("freq-base-CI")

PRED_LENS=(24)
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





parallel --lb -j ${TOTAL_JOBS} '
  npu_array=($NPU_STR)
  
  # Map parallel job slot to NPU ID
  slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
  NPU_ID=${npu_array[$slot_idx]}
  
  # export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}
  export CUDA_VISIBLE_DEVICES=${NPU_ID}
  SEED=0

  MODEL={1}
  
  # Parse Dataset Pair
  PAIR={2}
  DATASET=$(echo $PAIR | cut -d: -f1)
  TARGET=$(echo $PAIR | cut -d: -f2)

  PRED_LEN={3}
  LR={4}
  LAMBDA_ORTHO={5}
  N_BASES={6}
  QUERY_TYPE={7}

  CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}_ids${TRAIN_IDS_CLEAN}/"

  RESULT_DIR="./results/output_tta/"
  mkdir -p "${RESULT_DIR}"
  
  echo "Running experiment: ${MODEL} | ${DATASET} -> ${TARGET} | Len: ${PRED_LEN} | LR: ${LR}| N_BASE: ${N_BASES}"

  python main.py \
    SEED ${SEED} \
    DATA.NAME ${DATASET} \
    DATA.SEQ_LEN ${PRED_LEN} \
    DATA.PRED_LEN ${PRED_LEN} \
    DATA.LABEL_LEN 12 \
    DATA.DOMAIN_SHIFT_TARGET ${TARGET} \
    DATA.MIN_TEST_LEN 300 \
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
    TTA.METHOD 'Dual-tta' \
    TTA.DUAL.BATCH_SIZE 64 \
    TTA.DUAL.GATING_INIT 0.01 \
    TTA.SOLVER.BASE_LR ${LR} \
    TTA.DUAL.PRETRAIN_EPOCHS 5 \
    TTA.DUAL.PAAS True \
    TTA.DUAL.ADJUST_PRED True \
    TTA.DUAL.CALI_NAME RoCoBA_FreqDomain_Norm \
    TTA.DUAL.LOSS_NAME Freq-EW-CoBALoss \
    TTA.DUAL.QUERY_TYPE ${QUERY_TYPE} \
    TTA.DUAL.GCM_N_BASES ${N_BASES} \
    TTA.DUAL.LAMBDA_ORTHO ${LAMBDA_ORTHO} \
    TTA.DUAL.COBA_ONLINE_LR 1e-3 \
    TTA.DUAL.CALI_INPUT_ENABLE False \
    TTA.DUAL.CALI_OUTPUT_ENABLE True \
    TTA.DUAL.COBA_ONLINE_ENABLED False \
    TTA.VISUALIZE False \
    RESULT_DIR ${RESULT_DIR}

' ::: "${MODELS[@]}" ::: "${PAIRS[@]}" ::: "${PRED_LENS[@]}" ::: "${LRS[@]}" ::: "${LAMBDA_ORTHO[@]}" ::: "${BASE_NUMS[@]}" ::: "${QUERY_TYPES[@]}"

    # TTA.DUAL.CALI_NAME CoBA_GCM \
    # TTA.DUAL.LOSS_NAME CoBA_Loss \