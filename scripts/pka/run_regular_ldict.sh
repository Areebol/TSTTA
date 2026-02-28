#!/bin/bash
# NPUS=(0 1 2 3 4 5 6 7)          # Available NPU IDs
NPUS=(4 5 6 7)          # Available NPU IDs
NNPU=${#NPUS[@]}        # Number of NPUs

PER_NPU=1               # Parallel jobs per NPU
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
MODELS=("PatchTST")

# 固定迁移对: Source:Target
# PAIRS=("ETTh1:ETTh2" "ETTh2:ETTh1" "ETTm1:ETTm2" "ETTm2:ETTm1")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
DATASETS=("ETTh1")

PRED_LENS=(96 192 336 720)
PRED_LENS=(720)
# PATTERN_NUMS=(2 4 8 16 32 64 128 256 512)
PATTERN_NUMS=(64)
# PATTERN_NUMS=(16)
# PATTERN_NUMS=(128)

# LRS=(1e-1 5e-2 3e-2 1e-2)
# OFFLINE_LRS=(1e-1 5e-2 3e-2 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5)
# OFFLINE_LRS=(0.001 0.003 0.005 0.01 0.03 0.05 0.1)
# OFFLINE_LRS=(1e-1 1e-2 1e-3 1e-4 1e-5)
OFFLINE_LRS=(0.01)
# OFFLINE_LRS=(5e-4 1e-4 5e-5)
# OFFLINE_LRS=(5e-4 3e-4 1e-4 5e-5 1e-5)

# ONLINE_LRS=(0.1 0.05 0.03 0.01 0.005 0.001)
# OFFLINE_LRS=(0.01 0.03)
# OFFLINE_LRS=(0.03)
# ONLINE_LRS=(0.1 0.05 0.01 0.005 0.001 0.0005 0.0001)
ONLINE_LRS=(0.01)

LAMBDA_ORTHO=(1.0)
# LAMBDA_ORTHO=(0.0 0.001 0.01 0.1 1.0 10.0 100.0 1000.0)
QUERY_TYPES=("time-CI")

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
  DATASET={2}

  PRED_LEN={3}
  OFFLINE_LR={4}
  LAMBDA_ORTHO={5}
  N_PATTERNS={6}
  QUERY_TYPE={7}
  ONLINE_LR={8}

  CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}"

  RESULT_DIR="./results/output_tta/"
  mkdir -p "${RESULT_DIR}"
  
  echo "Running experiment: ${MODEL} | ${DATASET} | Len: ${PRED_LEN} | Offline LR: ${OFFLINE_LR} | Online LR: ${ONLINE_LR} | NPU ${NPU_ID}"

  python main.py \
    SEED ${SEED} \
    DATA.NAME ${DATASET} \
    DATA.PRED_LEN ${PRED_LEN} \
    MODEL.NAME ${MODEL} \
    MODEL.pred_len ${PRED_LEN} \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
    TEST.ENABLE False \
    TTA.ENABLE True \
    TTA.DOMAIN_SHIFT False \
    TTA.METHOD 'PKA_OnLine' \
    TTA.PKA.BATCH_SIZE 64 \
    TTA.PKA.GATING_INIT 0.01 \
    TTA.SOLVER.BASE_LR ${OFFLINE_LR} \
    TTA.PKA.PRETRAIN_EPOCHS 1 \
    TTA.PKA.PAAS True \
    TTA.PKA.ADJUST_PRED True \
    TTA.PKA.CALI_NAME PKA_LDict \
    TTA.PKA.LOSS_NAME CoBA_Loss \
    TTA.PKA.QUERY_TYPE ${QUERY_TYPE} \
    TTA.PKA.N_PATTERNS ${N_PATTERNS} \
    TTA.PKA.LAMBDA_ORTHO ${LAMBDA_ORTHO} \
    TTA.PKA.COBA_ONLINE_LR ${ONLINE_LR} \
    TTA.PKA.ENERGY_THRESHOLD 0.1 \
    TTA.PKA.CALI_INPUT_ENABLE False \
    TTA.PKA.CALI_OUTPUT_ENABLE True \
    TTA.PKA.COBA_ONLINE_ENABLED False \
    TTA.VISUALIZE False \
    RESULT_DIR ${RESULT_DIR}

' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${OFFLINE_LRS[@]}" ::: "${LAMBDA_ORTHO[@]}" ::: "${PATTERN_NUMS[@]}" ::: "${QUERY_TYPES[@]}" ::: "${ONLINE_LRS[@]}"