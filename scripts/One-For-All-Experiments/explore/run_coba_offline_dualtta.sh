#!/bin/bash
NPUS=(1 2 3)          # Available NPU IDs
NNPU=${#NPUS[@]}        # Number of NPUs

PER_NPU=4               # Parallel jobs per NPU
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
MODELS=("DLinear")
DATASETS=("ETTm2")
TARGETS=("ETTm1")
PRED_LENS=(96 192 336 720)
# PRED_LENS=(96)
# LRS=(0.5 0.3 0.1 0.08)

# LRS=(0.001)
# LRS=(0.01 0.005 0.003 0.001 0.0005 0.0001)
# LRS=(1e-2 5e-3 1e-3 5e-4 1e-4 5e-5)
LRS=(0.0001)
# LAMBDA_ORTHO=(1e1 1e0 1e-1 1e-2 1e-3 1e-4)
LAMBDA_ORTHO=(1e-2)

parallel --lb -j ${TOTAL_JOBS} '
  npu_array=($NPU_STR)
  
  # Map parallel job slot to NPU ID
  slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
  NPU_ID=${npu_array[$slot_idx]}
  
  export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}
  SEED=0

  MODEL={1}
  DATASET={2}
  PRED_LEN={3}
  TARGET={4}
  LR={5}
  LAMBDA_ORTHO={6}
  CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}"

  RESULT_DIR="./results/output_tta/"
  mkdir -p "${RESULT_DIR}"

  python main.py \
    SEED ${SEED} \
    DATA.NAME ${DATASET} \
    DATA.PRED_LEN ${PRED_LEN} \
    DATA.DOMAIN_SHIFT_TARGET ${TARGET} \
    MODEL.NAME ${MODEL} \
    MODEL.pred_len ${PRED_LEN} \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
    TEST.ENABLE False \
    TTA.ENABLE True \
    TTA.DOMAIN_SHIFT True \
    TTA.METHOD 'Dual-tta' \
    TTA.DUAL.BATCH_SIZE 64 \
    TTA.DUAL.GATING_INIT 0.01 \
    TTA.SOLVER.BASE_LR ${LR} \
    TTA.DUAL.PAAS True \
    TTA.DUAL.ADJUST_PRED True \
    TTA.DUAL.CALI_NAME lowrank-coba-GCM \
    TTA.DUAL.LOSS_NAME LOWRANK-COBA \
    TTA.DUAL.LAMBDA_ORTHO ${LAMBDA_ORTHO} \
    TTA.DUAL.COBA_ONLINE_LR 1e-3 \
    TTA.DUAL.CALI_INPUT_ENABLE False \
    TTA.DUAL.CALI_OUTPUT_ENABLE True \
    TTA.DUAL.COBA_ONLINE_ENABLED False \
    TTA.VISUALIZE False \
    RESULT_DIR ${RESULT_DIR}

' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${TARGETS[@]}" ::: "${LRS[@]}" ::: "${LAMBDA_ORTHO[@]}"
