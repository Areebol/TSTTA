#!/bin/bash
NPUS=(0 1 2 3)          # Available NPU IDs
NNPU=${#NPUS[@]}        # Number of NPUs

PER_NPU=4               # Parallel jobs per NPU
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
MODELS=("PatchTST")
DATASETS=("ETTh1")
TARGETS=("ETTh2")
PRED_LENS=(96 192 336 720)
# PRED_LENS=(96)
# LRS=(0.5 0.3 0.1 0.08)
# LRS=(0.003)
# ADAPTERS=("linear")
# LRS=(0.005 0.003 0.002 0.001 0.0005 0.0001)
ADAPTERS=("complex-freq")
LRS=(0.05 0.03 0.02 0.01 0.005 0.001)

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
  ADAPTER={6}
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
    TTA.METHOD 'Output' \
    TTA.OURS.LR ${LR} \
    TTA.OURS.STEPS_PER_BATCH 1 \
    TTA.OURS.BATCH_SIZE 64 \
    TTA.OURS.GATING.INIT 0.01 \
    TTA.OURS.GATING_LR_SCALE 1 \
    TTA.OURS.PAAS True \
    TTA.OURS.ADJUST_PRED True \
    TTA.OURS.RESET False \
    TTA.DUAL.LOSS_NAME 'MSE' \
    TTA.OURS.ADAPTER.NAME ${ADAPTER} \
    TTA.OURS.GATING.NAME 'tanh' \
    TTA.VISUALIZE False \
    RESULT_DIR ${RESULT_DIR}

' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${TARGETS[@]}" ::: "${LRS[@]}" ::: "${ADAPTERS[@]}"
