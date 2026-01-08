#!/bin/bash
NPUS=(0 1 2 3)          # Available NPU IDs
NNPU=${#NPUS[@]}        # Number of NPUs

PER_NPU=8               # Parallel jobs per NPU
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
MODELS=("PatchTST")
DATASETS=("ETTh1" "ETTh2")
PRED_LENS=(96 192 336 720)
# PRED_LENS=(96)
S_MAXS=(1)
WIN_SIZES=(48)
REG_COEFFS=(0.01)
LRS=(0.0001)

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
  S_MAX={4}
  WIN_SIZE={5}
  REG_COEFF={6}
  LR={7}
  CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}"

  RESULT_DIR="./results/OURS_tta/${MODEL}/${DATASET}_${PRED_LEN}_adapter_lr${LR}"
  mkdir -p "${RESULT_DIR}"

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
    TTA.METHOD "Ours-tta" \
    TTA.OURS.LR ${LR} \
    TTA.OURS.STEPS_PER_BATCH 1 \
    TTA.OURS.BATCH_SIZE 24 \
    TTA.OURS.GATING.INIT 0.01 \
    TTA.OURS.S_MAX ${S_MAX} \
    TTA.OURS.EPS 1e-6 \
    TTA.OURS.GATING_LR_SCALE 100 \
    TTA.OURS.PAAS False \
    TTA.OURS.ADJUST_PRED False \
    TTA.OURS.RESET False \
    TTA.OURS.ADAPTER.NAME 'linear' \
    TTA.OURS.GATING.WIN_SIZE ${WIN_SIZE} \
    TTA.OURS.GATING.NAME 'ci-loss-trend' \
    TTA.OURS.LOSS.REG_COEFF ${REG_COEFF} \
    TTA.VISUALIZE True \
    RESULT_DIR ${RESULT_DIR}

' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${S_MAXS[@]}" ::: "${WIN_SIZES[@]}" ::: "${REG_COEFFS[@]}" ::: "${LRS[@]}"

python build_table.py