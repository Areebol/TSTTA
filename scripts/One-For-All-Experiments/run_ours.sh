#!/bin/bash
GPUS=(0 5 6 7)
NUM_GPUS=${#GPUS[@]}

PRED_LENS=(720)
MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
MODELS=("DLinear")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
DATASETS=("ETTh1")
PRED_LENS=(96 192 336 720)
S_MAXS=(1.0)
WIN_SIZES=(48)
REG_COEFFS=(0.01)
LRS=(0.0001)

parallel -j 32 --delay 0 '
  export CUDA_VISIBLE_DEVICES="7"
  SEED=0

  MODEL={1}
  DATASET={2}
  PRED_LEN={3}
  S_MAX={4}
  WIN_SIZE={5}
  REG_COEFF={6}
  LR={7}
  CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}"

  OURS_STEPS=1
  OURS_BATCH_SIZE=24

  RESULT_DIR="./results/OURS_tta/${MODEL}/${DATASET}_${PRED_LEN}_adapter_lr${OURS_LR}"
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
    TTA.METHOD "Ours" \
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
    TTA.VISUALIZE False \
    RESULT_DIR ${RESULT_DIR}

' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${S_MAXS[@]}" ::: "${WIN_SIZES[@]}" ::: "${REG_COEFFS[@]}" ::: "${LRS[@]}"

python build_table.py