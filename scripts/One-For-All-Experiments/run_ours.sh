#!/bin/bash
GPUS=(0 5 6 7)
NUM_GPUS=${#GPUS[@]}

# MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
# DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
# PRED_LENS=(96 192 336 720)
MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
# MODELS=("DLinear")
DATASETS=("ETTh1")
# PRED_LENS=(96)
PRED_LENS=(96 192 336 720)

parallel -j 8 --delay 0 '
  export CUDA_VISIBLE_DEVICES="1"
  SEED=0

  MODEL={1}
  DATASET={2}
  PRED_LEN={3}
  CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}"

  OURS_LR=1e-3
  OURS_LR_SCALE=100
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
    TTA.OURS.LR 1e-4 \
    TTA.OURS.STEPS 1 \
    TTA.OURS.BATCH_SIZE 24 \
    TTA.OURS.GATING.INIT 0.01 \
    TTA.OURS.S_MAX 1.0 \
    TTA.OURS.EPS 1e-6 \
    TTA.OURS.GATING_LR_SCALE 100 \
    TTA.OURS.PAAS False \
    TTA.OURS.ADJUST_PRED False \
    TTA.RESET False \
    TTA.OURS.GATING.NAME 'ci-loss-trend' \
    TTA.OURS.GATING.WIN_SIZE 90 \
    TTA.OURS.LOSS.REG_COEFF 0.005 \
    TTA.VISUALIZE False \
    RESULT_DIR ${RESULT_DIR}

' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}"
    # TTA.OURS.GATING.NAME 'abs-tanh' \