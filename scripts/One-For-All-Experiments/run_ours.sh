#!/bin/bash
GPUS=(0 5 6 7)
NUM_GPUS=${#GPUS[@]}

# MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
# DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
# PRED_LENS=(96 192 336 720)
MODELS=("DLinear")
DATASETS=("ETTh1")
PRED_LENS=(96)

parallel -j 4 --delay 0 '
  GPU_ID=0
  SEED=0
  export CUDA_VISIBLE_DEVICES="5"

  SEQ_LEN=96

  patch_len=8
  stride=4

  TRAIN_EPOCHS=30
  TRAIN_BATCH_SIZE=512
  TRAIN_LR=1e-3
  MODEL={1}
  DATASET={2}
  PRED_LEN={3}

  CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}"

  FT_EPOCHS=1
  FT_LR=1e-2
  FT_MODE="linear_head"

  ADAPTER_ROOT="./checkpoints/norm_knowledge/${MODEL}/${DATASET}_${PRED_LEN}_${FT_MODE}_flr${FT_LR}_e${FT_EPOCHS}"

  OURS_LR=1e-4
  OURS_LR_SCALE=100
  OURS_STEPS=1
  OURS_BATCH_SIZE=24
  GATING_INIT=0.01      
  SOFTMAX=True
  S_MAX=1.0
  EPS=1e-6
  RESET=False

  RESULT_DIR="./results/OURS_tta/${MODEL}/${DATASET}_${PRED_LEN}_adapter_lr${OURS_LR}"
  mkdir -p "${RESULT_DIR}"

  python main.py \
    SEED ${SEED} \
    DATA.NAME {2} \
    DATA.PRED_LEN {3} \
    MODEL.NAME {1} \
    MODEL.pred_len {3} \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
    TEST.ENABLE False \
    TTA.ENABLE True \
    TTA.METHOD "Ours" \
    TTA.OURS.KNOWLEDGE_TYPE "adapter" \
    TTA.OURS.LR ${OURS_LR} \
    TTA.OURS.STEPS ${OURS_STEPS} \
    TTA.OURS.BATCH_SIZE ${OURS_BATCH_SIZE} \
    TTA.OURS.GATING_INIT ${GATING_INIT} \
    TTA.OURS.SOFTMAX ${SOFTMAX} \
    TTA.OURS.S_MAX ${S_MAX} \
    TTA.OURS.EPS ${EPS} \
    TTA.OURS.GATING_LR_SCALE ${OURS_LR_SCALE} \
    TTA.OURS.PAAS False \
    TTA.OURS.ADJUST_PRED False \
    TTA.RESET ${RESET} \
    TTA.OURS.USE_CONTEXT False \
    RESULT_DIR ${RESULT_DIR}

' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}"