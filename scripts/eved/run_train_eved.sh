#!/bin/bash
set -e

# usage: bash train.sh [GPU_ID]
GPU_ID="4"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "[INFO] Using GPU: ${CUDA_VISIBLE_DEVICES}"


# dataset / model config
DATASET="eVED"
SEQ_LEN=96
LABEL_LEN=12
PRED_LEN=96
patch_len=16
stride=8
# MODEL="LSTM"
# MODEL="PatchTST"
MODEL="PatchTSTPCD"

TRAIN_IDS="['455']"
TEST_IDS="['10']"
VAL_IDS=${TRAIN_IDS}
# Clean TRAIN_IDS for directory naming: remove [, ], ', " and spaces
TRAIN_IDS_CLEAN=$(echo "${TRAIN_IDS}" | tr -d "[]'\" ")

export TRAIN_IDS
export TEST_IDS
export VAL_IDS
export TRAIN_IDS_CLEAN

# training hyperparams (tune as needed)
EPOCHS=30
BATCH_SIZE=256
LR=1e-4

# checkpoint / save dirs
CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}_${LR}"
mkdir -p "${CHECKPOINT_DIR}"

echo "[INFO] Dataset: ${DATASET}, Model: ${MODEL}, Pred_len: ${PRED_LEN}, LR ${LR}"
echo "[INFO] Checkpoint dir: ${CHECKPOINT_DIR}"
echo "[INFO] Epochs: ${EPOCHS}, Batch size: ${BATCH_SIZE}, LR: ${LR}"

python main.py \
  DATA.NAME ${DATASET} \
  DATA.SEQ_LEN ${SEQ_LEN} \
  DATA.PRED_LEN ${PRED_LEN} \
  DATA.LABEL_LEN ${LABEL_LEN} \
  DATA.TRAIN_VEHICLE_IDS "${TRAIN_IDS}" \
  DATA.VAL_VEHICLE_IDS "${VAL_IDS}" \
  DATA.TEST_VEHICLE_IDS "${TEST_IDS}" \
  MODEL.NAME ${MODEL} \
  MODEL.pred_len ${PRED_LEN} \
  MODEL.label_len ${LABEL_LEN} \
  MODEL.seq_len ${SEQ_LEN} \
  MODEL.patch_len ${patch_len} \
  MODEL.stride ${stride} \
  TRAIN.ENABLE True \
  SOLVER.MAX_EPOCH ${EPOCHS} \
  TRAIN.BATCH_SIZE ${BATCH_SIZE} \
  SOLVER.BASE_LR ${LR} \
  TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
  TRAIN.FINETUNE False \
  TEST.ENABLE True \
  TTA.ENABLE False \
  VISIBLE_DEVICES ${GPU_ID}