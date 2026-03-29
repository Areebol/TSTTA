#!/bin/bash
set -e

# usage: bash train.sh [GPU_ID]
# GPU_ID="4"
# export CUDA_VISIBLE_DEVICES="${GPU_ID}"
# echo "[INFO] Using GPU: ${CUDA_VISIBLE_DEVICES}"


# dataset / model config
DATASET="eVED"
SEQ_LEN=24
LABEL_LEN=12
PRED_LEN=24
patch_len=16
stride=8
# MODEL="LSTM"
# MODEL="PatchTST"
# MODEL="LSTM"
# MODEL="DLinear"

# MODELS=("iTransformer" "Informer" "Reformer" "Autoformer" "ETSformer" "FEDformer" "Pyraformer" "TST")
# MODELS=("LSTM" "PatchTSTPCD" "TimeXer" "TimeXerPCD" "TimeXerHCM" "TimeXerRoadM" "TimeXerRoadMMh")
MODELS=("DLinearPCD" "FreTSPCD")
# DATASETS=("sVED")
DATASETS=("eVED")
# DATASETS=("oVED")
# DATASETS=("oeVED")
# PRED_LENS=(24 48 96)
# PRED_LENS=(192)
PRED_LENS=(24)

# training hyperparams (tune as needed)

# LRS=(1e-2 1e-3 1e-4 1e-5)
# LRS=(1e-1 1e-2 1e-3 1e-4)
LRS=(1e-4)

NPUS=(1 2 )          # 可用的 NPU ID
# NPUS=(0 1 2 3 4 5 6 7)  # 可用的 NPU ID
NNPU=${#NPUS[@]}        # NPU 数量

PER_NPU=1               # 每个 NPU 并行任务数
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR


parallel --lb -j ${TOTAL_JOBS} '

  npu_array=($NPU_STR)

  # slot_idx：parallel 的任务槽位 → 映射到某个 NPU
  slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
  NPU_ID=${npu_array[$slot_idx]}


  MODEL={1}
  DATASET={2}
  PRED_LEN={3}
  LR={4}

  SEQ_LEN={3}
  EPOCHS=30
  BATCH_SIZE=256
  patch_len=4
  stride=4
  LABEL_LEN=12

  CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}_${LR}_2"
  mkdir -p "${CHECKPOINT_DIR}"

  export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}

  echo "[INFO] Dataset: ${DATASET}, Model: ${MODEL}, Pred_len: ${PRED_LEN}, LR ${LR}, NPU: ${NPU_ID}"
  echo "[INFO] Checkpoint dir: ${CHECKPOINT_DIR}"
  echo "[INFO] Epochs: ${EPOCHS}, Batch size: ${BATCH_SIZE}, LR: ${LR}"

  python main.py \
    DATA.NAME ${DATASET} \
    DATA.SEQ_LEN ${SEQ_LEN} \
    DATA.PRED_LEN ${PRED_LEN} \
    DATA.LABEL_LEN ${LABEL_LEN} \
    MODEL.NAME ${MODEL} \
    MODEL.pred_len ${PRED_LEN} \
    MODEL.label_len ${LABEL_LEN} \
    MODEL.seq_len ${SEQ_LEN} \
    MODEL.patch_len ${patch_len} \
    MODEL.stride ${stride} \
    MODEL.use_norm False \
    TRAIN.ENABLE True \
    SOLVER.MAX_EPOCH ${EPOCHS} \
    TRAIN.BATCH_SIZE ${BATCH_SIZE} \
    SOLVER.BASE_LR ${LR} \
    TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
    TRAIN.FINETUNE False \
    TEST.ENABLE True \
    TTA.ENABLE False 
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${LRS[@]}"
