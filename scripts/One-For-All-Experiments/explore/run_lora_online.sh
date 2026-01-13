#!/bin/bash
MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
PRED_LENS=(96 192 336 720)
MODELS=("PatchTST")
DATASETS=("ETTh2")
TARGETS=("ETTh1")
PRED_LENS=(96)
LRS=(1e-2 5e-3 3e-3 1e-3 5e-4 1e-4)
LRS=(1e-3)

NPUS=(0 1 2 3)          # 可用的 NPU ID
# NPUS=(0 1 2 3 4 5 6 7)          # 可用的 NPU ID
NNPU=${#NPUS[@]}        # NPU 数量

PER_NPU=1               # 每个 NPU 并行任务数
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

parallel --lb -j ${TOTAL_JOBS} '
  npu_array=($NPU_STR)

  # parallel 的任务槽位 → 映射为某个 NPU ID
  slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
  NPU_ID=${npu_array[$slot_idx]}
  export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}

  SEED=0
  BASE_LR=0.001
  WEIGHT_DECAY=0.0001
  GATING_INIT=0.01
  RESULT_DIR="./results/TAFAS/"
  MODEL={1}
  DATASET={2}
  PRED_LEN={3}
  TARGET={4}
  TTA_LR={5}

  echo "Job slot {%}: NPU=${NPU_ID}  MODEL={1}  DATASET={2}  PRED={3} TARGET={4} TTA_LR={5}"

  CUDA_VISIBLE_DEVICES=0 python main.py \
      SEED ${SEED} \
      DATA.NAME ${DATASET} \
      DATA.PRED_LEN ${PRED_LEN} \
      DATA.DOMAIN_SHIFT_TARGET ${TARGET} \
      MODEL.NAME ${MODEL} \
      MODEL.pred_len ${PRED_LEN} \
      TRAIN.ENABLE False \
      TRAIN.CHECKPOINT_DIR checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/ \
      TEST.ENABLE False \
      TTA.ENABLE True \
      TTA.DOMAIN_SHIFT True \
      TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
      TTA.DUAL.GATING_INIT ${GATING_INIT} \
      TTA.DUAL.PETSA_LOWRANK 16 \
      TTA.DUAL.CALI_NAME lowrank-coba-GCM \
      TTA.DUAL.LOSS_NAME LOWRANK-COBA \
      TTA.DUAL.CALI_INPUT_ENABLE False \
      TTA.DUAL.CALI_OUTPUT_ENABLE True \
      TTA.DUAL.ADJUST_PRED True \
      RESULT_DIR ${RESULT_DIR} \
      TTA.SOLVER.BASE_LR 1e-4 \
      TTA.DUAL.GCM_N_BASES 6 \
      TTA.DUAL.COBA_ONLINE_ENABLED True \
      TTA.DUAL.COBA_ONLINE_LR ${TTA_LR} \
      TTA.DUAL.PRETRAIN_EPOCHS 4 \
      TRAIN.BATCH_SIZE 512 \
      TTA.METHOD Ours-tta
  ' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${TARGETS[@]}" ::: "${LRS[@]}"