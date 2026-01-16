# #!/bin/bash
# QUERY_TYPES=("freq-base")
# MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
# # DATASETS=("ETTm1" "ETTm2" "exchange_rate" "weather")
# # PRED_LENS=(96 192 336 720)
# # MODELS=("DLinear")
# # DATASETS=("weather")
# DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
# PRED_LENS=(96 192 336 720)
# BASE_NS=(6)
# # BASE_NS=(2 10 12 14 16 18 20 22 24)
# # LRS=(0.0001)
# LRS=(0.00001 0.00005 0.0001 0.0005 0.001)
# # ORTH_LOSSES=(0.1 0.05 0.01 0.005 0.001)
# ORTH_LOSSES=(0.01)
# # ONLINE_LRS=(0.001 0.0001)
# ONLINE_LRS=(0.001 0.0001 0.0003)

# NPUS=(1 2 3 4 5 6 7)          # 可用的 NPU ID
# NNPU=${#NPUS[@]}        # NPU 数量

# PER_NPU=1               # 每个 NPU 并行任务数
# TOTAL_JOBS=$(( NNPU * PER_NPU ))

# NPU_STR="${NPUS[*]}"
# export NPU_STR

# parallel --lb -j ${TOTAL_JOBS} '
#   npu_array=($NPU_STR)

#   # parallel 的任务槽位 → 映射为某个 NPU ID
#   slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
#   NPU_ID=${npu_array[$slot_idx]}
#   export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}

#   SEED=0
#   BASE_LR=0.001
#   WEIGHT_DECAY=0.0001
#   GATING_INIT=0.01
#   RESULT_DIR="./results/TAFAS/"
#   MODEL={1}
#   DATASET={2}
#   PRED_LEN={3}
#   GCM_N_BASES={4}
#   query_type={5}
#   BASE_LR={6}
#   ORTH={7}
#   online_lr={8}

#   echo "Job slot {%}: NPU=${NPU_ID}  MODEL={1}  DATASET={2}  PRED={3}  GCM_N_BASES={4} query_type={5} lr={6} ORTH={7} online_lr={8}"

#   CUDA_VISIBLE_DEVICES=0 python main.py \
#       SEED ${SEED} \
#       DATA.NAME ${DATASET} \
#       DATA.PRED_LEN ${PRED_LEN} \
#       MODEL.NAME ${MODEL} \
#       MODEL.pred_len ${PRED_LEN} \
#       TRAIN.ENABLE False \
#       TRAIN.CHECKPOINT_DIR checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/ \
#       TEST.ENABLE False \
#       TTA.ENABLE True \
#       TTA.DOMAIN_SHIFT False \
#       TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
#       TTA.DUAL.GATING_INIT ${GATING_INIT} \
#       TTA.DUAL.PETSA_LOWRANK 16 \
#       TTA.DUAL.CALI_NAME lowrank-coba-GCM \
#       TTA.DUAL.LOSS_NAME LOWRANK-COBA \
#       TTA.DUAL.CALI_INPUT_ENABLE False \
#       TTA.DUAL.CALI_OUTPUT_ENABLE True \
#       TTA.DUAL.ADJUST_PRED True \
#       RESULT_DIR ${RESULT_DIR} \
#       TTA.SOLVER.BASE_LR ${BASE_LR} \
#       TTA.DUAL.GCM_N_BASES ${GCM_N_BASES} \
#       TTA.DUAL.GCM_VAR_WISE True \
#       TTA.DUAL.PRETRAIN_EPOCHS 2 \
#       TRAIN.BATCH_SIZE 512 \
#       TTA.DUAL.COBA_ONLINE_ENABLED True \
#       TTA.DUAL.COBA_ONLINE_LR ${online_lr} \
#       TTA.DUAL.QUERY_TYPE ${query_type} \
#       TTA.DUAL.LAMBDA_ORTHO ${ORTH} \
#       TTA.METHOD Ours-tta
#   ' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${BASE_NS[@]}" ::: "${QUERY_TYPES[@]}" ::: "${LRS[@]}" ::: "${ORTH_LOSSES[@]}" ::: "${ONLINE_LRS[@]}"


#!/bin/bash
QUERY_TYPES=("freq-base-CI")
MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
# DATASETS=("ETTm1" "ETTm2" "exchange_rate" "weather")
# DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
BASE_NS=(6)
PRED_LENS=(96 192 336 720)
LRS=(0.0001 0.001 0.0005 0.00001)
ORTH_LOSSES=(0.0)
ONLINE_LRS=(0.001)
MODELS=("DLinear")
DATASETS=("ETTm1")
TARGETS=("ETTm2")
PRED_LENS=(720)
LRS=(0.0001)
ORTH_LOSSES=(0.01)
ONLINE_LRS=(0.001)

# NPUS=(1 2 3 4 5 6 7)          # 可用的 NPU ID
NPUS=(0 1 2 3)          # 可用的 NPU ID
NNPU=${#NPUS[@]}        # NPU 数量

PER_NPU=4               # 每个 NPU 并行任务数
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
  GCM_N_BASES={4}
  query_type={5}
  BASE_LR={6}
  ORTH={7}
  online_lr={8}

  echo "Job slot {%}: NPU=${NPU_ID}  MODEL={1}  DATASET={2}  PRED={3}  GCM_N_BASES={4} query_type={5} lr={6} ORTH={7} online_lr={8}"

  python main.py \
      SEED ${SEED} \
      DATA.NAME ${DATASET} \
      DATA.PRED_LEN ${PRED_LEN} \
      MODEL.NAME ${MODEL} \
      DATA.DOMAIN_SHIFT_TARGET {9} \
      MODEL.pred_len ${PRED_LEN} \
      TRAIN.ENABLE False \
      TRAIN.CHECKPOINT_DIR checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/ \
      TEST.ENABLE False \
      TTA.ENABLE True \
      TTA.DOMAIN_SHIFT False \
      TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
      TTA.DUAL.GATING_INIT ${GATING_INIT} \
      TTA.DUAL.PETSA_LOWRANK 16 \
      TTA.DUAL.CALI_NAME lowrank-coba-GCM \
      TTA.DUAL.LOSS_NAME LOWRANK-COBA \
      TTA.DUAL.CALI_INPUT_ENABLE False \
      TTA.DUAL.CALI_OUTPUT_ENABLE True \
      TTA.DUAL.ADJUST_PRED True \
      RESULT_DIR ${RESULT_DIR} \
      TTA.SOLVER.BASE_LR ${BASE_LR} \
      TTA.DUAL.GCM_N_BASES ${GCM_N_BASES} \
      TTA.DUAL.GCM_VAR_WISE True \
      TTA.DUAL.PRETRAIN_EPOCHS 2 \
      TRAIN.BATCH_SIZE 512 \
      TTA.DUAL.COBA_ONLINE_ENABLED False \
      TTA.DOMAIN_SHIFT True \
      TTA.DUAL.COBA_ONLINE_LR ${online_lr} \
      TTA.DUAL.QUERY_TYPE ${query_type} \
      TTA.DUAL.LAMBDA_ORTHO ${ORTH} \
      TTA.METHOD Ours-tta
  ' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${BASE_NS[@]}" ::: "${QUERY_TYPES[@]}" ::: "${LRS[@]}" ::: "${ORTH_LOSSES[@]}" ::: "${ONLINE_LRS[@]}" ::: "${TARGETS[@]}"