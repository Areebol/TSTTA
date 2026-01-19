# #!/bin/bash
# QUERY_TYPES=("freq-base")
# MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
# # DATASETS=("ETTm1" "ETTm2" "exchange_rate" "weather")
# PRED_LENS=(96 192 336 720)
# MODELS=("DLinear")
# # DATASETS=("weather")
# DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
# PRED_LENS=(96 192 336 720)
# TARGETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2")  # 添加域移目标数据集
# BASE_NS=(6)
# # BASE_NS=(2 10 12 14 16 18 20 22 24)
# # LRS=(0.0001)
# LRS=(0.00001 0.00005 0.0001 0.0005 0.001)
# # ORTH_LOSSES=(0.1 0.05 0.01 0.005 0.001)
# ORTH_LOSSES=(0.01)
# # ONLINE_LRS=(0.001 0.0001)
# ONLINE_LRS=(0.001 0.0001 0.0003)

# NPUS=(0 1 2 3)          # 可用的 NPU ID
# NNPU=${#NPUS[@]}        # NPU 数量

# PER_NPU=2             # 每个 NPU 并行任务数
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
#   TARGET={9}

#   echo "Job slot {%}: NPU=${NPU_ID}  MODEL={1}  DATASET={2}  PRED={3}  GCM_N_BASES={4} query_type={5} lr={6} ORTH={7} online_lr={8} TARGET={9}"

#   CUDA_VISIBLE_DEVICES=0 python main.py \
#       SEED ${SEED} \
#       DATA.NAME ${DATASET} \
#       DATA.PRED_LEN ${PRED_LEN} \
#       DATA.DOMAIN_SHIFT_TARGET ${TARGET} \  
#       MODEL.NAME ${MODEL} \
#       MODEL.pred_len ${PRED_LEN} \
#       TRAIN.ENABLE False \
#       TRAIN.CHECKPOINT_DIR checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/ \
#       TEST.ENABLE False \
#       TTA.ENABLE True \
#       TTA.DOMAIN_SHIFT True \
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
QUERY_TYPES=("freq-base-CD")
# MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
MODELS=("DLinear")
PRED_LENS=(96 192 336 720)
# PRED_LENS=(96)
DATASETS=("ETTh1")
TARGETS=("ETTh2")
BASE_NS=(1 2 3 4 5 6)
# BASE_NS=(6)
LRS=(0.00001)

# ORTH_LOSSES=(0.1 0.05 0.01 0.005 0.001)
ORTH_LOSSES=(0.01)
# ONLINE_LRS=(0.001)
ONLINE_LRS=(0.001 0.003)

# NPUS=(1 2 3 4 5 6 7)          # 可用的 NPU ID
NPUS=(0 1 2 3)          # 可用的 NPU ID
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

    SEED=0
    RESULT_DIR="./results/TAFAS/"

    echo "Job slot {%}: NPU=${NPU_ID}  MODEL={1}  DATASET={2}  PRED={3}  GCM_N_BASES={4} query_type={5} lr={6} ORTH={7} online_lr={8} TARGET={9}"

    export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}

    python main.py \
        SEED ${SEED} \
        DATA.NAME {2} \
        DATA.PRED_LEN {3} \
        DATA.DOMAIN_SHIFT_TARGET {9} \
        MODEL.NAME {1} \
        MODEL.pred_len {3} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR checkpoints/{1}/{2}_{3}/ \
        TEST.ENABLE False \
        TTA.ENABLE True \
        TTA.DOMAIN_SHIFT True \
        TTA.SOLVER.WEIGHT_DECAY 0.0001 \
        TTA.DUAL.GATING_INIT 0.01 \
        TTA.DUAL.PETSA_LOWRANK 16 \
        TTA.DUAL.CALI_NAME lowrank-coba-GCM \
        TTA.DUAL.LOSS_NAME LOWRANK-COBA \
        TTA.DUAL.CALI_INPUT_ENABLE False \
        TTA.DUAL.CALI_OUTPUT_ENABLE True \
        TTA.DUAL.ADJUST_PRED True \
        RESULT_DIR ${RESULT_DIR} \
        TTA.SOLVER.BASE_LR {6} \
        TTA.DUAL.GCM_N_BASES {4} \
        TTA.DUAL.GCM_VAR_WISE True \
        TTA.DUAL.PRETRAIN_EPOCHS 1 \
        TRAIN.BATCH_SIZE 512 \
        TTA.DUAL.COBA_ONLINE_ENABLED True \
        TTA.DUAL.COBA_ONLINE_LR {8} \
        TTA.DUAL.QUERY_TYPE {5} \
        TTA.DUAL.LAMBDA_ORTHO {7} \
        TTA.METHOD Ours-tta
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${BASE_NS[@]}" ::: "${QUERY_TYPES[@]}" ::: "${LRS[@]}" ::: "${ORTH_LOSSES[@]}" ::: "${ONLINE_LRS[@]}" ::: "${TARGETS[@]}"