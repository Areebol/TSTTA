#!/bin/bash
#  MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
MODELS=("DLinear")
PRED_LENS=(96 192 336 720)
# PRED_LENS=(336)
# TARGETS=("ETTm2" )
# DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
# MODELS=("iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" )

NPUS=(5 6 7)          # 可用的 NPU ID
NNPU=${#NPUS[@]}        # NPU 数量

# PER_NPU=8               # 每个 NPU 并行任务数
PER_NPU=1               # 每个 NPU 并行任务数
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

parallel -j 8 --delay 0 '
    npu_array=($NPU_STR)

    # parallel 的任务槽位 → 映射为某个 NPU ID
    slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
    NPU_ID=${npu_array[$slot_idx]}

    # === 固定参数 ===
    SEED=0
    RESULT_DIR="./results/PETSA/"
    BASE_LR=0.001
    WEIGHT_DECAY=0.0001
    LOW_RANK=16
    LOSS_ALPHA=0.1
    GATING_INIT=0.01

    echo "Job slot {%}: NPU=${NPU_ID}  MODEL={1}  DATASET={2}  PRED={3}  TARGET={4}"

    # === 设置 NPU 可见设备 ===
    export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}
    
     python main.py \
        SEED ${SEED} \
        DATA.NAME {2} \
        DATA.DOMAIN_SHIFT_TARGET {4} \
        DATA.PRED_LEN {3} \
        MODEL.NAME {1} \
        MODEL.pred_len {3} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR checkpoints/{1}/{2}_{3}/ \
        TEST.ENABLE False \
        TTA.ENABLE True \
        TTA.DOMAIN_SHIFT False \
        TTA.SOLVER.BASE_LR ${BASE_LR} \
        TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
        TTA.PETSA.GATING_INIT ${GATING_INIT} \
        TTA.PETSA.RANK ${LOW_RANK} \
        TTA.PETSA.LOSS_ALPHA ${LOSS_ALPHA} \
        RESULT_DIR ${RESULT_DIR} \
        TTA.METHOD PETSA
        
' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${TARGETS[@]}"

# #!/bin/bash
# GPUS=(0 5 6 7)
# NUM_GPUS=${#GPUS[@]}

# MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
# # DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
# # MODELS=("iTransformer" "MICN" "OLS" "PatchTST")
# DATASETS=("ETTh2")
# PRED_LENS=(96)
# PRED_LENS=(96 192 336 720)
# DATASETS=("ETTh1")
# TARGETS=("ETTh2")

# parallel -j 8 --delay 0 '
#     GPU=0
#     SEED=0
#     RESULT_DIR="./results/PETSA/"
#     BASE_LR=0.001
#     WEIGHT_DECAY=0.0001
#     LOW_RANK=16
#     LOSS_ALPHA=0.1
#     GATING_INIT=0.01

#     echo "Job {#}: MODEL={1} DATASET={2} PRED={3} -> Running on GPU $GPU"
    
#     CUDA_VISIBLE_DEVICES=$GPU python main.py \
#         SEED ${SEED} \
#         DATA.NAME {2} \
#         DATA.DOMAIN_SHIFT_TARGET {4} \
#         DATA.PRED_LEN {3} \
#         MODEL.NAME {1} \
#         MODEL.pred_len {3} \
#         TRAIN.ENABLE False \
#         TRAIN.CHECKPOINT_DIR checkpoints/{1}/{2}_{3}/ \
#         TEST.ENABLE False \
#         TTA.ENABLE True \
#         TTA.DOMAIN_SHIFT False \
#         TTA.SOLVER.BASE_LR ${BASE_LR} \
#         TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
#         TTA.PETSA.GATING_INIT ${GATING_INIT} \
#         TTA.PETSA.RANK ${LOW_RANK} \
#         TTA.PETSA.LOSS_ALPHA ${LOSS_ALPHA} \
#         RESULT_DIR ${RESULT_DIR} \
#         TTA.METHOD PETSA
        
# ' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${TARGETS[@]}"