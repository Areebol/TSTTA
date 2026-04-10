#!/bin/bash

############################################
# 实验参数：CoBA -> Linear offline 训练测试
############################################

# 1. 硬件设置
NPUS=(0 1 2 3)
NNPU=${#NPUS[@]}
PER_NPU=4
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

# 2. 模型与数据集 (原始非PCD模型)
MODELS=("PatchTST" )
# 3. 迁移对设置 (Source:Target)
# PAIRS=("ETTh1:ETTh2")
PAIRS=("ETTh2:ETTh1" )
PRED_LENS=(96 192 336 720)

# 3. Dual-TTA 核心参数 (使用 export 确保在 parallel 子进程中可见)
export OFFLINE_LRS=(1e-2)
export ONLINE_LRS=(1e-2)
export N_BASES=32
export QUERY_TYPE="time-CI"
export LAMBDA_ORTHO=1.0
export SEEDS=(0)

parallel --lb -j ${TOTAL_JOBS} '
    npu_array=($NPU_STR)
    slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
    NPU_ID=${npu_array[$slot_idx]}

    MODEL={1}
    DATASET={2}
    PRED_LEN={3}
    OFF_LR={4}
    ON_LR={5}

    PAIR={2}
    DATASET=$(echo $PAIR | cut -d: -f1)
    TARGET=$(echo $PAIR | cut -d: -f2)

    SEED={6}

    CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}"
    RESULT_DIR="./results/0401/Ablation/full_ETT/"
    mkdir -p "${RESULT_DIR}"

    echo "Job slot {%}: NPU=${NPU_ID} | ${MODEL} | ${DATASET} | PRED=${PRED_LEN} | ON_LR=${ON_LR}"
    export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}

    python main.py \
        SEED ${SEED} \
        DATA.NAME ${DATASET} \
        DATA.PRED_LEN ${PRED_LEN} \
        DATA.DOMAIN_SHIFT_TARGET ${TARGET} \
        MODEL.NAME ${MODEL} \
        MODEL.pred_len ${PRED_LEN} \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
        TEST.ENABLE True \
        TTA.ENABLE True \
        TTA.DOMAIN_SHIFT True \
        TTA.METHOD "COBA" \
        TTA.SOLVER.BASE_LR ${OFF_LR} \
        TTA.DUAL.COBA_ONLINE_LR ${ON_LR} \
        TTA.DUAL.GCM_N_BASES '"${N_BASES}"' \
        TTA.DUAL.QUERY_TYPE '"${QUERY_TYPE}"' \
        TTA.DUAL.LAMBDA_ORTHO '"${LAMBDA_ORTHO}"' \
        TTA.DUAL.CALI_NAME "CoBA_TF_Adapter" \
        TTA.DUAL.LOSS_NAME "CoBA_Loss" \
        TTA.DUAL.COBA_ONLINE_ENABLED True \
        TTA.DUAL.PRETRAIN_EPOCHS 1 \
        TTA.DUAL.PAAS True \
        TTA.DUAL.ADJUST_PRED True \
        TTA.DUAL.CALI_INPUT_ENABLE False \
        TTA.DUAL.CALI_OUTPUT_ENABLE True \
        RESULT_DIR ${RESULT_DIR}

' ::: "${MODELS[@]}" ::: "${PAIRS[@]}" ::: "${PRED_LENS[@]}" ::: "${OFFLINE_LRS[@]}" ::: "${ONLINE_LRS[@]}" ::: "${SEEDS[@]}"