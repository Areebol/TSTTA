#!/bin/bash

############################################
# 实验参数：ETT 数据集 Dual-TTA 迁移
############################################

# 1. 硬件设置 (根据你的 NPU 数量调整)
NPUS=(0 1 2 3 )
NNPU=${#NPUS[@]}
PER_NPU=1
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR

# 2. 模型设置 (原始非PCD模型)
MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")

# 3. 迁移对设置 (Source:Target)
PAIRS=("ETTh1:ETTh2" "ETTh2:ETTh1" "ETTm1:ETTm2" "ETTm2:ETTm1")

# 4. 预测长度
PRED_LENS=(96 192 336 720)

# 5. Dual-TTA 核心超参数
export OFFLINE_LRS=(1e-2 3e-2 5e-2 1e-3 3e-4 1e-4 1e-5 )      # 预训练适配阶段的学习率
export ONLINE_LRS=(1e-3 3e-3 5e-3 1e-4 3e-4 5e-4 1e-5)       # 在线流式更新阶段的学习率
export N_BASES=32
export QUERY_TYPE="time-CI"
export LAMBDA_ORTHO=1.0

# 6. 并行执行逻辑
parallel --lb -j ${TOTAL_JOBS} '
    npu_array=($NPU_STR)
    slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
    NPU_ID=${npu_array[$slot_idx]}

    MODEL={1}
    
    # 解析迁移对
    PAIR={2}
    DATASET=$(echo $PAIR | cut -d: -f1)
    TARGET=$(echo $PAIR | cut -d: -f2)
    
    PRED_LEN={3}
    OFFLINE_LR={4}
    ONLINE_LR={5}

    SEED=1

    # 路径定义
    CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}/"
    RESULT_DIR="./results/Dual_ETT_Transfer/${PAIR}/${MODEL}/"
    mkdir -p "${RESULT_DIR}"

    echo "Job slot {%}: NPU=${NPU_ID} | ${MODEL} | ${DATASET} -> ${TARGET} | PRED=${PRED_LEN}"
    
    # 华为 NPU 环境变量
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
        TEST.ENABLE False \
        TTA.ENABLE True \
        TTA.DOMAIN_SHIFT True \
        TTA.METHOD 'COBA' \
        TTA.DUAL.BATCH_SIZE 64 \
        TTA.DUAL.GATING_INIT 0.01 \
        TTA.SOLVER.BASE_LR ${OFFLINE_LR} \
        TTA.DUAL.COBA_ONLINE_LR ${ONLINE_LR} \
        TTA.DUAL.GCM_N_BASES ${N_BASES} \
        TTA.DUAL.QUERY_TYPE ${QUERY_TYPE} \
        TTA.DUAL.LAMBDA_ORTHO ${LAMBDA_ORTHO} \
        TTA.DUAL.CALI_NAME CoBA_TF_Adapter \
        TTA.DUAL.LOSS_NAME CoBA_Loss \
        TTA.DUAL.COBA_ONLINE_ENABLED True \
        TTA.DUAL.PRETRAIN_EPOCHS 1 \
        TTA.DUAL.PAAS True \
        TTA.DUAL.ADJUST_PRED True \
        TTA.DUAL.CALI_INPUT_ENABLE False \
        TTA.DUAL.CALI_OUTPUT_ENABLE True \
        RESULT_DIR ${RESULT_DIR}

' ::: "${MODELS[@]}" ::: "${PAIRS[@]}" ::: "${PRED_LENS[@]}" ::: "${OFFLINE_LRS[@]}" ::: "${ONLINE_LRS[@]}"