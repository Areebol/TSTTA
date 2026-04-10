#!/bin/bash

############################################
# 实验参数：eVED - COBA (Dual-TTA) PCD 
############################################

# # 参与实验的模型
MODELS=("DLinearPCD") 
# MODELS=("PatchTSTPCD")

# 迁移对设定
PAIRS=("eVED:eVED")

# 迁移 ID 设置：455 预训练 -> 10 测试
# TRAIN_IDS="['455']"
# TEST_IDS="['10']"

TRAIN_IDS="['10']"
TEST_IDS="['455']"

VAL_IDS=${TRAIN_IDS}
TRAIN_IDS_CLEAN=$(echo "${TRAIN_IDS}" | tr -d "[]'\" ")

# COBA 核心超参数
PRED_LENS=(24 48 96 192)
# export OFFLINE_LRS=(1e-1 5e-2 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5)
# export ONLINE_LRS=(1e-1 5e-2 3e-2 1e-2 5e-3 1e-3 5e-4 1e-4 5e-5 1e-5)
export OFFLINE_LRS=(1e-2)
export ONLINE_LRS=(5e-3)
BASE_NUMS=(32)          # Codebook 基向量数量
LAMBDA_ORTHOS=(1e-3 1e-2 1e-1 1.0 10.0)    # 正交约束权重
QUERY_TYPES=(time-CI)

export TRAIN_IDS
export TEST_IDS
export VAL_IDS

# NPU 资源设置
NPUS=(0 1 2 3)          
NNPU=${#NPUS[@]}        
PER_NPU=4            
TOTAL_JOBS=$(( NNPU * PER_NPU ))

NPU_STR="${NPUS[*]}"
export NPU_STR


parallel --lb -j ${TOTAL_JOBS} '
    npu_array=($NPU_STR)
    slot_idx=$(( ({%} - 1) % '"${NNPU}"' ))
    NPU_ID=${npu_array[$slot_idx]}

    SEED=0
    MODEL={1} 
    PAIR={2}
    PRED_LEN={3}
    OFFLINE_LR={4}
    ONLINE_LR={5}
    N_BASES={6}
    QUERY_TYPE={7}
    LAMBDA_ORTHO={8}

    DATASET=$(echo $PAIR | cut -d: -f1)
    TARGET=$(echo $PAIR | cut -d: -f2)

    # 路径根据 0327 训练好的 455 车模型设置
    CHECKPOINT_DIR="./checkpoints/0327/${MODEL}/${DATASET}_${PRED_LEN}_1e-4_ep_30_10_2_455/"
    RESULT_DIR="./results/0331_ablation/10_2_455/LAMBDA_ORTHO/${LAMBDA_ORTHO}/"

    echo "Job slot {%}: NPU=${NPU_ID} | MODEL=${MODEL} | COBA-Online | ${TRAIN_IDS} -> ${TEST_IDS} | PRED_LEN=${PRED_LEN} | LAMBDA_ORTHO=${LAMBDA_ORTHO}"
    export ASCEND_RT_VISIBLE_DEVICES=${NPU_ID}

    python main.py \
        SEED ${SEED} \
        DATA.NAME ${DATASET} \
        DATA.SEQ_LEN ${PRED_LEN} \
        DATA.PRED_LEN ${PRED_LEN} \
        DATA.LABEL_LEN 12 \
        DATA.DOMAIN_SHIFT_TARGET ${TARGET} \
        DATA.MIN_TEST_LEN 300 \
        DATA.TRAIN_VEHICLE_IDS "${TRAIN_IDS}" \
        DATA.VAL_VEHICLE_IDS "${VAL_IDS}" \
        DATA.TEST_VEHICLE_IDS "${TEST_IDS}" \
        MODEL.NAME ${MODEL} \
        MODEL.pred_len ${PRED_LEN} \
        MODEL.seq_len ${PRED_LEN} \
        MODEL.label_len 12 \
        MODEL.patch_len 8 \
        MODEL.stride 4 \
        TRAIN.ENABLE False \
        TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
        TEST.ENABLE False \
        TTA.ENABLE True \
        TTA.DOMAIN_SHIFT True \
        TTA.SOLVER.BASE_LR ${OFFLINE_LR} \
        TTA.DUAL.COBA_ONLINE_LR ${ONLINE_LR} \
        TTA.DUAL.GCM_N_BASES ${N_BASES} \
        TTA.DUAL.LAMBDA_ORTHO ${LAMBDA_ORTHO} \
        TTA.METHOD "COBA" \
        TTA.DUAL.QUERY_TYPE "time-CI" \
        TTA.DUAL.CALI_NAME "CoBA_TF_Adapter" \
        TTA.DUAL.LOSS_NAME "CoBA_Loss" \
        TTA.DUAL.COBA_ONLINE_ENABLED True \
        TTA.DUAL.PRETRAIN_EPOCHS 1 \
        TTA.DUAL.PAAS True \
        TTA.DUAL.ADJUST_PRED True \
        TTA.DUAL.CALI_INPUT_ENABLE False \
        TTA.DUAL.CALI_OUTPUT_ENABLE True \
        RESULT_DIR ${RESULT_DIR}
' ::: "${MODELS[@]}" ::: "${PAIRS[@]}" ::: "${PRED_LENS[@]}" ::: "${OFFLINE_LRS[@]}" ::: "${ONLINE_LRS[@]}" ::: "${BASE_NUMS[@]}" ::: "${QUERY_TYPES[@]}" ::: "${LAMBDA_ORTHOS[@]}"