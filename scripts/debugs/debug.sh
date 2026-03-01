#!/bin/bash

# --- 配置部分保持不变 ---
NPUS=(4 5 6 7)
NPU_ID=${NPUS[0]} # 调试时默认使用第一个可用的 NPU

MODELS=("PatchTST")
DATASETS=("ETTh1")
PRED_LENS=(720)
PATTERN_NUMS=(64)
OFFLINE_LRS=(0.01)
ONLINE_LRS=(0.01)
LAMBDA_ORTHO=(1.0)
QUERY_TYPES=("time-CI")

# --- 将 parallel 改为嵌套循环 ---

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for PRED_LEN in "${PRED_LENS[@]}"; do
            for OFFLINE_LR in "${OFFLINE_LRS[@]}"; do
                for LAM_ORTHO in "${LAMBDA_ORTHO[@]}"; do
                    for N_PATTERNS in "${PATTERN_NUMS[@]}"; do
                        for QUERY_TYPE in "${QUERY_TYPES[@]}"; do
                            for ONLINE_LR in "${ONLINE_LRS[@]}"; do

                                # 环境配置
                                export CUDA_VISIBLE_DEVICES=${NPU_ID}
                                SEED=0
                                CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${PRED_LEN}"
                                RESULT_DIR="./results/output_tta/"
                                mkdir -p "${RESULT_DIR}"

                                echo "---------------------------------------"
                                echo "Debugging: ${MODEL} | ${DATASET} | Len: ${PRED_LEN}"
                                echo "NPU ID: ${NPU_ID} | Online LR: ${ONLINE_LR}"
                                echo "---------------------------------------"

                                # 直接运行 Python 命令
                                # 如果你使用 VS Code 调试，可以直接在 python 前加 'python -m debugpy --wait-for-client --listen 5678'
                                python main.py \
                                    SEED ${SEED} \
                                    DATA.NAME ${DATASET} \
                                    DATA.PRED_LEN ${PRED_LEN} \
                                    MODEL.NAME ${MODEL} \
                                    MODEL.pred_len ${PRED_LEN} \
                                    TRAIN.ENABLE False \
                                    TRAIN.CHECKPOINT_DIR ${CHECKPOINT_DIR} \
                                    TEST.ENABLE False \
                                    TTA.ENABLE True \
                                    TTA.DOMAIN_SHIFT False \
                                    TTA.METHOD 'PKA_OnLine' \
                                    TTA.PKA.BATCH_SIZE 64 \
                                    TTA.PKA.GATING_INIT 0.01 \
                                    TTA.SOLVER.BASE_LR ${OFFLINE_LR} \
                                    TTA.PKA.PRETRAIN_EPOCHS 1 \
                                    TTA.PKA.PAAS True \
                                    TTA.PKA.ADJUST_PRED True \
                                    TTA.PKA.CALI_NAME PKA_LDict \
                                    TTA.PKA.LOSS_NAME CoBA_Loss \
                                    TTA.PKA.QUERY_TYPE ${QUERY_TYPE} \
                                    TTA.PKA.N_PATTERNS ${N_PATTERNS} \
                                    TTA.PKA.LAMBDA_ORTHO ${LAM_ORTHO} \
                                    TTA.PKA.COBA_ONLINE_LR ${ONLINE_LR} \
                                    TTA.PKA.ENERGY_THRESHOLD 0.1 \
                                    TTA.PKA.CALI_INPUT_ENABLE False \
                                    TTA.PKA.CALI_OUTPUT_ENABLE True \
                                    TTA.PKA.COBA_ONLINE_ENABLED False \
                                    TTA.VISUALIZE False \
                                    RESULT_DIR ${RESULT_DIR}

                            done
                        done
                    done
                done
            done
        done
    done
done