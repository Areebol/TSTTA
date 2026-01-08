#!/bin/bash

NGPU=${#GPUS[@]}
GPU_STR="${GPUS[*]}"
export GPU_STR
JOBS_PER_GPU=4
TOTAL_JOBS=$((NGPU * JOBS_PER_GPU))

MODELS=("DLinear" "FreTS" "iTransformer" "MICN" "OLS" "PatchTST")
DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "exchange_rate" "weather")
PRED_LENS=(96 192 336 720)

MODELS=("DLinear")
DATASETS=("ETTh1")
PRED_LENS=(336)

SEED=1
BASE_LR=0.001
WEIGHT_DECAY=0.0001
GATING_INIT=0.01
RESULT_DIR="./results/TAFAS/"

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python main.py \
            SEED ${SEED} \
            DATA.NAME ${DATASET} \
            DATA.PRED_LEN ${PRED_LEN} \
            MODEL.NAME ${MODEL} \
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
            TTA.SOLVER.BASE_LR 1e-3 \
            TTA.DUAL.GCM_N_BASES 6 \
            TTA.DUAL.LOWRANK_RANKS 8 \
            TTA.DUAL.COBA_ONLINE_ENABLED False \
            TTA.DUAL.COBA_ONLINE_LR 1e-3 \
            TTA.DUAL.PRETRAIN_EPOCHS 4 \
            TRAIN.BATCH_SIZE 512 \
            TTA.METHOD Ours-tta
done
done
done

# parallel --lb -j ${TOTAL_JOBS} '
#     gpu_array=($GPU_STR)
#     slot_idx=$(( ({%} - 1) % '"${NGPU}"' ))
#     GPU_ID=${gpu_array[$slot_idx]}

#     SEED=1
#     BASE_LR=0.001
#     WEIGHT_DECAY=0.0001
#     GATING_INIT=0.01
#     RESULT_DIR="./results/TAFAS/"

#     echo "MODEL={1} DATASET={2} PRED={3} -> Running on GPU $GPU_ID (Slot $slot_idx)"
    
#     CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
#         SEED ${SEED} \
#         DATA.NAME {2} \
#         DATA.PRED_LEN {3} \
#         MODEL.NAME {1} \
#         MODEL.pred_len {3} \
#         TRAIN.ENABLE False \
#         TRAIN.CHECKPOINT_DIR checkpoints/{1}/{2}_{3}/ \
#         TEST.ENABLE False \
#         TTA.ENABLE True \
#         TTA.DOMAIN_SHIFT False \
#         TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
#         TTA.DUAL.GATING_INIT ${GATING_INIT} \
#         TTA.DUAL.PETSA_LOWRANK 16 \
#         TTA.DUAL.CALI_NAME coba-GCM \
#         TTA.DUAL.LOSS_NAME COBA \
#         TTA.DUAL.CALI_INPUT_ENABLE False \
#         TTA.DUAL.CALI_OUTPUT_ENABLE True \
#         TTA.DUAL.ADJUST_PRED True \
#         RESULT_DIR ${RESULT_DIR} \
#         TTA.SOLVER.BASE_LR 1e-3 \
#         TTA.DUAL.GCM_N_BASES 6 \
#         TTA.DUAL.COBA_ONLINE_ENABLED False \
#         TTA.DUAL.COBA_ONLINE_LR 1e-4 \
#         TTA.DUAL.PRETRAIN_EPOCHS 4 \
#         TRAIN.BATCH_SIZE 512 \
#         TTA.METHOD Ours-tta
        
# ' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${TARGETS[@]}"

# python build_table.py
        # TTA.DUAL.CALI_NAME coba-GCM \