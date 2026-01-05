#!/bin/bash

# GPUS=(0 1 2 3 4 5 6 7)
GPUS=(7)
NGPU=${#GPUS[@]}
GPU_STR="${GPUS[*]}"
export GPU_STR
JOBS_PER_GPU=1
TOTAL_JOBS=$((NGPU * JOBS_PER_GPU))

MODELS=("PatchTST")
# DATASETS=("eVED")
PRED_LENS=(24)
SEQ_LEN=(24)
LABEL_LEN=(12)
PRED_LEN=(24)
patch_len=(8)
stride=(4)

DATASETS=("eVED")
TARGETS=("eVED")

# parallel --lb -j ${TOTAL_JOBS} '
#     gpu_array=($GPU_STR)
#     slot_idx=$(( ({%} - 1) % '"${NGPU}"' ))
#     GPU_ID=${gpu_array[$slot_idx]}

#     SEED=0
#     BASE_LR=0.01
#     WEIGHT_DECAY=0.0001
#     GATING_INIT=0.01
#     RESULT_DIR="./results/TAFAS/"

#     TRAIN_IDS="['455']"
#     TEST_IDS="['10']"
#     VAL_IDS=${TRAIN_IDS}

#     echo "MODEL={1} DATASET={2} PRED={3} -> Running on GPU $GPU_ID (Slot $slot_idx)"
    
#     CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
#         SEED ${SEED} \
#         DATA.NAME {2} \
#         DATA.DOMAIN_SHIFT_TARGET {4} \
#         DATA.PRED_LEN {3} \
#         DATA.SEQ_LEN {3} \
#         DATA.LABEL_LEN 12 \
#         DATA.TRAIN_VEHICLE_IDS "${TRAIN_IDS}" \
#         DATA.VAL_VEHICLE_IDS "${VAL_IDS}" \
#         DATA.TEST_VEHICLE_IDS "${TEST_IDS}" \
#         DATA.MIN_TEST_LEN 300 \
#         MODEL.NAME {1} \
#         MODEL.pred_len {3} \
#         MODEL.label_len 12 \
#         MODEL.seq_len {3} \
#         MODEL.patch_len 8 \
#         MODEL.stride 4 \
#         TRAIN.ENABLE False \
#         TRAIN.CHECKPOINT_DIR checkpoints/{1}/{2}_{3}_1e-3_p8_s4_bs512_ids455 \
#         TEST.ENABLE False \
#         TTA.ENABLE True \
#         TTA.DOMAIN_SHIFT True \
#         TTA.SOLVER.BASE_LR ${BASE_LR} \
#         TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
#         TTA.DUAL.GATING_INIT ${GATING_INIT} \
#         TTA.DUAL.PETSA_LOWRANK 16 \
#         TTA.DUAL.CALI_NAME coba-GCM \
#         TTA.DUAL.LOSS_NAME COBA \
#         TTA.DUAL.GCM_N_BASES 6 \
#         TTA.DUAL.CALI_INPUT_ENABLE False \
#         TTA.DUAL.CALI_OUTPUT_ENABLE True \
#         RESULT_DIR ${RESULT_DIR} \
#         TTA.METHOD Ours-tta
        
# ' ::: "${MODELS[@]}" ::: "${DATASETS[@]}" ::: "${PRED_LENS[@]}" ::: "${TARGETS[@]}"


slot_idx=0
GPU_ID=6

SEED=0
BASE_LR=0.01
WEIGHT_DECAY=0.0001
GATING_INIT=0.01
RESULT_DIR="./results/TAFAS/"
# TRAIN_BATCH_SIZE=256
TRAIN_BATCH_SIZE=64

MODELS="PatchTST"
DATASETS="eVED"
PRED_LENS=24
SEQ_LEN=24
LABEL_LEN=12

TRAIN_IDS="['455']"
TEST_IDS="['10']"
VAL_IDS=${TRAIN_IDS}


TTA_LR=1e-3
STEPS=10
GCM_N_BASES=10
GCM_FEA_DIM=32
ONLINE_TTA=False
PRETRAIN_EPOCHS=5


echo "MODEL=${MODELS} DATASET=${DATASETS} PRED=24 -> Running on GPU $GPU_ID (Slot $slot_idx)"

for PRETRAIN_EPOCHS in 1 5 10; do
for BASE_LR in 1e-2; do
for TRAIN_BATCH_SIZE in 512; do
# for STEPS in 1 10; do
echo "BASE_LR: ${BASE_LR}, PRETRAIN_EPOCHS: ${PRETRAIN_EPOCHS}, GCM_FEA_DIM: ${GCM_FEA_DIM}, GCM_N_BASES: ${GCM_N_BASES}, TTA_LR: ${TTA_LR}, STEPS: ${STEPS}, TRAIN_BATCH_SIZE ${TRAIN_BATCH_SIZE}"

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
    SEED ${SEED} \
    DATA.NAME ${DATASETS} \
    DATA.DOMAIN_SHIFT_TARGET ${DATASETS} \
    DATA.PRED_LEN ${PRED_LENS} \
    DATA.SEQ_LEN ${SEQ_LEN} \
    DATA.LABEL_LEN ${LABEL_LEN} \
    DATA.TRAIN_VEHICLE_IDS "${TRAIN_IDS}" \
    DATA.VAL_VEHICLE_IDS "${VAL_IDS}" \
    DATA.TEST_VEHICLE_IDS "${TEST_IDS}" \
    DATA.MIN_TEST_LEN 300 \
    MODEL.NAME ${MODELS} \
    MODEL.pred_len ${PRED_LENS} \
    MODEL.label_len ${LABEL_LEN} \
    MODEL.seq_len ${SEQ_LEN} \
    MODEL.patch_len 8 \
    MODEL.stride 4 \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR checkpoints/${MODELS}/${DATASETS}_${PRED_LENS}_1e-3_p8_s4_bs512_ids455 \
    TRAIN.BATCH_SIZE ${TRAIN_BATCH_SIZE} \
    TEST.ENABLE False \
    TTA.ENABLE True \
    TTA.DOMAIN_SHIFT True \
    TTA.SOLVER.BASE_LR ${BASE_LR} \
    TTA.SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
    TTA.DUAL.GATING_INIT ${GATING_INIT} \
    TTA.DUAL.PETSA_LOWRANK 16 \
    TTA.DUAL.CALI_NAME coba-GCM \
    TTA.DUAL.LOSS_NAME COBA \
    TTA.DUAL.COBA_ONLINE_ENABLED ${ONLINE_TTA} \
    TTA.DUAL.COBA_ONLINE_LR ${TTA_LR} \
    TTA.DUAL.STEPS ${STEPS} \
    TTA.DUAL.GCM_N_BASES ${GCM_N_BASES} \
    TTA.DUAL.GCM_FEA_DIM ${GCM_FEA_DIM} \
    TTA.DUAL.PRETRAIN_EPOCHS ${PRETRAIN_EPOCHS} \
    TTA.DUAL.CALI_INPUT_ENABLE False \
    TTA.DUAL.CALI_OUTPUT_ENABLE True \
    RESULT_DIR ${RESULT_DIR} \
    TTA.METHOD Ours-tta

done
done
done