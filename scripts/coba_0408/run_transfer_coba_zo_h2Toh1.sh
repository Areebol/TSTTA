#!/usr/bin/env bash
set -euo pipefail

# Reproducible ETTh2 -> ETTh1 comparison on physical GPU 2.
# Defaults reproduce the best H=720 plain-SGD ZO setting; every value is overridable.
# CONDA_ENV=tstta bash scripts/coba_0408/run_transfer_coba_zo_h2Toh1.sh all

MODE="${1:-all}"  # base, bp, zo, all
CONDA_ENV="${CONDA_ENV:-tstta}"
GPU_ID="${GPU_ID:-2}"
PRED_LEN="${PRED_LEN:-720}"
MODEL="${MODEL:-PatchTST}"
SEED="${SEED:-0}"
BP_STEPS="${BP_STEPS:-${ONLINE_STEPS:-1}}"
ZO_STEPS="${ZO_STEPS:-${ONLINE_STEPS:-5}}"
BP_LR="${BP_LR:-0.03}"
ZO_LR="${ZO_LR:-0.35}"
ZO_C="${ZO_C:-0.04}"
ZO_DIRECTIONS="${ZO_DIRECTIONS:-16}"
BP_OPTIMIZER="${BP_OPTIMIZER:-adam}"
ZO_OPTIMIZER="${ZO_OPTIMIZER:-${ONLINE_OPTIMIZER:-sgd}}"
SGD_MOMENTUM="${SGD_MOMENTUM:-0.0}"
SGD_NESTEROV="${SGD_NESTEROV:-False}"
SGD_DAMPENING="${SGD_DAMPENING:-0.0}"
ONLINE_WEIGHT_DECAY="${ONLINE_WEIGHT_DECAY:-0.0001}"
ZO_BLOCKWISE="${ZO_BLOCKWISE:-False}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/linyuanping/dzs/data/Electric_vehicles_dataset/checkpoints}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${MODEL}/ETTh2_${PRED_LEN}"
RESULT_ROOT="${RESULT_ROOT:-./results/zo_etth2_to_etth1/${MODEL}_${PRED_LEN}}"

if ! conda run -n "${CONDA_ENV}" python -c "import torch" >/dev/null 2>&1; then
  echo "Conda environment '${CONDA_ENV}' cannot import PyTorch." >&2
  echo "Install the project dependencies there or set CONDA_ENV to a ready environment." >&2
  exit 1
fi

run_base() {
  CUDA_VISIBLE_DEVICES="${GPU_ID}" conda run -n "${CONDA_ENV}" python main.py \
    SEED "${SEED}" \
    DATA.NAME ETTh2 \
    DATA.PRED_LEN "${PRED_LEN}" \
    DATA.DOMAIN_SHIFT_TARGET ETTh1 \
    MODEL.NAME "${MODEL}" \
    MODEL.pred_len "${PRED_LEN}" \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
    TEST.ENABLE True \
    TTA.ENABLE False \
    TTA.DOMAIN_SHIFT True \
    DATA_LOADER.NUM_WORKERS 0 \
    RESULT_DIR "${RESULT_ROOT}/base"
}

run_tta() {
  local update_method="$1"
  local online_steps="${BP_STEPS}"
  local zo_enabled=False
  local online_lr="${BP_LR}"
  local online_optimizer="${BP_OPTIMIZER}"
  local result_name="bp_s${BP_STEPS}_lr${BP_LR}"
  if [[ "${update_method}" == "zo" ]]; then
    zo_enabled=True
    online_steps="${ZO_STEPS}"
    online_lr="${ZO_LR}"
    online_optimizer="${ZO_OPTIMIZER}"
    result_name="zo_k${ZO_DIRECTIONS}_s${ZO_STEPS}_lr${ZO_LR}"
  fi
  if [[ "${online_optimizer}" != "adam" ]]; then
    result_name="${result_name}_${online_optimizer}"
  fi

  CUDA_VISIBLE_DEVICES="${GPU_ID}" conda run -n "${CONDA_ENV}" python main.py \
    SEED "${SEED}" \
    DATA.NAME ETTh2 \
    DATA.PRED_LEN "${PRED_LEN}" \
    DATA.DOMAIN_SHIFT_TARGET ETTh1 \
    MODEL.NAME "${MODEL}" \
    MODEL.pred_len "${PRED_LEN}" \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
    TEST.ENABLE False \
    TTA.ENABLE True \
    TTA.DOMAIN_SHIFT True \
    TTA.METHOD COBA_Vis \
    TTA.DUAL.BATCH_SIZE 64 \
    TTA.DUAL.STEPS "${online_steps}" \
    TTA.DUAL.GATING_INIT 0.01 \
    TTA.DUAL.COBA_ONLINE_OPTIMIZER "${online_optimizer}" \
    TTA.SOLVER.MOMENTUM "${SGD_MOMENTUM}" \
    TTA.SOLVER.NESTEROV "${SGD_NESTEROV}" \
    TTA.SOLVER.DAMPENING "${SGD_DAMPENING}" \
    TTA.SOLVER.WEIGHT_DECAY "${ONLINE_WEIGHT_DECAY}" \
    TTA.SOLVER.BASE_LR 0.01 \
    TTA.DUAL.PRETRAIN_EPOCHS 1 \
    TTA.DUAL.PAAS True \
    TTA.DUAL.ADJUST_PRED True \
    TTA.DUAL.CALI_NAME CoBA_TF_Adapter \
    TTA.DUAL.LOSS_NAME CoBA_Loss \
    TTA.DUAL.QUERY_TYPE time-CI \
    TTA.DUAL.GCM_N_BASES 32 \
    TTA.DUAL.LAMBDA_ORTHO 1.0 \
    TTA.DUAL.COBA_ONLINE_LR "${online_lr}" \
    TTA.DUAL.CALI_INPUT_ENABLE False \
    TTA.DUAL.CALI_OUTPUT_ENABLE True \
    TTA.DUAL.COBA_ONLINE_ENABLED True \
    TTA.ZO.ENABLE "${zo_enabled}" \
    TTA.ZO.PERTURBATION_SCALE "${ZO_C}" \
    TTA.ZO.SP_AVG "${ZO_DIRECTIONS}" \
    TTA.ZO.DISTRIBUTION rademacher \
    TTA.ZO.BLOCKWISE "${ZO_BLOCKWISE}" \
    TTA.ZO.PROFILE_MEMORY True \
    TTA.VISUALIZE False \
    DATA_LOADER.NUM_WORKERS 0 \
    RESULT_DIR "${RESULT_ROOT}/${result_name}"
}

case "${MODE}" in
  base) run_base ;;
  bp) run_tta bp ;;
  zo) run_tta zo ;;
  all)
    run_base
    run_tta bp
    run_tta zo
    ;;
  *)
    echo "Usage: $0 [base|bp|zo|all]" >&2
    exit 2
    ;;
esac
