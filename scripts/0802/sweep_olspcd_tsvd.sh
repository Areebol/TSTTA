#!/usr/bin/env bash
set -euo pipefail

# Serial truncated-SVD threshold sweep for deterministic OLSPCD fitting.
# The winning threshold is selected only by validation MSE.

PROJECT_ROOT="${PROJECT_ROOT:-/linyuanping/dzs/codes/TSTTA}"
SWEEP_TAG="${SWEEP_TAG:-0807_olspcd_tsvd_sweep}"
RCONDS_TEXT="${RCONDS:-0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40}"
PRED_LEN="${PRED_LEN:-24}"
TRAIN_ID="${TRAIN_ID:-455}"
TEST_ID="${TEST_ID:-10}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"

read -r -a RCOND_ARRAY <<< "${RCONDS_TEXT}"

SWEEP_RESULT_ROOT="${PROJECT_ROOT}/results/${SWEEP_TAG}"
SWEEP_LOG_ROOT="${PROJECT_ROOT}/logs/${SWEEP_TAG}"
SUMMARY_FILE="${SWEEP_RESULT_ROOT}/summary.tsv"
RANKING_FILE="${SWEEP_RESULT_ROOT}/rcond_ranking.tsv"
BEST_FILE="${SWEEP_RESULT_ROOT}/best_rcond.txt"

cd "${PROJECT_ROOT}"
mkdir -p "${SWEEP_RESULT_ROOT}" "${SWEEP_LOG_ROOT}"

if grep -q "outer_means" models/OLSPCD.py; then
    echo "[ERROR] rejected aligned preprocessing is still present in models/OLSPCD.py" >&2
    exit 1
fi
if ! grep -q 'fit_preprocessing.*legacy_olspcd_instance_norm' models/OLSPCD.py; then
    echo "[ERROR] restored OLSPCD preprocessing marker was not found" >&2
    exit 1
fi

printf "rcond\tmodel\tpred_len\ttrain_id\ttest_id\tcheckpoint\tmetrics\n" > "${SUMMARY_FILE}"

echo "================================================================"
echo "[TSVD SWEEP] rconds=${RCONDS_TEXT}"
echo "[TSVD SWEEP] pred_len=${PRED_LEN}, direction=${TRAIN_ID}->${TEST_ID}"

for RCOND_VALUE in "${RCOND_ARRAY[@]}"; do
    RCOND_TAG="${RCOND_VALUE//./p}"
    RCOND_TAG="${RCOND_TAG//-/m}"
    RUN_TAG="${SWEEP_TAG}/rcond_${RCOND_TAG}"
    CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/${RUN_TAG}/OLSPCD"
    RCOND_RESULT_ROOT="${SWEEP_RESULT_ROOT}/rcond_${RCOND_TAG}"
    RCOND_LOG_ROOT="${SWEEP_LOG_ROOT}/rcond_${RCOND_TAG}"

    echo "================================================================"
    echo "[RCOND] value=${RCOND_VALUE}, tag=${RCOND_TAG}"

    RUN_TAG="${RUN_TAG}" \
    OLSPCD_SOLVER=tsvd \
    OLSPCD_SVD_RCOND="${RCOND_VALUE}" \
    ALPHA=1e-6 \
    PRED_LENS="${PRED_LEN}" \
    DIRECTIONS="${TRAIN_ID}:${TEST_ID}" \
    GPU_ID="${GPU_ID}" \
    BATCH_SIZE="${BATCH_SIZE}" \
        bash scripts/0802/train_olspcd_repro_gpu.sh

    RESULT_ROOT="${RCOND_RESULT_ROOT}" \
    LOG_ROOT="${RCOND_LOG_ROOT}" \
    PRED_LENS="${PRED_LEN}" \
    DIRECTIONS="${TRAIN_ID}:${TEST_ID}" \
    GPU_ID="${GPU_ID}" \
        bash scripts/0802/eval_olspcd_fixed_gpu.sh "${CHECKPOINT_ROOT}"

    RCOND_SUMMARY="${RCOND_RESULT_ROOT}/summary.tsv"
    if [[ ! -s "${RCOND_SUMMARY}" ]]; then
        echo "[ERROR] evaluation summary was not created: ${RCOND_SUMMARY}" >&2
        exit 1
    fi

    while IFS= read -r RESULT_LINE; do
        [[ -n "${RESULT_LINE}" ]] || continue
        printf "%s\t%s\n" "${RCOND_VALUE}" "${RESULT_LINE}" >> "${SUMMARY_FILE}"
    done < <(tail -n +2 "${RCOND_SUMMARY}")
done

python - "${SUMMARY_FILE}" "${RANKING_FILE}" "${BEST_FILE}" <<'PY'
import csv
import re
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
ranking_path = Path(sys.argv[2])
best_path = Path(sys.argv[3])

rows = []
with summary_path.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        metrics = row.get("metrics", "")
        match = re.search(r"val_mse:\s*([0-9.eE+-]+)", metrics)
        if match is None:
            raise SystemExit(
                f"[ERROR] val_mse missing for rcond={row.get('rcond')}: {metrics}"
            )
        row["val_mse"] = float(match.group(1))
        rows.append(row)

if not rows:
    raise SystemExit("[ERROR] TSVD sweep produced no result rows")

rows.sort(key=lambda item: item["val_mse"])
with ranking_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["rank", "rcond", "val_mse", "metrics", "checkpoint"])
    for rank, row in enumerate(rows, start=1):
        writer.writerow(
            [rank, row["rcond"], row["val_mse"], row["metrics"], row["checkpoint"]]
        )

best = rows[0]
best_path.write_text(
    f"rcond={best['rcond']}\n"
    f"val_mse={best['val_mse']}\n"
    f"checkpoint={best['checkpoint']}\n"
    f"metrics={best['metrics']}\n",
    encoding="utf-8",
)

print("================================================================")
print("[RANKING] validation MSE, lower is better")
for rank, row in enumerate(rows, start=1):
    print(f"{rank:2d}. rcond={row['rcond']:<10} val_mse={row['val_mse']:.8f}")
print(f"[BEST] rcond={best['rcond']}, val_mse={best['val_mse']:.8f}")
print(f"[BEST] checkpoint={best['checkpoint']}")
PY

echo "================================================================"
echo "[PASS] TSVD sweep completed"
echo "[PASS] summary=${SUMMARY_FILE}"
echo "[PASS] ranking=${RANKING_FILE}"
echo "[PASS] best=${BEST_FILE}"
