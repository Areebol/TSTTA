#!/usr/bin/env bash
set -euo pipefail

# Serial alpha sweep for deterministic OLSPCD closed-form fitting.
# Selection is based only on validation MSE; test metrics are reported but are
# not used to choose alpha.
#
# Default smoke sweep:
#   bash scripts/0802/sweep_olspcd_alpha.sh
#
# Custom sweep:
#   ALPHAS="5.0 10.0 15.0 20.0 30.0" \
#   PRED_LEN=24 TRAIN_ID=455 TEST_ID=10 GPU_ID=0 \
#     bash scripts/0802/sweep_olspcd_alpha.sh

PROJECT_ROOT="${PROJECT_ROOT:-/linyuanping/dzs/codes/TSTTA}"
SWEEP_TAG="${SWEEP_TAG:-0805_olspcd_alpha_sweep}"
ALPHAS_TEXT="${ALPHAS:- 1000.0 10000.0 100000.0 1000000.0 10000000.0 100000000.0}"
PRED_LEN="${PRED_LEN:-24}"
TRAIN_ID="${TRAIN_ID:-455}"
TEST_ID="${TEST_ID:-10}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"

read -r -a ALPHA_ARRAY <<< "${ALPHAS_TEXT}"

SWEEP_RESULT_ROOT="${PROJECT_ROOT}/results/${SWEEP_TAG}"
SWEEP_LOG_ROOT="${PROJECT_ROOT}/logs/${SWEEP_TAG}"
SUMMARY_FILE="${SWEEP_RESULT_ROOT}/summary.tsv"
RANKING_FILE="${SWEEP_RESULT_ROOT}/alpha_ranking.tsv"
BEST_FILE="${SWEEP_RESULT_ROOT}/best_alpha.txt"

cd "${PROJECT_ROOT}"
mkdir -p "${SWEEP_RESULT_ROOT}" "${SWEEP_LOG_ROOT}"

if grep -q "outer_means" models/OLSPCD.py; then
    echo "[ERROR] models/OLSPCD.py still contains the rejected aligned-preprocessing experiment." >&2
    echo "        Sync the restored legacy_olspcd_instance_norm version first." >&2
    exit 1
fi

if ! grep -q 'fit_preprocessing.*legacy_olspcd_instance_norm' models/OLSPCD.py; then
    echo "[ERROR] restored OLSPCD preprocessing marker was not found." >&2
    exit 1
fi

printf "alpha\tmodel\tpred_len\ttrain_id\ttest_id\tcheckpoint\tmetrics\n" > "${SUMMARY_FILE}"

echo "================================================================"
echo "[SWEEP] alphas=${ALPHAS_TEXT}"
echo "[SWEEP] pred_len=${PRED_LEN}, direction=${TRAIN_ID}->${TEST_ID}"
echo "[SWEEP] summary=${SUMMARY_FILE}"

for ALPHA_VALUE in "${ALPHA_ARRAY[@]}"; do
    # Keep directory names shell-safe and stable: 10.0 -> 10p0.
    ALPHA_TAG="${ALPHA_VALUE//./p}"
    ALPHA_TAG="${ALPHA_TAG//-/m}"
    RUN_TAG="${SWEEP_TAG}/alpha_${ALPHA_TAG}"
    CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints/${RUN_TAG}/OLSPCD"
    ALPHA_RESULT_ROOT="${SWEEP_RESULT_ROOT}/alpha_${ALPHA_TAG}"
    ALPHA_LOG_ROOT="${SWEEP_LOG_ROOT}/alpha_${ALPHA_TAG}"

    echo "================================================================"
    echo "[ALPHA] value=${ALPHA_VALUE}, tag=${ALPHA_TAG}"

    RUN_TAG="${RUN_TAG}" \
    ALPHA="${ALPHA_VALUE}" \
    PRED_LENS="${PRED_LEN}" \
    DIRECTIONS="${TRAIN_ID}:${TEST_ID}" \
    GPU_ID="${GPU_ID}" \
    BATCH_SIZE="${BATCH_SIZE}" \
        bash scripts/0802/train_olspcd_repro_gpu.sh

    RESULT_ROOT="${ALPHA_RESULT_ROOT}" \
    LOG_ROOT="${ALPHA_LOG_ROOT}" \
    PRED_LENS="${PRED_LEN}" \
    DIRECTIONS="${TRAIN_ID}:${TEST_ID}" \
    GPU_ID="${GPU_ID}" \
        bash scripts/0802/eval_olspcd_fixed_gpu.sh "${CHECKPOINT_ROOT}"

    ALPHA_SUMMARY="${ALPHA_RESULT_ROOT}/summary.tsv"
    if [[ ! -s "${ALPHA_SUMMARY}" ]]; then
        echo "[ERROR] evaluation summary was not created: ${ALPHA_SUMMARY}" >&2
        exit 1
    fi

    while IFS= read -r RESULT_LINE; do
        [[ -n "${RESULT_LINE}" ]] || continue
        printf "%s\t%s\n" "${ALPHA_VALUE}" "${RESULT_LINE}" >> "${SUMMARY_FILE}"
    done < <(tail -n +2 "${ALPHA_SUMMARY}")
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
                f"[ERROR] val_mse was not found for alpha={row.get('alpha')}: {metrics}"
            )
        row["val_mse"] = float(match.group(1))
        rows.append(row)

if not rows:
    raise SystemExit("[ERROR] alpha sweep produced no result rows")

rows.sort(key=lambda item: item["val_mse"])
with ranking_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["rank", "alpha", "val_mse", "metrics", "checkpoint"])
    for rank, row in enumerate(rows, start=1):
        writer.writerow(
            [rank, row["alpha"], row["val_mse"], row["metrics"], row["checkpoint"]]
        )

best = rows[0]
best_path.write_text(
    f"alpha={best['alpha']}\n"
    f"val_mse={best['val_mse']}\n"
    f"checkpoint={best['checkpoint']}\n"
    f"metrics={best['metrics']}\n",
    encoding="utf-8",
)

print("================================================================")
print("[RANKING] validation MSE, lower is better")
for rank, row in enumerate(rows, start=1):
    print(f"{rank:2d}. alpha={row['alpha']:<12} val_mse={row['val_mse']:.8f}")
print(f"[BEST] alpha={best['alpha']}, val_mse={best['val_mse']:.8f}")
print(f"[BEST] checkpoint={best['checkpoint']}")
PY

echo "================================================================"
echo "[PASS] alpha sweep completed"
echo "[PASS] summary=${SUMMARY_FILE}"
echo "[PASS] ranking=${RANKING_FILE}"
echo "[PASS] best=${BEST_FILE}"

