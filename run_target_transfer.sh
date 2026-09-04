#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?Set SOURCE_CHECKPOINT to the learned source model.pt}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-target_transfer_runs}"
SEEDS="${SEEDS:-0 1 2}"
ALPHAS="${ALPHAS:-0 0.5}"

for seed in ${SEEDS}; do
  for alpha in ${ALPHAS}; do
    run_dir="${OUTPUT_ROOT}/alpha_${alpha}/seed_${seed}"
    checkpoint="${run_dir}/checkpoint_latest.pt"
    args=(
      --source-checkpoint "${SOURCE_CHECKPOINT}"
      --env ALE/Galaxian-v5
      --transfer-alpha "${alpha}"
      --total-steps "${TOTAL_STEPS}"
      --seed "${seed}"
      --device "${DEVICE}"
      --eval-interval 100000 --eval-episodes 10
      --random-eval-episodes 20 --eval-seed 100000
      --checkpoint-interval 100000
      --progress-interval-updates 10 --progress-interval-seconds 60
      --output-dir "${run_dir}"
    )
    if [[ -f "${checkpoint}" ]]; then
      echo "[resume] alpha=${alpha} seed=${seed}"
      args+=(--resume "${checkpoint}")
    fi
    "${PYTHON_BIN}" -u ocatari_target_transfer.py "${args[@]}"
  done
done

"${PYTHON_BIN}" compare_transfer_runs.py --root "${OUTPUT_ROOT}"
