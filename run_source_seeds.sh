#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-source_baseline_runs}"
SEEDS="${SEEDS:-0 1 2}"
REWARD_MODE="${REWARD_MODE:-scaled_raw}"
REWARD_SCALE="${REWARD_SCALE:-0.1}"
SURVIVAL_REWARD_PER_FRAME="${SURVIVAL_REWARD_PER_FRAME:-0.001}"
LIFE_LOSS_PENALTY="${LIFE_LOSS_PENALTY:--1.0}"

mkdir -p "${OUTPUT_ROOT}"

for seed in ${SEEDS}; do
  "${PYTHON_BIN}" ocatari_source_ppo.py \
    --env ALE/SpaceInvaders-v5 \
    --object-mode ram \
    --total-steps "${TOTAL_STEPS}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --reward-mode "${REWARD_MODE}" \
    --reward-scale "${REWARD_SCALE}" \
    --survival-reward-per-frame "${SURVIVAL_REWARD_PER_FRAME}" \
    --life-loss-penalty "${LIFE_LOSS_PENALTY}" \
    --slot-strategy temporal \
    --frameskip 4 \
    --rollout-steps 1024 \
    --eval-interval 100000 \
    --eval-episodes 10 \
    --random-eval-episodes 20 \
    --checkpoint-interval 100000 \
    --output-dir "${OUTPUT_ROOT}/seed_${seed}"
done

"${PYTHON_BIN}" aggregate_source_runs.py \
  --root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/aggregate"
