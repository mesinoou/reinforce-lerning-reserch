#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-representation_comparison_runs}"
SEEDS="${SEEDS:-0 1 2}"
REWARD_MODE="${REWARD_MODE:-scaled_raw}"
REWARD_SCALE="${REWARD_SCALE:-0.1}"

mkdir -p "${OUTPUT_ROOT}/objects" "${OUTPUT_ROOT}/pixels"

for seed in ${SEEDS}; do
  for input_mode in objects pixels; do
    mode_root="${OUTPUT_ROOT}/${input_mode}"
    run_dir="${mode_root}/seed_${seed}"
    checkpoint="${run_dir}/checkpoint_latest.pt"
    common_args=(
      --env ALE/SpaceInvaders-v5
      --input-mode "${input_mode}"
      --object-mode ram
      --total-steps "${TOTAL_STEPS}"
      --seed "${seed}"
      --device "${DEVICE}"
      --reward-mode "${REWARD_MODE}"
      --reward-scale "${REWARD_SCALE}"
      --slot-strategy temporal
      --stack-size 4
      --pixel-width 84
      --pixel-height 84
      --frameskip 4
      --rollout-steps 1024
      --ppo-epochs 4
      --minibatch-size 256
      --eval-interval 100000
      --eval-episodes 10
      --random-eval-episodes 20
      --eval-seed 100000
      --checkpoint-interval 100000
      --output-dir "${run_dir}"
    )
    if [[ -f "${checkpoint}" ]]; then
      echo "[resume] ${input_mode} seed=${seed} from ${checkpoint}"
      "${PYTHON_BIN}" ocatari_source_ppo.py \
        "${common_args[@]}" \
        --resume "${checkpoint}"
    else
      "${PYTHON_BIN}" ocatari_source_ppo.py "${common_args[@]}"
    fi
  done
done

for input_mode in objects pixels; do
  mode_root="${OUTPUT_ROOT}/${input_mode}"
  "${PYTHON_BIN}" aggregate_source_runs.py \
    --root "${mode_root}" \
    --output-dir "${mode_root}/aggregate"
done

"${PYTHON_BIN}" compare_representations.py \
  --objects-root "${OUTPUT_ROOT}/objects" \
  --pixels-root "${OUTPUT_ROOT}/pixels" \
  --output-dir "${OUTPUT_ROOT}/comparison"
