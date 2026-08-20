#!/usr/bin/env bash
set -euo pipefail

# Sample-efficiency comparison with an exactly replayed finite corpus.
# Space-separated environment variables can narrow or expand the matrix.
PYTHON_BIN="${PYTHON_BIN:-python}"
PROFILE="${PROFILE:-configs/channel_profiles/tdl_mix_normalized.json}"
TRAIN_SIZES="${TRAIN_SIZES:-500 2000 10000}"
SEEDS="${SEEDS:-0 1 2}"
MODELS="${MODELS:-su_mimo_phase_canonical su_mimo_phase_sensitive}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
NUM_VAL="${NUM_VAL:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VAL_INTERVAL_STEPS="${VAL_INTERVAL_STEPS:-500}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
CONSTANT_TAIL_STEPS="${CONSTANT_TAIL_STEPS:-2000}"
LR="${LR:-0.001}"
LR_MIN="${LR_MIN:-0.00001}"
SNR_DB_MIN="${SNR_DB_MIN:--5}"
SNR_DB_MAX="${SNR_DB_MAX:-20}"
TRAIN_TDL_MODELS="${TRAIN_TDL_MODELS:-A B C D E}"
TRAIN_DELAY_SPREAD_MIN_NS="${TRAIN_DELAY_SPREAD_MIN_NS:-10}"
TRAIN_DELAY_SPREAD_MAX_NS="${TRAIN_DELAY_SPREAD_MAX_NS:-300}"
TRAIN_COMPONENT_IDS="${TRAIN_COMPONENT_IDS:-}"
RUN_ROOT="${RUN_ROOT:-runs/su_mimo_finite_data}"

read -r -a train_sizes <<< "${TRAIN_SIZES}"
read -r -a seeds <<< "${SEEDS}"
read -r -a models <<< "${MODELS}"
read -r -a tdl_models <<< "${TRAIN_TDL_MODELS}"

scenario_args=(
  --train_tdl_models "${tdl_models[@]}"
  --train_delay_spread_min_ns "${TRAIN_DELAY_SPREAD_MIN_NS}"
  --train_delay_spread_max_ns "${TRAIN_DELAY_SPREAD_MAX_NS}"
)
if [[ -n "${TRAIN_COMPONENT_IDS}" ]]; then
  read -r -a component_ids <<< "${TRAIN_COMPONENT_IDS}"
  scenario_args+=(--train_component_ids "${component_ids[@]}")
fi

for num_train in "${train_sizes[@]}"; do
  for seed in "${seeds[@]}"; do
    for model in "${models[@]}"; do
      short_model="${model#su_mimo_}"
      save_dir="${RUN_ROOT}/n${num_train}/${short_model}_seed${seed}"
      echo "Training ${model}: N=${num_train}, seed=${seed}, output=${save_dir}"
      "${PYTHON_BIN}" -m training.train_su_mimo \
        --model "${model}" \
        --num_rx_ant 16 --num_layers 2 \
        --train_dataset_mode fixed \
        --num_train "${num_train}" \
        --train_steps "${TRAIN_STEPS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_val "${NUM_VAL}" \
        --validation_interval_steps "${VAL_INTERVAL_STEPS}" \
        --train_phase_mode uniform --val_phase_mode uniform \
        --snr_db_min "${SNR_DB_MIN}" --snr_db_max "${SNR_DB_MAX}" \
        --train_channel_profile "${PROFILE}" \
        --val_channel_profile "${PROFILE}" \
        "${scenario_args[@]}" \
        --lr "${LR}" --lr_scheduler cosine --lr_min "${LR_MIN}" \
        --warmup_steps "${WARMUP_STEPS}" \
        --constant_tail_steps "${CONSTANT_TAIL_STEPS}" \
        --seed "${seed}" \
        --train_generator_seed "$((700000 + seed))" \
        --val_generator_seed 900000 \
        --deterministic_algorithms \
        --save_dir "${save_dir}"
    done
  done
done
