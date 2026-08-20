#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
seed="${SEED:-0}"
epochs="${EPOCHS:-130}"
constant_tail_epochs="${CONSTANT_TAIL_EPOCHS:-30}"
train_seed="${TRAIN_GENERATOR_SEED:-$seed}"
val_seed="${VAL_GENERATOR_SEED:-$((seed + 100000))}"
output_root="${OUTPUT_ROOT:-runs/su_mimo_tdl_mix_phase_canonical_tail${constant_tail_epochs}_seed${seed}}"

export PYTHONHASHSEED="$seed"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

common_args=(
  --model su_mimo_phase_canonical
  --num_layers 2
  --total_tx_power 1.0
  --train_channel_profile configs/channel_profiles/tdl_mix_normalized.json
  --val_channel_profile configs/channel_profiles/tdl_mix_normalized.json
  --train_phase_mode fixed
  --val_phase_mode fixed
  --snr_db_min -5
  --snr_db_max 20
  --num_train 10000
  --num_val 2000
  --epochs "$epochs"
  --batch_size 64
  --hidden_complex 32
  --zero_real 22
  --hidden_real 66
  --num_iterations 2
  --kernel_size 3
  --zero_gate_hidden 16
  --lr 1e-3
  --lr_scheduler cosine
  --lr_min 1e-5
  --warmup_epochs 5
  --constant_tail_epochs "$constant_tail_epochs"
  --weight_decay 0
  --seed "$seed"
  --train_generator_seed "$train_seed"
  --val_generator_seed "$val_seed"
  --deterministic_algorithms
  --log_interval 50
  --device cuda
)

run_one() {
  local num_rx_ant="$1"
  local run_name="rx${num_rx_ant}_phase_canonical"
  local save_dir="$output_root/$run_name"
  if [[ -e "$save_dir/history.csv" || -e "$save_dir/best.pt" ]]; then
    echo "Refusing to overwrite an existing training run: $save_dir" >&2
    return 2
  fi
  mkdir -p "$save_dir"
  echo "Starting $run_name -> $save_dir"
  "$python_bin" -m training.train_su_mimo \
    "${common_args[@]}" \
    --num_rx_ant "$num_rx_ant" \
    --save_dir "$save_dir" \
    2>&1 | tee "$save_dir/train.log"
}

run_one 16
run_one 2

echo "Completed canonical-phase SU-MIMO runs under $output_root"
