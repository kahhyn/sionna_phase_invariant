#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
seed="${SEED:-0}"
epochs="${EPOCHS:-130}"
num_train="${NUM_TRAIN:-10000}"
num_val="${NUM_VAL:-2000}"
constant_tail_epochs="${CONSTANT_TAIL_EPOCHS:-30}"
rx_list="${RX_LIST:-2 16}"
models="${MODELS:-su_mimo_phase_sensitive su_mimo_widely_linear su_mimo_real_cnn}"
profile="${PROFILE:-configs/channel_profiles/tdl_a_10ns_normalized.json}"
output_root="${OUTPUT_ROOT:-runs/su_mimo_widely_linear_tdl_a_v1/seed${seed}/training}"

export PYTHONHASHSEED="$seed"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

label_for_model() {
  case "$1" in
    su_mimo_phase_sensitive) echo "standard_complex" ;;
    su_mimo_widely_linear) echo "widely_linear_complex" ;;
    su_mimo_real_cnn) echo "real_cnn" ;;
    *) echo "Unsupported model: $1" >&2; return 2 ;;
  esac
}

for rx in $rx_list; do
  for model in $models; do
    label="$(label_for_model "$model")"
    save_dir="$output_root/rx${rx}_${label}"
    if [[ -e "$save_dir/history.csv" || -e "$save_dir/best.pt" ]]; then
      echo "Refusing to overwrite an existing training run: $save_dir" >&2
      exit 2
    fi
    mkdir -p "$save_dir"
    echo "Training TDL-A streaming | ${rx}Rx | $model -> $save_dir"

    "$python_bin" -m training.train_su_mimo \
      --model "$model" \
      --num_layers 2 \
      --num_rx_ant "$rx" \
      --total_tx_power 1.0 \
      --train_dataset_mode streaming \
      --train_channel_profile "$profile" \
      --val_channel_profile "$profile" \
      --train_phase_mode uniform \
      --val_phase_mode uniform \
      --snr_db_min -5 \
      --snr_db_max 20 \
      --num_train "$num_train" \
      --num_val "$num_val" \
      --epochs "$epochs" \
      --batch_size 64 \
      --hidden_complex 32 \
      --zero_real 22 \
      --hidden_real 66 \
      --num_iterations 2 \
      --kernel_size 3 \
      --zero_gate_hidden 16 \
      --lr 1e-3 \
      --lr_scheduler cosine \
      --lr_min 1e-5 \
      --warmup_epochs 5 \
      --constant_tail_epochs "$constant_tail_epochs" \
      --weight_decay 0 \
      --seed "$seed" \
      --train_generator_seed "$seed" \
      --val_generator_seed "$((seed + 100000))" \
      --deterministic_algorithms \
      --log_interval 50 \
      --device cuda \
      --save_dir "$save_dir" \
      2>&1 | tee "$save_dir/train.log"
  done
done

echo "Completed TDL-A widely-linear screen training under $output_root"
