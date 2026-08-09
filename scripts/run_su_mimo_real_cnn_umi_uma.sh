#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
seed="${SEED:-0}"
epochs="${EPOCHS:-130}"
constant_tail_epochs="${CONSTANT_TAIL_EPOCHS:-30}"
rx_list="${RX_LIST:-2 16}"
profile="${PROFILE:-configs/channel_profiles/umi_uma_mix_normalized.json}"
output_root="${OUTPUT_ROOT:-runs/su_mimo_umi_uma_mix_warmup_cosine_tail${constant_tail_epochs}_seed${seed}}"

export PYTHONHASHSEED="$seed"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

for rx in $rx_list; do
  save_dir="$output_root/rx${rx}_real_cnn"
  if [[ -e "$save_dir/history.csv" || -e "$save_dir/best.pt" ]]; then
    echo "Refusing to overwrite an existing training run: $save_dir" >&2
    exit 2
  fi
  mkdir -p "$save_dir"
  echo "Starting parameter-matched real CNN with ${rx}Rx -> $save_dir"

  "$python_bin" -m training.train_su_mimo \
    --model su_mimo_real_cnn \
    --num_layers 2 \
    --num_rx_ant "$rx" \
    --total_tx_power 1.0 \
    --train_dataset_mode streaming \
    --train_channel_profile "$profile" \
    --val_channel_profile "$profile" \
    --train_phase_mode fixed \
    --val_phase_mode fixed \
    --snr_db_min -5 \
    --snr_db_max 20 \
    --num_train 10000 \
    --num_val 2000 \
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

echo "Completed real-CNN UMi/UMa runs under $output_root"
