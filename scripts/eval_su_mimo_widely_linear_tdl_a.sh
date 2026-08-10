#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
train_seed="${TRAIN_SEED:-0}"
eval_seed="${EVAL_SEED:-777000}"
rx_list="${RX_LIST:-2 16}"
models="${MODELS:-standard_complex widely_linear_complex real_cnn}"
profile="${PROFILE:-configs/channel_profiles/tdl_a_10ns_normalized.json}"
train_root="${TRAIN_ROOT:-runs/su_mimo_widely_linear_tdl_a_v1/seed${train_seed}/training}"
eval_root="${EVAL_ROOT:-runs/su_mimo_widely_linear_tdl_a_v1/seed${train_seed}/evaluation_seed${eval_seed}}"
ebno_list_rx2="${EBNO_LIST_RX2:-5,7,9}"
ebno_list_rx16="${EBNO_LIST_RX16:--7,-5,-3}"
batch_size="${BATCH_SIZE:-16}"
target_errors="${TARGET_ERRORS:-100}"
max_blocks="${MAX_BLOCKS:-5000}"

for rx in $rx_list; do
  if [[ "$rx" == "2" ]]; then
    ebno_list="$ebno_list_rx2"
  elif [[ "$rx" == "16" ]]; then
    ebno_list="$ebno_list_rx16"
  else
    echo "Set an explicit supported RX_LIST (2 and/or 16); got $rx." >&2
    exit 2
  fi

  for label in $models; do
    checkpoint="$train_root/rx${rx}_${label}/best.pt"
    output_dir="$eval_root/rx${rx}_${label}"
    output_csv="$output_dir/bler.csv"
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing checkpoint: $checkpoint" >&2
      exit 2
    fi
    if [[ -e "$output_csv" ]]; then
      echo "Refusing to overwrite an existing evaluation: $output_csv" >&2
      exit 2
    fi
    mkdir -p "$output_dir"
    echo "Evaluating TDL-A uniform phase | ${rx}Rx | $label | Eb/N0 $ebno_list"

    "$python_bin" -m evaluation.eval_bler_su_mimo \
      --receiver neural \
      --checkpoint "$checkpoint" \
      --ebno_list="$ebno_list" \
      --coderate 0.5 \
      --decoder_iterations 20 \
      --phase_mode uniform \
      --batch_size "$batch_size" \
      --target_block_errors "$target_errors" \
      --max_blocks "$max_blocks" \
      --seed "$eval_seed" \
      --common_random_numbers \
      --eval_channel_profile "$profile" \
      --out_csv "$output_csv" \
      2>&1 | tee "$output_dir/eval.log"
  done
done

echo "Completed TDL-A widely-linear screen evaluation under $eval_root"
