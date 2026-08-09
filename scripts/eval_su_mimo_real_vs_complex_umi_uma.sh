#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
train_root="${TRAIN_ROOT:-runs/su_mimo_umi_uma_mix_warmup_cosine_tail30_seed0}"
eval_root="${EVAL_ROOT:-runs/su_mimo_real_vs_complex_evaluation_seed777000}"
ebno_list="${EBNO_LIST:-3,5,7,9,11,13}"
batch_size="${BATCH_SIZE:-16}"
target_errors="${TARGET_ERRORS:-500}"
max_blocks="${MAX_BLOCKS:-20000}"
eval_seed="${EVAL_SEED:-777000}"
rx_list="${RX_LIST:-2 16}"
scenario_list="${SCENARIO_LIST:-umi uma}"
phase_list="${PHASE_LIST:-fixed uniform}"

for scenario in $scenario_list; do
  profile="configs/channel_profiles/${scenario}_normalized.json"
  for phase_mode in $phase_list; do
    for rx in $rx_list; do
      for model_label in real_cnn phase_sensitive; do
        checkpoint="$train_root/rx${rx}_${model_label}/best.pt"
        output_dir="$eval_root/$scenario/$phase_mode/rx${rx}_${model_label}"
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
        echo "Evaluating $scenario | $phase_mode | ${rx}Rx | $model_label"

        "$python_bin" -m evaluation.eval_bler_su_mimo \
          --receiver neural \
          --checkpoint "$checkpoint" \
          --ebno_list="$ebno_list" \
          --coderate 0.5 \
          --decoder_iterations 20 \
          --phase_mode "$phase_mode" \
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
  done
done

echo "Completed real-vs-complex evaluation under $eval_root"
