#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
device="${DEVICE:-cuda}"
run_root="${RUN_ROOT:-runs/su_mimo_lmmse_umi_uma_matrix_v1}"
ebno_list="${EBNO_LIST:-3,5,7,9,11,13}"
batch_size="${BATCH_SIZE:-16}"
target_errors="${TARGET_ERRORS:-500}"
max_blocks="${MAX_BLOCKS:-20000}"
eval_seed="${EVAL_SEED:-777000}"
skip_completed="${SKIP_COMPLETED:-1}"
checkpoint_root="checkpoints/generalization/umi_uma_mix_normalized/su_mimo_phase_sensitive_seed0"
contract="configs/experiment_matrices/su_mimo_lmmse_umi_uma_matrix_v1.json"

spatial_ids=(
  layer2_rx2 layer2_rx4 layer2_rx8 layer2_rx16
  layer4_rx4 layer4_rx8 layer4_rx16
)
checkpoint_names=(
  su_mimo_phase_sensitive_layer2_rx2_seed0.pt
  su_mimo_phase_sensitive_layer2_rx4_seed0.pt
  su_mimo_phase_sensitive_layer2_rx8_seed0.pt
  su_mimo_phase_sensitive_rx16_seed0.pt
  su_mimo_phase_sensitive_layer4_rx4_seed0.pt
  su_mimo_phase_sensitive_layer4_rx8_seed0.pt
  su_mimo_phase_sensitive_layer4_rx16_seed0.pt
)
profiles=(umi_normalized uma_normalized)
receivers=(lmmse_ls lmmse_perfect)

mkdir -p "$run_root/manifest"
cp "$contract" "$run_root/manifest/experiment_contract.json"
{
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "git_status_begin"
  git status --short 2>/dev/null || true
  echo "git_status_end"
  echo "python=$($python_bin --version 2>&1)"
  echo "device=$device"
  echo "ebno_list=$ebno_list"
  echo "batch_size=$batch_size"
  echo "target_errors=$target_errors"
  echo "max_blocks=$max_blocks"
  echo "eval_seed=$eval_seed"
} > "$run_root/manifest/environment.txt"

printf "spatial_id\tcheckpoint\tsha256\n" > "$run_root/manifest/checkpoints.tsv"
for index in "${!spatial_ids[@]}"; do
  checkpoint="$checkpoint_root/${checkpoint_names[$index]}"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 2
  fi
  printf "%s\t%s\t%s\n" \
    "${spatial_ids[$index]}" "$checkpoint" "$(sha256sum "$checkpoint" | cut -d' ' -f1)" \
    >> "$run_root/manifest/checkpoints.tsv"
done

for index in "${!spatial_ids[@]}"; do
  spatial_id="${spatial_ids[$index]}"
  checkpoint="$checkpoint_root/${checkpoint_names[$index]}"
  for profile_id in "${profiles[@]}"; do
    profile="configs/channel_profiles/${profile_id}.json"
    for receiver in "${receivers[@]}"; do
      output_dir="$run_root/$spatial_id/$profile_id/$receiver"
      output_csv="$output_dir/bler.csv"
      layer_csv="$output_dir/bler_per_layer.csv"
      if [[ -s "$output_csv" && -s "$layer_csv" && "$skip_completed" == "1" ]]; then
        echo "Skipping completed cell: $spatial_id | $profile_id | $receiver"
        continue
      fi
      if [[ -e "$output_csv" || -e "$layer_csv" ]]; then
        echo "Refusing to overwrite partial/existing cell: $output_dir" >&2
        exit 2
      fi
      mkdir -p "$output_dir"
      printf "%q " "$python_bin" -m evaluation.eval_bler_su_mimo \
        --receiver "$receiver" --checkpoint "$checkpoint" \
        --ebno_list="$ebno_list" --coderate 0.5 --decoder_iterations 20 \
        --cn_update boxplus-phi --phase_mode uniform --batch_size "$batch_size" \
        --target_block_errors "$target_errors" --max_blocks "$max_blocks" \
        --seed "$eval_seed" --common_random_numbers --device "$device" \
        --eval_channel_profile "$profile" --out_csv "$output_csv" \
        --out_layer_csv "$layer_csv" > "$output_dir/command.txt"
      printf "\n" >> "$output_dir/command.txt"
      echo "Evaluating $spatial_id | $profile_id | $receiver"
      "$python_bin" -m evaluation.eval_bler_su_mimo \
        --receiver "$receiver" \
        --checkpoint "$checkpoint" \
        --ebno_list="$ebno_list" \
        --coderate 0.5 \
        --decoder_iterations 20 \
        --cn_update boxplus-phi \
        --phase_mode uniform \
        --batch_size "$batch_size" \
        --target_block_errors "$target_errors" \
        --max_blocks "$max_blocks" \
        --seed "$eval_seed" \
        --common_random_numbers \
        --device "$device" \
        --eval_channel_profile "$profile" \
        --out_csv "$output_csv" \
        --out_layer_csv "$layer_csv" \
        2>&1 | tee "$output_dir/eval.log"
    done
  done
done

"$python_bin" -m evaluation.aggregate_su_mimo_lmmse_matrix \
  --run_root "$run_root" --require_complete
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_root/manifest/completed_utc.txt"
echo "Completed SU-MIMO LMMSE matrix under $run_root"
