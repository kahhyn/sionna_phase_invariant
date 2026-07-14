#!/usr/bin/env bash
set -euo pipefail

TRAIN_PROFILE=${TRAIN_PROFILE:-configs/channel_profiles/tdl_mix_normalized.json}
TEST_SUITE=${TEST_SUITE:-configs/channel_suites/generalization_normalized.json}
MODELS=${MODELS:-"single_branch_n0_gate strict_matched_complex_p_n0_gate"}
TRAIN_SEEDS=${TRAIN_SEEDS:-"0"}
EVAL_SEEDS=${EVAL_SEEDS:-"777000"}
EPOCHS=${EPOCHS:-50}
NUM_TRAIN=${NUM_TRAIN:-10000}
NUM_VAL=${NUM_VAL:-2000}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_EVAL=${NUM_EVAL:-4096}
SNR_LIST=${SNR_LIST:-"-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20"}
EBNO_LIST=${EBNO_LIST:-"2,2.5,3,3.5,4,4.5,5,5.5,6"}
TARGET_BLOCK_ERRORS=${TARGET_BLOCK_ERRORS:-100}
MAX_BLOCKS=${MAX_BLOCKS:-10000}
RUN_ROOT=${RUN_ROOT:-runs/channel_generalization}
RUN_BER=${RUN_BER:-1}
RUN_BLER=${RUN_BLER:-0}
INCLUDE_LMMSE=${INCLUDE_LMMSE:-0}
INCLUDE_LMMSE_PERFECT=${INCLUDE_LMMSE_PERFECT:-0}
SKIP_TRAINED=${SKIP_TRAINED:-1}
SKIP_EVALUATED=${SKIP_EVALUATED:-1}
DEVICE=${DEVICE:-cuda}

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/eval_ber" \
  "$RUN_ROOT/eval_bler" "$RUN_ROOT/manifests"

{
  echo "git_commit=$(git rev-parse HEAD)"
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "train_profile=$TRAIN_PROFILE"
  echo "test_suite=$TEST_SUITE"
  git status --short
} > "$RUN_ROOT/manifests/run_environment.txt"

python -m experiments.channel_generalization.suite_entries \
  --suite "$TEST_SUITE" > "$RUN_ROOT/manifests/suite_entries.tsv"

read -r -a model_array <<< "$MODELS"
read -r -a train_seed_array <<< "$TRAIN_SEEDS"
read -r -a eval_seed_array <<< "$EVAL_SEEDS"

for model in "${model_array[@]}"; do
  for train_seed in "${train_seed_array[@]}"; do
    checkpoint_dir="$RUN_ROOT/checkpoints/$model/seed_${train_seed}"
    checkpoint="$checkpoint_dir/best.pt"
    if [[ ! -f "$checkpoint" || "$SKIP_TRAINED" != "1" ]]; then
      mkdir -p "$checkpoint_dir"
      train_command=(
        python -m training.train_sionna
        --model "$model"
        --train_channel_profile "$TRAIN_PROFILE"
        --val_channel_profile "$TRAIN_PROFILE"
        --hidden 64
        --hidden_complex 32
        --zero_complex 32
        --branch_layers 2
        --epochs "$EPOCHS"
        --num_train "$NUM_TRAIN"
        --num_val "$NUM_VAL"
        --batch_size "$BATCH_SIZE"
        --snr_db_min -10
        --snr_db_max 20
        --seed "$train_seed"
        --device "$DEVICE"
        --save_dir "$checkpoint_dir"
      )
      printf '%q ' "${train_command[@]}" >> "$RUN_ROOT/manifests/commands.log"
      printf '\n' >> "$RUN_ROOT/manifests/commands.log"
      "${train_command[@]}"
    else
      echo "Skip existing checkpoint: $checkpoint"
    fi

    while IFS=$'\t' read -r test_id profile component_id; do
      component_args=()
      if [[ -n "$component_id" ]]; then
        component_args=(--eval_component_id "$component_id")
      fi
      for eval_seed in "${eval_seed_array[@]}"; do
        if [[ "$RUN_BER" == "1" ]]; then
          ber_out="$RUN_ROOT/eval_ber/$model/seed_${train_seed}/${test_id}/eval_${eval_seed}.csv"
          if [[ ! -f "$ber_out" || "$SKIP_EVALUATED" != "1" ]]; then
            mkdir -p "$(dirname "$ber_out")"
            python -m evaluation.eval_ber_sionna \
              --checkpoint "$checkpoint" \
              --eval_channel_profile "$profile" \
              "${component_args[@]}" \
              --phase_mode uniform \
              --snr_list="$SNR_LIST" \
              --num_samples "$NUM_EVAL" \
              --batch_size "$BATCH_SIZE" \
              --seed "$eval_seed" \
              --common_random_numbers \
              --device "$DEVICE" \
              --out_csv "$ber_out"
          else
            echo "Skip existing BER CSV: $ber_out"
          fi
        fi

        if [[ "$RUN_BLER" == "1" ]]; then
          bler_out="$RUN_ROOT/eval_bler/$model/seed_${train_seed}/${test_id}/eval_${eval_seed}.csv"
          if [[ ! -f "$bler_out" || "$SKIP_EVALUATED" != "1" ]]; then
            mkdir -p "$(dirname "$bler_out")"
            python -m evaluation.eval_bler_sionna \
              --receiver neural \
              --checkpoint "$checkpoint" \
              --eval_channel_profile "$profile" \
              "${component_args[@]}" \
              --phase_mode uniform \
              --ebno_list="$EBNO_LIST" \
              --batch_size "$BATCH_SIZE" \
              --target_block_errors "$TARGET_BLOCK_ERRORS" \
              --max_blocks "$MAX_BLOCKS" \
              --seed "$eval_seed" \
              --common_random_numbers \
              --device "$DEVICE" \
              --out_csv "$bler_out"
          else
            echo "Skip existing BLER CSV: $bler_out"
          fi
        fi

        if [[ "$RUN_BLER" == "1" && "$INCLUDE_LMMSE" == "1" ]]; then
          lmmse_out="$RUN_ROOT/eval_bler/lmmse_ls/${test_id}/eval_${eval_seed}.csv"
          if [[ ! -f "$lmmse_out" || "$SKIP_EVALUATED" != "1" ]]; then
            mkdir -p "$(dirname "$lmmse_out")"
            python -m evaluation.eval_bler_sionna \
              --receiver lmmse_ls --checkpoint "$checkpoint" \
              --eval_channel_profile "$profile" "${component_args[@]}" \
              --phase_mode uniform --ebno_list="$EBNO_LIST" \
              --batch_size "$BATCH_SIZE" \
              --target_block_errors "$TARGET_BLOCK_ERRORS" \
              --max_blocks "$MAX_BLOCKS" --seed "$eval_seed" \
              --common_random_numbers --device "$DEVICE" --out_csv "$lmmse_out"
          fi
        fi

        if [[ "$RUN_BLER" == "1" && "$INCLUDE_LMMSE_PERFECT" == "1" ]]; then
          perfect_out="$RUN_ROOT/eval_bler/lmmse_perfect/${test_id}/eval_${eval_seed}.csv"
          if [[ ! -f "$perfect_out" || "$SKIP_EVALUATED" != "1" ]]; then
            mkdir -p "$(dirname "$perfect_out")"
            python -m evaluation.eval_bler_sionna \
              --receiver lmmse_perfect --checkpoint "$checkpoint" \
              --eval_channel_profile "$profile" "${component_args[@]}" \
              --phase_mode uniform --ebno_list="$EBNO_LIST" \
              --batch_size "$BATCH_SIZE" \
              --target_block_errors "$TARGET_BLOCK_ERRORS" \
              --max_blocks "$MAX_BLOCKS" --seed "$eval_seed" \
              --common_random_numbers --device "$DEVICE" --out_csv "$perfect_out"
          fi
        fi
      done
    done < "$RUN_ROOT/manifests/suite_entries.tsv"
  done
done

if [[ "$RUN_BER" == "1" ]]; then
  python -m experiments.channel_generalization.aggregate_generalization \
    --input_root "$RUN_ROOT/eval_ber" --metric ber \
    --out_csv "$RUN_ROOT/ber_generalization_matrix.csv"
fi
if [[ "$RUN_BLER" == "1" ]]; then
  python -m experiments.channel_generalization.aggregate_generalization \
    --input_root "$RUN_ROOT/eval_bler" --metric bler \
    --out_csv "$RUN_ROOT/bler_generalization_matrix.csv"
fi
