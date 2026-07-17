#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-/home/ahhy/venvs/sionna-pi}"
source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"

RUN_ROOT="${RUN_ROOT:-runs/deeprx_11_vs_5_pilot}"
NUM_TRAIN="${NUM_TRAIN:-3000}"
NUM_VAL="${NUM_VAL:-750}"
EPOCHS="${EPOCHS:-10}"
MAX_BLOCKS="${MAX_BLOCKS:-2000}"
TARGET_BLOCK_ERRORS="${TARGET_BLOCK_ERRORS:-50}"

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/eval_ber" "$RUN_ROOT/eval_bler"

for spec in \
  "compact5:deeprx_compact_paper_input:110:5" \
  "paper11:deeprx_paper11:64:11"
do
  IFS=: read -r tag model hidden blocks <<< "$spec"
  save_dir="$RUN_ROOT/checkpoints/$tag"
  checkpoint="$save_dir/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    python -m training.train_sionna \
      --model "$model" \
      --train_phase_mode uniform \
      --val_phase_mode uniform \
      --num_train "$NUM_TRAIN" \
      --num_val "$NUM_VAL" \
      --epochs "$EPOCHS" \
      --batch_size 32 \
      --snr_db_min -10 \
      --snr_db_max 20 \
      --tdl_model A \
      --delay_spread_s 1e-8 \
      --dmrs_freq_spacing 1 \
      --hidden "$hidden" \
      --hidden_complex 2 \
      --zero_complex 2 \
      --branch_layers "$blocks" \
      --kernel_size 3 \
      --lr 1e-3 \
      --seed 0 \
      --save_dir "$save_dir" \
      --device cuda
  fi

  python -m evaluation.eval_ber_sionna \
    --checkpoint "$checkpoint" \
    --phase_mode uniform \
    --snr_list=-6,0,4,8,12 \
    --num_samples 2048 \
    --batch_size 128 \
    --seed 981000 \
    --common_random_numbers \
    --device cuda \
    --out_csv "$RUN_ROOT/eval_ber/$tag.csv"

  python -m evaluation.eval_bler_sionna \
    --checkpoint "$checkpoint" \
    --ebno_list=2,3,4,5,6 \
    --coderate 0.5 \
    --decoder_iterations 20 \
    --phase_mode uniform \
    --batch_size 128 \
    --target_block_errors "$TARGET_BLOCK_ERRORS" \
    --max_blocks "$MAX_BLOCKS" \
    --seed 982000 \
    --common_random_numbers \
    --device cuda \
    --out_csv "$RUN_ROOT/eval_bler/$tag.csv"
done

echo "Completed: $RUN_ROOT"
