#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-/home/ahhy/venvs/sionna-pi}"
source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"

RUN_ROOT="${RUN_ROOT:-runs/deeprx_ac_smoke}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0}"
MODELS="${MODELS:-deeprx deeprx_invariant_a deeprx_matched_c}"
NUM_TRAIN="${NUM_TRAIN:-4000}"
NUM_VAL="${NUM_VAL:-1000}"
EPOCHS="${EPOCHS:-15}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EVAL_SAMPLES="${EVAL_SAMPLES:-4096}"
SNR_LIST="${SNR_LIST:--10,-6,0,4,8,12}"
EVAL_SEED="${EVAL_SEED:-970000}"
HIDDEN="${HIDDEN:-64}"
ADAPTER_COMPLEX="${ADAPTER_COMPLEX:-2}"
ZERO_COMPLEX="${ZERO_COMPLEX:-2}"
NUM_BLOCKS="${NUM_BLOCKS:-5}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/eval"
CSV_FILES=()

for seed in $TRAIN_SEEDS; do
    for model in $MODELS; do
        save_dir="$RUN_ROOT/checkpoints/${model}_seed${seed}"
        checkpoint="$save_dir/best.pt"
        if [[ ! -f "$checkpoint" ]]; then
            python -m training.train_sionna \
                --model "$model" \
                --train_phase_mode fixed \
                --val_phase_mode uniform \
                --num_train "$NUM_TRAIN" \
                --num_val "$NUM_VAL" \
                --epochs "$EPOCHS" \
                --batch_size "$TRAIN_BATCH_SIZE" \
                --snr_db_min -10 \
                --snr_db_max 20 \
                --tdl_model A \
                --delay_spread_s 1e-8 \
                --dmrs_freq_spacing 1 \
                --hidden "$HIDDEN" \
                --hidden_complex "$ADAPTER_COMPLEX" \
                --zero_complex "$ZERO_COMPLEX" \
                --branch_layers "$NUM_BLOCKS" \
                --kernel_size 3 \
                --lr 1e-3 \
                --seed "$seed" \
                --save_dir "$save_dir" \
                --device "$DEVICE"
        fi

        out_csv="$RUN_ROOT/eval/${model}_seed${seed}.csv"
        python -m evaluation.eval_ber_sionna \
            --checkpoint "$checkpoint" \
            --phase_mode uniform \
            --snr_list="$SNR_LIST" \
            --num_samples "$EVAL_SAMPLES" \
            --batch_size "$EVAL_BATCH_SIZE" \
            --seed "$EVAL_SEED" \
            --common_random_numbers \
            --device "$DEVICE" \
            --out_csv "$out_csv"
        CSV_FILES+=("$out_csv")
    done
done

python -m evaluation.aggregate_ber_seeds \
    --input_files "${CSV_FILES[@]}" \
    --out_csv "$RUN_ROOT/ber_summary.csv"

echo "Completed: $RUN_ROOT/ber_summary.csv"
