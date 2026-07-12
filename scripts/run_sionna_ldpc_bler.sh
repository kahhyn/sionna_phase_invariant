#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/venvs/sionna-pi}"
source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-runs/sionna_multiseed_m10_p20}"
RUN_ROOT="${RUN_ROOT:-runs/sionna_ldpc_r050}"
read -r -a TRAIN_SEEDS_ARRAY <<< "${TRAIN_SEEDS:-0 1 2}"
read -r -a EVAL_SEEDS_ARRAY <<< "${EVAL_SEEDS:-777000 888000}"

EBNO_LIST="${EBNO_LIST:-0,1,2,3,4,5,6,7,8,9,10}"
CODERATE="${CODERATE:-0.5}"
DECODER_ITERATIONS="${DECODER_ITERATIONS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TARGET_BLOCK_ERRORS="${TARGET_BLOCK_ERRORS:-100}"
MAX_BLOCKS="${MAX_BLOCKS:-20000}"
INCLUDE_UNGATED="${INCLUDE_UNGATED:-1}"
INCLUDE_N0_ONLY="${INCLUDE_N0_ONLY:-0}"
INCLUDE_COMPLEX_ZERO_CONDITION="${INCLUDE_COMPLEX_ZERO_CONDITION:-0}"
INCLUDE_LMMSE="${INCLUDE_LMMSE:-0}"
INCLUDE_LMMSE_PERFECT="${INCLUDE_LMMSE_PERFECT:-0}"
SKIP_EVALUATED="${SKIP_EVALUATED:-1}"

EVAL_DIR="$RUN_ROOT/eval"
mkdir -p "$EVAL_DIR"
CSV_FILES=()
CONFIG_CHECKPOINT="${CONFIG_CHECKPOINT:-$CHECKPOINT_ROOT/complex_seed${TRAIN_SEEDS_ARRAY[0]}/best.pt}"

evaluate_checkpoint() {
    local checkpoint="$1"
    local eval_seed="$2"
    local out_csv="$3"
    if [[ "$SKIP_EVALUATED" == "1" && -f "$out_csv" ]]; then
        echo "Skipping existing evaluation: $out_csv"
        return
    fi
    python eval_bler_sionna.py \
        --checkpoint "$checkpoint" \
        --ebno_list="$EBNO_LIST" \
        --coderate "$CODERATE" \
        --decoder_iterations "$DECODER_ITERATIONS" \
        --phase_mode uniform \
        --batch_size "$BATCH_SIZE" \
        --target_block_errors "$TARGET_BLOCK_ERRORS" \
        --max_blocks "$MAX_BLOCKS" \
        --seed "$eval_seed" \
        --common_random_numbers \
        --out_csv "$out_csv"
}

evaluate_baseline() {
    local receiver="$1"
    local eval_seed="$2"
    local out_csv="$3"
    if [[ "$SKIP_EVALUATED" == "1" && -f "$out_csv" ]]; then
        echo "Skipping existing evaluation: $out_csv"
        return
    fi
    python eval_bler_sionna.py \
        --receiver "$receiver" \
        --checkpoint "$CONFIG_CHECKPOINT" \
        --ebno_list="$EBNO_LIST" \
        --coderate "$CODERATE" \
        --decoder_iterations "$DECODER_ITERATIONS" \
        --phase_mode uniform \
        --batch_size "$BATCH_SIZE" \
        --target_block_errors "$TARGET_BLOCK_ERRORS" \
        --max_blocks "$MAX_BLOCKS" \
        --seed "$eval_seed" \
        --common_random_numbers \
        --out_csv "$out_csv"
}

for eval_seed in "${EVAL_SEEDS_ARRAY[@]}"; do
    if [[ "$INCLUDE_LMMSE" == "1" ]]; then
        lmmse_csv="$EVAL_DIR/lmmse_ls_eval${eval_seed}.csv"
        evaluate_baseline "lmmse_ls" "$eval_seed" "$lmmse_csv"
        CSV_FILES+=("$lmmse_csv")
    fi

    if [[ "$INCLUDE_LMMSE_PERFECT" == "1" ]]; then
        perfect_csv="$EVAL_DIR/lmmse_perfect_eval${eval_seed}.csv"
        evaluate_baseline "lmmse_perfect" "$eval_seed" "$perfect_csv"
        CSV_FILES+=("$perfect_csv")
    fi

    for train_seed in "${TRAIN_SEEDS_ARRAY[@]}"; do
        complex_csv="$EVAL_DIR/complex_train${train_seed}_eval${eval_seed}.csv"
        gated_csv="$EVAL_DIR/gated_train${train_seed}_eval${eval_seed}.csv"
        evaluate_checkpoint \
            "$CHECKPOINT_ROOT/complex_seed${train_seed}/best.pt" \
            "$eval_seed" \
            "$complex_csv"
        evaluate_checkpoint \
            "$CHECKPOINT_ROOT/gated_seed${train_seed}/best.pt" \
            "$eval_seed" \
            "$gated_csv"
        CSV_FILES+=("$complex_csv" "$gated_csv")

        if [[ "$INCLUDE_UNGATED" == "1" ]]; then
            single_csv="$EVAL_DIR/single_train${train_seed}_eval${eval_seed}.csv"
            evaluate_checkpoint \
                "$CHECKPOINT_ROOT/single_seed${train_seed}/best.pt" \
                "$eval_seed" \
                "$single_csv"
            CSV_FILES+=("$single_csv")
        fi

        if [[ "$INCLUDE_N0_ONLY" == "1" ]]; then
            n0_csv="$EVAL_DIR/n0_only_train${train_seed}_eval${eval_seed}.csv"
            evaluate_checkpoint \
                "$CHECKPOINT_ROOT/n0_only_gate_seed${train_seed}/best.pt" \
                "$eval_seed" \
                "$n0_csv"
            CSV_FILES+=("$n0_csv")
        fi

        if [[ "$INCLUDE_COMPLEX_ZERO_CONDITION" == "1" ]]; then
            for model in \
                complex_p \
                complex_n0 \
                complex_p_n0 \
                complex_p_n0_gate \
                complex_p_n0_film
            do
                csv="$EVAL_DIR/${model}_train${train_seed}_eval${eval_seed}.csv"
                evaluate_checkpoint \
                    "$CHECKPOINT_ROOT/${model}_seed${train_seed}/best.pt" \
                    "$eval_seed" \
                    "$csv"
                CSV_FILES+=("$csv")
            done
        fi
    done
done

python aggregate_bler_seeds.py \
    --input_files "${CSV_FILES[@]}" \
    --out_csv "$RUN_ROOT/bler_multiseed_summary.csv"

echo "Completed. Summary: $RUN_ROOT/bler_multiseed_summary.csv"
