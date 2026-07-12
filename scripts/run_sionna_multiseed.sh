#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/venvs/sionna-pi}"
source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"

read -r -a TRAIN_SEEDS_ARRAY <<< "${TRAIN_SEEDS:-0 1 2}"
read -r -a EVAL_SEEDS_ARRAY <<< "${EVAL_SEEDS:-777000 888000}"

EPOCHS="${EPOCHS:-50}"
NUM_TRAIN="${NUM_TRAIN:-10000}"
NUM_VAL="${NUM_VAL:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SNR_DB_MIN="${SNR_DB_MIN:--5}"
SNR_DB_MAX="${SNR_DB_MAX:-20}"
NUM_EVAL="${NUM_EVAL:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
SKIP_TRAINED="${SKIP_TRAINED:-1}"
SKIP_EVALUATED="${SKIP_EVALUATED:-1}"
USE_EXISTING_SEED0="${USE_EXISTING_SEED0:-1}"
INCLUDE_N0_GATE="${INCLUDE_N0_GATE:-0}"
INCLUDE_GATE_ABLATIONS="${INCLUDE_GATE_ABLATIONS:-0}"
INCLUDE_COMPLEX_ZERO_CONDITION="${INCLUDE_COMPLEX_ZERO_CONDITION:-0}"
ZERO_GATE_HIDDEN="${ZERO_GATE_HIDDEN:-16}"
SNR_LIST="${SNR_LIST:--10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20}"

RUN_ROOT="${RUN_ROOT:-runs/sionna_multiseed}"
EVAL_DIR="$RUN_ROOT/eval"
mkdir -p "$EVAL_DIR"
if [[ "$INCLUDE_GATE_ABLATIONS" == "1" ]]; then
    INCLUDE_N0_GATE=1
fi

single_run_dir() {
    local train_seed="$1"
    if [[ "$train_seed" == "0" && "$USE_EXISTING_SEED0" == "1" ]]; then
        echo "runs/sionna_single_h64_hc32"
    else
        echo "$RUN_ROOT/single_seed${train_seed}"
    fi
}

complex_run_dir() {
    local train_seed="$1"
    if [[ "$train_seed" == "0" && "$USE_EXISTING_SEED0" == "1" ]]; then
        echo "runs/sionna_complex_h32_hc64_l3"
    else
        echo "$RUN_ROOT/complex_seed${train_seed}"
    fi
}

gated_run_dir() {
    local train_seed="$1"
    echo "$RUN_ROOT/gated_seed${train_seed}"
}

p_only_run_dir() {
    local train_seed="$1"
    echo "$RUN_ROOT/p_only_gate_seed${train_seed}"
}

n0_only_run_dir() {
    local train_seed="$1"
    echo "$RUN_ROOT/n0_only_gate_seed${train_seed}"
}

complex_zero_run_dir() {
    local model="$1"
    local train_seed="$2"
    echo "$RUN_ROOT/${model}_seed${train_seed}"
}

train_model() {
    local model="$1"
    local train_seed="$2"
    local run_dir="$3"
    shift 3

    if [[ "$SKIP_TRAINED" == "1" && -f "$run_dir/best.pt" ]]; then
        echo "Skipping existing checkpoint: $run_dir/best.pt"
        return
    fi

    python train_sionna.py \
        --model "$model" \
        --epochs "$EPOCHS" \
        --num_train "$NUM_TRAIN" \
        --num_val "$NUM_VAL" \
        --batch_size "$BATCH_SIZE" \
        --snr_db_min "$SNR_DB_MIN" \
        --snr_db_max "$SNR_DB_MAX" \
        --train_phase_mode fixed \
        --val_phase_mode uniform \
        --seed "$train_seed" \
        --save_dir "$run_dir" \
        "$@"
}

evaluate_checkpoint() {
    local checkpoint="$1"
    local eval_seed="$2"
    local out_csv="$3"
    if [[ "$SKIP_EVALUATED" == "1" && -f "$out_csv" ]]; then
        echo "Skipping existing evaluation: $out_csv"
        return
    fi
    python eval_ber_sionna.py \
        --checkpoint "$checkpoint" \
        --phase_mode uniform \
        --snr_list="$SNR_LIST" \
        --num_samples "$NUM_EVAL" \
        --batch_size "$EVAL_BATCH_SIZE" \
        --seed "$eval_seed" \
        --common_random_numbers \
        --out_csv "$out_csv"
}

for train_seed in "${TRAIN_SEEDS_ARRAY[@]}"; do
    single_dir="$(single_run_dir "$train_seed")"
    complex_dir="$(complex_run_dir "$train_seed")"
    train_model \
        single_branch \
        "$train_seed" \
        "$single_dir" \
        --hidden 64 \
        --hidden_complex 32

    train_model \
        complex_no_interaction \
        "$train_seed" \
        "$complex_dir" \
        --hidden 32 \
        --hidden_complex 64 \
        --branch_layers 3

    if [[ "$INCLUDE_N0_GATE" == "1" ]]; then
        gated_dir="$(gated_run_dir "$train_seed")"
        train_model \
            single_branch_n0_gate \
            "$train_seed" \
            "$gated_dir" \
            --hidden 64 \
            --hidden_complex 32 \
            --zero_gate_hidden "$ZERO_GATE_HIDDEN"
    fi

    if [[ "$INCLUDE_GATE_ABLATIONS" == "1" ]]; then
        p_only_dir="$(p_only_run_dir "$train_seed")"
        n0_only_dir="$(n0_only_run_dir "$train_seed")"
        train_model \
            single_branch_p_only_gate \
            "$train_seed" \
            "$p_only_dir" \
            --hidden 64 \
            --hidden_complex 32 \
            --zero_gate_hidden "$ZERO_GATE_HIDDEN"
        train_model \
            single_branch_n0_only_gate \
            "$train_seed" \
            "$n0_only_dir" \
            --hidden 64 \
            --hidden_complex 32 \
            --zero_gate_hidden "$ZERO_GATE_HIDDEN"
    fi

    if [[ "$INCLUDE_COMPLEX_ZERO_CONDITION" == "1" ]]; then
        for model in \
            complex_p \
            complex_n0 \
            complex_p_n0 \
            complex_p_n0_gate \
            complex_p_n0_film
        do
            run_dir="$(complex_zero_run_dir "$model" "$train_seed")"
            train_model \
                "$model" \
                "$train_seed" \
                "$run_dir" \
                --hidden 32 \
                --hidden_complex 64 \
                --branch_layers 3 \
                --zero_gate_hidden "$ZERO_GATE_HIDDEN"
        done
    fi
done

CSV_FILES=()
for eval_seed in "${EVAL_SEEDS_ARRAY[@]}"; do
    for train_seed in "${TRAIN_SEEDS_ARRAY[@]}"; do
        single_dir="$(single_run_dir "$train_seed")"
        complex_dir="$(complex_run_dir "$train_seed")"
        single_csv="$EVAL_DIR/single_train${train_seed}_eval${eval_seed}.csv"
        complex_csv="$EVAL_DIR/complex_train${train_seed}_eval${eval_seed}.csv"
        evaluate_checkpoint "$single_dir/best.pt" "$eval_seed" "$single_csv"
        evaluate_checkpoint "$complex_dir/best.pt" "$eval_seed" "$complex_csv"
        CSV_FILES+=("$single_csv" "$complex_csv")

        if [[ "$INCLUDE_N0_GATE" == "1" ]]; then
            gated_dir="$(gated_run_dir "$train_seed")"
            gated_csv="$EVAL_DIR/gated_train${train_seed}_eval${eval_seed}.csv"
            evaluate_checkpoint "$gated_dir/best.pt" "$eval_seed" "$gated_csv"
            CSV_FILES+=("$gated_csv")
        fi

        if [[ "$INCLUDE_GATE_ABLATIONS" == "1" ]]; then
            p_only_dir="$(p_only_run_dir "$train_seed")"
            n0_only_dir="$(n0_only_run_dir "$train_seed")"
            p_only_csv="$EVAL_DIR/p_only_train${train_seed}_eval${eval_seed}.csv"
            n0_only_csv="$EVAL_DIR/n0_only_train${train_seed}_eval${eval_seed}.csv"
            evaluate_checkpoint "$p_only_dir/best.pt" "$eval_seed" "$p_only_csv"
            evaluate_checkpoint "$n0_only_dir/best.pt" "$eval_seed" "$n0_only_csv"
            CSV_FILES+=("$p_only_csv" "$n0_only_csv")
        fi

        if [[ "$INCLUDE_COMPLEX_ZERO_CONDITION" == "1" ]]; then
            for model in \
                complex_p \
                complex_n0 \
                complex_p_n0 \
                complex_p_n0_gate \
                complex_p_n0_film
            do
                run_dir="$(complex_zero_run_dir "$model" "$train_seed")"
                csv="$EVAL_DIR/${model}_train${train_seed}_eval${eval_seed}.csv"
                evaluate_checkpoint "$run_dir/best.pt" "$eval_seed" "$csv"
                CSV_FILES+=("$csv")
            done
        fi
    done
done

python aggregate_ber_seeds.py \
    --input_files "${CSV_FILES[@]}" \
    --out_csv "$RUN_ROOT/ber_multiseed_summary.csv"

echo "Completed. Summary: $RUN_ROOT/ber_multiseed_summary.csv"
