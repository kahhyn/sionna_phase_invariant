#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/venvs/sionna-pi}"
source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"

read -r -a TRAIN_SEEDS_ARRAY <<< "${TRAIN_SEEDS:-0 1 2}"
read -r -a EVAL_SEEDS_ARRAY <<< "${EVAL_SEEDS:-777000 888000}"

MODELS=(
    single_branch_n0_gate_h_denoise
    strict_matched_complex_p_n0_gate_h_denoise
)

RUN_ROOT="${RUN_ROOT:-runs/sionna_h_denoise_three_stage_m10_p20}"
BASELINE_ROOT="${BASELINE_ROOT:-runs/sionna_invariance_matched_m10_p20}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-30}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-50}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-10}"
NUM_TRAIN="${NUM_TRAIN:-10000}"
NUM_VAL="${NUM_VAL:-2000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SNR_DB_MIN="${SNR_DB_MIN:--10}"
SNR_DB_MAX="${SNR_DB_MAX:-20}"
JOINT_H_WEIGHT="${JOINT_H_WEIGHT:-0.01}"
JOINT_LR="${JOINT_LR:-1e-4}"
SKIP_TRAINED="${SKIP_TRAINED:-1}"
SKIP_EVALUATED="${SKIP_EVALUATED:-1}"
RUN_BER="${RUN_BER:-1}"
INCLUDE_BASELINES="${INCLUDE_BASELINES:-1}"
NUM_EVAL="${NUM_EVAL:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
SNR_LIST="${SNR_LIST:--10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20}"
NMSE_SNR_LIST="${NMSE_SNR_LIST:--10,-5,0,5,10,15,20}"
NMSE_EVAL_SAMPLES="${NMSE_EVAL_SAMPLES:-2000}"

mkdir -p "$RUN_ROOT/eval"

COMMON_ARGS=(
    --hidden 64
    --hidden_complex 32
    --zero_complex 32
    --branch_layers 2
    --zero_gate_hidden 16
    --denoiser_hidden 16
    --denoiser_blocks 2
    --num_train "$NUM_TRAIN"
    --num_val "$NUM_VAL"
    --batch_size "$BATCH_SIZE"
    --snr_db_min "$SNR_DB_MIN"
    --snr_db_max "$SNR_DB_MAX"
    --train_phase_mode fixed
    --val_phase_mode uniform
    --device cuda
)

train_if_missing() {
    local run_dir="$1"
    shift
    if [[ "$SKIP_TRAINED" == "1" && -f "$run_dir/best.pt" ]]; then
        echo "Skipping existing checkpoint: $run_dir/best.pt"
        return
    fi
    python train_sionna_denoise.py \
        --save_dir "$run_dir" \
        "$@"
}

# Stage 1: one shared H denoiser per training seed.
for train_seed in "${TRAIN_SEEDS_ARRAY[@]}"; do
    denoiser_dir="$RUN_ROOT/denoiser_seed${train_seed}"
    train_if_missing \
        "$denoiser_dir" \
        --stage denoiser \
        --epochs "$STAGE1_EPOCHS" \
        --lr 1e-3 \
        --seed "$train_seed" \
        --nmse_snr_list="$NMSE_SNR_LIST" \
        --nmse_eval_samples "$NMSE_EVAL_SAMPLES" \
        "${COMMON_ARGS[@]}"
done

# Stage 2: train fresh A/C receivers while the shared denoiser is frozen.
for model in "${MODELS[@]}"; do
    for train_seed in "${TRAIN_SEEDS_ARRAY[@]}"; do
        denoiser_checkpoint="$RUN_ROOT/denoiser_seed${train_seed}/best.pt"
        frozen_dir="$RUN_ROOT/${model}_frozen_seed${train_seed}"
        train_if_missing \
            "$frozen_dir" \
            --stage frozen \
            --model "$model" \
            --denoiser_checkpoint "$denoiser_checkpoint" \
            --epochs "$STAGE2_EPOCHS" \
            --lr 1e-3 \
            --seed "$train_seed" \
            "${COMMON_ARGS[@]}"
    done
done

# Stage 3: jointly fine-tune from each frozen-stage best checkpoint.
for model in "${MODELS[@]}"; do
    for train_seed in "${TRAIN_SEEDS_ARRAY[@]}"; do
        frozen_checkpoint="$RUN_ROOT/${model}_frozen_seed${train_seed}/best.pt"
        joint_dir="$RUN_ROOT/${model}_joint_seed${train_seed}"
        train_if_missing \
            "$joint_dir" \
            --stage joint \
            --model "$model" \
            --init_checkpoint "$frozen_checkpoint" \
            --epochs "$STAGE3_EPOCHS" \
            --lr "$JOINT_LR" \
            --h_denoise_weight "$JOINT_H_WEIGHT" \
            --seed "$train_seed" \
            "${COMMON_ARGS[@]}"
    done
done

if [[ "$RUN_BER" != "1" ]]; then
    echo "Training completed. RUN_BER=$RUN_BER, so BER evaluation was skipped."
    exit 0
fi

evaluate_if_missing() {
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
        --device cuda \
        --out_csv "$out_csv"
}

CSV_FILES=()
for stage in frozen joint; do
    for model in "${MODELS[@]}"; do
        for train_seed in "${TRAIN_SEEDS_ARRAY[@]}"; do
            checkpoint="$RUN_ROOT/${model}_${stage}_seed${train_seed}/best.pt"
            for eval_seed in "${EVAL_SEEDS_ARRAY[@]}"; do
                csv="$RUN_ROOT/eval/${model}_${stage}_train${train_seed}_eval${eval_seed}.csv"
                evaluate_if_missing "$checkpoint" "$eval_seed" "$csv"
                CSV_FILES+=("$csv")
            done
        done
    done
done

# Reuse the already completed no-denoiser A0/C0 evaluations when available.
if [[ "$INCLUDE_BASELINES" == "1" ]]; then
    for model in single_branch_n0_gate strict_matched_complex_p_n0_gate; do
        for train_seed in "${TRAIN_SEEDS_ARRAY[@]}"; do
            for eval_seed in "${EVAL_SEEDS_ARRAY[@]}"; do
                csv="$BASELINE_ROOT/eval/${model}_train${train_seed}_eval${eval_seed}.csv"
                if [[ ! -f "$csv" ]]; then
                    echo "Missing baseline CSV: $csv" >&2
                    exit 1
                fi
                CSV_FILES+=("$csv")
            done
        done
    done
fi

python aggregate_ber_seeds.py \
    --input_files "${CSV_FILES[@]}" \
    --out_csv "$RUN_ROOT/ber_multiseed_summary.csv"

echo "Completed. Summary: $RUN_ROOT/ber_multiseed_summary.csv"
