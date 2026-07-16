#!/usr/bin/env bash
set -euo pipefail

CHANNEL_H5="${CHANNEL_H5:-data/dichasus/dichasus-0152_72sc_30khz.h5}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-checkpoints/generalization}"
TRAIN_PROFILES="${TRAIN_PROFILES:-tdl_mix_normalized umi_uma_mix_normalized}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2}"
ANTENNA_INDICES="${ANTENNA_INDICES:-all}"
SNR_LIST="${SNR_LIST:--10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20}"
PHASE_MODE="${PHASE_MODE:-fixed}"
NORMALIZATION="${NORMALIZATION:-checkpoint}"
POSITION_SOURCE="${POSITION_SOURCE:-lidar}"
MIN_MEASURED_SNR_DB="${MIN_MEASURED_SNR_DB:--inf}"
START_RECORD="${START_RECORD:-0}"
RECORD_STRIDE="${RECORD_STRIDE:-1}"
MAX_RECORDS="${MAX_RECORDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WINDOW_FRAMES="${WINDOW_FRAMES:-128}"
EVAL_SEED="${EVAL_SEED:-777000}"
DEVICE="${DEVICE:-cuda}"
RUN_ROOT="${RUN_ROOT:-runs/dichasus_fixed}"
SKIP_EVALUATED="${SKIP_EVALUATED:-1}"

if [[ ! -f "$CHANNEL_H5" ]]; then
    echo "DICHASUS HDF5 file not found: $CHANNEL_H5" >&2
    exit 1
fi
if [[ "$ANTENNA_INDICES" == "all" ]]; then
    ANTENNA_INDICES="$(seq 0 31)"
fi

mkdir -p "$RUN_ROOT"
for train_profile in $TRAIN_PROFILES; do
    for train_seed in $TRAIN_SEEDS; do
        invariant_checkpoint="$CHECKPOINT_BASE/$train_profile/single_branch_n0_gate_seed${train_seed}.pt"
        strict_checkpoint="$CHECKPOINT_BASE/$train_profile/strict_matched_complex_p_n0_gate_seed${train_seed}.pt"
        if [[ ! -f "$invariant_checkpoint" || ! -f "$strict_checkpoint" ]]; then
            echo "Missing A/C checkpoints for $train_profile seed $train_seed" >&2
            exit 1
        fi
        for antenna_index in $ANTENNA_INDICES; do
            output_dir="$RUN_ROOT/$train_profile/seed_${train_seed}/antenna_${antenna_index}"
            summary="$output_dir/dichasus_summary.csv"
            if [[ "$SKIP_EVALUATED" == "1" && -f "$summary" ]]; then
                echo "Skip existing: $summary"
                continue
            fi
            echo "Run $train_profile seed=$train_seed antenna=$antenna_index"
            python -m evaluation.eval_dichasus \
                --channel_h5 "$CHANNEL_H5" \
                --antenna_index "$antenna_index" \
                --invariant_checkpoint "$invariant_checkpoint" \
                --strict_checkpoint "$strict_checkpoint" \
                --output_dir "$output_dir" \
                --snr_list="$SNR_LIST" \
                --phase_mode "$PHASE_MODE" \
                --normalization "$NORMALIZATION" \
                --position_source "$POSITION_SOURCE" \
                --min_measured_snr_db="$MIN_MEASURED_SNR_DB" \
                --start_record "$START_RECORD" \
                --record_stride "$RECORD_STRIDE" \
                --max_records "$MAX_RECORDS" \
                --batch_size "$BATCH_SIZE" \
                --window_frames "$WINDOW_FRAMES" \
                --seed "$EVAL_SEED" \
                --device "$DEVICE"
        done
    done
done

python -m evaluation.aggregate_dichasus \
    --input_root "$RUN_ROOT" \
    --out_csv "$RUN_ROOT/dichasus_aggregate.csv" \
    --paired_out_csv "$RUN_ROOT/dichasus_paired_ac.csv"

echo "Completed: $RUN_ROOT"
