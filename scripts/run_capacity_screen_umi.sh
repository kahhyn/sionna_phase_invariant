#!/usr/bin/env bash
set -euo pipefail

# Exploratory capacity screen: one UMi training seed, UMi ID evaluation, and
# UMa OOD evaluation. This is intentionally smaller than the formal protocol.
CAPACITY=${CAPACITY:?Set CAPACITY to 50k or 20k.}
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-30}
NUM_TRAIN=${NUM_TRAIN:-5000}
NUM_VAL=${NUM_VAL:-1000}
NUM_EVAL=${NUM_EVAL:-1024}
TRAIN_SEED=${TRAIN_SEED:-0}
EVAL_SEED=${EVAL_SEED:-777000}
SNR_LIST=${SNR_LIST:-"-10,0,10,16,20"}

case "$CAPACITY" in
  50k)
    HIDDEN=32
    HIDDEN_COMPLEX=16
    ZERO_COMPLEX=16
    ZERO_GATE_HIDDEN=8
    TRUNK_HIDDEN=25
    EXPECTED_SINGLE=52255
    EXPECTED_REAL=52524
    ;;
  20k)
    HIDDEN=20
    HIDDEN_COMPLEX=10
    ZERO_COMPLEX=10
    ZERO_GATE_HIDDEN=5
    TRUNK_HIDDEN=15
    EXPECTED_SINGLE=20932
    EXPECTED_REAL=20322
    ;;
  *)
    echo "Unsupported CAPACITY=$CAPACITY; expected 50k or 20k." >&2
    exit 2
    ;;
esac

RUN_ROOT=${RUN_ROOT:-"runs/capacity_screen_umi/${CAPACITY}"}
TRAIN_PROFILE=configs/channel_profiles/umi_normalized.json
mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/eval" "$RUN_ROOT/manifests"

{
  echo "git_commit=$(git rev-parse HEAD)"
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "capacity=$CAPACITY"
  echo "epochs=$EPOCHS"
  echo "num_train=$NUM_TRAIN"
  echo "num_val=$NUM_VAL"
  echo "num_eval=$NUM_EVAL"
  echo "train_seed=$TRAIN_SEED"
  echo "eval_seed=$EVAL_SEED"
  echo "snr_list=$SNR_LIST"
  git status --short
} > "$RUN_ROOT/manifests/run_environment.txt"

python - "$HIDDEN" "$HIDDEN_COMPLEX" "$ZERO_COMPLEX" \
  "$ZERO_GATE_HIDDEN" "$TRUNK_HIDDEN" "$EXPECTED_SINGLE" \
  "$EXPECTED_REAL" <<'PY'
import sys

from models.factory import build_model

hidden, hidden_complex, zero_complex, gate_hidden, trunk_hidden = map(
    int, sys.argv[1:6]
)
expected_single, expected_real = map(int, sys.argv[6:8])
common = dict(
    bits_per_symbol=2,
    hidden=hidden,
    hidden_complex=hidden_complex,
    zero_complex=zero_complex,
    branch_layers=2,
    zero_gate_hidden=gate_hidden,
)
single = build_model("single_branch_n0_gate", **common)
real = build_model("real_imag_cnn", trunk_hidden=trunk_hidden, **common)
single_count = sum(p.numel() for p in single.parameters() if p.requires_grad)
real_count = sum(p.numel() for p in real.parameters() if p.requires_grad)
print(f"single_branch_n0_gate parameters: {single_count}")
print(f"real_imag_cnn parameters:         {real_count}")
assert single_count == expected_single
assert real_count == expected_real
PY

common_train_args=(
  --train_channel_profile "$TRAIN_PROFILE"
  --val_channel_profile "$TRAIN_PROFILE"
  --train_phase_mode fixed
  --val_phase_mode uniform
  --hidden "$HIDDEN"
  --hidden_complex "$HIDDEN_COMPLEX"
  --zero_complex "$ZERO_COMPLEX"
  --zero_gate_hidden "$ZERO_GATE_HIDDEN"
  --branch_layers 2
  --kernel_size 3
  --gate_type swiglu
  --single_readout_mode low_rank
  --epochs "$EPOCHS"
  --num_train "$NUM_TRAIN"
  --num_val "$NUM_VAL"
  --batch_size 64
  --snr_db_min -10
  --snr_db_max 20
  --seed "$TRAIN_SEED"
  --device "$DEVICE"
)

single_dir="$RUN_ROOT/checkpoints/single_branch_n0_gate/seed_${TRAIN_SEED}"
real_dir="$RUN_ROOT/checkpoints/real_imag_cnn/seed_${TRAIN_SEED}"

if [[ ! -f "$single_dir/best.pt" ]]; then
  python -m training.train_sionna \
    --model single_branch_n0_gate \
    "${common_train_args[@]}" \
    --save_dir "$single_dir"
else
  echo "Skip existing checkpoint: $single_dir/best.pt"
fi

if [[ ! -f "$real_dir/best.pt" ]]; then
  python -m training.train_sionna \
    --model real_imag_cnn \
    --trunk_hidden "$TRUNK_HIDDEN" \
    "${common_train_args[@]}" \
    --save_dir "$real_dir"
else
  echo "Skip existing checkpoint: $real_dir/best.pt"
fi

for model in single_branch_n0_gate real_imag_cnn; do
  checkpoint="$RUN_ROOT/checkpoints/$model/seed_${TRAIN_SEED}/best.pt"
  for entry in \
    "umi:configs/channel_profiles/umi_normalized.json" \
    "uma:configs/channel_profiles/uma_normalized.json"; do
    label=${entry%%:*}
    profile=${entry#*:}
    out_csv="$RUN_ROOT/eval/${model}_${label}.csv"
    if [[ ! -f "$out_csv" ]]; then
      python -m evaluation.eval_ber_sionna \
        --checkpoint "$checkpoint" \
        --eval_channel_profile "$profile" \
        --phase_mode uniform \
        --snr_list="$SNR_LIST" \
        --num_samples "$NUM_EVAL" \
        --batch_size 64 \
        --seed "$EVAL_SEED" \
        --common_random_numbers \
        --device "$DEVICE" \
        --out_csv "$out_csv"
    else
      echo "Skip existing evaluation: $out_csv"
    fi
  done
done

python - "$RUN_ROOT" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "eval"
print("\nscenario,snr_db,real_ber,single_ber,single_reduction_percent")
for scenario in ("umi", "uma"):
    rows = {}
    for model in ("real_imag_cnn", "single_branch_n0_gate"):
        path = root / f"{model}_{scenario}.csv"
        rows[model] = {
            float(row["snr_db"]): float(row["ber"])
            for row in csv.DictReader(path.open())
        }
    for snr_db in sorted(rows["real_imag_cnn"]):
        real = rows["real_imag_cnn"][snr_db]
        single = rows["single_branch_n0_gate"][snr_db]
        reduction = 100.0 * (real - single) / real if real else float("nan")
        print(
            f"{scenario},{snr_db:g},{real:.8g},{single:.8g},{reduction:+.2f}"
        )
PY
