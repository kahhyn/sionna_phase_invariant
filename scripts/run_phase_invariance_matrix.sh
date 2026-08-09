#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
mode="${MODE:-all}"
run_root="${RUN_ROOT:-runs/phase_invariance_matrix_v1}"
systems="${SYSTEMS:-siso_1l1rx,mimo_2l2rx,mimo_2l16rx}"
train_domains="${TRAIN_DOMAINS:-tdl_a,tdl_mix}"
test_domains="${TEST_DOMAINS:-id_fixed,phase_ood,delay_ood,doppler_ood,quadriga_ood}"
metrics="${METRICS:-bler}"
seeds="${SEEDS:-0}"
eval_seeds="${EVAL_SEEDS:-777000}"

args=(
  --mode "$mode"
  --run_root "$run_root"
  --systems "$systems"
  --train_domains "$train_domains"
  --test_domains "$test_domains"
  --metrics "$metrics"
  --seeds "$seeds"
  --eval_seeds "$eval_seeds"
  --python "$python_bin"
  --device "${DEVICE:-cuda}"
  "--snr_list=${SNR_LIST:--10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20}"
  "--ebno_list=${EBNO_LIST:--5,-3,-1,0,1,2,3,4,5,6,7,8,9,11,13}"
  --num_samples "${NUM_SAMPLES:-4096}"
  --target_block_errors "${TARGET_BLOCK_ERRORS:-500}"
  --max_blocks "${MAX_BLOCKS:-20000}"
  --coderate "${CODERATE:-0.5}"
  --decoder_iterations "${DECODER_ITERATIONS:-20}"
  --eval_batch_size "${EVAL_BATCH_SIZE:-16}"
  --epochs "${EPOCHS:-130}"
  --num_train "${NUM_TRAIN:-10000}"
  --num_val "${NUM_VAL:-2000}"
  --train_batch_size "${TRAIN_BATCH_SIZE:-64}"
  --warmup_epochs "${WARMUP_EPOCHS:-5}"
  --constant_tail_epochs "${CONSTANT_TAIL_EPOCHS:-30}"
  --quadriga_pattern "${QUADRIGA_PATTERN:-*.mat}"
  --quadriga_repetitions "${QUADRIGA_REPETITIONS:-1}"
  --mimo_mat_layout "${MIMO_MAT_LAYOUT:-frame_rx_layer_symbol_subcarrier}"
)

if [[ -n "${QUADRIGA_SISO_DIR:-}" ]]; then
  args+=(--quadriga_siso_dir "$QUADRIGA_SISO_DIR")
fi
if [[ -n "${QUADRIGA_MIMO_RX2_DIR:-}" ]]; then
  args+=(--quadriga_mimo_rx2_dir "$QUADRIGA_MIMO_RX2_DIR")
fi
if [[ -n "${QUADRIGA_MIMO_RX16_DIR:-}" ]]; then
  args+=(--quadriga_mimo_rx16_dir "$QUADRIGA_MIMO_RX16_DIR")
fi
if [[ "${REQUIRE_QUADRIGA:-0}" == "1" ]]; then
  args+=(--require_quadriga)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry_run)
fi
if [[ "${SKIP_EXISTING:-1}" == "0" ]]; then
  args+=(--no-skip_existing)
fi

"$python_bin" -m experiments.phase_invariance_matrix "${args[@]}"
