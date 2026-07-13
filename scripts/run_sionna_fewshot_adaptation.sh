#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-checkpoints/continual_source}"
RUN_ROOT="${RUN_ROOT:-runs/sionna_continual_fewshot}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0,1,2}"
TARGET_DELAY_NS="${TARGET_DELAY_NS:-50,100,300}"
SAMPLE_BUDGETS="${SAMPLE_BUDGETS:-0,16,64,256,1024,4096}"
EVAL_SNR_LIST="${EVAL_SNR_LIST:--5,0,5,10,15,20}"
ADAPT_EPOCHS="${ADAPT_EPOCHS:-5}"
ADAPT_BATCH_SIZE="${ADAPT_BATCH_SIZE:-16}"
NUM_EVAL_PER_SNR="${NUM_EVAL_PER_SNR:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
ADAPT_LR="${ADAPT_LR:-1e-4}"
ADAPT_SEED_BASE="${ADAPT_SEED_BASE:-500000}"
EVAL_SEED="${EVAL_SEED:-777000}"
DEVICE="${DEVICE:-cuda}"
FORCE="${FORCE:-0}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"

mkdir -p "${RUN_ROOT}"
IFS=',' read -r -a seeds <<< "${TRAIN_SEEDS}"

run_one() {
  local model="$1"
  local seed="$2"
  local checkpoint="${SOURCE_ROOT}/${model}_seed${seed}.pt"
  local output_dir="${RUN_ROOT}/${model}_seed${seed}"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing source checkpoint: ${checkpoint}" >&2
    return 1
  fi
  if [[ "${FORCE}" != "1" && -f "${output_dir}/fewshot_summary.csv" ]]; then
    echo "Skipping existing result: ${output_dir}/fewshot_summary.csv"
    return 0
  fi

  local extra_args=()
  if [[ "${SAVE_CHECKPOINTS}" == "1" ]]; then
    extra_args+=(--save_checkpoints)
  fi

  python -m experiments.continual_learning.fewshot_finetune \
    --checkpoint "${checkpoint}" \
    --output_dir "${output_dir}" \
    --target_delay_ns "${TARGET_DELAY_NS}" \
    --sample_budgets "${SAMPLE_BUDGETS}" \
    --adapt_epochs "${ADAPT_EPOCHS}" \
    --adapt_batch_size "${ADAPT_BATCH_SIZE}" \
    --adapt_snr_db_min -10 \
    --adapt_snr_db_max 20 \
    --adapt_phase_mode uniform \
    --lr "${ADAPT_LR}" \
    --eval_snr_list="${EVAL_SNR_LIST}" \
    --num_eval_per_snr "${NUM_EVAL_PER_SNR}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --eval_phase_mode uniform \
    --adapt_seed "$((ADAPT_SEED_BASE + seed * 10000))" \
    --eval_seed "${EVAL_SEED}" \
    --device "${DEVICE}" \
    "${extra_args[@]}"
}

for seed in "${seeds[@]}"; do
  run_one "single_branch_n0_gate" "${seed}"
  run_one "strict_matched_complex_p_n0_gate" "${seed}"
done

python -m experiments.continual_learning.aggregate_fewshot \
  --run_root "${RUN_ROOT}" \
  --out_csv "${RUN_ROOT}/fewshot_multiseed_summary.csv"

echo "Completed. Summary: ${RUN_ROOT}/fewshot_multiseed_summary.csv"
