#!/usr/bin/env bash
set -euo pipefail

# Formal matched RealImagCNN experiment. All optimization and data settings
# mirror the published single_branch_n0_gate protocol.
TRAIN_SEEDS=${TRAIN_SEEDS:-"0 1 2"}
EVAL_SEEDS=${EVAL_SEEDS:-"777000 888000"}
EPOCHS=${EPOCHS:-50}
NUM_TRAIN=${NUM_TRAIN:-10000}
NUM_VAL=${NUM_VAL:-2000}
BATCH_SIZE=${BATCH_SIZE:-64}
# Match the existing A/C generalization CSVs (1024 samples per SNR and
# checkpoint); pooled over 3 train seeds x 2 evaluation seeds.
NUM_EVAL=${NUM_EVAL:-1024}
SNR_LIST=${SNR_LIST:-"-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20"}
TEST_SUITE=${TEST_SUITE:-configs/channel_suites/generalization_normalized.json}
RUN_ROOT=${RUN_ROOT:-runs/realimag_matched_four_domains}
RUN_BLER=${RUN_BLER:-0}
DEVICE=${DEVICE:-cuda}

profiles=(
  "tdl_a:configs/channel_profiles/tdl_a_10_30_100_mix_normalized.json"
  "tdl_mix:configs/channel_profiles/tdl_mix_normalized.json"
  "umi:configs/channel_profiles/umi_normalized.json"
  "umi_uma_mix:configs/channel_profiles/umi_uma_mix_normalized.json"
)

for entry in "${profiles[@]}"; do
  label=${entry%%:*}
  profile=${entry#*:}
  domain_root="$RUN_ROOT/$label"
  mkdir -p "$domain_root"

  TRAIN_PROFILE="$profile" \
  TEST_SUITE="$TEST_SUITE" \
  MODELS="real_imag_cnn" \
  TRAIN_SEEDS="$TRAIN_SEEDS" \
  EVAL_SEEDS="$EVAL_SEEDS" \
  EPOCHS="$EPOCHS" \
  NUM_TRAIN="$NUM_TRAIN" \
  NUM_VAL="$NUM_VAL" \
  BATCH_SIZE="$BATCH_SIZE" \
  NUM_EVAL="$NUM_EVAL" \
  SNR_LIST="$SNR_LIST" \
  RUN_ROOT="$domain_root" \
  RUN_BER=1 \
  RUN_BLER="$RUN_BLER" \
  SKIP_TRAINED=1 \
  SKIP_EVALUATED=1 \
  DEVICE="$DEVICE" \
  bash scripts/run_channel_generalization.sh \
    |& tee -a "$domain_root/run.log"
done
