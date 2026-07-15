# Sionna migration

This branch keeps the existing PyTorch receiver models and replaces the
hand-written physical-layer data path with Sionna 2.x blocks.

The complete commands and parameter reference for the current TDL/UMi/UMa
experiments are in
[`docs/TRAINING_EVALUATION_GUIDE.md`](docs/TRAINING_EVALUATION_GUIDE.md).

## Implemented chain

```text
legacy-labeled QPSK
  -> Sionna ResourceGrid / custom DMRS PilotPattern
  -> 3GPP TR 38.901 TDL, UMi, or UMa channel profile
  -> Sionna AWGN
  -> Sionna LS estimation and interpolation
  -> legacy-compatible batch dictionary
  -> existing PyTorch receiver
```

The default setup is SISO, 14 OFDM symbols, 72 subcarriers, 30 kHz SCS,
TDL-A with 10 ns delay spread, up to 200 Hz Doppler, and full-symbol DMRS at
OFDM symbols 2 and 11. Noise power follows the legacy project's measured
symbol-SNR convention rather than an `Eb/N0` conversion.

## TDL / UMi / UMa channel profiles

The generator accepts JSON profiles that select a fixed channel or balance a
bank of channels at batch granularity. The normalized profiles under
`configs/channel_profiles/` keep pathloss and shadow fading disabled and keep
channel normalization enabled. They test small-scale channel-structure
generalization; they do not represent a realistic coverage/link-budget test.

Minimal UMi training:

```bash
python -m training.train_sionna \
  --model single_branch_n0_gate \
  --train_channel_profile configs/channel_profiles/umi_normalized.json \
  --val_channel_profile configs/channel_profiles/umi_normalized.json \
  --epochs 1 --num_train 128 --num_val 64 --batch_size 16 \
  --snr_db_min -10 --snr_db_max 20 \
  --save_dir runs/smoke_umi_single
```

To evaluate a checkpoint on one fixed domain from the TDL bank:

```bash
python -m evaluation.eval_ber_sionna \
  --checkpoint runs/smoke_umi_single/best.pt \
  --eval_channel_profile configs/channel_profiles/tdl_mix_normalized.json \
  --eval_component_id tdl_C_100ns \
  --snr_list=-4,0,4 --num_samples 256 --common_random_numbers \
  --out_csv runs/smoke_umi_single/ber_tdl_C_100ns.csv
```

The complete 22-domain suite and pooled result matrices are managed by
`scripts/run_channel_generalization.sh`. See
`experiments/channel_generalization/README.md` and
`docs/TDL_UMI_UMA_GENERALIZATION_PLAN.md`.

## Environment

```bash
source ~/venvs/sionna-pi/bin/activate
cd ~/projects/phase_invariant_receiver_sionna
python -m unittest tests.test_sionna_generator -v
```

## Parameter-matched training

The optional `single_branch_n0_gate` model injects the charge-zero features
`P` and `log(N0)` into the complex trunk. A real amplitude gate is applied
after the input projection and after each residual block, so common-phase
equivariance is preserved. The gate starts as the identity.

```bash
python -m training.train_sionna --model single_branch_n0_gate --hidden 64 --hidden_complex 32 --zero_gate_hidden 16 --snr_db_min -10 --snr_db_max 20 --epochs 50 --batch_size 64 --num_train 10000 --num_val 2000 --train_phase_mode fixed --val_phase_mode uniform --save_dir runs/sionna_single_n0_gate
```

Single-branch invariant receiver:

```bash
python -m training.train_sionna --model single_branch --hidden 64 --hidden_complex 32 --epochs 50 --batch_size 64 --num_train 10000 --num_val 2000 --train_phase_mode fixed --val_phase_mode uniform --save_dir runs/sionna_single_h64_hc32
```

Complex no-interaction receiver:

```bash
python -m training.train_sionna --model complex_no_interaction --hidden 32 --hidden_complex 64 --branch_layers 3 --epochs 50 --batch_size 64 --num_train 10000 --num_val 2000 --train_phase_mode fixed --val_phase_mode uniform --save_dir runs/sionna_complex_h32_hc64_l3
```

## BER evaluation

```bash
python -m evaluation.eval_ber_sionna --checkpoint runs/sionna_single_h64_hc32/best.pt --phase_mode uniform --num_samples 4096 --batch_size 128 --out_csv runs/sionna_single_h64_hc32/ber_uniform.csv
```

BER and BCE are aggregated by valid bit count. The last partial batch is not
overweighted.

To reuse exactly the same bits, channel realizations, normalized noise samples,
and phase samples at every SNR, add:

```bash
--common_random_numbers --seed 777000
```

## Multi-seed experiment

The following runs both architectures for training seeds `0 1 2`, evaluates
each checkpoint on evaluation seeds `777000 888000`, keeps each evaluation
set fixed across all SNR points, and writes a mean/std/SEM summary:

```bash
bash scripts/run_sionna_multiseed.sh | tee runs/sionna_multiseed.log
```

The defaults can be overridden without editing the script:

```bash
TRAIN_SEEDS="0 1 2" EVAL_SEEDS="777000" NUM_EVAL=8192 bash scripts/run_sionna_multiseed.sh
```

To retrain every seed over `-10...20 dB` without reusing the existing
`-5...20 dB` seed-0 checkpoints:

```bash
SNR_DB_MIN=-10 SNR_DB_MAX=20 USE_EXISTING_SEED0=0 RUN_ROOT=runs/sionna_multiseed_m10_p20 bash scripts/run_sionna_multiseed.sh | tee runs/sionna_multiseed_m10_p20.log
```

Existing `best.pt` files are skipped by default. Set `SKIP_TRAINED=0` to
force retraining. Seed 0 reuses the already trained
`runs/sionna_single_h64_hc32` and `runs/sionna_complex_h32_hc64_l3`
checkpoints by default; set `USE_EXISTING_SEED0=0` to create it again under
the multi-seed run directory.

To add the zero-order P/N0-gated SingleBranch model to an existing multi-seed
run, set `INCLUDE_N0_GATE=1`. Existing baseline checkpoints are skipped and
only missing gated checkpoints are trained:

```bash
SNR_DB_MIN=-10 SNR_DB_MAX=20 USE_EXISTING_SEED0=0 INCLUDE_N0_GATE=1 RUN_ROOT=runs/sionna_multiseed_m10_p20 bash scripts/run_sionna_multiseed.sh | tee runs/sionna_n0_gate_m10_p20.log
```

To isolate whether the trunk-conditioning gain comes from `P` or `N0`, run
the equal-parameter `P-only` and `N0-only` variants. Existing baseline and
`P+N0` checkpoints/evaluations are reused:

```bash
SNR_DB_MIN=-10 SNR_DB_MAX=20 USE_EXISTING_SEED0=0 INCLUDE_GATE_ABLATIONS=1 RUN_ROOT=runs/sionna_multiseed_m10_p20 bash scripts/run_sionna_multiseed.sh | tee runs/sionna_gate_ablation_m10_p20.log
```

## Current boundary

This stage uses an ideal frequency-domain OFDM channel without waveform-level
ICI/ISI. It supports full-symbol and comb DMRS patterns. Channel coding,
waveform modulation with cyclic prefix, LMMSE estimation/equalization,
higher-order QAM, and MIMO are intentionally left for later migration stages.

## 5G NR LDPC and BLER

`Sionna5GLDPCBatchGenerator` fills every data RE with one rate-matched 5G NR
LDPC codeword. For the default 14x72 grid with two full pilot symbols and
QPSK, `N=1728`; at rate 1/2, `K=864`. The sampled axis is Eb/N0 and noise is
computed with Sionna's `ebnodb2no`, including resource-grid overhead.

The neural receivers remain trained on uncoded random bits, following the
official Sionna tutorial. LDPC encoding/decoding is enabled only for BLER
evaluation. Sionna's decoder expects `log p(1)/p(0)`, which is already the
logit convention used by this project.

Single-checkpoint smoke evaluation:

```bash
python -m evaluation.eval_bler_sionna --checkpoint runs/sionna_multiseed_m10_p20/gated_seed0/best.pt --ebno_list=0,1,2,3,4,5,6,7,8 --coderate 0.5 --decoder_iterations 20 --batch_size 64 --target_block_errors 100 --max_blocks 20000 --seed 777000 --common_random_numbers --out_csv runs/sionna_ldpc_r050/gated_seed0.csv
```

Multi-seed comparison of No-interaction, ungated SingleBranch, and P+N0-gated
SingleBranch:

```bash
bash scripts/run_sionna_ldpc_bler.sh | tee runs/sionna_ldpc_r050.log
```

The summary is written to
`runs/sionna_ldpc_r050/bler_multiseed_summary.csv`. Each LDPC codeword is
counted as one block; this stage does not yet add transport-block CRC or code
block segmentation.
