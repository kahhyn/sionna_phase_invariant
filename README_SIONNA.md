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
waveform modulation with cyclic prefix, LMMSE estimation/equalization, and
higher-order QAM are intentionally left for later migration stages. The
SU-MIMO path below is a fixed-topology first stage rather than a general
precoding or variable-rank implementation.

### Minimal 2x2 SU-MIMO smoke stage

The repository now contains an isolated first MIMO migration stage with one
user, two spatial layers, two receive antennas, identity layer-to-antenna
mapping, FDM-orthogonal DMRS, and fixed total user transmit power. Data power
is split equally between the two layers. The accompanying receiver preserves
common-phase invariance and layer-permutation equivariance through shared
complex backbones and an equivariant masked-mean message-passing layer.

Run its data, forward/backward, checkpoint, phase, permutation, tiny-overfit,
training-resume, and BER-export checks with:

```bash
python -m unittest tests.test_su_mimo_smoke tests.test_su_mimo_train_eval -v
```

A complete default 2x2 training run is:

```bash
python -m training.train_su_mimo \
  --model su_mimo_phase_invariant \
  --num_layers 2 --num_rx_ant 2 --total_tx_power 1.0 \
  --train_channel_profile configs/channel_profiles/tdl_mix_normalized.json \
  --val_channel_profile configs/channel_profiles/tdl_mix_normalized.json \
  --train_phase_mode fixed --val_phase_mode uniform \
  --snr_db_min -5 --snr_db_max 20 \
  --num_train 10000 --num_val 2000 --epochs 50 --batch_size 64 \
  --seed 0 --save_dir runs/su_mimo_2x2_seed0
```

The default widths (`hidden_complex=32`, `zero_real=22`, `hidden_real=66`)
give exactly 204,599 trainable parameters, matching the existing
`tdl_mix_normalized` SingleBranch and strict matched checkpoints. Select the
equal-parameter phase-sensitive control with
`--model su_mimo_phase_sensitive`. Both models keep layer-permutation
equivariance; only the final readout's common-phase invariance differs.

`best.pt` is selected by minimum validation BCE. `last.pt`, `history.csv`, and
`resolved_config.json` are also saved. Resume from the full model and AdamW
state by setting a new total target epoch; data and model settings are restored
from the checkpoint:

```bash
python -m training.train_su_mimo \
  --resume_checkpoint runs/su_mimo_2x2_seed0/last.pt \
  --epochs 100
```

Evaluate the validation-selected checkpoint with aggregate and per-layer BER,
exact error counts, and Wilson 95% intervals:

```bash
python -m evaluation.eval_ber_su_mimo \
  --checkpoint runs/su_mimo_2x2_seed0/best.pt \
  --phase_mode uniform \
  --snr_list=-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20 \
  --num_samples 4096 --batch_size 128 --seed 777000 \
  --common_random_numbers \
  --eval_channel_profile configs/channel_profiles/tdl_mix_normalized.json \
  --out_csv runs/su_mimo_2x2_seed0/ber.csv
```

The same existing profile files can be used unchanged for SU-MIMO TDL, UMi,
UMa, and mixed-profile runs. Select one profile component for a fixed-scenario
test with `--eval_component_id`.

For 5G NR LDPC evaluation, each layer carries one independent rate-matched
codeword. The reported frame BLER counts a user frame as erroneous when any
layer fails; the companion CSV reports per-layer BLER.

```bash
python -m evaluation.eval_bler_su_mimo \
  --checkpoint runs/su_mimo_2x2_seed0/best.pt \
  --ebno_list=-2,0,2,4,6,8 \
  --coderate 0.5 --decoder_iterations 20 \
  --batch_size 64 --target_block_errors 100 --max_blocks 20000 \
  --seed 777000 --common_random_numbers \
  --eval_channel_profile configs/channel_profiles/tdl_mix_normalized.json \
  --out_csv runs/su_mimo_2x2_seed0/bler.csv
```

The implementation is in `data/sionna_su_mimo_generator.py`,
`models/su_mimo_invariant_net.py`, `training/train_su_mimo.py`, and
`evaluation/eval_ber_su_mimo.py`, and `evaluation/eval_bler_su_mimo.py`. The
initial setup does not yet model precoding, CDM DMRS, variable layer counts
during generation, or a classical MIMO evaluation baseline. UMi/UMa profiles
use the configured multi-antenna Sionna panel arrays and therefore include
their geometry-derived spatial channel behavior.

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
