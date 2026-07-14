# Phase-Invariant AI Receiver Prototype

This is a minimal PyTorch project for testing a U(1)-invariant neural receiver idea.

The first version uses a synthetic SISO-OFDM/QPSK dataset:

```text
Y[k,l] = H[k,l] X[k,l] + N[k,l]
```

The model input is:

```text
Y, H_hat, P, N0
```

The target is bit-level logits/LLRs on data REs. DMRS REs are masked out in the BCE and BER.

## Project structure

```text
phase_invariant_receiver/
├── models/
│   ├── baseline_cnn.py
│   ├── complex_layers.py
│   └── phase_invariant_net.py
├── data/
│   └── ofdm_dataset.py
├── training/
│   ├── train.py
│   ├── train_sionna.py
│   └── finetune.py
├── evaluation/
│   ├── eval_ber.py
│   ├── eval_ber_sionna.py
│   ├── eval_bler_sionna.py
│   ├── eval_invariance.py
│   └── eval_phase_sweep.py
├── experiments/
│   └── continual_learning/
├── scripts/
│   └── run_sionna_multiseed.sh
├── tools/
│   └── compute_model_size.py
├── utils/
│   ├── batching.py
│   └── metrics.py
├── config.py
└── requirements.txt
```

## Models

### 1. `real_imag_cnn`

Ordinary CNN baseline.

Input channels:

```text
Re(Y), Im(Y), Re(H_hat), Im(H_hat), P, log(N0)
```

This model does not structurally guarantee common phase invariance.

### 2. `physical_cnn`

CNN with hand-crafted physical invariant features.

Input channels:

```text
Re(conj(H_hat)Y), Im(conj(H_hat)Y), |H_hat|^2, |Y|^2, P, log(N0)
```

This is an important baseline because it obtains phase invariance by feature construction.

### 3. `phase_invariant`

Proposed U(1)-invariant network.

Structure:

```text
Y, H_hat                         charge +1
conj(Y), conj(H_hat)              charge -1
      ↓
separate complex equivariant branches
      ↓
charge +1 × charge -1
      ↓
charge 0 feature
      ↓
concat P, log(N0)
      ↓
real-valued LLR head
```

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the structural invariance test with random initialization:

```bash
python -m evaluation.eval_invariance --model real_imag_cnn
python -m evaluation.eval_invariance --model physical_cnn
python -m evaluation.eval_invariance --model phase_invariant
```

Expected result:

```text
real_imag_cnn: max |ΔLLR| is usually non-zero
physical_cnn: max |ΔLLR| near numerical precision
phase_invariant: max |ΔLLR| near numerical precision
```

Train models:

```bash
python -m training.train --model real_imag_cnn --train_phase_mode fixed --val_phase_mode uniform --save_dir runs/real_fixed
python -m training.train --model physical_cnn --train_phase_mode fixed --val_phase_mode uniform --save_dir runs/physical_fixed
python -m training.train --model phase_invariant --train_phase_mode fixed --val_phase_mode uniform --save_dir runs/invariant_fixed
```

Evaluate a trained model:

```bash
python -m evaluation.eval_ber --model phase_invariant --checkpoint runs/invariant_fixed/best.pt --phase_mode uniform --out_csv invariant_ber.csv
```

## Important experiment settings

To highlight the value of structural phase invariance, use a phase distribution shift:

```text
training: phase_mode = fixed
testing:  phase_mode = uniform
```

If training already uses `phase_mode=uniform`, the ordinary real/imag CNN may learn approximate invariance from data augmentation.

## Current limitations

This first version is intentionally simple.

It currently uses:

```text
SISO-OFDM
QPSK
synthetic smooth Rayleigh-like channel
H_hat = H + estimation noise
AWGN
```

Next extensions can include:

```text
real DMRS-based LS estimation and interpolation
16QAM / 64QAM
multi-user MIMO
interference covariance estimation
local phase noise or CFO
user/layer permutation-equivariant modules
```


## DMRS-LS channel estimation mode

The dataset supports two H_hat modes:

```bash
--h_hat_mode oracle_noisy
--h_hat_mode dmrs_ls_interp
```

`oracle_noisy` keeps the old simplified setting:

```text
H_hat = H + E
```

`dmrs_ls_interp` estimates the channel on DMRS REs:

```text
H_LS[pilot] = Y[pilot] / X_pilot[pilot]
```

Then it fills non-DMRS OFDM symbols by linear interpolation along the time dimension.

Example:

```bash
python -m training.train --model phase_invariant --h_hat_mode dmrs_ls_interp --train_phase_mode fixed --val_phase_mode uniform --save_dir runs/invariant_dmrs
python -m training.train --model physical_cnn --h_hat_mode dmrs_ls_interp --train_phase_mode fixed --val_phase_mode uniform --save_dir runs/physical_dmrs
```

This first DMRS version uses full-DMRS OFDM symbols across all subcarriers, so only time interpolation is required. A more realistic next step is comb-type DMRS in frequency, which requires both time and frequency interpolation.
