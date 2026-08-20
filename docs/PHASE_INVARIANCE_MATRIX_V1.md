# SISO/SU-MIMO phase-invariance evaluation matrix v1

## Scope

The formal matrix has 30 domain cells:

```text
3 systems × 2 source domains × 5 test domains
```

Systems:

- `siso_1l1rx`;
- `mimo_2l2rx`;
- `mimo_2l16rx`.

Source domains:

- `tdl_a`: TDL-A, 10 ns, 200 Hz (the existing extreme-narrow source);
- `tdl_mix`: TDL A-E, 10/30/100/300 ns, 200 Hz.

Test domains:

- `id_fixed`: source-matched channel and fixed common phase;
- `phase_ood`: source-matched channel and uniform common phase;
- `delay_ood`: TDL-A at 600/1000 ns and uniform phase;
- `doppler_ood`: TDL-A/10 ns at 0/400/800/1200 Hz and uniform phase;
- `quadriga_ood`: external QuaDRiGa CFR and uniform common phase.

The exact scientific contract is
`configs/experiment_matrices/phase_invariance_siso_mimo_v1.json`.

## Checkpoint readiness

For seed 0, four of twelve formal checkpoints are already available:

- all four TDL-mix SU-MIMO checkpoints (two models × 2Rx/16Rx).

The existing SISO checkpoints remain tracked historical evidence, but their
`best.pt` files were selected on uniform-phase validation. Because uniform
phase is a target OOD condition in this contract, reusing them would leak the
target shift into checkpoint selection. The runner therefore trains four new
fixed-train/fixed-validation SISO checkpoints:

- invariant and sensitive on TDL-A/10 ns;
- invariant and sensitive on TDL mix.

It also trains four TDL-A SU-MIMO checkpoints:

- canonical 2Rx;
- sensitive 2Rx;
- canonical 16Rx;
- sensitive 16Rx.

`MODE=all` trains these eight missing formal checkpoints and then evaluates the
selected matrix. Existing tracked checkpoints are never overwritten.

## QuaDRiGa MAT contract

Every MAT file must contain `H_real` and `H_imag` as `float32` arrays.

SISO:

```text
[frame, ofdm_symbol, subcarrier]
```

SU-MIMO default layout:

```text
[frame, rx_antenna, layer, ofdm_symbol, subcarrier]
```

The evaluator converts this to the internal `[frame, layer, rx, symbol,
subcarrier]` contract. Files for 2Rx and 16Rx must come from the corresponding
physical array simulation. A SISO channel repeated over Rx antennas is not a
valid MIMO control.

## Commands

Audit checkpoint readiness without using a GPU:

```bash
MODE=audit DRY_RUN=1 bash scripts/run_phase_invariance_matrix.sh
```

Train missing networks only:

```bash
MODE=train SEEDS=0 bash scripts/run_phase_invariance_matrix.sh
```

Run the full seed-0 matrix, including QuaDRiGa:

```bash
MODE=all \
SEEDS=0 \
EVAL_SEEDS=777000 \
METRICS=bler \
QUADRIGA_SISO_DIR=/path/to/quadriga/siso \
QUADRIGA_MIMO_RX2_DIR=/path/to/quadriga/mimo_rx2 \
QUADRIGA_MIMO_RX16_DIR=/path/to/quadriga/mimo_rx16 \
REQUIRE_QUADRIGA=1 \
bash scripts/run_phase_invariance_matrix.sh
```

Add uncoded BER for the synthetic TDL cells with `METRICS=ber,bler`. QuaDRiGa
is evaluated with LDPC and reports frame BLER, post-LDPC BER, pre-LDPC coded
BER, and BCE in one paired pass.

Results are isolated under:

```text
runs/phase_invariance_matrix_v1/
```

The runner writes the resolved matrix plan, every command/status record,
per-cell manifests, raw CSVs, and aggregate CSVs.

## Interpretation boundary

The Doppler profiles exercise the current frequency-domain channel backend.
They measure time-selective/channel-aging robustness across the OFDM grid, not
waveform-level within-symbol ICI unless the backend is later upgraded to a
time-domain waveform simulation.
