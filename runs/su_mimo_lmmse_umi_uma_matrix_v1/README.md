# Seven-configuration SU-MIMO LMMSE matrix

This directory contains the completed LS-CSI LMMSE and perfect-CSI LMMSE
evaluation for seven spatial configurations, evaluated separately on normalized
UMi and UMa channels.

## Fixed evaluation contract

- Spatial configurations: 2 layers with 2/4/8/16 Rx antennas, and 4 layers with
  4/8/16 Rx antennas.
- Receivers: `lmmse_ls` and `lmmse_perfect`.
- Eb/N0: 3, 5, 7, 9, 11, and 13 dB.
- 5G LDPC code rate 1/2, 20 decoder iterations, uniform common phase.
- Evaluation seed 777000 with common random numbers.
- Stop each point after 500 user-frame errors or 20,000 user frames.
- A user frame is in error if any layer's codeword fails.

The frozen machine-readable contract is in `manifest/experiment_contract.json`.
`lmmse_perfect` uses the true channel with zero estimation-error variance; it is
an oracle linear detector, not an ML detector or a capacity bound.

## Files

- `aggregate_bler.csv`: all 168 configuration/profile/receiver/Eb-N0 rows.
- `aggregate_per_layer.csv`: all 480 per-layer rows.
- `<spatial>/<profile>/<receiver>/`: raw aggregate and per-layer CSVs, exact
  command, and evaluation log for each of the 28 curves.
- `manifest/`: frozen contract, checkpoint SHA-256 hashes, environment, and
  start/completion timestamps.
- `formal_run.log`: complete top-level runner output.

At 13 dB, the two zero-error observations are 2-layer/16Rx perfect LMMSE on UMi
and UMa (0 errors in 20,000 frames). They must be read using the stored Wilson
95% upper bound of approximately 1.92e-4, not as proof that the true BLER is zero.
