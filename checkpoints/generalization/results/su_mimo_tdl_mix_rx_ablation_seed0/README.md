# SU-MIMO TDL-mix Rx/readout ablation, seed 0

These artifacts accompany the four inference checkpoints under
`../../../tdl_mix_normalized/su_mimo_rx_ablation_seed0/`.

All runs use two layers, fixed total transmit power 1, the unchanged
`tdl_mix_normalized` profile for training and validation, fixed common phase,
10,000 training samples, 2,000 validation samples, 50 epochs, batch size 64,
and SNR sampled uniformly from -5 to 20 dB. Checkpoint selection minimizes
fixed-phase validation BCE.

The comparison isolates the invariant versus phase-sensitive readout within
each receive-antenna count. The network widths are unchanged between 2 and 16
receive antennas, so the input projection raises the parameter count from
204,599 to 220,755. The four runs use seed 0 only. Three validation-best points
occur at the final epoch, so convergence and run-to-run stability remain open.
These artifacts are therefore `completed_exploratory`, not publication-ready.

Published `.pt` files intentionally exclude optimizer state and embedded
history. The CSV histories and resolved JSON configurations in this directory
preserve the training curves, exact commands, code revision, and runtime
configuration.
