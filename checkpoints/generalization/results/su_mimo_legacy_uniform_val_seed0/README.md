# Legacy SU-MIMO phase-sensitive checkpoint, seed 0

These artifacts preserve the historical checkpoint originally stored under
`runs/su_mimo_2x2_seed0/`. Despite that directory name, the resolved model has
two layers and **16 receive antennas**, not two receive antennas.

The model is `su_mimo_phase_sensitive` with 220,755 trainable parameters. It
was trained for 50 epochs on `tdl_mix_normalized`, with fixed-phase training,
uniform-phase validation, 10,000/2,000 training/validation samples, batch size
64, and an SNR range of -5 to 20 dB. The published checkpoint is the
validation-BCE minimum at epoch 49.

This run is archived separately from `su_mimo_tdl_mix_rx_ablation_seed0`
because that comparison uses fixed-phase validation for every model. Selecting
this historical checkpoint on uniform-phase validation makes it unsuitable as
a drop-in control for a zero-shot uniform-phase comparison.

The inference checkpoint excludes optimizer state and embedded history. The
training history and resolved configuration are preserved here. Existing
`runs/su_mimo_2x2_seed0/bler*.csv` files are intentionally excluded because
they record `checkpoint_epoch=30` and therefore do not correspond to the
published epoch-49 checkpoint.
