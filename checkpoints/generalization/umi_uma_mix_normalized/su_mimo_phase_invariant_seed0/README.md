# SU-MIMO phase-invariant checkpoints

These checkpoints were trained with `training.train_su_mimo` using the same
control settings as the UMI+UMA phase-canonical, phase-sensitive, and real-CNN
receivers:

- model: `su_mimo_phase_invariant`;
- source profile: `configs/channel_profiles/umi_uma_mix_normalized.json`;
- two transmit layers;
- fixed train/validation phase;
- streaming data with 10,000 training frames per epoch;
- 2,000 validation frames;
- 130 epochs, batch size 64;
- AdamW at `1e-3`, 5-epoch warmup, cosine decay to `1e-5`, and a 30-epoch constant tail;
- seed 0, training-generator seed 0, validation-generator seed 100000.

Files:

- `su_mimo_phase_invariant_rx2_seed0.pt`: 2Rx, selected at epoch 130,
  validation BCE 0.2499213197, 204,599 parameters, SHA-256
  `4f965cd9ba1461a48610a99992fc5c492ac51a8fbf4ca137e1f528631e34ed4e`.
- `su_mimo_phase_invariant_rx16_seed0.pt`: 16Rx, selected at epoch 130,
  validation BCE 0.1192794854, 220,755 parameters, SHA-256
  `7821752a3c806ec78d70738de0dec3ad262a7981476c9ce8e9d82e2f9a0afe40`.

Both best checkpoints occur at the final training epoch. They are valid for the
matched 130-epoch comparison, but the boundary optimum indicates that a small
additional optimization gain cannot be ruled out.

Evaluate with `python -m evaluation.eval_bler_su_mimo --checkpoint <file> ...`.
