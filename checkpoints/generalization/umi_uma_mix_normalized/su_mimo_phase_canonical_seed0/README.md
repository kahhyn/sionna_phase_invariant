# SU-MIMO phase-canonical checkpoints

These `su_mimo_phase_canonical` checkpoints use the matched UMI+UMA training
contract: two layers, fixed train/validation phase, 10,000 streaming training
frames per epoch, 2,000 validation frames, 130 epochs, batch size 64, AdamW at
`1e-3`, 5-epoch warmup, cosine decay to `1e-5`, a 30-epoch constant tail, and
seed 0 (train/validation generator seeds 0/100000).

- `su_mimo_phase_canonical_rx2_seed0.pt`: epoch 129, 204,599 parameters,
  SHA-256 `e10e13c26fa56f3583a3abac7f185515c8ad5144ceb15ee12049cf4feaed75bb`.
- `su_mimo_phase_canonical_rx16_seed0.pt`: epoch 129, 220,755 parameters,
  SHA-256 `dfe0433b4c39424365ca0b723a51e0069a4c2c89bef62911ca74a040d4c35bde`.

Evaluate with `python -m evaluation.eval_bler_su_mimo --checkpoint <file> ...`.
