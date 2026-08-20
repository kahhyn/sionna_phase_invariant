# SU-MIMO phase-sensitive checkpoints

These `su_mimo_phase_sensitive` checkpoints use the matched UMI+UMA training
contract: two layers, fixed train/validation phase, 10,000 streaming training
frames per epoch, 2,000 validation frames, 130 epochs, batch size 64, AdamW at
`1e-3`, 5-epoch warmup, cosine decay to `1e-5`, a 30-epoch constant tail, and
seed 0 (train/validation generator seeds 0/100000).

- `su_mimo_phase_sensitive_rx2_seed0.pt`: epoch 129, 204,599 parameters,
  SHA-256 `1a784511f4f4046752547acead434df18ce6c4a03d978bd71492a384c710bd53`.
- `su_mimo_phase_sensitive_rx16_seed0.pt`: epoch 128, 220,755 parameters,
  SHA-256 `cdb0f3898d36bc8e93536c8420808964d865fe78af7488455b1e84b34a7568ca`.

Evaluate with `python -m evaluation.eval_bler_su_mimo --checkpoint <file> ...`.
