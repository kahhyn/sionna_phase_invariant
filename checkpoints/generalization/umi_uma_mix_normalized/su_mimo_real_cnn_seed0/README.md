# SU-MIMO real-CNN checkpoints

These checkpoints were trained with `training.train_su_mimo` using:

- model: `su_mimo_real_cnn`;
- source profile: `configs/channel_profiles/umi_uma_mix_normalized.json`;
- two transmit layers;
- fixed train/validation phase;
- seed 0;
- 10,000 streaming training frames per epoch and 2,000 validation frames;
- 130 epochs with warmup, cosine decay, and a 30-epoch constant tail.

Files:

- `su_mimo_real_cnn_rx2_seed0.pt`: 2Rx, selected at epoch 127,
  204,656 parameters, SHA-256
  `9886392c34db8f23ecd0ecb9ebfcbc273e4266a67ce9931da2e85f98fddf13b4`.
- `su_mimo_real_cnn_rx16_seed0.pt`: 16Rx, selected at epoch 129,
  220,735 parameters, SHA-256
  `d80e0346cbd730055ff3abd90f2281bc07959b76de2decad9de68ea9ec9cd954`.

Both files can be evaluated directly with
`python -m evaluation.eval_bler_su_mimo --checkpoint <file> ...`.
