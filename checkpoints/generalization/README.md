# Published generalization checkpoints

This directory contains validation-best inference checkpoints and compact
result artifacts for the normalized channel-generalization experiments. The
checkpoints are small enough to be stored directly in Git; optimizer states,
`last.pt`, raw generated runs, and smoke-test checkpoints remain ignored.

## Layout

```text
generalization/
├── capacity_screen_umi/
│   ├── 20k/{single_branch_n0_gate,real_imag_cnn}_seed0.pt
│   └── 50k/{single_branch_n0_gate,real_imag_cnn}_seed0.pt
├── tdl_a_10_30_100_mix_normalized/
│   ├── single_branch_n0_gate_seed{0,1,2}.pt
│   └── real_imag_cnn_seed{0,1,2}.pt
├── tdl_mix_normalized/
│   ├── single_branch_n0_gate_seed{0,1,2}.pt
│   ├── strict_matched_complex_p_n0_gate_seed{0,1,2}.pt
│   └── real_imag_cnn_seed{0,1,2}.pt
├── umi_normalized/
│   ├── single_branch_n0_gate_seed{0,1,2}.pt
│   └── real_imag_cnn_seed{0,1,2}.pt
├── umi_uma_mix_normalized/
│   ├── single_branch_n0_gate_seed{0,1,2}.pt
│   ├── strict_matched_complex_p_n0_gate_seed{0,1,2}.pt
│   └── real_imag_cnn_seed{0,1,2}.pt
└── results/
    ├── realimag_matched_four_domains/
    ├── single_branch_missing_controls/
    └── capacity_screen_umi/
```

- `single_branch_n0_gate` is model A.
- `strict_matched_complex_p_n0_gate` is model C.
- `real_imag_cnn` is the depth- and parameter-matched ordinary real-valued
  baseline. Its formal configuration has 204,558 trainable parameters versus
  204,599 for model A.
- Formal checkpoints are validation-best checkpoints from a 50-epoch run with
  10,000 training samples, 2,000 validation samples, batch size 64, and the
  training SNR range -10 to 20 dB.
- Capacity-screen checkpoints use one UMi training seed, 30 epochs, 5,000
  training samples per epoch, and matched sizes of approximately 20k or 50k
  parameters. They are exploratory rather than publication-ready evidence.
- The TDL mix is balanced over TDL-A through TDL-E and RMS delay spreads
  10/30/100/300 ns.
- The urban mix is balanced over normalized UMi and UMa uplink channels with
  pathloss and shadow fading disabled.
- `results/` contains aggregate BER matrices, training histories, commands,
  environment manifests, and the compact UMi/UMa capacity-screen CSVs. Raw
  per-scenario formal-evaluation CSVs remain under ignored `runs/` directories.

Use `SHA256SUMS` to verify file integrity:

```bash
cd checkpoints/generalization
sha256sum -c SHA256SUMS
```

See [`../../docs/TRAINING_EVALUATION_GUIDE.md`](../../docs/TRAINING_EVALUATION_GUIDE.md)
for training and evaluation commands.
