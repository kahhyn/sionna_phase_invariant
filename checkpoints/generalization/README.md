# Published generalization checkpoints

This directory contains the commonly used checkpoints for the normalized
TDL-mix and UMi/UMa-mix generalization experiments. The checkpoints are small
enough to be stored directly in Git; generated runs and intermediate
checkpoints remain ignored.

## Layout

```text
generalization/
├── tdl_mix_normalized/
│   ├── single_branch_n0_gate_seed{0,1,2}.pt
│   └── strict_matched_complex_p_n0_gate_seed{0,1,2}.pt
└── umi_uma_mix_normalized/
    ├── single_branch_n0_gate_seed{0,1,2}.pt
    └── strict_matched_complex_p_n0_gate_seed{0,1,2}.pt
```

- `single_branch_n0_gate` is model A.
- `strict_matched_complex_p_n0_gate` is model C.
- Each checkpoint is the validation-best checkpoint from a 50-epoch run with
  10,000 training samples, 2,000 validation samples, batch size 64, and the
  training SNR range -10 to 20 dB.
- The TDL mix is balanced over TDL-A through TDL-E and RMS delay spreads
  10/30/100/300 ns.
- The urban mix is balanced over normalized UMi and UMa uplink channels with
  pathloss and shadow fading disabled.

Use `SHA256SUMS` to verify file integrity:

```bash
cd checkpoints/generalization
sha256sum -c SHA256SUMS
```

See [`../../docs/TRAINING_EVALUATION_GUIDE.md`](../../docs/TRAINING_EVALUATION_GUIDE.md)
for training and evaluation commands.
