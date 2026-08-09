# Phase-invariance SISO/SU-MIMO matrix checkpoints

This directory contains the already-trained SU-MIMO checkpoints required by
`configs/experiment_matrices/phase_invariance_siso_mimo_v1.json`. Existing SISO
checkpoints remain available as historical evidence:

- narrow TDL-A/10 ns: `checkpoints/continual_source/`;
- broad TDL mix: `checkpoints/generalization/tdl_mix_normalized/`.

The four files under `mimo_tdl_mix/` were copied byte-for-byte from their
completed run directories. For each Rx setting, canonical and sensitive
receivers have identical PHY settings, model configuration, parameter count,
training budget, optimizer schedule, source-validation policy, and seed. Only
the declared model/readout differs.

| Checkpoint | Parameters | Best epoch | Source run |
|---|---:|---:|---|
| `su_mimo_phase_canonical_rx2_seed0.pt` | 204,599 | 130 | `runs/su_mimo_tdl_mix_phase_canonical_tail30_seed0/rx2_phase_canonical/best.pt` |
| `su_mimo_phase_sensitive_rx2_seed0.pt` | 204,599 | 128 | `runs/su_mimo_tdl_mix_rx_ablation_warmup_cosine_tail30_seed0/rx2_phase_sensitive/best.pt` |
| `su_mimo_phase_canonical_rx16_seed0.pt` | 220,755 | 127 | `runs/su_mimo_tdl_mix_phase_canonical_tail30_seed0/rx16_phase_canonical/best.pt` |
| `su_mimo_phase_sensitive_rx16_seed0.pt` | 220,755 | 129 | `runs/su_mimo_tdl_mix_rx_ablation_warmup_cosine_tail30_seed0/rx16_phase_sensitive/best.pt` |

Those SISO `best.pt` files were selected on uniform-phase validation and are not
used by the formal matrix, where uniform phase is a target OOD condition. The
matrix runner trains four new fixed-validation SISO checkpoints per seed.

The matrix also needs four TDL-A/10 ns SU-MIMO networks per requested seed:
canonical/sensitive at 2Rx and 16Rx. The matrix runner trains these missing
networks automatically in `--mode all` or `--mode train`.

SHA-256:

```text
fcfd13f6978b002bd67ac0b1a09a671e2723e477eb3e181eb988d45aa9d79017  mimo_tdl_mix/su_mimo_phase_canonical_rx16_seed0.pt
db7cf3386e932c7769d02f41a43c5542f4af6ff9127bb782e5743249372b0464  mimo_tdl_mix/su_mimo_phase_canonical_rx2_seed0.pt
ea1f281ae47dd7bcfaad0b4067dd04f99421dd4df15c909f106ae53009201215  mimo_tdl_mix/su_mimo_phase_sensitive_rx16_seed0.pt
f260fbed09cad7084e6c33a89b5f623629f6d2d49c0cd04bd0bf7ffe42c782c1  mimo_tdl_mix/su_mimo_phase_sensitive_rx2_seed0.pt
```
