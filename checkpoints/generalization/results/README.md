# Published result artifacts

This directory contains compact evidence accompanying the published inference
checkpoints. It intentionally excludes raw per-scenario formal CSVs and all
smoke-test outputs.

- `realimag_matched_four_domains/` contains the four RealImagCNN aggregate BER
  matrices, validation histories, commands, and environment manifests.
- `single_branch_missing_controls/` contains the completed TDLA and UMi model-A
  controls needed to pair with those RealImagCNN runs.
- `capacity_screen_umi/` contains the exploratory 20k/50k UMi training histories
  and paired UMi/UMa evaluation CSVs.
- `su_mimo_tdl_mix_rx_ablation_seed0/` contains the four 2-layer SU-MIMO
  training histories, resolved configurations, and validation-best checkpoint
  summary for the 2/16-Rx phase-invariant and phase-sensitive models.
- `su_mimo_legacy_uniform_val_seed0/` preserves the historical 2-layer, 16-Rx
  phase-sensitive seed-0 run selected using uniform-phase validation. Its stale
  epoch-30 BLER CSVs are excluded from the epoch-49 checkpoint archive.

The capacity screen uses one training seed, 30 epochs, 5,000 online training
samples per epoch, one evaluation seed, and SNR points `-10,0,10,16,20`. It is
evidence of a trend, not a stability estimate. At 20 dB, the observed BERs were:

| Capacity | Scenario | RealImagCNN | Single branch |
| --- | --- | ---: | ---: |
| 50k | UMi | 0.00180788 | 0.00134390 |
| 50k | UMa OOD | 0.00903998 | 0.00502918 |
| 20k | UMi | 0.00255669 | 0.00140324 |
| 20k | UMa OOD | 0.01113157 | 0.00560054 |

`realimag_vs_single_branch.csv` was generated before the later TDLA/UMi
single-branch controls completed, so its paired comparison rows cover the
TDL-mix and UMi/UMa-mix training domains. The missing-control aggregate matrices
are preserved separately instead of silently regenerating that historical file.
