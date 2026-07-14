# TDL / UMi / UMa channel generalization

This experiment is the normalized small-scale phase defined in
`docs/TDL_UMI_UMA_GENERALIZATION_PLAN.md`. Pathloss and shadow fading are off,
the channel is normalized, and N0 is derived from each sample's received
signal power. Results therefore measure channel-structure generalization, not
coverage or link-budget generalization.

Channel profiles are batch-level distributions. The generalization suite
evaluates all 20 fixed TDL model/delay combinations plus fixed UMi and UMa
domains. `scripts/run_channel_generalization.sh` trains checkpoints, evaluates
the suite with common random numbers, and writes pooled BER/BLER matrices.

Use `configs/channel_suites/smoke_normalized.json` for a quick three-domain
runner check before launching the complete 22-domain suite.

Do not enable pathloss in these profiles. A realistic coverage experiment
requires a separate absolute transmit-power and thermal-noise implementation.
