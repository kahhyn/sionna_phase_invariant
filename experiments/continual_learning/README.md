# Oracle few-shot domain adaptation

This first continual-learning experiment compares the native receivers only:

- `single_branch_n0_gate`: exactly common-phase invariant (A)
- `strict_matched_complex_p_n0_gate`: phase-sensitive and parameter-matched (C)

The source checkpoint uses TDL-A with 10 ns delay spread. Each run adapts all
receiver parameters with simulator ground-truth bits after shifting the delay
spread to 50, 100, or 300 ns. Every budget restarts from the source checkpoint.
A/C receive identical deterministic adaptation and evaluation samples.

The experiment writes:

- `fewshot_per_snr.csv`: BER/BCE for every SNR, domain, and budget
- `fewshot_summary.csv`: target recovery, source forgetting, and per-run N90
- `fewshot_multiseed_summary.csv`: mean/std over source training seeds
- `experiment_config.json`: exact arguments and source-domain configuration

Use `scripts/run_sionna_fewshot_adaptation.sh` for the formal multi-seed run.
Set `FORCE=1` to replace existing results or `SAVE_CHECKPOINTS=1` to retain all
adapted checkpoints. Checkpoints are not saved by default to avoid disk growth.

The six source checkpoints required by the default three-seed run are tracked
under `checkpoints/continual_source/`. Set `SOURCE_ROOT` only when evaluating a
different set of pretrained source receivers.
