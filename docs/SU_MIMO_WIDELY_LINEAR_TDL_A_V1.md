# SU-MIMO widely-linear TDL-A screen

This screen tests whether the anti-linear branch in
`W*z + V*conj(z)` closes the receive-antenna-dependent gap between the
standard complex CNN and the ordinary real CNN.

The frozen contract is
`configs/experiment_matrices/su_mimo_widely_linear_tdl_a_v1.json`.

## Controlled comparison

- systems: two layers with 2Rx or 16Rx;
- source/test domain: normalized TDL-A, 10 ns;
- training corpus: streaming, with 10,000 newly generated frames per epoch;
- train/validation/evaluation phase: uniform;
- methods: standard complex, widely-linear complex, and real CNN;
- checkpoint selection: minimum fixed-seed source-validation BCE;
- train seed: 0; evaluation seed: 777000.

Default real-parameter counts are:

| Rx | standard complex | widely-linear complex | real CNN |
| ---: | ---: | ---: | ---: |
| 2 | 204,599 | 204,459 | 204,656 |
| 16 | 220,755 | 220,567 | 220,735 |

The widely-linear model keeps the existing amplitude gates, zero-order
conditioning, layer interaction, and real LLR head. Only the convolution
family and automatically resolved hidden/readout widths differ.

## Training

```bash
SEED=0 \
bash scripts/run_su_mimo_widely_linear_tdl_a.sh \
  |& tee runs/su_mimo_widely_linear_tdl_a_v1_train.log
```

The script calls `python -m training.train_su_mimo` for every method and Rx
count. `EPOCHS`, `NUM_TRAIN`, `NUM_VAL`, `RX_LIST`, `MODELS`, `PROFILE`, and
`OUTPUT_ROOT` can be overridden through environment variables.

## Fast BLER screen

```bash
TRAIN_SEED=0 EVAL_SEED=777000 \
bash scripts/eval_su_mimo_widely_linear_tdl_a.sh \
  |& tee runs/su_mimo_widely_linear_tdl_a_v1_eval.log
```

The default points are 5/7/9 dB for 2Rx and -7/-5/-3 dB for 16Rx, with 100
target block errors and 5,000 maximum blocks. Override `EBNO_LIST_RX2`,
`EBNO_LIST_RX16`, `TARGET_ERRORS`, or `MAX_BLOCKS` for a longer evaluation.
