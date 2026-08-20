# UMi/UMa SU-MIMO scale matrix checkpoints (seed 0)

This manifest archives the best validation-selected checkpoints from the
single-seed UMi/UMa scale experiment run under:

`runs/su_mimo_umi_uma_scale_matrix_tail30_seed0`

The matched training contract used `umi_uma_mix_normalized`, streaming channel
realizations, fixed train/validation phase, 10,000 training frames and 2,000
validation frames per epoch, batch size 64, 130 epochs, AdamW at `1e-3`, a
5-epoch warmup, cosine decay to `1e-5`, a 30-epoch constant tail, training seed
0, and validation-generator seed 100000. Total transmit power was fixed at 1.0.

The matrix contains L2-R2, L2-R4, L2-R8, L4-R4, L4-R8, and L4-R16 for the
real-CNN, phase-invariant, and phase-sensitive receivers. L2-R16 was not
retrained because matched checkpoints had already been archived. Older files
whose names omit `layer2` remain preserved; the explicit `layer*_rx*` files are
the checkpoints produced by this matrix run and do not overwrite them.

| Model | Layers | Rx | Best epoch | Validation BCE | Parameters | SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| phase-invariant | 2 | 2 | 129 | 0.2436435143 | 204599 | `86c4509f39393cdeee582f3e7e10273cf2e5819837dd31ff73b3a755500bb08b` |
| phase-invariant | 2 | 4 | 129 | 0.1720067865 | 206907 | `c3baf1e907c53903f058410a3af83abc7bb7e04eb0aa1258f04cfd801ec2e452` |
| phase-invariant | 2 | 8 | 129 | 0.1180060069 | 211523 | `56eb2f1aec0ff785f227b750c32eceb3c4975804ef430a17c428a95fb7182f23` |
| phase-invariant | 4 | 4 | 129 | 0.3751112597 | 206907 | `5dc6092ba36b72fe44a3be4842e396f0780685e14efe64557b207101d4e1cf31` |
| phase-invariant | 4 | 8 | 130 | 0.2926437929 | 211523 | `1d84f77f3e8021a0b7e439dd7e1d85e1709b823694026e91763b7760a81d5211` |
| phase-invariant | 4 | 16 | 128 | 0.2578365722 | 220755 | `723fd7b1bfe0298e53e66309f43543943d36649e0d8a454e15135c413e0e9db1` |
| phase-sensitive | 2 | 2 | 129 | 0.2438488230 | 204599 | `a48e6d7e8e55ef30595c032b7534b685f81d46d067d433350c7e4f2d42955024` |
| phase-sensitive | 2 | 4 | 127 | 0.1675987119 | 206907 | `e7dea8aae6dfcea3cdb382374f5e428255a6b3ab34c05b3b3442e841ea99f739` |
| phase-sensitive | 2 | 8 | 128 | 0.1123842809 | 211523 | `87f6e428f4bed96a4835fdcdefb8391afb0f62918cb092d1a8cd079a50446b56` |
| phase-sensitive | 4 | 4 | 128 | 0.3909224028 | 206907 | `68262768e51c66a24eb423df03d038c82d5174434f0ef44eaa6a5c2009b0c09d` |
| phase-sensitive | 4 | 8 | 125 | 0.3135417968 | 211523 | `e66caeb62b7d63f328d3a2384072a3607d00840a326f3ed803595368bc97bee1` |
| phase-sensitive | 4 | 16 | 130 | 0.2636907065 | 220755 | `420a9e43360a01af4b002890cb4ecda31dc66f918e812ff2bd53d1096058c7ec` |
| real CNN | 2 | 2 | 127 | 0.2540650869 | 204656 | `e9f5c303779f8715e18f41c5ef5737513ff6c64effa01141a8379758d0f1546e` |
| real CNN | 2 | 4 | 129 | 0.1660686227 | 206932 | `0633f6f7bca4ed7f2c8db957a8c6fa84c5efdc2a4068a238b22beb8792d09906` |
| real CNN | 2 | 8 | 127 | 0.1080071105 | 211420 | `d759c58fee45d6fde9a879c15817347e0c59d8aa6b672b6a122849dbf160b121` |
| real CNN | 4 | 4 | 130 | 0.3885706945 | 206932 | `2a3fd854cfa85ef9d03dd6b5170849f8a6712bcad562e0e4d972616019d2e3d7` |
| real CNN | 4 | 8 | 128 | 0.3351121176 | 211420 | `2484de82d25107f052db4e8aea39a0520899bfeb4418a94185f48da7b4d74198` |
| real CNN | 4 | 16 | 130 | 0.2754336686 | 220735 | `dac141c8aea51206735b95c40d0d0d45d34edd24d9e4d5f23e34012bb468c7b5` |

All 18 files were reconstructed successfully on CPU with
`utils.checkpoints.load_su_mimo_checkpoint`. Evaluate any file with:

```bash
python -m evaluation.eval_bler_su_mimo --checkpoint <checkpoint.pt> ...
```
