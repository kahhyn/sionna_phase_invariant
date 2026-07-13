# Continual-learning source checkpoints

These checkpoints are the native, non-denoised source receivers used by the
oracle few-shot adaptation experiment. All were trained on Sionna TDL-A with a
10 ns delay spread and an SNR range of -10 to 20 dB.

| Receiver | Seeds | Parameters |
| --- | --- | ---: |
| `single_branch_n0_gate` | 0, 1, 2 | 204,599 |
| `strict_matched_complex_p_n0_gate` | 0, 1, 2 | 204,599 |

SHA-256 checksums:

```text
2860d26d53230772c9582443e147c19d771cb45dce52d0773d7544e4bd733049  single_branch_n0_gate_seed0.pt
401221b70765f40be9f35d89a8d6c89dc231786b46ad30c0e2facbf916867cbd  single_branch_n0_gate_seed1.pt
4bbecacc787ef0ed788f2e515ef9b65c50db4e50e7a97abdf49294a07c3b670d  single_branch_n0_gate_seed2.pt
976f70992300db0a2fadfe202c6290909c7769f95699294278cdcb208313b111  strict_matched_complex_p_n0_gate_seed0.pt
3fe85c1e6b7b184a2137f43f8070284a1f12aa4fa6ac17a676cc1400d47086a7  strict_matched_complex_p_n0_gate_seed1.pt
693e05f97e53c8786422d7de7d7dd126f9b67529444956e16769e864965083b3  strict_matched_complex_p_n0_gate_seed2.pt
```
