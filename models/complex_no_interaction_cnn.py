import torch
import torch.nn as nn

from .baseline_cnn import _ensure_complex_grid, _prepare_zero_features
from .complex_layers import ChargeBranch


def _make_group_norm(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ComplexCNNNoInteraction(nn.Module):
    """
    Ablation baseline using complex convolution but without zero-order interaction.

    Purpose:
        Test whether performance gain comes merely from complex-valued convolution,
        or from the full phase-invariant framework.

    Input:
        Y, H_hat are treated as charge +1 features.

    Structure:
        [Y, H_hat]
            -> complex equivariant CNN branch
            -> Re/Im split
            -> concat P, log(N0)
            -> real-valued LLR head

    This model uses complex convolution, but it is NOT guaranteed to be
    invariant to common phase rotation, because it directly reads out Re/Im
    of charge +1 features.
    """

    def __init__(
        self,
        hidden_complex=64,
        hidden_real=32,
        bits_per_symbol=2,
        branch_layers=2,
        kernel_size=3,
        use_norm=True,
        gate_type="swiglu",
    ):
        super().__init__()

        self.complex_branch = ChargeBranch(
            in_channels=2,
            hidden_channels=hidden_complex,
            num_layers=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
            gate_type=gate_type,
        )

        # Directly read Re/Im of charge +1 features.
        # This intentionally does not construct zero-order features.
        llr_in_channels = 2 * hidden_complex + 2

        self.llr_head = nn.Sequential(
            nn.Conv2d(llr_in_channels, hidden_real, kernel_size=3, padding=1),
            _make_group_norm(hidden_real),
            nn.ReLU(),
            nn.Conv2d(hidden_real, hidden_real, kernel_size=3, padding=1),
            _make_group_norm(hidden_real),
            nn.ReLU(),
            nn.Conv2d(hidden_real, bits_per_symbol, kernel_size=1),
        )

    def forward(self, Y, H_hat, P, N0):
        Y = _ensure_complex_grid(Y)
        H_hat = _ensure_complex_grid(H_hat)

        b, t, f = Y.shape
        P, N0_grid = _prepare_zero_features(P, N0, b, t, f, Y.device)

        # Shape: (B, C, T, F), complex
        x = torch.stack([Y, H_hat], dim=1)

        feat = self.complex_branch(x)

        # Direct Re/Im readout. This breaks guaranteed phase invariance.
        feat_real = torch.cat(
            [
                feat.real,
                feat.imag,
            ],
            dim=1,
        )

        z = torch.cat(
            [
                feat_real,
                P,
                N0_grid,
            ],
            dim=1,
        )

        return self.llr_head(z)
