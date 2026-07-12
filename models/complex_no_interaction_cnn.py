import torch
import torch.nn as nn

from .baseline_cnn import _ensure_complex_grid, _prepare_zero_features
from .complex_layers import (
    AmplitudeGate,
    AmplitudeSwiGLUGate,
    ChargeBranch,
    ComplexConv2d,
    ComplexRMSNorm2d,
)


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


def _select_zero_condition(P, N0_grid, condition_mode):
    if condition_mode == "p":
        return P
    if condition_mode == "n0":
        return N0_grid
    if condition_mode == "p_n0":
        return torch.cat([P, N0_grid], dim=1)
    raise ValueError("condition_mode must be 'p', 'n0', or 'p_n0'.")


def _condition_channels(condition_mode):
    if condition_mode in {"p", "n0"}:
        return 1
    if condition_mode == "p_n0":
        return 2
    raise ValueError("condition_mode must be 'p', 'n0', or 'p_n0'.")


class ComplexCNNWithZeroInput(nn.Module):
    """Complex CNN baseline with zero-order features injected as input channels.

    The selected zero-order conditions are converted to complex channels with
    zero imaginary part and concatenated with Y and H_hat before the complex
    branch. This is an intentionally permissive ablation: it tests whether the
    complex branch benefits from seeing P/log(N0) early, but it does not
    preserve a clean charge convention.
    """

    def __init__(
        self,
        condition_mode,
        hidden_complex=64,
        hidden_real=32,
        bits_per_symbol=2,
        branch_layers=2,
        kernel_size=3,
        use_norm=True,
        gate_type="swiglu",
    ):
        super().__init__()
        self.condition_mode = condition_mode
        input_channels = 2 + _condition_channels(condition_mode)
        self.complex_branch = ChargeBranch(
            in_channels=input_channels,
            hidden_channels=hidden_complex,
            num_layers=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
            gate_type=gate_type,
        )

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
        condition = _select_zero_condition(P, N0_grid, self.condition_mode)
        condition_complex = torch.complex(condition, torch.zeros_like(condition))

        x = torch.cat(
            [
                torch.stack([Y, H_hat], dim=1),
                condition_complex,
            ],
            dim=1,
        )
        feat = self.complex_branch(x)
        feat_real = torch.cat([feat.real, feat.imag], dim=1)
        z = torch.cat([feat_real, P, N0_grid], dim=1)
        return self.llr_head(z)


class ComplexZeroOrderGate(nn.Module):
    """Identity-initialized zero-order amplitude gate for complex features."""

    def __init__(self, channels, condition_channels, condition_hidden=16):
        super().__init__()
        if condition_hidden <= 0:
            raise ValueError("condition_hidden must be positive.")
        self.condition_net = nn.Sequential(
            nn.Conv2d(condition_channels, condition_hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(condition_hidden, channels, kernel_size=1),
        )
        nn.init.zeros_(self.condition_net[-1].weight)
        nn.init.zeros_(self.condition_net[-1].bias)

    def forward(self, z, condition):
        scale = 2.0 * torch.sigmoid(self.condition_net(condition))
        return z * scale


class ComplexZeroOrderFiLM(nn.Module):
    """Identity-initialized real/imag FiLM conditioning for complex features."""

    def __init__(self, channels, condition_channels, condition_hidden=16):
        super().__init__()
        if condition_hidden <= 0:
            raise ValueError("condition_hidden must be positive.")
        self.condition_net = nn.Sequential(
            nn.Conv2d(condition_channels, condition_hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(condition_hidden, 4 * channels, kernel_size=1),
        )
        nn.init.zeros_(self.condition_net[-1].weight)
        nn.init.zeros_(self.condition_net[-1].bias)

    def forward(self, z, condition):
        gamma_r, beta_r, gamma_i, beta_i = self.condition_net(condition).chunk(4, dim=1)
        real = (1.0 + gamma_r) * z.real + beta_r
        imag = (1.0 + gamma_i) * z.imag + beta_i
        return torch.complex(real, imag)


class ComplexConditionedLayer(nn.Module):
    """One complex conv block followed by zero-order conditioning."""

    def __init__(
        self,
        in_channels,
        out_channels,
        condition_channels,
        condition_hidden,
        condition_method,
        kernel_size=3,
        use_norm=True,
        gate_type="swiglu",
    ):
        super().__init__()
        padding = kernel_size // 2
        if gate_type not in {"sigmoid", "swiglu"}:
            raise ValueError("gate_type must be 'sigmoid' or 'swiglu'.")
        if condition_method not in {"gate", "film"}:
            raise ValueError("condition_method must be 'gate' or 'film'.")

        self.conv = ComplexConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.norm = ComplexRMSNorm2d(out_channels) if use_norm else nn.Identity()
        if gate_type == "sigmoid":
            self.activation = AmplitudeGate(out_channels)
        else:
            self.activation = AmplitudeSwiGLUGate(out_channels)
        if condition_method == "gate":
            self.condition = ComplexZeroOrderGate(
                out_channels,
                condition_channels,
                condition_hidden=condition_hidden,
            )
        else:
            self.condition = ComplexZeroOrderFiLM(
                out_channels,
                condition_channels,
                condition_hidden=condition_hidden,
            )

    def forward(self, z, condition):
        z = self.conv(z)
        z = self.norm(z)
        z = self.activation(z)
        return self.condition(z, condition)


class ComplexCNNWithZeroConditioning(nn.Module):
    """Complex CNN baseline with P/log(N0) conditioning inside the trunk."""

    def __init__(
        self,
        condition_mode="p_n0",
        condition_method="gate",
        hidden_complex=64,
        hidden_real=32,
        bits_per_symbol=2,
        branch_layers=2,
        kernel_size=3,
        use_norm=True,
        gate_type="swiglu",
        condition_hidden=16,
    ):
        super().__init__()
        self.condition_mode = condition_mode
        condition_channels = _condition_channels(condition_mode)
        layers = []
        in_channels = 2
        for _ in range(branch_layers):
            layers.append(
                ComplexConditionedLayer(
                    in_channels,
                    hidden_complex,
                    condition_channels,
                    condition_hidden,
                    condition_method,
                    kernel_size=kernel_size,
                    use_norm=use_norm,
                    gate_type=gate_type,
                )
            )
            in_channels = hidden_complex
        self.layers = nn.ModuleList(layers)

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
        condition = _select_zero_condition(P, N0_grid, self.condition_mode)
        z = torch.stack([Y, H_hat], dim=1)
        for layer in self.layers:
            z = layer(z, condition)

        feat_real = torch.cat([z.real, z.imag], dim=1)
        llr_features = torch.cat([feat_real, P, N0_grid], dim=1)
        return self.llr_head(llr_features)
