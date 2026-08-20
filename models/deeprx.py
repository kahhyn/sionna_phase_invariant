"""Compact and paper-architecture DeepRx receivers plus A/C adapters.

The original DeepRx input contains the received resource grid, the known
pilot grid, and a raw pilot-domain channel estimate. The Sionna generator in
this repository uses unit pilots, so the raw estimate is simply ``P * Y``.

``DeepRxInvariantReceiver`` and ``DeepRxMatchedReceiver`` share every learned
module. They differ only in the algebraic operation at the input adapter:

    A: a * conj(b)                 (common-phase invariant)
    C: (a + b) / sqrt(2)           (common-phase sensitive)

The adapter exposes the same number of real channels in both cases, followed
by an identical DeepRx-style pre-activation dilated ResNet.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_layers import (
    AmplitudeSwiGLUGate,
    ComplexConv2d,
    ComplexRMSNorm2d,
)


def _as_complex_grid(x, name):
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.dim() != 4 or x.shape[1] != 1:
        raise ValueError(f"{name} must have shape (B,T,F) or (B,1,T,F).")
    if not torch.is_complex(x):
        raise TypeError(f"{name} must be a complex tensor.")
    return x


def _zero_features(P, N0, batch_size, num_symbols, num_subcarriers, device):
    if P.dim() == 3:
        P = P.unsqueeze(1)
    if P.shape != (batch_size, 1, num_symbols, num_subcarriers):
        raise ValueError("P must have shape (B,1,T,F) or (B,T,F).")
    P = P.to(device=device, dtype=torch.float32)

    if N0.dim() in (1, 2):
        N0 = N0.reshape(batch_size, 1, 1, 1)
    elif N0.dim() != 4:
        raise ValueError(f"Unsupported N0 shape: {tuple(N0.shape)}")
    log_n0 = torch.log(N0.to(device=device, dtype=torch.float32) + 1e-12)
    log_n0 = log_n0.expand(batch_size, 1, num_symbols, num_subcarriers)
    return P, log_n0


def _paper_input_features(Y, P):
    """Build the original DeepRx input for the repository's unit pilots.

    The Sionna setup currently exposes the transmitted pilot grid as the real
    tensor ``P``: pilot values are one and non-pilot resource elements are
    zero.  Thus ``X_p = P`` and ``H_raw = Y * conj(X_p) = Y * P``.
    """
    charge_one, charge_zero = _paper_input_components(Y, P)
    y = charge_one[:, :1]
    raw_h = charge_one[:, 1:]
    x_p_real = charge_zero[:, :1]
    x_p_imag = charge_zero[:, 1:]
    return torch.cat(
        [y.real, y.imag, x_p_real, x_p_imag, raw_h.real, raw_h.imag],
        dim=1,
    )


def _paper_input_components(Y, P):
    """Return charge-one ``[Y,H_raw]`` and charge-zero ``[Re Xp,Im Xp]``."""
    batch_size, _, num_symbols, num_subcarriers = Y.shape
    if P.dim() == 3:
        P = P.unsqueeze(1)
    if P.shape != (batch_size, 1, num_symbols, num_subcarriers):
        raise ValueError("P must have shape (B,1,T,F) or (B,T,F).")
    x_p = P.to(device=Y.device, dtype=Y.dtype)
    raw_h = Y * torch.conj(x_p)
    return (
        torch.cat([Y, raw_h], dim=1),
        torch.cat([x_p.real, x_p.imag], dim=1),
    )


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, depth_multiplier=1):
        super().__init__()
        if depth_multiplier <= 0:
            raise ValueError("depth_multiplier must be positive.")
        depthwise_channels = in_channels * depth_multiplier
        self.depthwise = nn.Conv2d(
            in_channels,
            depthwise_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            depthwise_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DeepRxPreActivationBlock(nn.Module):
    """Full-preactivation residual block with separable dilated convolutions."""

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU()
        self.conv1 = DepthwiseSeparableConv2d(
            in_channels, out_channels, dilation=dilation
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        self.conv2 = DepthwiseSeparableConv2d(
            out_channels, out_channels, dilation=dilation
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        preactivated = self.act1(self.norm1(x))
        residual = self.shortcut(preactivated)
        out = self.conv1(preactivated)
        out = self.conv2(self.act2(self.norm2(out)))
        return residual + out


def _compact_deeprx_specs(hidden, num_blocks):
    """Return the five-block DeepRx schedule used by the smoke reproduction."""
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive.")
    if hidden < 2:
        raise ValueError("hidden must be at least 2.")

    base_dilations = [(1, 1), (2, 4), (3, 8), (2, 4), (1, 1)]
    base_channels = [hidden, hidden, hidden // 2, hidden // 2, hidden // 2]
    return [
        (
            base_channels[min(index, len(base_channels) - 1)],
            base_dilations[index % len(base_dilations)],
        )
        for index in range(num_blocks)
    ]


class DeepRxBackbone(nn.Module):
    def __init__(self, in_channels, hidden, bits_per_symbol, num_blocks=5):
        super().__init__()
        self.input_conv = nn.Conv2d(
            in_channels, hidden, kernel_size=3, padding=1, bias=True
        )
        blocks = []
        current_channels = hidden
        for out_channels, dilation in _compact_deeprx_specs(hidden, num_blocks):
            blocks.append(
                DeepRxPreActivationBlock(
                    current_channels, out_channels, dilation=dilation
                )
            )
            current_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.output_norm = nn.BatchNorm2d(current_channels)
        self.output_act = nn.ReLU()
        self.output_conv = nn.Conv2d(
            current_channels, bits_per_symbol, kernel_size=1
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.blocks(x)
        return self.output_conv(self.output_act(self.output_norm(x)))


class DeepRxReceiver(nn.Module):
    """Real-valued compact DeepRx baseline with raw pilot-domain CSI input."""

    def __init__(self, hidden=64, bits_per_symbol=2, num_blocks=5):
        super().__init__()
        # Re/Im(Y), Re/Im(H_raw), P, log(N0)
        self.backbone = DeepRxBackbone(6, hidden, bits_per_symbol, num_blocks)

    def forward(self, Y, H_hat, P, N0):
        del H_hat  # DeepRx uses raw pilot-domain CSI, not interpolated CSI.
        Y = _as_complex_grid(Y, "Y")
        batch_size, _, num_symbols, num_subcarriers = Y.shape
        P, log_n0 = _zero_features(
            P, N0, batch_size, num_symbols, num_subcarriers, Y.device
        )
        raw_h = P.to(Y.dtype) * Y
        features = torch.cat(
            [Y.real, Y.imag, raw_h.real, raw_h.imag, P, log_n0], dim=1
        )
        return self.backbone(features)


class PaperInputCompactDeepRxReceiver(nn.Module):
    """Current five-block compact backbone with the original DeepRx input.

    This control isolates backbone depth/schedule from input information: it
    receives exactly ``[Y, X_p, H_raw]`` and does not use interpolated CSI or
    explicit noise variance.
    """

    def __init__(self, hidden=64, bits_per_symbol=2, num_blocks=5):
        super().__init__()
        self.backbone = DeepRxBackbone(6, hidden, bits_per_symbol, num_blocks)

    def forward(self, Y, H_hat, P, N0):
        del H_hat, N0
        Y = _as_complex_grid(Y, "Y")
        return self.backbone(_paper_input_features(Y, P))


_PAPER_DEEPRX_SPECS = (
    (64, (1, 1)),
    (64, (1, 1)),
    (128, (2, 3)),
    (128, (2, 3)),
    (256, (2, 3)),
    (256, (3, 6)),
    (256, (2, 3)),
    (128, (2, 3)),
    (128, (2, 3)),
    (64, (1, 1)),
    (64, (1, 1)),
)


class PaperDeepRxPreActivationBlock(nn.Module):
    """Full-preactivation block used by the paper-architecture reproduction."""

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.ReLU()
        self.conv1 = DepthwiseSeparableConv2d(
            in_channels,
            out_channels,
            dilation=dilation,
            depth_multiplier=2,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        self.conv2 = DepthwiseSeparableConv2d(
            out_channels,
            out_channels,
            dilation=dilation,
            depth_multiplier=2,
        )
        self.shortcut = (
            None
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        preactivated = self.act1(self.norm1(x))
        residual = x if self.shortcut is None else self.shortcut(preactivated)
        out = self.conv1(preactivated)
        out = self.conv2(self.act2(self.norm2(out)))
        return residual + out


class PaperDeepRx11Receiver(nn.Module):
    """Architecture-faithful 11-block DeepRx on the local Sionna interface."""

    def __init__(self, bits_per_symbol=2):
        super().__init__()
        self.input_conv = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        blocks = []
        current_channels = 64
        for out_channels, dilation in _PAPER_DEEPRX_SPECS:
            blocks.append(
                PaperDeepRxPreActivationBlock(
                    current_channels,
                    out_channels,
                    dilation=dilation,
                )
            )
            current_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.output_norm = nn.BatchNorm2d(current_channels)
        self.output_act = nn.ReLU()
        self.output_conv = nn.Conv2d(
            current_channels, bits_per_symbol, kernel_size=1
        )

    def forward(self, Y, H_hat, P, N0):
        del H_hat, N0
        Y = _as_complex_grid(Y, "Y")
        x = self.input_conv(_paper_input_features(Y, P))
        x = self.blocks(x)
        return self.output_conv(self.output_act(self.output_norm(x)))


class _DeepRxACReceiver(nn.Module):
    def __init__(
        self,
        interaction,
        hidden=64,
        adapter_complex=2,
        bits_per_symbol=2,
        num_blocks=5,
    ):
        super().__init__()
        if interaction not in {"invariant", "direct"}:
            raise ValueError("interaction must be invariant or direct.")
        if adapter_complex <= 0:
            raise ValueError("adapter_complex must be positive.")
        self.interaction = interaction
        self.proj_a = ComplexConv2d(
            2, adapter_complex, kernel_size=3, padding=1, bias=False
        )
        self.proj_b = ComplexConv2d(
            2, adapter_complex, kernel_size=3, padding=1, bias=False
        )
        self.norm_a = ComplexRMSNorm2d(adapter_complex)
        self.norm_b = ComplexRMSNorm2d(adapter_complex)
        self.backbone = DeepRxBackbone(
            2 * adapter_complex + 2,
            hidden,
            bits_per_symbol,
            num_blocks,
        )

    def forward(self, Y, H_hat, P, N0):
        del H_hat
        Y = _as_complex_grid(Y, "Y")
        batch_size, _, num_symbols, num_subcarriers = Y.shape
        P, log_n0 = _zero_features(
            P, N0, batch_size, num_symbols, num_subcarriers, Y.device
        )
        raw_h = P.to(Y.dtype) * Y
        charge_one = torch.cat([Y, raw_h], dim=1)
        a = self.norm_a(self.proj_a(charge_one))
        b = self.norm_b(self.proj_b(charge_one))

        if self.interaction == "invariant":
            adapted = a * torch.conj(b)
        else:
            adapted = (a + b) / math.sqrt(2.0)

        features = torch.cat([adapted.real, adapted.imag, P, log_n0], dim=1)
        return self.backbone(features)


class DeepRxInvariantReceiver(_DeepRxACReceiver):
    def __init__(
        self,
        hidden=64,
        adapter_complex=2,
        bits_per_symbol=2,
        num_blocks=5,
    ):
        super().__init__(
            "invariant",
            hidden=hidden,
            adapter_complex=adapter_complex,
            bits_per_symbol=bits_per_symbol,
            num_blocks=num_blocks,
        )


class DeepRxMatchedReceiver(_DeepRxACReceiver):
    def __init__(
        self,
        hidden=64,
        adapter_complex=2,
        bits_per_symbol=2,
        num_blocks=5,
    ):
        super().__init__(
            "direct",
            hidden=hidden,
            adapter_complex=adapter_complex,
            bits_per_symbol=bits_per_symbol,
            num_blocks=num_blocks,
        )


class ComplexDepthwiseSeparableConv2d(nn.Module):
    """Charge-preserving complex depthwise-separable dilated convolution."""

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        depthwise_kwargs = dict(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_real = nn.Conv2d(**depthwise_kwargs)
        self.depthwise_imag = nn.Conv2d(**depthwise_kwargs)
        self.pointwise_real = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.pointwise_imag = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

    @staticmethod
    def _complex_apply(real_layer, imag_layer, z):
        real = real_layer(z.real) - imag_layer(z.imag)
        imag = real_layer(z.imag) + imag_layer(z.real)
        return torch.complex(real, imag)

    def forward(self, z):
        if not torch.is_complex(z):
            raise TypeError("ComplexDepthwiseSeparableConv2d expects complex input.")
        z = self._complex_apply(self.depthwise_real, self.depthwise_imag, z)
        return self._complex_apply(self.pointwise_real, self.pointwise_imag, z)


class EquivariantDeepRxBlock(nn.Module):
    """Equivariant counterpart of a DeepRx pre-activation residual block."""

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.norm1 = ComplexRMSNorm2d(in_channels)
        self.gate1 = AmplitudeSwiGLUGate(in_channels)
        self.conv1 = ComplexDepthwiseSeparableConv2d(
            in_channels, out_channels, dilation
        )
        self.norm2 = ComplexRMSNorm2d(out_channels)
        self.gate2 = AmplitudeSwiGLUGate(out_channels)
        self.conv2 = ComplexDepthwiseSeparableConv2d(
            out_channels, out_channels, dilation
        )
        self.shortcut = (
            None
            if in_channels == out_channels
            else ComplexConv2d(
                in_channels, out_channels, kernel_size=1, padding=0, bias=False
            )
        )

    def forward(self, z):
        preactivated = self.gate1(self.norm1(z))
        residual = z if self.shortcut is None else self.shortcut(preactivated)
        out = self.conv1(preactivated)
        out = self.conv2(self.gate2(self.norm2(out)))
        return residual + out


class _LateInteractionDeepRx(nn.Module):
    """DeepRx macro-architecture with a late, parameter-matched A/C readout."""

    def __init__(
        self,
        interaction,
        hidden_real=64,
        hidden_complex=32,
        zero_complex=32,
        bits_per_symbol=2,
        num_blocks=5,
    ):
        super().__init__()
        if interaction not in {"invariant", "direct"}:
            raise ValueError("interaction must be invariant or direct.")
        self.interaction = interaction
        self.input_proj = ComplexConv2d(
            2, hidden_complex, kernel_size=3, padding=1, bias=False
        )
        self.input_norm = ComplexRMSNorm2d(hidden_complex)
        self.input_gate = AmplitudeSwiGLUGate(hidden_complex)

        blocks = []
        current_channels = hidden_complex
        for out_channels, dilation in _compact_deeprx_specs(
            hidden_complex, num_blocks
        ):
            blocks.append(
                EquivariantDeepRxBlock(
                    current_channels, out_channels, dilation=dilation
                )
            )
            current_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

        self.proj_a = ComplexConv2d(
            current_channels, zero_complex, kernel_size=3, padding=1, bias=False
        )
        self.proj_b = ComplexConv2d(
            current_channels, zero_complex, kernel_size=3, padding=1, bias=False
        )
        self.norm_a = ComplexRMSNorm2d(zero_complex)
        self.norm_b = ComplexRMSNorm2d(zero_complex)

        self.output_head = nn.Sequential(
            nn.Conv2d(2 * zero_complex + 2, hidden_real, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_real),
            nn.ReLU(),
            nn.Conv2d(hidden_real, bits_per_symbol, kernel_size=1),
        )

    def forward(self, Y, H_hat, P, N0):
        del H_hat
        Y = _as_complex_grid(Y, "Y")
        batch_size, _, num_symbols, num_subcarriers = Y.shape
        P, log_n0 = _zero_features(
            P, N0, batch_size, num_symbols, num_subcarriers, Y.device
        )
        raw_h = P.to(Y.dtype) * Y
        z = torch.cat([Y, raw_h], dim=1)
        z = self.input_gate(self.input_norm(self.input_proj(z)))
        for block in self.blocks:
            z = block(z)

        a = self.norm_a(self.proj_a(z))
        b = self.norm_b(self.proj_b(z))
        if self.interaction == "invariant":
            adapted = a * torch.conj(b)
        else:
            adapted = (a + b) / math.sqrt(2.0)
        features = torch.cat([adapted.real, adapted.imag, P, log_n0], dim=1)
        return self.output_head(features)


class LateInvariantDeepRx(_LateInteractionDeepRx):
    def __init__(self, **kwargs):
        super().__init__("invariant", **kwargs)


class LateMatchedDeepRx(_LateInteractionDeepRx):
    def __init__(self, **kwargs):
        super().__init__("direct", **kwargs)


class ConditionalAmplitudeSwiGLUGate(nn.Module):
    """One amplitude gate jointly conditioned on ``|z|`` and ``X_p``.

    Pilot features shift the real SwiGLU gate logits instead of introducing a
    second multiplicative gate.  Since both ``|z|`` and ``X_p`` are charge
    zero, the real multiplier preserves the charge of the complex input.
    """

    def __init__(self, channels, condition_hidden=8):
        super().__init__()
        if condition_hidden <= 0:
            raise ValueError("condition_hidden must be positive.")
        self.proj = nn.Conv2d(channels, 2 * channels, kernel_size=1)
        self.condition_net = nn.Sequential(
            nn.Conv2d(2, condition_hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(condition_hidden, channels, kernel_size=1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.condition_net[-1].weight)
        nn.init.zeros_(self.condition_net[-1].bias)

    def forward(self, z, x_p_features):
        if not torch.is_complex(z):
            raise TypeError(
                "ConditionalAmplitudeSwiGLUGate expects a complex tensor."
            )
        if x_p_features.shape[1] != 2:
            raise ValueError("x_p_features must contain Re(X_p) and Im(X_p).")
        gate, value = self.proj(torch.abs(z)).chunk(2, dim=1)
        gate = gate + self.condition_net(x_p_features)
        multiplier = 1.0 + self.residual_scale * F.silu(gate) * value
        return multiplier * z


class PilotConditionedEquivariantDeepRxBlock(nn.Module):
    """Equivariant block with pilot conditioning only in its residual update."""

    def __init__(self, in_channels, out_channels, dilation, condition_hidden=8):
        super().__init__()
        self.norm1 = ComplexRMSNorm2d(in_channels)
        self.gate1 = AmplitudeSwiGLUGate(in_channels)
        self.conv1 = ComplexDepthwiseSeparableConv2d(
            in_channels, out_channels, dilation
        )
        self.norm2 = ComplexRMSNorm2d(out_channels)
        self.gate2 = ConditionalAmplitudeSwiGLUGate(
            out_channels, condition_hidden=condition_hidden
        )
        self.conv2 = ComplexDepthwiseSeparableConv2d(
            out_channels, out_channels, dilation
        )
        self.shortcut = (
            None
            if in_channels == out_channels
            else ComplexConv2d(
                in_channels, out_channels, kernel_size=1, padding=0, bias=False
            )
        )

    def forward(self, z, x_p_features):
        preactivated = self.gate1(self.norm1(z))
        residual = z if self.shortcut is None else self.shortcut(preactivated)
        out = self.conv1(preactivated)
        out = self.conv2(self.gate2(self.norm2(out), x_p_features))
        return residual + out


class _PaperInputAC5Receiver(nn.Module):
    """Frozen paper-input A5/C5 architecture for the compact comparison.

    Both variants consume exactly ``[Y, X_p, H_raw]``. They share module
    topology and parameter count and differ only in the algebraic interaction:

    * A: ``a * conj(b)``
    * C: ``(a + b) / sqrt(2)``
    """

    def __init__(
        self,
        interaction,
        hidden_complex=32,
        readout_complex=32,
        hidden_real=60,
        condition_hidden=8,
        bits_per_symbol=2,
        num_blocks=5,
    ):
        super().__init__()
        if interaction not in {"invariant", "direct"}:
            raise ValueError("interaction must be invariant or direct.")
        if num_blocks != 5:
            raise ValueError("The frozen A5/C5 comparison requires five blocks.")
        self.interaction = interaction

        self.input_proj = ComplexConv2d(
            2, hidden_complex, kernel_size=3, padding=1, bias=False
        )
        self.input_norm = ComplexRMSNorm2d(hidden_complex)
        self.input_gate = ConditionalAmplitudeSwiGLUGate(
            hidden_complex, condition_hidden=condition_hidden
        )

        blocks = []
        current_channels = hidden_complex
        for out_channels, dilation in _compact_deeprx_specs(
            hidden_complex, num_blocks
        ):
            blocks.append(
                PilotConditionedEquivariantDeepRxBlock(
                    current_channels,
                    out_channels,
                    dilation=dilation,
                    condition_hidden=condition_hidden,
                )
            )
            current_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

        self.proj_a = ComplexConv2d(
            current_channels,
            readout_complex,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.proj_b = ComplexConv2d(
            current_channels,
            readout_complex,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm_a = ComplexRMSNorm2d(readout_complex)
        self.norm_b = ComplexRMSNorm2d(readout_complex)
        self.post_interaction_norm = ComplexRMSNorm2d(readout_complex)

        self.output_head = nn.Sequential(
            nn.Conv2d(
                2 * readout_complex + 2,
                hidden_real,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(hidden_real),
            nn.ReLU(),
            nn.Conv2d(hidden_real, bits_per_symbol, kernel_size=1),
        )

    def interaction_features(self, Y, P):
        """Return the normalized feature immediately before the real head."""
        Y = _as_complex_grid(Y, "Y")
        z, x_p_features = _paper_input_components(Y, P)
        z = self.input_gate(
            self.input_norm(self.input_proj(z)), x_p_features
        )
        for block in self.blocks:
            z = block(z, x_p_features)

        a = self.norm_a(self.proj_a(z))
        b = self.norm_b(self.proj_b(z))
        if self.interaction == "invariant":
            interaction = a * torch.conj(b)
        else:
            interaction = (a + b) / math.sqrt(2.0)
        return self.post_interaction_norm(interaction), x_p_features

    def forward(self, Y, H_hat, P, N0):
        del H_hat, N0
        interaction, x_p_features = self.interaction_features(Y, P)
        features = torch.cat(
            [interaction.real, interaction.imag, x_p_features], dim=1
        )
        return self.output_head(features)


class PaperInputInvariantA5(_PaperInputAC5Receiver):
    def __init__(self, **kwargs):
        super().__init__("invariant", **kwargs)


class PaperInputMatchedC5(_PaperInputAC5Receiver):
    def __init__(self, **kwargs):
        super().__init__("direct", **kwargs)
