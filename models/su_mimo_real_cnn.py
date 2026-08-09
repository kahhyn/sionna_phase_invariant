"""Parameter-matched real-valued CNN baseline for SU-MIMO OFDM."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .single_invariant_net import _make_group_norm
from .su_mimo_invariant_net import _prepare_n0_grid


class RealZeroOrderGate(nn.Module):
    """Condition real features on the pilot mask and log noise power."""

    def __init__(self, channels: int, condition_hidden: int = 16) -> None:
        super().__init__()
        if condition_hidden <= 0:
            raise ValueError("condition_hidden must be positive.")
        self.condition_net = nn.Sequential(
            nn.Conv2d(2, condition_hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(condition_hidden, channels, kernel_size=1),
        )
        nn.init.zeros_(self.condition_net[-1].weight)
        nn.init.zeros_(self.condition_net[-1].bias)

    def forward(self, z, zero_features):
        if z.is_complex():
            raise TypeError("RealZeroOrderGate expects real-valued features.")
        if zero_features.shape[1] != 2:
            raise ValueError("zero_features must contain P and log(N0).")
        scale = 2.0 * torch.sigmoid(self.condition_net(zero_features))
        return z * scale


class RealResidualBlock(nn.Module):
    """Conventional two-convolution residual block."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.norm1 = _make_group_norm(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size, padding=padding, bias=False
        )
        self.norm2 = _make_group_norm(channels)
        self.activation = nn.SiLU()

    def forward(self, z):
        residual = z
        z = self.activation(self.norm1(self.conv1(z)))
        z = self.norm2(self.conv2(z))
        return self.activation(z + residual)


class RealEquivariantLayerInteraction(nn.Module):
    """Real-valued DeepSets interaction preserving layer permutations."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.message_proj = nn.Conv2d(
            channels, channels, kernel_size=1, bias=False
        )
        self.message_norm = _make_group_norm(channels)
        self.update_proj = nn.Conv2d(
            2 * channels, channels, kernel_size=1, bias=False
        )
        self.update_norm = _make_group_norm(channels)
        self.activation = nn.SiLU()

    def forward(self, z, layer_mask=None):
        if z.dim() != 5 or z.is_complex():
            raise ValueError("z must be real with shape [B, L, C, T, F].")
        batch_size, num_layers, channels, num_symbols, num_subcarriers = z.shape
        if layer_mask is None:
            layer_mask = torch.ones(
                batch_size, num_layers, dtype=torch.bool, device=z.device
            )
        if layer_mask.shape != (batch_size, num_layers):
            raise ValueError("layer_mask must have shape [B, L].")

        flat = z.reshape(
            batch_size * num_layers, channels, num_symbols, num_subcarriers
        )
        message = self.activation(self.message_norm(self.message_proj(flat)))
        message = message.reshape(z.shape)

        active = layer_mask.to(dtype=z.dtype).view(
            batch_size, num_layers, 1, 1, 1
        )
        message_sum = (message * active).sum(dim=1, keepdim=True)
        other_sum = message_sum - message * active
        other_count = active.sum(dim=1, keepdim=True) - active
        other_mean = torch.where(
            other_count > 0,
            other_sum / other_count.clamp_min(1.0),
            torch.zeros_like(other_sum),
        )

        update_input = torch.cat([z, other_mean], dim=2).reshape(
            batch_size * num_layers,
            2 * channels,
            num_symbols,
            num_subcarriers,
        )
        update = self.activation(
            self.update_norm(self.update_proj(update_input))
        ).reshape(z.shape)
        return (z + update) * active


def _real_parameter_count(
    *,
    num_rx_ant: int,
    hidden_channels: int,
    readout_channels: int,
    hidden_real: int,
    bits_per_symbol: int,
    num_iterations: int,
    kernel_size: int,
    zero_gate_hidden: int,
) -> int:
    """Closed-form count for :class:`SUMIMORealCNNReceiver`."""
    h = hidden_channels
    z = readout_channels
    r = num_rx_ant
    q = kernel_size**2
    gate = 19 * zero_gate_hidden + zero_gate_hidden * h + h

    count = 4 * r
    count += 4 * r * h * q + 2 * h
    count += gate
    count += num_iterations * (
        (2 * h * h * q + 4 * h)
        + (3 * h * h + 4 * h)
        + gate
    )
    count += 2 * h * z * q + 4 * z
    count += 4 * z * z + 2 * z
    count += (2 * z + 2) * hidden_real * 9 + hidden_real
    count += 2 * hidden_real
    count += hidden_real * hidden_real * 9 + hidden_real
    count += 2 * hidden_real
    count += hidden_real * bits_per_symbol + bits_per_symbol
    return count


def resolve_real_cnn_widths(
    *,
    target_parameter_count: int,
    num_rx_ant: int,
    hidden_complex: int,
    zero_real: int,
    hidden_real: int,
    bits_per_symbol: int,
    num_iterations: int,
    kernel_size: int,
    zero_gate_hidden: int,
) -> tuple[int, int, int]:
    """Choose real backbone/readout widths nearest to a parameter budget."""
    if target_parameter_count <= 0:
        raise ValueError("target_parameter_count must be positive.")
    center = max(1, round(math.sqrt(2.0) * hidden_complex))
    hidden_candidates = range(
        max(2, center - hidden_complex), center + hidden_complex + 1
    )
    readout_candidates = range(max(2, zero_real // 2), 2 * zero_real + 1)
    candidates = []
    for hidden_channels in hidden_candidates:
        for readout_channels in readout_candidates:
            count = _real_parameter_count(
                num_rx_ant=num_rx_ant,
                hidden_channels=hidden_channels,
                readout_channels=readout_channels,
                hidden_real=hidden_real,
                bits_per_symbol=bits_per_symbol,
                num_iterations=num_iterations,
                kernel_size=kernel_size,
                zero_gate_hidden=zero_gate_hidden,
            )
            candidates.append(
                (
                    abs(count - target_parameter_count),
                    abs(hidden_channels - center),
                    abs(readout_channels - zero_real),
                    hidden_channels,
                    readout_channels,
                    count,
                )
            )
    _, _, _, hidden_channels, readout_channels, count = min(candidates)
    return hidden_channels, readout_channels, count


class SUMIMORealCNNReceiver(nn.Module):
    """Layer-equivariant ordinary real CNN matched to the complex baseline.

    The semantic inputs and layer interaction topology match the complex
    phase-sensitive receiver. Complex grids are represented by independent
    real and imaginary channels, and all learned operations are conventional
    real-valued convolutions, GroupNorm, and SiLU activations.
    """

    def __init__(
        self,
        num_rx_ant=2,
        hidden_complex=32,
        zero_real=22,
        hidden_real=66,
        bits_per_symbol=2,
        num_iterations=2,
        kernel_size=3,
        zero_gate_hidden=16,
        target_parameter_count=None,
    ) -> None:
        super().__init__()
        if num_rx_ant <= 0 or num_iterations <= 0:
            raise ValueError("num_rx_ant and num_iterations must be positive.")
        if target_parameter_count is None:
            raise ValueError(
                "target_parameter_count is required for the matched real CNN."
            )

        hidden_channels, readout_channels, predicted_count = (
            resolve_real_cnn_widths(
                target_parameter_count=int(target_parameter_count),
                num_rx_ant=int(num_rx_ant),
                hidden_complex=int(hidden_complex),
                zero_real=int(zero_real),
                hidden_real=int(hidden_real),
                bits_per_symbol=int(bits_per_symbol),
                num_iterations=int(num_iterations),
                kernel_size=int(kernel_size),
                zero_gate_hidden=int(zero_gate_hidden),
            )
        )
        self.num_rx_ant = int(num_rx_ant)
        self.bits_per_symbol = int(bits_per_symbol)
        self.real_hidden_channels = hidden_channels
        self.real_readout_channels = readout_channels
        self.target_parameter_count = int(target_parameter_count)
        self.predicted_parameter_count = predicted_count
        self.resolved_model_config = {
            "real_hidden_channels": hidden_channels,
            "real_readout_channels": readout_channels,
            "target_parameter_count": self.target_parameter_count,
            "predicted_parameter_count": predicted_count,
        }

        self.input_scale = nn.Parameter(torch.ones(4 * self.num_rx_ant))
        self.input_proj = nn.Conv2d(
            4 * self.num_rx_ant,
            hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.input_norm = _make_group_norm(hidden_channels)
        self.input_zero_gate = RealZeroOrderGate(
            hidden_channels, condition_hidden=zero_gate_hidden
        )
        self.input_activation = nn.SiLU()

        self.spatial_blocks = nn.ModuleList(
            [
                RealResidualBlock(hidden_channels, kernel_size=kernel_size)
                for _ in range(num_iterations)
            ]
        )
        self.interactions = nn.ModuleList(
            [
                RealEquivariantLayerInteraction(hidden_channels)
                for _ in range(num_iterations)
            ]
        )
        self.iteration_zero_gates = nn.ModuleList(
            [
                RealZeroOrderGate(
                    hidden_channels, condition_hidden=zero_gate_hidden
                )
                for _ in range(num_iterations)
            ]
        )

        self.readout_a = nn.Conv2d(
            hidden_channels,
            readout_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.readout_b = nn.Conv2d(
            hidden_channels,
            readout_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.readout_norm_a = _make_group_norm(readout_channels)
        self.readout_norm_b = _make_group_norm(readout_channels)
        readout_width = 2 * readout_channels
        self.readout_mix = nn.Conv2d(readout_width, readout_width, kernel_size=1)
        self.llr_head = nn.Sequential(
            nn.Conv2d(readout_width + 2, hidden_real, kernel_size=3, padding=1),
            _make_group_norm(hidden_real),
            nn.ReLU(),
            nn.Conv2d(hidden_real, hidden_real, kernel_size=3, padding=1),
            _make_group_norm(hidden_real),
            nn.ReLU(),
            nn.Conv2d(hidden_real, bits_per_symbol, kernel_size=1),
        )

        actual_count = sum(parameter.numel() for parameter in self.parameters())
        if actual_count != predicted_count:
            raise RuntimeError(
                "Real-CNN parameter-count formula drifted from implementation: "
                f"predicted={predicted_count}, actual={actual_count}."
            )

    def forward(self, y, h_hat, pilot_mask, n0, layer_mask=None):
        if y.dim() != 4 or not torch.is_complex(y):
            raise ValueError("Y must be complex with shape [B, N_rx, T, F].")
        if h_hat.dim() != 5 or not torch.is_complex(h_hat):
            raise ValueError(
                "H_hat must be complex with shape [B, L, N_rx, T, F]."
            )
        batch_size, num_rx_ant, num_symbols, num_subcarriers = y.shape
        num_layers = h_hat.shape[1]
        expected_h_shape = (
            batch_size,
            num_layers,
            num_rx_ant,
            num_symbols,
            num_subcarriers,
        )
        if tuple(h_hat.shape) != expected_h_shape:
            raise ValueError("Y and H_hat dimensions are inconsistent.")
        if num_rx_ant != self.num_rx_ant:
            raise ValueError(f"Expected {self.num_rx_ant} receive antennas.")
        if pilot_mask.shape != (
            batch_size,
            num_layers,
            1,
            num_symbols,
            num_subcarriers,
        ):
            raise ValueError("pilot_mask must have shape [B, L, 1, T, F].")
        if layer_mask is None:
            layer_mask = torch.ones(
                batch_size, num_layers, dtype=torch.bool, device=y.device
            )

        n0_grid = _prepare_n0_grid(
            n0, batch_size, num_symbols, num_subcarriers, y.device
        )
        layer_n0 = n0_grid.unsqueeze(1).expand(-1, num_layers, -1, -1, -1)
        zero_features = torch.cat(
            [pilot_mask.to(dtype=torch.float32), layer_n0], dim=2
        )
        flat_zero = zero_features.reshape(
            batch_size * num_layers, 2, num_symbols, num_subcarriers
        )

        shared_y = y.unsqueeze(1).expand(-1, num_layers, -1, -1, -1)
        complex_input = torch.cat([shared_y, h_hat], dim=2)
        real_input = torch.cat(
            [complex_input.real, complex_input.imag], dim=2
        ).reshape(
            batch_size * num_layers,
            4 * num_rx_ant,
            num_symbols,
            num_subcarriers,
        )
        z = real_input * self.input_scale.view(1, -1, 1, 1)
        z = self.input_activation(self.input_norm(self.input_proj(z)))
        z = self.input_zero_gate(z, flat_zero)

        hidden_channels = z.shape[1]
        for spatial_block, interaction, zero_gate in zip(
            self.spatial_blocks, self.interactions, self.iteration_zero_gates
        ):
            z = spatial_block(z)
            z = z.reshape(
                batch_size,
                num_layers,
                hidden_channels,
                num_symbols,
                num_subcarriers,
            )
            z = interaction(z, layer_mask=layer_mask)
            z = z.reshape(
                batch_size * num_layers,
                hidden_channels,
                num_symbols,
                num_subcarriers,
            )
            z = zero_gate(z, flat_zero)

        readout = torch.cat(
            [
                self.readout_norm_a(self.readout_a(z)),
                self.readout_norm_b(self.readout_b(z)),
            ],
            dim=1,
        )
        readout = self.readout_mix(readout)
        llr_input = torch.cat(
            [
                readout,
                pilot_mask.reshape(
                    batch_size * num_layers, 1, num_symbols, num_subcarriers
                ),
                layer_n0.reshape(
                    batch_size * num_layers, 1, num_symbols, num_subcarriers
                ),
            ],
            dim=1,
        )
        logits = self.llr_head(llr_input).reshape(
            batch_size,
            num_layers,
            self.bits_per_symbol,
            num_symbols,
            num_subcarriers,
        )
        active = layer_mask.to(dtype=logits.dtype).view(
            batch_size, num_layers, 1, 1, 1
        )
        return logits * active
