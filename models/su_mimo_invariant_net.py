"""Phase- and layer-permutation-equivariant receiver for SU-MIMO OFDM."""

from __future__ import annotations

import torch
import torch.nn as nn

from .complex_layers import (
    AmplitudeSwiGLUGate,
    ComplexConv2d,
    ComplexRMSNorm2d,
)
from .single_invariant_net import (
    ComplexResidualBlock,
    HermitianInvariantReadout,
    ZeroOrderAmplitudeGate,
    _make_group_norm,
)


def _prepare_n0_grid(n0, batch_size, num_symbols, num_subcarriers, device):
    if n0.dim() == 1:
        n0 = n0.view(batch_size, 1, 1, 1)
    elif n0.dim() == 2:
        n0 = n0.view(batch_size, 1, 1, 1)
    elif n0.dim() != 4:
        raise ValueError(f"Unsupported N0 shape: {n0.shape}")
    grid = torch.log(n0.to(device=device, dtype=torch.float32) + 1e-12)
    return grid.expand(batch_size, 1, num_symbols, num_subcarriers)


class EquivariantLayerInteraction(nn.Module):
    """Masked DeepSets interaction preserving charge one and layer order."""

    def __init__(self, channels):
        super().__init__()
        self.message_proj = ComplexConv2d(
            channels, channels, kernel_size=1, padding=0, bias=False
        )
        self.message_norm = ComplexRMSNorm2d(channels)
        self.message_gate = AmplitudeSwiGLUGate(channels)
        self.update_proj = ComplexConv2d(
            2 * channels, channels, kernel_size=1, padding=0, bias=False
        )
        self.update_norm = ComplexRMSNorm2d(channels)
        self.update_gate = AmplitudeSwiGLUGate(channels)

    def forward(self, z, layer_mask=None):
        if z.dim() != 5 or not torch.is_complex(z):
            raise ValueError("z must be complex with shape [B, L, C, T, F].")
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
        message = self.message_proj(flat)
        message = self.message_norm(message)
        message = self.message_gate(message)
        message = message.reshape(z.shape)

        active = layer_mask.to(dtype=z.real.dtype).view(
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
        update = self.update_proj(update_input)
        update = self.update_norm(update)
        update = self.update_gate(update)
        update = update.reshape(z.shape)
        return (z + update) * active


class EquivariantComplexReadout(nn.Module):
    """Parameter-matched phase-sensitive counterpart to Hermitian readout."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj_a = ComplexConv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.proj_b = ComplexConv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm_proj_a = ComplexRMSNorm2d(out_channels)
        self.norm_proj_b = ComplexRMSNorm2d(out_channels)

    def forward(self, z):
        projected = self.norm_proj_a(self.proj_a(z))
        projected = projected + self.norm_proj_b(self.proj_b(z))
        return torch.cat([projected.real, projected.imag], dim=1)


class SUMIMOPhaseInvariantReceiver(nn.Module):
    """Common-phase invariant and layer-permutation equivariant SU-MIMO RX.

    The model consumes one shared received grid and one channel estimate per
    layer. All per-layer modules share weights. Intermediate complex states
    transform with charge +1 under a common rotation of ``Y`` and ``H_hat``;
    only the final Hermitian readout converts them to charge-zero features.
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
        phase_invariant_readout=True,
    ):
        super().__init__()
        if num_rx_ant <= 0 or num_iterations <= 0:
            raise ValueError("num_rx_ant and num_iterations must be positive.")
        self.num_rx_ant = int(num_rx_ant)
        self.bits_per_symbol = int(bits_per_symbol)
        self.phase_invariant_readout = bool(phase_invariant_readout)
        self.input_scale = nn.Parameter(torch.ones(2 * self.num_rx_ant))

        self.input_proj = ComplexConv2d(
            2 * self.num_rx_ant,
            hidden_complex,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.input_norm = ComplexRMSNorm2d(hidden_complex)
        self.input_zero_gate = ZeroOrderAmplitudeGate(
            hidden_complex, condition_hidden=zero_gate_hidden
        )
        self.input_gate = AmplitudeSwiGLUGate(hidden_complex)

        self.spatial_blocks = nn.ModuleList(
            [
                ComplexResidualBlock(
                    hidden_complex,
                    kernel_size=kernel_size,
                    use_norm=True,
                    gate_type="swiglu",
                )
                for _ in range(num_iterations)
            ]
        )
        self.interactions = nn.ModuleList(
            [
                EquivariantLayerInteraction(hidden_complex)
                for _ in range(num_iterations)
            ]
        )
        self.iteration_zero_gates = nn.ModuleList(
            [
                ZeroOrderAmplitudeGate(
                    hidden_complex, condition_hidden=zero_gate_hidden
                )
                for _ in range(num_iterations)
            ]
        )

        if self.phase_invariant_readout:
            self.readout = HermitianInvariantReadout(
                in_channels=hidden_complex,
                out_channels=zero_real,
                mode="low_rank",
            )
        else:
            self.readout = EquivariantComplexReadout(
                in_channels=hidden_complex,
                out_channels=zero_real,
            )
        readout_channels = 2 * zero_real
        self.readout_mix = nn.Conv2d(
            readout_channels, readout_channels, kernel_size=1
        )
        self.llr_head = nn.Sequential(
            nn.Conv2d(readout_channels + 2, hidden_real, kernel_size=3, padding=1),
            _make_group_norm(hidden_real),
            nn.ReLU(),
            nn.Conv2d(hidden_real, hidden_real, kernel_size=3, padding=1),
            _make_group_norm(hidden_real),
            nn.ReLU(),
            nn.Conv2d(hidden_real, bits_per_symbol, kernel_size=1),
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

        shared_y = y.unsqueeze(1).expand(-1, num_layers, -1, -1, -1)
        z = torch.cat([shared_y, h_hat], dim=2).reshape(
            batch_size * num_layers,
            2 * num_rx_ant,
            num_symbols,
            num_subcarriers,
        )
        z = z * self.input_scale.view(1, -1, 1, 1)
        flat_zero = zero_features.reshape(
            batch_size * num_layers, 2, num_symbols, num_subcarriers
        )
        z = self.input_proj(z)
        z = self.input_norm(z)
        z = self.input_zero_gate(z, flat_zero)
        z = self.input_gate(z)

        hidden_complex = z.shape[1]
        for spatial_block, interaction, zero_gate in zip(
            self.spatial_blocks, self.interactions, self.iteration_zero_gates
        ):
            z = spatial_block(z)
            z = z.reshape(
                batch_size,
                num_layers,
                hidden_complex,
                num_symbols,
                num_subcarriers,
            )
            z = interaction(z, layer_mask=layer_mask)
            z = z.reshape(
                batch_size * num_layers,
                hidden_complex,
                num_symbols,
                num_subcarriers,
            )
            z = zero_gate(z, flat_zero)

        invariant = self.readout(z)
        invariant = self.readout_mix(invariant)
        llr_input = torch.cat(
            [
                invariant,
                pilot_mask.reshape(
                    batch_size * num_layers, 1, num_symbols, num_subcarriers
                ),
                layer_n0.reshape(
                    batch_size * num_layers, 1, num_symbols, num_subcarriers
                ),
            ],
            dim=1,
        )
        logits = self.llr_head(llr_input)
        logits = logits.reshape(
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


class SUMIMOPhaseSensitiveReceiver(SUMIMOPhaseInvariantReceiver):
    """Layer-equivariant matched control without common-phase invariance."""

    def __init__(self, **kwargs):
        super().__init__(phase_invariant_readout=False, **kwargs)
