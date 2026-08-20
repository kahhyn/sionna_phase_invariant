"""Parameter-matched widely-linear complex CNN for SU-MIMO OFDM."""

from __future__ import annotations

import math

from .complex_layers import WidelyLinearComplexConv2d
from .su_mimo_invariant_net import SUMIMOPhaseInvariantReceiver


def _zero_gate_parameter_count(channels: int, condition_hidden: int) -> int:
    return 19 * condition_hidden + condition_hidden * channels + channels


def _amplitude_swiglu_parameter_count(channels: int) -> int:
    return 2 * channels * channels + 2 * channels + 1


def _widely_linear_parameter_count(
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
    """Closed-form count for :class:`SUMIMOWidelyLinearReceiver`."""
    r = num_rx_ant
    h = hidden_channels
    z = readout_channels
    u = hidden_real
    b = bits_per_symbol
    q = kernel_size**2
    zero_gate = _zero_gate_parameter_count(h, zero_gate_hidden)
    amplitude_gate = _amplitude_swiglu_parameter_count(h)

    count = 2 * r  # real input scaling for complex Y/H channels
    count += 8 * r * h * q  # widely-linear input projection
    count += h + zero_gate + amplitude_gate

    residual_block = 8 * h * h * q + 2 * amplitude_gate + 2 * h
    layer_interaction = (
        4 * h * h
        + h
        + amplitude_gate
        + 8 * h * h
        + h
        + amplitude_gate
    )
    count += num_iterations * (
        residual_block + layer_interaction + zero_gate
    )

    count += 8 * h * z * 9 + 2 * z  # two widely-linear readout projections
    count += 4 * z * z + 2 * z  # real 1x1 readout mixer
    count += (2 * z + 2) * u * 9 + u + 2 * u
    count += u * u * 9 + u + 2 * u
    count += u * b + b
    return count


def resolve_widely_linear_widths(
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
    """Choose backbone/readout widths nearest to a real-parameter budget."""
    if target_parameter_count <= 0:
        raise ValueError("target_parameter_count must be positive.")
    center = max(2, round(hidden_complex / math.sqrt(2.0)))
    hidden_candidates = range(max(2, center - 12), hidden_complex + 1)
    readout_candidates = range(max(2, zero_real // 3), 2 * zero_real + 1)
    candidates = []
    for hidden_channels in hidden_candidates:
        for readout_channels in readout_candidates:
            count = _widely_linear_parameter_count(
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


class SUMIMOWidelyLinearReceiver(SUMIMOPhaseInvariantReceiver):
    """Phase-sensitive SU-MIMO receiver with ``Wz + V*conj(z)`` convolutions.

    The topology, amplitude gates, zero-order conditioning, layer interaction,
    and real LLR head match :class:`SUMIMOPhaseSensitiveReceiver`.  Only the
    complex-linear convolution constraint is relaxed.  Widths are resolved to
    the same real-parameter budget as the standard complex control.
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
        if target_parameter_count is None:
            raise ValueError(
                "target_parameter_count is required for the matched "
                "widely-linear CNN."
            )
        hidden_channels, readout_channels, predicted_count = (
            resolve_widely_linear_widths(
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
        super().__init__(
            num_rx_ant=num_rx_ant,
            hidden_complex=hidden_channels,
            zero_real=readout_channels,
            hidden_real=hidden_real,
            bits_per_symbol=bits_per_symbol,
            num_iterations=num_iterations,
            kernel_size=kernel_size,
            zero_gate_hidden=zero_gate_hidden,
            phase_invariant_readout=False,
            complex_conv_cls=WidelyLinearComplexConv2d,
        )
        self.widely_linear_hidden_channels = hidden_channels
        self.widely_linear_readout_channels = readout_channels
        self.target_parameter_count = int(target_parameter_count)
        self.predicted_parameter_count = predicted_count
        self.resolved_model_config = {
            "widely_linear_hidden_channels": hidden_channels,
            "widely_linear_readout_channels": readout_channels,
            "target_parameter_count": self.target_parameter_count,
            "predicted_parameter_count": predicted_count,
        }

        actual_count = sum(parameter.numel() for parameter in self.parameters())
        if actual_count != predicted_count:
            raise RuntimeError(
                "Widely-linear parameter-count formula drifted from the "
                f"implementation: predicted={predicted_count}, "
                f"actual={actual_count}."
            )
