"""Phase-equivariant residual denoising for interpolated LS channel estimates."""

from __future__ import annotations

import torch
import torch.nn as nn

from .complex_layers import AmplitudeGate, AmplitudeSwiGLUGate, ComplexConv2d
from .single_invariant_net import (
    ComplexResidualBlock,
    ZeroOrderAmplitudeGate,
    _prepare_zero_features,
)


def _ensure_single_complex_grid(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return ``x`` as ``[B, 1, T, F]`` and whether a channel was inserted."""
    if not torch.is_complex(x):
        raise TypeError("H_hat must be a complex tensor.")
    if x.dim() == 3:
        return x.unsqueeze(1), True
    if x.dim() == 4 and x.shape[1] == 1:
        return x, False
    raise ValueError("H_hat must have shape [B,T,F] or [B,1,T,F].")


class EquivariantHResidualDenoiser(nn.Module):
    """Refine an interpolated LS estimate while preserving charge +1.

    The final projection is initialized to zero, so the initial module is an
    exact identity map. All learned complex maps are bias-free and all P/N0
    conditioning is real-valued, hence

        D(exp(j phi) H_hat, P, N0) = exp(j phi) D(H_hat, P, N0).
    """

    def __init__(
        self,
        hidden_complex: int = 16,
        num_blocks: int = 2,
        kernel_size: int = 3,
        use_norm: bool = True,
        gate_type: str = "swiglu",
        condition_hidden: int = 16,
        condition_mode: str = "p_n0",
    ) -> None:
        super().__init__()
        if hidden_complex <= 0:
            raise ValueError("hidden_complex must be positive.")
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative.")
        if gate_type not in {"sigmoid", "swiglu"}:
            raise ValueError("gate_type must be sigmoid or swiglu.")
        if condition_mode not in {"p_n0", "p_only", "n0_only"}:
            raise ValueError(
                "condition_mode must be p_n0, p_only, or n0_only."
            )

        padding = kernel_size // 2
        self.condition_mode = condition_mode
        self.input_proj = ComplexConv2d(
            1,
            hidden_complex,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.input_gate = (
            AmplitudeGate(hidden_complex)
            if gate_type == "sigmoid"
            else AmplitudeSwiGLUGate(hidden_complex)
        )
        self.input_zero_gate = ZeroOrderAmplitudeGate(
            hidden_complex,
            condition_hidden=condition_hidden,
        )
        self.blocks = nn.ModuleList(
            [
                ComplexResidualBlock(
                    channels=hidden_complex,
                    kernel_size=kernel_size,
                    use_norm=use_norm,
                    gate_type=gate_type,
                )
                for _ in range(num_blocks)
            ]
        )
        self.block_zero_gates = nn.ModuleList(
            [
                ZeroOrderAmplitudeGate(
                    hidden_complex,
                    condition_hidden=condition_hidden,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_proj = ComplexConv2d(
            hidden_complex,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        # Start from H_refined == H_hat. The output layer learns first; after
        # it leaves zero, gradients also reach the earlier denoising layers.
        nn.init.zeros_(self.output_proj.real_conv.weight)
        nn.init.zeros_(self.output_proj.imag_conv.weight)

    def _select_condition(
        self,
        pilot_mask: torch.Tensor,
        n0_grid: torch.Tensor,
    ) -> torch.Tensor:
        if self.condition_mode == "p_only":
            n0_grid = torch.zeros_like(n0_grid)
        elif self.condition_mode == "n0_only":
            pilot_mask = torch.zeros_like(pilot_mask)
        return torch.cat([pilot_mask, n0_grid], dim=1)

    def forward(
        self,
        h_hat: torch.Tensor,
        pilot_mask: torch.Tensor,
        n0: torch.Tensor,
    ) -> torch.Tensor:
        h_grid, inserted_channel = _ensure_single_complex_grid(h_hat)
        batch_size, _, num_symbols, num_subcarriers = h_grid.shape
        pilot_mask, n0_grid = _prepare_zero_features(
            pilot_mask,
            n0,
            batch_size,
            num_symbols,
            num_subcarriers,
            h_grid.device,
        )
        condition = self._select_condition(pilot_mask, n0_grid)

        z = self.input_proj(h_grid)
        z = self.input_zero_gate(z, condition)
        z = self.input_gate(z)
        for block, zero_gate in zip(self.blocks, self.block_zero_gates):
            z = block(z)
            z = zero_gate(z, condition)

        h_refined = h_grid + self.output_proj(z)
        return h_refined[:, 0] if inserted_channel else h_refined


class HRefinedReceiver(nn.Module):
    """Apply a shared H denoiser before a receiver with Y/H/P/N0 inputs."""

    def __init__(
        self,
        receiver: nn.Module,
        denoiser: EquivariantHResidualDenoiser,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.receiver = receiver

    def forward(self, y, h_hat, pilot_mask, n0):
        h_refined = self.denoiser(h_hat, pilot_mask, n0)
        return self.receiver(y, h_refined, pilot_mask, n0)

    def forward_with_aux(self, y, h_hat, pilot_mask, n0):
        h_refined = self.denoiser(h_hat, pilot_mask, n0)
        logits = self.receiver(y, h_refined, pilot_mask, n0)
        return logits, {"H_refined": h_refined}


def complex_nmse_per_sample(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return one normalized channel-estimation error for each batch item."""
    if estimate.dim() == 4 and estimate.shape[1] == 1:
        estimate = estimate[:, 0]
    if target.dim() == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if estimate.shape != target.shape:
        raise ValueError(
            f"NMSE shape mismatch: estimate={estimate.shape}, "
            f"target={target.shape}."
        )
    reduce_dims = tuple(range(1, estimate.dim()))
    error_power = (estimate - target).abs().square().sum(dim=reduce_dims)
    target_power = target.abs().square().sum(dim=reduce_dims)
    return error_power / target_power.clamp_min(eps)


def complex_nmse(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Batch-averaged channel-estimation NMSE in linear scale."""
    return complex_nmse_per_sample(estimate, target, eps=eps).mean()
