import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexConv2d(nn.Module):
    """
    Complex-valued 2D convolution implemented by two real convolutions.

    Let z = zr + j zi, W = A + j B.
    Then Wz = (A*zr - B*zi) + j(A*zi + B*zr).

    For non-zero charge features, set bias=False. A complex bias breaks
    phase equivariance for charge +1 / -1 / etc.
    For zero-charge features, bias=True is allowed.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.real_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=False
        )
        self.imag_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=False
        )

        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_channels))
            self.bias_imag = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def forward(self, z):
        if not torch.is_complex(z):
            raise TypeError("ComplexConv2d expects a complex tensor.")

        zr = z.real
        zi = z.imag

        real = self.real_conv(zr) - self.imag_conv(zi)
        imag = self.real_conv(zi) + self.imag_conv(zr)

        if self.bias_real is not None:
            real = real + self.bias_real.view(1, -1, 1, 1)
            imag = imag + self.bias_imag.view(1, -1, 1, 1)

        return torch.complex(real, imag)


class WidelyLinearComplexConv2d(nn.Module):
    """Widely-linear complex convolution ``W*z + V*conj(z)``.

    A standard complex convolution only represents complex-linear maps.  The
    independent conjugate branch adds the anti-linear component required to
    represent an arbitrary real-linear map between complex feature spaces.
    Both branches are scaled by ``1/sqrt(2)`` so their summed initialization
    has approximately the same variance as :class:`ComplexConv2d`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        bias=False,
    ):
        super().__init__()
        self.linear = ComplexConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.antilinear = ComplexConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_channels))
            self.bias_imag = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def forward(self, z):
        if not torch.is_complex(z):
            raise TypeError(
                "WidelyLinearComplexConv2d expects a complex tensor."
            )
        out = (
            self.linear(z) + self.antilinear(torch.conj(z))
        ) / math.sqrt(2.0)
        if self.bias_real is not None:
            bias = torch.complex(self.bias_real, self.bias_imag).view(
                1, -1, 1, 1
            )
            out = out + bias
        return out


class AmplitudeGate(nn.Module):
    """
    Phase-equivariant nonlinearity:
        z -> gate(|z|) * z

    Since gate is real-valued and depends only on |z|, this preserves charge.
    """
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, z):
        gate = torch.sigmoid(self.scale * torch.abs(z) + self.bias)
        return gate * z


class AmplitudeSwiGLUGate(nn.Module):
    """
    Phase-equivariant residual SwiGLU-style gate.

    The gate and value branches only read |z|, so the multiplier is real-valued
    and invariant to a global phase rotation. The residual form starts close to
    the identity map and avoids repeatedly shrinking features with sigmoid gates.
    """
    def __init__(self, channels, residual_scale=0.1):
        super().__init__()
        self.proj = nn.Conv2d(channels, 2 * channels, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, z):
        if not torch.is_complex(z):
            raise TypeError("AmplitudeSwiGLUGate expects a complex tensor.")

        gate, value = self.proj(torch.abs(z)).chunk(2, dim=1)
        multiplier = 1.0 + self.residual_scale * F.silu(gate) * value
        return multiplier * z


class ComplexRMSNorm2d(nn.Module):
    """
    RMS normalization for non-zero charge complex features.

    The scale is computed from |z|, so a global phase rotation changes only the
    output phase and preserves equivariance:
        norm(exp(j phi) z) = exp(j phi) norm(z)
    """
    def __init__(self, channels, eps=1e-6, affine=True):
        super().__init__()
        self.eps = eps

        if affine:
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        else:
            self.register_parameter("weight", None)

    def forward(self, z):
        if not torch.is_complex(z):
            raise TypeError("ComplexRMSNorm2d expects a complex tensor.")

        rms = torch.sqrt(torch.mean(torch.abs(z) ** 2, dim=(-2, -1), keepdim=True) + self.eps)
        out = z / rms

        if self.weight is not None:
            out = self.weight * out

        return out


class ChargeBranch(nn.Module):
    """
    Branch for a fixed non-zero charge, e.g. +1 or -1.
    All operations preserve the charge.
    """
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        kernel_size=3,
        use_norm=True,
        gate_type="swiglu",
    ):
        super().__init__()
        padding = kernel_size // 2

        if gate_type not in ["sigmoid", "swiglu"]:
            raise ValueError("gate_type must be 'sigmoid' or 'swiglu'.")

        layers = []
        c_in = in_channels
        for _ in range(num_layers):
            layers.append(
                ComplexConv2d(
                    c_in, hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,  # critical for non-zero charge
                )
            )
            if use_norm:
                layers.append(ComplexRMSNorm2d(hidden_channels))
            if gate_type == "sigmoid":
                layers.append(AmplitudeGate(hidden_channels))
            else:
                layers.append(AmplitudeSwiGLUGate(hidden_channels))
            c_in = hidden_channels

        self.layers = nn.ModuleList(layers)

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z


class EquivariantInteraction(nn.Module):
    """
    Equivariant interaction:
        charge +1 feature * charge -1 feature -> charge 0 feature.

    The full outer-product interaction produces C_pos * C_neg channels.
    For large C this can be heavy, so keep hidden_complex modest in experiments.
    """
    def __init__(self, c_pos, c_neg, c_zero):
        super().__init__()
        self.c_pos = c_pos
        self.c_neg = c_neg

        self.compress = ComplexConv2d(
            c_pos * c_neg,
            c_zero,
            kernel_size=1,
            padding=0,
            bias=True,  # zero-charge features may use bias
        )

    def forward(self, feat_p1, feat_n1):
        if not (torch.is_complex(feat_p1) and torch.is_complex(feat_n1)):
            raise TypeError("EquivariantInteraction expects complex tensors.")

        b, c_pos, h, w = feat_p1.shape
        _, c_neg, _, _ = feat_n1.shape

        if c_pos != self.c_pos or c_neg != self.c_neg:
            raise ValueError(
                f"Expected channels ({self.c_pos}, {self.c_neg}), "
                f"got ({c_pos}, {c_neg})."
            )

        p = feat_p1.unsqueeze(2)   # (B, C_pos, 1, H, W)
        n = feat_n1.unsqueeze(1)   # (B, 1, C_neg, H, W)

        zero_raw = p * n           # (B, C_pos, C_neg, H, W)
        zero_raw = zero_raw.reshape(b, c_pos * c_neg, h, w)

        return self.compress(zero_raw)
