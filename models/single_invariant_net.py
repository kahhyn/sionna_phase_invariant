import torch
import torch.nn as nn

from .complex_layers import ComplexConv2d, AmplitudeGate,ComplexRMSNorm2d, AmplitudeSwiGLUGate


def _prepare_complex_input(Y, H_hat):
    if Y.dim() == 3:
        Y = Y.unsqueeze(1)
    if H_hat.dim() == 3:
        H_hat = H_hat.unsqueeze(1)

    if not torch.is_complex(Y) or not torch.is_complex(H_hat):
        raise TypeError("Y and H_hat must be complex tensors.")

    return torch.cat([Y, H_hat], dim=1)


def _prepare_zero_features(P, N0, batch_size, t, f, device):
    if P.dim() == 3:
        P = P.unsqueeze(1)

    P = P.to(device=device, dtype=torch.float32)

    if N0.dim() == 1:
        N0 = N0.view(batch_size, 1, 1, 1)
    elif N0.dim() == 2:
        N0 = N0.view(batch_size, 1, 1, 1)
    elif N0.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported N0 shape: {N0.shape}")

    N0_grid = torch.log(N0.to(device=device, dtype=torch.float32) + 1e-12)
    N0_grid = N0_grid.expand(batch_size, 1, t, f)

    return P, N0_grid

def _make_group_norm(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ComplexResidualBlock(nn.Module):
    """
    Charge +1 residual block.

    All complex conv layers use bias=False so that:
        F -> exp(jφ)F
    is preserved.
    """

    def __init__(self, channels, kernel_size=3, use_norm=True,gate_type="swiglu"):
        super().__init__()
        padding = kernel_size // 2
        if gate_type not in ["sigmoid", "swiglu"]:
            raise ValueError("gate_type must be 'sigmoid' or 'swiglu'.")

        self.conv1 = ComplexConv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        if gate_type == "sigmoid":
            self.gate1 = AmplitudeGate(channels)
        else:
            self.gate1 = AmplitudeSwiGLUGate(channels)

        self.conv2 = ComplexConv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        if gate_type == "sigmoid":
            self.gate2 = AmplitudeGate(channels)
        else:
            self.gate2 = AmplitudeSwiGLUGate(channels)
        
        self.norm1 = ComplexRMSNorm2d(channels) if use_norm else nn.Identity()
        self.norm2 = ComplexRMSNorm2d(channels) if use_norm else nn.Identity()

    def forward(self, z):
        residual = z

        out = self.conv1(z)
        out = self.norm1(out)
        out = self.gate1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + residual
        out = self.gate2(out)

        return out


class HermitianInvariantReadout(nn.Module):
    """
    Single-branch invariant readout.

    Input:
        F: charge +1 feature, shape (B, C, T, F)

    Output:
        zero-order real feature.

    Uses:
        G_ij = F_i * conj(F_j)

    This is invariant under:
        F -> exp(jφ)F
    """

    def __init__(self, in_channels, out_channels, mode="low_rank"):
        super().__init__()

        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "full":
            # full C x C Hermitian products
            invariant_channels = in_channels * in_channels * 2
            self.compress = nn.Conv2d(
                invariant_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )

        elif mode == "low_rank":
            # Learn two equivariant projections A(F), B(F),
            # then use A(F) * conj(B(F)).
            self.proj_a = ComplexConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.proj_b = ComplexConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            )
            self.norm_porj_a = ComplexRMSNorm2d(out_channels)
            self.norm_porj_b = ComplexRMSNorm2d(out_channels)
            
        else:
            raise ValueError("mode must be 'full' or 'low_rank'.")

    def forward(self, z):
        if self.mode == "full":
            b, c, t, f = z.shape

            zi = z.unsqueeze(2)             # (B, C, 1, T, F)
            zj = torch.conj(z).unsqueeze(1) # (B, 1, C, T, F)

            g = zi * zj                     # (B, C, C, T, F)
            g = g.reshape(b, c * c, t, f)

            g_real = torch.cat([g.real, g.imag], dim=1)
            return self.compress(g_real)

        if self.mode == "low_rank":
            a = self.proj_a(z)
            b = self.proj_b(z)
            a = self.norm_porj_a(a)
            b = self.norm_porj_b(b)
            

            g = a * torch.conj(b)

            # g is zero-order complex feature.
            # We expose real and imaginary parts to real LLR head.
            return torch.cat([g.real, g.imag], dim=1)


class SingleBranchPhaseInvariantReceiver(nn.Module):
    """
    Single-branch phase-invariant receiver.

    This avoids the duplicated positive/negative branches.

    Pipeline:
        [Y, H_hat] charge +1
            ↓
        complex equivariant backbone
            ↓
        invariant readout F * conj(F)
            ↓
        concat P, log(N0)
            ↓
        real-valued LLR head
    """

    def __init__(
        self,
        hidden_complex=32,
        zero_real=32,
        hidden_real=32,
        bits_per_symbol=2,
        num_blocks=3,
        kernel_size=3,
        use_norm=True,
        gate_type="swiglu",
        readout_mode="low_rank",
    ):
        super().__init__()
        if gate_type not in ["sigmoid", "swiglu"]:
            raise ValueError("gate_type must be 'sigmoid' or 'swiglu'.")
        

        self.input_proj = ComplexConv2d(
            in_channels=2,
            out_channels=hidden_complex,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        if gate_type == "sigmoid":
            self.input_gate = AmplitudeGate(hidden_complex)
        else:
            self.input_gate = AmplitudeSwiGLUGate(hidden_complex)

        self.blocks = nn.ModuleList(
            [
                ComplexResidualBlock(
                    channels=hidden_complex,
                    kernel_size=kernel_size,
                    gate_type=gate_type,
                    use_norm=use_norm,
                )
                for _ in range(num_blocks)
            ]
        )

        self.readout = HermitianInvariantReadout(
            in_channels=hidden_complex,
            out_channels=zero_real,
            mode=readout_mode,
        )

        if readout_mode == "low_rank":
            readout_channels = 2 * zero_real
        elif readout_mode == "full":
            readout_channels = zero_real
        else:
            raise ValueError("Unsupported readout_mode.")
        
        self.mixchannel = nn.Conv2d(
            readout_channels,
            readout_channels,
            kernel_size=1,
            )
        llr_in_channels = readout_channels + 2  # + P, log(N0)

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
        z = _prepare_complex_input(Y, H_hat)

        b, _, t, f = z.shape
        P, N0_grid = _prepare_zero_features(P, N0, b, t, f, z.device)

        z = self.input_proj(z)
        z = self.input_gate(z)

        for block in self.blocks:
            z = block(z)

        invariant_feat = self.readout(z)
        invariant_feat = self.mixchannel(invariant_feat)

        x = torch.cat(
            [
                invariant_feat,
                P,
                N0_grid,
            ],
            dim=1,
        )

        return self.llr_head(x)