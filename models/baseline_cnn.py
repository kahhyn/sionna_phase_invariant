import torch
import torch.nn as nn


def _make_group_norm(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def _ensure_complex_grid(x):
    """
    Accepts (B,T,F) or (B,1,T,F) complex tensor.
    Returns (B,T,F) complex tensor for the baseline models.
    """
    if x.dim() == 4:
        if x.shape[1] != 1:
            raise ValueError("Baseline models currently expect single complex channel.")
        x = x[:, 0]
    return x


def _prepare_zero_features(P, N0, batch_size, t, f, device):
    """
    P:  (B,1,T,F) or (B,T,F), real
    N0: (B,), (B,1), or (B,1,1,1), real
    Returns P and log(N0) grid, each shaped (B,1,T,F).
    """
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


class RealZeroOrderAmplitudeGate(nn.Module):
    """Condition a real feature map on the pilot mask and log noise power."""

    def __init__(self, channels, condition_hidden=16):
        super().__init__()
        if condition_hidden <= 0:
            raise ValueError("condition_hidden must be positive.")
        self.condition_net = nn.Sequential(
            nn.Conv2d(2, condition_hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(condition_hidden, channels, kernel_size=1),
        )

        # Match the invariant model's identity initialization for P/N0 gates.
        nn.init.zeros_(self.condition_net[-1].weight)
        nn.init.zeros_(self.condition_net[-1].bias)

    def forward(self, x, zero_features):
        if zero_features.shape[1] != 2:
            raise ValueError("zero_features must contain P and log(N0).")
        scale = 2.0 * torch.sigmoid(self.condition_net(zero_features))
        return x * scale


class RealResidualBlock(nn.Module):
    """Two-convolution residual block used by the matched real baseline."""

    def __init__(self, channels, kernel_size=3, use_norm=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding
        )
        self.norm1 = _make_group_norm(channels) if use_norm else nn.Identity()
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding
        )
        self.norm2 = _make_group_norm(channels) if use_norm else nn.Identity()
        self.activation = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class RealImagCNN(nn.Module):
    """Parameter/depth-matched ordinary real-valued CNN baseline.

    The semantic inputs are exactly those of ``single_branch_n0_gate``:
    ``Y``, ``H_hat``, the pilot mask ``P``, and ``N0``. Complex tensors are
    represented by their real and imaginary parts; no invariant feature is
    constructed. P/log(N0) condition the trunk at the same three macro
    locations as the two-block invariant model and are concatenated before
    the common LLR-head pattern.

    With the formal experiment settings (hidden=64, trunk_hidden=50,
    num_blocks=2, condition_hidden=16), this model has 204,558 trainable
    parameters versus 204,599 for ``single_branch_n0_gate`` (-0.02%). Both
    models have ten sequential convolutional stages on their longest spatial
    path. This model intentionally does not guarantee common-phase invariance.
    """

    def __init__(
        self,
        hidden=64,
        trunk_hidden=50,
        bits_per_symbol=2,
        num_blocks=2,
        kernel_size=3,
        use_norm=True,
        condition_hidden=16,
    ):
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive.")
        padding = kernel_size // 2

        # Y and H_hat are split into four ordinary real-valued channels.
        self.input_conv = nn.Conv2d(
            4,
            trunk_hidden,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.input_norm = (
            _make_group_norm(trunk_hidden) if use_norm else nn.Identity()
        )
        self.input_gate = RealZeroOrderAmplitudeGate(
            trunk_hidden, condition_hidden=condition_hidden
        )
        self.input_activation = nn.SiLU()

        self.blocks = nn.ModuleList(
            [
                RealResidualBlock(
                    trunk_hidden,
                    kernel_size=kernel_size,
                    use_norm=use_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.block_zero_gates = nn.ModuleList(
            [
                RealZeroOrderAmplitudeGate(
                    trunk_hidden, condition_hidden=condition_hidden
                )
                for _ in range(num_blocks)
            ]
        )

        # Match the invariant readout's 64 real output channels and 1x1 mix.
        self.readout = nn.Conv2d(
            trunk_hidden,
            hidden,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.readout_norm = _make_group_norm(hidden) if use_norm else nn.Identity()
        self.readout_activation = nn.SiLU()
        self.mixchannel = nn.Conv2d(hidden, hidden, kernel_size=1)

        self.llr_head = nn.Sequential(
            nn.Conv2d(hidden + 2, hidden, kernel_size=3, padding=1),
            _make_group_norm(hidden) if use_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            _make_group_norm(hidden) if use_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(hidden, bits_per_symbol, kernel_size=1),
        )

    def forward(self, Y, H_hat, P, N0):
        Y = _ensure_complex_grid(Y)
        H_hat = _ensure_complex_grid(H_hat)

        b, t, f = Y.shape
        P, N0_grid = _prepare_zero_features(P, N0, b, t, f, Y.device)
        zero_features = torch.cat([P, N0_grid], dim=1)
        x = torch.cat(
            [
                Y.real.unsqueeze(1),
                Y.imag.unsqueeze(1),
                H_hat.real.unsqueeze(1),
                H_hat.imag.unsqueeze(1),
            ],
            dim=1,
        )

        x = self.input_norm(self.input_conv(x))
        x = self.input_gate(x, zero_features)
        x = self.input_activation(x)
        for block, zero_gate in zip(self.blocks, self.block_zero_gates):
            x = zero_gate(block(x), zero_features)

        x = self.readout_activation(self.readout_norm(self.readout(x)))
        x = self.mixchannel(x)
        return self.llr_head(torch.cat([x, P, N0_grid], dim=1))


class PhysicalFeatureCNN_OLD(nn.Module):
    """
    Baseline using hand-crafted phase-invariant physical features.

    Input channels:
        Re(conj(H_hat)*Y), Im(conj(H_hat)*Y), |H_hat|^2, |Y|^2, P, log(N0)

    This is an important baseline because it obtains phase invariance through
    feature construction rather than through network equivariance.
    """
    def __init__(self, hidden=32, bits_per_symbol=2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(6, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, bits_per_symbol, kernel_size=1),
        )

    def forward(self, Y, H_hat, P, N0):
        Y = _ensure_complex_grid(Y)
        H_hat = _ensure_complex_grid(H_hat)

        b, t, f = Y.shape
        P, N0_grid = _prepare_zero_features(P, N0, b, t, f, Y.device)

        matched = torch.conj(H_hat) * Y
        h_power = torch.abs(H_hat) ** 2
        y_power = torch.abs(Y) ** 2

        x = torch.cat(
            [
                matched.real.unsqueeze(1),
                matched.imag.unsqueeze(1),
                h_power.unsqueeze(1),
                y_power.unsqueeze(1),
                P,
                N0_grid,
            ],
            dim=1,
        )

        return self.net(x)




class PhysicalFeatureCNN(nn.Module):
    """
    Parameter/depth-matched physical feature CNN.

    It uses hand-crafted phase-invariant physical features:
        Re(conj(H_hat)*Y), Im(conj(H_hat)*Y), |H_hat|^2, |Y|^2

    Then it mirrors the macro-depth of PhaseInvariantReceiver:

        physical zero-order features
            ↓
        branch_layers real Conv blocks
            ↓
        1x1 compression to 2 * zero_complex channels
            ↓
        concat P and log(N0)
            ↓
        real-valued LLR head

    This is a stronger and fairer physical baseline than a shallow CNN.
    """

    def __init__(
        self,
        hidden=42,
        zero_complex=16,
        hidden_real=32,
        bits_per_symbol=2,
        branch_layers=2,
        kernel_size=3,
        use_norm=True,
    ):
        super().__init__()

        padding = kernel_size // 2

        layers = []

        # Input physical invariant channels:
        # Re(H*Y), Im(H*Y), |H|^2, |Y|^2
        in_channels = 4

        layers.append(
            nn.Conv2d(
                in_channels,
                hidden,
                kernel_size=kernel_size,
                padding=padding,
            )
        )
        if use_norm:
            layers.append(_make_group_norm(hidden))
        layers.append(nn.ReLU())

        for _ in range(branch_layers - 1):
            layers.append(
                nn.Conv2d(
                    hidden,
                    hidden,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            if use_norm:
                layers.append(_make_group_norm(hidden))
            layers.append(nn.ReLU())

        # Match the real dimension of zero_complex complex zero-order features.
        # PhaseInvariantReceiver has zero_complex complex channels,
        # then concatenates real and imag -> 2 * zero_complex real channels.
        layers.append(
            nn.Conv2d(
                hidden,
                2 * zero_complex,
                kernel_size=1,
            )
        )
        if use_norm:
            layers.append(_make_group_norm(2 * zero_complex))
            layers.append(nn.ReLU())

        self.feature_net = nn.Sequential(*layers)

        # Then concat P and log(N0), so input is 2 * zero_complex + 2.
        llr_in_channels = 2 * zero_complex + 2

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

        matched = torch.conj(H_hat) * Y
        h_power = torch.abs(H_hat) ** 2
        y_power = torch.abs(Y) ** 2

        physical_features = torch.cat(
            [
                matched.real.unsqueeze(1),
                matched.imag.unsqueeze(1),
                h_power.unsqueeze(1),
                y_power.unsqueeze(1),
            ],
            dim=1,
        )

        zero_feat = self.feature_net(physical_features)

        zero_cond = torch.cat(
            [
                zero_feat,
                P,
                N0_grid,
            ],
            dim=1,
        )

        return self.llr_head(zero_cond)
