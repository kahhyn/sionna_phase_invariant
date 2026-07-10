import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class OFDMDataset(Dataset):
    """
    Synthetic SISO-OFDM/QPSK dataset.

    Model:
        Y[k,l] = H[k,l] X[k,l] + N[k,l]

    H_hat modes:
        oracle_noisy:
            H_hat = H + E

        dmrs_ls_interp:
            H_hat is estimated from DMRS REs:
                H_LS[pilot] = Y[pilot] / X_pilot[pilot]
            Then non-DMRS symbols are filled by linear interpolation along
            the time dimension.

    Current DMRS pattern:
        Two DMRS OFDM symbols with configurable comb spacing in frequency.
        dmrs_freq_spacing=1 recovers the original full-DMRS-symbol setting.
    """
    def __init__(
        self,
        num_samples=10000,
        num_ofdm_symbols=14,
        num_subcarriers=72,
        bits_per_symbol=2,
        snr_db_min=0.0,
        snr_db_max=20.0,
        channel_error_std=0.05,
        channel_kind="tdl",
        h_hat_mode="oracle_noisy",
        phase_mode="fixed",
        narrow_phase_range=math.pi / 8,
        seed=0,
        dmrs_freq_spacing=1,
        dmrs_freq_offset=0,
        # TDL parameters
        subcarrier_spacing_hz=30e3,
        num_paths=12,
        rms_delay_spread_s=10e-9,
        max_doppler_hz=200.0,
        rician_k_db=None,
    ):
        if bits_per_symbol != 2:
            raise ValueError("This first version supports QPSK only, bits_per_symbol=2.")

        if h_hat_mode not in ["oracle_noisy", "dmrs_ls_interp"]:
            raise ValueError("h_hat_mode must be 'oracle_noisy' or 'dmrs_ls_interp'.")
        if dmrs_freq_spacing <= 0:
            raise ValueError("dmrs_freq_spacing must be positive.")
        if not (0 <= dmrs_freq_offset < num_subcarriers):
            raise ValueError("dmrs_freq_offset must be in [0, num_subcarriers).")

        self.num_samples = num_samples
        self.T = num_ofdm_symbols
        self.F = num_subcarriers
        self.bits_per_symbol = bits_per_symbol
        self.snr_db_min = float(snr_db_min)
        self.snr_db_max = float(snr_db_max)
        self.channel_error_std = float(channel_error_std)
        self.channel_kind = channel_kind
        self.h_hat_mode = h_hat_mode
        self.phase_mode = phase_mode
        self.narrow_phase_range = float(narrow_phase_range)
        self.seed = int(seed)
        self.dmrs_freq_spacing = int(dmrs_freq_spacing)
        self.dmrs_freq_offset = int(dmrs_freq_offset)

        # TDL parameters init
        self.subcarrier_spacing_hz = float(subcarrier_spacing_hz)
        self.num_paths = int(num_paths)
        self.rms_delay_spread_s = float(rms_delay_spread_s)
        self.max_doppler_hz = float(max_doppler_hz)
        self.rician_k_db = rician_k_db

        dmrs_symbols = [2, 11] if self.T >= 12 else [self.T // 2]
        dmrs_subcarriers = torch.arange(self.dmrs_freq_offset, self.F, self.dmrs_freq_spacing)
        if dmrs_subcarriers.numel() == 0:
            raise ValueError("DMRS pattern has no pilot subcarriers.")

        P = torch.zeros(1, self.T, self.F, dtype=torch.float32)
        for s in dmrs_symbols:
            P[:, s, dmrs_subcarriers] = 1.0

        self.dmrs_symbols = dmrs_symbols
        self.dmrs_subcarriers = dmrs_subcarriers
        self.P = P
        self.loss_mask = 1.0 - P

    def __len__(self):
        return self.num_samples

    def _generator(self, idx):
        g = torch.Generator()
        g.manual_seed(self.seed + int(idx))
        return g

    def _make_bits_and_symbols(self, g):
        bits = torch.randint(
            low=0,
            high=2,
            size=(self.bits_per_symbol, self.T, self.F),
            generator=g,
            dtype=torch.float32,
        )

        b0 = bits[0]
        b1 = bits[1]
        x = ((2.0 * b0 - 1.0) + 1j * (2.0 * b1 - 1.0)) / math.sqrt(2.0)
        x = x.to(torch.complex64)

        # Known DMRS pilot symbol. The corresponding bits are ignored by loss_mask.
        pilot = torch.ones_like(x)
        x = torch.where(self.P[0].bool(), pilot, x)

        bits = bits * self.loss_mask
        return bits, x

    def _smooth_complex_grid(self, z):
        zr = z.real.unsqueeze(0).unsqueeze(0)
        zi = z.imag.unsqueeze(0).unsqueeze(0)

        for kernel in [(3, 9), (3, 5)]:
            pad_t = kernel[0] // 2
            pad_f = kernel[1] // 2
            zr = F.avg_pool2d(zr, kernel_size=kernel, stride=1, padding=(pad_t, pad_f))
            zi = F.avg_pool2d(zi, kernel_size=kernel, stride=1, padding=(pad_t, pad_f))

        out = torch.complex(zr[0, 0], zi[0, 0])
        power = torch.mean(torch.abs(out) ** 2)
        return (out / torch.sqrt(power + 1e-12)).to(torch.complex64)

    def _make_channel(self, g):
        if self.channel_kind == "tdl":
            return self._make_tdl_channel(g)

        h = torch.randn(self.T, self.F, generator=g) + 1j * torch.randn(self.T, self.F, generator=g)
        h = h.to(torch.complex64) / math.sqrt(2.0)

        if self.channel_kind == "smooth":
            h = self._smooth_complex_grid(h)
        elif self.channel_kind == "iid":
            power = torch.mean(torch.abs(h) ** 2)
            h = h / torch.sqrt(power + 1e-12)
        else:
            raise ValueError(f"Unknown channel_kind: {self.channel_kind}")

        return h

    def _make_tdl_channel(self, g):
        """
        Physics-based SISO TDL channel.

        H[n, k] = sum_l alpha_l
                  * exp(j 2π f_D,l t_n)
                  * exp(-j 2π f_k tau_l)

        Shape:
            H: (T, F)
        """

        # Basic OFDM parameters.
        # For 5G NR, common SCS values are 15/30/60 kHz.
        subcarrier_spacing = getattr(self, "subcarrier_spacing_hz", 30e3)

        # Approximate useful OFDM symbol duration.
        # CP is ignored in this first channel-frequency-response version.
        ofdm_symbol_duration = 1.0 / subcarrier_spacing

        num_paths = getattr(self, "num_paths", 12)
        rms_delay_spread = getattr(self, "rms_delay_spread_s", 100e-9)
        max_doppler_hz = getattr(self, "max_doppler_hz", 100.0)
        rician_k_db = getattr(self, "rician_k_db", None)

        # Time index: OFDM symbols.
        t = torch.arange(self.T, dtype=torch.float32) * ofdm_symbol_duration

        # Frequency index: subcarriers around carrier center.
        # f_k is baseband offset from center frequency.
        k = torch.arange(self.F, dtype=torch.float32) - (self.F - 1) / 2.0
        f = k * subcarrier_spacing

        # Generate random path delays.
        # Exponential delay distribution is a simple physical PDP approximation.
        # Sort delays so path 0 is the earliest path.
        u = torch.rand(num_paths, generator=g).clamp_min(1e-6)
        tau = -rms_delay_spread * torch.log(u)
        tau, _ = torch.sort(tau)

        # Power delay profile: later paths usually have lower power.
        power = torch.exp(-tau / rms_delay_spread)
        power = power / power.sum()

        # Complex path gains.
        alpha = (
                        torch.randn(num_paths, generator=g)
                        + 1j * torch.randn(num_paths, generator=g)
                ).to(torch.complex64) / math.sqrt(2.0)
        alpha = alpha * torch.sqrt(power).to(torch.complex64)

        # Optional Rician LOS component on the first path.
        if rician_k_db is not None:
            k_linear = 10.0 ** (rician_k_db / 10.0)

            # NLOS part scaled by 1/(K+1)
            alpha = alpha / math.sqrt(k_linear + 1.0)

            # LOS component scaled by K/(K+1)
            los_phase = 2.0 * math.pi * torch.rand((), generator=g)
            los = torch.sqrt(torch.tensor(k_linear / (k_linear + 1.0))).to(torch.float32)
            alpha[0] = alpha[0] + los.to(torch.complex64) * (
                    torch.cos(los_phase) + 1j * torch.sin(los_phase)
            ).to(torch.complex64)

        # Path Doppler.
        # Jakes-like: f_D,l = f_D,max cos(theta_l)
        theta = 2.0 * math.pi * torch.rand(num_paths, generator=g)
        doppler = max_doppler_hz * torch.cos(theta)

        # Build time and frequency phase terms.
        # time_phase: (T, L)
        # freq_phase: (F, L)
        time_phase = torch.exp(
            1j * 2.0 * math.pi * t[:, None] * doppler[None, :]
        ).to(torch.complex64)

        freq_phase = torch.exp(
            -1j * 2.0 * math.pi * f[:, None] * tau[None, :]
        ).to(torch.complex64)

        # H[n,k] = sum_l alpha_l * time_phase[n,l] * freq_phase[k,l]
        h = torch.einsum("l,tl,fl->tf", alpha, time_phase, freq_phase)

        # Normalize average channel power to 1.
        h = h / torch.sqrt(torch.mean(torch.abs(h) ** 2) + 1e-12)

        return h.to(torch.complex64)

    def _sample_snr_and_noise(self, h, x, g):
        snr_db = self.snr_db_min + (self.snr_db_max - self.snr_db_min) * torch.rand((), generator=g)
        snr_linear = 10.0 ** (snr_db / 10.0)

        signal_power = torch.mean(torch.abs(h * x) ** 2)
        n0 = signal_power / snr_linear

        noise = torch.sqrt(n0 / 2.0) * (
            torch.randn(self.T, self.F, generator=g) + 1j * torch.randn(self.T, self.F, generator=g)
        )
        return n0.to(torch.float32), noise.to(torch.complex64)

    def _make_h_hat_oracle_noisy(self, h, g):
        e = self.channel_error_std / math.sqrt(2.0) * (
            torch.randn(self.T, self.F, generator=g) + 1j * torch.randn(self.T, self.F, generator=g)
        )
        return h + e.to(torch.complex64)

    def _interp_1d_complex(self, xp, yp, out_len):
        """
        Linear interpolation for complex tensors.

        xp has shape (N,), yp has shape (N, ...). The first dimension of yp is
        interpolated to length out_len, while trailing dimensions are preserved.
        Values outside the pilot range use nearest-pilot extrapolation.
        """
        xp = xp.to(device=yp.device, dtype=torch.float32)
        if xp.numel() != yp.shape[0]:
            raise ValueError("xp and yp must have the same first-dimension length.")
        if xp.numel() == 1:
            return yp[0].unsqueeze(0).expand((out_len,) + yp.shape[1:]).clone()

        x = torch.arange(out_len, device=yp.device, dtype=torch.float32)
        right = torch.searchsorted(xp.contiguous(), x).clamp(1, xp.numel() - 1)
        left = right - 1

        x0 = xp[left]
        x1 = xp[right]
        alpha = (x - x0) / (x1 - x0 + 1e-12)
        while alpha.dim() < yp.dim():
            alpha = alpha.unsqueeze(-1)

        out = (1.0 - alpha) * yp[left] + alpha * yp[right]
        out[x <= xp[0]] = yp[0]
        out[x >= xp[-1]] = yp[-1]
        return out

    def _make_h_hat_dmrs_ls_interp(self, y, x):
        """
        LS on DMRS REs, then linear interpolation over frequency and time.

        For comb pilots, each DMRS symbol is first interpolated over
        subcarriers. Those full-frequency estimates are then interpolated
        along the OFDM-symbol dimension.
        """
        p_mask = self.P[0].to(device=y.device).bool()
        h_ls = torch.zeros_like(y)
        h_ls[p_mask] = y[p_mask] / (x[p_mask] + 1e-12)

        pilot_ts = self.dmrs_symbols
        h_pilot_time_full_freq = torch.empty(
            len(pilot_ts),
            self.F,
            dtype=y.dtype,
            device=y.device,
        )

        for i, t in enumerate(pilot_ts):
            pilot_fs = torch.where(p_mask[t])[0]
            if pilot_fs.numel() == 0:
                raise ValueError(f"DMRS symbol {t} has no pilot subcarriers.")
            h_pilot_time_full_freq[i] = self._interp_1d_complex(
                pilot_fs,
                h_ls[t, pilot_fs],
                self.F,
            )

        pilot_ts_tensor = torch.tensor(pilot_ts, device=y.device, dtype=torch.long)
        h_hat = self._interp_1d_complex(pilot_ts_tensor, h_pilot_time_full_freq, self.T)

        return h_hat.to(torch.complex64)

    def _make_h_hat(self, h, y, x, g):
        if self.h_hat_mode == "oracle_noisy":
            return self._make_h_hat_oracle_noisy(h, g)
        if self.h_hat_mode == "dmrs_ls_interp":
            return self._make_h_hat_dmrs_ls_interp(y, x)
        raise ValueError(f"Unknown h_hat_mode: {self.h_hat_mode}")

    def _sample_phase(self, g):
        if self.phase_mode == "fixed":
            phi = torch.tensor(0.0)
        elif self.phase_mode == "narrow":
            r = torch.rand((), generator=g)
            phi = (2.0 * r - 1.0) * self.narrow_phase_range
        elif self.phase_mode == "uniform":
            phi = 2.0 * math.pi * torch.rand((), generator=g)
        else:
            raise ValueError(f"Unknown phase_mode: {self.phase_mode}")

        rot = torch.cos(phi) + 1j * torch.sin(phi)
        return rot.to(torch.complex64), phi.to(torch.float32)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        g = self._generator(idx)

        bits, x = self._make_bits_and_symbols(g)
        h = self._make_channel(g)
        n0, noise = self._sample_snr_and_noise(h, x, g)

        y = h * x + noise
        h_hat = self._make_h_hat(h, y, x, g)

        # Apply common phase rotation. If h_hat is obtained by DMRS-LS,
        # rotating h_hat after LS is equivalent to rotating Y before LS.
        rot, phi = self._sample_phase(g)
        y = rot * y
        h = rot * h
        h_hat = rot * h_hat

        return {
            "Y": y,
            "H_hat": h_hat,
            "P": self.P.clone(),
            "N0": n0.view(1),
            "bits": bits,
            "X": x,
            "H": h,
            "loss_mask": self.loss_mask.clone(),
            "phi": phi.view(1),
        }
