"""Batched Sionna data generation for the phase-invariant receiver.

The public output contract intentionally matches :class:`OFDMDataset` so that
the existing PyTorch receivers can be used without architectural changes.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict

import torch
from sionna.phy import config as sionna_config
from sionna.phy.channel import AWGN, OFDMChannel
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.phy.mapping import Constellation, Mapper
from sionna.phy.ofdm import (
    LSChannelEstimator,
    PilotPattern,
    ResourceGrid,
    ResourceGridMapper,
)
from sionna.phy.utils import ebnodb2no


@dataclass(frozen=True)
class SionnaOFDMConfig:
    num_ofdm_symbols: int = 14
    fft_size: int = 72
    subcarrier_spacing_hz: float = 30e3
    cyclic_prefix_length: int = 0
    bits_per_symbol: int = 2

    dmrs_symbol_indices: tuple[int, ...] = (2, 11)
    dmrs_freq_spacing: int = 1
    dmrs_freq_offset: int = 0

    tdl_model: str = "A"
    delay_spread_s: float = 10e-9
    carrier_frequency_hz: float = 3.5e9
    max_doppler_hz: float = 200.0
    normalize_channel: bool = True
    ls_interpolation_type: str = "lin"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SionnaLDPC5GConfig:
    coderate: float = 0.5
    num_iter: int = 20
    cn_update: str = "boxplus-phi"
    prune_pcm: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def legacy_qpsk_points(device: torch.device | str) -> torch.Tensor:
    """Return the exact bit labeling used by the legacy dataset.

    Index order is the binary label ``00, 01, 10, 11``.
    """
    scale = 1.0 / math.sqrt(2.0)
    return scale * torch.tensor(
        [-1.0 - 1.0j, -1.0 + 1.0j, 1.0 - 1.0j, 1.0 + 1.0j],
        dtype=torch.complex64,
        device=device,
    )


class SionnaOFDMBatchGenerator:
    """Generate complete SISO-OFDM batches directly on a PyTorch device."""

    def __init__(
        self,
        config: SionnaOFDMConfig | None = None,
        *,
        snr_db_min: float = -5.0,
        snr_db_max: float = 20.0,
        phase_mode: str = "fixed",
        narrow_phase_range: float = math.pi / 8,
        seed: int = 0,
        device: torch.device | str = "cuda",
    ) -> None:
        self.config = config or SionnaOFDMConfig()
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.snr_db_min = float(snr_db_min)
        self.snr_db_max = float(snr_db_max)
        self.phase_mode = phase_mode
        self.narrow_phase_range = float(narrow_phase_range)
        self.seed = int(seed)

        if self.config.bits_per_symbol != 2:
            raise ValueError("The migrated receiver currently supports QPSK only.")
        if self.phase_mode not in {"fixed", "narrow", "uniform"}:
            raise ValueError("phase_mode must be fixed, narrow, or uniform.")
        if self.snr_db_min > self.snr_db_max:
            raise ValueError("snr_db_min must not exceed snr_db_max.")
        if self.config.dmrs_freq_spacing <= 0:
            raise ValueError("dmrs_freq_spacing must be positive.")

        self._torch_generator = torch.Generator(device=self.device)
        self._build_sionna_chain()
        self.reset()

    def _build_sionna_chain(self) -> None:
        cfg = self.config
        pilot_mask = torch.zeros(
            1,
            1,
            cfg.num_ofdm_symbols,
            cfg.fft_size,
            dtype=torch.int32,
            device=self.device,
        )
        pilot_subcarriers = torch.arange(
            cfg.dmrs_freq_offset,
            cfg.fft_size,
            cfg.dmrs_freq_spacing,
            device=self.device,
        )
        if pilot_subcarriers.numel() == 0:
            raise ValueError("The configured DMRS pattern contains no pilots.")
        for symbol_index in cfg.dmrs_symbol_indices:
            if not 0 <= symbol_index < cfg.num_ofdm_symbols:
                raise ValueError(f"Invalid DMRS OFDM symbol index: {symbol_index}")
            pilot_mask[0, 0, symbol_index, pilot_subcarriers] = 1

        num_pilots = int(pilot_mask.sum().item())
        pilots = torch.ones(
            1, 1, num_pilots, dtype=torch.complex64, device=self.device
        )
        pilot_pattern = PilotPattern(
            mask=pilot_mask,
            pilots=pilots,
            normalize=False,
            device=str(self.device),
        )

        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=cfg.num_ofdm_symbols,
            fft_size=cfg.fft_size,
            subcarrier_spacing=cfg.subcarrier_spacing_hz,
            num_tx=1,
            num_streams_per_tx=1,
            cyclic_prefix_length=cfg.cyclic_prefix_length,
            num_guard_carriers=(0, 0),
            dc_null=False,
            pilot_pattern=pilot_pattern,
            device=str(self.device),
        )

        constellation = Constellation(
            "custom",
            cfg.bits_per_symbol,
            points=legacy_qpsk_points(self.device),
            device=str(self.device),
        )
        self.mapper = Mapper(constellation=constellation, device=str(self.device))
        self.grid_mapper = ResourceGridMapper(
            self.resource_grid, device=str(self.device)
        )

        max_speed_mps = (
            cfg.max_doppler_hz * 299_792_458.0 / cfg.carrier_frequency_hz
        )
        self.tdl = TDL(
            model=cfg.tdl_model,
            delay_spread=cfg.delay_spread_s,
            carrier_frequency=cfg.carrier_frequency_hz,
            min_speed=0.0,
            max_speed=max_speed_mps,
            num_rx_ant=1,
            num_tx_ant=1,
            device=str(self.device),
        )
        self.ofdm_channel = OFDMChannel(
            channel_model=self.tdl,
            resource_grid=self.resource_grid,
            normalize_channel=cfg.normalize_channel,
            return_channel=True,
            device=str(self.device),
        )
        self.awgn = AWGN(device=str(self.device))
        self.ls_estimator = LSChannelEstimator(
            self.resource_grid,
            interpolation_type=cfg.ls_interpolation_type,
            device=str(self.device),
        )

        type_grid = self.resource_grid.build_type_grid()[0, 0]
        self._pilot_mask = (type_grid == 1).to(torch.float32).unsqueeze(0)
        self._loss_mask = 1.0 - self._pilot_mask
        self._data_mask = type_grid == 0

    def reset(self, seed: int | None = None) -> None:
        """Reset both local sampling and Sionna channel/noise random streams."""
        if seed is not None:
            self.seed = int(seed)
        self._torch_generator.manual_seed(self.seed)
        sionna_config.seed = self.seed
        self._sionna_rng_state = sionna_config.torch_rng(
            str(self.device)
        ).get_state()

    def _activate_sionna_rng(self) -> torch.Generator:
        rng = sionna_config.torch_rng(str(self.device))
        rng.set_state(self._sionna_rng_state)
        return rng

    def _sample_snr_db(self, batch_size: int) -> torch.Tensor:
        unit = torch.rand(
            batch_size, device=self.device, generator=self._torch_generator
        )
        return self.snr_db_min + (self.snr_db_max - self.snr_db_min) * unit

    def _sample_phase(self, batch_size: int) -> torch.Tensor:
        if self.phase_mode == "fixed":
            return torch.zeros(batch_size, device=self.device)
        unit = torch.rand(
            batch_size, device=self.device, generator=self._torch_generator
        )
        if self.phase_mode == "narrow":
            return (2.0 * unit - 1.0) * self.narrow_phase_range
        return 2.0 * math.pi * unit

    def _make_data_bits(self, batch_size: int):
        cfg = self.config
        num_data_symbols = self.resource_grid.num_data_symbols
        bits_data = torch.randint(
            0,
            2,
            (batch_size, num_data_symbols, cfg.bits_per_symbol),
            dtype=torch.int32,
            device=self.device,
            generator=self._torch_generator,
        )
        return bits_data, {}

    def _compute_noise_power(
        self, y_clean_full: torch.Tensor, snr_db: torch.Tensor
    ) -> torch.Tensor:
        signal_power = y_clean_full.abs().square().mean(dim=(1, 2, 3, 4))
        return signal_power / torch.pow(10.0, snr_db / 10.0)

    @torch.no_grad()
    def generate_batch(
        self, batch_size: int, *, return_aux: bool = False
    ) -> Dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        cfg = self.config
        bits_data, bit_metadata = self._make_data_bits(batch_size)
        mapped = self.mapper(bits_data.reshape(batch_size, 1, 1, -1))
        x_rg = self.grid_mapper(mapped)

        dense_bits = torch.zeros(
            batch_size,
            cfg.bits_per_symbol,
            cfg.num_ofdm_symbols,
            cfg.fft_size,
            dtype=torch.float32,
            device=self.device,
        )
        for bit_index in range(cfg.bits_per_symbol):
            dense_bits[:, bit_index, self._data_mask] = bits_data[
                :, :, bit_index
            ].to(torch.float32)

        sionna_rng = self._activate_sionna_rng()
        y_clean_full, h_full = self.ofdm_channel(x_rg)

        snr_db = self._sample_snr_db(batch_size)
        n0 = self._compute_noise_power(y_clean_full, snr_db)
        y_full = self.awgn(y_clean_full, n0)
        h_hat_full, err_var_full = self.ls_estimator(y_full, n0)
        self._sionna_rng_state = sionna_rng.get_state()

        y_unrotated = y_full[:, 0, 0]
        y_clean_unrotated = y_clean_full[:, 0, 0]
        h_unrotated = h_full[:, 0, 0, 0, 0]
        h_hat_unrotated = h_hat_full[:, 0, 0, 0, 0]
        err_var = err_var_full[:, 0, 0, 0, 0]
        x = x_rg[:, 0, 0]

        phi = self._sample_phase(batch_size)
        rot = torch.polar(torch.ones_like(phi), phi).view(batch_size, 1, 1)
        y = rot * y_unrotated
        h = rot * h_unrotated
        h_hat = rot * h_hat_unrotated

        p = self._pilot_mask.expand(batch_size, -1, -1, -1)
        loss_mask = self._loss_mask.expand(batch_size, -1, -1, -1)

        batch = {
            "Y": y.to(torch.complex64),
            "H_hat": h_hat.to(torch.complex64),
            "P": p,
            "N0": n0.view(batch_size, 1).to(torch.float32),
            "bits": dense_bits,
            "X": x.to(torch.complex64),
            "H": h.to(torch.complex64),
            "loss_mask": loss_mask,
            "phi": phi.view(batch_size, 1).to(torch.float32),
            "H_err_var": err_var.to(torch.float32),
            "snr_db": snr_db.view(batch_size, 1).to(torch.float32),
        }
        batch.update(bit_metadata)
        if return_aux:
            batch.update(
                {
                    "Y_unrotated": y_unrotated.to(torch.complex64),
                    "Y_clean_unrotated": y_clean_unrotated.to(torch.complex64),
                    "H_unrotated": h_unrotated.to(torch.complex64),
                    "H_hat_unrotated": h_hat_unrotated.to(torch.complex64),
                }
            )
        return batch


class Sionna5GLDPCBatchGenerator(SionnaOFDMBatchGenerator):
    """Sionna OFDM generator with 5G NR LDPC encoding and BP decoding.

    One rate-matched LDPC codeword fills all data REs of one OFDM frame. The
    sampled dB value is interpreted as Eb/N0 and converted to noise variance
    with :func:`sionna.phy.utils.ebnodb2no`.
    """

    def __init__(
        self,
        config: SionnaOFDMConfig | None = None,
        *,
        ldpc_config: SionnaLDPC5GConfig | None = None,
        ebno_db_min: float = -3.0,
        ebno_db_max: float = 5.0,
        phase_mode: str = "fixed",
        narrow_phase_range: float = math.pi / 8,
        seed: int = 0,
        device: torch.device | str = "cuda",
    ) -> None:
        self.ldpc_config = ldpc_config or SionnaLDPC5GConfig()
        super().__init__(
            config,
            snr_db_min=ebno_db_min,
            snr_db_max=ebno_db_max,
            phase_mode=phase_mode,
            narrow_phase_range=narrow_phase_range,
            seed=seed,
            device=device,
        )

        self.n = int(
            self.resource_grid.num_data_symbols * self.config.bits_per_symbol
        )
        self.k = int(round(self.n * self.ldpc_config.coderate))
        if not 0.0 < self.ldpc_config.coderate < 1.0:
            raise ValueError("LDPC coderate must be in (0, 1).")
        self.encoder = LDPC5GEncoder(
            k=self.k,
            n=self.n,
            num_bits_per_symbol=self.config.bits_per_symbol,
            device=str(self.device),
        )
        self.decoder = LDPC5GDecoder(
            self.encoder,
            hard_out=True,
            return_infobits=True,
            num_iter=self.ldpc_config.num_iter,
            cn_update=self.ldpc_config.cn_update,
            prune_pcm=self.ldpc_config.prune_pcm,
            device=str(self.device),
        )

    @property
    def coderate(self) -> float:
        return self.k / self.n

    def _make_data_bits(self, batch_size: int):
        info_bits = torch.randint(
            0,
            2,
            (batch_size, self.k),
            dtype=torch.int32,
            device=self.device,
            generator=self._torch_generator,
        ).to(torch.float32)
        codeword_bits = self.encoder(info_bits)
        bits_data = codeword_bits.reshape(
            batch_size,
            self.resource_grid.num_data_symbols,
            self.config.bits_per_symbol,
        ).to(torch.int32)
        return bits_data, {
            "info_bits": info_bits,
            "codeword_bits": codeword_bits.to(torch.float32),
        }

    def _compute_noise_power(
        self, y_clean_full: torch.Tensor, ebno_db: torch.Tensor
    ) -> torch.Tensor:
        nominal_n0 = ebnodb2no(
            ebno_db,
            num_bits_per_symbol=self.config.bits_per_symbol,
            coderate=self.coderate,
            resource_grid=self.resource_grid,
        ).to(device=self.device, dtype=torch.float32)
        signal_power = y_clean_full.abs().square().mean(dim=(1, 2, 3, 4))
        return signal_power * nominal_n0

    @torch.no_grad()
    def generate_batch(
        self, batch_size: int, *, return_aux: bool = False
    ) -> Dict[str, torch.Tensor]:
        batch = super().generate_batch(batch_size, return_aux=return_aux)
        batch["ebno_db"] = batch["snr_db"].clone()
        return batch

    def extract_codeword_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Extract data-RE logits in the exact LDPC rate-matched bit order."""
        expected_shape = (
            self.config.bits_per_symbol,
            self.config.num_ofdm_symbols,
            self.config.fft_size,
        )
        if logits.dim() != 4 or tuple(logits.shape[1:]) != expected_shape:
            raise ValueError(
                "logits must have shape [B, bits_per_symbol, T, F]."
            )
        bit_planes = [
            logits[:, bit_index, self._data_mask]
            for bit_index in range(self.config.bits_per_symbol)
        ]
        return torch.stack(bit_planes, dim=-1).reshape(logits.shape[0], self.n)

    @torch.no_grad()
    def decode_logits(
        self, logits: torch.Tensor, *, num_iter: int | None = None
    ) -> torch.Tensor:
        """Decode network logits, which already use log p(1)/p(0)."""
        codeword_logits = self.extract_codeword_logits(logits)
        if num_iter is None:
            return self.decoder(codeword_logits)
        return self.decoder(codeword_logits, num_iter=num_iter)
