"""Minimal single-user MIMO-OFDM batch generation with Sionna.

This module deliberately keeps the first MIMO migration stage narrow:

* one user;
* one spatial stream per transmit antenna;
* frequency-division multiplexed (FDM) pilots;
* equal data-power allocation across layers; and
* a fixed total user transmit power on every active resource element.

The output contract exposes the layer axis explicitly and is independent from
the legacy SISO generator so that existing experiments remain unchanged.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from sionna.phy import config as sionna_config
from sionna.phy.channel import AWGN
from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.phy.mapping import Constellation, Mapper
from sionna.phy.ofdm import (
    LSChannelEstimator,
    PilotPattern,
    ResourceGrid,
    ResourceGridMapper,
)
from sionna.phy.utils import ebnodb2no

from .sionna_channel_backends import (
    SionnaChannelBackend,
    legacy_channel_profile,
    load_channel_profile,
)
from .sionna_ofdm_generator import SionnaLDPC5GConfig, legacy_qpsk_points


@dataclass(frozen=True)
class SionnaSUMIMOConfig:
    """Configuration for the first fixed-topology SU-MIMO migration stage."""

    num_ofdm_symbols: int = 14
    fft_size: int = 72
    subcarrier_spacing_hz: float = 30e3
    cyclic_prefix_length: int = 0
    bits_per_symbol: int = 2

    num_layers: int = 2
    num_rx_ant: int = 2
    total_tx_power: float = 1.0

    dmrs_symbol_indices: tuple[int, ...] = (2, 11)

    tdl_model: str = "A"
    delay_spread_s: float = 10e-9
    carrier_frequency_hz: float = 3.5e9
    max_doppler_hz: float = 200.0
    normalize_channel: bool = True
    ls_interpolation_type: str = "lin"

    def to_dict(self) -> dict:
        return asdict(self)


class SionnaSUMIMOBatchGenerator:
    """Generate fixed-topology SU-MIMO OFDM batches on a PyTorch device.

    Tensor contract:

    * ``Y``: ``[B, N_rx, T, F]``
    * ``H_hat``: ``[B, L, N_rx, T, F]``
    * ``P``: ``[B, L, 1, T, F]`` (non-zero FDM pilot positions)
    * ``bits``: ``[B, L, bits_per_symbol, T, F]``
    * ``loss_mask``: ``[B, L, 1, T, F]``

    Unit-energy QPSK data symbols are scaled by
    ``sqrt(total_tx_power / num_layers)``. On a DMRS resource element only
    one layer transmits, with amplitude ``sqrt(total_tx_power)``. Consequently
    the sum power over all layers equals ``total_tx_power`` on every RE.

    ``snr_db_min`` and ``snr_db_max`` denote total-user transmit ``Es/N0``
    before the channel. Noise is therefore set explicitly from
    ``N0 = total_tx_power / 10**(snr_db/10)`` rather than from the realized
    aggregate received power.
    """

    def __init__(
        self,
        config: SionnaSUMIMOConfig | None = None,
        *,
        snr_db_min: float = -5.0,
        snr_db_max: float = 20.0,
        phase_mode: str = "fixed",
        narrow_phase_range: float = math.pi / 8,
        seed: int = 0,
        device: torch.device | str = "cuda",
        channel_profile: str | dict[str, Any] | None = None,
    ) -> None:
        self.config = config or SionnaSUMIMOConfig()
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.snr_db_min = float(snr_db_min)
        self.snr_db_max = float(snr_db_max)
        self.phase_mode = phase_mode
        self.narrow_phase_range = float(narrow_phase_range)
        self.seed = int(seed)
        self.channel_profile = (
            legacy_channel_profile(self.config)
            if channel_profile is None
            else load_channel_profile(channel_profile)
        )

        self._validate_config()
        self._torch_generator = torch.Generator(device=self.device)
        self._build_sionna_chain()
        self.reset()

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.bits_per_symbol != 2:
            raise ValueError("The SU-MIMO smoke stage currently supports QPSK only.")
        if cfg.num_layers <= 0 or cfg.num_rx_ant <= 0:
            raise ValueError("num_layers and num_rx_ant must be positive.")
        if cfg.fft_size % cfg.num_layers != 0:
            raise ValueError("fft_size must be divisible by num_layers for FDM DMRS.")
        if cfg.total_tx_power <= 0.0:
            raise ValueError("total_tx_power must be positive.")
        if not cfg.dmrs_symbol_indices:
            raise ValueError("At least one DMRS OFDM symbol is required.")
        if self.phase_mode not in {"fixed", "narrow", "uniform"}:
            raise ValueError("phase_mode must be fixed, narrow, or uniform.")
        if self.snr_db_min > self.snr_db_max:
            raise ValueError("snr_db_min must not exceed snr_db_max.")

    def _build_sionna_chain(self) -> None:
        cfg = self.config
        num_dmrs_symbols = len(cfg.dmrs_symbol_indices)

        # Every stream reserves the union of DMRS REs, preventing another
        # layer's data from contaminating the per-layer LS estimate. The pilot
        # values are non-zero only on that layer's FDM comb.
        pilot_mask = torch.zeros(
            1,
            cfg.num_layers,
            cfg.num_ofdm_symbols,
            cfg.fft_size,
            dtype=torch.int32,
            device=self.device,
        )
        pilot_values = torch.zeros(
            1,
            cfg.num_layers,
            num_dmrs_symbols,
            cfg.fft_size,
            dtype=torch.complex64,
            device=self.device,
        )
        pilot_amplitude = math.sqrt(cfg.total_tx_power)
        for dmrs_offset, symbol_index in enumerate(cfg.dmrs_symbol_indices):
            if not 0 <= symbol_index < cfg.num_ofdm_symbols:
                raise ValueError(f"Invalid DMRS OFDM symbol index: {symbol_index}")
            pilot_mask[:, :, symbol_index, :] = 1
            for layer_index in range(cfg.num_layers):
                pilot_values[
                    0,
                    layer_index,
                    dmrs_offset,
                    layer_index :: cfg.num_layers,
                ] = pilot_amplitude

        pilot_pattern = PilotPattern(
            mask=pilot_mask,
            pilots=pilot_values.reshape(1, cfg.num_layers, -1),
            normalize=False,
            device=str(self.device),
        )
        self.resource_grid = ResourceGrid(
            num_ofdm_symbols=cfg.num_ofdm_symbols,
            fft_size=cfg.fft_size,
            subcarrier_spacing=cfg.subcarrier_spacing_hz,
            num_tx=1,
            num_streams_per_tx=cfg.num_layers,
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

        self.channel_backend = SionnaChannelBackend(
            self.channel_profile,
            cfg,
            self.resource_grid,
            self.device,
        )
        self.awgn = AWGN(device=str(self.device))
        self.ls_estimator = LSChannelEstimator(
            self.resource_grid,
            interpolation_type=cfg.ls_interpolation_type,
            device=str(self.device),
        )

        type_grid = self.resource_grid.build_type_grid()[0]
        self._data_mask = type_grid == 0
        layer_pilot_mask = torch.zeros(
            cfg.num_layers,
            cfg.num_ofdm_symbols,
            cfg.fft_size,
            dtype=torch.bool,
            device=self.device,
        )
        for layer_index in range(cfg.num_layers):
            for symbol_index in cfg.dmrs_symbol_indices:
                layer_pilot_mask[
                    layer_index,
                    symbol_index,
                    layer_index :: cfg.num_layers,
                ] = True
        self._layer_pilot_mask = layer_pilot_mask

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self._torch_generator.manual_seed(self.seed)
        sionna_config.seed = self.seed
        self._sionna_rng_state = sionna_config.torch_rng(
            str(self.device)
        ).get_state()
        self.channel_backend.reset(self.seed + 31_337)

    def reset_profile_sampler(self, seed: int | None = None) -> None:
        """Start a deterministic profile-component order and clear counts."""
        sampler_seed = self.seed + 31_337 if seed is None else int(seed)
        self.channel_backend.reset_sampler(sampler_seed)

    @property
    def channel_profile_counts(self) -> dict[str, int]:
        return dict(self.channel_backend.counts)

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
        bits_data = torch.randint(
            0,
            2,
            (
                batch_size,
                cfg.num_layers,
                self.resource_grid.num_data_symbols,
                cfg.bits_per_symbol,
            ),
            dtype=torch.int32,
            device=self.device,
            generator=self._torch_generator,
        )
        return bits_data, {}

    def _compute_noise_power(self, snr_db: torch.Tensor) -> torch.Tensor:
        return self.config.total_tx_power / torch.pow(10.0, snr_db / 10.0)

    @torch.no_grad()
    def generate_batch(
        self, batch_size: int, *, return_aux: bool = False
    ) -> Dict[str, Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        cfg = self.config
        bits_data, bit_metadata = self._make_data_bits(batch_size)
        mapped = self.mapper(
            bits_data.reshape(batch_size, 1, cfg.num_layers, -1)
        )
        mapped = mapped * math.sqrt(cfg.total_tx_power / cfg.num_layers)
        x_rg = self.grid_mapper(mapped)

        dense_bits = torch.zeros(
            batch_size,
            cfg.num_layers,
            cfg.bits_per_symbol,
            cfg.num_ofdm_symbols,
            cfg.fft_size,
            dtype=torch.float32,
            device=self.device,
        )
        for layer_index in range(cfg.num_layers):
            for bit_index in range(cfg.bits_per_symbol):
                dense_bits[
                    :, layer_index, bit_index, self._data_mask[layer_index]
                ] = bits_data[:, layer_index, :, bit_index].to(torch.float32)

        sionna_rng = self._activate_sionna_rng()
        y_clean_full, h_full, channel_metadata = self.channel_backend.apply(
            x_rg, batch_size
        )
        snr_db = self._sample_snr_db(batch_size)
        n0 = self._compute_noise_power(snr_db)
        y_full = self.awgn(y_clean_full, n0)
        h_hat_full, err_var_full = self.ls_estimator(y_full, n0)
        self._sionna_rng_state = sionna_rng.get_state()

        y_unrotated = y_full[:, 0]
        y_clean_unrotated = y_clean_full[:, 0]
        h_unrotated = h_full[:, 0, :, 0].permute(0, 2, 1, 3, 4)
        h_hat_unrotated = h_hat_full[:, 0, :, 0].permute(0, 2, 1, 3, 4)
        err_var = err_var_full[:, 0, :, 0].permute(0, 2, 1, 3, 4)
        x = x_rg[:, 0]

        phi = self._sample_phase(batch_size)
        rot_y = torch.polar(torch.ones_like(phi), phi).view(batch_size, 1, 1, 1)
        rot_h = rot_y.unsqueeze(1)
        y = rot_y * y_unrotated
        h = rot_h * h_unrotated
        h_hat = rot_h * h_hat_unrotated

        p = self._layer_pilot_mask.to(torch.float32)
        p = p.view(1, cfg.num_layers, 1, cfg.num_ofdm_symbols, cfg.fft_size)
        p = p.expand(batch_size, -1, -1, -1, -1)
        loss_mask = self._data_mask.to(torch.float32).unsqueeze(1)
        loss_mask = loss_mask.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

        batch: Dict[str, Any] = {
            "Y": y.to(torch.complex64),
            "H_hat": h_hat.to(torch.complex64),
            "P": p,
            "N0": n0.view(batch_size, 1).to(torch.float32),
            "bits": dense_bits,
            "X": x.to(torch.complex64),
            "H": h.to(torch.complex64),
            "loss_mask": loss_mask,
            "layer_mask": torch.ones(
                batch_size,
                cfg.num_layers,
                dtype=torch.bool,
                device=self.device,
            ),
            "phi": phi.view(batch_size, 1).to(torch.float32),
            "H_err_var": err_var.to(torch.float32),
            "snr_db": snr_db.view(batch_size, 1).to(torch.float32),
            "total_tx_power": torch.full(
                (batch_size, 1),
                cfg.total_tx_power,
                dtype=torch.float32,
                device=self.device,
            ),
            "power_per_data_layer": torch.full(
                (batch_size, 1),
                cfg.total_tx_power / cfg.num_layers,
                dtype=torch.float32,
                device=self.device,
            ),
        }
        batch.update(bit_metadata)
        batch.update(channel_metadata)
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


class Sionna5GLDPCSUMIMOBatchGenerator(SionnaSUMIMOBatchGenerator):
    """One independent rate-matched 5G NR LDPC codeword per MIMO layer."""

    def __init__(
        self,
        config: SionnaSUMIMOConfig | None = None,
        *,
        ldpc_config: SionnaLDPC5GConfig | None = None,
        ebno_db_min: float = -3.0,
        ebno_db_max: float = 5.0,
        phase_mode: str = "fixed",
        narrow_phase_range: float = math.pi / 8,
        seed: int = 0,
        device: torch.device | str = "cuda",
        channel_profile: str | dict[str, Any] | None = None,
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
            channel_profile=channel_profile,
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
        num_layers = self.config.num_layers
        info_bits = torch.randint(
            0,
            2,
            (batch_size, num_layers, self.k),
            dtype=torch.int32,
            device=self.device,
            generator=self._torch_generator,
        ).to(torch.float32)
        codeword_bits = self.encoder(
            info_bits.reshape(batch_size * num_layers, self.k)
        ).reshape(batch_size, num_layers, self.n)
        bits_data = codeword_bits.reshape(
            batch_size,
            num_layers,
            self.resource_grid.num_data_symbols,
            self.config.bits_per_symbol,
        ).to(torch.int32)
        return bits_data, {
            "info_bits": info_bits,
            "codeword_bits": codeword_bits.to(torch.float32),
        }

    def _compute_noise_power(self, ebno_db: torch.Tensor) -> torch.Tensor:
        nominal_n0 = ebnodb2no(
            ebno_db,
            num_bits_per_symbol=self.config.bits_per_symbol,
            coderate=self.coderate,
            resource_grid=self.resource_grid,
        ).to(device=self.device, dtype=torch.float32)
        return self.config.total_tx_power * nominal_n0

    @torch.no_grad()
    def generate_batch(
        self, batch_size: int, *, return_aux: bool = False
    ) -> Dict[str, Any]:
        batch = super().generate_batch(batch_size, return_aux=return_aux)
        batch["ebno_db"] = batch["snr_db"].clone()
        return batch

    def extract_codeword_logits(self, logits: torch.Tensor) -> torch.Tensor:
        expected_shape = (
            self.config.num_layers,
            self.config.bits_per_symbol,
            self.config.num_ofdm_symbols,
            self.config.fft_size,
        )
        if logits.dim() != 5 or tuple(logits.shape[1:]) != expected_shape:
            raise ValueError("logits must have shape [B, L, bits, T, F].")
        layer_logits = []
        for layer_index in range(self.config.num_layers):
            bit_planes = [
                logits[
                    :, layer_index, bit_index, self._data_mask[layer_index]
                ]
                for bit_index in range(self.config.bits_per_symbol)
            ]
            layer_logits.append(
                torch.stack(bit_planes, dim=-1).reshape(logits.shape[0], self.n)
            )
        return torch.stack(layer_logits, dim=1)

    @torch.no_grad()
    def decode_logits(
        self, logits: torch.Tensor, *, num_iter: int | None = None
    ) -> torch.Tensor:
        codeword_logits = self.extract_codeword_logits(logits)
        flat_logits = codeword_logits.reshape(-1, self.n)
        if num_iter is None:
            decoded = self.decoder(flat_logits)
        else:
            decoded = self.decoder(flat_logits, num_iter=num_iter)
        return decoded.reshape(logits.shape[0], self.config.num_layers, self.k)
