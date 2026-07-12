"""Classical receiver baselines for the Sionna OFDM setup."""

from __future__ import annotations

import numpy as np
import torch
from sionna.phy.mapping import Constellation, Demapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import LMMSEEqualizer

from data.sionna_ofdm_generator import legacy_qpsk_points


class SionnaLMMSEBaseline:
    """LS/perfect-CSI LMMSE equalization followed by QPSK soft demapping."""

    def __init__(self, generator, *, csi: str = "ls") -> None:
        if csi not in {"ls", "perfect"}:
            raise ValueError("csi must be 'ls' or 'perfect'.")
        self.generator = generator
        self.csi = csi
        self.device = generator.device
        stream_management = StreamManagement(
            np.array([[1]], dtype=np.int32),
            num_streams_per_tx=1,
        )
        self.equalizer = LMMSEEqualizer(
            generator.resource_grid,
            stream_management,
            device=str(self.device),
        )
        constellation = Constellation(
            "custom",
            generator.config.bits_per_symbol,
            points=legacy_qpsk_points(self.device),
            device=str(self.device),
        )
        self.demapper = Demapper(
            "app",
            constellation=constellation,
            hard_out=False,
            device=str(self.device),
        )

    @property
    def model_name(self) -> str:
        return f"lmmse_{self.csi}"

    @torch.no_grad()
    def codeword_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return LDPC codeword logits in log p(bit=1)/p(bit=0) order."""
        y = batch["Y"].unsqueeze(1).unsqueeze(1)
        if self.csi == "perfect":
            h_hat = batch["H"]
            err_var = torch.zeros_like(batch["H"].real)
        else:
            h_hat = batch["H_hat"]
            err_var = batch["H_err_var"]

        h_hat = h_hat.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        err_var = err_var.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        no = batch["N0"].view(batch["N0"].shape[0], 1, 1)
        x_hat, no_eff = self.equalizer(y, h_hat, err_var, no)
        llr = self.demapper(x_hat, no_eff)
        return llr.reshape(batch["Y"].shape[0], self.generator.n)
