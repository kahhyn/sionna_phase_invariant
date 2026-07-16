import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from evaluation.aggregate_quadriga_interference import aggregate
from evaluation.eval_quadriga_interference import (
    build_interference_mask,
    channel_power,
    compute_interference_scale,
    load_two_link_trajectory,
    make_interference_batch,
)


class ZeroLSEstimator:
    def __call__(self, y_full, n0):
        batch, _, _, symbols, carriers = y_full.shape
        h_hat = torch.zeros(
            batch,
            1,
            1,
            1,
            1,
            symbols,
            carriers,
            dtype=torch.complex64,
            device=y_full.device,
        )
        return h_hat, torch.zeros_like(h_hat.real)


class QuaDRiGaInterferenceTest(unittest.TestCase):
    def setUp(self):
        self.pilot_mask = torch.zeros(14, 72, dtype=torch.bool)
        self.pilot_mask[[2, 11], :] = True

    def test_load_two_link_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two_link.mat"
            shape = (5, 14, 72)
            desired = np.ones(shape, dtype=np.complex64) * (1 + 2j)
            interferer = np.ones(shape, dtype=np.complex64) * (3 - 1j)
            savemat(
                path,
                {
                    "H_desired_real": desired.real,
                    "H_desired_imag": desired.imag,
                    "H_interferer_real": interferer.real,
                    "H_interferer_imag": interferer.imag,
                    "desired_positions_m": np.zeros((5, 3), dtype=np.float32),
                    "interferer_positions_m": np.ones((5, 3), dtype=np.float32),
                    "timestamps_s": np.arange(5, dtype=np.float32)[:, None],
                    "frame_index": np.arange(5, dtype=np.int32)[:, None],
                },
            )
            loaded = load_two_link_trajectory(path, max_frames=3)
            self.assertEqual(tuple(loaded.desired_h.shape), (3, 14, 72))
            self.assertEqual(tuple(loaded.interferer_h.shape), (3, 14, 72))
            torch.testing.assert_close(
                loaded.desired_h[0, 0, 0], torch.tensor(1 + 2j)
            )
            torch.testing.assert_close(
                loaded.interferer_h[0, 0, 0], torch.tensor(3 - 1j)
            )

    def test_interference_masks(self):
        full = build_interference_mask(
            "cochannel_full", self.pilot_mask, partial_band_fraction=0.25
        )
        data_only = build_interference_mask(
            "data_only", self.pilot_mask, partial_band_fraction=0.25
        )
        partial = build_interference_mask(
            "partial_band", self.pilot_mask, partial_band_fraction=0.25
        )
        self.assertTrue(full.all())
        self.assertFalse(data_only[self.pilot_mask].any())
        self.assertTrue(data_only[~self.pilot_mask].all())
        self.assertEqual(int(partial.sum()), 14 * 18)

    def test_requested_sir_is_realized(self):
        desired = torch.ones(4, 14, 72, dtype=torch.complex64)
        interferer = torch.ones_like(desired) * (2 + 0j)
        active = ~self.pilot_mask
        for normalization in ("per_frame", "global"):
            scale = compute_interference_scale(
                desired, interferer, active, 10.0, normalization
            )
            desired_power = channel_power(desired).sum()
            interference_power = channel_power(
                interferer * scale,
                active,
            ).sum()
            achieved = 10.0 * torch.log10(desired_power / interference_power)
            self.assertAlmostEqual(float(achieved), 10.0, places=5)

    def test_infinite_sir_disables_interference(self):
        desired = torch.ones(2, 14, 72, dtype=torch.complex64)
        interferer = torch.ones_like(desired)
        scale = compute_interference_scale(
            desired, interferer, ~self.pilot_mask, math.inf, "per_frame"
        )
        self.assertTrue(torch.equal(scale, torch.zeros_like(scale)))

    def test_batch_is_reproducible_and_n0_is_thermal(self):
        desired = torch.ones(2, 14, 72, dtype=torch.complex64)
        interferer = torch.ones_like(desired)
        active = build_interference_mask(
            "cochannel_full", self.pilot_mask, partial_band_fraction=0.25
        )
        scale = compute_interference_scale(
            desired, interferer, active, 5.0, "per_frame"
        )

        batches = []
        for _ in range(2):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(123)
            interference_generator = torch.Generator(device="cpu")
            interference_generator.manual_seed(456)
            batches.append(
                make_interference_batch(
                    desired,
                    interferer,
                    scale,
                    10.0,
                    self.pilot_mask,
                    active,
                    ZeroLSEstimator(),
                    "fixed",
                    math.pi / 8,
                    "thermal",
                    generator,
                    interference_generator,
                )
            )
        for key in ("Y", "bits", "N0", "interference_power", "thermal_n0"):
            torch.testing.assert_close(batches[0][key], batches[1][key])
        torch.testing.assert_close(
            batches[0]["thermal_n0"], batches[0]["desired_power"] / 10.0
        )

    def test_disabled_interference_does_not_change_baseline_rng_stream(self):
        desired = torch.ones(2, 14, 72, dtype=torch.complex64)
        interferer = torch.ones_like(desired)
        active = build_interference_mask(
            "cochannel_full", self.pilot_mask, partial_band_fraction=0.25
        )
        scale = torch.zeros(2, 1, 1)

        batches = []
        for interference_seed in (456, 789):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(123)
            interference_generator = torch.Generator(device="cpu")
            interference_generator.manual_seed(interference_seed)
            batches.append(
                make_interference_batch(
                    desired,
                    interferer,
                    scale,
                    10.0,
                    self.pilot_mask,
                    active,
                    ZeroLSEstimator(),
                    "fixed",
                    math.pi / 8,
                    "thermal",
                    generator,
                    interference_generator,
                )
            )

        for key in ("Y", "bits", "N0", "thermal_n0"):
            torch.testing.assert_close(batches[0][key], batches[1][key])

    def test_aggregator_preserves_ac_pairs(self):
        fields = [
            "train_profile",
            "receiver_label",
            "model",
            "snr_db",
            "sir_db",
            "interference_mode",
            "sir_normalization",
            "n0_mode",
            "phase_mode",
            "channel_normalization",
            "train_seed",
            "eval_seed",
            "ber",
            "bce",
            "bit_errors",
            "valid_bits",
            "h_hat_nmse_db",
            "achieved_sir_db",
            "achieved_sinr_db",
        ]
        base = {
            "train_profile": "tdl_mix_normalized",
            "snr_db": "10.0",
            "sir_db": "5.0",
            "interference_mode": "cochannel_full",
            "sir_normalization": "per_frame",
            "n0_mode": "thermal",
            "phase_mode": "fixed",
            "channel_normalization": "per_frame",
            "train_seed": "0",
            "eval_seed": "777000",
            "bce": "0.2",
            "valid_bits": "1000",
            "h_hat_nmse_db": "-3.0",
            "achieved_sir_db": "5.0",
            "achieved_sinr_db": "3.8",
        }
        rows = [
            {
                **base,
                "receiver_label": "A_phase_invariant",
                "model": "single_branch_n0_gate",
                "ber": "0.1",
                "bit_errors": "100",
            },
            {
                **base,
                "receiver_label": "C_strict_matched",
                "model": "strict_matched_complex_p_n0_gate",
                "ber": "0.12",
                "bit_errors": "120",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run" / "quadriga_interference_summary.csv"
            path.parent.mkdir()
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            aggregate_rows, paired_rows = aggregate(Path(directory))
            self.assertEqual(len(aggregate_rows), 2)
            self.assertEqual(len(paired_rows), 1)
            self.assertAlmostEqual(
                float(paired_rows[0]["a_minus_c_ber_mean"]), -0.02
            )


if __name__ == "__main__":
    unittest.main()
