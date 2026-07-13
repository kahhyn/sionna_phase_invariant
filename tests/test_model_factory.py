import unittest

import torch

from models.factory import build_model
from models.phase_equivariant_denoiser import EquivariantHResidualDenoiser


class ModelFactoryTest(unittest.TestCase):
    @staticmethod
    def _inputs():
        batch_size = 2
        num_symbols = 4
        num_subcarriers = 5
        y = torch.randn(
            batch_size, num_symbols, num_subcarriers, dtype=torch.complex64
        )
        h_hat = torch.randn(
            batch_size, num_symbols, num_subcarriers, dtype=torch.complex64
        )
        p = torch.zeros(batch_size, 1, num_symbols, num_subcarriers)
        p[:, :, 1, :] = 1.0
        n0 = torch.full((batch_size, 1), 0.1)
        return y, h_hat, p, n0

    def test_complex_zero_condition_models_forward(self):
        model_names = [
            "complex_p",
            "complex_n0",
            "complex_p_n0",
            "complex_p_n0_gate",
            "complex_p_n0_film",
        ]
        batch_size = 2
        num_symbols = 4
        num_subcarriers = 5
        y = torch.randn(batch_size, num_symbols, num_subcarriers) + 1j * torch.randn(
            batch_size,
            num_symbols,
            num_subcarriers,
        )
        h_hat = torch.randn(
            batch_size,
            num_symbols,
            num_subcarriers,
        ) + 1j * torch.randn(batch_size, num_symbols, num_subcarriers)
        p = torch.zeros(batch_size, 1, num_symbols, num_subcarriers)
        p[:, :, 1, :] = 1.0
        n0 = torch.full((batch_size, 1), 0.1)

        for name in model_names:
            with self.subTest(model=name):
                model = build_model(
                    name,
                    bits_per_symbol=2,
                    hidden=8,
                    hidden_complex=6,
                    branch_layers=2,
                    zero_gate_hidden=4,
                )
                logits = model(y.to(torch.complex64), h_hat.to(torch.complex64), p, n0)
                self.assertEqual(
                    logits.shape,
                    (batch_size, 2, num_symbols, num_subcarriers),
                )
                self.assertTrue(torch.isfinite(logits).all())

    def test_h_denoiser_starts_as_equivariant_identity(self):
        _, h_hat, p, n0 = self._inputs()
        denoiser = EquivariantHResidualDenoiser(
            hidden_complex=4,
            num_blocks=1,
            condition_hidden=3,
        )
        refined = denoiser(h_hat, p, n0)
        torch.testing.assert_close(refined, h_hat)

        phase = torch.exp(1j * torch.tensor(0.71))
        rotated = denoiser(phase * h_hat, p, n0)
        torch.testing.assert_close(rotated, phase * refined)

    def test_denoised_a_c_models_are_parameter_matched(self):
        y, h_hat, p, n0 = self._inputs()
        kwargs = {
            "bits_per_symbol": 2,
            "hidden": 8,
            "hidden_complex": 6,
            "zero_complex": 6,
            "branch_layers": 1,
            "zero_gate_hidden": 4,
            "denoiser_hidden": 4,
            "denoiser_blocks": 1,
        }
        invariant = build_model(
            "single_branch_n0_gate_h_denoise", **kwargs
        )
        matched = build_model(
            "strict_matched_complex_p_n0_gate_h_denoise", **kwargs
        )
        invariant_count = sum(p.numel() for p in invariant.parameters())
        matched_count = sum(p.numel() for p in matched.parameters())
        self.assertEqual(invariant_count, matched_count)

        for model in (invariant, matched):
            logits, aux = model.forward_with_aux(y, h_hat, p, n0)
            self.assertEqual(logits.shape, (2, 2, 4, 5))
            self.assertEqual(aux["H_refined"].shape, h_hat.shape)
            self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
