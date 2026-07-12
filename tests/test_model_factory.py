import unittest

import torch

from models.factory import build_model


class ModelFactoryTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
