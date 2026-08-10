import unittest

import torch

from models import (
    SU_MIMO_MODEL_CHOICES,
    SUMIMOWidelyLinearReceiver,
    build_su_mimo_model,
)
from models.complex_layers import WidelyLinearComplexConv2d


class WidelyLinearComplexTest(unittest.TestCase):
    def test_antilinear_branch_can_represent_complex_conjugation(self):
        layer = WidelyLinearComplexConv2d(
            1, 1, kernel_size=1, padding=0, bias=False
        )
        with torch.no_grad():
            for parameter in layer.parameters():
                parameter.zero_()
            layer.antilinear.real_conv.weight.fill_(2.0**0.5)

        z = torch.complex(
            torch.tensor([[[[1.0, -2.0], [0.5, 3.0]]]]),
            torch.tensor([[[[0.25, 4.0], [-1.5, 2.0]]]]),
        )
        torch.testing.assert_close(layer(z), torch.conj(z))

    def test_factory_matches_real_parameter_budget_for_rx2_and_rx16(self):
        self.assertIn("su_mimo_widely_linear", SU_MIMO_MODEL_CHOICES)
        base_config = {
            "hidden_complex": 32,
            "zero_real": 22,
            "hidden_real": 66,
            "bits_per_symbol": 2,
            "num_iterations": 2,
            "kernel_size": 3,
            "zero_gate_hidden": 16,
        }
        for num_rx_ant in (2, 16):
            config = {**base_config, "num_rx_ant": num_rx_ant}
            reference = build_su_mimo_model(
                "su_mimo_phase_sensitive", config
            )
            widely_linear = build_su_mimo_model(
                "su_mimo_widely_linear", config
            )
            self.assertIsInstance(
                widely_linear, SUMIMOWidelyLinearReceiver
            )
            reference_count = sum(p.numel() for p in reference.parameters())
            actual_count = sum(p.numel() for p in widely_linear.parameters())
            self.assertLessEqual(
                abs(actual_count - reference_count),
                max(100, round(0.001 * reference_count)),
            )
            self.assertEqual(
                widely_linear.resolved_model_config[
                    "predicted_parameter_count"
                ],
                actual_count,
            )

    def test_forward_backward_and_layer_permutation_on_cpu(self):
        torch.manual_seed(1234)
        config = {
            "num_rx_ant": 2,
            "hidden_complex": 4,
            "zero_real": 4,
            "hidden_real": 8,
            "bits_per_symbol": 2,
            "num_iterations": 1,
            "kernel_size": 3,
            "zero_gate_hidden": 4,
        }
        model = build_su_mimo_model(
            "su_mimo_widely_linear", config
        ).train()
        batch_size, num_layers, num_rx, symbols, subcarriers = 2, 2, 2, 4, 6
        y = torch.randn(
            batch_size, num_rx, symbols, subcarriers, dtype=torch.complex64
        )
        h_hat = torch.randn(
            batch_size,
            num_layers,
            num_rx,
            symbols,
            subcarriers,
            dtype=torch.complex64,
        )
        pilot_mask = torch.zeros(
            batch_size, num_layers, 1, symbols, subcarriers
        )
        n0 = torch.full((batch_size,), 0.1)
        layer_mask = torch.ones(batch_size, num_layers, dtype=torch.bool)

        logits = model(y, h_hat, pilot_mask, n0, layer_mask)
        self.assertEqual(
            logits.shape,
            (batch_size, num_layers, 2, symbols, subcarriers),
        )
        loss = logits.square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            all(
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )

        model.eval()
        permutation = torch.tensor([1, 0])
        with torch.no_grad():
            reference = model(y, h_hat, pilot_mask, n0, layer_mask)
            permuted = model(
                y,
                h_hat.index_select(1, permutation),
                pilot_mask.index_select(1, permutation),
                n0,
                layer_mask.index_select(1, permutation),
            )
        torch.testing.assert_close(
            permuted,
            reference.index_select(1, permutation),
            atol=3e-5,
            rtol=3e-5,
        )


if __name__ == "__main__":
    unittest.main()
