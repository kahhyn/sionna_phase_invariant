import unittest

import torch

from models.factory import build_model


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

    def test_strict_a_c_models_are_parameter_matched(self):
        y, h_hat, p, n0 = self._inputs()
        kwargs = {
            "bits_per_symbol": 2,
            "hidden": 8,
            "hidden_complex": 6,
            "zero_complex": 6,
            "branch_layers": 1,
            "zero_gate_hidden": 4,
        }
        invariant = build_model("single_branch_n0_gate", **kwargs)
        matched = build_model("strict_matched_complex_p_n0_gate", **kwargs)
        invariant_count = sum(p.numel() for p in invariant.parameters())
        matched_count = sum(p.numel() for p in matched.parameters())
        self.assertEqual(invariant_count, matched_count)

        for model in (invariant, matched):
            logits = model(y, h_hat, p, n0)
            self.assertEqual(logits.shape, (2, 2, 4, 5))
            self.assertTrue(torch.isfinite(logits).all())

        phase = torch.exp(1j * torch.tensor(0.71))
        reference = invariant(y, h_hat, p, n0)
        rotated = invariant(phase * y, phase * h_hat, p, n0)
        torch.testing.assert_close(rotated, reference, atol=1e-5, rtol=1e-5)

    def test_deeprx_a_c_models_are_parameter_matched_and_a_is_invariant(self):
        y, h_hat, p, n0 = self._inputs()
        kwargs = {
            "bits_per_symbol": 2,
            "hidden": 8,
            "hidden_complex": 2,
            "branch_layers": 5,
        }
        invariant = build_model("deeprx_invariant_a", **kwargs).eval()
        matched = build_model("deeprx_matched_c", **kwargs).eval()
        baseline = build_model("deeprx", **kwargs).eval()

        invariant_count = sum(param.numel() for param in invariant.parameters())
        matched_count = sum(param.numel() for param in matched.parameters())
        self.assertEqual(invariant_count, matched_count)

        for model in (baseline, invariant, matched):
            logits = model(y, h_hat, p, n0)
            self.assertEqual(logits.shape, (2, 2, 4, 5))
            self.assertTrue(torch.isfinite(logits).all())

        phase = torch.exp(1j * torch.tensor(0.71))
        reference = invariant(y, h_hat, p, n0)
        rotated = invariant(phase * y, phase * h_hat, p, n0)
        torch.testing.assert_close(rotated, reference, atol=1e-5, rtol=1e-5)

    def test_paper_input_deeprx_models_share_inputs_and_ignore_n0(self):
        y, h_hat, p, n0 = self._inputs()
        models = [
            build_model(
                "deeprx_compact_paper_input",
                bits_per_symbol=2,
                hidden=8,
                branch_layers=5,
            ).eval(),
            build_model("deeprx_paper11", bits_per_symbol=2).eval(),
        ]
        for model in models:
            reference = model(y, h_hat, p, n0)
            changed_n0 = model(y, h_hat, p, 10.0 * n0)
            self.assertEqual(reference.shape, (2, 2, 4, 5))
            self.assertTrue(torch.isfinite(reference).all())
            torch.testing.assert_close(changed_n0, reference)

        paper = models[1]
        self.assertEqual(len(paper.blocks), 11)

    def test_frozen_paper_input_a5_c5_contract(self):
        y, h_hat, p, n0 = self._inputs()
        d5 = build_model(
            "deeprx_compact_paper_input",
            bits_per_symbol=2,
            hidden=110,
            branch_layers=5,
        ).eval()
        a5 = build_model("deeprx_paper_input_a5", bits_per_symbol=2).eval()
        c5 = build_model("deeprx_paper_input_c5", bits_per_symbol=2).eval()

        d5_count = sum(parameter.numel() for parameter in d5.parameters())
        a5_count = sum(parameter.numel() for parameter in a5.parameters())
        c5_count = sum(parameter.numel() for parameter in c5.parameters())
        self.assertEqual(a5_count, c5_count)
        self.assertLess(abs(a5_count - d5_count) / d5_count, 0.01)
        self.assertEqual(a5_count, 90625)
        self.assertEqual(d5_count, 90972)

        self.assertEqual(a5.state_dict().keys(), c5.state_dict().keys())
        for key in a5.state_dict():
            self.assertEqual(a5.state_dict()[key].shape, c5.state_dict()[key].shape)

        for model in (a5, c5):
            conditional_gates = [
                module
                for module in model.modules()
                if module.__class__.__name__
                == "ConditionalAmplitudeSwiGLUGate"
            ]
            self.assertEqual(len(conditional_gates), 6)
            self.assertFalse(hasattr(model, "input_pilot_gate"))
            self.assertFalse(hasattr(model, "block_pilot_gates"))

        for model in (a5, c5):
            reference = model(y, h_hat, p, n0)
            changed_unused_inputs = model(y, 3.0 * h_hat, p, 10.0 * n0)
            self.assertEqual(reference.shape, (2, 2, 4, 5))
            self.assertTrue(torch.isfinite(reference).all())
            torch.testing.assert_close(changed_unused_inputs, reference)

        phase = torch.exp(1j * torch.tensor(0.73))
        a_reference, _ = a5.interaction_features(y, p)
        a_rotated, _ = a5.interaction_features(phase * y, p)
        torch.testing.assert_close(a_rotated, a_reference, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            a5(phase * y, h_hat, p, n0),
            a5(y, h_hat, p, n0),
            atol=1e-5,
            rtol=1e-5,
        )

        c_reference, _ = c5.interaction_features(y, p)
        c_rotated, _ = c5.interaction_features(phase * y, p)
        torch.testing.assert_close(
            c_rotated,
            phase * c_reference,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_frozen_deeprx_width_control_contract(self):
        y, h_hat, p, n0 = self._inputs()
        names = [
            "deeprx_width_control_d110",
            "deeprx_width_control_d64",
            "deeprx_width_control_a32",
            "deeprx_width_control_c32",
            "deeprx_width_control_a55",
        ]
        models = {
            name: build_model(
                name,
                bits_per_symbol=2,
                # Frozen aliases must ignore generic architecture knobs.
                hidden=7,
                hidden_complex=3,
                branch_layers=1,
                zero_gate_hidden=2,
            ).eval()
            for name in names
        }

        self.assertEqual(
            models["deeprx_width_control_d110"].backbone.input_conv.out_channels,
            110,
        )
        self.assertEqual(
            models["deeprx_width_control_d64"].backbone.input_conv.out_channels,
            64,
        )
        self.assertEqual(
            models["deeprx_width_control_a32"].input_proj.real_conv.out_channels,
            32,
        )
        self.assertEqual(
            models["deeprx_width_control_c32"].input_proj.real_conv.out_channels,
            32,
        )
        self.assertEqual(
            models["deeprx_width_control_a55"].input_proj.real_conv.out_channels,
            55,
        )
        expected_parameter_counts = {
            "deeprx_width_control_d110": 90972,
            "deeprx_width_control_d64": 34530,
            "deeprx_width_control_a32": 90625,
            "deeprx_width_control_c32": 90625,
            "deeprx_width_control_a55": 162054,
        }
        for name, model in models.items():
            self.assertEqual(
                sum(parameter.numel() for parameter in model.parameters()),
                expected_parameter_counts[name],
            )
        for model in models.values():
            logits = model(y, h_hat, p, n0)
            self.assertEqual(logits.shape, (2, 2, 4, 5))
            self.assertTrue(torch.isfinite(logits).all())

        phase = torch.exp(1j * torch.tensor(0.61))
        for name in ("deeprx_width_control_a32", "deeprx_width_control_a55"):
            model = models[name]
            torch.testing.assert_close(
                model(phase * y, h_hat, p, n0),
                model(y, h_hat, p, n0),
                atol=1e-5,
                rtol=1e-5,
            )

        a32 = models["deeprx_width_control_a32"]
        c32 = models["deeprx_width_control_c32"]
        self.assertEqual(a32.state_dict().keys(), c32.state_dict().keys())
        for key in a32.state_dict():
            self.assertEqual(a32.state_dict()[key].shape, c32.state_dict()[key].shape)

    def test_late_deeprx_a_c_are_parameter_matched_and_a_is_invariant(self):
        y, h_hat, p, n0 = self._inputs()
        kwargs = {
            "bits_per_symbol": 2,
            "hidden": 8,
            "hidden_complex": 4,
            "zero_complex": 4,
            "branch_layers": 5,
        }
        invariant = build_model("deeprx_late_invariant_a", **kwargs).eval()
        matched = build_model("deeprx_late_matched_c", **kwargs).eval()
        self.assertEqual(
            sum(param.numel() for param in invariant.parameters()),
            sum(param.numel() for param in matched.parameters()),
        )
        for model in (invariant, matched):
            logits = model(y, h_hat, p, n0)
            self.assertEqual(logits.shape, (2, 2, 4, 5))
            self.assertTrue(torch.isfinite(logits).all())

        phase = torch.exp(1j * torch.tensor(1.17))
        reference = invariant(y, h_hat, p, n0)
        rotated = invariant(phase * y, phase * h_hat, p, n0)
        torch.testing.assert_close(rotated, reference, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
