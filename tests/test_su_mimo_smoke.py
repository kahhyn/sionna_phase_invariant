import tempfile
import unittest
from pathlib import Path

import torch

from data import SionnaSUMIMOBatchGenerator, SionnaSUMIMOConfig
from models import (
    SUMIMOPhaseInvariantReceiver,
    SUMIMOPhaseSensitiveReceiver,
    build_su_mimo_model,
)
from utils.metrics import masked_bce_with_logits


@unittest.skipUnless(torch.cuda.is_available(), "Sionna SU-MIMO tests require CUDA")
class SUMIMOSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")
        cls.config = SionnaSUMIMOConfig(
            num_layers=2,
            num_rx_ant=2,
            total_tx_power=1.0,
        )
        cls.generator = SionnaSUMIMOBatchGenerator(
            cls.config,
            snr_db_min=8.0,
            snr_db_max=8.0,
            phase_mode="fixed",
            seed=2468,
            device=cls.device,
        )
        cls.batch = cls.generator.generate_batch(2, return_aux=True)

    def _build_model(self):
        return SUMIMOPhaseInvariantReceiver(
            num_rx_ant=2,
            hidden_complex=4,
            zero_real=4,
            hidden_real=8,
            bits_per_symbol=2,
            num_iterations=1,
            zero_gate_hidden=4,
        ).to(self.device)

    def test_batch_shapes_fdm_and_total_power(self):
        batch = self.batch
        self.assertEqual(batch["Y"].shape, (2, 2, 14, 72))
        self.assertEqual(batch["H_hat"].shape, (2, 2, 2, 14, 72))
        self.assertEqual(batch["P"].shape, (2, 2, 1, 14, 72))
        self.assertEqual(batch["bits"].shape, (2, 2, 2, 14, 72))
        self.assertEqual(batch["loss_mask"].shape, (2, 2, 1, 14, 72))
        self.assertEqual(batch["layer_mask"].shape, (2, 2))
        self.assertTrue(torch.isfinite(batch["H_hat"].real).all())
        self.assertTrue(torch.isfinite(batch["H_hat"].imag).all())

        p0 = batch["P"][:, 0]
        p1 = batch["P"][:, 1]
        self.assertFalse(torch.logical_and(p0.bool(), p1.bool()).any())
        self.assertEqual(int(p0[0].sum().item()), 72)
        self.assertEqual(int(p1[0].sum().item()), 72)

        summed_power = batch["X"].abs().square().sum(dim=1)
        torch.testing.assert_close(
            summed_power,
            torch.ones_like(summed_power),
            atol=2e-6,
            rtol=0.0,
        )
        expected_layer_power = torch.full_like(
            batch["power_per_data_layer"], 0.5
        )
        torch.testing.assert_close(
            batch["power_per_data_layer"], expected_layer_power
        )

    def test_forward_backward_and_checkpoint_reload(self):
        batch = self.batch
        model = self._build_model().train()
        logits = model(
            batch["Y"],
            batch["H_hat"],
            batch["P"],
            batch["N0"],
            batch["layer_mask"],
        )
        self.assertEqual(logits.shape, batch["bits"].shape)
        loss = masked_bce_with_logits(
            logits, batch["bits"], batch["loss_mask"]
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))

        model.eval()
        with torch.no_grad():
            reference = model(
                batch["Y"],
                batch["H_hat"],
                batch["P"],
                batch["N0"],
                batch["layer_mask"],
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "su_mimo_smoke.pt"
            torch.save(model.state_dict(), checkpoint)
            restored = self._build_model().eval()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))
            with torch.no_grad():
                reloaded = restored(
                    batch["Y"],
                    batch["H_hat"],
                    batch["P"],
                    batch["N0"],
                    batch["layer_mask"],
                )
        torch.testing.assert_close(reference, reloaded)

    def test_common_phase_invariance(self):
        batch = self.batch
        model = self._build_model().eval()
        phi = torch.tensor([0.37, 1.91], device=self.device)
        rot_y = torch.polar(torch.ones_like(phi), phi).view(2, 1, 1, 1)
        rot_h = rot_y.unsqueeze(1)
        with torch.no_grad():
            reference = model(
                batch["Y"],
                batch["H_hat"],
                batch["P"],
                batch["N0"],
                batch["layer_mask"],
            )
            rotated = model(
                rot_y * batch["Y"],
                rot_h * batch["H_hat"],
                batch["P"],
                batch["N0"],
                batch["layer_mask"],
            )
        torch.testing.assert_close(reference, rotated, atol=3e-5, rtol=3e-5)

    def test_model_choices_share_exact_tdl_mix_parameter_budget(self):
        model_config = {
            "num_rx_ant": 2,
            "hidden_complex": 32,
            "zero_real": 22,
            "hidden_real": 66,
            "bits_per_symbol": 2,
            "num_iterations": 2,
            "kernel_size": 3,
            "zero_gate_hidden": 16,
        }
        invariant = build_su_mimo_model(
            "su_mimo_phase_invariant", model_config
        )
        sensitive = build_su_mimo_model(
            "su_mimo_phase_sensitive", model_config
        )
        self.assertIsInstance(invariant, SUMIMOPhaseInvariantReceiver)
        self.assertIsInstance(sensitive, SUMIMOPhaseSensitiveReceiver)
        self.assertEqual(sum(p.numel() for p in invariant.parameters()), 204599)
        self.assertEqual(sum(p.numel() for p in sensitive.parameters()), 204599)

    def test_phase_sensitive_control_is_not_invariant(self):
        batch = self.batch
        model = SUMIMOPhaseSensitiveReceiver(
            num_rx_ant=2,
            hidden_complex=4,
            zero_real=4,
            hidden_real=8,
            bits_per_symbol=2,
            num_iterations=1,
            zero_gate_hidden=4,
        ).to(self.device).eval()
        rotation = torch.polar(
            torch.ones(2, device=self.device),
            torch.tensor([0.71, 1.37], device=self.device),
        ).view(2, 1, 1, 1)
        with torch.no_grad():
            reference = model(
                batch["Y"], batch["H_hat"], batch["P"], batch["N0"]
            )
            rotated = model(
                rotation * batch["Y"],
                rotation.unsqueeze(1) * batch["H_hat"],
                batch["P"],
                batch["N0"],
            )
        self.assertGreater(float((reference - rotated).abs().max()), 1e-4)

    def test_layer_permutation_equivariance(self):
        batch = self.batch
        model = self._build_model().eval()
        permutation = torch.tensor([1, 0], device=self.device)
        with torch.no_grad():
            reference = model(
                batch["Y"],
                batch["H_hat"],
                batch["P"],
                batch["N0"],
                batch["layer_mask"],
            )
            permuted = model(
                batch["Y"],
                batch["H_hat"].index_select(1, permutation),
                batch["P"].index_select(1, permutation),
                batch["N0"],
                batch["layer_mask"].index_select(1, permutation),
            )
        torch.testing.assert_close(
            permuted,
            reference.index_select(1, permutation),
            atol=3e-5,
            rtol=3e-5,
        )

    def test_reset_reproduces_bits_channel_noise_and_phase(self):
        generator = SionnaSUMIMOBatchGenerator(
            self.config,
            snr_db_min=8.0,
            snr_db_max=8.0,
            phase_mode="uniform",
            seed=97531,
            device=self.device,
        )
        first = generator.generate_batch(1, return_aux=True)
        generator.reset()
        second = generator.generate_batch(1, return_aux=True)
        for key in ("X", "H_unrotated", "Y_unrotated", "phi", "N0", "bits"):
            torch.testing.assert_close(first[key], second[key])

    def test_existing_umi_profile_runs_without_reconfiguration(self):
        profile_path = (
            Path(__file__).resolve().parents[1]
            / "configs/channel_profiles/umi_normalized.json"
        )
        config = SionnaSUMIMOConfig(
            num_ofdm_symbols=4,
            fft_size=12,
            dmrs_symbol_indices=(1, 3),
            num_layers=2,
            num_rx_ant=2,
        )
        generator = SionnaSUMIMOBatchGenerator(
            config,
            snr_db_min=8.0,
            snr_db_max=8.0,
            phase_mode="fixed",
            seed=8642,
            device=self.device,
            channel_profile=profile_path,
        )
        batch = generator.generate_batch(1)
        self.assertEqual(batch["Y"].shape, (1, 2, 4, 12))
        self.assertEqual(batch["H_hat"].shape, (1, 2, 2, 4, 12))
        self.assertEqual(batch["channel_profile_name"], "umi_normalized")
        self.assertEqual(batch["scenario"], "umi")

    def test_tiny_overfit_decreases_bce(self):
        torch.manual_seed(1357)
        generator = SionnaSUMIMOBatchGenerator(
            self.config,
            snr_db_min=8.0,
            snr_db_max=8.0,
            phase_mode="fixed",
            seed=1357,
            device=self.device,
        )
        batch = generator.generate_batch(1)
        model = self._build_model().train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(12):
            logits = model(
                batch["Y"],
                batch["H_hat"],
                batch["P"],
                batch["N0"],
                batch["layer_mask"],
            )
            loss = masked_bce_with_logits(
                logits, batch["bits"], batch["loss_mask"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
