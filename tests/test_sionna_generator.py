import math
import unittest

import torch

from data import (
    Sionna5GLDPCBatchGenerator,
    SionnaOFDMBatchGenerator,
    SionnaOFDMConfig,
)
from data.sionna_ofdm_generator import legacy_qpsk_points
from models.classical_receivers import SionnaLMMSEBaseline
from models import (
    N0GatedSingleBranchPhaseInvariantReceiver,
    SingleBranchPhaseInvariantReceiver,
)
from models.single_invariant_net import ZeroOrderAmplitudeGate


@unittest.skipUnless(torch.cuda.is_available(), "Sionna migration tests require CUDA")
class SionnaGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")
        cls.generator = SionnaOFDMBatchGenerator(
            snr_db_min=10.0,
            snr_db_max=10.0,
            phase_mode="fixed",
            seed=1234,
            device=cls.device,
        )
        cls.batch = cls.generator.generate_batch(2, return_aux=True)

    def test_legacy_qpsk_labeling(self):
        expected = torch.tensor(
            [-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j],
            dtype=torch.complex64,
            device=self.device,
        ) / math.sqrt(2.0)
        torch.testing.assert_close(legacy_qpsk_points(self.device), expected)

    def test_shapes_dtypes_and_device(self):
        batch = self.batch
        self.assertEqual(batch["Y"].shape, (2, 14, 72))
        self.assertEqual(batch["H_hat"].shape, (2, 14, 72))
        self.assertEqual(batch["bits"].shape, (2, 2, 14, 72))
        self.assertEqual(batch["P"].shape, (2, 1, 14, 72))
        self.assertEqual(batch["N0"].shape, (2, 1))
        self.assertEqual(batch["Y"].dtype, torch.complex64)
        self.assertEqual(batch["bits"].dtype, torch.float32)
        self.assertEqual(batch["Y"].device.type, "cuda")

    def test_dmrs_mask_and_symbols(self):
        batch = self.batch
        self.assertTrue(torch.all(batch["P"][:, :, 2, :] == 1))
        self.assertTrue(torch.all(batch["P"][:, :, 11, :] == 1))
        self.assertEqual(int(batch["P"][0].sum().item()), 144)
        self.assertTrue(torch.all(batch["X"][:, 2, :] == 1 + 0j))
        self.assertTrue(torch.all(batch["X"][:, 11, :] == 1 + 0j))

    def test_measured_snr_matches_requested_snr(self):
        signal = self.batch["Y_clean_unrotated"]
        noise = self.batch["Y_unrotated"] - signal
        measured = 10.0 * torch.log10(
            signal.abs().square().mean(dim=(1, 2))
            / noise.abs().square().mean(dim=(1, 2))
        )
        target = torch.full_like(measured, 10.0)
        torch.testing.assert_close(measured, target, atol=0.7, rtol=0.0)

    def test_single_branch_is_invariant_on_sionna_batch(self):
        model = SingleBranchPhaseInvariantReceiver(
            hidden_complex=4,
            zero_real=4,
            hidden_real=4,
            bits_per_symbol=2,
            num_blocks=1,
        ).to(self.device).eval()
        batch = self.batch
        phi = torch.tensor([0.37, 2.11], device=self.device).view(2, 1, 1)
        rot = torch.polar(torch.ones_like(phi), phi)
        with torch.no_grad():
            logits = model(batch["Y"], batch["H_hat"], batch["P"], batch["N0"])
            rotated_logits = model(
                rot * batch["Y"],
                rot * batch["H_hat"],
                batch["P"],
                batch["N0"],
            )
        torch.testing.assert_close(logits, rotated_logits, atol=2e-5, rtol=2e-5)

    def test_zero_order_gate_is_identity_at_initialization(self):
        gate = ZeroOrderAmplitudeGate(3, condition_hidden=4).to(self.device)
        z = torch.randn(2, 3, 5, 7, dtype=torch.complex64, device=self.device)
        condition = torch.randn(2, 2, 5, 7, device=self.device)
        torch.testing.assert_close(gate(z, condition), z)

    def test_zero_order_gate_can_respond_to_n0(self):
        gate = ZeroOrderAmplitudeGate(2, condition_hidden=1).to(self.device)
        with torch.no_grad():
            first = gate.condition_net[0]
            last = gate.condition_net[-1]
            first.weight.zero_()
            first.bias.zero_()
            first.weight[0, 1, 1, 1] = 1.0
            last.weight.zero_()
            last.bias.zero_()
            last.weight[:, 0, 0, 0] = 1.0
        z = torch.ones(1, 2, 3, 3, dtype=torch.complex64, device=self.device)
        low_n0 = torch.zeros(1, 2, 3, 3, device=self.device)
        high_n0 = low_n0.clone()
        high_n0[:, 1] = 1.0
        self.assertFalse(torch.allclose(gate(z, low_n0), gate(z, high_n0)))

    def test_n0_gated_single_branch_is_invariant(self):
        model = N0GatedSingleBranchPhaseInvariantReceiver(
            hidden_complex=4,
            zero_real=4,
            hidden_real=4,
            bits_per_symbol=2,
            num_blocks=1,
            zero_gate_hidden=3,
        ).to(self.device).eval()
        batch = self.batch
        phi = torch.tensor([0.61, 1.73], device=self.device).view(2, 1, 1)
        rot = torch.polar(torch.ones_like(phi), phi)
        with torch.no_grad():
            logits = model(batch["Y"], batch["H_hat"], batch["P"], batch["N0"])
            rotated_logits = model(
                rot * batch["Y"],
                rot * batch["H_hat"],
                batch["P"],
                batch["N0"],
            )
        torch.testing.assert_close(logits, rotated_logits, atol=2e-5, rtol=2e-5)

    def test_gate_condition_modes_mask_the_expected_input(self):
        batch = self.batch
        for mode in ["p_only", "n0_only", "p_n0"]:
            model = N0GatedSingleBranchPhaseInvariantReceiver(
                hidden_complex=4,
                zero_real=4,
                hidden_real=4,
                bits_per_symbol=2,
                num_blocks=1,
                zero_gate_hidden=3,
                zero_gate_condition=mode,
            ).to(self.device).eval()
            captured = []

            def capture_condition(_module, inputs):
                captured.append(inputs[1].detach())

            handle = model.input_zero_gate.register_forward_pre_hook(capture_condition)
            with torch.no_grad():
                model(batch["Y"], batch["H_hat"], batch["P"], batch["N0"])
            handle.remove()
            condition = captured[0]
            if mode == "p_only":
                self.assertTrue(torch.all(condition[:, 1] == 0))
                self.assertGreater(float(condition[:, 0].abs().sum()), 0.0)
            elif mode == "n0_only":
                self.assertTrue(torch.all(condition[:, 0] == 0))
                self.assertGreater(float(condition[:, 1].abs().sum()), 0.0)
            else:
                self.assertGreater(float(condition[:, 0].abs().sum()), 0.0)
                self.assertGreater(float(condition[:, 1].abs().sum()), 0.0)

    def test_comb_dmrs_generation(self):
        generator = SionnaOFDMBatchGenerator(
            SionnaOFDMConfig(dmrs_freq_spacing=2),
            snr_db_min=10.0,
            snr_db_max=10.0,
            seed=4321,
            device=self.device,
        )
        batch = generator.generate_batch(1)
        self.assertEqual(int(batch["P"].sum().item()), 72)
        self.assertTrue(torch.isfinite(batch["H_hat"].real).all())
        self.assertTrue(torch.isfinite(batch["H_hat"].imag).all())

    def test_same_seed_reuses_samples_across_snr(self):
        low_snr = SionnaOFDMBatchGenerator(
            snr_db_min=0.0,
            snr_db_max=0.0,
            phase_mode="uniform",
            seed=9876,
            device=self.device,
        ).generate_batch(2, return_aux=True)
        high_snr = SionnaOFDMBatchGenerator(
            snr_db_min=10.0,
            snr_db_max=10.0,
            phase_mode="uniform",
            seed=9876,
            device=self.device,
        ).generate_batch(2, return_aux=True)

        torch.testing.assert_close(low_snr["bits"], high_snr["bits"])
        torch.testing.assert_close(low_snr["X"], high_snr["X"])
        torch.testing.assert_close(low_snr["H"], high_snr["H"])
        torch.testing.assert_close(low_snr["phi"], high_snr["phi"])

        low_unit_noise = (
            low_snr["Y_unrotated"] - low_snr["Y_clean_unrotated"]
        ) / low_snr["N0"].sqrt().view(2, 1, 1)
        high_unit_noise = (
            high_snr["Y_unrotated"] - high_snr["Y_clean_unrotated"]
        ) / high_snr["N0"].sqrt().view(2, 1, 1)
        torch.testing.assert_close(
            low_unit_noise, high_unit_noise, atol=2e-6, rtol=2e-6
        )


@unittest.skipUnless(torch.cuda.is_available(), "Sionna migration tests require CUDA")
class SionnaLDPCGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda")
        cls.generator = Sionna5GLDPCBatchGenerator(
            ebno_db_min=0.0,
            ebno_db_max=0.0,
            seed=2468,
            device=cls.device,
        )
        cls.batch = cls.generator.generate_batch(2, return_aux=True)

    def test_ldpc_dimensions_fill_resource_grid(self):
        self.assertEqual(self.generator.n, 1728)
        self.assertEqual(self.generator.k, 864)
        self.assertEqual(self.batch["info_bits"].shape, (2, 864))
        self.assertEqual(self.batch["codeword_bits"].shape, (2, 1728))
        self.assertEqual(self.batch["ebno_db"].shape, (2, 1))

    def test_dense_grid_order_matches_rate_matched_codeword(self):
        dense_codeword = torch.stack(
            [
                self.batch["bits"][:, bit_index, self.generator._data_mask]
                for bit_index in range(2)
            ],
            dim=-1,
        ).reshape(2, self.generator.n)
        torch.testing.assert_close(dense_codeword, self.batch["codeword_bits"])

    def test_perfect_logits_decode_without_errors(self):
        perfect_logits = 20.0 * (2.0 * self.batch["codeword_bits"] - 1.0)
        info_hat = self.generator.decoder(perfect_logits)
        self.assertTrue(torch.equal(info_hat, self.batch["info_bits"]))

    def test_grid_logit_extraction_preserves_bit_order(self):
        codeword_logits = 20.0 * (2.0 * self.batch["codeword_bits"] - 1.0)
        grid_logits = torch.zeros(
            2, 2, 14, 72, dtype=torch.float32, device=self.device
        )
        reshaped = codeword_logits.reshape(2, -1, 2)
        for bit_index in range(2):
            grid_logits[:, bit_index, self.generator._data_mask] = reshaped[
                :, :, bit_index
            ]
        extracted = self.generator.extract_codeword_logits(grid_logits)
        torch.testing.assert_close(extracted, codeword_logits)
        info_hat = self.generator.decode_logits(grid_logits)
        self.assertTrue(torch.equal(info_hat, self.batch["info_bits"]))

    def test_lmmse_baseline_outputs_codeword_logits(self):
        receiver = SionnaLMMSEBaseline(self.generator, csi="ls")
        logits = receiver.codeword_logits(self.batch)
        self.assertEqual(logits.shape, self.batch["codeword_bits"].shape)
        self.assertTrue(torch.isfinite(logits).all())

    def test_lmmse_baseline_is_common_phase_invariant(self):
        receiver = SionnaLMMSEBaseline(self.generator, csi="ls")
        logits = receiver.codeword_logits(self.batch)
        phi = torch.tensor([0.53, 1.91], device=self.device).view(2, 1, 1)
        rot = torch.polar(torch.ones_like(phi), phi)
        rotated_batch = dict(self.batch)
        rotated_batch["Y"] = rot * self.batch["Y"]
        rotated_batch["H_hat"] = rot * self.batch["H_hat"]
        rotated_batch["H"] = rot * self.batch["H"]
        rotated_logits = receiver.codeword_logits(rotated_batch)
        torch.testing.assert_close(logits, rotated_logits, atol=2e-4, rtol=2e-4)

    def test_same_seed_reuses_ldpc_samples_across_ebno(self):
        low = Sionna5GLDPCBatchGenerator(
            ebno_db_min=0.0,
            ebno_db_max=0.0,
            phase_mode="uniform",
            seed=1357,
            device=self.device,
        ).generate_batch(1, return_aux=True)
        high = Sionna5GLDPCBatchGenerator(
            ebno_db_min=5.0,
            ebno_db_max=5.0,
            phase_mode="uniform",
            seed=1357,
            device=self.device,
        ).generate_batch(1, return_aux=True)
        torch.testing.assert_close(low["info_bits"], high["info_bits"])
        torch.testing.assert_close(low["codeword_bits"], high["codeword_bits"])
        torch.testing.assert_close(low["H"], high["H"])
        torch.testing.assert_close(low["phi"], high["phi"])

        low_noise = (low["Y_unrotated"] - low["Y_clean_unrotated"]) / low[
            "N0"
        ].sqrt().view(1, 1, 1)
        high_noise = (high["Y_unrotated"] - high["Y_clean_unrotated"]) / high[
            "N0"
        ].sqrt().view(1, 1, 1)
        torch.testing.assert_close(low_noise, high_noise, atol=2e-6, rtol=2e-6)
        expected_ratio = torch.tensor(10.0 ** (5.0 / 10.0), device=self.device)
        torch.testing.assert_close(
            low["N0"].squeeze() / high["N0"].squeeze(),
            expected_ratio,
            atol=2e-5,
            rtol=2e-5,
        )


if __name__ == "__main__":
    unittest.main()
