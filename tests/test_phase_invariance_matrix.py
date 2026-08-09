import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from evaluation.eval_quadriga_bler_matrix import load_channel_mat
from experiments import phase_invariance_matrix as matrix


class PhaseInvarianceMatrixTests(unittest.TestCase):
    def test_seed0_checkpoint_audit_excludes_uniform_val_siso_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            args = matrix.parse_args(
                [
                    "--mode",
                    "audit",
                    "--run_root",
                    directory,
                    "--seeds",
                    "0",
                ]
            )
            rows = matrix.audit_checkpoints(args)
        self.assertEqual(len(rows), 12)
        missing = [row for row in rows if not row["exists"]]
        self.assertEqual(len(missing), 8)
        siso_missing = [row for row in missing if row["system"] == "siso_1l1rx"]
        self.assertEqual(len(siso_missing), 4)
        mimo_missing = [row for row in missing if row["system"] != "siso_1l1rx"]
        self.assertEqual(len(mimo_missing), 4)
        self.assertEqual({row["train_domain"] for row in mimo_missing}, {"tdl_a"})

    def test_doppler_domain_expands_to_four_fixed_components(self):
        variants = matrix.test_variants("tdl_a", "doppler_ood")
        self.assertEqual([item[0] for item in variants], list(matrix.DOPPLER_COMPONENTS))
        self.assertTrue(all(item[2] == "uniform" for item in variants))

    def test_contract_declares_thirty_domain_cells(self):
        with (matrix.REPO_ROOT / matrix.CONTRACT_PATH).open() as stream:
            contract = json.load(stream)
        self.assertEqual(
            len(contract["systems"])
            * len(contract["train_domains"])
            * len(contract["test_domains"]),
            30,
        )

    def test_quadriga_loader_converts_rx_layer_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mimo.mat"
            real = np.zeros((2, 3, 2, 14, 72), dtype=np.float32)
            imag = np.zeros_like(real)
            real[:, 1, 0] = 7.0
            savemat(path, {"H_real": real, "H_imag": imag})
            channel = load_channel_mat(
                path,
                "su_mimo",
                "frame_rx_layer_symbol_subcarrier",
                max_frames=0,
            )
        self.assertEqual(tuple(channel.shape), (2, 2, 3, 14, 72))
        self.assertTrue((channel[:, 0, 1].real == 7.0).all())

    def test_quadriga_loader_rejects_siso_file_for_mimo(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "siso.mat"
            real = np.zeros((2, 14, 72), dtype=np.float32)
            savemat(path, {"H_real": real, "H_imag": real})
            with self.assertRaisesRegex(ValueError, "five-dimensional"):
                load_channel_mat(
                    path,
                    "su_mimo",
                    "frame_rx_layer_symbol_subcarrier",
                    max_frames=0,
                )


if __name__ == "__main__":
    unittest.main()
