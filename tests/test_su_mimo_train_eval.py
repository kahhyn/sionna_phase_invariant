import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


@unittest.skipUnless(torch.cuda.is_available(), "Sionna SU-MIMO tests require CUDA")
class SUMIMOTrainEvalCLITest(unittest.TestCase):
    def _run(self, *arguments):
        completed = subprocess.run(
            [sys.executable, *arguments],
            cwd=self.repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if completed.returncode != 0:
            self.fail(
                f"Command failed ({completed.returncode}): "
                f"{' '.join(arguments)}\nstdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        return completed

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_train_resume_and_ber_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            aggregate_csv = Path(tmpdir) / "ber.csv"
            layer_csv = Path(tmpdir) / "ber_layers.csv"

            first = self._run(
                "-m",
                "training.train_su_mimo",
                "--num_train",
                "1",
                "--num_val",
                "1",
                "--epochs",
                "1",
                "--batch_size",
                "1",
                "--snr_db_min",
                "8",
                "--snr_db_max",
                "8",
                "--num_ofdm_symbols",
                "4",
                "--fft_size",
                "12",
                "--dmrs_symbols",
                "1",
                "3",
                "--hidden_complex",
                "4",
                "--zero_real",
                "4",
                "--hidden_real",
                "8",
                "--num_iterations",
                "1",
                "--zero_gate_hidden",
                "4",
                "--save_dir",
                str(run_dir),
                "--seed",
                "31415",
                "--log_interval",
                "0",
            )
            self.assertIn("Epoch 001", first.stdout)
            self.assertTrue((run_dir / "best.pt").is_file())
            self.assertTrue((run_dir / "last.pt").is_file())
            self.assertTrue((run_dir / "history.csv").is_file())
            self.assertTrue((run_dir / "resolved_config.json").is_file())

            resumed = self._run(
                "-m",
                "training.train_su_mimo",
                "--resume_checkpoint",
                str(run_dir / "last.pt"),
                "--epochs",
                "2",
                "--log_interval",
                "0",
            )
            self.assertIn("from epoch 1", resumed.stdout)
            self.assertIn("Epoch 002", resumed.stdout)

            checkpoint = torch.load(
                run_dir / "last.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(len(checkpoint["history"]), 2)
            self.assertEqual(checkpoint["data_backend"], "sionna_su_mimo")
            self.assertEqual(checkpoint["sionna_su_mimo_config"]["fft_size"], 12)
            with (run_dir / "history.csv").open(newline="") as stream:
                history_rows = list(csv.DictReader(stream))
            self.assertEqual([row["epoch"] for row in history_rows], ["1", "2"])
            with (run_dir / "resolved_config.json").open() as stream:
                resolved = json.load(stream)
            self.assertEqual(resolved["selection_rule"], "minimum source-validation BCE")

            evaluated = self._run(
                "-m",
                "evaluation.eval_ber_su_mimo",
                "--checkpoint",
                str(run_dir / "best.pt"),
                "--snr_list",
                "4,8",
                "--num_samples",
                "1",
                "--batch_size",
                "1",
                "--seed",
                "2718",
                "--common_random_numbers",
                "--out_csv",
                str(aggregate_csv),
                "--out_layer_csv",
                str(layer_csv),
            )
            self.assertIn("errors", evaluated.stdout)
            with aggregate_csv.open(newline="") as stream:
                aggregate_rows = list(csv.DictReader(stream))
            with layer_csv.open(newline="") as stream:
                layer_rows = list(csv.DictReader(stream))
            self.assertEqual(len(aggregate_rows), 2)
            self.assertEqual(len(layer_rows), 4)
            self.assertEqual({row["snr_db"] for row in aggregate_rows}, {"4.0", "8.0"})
            self.assertEqual({row["layer"] for row in layer_rows}, {"0", "1"})
            self.assertTrue(all(int(row["valid_bits"]) == 96 for row in aggregate_rows))
            self.assertTrue(all(int(row["valid_bits"]) == 48 for row in layer_rows))
            self.assertTrue(
                all(
                    0.0 <= float(row["ber_ci95_low"])
                    <= float(row["ber"])
                    <= float(row["ber_ci95_high"])
                    <= 1.0
                    for row in aggregate_rows
                )
            )


if __name__ == "__main__":
    unittest.main()
