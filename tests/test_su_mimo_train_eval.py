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

    def test_fixed_finite_dataset_step_budget_and_scenario_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "finite"
            profile = (
                self.repo_root
                / "configs/channel_profiles/tdl_a_10_30_100_mix_normalized.json"
            )
            completed = self._run(
                "-m",
                "training.train_su_mimo",
                "--model",
                "su_mimo_phase_canonical",
                "--train_dataset_mode",
                "fixed",
                "--num_train",
                "3",
                "--train_steps",
                "3",
                "--validation_interval_steps",
                "2",
                "--num_val",
                "1",
                "--batch_size",
                "2",
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
                "--train_channel_profile",
                str(profile),
                "--val_channel_profile",
                str(profile),
                "--train_component_ids",
                "tdl_A_30ns",
                "--lr_scheduler",
                "cosine",
                "--lr_min",
                "0.0001",
                "--warmup_steps",
                "1",
                "--constant_tail_steps",
                "1",
                "--save_dir",
                str(run_dir),
                "--device",
                "cuda",
                "--seed",
                "101",
                "--log_interval",
                "0",
            )
            self.assertIn("Fixed training corpus: 3 distinct samples", completed.stdout)
            self.assertIn("Validation step 000003", completed.stdout)
            checkpoint = torch.load(
                run_dir / "last.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(checkpoint["global_step"], 3)
            self.assertIn(checkpoint["samples_seen"], {4, 5})
            self.assertEqual(checkpoint["training_dataset"]["num_distinct_samples"], 3)
            self.assertEqual(len(checkpoint["training_dataset"]["dataset_hash"]), 64)
            self.assertEqual(
                [c["id"] for c in checkpoint["train_channel_profile"]["components"]],
                ["tdl_A_30ns"],
            )
            self.assertEqual(
                checkpoint["val_channel_profile"]["name"],
                "tdl_A_10_30_100_mix_normalized",
            )
            self.assertEqual([row["global_step"] for row in checkpoint["history"]], [2, 3])

    def test_train_resume_and_ber_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            aggregate_csv = Path(tmpdir) / "ber.csv"
            layer_csv = Path(tmpdir) / "ber_layers.csv"
            bler_csv = Path(tmpdir) / "bler.csv"
            bler_layer_csv = Path(tmpdir) / "bler_layers.csv"
            profile = (
                self.repo_root
                / "configs/channel_profiles/tdl_a_10_30_100_mix_normalized.json"
            )

            first = self._run(
                "-m",
                "training.train_su_mimo",
                "--model",
                "su_mimo_phase_canonical",
                "--num_train",
                "1",
                "--num_val",
                "1",
                "--epochs",
                "2",
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
                "--train_channel_profile",
                str(profile),
                "--val_channel_profile",
                str(profile),
                "--seed",
                "31415",
                "--lr_scheduler",
                "cosine",
                "--lr_min",
                "0.0001",
                "--constant_tail_epochs",
                "1",
                "--deterministic_algorithms",
                "--log_interval",
                "0",
            )
            self.assertIn("Epoch 002", first.stdout)
            self.assertIn("initial model hash", first.stdout)
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
                "3",
                "--log_interval",
                "0",
            )
            self.assertIn("from epoch 2", resumed.stdout)
            self.assertIn("Epoch 003", resumed.stdout)

            checkpoint = torch.load(
                run_dir / "last.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(len(checkpoint["history"]), 3)
            self.assertEqual(checkpoint["data_backend"], "sionna_su_mimo")
            self.assertEqual(checkpoint["model_name"], "su_mimo_phase_canonical")
            self.assertEqual(checkpoint["lr_scheduler_state"]["name"], "cosine")
            self.assertEqual(checkpoint["lr_scheduler_state"]["last_epoch"], 3)
            self.assertAlmostEqual(
                checkpoint["lr_scheduler_state"]["last_lr"], 0.0001
            )
            self.assertEqual(len(checkpoint["initial_model_hash"]), 64)
            self.assertEqual(
                checkpoint["train_channel_profile"]["name"],
                "tdl_A_10_30_100_mix_normalized",
            )
            self.assertEqual(checkpoint["sionna_su_mimo_config"]["fft_size"], 12)
            with (run_dir / "history.csv").open(newline="") as stream:
                history_rows = list(csv.DictReader(stream))
            self.assertEqual(
                [row["epoch"] for row in history_rows], ["1", "2", "3"]
            )
            self.assertEqual(
                [float(row["lr"]) for row in history_rows],
                [0.001, 0.0001, 0.0001],
            )
            with (run_dir / "resolved_config.json").open() as stream:
                resolved = json.load(stream)
            self.assertEqual(resolved["selection_rule"], "minimum source-validation BCE")
            self.assertEqual(
                resolved["validation_policy"], "fixed-seed replay on every epoch"
            )
            self.assertEqual(resolved["lr_schedule"]["name"], "cosine")
            self.assertEqual(resolved["lr_schedule"]["constant_tail_epochs"], 1)
            self.assertTrue(resolved["args"]["deterministic_algorithms"])

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
                "--eval_channel_profile",
                str(profile),
                "--eval_component_id",
                "tdl_A_10ns",
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

            bler = self._run(
                "-m",
                "evaluation.eval_bler_su_mimo",
                "--checkpoint",
                str(run_dir / "best.pt"),
                "--ebno_list",
                "4",
                "--coderate",
                "0.5",
                "--decoder_iterations",
                "2",
                "--batch_size",
                "1",
                "--target_block_errors",
                "1",
                "--max_blocks",
                "1",
                "--eval_channel_profile",
                str(profile),
                "--eval_component_id",
                "tdl_A_10ns",
                "--out_csv",
                str(bler_csv),
                "--out_layer_csv",
                str(bler_layer_csv),
            )
            self.assertIn("frame BLER", bler.stdout)
            with bler_csv.open(newline="") as stream:
                bler_rows = list(csv.DictReader(stream))
            with bler_layer_csv.open(newline="") as stream:
                bler_layer_rows = list(csv.DictReader(stream))
            self.assertEqual(len(bler_rows), 1)
            self.assertEqual(len(bler_layer_rows), 2)
            self.assertEqual(int(bler_rows[0]["num_blocks"]), 1)
            self.assertEqual(int(bler_rows[0]["num_layer_blocks"]), 2)

            for receiver in ("neural_perfect", "lmmse_ls", "lmmse_perfect"):
                diagnostic_csv = Path(tmpdir) / f"{receiver}.csv"
                diagnostic = self._run(
                    "-m",
                    "evaluation.eval_bler_su_mimo",
                    "--receiver",
                    receiver,
                    "--checkpoint",
                    str(run_dir / "best.pt"),
                    "--ebno_list",
                    "8",
                    "--coderate",
                    "0.5",
                    "--decoder_iterations",
                    "2",
                    "--batch_size",
                    "1",
                    "--target_block_errors",
                    "1",
                    "--max_blocks",
                    "1",
                    "--eval_channel_profile",
                    str(profile),
                    "--eval_component_id",
                    "tdl_A_10ns",
                    "--out_csv",
                    str(diagnostic_csv),
                )
                self.assertIn(f"receiver: {receiver}", diagnostic.stdout)
                with diagnostic_csv.open(newline="") as stream:
                    diagnostic_rows = list(csv.DictReader(stream))
                self.assertEqual(len(diagnostic_rows), 1)
                self.assertEqual(diagnostic_rows[0]["receiver"], receiver)
                expected_csi = "perfect" if receiver.endswith("perfect") else "ls"
                self.assertEqual(diagnostic_rows[0]["csi"], expected_csi)


if __name__ == "__main__":
    unittest.main()
