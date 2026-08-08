import unittest

import torch

from data import filter_channel_profile
from models import build_su_mimo_model
from training.train_su_mimo import (
    FixedDatasetReplay,
    _model_state_hash,
    _seed_runtime,
    learning_rate_for_epoch,
    learning_rate_for_step,
)


class _FakeGenerator:
    def __init__(self):
        self.seed = 0

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)

    def generate_batch(self, batch_size):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        return {
            "value": torch.rand(batch_size, 3, generator=generator),
            "channel_profile_id": f"component_{self.seed % 3}",
        }


class SUMIMOTrainingControlTest(unittest.TestCase):
    def test_fixed_dataset_replays_exact_shards_and_exact_sample_count(self):
        replay = FixedDatasetReplay(
            _FakeGenerator(),
            num_samples=10,
            batch_size=4,
            seed=123,
            shuffle_batches=True,
        )
        self.assertEqual(replay.shard_sizes, (4, 4, 2))
        self.assertEqual(sum(replay.shard_sizes), 10)
        first_pass = {}
        second_pass = {}
        for step in range(replay.steps_per_pass):
            batch, data_pass, shard = replay.batch_for_step(step)
            self.assertEqual(data_pass, 0)
            first_pass[shard] = batch["value"].clone()
        for step in range(replay.steps_per_pass, 2 * replay.steps_per_pass):
            batch, data_pass, shard = replay.batch_for_step(step)
            self.assertEqual(data_pass, 1)
            second_pass[shard] = batch["value"].clone()
        self.assertEqual(set(first_pass), set(second_pass))
        for shard in first_pass:
            self.assertTrue(torch.equal(first_pass[shard], second_pass[shard]))
        self.assertEqual(replay.samples_seen(replay.steps_per_pass), 10)
        self.assertEqual(replay.samples_seen(2 * replay.steps_per_pass), 20)

    def test_fixed_dataset_is_identical_for_independent_model_runs(self):
        first = FixedDatasetReplay(_FakeGenerator(), 11, 4, 777)
        second = FixedDatasetReplay(_FakeGenerator(), 11, 4, 777)
        for step in range(7):
            batch_a, pass_a, shard_a = first.batch_for_step(step)
            batch_b, pass_b, shard_b = second.batch_for_step(step)
            self.assertEqual((pass_a, shard_a), (pass_b, shard_b))
            self.assertTrue(torch.equal(batch_a["value"], batch_b["value"]))

    def test_training_profile_can_be_filtered_by_model_delay_and_id(self):
        profile = {
            "schema_version": 1,
            "name": "mixed",
            "sampling": "balanced_batch",
            "components": [
                {
                    "id": "a30",
                    "backend": "tdl",
                    "weight": 1.0,
                    "tdl_model": "A",
                    "delay_spread_ns": 30,
                    "normalize_channel": True,
                },
                {
                    "id": "a100",
                    "backend": "tdl",
                    "weight": 1.0,
                    "tdl_model": "A",
                    "delay_spread_ns": 100,
                    "normalize_channel": True,
                },
                {
                    "id": "b100",
                    "backend": "tdl",
                    "weight": 1.0,
                    "tdl_model": "B",
                    "delay_spread_ns": 100,
                    "normalize_channel": True,
                },
            ],
        }
        filtered = filter_channel_profile(
            profile,
            component_ids=["a30", "a100", "b100"],
            tdl_models=["A"],
            delay_spread_min_ns=50,
            delay_spread_max_ns=150,
        )
        self.assertEqual([c["id"] for c in filtered["components"]], ["a100"])
        self.assertEqual(filtered["source_profile_name"], "mixed")

    def test_step_schedule_matches_declared_warmup_decay_and_tail(self):
        values = [
            learning_rate_for_step(
                step,
                total_steps=10,
                base_lr=1e-3,
                scheduler_name="cosine",
                lr_min=1e-5,
                warmup_steps=2,
                constant_tail_steps=2,
            )
            for step in range(1, 11)
        ]
        self.assertAlmostEqual(values[0], 5e-4)
        self.assertAlmostEqual(values[1], 1e-3)
        self.assertAlmostEqual(values[7], 1e-5)
        self.assertEqual(values[8:], [1e-5, 1e-5])

    def test_cosine_schedule_with_warmup_reaches_declared_endpoints(self):
        values = [
            learning_rate_for_epoch(
                epoch,
                total_epochs=10,
                base_lr=1e-3,
                scheduler_name="cosine",
                lr_min=1e-5,
                warmup_epochs=2,
            )
            for epoch in range(1, 11)
        ]
        self.assertAlmostEqual(values[0], 5e-4)
        self.assertAlmostEqual(values[1], 1e-3)
        self.assertAlmostEqual(values[2], 1e-3)
        self.assertAlmostEqual(values[-1], 1e-5)
        self.assertTrue(all(a >= b for a, b in zip(values[2:], values[3:])))

    def test_cosine_schedule_holds_lr_min_during_constant_tail(self):
        values = [
            learning_rate_for_epoch(
                epoch,
                total_epochs=13,
                base_lr=1e-3,
                scheduler_name="cosine",
                lr_min=1e-5,
                warmup_epochs=2,
                constant_tail_epochs=3,
            )
            for epoch in range(1, 14)
        ]
        self.assertAlmostEqual(values[0], 5e-4)
        self.assertAlmostEqual(values[1], 1e-3)
        self.assertAlmostEqual(values[2], 1e-3)
        self.assertAlmostEqual(values[9], 1e-5)
        self.assertEqual(values[10:], [1e-5, 1e-5, 1e-5])
        self.assertTrue(all(a >= b for a, b in zip(values[2:], values[3:])))

    def test_seed_precedes_model_construction_and_hashes_initialization(self):
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
        device = torch.device("cpu")
        _seed_runtime(1234, device)
        first = build_su_mimo_model("su_mimo_phase_sensitive", config)
        _seed_runtime(1234, device)
        second = build_su_mimo_model("su_mimo_phase_sensitive", config)
        _seed_runtime(1235, device)
        third = build_su_mimo_model("su_mimo_phase_sensitive", config)

        self.assertEqual(_model_state_hash(first), _model_state_hash(second))
        self.assertNotEqual(_model_state_hash(first), _model_state_hash(third))


if __name__ == "__main__":
    unittest.main()
