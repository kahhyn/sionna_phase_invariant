import copy
import unittest

from data import channel_profile_hash, load_channel_profile, validate_channel_profile
from data.sionna_channel_backends import ChannelProfileSampler
from experiments.channel_generalization.suite_entries import load_suite


class ChannelProfileTest(unittest.TestCase):
    def test_tdl_profile_has_twenty_components(self):
        profile = load_channel_profile(
            "configs/channel_profiles/tdl_mix_normalized.json"
        )
        self.assertEqual(len(profile["components"]), 20)
        self.assertEqual(len({c["id"] for c in profile["components"]}), 20)

    def test_fixed_component_selection(self):
        profile = load_channel_profile(
            "configs/channel_profiles/tdl_mix_normalized.json",
            component_id="tdl_C_100ns",
        )
        self.assertEqual(profile["name"], "tdl_C_100ns")
        self.assertEqual(profile["components"][0]["tdl_model"], "C")

    def test_hash_is_stable_under_key_order(self):
        profile = load_channel_profile(
            "configs/channel_profiles/umi_normalized.json"
        )
        reordered = {key: profile[key] for key in reversed(profile)}
        self.assertEqual(channel_profile_hash(profile), channel_profile_hash(reordered))

    def test_invalid_profiles_fail_clearly(self):
        profile = load_channel_profile(
            "configs/channel_profiles/umi_normalized.json"
        )
        invalid = copy.deepcopy(profile)
        invalid["components"][0]["weight"] = -1
        with self.assertRaisesRegex(ValueError, "negative weight"):
            validate_channel_profile(invalid)
        invalid = copy.deepcopy(profile)
        invalid["components"][0]["backend"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            validate_channel_profile(invalid)

    def test_balanced_sampler_differs_by_at_most_one(self):
        profile = load_channel_profile(
            "configs/channel_profiles/umi_uma_mix_normalized.json"
        )
        sampler = ChannelProfileSampler(profile, seed=123)
        sequence = [sampler.next_index() for _ in range(9)]
        self.assertLessEqual(abs(sequence.count(0) - sequence.count(1)), 1)
        sampler.reset(123)
        self.assertEqual(sequence, [sampler.next_index() for _ in range(9)])

    def test_generalization_suite_expands_fixed_domains(self):
        suite = load_suite(
            "configs/channel_suites/generalization_normalized.json"
        )
        self.assertEqual(len(suite["domains"]), 22)
        self.assertEqual(len({d["id"] for d in suite["domains"]}), 22)


if __name__ == "__main__":
    unittest.main()
