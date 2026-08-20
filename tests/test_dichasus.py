import csv
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from evaluation.aggregate_dichasus import aggregate
from evaluation.eval_dichasus import load_dichasus_antenna


def write_test_h5(path: Path) -> None:
    num_records = 5
    num_antennas = 2
    fft_size = 72
    record_value = np.arange(1, num_records + 1, dtype=np.float32)
    real = np.broadcast_to(
        record_value[:, None, None], (num_records, num_antennas, fft_size)
    ).copy()
    imag = 0.25 * real
    metadata = {
        "target_subcarrier_spacing_hz": 30000.0,
        "target_fft_size": fft_size,
    }
    with h5py.File(path, "w") as handle:
        handle.attrs["format_name"] = "dichasus_receiver_channel_v1"
        handle.attrs["dataset_id"] = "dichasus-test"
        handle.attrs["metadata_json"] = json.dumps(metadata)
        handle.create_dataset("H_real", data=real)
        handle.create_dataset("H_imag", data=imag)
        handle.create_dataset("antenna_indices", data=np.array([3, 7]))
        handle.create_dataset(
            "cfo_hz", data=np.arange(num_records * num_antennas).reshape(5, 2)
        )
        handle.create_dataset(
            "measured_snr_db",
            data=np.array([[1, 11], [2, 12], [3, 13], [4, 14], [5, 15]]),
        )
        handle.create_dataset(
            "positions_lidar_m",
            data=np.column_stack([record_value, -record_value]),
        )
        handle.create_dataset(
            "positions_tachy_m",
            data=np.column_stack([record_value, -record_value, record_value * 0]),
        )
        handle.create_dataset("source_record_index", data=np.arange(num_records))
        handle.create_dataset(
            "target_frequencies_hz",
            data=(np.arange(fft_size) - fft_size // 2) * 30000.0,
        )
        handle.create_dataset("timestamps_s", data=np.array([4, 1, 3, 0, 2]))


class DichasusLoaderTest(unittest.TestCase):
    def test_loader_sorts_time_and_selects_original_antenna_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.h5"
            write_test_h5(path)
            data = load_dichasus_antenna(path, antenna_index=7)

        np.testing.assert_array_equal(data.timestamps, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(data.source_record_indices, [3, 1, 4, 2, 0])
        np.testing.assert_allclose(data.h_frequency[:, 0].real, [4, 2, 5, 3, 1])
        self.assertEqual(data.h_frequency.shape, (5, 72))
        self.assertEqual(data.antenna_index, 7)

    def test_filter_then_chronological_stride(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.h5"
            write_test_h5(path)
            data = load_dichasus_antenna(
                path,
                antenna_index=7,
                min_measured_snr_db=12,
                start_record=1,
                record_stride=2,
            )
        np.testing.assert_array_equal(data.timestamps, [1, 3])
        np.testing.assert_array_equal(data.measured_snr_db, [12, 13])

    def test_unknown_antenna_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.h5"
            write_test_h5(path)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                load_dichasus_antenna(path, antenna_index=0)


class DichasusAggregatorTest(unittest.TestCase):
    def test_aggregator_pairs_a_and_c(self):
        fields = [
            "dataset_id",
            "train_profile",
            "receiver_label",
            "model",
            "snr_db",
            "phase_mode",
            "channel_normalization",
            "position_source",
            "start_record",
            "record_stride",
            "max_records",
            "min_measured_snr_db_filter",
            "train_seed",
            "antenna_index",
            "eval_seed",
            "bit_errors",
            "valid_bits",
            "ber",
            "bce",
            "h_hat_nmse_db",
            "measured_snr_db_mean",
        ]
        base = {
            "dataset_id": "dichasus-test",
            "train_profile": "tdl_mix_normalized",
            "snr_db": "10.0",
            "phase_mode": "fixed",
            "channel_normalization": "per_frame",
            "position_source": "lidar",
            "start_record": "0",
            "record_stride": "1",
            "max_records": "0",
            "min_measured_snr_db_filter": "-inf",
            "train_seed": "0",
            "antenna_index": "7",
            "eval_seed": "777000",
            "valid_bits": "1000",
            "h_hat_nmse_db": "-10",
            "measured_snr_db_mean": "15",
        }
        rows = [
            {
                **base,
                "receiver_label": "A_phase_invariant",
                "model": "single_branch_n0_gate",
                "bit_errors": "90",
                "ber": "0.09",
                "bce": "0.2",
            },
            {
                **base,
                "receiver_label": "C_strict_matched",
                "model": "strict_matched_complex_p_n0_gate",
                "bit_errors": "100",
                "ber": "0.10",
                "bce": "0.22",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run" / "dichasus_summary.csv"
            path.parent.mkdir()
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            aggregate_rows, paired_rows = aggregate(Path(directory))

        self.assertEqual(len(aggregate_rows), 2)
        self.assertEqual(len(paired_rows), 1)
        self.assertAlmostEqual(paired_rows[0]["a_minus_c_ber_mean"], -0.01)
        self.assertEqual(paired_rows[0]["a_wins"], 1)


if __name__ == "__main__":
    unittest.main()
