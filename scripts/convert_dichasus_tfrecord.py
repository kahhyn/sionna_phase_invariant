#!/usr/bin/env python3
"""Convert a DICHASUS TFRecord trajectory to a compact receiver HDF5 file.

The source CSI is measured on 32 antennas and 1024 subcarriers over 50 MHz.
This converter follows the official DICHASUS convention by applying fftshift,
then linearly resamples the central band to the receiver grid. It deliberately
stores one frequency response per measurement record; the 14-symbol OFDM
dimension is created as a constant-within-frame view during evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import h5py
import numpy as np
import tensorflow as tf


FORMAT_NAME = "dichasus_receiver_channel_v1"
SOURCE_NUM_ANTENNAS = 32
SOURCE_NUM_SUBCARRIERS = 1024


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert DICHASUS CSI TFRecords to a compact HDF5 channel file."
    )
    parser.add_argument(
        "--input_tfrecord",
        default=str(script_dir / "dichasus-0152.tfrecords"),
    )
    parser.add_argument(
        "--output_h5",
        default=str(script_dir / "dichasus-0152_72sc_30khz.h5"),
    )
    parser.add_argument("--dataset_id", default="dichasus-0152")
    parser.add_argument("--source_bandwidth_hz", type=float, default=50e6)
    parser.add_argument("--target_fft_size", type=int, default=72)
    parser.add_argument(
        "--target_subcarrier_spacing_hz", type=float, default=30e3
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_records",
        type=int,
        default=0,
        help="Convert only the first N records; zero converts the complete file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def feature_description() -> dict[str, Any]:
    return {
        "cfo": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "csi": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "pos-lidar": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "rot-lidar": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
        "snr": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "time": tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }


def parse_record(serialized: tf.Tensor) -> dict[str, tf.Tensor]:
    record = tf.io.parse_single_example(serialized, feature_description())
    return {
        "csi": tf.ensure_shape(
            tf.io.parse_tensor(record["csi"], out_type=tf.float32),
            (SOURCE_NUM_ANTENNAS, SOURCE_NUM_SUBCARRIERS, 2),
        ),
        "pos_lidar": tf.ensure_shape(
            tf.io.parse_tensor(record["pos-lidar"], out_type=tf.float64), (2,)
        ),
        "pos_tachy": tf.ensure_shape(
            tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64), (3,)
        ),
        "rotation_lidar_rad": tf.ensure_shape(record["rot-lidar"], ()),
        "snr_db": tf.ensure_shape(
            tf.io.parse_tensor(record["snr"], out_type=tf.float32),
            (SOURCE_NUM_ANTENNAS,),
        ),
        "cfo_hz": tf.ensure_shape(
            tf.io.parse_tensor(record["cfo"], out_type=tf.float32),
            (SOURCE_NUM_ANTENNAS,),
        ),
        "timestamp_s": tf.ensure_shape(record["time"], ()),
    }


def interpolation_plan(
    source_bandwidth_hz: float,
    target_fft_size: int,
    target_subcarrier_spacing_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_spacing_hz = source_bandwidth_hz / SOURCE_NUM_SUBCARRIERS
    source_frequencies_hz = (
        np.arange(SOURCE_NUM_SUBCARRIERS, dtype=np.float64)
        - SOURCE_NUM_SUBCARRIERS // 2
    ) * source_spacing_hz
    target_frequencies_hz = (
        np.arange(target_fft_size, dtype=np.float64) - target_fft_size // 2
    ) * target_subcarrier_spacing_hz
    if target_frequencies_hz[0] < source_frequencies_hz[0] or (
        target_frequencies_hz[-1] > source_frequencies_hz[-1]
    ):
        raise ValueError("The requested target grid extends beyond the source band.")

    right = np.searchsorted(source_frequencies_hz, target_frequencies_hz)
    right = np.clip(right, 1, SOURCE_NUM_SUBCARRIERS - 1)
    left = right - 1
    denominator = source_frequencies_hz[right] - source_frequencies_hz[left]
    weight_right = (target_frequencies_hz - source_frequencies_hz[left]) / denominator
    return source_frequencies_hz, target_frequencies_hz, left, weight_right


def create_extendable_dataset(
    handle: h5py.File,
    name: str,
    trailing_shape: tuple[int, ...],
    dtype: np.dtype | str,
    batch_size: int,
) -> h5py.Dataset:
    chunks = (max(1, batch_size), *trailing_shape)
    return handle.create_dataset(
        name,
        shape=(0, *trailing_shape),
        maxshape=(None, *trailing_shape),
        chunks=chunks,
        dtype=dtype,
        compression="lzf",
        shuffle=True,
    )


def append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    start = dataset.shape[0]
    end = start + values.shape[0]
    dataset.resize(end, axis=0)
    dataset[start:end] = values


def convert(args: argparse.Namespace) -> Path:
    input_path = Path(args.input_tfrecord).resolve()
    output_path = Path(args.output_h5).resolve()
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")

    if not input_path.is_file():
        raise FileNotFoundError(f"Input TFRecord does not exist: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}. Pass --overwrite to replace it."
        )
    if args.batch_size <= 0 or args.target_fft_size <= 0:
        raise ValueError("batch_size and target_fft_size must be positive.")
    if args.max_records < 0:
        raise ValueError("max_records must be non-negative.")
    if args.source_bandwidth_hz <= 0 or args.target_subcarrier_spacing_hz <= 0:
        raise ValueError("Bandwidth and subcarrier spacing must be positive.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if partial_path.exists():
        partial_path.unlink()

    (
        source_frequencies_hz,
        target_frequencies_hz,
        left_indices,
        weight_right,
    ) = interpolation_plan(
        args.source_bandwidth_hz,
        args.target_fft_size,
        args.target_subcarrier_spacing_hz,
    )
    weight_right = weight_right.astype(np.float32).reshape(1, 1, -1)

    raw = tf.data.TFRecordDataset([str(input_path)])
    if args.max_records > 0:
        raw = raw.take(args.max_records)
    dataset = (
        raw.map(parse_record, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    converted_records = 0
    previous_timestamp: float | None = None
    timestamp_decreases = 0
    try:
        with h5py.File(partial_path, "w") as handle:
            h_real = create_extendable_dataset(
                handle,
                "H_real",
                (SOURCE_NUM_ANTENNAS, args.target_fft_size),
                np.float32,
                args.batch_size,
            )
            h_imag = create_extendable_dataset(
                handle,
                "H_imag",
                (SOURCE_NUM_ANTENNAS, args.target_fft_size),
                np.float32,
                args.batch_size,
            )
            datasets = {
                "positions_lidar_m": create_extendable_dataset(
                    handle, "positions_lidar_m", (2,), np.float64, args.batch_size
                ),
                "positions_tachy_m": create_extendable_dataset(
                    handle, "positions_tachy_m", (3,), np.float64, args.batch_size
                ),
                "rotation_lidar_rad": create_extendable_dataset(
                    handle, "rotation_lidar_rad", (), np.float32, args.batch_size
                ),
                "measured_snr_db": create_extendable_dataset(
                    handle,
                    "measured_snr_db",
                    (SOURCE_NUM_ANTENNAS,),
                    np.float32,
                    args.batch_size,
                ),
                "cfo_hz": create_extendable_dataset(
                    handle,
                    "cfo_hz",
                    (SOURCE_NUM_ANTENNAS,),
                    np.float32,
                    args.batch_size,
                ),
                "timestamps_s": create_extendable_dataset(
                    handle, "timestamps_s", (), np.float32, args.batch_size
                ),
                "source_record_index": create_extendable_dataset(
                    handle, "source_record_index", (), np.int64, args.batch_size
                ),
            }
            handle.create_dataset(
                "antenna_indices", data=np.arange(SOURCE_NUM_ANTENNAS, dtype=np.int32)
            )
            handle.create_dataset(
                "source_frequencies_hz", data=source_frequencies_hz
            )
            handle.create_dataset(
                "target_frequencies_hz", data=target_frequencies_hz
            )

            for batch in dataset:
                csi_parts = batch["csi"].numpy()
                if not np.isfinite(csi_parts).all():
                    raise ValueError(
                        f"Non-finite CSI encountered near record {converted_records}."
                    )
                csi = np.fft.fftshift(
                    csi_parts[..., 0] + 1j * csi_parts[..., 1], axes=2
                )
                left = csi[:, :, left_indices]
                right = csi[:, :, left_indices + 1]
                resampled = left + weight_right * (right - left)

                timestamps = batch["timestamp_s"].numpy().astype(np.float32)
                if timestamps.size:
                    if previous_timestamp is not None:
                        timestamp_decreases += int(timestamps[0] < previous_timestamp)
                    timestamp_decreases += int(np.count_nonzero(np.diff(timestamps) < 0))
                    previous_timestamp = float(timestamps[-1])

                current_size = resampled.shape[0]
                append(h_real, np.asarray(resampled.real, dtype=np.float32))
                append(h_imag, np.asarray(resampled.imag, dtype=np.float32))
                append(datasets["positions_lidar_m"], batch["pos_lidar"].numpy())
                append(datasets["positions_tachy_m"], batch["pos_tachy"].numpy())
                append(
                    datasets["rotation_lidar_rad"],
                    batch["rotation_lidar_rad"].numpy().astype(np.float32),
                )
                append(
                    datasets["measured_snr_db"],
                    batch["snr_db"].numpy().astype(np.float32),
                )
                append(
                    datasets["cfo_hz"], batch["cfo_hz"].numpy().astype(np.float32)
                )
                append(datasets["timestamps_s"], timestamps)
                append(
                    datasets["source_record_index"],
                    np.arange(
                        converted_records,
                        converted_records + current_size,
                        dtype=np.int64,
                    ),
                )
                converted_records += current_size
                print(f"Converted {converted_records} records", end="\r", flush=True)

            if converted_records == 0:
                raise ValueError("The input TFRecord did not contain any records.")

            metadata = {
                "format_name": FORMAT_NAME,
                "format_version": 1,
                "dataset_id": args.dataset_id,
                "source_file_name": input_path.name,
                "source_num_antennas": SOURCE_NUM_ANTENNAS,
                "source_num_subcarriers": SOURCE_NUM_SUBCARRIERS,
                "source_bandwidth_hz": args.source_bandwidth_hz,
                "source_subcarrier_spacing_hz": (
                    args.source_bandwidth_hz / SOURCE_NUM_SUBCARRIERS
                ),
                "target_fft_size": args.target_fft_size,
                "target_subcarrier_spacing_hz": args.target_subcarrier_spacing_hz,
                "target_frequency_order": "fftshift ascending baseband frequencies",
                "frequency_resampling": "linear complex interpolation",
                "ofdm_symbol_expansion": "none; repeat each record during evaluation",
                "converted_records": converted_records,
                "source_record_order": "as stored in TFRecord; sort by timestamp for trajectories",
                "timestamp_decreases_in_source_order": timestamp_decreases,
            }
            handle.attrs["format_name"] = FORMAT_NAME
            handle.attrs["format_version"] = 1
            handle.attrs["dataset_id"] = args.dataset_id
            handle.attrs["converted_records"] = converted_records
            handle.attrs["timestamp_decreases_in_source_order"] = timestamp_decreases
            handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
            handle.flush()

        if output_path.exists():
            output_path.unlink()
        partial_path.replace(output_path)
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise

    print()
    print(f"Saved {converted_records} records to: {output_path}")
    print(
        "H shape: "
        f"[{converted_records}, {SOURCE_NUM_ANTENNAS}, {args.target_fft_size}]"
    )
    print(f"Output size: {output_path.stat().st_size / (1024 ** 2):.1f} MiB")
    return output_path


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
