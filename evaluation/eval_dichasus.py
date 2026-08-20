"""Zero-shot A/C receiver evaluation on measured DICHASUS CSI."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from data import SionnaOFDMBatchGenerator, SionnaOFDMConfig
from eval_quadriga_trajectory import (
    comparable_ofdm_config,
    evaluate_one_snr,
    load_receiver,
    parse_float_list,
    validate_experiment,
    write_csv,
)


EXPECTED_FORMAT = "dichasus_receiver_channel_v1"


@dataclass
class DichasusAntennaData:
    h_frequency: torch.Tensor
    positions: np.ndarray
    timestamps: np.ndarray
    source_record_indices: np.ndarray
    measured_snr_db: np.ndarray
    cfo_hz: np.ndarray
    target_frequencies_hz: np.ndarray
    dataset_id: str
    antenna_index: int
    total_source_records: int
    metadata: dict[str, Any]


def _decode_attribute(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_dichasus_antenna(
    path: Path,
    *,
    antenna_index: int,
    position_source: str = "lidar",
    start_record: int = 0,
    record_stride: int = 1,
    max_records: int = 0,
    min_measured_snr_db: float = -math.inf,
) -> DichasusAntennaData:
    if start_record < 0 or record_stride <= 0 or max_records < 0:
        raise ValueError(
            "start_record/max_records must be non-negative and record_stride positive."
        )
    if position_source not in {"lidar", "tachy"}:
        raise ValueError("position_source must be 'lidar' or 'tachy'.")

    required = {
        "H_real",
        "H_imag",
        "antenna_indices",
        "cfo_hz",
        "measured_snr_db",
        "positions_lidar_m",
        "positions_tachy_m",
        "source_record_index",
        "target_frequencies_hz",
        "timestamps_s",
    }
    with h5py.File(path, "r") as handle:
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise KeyError(f"Missing HDF5 datasets in {path}: {missing}")
        format_name = _decode_attribute(handle.attrs.get("format_name", ""))
        if format_name != EXPECTED_FORMAT:
            raise ValueError(
                f"Unsupported DICHASUS format {format_name!r}; expected "
                f"{EXPECTED_FORMAT!r}."
            )

        antenna_indices = np.asarray(handle["antenna_indices"], dtype=np.int64)
        matches = np.flatnonzero(antenna_indices == antenna_index)
        if matches.size != 1:
            raise ValueError(
                f"Antenna {antenna_index} is unavailable; choices are "
                f"{antenna_indices.tolist()}."
            )
        antenna_slot = int(matches[0])

        real = np.asarray(handle["H_real"][:, antenna_slot, :], dtype=np.float32)
        imag = np.asarray(handle["H_imag"][:, antenna_slot, :], dtype=np.float32)
        if real.shape != imag.shape or real.ndim != 2:
            raise ValueError("H_real/H_imag must have [record, subcarrier] shape.")
        total_source_records = real.shape[0]
        timestamps_all = np.asarray(handle["timestamps_s"], dtype=np.float64)
        source_indices_all = np.asarray(
            handle["source_record_index"], dtype=np.int64
        )
        measured_snr_all = np.asarray(
            handle["measured_snr_db"][:, antenna_slot], dtype=np.float32
        )
        cfo_all = np.asarray(handle["cfo_hz"][:, antenna_slot], dtype=np.float32)
        if position_source == "lidar":
            xy = np.asarray(handle["positions_lidar_m"], dtype=np.float32)
            positions_all = np.column_stack(
                [xy, np.zeros(total_source_records, dtype=np.float32)]
            )
        else:
            positions_all = np.asarray(handle["positions_tachy_m"], dtype=np.float32)
        target_frequencies_hz = np.asarray(
            handle["target_frequencies_hz"], dtype=np.float64
        )
        metadata_json = _decode_attribute(handle.attrs.get("metadata_json", "{}"))
        metadata = json.loads(metadata_json)
        dataset_id = _decode_attribute(handle.attrs.get("dataset_id", "unknown"))

    expected_lengths = (
        timestamps_all.size,
        source_indices_all.size,
        measured_snr_all.size,
        cfo_all.size,
        positions_all.shape[0],
    )
    if any(length != total_source_records for length in expected_lengths):
        raise ValueError("DICHASUS metadata lengths do not match the CSI records.")

    # TFRecord storage order is not chronological. A stable timestamp sort is
    # required before trajectory subsampling and window statistics.
    selection = np.argsort(timestamps_all, kind="stable")
    selection = selection[measured_snr_all[selection] >= min_measured_snr_db]
    selection = selection[start_record::record_stride]
    if max_records > 0:
        selection = selection[:max_records]
    if selection.size == 0:
        raise ValueError("No records remain after DICHASUS filtering/subsampling.")

    h = np.ascontiguousarray(
        real[selection] + 1j * imag[selection], dtype=np.complex64
    )
    if not np.isfinite(h).all() or np.any(np.mean(np.abs(h) ** 2, axis=1) <= 0):
        raise ValueError("Selected DICHASUS CSI contains non-finite or zero-power rows.")
    if np.any(np.diff(timestamps_all[selection]) < 0):
        raise AssertionError("Internal error: DICHASUS timestamps were not sorted.")

    return DichasusAntennaData(
        h_frequency=torch.from_numpy(h),
        positions=np.ascontiguousarray(positions_all[selection]),
        timestamps=np.ascontiguousarray(timestamps_all[selection]),
        source_record_indices=np.ascontiguousarray(source_indices_all[selection]),
        measured_snr_db=np.ascontiguousarray(measured_snr_all[selection]),
        cfo_hz=np.ascontiguousarray(cfo_all[selection]),
        target_frequencies_hz=target_frequencies_hz,
        dataset_id=dataset_id,
        antenna_index=antenna_index,
        total_source_records=total_source_records,
        metadata=metadata,
    )


def train_metadata(receiver) -> tuple[str, int]:
    profile = receiver.checkpoint.get("train_channel_profile")
    if isinstance(profile, dict):
        profile_name = profile.get("name", "unknown")
    else:
        profile_name = "legacy_tdl" if profile is None else str(profile)
    train_seed = int(receiver.checkpoint["args"].get("seed", -1))
    return profile_name, train_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate A/C receivers using measured DICHASUS CSI as the channel. "
            "QPSK, DMRS and AWGN are synthesized identically for both receivers."
        )
    )
    parser.add_argument("--channel_h5", required=True)
    parser.add_argument("--antenna_index", type=int, required=True)
    parser.add_argument("--invariant_checkpoint", required=True)
    parser.add_argument("--strict_checkpoint", required=True)
    parser.add_argument("--output_dir", default="runs/dichasus_zero_shot")
    parser.add_argument("--snr_list", default="-10,-5,0,5,10,15,20")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--window_frames", type=int, default=128)
    parser.add_argument("--start_record", type=int, default=0)
    parser.add_argument("--record_stride", type=int, default=1)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--min_measured_snr_db", type=float, default=-math.inf)
    parser.add_argument(
        "--position_source", choices=["lidar", "tachy"], default="lidar"
    )
    parser.add_argument("--seed", type=int, default=777000)
    parser.add_argument("--independent_snr_randomness", action="store_true")
    parser.add_argument(
        "--phase_mode", choices=["fixed", "narrow", "uniform"], default="fixed"
    )
    parser.add_argument("--narrow_phase_range", type=float, default=math.pi / 8)
    parser.add_argument(
        "--normalization",
        choices=["checkpoint", "per_frame", "none"],
        default="checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snr_values = parse_float_list(args.snr_list)
    if args.batch_size <= 0 or args.window_frames <= 0:
        raise ValueError("batch_size and window_frames must be positive.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in PyTorch.")

    receivers = [
        load_receiver("A_phase_invariant", Path(args.invariant_checkpoint), device),
        load_receiver("C_strict_matched", Path(args.strict_checkpoint), device),
    ]
    if receivers[0].model_name != "single_branch_n0_gate":
        raise ValueError("--invariant_checkpoint is not single_branch_n0_gate.")
    if receivers[1].model_name != "strict_matched_complex_p_n0_gate":
        raise ValueError(
            "--strict_checkpoint is not strict_matched_complex_p_n0_gate."
        )

    data = load_dichasus_antenna(
        Path(args.channel_h5),
        antenna_index=args.antenna_index,
        position_source=args.position_source,
        start_record=args.start_record,
        record_stride=args.record_stride,
        max_records=args.max_records,
        min_measured_snr_db=args.min_measured_snr_db,
    )
    num_symbols = int(receivers[0].checkpoint["sionna_config"]["num_ofdm_symbols"])
    h_cpu = data.h_frequency[:, None, :].expand(-1, num_symbols, -1)
    config = validate_experiment(receivers, h_cpu)
    expected_spacing = float(data.metadata["target_subcarrier_spacing_hz"])
    if data.target_frequencies_hz.size > 1:
        actual_spacing = np.diff(data.target_frequencies_hz)
        if not np.allclose(actual_spacing, expected_spacing, atol=1e-6, rtol=0):
            raise ValueError("Converted DICHASUS target frequency grid is irregular.")

    reference_generator = SionnaOFDMBatchGenerator(
        SionnaOFDMConfig(**receivers[0].checkpoint["sionna_config"]),
        snr_db_min=0.0,
        snr_db_max=0.0,
        phase_mode="fixed",
        seed=args.seed,
        device=device,
    )
    if args.normalization == "checkpoint":
        normalize_channel = bool(config["normalize_channel"])
    else:
        normalize_channel = args.normalization == "per_frame"
    train_profile, train_seed = train_metadata(receivers[0])
    strict_profile, strict_seed = train_metadata(receivers[1])
    if (train_profile, train_seed) != (strict_profile, strict_seed):
        raise ValueError(
            "A/C checkpoints must use the same training profile and training seed; "
            f"got {(train_profile, train_seed)} and {(strict_profile, strict_seed)}."
        )

    print(f"Device: {device}")
    print(
        f"DICHASUS: {data.dataset_id} | antenna {data.antenna_index} | "
        f"records {h_cpu.shape[0]}/{data.total_source_records}"
    )
    print(
        f"Measured SNR mean {data.measured_snr_db.mean():.2f} dB | "
        f"synthetic evaluation SNRs: {snr_values}"
    )
    print(
        "Each measured CSI record is repeated over "
        f"{num_symbols} OFDM symbols within one frame."
    )

    summary_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for snr_index, snr_db in enumerate(snr_values):
        eval_seed = (
            args.seed + 100000 * snr_index
            if args.independent_snr_randomness
            else args.seed
        )
        snr_summary, snr_windows = evaluate_one_snr(
            receivers,
            h_cpu,
            data.positions,
            data.timestamps,
            data.source_record_indices,
            config,
            reference_generator.ls_estimator,
            snr_db=snr_db,
            batch_size=args.batch_size,
            seed=eval_seed,
            phase_mode=args.phase_mode,
            narrow_phase_range=args.narrow_phase_range,
            normalize_channel=normalize_channel,
            window_frames=args.window_frames,
            device=device,
        )
        shared = {
            "dataset_id": data.dataset_id,
            "antenna_index": data.antenna_index,
            "train_profile": train_profile,
            "train_seed": train_seed,
            "source_records_total": data.total_source_records,
            "start_record": args.start_record,
            "record_stride": args.record_stride,
            "max_records": args.max_records,
            "min_measured_snr_db_filter": args.min_measured_snr_db,
            "measured_snr_db_mean": float(data.measured_snr_db.mean()),
            "measured_snr_db_std": float(data.measured_snr_db.std()),
            "measured_snr_db_min": float(data.measured_snr_db.min()),
            "measured_snr_db_max": float(data.measured_snr_db.max()),
            "cfo_hz_mean": float(data.cfo_hz.mean()),
            "position_source": args.position_source,
        }
        for row in snr_summary:
            row.update(shared)
        for row in snr_windows:
            start = int(row["window_start_offset"])
            end = int(row["window_end_offset_exclusive"])
            row.update(
                {
                    "dataset_id": data.dataset_id,
                    "antenna_index": data.antenna_index,
                    "train_profile": train_profile,
                    "train_seed": train_seed,
                    "measured_snr_db_mean": float(
                        data.measured_snr_db[start:end].mean()
                    ),
                    "cfo_hz_mean": float(data.cfo_hz[start:end].mean()),
                    "position_source": args.position_source,
                }
            )
        summary_rows.extend(snr_summary)
        window_rows.extend(snr_windows)
        print(f"\nSNR {snr_db:g} dB")
        for row in snr_summary:
            print(
                f"  {row['receiver_label']:18s} | BER {row['ber']:.6e} | "
                f"BCE {row['bce']:.6f} | Hhat NMSE {row['h_hat_nmse_db']:.2f} dB"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "dichasus_summary.csv"
    windows_path = output_dir / "dichasus_windows.csv"
    write_csv(summary_path, summary_rows)
    write_csv(windows_path, window_rows)
    manifest = {
        **vars(args),
        "channel_h5": str(Path(args.channel_h5)),
        "num_selected_records": int(h_cpu.shape[0]),
        "total_source_records": data.total_source_records,
        "dataset_id": data.dataset_id,
        "train_profile": train_profile,
        "train_seed": train_seed,
        "resolved_normalize_channel": normalize_channel,
        "ofdm_config": comparable_ofdm_config(config),
        "converted_channel_metadata": data.metadata,
    }
    manifest["ofdm_config"]["dmrs_symbol_indices"] = list(
        manifest["ofdm_config"]["dmrs_symbol_indices"]
    )
    with (output_dir / "experiment_config.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved windows to {windows_path}")


if __name__ == "__main__":
    main()
