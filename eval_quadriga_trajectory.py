"""Zero-shot evaluation of two neural receivers on a QuaDRiGa trajectory.

The script consumes the MAT file produced by ``generate_quadriga_trajectory.m``
and evaluates the native phase-invariant receiver against the strictly matched
phase-sensitive complex receiver. Bits, QPSK symbols, AWGN, LS channel
estimates, and optional common phase rotations are generated once per batch and
shared by both models.

This first trajectory experiment reports uncoded/pre-LDPC BER and BCE, matching
the metrics used by ``fewshot_finetune.py``. It also writes windowed results so
that performance changes along the continuous UE path can be inspected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from data import SionnaOFDMBatchGenerator, SionnaOFDMConfig
from utils.checkpoints import load_receiver_checkpoint


EXPECTED_MODELS = {
    "single_branch_n0_gate",
    "strict_matched_complex_p_n0_gate",
}


@dataclass
class ReceiverRecord:
    label: str
    model_name: str
    checkpoint_path: Path
    model: torch.nn.Module
    checkpoint: dict[str, Any]


def parse_float_list(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("The SNR list must not be empty.")
    return values


def batch_ranges(num_samples: int, batch_size: int):
    for start in range(0, num_samples, batch_size):
        yield start, min(start + batch_size, num_samples)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_trajectory(
    path: Path,
    *,
    max_frames: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw QuaDRiGa CFR and frame metadata.

    Returns a CPU complex64 tensor with shape [frame, symbol, subcarrier].
    """
    payload = loadmat(path, squeeze_me=False, struct_as_record=False)
    required = {"H_real", "H_imag", "positions_m", "timestamps_s"}
    missing = sorted(required.difference(payload))
    if missing:
        raise KeyError(f"Missing variables in {path}: {missing}")

    h_real = np.asarray(payload["H_real"], dtype=np.float32)
    h_imag = np.asarray(payload["H_imag"], dtype=np.float32)
    if h_real.shape != h_imag.shape or h_real.ndim != 3:
        raise ValueError(
            "H_real and H_imag must have the same [frame, symbol, carrier] shape."
        )
    h_numpy = np.ascontiguousarray(h_real + 1j * h_imag, dtype=np.complex64)

    positions = np.asarray(payload["positions_m"], dtype=np.float32)
    timestamps = np.asarray(payload["timestamps_s"], dtype=np.float32).reshape(-1)
    if "frame_index" in payload:
        frame_index = np.asarray(payload["frame_index"], dtype=np.int64).reshape(-1)
    else:
        frame_index = np.arange(h_numpy.shape[0], dtype=np.int64)

    num_frames = h_numpy.shape[0]
    if positions.shape != (num_frames, 3):
        raise ValueError(
            f"positions_m must have shape ({num_frames}, 3), got {positions.shape}."
        )
    if timestamps.size != num_frames or frame_index.size != num_frames:
        raise ValueError("Trajectory metadata length does not match the channel tensor.")

    if max_frames > 0:
        num_frames = min(max_frames, num_frames)
        h_numpy = h_numpy[:num_frames]
        positions = positions[:num_frames]
        timestamps = timestamps[:num_frames]
        frame_index = frame_index[:num_frames]

    return torch.from_numpy(h_numpy), positions, timestamps, frame_index


def load_receiver(
    label: str,
    checkpoint_path: Path,
    device: torch.device,
) -> ReceiverRecord:
    model, checkpoint = load_receiver_checkpoint(
        checkpoint_path,
        device,
        bits_per_symbol=2,
        required_data_backend="sionna",
    )
    model_name = checkpoint["model_name"]
    if model_name not in EXPECTED_MODELS:
        raise ValueError(
            f"{checkpoint_path} contains {model_name!r}; expected one of "
            f"{sorted(EXPECTED_MODELS)}."
        )
    model.eval()
    return ReceiverRecord(
        label=label,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        model=model,
        checkpoint=checkpoint,
    )


def comparable_ofdm_config(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "num_ofdm_symbols",
        "fft_size",
        "bits_per_symbol",
        "dmrs_symbol_indices",
        "dmrs_freq_spacing",
        "dmrs_freq_offset",
        "ls_interpolation_type",
        "normalize_channel",
    )
    comparable = {key: config[key] for key in keys}
    comparable["dmrs_symbol_indices"] = tuple(
        int(value) for value in comparable["dmrs_symbol_indices"]
    )
    return comparable


def validate_experiment(
    receivers: list[ReceiverRecord],
    h: torch.Tensor,
) -> dict[str, Any]:
    configs = [record.checkpoint["sionna_config"] for record in receivers]
    reference = comparable_ofdm_config(configs[0])
    for record, config in zip(receivers[1:], configs[1:]):
        if comparable_ofdm_config(config) != reference:
            raise ValueError(
                f"Checkpoint {record.checkpoint_path} uses a different OFDM/DMRS "
                "configuration. Common-random-number comparison is invalid."
            )

    if reference["bits_per_symbol"] != 2:
        raise ValueError("This evaluator currently supports QPSK checkpoints only.")
    if tuple(h.shape[1:]) != (
        reference["num_ofdm_symbols"],
        reference["fft_size"],
    ):
        raise ValueError(
            f"Channel shape {tuple(h.shape[1:])} does not match checkpoint grid "
            f"({reference['num_ofdm_symbols']}, {reference['fft_size']})."
        )
    return reference


def build_pilot_mask(config: dict[str, Any], device: torch.device) -> torch.Tensor:
    num_symbols = int(config["num_ofdm_symbols"])
    fft_size = int(config["fft_size"])
    mask = torch.zeros(num_symbols, fft_size, dtype=torch.bool, device=device)
    subcarriers = torch.arange(
        int(config["dmrs_freq_offset"]),
        fft_size,
        int(config["dmrs_freq_spacing"]),
        device=device,
    )
    for symbol_index in config["dmrs_symbol_indices"]:
        mask[int(symbol_index), subcarriers] = True
    return mask


def normalize_channel_per_frame(h: torch.Tensor) -> torch.Tensor:
    power = h.abs().square().mean(dim=(1, 2), keepdim=True)
    if torch.any(power <= 0):
        raise ValueError("The QuaDRiGa channel contains a zero-power frame.")
    return h / torch.sqrt(power)


def legacy_qpsk(bits: torch.Tensor) -> torch.Tensor:
    """Map [..., 2] bits using 00,01,10,11 -> (-1-j,-1+j,1-j,1+j)/sqrt(2)."""
    real = 2.0 * bits[..., 0].to(torch.float32) - 1.0
    imag = 2.0 * bits[..., 1].to(torch.float32) - 1.0
    return torch.complex(real, imag) / math.sqrt(2.0)


def sample_common_phase(
    batch_size: int,
    phase_mode: str,
    narrow_phase_range: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if phase_mode == "fixed":
        return torch.zeros(batch_size, device=device)
    unit = torch.rand(batch_size, device=device, generator=generator)
    if phase_mode == "narrow":
        return (2.0 * unit - 1.0) * narrow_phase_range
    if phase_mode == "uniform":
        return 2.0 * math.pi * unit
    raise ValueError(f"Unsupported phase mode: {phase_mode}")


@torch.no_grad()
def make_receiver_batch(
    h: torch.Tensor,
    snr_db: float,
    pilot_mask: torch.Tensor,
    ls_estimator: torch.nn.Module,
    phase_mode: str,
    narrow_phase_range: float,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Generate one common-random-number batch for all receivers."""
    device = h.device
    batch_size, num_symbols, fft_size = h.shape
    data_mask = ~pilot_mask
    num_data_symbols = int(data_mask.sum().item())

    bits_data = torch.randint(
        0,
        2,
        (batch_size, num_data_symbols, 2),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    bits = torch.zeros(
        batch_size,
        2,
        num_symbols,
        fft_size,
        dtype=torch.float32,
        device=device,
    )
    for bit_index in range(2):
        bits[:, bit_index, data_mask] = bits_data[:, :, bit_index].to(torch.float32)

    # QuaDRiGa supplies H only. Unit pilots and QPSK data are added here.
    x = torch.ones(
        batch_size,
        num_symbols,
        fft_size,
        dtype=torch.complex64,
        device=device,
    )
    x[:, data_mask] = legacy_qpsk(bits_data)
    y_clean = h * x

    signal_power = y_clean.abs().square().mean(dim=(1, 2))
    n0 = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise_scale = torch.sqrt(n0 / 2.0).view(batch_size, 1, 1)
    noise_real = torch.randn(
        y_clean.shape, device=device, generator=generator, dtype=torch.float32
    )
    noise_imag = torch.randn(
        y_clean.shape, device=device, generator=generator, dtype=torch.float32
    )
    y_unrotated = y_clean + noise_scale * torch.complex(noise_real, noise_imag)

    # Reuse the exact Sionna LSChannelEstimator configuration from training.
    # Its input/output dimensions match those in SionnaOFDMBatchGenerator.
    y_full = y_unrotated[:, None, None, :, :]
    h_hat_full, _ = ls_estimator(y_full, n0)
    h_hat_unrotated = h_hat_full[:, 0, 0, 0, 0]

    phi = sample_common_phase(
        batch_size,
        phase_mode,
        narrow_phase_range,
        generator,
        device,
    )
    rotation = torch.polar(torch.ones_like(phi), phi).view(batch_size, 1, 1)
    y = rotation * y_unrotated
    h_rotated = rotation * h
    h_hat = rotation * h_hat_unrotated

    p = pilot_mask.to(torch.float32).view(1, 1, num_symbols, fft_size)
    p = p.expand(batch_size, -1, -1, -1)
    loss_mask = (~pilot_mask).to(torch.float32).view(
        1, 1, num_symbols, fft_size
    )
    loss_mask = loss_mask.expand(batch_size, -1, -1, -1)

    return {
        "Y": y.to(torch.complex64),
        "H_hat": h_hat.to(torch.complex64),
        "P": p,
        "N0": n0.view(batch_size, 1).to(torch.float32),
        "bits": bits,
        "loss_mask": loss_mask,
        "H": h_rotated.to(torch.complex64),
    }


@torch.no_grad()
def per_frame_metrics(
    logits: torch.Tensor,
    bits: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = loss_mask.bool().expand_as(bits)
    bce = F.binary_cross_entropy_with_logits(logits, bits, reduction="none")
    bce_sum = (bce * valid).sum(dim=(1, 2, 3))
    errors = ((logits > 0) != bits.bool()).logical_and(valid).sum(dim=(1, 2, 3))
    valid_bits = valid.sum(dim=(1, 2, 3))
    return bce_sum, errors, valid_bits


@torch.no_grad()
def evaluate_one_snr(
    receivers: list[ReceiverRecord],
    h_cpu: torch.Tensor,
    positions: np.ndarray,
    timestamps: np.ndarray,
    frame_index: np.ndarray,
    config: dict[str, Any],
    ls_estimator: torch.nn.Module,
    *,
    snr_db: float,
    batch_size: int,
    seed: int,
    phase_mode: str,
    narrow_phase_range: float,
    normalize_channel: bool,
    window_frames: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pilot_mask = build_pilot_mask(config, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    stats: dict[str, dict[str, list[torch.Tensor]]] = {
        receiver.label: {"bce": [], "errors": [], "bits": []}
        for receiver in receivers
    }
    nmse_numerators: list[torch.Tensor] = []
    nmse_denominators: list[torch.Tensor] = []

    for start, end in batch_ranges(h_cpu.shape[0], batch_size):
        h = h_cpu[start:end].to(device=device, dtype=torch.complex64)
        if normalize_channel:
            h = normalize_channel_per_frame(h)

        batch = make_receiver_batch(
            h,
            snr_db,
            pilot_mask,
            ls_estimator,
            phase_mode,
            narrow_phase_range,
            generator,
        )
        error = (batch["H_hat"] - batch["H"]).abs().square().sum(dim=(1, 2))
        reference = batch["H"].abs().square().sum(dim=(1, 2))
        nmse_numerators.append(error.cpu())
        nmse_denominators.append(reference.cpu())

        # The same batch object is passed to both models before it is discarded.
        for receiver in receivers:
            logits = receiver.model(
                batch["Y"], batch["H_hat"], batch["P"], batch["N0"]
            )
            bce_sum, errors, valid_bits = per_frame_metrics(
                logits, batch["bits"], batch["loss_mask"]
            )
            stats[receiver.label]["bce"].append(bce_sum.cpu())
            stats[receiver.label]["errors"].append(errors.cpu())
            stats[receiver.label]["bits"].append(valid_bits.cpu())

    nmse_num = torch.cat(nmse_numerators).numpy()
    nmse_den = torch.cat(nmse_denominators).numpy()
    h_hat_nmse = float(nmse_num.sum() / nmse_den.sum())
    h_hat_nmse_db = 10.0 * math.log10(max(h_hat_nmse, 1e-30))

    summary_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for receiver in receivers:
        bce_per_frame = torch.cat(stats[receiver.label]["bce"]).numpy()
        errors_per_frame = torch.cat(stats[receiver.label]["errors"]).numpy()
        bits_per_frame = torch.cat(stats[receiver.label]["bits"]).numpy()
        total_bce = float(bce_per_frame.sum())
        total_errors = int(errors_per_frame.sum())
        total_bits = int(bits_per_frame.sum())

        summary_rows.append(
            {
                "snr_db": snr_db,
                "receiver_label": receiver.label,
                "model": receiver.model_name,
                "checkpoint": str(receiver.checkpoint_path),
                "num_frames": h_cpu.shape[0],
                "bce": total_bce / total_bits,
                "ber": total_errors / total_bits,
                "bit_errors": total_errors,
                "valid_bits": total_bits,
                "h_hat_nmse": h_hat_nmse,
                "h_hat_nmse_db": h_hat_nmse_db,
                "phase_mode": phase_mode,
                "channel_normalization": "per_frame" if normalize_channel else "none",
                "eval_seed": seed,
            }
        )

        for window_start in range(0, h_cpu.shape[0], window_frames):
            window_end = min(window_start + window_frames, h_cpu.shape[0])
            window_slice = slice(window_start, window_end)
            window_bce = float(bce_per_frame[window_slice].sum())
            window_errors = int(errors_per_frame[window_slice].sum())
            window_bits = int(bits_per_frame[window_slice].sum())
            window_nmse = float(
                nmse_num[window_slice].sum() / nmse_den[window_slice].sum()
            )
            mean_position = positions[window_slice].mean(axis=0)

            window_rows.append(
                {
                    "snr_db": snr_db,
                    "receiver_label": receiver.label,
                    "model": receiver.model_name,
                    "window_start_offset": window_start,
                    "window_end_offset_exclusive": window_end,
                    "start_frame_index": int(frame_index[window_start]),
                    "end_frame_index": int(frame_index[window_end - 1]),
                    "start_time_s": float(timestamps[window_start]),
                    "end_time_s": float(timestamps[window_end - 1]),
                    "mean_x_m": float(mean_position[0]),
                    "mean_y_m": float(mean_position[1]),
                    "mean_z_m": float(mean_position[2]),
                    "num_frames": window_end - window_start,
                    "bce": window_bce / window_bits,
                    "ber": window_errors / window_bits,
                    "bit_errors": window_errors,
                    "valid_bits": window_bits,
                    "h_hat_nmse": window_nmse,
                    "h_hat_nmse_db": 10.0 * math.log10(max(window_nmse, 1e-30)),
                }
            )

    return summary_rows, window_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate phase-invariant and strict matched receivers on QuaDRiGa UMi."
    )
    parser.add_argument("--channel_mat", required=True)
    parser.add_argument("--invariant_checkpoint", required=True)
    parser.add_argument("--strict_checkpoint", required=True)
    parser.add_argument("--output_dir", default="runs/quadriga_umi_zero_shot")
    parser.add_argument("--snr_list", default="-5,0,5,10,15,20")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--window_frames", type=int, default=64)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Use only the first N frames; zero means all frames.",
    )
    parser.add_argument("--seed", type=int, default=777000)
    parser.add_argument(
        "--independent_snr_randomness",
        action="store_true",
        help="Use different bits/noise at each SNR. Default reuses base randomness.",
    )
    parser.add_argument(
        "--phase_mode",
        default="fixed",
        choices=["fixed", "narrow", "uniform"],
    )
    parser.add_argument("--narrow_phase_range", type=float, default=math.pi / 8)
    parser.add_argument(
        "--normalization",
        default="checkpoint",
        choices=["checkpoint", "per_frame", "none"],
        help=(
            "checkpoint mirrors normalize_channel from source training; "
            "per_frame explicitly normalizes every QuaDRiGa frame."
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snr_values = parse_float_list(args.snr_list)
    if args.batch_size <= 0 or args.window_frames <= 0:
        raise ValueError("batch_size and window_frames must be positive.")
    if args.max_frames < 0:
        raise ValueError("max_frames must be non-negative.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in PyTorch.")

    invariant_checkpoint = Path(args.invariant_checkpoint)
    strict_checkpoint = Path(args.strict_checkpoint)
    receivers = [
        load_receiver("A_phase_invariant", invariant_checkpoint, device),
        load_receiver("C_strict_matched", strict_checkpoint, device),
    ]
    if receivers[0].model_name != "single_branch_n0_gate":
        raise ValueError("--invariant_checkpoint is not single_branch_n0_gate.")
    if receivers[1].model_name != "strict_matched_complex_p_n0_gate":
        raise ValueError(
            "--strict_checkpoint is not strict_matched_complex_p_n0_gate."
        )

    channel_path = Path(args.channel_mat)
    h_cpu, positions, timestamps, frame_index = load_trajectory(
        channel_path,
        max_frames=args.max_frames,
    )
    config = validate_experiment(receivers, h_cpu)

    # Building the existing generator once guarantees that H_hat is produced
    # by the same ResourceGrid, PilotPattern, and LSChannelEstimator used for
    # source training. Its TDL channel is constructed but never sampled here.
    source_ofdm_config = SionnaOFDMConfig(
        **receivers[0].checkpoint["sionna_config"]
    )
    sionna_reference = SionnaOFDMBatchGenerator(
        source_ofdm_config,
        snr_db_min=0.0,
        snr_db_max=0.0,
        phase_mode="fixed",
        seed=args.seed,
        device=device,
    )
    ls_estimator = sionna_reference.ls_estimator

    if args.normalization == "checkpoint":
        normalize_channel = bool(config["normalize_channel"])
    else:
        normalize_channel = args.normalization == "per_frame"

    print(f"Device: {device}")
    print(f"Trajectory: {channel_path}")
    print(f"Frames: {h_cpu.shape[0]} | grid: {tuple(h_cpu.shape[1:])}")
    print(
        "Channel normalization: "
        + ("per-frame unit power" if normalize_channel else "none")
    )
    for receiver in receivers:
        parameter_count = sum(
            parameter.numel() for parameter in receiver.model.parameters()
        )
        print(
            f"{receiver.label}: {receiver.model_name} | "
            f"parameters {parameter_count:,}"
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
            positions,
            timestamps,
            frame_index,
            config,
            ls_estimator,
            snr_db=snr_db,
            batch_size=args.batch_size,
            seed=eval_seed,
            phase_mode=args.phase_mode,
            narrow_phase_range=args.narrow_phase_range,
            normalize_channel=normalize_channel,
            window_frames=args.window_frames,
            device=device,
        )
        summary_rows.extend(snr_summary)
        window_rows.extend(snr_windows)
        print(f"\nSNR {snr_db:g} dB")
        for row in snr_summary:
            print(
                f"  {row['receiver_label']:18s} | "
                f"BER {row['ber']:.6e} | BCE {row['bce']:.6f} | "
                f"Hhat NMSE {row['h_hat_nmse_db']:.2f} dB"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "quadriga_umi_summary.csv", summary_rows)
    write_csv(output_dir / "quadriga_umi_trajectory_windows.csv", window_rows)

    manifest = {
        **vars(args),
        "channel_mat": str(channel_path),
        "invariant_checkpoint": str(invariant_checkpoint),
        "strict_checkpoint": str(strict_checkpoint),
        "snr_list_parsed": snr_values,
        "num_frames": int(h_cpu.shape[0]),
        "channel_shape": list(h_cpu.shape),
        "resolved_normalize_channel": normalize_channel,
        "ofdm_config": comparable_ofdm_config(config),
        "receiver_models": {
            receiver.label: receiver.model_name for receiver in receivers
        },
    }
    # Tuples are valid inputs to json.dump, but convert for a clearer manifest.
    manifest["ofdm_config"]["dmrs_symbol_indices"] = list(
        manifest["ofdm_config"]["dmrs_symbol_indices"]
    )
    with (output_dir / "experiment_config.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(f"\nSaved summary to {output_dir / 'quadriga_umi_summary.csv'}")
    print(
        "Saved trajectory windows to "
        f"{output_dir / 'quadriga_umi_trajectory_windows.csv'}"
    )


if __name__ == "__main__":
    main()
