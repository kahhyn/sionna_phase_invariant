"""Evaluate A/C receivers on aligned desired and interfering QuaDRiGa links."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat

from data import SionnaOFDMBatchGenerator, SionnaOFDMConfig
from eval_quadriga_trajectory import (
    ReceiverRecord,
    batch_ranges,
    build_pilot_mask,
    comparable_ofdm_config,
    legacy_qpsk,
    load_receiver,
    normalize_channel_per_frame,
    per_frame_metrics,
    sample_common_phase,
    validate_experiment,
    write_csv,
)


@dataclass
class TwoLinkTrajectory:
    desired_h: torch.Tensor
    interferer_h: torch.Tensor
    desired_positions: np.ndarray
    interferer_positions: np.ndarray
    timestamps: np.ndarray
    frame_index: np.ndarray


def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"Could not parse floating-point value {token!r}."
            ) from error
        if math.isnan(value):
            raise argparse.ArgumentTypeError("NaN is not a valid SNR or SIR.")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("The list must not be empty.")
    return values


def _complex_tensor(payload: dict[str, Any], real_key: str, imag_key: str) -> torch.Tensor:
    missing = [key for key in (real_key, imag_key) if key not in payload]
    if missing:
        raise KeyError(f"Missing MAT variables: {missing}")
    real = np.asarray(payload[real_key], dtype=np.float32)
    imag = np.asarray(payload[imag_key], dtype=np.float32)
    if real.shape != imag.shape or real.ndim != 3:
        raise ValueError(
            f"{real_key}/{imag_key} must have equal [frame, symbol, carrier] shapes."
        )
    data = np.ascontiguousarray(real + 1j * imag, dtype=np.complex64)
    return torch.from_numpy(data)


def load_two_link_trajectory(path: Path, *, max_frames: int = 0) -> TwoLinkTrajectory:
    payload = loadmat(path, squeeze_me=False, struct_as_record=False)
    desired_h = _complex_tensor(payload, "H_desired_real", "H_desired_imag")
    interferer_h = _complex_tensor(
        payload, "H_interferer_real", "H_interferer_imag"
    )
    if desired_h.shape != interferer_h.shape:
        raise ValueError(
            "Desired and interferer channel tensors must have identical shapes."
        )

    num_frames = desired_h.shape[0]
    required = ("desired_positions_m", "interferer_positions_m", "timestamps_s")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing MAT variables: {missing}")
    desired_positions = np.asarray(payload["desired_positions_m"], dtype=np.float32)
    interferer_positions = np.asarray(
        payload["interferer_positions_m"], dtype=np.float32
    )
    timestamps = np.asarray(payload["timestamps_s"], dtype=np.float32).reshape(-1)
    if "frame_index" in payload:
        frame_index = np.asarray(payload["frame_index"], dtype=np.int64).reshape(-1)
    else:
        frame_index = np.arange(num_frames, dtype=np.int64)

    for name, positions in (
        ("desired_positions_m", desired_positions),
        ("interferer_positions_m", interferer_positions),
    ):
        if positions.shape != (num_frames, 3):
            raise ValueError(
                f"{name} must have shape ({num_frames}, 3), got {positions.shape}."
            )
    if timestamps.size != num_frames or frame_index.size != num_frames:
        raise ValueError("Trajectory metadata length does not match channel tensors.")

    if max_frames > 0:
        num_frames = min(num_frames, max_frames)
        desired_h = desired_h[:num_frames]
        interferer_h = interferer_h[:num_frames]
        desired_positions = desired_positions[:num_frames]
        interferer_positions = interferer_positions[:num_frames]
        timestamps = timestamps[:num_frames]
        frame_index = frame_index[:num_frames]

    return TwoLinkTrajectory(
        desired_h=desired_h,
        interferer_h=interferer_h,
        desired_positions=desired_positions,
        interferer_positions=interferer_positions,
        timestamps=timestamps,
        frame_index=frame_index,
    )


def build_interference_mask(
    mode: str,
    pilot_mask: torch.Tensor,
    *,
    partial_band_fraction: float,
) -> torch.Tensor:
    num_symbols, fft_size = pilot_mask.shape
    if mode == "cochannel_full":
        return torch.ones_like(pilot_mask, dtype=torch.bool)
    if mode == "data_only":
        return ~pilot_mask
    if mode == "partial_band":
        if not 0.0 < partial_band_fraction <= 1.0:
            raise ValueError("partial_band_fraction must be in (0, 1].")
        width = max(1, int(round(fft_size * partial_band_fraction)))
        start = (fft_size - width) // 2
        mask = torch.zeros(
            num_symbols, fft_size, dtype=torch.bool, device=pilot_mask.device
        )
        mask[:, start : start + width] = True
        return mask
    raise ValueError(f"Unsupported interference mode: {mode}")


def channel_power(h: torch.Tensor, active_mask: torch.Tensor | None = None) -> torch.Tensor:
    power = h.abs().square()
    if active_mask is not None:
        power = power * active_mask.to(power.dtype).view(1, *active_mask.shape)
    return power.mean(dim=(1, 2))


def compute_interference_scale(
    desired_h: torch.Tensor,
    interferer_h: torch.Tensor,
    active_mask: torch.Tensor,
    sir_db: float,
    normalization: str,
) -> torch.Tensor:
    """Return [frame, 1, 1] scales that realize requested received SIR."""
    if math.isinf(sir_db) and sir_db > 0:
        return torch.zeros(
            desired_h.shape[0], 1, 1, dtype=torch.float32, device=desired_h.device
        )
    if not math.isfinite(sir_db):
        raise ValueError("SIR must be finite or positive infinity.")

    desired_power = channel_power(desired_h)
    interferer_power = channel_power(interferer_h, active_mask)
    if torch.any(desired_power <= 0) or torch.any(interferer_power <= 0):
        raise ValueError("Desired and active interferer powers must be positive.")
    ratio = 10.0 ** (float(sir_db) / 10.0)

    if normalization == "per_frame":
        scale = torch.sqrt(desired_power / (interferer_power * ratio))
    elif normalization == "global":
        scalar = torch.sqrt(desired_power.mean() / (interferer_power.mean() * ratio))
        scale = scalar.expand_as(desired_power)
    else:
        raise ValueError(f"Unsupported SIR normalization: {normalization}")
    return scale.to(torch.float32).view(-1, 1, 1)


def db_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return math.inf
    return 10.0 * math.log10(max(numerator, 1e-30) / denominator)


@torch.no_grad()
def make_interference_batch(
    desired_h: torch.Tensor,
    interferer_h: torch.Tensor,
    interference_scale: torch.Tensor,
    snr_db: float,
    pilot_mask: torch.Tensor,
    interference_mask: torch.Tensor,
    ls_estimator: torch.nn.Module,
    phase_mode: str,
    narrow_phase_range: float,
    n0_mode: str,
    generator: torch.Generator,
    interference_generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Generate one paired batch; SNR is referenced to desired signal power."""
    device = desired_h.device
    batch_size, num_symbols, fft_size = desired_h.shape
    if interferer_h.shape != desired_h.shape:
        raise ValueError("Desired and interferer batch shapes do not match.")
    data_mask = ~pilot_mask
    num_data_symbols = int(data_mask.sum().item())

    desired_bits_data = torch.randint(
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
        bits[:, bit_index, data_mask] = desired_bits_data[:, :, bit_index].float()

    desired_x = torch.ones(
        batch_size,
        num_symbols,
        fft_size,
        dtype=torch.complex64,
        device=device,
    )
    desired_x[:, data_mask] = legacy_qpsk(desired_bits_data)

    interferer_bits = torch.randint(
        0,
        2,
        (batch_size, num_symbols, fft_size, 2),
        dtype=torch.int32,
        device=device,
        generator=interference_generator,
    )
    interferer_x = legacy_qpsk(interferer_bits)
    interferer_x = interferer_x * interference_mask.to(torch.complex64).view(
        1, num_symbols, fft_size
    )

    desired_waveform = desired_h * desired_x
    unscaled_interference = interferer_h * interferer_x
    interference_waveform = interference_scale * unscaled_interference
    desired_power = desired_waveform.abs().square().mean(dim=(1, 2))
    interference_power = interference_waveform.abs().square().mean(dim=(1, 2))

    thermal_n0 = desired_power / (10.0 ** (float(snr_db) / 10.0))
    noise_scale = torch.sqrt(thermal_n0 / 2.0).view(batch_size, 1, 1)
    noise = torch.complex(
        torch.randn(
            desired_waveform.shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        ),
        torch.randn(
            desired_waveform.shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        ),
    )
    y_unrotated = desired_waveform + interference_waveform + noise_scale * noise

    y_full = y_unrotated[:, None, None, :, :]
    h_hat_full, _ = ls_estimator(y_full, thermal_n0)
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
    desired_h_rotated = rotation * desired_h
    h_hat = rotation * h_hat_unrotated

    if n0_mode == "thermal":
        model_n0 = thermal_n0
    elif n0_mode == "oracle_total":
        model_n0 = thermal_n0 + interference_power
    else:
        raise ValueError(f"Unsupported N0 mode: {n0_mode}")

    p = pilot_mask.float().view(1, 1, num_symbols, fft_size)
    p = p.expand(batch_size, -1, -1, -1)
    loss_mask = data_mask.float().view(1, 1, num_symbols, fft_size)
    loss_mask = loss_mask.expand(batch_size, -1, -1, -1)

    return {
        "Y": y.to(torch.complex64),
        "H_hat": h_hat.to(torch.complex64),
        "P": p,
        "N0": model_n0.view(batch_size, 1).float(),
        "bits": bits,
        "loss_mask": loss_mask,
        "H": desired_h_rotated.to(torch.complex64),
        "desired_power": desired_power,
        "interference_power": interference_power,
        "thermal_n0": thermal_n0,
    }


@torch.no_grad()
def evaluate_condition(
    receivers: list[ReceiverRecord],
    trajectory: TwoLinkTrajectory,
    config: dict[str, Any],
    ls_estimator: torch.nn.Module,
    *,
    snr_db: float,
    sir_db: float,
    interference_mode: str,
    partial_band_fraction: float,
    sir_normalization: str,
    n0_mode: str,
    batch_size: int,
    seed: int,
    phase_mode: str,
    narrow_phase_range: float,
    normalize_channel: bool,
    window_frames: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pilot_mask = build_pilot_mask(config, device)
    interference_mask = build_interference_mask(
        interference_mode,
        pilot_mask,
        partial_band_fraction=partial_band_fraction,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    # Keep interferer symbols on an independent stream. This makes SIR=inf
    # consume the same desired-bit/noise/phase sequence as the no-interference
    # QuaDRiGa evaluator, while retaining common interferer symbols across SIRs.
    interference_generator = torch.Generator(device=device)
    interference_generator.manual_seed(seed + 1_000_003)

    desired_all = trajectory.desired_h.to(device=device, dtype=torch.complex64)
    interferer_all = trajectory.interferer_h.to(
        device=device, dtype=torch.complex64
    )
    if normalize_channel:
        desired_all = normalize_channel_per_frame(desired_all)
        interferer_all = normalize_channel_per_frame(interferer_all)
    scales = compute_interference_scale(
        desired_all,
        interferer_all,
        interference_mask,
        sir_db,
        sir_normalization,
    )

    stats: dict[str, dict[str, list[torch.Tensor]]] = {
        receiver.label: {"bce": [], "errors": [], "bits": []}
        for receiver in receivers
    }
    nmse_num: list[torch.Tensor] = []
    nmse_den: list[torch.Tensor] = []
    desired_powers: list[torch.Tensor] = []
    interference_powers: list[torch.Tensor] = []
    noise_powers: list[torch.Tensor] = []

    for start, end in batch_ranges(desired_all.shape[0], batch_size):
        batch = make_interference_batch(
            desired_all[start:end],
            interferer_all[start:end],
            scales[start:end],
            snr_db,
            pilot_mask,
            interference_mask,
            ls_estimator,
            phase_mode,
            narrow_phase_range,
            n0_mode,
            generator,
            interference_generator,
        )
        error = (batch["H_hat"] - batch["H"]).abs().square().sum(dim=(1, 2))
        reference = batch["H"].abs().square().sum(dim=(1, 2))
        nmse_num.append(error.cpu())
        nmse_den.append(reference.cpu())
        desired_powers.append(batch["desired_power"].cpu())
        interference_powers.append(batch["interference_power"].cpu())
        noise_powers.append(batch["thermal_n0"].cpu())

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

    nmse_num_np = torch.cat(nmse_num).numpy()
    nmse_den_np = torch.cat(nmse_den).numpy()
    desired_power_np = torch.cat(desired_powers).numpy()
    interference_power_np = torch.cat(interference_powers).numpy()
    noise_power_np = torch.cat(noise_powers).numpy()
    total_desired_power = float(desired_power_np.sum())
    total_interference_power = float(interference_power_np.sum())
    total_noise_power = float(noise_power_np.sum())
    achieved_sir_db = db_ratio(total_desired_power, total_interference_power)
    achieved_sinr_db = db_ratio(
        total_desired_power, total_interference_power + total_noise_power
    )
    h_hat_nmse = float(nmse_num_np.sum() / nmse_den_np.sum())

    train_seed = int(receivers[0].checkpoint["args"].get("seed", -1))
    train_profile = receivers[0].checkpoint.get("train_channel_profile")
    if isinstance(train_profile, dict):
        train_profile_name = train_profile.get("name", "unknown")
    else:
        train_profile_name = "legacy_tdl" if train_profile is None else str(train_profile)

    summary_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for receiver in receivers:
        bce_per_frame = torch.cat(stats[receiver.label]["bce"]).numpy()
        errors_per_frame = torch.cat(stats[receiver.label]["errors"]).numpy()
        bits_per_frame = torch.cat(stats[receiver.label]["bits"]).numpy()
        total_bce = float(bce_per_frame.sum())
        total_errors = int(errors_per_frame.sum())
        total_bits = int(bits_per_frame.sum())

        common = {
            "snr_db": snr_db,
            "sir_db": sir_db,
            "achieved_sir_db": achieved_sir_db,
            "achieved_sinr_db": achieved_sinr_db,
            "interference_mode": interference_mode,
            "partial_band_fraction": partial_band_fraction,
            "sir_normalization": sir_normalization,
            "n0_mode": n0_mode,
            "receiver_label": receiver.label,
            "model": receiver.model_name,
            "checkpoint": str(receiver.checkpoint_path),
            "train_seed": train_seed,
            "train_profile": train_profile_name,
            "eval_seed": seed,
            "phase_mode": phase_mode,
            "channel_normalization": "per_frame" if normalize_channel else "none",
        }
        summary_rows.append(
            {
                **common,
                "num_frames": trajectory.desired_h.shape[0],
                "bce": total_bce / total_bits,
                "ber": total_errors / total_bits,
                "bit_errors": total_errors,
                "valid_bits": total_bits,
                "h_hat_nmse": h_hat_nmse,
                "h_hat_nmse_db": 10.0 * math.log10(max(h_hat_nmse, 1e-30)),
                "mean_desired_power": float(desired_power_np.mean()),
                "mean_interference_power": float(interference_power_np.mean()),
                "mean_thermal_n0": float(noise_power_np.mean()),
            }
        )

        for window_start in range(0, trajectory.desired_h.shape[0], window_frames):
            window_end = min(
                window_start + window_frames, trajectory.desired_h.shape[0]
            )
            sl = slice(window_start, window_end)
            window_bce = float(bce_per_frame[sl].sum())
            window_errors = int(errors_per_frame[sl].sum())
            window_bits = int(bits_per_frame[sl].sum())
            window_nmse = float(nmse_num_np[sl].sum() / nmse_den_np[sl].sum())
            desired_position = trajectory.desired_positions[sl].mean(axis=0)
            interferer_position = trajectory.interferer_positions[sl].mean(axis=0)
            window_rows.append(
                {
                    **common,
                    "window_start_offset": window_start,
                    "window_end_offset_exclusive": window_end,
                    "start_frame_index": int(trajectory.frame_index[window_start]),
                    "end_frame_index": int(trajectory.frame_index[window_end - 1]),
                    "start_time_s": float(trajectory.timestamps[window_start]),
                    "end_time_s": float(trajectory.timestamps[window_end - 1]),
                    "desired_mean_x_m": float(desired_position[0]),
                    "desired_mean_y_m": float(desired_position[1]),
                    "desired_mean_z_m": float(desired_position[2]),
                    "interferer_mean_x_m": float(interferer_position[0]),
                    "interferer_mean_y_m": float(interferer_position[1]),
                    "interferer_mean_z_m": float(interferer_position[2]),
                    "num_frames": window_end - window_start,
                    "bce": window_bce / window_bits,
                    "ber": window_errors / window_bits,
                    "bit_errors": window_errors,
                    "valid_bits": window_bits,
                    "h_hat_nmse": window_nmse,
                    "h_hat_nmse_db": 10.0
                    * math.log10(max(window_nmse, 1e-30)),
                    "achieved_sir_db": db_ratio(
                        float(desired_power_np[sl].sum()),
                        float(interference_power_np[sl].sum()),
                    ),
                    "achieved_sinr_db": db_ratio(
                        float(desired_power_np[sl].sum()),
                        float(
                            interference_power_np[sl].sum()
                            + noise_power_np[sl].sum()
                        ),
                    ),
                }
            )

    return summary_rows, window_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate A/C receivers with a second QuaDRiGa cochannel link."
    )
    parser.add_argument("--channel_mat", required=True)
    parser.add_argument("--invariant_checkpoint", required=True)
    parser.add_argument("--strict_checkpoint", required=True)
    parser.add_argument("--output_dir", default="runs/quadriga_interference")
    parser.add_argument("--snr_list", default="10")
    parser.add_argument("--sir_list", default="inf,20,10,5,0")
    parser.add_argument(
        "--interference_mode",
        default="cochannel_full",
        choices=["cochannel_full", "data_only", "partial_band"],
    )
    parser.add_argument("--partial_band_fraction", type=float, default=0.25)
    parser.add_argument(
        "--sir_normalization", default="per_frame", choices=["per_frame", "global"]
    )
    parser.add_argument(
        "--n0_mode", default="thermal", choices=["thermal", "oracle_total"]
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--window_frames", type=int, default=64)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=777000)
    parser.add_argument("--independent_condition_randomness", action="store_true")
    parser.add_argument(
        "--phase_mode", default="fixed", choices=["fixed", "narrow", "uniform"]
    )
    parser.add_argument("--narrow_phase_range", type=float, default=math.pi / 8)
    parser.add_argument(
        "--normalization",
        default="checkpoint",
        choices=["checkpoint", "per_frame", "none"],
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snr_values = parse_float_list(args.snr_list)
    sir_values = parse_float_list(args.sir_list)
    if args.batch_size <= 0 or args.window_frames <= 0:
        raise ValueError("batch_size and window_frames must be positive.")
    if args.max_frames < 0:
        raise ValueError("max_frames must be non-negative.")

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
    if receivers[0].checkpoint["args"].get("seed") != receivers[1].checkpoint[
        "args"
    ].get("seed"):
        raise ValueError("A/C checkpoints must use the same training seed.")

    channel_path = Path(args.channel_mat)
    trajectory = load_two_link_trajectory(channel_path, max_frames=args.max_frames)
    config = validate_experiment(receivers, trajectory.desired_h)
    if trajectory.interferer_h.shape != trajectory.desired_h.shape:
        raise ValueError("Interferer grid does not match desired grid.")

    source_ofdm_config = SionnaOFDMConfig(**receivers[0].checkpoint["sionna_config"])
    reference_generator = SionnaOFDMBatchGenerator(
        source_ofdm_config,
        snr_db_min=0.0,
        snr_db_max=0.0,
        phase_mode="fixed",
        seed=args.seed,
        device=device,
    )
    ls_estimator = reference_generator.ls_estimator

    if args.normalization == "checkpoint":
        normalize_channel = bool(config["normalize_channel"])
    else:
        normalize_channel = args.normalization == "per_frame"

    print(f"Device: {device}")
    print(f"Two-link trajectory: {channel_path}")
    print(f"Frames: {trajectory.desired_h.shape[0]}")
    print(f"Interference mode: {args.interference_mode}")
    print(f"SIR normalization: {args.sir_normalization} | N0 mode: {args.n0_mode}")

    summary_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    condition_index = 0
    for snr_db in snr_values:
        for sir_db in sir_values:
            eval_seed = (
                args.seed + 100000 * condition_index
                if args.independent_condition_randomness
                else args.seed
            )
            condition_summary, condition_windows = evaluate_condition(
                receivers,
                trajectory,
                config,
                ls_estimator,
                snr_db=snr_db,
                sir_db=sir_db,
                interference_mode=args.interference_mode,
                partial_band_fraction=args.partial_band_fraction,
                sir_normalization=args.sir_normalization,
                n0_mode=args.n0_mode,
                batch_size=args.batch_size,
                seed=eval_seed,
                phase_mode=args.phase_mode,
                narrow_phase_range=args.narrow_phase_range,
                normalize_channel=normalize_channel,
                window_frames=args.window_frames,
                device=device,
            )
            summary_rows.extend(condition_summary)
            window_rows.extend(condition_windows)
            print(f"\nSNR {snr_db:g} dB | SIR {sir_db:g} dB")
            for row in condition_summary:
                print(
                    f"  {row['receiver_label']:18s} | BER {row['ber']:.6e} | "
                    f"BCE {row['bce']:.6f} | achieved SIR "
                    f"{row['achieved_sir_db']:.2f} dB | "
                    f"Hhat NMSE {row['h_hat_nmse_db']:.2f} dB"
                )
            condition_index += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "quadriga_interference_summary.csv"
    windows_path = output_dir / "quadriga_interference_windows.csv"
    write_csv(summary_path, summary_rows)
    write_csv(windows_path, window_rows)

    manifest = {
        **vars(args),
        "channel_mat": str(channel_path),
        "snr_list_parsed": snr_values,
        "sir_list_parsed": sir_values,
        "num_frames": int(trajectory.desired_h.shape[0]),
        "channel_shape": list(trajectory.desired_h.shape),
        "resolved_normalize_channel": normalize_channel,
        "ofdm_config": comparable_ofdm_config(config),
        "receiver_models": {
            receiver.label: receiver.model_name for receiver in receivers
        },
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
