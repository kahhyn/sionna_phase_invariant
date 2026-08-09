"""Paired LDPC BLER evaluation on external SISO or SU-MIMO QuaDRiGa CFRs.

The MAT input must contain ``H_real`` and ``H_imag``. SISO tensors use
``[frame, symbol, subcarrier]``. SU-MIMO tensors use
``[frame, rx, layer, symbol, subcarrier]`` by default and are converted to the
project's ``[frame, layer, rx, symbol, subcarrier]`` contract. The evaluator
generates bits, pilots, noise, LS estimates, and common phase exactly once per
batch and reuses them for the invariant and phase-sensitive receivers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat

from data import (
    Sionna5GLDPCBatchGenerator,
    Sionna5GLDPCSUMIMOBatchGenerator,
    SionnaLDPC5GConfig,
    SionnaOFDMConfig,
)
from evaluation.eval_ber_su_mimo import wilson_interval
from utils.checkpoints import load_receiver_checkpoint, load_su_mimo_checkpoint


SISO_MODELS = ("single_branch_n0_gate", "strict_matched_complex_p_n0_gate")
MIMO_MODELS = ("su_mimo_phase_canonical", "su_mimo_phase_sensitive")


def parse_float_list(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("The Eb/N0 list must not be empty.")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_channel_mat(
    path: Path,
    system: str,
    layout: str,
    max_frames: int,
) -> torch.Tensor:
    payload = loadmat(path, squeeze_me=False, struct_as_record=False)
    missing = sorted({"H_real", "H_imag"}.difference(payload))
    if missing:
        raise KeyError(f"Missing variables in {path}: {missing}")
    real = np.asarray(payload["H_real"], dtype=np.float32)
    imag = np.asarray(payload["H_imag"], dtype=np.float32)
    if real.shape != imag.shape:
        raise ValueError("H_real and H_imag must have identical shapes.")
    channel = np.ascontiguousarray(real + 1j * imag, dtype=np.complex64)
    if system == "siso":
        if channel.ndim != 3:
            raise ValueError(
                "SISO QuaDRiGa CFR must have [frame, symbol, subcarrier] shape."
            )
    else:
        if channel.ndim != 5:
            raise ValueError(
                "SU-MIMO QuaDRiGa CFR must be five-dimensional."
            )
        if layout == "frame_rx_layer_symbol_subcarrier":
            channel = channel.transpose(0, 2, 1, 3, 4)
        elif layout != "frame_layer_rx_symbol_subcarrier":
            raise ValueError(f"Unsupported MIMO MAT layout: {layout}")
        channel = np.ascontiguousarray(channel)
    if max_frames > 0:
        channel = channel[:max_frames]
    if channel.shape[0] == 0:
        raise ValueError("The QuaDRiGa channel contains no frames.")
    return torch.from_numpy(channel)


def normalize_per_frame(channel: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(1, channel.ndim))
    power = channel.abs().square().mean(dim=dimensions, keepdim=True)
    if torch.any(power <= 0):
        raise ValueError("The QuaDRiGa channel contains a zero-power frame.")
    return channel / torch.sqrt(power)


def _noise(
    shape: torch.Size,
    n0: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    view_shape = (shape[0],) + (1,) * (len(shape) - 1)
    scale = torch.sqrt(n0 / 2.0).view(view_shape)
    real = torch.randn(shape, generator=generator, device=device)
    imag = torch.randn(shape, generator=generator, device=device)
    return scale * torch.complex(real, imag)


def _dense_siso_bits(reference, bits_data: torch.Tensor) -> torch.Tensor:
    cfg = reference.config
    dense = torch.zeros(
        bits_data.shape[0],
        cfg.bits_per_symbol,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.float32,
        device=reference.device,
    )
    for bit_index in range(cfg.bits_per_symbol):
        dense[:, bit_index, reference._data_mask] = bits_data[
            :, :, bit_index
        ].to(torch.float32)
    return dense


def _dense_mimo_bits(reference, bits_data: torch.Tensor) -> torch.Tensor:
    cfg = reference.config
    dense = torch.zeros(
        bits_data.shape[0],
        cfg.num_layers,
        cfg.bits_per_symbol,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.float32,
        device=reference.device,
    )
    for layer_index in range(cfg.num_layers):
        for bit_index in range(cfg.bits_per_symbol):
            dense[:, layer_index, bit_index, reference._data_mask[layer_index]] = (
                bits_data[:, layer_index, :, bit_index].to(torch.float32)
            )
    return dense


@torch.no_grad()
def make_siso_batch(reference, channel: torch.Tensor, ebno_db: float) -> dict:
    batch_size = channel.shape[0]
    bits_data, metadata = reference._make_data_bits(batch_size)
    mapped = reference.mapper(bits_data.reshape(batch_size, 1, 1, -1))
    x = reference.grid_mapper(mapped)[:, 0, 0]
    y_clean = channel * x
    y_clean_full = y_clean[:, None, None]
    ebno = torch.full(
        (batch_size,), ebno_db, dtype=torch.float32, device=reference.device
    )
    n0 = reference._compute_noise_power(y_clean_full, ebno)
    y_unrotated = y_clean + _noise(
        y_clean.shape, n0, reference._torch_generator, reference.device
    )
    h_hat_full, _ = reference.ls_estimator(y_unrotated[:, None, None], n0)
    h_hat = h_hat_full[:, 0, 0, 0, 0]
    phi = reference._sample_phase(batch_size)
    rotation = torch.polar(torch.ones_like(phi), phi).view(batch_size, 1, 1)
    batch = {
        "Y": (rotation * y_unrotated).to(torch.complex64),
        "H": (rotation * channel).to(torch.complex64),
        "H_hat": (rotation * h_hat).to(torch.complex64),
        "P": reference._pilot_mask.expand(batch_size, -1, -1, -1),
        "N0": n0.view(batch_size, 1),
        "bits": _dense_siso_bits(reference, bits_data),
        "loss_mask": reference._loss_mask.expand(batch_size, -1, -1, -1),
    }
    batch.update(metadata)
    return batch


@torch.no_grad()
def make_mimo_batch(reference, channel: torch.Tensor, ebno_db: float) -> dict:
    batch_size = channel.shape[0]
    cfg = reference.config
    bits_data, metadata = reference._make_data_bits(batch_size)
    mapped = reference.mapper(bits_data.reshape(batch_size, 1, cfg.num_layers, -1))
    mapped = mapped * math.sqrt(cfg.total_tx_power / cfg.num_layers)
    x = reference.grid_mapper(mapped)[:, 0]
    y_clean = torch.einsum("blrtf,bltf->brtf", channel, x)
    ebno = torch.full(
        (batch_size,), ebno_db, dtype=torch.float32, device=reference.device
    )
    n0 = reference._compute_noise_power(ebno)
    y_unrotated = y_clean + _noise(
        y_clean.shape, n0, reference._torch_generator, reference.device
    )
    h_hat_full, _ = reference.ls_estimator(y_unrotated[:, None], n0)
    h_hat = h_hat_full[:, 0, :, 0].permute(0, 2, 1, 3, 4)
    phi = reference._sample_phase(batch_size)
    rotation_y = torch.polar(torch.ones_like(phi), phi).view(
        batch_size, 1, 1, 1
    )
    rotation_h = rotation_y.unsqueeze(1)
    pilots = reference._layer_pilot_mask.to(torch.float32).view(
        1, cfg.num_layers, 1, cfg.num_ofdm_symbols, cfg.fft_size
    )
    loss_mask = reference._data_mask.to(torch.float32).unsqueeze(1).unsqueeze(0)
    batch = {
        "Y": (rotation_y * y_unrotated).to(torch.complex64),
        "H": (rotation_h * channel).to(torch.complex64),
        "H_hat": (rotation_h * h_hat).to(torch.complex64),
        "P": pilots.expand(batch_size, -1, -1, -1, -1),
        "N0": n0.view(batch_size, 1),
        "bits": _dense_mimo_bits(reference, bits_data),
        "loss_mask": loss_mask.expand(batch_size, -1, -1, -1, -1),
        "layer_mask": torch.ones(
            batch_size,
            cfg.num_layers,
            dtype=torch.bool,
            device=reference.device,
        ),
    }
    batch.update(metadata)
    return batch


def _checkpoint_train_seed(checkpoint: dict) -> int:
    return int(checkpoint.get("args", {}).get("seed", -1))


def load_receivers(system: str, invariant_path: Path, sensitive_path: Path, device):
    if system == "siso":
        invariant = load_receiver_checkpoint(
            invariant_path, device, bits_per_symbol=2, required_data_backend="sionna"
        )
        sensitive = load_receiver_checkpoint(
            sensitive_path, device, bits_per_symbol=2, required_data_backend="sionna"
        )
        pairs = [
            ("invariant", *invariant),
            ("sensitive", *sensitive),
        ]
        names = tuple(checkpoint["model_name"] for _, _, checkpoint in pairs)
        if names != SISO_MODELS:
            raise ValueError(f"Expected SISO models {SISO_MODELS}, got {names}.")
        configs = [checkpoint["sionna_config"] for _, _, checkpoint in pairs]
    else:
        invariant = load_su_mimo_checkpoint(invariant_path, device)
        sensitive = load_su_mimo_checkpoint(sensitive_path, device)
        pairs = [
            ("invariant", invariant[0], invariant[2]),
            ("sensitive", sensitive[0], sensitive[2]),
        ]
        names = tuple(checkpoint["model_name"] for _, _, checkpoint in pairs)
        if names != MIMO_MODELS:
            raise ValueError(f"Expected SU-MIMO models {MIMO_MODELS}, got {names}.")
        configs = [checkpoint["sionna_su_mimo_config"] for _, _, checkpoint in pairs]
    if configs[0] != configs[1]:
        raise ValueError("The paired checkpoints use different PHY configurations.")
    counts = [sum(parameter.numel() for parameter in model.parameters()) for _, model, _ in pairs]
    if counts[0] != counts[1]:
        raise ValueError(f"Parameter counts are not matched: {counts}.")
    for _, model, _ in pairs:
        model.eval()
    return pairs, configs[0], counts[0]


@torch.no_grad()
def evaluate_one_ebno(
    system: str,
    receivers,
    reference,
    channel_cpu: torch.Tensor,
    ebno_db: float,
    batch_size: int,
    repetitions: int,
    seed: int,
    normalize_channel: bool,
) -> tuple[list[dict], list[dict]]:
    reference.reset(seed)
    aggregate = {
        label: {
            "block_errors": 0,
            "info_errors": 0,
            "coded_errors": 0,
            "coded_bce": 0.0,
            "layer_block_errors": None,
            "layer_info_errors": None,
        }
        for label, _, _ in receivers
    }
    num_layers = 1 if system == "siso" else reference.config.num_layers
    for stats in aggregate.values():
        stats["layer_block_errors"] = torch.zeros(num_layers, dtype=torch.int64)
        stats["layer_info_errors"] = torch.zeros(num_layers, dtype=torch.int64)

    total_frames = 0
    for _ in range(repetitions):
        for start in range(0, channel_cpu.shape[0], batch_size):
            channel = channel_cpu[start : start + batch_size].to(
                device=reference.device, dtype=torch.complex64
            )
            if normalize_channel:
                channel = normalize_per_frame(channel)
            batch = (
                make_siso_batch(reference, channel, ebno_db)
                if system == "siso"
                else make_mimo_batch(reference, channel, ebno_db)
            )
            for label, model, _ in receivers:
                if system == "siso":
                    logits = model(batch["Y"], batch["H_hat"], batch["P"], batch["N0"])
                    codeword_logits = reference.extract_codeword_logits(logits)
                    info_hat = reference.decode_logits(logits)
                    info_errors = info_hat.bool() != batch["info_bits"].bool()
                    coded_errors = (codeword_logits > 0) != batch["codeword_bits"].bool()
                    layer_block_error = info_errors.any(dim=-1).view(-1, 1)
                    layer_info_errors = info_errors.sum(dim=-1).view(-1, 1)
                else:
                    logits = model(
                        batch["Y"],
                        batch["H_hat"],
                        batch["P"],
                        batch["N0"],
                        batch["layer_mask"],
                    )
                    codeword_logits = reference.extract_codeword_logits(logits)
                    info_hat = reference.decode_logits(logits)
                    info_errors = info_hat.bool() != batch["info_bits"].bool()
                    coded_errors = (codeword_logits > 0) != batch["codeword_bits"].bool()
                    layer_block_error = info_errors.any(dim=-1)
                    layer_info_errors = info_errors.sum(dim=-1)
                user_frame_error = layer_block_error.any(dim=-1)
                stats = aggregate[label]
                stats["block_errors"] += int(user_frame_error.sum().item())
                stats["info_errors"] += int(info_errors.sum().item())
                stats["coded_errors"] += int(coded_errors.sum().item())
                stats["coded_bce"] += float(
                    F.binary_cross_entropy_with_logits(
                        codeword_logits, batch["codeword_bits"], reduction="sum"
                    ).item()
                )
                stats["layer_block_errors"] += layer_block_error.sum(dim=0).cpu()
                stats["layer_info_errors"] += layer_info_errors.sum(dim=0).cpu()
            total_frames += channel.shape[0]

    rows = []
    layer_rows = []
    total_info_bits = total_frames * num_layers * reference.k
    total_coded_bits = total_frames * num_layers * reference.n
    for label, _, checkpoint in receivers:
        stats = aggregate[label]
        low, high = wilson_interval(stats["block_errors"], total_frames)
        common = {
            "ebno_db": ebno_db,
            "receiver_label": label,
            "model": checkpoint["model_name"],
            "train_seed": _checkpoint_train_seed(checkpoint),
            "eval_seed": seed,
            "system": system,
            "num_layers": num_layers,
            "num_rx_ant": 1 if system == "siso" else reference.config.num_rx_ant,
            "num_frames": total_frames,
            "repetitions": repetitions,
        }
        rows.append(
            {
                **common,
                "frame_bler": stats["block_errors"] / total_frames,
                "block_errors": stats["block_errors"],
                "bler_ci95_low": low,
                "bler_ci95_high": high,
                "post_ldpc_ber": stats["info_errors"] / total_info_bits,
                "pre_ldpc_ber": stats["coded_errors"] / total_coded_bits,
                "coded_bce": stats["coded_bce"] / total_coded_bits,
            }
        )
        for layer_index in range(num_layers):
            errors = int(stats["layer_block_errors"][layer_index].item())
            layer_low, layer_high = wilson_interval(errors, total_frames)
            layer_rows.append(
                {
                    **common,
                    "layer": layer_index,
                    "bler": errors / total_frames,
                    "block_errors": errors,
                    "bler_ci95_low": layer_low,
                    "bler_ci95_high": layer_high,
                    "post_ldpc_ber": int(
                        stats["layer_info_errors"][layer_index].item()
                    )
                    / (total_frames * reference.k),
                }
            )
    return rows, layer_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["siso", "su_mimo"], required=True)
    parser.add_argument("--channel_mat", required=True)
    parser.add_argument("--invariant_checkpoint", required=True)
    parser.add_argument("--sensitive_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ebno_list", default="-5,-3,-1,0,1,2,3,4,5,6,7,8")
    parser.add_argument("--coderate", type=float, default=0.5)
    parser.add_argument("--decoder_iterations", type=int, default=20)
    parser.add_argument("--cn_update", default="boxplus-phi")
    parser.add_argument("--phase_mode", choices=["fixed", "narrow", "uniform"], default="uniform")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=777000)
    parser.add_argument("--independent_snr_randomness", action="store_true")
    parser.add_argument("--normalization", choices=["checkpoint", "per_frame", "none"], default="checkpoint")
    parser.add_argument(
        "--mimo_mat_layout",
        choices=[
            "frame_rx_layer_symbol_subcarrier",
            "frame_layer_rx_symbol_subcarrier",
        ],
        default="frame_rx_layer_symbol_subcarrier",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size <= 0 or args.repetitions <= 0 or args.max_frames < 0:
        raise ValueError("Invalid batch/repetition/frame limit.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    system = "siso" if args.system == "siso" else "su_mimo"
    receivers, config_dict, parameter_count = load_receivers(
        system,
        Path(args.invariant_checkpoint),
        Path(args.sensitive_checkpoint),
        device,
    )
    channel = load_channel_mat(
        Path(args.channel_mat), system, args.mimo_mat_layout, args.max_frames
    )
    if system == "siso":
        config = SionnaOFDMConfig(**config_dict)
        expected = (config.num_ofdm_symbols, config.fft_size)
    else:
        from data import SionnaSUMIMOConfig

        config = SionnaSUMIMOConfig(**config_dict)
        expected = (
            config.num_layers,
            config.num_rx_ant,
            config.num_ofdm_symbols,
            config.fft_size,
        )
    if tuple(channel.shape[1:]) != expected:
        raise ValueError(
            f"Channel shape {tuple(channel.shape[1:])} does not match {expected}."
        )
    ldpc = SionnaLDPC5GConfig(
        coderate=args.coderate,
        num_iter=args.decoder_iterations,
        cn_update=args.cn_update,
    )
    reference = (
        Sionna5GLDPCBatchGenerator(
            config,
            ldpc_config=ldpc,
            phase_mode=args.phase_mode,
            seed=args.seed,
            device=device,
        )
        if system == "siso"
        else Sionna5GLDPCSUMIMOBatchGenerator(
            config,
            ldpc_config=ldpc,
            phase_mode=args.phase_mode,
            seed=args.seed,
            device=device,
        )
    )
    checkpoint_normalize = bool(config.normalize_channel)
    normalize = (
        checkpoint_normalize
        if args.normalization == "checkpoint"
        else args.normalization == "per_frame"
    )
    rows = []
    layer_rows = []
    for index, ebno_db in enumerate(parse_float_list(args.ebno_list)):
        eval_seed = args.seed + 100000 * index if args.independent_snr_randomness else args.seed
        snr_rows, snr_layer_rows = evaluate_one_ebno(
            system,
            receivers,
            reference,
            channel,
            ebno_db,
            args.batch_size,
            args.repetitions,
            eval_seed,
            normalize,
        )
        rows.extend(snr_rows)
        layer_rows.extend(snr_layer_rows)
        print(f"Eb/N0 {ebno_db:g} dB")
        for row in snr_rows:
            print(
                f"  {row['receiver_label']:10s} | BLER {row['frame_bler']:.6e} "
                f"| errors {row['block_errors']}/{row['num_frames']}"
            )
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "quadriga_bler_summary.csv", rows)
    write_csv(output_dir / "quadriga_bler_per_layer.csv", layer_rows)
    manifest = {
        **vars(args),
        "system": system,
        "channel_shape": list(channel.shape),
        "parameter_count_per_receiver": parameter_count,
        "resolved_normalize_channel": normalize,
        "phy_config": config_dict,
        "ldpc_config": ldpc.to_dict(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "experiment_config.json").open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
