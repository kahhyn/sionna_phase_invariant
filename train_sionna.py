"""Train the existing receivers with a batched Sionna PHY backend."""

import argparse
import math
from pathlib import Path

import torch

from data import SionnaOFDMBatchGenerator, SionnaOFDMConfig
from models.factory import MODEL_CHOICES, build_model_from_args
from utils.metrics import masked_bce_with_logits, masked_error_count


def batch_sizes(num_samples, batch_size):
    for start in range(0, num_samples, batch_size):
        yield min(batch_size, num_samples - start)


def build_data_config(args):
    return SionnaOFDMConfig(
        num_ofdm_symbols=args.num_ofdm_symbols,
        fft_size=args.fft_size,
        subcarrier_spacing_hz=args.subcarrier_spacing_hz,
        cyclic_prefix_length=args.cyclic_prefix_length,
        dmrs_symbol_indices=tuple(args.dmrs_symbols),
        dmrs_freq_spacing=args.dmrs_freq_spacing,
        dmrs_freq_offset=args.dmrs_freq_offset,
        tdl_model=args.tdl_model,
        delay_spread_s=args.delay_spread_s,
        carrier_frequency_hz=args.carrier_frequency_hz,
        max_doppler_hz=args.max_doppler_hz,
        normalize_channel=not args.no_normalize_channel,
        ls_interpolation_type=args.ls_interpolation_type,
    )


def build_generator(args, config, phase_mode, seed, device, snr_min=None, snr_max=None):
    return SionnaOFDMBatchGenerator(
        config,
        snr_db_min=args.snr_db_min if snr_min is None else snr_min,
        snr_db_max=args.snr_db_max if snr_max is None else snr_max,
        phase_mode=phase_mode,
        seed=seed,
        device=device,
    )


def train_one_epoch_sionna(model, generator, optimizer, num_samples, batch_size, log_interval):
    model.train()
    loss_weighted_sum = 0.0
    total_errors = 0
    total_bits = 0

    for step, current_batch_size in enumerate(batch_sizes(num_samples, batch_size), start=1):
        batch = generator.generate_batch(current_batch_size)
        logits = model(batch["Y"], batch["H_hat"], batch["P"], batch["N0"])
        loss = masked_bce_with_logits(logits, batch["bits"], batch["loss_mask"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        errors, valid_bits = masked_error_count(
            logits.detach(), batch["bits"], batch["loss_mask"]
        )
        count = int(valid_bits.item())
        loss_weighted_sum += loss.item() * count
        total_errors += int(errors.item())
        total_bits += count

        if log_interval > 0 and step % log_interval == 0:
            print(
                f"  step {step:05d} | "
                f"loss {loss_weighted_sum / total_bits:.5f} | "
                f"BER {total_errors / total_bits:.5f}"
            )

    return loss_weighted_sum / total_bits, total_errors / total_bits


@torch.no_grad()
def evaluate_sionna(model, generator, num_samples, batch_size):
    model.eval()
    generator.reset()
    loss_weighted_sum = 0.0
    total_errors = 0
    total_bits = 0

    for current_batch_size in batch_sizes(num_samples, batch_size):
        batch = generator.generate_batch(current_batch_size)
        logits = model(batch["Y"], batch["H_hat"], batch["P"], batch["N0"])
        loss = masked_bce_with_logits(logits, batch["bits"], batch["loss_mask"])
        errors, valid_bits = masked_error_count(
            logits, batch["bits"], batch["loss_mask"]
        )
        count = int(valid_bits.item())
        loss_weighted_sum += loss.item() * count
        total_errors += int(errors.item())
        total_bits += count

    return loss_weighted_sum / total_bits, total_errors / total_bits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="single_branch",
        choices=MODEL_CHOICES,
    )
    parser.add_argument("--train_phase_mode", default="fixed", choices=["fixed", "narrow", "uniform"])
    parser.add_argument("--val_phase_mode", default="uniform", choices=["fixed", "narrow", "uniform"])
    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_val", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--snr_db_min", type=float, default=-5.0)
    parser.add_argument("--snr_db_max", type=float, default=20.0)

    parser.add_argument("--num_ofdm_symbols", type=int, default=14)
    parser.add_argument("--fft_size", type=int, default=72)
    parser.add_argument("--subcarrier_spacing_hz", type=float, default=30e3)
    parser.add_argument("--cyclic_prefix_length", type=int, default=0)
    parser.add_argument("--dmrs_symbols", type=int, nargs="+", default=[2, 11])
    parser.add_argument("--dmrs_freq_spacing", type=int, default=1)
    parser.add_argument("--dmrs_freq_offset", type=int, default=0)
    parser.add_argument("--tdl_model", default="A", choices=["A", "B", "C", "D", "E", "A30", "B100", "C300"])
    parser.add_argument("--delay_spread_s", type=float, default=10e-9)
    parser.add_argument("--carrier_frequency_hz", type=float, default=3.5e9)
    parser.add_argument("--max_doppler_hz", type=float, default=200.0)
    parser.add_argument("--ls_interpolation_type", default="lin", choices=["nn", "lin", "lin_time_avg"])
    parser.add_argument("--no_normalize_channel", action="store_true")

    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--hidden_complex", type=int, default=16)
    parser.add_argument("--zero_complex", type=int, default=16)
    parser.add_argument("--branch_layers", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--gate_type", default="swiglu", choices=["sigmoid", "swiglu"])
    parser.add_argument("--single_readout_mode", default="low_rank", choices=["low_rank", "full"])
    parser.add_argument("--zero_gate_hidden", type=int, default=16)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", default="runs/sionna_debug")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_train <= 0 or args.num_val <= 0:
        raise ValueError("num_train and num_val must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in PyTorch.")
    device = requested_device
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    data_config = build_data_config(args)
    train_generator = build_generator(
        args, data_config, args.train_phase_mode, args.seed, device
    )
    val_generator = build_generator(
        args, data_config, args.val_phase_mode, args.seed + 100000, device
    )

    model = build_model_from_args(args, bits_per_symbol=2).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_val_loss = math.inf
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print("Data backend: Sionna 2.x / TDL-" + args.tdl_model)
    print(
        f"Train phase: {args.train_phase_mode} | "
        f"Validation phase: {args.val_phase_mode}"
    )

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_ber = train_one_epoch_sionna(
            model,
            train_generator,
            optimizer,
            args.num_train,
            args.batch_size,
            args.log_interval,
        )
        val_loss, val_ber = evaluate_sionna(
            model, val_generator, args.num_val, args.batch_size
        )
        print(
            f"Epoch {epoch:03d} | train loss {train_loss:.5f} | "
            f"train BER {train_ber:.5f} | val loss {val_loss:.5f} | "
            f"val BER {val_ber:.5f}"
        )

        checkpoint = {
            "model_name": args.model,
            "model_state": model.state_dict(),
            "args": vars(args),
            "sionna_config": data_config.to_dict(),
            "data_backend": "sionna",
        }
        torch.save(checkpoint, save_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, save_dir / "best.pt")
            print(f"  saved best checkpoint to {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
