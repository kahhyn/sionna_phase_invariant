"""Train the existing receivers with a batched Sionna PHY backend."""

import argparse
import csv
import math
from pathlib import Path

import torch

from data import (
    PROFILE_SCHEMA_VERSION,
    SionnaOFDMBatchGenerator,
    SionnaOFDMConfig,
    channel_profile_hash,
    legacy_channel_profile,
    load_channel_profile,
)
from models.factory import MODEL_CHOICES, build_model_from_args
from utils.batching import batch_sizes
from utils.metrics import masked_bce_with_logits, masked_error_count


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


def build_generator(
    args,
    config,
    phase_mode,
    seed,
    device,
    channel_profile,
    snr_min=None,
    snr_max=None,
):
    return SionnaOFDMBatchGenerator(
        config,
        snr_db_min=args.snr_db_min if snr_min is None else snr_min,
        snr_db_max=args.snr_db_max if snr_max is None else snr_max,
        phase_mode=phase_mode,
        seed=seed,
        device=device,
        channel_profile=channel_profile,
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
    parser.add_argument(
        "--train_channel_profile",
        help="JSON profile overriding the legacy TDL channel arguments.",
    )
    parser.add_argument(
        "--val_channel_profile",
        help="Validation JSON profile (defaults to the training profile).",
    )

    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument(
        "--trunk_hidden",
        type=int,
        help=(
            "RealImagCNN trunk width. Defaults to the matched 200k setting "
            "when omitted."
        ),
    )
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
    parser.add_argument(
        "--init_checkpoint",
        help="Initialize model weights from a Sionna checkpoint; optimizer starts fresh.",
    )
    parser.add_argument(
        "--epoch_offset",
        type=int,
        default=0,
        help="Epoch number already completed before this training stage.",
    )
    parser.add_argument(
        "--train_generator_seed",
        type=int,
        help="Override the training data RNG seed without changing model initialization.",
    )
    parser.add_argument(
        "--val_generator_seed",
        type=int,
        help="Override the validation data RNG seed; fixed validation remains reproducible.",
    )
    parser.add_argument("--save_dir", default="runs/sionna_debug")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_train <= 0 or args.num_val <= 0:
        raise ValueError("num_train and num_val must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.epoch_offset < 0:
        raise ValueError("epoch_offset must be non-negative.")

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
    train_profile = (
        load_channel_profile(args.train_channel_profile)
        if args.train_channel_profile
        else legacy_channel_profile(data_config)
    )
    val_profile = (
        load_channel_profile(args.val_channel_profile)
        if args.val_channel_profile
        else train_profile
    )
    train_generator_seed = (
        args.seed if args.train_generator_seed is None else args.train_generator_seed
    )
    val_generator_seed = (
        args.seed + 100000
        if args.val_generator_seed is None
        else args.val_generator_seed
    )
    train_generator = build_generator(
        args,
        data_config,
        args.train_phase_mode,
        train_generator_seed,
        device,
        train_profile,
    )
    val_generator = build_generator(
        args,
        data_config,
        args.val_phase_mode,
        val_generator_seed,
        device,
        val_profile,
    )

    model = build_model_from_args(args, bits_per_symbol=2).to(device)
    init_checkpoint = None
    if args.init_checkpoint:
        init_checkpoint = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False
        )
        if init_checkpoint.get("data_backend") != "sionna":
            raise ValueError("init_checkpoint is not a Sionna checkpoint.")
        if init_checkpoint.get("model_name") != args.model:
            raise ValueError(
                "init_checkpoint model mismatch: "
                f"{init_checkpoint.get('model_name')!r} != {args.model!r}."
            )
        source_config = init_checkpoint.get("sionna_config")
        if source_config is not None and source_config != data_config.to_dict():
            raise ValueError("init_checkpoint Sionna configuration does not match.")
        model.load_state_dict(init_checkpoint["model_state"], strict=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history_path = save_dir / "history.csv"
    history_fields = [
        "epoch",
        "lr",
        "train_loss",
        "train_ber",
        "val_loss",
        "val_ber",
    ]

    def make_checkpoint(epoch, train_loss, train_ber, val_loss, val_ber):
        return {
            "model_name": args.model,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ber": train_ber,
            "val_loss": val_loss,
            "val_ber": val_ber,
            "args": vars(args),
            "sionna_config": data_config.to_dict(),
            "data_backend": "sionna",
            "channel_profile_schema_version": PROFILE_SCHEMA_VERSION,
            "train_channel_profile": train_profile,
            "val_channel_profile": val_profile,
            "channel_profile_hash": channel_profile_hash(train_profile),
            "train_profile_counts": train_generator.channel_profile_counts,
            "val_profile_counts": val_generator.channel_profile_counts,
            "init_checkpoint": args.init_checkpoint,
            "train_generator_seed": train_generator_seed,
            "val_generator_seed": val_generator_seed,
        }

    best_val_loss = math.inf
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(
        f"Data backend: Sionna 2.x | train profile: {train_profile['name']} | "
        f"validation profile: {val_profile['name']}"
    )
    print(
        f"Train phase: {args.train_phase_mode} | "
        f"Validation phase: {args.val_phase_mode}"
    )
    print(
        f"Training data seed: {train_generator_seed} | "
        f"Validation data seed: {val_generator_seed}"
    )
    if init_checkpoint is not None:
        print(
            f"Initialized from: {args.init_checkpoint} | "
            f"fresh AdamW at lr={args.lr:g}"
        )
        initial_val_loss, initial_val_ber = evaluate_sionna(
            model, val_generator, args.num_val, args.batch_size
        )
        best_val_loss = initial_val_loss
        initial_checkpoint = make_checkpoint(
            args.epoch_offset, None, None, initial_val_loss, initial_val_ber
        )
        torch.save(initial_checkpoint, save_dir / "best.pt")
        with history_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=history_fields)
            writer.writeheader()
            writer.writerow(
                {
                    "epoch": args.epoch_offset,
                    "lr": args.lr,
                    "train_loss": "",
                    "train_ber": "",
                    "val_loss": initial_val_loss,
                    "val_ber": initial_val_ber,
                }
            )
        print(
            f"Initial validation | loss {initial_val_loss:.5f} | "
            f"BER {initial_val_ber:.5f}"
        )
    else:
        with history_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=history_fields)
            writer.writeheader()

    for epoch in range(1, args.epochs + 1):
        absolute_epoch = args.epoch_offset + epoch
        final_epoch = args.epoch_offset + args.epochs
        print(f"\nEpoch {absolute_epoch}/{final_epoch}")
        train_generator.reset_profile_sampler(
            train_generator_seed + absolute_epoch * 1009
        )
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
            f"Epoch {absolute_epoch:03d} | train loss {train_loss:.5f} | "
            f"train BER {train_ber:.5f} | val loss {val_loss:.5f} | "
            f"val BER {val_ber:.5f}"
        )
        print(
            f"  train profile batches: {train_generator.channel_profile_counts} | "
            f"validation profile batches: {val_generator.channel_profile_counts}"
        )

        checkpoint = make_checkpoint(
            absolute_epoch, train_loss, train_ber, val_loss, val_ber
        )
        torch.save(checkpoint, save_dir / "last.pt")
        with history_path.open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=history_fields).writerow(
                {
                    "epoch": absolute_epoch,
                    "lr": args.lr,
                    "train_loss": train_loss,
                    "train_ber": train_ber,
                    "val_loss": val_loss,
                    "val_ber": val_ber,
                }
            )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, save_dir / "best.pt")
            print(f"  saved best checkpoint to {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
