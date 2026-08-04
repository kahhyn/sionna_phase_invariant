"""Train the common-phase-invariant receiver on fixed-topology SU-MIMO data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from data import (
    PROFILE_SCHEMA_VERSION,
    SionnaSUMIMOBatchGenerator,
    SionnaSUMIMOConfig,
    channel_profile_hash,
    legacy_channel_profile,
    load_channel_profile,
)
from models import SU_MIMO_MODEL_CHOICES, build_su_mimo_model
from utils.batching import batch_sizes
from utils.checkpoints import load_su_mimo_checkpoint
from utils.metrics import masked_bce_sum, masked_bce_with_logits, masked_error_count


TDL_MIX_REFERENCE_PARAMETERS = 204599
HISTORY_FIELDS = [
    "epoch",
    "lr",
    "train_bce",
    "train_ber",
    "train_bit_errors",
    "train_valid_bits",
    "val_bce",
    "val_ber",
    "val_bit_errors",
    "val_valid_bits",
]
RESUME_RUNTIME_ARGUMENTS = {
    "resume_checkpoint",
    "epochs",
    "save_dir",
    "device",
    "log_interval",
}


def build_data_config(args):
    return SionnaSUMIMOConfig(
        num_ofdm_symbols=args.num_ofdm_symbols,
        fft_size=args.fft_size,
        subcarrier_spacing_hz=args.subcarrier_spacing_hz,
        cyclic_prefix_length=args.cyclic_prefix_length,
        bits_per_symbol=2,
        num_layers=args.num_layers,
        num_rx_ant=args.num_rx_ant,
        total_tx_power=args.total_tx_power,
        dmrs_symbol_indices=tuple(args.dmrs_symbols),
        tdl_model=args.tdl_model,
        delay_spread_s=args.delay_spread_s,
        carrier_frequency_hz=args.carrier_frequency_hz,
        max_doppler_hz=args.max_doppler_hz,
        normalize_channel=not args.no_normalize_channel,
        ls_interpolation_type=args.ls_interpolation_type,
    )


def build_model_config(args):
    return {
        "num_rx_ant": args.num_rx_ant,
        "hidden_complex": args.hidden_complex,
        "zero_real": args.zero_real,
        "hidden_real": args.hidden_real,
        "bits_per_symbol": 2,
        "num_iterations": args.num_iterations,
        "kernel_size": args.kernel_size,
        "zero_gate_hidden": args.zero_gate_hidden,
    }


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
    return SionnaSUMIMOBatchGenerator(
        config,
        snr_db_min=args.snr_db_min if snr_min is None else snr_min,
        snr_db_max=args.snr_db_max if snr_max is None else snr_max,
        phase_mode=phase_mode,
        seed=seed,
        device=device,
        channel_profile=channel_profile,
    )


def _git_revision():
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{revision}-dirty" if dirty else revision
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_history(path, history):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def _aggregate_metrics(logits, batch):
    bce_sum, bce_count = masked_bce_sum(
        logits, batch["bits"], batch["loss_mask"]
    )
    errors, valid_bits = masked_error_count(
        logits, batch["bits"], batch["loss_mask"]
    )
    if int(bce_count.item()) != int(valid_bits.item()):
        raise RuntimeError("BCE and BER masks disagree.")
    return float(bce_sum.item()), int(errors.item()), int(valid_bits.item())


def train_one_epoch(model, generator, optimizer, num_samples, batch_size, log_interval):
    model.train()
    total_bce = 0.0
    total_errors = 0
    total_bits = 0
    for step, current_batch_size in enumerate(
        batch_sizes(num_samples, batch_size), start=1
    ):
        batch = generator.generate_batch(current_batch_size)
        logits = model(
            batch["Y"],
            batch["H_hat"],
            batch["P"],
            batch["N0"],
            batch["layer_mask"],
        )
        loss = masked_bce_with_logits(
            logits, batch["bits"], batch["loss_mask"]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bce_sum, errors, valid_bits = _aggregate_metrics(logits.detach(), batch)
        total_bce += bce_sum
        total_errors += errors
        total_bits += valid_bits
        if log_interval > 0 and step % log_interval == 0:
            print(
                f"  step {step:05d} | BCE {total_bce / total_bits:.6f} | "
                f"BER {total_errors / total_bits:.6f}"
            )
    return total_bce / total_bits, total_errors / total_bits, total_errors, total_bits


@torch.no_grad()
def evaluate(model, generator, num_samples, batch_size):
    model.eval()
    generator.reset()
    total_bce = 0.0
    total_errors = 0
    total_bits = 0
    for current_batch_size in batch_sizes(num_samples, batch_size):
        batch = generator.generate_batch(current_batch_size)
        logits = model(
            batch["Y"],
            batch["H_hat"],
            batch["P"],
            batch["N0"],
            batch["layer_mask"],
        )
        bce_sum, errors, valid_bits = _aggregate_metrics(logits, batch)
        total_bce += bce_sum
        total_errors += errors
        total_bits += valid_bits
    return total_bce / total_bits, total_errors / total_bits, total_errors, total_bits


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="su_mimo_phase_invariant",
        choices=SU_MIMO_MODEL_CHOICES,
    )
    parser.add_argument(
        "--train_phase_mode", default="fixed", choices=["fixed", "narrow", "uniform"]
    )
    parser.add_argument(
        "--val_phase_mode", default="uniform", choices=["fixed", "narrow", "uniform"]
    )
    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_val", type=int, default=2000)
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Total target epoch, including epochs already present when resuming.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--snr_db_min", type=float, default=-5.0)
    parser.add_argument("--snr_db_max", type=float, default=20.0)

    parser.add_argument("--num_ofdm_symbols", type=int, default=14)
    parser.add_argument("--fft_size", type=int, default=72)
    parser.add_argument("--subcarrier_spacing_hz", type=float, default=30e3)
    parser.add_argument("--cyclic_prefix_length", type=int, default=0)
    parser.add_argument("--dmrs_symbols", type=int, nargs="+", default=[2, 11])
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_rx_ant", type=int, default=2)
    parser.add_argument("--total_tx_power", type=float, default=1.0)
    parser.add_argument(
        "--tdl_model", default="A", choices=["A", "B", "C", "D", "E"]
    )
    parser.add_argument("--delay_spread_s", type=float, default=10e-9)
    parser.add_argument("--carrier_frequency_hz", type=float, default=3.5e9)
    parser.add_argument("--max_doppler_hz", type=float, default=200.0)
    parser.add_argument(
        "--ls_interpolation_type",
        default="lin",
        choices=["nn", "lin", "lin_time_avg"],
    )
    parser.add_argument("--no_normalize_channel", action="store_true")
    parser.add_argument(
        "--train_channel_profile",
        help="Existing channel-profile JSON; legacy TDL arguments are used if omitted.",
    )
    parser.add_argument(
        "--val_channel_profile",
        help="Validation profile JSON; defaults to the training profile.",
    )

    parser.add_argument("--hidden_complex", type=int, default=32)
    parser.add_argument("--zero_real", type=int, default=22)
    parser.add_argument("--hidden_real", type=int, default=66)
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--zero_gate_hidden", type=int, default=16)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_generator_seed", type=int)
    parser.add_argument("--val_generator_seed", type=int)
    parser.add_argument(
        "--resume_checkpoint",
        help=(
            "Resume model, optimizer, data, and training settings. --epochs is "
            "the new total target epoch."
        ),
    )
    parser.add_argument(
        "--save_dir",
        help="Defaults to runs/su_mimo_debug, or the checkpoint directory on resume.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _restore_saved_arguments(args, checkpoint):
    saved_args = checkpoint.get("args", {})
    for key, value in saved_args.items():
        if key not in RESUME_RUNTIME_ARGUMENTS and hasattr(args, key):
            setattr(args, key, value)


def _validate_args(args, start_epoch):
    if args.num_train <= 0 or args.num_val <= 0:
        raise ValueError("num_train and num_val must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.epochs <= start_epoch:
        raise ValueError(
            f"--epochs must exceed the resumed epoch ({start_epoch})."
        )
    if args.lr <= 0.0:
        raise ValueError("lr must be positive.")


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in PyTorch.")

    resumed_checkpoint = None
    start_epoch = 0
    if args.resume_checkpoint:
        model, data_config, resumed_checkpoint = load_su_mimo_checkpoint(
            args.resume_checkpoint, device
        )
        _restore_saved_arguments(args, resumed_checkpoint)
        start_epoch = int(resumed_checkpoint["epoch"])
        model_config = resumed_checkpoint["model_config"]
    else:
        data_config = build_data_config(args)
        model_config = build_model_config(args)
        model = build_su_mimo_model(args.model, model_config).to(device)

    _validate_args(args, start_epoch)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if resumed_checkpoint is not None:
        if data_config.to_dict() != resumed_checkpoint["sionna_su_mimo_config"]:
            raise RuntimeError("Resumed SU-MIMO data configuration is inconsistent.")
        if model_config != resumed_checkpoint["model_config"]:
            raise RuntimeError("Resumed SU-MIMO model configuration is inconsistent.")

    train_seed = (
        args.seed if args.train_generator_seed is None else args.train_generator_seed
    )
    val_seed = (
        args.seed + 100000
        if args.val_generator_seed is None
        else args.val_generator_seed
    )
    if resumed_checkpoint is None:
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
    else:
        saved_train_profile = resumed_checkpoint.get("train_channel_profile")
        saved_val_profile = resumed_checkpoint.get("val_channel_profile")
        train_profile = (
            legacy_channel_profile(data_config)
            if saved_train_profile is None
            else load_channel_profile(saved_train_profile)
        )
        val_profile = (
            train_profile
            if saved_val_profile is None
            else load_channel_profile(saved_val_profile)
        )
    train_generator = build_generator(
        args,
        data_config,
        args.train_phase_mode,
        train_seed,
        device,
        train_profile,
    )
    val_generator = build_generator(
        args,
        data_config,
        args.val_phase_mode,
        val_seed,
        device,
        val_profile,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if resumed_checkpoint is not None:
        optimizer.load_state_dict(resumed_checkpoint["optimizer_state"])

    if args.save_dir:
        save_dir = Path(args.save_dir)
    elif args.resume_checkpoint:
        save_dir = Path(args.resume_checkpoint).resolve().parent
    else:
        save_dir = Path("runs/su_mimo_debug")
    save_dir.mkdir(parents=True, exist_ok=True)
    args.save_dir = str(save_dir)

    history = [] if resumed_checkpoint is None else list(
        resumed_checkpoint.get("history", [])
    )
    if resumed_checkpoint is not None and not history:
        history.append(
            {
                "epoch": start_epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_bce": resumed_checkpoint.get("train_bce", ""),
                "train_ber": resumed_checkpoint.get("train_ber", ""),
                "train_bit_errors": resumed_checkpoint.get("train_bit_errors", ""),
                "train_valid_bits": resumed_checkpoint.get("train_valid_bits", ""),
                "val_bce": resumed_checkpoint["val_bce"],
                "val_ber": resumed_checkpoint["val_ber"],
                "val_bit_errors": resumed_checkpoint.get("val_bit_errors", ""),
                "val_valid_bits": resumed_checkpoint.get("val_valid_bits", ""),
            }
        )
    _write_history(save_dir / "history.csv", history)

    best_val_bce = (
        math.inf
        if resumed_checkpoint is None
        else float(resumed_checkpoint.get("best_val_bce", math.inf))
    )
    if resumed_checkpoint is not None:
        source_best = Path(args.resume_checkpoint).resolve().parent / "best.pt"
        target_best = (save_dir / "best.pt").resolve()
        if source_best.exists() and source_best.resolve() != target_best:
            shutil.copy2(source_best, target_best)

    created_at = (
        datetime.now(timezone.utc).isoformat()
        if resumed_checkpoint is None
        else resumed_checkpoint.get("created_at_utc")
    )
    command = shlex.join([sys.executable, *sys.argv])
    resolved = {
        "model_name": args.model,
        "model_config": model_config,
        "data_backend": "sionna_su_mimo",
        "sionna_su_mimo_config": data_config.to_dict(),
        "args": vars(args),
        "selection_rule": "minimum source-validation BCE",
        "channel_profile_schema_version": PROFILE_SCHEMA_VERSION,
        "train_channel_profile": train_profile,
        "val_channel_profile": val_profile,
        "channel_profile_hash": channel_profile_hash(train_profile),
        "code_revision": _git_revision(),
        "created_at_utc": created_at,
        "last_command": command,
        "torch_version": str(torch.__version__),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    with (save_dir / "resolved_config.json").open("w") as stream:
        json.dump(resolved, stream, indent=2, sort_keys=True)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device} | model: {args.model} | parameters: {parameter_count}")
    print(
        f"tdl_mix reference: {TDL_MIX_REFERENCE_PARAMETERS} | "
        f"parameter delta: {parameter_count - TDL_MIX_REFERENCE_PARAMETERS:+d}"
    )
    print(
        f"Topology: 1 user, {data_config.num_layers} layers, "
        f"{data_config.num_rx_ant} Rx | total Tx power {data_config.total_tx_power:g}"
    )
    print(
        f"Train phase: {args.train_phase_mode} | val phase: {args.val_phase_mode} | "
        f"train seed: {train_seed} | val seed: {val_seed}"
    )
    print(
        f"Train profile: {train_profile['name']} | "
        f"validation profile: {val_profile['name']}"
    )
    if resumed_checkpoint is not None:
        print(f"Resuming {args.resume_checkpoint} from epoch {start_epoch}")

    def make_checkpoint(epoch, row):
        return {
            **resolved,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "train_bce": row["train_bce"],
            "train_ber": row["train_ber"],
            "train_bit_errors": row["train_bit_errors"],
            "train_valid_bits": row["train_valid_bits"],
            "val_bce": row["val_bce"],
            "val_ber": row["val_ber"],
            "val_bit_errors": row["val_bit_errors"],
            "val_valid_bits": row["val_valid_bits"],
            "best_val_bce": best_val_bce,
            "history": history,
            "train_generator_seed": train_seed,
            "val_generator_seed": val_seed,
            "train_profile_counts": train_generator.channel_profile_counts,
            "val_profile_counts": val_generator.channel_profile_counts,
            "resumed_from": args.resume_checkpoint,
        }

    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_generator.reset(train_seed + epoch * 1009)
        train_bce, train_ber, train_errors, train_bits = train_one_epoch(
            model,
            train_generator,
            optimizer,
            args.num_train,
            args.batch_size,
            args.log_interval,
        )
        val_bce, val_ber, val_errors, val_bits = evaluate(
            model, val_generator, args.num_val, args.batch_size
        )
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_bce": train_bce,
            "train_ber": train_ber,
            "train_bit_errors": train_errors,
            "train_valid_bits": train_bits,
            "val_bce": val_bce,
            "val_ber": val_ber,
            "val_bit_errors": val_errors,
            "val_valid_bits": val_bits,
        }
        history.append(row)
        _write_history(save_dir / "history.csv", history)
        print(
            f"Epoch {epoch:03d} | train BCE {train_bce:.6f} | "
            f"train BER {train_ber:.6f} ({train_errors}/{train_bits}) | "
            f"val BCE {val_bce:.6f} | val BER {val_ber:.6f} "
            f"({val_errors}/{val_bits})"
        )
        print(
            f"  train profile batches: {train_generator.channel_profile_counts} | "
            f"validation profile batches: {val_generator.channel_profile_counts}"
        )

        improved = val_bce < best_val_bce
        if improved:
            best_val_bce = val_bce
        checkpoint = make_checkpoint(epoch, row)
        checkpoint["best_val_bce"] = best_val_bce
        torch.save(checkpoint, save_dir / "last.pt")
        if improved:
            torch.save(checkpoint, save_dir / "best.pt")
            print(f"  saved best checkpoint to {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
