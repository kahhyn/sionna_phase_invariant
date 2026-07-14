"""BER evaluation for checkpoints trained with the Sionna backend."""

import argparse
import csv
from pathlib import Path

import torch

from data import (
    SionnaOFDMBatchGenerator,
    SionnaOFDMConfig,
    legacy_channel_profile,
    load_channel_profile,
    profile_backend_label,
    profile_component,
)
from utils.batching import batch_sizes
from utils.checkpoints import load_receiver_checkpoint
from utils.metrics import masked_bce_sum, masked_error_count


def parse_snr_list(text):
    return [float(value.strip()) for value in text.split(",") if value.strip()]


@torch.no_grad()
def evaluate_snr(
    model,
    config,
    channel_profile,
    snr_db,
    phase_mode,
    num_samples,
    batch_size,
    seed,
    device,
):
    generator = SionnaOFDMBatchGenerator(
        config,
        snr_db_min=snr_db,
        snr_db_max=snr_db,
        phase_mode=phase_mode,
        seed=seed,
        device=device,
        channel_profile=channel_profile,
    )
    model.eval()
    total_bce = 0.0
    total_errors = 0
    total_bits = 0
    for current_batch_size in batch_sizes(num_samples, batch_size):
        batch = generator.generate_batch(current_batch_size)
        logits = model(batch["Y"], batch["H_hat"], batch["P"], batch["N0"])
        bce_sum, bce_count = masked_bce_sum(
            logits, batch["bits"], batch["loss_mask"]
        )
        errors, valid_bits = masked_error_count(
            logits, batch["bits"], batch["loss_mask"]
        )
        total_bce += float(bce_sum.item())
        total_errors += int(errors.item())
        total_bits += int(valid_bits.item())
        if int(bce_count.item()) != int(valid_bits.item()):
            raise RuntimeError("BCE and BER masks disagree.")
    return total_bce / total_bits, total_errors / total_bits, total_errors, total_bits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase_mode", default="uniform", choices=["fixed", "narrow", "uniform"])
    parser.add_argument("--snr_list", default="-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20")
    parser.add_argument("--num_samples", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=777000)
    parser.add_argument(
        "--common_random_numbers",
        action="store_true",
        help=(
            "Reuse the same bits, channels, unit-variance noise, and phase "
            "samples at every SNR. Only the noise scale changes."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_csv", default="sionna_ber_results.csv")
    parser.add_argument(
        "--eval_channel_profile",
        help="JSON channel profile overriding the checkpoint training channel.",
    )
    parser.add_argument(
        "--eval_component_id",
        help="Evaluate one fixed component from --eval_channel_profile.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model, checkpoint = load_receiver_checkpoint(
        args.checkpoint,
        device,
        bits_per_symbol=2,
        required_data_backend="sionna",
    )
    train_args = checkpoint["args"]
    config = SionnaOFDMConfig(**checkpoint["sionna_config"])
    model_name = checkpoint["model_name"]
    train_seed = int(train_args.get("seed", -1))
    train_profile = checkpoint.get("train_channel_profile")
    if train_profile is None:
        train_profile = legacy_channel_profile(config)
    else:
        train_profile = load_channel_profile(train_profile)
    if args.eval_channel_profile:
        eval_profile = load_channel_profile(
            args.eval_channel_profile, component_id=args.eval_component_id
        )
    elif args.eval_component_id:
        raise ValueError("--eval_component_id requires --eval_channel_profile")
    else:
        eval_profile = train_profile
    fixed_component = profile_component(eval_profile)
    backend = profile_backend_label(eval_profile)
    scenario = fixed_component.get("backend", "mixed") if fixed_component else "mixed"
    tdl_model = fixed_component.get("tdl_model", "") if fixed_component else ""
    delay_ns = (
        fixed_component.get("delay_spread_ns", "") if fixed_component else ""
    )

    print(
        f"Train profile: {train_profile['name']} | "
        f"test profile: {eval_profile['name']} | backend: {backend}"
    )

    rows = []
    for index, snr_db in enumerate(parse_snr_list(args.snr_list)):
        eval_seed = args.seed if args.common_random_numbers else args.seed + index * 1000
        bce, ber, errors, valid_bits = evaluate_snr(
            model,
            config,
            eval_profile,
            snr_db,
            args.phase_mode,
            args.num_samples,
            args.batch_size,
            eval_seed,
            device,
        )
        print(
            f"SNR {snr_db:5.1f} dB | BCE {bce:.6f} | BER {ber:.6e} | "
            f"errors {errors}/{valid_bits}"
        )
        rows.append(
            {
                "snr_db": snr_db,
                "bce": bce,
                "ber": ber,
                "bit_errors": errors,
                "valid_bits": valid_bits,
                "model": model_name,
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "common_random_numbers": int(args.common_random_numbers),
                "train_profile": train_profile["name"],
                "test_profile": eval_profile["name"],
                "backend": backend,
                "scenario": scenario,
                "tdl_model": tdl_model,
                "delay_ns": delay_ns,
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "snr_db",
                "bce",
                "ber",
                "bit_errors",
                "valid_bits",
                "model",
                "train_seed",
                "eval_seed",
                "common_random_numbers",
                "train_profile",
                "test_profile",
                "backend",
                "scenario",
                "tdl_model",
                "delay_ns",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV to {out_path}")


if __name__ == "__main__":
    main()
