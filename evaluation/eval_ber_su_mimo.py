"""BER evaluation for SU-MIMO checkpoints, including per-layer results."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from data import SionnaSUMIMOBatchGenerator
from utils.batching import batch_sizes
from utils.checkpoints import load_su_mimo_checkpoint
from utils.metrics import masked_bce_sum, masked_error_count


def parse_snr_list(text):
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("--snr_list must contain at least one value.")
    return values


def wilson_interval(errors, trials, z=1.959963984540054):
    """Return a two-sided Wilson binomial confidence interval."""
    if trials <= 0:
        raise ValueError("trials must be positive.")
    rate = errors / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


@torch.no_grad()
def evaluate_snr(model, config, snr_db, phase_mode, num_samples, batch_size, seed, device):
    generator = SionnaSUMIMOBatchGenerator(
        config,
        snr_db_min=snr_db,
        snr_db_max=snr_db,
        phase_mode=phase_mode,
        seed=seed,
        device=device,
    )
    model.eval()
    total_bce = 0.0
    total_errors = 0
    total_bits = 0
    layer_bce = torch.zeros(config.num_layers, dtype=torch.float64)
    layer_errors = torch.zeros(config.num_layers, dtype=torch.int64)
    layer_bits = torch.zeros(config.num_layers, dtype=torch.int64)

    for current_batch_size in batch_sizes(num_samples, batch_size):
        batch = generator.generate_batch(current_batch_size)
        logits = model(
            batch["Y"],
            batch["H_hat"],
            batch["P"],
            batch["N0"],
            batch["layer_mask"],
        )
        bce_sum, bce_count = masked_bce_sum(
            logits, batch["bits"], batch["loss_mask"]
        )
        errors, valid_bits = masked_error_count(
            logits, batch["bits"], batch["loss_mask"]
        )
        if int(bce_count.item()) != int(valid_bits.item()):
            raise RuntimeError("BCE and BER masks disagree.")
        total_bce += float(bce_sum.item())
        total_errors += int(errors.item())
        total_bits += int(valid_bits.item())

        valid = batch["loss_mask"].bool().expand_as(batch["bits"])
        element_bce = F.binary_cross_entropy_with_logits(
            logits, batch["bits"], reduction="none"
        )
        element_errors = ((logits > 0) != batch["bits"].bool()) & valid
        reduce_dims = (0, 2, 3, 4)
        layer_bce += (element_bce * valid).sum(dim=reduce_dims).cpu().double()
        layer_errors += element_errors.sum(dim=reduce_dims).cpu()
        layer_bits += valid.sum(dim=reduce_dims).cpu()

    per_layer = []
    for layer_index in range(config.num_layers):
        bits = int(layer_bits[layer_index].item())
        errors = int(layer_errors[layer_index].item())
        per_layer.append(
            {
                "layer": layer_index,
                "bce": float(layer_bce[layer_index].item()) / bits,
                "ber": errors / bits,
                "bit_errors": errors,
                "valid_bits": bits,
            }
        )
    return {
        "bce": total_bce / total_bits,
        "ber": total_errors / total_bits,
        "bit_errors": total_errors,
        "valid_bits": total_bits,
        "per_layer": per_layer,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--phase_mode", default="uniform", choices=["fixed", "narrow", "uniform"]
    )
    parser.add_argument(
        "--snr_list", default="-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20"
    )
    parser.add_argument("--num_samples", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=777000)
    parser.add_argument(
        "--common_random_numbers",
        action="store_true",
        help=(
            "Reuse bits, channel, unit-variance noise, and phase samples at each SNR; "
            "only the noise scale changes."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_csv", default="su_mimo_ber_results.csv")
    parser.add_argument(
        "--out_layer_csv",
        help="Defaults to <out_csv stem>_per_layer.csv.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("num_samples and batch_size must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in PyTorch.")
    model, config, checkpoint = load_su_mimo_checkpoint(args.checkpoint, device)
    snr_values = parse_snr_list(args.snr_list)

    rows = []
    layer_rows = []
    for index, snr_db in enumerate(snr_values):
        eval_seed = args.seed if args.common_random_numbers else args.seed + index * 1000
        result = evaluate_snr(
            model,
            config,
            snr_db,
            args.phase_mode,
            args.num_samples,
            args.batch_size,
            eval_seed,
            device,
        )
        ci_low, ci_high = wilson_interval(
            result["bit_errors"], result["valid_bits"]
        )
        print(
            f"SNR {snr_db:5.1f} dB | BCE {result['bce']:.6f} | "
            f"BER {result['ber']:.6e} | errors "
            f"{result['bit_errors']}/{result['valid_bits']} | "
            f"95% CI [{ci_low:.6e}, {ci_high:.6e}]"
        )
        common = {
            "snr_db": snr_db,
            "phase_mode": args.phase_mode,
            "eval_seed": eval_seed,
            "common_random_numbers": int(args.common_random_numbers),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_epoch": checkpoint["epoch"],
            "train_seed": checkpoint["args"]["seed"],
            "num_layers": config.num_layers,
            "num_rx_ant": config.num_rx_ant,
            "total_tx_power": config.total_tx_power,
            "tdl_model": config.tdl_model,
        }
        rows.append(
            {
                **common,
                "bce": result["bce"],
                "ber": result["ber"],
                "bit_errors": result["bit_errors"],
                "valid_bits": result["valid_bits"],
                "ber_ci95_low": ci_low,
                "ber_ci95_high": ci_high,
            }
        )
        for layer_result in result["per_layer"]:
            layer_ci_low, layer_ci_high = wilson_interval(
                layer_result["bit_errors"], layer_result["valid_bits"]
            )
            layer_rows.append(
                {
                    **common,
                    **layer_result,
                    "ber_ci95_low": layer_ci_low,
                    "ber_ci95_high": layer_ci_high,
                }
            )

    aggregate_fields = [
        "snr_db",
        "bce",
        "ber",
        "bit_errors",
        "valid_bits",
        "ber_ci95_low",
        "ber_ci95_high",
        "phase_mode",
        "eval_seed",
        "common_random_numbers",
        "checkpoint",
        "checkpoint_epoch",
        "train_seed",
        "num_layers",
        "num_rx_ant",
        "total_tx_power",
        "tdl_model",
    ]
    layer_fields = [
        "snr_db",
        "layer",
        "bce",
        "ber",
        "bit_errors",
        "valid_bits",
        "ber_ci95_low",
        "ber_ci95_high",
        "phase_mode",
        "eval_seed",
        "common_random_numbers",
        "checkpoint",
        "checkpoint_epoch",
        "train_seed",
        "num_layers",
        "num_rx_ant",
        "total_tx_power",
        "tdl_model",
    ]
    out_path = Path(args.out_csv)
    layer_path = (
        Path(args.out_layer_csv)
        if args.out_layer_csv
        else out_path.with_name(f"{out_path.stem}_per_layer{out_path.suffix}")
    )
    for path, fieldnames, data in (
        (out_path, aggregate_fields, rows),
        (layer_path, layer_fields, layer_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    print(f"Saved aggregate CSV to {out_path}")
    print(f"Saved per-layer CSV to {layer_path}")


if __name__ == "__main__":
    main()
