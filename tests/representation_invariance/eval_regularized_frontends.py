"""Evaluate trained identity/MF/T_gamma/LMMSE CNN checkpoints on dense SNR grids.

Examples::

    # Default: -5, -4, ..., 20 dB
    python tests/representation_invariance/eval_regularized_frontends.py

    # Half-dB grid
    python tests/representation_invariance/eval_regularized_frontends.py \
        --snr_min -5 --snr_max 20 --snr_step 0.5

    # Explicit points (use '=' when the first value is negative)
    python tests/representation_invariance/eval_regularized_frontends.py \
        --snr_list=-5,-3,-1,0,1,2,3,5,7,9,11,13,15,17,19,20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data import legacy_channel_profile
from tests.representation_invariance.train_eval_regularized_frontends import (
    BASE_DATA_CONFIG,
    FRONTENDS,
    build_model,
    make_generator,
)
from utils.metrics import masked_bce_with_logits, masked_error_count


def parse_snr_points(args: argparse.Namespace) -> list[float]:
    if args.snr_list:
        values = [float(item.strip()) for item in args.snr_list.split(",") if item.strip()]
    else:
        if args.snr_step <= 0:
            raise ValueError("--snr_step must be positive.")
        values = torch.arange(
            args.snr_min,
            args.snr_max + 0.5 * args.snr_step,
            args.snr_step,
            dtype=torch.float64,
        ).tolist()
    if not values:
        raise ValueError("The SNR point list is empty.")
    # 保序去重，避免同一 SNR 被无意重复评估。
    return list(dict.fromkeys(values))


@torch.no_grad()
def evaluate_point(
    model,
    generator,
    min_frames: int,
    max_frames: int,
    target_bit_errors: int,
    batch_size: int,
    reset_seed: int,
) -> dict:
    """Evaluate one SNR, optionally continuing until enough bit errors occur."""
    model.eval()
    generator.reset(reset_seed)
    total_bce = 0.0
    total_errors = 0
    total_bits = 0
    total_frames = 0
    gamma_sum = 0.0
    noise_gain_sum = 0.0

    while total_frames < max_frames:
        enough_frames = total_frames >= min_frames
        enough_errors = target_bit_errors <= 0 or total_errors >= target_bit_errors
        if enough_frames and enough_errors:
            break

        current_bs = min(batch_size, max_frames - total_frames)
        batch = generator.generate_batch(current_bs)
        logits = model(
            batch["Y"], batch["H_hat"], batch["P"], batch["N0"], batch["layer_mask"]
        )
        bce = masked_bce_with_logits(logits, batch["bits"], batch["loss_mask"])
        errors, valid_bits = masked_error_count(logits, batch["bits"], batch["loss_mask"])
        valid = int(valid_bits.item())

        total_bce += float(bce.item()) * valid
        total_errors += int(errors.item())
        total_bits += valid
        gamma_sum += generator.last_stats["mean_gamma"] * current_bs
        noise_gain_sum += generator.last_stats["mean_noise_gain"] * current_bs
        total_frames += current_bs

    return {
        "bce": total_bce / max(total_bits, 1),
        "ber": total_errors / max(total_bits, 1),
        "bit_errors": total_errors,
        "valid_bits": total_bits,
        "num_frames": total_frames,
        "mean_gamma": gamma_sum / max(total_frames, 1),
        "mean_noise_gain": noise_gain_sum / max(total_frames, 1),
        "hit_max_frames": int(
            total_frames >= max_frames
            and target_bit_errors > 0
            and total_errors < target_bit_errors
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense-SNR checkpoint evaluation")
    parser.add_argument("--frontends", nargs="+", choices=FRONTENDS, default=list(FRONTENDS))
    parser.add_argument(
        "--checkpoint_dir",
        default="tests/representation_invariance/runs_regularized_frontends",
    )
    parser.add_argument("--checkpoint_seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: <checkpoint_dir>/eval_dense",
    )
    parser.add_argument("--snr_list", default=None, help="Comma-separated custom SNRs.")
    parser.add_argument("--snr_min", type=float, default=-5.0)
    parser.add_argument("--snr_max", type=float, default=20.0)
    parser.add_argument("--snr_step", type=float, default=1.0)
    parser.add_argument("--num_test", type=int, default=4096, help="Minimum frames per SNR.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--target_bit_errors", type=int, default=0,
        help="If >0, keep evaluating beyond num_test until this count or max_test.",
    )
    parser.add_argument("--max_test", type=int, default=100000)
    parser.add_argument("--eval_seed", type=int, default=777042)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow_missing", action="store_true",
        help="Skip a frontend when its checkpoint does not exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_test <= 0 or args.max_test < args.num_test or args.batch_size <= 0:
        raise ValueError("Require num_test>0, max_test>=num_test, and batch_size>0.")
    snr_points = parse_snr_points(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        device = torch.device("cpu")

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "eval_dense"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"SNR points ({len(snr_points)}): {snr_points}")

    checkpoint_paths = {
        frontend: checkpoint_dir / f"model_{frontend}_seed{args.checkpoint_seed}.pt"
        for frontend in args.frontends
    }
    missing = [str(path) for path in checkpoint_paths.values() if not path.exists()]
    if missing and not args.allow_missing:
        formatted = "\n  ".join(missing)
        raise FileNotFoundError(
            "Missing checkpoint(s):\n  " + formatted +
            "\nTrain them first, or evaluate a subset with --frontends."
        )

    all_results: dict[str, list[dict]] = {}

    for frontend in args.frontends:
        checkpoint_path = checkpoint_paths[frontend]
        if not checkpoint_path.exists():
            if args.allow_missing:
                print(f"[skip] missing checkpoint: {checkpoint_path}")
                continue

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_frontend = checkpoint.get("frontend", frontend)
        if checkpoint_frontend != frontend:
            raise ValueError(
                f"Checkpoint {checkpoint_path} says frontend={checkpoint_frontend}, "
                f"expected {frontend}."
            )
        gamma_scale = float(checkpoint.get("gamma_scale", 1.0))
        n0_mode = checkpoint.get("n0_mode", "original")
        interpolation = checkpoint.get("ls_interpolation_type", "nn")
        data_config = replace(BASE_DATA_CONFIG, ls_interpolation_type=interpolation)
        profile = legacy_channel_profile(data_config)

        torch.manual_seed(args.checkpoint_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.checkpoint_seed)
        model = build_model(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        print(
            f"\n[{frontend}] checkpoint={checkpoint_path} | "
            f"gamma_scale={gamma_scale:g} | N0={n0_mode} | interpolation={interpolation}"
        )
        rows = []
        for snr_index, snr in enumerate(snr_points):
            # 所有前端在同一 SNR 使用同一随机种子，保持公共随机样本。
            point_seed = args.eval_seed + snr_index * 1000
            generator = make_generator(
                data_config, profile, frontend, snr, snr, "uniform", point_seed,
                device, gamma_scale, n0_mode,
            )
            stats = evaluate_point(
                model, generator, args.num_test, args.max_test,
                args.target_bit_errors, args.batch_size, point_seed,
            )
            row = {
                "frontend": frontend,
                "snr_db": snr,
                **stats,
                "gamma_scale": gamma_scale,
                "n0_mode": n0_mode,
                "ls_interpolation_type": interpolation,
                "eval_seed": point_seed,
            }
            rows.append(row)
            max_tag = " [max frames]" if stats["hit_max_frames"] else ""
            print(
                f"  SNR {snr:6.2f} dB | BER {stats['ber']:.6e} "
                f"({stats['bit_errors']}/{stats['valid_bits']}, "
                f"frames={stats['num_frames']}){max_tag}"
            )
        all_results[frontend] = rows

    if not all_results:
        raise RuntimeError("No checkpoint was evaluated.")

    fieldnames = [
        "frontend", "snr_db", "bce", "ber", "bit_errors", "valid_bits",
        "num_frames", "mean_gamma", "mean_noise_gain", "hit_max_frames",
        "gamma_scale", "n0_mode", "ls_interpolation_type", "eval_seed",
    ]
    csv_path = output_dir / "results_dense_snr.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rows in all_results.values():
            writer.writerows(rows)

    markers = {"identity": "o", "mf": "^", "tgamma": "v", "lmmse": "s"}
    plt.figure(figsize=(10, 6))
    for frontend, rows in all_results.items():
        plt.semilogy(
            [row["snr_db"] for row in rows],
            [row["ber"] for row in rows],
            marker=markers[frontend], markersize=4, linewidth=1.6, label=frontend,
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("SU-MIMO Real CNN: Dense-SNR Evaluation")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "ber_vs_snr_dense.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    with (output_dir / "evaluation_config.json").open("w", encoding="utf-8") as file:
        json.dump({**vars(args), "snr_points": snr_points}, file, indent=2)
    print(f"\nCSV:  {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
