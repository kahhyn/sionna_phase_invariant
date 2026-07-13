"""Oracle-label few-shot adaptation under an OFDM channel-domain shift.

Every adaptation budget starts from the same source checkpoint. The target
examples are deterministic and nested when the budgets are multiples of the
adaptation batch size. This makes the invariant and strict complex receivers
comparable under exactly the same labels, channels, noise, and optimizer.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from data import SionnaOFDMBatchGenerator, SionnaOFDMConfig
from train_sionna import batch_sizes
from utils.checkpoints import load_receiver_checkpoint
from utils.metrics import masked_bce_sum, masked_bce_with_logits, masked_error_count


ALLOWED_MODELS = {
    "single_branch_n0_gate",
    "strict_matched_complex_p_n0_gate",
}


def parse_number_list(text: str, cast):
    values = [cast(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("The list must contain at least one value.")
    return values


def set_random_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_domain(
    model,
    config: SionnaOFDMConfig,
    snr_values: list[float],
    *,
    phase_mode: str,
    num_samples_per_snr: int,
    batch_size: int,
    seed: int,
    device: torch.device,
):
    """Evaluate one model with fixed, repeatable samples at every SNR."""
    model.eval()
    rows = []
    aggregate_bce = 0.0
    aggregate_errors = 0
    aggregate_bits = 0

    for snr_index, snr_db in enumerate(snr_values):
        generator = SionnaOFDMBatchGenerator(
            config,
            snr_db_min=snr_db,
            snr_db_max=snr_db,
            phase_mode=phase_mode,
            seed=seed + 1000 * snr_index,
            device=device,
        )
        bce_sum_total = 0.0
        error_total = 0
        bit_total = 0
        for current_batch_size in batch_sizes(num_samples_per_snr, batch_size):
            batch = generator.generate_batch(current_batch_size)
            logits = model(batch["Y"], batch["H_hat"], batch["P"], batch["N0"])
            bce_sum, bce_count = masked_bce_sum(
                logits, batch["bits"], batch["loss_mask"]
            )
            errors, valid_bits = masked_error_count(
                logits, batch["bits"], batch["loss_mask"]
            )
            if int(bce_count.item()) != int(valid_bits.item()):
                raise RuntimeError("BCE and BER masks disagree.")
            bce_sum_total += float(bce_sum.item())
            error_total += int(errors.item())
            bit_total += int(valid_bits.item())

        rows.append(
            {
                "snr_db": snr_db,
                "bce": bce_sum_total / bit_total,
                "ber": error_total / bit_total,
                "bit_errors": error_total,
                "valid_bits": bit_total,
            }
        )
        aggregate_bce += bce_sum_total
        aggregate_errors += error_total
        aggregate_bits += bit_total

    aggregate = {
        "bce": aggregate_bce / aggregate_bits,
        "ber": aggregate_errors / aggregate_bits,
        "bit_errors": aggregate_errors,
        "valid_bits": aggregate_bits,
    }
    return rows, aggregate


def adapt_with_oracle_labels(
    model,
    config: SionnaOFDMConfig,
    *,
    num_samples: int,
    epochs: int,
    batch_size: int,
    snr_db_min: float,
    snr_db_max: float,
    phase_mode: str,
    seed: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    device: torch.device,
):
    """Fine-tune all receiver parameters on a fixed oracle-labeled set."""
    if num_samples == 0:
        return {"updates": 0, "mean_loss": math.nan, "mean_ber": math.nan}

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    generator = SionnaOFDMBatchGenerator(
        config,
        snr_db_min=snr_db_min,
        snr_db_max=snr_db_max,
        phase_mode=phase_mode,
        seed=seed,
        device=device,
    )
    model.train()
    updates = 0
    loss_sum = 0.0
    error_sum = 0
    bit_sum = 0

    for _ in range(epochs):
        # Reuse the exact same target-domain examples on every epoch. Across
        # budgets, a fixed batch size also makes the adaptation sets nested.
        generator.reset(seed)
        for current_batch_size in batch_sizes(num_samples, batch_size):
            batch = generator.generate_batch(current_batch_size)
            logits = model(batch["Y"], batch["H_hat"], batch["P"], batch["N0"])
            loss = masked_bce_with_logits(
                logits, batch["bits"], batch["loss_mask"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            errors, valid_bits = masked_error_count(
                logits.detach(), batch["bits"], batch["loss_mask"]
            )
            count = int(valid_bits.item())
            loss_sum += float(loss.item()) * count
            error_sum += int(errors.item())
            bit_sum += count
            updates += 1

    return {
        "updates": updates,
        "mean_loss": loss_sum / bit_sum,
        "mean_ber": error_sum / bit_sum,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Oracle few-shot adaptation across TDL delay-spread domains."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_delay_ns", default="50,100,300")
    parser.add_argument(
        "--target_tdl_model",
        default=None,
        help="Defaults to the source checkpoint's TDL model.",
    )
    parser.add_argument("--sample_budgets", default="0,16,64,256,1024,4096")
    parser.add_argument("--adapt_epochs", type=int, default=5)
    parser.add_argument("--adapt_batch_size", type=int, default=16)
    parser.add_argument("--adapt_snr_db_min", type=float, default=-10.0)
    parser.add_argument("--adapt_snr_db_max", type=float, default=20.0)
    parser.add_argument("--adapt_phase_mode", default="uniform", choices=["fixed", "narrow", "uniform"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--eval_snr_list", default="-5,0,5,10,15,20")
    parser.add_argument("--num_eval_per_snr", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--eval_phase_mode", default="uniform", choices=["fixed", "narrow", "uniform"])
    parser.add_argument("--adapt_seed", type=int, default=500000)
    parser.add_argument("--eval_seed", type=int, default=777000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_checkpoints", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    budgets = sorted(set(parse_number_list(args.sample_budgets, int) + [0]))
    target_delay_ns = parse_number_list(args.target_delay_ns, float)
    eval_snr_values = parse_number_list(args.eval_snr_list, float)

    if any(value < 0 for value in budgets):
        raise ValueError("Adaptation budgets must be non-negative.")
    if any(value <= 0 for value in target_delay_ns):
        raise ValueError("Target delay spreads must be positive.")
    if args.adapt_epochs <= 0 or args.adapt_batch_size <= 0:
        raise ValueError("Adaptation epochs and batch size must be positive.")
    if args.num_eval_per_snr <= 0 or args.eval_batch_size <= 0:
        raise ValueError("Evaluation sample and batch sizes must be positive.")
    if args.adapt_snr_db_min > args.adapt_snr_db_max:
        raise ValueError("Adaptation SNR minimum exceeds its maximum.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in PyTorch.")
    set_random_seed(args.adapt_seed, device)

    base_model, checkpoint = load_receiver_checkpoint(
        args.checkpoint,
        device,
        bits_per_symbol=2,
        required_data_backend="sionna",
    )
    model_name = checkpoint["model_name"]
    if model_name not in ALLOWED_MODELS:
        raise ValueError(
            f"This experiment compares only native A/C receivers; got {model_name!r}."
        )
    source_config = SionnaOFDMConfig(**checkpoint["sionna_config"])
    target_tdl_model = args.target_tdl_model or source_config.tdl_model
    train_seed = int(checkpoint.get("args", {}).get("seed", -1))
    parameter_count = sum(parameter.numel() for parameter in base_model.parameters())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **vars(args),
        "checkpoint": str(Path(args.checkpoint)),
        "model": model_name,
        "train_seed": train_seed,
        "parameter_count": parameter_count,
        "source_sionna_config": source_config.to_dict(),
        "sample_budgets_parsed": budgets,
        "target_delay_ns_parsed": target_delay_ns,
        "eval_snr_list_parsed": eval_snr_values,
    }
    with (output_dir / "experiment_config.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    detail_rows = []
    summary_rows = []
    print(f"Model: {model_name} | parameters: {parameter_count:,}")
    print(
        f"Source: TDL-{source_config.tdl_model}, "
        f"delay={source_config.delay_spread_s * 1e9:g} ns"
    )

    for target_index, delay_ns in enumerate(target_delay_ns):
        target_config = replace(
            source_config,
            tdl_model=target_tdl_model,
            delay_spread_s=delay_ns * 1e-9,
        )
        target_adapt_seed = args.adapt_seed + target_index * 100000
        target_eval_seed = args.eval_seed + target_index * 100000
        per_target = []

        print(f"\nTarget: TDL-{target_tdl_model}, delay={delay_ns:g} ns")
        for budget in budgets:
            set_random_seed(target_adapt_seed, device)
            model = copy.deepcopy(base_model).to(device)
            adaptation = adapt_with_oracle_labels(
                model,
                target_config,
                num_samples=budget,
                epochs=args.adapt_epochs,
                batch_size=args.adapt_batch_size,
                snr_db_min=args.adapt_snr_db_min,
                snr_db_max=args.adapt_snr_db_max,
                phase_mode=args.adapt_phase_mode,
                seed=target_adapt_seed,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip_norm=args.grad_clip_norm,
                device=device,
            )
            target_rows, target_aggregate = evaluate_domain(
                model,
                target_config,
                eval_snr_values,
                phase_mode=args.eval_phase_mode,
                num_samples_per_snr=args.num_eval_per_snr,
                batch_size=args.eval_batch_size,
                seed=target_eval_seed,
                device=device,
            )
            source_rows, source_aggregate = evaluate_domain(
                model,
                source_config,
                eval_snr_values,
                phase_mode=args.eval_phase_mode,
                num_samples_per_snr=args.num_eval_per_snr,
                batch_size=args.eval_batch_size,
                seed=args.eval_seed,
                device=device,
            )

            common = {
                "checkpoint": str(Path(args.checkpoint)),
                "model": model_name,
                "train_seed": train_seed,
                "parameter_count": parameter_count,
                "source_tdl_model": source_config.tdl_model,
                "source_delay_ns": source_config.delay_spread_s * 1e9,
                "target_tdl_model": target_tdl_model,
                "target_delay_ns": delay_ns,
                "budget": budget,
                "adapt_epochs": args.adapt_epochs,
                "adapt_updates": adaptation["updates"],
                "adapt_seed": target_adapt_seed,
                "target_eval_seed": target_eval_seed,
                "source_eval_seed": args.eval_seed,
            }
            domain_rows = (
                ("target", target_rows, target_eval_seed),
                ("source", source_rows, args.eval_seed),
            )
            for domain, rows, domain_eval_seed in domain_rows:
                for row in rows:
                    detail_rows.append(
                        {
                            **common,
                            "domain": domain,
                            "eval_seed": domain_eval_seed,
                            **row,
                        }
                    )

            per_target.append(
                {
                    **common,
                    "adapt_train_loss": adaptation["mean_loss"],
                    "adapt_train_ber": adaptation["mean_ber"],
                    "target_post_bce": target_aggregate["bce"],
                    "target_post_ber": target_aggregate["ber"],
                    "target_bit_errors": target_aggregate["bit_errors"],
                    "target_valid_bits": target_aggregate["valid_bits"],
                    "source_post_bce": source_aggregate["bce"],
                    "source_post_ber": source_aggregate["ber"],
                    "source_bit_errors": source_aggregate["bit_errors"],
                    "source_valid_bits": source_aggregate["valid_bits"],
                }
            )
            print(
                f"  budget {budget:4d} | updates {adaptation['updates']:4d} | "
                f"target BER {target_aggregate['ber']:.6e} | "
                f"source BER {source_aggregate['ber']:.6e}"
            )

            if args.save_checkpoints and budget > 0:
                save_path = output_dir / f"target_{delay_ns:g}ns" / f"budget_{budget}.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                adapted_checkpoint = dict(checkpoint)
                adapted_checkpoint["model_state"] = model.state_dict()
                adapted_checkpoint["sionna_config"] = target_config.to_dict()
                adapted_checkpoint["adaptation"] = common
                adapted_checkpoint["source_sionna_config"] = source_config.to_dict()
                torch.save(adapted_checkpoint, save_path)

            del model

        target_pre = per_target[0]["target_post_ber"]
        source_pre = per_target[0]["source_post_ber"]
        best_target = min(row["target_post_ber"] for row in per_target)
        recoverable_gap = target_pre - best_target
        threshold_90 = target_pre - 0.9 * recoverable_gap
        n90 = 0
        if recoverable_gap > 0:
            n90 = next(
                (row["budget"] for row in per_target if row["target_post_ber"] <= threshold_90),
                -1,
            )

        for row in per_target:
            target_improvement = target_pre - row["target_post_ber"]
            recovery = (
                target_improvement / recoverable_gap if recoverable_gap > 0 else 0.0
            )
            summary_rows.append(
                {
                    **row,
                    "target_pre_ber": target_pre,
                    "target_best_ber": best_target,
                    "target_ber_improvement": target_improvement,
                    "recovery_fraction_to_best": recovery,
                    "source_pre_ber": source_pre,
                    "source_ber_forgetting": row["source_post_ber"] - source_pre,
                    "n90_samples": n90,
                }
            )

    write_csv(output_dir / "fewshot_per_snr.csv", detail_rows)
    write_csv(output_dir / "fewshot_summary.csv", summary_rows)
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
