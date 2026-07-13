"""Three-stage training for H-denoised Sionna neural receivers.

Stages
------
``denoiser``
    Train only the phase-equivariant H denoiser with channel NMSE.
``frozen``
    Load and freeze a pretrained denoiser, then train a fresh receiver with BCE.
``joint``
    Load a frozen-stage receiver and jointly fine-tune it with BCE + H-NMSE.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

from data import SionnaOFDMBatchGenerator
from models.factory import DENOISED_BASE_MODEL, MODEL_CHOICES, build_model_from_args
from models.phase_equivariant_denoiser import (
    EquivariantHResidualDenoiser,
    complex_nmse,
)
from train_sionna import batch_sizes, build_data_config, build_generator
from utils.metrics import masked_bce_with_logits, masked_error_count


def _load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def _assert_matching_config(checkpoint, data_config, label):
    saved = checkpoint.get("sionna_config")
    current = data_config.to_dict()
    if saved != current:
        raise ValueError(
            f"{label} uses a different Sionna OFDM configuration. "
            f"saved={saved}, current={current}"
        )


def run_denoiser_epoch(
    denoiser,
    generator,
    num_samples,
    batch_size,
    optimizer=None,
    log_interval=0,
):
    training = optimizer is not None
    denoiser.train(training)
    refined_sum = 0.0
    ls_sum = 0.0
    seen = 0

    if not training:
        generator.reset()

    for step, current_batch_size in enumerate(
        batch_sizes(num_samples, batch_size), start=1
    ):
        with torch.set_grad_enabled(training):
            batch = generator.generate_batch(current_batch_size)
            refined = denoiser(batch["H_hat"], batch["P"], batch["N0"])
            loss = complex_nmse(refined, batch["H"])
            ls_loss = complex_nmse(batch["H_hat"], batch["H"])
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        refined_sum += float(loss.detach().item()) * current_batch_size
        ls_sum += float(ls_loss.detach().item()) * current_batch_size
        seen += current_batch_size
        if log_interval > 0 and step % log_interval == 0:
            print(
                f"  step {step:05d} | LS-NMSE {ls_sum / seen:.6f} | "
                f"H-NMSE {refined_sum / seen:.6f}"
            )

    return ls_sum / seen, refined_sum / seen


def run_receiver_epoch(
    model,
    generator,
    num_samples,
    batch_size,
    h_denoise_weight,
    optimizer=None,
    log_interval=0,
):
    training = optimizer is not None
    model.train(training)
    total_sum = 0.0
    bce_sum = 0.0
    refined_nmse_sum = 0.0
    ls_nmse_sum = 0.0
    seen = 0
    total_errors = 0
    total_bits = 0

    if not training:
        generator.reset()

    for step, current_batch_size in enumerate(
        batch_sizes(num_samples, batch_size), start=1
    ):
        with torch.set_grad_enabled(training):
            batch = generator.generate_batch(current_batch_size)
            logits, aux = model.forward_with_aux(
                batch["Y"], batch["H_hat"], batch["P"], batch["N0"]
            )
            refined_nmse = complex_nmse(aux["H_refined"], batch["H"])
            ls_nmse = complex_nmse(batch["H_hat"], batch["H"])
            bce = masked_bce_with_logits(
                logits, batch["bits"], batch["loss_mask"]
            )
            loss = bce + h_denoise_weight * refined_nmse
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        errors, valid_bits = masked_error_count(
            logits.detach(), batch["bits"], batch["loss_mask"]
        )
        total_sum += float(loss.detach().item()) * current_batch_size
        bce_sum += float(bce.detach().item()) * current_batch_size
        refined_nmse_sum += (
            float(refined_nmse.detach().item()) * current_batch_size
        )
        ls_nmse_sum += float(ls_nmse.detach().item()) * current_batch_size
        seen += current_batch_size
        total_errors += int(errors.item())
        total_bits += int(valid_bits.item())

        if log_interval > 0 and step % log_interval == 0:
            print(
                f"  step {step:05d} | total {total_sum / seen:.5f} | "
                f"BCE {bce_sum / seen:.5f} | "
                f"BER {total_errors / total_bits:.5f} | "
                f"LS/H NMSE {ls_nmse_sum / seen:.5f}/"
                f"{refined_nmse_sum / seen:.5f}"
            )

    return {
        "total": total_sum / seen,
        "bce": bce_sum / seen,
        "ber": total_errors / total_bits,
        "ls_nmse": ls_nmse_sum / seen,
        "h_nmse": refined_nmse_sum / seen,
    }


@torch.no_grad()
def write_nmse_by_snr(denoiser, args, data_config, save_dir, device):
    rows = []
    for text in args.nmse_snr_list.split(","):
        snr_db = float(text.strip())
        generator = SionnaOFDMBatchGenerator(
            data_config,
            snr_db_min=snr_db,
            snr_db_max=snr_db,
            phase_mode="uniform",
            seed=args.seed + 200000,
            device=device,
        )
        ls_nmse, refined_nmse = run_denoiser_epoch(
            denoiser,
            generator,
            args.nmse_eval_samples,
            args.batch_size,
        )
        gain_db = 10.0 * math.log10(ls_nmse / refined_nmse)
        print(
            f"SNR {snr_db:5.1f} dB | LS-NMSE {ls_nmse:.6e} | "
            f"H-NMSE {refined_nmse:.6e} | gain {gain_db:+.3f} dB"
        )
        rows.append(
            {
                "snr_db": snr_db,
                "ls_nmse": ls_nmse,
                "refined_nmse": refined_nmse,
                "nmse_gain_db": gain_db,
            }
        )

    path = save_dir / "nmse_by_snr.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved NMSE curve to {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=["denoiser", "frozen", "joint"]
    )
    parser.add_argument(
        "--model",
        default="single_branch_n0_gate_h_denoise",
        choices=MODEL_CHOICES,
    )
    parser.add_argument("--denoiser_checkpoint")
    parser.add_argument("--init_checkpoint")
    parser.add_argument(
        "--train_phase_mode", default="fixed", choices=["fixed", "narrow", "uniform"]
    )
    parser.add_argument(
        "--val_phase_mode", default="uniform", choices=["fixed", "narrow", "uniform"]
    )
    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_val", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--snr_db_min", type=float, default=-10.0)
    parser.add_argument("--snr_db_max", type=float, default=20.0)

    parser.add_argument("--num_ofdm_symbols", type=int, default=14)
    parser.add_argument("--fft_size", type=int, default=72)
    parser.add_argument("--subcarrier_spacing_hz", type=float, default=30e3)
    parser.add_argument("--cyclic_prefix_length", type=int, default=0)
    parser.add_argument("--dmrs_symbols", type=int, nargs="+", default=[2, 11])
    parser.add_argument("--dmrs_freq_spacing", type=int, default=1)
    parser.add_argument("--dmrs_freq_offset", type=int, default=0)
    parser.add_argument(
        "--tdl_model",
        default="A",
        choices=["A", "B", "C", "D", "E", "A30", "B100", "C300"],
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

    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--hidden_complex", type=int, default=32)
    parser.add_argument("--zero_complex", type=int, default=32)
    parser.add_argument("--branch_layers", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument(
        "--gate_type", default="swiglu", choices=["sigmoid", "swiglu"]
    )
    parser.add_argument(
        "--single_readout_mode", default="low_rank", choices=["low_rank", "full"]
    )
    parser.add_argument("--zero_gate_hidden", type=int, default=16)
    parser.add_argument("--denoiser_hidden", type=int, default=16)
    parser.add_argument("--denoiser_blocks", type=int, default=2)
    parser.add_argument("--h_denoise_weight", type=float, default=0.01)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--nmse_snr_list", default="-10,-5,0,5,10,15,20"
    )
    parser.add_argument("--nmse_eval_samples", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_train <= 0 or args.num_val <= 0 or args.batch_size <= 0:
        raise ValueError("num_train, num_val, and batch_size must be positive.")
    if args.h_denoise_weight < 0.0:
        raise ValueError("h_denoise_weight must be non-negative.")
    if args.stage in {"frozen", "joint"} and args.model not in DENOISED_BASE_MODEL:
        raise ValueError(
            "Receiver stages require a denoised A/C model alias; got "
            f"{args.model!r}."
        )
    if args.stage == "frozen" and not args.denoiser_checkpoint:
        raise ValueError("--denoiser_checkpoint is required for stage=frozen.")
    if args.stage == "joint" and not args.init_checkpoint:
        raise ValueError("--init_checkpoint is required for stage=joint.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in PyTorch.")
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

    if args.stage == "denoiser":
        model = EquivariantHResidualDenoiser(
            hidden_complex=args.denoiser_hidden,
            num_blocks=args.denoiser_blocks,
            kernel_size=args.kernel_size,
            use_norm=not args.no_norm,
            gate_type=args.gate_type,
            condition_hidden=args.zero_gate_hidden,
            condition_mode="p_n0",
        ).to(device)
        h_weight = 1.0
    else:
        model = build_model_from_args(args, bits_per_symbol=2).to(device)
        if args.stage == "frozen":
            checkpoint = _load_checkpoint(args.denoiser_checkpoint, device)
            if checkpoint.get("checkpoint_type") != "h_denoiser":
                raise ValueError("Expected an H-denoiser checkpoint.")
            _assert_matching_config(checkpoint, data_config, "Denoiser checkpoint")
            model.denoiser.load_state_dict(checkpoint["model_state"])
            model.denoiser.requires_grad_(False)
            h_weight = 0.0
        else:
            checkpoint = _load_checkpoint(args.init_checkpoint, device)
            _assert_matching_config(checkpoint, data_config, "Initial checkpoint")
            if checkpoint.get("args", {}).get("model") != args.model:
                raise ValueError(
                    "Joint initialization model does not match --model."
                )
            model.load_state_dict(checkpoint["model_state"])
            model.requires_grad_(True)
            h_weight = args.h_denoise_weight

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_metric = math.inf
    print(f"Device: {device}")
    print(f"Stage: {args.stage}")
    print(f"Model: {args.model if args.stage != 'denoiser' else 'H denoiser'}")
    print(f"Train/validation phase: {args.train_phase_mode}/{args.val_phase_mode}")
    print(f"Effective H-NMSE weight: {h_weight}")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        if args.stage == "denoiser":
            train_ls, train_h = run_denoiser_epoch(
                model,
                train_generator,
                args.num_train,
                args.batch_size,
                optimizer,
                args.log_interval,
            )
            val_ls, val_h = run_denoiser_epoch(
                model, val_generator, args.num_val, args.batch_size
            )
            print(
                f"Epoch {epoch:03d} | train LS/H NMSE "
                f"{train_ls:.6f}/{train_h:.6f} | val LS/H NMSE "
                f"{val_ls:.6f}/{val_h:.6f}"
            )
            metric = val_h
            checkpoint_type = "h_denoiser"
            model_name = "equivariant_h_denoiser"
        else:
            train_metrics = run_receiver_epoch(
                model,
                train_generator,
                args.num_train,
                args.batch_size,
                h_weight,
                optimizer,
                args.log_interval,
            )
            val_metrics = run_receiver_epoch(
                model,
                val_generator,
                args.num_val,
                args.batch_size,
                h_weight,
            )
            print(
                f"Epoch {epoch:03d} | train BCE/BER "
                f"{train_metrics['bce']:.5f}/{train_metrics['ber']:.5f} | "
                f"val BCE/BER {val_metrics['bce']:.5f}/"
                f"{val_metrics['ber']:.5f}"
            )
            print(
                f"             train LS/H NMSE "
                f"{train_metrics['ls_nmse']:.5f}/"
                f"{train_metrics['h_nmse']:.5f} | val LS/H NMSE "
                f"{val_metrics['ls_nmse']:.5f}/"
                f"{val_metrics['h_nmse']:.5f}"
            )
            metric = val_metrics["bce"]
            checkpoint_type = "receiver"
            model_name = f"{args.model}_{args.stage}"

        checkpoint = {
            "checkpoint_type": checkpoint_type,
            "model_name": model_name,
            "model_state": model.state_dict(),
            "args": vars(args),
            "sionna_config": data_config.to_dict(),
            "data_backend": "sionna",
            "training_stage": args.stage,
        }
        torch.save(checkpoint, save_dir / "last.pt")
        if metric < best_metric:
            best_metric = metric
            torch.save(checkpoint, save_dir / "best.pt")
            print(f"  saved best checkpoint to {save_dir / 'best.pt'}")

    if args.stage == "denoiser":
        best = _load_checkpoint(save_dir / "best.pt", device)
        model.load_state_dict(best["model_state"])
        model.eval()
        write_nmse_by_snr(model, args, data_config, save_dir, device)


if __name__ == "__main__":
    main()
