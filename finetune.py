import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import OFDMDataset
from train import build_model, evaluate, move_batch, train_one_epoch


def optional_float(text):
    if text is None or str(text).lower() in ["none", "null"]:
        return None
    return float(text)


def make_dataset(
    num_samples,
    snr_db_min,
    snr_db_max,
    channel_error_std,
    h_hat_mode,
    phase_mode,
    dmrs_freq_spacing,
    dmrs_freq_offset,
    channel_kind,
    subcarrier_spacing_hz,
    num_paths,
    rms_delay_spread_s,
    max_doppler_hz,
    rician_k_db,
    seed,
):
    return OFDMDataset(
        num_samples=num_samples,
        snr_db_min=snr_db_min,
        snr_db_max=snr_db_max,
        channel_error_std=channel_error_std,
        h_hat_mode=h_hat_mode,
        phase_mode=phase_mode,
        dmrs_freq_spacing=dmrs_freq_spacing,
        dmrs_freq_offset=dmrs_freq_offset,
        channel_kind=channel_kind,
        subcarrier_spacing_hz=subcarrier_spacing_hz,
        num_paths=num_paths,
        rms_delay_spread_s=rms_delay_spread_s,
        max_doppler_hz=max_doppler_hz,
        rician_k_db=rician_k_db,
        seed=seed,
    )


def make_loader(dataset, batch_size, shuffle, num_workers, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def load_model(args, device):
    model = build_model(
        args.model,
        bits_per_symbol=2,
        hidden=args.hidden,
        hidden_complex=args.hidden_complex,
        zero_complex=args.zero_complex,
        branch_layers=args.branch_layers,
        kernel_size=args.kernel_size,
        use_norm=not args.no_norm,
        gate_type=args.gate_type,
        single_readout_mode=args.single_readout_mode,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    return model, ckpt if isinstance(ckpt, dict) else {}


def apply_freeze_mode(model, freeze_mode):
    for param in model.parameters():
        param.requires_grad = True

    if freeze_mode == "none":
        return

    for param in model.parameters():
        param.requires_grad = False

    if freeze_mode == "llr_head":
        for name, param in model.named_parameters():
            if name.startswith("llr_head."):
                param.requires_grad = True
        return

    raise ValueError(f"Unknown freeze_mode: {freeze_mode}")


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_target_loader(args, device, num_samples, seed, shuffle):
    dataset = make_dataset(
        num_samples=num_samples,
        snr_db_min=args.target_snr_db_min,
        snr_db_max=args.target_snr_db_max,
        channel_error_std=args.target_channel_error_std,
        h_hat_mode=args.target_h_hat_mode,
        phase_mode=args.target_phase_mode,
        dmrs_freq_spacing=args.target_dmrs_freq_spacing,
        dmrs_freq_offset=args.target_dmrs_freq_offset,
        channel_kind=args.target_channel_kind,
        subcarrier_spacing_hz=args.target_subcarrier_spacing_hz,
        num_paths=args.target_num_paths,
        rms_delay_spread_s=args.target_rms_delay_spread_s,
        max_doppler_hz=args.target_max_doppler_hz,
        rician_k_db=args.target_rician_k_db,
        seed=seed,
    )
    return make_loader(dataset, args.batch_size, shuffle, args.num_workers, device)


def save_checkpoint(path, model, optimizer, args, epoch, best_val_loss, source_args):
    torch.save(
        {
            "model_name": args.model,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "args": vars(args),
            "source_args": source_args,
        },
        path,
    )


def write_history(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=[
                            "real_imag_cnn",
                            "physical_cnn",
                            "phase_invariant",
                            "complex_no_interaction",
                            "single_branch",
                        ])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--hidden_complex", type=int, default=32)
    parser.add_argument("--zero_complex", type=int, default=32)
    parser.add_argument("--branch_layers", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--gate_type", type=str, default="swiglu",
                        choices=["sigmoid", "swiglu"])
    parser.add_argument("--single_readout_mode", type=str, default="low_rank",
                        choices=["low_rank", "full"])

    parser.add_argument("--num_adapt", type=int, default=512)
    parser.add_argument("--num_val", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--freeze_mode", type=str, default="none",
                        choices=["none", "llr_head"])
    parser.add_argument("--resample_each_epoch", action="store_true")

    parser.add_argument("--target_phase_mode", type=str, default="uniform",
                        choices=["fixed", "narrow", "uniform"])
    parser.add_argument("--target_snr_db_min", type=float, default=-5.0)
    parser.add_argument("--target_snr_db_max", type=float, default=20.0)
    parser.add_argument("--target_channel_error_std", type=float, default=0.05)
    parser.add_argument("--target_h_hat_mode", type=str, default="dmrs_ls_interp",
                        choices=["oracle_noisy", "dmrs_ls_interp"])
    parser.add_argument("--target_dmrs_freq_spacing", type=int, default=1)
    parser.add_argument("--target_dmrs_freq_offset", type=int, default=0)

    parser.add_argument("--target_channel_kind", type=str, default="tdl",
                        choices=["smooth", "iid", "tdl"])
    parser.add_argument("--target_subcarrier_spacing_hz", type=float, default=30e3)
    parser.add_argument("--target_num_paths", type=int, default=12)
    parser.add_argument("--target_rms_delay_spread_s", type=float, default=10e-9)
    parser.add_argument("--target_max_doppler_hz", type=float, default=200.0)
    parser.add_argument("--target_rician_k_db", type=optional_float, default=None)

    parser.add_argument("--adapt_seed", type=int, default=200000)
    parser.add_argument("--val_seed", type=int, default=300000)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model, source_ckpt = load_model(args, device)
    source_args = source_ckpt.get("args", {})
    apply_freeze_mode(model, args.freeze_mode)

    trainable_params = count_trainable_params(model)
    if trainable_params == 0:
        raise ValueError("No trainable parameters. Check freeze_mode.")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    val_loader = build_target_loader(
        args,
        device=device,
        num_samples=args.num_val,
        seed=args.val_seed,
        shuffle=False,
    )

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Target channel: {args.target_channel_kind} | target phase: {args.target_phase_mode}")
    print(
        "Target H_hat: "
        f"{args.target_h_hat_mode} | DMRS spacing {args.target_dmrs_freq_spacing} | "
        f"delay spread {args.target_rms_delay_spread_s:.3e} s | Doppler {args.target_max_doppler_hz:.1f} Hz"
    )
    print(f"Freeze mode: {args.freeze_mode} | trainable params: {trainable_params}")

    history = []
    val_loss, val_ber = evaluate(model, val_loader, device)
    print(f"\nBefore finetune | target val loss {val_loss:.5f} | target val BER {val_ber:.5f}")
    history.append(
        {
            "epoch": 0,
            "train_loss": "",
            "train_ber": "",
            "val_loss": val_loss,
            "val_ber": val_ber,
            "lr": args.lr,
        }
    )

    best_val_loss = val_loss
    save_checkpoint(save_dir / "best.pt", model, optimizer, args, 0, best_val_loss, source_args)
    save_checkpoint(save_dir / "last.pt", model, optimizer, args, 0, best_val_loss, source_args)

    for epoch in range(1, args.epochs + 1):
        train_seed = args.adapt_seed
        if args.resample_each_epoch:
            train_seed = args.adapt_seed + epoch * args.num_adapt

        train_loader = build_target_loader(
            args,
            device=device,
            num_samples=args.num_adapt,
            seed=train_seed,
            shuffle=True,
        )

        print(f"\nFinetune epoch {epoch}/{args.epochs} | adapt seed {train_seed}")
        train_loss, train_ber = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.log_interval,
        )
        val_loss, val_ber = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"adapt loss {train_loss:.5f} | adapt BER {train_ber:.5f} | "
            f"target val loss {val_loss:.5f} | target val BER {val_ber:.5f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ber": train_ber,
                "val_loss": val_loss,
                "val_ber": val_ber,
                "lr": args.lr,
            }
        )

        save_checkpoint(save_dir / "last.pt", model, optimizer, args, epoch, best_val_loss, source_args)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(save_dir / "best.pt", model, optimizer, args, epoch, best_val_loss, source_args)
            print(f"  saved best checkpoint to {save_dir / 'best.pt'}")

        write_history(save_dir / "history.csv", history)

    write_history(save_dir / "history.csv", history)
    print(f"\nSaved finetuned checkpoints and history to {save_dir}")


if __name__ == "__main__":
    main()
