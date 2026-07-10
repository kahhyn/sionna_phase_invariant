import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import OFDMDataset
from models import (
    ComplexCNNNoInteraction,
    PhysicalFeatureCNN,
    PhaseInvariantReceiver,
    RealImagCNN,
    SingleBranchPhaseInvariantReceiver,
)
from utils.metrics import masked_bce_with_logits, masked_ber


def build_model(
    name,
    bits_per_symbol,
    hidden=32,
    hidden_complex=16,
    zero_complex=16,
    branch_layers=2,
    kernel_size=3,
    use_norm=True,
    gate_type="swiglu",
    single_readout_mode="low_rank",
):
    if name == "real_imag_cnn":
        return RealImagCNN(hidden=hidden, bits_per_symbol=bits_per_symbol)
    if name == "physical_cnn":
        return PhysicalFeatureCNN(
            hidden=hidden,
            zero_complex=zero_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            branch_layers=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
        )
    if name == "phase_invariant":
        return PhaseInvariantReceiver(
            hidden_complex=hidden_complex,
            zero_complex=zero_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            branch_layers=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
            gate_type=gate_type
        )
    if name == "complex_no_interaction":
        return ComplexCNNNoInteraction(
            hidden_complex=hidden_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            branch_layers=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
            gate_type=gate_type,
        )
    if name == "single_branch":
        return SingleBranchPhaseInvariantReceiver(
            hidden_complex=hidden_complex,
            zero_real=zero_complex,
            hidden_real=hidden,
            bits_per_symbol=bits_per_symbol,
            num_blocks=branch_layers,
            kernel_size=kernel_size,
            use_norm=use_norm,
            gate_type=gate_type,
            readout_mode=single_readout_mode,
        )
    raise ValueError(f"Unknown model: {name}")


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train_one_epoch(model, loader, optimizer, device, log_interval):
    model.train()

    total_loss = 0.0
    total_ber = 0.0
    count = 0

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)

        logits = model(batch["Y"], batch["H_hat"], batch["P"], batch["N0"])
        loss = masked_bce_with_logits(logits, batch["bits"], batch["loss_mask"])
        ber = masked_ber(logits.detach(), batch["bits"], batch["loss_mask"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ber += ber.item()
        count += 1

        if log_interval > 0 and step % log_interval == 0:
            print(f"  step {step:05d} | loss {total_loss/count:.5f} | BER {total_ber/count:.5f}")

    return total_loss / max(count, 1), total_ber / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_ber = 0.0
    count = 0

    for batch in loader:
        batch = move_batch(batch, device)

        logits = model(batch["Y"], batch["H_hat"], batch["P"], batch["N0"])
        loss = masked_bce_with_logits(logits, batch["bits"], batch["loss_mask"])
        ber = masked_ber(logits, batch["bits"], batch["loss_mask"])

        total_loss += loss.item()
        total_ber += ber.item()
        count += 1

    return total_loss / max(count, 1), total_ber / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="phase_invariant",
                        choices=[
                            "real_imag_cnn",
                            "physical_cnn",
                            "phase_invariant",
                            "complex_no_interaction",
                            "single_branch",
                        ])

    parser.add_argument("--train_phase_mode", type=str, default="fixed",
                        choices=["fixed", "narrow", "uniform"])
    parser.add_argument("--val_phase_mode", type=str, default="uniform",
                        choices=["fixed", "narrow", "uniform"])

    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_val", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--snr_db_min", type=float, default=-5.0)
    parser.add_argument("--snr_db_max", type=float, default=20.0)
    parser.add_argument("--channel_error_std", type=float, default=0.05)
    parser.add_argument("--h_hat_mode", type=str, default="dmrs_ls_interp",
                        choices=["oracle_noisy", "dmrs_ls_interp"])
    parser.add_argument("--dmrs_freq_spacing", type=int, default=1)
    parser.add_argument("--dmrs_freq_offset", type=int, default=0)

    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--hidden_complex", type=int, default=16)
    parser.add_argument("--zero_complex", type=int, default=16)
    parser.add_argument("--branch_layers", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--gate_type", type=str, default="swiglu",
                        choices=["sigmoid", "swiglu"])
    parser.add_argument("--single_readout_mode", type=str, default="low_rank",
                        choices=["low_rank", "full"])

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument("--save_dir", type=str, default="runs/debug/")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_set = OFDMDataset(
        num_samples=args.num_train,
        snr_db_min=args.snr_db_min,
        snr_db_max=args.snr_db_max,
        channel_error_std=args.channel_error_std,
        h_hat_mode=args.h_hat_mode,
        phase_mode=args.train_phase_mode,
        dmrs_freq_spacing=args.dmrs_freq_spacing,
        dmrs_freq_offset=args.dmrs_freq_offset,
        seed=0,
    )

    val_set = OFDMDataset(
        num_samples=args.num_val,
        snr_db_min=args.snr_db_min,
        snr_db_max=args.snr_db_max,
        channel_error_std=args.channel_error_std,
        h_hat_mode=args.h_hat_mode,
        phase_mode=args.val_phase_mode,
        dmrs_freq_spacing=args.dmrs_freq_spacing,
        dmrs_freq_offset=args.dmrs_freq_offset,
        seed=100000,
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Train phase mode: {args.train_phase_mode} | Val phase mode: {args.val_phase_mode}")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_ber = train_one_epoch(
            model, train_loader, optimizer, device, args.log_interval
        )

        val_loss, val_ber = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.5f} | train BER {train_ber:.5f} | "
            f"val loss {val_loss:.5f} | val BER {val_ber:.5f}"
        )

        ckpt = {
            "model_name": args.model,
            "model_state": model.state_dict(),
            "args": vars(args),
        }

        torch.save(ckpt, save_dir / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  saved best checkpoint to {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
