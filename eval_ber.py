import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import OFDMDataset
from train import build_model, move_batch
from utils.metrics import masked_bce_with_logits, masked_ber


@torch.no_grad()
def evaluate_snr(
    model,
    snr_db,
    phase_mode,
    h_hat_mode,
    dmrs_freq_spacing,
    dmrs_freq_offset,
    num_samples,
    batch_size,
    device,
):
    dataset = OFDMDataset(
        num_samples=num_samples,
        snr_db_min=snr_db,
        snr_db_max=snr_db,
        h_hat_mode=h_hat_mode,
        phase_mode=phase_mode,
        dmrs_freq_spacing=dmrs_freq_spacing,
        dmrs_freq_offset=dmrs_freq_offset,
        seed=777000 + int(100 * snr_db),
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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


def parse_snr_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


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
    parser.add_argument("--phase_mode", type=str, default="uniform",
                        choices=["fixed", "narrow", "uniform"])
    parser.add_argument("--h_hat_mode", type=str, default="dmrs_ls_interp",
                        choices=["oracle_noisy", "dmrs_ls_interp"])
    parser.add_argument("--dmrs_freq_spacing", type=int, default=1)
    parser.add_argument("--dmrs_freq_offset", type=int, default=0)
    parser.add_argument("--snr_list", type=str, default="-10,-8,-6,-4,-2, 0,2,4,6,8,10,12,14,16,18,20")
    parser.add_argument("--num_samples", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")

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

    parser.add_argument("--out_csv", type=str, default="ber_results.csv")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

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
    model.load_state_dict(ckpt["model_state"])

    snr_values = parse_snr_list(args.snr_list)

    rows = []
    for snr in snr_values:
        loss, ber = evaluate_snr(
            model=model,
            snr_db=snr,
            phase_mode=args.phase_mode,
            h_hat_mode=args.h_hat_mode,
            dmrs_freq_spacing=args.dmrs_freq_spacing,
            dmrs_freq_offset=args.dmrs_freq_offset,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=device,
        )

        print(f"SNR {snr:5.1f} dB | BCE {loss:.6f} | BER {ber:.6e}")
        rows.append({"snr_db": snr, "bce": loss, "ber": ber})

    out_path = Path(args.out_csv)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["snr_db", "bce", "ber"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV to {out_path}")


if __name__ == "__main__":
    main()
