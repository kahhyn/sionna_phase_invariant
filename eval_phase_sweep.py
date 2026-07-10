import argparse
import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import OFDMDataset
from train import build_model, move_batch
from utils.metrics import masked_bce_with_logits, masked_ber


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def load_receiver(
    model_name,
    checkpoint,
    device,
    hidden,
    hidden_complex,
    zero_complex,
    branch_layers,
    kernel_size,
    use_norm,
    gate_type,
    single_readout_mode,
):
    model = build_model(
        model_name,
        bits_per_symbol=2,
        hidden=hidden,
        hidden_complex=hidden_complex,
        zero_complex=zero_complex,
        branch_layers=branch_layers,
        kernel_size=kernel_size,
        use_norm=use_norm,
        gate_type=gate_type,
        single_readout_mode=single_readout_mode,
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def expand_loss_mask(mask, target):
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.size(1) == 1 and target.size(1) != 1:
        mask = mask.expand(-1, target.size(1), -1, -1)
    return mask.to(device=target.device, dtype=torch.bool)


def masked_mean(x, mask):
    mask = expand_loss_mask(mask, x)
    return x[mask].mean()


def masked_max(x, mask):
    mask = expand_loss_mask(mask, x)
    return x[mask].max()


def make_fixed_batch(
    snr_db,
    phase_mode,
    h_hat_mode,
    dmrs_freq_spacing,
    dmrs_freq_offset,
    batch_size,
    seed,
    device,
):
    dataset = OFDMDataset(
        num_samples=batch_size,
        snr_db_min=snr_db,
        snr_db_max=snr_db,
        h_hat_mode=h_hat_mode,
        phase_mode=phase_mode,
        dmrs_freq_spacing=dmrs_freq_spacing,
        dmrs_freq_offset=dmrs_freq_offset,
        seed=seed + int(100 * snr_db),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return move_batch(next(iter(loader)), device)


@torch.no_grad()
def sweep_one_model(model, batch, phi_values):
    rows = []
    ref_logits = None
    ref_hard = None

    for phi in phi_values:
        rot = torch.exp(torch.tensor(1j * phi, dtype=batch["Y"].dtype, device=batch["Y"].device))
        logits = model(rot * batch["Y"], rot * batch["H_hat"], batch["P"], batch["N0"])

        if ref_logits is None:
            ref_logits = logits.detach()
            ref_hard = ref_logits > 0

        delta = (logits - ref_logits).abs()
        delta_sq = (logits - ref_logits).pow(2)
        hard = logits > 0
        sign_flip = (hard != ref_hard).float()
        loss = masked_bce_with_logits(logits, batch["bits"], batch["loss_mask"])
        ber = masked_ber(logits, batch["bits"], batch["loss_mask"])
        llr_mse = masked_mean(delta_sq, batch["loss_mask"])

        rows.append(
            {
                "phi_rad": phi,
                "phi_deg": phi * 180.0 / math.pi,
                "bce": loss.item(),
                "ber": ber.item(),
                "llr_mse": llr_mse.item(),
                "llr_rmse": torch.sqrt(llr_mse).item(),
                "max_abs_llr_delta": masked_max(delta, batch["loss_mask"]).item(),
                "mean_abs_llr_delta": masked_mean(delta, batch["loss_mask"]).item(),
                "sign_flip_rate": masked_mean(sign_flip, batch["loss_mask"]).item(),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase_checkpoint", type=str, default="runs/debug/best.pt")
    parser.add_argument("--nointer_checkpoint", type=str, default="runs/debug/nointer/best.pt")
    parser.add_argument("--single_checkpoint", type=str, default="")
    parser.add_argument("--snr_list", type=str, default="0,4,8,12,16,20")
    parser.add_argument("--phi_list", type=str, default="0,0.5235987756,1.0471975512,1.5707963268,2.0943951024,3.1415926536,4.1887902048,4.7123889804,5.235987756,6.2831853072")
    parser.add_argument("--phase_mode", type=str, default="uniform",
                        choices=["fixed", "narrow", "uniform"])
    parser.add_argument("--h_hat_mode", type=str, default="dmrs_ls_interp",
                        choices=["oracle_noisy", "dmrs_ls_interp"])
    parser.add_argument("--dmrs_freq_spacing", type=int, default=1)
    parser.add_argument("--dmrs_freq_offset", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=888000)
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

    parser.add_argument("--out_csv", type=str, default="phase_sweep_results.csv")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    use_norm = not args.no_norm

    models = {
        "phase_invariant": load_receiver(
            "phase_invariant",
            args.phase_checkpoint,
            device,
            args.hidden,
            args.hidden_complex,
            args.zero_complex,
            args.branch_layers,
            args.kernel_size,
            use_norm,
            args.gate_type,
            args.single_readout_mode,
        ),
        "complex_no_interaction": load_receiver(
            "complex_no_interaction",
            args.nointer_checkpoint,
            device,
            args.hidden,
            args.hidden_complex,
            args.zero_complex,
            args.branch_layers,
            args.kernel_size,
            use_norm,
            args.gate_type,
            args.single_readout_mode,
        ),
    }
    if args.single_checkpoint:
        models["single_branch"] = load_receiver(
            "single_branch",
            args.single_checkpoint,
            device,
            args.hidden,
            args.hidden_complex,
            args.zero_complex,
            args.branch_layers,
            args.kernel_size,
            use_norm,
            args.gate_type,
            args.single_readout_mode,
        )

    snr_values = parse_float_list(args.snr_list)
    phi_values = parse_float_list(args.phi_list)

    all_rows = []
    for snr in snr_values:
        batch = make_fixed_batch(
            snr_db=snr,
            phase_mode=args.phase_mode,
            h_hat_mode=args.h_hat_mode,
            dmrs_freq_spacing=args.dmrs_freq_spacing,
            dmrs_freq_offset=args.dmrs_freq_offset,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
        )

        print(f"\nSNR {snr:.1f} dB | fixed batch size {args.batch_size} | phase_mode={args.phase_mode}")
        for model_name, model in models.items():
            rows = sweep_one_model(model, batch, phi_values)
            print(f"  {model_name}")
            for row in rows:
                print(
                    "    phi {phi_deg:7.2f} deg | BCE {bce:.6f} | BER {ber:.6e} | "
                    "LLR_MSE {llr_mse:.6e} | LLR_RMSE {llr_rmse:.6e} | "
                    "mean_dLLR {mean_abs_llr_delta:.6e} | sign_flip {sign_flip_rate:.6e}".format(**row)
                )
                all_rows.append(
                    {
                        "model": model_name,
                        "snr_db": snr,
                        **row,
                    }
                )

    out_path = Path(args.out_csv)
    with out_path.open("w", newline="") as f:
        fieldnames = [
            "model",
            "snr_db",
            "phi_rad",
            "phi_deg",
            "bce",
            "ber",
            "llr_mse",
            "llr_rmse",
            "max_abs_llr_delta",
            "mean_abs_llr_delta",
            "sign_flip_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved CSV to {out_path}")


if __name__ == "__main__":
    main()
