import argparse
import math

import torch
from torch.utils.data import DataLoader

from data import OFDMDataset
from train import build_model, move_batch
from utils.metrics import max_mean_abs_diff


@torch.no_grad()
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
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--h_hat_mode", type=str, default="oracle_noisy",
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

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    dataset = OFDMDataset(
        num_samples=args.batch_size * args.num_batches,
        h_hat_mode=args.h_hat_mode,
        phase_mode="fixed",
        dmrs_freq_spacing=args.dmrs_freq_spacing,
        dmrs_freq_offset=args.dmrs_freq_offset,
        seed=12345,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

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

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])

    model.eval()

    max_diffs = []
    mean_diffs = []

    for batch in loader:
        batch = move_batch(batch, device)

        Y = batch["Y"]
        H_hat = batch["H_hat"]

        b = Y.shape[0]
        phi = 2.0 * math.pi * torch.rand(b, 1, 1, device=device)
        rot = torch.cos(phi) + 1j * torch.sin(phi)
        rot = rot.to(torch.complex64)

        Y_rot = rot * Y
        H_rot = rot * H_hat

        out = model(Y, H_hat, batch["P"], batch["N0"])
        out_rot = model(Y_rot, H_rot, batch["P"], batch["N0"])

        max_d, mean_d = max_mean_abs_diff(out, out_rot)
        max_diffs.append(max_d)
        mean_diffs.append(mean_d)

    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint if args.checkpoint else '[random init]'}")
    print(f"Global phase invariance test over {args.num_batches} batches")
    print(f"max |ΔLLR|  = {max(max_diffs):.8e}")
    print(f"mean |ΔLLR| = {sum(mean_diffs) / len(mean_diffs):.8e}")

    if args.model in ["phase_invariant", "single_branch"]:
        print("Expected: near numerical precision, e.g. around 1e-6 to 1e-5.")
    elif args.model == "physical_cnn":
        print("Expected: near numerical precision because input features are phase invariant.")
    else:
        print("Expected: usually non-zero, because ordinary real/imag CNN is not structurally invariant.")


if __name__ == "__main__":
    main()
