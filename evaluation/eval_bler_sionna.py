"""Evaluate trained neural receivers with 5G NR LDPC decoding."""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from data import (
    Sionna5GLDPCBatchGenerator,
    SionnaLDPC5GConfig,
    SionnaOFDMConfig,
)
from models.classical_receivers import SionnaLMMSEBaseline
from utils.checkpoints import load_receiver_checkpoint


def parse_float_list(text):
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def load_receiver(checkpoint_path, device):
    return load_receiver_checkpoint(
        checkpoint_path,
        device,
        bits_per_symbol=2,
        required_data_backend="sionna",
    )


def load_ofdm_config(checkpoint_path, device):
    if checkpoint_path is None:
        return SionnaOFDMConfig(), None
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if checkpoint.get("data_backend") != "sionna":
        raise ValueError(
            f"Checkpoint data_backend={checkpoint.get('data_backend')!r}, "
            "expected 'sionna'."
        )
    return SionnaOFDMConfig(**checkpoint["sionna_config"]), checkpoint


@torch.no_grad()
def evaluate_ebno(
    receiver,
    receiver_type,
    ofdm_config,
    ldpc_config,
    ebno_db,
    phase_mode,
    batch_size,
    target_block_errors,
    max_blocks,
    seed,
    device,
):
    generator = Sionna5GLDPCBatchGenerator(
        ofdm_config,
        ldpc_config=ldpc_config,
        ebno_db_min=ebno_db,
        ebno_db_max=ebno_db,
        phase_mode=phase_mode,
        seed=seed,
        device=device,
    )

    num_blocks = 0
    block_errors = 0
    info_bit_errors = 0
    coded_bit_errors = 0
    coded_bce_sum = 0.0
    num_batches = 0
    classical_receiver = None
    if receiver_type == "lmmse_ls":
        classical_receiver = SionnaLMMSEBaseline(generator, csi="ls")
    elif receiver_type == "lmmse_perfect":
        classical_receiver = SionnaLMMSEBaseline(generator, csi="perfect")

    while num_blocks < max_blocks and block_errors < target_block_errors:
        current_batch_size = min(batch_size, max_blocks - num_blocks)
        batch = generator.generate_batch(current_batch_size)
        if receiver_type == "neural":
            logits = receiver(batch["Y"], batch["H_hat"], batch["P"], batch["N0"])
            codeword_logits = generator.extract_codeword_logits(logits)
        elif receiver_type == "lmmse_ls":
            codeword_logits = classical_receiver.codeword_logits(batch)
        elif receiver_type == "lmmse_perfect":
            codeword_logits = classical_receiver.codeword_logits(batch)
        else:
            raise ValueError(f"Unknown receiver_type: {receiver_type}")
        info_hat = generator.decoder(codeword_logits)

        info_errors = info_hat.bool() != batch["info_bits"].bool()
        coded_errors = (
            codeword_logits > 0
        ) != batch["codeword_bits"].bool()

        block_errors += int(info_errors.any(dim=-1).sum().item())
        info_bit_errors += int(info_errors.sum().item())
        coded_bit_errors += int(coded_errors.sum().item())
        coded_bce_sum += float(
            F.binary_cross_entropy_with_logits(
                codeword_logits,
                batch["codeword_bits"],
                reduction="sum",
            ).item()
        )
        num_blocks += current_batch_size
        num_batches += 1

    return {
        "ebno_db": ebno_db,
        "bler": block_errors / num_blocks,
        "post_ldpc_ber": info_bit_errors / (num_blocks * generator.k),
        "pre_ldpc_coded_ber": coded_bit_errors / (num_blocks * generator.n),
        "coded_bce": coded_bce_sum / (num_blocks * generator.n),
        "block_errors": block_errors,
        "num_blocks": num_blocks,
        "info_bit_errors": info_bit_errors,
        "coded_bit_errors": coded_bit_errors,
        "k": generator.k,
        "n": generator.n,
        "coderate": generator.coderate,
        "decoder_iterations": ldpc_config.num_iter,
        "target_reached": int(block_errors >= target_block_errors),
        "num_batches": num_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--receiver",
        default="neural",
        choices=["neural", "lmmse_ls", "lmmse_perfect"],
        help=(
            "Receiver to evaluate. Neural receivers require --checkpoint. "
            "LMMSE receivers use --checkpoint only to reuse the saved OFDM config."
        ),
    )
    parser.add_argument("--ebno_list", default="-5,-3,-1,0,1,2,3,4,5,6,7,8")
    parser.add_argument("--coderate", type=float, default=0.5)
    parser.add_argument("--decoder_iterations", type=int, default=20)
    parser.add_argument("--cn_update", default="boxplus-phi")
    parser.add_argument("--phase_mode", default="uniform", choices=["fixed", "narrow", "uniform"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_block_errors", type=int, default=100)
    parser.add_argument("--max_blocks", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=777000)
    parser.add_argument("--common_random_numbers", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_csv", default="sionna_ldpc_bler.csv")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.max_blocks <= 0:
        raise ValueError("batch_size and max_blocks must be positive.")
    if args.target_block_errors <= 0:
        raise ValueError("target_block_errors must be positive.")
    if args.receiver == "neural" and args.checkpoint is None:
        raise ValueError("--checkpoint is required for --receiver neural.")

    device = torch.device(args.device)
    if args.receiver == "neural":
        model, checkpoint = load_receiver(args.checkpoint, device)
        ofdm_config = SionnaOFDMConfig(**checkpoint["sionna_config"])
        model.eval()
    else:
        model = None
        ofdm_config, checkpoint = load_ofdm_config(args.checkpoint, device)
    ldpc_config = SionnaLDPC5GConfig(
        coderate=args.coderate,
        num_iter=args.decoder_iterations,
        cn_update=args.cn_update,
    )

    rows = []
    for index, ebno_db in enumerate(parse_float_list(args.ebno_list)):
        eval_seed = args.seed if args.common_random_numbers else args.seed + index * 1000
        row = evaluate_ebno(
            model,
            args.receiver,
            ofdm_config,
            ldpc_config,
            ebno_db,
            args.phase_mode,
            args.batch_size,
            args.target_block_errors,
            args.max_blocks,
            eval_seed,
            device,
        )
        model_name = args.receiver
        train_seed = -1
        if args.receiver == "neural":
            model_name = checkpoint["model_name"]
            train_seed = int(checkpoint["args"].get("seed", -1))
        row.update(
            {
                "model": model_name,
                "train_seed": train_seed,
                "eval_seed": eval_seed,
                "common_random_numbers": int(args.common_random_numbers),
            }
        )
        print(
            f"Eb/N0 {ebno_db:5.1f} dB | BLER {row['bler']:.6e} | "
            f"post-BER {row['post_ldpc_ber']:.6e} | "
            f"pre-BER {row['pre_ldpc_coded_ber']:.6e} | "
            f"errors {row['block_errors']}/{row['num_blocks']}"
        )
        rows.append(row)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV to {out_path}")


if __name__ == "__main__":
    main()
