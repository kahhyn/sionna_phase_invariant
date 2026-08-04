"""Evaluate SU-MIMO neural receivers with one 5G NR LDPC codeword per layer."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from data import (
    Sionna5GLDPCSUMIMOBatchGenerator,
    SionnaLDPC5GConfig,
    legacy_channel_profile,
    load_channel_profile,
    profile_backend_label,
    profile_component,
)
from evaluation.eval_ber_su_mimo import wilson_interval
from utils.checkpoints import load_su_mimo_checkpoint


def parse_float_list(text):
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("--ebno_list must contain at least one value.")
    return values


@torch.no_grad()
def evaluate_ebno(
    model,
    config,
    channel_profile,
    ldpc_config,
    ebno_db,
    phase_mode,
    batch_size,
    target_block_errors,
    max_blocks,
    seed,
    device,
):
    generator = Sionna5GLDPCSUMIMOBatchGenerator(
        config,
        ldpc_config=ldpc_config,
        ebno_db_min=ebno_db,
        ebno_db_max=ebno_db,
        phase_mode=phase_mode,
        seed=seed,
        device=device,
        channel_profile=channel_profile,
    )
    model.eval()
    num_frames = 0
    frame_errors = 0
    info_bit_errors = 0
    coded_bit_errors = 0
    coded_bce_sum = 0.0
    num_batches = 0
    layer_frame_errors = torch.zeros(config.num_layers, dtype=torch.int64)
    layer_info_errors = torch.zeros(config.num_layers, dtype=torch.int64)

    while num_frames < max_blocks and frame_errors < target_block_errors:
        current_batch_size = min(batch_size, max_blocks - num_frames)
        batch = generator.generate_batch(current_batch_size)
        logits = model(
            batch["Y"],
            batch["H_hat"],
            batch["P"],
            batch["N0"],
            batch["layer_mask"],
        )
        codeword_logits = generator.extract_codeword_logits(logits)
        info_hat = generator.decode_logits(logits)
        info_errors = info_hat.bool() != batch["info_bits"].bool()
        coded_errors = (codeword_logits > 0) != batch["codeword_bits"].bool()
        layer_block_error = info_errors.any(dim=-1)
        user_frame_error = layer_block_error.any(dim=-1)

        frame_errors += int(user_frame_error.sum().item())
        info_bit_errors += int(info_errors.sum().item())
        coded_bit_errors += int(coded_errors.sum().item())
        coded_bce_sum += float(
            F.binary_cross_entropy_with_logits(
                codeword_logits,
                batch["codeword_bits"],
                reduction="sum",
            ).item()
        )
        layer_frame_errors += layer_block_error.sum(dim=0).cpu()
        layer_info_errors += info_errors.sum(dim=(0, 2)).cpu()
        num_frames += current_batch_size
        num_batches += 1

    total_info_bits = num_frames * config.num_layers * generator.k
    total_coded_bits = num_frames * config.num_layers * generator.n
    result = {
        "ebno_db": ebno_db,
        "bler": frame_errors / num_frames,
        "post_ldpc_ber": info_bit_errors / total_info_bits,
        "pre_ldpc_coded_ber": coded_bit_errors / total_coded_bits,
        "coded_bce": coded_bce_sum / total_coded_bits,
        "block_errors": frame_errors,
        "num_blocks": num_frames,
        "num_layer_blocks": num_frames * config.num_layers,
        "info_bit_errors": info_bit_errors,
        "coded_bit_errors": coded_bit_errors,
        "k_per_layer": generator.k,
        "n_per_layer": generator.n,
        "coderate": generator.coderate,
        "decoder_iterations": ldpc_config.num_iter,
        "target_reached": int(frame_errors >= target_block_errors),
        "num_batches": num_batches,
        "per_layer": [],
    }
    for layer in range(config.num_layers):
        errors = int(layer_frame_errors[layer].item())
        result["per_layer"].append(
            {
                "layer": layer,
                "bler": errors / num_frames,
                "block_errors": errors,
                "num_blocks": num_frames,
                "post_ldpc_ber": int(layer_info_errors[layer].item())
                / (num_frames * generator.k),
                "info_bit_errors": int(layer_info_errors[layer].item()),
            }
        )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ebno_list", default="-5,-3,-1,0,1,2,3,4,5,6,7,8")
    parser.add_argument("--coderate", type=float, default=0.5)
    parser.add_argument("--decoder_iterations", type=int, default=20)
    parser.add_argument("--cn_update", default="boxplus-phi")
    parser.add_argument(
        "--phase_mode", default="uniform", choices=["fixed", "narrow", "uniform"]
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--target_block_errors",
        type=int,
        default=100,
        help="Stop after this many user-frame errors; any failed layer fails a frame.",
    )
    parser.add_argument(
        "--max_blocks",
        type=int,
        default=20000,
        help="Maximum number of user frames, each containing one codeword per layer.",
    )
    parser.add_argument("--seed", type=int, default=777000)
    parser.add_argument("--common_random_numbers", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_csv", default="su_mimo_ldpc_bler.csv")
    parser.add_argument(
        "--out_layer_csv", help="Defaults to <out_csv stem>_per_layer.csv."
    )
    parser.add_argument(
        "--eval_channel_profile",
        help="Existing profile JSON overriding the checkpoint training profile.",
    )
    parser.add_argument(
        "--eval_component_id",
        help="Evaluate one component from --eval_channel_profile.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size <= 0 or args.max_blocks <= 0:
        raise ValueError("batch_size and max_blocks must be positive.")
    if args.target_block_errors <= 0:
        raise ValueError("target_block_errors must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in PyTorch.")
    model, config, checkpoint = load_su_mimo_checkpoint(args.checkpoint, device)
    ldpc_config = SionnaLDPC5GConfig(
        coderate=args.coderate,
        num_iter=args.decoder_iterations,
        cn_update=args.cn_update,
    )

    train_profile = checkpoint.get("train_channel_profile")
    train_profile = (
        legacy_channel_profile(config)
        if train_profile is None
        else load_channel_profile(train_profile)
    )
    if args.eval_channel_profile:
        eval_profile = load_channel_profile(
            args.eval_channel_profile, component_id=args.eval_component_id
        )
    elif args.eval_component_id:
        raise ValueError("--eval_component_id requires --eval_channel_profile")
    else:
        eval_profile = train_profile
    fixed_component = profile_component(eval_profile)
    backend = profile_backend_label(eval_profile)
    scenario = fixed_component.get("backend", "mixed") if fixed_component else "mixed"
    tdl_model = fixed_component.get("tdl_model", "") if fixed_component else ""
    delay_ns = (
        fixed_component.get("delay_spread_ns", "") if fixed_component else ""
    )
    print(
        f"Train profile: {train_profile['name']} | "
        f"test profile: {eval_profile['name']} | backend: {backend}"
    )

    rows = []
    layer_rows = []
    for index, ebno_db in enumerate(parse_float_list(args.ebno_list)):
        eval_seed = args.seed if args.common_random_numbers else args.seed + index * 1000
        result = evaluate_ebno(
            model,
            config,
            eval_profile,
            ldpc_config,
            ebno_db,
            args.phase_mode,
            args.batch_size,
            args.target_block_errors,
            args.max_blocks,
            eval_seed,
            device,
        )
        ci_low, ci_high = wilson_interval(
            result["block_errors"], result["num_blocks"]
        )
        common = {
            "ebno_db": ebno_db,
            "model": checkpoint["model_name"],
            "train_seed": checkpoint["args"]["seed"],
            "eval_seed": eval_seed,
            "common_random_numbers": int(args.common_random_numbers),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_epoch": checkpoint["epoch"],
            "phase_mode": args.phase_mode,
            "num_layers": config.num_layers,
            "num_rx_ant": config.num_rx_ant,
            "total_tx_power": config.total_tx_power,
            "train_profile": train_profile["name"],
            "test_profile": eval_profile["name"],
            "backend": backend,
            "scenario": scenario,
            "tdl_model": tdl_model,
            "delay_ns": delay_ns,
        }
        row = {key: value for key, value in result.items() if key != "per_layer"}
        row.update(common)
        row["bler_ci95_low"] = ci_low
        row["bler_ci95_high"] = ci_high
        rows.append(row)
        for layer_result in result["per_layer"]:
            layer_low, layer_high = wilson_interval(
                layer_result["block_errors"], layer_result["num_blocks"]
            )
            layer_rows.append(
                {
                    **common,
                    **layer_result,
                    "bler_ci95_low": layer_low,
                    "bler_ci95_high": layer_high,
                }
            )
        print(
            f"Eb/N0 {ebno_db:5.1f} dB | frame BLER {result['bler']:.6e} | "
            f"post-BER {result['post_ldpc_ber']:.6e} | "
            f"pre-BER {result['pre_ldpc_coded_ber']:.6e} | "
            f"errors {result['block_errors']}/{result['num_blocks']}"
        )

    out_path = Path(args.out_csv)
    layer_path = (
        Path(args.out_layer_csv)
        if args.out_layer_csv
        else out_path.with_name(f"{out_path.stem}_per_layer{out_path.suffix}")
    )
    for path, data in ((out_path, rows), (layer_path, layer_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
    print(f"Saved aggregate CSV to {out_path}")
    print(f"Saved per-layer CSV to {layer_path}")


if __name__ == "__main__":
    main()
