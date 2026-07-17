"""Measure Conditional-Amplitude-SwiGLU distribution shift across channels."""

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from data import (
    Sionna5GLDPCBatchGenerator,
    SionnaLDPC5GConfig,
    SionnaOFDMConfig,
    legacy_channel_profile,
    load_channel_profile,
)
from models.deeprx import ConditionalAmplitudeSwiGLUGate
from utils.batching import batch_sizes
from utils.checkpoints import load_receiver_checkpoint


class RunningMoments:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, value):
        value = value.detach().to(torch.float64)
        self.count += value.numel()
        self.total += value.sum().item()
        self.total_square += value.square().sum().item()
        self.minimum = min(self.minimum, value.min().item())
        self.maximum = max(self.maximum, value.max().item())

    @property
    def mean(self):
        return self.total / self.count

    @property
    def std(self):
        variance = self.total_square / self.count - self.mean**2
        return math.sqrt(max(variance, 0.0))

    @property
    def rms(self):
        return math.sqrt(self.total_square / self.count)


class GateAccumulator:
    def __init__(self):
        self.abs_z = RunningMoments()
        self.condition = RunningMoments()
        self.gate = RunningMoments()
        self.value = RunningMoments()
        self.residual = RunningMoments()
        self.multiplier = RunningMoments()
        self.output_abs = RunningMoments()
        self.residual_gt_010 = 0
        self.residual_gt_025 = 0
        self.residual_gt_050 = 0
        self.multiplier_nonpositive = 0

    def update(self, module, inputs, output):
        z, x_p_features = inputs
        abs_z = torch.abs(z)
        raw_gate, value = module.proj(abs_z).chunk(2, dim=1)
        condition = module.condition_net(x_p_features)
        gate = raw_gate + condition
        residual = module.residual_scale * F.silu(gate) * value
        multiplier = 1.0 + residual

        self.abs_z.update(abs_z)
        self.condition.update(condition)
        self.gate.update(gate)
        self.value.update(value)
        self.residual.update(residual)
        self.multiplier.update(multiplier)
        self.output_abs.update(torch.abs(output))
        abs_residual = residual.detach().abs()
        self.residual_gt_010 += int((abs_residual > 0.10).sum().item())
        self.residual_gt_025 += int((abs_residual > 0.25).sum().item())
        self.residual_gt_050 += int((abs_residual > 0.50).sum().item())
        self.multiplier_nonpositive += int((multiplier.detach() <= 0).sum().item())

    def row(self):
        count = self.residual.count
        return {
            "count": count,
            "abs_z_mean": self.abs_z.mean,
            "abs_z_std": self.abs_z.std,
            "condition_mean": self.condition.mean,
            "condition_std": self.condition.std,
            "gate_mean": self.gate.mean,
            "gate_std": self.gate.std,
            "value_mean": self.value.mean,
            "value_std": self.value.std,
            "residual_mean": self.residual.mean,
            "residual_std": self.residual.std,
            "residual_rms": self.residual.rms,
            "multiplier_mean": self.multiplier.mean,
            "multiplier_std": self.multiplier.std,
            "multiplier_min": self.multiplier.minimum,
            "multiplier_max": self.multiplier.maximum,
            "frac_abs_residual_gt_010": self.residual_gt_010 / count,
            "frac_abs_residual_gt_025": self.residual_gt_025 / count,
            "frac_abs_residual_gt_050": self.residual_gt_050 / count,
            "frac_multiplier_nonpositive": self.multiplier_nonpositive / count,
            "output_abs_rms": self.output_abs.rms,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_channel_profile")
    parser.add_argument("--eval_component_id")
    parser.add_argument("--ebno_db", type=float, default=6.0)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--coderate", type=float, default=0.5)
    parser.add_argument("--phase_mode", default="uniform")
    parser.add_argument("--seed", type=int, default=985000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_csv", required=True)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_receiver_checkpoint(
        args.checkpoint,
        device,
        bits_per_symbol=2,
        required_data_backend="sionna",
    )
    config = SionnaOFDMConfig(**checkpoint["sionna_config"])
    train_profile = checkpoint.get("train_channel_profile")
    if train_profile is None:
        train_profile = legacy_channel_profile(config)
    else:
        train_profile = load_channel_profile(train_profile)
    if args.eval_channel_profile:
        eval_profile = load_channel_profile(
            args.eval_channel_profile, component_id=args.eval_component_id
        )
    else:
        eval_profile = train_profile

    generator = Sionna5GLDPCBatchGenerator(
        config,
        ldpc_config=SionnaLDPC5GConfig(coderate=args.coderate),
        ebno_db_min=args.ebno_db,
        ebno_db_max=args.ebno_db,
        phase_mode=args.phase_mode,
        seed=args.seed,
        device=device,
        channel_profile=eval_profile,
    )
    accumulators = {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, ConditionalAmplitudeSwiGLUGate):
            accumulator = GateAccumulator()
            accumulators[name] = accumulator
            handles.append(module.register_forward_hook(accumulator.update))
    if not accumulators:
        raise ValueError("Checkpoint model has no ConditionalAmplitudeSwiGLUGate.")

    model.eval()
    for current_batch_size in batch_sizes(args.num_samples, args.batch_size):
        batch = generator.generate_batch(current_batch_size)
        model(batch["Y"], batch["H_hat"], batch["P"], batch["N0"])
    for handle in handles:
        handle.remove()

    rows = []
    for name, accumulator in accumulators.items():
        row = {
            "model": checkpoint["model_name"],
            "train_profile": train_profile["name"],
            "test_profile": eval_profile["name"],
            "layer": name,
            "ebno_db": args.ebno_db,
            "num_samples": args.num_samples,
            "residual_scale": float(
                dict(model.named_modules())[name].residual_scale.detach().item()
            ),
        }
        row.update(accumulator.row())
        rows.append(row)
        print(
            f"{name:24s} | |z| {row['abs_z_mean']:.4f} +/- "
            f"{row['abs_z_std']:.4f} | mult {row['multiplier_mean']:.4f} +/- "
            f"{row['multiplier_std']:.4f} | |delta|>0.25 "
            f"{row['frac_abs_residual_gt_025']:.4%}"
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV to {out_path}")


if __name__ == "__main__":
    main()
