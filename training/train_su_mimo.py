"""Train the common-phase-invariant receiver on fixed-topology SU-MIMO data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from data import (
    PROFILE_SCHEMA_VERSION,
    SionnaSUMIMOBatchGenerator,
    SionnaSUMIMOConfig,
    channel_profile_hash,
    filter_channel_profile,
    legacy_channel_profile,
    load_channel_profile,
)
from models import SU_MIMO_MODEL_CHOICES, build_su_mimo_model
from utils.batching import batch_sizes
from utils.checkpoints import load_su_mimo_checkpoint
from utils.metrics import masked_bce_sum, masked_bce_with_logits, masked_error_count


TDL_MIX_REFERENCE_PARAMETERS = 204599
HISTORY_FIELDS = [
    "epoch",
    "global_step",
    "data_pass",
    "samples_seen",
    "lr",
    "train_bce",
    "train_ber",
    "train_bit_errors",
    "train_valid_bits",
    "val_bce",
    "val_ber",
    "val_bit_errors",
    "val_valid_bits",
]
RESUME_RUNTIME_ARGUMENTS = {
    "resume_checkpoint",
    "epochs",
    "train_steps",
    "save_dir",
    "device",
    "log_interval",
}


class FixedDatasetReplay:
    """Regenerate a fixed finite set of batches without retaining it in memory.

    Every dataset shard has a stable seed and size. Dataset passes may reorder
    shards, but a shard always regenerates the exact same samples. This keeps a
    16-Rx finite corpus practical while making the data identical across models.
    """

    _SHARD_SEED_STRIDE = 1_000_003
    _ORDER_SEED_OFFSET = 2_000_003

    def __init__(
        self,
        generator,
        num_samples,
        batch_size,
        seed,
        *,
        shuffle_batches=True,
    ):
        if num_samples <= 0 or batch_size <= 0:
            raise ValueError("num_samples and batch_size must be positive")
        self.generator = generator
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle_batches = bool(shuffle_batches)
        self.shard_sizes = tuple(batch_sizes(self.num_samples, self.batch_size))
        self.shard_seeds = tuple(
            self.seed + self._SHARD_SEED_STRIDE * (index + 1)
            for index in range(len(self.shard_sizes))
        )

    @property
    def steps_per_pass(self):
        return len(self.shard_sizes)

    def shard_order(self, data_pass):
        if data_pass < 0:
            raise ValueError("data_pass must be non-negative")
        if not self.shuffle_batches or self.steps_per_pass == 1:
            return tuple(range(self.steps_per_pass))
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.seed + self._ORDER_SEED_OFFSET + int(data_pass))
        return tuple(torch.randperm(self.steps_per_pass, generator=rng).tolist())

    def batch_for_step(self, zero_based_step):
        if zero_based_step < 0:
            raise ValueError("zero_based_step must be non-negative")
        data_pass, offset = divmod(int(zero_based_step), self.steps_per_pass)
        shard_index = self.shard_order(data_pass)[offset]
        self.generator.reset(self.shard_seeds[shard_index])
        batch = self.generator.generate_batch(self.shard_sizes[shard_index])
        return batch, data_pass, shard_index

    def samples_seen(self, completed_steps):
        """Count sample presentations after ``completed_steps`` optimizer steps."""
        if completed_steps < 0:
            raise ValueError("completed_steps must be non-negative")
        passes, offset = divmod(int(completed_steps), self.steps_per_pass)
        total = passes * self.num_samples
        if offset:
            order = self.shard_order(passes)
            total += sum(self.shard_sizes[index] for index in order[:offset])
        return total


def _seed_runtime(seed, device, deterministic_algorithms=False):
    """Seed model initialization and configure optional deterministic CUDA ops."""
    seed = int(seed)
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic_algorithms)
    torch.use_deterministic_algorithms(bool(deterministic_algorithms))


def _model_state_hash(model):
    """Return a stable hash of model tensors for initialization provenance."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _fixed_dataset_hash(args, data_config, train_profile, train_seed):
    """Identify every input that determines a fixed generated corpus."""
    if args.train_dataset_mode != "fixed":
        return None
    identity = {
        "num_samples": args.num_train,
        "batch_size": args.batch_size,
        "generator_seed": int(train_seed),
        "shuffle_batches": args.fixed_dataset_shuffle_batches,
        "phase_mode": args.train_phase_mode,
        "snr_db_min": args.snr_db_min,
        "snr_db_max": args.snr_db_max,
        "data_config": data_config.to_dict(),
        "channel_profile_hash": channel_profile_hash(train_profile),
        "shard_seed_stride": FixedDatasetReplay._SHARD_SEED_STRIDE,
        "order_seed_offset": FixedDatasetReplay._ORDER_SEED_OFFSET,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def learning_rate_for_epoch(
    epoch,
    total_epochs,
    base_lr,
    scheduler_name="constant",
    lr_min=0.0,
    warmup_epochs=0,
    constant_tail_epochs=0,
):
    """Compute the optimizer LR for warmup, cosine decay, and a low-LR tail."""
    if not 1 <= epoch <= total_epochs:
        raise ValueError("epoch must be in [1, total_epochs].")
    if warmup_epochs < 0 or constant_tail_epochs < 0:
        raise ValueError("warmup_epochs and constant_tail_epochs must be non-negative.")
    if warmup_epochs + constant_tail_epochs >= total_epochs:
        raise ValueError(
            "warmup_epochs + constant_tail_epochs must be less than total_epochs."
        )
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    if scheduler_name == "constant":
        if constant_tail_epochs:
            raise ValueError("constant_tail_epochs requires the cosine scheduler.")
        return base_lr
    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported LR scheduler: {scheduler_name}")

    cosine_end = total_epochs - constant_tail_epochs
    if epoch > cosine_end:
        return lr_min
    decay_start = warmup_epochs + 1
    decay_span = cosine_end - decay_start
    if decay_span <= 0:
        return base_lr
    progress = (epoch - decay_start) / decay_span
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (base_lr - lr_min) * cosine


def learning_rate_for_step(
    step,
    total_steps,
    base_lr,
    scheduler_name="constant",
    lr_min=0.0,
    warmup_steps=0,
    constant_tail_steps=0,
):
    """Step-indexed counterpart used to compare finite datasets fairly."""
    return learning_rate_for_epoch(
        step,
        total_steps,
        base_lr,
        scheduler_name=scheduler_name,
        lr_min=lr_min,
        warmup_epochs=warmup_steps,
        constant_tail_epochs=constant_tail_steps,
    )


def _set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def build_data_config(args):
    return SionnaSUMIMOConfig(
        num_ofdm_symbols=args.num_ofdm_symbols,
        fft_size=args.fft_size,
        subcarrier_spacing_hz=args.subcarrier_spacing_hz,
        cyclic_prefix_length=args.cyclic_prefix_length,
        bits_per_symbol=2,
        num_layers=args.num_layers,
        num_rx_ant=args.num_rx_ant,
        total_tx_power=args.total_tx_power,
        dmrs_symbol_indices=tuple(args.dmrs_symbols),
        tdl_model=args.tdl_model,
        delay_spread_s=args.delay_spread_s,
        carrier_frequency_hz=args.carrier_frequency_hz,
        max_doppler_hz=args.max_doppler_hz,
        normalize_channel=not args.no_normalize_channel,
        ls_interpolation_type=args.ls_interpolation_type,
    )


def build_model_config(args):
    return {
        "num_rx_ant": args.num_rx_ant,
        "hidden_complex": args.hidden_complex,
        "zero_real": args.zero_real,
        "hidden_real": args.hidden_real,
        "bits_per_symbol": 2,
        "num_iterations": args.num_iterations,
        "kernel_size": args.kernel_size,
        "zero_gate_hidden": args.zero_gate_hidden,
    }


def build_generator(
    args,
    config,
    phase_mode,
    seed,
    device,
    channel_profile,
    snr_min=None,
    snr_max=None,
):
    return SionnaSUMIMOBatchGenerator(
        config,
        snr_db_min=args.snr_db_min if snr_min is None else snr_min,
        snr_db_max=args.snr_db_max if snr_max is None else snr_max,
        phase_mode=phase_mode,
        seed=seed,
        device=device,
        channel_profile=channel_profile,
    )


def _git_revision():
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{revision}-dirty" if dirty else revision
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_history(path, history):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def _aggregate_metrics(logits, batch):
    bce_sum, bce_count = masked_bce_sum(
        logits, batch["bits"], batch["loss_mask"]
    )
    errors, valid_bits = masked_error_count(
        logits, batch["bits"], batch["loss_mask"]
    )
    if int(bce_count.item()) != int(valid_bits.item()):
        raise RuntimeError("BCE and BER masks disagree.")
    return float(bce_sum.item()), int(errors.item()), int(valid_bits.item())


def train_one_batch(model, batch, optimizer):
    logits = model(
        batch["Y"],
        batch["H_hat"],
        batch["P"],
        batch["N0"],
        batch["layer_mask"],
    )
    loss = masked_bce_with_logits(logits, batch["bits"], batch["loss_mask"])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return _aggregate_metrics(logits.detach(), batch)


def train_one_epoch(model, generator, optimizer, num_samples, batch_size, log_interval):
    model.train()
    total_bce = 0.0
    total_errors = 0
    total_bits = 0
    for step, current_batch_size in enumerate(
        batch_sizes(num_samples, batch_size), start=1
    ):
        batch = generator.generate_batch(current_batch_size)
        bce_sum, errors, valid_bits = train_one_batch(model, batch, optimizer)
        total_bce += bce_sum
        total_errors += errors
        total_bits += valid_bits
        if log_interval > 0 and step % log_interval == 0:
            print(
                f"  step {step:05d} | BCE {total_bce / total_bits:.6f} | "
                f"BER {total_errors / total_bits:.6f}"
            )
    return total_bce / total_bits, total_errors / total_bits, total_errors, total_bits


def train_one_fixed_pass(model, replay, optimizer, data_pass, log_interval):
    model.train()
    total_bce = 0.0
    total_errors = 0
    total_bits = 0
    profile_counts = Counter()
    first_step = int(data_pass) * replay.steps_per_pass
    for offset in range(replay.steps_per_pass):
        batch, actual_pass, _ = replay.batch_for_step(first_step + offset)
        if actual_pass != data_pass:
            raise RuntimeError("Fixed dataset replay crossed a data-pass boundary.")
        bce_sum, errors, valid_bits = train_one_batch(model, batch, optimizer)
        profile_id = batch.get("channel_profile_id")
        if profile_id is not None:
            profile_counts[str(profile_id)] += 1
        total_bce += bce_sum
        total_errors += errors
        total_bits += valid_bits
        if log_interval > 0 and (offset + 1) % log_interval == 0:
            print(
                f"  step {offset + 1:05d} | BCE {total_bce / total_bits:.6f} | "
                f"BER {total_errors / total_bits:.6f}"
            )
    return (
        total_bce / total_bits,
        total_errors / total_bits,
        total_errors,
        total_bits,
        dict(profile_counts),
    )


@torch.no_grad()
def evaluate(model, generator, num_samples, batch_size, reset_seed=None):
    model.eval()
    generator.reset(reset_seed)
    total_bce = 0.0
    total_errors = 0
    total_bits = 0
    for current_batch_size in batch_sizes(num_samples, batch_size):
        batch = generator.generate_batch(current_batch_size)
        logits = model(
            batch["Y"],
            batch["H_hat"],
            batch["P"],
            batch["N0"],
            batch["layer_mask"],
        )
        bce_sum, errors, valid_bits = _aggregate_metrics(logits, batch)
        total_bce += bce_sum
        total_errors += errors
        total_bits += valid_bits
    return total_bce / total_bits, total_errors / total_bits, total_errors, total_bits


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="su_mimo_phase_invariant",
        choices=SU_MIMO_MODEL_CHOICES,
    )
    parser.add_argument(
        "--train_phase_mode", default="fixed", choices=["fixed", "narrow", "uniform"]
    )
    parser.add_argument(
        "--val_phase_mode", default="uniform", choices=["fixed", "narrow", "uniform"]
    )
    parser.add_argument(
        "--train_dataset_mode",
        default="streaming",
        choices=["streaming", "fixed"],
        help=(
            "streaming draws new samples every epoch; fixed defines exactly "
            "--num_train distinct samples and deterministically replays them."
        ),
    )
    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_val", type=int, default=2000)
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Total target epoch, including epochs already present when resuming.",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        help=(
            "Total optimizer-step budget. Requires --train_dataset_mode fixed "
            "and supersedes --epochs for stopping and LR scheduling."
        ),
    )
    parser.add_argument(
        "--validation_interval_steps",
        type=int,
        default=500,
        help="Validate/checkpoint every N optimizer steps in step-budget mode.",
    )
    parser.add_argument(
        "--fixed_dataset_shuffle_batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deterministically reorder fixed dataset shards on each data pass.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--snr_db_min", type=float, default=-5.0)
    parser.add_argument("--snr_db_max", type=float, default=20.0)

    parser.add_argument("--num_ofdm_symbols", type=int, default=14)
    parser.add_argument("--fft_size", type=int, default=72)
    parser.add_argument("--subcarrier_spacing_hz", type=float, default=30e3)
    parser.add_argument("--cyclic_prefix_length", type=int, default=0)
    parser.add_argument("--dmrs_symbols", type=int, nargs="+", default=[2, 11])
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_rx_ant", type=int, default=2)
    parser.add_argument("--total_tx_power", type=float, default=1.0)
    parser.add_argument(
        "--tdl_model", default="A", choices=["A", "B", "C", "D", "E"]
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
    parser.add_argument(
        "--train_channel_profile",
        help="Existing channel-profile JSON; legacy TDL arguments are used if omitted.",
    )
    parser.add_argument(
        "--val_channel_profile",
        help="Validation profile JSON; defaults to the training profile.",
    )
    parser.add_argument(
        "--train_component_ids",
        nargs="+",
        help="Only use these component IDs from the training profile.",
    )
    parser.add_argument(
        "--train_tdl_models",
        nargs="+",
        help="Only use these TDL model labels from the training profile.",
    )
    parser.add_argument(
        "--train_delay_spread_min_ns",
        type=float,
        help="Inclusive minimum TDL delay spread in the training profile.",
    )
    parser.add_argument(
        "--train_delay_spread_max_ns",
        type=float,
        help="Inclusive maximum TDL delay spread in the training profile.",
    )

    parser.add_argument("--hidden_complex", type=int, default=32)
    parser.add_argument("--zero_real", type=int, default=22)
    parser.add_argument("--hidden_real", type=int, default=66)
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--zero_gate_hidden", type=int, default=16)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        choices=["constant", "cosine"],
        help="Epoch- or optimizer-step-level learning-rate schedule.",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=1e-5,
        help="Final learning rate for the cosine schedule.",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=0,
        help="Linear warmup epochs before the constant/cosine schedule.",
    )
    parser.add_argument(
        "--constant_tail_epochs",
        type=int,
        default=0,
        help=(
            "Final epochs held at lr_min after cosine decay. This supports "
            "slow-convergence training without checkpoint-resume schedule changes."
        ),
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Linear warmup steps in --train_steps mode.",
    )
    parser.add_argument(
        "--constant_tail_steps",
        type=int,
        default=0,
        help="Final steps held at lr_min in --train_steps mode.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_generator_seed", type=int)
    parser.add_argument("--val_generator_seed", type=int)
    parser.add_argument(
        "--deterministic_algorithms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request deterministic PyTorch/CUDA algorithms for reproducible runs.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        help=(
            "Resume model, optimizer, data, and training settings. --epochs or "
            "--train_steps is the corresponding total target."
        ),
    )
    parser.add_argument(
        "--save_dir",
        help="Defaults to runs/su_mimo_debug, or the checkpoint directory on resume.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _restore_saved_arguments(args, checkpoint):
    saved_args = checkpoint.get("args", {})
    for key, value in saved_args.items():
        if key not in RESUME_RUNTIME_ARGUMENTS and hasattr(args, key):
            setattr(args, key, value)


def _validate_args(args, start_epoch, start_global_step=0):
    if args.num_train <= 0 or args.num_val <= 0:
        raise ValueError("num_train and num_val must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.train_steps is None:
        if args.epochs <= start_epoch:
            raise ValueError(
                f"--epochs must exceed the resumed epoch ({start_epoch})."
            )
    else:
        if args.train_dataset_mode != "fixed":
            raise ValueError("--train_steps requires --train_dataset_mode fixed.")
        if args.train_steps <= start_global_step:
            raise ValueError(
                "--train_steps must exceed the resumed global step "
                f"({start_global_step})."
            )
        if args.validation_interval_steps <= 0:
            raise ValueError("validation_interval_steps must be positive.")
    if args.lr <= 0.0:
        raise ValueError("lr must be positive.")
    if args.lr_min < 0.0:
        raise ValueError("lr_min must be non-negative.")
    if args.lr_scheduler == "cosine" and args.lr_min > args.lr:
        raise ValueError("lr_min must not exceed lr for the cosine schedule.")
    if args.train_steps is None:
        if args.warmup_steps or args.constant_tail_steps:
            raise ValueError("Step-based LR arguments require --train_steps.")
        if not 0 <= args.warmup_epochs < args.epochs:
            raise ValueError("warmup_epochs must be in [0, epochs).")
        if args.constant_tail_epochs < 0:
            raise ValueError("constant_tail_epochs must be non-negative.")
        if args.warmup_epochs + args.constant_tail_epochs >= args.epochs:
            raise ValueError(
                "warmup_epochs + constant_tail_epochs must be less than epochs."
            )
        if args.lr_scheduler != "cosine" and args.constant_tail_epochs:
            raise ValueError("constant_tail_epochs requires --lr_scheduler cosine.")
    else:
        if args.warmup_epochs or args.constant_tail_epochs:
            raise ValueError("Epoch-based LR arguments cannot be mixed with --train_steps.")
        if args.warmup_steps < 0 or args.constant_tail_steps < 0:
            raise ValueError("Step-based warmup and tail must be non-negative.")
        if args.warmup_steps + args.constant_tail_steps >= args.train_steps:
            raise ValueError(
                "warmup_steps + constant_tail_steps must be less than train_steps."
            )
        if args.lr_scheduler != "cosine" and args.constant_tail_steps:
            raise ValueError("constant_tail_steps requires --lr_scheduler cosine.")


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in PyTorch.")

    # This must precede model construction. The previous ordering seeded only
    # data generation and left nominally identical training seeds with random
    # model initializations.
    _seed_runtime(args.seed, device, args.deterministic_algorithms)

    resumed_checkpoint = None
    start_epoch = 0
    start_global_step = 0
    if args.resume_checkpoint:
        model, data_config, resumed_checkpoint = load_su_mimo_checkpoint(
            args.resume_checkpoint, device
        )
        _restore_saved_arguments(args, resumed_checkpoint)
        start_epoch = int(resumed_checkpoint["epoch"])
        start_global_step = int(resumed_checkpoint.get("global_step", 0))
        model_config = resumed_checkpoint["model_config"]
        _seed_runtime(args.seed, device, args.deterministic_algorithms)
    else:
        data_config = build_data_config(args)
        model_config = build_model_config(args)
        model = build_su_mimo_model(args.model, model_config).to(device)

    _validate_args(args, start_epoch, start_global_step)
    current_model_hash = _model_state_hash(model)
    initial_model_hash = (
        resumed_checkpoint.get("initial_model_hash")
        if resumed_checkpoint is not None
        else current_model_hash
    )

    if resumed_checkpoint is not None:
        if data_config.to_dict() != resumed_checkpoint["sionna_su_mimo_config"]:
            raise RuntimeError("Resumed SU-MIMO data configuration is inconsistent.")
        if model_config != resumed_checkpoint["model_config"]:
            raise RuntimeError("Resumed SU-MIMO model configuration is inconsistent.")

    train_seed = (
        args.seed if args.train_generator_seed is None else args.train_generator_seed
    )
    val_seed = (
        args.seed + 100000
        if args.val_generator_seed is None
        else args.val_generator_seed
    )
    if resumed_checkpoint is None:
        train_profile = (
            load_channel_profile(args.train_channel_profile)
            if args.train_channel_profile
            else legacy_channel_profile(data_config)
        )
        train_profile = filter_channel_profile(
            train_profile,
            component_ids=args.train_component_ids,
            tdl_models=args.train_tdl_models,
            delay_spread_min_ns=args.train_delay_spread_min_ns,
            delay_spread_max_ns=args.train_delay_spread_max_ns,
        )
        val_profile = (
            load_channel_profile(args.val_channel_profile)
            if args.val_channel_profile
            else train_profile
        )
    else:
        saved_train_profile = resumed_checkpoint.get("train_channel_profile")
        saved_val_profile = resumed_checkpoint.get("val_channel_profile")
        train_profile = (
            legacy_channel_profile(data_config)
            if saved_train_profile is None
            else load_channel_profile(saved_train_profile)
        )
        val_profile = (
            train_profile
            if saved_val_profile is None
            else load_channel_profile(saved_val_profile)
        )
    train_generator = build_generator(
        args,
        data_config,
        args.train_phase_mode,
        train_seed,
        device,
        train_profile,
    )
    val_generator = build_generator(
        args,
        data_config,
        args.val_phase_mode,
        val_seed,
        device,
        val_profile,
    )
    fixed_replay = None
    if args.train_dataset_mode == "fixed":
        fixed_replay = FixedDatasetReplay(
            train_generator,
            args.num_train,
            args.batch_size,
            train_seed,
            shuffle_batches=args.fixed_dataset_shuffle_batches,
        )
    fixed_dataset_hash = _fixed_dataset_hash(
        args, data_config, train_profile, train_seed
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if resumed_checkpoint is not None:
        optimizer.load_state_dict(resumed_checkpoint["optimizer_state"])

    if args.save_dir:
        save_dir = Path(args.save_dir)
    elif args.resume_checkpoint:
        save_dir = Path(args.resume_checkpoint).resolve().parent
    else:
        save_dir = Path("runs/su_mimo_debug")
    save_dir.mkdir(parents=True, exist_ok=True)
    args.save_dir = str(save_dir)

    history = [] if resumed_checkpoint is None else list(
        resumed_checkpoint.get("history", [])
    )
    if resumed_checkpoint is not None and not history:
        history.append(
            {
                "epoch": start_epoch,
                "global_step": start_global_step,
                "data_pass": resumed_checkpoint.get("data_pass", start_epoch),
                "samples_seen": resumed_checkpoint.get("samples_seen", ""),
                "lr": optimizer.param_groups[0]["lr"],
                "train_bce": resumed_checkpoint.get("train_bce", ""),
                "train_ber": resumed_checkpoint.get("train_ber", ""),
                "train_bit_errors": resumed_checkpoint.get("train_bit_errors", ""),
                "train_valid_bits": resumed_checkpoint.get("train_valid_bits", ""),
                "val_bce": resumed_checkpoint["val_bce"],
                "val_ber": resumed_checkpoint["val_ber"],
                "val_bit_errors": resumed_checkpoint.get("val_bit_errors", ""),
                "val_valid_bits": resumed_checkpoint.get("val_valid_bits", ""),
            }
        )
    _write_history(save_dir / "history.csv", history)

    best_val_bce = (
        math.inf
        if resumed_checkpoint is None
        else float(resumed_checkpoint.get("best_val_bce", math.inf))
    )
    if resumed_checkpoint is not None:
        source_best = Path(args.resume_checkpoint).resolve().parent / "best.pt"
        target_best = (save_dir / "best.pt").resolve()
        if source_best.exists() and source_best.resolve() != target_best:
            shutil.copy2(source_best, target_best)

    created_at = (
        datetime.now(timezone.utc).isoformat()
        if resumed_checkpoint is None
        else resumed_checkpoint.get("created_at_utc")
    )
    command = shlex.join([sys.executable, *sys.argv])
    resolved = {
        "model_name": args.model,
        "model_config": model_config,
        "data_backend": "sionna_su_mimo",
        "sionna_su_mimo_config": data_config.to_dict(),
        "args": vars(args),
        "selection_rule": "minimum source-validation BCE",
        "validation_policy": (
            f"fixed-seed replay every {args.validation_interval_steps} optimizer steps"
            if args.train_steps is not None
            else "fixed-seed replay on every epoch"
        ),
        "training_dataset": {
            "mode": args.train_dataset_mode,
            "num_distinct_samples": (
                args.num_train if args.train_dataset_mode == "fixed" else None
            ),
            "samples_per_epoch": (
                args.num_train if args.train_dataset_mode == "streaming" else None
            ),
            "batch_seed_policy": (
                "stable seeded shards" if args.train_dataset_mode == "fixed" else None
            ),
            "shuffle_batches_each_pass": (
                args.fixed_dataset_shuffle_batches
                if args.train_dataset_mode == "fixed"
                else None
            ),
            "dataset_hash": fixed_dataset_hash,
        },
        "lr_schedule": {
            "name": args.lr_scheduler,
            "base_lr": args.lr,
            "lr_min": args.lr_min,
            "warmup_epochs": args.warmup_epochs,
            "constant_tail_epochs": args.constant_tail_epochs,
            "total_epochs": args.epochs,
            "warmup_steps": args.warmup_steps,
            "constant_tail_steps": args.constant_tail_steps,
            "total_steps": args.train_steps,
        },
        "initial_model_hash": initial_model_hash,
        "resume_model_hash": current_model_hash if resumed_checkpoint else None,
        "channel_profile_schema_version": PROFILE_SCHEMA_VERSION,
        "train_channel_profile": train_profile,
        "val_channel_profile": val_profile,
        "channel_profile_hash": channel_profile_hash(train_profile),
        "code_revision": _git_revision(),
        "created_at_utc": created_at,
        "last_command": command,
        "torch_version": str(torch.__version__),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    with (save_dir / "resolved_config.json").open("w") as stream:
        json.dump(resolved, stream, indent=2, sort_keys=True)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device} | model: {args.model} | parameters: {parameter_count}")
    print(
        f"tdl_mix reference: {TDL_MIX_REFERENCE_PARAMETERS} | "
        f"parameter delta: {parameter_count - TDL_MIX_REFERENCE_PARAMETERS:+d}"
    )
    print(
        f"Topology: 1 user, {data_config.num_layers} layers, "
        f"{data_config.num_rx_ant} Rx | total Tx power {data_config.total_tx_power:g}"
    )
    print(
        f"Train phase: {args.train_phase_mode} | val phase: {args.val_phase_mode} | "
        f"train seed: {train_seed} | val seed: {val_seed}"
    )
    if args.train_steps is None:
        print(
            f"LR schedule: {args.lr_scheduler} | base {args.lr:g} | "
            f"min {args.lr_min:g} | warmup {args.warmup_epochs} epochs | "
            f"constant tail {args.constant_tail_epochs} epochs"
        )
    else:
        print(
            f"LR schedule: {args.lr_scheduler} | base {args.lr:g} | "
            f"min {args.lr_min:g} | warmup {args.warmup_steps} steps | "
            f"constant tail {args.constant_tail_steps} steps | "
            f"total {args.train_steps} steps"
        )
    print(
        f"Deterministic algorithms: {args.deterministic_algorithms} | "
        f"initial model hash: {initial_model_hash}"
    )
    print(
        f"Train profile: {train_profile['name']} | "
        f"validation profile: {val_profile['name']}"
    )
    if fixed_replay is not None:
        print(
            f"Fixed training corpus: {fixed_replay.num_samples} distinct samples | "
            f"{fixed_replay.steps_per_pass} shards/pass | "
            f"shuffle shards: {fixed_replay.shuffle_batches} | "
            f"dataset hash: {fixed_dataset_hash}"
        )
    if resumed_checkpoint is not None:
        print(f"Resuming {args.resume_checkpoint} from epoch {start_epoch}")

    def make_checkpoint(epoch, row, global_step, train_profile_counts):
        return {
            **resolved,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "lr_scheduler_state": {
                "name": args.lr_scheduler,
                "last_epoch": epoch,
                "last_global_step": global_step,
                "last_lr": row["lr"],
                "base_lr": args.lr,
                "lr_min": args.lr_min,
                "warmup_epochs": args.warmup_epochs,
                "constant_tail_epochs": args.constant_tail_epochs,
                "total_epochs": args.epochs,
                "warmup_steps": args.warmup_steps,
                "constant_tail_steps": args.constant_tail_steps,
                "total_steps": args.train_steps,
            },
            "epoch": epoch,
            "global_step": global_step,
            "data_pass": row.get("data_pass", epoch),
            "samples_seen": row.get("samples_seen", ""),
            "train_bce": row["train_bce"],
            "train_ber": row["train_ber"],
            "train_bit_errors": row["train_bit_errors"],
            "train_valid_bits": row["train_valid_bits"],
            "val_bce": row["val_bce"],
            "val_ber": row["val_ber"],
            "val_bit_errors": row["val_bit_errors"],
            "val_valid_bits": row["val_valid_bits"],
            "best_val_bce": best_val_bce,
            "history": history,
            "train_generator_seed": train_seed,
            "val_generator_seed": val_seed,
            "train_profile_counts": dict(train_profile_counts),
            "val_profile_counts": val_generator.channel_profile_counts,
            "resumed_from": args.resume_checkpoint,
        }

    def record_validation(epoch, global_step, row, train_profile_counts):
        nonlocal best_val_bce
        history.append(row)
        _write_history(save_dir / "history.csv", history)
        improved = row["val_bce"] < best_val_bce
        if improved:
            best_val_bce = row["val_bce"]
        checkpoint = make_checkpoint(epoch, row, global_step, train_profile_counts)
        checkpoint["best_val_bce"] = best_val_bce
        torch.save(checkpoint, save_dir / "last.pt")
        if improved:
            torch.save(checkpoint, save_dir / "best.pt")
            print(f"  saved best checkpoint to {save_dir / 'best.pt'}")

    if args.train_steps is not None:
        cumulative_profile_counts = Counter(
            {} if resumed_checkpoint is None else resumed_checkpoint.get(
                "train_profile_counts", {}
            )
        )
        interval_bce = 0.0
        interval_errors = 0
        interval_bits = 0
        model.train()
        for global_step in range(start_global_step + 1, args.train_steps + 1):
            step_lr = learning_rate_for_step(
                global_step,
                args.train_steps,
                args.lr,
                scheduler_name=args.lr_scheduler,
                lr_min=args.lr_min,
                warmup_steps=args.warmup_steps,
                constant_tail_steps=args.constant_tail_steps,
            )
            _set_optimizer_lr(optimizer, step_lr)
            batch, zero_based_pass, _ = fixed_replay.batch_for_step(global_step - 1)
            bce_sum, errors, valid_bits = train_one_batch(model, batch, optimizer)
            interval_bce += bce_sum
            interval_errors += errors
            interval_bits += valid_bits
            profile_id = batch.get("channel_profile_id")
            if profile_id is not None:
                cumulative_profile_counts[str(profile_id)] += 1

            if args.log_interval > 0 and global_step % args.log_interval == 0:
                print(
                    f"Step {global_step:06d}/{args.train_steps} | "
                    f"pass {zero_based_pass + 1} | lr {step_lr:.6e} | "
                    f"interval BCE {interval_bce / interval_bits:.6f} | "
                    f"BER {interval_errors / interval_bits:.6f}"
                )

            should_validate = (
                global_step % args.validation_interval_steps == 0
                or global_step == args.train_steps
            )
            if not should_validate:
                continue
            val_bce, val_ber, val_errors, val_bits = evaluate(
                model,
                val_generator,
                args.num_val,
                args.batch_size,
                reset_seed=val_seed,
            )
            completed_passes = math.ceil(global_step / fixed_replay.steps_per_pass)
            row = {
                "epoch": completed_passes,
                "global_step": global_step,
                "data_pass": completed_passes,
                "samples_seen": fixed_replay.samples_seen(global_step),
                "lr": step_lr,
                "train_bce": interval_bce / interval_bits,
                "train_ber": interval_errors / interval_bits,
                "train_bit_errors": interval_errors,
                "train_valid_bits": interval_bits,
                "val_bce": val_bce,
                "val_ber": val_ber,
                "val_bit_errors": val_errors,
                "val_valid_bits": val_bits,
            }
            print(
                f"Validation step {global_step:06d} | pass {completed_passes} | "
                f"lr {step_lr:.6e} | train BCE {row['train_bce']:.6f} | "
                f"train BER {row['train_ber']:.6f} "
                f"({interval_errors}/{interval_bits}) | val BCE {val_bce:.6f} | "
                f"val BER {val_ber:.6f} ({val_errors}/{val_bits})"
            )
            print(
                f"  cumulative train profile batches: {dict(cumulative_profile_counts)} | "
                f"validation profile batches: {val_generator.channel_profile_counts}"
            )
            record_validation(
                completed_passes,
                global_step,
                row,
                cumulative_profile_counts,
            )
            interval_bce = 0.0
            interval_errors = 0
            interval_bits = 0
            model.train()
    else:
        steps_per_epoch = math.ceil(args.num_train / args.batch_size)
        for epoch in range(start_epoch + 1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            epoch_lr = learning_rate_for_epoch(
                epoch,
                args.epochs,
                args.lr,
                scheduler_name=args.lr_scheduler,
                lr_min=args.lr_min,
                warmup_epochs=args.warmup_epochs,
                constant_tail_epochs=args.constant_tail_epochs,
            )
            _set_optimizer_lr(optimizer, epoch_lr)
            if fixed_replay is None:
                train_generator.reset(train_seed + epoch * 1009)
                train_bce, train_ber, train_errors, train_bits = train_one_epoch(
                    model,
                    train_generator,
                    optimizer,
                    args.num_train,
                    args.batch_size,
                    args.log_interval,
                )
                train_profile_counts = train_generator.channel_profile_counts
            else:
                (
                    train_bce,
                    train_ber,
                    train_errors,
                    train_bits,
                    train_profile_counts,
                ) = train_one_fixed_pass(
                    model,
                    fixed_replay,
                    optimizer,
                    epoch - 1,
                    args.log_interval,
                )
            val_bce, val_ber, val_errors, val_bits = evaluate(
                model,
                val_generator,
                args.num_val,
                args.batch_size,
                reset_seed=val_seed,
            )
            global_step = epoch * steps_per_epoch
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "data_pass": epoch,
                "samples_seen": epoch * args.num_train,
                "lr": epoch_lr,
                "train_bce": train_bce,
                "train_ber": train_ber,
                "train_bit_errors": train_errors,
                "train_valid_bits": train_bits,
                "val_bce": val_bce,
                "val_ber": val_ber,
                "val_bit_errors": val_errors,
                "val_valid_bits": val_bits,
            }
            print(
                f"Epoch {epoch:03d} | lr {epoch_lr:.6e} | "
                f"train BCE {train_bce:.6f} | "
                f"train BER {train_ber:.6f} ({train_errors}/{train_bits}) | "
                f"val BCE {val_bce:.6f} | val BER {val_ber:.6f} "
                f"({val_errors}/{val_bits})"
            )
            print(
                f"  train profile batches: {train_profile_counts} | "
                f"validation profile batches: {val_generator.channel_profile_counts}"
            )
            record_validation(epoch, global_step, row, train_profile_counts)


if __name__ == "__main__":
    main()
