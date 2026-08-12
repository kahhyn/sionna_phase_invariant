"""Train/evaluate the real CNN with identity, MF, T_gamma, and LMMSE frontends.

The three frontends are applied independently at every OFDM resource element::

    Identity: no spatial preprocessing
    MF:     T = H_hat^H
    Tgamma: T = (H_hat^H H_hat + gamma I)^(-1/2) H_hat^H
    LMMSE:  T = (H_hat^H H_hat + gamma I)^(-1) H_hat^H

where ``gamma = gamma_scale * N0 / Es`` and ``Es`` is the per-layer symbol
energy.  With the default four layers and unit total transmit power this is
``gamma = 4 * gamma_scale * N0``.

This is a new experiment entry point.  It reuses the existing data/model and
training utilities but does not modify the original five-representation test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data import SionnaSUMIMOBatchGenerator, legacy_channel_profile
from tests.representation_invariance.train_eval_transforms import (
    DATA_CONFIG as BASE_DATA_CONFIG,
    build_model,
    train_one_epoch,
)
from utils.batching import batch_sizes
from utils.metrics import masked_bce_with_logits, masked_error_count


FRONTENDS = ("identity", "mf", "tgamma", "lmmse")


def hermitian(x: torch.Tensor) -> torch.Tensor:
    """最后两个维度做共轭转置。"""
    return x.conj().transpose(-2, -1)


def apply_regularized_frontend(
    h_hat: torch.Tensor,
    y: torch.Tensor,
    frontend: str,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one frontend to a batch of flat channel snapshots.

    Args:
        h_hat: ``[N, Nr, Nt]`` complex estimated channel.
        y: ``[N, Nr]`` complex received vector.
        frontend: ``identity``, ``mf``, ``tgamma``, or ``lmmse``.
        gamma: ``[N]`` positive real regularization values.

    Returns:
        z: ``T y``, shape ``[N, Nt]``.
        h_seen: ``T H_hat``, shape ``[N, Nt, Nt]``.
        t_matrix: preprocessing matrix ``T``, shape ``[N, Nt, Nr]``.
        noise_gain: ``trace(T T^H)/Nt``, shape ``[N]``.
    """
    if frontend not in FRONTENDS:
        raise ValueError(f"Unknown frontend: {frontend}")
    if h_hat.ndim != 3 or y.ndim != 2:
        raise ValueError("Expected h_hat [N,Nr,Nt] and y [N,Nr].")
    if h_hat.shape[0] != y.shape[0] or h_hat.shape[1] != y.shape[1]:
        raise ValueError("h_hat and y batch/receive dimensions do not match.")

    n, nr, nt = h_hat.shape
    if nr < nt:
        raise ValueError("This implementation requires Nr >= Nt.")
    gamma = gamma.to(device=h_hat.device, dtype=h_hat.real.dtype).reshape(n)
    if bool(torch.any(gamma <= 0)):
        raise ValueError("gamma must be strictly positive.")

    if frontend == "identity":
        # 绝对基线：网络直接看到原始 Y 和 H_hat。
        t_matrix = (
            torch.eye(nr, device=h_hat.device, dtype=h_hat.dtype)
            .unsqueeze(0)
            .expand(n, -1, -1)
        )
    elif frontend == "mf":
        # MF 直接计算即可，不为它付出逐 RE 做 SVD 的开销。
        t_matrix = hermitian(h_hat)
    else:  # tgamma or lmmse
        # T_gamma/LMMSE 用 SVD 实现比直接求逆更稳定：H = U diag(s) V^H。
        u, singular_values, vh = torch.linalg.svd(h_hat, full_matrices=False)
        v = hermitian(vh)
        s2 = singular_values.square()
        if frontend == "tgamma":
            weights = singular_values / torch.sqrt(s2 + gamma[:, None])
        else:
            weights = singular_values / (s2 + gamma[:, None])
        # V diag(weights) U^H；不做额外归一化，保留前端的原始定义。
        t_matrix = (v * weights.unsqueeze(-2)) @ hermitian(u)
    z = (t_matrix @ y.unsqueeze(-1)).squeeze(-1)
    h_seen = t_matrix @ h_hat
    noise_gain = t_matrix.abs().square().sum(dim=(-2, -1)) / float(t_matrix.shape[-2])
    return z, h_seen, t_matrix, noise_gain


def _frame_values(value, batch_size: int, device: torch.device) -> torch.Tensor:
    """Convert a scalar/[B,...] value to one real value per frame."""
    tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
    if tensor.numel() == 1:
        return tensor.reshape(1).expand(batch_size)
    if tensor.shape[0] != batch_size:
        raise ValueError(f"Cannot map shape {tuple(tensor.shape)} to B={batch_size}.")
    return tensor.reshape(batch_size, -1)[:, 0]


class RegularizedFrontendWrapper:
    """Apply identity/MF/T_gamma/LMMSE to a batch from the Sionna generator."""

    def __init__(
        self,
        base_generator: SionnaSUMIMOBatchGenerator,
        frontend: str,
        gamma_scale: float = 1.0,
        n0_mode: str = "original",
    ):
        if frontend not in FRONTENDS:
            raise ValueError(f"frontend must be one of {FRONTENDS}")
        if gamma_scale <= 0:
            raise ValueError("gamma_scale must be positive.")
        if n0_mode not in ("original", "mean_effective"):
            raise ValueError("n0_mode must be original or mean_effective.")
        self._gen = base_generator
        self.frontend = frontend
        self.gamma_scale = float(gamma_scale)
        self.n0_mode = n0_mode
        self.last_stats: dict[str, float] = {}

    @property
    def config(self):
        return self._gen.config

    @property
    def channel_profile_counts(self) -> dict:
        return self._gen.channel_profile_counts

    def reset(self, seed: Optional[int] = None) -> None:
        self._gen.reset(seed)

    def reset_profile_sampler(self, seed: Optional[int] = None) -> None:
        self._gen.reset_profile_sampler(seed)

    @torch.no_grad()
    def generate_batch(self, batch_size: int, *, return_aux: bool = False) -> dict:
        batch = self._gen.generate_batch(batch_size, return_aux=return_aux)
        y = batch["Y"]                         # [B, Nr, T, F]
        h_hat = batch["H_hat"]                 # [B, L, Nr, T, F]
        b, nr, num_symbols, num_subcarriers = y.shape
        _, num_layers, _, _, _ = h_hat.shape

        n0_frame = _frame_values(batch["N0"], b, y.device)
        es_frame = _frame_values(batch["power_per_data_layer"], b, y.device)
        # gamma=N0/Es。当前默认设置下 Es=1/4，所以 gamma=4*N0。
        gamma_frame = self.gamma_scale * n0_frame / es_frame.clamp_min(1e-12)
        gamma_flat = (
            gamma_frame[:, None, None]
            .expand(b, num_symbols, num_subcarriers)
            .reshape(-1)
        )

        y_flat = y.permute(0, 2, 3, 1).reshape(-1, nr)
        h_flat = (
            h_hat.permute(0, 3, 4, 2, 1)
            .reshape(-1, nr, num_layers)
        )
        z_flat, h_seen_flat, _, noise_gain_flat = apply_regularized_frontend(
            h_flat, y_flat, self.frontend, gamma_flat
        )

        num_outputs = z_flat.shape[-1]
        y_new = (
            z_flat.reshape(b, num_symbols, num_subcarriers, num_outputs)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        h_new = (
            h_seen_flat.reshape(
                b, num_symbols, num_subcarriers, num_outputs, num_layers
            )
            .permute(0, 4, 3, 1, 2)
            .contiguous()
        )
        noise_gain_frame = noise_gain_flat.reshape(
            b, num_symbols, num_subcarriers
        ).mean(dim=(1, 2))

        batch["Y"] = y_new.to(torch.complex64)
        batch["H_hat"] = h_new.to(torch.complex64)
        batch["frontend"] = self.frontend
        batch["frontend_gamma"] = gamma_frame
        batch["frontend_noise_gain"] = noise_gain_frame

        # 默认仍给网络原始 N0，以便和原表示实验保持相同接口。
        # 可选模式仅传入每帧平均后的等效噪声功率，不能表达完整有色协方差。
        if self.n0_mode == "mean_effective":
            n0_shape = batch["N0"].shape
            batch["N0"] = (n0_frame * noise_gain_frame).reshape(n0_shape)

        self.last_stats = {
            "mean_gamma": float(gamma_frame.mean().item()),
            "mean_noise_gain": float(noise_gain_frame.mean().item()),
            "max_noise_gain": float(noise_gain_flat.max().item()),
        }
        return batch


@torch.no_grad()
def evaluate(
    model: nn.Module,
    generator: RegularizedFrontendWrapper,
    num_samples: int,
    batch_size: int,
    reset_seed: Optional[int] = None,
) -> tuple[float, float, int, int, float, float]:
    model.eval()
    generator.reset(reset_seed)
    total_bce = 0.0
    total_errors = 0
    total_bits = 0
    gamma_sum = 0.0
    noise_gain_sum = 0.0
    frame_count = 0

    for current_bs in batch_sizes(num_samples, batch_size):
        batch = generator.generate_batch(current_bs)
        logits = model(
            batch["Y"], batch["H_hat"], batch["P"], batch["N0"], batch["layer_mask"]
        )
        bce = masked_bce_with_logits(logits, batch["bits"], batch["loss_mask"])
        errors, valid_bits = masked_error_count(logits, batch["bits"], batch["loss_mask"])
        count = int(valid_bits.item())
        total_bce += float(bce.item()) * count
        total_errors += int(errors.item())
        total_bits += count
        gamma_sum += generator.last_stats["mean_gamma"] * current_bs
        noise_gain_sum += generator.last_stats["mean_noise_gain"] * current_bs
        frame_count += current_bs

    return (
        total_bce / max(total_bits, 1),
        total_errors / max(total_bits, 1),
        total_errors,
        total_bits,
        gamma_sum / max(frame_count, 1),
        noise_gain_sum / max(frame_count, 1),
    )


def make_generator(
    data_config,
    profile,
    frontend: str,
    snr_min: float,
    snr_max: float,
    phase_mode: str,
    seed: int,
    device: torch.device,
    gamma_scale: float,
    n0_mode: str,
) -> RegularizedFrontendWrapper:
    base = SionnaSUMIMOBatchGenerator(
        data_config,
        snr_db_min=snr_min,
        snr_db_max=snr_max,
        phase_mode=phase_mode,
        seed=seed,
        device=device,
        channel_profile=profile,
    )
    return RegularizedFrontendWrapper(base, frontend, gamma_scale, n0_mode)


def run_algebra_self_test(device: torch.device) -> None:
    """Numerically verify all four formulas against direct references."""
    torch.manual_seed(20260812)
    dtype = torch.complex128
    h = torch.randn(7, 4, 4, device=device, dtype=torch.float64) + 1j * torch.randn(
        7, 4, 4, device=device, dtype=torch.float64
    )
    h = h.to(dtype)
    y = (
        torch.randn(7, 4, device=device, dtype=torch.float64)
        + 1j * torch.randn(7, 4, device=device, dtype=torch.float64)
    ).to(dtype)
    gamma = torch.linspace(0.03, 2.0, 7, device=device, dtype=torch.float64)
    h_h = hermitian(h)
    gram = h_h @ h
    eye = torch.eye(4, device=device, dtype=dtype).expand(7, -1, -1)

    for frontend in FRONTENDS:
        z, h_seen, t_matrix, noise_gain = apply_regularized_frontend(
            h, y, frontend, gamma
        )
        assert z.shape == (7, 4) and h_seen.shape == (7, 4, 4)
        assert bool(torch.isfinite(z).all()) and bool(torch.isfinite(noise_gain).all())

        if frontend == "identity":
            reference = eye
        elif frontend == "mf":
            reference = h_h
        elif frontend == "lmmse":
            reference = torch.linalg.solve(gram + gamma[:, None, None] * eye, h_h)
        else:
            eigenvalues, eigenvectors = torch.linalg.eigh(
                gram + gamma[:, None, None] * eye
            )
            inv_sqrt = (
                eigenvectors * eigenvalues.rsqrt().unsqueeze(-2)
            ) @ hermitian(eigenvectors)
            reference = inv_sqrt @ h_h

        max_error = float((t_matrix - reference).abs().max().item())
        if max_error > 2e-10:
            raise AssertionError(f"{frontend} formula error too large: {max_error:.3e}")
        print(f"[self-test] {frontend:7s} max formula error = {max_error:.3e}")


@torch.no_grad()
def run_pipeline_smoke_test(
    data_config,
    profile,
    frontends: list[str],
    device: torch.device,
    seed: int,
    gamma_scale: float,
    n0_mode: str,
) -> None:
    """Generate one real Sionna batch and run one CNN forward pass per frontend."""
    for frontend in frontends:
        generator = make_generator(
            data_config, profile, frontend, 5.0, 5.0, "uniform",
            seed, device, gamma_scale, n0_mode,
        )
        model = build_model(device).eval()
        batch = generator.generate_batch(1)
        logits = model(
            batch["Y"], batch["H_hat"], batch["P"], batch["N0"], batch["layer_mask"]
        )
        loss = masked_bce_with_logits(logits, batch["bits"], batch["loss_mask"])
        if not bool(torch.isfinite(loss)):
            raise AssertionError(f"{frontend} smoke-test loss is not finite.")
        print(
            f"[pipeline-test] {frontend:7s} Y={tuple(batch['Y'].shape)} "
            f"H={tuple(batch['H_hat'].shape)} BCE={loss.item():.6f} "
            f"gamma={generator.last_stats['mean_gamma']:.4g} "
            f"noise_gain={generator.last_stats['mean_noise_gain']:.4g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-CNN experiment with identity/MF/T_gamma/LMMSE frontends"
    )
    parser.add_argument("--frontends", nargs="+", choices=FRONTENDS, default=list(FRONTENDS))
    parser.add_argument("--gamma_scale", type=float, default=1.0)
    parser.add_argument(
        "--n0_mode", choices=("original", "mean_effective"), default="original",
        help="N0 supplied to CNN; default preserves the old experiment interface.",
    )
    parser.add_argument(
        "--ls_interpolation_type", choices=("nn", "lin"), default="nn",
        help="Use nn by default, matching the stated experiment setting.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_train", type=int, default=10000)
    parser.add_argument("--num_val", type=int, default=2000)
    parser.add_argument("--num_test", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train_snr_min", type=float, default=-5.0)
    parser.add_argument("--train_snr_max", type=float, default=20.0)
    parser.add_argument("--eval_snr_min", type=float, default=-5.0)
    parser.add_argument("--eval_snr_max", type=float, default=20.0)
    parser.add_argument("--eval_snr_step", type=float, default=2.0)
    parser.add_argument(
        "--output_dir",
        default="tests/representation_invariance/runs_regularized_frontends",
    )
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument(
        "--smoke_test_only",
        action="store_true",
        help="Run formula and one-batch Sionna/CNN tests, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gamma_scale <= 0:
        raise ValueError("--gamma_scale must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # 旧脚本实际为 lin；本实验按已确认设置默认改为 nn，可用参数切回 lin。
    data_config = replace(
        BASE_DATA_CONFIG, ls_interpolation_type=args.ls_interpolation_type
    )
    profile = legacy_channel_profile(data_config)

    run_algebra_self_test(device)
    if args.smoke_test_only:
        run_pipeline_smoke_test(
            data_config, profile, args.frontends, device, args.seed,
            args.gamma_scale, args.n0_mode,
        )
        print("All smoke tests passed.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"arguments": vars(args), "data_config": asdict(data_config)},
            file, indent=2, ensure_ascii=False,
        )

    eval_snrs = torch.arange(
        args.eval_snr_min,
        args.eval_snr_max + 0.5 * args.eval_snr_step,
        args.eval_snr_step,
    ).tolist()
    all_results: dict[str, list[dict]] = {}

    print(f"Device: {device}")
    print(f"Frontends: {args.frontends}")
    print(f"gamma = {args.gamma_scale} * N0 / Es; N0 mode = {args.n0_mode}")
    print(f"LS interpolation: {args.ls_interpolation_type}")

    for frontend_index, frontend in enumerate(args.frontends, start=1):
        print(f"\n{'=' * 64}\n[{frontend_index}/{len(args.frontends)}] {frontend}\n{'=' * 64}")
        checkpoint_path = output_dir / f"model_{frontend}_seed{args.seed}.pt"

        # 每种前端使用相同初始化和相同随机数据序列，保证单 seed 对比公平。
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        model = build_model(device)

        if args.skip_train:
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
        else:
            train_generator = make_generator(
                data_config, profile, frontend,
                args.train_snr_min, args.train_snr_max, "fixed", args.seed,
                device, args.gamma_scale, args.n0_mode,
            )
            val_generator = make_generator(
                data_config, profile, frontend,
                args.train_snr_min, args.train_snr_max, "uniform", args.seed + 200000,
                device, args.gamma_scale, args.n0_mode,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            best_val_bce = math.inf

            for epoch in range(1, args.epochs + 1):
                train_generator.reset_profile_sampler(args.seed + epoch * 1009)
                train_bce, train_ber = train_one_epoch(
                    model, train_generator, optimizer,
                    args.num_train, args.batch_size, args.log_interval,
                )
                val_bce, val_ber, _, _, _, _ = evaluate(
                    model, val_generator, args.num_val, args.batch_size,
                    reset_seed=args.seed + 200000,
                )
                print(
                    f"Epoch {epoch:03d}/{args.epochs} | "
                    f"train BCE {train_bce:.6f} BER {train_ber:.6f} | "
                    f"val BCE {val_bce:.6f} BER {val_ber:.6f}"
                )
                if val_bce < best_val_bce:
                    best_val_bce = val_bce
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "frontend": frontend,
                            "gamma_scale": args.gamma_scale,
                            "n0_mode": args.n0_mode,
                            "ls_interpolation_type": args.ls_interpolation_type,
                        },
                        checkpoint_path,
                    )
                    print("  -> saved best checkpoint")

            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])

        frontend_results = []
        print(f"\nSNR sweep: {frontend}")
        for snr_index, snr in enumerate(eval_snrs):
            eval_seed = args.seed + 777000 + snr_index * 1000
            eval_generator = make_generator(
                data_config, profile, frontend, snr, snr, "uniform", eval_seed,
                device, args.gamma_scale, args.n0_mode,
            )
            bce, ber, errors, valid_bits, mean_gamma, mean_noise_gain = evaluate(
                model, eval_generator, args.num_test, args.batch_size,
                reset_seed=eval_seed,
            )
            result = {
                "frontend": frontend,
                "snr_db": snr,
                "bce": bce,
                "ber": ber,
                "bit_errors": errors,
                "valid_bits": valid_bits,
                "mean_gamma": mean_gamma,
                "mean_noise_gain": mean_noise_gain,
                "gamma_scale": args.gamma_scale,
                "n0_mode": args.n0_mode,
                "ls_interpolation_type": args.ls_interpolation_type,
            }
            frontend_results.append(result)
            print(
                f"  SNR {snr:5.1f} dB | BER {ber:.6e} ({errors}/{valid_bits}) | "
                f"gamma {mean_gamma:.3e} | noise gain {mean_noise_gain:.3e}"
            )
        all_results[frontend] = frontend_results

    markers = {"identity": "o", "mf": "^", "tgamma": "v", "lmmse": "s"}
    plt.figure(figsize=(10, 6))
    for frontend, results in all_results.items():
        plt.semilogy(
            [row["snr_db"] for row in results],
            [row["ber"] for row in results],
            marker=markers[frontend], linewidth=1.8, label=frontend,
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(
        "SU-MIMO Real CNN: Identity / MF / T_gamma / LMMSE Frontends\n"
        f"TDL-A, {data_config.num_layers}x{data_config.num_rx_ant} MIMO, "
        f"{data_config.num_ofdm_symbols} sym x {data_config.fft_size} SC"
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "ber_vs_snr_regularized_frontends.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    csv_path = output_dir / "results_regularized_frontends.csv"
    fieldnames = [
        "frontend", "snr_db", "bce", "ber", "bit_errors", "valid_bits",
        "mean_gamma", "mean_noise_gain", "gamma_scale", "n0_mode",
        "ls_interpolation_type",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for results in all_results.values():
            writer.writerows(results)

    print(f"\nPlot: {plot_path}")
    print(f"CSV:  {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
