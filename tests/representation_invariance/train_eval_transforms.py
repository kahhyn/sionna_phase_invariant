"""Train and evaluate the SU-MIMO real CNN with different unitary input representations.

For each transform (identity, haar, qr, svd, polar), the receiver Y and
estimated channel H_hat are rotated by a unitary W^H before being fed into
the network.  All transforms preserve mutual information, so performance
differences reveal sensitivity to the chosen linear basis.

Training uses the existing Sionna SU-MIMO pipeline (TDL-A, 4×4 MIMO, 14-symbol
72-subcarrier OFDM, DMRS at symbols 2 and 11, LS interpolation).

Usage::

    python -m tests.representation_invariance.train_eval_transforms

or::

    PYTHONPATH=. python tests/representation_invariance/train_eval_transforms.py

Outputs are written to ``tests/representation_invariance/runs/``.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Optional

# Ensure the project root is on sys.path so that direct invocation works.
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data import (
    SionnaSUMIMOBatchGenerator,
    SionnaSUMIMOConfig,
    legacy_channel_profile,
)
from models import build_su_mimo_model
from utils.batching import batch_sizes
from utils.metrics import masked_bce_with_logits, masked_error_count

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# helper: Hermitian (conjugate) transpose
# ---------------------------------------------------------------------------


def _hermitian(x: torch.Tensor) -> torch.Tensor:
    return x.conj().transpose(-2, -1)


# ---------------------------------------------------------------------------
# Polar decomposition
# ---------------------------------------------------------------------------


def thin_polar(h_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Thin Polar decomposition H_hat = U_p @ P via SVD (batched).

    Args:
        h_hat: ``[..., Nr, Nt]`` with Nr >= Nt.

    Returns:
        u_p: ``[..., Nr, Nt]``  -- semi-unitary factor.
        p:   ``[..., Nt, Nt]``  -- Hermitian positive semi-definite factor.
    """
    assert h_hat.ndim >= 2
    nr, nt = h_hat.shape[-2:]
    assert nr >= nt, "thin_polar requires Nr >= Nt"

    u, singular_values, vh = torch.linalg.svd(h_hat, full_matrices=False)
    v = _hermitian(vh)  # [..., Nt, Nt]
    u_p = u @ vh  # [..., Nr, Nt]
    # V diag(s) V^H  -- broadcast s along columns of V
    p = (v * singular_values.unsqueeze(-2)) @ vh  # [..., Nt, Nt]
    return u_p, p


# ---------------------------------------------------------------------------
# Representation transforms
# ---------------------------------------------------------------------------


def apply_representation(
    h_hat: torch.Tensor,
    y: torch.Tensor,
    transform: str,
    haar: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(z, h_seen, W)`` where ``z = W^H @ y``.

    Args:
        h_hat: ``[B, Nr, Nt]`` -- channel estimate.
        y:     ``[B, Nr]``     -- received signal.
        transform: one of ``identity, haar, qr, svd, polar``.
        haar:  pre-generated Haar unitary ``[Nr, Nr]`` (required for ``haar``).

    Returns:
        z:      ``[B, N_out]``   -- transformed received signal.
        h_seen: ``[B, N_out, Nt]`` -- transformed channel seen by the network.
        W:      ``[B, Nr, N_out]`` -- projection basis.
    """
    batch_size, nr, nt = h_hat.shape
    if transform == "identity":
        w = (
            torch.eye(nr, dtype=h_hat.dtype, device=h_hat.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        return y, h_hat, w

    if transform == "haar":
        if haar is None:
            raise ValueError("haar transform requires a pre-generated Haar matrix.")
        # Accept [Nr, Nr] or [B, Nr, Nr].
        w = haar.to(dtype=h_hat.dtype, device=h_hat.device)
        if w.dim() == 2:
            w = w.unsqueeze(0).expand(batch_size, -1, -1)
        elif w.shape[0] != batch_size:
            raise ValueError(f"Haar batch size {w.shape[0]} != {batch_size}.")
        z = (_hermitian(w) @ y.unsqueeze(-1)).squeeze(-1)
        h_seen = _hermitian(w) @ h_hat
        return z, h_seen, w

    if transform == "qr":
        q, r = torch.linalg.qr(h_hat, mode="reduced")  # Q: [B,Nr,Nt], R: [B,Nt,Nt]
        z = (_hermitian(q) @ y.unsqueeze(-1)).squeeze(-1)
        return z, r, q

    if transform == "svd":
        u, singular_values, vh = torch.linalg.svd(h_hat, full_matrices=False)
        h_seen = singular_values.unsqueeze(-1) * vh  # [B, Nt, Nt]
        z = (_hermitian(u) @ y.unsqueeze(-1)).squeeze(-1)
        return z, h_seen, u

    if transform == "polar":
        u_p, p = thin_polar(h_hat)
        z = (_hermitian(u_p) @ y.unsqueeze(-1)).squeeze(-1)
        return z, p, u_p

    raise ValueError(f"Unknown transform: {transform}")


# ---------------------------------------------------------------------------
# Haar random unitary generation
# ---------------------------------------------------------------------------


def generate_haar(batch_size: int, n: int, device: torch.device) -> torch.Tensor:
    """Sample Haar-distributed random unitary matrices.

    Uses the QR decomposition of a complex standard Gaussian matrix.
    Returns ``[B, n, n]``.
    """
    a = torch.randn(batch_size, n, n, dtype=torch.float32, device=device) + 1j * torch.randn(
        batch_size, n, n, dtype=torch.float32, device=device
    )
    q, _ = torch.linalg.qr(a)
    return q  # [B, n, n]


# ---------------------------------------------------------------------------
# Generator wrapper that applies per-subcarrier unitary transforms
# ---------------------------------------------------------------------------


class TransformGeneratorWrapper:
    """Wraps a :class:`SionnaSUMIMOBatchGenerator` and applies a unitary
    representation transform to ``Y`` and ``H_hat`` before returning the batch.
    """

    def __init__(
        self,
        base_generator: SionnaSUMIMOBatchGenerator,
        transform: str,
        haar_seed: int = 12345,
    ):
        self._gen = base_generator
        self._transform = transform
        self._haar_rng = torch.Generator(device="cpu")
        self._haar_rng.manual_seed(haar_seed)

        cfg = base_generator.config
        self._nt = cfg.num_layers
        self._nr = cfg.num_rx_ant
        self._t = cfg.num_ofdm_symbols
        self._f = cfg.fft_size
        self._device = base_generator.device

    # -- delegate -----------------------------------------------------------------
    @property
    def config(self):
        return self._gen.config

    @property
    def channel_profile_counts(self) -> dict:
        return self._gen.channel_profile_counts

    def reset(self, seed: int | None = None) -> None:
        self._gen.reset(seed)

    def reset_profile_sampler(self, seed: int | None = None) -> None:
        self._gen.reset_profile_sampler(seed)

    # -- core --------------------------------------------------------------------
    @torch.no_grad()
    def generate_batch(self, batch_size: int, *, return_aux: bool = False) -> dict:
        batch = self._gen.generate_batch(batch_size, return_aux=return_aux)
        y = batch["Y"]  # [B, Nr, T, F]
        h_hat = batch["H_hat"]  # [B, L, Nr, T, F]
        b, nr, t, f = y.shape
        _, l, _, _, _ = h_hat.shape

        # Reshape to treat each (symbol, subcarrier) independently.
        # y  -> [B, T, F, Nr] -> [B*T*F, Nr]
        # h  -> [B, T, F, Nr, L] -> [B*T*F, Nr, L]   (permute then reshape)
        y_flat = y.permute(0, 2, 3, 1).reshape(b * t * f, nr)
        h_flat = h_hat.permute(0, 3, 4, 2, 1).reshape(b * t * f, nr, l)

        n_total = b * t * f

        # Pre-generate Haar matrices if needed (per subcarrier).
        if self._transform == "haar":
            # One random unitary per (batch, symbol, subcarrier)
            haar_batch = generate_haar(n_total, nr, self._device)
        else:
            haar_batch = None

        z_flat, h_seen_flat, _ = apply_representation(
            h_flat, y_flat, self._transform, haar_batch
        )

        # z_flat:        [B*T*F, N_out]  -> [B, N_out, T, F]
        # h_seen_flat:   [B*T*F, N_out, L] -> [B, L, N_out, T, F]
        n_out = z_flat.shape[-1]

        y_new = z_flat.reshape(b, t, f, n_out).permute(0, 3, 1, 2)  # [B, N_out, T, F]
        h_new = (
            h_seen_flat.reshape(b, t, f, n_out, l)
            .permute(0, 4, 3, 1, 2)
            .contiguous()
        )  # [B, L, N_out, T, F]

        # For transforms that change the "receive" dimension, also reshape the
        # loss mask to match (the mask is spatial, so it's unaffected).
        batch["Y"] = y_new.to(torch.complex64)
        batch["H_hat"] = h_new.to(torch.complex64)
        batch["transform"] = self._transform

        # Store W for diagnostics (omit from batch to save memory).
        return batch


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_one_batch(
    model: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, int, int]:
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

    with torch.no_grad():
        errors, valid_bits = masked_error_count(
            logits.detach(), batch["bits"], batch["loss_mask"]
        )
    return float(loss.item()), int(errors.item()), int(valid_bits.item())


def train_one_epoch(
    model: nn.Module,
    generator,
    optimizer: torch.optim.Optimizer,
    num_samples: int,
    batch_size: int,
    log_interval: int,
) -> tuple[float, float]:
    model.train()
    total_bce = 0.0
    total_errors = 0
    total_bits = 0

    for step, current_bs in enumerate(batch_sizes(num_samples, batch_size), start=1):
        batch = generator.generate_batch(current_bs)
        bce_sum, errors, valid_bits = train_one_batch(model, batch, optimizer)
        count = max(valid_bits, 1)
        total_bce += bce_sum * valid_bits
        total_errors += errors
        total_bits += valid_bits

        if log_interval > 0 and step % log_interval == 0:
            print(
                f"  step {step:05d} | BCE {total_bce / max(total_bits, 1):.6f} | "
                f"BER {total_errors / max(total_bits, 1):.6f}"
            )

    return total_bce / max(total_bits, 1), total_errors / max(total_bits, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    generator,
    num_samples: int,
    batch_size: int,
    reset_seed: int | None = None,
) -> tuple[float, float, int, int]:
    model.eval()
    generator.reset(reset_seed)
    total_bce = 0.0
    total_errors = 0
    total_bits = 0

    for current_bs in batch_sizes(num_samples, batch_size):
        batch = generator.generate_batch(current_bs)
        logits = model(
            batch["Y"],
            batch["H_hat"],
            batch["P"],
            batch["N0"],
            batch["layer_mask"],
        )
        bce = masked_bce_with_logits(logits, batch["bits"], batch["loss_mask"])
        errors, valid_bits = masked_error_count(
            logits, batch["bits"], batch["loss_mask"]
        )
        count = int(valid_bits.item())
        total_bce += float(bce.item()) * count
        total_errors += int(errors.item())
        total_bits += count

    return total_bce / max(total_bits, 1), total_errors / max(total_bits, 1), total_errors, total_bits


@torch.no_grad()
def evaluate_snr_sweep(
    model: nn.Module,
    base_generator: SionnaSUMIMOBatchGenerator,
    transform: str,
    snr_list: list[float],
    num_samples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[dict]:
    """Evaluate BER at a fixed list of SNR points."""
    results = []
    for idx, snr in enumerate(snr_list):
        eval_seed = seed + idx * 1000
        gen = TransformGeneratorWrapper(
            SionnaSUMIMOBatchGenerator(
                base_generator.config,
                snr_db_min=snr,
                snr_db_max=snr,
                phase_mode="uniform",
                seed=eval_seed,
                device=device,
                channel_profile=base_generator.channel_profile,
            ),
            transform=transform,
            haar_seed=eval_seed + 50000,
        )
        _, ber, errors, valid_bits = evaluate(
            model, gen, num_samples, batch_size, reset_seed=eval_seed
        )
        results.append(
            {
                "snr_db": snr,
                "ber": ber,
                "bit_errors": errors,
                "valid_bits": valid_bits,
                "transform": transform,
            }
        )
        print(f"  SNR {snr:5.1f} dB | BER {ber:.6e} ({errors}/{valid_bits})")
    return results


# ---------------------------------------------------------------------------
# Build model config
# ---------------------------------------------------------------------------

MODEL_CONFIG = {
    "num_rx_ant": 4,
    "hidden_complex": 32,
    "zero_real": 22,
    "hidden_real": 66,
    "bits_per_symbol": 2,
    "num_iterations": 2,
    "kernel_size": 3,
    "zero_gate_hidden": 16,
}

DATA_CONFIG = SionnaSUMIMOConfig(
    num_ofdm_symbols=14,
    fft_size=72,
    subcarrier_spacing_hz=30e3,
    cyclic_prefix_length=0,
    bits_per_symbol=2,
    num_layers=4,
    num_rx_ant=4,
    total_tx_power=1.0,
    dmrs_symbol_indices=(2, 11),
    tdl_model="A",
    delay_spread_s=10e-9,
    carrier_frequency_hz=3.5e9,
    max_doppler_hz=200.0,
    normalize_channel=True,
    ls_interpolation_type="lin",
)


def build_model(device: torch.device) -> nn.Module:
    """Build the real CNN.  The factory internally computes the parameter budget
    from a reference phase-sensitive receiver."""
    model = build_su_mimo_model("su_mimo_real_cnn", dict(MODEL_CONFIG))
    return model.to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Train/eval real CNN under unitary transforms")
    p.add_argument("--transforms", nargs="+",
                   default=["identity", "haar", "qr", "svd", "polar"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--num_train", type=int, default=10000)
    p.add_argument("--num_val", type=int, default=2000)
    p.add_argument("--num_test", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--train_snr_min", type=float, default=-5.0)
    p.add_argument("--train_snr_max", type=float, default=20.0)
    p.add_argument("--eval_snr_min", type=float, default=-5.0)
    p.add_argument("--eval_snr_max", type=float, default=20.0)
    p.add_argument("--eval_snr_step", type=float, default=2.0)
    p.add_argument("--output_dir",
                   default="tests/representation_invariance/runs")
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training, load existing checkpoints.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_profile = legacy_channel_profile(DATA_CONFIG)
    eval_snr_list = list(
        torch.arange(args.eval_snr_min, args.eval_snr_max + 1e-9, args.eval_snr_step).tolist()
    )

    print(f"Device: {device}")
    print(f"Transforms: {args.transforms}")
    print(f"Topology: {DATA_CONFIG.num_layers}×{DATA_CONFIG.num_rx_ant} MIMO")
    print(f"Eval SNR: {eval_snr_list}")

    all_results: dict[str, list[dict]] = {}

    for idx, transform in enumerate(args.transforms):
        tag = f"{transform}_seed{args.seed}"
        ckpt_path = output_dir / f"model_{tag}.pt"
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(args.transforms)}] Transform: {transform}")
        print(f"{'='*60}")

        model = build_model(device)

        if args.skip_train and ckpt_path.exists():
            print(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
        else:
            # --- Build generators ---
            train_gen = TransformGeneratorWrapper(
                SionnaSUMIMOBatchGenerator(
                    DATA_CONFIG,
                    snr_db_min=args.train_snr_min,
                    snr_db_max=args.train_snr_max,
                    phase_mode="fixed",
                    seed=args.seed,
                    device=device,
                    channel_profile=train_profile,
                ),
                transform=transform,
                haar_seed=args.seed + 100,
            )
            val_gen = TransformGeneratorWrapper(
                SionnaSUMIMOBatchGenerator(
                    DATA_CONFIG,
                    snr_db_min=args.train_snr_min,
                    snr_db_max=args.train_snr_max,
                    phase_mode="uniform",
                    seed=args.seed + 200000,
                    device=device,
                    channel_profile=train_profile,
                ),
                transform=transform,
                haar_seed=args.seed + 200100,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            best_val_bce = math.inf

            for epoch in range(1, args.epochs + 1):
                train_gen.reset_profile_sampler(args.seed + epoch * 1009)
                train_bce, train_ber = train_one_epoch(
                    model, train_gen, optimizer,
                    args.num_train, args.batch_size, args.log_interval,
                )
                val_bce, val_ber, _, _ = evaluate(
                    model, val_gen, args.num_val, args.batch_size,
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
                        {"model_state": model.state_dict(), "transform": transform},
                        ckpt_path,
                    )
                    print(f"  -> best checkpoint saved")

            # Reload best
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])

        # --- Evaluate across SNR ---
        print(f"\nSNR sweep for transform={transform}:")
        base_gen_for_eval = SionnaSUMIMOBatchGenerator(
            DATA_CONFIG,
            snr_db_min=args.train_snr_min,
            snr_db_max=args.train_snr_max,
            phase_mode="uniform",
            seed=args.seed + 300000,
            device=device,
            channel_profile=train_profile,
        )
        results = evaluate_snr_sweep(
            model, base_gen_for_eval, transform,
            eval_snr_list, args.num_test, args.batch_size,
            args.seed + 777000, device,
        )
        all_results[transform] = results

    # ---- Plot ----
    print(f"\n{'='*60}")
    print("Plotting results...")
    plt.figure(figsize=(10, 6))

    markers = {"identity": "o", "haar": "s", "qr": "^", "svd": "D", "polar": "v"}
    colors = {
        "identity": "C0", "haar": "C1", "qr": "C2", "svd": "C3", "polar": "C4",
    }

    for transform in args.transforms:
        results = all_results[transform]
        snrs = [r["snr_db"] for r in results]
        bers = [r["ber"] for r in results]
        plt.semilogy(
            snrs, bers,
            marker=markers.get(transform, "x"),
            color=colors.get(transform, None),
            linestyle="-",
            label=transform,
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("SU-MIMO Real CNN: Unitary Representation Sensitivity\n"
              f"TDL-A, {DATA_CONFIG.num_layers}×{DATA_CONFIG.num_rx_ant} MIMO, "
              f"{DATA_CONFIG.num_ofdm_symbols} sym × {DATA_CONFIG.fft_size} SC")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = output_dir / "ber_vs_snr.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")

    # Also save raw CSV
    csv_path = output_dir / "results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["transform", "snr_db", "ber", "bit_errors", "valid_bits"])
        writer.writeheader()
        for transform, results in all_results.items():
            for r in results:
                writer.writerow(r)
    print(f"Results saved to {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
