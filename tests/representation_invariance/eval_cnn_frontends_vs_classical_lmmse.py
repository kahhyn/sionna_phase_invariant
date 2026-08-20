"""统一评估六种 CNN 预变换方案与经典 LMMSE 接收机。

所有接收机在每个 SNR 点共享完全相同的发送比特、信道、噪声和 LS
信道估计，避免不同随机样本给比较带来额外波动。经典基线使用项目已有的
Sionna LMMSE equalizer 和 QPSK APP soft demapper，不经过神经网络。

默认评估已经在 ``runs_unified_lin`` 中训练好的单 seed checkpoint::

    identity, qr, svd, polar, mf, lmmse, classical_lmmse

用法::

    python tests/representation_invariance/eval_cnn_frontends_vs_classical_lmmse.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data import SionnaSUMIMOBatchGenerator, legacy_channel_profile
from models import SionnaSUMIMOLMMSEBaseline
from tests.representation_invariance.train_eval_transforms import (
    DATA_CONFIG,
    apply_representation,
    build_model,
)


CNN_TRANSFORMS = ("identity", "qr", "svd", "polar", "mf", "lmmse")


def parse_snr_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--snr_list 至少需要一个 SNR 点。")
    return list(dict.fromkeys(values))


def _frame_values(value, batch_size: int, device: torch.device) -> torch.Tensor:
    """把标量或 [B,...] 张量转换成每帧一个数。"""
    tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
    if tensor.numel() == 1:
        return tensor.reshape(1).expand(batch_size)
    if tensor.shape[0] != batch_size:
        raise ValueError(f"无法把形状 {tuple(tensor.shape)} 映射到 B={batch_size}。")
    return tensor.reshape(batch_size, -1)[:, 0]


@torch.no_grad()
def transform_cnn_inputs(
    batch: dict,
    transform: str,
    gamma_scale: float,
    n0_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在共享原始 batch 上构造某个 CNN 所需的 Y、H_hat 和 N0。"""
    y = batch["Y"]
    h_hat = batch["H_hat"]
    b, nr, num_symbols, num_subcarriers = y.shape
    _, num_layers, _, _, _ = h_hat.shape

    n0_frame = _frame_values(batch["N0"], b, y.device)
    es_frame = _frame_values(batch["power_per_data_layer"], b, y.device)
    gamma_frame = gamma_scale * n0_frame / es_frame.clamp_min(1e-12)
    gamma_flat = (
        gamma_frame[:, None, None]
        .expand(b, num_symbols, num_subcarriers)
        .reshape(-1)
    )

    # 每个时频 RE 独立做空间预变换。
    y_flat = y.permute(0, 2, 3, 1).reshape(-1, nr)
    h_flat = h_hat.permute(0, 3, 4, 2, 1).reshape(-1, nr, num_layers)
    z_flat, h_seen_flat, w_flat = apply_representation(
        h_flat, y_flat, transform, gamma_flat
    )

    num_outputs = z_flat.shape[-1]
    y_new = (
        z_flat.reshape(b, num_symbols, num_subcarriers, num_outputs)
        .permute(0, 3, 1, 2)
        .contiguous()
        .to(torch.complex64)
    )
    h_new = (
        h_seen_flat.reshape(
            b, num_symbols, num_subcarriers, num_outputs, num_layers
        )
        .permute(0, 4, 3, 1, 2)
        .contiguous()
        .to(torch.complex64)
    )

    n0_for_cnn = batch["N0"]
    if n0_mode == "mean_effective":
        # 非酉预变换后噪声一般为有色噪声；该模式只传入每帧平均功率。
        noise_gain = (
            w_flat.abs().square().sum(dim=(-2, -1)) / float(num_outputs)
        )
        noise_gain = noise_gain.reshape(b, num_symbols, num_subcarriers).mean((1, 2))
        n0_for_cnn = (n0_frame * noise_gain).reshape(batch["N0"].shape)
    elif n0_mode != "original":
        raise ValueError(f"未知 n0_mode: {n0_mode}")

    return y_new, h_new, n0_for_cnn


def extract_data_bits_or_logits(
    dense: torch.Tensor,
    data_mask: torch.Tensor,
) -> torch.Tensor:
    """按 Sionna 的 symbol-major/bit-minor 顺序取出所有 data RE。"""
    b, num_layers, bits_per_symbol, _, _ = dense.shape
    flattened = []
    for layer in range(num_layers):
        # [B,bits,T,F] -> [B,T,F,bits]，随后只保留该层 data RE。
        layer_values = dense[:, layer].permute(0, 2, 3, 1)
        flattened.append(
            layer_values[:, data_mask[layer], :].reshape(b, -1)
        )
    return torch.stack(flattened, dim=1)


@torch.no_grad()
def classical_lmmse_logits(
    receiver: SionnaSUMIMOLMMSEBaseline,
    batch: dict,
) -> torch.Tensor:
    """复用项目正式经典基线，返回 [B,L,coded_bit] QPSK soft logits。"""
    y = batch["Y"].unsqueeze(1)
    h_hat = batch["H_hat"]
    err_var = batch["H_err_var"]

    # 生成器把总功率均分给各层；Sionna equalizer 采用单位能量符号模型，
    # 因而把每层符号功率吸收到信道和信道估计误差中。
    power_per_layer = (
        receiver.generator.config.total_tx_power / receiver.num_layers
    )
    h_hat = h_hat * power_per_layer**0.5
    err_var = err_var * power_per_layer

    # [B,L,R,T,F] -> [B,num_rx=1,R,num_tx=1,L,T,F]
    h_hat = h_hat.permute(0, 2, 1, 3, 4).unsqueeze(1).unsqueeze(3)
    err_var = err_var.permute(0, 2, 1, 3, 4).unsqueeze(1).unsqueeze(3)
    no = batch["N0"].view(batch["N0"].shape[0], 1, 1)
    x_hat, no_eff = receiver.equalizer(y, h_hat, err_var, no)
    llr = receiver.demapper(x_hat, no_eff)
    return llr.reshape(batch["Y"].shape[0], receiver.num_layers, -1)


def load_cnn_models(
    checkpoint_dir: Path,
    checkpoint_seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, dict]]:
    models = {}
    metadata = {}
    for transform in CNN_TRANSFORMS:
        checkpoint_path = checkpoint_dir / f"model_{transform}_seed{checkpoint_seed}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_transform = checkpoint.get("transform", transform)
        if checkpoint_transform != transform:
            raise ValueError(
                f"{checkpoint_path} 标记为 {checkpoint_transform}，预期为 {transform}。"
            )
        interpolation = checkpoint.get("ls_interpolation_type", "lin")
        if interpolation != "lin":
            raise ValueError(f"{checkpoint_path} 不是 lin checkpoint。")

        model = build_model(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models[transform] = model
        metadata[transform] = {
            "checkpoint": str(checkpoint_path),
            "gamma_scale": float(checkpoint.get("gamma_scale", 1.0)),
            "n0_mode": checkpoint.get("n0_mode", "original"),
            "ls_interpolation_type": interpolation,
        }
    return models, metadata


@torch.no_grad()
def evaluate_snr(
    snr_db: float,
    snr_index: int,
    models: dict[str, torch.nn.Module],
    metadata: dict[str, dict],
    num_test: int,
    batch_size: int,
    base_seed: int,
    device: torch.device,
    progress_interval: int,
) -> list[dict]:
    point_seed = base_seed + snr_index * 1000
    profile = legacy_channel_profile(DATA_CONFIG)
    generator = SionnaSUMIMOBatchGenerator(
        DATA_CONFIG,
        snr_db_min=snr_db,
        snr_db_max=snr_db,
        phase_mode="uniform",
        seed=point_seed,
        device=device,
        channel_profile=profile,
    )
    generator.reset(point_seed)
    classical_receiver = SionnaSUMIMOLMMSEBaseline(generator, csi="ls")
    data_mask = generator.resource_grid.build_type_grid()[0] == 0

    names = [f"{name}_cnn" for name in CNN_TRANSFORMS] + ["lmmse_classical"]
    totals = {
        name: {"bce_sum": 0.0, "bit_errors": 0, "valid_bits": 0}
        for name in names
    }

    num_batches = (num_test + batch_size - 1) // batch_size
    frames_done = 0
    for batch_index in range(num_batches):
        current_bs = min(batch_size, num_test - frames_done)
        batch = generator.generate_batch(current_bs)
        target = extract_data_bits_or_logits(batch["bits"], data_mask)

        for transform in CNN_TRANSFORMS:
            info = metadata[transform]
            y_input, h_input, n0_input = transform_cnn_inputs(
                batch,
                transform,
                info["gamma_scale"],
                info["n0_mode"],
            )
            dense_logits = models[transform](
                y_input,
                h_input,
                batch["P"],
                n0_input,
                batch["layer_mask"],
            )
            logits = extract_data_bits_or_logits(dense_logits, data_mask)
            name = f"{transform}_cnn"
            totals[name]["bce_sum"] += float(
                F.binary_cross_entropy_with_logits(logits, target, reduction="sum").item()
            )
            totals[name]["bit_errors"] += int(
                ((logits > 0) != target.bool()).sum().item()
            )
            totals[name]["valid_bits"] += target.numel()

        classical_logits = classical_lmmse_logits(classical_receiver, batch)
        if classical_logits.shape != target.shape:
            raise RuntimeError(
                f"经典 LMMSE 输出 {tuple(classical_logits.shape)} 与目标 "
                f"{tuple(target.shape)} 不一致。"
            )
        name = "lmmse_classical"
        totals[name]["bce_sum"] += float(
            F.binary_cross_entropy_with_logits(
                classical_logits, target, reduction="sum"
            ).item()
        )
        totals[name]["bit_errors"] += int(
            ((classical_logits > 0) != target.bool()).sum().item()
        )
        totals[name]["valid_bits"] += target.numel()

        frames_done += current_bs
        if progress_interval > 0 and (
            (batch_index + 1) % progress_interval == 0 or frames_done == num_test
        ):
            print(
                f"  SNR {snr_db:5.1f} dB | {frames_done:5d}/{num_test} frames",
                flush=True,
            )

    rows = []
    for receiver_name in names:
        total = totals[receiver_name]
        valid_bits = total["valid_bits"]
        rows.append(
            {
                "receiver": receiver_name,
                "snr_db": snr_db,
                "ber": total["bit_errors"] / valid_bits,
                "bce": total["bce_sum"] / valid_bits,
                "bit_errors": total["bit_errors"],
                "valid_bits": valid_bits,
                "num_frames": frames_done,
                "eval_seed": point_seed,
                "ls_interpolation_type": DATA_CONFIG.ls_interpolation_type,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare six frontend-specific CNNs with classical LMMSE"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="tests/representation_invariance/runs_unified_lin",
    )
    parser.add_argument("--checkpoint_seed", type=int, default=42)
    parser.add_argument(
        "--snr_list", default="-5,-3,-1,1,3,5,7,9,11,13,15,17,19"
    )
    parser.add_argument("--num_test", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_seed", type=int, default=777042)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress_interval", type=int, default=16)
    parser.add_argument(
        "--output_dir",
        default=(
            "tests/representation_invariance/runs_unified_lin/"
            "eval_cnn_vs_classical_lmmse"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_test <= 0 or args.batch_size <= 0:
        raise ValueError("--num_test 和 --batch_size 必须为正数。")
    if DATA_CONFIG.ls_interpolation_type != "lin":
        raise RuntimeError("本评估要求 DATA_CONFIG 使用 lin 插值。")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU。")
        device = torch.device("cpu")

    snr_values = parse_snr_list(args.snr_list)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models, metadata = load_cnn_models(
        checkpoint_dir, args.checkpoint_seed, device
    )

    print(f"Device: {device}")
    print(f"SNR: {snr_values}")
    print(f"Frames/SNR: {args.num_test}")
    print("Receivers: identity/QR/SVD/Polar/MF/LMMSE CNN + classical LMMSE")

    all_rows = []
    for snr_index, snr_db in enumerate(snr_values):
        print(f"\n[{snr_index + 1}/{len(snr_values)}] SNR={snr_db:g} dB", flush=True)
        rows = evaluate_snr(
            snr_db,
            snr_index,
            models,
            metadata,
            args.num_test,
            args.batch_size,
            args.eval_seed,
            device,
            args.progress_interval,
        )
        all_rows.extend(rows)
        for row in rows:
            print(
                f"    {row['receiver']:18s} BER={row['ber']:.6e} "
                f"({row['bit_errors']}/{row['valid_bits']})"
            )

    csv_path = output_dir / "results_cnn_vs_classical_lmmse.csv"
    fieldnames = [
        "receiver", "snr_db", "ber", "bce", "bit_errors", "valid_bits",
        "num_frames", "eval_seed", "ls_interpolation_type",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    labels = {
        "identity_cnn": "identity CNN",
        "qr_cnn": "QR CNN",
        "svd_cnn": "SVD CNN",
        "polar_cnn": "Polar CNN",
        "mf_cnn": "MF CNN",
        "lmmse_cnn": "LMMSE CNN",
        "lmmse_classical": "Classical LMMSE",
    }
    markers = {
        "identity_cnn": "o", "qr_cnn": "^", "svd_cnn": "D",
        "polar_cnn": "v", "mf_cnn": "P", "lmmse_cnn": "s",
        "lmmse_classical": "*",
    }
    colors = {
        "identity_cnn": "C0", "qr_cnn": "C2", "svd_cnn": "C3",
        "polar_cnn": "C4", "mf_cnn": "C1", "lmmse_cnn": "C6",
        "lmmse_classical": "black",
    }

    plt.figure(figsize=(11, 7))
    for receiver_name in labels:
        rows = [row for row in all_rows if row["receiver"] == receiver_name]
        plt.semilogy(
            [row["snr_db"] for row in rows],
            [row["ber"] for row in rows],
            marker=markers[receiver_name],
            color=colors[receiver_name],
            linestyle="--" if receiver_name == "lmmse_classical" else "-",
            linewidth=2.4 if receiver_name == "lmmse_classical" else 1.7,
            markersize=8 if receiver_name == "lmmse_classical" else 6,
            label=labels[receiver_name],
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(
        "SU-MIMO Real CNN Frontends vs. Classical LMMSE\n"
        f"TDL-A, {DATA_CONFIG.num_layers}x{DATA_CONFIG.num_rx_ant} MIMO, "
        f"{DATA_CONFIG.num_ofdm_symbols} sym x {DATA_CONFIG.fft_size} SC, LS-lin"
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plot_path = output_dir / "ber_cnn_frontends_vs_classical_lmmse.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()

    config_path = output_dir / "evaluation_config.json"
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                **vars(args),
                "snr_values": snr_values,
                "cnn_metadata": metadata,
                "classical_receiver": "SionnaSUMIMOLMMSEBaseline(csi='ls')",
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nCSV:  {csv_path}")
    print(f"Plot: {plot_path}")
    print(f"Config: {config_path}")


if __name__ == "__main__":
    main()
