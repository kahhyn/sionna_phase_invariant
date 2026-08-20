#!/usr/bin/env python3
"""Causal controls for Jacobian hotspot interpretations.

Consume artifacts from ``analyze_local_jacobian.py`` and compare a leading
Jacobian mode restricted to its highest-energy REs with three equal-L2 controls:

* the same RE energy with randomized complex phases;
* the same coefficients moved to non-hotspot REs matched on local SNR,
  edge/interior status, and pilot/data status;
* the same coefficients moved to unmatched random non-hotspot REs.

The primary metrics are masked raw-LLR central gain and RMS change. BER and
decision flips are saved only as secondary diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.analyze_local_jacobian import (  # noqa: E402
    EPS,
    apply_mask,
    get_mask,
    receiver_forward,
    sha256_file,
    tensor_rms,
)
from utils.checkpoints import load_receiver_checkpoint  # noqa: E402


CONDITIONS = (
    "hotspot_v1",
    "hotspot_phase_randomized",
    "matched_nonhotspot",
    "unmatched_nonhotspot",
)
METRICS = ("central_gain", "output_rms_change", "sign_flip_rate", "ber_change")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def deterministic_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_sample(path: Path, device: torch.device) -> dict[str, Tensor]:
    with np.load(path) as artifact:
        return {
            key: torch.from_numpy(np.asarray(artifact[key])).to(device)
            for key in artifact.files
            if key != "raw_llr"
        }


def load_mode(path: Path, device: torch.device) -> Tensor:
    with np.load(path) as artifact:
        modes = artifact["input_modes_complex"]
        if modes.shape[0] == 0:
            raise ValueError(f"No singular modes in {path}")
        return torch.from_numpy(np.asarray(modes[0])).to(device)


def tf_energy(value: Tensor) -> Tensor:
    result = value.abs().square()
    while result.ndim > 2:
        result = result.sum(dim=0)
    return result


def top_mask(energy: Tensor, fraction: float) -> Tensor:
    flat = energy.reshape(-1)
    count = max(1, int(math.ceil(fraction * flat.numel())))
    result = torch.zeros_like(flat, dtype=torch.bool)
    result[torch.topk(flat, count).indices] = True
    return result.reshape(energy.shape)


def make_edge_mask(shape: tuple[int, int], width: int, device: torch.device) -> Tensor:
    result = torch.zeros(shape, dtype=torch.bool, device=device)
    width = min(width, shape[0] // 2, shape[1] // 2)
    if width:
        result[:width, :] = True
        result[-width:, :] = True
        result[:, :width] = True
        result[:, -width:] = True
    return result


def expand_tf(mask: Tensor, reference: Tensor) -> Tensor:
    return mask.reshape([1] * (reference.ndim - 2) + list(mask.shape)).expand(reference.shape)


def normalize(direction: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(torch.view_as_real(direction).reshape(-1))
    if float(norm.item()) <= EPS:
        raise ValueError("Zero perturbation direction")
    return direction / norm


def restrict_mode(mode: Tensor, mask: Tensor) -> Tensor:
    return normalize(mode * expand_tf(mask, mode))


def randomize_phase(mode: Tensor, mask: Tensor, generator: torch.Generator) -> Tensor:
    phase = 2 * math.pi * torch.rand(
        mode.shape, dtype=mode.real.dtype, device=mode.device, generator=generator
    )
    return restrict_mode(mode * torch.exp(1j * phase), mask)


def random_destinations(mask: Tensor, count: int, generator: torch.Generator) -> Tensor:
    candidates = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
    if candidates.numel() < count:
        raise ValueError("Insufficient non-hotspot control positions")
    order = torch.randperm(candidates.numel(), device=mask.device, generator=generator)
    return candidates[order[:count]]


def matched_destinations(
    sources: Tensor,
    candidates: Tensor,
    local_snr: Tensor,
    edge: Tensor,
    pilot: Tensor,
    pool_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, float, float]:
    available = candidates.reshape(-1).clone()
    snr = local_snr.reshape(-1)
    edge = edge.reshape(-1)
    pilot = pilot.reshape(-1)
    order = torch.randperm(sources.numel(), device=sources.device, generator=generator)
    ordered_sources = sources[order]
    selected: list[Tensor] = []
    errors: list[float] = []
    for source in ordered_sources:
        valid = available & (edge == edge[source]) & (pilot == pilot[source])
        options = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if options.numel() == 0:
            raise ValueError("No exact edge/pilot-matched non-hotspot position")
        distances = (snr[options] - snr[source]).abs()
        pool = min(pool_size, options.numel())
        nearest = options[torch.topk(distances, pool, largest=False).indices]
        choice = nearest[
            torch.randint(pool, (1,), device=nearest.device, generator=generator).item()
        ]
        selected.append(choice)
        errors.append(float((snr[choice] - snr[source]).abs().item()))
        available[choice] = False
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    destination = torch.stack(selected)[inverse]
    return destination, float(np.mean(errors)), float(np.max(errors))


def relocate(mode: Tensor, sources: Tensor, destinations: Tensor) -> Tensor:
    freq_size = mode.shape[-1]
    result = torch.zeros_like(mode)
    for source, destination in zip(sources.tolist(), destinations.tolist()):
        st, sf = divmod(source, freq_size)
        dt, df = divmod(destination, freq_size)
        result[..., dt, df] = mode[..., st, sf]
    return normalize(result)


def masked_ber(raw: Tensor, sample: Mapping[str, Tensor], mask: Tensor | None) -> float:
    bits = sample.get("bits")
    if not isinstance(bits, Tensor) or bits.shape != raw.shape:
        return float("nan")
    return float(
        ((apply_mask(raw, mask) >= 0) != apply_mask(bits, mask).bool())
        .float()
        .mean()
        .item()
    )


def evaluate(
    model: nn.Module,
    sample: Mapping[str, Tensor],
    key: str,
    direction: Tensor,
    epsilon: float,
    mask_key: str,
) -> dict[str, float]:
    reference = sample[key]
    input_norm = torch.linalg.vector_norm(torch.view_as_real(reference).reshape(-1))
    step = epsilon * float(input_norm.item())
    delta = direction * step
    plus, minus = dict(sample), dict(sample)
    plus[key], minus[key] = reference + delta, reference - delta
    mask = get_mask(sample, mask_key)
    with torch.inference_mode():
        base_raw = receiver_forward(model, sample)
        plus_raw = receiver_forward(model, plus)
        minus_raw = receiver_forward(model, minus)
    base = apply_mask(base_raw, mask)
    plus_out = apply_mask(plus_raw, mask)
    minus_out = apply_mask(minus_raw, mask)
    central_l2 = float(torch.linalg.vector_norm(plus_out - minus_out).item()) / 2
    rms = np.mean(
        [float(tensor_rms(plus_out - base).item()), float(tensor_rms(minus_out - base).item())]
    )
    flips = np.mean(
        [
            float(((plus_out >= 0) != (base >= 0)).float().mean().item()),
            float(((minus_out >= 0) != (base >= 0)).float().mean().item()),
        ]
    )
    base_ber = masked_ber(base_raw, sample, mask)
    perturbed_ber = np.mean([masked_ber(plus_raw, sample, mask), masked_ber(minus_raw, sample, mask)])
    return {
        "input_step_l2": step,
        "central_gain": central_l2 / max(step, EPS),
        "output_rms_change": float(rms),
        "sign_flip_rate": float(flips),
        "base_ber": base_ber,
        "perturbed_ber": float(perturbed_ber),
        "ber_change": float(perturbed_ber - base_ber),
    }


def make_row(
    base: Mapping[str, Any],
    condition: str,
    trial: int,
    metrics: Mapping[str, float],
    match_mae: float = float("nan"),
    match_max: float = float("nan"),
) -> dict[str, Any]:
    return {
        **base,
        "condition": condition,
        "trial": trial,
        **metrics,
        "matched_local_snr_mae_db": match_mae,
        "matched_local_snr_max_error_db": match_max,
    }


def descriptive(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    return {
        "mean": float(data.mean()),
        "std": float(data.std(ddof=1)) if data.size > 1 else 0.0,
        "min": float(data.min()),
        "max": float(data.max()),
    }


def summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # Aggregate samples/trials within a generated batch, then report four-batch variation.
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["snr_db"], row["eval_seed"], row["key"], row["condition"])].append(row)
    batch_means: dict[tuple[Any, ...], dict[str, float]] = {}
    for group, selected in grouped.items():
        batch_means[group] = {
            metric: float(np.mean([float(row[metric]) for row in selected]))
            for metric in METRICS
        }
    output: list[dict[str, Any]] = []
    cells = sorted({(group[0], group[2], group[3]) for group in batch_means})
    for snr, key, condition in cells:
        selected = [
            metrics
            for group, metrics in batch_means.items()
            if group[0] == snr and group[2] == key and group[3] == condition
        ]
        row: dict[str, Any] = {
            "snr_db": snr,
            "key": key,
            "condition": condition,
            "num_batches": len(selected),
        }
        for metric in METRICS:
            row.update(
                {
                    f"{metric}_{name}": value
                    for name, value in descriptive([item[metric] for item in selected]).items()
                }
            )
        output.append(row)
    return output


def paired_effects(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    treatment = {
        (row["run"], row["sample_id"], row["key"]): row
        for row in rows
        if row["condition"] == "hotspot_v1"
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        if row["condition"] == "hotspot_v1":
            continue
        base = treatment[(row["run"], row["sample_id"], row["key"])]
        output.append(
            {
                "run": row["run"],
                "snr_db": row["snr_db"],
                "eval_seed": row["eval_seed"],
                "sample_id": row["sample_id"],
                "key": row["key"],
                "control": row["condition"],
                "trial": row["trial"],
                "rms_difference": float(base["output_rms_change"]) - float(row["output_rms_change"]),
                "rms_ratio": float(base["output_rms_change"]) / max(float(row["output_rms_change"]), EPS),
                "gain_difference": float(base["central_gain"]) - float(row["central_gain"]),
                "gain_ratio": float(base["central_gain"]) / max(float(row["central_gain"]), EPS),
            }
        )
    return output


def discover_runs(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    output: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob("snr*_evalseed*/config.json")):
        run_dir = path.parent
        status = json.loads((run_dir / "status.json").read_text())
        if status.get("failures"):
            raise RuntimeError(f"Source run contains failures: {run_dir}")
        output.append((run_dir, json.loads(path.read_text())))
    if not output:
        raise FileNotFoundError(f"No source runs found under {root}")
    return output


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = prepare_output(Path(args.output_dir), args.overwrite)
    source_runs = discover_runs(input_root)
    checkpoints = {str(config["checkpoint"]) for _, config in source_runs}
    checkpoint_hashes = {str(config["checkpoint_sha256"]) for _, config in source_runs}
    if len(checkpoints) != 1 or len(checkpoint_hashes) != 1:
        raise ValueError("Source runs do not share one checkpoint identity")
    checkpoint_path = Path(args.checkpoint or next(iter(checkpoints))).expanduser().resolve()
    if sha256_file(checkpoint_path) != next(iter(checkpoint_hashes)):
        raise ValueError("Checkpoint hash differs from source-run provenance")
    model, checkpoint = load_receiver_checkpoint(
        checkpoint_path, device, bits_per_symbol=2, required_data_backend="sionna"
    )
    model.to(device).eval().requires_grad_(False)

    rows: list[dict[str, Any]] = []
    sample_count = 0
    for run_dir, config in source_runs:
        input_paths = sorted((run_dir / "artifacts").glob("sample_*_inputs.npz"))
        if args.max_samples_per_run:
            input_paths = input_paths[: args.max_samples_per_run]
        sample_count += len(input_paths)
        for input_path in input_paths:
            sample_id = int(input_path.stem.split("_")[1])
            sample = load_sample(input_path, device)
            for key in args.keys:
                mode = load_mode(
                    run_dir / "artifacts" / f"sample_{sample_id:04d}_{key}_jacobian.npz",
                    device,
                )
                energy = tf_energy(mode)
                hotspot = top_mask(energy, args.hotspot_fraction)
                sources = torch.nonzero(hotspot.reshape(-1), as_tuple=False).reshape(-1)
                nonhotspot = ~hotspot
                edge = make_edge_mask(tuple(energy.shape), args.edge_width, device)
                pilot = tf_energy(sample["P"]) > 0
                local_snr = 10 * torch.log10(
                    torch.clamp(tf_energy(sample["H_hat"]) / max(float(sample["N0"].mean().item()), EPS), min=EPS)
                )
                base = {
                    "run": run_dir.name,
                    "snr_db": float(config["snr_db"]),
                    "eval_seed": int(config["seed"]),
                    "sample_id": sample_id,
                    "key": key,
                    "hotspot_re_count": int(sources.numel()),
                    "hotspot_edge_fraction": float(edge[hotspot].float().mean().item()),
                    "hotspot_pilot_fraction": float(pilot[hotspot].float().mean().item()),
                }
                treatment = restrict_mode(mode, hotspot)
                rows.append(
                    make_row(
                        base,
                        "hotspot_v1",
                        -1,
                        evaluate(model, sample, key, treatment, args.epsilon, str(config["mask_key"])),
                    )
                )
                for trial in range(args.trials):
                    generator = torch.Generator(device=device)
                    generator.manual_seed(
                        deterministic_seed(args.seed, config["seed"], sample_id, key, trial)
                    )
                    phase_control = randomize_phase(mode, hotspot, generator)
                    rows.append(
                        make_row(
                            base,
                            "hotspot_phase_randomized",
                            trial,
                            evaluate(model, sample, key, phase_control, args.epsilon, str(config["mask_key"])),
                        )
                    )
                    matched, match_mae, match_max = matched_destinations(
                        sources, nonhotspot, local_snr, edge, pilot, args.match_pool, generator
                    )
                    rows.append(
                        make_row(
                            base,
                            "matched_nonhotspot",
                            trial,
                            evaluate(model, sample, key, relocate(mode, sources, matched), args.epsilon, str(config["mask_key"])),
                            match_mae,
                            match_max,
                        )
                    )
                    unmatched = random_destinations(nonhotspot, int(sources.numel()), generator)
                    rows.append(
                        make_row(
                            base,
                            "unmatched_nonhotspot",
                            trial,
                            evaluate(model, sample, key, relocate(mode, sources, unmatched), args.epsilon, str(config["mask_key"])),
                        )
                    )

    effects = paired_effects(rows)
    summary = summaries(rows)
    write_csv(output_dir / "causal_per_trial.csv", rows)
    write_csv(output_dir / "paired_effects.csv", effects)
    write_csv(output_dir / "causal_summary.csv", summary)
    expected = sample_count * len(args.keys) * (1 + 3 * args.trials)
    provenance = {
        **vars(args),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_model_name": checkpoint.get("model_name", "unknown"),
        "source_run_count": len(source_runs),
        "source_sample_count": sample_count,
        "conditions": list(CONDITIONS),
        "primary_metric": "masked raw-LLR central gain and RMS change",
    }
    (output_dir / "config.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    status = {
        "rows": len(rows),
        "expected_rows": expected,
        "paired_effect_rows": len(effects),
        "summary_rows": len(summary),
        "complete": len(rows) == expected,
    }
    (output_dir / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if len(rows) != expected:
        raise RuntimeError(f"Incomplete result grid: {len(rows)} / {expected}")
    print(f"Completed {len(source_runs)} runs, {sample_count} samples, {len(rows)} rows")
    print(f"Saved to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Causal controls for saved Jacobian hotspots")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=991001)
    parser.add_argument("--keys", nargs="+", choices=("Y", "H_hat"), default=("Y", "H_hat"))
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--hotspot-fraction", type=float, default=0.01)
    parser.add_argument("--edge-width", type=int, default=1)
    parser.add_argument("--match-pool", type=int, default=1)
    parser.add_argument("--max-samples-per-run", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.epsilon <= 0 or args.trials <= 0 or args.match_pool <= 0:
        parser.error("epsilon, trials, and match-pool must be positive")
    if not 0 < args.hotspot_fraction < 1:
        parser.error("hotspot-fraction must be in (0, 1)")
    if args.edge_width < 0 or args.max_samples_per_run < 0:
        parser.error("edge-width and max-samples-per-run must be non-negative")
    run(args)


if __name__ == "__main__":
    main()
