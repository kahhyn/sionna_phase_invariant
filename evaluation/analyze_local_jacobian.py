#!/usr/bin/env python3
"""Local input-sensitivity, singular-mode, and Jacobian analysis for Sionna receivers.

The primary Jacobian metrics are computed on valid coded/data-bit logits only
(``loss_mask`` by default) and on raw LLRs by default.  The script supports
the uncoded BER/SNR generator and the LDPC/Eb/N0 generator explicitly so that
the operating point cannot be confused across evaluation protocols.

For Y and H_hat, finite differences use a relative L2 perturbation:

    ||delta x||_2 / ||x||_2 = epsilon.

For N0, the differentiated coordinate is log(N0), and epsilon is an absolute
L2 step in that coordinate.  ``common_phase`` is a separate directional
sensitivity measured per radian, not a partial Jacobian norm.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import (  # noqa: E402
    Sionna5GLDPCBatchGenerator,
    SionnaLDPC5GConfig,
    SionnaOFDMBatchGenerator,
    SionnaOFDMConfig,
    legacy_channel_profile,
    load_channel_profile,
    profile_backend_label,
    profile_component,
)
from utils.checkpoints import load_receiver_checkpoint  # noqa: E402


EPS = 1e-12


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def flatten_real(x: Tensor) -> Tensor:
    if torch.is_complex(x):
        return torch.view_as_real(x).reshape(-1)
    return x.reshape(-1)


def unflatten_real(vector: Tensor, reference: Tensor) -> Tensor:
    if torch.is_complex(reference):
        pair = vector.reshape(*reference.shape, 2).contiguous()
        return torch.view_as_complex(pair)
    return vector.reshape(reference.shape).to(reference.dtype)


def vector_norm(x: Tensor) -> Tensor:
    return torch.linalg.vector_norm(flatten_real(x))


def unit_vector(vector: Tensor) -> Tensor:
    return vector / torch.clamp(torch.linalg.vector_norm(vector), min=EPS)


def random_unit_vector(reference: Tensor, generator: torch.Generator) -> Tensor:
    vector = torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )
    return unit_vector(vector)


def tensor_rms(x: Tensor) -> Tensor:
    return torch.sqrt(torch.mean(torch.abs(x) ** 2))


def select_sample(batch: Mapping[str, Tensor], index: int) -> dict[str, Tensor]:
    batch_size = infer_batch_size(batch)
    sample: dict[str, Tensor] = {}
    for key, value in batch.items():
        if not isinstance(value, Tensor):
            continue
        if value.ndim > 0 and value.shape[0] == batch_size:
            sample[key] = value[index : index + 1]
        else:
            sample[key] = value
    return sample


def infer_batch_size(batch: Mapping[str, Tensor]) -> int:
    for key in ("Y", "H_hat", "N0"):
        value = batch.get(key)
        if isinstance(value, Tensor) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("Cannot infer batch size from Y, H_hat, or N0.")


def validate_batch(batch: Mapping[str, Tensor], mask_key: str) -> None:
    required = ["Y", "H_hat", "P", "N0"]
    if mask_key != "none":
        required.append(mask_key)
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Missing batch keys: {missing}")

    batch_size = infer_batch_size(batch)
    for key in ("Y", "H_hat", "P", "N0"):
        value = batch[key]
        if value.ndim == 0 or value.shape[0] != batch_size:
            raise ValueError(
                f"Batch tensor {key} has incompatible shape {tuple(value.shape)}."
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"Batch tensor {key} contains NaN or Inf.")
    if not torch.is_complex(batch["Y"]) or not torch.is_complex(batch["H_hat"]):
        raise TypeError("Y and H_hat must be complex tensors.")
    if torch.is_complex(batch["N0"]):
        raise TypeError("N0 must be real.")
    if bool((batch["N0"] <= 0).any().item()):
        raise ValueError("N0 must be strictly positive.")


def receiver_forward(model: nn.Module, batch: Mapping[str, Tensor]) -> Tensor:
    output = model(batch["Y"], batch["H_hat"], batch["P"], batch["N0"])
    if not isinstance(output, Tensor):
        raise TypeError(f"Receiver output must be a Tensor, got {type(output)!r}.")
    if output.ndim < 2 or output.shape[0] != batch["Y"].shape[0]:
        raise ValueError(f"Unexpected receiver output shape {tuple(output.shape)}.")
    if torch.is_complex(output):
        raise TypeError("Receiver LLR output must be real.")
    if not bool(torch.isfinite(output).all().item()):
        raise ValueError("Receiver output contains NaN or Inf.")
    return output


def output_transform(llr: Tensor, transform: str, tau: float) -> Tensor:
    if transform == "none":
        return llr
    if transform == "tanh":
        if tau <= 0:
            raise ValueError("--llr-tau must be positive.")
        return torch.tanh(llr / tau)
    if transform == "sigmoid":
        return torch.sigmoid(llr)
    raise ValueError(f"Unknown output transform: {transform}")


def get_mask(batch: Mapping[str, Tensor], mask_key: str) -> Tensor | None:
    if mask_key == "none":
        return None
    value = batch.get(mask_key)
    if not isinstance(value, Tensor):
        raise KeyError(f"Mask key {mask_key!r} is not a tensor in the batch.")
    return value.bool()


def apply_mask(output: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return output.reshape(output.shape[0], -1)
    try:
        expanded = torch.broadcast_to(mask, output.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} cannot broadcast to "
            f"output shape {tuple(output.shape)}."
        ) from exc
    counts = expanded.reshape(output.shape[0], -1).sum(dim=1)
    if not bool((counts == counts[0]).all().item()) or int(counts[0].item()) <= 0:
        raise ValueError("Mask must select the same positive count per sample.")
    return output[expanded].reshape(output.shape[0], int(counts[0].item()))


def get_outputs(
    model: nn.Module,
    sample: Mapping[str, Tensor],
    transform: str,
    tau: float,
    mask_key: str,
) -> tuple[Tensor, Tensor]:
    raw = apply_mask(receiver_forward(model, sample), get_mask(sample, mask_key))
    return raw, output_transform(raw, transform, tau)


def rms_change(a: Tensor, b: Tensor) -> Tensor:
    diff = (a - b).reshape(a.shape[0], -1)
    return torch.sqrt(torch.mean(diff.square(), dim=1))


def sign_flip_rate(a: Tensor, b: Tensor) -> Tensor:
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    return ((a >= 0) != (b >= 0)).float().mean(dim=1)


def rotate_tensor(x: Tensor, phase: Tensor, ri_dim: int) -> Tensor:
    if torch.is_complex(x):
        shape = [phase.shape[0]] + [1] * (x.ndim - 1)
        return x * torch.exp(1j * phase.reshape(shape))

    dim = ri_dim if ri_dim >= 0 else x.ndim + ri_dim
    if dim < 0 or dim >= x.ndim or x.shape[dim] != 2:
        raise ValueError(
            "Real-valued phase rotation requires a size-2 real/imag channel."
        )
    real = x.select(dim, 0)
    imag = x.select(dim, 1)
    shape = [phase.shape[0]] + [1] * (real.ndim - 1)
    cosine = torch.cos(phase).reshape(shape)
    sine = torch.sin(phase).reshape(shape)
    return torch.stack(
        (real * cosine - imag * sine, real * sine + imag * cosine), dim=dim
    )


@dataclass
class PerturbationPair:
    plus: dict[str, Tensor]
    minus: dict[str, Tensor]
    coordinate: str
    step_l2: float
    relative_step_l2: float


def make_perturbation_pair(
    sample: Mapping[str, Tensor],
    probe: str,
    epsilon: float,
    generator: torch.Generator,
    n0_floor: float,
    ri_dim: int,
) -> PerturbationPair:
    plus = dict(sample)
    minus = dict(sample)

    if probe in ("Y", "H_hat"):
        reference = sample[probe]
        coordinate = flatten_real(reference)
        input_norm = float(torch.linalg.vector_norm(coordinate).item())
        scale = max(input_norm, EPS)
        direction = random_unit_vector(coordinate, generator)
        delta_vector = epsilon * scale * direction
        delta = unflatten_real(delta_vector, reference)
        plus[probe] = reference + delta
        minus[probe] = reference - delta
        return PerturbationPair(
            plus=plus,
            minus=minus,
            coordinate=probe,
            step_l2=float(torch.linalg.vector_norm(delta_vector).item()),
            relative_step_l2=epsilon if input_norm > EPS else float("nan"),
        )

    if probe == "N0":
        reference = sample["N0"]
        eta = torch.log(torch.clamp(reference, min=n0_floor))
        coordinate = flatten_real(eta)
        direction = random_unit_vector(coordinate, generator)
        delta_vector = epsilon * direction
        delta = unflatten_real(delta_vector, eta)
        plus["N0"] = torch.exp(eta + delta)
        minus["N0"] = torch.exp(eta - delta)
        return PerturbationPair(
            plus=plus,
            minus=minus,
            coordinate="log_N0",
            step_l2=float(torch.linalg.vector_norm(delta_vector).item()),
            relative_step_l2=float("nan"),
        )

    if probe == "common_phase":
        batch_size = sample["Y"].shape[0]
        phase = torch.full(
            (batch_size,), epsilon, dtype=torch.float32, device=sample["Y"].device
        )
        plus["Y"] = rotate_tensor(sample["Y"], phase, ri_dim)
        plus["H_hat"] = rotate_tensor(sample["H_hat"], phase, ri_dim)
        minus["Y"] = rotate_tensor(sample["Y"], -phase, ri_dim)
        minus["H_hat"] = rotate_tensor(sample["H_hat"], -phase, ri_dim)
        return PerturbationPair(
            plus=plus,
            minus=minus,
            coordinate="common_phase_rad",
            step_l2=epsilon,
            relative_step_l2=float("nan"),
        )

    raise ValueError(f"Unsupported probe: {probe}")


@dataclass
class FiniteDifferenceMetrics:
    sample_id: int
    probe: str
    coordinate: str
    epsilon: float
    trials: int
    directional_lipschitz_mean: float
    directional_lipschitz_max: float
    directional_lipschitz_p95: float
    output_rms_change_mean: float
    raw_llr_rms_change_mean: float
    sign_flip_rate_mean: float
    input_step_l2_mean: float
    relative_input_step_l2: float
    num_outputs: int
    llr_abs_mean: float
    llr_abs_p95: float
    llr_abs_p99: float
    llr_abs_max: float


def finite_difference_one_sample(
    model: nn.Module,
    sample: Mapping[str, Tensor],
    sample_id: int,
    probe: str,
    args: argparse.Namespace,
) -> FiniteDifferenceMetrics:
    with torch.inference_mode():
        base_raw, base_out = get_outputs(
            model, sample, args.output_transform, args.llr_tau, args.mask_key
        )

    ratios: list[float] = []
    output_changes: list[float] = []
    raw_changes: list[float] = []
    flips: list[float] = []
    steps: list[float] = []
    relative_steps: list[float] = []
    coordinate_name = probe

    offset = {"Y": 10_000, "H_hat": 20_000, "N0": 30_000,
              "common_phase": 40_000}[probe]
    effective_trials = 1 if probe == "common_phase" else args.trials

    for trial in range(effective_trials):
        generator = torch.Generator(device=base_raw.device)
        generator.manual_seed(args.seed + offset + sample_id * 1000 + trial)
        pair = make_perturbation_pair(
            sample,
            probe,
            args.epsilon,
            generator,
            args.n0_floor,
            args.real_imag_channel_dim,
        )
        coordinate_name = pair.coordinate

        with torch.inference_mode():
            plus_raw, plus_out = get_outputs(
                model, pair.plus, args.output_transform, args.llr_tau, args.mask_key
            )
            minus_raw, minus_out = get_outputs(
                model, pair.minus, args.output_transform, args.llr_tau, args.mask_key
            )

        central_output_l2 = float(
            torch.linalg.vector_norm(flatten_real(plus_out - minus_out)).item()
        ) / 2.0
        ratios.append(central_output_l2 / max(pair.step_l2, EPS))
        output_changes.extend(
            [
                float(rms_change(plus_out, base_out)[0].item()),
                float(rms_change(minus_out, base_out)[0].item()),
            ]
        )
        raw_changes.extend(
            [
                float(rms_change(plus_raw, base_raw)[0].item()),
                float(rms_change(minus_raw, base_raw)[0].item()),
            ]
        )
        flips.extend(
            [
                float(sign_flip_rate(plus_raw, base_raw)[0].item()),
                float(sign_flip_rate(minus_raw, base_raw)[0].item()),
            ]
        )
        steps.append(pair.step_l2)
        relative_steps.append(pair.relative_step_l2)

    abs_llr = base_raw.detach().abs().reshape(-1)
    quantiles = torch.quantile(
        abs_llr,
        torch.tensor([0.95, 0.99], device=abs_llr.device, dtype=abs_llr.dtype),
    ).tolist()
    ratio_array = np.asarray(ratios, dtype=np.float64)
    finite_relative = np.asarray(relative_steps, dtype=np.float64)
    finite_relative = finite_relative[np.isfinite(finite_relative)]

    return FiniteDifferenceMetrics(
        sample_id=sample_id,
        probe=probe,
        coordinate=coordinate_name,
        epsilon=args.epsilon,
        trials=effective_trials,
        directional_lipschitz_mean=float(ratio_array.mean()),
        directional_lipschitz_max=float(ratio_array.max()),
        directional_lipschitz_p95=float(np.percentile(ratio_array, 95)),
        output_rms_change_mean=float(np.mean(output_changes)),
        raw_llr_rms_change_mean=float(np.mean(raw_changes)),
        sign_flip_rate_mean=float(np.mean(flips)),
        input_step_l2_mean=float(np.mean(steps)),
        relative_input_step_l2=(
            float(finite_relative.mean()) if finite_relative.size else float("nan")
        ),
        num_outputs=int(base_out.numel()),
        llr_abs_mean=float(abs_llr.mean().item()),
        llr_abs_p95=float(quantiles[0]),
        llr_abs_p99=float(quantiles[1]),
        llr_abs_max=float(abs_llr.max().item()),
    )


@dataclass
class JacobianMetrics:
    sample_id: int
    key: str
    mode_index: int
    input_coordinate: str
    spectral_norm: float
    num_inputs: int
    num_outputs: int
    restarts: int
    best_restart: int
    iterations_used: int
    converged: bool
    input_top_1pct_energy: float
    output_top_1pct_energy: float
    input_effective_support: float
    output_effective_support: float


def jacobian_coordinate(
    sample: Mapping[str, Tensor], key: str, n0_floor: float
) -> tuple[Tensor, Callable[[Tensor], Tensor], str]:
    if key in ("Y", "H_hat"):
        reference = sample[key].detach()
        coordinate = flatten_real(reference).detach()

        def decoder(vector: Tensor) -> Tensor:
            return unflatten_real(vector, reference)

        return coordinate, decoder, key

    if key == "N0":
        n0 = sample["N0"].detach()
        eta_reference = torch.log(torch.clamp(n0, min=n0_floor))
        coordinate = flatten_real(eta_reference).detach()

        def decoder(vector: Tensor) -> Tensor:
            return torch.exp(unflatten_real(vector, eta_reference))

        return coordinate, decoder, "log_N0"

    raise ValueError("Jacobian keys must be Y, H_hat, or N0.")


def orthogonalize(vector: Tensor, basis: Sequence[Tensor]) -> Tensor:
    """Project ``vector`` away from an existing orthonormal basis."""
    result = vector
    for item in basis:
        result = result - torch.dot(result, item) * item
    return result


def energy_summary(energy: Tensor, fraction: float = 0.01) -> tuple[float, float]:
    """Return top-fraction energy and inverse-participation support size."""
    flat = energy.detach().reshape(-1).to(torch.float64)
    total = float(flat.sum().item())
    if total <= EPS:
        return float("nan"), float("nan")
    probabilities = flat / total
    count = max(1, int(math.ceil(fraction * probabilities.numel())))
    top_energy = float(torch.topk(probabilities, count).values.sum().item())
    effective_support = float(1.0 / torch.sum(probabilities.square()).item())
    return top_energy, effective_support


def sum_to_tf(value: Tensor) -> Tensor:
    """Sum every axis before the final time/frequency axes."""
    result = value
    while result.ndim > 2:
        result = result.sum(dim=0)
    return result


def reduce_energy_to_tf(value: Tensor) -> Tensor:
    """Aggregate squared magnitude over every axis before the final T/F axes."""
    return sum_to_tf(torch.abs(value).square())


def input_mode_energy(mode: Tensor, reference: Tensor) -> Tensor | None:
    if reference.ndim < 2 or not torch.is_complex(reference):
        return None
    return reduce_energy_to_tf(unflatten_real(mode, reference))


def unmask_output_vector(
    vector: Tensor, raw_output: Tensor, mask: Tensor | None
) -> Tensor:
    if mask is None:
        return vector.reshape(raw_output.shape)
    expanded = torch.broadcast_to(mask, raw_output.shape)
    full = torch.zeros_like(raw_output)
    if int(expanded.sum().item()) != vector.numel():
        raise ValueError("Masked Jacobian output size does not match the output mask.")
    full[expanded] = vector
    return full


def hutchinson_input_diagonal(
    function: Callable[[Tensor], Tensor],
    coordinate: Tensor,
    samples: int,
    generator: torch.Generator,
) -> Tensor:
    """Estimate diag(J^T J) with Rademacher output probes."""
    if samples <= 0:
        return torch.full_like(coordinate, float("nan"))
    from torch.func import vjp

    output, vjp_fn = vjp(function, coordinate)
    accumulation = torch.zeros_like(coordinate)
    for _ in range(samples):
        signs = torch.randint(
            0,
            2,
            output.shape,
            generator=generator,
            device=output.device,
        ).to(output.dtype)
        signs = signs.mul_(2.0).sub_(1.0)
        jt_probe = vjp_fn(signs)[0]
        accumulation += jt_probe.detach().square()
    return accumulation / float(samples)


def jacobian_singular_modes(
    model: nn.Module,
    sample: Mapping[str, Tensor],
    sample_id: int,
    key: str,
    args: argparse.Namespace,
) -> tuple[list[JacobianMetrics], dict[str, Tensor]]:
    try:
        from torch.func import jvp, vjp
    except ImportError as exc:
        raise RuntimeError("This analysis requires torch.func in PyTorch 2.x.") from exc

    coordinate, decoder, coordinate_name = jacobian_coordinate(
        sample, key, args.n0_floor
    )

    def function(vector: Tensor) -> Tensor:
        modified = dict(sample)
        modified[key] = decoder(vector)
        raw = apply_mask(receiver_forward(model, modified), get_mask(modified, args.mask_key))
        return output_transform(raw, args.output_transform, args.llr_tau).reshape(-1)

    with torch.no_grad():
        num_outputs = int(function(coordinate).numel())
    raw_output = receiver_forward(model, sample).detach()
    output_mask = get_mask(sample, args.mask_key)
    max_modes = min(args.jacobian_top_k, coordinate.numel(), num_outputs)
    basis: list[Tensor] = []
    input_modes: list[Tensor] = []
    output_modes: list[Tensor] = []
    metrics: list[JacobianMetrics] = []
    key_offset = {"Y": 1, "H_hat": 2, "N0": 3}[key]
    for mode_index in range(max_modes):
        best_sigma = -1.0
        best_direction: Tensor | None = None
        best_jv: Tensor | None = None
        best_restart = -1
        best_iterations = 0
        best_converged = False

        for restart in range(args.jacobian_restarts):
            generator = torch.Generator(device=coordinate.device)
            generator.manual_seed(
                args.seed
                + 90_000
                + sample_id * 1000
                + key_offset * 100
                + mode_index * 10
                + restart
            )
            direction = orthogonalize(
                random_unit_vector(coordinate, generator), basis
            )
            if float(torch.linalg.vector_norm(direction).item()) <= EPS:
                continue
            direction = unit_vector(direction)
            previous_sigma: float | None = None
            converged = False
            iterations_used = 0

            for iteration in range(1, args.jacobian_iters + 1):
                _, jv = jvp(function, (coordinate,), (direction,))
                sigma = float(torch.linalg.vector_norm(jv).detach().item())
                iterations_used = iteration
                if sigma <= EPS:
                    converged = True
                    break

                _, vjp_fn = vjp(function, coordinate)
                jt_jv = orthogonalize(vjp_fn(jv)[0], basis)
                jt_norm = float(torch.linalg.vector_norm(jt_jv).detach().item())
                if jt_norm <= EPS:
                    converged = True
                    break
                next_direction = unit_vector(jt_jv)

                if previous_sigma is not None:
                    relative_change = abs(sigma - previous_sigma) / max(sigma, EPS)
                    if relative_change <= args.jacobian_tol:
                        direction = next_direction
                        converged = True
                        break
                direction = next_direction
                previous_sigma = sigma

            _, final_jv = jvp(function, (coordinate,), (direction,))
            final_sigma = float(torch.linalg.vector_norm(final_jv).detach().item())
            if final_sigma > best_sigma:
                best_sigma = final_sigma
                best_direction = direction.detach()
                best_jv = final_jv.detach()
                best_restart = restart
                best_iterations = iterations_used
                best_converged = converged

        if best_direction is None or best_jv is None or best_sigma <= EPS:
            break
        best_direction = unit_vector(orthogonalize(best_direction, basis)).detach()
        _, best_jv = jvp(function, (coordinate,), (best_direction,))
        best_jv = best_jv.detach()
        best_sigma = float(torch.linalg.vector_norm(best_jv).item())
        output_mode = best_jv / max(best_sigma, EPS)
        full_output_mode = unmask_output_vector(output_mode, raw_output, output_mask)
        input_energy = input_mode_energy(best_direction, sample[key])
        output_energy = reduce_energy_to_tf(full_output_mode)
        if input_energy is None:
            input_top, input_support = 1.0, 1.0
        else:
            input_top, input_support = energy_summary(input_energy)
        output_top, output_support = energy_summary(output_energy)

        basis.append(best_direction)
        input_modes.append(best_direction)
        output_modes.append(output_mode)
        metrics.append(
            JacobianMetrics(
                sample_id=sample_id,
                key=key,
                mode_index=mode_index + 1,
                input_coordinate=coordinate_name,
                spectral_norm=best_sigma,
                num_inputs=int(coordinate.numel()),
                num_outputs=num_outputs,
                restarts=args.jacobian_restarts,
                best_restart=best_restart,
                iterations_used=best_iterations,
                converged=best_converged,
                input_top_1pct_energy=input_top,
                output_top_1pct_energy=output_top,
                input_effective_support=input_support,
                output_effective_support=output_support,
            )
        )

    order = sorted(
        range(len(metrics)),
        key=lambda index: metrics[index].spectral_norm,
        reverse=True,
    )
    metrics = [metrics[index] for index in order]
    input_modes = [input_modes[index] for index in order]
    output_modes = [output_modes[index] for index in order]
    for index, row in enumerate(metrics, start=1):
        row.mode_index = index

    probe_generator = torch.Generator(device=coordinate.device)
    probe_generator.manual_seed(
        args.seed + 190_000 + sample_id * 100 + key_offset
    )
    sensitivity_diagonal = hutchinson_input_diagonal(
        function,
        coordinate,
        args.hutchinson_samples,
        probe_generator,
    )
    artifacts = {
        "coordinate": coordinate.detach(),
        "input_modes": torch.stack(input_modes) if input_modes else coordinate.new_empty((0, coordinate.numel())),
        "output_modes": torch.stack(output_modes) if output_modes else coordinate.new_empty((0, num_outputs)),
        "singular_values": torch.tensor(
            [row.spectral_norm for row in metrics],
            dtype=coordinate.dtype,
            device=coordinate.device,
        ),
        "sensitivity_diagonal": sensitivity_diagonal.detach(),
        "raw_output": raw_output,
    }
    return metrics, artifacts


def valid_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def summarize(values: Iterable[float]) -> dict[str, float]:
    array = valid_array(values)
    if array.size == 0:
        return {
            key: float("nan")
            for key in ("mean", "median", "p90", "p95", "min", "max")
        }
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty result file: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_numpy(value: Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def save_sample_artifact(
    output_dir: Path, sample_id: int, sample: Mapping[str, Tensor], raw_output: Tensor
) -> Path:
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    keys = (
        "Y",
        "H_hat",
        "P",
        "N0",
        "bits",
        "loss_mask",
        "H_err_var",
        "snr_db",
        "phi",
        "ut_distance_m",
        "ut_speed_mps",
        "indoor_state",
    )
    payload = {
        key: to_numpy(sample[key])
        for key in keys
        if isinstance(sample.get(key), Tensor)
    }
    payload["raw_llr"] = to_numpy(raw_output)
    path = artifact_dir / f"sample_{sample_id:04d}_inputs.npz"
    np.savez_compressed(path, **payload)
    return path


def sensitivity_tf_map(diagonal: Tensor, reference: Tensor) -> Tensor | None:
    if reference.ndim < 2 or not torch.is_complex(reference):
        return None
    if torch.is_complex(reference):
        pair = diagonal.reshape(*reference.shape, 2)
        energy = pair.sum(dim=-1)
    else:
        energy = diagonal.reshape(reference.shape)
    return torch.sqrt(torch.clamp(sum_to_tf(energy), min=0.0))


def save_jacobian_artifact(
    output_dir: Path,
    sample_id: int,
    key: str,
    sample: Mapping[str, Tensor],
    metrics: Sequence[JacobianMetrics],
    artifacts: Mapping[str, Tensor],
    mask_key: str,
) -> Path:
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reference = sample[key]
    raw_output = artifacts["raw_output"]
    output_mask = get_mask(sample, mask_key)
    input_modes = artifacts["input_modes"]
    output_modes = artifacts["output_modes"]
    full_output_modes = torch.stack(
        [
            unmask_output_vector(mode, raw_output, output_mask)
            for mode in output_modes
        ]
    ) if output_modes.shape[0] else raw_output.new_empty((0, *raw_output.shape))

    payload: dict[str, np.ndarray] = {
        "singular_values": to_numpy(artifacts["singular_values"]),
        "input_modes_real": to_numpy(input_modes),
        "output_modes_masked": to_numpy(output_modes),
        "output_modes_full": to_numpy(full_output_modes),
        "sensitivity_diagonal": to_numpy(artifacts["sensitivity_diagonal"]),
        "converged": np.asarray([row.converged for row in metrics], dtype=np.bool_),
        "iterations_used": np.asarray(
            [row.iterations_used for row in metrics], dtype=np.int64
        ),
    }
    if key in ("Y", "H_hat"):
        decoded_modes = torch.stack(
            [unflatten_real(mode, reference) for mode in input_modes]
        )
        payload["input_modes_complex"] = to_numpy(decoded_modes)
        payload["input_mode_energy_tf"] = to_numpy(
            torch.stack([reduce_energy_to_tf(mode) for mode in decoded_modes])
        )
        sensitivity = sensitivity_tf_map(
            artifacts["sensitivity_diagonal"], reference
        )
        if sensitivity is not None:
            payload["input_sensitivity_tf"] = to_numpy(sensitivity)
    payload["output_mode_energy_tf"] = to_numpy(
        torch.stack([reduce_energy_to_tf(mode) for mode in full_output_modes])
    ) if full_output_modes.shape[0] else np.empty((0,))

    path = artifact_dir / f"sample_{sample_id:04d}_{key}_jacobian.npz"
    np.savez_compressed(path, **payload)
    return path


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    left = left[valid]
    right = right[valid]
    if float(left.std()) <= EPS or float(right.std()) <= EPS:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def top_energy_mask(energy: np.ndarray, fraction: float) -> np.ndarray:
    flat = np.asarray(energy, dtype=np.float64).reshape(-1)
    count = max(1, int(math.ceil(fraction * flat.size)))
    indices = np.argpartition(flat, -count)[-count:]
    result = np.zeros(flat.shape, dtype=bool)
    result[indices] = True
    return result.reshape(energy.shape)


def hotspot_diagnostics(
    sample_id: int,
    key: str,
    sample: Mapping[str, Tensor],
    metrics: Sequence[JacobianMetrics],
    artifacts: Mapping[str, Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw_output = artifacts["raw_output"]
    reference = sample[key]
    input_modes = artifacts["input_modes"]
    output_modes = artifacts["output_modes"]
    output_mask = get_mask(sample, args.mask_key)
    singular_values = [row.spectral_norm for row in metrics]
    sigma_ratio = (
        singular_values[0] / max(singular_values[1], EPS)
        if len(singular_values) > 1
        else float("nan")
    )
    full_output = unmask_output_vector(output_modes[0], raw_output, output_mask)
    output_energy = to_numpy(reduce_energy_to_tf(full_output))
    output_top, output_support = energy_summary(
        torch.from_numpy(output_energy), args.hotspot_fraction
    )
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "key": key,
        "sigma_1": singular_values[0],
        "sigma_2": singular_values[1] if len(singular_values) > 1 else float("nan"),
        "sigma_1_over_sigma_2": sigma_ratio,
        "output_top_energy_fraction": output_top,
        "output_effective_support": output_support,
    }
    input_energy_tensor = input_mode_energy(input_modes[0], reference)
    if input_energy_tensor is None:
        return row
    input_energy = to_numpy(input_energy_tensor)
    top_mask = top_energy_mask(input_energy, args.hotspot_fraction)
    input_top, input_support = energy_summary(
        input_energy_tensor, args.hotspot_fraction
    )
    num_symbols, num_subcarriers = input_energy.shape
    edge = np.zeros_like(top_mask)
    width = min(args.edge_width, num_symbols // 2, num_subcarriers // 2)
    if width > 0:
        edge[:width, :] = True
        edge[-width:, :] = True
        edge[:, :width] = True
        edge[:, -width:] = True

    h_power = to_numpy(reduce_energy_to_tf(sample["H_hat"].detach()))
    n0 = float(sample["N0"].detach().mean().item())
    local_snr_db = 10.0 * np.log10(np.maximum(h_power / max(n0, EPS), EPS))
    low_snr = local_snr_db <= np.nanpercentile(local_snr_db, 10.0)

    pilot = to_numpy(reduce_energy_to_tf(sample["P"].detach())).astype(bool)
    llr_abs = to_numpy(raw_output.detach().abs().mean(dim=(0, 1)))
    data_mask = (
        to_numpy(reduce_energy_to_tf(sample[args.mask_key].detach())).astype(bool)
        if args.mask_key != "none"
        else np.ones_like(top_mask)
    )
    low_llr = np.zeros_like(top_mask)
    if data_mask.any():
        threshold = np.nanpercentile(llr_abs[data_mask], 10.0)
        low_llr = data_mask & (llr_abs <= threshold)

    error_map = np.zeros_like(top_mask)
    bits = sample.get("bits")
    if isinstance(bits, Tensor) and bits.shape == raw_output.shape:
        error_map = to_numpy(
            ((raw_output >= 0) != bits.bool()).any(dim=(0, 1))
        ).astype(bool)
        error_map &= data_mask

    peak_flat = int(np.argmax(input_energy))
    peak_t, peak_f = np.unravel_index(peak_flat, input_energy.shape)
    denominator = max(int(top_mask.sum()), 1)
    energy_log = np.log10(np.maximum(input_energy, EPS))
    row.update(
        {
            "input_top_energy_fraction": input_top,
            "input_effective_support": input_support,
            "peak_t": int(peak_t),
            "peak_f": int(peak_f),
            "peak_is_edge": bool(edge[peak_t, peak_f]),
            "peak_is_pilot": bool(pilot[peak_t, peak_f]),
            "peak_local_snr_db": float(local_snr_db[peak_t, peak_f]),
            "peak_abs_llr": float(llr_abs[peak_t, peak_f]),
            "peak_has_bit_error": bool(error_map[peak_t, peak_f]),
            "hotspot_edge_fraction": float((top_mask & edge).sum() / denominator),
            "hotspot_pilot_fraction": float((top_mask & pilot).sum() / denominator),
            "hotspot_low_snr_fraction": float((top_mask & low_snr).sum() / denominator),
            "hotspot_low_llr_fraction": float((top_mask & low_llr).sum() / denominator),
            "hotspot_error_fraction": float((top_mask & error_map).sum() / denominator),
            "corr_log_energy_neg_local_snr": safe_correlation(
                energy_log, -local_snr_db
            ),
            "corr_log_energy_neg_abs_llr": safe_correlation(
                energy_log[data_mask], -llr_abs[data_mask]
            ) if data_mask.any() else float("nan"),
        }
    )
    return row


def plot_jacobian_diagnostics(
    output_dir: Path,
    sample_id: int,
    key: str,
    sample: Mapping[str, Tensor],
    metrics: Sequence[JacobianMetrics],
    artifacts: Mapping[str, Tensor],
    args: argparse.Namespace,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    raw_output = artifacts["raw_output"]
    output_mask = get_mask(sample, args.mask_key)
    full_output_mode = unmask_output_vector(
        artifacts["output_modes"][0], raw_output, output_mask
    )
    output_energy = to_numpy(reduce_energy_to_tf(full_output_mode))
    input_energy_tensor = input_mode_energy(
        artifacts["input_modes"][0], sample[key]
    )
    sensitivity = sensitivity_tf_map(
        artifacts["sensitivity_diagonal"], sample[key]
    )
    h_power = to_numpy(reduce_energy_to_tf(sample["H_hat"].detach()))
    n0 = float(sample["N0"].detach().mean().item())
    local_snr_db = 10.0 * np.log10(np.maximum(h_power / max(n0, EPS), EPS))
    llr_abs = to_numpy(raw_output.detach().abs().mean(dim=(0, 1)))
    pilot = to_numpy(reduce_energy_to_tf(sample["P"].detach())).astype(bool)
    error_map = np.zeros_like(pilot)
    data_mask = (
        to_numpy(reduce_energy_to_tf(sample[args.mask_key].detach())).astype(bool)
        if args.mask_key != "none"
        else np.ones_like(pilot)
    )
    bits = sample.get("bits")
    if isinstance(bits, Tensor) and bits.shape == raw_output.shape:
        error_map = to_numpy(
            ((raw_output >= 0) != bits.bool()).any(dim=(0, 1))
        ).astype(bool)
        error_map &= data_mask

    figure, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)

    def show(axis, array, title, cmap="viridis"):
        image_handle = axis.imshow(array, aspect="auto", origin="lower", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("Subcarrier")
        axis.set_ylabel("OFDM symbol")
        figure.colorbar(image_handle, ax=axis, fraction=0.046, pad=0.04)

    if input_energy_tensor is None:
        axes[0, 0].text(0.5, 0.5, "Scalar input", ha="center", va="center")
        axes[0, 0].set_title("Input mode v1")
    else:
        show(
            axes[0, 0],
            np.log10(np.maximum(to_numpy(input_energy_tensor), EPS)),
            "Input v1 energy (log10)",
        )
    show(
        axes[0, 1],
        np.log10(np.maximum(output_energy, EPS)),
        "Output u1 energy (log10)",
    )
    if sensitivity is None:
        axes[0, 2].text(0.5, 0.5, "Scalar input", ha="center", va="center")
        axes[0, 2].set_title("diag(J^T J)")
    else:
        show(
            axes[0, 2],
            np.log10(np.maximum(to_numpy(sensitivity), EPS)),
            "Per-RE sensitivity (log10)",
        )
    show(axes[1, 0], local_snr_db, "Local |H_hat|^2/N0 (dB)", "magma")
    show(axes[1, 1], llr_abs, "Mean |raw LLR|", "magma")
    pilot_t, pilot_f = np.nonzero(pilot)
    error_t, error_f = np.nonzero(error_map)
    axes[1, 1].scatter(pilot_f, pilot_t, s=7, facecolors="none", edgecolors="cyan", label="pilot")
    axes[1, 1].scatter(error_f, error_t, s=18, marker="x", c="red", label="bit error")
    if pilot_t.size or error_t.size:
        axes[1, 1].legend(loc="upper right", fontsize=8)
    singular_values = [row.spectral_norm for row in metrics]
    axes[1, 2].bar(np.arange(1, len(singular_values) + 1), singular_values)
    axes[1, 2].set_title("Leading singular values")
    axes[1, 2].set_xlabel("Mode")
    axes[1, 2].set_ylabel("Singular value")
    figure.suptitle(f"sample={sample_id}  input={key}")
    path = plot_dir / f"sample_{sample_id:04d}_{key}_diagnostics.png"
    figure.savefig(path, dpi=args.plot_dpi)
    plt.close(figure)
    return path


def build_summary_rows(
    finite_rows: Sequence[Mapping[str, Any]],
    jacobian_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probe in sorted({str(row["probe"]) for row in finite_rows}):
        selected = [row for row in finite_rows if row["probe"] == probe]
        for metric in (
            "directional_lipschitz_mean",
            "directional_lipschitz_max",
            "directional_lipschitz_p95",
            "output_rms_change_mean",
            "raw_llr_rms_change_mean",
            "sign_flip_rate_mean",
        ):
            rows.append(
                {
                    "analysis": "finite_difference",
                    "group": probe,
                    "metric": metric,
                    **summarize(float(row[metric]) for row in selected),
                }
            )
    for key in sorted({str(row["key"]) for row in jacobian_rows}):
        selected = [
            row
            for row in jacobian_rows
            if row["key"] == key and int(row["mode_index"]) == 1
        ]
        rows.append(
            {
                "analysis": "jacobian",
                "group": key,
                "metric": "spectral_norm",
                **summarize(float(row["spectral_norm"]) for row in selected),
            }
        )
        mode_indices = sorted(
            {int(row["mode_index"]) for row in jacobian_rows if row["key"] == key}
        )
        for mode_index in mode_indices:
            mode_rows = [
                row
                for row in jacobian_rows
                if row["key"] == key and int(row["mode_index"]) == mode_index
            ]
            rows.append(
                {
                    "analysis": "jacobian_spectrum",
                    "group": key,
                    "metric": f"spectral_norm_mode_{mode_index}",
                    **summarize(
                        float(row["spectral_norm"]) for row in mode_rows
                    ),
                }
            )
    return rows


def build_hotspot_summary_rows(
    hotspot_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    metrics = (
        "sigma_1_over_sigma_2",
        "input_top_energy_fraction",
        "input_effective_support",
        "output_top_energy_fraction",
        "output_effective_support",
        "hotspot_edge_fraction",
        "hotspot_pilot_fraction",
        "hotspot_low_snr_fraction",
        "hotspot_low_llr_fraction",
        "hotspot_error_fraction",
        "corr_log_energy_neg_local_snr",
        "corr_log_energy_neg_abs_llr",
    )
    rows: list[dict[str, Any]] = []
    for key in sorted({str(row["key"]) for row in hotspot_rows}):
        selected = [row for row in hotspot_rows if row["key"] == key]
        for metric in metrics:
            values = [float(row[metric]) for row in selected if metric in row]
            if values:
                rows.append(
                    {
                        "analysis": "hotspot",
                        "group": key,
                        "metric": metric,
                        **summarize(values),
                    }
                )
    return rows


def print_summary(
    title: str, rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> None:
    print(f"\n===== {title} =====")
    for metric in metrics:
        stats = summarize(float(row[metric]) for row in rows)
        print(
            f"{metric:<32} mean={stats['mean']:.6e}  "
            f"median={stats['median']:.6e}  "
            f"p95={stats['p95']:.6e}  max={stats['max']:.6e}"
        )


def resolve_profiles(
    checkpoint: Mapping[str, Any],
    ofdm_config: SionnaOFDMConfig,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_profile = checkpoint.get("train_channel_profile")
    if train_profile is None:
        train_profile = legacy_channel_profile(ofdm_config)
    else:
        train_profile = load_channel_profile(train_profile)

    if args.eval_channel_profile:
        eval_profile = load_channel_profile(
            args.eval_channel_profile, component_id=args.eval_component_id
        )
    elif args.eval_component_id:
        raise ValueError("--eval-component-id requires --eval-channel-profile.")
    else:
        eval_profile = train_profile
    return train_profile, eval_profile


def generate_batch(
    ofdm_config: SionnaOFDMConfig,
    eval_profile: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Any], dict[str, Any]]:
    if args.eval_mode == "ber":
        generator = SionnaOFDMBatchGenerator(
            ofdm_config,
            snr_db_min=args.snr_db,
            snr_db_max=args.snr_db,
            phase_mode=args.phase_mode,
            seed=args.seed,
            device=device,
            channel_profile=eval_profile,
        )
        operating_point = {
            "eval_mode": "ber",
            "snr_db": args.snr_db,
            "ebno_db": None,
            "coderate": None,
        }
    else:
        ldpc_config = SionnaLDPC5GConfig(coderate=args.coderate)
        generator = Sionna5GLDPCBatchGenerator(
            ofdm_config,
            ldpc_config=ldpc_config,
            ebno_db_min=args.ebno_db,
            ebno_db_max=args.ebno_db,
            phase_mode=args.phase_mode,
            seed=args.seed,
            device=device,
            channel_profile=eval_profile,
        )
        operating_point = {
            "eval_mode": "ldpc",
            "snr_db": None,
            "ebno_db": args.ebno_db,
            "coderate": generator.coderate,
        }
    loaded = generator.generate_batch(args.batch_size)
    batch = {
        key: value.to(device)
        for key, value in loaded.items()
        if isinstance(value, Tensor)
    }
    batch_metadata = {
        key: value
        for key, value in loaded.items()
        if not isinstance(value, Tensor)
        and isinstance(value, (str, int, float, bool, type(None)))
    }
    return batch, operating_point, batch_metadata


def default_output_dir(args: argparse.Namespace, checkpoint_path: Path) -> Path:
    point = (
        f"snr{args.snr_db:g}" if args.eval_mode == "ber" else f"ebno{args.ebno_db:g}"
    )
    return (
        PROJECT_ROOT
        / "runs"
        / "jacobian_analysis"
        / f"{checkpoint_path.parent.name}_{checkpoint_path.stem}_{args.eval_mode}_{point}_seed{args.seed}"
    )


def prepare_output_dir(
    args: argparse.Namespace, checkpoint_path: Path
) -> Path:
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir(args, checkpoint_path)
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use a unique --output-dir or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = prepare_output_dir(args, checkpoint_path)

    model, checkpoint = load_receiver_checkpoint(
        checkpoint_path,
        device,
        bits_per_symbol=2,
        required_data_backend="sionna",
    )
    model.to(device).eval()
    model.requires_grad_(False)

    ofdm_config = SionnaOFDMConfig(**checkpoint["sionna_config"])
    if args.eval_dmrs_freq_spacing is not None:
        if args.eval_dmrs_freq_spacing <= 0:
            raise ValueError("--eval-dmrs-freq-spacing must be positive.")
        ofdm_config = replace(
            ofdm_config, dmrs_freq_spacing=args.eval_dmrs_freq_spacing
        )
    train_profile, eval_profile = resolve_profiles(checkpoint, ofdm_config, args)
    batch, operating_point, batch_metadata = generate_batch(
        ofdm_config, eval_profile, args, device
    )
    if len(eval_profile["components"]) > 1:
        print(
            "Warning: channel components are sampled once per generated batch. "
            f"This run analyzes only component {batch_metadata.get('channel_profile_id', 'unknown')!r} "
            "from the multi-component evaluation profile."
        )
    validate_batch(batch, args.mask_key)

    batch_size = infer_batch_size(batch)
    sample_count = min(batch_size, args.max_samples)
    jacobian_count = min(sample_count, args.jacobian_max_samples)
    if sample_count <= 0:
        raise ValueError("No samples selected for analysis.")

    fixed_component = profile_component(eval_profile)
    provenance = {
        **vars(args),
        **operating_point,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "checkpoint_model_name": checkpoint.get("model_name", "unknown"),
        "checkpoint_train_seed": checkpoint.get("args", {}).get("seed"),
        "git_revision": git_revision(),
        "torch_version": torch.__version__,
        "device_resolved": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_profile": train_profile.get("name", "unknown"),
        "eval_profile": eval_profile.get("name", "unknown"),
        "eval_backend": profile_backend_label(eval_profile),
        "eval_component": fixed_component,
        "eval_profile_components": [
            {"id": component["id"], "backend": component["backend"]}
            for component in eval_profile["components"]
        ],
        "channel_component_sampling_granularity": "one_component_per_generated_batch",
        "batch_channel_metadata": batch_metadata,
        "batch_size_loaded": batch_size,
        "samples_analyzed": sample_count,
        "jacobian_samples_analyzed": jacobian_count,
        "batch_tensors": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in batch.items()
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    failures: list[str] = []
    finite_rows: list[dict[str, Any]] = []
    for sample_id in range(sample_count):
        sample = select_sample(batch, sample_id)
        for probe in args.probes:
            try:
                finite_rows.append(
                    asdict(
                        finite_difference_one_sample(
                            model, sample, sample_id, probe, args
                        )
                    )
                )
            except Exception as exc:  # preserve all failures, then fail the run
                failures.append(
                    f"finite_difference sample={sample_id} probe={probe}: {exc!r}"
                )

    jacobian_rows: list[dict[str, Any]] = []
    hotspot_rows: list[dict[str, Any]] = []
    for sample_id in range(jacobian_count):
        sample = select_sample(batch, sample_id)
        with torch.inference_mode():
            raw_output = receiver_forward(model, sample).detach()
        if args.save_directions:
            save_sample_artifact(output_dir, sample_id, sample, raw_output)
        for key in args.jacobian_keys:
            try:
                mode_metrics, artifacts = jacobian_singular_modes(
                    model, sample, sample_id, key, args
                )
                jacobian_rows.extend(asdict(row) for row in mode_metrics)
                if not mode_metrics:
                    raise RuntimeError("No non-zero singular mode was recovered.")
                hotspot_rows.append(
                    hotspot_diagnostics(
                        sample_id, key, sample, mode_metrics, artifacts, args
                    )
                )
                if args.save_directions:
                    save_jacobian_artifact(
                        output_dir,
                        sample_id,
                        key,
                        sample,
                        mode_metrics,
                        artifacts,
                        args.mask_key,
                    )
                if args.save_plots:
                    plot_jacobian_diagnostics(
                        output_dir,
                        sample_id,
                        key,
                        sample,
                        mode_metrics,
                        artifacts,
                        args,
                    )
            except Exception as exc:
                failures.append(
                    f"jacobian sample={sample_id} key={key}: {exc!r}"
                )

    expected_finite = sample_count * len(args.probes)
    expected_jacobian_groups = jacobian_count * len(args.jacobian_keys)
    observed_jacobian_groups = len(
        {(int(row["sample_id"]), str(row["key"])) for row in jacobian_rows}
    )
    status = {
        "finite_rows": len(finite_rows),
        "expected_finite_rows": expected_finite,
        "jacobian_mode_rows": len(jacobian_rows),
        "jacobian_groups": observed_jacobian_groups,
        "expected_jacobian_groups": expected_jacobian_groups,
        "hotspot_rows": len(hotspot_rows),
        "failures": failures,
    }
    (output_dir / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures and not args.allow_partial:
        raise RuntimeError(
            f"Analysis had {len(failures)} failure(s). See {output_dir / 'status.json'}."
        )
    if len(finite_rows) != expected_finite and not args.allow_partial:
        raise RuntimeError("Finite-difference result grid is incomplete.")
    if observed_jacobian_groups != expected_jacobian_groups and not args.allow_partial:
        raise RuntimeError("Jacobian sample/key grid is incomplete.")

    finite_path = output_dir / "finite_difference_per_sample.csv"
    jacobian_path = output_dir / "jacobian_per_sample.csv"
    summary_path = output_dir / "summary.csv"
    hotspot_path = output_dir / "hotspot_per_sample.csv"
    hotspot_summary_path = output_dir / "hotspot_summary.csv"
    if finite_rows:
        write_csv(finite_path, finite_rows)
    if jacobian_rows:
        write_csv(jacobian_path, jacobian_rows)
    if hotspot_rows:
        write_csv(hotspot_path, hotspot_rows)
        write_csv(
            hotspot_summary_path,
            build_hotspot_summary_rows(hotspot_rows),
        )
    summary_rows = build_summary_rows(finite_rows, jacobian_rows)
    write_csv(summary_path, summary_rows)

    for probe in args.probes:
        selected = [row for row in finite_rows if row["probe"] == probe]
        if selected:
            print_summary(
                f"Finite-difference probe: {probe}",
                selected,
                (
                    "directional_lipschitz_max",
                    "directional_lipschitz_mean",
                    "output_rms_change_mean",
                    "raw_llr_rms_change_mean",
                    "sign_flip_rate_mean",
                ),
            )
    for key in args.jacobian_keys:
        selected = [
            row
            for row in jacobian_rows
            if row["key"] == key and int(row["mode_index"]) == 1
        ]
        if selected:
            print_summary(
                f"Jacobian spectral norm: {key}", selected, ("spectral_norm",)
            )

    print("\nSaved:")
    if finite_rows:
        print(f"  {finite_path}")
    if jacobian_rows:
        print(f"  {jacobian_path}")
    if hotspot_rows:
        print(f"  {hotspot_path}")
        print(f"  {hotspot_summary_path}")
    if args.save_directions and (output_dir / "artifacts").exists():
        print(f"  {output_dir / 'artifacts'}")
    if args.save_plots and (output_dir / "plots").exists():
        print(f"  {output_dir / 'plots'}")
    print(f"  {summary_path}")
    print(f"  {output_dir / 'config.json'}")
    print(f"  {output_dir / 'status.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local input sensitivity and Jacobian analysis for AI receivers."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=777000)
    parser.add_argument("--phase-mode", default="uniform", choices=("fixed", "narrow", "uniform"))
    parser.add_argument("--eval-mode", choices=("ber", "ldpc"), default="ber")
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--ebno-db", type=float, default=20.0)
    parser.add_argument("--coderate", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--eval-channel-profile")
    parser.add_argument("--eval-component-id")
    parser.add_argument("--eval-dmrs-freq-spacing", type=int)

    parser.add_argument(
        "--probes",
        nargs="+",
        choices=("Y", "H_hat", "N0", "common_phase"),
        default=("Y", "H_hat", "N0", "common_phase"),
    )
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument(
        "--output-transform", choices=("none", "tanh", "sigmoid"), default="none"
    )
    parser.add_argument("--llr-tau", type=float, default=5.0)
    parser.add_argument("--mask-key", default="loss_mask")
    parser.add_argument("--n0-floor", type=float, default=1e-12)
    parser.add_argument("--real-imag-channel-dim", type=int, default=1)

    parser.add_argument(
        "--jacobian-keys",
        nargs="*",
        choices=("Y", "H_hat", "N0"),
        default=("Y", "H_hat", "N0"),
    )
    parser.add_argument("--jacobian-iters", type=int, default=30)
    parser.add_argument("--jacobian-restarts", type=int, default=2)
    parser.add_argument("--jacobian-tol", type=float, default=1e-4)
    parser.add_argument("--jacobian-max-samples", type=int, default=4)
    parser.add_argument("--jacobian-top-k", type=int, default=5)
    parser.add_argument("--hutchinson-samples", type=int, default=32)
    parser.add_argument("--hotspot-fraction", type=float, default=0.01)
    parser.add_argument("--edge-width", type=int, default=1)
    parser.add_argument("--plot-dpi", type=int, default=160)
    parser.add_argument(
        "--no-save-directions",
        action="store_false",
        dest="save_directions",
        help="Do not save raw inputs and singular-mode NPZ artifacts.",
    )
    parser.add_argument(
        "--no-save-plots",
        action="store_false",
        dest="save_plots",
        help="Do not create per-sample diagnostic PNG files.",
    )

    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.epsilon <= 0:
        parser.error("--epsilon must be positive.")
    if args.trials <= 0:
        parser.error("--trials must be positive.")
    if args.batch_size <= 0 or args.max_samples <= 0:
        parser.error("--batch-size and --max-samples must be positive.")
    if args.max_samples > args.batch_size:
        parser.error("--max-samples cannot exceed --batch-size.")
    if not 0.0 < args.coderate < 1.0:
        parser.error("--coderate must be in (0, 1).")
    if args.jacobian_iters <= 0 or args.jacobian_restarts <= 0:
        parser.error("Jacobian iterations and restarts must be positive.")
    if args.jacobian_tol <= 0:
        parser.error("--jacobian-tol must be positive.")
    if args.jacobian_max_samples < 0:
        parser.error("--jacobian-max-samples must be non-negative.")
    if args.jacobian_max_samples > args.max_samples:
        parser.error("--jacobian-max-samples cannot exceed --max-samples.")
    if args.jacobian_top_k <= 0:
        parser.error("--jacobian-top-k must be positive.")
    if args.hutchinson_samples <= 0:
        parser.error("--hutchinson-samples must be positive.")
    if not 0.0 < args.hotspot_fraction <= 1.0:
        parser.error("--hotspot-fraction must be in (0, 1].")
    if args.edge_width < 0:
        parser.error("--edge-width must be non-negative.")
    if args.plot_dpi <= 0:
        parser.error("--plot-dpi must be positive.")
    if args.output_transform == "tanh" and args.llr_tau <= 0:
        parser.error("--llr-tau must be positive for tanh output transform.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    run(args)


if __name__ == "__main__":
    main()
