#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 PyTorch 模型参数统计脚本

功能：
1. 统计 torch.nn.Module 的参数量、可训练参数量、buffer 数量、参数内存大小。
2. 统计 checkpoint/state_dict 中的参数量和 dtype 内存大小。
3. 支持按层打印参数明细。
4. 支持从 Python 代码中 import 使用，也支持命令行使用。

典型用法：
    # 方式1：只统计 checkpoint，不需要模型定义
    python count_model_params.py --checkpoint model.pth

    # 方式2：从 Python 模块中实例化模型后统计
    python count_model_params.py \
        --model_module models.my_model \
        --model_class MyModel \
        --model_kwargs '{"num_classes": 10}'

    # 方式3：模型 + checkpoint，一般用于确认加载后模型参数
    python count_model_params.py \
        --model_module models.my_model \
        --model_class MyModel \
        --model_kwargs '{"num_classes": 10}' \
        --checkpoint model.pth \
        --checkpoint_key model_state_dict

在 Python 中使用：
    from count_model_params import summarize_model
    summary = summarize_model(model, verbose=True)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    raise ImportError(
        "This script requires PyTorch. Please install torch first. "
        "Example: pip install torch"
    ) from exc


# -----------------------------
# Basic utilities
# -----------------------------


def human_count(num: int) -> str:
    """Convert an integer count to a readable string."""
    abs_num = abs(num)
    if abs_num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.3f}B"
    if abs_num >= 1_000_000:
        return f"{num / 1_000_000:.3f}M"
    if abs_num >= 1_000:
        return f"{num / 1_000:.3f}K"
    return str(num)


def human_bytes(num_bytes: int) -> str:
    """Convert bytes to KB/MB/GB."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0:
            return f"{size:.3f} {unit}"
        size /= 1024.0
    return f"{size:.3f} PB"


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """Return tensor memory bytes."""
    return int(tensor.numel() * tensor.element_size())


def safe_shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(x) for x in tensor.shape)


def unwrap_state_dict(obj: Any, checkpoint_key: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """
    Extract a state_dict-like mapping from a loaded checkpoint.

    Supports common formats:
    - pure state_dict: {name: tensor}
    - {'state_dict': ...}
    - {'model_state_dict': ...}
    - {'model': ...}
    - custom key specified by checkpoint_key
    """
    if checkpoint_key is not None:
        if not isinstance(obj, dict) or checkpoint_key not in obj:
            raise KeyError(f"checkpoint_key='{checkpoint_key}' not found in checkpoint.")
        obj = obj[checkpoint_key]

    if isinstance(obj, dict):
        # If it already looks like a state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj

        # Try common keys
        for key in ["state_dict", "model_state_dict", "model", "net", "network", "module"]:
            if key in obj and isinstance(obj[key], dict):
                maybe = obj[key]
                if all(isinstance(v, torch.Tensor) for v in maybe.values()):
                    return maybe

    raise TypeError(
        "Cannot find a valid state_dict in checkpoint. "
        "Please pass --checkpoint_key explicitly if your checkpoint uses a custom key."
    )


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove DataParallel/DistributedDataParallel prefix 'module.' if present."""
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class TensorStat:
    name: str
    shape: Tuple[int, ...]
    numel: int
    dtype: str
    nbytes: int
    trainable: Optional[bool] = None


@dataclass
class ParamSummary:
    total_params: int
    trainable_params: int
    non_trainable_params: int
    buffer_params: int
    total_param_bytes: int
    trainable_param_bytes: int
    non_trainable_param_bytes: int
    buffer_bytes: int
    total_bytes_with_buffers: int
    dtype_bytes: Dict[str, int]
    dtype_params: Dict[str, int]
    layer_stats: List[TensorStat]
    checkpoint_file_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["total_params_readable"] = human_count(self.total_params)
        data["trainable_params_readable"] = human_count(self.trainable_params)
        data["non_trainable_params_readable"] = human_count(self.non_trainable_params)
        data["buffer_params_readable"] = human_count(self.buffer_params)
        data["total_param_size_readable"] = human_bytes(self.total_param_bytes)
        data["trainable_param_size_readable"] = human_bytes(self.trainable_param_bytes)
        data["non_trainable_param_size_readable"] = human_bytes(self.non_trainable_param_bytes)
        data["buffer_size_readable"] = human_bytes(self.buffer_bytes)
        data["total_size_with_buffers_readable"] = human_bytes(self.total_bytes_with_buffers)
        if self.checkpoint_file_bytes is not None:
            data["checkpoint_file_size_readable"] = human_bytes(self.checkpoint_file_bytes)
        return data


# -----------------------------
# Model statistics
# -----------------------------


def summarize_model(model: nn.Module, verbose: bool = False) -> ParamSummary:
    """
    Summarize a PyTorch nn.Module.

    Args:
        model: PyTorch model.
        verbose: Whether to print per-layer parameter details.

    Returns:
        ParamSummary object.
    """
    layer_stats: List[TensorStat] = []
    dtype_bytes: Dict[str, int] = OrderedDict()
    dtype_params: Dict[str, int] = OrderedDict()

    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    total_param_bytes = 0
    trainable_param_bytes = 0
    non_trainable_param_bytes = 0

    for name, param in model.named_parameters():
        n = int(param.numel())
        b = tensor_nbytes(param)
        dtype = str(param.dtype).replace("torch.", "")
        trainable = bool(param.requires_grad)

        total_params += n
        total_param_bytes += b
        if trainable:
            trainable_params += n
            trainable_param_bytes += b
        else:
            non_trainable_params += n
            non_trainable_param_bytes += b

        dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + b
        dtype_params[dtype] = dtype_params.get(dtype, 0) + n

        layer_stats.append(
            TensorStat(
                name=name,
                shape=safe_shape(param),
                numel=n,
                dtype=dtype,
                nbytes=b,
                trainable=trainable,
            )
        )

    buffer_params = 0
    buffer_bytes = 0
    for name, buf in model.named_buffers():
        n = int(buf.numel())
        b = tensor_nbytes(buf)
        dtype = str(buf.dtype).replace("torch.", "")

        buffer_params += n
        buffer_bytes += b
        dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + b
        dtype_params[dtype] = dtype_params.get(dtype, 0) + n

        layer_stats.append(
            TensorStat(
                name=f"[buffer] {name}",
                shape=safe_shape(buf),
                numel=n,
                dtype=dtype,
                nbytes=b,
                trainable=None,
            )
        )

    summary = ParamSummary(
        total_params=total_params,
        trainable_params=trainable_params,
        non_trainable_params=non_trainable_params,
        buffer_params=buffer_params,
        total_param_bytes=total_param_bytes,
        trainable_param_bytes=trainable_param_bytes,
        non_trainable_param_bytes=non_trainable_param_bytes,
        buffer_bytes=buffer_bytes,
        total_bytes_with_buffers=total_param_bytes + buffer_bytes,
        dtype_bytes=dict(dtype_bytes),
        dtype_params=dict(dtype_params),
        layer_stats=layer_stats,
    )

    if verbose:
        print_summary(summary, show_layers=True)

    return summary


def summarize_state_dict(
    state_dict: Dict[str, torch.Tensor],
    checkpoint_file_bytes: Optional[int] = None,
    verbose: bool = False,
) -> ParamSummary:
    """
    Summarize a checkpoint state_dict without requiring model definition.

    Note:
        Checkpoint state_dict does not know which parameters are trainable.
        Therefore trainable_params is set to 0 and all tensors are counted as non-trainable-like tensors.
    """
    layer_stats: List[TensorStat] = []
    dtype_bytes: Dict[str, int] = OrderedDict()
    dtype_params: Dict[str, int] = OrderedDict()

    total_params = 0
    total_param_bytes = 0

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        n = int(tensor.numel())
        b = tensor_nbytes(tensor)
        dtype = str(tensor.dtype).replace("torch.", "")

        total_params += n
        total_param_bytes += b
        dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + b
        dtype_params[dtype] = dtype_params.get(dtype, 0) + n

        layer_stats.append(
            TensorStat(
                name=name,
                shape=safe_shape(tensor),
                numel=n,
                dtype=dtype,
                nbytes=b,
                trainable=None,
            )
        )

    summary = ParamSummary(
        total_params=total_params,
        trainable_params=0,
        non_trainable_params=total_params,
        buffer_params=0,
        total_param_bytes=total_param_bytes,
        trainable_param_bytes=0,
        non_trainable_param_bytes=total_param_bytes,
        buffer_bytes=0,
        total_bytes_with_buffers=total_param_bytes,
        dtype_bytes=dict(dtype_bytes),
        dtype_params=dict(dtype_params),
        layer_stats=layer_stats,
        checkpoint_file_bytes=checkpoint_file_bytes,
    )

    if verbose:
        print_summary(summary, show_layers=True)

    return summary


# -----------------------------
# Printing and export
# -----------------------------


def print_summary(summary: ParamSummary, show_layers: bool = False, topk: Optional[int] = None) -> None:
    """Pretty print summary."""
    print("\n========== Model Parameter Summary ==========")
    print(f"Total parameters          : {summary.total_params:,} ({human_count(summary.total_params)})")
    print(f"Trainable parameters      : {summary.trainable_params:,} ({human_count(summary.trainable_params)})")
    print(f"Non-trainable parameters  : {summary.non_trainable_params:,} ({human_count(summary.non_trainable_params)})")
    print(f"Buffer elements           : {summary.buffer_params:,} ({human_count(summary.buffer_params)})")
    print("---------------------------------------------")
    print(f"Parameter memory          : {human_bytes(summary.total_param_bytes)}")
    print(f"Trainable parameter memory: {human_bytes(summary.trainable_param_bytes)}")
    print(f"Non-trainable param memory: {human_bytes(summary.non_trainable_param_bytes)}")
    print(f"Buffer memory             : {human_bytes(summary.buffer_bytes)}")
    print(f"Total memory with buffers : {human_bytes(summary.total_bytes_with_buffers)}")

    if summary.checkpoint_file_bytes is not None:
        print(f"Checkpoint file size      : {human_bytes(summary.checkpoint_file_bytes)}")

    if summary.dtype_params:
        print("---------------------------------------------")
        print("By dtype:")
        for dtype, n in summary.dtype_params.items():
            b = summary.dtype_bytes.get(dtype, 0)
            print(f"  {dtype:<12} params/elements: {n:,} | memory: {human_bytes(b)}")

    if show_layers:
        stats = summary.layer_stats
        if topk is not None and topk > 0:
            stats = sorted(stats, key=lambda x: x.numel, reverse=True)[:topk]
            print(f"---------------------------------------------")
            print(f"Top-{topk} largest tensors:")
        else:
            print("---------------------------------------------")
            print("Per-layer/tensor details:")

        header = f"{'Name':<60} {'Shape':<24} {'Params':>14} {'Memory':>14} {'DType':>10} {'Trainable':>10}"
        print(header)
        print("-" * len(header))
        for s in stats:
            trainable_str = "-" if s.trainable is None else str(s.trainable)
            shape_str = str(s.shape)
            if len(s.name) > 58:
                name = s.name[:55] + "..."
            else:
                name = s.name
            print(
                f"{name:<60} {shape_str:<24} {s.numel:>14,} "
                f"{human_bytes(s.nbytes):>14} {s.dtype:>10} {trainable_str:>10}"
            )

    print("=============================================\n")


def export_json(summary: ParamSummary, output_path: str) -> None:
    """Export summary to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"Saved JSON summary to: {output_path}")


# -----------------------------
# Dynamic model loading
# -----------------------------


def build_model_from_module(
    model_module: str,
    model_class: str,
    model_kwargs: Optional[Union[str, Dict[str, Any]]] = None,
) -> nn.Module:
    """
    Dynamically import and build a model.

    Args:
        model_module: Python module path, e.g. 'models.resnet'.
        model_class: Class/function name, e.g. 'ResNet18'.
        model_kwargs: JSON string or dict for model constructor.
    """
    if model_kwargs is None:
        kwargs = {}
    elif isinstance(model_kwargs, str):
        kwargs = json.loads(model_kwargs) if model_kwargs.strip() else {}
    elif isinstance(model_kwargs, dict):
        kwargs = model_kwargs
    else:
        raise TypeError("model_kwargs must be None, JSON string, or dict.")

    module = importlib.import_module(model_module)
    cls_or_fn = getattr(module, model_class)
    model = cls_or_fn(**kwargs)

    if not isinstance(model, nn.Module):
        raise TypeError(f"{model_module}.{model_class} did not return a torch.nn.Module.")
    return model


def load_checkpoint_to_model(
    model: nn.Module,
    checkpoint_path: str,
    checkpoint_key: Optional[str] = None,
    strict: bool = False,
    map_location: str = "cpu",
) -> nn.Module:
    """Load checkpoint into model."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = unwrap_state_dict(checkpoint, checkpoint_key=checkpoint_key)
    state_dict = strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        print(f"[Warning] Missing keys: {len(missing)}")
        for k in missing[:20]:
            print(f"  missing: {k}")
        if len(missing) > 20:
            print("  ...")

    if unexpected:
        print(f"[Warning] Unexpected keys: {len(unexpected)}")
        for k in unexpected[:20]:
            print(f"  unexpected: {k}")
        if len(unexpected) > 20:
            print("  ...")

    return model


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Universal PyTorch model/checkpoint parameter counter."
    )

    parser.add_argument(
        "--model_module",
        type=str,
        default=None,
        help="Python module path containing the model, e.g. models.my_model.",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default=None,
        help="Model class/function name in model_module, e.g. MyModel.",
    )
    parser.add_argument(
        "--model_kwargs",
        type=str,
        default="{}",
        help="JSON string for model constructor arguments.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pth/.pt checkpoint or state_dict.",
    )
    parser.add_argument(
        "--checkpoint_key",
        type=str,
        default=None,
        help="Key to state_dict in checkpoint, e.g. model_state_dict.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True when loading checkpoint into model.",
    )
    parser.add_argument(
        "--show_layers",
        action="store_true",
        help="Print per-layer/tensor details.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Show only top-k largest tensors when --show_layers is enabled.",
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="Optional path to save summary as JSON.",
    )
    parser.add_argument(
        "--map_location",
        type=str,
        default="cpu",
        help="torch.load map_location. Default: cpu.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    has_model = args.model_module is not None or args.model_class is not None
    if has_model and not (args.model_module and args.model_class):
        raise ValueError("Please provide both --model_module and --model_class.")

    if has_model:
        model = build_model_from_module(
            model_module=args.model_module,
            model_class=args.model_class,
            model_kwargs=args.model_kwargs,
        )

        if args.checkpoint:
            model = load_checkpoint_to_model(
                model=model,
                checkpoint_path=args.checkpoint,
                checkpoint_key=args.checkpoint_key,
                strict=args.strict,
                map_location=args.map_location,
            )

        summary = summarize_model(model, verbose=False)

    elif args.checkpoint:
        checkpoint_file_bytes = os.path.getsize(args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=args.map_location)
        state_dict = unwrap_state_dict(checkpoint, checkpoint_key=args.checkpoint_key)
        state_dict = strip_module_prefix(state_dict)
        summary = summarize_state_dict(
            state_dict=state_dict,
            checkpoint_file_bytes=checkpoint_file_bytes,
            verbose=False,
        )
    else:
        raise ValueError(
            "Nothing to summarize. Please provide either:\n"
            "  1) --checkpoint model.pth\n"
            "  2) --model_module xxx --model_class XxxModel\n"
        )

    print_summary(summary, show_layers=args.show_layers, topk=args.topk)

    if args.json_out:
        export_json(summary, args.json_out)


if __name__ == "__main__":
    main()