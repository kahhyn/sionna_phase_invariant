"""Checkpoint helpers for receiver reconstruction."""

import torch

from data import (
    PROFILE_SCHEMA_VERSION,
    SionnaSUMIMOConfig,
    validate_channel_profile,
)
from models import SUMIMOPhaseInvariantReceiver
from models.factory import build_model_from_args


def load_receiver_checkpoint(
    checkpoint_path,
    device,
    bits_per_symbol=2,
    required_data_backend=None,
):
    """Load a receiver checkpoint and reconstruct the matching model."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if required_data_backend is not None:
        data_backend = checkpoint.get("data_backend")
        if data_backend != required_data_backend:
            raise ValueError(
                f"Checkpoint data_backend={data_backend!r}, "
                f"expected {required_data_backend!r}."
            )

    profile_version = checkpoint.get("channel_profile_schema_version")
    if profile_version is not None:
        if profile_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoint channel profile schema={profile_version!r}, "
                f"expected {PROFILE_SCHEMA_VERSION}."
            )
        validate_channel_profile(checkpoint["train_channel_profile"])
        validate_channel_profile(checkpoint["val_channel_profile"])
    saved_bits = checkpoint.get("sionna_config", {}).get("bits_per_symbol")
    if saved_bits is not None and int(saved_bits) != int(bits_per_symbol):
        raise ValueError(
            f"Checkpoint bits_per_symbol={saved_bits}, expected {bits_per_symbol}."
        )

    model = build_model_from_args(
        checkpoint["args"],
        bits_per_symbol=bits_per_symbol,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def load_su_mimo_checkpoint(checkpoint_path, device):
    """Load and reconstruct a checkpoint from ``training.train_su_mimo``."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if checkpoint.get("data_backend") != "sionna_su_mimo":
        raise ValueError(
            "Checkpoint data_backend="
            f"{checkpoint.get('data_backend')!r}, expected 'sionna_su_mimo'."
        )
    if checkpoint.get("model_name") != "su_mimo_phase_invariant":
        raise ValueError(
            f"Unsupported SU-MIMO model: {checkpoint.get('model_name')!r}."
        )

    config = SionnaSUMIMOConfig(**checkpoint["sionna_su_mimo_config"])
    model = SUMIMOPhaseInvariantReceiver(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, config, checkpoint
