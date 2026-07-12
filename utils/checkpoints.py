"""Checkpoint helpers for receiver reconstruction."""

import torch

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

    model = build_model_from_args(
        checkpoint["args"],
        bits_per_symbol=bits_per_symbol,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
