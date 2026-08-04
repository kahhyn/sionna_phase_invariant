import torch
import torch.nn.functional as F


def masked_bce_with_logits(logits, bits, loss_mask):
    """
    SISO tensors use ``(B, bits_per_symbol, T, F)`` and MIMO tensors may
    prepend an explicit layer axis, ``(B, L, bits_per_symbol, T, F)``.
    ``loss_mask`` must be broadcastable to ``logits`` and use one for valid
    data bits and zero for ignored REs.
    """
    bce = F.binary_cross_entropy_with_logits(logits, bits, reduction="none")
    valid = loss_mask.expand_as(logits)
    denom = valid.sum() + 1e-12
    return (bce * valid).sum() / denom


@torch.no_grad()
def masked_ber(logits, bits, loss_mask):
    """
    Decision rule:
    logits > 0 -> bit 1
    logits < 0 -> bit 0

    ``loss_mask`` must be broadcastable to ``logits``.
    """
    pred = (logits > 0).to(bits.dtype)
    err = (pred != bits).to(bits.dtype)
    valid = loss_mask.expand_as(logits)
    denom = valid.sum() + 1e-12
    return (err * valid).sum() / denom


@torch.no_grad()
def masked_error_count(logits, bits, loss_mask):
    """Return integer bit errors and valid-bit count for exact aggregation."""
    pred = logits > 0
    valid = loss_mask.bool().expand_as(bits)
    errors = ((pred != bits.bool()) & valid).sum()
    return errors, valid.sum()


@torch.no_grad()
def masked_bce_sum(logits, bits, loss_mask):
    """Return summed BCE and valid-bit count for exact aggregation."""
    bce = F.binary_cross_entropy_with_logits(logits, bits, reduction="none")
    valid = loss_mask.expand_as(bits)
    return (bce * valid).sum(), valid.sum()


@torch.no_grad()
def max_mean_abs_diff(a, b):
    diff = torch.abs(a - b)
    return diff.max().item(), diff.mean().item()
