"""Shared helpers for deterministic sample-count batching."""


def batch_sizes(num_samples, batch_size):
    """Yield batch sizes that exactly cover ``num_samples`` examples."""
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, num_samples, batch_size):
        yield min(batch_size, num_samples - start)
