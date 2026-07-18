"""Cache slicing and position correction.

Pure functions. No I/O, no model loading, no global state. Everything here
is testable against a synthetic cache without running a model.

Cache tensor layout, per layer: [batch, n_kv_heads, seq, head_dim].
The sequence axis is -2.
"""

import torch
from transformers.cache_utils import DynamicCache

SEQ_AXIS = -2


def slice_cache(cache: DynamicCache, len_E: int, len_X: int) -> DynamicCache:
    """Return the question-aligned slice of a cache built from prefilling E ⊕ X.

    Implements C*(X) = C[|E| : |E|+|X|](E ⊕ X). Applied to both keys and
    values, at every layer.

    The slice is cloned, not viewed. A view of a tensor keeps the whole
    parent tensor alive in memory, so a viewed slice would report a shorter
    sequence length while still holding every byte of E. Cloning makes the
    reported cache size and the actual footprint agree.

    Args:
        cache: cache produced by prefilling E ⊕ X.
        len_E: token length of E.
        len_X: token length of X.

    Returns:
        A new DynamicCache holding only X's positions. The input is not
        modified.
    """
    if len_E < 0:
        raise ValueError(f"len_E must be non-negative, got {len_E}")
    if len_X <= 0:
        raise ValueError(f"len_X must be positive, got {len_X}")
    if len(cache.layers) == 0:
        raise ValueError("cache has no layers")

    start, stop = len_E, len_E + len_X

    sliced_data = []
    for layer_idx, layer in enumerate(cache.layers):
        if layer.keys is None or layer.values is None:
            raise ValueError(f"layer {layer_idx} is empty")

        seq_len = layer.keys.shape[SEQ_AXIS]
        if seq_len < stop:
            raise ValueError(
                f"layer {layer_idx}: cache holds {seq_len} tokens, "
                f"slice needs {stop} (len_E={len_E} + len_X={len_X})"
            )

        k = layer.keys[..., start:stop, :].clone()
        v = layer.values[..., start:stop, :].clone()
        sliced_data.append((k, v))

    return DynamicCache(ddp_cache_data=sliced_data)


def correct_position_ids(
    len_E: int,
    len_X: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return absolute position ids for a cache sliced out of E ⊕ X.

    The sliced tokens keep the positions they held in the full prefill:
    len_E, len_E + 1, ..., len_E + len_X - 1.

    Resetting these to 0 .. len_X - 1 corrupts RoPE silently. The rotation
    stays orthogonal, norms are preserved, nothing raises and nothing is
    NaN. Only the attention scores are wrong.

    Args:
        len_E: token length of E.
        len_X: token length of X.
        device: device for the returned tensor.

    Returns:
        Tensor of shape [1, len_X], dtype long.
    """
    if len_E < 0:
        raise ValueError(f"len_E must be non-negative, got {len_E}")
    if len_X <= 0:
        raise ValueError(f"len_X must be positive, got {len_X}")

    return torch.arange(len_E, len_E + len_X, dtype=torch.long, device=device).unsqueeze(0)


def next_position_id(len_E: int, len_X: int) -> int:
    """Return the absolute position of the first token generated after the slice.

    This is len_E + len_X, not len_X. The distinction is the whole point of
    correct_position_ids, applied to the decode step rather than the prefill.
    """
    if len_E < 0:
        raise ValueError(f"len_E must be non-negative, got {len_E}")
    if len_X <= 0:
        raise ValueError(f"len_X must be positive, got {len_X}")

    return len_E + len_X