"""Flatten/unflatten the Gemma 4 KV-sharing side channel for the wire.

Gemma 4's last ``num_kv_shared_layers`` layers reuse the full-length keys/values of the last
non-shared layer of the same attention type (the *donor*), addressed by ``layer_type`` in a
``shared_kv_states`` dict (``"full_attention"`` / ``"sliding_attention"`` -> ``(key, value)``).
When a donor and its consumer land on **different servers**, that dict must cross the wire: the
donor's server emits it, the client accumulates it across spans, and the consumer's server is
seeded from it.

A ``shared_kv_states`` dict is carried as a parallel ``(keys, tensors)`` pair: ``keys`` is an
ordered list of layer-type strings (msgpack-friendly, goes in request/response metadata) and
``tensors`` is the flat ``[k0, v0, k1, v1, ...]`` list appended to the tensor payload. The order
is deterministic (sorted by layer type) so both sides agree without extra bookkeeping.
"""
from typing import Dict, List, Tuple

import torch


def flatten_shared_kv(
    shared_kv_states: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[List[str], List[torch.Tensor]]:
    """Split a ``shared_kv_states`` dict into an ordered ``(layer_types, [k0, v0, k1, v1, ...])`` pair."""
    keys: List[str] = []
    tensors: List[torch.Tensor] = []
    for layer_type in sorted(shared_kv_states):
        key, value = shared_kv_states[layer_type]
        keys.append(layer_type)
        tensors.extend((key, value))
    return keys, tensors


def unflatten_shared_kv(keys: List[str], tensors: List[torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Rebuild a ``shared_kv_states`` dict from the ordered ``(layer_types, flat_tensors)`` pair."""
    assert len(tensors) == 2 * len(keys), f"expected 2 tensors per layer type, got {len(tensors)} for {len(keys)} keys"
    return {layer_type: (tensors[2 * i], tensors[2 * i + 1]) for i, layer_type in enumerate(keys)}
