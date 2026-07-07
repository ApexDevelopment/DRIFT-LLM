"""
Per-architecture tensor-parallel slicing configs for the transformer blocks DRIFT-LLM serves.

Unlike the upstream ``tensor_parallel`` library (which shipped a config only for BLOOM and fell
back to a correct-but-inefficient auto config for everything else), these configs express real
*head-parallel* sharding for each architecture: attention heads (and MoE experts) are split
across devices, matching the head-split KV cache in ``drift/server/backend.py``.

DRIFT-LLM manages the KV cache at the *block* level (each ``Wrapped*Block.forward`` owns a
``DynamicCache`` and returns BLOOM-layout key/value tensors), so the cache plumbing lives on the
root block: an input rule selects this rank's slice out of the incoming ``PerDeviceTensors`` and
an output rule gathers every rank's key/value back into a ``PerDeviceTensors`` for the caller.
"""
from functools import partial
from itertools import chain
from typing import Optional, Sequence

import torch
from transformers import PretrainedConfig

from drift.utils.tensor_parallel.communications import CollectiveOperation
from drift.utils.tensor_parallel.slicer import Config
from drift.utils.tensor_parallel.wrapper import PerDeviceTensors

__all__ = ["get_tensor_parallel_config", "get_bloom_config"]


def split_heads(tensor: torch.Tensor, *, dim: int, head_dim: int, rank: int, world_size: int, optional: bool = False):
    """Split a tensor along ``dim`` so that each part is a whole number of ``head_dim``-sized heads."""
    if tensor is None and optional:
        return None
    assert tensor.shape[dim] % head_dim == 0, tensor.shape
    if dim < 0:
        dim = (tensor.ndim + dim) % tensor.ndim
    shape = list(tensor.shape)
    shape[dim] //= head_dim
    shape.insert(dim + 1, head_dim)
    tensor_part = tensor.reshape(shape).tensor_split(world_size, dim=dim)[rank].flatten(dim, dim + 1)
    return tensor_part


def split_num_heads(num_heads: int, *, rank: int, world_size: int) -> int:
    return torch.empty(num_heads, device="meta").tensor_split(world_size)[rank].numel()


def split_alibi(alibi: torch.Tensor, *, rank: int, num_heads: int, world_size: int) -> torch.Tensor:
    """Split an alibi tensor of shape ``[batch_size * num_heads, ...]`` over attention heads."""
    alibi_expanded = alibi.reshape(-1, num_heads, *alibi.shape[1:])
    alibi_part = alibi_expanded.tensor_split(world_size, dim=1)[rank]
    return alibi_part.reshape(-1, *alibi.shape[1:])


def _make_kv_collectives(world_size: int):
    """Build the block-level KV cache collectives shared by all architectures.

    ``gather_kv`` runs after the block forward: it takes each rank's ``(key, value)`` (BLOOM layout,
    holding only that rank's heads) and returns the same ``PerDeviceTensors(k0, v0, k1, v1, ...)`` to
    every rank. ``select_kv`` runs before the block forward: it picks this rank's ``(key, value)`` out
    of that ``PerDeviceTensors`` (or returns ``None`` on the first step, when there is no past).
    """
    gather_kv = CollectiveOperation(
        world_size=world_size,
        func=lambda *kvs: [PerDeviceTensors(*chain(*(x or [None] for x in kvs)))] * world_size,
    )
    # Only a gathered multi-shard cache (PerDeviceTensors) is selected per rank; anything else --
    # None (prefill) or the DUMMY empty cache -- passes through untouched so the block's own
    # ``is_dummy`` handling applies.
    select_kv = lambda kvs, rank: (kvs[2 * rank], kvs[2 * rank + 1]) if isinstance(kvs, PerDeviceTensors) else kvs
    return gather_kv, select_kv


def get_bloom_config(model_config: PretrainedConfig, devices: Sequence[torch.device]) -> Config:
    world_size = len(devices)
    num_heads = model_config.n_head
    head_dim = model_config.hidden_size // num_heads
    gather_kv, select_kv = _make_kv_collectives(world_size)
    _split_alibi = partial(split_alibi, num_heads=num_heads, world_size=world_size)

    return Config(
        state_rules={
            r".*self_attention\.query_key_value\.(weight|bias)$": partial(
                split_heads, dim=0, head_dim=head_dim * 3, world_size=world_size
            ),
            r".*self_attention\.dense\.weight$": partial(split_heads, dim=1, head_dim=head_dim, world_size=world_size),
            r".*self_attention\.dense\.bias$": "scale",
            r".*mlp\.dense_h_to_4h\.(weight|bias)$": "split 0",
            r".*mlp\.dense_4h_to_h\.weight$": "split 1",
            r".*mlp\.dense_4h_to_h\.bias$": "scale",
        },
        input_rules={
            r".*self_attention$": {"alibi": _split_alibi},
            r"^$": {"layer_past": select_kv},  # root block: pick this rank's past key/value
        },
        output_rules={
            r".*self_attention\.dense$": {0: "sum"},
            r".*mlp\.dense_4h_to_h$": {0: "sum"},
            r"^$": {1: gather_kv},  # root block: gather every rank's key/value
        },
        attr_rules={r".*self_attention$": {"num_heads": partial(split_num_heads, world_size=world_size)}},
    )


def _gqa_head_dims(model_config: PretrainedConfig):
    num_heads = model_config.num_attention_heads
    num_kv_heads = getattr(model_config, "num_key_value_heads", None) or num_heads
    head_dim = getattr(model_config, "head_dim", None) or model_config.hidden_size // num_heads
    groups = num_heads // num_kv_heads  # query heads per kv head; stays constant per shard
    return num_heads, num_kv_heads, head_dim, groups


def _gqa_attention_state_rules(head_dim: int, groups: int, world_size: int) -> dict:
    """State rules that split ``self_attn`` query/key/value/output projections at kv-group granularity.

    Query (and output) heads are split in blocks of ``groups`` query heads -- one kv group -- while
    key/value heads are split one at a time, so query head ``i`` stays paired with kv head ``i // groups``
    and ``num_key_value_groups`` (which the stock ``repeat_kv`` reads off the module) is unchanged per shard.
    """
    return {
        r".*self_attn\.q_proj\.(weight|bias)$": partial(
            split_heads, dim=0, head_dim=groups * head_dim, world_size=world_size
        ),
        r".*self_attn\.k_proj\.(weight|bias)$": partial(split_heads, dim=0, head_dim=head_dim, world_size=world_size),
        r".*self_attn\.v_proj\.(weight|bias)$": partial(split_heads, dim=0, head_dim=head_dim, world_size=world_size),
        r".*self_attn\.o_proj\.weight$": partial(split_heads, dim=1, head_dim=groups * head_dim, world_size=world_size),
        r".*self_attn\.o_proj\.bias$": "scale",
    }


def get_llama_config(model_config: PretrainedConfig, devices: Sequence[torch.device]) -> Config:
    """Head-parallel config for Llama-family GQA blocks (Llama, Mistral, Qwen2/3, Gemma, ...).

    Requires ``num_key_value_heads`` >= ``world_size`` (at least one kv head per device).
    """
    world_size = len(devices)
    num_heads, num_kv_heads, head_dim, groups = _gqa_head_dims(model_config)
    assert num_kv_heads >= world_size, (
        f"Cannot tensor-parallelize {num_kv_heads} kv heads across {world_size} devices; "
        f"need at least one kv head per device"
    )
    gather_kv, select_kv = _make_kv_collectives(world_size)

    return Config(
        state_rules={
            **_gqa_attention_state_rules(head_dim, groups, world_size),
            r".*mlp\.(gate_proj|up_proj)\.(weight|bias)$": "split 0",
            r".*mlp\.down_proj\.weight$": "split 1",
            r".*mlp\.down_proj\.bias$": "scale",
        },
        input_rules={r"^$": {"layer_past": select_kv}},
        output_rules={
            r".*self_attn\.o_proj$": {0: "sum"},
            r".*mlp\.down_proj$": {0: "sum"},
            r"^$": {1: gather_kv},
        },
        attr_rules={},
    )


def split_gate_up_experts(tensor: torch.Tensor, *, rank: int, world_size: int) -> torch.Tensor:
    """Split fused MoE ``gate_up_proj`` ``[num_experts, 2 * intermediate, hidden]`` over the intermediate dim.

    The output dim packs ``[gate; up]``, so each half's intermediate slice must be split independently
    and kept aligned with the ``down_proj`` input split.
    """
    num_experts, two_intermediate, hidden = tensor.shape
    reshaped = tensor.reshape(num_experts, 2, two_intermediate // 2, hidden)
    part = reshaped.tensor_split(world_size, dim=2)[rank]
    return part.reshape(num_experts, -1, hidden)


def get_mixtral_config(model_config: PretrainedConfig, devices: Sequence[torch.device]) -> Config:
    """Head-parallel config for Mixtral-style MoE blocks: GQA attention + expert MLPs sharded over
    the intermediate dim (fused ``experts.gate_up_proj``/``down_proj``), router (``gate``) replicated.
    """
    world_size = len(devices)
    num_heads, num_kv_heads, head_dim, groups = _gqa_head_dims(model_config)
    assert num_kv_heads >= world_size, (
        f"Cannot tensor-parallelize {num_kv_heads} kv heads across {world_size} devices; "
        f"need at least one kv head per device"
    )
    gather_kv, select_kv = _make_kv_collectives(world_size)

    return Config(
        state_rules={
            **_gqa_attention_state_rules(head_dim, groups, world_size),
            r".*mlp\.experts\.gate_up_proj$": partial(split_gate_up_experts, world_size=world_size),
            r".*mlp\.experts\.down_proj$": "split 2",  # [num_experts, hidden, intermediate] -> split intermediate
            # note: the router (mlp.gate.weight) is replicated so every shard routes identically
        },
        input_rules={r"^$": {"layer_past": select_kv}},
        output_rules={
            r".*self_attn\.o_proj$": {0: "sum"},
            r".*mlp$": {0: "sum"},  # sum each rank's partial MoE output (down_proj is row-parallel)
            r"^$": {1: gather_kv},
        },
        attr_rules={},
    )


def split_falcon_alibi(alibi, *, rank: int, world_size: int):
    """Split Falcon's per-head alibi ``[batch, num_heads, 1, seq]`` over heads (``None`` for rotary variants)."""
    if alibi is None:
        return alibi
    return alibi.tensor_split(world_size, dim=1)[rank]


def split_falcon_attention_mask(mask, *, rank: int, world_size: int):
    """Split Falcon's attention mask over heads only when it is per-head.

    DRIFT-LLM folds alibi into the 4D mask, giving ``[batch, num_heads, q, kv]`` (per head) for the alibi
    variants; the rotary variants pass a head-agnostic ``[batch, 1, q, kv]`` mask that must stay shared.
    """
    if mask is None or mask.ndim != 4 or mask.shape[1] == 1:
        return mask
    return mask.tensor_split(world_size, dim=1)[rank]


def get_falcon_config(model_config: PretrainedConfig, devices: Sequence[torch.device]) -> Config:
    """Head-parallel config for Falcon blocks (fused ``query_key_value``; alibi or rotary; GQA/MHA).

    Falcon fuses q/k/v into one projection laid out as ``num_kv_heads`` groups of
    ``[groups query heads, 1 key head, 1 value head]`` (regular MHA is the ``groups == 1`` special
    case), so both split cleanly at kv-group granularity. ``multi_query`` (a single shared kv head)
    cannot be head-split and is rejected for ``world_size > 1``. Requires
    ``num_kv_heads`` % ``world_size`` == 0 because ``_split_heads`` re-derives ``groups`` from the
    (sharded) ``num_heads`` / ``num_kv_heads`` attributes.
    """
    world_size = len(devices)
    num_heads = model_config.num_attention_heads
    head_dim = getattr(model_config, "head_dim", None) or model_config.hidden_size // num_heads
    if model_config.new_decoder_architecture or not model_config.multi_query:
        num_kv_heads = model_config.num_kv_heads
    else:  # multi_query: a single kv head shared by every query head
        num_kv_heads = 1
    if world_size > 1:
        if num_kv_heads < world_size or num_kv_heads % world_size != 0:
            raise NotImplementedError(
                f"Falcon tensor parallelism requires num_kv_heads ({num_kv_heads}) to be divisible by the number "
                f"of devices ({world_size}); multi-query Falcon (1 kv head) cannot be head-split -- run it on a "
                f"single device or use a Falcon checkpoint with more kv heads"
            )
    groups = num_heads // num_kv_heads  # query heads per kv head; must stay constant per shard
    gather_kv, select_kv = _make_kv_collectives(world_size)

    return Config(
        state_rules={
            # fused qkv split in kv-group blocks of [groups q heads, 1 k head, 1 v head]
            r".*self_attention\.query_key_value\.(weight|bias)$": partial(
                split_heads, dim=0, head_dim=(groups + 2) * head_dim, world_size=world_size
            ),
            r".*self_attention\.dense\.weight$": partial(
                split_heads, dim=1, head_dim=groups * head_dim, world_size=world_size
            ),
            r".*self_attention\.dense\.bias$": "scale",
            r".*mlp\.dense_h_to_4h\.(weight|bias)$": "split 0",
            r".*mlp\.dense_4h_to_h\.weight$": "split 1",
            r".*mlp\.dense_4h_to_h\.bias$": "scale",
        },
        input_rules={
            r".*self_attention$": {
                "alibi": partial(split_falcon_alibi, world_size=world_size),
                "attention_mask": partial(split_falcon_attention_mask, world_size=world_size),
            },
            r"^$": {"layer_past": select_kv},
        },
        output_rules={
            r".*self_attention\.dense$": {0: "sum"},
            r".*mlp\.dense_4h_to_h$": {0: "sum"},
            r"^$": {1: gather_kv},
        },
        attr_rules={
            r".*self_attention$": {
                "num_heads": partial(split_num_heads, world_size=world_size),
                "num_kv_heads": partial(split_num_heads, world_size=world_size),
            }
        },
    )


# model_type -> config factory. Architectures absent here fall back to the generic auto config
# (correct, but gathers after every linear).
_CONFIG_GETTERS = {
    "bloom": get_bloom_config,
    "llama": get_llama_config,
    "mixtral": get_mixtral_config,
    "falcon": get_falcon_config,
}


def get_tensor_parallel_config(model_config: PretrainedConfig, devices: Sequence[torch.device]) -> Optional[Config]:
    """Return the tensor-parallel :class:`Config` for a model, or ``None`` to use the generic auto config."""
    getter = _CONFIG_GETTERS.get(model_config.model_type)
    return getter(model_config, devices) if getter is not None else None
