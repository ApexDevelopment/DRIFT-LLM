"""
Bloom intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/bloom/modeling_bloom.py
See commit history for authorship.
"""
from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.bloom.modeling_bloom import BloomBlock, build_alibi_tensor

from drift.utils.misc import is_dummy


class WrappedBloomBlock(BloomBlock):
    """A DRIFT-LLM wrapper around a stock transformers ``BloomBlock`` (ALiBi attention).

    See ``drift.models.llama.block.WrappedLlamaBlock`` for the BLOOM-layout KV bridging rationale
    (Bloom is multi-head attention, so the number of KV heads equals ``num_attention_heads``).
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__(config, layer_idx=0)
        self.config = config
        if getattr(config, "_attn_implementation", None) is None:
            # Bloom is always ALiBi, which DRIFT-LLM folds into a 4D mask the sdpa path can't consume,
            # so eager is required here (not merely the default) for correct results.
            config._attn_implementation = "eager"
        # Disable the legacy Megatron `pretraining_tp` slow path: it slices `dense.weight` /
        # `dense_4h_to_h.weight` by hand (bypassing the module forward), which is incompatible with
        # tensor-parallel sharding, and is numerically equivalent to the standard path anyway.
        self.self_attention.pretraining_tp = 1
        self.mlp.pretraining_tp = 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        alibi: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        batch_size, seq_length = hidden_states.shape[:2]

        past_key_values = DynamicCache()
        past_length = 0
        if layer_past is not None and not is_dummy(layer_past[0]):
            past_key, past_value = self._reorder_cache_from_bloom(layer_past, batch_size)
            past_length = past_key.shape[2]
            past_key_values.update(past_key, past_value, self.self_attention.layer_idx)

        cache_position = torch.arange(past_length, past_length + seq_length, device=hidden_states.device)
        ones_mask = torch.ones((batch_size, past_length + seq_length), device=hidden_states.device)
        if alibi is None:
            alibi = build_alibi_tensor(ones_mask, self.num_heads, dtype=hidden_states.dtype)

        causal_mask = create_causal_mask(self.config, hidden_states, None, past_key_values, cache_position.unsqueeze(0))

        output = super().forward(
            hidden_states,
            alibi=alibi,
            attention_mask=causal_mask,
            layer_past=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = output[0] if isinstance(output, tuple) else output

        if use_cache:
            present = past_key_values.layers[self.self_attention.layer_idx]
            return hidden_states, self._reorder_cache_to_bloom((present.keys, present.values), batch_size)
        return (hidden_states,)

    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Derive head count / head_dim from the tensor shapes (not config) so that this stays correct
        # when tensor parallelism gives a shard only a subset of the attention heads.
        key_states, value_states = key_value
        seq_length, head_dim = value_states.shape[1], value_states.shape[2]  # value: [batch * heads, seq, head_dim]
        num_heads = value_states.shape[0] // batch_size
        key_states = key_states.permute(0, 2, 1)  # key (BLOOM): [batch * heads, head_dim, seq_length]
        key_states = key_states.reshape(batch_size, num_heads, seq_length, head_dim)
        value_states = value_states.reshape(batch_size, num_heads, seq_length, head_dim)
        return key_states, value_states

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states, value_states = key_value  # both: [batch, heads, seq_length, head_dim]
        num_heads, seq_length, head_dim = key_states.shape[1], key_states.shape[2], key_states.shape[3]
        value_states = value_states.reshape(batch_size * num_heads, seq_length, head_dim)
        key_states = key_states.reshape(batch_size * num_heads, seq_length, head_dim)
        key_states = key_states.permute(0, 2, 1)  # [batch * heads, head_dim, seq_length]
        return key_states, value_states
