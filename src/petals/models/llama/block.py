"""
LLaMA intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
See commit history for authorship.
"""
from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding


class WrappedLlamaBlock(LlamaDecoderLayer):
    """
    A Petals wrapper around a stock transformers ``LlamaDecoderLayer``.

    Petals stores past keys/values in a BLOOM-style layout (see ``TransformerBackend``) and
    passes them to the block as ``layer_past``. Since transformers >=5.0 the decoder layer
    consumes a ``transformers.cache_utils.Cache`` and computes rotary embeddings at the model
    level, so this wrapper:

      * owns a ``LlamaRotaryEmbedding`` (moved out of the attention module upstream),
      * converts ``layer_past`` <-> a ``DynamicCache`` around the stock forward, and
      * returns the updated keys/values back in the BLOOM layout,

    keeping the server's memory-cache format unchanged.
    """

    def __init__(self, config, layer_idx: int = 0):
        # layer_idx only matters for KV caching, which Petals re-implements, so we always use 0
        super().__init__(config, layer_idx=0)
        self.config = config
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = "eager"
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        batch_size, seq_length, _ = hidden_states.shape

        past_key_values = DynamicCache()
        past_length = 0
        if layer_past is not None:
            past_key, past_value = self._reorder_cache_from_bloom_to_llama(layer_past, batch_size)
            past_length = past_key.shape[2]
            past_key_values.update(past_key, past_value, self.self_attn.layer_idx)

        cache_position = torch.arange(past_length, past_length + seq_length, device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        causal_mask = create_causal_mask(
            self.config,
            hidden_states,
            attention_mask,
            past_key_values,
            position_ids,
        )

        output = super().forward(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )
        hidden_states = output[0] if isinstance(output, tuple) else output

        if use_cache:
            present = past_key_values.layers[self.self_attn.layer_idx]
            present_key_value = self._reorder_cache_from_llama_to_bloom((present.keys, present.values), batch_size)
            return hidden_states, present_key_value
        return (hidden_states,)

    def _reorder_cache_from_bloom_to_llama(
        self, key_value: Tuple[torch.Tensor], batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states, value_states = key_value
        kv_heads = self.config.num_key_value_heads
        head_dim = self.self_attn.head_dim
        seq_length = value_states.shape[1]  # value (BLOOM): [batch * kv_heads, seq_length, head_dim]
        key_states = key_states.permute(0, 2, 1)  # key (BLOOM): [batch * kv_heads, head_dim, seq_length]
        key_states = key_states.reshape(batch_size, kv_heads, seq_length, head_dim)
        value_states = value_states.reshape(batch_size, kv_heads, seq_length, head_dim)
        return key_states, value_states

    def _reorder_cache_from_llama_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states, value_states = key_value  # both: [batch, kv_heads, seq_length, head_dim]
        kv_heads = self.config.num_key_value_heads
        head_dim = self.self_attn.head_dim
        seq_length = key_states.shape[2]
        value_states = value_states.reshape(batch_size * kv_heads, seq_length, head_dim)
        key_states = key_states.reshape(batch_size * kv_heads, seq_length, head_dim)
        key_states = key_states.permute(0, 2, 1)  # [batch * kv_heads, head_dim, seq_length]
        return key_states, value_states
