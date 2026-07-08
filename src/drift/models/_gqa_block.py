"""
Shared DRIFT-LLM wrapper for standard dense (MHA/GQA) transformer decoder layers.

Llama, Mistral, Qwen2 and Qwen3 expose the same post-refactor ``transformers`` decoder-layer
contract (rotary computed outside the layer, a ``Cache`` passed in, identical ``forward``
signature), so a single mixin bridges DRIFT-LLM' BLOOM-style KV layout to/from that contract for
all of them. A concrete block just inherits ``(WrappedGQABlock, <HF>DecoderLayer)`` and points
``rotary_class`` at the matching rotary-embedding module.

Architectures that add their own wrinkles on top of this contract keep their own wrappers:
Bloom/Falcon (fused QKV, ALiBi), Mixtral (MoE), and Gemma (dual local/global rotary + softcap).
"""
from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from drift.utils.misc import default_attn_implementation, is_dummy, mps_gqa_eager_attention


class BloomLayoutCacheMixin:
    """Converts DRIFT-LLM' BLOOM-layout ``layer_past`` to/from the ``transformers`` Cache layout.

    DRIFT-LLM stores keys as ``[batch * kv_heads, head_dim, seq]`` and values as
    ``[batch * kv_heads, seq, head_dim]`` (the historical BLOOM layout, see ``TransformerBackend``);
    ``transformers`` decoder layers want ``[batch, kv_heads, seq, head_dim]``. Shared by every
    wrapper whose underlying attention uses this standard per-head K/V cache.
    """

    def _reorder_cache_from_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Derive head counts / dims from the tensor shapes (not config) so this stays correct when
        # tensor parallelism gives a shard a subset of heads, and when the key and value head dims
        # differ (MLA: key = qk_nope + qk_rope, value = v_head_dim).
        key_states, value_states = key_value
        seq_length, v_head_dim = value_states.shape[1], value_states.shape[2]  # value: [b*kv, seq, v_head_dim]
        kv_heads = value_states.shape[0] // batch_size
        k_head_dim = key_states.shape[1]  # key (BLOOM): [b*kv, k_head_dim, seq]
        key_states = key_states.permute(0, 2, 1)  # -> [b*kv, seq, k_head_dim]
        key_states = key_states.reshape(batch_size, kv_heads, seq_length, k_head_dim)
        value_states = value_states.reshape(batch_size, kv_heads, seq_length, v_head_dim)
        return key_states, value_states

    def _reorder_cache_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states, value_states = key_value  # both: [batch, kv_heads, seq_length, head_dim] (dims may differ)
        kv_heads, seq_length = key_states.shape[1], key_states.shape[2]
        value_states = value_states.reshape(batch_size * kv_heads, seq_length, value_states.shape[3])
        key_states = key_states.reshape(batch_size * kv_heads, seq_length, key_states.shape[3])
        key_states = key_states.permute(0, 2, 1)  # [batch * kv_heads, head_dim, seq_length]
        return key_states, value_states


class WrappedGQABlock(BloomLayoutCacheMixin):
    """Mixin implementing DRIFT-LLM' cache bridging around a stock dense decoder layer.

    DRIFT-LLM stores past keys/values in a BLOOM-style layout (see ``TransformerBackend``) and
    passes them to the block as ``layer_past``. Since ``transformers`` >=5.0 the decoder layer
    consumes a ``transformers.cache_utils.Cache`` and computes rotary embeddings at the model
    level, so this mixin:

      * owns a rotary-embedding module (``rotary_class``, moved out of attention upstream),
      * converts ``layer_past`` <-> a ``DynamicCache`` around the stock forward, and
      * returns the updated keys/values back in the BLOOM layout,

    keeping the server's memory-cache format unchanged.
    """

    rotary_class = None  # set by concrete subclasses to the matching *RotaryEmbedding

    def __init__(self, config, layer_idx: int = 0):
        # layer_idx only matters for KV caching, which DRIFT-LLM re-implements, so we always use 0
        super().__init__(config, layer_idx=0)
        self.config = config
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = default_attn_implementation(config)
        self.rotary_emb = self.rotary_class(config=config)

    def _build_causal_mask(self, hidden_states, attention_mask, past_key_values, position_ids):
        # Mistral/Qwen enable a sliding window via config; honor it so tokens past the window
        # don't attend (Llama leaves sliding_window unset and falls through to the dense mask).
        use_sliding = getattr(self.config, "sliding_window", None) is not None and getattr(
            self.config, "use_sliding_window", True
        )
        mask_fn = create_sliding_window_causal_mask if use_sliding else create_causal_mask
        return mask_fn(self.config, hidden_states, attention_mask, past_key_values, position_ids)

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
        if layer_past is not None and not is_dummy(layer_past[0]):
            past_key, past_value = self._reorder_cache_from_bloom(layer_past, batch_size)
            past_length = past_key.shape[2]
            past_key_values.update(past_key, past_value, self.self_attn.layer_idx)

        cache_position = torch.arange(past_length, past_length + seq_length, device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        causal_mask = self._build_causal_mask(hidden_states, attention_mask, past_key_values, position_ids)

        with mps_gqa_eager_attention(self.config, hidden_states.device):
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
            return hidden_states, self._reorder_cache_to_bloom((present.keys, present.values), batch_size)
        return (hidden_states,)
