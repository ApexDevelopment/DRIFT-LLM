"""
Gemma 4 Unified intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py
See commit history for authorship.

Gemma 4 Unified is the dense mid-size branch of the Gemma 4 family (gemma-4-12B-it and friends).
It drops Gemma 4's per-layer embeddings entirely but keeps the KV-sharing machinery (config-gated,
off in the released 12B), and adds two per-layer-type twists that stay inside the block:

  * **attention_k_eq_v.** Full-attention layers have no ``v_proj``; values reuse the key projection
    through their own norm. They also get their own kv-head count (``num_global_key_value_heads``)
    on top of Gemma 4's wider ``global_head_dim`` -- the backend derives both off the attention
    submodule when sizing this block's KV cache.

  * **layer_scalar.** Every layer ends with a trained scalar multiply (a persistent buffer, loaded
    by the block loader like any other checkpoint tensor).

The KV-sharing bridge below (donor writes ``shared_kv_states[layer_type]``, consumers attend
against it with a length-only mask cache) mirrors the Gemma 4 block.
"""
from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedTextDecoderLayer,
    Gemma4UnifiedTextRotaryEmbedding,
)

from drift.models._gemma_block import WrappedGemmaBlock
from drift.utils.misc import is_dummy, mps_gqa_eager_attention


class WrappedGemma4UnifiedBlock(WrappedGemmaBlock, Gemma4UnifiedTextDecoderLayer):
    """A DRIFT-LLM wrapper around a stock transformers ``Gemma4UnifiedTextDecoderLayer`` (text).

    Extends the shared Gemma wrapper (``WrappedGemmaBlock``: sliding/full pattern + per-layer-type
    rotary base) with Gemma 4's KV-sharing side channel. ``shared_kv_states`` is supplied by the
    caller (an upstream donor block, possibly on another server) rather than derived locally.
    """

    rotary_class = Gemma4UnifiedTextRotaryEmbedding
    rotary_takes_layer_type = True

    def __init__(self, config, layer_idx: int = 0):
        super().__init__(config, layer_idx=layer_idx)
        # Derived from the *true* global layer index by the stock attention in __init__, before the
        # cache index is normalized to 0. A sharing layer has no k/v/norm weights of its own.
        self.is_kv_shared_layer = self.self_attn.is_kv_shared_layer
        self.store_full_length_kv = self.self_attn.store_full_length_kv
        if self.is_kv_shared_layer:
            # Checkpoints ship k/v projections + norms for KV-shared layers even though the module
            # has none (it attends against the donor's K/V); tell the block loader these leftovers
            # are dropped by design, not lost trained state.
            self._keys_to_ignore_on_load_unexpected = [r"^self_attn\.(k_proj|v_proj|k_norm|v_norm)\."]

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        shared_kv_states: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        batch_size, seq_length, _ = hidden_states.shape
        if shared_kv_states is None:
            shared_kv_states = {}

        # Sharing layers attend against the donor's K/V (from `shared_kv_states`), not a local cache,
        # so they keep no BLOOM cache. We still build a length-only cache from the donor's key length
        # so the mask sees the correct number of past keys. Non-sharing layers use the real cache.
        mask_cache = DynamicCache()
        if self.is_kv_shared_layer:
            donor_key = shared_kv_states[self.layer_type][0]  # [batch, kv_heads, donor_len, head_dim]
            past_length = donor_key.shape[2] - seq_length
            if past_length > 0:
                mask_cache.update(donor_key[:, :, :past_length], donor_key[:, :, :past_length], 0)
            past_key_values = None  # stock attention reads shared_kv_states, must not touch a cache
        else:
            past_length = 0
            if layer_past is not None and not is_dummy(layer_past[0]):
                past_key, past_value = self._reorder_cache_from_bloom(layer_past, batch_size)
                past_length = past_key.shape[2]
                mask_cache.update(past_key, past_value, 0)
            past_key_values = mask_cache

        cache_position = torch.arange(past_length, past_length + seq_length, device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids, self.layer_type)

        mask_fn = create_sliding_window_causal_mask if self.layer_type == "sliding_attention" else create_causal_mask
        causal_mask = mask_fn(self.config, hidden_states, attention_mask, mask_cache, position_ids)

        with mps_gqa_eager_attention(self.config, hidden_states.device):
            hidden_states = Gemma4UnifiedTextDecoderLayer.forward(
                self,
                hidden_states,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )

        if use_cache:
            # Sharing layers keep no cache of their own -- they attend against the donor's K/V from
            # `shared_kv_states`. Return `None` in the cache slot so the backend skips the write-back.
            if self.is_kv_shared_layer:
                return hidden_states, None
            present = past_key_values.layers[0]
            return hidden_states, self._reorder_cache_to_bloom((present.keys, present.values), batch_size)
        return (hidden_states,)
