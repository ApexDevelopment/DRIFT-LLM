"""
DeepSeek-V3 intermediate layer (Multi-head Latent Attention + MoE).
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py
See commit history for authorship.
"""
from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3DecoderLayer, DeepseekV3RotaryEmbedding

from drift.models._gqa_block import BloomLayoutCacheMixin
from drift.utils.misc import default_attn_implementation, is_dummy


class WrappedDeepseekV3Block(BloomLayoutCacheMixin, DeepseekV3DecoderLayer):
    """A DRIFT-LLM wrapper around a stock transformers ``DeepseekV3DecoderLayer``.

    Two DeepSeek-specific points, both handled here:

      * whether a layer's FFN is dense or MoE is chosen by the global layer index
        (``layer_idx >= first_k_dense_replace``), so -- like Gemma -- the block is built with its
        *true* index, then the cache index is normalized to 0 (DRIFT-LLM owns a single-layer cache), and
      * MLA decompresses the KV latent before caching (HF does this in attention), so DRIFT-LLM keeps
        the per-head BLOOM layout, but keys and values have different head dims -- see
        ``BloomLayoutCacheMixin`` / ``MLACache``.
    """

    def __init__(self, config, layer_idx: int = 0):
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = default_attn_implementation(config)
        # Build with the true index so the dense-vs-MoE FFN is chosen correctly, then normalize the
        # cache index to 0.
        super().__init__(config, layer_idx=layer_idx)
        self.self_attn.layer_idx = 0
        self.config = config
        self.rotary_emb = DeepseekV3RotaryEmbedding(config=config)

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
            past_key_values.update(past_key, past_value, 0)

        cache_position = torch.arange(past_length, past_length + seq_length, device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        causal_mask = create_causal_mask(self.config, hidden_states, attention_mask, past_key_values, position_ids)

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
            present = past_key_values.layers[0]
            return hidden_states, self._reorder_cache_to_bloom((present.keys, present.values), batch_size)
        return (hidden_states,)
