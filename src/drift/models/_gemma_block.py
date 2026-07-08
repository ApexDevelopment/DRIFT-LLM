"""
Shared DRIFT-LLM wrapper for Gemma 2 / Gemma 3 decoder layers.

Gemma differs from the plain GQA models in two position-dependent ways that DRIFT-LLM' otherwise
position-agnostic block abstraction has to account for:

  * layers alternate between **sliding** and **full** attention (``config.layer_types[layer_idx]``),
    which selects both the attention mask and, for Gemma 3, the rotary base (local vs global), and
  * the rotary embeddings are computed at the model level and passed into the layer.

So the block is constructed with its **true** global ``layer_idx`` (letting the stock attention
derive the correct ``layer_type`` / ``sliding_window`` / scaling), after which the *cache* index is
normalized to 0 because DRIFT-LLM owns a single-layer KV cache. Softcapping, query pre-attention
scaling and QK-norm all live inside the stock attention and need no special handling here.
"""
from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from drift.models._gqa_block import BloomLayoutCacheMixin
from drift.utils.misc import default_attn_implementation, is_dummy, mps_gqa_eager_attention


class WrappedGemmaBlock(BloomLayoutCacheMixin):
    """Mixin bridging DRIFT-LLM' BLOOM-layout cache around a stock Gemma decoder layer.

    Concrete blocks inherit ``(WrappedGemmaBlock, <HF>DecoderLayer)`` and set ``rotary_class``;
    Gemma 3 additionally sets ``rotary_takes_layer_type`` because its rotary module produces a
    different base frequency for sliding vs full layers.
    """

    rotary_class = None  # set by concrete subclasses to the matching *RotaryEmbedding
    rotary_takes_layer_type = False  # Gemma 3's rotary picks local/global by layer_type

    def __init__(self, config, layer_idx: int = 0):
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = default_attn_implementation(config)
        # Build attention from the true global index so it derives the right attention type,
        # sliding window and rotary base; then normalize the cache index (DRIFT-LLM owns one layer).
        super().__init__(config, layer_idx=layer_idx)
        self.self_attn.layer_idx = 0
        self.config = config
        self.rotary_emb = self.rotary_class(config)
        self.layer_type = self.self_attn.layer_type  # 'sliding_attention' | 'full_attention'

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
        if self.rotary_takes_layer_type:
            position_embeddings = self.rotary_emb(hidden_states, position_ids, self.layer_type)
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        mask_fn = create_sliding_window_causal_mask if self.layer_type == "sliding_attention" else create_causal_mask
        causal_mask = mask_fn(self.config, hidden_states, attention_mask, past_key_values, position_ids)

        with mps_gqa_eager_attention(self.config, hidden_states.device):
            output = super().forward(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )
        hidden_states = output[0] if isinstance(output, tuple) else output

        if use_cache:
            present = past_key_values.layers[0]
            return hidden_states, self._reorder_cache_to_bloom((present.keys, present.values), batch_size)
        return (hidden_states,)
