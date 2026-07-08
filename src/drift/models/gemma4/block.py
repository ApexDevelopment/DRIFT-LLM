"""
Gemma 4 intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py
See commit history for authorship.

Gemma 4 keeps Gemma 3's alternating sliding/full attention and dual local/global rotary base, and
adds two features that reach *outside* a single decoder layer -- which DRIFT-LLM' otherwise
position- and layer-agnostic block has to bridge:

  * **Per-layer inputs.** ``Gemma4TextModel`` derives a per-token, per-layer embedding from the raw
    ``input_ids`` (a separate ``embed_tokens_per_layer`` table) and feeds layer ``i`` its slice,
    consumed multiplicatively at the *end* of the layer. DRIFT-LLM servers only receive
    ``hidden_states`` over the wire, so the client computes the per-layer inputs and threads the
    matching slice into each block as ``per_layer_input``.

  * **KV sharing.** The last ``num_kv_shared_layers`` layers carry no k/v projections; they reuse
    the full-length keys/values of the last non-shared layer of the same attention type (the
    *donor*). The donor writes its K/V into ``shared_kv_states[layer_type]`` and the sharing layers
    read it back. Across a swarm the donor's K/V is propagated downstream via the same side channel.

Because sharing layers have no local KV cache, this block seeds a length-only cache from the donor's
key length purely so the causal / sliding-window mask uses the right key offset.
"""
from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

from drift.models._gemma_block import WrappedGemmaBlock
from drift.utils.misc import is_dummy, mps_gqa_eager_attention


class WrappedGemma4Block(WrappedGemmaBlock, Gemma4TextDecoderLayer):
    """A DRIFT-LLM wrapper around a stock transformers ``Gemma4TextDecoderLayer`` (text).

    Extends the shared Gemma wrapper (``WrappedGemmaBlock``: sliding/full pattern + per-layer-type
    rotary base) with Gemma 4's per-layer-input and KV-sharing side channels. ``per_layer_input`` and
    ``shared_kv_states`` are supplied by the caller (the client for per-layer inputs, an upstream
    donor block for shared K/V) rather than derived locally.
    """

    rotary_class = Gemma4TextRotaryEmbedding
    rotary_takes_layer_type = True

    def __init__(self, config, layer_idx: int = 0):
        super().__init__(config, layer_idx=layer_idx)
        # Derived from the *true* global layer index by the stock attention in __init__, before the
        # cache index is normalized to 0. A sharing layer has no k/v/norm weights of its own.
        self.is_kv_shared_layer = self.self_attn.is_kv_shared_layer
        self.store_full_length_kv = self.self_attn.store_full_length_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        per_layer_input: Optional[torch.Tensor] = None,
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
            hidden_states = Gemma4TextDecoderLayer.forward(
                self,
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )

        if use_cache and not self.is_kv_shared_layer:
            present = past_key_values.layers[0]
            return hidden_states, self._reorder_cache_to_bloom((present.keys, present.values), batch_size)
        return (hidden_states,)
