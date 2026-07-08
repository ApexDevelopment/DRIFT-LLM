"""
Mixtral intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py
See commit history for authorship.
"""
from typing import Optional, Tuple

import torch
from transformers import MixtralConfig
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer, MixtralRotaryEmbedding

from drift.utils.misc import default_attn_implementation, is_dummy, mps_gqa_eager_attention


class WrappedMixtralBlock(MixtralDecoderLayer):
    """A DRIFT-LLM wrapper around a stock transformers ``MixtralDecoderLayer`` (GQA + sliding window + MoE).

    See ``drift.models.llama.block.WrappedLlamaBlock`` for the BLOOM-layout KV bridging rationale.
    """

    def __init__(self, config: MixtralConfig, layer_idx: int = 0):
        # layer_idx only matters for KV caching, which DRIFT-LLM re-implements, so we always use 0
        super().__init__(config, layer_idx=0)
        self.config = config
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = default_attn_implementation(config)
        self.rotary_emb = MixtralRotaryEmbedding(config=config)

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
            past_key, past_value = self._reorder_cache_from_bloom_to_mixtral(layer_past, batch_size)
            past_length = past_key.shape[2]
            past_key_values.update(past_key, past_value, self.self_attn.layer_idx)

        cache_position = torch.arange(past_length, past_length + seq_length, device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            self.config,
            hidden_states,
            attention_mask,
            past_key_values,
            position_ids,
        )

        with mps_gqa_eager_attention(self.config, hidden_states.device):
            output = super().forward(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
            )
        hidden_states = output[0] if isinstance(output, tuple) else output

        if use_cache:
            present = past_key_values.layers[self.self_attn.layer_idx]
            present_key_value = self._reorder_cache_from_mixtral_to_bloom((present.keys, present.values), batch_size)
            return hidden_states, present_key_value
        return (hidden_states,)

    def _reorder_cache_from_bloom_to_mixtral(
        self, key_value: Tuple[torch.Tensor], batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Derive the KV-head count from the tensor shapes (not config) so this stays correct when
        # tensor parallelism gives a shard only a subset of the key/value heads.
        key_states, value_states = key_value
        seq_length, head_dim = value_states.shape[1], value_states.shape[2]  # value: [batch * kv_heads, seq, head_dim]
        kv_heads = value_states.shape[0] // batch_size
        key_states = key_states.permute(0, 2, 1)  # key (BLOOM): [batch * kv_heads, head_dim, seq_length]
        key_states = key_states.reshape(batch_size, kv_heads, seq_length, head_dim)
        value_states = value_states.reshape(batch_size, kv_heads, seq_length, head_dim)
        return key_states, value_states

    def _reorder_cache_from_mixtral_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states, value_states = key_value  # both: [batch, kv_heads, seq_length, head_dim]
        kv_heads, seq_length, head_dim = key_states.shape[1], key_states.shape[2], key_states.shape[3]
        value_states = value_states.reshape(batch_size * kv_heads, seq_length, head_dim)
        key_states = key_states.reshape(batch_size * kv_heads, seq_length, head_dim)
        key_states = key_states.permute(0, 2, 1)  # [batch * kv_heads, head_dim, seq_length]
        return key_states, value_states
