"""
Falcon intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
See commit history for authorship.
"""
import math
from typing import Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer, FalconRotaryEmbedding, build_alibi_tensor

from drift.utils.misc import default_attn_implementation, is_dummy


class WrappedFalconBlock(FalconDecoderLayer):
    """A DRIFT-LLM wrapper around a stock transformers ``FalconDecoderLayer`` (ALiBi or rotary, GQA)."""

    def __init__(self, config, layer_idx: int = 0):
        # FalconDecoderLayer.__init__ selects the attention class by config._attn_implementation,
        # so it must be set before calling super().__init__. default_attn_implementation keeps ALiBi
        # Falcon on eager (folded 4D mask) while letting rotary Falcon use the faster sdpa path.
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = default_attn_implementation(config)
        super().__init__(config, layer_idx=0)
        self.config = config
        self.num_heads = config.num_attention_heads
        self.rotary_emb = FalconRotaryEmbedding(config=config)

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
            past_key, past_value = self._reorder_cache_from_bloom_to_falcon(layer_past, batch_size)
            past_length = past_key.shape[2]
            past_key_values.update(past_key, past_value, self.self_attention.layer_idx)

        cache_position = torch.arange(past_length, past_length + seq_length, device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)

        if alibi is None and self.config.alibi:
            ones_mask = torch.ones((batch_size, past_length + seq_length), device=hidden_states.device)
            alibi = build_alibi_tensor(ones_mask, self.num_heads, dtype=hidden_states.dtype)

        # With the eager implementation create_causal_mask already returns a 4D float mask, into
        # which we fold alibi (matching FalconModel.forward, which uses and_mask_function to force
        # 4D creation for the sdpa/flash paths -- that arg needs torch>=2.6, so we avoid it here).
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=None,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        if alibi is not None and causal_mask is not None and causal_mask.ndim == 4:
            min_dtype = torch.finfo(hidden_states.dtype).min
            if causal_mask.dtype == torch.bool:
                causal_mask = torch.where(
                    causal_mask, torch.tensor(0.0, device=causal_mask.device, dtype=hidden_states.dtype), min_dtype
                )
            alibi = alibi.reshape(batch_size, -1, *alibi.shape[1:])
            causal_mask = torch.masked_fill(
                alibi / math.sqrt(self.config.hidden_size // self.num_heads),
                causal_mask < -1,
                min_dtype,
            )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        output = super().forward(
            hidden_states,
            alibi,
            attention_mask=causal_mask,
            position_ids=position_ids,
            layer_past=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
        )
        hidden_states = output[0] if isinstance(output, tuple) else output

        if use_cache:
            present = past_key_values.layers[self.self_attention.layer_idx]
            return hidden_states, self._reorder_cache_from_falcon_to_bloom((present.keys, present.values), batch_size)
        return (hidden_states,)

    def _reorder_cache_from_bloom_to_falcon(
        self, key_value: Tuple[torch.Tensor], batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # value (BLOOM): [batch * kv_heads, seq_length, head_dim] -- derive kv_heads from the shape
        key_states, value_states = key_value
        seq_length, head_dim = value_states.shape[1], value_states.shape[2]
        kv_heads = value_states.shape[0] // batch_size
        key_states = key_states.permute(0, 2, 1)  # key (BLOOM): [batch * kv_heads, head_dim, seq_length]
        key_states = key_states.reshape(batch_size, kv_heads, seq_length, head_dim)
        value_states = value_states.reshape(batch_size, kv_heads, seq_length, head_dim)
        return key_states, value_states

    def _reorder_cache_from_falcon_to_bloom(
        self, key_value: Tuple[torch.Tensor], batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states, value_states = key_value  # both: [batch, kv_heads, seq_length, head_dim]
        kv_heads, seq_length, head_dim = key_states.shape[1], key_states.shape[2], key_states.shape[3]
        value_states = value_states.reshape(batch_size * kv_heads, seq_length, head_dim)
        key_states = key_states.reshape(batch_size * kv_heads, seq_length, head_dim)
        key_states = key_states.permute(0, 2, 1)  # [batch * kv_heads, head_dim, seq_length]
        return key_states, value_states
