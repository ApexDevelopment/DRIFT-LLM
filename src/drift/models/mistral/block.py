"""
Mistral intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py
See commit history for authorship.
"""
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralRotaryEmbedding

from drift.models._gqa_block import WrappedGQABlock


class WrappedMistralBlock(WrappedGQABlock, MistralDecoderLayer):
    """A DRIFT-LLM wrapper around a stock transformers ``MistralDecoderLayer`` (GQA + sliding window).

    See ``drift.models._gqa_block.WrappedGQABlock`` for the BLOOM-layout KV bridging rationale;
    the sliding-window mask is applied there whenever ``config.sliding_window`` is set.
    """

    rotary_class = MistralRotaryEmbedding
