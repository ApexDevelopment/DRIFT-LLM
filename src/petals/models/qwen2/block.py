"""
Qwen2 intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
See commit history for authorship.
"""
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

from petals.models._gqa_block import WrappedGQABlock


class WrappedQwen2Block(WrappedGQABlock, Qwen2DecoderLayer):
    """A Petals wrapper around a stock transformers ``Qwen2DecoderLayer`` (see ``WrappedGQABlock``)."""

    rotary_class = Qwen2RotaryEmbedding
