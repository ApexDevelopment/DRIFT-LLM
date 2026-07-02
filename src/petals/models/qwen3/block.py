"""
Qwen3 intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
See commit history for authorship.
"""
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

from petals.models._gqa_block import WrappedGQABlock


class WrappedQwen3Block(WrappedGQABlock, Qwen3DecoderLayer):
    """A Petals wrapper around a stock transformers ``Qwen3DecoderLayer`` (GQA + per-head QK-norm).

    The QK RMSNorms live inside ``Qwen3Attention`` and run as part of the stock forward, so the
    BLOOM-layout KV bridging in ``WrappedGQABlock`` needs no Qwen3-specific handling.
    """

    rotary_class = Qwen3RotaryEmbedding
