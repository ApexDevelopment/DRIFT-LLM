"""
LLaMA intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
See commit history for authorship.
"""
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

from drift.models._gqa_block import WrappedGQABlock


class WrappedLlamaBlock(WrappedGQABlock, LlamaDecoderLayer):
    """A DRIFT-LLM wrapper around a stock transformers ``LlamaDecoderLayer`` (see ``WrappedGQABlock``)."""

    rotary_class = LlamaRotaryEmbedding
