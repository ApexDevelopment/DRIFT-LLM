"""
Gemma 2 intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py
See commit history for authorship.
"""
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer, Gemma2RotaryEmbedding

from petals.models._gemma_block import WrappedGemmaBlock


class WrappedGemma2Block(WrappedGemmaBlock, Gemma2DecoderLayer):
    """A Petals wrapper around a stock transformers ``Gemma2DecoderLayer``.

    See ``petals.models._gemma_block.WrappedGemmaBlock``: the alternating sliding/full attention
    pattern is honored via the block's true global index; logit softcapping and query pre-attention
    scaling live inside ``Gemma2Attention``.
    """

    rotary_class = Gemma2RotaryEmbedding
