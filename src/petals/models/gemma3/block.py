"""
Gemma 3 intermediate layer
Based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py
See commit history for authorship.
"""
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer, Gemma3RotaryEmbedding

from petals.models._gemma_block import WrappedGemmaBlock


class WrappedGemma3Block(WrappedGemmaBlock, Gemma3DecoderLayer):
    """A Petals wrapper around a stock transformers ``Gemma3DecoderLayer`` (text).

    On top of the alternating sliding/full pattern (see ``WrappedGemmaBlock``), Gemma 3 uses a
    different rotary base for sliding vs full layers, so the block feeds its layer type into the
    shared rotary module.
    """

    rotary_class = Gemma3RotaryEmbedding
    rotary_takes_layer_type = True
