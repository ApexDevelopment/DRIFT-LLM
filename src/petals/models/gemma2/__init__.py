from petals.models.gemma2.block import WrappedGemma2Block
from petals.models.gemma2.config import DistributedGemma2Config
from petals.models.gemma2.model import (
    DistributedGemma2ForCausalLM,
    DistributedGemma2ForSequenceClassification,
    DistributedGemma2Model,
)
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedGemma2Config,
    model=DistributedGemma2Model,
    model_for_causal_lm=DistributedGemma2ForCausalLM,
    model_for_sequence_classification=DistributedGemma2ForSequenceClassification,
)
