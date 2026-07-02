from petals.models.qwen2.block import WrappedQwen2Block
from petals.models.qwen2.config import DistributedQwen2Config
from petals.models.qwen2.model import (
    DistributedQwen2ForCausalLM,
    DistributedQwen2ForSequenceClassification,
    DistributedQwen2Model,
)
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedQwen2Config,
    model=DistributedQwen2Model,
    model_for_causal_lm=DistributedQwen2ForCausalLM,
    model_for_sequence_classification=DistributedQwen2ForSequenceClassification,
)
