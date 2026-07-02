from petals.models.mistral.block import WrappedMistralBlock
from petals.models.mistral.config import DistributedMistralConfig
from petals.models.mistral.model import (
    DistributedMistralForCausalLM,
    DistributedMistralForSequenceClassification,
    DistributedMistralModel,
)
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedMistralConfig,
    model=DistributedMistralModel,
    model_for_causal_lm=DistributedMistralForCausalLM,
    model_for_sequence_classification=DistributedMistralForSequenceClassification,
)
