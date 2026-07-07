from drift.models.mistral.block import WrappedMistralBlock
from drift.models.mistral.config import DistributedMistralConfig
from drift.models.mistral.model import (
    DistributedMistralForCausalLM,
    DistributedMistralForSequenceClassification,
    DistributedMistralModel,
)
from drift.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedMistralConfig,
    model=DistributedMistralModel,
    model_for_causal_lm=DistributedMistralForCausalLM,
    model_for_sequence_classification=DistributedMistralForSequenceClassification,
)
