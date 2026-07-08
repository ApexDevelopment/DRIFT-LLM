from drift.models.gemma4.block import WrappedGemma4Block
from drift.models.gemma4.config import DistributedGemma4Config
from drift.models.gemma4.model import (
    DistributedGemma4ForCausalLM,
    DistributedGemma4ForSequenceClassification,
    DistributedGemma4Model,
)
from drift.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedGemma4Config,
    model=DistributedGemma4Model,
    model_for_causal_lm=DistributedGemma4ForCausalLM,
    model_for_sequence_classification=DistributedGemma4ForSequenceClassification,
)
