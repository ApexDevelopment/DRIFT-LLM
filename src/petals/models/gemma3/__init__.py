from petals.models.gemma3.block import WrappedGemma3Block
from petals.models.gemma3.config import DistributedGemma3Config
from petals.models.gemma3.model import (
    DistributedGemma3ForCausalLM,
    DistributedGemma3ForSequenceClassification,
    DistributedGemma3Model,
)
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedGemma3Config,
    model=DistributedGemma3Model,
    model_for_causal_lm=DistributedGemma3ForCausalLM,
    model_for_sequence_classification=DistributedGemma3ForSequenceClassification,
)
