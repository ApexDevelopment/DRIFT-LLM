from petals.models.qwen3.block import WrappedQwen3Block
from petals.models.qwen3.config import DistributedQwen3Config
from petals.models.qwen3.model import (
    DistributedQwen3ForCausalLM,
    DistributedQwen3ForSequenceClassification,
    DistributedQwen3Model,
)
from petals.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedQwen3Config,
    model=DistributedQwen3Model,
    model_for_causal_lm=DistributedQwen3ForCausalLM,
    model_for_sequence_classification=DistributedQwen3ForSequenceClassification,
)
