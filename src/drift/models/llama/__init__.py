from drift.models.llama.block import WrappedLlamaBlock
from drift.models.llama.config import DistributedLlamaConfig
from drift.models.llama.model import (
    DistributedLlamaForCausalLM,
    DistributedLlamaForSequenceClassification,
    DistributedLlamaModel,
)
from drift.models.llama.speculative_model import DistributedLlamaForSpeculativeGeneration
from drift.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedLlamaConfig,
    model=DistributedLlamaModel,
    model_for_causal_lm=DistributedLlamaForCausalLM,
    model_for_speculative=DistributedLlamaForSpeculativeGeneration,
    model_for_sequence_classification=DistributedLlamaForSequenceClassification,
)
