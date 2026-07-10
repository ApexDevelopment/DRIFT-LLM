from drift.models.gemma4_unified.block import WrappedGemma4UnifiedBlock
from drift.models.gemma4_unified.config import DistributedGemma4UnifiedConfig
from drift.models.gemma4_unified.model import (
    DistributedGemma4UnifiedForCausalLM,
    DistributedGemma4UnifiedForSequenceClassification,
    DistributedGemma4UnifiedModel,
)
from drift.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedGemma4UnifiedConfig,
    model=DistributedGemma4UnifiedModel,
    model_for_causal_lm=DistributedGemma4UnifiedForCausalLM,
    model_for_sequence_classification=DistributedGemma4UnifiedForSequenceClassification,
)
