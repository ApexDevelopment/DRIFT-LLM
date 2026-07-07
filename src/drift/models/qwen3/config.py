import os
from typing import Optional, Union

from hivemind.utils.logging import get_logger
from transformers.models.qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from drift.client.config import ClientConfig
from drift.client.lm_head import LMHeadConfig
from drift.client.ptune import PTuneConfig
from drift.models.qwen3.block import WrappedQwen3Block

logger = get_logger(__name__)


class DistributedQwen3Config(Qwen3Config, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = WrappedQwen3Block
    attn_class = Qwen3Attention
    block_prefix = "model.layers"

    @property
    def num_key_value_groups(self):
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path).split("/")[-1]  # Use only repo name to merge blocks across accounts
            dht_prefix = dht_prefix.replace(".", "-")
            logger.info(f"Using DHT prefix: {dht_prefix}")

        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        config.use_cache = (
            True  # use_cache=False leads to identical results but is slower and not supported by DRIFT-LLM
        )
        return result
