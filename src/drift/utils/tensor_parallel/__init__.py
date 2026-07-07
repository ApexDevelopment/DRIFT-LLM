"""
Self-contained tensor parallelism for DRIFT-LLM, vendored from the unmaintained ``tensor_parallel``
library (https://github.com/BlackSamorez/tensor_parallel) and modernized for current torch /
transformers.

Why vendor instead of depend: the upstream package is unmaintained, pins old internals
(private ``torch._utils`` device helpers, NCCL/`torch.distributed` fast paths) and only ever
shipped a *correct* head-parallel sharding config for BLOOM. Here we drop the parts DRIFT-LLM never
used (the ``tensor_parallel()`` factory, ``TensorParallelPreTrainedModel``, conv sharding, the
CUDA-stream apply path) and add real head-parallel configs for every architecture DRIFT-LLM serves
(see ``configs.py``). DRIFT-LLM only ever needs :class:`TensorParallel`, :class:`PerDeviceTensors`
and the per-architecture configs.
"""
from drift.utils.tensor_parallel.configs import get_bloom_config, get_tensor_parallel_config
from drift.utils.tensor_parallel.slicer import Config
from drift.utils.tensor_parallel.wrapper import PerDeviceTensors, TensorParallel

__all__ = [
    "TensorParallel",
    "PerDeviceTensors",
    "Config",
    "get_tensor_parallel_config",
    "get_bloom_config",
]
