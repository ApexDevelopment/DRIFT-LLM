"""
Self-contained tensor parallelism for Petals, vendored from the unmaintained ``tensor_parallel``
library (https://github.com/BlackSamorez/tensor_parallel) and modernized for current torch /
transformers.

Why vendor instead of depend: the upstream package is unmaintained, pins old internals
(private ``torch._utils`` device helpers, NCCL/`torch.distributed` fast paths) and only ever
shipped a *correct* head-parallel sharding config for BLOOM. Here we drop the parts Petals never
used (the ``tensor_parallel()`` factory, ``TensorParallelPreTrainedModel``, conv sharding, the
CUDA-stream apply path) and add real head-parallel configs for every architecture Petals serves
(see ``configs.py``). Petals only ever needs :class:`TensorParallel`, :class:`PerDeviceTensors`
and the per-architecture configs.
"""
from petals.utils.tensor_parallel.configs import get_bloom_config, get_tensor_parallel_config
from petals.utils.tensor_parallel.slicer import Config
from petals.utils.tensor_parallel.wrapper import PerDeviceTensors, TensorParallel

__all__ = [
	"TensorParallel",
	"PerDeviceTensors",
	"Config",
	"get_tensor_parallel_config",
	"get_bloom_config",
]
