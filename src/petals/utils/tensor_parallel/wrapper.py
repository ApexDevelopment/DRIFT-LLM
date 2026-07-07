"""
The main :class:`TensorParallel` module wrapper.

Vendored (and trimmed) from the ``tensor_parallel`` library. The CUDA-stream ``parallel_apply``
fast path and the private ``torch._utils`` device-index helpers were dropped; shards always run
via the thread-based ``parallel_apply_simple`` (correct on both CPU and CUDA). Petals always
passes explicit device ids, so device auto-discovery is unnecessary.
"""
from __future__ import annotations

import logging
import threading
from contextlib import nullcontext
from typing import Any, Optional, Sequence, Union

import torch
from torch import nn
from torch._utils import ExceptionWrapper

from petals.utils.tensor_parallel.cross_device_ops import broadcast_coalesced
from petals.utils.tensor_parallel.slicer import Config
from petals.utils.tensor_parallel.utils import nested_flatten, nested_pack

logger = logging.getLogger(__file__)


class TensorParallel(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        device_ids: Optional[Sequence[torch.device]] = None,
        output_device: Optional[torch.device] = None,
        output_device_index: Optional[int] = None,
        config: Optional[Config] = None,
        delay_init: bool = False,
    ):
        super().__init__()
        original_params = sum(p.numel() for p in module.parameters())
        assert output_device is None or output_device_index is None, "please specify either device or index, not both"
        device_ids = check_device_ids(device_ids)

        if output_device is not None:
            output_device = canonicalize_device(output_device)
            assert output_device in device_ids, f"Output device {output_device} not in {device_ids}"
            output_device_index = device_ids.index(output_device)
            del output_device
        elif output_device_index is None:
            output_device_index = 0

        self.module_shards = nn.ModuleList()
        self.devices = device_ids
        self.output_device_index = output_device_index
        self.all_cuda = all(device.type == "cuda" for device in self.devices)
        self.need_delayed_init = delay_init
        world_size = len(self.devices)

        if world_size <= 1:
            self.module_shards.append(module)
            if world_size == 1 and not delay_init:
                self.module_shards[0].to(device_ids[0])
            return

        if config is None:
            config = Config.get_default_config(module, self.devices)
            logger.info("Using automatic config: sharding individual linear/embedding layers")

        # create a copy of the config with collective op instances (AllReduce/AllGather) baked in
        config_with_ops = config.create_collective_ops(self.devices)

        for rank, device in enumerate(self.devices):
            shard_device = torch.device("cpu") if delay_init else device
            self.module_shards.append(
                config.make_shard(module, shard_device, config_with_ops, rank=rank, world_size=world_size)
            )

        # self-diagnostics: check if the model was sharded properly
        params_per_shard = [sum(p.numel() for p in shard.parameters()) for shard in self.module_shards]
        assert sum(params_per_shard) >= original_params, "Internal assert failed: lost some parameters during sharding"
        self.param_fractions = tuple(params_i / original_params for params_i in params_per_shard)
        inefficiency_rate = (sum(self.param_fractions) - 1) / len(device_ids)  # extra params rate per device
        log_level = logging.DEBUG if inefficiency_rate < 0.1 else logging.WARNING
        logger.log(
            log_level,
            f"Inefficiency warning: model has {original_params} params but shards have {params_per_shard} params. "
            f"This means that each device uses {inefficiency_rate * 100:.3f}% extra memory for parameters",
        )

        # more self-diagnostics: make sure the model was not cast .to one device
        self._sanity_check_params = nn.ParameterList(
            [nn.Parameter(torch.empty(0, device=device), requires_grad=False) for device in self.devices]
        )

    def forward(self, *args, **kwargs):
        if self.need_delayed_init:
            for shard, device in zip(self.module_shards, self.devices):
                shard.to(device)
            self.need_delayed_init = False

        if len(self.module_shards) <= 1:
            return [self.module_shards[0](*args, **kwargs)][self.output_device_index]

        if not all(p.device == d for p, d in zip(self._sanity_check_params, self.devices)):
            raise ValueError(
                "Model parameters were moved to incorrect devices; did you call model.cuda() or model.to(device)? "
                "If so, please avoid doing that."
            )
        args_and_kwargs = (args, kwargs)
        flat_tensors = [obj for obj in nested_flatten(args_and_kwargs) if isinstance(obj, torch.Tensor)]
        flat_tensors_replicated = broadcast_coalesced(flat_tensors, self.devices, all_cuda=self.all_cuda)
        next_tensor_index = 0
        args_and_kwargs_replicated = [list() for _ in self.devices]
        for obj in nested_flatten(args_and_kwargs):
            if isinstance(obj, torch.Tensor):
                for idx in range(len(self.module_shards)):
                    args_and_kwargs_replicated[idx].append(flat_tensors_replicated[idx][next_tensor_index])
                next_tensor_index += 1
            else:
                for idx in range(len(self.module_shards)):
                    args_and_kwargs_replicated[idx].append(obj)
        for idx in range(len(self.module_shards)):
            args_and_kwargs_replicated[idx] = nested_pack(args_and_kwargs_replicated[idx], args_and_kwargs)
        inputs, kwargs_tup = zip(*args_and_kwargs_replicated)
        return parallel_apply_simple(self.module_shards, inputs, kwargs_tup, self.devices)[self.output_device_index]


def parallel_apply_simple(
    modules: Sequence[nn.Module],
    inputs: Sequence[Sequence[torch.Tensor]],
    kwargs_tup: Optional[Any],
    devices: Sequence[torch.device],
) -> Sequence[Sequence[torch.Tensor]]:
    """Run each shard in its own thread (no CUDA streams); correct on both CPU and CUDA."""
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, kwargs, device):
        torch.set_grad_enabled(grad_enabled)
        try:
            device_ctx = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
            with device_ctx:
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(where=f"in replica {i} on device {device}")

    if len(modules) > 1:
        threads = [
            threading.Thread(target=_worker, args=(i, module, input, kwargs, torch.device(device)))
            for i, (module, input, kwargs, device) in enumerate(zip(modules, inputs, kwargs_tup, devices))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], torch.device(devices[0]))

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs


def canonicalize_device(device: Union[torch.device, str]) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device(device.type, index=0)
    return device


def check_device_ids(device_ids: Optional[Sequence[torch.device]]) -> Sequence[torch.device]:
    if device_ids is None:
        device_ids = (
            [torch.device("cuda", i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
        )
    return tuple(map(canonicalize_device, device_ids))


class PerDeviceTensors:
    """Tensors located on different devices that will *not* be broadcast when passed to TensorParallel.forward."""

    def __init__(self, *tensors: torch.Tensor):
        # note: this is not broadcast because broadcast_coalesced does not broadcast class properties
        self.tensors = tuple(tensors)

    def __getitem__(self, i: int):
        return self.tensors[i]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.tensors})"
