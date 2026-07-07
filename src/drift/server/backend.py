from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from hivemind.moe.expert_uid import ExpertUID
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils import get_logger
from hivemind.utils.tensor_descr import BatchTensorDescriptor, TensorDescriptor
from transformers import PretrainedConfig

from drift.data_structures import InferenceMetadata
from drift.server.memory_cache import MemoryCache
from drift.server.task_pool import PrioritizedTaskPool
from drift.utils.kv_cache import StandardGQACache
from drift.utils.misc import get_num_attention_heads, get_size_in_bytes, is_dummy
from drift.utils.tensor_parallel import TensorParallel

logger = get_logger(__name__)


class TransformerBackend(ModuleBackend):
    """A wrapper for a transformer block that can process requests for forward, backward and inference"""

    _peft_module = None

    def __init__(
        self,
        *args,
        config: PretrainedConfig,
        memory_cache: MemoryCache,
        backend_dtype: torch.dtype,
        max_chunk_size_bytes: int,
        **kwargs,
    ):
        import drift.utils.peft as _peft_module

        self._peft_module = _peft_module

        super().__init__(*args, **kwargs)
        assert isinstance(self.module, TensorParallel)
        self.config = config
        self.memory_cache = memory_cache
        self.max_chunk_size_bytes = max_chunk_size_bytes

        for name, param in self.module.named_parameters():
            assert not param.requires_grad, f"Block parameters must not accumulate gradients, but {name} does"
        for name, buf in self.module.named_buffers():
            assert not buf.requires_grad, f"Block parameters must not accumulate gradients, but {name} does"

        max_batch_size = self.forward_pool.max_batch_size
        device = self.module.devices[self.module.output_device_index]
        self.inference_pool = PrioritizedTaskPool(
            self.inference_step, max_batch_size=max_batch_size, device=device, name=f"{self.name}_inference"
        )  # note: inference_pools may be merged later, see merge_inference_pools_inplace
        self.forward_pool = PrioritizedTaskPool(
            self.forward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_forward"
        )
        self.backward_pool = PrioritizedTaskPool(
            self.backward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_backward"
        )

        self.dtype = backend_dtype
        self.dtype_bytes = get_size_in_bytes(self.dtype)
        self.shard_num_heads = []
        for shard in self.module.module_shards:
            for submodule in shard.modules():
                if isinstance(submodule, config.attn_class):
                    self.shard_num_heads.append(get_num_attention_heads(submodule, config))
        assert len(self.shard_num_heads) == len(self.module.devices)
        assert sum(self.shard_num_heads) == config.num_attention_heads

        # The cache layout (descriptor shapes, prefix selection, write-back) is owned by a
        # pluggable strategy so that non-standard layouts (sliding window, MLA) can be added
        # without touching the backend. Models select one via `config.kv_cache_strategy`.
        self.cache_strategy = getattr(config, "kv_cache_strategy", StandardGQACache)(config)

        self.inference_schema = (
            (
                *self.args_schema,
                BatchTensorDescriptor((), dtype=self.dtype),
                BatchTensorDescriptor((), dtype=torch.int64),
            ),
            self.kwargs_schema,
        )

        self.cache_bytes_per_token: Dict[torch.device, int] = Counter()
        for descr in self.get_inference_cache_descriptors(batch_size=1, max_length=1):
            self.cache_bytes_per_token[descr.device] += descr.numel() * get_size_in_bytes(descr.dtype)

        if self.memory_cache.paged:
            self._configure_paged_pool()

    def _configure_paged_pool(self) -> None:
        """Register this block's slice of the shared paged pool (single tensor-parallel shard only)."""
        assert len(self.module.devices) == 1, "paged KV cache does not support tensor parallelism yet"
        key_descr, value_descr = self.get_inference_cache_descriptors(batch_size=1, max_length=1)
        device = key_descr.device
        page_bytes = self.cache_bytes_per_token[device] * self.memory_cache.page_size
        self.memory_cache.configure_paged_pool(
            num_pages=int(self.memory_cache.max_size_bytes // page_bytes),
            num_kv_heads=key_descr.size[1],  # key descr: [batch, kv_heads, k_head_dim, len]
            k_head_dim=key_descr.size[2],
            v_head_dim=value_descr.size[3],  # value descr: [batch, kv_heads, len, v_head_dim]
            dtype=self.dtype,
            device=device,
        )

    def get_inference_cache_descriptors(self, batch_size: int, max_length: int) -> Sequence[TensorDescriptor]:
        """Create tensor descriptors for attention cache tensors used during inference_step"""
        return self.cache_strategy.get_cache_descriptors(
            batch_size,
            max_length,
            dtype=self.dtype,
            devices=self.module.devices,
            shard_num_heads=self.shard_num_heads,
        )

    def forward(self, *inputs: Union[torch.Tensor, str]) -> Tuple[torch.Tensor, ...]:
        *inputs, active_adapter = inputs
        with self._peft_module.using_adapter(active_adapter):
            return super().forward(*inputs)

    def backward(self, *inputs: Union[torch.Tensor, str]) -> Tuple[torch.Tensor, ...]:
        *inputs, active_adapter = inputs
        with self._peft_module.using_adapter(active_adapter):
            return super().backward(*inputs)

    @torch.inference_mode()
    def inference_step(
        self,
        hidden_states: torch.Tensor,
        hypo_ids: torch.LongTensor,
        inference_info: InferenceMetadata,
    ) -> Tuple[torch.Tensor, ...]:
        assert hidden_states.ndim == 3, "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]"

        with self._peft_module.using_adapter(inference_info.active_adapter):
            if self.memory_cache.paged:
                return self._paged_inference_step(hidden_states, hypo_ids, inference_info)

            with self.memory_cache.use_cache(*inference_info.cache_handles) as cache_tensors:
                self._reorder_cache_inplace(cache_tensors, hypo_ids)
                max_chunk_length = self._estimate_max_chunk_length(hidden_states, inference_info)
                layer_past = self.cache_strategy.select_layer_past(
                    cache_tensors, inference_info.prefix_length, num_shards=len(self.module.module_shards)
                )
                output_hidden_states, new_kvs = self._forward_chunked(hidden_states, layer_past, max_chunk_length)
                self.cache_strategy.update_cache(cache_tensors, new_kvs, inference_info.prefix_length)
                return (output_hidden_states,)

    def _paged_inference_step(
        self, hidden_states: torch.Tensor, hypo_ids: torch.LongTensor, inference_info: InferenceMetadata
    ) -> Tuple[torch.Tensor, ...]:
        (slot_id,) = inference_info.cache_handles
        with self.memory_cache.use_paged_pool() as pool:
            if not is_dummy(hypo_ids):
                pool.reorder(slot_id, hypo_ids)
            max_chunk_length = self._estimate_max_chunk_length(hidden_states, inference_info)
            layer_past = pool.gather(slot_id, inference_info.prefix_length)
            output_hidden_states, new_kvs = self._forward_chunked(hidden_states, layer_past, max_chunk_length)
            pool.scatter(slot_id, new_kvs, inference_info.prefix_length)
            return (output_hidden_states,)

    def _forward_chunked(self, hidden_states: torch.Tensor, layer_past, max_chunk_length: int):
        # We chunk the inputs so that peak memory for long sequences fits into `autograd_memory`
        # reserved in `Server._choose_num_blocks()`. This saves us from OOMs if `max_chunk_size_bytes`
        # is at least 4-6x less than `autograd_memory`.
        seq_len = hidden_states.shape[1]
        output_hidden_states = torch.empty_like(hidden_states) if seq_len > max_chunk_length else None
        for offset in range(0, seq_len, max_chunk_length):
            hidden_states_chunk = hidden_states[:, offset : offset + max_chunk_length, :]
            output_hidden_states_chunk, new_kvs = self.module.forward(
                hidden_states_chunk, layer_past=layer_past, use_cache=True
            )
            if seq_len > max_chunk_length:
                output_hidden_states[:, offset : offset + max_chunk_length] = output_hidden_states_chunk
            else:
                output_hidden_states = output_hidden_states_chunk  # saves one memcopy
            layer_past = new_kvs
        return output_hidden_states, new_kvs

    def _estimate_max_chunk_length(self, hidden_states: torch.Tensor, inference_info: InferenceMetadata) -> int:
        # We assume that attention logit matrices are the main thing that consumes memory, given that
        # the model uses multi-query attention
        batch_size, seq_length, hidden_size = hidden_states.shape
        worst_case_length = inference_info.prefix_length + seq_length
        attn_bytes_per_token = max(self.shard_num_heads) * batch_size * self.dtype_bytes * worst_case_length
        return max(1, self.max_chunk_size_bytes // attn_bytes_per_token)

    def _reorder_cache_inplace(self, cache_tensors: torch.Tensor, hypo_ids: torch.Tensor):
        """If hypo_ids is specified, reorder elements of each cache tensor in-place by taking indices from hypo_ids"""
        if not is_dummy(hypo_ids):
            for cache_tensor in cache_tensors:
                cache_tensor[...] = cache_tensor[hypo_ids.to(cache_tensor.device)]  # in-place reorder cache by hypo ids

    def get_pools(self) -> Sequence[PrioritizedTaskPool]:
        return self.forward_pool, self.backward_pool, self.inference_pool

    def get_info(self) -> Dict[str, Any]:
        """Get module parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration."""
        return dict(super().get_info(), inference_schema=self.inference_schema)

    def shutdown(self):
        # Break the cyclic references, otherwise TransformerBackend may be not garbage-collected
        self.forward_pool = self.backward_pool = self.inference_pool = None

        # Explicitly free the GPU memory. This is not necessary at the time this code is written,
        # but may help to avoid future issues when the module is not garbage-collected for some reasons
        dummy = torch.tensor([])
        for p in self.module.parameters():
            p.data = dummy


def merge_inference_pools_inplace(backends: Dict[ExpertUID, TransformerBackend]):
    """Replace each backend's rpc_inference pools with a combined pool runs multiple blocks in one call"""
    assert len(backends) != 0 and all(isinstance(b, TransformerBackend) for b in backends.values())
    first_pool = next(iter(backends.values())).inference_pool
    merged_pool = PrioritizedTaskPool(
        _MergedInferenceStep(backends),
        max_batch_size=first_pool.max_batch_size,
        device=first_pool.device,
        name=f"merged_inference",
    )
    for backend in backends.values():
        assert not backend.inference_pool.is_alive()
        backend.inference_pool = merged_pool


class _MergedInferenceStep:
    def __init__(self, backends: Dict[ExpertUID, TransformerBackend]):
        self.backends = backends

    @torch.inference_mode()
    def __call__(
        self,
        hidden_states: torch.Tensor,
        hypo_ids: torch.LongTensor,
        inference_infos: Sequence[InferenceMetadata],
        *optional_prompts: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        assert len(inference_infos) == len(
            optional_prompts
        ), f"found {len(inference_infos)} blocks but {len(optional_prompts)} prompts"
        for inference_info, optional_prompt in zip(inference_infos, optional_prompts):
            if optional_prompt is not None:
                hidden_states[:, : optional_prompt.shape[1]] += optional_prompt
            (hidden_states,) = self.backends[inference_info.uid].inference_step(hidden_states, hypo_ids, inference_info)
        return (hidden_states,)
