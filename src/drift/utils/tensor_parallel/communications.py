"""
Thread-parallel collective operations for use in :class:`TensorParallel`.

Vendored (and trimmed) from the ``tensor_parallel`` library. Each collective coordinates the
per-shard worker threads spawned by ``parallel_apply_simple`` via barriers/events, so it works
identically on CPU and CUDA without touching ``torch.distributed`` or NCCL.
"""
from __future__ import annotations

import threading
from typing import Any, List, Optional

import torch

from drift.utils.tensor_parallel import cross_device_ops


class CollectiveOperationBase:
    def __call__(self, x: torch.Tensor, rank: int):
        raise NotImplementedError()


class CollectiveOperation(CollectiveOperationBase):
    def __init__(self, world_size: int, func: callable, authoritative_rank: int = 0):
        """
        Apply a user-defined collective function in a way that is compatible with TensorParallel.

        :param func: ``function(input0, input1, ..., input_worldsize) -> (out0, ..., out_worldsize)``
        """
        self.world_size = world_size
        self.func = func
        self.authoritative_rank = authoritative_rank
        self.rank_inputs: List[Any] = [None for _ in range(world_size)]
        self.rank_outputs: List[Any] = [None for _ in range(world_size)]
        self.barrier = threading.Barrier(world_size)

    def __call__(self, x: torch.Tensor, rank: int):
        try:
            self.rank_inputs[rank] = x
            self.barrier.wait()
            if rank == self.authoritative_rank:
                try:
                    result = self.func(*self.rank_inputs)
                    for i in range(self.world_size):
                        self.rank_outputs[i] = (result[i], None)
                except Exception as e:
                    for i in range(self.world_size):
                        self.rank_outputs[i] = (None, e)
            self.barrier.wait()
            result, exception = self.rank_outputs[rank]
            if exception:
                raise exception
            return result
        finally:
            self.rank_inputs[rank] = self.rank_outputs[rank] = None


class AllReduce(CollectiveOperationBase):
    def __init__(
        self,
        world_size: int,
        reduce_op: callable = cross_device_ops.reduce_add,
        gather_op: callable = cross_device_ops.gather,
    ):
        self.scatter_reduce = ScatterReduce(world_size, reduce_op)
        self.all_gather = AllGather(world_size, gather_op, barrier=False)
        # note: AllGather does not need a barrier here because scatter_reduce's ready event serves as one

    def __call__(self, x: torch.Tensor, rank: int):
        reduced_part = self.scatter_reduce(x, rank)
        return self.all_gather(reduced_part, rank).view_as(x)


class ScatterReduce(CollectiveOperationBase):
    def __init__(self, world_size: int, reduce_op: callable = cross_device_ops.reduce_add):
        self.world_size = world_size
        self.tensor_parts = [[] for _ in range(world_size)]
        self.parts_ready = [threading.Event() for _ in range(world_size)]
        self.reduce_op = reduce_op

    def __call__(self, x: torch.Tensor, rank: int):
        try:
            for i, part in enumerate(x.flatten().tensor_split(self.world_size)):
                self.tensor_parts[i].append(part)  # append is thread-safe. thanks, GIL!
                if len(self.tensor_parts[i]) == self.world_size:
                    self.parts_ready[i].set()  # can be called more than once; we don't care

            self.parts_ready[rank].wait()
            reduced_part = self.reduce_op(self.tensor_parts[rank], x.device)
            return reduced_part
        finally:
            # prepare for next forward; each rank clears its own data
            self.tensor_parts[rank].clear()
            self.parts_ready[rank].clear()


class AllGather(CollectiveOperationBase):
    def __init__(self, world_size: int, gather_op: callable = cross_device_ops.gather, barrier: bool = True):
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size) if barrier else None
        self.parts: List[Optional[torch.Tensor]] = [None for _ in range(world_size)]
        self.ranks_updated = []
        self.parts_ready = threading.Event()
        self.gather_op = gather_op

    def __call__(self, x: torch.Tensor, rank: int):
        if self.barrier is not None:
            self.barrier.wait()  # if this code is run multiple times in quick succession,
        # this event will wait for the previous call to finish before starting a new one
        parts, ranks_updated, parts_ready = self.parts, self.ranks_updated, self.parts_ready
        # ^-- note: we copy properties to locals so that the "finally" clause is thread-safe
        try:
            parts[rank] = x  # no race b/c each rank writes to a separate location
            ranks_updated.append(rank)  # append is thread-safe. thanks, GIL!
            if len(ranks_updated) == self.world_size:
                parts_ready.set()  # can be called more than once; we don't care
            parts_ready.wait()
            # note: for one of the parts with r == rank, part.to(device) is a no-op
            return self.gather_op(parts, x.device)
        finally:
            if ranks_updated[-1] == rank:
                self.parts = [None for _ in range(self.world_size)]
                self.ranks_updated = []
                self.parts_ready = threading.Event()
            # note: we can safely update these properties because all ranks have
            # copied self.parts_* to locals before passing parts_ready.wait
