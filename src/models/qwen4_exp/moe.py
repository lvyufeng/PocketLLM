"""Qwen4-Exp routed-expert backends.

Two implementations share one interface:

* `InMemoryMoE` keeps `gate_up_proj`/`down_proj` as dense tensors on whatever
  device they were handed.  Used by tests and small models.
* `HostExpertMoE` reads expert rows from the host through
  `reader.expert_rows()` and runs the SwiGLU on the GPU, staging only the active
  experts.  This is what makes the 225 GiB expert set usable on 4x22 GiB cards: a
  decode step touches at most `top_k` experts per layer (6.6 MiB each in BF16).
  The reader normally serves those rows from this rank's resident host shard (see
  `weights.HostExpertShard`); the mmap is only the fallback.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

import torch

from src.kernels.cuda_loader import load_cuda_kernel
from src.models.qwen4_exp.layers import swiglu_expert


@dataclass
class _PendingExpert:
    gate_up: torch.Tensor
    down: torch.Tensor
    ready: torch.cuda.Event | None


class MoEBackend(Protocol):
    def __call__(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """hidden_states: (tokens, hidden); indices/weights: (tokens, top_k)."""
        ...


def _dispatch_experts(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
    expert_weights,
    stats: dict | None = None,
    stager=None,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Group tokens by expert, run each expert once, scatter-add the results.

    `expert_weights(expert_id)` returns `(gate_up, down)` on the compute device.
    Experts are visited in ascending id so the accumulation order is fixed and
    the result is run-to-run reproducible.

    When `stats` is given, per-phase wall time is accumulated into it.  The
    timing path synchronizes and is meant for profiling runs only.
    """
    out = torch.zeros_like(hidden_states)
    flat_experts = indices.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[order]
    unique, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
    top_k = indices.shape[1]

    if stats is None:
        offset = 0
        unique_list = unique.tolist()
        counts_list = counts.tolist()
        expert_ids = [int(expert_id) for expert_id in unique_list if expert_id < num_experts]
        prefetched = stager(expert_ids) if stager is not None else None
        for expert_id, count in zip(unique_list, counts_list):
            slot = order[offset : offset + count]
            offset += count
            if expert_id >= num_experts:
                continue
            token_idx = slot // top_k
            k_idx = slot % top_k
            gate_up, down = next(prefetched) if prefetched is not None else expert_weights(int(expert_id))
            expert_input = hidden_states[token_idx]
            if compute_dtype is not None:
                expert_input = expert_input.to(compute_dtype)
                gate_up = gate_up.to(compute_dtype)
                down = down.to(compute_dtype)
            contrib = swiglu_expert(expert_input, gate_up, down)
            contrib = contrib * weights[token_idx, k_idx].unsqueeze(-1).to(contrib.dtype)
            out.index_add_(0, token_idx, contrib.to(out.dtype))
        return out

    import time

    dev = hidden_states.device
    sync = torch.cuda.synchronize if dev.type == "cuda" else lambda *a: None

    offset = 0
    unique_list = unique.tolist()
    counts_list = counts.tolist()
    for expert_id, count in zip(unique_list, counts_list):
        slot = order[offset : offset + count]
        offset += count
        if expert_id >= num_experts:
            continue
        token_idx = slot // top_k
        k_idx = slot % top_k

        sync(dev)
        t0 = time.perf_counter()
        gate_up, down = expert_weights(int(expert_id))
        sync(dev)
        t1 = time.perf_counter()
        expert_input = hidden_states[token_idx]
        if compute_dtype is not None:
            expert_input = expert_input.to(compute_dtype)
            gate_up = gate_up.to(compute_dtype)
            down = down.to(compute_dtype)
        contrib = swiglu_expert(expert_input, gate_up, down)
        contrib = contrib * weights[token_idx, k_idx].unsqueeze(-1).to(contrib.dtype)
        out.index_add_(0, token_idx, contrib.to(out.dtype))
        sync(dev)
        t2 = time.perf_counter()

        stats["stage_s"] = stats.get("stage_s", 0.0) + (t1 - t0)
        stats["compute_s"] = stats.get("compute_s", 0.0) + (t2 - t1)
        stats["experts"] = stats.get("experts", 0) + 1
    stats["calls"] = stats.get("calls", 0) + 1
    return out


class InMemoryMoE:
    """All experts resident as dense per-layer tensors."""

    def __init__(self, num_experts: int, gate_up: dict[int, torch.Tensor], down: dict[int, torch.Tensor]) -> None:
        self.num_experts = num_experts
        self.gate_up = gate_up  # layer_idx -> (num_experts, 2*inter, hidden)
        self.down = down  # layer_idx -> (num_experts, hidden, inter)

    def __call__(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        gu = self.gate_up[layer_idx]
        dn = self.down[layer_idx]
        return _dispatch_experts(
            hidden_states,
            indices,
            weights,
            self.num_experts,
            lambda e: (gu[e], dn[e]),
        )


class HostExpertMoE:
    """Experts stay in host RAM; active ones are staged to GPU per step.

    The reader must expose `expert_rows(layer_idx, expert_id)` returning host
    tensors — resident shard rows when preloaded, mmap views otherwise.  A small
    LRU keeps recently used experts on device, which pays off during prefill where
    consecutive chunks reuse the same hot experts.
    """

    def __init__(
        self,
        reader,
        num_experts: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        cache_capacity: int = 0,
    ) -> None:
        self.reader = reader
        self.num_experts = num_experts
        self.device = device
        self.dtype = dtype
        self.cache_capacity = cache_capacity
        self._cache: OrderedDict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self.stage_bytes = 0
        self.stage_calls = 0
        self.cache_hits = 0
        # Set to a dict to collect per-phase timings (profiling only).
        self.phase_stats: dict | None = None
        self.prefetch_enabled = os.environ.get("DSV4_QWEN4_MOE_PREFETCH", "1").lower() in {
            "1", "true", "yes"
        }
        self._copy_stream: torch.cuda.Stream | None = None
        self._cuda_ext = None
        self.grouped_enabled = os.environ.get("DSV4_QWEN4_MOE_GROUPED", "0").lower() in {
            "1", "true", "yes"
        }
        self.grouped_min_tokens = int(os.environ.get("DSV4_QWEN4_MOE_GROUPED_MIN_TOKENS", "128"))
        self.grouped_max_padding = float(os.environ.get("DSV4_QWEN4_MOE_GROUPED_MAX_PADDING", "2.5"))
        self.grouped_min_free_gib = float(
            os.environ.get("DSV4_QWEN4_MOE_GROUPED_MIN_FREE_GIB", "2.0")
        )
        self.fp16_compute_min_tokens = max(
            1,
            int(os.environ.get("DSV4_QWEN4_MOE_FP16_MIN_TOKENS", "128")),
        )

    def _cache_put(self, key: tuple[int, int], pair: tuple[torch.Tensor, torch.Tensor]) -> None:
        if self.cache_capacity <= 0:
            return
        self._cache[key] = pair
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)

    def _cache_get(self, key: tuple[int, int]):
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            self.cache_hits += 1
        return hit

    def _fetch(self, layer_idx: int, expert_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer_idx, expert_id)
        hit = self._cache_get(key)
        if hit is not None:
            return hit
        gate_up_cpu, down_cpu = self.reader.expert_rows(layer_idx, expert_id)
        gate_up = gate_up_cpu.to(self.device, dtype=self.dtype, non_blocking=True)
        down = down_cpu.to(self.device, dtype=self.dtype, non_blocking=True)
        self.stage_bytes += gate_up.numel() * gate_up.element_size() + down.numel() * down.element_size()
        self.stage_calls += 1
        self._cache_put(key, (gate_up, down))
        return gate_up, down

    def _grouped_bf16(self, layer_idx: int, hidden_states, indices, weights):
        """Run a dense-local-expert BF16 grouped prefill, or return None."""
        if hidden_states.shape[0] < self.grouped_min_tokens:
            return None
        shard = getattr(self.reader, "expert_shard", None)
        if shard is None or not shard.has_layer(layer_idx):
            return None
        if self.dtype is not torch.bfloat16:
            return None
        if shard.world_size <= 1:
            # A single-rank shard contains all experts; copying a full layer is
            # unnecessarily large, so keep the per-expert path there.
            return None
        if self._cuda_ext is None:
            self._cuda_ext = load_cuda_kernel()
        if self._cuda_ext is None or not hasattr(
            self._cuda_ext, "qwen4_exp_moe_prefill_bf16_forward"
        ):
            return None

        gate_up_cpu, down_cpu = shard.layer_tensors(layer_idx)
        layer_bytes = (
            gate_up_cpu.numel() * gate_up_cpu.element_size()
            + down_cpu.numel() * down_cpu.element_size()
        )
        try:
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
        except RuntimeError:
            return None
        reserve_bytes = max(0, int(self.grouped_min_free_gib * (1 << 30)))
        if free_bytes < layer_bytes + reserve_bytes:
            return None

        flat = indices.reshape(-1)
        top_k = indices.shape[1]
        order = torch.argsort(flat, stable=True)
        sorted_ids = flat[order]
        unique, counts = torch.unique_consecutive(sorted_ids, return_counts=True)

        unique_list = [int(e) for e in unique.tolist()]
        counts_list = [int(c) for c in counts.tolist()]
        local_experts = shard.num_local_experts
        offset = 0
        route_tokens = []
        route_weights = []
        counts_by_local = [0] * local_experts
        for expert_id, count in zip(unique_list, counts_list):
            begin, offset = offset, offset + count
            if expert_id >= self.num_experts or not shard.owns(expert_id):
                continue
            local_id = shard.local_index(expert_id)
            if local_id < 0 or local_id >= local_experts:
                return None
            slots = order[begin:offset]
            route_tokens.append((slots // top_k).to(torch.int64))
            route_weights.append(weights.reshape(-1)[slots].to(torch.float32))
            counts_by_local[local_id] = count
        if not route_tokens:
            # This is the expected result for a ShardedMoE rank with no local
            # routes.  The caller's all-reduce will combine the zero partial.
            return torch.zeros_like(hidden_states)

        route_count = sum(counts_by_local)
        max_count = max(counts_by_local)
        padding_ratio = shard.num_local_experts * max_count / max(1, route_count)
        if padding_ratio > self.grouped_max_padding:
            return None

        route_tokens = torch.cat(route_tokens).contiguous()
        route_weights = torch.cat(route_weights).contiguous()
        seg = [0]
        for count in counts_by_local:
            seg.append(seg[-1] + count)
        seg_starts = torch.tensor(seg, device=hidden_states.device, dtype=torch.int32)

        # Keep the local shard's expert-id layout intact.  This replaces many
        # small H2D copies with two contiguous layer copies; zero-route experts
        # are represented by empty segments and never contribute to the output.
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(self._copy_stream):
            gate_up = gate_up_cpu.to(self.device, dtype=self.dtype, non_blocking=True)
            down = down_cpu.to(self.device, dtype=self.dtype, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self._copy_stream)
        current.wait_event(ready)
        gate_up.record_stream(current)
        down.record_stream(current)
        self.stage_bytes += gate_up.numel() * gate_up.element_size() + down.numel() * down.element_size()
        self.stage_calls += 1
        return self._cuda_ext.qwen4_exp_moe_prefill_bf16_forward(
            hidden_states.contiguous(),
            route_tokens,
            route_weights,
            seg_starts,
            gate_up,
            down,
            0.0,
        )

    def _prefetch(self, layer_idx: int, expert_ids: list[int]):
        """Stage one layer's active experts on a copy stream in expert-id order."""
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        pending: list[_PendingExpert] = []
        with torch.cuda.stream(self._copy_stream):
            for expert_id in expert_ids:
                key = (layer_idx, expert_id)
                hit = self._cache_get(key)
                if hit is not None:
                    pending.append(_PendingExpert(hit[0], hit[1], None))
                    continue
                gate_up_cpu, down_cpu = self.reader.expert_rows(layer_idx, expert_id)
                gate_up = gate_up_cpu.to(self.device, dtype=self.dtype, non_blocking=True)
                down = down_cpu.to(self.device, dtype=self.dtype, non_blocking=True)
                event = torch.cuda.Event()
                event.record(self._copy_stream)
                pending.append(_PendingExpert(gate_up, down, event))
                self.stage_bytes += gate_up.numel() * gate_up.element_size() + down.numel() * down.element_size()
                self.stage_calls += 1
                self._cache_put(key, (gate_up, down))

        def consume():
            for item in pending:
                if item.ready is not None:
                    current.wait_event(item.ready)
                item.gate_up.record_stream(current)
                item.down.record_stream(current)
                yield item.gate_up, item.down

        return consume()

    def __call__(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.grouped_enabled
            and self.phase_stats is None
            and hidden_states.device.type == "cuda"
        ):
            grouped = self._grouped_bf16(layer_idx, hidden_states, indices, weights)
            if grouped is not None:
                return grouped

        use_prefetch = (
            self.prefetch_enabled
            and self.phase_stats is None
            and hidden_states.device.type == "cuda"
        )
        fp16_compute_enabled = os.environ.get(
            "DSV4_QWEN4_MOE_FP16_COMPUTE", "0"
        ).lower() in {"1", "true", "yes"}
        compute_dtype = (
            torch.float16
            if fp16_compute_enabled
            and hidden_states.device.type == "cuda"
            and hidden_states.dtype is torch.bfloat16
            and hidden_states.shape[0] >= self.fp16_compute_min_tokens
            else None
        )
        return _dispatch_experts(
            hidden_states,
            indices,
            weights,
            self.num_experts,
            lambda e: self._fetch(layer_idx, e),
            stats=None if use_prefetch else self.phase_stats,
            stager=(lambda expert_ids: self._prefetch(layer_idx, expert_ids)) if use_prefetch else None,
            compute_dtype=compute_dtype,
        )


class ShardedMoE:
    """Wraps a backend so each rank only runs the experts it owns.

    Experts are partitioned round-robin by id across `world_size` ranks; the
    per-layer all-reduce that already follows the MLP block sums the partial
    results, so no extra communication is needed.
    """

    def __init__(self, inner: MoEBackend, rank: int, world_size: int) -> None:
        self.inner = inner
        self.rank = rank
        self.world_size = world_size

    def __call__(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.world_size == 1:
            return self.inner(layer_idx, hidden_states, indices, weights)
        # Mask out foreign experts by pointing them at an out-of-range id, which
        # `_dispatch_experts` skips.
        mine = (indices % self.world_size) == self.rank
        local = torch.where(mine, indices, torch.full_like(indices, self.inner.num_experts))
        return self.inner(layer_idx, hidden_states, local, weights)
