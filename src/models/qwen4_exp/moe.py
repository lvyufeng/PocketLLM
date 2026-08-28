"""Qwen4-Exp routed-expert backends.

Two implementations share one interface:

* `InMemoryMoE` keeps `gate_up_proj`/`down_proj` as dense tensors on whatever
  device they were handed.  Used by tests and small models.
* `HostExpertMoE` reads expert rows straight out of the mmap'd safetensors on
  demand and runs the SwiGLU on the GPU, staging only the active experts.  This
  is what makes the 225 GiB expert set usable on 4x22 GiB cards: a decode step
  touches at most `top_k` experts per layer (6.6 MiB each in BF16).
"""

from __future__ import annotations

from typing import Protocol

import torch

from src.models.qwen4_exp.layers import swiglu_expert


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
) -> torch.Tensor:
    """Group tokens by expert, run each expert once, scatter-add the results.

    `expert_weights(expert_id)` returns `(gate_up, down)` on the compute device.
    Experts are visited in ascending id so the accumulation order is fixed and
    the result is run-to-run reproducible.
    """
    out = torch.zeros_like(hidden_states)
    flat_experts = indices.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[order]
    unique, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
    top_k = indices.shape[1]

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
        gate_up, down = expert_weights(int(expert_id))
        contrib = swiglu_expert(hidden_states[token_idx], gate_up, down)
        contrib = contrib * weights[token_idx, k_idx].unsqueeze(-1).to(contrib.dtype)
        out.index_add_(0, token_idx, contrib.to(out.dtype))
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
    """Experts stay in host RAM (mmap'd checkpoint); active ones are staged to GPU.

    The reader must expose `expert_rows(layer_idx, expert_id)` returning CPU
    tensors that alias the mapping.  A small LRU keeps recently used experts on
    device, which pays off during prefill where consecutive chunks reuse the same
    hot experts.
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
        self._cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._order: list[tuple[int, int]] = []
        self.stage_bytes = 0
        self.stage_calls = 0

    def _fetch(self, layer_idx: int, expert_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer_idx, expert_id)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        gate_up_cpu, down_cpu = self.reader.expert_rows(layer_idx, expert_id)
        gate_up = gate_up_cpu.to(self.device, dtype=self.dtype, non_blocking=True)
        down = down_cpu.to(self.device, dtype=self.dtype, non_blocking=True)
        self.stage_bytes += gate_up.numel() * gate_up.element_size() + down.numel() * down.element_size()
        self.stage_calls += 1
        if self.cache_capacity > 0:
            self._cache[key] = (gate_up, down)
            self._order.append(key)
            while len(self._order) > self.cache_capacity:
                evicted = self._order.pop(0)
                self._cache.pop(evicted, None)
        return gate_up, down

    def __call__(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return _dispatch_experts(
            hidden_states,
            indices,
            weights,
            self.num_experts,
            lambda e: self._fetch(layer_idx, e),
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
