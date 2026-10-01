"""The KV cache: how much of it a model needs, and where it lives.

The cache is the one part of a decode loop that is *not* a pure function of the
weights.  It grows with context, it is written one row at a time, and on the
capacity it needs.  So a model does not carry it as a weight: it declares a
:class:`CachePlan`, the loader sizes an allocation from it, and the engine binds
that buffer to the cache values the graph reads.

**Why capacity is a parameter, not a constant.**  A phone and a card run the same
model at very different contexts, and a context limit is a *deployment* choice --
the same checkpoint should serve a 4K chat and a 32K summarisation without being
rebuilt.  So the plan takes the sequence length and returns descriptors; nothing
is allocated here and no device is named.

**Why there is no paging in this layer.**  A paged cache is a backend's
implementation of ``cache_append`` and ``attention``, not a different plan.  The
plan describes the logical shape -- ``(capacity, kv_heads, d)`` per layer -- and a
backend that pages it maps that shape onto its own blocks.  Keeping the plan
logical is what lets the reference backend and a card agree on what a ``cache_append``
means while disagreeing completely on where the bytes are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pocketllm.kernels.dtypes import DType
from pocketllm.kernels.tensor import TensorDesc

__all__ = ["CachePlan", "CacheLayout"]


@dataclass(frozen=True, slots=True)
class CacheLayout:
    """One cache tensor: its name in the graph, and its per-layer shape."""

    name: str
    layers: int
    kv_heads: int
    head_dim: int
    dtype: DType = DType.F32

    def shape(self, capacity: int) -> tuple[int, int, int, int]:
        """``(layers, capacity, kv_heads, head_dim)`` at a given context length.

        The layer axis is first because the cache is allocated once, whole: a
        decode step indexes into it by layer, and a per-layer tensor would mean a
        hundred allocations and a hundred bindings per step for no gain.
        """
        if capacity <= 0:
            raise ValueError(f"cache capacity must be positive, got {capacity}")
        return (self.layers, int(capacity), self.kv_heads, self.head_dim)

    def desc(self, capacity: int) -> TensorDesc:
        return TensorDesc(self.shape(capacity), dtype=self.dtype)


@dataclass(frozen=True, slots=True)
class CachePlan:
    """The cache tensors a model needs, and what one context length costs."""

    layouts: tuple[CacheLayout, ...] = ()
    #: The context the model was built for, used by :meth:`specs` when a caller
    #: does not want to pass one.
    default_capacity: int = 0

    def add(self, layout: CacheLayout) -> "CachePlan":
        return CachePlan(self.layouts + (layout,), self.default_capacity)

    def names(self) -> tuple[str, ...]:
        return tuple(layout.name for layout in self.layouts)

    def specs(self, capacity: int | None = None) -> tuple[tuple[str, TensorDesc], ...]:
        context = int(capacity if capacity is not None else self.default_capacity)
        if context <= 0:
            raise ValueError("no cache capacity was given and the plan has no default")
        return tuple((layout.name, layout.desc(context)) for layout in self.layouts)

    def bytes_for(self, capacity: int | None = None) -> int:
        """How much device memory this cache needs -- the number a phone budgets."""
        return sum(desc.nbytes for _, desc in self.specs(capacity))

    def summary(self, capacity: int | None = None) -> str:
        context = int(capacity if capacity is not None else self.default_capacity)
        shapes = ", ".join(f"{name}{layout.shape(context)}" for layout, (name, _) in zip(self.layouts, self.specs(context)))
        head = f"{len(self.layouts)} tensors at context {context}" if self.layouts else "no cache"
        return f"{head}: {shapes}" if self.layouts else head


def uniform_cache(
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    default_capacity: int,
    names: Sequence[str] = ("k_cache", "v_cache"),
    dtype: DType = DType.F32,
) -> CachePlan:
    """The common case: one K and one V tensor, the same shape in every layer.

    A checkpoint with a different arrangement -- per-layer window sizes, a
    shared-KV layer, a compressed cache -- builds its own :class:`CachePlan`
    instead.  That is the point of the plan being data: the unusual case writes a
    different one rather than special-casing this function.
    """
    if layers <= 0:
        raise ValueError(f"a model needs at least one layer, got {layers}")
    plan = CachePlan(default_capacity=default_capacity)
    for name in names:
        plan = plan.add(CacheLayout(name=name, layers=int(layers), kv_heads=int(kv_heads), head_dim=int(head_dim), dtype=dtype))
    return plan