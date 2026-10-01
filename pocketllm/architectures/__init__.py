"""Model architectures: what a model *is*, as a graph the engine can run.

An architecture turns a config into a :class:`~pocketllm.architectures.ir.ModelSpec`
-- a graph, the weights it reads, and the cache it needs -- and stops there.  It
names no device, allocates nothing, and does not import a backend.  Running the
spec is the engine's job, which is why the same executor drives ``toy`` and will
drive ``xing4_0`` when it is ported.

The distinction the naming is keeping straight, since three similar words appear
throughout:

* **architecture** (here) -- the model's structure;
* **backend** (:mod:`pocketllm.backends`) -- the device that runs it;
* **engine** (:mod:`pocketllm.engine`) -- the execution and serving contract.

Only ``toy`` ships today.  It exists so the scaffold is *runnable*: a builder
that has never produced a graph an executor accepted is a design, not code.
"""

from __future__ import annotations

from .cache import CacheLayout, CachePlan, uniform_cache
from .ir import GraphBuilder, ModelSpec, WeightSpec, WeightTable
from .registry import ARCHITECTURES, ArchitectureEntry, build, get, names

__all__ = [
    "ARCHITECTURES",
    "ArchitectureEntry",
    "CacheLayout",
    "CachePlan",
    "GraphBuilder",
    "ModelSpec",
    "WeightSpec",
    "WeightTable",
    "build",
    "get",
    "names",
    "uniform_cache",
]