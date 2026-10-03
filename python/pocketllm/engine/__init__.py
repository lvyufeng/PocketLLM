"""The engine: turn a device and a graph into running work.

The layers below this one are declarations.  :mod:`pocketllm.kernels` says what
an op *is*; :mod:`pocketllm.backends` says which device can run it and how;
:mod:`pocketllm.architectures` says what a model's graph looks like.  The engine
is where those meet: it selects a device, lays out the graph's memory, splits it
into regions for a backend's optional graph path, and walks it.

What this package deliberately does **not** own:

* **A model's structure.**  ``LLM`` holds a spec, but it does not *know* one.
  An architecture builds the :class:`~pocketllm.kernels.graph.Graph`; the engine
  runs whatever graph it is handed.  That is what lets the same executor drive a
  toy two-op graph in a test and a 29B model on a card, which is the property
  the rebuild is for.
* **A tokenizer, or sampling policy.**  Those are the serving layer's, and they
  are device-neutral by construction.
* **A device runtime.**  The engine names ``session``, ``alloc`` and ``run``;
  which library is behind those is the backend's business.

``LLM`` and ``AsyncLLM`` live here rather than in the package façade because they
are the first thing that would import a backend, and the façade must stay
torch-free on ``import pocketllm``.
"""

from __future__ import annotations

from .captured import CaptureRefused, RegionOutcome, run_region
from .decode import (
    STEP_INPUTS,
    Decoder,
    Generation,
    Sampler,
    cache_descriptors,
    pick_token,
    singleton_plan,
)
from .executor import Execution, ExecutionTrace, Executor, NodeResult
from .llm import LLM, AsyncLLM
from .memory import BufferArena, CountingSession, Liveness, MemoryPlan, plan_memory, value_liveness
from .planner import ExecutionPlan, PlanRegion, plan_execution
from .session import EngineExecutor, EngineSession, NoUsableBackend, SessionPolicy

__all__ = [
    "AsyncLLM",
    "BufferArena",
    "CaptureRefused",
    "CountingSession",
    "Decoder",
    "EngineExecutor",
    "EngineSession",
    "Execution",
    "ExecutionPlan",
    "ExecutionTrace",
    "Executor",
    "Generation",
    "LLM",
    "Liveness",
    "MemoryPlan",
    "NoUsableBackend",
    "NodeResult",
    "PlanRegion",
    "RegionOutcome",
    "STEP_INPUTS",
    "Sampler",
    "SessionPolicy",
    "cache_descriptors",
    "pick_token",
    "plan_execution",
    "plan_memory",
    "run_region",
    "singleton_plan",
    "value_liveness",
]