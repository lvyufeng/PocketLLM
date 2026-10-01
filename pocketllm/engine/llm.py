"""``LLM``: a model spec and an open device, in one object.

This is the smallest thing that deserves the name.  It does **not** tokenize, it
does **not** sample, and it has no request queue -- those are the serving layer's,
and the serving layer is a later phase.  What it does is the part that has to be
right before any of them can be: take a
:class:`~pocketllm.architectures.ir.ModelSpec`, open one device for it, hold the
tensors the spec names as weights, and run its graph.

Two properties come straight out of the engine below it, and are worth stating
because they are the ones a caller will notice:

* **One device, chosen once.**  ``LLM`` opens exactly one session and holds it.
  Nothing here spawns a rank or splits a model; a spec that does not fit is a
  spec whose *weights* should have been quantized further, which is a decision
  made before this object exists.
* **Weights are bound, not copied per call.**  A weight is a graph input the
  caller supplies once; the executor reads it from the bound mapping on every
  step.  Getting this wrong is the difference between a decode loop and a leak.

The async form is a thin wrapper, and honestly so: the compute below it is
synchronous, and a worker thread is how a serving loop gets concurrency out of
one device.  Pretending otherwise -- an ``async def`` that never awaits -- would
make callers think they can overlap two runs on one session, which they cannot.
"""

from __future__ import annotations

import asyncio
from typing import Any, Mapping, Sequence

from pocketllm.architectures.ir import ModelSpec
from pocketllm.kernels.device import Device
from pocketllm.kernels.tensor import Tensor, TensorDesc

from .executor import Executor
from .session import EngineSession, SessionPolicy

__all__ = ["LLM", "AsyncLLM"]


class LLM:
    """A model on one device: hold the weights, run the graph.

    ```python
    model = LLM(spec, device="cpu")
    model.bind({"embedding": t, "ffn.gate": t, ...})   # or let a loader do it
    y = model({"tokens": ids})["y"]
    ```
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        device: Device | str | int | None = None,
        policy: SessionPolicy | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        self.session = EngineSession.open(device, policy=policy, options=options)
        self._executor = self.session.executor()
        self._weights: dict[str, Tensor] = {}
        # Verified once, at construction, rather than on every ``bind`` and every
        # call: the descriptor of a weight is a property of the graph, and a graph
        # does not change shape underneath a live model.
        self._env: dict[str, TensorDesc] = spec.graph.verify()

    @classmethod
    def from_spec(cls, spec: ModelSpec, **kwargs: Any) -> "LLM":
        return cls(spec, **kwargs)

    # -- weights ------------------------------------------------------------

    def bind(self, tensors: Mapping[str, Tensor]) -> "LLM":
        """Hold ``tensors`` under the names the graph knows, checking each one.

        A bound tensor must name a graph input, and its descriptor must be the
        one the graph inferred.  Checking here rather than at the first op is the
        whole point: a weight bound at the wrong shape produces a graph that
        fails three nodes later, inside a kernel, with an error naming a value
        nobody recognises.  The graph already knows every descriptor; use it.

        Weights and request inputs are bound through the same door on purpose --
        both are graph inputs, and the engine draws no distinction between them.
        What separates them is *when* they arrive: weights once, ``tokens`` every
        step.
        """
        for name, tensor in tensors.items():
            desc = self._env.get(name)
            if desc is None:
                raise KeyError(
                    f"{name!r} is not a graph input of {self.spec.name}; "
                    f"it takes {list(self.spec.graph.input_names)}"
                )
            if tensor.desc != desc:
                raise ValueError(
                    f"tensor {name!r} has descriptor {tensor.desc}, but the graph declares {desc}"
                )
        self._weights.update(tensors)
        return self

    @property
    def missing_weights(self) -> tuple[str, ...]:
        """Spec weights that have not been bound yet, in declaration order."""
        return tuple(name for name in self.spec.weight_values if name not in self._weights)

    def ready(self) -> bool:
        """Whether every weight the spec names is bound.

        A graph with no weights is always ready; a spec with weights is ready
        exactly when ``missing_weights`` is empty.  This is a statement about
        *weights*, not about whether a particular call was given its inputs.
        """
        return not self.missing_weights

    def weights_from(self, tensors: Sequence[Tensor], names: Sequence[str] | None = None) -> "LLM":
        """Bind a positional run of tensors to the weights in declaration order."""
        ordered = self.spec.weight_values if names is None else tuple(names)
        if len(tensors) != len(ordered):
            raise ValueError(f"{len(tensors)} tensors for {len(ordered)} weights")
        return self.bind(dict(zip(ordered, tensors)))

    # -- running ------------------------------------------------------------

    def __call__(self, inputs: Mapping[str, Tensor], *, trace: bool = False):
        missing = self.missing_weights
        if missing:
            raise RuntimeError(
                f"{self.spec.name} is missing {len(missing)} weight(s): {list(missing)}"
            )
        bound = dict(self._weights)
        bound.update(inputs)
        return self._executor.run(self.spec.graph, bound, trace=trace)

    def run(self, inputs: Mapping[str, Tensor], *, trace: bool = False):
        return self(inputs, trace=trace)

    # -- lifetime -----------------------------------------------------------

    def close(self) -> None:
        self._weights.clear()
        self._executor.close()
        self.session.close()

    def __enter__(self) -> "LLM":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def describe(self) -> str:
        state = "ready" if self.ready() else f"{len(self.missing_weights)} weights unbound"
        return f"{self.spec.describe()} [{state}] on {self.session.describe()}"


class AsyncLLM:
    """The same object with an awaitable call, for a serving loop.

    The compute is synchronous -- one device, one session, no overlap -- so each
    call goes to a worker thread and the event loop stays free to accept the next
    request.  A caller that awaits two calls concurrently gets two threads
    contending for one device, which is slower than awaiting them in turn; that
    is a property of the hardware, not of this wrapper.  The lock makes the
    serialization explicit instead of leaving it to a race.
    """

    def __init__(self, llm: LLM) -> None:
        self._llm = llm
        self._lock = asyncio.Lock()

    @classmethod
    def from_spec(cls, spec: ModelSpec, **kwargs: Any) -> "AsyncLLM":
        return cls(LLM.from_spec(spec, **kwargs))

    @property
    def spec(self) -> ModelSpec:
        return self._llm.spec

    def bind(self, tensors: Mapping[str, Tensor]) -> "AsyncLLM":
        self._llm.bind(tensors)
        return self

    async def __call__(self, inputs: Mapping[str, Tensor], *, trace: bool = False):
        async with self._lock:
            return await asyncio.to_thread(self._llm, inputs, trace=trace)

    async def run(self, inputs: Mapping[str, Tensor], *, trace: bool = False):
        return await self(inputs, trace=trace)

    async def close(self) -> None:
        await asyncio.to_thread(self._llm.close)

    def describe(self) -> str:
        return self._llm.describe()