"""Running a planned region, by capture or by compilation, or not at all.

:mod:`pocketllm.engine.planner` decides *which* regions a backend claims to be
able to take whole.  This module is what happens next, and its governing rule is
that a backend's graph path is **optional and may decline at any moment**.

Both graph methods can return ``None``:

* ``session.compile_graph`` on a backend whose AOT toolchain is not installed;
* ``session.capture`` on a stream-capture backend that met a shape its recorded
  graph cannot be replayed at.

Neither is an error.  The ABI says so explicitly (``GraphCapability.supported``
defaults to ``False``, and the docstrings on both methods say a ``None`` means
"run this eagerly"), and the whole rebuild would be pointless if a phone whose
DSP refused one region could not fall back to the CPU for that region and carry
on.  So this file's job is to make that fallback the *default* path rather than a
special case: :func:`run_region` either returns a result or returns ``None``, and
a caller that gets ``None`` runs the region through the executor with no branch
of its own.

The one thing that is *not* silently absorbed is a mismatch in what a capture
returns: if a backend replays a region and hands back a different number of
values than the region has outputs, that is a bug in the backend rather than a
fallback, and it raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from pocketllm.kernels.graph import GraphRegion
from pocketllm.kernels.tensor import Tensor

__all__ = ["RegionOutcome", "run_region", "CaptureRefused"]


class CaptureRefused(Exception):
    """A backend declined a region.

    Raised internally and caught by :func:`run_region`; it is not part of the
    public surface, but it is a named type rather than a bare ``None`` so a
    backend that wants to explain itself can put the reason in the message and a
    caller that wanted a trace can read it.
    """


@dataclass(slots=True)
class RegionOutcome:
    """What a graph-path run produced, and how it was produced."""

    outputs: Mapping[str, Tensor]
    #: ``"captured"`` (a replayed stream) or ``"compiled"`` (an AOT artifact).
    mode: str

    def summary(self) -> str:
        names = ", ".join(self.outputs)
        return f"{self.mode} [{names}]"


def run_region(
    session,
    region: GraphRegion,
    inputs: Mapping[str, Tensor],
) -> RegionOutcome | None:
    """Run ``region`` through the backend's graph path, or ``None`` to run it eagerly.

    ``inputs`` maps the region's input names to tensors, in the order
    ``region.inputs`` declares them.  The two modes are tried in the order the
    backend's declared :class:`~pocketllm.kernels.backend.GraphMode` prefers:

    * ``aot_compile`` first compiles the region and runs the artifact -- the
      result is cached by the session's own compiled-graph handle, so a decode
      loop compiles once;
    * ``stream_capture`` records a replay and re-runs it.

    Both are attempted for a backend that declares both, which is not
    hypothetical: Ascend offers ``aclgraph`` capture and ``atc`` compilation, and
    which one wins depends on the region and on the board.

    A ``None`` from either method, or a :class:`CaptureRefused`, returns ``None``
    and the caller runs the region op by op.  A *shape* is all this function
    enforces on the way out: a graph path that returns the wrong number of values
    is a backend bug, and absorbing it would turn it into a wrong answer.
    """
    capability = session.backend.graph()
    if not capability.supported:
        return None

    ordered = [inputs[name] for name in region.inputs]

    for mode in _modes_in_order(capability):
        try:
            if mode == "compiled":
                outcome = _try_compile(session, region, ordered)
            else:
                outcome = _try_capture(session, region, ordered)
        except CaptureRefused:
            continue
        if outcome is not None:
            return outcome
    return None


def _modes_in_order(capability) -> tuple[str, ...]:
    """``("compiled", "captured")`` or the reverse, per the declared primary mode."""
    from pocketllm.kernels.backend import GraphMode

    if capability.mode is GraphMode.AOT_COMPILE:
        return ("compiled", "captured")
    if capability.mode is GraphMode.STREAM_CAPTURE:
        return ("captured", "compiled")
    return ()


def _try_compile(session, region: GraphRegion, ordered: Sequence[Tensor]) -> RegionOutcome | None:
    graph = session.compile_graph(region)
    if graph is None:
        return None
    values = graph.run(ordered)
    _expect(len(values), region)
    return RegionOutcome(outputs=_bind(region, values), mode="compiled")


def _try_capture(session, region: GraphRegion, ordered: Sequence[Tensor]) -> RegionOutcome | None:
    captured = session.capture(region)
    if captured is None:
        return None
    values = captured.replay(ordered)
    _expect(len(values), region)
    return RegionOutcome(outputs=_bind(region, values), mode="captured")


def _bind(region: GraphRegion, values: Sequence[Tensor]) -> dict[str, Tensor]:
    return {name: tensor for name, tensor in zip(region.outputs, values)}


def _expect(count: int, region: GraphRegion) -> None:
    if count != len(region.outputs):
        raise ValueError(
            f"the graph path returned {count} values for a region with "
            f"{len(region.outputs)} outputs ({', '.join(region.outputs)})"
        )