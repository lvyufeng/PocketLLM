"""Resolution: which backend runs this op, and why.

Dispatch is a pure function over declarations -- it reads schemas and
capabilities, never a device -- so a resolution can be tested on a host with no
accelerator at all, and a given tree resolves identically on two machines.

Two things make this more than a lookup:

* ``explain`` returns *why* each candidate was accepted or rejected, as data.
  The old tree's dispatch raised "unsupported" with no trace, and finding out
  which layer refused a call was the expensive part of adding a format.
* The reference backend is always a candidate but never a *preference*: a wrong
  device still gets a correct-but-slow answer rather than "no backend", unless
  the session's fallback policy forbids it (``serve`` does, because a 29B model
  on numpy is a ten-minute first token, not graceful degradation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .backend import Backend, Capability
from .device import Device
from .dtypes import DType, QuantFormat
from .errors import NoBackendError, ShapeError
from .registry import OPS, OpRegistry
from .tensor import Tensor

__all__ = ["ResolvedOp", "Resolution", "Dispatcher"]


@dataclass(frozen=True, slots=True)
class ResolvedOp:
    """The chosen backend, and one declaration of the op it will run."""

    backend: Backend
    capability: Capability
    op: str
    #: True when the chosen backend is the reference one, i.e. a host round-trip
    #: for a call the device could not take.  A region containing one is not
    #: capturable.
    is_reference: bool = False


@dataclass(frozen=True, slots=True)
class Resolution:
    """The outcome of resolving one call, including the rejects and their reasons."""

    op: str
    chosen: ResolvedOp | None
    rejected: tuple[tuple[str, str], ...]

    def reason(self) -> str:
        if self.chosen is not None:
            return f"{self.op} -> {self.chosen.backend.name}"
        lines = [f"no backend for {self.op!r}:"]
        lines.extend(f"  {name}: {why}" for name, why in self.rejected)
        return "\n".join(lines)


class Dispatcher:
    """Resolves ops against a fixed, ordered set of backends."""

    def __init__(
        self,
        backends: Sequence[Backend],
        *,
        preference: Sequence[str] = (),
        registry: OpRegistry | None = None,
        allow_reference_fallback: bool = True,
    ) -> None:
        self.backends = tuple(backends)
        self.preference = tuple(preference)
        self.registry = registry or OPS
        self.allow_reference_fallback = allow_reference_fallback

    def candidates(self, op: str, args: Sequence[Any], device: Device, *, attrs: Mapping[str, Any] | None = None):
        """Every backend that could run ``op`` over ``args`` on ``device``, best first."""
        schema = self.registry.get(op)  # raises OpNotDeclaredError for an unknown op
        dtypes, quants = _operand_types(schema, args)
        out: list[ResolvedOp] = []
        for backend in self.backends:
            if backend.device_kind != device.kind and not _is_reference(backend):
                continue
            if not backend.available():
                continue
            for cap in backend.capabilities():
                if cap.op != op:
                    continue
                if not cap.admits(dtypes, quants):
                    continue
                if cap.accepts is not None and not cap.accepts(args, dict(attrs or {})):
                    continue
                out.append(ResolvedOp(backend, cap, op, is_reference=_is_reference(backend)))
        out.sort(key=self._sort_key)
        return out

    def resolve(self, op: str, args: Sequence[Any], device: Device, *, attrs: Mapping[str, Any] | None = None) -> ResolvedOp:
        resolution = self.explain(op, args, device, attrs=attrs)
        if resolution.chosen is None:
            raise NoBackendError(resolution.reason())
        return resolution.chosen

    def explain(self, op: str, args: Sequence[Any], device: Device, *, attrs: Mapping[str, Any] | None = None) -> Resolution:
        schema = self.registry.get(op)
        dtypes, quants = _operand_types(schema, args)
        rejected: list[tuple[str, str]] = []
        chosen: ResolvedOp | None = None
        for backend in self.backends:
            name = backend.name
            if backend.device_kind != device.kind and not _is_reference(backend):
                rejected.append((name, f"device kind {backend.device_kind!r} != {device.kind!r}"))
                continue
            if not backend.available():
                rejected.append((name, "not available on this host"))
                continue
            caps = [c for c in backend.capabilities() if c.op == op]
            if not caps:
                rejected.append((name, f"does not declare {op!r}"))
                continue
            if _is_reference(backend) and not self.allow_reference_fallback and chosen is None:
                # Keep it as a last resort only when the policy allows it.
                rejected.append((name, "reference fallback is disabled"))
                continue
            match = next((c for c in caps if c.admits(dtypes, quants) and (c.accepts is None or c.accepts(args, dict(attrs or {})))), None)
            if match is None:
                rejected.append((name, _domain_reason(caps[0], dtypes, quants)))
                continue
            candidate = ResolvedOp(backend, match, op, is_reference=_is_reference(backend))
            if chosen is None or self._sort_key(candidate) < self._sort_key(chosen):
                if chosen is not None:
                    rejected.append((chosen.backend.name, "a better candidate was found"))
                chosen = candidate
            else:
                rejected.append((name, "lower preference"))
        return Resolution(op=op, chosen=chosen, rejected=tuple(rejected))

    def _sort_key(self, resolved: ResolvedOp) -> tuple[int, int, str]:
        try:
            explicit = self.preference.index(resolved.backend.name)
        except ValueError:
            explicit = len(self.preference)
        # The reference backend sorts last among equals, so a real device wins.
        reference_penalty = 1 if resolved.is_reference else 0
        return (explicit, reference_penalty * 10_000 + resolved.capability.rank, resolved.backend.name)


def _is_reference(backend: Backend) -> bool:
    return getattr(backend, "is_reference", backend.name == "reference")


def _operand_types(schema, args: Sequence[Any]) -> tuple[frozenset[DType], frozenset[QuantFormat]]:
    """The types a backend must admit to run this call.

    Only the *data* arguments count.  An index or a token id is a tensor too --
    ``embedding`` takes ``tokens: i32`` next to a float table, ``attention``
    takes ``positions: i32`` next to f32 queries -- but no backend's capability
    domain ranges over i32, and it should not have to declare that it does.  The
    schema fixes an auxiliary argument's type with ``ArgSpec.dtype``, which is
    the same marker the return type uses; anything declared that way is data the
    op consumes, not a domain the op operates over.
    """
    dtypes: set[DType] = set()
    quants: set[QuantFormat] = set()
    for spec, value in zip(schema.args, args):
        if spec.dtype is not None:
            continue
        desc = getattr(value, "desc", None)
        if desc is None or not isinstance(value, Tensor):
            continue
        if desc.dtype is not None:
            dtypes.add(desc.dtype)
        if desc.quant is not None:
            quants.add(desc.quant)
    return frozenset(dtypes), frozenset(quants)


def _domain_reason(cap: Capability, dtypes: frozenset[DType], quants: frozenset[QuantFormat]) -> str:
    missing_dtypes = dtypes - cap.dtypes if cap.dtypes else frozenset()
    missing_quants = quants - cap.quants if cap.quants else frozenset()
    parts = []
    if missing_dtypes:
        parts.append("no dtype " + ", ".join(sorted(d.value for d in missing_dtypes)))
    if missing_quants:
        parts.append("no quant " + ", ".join(sorted(q.name for q in missing_quants)))
    return "; ".join(parts) or "domain predicate rejected the call"