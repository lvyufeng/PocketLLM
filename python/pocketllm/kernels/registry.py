"""The op registry: the ABI's fixed vocabulary.

The set of op names in :data:`OPS` *is* the ABI.  A backend declares which of
these it implements; it does not invent new ones.  A third-party backend may
extend the vocabulary through ``OpRegistry.extend`` (an explicit, named call),
but the tree's own ops are all declared under :mod:`pocketllm.kernels.ops`, and
a new op there is not complete until the reference backend implements it
(``tests/abi/test_reference_completeness.py``).
"""

from __future__ import annotations

from .errors import OpNotDeclaredError
from .schema import OpSchema

__all__ = ["OpRegistry", "OPS"]


class OpRegistry:
    """A name -> :class:`OpSchema` table that refuses to be scribbled on silently."""

    def __init__(self) -> None:
        self._ops: dict[str, OpSchema] = {}
        self._frozen = False

    def declare(self, schema: OpSchema) -> OpSchema:
        if self._frozen:
            raise RuntimeError(
                f"cannot declare {schema.name!r}: the core vocabulary is frozen; "
                "use OpRegistry.extend for a backend-specific op"
            )
        existing = self._ops.get(schema.name)
        if existing is not None and existing != schema:
            raise ValueError(f"op {schema.name!r} is already declared with a different schema")
        self._ops[schema.name] = schema
        return schema

    def extend(self, schema: OpSchema) -> OpSchema:
        """Add an op outside the frozen core vocabulary, for a backend extension."""
        self._ops[schema.name] = schema
        return schema

    def freeze(self) -> None:
        self._frozen = True

    def get(self, name: str) -> OpSchema:
        try:
            return self._ops[name]
        except KeyError as exc:
            raise OpNotDeclaredError(
                f"op {name!r} is not declared; the vocabulary is {sorted(self._ops)}"
            ) from exc

    def names(self) -> frozenset[str]:
        return frozenset(self._ops)

    def schemas(self) -> tuple[OpSchema, ...]:
        return tuple(self._ops[name] for name in sorted(self._ops))

    def __contains__(self, name: object) -> bool:
        return name in self._ops

    def __len__(self) -> int:
        return len(self._ops)

    def __iter__(self):
        return iter(self._ops)


#: The one registry.  Populated by importing :mod:`pocketllm.kernels.ops`.
OPS = OpRegistry()