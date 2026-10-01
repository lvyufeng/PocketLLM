"""The reference backend: numpy on the host, the normative implementation.

It is the backend that is always there.  On a machine with no device runtime --
which includes CI -- it is the only one that runs, and it runs *everything*,
because a declared op with no reference implementation has no definition the
accelerated backends can be checked against.  That is why
``tests/abi/test_reference_completeness.py`` treats "declared but not
implemented" as a failure and not a gap.

It is also, for the same reason, never the *chosen* backend when a real one
matches: dispatch sorts it last, and the serving path can turn it off entirely,
because a correct answer that takes ten minutes is not graceful degradation.
"""

from __future__ import annotations

from .session import ReferenceBackend, ReferenceSession

BACKEND = ReferenceBackend()

__all__ = ["BACKEND", "ReferenceBackend", "ReferenceSession"]