"""One forward's starting position, in the two forms the stack has to read it in.

`start_pos` is the only number that changes from one decode step to the next, and every use of it in
this model is shaped by that number being a Python `int`. A CUDA graph cannot record any of that: a
slice's bounds are Python values at *record* time, so a capture bakes in the position it was recorded
at and every replay would be that same step. What a graph can record is an *index tensor*, so the
position has to reach the card as one — and it is also the reason the cache has to be *read* at a
width that does not depend on the position at all. That width is the second thing this type carries,
and it is the subject of the next section.

## What a position decides, and in which form

Four things in this model read a position, and they split two and two:

* the **row** the cache writes (`KVLatentCache.append`) and the **row** the rotary table is built
  from (`gguf_model.forward`'s `positions`) are discrete indices, and each has a spelling that
  survives a capture: `index_copy_` with a 1-element index tensor for the write, and
  `arange(seq) + pos` for the table. `Pos.row` is both.
* the **mask** and the **read width** are the same number seen twice, because the width the
  attention reads is what the mask has to describe. Those are `Pos.row` and `Pos.width`.

## Why the read width is on this type at all

A decode step reads the cache at the length it has reached, and that length is a Python `int` — so a
capture freezes it, and a graph recorded at 4096 rows can only ever be replayed at 4096 rows. The
attention's `N` is that width: it is the second dimension of the score GEMM and the width of the
softmax, and no index tensor can make it vary.

So the width is **rounded up to a bucket** and the rows past the position are masked out. That turns
`N` into a per-bucket constant, which is something a capture can hold. It is the same problem
`deepseek_v4_1`'s `Pos.upto` solves, and the two solve it differently: that version reads the *whole*
cache on every step, so its `N` is one constant for the life of the run, and this one reads the next
bucket end, which is at most twice the rows a step needs. The trade is symmetric — a constant `N`
costs a 32K context the whole 32K rows a token, a ladder costs a recording a rung — and the choice
here is the ladder, because the eager path is what a server runs and it should not read rows a step
does not need.

Two properties fall out of that choice and both are checked rather than assumed:

* **At a bucket end the arithmetic is exactly the unbucketed arithmetic.** If the position is the
  last one in its bucket then no row is masked and the GEMM, the softmax and the value contraction
  are over the same width they would have been — so the logits are bit-identical, not merely close.
* **Everywhere else the mask is what makes it right, and the mask is a device value.** `k_pos > pos`
  with `pos` a 0-dim tensor is a comparison the graph can hold; the eager path's mask, which is cell
  for cell the same rule, is the identity when no row needs hiding.

`width` is `None` on the host path, and that is not a shorthand for "no width" — it is "whatever the
cache holds", which is what every caller that is not a captured decode wants. A device `Pos` without
a width is refused rather than defaulted, because defaulting it would silently capture a length.

`tests/test_xing4_0_decode_pos.py` pins the host and device spellings against each other, on the
values and through a real capture; `tests/test_xing4_0_decode_graph.py` holds the two properties
above against the released checkpoint.
"""

from __future__ import annotations

import torch

__all__ = ["Pos", "write_row"]


def write_row(dst: torch.Tensor, index: "int | torch.Tensor", rows: torch.Tensor, dim: int = 1) -> None:
    """`dst[:, index] = rows` along `dim`, in the one spelling that survives a capture.

    The obvious spelling is what the eager path runs and what it still runs when the index is a
    Python `int`: `dst[:, i] = rows` is basic indexing and lowers to a copy into a view. Hand the
    same line a 0-dim CUDA tensor — the only kind of index an eager free graph has — and it raises
    `cudaErrorStreamCaptureUnsupported`: `at::indexing` treats a 0-dim *integer* tensor as an integer
    index, so it reads the value back to the host before the copy can be formed, and a capture
    forbids the read. A 1-element 1-d tensor is an index tensor instead and lowers to `index_copy_`,
    which is a kernel.

    So a write is the one place the two spellings cannot share an index object. `rows` is shaped
    like `dst` with `dim` of length `index.numel()` — which for a decode step is `(batch, 1, width)`
    against a `(batch, capacity, width)` cache, i.e. the shape `append` already receives. The two
    spellings write the same slots, which `tests/test_xing4_0_decode_pos.py` pins on the values.

    This is `deepseek_v4_1.decode_pos.write_row`'s mechanism, narrowed to the shapes this model
    writes: that version also accepts a run of indices, and nothing here does — a prefill chunk is
    never captured, so its `append` keeps the slice.
    """
    if isinstance(index, torch.Tensor):
        dst.index_copy_(dim, index.reshape(-1), rows)
    else:
        dst[(slice(None),) * dim + (index,)] = rows


class Pos:
    """A forward's starting position: a Python `int`, or a 0-dim int64 CUDA tensor.

    Built with `Pos.of` from whatever a caller passed as `start_pos`, `Pos.bucket` for a host-path
    step that should read a bucket width anyway, or `Pos.device` for a step that has to reach the
    card. Everything below returns a *value*, never a copy of the position: every layer of one
    forward shares one `Pos`, and only `set` and `advance` move it.
    """

    __slots__ = ("_host", "_tensor", "_width")

    def __init__(self, host: int, tensor: torch.Tensor | None = None, width: int | None = None) -> None:
        self._host = int(host)
        # A 0-dim tensor and not a `[1]`: it is used as an *index*, and an index tensor's shape is
        # the shape of the axis it selects. `latent[:, [3]]` is not `latent[:, 3]`.
        self._tensor = tensor
        self._width = None if width is None else int(width)
        if self._tensor is not None:
            self._tensor.fill_(self._host)

    # -- construction ------------------------------------------------------------------------ #

    @classmethod
    def of(cls, value: "int | Pos") -> "Pos":
        """Wrap a caller's `start_pos`, which the eager path passes as a plain `int`."""
        if isinstance(value, cls):
            return value
        if isinstance(value, torch.Tensor):
            raise TypeError(
                "start_pos arrived as a tensor with no host value beside it; a step that has to "
                "reach the card builds it with Pos.device and carries the counter on the host, "
                "where it is free"
            )
        return cls(int(value))

    @classmethod
    def bucket(cls, step: int, width: int) -> "Pos":
        """The host path at `step`, reading a `width`-wide cache instead of its own length.

        This is what makes a bucketed and a graphed step comparable: the two differ in how the ops
        are submitted and in nothing else, so a logit difference between them is the capture's and
        not the bucket's.
        """
        return cls(step, None, width)

    @classmethod
    def device(cls, step: int, device: torch.device | str, width: int) -> "Pos":
        """A position that reaches the card as an index tensor, at `step`, reading `width` rows.

        `width` is required here and has no default. A default would be the cache's length, which is
        a Python `int` — legal inside a capture and read once, at record time, so the graph would
        silently hold the bucket of the step it happened to be recorded at.
        """
        return cls(step, torch.zeros((), dtype=torch.int64, device=device), width)

    def set(self, step: int) -> "Pos":
        """Move to `step` on both sides. The two are filled from one integer, so they cannot drift."""
        self._host = int(step)
        if self._tensor is not None:
            self._tensor.fill_(self._host)
        return self

    def advance(self) -> "Pos":
        """The next step. The one place a decode loop should move the position from."""
        return self.set(self._host + 1)

    # -- what the caller reads ---------------------------------------------------------------- #

    @property
    def host(self) -> int:
        """The position as a Python int, on both paths."""
        return self._host

    @property
    def on_device(self) -> bool:
        """Whether this is the tensor path. Read it where the *shape* of a result differs."""
        return self._tensor is not None

    @property
    def width(self) -> int | None:
        """The cache width this position reads, or `None` for the cache's own length."""
        return self._width

    def first(self) -> bool:
        """Whether this is the forward's first position, which is the prefill chunk's own token."""
        return self._host == 0

    def __index__(self) -> int:
        """`int(pos)` is the position itself, which is what a host-side counter wants.

        On the device path this is a synchronising device read. It is fine everywhere this model
        calls it — the generation loop's bookkeeping, a test, a log line — and it is fatal inside a
        capture, which is why nothing on the captured path calls it. `row()` is the accessor for the
        slot the graph writes.
        """
        return self._host

    # -- the index objects the call sites use -------------------------------------------------- #

    def row(self, offset: int = 0) -> "int | torch.Tensor":
        """One position, `offset` away: the row of the cache this step writes.

        An `int` on the host path and a 0-dim tensor on the device path, which is exactly what
        `write_row` and the mask both want.
        """
        if self._tensor is None:
            return self._host + offset
        return self._tensor + offset if offset else self._tensor

    def span(self, n: int) -> "slice":
        """`n` consecutive rows from this one, for the write a *chunk* makes.

        Only the host path has this: a chunk's write is a slice, and a decode step's is a single
        row through `write_row`. Returning a slice on the device path would be a lie a capture
        cannot honour, so it is not offered there.
        """
        if self._tensor is not None:
            raise TypeError("a span is a slice, and a slice's bounds are Python values a capture freezes")
        return slice(self._host, self._host + n)

    def __repr__(self) -> str:
        where = f"{self._tensor.device} tensor" if self._tensor is not None else "host int"
        width = "cache length" if self._width is None else f"width {self._width}"
        return f"Pos({self._host}, {where}, {width})"
