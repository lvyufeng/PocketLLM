"""One decode position, in the two forms the stack has to read it in.

`start_pos` is the only number that changes from one decode step to the next, and every use of it in
`attention.py` is shaped by that number being a Python `int`: `freqs_cis[start_pos : start_pos +
seqlen]` is a slice, `window_kv_cache[:bsz, start_pos % window_size] = ...` writes one row, and
`self.freqs_cis[start_pos + 1 - ratio]` picks one. A CUDA graph cannot record any of that -- a
slice's bounds and an integer index are Python values at *record* time, so a capture bakes in the
position it was recorded at and every replay is that same step. What a graph can record is an *index
tensor*, so the position has to reach the card as one.

`Pos` is the whole of the difference. It wraps either the Python int the eager path already carries
or a 0-dim int64 CUDA tensor, and each method returns the kind of index object the current code
builds by hand: an `int` or a 0-dim tensor for a single row, a `slice` or a fixed-width index tensor
for a run of positions. PyTorch takes ints, slices and index tensors interchangeably in `cache[...]`,
so a *read* like `freqs_cis[pos.span(seqlen)]` is one line that means the same thing on both paths and
no call site branches.

A *write* is not, and `write_row` below is the one exception: a 0-dim tensor used as the index of an
assignment does not survive a capture, while an `int` does. That is the only place the two paths
diverge, and it is measured rather than assumed.

`publish` is the same idea one level up, for the two tensors a layer hands to the layers after it
(`shared.topk_idxs` and `shared.candidates`): they are produced inside the body that is being
*recorded*, so they cannot be placed in the recording's own memory and read by the next layer's
recording. What publishes them is a buffer outside every capture, written through `copy_`.

Three things are deliberately *not* on the device, and the reason is the same for all three: a
Python int is known when the graph is recorded, a tensor read is not.

* `first()` -- whether this is position 0 -- picks between the prefill body and the decode body,
  which are different kernels rather than different arguments.
* `emits(ratio)` picks between the compressor's fill body and its emit body, for the same reason.
  Its answer is also what decides which of two captured variants a step replays, so it has to be
  known before the launch rather than inside it.
* `host` is what the remaining host-side arithmetic reads. A device `Pos` carries its own integer,
  kept in step by `set`/`advance`, and the contract is that the two agree -- it is the step counter
  that produced the tensor, not a second source of truth read back off the card, which would cost a
  synchronization per layer and a second one per step.

The int path is not an afterthought: it is what the eager path runs, unchanged, and its methods
return exactly the objects the lines they replace used to build. `tests/test_models_deepseek_v4_1_decode_pos.py`
pins that against the literal expressions.
"""

from __future__ import annotations

import torch

__all__ = ["Pos", "publish", "write_row"]


def write_row(dst: torch.Tensor, index: "int | slice | torch.Tensor", value: torch.Tensor, dim: int = 1) -> None:
    """`dst[:, index] = value` along `dim`, in the one spelling a capture survives on both paths.

    The obvious spelling is what the eager path ran and it is what the int path still runs:
    `dst[:, i] = v` with a Python integer is basic indexing and lowers to a copy into a view. Hand
    the same line a 0-dim CUDA tensor -- the only kind of index an eager free graph has -- and it
    raises `cudaErrorStreamCaptureUnsupported`. A 0-dim integer tensor is *an integer index* to
    `at::indexing`, so it is read back to the host before the copy can be formed, and a capture
    forbids the read. A 1-element 1-d tensor is an index tensor instead and lowers to `index_copy_`,
    which is a kernel.

    So a write is the one place the two paths cannot share an index object, and this is that place.
    `value` is shaped like `dst` minus `dim` for one index and like `dst` for a run of them -- the
    shapes the call sites already pass. Both spellings write the same slots, which
    `tests/test_models_deepseek_v4_1_decode_pos.py` pins on the values and on a real capture.

    Measured on this host before it was written, rather than reasoned about: see
    `/tmp/repro_capture_setitem.py`, where `dst[:, slot0d] = v` failed the capture and
    `dst.index_copy_(1, slot1d, v.unsqueeze(1))` replayed onto exactly the rows the int spelling
    wrote, at both widths the decode step uses.
    """
    if isinstance(index, torch.Tensor):
        dst.index_copy_(dim, index.reshape(-1), value.unsqueeze(dim))
    else:
        dst[(slice(None),) * dim + (index,)] = value


def publish(slot: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    """The buffer a captured body puts a value into, so that a *later* recording can read it.

    `shared.topk_idxs` and `shared.candidates` outlive the body that produces them: one layer's
    indexer publishes them and the layers between it and the next index source read them, and on the
    graph path every one of those bodies is a *recording*. A tensor a recording allocates lives in
    the graph's pool, which is the wrong place for a published value on three counts:

    * the value arrives when a replay runs and at no other time, so between two replays the address
      holds whatever the pool handed out next -- and a consumer that recorded that address reads it
      without a producer having written it in the same step;
    * a layer with two A bodies has two of them. The emitting body and the filling body each
      allocate their own, and a replay writes only the one the step picked, while every consumer's
      recording holds the address that belonged to whichever body happened to be recorded last. The
      consumers then read a value from the *previous* step, which is not an out-of-bounds index and
      not a crash: it is a wrong top-k, and a wrong answer that still looks like one;
    * the pool is shared by every recording in the round, so an address a consumer recorded is an
      address that a later layer's capture may already have handed to an unrelated tensor.

    So the value goes into a buffer allocated outside any capture and a recording contains a
    `copy_` into it rather than the allocation itself -- the rule `graphs.Sink` follows for the
    activations, for the same reason. Shapes are fixed within a decode and change only between
    prefill and decode (where the indexer's output is `seqlen` wide, not one), so a `value` of a
    different shape allocates a new buffer rather than broadcasting into the old one, and the eager
    path rebinds exactly where it always did.

    Only the shape and dtype are compared, not the identity of `value`: the point is to hand back
    the same tensor object for every value a decode step can produce.
    """
    if slot is None or slot.shape != value.shape or slot.dtype != value.dtype or slot.device != value.device:
        return value.detach().clone()
    slot.copy_(value)
    return slot


class Pos:
    """A decode position that is either a Python `int` or a 0-dim int64 CUDA tensor.

    Constructed with `Pos.of` from whatever a caller passed as `start_pos`, or with `Pos.device`
    for a step that has to reach the card. Everything below returns a *value*, never a copy of the
    position: two layers in one step share one `Pos`, and nothing here mutates it except `set`.
    """

    __slots__ = ("_host", "_tensor")

    def __init__(self, host: int, tensor: torch.Tensor | None = None) -> None:
        self._host = int(host)
        # A 0-dim tensor and not a `[1]` one: it is used as an *index*, and an index tensor's shape
        # is the shape of the axis it selects. A `[1]` would add a dimension to every `cache[...]`
        # it appears in, and `freqs_cis[[3]]` is a different object from `freqs_cis[3]`.
        self._tensor = tensor
        if self._tensor is not None:
            self._tensor.fill_(self._host)

    # -- construction and advance ---------------------------------------------------------------

    @classmethod
    def of(cls, value: "int | Pos") -> "Pos":
        """Wrap a caller's `start_pos`, which the eager path passes as a plain `int`."""
        if isinstance(value, cls):
            return value
        if isinstance(value, torch.Tensor):
            raise TypeError(
                "start_pos arrived as a tensor with no host value beside it; build the step's "
                "position with Pos.device and carry the counter on the host, where it is free"
            )
        return cls(int(value))

    @classmethod
    def device(cls, step: int, device: torch.device | str) -> "Pos":
        """A position that reaches the card as an index tensor, at `step`."""
        return cls(step, torch.zeros((), dtype=torch.int64, device=device))

    def set(self, step: int) -> "Pos":
        """Move to `step` on both sides. The two are filled from one integer, so they cannot drift."""
        self._host = int(step)
        if self._tensor is not None:
            self._tensor.fill_(self._host)
        return self

    def advance(self) -> "Pos":
        """The next step. The one place a decode loop should move the position from."""
        return self.set(self._host + 1)

    # -- what the host needs to know -------------------------------------------------------------

    @property
    def host(self) -> int:
        """The position as a Python int, on both paths."""
        return self._host

    @property
    def on_device(self) -> bool:
        """Whether this is the tensor path. Read it where the *shape* of a result differs."""
        return self._tensor is not None

    @property
    def torch_device(self) -> torch.device:
        """Where the index tensor lives. Only meaningful on the device path.

        Not named `device`, which is the constructor for that path -- `Pos.device(step, device)`.
        """
        assert self._tensor is not None, "the host path has no device: it indexes with Python ints"
        return self._tensor.device

    def first(self) -> bool:
        """Whether this is the forward's first position, i.e. the prefill body."""
        return self._host == 0

    def __index__(self) -> int:
        """A `Pos` is an integer position, so `int(pos)` is the position itself.

        That is what a caller that needs the number rather than an index object asks for: the
        Engram hash front end indexes its cache with a Python slice
        (`loader.py`, `self.cache[:batch, start_pos : start_pos + seqlen]`), so it takes `int(...)`
        of whatever it is handed and gets the same value on both paths.
        """
        return self._host

    def emits(self, ratio: int) -> bool:
        """Whether the compressor of a `ratio`-source completes a group on this position.

        The compressor's own test, `(start_pos + 1) % ratio == 0`, and the same expression: what
        changes is only that it is asked before the launch instead of inside the captured body.
        """
        return (self._host + 1) % ratio == 0

    # -- the index objects the call sites use ----------------------------------------------------

    def row(self, offset: int = 0) -> "int | torch.Tensor":
        """One position, `offset` away. Indexes a row of a table shaped one row per position."""
        if self._tensor is None:
            return self._host + offset
        return self._tensor + offset if offset else self._tensor

    def pick(self, table: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """The one row of `table` this position `offset` away names, as a row and not a new axis.

        `table[pos.row(offset)]` is the expression this replaces, and what the int path below still
        runs. On the device path that read is a *host* read for the same reason the write is:
        `at::indexing` treats a 0-dim integer tensor as an integer index, so `Tensor.__getitem__`
        lowers it to `select(dim, index.item())` and a capture refuses the `.item()`. Measured, not
        inferred -- the capture failed here, on `freqs_cis[pos.row(1 - ratio)]`, after the write was
        fixed.

        A one-element index tensor takes the index-tensor route instead, and `index_select` is a
        kernel; the `squeeze` puts back the axis the tensor route added, so both paths return the
        same shape and `apply_rotary_emb` cannot tell them apart.
        """
        if self._tensor is None:
            return table[self._host + offset]
        return table.index_select(0, (self._tensor + offset).reshape(1)).squeeze(0)

    def slot(self, mod: int) -> "int | torch.Tensor":
        """Which slot of a `mod`-long ring this position is. Indexes the ring's row."""
        return self._tensor % mod if self._tensor is not None else self._host % mod

    def group(self, mod: int, offset: int = 0) -> "int | torch.Tensor":
        """Which `mod`-long group the position `offset` ahead falls in; `pos.group(r, seqlen)` is
        therefore the count of groups the forward reaches."""
        return self.row(offset) // mod

    def span(self, n: int, start: "int | torch.Tensor | None" = None) -> "slice | torch.Tensor":
        """`n` consecutive positions from `start`, which defaults to this one.

        An index tensor rather than a slice on the device path, and the same width on both: `n` is a
        Python int at record time either way, so a capture's shapes do not depend on the position.
        """
        if self._tensor is None:
            base = self._host if start is None else start
            return slice(base, base + n)
        base = self._tensor if start is None else start
        return torch.arange(n, device=self._tensor.device, dtype=torch.int64) + base

    def upto(self, stop: "int | torch.Tensor", full: int) -> slice:
        """The first `stop` positions of a cache that is `full` wide.

        `stop` grows by one every step, which is the one thing a capture cannot hold -- a slice
        recorded at `stop` is that `stop` forever -- so a decode reads the whole cache and masks the
        tail out downstream, the way the prefill branch already masks with
        `arange(width) >= compress_lens`. `full` is a per-config constant, so every shape after it is
        fixed.

        **Both paths read the whole cache, not just the tensor one.** A tail-masked read and a
        prefix read differ in the width of the softmax that consumes them, and a softmax over 640
        slots is not the same arithmetic as one over 1152: the reduction is a different kernel with
        a different summation order, so the two would differ in the last bits with no bug anywhere.
        That would make `max |logit diff| == 0` -- the bar this exists to hold the graph to --
        unreachable for a reason that has nothing to do with the graph. Reading one width is what
        makes the comparison a statement about the capture.

        The prefill keeps its prefix: its `stop` is the chunk's own group count rather than a
        position in the sequence, and the mask it applies downstream is shaped for it.
        """
        return slice(0, stop) if self.first() else slice(0, full)

    def __repr__(self) -> str:
        where = f"{self._tensor.device} tensor" if self._tensor is not None else "host int"
        return f"Pos({self._host}, {where})"
