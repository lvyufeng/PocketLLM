"""A decode position in the two forms the stack reads it in, and what a capture does to each.

`src/models/xing4_0/decode_pos.py` carries one number in two spellings, and this file pins them
against each other three ways: on the values (`row`, `span`, `advance`), on the refusals (a tensor
with no host value, a device `Pos` without a width, a `span` on the device path), and **through a real
CUDA capture** — which is the only test here that can distinguish a spelling that works from one that
merely looks right. `dst[:, i] = rows` and `dst.index_copy_(1, [i], rows)` compute the same tensor and
only the second can be recorded; a test that never records cannot tell them apart.

The capture tests build their own buffers rather than a model. That is deliberate: what they pin is
the mechanism any captured step rests on — an index tensor written by a graph, and `arange + pos` read
by one — and a buffer test can state the property exactly. `tests/test_xing4_0_decode_graph.py` is
where the same mechanism is held against the released checkpoint's arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.models.xing4_0.attention import KVLatentCache
from src.models.xing4_0.config import Xing4_0Params
from src.models.xing4_0.decode_pos import Pos, write_row


CONFIG = Path("/mnt/data2/Xing4.0-29B-A4B/config.json")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="a capture needs a card")


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _params(*, kv_lora_rank: int = 8, qk_rope_head_dim: int = 4) -> Xing4_0Params:
    """The released config at fixture widths, so a cache here is a real `KVLatentCache`.

    Only `kv_lora_rank` and `qk_rope_head_dim` are read by the cache — the widths that decide how
    wide a row is — but the rest comes from the checkpoint rather than from a stub, because the
    subject of the capture test below is the write and a stub could have got the row width wrong
    without anything noticing.
    """
    if not CONFIG.exists():
        pytest.skip(f"{CONFIG} is not on disk")
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw.update(kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim)
    return Xing4_0Params.from_config(raw)


# --------------------------------------------------------------------------- #
# `write_row`: the two spellings, and the one a capture forces
# --------------------------------------------------------------------------- #


def test_the_two_spellings_write_the_same_slot() -> None:
    """A Python index and a 1-element index tensor land on the same row, byte for byte.

    This is the property that lets `KVLatentCache.append` pick between them without an arithmetic
    branch: the chunk's `int` and the decode step's tensor are the same write, and only the second
    can be recorded.
    """
    width = 3
    rows = torch.tensor([[[1.0, 2.0, 3.0]]])
    by_int = torch.zeros((1, 5, width))
    by_tensor = torch.zeros((1, 5, width))
    write_row(by_int, 2, rows)
    write_row(by_tensor, torch.tensor([2]), rows)
    assert torch.equal(by_int, by_tensor)
    assert by_int[0, 2].tolist() == [1.0, 2.0, 3.0]
    # And nothing else moved: a write is one row, not a shift or a fill.
    assert float(by_int.sum()) == 6.0


def test_a_zero_dim_index_is_the_shape_write_row_takes() -> None:
    """`Pos.row()` is 0-dim, which is an *index*, and `write_row` reshapes it to 1-d.

    The distinction is not cosmetic and it is why `write_row` exists at all: `at::indexing` reads a
    0-dim integer tensor back to the host to form the copy, which is exactly what a capture forbids,
    and a 1-element 1-d tensor lowers to `index_copy_`, which is a kernel. So the 0-dim form is what
    `Pos.row()` returns — `latent[:, pos]` and `latent[:, [pos]]` are different axes — and
    `write_row` is the one place that widens it.
    """
    pos = Pos.device(0, "cpu", 5)
    row = pos.row()
    assert isinstance(row, torch.Tensor) and row.ndim == 0
    dst = torch.zeros((1, 5, 2))
    write_row(dst, pos.set(3).row(), torch.tensor([[[7.0, 8.0]]]))
    assert dst[0, 3].tolist() == [7.0, 8.0]


def test_a_run_of_rows_lands_on_the_rows_the_index_names() -> None:
    """`index_copy_` takes a run, so the helper does too — and no call site gives it one.

    A prefill chunk is never captured, so its `append` keeps the slice and only ever hands this one
    index. The run is left working rather than refused because refusing it would be a check on a code
    path that does not exist; what this pins is that the helper's reshape to 1-d is a shape fix and
    not a narrowing to one element.
    """
    dst = torch.zeros((1, 5, 2))
    write_row(dst, torch.tensor([1, 4]), torch.tensor([[[1.0, 1.0], [4.0, 4.0]]]))
    assert dst[0, 1].tolist() == [1.0, 1.0]
    assert dst[0, 4].tolist() == [4.0, 4.0]
    assert float(dst.sum()) == 10.0


# --------------------------------------------------------------------------- #
# The two constructions, and the refusals that make them honest
# --------------------------------------------------------------------------- #


def test_of_wraps_an_int_and_passes_a_pos_through() -> None:
    host = Pos.of(7)
    assert not host.on_device and host.host == 7 and host.width is None
    assert Pos.of(host) is host


def test_a_bare_tensor_is_refused_rather_than_read_back() -> None:
    """A tensor `start_pos` with no host value is a caller mistake, and reading it back hides it.

    `int(tensor)` would work — it synchronises and returns the value — so the convenient behaviour is
    the dangerous one: a loop that passes a tensor position is a loop that is already doing a device
    read a step, and the refusal is what tells it so.
    """
    with pytest.raises(TypeError):
        Pos.of(torch.tensor(3))


def test_a_device_pos_without_a_width_is_refused() -> None:
    """The width has no default, because its natural default is a capture's trap.

    "The cache's own length" is a Python int. A default would read it once, at record time, and the
    graph would hold the bucket of whichever step happened to be recorded first — wrong on every
    later step and invisible until a logits comparison caught it.
    """
    with pytest.raises(TypeError):
        Pos.device(0, "cpu")  # type: ignore[call-arg]


def test_a_device_pos_carries_the_step_from_the_start() -> None:
    pos = Pos.device(5, "cpu", 64)
    assert pos.host == 5 and pos.width == 64 and pos.on_device
    assert int(pos) == 5
    assert pos.row().tolist() == 5


# --------------------------------------------------------------------------- #
# What the call sites read
# --------------------------------------------------------------------------- #


def test_row_is_an_int_on_the_host_and_a_tensor_on_the_device() -> None:
    host, device = Pos.of(4), Pos.device(4, "cpu", 64)
    assert isinstance(host.row(), int) and host.row() == 4
    assert isinstance(device.row(), torch.Tensor)
    assert device.row().item() == 4
    # The offset form is the same accessor one step on, and it is how the mask names the position.
    assert host.row(1) == 5
    assert device.row(1).item() == 5
    assert host.row() == 4, "the offset must not move the position it was read from"
    assert host.host == 4


def test_span_is_a_slice_and_the_device_path_does_not_have_one() -> None:
    """A chunk's write is `pos.span(n)`; a decode step's is one row. Only one of them is a slice.

    Returning a slice on the device path would be a slice whose bounds are Python values a capture
    freezes, which is the failure this whole module exists to avoid — so it raises instead.
    """
    assert Pos.of(3).span(4) == slice(3, 7)
    with pytest.raises(TypeError):
        Pos.device(3, "cpu", 64).span(4)


def test_set_and_advance_move_both_spellings_together() -> None:
    """One integer fills both sides, so a device `Pos` cannot drift from its host counter."""
    pos = Pos.device(0, "cpu", 64)
    pos.set(9)
    assert (pos.host, int(pos.row())) == (9, 9)
    pos.advance()
    assert (pos.host, int(pos.row())) == (10, 10)
    assert not (pos.first()) and Pos.of(0).first()


def test_first_is_the_prefill_chunk_s_own_token() -> None:
    """`first()` is what the callers branch on to tell a chunk's position from a step's."""
    assert Pos.of(0).first()
    assert not Pos.of(1).first()


# --------------------------------------------------------------------------- #
# Through a real capture
# --------------------------------------------------------------------------- #


def _capture(body) -> torch.cuda.CUDAGraph:
    """Record `body` after warming it on a side stream, in the shape `graphs._Step` records in."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            body()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    torch.cuda.synchronize()
    return graph


@requires_cuda
def test_a_captured_write_lands_on_the_row_the_position_names() -> None:
    """The replay writes row `pos` — a different row at every replay, from one recording.

    This is the whole point of putting the position in a tensor: the recording holds an *address* and
    the row it writes is decided at replay time, on the card. A recording that had baked in the row
    would write the same one forever, which is the bug `graph-frozen` in the stage-1 probe *is*.
    """
    device = _device()
    width = 4
    dst = torch.zeros((1, 8, width), device=device)
    rows = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], device=device)
    pos = Pos.device(0, device, 8)
    graph = _capture(lambda: write_row(dst, pos.row(), rows))

    for step in (0, 3, 7):
        pos.set(step)
        graph.replay()
        torch.cuda.synchronize()
        assert dst[0, step].tolist() == [1.0, 2.0, 3.0, 4.0], f"row {step} was not the one written"
    # And only those three: a write is one row, not a fill.
    assert float(dst.sum()) == 3 * 10.0


@requires_cuda
def test_a_captured_arange_reads_the_position_the_replay_names() -> None:
    """`arange(seq) + pos` is the rotary table's row, and it is read from the card too.

    The alternative — `arange(start_pos, start_pos + seq)` — takes its bounds from a Python int, so a
    recording would rope every replay at the position it was recorded at. The device spelling is the
    one that makes a captured step's RoPE follow the loop.
    """
    device = _device()
    sink = torch.zeros(1, device=device)
    pos = Pos.device(0, device, 8)
    graph = _capture(lambda: sink.copy_(torch.arange(1, device=device, dtype=torch.float32) + pos.row()))

    for step in (0, 5, 63):
        pos.set(step)
        graph.replay()
        torch.cuda.synchronize()
        assert sink.item() == float(step)


@requires_cuda
def test_a_captured_cache_append_writes_the_row_its_position_names() -> None:
    """The same property where it is actually used: `KVLatentCache.append` on the tensor path.

    The cache advances no `length` on this path — comparing an index tensor against the capacity
    would be a host read — so what this pins is the write, and the length being the caller's is what
    `graphs.DecodeGraphs._advance` is for.
    """
    device = _device()
    cache = KVLatentCache(1, 16, _params(), device=device, dtype=torch.float32)
    rows = torch.ones((1, 1, cache.width), device=device) * 2.0
    pos = Pos.device(0, device, 16)
    graph = _capture(lambda: cache.append(rows, pos.row()))
    before = cache.length

    for step in (0, 11, 15):
        pos.set(step)
        graph.replay()
        torch.cuda.synchronize()
        assert cache.latent[0, step].tolist() == [2.0] * cache.width
    assert cache.length == before, "the device path must leave the length to its caller"


@requires_cuda
def test_the_slice_spelling_cannot_be_recorded_at_all() -> None:
    """The negative control for everything above: `dst[:, pos] = rows` is refused by a capture.

    `write_row`'s docstring gives a mechanism for why the two spellings cannot share an index object
    — `at::indexing` reads a 0-dim integer tensor back to the host to form the copy, and a capture
    forbids the read. A mechanism stated but never exercised is a comment, so this records the naive
    spelling and asserts the refusal. Without it, `write_row`'s reshape would look like a style
    preference and a later simplification could quietly reintroduce the slice.

    The refusal is reported at `capture_end` as `cudaErrorStreamCaptureInvalidated`, one step after
    the read that caused it, and it leaves the context usable — which is why this can live in the
    same process as the tests that do capture successfully.
    """
    device = _device()
    dst = torch.zeros((1, 8, 2), device=device)
    rows = torch.ones((1, 1, 2), device=device)
    pos = Pos.device(0, device, 8)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            dst[(slice(None), pos.row())] = rows
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    with pytest.raises(RuntimeError):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            dst[(slice(None), pos.row())] = rows
    torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# The torch facts the bucketed host path rests on
# --------------------------------------------------------------------------- #


def test_an_all_false_mask_returns_its_input_bit_for_bit() -> None:
    """Why `_mask` applies the decode mask unconditionally instead of branching on the width.

    On the host path at the cache's own length no row needs hiding, and the tempting shape is `if
    tokens > pos: mask`. Applying it anyway is what makes a bucketed eager step and a captured one
    differ in their *submission* and in nothing else — and it is only legal because the identity case
    is exact. That is what is pinned here rather than assumed, on the fp32 tensor the path actually
    masks: `masked_fill` with no cell set returns its input, so the branch would be free to remove
    and this test says so.

    `pos` is a 0-dim tensor on the device path, and the comparison it forms with `k_pos` is the other
    half of the same claim: nothing about `k_pos > pos` needs the position to be a Python value.
    """
    scores = torch.randn(1, 2, 1, 6, dtype=torch.float32)
    k_pos = torch.arange(6).view(1, 1, 1, 6)
    nothing = k_pos > 5  # the position is the last row in a six-row read
    assert not bool(nothing.any())
    assert torch.equal(scores.masked_fill(nothing, float("-inf")), scores)
    # And the same rule as a device value, which is what a capture holds: the five rows past a
    # position of one are hidden and the first two are not.
    pos = Pos.device(1, "cpu", 6)
    hidden = k_pos > pos.row()
    assert hidden.view(-1).tolist() == [False, False, True, True, True, True]
