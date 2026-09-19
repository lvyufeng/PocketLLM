"""Does `Pos` return the objects the lines it replaces used to build, on both paths?

`src/models/deepseek_v4_1/decode_pos.py` exists for one reason: a CUDA graph freezes every Python
value it records, so `start_pos` has to reach the card as an index tensor before a decode step can be
replayed at a moving position. The whole risk of that change is that the *eager* path -- which is
what every number on the V4.1 pages was measured with, and the column the graph is compared against
token by token -- quietly starts computing something else.

So the test is a differential one. For every site the six methods serve, the int path has to produce
exactly what the expression it replaced produced: `slice(start_pos, start_pos + n)` for `span`,
`start_pos % win` for `slot`, `start_pos + offset` for `row`, and the same `bool` for `first` and
`emits`. The device path is then held to the same values through the index tensors, checked by
indexing a real cache with both and requiring the results to be equal -- which is the property the
call sites actually rely on, since PyTorch takes ints, slices and index tensors interchangeably in
`cache[...]` and nothing else in the module promises the two are the same object.

One thing here is not a value check: **`emits` is computed from `pos.host` rather than from the
tensor**, because the answer picks which of two captured bodies a replay runs and therefore has to
be known before the launch. A test that only compared it to the literal expression on a host `int`
would not notice if a later edit read it off the card, so the device path is checked at a position
where the tensor and the host integer are deliberately different.

`publish` is the module's other half and its risk is the opposite one: it is *only* correct in a
capture, an ordinary call cannot tell the two behaviours apart. Its section below therefore ends
with a real capture, refills the buffer, replays, and requires the replayed value to be there --
which a value that lived in the recording's own pool would not be.

Every device case skips on a host without a card rather than being asserted in software, because the
dtype, the layout and the cost of the object are the point and a CPU tensor is not any of them.
"""

from __future__ import annotations

import pytest
import torch

from src.models.deepseek_v4_1.decode_pos import Pos, publish, write_row

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="the device path needs a card")

WINDOW = 8
RATIO = 4
WIDTH = 64


def _positions():
    """The positions worth checking: the first, one that emits at both ratios, one that does not."""
    return [0, 1, 3, 7, 8, 1024, 1025]


# -- the int path is the expressions it replaces ------------------------------------------------


def test_int_path_row_is_the_bare_integer():
    for start in _positions():
        pos = Pos.of(start)
        assert pos.row() == start
        assert pos.row(1 - RATIO) == start + 1 - RATIO
        assert isinstance(pos.row(), int)


def test_int_path_slot_is_the_modulo():
    for start in _positions():
        assert Pos.of(start).slot(WINDOW) == start % WINDOW
        assert isinstance(Pos.of(start).slot(WINDOW), int)


def test_int_path_span_is_the_literal_slice():
    for start in _positions():
        assert Pos.of(start).span(5) == slice(start, start + 5)
        assert Pos.of(start).span(4, start // 2) == slice(start // 2, start // 2 + 4)


def test_int_path_group_is_the_floor_division():
    for start in _positions():
        assert Pos.of(start).group(RATIO) == start // RATIO
        assert Pos.of(start).group(RATIO, 6) == (start + 6) // RATIO


def test_int_path_first_is_the_zero_test():
    for start in _positions():
        assert Pos.of(start).first() == (start == 0)


def test_int_path_emits_is_the_compressor_s_own_test():
    for start in _positions():
        for ratio in (1, 2, 4):
            assert Pos.of(start).emits(ratio) == ((start + 1) % ratio == 0)


def test_int_path_upto_reads_the_whole_cache_past_the_first_position():
    """The width rule, which is what makes `max |logit diff| == 0` reachable at all.

    A prefix slice here and a tail-masked full read on the graph path would be two softmaxes of
    different widths over the same values -- different kernels, different summation order, different
    last bits -- so the two columns could not agree bit for bit with no bug anywhere. Both paths
    read `full`; the prefill keeps its prefix because its `stop` is a chunk's group count.
    """
    assert Pos.of(0).upto(0, WIDTH) == slice(0, 0)
    for start in _positions():
        if start:
            assert Pos.of(start).upto(3, WIDTH) == slice(0, WIDTH)


def test_of_accepts_an_int_and_passes_a_pos_through():
    same = Pos.device(11, "meta")
    assert Pos.of(4).host == 4
    assert Pos.of(same) is same


def test_of_refuses_a_bare_tensor():
    """A tensor with no host counter beside it would cost a sync a layer to read back."""
    with pytest.raises(TypeError):
        Pos.of(torch.zeros((), dtype=torch.int64))


# -- the device path is the same values through an index tensor ---------------------------------


@CUDA
def test_device_path_is_zero_dim_int64():
    pos = Pos.device(7, "cuda:0")
    assert pos.on_device and pos.torch_device == torch.device("cuda:0")
    for obj in (pos.row(), pos.slot(WINDOW), pos.group(RATIO), pos.span(3)):
        assert isinstance(obj, torch.Tensor)
        assert obj.dtype == torch.int64 and obj.device == pos.torch_device
    # A 0-dim index preserves the rank of the axis it selects; `[1]` would add one, and the call
    # sites index rows of a cache whose layout is fixed.
    assert pos.row().shape == () and pos.slot(WINDOW).shape == ()


@CUDA
def test_device_path_selects_the_same_rows_as_the_int_path():
    """The property the call sites actually rely on, not the object identity.

    `cache[:bsz, pos.slot(win)] = v` is one line for both paths only because PyTorch treats an int
    and a 0-dim index tensor alike. So each pair is used to pull a row out of the same table and the
    two rows are required to be equal, which a `[1]`-shaped index or a wrong modulo would break.
    """
    table = torch.arange(WIDTH * 4, dtype=torch.float32, device="cuda:0").reshape(WIDTH, 4)

    def rows(obj) -> torch.Tensor:
        return table[obj % WIDTH]

    for start in _positions():
        eager, graph = Pos.of(start), Pos.device(start, "cuda:0")
        assert torch.equal(rows(eager.row()), rows(graph.row()))
        assert torch.equal(rows(eager.slot(WINDOW)), rows(graph.slot(WINDOW)))
        # A span selects several rows at once and is a slice on one path, a tensor on the other, so
        # the comparison is on the rows it reaches rather than on the index object.
        span = eager.span(3)
        assert span == slice(start, start + 3)
        assert graph.span(3).tolist() == [start, start + 1, start + 2]
        assert torch.equal(
            table[graph.span(3) % WIDTH], table[[i % WIDTH for i in range(span.start, span.stop)]]
        )


@CUDA
def test_device_path_span_is_fixed_width_and_starts_where_it_says():
    pos = Pos.device(9, "cuda:0")
    assert pos.span(5).tolist() == [9, 10, 11, 12, 13]
    assert pos.span(4, pos.group(RATIO)).tolist() == [2, 3, 4, 5]


@CUDA
def test_device_path_upto_reads_the_whole_cache():
    assert Pos.device(5, "cuda:0").upto(2, WIDTH) == slice(0, WIDTH)


@CUDA
def test_emits_reads_the_host_counter_not_the_tensor():
    """The variant is chosen before the launch, so the tensor must not be what answers.

    The two are moved apart on purpose: `set` fills the tensor from the host integer, so the only
    way to make them disagree is to stamp the tensor behind `Pos`'s back -- which is what a caller
    that advanced the card by some other route would do, and what this refuses to accept.
    """
    pos = Pos.device(7, "cuda:0")
    assert pos.emits(4) == (8 % 4 == 0)
    pos._tensor.fill_(11)  # deliberate: the host integer stays 7
    assert pos.host == 7
    assert pos.emits(4) == (8 % 4 == 0), "emits read the card, so the variant is not launch-time"


# -- the write is the one place the paths cannot share an index object --------------------------
#
# The buffer shape is the real one: `[batch, width, dim]` with the position along dimension 1, so
# `value` is the buffer minus that dimension -- `[batch, dim]` for one slot.

BSZ = 1
DIM = 4


def test_write_row_matches_the_literal_assignment_on_the_int_path():
    for start in _positions():
        table = torch.zeros(BSZ, WIDTH, DIM)
        value = torch.full((BSZ, DIM), float(start) + 0.5)
        write_row(table, Pos.of(start).slot(WINDOW), value)
        literal = torch.zeros(BSZ, WIDTH, DIM)
        literal[:, start % WINDOW] = value
        assert torch.equal(table, literal), f"int path at {start}"


@CUDA
def test_write_row_lands_on_the_same_row_as_the_int_path():
    for start in _positions():
        value = torch.full((BSZ, DIM), float(start) + 0.5, device="cuda:0")
        eager = torch.zeros(BSZ, WIDTH, DIM, device="cuda:0")
        graph = torch.zeros(BSZ, WIDTH, DIM, device="cuda:0")
        write_row(eager, Pos.of(start).slot(WINDOW), value)
        write_row(graph, Pos.device(start, "cuda:0").slot(WINDOW), value)
        assert torch.equal(eager.cpu(), graph.cpu()), f"device path at {start}"


@CUDA
def test_write_row_survives_a_capture_and_writes_nothing_else():
    """The reason `write_row` exists, held as a test rather than as a comment.

    `cache[:bsz, pos.slot(win)] = v` is the natural line and it is what `_window_kv` ran until this
    was measured: with a 0-dim CUDA tensor in that index position the capture fails with
    `cudaErrorStreamCaptureUnsupported`, because a 0-dim integer tensor is an integer index and has
    to be read back to the host. The `write_row` spelling is `index_copy_` and replays.

    The table is refilled between the capture and the replay, so what the assertion reads can only
    have come from the replay: a recording that ran its kernels at record time would leave the canary
    where the test then looks, and the two would otherwise be indistinguishable.
    """
    canary = 0.25
    table = torch.full((BSZ, WIDTH, DIM), canary, device="cuda:0")
    value = torch.ones(BSZ, DIM, device="cuda:0")
    pos = Pos.device(1024, "cuda:0")
    slot = int(pos.slot(WINDOW))

    def body() -> None:
        write_row(table, pos.slot(WINDOW), value)

    for _ in range(2):  # the warm launches the capture pass runs before recording
        body()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    table.fill_(canary)
    graph.replay()
    torch.cuda.synchronize()

    assert bool((table[0, slot] == 1.0).all()), "the replay did not write its row"
    assert int((table != 1.0).any(dim=-1).sum()) == WIDTH - 1, "the replay wrote more than its row"


# -- the published slots outlive the body that fills them ---------------------------------------


def test_publish_clones_on_the_first_call_and_copies_after():
    """A buffer, not an alias: the caller keeps handing the same object and it keeps being updated."""
    source = torch.ones(BSZ, DIM)
    slot = publish(None, source)
    assert slot is not source, "an alias would be overwritten by whatever the source holds next"
    assert torch.equal(slot, source)
    slot.fill_(9.0)
    assert bool((source == 1.0).all())
    assert publish(slot, source) is slot
    assert bool((slot == 1.0).all())


def test_publish_rebinds_when_the_shape_changes():
    """Prefill publishes `seqlen` rows and a decode step publishes one, in that order.

    The indexer's output is `[batch, seqlen, topk]` and `seqlen` is 1024 at the prompt and 1 for
    every step after it. A `copy_` across that difference is a broadcast or an error rather than the
    step's value, so the shapes are compared and a mismatch starts a new buffer -- which the decode
    then keeps for good, since nothing after the prompt changes its shape.
    """
    slot = publish(None, torch.ones(BSZ, WIDTH, DIM))
    one = publish(slot, torch.ones(BSZ, 1, DIM))
    assert one is not slot and one.shape == (BSZ, 1, DIM)
    assert publish(one, torch.full((BSZ, 1, DIM), 2.0)) is one
    assert bool((one == 2.0).all())
    assert bool((slot == 1.0).all()), "the prefill's buffer was written through"


def test_publish_rebinds_when_the_dtype_changes():
    """`topk_idxs` is int32 and `candidates` is bool; one slot per source, but the check is cheap."""
    slot = publish(None, torch.ones(BSZ, 1, DIM))
    other = publish(slot, torch.ones(BSZ, 1, DIM, dtype=torch.int32))
    assert other.dtype == torch.int32 and other is not slot


@CUDA
def test_publish_keeps_the_buffer_out_of_the_capture():
    """The reason `publish` exists, held as a test rather than as a comment.

    A body under capture allocates into the graph's pool, so the natural line -- `slot = fresh` --
    leaves the *consumer's* recording holding an address that only the producer's replay ever
    writes, and the pool is shared with every other recording in the round. `publish` puts the value
    in a buffer allocated before the capture and the recording contains a `copy_` into it.

    The buffer is refilled with a canary between the capture and the replay, so the value the
    assertion reads can only have come from the replay. A body that had published the allocation
    itself would leave the canary where the test looks.
    """
    canary = 0.25
    source = torch.full((BSZ, DIM), 2.0, device="cuda:0")
    slot: torch.Tensor | None = None

    def body() -> None:
        nonlocal slot
        slot = publish(slot, source * 3.0)

    for _ in range(2):  # the warm launches a capture pass runs before recording
        body()
    torch.cuda.synchronize()
    assert slot is not None
    address = slot.data_ptr()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    torch.cuda.synchronize()
    assert slot.data_ptr() == address, "the recording rebound the slot"

    slot.fill_(canary)
    graph.replay()
    torch.cuda.synchronize()
    assert bool((slot == 6.0).all()), "the replay did not write through the published buffer"


# -- the counter is one integer, kept on both sides ---------------------------------------------


@CUDA
def test_set_and_advance_keep_the_two_in_step():
    pos = Pos.device(1024, "cuda:0")
    assert int(pos) == 1024 and pos.host == 1024
    assert pos.row().item() == 1024
    pos.advance()
    assert pos.host == 1025 and pos.row().item() == 1025
    pos.set(3)
    assert pos.host == 3 and pos.slot(WINDOW).item() == 3


def test_host_path_advance_moves_only_the_integer():
    pos = Pos.of(8)
    pos.advance()
    assert pos.host == 9 and not pos.on_device


def test_repr_names_both_paths():
    assert "host int" in repr(Pos.of(1))
