"""Contract tests for `src/models/deepseek_v4_1/resident_bank.py`.

The bank exists to make one claim true: after it is filled, a routed expert's bytes and an Engram
row's bytes come from host memory and not from `/mnt/data3`. Two things can make that claim false
while every counter still looks right, and they are what this file tests.

**The offsets.** The bank reproduces the shard's own byte order and then hands out views into its own
segment, so its offset arithmetic is the only thing standing between a kernel and a permutation of
the right bytes. The release does not order experts numerically -- `layers.0.ffn.experts.100.*` sits
between expert 10's and expert 11's, because the names are sorted as strings -- so an implementation
that placed expert `e` at `e * per_expert_bytes` would be plausible, self-consistent, and wrong for
all 384 experts. The test that catches it fills a real segment from a real shard and compares every
one of a layer's 2,304 tensors against the checkpoint's own view of the same tensor. It needs the
checkpoint and skips without it. The same arithmetic moves the two Engram tables, and
`test_a_filled_bank_hands_back_what_the_shard_holds` runs that half end to end on a synthetic
checkpoint, so the wiring is checked on the hosts that have no `/mnt/data3`.

**The layout check.** `_layout` is asserted rather than assumed, and the assertions are the design:
each kind is one contiguous range (that is what makes the fill 80 reads rather than 92,160), and the
projections cycle `w1, w2, w3` across that range (that is what makes the slot map a grouping and not
a coincidence). These are checked against synthetic entries, so they run everywhere and pin the
failure mode -- a release that grouped by projection would still be contiguous, and would put expert
5's `w1` into expert 0's `w3` slot.

A third claim, and the reason both regions are in one segment rather than two: a key the bank does
not hold must still be readable. The dense tree is 16.79 GiB that is deliberately *not* resident, so
`packed` has to answer for it from the mapping -- and a bank that answered with a plausible zero, or
that claimed a key it had no slot for, would fail silently rather than loudly.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch

from src.loader.safetensors import TensorEntry
from src.models.deepseek_v4_1 import resident_bank as rb


REPO_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = os.environ.get("DEEPSEEK_V41_CHECKPOINT", "/mnt/data3/DeepSeek-V4.1-Flash")

# What the release actually holds, measured from its index: 384 experts of three projections in two
# kinds, 18,800,640 bytes each. Spelled out because a change to the checkpoint that this repository
# has not seen should fail here rather than quietly reshape the segment.
EXPERTS = 384
PER_EXPERT_BYTES = 18_800_640
LAYER_BYTES = EXPERTS * PER_EXPERT_BYTES
SHAPES = {
    ("w1", "q"): (2304, 2560),
    ("w1", "s"): (2304, 160),
    ("w2", "q"): (5120, 1152),
    ("w2", "s"): (5120, 72),
    ("w3", "q"): (2304, 2560),
    ("w3", "s"): (2304, 160),
}
# Where the synthetic layer's second kind starts, so a test can name a slot's absolute position.
KIND_GAP = 10_000_000

# The synthetic Engram table's geometry, and the two sizes the offset assertions are written against.
TABLE_SHAPES = {"weight": (64, 4), "scale": (64, 1)}

# The release's two Engram layers and what their tables weigh, measured from its index: the codes
# and their E8M0 scales, 94.42 GiB a table and 188.83 GiB together.
ENGRAM_LAYERS = (1, 14)
ENGRAM_BYTES = 202_758_032_400


def _nbytes(shape) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


TABLE_WEIGHT_BYTES = _nbytes(TABLE_SHAPES["weight"])
TABLE_SCALE_BYTES = _nbytes(TABLE_SHAPES["scale"])


class _FakeReader:
    """Just enough of `MmapSafetensors` for `_layout`: a name -> `TensorEntry` mapping, and the
    shard's data-section base.

    The base is deliberately not zero. `TensorEntry.begin` counts from the data section, so a run's
    `file_offset` is `data_offset(file_name) + begin`; a fake that reported 0 would let an
    implementation that forgot the header pass here and read garbage out of the real shard.
    """

    def __init__(self, entries: dict[str, TensorEntry], *, data_offset: int = 4096) -> None:
        self.entries = entries
        self._data_offset = data_offset

    def data_offset(self, file_name: str) -> int:
        return self._data_offset


def _synthetic_layer(layer_id: int, *, order=None, files=None, kinds=("s", "q")):
    """A layer whose entries are laid out one kind at a time, in `order`.

    `order` is the sequence of `(expert, projection)` in file order and defaults to the release's:
    the experts in string order, each expert's three projections together.
    """
    if order is None:
        order = [
            (int(expert), which)
            for expert in sorted((str(e) for e in range(EXPERTS)), key=str)
            for which in rb.PROJECTIONS
        ]
    if files is None:
        files = {(expert, which): f"model-{layer_id:05d}.safetensors" for expert, which in order}
    entries: dict[str, TensorEntry] = {}
    for kind in kinds:
        # The two kinds are separate contiguous ranges in the shard, the second one well past the
        # first -- which is what makes each layer two runs and not one.
        cursor = 1_000 * layer_id + KIND_GAP * kinds.index(kind)
        for expert, which in order:
            shape = SHAPES[(which, kind)]
            entries[f"layers.{layer_id}.ffn.experts.{expert}.{which}."
                    f"{'weight' if kind == 'q' else 'scale'}"] = TensorEntry(
                file_name=files[(expert, which)], dtype="I8" if kind == "q" else "F8_E8M0",
                shape=shape, begin=cursor, end=cursor + _nbytes(shape),
            )
            cursor += _nbytes(shape)
    return entries


def _synthetic_table(layer_id: int, *, files=None, order=rb.TABLE_LEAVES):
    """An Engram layer as the release stores it: the codes from the data section's first byte, the
    scales straight after, both in one shard.

    `order` is the order the two live in, which the bank takes from the index rather than assuming --
    a release that wrote the scales first is read correctly, and this is what says so.
    """
    default = f"model-{layer_id:05d}.safetensors"
    files = {leaf: (files or {}).get(leaf, default) for leaf in rb.TABLE_LEAVES}
    entries: dict[str, TensorEntry] = {}
    cursor = 0
    for leaf in order:
        shape = TABLE_SHAPES[leaf]
        entries[f"layers.{layer_id}.engram.embed.{leaf}"] = TensorEntry(
            file_name=files[leaf], dtype="F8_E4M3" if leaf == "weight" else "F8_E8M0",
            shape=shape, begin=cursor, end=cursor + _nbytes(shape),
        )
        cursor += _nbytes(shape)
    return entries


def _pread_window(root: str, file_name: str, offset: int, count: int) -> torch.Tensor:
    """`count` bytes at `offset` in a shard, read the way the fill reads them.

    Not `entry_view`, which is the thing being checked: this goes to the file with the same syscall
    pair so that a difference is the offset arithmetic and nothing else.
    """
    buffer = bytearray(count)
    fd = os.open(os.path.join(root, file_name), os.O_RDONLY)
    try:
        got = os.preadv(fd, [memoryview(buffer)], offset)
    finally:
        os.close(fd)
    assert got == count, f"{file_name}: asked for {count} bytes at {offset}, got {got}"
    return torch.frombuffer(buffer, dtype=torch.uint8)


def test_layout_reproduces_the_release_geometry() -> None:
    reader = _FakeReader(_synthetic_layer(0))
    layers, size, n_experts = rb._layout(reader)
    assert n_experts == EXPERTS
    assert len(layers) == 1
    layer = layers[0]
    assert layer.nbytes == LAYER_BYTES
    assert len(layer.slots) == EXPERTS * 6
    # two runs, because the two kinds are contiguous ranges and nothing in the layout merges them
    assert len(layer.runs) == 2


def test_layout_follows_the_shards_string_order_not_the_numeric_one() -> None:
    """A slot is where the shard put it, which for expert 11 is after expert 109 and not after 10."""
    reader = _FakeReader(_synthetic_layer(0))
    layers, _, _ = rb._layout(reader)
    layer = layers[0]
    offsets = {expert: layer.slots[(expert, "w1", "q")][0] for expert in range(EXPERTS)}

    # Derived from the file order the layer was built in, not from the layout code: the scales range
    # first, then each expert's three packed weights back to back, the experts in string order.
    scales_bytes = EXPERTS * sum(_nbytes(SHAPES[(which, "s")]) for which in rb.PROJECTIONS)
    stride = sum(_nbytes(SHAPES[(which, "q")]) for which in rb.PROJECTIONS)
    order = [int(expert) for expert in sorted((str(e) for e in range(EXPERTS)), key=str)]
    expected = {}
    cursor = scales_bytes
    for expert in order:
        expected[expert] = cursor
        cursor += stride
    assert offsets == expected

    # 11 is the shard's fourteenth expert -- after 10, 100, 101, ... 109 -- so it sits thirteen
    # strides in, not eleven, and after 109 rather than between 1 and 2.
    assert order.index(11) == 13
    assert offsets[11] == scales_bytes + 13 * stride != scales_bytes + 11 * stride
    assert offsets[11] > offsets[109] > offsets[10]


def test_a_run_offset_is_a_file_offset_not_a_data_section_one() -> None:
    """The fill `pread`s at `run.file_offset`, so it has to count past the shard's header.

    `TensorEntry.begin` counts from the data section; a run built from it directly reads whichever
    tensor happens to live 258,144 bytes earlier in the file -- a real tensor, of the right size, in
    the right shape, and the wrong one. This pins the addition rather than trusting it.
    """
    entries = _synthetic_layer(0)
    layers, _, _ = rb._layout(_FakeReader(entries, data_offset=258_144))
    layer = layers[0]
    first_scale = min(entry.begin for entry in entries.values())
    assert layer.runs[0].file_offset == 258_144 + first_scale
    assert layer.runs[0].file_offset != first_scale


def test_layout_rejects_a_projection_major_order() -> None:
    """The failure a contiguity check alone would miss: one range, every expert's w1 in w3's slot."""
    order = [
        (int(expert), which)
        for which in rb.PROJECTIONS
        for expert in sorted((str(e) for e in range(EXPERTS)), key=str)
    ]
    reader = _FakeReader(_synthetic_layer(0, order=order))
    with pytest.raises(ValueError, match="do not cycle"):
        rb._layout(reader)


def test_layout_rejects_a_hole_inside_a_kind() -> None:
    entries = _synthetic_layer(0)
    # push one tensor 64 bytes further along, leaving a gap in the middle of the scales group
    key = "layers.0.ffn.experts.200.w2.scale"
    entry = entries[key]
    entries[key] = TensorEntry(entry.file_name, entry.dtype, entry.shape,
                               entry.begin + 64, entry.end + 64)
    with pytest.raises(ValueError, match="byte gap"):
        rb._layout(_FakeReader(entries))


def test_layout_rejects_a_kind_split_across_shards() -> None:
    files = {(expert, which): f"model-{0 if expert < 100 else 1:05d}.safetensors"
             for expert in range(EXPERTS) for which in rb.PROJECTIONS}
    with pytest.raises(ValueError, match="span 2 shards"):
        rb._layout(_FakeReader(_synthetic_layer(0, files=files)))


def test_layout_rejects_a_missing_expert() -> None:
    entries = _synthetic_layer(0)
    entries.pop("layers.0.ffn.experts.383.w3.weight")
    with pytest.raises(ValueError, match="missing 1"):
        rb._layout(_FakeReader(entries))


def test_layout_rejects_experts_not_numbered_from_zero() -> None:
    """A layer whose experts are 0..382 and 384 -- 384 present, 383 absent -- is not a layer this
    bank can index by expert number, so it has to say so rather than place 384 where 383 was."""
    entries = _synthetic_layer(0)
    for which in rb.PROJECTIONS:
        for leaf in ("weight", "scale"):
            entry = entries.pop(f"layers.0.ffn.experts.383.{which}.{leaf}")
            entries[f"layers.0.ffn.experts.384.{which}.{leaf}"] = entry
    with pytest.raises(ValueError, match="without gaps"):
        rb._layout(_FakeReader(entries))


# ---------------------------------------------------------------- the Engram half

# A geometry small enough to write to disk in a test and shaped like the release: the two kinds are
# separate ranges with the dense tensors between them, the projections cycle inside each range, and
# the table is in a shard of its own.
SMALL_EXPERTS = 3
SMALL_SHAPES = {
    ("w1", "q"): (6, 4),
    ("w1", "s"): (6, 1),
    ("w2", "q"): (4, 6),
    ("w2", "s"): (4, 1),
    ("w3", "q"): (6, 4),
    ("w3", "s"): (6, 1),
}
SMALL_TABLE_SHAPES = {"weight": (24, 4), "scale": (24, 1)}


def _pattern(size: int, seed: int) -> bytes:
    """Deterministic bytes, so a run that is read from the wrong offset is caught.

    A payload of zeros would let an off-by-`data_offset` fill pass: the wrong bytes are real bytes of
    the right length, and zero is a perfectly ordinary fp8 code.
    """
    return bytes((seed + index * 7) % 251 for index in range(size))


def _write_shard(path: str, tensors: dict[str, tuple[str, tuple, bytes]]) -> None:
    """One safetensors shard in the format the reader parses: an 8-byte header length, the header
    JSON, then a data section that every `data_offsets` counts from."""
    header, payload, cursor = {}, bytearray(), 0
    for key, (dtype, shape, data) in tensors.items():
        header[key] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [cursor, cursor + len(data)],
        }
        payload += data
        cursor += len(data)
    blob = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        handle.write(bytes(payload))


def _synthetic_checkpoint(root: str) -> str:
    """A whole tiny checkpoint on disk: one expert layer and one Engram table, and nothing else.

    Written through the real reader rather than a fake one. The reader is what turns a safetensors
    header into the `begin`/`end` the fill `pread`s at, and the difference between the two -- the
    header length -- is the one bug a fake reader cannot catch. A dense tensor sits between the
    layer's two kinds, which is the hole the release has there and the thing the run boundary has to
    skip rather than copy.
    """
    shard = "model-00000-of-00002.safetensors"
    table_shard = "model-00001-of-00002.safetensors"

    tensors: dict[str, tuple[str, tuple, bytes]] = {}
    cursor = 0
    for kind in rb.KINDS:
        if kind == "q":
            tensors["layers.0.ffn.dense.weight"] = ("BF16", (8, 2), _pattern(32, 5))
            cursor += 32
        for expert in range(SMALL_EXPERTS):
            for which in rb.PROJECTIONS:
                shape = SMALL_SHAPES[(which, kind)]
                size = _nbytes(shape)
                tensors[f"layers.0.ffn.experts.{expert}.{which}."
                        f"{'weight' if kind == 'q' else 'scale'}"] = (
                    "I8" if kind == "q" else "F8_E8M0", shape, _pattern(size, cursor),
                )
                cursor += size

    table: dict[str, tuple[str, tuple, bytes]] = {}
    table_cursor = 0
    for leaf in rb.TABLE_LEAVES:
        shape = SMALL_TABLE_SHAPES[leaf]
        size = _nbytes(shape)
        table[f"layers.1.engram.embed.{leaf}"] = (
            "F8_E4M3" if leaf == "weight" else "F8_E8M0",
            shape,
            _pattern(size, 100 + table_cursor),
        )
        table_cursor += size
    # The rest of an Engram layer is in the same shard and is not the table; the scan has to leave it
    # behind, and a bank that copied the shard whole would be holding 27 GiB of the wrong tensors.
    table["layers.1.engram.wkv.weight"] = ("F8_E4M3", (4, 4), _pattern(16, 200))

    os.makedirs(root, exist_ok=True)
    _write_shard(os.path.join(root, shard), tensors)
    _write_shard(os.path.join(root, table_shard), table)
    weight_map = {key: shard for key in tensors} | {key: table_shard for key in table}
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as handle:
        json.dump(
            {"metadata": {"total_size": cursor + table_cursor + 16}, "weight_map": weight_map},
            handle,
        )
    return root


def test_the_engram_region_lands_after_the_experts_and_past_the_header() -> None:
    """Two regions, one segment: the tables come after every layer and their runs count the header.

    The order is not cosmetic. The segment is created once and attached by name, so a region that
    moved between the writer's arithmetic and a reader's would hand out a byte range that is real and
    wrong; `_validate_header` catches a change in the totals but not in where they sit.
    """
    entries = _synthetic_layer(0) | _synthetic_table(1) | _synthetic_table(14)
    layers, tables, size, n_experts = rb.layout(_FakeReader(entries, data_offset=258_144))

    assert [table.layer_id for table in tables] == [1, 14]
    assert layers[0].base == rb._HEADER_BYTES
    assert tables[0].base >= layers[0].base + layers[0].nbytes
    assert tables[1].base >= tables[0].base + tables[0].nbytes
    assert size >= tables[1].base + tables[1].nbytes
    # one shard per table, so the fill is two sequential reads and not a scatter
    assert len({run.file_name for table in tables for run in table.runs}) == 2
    # the offsets are file offsets, and the table is two runs whatever the release's own order was
    for table in tables:
        assert [run.file_offset for run in table.runs] == [258_144, 258_144 + TABLE_WEIGHT_BYTES]
        assert [run.bank_offset for run in table.runs] == [
            table.base, table.base + TABLE_WEIGHT_BYTES
        ]
        assert table.nbytes == TABLE_WEIGHT_BYTES + TABLE_SCALE_BYTES


def test_a_filled_bank_hands_back_what_the_shard_holds(tmp_path) -> None:
    """Fill both regions from a checkpoint on disk and read them back through the checkpoint.

    This is the wiring test, and the one that runs on a host with no `/mnt/data3`: `fill` into the
    segment, then `packed` and `rows` on the same keys before and after `attach_bank`. Before, they
    must come off the mapping with the shard's own bytes; after, off the segment with the same
    bytes -- asserted by provenance, because the two sources agreeing is what the first half
    established. The Engram half is the half that matters here: `rows` is called once per gather and
    it is where "no disk in the steady state" is either true or not.
    """
    from src.models.deepseek_v4_1.loader import V41Checkpoint

    root = _synthetic_checkpoint(os.path.join(str(tmp_path), "ckpt"))
    checkpoint = V41Checkpoint(root)
    layers, tables, size, n_experts = rb.layout(checkpoint.reader)
    assert len(layers) == 1 and len(tables) == 1 and n_experts == SMALL_EXPERTS

    segment = os.path.join(str(tmp_path), "shm")
    bank = rb.ResidentExpertBank(
        segment, layers, size, f"pocketllm_test_bank_{tmp_path.name}", create=True,
        n_routed_experts=n_experts, engram=tables,
    )
    try:
        assert bank.payload_bytes == sum(layer.nbytes for layer in layers) + tables[0].nbytes
        assert bank.fill(checkpoint) == bank.payload_bytes

        mismatched = []
        for layer in layers:
            for expert in range(SMALL_EXPERTS):
                for which, kind in bank.expert_views(layer.layer_id, expert):
                    key = (f"layers.{layer.layer_id}.ffn.experts.{expert}.{which}."
                           f"{'weight' if kind == 'q' else 'scale'}")
                    want = checkpoint.reader.entry_view(checkpoint.reader.entry(key))
                    if not torch.equal(bank.tensor(layer.layer_id, expert, which, kind),
                                       want.view(torch.uint8)):
                        mismatched.append(key)
        for leaf in rb.TABLE_LEAVES:
            key = f"layers.1.engram.embed.{leaf}"
            want = checkpoint.reader.entry_view(checkpoint.reader.entry(key))
            if not torch.equal(bank.table_tensor(1, leaf), want.view(torch.uint8)):
                mismatched.append(key)
        assert not mismatched, f"{mismatched} do not match the shard"

        # Before `attach_bank` the dense tensor and the table both come off the mapping, which is
        # what the un-resident run does and what the resident one has to keep doing for the tree.
        rows = torch.tensor([[0, 3, 3, 23]])
        want_rows = (checkpoint.reader.entry_view(checkpoint.reader.entry("layers.1.engram.embed.weight"))
                     .view(torch.uint8)[rows])
        assert torch.equal(checkpoint.rows("layers.1.engram.embed.weight", rows).view(torch.uint8),
                           want_rows)
        assert checkpoint.banked("layers.1.engram.embed.weight") is None
        assert checkpoint.banked("layers.1.engram.wkv.weight") is None

        checkpoint.attach_bank(bank)
        table = checkpoint.packed("layers.1.engram.embed.weight")
        assert table.data_ptr() == bank.table_tensor(1, "weight").data_ptr()
        assert torch.equal(checkpoint.rows("layers.1.engram.embed.weight", rows).view(torch.uint8),
                           want_rows)
        # A key the bank does not hold is answered from the mapping, not with an error and not with
        # a plausible zero: the dense tree is 16.79 GiB that is deliberately not in the segment.
        assert checkpoint.banked("layers.1.engram.wkv.weight") is None
        assert torch.equal(checkpoint.packed("layers.1.engram.wkv.weight").view(torch.uint8),
                           checkpoint.reader.entry_view(
                               checkpoint.reader.entry("layers.1.engram.wkv.weight")).view(torch.uint8))
    finally:
        bank.close(unlink=True)
    checkpoint.close()


def test_the_segment_outlives_the_process_that_filled_it(tmp_path) -> None:
    """A filled segment is per host boot, not per run: a child fills and exits, the parent attaches.

    `multiprocessing`'s resource tracker unlinks every shared segment the process that created it
    made, so without the explicit `unregister` in `_open` the 36.6 minute read of `/mnt/data3` would
    be paid again on every restart and `bank.ready` would mark a segment no run can attach to. The
    claim is cross-process, so the check is too -- a second object in the same process proves nothing.
    """
    from src.models.deepseek_v4_1.loader import V41Checkpoint

    root = _synthetic_checkpoint(os.path.join(str(tmp_path), "ckpt"))
    bank_dir = os.path.join(str(tmp_path), "shm")
    child = (
        "import sys\n"
        "from src.models.deepseek_v4_1.loader import V41Checkpoint\n"
        "from src.models.deepseek_v4_1 import resident_bank as rb\n"
        "checkpoint = V41Checkpoint(sys.argv[1])\n"
        "layers, tables, size, n = rb.layout(checkpoint.reader)\n"
        "name = rb._shm_name(checkpoint)\n"
        "bank = rb.ResidentExpertBank(rb._root_dir(sys.argv[2]), layers, size, name, create=True,\n"
        "                             n_routed_experts=n, engram=tables)\n"
        "assert bank.fill(checkpoint) == bank.payload_bytes\n"
        "bank.mark_ready()\n"
        "bank.close()\n"
        "print(name)\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    done = subprocess.run(
        [sys.executable, "-c", child, root, bank_dir], cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert done.returncode == 0, done.stderr
    name = done.stdout.strip().splitlines()[-1]
    assert name == rb._shm_name(V41Checkpoint(root))
    # The tracker is a separate process that unlinks on the child's exit; give it the moment it would
    # need, so that the failure this test is about is a failure and not a race.
    time.sleep(0.5)

    checkpoint = V41Checkpoint(root)
    layers, tables, size, n_experts = rb.layout(checkpoint.reader)
    segment = rb._segment_path(name)
    try:
        assert os.path.exists(segment), (
            f"{name} went away with the process that filled it, so the ready marker at "
            f"{os.path.join(bank_dir, 'bank.ready')} is a claim no run can act on"
        )
        bank = rb.ResidentExpertBank(
            bank_dir, layers, size, name, create=False, n_routed_experts=n_experts, engram=tables
        )
        try:
            for leaf in rb.TABLE_LEAVES:
                key = f"layers.1.engram.embed.{leaf}"
                want = checkpoint.reader.entry_view(checkpoint.reader.entry(key)).view(torch.uint8)
                assert torch.equal(bank.table_tensor(1, leaf), want), f"{key} did not survive"
        finally:
            bank.close(unlink=True)
        assert not os.path.exists(segment)
    finally:
        checkpoint.close()
        # Only reached when an assertion above already failed; the segment is 8 MiB and leaving it in
        # `/dev/shm` under a fixed name would leak into the next run of this test.
        if os.path.exists(segment):
            os.unlink(segment)


def test_a_ready_marker_with_no_segment_is_refilled(tmp_path, monkeypatch) -> None:
    """A marker is not the segment: a bank directory that outlived `/dev/shm` refills rather than
    failing to attach for the rest of the host's life.

    `DIR_ENV` is what makes this reachable. The default directory is itself under `/dev/shm`, so a
    reboot clears marker and segment together -- but a directory on disk keeps the marker and loses
    the segment, and `SharedMemory(name, create=False)` then raises `FileNotFoundError` on every
    start, forever, with a `bank.ready` file present to argue that it should not.
    """
    from src.models.deepseek_v4_1.loader import V41Checkpoint

    root = _synthetic_checkpoint(os.path.join(str(tmp_path), "ckpt"))
    bank_dir = os.path.join(str(tmp_path), "bank")
    os.makedirs(bank_dir)
    with open(os.path.join(bank_dir, "bank.ready"), "w") as handle:
        handle.write("ready\n")

    monkeypatch.setenv(rb.ENABLE_ENV, "1")
    checkpoint = V41Checkpoint(root)
    name = rb._shm_name(checkpoint)
    assert not os.path.exists(rb._segment_path(name))
    bank = rb.open_expert_bank(checkpoint, rank=0, root_dir=bank_dir)
    try:
        layers, tables, size, _ = rb.layout(checkpoint.reader)
        assert bank.size == size
        assert os.path.exists(rb._segment_path(name))
        key = "layers.1.engram.embed.weight"
        assert torch.equal(
            bank.table_tensor(1, "weight"),
            checkpoint.reader.entry_view(checkpoint.reader.entry(key)).view(torch.uint8),
        )
    finally:
        bank.close(unlink=True)
        checkpoint.close()


def test_a_table_is_read_in_whatever_order_the_release_stored_it() -> None:
    """Scales first is still one table: each run's offset comes from the index, not from a fixed
    leaf order, so the segment holds weight then scale either way and both are read correctly."""
    tables, cursor = rb._table_layout(
        _FakeReader(_synthetic_table(1, order=("scale", "weight")), data_offset=1_000), 0
    )
    table = tables[0]
    # the segment's own order is always weight then scale, whatever the shard did
    assert [run.nbytes for run in table.runs] == [TABLE_WEIGHT_BYTES, TABLE_SCALE_BYTES]
    assert table.slots["weight"][0] == 0 and table.slots["scale"][0] == TABLE_WEIGHT_BYTES
    # ... and the file offsets are the other way round, because the shard wrote the scales first
    assert [run.file_offset for run in table.runs] == [1_000 + TABLE_SCALE_BYTES, 1_000]
    assert cursor == rb._align_up(TABLE_WEIGHT_BYTES + TABLE_SCALE_BYTES)


def test_layout_rejects_half_a_table() -> None:
    """Codes with no scales cannot be dequantized, so holding half of one is not useful."""
    entries = _synthetic_layer(0) | _synthetic_table(1)
    entries.pop("layers.1.engram.embed.scale")
    with pytest.raises(ValueError, match="has no scale"):
        rb.layout(_FakeReader(entries))


def test_layout_rejects_a_table_split_across_shards() -> None:
    entries = _synthetic_layer(0) | _synthetic_table(1, files={"scale": "model-00048.safetensors"})
    with pytest.raises(ValueError, match="table spans 2 shards"):
        rb.layout(_FakeReader(entries))


def test_parse_engram_key_leaves_an_engram_layers_other_tensors_alone() -> None:
    """`wkv`, `q_weight` and `k_weight` live beside the table and are not part of it."""
    for leaf in rb.TABLE_LEAVES:
        assert rb.parse_engram_key(f"layers.1.engram.embed.{leaf}") == (1, leaf)
    for key in (
        "layers.1.engram.wkv.weight",
        "layers.1.engram.wkv.scale",
        "layers.1.engram.q_weight",
        "layers.1.engram.k_weight",
        "layers.1.engram.embed.weight.scale",
        "layers.1.ffn.experts.0.w1.weight",
        "layers.1.engram.weight",
        "engram.embed.weight",
    ):
        assert rb.parse_engram_key(key) is None, key


@pytest.mark.skipif(not os.path.isdir(CHECKPOINT), reason=f"no checkpoint at {CHECKPOINT}")
def test_a_filled_segment_matches_the_checkpoint_tensor_for_tensor() -> None:
    """Fill one real layer and compare all 2,304 tensors against the shard's own view of them.

    This is the test the offset arithmetic has to pass, and it is not a shape check: each comparison
    is `torch.equal` over bytes. A bank that placed experts numerically would fail on expert 10,
    which in the shard's order is the fourth expert and in the numeric order the eleventh.

    The second comparison is the wiring. With no bank attached `packed` goes to the shard mapping, so
    it has to hand back the shard's own bytes -- that is the function both the host path and the
    device path read through, and one that changed what it returns when a bank showed up would break
    both at once. Attaching afterwards is checked by provenance, further down.
    """
    from src.models.deepseek_v4_1.loader import V41Checkpoint

    checkpoint = V41Checkpoint(CHECKPOINT)
    layers, _, n_experts = rb._layout(checkpoint.reader)
    layer = layers[0]
    with tempfile.TemporaryDirectory() as root:
        bank = rb.ResidentExpertBank(
            root, [layer], rb._align_up(rb._HEADER_BYTES + layer.nbytes), "pocketllm_test_bank",
            create=True, n_routed_experts=n_experts,
        )
        try:
            bank.fill(checkpoint)
            kind_leaf = {"q": "weight", "s": "scale"}
            mismatched, via_mapping = [], []
            for expert in range(n_experts):
                for which in rb.PROJECTIONS:
                    for kind in rb.KINDS:
                        key = f"layers.0.ffn.experts.{expert}.{which}.{kind_leaf[kind]}"
                        want = checkpoint.reader.entry_view(checkpoint.reader.entry(key))
                        if not torch.equal(bank.tensor(0, expert, which, kind), want.view(torch.uint8)):
                            mismatched.append((expert, which, kind))
                        # No `attach_bank` yet, so this reads the shard; it must land on the same bytes.
                        if not torch.equal(checkpoint.packed(key).view(torch.uint8),
                                            want.view(torch.uint8)):
                            via_mapping.append((expert, which, kind))
            assert not mismatched, f"{len(mismatched)} tensors differ, first {mismatched[:4]}"
            assert not via_mapping, (
                f"{len(via_mapping)} tensors read differently through packed() than through the "
                f"reader, first {via_mapping[:4]}"
            )

            # And with the bank attached it reads the segment and not the shard. Asserted by identity
            # of the underlying storage rather than by a byte comparison, which the loop above already
            # made: the two sources agree, so only provenance tells them apart.
            checkpoint.attach_bank(bank)
            got = checkpoint.packed("layers.0.ffn.experts.0.w1.weight")
            assert got.data_ptr() == bank.tensor(0, 0, "w1", "q").data_ptr(), (
                "packed() returned bytes that do not live in the segment"
            )
        finally:
            bank.close(unlink=True)
    checkpoint.close()


@pytest.mark.skipif(not os.path.isdir(CHECKPOINT), reason=f"no checkpoint at {CHECKPOINT}")
def test_the_real_engram_runs_point_at_the_tables_and_not_at_the_shards() -> None:
    """The two tables, their sizes, and both ends of each of their four runs.

    Reading 188.83 GiB off the SMR disk to check an offset would be fifteen minutes, so this reads
    the first and last 4 KiB of every run through the same `pread` the fill uses and compares them
    against the shard's own view. A run built from `TensorEntry.begin` without the shard's header
    lands a whole header's length early, returns the requested number of bytes, and is wrong -- and
    at the end of a 91.55 GiB run it is wrong by 91.55 GiB rather than by 258 KiB, which is what the
    tail comparison is for.
    """
    from src.models.deepseek_v4_1.loader import V41Checkpoint

    checkpoint = V41Checkpoint(CHECKPOINT)
    try:
        _, tables, _, _ = rb.layout(checkpoint.reader)
        assert [table.layer_id for table in tables] == list(ENGRAM_LAYERS)
        assert sum(table.nbytes for table in tables) == ENGRAM_BYTES
        # every Engram layer in the index has a table, and nothing else produced one
        engram_keys = [key for key in checkpoint.reader.entries
                       if key.startswith("layers.") and ".engram." in key]
        assert {int(key.split(".")[1]) for key in engram_keys} == set(ENGRAM_LAYERS)
        assert {key for key in engram_keys if rb.parse_engram_key(key) is not None} == {
            f"layers.{layer}.engram.embed.{leaf}"
            for layer in ENGRAM_LAYERS
            for leaf in rb.TABLE_LEAVES
        }
        # ... and the rest of an Engram layer is in the same shards without being part of a table
        assert len(engram_keys) > 2 * len(ENGRAM_LAYERS)

        checked = []
        for table in tables:
            for leaf, run in zip(rb.TABLE_LEAVES, table.runs):
                key = f"layers.{table.layer_id}.engram.embed.{leaf}"
                entry = checkpoint.reader.entry(key)
                assert run.file_offset == checkpoint.reader.data_offset(entry.file_name) + entry.begin
                assert run.nbytes == entry.nbytes
                flat = checkpoint.reader.entry_view(entry).view(torch.uint8).reshape(-1)
                for offset, want in (
                    (run.file_offset, flat[:4096]),
                    (run.file_offset + run.nbytes - 4096, flat[-4096:]),
                ):
                    assert torch.equal(_pread_window(checkpoint.root, run.file_name, offset, 4096),
                                       want), f"{key} differs at file offset {offset}"
                checked.append((key, run.nbytes))
        assert dict(checked) == {
            "layers.1.engram.embed.weight": 98_305_579_008,
            "layers.1.engram.embed.scale": 3_072_049_344,
            "layers.14.engram.embed.weight": 98_308_270_592,
            "layers.14.engram.embed.scale": 3_072_133_456,
        }
    finally:
        checkpoint.close()


@pytest.mark.skipif(not os.path.isdir(CHECKPOINT), reason=f"no checkpoint at {CHECKPOINT}")
def test_the_real_layout_is_what_the_release_holds() -> None:
    """40 layers, 384 experts, 18,800,640 bytes each, 80 runs, one shard per layer."""
    from src.models.deepseek_v4_1.loader import V41Checkpoint

    checkpoint = V41Checkpoint(CHECKPOINT)
    try:
        layers, size, n_experts = rb._layout(checkpoint.reader)
    finally:
        checkpoint.close()

    assert n_experts == EXPERTS and len(layers) == 40
    assert {layer.nbytes for layer in layers} == {LAYER_BYTES}
    assert sum(len(layer.runs) for layer in layers) == 80
    assert size >= 40 * LAYER_BYTES
    # one shard per layer, and no shard shared between two layers
    per_layer = [layer.runs[0].file_name for layer in layers]
    assert len(set(per_layer)) == 40
    # the derived segment is the expert bytes plus the header and at most one alignment pad a layer
    assert size - 40 * LAYER_BYTES <= rb._HEADER_BYTES + 40 * rb._ALIGN


def test_the_environment_variable_is_what_turns_it_on(monkeypatch) -> None:
    monkeypatch.delenv(rb.ENABLE_ENV, raising=False)
    assert rb.enabled() is False
    monkeypatch.setenv(rb.ENABLE_ENV, "1")
    assert rb.enabled() is True
    monkeypatch.setenv(rb.ENABLE_ENV, "no")
    assert rb.enabled() is False
