"""A decode step with each block captured, split around the one call that cannot be.

`Block.forward` is 213 kernel launches and 244 host API calls for one token, and device-busy is 11%
of it: the step is not short of arithmetic, it is short of work to hide the launches behind. A CUDA
graph removes the launches, and one graph per block is the largest scope a decode step allows --
what it is split around is not a choice:

    graph A  ->  eager expert call  ->  graph B        (per layer, forty times)

`routed.forward` cannot be captured. `_route_ids` does a pinned device-to-host copy and then
`torch.cuda.current_stream(...).synchronize()`, and `_resolve_row`'s `[int(e) for e in
ids_row.tolist()]` is a host read per row (`device_experts.py`) -- that is the *point* of a
host-resident expert bank, so the host has to act in the middle of the step and the step cannot be
one graph. **A** is everything up to and including the gate, **B** is the shared expert, the
tensor-parallel reduce, the residual fold and the block's `hc_post`; between them `MoE.forward`'s
routed line runs as it always did.

**The position has to reach the card as a tensor, and `decode_pos.Pos` is how.** `start_pos` picks
the `freqs_cis` row, the window ring's slot, and the compressed group a step writes; every one of
those is a Python value at *record* time, so a capture would freeze the step it was recorded at.
With `Pos` they are index tensors the graph reads, and the graphs are the same graphs at every
position. Two things about the position stay on the host, and both are answered before the launch
rather than inside it: `first()` (which picks the prefill body, and prefill is not graphed) and
`emits(ratio)` (which picks the compressor's variant, below).

**A compressor that emits has a different body from one that fills, so those layers get two A
graphs.** `Compressor.forward` runs its quantizer and writes the compressed cache only when
`(start_pos + 1) % ratio == 0`; on the other steps it fills a slot in `kv_state` and returns None.
That is a branch inside A, so it is a branch inside the recording, and `Compressor.compress_variant`
is what lets both bodies be recorded: `True` records the emitting body, `False` the filling one, and
the replay picks between the two recordings from the step's own position. Only the layers in
`kv_source_layers` carry a compressor at all -- four of the forty -- and only the ones whose ratio
is above 1 have two bodies, which is three of them here.

**The capture pass is a real step, and then it is rewound.** Each layer's graphs are recorded from
an activation the model really produced, which means the recording is driven by a forward that has
to happen anyway; the per-layer caches are snapshotted before it and copied back after, so the step
that *follows* starts from the same state the eager column's first step starts from. Without the
rewind the capture pass's own cache writes would be the graphed column's starting state and the two
columns would be one step apart, which on a recurrent stack reads exactly like a wrong graph. The
capture pass's own logits are discarded rather than returned: the caches no longer describe the
forward that produced them, so keeping them would be inconsistent with the state the next step runs
on.

**The graphs share one pool, and the pool is re-made for every capture round.** `graph_pool_handle`
is a bare `(id, generation)` pair with no lifetime at all (`torch/cuda/graphs.py`): the allocator's
entry for an id lives only as long as some graph captured into it, so a round whose graphs are all
gone leaves nothing for the next `capture_begin`'s `create_or_incref_pool` to incref and it asserts
(`CUDACachingAllocator.cpp`, `TORCH_INTERNAL_ASSERT(it->second->use_count > 0)`). A fresh handle per
round costs nothing -- one round's graphs are replayed together and never share memory with another
round's -- and `release` also gives the dead pool back with `empty_cache`. Sharing one pool across
all forty layers is what the pool is for and what makes the cost a per-round cost rather than a
per-layer one: 0.254 GiB for the whole tree, against a 3.7 GiB-per-layer projection if each graph
had a pool of its own.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable

import torch

from src.models.deepseek_v4_1.decode_pos import Pos

# The cache set and the walk over it live in `prefix_cache`, which uses the same names for the store
# a request leaves behind for the next one; a capture pass is the other caller of that walk, and the
# names are re-exported here because this module was where they were defined.
from src.models.deepseek_v4_1.prefix_cache import CACHE_NAMES, restore, snapshot

__all__ = ["CACHE_NAMES", "DecodeGraphs", "LayerGraphs", "Sink", "restore", "snapshot"]

# How many real bodies run on a side stream before each recording. They are not a correctness
# device -- the recorded body is what runs on replay -- they are what the caching allocator needs: a
# capture wants every allocation it will make already in the free list, or the recording grows the
# pool a block at a time. Measured as the right convention by `probe_graph_sweep.py
# --align-extra 0`.
CAPTURE_WARMUP = 2


class Sink:
    """The per-layer buffers a captured step reads and writes.

    Every slot is allocated by the first real pass and copied into afterwards. Allocating them
    outside a capture is not a preference -- a tensor allocated *inside* one lives in the graph's
    private pool, and after `capture_end` returns that memory is the pool's and not the tensor's, so
    reading it back gives whatever the pool holds next. The graphs also hold raw pointers into these
    buffers, which is the same reason they are `copy_`-ed into and never rebound.
    """

    __slots__ = (
        "x_in",
        "pre_in",
        "residual",
        "ffn_pre",
        "ffn_post",
        "ffn_comb",
        "ffn_in",
        "weights",
        "indices",
        "y_in",
        "x_out",
    )

    def __init__(self) -> None:
        for name in self.__slots__:
            setattr(self, name, None)


def _put(sink: Sink, key: str, value: torch.Tensor) -> torch.Tensor:
    """Record one stage's output and hand the buffer back.

    The first pass allocates -- a clone, so the value survives the next stage overwriting its
    source -- and every pass after that copies into the same buffer. The clone is the whole of what
    makes the sink a snapshot rather than an alias: `_body_a`'s `residual` and `x_in` are the same
    tensor at the top of the layer, and B reads `residual` after the routed call has been given
    `ffn_in`, so a slot that aliased its source would hold whatever that source holds *now*.
    """
    got = getattr(sink, key)
    if got is None:
        got = value.detach().clone()
        setattr(sink, key, got)
    else:
        got.copy_(value)
    return got


def _record(body: Callable[[], None], pool, warmup: int = CAPTURE_WARMUP) -> torch.cuda.CUDAGraph:
    """Record `body` into `pool`, after running it for real a few times on a side stream.

    The warm bodies execute for real and their results are discarded. What they write to the caches
    is either the same value the real pass wrote (the ring slot and the group index do not depend on
    which compressor variant is running) or a row the read masks out (`_compress_kv` writes group
    `pos.row() // ratio` and the mask admits `(pos.row() + 1) // ratio` of them, so a forced emitting
    body at a filling position writes exactly the first invisible one). `DecodeGraphs.capture_pass`
    rewinds the caches regardless, so neither case is load-bearing.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            body()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        body()
    return graph


class LayerGraphs:
    """One block's decode forward, cut at the expert call, with each half a captured graph.

    `Block.decode_graph` holds one of these and `Block.forward` hands the layer to it, so the
    chaining of the layers stays `Backbone.forward`'s loop and nothing here advances anything
    itself. Two entry points: `step`, which is what `Block.forward` calls, and `_capture`, which
    `step` calls on the first one.

    A layer whose compressor can emit has two A bodies and therefore two A graphs, picked per step
    from `pos.emits(ratio)` -- a host answer, because which recording runs is decided before the
    launch. B is one graph either way: the compressor is inside A, so the variant changes B's input
    values and never its shape.
    """

    def __init__(self, block: torch.nn.Module, pool, warmup: int = CAPTURE_WARMUP) -> None:
        self.block = block
        self.pool = pool
        self.warmup = warmup
        self.sink = Sink()
        self.compressor = block.attn.compressor
        self.ratio = 0 if self.compressor is None else self.compressor.compress_ratio
        # Only a ratio above 1 has an emitting body to record: `Compressor.forward` returns
        # `self.norm(self.wkv(x))` outright at ratio 1, before anything the position could branch.
        self.dual = self.ratio > 1
        self.graph_a: torch.cuda.CUDAGraph | None = None
        self.graph_a_emit: torch.cuda.CUDAGraph | None = None
        self.graph_a_fill: torch.cuda.CUDAGraph | None = None
        self.graph_b: torch.cuda.CUDAGraph | None = None
        # The `SharedAttentionRuntime` the recordings were made against, held so that the buffers
        # `publish` put `topk_idxs` and `candidates` in cannot go back to the allocator. That is not
        # a precaution, it is the whole reason this attribute exists: the runtime is a local of
        # `Backbone.forward` and dies with the step that recorded, while every recording that wrote
        # a published value holds the raw address of its buffer. Measured, not reasoned about --
        # `DSV41_ENGRAM_TRACE` caught layer 2's replay overwriting the step's own hash ids with
        # `256..351`, which is this layer's top-k (the compressed positions `128..639`) read 128
        # elements in: the 2048-byte publish buffer had been split into an 8-byte tensor, the 384-
        # byte hash ids and a free remainder, and the hash ids landed on bytes 512..896 of it.
        self.runtime = None
        self.pool_bytes = 0
        # The three host-side clocks, summed over the layers by `DecodeGraphs.marks`. Indicative
        # only: the routed call synchronizes, so graph A's tail lands in `routed` and the step total
        # is the number that means anything.
        self.marks = {"a": 0.0, "routed": 0.0, "b": 0.0}

    @property
    def ready(self) -> bool:
        """Whether every graph this layer needs has been recorded."""
        return self.graph_b is not None and (self.dual or self.graph_a is not None)

    # -- the entry point -------------------------------------------------------------------------

    def step(
        self,
        x: torch.Tensor,
        start_pos: "int | Pos",
        pre_mix: torch.Tensor,
        image_mask: torch.Tensor | None,
        shared,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`Block.forward`'s contract, replayed. Records the graphs on the first call."""
        if image_mask is not None:
            # The gate reads the mask through `torch.where`, which a recording can hold, but the
            # mask a step carries is per-position and the graph path has no buffer for it yet. The
            # text path is the one that exists; refusing is better than recording a body that
            # silently ignores it.
            raise ValueError("the decode graphs are recorded for a text forward and take no image mask")
        sink = self.sink
        _put(sink, "x_in", x)
        _put(sink, "pre_in", pre_mix)
        pos = Pos.of(start_pos)
        if not self.ready:
            self._capture(pos, shared)
        return self._replay(pos, shared)

    # -- the two bodies --------------------------------------------------------------------------

    def _body_a(self, pos: "int | Pos", shared) -> None:
        """`Block.forward` up to and including the gate.

        Ends with everything `MoE.forward` needs to make its routed call -- the ffn input, and the
        gate's own `(weights, indices)` -- which is the last thing before the split, because the
        call itself is the one thing that has to run on the host.
        """
        block, sink = self.block, self.sink
        x, pre_mix = sink.x_in, sink.pre_in
        residual = x
        attn_pre, attn_post, attn_comb = block.hc_mixes(
            x, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
        )
        h = block.hc_pre(x, pre_mix)
        h = block.attn_norm(h)
        h = block.attn(h, pos, shared)
        h = block.hc_post(h, residual, attn_post, attn_comb)
        # B's `hc_post` folds the residual back in, and by then `x_in` is long overwritten.
        _put(sink, "residual", h)

        ffn_pre, ffn_post, ffn_comb = block.hc_mixes(
            h, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base
        )
        _put(sink, "ffn_pre", ffn_pre)
        _put(sink, "ffn_post", ffn_post)
        _put(sink, "ffn_comb", ffn_comb)
        h = block.hc_pre(h, attn_pre)
        h = block.ffn_norm(h)
        ffn_in = _put(sink, "ffn_in", h)
        # `MoE.forward` views to (rows, dim) before the gate; the view is metadata, so B takes it
        # from the same buffer rather than carrying a second copy of it.
        weights, indices = block.ffn.gate(ffn_in.view(-1, block.ffn.dim), None)
        _put(sink, "weights", weights)
        _put(sink, "indices", indices)

    def _routed(self) -> None:
        """`MoE.forward`'s routed line: the eager half, and the only thing between the two replays.

        `.forward(...)` rather than a call: `RoutedExperts` is an interface, and a store that keeps
        its experts in host memory is not an `nn.Module`, so it has no `__call__` to fall back on.
        `MoE.forward` spells the call the same way.
        """
        sink, ffn = self.sink, self.block.ffn
        _put(sink, "y_in", ffn.routed.forward(sink.ffn_in.view(-1, ffn.dim), sink.weights, sink.indices))

    def _body_b(self) -> None:
        """`MoE.forward` from the shared expert down, plus the block's own fold."""
        block, sink = self.block, self.sink
        ffn = block.ffn
        view = sink.ffn_in.view(-1, ffn.dim)
        y, shared_out = sink.y_in, ffn.shared_experts(view)
        tp = ffn.tp
        if tp is None:
            y = y + shared_out
        elif ffn.routed.partial:
            y = tp.reduce(y + shared_out)
        else:
            y = y + tp.reduce(shared_out)
        y = y.type_as(view).view(sink.ffn_in.shape)
        _put(sink, "x_out", block.hc_post(y, sink.residual, sink.ffn_post, sink.ffn_comb))

    # -- capture and replay ----------------------------------------------------------------------

    def _capture(self, pos: "int | Pos", shared) -> None:
        """Record both halves. The step's own value comes from the replay that follows."""
        # Held for as long as the recordings live, and read by every layer because they all recorded
        # against the same one. See the attribute's own comment: a published buffer whose runtime is
        # only referenced by the capture's closure is freed at the end of that forward, and the next
        # step's allocations take the block. Recording happens once per layer, so this assigns once.
        self.runtime = shared
        # The real bodies run first and they are not decoration: they are what allocates every sink
        # slot outside the pool, and every recording is made against the addresses they hand out.
        self._body_a(pos, shared)
        self._routed()
        self._body_b()
        before = torch.cuda.memory_allocated()
        if self.dual:
            # Both bodies, whichever one this position would have picked. `compress_variant` is what
            # makes the choice a parameter rather than a host answer read inside the body.
            self.compressor.compress_variant = True
            self.graph_a_emit = _record(self._body_a_for(pos, shared), self.pool, self.warmup)
            self.compressor.compress_variant = False
            self.graph_a_fill = _record(self._body_a_for(pos, shared), self.pool, self.warmup)
            self.compressor.compress_variant = None
        else:
            self.graph_a = _record(self._body_a_for(pos, shared), self.pool, self.warmup)
        self.graph_b = _record(self._body_b, self.pool, self.warmup)
        self.pool_bytes = torch.cuda.memory_allocated() - before

    def _body_a_for(self, pos: "int | Pos", shared) -> Callable[[], None]:
        """`_body_a` bound to one step's arguments, for a recording that takes none."""

        def body() -> None:
            self._body_a(pos, shared)

        return body

    def _graph_for(self, pos: Pos) -> torch.cuda.CUDAGraph:
        """Which recording this step replays. A host answer, and the reason `emits` is on `Pos`."""
        if not self.dual:
            return self.graph_a
        return self.graph_a_emit if pos.emits(self.ratio) else self.graph_a_fill

    def _replay(self, pos: Pos, shared) -> tuple[torch.Tensor, torch.Tensor]:
        sink = self.sink
        t0 = time.perf_counter()
        self._graph_for(pos).replay()
        t1 = time.perf_counter()
        self._routed()
        t2 = time.perf_counter()
        self.graph_b.replay()
        t3 = time.perf_counter()
        self.marks["a"] += t1 - t0
        self.marks["routed"] += t2 - t1
        self.marks["b"] += t3 - t2
        return sink.x_out, sink.ffn_pre


class DecodeGraphs:
    """The per-layer graphs a decode step replays, and the pass that records them.

    Installing one sets `block.decode_graph` on every layer it covers, and `Block.forward` hands
    each of them to its own `LayerGraphs` from then on. Everything outside the blocks -- the
    embedding, the Engram gather, the final `hc_pre`, the norm and the head -- stays eager, which is
    what keeps a decode step's graph shapes fixed: one token in, one token out, no matter where in
    the sequence it is.
    """

    def __init__(
        self, model: torch.nn.Module, layer_ids: Iterable[int] | None = None, warmup: int = CAPTURE_WARMUP
    ) -> None:
        self.model = model
        self.warmup = warmup
        self.pool = torch.cuda.graph_pool_handle()
        self.captured = False
        self.layers: list[LayerGraphs] = []
        wanted = None if layer_ids is None else set(layer_ids)
        for block in model.layers:
            if wanted is not None and block.layer_id not in wanted:
                continue
            graphs = LayerGraphs(block, self.pool, warmup)
            block.decode_graph = graphs
            self.layers.append(graphs)

    def release(self) -> None:
        """Uninstall and hand the pool back. A round after this needs a new `DecodeGraphs`.

        The graphs are dropped before `empty_cache` so their pool's memory is freeable at the next
        release, and `block.decode_graph` is cleared so a layer that outlives the graphs runs the
        eager body again rather than replaying into memory nothing owns.
        """
        for graphs in self.layers:
            graphs.block.decode_graph = None
        self.layers = []
        self.captured = False
        self.pool = None
        torch.cuda.empty_cache()

    # -- what the caller reports with ------------------------------------------------------------

    @property
    def pool_bytes(self) -> int:
        """What the recordings cost, summed over the layers. They share one pool, so this is the
        pool's own size and not a per-layer figure that could be multiplied by forty."""
        return sum(graphs.pool_bytes for graphs in self.layers)

    def marks(self) -> dict[str, float]:
        """The three host-side clocks, summed over the layers. Indicative; see `LayerGraphs.marks`."""
        out = {"a": 0.0, "routed": 0.0, "b": 0.0}
        for graphs in self.layers:
            for key in out:
                out[key] += graphs.marks[key]
        return out

    def reset_marks(self) -> None:
        for graphs in self.layers:
            graphs.marks = {"a": 0.0, "routed": 0.0, "b": 0.0}

    # -- the capture pass ------------------------------------------------------------------------

    def capture_pass(self, forward: Callable[[], object]) -> None:
        """Record every graph from one real step, then rewind the caches.

        `forward` is the model's own forward -- `LoadedBackbone.__call__`, one token at one
        position -- and it is what drives each layer's `LayerGraphs.step`, which records instead of
        replaying the first time it is reached. Recording it from a real step is not an
        optimization: a body recorded from a synthetic activation is a body recorded against
        synthetic addresses and shapes, and the whole point of capturing per layer is that the
        activation is the one the model actually produces.

        The caches are put back afterwards, so the step that follows is the step the eager column
        would have taken. What this returns is nothing on purpose: the forward's own logits belong
        to a state that no longer exists once the caches are rewound, and a caller that used them
        would be comparing against a step that did not happen.
        """
        saved = snapshot(self.model)
        try:
            forward()
        finally:
            restore(self.model, saved)
        self.captured = True
