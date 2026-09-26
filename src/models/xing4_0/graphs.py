"""A Xing4.0-29B-A4B decode step, captured whole and replayed a bucket at a time.

Stage 1 of this work measured what such a capture is worth before any of it was written
(`docs/performance/xing4_0_decode_launch_gap.md`): a step at a frozen position is **37.9 ms against
an eager 178.6**, the graph submits no `cudaLaunchKernel` at all, and the logits are bit-identical.
The two things that measurement did not have and this module supplies are the two things a *loop*
needs:

* **the position**, which `decode_pos.Pos` carries to the card as an index tensor so that a capture
  is not frozen at the step it was recorded at; and
* **the cache width**, which the attention's score GEMM reads as its `N` and which therefore has to
  be a constant inside a recording. It is rounded up to a bucket and the rows past the position are
  masked out, so a step at position 4097 reads 8192 rows and hides 4095 of them.

## One graph a bucket, not one graph a position

The bucket is chosen on the host, before the launch, exactly where `deepseek_v4_1` chooses between
its compressor's two bodies: which recording runs is not something a recording can decide. The
ladder is the powers of two from `floor` up to the cache's capacity, so a step wastes at most a
factor of two on the attention's read and never more — a 4097-row cache is read at 8192 and a
8192-row one at 8192.

**At a bucket end the capture is not an approximation.** A step at position `B - 1` reads a
`B`-wide cache, which is exactly the width it would read unbucketed, and the mask hides nothing —
so its logits are bit-identical to the eager path's, not close to them.
`tests/test_xing4_0_decode_graph.py` asserts that identity at every rung of the ladder it can reach,
which is the strongest statement the mechanism admits.

## What is captured, and what is rewound

The body is one `Xing4_0GGUFModel.forward` with one token at one position, and it is a *real* step:
`_Step.capture` runs it for real first, on this stream and then on a side stream, and only then
records it. That is not a formality — it is what allocates every buffer the recording will touch
outside the pool, which `torch.cuda.graph` needs and a body recorded from synthetic tensors would not
have done.

Recording runs the body four times over, and every one of them writes the same cache row. The row is
snapshotted before and put back after, so a capture leaves the cache exactly as it found it and the
step that follows starts where the eager column's step would. The capture's own logits are discarded
for the same reason: they describe a state that has been rewound.

## What it costs, and where

Recording a rung is four forwards at that width — one real, `warmup` on a side stream, one captured.
At this checkpoint's measured eager rate that is ~0.7 s a rung, and it happens on the **first decode
step** of the first request that needs the rung. `reserve` is how a caller tells the holder every
rung a run can reach, so the cost lands in one step rather than in whichever step happens to cross a
boundary — and it lands in `Generation.first_step_seconds`, which the record and the bench already
report apart from the steady rate for exactly this kind of one-off.

Rungs are recorded once and live as long as the holder, so a served process pays this on its first
request and no request after it. `release` is what ends that.
"""

from __future__ import annotations

import time

import torch

from src.models.xing4_0.decode_pos import Pos

__all__ = ["CAPTURE_WARMUP", "MIN_BUCKET", "DecodeGraphs", "bucket_ladder"]

MIN_BUCKET = 64
"""The narrowest cache a graph is recorded for.

Below this the read is a few MiB and the eager step's own overhead is what the width would have been
hiding; more to the point, a rung costs a capture, and capturing 64-row rungs for a 32K context buys
six of them to save nothing. A step that needs less than this reads the 64-row rung.
"""

CAPTURE_WARMUP = 2
"""Real bodies run on a side stream before each recording.

Not a correctness device -- the recorded body is what runs on replay -- but what the caching
allocator needs: a capture wants every allocation it will make already in the free list, or the
recording grows the pool a block at a time. Same convention and same value as
`deepseek_v4_1.graphs.CAPTURE_WARMUP`, which measured it there.
"""


def bucket_ladder(capacity: int, floor: int = MIN_BUCKET) -> list[int]:
    """The cache widths a step may be captured at: powers of two, then the capacity itself.

    The last rung is the capacity and not the next power of two above it, because a slice cannot
    read past the buffer: at a capacity of 8192 the rungs end at 8192, and at 30000 they end at
    30000 rather than at 32768. A rung that lands exactly on a power of two appears once.
    """
    capacity = int(capacity)
    if capacity < 1:
        raise ValueError(f"a cache of {capacity} positions is not a cache")
    rungs: list[int] = []
    width = max(1, int(floor))
    while width < capacity:
        rungs.append(width)
        width *= 2
    rungs.append(capacity)
    return rungs


class _Step:
    """One captured decode step, at one fixed cache width.

    Every buffer a replay reads or writes lives outside the graph's pool and is never rebound: the
    recording holds raw addresses, so the position and the token are `fill_`-ed in place and the
    logits are `copy_`-ed into a tensor allocated before the capture began.
    """

    __slots__ = ("model", "cache", "width", "pool", "warmup", "pos", "token", "logits", "graph")

    def __init__(self, model, cache, width: int, pool, device, warmup: int) -> None:
        self.model = model
        self.cache = cache
        self.width = int(width)
        self.pool = pool
        self.warmup = max(1, int(warmup))
        self.pos = Pos.device(0, device, self.width)
        self.token = torch.zeros(1, dtype=torch.int64, device=device)
        self.logits: torch.Tensor | None = None
        self.graph: torch.cuda.CUDAGraph | None = None

    def _body(self) -> None:
        """One decode forward, at wherever `self.pos` and `self.token` currently are.

        The first call allocates the logits buffer — outside any capture — and every call after it
        copies into that same buffer, which is the whole of what makes the sink a snapshot rather
        than an alias into the pool.
        """
        out = self.model.forward(self.token, cache=self.cache, start_pos=self.pos)
        if self.logits is None:
            self.logits = out.detach().clone()
        else:
            self.logits.copy_(out)

    def capture(self, token_id: int, start_pos: int) -> tuple[float, int]:
        """Record the body at `(token_id, start_pos)`, leaving the cache as it found it.

        Returns the recording's seconds and the bytes the pool grew by. The bytes are this rung's
        *increment* -- the rungs share one pool -- so a caller summing them gets the pool, and the
        accounting works because nothing is allocated between one rung's `capture_end` and the next
        rung's baseline.
        """
        self.pos.set(start_pos)
        self.token.fill_(int(token_id))
        row = int(start_pos)
        # What the recording writes, and the only thing in the model a capture mutates: one row a
        # layer, because a decode step appends one token. Cloning the whole cache would be 1.5 GiB
        # at a 32K context to put back 46 KB of it.
        saved = [(layer, layer.latent[:, row : row + 1].clone()) for layer in self.cache]
        started = time.perf_counter()
        try:
            self._body()
            torch.cuda.synchronize()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(self.warmup):
                    self._body()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            # The baseline is read here, after the bodies and after an `empty_cache`, and both halves
            # of that order are load-bearing. `torch.cuda.graph.__enter__` calls `empty_cache` before
            # `capture_begin`, so a baseline taken with the bodies' transients still resident is a
            # baseline the capture throws away and the delta comes out *negative* -- measured at
            # -2 MiB on the 64 rung before this line. Reading it at the allocator's own floor is what
            # makes `grew` the pool rather than the difference between two floors.
            torch.cuda.empty_cache()
            before = torch.cuda.memory_stats()["reserved_bytes.all.current"]
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, pool=self.pool):
                self._body()
            torch.cuda.synchronize()
            grew = torch.cuda.memory_stats()["reserved_bytes.all.current"] - before
        finally:
            for layer, rows in saved:
                layer.latent[:, row : row + 1] = rows
        return time.perf_counter() - started, grew


class DecodeGraphs:
    """The captured decode steps a run replays, one per cache-width bucket.

    Constructed against one cache, because the graphs hold that cache's addresses. Every replay
    writes the row its position names, so the graphs outlive a request: a caller that keeps its cache
    — a server does, and so does the bench — keeps one holder for its whole life and records each
    rung once.
    """

    def __init__(
        self,
        model,
        cache,
        *,
        device: torch.device | str | None = None,
        floor: int = MIN_BUCKET,
        warmup: int = CAPTURE_WARMUP,
    ) -> None:
        self.model = model
        self.cache = cache
        layers = cache if isinstance(cache, list) else [cache]
        if not layers:
            raise ValueError("a decode graph needs the cache it will replay against")
        device = torch.device(device) if device is not None else layers[0].latent.device
        # Before anything else, and not cosmetic: `torch.cuda.graph` captures on the *current*
        # device's stream, while the ops inside a body launch on the stream of the device their
        # tensors are on. Naming a card in a command line does not set the current device, and with
        # the current device left at 0 and the weights on 2 every capture fails with
        # `cudaErrorStreamCaptureUnsupported` reported at whatever call happened to come next.
        # `scripts/probe_xing4_0_decode_graph.py` hit exactly that before this line existed.
        torch.cuda.set_device(device.index)
        self.device = device
        self.capacity = min(int(layer.capacity) for layer in layers)
        self.ladder = bucket_ladder(self.capacity, floor)
        self.warmup = int(warmup)
        # One handle for every rung, and a fresh one per holder. A pool's id has no lifetime of its
        # own (`torch/cuda/graphs.py`: a bare `(id, generation)` pair), so a round after `release`
        # needs a new holder rather than a new capture into a dead pool — `capture_begin`'s
        # `create_or_incref_pool` asserts on the freed entry.
        self.pool = torch.cuda.graph_pool_handle()
        self.steps: dict[int, _Step] = {}
        self.reserved: int = 0
        self.capture_seconds = 0.0
        self.replay_seconds = 0.0
        self.replays = 0
        self.pool_bytes = 0
        self.rung_bytes: dict[int, int] = {}

    # -- what a caller tells it, and what it reports -------------------------------------------- #

    def reserve(self, upto: int) -> None:
        """Every rung a run reaching position `upto` can need, to be recorded at the first step.

        A hint and not a contract: a rung nobody reserved is recorded when a step first needs it, so
        a caller that skips this pays for its rungs one boundary at a time rather than wrongly.
        """
        self.reserved = max(self.reserved, int(upto))

    def bucket_for(self, positions: int) -> int:
        """The narrowest rung that can hold `positions` rows."""
        for width in self.ladder:
            if width >= positions:
                return width
        raise ValueError(
            f"{positions} positions need a cache wider than this holder's {self.capacity}; the "
            f"cache was sized for fewer tokens than the run is asking for"
        )

    @property
    def recorded(self) -> list[int]:
        """The rungs recorded so far, narrowest first. What a test or a log line reads."""
        return sorted(self.steps)

    @property
    def replay_millis(self) -> float:
        """The average host cost of one `replay()` submission, or zero before any.

        A per-replay average and not a cumulative total, because the total is a property of the run
        and this is a property of the step — the number that says whether the host can keep up with
        the card once the card is what the step costs.
        """
        return 0.0 if not self.replays else self.replay_seconds / self.replays * 1000

    # -- the step ------------------------------------------------------------------------------- #

    def step(self, token_id: int, cache, start_pos: int) -> torch.Tensor:
        """One decode forward, replayed. Records the rungs it needs on the first call.

        Returns the logits in `Xing4_0GGUFModel.forward`'s own shape, as a view of the step's sink
        buffer — the next replay of the same rung overwrites it, which is the contract
        `sample_token` already relies on when it reads the row it was handed.
        """
        entry = self._entry(token_id, cache, start_pos)
        started = time.perf_counter()
        entry.graph.replay()
        self.replay_seconds += time.perf_counter() - started
        self.replays += 1
        self._advance(cache, start_pos)
        return entry.logits

    def step_eager(self, token_id: int, cache, start_pos: int) -> torch.Tensor:
        """The same step at the same bucket width, submitted one launch at a time.

        `step` and this differ in exactly one thing — whether the ops are replayed or launched —
        which makes this the arm that separates what the *bucket* costs from what the *graph* buys.
        Without it "a graphed step is 4.7x faster" would be a claim about two changes at once: a
        step read at `width` rows instead of at its length is not the eager step either.

        It is also the parity oracle. `tests/test_xing4_0_decode_graph.py` asserts the replayed step
        is bit-identical to this one at every position it checks, which is a statement about the
        capture rather than about the bucket; and a served run whose two disagree has a way to find
        out which of them moved. It records nothing, so it is usable before any rung exists.
        """
        return self.model.forward(
            [int(token_id)],
            cache=cache,
            start_pos=Pos.bucket(int(start_pos), self.bucket_for(int(start_pos) + 1)),
        )

    def _entry(self, token_id: int, cache, start_pos: int) -> "_Step":
        if cache is not self.cache:
            raise ValueError(
                "these graphs hold the addresses of the cache they were recorded against; a step "
                "against a different cache would replay into the wrong buffer and answer wrong"
            )
        start_pos = int(start_pos)
        need = start_pos + 1
        width = self.bucket_for(need)
        if width not in self.steps:
            self._record(self._wanted(need, start_pos), token_id, start_pos)
        entry = self.steps[width]
        entry.pos.set(start_pos)
        entry.token.fill_(int(token_id))
        return entry

    def _advance(self, cache, start_pos: int) -> None:
        """`start_pos + 1` into every layer's `length`.

        `KVLatentCache.append` cannot move it on the device path — comparing an index tensor against
        the capacity is a host read, which is the one thing a capture forbids — so the loop that
        knows its own position keeps it honest. Forty attribute writes against a step that is
        otherwise 38 ms of device work.
        """
        need = int(start_pos) + 1
        for layer in (cache if isinstance(cache, list) else [cache]):
            layer.length = need

    def _wanted(self, need: int, start_pos: int) -> list[int]:
        """The rungs to record: the one this step needs, and — once — the reserved run's.

        All of them at one position, which is what puts the whole capture cost in one
        `first_step_seconds` instead of spreading it over whichever steps happen to cross a
        boundary. Each recording is a body the same position would have run under that rung's width,
        so none of them is a forward the model could not produce.

        The floor is the rung this step needs and not `need` itself: `need` is a row count, and a
        ladder has no rung at 101. A limit of `need` would select nothing at all and leave `_entry`
        replaying a rung it never recorded, which is what this did before a test with no `reserve`
        call found it.
        """
        limit = self.bucket_for(need)
        if self.reserved > start_pos:
            limit = max(limit, self.bucket_for(min(self.reserved, self.capacity)))
        return [width for width in self.ladder if need <= width <= limit and width not in self.steps]

    def _record(self, widths: list[int], token_id: int, start_pos: int) -> None:
        """Record `widths`, and price the pool they share.

        The pool is read as a per-rung increment and summed, rather than as one delta around the
        batch, because the rungs share it and a caller wants the profile as well as the total. Each
        increment is `_Step.capture`'s, measured against the allocator's floor.
        """
        torch.cuda.synchronize()
        for width in widths:
            entry = _Step(self.model, self.cache, width, self.pool, self.device, self.warmup)
            seconds, grew = entry.capture(token_id, start_pos)
            self.capture_seconds += seconds
            self.pool_bytes += grew
            self.rung_bytes[width] = grew
            self.steps[width] = entry

    # -- lifetime ------------------------------------------------------------------------------- #

    def release(self) -> None:
        """Drop the recordings and hand the pool back.

        The graphs go before the pool is freed so that their memory is freeable at the
        `empty_cache`, and the handle is dropped because it is spent: `torch.cuda.graph_pool_handle`
        is a bare id and `create_or_incref_pool` asserts on a freed entry, so a capture into it
        again would abort rather than fail.

        **The holder survives it.** A later `step` finds no rung recorded and records one — into the
        pool `torch.cuda.graph` creates for itself when handed none — so a release is a way to end a
        run and give its memory back, not a way to make the holder refuse. That is the honest
        semantics for an object whose whole job is to be re-recorded, and `tests/
        test_xing4_0_decode_graph.py` asserts the round trip rather than the refusal.
        """
        self.steps = {}
        self.pool = None
        torch.cuda.empty_cache()
