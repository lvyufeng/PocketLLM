"""The routed experts, computed on the card, read out of host memory as they are drawn.

This is the half of the heterogeneous path that does arithmetic, and the split it
makes is the whole design: the experts' **bytes** live in the host bank, where all
149.81 GiB of them fit, and the **compute** happens on the card, where one token's
eight experts a layer do. A step therefore moves 8 x 12.75 MiB a layer instead of
reading 256 of them from a disk, and it moves them into an arena laid out the way
the kernel wants rather than the way the checkpoint stores them.

The kernel is `moe_single_token_fp4_forward` from `src/csrc`, which takes the
released storage format directly: `[N, K/2]` uint8 holding two E2M1 codes a byte
and `[N, K/32]` uint8 of E8M0 scales beside them. That is the same format the
DeepSeek-V4.1 checkpoint uses, so the expert arithmetic is shared with that path
rather than reimplemented, and it is the reason this module needs no new CUDA.

What it does *not* take directly is the checkpoint's own arrangement. A bank holds
one expert as a contiguous 12.75 MiB record in file order -- `down_proj` then
`gate_proj` then `up_proj`, each with its scale -- while the kernel wants three
separate `[rows, N, K/2]` tensors named `w1`/`w2`/`w3`. So a drawn expert is copied
into the arena row by row and tensor by tensor: six copies per expert, from six
ranges of the bank that are nowhere near each other. That is why this is a loop of
small copies and not one large one, and it is why the source has to be page-locked
for the copies to be asynchronous -- a pageable source would make PyTorch stage
every one of them through its own pinned ring, which is the copy the bank exists to
delete.

Ordering, which is the part that is easy to get subtly wrong
------------------------------------------------------------

There are two streams a call. The copy stream owns every H2D; the compute stream
owns the kernel. A staging pass waits on the compute stream before it starts,
because the arena rows it is about to overwrite are the ones the previous call's
kernel read; and the kernel waits on the copy stream, because its inputs are the
rows the copies are still writing. Neither wait is optional and neither is implied
by anything else: they are on different streams, so nothing but these two orders
them.

A slot is only reused once its own kernel has been drained, which is what the
per-slot event records and what `_take_slots` waits on. Without that wait the next
layer's copies could start writing a row a kernel launched two calls ago is still
reading, and the two are on the copy stream and the compute stream respectively, so
the wait on the compute stream inside the staging pass is what covers it.

One rank's share, and not the world's
-------------------------------------

When the experts are dealt out over several ranks this module computes **this rank's share of the
draw and nothing else**, which is a partial sum and not the layer's output. The deal and the
collective are `ep.py`'s; what belongs here is only the selection -- which of the draw's `top_k`
positions this rank stages, stages being the one second of the step that costs the link.

That split is deliberate. A module that returned the summed answer would have to own a process
group, and a caller that handed it the wrong group would get a plausible tensor with a quarter of
the arithmetic in it. A module that returns a summand cannot: the sum is the model's, and it is
taken in fp32 before the residual is rounded, once, where the reference rounds it once.

A chunk is the other shape of the same question
-----------------------------------------------

`forward` is one token: a draw of `top_k` experts, an arena `ceil(top_k / world)` rows wide, and a
kernel whose arithmetic is a per-row activation quantisation. `forward_chunk` is a chunk of tokens,
and it is a different question in one respect that matters -- a chunk's drawings name nearly every
expert the layer has, so what is selected is no longer "which positions of this draw" but "which
experts of the quarter this rank owns", and the layout the kernel wants is pairs grouped by expert
rather than one row a drawing. That is what `moe_multi_token_fp4_forward` is for, and the arithmetic
of the two is the same arithmetic on the same arena bytes -- not bit for bit, because the grouped
kernel tiles K in two stages of shared memory where the single-token one tiles it in one, so it costs
about 1e-7 of the answer's own peak and not zero. It is a regrouping and not a second answer.

So this file has two entry points and they are not interchangeable. A chunk under a deal that
partitions draws rather than experts would stage every expert on every rank, and a single token
under a chunk-sized arena would allocate 816 MiB a slot to read two experts out of it. The deal is
`ep.py`'s to define and the caller's to choose; `forward_chunk` refuses the one that would work and
be pointless, and the module's docstrings say which configuration is which.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import torch

from src.kernels.cuda_loader import load_cuda_kernel
from src.models.mimo_v2.ep import deal_rule, owned_experts, owned_positions, rows_per_card

__all__ = [
    "MimoV2DeviceExperts",
    "MimoV2ExpertSource",
    "MmapExpertSource",
    "SWIGLU_LIMIT",
]

#: MiMo-V2.6's experts are plain SwiGLU: the host reference clamps nothing, and the
#: kernel's clamp is off at zero or below. Not a tuning constant -- a statement about the
#: checkpoint, and one a future release that does clamp would have to change.
SWIGLU_LIMIT = 0.0


class MimoV2ExpertSource(Protocol):
    """Where an expert's six tensors come from, as `uint8` views of memory that stays put.

    Deliberately narrow. `MimoV2ExpertBank` is one implementation and `MmapExpertSource`
    another, and a test can be a third -- what the staging loop needs is the six tensors in
    the shapes the kernel declares, in an order the caller can name, not a bank.
    """

    def expert_views(self, layer_id: int, expert: int) -> dict[tuple[str, str], torch.Tensor]:
        """One expert's six tensors, keyed `(projection, kind)`."""


class MmapExpertSource:
    """The checkpoint's own mapped views, for a run that has not filled a bank.

    Correct and slower. The views are the shard's pages, so they are not page-locked, and a
    `non_blocking` copy out of them stages through PyTorch's pinned ring -- the copy the
    bank exists to delete -- but the bytes are the same bytes and the arithmetic is the
    same arithmetic. That makes this the source a test can use on the release without
    spending twelve minutes filling 149.81 GiB, and the source a host without a bank falls
    back to rather than failing.
    """

    def __init__(self, checkpoint: object) -> None:
        self.checkpoint = checkpoint

    def expert_views(self, layer_id: int, expert: int) -> dict[tuple[str, str], torch.Tensor]:
        views = self.checkpoint.expert_arrays(layer_id, expert)
        # `expert_arrays` builds its six views fresh on every call -- it is a handful of
        # `from_numpy` wrappers and no copy -- so nothing is cached here. The staging loop
        # draws any given expert once.
        return views


class MimoV2DeviceExperts:
    """A card's routed experts, computed out of a host-resident source.

    The arena is shared across layers on purpose. A layer at decode width holds eight
    experts, so one layer's own arena would be 102 MiB and forty-seven of them 4.8 GiB of a
    22 GiB card for state that is read once per layer per token; one arena, refilled by
    whichever layer is running, is safe because the layers run one at a time and each call
    drains the kernel it launched before it returns. Which layer is being served is
    therefore a property of the *call* and not of the module -- `layer_id` is a
    constructor argument only so a single-layer caller can set it once.

    `slots` is what makes the copies overlap the kernels. Two slots is the smallest number
    that lets a layer's staging start while the previous layer's kernel is still running,
    and it is what bounds how far the host may run ahead of the card.

    `world` and `rank` are the deal: which of the draw's experts this instance stages and
    computes. The width of the arena follows from the deal rather than from `top_k` -- a
    `sorted` deal over four ranks can only ever be handed two of a top-8 draw, and the
    rows that saves are rows of expert bytes. See `ep.py` for what the two deals cost.

    `n_experts` and `chunk_rows` are the chunk path, and both are needed as soon as a
    caller wants one: a chunk holds a *share* of the layer's experts rather than a draw,
    which is a quarter of them at a world of four and all of them on one rank, and
    `chunk_rows` is how many of that share one kernel call stages -- `0` for all of it in
    one call, a smaller number for bands. Without them the arena is one draw wide and
    `forward_chunk` says so rather than reading a chunk out of two rows.
    """

    def __init__(
        self,
        source: MimoV2ExpertSource,
        layer_id: int | None = None,
        *,
        device: torch.device | str = "cuda",
        top_k: int,
        dim: int,
        inter_dim: int,
        slots: int = 2,
        arena_rows: int | None = None,
        world: int = 1,
        rank: int = 0,
        deal: str | None = None,
        n_experts: int | None = None,
        chunk_rows: int | None = None,
    ) -> None:
        self.source = source
        self.layer_id = None if layer_id is None else int(layer_id)
        self.device = torch.device(device)
        self.top_k = int(top_k)
        self.dim = int(dim)
        self.inter_dim = int(inter_dim)
        self.slots = max(1, int(slots))
        self.world = int(world)
        self.rank = int(rank)
        if self.world < 1:
            raise ValueError(f"world must be at least 1, got {self.world}")
        if not 0 <= self.rank < self.world:
            raise ValueError(f"rank {self.rank} is not a rank of a world of {self.world}")
        self.deal = deal_rule() if deal is None else str(deal)
        # A decode call draws exactly `top_k` experts, so that is the arena width unless a
        # caller asks for more -- which is what a grouped prefill will do, and what makes
        # this a parameter rather than a constant. A deal that bounds what one rank can be
        # dealt narrows it further, and that is the width that gets allocated.
        required = rows_per_card(self.deal, self.top_k, self.world)
        self.arena_rows = required if arena_rows is None else int(arena_rows)
        if self.arena_rows < required:
            raise ValueError(
                f"an arena of {self.arena_rows} rows cannot hold the {required} experts a "
                f"`{self.deal}` deal can hand rank {self.rank} of {self.world}"
            )
        # The chunk path, when a caller asks for one. `chunk_rows` is how many of this rank's
        # share one kernel call holds -- its band -- and `None` means the module is a decode
        # module: the arena is then one token's draw wide and a chunk has nowhere to live.
        # `0` is the whole share in one band, which is what a chunk-size bands' worth of rows
        # buys and what the default serving configuration uses. See `forward_chunk`.
        self.n_experts = None if n_experts is None else int(n_experts)
        self.chunk_rows: int | None = None
        self.n_local = 0
        self._owned_rows: tuple[int, ...] | None = None
        self._row_of_local: torch.Tensor | None = None
        self._slot_ids: torch.Tensor | None = None
        self.chunks = 0
        if chunk_rows is not None:
            if self.n_experts is None:
                raise ValueError(
                    "a chunk band holds a share of the experts, so the module has to be told "
                    "how many the layer routes to: pass `n_experts`"
                )
            owned = owned_experts(
                self.n_experts, rank=self.rank, world=self.world, deal=self.deal
            )
            if not owned:
                raise ValueError(
                    f"a world of {self.world} over {self.n_experts} experts leaves rank "
                    f"{self.rank} nothing under the `{self.deal}` deal"
                )
            self._owned_rows = tuple(owned)
            self.n_local = len(owned)
            self.chunk_rows = self.n_local if int(chunk_rows) <= 0 else int(chunk_rows)
            self.chunk_rows = min(self.chunk_rows, self.n_local)
            self.arena_rows = max(self.arena_rows, self.chunk_rows)
            with torch.cuda.device(self.device):
                # The whole table once: which arena row holds an expert this rank owns, and a
                # sentinel for the ones it does not. A chunk's routing is then one gather, and
                # the sentinel needs no branch -- every unowned pair sorts to the end.
                table = torch.full(
                    (self.n_experts,), self.n_local, dtype=torch.int32, device=self.device
                )
                table[torch.tensor(owned, dtype=torch.int64, device=self.device)] = torch.arange(
                    self.n_local, dtype=torch.int32, device=self.device
                )
                self._row_of_local = table
                self._slot_ids = torch.arange(
                    self.n_local + 1, dtype=torch.int32, device=self.device
                )

        if self.dim % 32 or self.inter_dim % 32:
            raise ValueError(
                f"the fp4 kernel needs dim and inter_dim divisible by 32, got {self.dim} and "
                f"{self.inter_dim}"
            )

        self._kernel = load_cuda_kernel()
        if self._kernel is None:
            # `load_cuda_kernel` returns None both when nothing is built and when the build does
            # not match this interpreter, and it swallows the reason. This path has no fallback
            # -- the expert arithmetic *is* the extension -- so the failure has to be here and
            # not an `AttributeError` on a `None` a hundred calls later.
            raise RuntimeError(
                "the `cuda_kernel` extension did not load: build it, or run under the "
                "interpreter it was built for (`scripts/build_extensions.sh` names one)"
            )
        self._arenas = [self._allocate() for _ in range(self.slots)]
        with torch.cuda.device(self.device):
            self._events = [torch.cuda.Event() for _ in range(self.slots)]
            self._copy_events = [torch.cuda.Event() for _ in range(self.slots)]
            self._copy_stream = torch.cuda.Stream(device=self.device)
        self._pending: list[bool] = [False] * self.slots
        # The draw's bookkeeping, staged through a pinned pair rather than built on the card.
        # `mine` is a list of positions on the host and the kernel wants them as a device tensor;
        # the obvious `torch.tensor(mine, device=...)` is a *pageable* H2D, which is synchronous
        # and therefore a host wait per layer -- forty-seven of them a token, on a path whose
        # whole problem is that the host is in it. A pinned source and a non-blocking copy are
        # ordered by the stream instead, and the width is `top_k` because a deal's share of a
        # drawing is at most all of it.
        with torch.cuda.device(self.device):
            self._picked_host = torch.zeros(self.top_k, dtype=torch.int64, pin_memory=True)
            self._picked_device = torch.zeros(self.top_k, dtype=torch.int64, device=self.device)
            self._rows = torch.arange(self.top_k, dtype=torch.int64, device=self.device)
        self._next = 0
        # Counters, because "how much did the bank save" is a question about the copies and
        # not about the wall clock: `staged_experts` is the draws that cost an H2D.
        self.staged_experts = 0
        self.staged_bytes = 0

    # -- allocation -------------------------------------------------------------------------

    def _shapes(self) -> dict[tuple[str, str], tuple[int, ...]]:
        return {
            ("gate_proj", "weight"): (self.inter_dim, self.dim // 2),
            ("gate_proj", "weight_scale"): (self.inter_dim, self.dim // 32),
            ("down_proj", "weight"): (self.dim, self.inter_dim // 2),
            ("down_proj", "weight_scale"): (self.dim, self.inter_dim // 32),
            ("up_proj", "weight"): (self.inter_dim, self.dim // 2),
            ("up_proj", "weight_scale"): (self.inter_dim, self.dim // 32),
        }

    def _allocate(self) -> dict[tuple[str, str], torch.Tensor]:
        with torch.cuda.device(self.device):
            return {
                key: torch.empty((self.arena_rows,) + shape, dtype=torch.uint8, device=self.device)
                for key, shape in self._shapes().items()
            }

    @property
    def arena_bytes(self) -> int:
        """Packed fp4 bytes one slot occupies on the card."""
        return self.slots * self.arena_rows * sum(
            shape[0] * shape[1] for shape in self._shapes().values()
        )

    @property
    def expert_bytes(self) -> int:
        """One expert's packed bytes, which is what one draw costs the PCIe link."""
        return sum(shape[0] * shape[1] for shape in self._shapes().values())

    # -- staging ----------------------------------------------------------------------------

    def _take_slots(self, count: int) -> int:
        """The next slot, with the overwrite ordered behind the kernel that still reads it.

        Rotating rather than searching is what keeps the copies and the kernels apart: with
        two slots and one call in flight, `_next` names the slot the *previous* call used, so
        the copy that overwrites it is ordered behind a kernel that is a full layer behind.

        **Ordered by the copy stream and not by the host.** The overwrite is safe because
        `_stage` makes the copy stream wait on the compute stream's tail, which at that moment
        already holds the kernel that read the slot two layers ago -- so the host does not have
        to wait for anything, and a host that waits is a host that has stopped issuing. That
        matters here more than it looks: this arena is one of forty-seven in a layer loop, so
        the wait this dropped was one a layer and forty-seven a token, on a path whose whole
        problem is that the host is in it. `drain` is where a caller that genuinely needs the
        kernels to have happened says so.
        """
        if count > self.arena_rows:
            raise ValueError(f"a call needs {count} arena rows and the arena holds {self.arena_rows}")
        slot = self._next
        self._next = (self._next + 1) % self.slots
        return slot

    def _stage(self, slot: int, layer_id: int, experts: Sequence[int]) -> None:
        """Copy `layer_id`'s `experts` into the slot's arena rows, on the copy stream.

        Ordered behind the compute stream because the rows are the previous call's inputs
        **and** ahead of it because the kernel about to read them is on the other stream.
        Both halves are load-bearing; see the module docstring.

        Six copies an expert and not one: the source holds an expert as one record in
        `down, gate, up` order and the kernel wants three tensors in `w1, w2, w3`, so
        neither arrangement is a prefix of the other. The destination rows are contiguous
        within each tensor, which is what lets the kernel read them as `[E, N, K/2]`.
        """
        arena = self._arenas[slot]
        with torch.cuda.device(self.device):
            self._copy_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._copy_stream):
                for row, expert in enumerate(experts):
                    views = self.source.expert_views(layer_id, int(expert))
                    for (proj, kind), target in arena.items():
                        target[row].copy_(
                            views[(proj, kind)].view(torch.uint8), non_blocking=True
                        )
                self._copy_events[slot].record(self._copy_stream)
        self.staged_experts += len(experts)
        self.staged_bytes += len(experts) * self.expert_bytes

    # -- the call ---------------------------------------------------------------------------

    def forward(
        self,
        hidden: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        *,
        layer_id: int | None = None,
    ) -> torch.Tensor:
        """This rank's share of one token's routed output, `[1, dim]`, in float32.

        `indices` is `[top_k]` int64 of global expert ids and `weights` is `[top_k]` float32,
        both as `gate_and_route` leaves them. The kernel is handed `arange(rows)` instead of
        the ids, because the arena holds the staged experts in the order they were staged
        and the id has already been spent -- the row that holds expert 137 is whatever row
        the draw gave it. That is also why the weights can be passed through unchanged: they
        follow the same order, which is the draw order and not the id order.

        `layer_id` names whose experts to draw, and defaults to the one the module was built
        with. A whole-model caller leaves it unset at construction and passes it here, which
        is what lets forty-seven layers share one arena.

        **A partial when the world is more than one.** The kernel sums a weighted set of
        rows, so a rank that staged some of a draw's experts holds the part of the sum that
        those experts contribute, and the layer's answer is the sum over ranks. Rank's whose
        deal gave it nothing returns zeros -- correctly, and it is not a corner: under the
        `id` deal a rank of four owns nothing in 10% of top-8 draws, and the collective is
        unconditional, so the zero has to be a value and not a skipped call.

        One token, and deliberately: the kernel is the single-token one, whose arithmetic is
        a per-row int8 activation quantisation. A batch goes through
        `moe_multi_token_fp4_forward`, which is defined to agree with this one bit for bit
        when fed one token at a time, and through the grouped prefill path for a chunk.
        """
        if hidden.dim() != 2 or hidden.shape[0] != 1:
            raise ValueError(f"the single-token path takes [1, dim], got {tuple(hidden.shape)}")
        layer = self.layer_id if layer_id is None else int(layer_id)
        if layer is None:
            raise ValueError(
                "no layer was named: this arena holds whichever layer's experts were last "
                "staged, and it has no way to know a caller forgot to say which"
            )
        indices = indices.reshape(-1)
        weights = weights.reshape(-1)
        if indices.numel() != weights.numel():
            raise ValueError(
                f"{indices.numel()} indices and {weights.numel()} weights is not a routing"
            )
        if indices.numel() > self.top_k:
            raise ValueError(
                f"a draw of {indices.numel()} experts is wider than the {self.top_k} this "
                f"module was built for"
            )

        drawn = indices.tolist()
        mine = owned_positions(drawn, rank=self.rank, world=self.world, deal=self.deal)
        if len(mine) > self.arena_rows:
            raise ValueError(
                f"the deal handed this rank {len(mine)} of the draw's {len(drawn)} experts and "
                f"the arena holds {self.arena_rows}"
            )
        if not mine:
            return torch.zeros((1, self.dim), dtype=torch.float32, device=self.device)

        slot = self._take_slots(len(mine))
        self._stage(slot, layer, [drawn[position] for position in mine])
        arena = self._arenas[slot]
        compute = torch.cuda.current_stream(self.device)
        compute.wait_event(self._copy_events[slot])
        self._picked_host[: len(mine)] = torch.tensor(mine, dtype=torch.int64)
        picked = self._picked_device[: len(mine)].copy_(
            self._picked_host[: len(mine)], non_blocking=True
        )
        rows = self._rows[: len(mine)]
        out = self._kernel.moe_single_token_fp4_forward(
            hidden.to(self.device),
            rows,
            weights.index_select(0, picked).to(self.device, dtype=torch.float32),
            arena[("gate_proj", "weight")],
            arena[("gate_proj", "weight_scale")],
            arena[("down_proj", "weight")],
            arena[("down_proj", "weight_scale")],
            arena[("up_proj", "weight")],
            arena[("up_proj", "weight_scale")],
            0,
            SWIGLU_LIMIT,
        )
        self._events[slot].record(compute)
        self._pending[slot] = True
        return out

    # -- a chunk ----------------------------------------------------------------------------

    def forward_chunk(
        self,
        hidden: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        *,
        layer_id: int | None = None,
    ) -> torch.Tensor:
        """This rank's share of a chunk's routed output, `[rows, dim]`, in float32.

        `indices` is `[rows, top_k]` int64 and `weights` `[rows, top_k]` float32, as
        `gate_and_route` leaves a chunk of tokens, and every column of every row is one drawing.
        The sibling `forward` takes one row and one drawing list; this takes a chunk, and the
        difference is not the batch -- the kernel has always been batched -- but the *layout*.
        A chunk's drawings name nearly every expert the layer routes to, so what this rank
        computes is the subset of the chunk's pairs whose expert it owns, and the kernel wants
        those pairs grouped by expert with one arena row a group.

        **The layout.** The three tensors are built on the card, from `indices` alone:

        * `_row_of_local`, one int32 an expert, maps a global id to the row that holds it, or to
          the sentinel `len(owned)` for an expert this rank does not hold. One gather turns the
          chunk's `[rows, top_k]` of ids into slots, and the sentinel needs no branch: a stable
          sort by slot puts every unowned pair behind every owned one, so the pairs this call
          computes are a prefix of the sorted order and their count is the last band bound.
        * The pairs in that prefix, in slot order, are `slot_tokens` -- the row of the chunk
          each belongs to -- and `pair_weights`, this rank's share of the pair's own routing
          weight. Slot order and not encounter order, so that two runs of the same routing hand
          the kernel the same layout; the sort is stable, so each slot's pairs arrive in the
          order the chunk produced them.
        * `slot_starts` is `searchsorted` of the sorted slots against `0..len(owned)`, which is
          the counts' exclusive prefix sum in one op and no host round trip. It is deliberately
          not `bincount`: `_bincount_cuda` bounds-checks with two blocking device-to-host reads,
          and this is a drain the chunk path cannot afford. V4.1 replaced the same call for the
          same reason.

        **The bands.** One kernel call stages its whole working set in the arena and sums the
        chunk's rows at once, so the arena has to be as wide as the experts one call holds --
        `chunk_rows`. A rank that owns more than that is computed in bands of that width, one
        call each, summed in `[rows, dim]` float32. The bands are slices of the one sorted pair
        list, so a band costs no layout work: band `b` owns slots `[b*chunk_rows, (b+1)*chunk_rows)`
        and its pairs are the contiguous run between those two prefix sums. A band whose experts
        the chunk never drew is skipped, which is what makes a narrow band cheap at a small chunk
        and why the width is a knob rather than a constant -- it trades arena bytes for call
        count, and the calls of adjacent bands overlap the way the layers do, on the slots.

        **One host read a call.** `slot_starts` lives on the card, but the band boundaries have to
        become Python integers before the pairs can be sliced with them, so `starts` at the band
        edges is gathered into one small tensor and read back once -- one synchronisation a layer,
        and the ownership count comes out of the same read rather than a second one. That is the
        decode path's own price, which pulls the whole draw across with `indices.tolist()` every
        layer, and nothing else here touches the host: the staged experts are the rank's own
        quarter, known at construction, and no routing id crosses the bus.

        **A partial when the world is more than one**, exactly as `forward` is.
        """
        if self._row_of_local is None:
            raise ValueError(
                "this module was built without a chunk band (`chunk_rows`), so its arena is "
                "one token's draw wide and a chunk has nowhere to live; build it with "
                "`chunk_rows=0` for the whole share in one band"
            )
        if self.world > 1 and self.deal != "id":
            raise ValueError(
                f"a chunk is the `id` deal's: under `{self.deal}` a chunk's drawings reach "
                f"every one of the {self.n_experts} experts on every rank, so each rank would "
                f"stage them all and the deal would save nothing. Build the module with "
                f'`deal="id"`'
            )
        layer = self.layer_id if layer_id is None else int(layer_id)
        if layer is None:
            raise ValueError(
                "no layer was named: this arena holds whichever layer's experts were last "
                "staged, and it has no way to know a caller forgot to say which"
            )
        if hidden.dim() != 2:
            raise ValueError(f"a chunk is [rows, dim], got {tuple(hidden.shape)}")
        rows = int(hidden.shape[0])
        if rows == 0:
            return torch.empty((0, self.dim), dtype=torch.float32, device=self.device)
        indices = indices.reshape(rows, -1)
        weights = weights.reshape(rows, -1).to(torch.float32)
        if indices.shape != weights.shape:
            raise ValueError(f"{tuple(indices.shape)} indices and {tuple(weights.shape)} weights")
        width = int(indices.shape[1])
        if width > self.top_k:
            raise ValueError(
                f"a draw of {width} experts is wider than the {self.top_k} this module was "
                f"built for"
            )

        flat_index = indices.reshape(-1)
        slots = self._row_of_local[flat_index]
        # The whole list is sorted and not a prefix of it: how many pairs this rank owns is
        # `starts[n_local]`, which the band bounds already carry, and a count of its own would be
        # a second device-wide read of an answer the next line has. One read a call is the budget
        # here, so `slots < n_local` is never reduced on the host.
        take = torch.argsort(slots, stable=True)
        slot_of_pair = slots.index_select(0, take)
        token_of_pair = (
            torch.arange(rows * width, dtype=torch.int32, device=self.device) // width
        ).index_select(0, take)
        weight_of_pair = weights.reshape(-1).index_select(0, take)
        starts = torch.searchsorted(slot_of_pair, self._slot_ids).to(torch.int32)

        band = self.chunk_rows
        edges = list(range(0, self.n_local, band)) + [self.n_local]
        bounds = starts[edges].tolist()
        if bounds[-1] == 0:
            return torch.zeros((rows, self.dim), dtype=torch.float32, device=self.device)
        source = hidden.to(self.device)
        if not source.is_contiguous():
            source = source.contiguous()
        out = None
        for index, (lo, hi) in enumerate(zip(edges, edges[1:])):
            first, last = bounds[index], bounds[index + 1]
            if first == last:
                continue
            slot = self._take_slots(hi - lo)
            self._stage(slot, layer, self._owned_rows[lo:hi])
            arena = self._arenas[slot]
            compute = torch.cuda.current_stream(self.device)
            compute.wait_event(self._copy_events[slot])
            partial = self._kernel.moe_multi_token_fp4_forward(
                source,
                torch.arange(hi - lo, dtype=torch.int32, device=self.device),
                (starts[lo : hi + 1] - first).to(torch.int32),
                token_of_pair[first:last],
                weight_of_pair[first:last],
                arena[("gate_proj", "weight")],
                arena[("gate_proj", "weight_scale")],
                arena[("down_proj", "weight")],
                arena[("down_proj", "weight_scale")],
                arena[("up_proj", "weight")],
                arena[("up_proj", "weight_scale")],
                SWIGLU_LIMIT,
            )
            self._events[slot].record(compute)
            self._pending[slot] = True
            out = partial if out is None else out.add_(partial)
        self.chunks += 1
        if out is None:
            return torch.zeros((rows, self.dim), dtype=torch.float32, device=self.device)
        return out

    def drain(self) -> None:
        """Wait for every kernel this module has launched.

        The arena holds no state between calls -- only bytes -- so there is nothing to
        reconcile; what this is for is a caller that is about to read the host side, or to
        tear the process down, and wants the copies it ordered to have happened.
        """
        for slot in range(self.slots):
            if self._pending[slot]:
                self._events[slot].synchronize()
                self._pending[slot] = False
