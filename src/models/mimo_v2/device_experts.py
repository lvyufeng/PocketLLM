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


class _Residents:
    """Which of a layer's experts are on the card, learned from the draws as they arrive.

    The router's marginal distribution is heavily skewed -- an effective expert count of 36.6 to
    118.2 of 256 on a real document, measured by `probe_mimo_v2_expert_residency.py` -- so a small
    set of a layer's hottest experts answers a large share of its draws, and an expert held on the
    card is an expert not copied over PCIe. That is the entire value: the copy is 1198.5 MiB a rank
    a token and it is a strictly serial chain of 47 layers, so the bytes are the step's floor and
    nothing else in this file can move it.

    **The policy is least-frequently-used over the run, and it is continuous.** A draw that misses
    is compared against the coldest resident and takes its slot when it has been drawn more often;
    a slot the layer has not filled yet is always taken. That converges on the marginal hot set
    without the two things a periodic rebuild would cost -- a calibration pass whose statistics
    belong to *its* prompt, and a batch of evictions every refresh that is itself a burst of
    copies.

    **It is exact for any policy and that is why this one can be this simple.** A resident row
    holds the same bytes the staging row would have held and the kernel is handed rows, so the
    arithmetic and the order of the sum are the same whether an expert was copied or not:
    residency changes the traffic and not the answer, in either direction and at any moment. So
    the policy is a performance question with no correctness constraint on it, and the tests that
    hold it to the shipped path do not have to know which experts happened to be resident.

    Host-side and not on the card on purpose. The count is over draws, and the draw is already on
    the host -- `forward` reads it back a layer to know which experts to copy -- so a table here
    costs nothing that is not already paid.
    """

    def __init__(self, rows: int, layers: int) -> None:
        self.rows = int(rows)
        #: How many layers may hold a set. A capacity and not a count: a layer's block is claimed
        #: on its first draw, which is what makes a model built for a *subset* of the layers
        #: correct. The layers of such a model are named by their absolute index -- `layers=[2, 5]`
        #: strides the arena by the gap between them -- so an offset computed as `layer_id *
        #: resident_rows` walks off the end of the region as soon as any layer is left out.
        self.layers = int(layers)
        # A layer's resident expert ids in slot order, `-1` for a slot not yet filled, and the two
        # tables a draw is answered out of: the slot an expert occupies, so a hit is a lookup and
        # not a scan, and how often it has been drawn.
        self.ids: list[list[int]] = []
        self.slot: list[dict[int, int]] = []
        self.counts: list[dict[int, int]] = []
        self.block: dict[int, int] = {}
        self.hits = 0
        self.misses = 0
        self.admits = 0
        self.swaps = 0

    def position(self, layer: int) -> int:
        """The block of resident rows `layer` owns, claimed on its first draw."""
        index = self.block.get(layer)
        if index is None:
            if len(self.ids) >= self.layers:
                raise ValueError(
                    f"layer {layer} draws from a resident set built for {self.layers} layers"
                )
            index = len(self.ids)
            self.block[layer] = index
            self.ids.append([-1] * self.rows)
            self.slot.append({})
            self.counts.append({})
        return index

    def row(self, layer: int, slot: int) -> int:
        """The arena row a layer's resident slot is: a block a layer and a row a slot."""
        return self.block[layer] * self.rows + slot

    def take(self, layer: int, expert: int) -> tuple[int, bool] | None:
        """`(slot, needs filling)` for a draw, or `None` to stage it in the slot's own rows.

        A draw the set already holds is `(slot, False)` and is the whole of what the set buys: no
        copy. A draw that missed is offered a slot -- `(slot, True)`, the copy is what filling it
        costs -- or refused, which is the ordinary staging path.

        The offer is least-frequently-used: an unfilled slot is taken outright, and otherwise the
        coldest resident is compared and the incumbent only leaves when the challenger has been
        drawn *strictly* more often, so a tie never evicts and the set cannot thrash on noise.
        """
        block = self.position(layer)
        table = self.counts[block]
        table[expert] = table.get(expert, 0) + 1
        slot = self.slot[block].get(expert)
        if slot is not None:
            self.hits += 1
            return slot, False
        self.misses += 1
        ids = self.ids[block]
        for index, held in enumerate(ids):
            if held < 0:
                self.admits += 1
                return self._place(block, expert, index), True
        coldest = min(range(self.rows), key=lambda index: table.get(ids[index], 0))
        if table.get(expert, 0) <= table.get(ids[coldest], 0):
            return None
        self.admits += 1
        self.swaps += 1
        return self._place(block, expert, coldest), True

    def _place(self, block: int, expert: int, slot: int) -> int:
        evicted = self.ids[block][slot]
        if evicted >= 0:
            del self.slot[block][evicted]
        self.ids[block][slot] = expert
        self.slot[block][expert] = slot
        return slot

    @property
    def held(self) -> int:
        """How many rows of the resident region actually hold an expert."""
        return sum(1 for ids in self.ids for expert in ids if expert >= 0)

    @property
    def drawn(self) -> int:
        """How many draws the set has been asked about, hits and misses together."""
        return self.hits + self.misses


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

    `resident_rows` is the one lever on the bytes and it is not an optimization of the
    copy -- it is the only way the copy gets smaller. A token moves `top_k / world` experts
    a layer over PCIe, which at a world of four is 1198.5 MiB a rank, and the link sustains
    about 11.7 GiB/s: a hundred milliseconds a token that no amount of host work can hide,
    because the layers are a chain and each one's copy waits on the one before. Holding the
    `resident_rows` hottest experts of a layer on the card is what shrinks the chain, and
    the arena carries them as rows that are never overwritten -- see `_residents`.
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
        resident_rows: int = 0,
        resident_layers: int = 1,
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
        #: How many of a layer's experts are held on the card, and how many layers there are to
        #: hold them for. The two together are the arena's resident region, which is one block of
        #: `resident_layers * resident_rows` rows in front of the slots' own rows -- one
        #: allocation and not two, because the kernel takes one tensor and indexes its first axis,
        #: so a resident row and a staging row have to be rows of the same tensor for one call to
        #: read both. `_residents` is the policy that decides which experts those rows hold.
        self.resident_rows = int(resident_rows)
        self.resident_layers = int(resident_layers)
        if self.resident_rows and self.chunk_rows is not None:
            raise ValueError(
                "a chunk's arena is its share of the layer's experts and a resident set is a "
                "draw's hottest few; the two do not share an arena"
            )
        self._residents = (
            _Residents(self.resident_rows, self.resident_layers) if self.resident_rows else None
        )
        self._arena, self._arenas, self._staging_base = self._allocate()
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
            # Which arena row each of this call's positions reads. Without a resident set it is
            # `arange`, which is what this used to be; with one it is a different row a position
            # and has to be built per call, so it has a pinned source of its own either way.
            self._row_host = torch.zeros(self.top_k, dtype=torch.int64, pin_memory=True)
            self._row_device = torch.zeros(self.top_k, dtype=torch.int64, device=self.device)
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

    @property
    def resident_bytes(self) -> int:
        """What the resident set costs the card: one expert a row, kept for the whole run."""
        return self.resident_rows * self.resident_layers * self.expert_bytes

    @property
    def resident_hits(self) -> int:
        """Draws this rank answered out of the resident region, so draws that cost no copy."""
        return 0 if self._residents is None else self._residents.hits

    def resident_report(self) -> dict[str, float]:
        """The set's own numbers, for a probe: it is a policy and a policy has to be scored.

        `hit_rate` is the share of this rank's draws that were already on the card, which is the
        share of `staged_bytes` the set removed; `swaps` is how much of the region's traffic went
        into holding the set up to date rather than into a slot.
        """
        if self._residents is None:
            return {
                "rows": 0,
                "layers": 0,
                "held": 0,
                "hits": 0,
                "drawn": 0,
                "swaps": 0,
                "hit_rate": 0.0,
            }
        drawn = self._residents.drawn
        return {
            "rows": float(self.resident_rows),
            "layers": float(len(self._residents.ids)),
            "held": float(self._residents.held),
            "hits": float(self._residents.hits),
            "drawn": float(drawn),
            "swaps": float(self._residents.swaps),
            "hit_rate": 0.0 if not drawn else self._residents.hits / drawn,
        }

    def _allocate(
        self,
    ) -> tuple[dict[tuple[str, str], torch.Tensor], list[dict], list[int]]:
        """The arena, as `(the allocation, the per-slot views, each slot's first row)`.

        Without a resident set this is one tensor a slot, which is what it always was. With one,
        the resident region has to be a *single* allocation shared by the slots -- the kernel reads
        one tensor and a row is an index into its first axis -- so the allocation is one tensor of
        `resident_layers * resident_rows + slots * arena_rows` rows and the slots are row ranges
        of it. The rows a slot owns start after the resident region, which is what `_staging_base`
        carries into `_stage` and `forward`.
        """
        resident = self.resident_rows * self.resident_layers
        with torch.cuda.device(self.device):
            if not resident:
                arenas = [
                    {
                        key: torch.empty(
                            (self.arena_rows,) + shape, dtype=torch.uint8, device=self.device
                        )
                        for key, shape in self._shapes().items()
                    }
                    for _ in range(self.slots)
                ]
                return arenas[0], arenas, [0] * self.slots
            rows = resident + self.slots * self.arena_rows
            arena = {
                key: torch.empty((rows,) + shape, dtype=torch.uint8, device=self.device)
                for key, shape in self._shapes().items()
            }
            base = [resident + slot * self.arena_rows for slot in range(self.slots)]
            return arena, [arena] * self.slots, base

    @property
    def arena_bytes(self) -> int:
        """Packed fp4 bytes the arena occupies on the card, resident rows and slots together."""
        return (
            self.slots * self.arena_rows + self.resident_rows * self.resident_layers
        ) * sum(shape[0] * shape[1] for shape in self._shapes().values())

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

    def _stage(
        self,
        slot: int,
        layer_id: int,
        experts: Sequence[int],
        rows: Sequence[int] | None = None,
    ) -> None:
        """Copy `layer_id`'s `experts` into the slot's arena rows, on the copy stream.

        Ordered behind the compute stream because the rows are the previous call's inputs
        **and** ahead of it because the kernel about to read them is on the other stream.
        Both halves are load-bearing; see the module docstring.

        Six copies an expert and not one: the source holds an expert as one record in
        `down, gate, up` order and the kernel wants three tensors in `w1, w2, w3`, so
        neither arrangement is a prefix of the other. The destination rows are contiguous
        within each tensor, which is what lets the kernel read them as `[E, N, K/2]`.

        `rows` is where they go, and it exists for the resident set: a row of this call's
        arena is either one of the slot's own -- `_staging_base[slot]` plus the position --
        or a resident row, when the expert is one the layer already holds and the copy is
        what an eviction into it costs. Without a resident set the two are the same list and
        the rows are the slot's own.
        """
        arena = self._arenas[slot]
        base = self._staging_base[slot]
        with torch.cuda.device(self.device):
            self._copy_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self._copy_stream):
                for index, expert in enumerate(experts):
                    row = base + index if rows is None else rows[index]
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
        stage = self._staging_base[slot]
        rows: list[int] = []
        staged: list[int] = []
        staged_rows: list[int] = []
        next_staging = 0
        for position in mine:
            expert = drawn[position]
            row = None
            fill = True
            if self._residents is not None:
                # A held expert is a row this call does not copy, which is the whole of what the
                # resident set buys. A draw the set does not hold may be admitted -- into a free
                # slot, or into the coldest one when it has been drawn more often -- and *that*
                # row has to be filled, which is why an admitted expert is listed with the staged
                # ones: it is a copy either way, into the resident region instead of a slot.
                taken = self._residents.take(layer, expert)
                if taken is not None:
                    row = self._residents.row(layer, taken[0])
                    fill = taken[1]
            if row is None:
                row = stage + next_staging
                next_staging += 1
                fill = True
            if fill:
                staged.append(expert)
                staged_rows.append(row)
            rows.append(row)
        if staged:
            self._stage(slot, layer, staged, rows=staged_rows)
        else:
            # Nothing to copy, and the kernel still has to wait on this slot's event -- but the
            # event it would wait on is the one a *previous* call recorded for the same slot, and
            # that copy is long done. Recording a fresh one on the copy stream makes the wait an
            # edge on an empty stream instead of a stale edge on a real one.
            with torch.cuda.device(self.device):
                self._copy_events[slot].record(self._copy_stream)
        arena = self._arenas[slot]
        compute = torch.cuda.current_stream(self.device)
        compute.wait_event(self._copy_events[slot])
        self._picked_host[: len(mine)] = torch.tensor(mine, dtype=torch.int64)
        picked = self._picked_device[: len(mine)].copy_(
            self._picked_host[: len(mine)], non_blocking=True
        )
        self._row_host[: len(mine)] = torch.tensor(rows, dtype=torch.int64)
        row_index = self._row_device[: len(mine)].copy_(
            self._row_host[: len(mine)], non_blocking=True
        )
        out = self._kernel.moe_single_token_fp4_forward(
            hidden.to(self.device),
            row_index,
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
