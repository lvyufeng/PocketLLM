"""One layer's routed experts on the cards, fed the packed fp4 straight out of the shards.

`CheckpointRoutedExperts` in `loader.py` is the correctness path and it is also the whole cost of a
host token: it expands each miss to bf16 on the CPU, measured at 0.122 s per expert, so a step that
misses all 240 of a token's experts is 29 s and the 119 of 240 a warm step still misses is 14.5 s.
This class is the same subject on the other side of PCIe. It never builds a bf16 expert matrix
anywhere -- `moe_single_token_fp4_forward` consumes the checkpoint's own `[E, N, K/2]` uint8 weights
and `[E, N, K/32]` E8M0 scales and dequantizes inside the kernel -- so what crosses the link is the
17.9 MiB the checkpoint stores rather than the 67.5 MiB an expansion would be, and what the host runs
out of is arithmetic on 240 experts rather than a memcpy of the same bytes into the same shape.

Two conventions had to be settled before this could be written, and both came out in the kernel's
favour; they are recorded here because each of them looked like a blocker first:

* **The scale.** `fp4_block_scale` (`cuda_kernel_impl.cu:2388`) is
  `__int_as_float(max(0, byte - 1) << 23)`. `__int_as_float` places its argument in the *exponent
  field*, so the value is `2**(byte - 1 - 127)` and not `2**(byte - 1)`; with the LUT's doubled e2m1
  levels that is `level * 2**(byte - 127)`, bit for bit what torch's `float8_e8m0fnu` reads on the
  same byte and what `cpp_engine/backends/cuda/kernels/fp4_ops.cu:243` computes as `exp2f(code - 127)`.
  The `- 1` absorbs the LUT's `* 2`. `tests/test_moe_single_token_fp4.py` draws scales 124..130
  against a reference that reads them as `2**(byte - 127)` and reproduces the kernel to 2.9e-5
  relative, and the kernel's own header says the same thing.
* **The layout.** The release stores an expert's `w1.weight` as `(2304, 2560)` `I8` beside
  `w1.scale` `(2304, 160)` `F8_E8M0` -- `[N, K/2]` beside `[N, K/32]` with `N = inter_dim` and
  `K = dim`, which is the op's ABI verbatim -- and `w2` as `(5120, 1152)` beside `(5120, 72)` as
  `[dim, inter/2]`. So an arena row is the checkpoint's own tensor and no transpose, no per-element
  scale expansion and no rebasing happens here.

The split is expert parallelism over `world` cards, and it is static. Per row the six routed experts
are sorted by global id and dealt round-robin, so card `c` owns sorted positions `c` and `c + world`
-- a **2, 2, 1, 1** split over four cards -- and each card's arena is a fixed
`ceil(topk / world)` rows. Nothing resizes between steps and nothing is allocated per token: the
pinned arena is those rows for every card laid end to end, 35.9 MiB per card and 143.4 MiB for four,
and the device holds the matching per-card slice.

The cards never talk to each other. Each holds its own arena, is handed the same `[1, 5120]`
activation, and returns `[1, 5120]` fp32; whoever asked for the row sums the partials. That is
20 KiB back per card per layer, 3.2 MB per token across four cards, and it is why this needs no
all-to-all and no collective of its own. It also makes `world=1` the single-card configuration
rather than a second implementation.

**Who does the summing is the one thing `ranks` decides.** Deal every rank of the layer into one
process -- `ranks` left unset, `world` devices -- and that process returns the layer's whole routed
output, which is what the host-side probes measure. Deal one rank per process -- `ranks=[r]`, one
device -- and it returns that rank's share, and the routed partial then joins the shared expert's on
the ffn's all-reduce instead of at a host `+`. The deal itself is unchanged: `world` is still how
many ways the sorted ids are dealt, so rank `r` stages exactly the experts it staged before, into
exactly the rows it staged them into, and the two configurations differ only in who adds the pieces
up. `partial` is that distinction, and `MoE.forward` reads it.

How it is checked: `/tmp/probe_device_experts.py` runs this class against `expert_forward` on layer
0's captured prefill activation. `world=4` and `world=1` agree with each other to **5.960e-08**, fp32
association order across the four partial sums and nothing else, and both sit 1.5-2.4% of the output
scale from the host result at cosine 0.9993-0.9997 -- the fp4 kernel's own error, which
`/tmp/probe_fp4_parity.py` measures independently. So the split, the deal, the staging and the weight
permutation are all accounted for: what is left between this and the host path is the arithmetic the
checkpoint ships with.

What it costs, measured on this host (`/tmp/probe_device_experts.py` for the class on its own,
`/tmp/probe_launch_cost.py` and `/tmp/probe_launch_split.py` for the terms below, and
`/tmp/probe_stage.py` and `/tmp/probe_fp4_parity.py` for staging and for the kernel's own error):

| Term | Per token, 40 layers x 6 experts | Source |
| --- | --- | --- |
| this class alone, both sides of PCIe, `world=4` | 0.59 s | measured, 14.8 ms per row-layer |
| a whole step, this class plus the host dense tree | **1.14 s** | measured, warm decode |
| the same class at `world=1` | 0.88 s | measured, same probe, one card |
| a whole step at `world=1` | 1.40 s | measured before the ordering fix below |
| `_take_buffer`, the wait for the previous DMA | **0.00 s** | measured in a real step |
| host staging, page cache -> pinned, 4.20 GiB | 0.30 s, 14 GiB/s | measured in a real step |
| the four copy chains, enqueued | 0.03-0.04 s | measured; the transfer is awaited later |
| a row's card side, four kernels plus the H2D | 0.15 s | measured in a real step, 160 calls |
| the host dense tree and the layer glue around it | 0.55-0.67 s | measured, the step's unattributed remainder |

The probes are `/tmp/probe_device_experts.py`, which loops one layer's rows against the captured
activation and sees 14.8 ms per row-layer; `/tmp/probe_launch_cost.py`, which wraps all four of the
class's phases in a real step over the prompt and sees 11.4 ms per row-layer inside this class --
7.3 ms of staging, 0.8 ms of upload issue and 3.75 ms of card time -- with 0.55-0.67 s per token
outside it in the attention stack, the gate, the shared experts, the head and the layer glue; and
`/tmp/probe_launch_split.py`, which prices one row's four calls three ways and is where the 3.42x
below comes from. The last two were written when the row's card side was one method named `_launch`;
the split into `_issue` and `_drain` is the pipeline below, and the numbers they record are the same
two halves of it.

**The world size is worth 0.1-0.2 s of that and no more**, which is the one prediction of the plan
this measurement does not support. At `world=1` the same probe measures 1.40 s against 1.23-1.42 s,
before the ordering fix below: the staging is identical at 0.32-0.33 s, because it is host work and
does not care how many cards read the bytes, while the card side falls from 0.46 s to 0.27 s. The split
is what makes the launch a fifth of a step instead of a third; it does nothing for the staging, which
the four cards have made the largest single term. That is the honest reading of the four-against-one
comparison, and it is why the follow-on that would matter keeps the packed rows on the card rather
than staging them again -- the wider-arena bullet at the end of this docstring.

Where the 1.14 s goes, measured rather than inferred (`/tmp/probe_launch_cost.py`, warm decode):
**0.30 s staging, 0.15 s of card time, 0.03 s of upload issue, 0.00 s in `_take_buffer`, and
0.55-0.67 s of everything else** -- the dense tree, the gate, the shared experts, the head and the
loop. Against the 29 s the same token costs on the host path cold, and the 14.5 s it costs warm, none
of this is close; against the step's own terms, what the host still runs is now the largest of them.

`_take_buffer`'s zero is worth recording because it was this docstring's standing inference: the wait
for the previous upload's DMA is already satisfied, because with two pinned buffers the buffer being
staged was read by a DMA issued a row and a launch earlier. A third buffer would buy nothing, and
`_upload`'s 0.03-0.04 s really is issue and not transfer. The row's card side decomposes the same way:
a row of four calls costs **3.63 ms** with the drain inside the card loop and **1.01 ms** with the four
kernels issued before any is drained, so its 0.15 s is one kernel's worth of arithmetic plus the
**~0.11 s** of H2D the kernels wait on -- a card's arena copy is ordered behind that card's previous
kernel, so the transfer, unlike the issue, is on the device's critical path and not the host's.

**Every number above is a warm-page-cache number, and on this host the cache holds 14-21% of the
checkpoint.** This class stages out of the checkpoint mapping, so `_stage`'s 0.30 s is a read of
4.20 GiB out of RAM, and `/tmp/fadvise_drop.py` -- `POSIX_FADV_DONTNEED` over the 48 shards,
confirmed with `/tmp/mincore_resident.py` -- is the other column. With the pages dropped, the same
8-token row measures **9.91 s of staging and 17.01 s a step** against 0.30 and 0.83-0.88 resident --
33x and 19x, the same bytes and the same row. `_upload` (16 -> 26 ms) and `_launch` (161 -> 157 ms) do
not move, so it is not the cards, the kernels or the link -- it is one phase reading the same bytes
off an SMR disk. The reason is arithmetic: **457.78 GiB of this host's 1007 GiB is the resident
bank's tmpfs segment**, which is not reclaimable, so the 475.25 GiB this path reads cannot also be in
the page cache.

Quote the step with the cache it was measured on, and note that the source is a choice: `banked()`
below reaches the segment when one is attached, so `DEEPSEEK_V41_RESIDENT_EXPERTS=1` moves `_stage`
off the mapping and the step stops depending on the cache. Measured on the same cold row, that is
**782.9 ms a step and 242.1 ms of staging** against the 17.01 s and 9.91 s above -- 21.7x and 41x.
Warm it is a wash, and for the obvious reason: 244.0 ms banked against 224.7 unbanked, both memcpys
of the same 4.20 GiB into the same pinned arena, the first out of tmpfs and the second out of the page
cache. The bank removes the disk; it does not remove the copy, so `_stage` is still the largest term
inside the class and still what the pipeline at the end of this docstring is for.

The staging rate is the one term that had to be measured rather than argued, because the whole plan
turns on it: a step stages 40 rows x 6 experts x 17.9 MiB = 4.20 GiB and the phase costs 0.30 s, so
**14 GiB/s**, which is the 11.49 GiB/s an isolated `copy_` into a pinned arena reaches and not the
0.33 GiB/s that `pin_memory()` inside the loop reaches. Prefill is the same rate over 200 rows:
1.89-1.94 s for 21.0 GiB, 10.8-11.1 GiB/s.

The measured rows are warm page cache. The staging row is a read of `/mnt/data3`, an SMR disk, and a
cold one is 1308.0 ms against 11.2 ms -- which is the floor a first token on a cold cache pays and
nothing after it does. Multi-threading the staging was measured and regresses -- 733 ms against 365
ms -- because a 3 MiB `copy_` is already at what one core pulls out of the page cache, so the loop
here is deliberately single-threaded.

**The row loop is one row deep, and it is worth 1.10x on a prefill.** A row used to be strictly
serialized -- the host could not stage row `k+1` until the card side of row `k` had returned, and that
returned only once row `k`'s partials had landed -- so the four cards' work and the *next* row's
staging were never in flight together. `forward` now holds one row: it stages row `k+1` and issues its
kernels, and only then drains row `k`. A decode step is one row a layer, so a row-deep pipeline has
nothing to overlap there and the whole of what it can be worth is a prefill's, `n` rows a layer, `n` =
the prompt length. Measured with `/tmp/probe_v41_tp4_pipe_matrix.py` -- four ranks under `torchrun`,
one process a card, the resident bank on, 22 threads, and both orders alternated inside one process so
that the node's own 20% drift across a sitting lands on both columns instead of on one:

| prompt | order | prefill | tok/s | `_stage_row` | `_stage` | `_upload` | `_issue` | `_drain` |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | serial | 13.51 s, 13.77 s | 2.4 | 7488 ms | 6850 ms | 482 ms | 849 ms | 4792 ms |
| 32 | pipelined | 12.43 s, 12.42 s | 2.6 | 8720 ms | 8048 ms | 594 ms | 989 ms | 2305 ms |
| 128 | serial | 53.61 s, 53.45 s | 2.4 | 30096 ms | 27542 ms | 1964 ms | 3434 ms | 19264 ms |
| 128 | pipelined | 48.44 s, 48.18 s | 2.7 | 34975 ms | 32717 ms | 2007 ms | 3501 ms | 9033 ms |

That is **1.098x** at 32 tokens and **1.108x** at 128, and the per-row-layer account is flat across
both: the drain falls from **3.75 ms to 1.8 ms** a row-layer and the staging rises from **5.35 ms to
6.3 ms**, for a net **~1.0 ms** off a row-layer that costs about 12. The rise is not a mystery and not
a cost the pipeline introduces so much as one it stops hiding: the staging is a memcpy out of tmpfs
and, with a row in flight, it now shares the memory system with that row's H2D reading the *other*
pinned arena. What is left of the drain -- 1.8 ms -- is the row's own arena copy plus its kernel,
which a one-row-deep pipeline cannot go below; a two-row pipeline would have to issue row `k+1`'s H2D
while row `k`'s kernel ran, and it needs a third pinned arena to do it. That is the follow-on, and the
measurement above is the bar it has to clear. The pinned pool is not it: swept at two, three and four
arenas a layer, `_take_buffer` is 10-29 ms against a stage of 4-8 s, so `pinned_buffers` stays at 2.

Two things had to change for the pipeline to pay, and the first version of it did not. It measured
**1.012x**, and the reason was the routing: `indices_row.tolist()` inside the row loop synchronizes
the card's stream, and under the pipeline that stream already has the *previous* row's kernel on it,
so those reads cost **2.9 ms a row** where they cost 0.1 ms in a loop that has already drained. That
was as much as the drain saving, given straight back. `_route_ids` takes the whole `[n, topk]` to host
memory in one pinned copy ahead of the loop, and every row's read is then a read of host memory. The
second is the route weights: `_issue` gathers them on the card and copies them back `non_blocking`, so
nothing in the loop waits on a pageable transfer.

It does not change the answer, and the probe is arranged to show that rather than assert it. A prefill
of this path is not bit-reproducible -- the same order twice differs by 3.4e-02-7.1e-02 max abs on the
last token's logits -- and the pipelined run differs from the serial one by 2.0e-02, inside that
spread. Comparing a pipelined prefill against a serial one and reading the difference as the pipeline
would be reading run-to-run noise.

Two things this deliberately does not do yet, all because they are separate measurements rather
than separate opinions:

* **Nothing is cached on the device between rows.** The arena is the two rows this row needs and it
  is refilled every row, so a prefill of `n` rows pays the 4.20 GiB `n` times. A wider arena plus
  `moe_multi_token_fp4_forward` -- one slot per distinct expert the batch hit, its tokens contiguous
  -- is the shape that fixes it, and it is a follow-on rather than a knob, because an arena size and
  an eviction policy only mean something once that measurement exists. The row-layer table above is
  the number it is worth: at 128 tokens the staging is 6.4 ms of a 12 ms row-layer and 32,717 ms of a
  48 s prefill, and it is paid once per route. What a per-layer slot set saves is the ratio of a
  rank's routes to the distinct experts among them -- at 128 tokens, four ranks dealing six sorted ids
  round-robin leave a rank 256 draws from the 384-expert space over the layer's 128 rows, and the
  duplicates in those draws are what the design would stop paying for. It is worth more the longer
  the prompt, which is the opposite of the pipeline above.
* **The dense tree's own quarter is now on the card beside this class, and the split moved.** With
  the tree on the host this class was the whole step's smaller half; with the tree cut across the
  four cards (TP4, `src/cli/generate_v41.py`) the activation arrives on a card rather than from the
  host, so the input copy is a device-to-device one and the partial goes back to the card the `MoE`
  wants it on. Measured that way at 8 tokens of decode, **722-747 ms per step, 409-444 of it in this
  class and 303-313 in the tree** -- against the recorded 1060-1140 ms per step with the tree on the
  host, so the move took 620 ms of dense tree down to 310 and left the staging as the larger half.
  Read that pair with the warm cache the paragraph above describes, and read the banked cold pair with
  the same paragraph's last table: the resident source is a `DEEPSEEK_V41_RESIDENT_EXPERTS` away, and
  what it does not take away is the 242 ms `_stage` still spends copying into pinned. This is also why
  the class no longer raises when a rank holds none of a row's routes: under a deal that rank's share
  is zero and it has to stay in the all-reduce to say so.
"""

from __future__ import annotations

from typing import Sequence

import torch

from src.kernels.cuda_loader import load_cuda_kernel
from src.models.deepseek_v4_1.modules import RoutedExperts

__all__ = ["DeviceRoutedExperts", "device_experts_available"]

# Reused across instances and across layers: a `torch.cuda.Stream` is a per-process resource, and a
# 40-layer backbone would otherwise create 160 of them for the four cards to share four at a time.
_COPY_STREAMS: dict[int, torch.cuda.Stream] = {}

# The three projections, in the order the checkpoint names them. `loader.py` has the same tuple for
# the host path; the two are the same list of what an expert is made of.
PROJECTIONS = ("w1", "w2", "w3")

# The two halves of a quantized projection: the packed codes, and the E8M0 scale beside them. Every
# loop here does the same thing to both, so they travel together.
KINDS = ("q", "s")


def device_experts_available() -> bool:
    """Whether the extension this class calls is loadable and carries the op it needs."""
    extension = load_cuda_kernel()
    return extension is not None and hasattr(extension, "moe_single_token_fp4_forward")


def _copy_stream(device: torch.device) -> torch.cuda.Stream:
    """The one H2D stream for `device`, created on first use and never freed."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _COPY_STREAMS.get(index)
    if stream is None:
        with torch.cuda.device(index):
            stream = torch.cuda.Stream(device=index)
        _COPY_STREAMS[index] = stream
    return stream


class DeviceRoutedExperts(RoutedExperts):
    """One layer's routed experts held as fixed `ceil(topk / world)`-row arenas, one per card driven.

    Not an `nn.Module`, for the same reason `CheckpointRoutedExperts` is not: the experts are not
    part of the tree, and a module would put them in `state_dict()` and in `parameters()`.

    The class owns exactly the state a step needs and nothing it allocates: the pinned and device
    arenas, the copy streams, one event per pinned buffer so a buffer is not overwritten while the
    DMA that reads it is still in flight, one event per card for the result copy back, and the
    per-row scratch. Nothing on either side of PCIe is allocated per card or per row.

    `ranks` is which of the `world` dealt shares this process owns, and it is what makes the class
    usable one-card-per-process as well as one-process-for-every-card. See the module docstring for
    the deal and `partial` for what changes downstream.
    """

    def __init__(
        self,
        checkpoint,
        layer_id: int,
        *,
        n_experts: int,
        dim: int,
        inter_dim: int,
        topk: int,
        swiglu_limit: float = 0.0,
        world: int = 1,
        ranks: Sequence[int] | None = None,
        devices: Sequence[torch.device | str] | None = None,
        pinned_buffers: int = 2,
    ) -> None:
        # Bound here rather than imported at module scope: `loader.py` is what builds this class, so
        # a module-level import of it would be a cycle. By the time an instance exists `loader` is
        # fully imported and this is one attribute lookup, not one per tensor.
        from src.models.deepseek_v4_1.loader import scale_key

        if world < 1:
            raise ValueError(f"world must be at least 1, got {world}")
        if topk < 1:
            raise ValueError(f"topk must be at least 1, got {topk}")
        extension = load_cuda_kernel()
        if extension is None or not hasattr(extension, "moe_single_token_fp4_forward"):
            raise RuntimeError("moe_single_token_fp4_forward is not available in the built extension")

        self._extension = extension
        self._scale_key = scale_key
        self.checkpoint = checkpoint
        self.layer_id = layer_id
        self.n_experts = n_experts
        self.dim = dim
        self.inter_dim = inter_dim
        self.topk = topk
        self.swiglu_limit = swiglu_limit
        # `world` is how the layer's experts are dealt and `ranks` is which of those shares *this*
        # process owns. The default -- every rank of the deal, one device each -- is the
        # one-process-drives-every-card configuration this class was written for; one entry is a
        # single card in a process of its own, which is what TP4 needs.
        self.world = world
        self.ranks = list(range(world)) if ranks is None else [int(r) for r in ranks]
        if not self.ranks:
            raise ValueError("a rank list with nothing in it drives no card")
        if len(set(self.ranks)) != len(self.ranks) or any(
            not 0 <= r < world for r in self.ranks
        ):
            raise ValueError(f"ranks {self.ranks} are not distinct ranks of a world of {world}")
        self.devices = [
            d if isinstance(d, torch.device) else torch.device(d)
            for d in (devices if devices is not None else [f"cuda:{r}" for r in self.ranks])
        ]
        if len(self.devices) != len(self.ranks):
            raise ValueError(f"{len(self.devices)} devices for {len(self.ranks)} ranks")

        # Counters rather than a timing harness: how many rows and expert rows this class has moved
        # is what says whether a run is behaving like the model it started as.
        self.rows = 0
        self.expert_rows = 0

        # The split is round-robin over the sorted expert ids, so a card's widest case is
        # `ceil(topk / world)` and every row is the same shape -- which is what makes it fixed.
        self.rows_per_card = -(-topk // world)
        self._buffers = max(1, pinned_buffers)

        # Shapes from the checkpoint rather than from the config: the arena has to match the tensor
        # `copy_` is handed, so deriving it from the same place the tensor comes from means a
        # checkpoint laid out differently fails here rather than in a kernel's index arithmetic.
        self._shapes: dict[tuple[str, str], tuple[int, ...]] = {}
        for which in PROJECTIONS:
            weight = f"layers.{layer_id}.ffn.experts.0.{which}.weight"
            for kind, key in (("q", weight), ("s", scale_key(weight))):
                self._shapes[(which, kind)] = tuple(checkpoint.reader.entry(key).shape)
        self._check_shapes()

        # One pinned arena per buffer, holding **every card this process drives** laid out end to
        # end: local card `c` owns `[c * rows_per_card, (c + 1) * rows_per_card)`. It is one
        # allocation rather than one per card so that a row stages into it the same way whichever
        # card will read it, and the device side is the matching per-card slice -- a card's arena is
        # `rows_per_card` rows and not the whole token's.
        self._pinned = [
            {
                (which, kind): torch.empty(
                    (len(self.ranks) * self.rows_per_card,) + self._shapes[(which, kind)],
                    dtype=torch.uint8,
                    pin_memory=True,
                )
                for which in PROJECTIONS
                for kind in KINDS
            }
            for _ in range(self._buffers)
        ]
        self._on_device = []
        for device in self.devices:
            with torch.cuda.device(device):
                self._on_device.append(
                    {
                        (which, kind): torch.empty(
                            (self.rows_per_card,) + self._shapes[(which, kind)],
                            dtype=torch.uint8,
                            device=device,
                        )
                        for which in PROJECTIONS
                        for kind in KINDS
                    }
                )
        # `None` means "no copy in flight from this buffer", which is the state a fresh buffer is in.
        # The events themselves are allocated here and reused: one per card per buffer, 8 for the
        # four-card case, rather than one per copy per row, which would be 320 a token.
        self._events: list[list[torch.cuda.Event]] = []
        for device in self.devices:
            with torch.cuda.device(device):
                self._events.append([torch.cuda.Event() for _ in range(self._buffers)])
        self._uploaded: list[list[bool]] = [[False] * self._buffers for _ in self.ranks]
        self._next_buffer = 0

        # One event per card for the *result* copy back, reused the same way. `_issue` records it
        # after that card's D2H is issued and `_drain` waits on it once every card has been issued, so
        # the wait costs what the slowest card costs and not the sum of four.
        self._drained: list[torch.cuda.Event] = []
        for device in self.devices:
            with torch.cuda.device(device):
                self._drained.append(torch.cuda.Event())

        # The per-row buffers, built on the first row and reused after it. See `_row_scratch`.
        self._scratch: dict | None = None

        # The routing as host memory, one row set at a time. Allocated on the first `forward` and
        # kept, because the alternative is a pinned allocation per layer per token. See `_route_ids`.
        self._route: torch.Tensor | None = None

    # -- what the checkpoint gave us ---------------------------------------------------------

    def _expected(self, which: str) -> tuple[int, int]:
        """The `[rows, K/2]` an expert's `which` projection must be, from this layer's geometry."""
        if which == "w2":
            return self.dim, self.inter_dim // 2
        return self.inter_dim, self.dim // 2

    def _check_shapes(self) -> None:
        """Refuse a checkpoint whose expert is laid out the other way round, before any kernel runs.

        `w1` is `[inter, dim]` and `w2` is `[dim, inter]`, so a transposed release would still give
        every arena a plausible shape and every kernel a plausible answer. The two widths are the
        only thing that distinguishes them, so they are what this checks -- and the message names
        both orders, because "which one is right" is the whole question when this fires.
        """
        for which in PROJECTIONS:
            codes = self._shapes[(which, "q")]
            scales = self._shapes[(which, "s")]
            expected = self._expected(which)
            if codes != expected:
                transposed = (expected[1], expected[0])
                hint = " -- the two orders are swapped" if codes == transposed else ""
                raise ValueError(
                    f"layer {self.layer_id}'s {which} is {codes} where a {self.dim}-wide layer with "
                    f"a {self.inter_dim}-wide expert needs {expected}{hint}"
                )
            # One scale byte per 32 codes along K, and two codes per byte, so a scale row is a
            # sixteenth of the packed row it sits beside.
            if scales != (codes[0], codes[1] // 16):
                raise ValueError(
                    f"layer {self.layer_id}'s {which} is {codes} with a {scales} scale; a 32-wide "
                    f"block grid over {codes[1] * 2} codes needs {(codes[0], codes[1] // 16)}"
                )

    @property
    def partial(self) -> bool:
        """Whether a `forward` is one rank's share of the routed sum rather than all of it.

        It is `ranks` and not `world` that decides: an instance driving every rank of the deal sums
        them here and returns the whole routed sum, which is what the one-process configuration does
        and what `RoutedExperts.partial` documents as the default. An instance driving one of them
        returns a partial, and the `MoE` above has to complete it -- which it does on the same
        all-reduce that carries the shared expert's, since both are cut the same way.
        """
        return len(self.ranks) < self.world

    @property
    def arena_bytes(self) -> int:
        """Packed fp4 bytes this instance holds per side of PCIe, summed over its cards."""
        per_card = sum(
            self.rows_per_card * shape[0] * shape[1] for shape in self._shapes.values()
        )
        return per_card * len(self.ranks)

    # -- the plan ---------------------------------------------------------------------------

    def _split(self, ids: Sequence[int]) -> list[list[tuple[int, int]]]:
        """Per card this process drives, the `(arena row, route slot)` pairs that card owns.

        Sorted by global expert id and dealt round-robin, which makes the split a property of the
        routing rather than of the order the gate happened to emit it in: two runs that route to the
        same six experts stage the same bytes into the same rows. The host path walks a layer's
        experts in id order for the same reason.

        Round-robin over `world` and *select* the ranks this process owns, rather than dealing over
        `len(self.ranks)`: the deal is a property of the layer and has to come out the same in every
        process, or two ranks would stage the same expert into different rows and the all-reduce
        would sum the layer twice and never once.
        """
        order = sorted(range(len(ids)), key=lambda slot: ids[slot])
        cards: list[list[tuple[int, int]]] = [[] for _ in self.ranks]
        for position, slot in enumerate(order):
            if position % self.world in self.ranks:
                cards[self.ranks.index(position % self.world)].append((position // self.world, slot))
        return cards

    def _key(self, expert: int, which: str) -> str:
        return f"layers.{self.layer_id}.ffn.experts.{expert}.{which}.weight"

    def _take_buffer(self) -> int:
        """The next pinned buffer to stage into, after the DMA that last read it has finished.

        The pinned arena is what the H2D copies read from, and the host runs ahead of the cards, so
        a buffer two rows old is the earliest one whose copy has a chance of being done. Waiting on
        its event turns that into a guarantee; with two buffers and the copy taking about as long as
        the staging, the wait is usually already satisfied and costs nothing.
        """
        slot = self._next_buffer
        self._next_buffer = (self._next_buffer + 1) % self._buffers
        for card in range(len(self.ranks)):
            if self._uploaded[card][slot]:
                self._events[card][slot].synchronize()
                self._uploaded[card][slot] = False
        return slot

    def _row(self, card: int, arena_row: int) -> slice:
        """The pinned rows the local card `card`'s `arena_row` lives in."""
        start = card * self.rows_per_card + arena_row
        return slice(start, start + 1)

    def _stage(self, buffer: int, ids: Sequence[int], cards: list[list[tuple[int, int]]]) -> None:
        """One `copy_` per tensor out of the host source into the row an expert owns.

        `checkpoint.packed` is the subject here, and it decides the source: the shard mapping, or the
        resident bank when one is attached. Out of the mapping it uses `entry_view` and not `view`,
        deliberately -- uncached, which is exactly right for a 3 MiB expert that will not be read
        again this step, and it is why this loop is one pass over the shards' address space and not a
        growing set of handles on whole layers. Out of the bank the mapping's slowest case -- a cold
        page, 1308.0 ms for a row against 11.2 ms warm -- stops being possible, because the bank was
        filled from the disk once, before the first step. It is not a faster `copy_`: 14 GiB/s either
        way. What it removes is the disk.
        """
        for card, members in enumerate(cards):
            arena = self._pinned[buffer]
            for arena_row, slot in members:
                for which in PROJECTIONS:
                    weight = self._key(ids[slot], which)
                    for kind, key in (("q", weight), ("s", self._scale_key(weight))):
                        # F8_E8M0 has no CPU `copy_` from a pyloaded view, so both halves travel as
                        # the `uint8` the kernel reads them as.
                        arena[(which, kind)][self._row(card, arena_row)].copy_(
                            self.checkpoint.packed(key).view(torch.uint8)
                        )
            self.expert_rows += len(members)

    def _upload(self, buffer: int, cards: list[list[tuple[int, int]]]) -> None:
        """One asynchronous copy stream per card, each recording an event the host waits on later.

        Four links at once is the whole point of the split: measured, one card takes 10.47 GiB/s and
        four take 38.56 GiB/s aggregate, so the same bytes cost 430 ms staged to one card and 117 ms
        staged to four.

        A card's copy is also ordered behind that card's own compute stream, which is where its
        previous kernel was launched: without that the copy could refill an arena a kernel is still
        reading, and the two are on different streams so nothing else would order them.
        """
        for card, members in enumerate(cards):
            if not members:
                continue
            device = self.devices[card]
            stream = _copy_stream(device)
            rows = slice(card * self.rows_per_card, (card + 1) * self.rows_per_card)
            with torch.cuda.device(device):
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for key, destination in self._on_device[card].items():
                        destination.copy_(self._pinned[buffer][key][rows], non_blocking=True)
                    self._events[card][buffer].record(stream)
            self._uploaded[card][buffer] = True

    def _row_scratch(self, x_row: torch.Tensor) -> dict:
        """The per-row buffers, built on the first row and reused for every one after it.

        A row is 5120 fp32 of weights, 5120 bf16 of activation and 20 KiB of result per card, so
        allocating them per call looks free and is not: measured on this host, the six allocations
        and four pageable transfers a card-call cost 145 us of a 1.7 ms call, and -- more to the
        point -- a pageable copy is *synchronous*. `_issue` below is arranged around that second
        fact, and this is the rest of it.

        Pinned and not pageable on the host side, one activation for the whole row rather than one
        per card, and the route weights in the order the cards read them.
        """
        if self._scratch is not None and self._scratch["x"].dtype == x_row.dtype:
            return self._scratch
        dim = self.dim
        scratch = {
            "x": torch.empty((1, dim), dtype=x_row.dtype, pin_memory=True),
            "w": torch.empty((self.topk,), dtype=torch.float32, pin_memory=True),
            "x_device": [torch.empty((1, dim), dtype=x_row.dtype, device=d) for d in self.devices],
            "w_device": [
                torch.empty((self.rows_per_card,), dtype=torch.float32, device=d)
                for d in self.devices
            ],
            # `experts_start_idx` is 0 and the arena's ids are relative -- a card's rows are 0 and 1
            # -- so this is the whole index arithmetic a call needs, and it never changes.
            "idx_device": [
                torch.arange(self.rows_per_card, dtype=torch.int64, device=d)
                for d in self.devices
            ],
            "y_device": [None] * len(self.devices),
            "y": [torch.empty((1, dim), dtype=torch.float32, pin_memory=True) for _ in self.devices],
        }
        self._scratch = scratch
        return scratch

    def _issue(
        self,
        x_row: torch.Tensor,
        weights_row: torch.Tensor,
        cards: list[list[tuple[int, int]]],
    ) -> list[int]:
        """One kernel call per local card, all of them issued before any is drained, and none waited on.

        This is the half of what used to be `_launch` that talks to the card; `_drain` is the half
        that waits. The split exists so that a row's card-side time and the *next* row's host-side
        staging can be in flight together -- see `forward`.

        The sum is over the cards *this process drives*, and `partial` says what that sum is: with
        every rank of the deal in `ranks` it is the layer's whole routed output, and with one it is
        that rank's share, which the `MoE` above completes on the same all-reduce as the shared
        expert's. Nothing else about a row changes -- the arena is the same, the deal is the same,
        and a card that is not in `ranks` is staged by whoever owns it.

        The arena's expert ids are relative -- a card's rows are 0 and 1 -- so the 384-expert space
        never reaches a kernel, and `experts_start_idx` is 0 because there is nothing to rebase:
        `local` is only ever an arena row. A card holding one expert is called with one index rather
        than with a zero second row, so the unused row is never read.

        The routing weights are gathered on the host, in the same order the indices are handed over:
        route `r` of the call is arena row `r`, which is `members[r]`, so the two have to be permuted
        together or the kernel would scale one expert's output by another's weight -- a wrong answer
        that no shape check would catch.

        **Issue, then drain.** The two loops below are the same four calls the card loop used to make
        one at a time, and the split between them is the whole of what a card is worth here: a
        pageable D2H cannot return until the kernel that produced it has finished, so draining card 0
        inside the loop left cards 1..3 unlaunched until card 0 was done, and a row cost four kernels
        added up instead of one plus copies. Measured on this host with the real arenas, one row of
        four calls: **3.63 ms** drained in the loop, **1.01 ms** with the drain moved after it, and
        **3.45 ms** with the allocation and the pageable copies removed but the drain left in place --
        so 3.42x of the 3.60x is the ordering and not the transfers.

        What is left here is all issue: the copies are `non_blocking` into buffers that are already
        pinned and already sized, the kernel is launched, the D2H that reads the result is issued and
        an event recorded. Nothing below blocks: the activation, the route weights and the result all
        cross the link as issued copies whose only reader is the next copy on the same stream. The
        routing used to block here and does not any more -- `_route_ids` takes it once for the whole
        row set, ahead of the loop, precisely so that nothing in the loop has to.
        """
        scratch = self._row_scratch(x_row)
        # The activation goes to every card, and with the dense tree on a card it is already on one:
        # then it is a device-to-device copy of the same 10 KiB with no pageable D2H in between, and
        # a pageable copy is synchronous -- which is the one cost the issue-then-drain ordering below
        # exists to keep off the per-row path.
        source = scratch["x"]
        if x_row.is_cuda:
            source = x_row.reshape(1, -1)
        else:
            scratch["x"].copy_(x_row)
        # One gather for the row, in card order, so card `c`'s weights are a contiguous slice. The
        # permutation follows the weights to whichever device the gate produced them on.
        perm = torch.tensor(
            [slot for members in cards for _, slot in members],
            dtype=torch.int64,
            device=weights_row.device,
        )
        # `perm` is the routes *this process* holds, which under a deal is fewer than `topk` -- the
        # else of them are a sibling rank's arena rows and its share of the sum. `scratch["w"]` is
        # sized for `topk` because that is the widest a row's own routes can be, so the copy takes a
        # view of it rather than the whole buffer. Non-blocking: the six values land in pinned memory
        # and the only reader is the H2D below, on this same stream, so ordering them is the stream's
        # job and not the host's -- which is what keeps this from being the one sync left in the loop.
        scratch["w"].narrow(0, 0, perm.numel()).copy_(
            weights_row.reshape(-1).index_select(0, perm), non_blocking=True
        )

        issued: list[int] = []
        offset = 0
        for card, members in enumerate(cards):
            k = len(members)
            if not k:
                continue
            device = self.devices[card]
            # Order the compute behind its own copy stream and no one else's: a card's kernel reads
            # only its own arena, so four independent chains is what the hardware is.
            torch.cuda.current_stream(device).wait_stream(_copy_stream(device))
            with torch.cuda.device(device):
                scratch["x_device"][card].copy_(source, non_blocking=True)
                scratch["w_device"][card].narrow(0, 0, k).copy_(
                    scratch["w"].narrow(0, offset, k), non_blocking=True
                )
            offset += k
            scratch["y_device"][card] = self._extension.moe_single_token_fp4_forward(
                scratch["x_device"][card],
                scratch["idx_device"][card].narrow(0, 0, k),
                scratch["w_device"][card].narrow(0, 0, k),
                self._on_device[card][("w1", "q")], self._on_device[card][("w1", "s")],
                self._on_device[card][("w2", "q")], self._on_device[card][("w2", "s")],
                self._on_device[card][("w3", "q")], self._on_device[card][("w3", "s")],
                0,
                float(self.swiglu_limit),
            )
            issued.append(card)

        for card in issued:
            with torch.cuda.device(self.devices[card]):
                scratch["y"][card].copy_(scratch["y_device"][card], non_blocking=True)
                self._drained[card].record()
        if not issued and not self.partial:
            # Under a deal a rank can hold none of a row's routes -- `topk` below `world`, or a
            # routing that landed every one of them on a sibling. That rank's share of the sum is
            # zero, and returning it is what keeps it in the all-reduce the `MoE` is about to run;
            # raising here would take the process group down over an answer that is not an error.
            # Without a deal there is no sibling to hold the routes, so nobody routed is a real one,
            # and it is raised here rather than in `_drain` so that it names the row it happened on.
            raise RuntimeError(f"layer {self.layer_id} routed a row to no expert at all")
        return issued

    def _drain(self, issued: list[int]) -> torch.Tensor:
        """The partials `_issue` left in flight, waited for and summed on the host.

        All the waits come after all the issues -- see `_issue` -- so this costs what the slowest
        card costs and not the sum of four. A row that reached none of this process's cards sums to
        zero rather than to nothing, because zero is the rank's share of the all-reduce.
        """
        if not issued:
            return torch.zeros(self.dim, dtype=torch.float32)
        y = None
        for card in issued:
            self._drained[card].synchronize()
            y = self._scratch["y"][card] if y is None else y + self._scratch["y"][card]
        # A tensor of its own and not the view of a reused buffer: the caller assigns this into its
        # own `[n, dim]`, and one that kept the result would otherwise watch the next row overwrite
        # it. `_issue` may already have started filling that buffer for the next row by the time a
        # pipelined caller adds this one up, which is the same reason.
        return y.squeeze(0).clone()

    # -- one row ----------------------------------------------------------------------------

    def forward(self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """`x` is `[n, dim]` bf16, `weights`/`indices` are `[n, topk]`. Returns `[n, dim]` fp32.

        A MoE is a row-wise map, so `n` is a loop here and not a batch: the kernel is single-token,
        which is what the arenas are sized for. See the module docstring for what that costs a
        prefill and what replaces it.
        """
        n, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"x is [_, {dim}] where this layer is {self.dim}-wide")
        if indices.shape[1] != self.topk:
            raise ValueError(
                f"a row routed to {indices.shape[1]} experts where this layer has {self.topk}"
            )
        y = torch.empty((n, dim), dtype=torch.float32)
        # The routing comes back to the host once, before the first row, and not once a row inside the
        # loop: under the pipeline below the card's stream already has the previous row's kernel on
        # it when a row would ask, so a per-row read is a per-row wait. See `_route_ids`.
        route = self._route_ids(indices)
        # One row in flight, and the row loop is arranged so that the *host* work of row `k+1` runs
        # while row `k`'s card-side chain does. The order inside the loop is what does it: the stage
        # and the H2D are issued for row `k+1` first, and only then is row `k` drained -- so the wait
        # that used to open every row is entered after the staging that can fill it. Measured at 32
        # and 128 tokens of prefill on layer sizes of the real model, the drain falls from 3.75 ms a
        # row-layer to 1.8 and the stage rises from 5.35 to 6.3, for 1.10x of wall clock either way.
        # See the module docstring for the table and for what it does not reach.
        pending: list[int] | None = None
        for row in range(n):
            cards = self._stage_row(x[row], weights[row], route[row])
            if pending is not None:
                y[row - 1] = self._drain(pending)
            pending = self._issue(x[row], weights[row], cards)
        if pending is not None:
            y[n - 1] = self._drain(pending)
        # The result belongs where the activation came from. With the tree on a card this is the
        # half of the layer that follows: the `MoE` above adds the shared expert's output and hands
        # the sum to the all-reduce, and a CPU tensor would make that line a cross-device add. The
        # sum itself is still the host's -- each card's copy is drained into pinned memory and added
        # there, which is what `_issue` and `_drain` measure between them -- so this is one H2D of
        # `[n, dim]` per layer and not a fourth card in the reduction.
        return y if x.device.type == "cpu" else y.to(x.device)

    def _route_ids(self, indices: torch.Tensor) -> torch.Tensor:
        """The routing as host memory, in one copy, before the first row of the set is opened.

        `indices_row.tolist()` inside the row loop synchronizes the card's stream, and under the row
        pipeline that stream already has the *previous* row's kernel on it. Measured at 32 tokens of
        prefill: those reads cost **2.9 ms a row** there against **0.1 ms** in a loop that has already
        drained its last row, so as much as the pipeline saves on the drain it gives back here -- and
        that is what the first version of it did, to 1.01x. One copy of the whole `[n, topk]` ahead of
        the loop costs the gate's latency once, which is a wait the loop was already paying for its
        first row, and makes every row's read a read of host memory.

        `weights` is deliberately left on the card. `_issue` indexes it there and copies six values
        back on the same stream, so it is a copy the host never waits for once it is issued
        non-blocking, and pulling the whole `[n, topk]` across would be a second round trip for
        scalars the card already has.
        """
        if indices.device.type == "cpu":
            return indices
        host = self._route
        if host is None or host.shape != indices.shape or host.dtype != indices.dtype:
            host = torch.empty(indices.shape, dtype=indices.dtype, pin_memory=True)
            self._route = host
        host.copy_(indices, non_blocking=True)
        # The one wait: the copy is ordered behind whatever the gate produced `indices` from, which
        # is earlier in the layer and not part of this loop. Nothing of this class is on the stream
        # yet, which is the whole difference between this sync and the one it replaces.
        torch.cuda.current_stream(indices.device).synchronize()
        return host

    def _stage_row(
        self, x_row: torch.Tensor, weights_row: torch.Tensor, ids_row: torch.Tensor
    ) -> list[list[tuple[int, int]]]:
        """Put one row's bytes where its cards can reach them, and return the deal they were put in.

        Everything here is host-side and none of it reads a result: the routing, the arena row, the
        pinned staging and the H2D. That is what lets `forward` run it ahead of the *previous* row's
        drain, and it is the whole of why this is not inside `_issue`. `ids_row` is a row of the host
        copy `_route_ids` took, so the list comprehension below is a read of host memory.
        """
        ids = [int(e) for e in ids_row.tolist()]
        cards = self._split(ids)
        buffer = self._take_buffer()
        self._stage(buffer, ids, cards)
        self._upload(buffer, cards)
        self.rows += 1
        return cards
