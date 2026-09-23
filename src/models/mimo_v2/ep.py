"""The routed experts dealt out over the ranks, and the one collective a routed layer needs.

A decode step's expert traffic is the thing this module exists to divide. On one rank a step draws
eight experts a layer out of the host bank, which is 102 MiB a layer and **4.68 GiB a token**, and
at the 10.4 GiB/s the pinned link sustains that is 450 ms of a 610 ms token -- 73.8% of it, and
more than everything else the step does put together. Nothing about the arithmetic changes when the
experts are dealt out: each rank still holds the same draw, stages what it owns and computes it.

**The router is replicated, and that is load-bearing.** Every rank runs the same gate over the same
hidden state and draws the same eight experts, so there is no dispatch message, no routing
collective and no all-to-all: the only thing that crosses the fabric is the sum of what came back.
That is affordable for two reasons. The gate is a `[1, 4096] x [4096, 256]` matmul against the
layer's own bf16 copy, which is nothing beside the expert bytes. And the draw is a *deterministic
function of the row*: `gate_and_route` is one function that every rank runs in the same dtype on the
same bytes, so every rank agrees on the draw without being told it. A path that had to guess would
need the ids on the wire and the whole design would change.

**The one collective is the layer's own output.** The expert kernel sums a weighted set of drawn
rows, so a rank holding a subset of the draw holds a *partial* sum and the layer's answer is the
sum of the partials: one `all_reduce` of `[1, 4096]` fp32 a routed layer -- sixteen kilobytes,
forty-seven times a token, about 1.4 ms of the step in all. That price is what makes the deal a
modulus instead of a dispatcher, and it does not grow with the number of experts.

Two shapes of deal, because they pay in different places.

* `sorted` gives sorted position `p` to rank `p % world`, which is a property of the *routing*
  rather than of the drawn ids, so a top-8 draw over four ranks is exactly **2, 2, 2, 2**: no
  variance, no empty rank, and an arena of `ceil(top_k / world)` rows. This is the default.
  Measured on four cards, a decode step is **275.2 ms** under it against **348.6** under `id`,
  and the whole of the difference is that the deal makes the ranks equal: a per-layer collective
  charges every rank the *widest* draw's work, and the widest of four ranks under `id` is 3.4
  experts on average where `sorted` is exactly 2.
* `id` gives the drawing to rank `expert % world`. It partitions the **experts** over the ranks, so
  a rank is ever dealt from `n_experts / world` of them, and a chunk that draws nearly every expert
  -- which is what a prefill chunk does, 4096 tokens at top-8 being 32768 draws over 256 experts --
  stages a quarter of the bytes rather than all of them. Under `sorted` a chunk's sorted positions
  reach every rank, so the rank stages the whole expert set: the prefill's deal is `id`, and this
  default is a decode default that the prefill path overrides per module -- `owned_experts` is
  where that share is computed and `MimoV2DeviceExperts.forward_chunk` is where it is spent.

**The empty rank is why the deal is not a detail.** Under `id` with `top_k` 8 and a world of 4, a
rank owns nothing `(3/4)^8` of the time -- 10% of the draws -- and it still has to arrive at the
collective with a zero, because the collective is unconditional. Under `sorted` the case cannot
happen. Neither is a corner to be excused: the first is one draw in ten and the second is most of
the bytes a chunk moves.

**The second job this module has is the attention.** A decode step at a long context is a host
bound step, and the largest single thing on the host is the attention: at 262144 keys the nine
global layers are 12.5 ms a layer, and a quarter of a layer -- sixteen of its sixty-four query
heads, with the one key head and the value head they attend to -- is 3.0 ms. The checkpoint is
already partitioned that way (`quant.QKV_SHARDS`, four groups of `[q | k | v]`), so the split costs
no arithmetic agreement and no weight remapping: `attention_shards` decides how many pieces, and
the pieces are joined by `make_all_gather`, which is the reason the split is exact rather than
merely close. Four is also the number of ranks here, which is a coincidence of this machine and not
of the design -- a world of two replicates the attention and deals the experts, which is correct
and is what the prefill test uses.

`world=1` is not a special case anywhere. No rank divides anything by a deal, `make_all_reduce`
returns `None`, and a partial is the whole. That is the control column, and it is free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Sequence

import torch

__all__ = [
    "ATTENTION_SHARDS",
    "DEALS",
    "DEAL_ENV",
    "EpGroup",
    "SHARDS_ENV",
    "attention_shards",
    "deal_card",
    "deal_rule",
    "make_all_gather",
    "make_all_reduce",
    "owned_experts",
    "owned_positions",
    "rows_per_card",
]

#: How a draw's experts are shared out. See the module docstring for what each costs.
DEAL_ENV = "POCKETLLM_MIMO_EXPERT_DEAL"
DEALS = ("sorted", "id")


def deal_rule() -> str:
    """Which deal a module makes unless its constructor is told. `sorted` by default.

    The default is the *decode* deal, because decode is the only path that exists: it is 27% faster
    than `id` on four cards where the attention is replicated, and 41 to 53% where it is split, and
    the reason is in the module docstring. `id` is the prefill deal, and
    a prefill stage will have to say so -- a variable a run does not set must not be able to change
    what a run computes, but this one does change it, so the prefill path names its deal and does
    not inherit this one.
    """
    asked = os.environ.get(DEAL_ENV, "").strip().lower()
    return asked if asked in DEALS else "sorted"


def deal_card(expert: int, position: int, *, deal: str, world: int) -> int:
    """Which rank of the deal owns one drawing of `expert`.

    `position` is the drawing's place among the draw's ids *sorted*, which is what `sorted` reads
    and what makes that deal independent of the order the gate emitted the experts in. `id` reads
    the expert's own number, so position is not part of its rule and one draw can hand a single
    rank all `top_k` of its drawings -- which is why `rows_per_card` is `top_k` under it.
    """
    if deal == "id":
        return int(expert) % world
    if deal == "sorted":
        return int(position) % world
    raise ValueError(f"{deal!r} is not a deal; {DEALS} are")


def rows_per_card(deal: str, top_k: int, world: int) -> int:
    """How wide one rank's arena has to be under `deal`, for the drawings it is dealt.

    `sorted` deals every rank at most one drawing of each of the `world` slots it can own, so
    `ceil(top_k / world)` covers it exactly and a top-8 draw over four ranks is two rows. `id` has
    no such ceiling -- a draw's eight experts may all be congruent mod `world` -- so the only width
    that always covers a draw is `top_k`. This is the `id` deal's whole cost and it is paid in
    arena rows.
    """
    if world < 1:
        raise ValueError(f"world must be at least 1, got {world}")
    if deal == "id":
        return int(top_k)
    if deal == "sorted":
        return -(-int(top_k) // world)
    raise ValueError(f"{deal!r} is not a deal; {DEALS} are")


def owned_positions(indices: Sequence[int], *, rank: int, world: int, deal: str) -> list[int]:
    """Which of a draw's `top_k` positions this rank computes, in the order the draw gave them.

    The positions and not the ids, because the kernel is handed arena rows in draw order and the
    weight a row is multiplied by is the draw's own weight: a caller that reordered the positions
    would have to reorder the weights with them, and this way it does not.

    Sorted order is computed here rather than sent, because every rank holds the same draw and a
    sort of eight numbers is not worth a message. Duplicate ids cannot occur -- a top-k is over
    distinct experts -- so the sort needs no tie-break and is deterministic.
    """
    if world <= 1:
        return list(range(len(indices)))
    if deal == "sorted":
        order = sorted(range(len(indices)), key=lambda position: indices[position])
        return [position for slot, position in enumerate(order) if slot % world == rank]
    return [position for position, expert in enumerate(indices) if expert % world == rank]


def owned_experts(n_experts: int, *, rank: int, world: int, deal: str) -> list[int]:
    """Every expert of the layer's set this rank can be dealt, ascending.

    `owned_positions` answers the question for *one* drawing, which is what a decode step is. This
    answers it for a chunk, and a chunk is a different question with a different answer, because a
    chunk draws nearly every expert there is: 4096 tokens at top-8 is 32768 drawings over 256
    experts, so "the experts this rank can be dealt" is not "the experts one draw reaches" but the
    whole set the deal gives it a share of.

    The two deals divide that set very differently, and this is where the prefill's arithmetic
    lives. `id` gives a rank the experts congruent to it, a quarter of them at a world of four,
    and that quarter *is* the prefill's win: a quarter of the bytes, a quarter of the arithmetic,
    one collective to close it. `sorted` gives a rank a position within each drawing, which for
    one token is exactly two experts and for a chunk of four thousand tokens is all of them: a
    `sorted` chunk stages the whole set on every rank, so it costs one rank's copy four times over
    and saves nothing. Not a wrong answer -- the partials still sum -- but the wrong deal, which
    is why `MimoV2DeviceExperts.forward_chunk` refuses it rather than quietly paying it.
    """
    if world < 1:
        raise ValueError(f"world must be at least 1, got {world}")
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} is not a rank of a world of {world}")
    if deal == "id":
        return [expert for expert in range(n_experts) if expert % world == rank]
    if deal == "sorted":
        # Every position of a chunk's drawings, and therefore every expert, on every rank.
        return list(range(n_experts))
    raise ValueError(f"{deal!r} is not a deal; {DEALS} are")


def make_all_reduce(world: int) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """`dist.all_reduce` as a closure, injected rather than imported.

    Same shape as `src/models/deepseek_v4_1/tp.py:make_all_reduce`, for the same reason: nothing in
    the model layer imports `torch.distributed`, so a single-card run pays nothing for the
    collective machinery existing, and a caller that has already built a process group decides how
    a partial is summed rather than inheriting a decision. `world=1` returns `None`, which is what
    keeps the unsharded configuration the same code path as the control.

    fp32 on the wire, the caller's dtype on either side of it. The message is a routed layer's
    `[1, 4096]`, sixteen kilobytes, which is not the message V4.1's `REDUCE_BITS_ENV` exists to
    argue about -- that one is an 80 MiB activation tile where the wire dtype halves a real cost.
    Here a wire dtype would be a second arithmetic behind a parity check for a fraction of a
    millisecond, so there is no variable to set.
    """
    if world <= 1:
        return None

    def reduce(tensor: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        wide = tensor.float().contiguous()
        dist.all_reduce(wide, op=dist.ReduceOp.SUM)
        return wide.to(tensor.dtype)

    return reduce


#: How many ranks the attention is split over, when it is split at all. Four is the
#: checkpoint's own number and not a choice: the fused `qkv_proj` is stored as four shards of
#: `[q | k | v]` (`quant.QKV_SHARDS`) and `fused_qkv_row_order` refuses any other reading, so a
#: rank of four takes one contiguous quarter of the projection's rows and the query heads, key
#: heads and value heads that go with it -- see `attention_shards`.
ATTENTION_SHARDS = 4

#: Turns the split off, for the A/B and for a host where it turns out not to pay. Unset means the
#: checkpoint's own four at a world of four; `1` means the whole attention on every rank, which is
#: what this model did before the split existed. See `attention_shards`.
SHARDS_ENV = "POCKETLLM_MIMO_ATTENTION_SHARDS"


def attention_shards(world: int) -> int:
    """How many pieces to cut the attention into at this world size: one, four, or one.

    A world of four is the checkpoint's own partition and gets it. A world of one is the control
    column and needs nothing. Anything else -- two, three -- has no partition the weights admit,
    so the attention stays whole and replicated and the experts are dealt as usual: correct, and
    the same shape of run this module had before the attention could be split at all. A world that
    *could* be split and is not is a slower run, not a wrong one, which is why this answers with a
    number rather than raising.

    `SHARDS_ENV` can hold the answer back to one, which is the only value it may ask for besides
    four: one is an arm of an A/B and four is the default, and any other number is not a partition
    this checkpoint admits, so it is ignored rather than refused.

    **The knob moves the clock and not the answer.** A share's projection is the whole's rows for
    that share bit for bit, and the shares' outputs are concatenated rather than projected and
    summed, so the joined answer is the whole path's answer -- exactly, for every chunk of queries
    and for a decode step of a global layer past `FOLD_KEYS` keys. The one place it is not exact is
    a decode step of a *windowed* layer, whose folded path is batched over the key heads and whose
    output gemm cuBLAS therefore tiles differently for a share's two heads than for the layer's
    eight: measured on layer 1 of the release, `1.2e-07` of a `5.1e-01` peak in `pre_o`, the last
    bit of a float32. `probe_mimo_v2_split_tokens.py` is what says no *token* moves: at 8192 tokens
    of context, on a released prompt and on a drawn one, the greedy streams and the prefill's
    logits are identical with the split and without it.
    """
    if world < 1:
        raise ValueError(f"world must be at least 1, got {world}")
    if world != ATTENTION_SHARDS:
        return 1
    asked = os.environ.get(SHARDS_ENV, "").strip()
    return 1 if asked == "1" else ATTENTION_SHARDS


def make_all_gather(world: int) -> Callable[[torch.Tensor], torch.Tensor] | None:
    """`dist.all_gather_into_tensor` as a closure, the attention's own way of joining its rank.

    A split attention produces `[rows, o_in / world]` a rank -- its own query heads' output, before
    the projection that mixes them -- and the layer's answer is those pieces *concatenated*, not
    summed: `o_proj` reads all `o_in` of its input at once, so the pieces have to be back in one
    tensor before it runs. The head-major layout wants rank `r`'s piece at columns
    `[r * o_in / world, (r + 1) * o_in / world)`, which is the concatenation along the *last* axis.

    **`all_gather_into_tensor` joins along the first.** It takes a buffer of `world * rows` rows and
    hands rank `r` the rows `[r * rows, (r + 1) * rows)`, so a caller that allocates the output it
    wants (`[rows, world * width]`) gets a *silent* scramble for any `rows` over one: the pieces are
    all there, each in the wrong place, and the output is the same size and made of the same
    numbers. It is exact for a decode step, where `rows` is one and the flat layout happens to be
    the concatenation, which is the worst possible failure mode for this bug -- a longer prompt
    prefill is where it goes wrong and a token-at-a-time run never sees it. Hence the transpose
    below, which is not an optimisation of the join but the join.

    **Gathered rather than reduced, and that is an exactness argument.** The other way to join them
    is to let each rank run its own quarter of `o_proj` and sum the four partials in fp32, which is
    how the routed experts' split is joined. It is not the same number: each rank's partial is
    rounded to bfloat16 before it is summed, so the answer rounds four times where the whole path
    rounds once. Measured on layer 0 of the release at 32768 keys, the reduced join is 7.5e-3 of
    the attention output's own peak and the gathered one is `0.00e+00` -- bit for bit the whole
    path's answer. The two cost the same on the wire: sixteen kilobytes a token either way, a
    `[1, 4096]` float32 reduction against a `[1, 8192]` bfloat16 gather.

    `world=1` returns `None`, which is the identity and not a collective.
    """
    if world <= 1:
        return None

    def gather(part: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        if part.dim() != 2:
            raise ValueError(f"a gathered piece is [rows, width], got {tuple(part.shape)}")
        part = part.contiguous()
        rows, width = part.shape
        flat = torch.empty((world * rows, width), dtype=part.dtype, device=part.device)
        dist.all_gather_into_tensor(flat, part)
        return flat.view(world, rows, width).transpose(0, 1).reshape(rows, world * width)

    return gather


@dataclass
class EpGroup:
    """Which rank this process is, and how a partial becomes the answer.

    Held by the model rather than by the experts: the experts return *this rank's share* and the
    layer sums it, so the collective and the arithmetic it completes sit in the same place, and a
    module that is asked for its share cannot silently be asked for the world's.

    `reduce` is required as soon as the world is more than one. A group of four with no way to sum
    is not a configuration, it is a wrong answer that costs one collectible per layer to produce.
    """

    world: int = 1
    rank: int = 0
    reduce: Callable[[torch.Tensor], torch.Tensor] | None = None
    gather: Callable[[torch.Tensor], torch.Tensor] | None = None
    device: torch.device | None = None

    def __post_init__(self) -> None:
        self.world = int(self.world)
        self.rank = int(self.rank)
        if self.world < 1:
            raise ValueError(f"world must be at least 1, got {self.world}")
        if not 0 <= self.rank < self.world:
            raise ValueError(f"rank {self.rank} is not a rank of a world of {self.world}")
        if self.world > 1 and self.reduce is None:
            raise ValueError(
                f"a world of {self.world} needs a way to sum a partial; `make_all_reduce` is one"
            )
        if self.device is not None:
            self.device = torch.device(self.device)

    @property
    def partial(self) -> bool:
        """Whether one rank's output is a summand rather than the sum."""
        return self.world > 1

    @property
    def attention_shards(self) -> int:
        """How many pieces this group's attention is cut into. See `attention_shards`.

        A group that holds no way to join the pieces does not split the attention -- which is what a
        caller that injects its own collectives, or its own arithmetic in place of them, gets: the
        model keeps the whole attention on every rank, which is what it did before the split
        existed. Splitting is not something a group can be *told* to do and then fail to finish.
        """
        return attention_shards(self.world) if self.gather is not None else 1

    @property
    def attention_shard(self) -> int:
        """Which piece this rank computes, or zero for a group whose attention is not cut.

        A rank is a piece index only when the attention is actually cut: the piece and the count
        come from this one rule so that a caller cannot take one without the other, which would be
        a rank asking for the fourth quarter of a tensor that was never divided.
        """
        return self.rank if self.attention_shards > 1 else 0

    @classmethod
    def from_env(cls, *, device: torch.device | str | None = None, timeout_hours: float = 2.0):
        """The group `torchrun` left in the environment, with the process group built if needed.

        Reads `WORLD_SIZE`/`RANK`/`LOCAL_RANK`, which is the convention `torchrun` sets and the one
        `src/cli/generate_v41.py:setup_distributed` reads. Outside `torchrun` the world is one, no
        process group is touched and this returns the control column.

        Two side effects, both of them the launcher's usual work rather than a model's: the card
        this rank drives is selected with `torch.cuda.set_device`, and a world over one initialises
        NCCL. The timeout is two hours rather than the ten-minute default because rank 0 may be
        paying the resident bank's one-time fill -- twelve minutes of `/dev/shm` write and twenty
        seconds of registration -- while the rest wait at the first collective.
        """
        world = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        if world <= 1:
            return cls()
        import torch.distributed as dist

        if not torch.cuda.is_available():
            raise RuntimeError(f"a world of {world} needs the cards, and this host has none")
        count = torch.cuda.device_count()
        if local_rank >= count:
            raise RuntimeError(
                f"rank {rank} was told to drive card {local_rank} of {count} on this host"
            )
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group("nccl", timeout=timedelta(hours=timeout_hours))
        # `device` is resolved *after* the group exists so that the caller's own card and the
        # group's rank agree; a caller that named a card explicitly keeps it.
        if device is None:
            device = torch.device("cuda", local_rank)
        return cls(
            world=world,
            rank=rank,
            reduce=make_all_reduce(world),
            gather=make_all_gather(attention_shards(world)),
            device=device,
        )
