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
  default is a decode default that the prefill stage will have to override per path.

**The empty rank is why the deal is not a detail.** Under `id` with `top_k` 8 and a world of 4, a
rank owns nothing `(3/4)^8` of the time -- 10% of the draws -- and it still has to arrive at the
collective with a zero, because the collective is unconditional. Under `sorted` the case cannot
happen. Neither is a corner to be excused: the first is one draw in ten and the second is most of
the bytes a chunk moves.

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
    "DEALS",
    "DEAL_ENV",
    "EpGroup",
    "deal_card",
    "deal_rule",
    "make_all_reduce",
    "owned_positions",
    "rows_per_card",
]

#: How a draw's experts are shared out. See the module docstring for what each costs.
DEAL_ENV = "POCKETLLM_MIMO_EXPERT_DEAL"
DEALS = ("sorted", "id")


def deal_rule() -> str:
    """Which deal a module makes unless its constructor is told. `sorted` by default.

    The default is the *decode* deal, because decode is the only path that exists: it is 21% faster
    than `id` on four cards and the reason is in the module docstring. `id` is the prefill deal, and
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
        return cls(world=world, rank=rank, reduce=make_all_reduce(world), device=device)
