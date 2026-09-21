"""The V4.1 dense tree cut across cards, and the three collectives a layer needs.

The tree is the half of a decode step the cards do not touch: 16.79 GiB of bf16 dense weights,
measured at 620 ms of a 1150 ms step. `/tmp/probe_tp4_block.py` runs one whole block four ways and
finds the split worth 1.21x on the heaviest layer (16.56 ms on the host at 44 threads, 13.70 ms at
TP4) -- which is the platform for a per-layer CUDA graph rather than a win on its own, since the
step it is a fifth of is Python-dispatch-bound and 89% idle. This module is the split itself: what a
rank owns, what it replicates, and where the three all-reduces go.

**Every boundary is a ratio of V4.1's own constants, and two of them are deliberately not cut.**

* `wq_b` splits by head -- 64 heads, 16 a rank -- which is also what cuts the attention.
* `wo_a` is block-diagonal over `o_groups`, 8 groups of 8 heads, so a rank holding 16 contiguous
  heads holds exactly groups `2r` and `2r+1` whole, and `wo_a` needs **no** collective.
* `wo_b` is row-parallel: one `all_reduce` of `[1, 5120]` per layer.
* the shared expert splits the same way -- `w1`/`w3` by intermediate, `w2` row-parallel -- and it is
  the one half the ffn's all-reduce completes *when the routed store is whole*: with no rank dealt
  out, every process holds the same 384 experts and computes the same routed sum, so reducing
  `routed + shared` would multiply that sum by the world. `DeviceRoutedExperts` given a rank deals
  the experts out, and then the routed partial joins the shared one on **one** message;
  `RoutedExperts.partial` is how `MoE.forward` tells the two cases apart, and the two lines change
  together.
* the indexer splits its 32 heads 8 ways. Its score sums *over heads*, so a sharded indexer's score
  is a partial and needs its own `all_reduce` of `[1, 1, t]` before the top-k -- and a flip there is
  discrete, which is why the parity check compares the selected blocks and not only the output.
* **`wq_a` stays whole**, because `q_norm` normalizes across all 1280 LoRA dims: a column-parallel
  `wq_a` would put an all-reduce in *front* of every attention instead of behind it.
* **`wkv` stays whole**, because this is MQA with one KV head: slicing its 512 dims makes every
  attention score a partial sum that no longer has a softmax to belong to.

Three collectives a layer, then, and at the sizes a layer actually sends: fp32 in and fp32 out, 20
KiB for the two block outputs. Summing bf16 partials in bf16 would cost about a bit per partial for
no measurable saving on a message this size, and would put a rounding difference into the parity
check that is not the sharding's.

**The split is by construction, not by post-hoc slicing.** Each module derives its own local widths
from a `world`, so a rank allocates a quarter of the tree and never holds the whole thing -- which
matters, because the full tree plus its own quarter does not fit: 20.6 GiB of unsharded dense
weights against a 22 GiB card. It also keeps the loader's one invariant -- *the name in the tree is
the name in the file* -- from being quietly broken: `checkpoint_weights` matches a parameter to the
file's tensor by name and then refuses a shape mismatch, so a sharded tree needs the file's tensor
cut the same way the constructor cut the parameter. `ShardPlan.local_value` below is that cut, and
it is the same arithmetic the constructors in `attention.py`/`modules.py` were written with.

That duplication is the one thing worth watching here, because it cannot be removed: a constructor
divides `n_heads` by `world` while the loader has to divide `n_heads * head_dim` by four and land on
a row boundary that is a whole number of heads. `test_models_deepseek_v4_1_tp.py` closes the loop by
building the reference tree both ways -- sharded by construction, and one card sliced post hoc -- and
asserting every parameter is identical, so a change to one half fails against the other.

`world=1` is not a special case anywhere: no module divides by anything but one, `attach_tp` attaches
nothing, and every forward is the host forward exactly. That is the control column, and it is free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import torch

__all__ = ["REDUCE_BITS_ENV", "ShardPlan", "attach_tp", "indexer_row_split", "make_all_gather",
           "make_all_reduce", "reduce_bits", "split_names"]


# The indexer's `wq_b` is the one parameter whose cut depends on *how* the indexer is parallel, not
# just on the world: see `indexer_row_split`.
INDEXER_ROW_SPLIT_ENV = "DEEPSEEK_V41_INDEXER_ROW_SPLIT"

# The wire dtype of an activation collective. `32` is the shipped fp32; `16` sends the same sum in
# half the bytes; `0` sends nothing at all. **`0` is a diagnostic and not a configuration** -- it
# returns each rank's partial unscaled, so the answer is wrong and only the timing is worth reading,
# the same way `INDEXER_CAND_SKIP_TEST`'s test branch is. See `make_all_reduce`.
REDUCE_BITS_ENV = "DEEPSEEK_V41_REDUCE_BITS"


def reduce_bits() -> int:
    """`REDUCE_BITS_ENV` as one of 32, 16 or 0; anything else is 32, the shipped wire.

    Read per collective rather than once in `make_all_reduce`, for `INDEXER_CAND_SKIP_TEST`'s reason:
    a probe can then switch arms inside one run and the second arm pays for a `getenv` instead of for
    a re-load. A capture bakes whichever value was in force when the graph was recorded -- the wire
    dtype is part of the recorded body -- so a decode graph and a prefill step can disagree across a
    change of this variable.
    """
    raw = os.environ.get(REDUCE_BITS_ENV, "32").strip()
    return int(raw) if raw in ("0", "16") else 32


def indexer_row_split(world: int) -> bool:
    """Whether the indexer shards query rows instead of index heads.

    The indexer's score is a sum over the 32 index heads, so a head split leaves each rank holding a
    *partial* score and the top-k over it has to wait for a collective. That collective is the whole
    of the indexer's context term -- it is issued once per key tile and its payload grows with the
    compressed width -- so it, and not the arithmetic, is what makes the indexer the expensive row of
    a long prefill.

    Sharding the query rows instead costs the same arithmetic: folding the query and head axes
    together, a rank computing `seqlen/world x 32 heads` is the same GEMM against the same `index_k`
    as one computing `seqlen x 8 heads`. What changes is that the head sum is now over all 32 heads,
    so the score is *complete for the rank's own rows* and there is nothing to reduce. The picks then
    have to come back together before `sparse_attn`, which is head-parallel and needs every query's
    ids -- but that gather is one message an index source over `index_topk` ints a query, against a
    fp32 score tile an index source over the whole compressed prefix.

    The price is that `wq_b` cannot be cut: a rank needs all 32 heads of its own rows even though it
    owns a quarter of the rows. It is the only parameter that changes -- `wq_a` is not in the table
    at all, so `qr` is already full-width everywhere, and `weights_proj` is already computed whole
    and sliced on the way out. The `index_heads` scale in `Indexer.forward` stays the *global* count
    for the same reason it is global today: the sum runs over all 32 heads on every rank.

    Only a prefill whose length divides the world takes this path. The gather is a fixed-shape
    `all_gather`, so a ragged band would need a second code path for no case this model has, and a
    decode step -- one query -- is below the world for every sensible one.
    """
    if world <= 1:
        return False
    return os.environ.get(INDEXER_ROW_SPLIT_ENV, "0").strip() not in ("", "0")


def make_all_reduce(world: int):
    """`dist.all_reduce` as a closure, injected rather than imported.

    Same shape as `src/models/qwen4_exp/runtime.py:make_all_reduce`, for the same reason: nothing in
    the model layer imports `torch.distributed`, so a single-card run pays nothing for the collective
    machinery existing, and a caller that has already built a process group decides how a partial is
    summed rather than inheriting a decision. `world=1` returns `None`, which is what makes the
    unsharded configuration the same code path as the control.

    fp32 on the wire, the caller's dtype on either side of it: the closure upcasts to fp32, sums in
    fp32 and casts the answer back to `tensor.dtype` before returning. On the indexer's level-one
    collective both sides are bf16, because the partial is an `einsum` of the bf16 query tile against
    the bf16 key cache times a bf16 `weights` -- so that message travels at 33.6 MB where the tensor
    it carries is 16.8 MB.

    That upcast is a measured price and not an oversight. A `[2048, 4096]` level-one tile is 5469.8
    us at fp32 against 2877.5 us at bf16 on this fabric -- 32 MiB of wire at 6.1 GB/s against 16 MiB
    at 5.8 GB/s -- which over a chunk's tile counts is 0.469 s at 32768 and 2.001 s at 262144 of
    fp32 wire against 0.267 and 1.073 in half the bytes. Halving it is closed on the picked ids
    rather than on size: NCCL sums in the wire dtype, the score's O(600) values carry 8 mantissa bits
    there, and the fp16 arm of that experiment moves the selection on all eight indexer layers against
    a baseline that reproduces to the digit. If the dtype moves at all it should move to fp16 -- the
    same 2 bytes with 10 mantissa bits -- which is a different function and so its own parity run.
    Measurements and both gates: `docs/performance/deepseek_v4_1_flash_chunked_prefill.md`, "Two
    levers, and what gates each".

    **The two messages at the tail of a layer are not that tensor, and they are the ones that pay.**
    `Attention.forward`'s `wo_b` and `MoE.forward`'s join each carry one `[1, 4096, 5120]` fp32 sum,
    which is 80 MiB, and a 4096-token chunk sends 80 of them -- 6.4 GiB of wire, where the indexer's
    1066 collectives together send 1.4. `REDUCE_BITS_ENV` is what separates the two: a caller that
    passes `discrete=True` -- the two score sites, whose result a top-k reads and a rounding in which
    is a different selection -- keeps the fp32 path above whatever the variable says, and every other
    call site follows it. `reduce_bits`'s docstring names the third setting, which is a control column
    rather than a configuration.

    `/tmp/probe_allreduce_rate.py` prices those messages alone on this fabric, and the curve is flat
    at ~5.7 GiB/s from 1 MiB to 80 MiB, so on this wire a byte is a byte: 80 MiB fp32 13.582 ms, the
    same `4096x5120` in bf16 7.080 -- **exactly half** -- 32 MiB fp32 5.460 ms, 1 MiB fp32 0.193 ms,
    and 11.6 us of event pair with nothing between them. Over one 4096-token chunk that is 1.087 s
    for the 80 activation messages, 0.245 s for the indexer's 48 prefix tiles and 0.198 s for its
    1024 candidate tiles: **1.53 s in all**, against 5.96 s of
    `ncclDevKernel_AllReduce_Sum_f32_RING_LL` in the chunk's own profile
    (`/tmp/chunk_nccl_attr.log`, 1152 calls, 26.04 s of card timeline). So the fabric's own price is a
    quarter of what the row costs in situ, and that gap is the question `REDUCE_BITS_ENV=16` answers:
    if the row follows the bytes it is wire and the halving is worth up to 2.1 s of a 26.0 s chunk,
    and if it does not then the other 4.4 s is the ring waiting for the last rank to arrive and no
    message in the model is worth shrinking.
    """
    if world <= 1:
        return None

    def reduce(tensor: torch.Tensor, *, discrete: bool = False) -> torch.Tensor:
        """Sum `tensor` across the ranks, in the wire dtype `reduce_bits` names.

        `discrete` marks the message whose result a top-k selects on -- the indexer's two score sites
        -- and pins it to the fp32 path whatever the variable says.
        """
        import torch.distributed as dist

        bits = 32 if discrete else reduce_bits()
        if bits == 0:  # the control column: no message, so a wrong answer and a true floor
            return tensor
        if bits == 16:
            narrow = tensor.to(torch.float16).contiguous()
            dist.all_reduce(narrow, op=dist.ReduceOp.SUM)
            return narrow.to(tensor.dtype)
        wide = tensor.float().contiguous()
        dist.all_reduce(wide, op=dist.ReduceOp.SUM)
        return wide.to(tensor.dtype)

    return reduce


def make_all_gather(world: int):
    """`dist.all_gather` along the sequence axis as a closure, injected for `make_all_reduce`'s reason.

    Only the indexer's row split uses it. The split leaves each rank with the picked positions for
    its own query rows, and `sparse_attn` is head-parallel: it needs every query's ids, so the rows
    have to come back together before the layer reads them. `cat` in rank order reassembles exactly
    the row order the bands were cut in, which is what makes the gathered tensor the one a
    head-split rank would have produced.

    `world=1` returns `None`, so the unsharded configuration stays the same code path as the control.
    """
    if world <= 1:
        return None

    def gather(tensor: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        parts = [torch.empty_like(tensor) for _ in range(world)]
        dist.all_gather(parts, tensor.contiguous())
        return torch.cat(parts, dim=1)

    return gather


# Which parameters are cut, and along which axis. The key is the parameter name *below* its layer --
# `layers.7.attn.wq_b.weight` is matched on `attn.wq_b.weight` -- because the same block is built 40
# times and the cut does not depend on which one it is. Exact equality, not a suffix test: `attn.wq_b`
# and `attn.indexer.wq_b` are both `wq_b.weight` at the end and two different tensors.
#
#   rows         -- axis 0, in units of `head_dim` heads or a whole intermediate width
#   cols         -- axis 1, the row-parallel half of an output projection
#   grouped_rows -- axis 0 of the `[groups, o_lora_rank, -1]` view of a flat matrix
#   index_heads  -- axis 0 of the `[n_heads, index_head_dim, -1]` view, same reason as wo_a
#
# Every entry is cut on every sharded load. The one exception is the indexer's `wq_b`, which
# replicates under `indexer_row_split` -- the table names the axis it is cut on by default, and
# `ShardPlan._kind` is what makes the exception, in one place rather than two.
_SPLITS: tuple[tuple[str, str], ...] = (
    ("attn.wq_b.weight", "rows"),
    ("attn.attn_sink", "rows"),
    ("attn.wo_a.weight", "grouped_rows"),
    ("attn.wo_b.weight", "cols"),
    ("attn.indexer.wq_b.weight", "index_heads"),
    ("ffn.shared_experts.w1.weight", "rows"),
    ("ffn.shared_experts.w3.weight", "rows"),
    ("ffn.shared_experts.w2.weight", "cols"),
)


def split_names() -> tuple[str, ...]:
    """The parameter tails this module cuts. Exported so a test can assert it saw every one.

    The table, not the cut: `attn.indexer.wq_b.weight` is listed here and is not cut when
    `indexer_row_split` is on, because that configuration shards the indexer by query rows and needs
    all of the heads on every rank. `ShardPlan._kind` is the one place the two are reconciled.
    """
    return tuple(name for name, _ in _SPLITS)


def _tail(name: str) -> str:
    """`layers.7.attn.wq_b.weight` -> `attn.wq_b.weight`; leaves an already-bare name alone."""
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] in ("layers", "blocks") and parts[1].isdigit():
        return ".".join(parts[2:])
    return name


@dataclass(frozen=True)
class ShardPlan:
    """What rank `rank` of `world` owns, and how its partials get summed.

    The head/group counts here are the *global* ones; `heads` is what one rank owns. Both are needed
    and the distinction is not cosmetic: the indexer's score is scaled by
    `softmax_scale * n_heads**-0.5` over all 32 heads, so a sharded indexer that used its own 8 would
    scale its partial by `2x` -- every one of the eight index-source layers, silently.

    Built by `build`, which is also where the split is refused if it does not come out whole.
    """

    rank: int
    world: int
    heads: int
    groups: int
    index_heads: int
    heads_global: int
    groups_global: int
    index_heads_global: int
    # Read once, in `build`, and not again. `_kind` is called from `local_shape` and `local_value`,
    # which the loader calls for every parameter long after the plan was made, so a plan that read
    # the switch lazily would answer as a head split or a row split depending on when it was asked --
    # and the one place that disagreement would show up is the cut itself. A plan is a value: the
    # same question gets the same answer.
    row_split: bool
    head_dim: int
    o_lora_rank: int
    index_head_dim: int
    moe_inter_dim: int
    reduce: Callable[[torch.Tensor], torch.Tensor] | None
    gather: Callable[[torch.Tensor], torch.Tensor] | None

    @classmethod
    def build(
        cls,
        cfg: Any,
        rank: int,
        world: int,
        *,
        moe_inter_dim: int | None = None,
        reduce: Callable[[torch.Tensor], torch.Tensor] | None = None,
        gather: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> "ShardPlan":
        """Derive the plan, or raise on a split that is not a partition.

        Every division below has to be exact for the split to be a partition rather than an
        approximation. The one that looks like it needs its own check does not, and it is worth
        writing down why: a rank owns `n_heads / world` *contiguous* heads and `o_groups / world`
        contiguous groups, and the two are the same set only if a whole number of groups fits in a
        rank's slice. Write `hpg = n_heads / o_groups`, which the check below makes an integer, so
        `heads / world = (o_groups / world) * hpg` and a rank's head count is its group count times
        `hpg` -- always a whole number of groups. The `o_groups = 3, world = 2` case that would break
        this is already refused by `groups % world`, so no third check is needed and none is added.
        """
        if world < 1:
            raise ValueError(f"a TP world of {world} is not a split of anything")
        if not 0 <= rank < world:
            raise ValueError(f"rank {rank} is outside a world of {world}")

        heads = int(cfg.n_heads)
        groups = int(cfg.o_groups)
        index_heads = int(cfg.index_n_heads)
        inter = int(moe_inter_dim if moe_inter_dim is not None else cfg.moe_inter_dim)

        for count, what in (
            (heads, "attention heads"),
            (groups, "o-groups"),
            (index_heads, "indexer heads"),
            (inter, "the shared expert's intermediate width"),
        ):
            if count % world:
                raise ValueError(f"{count} {what} do not split {world} ways")
        if heads % groups:
            raise ValueError(
                f"{heads} attention heads do not divide into {groups} o-groups, so wo_a's blocks "
                "are not a fixed number of heads wide"
            )

        if reduce is None:
            reduce = make_all_reduce(world)
        if gather is None:
            gather = make_all_gather(world)

        return cls(
            rank=rank,
            world=world,
            heads=heads // world,
            groups=groups // world,
            index_heads=index_heads // world,
            heads_global=heads,
            groups_global=groups,
            index_heads_global=index_heads,
            row_split=indexer_row_split(world),
            head_dim=int(cfg.head_dim),
            o_lora_rank=int(cfg.o_lora_rank),
            index_head_dim=int(cfg.index_head_dim),
            moe_inter_dim=inter,
            reduce=reduce,
            gather=gather,
        )

    def describe(self) -> str:
        return (
            f"TP{self.world} rank {self.rank}: {self.heads}/{self.heads_global} heads, "
            f"{self.groups}/{self.groups_global} o-groups, "
            f"{self.index_heads}/{self.index_heads_global} index heads"
        )

    # -- the cut, in the two forms the two callers need -----------------------------------------

    def local_shape(self, name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
        """The shape this rank's parameter has, given the file's. Unsharded names pass through."""
        kind = self._kind(name)
        if kind is None:
            return tuple(shape)
        shape = tuple(shape)
        if kind == "rows":
            return (self._rows_total(name) // self.world,) + shape[1:]
        if kind == "cols":
            return shape[:-1] + (shape[-1] // self.world,)
        if kind == "grouped_rows":
            inner = shape[1]
            return (self.groups * self.o_lora_rank, inner)
        if kind == "index_heads":
            inner = shape[1]
            return (self.index_heads * self.index_head_dim, inner)
        raise AssertionError(kind)

    def local_value(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Cut the file's tensor for `name` down to this rank's parameter.

        A copy, not a view: the caller hands the result to `Parameter.copy_` and the source may be
        an mmap-backed tensor that has to outlive the load. `contiguous` for the same reason -- the
        `wo_a` and indexer `wq_b` cases below take a row band out of a reshaped view of a matrix
        that is itself a stack, and the second row of that view is not contiguous with the first.
        """
        kind = self._kind(name)
        if kind is None:
            return value
        if kind == "rows":
            total = self._rows_total(name)
            width = total // self.world
            return value[self.rank * width : (self.rank + 1) * width].contiguous()
        if kind == "cols":
            width = value.size(-1) // self.world
            return value[..., self.rank * width : (self.rank + 1) * width].contiguous()
        if kind == "grouped_rows":
            # The checkpoint stores wo_a as one flat `[groups * o_lora_rank, heads * head_dim /
            # groups]`, which is really `groups` stacked `[o_lora_rank, -1]` blocks -- one per head
            # group, each projecting only its own heads. Slicing the flat matrix by rows would take
            # an arbitrary `o_lora_rank`-wide band across *all* eight groups instead of two whole
            # ones, which is a different (and wrong) tensor that happens to have the right shape.
            stacked = value.view(self.groups_global, self.o_lora_rank, -1)
            return stacked[self.rank * self.groups : (self.rank + 1) * self.groups].reshape(
                self.groups * self.o_lora_rank, -1
            ).contiguous()
        if kind == "index_heads":
            stacked = value.view(self.index_heads_global, self.index_head_dim, -1)
            return stacked[self.rank * self.index_heads : (self.rank + 1) * self.index_heads].reshape(
                self.index_heads * self.index_head_dim, -1
            ).contiguous()
        raise AssertionError(kind)

    # -- internals ------------------------------------------------------------------------------

    def _kind(self, name: str):
        tail = _tail(name)
        # The one entry whose cut depends on the world's *axis* and not just its size. A row split
        # needs all 32 heads of a rank's rows, so the parameter replicates; see `indexer_row_split`.
        if tail == "attn.indexer.wq_b.weight" and self.row_split:
            return None
        for candidate, kind in _SPLITS:
            if tail == candidate:
                return kind
        return None

    def _rows_total(self, name: str) -> int:
        tail = _tail(name)
        if tail == "attn.attn_sink":
            return self.heads_global
        if tail == "attn.wq_b.weight":
            return self.heads_global * self.head_dim
        if tail in ("ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight"):
            return self.moe_inter_dim
        raise AssertionError(tail)


def attach_tp(model: torch.nn.Module, plan: ShardPlan | None) -> int:
    """Hang `plan` on every module whose forward has a collective in it. Returns how many.

    An attribute rather than a constructor argument, because the collectives are *inside*
    `Attention.forward`, `Indexer.forward` and `MoE.forward`, and threading a fifth argument through
    `Backbone.forward` -> `Block.forward` -> those three would put the plan on the hot path of the
    host configuration too. `getattr(module, "tp", None) is None` is the host path, unmodified.

    The count is returned so a caller can assert it matched what it expected: a plan that reached no
    module would otherwise look exactly like a plan that worked, and the run would be four ranks each
    computing a quarter of the answer and printing it.
    """
    if plan is None or plan.world <= 1:
        return 0
    from src.models.deepseek_v4_1.attention import Attention, Indexer
    from src.models.deepseek_v4_1.modules import MoE

    attached = 0
    for module in model.modules():
        if isinstance(module, (Attention, Indexer, MoE)):
            module.tp = plan
            attached += 1
    # also on the root, where a caller can find it: the plan carries the rank and world a launcher
    # wants to print, and building a second one to ask would give a second all-reduce closure.
    model.tp = plan
    return attached
