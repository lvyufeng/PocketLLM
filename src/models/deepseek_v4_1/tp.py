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

from dataclasses import dataclass
from typing import Any, Callable

import torch

__all__ = ["ShardPlan", "attach_tp", "make_all_reduce", "split_names"]


def make_all_reduce(world: int):
    """`dist.all_reduce` as a closure, injected rather than imported.

    Same shape as `src/models/qwen4_exp/runtime.py:make_all_reduce`, for the same reason: nothing in
    the model layer imports `torch.distributed`, so a single-card run pays nothing for the collective
    machinery existing, and a caller that has already built a process group decides how a partial is
    summed rather than inheriting a decision. `world=1` returns `None`, which is what makes the
    unsharded configuration the same code path as the control.

    fp32 in, fp32 out, `SUM`. Summing four bf16 partials in bf16 would cost about a bit per partial
    for no measurable saving on a 20 KiB message, and would put a rounding difference into the
    parity check that is not the sharding's.
    """
    if world <= 1:
        return None

    def reduce(tensor: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        wide = tensor.float().contiguous()
        dist.all_reduce(wide, op=dist.ReduceOp.SUM)
        return wide.to(tensor.dtype)

    return reduce


# Which parameters are cut, and along which axis. The key is the parameter name *below* its layer --
# `layers.7.attn.wq_b.weight` is matched on `attn.wq_b.weight` -- because the same block is built 40
# times and the cut does not depend on which one it is. Exact equality, not a suffix test: `attn.wq_b`
# and `attn.indexer.wq_b` are both `wq_b.weight` at the end and two different tensors.
#
#   rows         -- axis 0, in units of `head_dim` heads or a whole intermediate width
#   cols         -- axis 1, the row-parallel half of an output projection
#   grouped_rows -- axis 0 of the `[groups, o_lora_rank, -1]` view of a flat matrix
#   index_heads  -- axis 0 of the `[n_heads, index_head_dim, -1]` view, same reason as wo_a
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
    """The parameter tails this module cuts. Exported so a test can assert it saw every one."""
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
    head_dim: int
    o_lora_rank: int
    index_head_dim: int
    moe_inter_dim: int
    reduce: Callable[[torch.Tensor], torch.Tensor] | None

    @classmethod
    def build(
        cls,
        cfg: Any,
        rank: int,
        world: int,
        *,
        moe_inter_dim: int | None = None,
        reduce: Callable[[torch.Tensor], torch.Tensor] | None = None,
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

        return cls(
            rank=rank,
            world=world,
            heads=heads // world,
            groups=groups // world,
            index_heads=index_heads // world,
            heads_global=heads,
            groups_global=groups,
            index_heads_global=index_heads,
            head_dim=int(cfg.head_dim),
            o_lora_rank=int(cfg.o_lora_rank),
            index_head_dim=int(cfg.index_head_dim),
            moe_inter_dim=inter,
            reduce=reduce,
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

    @staticmethod
    def _kind(name: str):
        tail = _tail(name)
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
