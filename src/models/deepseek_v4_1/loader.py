"""Feed the released DeepSeek-V4.1-Flash shards into `modules.Backbone`.

`modules.py` builds the tree over `torch.empty` and `src/loader/safetensors.py` maps the 48 shards;
this is the one place that knows which name in the file is which parameter in the tree, and the one
place that knows which of them are quantized. `checkpoint_weights` is the general half -- it walks a
module's own parameter names and copies the same-named tensor in -- and the two `Checkpoint*` classes
are the specific half, for the two things the checkpoint holds in a shape the tree does not.

Four properties of the release decide the shape of this file. All four are measured from the shards
rather than read off the reference, because the reference's own loader runs against a *converted*
checkpoint (`inference/convert.py` rewrites the fp4 experts and splits the model across ranks) and so
never has to decide any of them:

* **Quantization is not a config field, so it is detected per tensor.** A weight is quantized exactly
  when a tensor with the same stem and a `scale` leaf sits beside it: 162.0M elements of dense
  projection per layer arrive fp8 -- 126.6M of that attention, the rest the shared experts -- the 384
  experts arrive as packed fp4, and the norms, the gates, the hyper-connection coefficients and the
  embedding arrive plain. `cfg.dtype` says `'fp8'` and `cfg.expert_dtype` says `'fp4'`, but no layer
  stores its `compressor.wkv` fp8 -- all four KV sources store it bf16, and the tree then holds layer
  20's bf16 and the other three fp32, because `Compressor` builds that one in the parameter dtype
  only when `compress_ratio` is 1 -- so the config's dtype is a statement about the model and not a
  lookup key.
* **The scale beside a weight comes in two layouts, and which one applies follows from the dtype.**
  A dense fp8 weight's scale is a two-dimensional `[out // 32, in // 32]` grid -- `wq_a`'s is
  `(40, 160)` for a `(1280, 5120)` weight -- while a packed fp4 expert's and an Engram table's keep a
  row per output row along K only (`(2304, 160)` beside `(2304, 2560)`, and `(rows, 8)` beside
  `(rows, 256)`). Both put the block structure along K, so `block_size` is derived the same way from
  either and holds for the release's 32; the axis a scale is *expanded* along does not, and
  `kernels.dequant_fp8_weight` versus `dequantize_rows` is that difference.
* **`compressor.wgate` exists on three of the four KV source layers, not four.** `compress_ratios[20]`
  is 1, and `Compressor.__init__` returns before creating `wgate` in that case, so layer 20 has
  `compressor.norm` and `compressor.wkv` and no `wgate`. Nothing in the loader has to special-case it
  -- the name simply is not there and the tree does not ask for it -- but it is the reason a naive
  "40 layers times 4 KV sources" key count comes up one short. The same `ratio <= 1` branch is why the
  tree keeps layer 20's `compressor.wkv` bf16 while layers 2, 8 and 14 hold fp32 there: 176 of the 924
  parameters are bf16 in the file and fp32 in the tree -- the four norms on every layer, the
  compressor's norm, `wgate` and `wkv` on the ratio-2 KV sources, the indexer's `k_norm`, the root
  `norm` and `head` -- and `checkpoint_weights` casts each into its parameter's own dtype rather than
  special-casing any of them.
* **Neither Engram table belongs anywhere but host RAM, and gathering from the shards is not an
  option either.** They are 91.55 and 91.56 GiB of rows, and a forward touches one row per hash
  column -- 24 per position, per table -- so `CheckpointEngramTable` either copies the table into RAM
  (189.1 GiB of codes and scales for both) or gathers each row out of the mapping. The gather is what
  decides it, and it is a page-cache question rather than a bandwidth one: measured on a quiet disk,
  a row whose page is not yet resident costs **21 to 49 ms** out of the mapping -- one seek per row on
  this shingled drive, the low end when a batch of them queues up behind each other -- and the same
  row costs **0.004 ms** once it is. A 512-token prefill gathers 12,288 rows per table, which measured
  **253 s** from a cold cache, where the same rows out of the copy take **9 ms**. The copy is one
  sequential pass -- **373 s** per table here, 271 MiB/s, and 747 s for both -- and it is paid once.
  Both paths gather through a `uint8` alias because the CPU has no
  float8 `index_select`: `view[rows]` raises `NotImplementedError: "index_cpu" not implemented for
  'Float8_e4m3fn'` and the E8M0 scales raise it too.
* **The three DSpark draft layers, the vision tower and the aligner are simply not asked for.**
  `Backbone` is the text backbone, so 94,831 of the checkpoint's 96,085 tensors are left where they
  are, and 92,160 of those are the expert projections. `LoadReport.unloaded_groups` counts 95,161
  rather than 94,831 because it tallies a quantized name as the one tensor it is named by and not as
  the two it reads; either way it counts them rather than passing over them in silence.

`checkpoint_weights` raises on a parameter it could not fill. That is deliberate: every module in the
tree is built out of `torch.empty`, so a parameter the checkpoint does not name stays uninitialized
memory, and uninitialized memory that happens to be finite is not detectable at the point of use.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Sequence

import torch

from src.encoding.engram import EngramLayout, NgramHasher, build_compressed_token_map
from src.loader.safetensors import SAFETENSORS_DTYPES, MmapSafetensors
from src.models.deepseek_v4_1.config import V41TextConfig
from src.models.deepseek_v4_1.kernels import dequant_fp4_weight, dequant_fp8_weight
from src.models.deepseek_v4_1.modules import (
    LINEAR_DTYPE,
    Backbone,
    EngramTable,
    RoutedExperts,
    _moe_shape,
    dequantize_rows,
    expert_forward,
)

__all__ = [
    "CheckpointEngramTable",
    "CheckpointRoutedExperts",
    "EngramHashIds",
    "LoadReport",
    "LoadedBackbone",
    "V41Checkpoint",
    "build_hasher",
    "checkpoint_weights",
    "load_backbone",
    "scale_key",
]

# The two dtype strings a quantized weight arrives in, both paired with an E8M0
# scale tensor. `F8_E4M3` is one code per byte; `I8` is two fp4 codes per byte,
# packed along the input axis, which is why the two expanders are not
# interchangeable even though the scale names are spelled the same.
FP8_WEIGHT = "F8_E4M3"
FP4_PACKED_WEIGHT = "I8"

# Dequantized experts one layer keeps on the host. The released layer is 384
# experts of 16.9 MiB of packed fp4 each, 17.9 MiB with their scales, and 67.5 MiB
# each once expanded to bf16 -- 25.3 GiB for the whole layer -- and the correctness
# path re-reads and re-expands on a miss. 16 experts is 1.05 GiB per layer and 42
# GiB across the backbone, which bounds the cache without pretending to be the
# serving design: a device-side expert cache fed by fp4 kernels is what a measured
# run needs, and it replaces this, not tunes it.
DEFAULT_EXPERT_CACHE = 16

_EXPERT_PROJECTIONS = ("w1", "w2", "w3")


def scale_key(weight_key: str) -> str:
    """The scale tensor paired with a quantized weight: `...wq_a.weight` -> `...wq_a.scale`."""
    stem, dot, leaf = weight_key.rpartition(".")
    if not dot or leaf != "weight":
        raise ValueError(f"{weight_key!r} is not a weight name, so it has no scale beside it")
    return f"{stem}.scale"


def _group_of(name: str) -> str:
    """The unit a load report and a progress line count in: one layer, or one root tensor."""
    parts = name.split(".")
    if parts[0] == "layers" and len(parts) > 1:
        return f"layers.{parts[1]}"
    return parts[0]


class V41Checkpoint:
    """A read-only view of a released DeepSeek-V4.1-Flash directory.

    Wraps `MmapSafetensors`, which does the mapping, and adds the three things this checkpoint needs
    on top of it: the `weight`/`scale` pairing, the fp8-versus-fp4 decision, and row gathers for the
    two Engram tables. Nothing here copies a tensor into memory except `weight` and `rows`; until one
    of those is called the shards are 476 GiB of address space and no I/O.
    """

    def __init__(self, root: str, *, device: torch.device | str | None = None) -> None:
        self.root = root
        self.device = device
        self.reader = MmapSafetensors(root)
        self._blocks: dict[str, int] = {}

    def __contains__(self, key: str) -> bool:
        return key in self.reader

    def __len__(self) -> int:
        return len(self.reader)

    def keys(self) -> Iterable[str]:
        return self.reader.keys()

    def close(self) -> None:
        self.reader.close()

    # -- what a tensor is -----------------------------------------------------

    def stored_dtype(self, key: str) -> torch.dtype:
        """The dtype `key` is stored in, before any expansion."""
        return SAFETENSORS_DTYPES[self.reader.entry(key).dtype][1]

    def nbytes(self, key: str) -> int:
        return self.reader.entry(key).nbytes

    def is_quantized(self, key: str) -> bool:
        """Quantized means: this checkpoint stores a `scale` beside it. Nothing else has one."""
        try:
            sibling = scale_key(key)
        except ValueError:
            return False
        return sibling in self.reader

    def is_packed_fp4(self, key: str) -> bool:
        return self.reader.entry(key).dtype == FP4_PACKED_WEIGHT

    def block_size(self, key: str) -> int:
        """The scale block's width along the input axis, derived rather than assumed.

        The two expanders take a `block_size` whose defaults are not this checkpoint's: the shared
        fp8 dequantizer defaults to 128 and the fp4 one to 32, and this release uses a plain 32x32
        grid everywhere. Passing the derived value means a checkpoint laid out differently fails
        here with the two widths in the message instead of expanding to plausible wrong numbers.
        """
        cached = self._blocks.get(key)
        if cached is not None:
            return cached
        entry = self.reader.entry(key)
        scale = self.reader.entry(scale_key(key))
        if len(entry.shape) != 2 or len(scale.shape) != 2:
            raise ValueError(f"{key} is not a matrix, so it has no block grid")
        # A packed fp4 weight holds two codes per byte along the input axis, so the
        # axis its scale divides is twice the stored width.
        width = entry.shape[-1] * (2 if entry.dtype == FP4_PACKED_WEIGHT else 1)
        if scale.shape[-1] == 0 or width % scale.shape[-1]:
            raise ValueError(
                f"{key} is {entry.shape} with a {scale.shape} scale, which is not a whole "
                f"number of blocks over a width of {width}"
            )
        self._blocks[key] = width // scale.shape[-1]
        return self._blocks[key]

    # -- reading --------------------------------------------------------------

    def weight(self, key: str, *, dtype: torch.dtype | None = None, device=None) -> torch.Tensor:
        """One weight, expanded if it is quantized, in `dtype` and on `device`."""
        value = self.dequantize(key) if self.is_quantized(key) else self.reader.load(key)
        if dtype is not None and value.dtype != dtype:
            value = value.to(dtype)
        device = self.device if device is None else device
        if device is not None:
            value = value.to(device)
        return value

    def dequantize(self, key: str) -> torch.Tensor:
        """Expand one quantized weight to fp32.

        fp8 and packed fp4 both arrive as a byte tensor with E8M0 scales on a square block grid, and
        they expand differently: `dequant_fp8_weight` reads one code per byte and
        `dequant_fp4_weight` unpacks two. The checkpoint stores both beside a `.scale`, so which one
        applies is decided by the weight's own stored dtype and not by its name.
        """
        weight, scale = self.reader.view(key), self.reader.view(scale_key(key))
        block = self.block_size(key)
        if self.is_packed_fp4(key):
            return dequant_fp4_weight(weight, scale, block)
        return dequant_fp8_weight(weight, scale, block)

    def rows(self, key: str, rows: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Gather scattered rows out of a mapped matrix, without reading the rest of it.

        The gather goes through a `uint8` alias. The CPU has no float8 `index_select`, so
        `view[rows]` raises `NotImplementedError: "index_cpu" not implemented for 'Float8_e4m3fn'`
        -- and the same for `'Float8_e8m0fnu'` on the scales -- while the byte alias gathers fine
        and is reinterpreted afterwards. Duplicate rows are copies rather than views, which is what
        a repeated hash id needs.
        """
        entry = self.reader.entry(key)
        if len(entry.shape) != 2:
            raise ValueError(f"{key} is {len(entry.shape)}-dimensional; only rows of a matrix gather")
        flat = rows.reshape(-1).to(device="cpu", dtype=torch.int64)
        if flat.numel():
            low, high = int(flat.min()), int(flat.max())
            if low < 0 or high >= entry.shape[0]:
                raise IndexError(f"{key} has {entry.shape[0]} rows, so {low}..{high} is out of range")
        gathered = self.reader.entry_view(entry).view(torch.uint8)[flat]
        gathered = gathered.view(SAFETENSORS_DTYPES[entry.dtype][1])
        if dtype is not None:
            gathered = gathered.to(dtype)
        return gathered.reshape(*rows.shape, *entry.shape[1:])


@dataclass
class LoadReport:
    """What a load moved and what it did not.

    `missing` is the field that matters. Every module in the tree is built out of `torch.empty`, so
    a parameter the checkpoint does not name is uninitialized memory, and a NaN from one shows up
    forty layers later as a flat logit distribution rather than as an error here.

    `unloaded_groups` is the complement and is a count rather than a verdict: `Backbone` is the text
    backbone, so the vision tower, the aligner and the three DSpark draft layers are genuinely not
    wanted, and counting them is how the difference between "left behind on purpose" and "forgotten"
    stays visible.
    """

    loaded: list[str] = field(default_factory=list)
    quantized: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unloaded_groups: dict[str, int] = field(default_factory=dict)
    bytes_read: int = 0

    @property
    def ok(self) -> bool:
        return not self.missing

    def summary(self) -> str:
        parts = [f"{len(self.loaded)} tensors ({len(self.quantized)} quantized), {self.bytes_read / 2**30:.2f} GiB read"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        if self.unloaded_groups:
            left = ", ".join(f"{group} {count}" for group, count in sorted(self.unloaded_groups.items()))
            parts.append(f"not asked for: {left}")
        # Named in full rather than counted, because the whole point of the field is that a
        # parameter nothing will ever write to is not something to be summarized away.
        parts.append(f"MISSING {self.missing}" if self.missing else "no parameter left unfilled")
        return "; ".join(parts)


def checkpoint_weights(
    module: torch.nn.Module,
    checkpoint: V41Checkpoint,
    *,
    prefix: str = "",
    skip: Callable[[str], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> LoadReport:
    """Copy every parameter `module` names out of the checkpoint, expanding what is quantized.

    The name in the tree *is* the name in the file, which is why this can be a walk rather than a
    table: `tests/test_models_deepseek_v4_1_reference_parity.py` proves the convention on the
    reference's own `state_dict`, and the one structural difference -- the checkpoint's
    `ffn.experts.{j}.w{k}.weight` against the tree's stacked `ffn.routed.w{k}` -- is not a parameter
    here at all, because `MoE` holds a `RoutedExperts` store instead of a bank when one is supplied.

    `skip` exists for a caller that built the resident bank or a truncated table, whose storage is
    not named after anything in the file. A skipped name is not a missing one; only `missing` means
    the tree asked for something the checkpoint does not have.

    Raises only through the caller's own check on `missing`. Filling is to the parameter's own dtype
    and device, so `Head`, which is fp32 where the file is bf16, keeps full precision in the logits.
    """
    report = LoadReport()
    seen_group: str | None = None
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            group = _group_of(name)
            if progress is not None and group != seen_group:
                progress(f"{group}  ({len(report.loaded)} filled)")
                seen_group = group
            key = f"{prefix}{name}"
            if skip is not None and skip(key):
                report.skipped.append(key)
                continue
            if key not in checkpoint:
                report.missing.append(key)
                continue
            report.bytes_read += checkpoint.nbytes(key)
            if checkpoint.is_quantized(key):
                report.quantized.append(key)
            value = checkpoint.weight(key, dtype=parameter.dtype)
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"{key} is {tuple(value.shape)} in the checkpoint and {tuple(parameter.shape)} "
                    f"in the tree at {name}"
                )
            parameter.copy_(value)
            report.loaded.append(key)
    return report


class CheckpointRoutedExperts(RoutedExperts):
    """One layer's 384 routed experts, left in the shards.

    Not an `nn.Module`, deliberately: a module would put its weights in `state_dict()` and in
    `parameters()`, and the point of this class is that the layer's experts are *not* part of the
    tree. `MoE` holds one of these where `ResidentRoutedExperts` would hold a bank, so the backbone
    has no `ffn.routed.*` tensor to fill and the 476 GiB never becomes an allocation.

    A miss reads the expert's three matrices out of the mapping and expands them to bf16, which is
    the same width the resident bank holds and the width `expert_forward` consumes. bf16 and not
    fp32: the released fp4 is four bits of mantissa, so bf16's eight are already more than the
    expansion can recover, and the cache below is what makes the cost per miss bearable.

    It barely does. A token routes to 8 of the layer's 384 experts, so a 16-expert window returns
    about 2 of them: measured, 6 misses per layer per token, warm. That is 25% saved for the 42 GiB
    the window costs across the backbone, and it puts the whole model's token time on the expansion
    rather than on the disk -- one miss is 0.3% mapping read and 99.7% the arithmetic that turns fp4
    codes into fp32 and then bf16, at 0.122 s per expert, so a step that misses 240 of them costs
    about 29 s against 1.0 s for a forward whose experts are already expanded and 27.2 s for a first,
    entirely cold one. 240 fresh
    experts is 4.2 GiB read and 15.8 GiB expanded per token, on a host whose RAM holds all 269 GiB of
    them in their packed form and hands them over for free; what the host cannot do cheaply is turn
    them into numbers. A device-side cache is what replaces it, not a larger `cache_size`.
    """

    def __init__(
        self,
        checkpoint: V41Checkpoint,
        layer_id: int,
        *,
        n_experts: int,
        dim: int,
        inter_dim: int,
        swiglu_limit: float = 0.0,
        cache_size: int = DEFAULT_EXPERT_CACHE,
    ) -> None:
        self.checkpoint = checkpoint
        self.layer_id = layer_id
        self.n_experts = n_experts
        self.dim = dim
        self.inter_dim = inter_dim
        self.swiglu_limit = swiglu_limit
        self.cache_size = cache_size
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._order: list[int] = []
        self.misses = 0

    def _key(self, expert: int, which: str) -> str:
        return f"layers.{self.layer_id}.ffn.experts.{expert}.{which}.weight"

    def expert(self, expert: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(w1, w2, w3) of one expert as `[inter, dim]`, `[dim, inter]`, `[inter, dim]` bf16.

        First-in-first-out eviction rather than least-recently-used: what a step touches is the
        experts its own token routed to, so recency within a layer is not informative, and the
        access pattern a real run has -- a different handful of experts every step -- makes a
        simple queue as good as anything more elaborate.
        """
        cached = self._cache.get(expert)
        if cached is not None:
            return cached
        self.misses += 1
        weights = tuple(
            self.checkpoint.weight(self._key(expert, which), dtype=LINEAR_DTYPE)
            for which in _EXPERT_PROJECTIONS
        )
        if self.cache_size > 0:
            if len(self._order) >= self.cache_size:
                self._cache.pop(self._order.pop(0), None)
            self._cache[expert] = weights
            self._order.append(expert)
        return weights

    def forward(self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """x: [n, dim] bf16, weights/indices: [n, topk]. Returns [n, dim] fp32."""
        y = torch.zeros_like(x, dtype=torch.float32)
        # Walked in expert id order, like `ResidentRoutedExperts` and like the reference: a token's
        # contributions land in the same order either way, so the two agree bit for bit.
        counts = torch.bincount(indices.flatten(), minlength=self.n_experts).tolist()
        for expert in range(self.n_experts):
            if counts[expert] == 0:
                continue
            rows, slot = torch.where(indices == expert)
            w1, w2, w3 = self.expert(expert)
            y[rows] += expert_forward(x[rows], w1, w2, w3, self.swiglu_limit, weights[rows, slot, None])
        return y


class CheckpointEngramTable(EngramTable):
    """One Engram layer's n-gram table, either left in the shards or copied into host RAM.

    The two tables are 91.55 and 91.56 GiB of rows, and a forward touches one row per hash column --
    24 per position, two tables -- so the question is not whether the table fits but what a gather
    costs. Out of the mapping it is one seek per row, and on the shingled disk this checkpoint lives
    on that is expensive enough to decide how the model is served: measured 21 to 49 ms for a row
    whose page is not resident, against 0.004 ms for one that is. A 512-token prefill gathers 12,288
    rows per table, so a cold one is 253 s per table and a corpus would keep paying that as it moved
    onto new n-grams, where the same 12,288 rows out of the copy take 9 ms.

    So `resident` is not a tuning knob but a choice of when to pay, and the answer is not close. A
    real run sets it and pays one sequential pass -- measured 373 s per table, 747 s for both, at 271
    MiB/s -- from which every later gather is a memcpy out of a 91.6 GiB array. The host has the room:
    189.1 GiB of codes and scales against the 930 GiB this machine reports available. The streaming
    path exists for a host that cannot spare the memory, and for a test that wants the two to be
    comparable. It is not wrong, it is just cold on every n-gram it has not seen before.

    Dequantization is `modules.dequantize_rows`, the same function `ResidentEngramTable` calls, so a
    truncated table and the real one agree element for element instead of being two writings of one
    formula. It is the reference's own order -- fp32 product, then narrow -- because an E8M0 scale
    over a 256-wide row reaches 2**-13 and narrowing first would round the scale away.
    """

    def __init__(
        self,
        checkpoint: V41Checkpoint,
        layer_id: int,
        *,
        resident: bool = False,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.layer_id = layer_id
        self.weight_key = f"layers.{layer_id}.engram.embed.weight"
        self.scale_key = scale_key(self.weight_key)
        self.block_size = checkpoint.block_size(self.weight_key)
        weight_entry = checkpoint.reader.entry(self.weight_key)
        self.rows_total, self.head_dim = weight_entry.shape
        self.rows_gathered = 0
        # Held as bytes and reinterpreted at the gather, not as float8: the CPU has no float8
        # `index_select` either, so the copy is only useful if it is addressed one byte wide.
        self.codes = self.scales = None
        self.code_dtype = SAFETENSORS_DTYPES[weight_entry.dtype][1]
        self.scale_dtype = SAFETENSORS_DTYPES[checkpoint.reader.entry(self.scale_key).dtype][1]
        if resident:
            self.codes = self._copy_into_ram(self.weight_key, progress)
            self.scales = self._copy_into_ram(self.scale_key, progress)

    def _copy_into_ram(self, key: str, progress: Callable[[str], None] | None) -> torch.Tensor:
        """One sequential pass over one table, then hand its pages back.

        `advise_dontneed_entry` after the copy matters: the pass fills the page cache with as many
        bytes as the copy itself occupies, and the second table needs the same room again.
        """
        entry = self.checkpoint.reader.entry(key)
        if progress is not None:
            progress(f"engram {key} ({entry.nbytes / 2**30:.1f} GiB)")
        tensor = self.checkpoint.reader.load(key)
        self.checkpoint.reader.advise_dontneed_entry(entry)
        return tensor.view(torch.uint8)

    def lookup(self, indices: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        """Dequantize the rows `indices` names into bf16 `[..., head_dim]`."""
        flat = indices.reshape(-1)
        self.rows_gathered += int(flat.numel())
        if self.codes is None:
            values = self.checkpoint.rows(self.weight_key, flat)
            scales = self.checkpoint.rows(self.scale_key, flat)
        else:
            flat = flat.to(device="cpu", dtype=torch.int64)
            values = self.codes[flat].view(self.code_dtype)
            scales = self.scales[flat].view(self.scale_dtype)
        out = dequantize_rows(values, scales, self.block_size)
        out = out.reshape(*indices.shape, self.head_dim)
        return out if device is None else out.to(device)


class EngramHashIds:
    """`NgramHasher`'s row ids as the `[b, s, n_engram_layers, n_hash_cols]` tensor the tree reads.

    `src/encoding/engram.py` is stdlib-only on purpose -- it answers a question about a config and a
    tokenizer and has no reason to need a tensor library -- and `NgramHasher.hash_ids` returns
    `[position][layer][column]` nested lists, which is the shape that is readable and testable there.
    `Backbone.forward` wants the tensor, so this is the bridge, and it is a second *formulation* of
    the same arithmetic rather than a wrapper around the lists: vectorizing is what makes a 5120-token
    prefill cost one gather per look-back instead of 122,880 list operations.
    `tests/test_models_deepseek_v4_1_loader.py` holds the two to the same ids on a token stream split
    across a prefill and a decode, which is the only way a second formulation is worth having.

    The token map, the multipliers, the primes and the bucket offsets are read off the hasher rather
    than recomputed, so the two cannot drift; the cache is this class's own, because the two are
    driven independently and one reset would otherwise clear the other's history.
    """

    DEAD = NgramHasher.DEAD

    def __init__(
        self,
        hasher: NgramHasher,
        *,
        max_batch_size: int = 1,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        self.hasher = hasher
        self.layout = hasher.layout
        self.max_seq_len = int(max_seq_len)
        self.device = device
        self.token_map = torch.tensor(hasher.token_map, dtype=torch.int64, device=device)
        self.pad_id = int(hasher.pad_id)
        self.multipliers = torch.tensor(hasher.multipliers, dtype=torch.int64, device=device)
        # `primes[layer][n-gram index][head]`, flattened for a broadcast modulo against a
        # `[b, s, layers, 1]` running product: the result is `[b, s, layers, heads]` per look-back,
        # and concatenating the look-backs gives the (n-gram, head) column order the offsets assume.
        self.primes = torch.tensor(self.layout.primes, dtype=torch.int64, device=device)
        self.offsets = torch.tensor(
            [list(hasher.offsets[layer]) for layer in range(len(self.layout.layer_ids))],
            dtype=torch.int64,
            device=device,
        )
        self.cache = torch.full(
            (max_batch_size, self.max_seq_len), self.DEAD, dtype=torch.int64, device=device
        )

    @property
    def n_hash_cols(self) -> int:
        return self.layout.n_hash_columns

    def reset(self) -> None:
        """Forget the positions carried across a prefill, exactly as `NgramHasher.reset` does."""
        self.cache.fill_(self.DEAD)

    @torch.inference_mode()
    def __call__(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """token_ids: [b, s] int64. token_mask: [b, s], False for tokens outside any n-gram.
        Returns the row ids, shaped [b, s, n_engram_layers, n_hash_cols]."""
        batch, seqlen = token_ids.shape
        compressed = self.token_map[token_ids.to(device=self.device, dtype=torch.int64)]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, torch.full_like(compressed, self.DEAD))
        self.cache[:batch, start_pos : start_pos + seqlen] = compressed

        positions = torch.arange(
            start_pos, start_pos + seqlen, device=self.device
        ).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.layout.max_ngram_size):
            # `clamp_min(0)` is what makes the read in range; the mask below is what makes the value
            # irrelevant, since a position shorter than `shift` is blocked by its own index.
            source = self.cache[:batch].gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift) | (source == self.DEAD)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)  # [b, s, max_ngram_size]

        # XOR one look-back at a time, so the running value after step i is the (i+1)-gram's hash,
        # and each lands in its own prime-sized bucket range.
        products = tokens.unsqueeze(2) * self.multipliers  # [b, s, layers, max_ngram_size]
        rolling = products[..., 0]
        hashes = []
        for index in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., index])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, index - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


def build_hasher(
    config: V41TextConfig,
    tokenizer_dir: str | None = None,
    *,
    tokenizer=None,
) -> tuple[EngramLayout | None, NgramHasher | None]:
    """The Engram front end a config implies, derived from its tokenizer.

    Both halves are needed to name a row: the layout gives the bucket ranges, and the tokenizer gives
    the compressed id space every hash multiplier derives from. `(None, None)` for a config that
    declares no Engram layers, which is a model with nothing to hash rather than an error.
    """
    layout = EngramLayout.from_config(config.__dict__)
    if layout is None:
        return None, None
    if tokenizer is None:
        if tokenizer_dir is None:
            raise ValueError("an Engram model needs a tokenizer: pass `tokenizer` or `tokenizer_dir`")
        # Imported here rather than at module scope: the tokenizer is the one thing in this file a
        # caller without Engram layers does not need, and `transformers` is a heavy import.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    token_map, size = build_compressed_token_map(tokenizer)
    hasher = NgramHasher(
        layout,
        token_map,
        int(config.engram_pad_id),
        expected_token_map_size=config.engram_compressed_vocab_size,
    )
    if size != hasher.token_map_size:  # pragma: no cover - build_compressed_token_map returns both
        raise ValueError(f"compressed token map has {hasher.token_map_size} ids but reports {size}")
    return layout, hasher


@dataclass
class LoadedBackbone:
    """A `Backbone` fed from a checkpoint, with the hash front end its forward needs.

    `Backbone.forward` takes `hash_ids` rather than computing them, because the ids come from the
    tokenizer and the cache that carries them across a prefill is state the model does not otherwise
    own. This pairs the two so that a caller has one object to call and one to reset, which is the
    smallest thing that is actually runnable.
    """

    model: Backbone
    report: LoadReport
    hasher: NgramHasher | None = None
    hash_ids: EngramHashIds | None = None

    @torch.inference_mode()
    def __call__(
        self,
        token_ids: torch.Tensor,
        start_pos: int = 0,
        image_mask: torch.Tensor | None = None,
    ):
        """One forward over `token_ids`, hashing them first. Returns `Backbone.forward`'s triple."""
        hashes = None
        if self.hash_ids is not None:
            # An image span takes no part in an n-gram, which is the same statement `Backbone` makes
            # when it turns the image mask into an Engram token mask; spelled once, here.
            hashes = self.hash_ids(token_ids, start_pos, None if image_mask is None else ~image_mask)
        return self.model(token_ids, start_pos, hashes, image_mask)

    def reset_state(self, batch_size: int) -> None:
        self.model.reset_state(batch_size)
        if self.hash_ids is not None:
            self.hash_ids.reset()


def load_backbone(
    config: V41TextConfig,
    checkpoint: V41Checkpoint,
    *,
    device: torch.device | str | None = None,
    max_batch_size: int = 1,
    max_seq_len: int | None = None,
    engram: bool = True,
    resident_engram: bool = False,
    layout: EngramLayout | None = None,
    hasher: NgramHasher | None = None,
    tokenizer_dir: str | None = None,
    expert_cache: int = DEFAULT_EXPERT_CACHE,
    progress: Callable[[str], None] | None = None,
) -> LoadedBackbone:
    """Build the V4.1 text backbone and fill it from `checkpoint`.

    The routed experts stay in the shards, so what this allocates is the dense half: about 16 GiB in
    bf16 for the released geometry, against the 476 GiB on disk. `resident_engram` adds the two
    Engram tables on top of that -- 189.1 GiB -- and is what a real run wants; see
    `CheckpointEngramTable` for why gathering them from the shards is not an option here.

    The forward that follows is the host-offload correctness path, and it is measured now rather than
    assumed: a generated token costs 30 to 40 s, of which the attention stack is 0.37 s, and the rest
    is the routed experts -- 320 of them expanded on a first, cold forward, 240 on a warm step, at
    0.122 s each. The cost is the expansion and not the bytes: 0.3% of a miss is reading the mapping
    and 99.7% is turning fp4 codes into fp32 and then bf16. `CheckpointRoutedExperts` is that
    subject, and this function's is only the dense half.
    """
    if hasher is not None:
        layout = hasher.layout
    elif engram and layout is None:
        layout, hasher = build_hasher(config, tokenizer_dir or checkpoint.root)
    elif not engram:
        layout, hasher = None, None

    max_seq_len = config.max_position_embeddings if max_seq_len is None else max_seq_len
    n_layers = config.n_layers if config.n_layers is not None else len(config.compress_ratios)
    routed = {
        layer_id: CheckpointRoutedExperts(
            checkpoint,
            layer_id,
            n_experts=_moe_shape(config, layer_id)[0],
            dim=config.dim,
            inter_dim=config.moe_inter_dim,
            swiglu_limit=config.swiglu_limit or 0.0,
            cache_size=expert_cache,
        )
        for layer_id in range(n_layers)
    }
    tables = (
        {
            layer_id: CheckpointEngramTable(
                checkpoint, layer_id, resident=resident_engram, progress=progress
            )
            for layer_id in layout.layer_ids
        }
        if layout is not None
        else {}
    )

    model = Backbone(
        config,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        layout=layout,
        engram_tables=tables,
        routed=routed,
    )
    report = checkpoint_weights(model, checkpoint, progress=progress)
    if report.missing:
        raise RuntimeError(
            f"{len(report.missing)} parameters have no tensor in the checkpoint and would have "
            f"stayed uninitialized: {report.missing}"
        )
    counted = Counter(_group_of(key) for key in checkpoint.keys())
    for key in report.loaded:
        counted.subtract([_group_of(key)])
    report.unloaded_groups = {group: count for group, count in counted.items() if count > 0}

    front_end = (
        EngramHashIds(hasher, max_batch_size=max_batch_size, max_seq_len=max_seq_len, device=device)
        if hasher is not None
        else None
    )
    return LoadedBackbone(model=model, report=report, hasher=hasher, hash_ids=front_end)
