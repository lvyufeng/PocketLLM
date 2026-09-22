"""MiMo-V2.6's checkpoint: where every tensor is, and how to get at it.

The release is 64 expert-parallel shards (`model_pp0_ep<N>_shard0.safetensors`)
plus `model_mtp.safetensors`, and it does ship a `model.safetensors.index.json`
(73,081 names, `tp_size: 4`). The index is a name-to-file map and nothing more,
and the two things a bank needs are not in it: that **expert `e` is in shard
`e // 4`**, and that **one expert is one contiguous run of six tensors**. So the
layout below is derived from the shards' own headers -- every one of them is read
at open, index or not -- and checked against the config, and the index serves only
as the list of files to read. A release that interleaved the experts, or wrote
their tensors in a different order, has to fail here rather than stage the right
bytes into a kernel in the wrong arrangement.

Three quantization schemes are live at once and they are selected by *where* a
tensor is, not by a flag:

* the routed experts are MXFP4 -- `[N, K/2]` uint8 holding two E2M1 codes per
  byte, beside an `[N, K/32]` uint8 E8M0 scale. That is the ABI
  `moe_single_token_fp4_cuda` already takes verbatim, so expert weights are never
  expanded in this repository;
* every dense linear except `o_proj` is FP8 E4M3 under 128x128 tile scales
  (`<name>.weight_scale_inv`, float32), tile-normalised so that
  `w = w_fp8 * scale`;
* `o_proj`, the norms, the router, the embedding and the head are BF16.

And one anomaly, because a loader that gets it wrong reads a plausible tensor
from the right place: on a **global-attention** layer `qkv_proj.weight_scale_inv`
has 108 rows for a weight with 106 row-blocks, while every sliding-window layer
matches exactly. The 108 is 4 x 27 and the 106 is 106: the projection is stored
as four tensor-parallel shards of `[q | k | v]` and was quantised one shard at a
time, so its tiles restart at every 3392-row shard boundary. That is a multiple
of 128 on a sliding-window layer (3712 = 29 tiles) and is not on a global one
(3392 = 26.5 tiles), which is why only the global layers show it. Reading the
scale as one run of 106 tiles instead pulls each shard's first 128 rows -- the
head of its query block -- onto the previous shard's last tile, which covers its
small value block, so `dense_tensor` passes `QKV_SHARDS` for that weight.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from src.loader.safetensors import SAFETENSORS_DTYPES, MmapSafetensors, TensorEntry
from src.models.mimo_v2.config import MimoV2Config, MimoV2TextConfig
from src.models.mimo_v2.quant import FP8_BLOCK, QKV_SHARDS, dequant_fp8_block

__all__ = [
    "EXPERT_PROJECTIONS",
    "FUSED_QKV_SUFFIX",
    "MimoV2Checkpoint",
    "MimoV2ExpertLayout",
    "checkpoint_name",
    "dequant_fp8_block",
    "dense_weight_keys",
    "layer_weight_keys",
    "shard_of_expert",
]

#: The tail of the one FP8 weight whose scale is blocked per tensor-parallel
#: shard. Everything else in the checkpoint is blocked across the whole tensor.
FUSED_QKV_SUFFIX = "self_attn.qkv_proj.weight"

#: The three routed projections, in the order a shard stores one expert's bytes.
#: `gate`/`up` are the two halves of the SwiGLU and `down` is its output
#: projection, which is the same order the fp4 MoE kernels take as `w1`/`w3`/`w2`.
EXPERT_PROJECTIONS = ("down_proj", "gate_proj", "up_proj")

#: What accompanies each projection: packed codes then the E8M0 scale row.
EXPERT_KINDS = ("weight", "weight_scale")

_SHARD_RE = re.compile(r"^model_pp0_ep(\d+)_shard0\.safetensors$")
_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>down_proj|gate_proj|up_proj)\.(?P<kind>weight|weight_scale)$"
)

def shard_of_expert(expert: int, experts_per_shard: int) -> int:
    """Which expert-parallel rank holds this expert. Contiguous, never strided."""
    return expert // experts_per_shard


def checkpoint_name(layer_idx: int, suffix: str, prefix: str = "model.layers") -> str:
    """A backbone tensor's name, so call sites do not each format it differently."""
    return f"{prefix}.{layer_idx}.{suffix}"


@dataclass(frozen=True)
class _ExpertRun:
    """One expert's six tensors, in file order, with their byte ranges."""

    layer: int
    expert: int
    file_name: str
    begin: int
    end: int
    parts: tuple[tuple[str, str, int, int], ...]

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    def offsets(self, proj: str, kind: str) -> tuple[int, int]:
        for name, part_kind, begin, end in self.parts:
            if name == proj and part_kind == kind:
                return begin - self.begin, end - self.begin
        raise KeyError(f"{proj}.{kind} is not part of expert ({self.layer}, {self.expert})")

    def shapes(self) -> dict[tuple[str, str], tuple[int, ...]]:
        raise NotImplementedError


@dataclass(frozen=True)
class MimoV2ExpertLayout:
    """Where the routed experts are, verified against the shards rather than assumed.

    `experts_per_shard` comes from the checkpoint's own file names, and the
    contiguity property is asserted: if a release interleaved the experts, or
    ordered a shard's tensors differently, `expert_runs()` would hand a kernel the
    right bytes in the wrong arrangement and nothing downstream would notice.
    """

    root: str
    experts_per_shard: int
    n_experts: int
    n_layers: int
    moe_layers: tuple[int, ...]
    expert_bytes: int
    projections: tuple[str, ...]
    kinds: tuple[str, ...]
    shapes: dict[str, tuple[int, ...]]
    scale_shapes: dict[str, tuple[int, ...]]
    files: tuple[str, ...]
    #: layer -> expert -> run. Built once; a 47 x 256 mapping of six parts each.
    runs: dict[int, dict[int, _ExpertRun]] = field(default_factory=dict)

    def shard_of(self, expert: int) -> int:
        return shard_of_expert(expert, self.experts_per_shard)

    def run(self, layer: int, expert: int) -> _ExpertRun:
        try:
            return self.runs[layer][expert]
        except KeyError:
            raise KeyError(
                f"layer {layer} expert {expert} is not in the checkpoint's routed experts "
                f"(layers {self.moe_layers[0]}..{self.moe_layers[-1]}, experts 0..{self.n_experts - 1})"
            ) from None

    def layer_runs(self, layer: int) -> list[_ExpertRun]:
        return [self.runs[layer][e] for e in sorted(self.runs[layer])]

    def describe(self) -> str:
        gib = self.expert_bytes * self.n_experts * len(self.moe_layers) / 2**30
        return (
            f"{len(self.moe_layers)} routed layers x {self.n_experts} experts x "
            f"{self.expert_bytes / 2**20:.2f} MiB = {gib:.2f} GiB; "
            f"{self.experts_per_shard} experts per shard, expert e in shard e // "
            f"{self.experts_per_shard}"
        )


def _read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return header, 8 + header_len


def _expert_layout(root: str, files: list[str], n_experts: int) -> MimoV2ExpertLayout:
    """Walk the shards' headers and build the expert map, refusing anything unexpected."""
    shards = {}
    for name in files:
        match = _SHARD_RE.match(name)
        if match:
            shards[int(match.group(1))] = name
    if not shards:
        raise ValueError(
            f"no model_pp0_ep<N>_shard0.safetensors in {root}; this is not a MiMo-V2 "
            f"expert-parallel release"
        )
    indices = sorted(shards)
    if indices != list(range(len(indices))):
        raise ValueError(f"the expert shards are not contiguous: {indices}")

    per_shard: dict[int, int] = {}
    runs: dict[int, dict[int, _ExpertRun]] = {}
    projections: tuple[str, ...] | None = None
    kinds: tuple[str, ...] | None = None
    shapes: dict[str, tuple[int, ...]] = {}
    scale_shapes: dict[str, tuple[int, ...]] = {}
    sizes: set[int] = set()

    for index in indices:
        header, _ = _read_header(os.path.join(root, shards[index]))
        grouped: dict[tuple[int, int], list[tuple[int, int, str, str]]] = {}
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            match = _EXPERT_RE.match(key)
            if not match:
                continue
            layer, expert = int(match.group("layer")), int(match.group("expert"))
            begin, end = meta["data_offsets"]
            grouped.setdefault((layer, expert), []).append(
                (begin, end, match.group("proj"), match.group("kind"))
            )
            if match.group("kind") == "weight":
                shapes[match.group("proj")] = tuple(meta["shape"])
            else:
                scale_shapes[match.group("proj")] = tuple(meta["shape"])

        counts = {len(parts) for parts in grouped.values()}
        if counts != {len(EXPERT_PROJECTIONS) * len(EXPERT_KINDS)}:
            raise ValueError(
                f"{shards[index]}: experts carry {sorted(counts)} tensors each, expected "
                f"{len(EXPERT_PROJECTIONS) * len(EXPERT_KINDS)}"
            )

        experts_here = sorted({expert for _, expert in grouped})
        per_shard[index] = len(experts_here)
        expected_first = index * len(experts_here)
        if experts_here != list(range(expected_first, expected_first + len(experts_here))):
            raise ValueError(
                f"{shards[index]} holds experts {experts_here[0]}..{experts_here[-1]}, but "
                f"expert e is expected in shard e // {len(experts_here)}, i.e. "
                f"{expected_first}..{expected_first + len(experts_here) - 1}"
            )

        for (layer, expert), parts in grouped.items():
            parts.sort()
            order = tuple((p[2], p[3]) for p in parts)
            wanted = tuple(
                (proj, kind) for proj in EXPERT_PROJECTIONS for kind in EXPERT_KINDS
            )
            if order != wanted:
                raise ValueError(
                    f"{shards[index]} stores expert ({layer}, {expert}) as {order}, "
                    f"expected {wanted}; the bank reads one expert as one contiguous run"
                )
            begin, end = parts[0][0], parts[-1][1]
            if sum(p[1] - p[0] for p in parts) != end - begin:
                raise ValueError(
                    f"{shards[index]}: expert ({layer}, {expert}) spans {end - begin} bytes "
                    f"but its tensors are {sum(p[1] - p[0] for p in parts)}; not contiguous"
                )
            sizes.add(end - begin)
            runs.setdefault(layer, {})[expert] = _ExpertRun(
                layer=layer,
                expert=expert,
                file_name=shards[index],
                begin=begin,
                end=end,
                parts=tuple(parts),
            )
            projections = tuple(EXPERT_PROJECTIONS)
            kinds = tuple(EXPERT_KINDS)

    if len(sizes) != 1:
        raise ValueError(f"experts are not all the same size: {sorted(sizes)}")
    if len(set(per_shard.values())) != 1:
        raise ValueError(f"shards hold different expert counts: {sorted(set(per_shard.values()))}")
    experts_per_shard = next(iter(per_shard.values()))
    if experts_per_shard * len(indices) != n_experts:
        raise ValueError(
            f"{len(indices)} shards x {experts_per_shard} experts != the config's {n_experts}"
        )

    moe_layers = tuple(sorted(runs))
    for layer, experts in runs.items():
        if sorted(experts) != list(range(n_experts)):
            raise ValueError(
                f"layer {layer} has {len(experts)} experts, expected {n_experts}"
            )

    return MimoV2ExpertLayout(
        root=root,
        experts_per_shard=experts_per_shard,
        n_experts=n_experts,
        n_layers=len(moe_layers),
        moe_layers=moe_layers,
        expert_bytes=next(iter(sizes)),
        projections=projections or EXPERT_PROJECTIONS,
        kinds=kinds or EXPERT_KINDS,
        shapes=shapes,
        scale_shapes=scale_shapes,
        files=tuple(shards[i] for i in indices),
        runs=runs,
    )


def layer_weight_keys(layer_idx: int, config: MimoV2TextConfig) -> list[str]:
    """Every backbone tensor one decoder layer owns, in the checkpoint's names."""
    root = f"model.layers.{layer_idx}"
    keys = [
        f"{root}.input_layernorm.weight",
        f"{root}.post_attention_layernorm.weight",
        f"{root}.self_attn.qkv_proj.weight",
        f"{root}.self_attn.qkv_proj.weight_scale_inv",
        f"{root}.self_attn.o_proj.weight",
    ]
    if config.attention(layer_idx).has_sink:
        keys.append(f"{root}.self_attn.attention_sink_bias")
    if config.ffn_kind(layer_idx) == "moe":
        keys += [f"{root}.mlp.gate.weight", f"{root}.mlp.gate.e_score_correction_bias"]
    else:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            keys += [f"{root}.mlp.{proj}.weight", f"{root}.mlp.{proj}.weight_scale_inv"]
    return keys


def dense_weight_keys(config: MimoV2TextConfig) -> list[str]:
    """Every non-expert tensor of the text backbone."""
    keys = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for layer in range(config.num_hidden_layers):
        keys += layer_weight_keys(layer, config)
    return keys


class MimoV2Checkpoint:
    """Header-level access to the release, plus the two ways to read a tensor.

    `view` aliases the mapped bytes and is what a `pread`-free loader wants;
    `read` copies into a tensor of the requested dtype and device. Neither
    expands a quantized weight -- the FP8 tiles are dequantized by
    `dense_tensor`, and MXFP4 experts are never expanded at all.
    """

    def __init__(self, root: str, config: MimoV2Config | None = None) -> None:
        self.root = os.path.abspath(root)
        self.config = config if config is not None else MimoV2Config.from_pretrained(self.root)
        self.mmap = MmapSafetensors(self.root)
        self.layer = self.config.text
        self.layout = _expert_layout(
            self.root, self.mmap._shard_files(), self.layer.n_routed_experts
        )
        if tuple(self.layout.moe_layers) != self.layer.moe_layer_indices:
            raise ValueError(
                f"the shards carry experts on layers {self.layout.moe_layers} but the config "
                f"says MoE is on {self.layer.moe_layer_indices}"
            )
        missing = [key for key in dense_weight_keys(self.layer) if key not in self.mmap]
        if missing:
            raise ValueError(
                f"the checkpoint is missing {len(missing)} backbone tensor(s): {missing[:6]}"
            )

    # -- header level ----------------------------------------------------

    def __contains__(self, key: str) -> bool:
        return key in self.mmap

    def entry(self, key: str) -> TensorEntry:
        return self.mmap.entry(key)

    @property
    def device_backbone(self) -> str:
        return "the 841 non-expert tensors of model_pp0_ep0_shard0.safetensors"

    def dense_bytes(self) -> int:
        return sum(self.entry(key).nbytes for key in dense_weight_keys(self.layer))

    def expert_bytes(self) -> int:
        return self.layout.expert_bytes * self.layout.n_experts * len(self.layout.moe_layers)

    # -- reading ---------------------------------------------------------

    def view(self, key: str) -> torch.Tensor:
        """A torch tensor aliasing the mapped bytes, at the stored dtype."""
        return self.mmap.view(key)

    def read(
        self,
        key: str,
        dtype: torch.dtype | None = None,
        device: torch.device | str = "cpu",
        copy: bool = False,
    ) -> torch.Tensor:
        """A standalone tensor, cast if asked.

        The default is a copy rather than a view: a mapped view of a 2 GB embedding
        keeps every page it touches referenced, and a loader that moves a whole
        checkpoint to the device wants the process to be able to drop them again.
        """
        entry = self.entry(key)
        np_dtype, torch_dtype = SAFETENSORS_DTYPES[entry.dtype]
        base = self.mmap.data_offset(entry.file_name)
        raw = self.mmap._map(entry.file_name)[base + entry.begin : base + entry.end]
        array = np.array(raw.view(np_dtype), copy=True) if copy else np.asarray(raw.view(np_dtype))
        tensor = torch.from_numpy(array)
        if torch_dtype != tensor.dtype:
            tensor = tensor.view(torch_dtype)
        tensor = tensor.reshape(entry.shape)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor.to(device)

    def dense_tensor(
        self,
        key: str,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """One backbone tensor at a compute dtype, dequantizing FP8 block weights.

        A tensor stored as FP8 E4M3 is returned as `w_fp8 * scale` at `dtype`; MXFP4
        codes are refused, because expanding them is exactly what this repository
        does not do.
        """
        entry = self.entry(key)
        if entry.dtype == "F8_E4M3":
            scale_key = f"{key}_scale_inv" if not key.endswith("_scale_inv") else key
            if key.endswith("_scale_inv"):
                raise ValueError(f"{key} is a scale and not a weight")
            if scale_key not in self.mmap:
                raise ValueError(f"{key} is FP8 but {scale_key} is not in the checkpoint")
            codes = self.read(key, copy=False)
            scale = self.read(scale_key, copy=False)
            shards = QKV_SHARDS if key.endswith(FUSED_QKV_SUFFIX) else 1
            return dequant_fp8_block(codes, scale, FP8_BLOCK, dtype, shards).to(device)
        if entry.dtype == "U8" and "experts" in key:
            raise ValueError(
                f"{key} is a packed MXFP4 expert weight; pass it to the expert path "
                f"instead of expanding it"
            )
        return self.read(key, dtype, device, copy=False)

    def expert_arrays(self, layer: int, expert: int) -> dict[tuple[str, str], torch.Tensor]:
        """One expert's six tensors as zero-copy views, keyed `(projection, kind)`.

        The views are the checkpoint's own `[N, K/2]` and `[N, K/32]` uint8 arrays.
        """
        run = self.layout.run(layer, expert)
        mm = self.mmap._map(run.file_name)
        base = self.mmap.data_offset(run.file_name)
        out: dict[tuple[str, str], torch.Tensor] = {}
        for proj in self.layout.projections:
            for kind in self.layout.kinds:
                entry = self.mmap.entry(
                    f"model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}"
                )
                raw = mm[base + entry.begin : base + entry.end]
                tensor = torch.from_numpy(np.asarray(raw.view(np.uint8)))
                out[(proj, kind)] = tensor.reshape(entry.shape)
        return out

    def expert_region(self, layer: int, shard_index: int) -> tuple[str, int, int]:
        """The byte range covering one shard's share of one layer's experts.

        Returns `(file_name, begin, end)`. A shard holds `experts_per_shard`
        experts of *every* routed layer, and those are one contiguous run, so this
        is the `pread` a bank wants per (shard, layer): 4 experts, 51 MiB, 3008
        reads for the whole checkpoint. Layers whose experts are not one run in
        that shard raise here rather than being read as a plausible superset.
        """
        files = self.layout.files
        runs = [run for run in self.layout.layer_runs(layer) if run.file_name == files[shard_index]]
        if not runs:
            raise KeyError(f"shard {shard_index} holds no experts of layer {layer}")
        begin, end = runs[0].begin, runs[-1].end
        if sum(run.nbytes for run in runs) != end - begin:
            raise ValueError(
                f"layer {layer}'s {len(runs)} experts are not one contiguous run in "
                f"{runs[0].file_name}"
            )
        return runs[0].file_name, begin, end

    def experts_of_shard(self, shard_index: int) -> tuple[int, ...]:
        """Which global expert ids one shard owns. Contiguous by construction."""
        lo = shard_index * self.layout.experts_per_shard
        return tuple(range(lo, lo + self.layout.experts_per_shard))

    def describe(self) -> str:
        return (
            f"{self.root}\n"
            f"  {len(self.mmap)} tensors, {self.mmap.nbytes_total() / 1e9:.2f} GB\n"
            f"  backbone {self.dense_bytes() / 1e9:.2f} GB in ep0\n"
            f"  experts: {self.layout.describe()}"
        )
