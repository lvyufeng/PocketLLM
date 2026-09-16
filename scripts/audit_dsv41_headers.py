#!/usr/bin/env python3
"""Read-only audit of a DeepSeek-V4.1-Flash checkpoint's safetensors headers.

DeepSeek-V4.1-Flash (552B backbone, CED, CSA2, Engram, DSpark) is a different
architecture from the DeepSeek-V4-Flash this repository supports, and none of it
is implemented yet. Before any of it can be, two things have to be facts rather
than readings of a model card: the config's own consistency, and the tensor
inventory the checkpoint actually ships.

This script answers both without downloading or mapping a single weight. A
safetensors file starts with an 8-byte little-endian header length, then that
many bytes of JSON describing every tensor's dtype, shape and byte offsets. That
JSON is all this needs, so it runs against a header-only prefix tree (what you
get from fetching the first few MB of each shard) exactly as it does against a
complete checkpoint -- see `--header-prefix` / automatic detection.

What it checks, and why each one matters:

- Config self-consistency. `compress_ratios`, the KV/index source layers and the
  candidate source must agree with each other and with `n_layers`; the Engram
  hash-table row count is *re-derived* from `engram_vocab_size`, `engram_n_heads`
  and `engram_max_ngram_size` and compared against `engram_num_embeddings`.
- Quantization granularity. Every `weight`/`scale` pair must block evenly, and
  every tensor's byte extent must equal its shape times its dtype size -- which
  is what proves the routed experts are nibble-packed (I8 holding 2 FP4 values
  per byte) rather than stored unpacked.
- The tensor inventory. Which layers own a compressor, which own which half of
  the indexer, where the Engram tables and the DSpark heads live, and how the
  bytes divide between them.

Exit status is 0 only when every check passes; failures are listed with the
expected and observed values.

Usage:

    python scripts/audit_dsv41_headers.py --checkpoint-dir /path/to/DeepSeek-V4.1-Flash
    python scripts/audit_dsv41_headers.py --checkpoint-dir /tmp/dsv41 --header-prefix
    python scripts/audit_dsv41_headers.py --config cfg.json --list-tensors 'engram'
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict

# The config is read through `src.models.deepseek_v4_1.config`, which is itself
# standard library only, so the "runs under any interpreter" property holds --
# but running this file as a script puts `scripts/` on sys.path rather than the
# repository root, so the root has to be added before that import resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# safetensors stores the header length as a little-endian u64 ahead of the JSON.
HEADER_LEN = struct.Struct("<Q")

# Bytes per element, by safetensors dtype name. F4 is the one special case: two
# nibbles per byte, so its byte count is half its logical element count. The
# published checkpoint does not use it -- it ships routed experts as I8 with the
# packed shape, and `check_packing` is what proves that -- but a repacked export
# might, and the extent check has to stay honest either way.
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
NIBBLE_DTYPES = {"F4"}

# The two Engram tables are the largest tensors in the checkpoint by a wide
# margin; naming them keeps the report readable.
ENGRAM_ROW_TOLERANCE = 0


class Report:
    """Collects check outcomes so the summary can be printed in one place."""

    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((name, bool(ok), detail))
        return bool(ok)

    def fail(self, name: str, detail: str) -> bool:
        return self.check(name, False, detail)

    @property
    def failures(self) -> list[tuple[str, str]]:
        return [(n, d) for n, ok, d in self.checks if not ok]


# ---------------------------------------------------------------------------
# safetensors header reading
# ---------------------------------------------------------------------------


def read_header(path: str) -> tuple[dict, int, bool]:
    """Return (header, header_len, complete) for one shard.

    `complete` is False when the file is a header-only prefix shorter than the
    payload its own header declares, which is the offline auditing case.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) < 8:
            raise ValueError(f"{path}: too short to hold a header length")
        header_len = HEADER_LEN.unpack(raw)[0]
        payload = handle.read(header_len)
        if len(payload) < header_len:
            raise ValueError(f"{path}: header claims {header_len} bytes, file has {len(payload)}")
    header = json.loads(payload)
    payload_bytes = max((entry["data_offsets"][1] for key, entry in header.items() if key != "__metadata__"), default=0)
    complete = size >= 8 + header_len + payload_bytes
    return header, header_len, complete


def shard_paths(directory: str) -> list[str]:
    names = sorted(name for name in os.listdir(directory) if name.endswith(".safetensors"))
    if names:
        return [os.path.join(directory, name) for name in names]
    # A header-prefix tree keeps whatever extension the fetch used; accept any
    # file whose first eight bytes look like a plausible header length.
    candidates = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path) or os.path.getsize(path) < 8:
            continue
        with open(path, "rb") as handle:
            header_len = HEADER_LEN.unpack(handle.read(8))[0]
        if 0 < header_len <= os.path.getsize(path) - 8 and header_len < 256 * 1024 * 1024:
            candidates.append(path)
    return candidates


def load_inventory(paths: list[str]) -> tuple[dict, dict, bool, list[str]]:
    """Read every shard header into {tensor: (dtype, shape, bytes, shard)}.

    A header-fetch scratch directory can end up holding the same shard twice under
    two names (`.safetensors` and a `.head` prefix, say). Byte-identical headers
    are skipped as duplicates; two *different* headers claiming the same tensor is
    an error, because that is a real inconsistency in the checkpoint.
    """
    tensors: dict[str, tuple[str, tuple, int, str]] = {}
    per_shard: dict[str, dict] = {}
    all_complete = True
    duplicates: list[str] = []
    seen_headers: dict[bytes, str] = {}
    for path in paths:
        header, header_len, complete = read_header(path)
        with open(path, "rb") as handle:
            handle.seek(8)
            digest = handle.read(header_len)
        if digest in seen_headers:
            duplicates.append(f"{os.path.basename(path)} == {seen_headers[digest]}")
            continue
        seen_headers[digest] = os.path.basename(path)
        all_complete = all_complete and complete
        entries = {key: value for key, value in header.items() if key != "__metadata__"}
        per_shard[os.path.basename(path)] = entries
        for key, entry in entries.items():
            shape = tuple(entry["shape"])
            span = entry["data_offsets"][1] - entry["data_offsets"][0]
            if key in tensors:
                raise ValueError(f"{key}: declared in both {tensors[key][3]} and {os.path.basename(path)}")
            tensors[key] = (entry["dtype"], shape, span, os.path.basename(path))
    return tensors, per_shard, all_complete, duplicates


# ---------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = (
    "vocab_size",
    "dim",
    "moe_inter_dim",
    "n_layers",
    "n_mtp_layers",
    "n_heads",
    "n_routed_experts",
    "n_activated_experts",
    "n_shared_experts",
    "head_dim",
    "rope_head_dim",
    "q_lora_rank",
    "o_lora_rank",
    "o_groups",
    "window_size",
    "compress_ratios",
    "kv_source_layers",
    "index_source_layers",
    "candidate_source_layer",
    "index_topk",
    "index_n_heads",
    "index_head_dim",
    "hc_mult",
    "engram_layer_ids",
    "engram_num_embeddings",
    "engram_max_ngram_size",
    "engram_n_heads",
    "engram_head_dim",
    "engram_vocab_size",
    "engram_compressed_vocab_size",
    "dspark_target_layer_ids",
    "dspark_markov_rank",
    "dspark_n_routed_experts",
    "vision_n_layers",
    "vision_dim",
    "vision_patch_size",
)


def load_config(path: str) -> dict:
    """The config as the flat reference key set, from either released shape.

    The checkpoint ships `config.json` in the Transformers layout -- the text
    hyper-parameters nested under `text_config` and the layer lists spelled
    `*_layer_ids` -- and `inference/config.json` in the reference runtime's flat
    one. `V41Config` reads both and `as_reference_dict` writes the flat one back
    out, so every check below this line sees the same keys whichever file was
    passed and only one place has to know how the two relate.
    """
    from src.models.deepseek_v4_1.config import from_dict

    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    return from_dict(raw, source=path).as_reference_dict()


def check_config(config: dict, report: Report) -> None:
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    report.check("config: every required key is present", not missing, f"missing {missing}" if missing else "")
    if missing:
        return

    layers, mtp = config["n_layers"], config["n_mtp_layers"]
    ratios = list(config["compress_ratios"])
    report.check(
        "config: len(compress_ratios) == n_layers + n_mtp_layers",
        len(ratios) == layers + mtp,
        f"{len(ratios)} vs {layers} + {mtp} = {layers + mtp}",
    )

    backbone = ratios[:layers]
    nonzero = {i for i, ratio in enumerate(backbone) if ratio > 0}
    # `compress_ratios[l] > 0` marks a layer whose attention reads the compressed
    # positions, not one that produces them -- only kv_source_layers pool their own
    # KV (inference_model.py: `compress_ratio > 0 does not mean the layer compresses
    # its own KV: only kv_source_layers do`). Every source layer is therefore also a
    # consumer, but not the other way round.
    for source_key in ("kv_source_layers", "index_source_layers"):
        sources = sorted(config[source_key])
        unread = [layer for layer in sources if layer not in nonzero]
        report.check(
            f"config: every {source_key} entry reads compressed positions",
            not unread,
            f"compress_ratios==0 at {unread}" if unread else "",
        )
    report.check(
        "config: MTP layers compress nothing",
        all(ratio == 0 for ratio in ratios[layers:]),
        f"tail={ratios[layers:]}",
    )
    report.check(
        "config: kv_source_layers is a subset of index_source_layers",
        set(config["kv_source_layers"]) <= set(config["index_source_layers"]),
        f"kv={config['kv_source_layers']} index={config['index_source_layers']}",
    )

    # The indexer K is shared from a KV source, so a layer that owns an indexer
    # but no K must still have a source below it to read from.
    sources = sorted(config["index_source_layers"])
    stray = [i for i in sources if not [k for k in config["kv_source_layers"] if k <= i]]
    report.check("config: every index source has a KV source at or below it", not stray, f"uncovered {stray}")

    candidate = config["candidate_source_layer"]
    report.check(
        "config: candidate_source_layer is the first layer after kv_source_layers[-1]",
        candidate == sorted(config["kv_source_layers"])[-1] == layers // 2,
        f"candidate={candidate} last_kv_source={sorted(config['kv_source_layers'])[-1]} n_layers//2={layers // 2}",
    )

    targets = list(config["dspark_target_layer_ids"])
    report.check(
        "config: dspark_target_layer_ids are the last n_mtp_layers backbone layers",
        targets == list(range(layers - mtp, layers)),
        f"{targets} vs {list(range(layers - mtp, layers))}",
    )

    layer_ids = list(config["engram_layer_ids"])
    report.check(
        "config: one Engram table size per Engram layer",
        len(config["engram_num_embeddings"]) == len(layer_ids),
        f"{len(config['engram_num_embeddings'])} sizes for {layer_ids}",
    )
    report.check(
        "config: engram_layer_ids are inside the backbone",
        all(0 <= layer_id < layers for layer_id in layer_ids),
        f"{layer_ids}",
    )
    # 99092 is the one Engram constant that is not arithmetic -- it is the size of
    # the compressed token map, i.e. the number of distinct normalized token texts
    # the tokenizer produces. The reference asserts the same equality at load
    # time, and a mismatch silently rehashes the whole table, so it is worth
    # stating in the report even though the audit cannot recompute it alone.
    report.check(
        "config: engram_compressed_vocab_size is set",
        config["engram_compressed_vocab_size"] > 0,
        f"engram_compressed_vocab_size={config['engram_compressed_vocab_size']} "
        f"(must equal the compressed token-map size; see --tokenizer)",
    )


# ---------------------------------------------------------------------------
# Engram bucket layout
# ---------------------------------------------------------------------------


def is_prime(value: int) -> bool:
    """Deterministic Miller-Rabin for the 64-bit range these primes live in."""
    if value < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % small == 0:
            return value == small
    d, s = value - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for base in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(base, d, value)
        if x in (1, value - 1):
            continue
        for _ in range(s - 1):
            x = x * x % value
            if x == value - 1:
                break
        else:
            return False
    return True


def next_prime(start: int, seen: set) -> int:
    """The smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def check_engram_tables(config: dict, report: Report) -> list[int]:
    """Re-derive the Engram row counts and compare them to the config.

    Each (n-gram size, head) pair owns a prime-sized bucket range; the primes are
    drawn in order starting just above `engram_vocab_size`, handed out across every
    layer without reuse, so the ranges stay disjoint. A position's bucket id is
    `hash % prime + offset`, and the offsets of a layer are a running sum over that
    layer's full prime list -- all `(max_ngram_size - 1) * n_heads` of them, in
    order -- so the largest id any layer can produce is `sum(primes) - 1` and the
    table needs exactly `sum(primes)` rows. This is pure arithmetic on config
    values, so it is checkable with no tokenizer and no weights.
    """
    layer_ids = list(config["engram_layer_ids"])
    if not layer_ids:
        return []
    max_ngram_size = config["engram_max_ngram_size"]
    n_heads = config["engram_n_heads"]

    primes: list[list[tuple[int, ...]]] = []
    seen: set[int] = set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, current = [], config["engram_vocab_size"] - 1
            for _ in range(n_heads):
                current = next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        primes.append(per_ngram)

    required = [sum(prime for group in layer for prime in group) for layer in primes]

    declared = list(config["engram_num_embeddings"])
    report.check(
        "engram: the tables are large enough for the derived bucket ranges",
        all(declared[i] >= required[i] for i in range(len(layer_ids))),
        f"declared={declared} required_min={required}",
    )
    report.check(
        "engram: the tables are no larger than the derived ranges plus tolerance",
        all(abs(declared[i] - required[i]) <= ENGRAM_ROW_TOLERANCE for i in range(len(layer_ids))),
        f"declared={declared} derived={required}",
    )
    return required


# ---------------------------------------------------------------------------
# tensor inventory
# ---------------------------------------------------------------------------

# Ordered: the first pattern that matches wins, so the specific cases come first.
CATEGORY_PATTERNS = (
    ("routed_experts", re.compile(r"(^|\.)ffn\.experts\.\d+\.")),
    ("engram", re.compile(r"\.engram\.")),
    ("vision_aligner", re.compile(r"^(vision\.|aligner\.|image_(start|end|newline)$)")),
    ("mtp_dspark", re.compile(r"^mtp\.")),
    ("shared_experts", re.compile(r"\.ffn\.shared_experts\.")),
    ("attention", re.compile(r"^layers\.\d+\.(attn\.|attn_norm\.)")),
    ("embed_head", re.compile(r"^(embed\.|head\.|norm\.)")),
)


def categorize(name: str) -> str:
    for label, pattern in CATEGORY_PATTERNS:
        if pattern.search(name):
            return label
    return "layer_other"


def check_packing(tensors: dict, report: Report) -> list[str]:
    """Every tensor's byte extent must match its shape and dtype.

    This is the check that proves the routed experts are packed: an I8 tensor of
    [2304, 2560] is 5,898,240 bytes, i.e. two FP4 values per byte over a logical
    [2304, 5120] weight, and an unpacked export would not match.
    """
    mismatched = []
    for name, (dtype, shape, span, _shard) in tensors.items():
        elements = 1
        for dim in shape:
            elements *= dim
        if dtype in NIBBLE_DTYPES:
            expected = (elements + 1) // 2
        elif dtype in DTYPE_BYTES:
            expected = elements * DTYPE_BYTES[dtype]
        else:
            mismatched.append(f"{name}: unknown dtype {dtype}")
            continue
        if expected != span:
            mismatched.append(f"{name}: {dtype}{list(shape)} declares {span} bytes, shape implies {expected}")
    report.check(
        "packing: every tensor's byte extent matches its shape and dtype",
        not mismatched,
        "; ".join(mismatched[:6]) + (f" (+{len(mismatched) - 6} more)" if len(mismatched) > 6 else ""),
    )
    return mismatched


def block_of(weight_shape, scale_shape) -> tuple[int, int] | None:
    if len(weight_shape) != 2 or len(scale_shape) != 2:
        return None
    if weight_shape[0] % scale_shape[0] or weight_shape[1] % scale_shape[1]:
        return None
    return weight_shape[0] // scale_shape[0], weight_shape[1] // scale_shape[1]


def check_scale_pairs(tensors: dict, report: Report, expect_block: tuple[int, int]) -> None:
    """Quantized weights must divide evenly by their scale, at the expected block.

    The two Engram tables are the exception: they are stored fp8 with one E8M0
    scale per 32 channels and no row blocking at all, so their block is 1 x 32
    rather than the 32 x 32 every other fp8 weight uses. That is a fact about the
    checkpoint, not a rounding of the rule, so it is checked separately.
    """
    blocks: Counter = Counter()
    engram_blocks: Counter = Counter()
    broken = []
    for name, (dtype, shape, _span, _shard) in tensors.items():
        if not name.endswith(".weight") or dtype == "I8":
            continue
        scale_name = name[: -len(".weight")] + ".scale"
        if scale_name not in tensors:
            continue
        block = block_of(shape, tensors[scale_name][1])
        if block is None:
            broken.append(f"{name}{list(shape)} vs {list(tensors[scale_name][1])}")
            continue
        (engram_blocks if ".engram.embed." in name else blocks)[block] += 1
    report.check("scales: every weight/scale pair blocks evenly", not broken, "; ".join(broken[:6]))
    report.check(
        f"scales: all non-Engram FP8 weights use a {expect_block[0]}x{expect_block[1]} block",
        set(blocks) == {expect_block},
        f"observed blocks {dict(blocks)}",
    )
    report.check(
        "scales: the Engram tables use a 1x32 per-row block",
        set(engram_blocks) == {(1, 32)},
        f"observed blocks {dict(engram_blocks)}",
    )


def check_backbone(config: dict, tensors: dict, report: Report) -> None:
    layers, mtp = config["n_layers"], config["n_mtp_layers"]
    dim, inter = config["dim"], config["moe_inter_dim"]
    hc_mult = config["hc_mult"]
    mix_hc = (2 + hc_mult) * hc_mult

    missing = []
    for layer in range(layers):
        prefix = f"layers.{layer}."
        for suffix in (
            "attn.wq_a.weight",
            "attn.wq_b.weight",
            "attn.wkv.weight",
            "attn.wo_a.weight",
            "attn.wo_b.weight",
            "attn.q_norm.weight",
            "attn.kv_norm.weight",
            "attn.attn_sink",
            "attn_norm.weight",
            "ffn.gate.weight",
            "ffn_norm.weight",
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
        ):
            if prefix + suffix not in tensors:
                missing.append(prefix + suffix)
    report.check("inventory: every backbone layer has its full tensor set", not missing, "; ".join(missing[:6]))

    def shape_of(name):
        return tensors[name][1] if name in tensors else None

    expected = {
        "embed.weight": (config["vocab_size"], dim),
        "head.weight": (config["vocab_size"], dim),
        "norm.weight": (dim,),
        "layers.0.attn.attn_sink": (config["n_heads"],),
        "layers.0.attn.wq_a.weight": (config["q_lora_rank"], dim),
        "layers.0.attn.wq_b.weight": (config["n_heads"] * config["head_dim"], config["q_lora_rank"]),
        "layers.0.attn.wkv.weight": (config["head_dim"], dim),
        "layers.0.attn.wo_a.weight": (
            config["o_groups"] * config["o_lora_rank"],
            config["n_heads"] * config["head_dim"] // config["o_groups"],
        ),
        "layers.0.attn.wo_b.weight": (dim, config["o_groups"] * config["o_lora_rank"]),
        "layers.0.ffn.gate.weight": (config["n_routed_experts"], dim),
        "layers.0.hc_attn_fn": (mix_hc, hc_mult * dim),
        "layers.0.hc_attn_base": (mix_hc,),
        "layers.0.hc_attn_scale": (3,),
    }
    wrong = [f"{n}: {shape_of(n)} != {s}" for n, s in expected.items() if shape_of(n) != s]
    report.check("inventory: the known shapes match the config", not wrong, "; ".join(wrong))

    dtype_wrong = [
        f"{n}: {tensors[n][0]} != {d}"
        for n, d in (
            ("layers.0.attn.attn_sink", "F32"),
            ("layers.0.ffn.gate.weight", "BF16"),
            ("layers.0.hc_attn_fn", "F32"),
            ("embed.weight", "BF16"),
        )
        if n in tensors and tensors[n][0] != d
    ]
    report.check("inventory: the F32/BF16 tensors are not quantized", not dtype_wrong, "; ".join(dtype_wrong))

    # Routed experts: FP4 packed into I8, scales as F8_E8M0 over 32 columns.
    expert_rows = []
    shape_wrong = []
    for layer in range(layers + mtp):
        prefix = f"layers.{layer}." if layer < layers else f"mtp.{layer - layers}."
        count = 0
        while f"{prefix}ffn.experts.{count}.w1.weight" in tensors:
            count += 1
        expert_rows.append(count)
        for index in range(count):
            w1 = tensors[f"{prefix}ffn.experts.{index}.w1.weight"]
            w2 = tensors[f"{prefix}ffn.experts.{index}.w2.weight"]
            s1 = tensors[f"{prefix}ffn.experts.{index}.w1.scale"]
            if w1[:2] != ("I8", (inter, dim // 2)) or w2[:2] != ("I8", (dim, inter // 2)):
                shape_wrong.append(f"{prefix}ffn.experts.{index}: {w1[0]}{list(w1[1])} {w2[0]}{list(w2[1])}")
            if s1[:2] != ("F8_E8M0", (inter, dim // 32)):
                shape_wrong.append(f"{prefix}ffn.experts.{index}.w1.scale: {s1[0]}{list(s1[1])}")
            if len(shape_wrong) > 4:
                break
    report.check(
        "experts: w1/w2/w3 are FP4 packed into I8 with FP4-block-32 E8M0 scales",
        not shape_wrong,
        f"expected I8[{inter},{dim // 2}] and F8_E8M0[{inter},{dim // 32}]; "
        + "; ".join(shape_wrong[:4]),
    )
    report.check(
        f"experts: every backbone layer has {config['n_routed_experts']} routed experts",
        all(count == config["n_routed_experts"] for count in expert_rows[:layers]),
        f"observed {sorted(set(expert_rows[:layers]))}",
    )
    report.check(
        f"experts: every MTP layer has {config['dspark_n_routed_experts']} routed experts",
        all(count == config["dspark_n_routed_experts"] for count in expert_rows[layers:]),
        f"observed {sorted(set(expert_rows[layers:])) if expert_rows[layers:] else 'none'}",
    )


def check_csa2(config: dict, tensors: dict, report: Report) -> None:
    """Compressor and indexer ownership, and the three CSA2 modes they define.

    The model card names three static modes -- Full, Reindex and Reuse. The
    checkpoint expresses them as tensor presence: a Full layer owns both the
    shared compressed KV (`compressor.wkv`) and the indexer's K (`indexer.wk`),
    a Reindex layer owns only the indexer's query side (`indexer.wq_b`,
    `indexer.weights_proj`) and reads K from the source below it, and a Reuse
    layer owns neither. Those three groups must partition the backbone.
    """
    layers = config["n_layers"]
    kv_sources = set(config["kv_source_layers"])
    index_sources = set(config["index_source_layers"])
    ratios = list(config["compress_ratios"])

    present = defaultdict(list)
    for layer in range(layers):
        prefix = f"layers.{layer}."
        for label, suffix in (
            ("compressor.wkv", "attn.compressor.wkv.weight"),
            ("compressor.wgate", "attn.compressor.wgate.weight"),
            ("indexer.wk", "attn.indexer.wk.weight"),
            ("indexer.wq_b", "attn.indexer.wq_b.weight"),
            ("indexer.weights_proj", "attn.indexer.weights_proj.weight"),
        ):
            if prefix + suffix in tensors:
                present[label].append(layer)

    report.check(
        "csa2: compressor.wkv is present exactly on kv_source_layers",
        present["compressor.wkv"] == sorted(kv_sources),
        f"{present['compressor.wkv']} vs {sorted(kv_sources)}",
    )
    # A ratio-1 group is one token wide, so there is nothing to softmax over and no
    # gate; only the compressors that pool more than one token carry one.
    gated = [layer for layer in sorted(kv_sources) if ratios[layer] > 1]
    report.check(
        "csa2: compressor.wgate is present exactly on the ratio>1 KV sources",
        present["compressor.wgate"] == gated,
        f"{present['compressor.wgate']} vs {gated}",
    )
    report.check(
        "csa2: indexer.wk is present exactly on kv_source_layers",
        present["indexer.wk"] == sorted(kv_sources),
        f"{present['indexer.wk']} vs {sorted(kv_sources)}",
    )
    report.check(
        "csa2: indexer.wq_b is present exactly on index_source_layers",
        present["indexer.wq_b"] == sorted(index_sources),
        f"{present['indexer.wq_b']} vs {sorted(index_sources)}",
    )

    full = sorted(set(present["indexer.wk"]) & set(present["indexer.wq_b"]))
    reindex = sorted(set(present["indexer.wq_b"]) - set(present["indexer.wk"]))
    reuse = [layer for layer in range(layers) if layer not in full and layer not in reindex]
    report.check(
        "csa2: Full/Reindex/Reuse partition the backbone",
        sorted(full + reindex + reuse) == list(range(layers)) and not (set(full) & set(reindex)),
        f"Full={full} Reindex={reindex} Reuse={len(reuse)} layers",
    )
    print(f"    CSA2 modes: Full={full}  Reindex={reindex}  Reuse={len(reuse)} layers")


def check_engram_tensors(config: dict, tensors: dict, report: Report, derived_rows: list[int]) -> None:
    layer_ids = list(config["engram_layer_ids"])
    head_dim = config["engram_head_dim"]
    expected_rows = dict(zip(layer_ids, config["engram_num_embeddings"]))

    wrong, found = [], []
    for layer in range(config["n_layers"]):
        prefix = f"layers.{layer}.engram."
        has = prefix + "embed.weight" in tensors
        if has != (layer in layer_ids):
            wrong.append(f"layers.{layer}: engram present={has}, expected={layer in layer_ids}")
        if not has:
            continue
        found.append(layer)
        weight = tensors[prefix + "embed.weight"]
        scale = tensors[prefix + "embed.scale"]
        if weight[0] != "F8_E4M3" or weight[1] != (expected_rows[layer], head_dim):
            wrong.append(f"{prefix}embed.weight: {weight[0]}{list(weight[1])}")
        if scale[0] != "F8_E8M0" or scale[1] != (expected_rows[layer], head_dim // 32):
            wrong.append(f"{prefix}embed.scale: {scale[0]}{list(scale[1])}")
    report.check(
        "engram: the tables sit on exactly engram_layer_ids, F8_E4M3 with E8M0 scales",
        not wrong,
        "; ".join(wrong[:4]),
    )

    per_layer = []
    for layer in found:
        prefix = f"layers.{layer}.engram."
        needed = ("wkv.weight", "q_weight", "k_weight")
        absent = [prefix + name for name in needed if prefix + name not in tensors]
        if absent:
            per_layer.append("; ".join(absent))
        elif tensors[prefix + "q_weight"][1] != (config["hc_mult"], config["dim"]):
            per_layer.append(f"{prefix}q_weight: {list(tensors[prefix + 'q_weight'][1])}")
    report.check("engram: each Engram layer has its gate and value projection", not per_layer, "; ".join(per_layer))

    if found:
        bytes_total = sum(
            tensors[f"layers.{layer}.engram.embed.{part}"][2] for layer in found for part in ("weight", "scale")
        )
        rows = sum(tensors[f"layers.{layer}.engram.embed.weight"][1][0] for layer in found)
        print(
            f"    Engram tables: {len(found)} layers, {rows:,} rows, {bytes_total / 2**30:.2f} GiB "
            f"(derived bucket minimum {derived_rows})"
        )


def check_vision(config: dict, tensors: dict, report: Report) -> None:
    n_layers, dim = config["vision_n_layers"], config["vision_dim"]
    patch = config["vision_patch_size"]
    downsample = config.get("vision_downsample_ratio", 3)

    missing = []
    for block in range(n_layers):
        prefix = f"vision.blocks.{block}."
        for suffix in ("attn.wqkv.weight", "attn.wo.weight", "mlp.w1.weight", "mlp.w2.weight", "norm1.weight", "norm2.weight"):
            if prefix + suffix not in tensors:
                missing.append(prefix + suffix)
    report.check(f"vision: all {n_layers} blocks are present", not missing, "; ".join(missing[:6]))

    expected = {
        "vision.patch_embed.proj.weight": (dim, 3 * patch * patch),
        "vision.blocks.0.attn.wqkv.weight": (3 * dim, dim),
        "aligner.w1.weight": (config["dim"], dim * downsample * downsample),
    }
    wrong = [f"{n}: {tensors[n][1]} != {s}" for n, s in expected.items() if n in tensors and tensors[n][1] != s]
    report.check("vision: the encoder and projector shapes match the config", not wrong, "; ".join(wrong))


def check_mtp(config: dict, tensors: dict, report: Report) -> None:
    mtp = config["n_mtp_layers"]
    markov_rank, dim = config["dspark_markov_rank"], config["dim"]
    vocab = config["vocab_size"]

    expected = {
        "mtp.2.markov_head.embed.weight": (vocab, markov_rank),
        "mtp.2.markov_head.head.weight": (vocab, markov_rank),
        "mtp.2.confidence_head.proj.weight": (1, dim + markov_rank),
    }
    wrong = [f"{n}: {tensors[n][1]} != {s}" for n, s in expected.items() if n in tensors and tensors[n][1] != s]
    report.check("dspark: the Markov and confidence heads match the config", not wrong, "; ".join(wrong))

    # The DSpark block consumes the attention input of its target layers, so its
    # projection is as wide as there are targets.
    main_proj = [k for k in tensors if re.match(r"^mtp\.\d+\.main_proj\.weight$", k)]
    report.check(
        "dspark: main_proj is n_mtp_layers * dim wide",
        all(tensors[k][1] == (dim, mtp * dim) for k in main_proj) and bool(main_proj),
        f"{[(k, list(tensors[k][1])) for k in main_proj]}",
    )
    report.check(
        "dspark: every MTP layer carries attn and ffn but only the last carries the heads",
        all(f"mtp.{i}.attn.wq_a.weight" in tensors and f"mtp.{i}.ffn.gate.weight" in tensors for i in range(mtp)),
        f"mtp layers with the DSpark heads: "
        f"{[i for i in range(mtp) if f'mtp.{i}.markov_head.embed.weight' in tensors]}",
    )


def report_inventory(tensors: dict, per_shard: dict, report: Report) -> None:
    categories: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    dtypes: dict[str, Counter] = defaultdict(Counter)
    for name, (dtype, _shape, span, _shard) in tensors.items():
        label = categorize(name)
        categories[label][0] += 1
        categories[label][1] += span
        dtypes[label][dtype] += 1

    total = sum(spans for _count, spans in categories.values())
    print("\n  Tensor inventory")
    for label, (count, spans) in sorted(categories.items(), key=lambda item: -item[1][1]):
        kinds = " ".join(f"{dtype}x{n}" for dtype, n in sorted(dtypes[label].items()))
        print(f"    {label:16} {count:8,} tensors  {spans / 2**30:9.2f} GiB  {kinds}")
    print(f"    {'TOTAL':16} {len(tensors):8,} tensors  {total / 2**30:9.2f} GiB")

    largest = max(tensors.items(), key=lambda item: item[1][2])
    print(f"    largest tensor: {largest[0]} {largest[1][0]}{list(largest[1][1])} {largest[1][2] / 2**30:.2f} GiB")
    report.check("inventory: the tensor count and byte total are non-zero", len(tensors) > 0 and total > 0)

    print("\n  Shards")
    for shard, entries in sorted(per_shard.items()):
        spans = sum(entry["data_offsets"][1] - entry["data_offsets"][0] for entry in entries.values())
        print(f"    {shard:24} {len(entries):7,} tensors  {spans / 2**30:9.2f} GiB")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True, help="directory holding the shards or header prefixes")
    parser.add_argument(
        "--config",
        default=None,
        help="config JSON in either released shape; defaults to <checkpoint-dir>/config.json, "
        "then to <checkpoint-dir>/inference/config.json",
    )
    parser.add_argument(
        "--header-prefix",
        action="store_true",
        help="assert that the shards are header-only prefixes, not complete files",
    )
    parser.add_argument("--expect-fp8-block", type=int, nargs=2, default=(32, 32), metavar=("OUT", "IN"))
    parser.add_argument("--json", default=None, help="write the machine-readable result here")
    parser.add_argument("--list-tensors", default=None, help="print the tensors matching this regex and exit")
    return parser.parse_args(argv)


def resolve_config(checkpoint_dir: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    for name in ("config.json", "inference/config.json", "inference_config.json"):
        candidate = os.path.join(checkpoint_dir, name)
        if os.path.exists(candidate):
            return candidate
    raise SystemExit(f"no config found: pass --config, or place config.json in {checkpoint_dir}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = shard_paths(args.checkpoint_dir)
    if not paths:
        print(f"no safetensors shards or header prefixes found in {args.checkpoint_dir}", file=sys.stderr)
        return 2

    tensors, per_shard, complete, duplicates = load_inventory(paths)

    if args.list_tensors:
        pattern = re.compile(args.list_tensors)
        for name in sorted(tensors):
            if pattern.search(name):
                dtype, shape, span, shard = tensors[name]
                print(f"{name}\t{dtype}\t{list(shape)}\t{span}\t{shard}")
        return 0

    config = load_config(resolve_config(args.checkpoint_dir, args.config))
    report = Report()

    print(f"DeepSeek-V4.1-Flash header audit: {len(per_shard)} shards in {args.checkpoint_dir}")
    for duplicate in duplicates:
        print(f"  skipped duplicate shard: {duplicate}")
    print(f"  mode: {'header-only prefixes' if not complete else 'complete shards'}")
    if args.header_prefix and complete:
        print("  warning: --header-prefix was given but the shards look complete", file=sys.stderr)
    if not complete and not args.header_prefix:
        print("  note: payloads are absent, so only the headers were read")

    check_config(config, report)
    derived_rows = check_engram_tables(config, report)
    check_packing(tensors, report)
    check_scale_pairs(tensors, report, tuple(args.expect_fp8_block))
    check_backbone(config, tensors, report)
    check_csa2(config, tensors, report)
    check_engram_tensors(config, tensors, report, derived_rows)
    check_vision(config, tensors, report)
    check_mtp(config, tensors, report)
    report_inventory(tensors, per_shard, report)

    failures = report.failures
    print(f"\n  {len(report.checks) - len(failures)}/{len(report.checks)} checks passed")
    for name, detail in failures:
        print(f"    [FAIL] {name}")
        if detail:
            print(f"           {detail}")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "checkpoint_dir": args.checkpoint_dir,
                    "shards": len(paths),
                    "complete": complete,
                    "tensors": len(tensors),
                    "checks": [{"name": n, "passed": ok, "detail": d} for n, ok, d in report.checks],
                },
                handle,
                indent=2,
            )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
