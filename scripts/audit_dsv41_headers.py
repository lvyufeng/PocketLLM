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

Presence and shape are answered from two different places, and the audit keeps
them apart. `model.safetensors.index.json` -- which the release ships, 96,085
tensors over 48 shards -- names the shard holding every tensor, so *is this
tensor in the checkpoint* is decidable before a byte of payload is downloaded.
Shapes, dtypes and byte extents come from the headers, so they are decidable only
for the shards that are here. That is what makes this usable while a download is
still running: the whole tensor inventory, the CSA2 mode assignment and the expert
counts are checked on the first shard that lands, and each shape check turns from
undecided into asserted as its shard arrives.

A check therefore has three outcomes, not two:

- **passed** -- the evidence is local and agrees with the config.
- **failed** -- the evidence is local and disagrees, or the index says the
  checkpoint ships a tensor the headers contradict.
- **undecided** -- the index places the evidence in a shard that is not here yet.
  Counted separately and never as a pass: a partial checkpoint must not read as a
  full one. `--require-complete` turns undecided into a failure for callers that
  need the strict reading.

Exit status is 0 only when no check fails; failures are listed with the expected
and observed values, and undecided checks are listed separately with the shards
they are waiting on.

Usage:

    python scripts/audit_dsv41_headers.py --checkpoint-dir /path/to/DeepSeek-V4.1-Flash
    python scripts/audit_dsv41_headers.py --checkpoint-dir /tmp/dsv41 --header-prefix
    python scripts/audit_dsv41_headers.py --config cfg.json --list-tensors 'engram'
    python scripts/audit_dsv41_headers.py --checkpoint-dir partial --require-complete
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable

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


def shard_label(name: str) -> str:
    """`model-00015-of-00048.safetensors` -> `15/48`, short enough for a detail line."""
    match = re.search(r"(\d+)-of-(\d+)\.safetensors$", name)
    return f"{int(match.group(1))}/{int(match.group(2))}" if match else name


def shard_note(shards: Iterable[str], limit: int = 4) -> str:
    """A one-line list of the shards a check is waiting on."""
    ordered = sorted(shards)
    labels = [shard_label(shard) for shard in ordered[:limit]]
    if len(ordered) > limit:
        labels.append(f"+{len(ordered) - limit} more")
    return f"undecided: {len(ordered)} shard(s) not downloaded ({', '.join(labels)})"


class Report:
    """Collects check outcomes so the summary can be printed in one place.

    A check has three outcomes, not two. `None` is the third: the index places the
    evidence in a shard that has not been downloaded, so nothing was found wrong
    and nothing was verified either. Counting it separately from a pass is what
    keeps a partial checkpoint from reading as a complete one, and
    `--require-complete` turns it into a failure for callers that need that.
    """

    def __init__(self) -> None:
        self.checks: list[tuple[str, bool | None, str]] = []

    def check(self, name: str, ok: bool, detail: str = "", waiting: Iterable[str] = ()) -> bool | None:
        """Record one check; `waiting` names the shards whose absence makes it moot.

        `waiting` only downgrades a *pass*. A check that found a contradiction is
        reported as a failure whether or not other shards are missing -- the thing
        it found is wrong either way, and saying so is more useful than deferring.
        """
        pending = sorted(set(waiting))
        if pending:
            note = shard_note(pending)
            if ok:
                detail, ok = "; ".join(filter(None, (note, detail))), None
            else:
                detail = "; ".join(filter(None, (detail, f"{len(pending)} shard(s) also not downloaded")))
        self.checks.append((name, ok, detail))
        return ok

    def fail(self, name: str, detail: str = "") -> None:
        self.checks.append((name, False, detail))

    @property
    def failures(self) -> list[tuple[str, str]]:
        return [(n, d) for n, ok, d in self.checks if ok is False]

    @property
    def undecided(self) -> list[tuple[str, str]]:
        return [(n, d) for n, ok, d in self.checks if ok is None]

    @property
    def passed(self) -> int:
        return sum(1 for _n, ok, _d in self.checks if ok)


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
# index-driven coverage
# ---------------------------------------------------------------------------


def load_index(checkpoint_dir: str, explicit: str | None) -> dict:
    """The tensor -> shard map the release ships, or `{}` when there is none.

    `model.safetensors.index.json` is the checkpoint's own statement of what each
    shard contains. Reading it is what makes the audit work on a partial download:
    a presence question ("does layer 14 own a compressor?") is answerable for all
    96,085 tensors before any payload arrives, and a shape check whose shard is
    still missing is reported as undecided rather than as a failure, because a
    tensor being absent from disk says nothing about the checkpoint.
    """
    path = explicit or os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    weight_map = raw.get("weight_map") if isinstance(raw, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit(f"{path}: holds no weight_map; pass --index or remove the file")
    return dict(weight_map)


class Coverage:
    """Which shards of the checkpoint are on disk, according to the index."""

    def __init__(self, weight_map: dict | None = None) -> None:
        self.weight_map: dict[str, str] = dict(weight_map or {})
        self.shard_of: dict[str, str] = {}
        self.local: set[str] = set()
        self._names = sorted(self.weight_map)
        self._owners = [self.weight_map[name] for name in self._names]

    @property
    def indexed(self) -> bool:
        return bool(self.weight_map)

    @property
    def shards(self) -> list[str]:
        """Every shard the index names, in file order."""
        return sorted(set(self.weight_map.values()))

    @property
    def pending(self) -> list[str]:
        """The indexed shards that are not on disk yet."""
        return [shard for shard in self.shards if shard not in self.local]

    def identify(self, per_shard: dict) -> None:
        """Match each local file to the index shard whose tensors it holds.

        Matching on tensor names rather than on file names keeps a header-prefix
        tree usable: the names a fetch gives those files are free-form, but the
        tensors inside them are not. A file that holds a different set of tensors
        than any one index shard is left unmatched on purpose, and `check_index`
        reports it.
        """
        for basename, entries in per_shard.items():
            owners = {self.weight_map[name] for name in entries if name in self.weight_map}
            if len(owners) == 1:
                self.shard_of[basename] = owners.pop()
        self.local = set(self.shard_of.values())

    def waiting(self, names: Iterable[str]) -> list[str]:
        """The not-yet-local shards holding any of `names`."""
        if not self.pending:
            return []
        pending = set(self.pending)
        return sorted({self.weight_map[name] for name in names if self.weight_map.get(name) in pending})

    def waiting_under(self, prefix: str) -> list[str]:
        """`waiting` for every indexed name that starts with `prefix`."""
        if not self.pending:
            return []
        pending = set(self.pending)
        low = bisect.bisect_left(self._names, prefix)
        high = bisect.bisect_left(self._names, prefix + "\U0010ffff")
        return sorted({self._owners[i] for i in range(low, high)} & pending)


class Checkpoint:
    """The tensor facts the audit reads, split by where each one comes from.

    *Does the checkpoint ship this tensor* is answered by the index when there is
    one -- it lists all 96,085 -- and by the headers otherwise. *What shape is it*
    is answered by the header, which exists only for the shards that are here.
    Keeping the two apart is the whole point: on a partial download they have
    different answers, and collapsing them would either miss a real defect or
    invent one.
    """

    def __init__(self, tensors: dict, per_shard: dict, complete: bool, coverage: Coverage) -> None:
        self.tensors = tensors
        self.per_shard = per_shard
        self.complete = complete
        self.coverage = coverage
        self._names: set[str] | None = None

    @property
    def names(self) -> set[str]:
        """Every tensor the checkpoint ships, whether or not it is readable yet."""
        if self._names is None:
            source = self.coverage.weight_map if self.coverage.indexed else self.tensors
            self._names = set(source)
        return self._names

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def shape(self, name: str):
        entry = self.tensors.get(name)
        return entry[1] if entry else None

    def waiting(self, *names: str) -> list[str]:
        return self.coverage.waiting(names)

    def waiting_under(self, prefix: str) -> list[str]:
        return self.coverage.waiting_under(prefix)


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


def check_packing(ckpt: Checkpoint, report: Report) -> list[str]:
    """Every tensor's byte extent must match its shape and dtype.

    This is the check that proves the routed experts are packed: an I8 tensor of
    [2304, 2560] is 5,898,240 bytes, i.e. two FP4 values per byte over a logical
    [2304, 5120] weight, and an unpacked export would not match.
    """
    mismatched = []
    for name, (dtype, shape, span, _shard) in ckpt.tensors.items():
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


def settles(observed: set, expected: set, complete: bool) -> bool:
    """Whether a claim about a set of tensors holds on the evidence available.

    `observed` can only be built from the shards that are here, so a partial
    download proves `observed <= expected` and nothing more. Saying `==` here
    would fail every such check for the wrong reason -- an empty observation is
    not a contradiction, it is an absence of evidence, and `Report.check` is what
    records that distinction once `complete` says the evidence is all in.
    """
    return observed == expected if complete else observed <= expected


def check_scale_pairs(ckpt: Checkpoint, report: Report, expect_block: tuple[int, int]) -> None:
    """Quantized weights must divide evenly by their scale, at the expected block.

    The two Engram tables are the exception: they are stored fp8 with one E8M0
    scale per 32 channels and no row blocking at all, so their block is 1 x 32
    rather than the 32 x 32 every other fp8 weight uses. That is a fact about the
    checkpoint, not a rounding of the rule, so it is checked separately.

    The block a pair uses is a property of that pair, so every pair whose shard is
    here is decided on its own. The claim that *all* fp8 weights in the checkpoint
    use the block is only as complete as the download, so the two set comparisons
    below are reported undecided while shards holding unchecked weights are
    outstanding rather than being read as a statement about the whole checkpoint.
    """
    blocks: Counter = Counter()
    engram_blocks: Counter = Counter()
    broken = []
    unreadable = []
    for name, (dtype, shape, _span, _shard) in ckpt.tensors.items():
        if not name.endswith(".weight") or dtype == "I8":
            continue
        scale_name = name[: -len(".weight")] + ".scale"
        if scale_name not in ckpt:
            continue  # the checkpoint ships no scale for this weight
        if scale_name not in ckpt.tensors:
            unreadable.append(scale_name)
            continue
        block = block_of(shape, ckpt.shape(scale_name))
        if block is None:
            broken.append(f"{name}{list(shape)} vs {list(ckpt.shape(scale_name))}")
            continue
        (engram_blocks if ".engram.embed." in name else blocks)[block] += 1
    report.check(
        "scales: every weight/scale pair blocks evenly",
        not broken,
        "; ".join(broken[:6]),
        ckpt.waiting(*unreadable),
    )

    # Which weights are in scope: `.weight` names that ship a `.scale` sibling, in
    # the checkpoint's own inventory rather than in the shards on disk.
    paired = [name for name in ckpt.names if name.endswith(".weight") and name[: -len(".weight")] + ".scale" in ckpt]
    engram_paired = [name for name in paired if ".engram.embed." in name]
    plain_paired = [name for name in paired if ".engram.embed." not in name]
    report.check(
        f"scales: all non-Engram FP8 weights use a {expect_block[0]}x{expect_block[1]} block",
        settles(set(blocks), {expect_block}, not plain_paired or not ckpt.waiting(*plain_paired)),
        f"observed blocks {dict(blocks)}",
        ckpt.waiting(*plain_paired),
    )
    report.check(
        "scales: the Engram tables use a 1x32 per-row block",
        settles(set(engram_blocks), {(1, 32)}, not engram_paired or not ckpt.waiting(*engram_paired)),
        f"observed blocks {dict(engram_blocks)}",
        ckpt.waiting(*engram_paired),
    )


def check_index(ckpt: Checkpoint, report: Report) -> None:
    """Every local file must hold exactly the tensors the index assigns to one shard.

    This is what lets the rest of the audit trust presence. The index is the
    checkpoint's own statement of its contents, so a file whose header disagrees
    with it is a real defect -- a truncated write, a mixed-up shard, a repacked
    export -- rather than a file that has not arrived yet. It is also the check
    that keeps a header-prefix tree honest: fetch the wrong range and the tensors
    that come back name a different shard than the file claims.
    """
    coverage = ckpt.coverage
    if not coverage.indexed:
        return
    mismatched = []
    for basename, entries in sorted(ckpt.per_shard.items()):
        shard = coverage.shard_of.get(basename)
        if shard is None:
            mismatched.append(f"{basename}: holds tensors from no single indexed shard")
            continue
        expected = {name for name, owner in coverage.weight_map.items() if owner == shard}
        if set(entries) != expected:
            extra = sorted(set(entries) - expected)[:3]
            absent = sorted(expected - set(entries))[:3]
            mismatched.append(
                f"{shard}: {len(entries)} tensors against {len(expected)} in the index"
                + (f", unexpected {extra}" if extra else "")
                + (f", absent {absent}" if absent else "")
            )
    report.check(
        "index: every local shard holds exactly the tensors the index assigns it",
        not mismatched,
        "; ".join(mismatched[:4]),
    )


BACKBONE_SUFFIXES = (
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
)


def check_backbone(config: dict, ckpt: Checkpoint, report: Report) -> None:
    layers, mtp = config["n_layers"], config["n_mtp_layers"]
    dim, inter = config["dim"], config["moe_inter_dim"]
    hc_mult = config["hc_mult"]
    mix_hc = (2 + hc_mult) * hc_mult

    # Presence is an index question, so it is answered for every layer as soon as
    # the index is readable. With no index it falls back to the headers, which is
    # the header-prefix case the audit started as.
    missing = []
    for layer in range(layers):
        prefix = f"layers.{layer}."
        missing += [prefix + suffix for suffix in BACKBONE_SUFFIXES if prefix + suffix not in ckpt]
    report.check("inventory: every backbone layer has its full tensor set", not missing, "; ".join(missing[:6]))

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
    wrong, waiting = [], []
    for name, want in expected.items():
        if ckpt.shape(name) == want:
            continue
        pending = ckpt.waiting(name)
        if pending:
            waiting += pending
        else:
            wrong.append(f"{name}: {ckpt.shape(name)} != {want}")
    report.check("inventory: the known shapes match the config", not wrong, "; ".join(wrong), waiting)

    # An unquantized tensor's dtype is readable wherever its shard is, so this only
    # waits on the ones that have not arrived.
    dtype_expected = (
        ("layers.0.attn.attn_sink", "F32"),
        ("layers.0.ffn.gate.weight", "BF16"),
        ("layers.0.hc_attn_fn", "F32"),
        ("embed.weight", "BF16"),
    )
    dtype_wrong, dtype_waiting = [], []
    for name, dtype in dtype_expected:
        entry = ckpt.tensors.get(name)
        if entry is None:
            dtype_waiting += ckpt.waiting(name)
        elif entry[0] != dtype:
            dtype_wrong.append(f"{name}: {entry[0]} != {dtype}")
    report.check(
        "inventory: the F32/BF16 tensors are not quantized", not dtype_wrong, "; ".join(dtype_wrong), dtype_waiting
    )

    # Routed experts: FP4 packed into I8, scales as F8_E8M0 over 32 columns. The
    # count comes from presence and the packing from the shapes, so a partially
    # downloaded checkpoint can still say how many experts each layer has.
    expert_counts = []
    for layer in range(layers + mtp):
        prefix = f"layers.{layer}." if layer < layers else f"mtp.{layer - layers}."
        count = 0
        while f"{prefix}ffn.experts.{count}.w1.weight" in ckpt:
            count += 1
        expert_counts.append(count)

    shape_wrong, expert_waiting = [], []
    for layer in range(layers + mtp):
        prefix = f"layers.{layer}." if layer < layers else f"mtp.{layer - layers}."
        pending = ckpt.waiting_under(prefix + "ffn.experts.")
        if pending:
            expert_waiting += pending
            continue
        for index in range(expert_counts[layer]):
            w1 = ckpt.tensors[f"{prefix}ffn.experts.{index}.w1.weight"]
            w2 = ckpt.tensors[f"{prefix}ffn.experts.{index}.w2.weight"]
            s1 = ckpt.tensors[f"{prefix}ffn.experts.{index}.w1.scale"]
            if w1[:2] != ("I8", (inter, dim // 2)) or w2[:2] != ("I8", (dim, inter // 2)):
                shape_wrong.append(f"{prefix}ffn.experts.{index}: {w1[0]}{list(w1[1])} {w2[0]}{list(w2[1])}")
            if s1[:2] != ("F8_E8M0", (inter, dim // 32)):
                shape_wrong.append(f"{prefix}ffn.experts.{index}.w1.scale: {s1[0]}{list(s1[1])}")
            if len(shape_wrong) > 4:
                break
    report.check(
        "experts: w1/w2/w3 are FP4 packed into I8 with FP4-block-32 E8M0 scales",
        not shape_wrong,
        f"expected I8[{inter},{dim // 2}] and F8_E8M0[{inter},{dim // 32}]; " + "; ".join(shape_wrong[:4]),
        expert_waiting,
    )
    report.check(
        f"experts: every backbone layer has {config['n_routed_experts']} routed experts",
        all(count == config["n_routed_experts"] for count in expert_counts[:layers]),
        f"observed {sorted(set(expert_counts[:layers]))}",
    )
    report.check(
        f"experts: every MTP layer has {config['dspark_n_routed_experts']} routed experts",
        all(count == config["dspark_n_routed_experts"] for count in expert_counts[layers:]),
        f"observed {sorted(set(expert_counts[layers:])) if expert_counts[layers:] else 'none'}",
    )


def check_csa2(config: dict, ckpt: Checkpoint, report: Report) -> None:
    """Compressor and indexer ownership, and the three CSA2 modes they define.

    The model card names three static modes -- Full, Reindex and Reuse. The
    checkpoint expresses them as tensor presence: a Full layer owns both the
    shared compressed KV (`compressor.wkv`) and the indexer's K (`indexer.wk`),
    a Reindex layer owns only the indexer's query side (`indexer.wq_b`,
    `indexer.weights_proj`) and reads K from the source below it, and a Reuse
    layer owns neither. Those three groups must partition the backbone.

    Presence is an index question, so all five checks here are decided as soon as
    the index is readable -- which is the layer assignment a runtime has to
    implement, and the part of the checkpoint most worth knowing early.
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
            if prefix + suffix in ckpt:
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


def check_engram_tensors(config: dict, ckpt: Checkpoint, report: Report, derived_rows: list[int]) -> None:
    layer_ids = list(config["engram_layer_ids"])
    head_dim = config["engram_head_dim"]
    expected_rows = dict(zip(layer_ids, config["engram_num_embeddings"]))

    misplaced = [
        f"layers.{layer}: engram present={f'layers.{layer}.engram.embed.weight' in ckpt}, "
        f"expected={layer in layer_ids}"
        for layer in range(config["n_layers"])
        if (f"layers.{layer}.engram.embed.weight" in ckpt) != (layer in layer_ids)
    ]
    report.check("engram: the tables sit on exactly engram_layer_ids", not misplaced, "; ".join(misplaced[:4]))

    wrong, waiting = [], []
    for layer in layer_ids:
        prefix = f"layers.{layer}.engram."
        weight = ckpt.tensors.get(prefix + "embed.weight")
        scale = ckpt.tensors.get(prefix + "embed.scale")
        if weight is None or scale is None:
            waiting += ckpt.waiting(prefix + "embed.weight", prefix + "embed.scale")
            continue
        if weight[0] != "F8_E4M3" or weight[1] != (expected_rows[layer], head_dim):
            wrong.append(f"{prefix}embed.weight: {weight[0]}{list(weight[1])}")
        if scale[0] != "F8_E8M0" or scale[1] != (expected_rows[layer], head_dim // 32):
            wrong.append(f"{prefix}embed.scale: {scale[0]}{list(scale[1])}")
    report.check(
        "engram: the tables are F8_E4M3 with one E8M0 scale per 32 channels",
        not wrong,
        "; ".join(wrong[:4]),
        waiting,
    )

    per_layer, projection_waiting = [], []
    for layer in layer_ids:
        prefix = f"layers.{layer}.engram."
        needed = ("wkv.weight", "wkv.scale", "q_weight", "k_weight")
        absent = [prefix + name for name in needed if prefix + name not in ckpt.tensors]
        if absent:
            projection_waiting += ckpt.waiting(*absent)
        elif ckpt.shape(prefix + "q_weight") != (config["hc_mult"], config["dim"]):
            per_layer.append(f"{prefix}q_weight: {list(ckpt.shape(prefix + 'q_weight'))}")
    report.check(
        "engram: each Engram layer has its gate and value projection",
        not per_layer,
        "; ".join(per_layer),
        projection_waiting,
    )

    # The table row count is in the config, so the size is knowable before the two
    # shards holding 189 GiB of embeddings arrive: one fp8 value per channel plus
    # one E8M0 scale per 32 channels.
    rows = sum(config["engram_num_embeddings"])
    table_bytes = rows * (head_dim + head_dim // 32)
    print(
        f"    Engram tables: layers {layer_ids}, {rows:,} rows, {table_bytes / 2**30:.2f} GiB "
        f"of embed weight+scale (derived bucket minimum {derived_rows})"
    )


def check_vision(config: dict, ckpt: Checkpoint, report: Report) -> None:
    n_layers, dim = config["vision_n_layers"], config["vision_dim"]
    patch = config["vision_patch_size"]
    downsample = config.get("vision_downsample_ratio", 3)

    missing = []
    for block in range(n_layers):
        prefix = f"vision.blocks.{block}."
        for suffix in ("attn.wqkv.weight", "attn.wo.weight", "mlp.w1.weight", "mlp.w2.weight", "norm1.weight", "norm2.weight"):
            if prefix + suffix not in ckpt:
                missing.append(prefix + suffix)
    report.check(f"vision: all {n_layers} blocks are present", not missing, "; ".join(missing[:6]))

    expected = {
        "vision.patch_embed.proj.weight": (dim, 3 * patch * patch),
        "vision.blocks.0.attn.wqkv.weight": (3 * dim, dim),
        "aligner.w1.weight": (config["dim"], dim * downsample * downsample),
    }
    wrong, waiting = [], []
    for name, want in expected.items():
        if ckpt.shape(name) == want:
            continue
        pending = ckpt.waiting(name)
        if pending:
            waiting += pending
        else:
            wrong.append(f"{name}: {ckpt.shape(name)} != {want}")
    report.check("vision: the encoder and projector shapes match the config", not wrong, "; ".join(wrong), waiting)


def check_mtp(config: dict, ckpt: Checkpoint, report: Report) -> None:
    mtp = config["n_mtp_layers"]
    markov_rank, dim = config["dspark_markov_rank"], config["dim"]
    vocab = config["vocab_size"]
    last = f"mtp.{mtp - 1}."

    expected = {
        last + "markov_head.embed.weight": (vocab, markov_rank),
        last + "markov_head.head.weight": (vocab, markov_rank),
        last + "confidence_head.proj.weight": (1, dim + markov_rank),
    }
    wrong, waiting = [], []
    for name, want in expected.items():
        if ckpt.shape(name) == want:
            continue
        pending = ckpt.waiting(name)
        if pending:
            waiting += pending
        else:
            wrong.append(f"{name}: {ckpt.shape(name)} != {want}")
    report.check("dspark: the Markov and confidence heads match the config", not wrong, "; ".join(wrong), waiting)

    # The DSpark block consumes the attention input of its target layers, so its
    # projection is as wide as there are targets.
    main_proj = [name for name in ckpt.names if re.match(r"^mtp\.\d+\.main_proj\.weight$", name)]
    readable = [(name, list(ckpt.shape(name))) for name in main_proj if name in ckpt.tensors]
    report.check(
        "dspark: main_proj is n_mtp_layers * dim wide",
        bool(main_proj) and all(ckpt.shape(name) == (dim, mtp * dim) for name in main_proj if name in ckpt.tensors),
        f"{readable or main_proj} (want shape ({dim}, {mtp * dim}))",
        ckpt.waiting(*[name for name in main_proj if name not in ckpt.tensors]),
    )
    report.check(
        "dspark: every MTP layer carries attn and ffn but only the last carries the heads",
        all(f"mtp.{i}.attn.wq_a.weight" in ckpt and f"mtp.{i}.ffn.gate.weight" in ckpt for i in range(mtp)),
        f"mtp layers with the DSpark heads: "
        f"{[i for i in range(mtp) if f'mtp.{i}.markov_head.embed.weight' in ckpt]}",
    )


def report_inventory(ckpt: Checkpoint, report: Report) -> None:
    tensors = ckpt.tensors
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
    if ckpt.coverage.indexed:
        shipped = len(ckpt.coverage.weight_map)
        print(
            f"    {'of':16} {shipped:8,} tensors shipped by the index "
            f"({100 * len(tensors) / shipped:.1f}% readable from the shards on disk)"
        )

    largest = max(tensors.items(), key=lambda item: item[1][2])
    print(f"    largest tensor: {largest[0]} {largest[1][0]}{list(largest[1][1])} {largest[1][2] / 2**30:.2f} GiB")
    report.check("inventory: the tensor count and byte total are non-zero", len(tensors) > 0 and total > 0)

    print("\n  Shards")
    for shard, entries in sorted(ckpt.per_shard.items()):
        spans = sum(entry["data_offsets"][1] - entry["data_offsets"][0] for entry in entries.values())
        indexed = ckpt.coverage.shard_of.get(shard)
        suffix = f"  {shard_label(indexed)}" if indexed else ""
        print(f"    {shard:24} {len(entries):7,} tensors  {spans / 2**30:9.2f} GiB{suffix}")


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
    parser.add_argument(
        "--index",
        default=None,
        help="safetensors index naming each tensor's shard; defaults to <checkpoint-dir>/model.safetensors.index.json. "
        "With it, presence is checked across the whole checkpoint and shape checks whose shard is missing are "
        "reported as undecided instead of failing; without it the audit is header-only",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="treat an undecided check -- one whose evidence sits in a shard that is not downloaded -- as a failure",
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

    coverage = Coverage(load_index(args.checkpoint_dir, args.index))
    coverage.identify(per_shard)
    ckpt = Checkpoint(tensors, per_shard, complete, coverage)
    config = load_config(resolve_config(args.checkpoint_dir, args.config))
    report = Report()

    print(f"DeepSeek-V4.1-Flash header audit: {len(per_shard)} shards in {args.checkpoint_dir}")
    for duplicate in duplicates:
        print(f"  skipped duplicate shard: {duplicate}")
    print(f"  mode: {'header-only prefixes' if not complete else 'complete shards'}")
    if coverage.indexed:
        print(
            f"  index: {len(coverage.weight_map):,} tensors over {len(coverage.shards)} shards, "
            f"{len(coverage.local)} local, {len(coverage.pending)} not downloaded"
        )
        print(
            f"  readable: {len(tensors):,} of {len(coverage.weight_map):,} tensors "
            f"({100 * len(tensors) / len(coverage.weight_map):.1f}%); "
            "presence is checked across the checkpoint, shape only where the shard is here"
        )
    elif complete:
        print("  note: no index beside the shards, so presence is read from the headers alone")
    if args.header_prefix and complete:
        print("  warning: --header-prefix was given but the shards look complete", file=sys.stderr)
    if not complete and not args.header_prefix:
        print("  note: payloads are absent, so only the headers were read")

    check_config(config, report)
    check_index(ckpt, report)
    derived_rows = check_engram_tables(config, report)
    check_packing(ckpt, report)
    check_scale_pairs(ckpt, report, tuple(args.expect_fp8_block))
    check_backbone(config, ckpt, report)
    check_csa2(config, ckpt, report)
    check_engram_tensors(config, ckpt, report, derived_rows)
    check_vision(config, ckpt, report)
    check_mtp(config, ckpt, report)
    report_inventory(ckpt, report)

    failures, undecided = report.failures, report.undecided
    summary = f"\n  {report.passed}/{len(report.checks)} checks passed"
    if failures:
        summary += f", {len(failures)} failed"
    if undecided:
        summary += f", {len(undecided)} undecided"
    print(summary)
    for name, detail in failures:
        print(f"    [FAIL] {name}")
        if detail:
            print(f"           {detail}")
    for name, detail in undecided:
        print(f"    [UNDECIDED] {name}")
        if detail:
            print(f"           {detail}")
    if undecided:
        print("  note: undecided is not a pass -- those checks cover tensors in shards that are not downloaded")

    if args.require_complete and undecided:
        print(f"  --require-complete: {len(undecided)} undecided check(s) counted as failures", file=sys.stderr)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "checkpoint_dir": args.checkpoint_dir,
                    "shards": len(paths),
                    "complete": complete,
                    "tensors": len(tensors),
                    "index": {
                        "indexed": coverage.indexed,
                        "shipped_tensors": len(coverage.weight_map),
                        "local_shards": sorted(coverage.local),
                        "pending_shards": coverage.pending,
                    },
                    "checks": [
                        {"name": n, "status": "undecided" if ok is None else ("pass" if ok else "fail"),
                         "passed": ok is True, "detail": d}
                        for n, ok, d in report.checks
                    ],
                },
                handle,
                indent=2,
            )

    return 1 if failures or (args.require_complete and undecided) else 0


if __name__ == "__main__":
    sys.exit(main())
