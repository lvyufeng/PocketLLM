"""Engram conditional memory for DeepSeek-V4.1: the tokenizer-side front end.

DeepSeek-V4.1-Flash ships two Engram hash tables totalling 189.13 GiB of FP8 --
196B embedding parameters, 39.8% of the checkpoint, the largest single thing in
it. Storing those weights is not the hard part. The tables are addressed by
n-gram hash ids that are computed from the *tokenizer* and appear nowhere in the
checkpoint, so a loader holding all 189 GiB still has no way to name a row.
This module is that missing front end, reproduced from the released
`inference_engram.py` and checked against the checkpoint's own declared
metadata rather than against a running model:

* `EngramLayout` re-derives the per-(layer, n-gram size, head) bucket ranges.
  The reference hands out consecutive primes above `engram_vocab_size` without
  reuse, so a layer's table needs exactly `sum(primes)` rows. Both V4.1 Engram
  layers reproduce the declared `engram_num_embeddings` with difference 0 --
  384,006,168 and 384,016,682 -- which is the closed loop
  `tests/test_encoding_engram.py` asserts.
* `NgramHasher` maps token ids onto those rows, and its largest possible id is
  `sum(primes) - 1`. That is what makes the row count in the safetensors header
  the whole address space rather than a lower bound: every declared row is
  reachable and no reachable id falls outside the table.
* `build_compressed_token_map` collapses tokens that normalize alike onto one id,
  so " The", "the" and "THE" hash the same way. Its output size is not merely the
  size of `token_map`: `compute_hash_multipliers` derives every multiplier from
  it, so a mismatch there rehashes all 189 GiB silently. The reference gates this
  with an `assert`; `NgramHasher` takes the same value as
  `expected_token_map_size` so the check happens in the same place in the
  pipeline.

Nothing here needs a V4.1 checkpoint, and nothing here downloads anything: the
config and safetensors headers are enough for the layout, and the compressed
token map needs only a tokenizer directory.

Stdlib only, with two lazy exceptions. The multipliers are drawn from numpy's
PCG64 stream because that is the stream the reference draws them from, and
re-deriving PCG64 in pure Python would add a second source of truth rather than
a check -- `tests/test_encoding_engram.py` pins the resulting values instead. The
compressed token map needs `tokenizers`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# The private-use sentinel the reference substitutes for a lone space. Without it
# a token that is exactly " " normalizes to "" under Strip() and merges with every
# other token that normalizes to nothing. Escaped rather than literal so the
# character stays visible in review.
_SENTINEL = "\ue000"
_BLANK_RUN = r"[ \t\r\n]+"
_LONE_SPACE = r"^ $"

# Deterministic Miller-Rabin. These bases decide every n below
# 3,317,044,064,679,887,385,961,981, and the primes here are just above 1.6e7, so
# the answer is decided rather than probabilistic. The reference calls
# `sympy.isprime`; pulling sympy in to answer a 1.6e7 primality question would be
# a large dependency for a cheap answer.
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n: int) -> bool:
    """Deterministic primality for every n < 3.3e24, which covers this module's use."""
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen: set[int]) -> int:
    """The smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables.

    A position is hashed as `max_ngram_size - 1` n-grams (2-gram through
    max_ngram_size-gram), each split over `n_heads` heads. Every (n-gram size,
    head) pair owns its own prime-sized bucket range in the layer's table; the
    primes are drawn in order and never reused, which is what keeps the ranges
    disjoint, and drawn in ascending order within a layer, which is what makes
    the last bucket the one holding the highest id.

    `primes` is indexed `[layer][n-gram index][head]`, matching the reference's
    `primes[:, i - 1]` lookup once its buffer is flattened.
    """

    layer_ids: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    head_dim: int
    num_embeddings: tuple[int, ...]  # table rows declared by the config, per layer
    primes: tuple[tuple[tuple[int, ...], ...], ...]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EngramLayout | None":
        """Derive the layout a V4.1 config implies, or None if it declares no Engram layers."""
        layer_ids = tuple(int(v) for v in config["engram_layer_ids"])
        if not layer_ids:
            return None
        max_ngram_size = int(config["engram_max_ngram_size"])
        n_heads = int(config["engram_n_heads"])
        # `seen` is carried across layers as well as within one, so no prime is ever
        # shared between the two tables. The search starts one below the vocabulary
        # size so that the first prime returned is above it.
        primes: list[tuple[tuple[int, ...], ...]] = []
        seen: set[int] = set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], int(config["engram_vocab_size"]) - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            layer_ids=layer_ids,
            max_ngram_size=max_ngram_size,
            n_heads=n_heads,
            head_dim=int(config["engram_head_dim"]),
            num_embeddings=tuple(int(v) for v in config["engram_num_embeddings"]),
            primes=tuple(primes),
        )

    @property
    def n_ngram_sizes(self) -> int:
        """2-gram through max_ngram_size-gram."""
        return self.max_ngram_size - 1

    @property
    def n_hash_columns(self) -> int:
        """Hash ids produced per position: one bucket range per (n-gram size, head)."""
        return self.n_ngram_sizes * self.n_heads

    def flat_primes(self, layer: int) -> tuple[int, ...]:
        """The layer's bucket sizes in the order the hash columns use them.

        Column `c` is `(n-gram index = c // n_heads, head = c % n_heads)`: the
        reference flattens `[n-gram][head]` for the offsets but concatenates the
        per-head moduli n-gram by n-gram, and the two orders agree.
        """
        return tuple(p for per_ngram in self.primes[layer] for p in per_ngram)

    def bucket_offsets(self, layer: int) -> tuple[int, ...]:
        """Start of each bucket in the layer's table.

        `np.cumsum([0, *sizes[:-1]])` in the reference: the buckets sit back to
        back, in the same order as the hash columns, so the table is covered with
        no gap and no overlap.
        """
        offsets, running = [], 0
        for size in self.flat_primes(layer):
            offsets.append(running)
            running += size
        return tuple(offsets)

    def row_counts(self) -> tuple[int, ...]:
        """Rows each table needs, derived from the primes rather than read from the config."""
        return tuple(sum(self.flat_primes(layer)) for layer in range(len(self.layer_ids)))

    def verify(self) -> list[str]:
        """Ways the derived layout disagrees with the declared one. Empty means consistent."""
        problems: list[str] = []
        if len(self.num_embeddings) != len(self.layer_ids):
            problems.append(
                f"engram_layer_ids has {len(self.layer_ids)} entries but "
                f"engram_num_embeddings has {len(self.num_embeddings)}"
            )
        shared: set[int] = set()
        for layer, layer_id in enumerate(self.layer_ids):
            sizes = self.flat_primes(layer)
            if len(sizes) != self.n_columns_expected(layer):
                problems.append(
                    f"layer {layer_id}: {len(sizes)} buckets, expected {self.n_columns_expected(layer)}"
                )
                continue
            if list(sizes) != sorted(sizes):
                problems.append(
                    f"layer {layer_id}: buckets are not in ascending order, so the "
                    "highest id would not be in the last bucket"
                )
            if len(shared & set(sizes)) or len(set(sizes)) != len(sizes):
                problems.append(f"layer {layer_id}: a bucket size is used twice")
            shared |= set(sizes)
            if layer >= len(self.num_embeddings):
                continue
            derived, declared = sum(sizes), self.num_embeddings[layer]
            if derived != declared:
                problems.append(
                    f"layer {layer_id}: derived {derived} rows, declared {declared} "
                    f"(difference {derived - declared:+d})"
                )
            if self.bucket_offsets(layer)[-1] + sizes[-1] != declared:
                problems.append(
                    f"layer {layer_id}: buckets do not cover [0, {declared}) exactly"
                )
        return problems

    def n_columns_expected(self, layer: int) -> int:
        """Buckets the shape of `primes[layer]` implies, so a truncated layout is caught."""
        return sum(len(heads) for heads in self.primes[layer])


def build_compressed_token_map(tokenizer: Any) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike collapse.

    Returns the lookup plus the size of the compressed vocabulary. That size is
    the value the config declares as `engram_compressed_vocab_size` and the value
    every hash multiplier is derived from, so it is worth asserting rather than
    trusting -- see `NgramHasher`'s `expected_token_map_size`.

    A token that decodes to a partial UTF-8 byte (`\\ufffd` in its text) has
    nothing to normalize and is keyed by its raw form instead.
    """
    from tokenizers import Regex, normalizers

    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(_BLANK_RUN), " "),
            normalizers.Replace(Regex(_LONE_SPACE), _SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(_SENTINEL, " "),
        ]
    )

    # The raw Rust tokenizer, matching what training decoded with. Going through the
    # Python wrapper would apply clean_up_tokenization_spaces and change the key.
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: Sequence[int], max_ngram_size: int, hash_vocab_size: int
) -> list[list[int]]:
    """One multiplier per (layer, lookback), from a per-layer RNG so layers hash differently.

    Kept odd, and bounded so that `token_id * multiplier` cannot overflow int64:
    the bound is `(int64 max // hash_vocab_size) // 2` and the multiplier doubles
    it, so the product stays under `int64 max` for every compressed id.

    The seed is `10007 * layer_id` -- the layer *id*, not its index -- so two
    configs that list the same layers in a different order hash identically.

    `hash_vocab_size` is the *compressed* vocabulary size. The reference names
    this parameter `tokenizer_vocab_size` while passing it
    `build_compressed_token_map`'s second return value; substituting the
    uncompressed `vocab_size` (129,280 for V4.1) changes `multiplier_bound` and
    silently rehashes every table.
    """
    import numpy as np

    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // hash_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append([int(v) * 2 + 1 for v in values])
    return rows


class NgramHasher:
    """Maps each position to the row ids of the n-grams ending there.

    Ids go through the compressed table first, then each position is hashed with
    the `max_ngram_size - 1` tokens before it. Look-back stops at the start of the
    sequence and at any dead token (an image span, cached as `DEAD`), so an n-gram
    never spans one; once a position is blocked every longer n-gram ending there
    is blocked too. The cache carries all of that across the prefill/decode split,
    which is why `hash_ids` takes `start_pos` and remembers earlier positions
    rather than hashing each call in isolation.
    """

    DEAD = -1

    def __init__(
        self,
        layout: EngramLayout,
        token_map: Sequence[int],
        pad_id: int,
        expected_token_map_size: int | None = None,
        multipliers: Sequence[Sequence[int]] | None = None,
    ) -> None:
        self.layout = layout
        self.token_map = list(token_map)
        self.token_map_size = len(set(self.token_map))
        # The reference asserts this at construction, and it has to be here rather
        # than at the call site: the compressed size feeds every multiplier, so a
        # mismatch produces a full set of plausible-looking but wrong row ids.
        if expected_token_map_size is not None and self.token_map_size != expected_token_map_size:
            raise ValueError(
                f"compressed vocabulary is {self.token_map_size}, but the config declares "
                f"{expected_token_map_size}; every Engram hash multiplier derives from this "
                "size, so the whole table would be hashed differently"
            )
        # The reference indexes the compressed table for the pad token rather than
        # using `engram_pad_id` directly, so a padded lookback slot carries the pad
        # token's compressed id.
        self.pad_id = self.token_map[pad_id]
        self.multipliers = (
            [list(row) for row in multipliers]
            if multipliers is not None
            else compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, self.token_map_size)
        )
        self.offsets = [layout.bucket_offsets(layer) for layer in range(len(layout.layer_ids))]
        self._cache: list[int] = []

    def reset(self) -> None:
        """Forget the positions carried across a prefill."""
        self._cache = []

    def hash_ids(
        self,
        token_ids: Sequence[int],
        start_pos: int = 0,
        token_mask: Sequence[bool] | None = None,
    ) -> list[list[list[int]]]:
        """Row ids of the n-grams ending at each position.

        `token_mask` is per token, False for tokens that take no part in an n-gram
        (image spans). Returns `[position][layer][hash column]`, the innermost axis
        ordered as `EngramLayout.flat_primes`, and every id is in
        `[0, EngramLayout.row_counts()[layer])`.

        Positions before `start_pos` come from the cache, so a decode step has to
        follow a prefill that wrote them; unwritten positions count as `DEAD`
        rather than as a stale hash.
        """
        if token_mask is not None and len(token_mask) != len(token_ids):
            raise ValueError("token_mask must have one entry per token")
        if start_pos < 0:
            raise ValueError("start_pos must not be negative")
        seqlen = len(token_ids)
        while len(self._cache) < start_pos + seqlen:
            self._cache.append(self.DEAD)

        for i, token_id in enumerate(token_ids):
            value = self.token_map[token_id]
            if token_mask is not None and not token_mask[i]:
                value = self.DEAD
            self._cache[start_pos + i] = value

        positions = [start_pos + i for i in range(seqlen)]
        blocked = [False] * seqlen
        tokens: list[list[int]] = []
        for shift in range(self.layout.max_ngram_size):
            source = [self._cache[max(0, position - shift)] for position in positions]
            blocked = [
                blocked[i] or (positions[i] < shift) or (source[i] == self.DEAD)
                for i in range(seqlen)
            ]
            tokens.append([self.pad_id if blocked[i] else source[i] for i in range(seqlen)])

        # XOR the multiplied ids one lookback at a time, so the running value after
        # step `i` is the hash of the (i + 2)-gram. Each of those lands in its own
        # bucket range: the modulus is per (layer, n-gram size, head) and the offset
        # is the range's start, so the largest id any column can produce is
        # `sum(primes) - 1`.
        result: list[list[list[int]]] = []
        for i in range(seqlen):
            per_layer = []
            for layer in range(len(self.layout.layer_ids)):
                multipliers = self.multipliers[layer]
                rolling = multipliers[0] * tokens[0][i]
                columns = []
                for index in range(self.layout.n_ngram_sizes):
                    rolling ^= multipliers[index + 1] * tokens[index + 1][i]
                    base = index * self.layout.n_heads
                    for head in range(self.layout.n_heads):
                        columns.append(
                            rolling % self.layout.primes[layer][index][head]
                            + self.offsets[layer][base + head]
                        )
                per_layer.append(columns)
            result.append(per_layer)
        return result


def _load_config(path: str) -> Mapping[str, Any]:
    """A V4.1 inference config, flat or nested either way the release ships it.

    The released `config.json` nests these fields under `text_config` and spells
    the pad id `engram_pad_token_id`; `inference/config.json` is flat. Six of the
    keys this module reads -- the layer ids, the n-gram size, the head count, the
    vocabulary, the head dim and the row counts -- are spelled identically in
    both, which is why the derivation is shape-independent and only the pad id
    has to be aliased. `V41Config` owns that mapping, so this defers to it rather
    than restating it.
    """
    from src.models.deepseek_v4_1.config import load_config

    return load_config(path).engram_block()


def main(argv: Sequence[str] | None = None) -> int:
    """Check a config's Engram arithmetic, and optionally its tokenizer, against each other.

    Exists because the two halves of the derivation live in different files: the row
    count is in the safetensors header and the bucket layout is in the config, and
    nothing else in the release compares them. Exit code 0 means they agree.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m src.encoding.engram",
        description="Verify the Engram hash layout a DeepSeek-V4.1 config implies.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="a released config.json or inference/config.json",
    )
    parser.add_argument(
        "--tokenizer",
        help="a tokenizer directory; without it the compressed vocab size cannot be checked",
    )
    parser.add_argument("--json", help="write the report here as JSON as well as to stdout")
    args = parser.parse_args(argv)

    config = _load_config(args.config)
    layout = EngramLayout.from_config(config)
    if layout is None:
        print("engram_layer_ids is empty: this config has no Engram tables")
        return 0

    report: dict[str, Any] = {
        "layer_ids": list(layout.layer_ids),
        "max_ngram_size": layout.max_ngram_size,
        "n_heads": layout.n_heads,
        "head_dim": layout.head_dim,
        "n_hash_columns": layout.n_hash_columns,
        "layers": [],
    }
    failed = False

    print(f"Engram layers {list(layout.layer_ids)} | {layout.n_hash_columns} hash columns per position")
    for index, layer_id in enumerate(layout.layer_ids):
        sizes = layout.flat_primes(index)
        offsets = layout.bucket_offsets(index)
        derived = sum(sizes)
        declared = layout.num_embeddings[index] if index < len(layout.num_embeddings) else None
        row = {
            "layer_id": layer_id,
            "buckets": len(sizes),
            "prime_min": min(sizes),
            "prime_max": max(sizes),
            "derived_rows": derived,
            "declared_rows": declared,
            "max_hash_id": declared - 1 if declared else None,
            "head_dim": layout.head_dim,
        }
        report["layers"].append(row)
        status = "ok" if declared == derived else f"MISMATCH ({derived - declared:+d})"
        print(
            f"  layer {layer_id:>3}: {len(sizes)} buckets, primes {min(sizes)}..{max(sizes)}, "
            f"rows derived {derived} vs declared {declared} -> {status}"
        )
        print(
            f"            embedding rows [{declared} x {layout.head_dim}] = "
            f"{declared * layout.head_dim / 2**30:.2f} GiB at one byte per element, "
            f"plus {declared * 8 / 2**30:.2f} GiB of row scales"
        )

    problems = layout.verify()
    if problems:
        failed = True
        for problem in problems:
            print(f"  [FAIL] {problem}", file=sys.stderr)
    else:
        print("  [ok] the primes add up to the declared row counts")
    report["layout_problems"] = problems

    if args.tokenizer is None:
        print("\n[SKIP] compressed token map: pass --tokenizer to check it against the config")
        report["compressed_vocab_size"] = None
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        token_map, size = build_compressed_token_map(tokenizer)
        declared_size = config.get("engram_compressed_vocab_size")
        print(f"\ntokenizer {args.tokenizer}: {len(tokenizer)} tokens -> {size} compressed ids")
        report["compressed_vocab_size"] = size
        report["tokenizer_vocab_size"] = len(tokenizer)
        if declared_size != size:
            failed = True
            print(
                f"  [FAIL] engram_compressed_vocab_size is {declared_size}, but the tokenizer "
                f"compresses to {size}; every hash multiplier derives from this size, so the "
                "hash ids below cannot be the ones this checkpoint was built with",
                file=sys.stderr,
            )
        else:
            print(f"  [ok] matches engram_compressed_vocab_size ({declared_size})")

            hasher = NgramHasher(
                layout,
                token_map,
                int(config["engram_pad_id"]),
                expected_token_map_size=declared_size,
            )
            rows = layout.row_counts()
            print(f"  pad id {config['engram_pad_id']} -> compressed {hasher.pad_id}")
            for layer_id, multipliers in zip(layout.layer_ids, hasher.multipliers):
                print(f"  layer {layer_id} multipliers: {multipliers}")

            sample = [int(v) for v in range(len(token_map) - 8, len(token_map))]
            ids = hasher.hash_ids(sample)
            for layer in range(len(layout.layer_ids)):
                values = [value for position in ids for value in position[layer]]
                inside = all(0 <= value < rows[layer] for value in values)
                failed = failed or not inside
                print(
                    f"  layer {layout.layer_ids[layer]}: {len(values)} ids from {len(sample)} tokens, "
                    f"min {min(values)} max {max(values)} of {rows[layer]} rows -> "
                    f"{'all in range' if inside else 'OUT OF RANGE'}"
                )
            report["sample_max_hash_id"] = [max(v for p in ids for v in p[l]) for l in range(len(rows))]

    if args.json:
        import json

        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\nwrote {args.json}")

    print("\n[FAIL] Engram layout verification" if failed else "\n[PASS] Engram layout verification")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
