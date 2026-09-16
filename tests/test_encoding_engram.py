"""Tests for `src/encoding/engram.py`.

The Engram hash tables are 189.13 GiB of FP8 whose rows are named by ids computed
from the tokenizer, so the only thing that can be checked without the checkpoint
is whether the derivation closes: the primes a V4.1 config implies must add up to
the row count the same config declares, and the ids the hasher can produce must
be exactly that row range. Both are asserted here.

Two verification levels are used deliberately:

* The arithmetic is checked against values derived from the released reference on
  the host -- the primes, the offsets and the eight hash multipliers are pinned as
  literals, so a change in the derivation fails the test rather than quietly
  re-hashing all 189 GiB.
* `is_prime` is checked against `sympy.isprime`, which is what the reference calls,
  and the compressed token map against the V4-Flash tokenizer that is present on
  this host.

numpy is not a declared dependency of this repository, so the tests that draw the
multipliers -- the pinned values and everything built on `NgramHasher` -- ask for
it and skip without it. A skip is a skip, not a pass: none of these three
dependencies is optional to the claims they cover.

What is *not* covered, and cannot be on this host: no V4.1 checkpoint exists here,
so no assertion in this file has been run against V4.1's own tokenizer or against a
row actually read out of the table. See the V4.1 model page for that limitation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from src.encoding.engram import (
    EngramLayout,
    NgramHasher,
    build_compressed_token_map,
    compute_hash_multipliers,
    find_next_prime,
    is_prime,
)

# The Engram block of the V4.1 config, transcribed from the model page rather than
# read from disk: there is no V4.1 checkpoint on this host, and the values are what
# the safetensors headers were audited against.
ENGRAM_CONFIG = {
    "engram_layer_ids": [1, 14],
    "engram_max_ngram_size": 4,
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_vocab_size": 16_000_000,
    "engram_num_embeddings": [384_006_168, 384_016_682],
    "engram_compressed_vocab_size": 99_092,
    "engram_pad_id": 2,
}

# A real tokenizer, for the compressed-token-map test only. Not the V4.1 one: the
# checkpoint is not on this host, and the V4-Flash tokenizer reproduces the same
# 99,092 count while being 111 bytes smaller than V4.1's.
V4_FLASH_TOKENIZER = Path("/mnt/data3/DeepSeek-V4-Flash-0731")

# One multiplier per (layer, lookback), from numpy's PCG64 stream at seed
# 10007 * layer_id with the compressed vocabulary (99,092) as the bound.
EXPECTED_MULTIPLIERS = [
    [76632096046245, 4839876093313, 35959672319349, 73987337458391],  # layer 1
    [67716810739261, 51510806800915, 30921347202721, 82619226485591],  # layer 14
]

# The 24 bucket sizes per layer, in hash-column order. All 48 are distinct and all
# start above engram_vocab_size.
EXPECTED_PRIME_EXTREMES = [(16_000_057, 16_000_463), (16_000_477, 16_000_889)]


@pytest.fixture(scope="module")
def layout() -> EngramLayout:
    derived = EngramLayout.from_config(ENGRAM_CONFIG)
    assert derived is not None
    return derived


def test_row_counts_match_the_declared_embeddings(layout: EngramLayout) -> None:
    """The closed loop: the primes a config implies add up to the rows it declares."""
    assert layout.row_counts() == (384_006_168, 384_016_682)
    assert layout.row_counts() == tuple(ENGRAM_CONFIG["engram_num_embeddings"])
    assert layout.verify() == []


def test_verify_reports_a_tampered_row_count() -> None:
    """Negative control, so a `verify` that always returns [] would fail."""
    tampered = dict(ENGRAM_CONFIG, engram_num_embeddings=[384_006_169, 384_016_682])
    derived = EngramLayout.from_config(tampered)
    assert derived is not None
    problems = derived.verify()
    assert any("derived 384006168" in p and "difference -1" in p for p in problems)


def test_primes_are_distinct_across_both_layers(layout: EngramLayout) -> None:
    """The two tables share one prime stream, so a repeat would overlap two ranges."""
    everything = [p for layer in range(2) for p in layout.flat_primes(layer)]
    assert len(everything) == 48
    assert len(set(everything)) == 48
    assert min(everything) > ENGRAM_CONFIG["engram_vocab_size"]
    assert [(min(layout.flat_primes(i)), max(layout.flat_primes(i))) for i in range(2)] == (
        EXPECTED_PRIME_EXTREMES
    )


def test_ascending_within_a_layer(layout: EngramLayout) -> None:
    """Ascending order is what puts the highest id in the last bucket."""
    for layer in range(2):
        sizes = layout.flat_primes(layer)
        assert list(sizes) == sorted(sizes)


def test_buckets_cover_the_table_exactly(layout: EngramLayout) -> None:
    """No gap and no overlap, so `sum(primes)` is the whole address space."""
    for layer, declared in enumerate(layout.row_counts()):
        sizes, offsets = layout.flat_primes(layer), layout.bucket_offsets(layer)
        assert offsets[0] == 0
        assert all(
            offsets[i + 1] == offsets[i] + sizes[i] for i in range(len(sizes) - 1)
        )
        assert offsets[-1] + sizes[-1] == declared
        # the highest id any column can emit
        assert max(offsets[i] + sizes[i] - 1 for i in range(len(sizes))) == declared - 1


def test_column_count(layout: EngramLayout) -> None:
    """2-gram through 4-gram, eight heads each."""
    assert layout.n_ngram_sizes == 3
    assert layout.n_hash_columns == 24
    assert all(len(layout.flat_primes(i)) == 24 for i in range(2))


def test_multipliers_are_pinned(layout: EngramLayout) -> None:
    """Pinned so a numpy change is a failure and not a silent re-hash of both tables."""
    pytest.importorskip("numpy")  # the multipliers come from numpy's PCG64 stream
    assert compute_hash_multipliers(
        layout.layer_ids, layout.max_ngram_size, ENGRAM_CONFIG["engram_compressed_vocab_size"]
    ) == EXPECTED_MULTIPLIERS
    assert all(m % 2 == 1 for row in EXPECTED_MULTIPLIERS for m in row)


def test_multiplier_seed_is_the_layer_id_not_its_index() -> None:
    """The seed is `10007 * layer_id`, so a layer's own row does not depend on where it is listed."""
    pytest.importorskip("numpy")
    vocab = ENGRAM_CONFIG["engram_compressed_vocab_size"]
    forward = compute_hash_multipliers((1, 14), 4, vocab)
    reversed_order = compute_hash_multipliers((14, 1), 4, vocab)
    assert reversed_order == list(reversed(forward))
    assert forward == EXPECTED_MULTIPLIERS


def test_multiplier_bound_uses_the_compressed_vocabulary() -> None:
    """The trap this guards: the bound divides by the compressed size, not `vocab_size`."""
    numpy = pytest.importorskip("numpy")
    max_long = numpy.iinfo(numpy.int64).max
    compressed = ENGRAM_CONFIG["engram_compressed_vocab_size"]
    assert max(1, (max_long // compressed) // 2) == 46_539_438_283_891
    # the uncompressed `vocab_size` would give a different bound, hence different
    # multipliers, hence a differently hashed table with the same weights
    assert max(1, (max_long // 129_280) // 2) != 46_539_438_283_891
    assert compute_hash_multipliers((1,), 4, compressed) != compute_hash_multipliers((1,), 4, 129_280)


def test_is_prime_matches_sympy_over_the_bucket_window() -> None:
    """`sympy.isprime` is the reference's oracle; the rest of the file depends on agreeing."""
    sympy = pytest.importorskip("sympy")
    for n in range(16_000_000, 16_001_000):
        assert is_prime(n) == bool(sympy.isprime(n)), n
    for n in (0, 1, 2, 3, 4, 25, 49, 2**31 - 1):
        assert is_prime(n) == bool(sympy.isprime(n)), n


def test_find_next_prime_skips_ones_already_handed_out() -> None:
    seen = {16_000_057}
    assert find_next_prime(16_000_000, seen) == 16_000_079
    assert find_next_prime(16_000_000, set()) == 16_000_057


def _hasher(layout: EngramLayout, pad_id: int | None = None) -> NgramHasher:
    """A hasher over an identity compressed map of the real compressed size.

    An identity map keeps the multipliers identical to the real ones while making
    the expected ids computable by hand.
    """
    pytest.importorskip("numpy")  # NgramHasher derives its multipliers from numpy's stream
    size = ENGRAM_CONFIG["engram_compressed_vocab_size"]
    return NgramHasher(
        layout,
        list(range(size)),
        ENGRAM_CONFIG["engram_pad_id"] if pad_id is None else pad_id,
        expected_token_map_size=size,
    )


def test_hash_ids_stay_inside_the_table(layout: EngramLayout) -> None:
    """Every reachable id is a declared row, which is why the header row count is the address space."""
    hasher = _hasher(layout)
    rows = layout.row_counts()
    sequence = [7, 41, 1_000, 99_091, 3, 512, 65_535, 2, 88_888]
    ids = hasher.hash_ids(sequence)
    assert len(ids) == len(sequence)
    for position in ids:
        for layer, columns in enumerate(position):
            assert len(columns) == layout.n_hash_columns
            assert all(0 <= value < rows[layer] for value in columns)
    # a run this long has to reach the top bucket of at least one column, otherwise
    # the row count would be a loose upper bound rather than the address space
    assert max(value for position in ids for layer in position for value in layer) > rows[0] // 2


def test_first_position_pads_the_longer_ngrams(layout: EngramLayout) -> None:
    """Look-back stops at the start of the sequence, so shifts past it take the pad id."""
    hasher = _hasher(layout)
    pad = hasher.pad_id
    assert pad == ENGRAM_CONFIG["engram_pad_id"]  # identity map, so uncompressed here
    columns = hasher.hash_ids([12345])[0][0]
    multipliers = EXPECTED_MULTIPLIERS[0]
    rolling = multipliers[0] * 12345
    for index in range(3):
        rolling ^= multipliers[index + 1] * pad
        sizes = layout.flat_primes(0)
        offsets = layout.bucket_offsets(0)
        base = index * layout.n_heads
        for head in range(layout.n_heads):
            assert columns[base + head] == rolling % sizes[base + head] + offsets[base + head]


def test_dead_tokens_block_longer_ngrams(layout: EngramLayout) -> None:
    """An n-gram never spans a dead token, and blocking is sticky once it happens."""
    hasher = _hasher(layout)
    pad = hasher.pad_id
    multipliers = EXPECTED_MULTIPLIERS[0]
    sizes, offsets = layout.flat_primes(0), layout.bucket_offsets(0)

    def expected_row(terms: Sequence[int]) -> list[int]:
        """The 24 ids a position gets when its four lookback slots hold `terms`."""
        rolling = multipliers[0] * terms[0]
        columns = []
        for index in range(layout.n_ngram_sizes):
            rolling ^= multipliers[index + 1] * terms[index + 1]
            base = index * layout.n_heads
            columns.extend(rolling % sizes[base + h] + offsets[base + h] for h in range(layout.n_heads))
        return columns

    # a dead token two back blocks the 3-gram as well: `blocked` is sticky, so the
    # 4-gram is padded even though its lookback reach is live
    ids = hasher.hash_ids([100, 200, 300, 400], token_mask=[True, True, False, True])
    assert ids[3][0] == expected_row([400, pad, pad, pad])

    # a dead token exactly at the 4-gram's reach blocks only that column group
    ids = hasher.hash_ids([100, 200, 300, 400], token_mask=[False, True, True, True])
    assert ids[3][0] == expected_row([400, 300, 200, pad])

    # nothing blocks the 4-gram when all four tokens are live
    ids = hasher.hash_ids([100, 200, 300, 400], token_mask=[True] * 4)
    assert ids[3][0] == expected_row([400, 300, 200, 100])

    # the dead token itself is padded in every column group, not just the longest one
    ids = hasher.hash_ids([100, 200, 300], token_mask=[True, False, True])
    assert ids[1][0] == expected_row([pad, pad, pad, pad])


def test_fully_masked_sequence_is_position_independent(layout: EngramLayout) -> None:
    """When nothing is live every n-gram is pad-only, so the ids stop depending on position."""
    hasher = _hasher(layout)
    ids = hasher.hash_ids([1, 2, 3, 4, 5], token_mask=[False] * 5)
    first = ids[0][0]
    assert all(position[0] == first for position in ids)


def test_prefill_then_decode_equals_one_call(layout: EngramLayout) -> None:
    """The cache carries the look-back across the split, so chunking must not change ids."""
    sequence = [11, 22, 33, 44, 55, 66, 77, 88]
    one_call = _hasher(layout).hash_ids(sequence)

    split = _hasher(layout)
    first = split.hash_ids(sequence[:5], start_pos=0)
    rest = split.hash_ids(sequence[5:], start_pos=5)
    assert first + rest == one_call


def test_chunk_boundaries_do_not_leak_across_calls(layout: EngramLayout) -> None:
    """A fresh hasher starting mid-sequence must not see the earlier tokens."""
    sequence = [11, 22, 33, 44, 55, 66, 77, 88]
    pad = _hasher(layout).pad_id
    fresh = _hasher(layout).hash_ids([sequence[5], sequence[6]], start_pos=0)
    cached = _hasher(layout).hash_ids(sequence[:7], start_pos=0)[5:7]
    assert fresh != cached
    # the fresh call sees position 0, so the 3-gram and 4-gram run out of look-back
    # and take the pad id instead of reaching into the real sequence
    multipliers = EXPECTED_MULTIPLIERS[0]
    rolling = multipliers[0] * sequence[6]
    rolling ^= multipliers[1] * sequence[5]
    rolling ^= multipliers[2] * pad
    rolling ^= multipliers[3] * pad
    sizes, offsets = layout.flat_primes(0), layout.bucket_offsets(0)
    base = 2 * layout.n_heads
    for head in range(layout.n_heads):
        assert fresh[1][0][base + head] == rolling % sizes[base + head] + offsets[base + head]


def test_wrong_compressed_vocab_size_is_rejected(layout: EngramLayout) -> None:
    """The reference asserts this at construction because it feeds every multiplier."""
    pytest.importorskip("numpy")
    size = ENGRAM_CONFIG["engram_compressed_vocab_size"]
    with pytest.raises(ValueError, match="every Engram hash multiplier derives from this size"):
        NgramHasher(
            layout,
            list(range(size)),
            ENGRAM_CONFIG["engram_pad_id"],
            expected_token_map_size=size + 1,
        )


def test_hash_ids_rejects_bad_arguments(layout: EngramLayout) -> None:
    hasher = _hasher(layout)
    with pytest.raises(ValueError, match="one entry per token"):
        hasher.hash_ids([1, 2, 3], token_mask=[True])
    with pytest.raises(ValueError, match="must not be negative"):
        hasher.hash_ids([1, 2], start_pos=-1)
    with pytest.raises(IndexError):
        hasher.hash_ids([ENGRAM_CONFIG["engram_compressed_vocab_size"]])


def test_two_engram_layers_hash_differently(layout: EngramLayout) -> None:
    """Layers 1 and 14 have different primes and multipliers, so ids must not coincide."""
    ids = _hasher(layout).hash_ids([5, 10, 15, 20, 25])[-1]
    assert ids[0] != ids[1]


@pytest.mark.skipif(not V4_FLASH_TOKENIZER.exists(), reason="V4-Flash tokenizer not on this host")
def test_compressed_token_map_on_a_real_tokenizer() -> None:
    """The 99,092 figure, reproduced zero-download against the tokenizer on this host.

    This is the V4-Flash tokenizer, which is not byte-identical to V4.1's: its
    `tokenizer.json` is 6,367,146 bytes against V4.1's 6,367,257, and it lacks the
    `<｜deepseek_image｜>` and `<｜System｜>` added tokens V4.1's prompt format
    needs. The count agreeing is therefore strong evidence, not a self-contained
    proof -- see the V4.1 model page.
    """
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(V4_FLASH_TOKENIZER))
    assert len(tokenizer) == 129_280
    lookup, size = build_compressed_token_map(tokenizer)
    assert size == 99_092 == ENGRAM_CONFIG["engram_compressed_vocab_size"]

    # the compressed ids are exactly [0, size) with no gaps, which is what the
    # `token_map_size` NgramHasher derives from `len(set(...))` assumes
    assert len(lookup) == len(tokenizer)
    assert set(lookup) == set(range(size))

    # the U+E000 sentinel: the lone-space token must not fold into the id of a token
    # that normalizes to nothing
    backend = tokenizer.backend_tokenizer
    space_ids = [i for i in range(len(tokenizer)) if backend.decode([i], skip_special_tokens=False) == " "]
    assert space_ids == [223]
    assert lookup[223] != lookup[0]
    assert lookup[223] == 174

    # the case/whitespace folding the map exists for
    folded = set()
    for text in (" The", "the", "THE", " the"):
        ids = backend.encode(text, add_special_tokens=False).ids
        assert ids
        folded.add(lookup[ids[0]])
    assert len(folded) == 1
