"""Is a restored prefix the state the same prefix would have built from scratch?

A prefix cache is only sound if a *resume* is the same forward as a cold prefill cut at the same
place, and that is a claim about the layers' caches rather than about the logits one request happens
to produce. `tests/test_models_deepseek_v4_1_chunked_prefill.py` pins the first half of it -- a chunk
boundary is not a new kind of forward -- and this file pins the second: the state *before* that
boundary can be put back on the host and reloaded into a wiped module without the continuation
noticing. The two together are the whole of what makes a request that starts with another request's
tokens safe to answer from the first request's leftovers.

The arms below are cut at every split point of a twelve-token prompt rather than at a few, because
the anchoring rules have no privileged length: an odd cut lands inside a compressor group, a cut at
the window lands on the ring's own boundary, and `N_TOKENS - 1` leaves a one-token tail, which is not
a continuation at all but the decode path -- the case an exact repeat of a prompt degenerates to, and
the one `lookup`'s clamp to `len(ids) - 1` exists to produce.

The store's own rules are tested apart from the model, on a payload that is a few tensors: which
anchor wins, what a miss falls back to, what the budget evicts, and what a sliced snapshot carries.
None of those needs a card and none of them is about the numbers.
"""

from __future__ import annotations

import pytest
import torch

from src.models.deepseek_v4_1 import attention as attention_module
from src.models.deepseek_v4_1.prefix_cache import (
    BLOCK_TOKENS,
    PrefixCache,
    geometry_tag,
    prefix_hashes,
    restore_rows,
    snapshot_rows,
)
from test_models_deepseek_v4_1_chunked_prefill import (  # noqa: E402 - same directory, as the other tests do it
    N_TOKENS,
    _build,
    _compare,
    _run,
    _states,
)

# The width the model was built at, which is what the row rule divides `max_seq_len` by. `TOY`'s
# `max_position_embeddings` is larger; the buffers are the thing a ratio is read off, so this is the
# number that has to match the stack and not the config's ceiling.
MAX_SEQ_LEN = 64

# Every split point of the prompt. 11 is the one-token tail, 4 is the window, and the odd ones cut a
# group in half -- the three placements the chunked-prefill file calls out, here on the far side of a
# restore rather than of a chunk.
CUTS = tuple(range(1, N_TOKENS))


def _resume(stack, x, cut: int, chunk: int) -> torch.Tensor:
    """Prefill to `cut`, snapshot, wipe, restore, and forward the rest.

    The wipe between the two halves is not a test artefact: `reset_state` is what the served path
    calls first, because the ring and the three position tables are the one piece of state that
    outlives a request and the restore is a copy over zeros rather than over the last request's
    tokens. A restore into a state that was not reset would be reading the snapshot's own gaps as
    whatever the previous forward left there.
    """
    stack.reset_state(x.size(0))
    out = []
    for c0 in range(0, cut, chunk):
        out.append(stack(x[:, c0 : min(c0 + chunk, cut)], c0))
    saved = snapshot_rows(stack, cut, MAX_SEQ_LEN)

    stack.reset_state(x.size(0))
    restore_rows(stack, saved)
    for c0 in range(cut, x.size(1), chunk):
        out.append(stack(x[:, c0 : min(c0 + chunk, x.size(1))], c0))
    return torch.cat(out, dim=1)


def _resumed_against_one_shot(cut: int, chunk: int = 5) -> None:
    """One arm: the same prompt prefilled in one forward, and prefilled to `cut` then resumed."""
    _, cfg = _build()
    x = torch.randn(1, N_TOKENS, cfg.dim, dtype=attention_module.LINEAR_DTYPE)

    whole, _ = _build()
    reference = _run(whole, x, N_TOKENS)

    stack, _ = _build()
    out = _resume(stack, x, cut, chunk)
    assert out.shape == reference.shape
    assert torch.equal(out, reference), (
        f"cut {cut}, chunk {chunk}: the tokens past the cut differ, max abs diff "
        f"{(out[:, cut:].float() - reference[:, cut:].float()).abs().max().item()}"
    )
    _compare(_states(stack, N_TOKENS), _states(whole, N_TOKENS), f"cut {cut}, chunk {chunk}")


def test_a_restored_prefix_leaves_the_caches_a_one_shot_prefill_leaves() -> None:
    """The property the store rests on, at every split point of the prompt.

    Only the tokens past the cut can carry a difference, and they are compared along with every cache
    the resumed forward leaves behind, because a cache that was restored a row short would show up as
    a logit difference several tokens later rather than as a shape anything checks.
    """
    for cut in CUTS:
        _resumed_against_one_shot(cut)


@pytest.mark.parametrize("cut", CUTS)
def test_every_split_point_resumes_to_the_same_state(cut: int) -> None:
    """The parametrized form, one split point per test id.

    The loop above reports the first cut that broke and stops; this one names each cut on its own,
    which is what separates the failure of a cut inside a group from the failure of a one-token tail.
    """
    _resumed_against_one_shot(cut)


def test_a_one_token_tail_is_the_decode_path_and_not_the_continuation_one() -> None:
    """An exact repeat of a prompt forwards one token, and it is not a chunk of one.

    `lookup` clamps a matched length to `len(ids) - 1`, because a forward is always needed for the
    logits the first new token comes from; for a prompt the store already holds exactly, that is a
    single query at a position past zero, which `_is_continuation` sends to the *decode* bodies. The
    chunked-prefill file pins that a one-token forward at a position past zero is the decode path and
    equals the one-shot's last row; this says the same row comes out of a restored prefix, which is
    the shape a repeated prompt actually takes in the served path.
    """
    _, cfg = _build()
    x = torch.randn(1, N_TOKENS, cfg.dim, dtype=attention_module.LINEAR_DTYPE)

    whole, _ = _build()
    reference = _run(whole, x, N_TOKENS)

    stack, _ = _build()
    out = _resume(stack, x, N_TOKENS - 1, chunk=N_TOKENS - 1)
    assert torch.equal(out[:, -1], reference[:, -1])
    assert torch.equal(out, reference)


def test_a_snapshot_at_the_head_anchor_is_a_chunk_boundary_like_any_other() -> None:
    """The head anchor is a cut whose head is one forward, which is what a cold prefill would run.

    A cold prefill under a head anchor is `front(ids[:p_head], 0, chunk=p_head)` and then the rest --
    one extra boundary on a miss, and none on a hit. There is nothing special about it beyond that:
    the arm here is a cut at exactly one chunk, which is the shape the anchor forces.
    """
    _resumed_against_one_shot(N_TOKENS // 2, chunk=N_TOKENS // 2)


def test_a_row_limited_snapshot_holds_the_rows_a_prefix_has_written() -> None:
    """The slice rule, asserted against the toy geometry rather than against a byte count.

    Two of the five buffers are position tables -- one row per `compress_ratio` positions -- and a
    prefix of six tokens has written three rows of a ratio-2 layer and six of a ratio-1 layer. The
    rule is read off each buffer's own width, so this is where a buffer that moved to a different
    ratio would be caught, and the three windows and the compressor state are checked to be whole,
    since cutting *them* would be the other half of the same mistake.
    """
    _, cfg = _build()
    x = torch.randn(1, N_TOKENS, cfg.dim, dtype=attention_module.LINEAR_DTYPE)
    stack, _ = _build()
    stack.reset_state(1)
    stack(x[:, :6], 0)
    saved = snapshot_rows(stack, 6, MAX_SEQ_LEN)

    compress = {name: value.shape[1] for name, value in saved.items() if name.endswith("compress_kv_cache")}
    keys = {name: value.shape[1] for name, value in saved.items() if name.endswith("k_cache")}
    assert compress == {"layers.2.compress_kv_cache": 3, "layers.4.compress_kv_cache": 6}
    # The index keys are cut from the compressor's latent, so only a layer that compresses its own KV
    # owns a `k_cache` at all -- which is why there are two of these and not three.
    assert keys == {"layers.2.indexer.k_cache": 3, "layers.4.indexer.k_cache": 6}
    # The ring is a fixed table of slots keyed by `position % window_size`, and the compressor's state
    # is the group still filling: neither is a function of how many rows a prefix has written.
    assert {value.shape[1] for name, value in saved.items() if name.endswith("window_kv_cache")} == {4}
    assert {value.shape[1] for name, value in saved.items() if name.endswith(("kv_state", "score_state"))} == {2}
    # And every one of them is on the host, because the store outlives the request and
    # `DecodeGraphs.release` calls `torch.cuda.empty_cache`.
    assert all(value.device.type == "cpu" for value in saved.values())


def test_a_whole_snapshot_is_the_buffers_themselves() -> None:
    """`limit=None` is the capture pass's rewind: every buffer, uncut, and left where it is.

    Left on the card on purpose. The capture pass snapshots around one forward and copies back inside
    the same call, so a snapshot that went to the host would be a round trip with nothing on the
    other side of it -- and at 256K the caches are a gigabyte. The store is the other caller and the
    other choice, which is why both are asserted here rather than one.
    """
    stack, _ = _build()
    saved = snapshot_rows(stack, to_host=False)
    by_name = dict(stack.named_buffers())
    assert saved
    for name, value in saved.items():
        assert value.shape == by_name[name].shape
        assert value.device == by_name[name].device
    assert all(value.device.type == "cpu" for value in snapshot_rows(stack).values())


# -- the store ------------------------------------------------------------------------------------

PROMPT = list(range(200))


def _payload(nbytes: int) -> dict[str, torch.Tensor]:
    """A stand-in for a snapshot: the store counts bytes and never looks inside."""
    return {"fill": torch.zeros(max(nbytes // 4, 1), dtype=torch.float32)}


def _cache(budget: int = 1 << 20, min_tokens: int = BLOCK_TOKENS) -> PrefixCache:
    return PrefixCache(budget_bytes=budget, max_seq_len=1024, min_tokens=min_tokens)


def test_an_exact_repeat_is_clamped_by_one_token() -> None:
    """The store holds the prompt's own length, and the request forwards one token anyway."""
    ids = PROMPT[:130]
    cache = _cache()
    cache.store(ids, 130, _payload(64))
    hit = cache.lookup(ids)
    assert hit is not None
    length, entry = hit
    assert entry.length == 130
    assert length == 129
    assert cache.stats()["reused_tokens"] == 130


def test_a_longer_prompt_hits_the_anchor_whole() -> None:
    """The conversational case: the next turn is the previous prompt plus an answer plus a question."""
    cache = _cache()
    cache.store(PROMPT, 130, _payload(64))
    hit = cache.lookup(PROMPT + [1, 2, 3])
    assert hit is not None and hit[0] == 130


def test_the_head_anchor_and_the_end_anchor_serve_different_tails() -> None:
    """Two conversations under one rendered header: the head hits, the end does not."""
    header = PROMPT[:BLOCK_TOKENS]
    first = header + list(range(1000, 1100))
    second = header + list(range(2000, 2100))
    cache = _cache()
    cache.store(first, BLOCK_TOKENS, _payload(64))
    cache.store(first, 164, _payload(64))

    assert cache.lookup(first + [7])[0] == 164
    assert cache.lookup(second + [7])[0] == BLOCK_TOKENS


def test_the_longest_anchor_wins_and_a_miss_falls_back() -> None:
    """A length whose key does not match is skipped and the walk continues below it."""
    cache = _cache()
    cache.store(PROMPT, 66, _payload(64))
    cache.store(PROMPT, 130, _payload(64))
    assert cache.lookup(PROMPT)[0] == 130

    diverged = list(range(129)) + [999]
    assert cache.lookup(diverged + [5])[0] == 66


def test_a_length_that_did_not_hash_alike_is_a_miss() -> None:
    """Same length, different tokens: the key is the tokens, so this is not a hit."""
    cache = _cache()
    cache.store(PROMPT, 130, _payload(64))
    assert cache.lookup(list(range(129)) + [999] + [5]) is None
    assert cache.stats()["misses"] == 1


def test_a_length_that_is_not_on_a_block_boundary_is_keyed_all_the_same() -> None:
    """The chain answers every multiple of the block from its own list, and a tail with one hash."""
    cache = _cache()
    for length in (64, 66, 128, 130):
        assert cache.store(PROMPT, length, _payload(64)) is not None
    assert cache.lengths() == [130, 128, 66, 64]
    assert cache.lookup(PROMPT)[0] == 130


def test_a_length_below_the_floor_or_past_the_prompt_is_not_taken() -> None:
    """A refused store is not an error -- the next request just pays for the prompt again."""
    cache = _cache(min_tokens=64)
    assert cache.store(PROMPT, 63, _payload(64)) is None
    assert cache.store(PROMPT[:10], 11, _payload(64)) is None
    assert cache.store(PROMPT, 1025, _payload(64)) is None
    assert len(cache) == 0


def test_an_entry_that_does_not_fit_the_budget_on_its_own_is_not_taken() -> None:
    """A budget the store cannot honour for this entry is a budget it does not overshoot."""
    cache = _cache(budget=100)
    assert cache.store(PROMPT, 64, _payload(128)) is None
    assert len(cache) == 0 and cache.bytes == 0


def test_eviction_takes_the_least_recently_used_and_holds_the_budget() -> None:
    """Four 64-byte anchors into a 200-byte budget: the one touched last survives."""
    cache = _cache(budget=200, min_tokens=1)
    for length in (32, 64, 96):
        cache.store(PROMPT, length, _payload(64))
    assert cache.lookup(PROMPT)[0] == 96  # touches the newest, which is then the most recent
    cache.store(PROMPT, 128, _payload(64))

    assert cache.bytes == 192
    assert cache.lengths() == [128, 96, 64]


def test_a_second_store_of_the_same_prefix_replaces_rather_than_grows() -> None:
    """A chat loop resends its history; the store holds one copy of it, not one per turn."""
    cache = _cache()
    cache.store(PROMPT, 64, _payload(64))
    second = cache.store(PROMPT, 64, _payload(64))
    assert len(cache) == 1 and cache.bytes == 64
    assert cache.lookup(PROMPT)[1] is second


def test_a_cleared_store_holds_nothing() -> None:
    cache = _cache()
    cache.store(PROMPT, 64, _payload(64))
    cache.clear()
    assert len(cache) == 0 and cache.bytes == 0 and cache.lookup(PROMPT) is None


def test_the_geometry_is_mixed_into_every_key() -> None:
    """A chain is only meaningful inside the geometry it was taken in: world, context and shapes."""
    stack, _ = _build()
    tag = geometry_tag(stack, 4, MAX_SEQ_LEN)
    assert tag != geometry_tag(stack, 1, MAX_SEQ_LEN)
    assert tag != geometry_tag(stack, 4, MAX_SEQ_LEN * 2)
    # and the chain is seeded by it, so the same tokens under two geometries are two keys
    assert prefix_hashes(PROMPT, b"") != prefix_hashes(PROMPT, b"\x01")
    assert prefix_hashes(PROMPT, b"") == prefix_hashes(PROMPT, b"")


def test_the_chain_keys_every_block_and_the_length_beside_it() -> None:
    """`out[j - 1]` is the key for the first `64 * j` tokens, and the length is not decoration.

    Two prompts that share a block at the same place key alike below it and apart from a prompt that
    carries the same block at a different offset, which is the property that makes a chain a prefix
    key rather than a set of block keys.
    """
    blocks = prefix_hashes(PROMPT)
    assert len(blocks) == len(PROMPT) // BLOCK_TOKENS == 3
    assert prefix_hashes(PROMPT[:BLOCK_TOKENS]) == blocks[:1]
    assert prefix_hashes(PROMPT, b"") == prefix_hashes(PROMPT, b"")

    shifted = list(range(64)) + PROMPT[:64]  # the same block one block later
    assert prefix_hashes(shifted)[1] != blocks[0]
