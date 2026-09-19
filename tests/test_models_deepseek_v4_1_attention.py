"""Does the CSA2 layer stack hold together as a stack, without a checkpoint to compare against?

`src/models/deepseek_v4_1/attention.py` is the V4.1 attention half written out in pure PyTorch.
There is no oracle for it on this host: the released runtime implements its ops in TileLang, which
needs `torch>=2.10.0` (this environment is on 2.9.1), and the sm_75 cards here have no FP4 tensor
core. So, as with `test_models_deepseek_v4_1_kernels.py`, nothing below claims to reproduce the
released model's tokens. What is checked is the internal arithmetic that *is* decidable:

* **Prefill and decode agree.** The same twelve tokens, once in a single forward and once as twelve
  one-token forwards, must produce the same output. That is the property every later measurement
  rests on -- a KV cache that only works because prefill and decode are never compared is not a
  cache -- and it exercises the whole mechanism at once: the sliding-window ring buffer, the
  compressor's group accumulation across forwards, the index-key cache, and the four shared slots
  `SharedAttentionRuntime` carries down the stack.
* **The compressed indices are causal.** A query at position `p` may attend to compressed position
  `j` only when group `j` has already closed, i.e. `j < (p + 1) // compress_ratio`, and the `-1`
  padding fills exactly the rest of the `index_topk` width.
* **The shared slots are wired the way the modes say.** Only `kv_source_layers` own an index key
  cache; only `candidate_source_layer` writes the candidate mask; only layers *after* it read one.

One measurement shaped the config below and is worth stating, because it looks like a bug otherwise.
`index_topk` truncation makes prefill and decode differ, and the difference is a **tie artifact, not
an arithmetic error**: with `index_topk` smaller than the number of reachable compressed positions,
every mismatch lands on a query whose k-th and (k+1)-th scores are equal, so prefill's wider score
matrix breaks the tie differently than decode's narrower one. Sweeping eight seeds found no mismatch
at a position whose k-th score was distinct, and layers isolated one at a time agreed exactly until
the first truncated index source. With `index_topk` and `candidate_topk_blocks` large enough that
nothing is truncated, the two orderings are bit-identical. So the toy config sets both wide, and the
assertion is equality rather than a tolerance -- a tolerance here would hide exactly the class of
error the test exists to catch.
"""

from __future__ import annotations

import torch

from src.models.deepseek_v4_1 import attention as attention_module
from src.models.deepseek_v4_1.attention import (
    AttentionStack,
    Indexer,
    get_window_topk_idxs,
    select_candidate_blocks,
)
from src.models.deepseek_v4_1.config import V41TextConfig
from src.models.deepseek_v4_1.decode_pos import Pos

# Toy geometry: six layers, two of them KV sources, three of them index sources, and the second KV
# source doubling as the candidate source so that one layer owns both published caches. The ratios
# mix 2 and 1 so that both compressor branches (softmax pooling and a plain projection) run.
TOY = dict(
    dim=64,
    n_layers=6,
    n_mtp_layers=0,
    n_heads=4,
    head_dim=32,
    rope_head_dim=8,
    q_lora_rank=32,
    o_groups=2,
    o_lora_rank=16,
    window_size=4,
    compress_ratios=(0, 0, 2, 2, 1, 1),
    kv_source_layers=(2, 4),
    index_source_layers=(2, 4, 5),
    index_n_heads=2,
    index_head_dim=32,
    # wide enough that no query is truncated: the widest reachable set is 12 positions (ratio 1)
    index_topk=16,
    candidate_source_layer=4,
    candidate_topk_blocks=16,
    candidate_block_size=2,
    norm_eps=1e-6,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    rope_factor=40.0,
    beta_fast=32,
    beta_slow=1,
    original_seq_len=512,
    max_position_embeddings=1024,
)

N_TOKENS = 12
WINDOW_SIZE = 4
# layer 5 has ratio 1, so it reaches one compressed position per token
LAST_INDEX_SOURCE = 5
LAST_RATIO = 1
# layer 4 is the candidate source of the toy config, and the only layer that publishes blocks
CANDIDATE_LAYER = 4


def _build(seed: int = 0) -> tuple[AttentionStack, V41TextConfig]:
    """A stack whose weights are finite and reproducible, and the config it was built from.

    The fill is not decoration. `AttentionStack` is built out of `torch.empty` because its real
    weights arrive from a checkpoint, so without this the tests would be comparing uninitialized
    memory: a NaN in a weight makes every `torch.equal` below false, and whether a given allocation
    holds one depends on what the process allocated and freed before it. That is not hypothetical --
    with this fill absent the file passes alone and fails when `test_models_deepseek_v4_1_config.py`
    runs first, which is a property of the allocator and not of the layer stack.
    """
    torch.manual_seed(seed)
    cfg = V41TextConfig(**TOY)
    stack = AttentionStack(cfg, max_batch_size=1, max_seq_len=64)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in stack.parameters():
            values = torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 0.2
            parameter.copy_(values.to(parameter.dtype))
    return stack, cfg


def _run_prefill(stack: AttentionStack, x: torch.Tensor) -> torch.Tensor:
    stack.reset_state(x.size(0))
    return stack(x, 0)


def _run_decode(stack: AttentionStack, x: torch.Tensor) -> torch.Tensor:
    stack.reset_state(x.size(0))
    return torch.cat([stack(x[:, pos : pos + 1], pos) for pos in range(x.size(1))], dim=1)


def test_prefill_and_token_by_token_decode_agree() -> None:
    stack, cfg = _build()
    x = torch.randn(1, N_TOKENS, cfg.dim, dtype=torch.bfloat16)

    prefill = _run_prefill(stack, x)
    decode = _run_decode(stack, x)

    assert prefill.shape == decode.shape == x.shape
    # not a tolerance: with no truncation the two orderings are the same arithmetic in a different
    # grouping, and bf16 rounding of a different grouping would already be a finding
    assert torch.equal(prefill, decode), f"max abs diff {(prefill.float() - decode.float()).abs().max().item()}"


def _offset_at(pos: int | None) -> int:
    """What `Attention.forward` hands the indexer as its offset: `kv.size(1)`, the number of KV
    positions the indices are relative to. That is the whole chunk during prefill and the ring
    during decode -- with one wrinkle, that a decode step at `start_pos == 0` takes the
    start-of-sequence branch of `_window_kv`, where the chunk is a single token. So the raw indices
    are not comparable between the two orderings; `raw - _offset_at(pos)` is."""
    return N_TOKENS if pos is None else (1 if pos == 0 else WINDOW_SIZE)


def test_both_orderings_select_exactly_the_reachable_compressed_positions() -> None:
    """The parity above could in principle survive two *different* index selections cancelling out.
    This checks the selection itself, on the underlying positions rather than the raw indices -- the
    two orderings hand `sparse_attn` different offsets (the whole chunk during prefill, the ring
    during decode), so the raw integers legitimately differ.

    Layer 5 has `compress_ratio` 1, so group `j` is token `j`: a query at position `p` has exactly
    `p + 1` groups closed, and must list precisely those, padded with `-1` to the width the top-k
    asked for. That the two orderings agree here is the shared-slot mechanism working -- the decode
    path's `topk_idxs` comes from an indexer reading the key cache layer 2/4 published, not from
    anything recomputed per step.
    """
    stack, cfg = _build()
    x = torch.randn(1, N_TOKENS, cfg.dim, dtype=torch.bfloat16)

    _run_prefill(stack, x)
    assert stack.shared.topk_idxs is not None
    prefill = stack.shared.topk_idxs.clone()
    assert prefill.size(1) == N_TOKENS

    stack.reset_state(1)
    decode = []
    for pos in range(N_TOKENS):
        stack(x[:, pos : pos + 1], pos)
        assert stack.shared.topk_idxs is not None
        decode.append(stack.shared.topk_idxs.clone())

    for pos in range(N_TOKENS):
        # every group that has closed is reachable, and the top-k is wide enough to keep all of them
        reachable = (pos + 1) // LAST_RATIO
        rows = {
            "prefill": (prefill[0, pos].tolist(), _offset_at(None)),
            "decode": (decode[pos][0, 0].tolist(), _offset_at(pos)),
        }
        for which, (row, offset) in rows.items():
            valid = sorted(int(v) - offset for v in row if int(v) >= 0)
            assert valid == list(range(reachable)), f"{which} position {pos}: {valid}"
            assert row.count(-1) == len(row) - reachable


def test_shared_slots_follow_the_layer_modes() -> None:
    stack, cfg = _build()
    owners = [layer.layer_id for layer in stack.layers if layer.indexer is not None and layer.indexer.owns_k]
    assert owners == list(cfg.kv_source_layers)
    assert [layer.is_kv_source for layer in stack.layers] == [
        layer.layer_id in cfg.kv_source_layers for layer in stack.layers
    ]
    assert [layer.is_index_source for layer in stack.layers] == [
        layer.layer_id in cfg.index_source_layers for layer in stack.layers
    ]

    candidate_source = cfg.candidate_source_layer
    assert [layer.indexer.is_candidate_source for layer in stack.layers if layer.indexer is not None] == [
        layer.layer_id == candidate_source for layer in stack.layers if layer.indexer is not None
    ]
    # exactly the index sources after the candidate source narrow their field first
    users = [layer.layer_id for layer in stack.layers if layer.indexer is not None and layer.indexer.uses_candidates]
    assert users == [layer_id for layer_id in cfg.index_source_layers if layer_id > candidate_source]


def test_reset_state_makes_two_forwards_independent() -> None:
    """Every test above relies on this: the reference is one conversation per process and never
    resets, so without it a second forward would silently continue the first."""
    stack, cfg = _build()
    x = torch.randn(1, N_TOKENS, cfg.dim, dtype=torch.bfloat16)

    first = _run_prefill(stack, x)
    second = _run_prefill(stack, x)
    assert torch.equal(first, second)


def _place_scores(indexer, scores: torch.Tensor, compress_lens: torch.Tensor):
    """Drive `_stream_prefix` with scores given directly rather than through the indexer's weights.

    One index head is given unit weight and the key cache is filled with the numbers wanted, so the
    score row the selection sees is exactly `scores`: `relu(dot(q_h, k_t)) * w_h` sums to `k_t[0]` when
    `q_h` is the first basis vector and `w_h` is 1. Everything else is zero and contributes nothing.
    A test that wanted these numbers out of random weights would have to invert a `relu` sum to name
    them, and the point of the digits below is that they are readable.
    """
    block_size = indexer.candidate_block_size
    seqlen = scores.size(-1)
    index_k = torch.zeros(1, seqlen, indexer.index_head_dim)
    index_k[0, :, 0] = scores
    q = torch.zeros(1, seqlen, indexer.n_heads, indexer.index_head_dim)
    q[..., 0, 0] = 1.0
    weights = torch.zeros(1, seqlen, indexer.n_heads)
    weights[..., 0] = 1.0
    return indexer._stream_prefix(q, weights, index_k, compress_lens, None, block_size)


def test_candidate_blocks_pin_the_open_block_and_drop_unreachable_ones() -> None:
    """Level one of the two-level top-k, through the tiling it actually runs in.

    Two things here are not a plain top-k. The block holding a query's newest reachable position is
    only partly filled, so it holds the most recent tokens and is *pinned*: it is kept even though
    every older block out-scores it. And a block the query cannot reach yet is `-inf`, so it is
    dropped even when `topk_blocks` would have had room for it.

    Eight queries are run and all eight get the same row -- they are handed the same scores and the
    same reachable count -- so the digits below are one query's answer `seqlen` times over.
    """
    indexer = Indexer(V41TextConfig(**TOY), CANDIDATE_LAYER, max_batch_size=1, max_seq_len=64)
    # one query per position, so the score row is `seqlen` queries of the same numbers
    rows = 8

    # 7 reachable positions in 4 blocks of 2, so block 3 is half filled. Scores say block 0 and 1 beat
    # it; the pin is what keeps it anyway, and `topk_blocks = 2` is what makes that matter.
    indexer.candidate_topk_blocks = 2
    scores = torch.tensor([4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 0.0, 0.0])
    _, _, blocks = _place_scores(indexer, scores, torch.full((rows, 1), 7))
    assert blocks.tolist() == [[[0, 3]] * rows]

    # 4 reachable positions, so the pinned block is 1; blocks 2 and 3 are past the reachable count and
    # come back padding even though `topk_blocks = 4` leaves room for all four.
    indexer.candidate_topk_blocks = 4
    scores = torch.tensor([2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    _, _, blocks = _place_scores(indexer, scores, torch.full((rows, 1), 4))
    assert blocks.tolist() == [[[-1, -1, 0, 1]] * rows]

    # The whole-axis form and the tiled one on one row: the tiling is a merge of per-tile top-k's, and
    # the pin is the one thing that makes the merge not commute with it, so this is the case worth
    # holding the two against each other on. Keys forced apart so the selection is not a tie.
    indexer.candidate_topk_blocks = 3
    scores = torch.tensor([5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 0.0, 0.0])
    _, _, blocks = _place_scores(indexer, scores, torch.full((rows, 1), 7))
    whole = select_candidate_blocks(
        scores.reshape(1, -1).unflatten(-1, (-1, indexer.candidate_block_size)).amax(dim=-1),
        torch.tensor([[7]]),
        topk_blocks=3,
        block_size=indexer.candidate_block_size,
    )
    assert torch.equal(blocks, whole.expand(rows, -1).unsqueeze(0)), f"{blocks.tolist()} vs {whole.tolist()}"


def test_window_topk_idxs_agree_across_the_int_and_device_paths() -> None:
    """The decode branch of `get_window_topk_idxs` became rotation, and the eager path kept the build.

    The graph path cannot call the old code -- it built the table on the CPU and copied it, which is
    a pageable H2D inside a capture -- so the branch is now arithmetic on the device. That makes the
    two paths two implementations of one function, which is exactly the shape that drifts. Both are
    therefore held against the literal `cat` they replaced, and against each other, at every position
    that matters: below the window, at the wrap, and past it.

    The ring is read oldest-first from `pos % win + 1`, and every slot past `P` is `-1`, so a
    position near the start of a sequence has a partly-filled ring and the two must still agree.
    """
    positions = [1, 2, 7, 8, 63, 127, 128, 129, 255, 1024, 1025, 4095, 4096]
    for window in (8, 128):
        for p in positions:
            oldest = p % window + 1
            literal = torch.cat([torch.arange(oldest, window), torch.arange(oldest)])
            literal = torch.where(literal > p, -1, literal).int().unsqueeze(0)

            eager = get_window_topk_idxs(window, 1, 1, p)
            assert torch.equal(eager, literal.expand(1, -1, -1).contiguous()), f"host path at {p}"

            if torch.cuda.is_available():
                device = get_window_topk_idxs(window, 1, 1, Pos.device(p, "cuda:0"))
                assert torch.equal(device.cpu(), eager), f"device path at {p}"


def test_window_topk_idxs_prefill_branch_is_untouched() -> None:
    """`Pos.first()` still routes position 0 to the chunk build, on both paths."""

    seqlen = 8
    end = torch.arange(seqlen).unsqueeze(1)
    literal = (end - 128 + 1).clamp(0) + torch.arange(min(seqlen, 128))
    literal = torch.where(literal > end, -1, literal).int().unsqueeze(0)

    assert torch.equal(get_window_topk_idxs(128, 1, seqlen, 0), literal.expand(1, -1, -1))
    if torch.cuda.is_available():
        assert torch.equal(
            get_window_topk_idxs(128, 1, seqlen, Pos.device(0, "cuda:0")).cpu(),
            literal.expand(1, -1, -1),
        )


def test_a_query_tile_smaller_than_the_chunk_keeps_every_query(monkeypatch) -> None:
    """The tiles cover both axes, and only the key axis can be merged into one buffer.

    `_TopKStream` merges along the key axis, so a second query tile pushed into the same buffer is
    merged as if it were more keys of the first: the buffer keeps the first tile's row count and
    every later query is dropped from the selection without a shape error anywhere. That is invisible
    for exactly as long as the chunk fits in one query tile, which the toy config's twelve tokens
    always do -- the real model's `INDEXER_CAND_QUERY_TILE` is 512, so a 1024-token prefill was the
    first length that did not, and it failed as a `torch.cat` of the two widths in `Attention`.

    Shrinking both tiles to four makes the twelve-token chunk span three query tiles. The tiled run
    has to agree with the untiled one in both the output and the published selection, and the widths
    have to stay the chunk's rather than the tile's -- an assertion on equality alone would pass
    again the moment anything downstream broadcast the short table back up.

    The third run shrinks the score budget instead, so the key axis splits too and the same query
    walks several key tiles of one prefix.
    """
    stack, cfg = _build()
    x = torch.randn(1, N_TOKENS, cfg.dim, dtype=torch.bfloat16)

    whole = _run_prefill(stack, x)
    assert stack.shared.topk_idxs is not None
    whole_idxs = stack.shared.topk_idxs.clone()

    monkeypatch.setattr(attention_module, "INDEXER_QUERY_TILE", 4)
    monkeypatch.setattr(attention_module, "INDEXER_CAND_QUERY_TILE", 4)
    stack, _ = _build()
    tiled = _run_prefill(stack, x)
    assert stack.shared.topk_idxs is not None
    tiled_idxs = stack.shared.topk_idxs

    assert tiled_idxs.shape == whole_idxs.shape
    assert tiled_idxs.size(1) == N_TOKENS
    assert torch.equal(tiled_idxs, whole_idxs)
    assert torch.equal(tiled, whole)

    # eight columns a key tile against a width of twelve: the local index head count times the query
    # tile times eight, which is what the budget divides
    monkeypatch.setattr(attention_module, "INDEXER_MIN_KEY_TILE", 1)
    monkeypatch.setattr(attention_module, "INDEXER_SCORE_BUDGET", TOY["index_n_heads"] * 4 * 8)
    stack, _ = _build()
    split = _run_prefill(stack, x)
    assert stack.shared.topk_idxs is not None
    assert stack.shared.topk_idxs.shape == whole_idxs.shape
    assert torch.equal(stack.shared.topk_idxs, whole_idxs)
    assert torch.equal(split, whole)


def test_a_partial_last_key_tile_does_not_widen_the_position_keys(monkeypatch) -> None:
    """The block stream pads the tail of the last key tile out to a whole block; the position stream
    must not inherit that pad.

    Both streams read the same `score`, and the pad is only there so a block is never split across
    two tiles. A position push that carried the padded score would arrive at `_TopKStream` one column
    wider than the `ids` it is picked with. That does not raise where it happens: the merge takes the
    top-k of the wider values and indexes the narrower keys, which stays in bounds until a tie makes
    it pick a pad column. So the invariant is asserted at the push instead of looked for in the
    result. Nine positions against a key tile of four and a block size of two ends on a tile of one,
    which is the geometry that has a pad to inherit.
    """
    monkeypatch.setattr(attention_module, "INDEXER_MIN_KEY_TILE", 1)
    width = 9
    indexer = Indexer(V41TextConfig(**TOY), CANDIDATE_LAYER, max_batch_size=1, max_seq_len=64)
    monkeypatch.setattr(attention_module, "INDEXER_SCORE_BUDGET", indexer.n_heads * width * 4)

    widths: list[int] = []
    original = attention_module._TopKStream.push

    def checked(self, values, keys):
        assert values.shape == keys.shape, "a stream picks its keys with its values"
        widths.append(values.size(-1))
        return original(self, values, keys)

    monkeypatch.setattr(attention_module._TopKStream, "push", checked)
    values, keys, blocks = _place_scores(indexer, torch.arange(width, dtype=torch.float32), width)

    # one column in the last tile, and a block of two to pad it out to
    assert widths[-1] == 1
    assert values.shape == keys.shape == (1, width, width)
    assert torch.equal(keys.sort(dim=-1).values, torch.arange(width).expand(1, width, width))
    assert blocks is not None and blocks.shape == (1, width, -(-width // indexer.candidate_block_size))
