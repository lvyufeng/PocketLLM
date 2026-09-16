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

from src.models.deepseek_v4_1.attention import AttentionStack, select_candidate_blocks
from src.models.deepseek_v4_1.config import V41TextConfig

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


def _build(seed: int = 0) -> tuple[AttentionStack, V41TextConfig]:
    torch.manual_seed(seed)
    cfg = V41TextConfig(**TOY)
    return AttentionStack(cfg, max_batch_size=1, max_seq_len=64), cfg


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


def test_select_candidate_blocks_pins_the_open_block_and_drops_unreachable_ones() -> None:
    # 7 reachable positions in 4 blocks of 2, so the newest block is half filled. It is pinned in
    # even though every older block out-scores it, because it holds the most recent tokens and
    # would otherwise be at the mercy of a partial amax.
    logits = torch.tensor([[4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 0.0, -torch.inf]])
    keep = select_candidate_blocks(logits, torch.tensor([[7]]), topk_blocks=2, block_size=2)
    assert keep.tolist() == [[True, True, False, False, False, False, True, True]]

    # Positions the query cannot reach yet are -inf, so their blocks score -inf and are dropped even
    # when topk_blocks would have room for them.
    unreachable = torch.tensor([[2.0, 2.0, 1.0, 1.0, -torch.inf, -torch.inf, -torch.inf, -torch.inf]])
    keep = select_candidate_blocks(unreachable, torch.tensor([[4]]), topk_blocks=4, block_size=2)
    assert keep.tolist() == [[True, True, True, True, False, False, False, False]]
