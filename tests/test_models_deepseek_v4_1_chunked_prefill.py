"""Does a prompt split into chunks produce what the same prompt in one forward produces?

`Backbone.forward(..., chunk=n)` exists because two activations a forward holds are linear in the
sequence length and neither is needed past the row it is computed on: the Hyper-Connections mixing
is `[s, hc_mult * dim]` fp32 -- 21 GiB at 256K -- and `main_hiddens` is another 8 GiB. Splitting the
prompt is what puts 256K inside a 22 GiB card, and it is only a *reasonable* way to do that if the
chunks compose into the same forward. That is what this file checks, and it is checked at the level
where the composing happens: the layers' caches. A chunked prompt leaves the sliding-window ring, the
compressed-KV cache, the index-key cache and the compressor's partial group in the same state a
one-shot prompt leaves them in, so the tokens that come out of the two are the same tokens.

Three things have to hold for that and each is a test below.

* **The window is a window, not a chunk.** A query in a later chunk sees `window_size` positions
  that reach back *past its own chunk*, into rows the earlier chunk wrote into the ring. The ring is
  written and read by absolute position (`slot = position % window_size`) on both paths, and the
  continuation body hands `sparse_attn` the ring and the chunk concatenated so that a position can
  be named by which half it fell in. The row that comes out is the same *positions* in the same
  *order* as the one-shot row -- oldest first, which is the order the prefill branch emits and the
  order the sparse attention's denominator sums in. Same vectors, same order, same sum.
* **A group is `compress_ratio` positions, not `compress_ratio` tokens of a chunk.** A chunk boundary
  lands inside a group in general, so the first tokens of a chunk close a group the previous chunk
  opened; those tokens pool *in the state*, where that group's earlier tokens are. A trailing partial
  group waits in the state for the next chunk to close it. The compressor's own state is compared
  after a chunked prompt and after a one-shot prompt for exactly this reason.
* **The indexer reaches what the query can reach.** `compress_lens` is counted from absolute
  position, and the reachable prefix is `pos.group(ratio, seqlen)` groups wide -- a chunk's first
  query can see the whole history in front of it, and a count that started at 0 would mask all of it.

The comparison is `torch.equal` and not a tolerance, at a chunk width chosen so nothing is
truncated: `index_topk` is wide enough for every reachable compressed position, so the two orderings
select the same `k` vectors and the difference between them is a different grouping of the same
additions rather than a different answer. That is the same standard
`test_models_deepseek_v4_1_attention.py` holds prefill and token-by-token decode to, and the same
reason a tolerance would be wrong here: an off-by-one in the ring would land inside a tolerance and
change the answer those cards produce.
"""

from __future__ import annotations

import pytest
import torch

from src.models.deepseek_v4_1.attention import AttentionStack, get_window_topk_idxs
from src.models.deepseek_v4_1.config import V41TextConfig
from src.models.deepseek_v4_1.decode_pos import Pos

# The same toy geometry `test_models_deepseek_v4_1_attention.py` uses, for the same reasons: two KV
# sources, three index sources, one layer owning both published caches, and ratios that run both
# compressor branches. The two numbers that matter *here* are the window and the group size: a
# window of 4 makes a chunk boundary land outside it, so the ring is actually read across chunks,
# and ratio 2 makes a chunk boundary land inside a group, so the compressor's carried partial group
# is actually exercised. A chunk of odd length does both without any arranging.
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
GROUP = 2
# The chunk widths below run from one token -- the decode path, which is the other ordering that has
# to agree -- up to the whole prompt in two. 5 and 7 put a boundary inside a group with tokens left
# over on both sides, 3 and 9 put one inside a group that the *same* chunk then closes, and 4 and 8
# land on the window.
CHUNKS = (1, 2, 3, 4, 5, 7, 8, 9, 11, 12)


def _build(seed: int = 0) -> tuple[AttentionStack, V41TextConfig]:
    """A stack whose weights are finite and reproducible, and the config it was built from.

    `AttentionStack` is built out of `torch.empty` because its real weights arrive from a checkpoint,
    so without the fill the comparisons below would be of whatever the allocator last handed out --
    and a NaN in a weight makes every `torch.equal` false.
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


def _run(stack: AttentionStack, x: torch.Tensor, chunk: int) -> torch.Tensor:
    """Prefill `x` at `chunk` tokens a forward, exactly as `Backbone.forward` drives it.

    The `start_pos` of each chunk is its own offset into the prompt, which is the whole of what the
    continuation bodies key off -- a chunk past zero is not a decode step, and which of the two a
    forward is is decided by the count of queries rather than by the position.
    """
    stack.reset_state(x.size(0))
    out = []
    for c0 in range(0, x.size(1), chunk):
        stop = min(c0 + chunk, x.size(1))
        out.append(stack(x[:, c0:stop], c0))
    return torch.cat(out, dim=1)


def _states(
    stack: AttentionStack,
    total: int,
    suffixes: tuple[str, ...] = (
        "window_kv_cache",
        "compress_kv_cache",
        "kv_state",
        "score_state",
        "k_cache",
    ),
) -> dict[str, torch.Tensor]:
    """Every cache a chunk leaves behind, keyed by the buffer's own name.

    Named off the module tree rather than listed by hand so that a cache added to `Attention` later
    is compared without this file being edited -- the failure this file exists to catch is a cache
    that a chunk forgot to carry, and a hand-written list is exactly how one goes unnoticed.

    Two of these are compared only over the rows a future forward can read: `kv_state` and
    `score_state` reserve a slot per position of a group and hold the group that is still filling,
    so after a prompt of `total` positions only the first `total % compress_ratio` rows are that
    group. The rest are rows of groups already pooled and emitted, and a forward never reads them
    again -- the step that completes a group is the step that writes the group's last slot, so every
    slot is rewritten before the next read. They are *not* equal between a chunked and a one-shot
    prompt and nothing depends on their being equal, which is why comparing them would be a test
    that fails for a reason the model does not care about.
    """
    ratios = {
        name: module.compress_ratio
        for name, module in stack.named_modules()
        if hasattr(module, "compress_ratio")
    }
    found: dict[str, torch.Tensor] = {}
    for name, buffer in stack.named_buffers():
        if not name.endswith(suffixes):
            continue
        if name.endswith(("kv_state", "score_state")):
            ratio = ratios[name.rsplit(".", 1)[0]]
            buffer = buffer[:, : total % ratio]
        found[name] = buffer.clone()
    return found


def _compare(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], what: str) -> None:
    assert left.keys() == right.keys()
    for name in left:
        a, b = left[name], right[name]
        assert torch.equal(a, b), (
            f"{what}: `{name}` differs, max abs diff {(a.float() - b.float()).abs().max().item()}"
        )


def test_chunked_prefill_leaves_the_caches_a_one_shot_prefill_leaves() -> None:
    """The property 256K rests on: the chunks compose.

    Every width in `CHUNKS` is run against the same one-shot prefill, on a fresh pair of stacks built
    from one seed, so the only thing that differs between an arm and the reference is where the
    forward boundaries fell. Both the output stream and every cache are compared.
    """
    _, cfg = _build()
    x = torch.randn(1, N_TOKENS, cfg.dim, dtype=torch.bfloat16)

    whole, _ = _build()
    reference = _run(whole, x, N_TOKENS)
    reference_states = _states(whole, N_TOKENS)

    for chunk in CHUNKS:
        stack, _ = _build()
        out = _run(stack, x, chunk)
        assert out.shape == reference.shape
        assert torch.equal(out, reference), (
            f"chunk {chunk}: max abs diff {(out.float() - reference.float()).abs().max().item()}"
        )
        _compare(_states(stack, N_TOKENS), reference_states, f"chunk {chunk}")


def test_a_chunk_of_one_token_is_the_decode_path_and_not_the_continuation_one() -> None:
    """`chunk=1` is not `chunk>1` with a boundary every token, it is the decode body.

    `_is_continuation` turns on the count of queries, so a one-token forward at a position past zero
    runs the decode bodies -- which is what a caller stepping a prompt through one token at a time
    already got and what `test_models_deepseek_v4_1_attention.py` already pins. It is in `CHUNKS`
    above for that reason and called out here, because a change to `_is_continuation` that made a
    one-token forward take the continuation body would keep that test passing and quietly move the
    capture path's arithmetic.
    """
    _, cfg = _build()
    x = torch.randn(1, N_TOKENS, cfg.dim, dtype=torch.bfloat16)

    whole, _ = _build()
    reference = _run(whole, x, N_TOKENS)

    stack, _ = _build()
    stack.reset_state(1)
    stepped = torch.cat([stack(x[:, pos : pos + 1], pos) for pos in range(N_TOKENS)], dim=1)
    assert torch.equal(stepped, reference)


def test_the_ring_holds_the_last_window_positions_after_a_chunked_prompt() -> None:
    """The ring is a function of the prompt, not of how the prompt was cut.

    A chunk longer than the window puts only its own last `window_size` rows into the ring, and a
    chunk shorter than the window puts all of them while leaving the older slots alone -- both are
    read back by absolute position, so both have to land on the slots a one-shot prefill put the same
    positions in. This is the assertion `_window_kv`'s continuation body and `get_window_topk_idxs`'
    slot arithmetic have to agree on, and they are two different expressions over the same ring.
    """
    _, cfg = _build()
    x = torch.randn(1, N_TOKENS, cfg.dim, dtype=torch.bfloat16)

    whole, _ = _build()
    _run(whole, x, N_TOKENS)
    expected = {name: buf.clone() for name, buf in whole.named_buffers() if name.endswith("window_kv_cache")}
    assert expected

    for chunk in CHUNKS:
        stack, _ = _build()
        _run(stack, x, chunk)
        found = {name: buf for name, buf in stack.named_buffers() if name.endswith("window_kv_cache")}
        _compare(found, expected, f"chunk {chunk}")


def test_the_compressor_closes_a_group_a_chunk_boundary_cut_in_half() -> None:
    """A group is `compress_ratio` consecutive *positions*, and a boundary does not move it.

    Run at the widths that cut a group and leave the far half in the chunk before, in the chunk
    after, and in both -- a boundary inside a group is the case the continuation body exists for, and
    the three placements take three different paths through it: the carried tokens complete a group
    in the state, the groups wholly inside the chunk pool directly, and the trailing partial group
    waits in the state. The compressed-KV cache is what the three of them produce, and it has to be
    the six latents the one-shot forward produced, in the same columns.
    """
    _, cfg = _build()
    x = torch.randn(1, N_TOKENS, cfg.dim, dtype=torch.bfloat16)
    # 5 cuts a group with the earlier half in the previous chunk and the later half in this one; 3 and
    # 9 cut one that the chunk containing the *later* half then closes on its own; 7 does both kinds
    # of cut in one prompt.
    cutting = (3, 5, 7, 9, 11)

    whole, _ = _build()
    _run(whole, x, N_TOKENS)
    expected = _states(whole, N_TOKENS, ("compress_kv_cache", "kv_state", "score_state", "k_cache"))
    assert expected

    for chunk in cutting:
        stack, _ = _build()
        _run(stack, x, chunk)
        found = _states(stack, N_TOKENS, ("compress_kv_cache", "kv_state", "score_state", "k_cache"))
        _compare(found, expected, f"chunk {chunk}")


def _absolute(idxs: torch.Tensor, row: int, chunk_start: int, window_size: int) -> list[int]:
    """A continuation row's columns as absolute positions, oldest first.

    The row is read out of two tensors, so a column below `window_size` is a ring slot and a column
    at or above it is an offset into the chunk. The query's own position is the chunk's start plus
    its row, and it is what a slot is turned back into a position against; that the ring's rotation
    uses the *chunk's* start rather than the query's is the whole of what makes the two halves one
    contiguous run of positions.
    """
    pos = chunk_start + row
    out = []
    for column in idxs[0, row].tolist():
        if column < 0:
            continue
        if column >= window_size:
            out.append(chunk_start + column - window_size)
        else:
            out.append(pos - ((pos - column) % window_size))
    return out


def test_a_continuation_row_lists_the_same_positions_as_the_prefill_row() -> None:
    """The two window expressions, compared where they can be compared: on positions.

    `get_window_topk_idxs` builds a prefill row as chunk offsets and a continuation row as ring slots
    below `window_size` and chunk offsets above it. They are different expressions, they index
    different halves of a concatenated K, and the only reason the model can use both is that they
    name the same positions in the same order. A window that lost its rotation -- the classic ring
    bug -- changes only the continuation expression, and the end-to-end tests above would catch it as
    a diff they cannot explain; this one says where.

    The comparison is run at every query position of a chunk that starts past the window, so the
    rows where the ring is partly unwritten (`pos < window_size`) are covered along with the full
    ones, and at both the widths that keep a boundary inside a group and outside it.
    """
    seqlen = 6
    for start in (WINDOW_SIZE, WINDOW_SIZE + 3, 12):
        for chunk in (seqlen, seqlen - 1, seqlen - 3):
            continuation = get_window_topk_idxs(WINDOW_SIZE, 1, chunk, Pos(start))
            # the same rows as a one-shot prefill would emit them, by absolute position: the chunk's
            # own columns are `start + j`, and the ring's are the slots those positions live in
            for row in range(chunk):
                at = start + row
                wanted = list(range(max(0, at - WINDOW_SIZE + 1), at + 1))
                assert _absolute(continuation, row, start, WINDOW_SIZE) == wanted, (
                    f"start {start}, chunk {chunk}, row {row}: "
                    f"{_absolute(continuation, row, start, WINDOW_SIZE)} != {wanted}"
                )


def test_a_chunk_wider_than_the_window_still_names_only_the_window() -> None:
    """A chunk of 512 into a ring of 128: the rows past the window may not name a ring slot.

    The continuation body hands the whole chunk to `sparse_attn` beside the ring, so a row's columns
    above `window_size` are its own chunk -- and a row that named a *ring* slot it does not own would
    be reading a position another query in the same chunk is about to overwrite. The partition is
    what stops that: below `window_size` is the ring and above it is the chunk, and the two halves
    are read as one contiguous run of positions by `_absolute` below.
    """
    window_size, chunk, start = 4, 9, 7
    idxs = get_window_topk_idxs(window_size, 1, chunk, Pos(start))
    for row in range(chunk):
        at = start + row
        named = [c for c in idxs[0, row].tolist() if c >= 0]
        # a row never names more positions than the window holds, and never the same one twice
        assert len(named) == len(set(named)) <= window_size
        # and they are exactly the reachable window, oldest first
        assert _absolute(idxs, row, start, window_size) == list(range(max(0, at - window_size + 1), at + 1))


@pytest.mark.parametrize("chunk", CHUNKS)
def test_every_chunk_width_leaves_the_same_compressed_cache(chunk: int) -> None:
    """The parametrized form of the cache comparison, one width per test id.

    The loop above reports the first width that broke and stops; this one reports each width on its
    own, which is what makes the failure legible when a boundary rule is changed -- the widths that
    cut a group and the widths that do not fail under different edits.
    """
    _, cfg = _build()
    x = torch.randn(1, N_TOKENS, cfg.dim, dtype=torch.bfloat16)

    whole, _ = _build()
    reference = _run(whole, x, N_TOKENS)
    reference_states = _states(whole, N_TOKENS)

    stack, _ = _build()
    out = _run(stack, x, chunk)
    assert torch.equal(out, reference)
    _compare(_states(stack, N_TOKENS), reference_states, f"chunk {chunk}")
