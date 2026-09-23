"""One layer's attention on the card: the arithmetic, the bounds, the cache, the release.

Everything here is one of two questions. The first is whether the device attention is
the reference's attention, which is answered against `layers.py` rather than against a
second copy of the same idea: the reference is the oracle, and a test that recomputes
what it is testing only tests its own arithmetic. The synthetic half runs without the
checkpoint and covers the places where two readings of the model differ -- the sink as
a column against the sink as a bias, the window against the prefix, the folded query
groups against `repeat_kv` -- because those are silent: every one of them produces a
tensor of the right shape and a plausible number.

The second question is whether the cache is a cache. A ring that wraps has to return
the *last* window in time order, a decode step has to read back what a prefill stored,
and a chunked prefill has to agree with a one-shot one -- to the rounding of its own
summations, since the gemms are not the same shape, but to nothing else. The regression
that motivates the last test in the file is worth naming because no single-call
comparison can see it: the online softmax's running maximum once aliased a one-query
sink, so a call wrote into the layer's own parameter and the *next* call was wrong.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from src.models.mimo_v2.device_attention import (  # noqa: E402
    FOLD_KEYS,
    MimoV2DeviceAttention,
    MimoV2KVCache,
    attention,
    blocked_attention,
    fused_qkv_row_order,
    single_pass_attention,
)
from src.models.mimo_v2.layers import (  # noqa: E402
    MimoV2DecoderLayer,
    attention as host_attention,
    build_attention_masks,
    repeat_kv,
    split_fused_qkv,
)
from src.models.mimo_v2.loader import MimoV2Checkpoint  # noqa: E402
from src.models.mimo_v2.weights import layer_weights_from_checkpoint  # noqa: E402

RELEASE = os.environ.get("POCKETLLM_MIMO_CHECKPOINT", "/mnt/data3/MiMo-V2.6-Flash-RL")
HAS_RELEASE = os.path.isfile(os.path.join(RELEASE, "config.json"))

needs_release = pytest.mark.skipif(
    not HAS_RELEASE, reason=f"MiMo-V2.6 checkpoint not present at {RELEASE}"
)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
needs_release_cuda = pytest.mark.skipif(
    not (HAS_RELEASE and torch.cuda.is_available()),
    reason="the device attention needs both the release and a CUDA device",
)

#: A windowed layer and a global one: 2 carries a sink, 5 does not, and their fused
#: projections are not even the same width.
SWA_LAYER, GA_LAYER = 2, 5


def chunk_bounds(queries: int, keys: int, window: int | None = None, *, device: str = "cuda"):
    """The bounds of a chunk of `queries` keys whose last query is the last key.

    `window` of `None` is a prefix, not a mask: every query sees every key below it.
    """
    upper = torch.arange(keys - queries, keys, device=device)
    if window is None:
        lower = torch.zeros_like(upper)
    else:
        lower = (upper - window + 1).clamp_min(0)
    return lower, upper


def full_bounds(queries: int, keys: int, *, device: str = "cuda"):
    """Every query sees every key, which is what the reference's `None` mask stands for."""
    return (
        torch.zeros(queries, dtype=torch.int64, device=device),
        torch.full((queries,), keys - 1, dtype=torch.int64, device=device),
    )


# ---------------------------------------------------------------------------
# The softmax, against the reference's own
# ---------------------------------------------------------------------------


@needs_cuda
def test_the_device_softmax_is_the_reference_softmax():
    """Full attention, no window, no sink: the host function is the oracle.

    `layers.attention` repeats the key heads and concatenates the sink column, so this
    agrees with it only if the fold, the bounds and the online maximum are all right at
    once. The `lower`/`upper` pair is what the mask would have been.
    """
    torch.manual_seed(0)
    heads, kv_heads, queries, keys, dim, vdim = 8, 2, 6, 9, 12, 5
    query = torch.randn(heads, queries, dim, device="cuda") * 0.5
    key = torch.randn(kv_heads, keys, dim, device="cuda") * 0.5
    value = torch.randn(kv_heads, keys, vdim, device="cuda") * 0.5
    sink = torch.randn(heads, device="cuda")
    scaling = dim**-0.5
    lower, upper = full_bounds(queries, keys)

    want, _ = host_attention(
        query.unsqueeze(0),
        key.unsqueeze(0),
        value.unsqueeze(0),
        None,
        scaling,
        heads // kv_heads,
        sink,
    )
    want = want[0].transpose(0, 1)

    got, stats = single_pass_attention(query, key, value, lower, upper, scaling=scaling, sink=sink)
    assert stats.path == "single"
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max().item()

    looped, _ = blocked_attention(query, key, value, lower, upper, scaling=scaling, sink=sink, block=4)
    assert torch.allclose(looped, want, atol=1e-5), (looped - want).abs().max().item()


@needs_cuda
def test_the_sink_is_a_column_and_not_a_bias():
    """Falsifies the additive reading, which is shape-correct and wrong.

    An additive sink shifts every logit and cancels in the softmax -- it would change
    nothing at all -- while the reference's sink takes part in the row maximum and in
    the denominator. The two halves below are that difference: a sink that wins takes
    the row, and a sink that loses is invisible.
    """
    torch.manual_seed(1)
    heads, kv_heads, queries, keys, dim, vdim = 4, 1, 1, 7, 8, 4
    query = torch.randn(heads, queries, dim, device="cuda")
    key = torch.randn(kv_heads, keys, dim, device="cuda")
    value = torch.randn(kv_heads, keys, vdim, device="cuda")
    lower, upper = chunk_bounds(queries, keys, keys)

    plain, _ = single_pass_attention(query, key, value, lower, upper, scaling=1.0, sink=None)
    assert plain.abs().max() > 0

    winning, _ = single_pass_attention(
        query, key, value, lower, upper, scaling=1.0, sink=torch.full((heads,), 40.0, device="cuda")
    )
    assert winning.abs().max() < plain.abs().max() * 1e-3, (
        "a sink of 40 logits has to take almost every row's mass",
        winning.abs().max().item(),
        plain.abs().max().item(),
    )

    losing, _ = single_pass_attention(
        query, key, value, lower, upper, scaling=1.0, sink=torch.full((heads,), -40.0, device="cuda")
    )
    assert torch.allclose(losing, plain, atol=1e-6)


@needs_cuda
def test_the_bounds_are_the_only_thing_keeping_a_key_out():
    """A query's output depends on exactly `lower[i] .. upper[i]` and on nothing else."""
    torch.manual_seed(2)
    heads, kv_heads, queries, keys, dim, vdim = 4, 2, 3, 10, 8, 4
    query = torch.randn(heads, queries, dim, device="cuda")
    key = torch.randn(kv_heads, keys, dim, device="cuda")
    value = torch.randn(kv_heads, keys, vdim, device="cuda")
    lower, upper = chunk_bounds(queries, keys, 4)

    got, _ = single_pass_attention(query, key, value, lower, upper, scaling=1.0)

    # Poison every key the bounds exclude, on both sides of the product, so a leak
    # shows up in the output rather than in a number nobody reads.
    keep = torch.zeros(keys, dtype=torch.bool, device="cuda")
    for row in range(queries):
        keep[int(lower[row]) : int(upper[row]) + 1] = True
    assert not bool(keep.all()) and bool(keep.any())
    poisoned_key, poisoned_value = key.clone(), value.clone()
    poisoned_key[:, ~keep] = 1e4
    poisoned_value[:, ~keep] = 1e4
    clean, _ = single_pass_attention(
        query, poisoned_key, poisoned_value, lower, upper, scaling=1.0
    )
    assert torch.allclose(clean, got, atol=1e-4), (clean - got).abs().max().item()

    # ... and a bound that drops one key that was visible does change it, so the
    # poisoning above is not simply unreachable.
    tighter, _ = single_pass_attention(
        query, key, value, (lower + 1).clamp_max(upper), upper, scaling=1.0
    )
    assert not torch.allclose(tighter, got, atol=1e-3)


@needs_cuda
def test_the_block_loop_and_the_single_pass_agree():
    """Two implementations of one softmax: one bounded by memory, one by launch count."""
    for heads, kv_heads, queries, keys, dim, vdim, has_sink in (
        (8, 2, 5, 11, 8, 4, True),
        (8, 2, 5, 11, 8, 4, False),
        (4, 1, 1, 40, 8, 4, True),
    ):
        torch.manual_seed(3)
        query = torch.randn(heads, queries, dim, device="cuda") * 0.5
        key = torch.randn(kv_heads, keys, dim, device="cuda") * 0.5
        value = torch.randn(kv_heads, keys, vdim, device="cuda") * 0.5
        sink = torch.randn(heads, device="cuda") if has_sink else None
        lower, upper = chunk_bounds(queries, keys, 7)

        one, single_stats = single_pass_attention(
            query, key, value, lower, upper, scaling=0.5, sink=sink
        )
        looped, block_stats = blocked_attention(
            query, key, value, lower, upper, scaling=0.5, sink=sink, block=3
        )
        assert torch.allclose(one, looped, atol=1e-5), (one - looped).abs().max().item()
        assert single_stats.pairs == heads * queries * keys
        assert block_stats.full_pairs == heads * queries * keys
        assert block_stats.blocks >= 1


@needs_cuda
def test_the_queries_are_grouped_the_way_repeat_kv_groups_them():
    """The fold assumes head `kv * groups + j` is key head `kv`; `repeat_kv` says so."""
    torch.manual_seed(4)
    heads, kv_heads, queries, keys, dim, vdim = 6, 2, 2, 4, 4, 3
    assert keys <= FOLD_KEYS, "this test is about the folded product"
    query = torch.randn(heads, queries, dim, device="cuda")
    key = torch.randn(kv_heads, keys, dim, device="cuda")
    value = torch.randn(kv_heads, keys, vdim, device="cuda")
    lower, upper = chunk_bounds(queries, keys, None)

    folded, _ = single_pass_attention(query, key, value, lower, upper, scaling=1.0)

    # The same product with the key heads materialised the reference's way, one key
    # head a query head. Same numbers, six times the bytes.
    expanded, stats = single_pass_attention(
        query,
        repeat_kv(key.unsqueeze(0), heads // kv_heads)[0],
        repeat_kv(value.unsqueeze(0), heads // kv_heads)[0],
        lower,
        upper,
        scaling=1.0,
    )
    assert stats.pairs == heads * queries * keys
    assert torch.allclose(folded, expanded, atol=1e-5), (folded - expanded).abs().max().item()

    # And a different grouping is a different answer, which is what makes the above
    # a comparison rather than an identity: head 0 takes the *other* key head.
    swapped, _ = single_pass_attention(query, key.flip(0), value.flip(0), lower, upper, scaling=1.0)
    assert not torch.allclose(folded, swapped, atol=1e-3)


@needs_cuda
def test_the_bounds_are_checked_rather_than_trusted():
    torch.manual_seed(5)
    query = torch.randn(4, 2, 8, device="cuda")
    key = torch.randn(2, 5, 8, device="cuda")
    value = torch.randn(2, 5, 4, device="cuda")
    lower, upper = chunk_bounds(2, 5, 5)

    with pytest.raises(ValueError, match="allowed key"):
        single_pass_attention(query, key, value, lower, upper + 3, scaling=1.0)
    with pytest.raises(ValueError, match="below its lower bound"):
        single_pass_attention(query, key, value, upper + 1, upper, scaling=1.0)
    with pytest.raises(ValueError, match="empty key tensor"):
        single_pass_attention(query, key[:, :0], value[:, :0], lower, upper, scaling=1.0)
    with pytest.raises(ValueError, match="do not group"):
        single_pass_attention(query[:3], key, value, lower, upper, scaling=1.0)


# ---------------------------------------------------------------------------
# The row order
# ---------------------------------------------------------------------------


@needs_release
def test_reordering_the_projection_is_the_same_as_cutting_its_output(release):
    """`fused_qkv_row_order` against `split_fused_qkv`: two descriptions of one layout.

    Both come from the same sentence about the released checkpoint, so neither is
    evidence alone; agreeing is what makes either usable.
    """
    for layer_idx in (SWA_LAYER, GA_LAYER):
        shape = release.layer.attention(layer_idx)
        order = fused_qkv_row_order(shape)
        assert sorted(order.tolist()) == list(range(shape.qkv_out))

        generator = torch.Generator().manual_seed(6)
        rows = torch.randn(shape.qkv_out, generator=generator)
        want = split_fused_qkv(rows.unsqueeze(0), shape, shape.qkv_row_layout)
        got = split_fused_qkv(rows[order].unsqueeze(0), shape, "contiguous")
        for a, b in zip(got, want):
            assert torch.equal(a, b), layer_idx


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


@needs_release
def test_the_cache_holds_a_window_for_a_windowed_layer_and_the_context_for_a_global_one(release):
    window = release.layer.attention(SWA_LAYER).sliding_window
    cache = MimoV2KVCache(
        release.layer, 8192, [SWA_LAYER, GA_LAYER], device="cpu", dtype=torch.bfloat16
    )
    assert cache.slots(SWA_LAYER) == window
    assert cache.slots(GA_LAYER) == 8192
    assert cache.context_capacity == 8192
    assert cache.length(SWA_LAYER) == 0 and cache.written(GA_LAYER) == 0

    expected = 0
    for layer_idx, slots in ((SWA_LAYER, window), (GA_LAYER, 8192)):
        shape = release.layer.attention(layer_idx)
        expected += shape.num_kv_heads * slots * (shape.head_dim + shape.v_head_dim) * 2
    assert cache.memory_bytes == expected

    cache.reset()
    assert cache.written(SWA_LAYER) == 0


@needs_release_cuda
def test_a_ring_returns_the_last_window_in_time_order(release):
    """The ring wraps, and what comes out is the *newest* positions, in time order."""
    window = release.layer.attention(SWA_LAYER).sliding_window
    shape = release.layer.attention(SWA_LAYER)
    cache = MimoV2KVCache(release.layer, 4096, [SWA_LAYER], device="cuda", dtype=torch.float32)

    written = window * 3 + 7
    positions = torch.arange(written, device="cuda").float()
    key = positions.reshape(1, written, 1).expand(shape.num_kv_heads, written, shape.head_dim)
    value = torch.ones(shape.num_kv_heads, written, shape.v_head_dim, device="cuda")
    cache.append(SWA_LAYER, key, value)

    got, _, length = cache.prefix(SWA_LAYER, written)
    assert length == window
    want = positions[written - window :]
    assert torch.equal(got[0, :, 0], want), (got[0, :8, 0], want[:8])
    assert cache.written(SWA_LAYER) == written
    assert cache.length(SWA_LAYER) == window

    # A chunk longer than the ring is trimmed to the ring rather than refused: only the
    # newest `window` positions can reach a query either way.
    cache.append(SWA_LAYER, key, value)
    assert cache.written(SWA_LAYER) == 2 * written
    assert cache.length(SWA_LAYER) == window


@needs_release_cuda
def test_a_global_cache_refuses_to_run_past_its_capacity(release):
    shape = release.layer.attention(GA_LAYER)
    cache = MimoV2KVCache(release.layer, 64, [GA_LAYER], device="cuda", dtype=torch.float32)
    key = torch.zeros(shape.num_kv_heads, 64, shape.head_dim, device="cuda")
    value = torch.zeros(shape.num_kv_heads, 64, shape.v_head_dim, device="cuda")
    cache.append(GA_LAYER, key, value)
    with pytest.raises(ValueError, match="runs past it"):
        cache.append(GA_LAYER, key[:, :1], value[:, :1])
    with pytest.raises(ValueError, match="has not appended"):
        cache.prefix(GA_LAYER, 65)


@needs_release_cuda
def test_a_cache_does_not_answer_for_a_layer_it_does_not_hold(release):
    shape = release.layer.attention(GA_LAYER)
    cache = MimoV2KVCache(release.layer, 64, [SWA_LAYER], device="cuda", dtype=torch.float32)
    with pytest.raises(KeyError, match="not in this cache"):
        cache.prefix(GA_LAYER, 0)
    with pytest.raises(KeyError, match="not in this cache"):
        cache.append(
            GA_LAYER,
            torch.zeros(shape.num_kv_heads, 1, shape.head_dim, device="cuda"),
            torch.zeros(shape.num_kv_heads, 1, shape.v_head_dim, device="cuda"),
        )
    with pytest.raises(ValueError, match="one value a key"):
        cache.append(
            SWA_LAYER,
            torch.zeros(shape.num_kv_heads, 1, shape.head_dim, device="cuda"),
            torch.zeros(shape.num_kv_heads, 2, shape.v_head_dim, device="cuda"),
        )


# ---------------------------------------------------------------------------
# The released layer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def release() -> MimoV2Checkpoint:
    return MimoV2Checkpoint(RELEASE)


def cache_for(release: MimoV2Checkpoint, layer_idx: int, capacity: int, **kwargs) -> MimoV2KVCache:
    return MimoV2KVCache(release.layer, capacity, [layer_idx], device="cuda", **kwargs)


def host_attend(layer: MimoV2DecoderLayer, hidden: torch.Tensor, window) -> dict:
    masks = build_attention_masks(hidden.shape[1], window, hidden.dtype)
    key = "sliding_window_attention" if window is not None else "full_attention"
    return layer.attend(hidden, masks[key])


@needs_release_cuda
@pytest.mark.parametrize("layer_idx", (SWA_LAYER, GA_LAYER))
def test_the_released_layer_is_the_host_layer(release, layer_idx):
    """The whole attention block against the reference, in float32, on the release.

    The device dequantizes the same FP8 projection the same way, so what is left is the
    attention itself: the row order, the value scale, the partial rope, the window, the
    sink and the output projection. Measured agreement is around 1e-5 of the peak, which
    is the reassociation of the same float32 sums.
    """
    sequence = 96
    shape = release.layer.attention(layer_idx)
    host = MimoV2DecoderLayer(
        release.layer, layer_idx, layer_weights_from_checkpoint(release, layer_idx, torch.float32, "cpu")
    )
    device = MimoV2DeviceAttention(release, layer_idx, "cuda", torch.float32)
    assert torch.equal(host.weights.qkv_proj, device.qkv_proj.cpu()), (
        "the two paths do not even read the same projection"
    )

    torch.manual_seed(7)
    hidden = torch.randn(1, sequence, release.layer.hidden_size) * 0.5
    reference = host_attend(host, hidden, shape.sliding_window)
    out = device.forward(hidden[0].cuda())

    for key in ("attn_out_pre_o", "attn_out_post_o"):
        want = reference[key].reshape(sequence, -1)
        got = out[key].float().cpu()
        peak = want.abs().max().item()
        assert (want - got).abs().max().item() < 1e-4 * max(peak, 1.0), (
            key,
            (want - got).abs().max().item(),
            peak,
        )


@needs_release_cuda
def test_the_windowed_layer_skips_the_key_blocks_it_cannot_see(release):
    """A window of 128 inside a 4096-token chunk is worth about four times the work.

    A causal layer with no window visits every key block for every query row above the
    diagonal, which is the triangle a flash attention saves half of. A windowed layer
    visits a block for the rows that can *reach* it, which is the window plus the block
    and no more -- so the pair count should be a quarter of the dense one, and a reader
    that ignored the window would report the dense one.
    """
    sequence = 4096
    device = MimoV2DeviceAttention(release, SWA_LAYER, "cuda", torch.bfloat16)
    torch.manual_seed(12)
    hidden = torch.randn(
        sequence, release.layer.hidden_size, device="cuda", dtype=torch.bfloat16
    ) * 0.5
    out = device.forward(hidden)
    stats = device.last_stats
    assert stats.path == "blocked", stats.path
    assert stats.pairs < stats.full_pairs / 3, (stats.pairs, stats.full_pairs)
    assert out["attn_out_post_o"].shape == (sequence, release.layer.hidden_size)


@needs_release_cuda
@pytest.mark.parametrize("layer_idx", (SWA_LAYER, GA_LAYER))
def test_a_chunked_prefill_is_a_one_shot_prefill(release, layer_idx):
    """The cache in the middle must not be a source of difference.

    It is not nothing: the gemms are not the same shape, so the float32 sums reassociate
    and the answers differ by their rounding. What it must not be is large, and the
    tolerance here is far tighter than the smallest thing a wrong key, a wrong window or
    a stale slot produces.
    """
    sequence = 96
    device = MimoV2DeviceAttention(release, layer_idx, "cuda", torch.float32)
    torch.manual_seed(8)
    hidden = torch.randn(1, sequence, release.layer.hidden_size) * 0.5
    one_shot = device.forward(hidden[0].cuda())["attn_out_post_o"]

    pieces, start = [], 0
    cache = cache_for(release, layer_idx, sequence + 4096, dtype=torch.float32)
    for size in (1, 7, 16, 40, 32):
        pieces.append(
            device.forward(hidden[0, start : start + size].cuda(), start_pos=start, cache=cache)[
                "attn_out_post_o"
            ]
        )
        start += size
    assert start == sequence
    chunked = torch.cat(pieces, dim=0)
    assert torch.allclose(one_shot, chunked, atol=1e-4), (one_shot - chunked).abs().max().item()


@needs_release_cuda
@pytest.mark.parametrize("layer_idx", (SWA_LAYER, GA_LAYER))
def test_a_decode_step_reads_what_the_prefill_stored(release, layer_idx):
    """One token against the cache, against the same token inside the full prefix.

    This is the shape of a real decode: the prefix is one call and the token is another,
    and the token's answer has to come out of the cache rather than out of the chunk it
    arrived in.
    """
    prefix, steps = 40, 8
    device = MimoV2DeviceAttention(release, layer_idx, "cuda", torch.float32)
    torch.manual_seed(9)
    hidden = torch.randn(1, prefix + steps, release.layer.hidden_size) * 0.5

    whole = device.forward(hidden[0].cuda())["attn_out_post_o"]
    cache = cache_for(release, layer_idx, prefix + 4096, dtype=torch.float32)
    device.forward(hidden[0, :prefix].cuda(), start_pos=0, cache=cache)
    stepped = torch.cat(
        [
            device.forward(
                hidden[0, prefix + i : prefix + i + 1].cuda(), start_pos=prefix + i, cache=cache
            )["attn_out_post_o"]
            for i in range(steps)
        ],
        dim=0,
    )
    assert torch.allclose(whole[prefix:], stepped, atol=1e-4), (
        whole[prefix:] - stepped
    ).abs().max().item()


@needs_release_cuda
def test_a_windowed_layer_does_not_read_past_its_window_and_a_global_one_does(release):
    """The window is a window: poison the oldest keys and only one of the two moves.

    Two layers of one model, the same prefix length and the same edit -- the first 100
    positions written from different hidden states. The global layer's decode step has
    to change, because those are keys it reads; the windowed layer's must not, because
    at a prefix past 128 the oldest key is a key no query can reach. A reader that
    windowed the prefill but not the decode, or that read the ring from the wrong end,
    passes every other test in this file and fails this one.
    """
    prefix = 300
    assert prefix > release.layer.attention(SWA_LAYER).sliding_window
    torch.manual_seed(10)
    true_hidden = torch.randn(1, prefix, release.layer.hidden_size) * 0.5
    poisoned = true_hidden.clone()
    poisoned[:, :100] = torch.randn(1, 100, release.layer.hidden_size) * 0.5

    moved = {}
    for layer_idx in (SWA_LAYER, GA_LAYER):
        device = MimoV2DeviceAttention(release, layer_idx, "cuda", torch.float32)
        answers = []
        for hidden in (true_hidden, poisoned):
            cache = cache_for(release, layer_idx, prefix + 4096, dtype=torch.float32)
            device.forward(hidden[0].cuda(), start_pos=0, cache=cache)
            # The step is the same token in both arms; only the cache's oldest 100
            # positions differ.
            answers.append(
                device.forward(true_hidden[0, prefix - 1 :].cuda(), start_pos=prefix - 1, cache=cache)[
                    "attn_out_post_o"
                ]
            )
        moved[layer_idx] = (answers[0] - answers[1]).abs().max().item()

    assert moved[SWA_LAYER] == 0, (
        f"the windowed layer read a key outside its window: {moved[SWA_LAYER]}"
    )
    assert moved[GA_LAYER] > 1e-3, f"the global layer ignored the keys it holds: {moved[GA_LAYER]}"


@needs_release_cuda
def test_a_call_that_is_wrong_does_not_poison_the_next_one(release):
    """The running maximum once aliased the sink and wrote into the layer's parameter.

    A one-query call is the shape that does it -- the expansion of a `[heads, 1]` sink
    over one query is already contiguous, so the copy that was meant to protect the
    parameter was not made -- and the damage lands on the *next* call, which is why the
    test calls twice and compares the parameter with itself.
    """
    device = MimoV2DeviceAttention(release, SWA_LAYER, "cuda", torch.float32)
    before = device.sink.clone()
    torch.manual_seed(11)
    hidden = torch.randn(1, 4, release.layer.hidden_size) * 0.5
    for i in range(4):
        device.forward(hidden[0, i : i + 1].cuda(), start_pos=i)

    assert torch.equal(device.sink, before), (
        "a call wrote into the sink",
        (device.sink - before).abs().max().item(),
    )
    one = device.forward(hidden[0, :1].cuda(), start_pos=0)["attn_out_post_o"]
    two = device.forward(hidden[0, :1].cuda(), start_pos=0)["attn_out_post_o"]
    assert torch.equal(one, two)
    assert one.shape == (1, release.layer.hidden_size)


@needs_cuda
def test_a_narrower_row_step_is_the_same_answer_to_a_rounding():
    """The tile a long-context chunk can afford is bought in rows, and rows are not a re-derivation.

    A prefill at 256k passes the score budget on its first key block and then runs a loop whose
    tile is `kv_heads x groups x rows x block` -- a product whose width is the chunk, which makes
    the chunk a memory bound as well as a speed one. The fix is to step the rows, and this is the
    test that says the fix is arithmetic rather than an approximation: a row still sees its own
    key blocks in the same order with the same running maximum, and the only thing the step changes
    is the row count of a gemm, which is a summation order inside a dot product.

    A rounding and not an identity, and the bound below is the measured one rather than a hope:
    the whole-slice loop and the narrowest step differ by 4.5e-08 at this size, against the 1e-5
    the block loop and the single pass are already held to. Anything that moved by a *tolerance*
    would show up here as a much larger number, so this is the assertion that separates the two.
    """
    torch.manual_seed(11)
    heads, kv_heads, queries, keys, dim, vdim = 8, 2, 37, 211, 16, 8
    query = torch.randn(heads, queries, dim, device="cuda") * 0.5
    key = torch.randn(kv_heads, keys, dim, device="cuda") * 0.5
    value = torch.randn(kv_heads, keys, vdim, device="cuda") * 0.5
    sink = torch.randn(heads, device="cuda")
    lower, upper = chunk_bounds(queries, keys, 64)

    whole, _ = blocked_attention(query, key, value, lower, upper, scaling=0.5, sink=sink, block=16)
    for step in (1, 3, 8, 36):
        split, stats = blocked_attention(
            query, key, value, lower, upper, scaling=0.5, sink=sink, block=16, row_step=step
        )
        assert torch.allclose(whole, split, atol=1e-6), (step, (whole - split).abs().max().item())
        assert stats.blocks and stats.full_pairs == heads * queries * keys

    # And the bounds still decide which keys a row sees, so a narrow step is not narrow *keys*:
    # dropping the window changes the answer, which is what makes the above a comparison.
    narrower, _ = blocked_attention(
        query, key, value, lower, upper, scaling=0.5, sink=sink, block=16, row_step=3
    )
    open_lower, open_upper = full_bounds(queries, keys)
    unwindowed, _ = blocked_attention(
        query, key, value, open_lower, open_upper, scaling=0.5, sink=sink, block=16, row_step=3
    )
    assert not torch.allclose(narrower, unwindowed)


@needs_cuda
def test_the_call_bounds_its_tile_and_not_only_its_total():
    """`attention` hands the loop a row budget, so a chunk width is a memory knob and not a wall.

    The two bounds are different questions. `budget` decides whether the whole grid fits at once,
    and a 256k prefill answers it "no" however it is chopped. `tile_budget` decides how much of the
    grid the loop holds at a time, and that one has an answer at any chunk width -- which is what
    lets a 2048-token chunk keep running at 262144 positions instead of being refused by a card
    with two gigabytes free. The tile only moves where the gemm's rows end, so every
    budget below is the same answer to a rounding.
    """
    torch.manual_seed(12)
    heads, kv_heads, queries, keys, dim, vdim = 8, 2, 64, 4096, 16, 8
    query = torch.randn(heads, queries, dim, device="cuda") * 0.5
    key = torch.randn(kv_heads, keys, dim, device="cuda") * 0.5
    value = torch.randn(kv_heads, keys, vdim, device="cuda") * 0.5
    lower, upper = full_bounds(queries, keys)

    reference, _ = blocked_attention(query, key, value, lower, upper, scaling=0.5)
    for tile in (1 << 12, 1 << 16, 1 << 23):
        wide, _ = attention(query, key, value, lower, upper, scaling=0.5, tile_budget=tile)
        assert torch.allclose(reference, wide, atol=1e-6), (
            tile,
            (reference - wide).abs().max().item(),
        )


# ---------------------------------------------------------------------------
# The split over the ranks
# ---------------------------------------------------------------------------


@needs_release_cuda
@pytest.mark.parametrize("layer_idx", (SWA_LAYER, GA_LAYER))
def test_a_share_reads_the_whole_projection_rows_it_owns(release, layer_idx):
    """A share's fused `qkv_proj` is the whole's rows for that share, bit for bit.

    This is the one reading the split cannot get wrong quietly. `quant.QKV_SHARDS` records that the
    released projection was quantised a tensor-parallel shard at a time, so the FP8 scale restarts
    inside each share; a reader that dequantized a share with a whole-tensor scale -- or that asked
    for the wrong window of rows -- would produce a plausible projection of the right shape whose
    numbers are wrong from the first shard boundary on. The check is on the *product* and not on the
    weights, because the two readings agree on the weights and differ on the rows they scale.
    """
    shape = release.layer.attention(layer_idx)
    whole = MimoV2DeviceAttention(release, layer_idx, "cuda", torch.bfloat16)
    torch.manual_seed(5)
    hidden = torch.randn(4, release.layer.hidden_size, device="cuda", dtype=torch.bfloat16)
    reference = torch.nn.functional.linear(hidden, whole.qkv_proj)

    shares = [
        MimoV2DeviceAttention(
            release, layer_idx, "cuda", torch.bfloat16, shard=rank, shards=4, gather=lambda x: x
        )
        for rank in range(4)
    ]
    joined = torch.cat(
        [torch.nn.functional.linear(hidden, share.qkv_proj) for share in shares], dim=-1
    )
    assert torch.equal(joined, reference), (joined - reference).abs().max().item()
    assert shares[0].shape.num_q_heads == shape.num_q_heads // 4
    assert shares[0].shape.num_kv_heads == shape.num_kv_heads // 4
    assert shares[0].shape.o_in * 4 == shape.o_in
    # The share's stored order is `[q | k | v]` already, so a share pays no permutation where the
    # whole layer pays one -- see `group_qkv_order`.
    assert shares[0]._qkv_order is None


@needs_release_cuda
@pytest.mark.parametrize("layer_idx", (SWA_LAYER, GA_LAYER))
@pytest.mark.parametrize("rows,keys", ((512, 8192), (512, 129)))
def test_the_four_shares_are_the_whole_attention(release, layer_idx, rows, keys):
    """Four shares over the key heads, joined along the head axis, are the whole layer.

    The join is a concatenation and not a sum (`ep.make_all_gather` is the collective), and the
    shares are run against the *whole's own* cache, filled once and sliced -- so the comparison is
    of the head split and of nothing else. A chunk of rows comes out exact; a decode step's
    windowed layer can differ by the last bit of a float32, because the folded path's output gemm
    is batched over the key heads and cuBLAS tiles a batch of two differently from a batch of
    eight. That is a property of the library and not of the split, and it is why the tolerance
    below is a float32 ULP and not zero -- `probe_mimo_v2_split_tokens.py` is what says no token
    moves.
    """
    shape = release.layer.attention(layer_idx)
    whole = MimoV2DeviceAttention(release, layer_idx, "cuda", torch.bfloat16)
    torch.manual_seed(6)
    full_key = torch.randn(shape.num_kv_heads, keys, shape.head_dim, device="cuda")
    full_value = torch.randn(shape.num_kv_heads, keys, shape.v_head_dim, device="cuda")
    cache = cache_for(release, layer_idx, keys + rows + 8, dtype=torch.bfloat16)
    cache.append(
        layer_idx, full_key.to(torch.bfloat16), full_value.to(torch.bfloat16)
    )
    hidden = torch.randn(
        (rows, release.layer.hidden_size), device="cuda", dtype=torch.bfloat16
    )
    want = whole.forward(hidden, start_pos=keys, cache=cache)

    pieces = []
    for rank in range(4):
        share = MimoV2DeviceAttention(
            release, layer_idx, "cuda", torch.bfloat16, shard=rank, shards=4, gather=lambda x: x
        )
        own = cache_for(release, layer_idx, keys + rows + 8, dtype=torch.bfloat16, shard=rank,
                        shards=4)
        kv = shape.num_kv_heads // 4
        own.append(
            layer_idx,
            full_key[rank * kv : (rank + 1) * kv].to(torch.bfloat16),
            full_value[rank * kv : (rank + 1) * kv].to(torch.bfloat16),
        )
        pieces.append(share.attention_output(hidden, start_pos=keys, cache=own)[0])

    joined = torch.cat(pieces, dim=-1)
    assert joined.shape == want["attn_out_pre_o"].shape
    off = (joined.float() - want["attn_out_pre_o"].float()).abs().max().item()
    peak = want["attn_out_pre_o"].float().abs().max().item()
    assert off <= peak * 2.0**-23 * 2, (off, peak)
    # And the join is a *join*: the pieces are each a quarter of the width and a permutation of
    # them is not the answer, so the test above is not passing on symmetry.
    reordered = torch.cat(pieces[1:] + pieces[:1], dim=-1)
    assert not torch.equal(reordered, want["attn_out_pre_o"])
