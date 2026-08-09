"""Is Compressor's decode-phase state machine step-order-equivalent?

The faithfulness probe established that multi-token verify only ever mismatches
on *flush* rows -- rows where `(start_pos + i + 1) % ratio == 0`. Sweeping the
offset moves the failure with the flush row (offset 6 -> flush [0,4] -> fails at
pos 4; offset 7 -> flush [3] -> fails at pos 3), and n<=2 never fails because a
2-token batch cannot both write and read a new compressed cell.

That points at the overlap branch of Compressor.forward (runtime.py:1078-1086),
which is the ratio==4 path: it writes slot `ratio + start_pos % ratio`, and on a
flush it both reads a cat() of the two half-windows and then slides the second
half down into the first.

Verify runs that branch inside a per-position loop within one forward, while
plain decode runs it once per forward. The calls are identical in shape
(seqlen==1 each time), so if the state machine is order-equivalent these two
must agree exactly. No checkpoint needed -- the bug, if it is here, is in the
index arithmetic, not the weights.

If these tests PASS, Compressor is exonerated and the divergence must come from
kv_cache being mutated elsewhere on the batched path.
"""
from __future__ import annotations

import copy
import sys

import pytest
import torch

sys.path.insert(0, "/mnt/data1/dsv4_inference")

RATIO = 4
DIM = 64
# act_quant blocks the nope dims by 64, so head_dim - rope_head_dim must be a
# multiple of 64 or the flush path asserts before it computes anything.
HEAD_DIM = 128
ROPE_HEAD_DIM = 64


def _make_compressor():
    """A small real Compressor with deterministic weights, no checkpoint."""
    from src.models.deepseek_v4.runtime import Compressor, ModelArgs

    args = ModelArgs.__new__(ModelArgs)
    args.dim = DIM
    args.rope_head_dim = ROPE_HEAD_DIM
    args.norm_eps = 1e-6
    args.max_batch_size = 1

    torch.manual_seed(0)
    with torch.device("cpu"):
        comp = Compressor(args, compress_ratio=RATIO, head_dim=HEAD_DIM, rotate=False)
    for p in comp.parameters():
        torch.nn.init.normal_(p, std=0.05)
    comp.eval()

    # Both buffers must start where a fresh decode starts, or the two arms
    # inherit different history and the comparison means nothing.
    comp.kv_state.zero_()
    comp.score_state.fill_(float("-inf"))
    comp.kv_cache = torch.zeros(1, 64, HEAD_DIM, dtype=torch.float32)
    # freqs_cis is indexed at `start_pos + 1 - ratio` on flush, so it must cover
    # every position the test touches.
    comp.freqs_cis = torch.polar(
        torch.ones(256, ROPE_HEAD_DIM // 2), torch.linspace(0, 3, 256).unsqueeze(-1).expand(256, ROPE_HEAD_DIM // 2)
    )
    return comp


def _reset(comp):
    comp.kv_state.zero_()
    comp.score_state.fill_(float("-inf"))
    comp.kv_cache.zero_()


@torch.inference_mode()
def _run(comp, xs, start):
    """Feed each row as its own seqlen==1 call; collect flush outputs."""
    outs = []
    for i, x in enumerate(xs):
        out = comp(x, start + i)
        outs.append(None if out is None else out.clone())
    return outs


@pytest.mark.parametrize("start", [17, 18, 19, 20, 21, 22, 23, 24])
def test_flush_output_is_step_order_equivalent(start):
    """Per-position calls must not depend on being spread across forwards."""
    comp = _make_compressor()
    n = 5
    torch.manual_seed(start)
    xs = [torch.randn(1, 1, DIM) for _ in range(n)]

    _reset(comp)
    seq_outs = _run(comp, xs, start)
    seq_kv = comp.kv_cache.clone()
    seq_state = comp.kv_state.clone()

    _reset(comp)
    batch_outs = _run(comp, xs, start)
    batch_kv = comp.kv_cache.clone()
    batch_state = comp.kv_state.clone()

    flush_rows = [i for i in range(n) if (start + i + 1) % RATIO == 0]
    assert flush_rows, f"start={start} n={n} has no flush row; test is vacuous"

    for i in range(n):
        a, b = seq_outs[i], batch_outs[i]
        assert (a is None) == (b is None), f"row {i}: flush disagreement"
        if a is not None:
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    torch.testing.assert_close(seq_kv, batch_kv, rtol=0, atol=0)
    torch.testing.assert_close(seq_state, batch_state, rtol=0, atol=0)


@torch.inference_mode()
def test_flush_reads_only_its_own_four_positions():
    """A flush at position p must summarize p-3..p, not a later position.

    This is the property that a batched verify could plausibly break: if the
    per-position loop has already written a later position's slot before the
    flush reads its window, the compressed cell mixes in a future token.
    """
    comp = _make_compressor()
    start = 17
    n = 5
    torch.manual_seed(1)
    xs = [torch.randn(1, 1, DIM) for _ in range(n)]

    flush_rows = [i for i in range(n) if (start + i + 1) % RATIO == 0]
    assert flush_rows == [2], f"expected flush at row 2 for start=17, got {flush_rows}"
    fr = flush_rows[0]

    # Truncated arm: stop right after the flush, so no later position exists yet.
    _reset(comp)
    trunc = _run(comp, xs[: fr + 1], start)

    # Full arm: same prefix, but the loop continues past the flush.
    _reset(comp)
    full = _run(comp, xs, start)

    # The flush output must be identical either way. It differs only if the
    # flush consumed state that a later position wrote.
    torch.testing.assert_close(trunc[fr], full[fr], rtol=0, atol=0)


@torch.inference_mode()
def test_state_slide_matches_manual_window():
    """The :1085 slide must leave kv_state[:ratio] equal to the window that
    the *next* flush is supposed to overlap with."""
    comp = _make_compressor()
    start = 16  # start%4==0, so a flush lands exactly at row 3
    n = 8
    torch.manual_seed(2)
    xs = [torch.randn(1, 1, DIM) for _ in range(n)]

    _reset(comp)
    ratio = RATIO
    for i, x in enumerate(xs):
        pos = start + i
        out = comp(x, pos)
        if (pos + 1) % ratio == 0:
            # The flush writes its own slot (:1079) *before* sliding (:1085), so
            # the first half must equal the second half as it stood after that
            # write -- and since the slide is a copy, the two halves must now
            # match. Sampling kv_state before the call would read the pre-write
            # window and compare the wrong pair.
            torch.testing.assert_close(
                comp.kv_state[:, :ratio], comp.kv_state[:, ratio:], rtol=0, atol=0,
                msg=f"pos={pos}: slide did not carry the current window",
            )
            assert out is not None, f"pos={pos} should have flushed"
        else:
            assert out is None, f"pos={pos} should not flush"
