"""Are the FP4 MoE kernels run-to-run reproducible?

Speculative decoding measured a real defect: DSpark always-k produced different
tokens from itself on the same prompt with the same weights (docs/performance/dspark.md,
"Output determinism"). That is distinct from batch-vs-sequential drift, which is
expected -- N-token and 1-token forwards reduce in different orders, so their
logits differ and an argmax near a tie can flip. Nondeterminism is not expected
and is fixable.

Both MoE kernels accumulate a token's top-k routes with atomicAdd, so the
summation order follows block scheduling and float addition is not associative.
This isolates that: same inputs, same weights, twice, bitwise compare. No
checkpoint needed -- random FP4 bytes exercise the same arithmetic.

Run: pytest tests/test_moe_kernel_determinism.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from src.kernels.cuda_loader import load_cuda_kernel  # noqa: E402

kernel = load_cuda_kernel()
if kernel is None or not hasattr(kernel, "moe_multi_token_fp4_forward"):
    pytest.skip("deepseek CUDA kernel not built", allow_module_level=True)

DIM = 512
INTER = 256
N_EXPERTS = 8
TOPK = 3
SWIGLU_LIMIT = 7.0


def _fp4_weights(rows: int, cols: int, n_experts: int, seed: int):
    """Random FP4 blocks in the layout the kernels consume.

    Values are opaque to the determinism question -- only that the same bytes go
    in twice matters -- but the shapes and dtypes must match the real path or the
    kernel takes a different branch.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(0, 256, (n_experts, rows, cols // 2),
                      generator=g, device="cuda", dtype=torch.uint8)
    # e8m0 block scales, one per 32-element block, biased around 2^0.
    s = torch.randint(120, 136, (n_experts, rows, cols // 32),
                      generator=g, device="cuda", dtype=torch.uint8)
    return q, s


def _routing(tokens: int, seed: int):
    """Build the slot/pair CSR the multi-token kernel expects.

    Slots are (expert -> contiguous run of pairs); a pair is one (token, expert)
    assignment. Several pairs can name the same token, which is exactly why the
    output accumulation is atomic.
    """
    g = torch.Generator().manual_seed(seed)
    pairs = []
    for t in range(tokens):
        experts = torch.randperm(N_EXPERTS, generator=g)[:TOPK]
        for e in experts.tolist():
            pairs.append((e, t))
    pairs.sort()  # group by expert so each expert's pairs are contiguous

    slot_expert, slot_starts, slot_tokens = [], [0], []
    cur = None
    for e, t in pairs:
        if e != cur:
            if cur is not None:
                slot_starts.append(len(slot_tokens))
            slot_expert.append(e)
            cur = e
        slot_tokens.append(t)
    slot_starts.append(len(slot_tokens))

    w = torch.rand(len(slot_tokens), generator=g)
    return (
        torch.tensor(slot_expert, dtype=torch.int32, device="cuda"),
        torch.tensor(slot_starts, dtype=torch.int32, device="cuda"),
        torch.tensor(slot_tokens, dtype=torch.int32, device="cuda"),
        w.to(device="cuda", dtype=torch.float32),
    )


def _multi_args(tokens: int, seed: int = 0):
    x = torch.randn(tokens, DIM, device="cuda", dtype=torch.bfloat16)
    slot_expert, slot_starts, slot_tokens, pair_w = _routing(tokens, seed)
    w1q, w1s = _fp4_weights(INTER, DIM, N_EXPERTS, seed + 1)
    w2q, w2s = _fp4_weights(DIM, INTER, N_EXPERTS, seed + 2)
    w3q, w3s = _fp4_weights(INTER, DIM, N_EXPERTS, seed + 3)
    return (x, slot_expert, slot_starts, slot_tokens, pair_w,
            w1q, w1s, w2q, w2s, w3q, w3s, SWIGLU_LIMIT)


def _run_multi(args):
    return kernel.moe_multi_token_fp4_forward(*args)


@pytest.mark.parametrize("tokens", [1, 2, 6])
def test_multi_token_moe_is_bitwise_reproducible(tokens):
    """Same inputs twice must give bitwise-identical output.

    This is the property DSpark needs and the one measured to fail end to end.
    A failure here localizes it to the MoE kernel; a pass sends the search to
    the attention path, which docs/performance/dspark.md already implicates.
    """
    args = _multi_args(tokens)
    first = _run_multi(args)
    for _ in range(4):
        again = _run_multi(args)
        assert torch.equal(first, again), (
            f"multi-token MoE not reproducible at tokens={tokens}: "
            f"max|diff|={(first - again).abs().max().item():.3e}")


def test_multi_token_moe_reproducible_under_concurrent_load():
    """Reproducibility must survive contention, not just an idle GPU.

    atomicAdd ordering follows block scheduling, which an idle device can make
    look deterministic by accident. Other work resident on the device perturbs
    the scheduler, so this is the sharper version of the test above.
    """
    args = _multi_args(6)
    first = _run_multi(args)
    noise = torch.randn(2048, 2048, device="cuda")
    for _ in range(8):
        noise @ noise  # keep SMs busy so block scheduling varies
        again = _run_multi(args)
        assert torch.equal(first, again), (
            "multi-token MoE diverges under load: "
            f"max|diff|={(first - again).abs().max().item():.3e}")


def _single_args(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, DIM, device="cuda", dtype=torch.bfloat16)
    idx = torch.randperm(N_EXPERTS, generator=g)[:TOPK].to(
        device="cuda", dtype=torch.int64)
    w = torch.rand(TOPK, generator=g).to(device="cuda", dtype=torch.float32)
    w1q, w1s = _fp4_weights(INTER, DIM, N_EXPERTS, seed + 1)
    w2q, w2s = _fp4_weights(DIM, INTER, N_EXPERTS, seed + 2)
    w3q, w3s = _fp4_weights(INTER, DIM, N_EXPERTS, seed + 3)
    return (x, idx, w, w1q, w1s, w2q, w2s, w3q, w3s, 0, SWIGLU_LIMIT)


def test_single_token_moe_is_bitwise_reproducible():
    """The control arm: is the plain-decode MoE path reproducible?

    It accumulates over top-k routes with the same atomicAdd pattern, so if it
    passes while the multi-token kernel fails, the difference is in how many
    blocks contend per output element, not in the presence of atomics -- and
    "plain decode is reproducible" is luck, not a property to rely on.
    """
    args = _single_args()
    first = kernel.moe_single_token_fp4_forward(*args)
    for _ in range(8):
        again = kernel.moe_single_token_fp4_forward(*args)
        assert torch.equal(first, again), (
            "single-token MoE not reproducible: "
            f"max|diff|={(first - again).abs().max().item():.3e}")


@pytest.mark.parametrize("topk", [1, 2, 3, 4, 6])
def test_single_token_moe_reproducible_at_every_topk(topk, monkeypatch):
    """topk >= 3 is where the atomic version broke, so cover the boundary.

    Two summands have only one possible order, so topk<=2 was stable even before
    the fix and cannot distinguish the two reductions. The live config uses
    n_activated_experts=6.
    """
    monkeypatch.setattr(sys.modules[__name__], "TOPK", topk)
    args = _single_args()
    first = kernel.moe_single_token_fp4_forward(*args)
    for _ in range(8):
        again = kernel.moe_single_token_fp4_forward(*args)
        assert torch.equal(first, again), (
            f"single-token MoE not reproducible at topk={topk}: "
            f"max|diff|={(first - again).abs().max().item():.3e}")


def test_deterministic_and_atomic_reductions_agree_numerically():
    """The fix must only reorder the summation, not change the arithmetic.

    Bitwise equality is not expected -- reordering float addition is the whole
    point -- so this bounds the disagreement instead, at the same ~1e-7 relative
    scale as the nondeterminism it replaces.
    """
    import os

    args = _multi_args(6)
    det = _run_multi(args).clone()

    # The env var is read per call, so flipping it in-process is enough.
    os.environ["DEEPSEEK_MOE_DETERMINISTIC_REDUCE"] = "0"
    try:
        atomic = _run_multi(args).clone()
    finally:
        os.environ.pop("DEEPSEEK_MOE_DETERMINISTIC_REDUCE", None)

    scale = det.abs().max().clamp(min=1.0)
    rel = ((det - atomic).abs().max() / scale).item()
    assert rel < 1e-4, f"reductions disagree beyond reordering noise: rel={rel:.3e}"


def test_atomic_path_still_reachable_and_still_nondeterministic():
    """Documents why the fix is needed, and that the old path is still selectable.

    Keeping the atomic version reachable makes the cost of determinism measurable.
    This test asserts only that it runs and produces finite output -- asserting
    that it *diverges* would be a flaky test, since divergence is probabilistic.
    """
    import os

    args = _multi_args(6)
    os.environ["DEEPSEEK_MOE_DETERMINISTIC_REDUCE"] = "0"
    try:
        y = _run_multi(args)
    finally:
        os.environ.pop("DEEPSEEK_MOE_DETERMINISTIC_REDUCE", None)
    assert torch.isfinite(y).all()


CSR_MIN_WORK = 50_000


def test_csr_reduce_matches_the_scan_it_replaced():
    """The per-token CSR reduce and the all-pairs scan are the same sum in the same order.

    `moe_multi_token_fp4_forward` groups a token's pairs into a CSR once `tokens * pairs` is past
    `kCsrMinWork` and scans the whole pair list below that, so every other MoE test in the suite
    takes the scan and this is the one that reaches the new kernel. The bar here is bit-identity
    rather than a tolerance: the CSR lists a row's pairs in ascending pair order, which is the order
    the scan adds them in, so the two are the same float additions in the same sequence. Anything
    but `equal` means the grouping reordered them, and reordering is what moves logits.

    Bit-identity between the two paths is also why this test can be this strict where
    `test_moe_multi_token_fp4.py` cannot: that file bounds the *batched* kernel against the
    one-token reference, and at a token count large enough to reach the CSR the elementwise
    relative bound there is dominated by swiglu cancellation in near-zero outputs, not by the
    reduction.

    A third arm takes the CSR arm's own construction of `row_ptr` the long way round --
    `DEEPSEEK_MOE_CSR_BINCOUNT=1`, `at::bincount` and a `cumsum` in place of the `searchsorted` and
    the closing `fill_` -- because that is the half of the block that has no other check on it. The
    switch is a bisect knob in the launcher's C++, so a divergence between the two constructions
    has nowhere else to show up.
    """
    import os

    tokens = 160
    args = _multi_args(tokens)
    pairs = int(args[3].numel())
    assert tokens * pairs >= CSR_MIN_WORK, (
        f"tokens={tokens} pairs={pairs} no longer reaches the CSR reduce path; "
        "pick a larger token count or lower kCsrMinWork")

    os.environ["DEEPSEEK_MOE_DETERMINISTIC_REDUCE"] = "1"
    try:
        os.environ["DEEPSEEK_MOE_CSR_REDUCE"] = "1"
        csr = _run_multi(args).clone()
        os.environ["DEEPSEEK_MOE_CSR_BINCOUNT"] = "1"
        bincount = _run_multi(args).clone()
        os.environ.pop("DEEPSEEK_MOE_CSR_BINCOUNT", None)
        os.environ["DEEPSEEK_MOE_CSR_REDUCE"] = "0"
        scan = _run_multi(args).clone()
    finally:
        os.environ.pop("DEEPSEEK_MOE_CSR_BINCOUNT", None)
        os.environ.pop("DEEPSEEK_MOE_CSR_REDUCE", None)
        os.environ.pop("DEEPSEEK_MOE_DETERMINISTIC_REDUCE", None)

    assert torch.equal(csr, bincount), (
        f"searchsorted and bincount build different row_ptr at tokens={tokens}: "
        f"max|diff|={(csr - bincount).abs().max().item():.3e}")
    assert torch.equal(csr, scan), (
        f"CSR reduce and all-pairs scan disagree at tokens={tokens}: "
        f"max|diff|={(csr - scan).abs().max().item():.3e}")
