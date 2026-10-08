"""Each C kernel, on inputs this test authors, against the numpy reference.

`test_forward.py` checks the whole graph against llama.cpp, which is the right
question and not the only one.  A forward pass that matches on one prompt tells
you the composition is right at the shapes that prompt produced; it does not
tell you a kernel is *correct*.  A row of attention with a span of one, a GEMM
with `k` not a multiple of four, a `rope` at a nonzero `start_pos` -- none of
those are reachable from a five-token prefill, and each is a place a kernel can
be wrong while the graph is right.

So this file drives one op at a time.  It reaches the kernels through
`src/tools/opcheck.cpp`, which links them directly rather than through the ABI,
because the ABI's surface is the graph call and the point here is to get at the
ops underneath it.

**The direction of the exchange is deliberate.**  This test writes a *request*
-- the op, its parameters, and the values to run on -- and the tool answers with
what the kernel produced.  The alternative, having the C tool generate its own
inputs and this test replicate the generator, would put the definition of "the
canonical call to `gemm`" in two languages, and the two would drift.  Here the
reference is the author: the same numpy arrays that go into the request file are
the ones `backends/reference` is called on, so the comparison is on identical
input by construction rather than by agreement.

What is *not* compared: `cache_append` and `cache_truncate`.  Those are declared
ops with reference implementations, and the C graph does not implement them as
ops at all -- the KV append is a `copy_device_to_device` inline in `qwen3.cpp`,
which is a decision about where the copy belongs rather than a missing kernel.
Driving them here would test a surface that does not exist.  `add` is the same
shape of gap for a different reason: the residual is `gemm(accumulate=true)`, so
the reference's separate `add` op has no C counterpart to compare against, and
pretending otherwise would be comparing a decomposition to a decomposition.

Every case runs on `cpu` and on `cuda`, and the two are compared to each other
as well as to the reference.  The cross-backend comparison is the one that will
matter for `gemm_quant`: a quantized kernel has no numpy *oracle* in the sense
llama.cpp is one -- the reference decoder is the definition -- so what it can be
checked against is the CPU kernel, which this file is what establishes is
trustworthy.

`ascend` joins them where it can.  It is a native backend reached through the
same tool, so a case that runs on it is compared to the same reference -- and
because it is fp16-hardware it is compared at :data:`AS_F16_RTOL`, a second bound
with its own justification and not a widened :data:`BACKEND_RTOL`.  Where a case
cannot run -- an op the backend has not implemented, or a shape its op refuses --
it is named in :data:`DEVICE_EXCLUDED_CASES` with the reason rather than skipped
in the body, so the set of what is *not* covered is as legible as the set of what
is.  There is no cross-backend comparison for `ascend` yet: the CPU-versus-card
check is meaningful because both contract in f32, and against an fp16 kernel it
would be measuring the two widths against each other.
"""

from __future__ import annotations

import pathlib
import subprocess

import numpy as np
import pytest

from pocketllm import native
from pocketllm.backends.reference import kernels as ref
from pocketllm.quant import formats

#: The ABI's tolerance for a dense product, the same one `test_forward.py` uses
#: against llama.cpp -- but here the comparison is two float32 computations of
#: the same expression, so the measured agreement should be many orders tighter
#: and the headroom is only for association order.
DENSE_RTOL = 2e-3

#: The ABI's tolerance for a quantized product.  Not used yet -- `gemm_quant` is
#: the next op to land -- and defined here so the number is in one place when it
#: does.
QUANTIZED_RTOL = 5e-2

#: How far `cpu` and `cuda` may drift from each other.  Tighter than
#: :data:`DENSE_RTOL` by an order: no third-party kernel choices are in between,
#: only two association orders for the same sum.
BACKEND_RTOL = 2e-4

#: How far an `ascend` kernel may drift from the reference *for the operations
#: the 310B executes in fp16*.  This is not slack and it is not a loosening of
#: :data:`BACKEND_RTOL` -- it is a second, wider bound with a different
#: justification, applied only to the four op families in
#: :data:`ASCEND_F16_OPS` and only on `ascend`.
#:
#: The fact behind it: every custom op in the package this backend drives
#: (`RmsNormNdCustom`, `SiluMulCustom`, `MatmulCubeCustom`,
#: `AttentionStepCustom`, `MatmulW4a16Custom`) has an fp16 kernel -- there is no
#: f32 variant -- so the backend narrows every f32 operand to fp16 on the way in
#: and widens the fp16 result on the way out.  Every other backend here, the
#: reference included, contracts in f32.  An fp16 relative epsilon is 2^-11
#: (4.9e-4), and a K- or context-long reduction that rounds each partial to fp16
#: accumulates several of those, so a ~1e-3 bound is the width of the format and
#: not headroom over it.  Measured across these very cases on the board, the worst
#: relative error per op is 6.0e-4 (silu_mul), 5.7e-4 (rms_norm, from the fp16
#: gamma alone), 5.2e-4 (gemm) and 5.3e-4 (attention); a bound set at
#: :data:`BACKEND_RTOL` (2e-4) would fail every one of them while the kernels are
#: bit-honest, which is what makes this a width bound rather than a fudge.
#:
#: What it still catches is the thing the case is for: a mis-transposed GEMM,
#: an attention head group read wrong, a norm that divided by the wrong `d` --
#: all of which move a value by a large fraction of its range, orders past 1e-3.
AS_F16_RTOL = 1.5e-3

#: The op families the 310B executes in fp16, and so the ones :data:`AS_F16_RTOL`
#: applies to.  Named as op families rather than case names because it is the
#: *kernel* that has the width, not the individual case -- a new rms_norm case is
#: fp16 on this backend for the same reason the existing one is.
#:
#: `rope` is the fifth: `aclnnRopeCustom` is another fp16-only custom-op kernel
#: (its internal arithmetic is fp32 but it reads and writes half), measured at
#: 5.1e-4 relative on the two `d = 128` cases -- the same shape of error as the
#: other four, and over :data:`BACKEND_RTOL` for the same reason.  `embedding`,
#: `argmax` and `softmax` are deliberately *not* here: they run through f32
#: built-ins on this board, so they are compared at the f32 bounds.
ASCEND_F16_OPS = frozenset({"rms_norm", "silu_mul", "gemm", "attention", "rope"})

#: How far `softmax` may drift from the reference, as an *absolute* bound.
#:
#: Not :data:`BACKEND_RTOL`, and the difference is not slack: a relative bound
#: applied to a probability is a bound on the reduction order, because the
#: kernel sums 151936 exponentials sequentially while numpy sums them pairwise,
#: and the two disagree in the last ulps of the row's *total*.  The measured
#: error at that width is ~2e-5 absolute; the bound is set an order above it.
#:
#: The shape of the error is worth knowing: a probability that is large relative
#: to its row -- the argmax of a peaked distribution -- is also the one whose own
#: `exp` dominates the total, so it is the *small* probabilities in a flat row
#: that carry the absolute error.  A row scaled so its maximum probability is
#: small is therefore the worst case, and that is why the value below is a
#: little above the 1.2e-5 measured on a standard-normal row.
SOFTMAX_ATOL = 5e-5

#: The seed every case draws from.  Fixed so a failure is reproducible from the
#: test name alone -- the generated request is not checked in, so the inputs have
#: to be recoverable from the code.
SEED = 20261002


def repository_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def opcheck_path() -> pathlib.Path:
    return repository_root() / "build" / "pocketllm-opcheck"


def _needs_tool() -> str | None:
    if not opcheck_path().is_file():
        return "build/pocketllm-opcheck is not built"
    return None


needs_tool = pytest.mark.skipif(_needs_tool() is not None, reason=_needs_tool() or "")

pytestmark = [needs_tool]


# --------------------------------------------------------------------------
# The request format, from the writing side.
# --------------------------------------------------------------------------


def _format(values: np.ndarray) -> str:
    """One line of values, at nine significant digits.

    Nine is not a preference: it is the number that round-trips a float32
    exactly, which is what makes the input the kernel sees bit-identical to the
    input the reference sees rather than a printed approximation of it.
    """
    flat = np.asarray(values).reshape(-1)
    if flat.dtype == np.uint8:
        return " ".join(f"{int(v)}" for v in flat)
    return " ".join(f"{int(v)}" if flat.dtype.kind in "iu" else f"{float(v):.9g}" for v in flat)


def write_request(op: str, tensors: dict, params: dict | None = None, poison: bool = False) -> str:
    lines = [f"op {op}"]
    if poison:
        lines.append("poison")
    for key, value in (params or {}).items():
        lines.append(f"param {key} {value}")
    for name, array in tensors.items():
        arr = np.asarray(array)
        if arr.dtype == np.uint8:
            dtype = "u8"
        else:
            dtype = "f32" if arr.dtype.kind == "f" else "i32"
        shape = " ".join(str(d) for d in arr.shape)
        lines.append(f"tensor {name} {dtype} {shape}" if shape else f"tensor {name} {dtype}")
        lines.append(_format(arr))
    return "\n".join(lines) + "\n"


def run(request: str, device: str = "cpu") -> dict:
    """One call to the tool, parsed.

    Raises rather than returning a failure dict: a request this test built that
    the engine refuses is a bug in the test or in the kernels, and either way it
    should stop the run with the engine's own words rather than a comparison
    against nothing.
    """
    result = subprocess.run(
        [str(opcheck_path()), "--request", "/dev/stdin", "--device", device],
        input=request,
        capture_output=True,
        text=True,
    )
    parsed: dict = {"device": device}
    values: list[float] = []
    ints: dict[str, int] = {}
    collecting = False
    for line in result.stdout.splitlines():
        if collecting:
            values = [float(v) for v in line.split()] if line.strip() else []
            collecting = False
            continue
        key, _, rest = line.partition(" ")
        if key == "status":
            parsed["status"] = rest
        elif key == "message":
            parsed["message"] = rest
        elif key == "tensor":
            parts = rest.split()
            parsed["out_shape"] = tuple(int(d) for d in parts[2:])
            collecting = True
        elif key == "int":
            name, _, value = rest.partition(" ")
            ints[name] = int(value)
    parsed["ints"] = ints
    parsed["values"] = values
    parsed["stdout"] = result.stdout

    if parsed.get("status") != "ok":
        raise AssertionError(
            f"opcheck refused the request (exit {result.returncode}): "
            f"{parsed.get('message', result.stderr.strip())}\n{request}"
        )
    return parsed


def outcome(request: str, device: str = "cpu") -> dict:
    """Like `run`, but returns a failure the way the tool reported it.

    The one place a refusal is the thing under test rather than a bug, so this
    is the door for those cases and `run` stays strict for the rest.
    """
    result = subprocess.run(
        [str(opcheck_path()), "--request", "/dev/stdin", "--device", device],
        input=request,
        capture_output=True,
        text=True,
    )
    parsed = {"status": "unknown", "stdout": result.stdout, "returncode": result.returncode}
    for line in result.stdout.splitlines():
        key, _, rest = line.partition(" ")
        if key in {"status", "message"}:
            parsed[key] = rest
        elif key == "op":
            parsed["op"] = rest
    return parsed


# --------------------------------------------------------------------------
# The backends this host can offer.
# --------------------------------------------------------------------------


def _cuda_reason() -> str | None:
    """Why `cuda` is unusable here, or None if it works.

    Asked by running a one-element request through the tool, which is the
    cheapest possible question: the tool resolves the backend before it reads
    the request's tensors, so a build without the backend answers with the
    message `registry.cpp` writes and a build with it answers with an output.
    Nothing is loaded and no `libpocketllm.so` is needed, which matters because
    this file's subject is the tool, not the ABI.
    """
    if not opcheck_path().is_file():
        return "build/pocketllm-opcheck is not built"
    probe = write_request("argmax", {"values": np.arange(3, dtype=np.float32)})
    try:
        run(probe, "cuda")
    except AssertionError as exc:
        message = str(exc)
        return message if "cuda" in message else None
    return None


def _ascend_reason() -> str | None:
    """Why `ascend` is unusable here, or None if it works.

    The same shape as :func:`_cuda_reason` -- a one-element request through the
    tool -- but its answer is read differently, and the difference is worth
    stating because the two obvious short cuts are both wrong here.

    "Converts to `--device ascent`" is the same *question*: the tool resolves the
    backend before it reads a tensor, so the answer comes back without a kernel
    running.  But the *text* cannot be inverted the way the CUDA check inverts
    "cuda", because this backend's unimplemented ops throw "ascend: <op> not
    implemented yet" -- so seeing the word "ascend" proves only that the backend
    was reached, which is the useful case and not the excluded one.

    What separates a build that has this backend from one that does not is
    `registry.cpp`'s refusal, "is not in this build".  What separates a host that
    can *load* it from one that cannot is the dynamic loader's own message: the
    backend links the CANN runtime, and a run without the toolkit's `set_env.sh`
    sourced dies before `main` with "error while loading shared libraries".  And
    a host whose CANN is present but has no NPU to bind fails inside the
    backend's constructor, named by the call that failed (`aclInit`,
    `aclrtSetDevice`, `aclrtCreateStream`).  All three are "no ascend here"; a
    kernel-level error is not.
    """
    if not opcheck_path().is_file():
        return "build/pocketllm-opcheck is not built"
    probe = write_request("argmax", {"values": np.arange(3, dtype=np.float32)})
    try:
        run(probe, "ascend")
    except AssertionError as exc:
        message = str(exc)
        if "error while loading shared libraries" in message:
            return (
                "the ascend backend needs the CANN runtime on LD_LIBRARY_PATH; "
                "source <toolkit>/set_env.sh and the custom-op set_env.bash first"
            )
        if "is not in this build" in message:
            return message.splitlines()[0]
        for call in ("aclInit", "aclrtSetDevice", "aclrtCreateStream", "aclrtGetSocName"):
            if call in message:
                return f"ascend is built but this host cannot bind a device ({call} failed)"
        return None
    return None


BACKENDS = ["cpu"]
_cuda_reason = _cuda_reason()
if _cuda_reason is None:
    BACKENDS.append("cuda")
_ascend_reason = _ascend_reason()
if _ascend_reason is None:
    BACKENDS.append("ascend")


DEVICE_REASONS = {"cuda": _cuda_reason, "ascend": _ascend_reason}


def _device_param(name: str) -> pytest.ParameterSet:
    if name in BACKENDS:
        return pytest.param(name, id=name)
    return pytest.param(
        name, marks=pytest.mark.skipif(True, reason=DEVICE_REASONS[name] or ""), id=name
    )


DEVICES = [_device_param(name) for name in ("cpu", "cuda", "ascend")]

#: Cases a device cannot drive, keyed by the case's op, each with the reason.
#:
#: Not a loosened tolerance and not a deleted case: these are ops the 310B
#: backend does not implement, so a request that reaches one is refused *by
#: name* rather than answered with a wrong number, and there is no expected value
#: to compare against.  Recording them here rather than skipping inline keeps the
#: divergence in one place a reader can audit, and keeps the cases alive on `cpu`
#: and `cuda` where they are the point.
#:
#: Two kinds of reason live here and they are worth telling apart:
#:
#:   * The op is absent from the backend ("not implemented yet").  These are the
#:     ops the backend's file header says it still throws for -- the sampler
#:     family (`logits_temperature`, `topk_sample`) and the shape limits on the
#:     packed and dense GEMMs.  Removing an entry here is how a newly implemented
#:     op gets its conformance coverage, so the list is a to-do and not a "these
#:     do not matter".
#:   * The op exists but its interface is narrower than the harness's case, and
#:     refuses the case by name: `attention` on this backend is a one-token
#:     decode step (`q_len == 1`, `first_key == 0`), which is the shape the
#:     graph's decode uses, so the chunked and sliding-window cases cannot run;
#:     `rms_norm` and `rope_neox` need a width that is a whole number of fp16
#:     vector lanes (see their entries).
DEVICE_EXCLUDED_CASES: dict[str, dict[str, str]] = {
    "ascend": {
        # --- ops the backend still throws by name for ---
        "logits_temperature": "ascend: logits_temperature not implemented yet",
        "topk_sample": "ascend: topk_sample not implemented yet",
        "topk_sample_uniform_on_a_boundary": "ascend: topk_sample not implemented yet",
        "topk_sample_min_p": "ascend: topk_sample not implemented yet",
        "topk_sample_top_p": "ascend: topk_sample not implemented yet",
        "topk_sample_all_tied": "ascend: topk_sample not implemented yet",
        # --- ops that exist but take a narrower shape ---
        #
        # `gemm_quant` here is q4_k only, M=1 only, and needs n and k multiples
        # of 128 for the W4A16 packing; these two cases are f32 (M=3/4) and
        # q6_k, so the op refuses them by name rather than running them wrong.
        "gemm_quant_q4_k": "ascend: gemm_quant is M=1 and n,k % 128 == 0 only",
        "gemm_quant_q6_k": "ascend: gemm_quant decodes q4_k only",
        "gemm_bias": "ascend: gemm bias not implemented yet (MatmulCubeCustom has no bias input)",
        # `AttentionStepCustom` is a one-token decode step; `q_len > 1` throws
        # "not implemented for prefill" and `first_key != 0` a sliding window the
        # op does not implement.
        "attention_chunk": "ascend: attention is decode-only (q_len == 1)",
        "attention_grouped": "ascend: attention is decode-only (q_len == 1)",
        "attention_first_key": "ascend: attention has no sliding window (first_key != 0)",
        # --- cases that do not apply to an fp16/32B-lane kernel ---
        #
        # `rms_norm_single_row` is `d = 1`, which is below the 16-lane vector
        # repeat the AscendC kernel works in: `RmsNormNdCustom` writes nothing for
        # `d < 16` (measured -- `d = 1, 8, 15` come back zeroed, `d = 16` onwards
        # is right), and this backend refuses that width by name rather than
        # returning the zeros.  The case pins a division-by-`d` bug the reference
        # and the CPU/CUDA kernels can have; on a kernel that cannot run the
        # shape, there is nothing to compare.
        "rms_norm_single_row": "ascend: rms_norm needs d a multiple of 16 (RmsNormNdCustom's "
        "vector repeat)",
        # `d = 4`, whose split half is two half-words -- a quarter of the 16-lane
        # vector repeat `aclnnRopeCustom` moves its halves in, so the `DataCopy`
        # over-reads into the next row (measured: 100% error, and the same at
        # d = 16).  The width the graph actually uses is 128, which is a whole
        # number of repeats; the backend refuses `d % 32 != 0` by name rather
        # than returning the garbage.
        "rope_small_head_dim": "ascend: rope_neox needs d a multiple of 32 (RopeCustom's "
        "16-lane repeat over one split half)",
        # `silu_mul_large_negative` is a gate of -50, whose correct output is
        # -9.6e-21 -- about 1e13 below the smallest positive fp16 subnormal
        # (6e-8).  An fp16 kernel has no way to represent it and returns signed
        # zero, so the two answers differ by 100% of a quantity that is zero in
        # the format doing the computing.  Not a divergence to widen a bound for:
        # the case is about f32 overflow in `exp(50)`, and the 310B kernel never
        # evaluates it in f32 to begin with.
        "silu_mul_large_negative": "ascend: result 9.6e-21 is below fp16's smallest subnormal "
        "(6e-8), so an fp16 kernel returns signed zero",
    }
}


def _device_case_skip_reason(device: str, case: str) -> str | None:
    """The recorded reason this (device, case) is not run, or None."""
    return DEVICE_EXCLUDED_CASES.get(device, {}).get(case)


# --------------------------------------------------------------------------
# The cases.
#
# Each is a function of the rng returning (tensors, params, expected).  The
# expected array is computed by the reference on the *same* arrays that are
# written into the request.
# --------------------------------------------------------------------------


def case_rms_norm(rng):
    x = rng.standard_normal((3, 256), dtype=np.float32)
    weight = rng.standard_normal(256, dtype=np.float32)
    return ({"x": x, "weight": weight}, {"eps": 1e-6}, ref.rms_norm(x, weight, eps=1e-6))


def case_rms_norm_single_row(rng):
    """A row of one element.  `mean` over a length-one axis is the element, so a
    kernel that divides by `d` instead of reading it passes every other case."""
    x = rng.standard_normal((1, 1), dtype=np.float32)
    weight = rng.standard_normal(1, dtype=np.float32)
    return ({"x": x, "weight": weight}, {"eps": 1e-6}, ref.rms_norm(x, weight, eps=1e-6))


def case_gemm_square(rng):
    x = rng.standard_normal((4, 64), dtype=np.float32)
    w = rng.standard_normal((32, 64), dtype=np.float32)
    return ({"x": x, "w": w}, {}, ref.gemm(x, w))


def case_gemm_k_not_multiple_of_four(rng):
    """`k = 6`.  The CPU kernel unrolls four at a time and the CUDA kernel's
    `dot4` does the same; a tail that is dropped or double-counted is invisible
    at `k` a multiple of four, which is every shape the graph produces."""
    x = rng.standard_normal((2, 6), dtype=np.float32)
    w = rng.standard_normal((5, 6), dtype=np.float32)
    return ({"x": x, "w": w}, {}, ref.gemm(x, w))


def case_gemm_bias(rng):
    x = rng.standard_normal((3, 16), dtype=np.float32)
    w = rng.standard_normal((8, 16), dtype=np.float32)
    bias = rng.standard_normal(8, dtype=np.float32)
    return ({"x": x, "w": w, "bias": bias}, {}, ref.gemm(x, w, bias))


def case_gemm_accumulate(rng):
    """The residual form.  The seed is what the product is *added to*, so a
    kernel that overwrites passes every other GEMM case here and loses the
    residual -- which is most of the signal in the network."""
    x = rng.standard_normal((3, 32), dtype=np.float32)
    w = rng.standard_normal((16, 32), dtype=np.float32)
    seed = rng.standard_normal((3, 16), dtype=np.float32)
    return (
        {"x": x, "w": w, "out_seed": seed},
        {"accumulate": 1},
        ref.gemm(x, w) + seed,
    )


def case_gemm_quant_q4_k(rng):
    """The packed product in the format a `q4_k_m` checkpoint spends most of its
    bytes on. `k = 512` is two blocks a row, which is the smallest shape where a
    kernel that read only the first block would still produce the right shape."""
    x = rng.standard_normal((4, 512), dtype=np.float32)
    blocks = packed_row(rng, rows=32, cols=512, fmt="q4_k")
    return (
        {"x": x, "w_blocks": blocks},
        {"type_id": 12},
        ref.gemm_quant(x, blocks, w_blocks_fmt="q4_k"),
    )


def case_gemm_quant_q6_k(rng):
    """`q6_k`, which a `q4_k_m` file mixes in for `attn_v` and `ffn_down`.

    A separate case rather than a parameter, because the two formats share
    nothing but the block width: a single test would report a decode error in
    either one under the same name, and the second format is where the
    interleaved `ql`/`qh` layout is.
    """
    x = rng.standard_normal((3, 256), dtype=np.float32)
    blocks = packed_row(rng, rows=16, cols=256, fmt="q6_k")
    return (
        {"x": x, "w_blocks": blocks},
        {"type_id": 14},
        ref.gemm_quant(x, blocks, w_blocks_fmt="q6_k"),
    )


def case_embedding(rng):
    table = rng.standard_normal((32, 48), dtype=np.float32)
    tokens = np.array([0, 31, 7, 7, 0], dtype=np.int32)
    return ({"tokens": tokens, "table": table}, {}, ref.embedding(tokens, table))


def case_embedding_quant_q4_k(rng):
    """The gather from a packed table, which is the tied-embedding path.

    `token_embd.weight` is both the first op of the graph and — in a checkpoint
    with no separate head — the matrix the final projection contracts against.
    A `q4_k_m` file leaves it packed, so a build that could only gather from a
    float table would expand 155 MB to 622 MB to read one row per token.
    """
    tokens = np.array([0, 7, 7, 3], dtype=np.int32)
    blocks = packed_row(rng, rows=8, cols=512, fmt="q4_k")
    return (
        {"tokens": tokens, "table_blocks": blocks},
        {"type_id": 12},
        ref.embedding(tokens, blocks, table_fmt="q4_k"),
    )


def case_embedding_quant_q6_k(rng):
    tokens = np.array([5, 0, 2], dtype=np.int32)
    blocks = packed_row(rng, rows=6, cols=256, fmt="q6_k")
    return (
        {"tokens": tokens, "table_blocks": blocks},
        {"type_id": 14},
        ref.embedding(tokens, blocks, table_fmt="q6_k"),
    )


def packed_row(rng, rows: int, cols: int, fmt: str) -> np.ndarray:
    """Random bytes with the geometry of `fmt`'s blocks, shaped as GGUF stores
    the tensor: ``(rows, cols // 256, block_bytes)``.

    The bytes are random *within* the format's fields and not uniform over the
    whole block: the pair of fp16 header values is drawn from a small positive
    range rather than from all 65536 bit patterns, because a uniform draw spends
    most of its mass on NaNs and infinities and a block whose `d` is NaN would
    make the comparison vacuous -- both sides would agree on "not a number".

    This is a *synthetic* weight and not a quantized one: no quantizer produced
    it, so it exercises the decode and the product on values the quantizer would
    never write. That is the point -- a real `q4_k` tensor is close to what it
    approximates, and the fields that matter (`d`, `dmin`, the 6-bit pairs) are
    within a few percent of each other's, so an off-by-one in a bit field can
    land on a plausible value. Random fields cannot.
    """
    fmt_desc = formats.format_for(fmt)
    blocks = rng.integers(0, 256, size=(rows, cols // 256, fmt_desc.block_bytes), dtype=np.uint8)

    def put_half(block: np.ndarray, offset: int, values: np.ndarray) -> None:
        block[..., offset : offset + 2] = (
            values.astype(np.float16).view(np.uint8).reshape(*block.shape[:-1], 2)
        )

    shape = blocks.shape[:-1]
    if fmt == "q4_k":
        # `d` at bytes 0..1 and `dmin` at 2..3, both fp16 and both positive.
        put_half(blocks, 0, rng.uniform(0.01, 0.1, size=shape))
        put_half(blocks, 2, rng.uniform(0.0, 0.05, size=shape))
    elif fmt == "q6_k":
        # `d` is the *last* field of a q6_k block, at 208..209. Writing it at
        # offset 0 instead -- which is where q4_k's is -- puts a random fp16 in
        # the scale slot and a random fp16 in `d`, and a random pair of bytes is
        # a NaN or an infinity about one time in three.
        put_half(blocks, 208, rng.uniform(0.01, 0.1, size=shape))
    else:
        raise ValueError(f"packed_row has no header layout for {fmt}")
    return blocks


def case_silu_mul(rng):
    gate = rng.standard_normal((3, 64), dtype=np.float32) * 4.0
    up = rng.standard_normal((3, 64), dtype=np.float32)
    return ({"gate": gate, "up": up}, {}, ref.silu_mul(gate, up))


def case_silu_mul_large_negative(rng):
    """A gate of -50.  `silu(-50)` is -1e-22 to f32, and a kernel that takes the
    obvious `g / (1 + exp(-g))` overflows `exp(50)` on the way there -- both
    implementations avoid it, and this is the case that says so."""
    gate = np.full((2, 8), -50.0, dtype=np.float32)
    up = np.ones((2, 8), dtype=np.float32)
    return ({"gate": gate, "up": up}, {}, ref.silu_mul(gate, up))


def rope_tables(capacity: int, half: int, theta: float = 1e6) -> tuple:
    """The `(capacity, d/2)` cos/sin tables the kernels take.

    Built the way `qwen3.cpp` builds them -- `theta^(-2i/d)` per frequency, times
    the absolute position -- so the case is the graph's arithmetic and not an
    invented table that happens to be in range.
    """
    freqs = theta ** (-np.arange(half, dtype=np.float64) / half)
    positions = np.arange(capacity, dtype=np.float64)[:, None]
    angles = positions * freqs[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def case_rope_start_zero(rng):
    """The prefill case: positions `0..n-1` with the head geometry Qwen3 uses."""
    d, heads, tokens = 128, 16, 3
    x = rng.standard_normal((tokens, heads, d), dtype=np.float32)
    cos, sin = rope_tables(tokens, d // 2)
    positions = np.arange(tokens)
    return (
        {"x": x, "cos": cos, "sin": sin},
        {"start_pos": 0},
        ref.rope(x, positions, cos, sin, layout="split"),
    )


def case_rope_start_pos_offsets(rng):
    """The decode case, which is the one that can be wrong and pass everything
    else: `start_pos = 300` with three tokens.  The table is indexed by
    `start_pos + t`, so a kernel that ignores the offset rotates by position 0
    and agrees with the prefill case exactly where the prefill case is tested.

    This is also the case that found the `scores_` sizing bug in `qwen3.cpp` --
    a decode at a high position writes past a scratch buffer sized to the batch.
    """
    d, heads, tokens, start = 128, 16, 3, 300
    x = rng.standard_normal((tokens, heads, d), dtype=np.float32)
    cos, sin = rope_tables(start + tokens, d // 2)
    positions = np.arange(start, start + tokens)
    return (
        {"x": x, "cos": cos, "sin": sin},
        {"start_pos": start},
        ref.rope(x, positions, cos, sin, layout="split"),
    )


def case_rope_small_head_dim(rng):
    """`d = 4`, where `d/2 = 2` and the split-half pairing is easy to get wrong
    by a factor of two in a way `d = 128` hides."""
    d, heads, tokens = 4, 2, 5
    x = rng.standard_normal((tokens, heads, d), dtype=np.float32)
    cos, sin = rope_tables(tokens, d // 2)
    positions = np.arange(tokens)
    return (
        {"x": x, "cos": cos, "sin": sin},
        {"start_pos": 0},
        ref.rope(x, positions, cos, sin, layout="split"),
    )


def _attention_case(rng, q_len, heads, kv_heads, d, first_key, q_offset, capacity):
    q = rng.standard_normal((q_len, heads, d), dtype=np.float32)
    k_cache = rng.standard_normal((capacity, kv_heads, d), dtype=np.float32)
    v_cache = rng.standard_normal((capacity, kv_heads, d), dtype=np.float32)
    scale = 1.0 / np.sqrt(d)
    positions = q_offset + np.arange(q_len)
    # The reference's `window` is a property of the query and the C `first_key` a
    # property of the cache, so the two only translate when there is one query;
    # see `case_attention_first_key`.  `first_key = 0` means no lower bound, which
    # a window of the whole span expresses.
    window = None if first_key == 0 else q_offset + 1 - first_key
    if window is not None and q_len != 1:
        raise ValueError("first_key translates to a window only for a single query")
    expected = ref.attention(
        q,
        k_cache,
        v_cache,
        positions,
        window=window,
        softmax_scale=float(scale),
        num_kv_heads=kv_heads,
    )
    tensors = {"q": q, "k_cache": k_cache, "v_cache": v_cache}
    params = {"first_key": first_key, "q_offset": q_offset, "scale": f"{scale:.9g}"}
    return tensors, params, expected


def case_attention_single_query(rng):
    """A decode step: one query, one visible key.  The softmax is over a single
    score, so it is 1.0 and the output is exactly `v` -- which makes any error in
    the max/subtract/normalize chain show up as an output that is not `v`."""
    return _attention_case(rng, 1, 16, 8, 128, 0, 7, capacity=8)


def case_attention_chunk(rng):
    """A chunked prefill, four queries at offsets 4..7 against a cache of 8."""
    return _attention_case(rng, 4, 16, 8, 128, 0, 4, capacity=8)


def case_attention_first_key(rng):
    """A nonzero `first_key` with a single query -- the sliding-window hook.

    One query rather than a chunk, and that is not a convenience.  The C
    parameter is "the cache's live span begins at `first_key`", which is a
    property of *where the cache was populated*; the reference's equivalent is
    `window`, "attend to the last `window` positions", a property of the query.
    For one query at offset `q` the two coincide at ``window = q + 1 -
    first_key``, and for a chunk they do not: a fixed `first_key` over queries at
    different offsets is a window that *widens*, which the reference's constant
    `window` cannot express.

    So this case is stated in the parameterization that exists today and pins
    the one property that matters -- the span's lower bound is honoured -- while
    the general chunked-cache case is left to whoever first needs it.  Nothing
    in the graph sets `first_key` nonzero, which is exactly why it is worth a
    case: it is carried through three implementations and otherwise unexercised.
    """
    return _attention_case(rng, 1, 4, 2, 16, first_key=3, q_offset=6, capacity=8)


def case_attention_grouped(rng):
    """Four query heads over two KV heads, small enough to read.  `group` is 2,
    so heads 0/1 read KV head 0 and 2/3 read KV head 1 -- a kernel that ignores
    the grouping passes at `n_heads == n_head_kv` and fails here."""
    return _attention_case(rng, 3, 4, 2, 8, first_key=0, q_offset=0, capacity=3)


def case_attention_decode_group2(rng):
    """A decode step at the shipped GQA group -- Qwen3-0.6B/1.7B, 16 heads over
    8 KV heads -- and d = 128, the width every Qwen3 layer uses.

    One query token is the whole point: `q_len == 1` is what selects the flash
    decode path, and neither `case_attention_grouped` (three tokens) nor any
    other case reaches it.  The span (12 keys) is short enough to walk in one
    flash block, which keeps the case about the per-head scoring rather than
    about the chunk merge `case_attention_chunk` already covers."""
    return _attention_case(rng, 1, 16, 8, 128, first_key=0, q_offset=11, capacity=12)


def case_attention_decode_group4(rng):
    """A decode step at GQA group 4 -- Qwen3-4B/8B, 32 query heads over 8 KV
    heads.

    The flash decode path scored its `group` query heads into a stack array
    sized `kAttentionHeadBatch` (2) and then folded all `group` of them, so at
    group 4 it folded two stale entries per key.  The first generated token
    stayed correct -- it comes from prefill, which batches heads differently --
    and every decode step after it was garbage, which is why a whole-model test
    on the 0.6B/1.7B ladder never caught it.  This case is that regression."""
    return _attention_case(rng, 1, 32, 8, 128, first_key=0, q_offset=11, capacity=12)


def case_argmax(rng):
    values = rng.standard_normal(151936, dtype=np.float32)
    values[90432] = 100.0
    return ({"values": values}, {}, np.array(ref.argmax(values), dtype=np.int64))


def case_argmax_ties(rng):
    """Ties go to the lowest index.  `np.argmax` does the same, so the two
    agree by definition -- and a kernel that uses `>` where it means `>=` picks
    the *last* of the tie, which is a different token."""
    values = np.zeros(16, dtype=np.float32)
    values[4] = 5.0
    values[9] = 5.0
    return ({"values": values}, {}, np.array(ref.argmax(values), dtype=np.int64))


def case_softmax(rng):
    """A narrow row, where the error is the last few ulps of a short sum."""
    x = rng.standard_normal((3, 64), dtype=np.float32) * 4.0
    return ({"x": x}, {}, ref.softmax(x))


def case_softmax_wide_row(rng):
    """A whole vocabulary in one row, which is the shape the op is called at.

    151936 is the Qwen3 vocabulary and not a round number: the error of a
    sequential float32 sum against numpy's pairwise one grows with the length of
    the row, and the bound this case is compared at is the measured error at
    *this* width rather than at a width chosen to make the test easy.
    """
    x = rng.standard_normal((1, 151936), dtype=np.float32)
    return ({"x": x}, {}, ref.softmax(x))


def case_logits_temperature(rng):
    """The one logit transform a decode applies, at a temperature a user would
    actually pass -- and not 1.0, where a kernel that ignored the parameter
    would still agree."""
    logits = rng.standard_normal((2, 32), dtype=np.float32) * 8.0
    return ({"logits": logits}, {"temperature": 0.7}, ref.logits_temperature(logits, temperature=0.7))


def case_topk_sample(rng):
    """Authored logits with a clear ranking and a uniform draw that lands in the
    middle, so the answer is a token a reader can check by hand.

    `rng` is deliberately unused: a sampled case built from a random draw would
    change meaning if the seed ever moved, and the point of this one is that its
    answer is obvious. The truncation rules each get a case of their own below.
    """
    logits = np.array([3.0, 2.0, 1.0, 0.5, -1.0, -4.0], dtype=np.float32)
    uniform = np.array([0.5], dtype=np.float32)
    want = ref.topk_sample(logits, uniform)
    return ({"logits": logits, "uniform": uniform}, {}, np.array(want, dtype=np.int64))


def case_topk_sample_uniform_on_a_boundary(rng):
    """Four tied logits, so the cumulative is exactly 0.25/0.5/0.75/1.0 in both
    implementations, and a draw of exactly 0.5.

    This is the `side="left"` case. The reference takes the *first* index whose
    cumulative reaches the draw, so the boundary belongs to the token at it;
    an implementation that used `>` instead of `>=` picks the token after. The
    ties are what make the cumulative exact: no `exp` rounding stands between
    the two implementations and the number this is testing.
    """
    logits = np.zeros(4, dtype=np.float32)
    uniform = np.array([0.5], dtype=np.float32)
    want = ref.topk_sample(logits, uniform)
    return ({"logits": logits, "uniform": uniform}, {}, np.array(want, dtype=np.int64))


def case_topk_sample_min_p(rng):
    """`min_p=1.0` keeps exactly the argmax: the cutoff is *inclusive*, so a
    kernel that used `>` here would keep nothing and fall back to `order[0]` --
    the same token, by a different route, which is why the case that would catch
    it is a `min_p` that sits on a boundary between two tokens and not on 1.0.

    The logits are spaced so the top probability is far above `min_p * top` for
    the second token and the answer is the argmax either way; the *value* being
    pinned is that the second token is dropped for the right reason.
    """
    logits = np.array([2.0, 1.0, 1.0, 0.0, -3.0], dtype=np.float32)
    uniform = np.array([0.9], dtype=np.float32)
    want = ref.topk_sample(logits, uniform, min_p=1.0)
    return ({"logits": logits, "uniform": uniform}, {"min_p": 1.0}, np.array(want, dtype=np.int64))


def case_topk_sample_top_p(rng):
    """`top_p` cuts a distribution with a dominant token: a draw past the kept
    prefix still returns a token from it, which is what makes the truncation
    observable at all."""
    logits = np.array([6.0, 5.0, 1.0, 0.0, -2.0], dtype=np.float32)
    uniform = np.array([0.99], dtype=np.float32)
    want = ref.topk_sample(logits, uniform, top_p=0.8)
    return ({"logits": logits, "uniform": uniform}, {"top_p": 0.8}, np.array(want, dtype=np.int64))


def case_topk_sample_all_tied(rng):
    """Every logit equal, so the ranking is decided by the tie rule alone.

    The reference's `argsort(..., kind="stable")` puts the lower index first,
    and with eight equal probabilities the draw at 0.5 lands on the fifth. A
    kernel that let ties fall the other way answers 3, which is a different
    token and not a rounding.
    """
    logits = np.zeros(8, dtype=np.float32)
    uniform = np.array([0.5], dtype=np.float32)
    want = ref.topk_sample(logits, uniform)
    return ({"logits": logits, "uniform": uniform}, {}, np.array(want, dtype=np.int64))


CASES = {
    "rms_norm": case_rms_norm,
    "rms_norm_single_row": case_rms_norm_single_row,
    "gemm_square": case_gemm_square,
    "gemm_k_not_multiple_of_four": case_gemm_k_not_multiple_of_four,
    "gemm_bias": case_gemm_bias,
    "gemm_accumulate": case_gemm_accumulate,
    "gemm_quant_q4_k": case_gemm_quant_q4_k,
    "gemm_quant_q6_k": case_gemm_quant_q6_k,
    "embedding": case_embedding,
    "embedding_quant_q4_k": case_embedding_quant_q4_k,
    "embedding_quant_q6_k": case_embedding_quant_q6_k,
    "silu_mul": case_silu_mul,
    "silu_mul_large_negative": case_silu_mul_large_negative,
    "rope_start_zero": case_rope_start_zero,
    "rope_start_pos_offsets": case_rope_start_pos_offsets,
    "rope_small_head_dim": case_rope_small_head_dim,
    "attention_single_query": case_attention_single_query,
    "attention_chunk": case_attention_chunk,
    "attention_first_key": case_attention_first_key,
    "attention_grouped": case_attention_grouped,
    "attention_decode_group2": case_attention_decode_group2,
    "attention_decode_group4": case_attention_decode_group4,
    "argmax": case_argmax,
    "argmax_ties": case_argmax_ties,
    "softmax": case_softmax,
    "softmax_wide_row": case_softmax_wide_row,
    "logits_temperature": case_logits_temperature,
    "topk_sample": case_topk_sample,
    "topk_sample_uniform_on_a_boundary": case_topk_sample_uniform_on_a_boundary,
    "topk_sample_min_p": case_topk_sample_min_p,
    "topk_sample_top_p": case_topk_sample_top_p,
    "topk_sample_all_tied": case_topk_sample_all_tied,
}

#: Which op each case drives.  Stated rather than parsed out of the function
#: name because `rms_norm_single_row` -> `rms_norm` is a convention, and a
#: convention that a new case name can quietly violate.
OP_OF_CASE = {
    "rms_norm": "rms_norm",
    "rms_norm_single_row": "rms_norm",
    "gemm_square": "gemm",
    "gemm_k_not_multiple_of_four": "gemm",
    "gemm_bias": "gemm",
    "gemm_accumulate": "gemm",
    "gemm_quant_q4_k": "gemm_quant",
    "gemm_quant_q6_k": "gemm_quant",
    "embedding": "embedding",
    "embedding_quant_q4_k": "embedding_quant",
    "embedding_quant_q6_k": "embedding_quant",
    "silu_mul": "silu_mul",
    "silu_mul_large_negative": "silu_mul",
    "rope_start_zero": "rope",
    "rope_start_pos_offsets": "rope",
    "rope_small_head_dim": "rope",
    "attention_single_query": "attention",
    "attention_chunk": "attention",
    "attention_first_key": "attention",
    "attention_grouped": "attention",
    "attention_decode_group2": "attention",
    "attention_decode_group4": "attention",
    "argmax": "argmax",
    "argmax_ties": "argmax",
    "softmax": "softmax",
    "softmax_wide_row": "softmax",
    "logits_temperature": "logits_temperature",
    "topk_sample": "topk_sample",
    "topk_sample_uniform_on_a_boundary": "topk_sample",
    "topk_sample_min_p": "topk_sample",
    "topk_sample_top_p": "topk_sample",
    "topk_sample_all_tied": "topk_sample",
}

#: Cases whose answer is an index rather than a value.  They are compared
#: exactly -- a float one off is rounding and an index one off is a different
#: token -- and they are excluded from the poison test below, because `argmax`
#: and `topk_sample` allocate a scalar rather than a tensor and reporting an
#: unwritten element count for one would be counting something the op never
#: claimed to fill.
#:
#: Exactness is the whole claim for `topk_sample` and it is meaningful here in a
#: way it is not for a float op: the ranked probabilities the two implementations
#: compute differ in the last ulps, yet the *token* is the same token unless the
#: draw lands on a boundary between two of them -- and the boundary cases above
#: put it there on purpose.
INDEX_CASES = {
    "argmax",
    "argmax_ties",
    "topk_sample",
    "topk_sample_uniform_on_a_boundary",
    "topk_sample_min_p",
    "topk_sample_top_p",
    "topk_sample_all_tied",
}

#: Cases the poison test cannot drive.  `gemm_accumulate` is refused by the tool
#: when poisoned -- the sentinel would be summed into the residual -- and that
#: refusal has its own test below; the argmax cases have no float tensor to
#: count.
POISONABLE = [case for case in CASES if case not in INDEX_CASES | {"gemm_accumulate"}]


#: Cases whose right operand is a packed weight, and the tolerance they are
#: compared at. A quantized product has a larger error than a dense one for a
#: reason that is not a bug: the reference decodes the blocks with numpy and
#: contracts in float64 through `einsum`, while the kernel contracts in float32
#: as it decodes. On a `q4_k` block the decoded values are exact in both, so the
#: difference is the accumulation -- but the accumulation is over 256 wide
#: groups of values that are themselves the product of a coarse quantizer, and
#: the measured spread is what :data:`QUANTIZED_RTOL` is set for.
QUANTIZED_CASES = frozenset(
    {
        "gemm_quant_q4_k",
        "gemm_quant_q6_k",
        "embedding_quant_q4_k",
        "embedding_quant_q6_k",
    }
)


def _bound(device: str, case: str) -> float:
    """The relative bound this (device, case) is compared at.

    One place, so the reason a bound differs is next to the bound.  `ascend` gets
    :data:`AS_F16_RTOL` for the ops it executes in fp16 and :data:`BACKEND_RTOL`
    for everything else it runs -- but every other op it runs is excluded, so in
    practice the wider bound is what its dense cases use.
    """
    if device == "ascend" and OP_OF_CASE[case] in ASCEND_F16_OPS:
        return AS_F16_RTOL
    return QUANTIZED_RTOL if case in QUANTIZED_CASES else BACKEND_RTOL


def _compare(case: str, result: dict, want: np.ndarray, device: str = "cpu") -> None:
    """The request's answer against the reference's, at the case's tolerance."""
    if case in INDEX_CASES:
        got = int(result["ints"]["out"])
        assert got == int(want), f"{case}: got index {got}, expected {int(want)}"
        return
    got = np.array(result["values"], dtype=np.float64).reshape(want.shape)
    assert got.shape == want.shape, f"{case}: got shape {got.shape}, expected {want.shape}"
    if case in QUANTIZED_CASES:
        # A relative bound on the *error*, not on the value: the weight is
        # quantized, so the answer the reference produces is itself an
        # approximation of the exact product, and comparing the two at a bound
        # derived from the output's magnitude would be measuring the quantizer
        # rather than the kernel. What is being asked is whether the two
        # implementations decoded the same blocks, and a misread bit field
        # changes a weight by a large fraction of its range.
        error = float(np.max(np.abs(got - want.astype(np.float64))))
        scale = float(np.max(np.abs(want))) or 1.0
        assert error <= _bound(device, case) * scale, (
            f"{case} on {device}: max |c - reference| = {error} over a scale of {scale}"
        )
        return
    if case == "softmax" or case == "softmax_wide_row":
        worst = float(np.max(np.abs(got - want.astype(np.float64))))
        assert worst <= SOFTMAX_ATOL, f"{case}: max |c - reference| = {worst}"
        return
    spread = float(np.max(np.abs(want))) or 1.0
    worst = float(np.max(np.abs(got - want.astype(np.float64))))
    assert worst <= _bound(device, case) * spread, (
        f"{case} on {device}: max |c - reference| = {worst} over a scale of {spread}"
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", list(CASES))
def test_the_kernel_matches_the_reference(device: str, case: str) -> None:
    """One op, one shape, against `backends/reference`."""
    reason = _device_case_skip_reason(device, case)
    if reason is not None:
        pytest.skip(reason)
    rng = np.random.default_rng(SEED)
    tensors, params, expected = CASES[case](rng)
    request = write_request(OP_OF_CASE[case], tensors, params)
    _compare(case, run(request, device), expected, device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", POISONABLE)
def test_the_kernel_writes_every_element(device: str, case: str) -> None:
    """A kernel that leaves part of its output untouched fails here.

    Under `--poison` the output buffer is filled with a sentinel before the call
    and the tool reports how many elements still hold it.  This is the failure a
    value comparison cannot see: an element the kernel never wrote holds
    whatever the previous token left in that buffer, which is finite, in range,
    and the right shape -- so it compares as *some* number rather than as an
    error, and on a decode loop it is the previous position's activation
    silently reused.

    The poison is not a value any kernel here produces, but nothing *forbids* an
    output to be it, which is why the tool reports a count and this asserts on
    it rather than the tool refusing.  A real collision would report as a
    nonzero count on a passing kernel, and would be obvious from the value.
    """
    reason = _device_case_skip_reason(device, case)
    if reason is not None:
        pytest.skip(reason)
    rng = np.random.default_rng(SEED)
    tensors, params, _ = CASES[case](rng)
    request = write_request(OP_OF_CASE[case], tensors, params, poison=True)
    result = run(request, device)
    unwritten = result["ints"].get("unwritten")
    assert unwritten == 0, (
        f"{case} on {device} left {unwritten} of "
        f"{np.prod(result['out_shape'])} elements unwritten"
    )


@pytest.mark.parametrize("case", list(CASES))
def test_the_backends_agree_with_each_other(case: str) -> None:
    """`cpu` and `cuda` are the same computation.

    This is the check that makes the CPU kernel usable as the oracle for a
    quantized one.  A typed Q4_K block has no third-party implementation to
    compare against -- the Python decoder *is* the definition -- so the question
    a new quantized kernel will be asked is "does the card agree with the host",
    and the answer is only meaningful if the two already agree on the arithmetic
    they share.

    For a dense case the tolerance is tighter than the reference comparison
    because there is no third party in between: two float32 evaluations of the
    same expression differ by association order and nothing else.  A packed
    `gemm_quant` is not that, and pretending otherwise would be the dishonest
    half of this file.  The CPU kernel quantizes its *activation* to int8 and
    takes the integer route (`src/quant/q8k.h`); the card kernel decodes each
    weight to float and multiplies the float activation.  Those are two
    different computations that both decode the same blocks, and their
    difference is the activation quantization's own error -- an absolute error
    that is a percent of the output's scale, which is what
    :data:`QUANTIZED_RTOL` bounds.  What this case can still catch is a card
    kernel that mis-reads a nibble, and that moves a weight by a large fraction
    of its range, far past the bound.
    """
    if "cuda" not in BACKENDS:
        pytest.skip(_cuda_reason or "cuda is not available")

    rng = np.random.default_rng(SEED)
    tensors, params, wanted = CASES[case](rng)
    request = write_request(OP_OF_CASE[case], tensors, params)
    host_result = run(request, "cpu")
    card_result = run(request, "cuda")

    if case in INDEX_CASES:
        host, card = int(host_result["ints"]["out"]), int(card_result["ints"]["out"])
        assert host == card, f"{case}: cpu says {host}, cuda {card}"
        return
    host = np.array(host_result["values"], dtype=np.float64).reshape(wanted.shape)
    card = np.array(card_result["values"], dtype=np.float64).reshape(wanted.shape)
    scale = float(np.max(np.abs(host))) or 1.0
    worst = float(np.max(np.abs(host - card)))
    bound = QUANTIZED_RTOL if case in QUANTIZED_CASES else BACKEND_RTOL
    assert worst <= bound * scale, (
        f"{case}: max |cpu - cuda| = {worst} over a scale of {scale}"
    )


def test_a_poisoned_accumulate_is_refused() -> None:
    """The two flags ask for incompatible things, and the tool says so.

    Worth a test because the alternative is silent: poisoning an accumulating
    GEMM adds the sentinel to the seed, so every element comes back wrong by the
    same large amount and the run looks like a kernel bug rather than a request
    that could not be honoured.
    """
    tensors = {
        "x": np.ones((1, 4), dtype=np.float32),
        "w": np.ones((2, 4), dtype=np.float32),
        "out_seed": np.zeros((1, 2), dtype=np.float32),
    }
    result = outcome(write_request("gemm", tensors, {"accumulate": 1}, poison=True))
    assert result["status"] == "error"
    assert "accumulate" in result.get("message", "")


@pytest.mark.parametrize("device", DEVICES)
def test_an_out_of_range_token_id_zeroes_the_row(device: str) -> None:
    """A deliberate divergence from the reference, pinned so it stays deliberate.

    `kernels.cpp` zeroes the output row for an id outside `[0, vocab)` and says
    why: the alternative -- reading where the id points -- is a read past the
    mapping at one end and a *valid* table row at the other, so a corrupt id
    would produce fluent output rather than a visibly wrong one.

    The reference does neither.  `table[ids]` with a negative id indexes from
    the end of the table in numpy -- it returns a real embedding row, silently;
    with an id past the end it raises `IndexError`.  So the two do not agree
    here, and no tolerance makes them: this is not two roundings of one sum, it
    is two answers to "what is an out-of-range id".

    The C behaviour is the one that stays, because on a device the reference's
    version is either a segfault or a plausible wrong answer, and neither is
    something a caller can act on.  The reference's job is to define the op, not
    to define the error handling around it -- and this test is where that
    distinction is written down rather than assumed.
    """
    # A device that does not implement `embedding` is named in the exclusion set
    # and skipped rather than driven.  `ascend` now does implement it -- its
    # gather is the built-in `aclnnEmbedding`, and the zeroing is done over the
    # out-of-range rows on the host, because the built-in itself returns garbage
    # for an id it cannot reject (measured: `19023.8` for `-1`).  The divergence
    # this pins is the one in `kernels.cpp`, and this is where the 310B is held
    # to it too.
    if _device_case_skip_reason(device, "embedding") is not None:
        pytest.skip(_device_case_skip_reason(device, "embedding"))
    table = np.arange(12, dtype=np.float32).reshape(4, 3)
    tokens = np.array([-1, 1, 99], dtype=np.int32)
    request = write_request("embedding", {"tokens": tokens, "table": table})
    got = np.array(run(request, device)["values"], dtype=np.float32).reshape(3, 3)
    assert np.array_equal(got[0], np.zeros(3)), "a negative id must zero its row, not wrap"
    assert np.array_equal(got[1], table[1]), "an in-range id must still gather"
    assert np.array_equal(got[2], np.zeros(3)), "an id past the end must zero its row"

    # And the reference really does disagree, which is what makes this a
    # divergence rather than a case where the two happen to match.
    assert np.array_equal(ref.embedding(np.array([-1], np.int32), table)[0], table[-1])
    with pytest.raises(IndexError):
        ref.embedding(np.array([99], np.int32), table)


def test_argmax_of_a_nan_row_differs_from_the_reference() -> None:
    """A NaN is never the answer here and always the answer for numpy.

    `np.argmax` returns the index of the first NaN it meets, because every
    comparison against a NaN is false so the running maximum is displaced by
    nothing and the NaN is simply the first element that is not `>` the max.
    The C kernel compares with a strict `>` and so skips every NaN, returning
    the largest finite index instead.

    Not a rounding and not a tolerance: the two return different integers.  The
    C rule is the one kept -- a NaN in the logits means a kernel upstream has
    already produced one, and the useful response is to keep generating rather
    than to emit whatever index the NaN happened to land at -- but the reference
    is the definition of the op, so the difference is recorded here and the
    reference comparison above deliberately does not feed it a NaN.
    """
    values = np.array([1.0, np.nan, 5.0, 2.0], dtype=np.float32)
    request = write_request("argmax", {"values": values})
    assert run(request, "cpu")["ints"]["out"] == 2, "the largest finite value is the answer"

    assert int(ref.argmax(values)) == 1, "numpy returns the NaN's index, not the largest"


@pytest.mark.parametrize("device", DEVICES)
def test_a_flat_tail_samples_where_an_exact_reference_would(device: str) -> None:
    """The one place the sampler's answer and the reference's may differ, measured.

    Both implementations rank *probabilities* off a shifted softmax and take the
    first index whose cumulative reaches the draw.  The difference is the
    accumulator: the reference's `np.cumsum` is float32 and the kernel's is
    double, and over 151936 additions a float32 running sum drifts by ~4e-7 --
    which sounds like nothing until the tail of the distribution is as flat as
    the vocabulary makes it.  There a step of 4e-7 in the cumulative is a step of
    tens of thousands of token ids, so a draw at the very top of the range lands
    in a different place.

    The reference's answer is the one that is wrong here: this test recomputes
    the same cumulative in float64 and shows the kernel agrees with *that*, to
    the index.  So the divergence is not "the kernel is approximate where the
    reference is exact" -- the C implementation is the more accurate of the two
    and the float32 cumsum is the outlier.

    The check is stated rather than left implicit because a naive port that
    matched `np.cumsum` bit for bit would be a port that reproduced numpy's
    rounding error, and the next person to read `softmax_row`'s comment about
    two accumulators deserves to find out why.
    """
    # This drives `topk_sample`, which `ascend` does not implement yet, so it is
    # recorded in the exclusion set rather than run into a refusal.  The op it
    # pins is `kernels.cpp`'s, which this backend will share when its sampler
    # lands -- at which point the entry comes out and this runs on it too.
    reason = _device_case_skip_reason(device, "topk_sample")
    if reason is not None:
        pytest.skip(reason)
    rng = np.random.default_rng(SEED)
    logits = rng.standard_normal(151936, dtype=np.float32)
    uniform = np.array([0.999], dtype=np.float32)

    probs = ref.softmax(logits).astype(np.float64)
    probs /= probs.sum()
    order = np.argsort(-probs, kind="stable")
    cumulative = np.cumsum(probs[order])
    exact = int(order[int(np.searchsorted(cumulative, float(uniform[0]), side="left"))])

    request = write_request("topk_sample", {"logits": logits, "uniform": uniform})
    assert run(request, device)["ints"]["out"] == exact

    # And the float32 reference is where the difference comes from: it is not
    # the same index, which is what makes this a divergence and not a case
    # where the two implementations happen to agree.
    assert int(ref.topk_sample(logits, uniform)) != exact


def test_a_malformed_request_is_refused() -> None:
    """A tensor whose value count disagrees with its shape.

    The tool has to refuse this rather than read on: the values would come out
    of the *next* line, so a short tensor is not a small error, it is a request
    whose every subsequent tensor is shifted by one.  Checked here because the
    test above it depends on the tool having parsed the request it was given.
    """
    request = "op rms_norm\ntensor x f32 2 4\n0 1 2 3\ntensor weight f32 4\n0 1 2 3\n"
    result = outcome(request)
    assert result["status"] == "error"
    assert "values" in result.get("message", "")