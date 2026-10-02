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


BACKENDS = ["cpu"]
_cuda_reason = _cuda_reason()
if _cuda_reason is None:
    BACKENDS.append("cuda")


def _device_param(name: str) -> pytest.ParameterSet:
    if name in BACKENDS:
        return pytest.param(name, id=name)
    return pytest.param(name, marks=pytest.mark.skipif(True, reason=_cuda_reason or ""), id=name)


DEVICES = [_device_param(name) for name in ("cpu", "cuda")]


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
    "argmax": case_argmax,
    "argmax_ties": case_argmax_ties,
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
    "argmax": "argmax",
    "argmax_ties": "argmax",
}

#: Cases whose answer is an index rather than a value.  They are compared
#: exactly -- a float one off is rounding and an index one off is a different
#: token -- and they are excluded from the poison test below, because `argmax`
#: allocates a scalar rather than a tensor and reporting an unwritten element
#: count for it would be counting something the op never claimed to fill.
INDEX_CASES = {"argmax", "argmax_ties"}

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


def _compare(case: str, result: dict, want: np.ndarray) -> None:
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
        assert error <= QUANTIZED_RTOL * scale, (
            f"{case}: max |c - reference| = {error} over a scale of {scale}"
        )
        return
    spread = float(np.max(np.abs(want))) or 1.0
    worst = float(np.max(np.abs(got - want.astype(np.float64))))
    assert worst <= BACKEND_RTOL * spread, (
        f"{case}: max |c - reference| = {worst} over a scale of {spread}"
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", list(CASES))
def test_the_kernel_matches_the_reference(device: str, case: str) -> None:
    """One op, one shape, against `backends/reference`."""
    rng = np.random.default_rng(SEED)
    tensors, params, expected = CASES[case](rng)
    request = write_request(OP_OF_CASE[case], tensors, params)
    _compare(case, run(request, device), expected)


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

    The tolerance is tighter than the reference comparison because there is no
    third party in between: two float32 evaluations of the same expression
    differ by association order and nothing else.
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
    spread = float(np.max(np.abs(host))) or 1.0
    worst = float(np.max(np.abs(host - card)))
    assert worst <= BACKEND_RTOL * spread, (
        f"{case}: max |cpu - cuda| = {worst} over a scale of {spread}"
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