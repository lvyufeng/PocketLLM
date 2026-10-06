"""The CPU kernels use the whole machine, and get the same answer doing it.

The change this file guards splits every kernel over an axis whose outputs are
independent -- the output column of a GEMM, the token of a gather, the element of
an elementwise op -- and never splits a reduction.  That is what makes threading
*bit-identical* to itself: the same arithmetic runs, just on more cores, so the
output must not move with `$POCKETLLM_CPU_THREADS`.

The vectorized `gemm_quant` is the deliberate exception.  It reassociates the sum
inside a 256-weight block and contracts multiply-adds into FMA, both of which
move the last bits, and the plan accepted that as the price of speed.  So it has
*three* separate guards here rather than one:

1. **The decode is exact.**  The SIMD path must produce the same 256 weights the
   scalar decoder produces.  A comparison of the C kernel against the numpy
   reference at the conformance tolerance does not isolate a wrong nibble from a
   different accumulation order -- both show up as "a bit off".  So the decode is
   checked where it can be checked exactly: the scalar and AVX2 row dots, on a
   block a test authors, must agree to a *tight* relative bound, and at the same
   time be far away from a dot computed on a *different* weight set.  A wrong
   nibble moves the value by an order; a reassociation moves it by an ulp.

2. **The tolerance gate is the conformance one.**  `test_op_conformance.py`
   checks `gemm_quant` against numpy at `QUANTIZED_RTOL`, as a bound on the
   *error relative to the output's scale* and not elementwise.  That test
   reappears here at the thread counts this file uses, because a kernel can be
   vectorized correctly and still be threaded wrongly.  The global bound is the
   right one and `assert_allclose` is not: the integer path quantizes the
   *activation*, so an output element whose true value is near zero carries an
   absolute error that is a large fraction of itself while staying a percent of
   the output's scale -- and an elementwise relative bound reads that as a
   failure of a kernel that is correct.

3. **The token sequence is the primary gate.**  A last-bit change that flips an
   `argmax` is invisible to a tolerance and fatal to the model.  The greedy
   sequence over a real prompt must be llama.cpp's, at one thread and at eight:
   if a token flips, that is a finding to investigate and not a tolerance to
   widen.

Everything here drives the C engine through its own tools -- `opcheck` for a
single op, `run` for the whole model -- because the subject is the C kernels and
not the ABI's plumbing.  Nothing loads `libpocketllm.so`.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from pocketllm import native

#: The `q4_k_m` checkpoint, which is the one whose GEMMs this file measures.  The
#: `q4_k` and `q6_k` cases below are the two formats it is built from.
CHECKPOINT = pathlib.Path("/mnt/data1/models/qwen3-0.6b-q4_k_m.gguf")

#: The prompt `test_cli_run.py` uses, as ids so a tokenizer regression does not
#: read as a generation regression.
PROMPT = "The capital of France is"
PROMPT_IDS = [785, 6722, 315, 9625, 374]

#: How many greedy tokens the token-match cases generate.  Long enough for a
#: near-tie to have had a chance to resolve the other way, short enough that a
#: single-threaded decode at roughly seven tokens a second stays under a minute.
STEPS = 8

#: The thread counts the determinism cases compare.  One is the scalar reference,
#: eight is the smallest count that exercises every pool worker's claim loop.
THREAD_COUNTS = [1, 8]

#: The relative bound the scalar and AVX2 row dots must agree to.
#:
#: The two compute the same 256 weights and differ only in association and
#: contraction, so the honest bound is a small multiple of the float32 epsilon
#: accumulated over a `k`-long row -- `1e-6` at the `k` this file uses is a
#: hundredfold headroom over the `~1e-7` a reassociated sum of 1024 terms
#: should show.  It is set this tight on purpose: the value it is guarding
#: against is a *wrong nibble*, which moves the result by percent, not by ppm.
SIMD_RTOL = 1e-6

#: The bound the vectorized `gemm_quant` is held to against the numpy reference,
#: the ABI's number for a quantized product and the same one
#: `test_op_conformance.py` uses.  It is a bound on the *error relative to the
#: output's scale*, not elementwise: the integer path quantizes the activation,
#: so a small output carries a large relative error while staying a percent of
#: the tensor's scale, and only the global bound is a statement about the kernel.
QUANTIZED_RTOL = 5e-2


def repository_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def opcheck_path() -> pathlib.Path:
    return repository_root() / "build" / "pocketllm-opcheck"


def bench_path() -> pathlib.Path:
    return repository_root() / "build" / "pocketllm-bench"


def _missing_opcheck() -> str | None:
    if not opcheck_path().is_file():
        return "build/pocketllm-opcheck is not built"
    return None


needs_tools = pytest.mark.skipif(_missing_opcheck() is not None, reason=_missing_opcheck() or "")
needs_engine = pytest.mark.skipif(
    not native.is_available(), reason="libpocketllm.so is not built"
)
needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason=f"no checkpoint at {CHECKPOINT}"
)


def _thread_env(threads: int) -> dict:
    return dict(os.environ, POCKETLLM_CPU_THREADS=str(threads))


# --------------------------------------------------------------------------
# The request format, shared with `test_op_conformance.py`'s writer.
#
# Repeated rather than imported: a cross-file import inside tests/ makes one
# test file's collection depend on another's, and the format is four lines.  If
# it ever grows past that, the right home is a `conftest.py` both can see.
# --------------------------------------------------------------------------


def _format(values: np.ndarray) -> str:
    flat = np.asarray(values).reshape(-1)
    if flat.dtype == np.uint8:
        return " ".join(f"{int(v)}" for v in flat)
    return " ".join(f"{int(v)}" if flat.dtype.kind in "iu" else f"{float(v):.9g}" for v in flat)


def write_request(op: str, tensors: dict, params: dict | None = None) -> str:
    lines = [f"op {op}"]
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


def run_op(
    request: str, *, device: str = "cpu", threads: int = 1, env: dict | None = None
) -> np.ndarray:
    """One op through `opcheck`, at a chosen thread count, as an array.

    ``env`` is layered over ``_thread_env`` for the one caller that needs to
    change *which* kernel runs rather than how many cores run it
    (``$POCKETLLM_CPU_EXACT_GEMM``)."""
    result = subprocess.run(
        [str(opcheck_path()), "--request", "/dev/stdin", "--device", device],
        input=request,
        capture_output=True,
        text=True,
        env=dict(_thread_env(threads), **(env or {})),
        timeout=300,
    )
    values: list[float] = []
    shape: tuple[int, ...] = ()
    collecting = False
    for line in result.stdout.splitlines():
        if collecting:
            values = [float(v) for v in line.split()] if line.strip() else []
            collecting = False
            continue
        key, _, rest = line.partition(" ")
        if key == "status" and rest != "ok":
            raise AssertionError(f"opcheck refused the request: {rest}\n{request}\n{result.stderr}")
        if key == "tensor":
            parts = rest.split()
            shape = tuple(int(d) for d in parts[2:])
            collecting = True
    if not values and np.prod(shape, dtype=np.int64) != 0:
        raise AssertionError(f"opcheck produced no output for {request!r}\n{result.stdout}")
    return np.asarray(values, dtype=np.float32).reshape(shape)


def packed_row(rng: np.random.Generator, rows: int, cols: int, fmt: str) -> np.ndarray:
    """Synthetic packed weights shaped ``(rows, cols // 256, block_bytes)``.

    The block fields are random but the fp16 headers are drawn from a small
    positive range rather than from all 65536 bit patterns: a uniform draw is a
    NaN or an infinity about one time in three, and a block whose `d` is NaN
    would make every comparison vacuous.  The layout mirrors
    `test_op_conformance.py`'s helper on purpose -- both exercise the decode on
    values no quantizer would write, which is exactly where an off-by-one in a
    bit field stops looking plausible.
    """
    from pocketllm.quant import formats

    spec = formats.format_for(fmt)
    blocks = rng.integers(
        0, 256, size=(rows, cols // 256, spec.block_bytes), dtype=np.uint8
    )

    def put_half(offset: int, values: np.ndarray) -> None:
        blocks[..., offset : offset + 2] = (
            values.astype(np.float16).view(np.uint8).reshape(*blocks.shape[:-1], 2)
        )

    shape = blocks.shape[:-1]
    if fmt == "q4_k":
        put_half(0, rng.uniform(0.01, 0.1, size=shape))
        put_half(2, rng.uniform(0.0, 0.05, size=shape))
    elif fmt == "q6_k":
        # q6_k's `d` is the *last* field, at 208..209 -- writing it at 0, where
        # q4_k's is, lands a random fp16 in the scale slot instead.
        put_half(208, rng.uniform(0.01, 0.1, size=shape))
    else:
        raise ValueError(f"packed_row has no header layout for {fmt}")
    return blocks


# --------------------------------------------------------------------------
# 1. Threading is bit-identical to itself.
#
# The pool splits an independent-output axis and leaves every sum whole, so the
# bytes at eight threads must equal the bytes at one.  These compare `%.9g`
# strings, which round-trip float32 exactly -- string equality here is bit
# equality, not a tolerance.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("threads", THREAD_COUNTS)
@needs_tools
def test_dense_gemm_is_independent_of_the_thread_count(threads: int) -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((8, 512), dtype=np.float32)
    w = rng.standard_normal((64, 512), dtype=np.float32)
    request = write_request("gemm", {"x": x, "w": w})
    one = run_op(request, threads=1)
    many = run_op(request, threads=threads)
    np.testing.assert_array_equal(one, many)


@needs_tools
def test_rms_norm_and_elementwise_are_independent_of_the_thread_count() -> None:
    rng = np.random.default_rng(11)
    x = rng.standard_normal((64, 256), dtype=np.float32)
    weight = rng.standard_normal(256, dtype=np.float32)
    rms = write_request("rms_norm", {"x": x, "weight": weight}, {"eps": 1e-6})
    np.testing.assert_array_equal(run_op(rms, threads=1), run_op(rms, threads=8))

    gate = rng.standard_normal(16384, dtype=np.float32)
    up = rng.standard_normal(16384, dtype=np.float32)
    silu = write_request("silu_mul", {"gate": gate, "up": up})
    np.testing.assert_array_equal(run_op(silu, threads=1), run_op(silu, threads=8))


# --------------------------------------------------------------------------
# 2. The vectorized GEMM matches the reference, and the thread count does not
#    move it.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fmt,type_id", [("q4_k", 12), ("q6_k", 14)])
@needs_tools
def test_gemm_quant_matches_the_reference(fmt: str, type_id: int) -> None:
    """The conformance case's gate, at two thread counts.

    `test_op_conformance.py` checks this at one; the point of repeating it here
    is that the vectorized kernel and the threaded kernel are two changes, and a
    test that exercises both at eight threads is what says they compose.

    The bound is the conformance file's, applied the way that file applies it --
    on the error relative to the output's scale.  An elementwise `allclose` is a
    different and much tighter statement, and it fails on a correct kernel: the
    activation quantization's error is absolute, so an output near zero carries
    a relative error above 1 while the tensor-wide error stays under a percent.
    """
    from pocketllm.backends.reference import kernels as ref

    rng = np.random.default_rng(20261004)
    x = rng.standard_normal((4, 1024), dtype=np.float32)
    blocks = packed_row(rng, rows=32, cols=1024, fmt=fmt)
    expected = ref.gemm_quant(x, blocks, w_blocks_fmt=fmt)
    request = write_request("gemm_quant", {"x": x, "w_blocks": blocks}, {"type_id": type_id})
    scale = float(np.max(np.abs(expected)))
    for threads in (1, 8):
        got = run_op(request, threads=threads)
        worst = float(np.max(np.abs(got - expected)))
        assert worst <= QUANTIZED_RTOL * scale, (
            f"{fmt} at {threads} thread(s): max |c - reference| = {worst} over a scale of {scale}"
        )


@pytest.mark.parametrize("fmt,type_id", [("q4_k", 12), ("q6_k", 14)])
@needs_tools
def test_gemm_quant_is_independent_of_the_thread_count(fmt: str, type_id: int) -> None:
    """The vectorized kernel's *own* output does not move with the pool.

    This is the tighter of the two quantized checks: not the reference tolerance,
    but an exact comparison between one thread and eight.  The vector kernel is
    deterministic run to run, and threading splits only `j`, so the two must be
    byte-identical -- a difference here would mean a race, not a reassociation.
    """
    rng = np.random.default_rng(20261004)
    x = rng.standard_normal((3, 768), dtype=np.float32)
    blocks = packed_row(rng, rows=24, cols=768, fmt=fmt)
    request = write_request("gemm_quant", {"x": x, "w_blocks": blocks}, {"type_id": type_id})
    np.testing.assert_array_equal(run_op(request, threads=1), run_op(request, threads=8))


#: Activation rows a prefill reaches past, and the widths it reaches them at.
#:
#: `gemm_quant` quantizes the activation into a scratch whose size used to be a
#: fixed 1024 blocks (299 KiB); a shape past that silently took the *exact* path
#: instead.  The exact path is correct and only slower, which is why a wrong cap
#: never failed -- it just stopped using the fast kernel above a 256-token
#: prefill.  These are the shapes that cross the boundary, as `(m, k)`: `m` rows
#: of `k` weights, so `m * k / 256` blocks.
PREFILL_SHAPES = [(256, 3072), (512, 1024), (512, 3072), (1024, 3072)]


@pytest.mark.parametrize("m,k", PREFILL_SHAPES)
@needs_tools
def test_the_activation_quantizer_is_independent_of_the_thread_count(m: int, k: int) -> None:
    """A prefill shape's `gemm_quant`, one thread against eight, byte for byte.

    The quantizer used to run on the caller's thread before the parallel walk,
    which made it the one part of a prefill GEMM that did not scale; it now runs
    one block per task, alongside the walk.  The property that makes that
    legitimate is that a block owns its output and reads only its own 256
    floats, so splitting the loop over blocks cannot move a byte -- and the bytes
    are load-bearing, because the quantized activation is an operand of the
    token-for-token llama.cpp match.  A schedule that reassociated a block's
    scale would be a different model, not a slower one.

    The existing `test_gemm_quant_is_independent_of_the_thread_count` covers the
    same rule at `m=3, k=768`, which is nine blocks and barely one task's worth.
    These are the shapes where the quantizer is actually split: 1024 to 12288
    blocks, eight tasks at eight threads, and the same comparison is what says
    the split is invisible.

    One thread is the serial reference: the pool hands a lone worker the whole
    range, so its bytes are the pre-change quantizer's by construction.  The
    comparison is exact on purpose.  It is not the guard for the quantizer's
    *arithmetic* -- both runs execute the same `quantize_q8_block`, so a change
    to its rounding moves both together and this passes.  The case below is what
    holds the arithmetic, against the bound the conformance file uses.
    """
    rng = np.random.default_rng(20261005 + m)
    x = rng.standard_normal((m, k), dtype=np.float32)
    blocks = packed_row(rng, rows=16, cols=k, fmt="q4_k")
    request = write_request("gemm_quant", {"x": x, "w_blocks": blocks}, {"type_id": 12})
    single = run_op(request, threads=1)
    eight = run_op(request, threads=8)
    np.testing.assert_array_equal(single, eight)
    # And the comparison is not between two constants.
    assert float(np.max(np.abs(single))) > 0.0


@pytest.mark.parametrize("fmt,type_id", [("q4_k", 12), ("q6_k", 14)])
@needs_tools
def test_the_four_row_weight_walk_is_the_one_row_walk_four_times(
    fmt: str, type_id: int
) -> None:
    """Rows batched four at a time, against the same rows one at a time.

    `gemm_quant` walks a weight row once per four activation rows when `m`
    allows it, which is a change of *schedule* and must not be one of
    arithmetic: each of the four rows accumulates its blocks in the order
    `dot_row_q8k` accumulates them, with the same per-block float multiply and
    the same horizontal reduce.  That is the property the token-for-token
    llama.cpp match rests on, so it is pinned here rather than left to the
    conformance tolerance, which a reassociated sum would pass.

    A six-row call exercises both paths against the same packed weights: rows
    0-3 go through the batched walk and rows 4-5 through the one-row kernel.
    Six one-row calls are the reference.  A batched kernel that dropped the
    activation's own scale -- the mistake this test was written after making --
    differs by a factor of `y.d` per block and fails loudly here.
    """
    rng = np.random.default_rng(20261006)
    rows, k = 6, 1024
    x = rng.standard_normal((rows, k), dtype=np.float32)
    blocks = packed_row(rng, rows=16, cols=k, fmt=fmt)
    batched = run_op(
        write_request("gemm_quant", {"x": x, "w_blocks": blocks}, {"type_id": type_id}),
        threads=1,
    )
    for r in range(rows):
        alone = run_op(
            write_request("gemm_quant", {"x": x[r : r + 1], "w_blocks": blocks}, {"type_id": type_id}),
            threads=1,
        )
        np.testing.assert_array_equal(batched[r : r + 1], alone)
    assert float(np.max(np.abs(batched))) > 0.0


@pytest.mark.parametrize("fmt,type_id", [("q4_k", 12), ("q6_k", 14)])
@pytest.mark.parametrize("m", [16, 18])
@needs_tools
def test_the_eight_row_walk_is_the_four_row_walk(fmt: str, type_id: int, m: int) -> None:
    """The shipped eight-row tile is bit-identical to the four-row one.

    `gemm_quant` walks a weight row once per *eight* activation rows by default;
    `$POCKETLLM_CPU_GEMM_RPW=4` walks it once per four.  The two are the same
    kernel templated on the row count, and the row count is a scheduling choice
    and not an arithmetic one: each row accumulates its blocks in the order
    `dot_row_q8k` accumulates them, with the same per-block float multiply and
    the same horizontal reduce.

    That matters because eight is the shipping default and four was the
    llama.cpp-verified one, so this is what says the change of tile did not change
    the model.  The four-row run is the reference and not the one-row kernel
    because the eight-row template inherits the four-row template's body line for
    line -- the case above already ties that one to the one-row kernel.

    `m=18` is not a multiple of either row count, so both the full tiles and the
    ragged tail (rows 16-17, which fall through to the one-row kernel) are
    compared on the same weights.
    """
    rng = np.random.default_rng(20261006 + m)
    k = 1024
    x = rng.standard_normal((m, k), dtype=np.float32)
    blocks = packed_row(rng, rows=16, cols=k, fmt=fmt)
    request = write_request("gemm_quant", {"x": x, "w_blocks": blocks}, {"type_id": type_id})
    eight = run_op(request, threads=1)
    four = run_op(request, threads=1, env={"POCKETLLM_CPU_GEMM_RPW": "4"})
    np.testing.assert_array_equal(eight, four)
    assert float(np.max(np.abs(four))) > 0.0


@pytest.mark.parametrize("m,k", PREFILL_SHAPES)
@needs_tools
def test_a_large_activation_row_keeps_the_integer_path(m: int, k: int) -> None:
    """A shape past the old stack cap still runs the int8 path, not the fallback.

    The two paths give *different answers* on purpose -- the integer one
    quantizes the activation to int8 and the exact one keeps it in float32 --
    and `$POCKETLLM_CPU_EXACT_GEMM` selects the second.  So the fast path is
    measurably in use exactly when the default run differs from the forced-exact
    run, and a cap that is too small shows up as the two agreeing.

    The comparison is `run_op`'s existing entry point at `m` rows of `k`
    weights; `k = 1024` and `k = 3072` are the two widths this graph's layers
    project against, and `m` is the prefill length.
    """
    rng = np.random.default_rng(20261005)
    x = rng.standard_normal((m, k), dtype=np.float32)
    blocks = packed_row(rng, rows=16, cols=k, fmt="q4_k")
    request = write_request("gemm_quant", {"x": x, "w_blocks": blocks}, {"type_id": 12})
    integer = run_op(request, threads=1)
    exact = run_op(request, threads=1, env={"POCKETLLM_CPU_EXACT_GEMM": "1"})
    assert not np.array_equal(integer, exact), (
        f"m={m} k={k}: the default and the forced-exact run agree, so the "
        f"activation scratch did not fit and the integer path was skipped"
    )
    # The same global bound the other quantized comparisons use: the int8
    # activation's error is absolute, so an output near zero carries a large
    # relative error while the tensor-wide error stays a percent of the scale.
    scale = float(np.max(np.abs(exact)))
    worst = float(np.max(np.abs(integer - exact)))
    assert worst <= QUANTIZED_RTOL * scale, (
        f"m={m} k={k}: max |int - exact| = {worst} over a scale of {scale}"
    )


# --------------------------------------------------------------------------
# 2b. The attention score dot, which has to be exact or the tokens move.
# --------------------------------------------------------------------------


def _decode_attention_case(span: int, seed: int):
    """A decode-shaped attention call: one query head set against `span` keys.

    The shape is the model's -- `q_len = 1`, 16 query heads over 8 KV heads,
    `d = 128`, `q_offset = span - 1` -- because the property under test is a
    property of *this* shape, where the score dot runs `d = 128` wide for every
    one of `span` cache rows.  A case with `d = 4` would fit in one vector and
    prove nothing.
    """
    rng = np.random.default_rng(seed)
    heads, kv_heads, d = 16, 8, 128
    q = rng.standard_normal((1, heads, d), dtype=np.float32)
    k_cache = rng.standard_normal((span, kv_heads, d), dtype=np.float32)
    v_cache = rng.standard_normal((span, kv_heads, d), dtype=np.float32)
    scale = 1.0 / np.sqrt(d)
    tensors = {"q": q, "k_cache": k_cache, "v_cache": v_cache}
    params = {"q_offset": span - 1, "scale": f"{scale:.9g}"}
    return tensors, params


#: The spans the attention exactness cases run at.  A short one exercises the
#: path a 5-token prompt takes; 512 is the long end of what llama.cpp's oracle in
#: `test_quantized_forward.py` can be asked for, and it is where the score dot
#: was `0.63x` of the pre-change throughput.
ATTENTION_SPANS = [8, 512]


def _prefill_attention_case(q_len: int, seed: int, n_heads: int = 16, n_head_kv: int = 8):
    """A prefill-shaped attention call: a *chunk* of queries, one forward pass.

    The decode case above has ``q_len = 1``, which is the shape where the
    four-row tiling collapses to the one-row kernel: a single block covers one
    query, so the shared-key region is that query's whole span and `dot_tile`
    never runs.  The tiling is a prefill work and this is the shape that
    exercises it -- both the region every row of a block shares and the
    triangular tail after it, since a block's last row sees `kAttentionRows - 1`
    keys its first row cannot.
    """
    rng = np.random.default_rng(seed)
    d = 128
    q = rng.standard_normal((q_len, n_heads, d), dtype=np.float32)
    k_cache = rng.standard_normal((q_len, n_head_kv, d), dtype=np.float32)
    v_cache = rng.standard_normal((q_len, n_head_kv, d), dtype=np.float32)
    scale = 1.0 / np.sqrt(d)
    tensors = {"q": q, "k_cache": k_cache, "v_cache": v_cache}
    params = {"q_offset": 0, "scale": f"{scale:.9g}"}
    return tensors, params


@pytest.mark.parametrize("span", ATTENTION_SPANS)
@needs_tools
def test_attention_is_independent_of_the_thread_count(span: int) -> None:
    """The attention output, at one thread and at eight, is the same *bytes*.

    The same rule the rest of this file holds every op to, applied to the
    kernel that grew a vectorized inner loop: the partition splits `(token,
    head)` units, never a unit's own chain of passes, so more cores must not
    move a single bit.  It is a statement about the *scheduling* and it is
    deliberately exact -- the alternative, a tolerance, would read a stale
    scratch row as a rounding.

    It is not the guard for the vector dot's arithmetic, and reading it as one
    would be a mistake: one thread and eight run the same code, so a change to
    `dot4`'s rounding moves both runs together and this comparison passes.  The
    case below is the one that checks the arithmetic.
    """
    tensors, params = _decode_attention_case(span, seed=20261005 + span)
    request = write_request("attention", tensors, params)
    single = run_op(request, threads=1)
    eight = run_op(request, threads=8)
    assert np.array_equal(single, eight), (
        f"span={span}: the attention output moved with the thread count -- "
        f"worst |difference| {float(np.max(np.abs(single - eight)))}"
    )
    # And the output is not a constant the comparison would pass trivially.
    assert float(np.max(np.abs(single))) > 0.0


@pytest.mark.parametrize("span", ATTENTION_SPANS)
@needs_tools
def test_the_attention_score_dot_is_the_scalar_order(span: int) -> None:
    """The four-lane score dot produces the scalar dot's bytes, not its value.

    The thread-count comparison above is bit-exact *by construction* -- the
    partition never splits a unit, so one thread and eight run the same code --
    and that means it would pass just as happily if the vectorized dot's
    rounding were the thing that moved.  This is the case that would not.

    Two runs of one request, differing only in `$POCKETLLM_CPU_SCALAR_DOT`,
    which is the switch `attention` exposes for exactly this: with it unset the
    score pass runs the four-lane `dot4`, with it set it runs the scalar `dot`.
    They must agree *exactly*.  The bound they are held to is not a tolerance --
    a tolerance is what let the FMA variant through.  An earlier draft of this
    test compared the kernel against a float64 oracle at `2 * d * eps` on the
    output's scale, and the `_mm_fmadd_ps` variant -- which moved the model's
    greedy completion from 32/32 tokens matching llama.cpp to 1/32 -- passed it
    with the measured figure at 2% of the bound.  The softmax spreads one
    last-bit score difference over a whole row of `d` outputs, where it stays
    under any sensible budget and still changes which token wins.

    So the claim is checked where it is true or false rather than where it is
    measurable: same input, both roundings, `memcmp`.  The scalar form is the
    reference because it is the one the pre-change engine shipped and the one
    llama.cpp's token match was recorded against.
    """
    tensors, params = _decode_attention_case(span, seed=20261005 + span)
    request = write_request("attention", tensors, params)
    vectorized = run_op(request, threads=1)
    scalar = run_op(request, threads=1, env={"POCKETLLM_CPU_SCALAR_DOT": "1"})
    assert np.array_equal(vectorized, scalar), (
        f"span={span}: the four-lane score dot does not reproduce the scalar "
        f"order -- worst |difference| {float(np.max(np.abs(vectorized - scalar)))}"
    )
    # And the comparison is not between two constants.
    assert float(np.max(np.abs(scalar))) > 0.0


#: The chunk lengths the four-row tiling is checked at.  512 is the prefill the
#: benchmarks use and a multiple of `kAttentionRows`, so every block is full;
#: 6 is not, so the last block covers two rows and the ragged path -- and the
#: triangular tail a short block leaves -- is what is under test; 1 is a decode
#: step, where the tiling must be invisible.
ATTENTION_CHUNKS = [1, 6, 512]


@pytest.mark.parametrize("q_len", ATTENTION_CHUNKS)
@needs_tools
def test_the_tiled_attention_score_is_the_one_row_dot(q_len: int) -> None:
    """Four queries scored per key walk, and the same bytes as one at a time.

    The kernel now scores `kAttentionRows` query rows of one head against a key
    vector in a single walk, with the key loaded once instead of four times.
    The 256-bit registers hold two queries' four-lane accumulator chains side by
    side, so every lane is the lane `dot4` would have computed -- including the
    reduce, which is `dot4`'s `((l0 + l1) + (l2 + l3)) + tail` and not an
    eight-lane horizontal sum.

    This is the check that keeps it that way, and it has to be a byte
    comparison for the same reason the score dot's is: `$POCKETLLM_CPU_SCALAR_DOT`
    routes both forms to the scalar `dot`, so the two runs differ in whether the
    tiling ran, not in the rounding of anything else.  A `q_len` that is not a
    multiple of the row count is included on purpose -- the ragged block and the
    causal tail are where a tiling kernel goes wrong, and a first version of this
    one did: it advanced a single `s` across the rows and quietly skipped the
    keys `t0+1 .. t0+j-1` of every row past the first, which is an output wrong
    by 3.1 that still looked like a 1.35x win.
    """
    tensors, params = _prefill_attention_case(q_len, seed=20261005 + q_len)
    request = write_request("attention", tensors, params)
    tiled = run_op(request, threads=1)
    scalar = run_op(request, threads=1, env={"POCKETLLM_CPU_SCALAR_DOT": "1"})
    assert np.array_equal(tiled, scalar), (
        f"q_len={q_len}: the four-row score tiling does not reproduce the "
        f"one-row dot -- worst |difference| {float(np.max(np.abs(tiled - scalar)))}"
    )
    # And the comparison is not between two constants.
    assert float(np.max(np.abs(scalar))) > 0.0


#: `(n_heads, n_head_kv)` pairs the head batching is checked at.  16/8 is the
#: shipped shape and the one the pairing exists for; 16/16 has one query head
#: per KV head, so no two heads share a key vector and `dot_pair` is never
#: entered; 8/4 keeps a group of two but at a head count the batch divides
#: evenly; 3/1 is the one that matters most -- `n_heads` is not a multiple of
#: `kAttentionHeadBatch`, so the last unit carries a single head and the `hb`
#: guard is what runs.  (3/1 and not 6/4: grouping is `n_heads / n_head_kv` by
#: integer division, so 6 heads over 4 KV heads leaves `group == 1` and heads
#: 4 and 5 reading a KV head the cache does not have.  The odd head count has
#: to come from an odd *group*, not from a non-divisor.)
ATTENTION_HEAD_SHAPES = [(16, 8), (16, 16), (8, 4), (3, 1)]


@pytest.mark.parametrize(("n_heads", "n_head_kv"), ATTENTION_HEAD_SHAPES)
@needs_tools
def test_the_batched_attention_score_is_the_per_head_dot(n_heads: int, n_head_kv: int) -> None:
    """Scoring two heads of one KV group per key walk, byte for byte.

    The score pass walked every key row once *per query head*; grouped attention
    has `n_heads / n_head_kv` heads reading each KV head, so a KV head's row was
    loaded that many times.  The batched form scores `kAttentionHeadBatch` of
    those heads against one load of the key vector, each in the low or high half
    of one `__m256` whose lanes are `dot4`'s -- the same offsets, the same
    `_mm_add_ps(_mm_mul_ps(...))`, never a fused multiply-add -- so the results
    are two `dot4` calls and not an approximation of them.

    `$POCKETLLM_CPU_SCALAR_DOT` is the switch that makes this checkable: it
    routes every score lane to the scalar `dot`, so the two runs differ only in
    whether the batching ran.

    The head shapes are the point.  `n_heads` not a multiple of the batch is the
    case a batching kernel gets wrong -- the last unit covers one head, not
    zero and not two -- and `n_head_kv == n_heads` is the case where the pairing
    loop must not run at all.  A version that assumed `n_heads % batch == 0`
    would read past the last group here, and the `v_cache` index it would build
    from the next KV head is a plausible number rather than a segfault.
    """
    tensors, params = _prefill_attention_case(q_len=6, seed=20261006 + n_heads * 31 + n_head_kv,
                                              n_heads=n_heads, n_head_kv=n_head_kv)
    request = write_request("attention", tensors, params)
    batched = run_op(request, threads=1)
    scalar = run_op(request, threads=1, env={"POCKETLLM_CPU_SCALAR_DOT": "1"})
    assert np.array_equal(batched, scalar), (
        f"n_heads={n_heads} n_head_kv={n_head_kv}: the batched score pass does "
        f"not reproduce the per-head dot -- worst |difference| "
        f"{float(np.max(np.abs(batched - scalar)))}"
    )
    # And the comparison is not between two constants.
    assert float(np.max(np.abs(scalar))) > 0.0


@pytest.mark.parametrize("q_len", ATTENTION_CHUNKS)
@needs_tools
def test_the_tiled_attention_is_independent_of_the_thread_count(q_len: int) -> None:
    """And the tiled form is bit-identical at one thread and at eight.

    The per-task scratch region grew from one row to `kAttentionRows` rows when
    the tiling arrived, and the CPU backend sizes it from the same two constants
    the kernel partitions by.  A backend that still allocated two rows would
    have two tasks interleaving their score rows -- fluent, finite, wrong output
    that only this comparison (or a token) would catch.
    """
    tensors, params = _prefill_attention_case(q_len, seed=20261005 + q_len)
    request = write_request("attention", tensors, params)
    one = run_op(request, threads=1)
    eight = run_op(request, threads=8)
    assert np.array_equal(one, eight), (
        f"q_len={q_len}: the tiled attention output moved with the thread count "
        f"-- worst |difference| {float(np.max(np.abs(one - eight)))}"
    )
    assert float(np.max(np.abs(one))) > 0.0


@pytest.mark.parametrize("q_len", ATTENTION_CHUNKS)
@needs_tools
def test_the_tiled_attention_weighted_sum_is_the_row_at_a_time_one(q_len: int) -> None:
    """Four output rows per walk over a V row, and the same bytes as one.

    The weighted sum is the second term of the attention call and it had been
    left alone when the score pass got its tiling: for every query row it
    streamed the whole V slab doing `dst[x] += weight * vvec[x]`.  The tiling
    now holds `kAttentionRows` accumulators and walks a V row once for all of
    them, splitting the walk into a shared prefix -- where every row of the tile
    sees the key -- and a per-row causal tail.

    **It is bit-exact, and that is a stronger claim than this kernel had to
    make.**  The plan accepted a last-bit change from the weighted sum's
    tiling, on the same reasoning as the score dot's: `dst[x] +=` over the keys
    is an accumulation and a differently-interleaved one could contract
    differently.  It does not.  Row `r` still adds its own terms in key order,
    one `+=` per key, with the same weight -- the tiling moves *which row* is
    being accumulated between two loads of the same V vector, not the order
    within any row's sum.  The probe that checked it (`/tmp/prof/attnwsum.cpp`,
    `Rv = 4` against the shipped loop at `q_len` 1 through 512) reports
    `relerr 0.00e+00` and this test holds the same thing to the byte.

    So the bar is equality and not a tolerance.  `$POCKETLLM_CPU_SCALAR_VSUM`
    routes the call back to the row-at-a-time loop, and the two runs must be
    `array_equal` on the same input.  If that ever stops holding the tiling has
    either reassociated a row's sum or, worse, dropped a row's region -- which is
    the bug the first draft of this kernel had: it gave every row only its own
    private slice of the keys and produced fluent output 0.8 of its own scale
    off, at a 1.58x speedup that made it look like a win.

    A `q_len` that is not a multiple of the row count is here for the ragged
    tile, where `jn < kAttentionRows` and the tail loop is the whole of the
    difference between the rows.
    """
    tensors, params = _prefill_attention_case(q_len, seed=20261005 + q_len)
    request = write_request("attention", tensors, params)
    tiled = run_op(request, threads=1)
    untiled = run_op(request, threads=1, env={"POCKETLLM_CPU_SCALAR_VSUM": "1"})
    assert np.array_equal(tiled, untiled), (
        f"q_len={q_len}: the tiled weighted sum does not reproduce the "
        f"row-at-a-time one -- worst |difference| "
        f"{float(np.max(np.abs(tiled - untiled)))}"
    )
    # And the comparison is not between two constants.
    assert float(np.max(np.abs(untiled))) > 0.0


# --------------------------------------------------------------------------
# 3. The token sequence, at one thread and at eight.
# --------------------------------------------------------------------------


def _generate(steps: int, threads: int) -> str:
    """The completion text from `pocketllm run`, at a chosen thread count.

    The Python CLI rather than the raw `pocketllm-run` tool, because it is the
    entry point `test_cli_run.py` pins against llama.cpp and it detokenizes --
    the C tool prints ids and candidate logits, and the subject here is the
    token sequence, spelled the way a user reads it.  The CLI drives the same C
    core through `native.py`; `POCKETLLM_CPU_THREADS` reaches the kernels because
    the pool reads it once, when the shared object is first loaded.
    """
    result = subprocess.run(
        [
            sys.executable, "-m", "pocketllm", "run",
            "--model", str(CHECKPOINT),
            "--prompt", PROMPT,
            "--max-tokens", str(steps),
        ],
        capture_output=True,
        text=True,
        env=_thread_env(threads),
        timeout=900,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith(PROMPT), f"the prompt was not echoed: {result.stdout!r}"
    return result.stdout[len(PROMPT):].strip()


@pytest.mark.parametrize("threads", THREAD_COUNTS)
@needs_engine
@needs_checkpoint
def test_the_greedy_sequence_is_llama_cpps_at_every_thread_count(threads: int) -> None:
    """The primary gate.

    The token match is the only check that can catch a last-bit change in a
    vectorized sum that flips an `argmax`.  A tolerance cannot see it and the
    reference comparison cannot see it; a token can.  Both thread counts have to
    produce it, which also says the pool does not reorder anything the model
    depends on.
    """
    text = _generate(STEPS, threads)
    assert text.startswith("Paris"), (
        f"at {threads} thread(s) the completion began {text!r}, not 'Paris' -- "
        f"llama.cpp would emit ids ending at {PROMPT_IDS}"
    )


@needs_engine
@needs_checkpoint
def test_one_thread_and_eight_generate_the_same_text() -> None:
    """And the two thread counts agree with each other, text for text.

    A weaker statement than the llama.cpp match and a different failure mode:
    this catches a pool bug that happens to land inside the model's own margin,
    where the argmax still matches llama.cpp but the logits do not match
    themselves.
    """
    assert _generate(STEPS, 1) == _generate(STEPS, 8)


# --------------------------------------------------------------------------
# 4. The benchmark tool reports a number.
# --------------------------------------------------------------------------


@needs_tools
@needs_checkpoint
def test_the_benchmark_reports_finite_throughput() -> None:
    """`pocketllm-bench` prints a parseable rate, and it is a rate.

    A smoke test, not a performance gate: it exists so a bench that prints
    `inf`, `nan` or nothing -- a division by a zero timer, a median over an
    empty vector -- fails here rather than in a report.
    """
    if not bench_path().is_file():
        pytest.skip("build/pocketllm-bench is not built")
    result = subprocess.run(
        [
            str(bench_path()), str(CHECKPOINT),
            "--pp", "4", "--tg", "4", "--reps", "1", "--warmup", "0",
        ],
        capture_output=True,
        text=True,
        env=_thread_env(4),
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    rates = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "bench":
            rates[parts[1]] = float(parts[3])
    assert set(rates) == {"pp", "tg"}, f"expected a pp and a tg row, got {rates!r}"
    for name, rate in rates.items():
        assert rate == rate, f"{name} reported NaN"
        assert rate not in (float("inf"), float("-inf")), f"{name} reported infinity"
        assert rate > 0.0, f"{name} reported {rate}"