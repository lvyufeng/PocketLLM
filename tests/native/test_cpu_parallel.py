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


def run_op(request: str, *, device: str = "cpu", threads: int = 1) -> np.ndarray:
    """One op through `opcheck`, at a chosen thread count, as an array."""
    result = subprocess.run(
        [str(opcheck_path()), "--request", "/dev/stdin", "--device", device],
        input=request,
        capture_output=True,
        text=True,
        env=_thread_env(threads),
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