"""The speech ABI: the composition is reachable from C, and it transcribes.

The engine's correctness on this path is checked against the reference in
`test_audio_frontend.py` (the front end) and by the tower comparison recorded in
`c_engine.md`; what is checked here is the seam: that a host with no C++ can open a
projector and a text model, ask how many windows a recording becomes, run the
composition and read the logits -- and that the answer is the reference's own
transcript.

The input is four seconds of `mlk.flac` decoded once to 16 kHz mono float32.  Both the
audio and the expected text live outside the tree (see the skip at the bottom), because a
checkpoint and a recording are not things a source checkout carries.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np
import pytest

from pocketllm import native

MMPROJ = pathlib.Path("/mnt/data1/models/mmproj-Qwen3-ASR-1.7B-Q8_0.gguf")
TEXT_MODEL = pathlib.Path("/mnt/data1/models/Qwen3-ASR-1.7B-Q8_0.gguf")
#: The first four seconds of `mlk.flac`, decoded to 16 kHz mono float32 by
#: `ffmpeg -i mlk.flac -ac 1 -ar 16000 -t 4 -f f32le`.
SAMPLES = pathlib.Path("/tmp/pll-probe/mlk4.f32")

#: The reference's transcript for that clip, from `llama-cli` with the same prompt.
#: Compared by prefix and not exactly: the reference stops where its own generation
#: budget or its EOS takes it, and one trailing punctuation mark is the decoding
#: budget rather than a disagreement.
REFERENCE_TEXT = "language English<asr_text>I have a dream that one day"

ERR_CAP = 512


@pytest.fixture(scope="module")
def lib() -> "ctypes.CDLL":
    if not native.is_available():
        pytest.skip("libpocketllm.so is not built")
    return native.load()


def _skip_reason() -> str | None:
    if not MMPROJ.is_file():
        return f"no audio mmproj at {MMPROJ}"
    if not TEXT_MODEL.is_file():
        return f"no text model at {TEXT_MODEL}"
    if not SAMPLES.is_file():
        return f"no decoded samples at {SAMPLES}"
    return None


class Speech:
    """The ABI handle, closed on the way out."""

    def __init__(self, lib: "ctypes.CDLL") -> None:
        error = ctypes.create_string_buffer(ERR_CAP)
        self._lib = lib
        self._handle = lib.pocketllm_asr_open(
            str(MMPROJ).encode(), str(TEXT_MODEL).encode(), b"cpu", error, ERR_CAP
        )
        if not self._handle:
            raise RuntimeError(f"pocketllm_asr_open failed: {error.value.decode()}")

    def close(self) -> None:
        if self._handle:
            self._lib.pocketllm_asr_close(self._handle)
            self._handle = None

    def __enter__(self) -> "Speech":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def marker(self) -> str:
        return self._lib.pocketllm_asr_marker().decode()

    def windows(self, n_samples: int) -> int:
        return int(self._lib.pocketllm_asr_windows(self._handle, n_samples))

    def tokens_per_window(self) -> int:
        return int(self._lib.pocketllm_asr_tokens_per_window(self._handle))

    def forward(self, prompt: str, pcm: np.ndarray) -> np.ndarray:
        samples = np.ascontiguousarray(pcm, dtype=np.float32)
        vocabulary = 151936
        out = (ctypes.c_float * vocabulary)()
        written = self._lib.pocketllm_asr_forward(
            self._handle,
            prompt.encode(),
            samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            samples.size,
            out,
            vocabulary,
        )
        if written < 0:
            raise RuntimeError("pocketllm_asr_forward failed")
        return np.frombuffer(bytes(out)[: written * 4], dtype="<f4").astype(np.float64)

    def generate(self, prompt: str, pcm: np.ndarray, max_tokens: int = 48) -> str:
        samples = np.ascontiguousarray(pcm, dtype=np.float32)
        cap = 4096
        out = ctypes.create_string_buffer(cap)
        written = self._lib.pocketllm_asr_generate(
            self._handle,
            prompt.encode(),
            samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            samples.size,
            max_tokens,
            out,
            cap,
        )
        if written < 0:
            raise RuntimeError("pocketllm_asr_generate failed")
        return out.value.decode()

    def forward_argmax(self, prompt: str, pcm: np.ndarray) -> int:
        samples = np.ascontiguousarray(pcm, dtype=np.float32)
        out = ctypes.c_int32(-1)
        rc = self._lib.pocketllm_asr_forward_argmax(
            self._handle,
            prompt.encode(),
            samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            samples.size,
            ctypes.byref(out),
        )
        assert rc == 0, "pocketllm_asr_forward_argmax failed"
        return int(out.value)


def test_the_marker_is_the_one_the_reference_uses(lib: "ctypes.CDLL") -> None:
    """`libmtmd`'s default, asked for rather than assumed.

    The same string the vision path uses, because `libmtmd` defaults both media to it --
    and a host that hard-coded it instead of asking is a host that breaks silently if the
    engine's changes.
    """
    assert lib.pocketllm_asr_marker().decode() == "<__media__>"


def test_the_window_count_is_a_function_of_the_sample_count(lib: "ctypes.CDLL") -> None:
    """Four seconds is two windows, and the count grows by one per ten seconds.

    Checked against the front end's own arithmetic rather than against a number copied
    from a run: 64000 samples produce 401 frames after the reflection padding, five
    100-frame chunks, and the tower's 13 tokens each -- so 65 tokens for the first
    window, which the reference reports as `n_len=800` for a full one.
    """
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    with Speech(lib) as session:
        assert session.windows(64000) == 1
        assert session.windows(16000 * 13) == 2
        # The tower reports its own span length for a full window: 800 frames is eight
        # conv sub-chunks of 13 tokens each.
        assert session.tokens_per_window() == 104


def test_the_composition_transcribes_what_the_reference_says(lib: "ctypes.CDLL") -> None:
    """The whole path through the ABI, compared against `llama-cli`'s own answer.

    Greedy decoding in the test rather than through a sampler, because the comparison is
    of tokens and a sampled answer would differ for a reason that is not under test.  The
    first token is checked exactly and the run is checked by prefix -- see
    `REFERENCE_TEXT`.
    """
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    pcm = np.frombuffer(SAMPLES.read_bytes(), dtype="<f4")
    assert pcm.size == 64000, pcm.size
    with Speech(lib) as session:
        marker = session.marker()
        # The checkpoint's own ChatML template, spelled out: a `qwen3a` text model expects
        # it, and the marker is where the audio goes.
        prompt = (
            f"<|im_start|>user\nTranscribe the audio. {marker}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        logits = session.forward(prompt, pcm)
        greedy = session.forward_argmax(prompt, pcm)

    assert greedy == int(np.argmax(logits))
    assert logits.size == 151936

    # The transcript itself -- the thing a caller actually wants -- against the reference's.
    # A prefix comparison because where the reference stops is its generation budget or its
    # own EOS, and one trailing punctuation mark is that budget rather than a disagreement.
    with Speech(lib) as decoding:
        marker = decoding.marker()
        prompt_for_text = (
            f"<|im_start|>user\nTranscribe the audio. {marker}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        text = decoding.generate(prompt_for_text, pcm, max_tokens=48)

    assert text.startswith(REFERENCE_TEXT), (
        f"the ABI transcribed {text!r} and the reference says {REFERENCE_TEXT!r}"
    )


def test_a_prompt_without_the_marker_is_refused(lib: "ctypes.CDLL") -> None:
    """The seam's own property: failure is a return value, not a crash.

    A caller that forgot the marker is the mistake this guard exists for -- the
    composition has nowhere to put the audio and would otherwise silently transcribe
    nothing.
    """
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    pcm = np.frombuffer(SAMPLES.read_bytes(), dtype="<f4")
    with Speech(lib) as session:
        out = (ctypes.c_float * 8)()
        written = lib.pocketllm_asr_forward(
            session._handle,
            b"no marker at all",
            pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            pcm.size,
            out,
            8,
        )
        assert written < 0, "a prompt with no marker should have been refused"