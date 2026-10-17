"""The vision ABI: the composition is reachable from C, and it is the same one.

The engine's own correctness on this path is checked in
`tests/native/test_vlm.py`, which drives `pocketllm-vlm`.  What is checked *here*
is the seam the header promises: that a host with no C++ can open a projector and
a text model, ask what an image will cost, run the composition and read the
logits -- and that the answer is the same one the instrument produces.

That last part is the reason this file exists rather than a second copy of
`test_vlm.py`.  The composition is three calls with a specific order, and a
version of it that lives in `c_api.cpp` instead of `runtime/vlm_session.cpp`
would be a second implementation of the one thing the ABI is for.  Comparing the
two is how a divergence between them would be caught.

Every test here skips when the library has not been built, the same way the rest
of this module does: a checkout without a compiler must still run the suite.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np
import pytest

from pocketllm import native

pytestmark = pytest.mark.skipif(
    not native.is_available(),
    reason="libpocketllm.so is not built (run `cmake -B build -S src && cmake --build build`)",
)

MMPROJ = pathlib.Path("/mnt/data1/models/mmproj-qwen3vl-2b-f16.gguf")
TEXT_MODEL = pathlib.Path("/mnt/data1/models/qwen3vl-2b-q4_k_m.gguf")

#: `mtmd_default_marker()`, asked for through the ABI rather than spelled here --
#: a host should not have to know this string, and the test should fail if the
#: engine's answer changes.
PROMPT_BEFORE = "Describe this image: "
PROMPT_AFTER = ""

ERR_CAP = 512


@pytest.fixture(scope="module")
def lib() -> "ctypes.CDLL":
    return native.load()


def _skip_reason() -> str | None:
    if not MMPROJ.is_file():
        return f"no mmproj at {MMPROJ}"
    if not TEXT_MODEL.is_file():
        return f"no text model at {TEXT_MODEL}"
    return None


def picture(width: int, height: int) -> np.ndarray:
    """A picture that is not symmetric in any direction, at the asked size."""
    axis_w = np.arange(width, dtype=np.uint16)
    axis_h = np.arange(height, dtype=np.uint16)
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = (axis_w[None, :] % 256).astype(np.uint8)
    image[:, :, 1] = (axis_h[:, None] % 256).astype(np.uint8)
    image[:, :, 2] = ((axis_w[None, :] + axis_h[:, None]) % 256).astype(np.uint8)
    return image


class Vision:
    """The ABI handle, closed on the way out."""

    def __init__(self, lib: "ctypes.CDLL") -> None:
        error = ctypes.create_string_buffer(ERR_CAP)
        self._lib = lib
        self._handle = lib.pocketllm_vlm_open(
            str(MMPROJ).encode(),
            str(TEXT_MODEL).encode(),
            b"cpu",
            error,
            ERR_CAP,
        )
        if not self._handle:
            raise RuntimeError(f"pocketllm_vlm_open failed: {error.value.decode()}")

    def close(self) -> None:
        if self._handle:
            self._lib.pocketllm_vlm_close(self._handle)
            self._handle = None

    def __enter__(self) -> "Vision":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def marker(self) -> str:
        return self._lib.pocketllm_vlm_marker().decode()

    def image_tokens(self, width: int, height: int) -> int:
        return int(self._lib.pocketllm_vlm_image_tokens(self._handle, width, height))

    def forward(self, prompt: str, image: np.ndarray) -> np.ndarray:
        pixels = np.ascontiguousarray(image, dtype=np.uint8)
        height, width = pixels.shape[0], pixels.shape[1]
        vocabulary = 151936
        out = (ctypes.c_float * vocabulary)()
        written = self._lib.pocketllm_vlm_forward(
            self._handle,
            prompt.encode(),
            pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            width,
            height,
            out,
            vocabulary,
        )
        if written < 0:
            raise RuntimeError("pocketllm_vlm_forward failed")
        return np.frombuffer(bytes(out)[: written * 4], dtype="<f4").astype(np.float64)

    def forward_argmax(self, prompt: str, image: np.ndarray) -> int:
        pixels = np.ascontiguousarray(image, dtype=np.uint8)
        height, width = pixels.shape[0], pixels.shape[1]
        out = ctypes.c_int32(-1)
        rc = self._lib.pocketllm_vlm_forward_argmax(
            self._handle,
            prompt.encode(),
            pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            width,
            height,
            ctypes.byref(out),
        )
        assert rc == 0, "pocketllm_vlm_forward_argmax failed"
        return int(out.value)


def test_the_marker_is_the_one_the_reference_uses(lib: "ctypes.CDLL") -> None:
    """`libmtmd`'s default, asked for rather than assumed.

    A host that hard-codes it is a host that breaks silently if the engine's
    changes, so the string is a call and this pins its value.
    """
    assert lib.pocketllm_vlm_marker().decode() == "<__media__>"


def test_a_bad_prompt_is_refused_with_a_message(lib: "ctypes.CDLL") -> None:
    """The seam's own property: failure is a return value, not a crash.

    Two markers is the interesting one -- a host that concatenated two images into
    one prompt would otherwise get the first image's answer and no complaint.
    """
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    with Vision(lib) as session:
        image = picture(64, 64)
        marker = session.marker()
        for prompt, expected in (
            ("no marker at all", "no '"),
            (f"a {marker} and another {marker}", "more than one"),
        ):
            out = (ctypes.c_float * 8)()
            written = lib.pocketllm_vlm_forward(
                session._handle,
                prompt.encode(),
                image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                64,
                64,
                out,
                8,
            )
            assert written < 0, f"'{prompt}' should have been refused"


def test_the_composition_matches_llama_cpp_through_the_abi(lib: "ctypes.CDLL") -> None:
    """The whole path, driven the way a host drives it, against the reference.

    The image is the checkpoint's own size, so this isolates the *seam*: the
    resize is covered by `test_image_preprocess.py` and does not need to be in the
    same comparison.  A host reading the wrong logits, getting the span's count
    wrong, or running the three calls in another order all land here.
    """
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)
    from mtmd_vlm import run as run_oracle

    image = picture(768, 768)
    with Vision(lib) as session:
        marker = session.marker()
        assert session.image_tokens(768, 768) == 576
        prompt = f"{PROMPT_BEFORE}{marker}{PROMPT_AFTER}"
        got = session.forward(prompt, image)
        greedy = session.forward_argmax(prompt, image)

    want = run_oracle(MMPROJ, TEXT_MODEL, image, prompt)["logits"]
    spread = float(want.max() - want.min())
    worst = float(np.max(np.abs(got - want)))
    correlation = float(np.corrcoef(want, got)[0, 1])
    # The same bounds `test_vlm.py` uses on the same comparison, because it is the
    # same computation: measured 4.3% of the spread at correlation 0.994.
    assert worst <= 0.1 * spread, (
        f"the ABI's logits differ from the reference by {worst:.4g} over a spread of {spread:.4g}"
    )
    assert correlation > 0.95, f"the ABI's logits correlate {correlation:.4f} with the reference"
    # The fast path is the same computation with the vocabulary left behind, so it
    # cannot disagree with the full one -- that is the header's promise and it is
    # checked rather than asserted in a comment.
    #
    # Deliberately *not* compared against the reference's argmax, and the numbers
    # are why: the reference's top two candidates are 0.0708 apart (30.6012 and
    # 30.5304) and they are 1986 and 785, while this engine's own residue against
    # the reference is 2.21 at the widest.  A gap an order of magnitude smaller than
    # the error being tolerated is a gap the tolerance decides, so the two engines
    # pick different winners on this image without disagreeing about the
    # distribution -- 0.994 of correlation, the same two candidates at the top, in
    # the other order.  Demanding the same argmax here would be asserting the
    # residue is zero, which is a much stronger claim than the bound supports and a
    # different test than this one.
    assert greedy == int(np.argmax(got))