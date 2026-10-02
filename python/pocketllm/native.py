"""The `ctypes` bridge to the native engine.

The Python package is a host shell over the C core, and this module is the seam
between them. It is deliberately the *only* place that knows a shared library
exists: everything above it — the CLI, the server — calls the small Python
wrappers here and never `ctypes` directly, so there is one place to change when
the ABI moves and one place to look when a symbol is missing.

The engine is **optional at import time**. A wheel installs on a phone, and a
builder that never ran CMake has no `libpocketllm.so`; `import pocketllm` must
still work, and a test asserts it pulls in neither torch nor numpy. So the
library is located lazily by :func:`load` and its absence is an
:class:`EngineUnavailable` the caller reports, not an `ImportError` at import.

Search order, first hit wins:

1. ``$POCKETLLM_CORE_LIB`` — an explicit path, which is what a packaged app or
   a test harness with an out-of-tree build uses.
2. ``build/libpocketllm.so`` under the repository root — the default the
   CMake instructions in the README produce.
3. Nothing, and :class:`EngineUnavailable`.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ctypes import CDLL

__all__ = [
    "ABI_VERSION",
    "EngineUnavailable",
    "Engine",
    "engine_path",
    "is_available",
    "load",
]

#: The ABI major the Python side is written against. The native library reports
#: its own on :func:`Engine.abi_version`, and :func:`load` refuses a mismatch:
#: a major bump is a signature change, so a mismatch is a broken pairing and
#: calling into it anyway is the failure mode that produces a crash rather than
#: an exception.
ABI_VERSION = 1

#: The error buffer size the header documents. Duplicated rather than imported
#: because reading it from the header would mean this module parses C, and a
#: disagreement here is caught by the ABI test either way.
_ERR_CAP = 512


class EngineUnavailable(RuntimeError):
    """The native library is not built, or cannot be loaded on this host."""


def _repository_root() -> pathlib.Path:
    """The checkout root, from this file's location.

    ``python/pocketllm/native.py`` -> the directory two levels up from the
    package, which is where ``build/`` sits when the engine is built out of
    tree. An installed wheel has no such directory, and the walk simply finds
    nothing; that is the intended outcome, not an error.
    """
    return pathlib.Path(__file__).resolve().parents[2]


def engine_path() -> pathlib.Path | None:
    """The engine library this host would load, or ``None`` if there is none.

    Returning the path rather than a boolean so the CLI's diagnostics can print
    *which* library it found — "the engine is missing" and "the engine is
    loading the wrong build" are different problems.
    """
    override = os.environ.get("POCKETLLM_CORE_LIB")
    if override:
        candidate = pathlib.Path(override)
        return candidate if candidate.is_file() else None

    candidate = _repository_root() / "build" / "libpocketllm.so"
    return candidate if candidate.is_file() else None


def is_available() -> bool:
    """Whether :func:`load` would succeed on this host.

    A filesystem probe, not an import: it is the question the CLI asks when
    something is broken, so it must not itself be able to fail.
    """
    return engine_path() is not None


def _bind(lib: "CDLL") -> None:
    """Declare each function's types.

    `ctypes` defaults every argument and return to a C ``int``, which silently
    truncates a pointer on a 64-bit host. Declaring them is not optional
    documentation — without it the first call with a pointer argument crashes
    in a way that looks like a bug in the engine.
    """
    lib.pocketllm_abi_version.restype = ctypes.c_char_p
    lib.pocketllm_abi_version.argtypes = []

    lib.pocketllm_open.restype = ctypes.c_void_p
    lib.pocketllm_open.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]

    lib.pocketllm_close.restype = None
    lib.pocketllm_close.argtypes = [ctypes.c_void_p]

    lib.pocketllm_encode.restype = ctypes.c_int
    lib.pocketllm_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
    ]

    lib.pocketllm_decode.restype = ctypes.c_int
    lib.pocketllm_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]

    lib.pocketllm_forward.restype = ctypes.c_int
    lib.pocketllm_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]

    lib.pocketllm_reset.restype = ctypes.c_int
    lib.pocketllm_reset.argtypes = [ctypes.c_void_p]

    lib.pocketllm_argmax.restype = ctypes.c_int
    lib.pocketllm_argmax.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]

    lib.pocketllm_temperature.restype = ctypes.c_int
    lib.pocketllm_temperature.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float,
    ]

    lib.pocketllm_sample.restype = ctypes.c_int
    lib.pocketllm_sample.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
    ]


def load(path: pathlib.Path | str | None = None) -> "CDLL":
    """Load and bind the engine library.

    Raises :class:`EngineUnavailable` when there is no library to load, or when
    the one found was built against a different ABI major.
    """
    if path is None:
        found = engine_path()
        if found is None:
            raise EngineUnavailable(
                "the native engine is not built; run `cmake -B build -S src && "
                "cmake --build build` from the repository root, or set "
                "POCKETLLM_CORE_LIB to an existing libpocketllm.so"
            )
        path = found
    else:
        path = pathlib.Path(path)
        if not path.is_file():
            raise EngineUnavailable(f"no engine library at {path}")

    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:  # a bad ELF, a wrong architecture, a missing dep
        raise EngineUnavailable(f"cannot load {path}: {exc}") from exc

    _bind(lib)

    reported = lib.pocketllm_abi_version()
    major = _major(reported.decode("ascii", "replace") if reported else "")
    if major != ABI_VERSION:
        raise EngineUnavailable(
            f"{path} speaks ABI {reported.decode('ascii', 'replace') if reported else '?'}; "
            f"this build of pocketllm needs major {ABI_VERSION}"
        )
    return lib


def _major(version: str) -> int:
    """The integer before the dot in an ``"M.m"`` version, or ``-1``.

    A library that reports something unparseable is treated as a mismatch
    rather than as agreement: the string is the handshake, and a handshake that
    cannot be read has failed.
    """
    head, _, _ = version.partition(".")
    try:
        return int(head)
    except ValueError:
        return -1


class Engine:
    """An owned session, with the C pointers kept in one place.

    This is a context manager because the session owns a file descriptor, an
    mmap and an arena: leaving one open per call is the leak that matters. Each
    method raises :class:`EngineUnavailable` on a negative return rather than
    returning a sentinel, so a caller cannot mistake a failure for a result.
    """

    def __init__(self, lib: "CDLL", handle: int) -> None:
        self._lib = lib
        self._handle = handle

    @classmethod
    def open(cls, gguf_path: str, backend: str = "cpu", *, lib: "CDLL | None" = None) -> "Engine":
        lib = load() if lib is None else lib
        err = ctypes.create_string_buffer(_ERR_CAP)
        handle = lib.pocketllm_open(
            gguf_path.encode(),
            backend.encode(),
            err,
            _ERR_CAP,
        )
        if not handle:
            raise EngineUnavailable(_message(err, "pocketllm_open failed"))
        return cls(lib, handle)

    def close(self) -> None:
        if self._handle:
            self._lib.pocketllm_close(self._handle)
            self._handle = 0

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @staticmethod
    def abi_version() -> str:
        reported = load().pocketllm_abi_version()
        return reported.decode("ascii", "replace") if reported else ""

    def encode(self, text: str, add_special: bool = True, parse_special: bool = False) -> list[int]:
        """Tokenize ``text``, sizing the output buffer from a first call.

        The header documents that an undersized buffer fails with a count
        rather than writing partially, so the two-call pattern is: ask with a
        null pointer to learn the length, then ask again with room for it.

        ``parse_special`` defaults to false and mirrors ``llama_tokenize``: a
        caller feeding a chat template passes true so that ``<|im_start|>``
        becomes one token, and a caller feeding user-typed text leaves it
        false so that a literal mention stays literal.
        """
        n = self._lib.pocketllm_encode(
            self._handle, text.encode(), int(bool(add_special)), int(bool(parse_special)), None, 0
        )
        if n < 0:
            raise EngineUnavailable(f"pocketllm_encode failed ({n})")
        if n == 0:
            return []
        out = (ctypes.c_int32 * n)()
        written = self._lib.pocketllm_encode(
            self._handle, text.encode(), int(bool(add_special)), int(bool(parse_special)), out, n
        )
        if written < 0:
            raise EngineUnavailable(f"pocketllm_encode failed ({written})")
        return [int(out[i]) for i in range(written)]

    def decode(self, ids: list[int]) -> str:
        n = len(ids)
        if n == 0:
            return ""
        arr = (ctypes.c_int32 * n)(*ids)
        # A generous fixed buffer for the skeleton; the real growth strategy
        # belongs with the tokenizer, where the bytes-per-token bound is known.
        cap = max(256, 8 * n)
        out = ctypes.create_string_buffer(cap)
        written = self._lib.pocketllm_decode(self._handle, arr, n, out, cap)
        if written < 0:
            raise EngineUnavailable(f"pocketllm_decode failed ({written})")
        return out.value.decode("utf-8", "replace")

    def forward(self, tokens: list[int], vocab: int = 151_936) -> list[float]:
        """Run ``tokens`` and return the logits for the last position."""
        n = len(tokens)
        if n == 0:
            raise ValueError("forward needs at least one token")
        arr = (ctypes.c_int32 * n)(*tokens)
        out = (ctypes.c_float * vocab)()
        written = self._lib.pocketllm_forward(self._handle, arr, n, out, vocab)
        if written < 0:
            raise EngineUnavailable(f"pocketllm_forward failed ({written})")
        return [float(out[i]) for i in range(written)]

    def reset(self) -> None:
        if self._lib.pocketllm_reset(self._handle) < 0:
            raise EngineUnavailable("pocketllm_reset failed")

    @staticmethod
    def argmax(logits: list[float]) -> int:
        if not logits:
            raise ValueError("argmax needs at least one logit")
        arr = (ctypes.c_float * len(logits))(*logits)
        return int(load().pocketllm_argmax(arr, len(logits)))


def _message(err: "ctypes.Array[ctypes.c_char]", fallback: str) -> str:
    """The engine's own error message, or ``fallback`` when it wrote none."""
    text = err.value.decode("utf-8", "replace") if err.value else ""
    return text or fallback