"""The `ctypes` bridge to the S600 `libxlm.so` delegate.

This is the second of the tree's two native seams, and it is worth saying at the
top why there are two.  :mod:`pocketllm.native` bridges `libpocketllm.so`, this
tree's own engine, which exposes a token-level surface — encode, forward, the
logits — and lets the host do everything above the kernel.  `libxlm.so` is not
that.  It is Horizon's text-in / text-out runtime for the Nash BPU: a caller
hands it a **string**, and text comes back through a callback as it decodes.
There is no logits accessor, no token-id surface (`XLM_INPUT_TOKEN` is annotated
"not support yet" in the header), and no way to ask it for one token at a time.
The model it runs is an AOT-compiled `.hbm` graph whose shapes were frozen when
it was built.

So this bridge exposes a **different shape** from `native.py`, and the shape is
the point: an :class:`XlmEngine` is opened over a checkpoint *directory* and a
config *file* rather than a single `.gguf`, a request carries a prompt rather
than token ids, and sampling is the delegate's rather than the host's.

Like `native.py`, the library is optional at import time and located lazily: a
host without the SDK gets an :class:`XlmUnavailable` from :func:`load`, not an
`ImportError` from importing this module.  Search order, first hit wins:

1. ``$POCKETLLM_XLM_LIB`` — an explicit path, for a relocated SDK or a test.
2. The SDK's own tree under ``$HOME``, newest version first — the layout the
   D-Robotics installer produces.
3. ``libxlm.so`` on the system loader path.
4. Nothing, and :class:`XlmUnavailable`.

The struct layouts below are transcribed from ``offsetof`` output produced by
the board's own compiler against the SDK header, not derived by hand: an
anonymous union that is one field out does not fail to compile, it reads a
garbage pointer and segfaults inside `xlm_init` — which is exactly what an
earlier hand-written probe of this API did.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import pathlib
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ctypes import CDLL

__all__ = [
    "CALLBACK",
    "CommonParams",
    "Input",
    "InputToken",
    "LmRequest",
    "MultiModal",
    "Performance",
    "Priority",
    "Result",
    "Sampling",
    "VlaParams",
    "VlmParams",
    "XlmEngine",
    "XlmUnavailable",
    "XlmModelType",
    "is_available",
    "library_path",
    "load",
]

#: The failure :func:`load` raises when the delegate is not on this host.
class XlmUnavailable(RuntimeError):
    """The S600 `libxlm.so` is not installed, or cannot be loaded here."""


class XlmModelType:
    """`xlm_model_type`, as named constants.

    A class rather than an ``enum.IntEnum`` because the values are handed
    straight to ``ctypes`` as ``c_int`` and an ``IntEnum`` would have to be
    unwrapped at every call site for no gain.
    """

    INTERNVL = 0
    DEEPSEEK = 1
    QWEN = 2
    LLAMA = 3  # not supported by the delegate
    INTERNLM = 4
    OMNI = 5
    QWEN_VL = 6
    QWEN2_5 = 7
    PI0 = 8
    QWEN3 = 9
    WHISPER = 10
    VLM = 11


#: `xlm_state_t`, the values the callback is handed.
STATE_START = 0
STATE_END = 1
STATE_RUNNING = 2
STATE_ERROR = 3

#: `xlm_input_type_e` — the only one this bridge builds is a prompt.
INPUT_PROMPT = 0

#: `xlm_infer_backend_e` — let the scheduler pick a BPU core.
INFER_BACKEND_ANY = 0


class Sampling(ctypes.Structure):
    """`common_params_sampling_t` (44 bytes).

    **The delegate samples, not the host.**  There is no logits surface for the
    host to draw from, so a caller's temperature and top-k travel here and the
    text that comes back is already the delegate's choice.  ``temp <= 0`` is the
    header's spelling of "sample greedily", and it is what :meth:`XlmEngine.infer`
    sets when no sampling is asked for, so the default is deterministic.
    """

    _fields_ = [
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("min_p", ctypes.c_float),
        ("temp", ctypes.c_float),
        ("typ_p", ctypes.c_float),
        ("min_keep", ctypes.c_int32),
        ("penalty_last_n", ctypes.c_int32),
        ("penalty_repeat", ctypes.c_float),
        ("penalty_freq", ctypes.c_float),
        ("penalty_present", ctypes.c_float),
        ("do_sample", ctypes.c_bool),
    ]


class VlmParams(ctypes.Structure):
    _fields_ = [
        ("visual_model_path", ctypes.c_char_p),
        ("audio_model_path", ctypes.c_char_p),
        ("text_model_path", ctypes.c_char_p),
        ("embed_tokens", ctypes.c_char_p),
        ("online_mode", ctypes.c_bool),
    ]


class VlaParams(ctypes.Structure):
    _fields_ = [
        ("siglip_model_path", ctypes.c_char_p),
        ("paligemma_model_path", ctypes.c_char_p),
        ("action_model_path", ctypes.c_char_p),
        ("norm_stats_path", ctypes.c_char_p),
    ]


class CommonParams(ctypes.Structure):
    """`xlm_common_params_t` (176 bytes), returned **by value** by the factory.

    Three of these fields are what an init actually needs — the `.hbm`, the
    tokenizer directory, and the JSON config the demo passes on its command
    line.  The config path matters more than its name suggests: the BPU core
    list, the context size and the model type all reach the delegate through
    that file, so a caller who omits it gets whatever the model was compiled
    with rather than what the request asked for.
    """

    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("vlm_param", VlmParams),
        ("vla_param", VlaParams),
        ("token_config_path", ctypes.c_char_p),
        ("config_path", ctypes.c_char_p),
        ("k_cache_int8", ctypes.c_bool),
        ("model_type", ctypes.c_int),
        ("context_size", ctypes.c_int32),
        ("max_img_cnt", ctypes.c_int32),
        ("sampling", Sampling),
        ("prompt_file", ctypes.c_char_p),
        ("path_prompt_cache", ctypes.c_char_p),
    ]


class InputToken(ctypes.Structure):
    _fields_ = [
        ("tokens", ctypes.POINTER(ctypes.c_int32)),
        ("tokens_size", ctypes.c_int32),
        ("prompt", ctypes.c_char_p),
    ]


class MultiModal(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_num", ctypes.c_int32),
        ("images", ctypes.c_void_p),
        ("has_prompt", ctypes.c_bool),
    ]


class Priority(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("priority", ctypes.c_int32)]


class _RequestUnion(ctypes.Union):
    """The anonymous union inside `xlm_lm_request_t`, at offset 32.

    Only ``prompt`` is spelled with its true type; the VLA arm is left as raw
    bytes because its internal layout is not needed to *size* the union, and a
    transcription of a struct this bridge never fills is one more thing to be
    wrong about.  The size is pinned by :data:`_EXPECTED_SIZES` below rather
    than trusted.
    """

    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("token", InputToken),
        ("multi_modal_request", MultiModal),
        ("vla_request", ctypes.c_byte * 72),
    ]


class LmRequest(ctypes.Structure):
    """`xlm_lm_request_t` (144 bytes).  One request in a batch.

    ``new_chat`` is the equivalent of the engine's context reset: the delegate
    keeps conversation state between calls, so a fresh prompt has to say so or
    it is appended to the previous turn.  The demo sets it back to ``True``
    after every prompt in single-turn mode, which is the behaviour a stateless
    server wants and what :meth:`XlmEngine.infer` does.
    """

    _fields_ = [
        ("request_id", ctypes.c_int32),
        ("type", ctypes.c_int),
        ("new_chat", ctypes.c_bool),
        ("prompt_json", ctypes.c_char_p),
        ("need_partial_result", ctypes.c_bool),
        ("_payload", _RequestUnion),
        ("system_prompt", ctypes.c_char_p),
        ("chat_template", ctypes.c_char_p),
        ("infer_backend", ctypes.c_int),
        ("priority", Priority),
        ("ppl", ctypes.c_void_p),
    ]


class Input(ctypes.Structure):
    _fields_ = [("request_num", ctypes.c_int32), ("requests", ctypes.POINTER(LmRequest))]


class Performance(ctypes.Structure):
    """`xlm_model_performance_t` (88 bytes), reported once at ``STATE_END``.

    The delegate measures its own decode, which is the one number worth having
    from a board whose whole question is whether the substitution is fast
    enough.  ``prefill_tps``/``decode_tps``/``ttft`` are the same three the
    demo prints.
    """

    _fields_ = [
        ("vit_cost", ctypes.c_double),
        ("vit_infer_cost", ctypes.c_double),
        ("prefill_token_num", ctypes.c_int64),
        ("prefill_tps", ctypes.c_double),
        ("decode_token_num", ctypes.c_int64),
        ("decode_tps", ctypes.c_double),
        ("ttft", ctypes.c_double),
        ("tpot", ctypes.c_double),
        ("end_to_end_cost", ctypes.c_double),
        ("accept_rate", ctypes.c_float),
        ("avg_accept_count", ctypes.c_float),
        ("asr_rtf", ctypes.c_double),
    ]


class Result(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("request_id", ctypes.c_int32),
        ("performance", Performance),
    ]


#: `void (*)(xlm_result_t *, xlm_state_t, void *)`.
CALLBACK = ctypes.CFUNCTYPE(None, ctypes.POINTER(Result), ctypes.c_int, ctypes.c_void_p)

#: Sizes the transcription is checked against at import.  A layout that has moved
#: is caught here, where the message names the struct, rather than as a garbage
#: read inside `xlm_init`.
_EXPECTED_SIZES = {
    Sampling: 44,
    VlmParams: 40,
    VlaParams: 32,
    CommonParams: 176,
    LmRequest: 144,
    Input: 16,
    Performance: 88,
    Result: 104,
}

for _struct, _size in _EXPECTED_SIZES.items():
    _actual = ctypes.sizeof(_struct)
    if _actual != _size:  # pragma: no cover - a transcription slip, not a runtime path
        raise RuntimeError(
            f"the {_struct.__name__} layout is {_actual} bytes, the SDK header's is {_size}; "
            "the structs in pocketllm/xlm.py no longer match the board's libxlm.so"
        )
del _struct, _size, _actual


def library_path() -> pathlib.Path | None:
    """The `libxlm.so` this host would load, or ``None``.

    Returning the path rather than a boolean so a diagnostic can print *which*
    SDK was found: two SDK versions under ``$HOME`` is the normal state after an
    upgrade, and "the delegate is missing" and "the delegate is the old one" are
    different problems.
    """
    override = os.environ.get("POCKETLLM_XLM_LIB")
    if override:
        candidate = pathlib.Path(override)
        return candidate if candidate.is_file() else None

    # Newest SDK first, so an upgrade that leaves the old tree in place does not
    # silently keep loading it.  The version is the directory's own name, which
    # sorts correctly as a string for the ``1.0.2``-style releases the installer
    # produces.
    pattern = str(
        pathlib.Path.home()
        / "llm_sdk"
        / "D-Robotics_LLM_S600_*"
        / "oellm_runtime"
        / "lib"
        / "libxlm.so"
    )
    for match in sorted(glob.glob(pattern), reverse=True):
        candidate = pathlib.Path(match)
        if candidate.is_file():
            return candidate

    try:
        found = ctypes.util.find_library("xlm")
    except Exception:  # pragma: no cover - find_library shells out; absence is normal
        found = None
    return pathlib.Path(found) if found else None


def is_available() -> bool:
    """Whether :func:`load` would succeed here.  A filesystem probe, never throws."""
    return library_path() is not None


def _bind(lib: "CDLL") -> None:
    """Declare each function's types.

    `ctypes` defaults every argument and return to a C ``int``, which silently
    truncates a 64-bit handle.  That is not a hypothetical here: ``xlm_init``
    takes its handle as an **out-parameter** (``void **``), so a wrong
    ``argtypes`` on it is what turns a valid call into a segfault.
    """
    lib.xlm_create_default_param.restype = CommonParams
    lib.xlm_create_default_param.argtypes = []

    lib.xlm_init.restype = ctypes.c_int
    lib.xlm_init.argtypes = [
        ctypes.POINTER(CommonParams),
        CALLBACK,
        ctypes.POINTER(ctypes.c_void_p),
    ]

    lib.xlm_infer.restype = ctypes.c_int
    lib.xlm_infer.argtypes = [ctypes.c_void_p, ctypes.POINTER(Input), ctypes.c_void_p]

    lib.xlm_destroy.restype = ctypes.c_int
    lib.xlm_destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]


def load(path: pathlib.Path | str | None = None) -> "CDLL":
    """Load `libxlm.so`, or raise :class:`XlmUnavailable` with a next step.

    The SDK's own libraries must already be findable — ``libxlm.so`` links
    against `libhbrt4.so` from the same directory — so the failure this reports
    most often is not a missing file but an unset ``LD_LIBRARY_PATH``.  The
    message names both, because the two fixes are different.
    """
    resolved = pathlib.Path(path) if path is not None else library_path()
    if resolved is None:
        raise XlmUnavailable(
            "the S600 `libxlm.so` is not installed on this host. Set POCKETLLM_XLM_LIB "
            "to an existing libxlm.so, or install the D-Robotics LLM SDK under "
            "~/llm_sdk/."
        )
    try:
        lib = ctypes.CDLL(str(resolved))
    except OSError as exc:
        raise XlmUnavailable(
            f"{resolved} failed to load: {exc}. The SDK's own libraries must be on the "
            "loader path — source the SDK environment or add its lib/ directory to "
            "LD_LIBRARY_PATH."
        ) from exc
    _bind(lib)
    return lib


class XlmEngine:
    """An open delegate session: one `.hbm`, one config, text in, text out.

    **One request at a time, and the delegate is not asked about it.**  The
    same constraint the C session has, for the same reason: the delegate holds
    one conversation state and one KV cache, and concurrent calls do not fail —
    they interleave one prompt's tokens into another's answer.  This class does
    not lock; :class:`~pocketllm.server.xlm_backend.XlmBackend` does, the way
    its native twin does, because serialization is a *serving* decision and a
    library that took a lock would hide it from a caller who knows better.
    """

    def __init__(self, lib: "CDLL", handle: ctypes.c_void_p, params: CommonParams) -> None:
        self._lib = lib
        self._handle = handle
        #: Held for the session's life so the pointers it carries stay valid.
        #: `xlm_init` copies the values it needs, but the SDK does not promise
        #: it, and a freed path string read at the first decode is a crash with
        #: no stack that points here.
        self._params = params
        self._closed = False

    @classmethod
    def open(
        cls,
        model_path: str,
        tokenizer_dir: str,
        config_path: str,
        *,
        model_type: int = XlmModelType.QWEN3,
        context_size: int = 0,
        sampling: Sampling | None = None,
        callback: "Callable[[Result, int], None] | None" = None,
        lib: "CDLL | None" = None,
    ) -> "XlmEngine":
        """Open a delegate session.  Raises :class:`XlmUnavailable` on any refusal.

        ``context_size`` of 0 keeps the config file's own value rather than
        forcing the header's default: the `.hbm` was compiled for a specific
        chunk and cache size, so a caller overriding it from the host is more
        likely to disagree with the graph than to improve it.
        """
        loaded = lib if lib is not None else load()
        param = loaded.xlm_create_default_param()
        param.model_path = _encode(model_path)
        param.token_config_path = _encode(tokenizer_dir)
        param.config_path = _encode(config_path)
        param.model_type = int(model_type)
        if context_size:
            param.context_size = int(context_size)
        if sampling is not None:
            param.sampling = sampling

        state: dict[str, object] = {"sink": callback}
        ffi = _trampoline(state)

        handle = ctypes.c_void_p()
        status = loaded.xlm_init(ctypes.byref(param), ffi, ctypes.byref(handle))
        if status != 0 or not handle.value:
            raise XlmUnavailable(f"xlm_init refused the checkpoint (status {status})")
        engine = cls(loaded, handle, param)
        # The trampoline must outlive the session: the delegate keeps the
        # function pointer and calls it on every decode, so a collected
        # `CFUNCTYPE` object is a jump into freed memory.
        engine._ffi = ffi  # type: ignore[attr-defined]
        engine._state = state  # type: ignore[attr-defined]
        return engine

    def infer(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        on_chunk: "Callable[[str, int], None] | None" = None,
        timeout: float = 300.0,
    ) -> str:
        """Run one prompt to completion and return the text.

        ``on_chunk`` is called with ``(piece, state)`` for each callback the
        delegate makes as it decodes, which is the streaming surface: the
        delegate is synchronous — `xlm_infer` does not return until the answer
        is whole — so the pieces arrive during the call, not after it.

        **The returned text includes the delegate's own reasoning block.**  On
        a Qwen3 checkpoint with thinking enabled the text opens with ` thinking`,
        which is a property of the model, not of this bridge; splitting it is
        the protocol layer's job (``pocketllm.protocol.templating``), and doing
        it here would duplicate that decision in a second place.
        """
        if self._closed:
            raise XlmUnavailable("this XlmEngine has been closed")

        state = self._state  # type: ignore[attr-defined]
        # Reset both per call: the trampoline accumulates into `chunks` and
        # reports through `sink`, and a call that inherited the previous one's
        # list would return the concatenation of every answer so far.
        state["sink"] = on_chunk
        state["chunks"] = []
        state["error"] = False
        state["performance"] = None

        request = LmRequest()
        ctypes.memset(ctypes.byref(request), 0, ctypes.sizeof(request))
        request.type = INPUT_PROMPT
        # Single-turn: every `infer` starts a new conversation, so a prompt is
        # never appended to the previous one's context.
        request.new_chat = True
        request.infer_backend = INFER_BACKEND_ANY
        # `prompt` and `system_prompt` are `c_char_p`; assigning a bytes object
        # to the field stores the pointer, so the bytes have to outlive the call
        # — `_keep_alive` is what holds them.
        keep_alive = _set_pointer(request, "_payload", "prompt", prompt)
        if system_prompt is not None:
            keep_alive.append(_set_field(request, "system_prompt", system_prompt))

        requests = (LmRequest * 1)(request)
        batch = Input(request_num=1, requests=requests)
        status = self._lib.xlm_infer(self._handle, ctypes.byref(batch), None)
        # `keep_alive` and `requests` are read through the call; the call is
        # synchronous, so nothing past this line may touch them.  Binding them
        # to `_` keeps the references alive until here without a bare `del`.
        _ = keep_alive, requests
        if status != 0:
            raise XlmUnavailable(f"xlm_infer refused the request (status {status})")
        chunks: list[str] = state.get("chunks", [])  # type: ignore[assignment]
        return "".join(chunks)

    @property
    def last_performance(self) -> Any:
        """The `xlm_model_performance_t` the delegate reported at the last ``STATE_END``.

        A copy, or ``None`` before the first completed request.  It is the one
        place the delegate's own token counts and throughput are available --
        there is no token-id surface to count tokens any other way -- so a
        serving layer reads its ``prefill_token_num``/``decode_token_num`` here
        rather than inventing a count.
        """
        return self._state.get("performance")  # type: ignore[attr-defined]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._lib.xlm_destroy(ctypes.byref(self._handle))

    def __enter__(self) -> "XlmEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _trampoline(state: "dict[str, object]") -> CALLBACK:
    """The C callback, closing over the live-chunk sink and accumulator.

    The accumulator is reset per :meth:`XlmEngine.infer` by clearing ``chunks``,
    and the sink is swapped there too, so one trampoline serves every call.  The
    state dict is created in :meth:`open` and kept on the engine for exactly
    that reason: a closure that captured a local would be collected.
    """

    @CALLBACK
    def _callback(result: "ctypes.POINTER[Result]", status: int, _userdata: object) -> None:
        if not result:
            return
        if status == STATE_ERROR:
            state["error"] = True
            return
        text = result.contents.text
        if text:
            piece = text.decode("utf-8", "replace")
            chunks = state.setdefault("chunks", [])
            assert isinstance(chunks, list)
            chunks.append(piece)
            sink = state.get("sink")
            if callable(sink):
                sink(piece, status)
        if status == STATE_END:
            state["performance"] = result.contents.performance

    return _callback


def _encode(value: str) -> bytes:
    return str(value).encode("utf-8")


def _set_field(struct: ctypes.Structure, field: str, value: str) -> bytes:
    """Assign a ``c_char_p`` field a Python string and return the bytes to hold."""
    raw = _encode(value)
    setattr(struct, field, raw)
    return raw


def _set_pointer(
    struct: ctypes.Structure, union_field: str, member: str, value: str
) -> list[bytes]:
    """Assign ``struct.<union>.<member>`` a Python string and return what to hold.

    A ``c_char_p`` inside an anonymous union has no ``setattr`` path from the
    outer struct, so the pointer is written through the field's own address.
    The address is built from the field's declared ``offset`` rather than from
    ``addressof(getattr(...))``: reading a nested struct or union through
    ctypes hands back an object whose relation to the parent buffer is an
    implementation detail, while the offsets are the layout the size assertions
    above already pin down.
    """
    raw = _encode(value)
    offset = getattr(type(struct), union_field).offset + getattr(
        type(getattr(struct, union_field)), member
    ).offset
    ctypes.cast(
        ctypes.addressof(struct) + offset, ctypes.POINTER(ctypes.c_char_p)
    )[0] = raw
    return [raw]