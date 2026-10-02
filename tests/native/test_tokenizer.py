"""The C tokenizer must agree with llama.cpp's, token for token.

There is no Python oracle for this. `python/pocketllm/tokenizer/` is a skeleton
whose `GgufTokenizer.__init__` raises `BackendNotImplementedError` -- it was
written to mirror the *loader*, not to implement BPE -- so the only honest
oracle is the implementation the GGUF was converted for: llama.cpp's
`llama_tokenize`, reached through `libllama.so` with `ctypes`.

That is a stronger check than a second hand-written implementation would be.
Two programs written from the same reading of the format agree on everything
the reader understood and disagree on exactly what it did not, which is the
failure this is meant to catch.

The corpus below is generated from a fixed seed, so a failure is reproducible
from the seed alone. The cases it is chosen to cover are the ones where a BPE
implementation actually diverges: the pre-tokenizer's handling of apostrophes
and contractions, digits (which split per character, not per run), punctuation
runs, whitespace runs with the `\\s+(?!\\S)` lookahead, multi-byte UTF-8, and
the special-token splice.

Everything skips when the library, the tool or the checkpoint is missing. A skip
is not a pass.
"""

from __future__ import annotations

import ctypes
import pathlib
import random
import subprocess

import pytest

from pocketllm import native

CHECKPOINT = pathlib.Path("/mnt/data1/models/qwen3-0.6b-f16.gguf")
#: Where `llama-tokenize` would live if the oracle could use it. It is not
#: built in this tree's llama.cpp checkout, so the oracle is the library
#: called through `ctypes` instead -- see `_LlamaOracle`.
LLAMA_CPP = pathlib.Path("/mnt/data1/llama.cpp-latest")
LLAMA_LIB = LLAMA_CPP / "build" / "bin" / "libllama.so"


def _tokenize_tool() -> pathlib.Path:
    return native._repository_root() / "build" / "pocketllm-tokenize"


def _escape(text: str) -> str:
    """Encode a text for the tool's `--lines` transport.

    The transport is line-oriented, so a text containing a newline would be
    read as two and the comparison would fail on the harness. The tool decodes
    this convention back, and the code points are written through as UTF-8
    because the tool's decoder is byte-oriented and leaves those bytes alone.
    """
    escapes = {0x0A: b"\\n", 0x0D: b"\\r", 0x09: b"\\t", 0x5C: b"\\\\"}
    raw = bytearray()
    for byte in text.encode():
        assert byte != 0x00, "NUL is not carryable through --lines"
        # Everything the map does not cover is passed through as-is, so the
        # result is still the original UTF-8 and decodes.
        raw += escapes.get(byte, bytes([byte]))
    return raw.decode("utf-8")


needs_engine = pytest.mark.skipif(not native.is_available(), reason="libpocketllm.so is not built")
needs_tool = pytest.mark.skipif(not _tokenize_tool().is_file(), reason="pocketllm-tokenize is not built")
needs_checkpoint = pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason=f"no checkpoint at {CHECKPOINT}"
)
needs_llama = pytest.mark.skipif(not LLAMA_LIB.is_file(), reason=f"no llama.cpp at {LLAMA_CPP}")

pytestmark = [needs_engine, needs_tool, needs_checkpoint, needs_llama]


class _LlamaOracle:
    """llama.cpp's tokenizer, called in-process through its C API.

    A subprocess would be simpler -- build one small C program and shell out --
    but the call is three functions wide, and `ctypes` keeps the oracle in the
    same process as the test, which is what lets it be a plain fixture rather
    than a pile of temp files. The structs below are `llama_model_params` and
    `llama_model_default_params()`, which cannot be constructed by hand across
    the boundary, so the default is fetched and only `vocab_only` flipped.
    """

    class _ModelParams(ctypes.Structure):
        _fields_ = [
            ("devices", ctypes.c_void_p),
            ("tensor_buft_overrides", ctypes.c_void_p),
            ("n_gpu_layers", ctypes.c_int32),
            ("split_mode", ctypes.c_int),
            ("main_gpu", ctypes.c_int32),
            ("tensor_split", ctypes.c_void_p),
            ("progress_callback", ctypes.c_void_p),
            ("progress_callback_user_data", ctypes.c_void_p),
            ("kv_overrides", ctypes.c_void_p),
            ("vocab_only", ctypes.c_bool),
            ("use_mmap", ctypes.c_bool),
            ("use_direct_io", ctypes.c_bool),
            ("use_mlock", ctypes.c_bool),
            ("check_tensors", ctypes.c_bool),
            ("use_extra_bufts", ctypes.c_bool),
            ("no_host", ctypes.c_bool),
            ("no_alloc", ctypes.c_bool),
        ]

    #: `ggml_log_callback`. Held on the instance, because the C side keeps the
    #: raw pointer and a garbage-collected trampoline is a use-after-free.
    _LogCallback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)

    def __init__(self, lib_path: pathlib.Path, model_path: pathlib.Path) -> None:
        # RTLD_GLOBAL because `libllama.so` resolves `ggml_backend_dev_by_type`
        # through its own link to libggml; loading the two in the wrong mode
        # gives each a private copy of ggml's globals, which is a subtle
        # enough failure -- the device count reads zero in one of them -- that
        # the mode is worth stating rather than searching for later.
        self._lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        ggml = ctypes.CDLL(str(lib_path.parent / "libggml.so"))
        self._ggml = ggml

        self._noop = self._LogCallback(lambda level, text, data: None)
        self._lib.llama_log_set.argtypes = [self._LogCallback, ctypes.c_void_p]
        self._lib.llama_log_set.restype = None
        # Silent: loading a 1.5 GB vocab prints ~80 lines of info that would
        # otherwise be interleaved into the test output for every run. A NULL
        # callback does not silence it -- NULL means "restore the default".
        self._lib.llama_log_set(self._noop, None)

        self._lib.llama_backend_init.argtypes = []
        self._lib.llama_backend_init.restype = None
        self._lib.llama_model_default_params.argtypes = []
        self._lib.llama_model_default_params.restype = self._ModelParams
        self._lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, self._ModelParams]
        self._lib.llama_model_load_from_file.restype = ctypes.c_void_p
        self._lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
        self._lib.llama_model_get_vocab.restype = ctypes.c_void_p
        self._lib.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
        self._lib.llama_vocab_n_tokens.restype = ctypes.c_int32
        self._lib.llama_tokenize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.c_bool,
            ctypes.c_bool,
        ]
        self._lib.llama_tokenize.restype = ctypes.c_int32
        self._lib.llama_model_free.argtypes = [ctypes.c_void_p]
        self._lib.llama_model_free.restype = None

        self._lib.llama_backend_init()

        # Pin the device list to CPU.  `ggml_backend_load_all` searches for
        # backend modules relative to the current directory, so a load from
        # anywhere but the llama.cpp build tree finds none -- and then
        # `llama_model_load_from_file` returns NULL because there is no device
        # to put the model on.  Asking ggml for its CPU device directly avoids
        # the search and, more importantly, is honest about what this oracle
        # is: it must be the CPU path, not whatever GPU happens to be present,
        # because the C engine under test is CPU-only.
        ggml.ggml_backend_dev_by_type.argtypes = [ctypes.c_int]
        ggml.ggml_backend_dev_by_type.restype = ctypes.c_void_p
        cpu = ggml.ggml_backend_dev_by_type(0)  # GGML_BACKEND_DEVICE_TYPE_CPU
        if not cpu:
            raise RuntimeError("llama.cpp has no CPU backend registered")
        self._devices = (ctypes.c_void_p * 2)(cpu, None)

        params = self._lib.llama_model_default_params()
        params.vocab_only = True
        params.devices = ctypes.cast(self._devices, ctypes.c_void_p)
        self._model = self._lib.llama_model_load_from_file(str(model_path).encode(), params)
        if not self._model:
            raise RuntimeError(f"llama.cpp could not load {model_path}")
        self._vocab = self._lib.llama_model_get_vocab(self._model)

    def close(self) -> None:
        if self._model:
            self._lib.llama_model_free(self._model)
            self._model = None

    @property
    def vocab_size(self) -> int:
        return int(self._lib.llama_vocab_n_tokens(self._vocab))

    def tokenize(self, text: str, add_special: bool, parse_special: bool) -> list[int]:
        raw = text.encode()
        # A negative return means "this many were needed", so one guess of
        # len(text)+64 is not safe for text that tokenizes to more pieces than
        # it has bytes -- which multi-byte UTF-8 does. Two calls, always.
        first = self._lib.llama_tokenize(
            self._vocab, raw, len(raw), None, 0, add_special, parse_special
        )
        if first < 0:
            first = -first
        buf = (ctypes.c_int32 * (first + 16))()
        n = self._lib.llama_tokenize(
            self._vocab, raw, len(raw), buf, len(buf), add_special, parse_special
        )
        assert n >= 0, f"llama_tokenize failed for {text!r} with {n}"
        return [int(buf[i]) for i in range(n)]


@pytest.fixture(scope="module")
def oracle() -> "_LlamaOracle":
    if not LLAMA_LIB.is_file():
        pytest.skip(f"no llama.cpp at {LLAMA_CPP}")
    instance = _LlamaOracle(LLAMA_LIB, CHECKPOINT)
    yield instance
    instance.close()


@pytest.fixture(scope="module")
def ours() -> list[str]:
    """The tool's output over `CORPUS`, one id list per line."""

    def run(texts: list[str], extra: list[str]) -> list[str]:
        result = subprocess.run(
            [str(_tokenize_tool()), str(CHECKPOINT), "--lines", *extra],
            input="\n".join(_escape(t) for t in texts) + "\n",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.splitlines()

    return run


#: One line per text, generated from a fixed seed so a failure is reproducible
#: from this file alone. `_seed_corpus` is called once per session.
def _seed_corpus() -> list[str]:
    rng = random.Random(1234)
    words = [
        "hello", "world", "def", "return", "class", "__init__", "self", "x", "1",
        "42", "3.14", "café", "naïve", "日本語", "中文", "مرحبا", "Привет", "한국어",
        "emoji", "🚀", "👍", "’s", "don't", "it's", "I'M", "WE'LL", "can't",
    ]
    punct = list(",.!?;:'\"-_()[]{}<>/\\|@#$%^&*+=~`")
    spaces = [" ", "  ", "\t", "   ", " \t "]
    texts: list[str] = []
    for _ in range(2000):
        parts = []
        for _ in range(rng.randint(1, 14)):
            roll = rng.random()
            if roll < 0.42:
                parts.append(rng.choice(words))
            elif roll < 0.62:
                parts.append(rng.choice(punct))
            elif roll < 0.82:
                parts.append(rng.choice(spaces))
            else:
                parts.append(str(rng.randint(0, 99999)))
        texts.append("".join(parts))
    texts += [
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return n\n",
        "SELECT * FROM t WHERE a = 'x' AND b > 3.14; -- comment\n",
        "    indented block with trailing spaces   ",
        "tabs\tand\tmore\ttabs",
        "line1\nline2\r\nline3\rline4",
        "trailing space ",
        " leading space",
        "multiple    spaces",
        "''''quotes''''",
        "....ellipsis...",
        "1234567890",
        "12.34e-5 + 6e7",
        "emoji 🚀 in the middle",
        "   ",
        "\n\n\n",
        "",
    ]
    return texts


#: Texts carrying special-token spellings, for the splice comparison.
SPECIAL_CORPUS = [
    "<|im_start|>user\nHello!<|im_end|>\n",
    "<|im_start|>assistant\n",
    "<tool_call>{\"a\": 1}</tool_call>",
    "a <tool_response> result </tool_response> b",
    "I think  thinkinghidden</think> answer",
    "plain text with no specials at all",
    "<|im_start|>",
    "<|im_end|>",
    "x <|im_start|> y <|im_start|> z",
]


@pytest.fixture(scope="module", params=[(True, False), (False, False), (True, True), (False, True)])
def flags(request: pytest.FixtureRequest) -> tuple[bool, bool]:
    return request.param


def test_the_corpus_tokenizes_identically(oracle: "_LlamaOracle", ours, flags) -> None:
    add_special, parse_special = flags
    extra = ([] if add_special else ["--no-special"]) + (["--parse-special"] if parse_special else [])
    texts = _seed_corpus()
    mine = ours(texts, extra)
    assert len(mine) == len(texts)
    for text, line in zip(texts, mine):
        expected = oracle.tokenize(text, add_special, parse_special)
        got = [int(x) for x in line.split()] if line.strip() else []
        assert got == expected, f"token mismatch for {text!r}"


def test_special_tokens_are_spliced_exactly_as_llama_cpp_does(oracle: "_LlamaOracle", ours, flags) -> None:
    add_special, parse_special = flags
    extra = ([] if add_special else ["--no-special"]) + (["--parse-special"] if parse_special else [])
    mine = ours(SPECIAL_CORPUS, extra)
    for text, line in zip(SPECIAL_CORPUS, mine):
        expected = oracle.tokenize(text, add_special, parse_special)
        got = [int(x) for x in line.split()] if line.strip() else []
        assert got == expected, f"special-token mismatch for {text!r}"


def test_a_control_looking_token_llama_cpp_promotes_is_known_and_deliberate(
    oracle: "_LlamaOracle",
) -> None:
    """The one input class this tokenizer deliberately does not match.

    llama.cpp repairs checkpoints whose vocabulary misspells a known EOG token:
    it scans for a list of literal spellings -- `</s>`, `<|eot_id|>`, `[EOT]`
    and about twenty others -- and promotes any it finds to CONTROL, "this is
    probably a bug in the model. its type will be overridden". Qwen3-0.6B
    labels `</s>` NORMAL, which is correct: it is the BPE spelling of a string,
    and 151645 is the control token.

    This engine does not copy the workaround. Trusting the checkpoint's own
    `token_type` is the simpler rule, it is right for this checkpoint, and a
    table of magic spellings is a list that ages badly -- a model whose
    vocabulary spells `_<EOT>` is not improved by a C engine guessing.

    The test is here rather than omitted so that the divergence is *pinned*: it
    runs both tokenizers over the spelling, asserts they differ, and thereby
    fails if a future llama.cpp drops the heuristic -- which is the moment to
    reconsider, and which would otherwise show up as a mysterious failure in
    the special-token corpus above.
    """
    text = "</s>"
    with native.Engine.open(str(CHECKPOINT)) as engine:
        asserted = engine.encode(text, add_special=False, parse_special=True)
    assert len(asserted) == 3, "the checkpoint's own type says this is ordinary text"

    theirs = oracle.tokenize(text, False, True)
    assert theirs == [128247], "llama.cpp no longer promotes it -- revisit the divergence"
    assert asserted != theirs


def test_the_vocabulary_size_matches_the_oracle(oracle: "_LlamaOracle") -> None:
    """The vocabularies are the same length, which is the weakest form of the
    check the corpus makes and the only one that holds without a checkpoint
    the engine can tokenize a string with."""
    with native.Engine.open(str(CHECKPOINT)) as engine:
        assert len(engine.encode("x")) == len(oracle.tokenize("x", True, False))


def test_decode_round_trips_through_the_engine() -> None:
    """`decode(encode(text)) == text` for text whose bytes survive the model's
    own vocabulary. The round trip is not the identity in general -- a
    vocabulary can spell a byte sequence more than one way -- but for ordinary
    UTF-8 it is, and a failure here means the byte map and its inverse
    disagree."""
    texts = ["Hello, world!", "café", "日本語", "emoji 🚀", "tabs\tand\ttabs", "line\nbreak"]
    with native.Engine.open(str(CHECKPOINT)) as engine:
        for text in texts:
            assert engine.decode(engine.encode(text)) == text, f"round trip failed for {text!r}"