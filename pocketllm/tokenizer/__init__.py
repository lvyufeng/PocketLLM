"""The tokenizer: a GGUF vocabulary, and BPE over it.

``templating``/``prompt``/``server`` are device-neutral and take *token ids* or
*text*; they never tokenize.  This package is the thing on the other side of that
line, and it is deliberately the only part of the tree that needs a vocabulary.

Why a new implementation rather than ``transformers``: this wheel installs on a
phone.  ``transformers`` drags in ``tokenizers`` (a Rust extension) and a model
registry, neither of which a GGUF-only inference engine needs, and the vocabulary
is already in the file being loaded.  A GGUF carries its own token list and its own
BPE merges, so the tokenizer is a decoder over data the loader already reads.

**This is a skeleton.**  The interface is here and :class:`WhitespaceTokenizer` is
a working degenerate case used by the serving tests; :class:`GgufTokenizer` raises
:class:`~pocketllm.api.BackendNotImplementedError` naming what is left.  The
boundary is stated now -- so ``protocol`` and ``server`` can be written against it
without importing a tokenizer that does not exist -- and the merge table is a
later PR.
"""

from __future__ import annotations

from typing import Sequence

from pocketllm.api import BackendNotImplementedError

__all__ = ["Tokenizer", "WhitespaceTokenizer", "GgufTokenizer", "from_gguf"]


class Tokenizer:
    """Text to ids and back.  The only interface the serving layer needs.

    ``encode`` returns a list of ids rather than one id per call so a caller can
    decide about special tokens once; ``decode`` takes a sequence so a streaming
    response can decode the whole answer so far and re-emit only its tail, which
    is the only way a byte-level BPE can guarantee the bytes it printed are the
    bytes the model chose.
    """

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError


class WhitespaceTokenizer(Tokenizer):
    """A hash-free stand-in: whitespace splits, ids are the word's index in a table.

    Not a real tokenizer and not trying to be -- it exists so the serving tests
    can round-trip text without a vocabulary file, and so a caller has something
    concrete to pass when it does not care what the ids mean.  ``GgufTokenizer``
    is the one that ships for real, over a checkpoint's own merge table.
    """

    def __init__(self, words: Sequence[str]) -> None:
        if len(set(words)) != len(words):
            raise ValueError("the vocabulary has a duplicate entry")
        self._words = tuple(words)
        self._index = {word: i for i, word in enumerate(self._words)}
        self._unknown = len(self._words)

    @property
    def vocab_size(self) -> int:
        return len(self._words) + 1

    def encode(self, text: str) -> list[int]:
        return [self._index.get(word, self._unknown) for word in text.split()]

    def decode(self, ids: Sequence[int]) -> str:
        parts = []
        for token in ids:
            if token == self._unknown:
                continue
            if 0 <= token < len(self._words):
                parts.append(self._words[token])
        return " ".join(parts)


class GgufTokenizer(Tokenizer):
    """A ``.gguf`` vocabulary with its BPE merge table.  Not implemented yet.

    The declaration is the point of the stub: the loader already reads
    ``tokenizer.ggml.tokens`` and ``tokenizer.ggml.merges`` out of the metadata,
    so the missing piece is the merge algorithm and the byte fallback, not a way
    to reach the data.  When it lands, the constructor takes the bundle and this
    docstring goes away.
    """

    def __init__(self, path: str) -> None:
        raise BackendNotImplementedError(
            "the GGUF tokenizer is not implemented yet; it needs the BPE merge "
            "algorithm over tokenizer.ggml.merges and a byte fallback for "
            "tokenizer.ggml.byte_fallback. Use WhitespaceTokenizer in tests, or "
            "pass pre-tokenized ids."
        )


def from_gguf(path: str) -> Tokenizer:
    """The tokenizer for a ``.gguf`` checkpoint."""
    return GgufTokenizer(path)