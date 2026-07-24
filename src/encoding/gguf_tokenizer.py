from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Iterable

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers

from src.loader.gguf.bundle import resolve_gguf_bundle

# GGUF ``tokenizer.ggml.pre`` values whose reference (llama.cpp) pre-tokenizer
# is the llama3/CHATGLM4 split regex, NOT plain byte-level. GLM-4/GLM-5.2 use
# ``glm4``; llama3 shares the identical adapted pattern. Any pre value NOT in
# this set keeps the historical plain-ByteLevel behavior (e.g. MiniMax-M2),
# so this table is additive and cannot change existing models' tokenization.
_LLAMA3_STYLE_PRE = frozenset(
    {"glm4", "chatglm-bpe", "llama3", "llama-bpe", "llama-v3", "llama3-bpe"}
)

# The llama.cpp CHATGLM4 / LLAMA3 pre-tokenizer split pattern (two source
# segments concatenated). Matches contractions (case-insensitive), words,
# 1-3 digit number groups, punctuation runs, and whitespace/newline runs.
_LLAMA3_STYLE_SPLIT_REGEX = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _pre_tokenizer_for(pre: str | None):
    """Pick the pre-tokenizer for a GGUF ``tokenizer.ggml.pre`` value.

    llama3/glm4-family models need a regex ``Split`` ahead of a
    ``use_regex=False`` ByteLevel so token boundaries (digits, whitespace,
    contractions) match the reference tokenization. Every other value keeps
    the plain ``ByteLevel(add_prefix_space=False)`` used historically, so
    already-supported models (e.g. MiniMax-M2) are unaffected.
    """

    if pre and str(pre) in _LLAMA3_STYLE_PRE:
        return pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(_LLAMA3_STYLE_SPLIT_REGEX), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
    return pre_tokenizers.ByteLevel(add_prefix_space=False)

_METADATA_TYPES = {
    0: ("uint8", "<B", 1),
    1: ("int8", "<b", 1),
    2: ("uint16", "<H", 2),
    3: ("int16", "<h", 2),
    4: ("uint32", "<I", 4),
    5: ("int32", "<i", 4),
    6: ("float32", "<f", 4),
    7: ("bool", "<?", 1),
    8: ("string", None, None),
    9: ("array", None, None),
    10: ("uint64", "<Q", 8),
    11: ("int64", "<q", 8),
    12: ("float64", "<d", 8),
}

TOKENIZER_ARRAY_KEYS = frozenset(
    {
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.token_type",
    }
)

TOKENIZER_SCALAR_KEYS = frozenset(
    {
        "general.architecture",
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_sep_token",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.padding_token_id",
    }
)


class GGUFTokenizerMetadataError(RuntimeError):
    pass


def primary_gguf_path(path: str | Path) -> Path:
    """Return the first/primary GGUF shard for a file, shard, or directory."""

    paths = resolve_gguf_bundle(path)
    if not paths:
        raise FileNotFoundError(f"no GGUF files resolved from {path}")
    return Path(paths[0])


def _metadata_type(type_id: int) -> tuple[str, str | None, int | None]:
    try:
        return _METADATA_TYPES[type_id]
    except KeyError as exc:
        raise GGUFTokenizerMetadataError(f"unsupported GGUF metadata type {type_id}") from exc


def _read_struct(f, fmt: str):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise EOFError("unexpected EOF")
    values = struct.unpack(fmt, data)
    return values[0] if len(values) == 1 else values


def _read_string(f) -> str:
    length = _read_struct(f, "<Q")
    data = f.read(length)
    if len(data) != length:
        raise EOFError("unexpected EOF in GGUF string")
    return data.decode("utf-8", errors="replace")


def _read_or_skip_value(f, value_type: int, *, keep: bool) -> Any:
    type_name, fmt, size = _metadata_type(value_type)
    if type_name == "string":
        value = _read_string(f)
        return value if keep else None
    if type_name == "array":
        item_type = _read_struct(f, "<I")
        length = _read_struct(f, "<Q")
        item_name, item_fmt, item_size = _metadata_type(item_type)
        if keep:
            if item_name == "string":
                return [_read_string(f) for _ in range(length)]
            if item_fmt is not None:
                return [_read_struct(f, item_fmt) for _ in range(length)]
            raise GGUFTokenizerMetadataError(f"unsupported GGUF array item type {item_type}")
        if item_name == "string":
            for _ in range(length):
                f.seek(_read_struct(f, "<Q"), 1)
        else:
            if item_size is None:
                raise GGUFTokenizerMetadataError(f"unsupported GGUF array item type {item_type}")
            f.seek(length * item_size, 1)
        return None
    if fmt is None or size is None:
        raise GGUFTokenizerMetadataError(f"unsupported GGUF metadata type {value_type}")
    value = _read_struct(f, fmt)
    return value if keep else None


def read_gguf_metadata(
    path: str | Path,
    *,
    keep_array_keys: Iterable[str] = (),
    keep_all_scalars: bool = True,
) -> dict[str, Any]:
    """Read selected GGUF metadata without loading tensor payloads.

    Scalar/string metadata is cheap and kept by default. Large arrays, such as
    tokenizer vocab/merges, are only materialized when named in ``keep_array_keys``.
    """

    keep_arrays = set(keep_array_keys)
    metadata: dict[str, Any] = {}
    gguf_path = primary_gguf_path(path)
    with open(gguf_path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise GGUFTokenizerMetadataError(f"not a GGUF file: {gguf_path}")
        _version = _read_struct(f, "<I")
        _tensor_count = _read_struct(f, "<Q")
        metadata_count = _read_struct(f, "<Q")
        for _ in range(metadata_count):
            key = _read_string(f)
            value_type = _read_struct(f, "<I")
            type_name = _metadata_type(value_type)[0]
            keep = key in keep_arrays if type_name == "array" else keep_all_scalars
            value = _read_or_skip_value(f, value_type, keep=keep)
            if keep:
                metadata[key] = value
    return metadata


def read_gguf_tokenizer_metadata(path: str | Path) -> dict[str, Any]:
    metadata = read_gguf_metadata(path, keep_array_keys=TOKENIZER_ARRAY_KEYS, keep_all_scalars=True)
    missing = sorted(TOKENIZER_ARRAY_KEYS - set(metadata))
    if missing:
        raise GGUFTokenizerMetadataError(f"GGUF tokenizer metadata missing keys: {missing}")
    return metadata


def metadata_int(metadata: dict[str, Any], key: str, default: int = 0) -> int:
    value = metadata.get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    return int(default)


def context_length_from_metadata(metadata: dict[str, Any], *, architecture: str | None = None) -> int:
    arch = architecture or str(metadata.get("general.architecture", ""))
    candidates = []
    if arch:
        candidates.append(f"{arch}.context_length")
    candidates.extend(["minimax-m2.context_length", "deepseek4.context_length", "llama.context_length"])
    for key in candidates:
        value = metadata_int(metadata, key, 0)
        if value > 0:
            return value
    return 0


def build_gguf_bpe_tokenizer(path: str | Path) -> tuple[Tokenizer, dict[str, Any]]:
    metadata = read_gguf_tokenizer_metadata(path)
    tokens = metadata["tokenizer.ggml.tokens"]
    merges = metadata["tokenizer.ggml.merges"]
    token_types = metadata["tokenizer.ggml.token_type"]
    if len(tokens) != len(token_types):
        raise GGUFTokenizerMetadataError("tokenizer token/type array length mismatch")

    vocab = {token: idx for idx, token in enumerate(tokens)}
    merge_pairs: list[tuple[str, str]] = []
    for merge in merges:
        parts = merge.split(" ")
        if len(parts) != 2:
            raise GGUFTokenizerMetadataError(f"bad BPE merge line: {merge!r}")
        merge_pairs.append((parts[0], parts[1]))

    unk_id = metadata_int(metadata, "tokenizer.ggml.unknown_token_id", 0)
    unk_token = tokens[unk_id]
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merge_pairs, unk_token=unk_token, fuse_unk=False))
    tokenizer.pre_tokenizer = _pre_tokenizer_for(metadata.get("tokenizer.ggml.pre"))
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = [token for token, token_type in zip(tokens, token_types) if int(token_type) != 1]
    tokenizer.add_special_tokens(
        [
            AddedToken(token, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)
            for token in special_tokens
        ]
    )
    return tokenizer, metadata


def parse_ids_csv(text: str) -> list[int]:
    return [int(item.strip()) for item in text.replace("\n", ",").split(",") if item.strip()]


def decode_ids(tokenizer: Tokenizer, ids: Iterable[int], *, skip_special: bool = False) -> str:
    return tokenizer.decode([int(token_id) for token_id in ids], skip_special_tokens=skip_special)
