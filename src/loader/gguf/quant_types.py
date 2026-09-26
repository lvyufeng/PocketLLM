from __future__ import annotations

# Raw-block runtime dispatch table: GGUF type name -> the compact type id the
# GGUF kernels switch on.  Deliberately not the file's own GGML id -- `iq2_xxs`
# is GGML type 16 and runtime type 0, and `iq4_nl` is GGML type 20 and runtime
# type 20 only because 20 was free -- because the kernels only ever see the
# formats in here.
GGUF_DENSE_TYPE_IDS = {
    "iq2_xxs": 0,
    "q2_k": 1,
    "iq1_m": 2,
    "q4_k": 3,
    "q5_k": 4,
    "iq2_xs": 5,
    "iq3_xxs": 6,
    "iq4_xs": 7,
    "q6_k": 8,
    "iq4_nl": 20,
}

GGUF_DENSE_TYPE_NAMES = {value: key for key, value in GGUF_DENSE_TYPE_IDS.items()}

# Fork-private ternary types from PrismML-Eng/llama.cpp's `prism` branch, as
# GGML *file* ids: 143 is PTQ1_0 (128 weights per 28 bytes) and 142 is PQ2_0
# (128 per 34).  The geometry is keyed the same way here as it is in
# `reader.GGML_TYPES`, which is what a file's own header is read through.
#
# They are deliberately **absent** from `GGUF_DENSE_TYPE_IDS` above.  That map
# is the raw-block runtime's dispatch table, so a name in it is a claim that a
# kernel exists to consume the blocks; there is none yet, and the failure it
# would produce is a silent F16 upcast -- ten times the memory and a wrong
# kernel that looks right.  The loader addresses the bytes and refuses to
# interpret them.  `src/loader/gguf/tensor_reader.py` carries the refusal and
# says so in its message.
GGUF_TERNARY_FILE_TYPE_IDS = {
    "ptq1_0": 143,
    "pq2_0": 142,
}

GGUF_TERNARY_TYPE_NAMES = frozenset(GGUF_TERNARY_FILE_TYPE_IDS)

# Formats the loader can both address and dequantize but which no raw-block
# kernel consumes, so they are absent from `GGUF_DENSE_TYPE_IDS` above for the
# same reason the ternary packs are: a name in that map is a claim that a kernel
# switches on its id.  The distinction from the ternary pair is the loader's
# behaviour, not the file's: `read_tensor` decodes these honestly, while it
# refuses a ternary tensor by name.
#
# `iq4_nl` was this category's only member when the set was introduced in #390,
# and Xing4.0-29B-A4B's released GGUF is 243 tensors of it.  #393 gave it a
# kernel -- `iq4nl_block_dot_256` in `src/csrc/cuda_kernel_impl.cu` -- so it
# graduated into `GGUF_DENSE_TYPE_IDS` above and the set is empty.  It is kept
# rather than deleted because the category recurs, and because empty-and-
# deliberate is a different statement from a name that was forgotten.
GGUF_LOADER_TYPE_NAMES: frozenset[str] = frozenset()

#: Weights one raw-block kernel call consumes for `iq4_nl`.  The runtime's other
#: ten formats have a 256-weight block with an internal scale table; IQ4_NL's
#: scale is per 32, so a 256-wide span is eight consecutive native 18-byte blocks
#: and the loader folds them into one 144-byte row element before the tensor
#: reaches a kernel.  See `src/loader/gguf/iq4_nl.fold_to_runtime_span`.
IQ4_NL_RUNTIME_SPAN = 256

#: Every GGUF type whose blocks the loader can address by offset, whether or not
#: anything downstream can consume them.
GGUF_ADDRESSABLE_TYPE_NAMES = (
    frozenset(GGUF_DENSE_TYPE_IDS) | GGUF_TERNARY_TYPE_NAMES | GGUF_LOADER_TYPE_NAMES
)
