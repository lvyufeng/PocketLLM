# Vendored GGML tables

`ggml-common.h` is a verbatim copy of the table header from **llama.cpp**, taken from upstream's
`ggml-common.h`. The loader reads it as **text** and extracts three tables by name; it is never
compiled here, so the file is data, not a build input.

| | |
| --- | --- |
| Origin | llama.cpp (MIT), `ggml-common.h` |
| sha256 | `d09a7116254352959002c50efd0f0a6008bb6109d342f3285c94ad461f877d9a` |
| Size | 136,181 bytes, 1,925 lines |
| Tables read | `iq2xs_grid` (512), `iq3xxs_grid` (256), `kvalues_iq4nl` (16) |

The upstream revision is **not recorded** — the copy does not carry a provenance note, so the only
revision identity available is the sha256 above. If it ever matters, that hash is what to match
against.

## Why this file is here

Reading a checkpoint must not require a kernel library to be installed. The tree's core is
**torch-free and native-toolchain-free**: a phone or edge install reads its codebooks from this
vendored header and nothing else. Vendoring the header is what makes that true.

`quant/ggml_tables.py` resolves the header in this order, and the vendored copy is the one a
normal install uses:

1. `$POCKETLLM_GGML_COMMON`, an explicit override;
2. this vendored file.

There is deliberately no further fallback: a lookup that silently succeeds on a developer's machine
and fails on a phone is worse than one that fails loudly on both. `$POCKETLLM_GGML_COMMON` is how a
non-default revision is named.

The tables are read from the header rather than transcribed into Python so there is one statement of
each table's bytes. To move to a different upstream revision, re-vendor the file and update the
sha256 here and in `ggml_tables.VENDORED_SHA256` in the same change — do not edit either copy by
hand.