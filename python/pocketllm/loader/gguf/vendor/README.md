# Vendored GGML tables

`ggml-common.h` is a verbatim copy of the table header from **llama.cpp**, taken from
`relic_core/csrc/llama_mmq/ggml-common.h` in the [relic-core](https://github.com/lvyufeng/relic-core)
checkout that first vendored it. The loader reads it as **text** and extracts three tables by name;
it is never compiled here, so the file is data, not a build input.

| | |
| --- | --- |
| Origin | llama.cpp (MIT), `ggml-common.h` |
| Copied from | `relic-core/csrc/llama_mmq/ggml-common.h` |
| sha256 | `d09a7116254352959002c50efd0f0a6008bb6109d342f3285c94ad461f877d9a` |
| Size | 136,181 bytes, 1,925 lines |
| Tables read | `iq2xs_grid` (512), `iq3xxs_grid` (256), `kvalues_iq4nl` (16) |

The upstream revision is **not recorded** — the relic-core copy does not carry a provenance note,
so the only revision identity available is the sha256 above. If it ever matters, that hash is what
to match against.

## Why this file is here

Before the rebuild, `python/pocketllm/loader/gguf/iq4_nl.py` resolved this header through
`Path(relic_core.__file__).parent / "csrc" / "llama_mmq" / "ggml-common.h"`, which made the loader
depend on the kernel library being installed. The rebuilt tree's core is **torch-free and
native-toolchain-free**: a phone or edge install must not need relic-core. Vendoring the header is
what removes that dependency.

`quant/ggml_tables.py` resolves the header in this order, and the vendored copy is the one a
normal install uses:

1. `$POCKETLLM_GGML_COMMON`, an explicit override;
2. this vendored file;
3. relic-core's copy, when it happens to be installed — last, and its absence is not an error.

The tables are read from the header rather than transcribed into Python so there is one statement of
each table's bytes. If relic-core's copy is ever found to disagree with this one, the sha256 above
is the thing that changed, and the fix is to re-vendor and update this note rather than to edit
either copy by hand.