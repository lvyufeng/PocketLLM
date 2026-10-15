#!/usr/bin/env python3
"""Generate ``src/quant/iq2_grid.h``: the IQ2 codebooks, in C.

Two of the formats this engine decodes are *codebook* formats. A block of
``iq2_xxs`` or ``iq2_xs`` holds indices into a fixed table of small integer
weights, so decoding it is a lookup and the table *is* the format. There is no
arithmetic to write the table from, which leaves the copy in this repository as
the only thing that can be wrong.

``python/pocketllm/quant/ggml_tables.py`` answers that at runtime by reading
ggml's own header as text and refusing a header whose sha256 is not the pinned
one. C cannot read text at runtime, so this script generates the same two tables
into a header -- from the same pinned source and through the same reader,
``table_literals`` -- and records that source's sha256 in the output.
``tests/loader/test_vendored_header.py`` fails if the checked-in file is not what
this script would write, so the C copy cannot drift from the header every other
decoder in the tree reads.

Usage::

    python scripts/gen_iq_tables.py            # write src/quant/iq2_grid.h
    python scripts/gen_iq_tables.py --check    # exit 1 if the checked-in file is stale

The *shape* of the generated header is this script's other decision, and it is
not just two arrays. A host array is invisible to device code (nvcc: "identifier
is undefined in device code") and a ``__device__`` array is invisible to the host
half of a ``__host__ __device__`` function, while ``src/quant/blocks.h``'s
decoders are compiled for both. So each table's entries are a macro, and the
macro is instantiated once per pass that can read it, with ``__CUDA_ARCH__`` --
defined only in the device pass -- selecting at the use site.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from pocketllm.quant import ggml_tables  # noqa: E402  (path set above)

DEFAULT_OUTPUT = REPO_ROOT / "src" / "quant" / "iq2_grid.h"

#: The tables IQ2 needs: (source name, entry count, macro, host array, device array).
TABLES = (
    ("iq2xxs_grid", 256, "POCKETLLM_IQ2XXS_GRID_VALUES", "kIq2xxsGrid", "kIq2xxsGridDevice"),
    ("iq2xs_grid", 512, "POCKETLLM_IQ2XS_GRID_VALUES", "kIq2xsGrid", "kIq2xsGridDevice"),
)

#: Entries per line of the macro body: ggml's own header breaks these tables four
#: to a line, and keeping that makes a diff against the source readable.
PER_LINE = 4

_PREAMBLE = """\
/* The IQ2 codebooks, generated from ggml's table header.  Do not edit by hand.
 *
 * Source: {source}
 * sha256: {sha256}
 *
 * Regenerate with:
 *
 *   python scripts/gen_iq_tables.py
 *
 * The values are the entries of the source's `GGML_TABLE_BEGIN` blocks for
 * {names}, in source order, four to a line.
 *
 * A codebook is not arithmetic: there is nothing to derive the entries from,
 * so the only question is whether
 * this copy is the source's -- and the answer is that the same reader the Python
 * decoders use (`table_literals` in `pocketllm.quant.ggml_tables`) wrote it,
 * over the same sha256-pinned header.
 *
 * **Why the entries are a macro and the arrays are two.**  A host array is not
 * visible from device code and a `__device__` array cannot be read from the host
 * half of a `__host__ __device__` function, while the decoders in `blocks.h` are
 * compiled for both.  Each pass therefore reads the copy it can and
 * `__CUDA_ARCH__` -- defined only in the device pass -- chooses between them.
 * The entries are stated once, in the `..._VALUES` macros, and instantiated once
 * per pass that can use them.
 */

#ifndef POCKETLLM_QUANT_IQ2_GRID_H
#define POCKETLLM_QUANT_IQ2_GRID_H

#include <cstdint>

namespace pocketllm {{
namespace quant {{
"""

_TABLE_BLOCK = """
/* One `{name}` entry is eight 2-bit magnitudes packed little-endian: weight `j`
 * is byte `j` of the entry, so a magnitude is `(entry >> (8 * j)) & 0xFF`.  The
 * sign does not live here; it comes from the block. */
#define {macro} \\
{body}

[[maybe_unused]] constexpr uint64_t {array}[{count}] = {{{macro}}};
#ifdef __CUDACC__
static const __device__ uint64_t {device}[{count}] = {{{macro}}};
#endif
"""

_EPILOGUE = """
}  // namespace quant
}  // namespace pocketllm

#endif /* POCKETLLM_QUANT_IQ2_GRID_H */
"""


def _entries(name: str, count: int) -> list[str]:
    """One table's literals, as the pinned header spells them."""
    values = ggml_tables.table_literals(name, str(count))
    if len(values) != count:
        raise SystemExit(f"{name}: expected {count} entries, header has {len(values)}")
    return values


def _macro_body(values: list[str]) -> str:
    """The entries as macro lines, each but the last continued with a backslash."""
    lines: list[str] = []
    for start in range(0, len(values), PER_LINE):
        chunk = values[start : start + PER_LINE]
        last = start + PER_LINE >= len(values)
        lines.append("  " + ", ".join(chunk) + ("" if last else ", \\"))
    return "\n".join(lines)


def _source_label() -> str:
    """The header's path, relative to the repository when it is inside it."""
    path = ggml_tables.header_path()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def generate() -> str:
    """The exact contents of ``src/quant/iq2_grid.h``."""
    digest = hashlib.sha256(ggml_tables.header_path().read_bytes()).hexdigest()
    parts = [
        _PREAMBLE.format(
            source=_source_label(),
            sha256=digest,
            names=" and ".join(f"`{name}`" for name, *_ in TABLES),
        )
    ]
    for name, count, macro, array, device in TABLES:
        parts.append(
            _TABLE_BLOCK.format(
                name=name,
                macro=macro,
                body=_macro_body(_entries(name, count)),
                array=array,
                device=device,
                count=count,
            )
        )
    parts.append(_EPILOGUE)
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the checked-in file is stale")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    generated = generate()
    output: Path = args.output
    if args.check:
        if output.is_file() and output.read_text(encoding="utf-8") == generated:
            print(f"{output} is current ({generated.count(chr(10))} lines)")
            return 0
        print(
            f"{output} is stale or missing: regenerate it with "
            f"`python {Path(__file__).resolve().relative_to(REPO_ROOT)}`",
            file=sys.stderr,
        )
        return 1

    output.write_text(generated, encoding="utf-8")
    print(f"wrote {output} ({generated.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
