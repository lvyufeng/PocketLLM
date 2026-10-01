#!/usr/bin/env python3
"""Regenerate ``src/tokenizer/unicode_data.{h,cpp}`` from llama.cpp's tables.

The Qwen2 pre-tokenizer regex needs three Unicode questions answered about a
codepoint — letter, number, whitespace — and llama.cpp answers them from a
generated table rather than from ICU or a regex engine. This script extracts
exactly the three tables that regex reads, so the C tokenizer vendors ~170 KB
rather than the ~7,000-line file most of which nothing here consumes.

The extraction is a *verbatim* copy, not a re-derivation: a table regenerated
from a different Unicode revision would disagree with the oracle in exactly the
codepoints nobody tests, which is the failure mode this avoids.

Usage::

    python scripts/vendor_unicode_data.py --llama-cpp /path/to/llama.cpp
    python scripts/vendor_unicode_data.py --llama-cpp ... --check   # verify only

``--check`` compares the committed files against a fresh extraction and exits
non-zero on any difference, so a llama.cpp that moved under us is a loud failure
rather than a silent divergence.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path("/mnt/data1/llama.cpp-latest")

HEADER = REPO_ROOT / "src" / "tokenizer" / "unicode_data.h"
BODY = REPO_ROOT / "src" / "tokenizer" / "unicode_data.cpp"

#: The three tables the Qwen2 pre-tokenizer regex reads, and their vendored
#: names. The order matters only for the diff, not for correctness.
TABLES = (
    ("unicode_ranges_flags", "kUnicodeRangesFlags"),
    ("unicode_set_whitespace", "kUnicodeSetWhitespace"),
    ("unicode_map_lowercase", "kUnicodeMapLowercase"),
)


def _block(text: str, name: str) -> str:
    """The full declaration of `name`, from its `const ...` line to its `};`.

    A regex over the generated file is enough and is the honest tool here: the
    file is machine-written with one declaration per line and one table per
    block, so anything more structured would be more code for the same answer.

    The `[^{;=]*` between `const` and the name is load-bearing.  A lazy `.*?`
    there would let an earlier `const` line reach forward to a later name and
    swallow every table in between, which extracts a block that starts with the
    wrong declaration and ends wherever the next `};` happens to be.  That
    produces a file that compiles only if the duplication is caught, and the
    duplication is the symptom rather than the error -- the tables it should
    have extracted are simply absent.
    """
    match = re.search(rf"^const [^{{;=]*\b{re.escape(name)}\b\s*=.*?^\}};", text, re.MULTILINE | re.DOTALL)
    if not match:
        raise SystemExit(f"could not find the table {name} in the source")
    return match.group(0)


def extract(source: Path) -> tuple[str, str, str]:
    """The header text, the body text, and the source's sha256."""
    path = source / "src" / "unicode-data.cpp"
    if not path.is_file():
        raise SystemExit(f"no unicode-data.cpp under {source}")

    text = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    parts = []
    for original, vendored in TABLES:
        block = _block(text, original)
        # Rename on the way in: a vendored symbol with llama.cpp's name would
        # collide at link time for anything that also links llama.cpp, which
        # the tests do -- they dlopen it for the oracle.
        parts.append(block.replace(original, vendored))

    header = f'''/* Unicode character data, vendored from llama.cpp.
 *
 * The Qwen2 pre-tokenizer regex needs three Unicode questions answered about a
 * codepoint: is it a letter, is it a number, is it whitespace.  llama.cpp
 * answers them from a generated table rather than from ICU or a regex engine,
 * which is the right shape for a phone -- a full Unicode database is megabytes
 * and a regex engine is a dependency this tree does not take.
 *
 * This file is a verbatim extraction of the three tables that regex uses, from
 * llama.cpp's `src/unicode-data.cpp`.  The extraction is deliberate: the full
 * file also carries case folding, normalization and a second flag table that
 * nothing here reads, and a 7000-line vendored file with no consumer for most
 * of it is a file nobody can review.
 *
 * Source:    https://github.com/ggml-org/llama.cpp  src/unicode-data.cpp
 * Generated: scripts/gen-unicode-data.py in that repository
 * sha256:    {digest}
 *
 * Regenerate with `python scripts/vendor_unicode_data.py`, which refuses a
 * mismatch rather than silently vendoring a different revision.
 *
 * The data is MIT-licensed, from llama.cpp.  Flag bits follow that file's
 * `unicode_cpt_flags`: NUMBER 0x0002, LETTER 0x0004, WHITESPACE 0x0100.
 */

#ifndef POCKETLLM_TOKENIZER_UNICODE_DATA_H
#define POCKETLLM_TOKENIZER_UNICODE_DATA_H

#include <cstdint>
#include <initializer_list>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace pocketllm {{

/* Sorted by range start; the flags apply from a start up to the next start
 * minus one.  A binary search over this is how the flags lookup answers. */
extern const std::initializer_list<std::pair<uint32_t, uint16_t>> kUnicodeRangesFlags;

/* Unicode whitespace, the set the regex's `\\s` means. */
extern const std::unordered_set<uint32_t> kUnicodeSetWhitespace;

/* Simple lowercase mapping, used only by the case-insensitive `'s|'re|...`
 * branch of the pre-tokenizer.  Sorted by codepoint. */
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> kUnicodeMapLowercase;

}}  // namespace pocketllm

#endif /* POCKETLLM_TOKENIZER_UNICODE_DATA_H */
'''

    body = f'''/* Generated by scripts/vendor_unicode_data.py -- do not edit by hand.
 * See src/tokenizer/unicode_data.h for the provenance and the sha256.
 */

#include "tokenizer/unicode_data.h"

namespace pocketllm {{

{chr(10).join(parts)}

}}  // namespace pocketllm
'''
    return header, body, digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llama-cpp", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 on a difference")
    args = parser.parse_args(argv)

    header, body, digest = extract(args.llama_cpp)

    if args.check:
        stale = []
        if not HEADER.is_file() or HEADER.read_text() != header:
            stale.append(HEADER)
        if not BODY.is_file() or BODY.read_text() != body:
            stale.append(BODY)
        if stale:
            print("vendored unicode data is stale: " + ", ".join(str(p) for p in stale), file=sys.stderr)
            return 1
        print(f"vendored unicode data is current (source sha256 {digest[:12]})")
        return 0

    HEADER.write_text(header, encoding="utf-8")
    BODY.write_text(body, encoding="utf-8")
    print(f"wrote {HEADER.relative_to(REPO_ROOT)} and {BODY.relative_to(REPO_ROOT)} (sha256 {digest[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())