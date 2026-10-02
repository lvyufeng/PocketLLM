/* Unicode character data, vendored from llama.cpp.
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
 * sha256:    95170cd1c105a5b41a1b2dce73b0fae8ce8011ef7897600828bb2babe8b26e5d
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

namespace pocketllm {

/* Sorted by range start; the flags apply from a start up to the next start
 * minus one.  A binary search over this is how the flags lookup answers. */
extern const std::initializer_list<std::pair<uint32_t, uint16_t>> kUnicodeRangesFlags;

/* Unicode whitespace, the set the regex's `\s` means. */
extern const std::unordered_set<uint32_t> kUnicodeSetWhitespace;

/* Simple lowercase mapping, used only by the case-insensitive `'s|'re|...`
 * branch of the pre-tokenizer.  Sorted by codepoint. */
extern const std::initializer_list<std::pair<uint32_t, uint32_t>> kUnicodeMapLowercase;

}  // namespace pocketllm

#endif /* POCKETLLM_TOKENIZER_UNICODE_DATA_H */
