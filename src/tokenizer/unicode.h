/* The Unicode questions the Qwen2 pre-tokenizer regex asks.
 *
 * llama.cpp answers these from a generated table (see `unicode_data.h`) rather
 * than from ICU or a regex engine, and this is the same decision: the regex
 * only needs "is this a letter", "is this a number" and "is this whitespace",
 * and a table lookup for three predicates is smaller and faster than a regex
 * engine that would have to be able to evaluate arbitrary patterns.
 *
 * `cpt_flags` returns the raw flag word from the vendored table.  The two
 * predicates below are the ones the pre-tokenizer uses; the rest of the word
 * is unused here and is not given a name, so nothing reads a bit by accident.
 */

#ifndef POCKETLLM_TOKENIZER_UNICODE_H
#define POCKETLLM_TOKENIZER_UNICODE_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace pocketllm {

/* Flag bits from the vendored table, matching llama.cpp's `unicode_cpt_flags`. */
constexpr uint16_t kCptNumber = 0x0002;
constexpr uint16_t kCptLetter = 0x0004;
constexpr uint16_t kCptWhitespace = 0x0100;
constexpr uint16_t kCptLowercase = 0x0200;
constexpr uint16_t kCptUppercase = 0x0400;

/* The flags for a codepoint, or 0 outside the table's range.  A binary search
 * over the sorted ranges. */
uint16_t cpt_flags(uint32_t codepoint);

/* Whether a codepoint is Unicode whitespace, per the vendored set. */
bool cpt_is_whitespace(uint32_t codepoint);

/* Simple lowercase, or the codepoint unchanged.  Only the pre-tokenizer's
 * case-insensitive `'s|'t|'re|'ve|'m|'ll|'d` branch needs it. */
uint32_t cpt_tolower(uint32_t codepoint);

/* Decode `text` into codepoints.
 *
 * Bytes that are not valid UTF-8 are treated as their own codepoint rather than
 * rejected: the input to `encode` has already been through `bytes_to_unicode`,
 * so every byte is a valid codepoint and this only matters for text a caller
 * passed without mapping.  Continuing is right -- refusing would make the
 * tokenizer the thing that decides what bytes a caller may have.
 */
std::vector<uint32_t> cpts_from_utf8(const std::string &text);

/* The UTF-8 encoding of one codepoint. */
std::string cpt_to_utf8(uint32_t codepoint);

/* The GPT-2 byte-to-unicode map.
 *
 * The 256 bytes are mapped onto printable codepoints so that every byte has a
 * representation and none of them is a control character: the identity for the
 * printable ASCII range, and `0x100 + n` for the bytes that are not — the same
 * assignment GPT-2 uses, which is what makes the vocabulary in a GGUF match.
 *
 * `map()` is 256 entries from byte to codepoint; `unmap()` is the inverse, by
 * codepoint.  Both are built once, on first use.
 */
const uint32_t *byte_to_cpt();
const std::unordered_map<uint32_t, uint8_t> &cpt_to_byte();

}  // namespace pocketllm

#endif /* POCKETLLM_TOKENIZER_UNICODE_H */