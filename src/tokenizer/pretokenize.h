/* The pre-tokenizer: split mapped text into the words BPE merges within.
 *
 * The regex, in its own notation, is Qwen2's:
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)
 *   | [^\r\n\p{L}\p{N}]?\p{L}+
 *   | \p{N}
 *   |  ?[^\s\p{L}\p{N}]+[\r\n]*
 *   | \s*[\r\n]+
 *   | \s+(?!\S)
 *   | \s+
 *
 * This is a hand-written matcher for that one alternation rather than a regex
 * engine, for the same reason llama.cpp does it that way: the pattern is fixed,
 * the engine would be a dependency, and the alternatives are mutually exclusive
 * enough that a scanner is both clearer and faster.
 *
 * The function returns the length of each word in *codepoints*, not the words
 * themselves: the caller already holds the codepoint vector and only needs the
 * boundaries, and codepoint lengths are what llama.cpp's `bpe_offsets` carries,
 * so a self-test can compare the two splits directly.
 */

#ifndef POCKETLLM_TOKENIZER_PRETOKENIZE_H
#define POCKETLLM_TOKENIZER_PRETOKENIZE_H

#include <cstdint>
#include <string>
#include <vector>

namespace pocketllm {

/* The Qwen2 pre-tokenizer, over the codepoints of `text`.
 *
 * `text` must be the *mapped* text -- every byte already replaced by its
 * printable codepoint -- because the regex's `\p{L}` and `\p{N}` are asking
 * about the mapped characters.  `offsets` is the split the caller has already
 * made (a single element covering the whole text, for a one-shot encode); the
 * regex is applied within each element.
 *
 * Returns the codepoint length of each word, in order.  The caller turns those
 * lengths back into byte ranges.
 */
std::vector<std::size_t> qwen2_pretokenize(const std::vector<uint32_t> &cpts,
                                           const std::vector<std::size_t> &offsets);

}  // namespace pocketllm

#endif /* POCKETLLM_TOKENIZER_PRETOKENIZE_H */