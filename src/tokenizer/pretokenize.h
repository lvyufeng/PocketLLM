/* The pre-tokenizer: split mapped text into the words BPE merges within.
 *
 * Two alternations are implemented, because the vocabularies in this project's
 * scope are built on different ones and they differ in exactly one place:
 *
 *   qwen2      (?i:'s|'t|'re|'ve|'m|'ll|'d)
 *              | [^\r\n\p{L}\p{N}]?\p{L}+
 *              | \p{N}
 *              |  ?[^\s\p{L}\p{N}]+[\r\n]*
 *              | \s*[\r\n]+
 *              | \s+(?!\S)
 *              | \s+
 *
 *   llama-bpe  the same, with `\p{N}{1,3}` in place of `\p{N}`.
 *
 * That one difference is not cosmetic: it decides where a number is cut into
 * words, so `1234` is four words under `qwen2` and two (`123`, `4`) under
 * `llama-bpe`, and the digit pieces of a vocabulary are different ids in each.
 * Running the wrong one produces a sequence that is still integers, still in
 * range, and wrong -- there is nothing to notice it by -- which is why the
 * variant is named by the file (`tokenizer.ggml.pre`) and an unknown name is
 * refused rather than defaulted.
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
enum class PreTokenizer {
  /* `\p{N}` -- one digit per word.  Qwen2's and Qwen3's rule. */
  kQwen2,
  /* `\p{N}{1,3}` -- a run of up to three digits.  Llama-3's rule, which is what
   * a file naming `llama-bpe` asks for (MiniCPM5 among them). */
  kLlamaBpe,
};

std::vector<std::size_t> pretokenize(PreTokenizer rules, const std::vector<uint32_t> &cpts,
                                     const std::vector<std::size_t> &offsets);

}  // namespace pocketllm

#endif /* POCKETLLM_TOKENIZER_PRETOKENIZE_H */
