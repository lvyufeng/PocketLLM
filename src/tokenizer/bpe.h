/* Byte-level BPE, over the vocabulary a GGUF carries.
 *
 * A GGUF checkpoint is self-describing about its tokenizer: it carries the
 * token list, the merge ranks and the pre-tokenizer name, so nothing here reads
 * a separate `tokenizer.json` and nothing here imports `transformers`.  That is
 * not an optimisation -- a wheel that installs on a phone cannot take a Rust
 * extension and a model registry to turn text into ids.
 *
 * The algorithm is GPT-2's byte-level BPE, which is what Qwen2/Qwen3 use:
 *
 *   1. Split the text into words with the model's pre-tokenizer regex.
 *   2. Map each byte of a word to a printable codepoint (`bytes_to_unicode`),
 *      so every byte has a piece and no input is unrepresentable.
 *   3. Within each word, repeatedly merge the adjacent pair with the lowest
 *      merge rank, until no adjacent pair has a rank.
 *   4. Turn each merged piece into a token id, falling back to per-byte pieces
 *      when a whole-piece token does not exist.
 *
 * Steps 1 and 2 are in that order and not the other: the regex asks about
 * `\p{L}` and `\p{N}`, which are questions about the text the caller wrote,
 * not about the byte alphabet.
 *
 * Control and user-defined tokens are spliced out of the text before any of
 * this, so a prompt carrying `<|im_start|>` reaches the model as that one
 * token rather than as the characters it is spelled with.
 *
 * Steps 2 and 4 are where an implementation diverges from llama.cpp, because
 * both depend on which pre-tokenizer the vocabulary names.  The pre-tokenizer
 * here is the Qwen2 one, matching `tokenizer.ggml.pre == "qwen2"`, and a
 * checkpoint naming anything else is refused by name rather than tokenized
 * wrongly.
 *
 * `python/pocketllm/tokenizer/` is a skeleton with no BPE, so there is no
 * Python oracle for this.  The oracle is llama.cpp's own `llama_tokenize`,
 * reached from the test through `libllama.so`; that is a stronger check than a
 * comparison against a second implementation written from the same reading.
 */

#ifndef POCKETLLM_TOKENIZER_BPE_H
#define POCKETLLM_TOKENIZER_BPE_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace pocketllm {

class GgufReader;

/* Special tokens the engine needs by id, not by name.  A checkpoint that does
 * not declare one leaves it at -1, and the callers that need it say so. */
struct SpecialTokens {
  int32_t bos = -1;
  int32_t eos = -1;
  int32_t pad = -1;
  bool add_bos = false;  /* `tokenizer.ggml.add_bos_token` */
  bool add_eos = false;
};

/* A token that can be spliced out of the text, and whether doing so requires
 * `parse_special`.  CONTROL and UNKNOWN do; USER_DEFINED does not, which is
 * llama.cpp's rule and the reason `<tool_call>` is one token in a plain prompt
 * while `<|im_start|>` is not. */
struct SpecialToken {
  int32_t id = 0;
  bool needs_parse = false;
};

class Tokenizer {
 public:
  /* Build a tokenizer from a checkpoint's metadata.
   *
   * Throws `Error` if the vocabulary is absent, or if it names a pre-tokenizer
   * this build does not implement -- a wrong split is a wrong token sequence,
   * and a wrong token sequence is a wrong answer with no other symptom. */
  explicit Tokenizer(const GgufReader &checkpoint);
  ~Tokenizer();

  Tokenizer(const Tokenizer &) = delete;
  Tokenizer &operator=(const Tokenizer &) = delete;

  /* Text to ids.
   *
   * `add_special` asks for the model's BOS/EOS convention: a BOS is prepended
   * only when the checkpoint's `add_bos_token` says to, which is how a prompt
   * that already carries its own control tokens avoids a doubled one.
   *
   * `parse_special` decides whether a control token spelled inside `text`
   * becomes that token or the characters it is spelled with.  It is the
   * difference between running a prompt that *is* a chat template -- where
   * `<|im_start|>` has to be one id or the model sees the wrong prompt -- and
   * running text that merely mentions one.  User-defined tokens are spliced
   * either way; llama.cpp draws the line in the same place, and the two flags
   * are what its `llama_tokenize` takes. */
  std::vector<int32_t> encode(const std::string &text, bool add_special, bool parse_special) const;

  /* Ids to text.  A byte piece becomes its byte -- which may be a partial UTF-8
   * sequence, so the result is bytes rather than a validated string.  That is
   * deliberate: a streaming consumer decodes the whole answer so far and
   * re-emits the tail, and validating here would fail on the first fragment of
   * a three-byte character. */
  std::string decode(const std::vector<int32_t> &ids) const;

  int vocab_size() const { return static_cast<int>(vocab_.size()); }
  const SpecialTokens &special() const { return special_; }

 private:
  /* BPE over one word -- already in the byte alphabet -- appending ids. */
  void merge_word(const std::string &word, std::vector<int32_t> &out) const;

  std::vector<std::string> vocab_;
  std::unordered_map<std::string, int32_t> token_to_id_;
  /* Keyed by "left\x01right": a merge's two halves can contain a space, so a
   * space is not a separator this could be looked up with. */
  std::unordered_map<std::string, int> merge_rank_;
  /* Sorted longest spelling first, so a longer token wins over any token it
   * starts with. */
  std::vector<SpecialToken> special_tokens_;
  SpecialTokens special_;
  std::string pre_type_;
};

}  // namespace pocketllm

#endif /* POCKETLLM_TOKENIZER_BPE_H */