#include "tokenizer/bpe.h"

#include <algorithm>
#include <cstddef>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gguf/reader.h"
#include "runtime/status.h"
#include "tokenizer/pretokenize.h"
#include "tokenizer/unicode.h"

namespace pocketllm {

namespace {

/* One piece of a word while it is being merged.  The pieces form a doubly
 * linked list over a vector, which is what makes a merge O(1) -- the two
 * neighbours are joined and the merged piece takes the left one's slot. */
struct Symbol {
  std::string text;
  int prev = -1;
  int next = -1;
};

/* A candidate merge.  Ordering is by rank, then by left position: that second
 * key is not decoration.  Two disjoint pairs can carry the same rank (a rank is
 * unique per pair, but different pairs are different entries and a vocabulary
 * can repeat one), and without a tiebreak the pop order would depend on the
 * container's internals.  llama.cpp orders the same way, and the tiebreak is
 * what makes the two agree. */
struct Bigram {
  int left = 0;
  int right = 0;
  std::string text;
  int rank = 0;

  struct Worse {
    bool operator()(const Bigram &a, const Bigram &b) const {
      return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
    }
  };
};

using BigramQueue = std::priority_queue<Bigram, std::vector<Bigram>, Bigram::Worse>;

/* A fragment of the input: either raw text to be BPE'd, or a special token that
 * was spliced out of it.  The splice happens first so that a control token in
 * the prompt is never re-tokenized as the characters it is spelled with. */
struct Fragment {
  bool is_token = false;
  std::string text;
  int32_t token = 0;
};

/* The multi-byte characters a word is built from, exactly as
 * `unicode_len_utf8` counts them: a continuation byte counts as one, so
 * malformed input still advances instead of looping. */
std::size_t utf8_char_len(const std::string &word, std::size_t pos) {
  /* The high nibble decides, which is exactly llama.cpp's `unicode_len_utf8`
   * table: 0xC0-0xDF is two bytes, 0xE0-0xEF three, 0xF0 and up four,
   * everything else -- ASCII and continuation bytes alike -- one. */
  static const std::size_t kLookup[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  return kLookup[static_cast<uint8_t>(word[pos]) >> 4];
}

/* Rewrite raw text into the GPT-2 byte alphabet: every *byte* of the input,
 * UTF-8 continuation bytes included, becomes the printable codepoint the
 * vocabulary spells it with.  The loop is over bytes and not codepoints --
 * a `\xc3\xa4` is two pieces here, because that is how the model's vocabulary
 * carries it. */
std::string byte_map(const std::string &text) {
  const uint32_t *table = byte_to_cpt();
  std::string out;
  out.reserve(text.size());
  for (const char c : text) {
    out += cpt_to_utf8(table[static_cast<uint8_t>(c)]);
  }
  return out;
}

/* Undo the byte mapping over one piece. */
std::string byte_unmap(const std::string &piece) {
  const std::unordered_map<uint32_t, uint8_t> &inverse = cpt_to_byte();
  std::string out;
  out.reserve(piece.size());
  for (const uint32_t cpt : cpts_from_utf8(piece)) {
    /* A piece outside the alphabet is not text this tokenizer produced.  It
     * cannot happen for a vocabulary a ggml conversion wrote, so rather than
     * invent a byte the codepoint is kept as it stands -- the result is then
     * wrong in a way the caller can see, instead of silently zeroed. */
    const auto it = inverse.find(cpt);
    if (it == inverse.end()) {
      out += cpt_to_utf8(cpt);
    } else {
      out.push_back(static_cast<char>(it->second));
    }
  }
  return out;
}

}  // namespace

Tokenizer::Tokenizer(const GgufReader &checkpoint) {
  /* The pre-tokenizer is a hard requirement, not a hint.  A vocabulary split
   * by the wrong pattern produces different ids and a model that answers
   * wrongly with no other symptom, so an unimplemented one is refused by name
   * rather than approximated.  `deepseek-r1-qwen`, `kormo` and `f2llmv2` name
   * the same pattern in llama.cpp; they are refused here only because this
   * build has not been checked against them. */
  pre_type_ = checkpoint.get_string("tokenizer.ggml.pre", "");
  if (pre_type_ != "qwen2") {
    throw Error("unsupported pre-tokenizer '" + pre_type_ +
                "': this build implements 'qwen2' only");
  }

  const std::string model = checkpoint.get_string("tokenizer.ggml.model", "");
  if (model != "gpt2") {
    throw Error("unsupported tokenizer model '" + model +
                "': this build implements 'gpt2' (byte-level BPE) only");
  }

  const GgufValue *tokens_value = checkpoint.find("tokenizer.ggml.tokens");
  if (tokens_value == nullptr || !std::holds_alternative<GgufArray>(*tokens_value)) {
    throw Error("the checkpoint has no tokenizer.ggml.tokens array");
  }
  const GgufArray &tokens = std::get<GgufArray>(*tokens_value);
  if (tokens.strings.empty()) {
    throw Error("the checkpoint's tokenizer.ggml.tokens array is empty");
  }
  vocab_ = tokens.strings;
  vocab_.reserve(vocab_.size());

  token_to_id_.reserve(vocab_.size() * 2);
  for (std::size_t i = 0; i < vocab_.size(); ++i) {
    /* First id wins for a duplicated token, matching the map insert in the
     * reader that produced the vocabulary. */
    token_to_id_.emplace(vocab_[i], static_cast<int32_t>(i));
  }

  const GgufValue *merges_value = checkpoint.find("tokenizer.ggml.merges");
  if (merges_value != nullptr && std::holds_alternative<GgufArray>(*merges_value)) {
    const GgufArray &merges = std::get<GgufArray>(*merges_value);
    merge_rank_.reserve(merges.strings.size() * 2);
    for (std::size_t i = 0; i < merges.strings.size(); ++i) {
      /* A merge is "left right", and the split is on the first space *after
       * the first character* -- a merge whose left half is a single space is
       * how the byte-level encoding spells "a space and something", and
       * finding that space at index 0 would split it into an empty half. */
      const std::string &merge = merges.strings[i];
      const std::size_t space = merge.find(' ', 1);
      if (space == std::string::npos) {
        continue;
      }
      const std::string pair = merge.substr(0, space) + '\x01' + merge.substr(space + 1);
      merge_rank_.emplace(pair, static_cast<int>(i));
    }
  }

  /* Token types: 2 is UNKNOWN, 3 is CONTROL and 4 is USER_DEFINED, and those
   * three are the ones that can be spliced.  A checkpoint without the type
   * array leaves every token NORMAL, which is a vocabulary with no specials at
   * all -- and that is the correct reading, not a fallback. */
  std::vector<int64_t> token_types;
  const GgufValue *types_value = checkpoint.find("tokenizer.ggml.token_type");
  if (types_value != nullptr && std::holds_alternative<GgufArray>(*types_value)) {
    token_types = std::get<GgufArray>(*types_value).ints;
  }
  for (std::size_t i = 0; i < vocab_.size() && i < token_types.size(); ++i) {
    const int64_t type = token_types[i];
    if (type < 2 || type > 4) {
      continue;
    }
    special_tokens_.push_back(SpecialToken{static_cast<int32_t>(i), /*needs_parse=*/type != 4});
  }
  /* Longest first, so that `<tool_response>` is preferred over any shorter
   * token that is a prefix of it.  The id breaks a length tie, which is the
   * order llama.cpp's sort produces. */
  std::sort(special_tokens_.begin(), special_tokens_.end(),
            [&](const SpecialToken &a, const SpecialToken &b) {
              if (vocab_[a.id].size() != vocab_[b.id].size()) {
                return vocab_[a.id].size() > vocab_[b.id].size();
              }
              return a.id < b.id;
            });

  special_.bos = static_cast<int32_t>(checkpoint.get_int("tokenizer.ggml.bos_token_id", -1));
  special_.eos = static_cast<int32_t>(checkpoint.get_int("tokenizer.ggml.eos_token_id", -1));
  special_.pad = static_cast<int32_t>(checkpoint.get_int("tokenizer.ggml.padding_token_id", -1));
  special_.add_bos = checkpoint.get_int("tokenizer.ggml.add_bos_token", 0) != 0;
  special_.add_eos = checkpoint.get_int("tokenizer.ggml.add_eos_token", 0) != 0;
}

Tokenizer::~Tokenizer() = default;

namespace {

/* The special-token splice: carve every occurrence of a special token out of
 * `text`, leaving the pieces between them as raw text.  The result is in order,
 * and the token fragments carry ids rather than text so that the BPE never
 * sees them. */
std::vector<Fragment> split_specials(const std::vector<SpecialToken> &specials,
                                     const std::vector<std::string> &vocab, const std::string &text,
                                     bool parse_special) {
  std::vector<Fragment> fragments;
  fragments.push_back(Fragment{false, text, 0});
  if (text.empty()) {
    return fragments;
  }

  for (const SpecialToken &special : specials) {
    if (special.needs_parse && !parse_special) {
      continue;
    }
    const int32_t id = special.id;
    const std::string &needle = vocab[static_cast<std::size_t>(id)];
    if (needle.empty()) {
      continue;
    }
    std::vector<Fragment> next;
    next.reserve(fragments.size());
    for (const Fragment &fragment : fragments) {
      if (fragment.is_token) {
        next.push_back(fragment);
        continue;
      }
      std::size_t pos = 0;
      while (true) {
        const std::size_t match = fragment.text.find(needle, pos);
        if (match == std::string::npos) {
          if (pos < fragment.text.size()) {
            next.push_back(Fragment{false, fragment.text.substr(pos), 0});
          }
          break;
        }
        if (match > pos) {
          next.push_back(Fragment{false, fragment.text.substr(pos, match - pos), 0});
        }
        next.push_back(Fragment{true, "", id});
        pos = match + needle.size();
      }
    }
    fragments = std::move(next);
  }
  return fragments;
}

}  // namespace

/* BPE over one word, appending ids.  The word has already been byte-mapped, so
 * every piece is a token spelling and the byte fallback at the end is for a
 * vocabulary that simply does not carry every single byte -- which a ggml
 * conversion always does, so the fallback is the difference between "unlikely"
 * and "silently dropping input". */
void Tokenizer::merge_word(const std::string &word, std::vector<int32_t> &out) const {
  if (word.empty()) {
    return;
  }

  std::vector<Symbol> symbols;
  symbols.reserve(word.size());
  std::size_t offset = 0;
  while (offset < word.size()) {
    const std::size_t len = std::min(utf8_char_len(word, offset), word.size() - offset);
    symbols.push_back(Symbol{word.substr(offset, len), static_cast<int>(symbols.size()) - 1, -1});
    if (symbols.size() > 1) {
      symbols[symbols.size() - 2].next = static_cast<int>(symbols.size()) - 1;
    }
    offset += len;
  }

  const auto rank_of = [&](int left, int right) -> int {
    if (left < 0 || right < 0) {
      return -1;
    }
    const std::string pair = symbols[left].text + '\x01' + symbols[right].text;
    const auto it = merge_rank_.find(pair);
    return it == merge_rank_.end() ? -1 : it->second;
  };

  BigramQueue queue;
  const auto push_bigram = [&](int left, int right) {
    if (left < 0 || right < 0) {
      return;
    }
    const int rank = rank_of(left, right);
    if (rank < 0) {
      return;
    }
    queue.push(Bigram{left, right, symbols[left].text + symbols[right].text, rank});
  };

  for (std::size_t i = 1; i < symbols.size(); ++i) {
    push_bigram(static_cast<int>(i) - 1, static_cast<int>(i));
  }

  while (!queue.empty()) {
    const Bigram bigram = queue.top();
    queue.pop();

    Symbol &left = symbols[static_cast<std::size_t>(bigram.left)];
    Symbol &right = symbols[static_cast<std::size_t>(bigram.right)];
    if (left.text.empty() || right.text.empty()) {
      continue;
    }
    /* The queue is not updated in place: a merge changes what a queued pair
     * would spell, and the entry already in the queue spells the old text.  A
     * stale entry is dropped here rather than repaired, which is cheaper than
     * a decrease-key and is what makes the linked list okay to mutate. */
    if (left.text + right.text != bigram.text) {
      continue;
    }

    left.text += right.text;
    right.text.clear();
    left.next = right.next;
    if (right.next >= 0) {
      symbols[static_cast<std::size_t>(right.next)].prev = bigram.left;
    }

    push_bigram(left.prev, bigram.left);
    push_bigram(bigram.left, left.next);
  }

  for (const Symbol &symbol : symbols) {
    if (symbol.text.empty()) {
      continue;
    }
    const auto it = token_to_id_.find(symbol.text);
    if (it != token_to_id_.end()) {
      out.push_back(it->second);
      continue;
    }
    /* The piece is not a token, which for a vocabulary a ggml conversion
     * wrote means a word that BPE could not merge any further and that is
     * itself not in the vocabulary.  Its characters are mapped bytes, each of
     * which is a token on its own, so the fallback is a lookup per character.
     *
     * llama.cpp passes *every* byte of the piece to `text_to_token` here
     * rather than a character, which finds the same token for a mapped byte
     * and misses the multi-byte ones; the difference only shows on input that
     * is not valid UTF-8 in the first place, and this is the behaviour that
     * leaves the caller's bytes intact. */
    for (std::size_t i = 0; i < symbol.text.size();) {
      const std::size_t len = std::min(utf8_char_len(symbol.text, i), symbol.text.size() - i);
      const auto byte_it = token_to_id_.find(symbol.text.substr(i, len));
      if (byte_it != token_to_id_.end()) {
        out.push_back(byte_it->second);
      }
      i += len;
    }
  }
}

std::vector<int32_t> Tokenizer::encode(const std::string &text, bool add_special,
                                       bool parse_special) const {
  std::vector<int32_t> ids;
  if (add_special && special_.add_bos && special_.bos >= 0) {
    ids.push_back(special_.bos);
  }

  const std::vector<Fragment> fragments =
      split_specials(special_tokens_, vocab_, text, parse_special);
  for (const Fragment &fragment : fragments) {
    if (fragment.is_token) {
      ids.push_back(fragment.token);
      continue;
    }
    if (fragment.text.empty()) {
      continue;
    }

    /* Split first, map second.  The order is not interchangeable: the
     * pre-tokenizer's `\p{L}` and `\p{N}` are asking about the *real* text, so
     * the regex runs over the original codepoints, and only then is each word
     * rewritten into the byte alphabet the vocabulary is spelled in.  Mapping
     * first would ask the regex about `Ä` where the user wrote an emoji. */
    const std::vector<uint32_t> cpts = cpts_from_utf8(fragment.text);
    const std::vector<std::size_t> words = qwen2_pretokenize(cpts, {cpts.size()});

    std::size_t start = 0;
    for (const std::size_t length : words) {
      std::string word;
      for (std::size_t i = start; i < start + length; ++i) {
        word += cpt_to_utf8(cpts[i]);
      }
      merge_word(byte_map(word), ids);
      start += length;
    }
  }

  if (add_special && special_.add_eos && special_.eos >= 0) {
    ids.push_back(special_.eos);
  }
  return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t> &ids) const {
  std::string out;
  for (const int32_t id : ids) {
    if (id < 0 || static_cast<std::size_t>(id) >= vocab_.size()) {
      continue;
    }
    const std::string &piece = vocab_[static_cast<std::size_t>(id)];
    if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece[5] == '>') {
      /* A `<0xXX>` byte token, which is how a vocabulary without byte-level
       * coverage spells a byte.  Parsed rather than mapped, because the bytes
       * inside it are hex, not the mapped alphabet. */
      out.push_back(static_cast<char>(std::stoi(piece.substr(3, 2), nullptr, 16)));
      continue;
    }
    out += byte_unmap(piece);
  }
  return out;
}

}  // namespace pocketllm