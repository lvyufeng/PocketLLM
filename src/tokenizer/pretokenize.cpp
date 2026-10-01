#include "tokenizer/pretokenize.h"

#include <cstddef>

#include "tokenizer/unicode.h"

namespace pocketllm {

namespace {

/* Sentinel for "past the end of this word".  No codepoint is ever this value,
 * so a lookup past the boundary and a lookup past the end of the text answer
 * the same way -- both are "nothing here". */
constexpr uint32_t kOutOfRange = 0xFFFFFFFFu;

}  // namespace

std::vector<std::size_t> qwen2_pretokenize(const std::vector<uint32_t> &cpts,
                                           const std::vector<std::size_t> &offsets) {
  std::vector<std::size_t> words;
  words.reserve(offsets.size());

  std::size_t start = 0;
  for (const std::size_t offset : offsets) {
    const std::size_t ini = start;
    const std::size_t end = start + offset;
    start = end;

    /* The three accessors are closures over this word, and each answers
     * `kOutOfRange` / zero flags outside it.  That is what makes the `+\r\n`
     * loops and the `\s+(?!\S)` lookahead terminate at the word boundary
     * instead of running into the next one. */
    const auto cpt_at = [&](std::size_t pos) -> uint32_t {
      return (ini <= pos && pos < end) ? cpts[pos] : kOutOfRange;
    };
    const auto flags_at = [&](std::size_t pos) -> uint16_t {
      return (ini <= pos && pos < end) ? cpt_flags(cpts[pos]) : 0;
    };

    std::size_t emitted = ini;
    const auto emit_through = [&](std::size_t stop) -> std::size_t {
      const std::size_t len = stop - emitted;
      if (len > 0) {
        words.push_back(len);
      }
      emitted = stop;
      return len;
    };

    for (std::size_t pos = ini; pos < end;) {
      const uint32_t cpt = cpt_at(pos);
      const uint16_t flags = flags_at(pos);

      /* (?i:'s|'t|'re|'ve|'m|'ll|'d) */
      if (cpt == '\'' && pos + 1 < end) {
        const uint32_t next = cpt_tolower(cpt_at(pos + 1));
        if (next == 's' || next == 't' || next == 'm' || next == 'd') {
          pos += emit_through(pos + 2);
          continue;
        }
        if (pos + 2 < end) {
          const uint32_t next2 = cpt_tolower(cpt_at(pos + 2));
          if ((next == 'r' && next2 == 'e') || (next == 'v' && next2 == 'e') ||
              (next == 'l' && next2 == 'l')) {
            pos += emit_through(pos + 3);
            continue;
          }
        }
      }

      /* [^\r\n\p{L}\p{N}]?\p{L}+ -- an optional non-letter lead-in, then a run
       * of letters.  The lead-in is consumed because it is part of the match;
       * what disqualifies it is being a newline or a digit. */
      if (!(cpt == '\r' || cpt == '\n' || (flags & kCptNumber))) {
        if ((flags & kCptLetter) || (flags_at(pos + 1) & kCptLetter)) {
          ++pos;
          while (flags_at(pos) & kCptLetter) {
            ++pos;
          }
          emit_through(pos);
          continue;
        }
      }

      /* \p{N} -- one digit per word, not a run: a number is split digit by
       * digit so that the vocabulary's digit tokens are what the model sees. */
      if (flags & kCptNumber) {
        ++pos;
        emit_through(pos);
        continue;
      }

      /* ` ?[^\s\p{L}\p{N}]+[\r\n]*` -- optional leading space, then a run of
       * punctuation and symbols, then any newlines that follow it. */
      uint16_t lead_flags = (cpt == ' ') ? flags_at(pos + 1) : flags;
      if (!(lead_flags & (kCptWhitespace | kCptLetter | kCptNumber)) && lead_flags != 0) {
        pos += (cpt == ' ') ? 1 : 0;
        while (!(lead_flags & (kCptWhitespace | kCptLetter | kCptNumber)) && lead_flags != 0) {
          lead_flags = flags_at(++pos);
        }
        uint32_t tail = cpt_at(pos);
        while (tail == '\r' || tail == '\n') {
          tail = cpt_at(++pos);
        }
        emit_through(pos);
        continue;
      }

      /* The whitespace runs.  All three branches below start from the same
       * scan, which counts the run and remembers where the last newline in it
       * ended. */
      std::size_t run = 0;
      std::size_t last_newline_end = 0;
      while (flags_at(pos + run) & kCptWhitespace) {
        const uint32_t c = cpt_at(pos + run);
        if (c == '\r' || c == '\n') {
          last_newline_end = pos + run + 1;
        }
        ++run;
      }

      /* \s*[\r\n]+ -- anything up to and including the last newline is one
       * word; whitespace *after* the last newline belongs to the next match. */
      if (last_newline_end > 0) {
        pos = last_newline_end;
        emit_through(pos);
        continue;
      }

      /* \s+(?!\S) -- a trailing run leaves its final space for the next word,
       * so that a space-then-word splits as " " + "word".  The lookahead fails
       * at the end of the text, where there is nothing left to attach the last
       * space to, and that is what the `kOutOfRange` comparison is for. */
      if (run > 1 && cpt_at(pos + run) != kOutOfRange) {
        pos += run - 1;
        emit_through(pos);
        continue;
      }

      /* \s+ */
      if (run > 0) {
        pos += run;
        emit_through(pos);
        continue;
      }

      /* Nothing matched -- one codepoint on its own, which is how an
       * unassigned or control codepoint gets through. */
      emit_through(++pos);
    }
  }

  return words;
}

}  // namespace pocketllm