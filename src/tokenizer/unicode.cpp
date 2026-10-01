#include "tokenizer/unicode.h"

#include <algorithm>
#include <array>

#include "tokenizer/unicode_data.h"

namespace pocketllm {

bool cpt_is_whitespace(uint32_t codepoint) {
  return kUnicodeSetWhitespace.find(codepoint) != kUnicodeSetWhitespace.end();
}

uint16_t cpt_flags(uint32_t codepoint) {
  /* The table is a sorted list of (range_start, flags); flags apply from a
   * start up to the next start minus one.  `upper_bound` finds the first entry
   * whose start is past the codepoint, and the one before it is the covering
   * range. */
  const auto &ranges = kUnicodeRangesFlags;
  auto it = std::upper_bound(
      ranges.begin(), ranges.end(), codepoint,
      [](uint32_t value, const std::pair<uint32_t, uint16_t> &entry) { return value < entry.first; });
  uint16_t flags = it == ranges.begin() ? 0 : std::prev(it)->second;

  /* Whitespace is not one of the range flags -- the table stores the *category*
   * bits and whitespace is a property on top of them, which is why ` ` is
   * SEPARATOR in the table and whitespace in the set.  Composing the two here
   * is what llama.cpp's `unicode_cpt_flags_from_cpt` does, and leaving it out
   * is not a subtle bug: a space stops being whitespace, and then the
   * `<space>?[^\s\p{L}\p{N}]+` branch eats the space before every word. */
  if (kUnicodeSetWhitespace.find(codepoint) != kUnicodeSetWhitespace.end()) {
    flags |= kCptWhitespace;
  }
  return flags;
}

uint32_t cpt_tolower(uint32_t codepoint) {
  const auto &mappings = kUnicodeMapLowercase;
  auto it = std::lower_bound(
      mappings.begin(), mappings.end(), codepoint,
      [](const std::pair<uint32_t, uint32_t> &entry, uint32_t value) { return entry.first < value; });
  if (it == mappings.end() || it->first != codepoint) {
    return codepoint;
  }
  return it->second;
}

std::vector<uint32_t> cpts_from_utf8(const std::string &text) {
  std::vector<uint32_t> out;
  out.reserve(text.size());
  const auto *bytes = reinterpret_cast<const uint8_t *>(text.data());
  std::size_t i = 0;
  const std::size_t n = text.size();
  while (i < n) {
    const uint8_t b0 = bytes[i];
    uint32_t codepoint = b0;
    std::size_t extra = 0;
    if (b0 < 0x80) {
      extra = 0;
      codepoint = b0;
    } else if ((b0 & 0xE0) == 0xC0) {
      extra = 1;
      codepoint = b0 & 0x1Fu;
    } else if ((b0 & 0xF0) == 0xE0) {
      extra = 2;
      codepoint = b0 & 0x0Fu;
    } else if ((b0 & 0xF8) == 0xF0) {
      extra = 3;
      codepoint = b0 & 0x07u;
    } else {
      /* A continuation byte or a 5/6-byte lead: not valid UTF-8.  Treat the
       * byte as its own codepoint and move on, which keeps every byte in the
       * output exactly once. */
      out.push_back(b0);
      ++i;
      continue;
    }

    if (i + extra >= n) {
      out.push_back(b0);
      ++i;
      continue;
    }
    bool valid = true;
    for (std::size_t k = 1; k <= extra; ++k) {
      if ((bytes[i + k] & 0xC0) != 0x80) {
        valid = false;
        break;
      }
      codepoint = (codepoint << 6) | (bytes[i + k] & 0x3Fu);
    }
    if (!valid) {
      out.push_back(b0);
      ++i;
      continue;
    }
    out.push_back(codepoint);
    i += extra + 1;
  }
  return out;
}

std::string cpt_to_utf8(uint32_t codepoint) {
  std::string out;
  if (codepoint < 0x80) {
    out.push_back(static_cast<char>(codepoint));
  } else if (codepoint < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else if (codepoint < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  }
  return out;
}

namespace {

/* GPT-2's byte-to-unicode assignment.
 *
 * Printable ASCII and a span of Latin-1 map to themselves; every other byte is
 * given a codepoint starting at 256, in increasing byte order.  Building it by
 * the same loop GPT-2 uses is the only way to be sure the assignment matches --
 * a table copied by hand from a blog post is a table that is off by one
 * somewhere, and the symptom is a vocabulary mismatch on a handful of bytes. */
std::array<uint32_t, 256> build_byte_to_cpt() {
  std::array<uint32_t, 256> table{};
  uint32_t next = 256;
  for (int b = 0; b < 256; ++b) {
    const bool printable_ascii = (b >= 33 && b <= 126);
    const bool printable_latin = (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    if (printable_ascii || printable_latin) {
      table[static_cast<std::size_t>(b)] = static_cast<uint32_t>(b);
    } else {
      table[static_cast<std::size_t>(b)] = next++;
    }
  }
  return table;
}

std::unordered_map<uint32_t, uint8_t> build_cpt_to_byte(const std::array<uint32_t, 256> &forward) {
  std::unordered_map<uint32_t, uint8_t> inverse;
  inverse.reserve(256);
  for (std::size_t b = 0; b < forward.size(); ++b) {
    inverse[forward[b]] = static_cast<uint8_t>(b);
  }
  return inverse;
}

}  // namespace

const uint32_t *byte_to_cpt() {
  static const std::array<uint32_t, 256> table = build_byte_to_cpt();
  return table.data();
}

const std::unordered_map<uint32_t, uint8_t> &cpt_to_byte() {
  static const std::array<uint32_t, 256> forward = build_byte_to_cpt();
  static const std::unordered_map<uint32_t, uint8_t> inverse = build_cpt_to_byte(forward);
  return inverse;
}

}  // namespace pocketllm