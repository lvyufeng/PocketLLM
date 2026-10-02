/* Tokenize text with the engine's own BPE, for comparison against llama.cpp.
 *
 * An internal tool, not part of the ABI.  The tokenizer test drives it two
 * ways:
 *
 *   - `--lines` reads newline-separated texts from stdin and prints the ids of
 *     each on its own line, so a corpus can be diffed in one process rather
 *     than one session per line;
 *   - `--text` takes one text on the command line, for eyeballing against
 *     `llama-tokenize` by hand.
 *
 * Both print ids space-separated with no other decoration, because the point is
 * `diff`, not report.  Decoding is available too -- `--decode` reads ids and
 * prints the text, hex-escaped so that an unprintable byte is still visible in
 * the diff.
 */

#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gguf/reader.h"
#include "runtime/status.h"
#include "tokenizer/bpe.h"

namespace {

void print_ids(const std::vector<int32_t> &ids) {
  for (std::size_t i = 0; i < ids.size(); ++i) {
    std::printf("%s%d", i ? " " : "", ids[i]);
  }
  std::printf("\n");
}

/* One text per line, in the same escape convention `print_escaped` writes.
 *
 * The escaping is not a convenience. Newlines are among the cases the
 * pre-tokenizer has to get right -- the `\s*[\r\n]+` branch exists for them --
 * so a transport that cannot carry one cannot test the tokenizer on the inputs
 * most likely to be wrong. `\` starts an escape in this format and there is no
 * unescaped form of it, so a text containing a literal backslash round-trips
 * rather than being reinterpreted. */
std::string unescape_line(const std::string &line) {
  std::string out;
  out.reserve(line.size());
  for (std::size_t i = 0; i < line.size(); ++i) {
    if (line[i] != '\\' || i + 1 >= line.size()) {
      out.push_back(line[i]);
      continue;
    }
    switch (line[++i]) {
      case 'n': out.push_back('\n'); break;
      case 'r': out.push_back('\r'); break;
      case 't': out.push_back('\t'); break;
      case '0': out.push_back('\0'); break;
      case '\\': out.push_back('\\'); break;
      case 'x': {
        if (i + 2 < line.size()) {
          out.push_back(static_cast<char>(std::stoi(line.substr(i + 1, 2), nullptr, 16)));
          i += 2;
        }
        break;
      }
      default: out.push_back(line[i]); break;
    }
  }
  return out;
}

/* The text with every byte that is not printable-and-unambiguous escaped, so
 * that `--lines` can carry any text at all and a test diffing two decoders
 * sees an off-by-one byte instead of an ambiguous line. */
void print_escaped(const std::string &text) {
  for (const unsigned char c : text) {
    if (c == '\\') {
      std::printf("\\\\");
    } else if (c >= 0x20 && c < 0x7F) {
      std::printf("%c", c);
    } else {
      std::printf("\\x%02x", c);
    }
  }
  std::printf("\n");
}

int usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s <checkpoint.gguf> --text <text> [--no-special] [--parse-special]\n"
               "       %s <checkpoint.gguf> --lines [--no-special] [--parse-special]\n"
               "       %s <checkpoint.gguf> --decode   # stdin, one id list per line\n",
               argv0, argv0, argv0);
  return 2;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    return usage(argv[0]);
  }

  const std::string path = argv[1];
  std::string text;
  bool lines = false;
  bool decode = false;
  bool add_special = true;
  /* Off by default, which is llama.cpp's default and what makes the oracle
   * comparison an exact one: `llama_tokenize`'s `parse_special` defaults to
   * false, and a tool that spliced control tokens by default could only be
   * diffed against it with a flag on both sides. */
  bool parse_special = false;

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--text" && i + 1 < argc) {
      text = argv[++i];
    } else if (arg == "--lines") {
      lines = true;
    } else if (arg == "--decode") {
      decode = true;
    } else if (arg == "--no-special") {
      add_special = false;
    } else if (arg == "--parse-special") {
      parse_special = true;
    } else {
      return usage(argv[0]);
    }
  }

  try {
    pocketllm::GgufReader reader(path);
    const pocketllm::Tokenizer tokenizer(reader);

    if (decode) {
      std::string line;
      while (std::getline(std::cin, line)) {
        std::istringstream stream(line);
        std::vector<int32_t> ids;
        int32_t id = 0;
        while (stream >> id) {
          ids.push_back(id);
        }
        print_escaped(tokenizer.decode(ids));
      }
      return 0;
    }

    if (lines) {
      std::string line;
      while (std::getline(std::cin, line)) {
        print_ids(tokenizer.encode(unescape_line(line), add_special, parse_special));
      }
      return 0;
    }

    print_ids(tokenizer.encode(text, add_special, parse_special));
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s: %s\n", argv[0], e.what());
    return 1;
  }
}