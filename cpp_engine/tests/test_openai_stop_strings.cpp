// Unit tests for client stop-sequence matching.
//
// The scan runs on every generated token of a streaming response, so it is
// worth pinning down separately from the server: it is pure string logic, needs
// no checkpoint or device, and links pocket_core.
//
// Plain `assert` is deliberately not used. Release builds define NDEBUG, which
// would compile every assertion in this file out and report success without
// running anything.

#include "openai_stop_strings.hpp"

#include <iostream>
#include <string>
#include <vector>

using namespace pocket;

namespace {

int g_failures = 0;

void check(bool condition, const char* expression, int line) {
    if (condition) return;
    std::cout << "FAIL line " << line << ": " << expression << "\n";
    ++g_failures;
}

#define CHECK(condition) check((condition), #condition, __LINE__)

const std::vector<std::string> kNoStops;

// Asserts the three lengths a scan reports, since they are what every caller
// branches on and a wrong one either truncates text that should have been sent
// or leaks a stop sequence into the answer.
void check_scan(const std::string& text, const std::vector<std::string>& sequences,
                bool matched, std::size_t offset, std::size_t safe_length,
                std::size_t final_length, int line) {
    const StopScan scan = scan_stop_strings(text, sequences);
    bool ok = true;
    if (scan.matched != matched) {
        std::cout << "FAIL line " << line << ": matched=" << scan.matched
                  << " for text " << text << "\n";
        ok = false;
    }
    if (scan.offset != offset || scan.safe_length != safe_length ||
        scan.final_length != final_length) {
        std::cout << "FAIL line " << line << ": offset=" << scan.offset
                  << " safe=" << scan.safe_length << " final=" << scan.final_length
                  << ", expected offset=" << offset << " safe=" << safe_length
                  << " final=" << final_length << " for text " << text << "\n";
        ok = false;
    }
    if (!ok) ++g_failures;
}

// `sequences` is deliberately not parenthesised: it is written as a braced list
// at the call site, and parentheses around one do not parse. A list with more
// than one entry has to arrive through a named vector, because the preprocessor
// splits the macro arguments on those commas.
#define SCAN(text, sequences, matched, offset, safe, final) \
    check_scan((text), sequences, (matched), (offset), (safe), (final), __LINE__)

// A request without stop sequences must behave exactly as the server did
// before they were supported: the whole text is deliverable at every point.
void test_without_sequences() {
    SCAN("", kNoStops, false, 0, 0, 0);
    SCAN("hello", kNoStops, false, 0, 5, 5);
    SCAN("hello\n", kNoStops, false, 0, 6, 6);

    // An empty string is how a client spells "no stop sequence", so it must not
    // match at position 0 -- which would truncate every response to nothing.
    const std::vector<std::string> empty_sequences = {"", ""};
    SCAN("", empty_sequences, false, 0, 0, 0);
    SCAN("hello", empty_sequences, false, 0, 5, 5);
}

void test_matches() {
    // A match at the very start: the answer is empty, and that is the whole
    // point of the sequence.
    SCAN("OK", {"OK"}, true, 0, 0, 0);
    SCAN("STOP!", {"STOP"}, true, 0, 0, 0);

    // The sequence itself is never part of the answer, but text before it is.
    SCAN("hello STOP world", {" STOP"}, true, 5, 5, 5);
    SCAN("hello\n\nworld", {"\n\n"}, true, 5, 5, 5);

    // The earliest occurrence wins, not the first sequence in the list.
    const std::vector<std::string> reversed = {"cd", "ab"};
    SCAN("ab cd", reversed, true, 0, 0, 0);
    const std::vector<std::string> both = {"a", "b"};
    SCAN("xx b yy a", both, true, 3, 3, 3);

    // Overlapping occurrences: the leftmost start is the offset.
    SCAN("aaab", {"aab"}, true, 1, 1, 1);

    // A sequence that occurs twice stops at the first one.
    SCAN("END x END", {"END"}, true, 0, 0, 0);
}

// The holdback: bytes that may still become a match must not be delivered while
// generation is running, but they are part of the answer once it has ended.
void test_partial_sequences() {
    SCAN("hello\n", {"\n\n"}, false, 0, 5, 6);
    SCAN("hello\n\n", {"\n\n"}, true, 5, 5, 5);

    // Two sequences, and the suffix is tested against both: the longer overlap
    // wins whichever sequence it belongs to.
    const std::vector<std::string> abcd = {"ABC", "BCD"};
    SCAN("xAB", abcd, false, 0, 1, 3);
    SCAN("xBC", abcd, false, 0, 1, 3);

    // A one-byte sequence is never partially matched: a trailing byte either
    // makes the sequence present or has nothing to do with it.
    SCAN("hello", {"Z"}, false, 0, 5, 5);

    // A byte-level holdback rather than a character-level one. A stream delivers
    // text in UTF-8 chunks, so the partial match can be the first byte of a
    // multi-byte character.
    SCAN("h\xC3", {"\xC3\xA9"}, false, 0, 1, 2);
    SCAN("h\xC3\xA9", {"\xC3\xA9"}, true, 1, 1, 1);

    // The holdback is bounded by the sequence length, not by the text: a text
    // that is longer than every sequence has already been delivered up to it.
    SCAN("abcdef", {"xy"}, false, 0, 6, 6);
}

// The scan is what the four response emitters share, so it has to describe the
// case where the sequence is longer than everything generated so far.
void test_text_shorter_than_sequence() {
    SCAN("a", {"abc"}, false, 0, 0, 1);
    SCAN("ab", {"abc"}, false, 0, 0, 2);
    SCAN("abc", {"abc"}, true, 0, 0, 0);
    SCAN("xabc", {"abc"}, true, 1, 1, 1);
}

void test_parse_stop_sequences() {
    const auto parse = [](const std::string& json) {
        return parse_stop_sequences(parse_json(json).object());
    };

    CHECK(parse("{}").empty());
    CHECK(parse(R"({"stop":null})").empty());
    CHECK(parse(R"({"stop":""})").empty());
    CHECK(parse(R"({"stop":[]})").empty());
    CHECK(parse(R"({"stop":["",""]})").empty());

    // A bare string is the one-sequence spelling of the list.
    const std::vector<std::string> one = parse(R"({"stop":"\n\n"})");
    CHECK(one == std::vector<std::string>({"\n\n"}));
    CHECK(parse(R"({"stop":["\n\n"]})") == one);

    // Empty entries are dropped and the rest keep their order.
    const std::vector<std::string> mixed = parse(R"({"stop":["", "USER:", "","\n"]})");
    CHECK(mixed == std::vector<std::string>({"USER:", "\n"}));

    // A malformed entry is not a sequence; check_request_fields refuses the
    // request on shape, and nothing here treats the value as something else.
    CHECK(parse(R"({"stop":["ok", 5]})") == std::vector<std::string>({"ok"}));
    CHECK(parse(R"({"stop":5})").empty());
}

void test_is_stop_shape() {
    const auto shape = [](const std::string& json) {
        const JsonValue value = parse_json(json);
        const JsonValue* stop = object_get(value.object(), "stop");
        return stop != nullptr && is_stop_shape(*stop);
    };

    CHECK(shape(R"({"stop":"END"})"));
    CHECK(shape(R"({"stop":""})"));
    CHECK(shape(R"({"stop":[]})"));
    CHECK(shape(R"({"stop":["a", "b"]})"));

    CHECK(!shape(R"({"stop":5})"));
    CHECK(!shape(R"({"stop":true})"));
    CHECK(!shape(R"({"stop":{"a":1}})"));
    CHECK(!shape(R"({"stop":["a", 5]})"));
    CHECK(!shape(R"({"stop":[["a"]]})"));
    // null is "not set" in several SDKs, so check_request_fields treats it as
    // absent and never asks this question about it.
    CHECK(!shape(R"({"stop":null})"));
}

}  // namespace

int main() {
    test_without_sequences();
    test_matches();
    test_partial_sequences();
    test_text_shorter_than_sequence();
    test_parse_stop_sequences();
    test_is_stop_shape();

    if (g_failures != 0) {
        std::cout << g_failures << " stop sequence test(s) failed\n";
        return 1;
    }
    std::cout << "All stop sequence tests passed\n";
    return 0;
}
