#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "json_lite.hpp"

namespace pocket {

// Client stop sequences are matched against the *decoded* text rather than
// against token ids. A stop string is not one token -- "USER:" is three in most
// vocabularies, and a sequence can start inside one token and end inside the
// next -- so the only place the sequence is visible as a unit is the text the
// caller is going to read anyway.
//
// The scan is therefore expressed over the cumulative decoded text: the caller
// appends each generated token, decodes what it has so far, and asks this how
// much of that text it may hand over. Nothing here is stateful, so the same
// answer is produced whether the caller asks once at the end (a non-streaming
// response) or after every token (a stream).

// How much of a decoded text the client's stop sequences leave deliverable.
struct StopScan {
    // Whether one of the sequences is present in the text.
    bool matched = false;
    // Where the earliest matched sequence starts. Meaningful only when
    // `matched`; 0 otherwise.
    std::size_t offset = 0;
    // Bytes deliverable *now*, while more tokens may still arrive: everything
    // before `offset` when matched, and everything except a trailing partial
    // sequence otherwise. A partial sequence is withheld because the next token
    // may complete it, and text already sent cannot be taken back.
    std::size_t safe_length = 0;
    // Bytes deliverable once generation has ended: `offset` when matched, the
    // whole text otherwise. A trailing partial sequence can no longer complete,
    // so it is part of the answer rather than a withheld prefix.
    std::size_t final_length = 0;
};

// Finds the earliest client stop sequence in `text` and reports how much of it
// may be delivered. Empty sequences are ignored: they match nothing, which is
// what makes an empty `stop` list -- or the empty strings some clients pad it
// with -- harmless.
StopScan scan_stop_strings(const std::string& text,
                           const std::vector<std::string>& sequences);

// The sequences a request asks to stop on, in the order given, with empty
// strings dropped. A request without the field yields an empty list, so callers
// do not need a separate presence check.
std::vector<std::string> parse_stop_sequences(const JsonObject& body);

// Whether a `stop` value has the documented shape: a string, or a list of
// strings. A value of another shape is a request error rather than an empty
// stop list -- dropping it would be the silent-ignore failure this field's
// handling exists to avoid.
bool is_stop_shape(const JsonValue& value);

}  // namespace pocket
