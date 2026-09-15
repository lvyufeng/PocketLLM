#include "openai_stop_strings.hpp"

#include <algorithm>

namespace pocket {

namespace {

// Length of the longest suffix of `text` that is a proper prefix of one of the
// sequences -- the bytes a scan has to withhold because the next token may turn
// them into a match. `text` ending in the complete sequence is not this case:
// that is a match, and the scan reports it instead.
//
// Each sequence is tried from its longest possible overlap downwards and stops
// at the first hit, so the common case -- a text whose last byte is not the
// first byte of anything -- costs one comparison per sequence.
std::size_t partial_suffix_length(const std::string& text,
                                  const std::vector<std::string>& sequences) {
    std::size_t held = 0;
    for (const std::string& sequence : sequences) {
        if (sequence.empty()) continue;
        // A sequence of one byte can never be partially matched: if its only
        // byte is already the last byte of `text`, the sequence is present and
        // the scan matched.
        for (std::size_t length = std::min(sequence.size() - 1, text.size());
             length > held; --length) {
            if (text.compare(text.size() - length, length, sequence, 0, length) == 0) {
                held = length;
                break;
            }
        }
    }
    return held;
}

}  // namespace

StopScan scan_stop_strings(const std::string& text,
                           const std::vector<std::string>& sequences) {
    StopScan scan;
    std::size_t earliest = std::string::npos;
    for (const std::string& sequence : sequences) {
        if (sequence.empty()) continue;
        const std::size_t at = text.find(sequence);
        if (at < earliest) earliest = at;
    }

    scan.matched = earliest != std::string::npos;
    scan.offset = scan.matched ? earliest : 0;
    scan.safe_length = scan.matched
                           ? earliest
                           : text.size() - partial_suffix_length(text, sequences);
    scan.final_length = scan.matched ? earliest : text.size();
    return scan;
}

std::vector<std::string> parse_stop_sequences(const JsonObject& body) {
    std::vector<std::string> sequences;
    const JsonValue* value = object_get(body, "stop");
    if (value == nullptr || value->is_null()) return sequences;

    if (value->is_string()) {
        if (!value->string().empty()) sequences.push_back(value->string());
        return sequences;
    }
    if (value->is_array()) {
        for (const JsonValue& item : value->array()) {
            if (item.is_string() && !item.string().empty()) {
                sequences.push_back(item.string());
            }
        }
    }
    return sequences;
}

bool is_stop_shape(const JsonValue& value) {
    if (value.is_string()) return true;
    if (!value.is_array()) return false;
    for (const JsonValue& item : value.array()) {
        if (!item.is_string()) return false;
    }
    return true;
}

}  // namespace pocket
