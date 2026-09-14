#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pocket {

class JsonValue;
class Tokenizer;

// Token-level constraint applied immediately before sampling. The implementation
// owns its vocabulary strings, so the inference engine never needs a Tokenizer.
class TokenConstraint {
public:
    virtual ~TokenConstraint() = default;

    // Set allowed[i - vocab_begin] for every token ID in [vocab_begin, vocab_end).
    virtual void fill_mask(bool* allowed, int vocab_begin, int vocab_end) const = 0;

    // Commit a sampled token and advance the constraint state.
    virtual bool accept_token(int token_id) = 0;

    virtual bool is_complete() const = 0;
    virtual void reset() = 0;
};

std::unique_ptr<TokenConstraint> make_json_object_constraint(
    const Tokenizer& tokenizer);

std::unique_ptr<TokenConstraint> make_json_schema_constraint(
    const Tokenizer& tokenizer, const JsonValue& schema);

// Host-only factory useful for tests and for runtimes that already have a decoded
// vocabulary. `special_token_ids` may be empty, in which case every piece is
// treated as a normal token.
std::unique_ptr<TokenConstraint> make_json_object_constraint(
    std::vector<std::string> token_pieces,
    std::vector<uint8_t> special_token_ids = {});

}  // namespace pocket
